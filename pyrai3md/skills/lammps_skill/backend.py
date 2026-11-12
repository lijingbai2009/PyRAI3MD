"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the backend for the LAMMPS skill.
It is the single public computational API entry point.

Responsibilities:
- Initialize LAMMPS configuration (build -> normalize -> validate)
- Compile calculation workflow from profile
- Resolve calculation directory (one fixed namespace per instance)
- Execute calculations, in one of two modes
- Return standardized results (energy in hartree, gradient in
  hartree/bohr, converted from the lammps_units style; the
  internal-iteration profiles additionally return the final geometry
  through the fin_coord channel)

Core principle:
Compile the computational workflow once during backend initialization,
then execute lightweight calculate() calls during dynamics.

Subprocess mode (python_enabled=false, the default) keeps the historical
behavior: every calculate() call updates the coordinates in the topology
(data) file, writes the input script, runs lmp and parses its output.
Single-point profiles run a 'run 0' evaluation; the ff_optimize and
ff_nve/ff_nvt/ff_npt profiles run an internal minimization or dynamics
propagation and return the final structure through the fin_coord
channel.

Resident mode (python_enabled=true, 'local' execution) moves the
topology and force-field loading into the constructor: a service process
in its own interpreter holds a live LAMMPS instance, and every
calculate() call ships only the coordinates and the box of that step over
a Unix socket. Nothing is written per step, no process is started per
step, and the framework process never imports LAMMPS or MPI. The switch
is an intent, not a fallback: a resident run that cannot be served raises
rather than quietly running the subprocess path instead.

"""

import os
import shutil
import sys
import threading
import warnings
import weakref

import numpy as np
from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import validate, is_persistent_enabled
from .service.workflow import compile_workflow
from .service.builder import (write_input_file, write_static_input_files,
                              resolve_live_box)
from .service.builder.ff_model import (FFModelUnsupported, build_ff_model,
                                       load_cached_model, model_path,
                                       read_templates, save_model)
from .service.builder.input_builder import _resolve_periodic_lattice
from .service.builder.qmmm import validate_partition_charges
from .service.builder.qmmm_energy import write_energy_input_file
from .service.builder.qmmm_field import (write_field_input_file,
                                         write_field_static_input_files)
from .service.runner import get_runner, validate_environment
from .service.parser import (format_persistent_results,
                             format_results, parse_output,
                             pressure6_to_stress_tensor, to_native_gradient)
from .service.read import read_calculation
from .service.parser.output_parser import (_build_stress_tensor,
                                           _parse_force_dump,
                                           read_energy_file)
from .service.parser.qmmm_output import (assemble_qmmm_energy,
                                         assemble_qmmm_field,
                                         read_esfp, read_field_groups,
                                         read_probe_force, read_probe_pe,
                                         write_esfp, write_grad,
                                         write_qm_force,
                                         write_qm_multipole,
                                         write_qmmm_energy)
from .service.persistent import launcher
from .service.persistent.client import PersistentSession
from .service.persistent.profile import PROFILE_EVERY, StepProfiler
from .service.persistent.protocol import (
    STARTUP_TIMEOUT_S,
    LammpsPersistentError,
    LammpsPersistentUnsupported,
    compute_fingerprint,
)
from .service.utils import (build_calcdir, flatten_extended_config,
                            output_basename, step_data_files)

# Profiles the resident service implements; schema.validate_persistent
# enforces the same list, this one guards the constructor path.
_RESIDENT_PROFILES = ('ff_energy', 'ff_gradient',
                      'qmmm_field', 'qmmm_energy')
_QM_MM_PROFILES = ('qmmm_field', 'qmmm_energy')


def _client_profile_rows(timings):
    """Render the client's socket buckets as '[client] op=...' rows.

    Parameters
    ----------
    timings : dict
        What :meth:`LammpsBackend.resident_timings` reports.

    Returns
    -------
    list of str
        One row per served operation, in milliseconds per call.
    """
    rows = []
    for name, record in sorted(timings.items()):
        rows.append('[client] op=%s calls=%d send_ms=%.3f wait_ms=%.3f '
                    'parse_ms=%.3f'
                    % (name.lower(), record['calls'], record['send_ms'],
                       record['wait_ms'], record['parse_ms']))
    return rows


class LammpsBackend:
    """Main backend class for LAMMPS calculations.

    Initializes once, then serves lightweight calculate() calls.
    The calculation directory is determined at initialization and
    remains fixed across all calculate() calls.

    In resident mode the backend owns a service process for its whole
    lifetime and releases it in close(); the context-manager protocol and
    a weakref finalizer both reach that same idempotent teardown, so a
    dropped or abandoned backend cannot leave a LAMMPS world running.

    Supports initialization from:
    1. Dictionary: direct configuration dict
    2. JSON path: path to JSON config file
    3. BaseModel: pydantic model instance (standalone flat config or
       a prebuilt extended PyRAIConfig from PyRAIState)

    Parameters
    ----------
    config : dict or str or BaseModel
        Configuration dictionary, path to JSON config file,
        or pydantic model.
    calcdir : str, optional
        Explicit full calculation directory path. Takes highest priority.
    execution_id : str, optional
        Execution namespace identifier assigned by the Calculator
        (e.g. 'tmp_LAMMPS' (single task) or 'tmp_low_mm_LAMMPS'). Used directly as the calculation
        directory name; empty string falls back to the standalone
        tmp_LAMMPS naming. A resident service is identified by the socket
        this directory yields, not by this name: an isolated-workdir batch
        gives every unit its own directory under one shared namespace.
    """

    def __init__(self, config, calcdir=None, execution_id=''):

        # Resolve the raw skill config from the input source.
        # Task config dict mode: dict with 'skill_config' from framework
        #   task queue (new design, independent per task).
        # Legacy state mode: PyRAIConfig from PyRAIState with _skill_config
        #   attached by the framework (defaults filled, profile resolved).
        # Standalone mode: dict, JSON path or flat config model.
        if isinstance(config, dict) and 'skill_config' in config:
            # New: task config dict from framework task queue
            self._task_config = config
            raw_config = dict(config['skill_config'])
        elif isinstance(config, BaseModel) and 'control' in type(config).model_fields:
            # Legacy: PyRAIConfig from PyRAIState with _skill_config
            self._task_config = None
            self._pyrai_config = config
            skill_config = getattr(config, '_skill_config', None)
            raw_config = (dict(skill_config) if skill_config is not None
                          else flatten_extended_config(config))
        else:
            # Standalone mode: dict, JSON path or flat config model
            self._task_config = None
            self._pyrai_config = None
            raw_config = config
            if isinstance(config, str):
                raw_config = read_json(config)

        # Run the skill schema pipeline (build -> normalize -> validate)
        # in both modes. The framework merge only carries raw values and
        # defaults; type conversion and derived fields (natom, ground-state
        # info, ...) are computed here.
        # The result is a nested SkillConfig held as self.config
        # throughout the backend lifetime (no config_dict).
        input_config = build_config(raw_config)
        normalized_config = normalize(input_config)
        self.config = validate(normalized_config)

        # Ensure environment scratch_path is valid (default to cwd)
        scratch_path = self.config.environment.scratch_path
        if scratch_path and scratch_path != 'AUTO':
            scratch_path = os.path.expanduser(scratch_path)
            os.makedirs(scratch_path, exist_ok=True)
        else:
            scratch_path = ''
        self.config.environment.scratch_path = scratch_path

        # Resident-mode bookkeeping, before anything that can fail.
        self._lock = threading.RLock()
        self._closed = False
        self._service = None
        self._session = None
        self._finalizer = None
        self._execution_id = execution_id or ''
        self._claim = None
        self._step = 0
        self._natom = int(self.config.natom)
        self._box_origin = None
        self._paths = None

        profile = self.config.calculation.profile
        self._persistent = is_persistent_enabled(self.config)
        if self._persistent and profile not in _RESIDENT_PROFILES:
            raise LammpsPersistentUnsupported(
                f'python_enabled=true is implemented for '
                f'{", ".join(_RESIDENT_PROFILES)} only; profile '
                f'{profile!r} keeps the subprocess path. Set '
                f'python_enabled=false.')
        if (getattr(self.config, 'python_enabled', False)
                and not self._persistent):
            # A designed route, not a degradation: residency cannot help
            # when every step is its own blocking batch job.
            warnings.warn(
                'LAMMPS python_enabled ignored: execution_mode=slurm '
                'submits one blocking job per calculation, so there is '
                'nothing to keep resident. Using the subprocess path.')

        # No environment check on the resident path. It never runs lmp,
        # and whether its interpreter can load LAMMPS is answered by the
        # service itself: it imports lammps, builds the instance and
        # proves its world before binding a socket, so a service that
        # cannot work has already exited with its reason in the log by the
        # time the launcher looks for it.
        if not self._persistent:
            # Validate external program environment
            validate_environment(self.config)

            # Resolve the execution runner once: execution_mode is fixed
            # at initialization (no runtime path mutates it), so the mode
            # decision stays with the service (get_runner) and the result
            # is cached for lightweight calculate() calls.
            self.run = get_runner(self.config)

        # Compile workflow from profile
        self.workflow = compile_workflow(profile)

        # Resolve calculation directory under the scratch path
        # (fixed for this backend instance)
        if calcdir:
            self.calcdir = calcdir
        else:
            self.calcdir = build_calcdir(
                execution_id=execution_id, base_path=scratch_path or None,
                project=self.config.calculation.project,
            )

        # Compress the force field once. The model is the authoritative
        # description of what every step evaluates, and it is held here in
        # memory for the life of this backend: no step re-reads it from
        # disk, and the package writer saves it into the calculation
        # directory beside the files it produced, so keep_tmp decides
        # whether anyone can inspect it afterwards. Compression is a
        # mandatory stage, not a switch: it only degrades to the verbatim
        # templates when a template shape cannot be proven equivalent.
        self.ff_model = self._resolve_ff_model(profile)

        if self._persistent:
            # Resident mode evaluates the model, never a guessed-at
            # template: an unprovable template is a refusal here rather
            # than a silent fallback to the subprocess path.
            if self.ff_model is None:
                raise LammpsPersistentUnsupported(
                    'python_enabled=true needs the compressed force-field '
                    f'model of {self.config.lammps_project!r}, and the '
                    'compressor could not prove it equivalent (see the '
                    'warning above). Set python_enabled=false to run the '
                    'verbatim templates through the subprocess path.')
            self._start_persistent_service(profile)

    def _resolve_ff_model(self, profile: str):
        """Build or reload the compressed force-field model.

        Parameters
        ----------
        profile : str
            Compiled workflow profile.

        Returns
        -------
        dict or None
            The model, or None when these templates cannot be compressed
            provably and the verbatim templates have to be used.
        """
        project = self.config.lammps_project
        # A cache left in this directory by an earlier run of the same
        # project and profile is reused when it still describes this
        # calculation. The save below and the one the package writer makes
        # are the same call to the same function: this one covers a model
        # that has not produced a package yet, that one restores the file
        # after the subprocess path clears the directory for its next step.
        path = model_path(project, self.calcdir)
        try:
            sources = read_templates(project)
            self._template_sources = sources
            model = load_cached_model(path, sources, self.config, profile)
            if model is None:
                model = build_ff_model(self.config, profile)
                os.makedirs(self.calcdir, exist_ok=True)
                save_model(path, model)
            return model
        except FFModelUnsupported as exc:
            if profile in _QM_MM_PROFILES:
                # A QM/MM task may never fall back to a template whose
                # exclusion was never proven: the run would answer with a
                # potential that still contains the QM region's classical
                # terms, and nothing downstream could tell.
                raise RuntimeError(
                    f'QM/MM needs the compressed force-field model so the QM '
                    f"region's bonded rows can be removed from the topology, "
                    f'and the compressor refused these templates: {exc}'
                ) from exc
            warnings.warn(
                f'LAMMPS force-field compression disabled ({exc}); '
                f'using the verbatim templates.')
            return None

    def _qmmm_lattice(self):
        """The cell the in-cell Q-Q correction must use, or None.

        The correction is a bare Coulomb sum, so it must use the same
        minimum image LAMMPS used inside the cell. A periodic run
        without a cell cannot answer that and is refused here rather
        than summed against a literal displacement.
        """
        if not any(int(flag) for flag in (self.config.pbc or [])):
            return None
        lattice = _resolve_periodic_lattice(self.config)
        if lattice is None:
            raise RuntimeError(
                'qmmm needs the cell of a periodic run to remove the QM '
                "region's in-cell Coulomb term, and pbc is on while no "
                'lattice or cell was handed over')
        return lattice

    def _qmmm_positions(self, coordinates):
        """QM positions in qm_atoms order, Angstrom."""
        partition = self.ff_model['qmmm']
        return [list(map(float, coordinates[index]))
                for index in partition['qm_index']]

    def _qmmm_embedding(self):
        """Whether the QM part owns the QM-MM electrostatics."""
        value = getattr(self.config, 'embedding', False)
        if isinstance(value, str):
            return value.strip().lower() not in ('', 'none', 'false', '0')
        return bool(value)

    def _resolve_qmmm_inputs(self, q_qm=None, v_mm=None, e_mm=None,
                             esfp=None, embedding=True):
        """Resolve the in-memory q_qm / native V/E for qmmm_energy.

        The static config fields are the cold/standalone fallback; a
        runtime caller passes ``qmmm_inputs`` and wins. ``.esfp`` content
        is accepted as a last resort and converted once in memory. When
        the QM calculation is not embedded (embedding=False), the V/E are
        unused and returned as zeros; q_qm is still required by the full
        force field.
        """
        units = self.config.lammps_units
        if q_qm is None:
            q_qm = list(getattr(self.config, 'q_qm', None) or [])
        if not q_qm:
            raise ValueError(
                'qmmm_energy needs q_qm: pass calculate(qmmm_inputs=...), '
                'set System.q_qm, or supply qmmm_field_file')
        q_qm = np.asarray(q_qm, dtype=np.float64)
        if not embedding:
            n_qm = len(q_qm)
            return (q_qm, np.zeros(n_qm, dtype=np.float64),
                    np.zeros((n_qm, 3), dtype=np.float64))
        if v_mm is None or e_mm is None:
            if esfp is None:
                path = str(getattr(self.config, 'qmmm_field_file', '') or '')
                if path:
                    esfp = open(path, 'r').read()
            if esfp is not None:
                v_mm, e_mm = read_esfp(esfp, units)
        if v_mm is None or e_mm is None:
            raise ValueError(
                'qmmm_energy needs the Flow-1 V/E: pass qmmm_inputs '
                'v_mm/e_mm or set qmmm_field_file')
        return (q_qm,
                np.asarray(v_mm, dtype=np.float64),
                np.asarray(e_mm, dtype=np.float64))

    def _field_environment_charges(self, qmmm_inputs, partition):
        """Resolve the charges the probe's environment carries, or None.

        With one center the environment is the MM subsystem and nothing has
        to be supplied: the return is None and the deck keeps the topology
        charges, exactly as every existing run does. With several centers
        the atoms of the other centers sit in the environment at their
        current 0-order values, so the caller must hand the whole QM set of
        charges here -- the first iteration round passes zeros, which is the
        statement 'this center sees no other center yet'.

        The neutrality of that environment is checked on the way in, not
        after the probe: the A/B difference only cancels the Ewald
        background when the environment is neutral as a whole.
        """
        env_ids = [int(value) for value in (partition.get('env_ids') or [])]
        payload = dict(qmmm_inputs or {})
        q_qm = payload.get('q_qm')
        if q_qm is None:
            q_qm = list(getattr(self.config, 'q_qm', None) or [])
        if not env_ids:
            return None
        if not q_qm:
            raise ValueError(
                'qmmm_field with an unprobed center needs q_qm: the probe '
                'environment carries the other centers 0-order charges, so '
                'calculate(qmmm_inputs={"q_qm": ...}) must supply the whole '
                'QM set (zeros are a valid first round).')
        validate_partition_charges(partition, q_qm, 'field')
        return [float(value) for value in q_qm]

    def _calculate_qmmm_field(self, atoms, coordinates, cell,
                              qmmm_inputs=None):
        """Flow 1: A/B Coulomb probe, native V/E plus the Molcas .esfp."""
        self._apply_step_inputs(atoms, coordinates, cell)
        os.makedirs(self.calcdir, exist_ok=True)
        partition = self.ff_model['qmmm']
        env_q = self._field_environment_charges(qmmm_inputs, partition)
        basename = write_field_input_file(
            self.config, self.workflow, self.calcdir, self.ff_model,
            q_qm=env_q)
        returncode = self.run(self.config, self.calcdir)
        if returncode not in (0, None):
            raise RuntimeError(
                'LAMMPS qmmm_field calculation failed with return code %s '
                '(calcdir: %s)' % (returncode, self.calcdir))
        ids = [int(value) for value in
               (partition.get('probe_ids') or partition['qm_ids'])]
        raw = {
            'pe_a': read_probe_pe(
                os.path.join(self.calcdir, f'{basename}.probe_A.pe'), ids),
            'pe_b': read_probe_pe(
                os.path.join(self.calcdir, f'{basename}.probe_B.pe'), ids),
            'forces_a': read_probe_force(
                os.path.join(self.calcdir, f'{basename}.probe_A.force'), ids),
            'forces_b': read_probe_force(
                os.path.join(self.calcdir, f'{basename}.probe_B.force'), ids),
            'groups': {
                **read_field_groups(os.path.join(
                    self.calcdir, f'{basename}.field_A.groups')),
                **read_field_groups(os.path.join(
                    self.calcdir, f'{basename}.field_B.groups')),
            },
        }
        block = assemble_qmmm_field(
            raw, partition, self.config.lammps_units)
        write_esfp(self.calcdir, basename,
                   block['v_mm_native'], block['e_mm_native'],
                   self.config.lammps_units,
                   qm_ids=block['qm_atom_ids'])
        results = {
            'energy': [0.0],
            # The probe contributes no energy and no gradient; the channel
            # exists so a task without a surface is still a shaped result.
            'gradient': to_native_gradient(np.zeros((self._natom, 3))),
            'stress_tensor': np.zeros((1, 3, 3)),
            'qmmm_field': block,
            '_native_units': self.config.lammps_units,
        }
        formatted = format_results(results, self.config)
        self._store_results(formatted)
        if self.config.environment.keep_tmp == 0:
            try:
                shutil.rmtree(self.calcdir)
            except OSError:
                pass
        return formatted

    def _calculate_qmmm_energy(self, atoms, coordinates, cell,
                               qmmm_inputs=None):
        """Flow 2: full-FF U_main plus the Python QM/MM decomposition."""
        payload = dict(qmmm_inputs or {})
        q_qm, v_mm, e_mm = self._resolve_qmmm_inputs(
            q_qm=payload.get('q_qm'),
            v_mm=payload.get('v_mm'),
            e_mm=payload.get('e_mm'),
            esfp=payload.get('esfp'),
            embedding=self._qmmm_embedding())
        partition = self.ff_model['qmmm']
        validate_partition_charges(partition, q_qm, 'energy')
        self._apply_step_inputs(atoms, coordinates, cell)
        os.makedirs(self.calcdir, exist_ok=True)
        title = write_energy_input_file(
            self.config, self.workflow, self.calcdir,
            self.ff_model, q_qm)
        returncode = self.run(self.config, self.calcdir)
        if returncode not in (0, None):
            raise RuntimeError(
                'LAMMPS qmmm_energy calculation failed with return code %s '
                '(calcdir: %s)' % (returncode, self.calcdir))
        names = step_data_files(
            title, gradient=bool(self.workflow.needs_gradient),
            stress=bool(getattr(self.config, 'stress', False)))
        energy = read_energy_file(
            os.path.join(self.calcdir, names['energy']),
            os.path.join(self.calcdir, f'{title}.log'))
        forces = _parse_force_dump(
            os.path.join(self.calcdir, names['force']), self._natom)
        stress = _build_stress_tensor(
            self.config, self.config.lammps_units,
            os.path.join(self.calcdir, names['stress'])
            if 'stress' in names else None)
        block = assemble_qmmm_energy(
            energy, forces, partition, q_qm, v_mm, e_mm,
            self._qmmm_positions(coordinates), self._qmmm_lattice(),
            self.config.lammps_units, embedding=self._qmmm_embedding())
        write_qm_multipole(self.calcdir, title,
                           partition['qm_ids'], q_qm)
        write_qmmm_energy(
            self.calcdir, title, block['u_main_native'],
            block['u_gas_native'], block['qmmm_el_native'],
            block['e_classic_native'])
        write_qm_force(self.calcdir, title, 'u_gas', partition['qm_ids'],
                       -block['u_gas_gradient_native'])
        write_qm_force(self.calcdir, title, 'qmmm_el', partition['qm_ids'],
                       -block['qmmm_el_gradient_native'])
        write_grad(self.calcdir, title, block['mm_gradient_native'],
                   self.config.lammps_units)
        results = {
            'energy': [float(energy)],
            # The classical surface the composite adds to the QM part: the
            # QM rows already have the terms the QM provider owns removed.
            'gradient': block['mm_gradient_native'],
            'stress_tensor': stress,
            'qmmm': block,
            '_native_units': self.config.lammps_units,
        }
        formatted = format_results(results, self.config)
        self._store_results(formatted)
        if self.config.environment.keep_tmp == 0:
            try:
                shutil.rmtree(self.calcdir)
            except OSError:
                pass
        return formatted

    def _calculate_persistent_field(self, atoms, coordinates, cell,
                                    qmmm_inputs=None):
        """Resident Flow 1: A/B probe on the loaded instance."""
        partition = self.ff_model['qmmm']
        env_q = self._field_environment_charges(qmmm_inputs, partition)
        with self._lock:
            if self._closed or self._session is None:
                raise LammpsPersistentError(
                    'the resident LAMMPS backend is closed; a resident '
                    'session is not restartable')
            self._apply_step_inputs(atoms, coordinates, cell)
            box = resolve_live_box(self.config, self._box_origin)
            self._step += 1
            response = self._session.evaluate(
                coordinates, box, self._step,
                image=self.config.system.image, q_qm=env_q)
        raw = response.get('qmmm_raw') or {}
        required = {'pe_a', 'pe_b', 'forces_a', 'forces_b', 'u_gg', 'f_gg'}
        if not required.issubset(raw):
            raise LammpsPersistentError(
                'the resident qmmm_field step returned no A/B probe channels')
        payload = {
            'pe_a': {int(key): float(value)
                     for key, value in raw['pe_a'].items()},
            'pe_b': {int(key): float(value)
                     for key, value in raw['pe_b'].items()},
            'forces_a': np.asarray(raw['forces_a'], dtype=float),
            'forces_b': np.asarray(raw['forces_b'], dtype=float),
            'groups': {
                'A': {'energy': float(raw['u_gg']),
                      'force': [float(value) for value in raw['f_gg']]},
                'B': {'energy': float(raw.get('u_gg', 0.0)),
                      'force': [0.0, 0.0, 0.0]},
            },
        }
        block = assemble_qmmm_field(
            payload, partition, self.config.lammps_units)
        title = output_basename(self.config.lammps_project,
                                self.workflow.profile)
        write_esfp(self.calcdir, title, block['v_mm_native'],
                   block['e_mm_native'], self.config.lammps_units,
                   qm_ids=block['qm_atom_ids'])
        results = {
            'energy': [0.0],
            # The probe contributes no energy and no gradient; the channel
            # exists so a task without a surface is still a shaped result.
            'gradient': to_native_gradient(np.zeros((self._natom, 3))),
            'stress_tensor': np.zeros((1, 3, 3)),
            'qmmm_field': block,
            '_native_units': self.config.lammps_units,
        }
        formatted = format_results(results, self.config)
        self._store_results(formatted)
        self._report_step_profile()
        return formatted

    def _calculate_persistent_energy(self, atoms, coordinates, cell,
                                     qmmm_inputs=None):
        """Resident Flow 2: full FF + Python gas/qmmm_el assembly."""
        payload = dict(qmmm_inputs or {})
        q_qm, v_mm, e_mm = self._resolve_qmmm_inputs(
            q_qm=payload.get('q_qm'),
            v_mm=payload.get('v_mm'),
            e_mm=payload.get('e_mm'),
            esfp=payload.get('esfp'),
            embedding=self._qmmm_embedding())
        partition = self.ff_model['qmmm']
        validate_partition_charges(partition, q_qm, 'energy')
        with self._lock:
            if self._closed or self._session is None:
                raise LammpsPersistentError(
                    'the resident LAMMPS backend is closed; a resident '
                    'session is not restartable')
            self._apply_step_inputs(atoms, coordinates, cell)
            box = resolve_live_box(self.config, self._box_origin)
            self._step += 1
            response = self._session.evaluate(
                coordinates, box, self._step,
                image=self.config.system.image,
                q_qm=q_qm)
        energy = float(response['energy_raw'])
        forces = np.asarray(response['forces_raw'], dtype=np.float64)
        stress = np.zeros((1, 3, 3))
        if getattr(self.config, 'stress', False):
            stress[0] = pressure6_to_stress_tensor(
                response['pressure6_raw'], self.config.lammps_units)
        block = assemble_qmmm_energy(
            energy, forces, partition, q_qm, v_mm, e_mm,
            self._qmmm_positions(coordinates), self._qmmm_lattice(),
            self.config.lammps_units, embedding=self._qmmm_embedding())
        title = output_basename(self.config.lammps_project,
                                self.workflow.profile)
        self._write_resident_energy_force(title, energy, forces, box)
        write_qm_multipole(self.calcdir, title,
                           partition['qm_ids'], q_qm)
        write_qmmm_energy(
            self.calcdir, title, block['u_main_native'],
            block['u_gas_native'], block['qmmm_el_native'],
            block['e_classic_native'])
        write_qm_force(self.calcdir, title, 'u_gas', partition['qm_ids'],
                       -block['u_gas_gradient_native'])
        write_qm_force(self.calcdir, title, 'qmmm_el', partition['qm_ids'],
                       -block['qmmm_el_gradient_native'])
        write_grad(self.calcdir, title, block['mm_gradient_native'],
                   self.config.lammps_units)
        results = {
            'energy': [energy],
            'gradient': block['mm_gradient_native'],
            'stress_tensor': stress,
            'qmmm': block,
            '_native_units': self.config.lammps_units,
        }
        formatted = format_results(results, self.config)
        self._store_results(formatted)
        self._report_step_profile()
        return formatted

    def _write_resident_energy_force(self, title, energy, forces,
                                     box=None):
        """Write the native standard energy/force files for a resident step.

        The resident path normally leaves engine files to ``debug``; the
        QM/MM contract asks for the standard files on both paths, so this
        writes the same two numbers the subprocess path writes, in the
        same dump shape the parser reads.
        """
        if getattr(getattr(self.config, 'environment', None),
                   'debug', False):
            # debug=true already appends the energy print and the force
            # write_dump to step_tail; writing them again here would give
            # one path two producers and two formats.
            return
        with open(os.path.join(self.calcdir, f'{title}.energy'),
                  'w') as handle:
            handle.write(f'{float(energy):.16e}\n')
        boundary = ('pp' if any(int(flag) for flag in
                               (self.config.pbc or [])) else 'ff')
        box = box or resolve_live_box(self.config, self._box_origin)
        if box is None:
            # Non-periodic runs have no live box; the data file's bound
            # lines still describe the dump frame.
            bounds = []
            for raw in (self.ff_model.get('box') or [])[:3]:
                tokens = raw.split('#')[0].split()[:2]
                bounds.append((float(tokens[0]), float(tokens[1])))
            if len(bounds) == 3:
                box = {'xlo': bounds[0][0], 'xhi': bounds[0][1],
                       'ylo': bounds[1][0], 'yhi': bounds[1][1],
                       'zlo': bounds[2][0], 'zhi': bounds[2][1]}
        if box is None:
            raise LammpsPersistentError(
                'cannot write the resident force dump: no box is available')
        lines = [
            'ITEM: TIMESTEP',
            str(int(self._step)),
            'ITEM: NUMBER OF ATOMS',
            str(int(self._natom)),
            f'ITEM: BOX BOUNDS {boundary} {boundary} {boundary}',
            f"{float(box['xlo']):.16e} {float(box['xhi']):.16e}",
            f"{float(box['ylo']):.16e} {float(box['yhi']):.16e}",
            f"{float(box['zlo']):.16e} {float(box['zhi']):.16e}",
            'ITEM: ATOMS id fx fy fz',
        ]
        for atom_id, force in enumerate(np.asarray(forces), start=1):
            lines.append(f'{atom_id} ' + ' '.join(
                f'{float(value):.16e}' for value in force))
        with open(os.path.join(self.calcdir, f'{title}.force'),
                  'w') as handle:
            handle.write('\n'.join(lines) + '\n')

    def _qmmm_descriptor(self, package=None):
        """The computes a resident QM/MM step reads, or None.

        The instance is loaded with the task's own static script, so the
        computes exist under these ids. The descriptor travels with INIT
        because the service, not the backend, extracts them; nothing is
        written per step.
        """
        profile = str(self.config.calculation.profile).lower()
        partition = self.ff_model.get('qmmm') if self.ff_model else None
        if not partition:
            return None
        if profile == 'qmmm_field':
            return {
                'profile': 'qmmm_field',
                # The QM set the environment charges are given for, and the
                # center whose rows the probe extracts. They differ only
                # under several centers; with one the descriptor is what it
                # always was.
                'qm_ids': list(partition['qm_ids']),
                'probe_ids': list(partition.get('probe_ids')
                                  or partition['qm_ids']),
                'peq': 'QMMM_PEQ',
                'gg': 'QMMM_GG',
                'restore_groups': list(
                    (package or {}).get('restore_groups') or []),
            }
        if profile == 'qmmm_energy':
            return {'profile': 'qmmm_energy',
                    'qm_ids': list(partition['qm_ids'])}
        return None

    def _start_persistent_service(self, profile: str) -> None:
        """Write the static package and bring the resident service up.

        Everything expensive about a LAMMPS calculation happens exactly
        here: template reading, input assembly, process launch, topology
        and force-field loading. A failure at any point releases what was
        already started, so a constructor that raises never leaves a
        service process behind.

        Parameters
        ----------
        profile : str
            Compiled workflow profile.
        """
        # The package goes into the calculation directory, under the same
        # names the subprocess path writes, so a resident run is inspected
        # with the same habits and any relative LAMMPS output lands
        # beside them.
        # One rank for the probe. With two ranks the deployed build's
        # per-atom `kspace` allocation stops summing to the group/group
        # tally once the domain decomposition straddles the moving centers:
        # round `tmp_20261104_162054` failed every multi-center `md`/`mep`
        # leg at its second evaluation (`sum(V)=-27.967` against a tally of
        # `-30.616`), the same input pinned to one rank completed, and the
        # cold path - a fresh process per evaluation - was correct at two
        # ranks as well. The probe samples a handful of atoms, so the rank
        # count buys nothing here.
        if profile == 'qmmm_field' and int(self.config.nproc) > 1:
            # `nproc` is a flat spelling of `environment.nproc`; the flat
            # name has no pydantic field, so the submodel is what gets set.
            self.config.environment.nproc = 1
            self.config.environment.lammps_nproc = 1
        if profile == 'qmmm_field':
            package = write_field_static_input_files(
                self.config, self.workflow, self.calcdir, self.ff_model)
        else:
            package = write_static_input_files(
                self.config, self.workflow, self.calcdir, self.ff_model)
        paths = launcher.service_paths(self.calcdir, self._execution_id,
                                       package['script'])
        self._paths = paths
        self._box_origin = tuple(package['origin'])

        claim = None
        handle = None
        stream = None
        try:
            claim = launcher.claim_endpoint(paths.endpoint,
                                            self._execution_id)
            self._claim = claim
            handle = launcher.start_service(self.config, paths,
                                            timeout=STARTUP_TIMEOUT_S)
            self._service = handle
            stream = launcher.connect_service(handle, STARTUP_TIMEOUT_S)
            fingerprint = self._fingerprint(profile, package)
            self._session = PersistentSession(
                stream, session=paths.endpoint, fingerprint=fingerprint,
                natom=package['natom'], order=package['order'],
                units=self.config.lammps_units,
                capabilities={
                    'gradient': bool(self.workflow.needs_gradient)
                    and profile != 'qmmm_field',
                    'stress': bool(getattr(self.config, 'stress', False))},
                expected_world=int(self.config.nproc),
                log_path=paths.log)
            info = self._session.hello()
            handle.info = info
            self._session.initialize(
                cwd=self.calcdir, script=package['script'],
                files=package['files'], profile=profile,
                pbc=list(self.config.pbc or []),
                step_tail=package['step_tail'],
                kspace=package['kspace'],
                qmmm=self._qmmm_descriptor(package))
        except Exception as exc:
            # Close reclaims the process group and the socket claim.
            # TypeError is rewritten because Calculator.initialize()
            # retries a backend whose constructor raised TypeError with a
            # single positional argument, which would drop execution_id
            # and could start a second service.
            self.close()
            if isinstance(exc, TypeError):
                raise LammpsPersistentError(
                    f'{type(exc).__name__}: {exc}') from exc
            raise
        # Bound only to the plain teardown arguments, never to self: the
        # interpreter-exit and garbage-collection paths must be able to
        # run after this object is gone. The cleanup target is the whole
        # calculation directory, the same rule the subprocess path applies
        # at the end of a step; a resident run cannot clear it per step
        # because the service loads its package from there.
        self._finalizer = weakref.finalize(
            self, _finalize_service, handle, self.calcdir,
            paths.endpoint, int(self.config.environment.keep_tmp))
        self._finalizer.atexit = True

    def _fingerprint(self, profile: str, package: dict) -> str:
        """Compute this session's identity from the loaded model.

        The same digest travels with every request, so a stale cache, an
        edited template or a swapped project is a refused step rather
        than a silently different calculation.
        """
        project = self.config.lammps_project
        data_file = os.path.join(self.calcdir, f'{project}.data')
        with open(data_file, 'r') as handle:
            data_text = handle.read()
        return compute_fingerprint(
            model=self.ff_model, init_files=package['files'],
            script=package['script'], data=data_text,
            order=package['order'], box_lines=self.ff_model.get('box', []),
            units=self.config.lammps_units,
            pbc=list(self.config.pbc or []), profile=profile,
            stress=bool(getattr(self.config, 'stress', False)),
            needs_gradient=bool(self.workflow.needs_gradient),
            shake=self.ff_model.get('constraints', {}),
            nproc=int(self.config.nproc),
            step_tail=package['step_tail'])

    def _store_results(self, results):
        """Write formatted results into the task config results dict.

        Results dict keys follow the Properties schema field names
        ('energy', 'gradient', 'nac', 'soc', 'hessian', 'frequencies', etc.)
        so the framework composite can aggregate them.
        """
        if self._task_config is not None:
            self._task_config['results'] = {
                key: (value.tolist()
                      if hasattr(value, 'tolist') else value)
                for key, value in results.items()
            }

    def _apply_step_inputs(self, atoms, coordinates, cell):
        """Record this step's dynamic inputs on the config.

        Shared by both execution paths so the geometry and cell snapshot
        semantics (not least 'cell=None keeps the initialization cell')
        cannot drift between them.

        Parameters
        ----------
        atoms : list of str
            Atom symbols.
        coordinates : list of list of float
            Atomic coordinates in Angstrom, wrapped into the main image.
        cell : dict, optional
            Frozen periodic snapshot of this evaluation: the cell that
            belongs to these coordinates plus the crossing counts that say
            which image each atom sits in.
        """
        self.config.system.atoms = atoms
        self.config.system.coordinates = coordinates
        if cell is not None:
            # Per-evaluation cell snapshot: refresh the periodic cell so
            # the simulation box is rebuilt from the lattice matching
            # these coordinates.
            self.config.system.lattice_vector = cell.get(
                'lattice_vector', self.config.system.lattice_vector)
            self.config.system.cell = cell.get(
                'cell', self.config.system.cell)
            self.config.system.pbc = cell.get(
                'pbc', self.config.system.pbc)
            if cell.get('volume') is not None:
                self.config.system.volume = cell['volume']
            # A snapshot from before the image channel carries no key at
            # all: read that as "no crossings", which is what such a run
            # meant then and what both execution paths agree on.
            self.config.system.image = [list(row)
                                        for row in (cell.get('image') or [])]

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, qmmm_inputs=None):
        """Execute a lightweight calculation step.

        Only updates dynamic information (geometry). The workflow is
        already compiled during initialization. Subprocess calls rewrite
        the coordinates in the data file and run the compiled workflow;
        resident calls send the same geometry and box to the loaded
        LAMMPS instance and take the numbers back over the socket.
        Single-point profiles perform a 'run 0' evaluation returning the
        energy in hartree and the gradient in hartree/bohr; the
        ff_optimize and MD profiles run an internal minimization /
        dynamics propagation and additionally return the final structure
        through the fin_coord channel (Angstrom).

        Parameters
        ----------
        atoms : list of str
            Atom symbols.
        coordinates : list of list of float
            Atomic coordinates in Angstrom.
        point_charges : list of list of float, optional
            Background point charges [charge, x, y, z]. Accepted for
            interface compatibility; unused by classical force fields.
        target_states : list of int, optional
            Per-step target states chosen by the workflow. Accepted
            for interface compatibility; LAMMPS is a ground-state
            method, so this argument is ignored.
        cell : dict, optional
            Frozen periodic-cell snapshot of this evaluation
            ({'lattice_vector', 'cell', 'pbc', 'volume'}); the
            authoritative per-step cell channel. In subprocess mode
            write_data_file rebuilds the simulation box from it, so a
            variable-cell (NPT) step evaluates in its own deformed box
            instead of the template box; in resident mode the same
            numbers are sent as the box of the step. None keeps the
            initialization cell
            (dev/methods/PBC_DEVELOPMENT.md section 5.3).

        Returns
        -------
        dict
            Formatted calculation results.
        """
        profile = str(self.config.calculation.profile).lower()
        if profile == 'qmmm_field':
            if self._persistent:
                return self._calculate_persistent_field(
                    atoms, coordinates, cell, qmmm_inputs=qmmm_inputs)
            return self._calculate_qmmm_field(
                atoms, coordinates, cell, qmmm_inputs=qmmm_inputs)
        if profile == 'qmmm_energy':
            if self._persistent:
                return self._calculate_persistent_energy(
                    atoms, coordinates, cell, qmmm_inputs=qmmm_inputs)
            return self._calculate_qmmm_energy(
                atoms, coordinates, cell, qmmm_inputs=qmmm_inputs)

        if self._persistent:
            return self._calculate_persistent(atoms, coordinates, cell)

        # Update dynamic information via attribute access
        self._apply_step_inputs(atoms, coordinates, cell)

        calcdir = self.calcdir

        # Create calculation directory
        os.makedirs(calcdir, exist_ok=True)

        # Write the data file (coordinates), auxiliary templates and
        # the assembled input script
        write_input_file(self.config, self.workflow, calcdir, self.ff_model)

        # Execute calculation through the runner resolved during
        # initialization
        returncode = self.run(self.config, calcdir)
        if returncode not in (0, None):
            raise RuntimeError(
                "LAMMPS calculation failed with return code %s "
                "(calcdir: %s)" % (returncode, calcdir))

        # Parse results
        results = parse_output(self.config, calcdir, self.workflow)
        results['_native_units'] = self.config.lammps_units
        formatted_results = format_results(results, self.config)

        # Write results back into the config following the schema
        self._store_results(formatted_results)

        # Cleanup if requested
        if self.config.environment.keep_tmp == 0:
            try:
                shutil.rmtree(calcdir)
            except OSError:
                pass

        return formatted_results

    def _calculate_persistent(self, atoms, coordinates, cell):
        """Evaluate one step against the resident LAMMPS instance.

        No directory is created, no input file is written, no process is
        started, and the main process does not change its working
        directory: the only thing that moves is this step's geometry.

        Parameters
        ----------
        atoms : list of str
            Atom symbols.
        coordinates : list of list of float
            Atomic coordinates in Angstrom.
        cell : dict, optional
            Frozen periodic-cell snapshot of this evaluation.

        Returns
        -------
        dict
            Results in framework units, as numpy arrays.
        """
        with self._lock:
            if self._closed or self._session is None:
                raise LammpsPersistentError(
                    'the resident LAMMPS backend is closed; a resident '
                    'session is not restartable, build a new backend '
                    'instead of falling back to the subprocess path')
            with self._phase('apply_inputs'):
                self._apply_step_inputs(atoms, coordinates, cell)

            if len(coordinates) != self._natom:
                raise LammpsPersistentError(
                    f'this backend was initialized for {self._natom} '
                    f'atoms and its resident instance cannot be resized; '
                    f'got {len(coordinates)} coordinates')

            with self._phase('resolve_box'):
                box = resolve_live_box(self.config, self._box_origin)
            self._step += 1
            with self._phase('session_evaluate'):
                response = self._session.evaluate(
                    coordinates, box, self._step,
                    image=self.config.system.image)

        with self._phase('format_results'):
            results = format_persistent_results(
                response, self.config, self.workflow)
            results['_native_units'] = self.config.lammps_units
        with self._phase('store_results'):
            self._store_results(results)
        self._report_step_profile()
        return results

    def _phase(self, name):
        """Time one part of a resident step.

        The service reports what a step costs the engine; this reports
        what it costs the framework process around that engine - the
        snapshot it applies, the box it recomputes, the round trip it
        waits on, and the two conversions that turn raw LAMMPS numbers
        into framework arrays. Without these rows the difference between
        the engine's milliseconds and the run's milliseconds has no owner,
        which is exactly the question a slow MD step asks.

        Parameters
        ----------
        name : str
            Phase label.

        Returns
        -------
        context manager
            The step profiler's timer, or a no-op before the profiler
            exists (a backend that never reached a resident step).
        """
        profiler = getattr(self, '_profiler', None)
        if profiler is None:
            profiler = self._profiler = StepProfiler()
        return profiler.measure(name)

    def _report_step_profile(self):
        """Refresh this session's framework-side table in its calcdir.

        On a cadence, and from the step itself rather than from close():
        nothing in the framework closes a resident backend, so an artifact
        written at shutdown is an artifact nobody collects. The client's
        own buckets ride along under their own label because they answer a
        different question: one table says where this process spent its
        time, the other says how the wait for the service broke into
        encode, pipe and validation.
        """
        profiler = getattr(self, '_profiler', None)
        if profiler is None or self._step % PROFILE_EVERY:
            return
        profiler.refresh(self.calcdir, self._step,
                         extra=_client_profile_rows(self.resident_timings()))

    def resident_timings(self):
        """Client-side socket time per operation of this resident session.

        Returns
        -------
        dict
            What the session's timings() reports, or empty for a
            subprocess backend, which has no session to time. The service
            writes the engine side of the same question into its own log.
        """
        session = getattr(self, '_session', None)
        return dict(session.timings()) if session is not None else {}

    def close(self, reason: str = 'owner_close') -> None:
        """Release the resident service. Idempotent, never raising.

        Ordering matters: the polite CLOSE lets every rank leave through
        the normal path, the process-group signal is what guarantees they
        actually went, and only then is the socket claim and the session
        directory released.

        Parameters
        ----------
        reason : str
            Why the session ends; recorded in the service log.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            handle = self._service
            paths = self._paths
            self._session = None
            self._service = None

            if self._finalizer is not None and self._finalizer.alive:
                # detach(), not a call: the teardown belongs to close()
                # from here on, so the interpreter-exit hook must not run
                # a second time over an already-reclaimed service.
                self._finalizer.detach()
            self._finalizer = None

            if session is not None:
                try:
                    session.close(reason)
                except Exception as error:           # noqa: BLE001
                    print(f'resident close failed: {type(error).__name__}: '
                          f'{error}', file=sys.stderr, flush=True)
            if handle is not None:
                try:
                    launcher.terminate_service(handle)
                except Exception as error:           # noqa: BLE001
                    print(f'resident terminate failed: '
                          f'{type(error).__name__}: {error}',
                          file=sys.stderr, flush=True)
            if self._claim:
                # A claim is only ever taken after self._paths is set, so
                # the socket this backend reserved is the one handed back.
                launcher.release_endpoint(paths.endpoint)
                self._claim = None
            # keep_tmp means exactly what it means on the subprocess path:
            # 0 clears the calculation directory, which holds the input
            # package, the script, the log and the socket directory.
            if (paths is not None
                    and int(self.config.environment.keep_tmp) == 0):
                shutil.rmtree(self.calcdir, ignore_errors=True)

    def __enter__(self):
        """Return the live backend for a ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the backend, including on the exception path."""
        self.close('context_exit')
        return False


def _finalize_service(handle, calcdir, endpoint, keep_tmp):
    """Reap a resident service whose backend is gone or exiting.

    Reached through weakref.finalize, so it holds no reference to the
    backend. It cannot send a polite CLOSE (the socket belongs to the
    object that just died), which is why terminating the process group is
    enough: every rank is a child of that group.

    Parameters
    ----------
    handle : launcher.ServiceHandle or None
        Service to terminate.
    calcdir : str
        Calculation directory to clean when keep_tmp is off.
    endpoint : str
        Socket claim to release.
    keep_tmp : int
        Resource-management setting, same meaning as on the cold path.
    """
    try:
        if handle is not None:
            launcher.terminate_service(handle)
    except Exception:                               # noqa: BLE001
        pass
    finally:
        if endpoint:
            launcher.release_endpoint(endpoint)
        if keep_tmp == 0 and calcdir:
            shutil.rmtree(calcdir, ignore_errors=True)
