"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the backend for the Molcas skill.
It is the single public computational API entry point.

Responsibilities:
- Initialize Molcas configuration (build -> normalize -> validate)
- Compile calculation workflow from profile
- Resolve calculation directory (auto-increment for multiple backends)
- Execute calculations (local/slurm via subprocess)
- Return standardized results

Core principle:
Compile the computational workflow once during backend initialization,
then execute lightweight calculate() calls during dynamics.
The calculation directory is fixed per backend instance, so that
orbital files (RasOrb) from previous steps can be reused as
initial guesses for subsequent calculations.

"""

import os
import shutil

import numpy as np
from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import validate
from .schema.method import has_transition_request
from .service.workflow import compile_workflow
from .service.builder import (write_input_file, write_xyz_file,
                              write_velocity_file, manage_orbitals)
from .service.runner import get_runner, validate_environment, resolve_workdir
from .service.parser import parse_output, format_results
from .service.read import read_calculation
from .service.parser.espf_charges import parse_espf_charges
from .service.utils import build_calcdir, flatten_extended_config

# The embedding words whose run leaves post-ESPF 0-order charges behind to
# be read. These are the electrostatic choices the skill's own schema accepts
# minus 'none'; kept as a name so a test can hold it against
# schema/config.py's vocabulary instead of trusting this line, which is what
# let a retired word sit here while the word QMMM actually sends was missing.
_ESPF_CHARGE_EMBEDDINGS = ('charge', 'field')


def _run_not_converged(config, calcdir: str) -> bool:
    """Decide whether a failed return code is an unconverged run.

    SLAPAF exhausts its step budget with a Stop Module line carrying
    /rc=_RC_NOT_CONVERGED_ (pymolcas exit code 96); that is a result
    status rather than a crash, so the backend falls through to
    parsing instead of raising. Missing logs or logs without the
    marker stay fail-fast.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container (project name).
    calcdir : str
        Calculation directory.

    Returns
    -------
    bool
        True when the log records an unconverged SLAPAF stop.
    """
    names = [f'{config.project}.log']
    if str(getattr(config, 'execution_mode', 'local')).lower() == 'slurm':
        # The skill's Slurm runner redirects the engine log to the
        # `slurm_artifacts.log` suffix declared in manifest.yaml.
        names.append(f'{config.project}_slurm.out')
    for name in names:
        log_path = os.path.join(calcdir, name)
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if ('Stop Module' in line and '/rc=' in line
                            and '_RC_NOT_CONVERGED_' in line):
                        return True
        except OSError:
            continue
    return False


class MolcasBackend:
    """Main backend class for OpenMolcas calculations.

    Initializes once, then serves lightweight calculate() calls.
    The calculation directory is determined at initialization and
    remains fixed across all calculate() calls, enabling orbital
    file reuse (RasOrb -> StrOrb) as initial guesses.

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
        (e.g. 'tmp_MOLCAS' (single task) or 'tmp_high_qm_task1_MOLCAS'). Used directly as the calculation
        directory name; empty string falls back to the standalone
        tmp_MOLCAS naming.
    """

    # QMMM potential embedding is delivered as an optional per-step array.
    supports_external_potential = True
    supports_nac_request = True
    supports_soc_request = True

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
        # defaults; type conversion and derived fields (ras2_orb,
        # inactive_orb, ci_roots, state labels, ...) are computed here.
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

        # Validate external program environment
        validate_environment(self.config)

        # Resolve the execution runner once: execution_mode is fixed
        # at initialization (no runtime path mutates it), so the mode
        # decision stays with the service (get_runner) and the result
        # is cached for lightweight calculate() calls.
        self.run = get_runner(self.config)

        # Compile workflow from profile. A transition request in the
        # framework jobtype appends a RASSI step to the module chain
        # (energy template upgraded to the absorption one).
        profile = self.config.calculation.profile
        transition = has_transition_request(
            self.config.calculation.jobtype)
        self.workflow = compile_workflow(
            profile, transition=transition, config=self.config)

        # Resolve calculation directory under the scratch path
        # (fixed for this backend instance)
        if calcdir:
            self.calcdir = calcdir
        else:
            self.calcdir = build_calcdir(
                execution_id=execution_id, base_path=scratch_path or None,
                project=self.config.calculation.project,
            )

        # Resolve working directory from calculation.scratch_path
        # (molcas decides the AUTO fallback for its own IO need). The
        # calculation directory names the term underneath it, so two
        # terms never share the scratch the other one deletes.
        self.workdir = resolve_workdir(
            self.config, self.config.environment.scratch_path,
            term=os.path.basename(str(self.calcdir).rstrip('/')),
            run_id=self.calcdir)

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

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None,
                  external_potential=None, nac_pairs=None,
                  soc_pairs=None):
        """Execute a lightweight calculation step.

        Only updates dynamic information (geometry, charges, target
        states, requested couplings). The workflow is already compiled
        during initialization.
        The calculation directory remains fixed, allowing orbital
        files from previous steps to be reused as initial guesses.

        Parameters
        ----------
        atoms : list of str
            Atom symbols.
        coordinates : list of list of float
            Atomic coordinates in Angstrom.
        point_charges : list of list of float, optional
            Background point charges [charge, x, y, z].
        target_states : list of int, optional
            Per-step target states chosen by the workflow. None keeps
            the target_states fixed at initialization; otherwise
            overrides them for this step.
        soc_pairs : list of list of int, optional
            State pairs this step wants spin-orbit elements for. None keeps
            the pairs from initialization.
        nac_pairs : list of list of int, optional
            State pairs this step wants derivative couplings for, as
            0-based global state indices. None keeps the pairs from
            initialization.
        cell : dict, optional
            Periodic-cell snapshot, accepted for interface
            compatibility. This is a molecular (non-periodic) backend
            and the skill validation rejects pbc, so cell is always
            None here and is ignored.

        Returns
        -------
        dict
            Formatted calculation results.
        """
        # Update dynamic information via attribute access
        self.config.system.atoms = atoms
        self.config.system.coordinates = coordinates
        # A step that brings no rows has none: the alternative is carrying the
        # previous step's charges into the new geometry, and a run whose field
        # comes from a file then never gets back to the file.
        if point_charges is not None:
            self.config.system.point_charges = point_charges
        else:
            self.config.system.point_charges = []
        if external_potential is not None:
            self.config.system.external_potential = external_potential
        else:
            # The same rule as the charges above, and the same reason: a
            # charge iteration calls this backend once per round, and a
            # round that hands over no rows must not be solved in the
            # previous round's field.
            self.config.system.external_potential = []
        if target_states is not None:
            self.config.calculation.target_states = target_states
        if nac_pairs is not None:
            # The pair travels per step, the same way the target states do:
            # `&ALASKA` prints a coupling table only for a pair the deck
            # names, and the deck is rebuilt here every step.
            self.config.calculation.target_nac_pair = [
                list(pair) for pair in nac_pairs]
        if soc_pairs is not None:
            # Same rule on the spin-orbit channel: it is the list the reader
            # addresses its rows by, so a step that wants a subset says so
            # here instead of getting every pair the config declares.
            self.config.calculation.target_soc_pair = [
                list(pair) for pair in soc_pairs]

        calcdir = self.calcdir

        # Create calculation directory
        os.makedirs(calcdir, exist_ok=True)

        # Write xyz file (coordinates referenced by $MOLCAS_PROJECT.xyz)
        write_xyz_file(self.config, calcdir)

        # Write initial velocity file for dynamics (read by DYNAMIX
        # via $MOLCAS_PROJECT.velocity.xyz)
        if self.config.system.velocities:
            write_velocity_file(self.config, calcdir)

        # Manage orbital files (RasOrb/StrOrb/JobIph)
        manage_orbitals(self.config, calcdir)

        # Build and write Molcas input file
        write_input_file(self.config, self.workflow, calcdir)

        # Execute calculation through the runner resolved during
        # initialization
        returncode = self.run(self.config, calcdir, self.workdir)

        # Propagate execution failure: the run script exits with the
        # pymolcas return code, so a non-zero value usually means a
        # module crashed (e.g. missing shared libraries). Raising
        # here prevents parse_output from masking the failure with
        # zero-filled results. The unconverged-optimization stop
        # (_RC_NOT_CONVERGED_, rc=96) is an exception: it is a
        # result status, recorded through the opt_converge contract.
        if returncode != 0:
            if not _run_not_converged(self.config, calcdir):
                raise RuntimeError(
                    f'Molcas calculation failed with return code '
                    f'{returncode} (calcdir: {calcdir}). Check the '
                    f'Molcas log for the failing module.'
                )
            # Not converged: fall through to parse_output, which
            # records opt_converge=False and collects the final
            # structure written by SLAPAF.

        # Parse results
        results = parse_output(self.config, calcdir, self.workflow)
        # The post-ESPF 0-order charges are the framework's
        # computed_charges channel: the QMMM classical seed and the
        # charge block the run log prints. A layout this reader does not
        # recognise yields no charges rather than a guessed number.
        #
        # The vocabulary is the skill's own, which holds 'charge' and
        # 'field' and refuses 'potential' at config build
        # (schema/config.py:_normalize_embedding). Asking for 'potential'
        # here therefore reached nothing, while the one word a QMMM run
        # actually carries was left out: the reader never ran on that path,
        # and every shipped QMMM example had to hand the framework its
        # `molecule.q_qm` because no charge could come back.
        _embedding = str(
            getattr(self.config.system, 'embedding', 'none') or 'none').lower()
        if _embedding in _ESPF_CHARGE_EMBEDDINGS:
            # The charges are returned one row per state, in the order
            # `&RASSCF` prints its roots across the run's spin manifolds -
            # which is the global 0-based state numbering the framework
            # uses. Which state answers is not this reader's question: a run
            # under MD embeds with `current_state`, a crossing search needs
            # the pair, and only the framework knows which it is. Binding the
            # blocks to a state is what makes that possible; the old rule of
            # taking whichever block came last handed a `caspt2` response
            # vector over as if it were a state.
            _charges = parse_espf_charges(calcdir, len(atoms))
            if _charges is not None:
                results['computed_charges'] = np.asarray(
                    _charges, dtype=np.float64)
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

