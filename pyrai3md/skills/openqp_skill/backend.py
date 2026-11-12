"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the backend for the OpenQP skill.
It is the single public computational API entry point.

Responsibilities:
- Initialize OpenQP configuration (build -> normalize -> validate)
- Compile calculation workflow from profile
- Resolve calculation directory (fixed per backend instance)
- Execute calculations (local/slurm via subprocess)
- Return standardized results

Core principle:
Compile the computational workflow once during backend initialization,
then execute lightweight calculate() calls during dynamics.
The calculation directory is fixed per backend instance, so that the
orbital json (${OPENQP_PROJECT}.json, saved via save_mol=true) from
previous steps is automatically reused as the initial guess by
OpenQP through the fixed file name mechanism (no explicit orbital
file management is needed).

"""

import os
import shutil

from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import soc_is_requested, validate
from .schema.method import resolve_profile
from .service.workflow import compile_workflow
from .service.builder import write_input_file, write_xyz_file
from .service.runner import get_runner, validate_environment
from .service.parser import parse_output, format_results
from .service.read import read_calculation
from .service.utils import build_calcdir, flatten_extended_config


class OpenQPBackend:
    """Main backend class for OpenQP calculations.

    Initializes once, then serves lightweight calculate() calls.
    The calculation directory is determined at initialization and
    remains fixed across all calculate() calls, enabling automatic
    orbital continuation: every step saves the orbital json as
    ${OPENQP_PROJECT}.json (save_mol=true), and OpenQP reads it back
    through [guess]file (and [guess]file2 for nacme/dcm).

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
        (e.g. 'tmp_OPENQP' (single task) or 'tmp_high_qm_task1_OPENQP'). Used directly as the calculation
        directory name; empty string falls back to the standalone
        tmp_OPENQP naming.
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

        # Derive profile from method + jobtype if not explicitly set
        # (classic input / standalone flow carries method + jobtype only).
        if isinstance(raw_config, dict) and not raw_config.get('profile'):
            method = raw_config.get('method', '')
            jobtype = raw_config.get('jobtype', 'energy')
            if method:
                raw_config = dict(raw_config)
                raw_config['profile'] = resolve_profile(method, jobtype)

        # Run the skill schema pipeline (build -> normalize -> validate)
        # in both modes. The framework merge only carries raw values and
        # defaults; type conversion and derived fields (nact_elec,
        # state labels, nac pair maps, ...) are computed here.
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

        # Compile workflow from profile (nac_type selects the NAC
        # runtype for *_nac profiles: nac vector vs nacme; dosoc and an
        # explicit grad_t list select the two-manifold spin-orbit path)
        profile = self.config.calculation.profile
        nac_type = self.config.calculation.nac_type
        self.workflow = compile_workflow(
            profile,
            nac_type=nac_type,
            is_soc=soc_is_requested(self.config.calculation),
            has_grad_t=bool(self.config.calculation.grad_t),
        )

        # Resolve calculation directory under the scratch path
        # (fixed for this backend instance)
        if calcdir:
            self.calcdir = calcdir
        else:
            self.calcdir = build_calcdir(
                execution_id=execution_id, base_path=scratch_path or None,
                project=self.config.calculation.project,
            )

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

    def _cleanup_calcdir(self, calcdir: str):
        """Clean the calculation directory while protecting the orbital json.

        The orbital json (${OPENQP_PROJECT}.json) is the fixed-name
        carrier of the orbital continuation mechanism; it must stay
        in place so that the next step can read it back. All other
        files are removed.

        Parameters
        ----------
        calcdir : str
            Calculation directory.
        """
        project = self.config.project
        protected = {f'{project}.json'}

        try:
            for entry in os.listdir(calcdir):
                if entry in protected:
                    continue
                entry_path = os.path.join(calcdir, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    os.remove(entry_path)
        except OSError:
            pass

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        """Execute a lightweight calculation step.

        Only updates dynamic information (geometry, charges, target
        states). The workflow is already compiled during
        initialization.
        The calculation directory remains fixed, allowing the orbital
        json from previous steps to be reused as initial guess via
        the OpenQP fixed file name mechanism.

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
        if point_charges is not None:
            self.config.system.point_charges = point_charges
        if target_states is not None:
            self.config.calculation.target_states = target_states

        calcdir = self.calcdir

        # Create calculation directory
        os.makedirs(calcdir, exist_ok=True)

        # Write xyz file (coordinates referenced by $OPENQP_PROJECT.xyz)
        write_xyz_file(self.config, calcdir)

        # Build and write OpenQP input file
        write_input_file(self.config, self.workflow, calcdir)

        # Execute calculation through the runner resolved during
        # initialization. A non-zero exit code means the OpenQP run
        # failed (aborted input, crash, scheduler kill): raise instead
        # of parsing whatever partial exports happen to exist.
        returncode = self.run(self.config, calcdir)
        if returncode != 0:
            raise RuntimeError(
                "OpenQP calculation failed with return code %s "
                "(calcdir: %s)" % (returncode, calcdir))

        # Parse results from the exported fixed-name text files
        results = parse_output(self.config, calcdir, self.workflow)
        formatted_results = format_results(results, self.config)

        # Write results back into the config following the schema
        self._store_results(formatted_results)

        # Cleanup if requested (the orbital json is protected so the
        # fixed-name continuation mechanism keeps working)
        if self.config.environment.keep_tmp == 0:
            self._cleanup_calcdir(calcdir)

        return formatted_results
