"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the backend for the CP2K skill.
It is the single public computational API entry point.

Responsibilities:
- Initialize CP2K configuration (build -> normalize -> validate)
- Compile calculation workflow from profile
- Resolve calculation directory (auto-increment for multiple backends)
- Execute calculations (local/slurm via subprocess): one CP2K run per spin
  manifold, as `service/jobs.py` plans them, stitched back into one result
- Return standardized results

Core principle:
Compile the computational workflow once during backend initialization,
then execute lightweight calculate() calls during dynamics.
The calculation directory is fixed per backend instance.

"""

import os
import shutil

from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import validate
from .service.workflow import compile_workflow
from .service.builder import write_input_file
from .service.jobs import plan_jobs
from .service.runner import get_runner, validate_environment
from .service.parser import (parse_output, merge_job_results,
                             format_results)
from .service.read import read_calculation
from .service.stress import compute_num_stress
from .service.utils import build_calcdir, flatten_extended_config


class Cp2kBackend:
    """Main backend class for CP2K calculations.

    Initializes once, then serves lightweight calculate() calls.
    The calculation directory is determined at initialization and
    remains fixed across all calculate() calls.

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
        (e.g. 'tmp_CP2K' (single task) or 'tmp_high_qm_task1_CP2K'). Used directly as the calculation
        directory name; empty string falls back to the standalone
        tmp_CP2K naming.
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
        # defaults; type conversion and derived fields (tddft_nstates,
        # element-level basis, state labels, ...) are computed here.
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

        # Compile workflow from profile
        profile = self.config.calculation.profile
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
                  target_states=None, cell=None):
        """Execute a lightweight calculation step.

        Only updates dynamic information (geometry, charges, target
        states). The workflow is already compiled during
        initialization.
        The calculation directory remains fixed across steps.

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
            Frozen periodic-cell snapshot of this evaluation
            ({'lattice_vector', 'cell', 'pbc', 'volume'}). The
            authoritative per-step cell channel: it is applied to this
            backend's config.system before the input is written, so a
            variable-cell (NPT) step evaluates with its own deformed
            cell instead of the initialization-time cell. None keeps
            the initialization cell unchanged (dev/methods/
            PBC_DEVELOPMENT.md section 5.3).

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
        if cell is not None:
            # Per-evaluation cell snapshot: refresh the periodic cell so
            # the generated input uses the cell that matches these
            # coordinates (the __init__ config is frozen; sync() only
            # runs after the step). The skill pbc field has a
            # BeforeValidator, so assigning the plain list is normalized.
            self.config.system.lattice_vector = cell.get(
                'lattice_vector', self.config.system.lattice_vector)
            self.config.system.cell = cell.get(
                'cell', self.config.system.cell)
            self.config.system.pbc = cell.get(
                'pbc', self.config.system.pbc)
            if cell.get('volume') is not None:
                self.config.system.volume = cell['volume']

        calcdir = self.calcdir

        # Create calculation directory
        os.makedirs(calcdir, exist_ok=True)

        # One evaluation is a LIST of CP2K runs, decided by `service/jobs.py`: a
        # request naming one spin manifold is one run and the original config, so
        # nothing below changes shape for the shipped cases; a request naming both
        # singlet and triplet excited rows is two runs of the same reference,
        # because one run solves exactly one manifold (measured in the test environment,
        # and `qs_tddfpt2_methods.F:294-297/421/1190`).
        jobs = plan_jobs(self.config, self.workflow)
        results = []
        for job in jobs:
            # Build and write this job's CP2K input file
            write_input_file(job.config, self.workflow, calcdir)

            # Execute calculation through the runner resolved during
            # initialization
            returncode = self.run(job.config, calcdir)

            # Propagate execution failure: the run script exits with the
            # CP2K return code, so a non-zero value means a module
            # crashed. Raising here prevents parse_output from masking
            # the failure with zero-filled results.
            if returncode != 0:
                raise RuntimeError(
                    f'CP2K calculation failed with return code '
                    f'{returncode} (calcdir: {calcdir}, project: '
                    f'{job.config.project}). Check the '
                    f'CP2K log for the failing module.'
                )

            # Parse results
            results.append(parse_output(job.config, calcdir, self.workflow))

        # Stitch the runs back into the framework's single set of arrays: each
        # state row is taken from the run that solved its manifold, and the two
        # readings of the ground state have to agree.
        results = merge_job_results(jobs, results, self.config)
        formatted_results = format_results(results, self.config)

        # Numerical (external strain) stress: the tensor is not read
        # from this run's output at all -- the route asks CP2K for no
        # stress keyword -- but differentiated out of the perturbed
        # single points planned by the skill. It replaces the
        # zero-filled array for every state, then travels to the
        # framework through the same stress_tensor channel and the same
        # manifest-declared unit as the analytical tensor.
        if (getattr(self.config, 'stress', False)
                and str(getattr(self.config, 'stress_method',
                                'analytical')).lower() == 'num'):
            formatted_results['stress_tensor'] = compute_num_stress(
                self.config, calcdir, results).tolist()

        # Write results back into the config following the schema
        self._store_results(formatted_results)

        # Cleanup if requested
        if self.config.environment.keep_tmp == 0:
            try:
                shutil.rmtree(calcdir)
            except OSError:
                pass

        return formatted_results
