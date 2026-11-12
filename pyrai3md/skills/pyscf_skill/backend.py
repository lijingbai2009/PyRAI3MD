"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the backend for the PySCF skill.
It is the single public computational API entry point.

Responsibilities:
- Initialize PySCF configuration (build -> normalize -> validate)
- Compile calculation workflow from profile
- Execute calculations (local in-process or slurm via subprocess)
- Return standardized results

Core principle:
Compile the computational workflow once during backend initialization,
then execute lightweight calculate() calls during dynamics.

"""

import os
import shutil

from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import validate
from .schema.method import resolve_profile
from .service.workflow import compile_workflow
from .service.runner import get_runner, results_file_name, validate_environment
from .service.parser import parse_output, format_results
from .service.utils import build_calcdir, flatten_extended_config


class PySCFBackend:
    """Main backend class for PySCF calculations.

    Initializes once, then serves lightweight calculate() calls.
    The workflow is compiled from the profile during initialization.

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
        Explicit calculation directory path.
    execution_id : str, optional
        Execution namespace identifier assigned by the Calculator
        (e.g. 'tmp_PYSCF' (single task) or 'tmp_high_qm_PYSCF'). Used directly as the calculation
        directory name; empty string falls back to the standalone
        tmp_PYSCF naming.
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
        if isinstance(raw_config, dict) and not raw_config.get('profile'):
            method = raw_config.get('method', '')
            jobtype = raw_config.get('jobtype', 'energy')
            if method:
                raw_config = dict(raw_config)
                raw_config['profile'] = resolve_profile(method, jobtype)

        # Run the skill schema pipeline (build -> normalize -> validate)
        # in both modes. The result is a nested SkillConfig held as
        # self.config throughout the backend lifetime (no config_dict).
        input_config = build_config(raw_config)
        normalized_config = normalize(input_config)
        self.config = validate(normalized_config)

        # Ensure scratch_path exists (default to cwd if not set)
        scratch_path = self.config.environment.scratch_path or os.getcwd()
        scratch_path = os.path.expanduser(scratch_path)
        os.makedirs(scratch_path, exist_ok=True)
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
        if calcdir:
            self.calcdir = calcdir
        else:
            self.calcdir = build_calcdir(
                execution_id=execution_id, base_path=scratch_path,
                project=self.config.calculation.project,
            )

        # A local run keys its artifacts (log, chk, molden, opt.xyz) on
        # config.scratch_path, which every term of the skill shares; the
        # slurm arm already writes them under calcdir. Pointing the local
        # arm at its own calculation directory makes those files per-term
        # without renaming any of them (#302). The slurm arm is left alone
        # on purpose: its config travels to the child as a json file, and
        # a calcdir-valued scratch would nest a second calculation
        # directory inside the first.
        if self.config.environment.execution_mode != 'slurm':
            os.makedirs(self.calcdir, exist_ok=True)
            self.config.environment.scratch_path = self.calcdir

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

        Parameters
        ----------
        atoms : list of str
            Atom symbols.
        coordinates : list of list of float
            Atomic coordinates in Angstrom.
        point_charges : list of list of float, optional
            Background point charges.
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

        # Execute calculation through the runner resolved during
        # initialization; the execution_mode keyword (local/slurm) is
        # honored as decided by the service (get_runner).
        execution_mode = self.config.environment.execution_mode
        if execution_mode == 'slurm':
            # Cluster run: run_slurm serializes the config into the
            # calculation directory and submits a blocking sbatch
            # job; the agent writes the formatted results next to
            # the config json for collection here.
            os.makedirs(self.calcdir, exist_ok=True)
            self.run(self.config, self.calcdir)
            project = self.config.calculation.project or 'PYSCF'
            config_file = os.path.join(
                self.calcdir, f"{project}.json")
            formatted_results = read_json(
                results_file_name(config_file))
        else:
            # Local run: in-process execution through the service
            # Log file name follows Calculation.project so each
            # profile gets its own log (e.g. hf_energy.log).
            project = self.config.calculation.project or 'PYSCF'
            log_name = f'{project}.log'
            scratch_path = self.config.environment.scratch_path
            log_file = os.path.join(scratch_path, log_name)

            runtime = self.run(self.config, self.workflow,
                               log_file=log_file)

            # Parse and format results
            results = parse_output(runtime, self.config, self.workflow)
            formatted_results = format_results(results, self.config)

        # Guarantee the fin_coord contract on every path: skills
        # performing internal iteration (optimize/md/...) report the
        # final structure through this key; paths without internal
        # iteration return an empty value explicitly.
        formatted_results.setdefault('fin_coord', [])

        # Reserved fin_velocity return channel: the pyscf skill has no
        # MD velocity output, so the channel stays empty on every path
        formatted_results.setdefault('fin_velocity', [])

        # Write results back into the config following the schema
        self._store_results(formatted_results)

        # Cleanup if requested: the log, chk, molden and trajectory
        # files are what a run reads back, so keep_tmp decides whether
        # the whole calculation directory survives (#306). The next
        # local run recreates it (service/runner/local_runner.py makes
        # the log directory), so a stepped workflow is not left behind.
        if int(self.config.environment.keep_tmp) == 0:
            try:
                shutil.rmtree(self.calcdir)
            except OSError:
                pass

        return formatted_results
