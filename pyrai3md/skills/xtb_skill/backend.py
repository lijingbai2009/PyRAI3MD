"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the backend for the xTB skill.
It is the single public computational API entry point.

Responsibilities:
- Initialize xTB configuration (build -> normalize -> validate)
- Compile calculation workflow from profile
- Resolve calculation directory (auto-increment for multiple backends)
- Execute calculations (local/slurm via subprocess)
- Return standardized results

Core principle:
Compile the computational workflow once during backend initialization,
then execute lightweight calculate() calls during dynamics.
The calculation directory is fixed per backend instance, so that
the GFN-FF topology file (gfnff_topo) built in the first step can
be reused by subsequent calculations.

"""

import os
import shutil

from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import validate
from .service.workflow import compile_workflow
from .service.builder import write_input_file
from .service.runner import get_runner, validate_environment
from .service.parser import parse_output, format_results
from .service.read import read_calculation
from .service.utils import build_calcdir, flatten_extended_config


class XtbBackend:
    """Main backend class for xTB calculations.

    Initializes once, then serves lightweight calculate() calls.
    The calculation directory is determined at initialization and
    remains fixed across all calculate() calls. A dynamics run has the
    GFN-FF topology file from its first step reused by the steps after it
    (`keep_topology`, derived by the framework from the jobtype); every
    other route drops that file before each call, so a geometry is always
    scored on the bond list built from that geometry.

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
        (e.g. 'tmp_XTB' (single task) or 'tmp_high_qm_XTB'). Used directly as the calculation
        directory name; empty string falls back to the standalone
        tmp_XTB naming.
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
        # defaults; type conversion and derived fields (uhf, state labels,
        # ...) are computed here. The result is a nested SkillConfig held
        # as self.config throughout the backend lifetime (no config_dict).
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

        Only updates dynamic information (geometry, charges).
        The workflow is already compiled during initialization.
        The calculation directory remains fixed, allowing the GFN-FF
        topology file from the first step to be reused.

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
            overrides them for this step. xTB computes the ground
            state only, so this argument is accepted for interface
            compatibility.
        cell : dict, optional
            Frozen periodic-cell snapshot of this evaluation
            ({'lattice_vector', 'cell', 'pbc', 'volume'}); the
            authoritative per-step cell channel applied before the
            coord file is written, so a variable-cell step evaluates
            with its own deformed lattice. None keeps the
            initialization cell (dev/methods/PBC_DEVELOPMENT.md
            section 5.3).

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
            # write_coord_file uses the lattice matching these
            # coordinates. The skill pbc field has a BeforeValidator, so
            # assigning the plain list is normalized.
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

        if not getattr(self.config, 'keep_topology', False):
            # Drop the bond list the previous evaluation left behind. The
            # directory is fixed for this backend's whole life, so xTB reads
            # that file back and scores the current geometry against a topology
            # it did not build. For a dynamics run that is the point, and the
            # framework says so with keep_topology; for any other route the
            # geometry moves on the search's own terms and the stale list costs
            # the energy: measured in the test environment, carrying a clashed run's
            # gfnff_topo onto a sane geometry moved it by +0.3508098 Hartree,
            # and round tmp_20261017_234656's variable-cell rows 6-21 read
            # +0.20 while those very frames evaluate fresh to -0.929.
            stale_topo = os.path.join(calcdir, 'gfnff_topo')
            if os.path.exists(stale_topo):
                os.remove(stale_topo)

        # Write geometry, embedding, and inp files; build xTB arguments
        write_input_file(self.config, self.workflow, calcdir)

        # Execute calculation through the runner resolved during
        # initialization
        returncode = self.run(self.config, calcdir)

        # Parse results
        results = parse_output(self.config, calcdir, self.workflow)
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
