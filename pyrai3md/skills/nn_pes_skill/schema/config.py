"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the unified configuration schema for the nn_pes
skill: where a trained ensemble lives, which ensemble members to serve,
what a training run needs, and the runtime environment. No numpy array
appears in the schema, because arrays are not JSON serializable.

"""

from typing import List

from pydantic import BaseModel, Field, model_validator

# Here we define some default values
DEFAULT = {
    'execution_mode': 'local',
    'project': 'NNPES',
    'profile': 'mlp_eg_gradient',
    'method': 'mlp_eg',
    # Shared machine-learning settings, named as the original project named
    # them. One folder per role lives inside modeldir, named after the role:
    # 'energy_grad' and 'nac'. A training file carries Hartree energies and
    # Hartree/Bohr gradients and couplings, the framework's own convention;
    # there is no unit keyword, because the only rule a file has to satisfy
    # is that its energy and its gradient describe one surface, and the
    # trainer enforces that itself (service/trainer_data.py:match_labels).
    'modeldir': '',
    'train_data': '',
    # The same training set handed over inline instead of by file name.
    # The framework expands a permutation map into this mapping before
    # the backend is initialized (workflow/train.py), so the skill never
    # reads a map or reorders atoms itself. `train_data` and this key are
    # mutually exclusive; an inline dataset takes the loader's same
    # shape/unit rules as a file.
    'train_dataset': {},
    # The test set a 'prediction' request measures the trained surface
    # against. It is read by the same rules as train_data (same loader,
    # same unit convention); a training file is not needed for it, because
    # the scaler a metric needs lives in the model folder. A serving
    # request that carries this field is a test-set evaluation: one pass
    # over the file's frames, and an error table, instead of one geometry.
    'pred_data': '',
    # The framework's control-level seed, under its own name: the merge that
    # hands a skill its configuration flattens `control` into bare keys, so a
    # run that sets gl_seed reaches training without a translation table. The
    # default is the framework's own default, and it is what makes a fit
    # reproducible on its own terms - a child process inherits no random state
    # from its parent, so without a number on the command line every rerun
    # draws fresh weights. It travels to each member as gl_seed + member index
    # (service/runner/train_runner.py), the rule an MD run uses per trajectory.
    'gl_seed': 1,
    'train_mode': 'training',
    # Ensemble size. This is what decides how many member groups of a
    # hyperparameter are read: fewer broadcast, surplus are ignored.
    'mult_nn': 2,
    # Serving predictor selection: 0 returns the member mean; k>0 (1-based)
    # returns member k as the prediction and writes the other members to the
    # alternative_* channels for the adaptive analysis layer.
    'nn_predict_mode': 0,
    'nstate': 2,
    'nnac': 0,
    # Per-member GPU index of the training subprocesses; ';' separates
    # members, -1 means no device.
    'gpu': '-1',
    # Role sections carry the per-member hyperparameters of one network
    # family; schema/hyper.py owns their vocabulary. The section a profile
    # trains is named after the ensemble it fits: 'eg', 'nac', 'soc'.
    'eg': {},
    'nac': {},
    'soc': {},
    'target_states': [0, 1],
    'target_nac_pair': [[]],
    'target_soc_pair': [[]],
    'charges': 0,
    'multiplicity': 1,
    # State information (from framework)
    'ci': [1],
    'spin': [0],
    'coupling': [],
    'pbc': [False, False, False],
    'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0],
    'lattice_vector': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'kpoint': [1, 1, 1],
    'nn_project': 'NNPES',
    'scratch_path': '',
    # Threads, in two spellings (#306): the bare name is what a standalone
    # input writes and what this skill reads, the prefixed name what the
    # framework forwards; the larger wins.
    'thread': 1,
    'nn_pes_thread': 1,
    # Separate budgets for the two TensorFlow lifetimes: training runs one
    # subprocess per ensemble member, serving runs in the MD child process.
    # 0 means "inherit thread".
    'train_thread': 0,
    'predict_thread': 0,
    'slurm_time': '01:00:00',
    'slurm_partition': 'default',
    'slurm_ntasks': 1,
    'slurm_command': '',
    'slurm_memory': 4000,
    'keep_tmp': 0,
    # Per-step diagnostic files (step log and uncertainty dump). 0 keeps
    # a dynamics run from paying the I/O; a test that reads those files
    # sets it to 1. The ensemble spread itself is still computed because
    # it is the adaptive sampling signal.
    'write_step_log': 0,
    # Post-fit prediction spread on the training frames, reported as
    # `&nn prediction max std`. Training/prediction only; a serving call
    # never enters this analysis.
    'predict_std': 0,
}


class System(BaseModel):
    # system information
    atoms: List[str] = []                      # (na, 1)
    mass: List[float] = []                     # (na, 1)
    coordinates: List[List[float]] = []        # (na, 3) Angstrom
    velocities: List[List[float]] = []         # (na, 3) Bohr/au
    point_charges: List[List[float]] = []      # (ns, na)

    charges: int = DEFAULT['charges']
    multiplicity: int = DEFAULT['multiplicity']
    natom: int = 0

    # State information (from framework)
    ci: List[int] = DEFAULT['ci']
    spin: List[int] = DEFAULT['spin']
    coupling: List[List[int]] = DEFAULT['coupling']
    nstate: int = DEFAULT['nstate']
    current_state: int = 0      # Currently active electronic state (0-based)
    state_label: List[str] = []      # (ns,)
    state_spin: List[int] = []       # (ns, 1)

    npair: int = 0
    nnac: int = DEFAULT['nnac']
    nsoc: int = 0                             # (nsoc,) declared SOC pairs

    transition_pair: List[List[int]] = [[]]    # (npair, 2)
    nac_pair_map: List[List[int]] = [[]]       # (nnac, 2)
    soc_pair_map: List[List[int]] = [[]]      # (nsoc, 2) spin-orbit pairs

    pbc: List[bool] = DEFAULT['pbc']                    # (3,)
    cell: List[float] = DEFAULT['cell']                 # (6,)
    lattice_vector: List[List[float]] = DEFAULT['lattice_vector']  # (3, 3)
    kpoint: List[int] = DEFAULT['kpoint']               # (3,)


class Properties(BaseModel):
    # computed properties reported by the nn_pes backend
    energy: List[float] = [0.0]                         # (ns,) Hartree
    gradient: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]   # (ns, na, 3)
    # Hartree/Bohr, ordered exactly as the requested nac pairs
    nac: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]      # (nnac, na, 3)
    # One spin-orbit matrix element per declared pair, in cm-1. A pair that
    # was never declared is an empty channel, not a zero-filled one, so a
    # couplings consumer can still tell "none" from "missing".
    soc: List[List[float]] = [[0.0]]                  # (nsoc, 1) cm-1

    fin_coord: List[List[float]] = []                 # (na, 3); never moved
    fin_velocity: List[List[float]] = []              # (na, 3); never moved


class Calculation(BaseModel):
    # calculation settings
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']

    modeldir: str = DEFAULT['modeldir']
    train_data: str = DEFAULT['train_data']
    train_dataset: dict = Field(default_factory=dict)
    pred_data: str = DEFAULT['pred_data']
    gl_seed: int = DEFAULT['gl_seed']
    train_mode: str = DEFAULT['train_mode']
    # Post-fit prediction spread on the training frames (training and
    # prediction only; the serving path never reads it).
    predict_std: int = DEFAULT['predict_std']
    mult_nn: int = DEFAULT['mult_nn']
    nn_predict_mode: int = DEFAULT['nn_predict_mode']
    # Device indices of the training subprocesses, one per ensemble
    # member. The ';' grouping is the member axis, as in the hyper roles.
    gpu: str = DEFAULT['gpu']
    # Role sections carry per-member hyperparameters. They stay raw
    # dicts here: schema/hyper.py owns their vocabulary, value shapes and
    # the member broadcasting rule.
    eg: dict = Field(default_factory=dict)
    nac: dict = Field(default_factory=dict)
    soc: dict = Field(default_factory=dict)

    target_states: List[int] = DEFAULT['target_states']           # (ntarget,)
    target_nac_pair: List[List[int]] = DEFAULT['target_nac_pair']  # (nnac, 2)
    target_soc_pair: List[List[int]] = DEFAULT['target_soc_pair']  # (nsoc, 2)

    comment: str = ''


class Environment(BaseModel):
    # computational environment settings
    skill_name: str = 'nn_pes_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # multitask/remotetask are dispatch-layer keywords: they are NOT
    # validated by the skill schema. The dispatch layer reads them from
    # the raw config dict when building the Dispatcher.
    nn_project: str = DEFAULT['nn_project']
    scratch_path: str = DEFAULT['scratch_path']
    # Threads per calculation (#306): `thread` is what this skill reads,
    # `nn_pes_thread` the spelling the framework forwards from the nn_pes
    # section; the larger wins. nn_pes is in-process TensorFlow, so the
    # value is exported into the environment the training children inherit
    # (service/trainer.py) rather than into a run script.
    thread: int = DEFAULT['thread']
    nn_pes_thread: int = DEFAULT['thread']
    # Empty/0 inherits `thread`; otherwise the value is this lifetime's own.
    train_thread: int = DEFAULT['train_thread']
    predict_thread: int = DEFAULT['predict_thread']

    keep_tmp: int = DEFAULT['keep_tmp']
    write_step_log: int = DEFAULT['write_step_log']
    slurm_time: str = DEFAULT['slurm_time']
    slurm_partition: str = DEFAULT['slurm_partition']
    slurm_ntasks: int = DEFAULT['slurm_ntasks']
    slurm_memory: int = DEFAULT['slurm_memory']
    slurm_command: str = DEFAULT['slurm_command']

    @model_validator(mode='after')
    def _resolve_resources(self):
        """Fold the two thread spellings into the name this skill reads."""
        def _larger(*values):
            numbers = []
            for value in values:
                try:
                    numbers.append(max(1, int(value or 1)))
                except (TypeError, ValueError):
                    continue
            return max(numbers or [1])

        self.thread = _larger(self.thread, self.nn_pes_thread)
        for name in ('train_thread', 'predict_thread'):
            try:
                value = int(getattr(self, name) or 0)
            except (TypeError, ValueError):
                value = 0
            setattr(self, name, value if value >= 1 else self.thread)
        return self

# Field-to-submodel mapping for flat access routing
# (aligned with framework FIELD_TO_SUBMODEL pattern).
#
# The name 'nac' means two things here: the couplings the backend reports
# (Properties) and the coupling role's hyperparameters (Calculation). A
# flat carrier can only mean the second, since results are never input, so
# Calculation is consulted first for a duplicated name.

FIELD_TO_SUBMODEL = {}
for _cls, _name in [(System, 'system'), (Calculation, 'calculation'),
                    (Properties, 'properties'), (Environment, 'environment')]:
    for _field in _cls.model_fields:
        FIELD_TO_SUBMODEL.setdefault(_field, _name)

SUBMODEL_CLASSES = {
    'system': System,
    'properties': Properties,
    'calculation': Calculation,
    'environment': Environment,
}


class SkillConfig(BaseModel):
    """nn_pes skill runtime configuration container.

    Supports both nested access (config.calculation.method) and flat
    access (config.method) via __getattr__, aligned with the framework
    PyRAIConfig pattern.
    """
    system: System = Field(default_factory=System)
    properties: Properties = Field(default_factory=Properties)
    calculation: Calculation = Field(default_factory=Calculation)
    environment: Environment = Field(default_factory=Environment)

    def __init__(self, **data):
        # Support flat kwargs auto-routing to submodels
        declared = set(type(self).model_fields.keys())
        submodel_kwargs = {}
        for k, v in data.items():
            if k in declared:
                submodel_kwargs[k] = v
            else:
                parent = FIELD_TO_SUBMODEL.get(k)
                if parent is not None:
                    if parent not in submodel_kwargs:
                        submodel_kwargs[parent] = {}
                    if isinstance(submodel_kwargs[parent], dict):
                        submodel_kwargs[parent][k] = v
        super().__init__(**submodel_kwargs)

    def __getattr__(self, name):
        # Flat field lookup across submodels. The FIELD_TO_SUBMODEL
        # lookup runs first so skill fields keep resolving; the 'model_'
        # guard only rejects pydantic internals.
        if name.startswith('__'):
            raise AttributeError(name)
        parent = FIELD_TO_SUBMODEL.get(name)
        if parent is not None:
            try:
                submodel = object.__getattribute__(self, parent)
                return getattr(submodel, name)
            except AttributeError:
                pass
        if name.startswith('model_'):
            raise AttributeError(name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# Backward compatibility alias
Config = SkillConfig
