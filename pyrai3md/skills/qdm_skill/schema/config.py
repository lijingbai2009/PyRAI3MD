"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the unified configuration schema for the QDM skill,
including system information, calculation parameters, and runtime settings.
It does not include numpy arrays in the schema, as they are not JSON serializable.

"""

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

# Here we define some default values
DEFAULT = {
    'execution_mode': 'local',
    'project': 'QDM',
    'profile': 'tully1_energy',
    'method': 'tully1',
    'target_states': [0, 1],
    'target_nac_pair': [[]],
    # Analytic model settings. tully_a/b/c/d/e = None keeps the
    # standard Tully 1990 parameters of the selected model; lz_a/lz_b
    # = None keeps the Landau-Zener model class defaults.
    'model_mass': 2000.0,
    'tully_a': None,
    'tully_b': None,
    'tully_c': None,
    'tully_d': None,
    'tully_e': None,
    'lz_a': None,
    'lz_b': None,
    # Reference quantum dynamics (split-operator) settings.
    # ref_xmin/ref_xmax = None keeps the fixed default window
    # [-30, +80] Bohr defined by the reference service.
    'ref_ngrid': 2048,
    'ref_xmin': None,
    'ref_xmax': None,
    'ref_dt': 20.0,
    'ref_nsteps': 500,
    'ref_save_every': 1,
    'wavepacket_sigma': 0.5,
    'wavepacket_k0': 10.0,
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
    'qdm_project': 'QDM',
    'scratch_path': '',
    # Threads, in two spellings (#306): `thread` is what a standalone
    # input writes and what this skill reads, `qdm_thread` what the
    # framework forwards; the larger wins.
    'thread': 1,
    'qdm_thread': 1,
    'slurm_time': '01:00:00',
    'slurm_partition': 'default',
    'slurm_ntasks': 1,
    'slurm_command': '',
    'slurm_memory': 4000,
    'keep_tmp': 0,
}


class System(BaseModel):
    # system information
    atoms: List[str] = []                      # (na, 1)
    mass: List[float] = []                     # (na, 1)
    coordinates: List[List[float]] = []        # (na, 3)
    velocities: List[List[float]] = []         # (na, 3) Bohr/au
    point_charges: List[List[float]] = []      # (ns, na)

    charges: int = DEFAULT['charges']
    multiplicity: int = DEFAULT['multiplicity']
    natom: int = 0

    # State information (from framework)
    ci: List[int] = DEFAULT['ci']
    spin: List[int] = DEFAULT['spin']
    coupling: List[List[int]] = DEFAULT['coupling']
    nstate: int = 2
    current_state: int = 0      # Currently active electronic state (0-based)
    state_label: List[str] = []      # (ns,)
    state_spin: List[int] = []       # (ns, 1)

    npair: int = 0
    nnac: int = 0
    nsoc: int = 0

    transition_pair: List[List[int]] = [[]]    # (npair, 2)
    nac_pair_map: List[List[int]] = [[]]       # (nnac, 2)
    soc_pair_map: List[List[int]] = [[]]       # (nsoc, 2)

    pbc: List[bool] = DEFAULT['pbc']                          # (3,)
    cell: List[float] = DEFAULT['cell']                       # (6,)
    lattice_vector: List[List[float]] = DEFAULT['lattice_vector']  # (3, 3)
    kpoint: List[int] = DEFAULT['kpoint']                     # (3,)


class Properties(BaseModel):
    # computed properties reported by the QDM backend
    energy: List[float] = [0.0]                            # (ns,)

    fin_coord: List[List[float]] = []                      # (na, 3); QDM never returns a final structure
    fin_velocity: List[List[float]] = []                   # (na, 3); QDM never returns final velocities

    gradient: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]   # (ns, na, 3)
    nac: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]        # (nnac, na, 3)


class Calculation(BaseModel):
    # calculation settings
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']

    target_states: List[int] = DEFAULT['target_states']                # (ntarget,)
    target_nac_pair: List[List[int]] = DEFAULT['target_nac_pair']      # (ntarget_nac, 2)

    # Analytic model settings. tully_a/b/c/d/e = None keeps the
    # standard Tully 1990 parameters of the selected model; lz_a/lz_b
    # = None keeps the Landau-Zener model class defaults.
    # model_mass is the wavepacket mass ('mass' is reserved for the
    # framework atomic masses).
    model_mass: float = DEFAULT['model_mass']
    tully_a: Optional[float] = DEFAULT['tully_a']
    tully_b: Optional[float] = DEFAULT['tully_b']
    tully_c: Optional[float] = DEFAULT['tully_c']
    tully_d: Optional[float] = DEFAULT['tully_d']
    tully_e: Optional[float] = DEFAULT['tully_e']
    lz_a: Optional[float] = DEFAULT['lz_a']
    lz_b: Optional[float] = DEFAULT['lz_b']

    # Reference quantum dynamics (split-operator) settings (au)
    ref_ngrid: int = DEFAULT['ref_ngrid']
    ref_xmin: Optional[float] = DEFAULT['ref_xmin']
    ref_xmax: Optional[float] = DEFAULT['ref_xmax']
    ref_dt: float = DEFAULT['ref_dt']
    ref_nsteps: int = DEFAULT['ref_nsteps']
    ref_save_every: int = DEFAULT['ref_save_every']
    wavepacket_sigma: float = DEFAULT['wavepacket_sigma']
    wavepacket_k0: float = DEFAULT['wavepacket_k0']

    comment: str = ''


class Environment(BaseModel):
    # computational environment settings
    skill_name: str = 'qdm_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # multitask/remotetask are dispatch-layer keywords: they are NOT
    # validated by the skill schema. The dispatch layer reads them
    # from the raw config dict when building the Dispatcher. Backend
    # initialization ignores them (not in the schema).
    qdm_project: str = DEFAULT['qdm_project']
    scratch_path: str = DEFAULT['scratch_path']
    # Threads per calculation (#306): `thread` is what this skill reads and
    # what a standalone input writes, `qdm_thread` the spelling the framework
    # forwards from the qdm section; the larger of the pair wins.
    thread: int = DEFAULT['thread']
    qdm_thread: int = DEFAULT['thread']

    keep_tmp: int = DEFAULT['keep_tmp']
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

        self.thread = _larger(self.thread, self.qdm_thread)
        return self

# Field-to-submodel mapping for flat access routing
# (aligned with framework FIELD_TO_SUBMODEL pattern)

FIELD_TO_SUBMODEL = {}
for _cls, _name in [(System, 'system'), (Properties, 'properties'),
                    (Calculation, 'calculation'),
                    (Environment, 'environment')]:
    for _field in _cls.model_fields:
        FIELD_TO_SUBMODEL[_field] = _name

SUBMODEL_CLASSES = {
    'system': System,
    'properties': Properties,
    'calculation': Calculation,
    'environment': Environment,
}


class SkillConfig(BaseModel):
    """QDM skill runtime configuration container.

    Supports both nested access (config.calculation.method)
    and flat access (config.method) via __getattr__.
    Aligned with the framework PyRAIConfig pattern.
    """
    system: System = Field(default_factory=System)
    properties: Properties = Field(default_factory=Properties)
    calculation: Calculation = Field(default_factory=Calculation)
    environment: Environment = Field(default_factory=Environment)

    def __init__(self, **data):
        # Support flat kwargs auto-routing to submodels
        # (aligned with PyRAIConfig.__init__ pattern)
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
        # Flat field lookup across submodels
        # (aligned with PyRAIConfig.__getattr__ pattern). The
        # FIELD_TO_SUBMODEL lookup runs first so skill fields that
        # merely start with 'model_' (model_mass) still resolve;
        # the 'model_' guard only rejects pydantic internals.
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
