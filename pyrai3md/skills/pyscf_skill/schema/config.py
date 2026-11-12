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

This module provides the unified configuration schema for the PySCF skill,
including system information, calculation parameters, and runtime settings.
It does not include numpy arrays in the schema, as they are not JSON serializable.

"""

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

# Here we define some default values
DEFAULT = {
    'execution_mode': 'local',
    'project': 'PYSCF',
    'profile': 'dft_energy',
    'method': 'dft',
    'functional': 'pbe0',
    'basis': 'ccpvdz',
    'chk_file': '',
    'save_mo': False,
    'save_norm_mode': False,
    'maxit_scf': 100,
    'maxit_opt': 100,
    'maxit_md': 1,
    'density_fitting': True,
    'non_local_correction': True,
    'dispersion_correction': '',
    'scf_converge': 1e-9,
    'td_converge': 1e-6,
    'opt_converge': False,
    'target_states': [0],
    'target_nac_pair': [[]],
    'target_soc_pair': [[]],
    'opt_energy': 1e-6,
    'opt_grms': 3e-4,
    'opt_gmax': 4.5e-4,
    'opt_drms': 1.2e-3,
    'opt_dmax': 1.8e-3,
    'opt_sp': True,
    'opt_freq': False,
    'ir_width': 100.0,
    'ir_scale': 1.0,
    'uv_width': 100.0,
    'nics_ring': [0],
    'nics_scan': [0.0],
    'grid_level': 3,
    'temperature': 298.15,
    'pressure': 1.0,
    'thermostat': 'nose-hoover',
    'barostat': 'berendsen',
    'ensemble': 'nve',
    'verbose': 4,
    'solvent': 'gas',
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
    # External potential (from framework)
    'shape': 'ellipsoid',
    'factor': 40,
    'scale': 1.0,
    'pyscf_path': 'import',
    'pyscf_project': 'PYSCF',
    'scratch_path': '',
    # Threads, in two spellings (#306): `thread` is what a standalone
    # input writes and what this skill reads, `pyscf_thread` what the
    # framework forwards; the larger wins.
    'thread': 1,
    'pyscf_thread': 1,
    'slurm_time': '01:00:00',
    # Empty means "no --partition directive": the cluster's default
    # partition answers. A named partition is a site choice, so it comes
    # from the input or the deployment's skill section, never from the
    # shipped default (a site rarely names its queue 'default').
    'slurm_partition': '',
    'slurm_ntasks': 1,
    'slurm_command': '',
    'slurm_memory': 4000,
    'keep_tmp': 0,
}


class System(BaseModel):
    # system information
    atoms: List[str] = []         # (na, 1)
    mass: List[float] = []             # (na, 1)
    coordinates: List[List[float]] = []      # (na, 3)
    target_coord: List[List[float]] = []     # (na, 3)
    point_charges: List[List[float]] = []      # (ns, na)

    charges: int = DEFAULT['charges']
    total_electron: int = 0
    nalpha: int = 0
    nbeta: int = 0
    multiplicity: int = DEFAULT['multiplicity']
    natom: int = 0

    # State information (from framework)
    ci: List[int] = DEFAULT['ci']
    spin: List[int] = DEFAULT['spin']
    coupling: List[List[int]] = DEFAULT['coupling']
    nstate: int = 1
    current_state: int = 0      # Currently active electronic state (0-based)
    state_label: List[str] = []      # (ns,)
    state_spin: List[int] = []      # (ns, 1)

    npair: int = 0
    nnac: int = 0
    nsoc: int = 0

    transition_pair: List[List[int]] = [[]]    # (npair, 2)
    nac_pair_map: List[List[int]] = [[]]    # (nnac, 2)
    soc_pair_map: List[List[int]] = [[]]    # (nsoc, 2)

    # QM/MM settings (from framework)
    qmmm_key: Optional[str] = None
    qmmm_xyz: str = 'Input'
    highlevel: List[int] = []
    midlevel: List[int] = []
    embedding: bool = False
    read_charge: bool = False
    boundary: List[List[int]] = []

    # The framework used to carry a restraint as nineteen flat keys
    # (`freeze`, `cbond`, `fbond`, `cavity`, `shape`, ...), and each skill
    # schema mirrored them so the merge would not drop a field. Those keys are
    # gone from `System` and are refused when an input writes one
    # (`schema.config.RETIRE_CONSTRAINT_KEYS`); a declaration now arrives as the
    # single nested `system.constraints` block, which no skill reads and no
    # skill mirror needs (#272).

    # Geometry tracking (from framework)
    track_type: Optional[str] = None
    track_index: List[List[int]] = []
    track_thrhd: List[float] = []
    track_stop: int = 0

    solvent: str = DEFAULT['solvent']

    pbc: List[bool] = DEFAULT['pbc'] # (3,)
    cell: List[float] = DEFAULT['cell'] # (6,)
    lattice_vector: List[List[float]] = DEFAULT['lattice_vector'] # (3, 3)
    kpoint: List[int] = DEFAULT['kpoint'] # (3,)

    volume: float = 0


class Properties(BaseModel):
    # computed properties for each state
    energy: List[float] = [0.0]         # (ns,)
    kinetic_energy: float = 0
    pot_energy: List[float] = [0.0]      # (ns,)
    total_energy: List[float] = [0.0]    # (ns,)

    fin_coord: List[List[float]] = []    # (na, 3) final coordinates after skill internal iteration (optimize/md/...), Angstrom; empty = no internal iteration

    fin_velocity: List[List[float]] = []    # (na, 3) final velocities after skill internal MD propagation, Bohr/au; empty = no velocity output

    gradient: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]        # (ns, na, 3)
    # MISSING-PHASE1-STRESS (dev/methods/ENSEMBLE_DEVELOPMENT.md section 8):
    # declared for the NPT barostats but this skill's service layer does
    # not compute/write it yet. When added it must be the FULL
    # (nstate, 3, 3) stress tensor (not a scalar pressure) in
    # Hartree/Bohr^3, and only the whole-system provider task must supply
    # it (single center: theory[0]; multiscale / multi-center: theory[-1],
    # see resolve_system_task).
    stress_tensor: List[List[List[float]]] = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]   # (ns, 3, 3)
    hessian: List[List[List[float]]] = [[[0.0]]]         # (ns, 3na, 3na)
    frequencies: List[List[float]] = [[0.0]]        # (ns, 3na)
    norm_mode: List[List[List[List[float]]]] = [[[[0.0, 0.0, 0.0]]]] # (ns, 3na, na, 3)
    reduced_mass: List[List[float]] = [[0.0]]        # (ns, 3na)
    rotation_constants: List[List[float]] = [[0.0]]  # (ns, 3)
    heat_capacity_cv: List[float] = [0.0] # (ns, 1)
    heat_capacity_cp: List[float] = [0.0] # (ns, 1)
    zero_point_energy: List[float] = [0.0]  # (ns, 1)
    internal_energy: List[float] = [0.0]    # (ns, 1)
    enthalpy: List[float] = [0.0]  # (ns, 1)
    entropy: List[float] = [0.0]   # (ns, 1)
    free_energy: List[float] = [0.0]  # (ns, 1)

    atom_dipole: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]          # (ns, na, 3)
    dipole: List[List[float]] = [[0.0, 0.0, 0.0]]      # (ns, 3)
    computed_charges: List[List[float]] = []            # (ns, na) atom charges computed by skill, extracted by framework to update background charges

    atom_trans_dipole: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]    # (npair, na, 3)
    transition_dipole: List[List[float]] = [[0.0, 0.0, 0.0]]   # (npair, 3)

    osc_strength: List[float] = [0.0]    # (npair, 1)
    excitation: List[float] = [0.0]

    nac: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]  # (nnac, na, 3)
    soc: List[List[float]] = [[0.0]]             # (nsoc, 1)

    mo: str = ''
    mo1: str = ''
    mo2: str = ''
    esp: str = ''
    nto_molden: List[str] = ['']  # (nstate, 1)
    nto_cube: List[List[dict]]  = [[{}]] # (nstate, ncube)
    electronic_density: str = ''
    ir_width: float = DEFAULT['ir_width']
    ir_scale: float = DEFAULT['ir_scale']
    ir_intensity: List[float] = [0.0]    # (3na-6,)
    ir_spectrum: str = ''
    ir_data: List[List[float]] = [[0, 0.0]]    # (npoints, 2)
    uv_width: float = DEFAULT['uv_width']
    uv_spectrum: str = ''
    uv_data: List[List[float]] = [[0, 0.0]]    # (npoints, 2)
    nics_ring: List[int] = DEFAULT['nics_ring']
    nics_scan: List[float] = DEFAULT['nics_scan']
    nics: List[List[float]] = [[0.0]]  # ns, npoints
    nics_zz: List[List[float]] = [[0, 0]] # ns, npoints
    shielding_tensors: List[List[List[List[float]]]] = [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]] # (ns, na, 3, 3)
    shielding_constants: List[List[float]] = [[0.0]] # (ns, na + nnics)


class Dynamics(BaseModel):
    # dynamics settings and results
    state_population: List[float] = [1.0]     # (ns, 1)

    temperature: float = DEFAULT['temperature']  # in K
    pressure: float = DEFAULT['pressure'] # in bar

    thermostat: str = DEFAULT['thermostat']
    barostat: str = DEFAULT['barostat']
    ensemble: str = DEFAULT['ensemble']


class Calculation(BaseModel):
    # calculation settings
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']
    functional: str = DEFAULT['functional']
    basis: str = DEFAULT['basis']
    chk_file: str = DEFAULT['chk_file']
    save_mo: bool = DEFAULT['save_mo']
    save_norm_mode: bool = DEFAULT['save_norm_mode']

    target_states: List[int] = DEFAULT['target_states']     # (ntarget,)
    target_nac_pair: List[List[int]] = DEFAULT['target_nac_pair']   # (ntarget_nac, 2)
    target_soc_pair: List[List[int]] = DEFAULT['target_soc_pair']   # (ntarget_soc, 2)
    maxit_scf: int = DEFAULT['maxit_scf']
    maxit_opt: int = DEFAULT['maxit_opt']
    maxit_md: int = DEFAULT['maxit_md']

    density_fitting: bool = DEFAULT['density_fitting']
    non_local_correction: bool = DEFAULT['non_local_correction']
    dispersion_correction: str = DEFAULT['dispersion_correction']
    grid_level: int = DEFAULT['grid_level']
    scf_converge: float = DEFAULT['scf_converge']
    td_converge: float =  DEFAULT['td_converge']
    opt_converge: bool =  DEFAULT['opt_converge']
    opt_energy: float =  DEFAULT['opt_energy']
    opt_grms: float =  DEFAULT['opt_grms']
    opt_gmax: float =  DEFAULT['opt_gmax']
    opt_drms: float =  DEFAULT['opt_drms']
    opt_dmax: float =  DEFAULT['opt_dmax']
    opt_sp: bool =  DEFAULT['opt_sp']
    opt_freq: bool =  DEFAULT['opt_freq']

    # Reserved internal-iteration algorithm selectors (empty =
    # program default). Consumed by the skill once it exposes its
    # own optimizer options.
    optimizer: str = ''      # internal geometry optimizer algorithm
    scf_converger: str = ''  # SCF convergence optimizer (diis/soscf/...)

    verbose: int = DEFAULT['verbose']
    comment: str = ''


class Environment(BaseModel):
    # computational environment settings
    skill_name: str = 'pyscf_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # multitask/remotetask are dispatch-layer keywords: they are NOT
    # validated by the skill schema. The dispatch layer reads them
    # from the raw config dict when building the Dispatcher. Backend
    # initialization ignores them (not in the schema).
    pyscf_path: str = DEFAULT['pyscf_path']
    pyscf_project: str = DEFAULT['pyscf_project']
    scratch_path: str = DEFAULT['scratch_path']
    # Threads per calculation (#306): `thread` is what this skill reads and
    # what a standalone input writes, `pyscf_thread` the spelling the framework
    # forwards from the pyscf section; the larger of the pair wins.
    thread: int = DEFAULT['thread']
    pyscf_thread: int = DEFAULT['thread']

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

        self.thread = _larger(self.thread, self.pyscf_thread)
        return self

# Field-to-submodel mapping for flat access routing
# (aligned with framework FIELD_TO_SUBMODEL pattern)

FIELD_TO_SUBMODEL = {}
for _cls, _name in [(System, 'system'), (Properties, 'properties'),
                    (Dynamics, 'dynamics'), (Calculation, 'calculation'),
                    (Environment, 'environment')]:
    for _field in _cls.model_fields:
        FIELD_TO_SUBMODEL[_field] = _name

SUBMODEL_CLASSES = {
    'system': System,
    'properties': Properties,
    'dynamics': Dynamics,
    'calculation': Calculation,
    'environment': Environment,
}


class SkillConfig(BaseModel):
    """PySCF skill runtime configuration container.

    Supports both nested access (config.calculation.basis)
    and flat access (config.basis) via __getattr__.
    Aligned with the framework PyRAIConfig pattern.
    """
    system: System = Field(default_factory=System)
    properties: Properties = Field(default_factory=Properties)
    dynamics: Dynamics = Field(default_factory=Dynamics)
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
        # (aligned with PyRAIConfig.__getattr__ pattern)
        if name.startswith('model_') or name.startswith('__'):
            raise AttributeError(name)
        parent = FIELD_TO_SUBMODEL.get(name)
        if parent is not None:
            try:
                submodel = object.__getattribute__(self, parent)
                return getattr(submodel, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# Backward compatibility alias
Config = SkillConfig
