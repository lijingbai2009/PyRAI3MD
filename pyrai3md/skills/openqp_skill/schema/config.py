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

This module provides the unified configuration schema for the OpenQP skill,
including system information, calculation parameters, and runtime settings.
It does not include numpy arrays in the schema, as they are not JSON serializable.

"""

from typing import Annotated, List, Optional
from pydantic import (BaseModel, BeforeValidator, Field,
                     model_validator)

from .reference import DEFAULT


def _normalize_shell(value):
    """Normalize shell commands to one command per line.

    Accepts a list of commands (yaml), a multi-line string, or a
    single-line string with ';' separators (classic input),
    matching the kernel Environment.get_shell normalization. The
    annotation survives the flat-keyword routing into the owning sub-model, unlike a
    class-level field validator.
    """
    fragments = value if isinstance(value, list) else [value]
    commands = []
    for fragment in fragments:
        if fragment is None:
            continue
        for piece in str(fragment).split(';'):
            piece = piece.strip()
            if piece:
                commands.append(piece)
    return '\n'.join(commands)


class System(BaseModel):
    """System information: atoms, coordinates, charges, electronic structure."""
    atoms: List[str] = []
    mass: List[float] = []
    coordinates: List[List[float]] = []
    target_coord: List[List[float]] = []
    point_charges: List[List[float]] = []

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
    # Spin-ladder sizes of a two-manifold (spin-orbit) run.  0 means unset,
    # which normalize derives from state_spin; a non-zero value must cover the
    # derived count of that ladder.  [tdhf] nstate stays the configured
    # ladder's count, so a two-manifold run never sizes from the total.
    nstate_s: int = 0
    nstate_t: int = 0
    current_state: int = 0
    state_label: List[str] = []
    state_spin: List[int] = []               # (nstate,) 2S per state (0=singlet, 2=triplet)

    npair: int = 0
    nnac: int = 0
    nsoc: int = 0

    transition_pair: List[List[int]] = [[]]
    nac_pair_map: List[List[int]] = [[]]
    soc_pair_map: List[List[int]] = [[]]

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
    pbc: List[int] = DEFAULT['pbc']
    cell: List[float] = DEFAULT['cell']
    lattice_vector: List[List[float]] = DEFAULT['lattice_vector']
    kpoint: List[int] = DEFAULT['kpoint']
    volume: float = 0


class Properties(BaseModel):
    """Computed properties for each electronic state."""
    energy: List[float] = [0.0]
    kinetic_energy: float = 0.0
    pot_energy: List[float] = [0.0]
    total_energy: List[float] = [0.0]

    fin_coord: List[List[float]] = []   # (na, 3) final coordinates after skill internal iteration (optimize/md/...), Angstrom; empty = no internal iteration

    fin_velocity: List[List[float]] = []   # (na, 3) final velocities after skill internal MD propagation, Bohr/au; empty = no velocity output

    gradient: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    # MISSING-PHASE1-STRESS (dev/methods/ENSEMBLE_DEVELOPMENT.md section 8):
    # declared for the NPT barostats but this skill's service layer does
    # not compute/write it yet. When added it must be the FULL
    # (nstate, 3, 3) stress tensor (not a scalar pressure) in
    # Hartree/Bohr^3, and only the whole-system provider task must supply
    # it (single center: theory[0]; multiscale / multi-center: theory[-1],
    # see resolve_system_task).
    stress_tensor: List[List[List[float]]] = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    hessian: List[List[List[float]]] = [[[0.0]]]
    frequencies: List[List[float]] = [[0.0]]
    norm_mode: List[List[List[List[float]]]] = [[[[0.0, 0.0, 0.0]]]]
    reduced_mass: List[List[float]] = [[0.0]]
    rotation_constants: List[List[float]] = [[0.0]]

    # Thermodynamic properties (from framework)
    heat_capacity_cv: List[float] = [0.0]
    heat_capacity_cp: List[float] = [0.0]
    zero_point_energy: List[float] = [0.0]
    internal_energy: List[float] = [0.0]
    enthalpy: List[float] = [0.0]
    entropy: List[float] = [0.0]
    free_energy: List[float] = [0.0]

    atom_dipole: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    dipole: List[List[float]] = [[0.0, 0.0, 0.0]]
    computed_charges: List[List[float]] = []            # (ns, na) atom charges computed by skill, extracted by framework to update background charges

    atom_trans_dipole: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    transition_dipole: List[List[float]] = [[0.0, 0.0, 0.0]]

    osc_strength: List[float] = [0.0]
    excitation: List[float] = [0.0]

    nac: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    soc: List[List[float]] = [[0.0]]

    # Orbital and density data (from framework)
    mo: str = ''
    mo1: str = ''
    mo2: str = ''
    esp: str = ''
    nto_molden: List[str] = ['']
    nto_cube: List[List[dict]] = [[{}]]
    electronic_density: str = ''

    # Spectroscopy (from framework)
    ir_width: float = DEFAULT['ir_width']
    ir_scale: float = DEFAULT['ir_scale']
    ir_intensity: List[float] = [0.0]
    ir_spectrum: str = ''
    ir_data: List[List[float]] = [[0, 0.0]]
    uv_width: float = DEFAULT['uv_width']
    uv_spectrum: str = ''
    uv_data: List[List[float]] = [[0, 0.0]]

    # NICS (from framework)
    nics_ring: List[int] = DEFAULT['nics_ring']
    nics_scan: List[float] = DEFAULT['nics_scan']
    nics: List[List[float]] = [[0.0]]
    nics_zz: List[List[float]] = [[0.0]]
    shielding_tensors: List[List[List[List[float]]]] = [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    shielding_constants: List[List[float]] = [[0.0]]


class Dynamics(BaseModel):
    """Dynamics settings and results."""
    state_population: List[float] = [1.0]
    temperature: float = DEFAULT['temperature']


class Calculation(BaseModel):
    """Calculation settings: method, profile, basis, functional, SCF/TD parameters."""
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']
    basis: str = DEFAULT['basis']
    functional: str = DEFAULT['functional']

    # TD-DFT type: rpa, tda (tddft) or mrsf (forced for the mrsf method)
    td_type: str = DEFAULT['td_type']

    # SCF settings (scf_type derived by normalize: rhf/rohf/uhf)
    scf_type: str = DEFAULT['scf_type']
    scf_maxit: int = DEFAULT['scf_maxit']
    scf_converge: float = DEFAULT['scf_converge']

    # TD settings
    td_maxit: int = DEFAULT['td_maxit']
    td_converge: float = DEFAULT['td_converge']

    # Guess and NAC settings
    guess_type: str = DEFAULT['guess_type']
    nac_type: str = DEFAULT['nac_type']

    # NAC vector settings (runtype=nac, single state pair only;
    # OpenQP does not support batch NAC vector calculations)
    nac_dt: float = DEFAULT['nac_dt']
    nac_dx: float = DEFAULT['nac_dx']
    nac_bp: bool = DEFAULT['nac_bp']

    # Target states
    target_states: List[int] = DEFAULT['target_states']
    target_nac_pair: List[List[int]] = [[]]
    target_soc_pair: List[List[int]] = [[]]

    # Spin-orbit coupling.  `dosoc` is the framework &md spelling and the
    # request side of SOC; Properties.soc stays result-only because the
    # merge loop below treats a falsy value as "unset", which a result
    # placeholder cannot distinguish from an empty request.
    dosoc: int = DEFAULT['dosoc']
    # OpenQP 1.3 takes one gradient list per spin ladder; a spin-orbit run
    # refuses the legacy combined `grad` selector.  Empty means the builder
    # derives them per step from target_states (the activestate channel).
    grad_s: List[int] = []
    grad_t: List[int] = []
    # Two-electron mean-field SOC term.  None means "not stated", which
    # normalize resolves to OpenQP's own default of 1; an explicit 0 must
    # survive, so the field cannot default to a falsy int.
    soc_2e: Optional[int] = None
    # Scalar-relativistic DKH order used by the reference (0 = OpenQP default).
    scal_rel: int = DEFAULT['scal_rel']
    # Reactive SCF convergence ladder ("soscf,trah"); empty keeps OpenQP's
    # own default chain.  Naming a stage replaces the chain rather than
    # extending it, so nothing is emitted unless the user asks.
    escalation: str = ''

    # Geometry optimization settings
    opt_maxit: int = DEFAULT['opt_maxit']
    opt_optimizer: str = DEFAULT['opt_optimizer']
    opt_rmsd_grad: float = DEFAULT['opt_rmsd_grad']
    opt_max_grad: float = DEFAULT['opt_max_grad']

    # Hessian settings. The displacement batch is bounded by the launcher,
    # not the deck: `openqp_nproc` sets the MPI ranks (each rank is
    # one displaced gradient), `thread` / `openqp_thread` sets the OpenMP
    # threads per rank. There is deliberately no hess_*_thread key.
    hess_dx: float = DEFAULT['hess_dx']

    # Reserved internal-iteration algorithm selectors (empty =
    # program default). Consumed by the skill once it exposes its
    # own optimizer options. Note: opt_optimizer above is OpenQP's
    # existing native optimizer selector and stays as-is.
    optimizer: str = ''      # internal geometry optimizer algorithm
    # SCF convergence algorithm, written to [scf] converger_type
    # (diis/soscf/trah).  OpenQP's own spelling is `converger_type`;
    # the adapter accepts it as an alias of this cross-skill field.
    scf_converger: str = ''

    verbose: int = DEFAULT['verbose']
    comment: str = ''


class Environment(BaseModel):
    """Computational environment settings for OpenQP."""
    skill_name: str = 'openqp_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # Skill-level shell commands (skills.openqp_skill.shell of
    # pmd_env.yaml / test input; module loads, ...), copied verbatim
    # into the local/slurm run scripts. The global environment.shell
    # is reserved for the main framework (#134).
    shell: Annotated[str, BeforeValidator(_normalize_shell)] = DEFAULT['shell']
    # multitask/remotetask are dispatch-layer keywords: they are NOT
    # validated by the skill schema. The dispatch layer reads them
    # from the raw config dict when building the Dispatcher. Backend
    # initialization ignores them (not in the schema).
    program: str = 'openqp'

    # OpenQP-specific runtime
    openqp_project: str = DEFAULT['openqp_project']
    openqp_path: str = DEFAULT['openqp_path']

    # MPI switch: default no-MPI mode (OpenMP only, --nompi).
    # use_mpi=True switches the run command to mpirun (no --nompi);
    # openqp_nproc controls the number of MPI ranks; mpi_command allows
    # a custom launcher prefix (e.g. 'mpirun -n 4'; the launcher every
    # skill standardizes on, #305).
    use_mpi: bool = DEFAULT['use_mpi']
    mpi_command: str = DEFAULT['mpi_command']

    # Resource management
    keep_tmp: int = DEFAULT['keep_tmp']
    # Compute resources (#306). `thread` is what a standalone input writes;
    # `openqp_thread` is the spelling the framework forwards from &openqp.
    # The larger of the pair wins, folded once below, so neither spelling
    # can shrink a request made in the other.
    # The rank count has no pair (#427): `openqp_nproc` is the only spelling
    # this skill reads, and the bare `nproc` is refused by name.
    thread: int = DEFAULT['thread']
    openqp_thread: int = DEFAULT['thread']
    openqp_nproc: int = DEFAULT['openqp_nproc']

    scratch_path: str = ''

    # Slurm settings
    slurm_time: str = DEFAULT['slurm_time']
    slurm_partition: str = DEFAULT['slurm_partition']
    slurm_ntasks: int = DEFAULT['slurm_ntasks']
    slurm_memory: int = DEFAULT['slurm_memory']
    slurm_command: str = DEFAULT['slurm_command']

    @model_validator(mode='after')
    def _resolve_resources(self):
        """Fold the thread pair into the name this skill reads."""
        def _larger(*values):
            numbers = []
            for value in values:
                try:
                    numbers.append(max(1, int(value or 1)))
                except (TypeError, ValueError):
                    continue
            return max(numbers or [1])

        self.thread = _larger(self.thread, self.openqp_thread)
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
    """OpenQP skill runtime configuration container.

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
