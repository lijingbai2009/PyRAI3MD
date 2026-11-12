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

This module provides the unified configuration schema for the Molcas skill,
including system information, calculation parameters, and runtime settings.
It does not include numpy arrays in the schema, as they are not JSON serializable.

"""

from typing import Annotated, List, Optional
from pydantic import (BaseModel, BeforeValidator, Field, field_validator,
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
    velocities: List[List[float]] = []       # (natom, 3) Cartesian velocities in Bohr/au
    target_coord: List[List[float]] = []
    point_charges: List[List[float]] = []
    # (natom, 4) [V, Fx, Fy, Fz] external potential/field at each QM
    # atom, atomic units; used by embedding='potential'.
    external_potential: List[List[float]] = []

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
    current_state: int = 0                   # Currently active electronic state (0-based)
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
    embedding: str = 'none'
    read_charge: str = 'none'
    boundary: List[List[int]] = []

    @field_validator('embedding', mode='before')
    @classmethod
    def _normalize_embedding(cls, value):
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('', 'none', 'false', '0'):
                return 'none'
            if text in ('true', '1'):
                return 'charge'
            if text == 'potential':
                raise ValueError(
                    "embedding='potential' is retired: write 'field', the "
                    "V/E block per QM atom. A charge file declares which "
                    "representation it holds, so this key is only consulted "
                    "when the data arrives as an array.")
            if text in ('charge', 'field'):
                return text
            raise ValueError(
                "embedding must be none, charge, or field; got %r" % (value,))
        return 'charge' if value else 'none'

    @field_validator('read_charge', mode='before')
    @classmethod
    def _normalize_read_charge(cls, value):
        if isinstance(value, str):
            text = value.strip()
            lowered = text.lower()
            if lowered in ('', 'none', 'false', '0'):
                return 'none'
            if lowered in ('true', '1'):
                return 'legacy'
            return text
        return 'legacy' if value else 'none'

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
    # MD propagation keywords forwarded to DYNAMIX/SURFACEHOP:
    # timestep in au, decoherence correction, thermostat selection.
    size: float = DEFAULT['size']
    deco: str = DEFAULT['deco']
    thermostat: str = DEFAULT['thermostat']
    # Initial MD state (0-based); becomes RASSCF MDRLXR (1-based).
    root: int = DEFAULT['root']


class Calculation(BaseModel):
    """Calculation settings: method, profile, active space, convergence."""
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']
    basis: str = DEFAULT['basis']
    # Task request from the framework (&CONTROL jobtype).  Used to
    # decide whether gradient ALASKA blocks are emitted (SOC workflow).
    jobtype: str = ''

    # Active space settings
    nact_elec: int = DEFAULT['nact_elec']
    nact_orb: int = DEFAULT['nact_orb']
    ras1_orb: int = DEFAULT['ras1_orb']
    ras2_orb: int = DEFAULT['ras2_orb']
    ras3_orb: int = DEFAULT['ras3_orb']
    inactive_orb: int = DEFAULT['inactive_orb']  # computed by normalize
    # Active orbital indices selected by the user (1-based at every
    # boundary, stored 0-based internally; converted by build_config
    # and turned into ALTEr pairs by the RASSCF builder).
    active: List[int] = DEFAULT['active']

    # CI settings
    ci_roots: List[int] = DEFAULT['ci_roots']
    ci_spin: List[int] = DEFAULT['ci_spin']

    # CASPT2 multistate switch (xmultistate all); default enabled.
    # Kept as the legacy alias for caspt2_type.
    xms: bool = DEFAULT['xms']
    # ss / ms / xms / rms / xdw. Empty resolves from xms.
    caspt2_type: str = DEFAULT['caspt2_type']
    dwms_zeta: float = DEFAULT['dwms_zeta']
    dwtype: int = DEFAULT['dwtype']
    nocsf: bool = DEFAULT['nocsf']
    caspt2_ipea_shift: Optional[float] = DEFAULT['caspt2_ipea_shift']
    caspt2_real_shift: float = DEFAULT['caspt2_real_shift']
    caspt2_imaginary_shift: float = DEFAULT['caspt2_imaginary_shift']

    # SOC dual RASSCF independent parameters (singlet / triplet separated)
    singlet_ci_roots: List[int] = DEFAULT['singlet_ci_roots']
    singlet_ci_spin: List[int] = DEFAULT['singlet_ci_spin']
    triplet_ci_roots: List[int] = DEFAULT['triplet_ci_roots']
    triplet_ci_spin: List[int] = DEFAULT['triplet_ci_spin']

    # Target states
    target_states: List[int] = [0]
    target_nac_pair: List[List[int]] = [[]]
    target_soc_pair: List[List[int]] = [[]]

    # Convergence
    scf_converge: float = DEFAULT['scf_converge']
    opt_converge: bool = DEFAULT['opt_converge']
    opt_maxit: int = DEFAULT['opt_maxit']
    # RASSCF ITERATIONS first number (max CI iterations)
    maxit_scf: int = DEFAULT['maxit_scf']

    # Optimization target states (0-based): equal values give a
    # single-state relaxation; a same-spin pair triggers MECI and a
    # cross-spin pair triggers MECP (i = singlet, j = triplet).
    opt_state_i: int = DEFAULT['opt_state_i']
    opt_state_j: Optional[int] = DEFAULT['opt_state_j']

    # Molecular dynamics: number of MD steps (FOREACH loop bound)
    # and the MOLCAS_MAXITER export limit (0 = auto, maxit_md x 10).
    maxit_md: int = DEFAULT['maxit_md']
    md_maxiter: int = DEFAULT['md_maxiter']

    # Orbital management
    guess_orbital: str = ''  # path to StrOrb or JobIph file

    # Reserved internal-iteration algorithm selectors (empty =
    # program default). Consumed by the skill once it exposes its
    # own optimizer options.
    optimizer: str = ''      # internal geometry optimizer algorithm
    scf_converger: str = ''  # SCF convergence optimizer (diis/soscf/...)

    verbose: int = DEFAULT['verbose']
    comment: str = ''


class Environment(BaseModel):
    """Computational environment settings for OpenMolcas."""
    skill_name: str = 'molcas_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # Skill-level shell commands (skills.molcas_skill.shell of
    # pmd_env.yaml / test input; module loads, ...), copied verbatim
    # into the local/slurm run scripts. The global environment.shell
    # is reserved for the main framework (#134).
    shell: Annotated[str, BeforeValidator(_normalize_shell)] = DEFAULT['shell']
    # multitask/remotetask are dispatch-layer keywords: they are NOT
    # validated by the skill schema. The dispatch layer reads them
    # from the raw config dict when building the Dispatcher. Backend
    # initialization ignores them (not in the schema).
    program: str = 'molcas'

    # Molcas-specific runtime
    molcas_project: str = DEFAULT['molcas_project']
    molcas_mem: int = DEFAULT['molcas_mem']
    molcas_nproc: int = DEFAULT['molcas_nproc']
    molcas_print: int = DEFAULT['molcas_print']
    molcas_path: str = DEFAULT['molcas_path']

    # Resource management
    keep_tmp: int = DEFAULT['keep_tmp']
    # Compute resources (#306). `thread` and `nproc` are what this skill
    # reads and what a standalone input writes; `molcas_thread` and
    # `molcas_nproc` are the spellings the framework forwards from &molcas. The
    # larger of each pair wins, folded once below, so neither spelling can
    # shrink a request made in the other.
    thread: int = DEFAULT['thread']
    molcas_thread: int = DEFAULT['thread']
    nproc: int = DEFAULT['nproc']

    scratch_path: str = ''

    # Slurm settings
    slurm_time: str = DEFAULT['slurm_time']
    slurm_partition: str = DEFAULT['slurm_partition']
    slurm_ntasks: int = DEFAULT['slurm_ntasks']
    slurm_memory: int = DEFAULT['slurm_memory']
    slurm_command: str = DEFAULT['slurm_command']

    @model_validator(mode='after')
    def _resolve_resources(self):
        """Fold each resource pair into the name this skill reads."""
        def _larger(*values):
            numbers = []
            for value in values:
                try:
                    numbers.append(max(1, int(value or 1)))
                except (TypeError, ValueError):
                    continue
            return max(numbers or [1])

        self.thread = _larger(self.thread, self.molcas_thread)
        self.nproc = _larger(self.nproc, self.molcas_nproc)
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
    """Molcas skill runtime configuration container.

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
