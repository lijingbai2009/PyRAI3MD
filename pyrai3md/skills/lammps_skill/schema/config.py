"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the unified configuration schema for the LAMMPS skill,
including system information, calculation parameters, and runtime settings.
It does not include numpy arrays in the schema, as they are not JSON serializable.

"""

from typing import Annotated, List, Optional, Union
from pydantic import (BaseModel, BeforeValidator, Field,
                     model_validator)

from .reference import DEFAULT


def _normalize_int_list(value):
    """Accept a whitespace/comma separated string or a list of ints.

    The classic-input carrier writes a flat int list as one
    space-separated token string (`qm_atoms 0 1 2`), and the framework
    hands that string through unchanged; a standalone YAML or the
    framework task config hands a real list. Both must reach the model
    as a list, otherwise pydantic refuses the string for `List[int]`.
    """
    if isinstance(value, str):
        text = value.replace(',', ' ')
        return [int(token) for token in text.split() if token]
    return value


def _normalize_float_list(value):
    """Accept a whitespace/comma separated string or a list of floats."""
    if isinstance(value, str):
        text = value.replace(',', ' ')
        return [float(token) for token in text.split() if token]
    return value


def _normalize_accuracy(value):
    """Fold the reciprocal accuracy down to the token the deck carries.

    YAML reads `0.0001` as a float and `1e-8` as a string, so this one key
    arrives under two Python types depending on how the case spells the
    number its template already uses. They are the same request, and LAMMPS
    wants digits rather than a float: without this, a float dies in model
    construction with a `string_type` error that names neither the key's
    meaning nor the file it came from, while the same value written `1e-8`
    runs. Validation of what the token may say stays in schema/validate.py.
    """
    return str(value).strip().lower()


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
    velocities: List[List[float]] = []   # (natom, 3) Cartesian velocities, Bohr/au from the framework; written into the data file for MD profiles
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
    current_state: int = 0
    state_label: List[str] = []
    state_spin: List[int] = []               # (nstate,) 2S per state (0=singlet, 2=triplet)

    npair: int = 0
    nnac: int = 0
    nsoc: int = 0

    transition_pair: List[List[int]] = [[]]
    nac_pair_map: List[List[int]] = [[]]
    soc_pair_map: List[List[int]] = [[]]

    # Legacy ONIOM mirrors (from framework). Inert in this skill: they are
    # declared so the framework-to-skill merge routes them instead of
    # dropping them, and nothing here reads them. The QM/MM region this
    # skill implements is `qmmm_field`/`qmmm_energy` + `qm_atoms`, with its own 0-based
    # contract; reusing `embedding` or `highlevel` would put the ONIOM
    # shell semantics and its 1-based -> 0-based migration in the path.
    qmmm_key: Optional[str] = None
    qmmm_xyz: str = 'Input'
    highlevel: List[int] = []
    midlevel: List[int] = []
    # Inert ONIOM mirrors. The framework's `System.embedding` /
    # `read_charge` are normalized to strings ('none'/'charge'/
    # 'potential'), and its config merge forwards them here; the
    # union keeps that value legal even though nothing in this skill
    # reads it. A bool is still accepted for the standalone spelling.
    embedding: Union[bool, str] = False
    read_charge: Union[bool, str] = False
    boundary: List[List[int]] = []

    # QM/MM task inputs (service/builder/qmmm.py, skill_guide.md).
    # `qm_atoms` names the QM subsystem as 0-based framework atom indices
    # (the atom order, not LAMMPS ids). The task is selected by profile
    # (`qmmm_field` / `qmmm_energy`); the retired `qmmm` switch is refused
    # by schema/adapter.py. Initialization-time only: the partition is a
    # topology property, so a different region needs a new backend.
    qm_atoms: Annotated[List[int], BeforeValidator(_normalize_int_list)] = \
        DEFAULT['qm_atoms']
    # qmmm_field only: the center this probe measures. Empty means the whole
    # QM set (single center). It is a separate keyword from qm_atoms because
    # the two questions differ: which atoms the classical surface carries
    # charges for, and which of them this deck is sampling the potential at.
    qm_probe: Annotated[List[int], BeforeValidator(_normalize_int_list)] = \
        DEFAULT['qm_probe']
    # qmmm_energy only: the QM set tiled into centers. The framework writes
    # it from the partition plan, so a QM atom never carries two centers'
    # 0-order charges.
    qm_centers: List[List[int]] = DEFAULT['qm_centers']
    # qmmm_energy inputs. `q_qm` is the QM 0-order charge set in qm_atoms
    # order; `qmmm_field_file` is an optional precomputed four-column
    # ESPF V/E file. Runtime callers may pass the same data through
    # calculate(qmmm_inputs=...) instead; these static fields keep cold
    # standalone runs and test yaml files self-contained.
    q_qm: Annotated[List[float], BeforeValidator(_normalize_float_list)] = \
        DEFAULT['q_qm']
    qmmm_field_file: str = DEFAULT['qmmm_field_file']
    # atom_selection: 1-based row numbers of the topology template this task
    # computes, in the order the framework hands over the sliced geometry (it
    # canonicalises a center to ascending at input time, so this arrives
    # ascending; the skill renumbers by position and never re-sorts).
    # Empty means the whole system, which is every existing path. A non-empty
    # list cuts the user's `{project}.data` down to those rows at backend
    # initialization (dev/skills/LAMMPS_SKILL_DEVELOPMENT.md), so a
    # multi-center run's low-layer correction term `L(c)` evaluates the center
    # with the force field the user declared for it rather than failing on a
    # template it cannot satisfy.
    #
    # It is NOT `highlevel`. `highlevel`/`midlevel` are the inner QM/MM
    # partition of a task and the framework clears them for a region task
    # (`planning/assembly.py`), because the sliced subsystem is then the whole system
    # for that task. This key says which rows of the topology file survive;
    # it is topology identity, not a run parameter, so it lives beside
    # `atoms`.
    atom_selection: List[int] = []

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
    # image: (natom, 3) count of cell lengths each atom crossed, handed
    # over with the coordinates by the framework's evaluation snapshot.
    # coordinates stay the wrapped (main image) values the framework
    # stores, and LAMMPS keys its own unwrapped positions off this count:
    # handing it wrapped coordinates while its live instance still holds
    # the previous step's counts is what makes the two disagree about
    # where an atom is. Runtime state, never an input keyword; an empty
    # list reads as "no crossings".
    image: List[List[int]] = []


class Properties(BaseModel):
    """Computed properties for each electronic state."""
    energy: List[float] = [0.0]
    kinetic_energy: float = 0.0
    pot_energy: List[float] = [0.0]
    total_energy: List[float] = [0.0]

    fin_coord: List[List[float]] = []   # (na, 3) final coordinates after skill internal iteration (optimize/md/...), Angstrom; empty = no internal iteration

    fin_velocity: List[List[float]] = []   # (na, 3) final velocities after skill internal MD propagation, Bohr/au; empty = no velocity output

    gradient: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    # Whole-system stress tensor (nstate, 3, 3) in bar (units.stress).
    # LAMMPS is a ground-state method, so only the S0 block is filled
    # from the virial pressure compute; the rest stays zero.
    stress_tensor: List[List[List[float]]] = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    hessian: List[List[List[float]]] = [[[0.0]]]

    dipole: List[List[float]] = [[0.0, 0.0, 0.0]]
    computed_charges: List[List[float]] = []            # (ns, na) atom charges computed by skill, extracted by framework to update background charges
    transition_dipole: List[List[float]] = [[0.0, 0.0, 0.0]]

    osc_strength: List[float] = [0.0]
    excitation: List[float] = [0.0]

    nac: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    soc: List[List[float]] = [[0.0]]

    # Atomic charges from the force field topology (natom,)
    atom_charges: List[float] = []


class Dynamics(BaseModel):
    """Dynamics settings and results."""
    state_population: List[float] = [1.0]
    temperature: float = DEFAULT['temperature']

    # Standalone MD parameters for the ff_nve/ff_nvt/ff_npt profiles
    # (aligned with the reference nve/nvt/npt inputs)
    md_nstep: int = DEFAULT['md_nstep']              # number of MD steps (run N)
    md_timestep: float = DEFAULT['md_timestep']      # timestep in fs (real) or ps (metal)
    md_seed: int = DEFAULT['md_seed']                # RNG seed for velocity all create
    dump_freq: int = DEFAULT['dump_freq']            # trajectory dump / thermo frequency
    thermo_damp: float = DEFAULT['thermo_damp']      # thermostat damping parameter
    pressure: float = DEFAULT['pressure']            # npt target pressure (atm)
    pressure_damp: float = DEFAULT['pressure_damp']  # barostat damping parameter
    # Cell coupling of the ff_npt fix: 'isotropic' (one scalar volume) or
    # 'anisotropic' (three independent axes). See reference.py for why it
    # is declared: it mirrors the framework's md.barostat_mode, so the two
    # drivers of one input deform the cell the same way.
    barostat_mode: str = DEFAULT['barostat_mode']


class Calculation(BaseModel):
    """Calculation settings: method and profile."""
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']

    # Target states (LAMMPS is a ground-state method, always [0])
    target_states: List[int] = [0]

    # Whole-system stress tensor request. The framework planning route
    # (md+ensemble=npt or optimize+optimize_cell) injects stress=True;
    # a standalone test yaml sets it directly. The single-point section
    # then adds a virial-only 'compute pressure' whose six components
    # the parser reads non-blocking (zero-filled when absent).
    stress: bool = False

    # Reserved internal-iteration algorithm selectors (empty =
    # program default). Consumed by the skill once it exposes its
    # own optimizer options.
    optimizer: str = ''      # internal geometry optimizer algorithm
    scf_converger: str = ''  # SCF convergence optimizer (diis/soscf/...)

    # Geometry optimization parameters for the ff_optimize profile
    # (minimize command, aligned with the reference 'min_style cg' +
    # 'minimize 0.0001 1.0e-8 1000 1000000' input)
    # Engine-side box relaxation: which lattice degrees of freedom LAMMPS may
    # move during `minimize`, and the pressure it moves them against (bar).
    box_relax: str = DEFAULT['box_relax']
    box_relax_pressure: float = DEFAULT['box_relax_pressure']
    min_style: str = DEFAULT['min_style']        # minimization algorithm (cg/sd/quickmin/fire/hftn)
    min_etol: float = DEFAULT['min_etol']        # energy tolerance (kcal/mol or eV)
    min_ftol: float = DEFAULT['min_ftol']        # force tolerance (kcal/mol/A or eV/A)
    min_maxiter: int = DEFAULT['min_maxiter']    # max minimization iterations
    min_maxeval: int = DEFAULT['min_maxeval']    # max force/energy evaluations

    verbose: int = DEFAULT['verbose']
    comment: str = ''


class Environment(BaseModel):
    """Computational environment settings for LAMMPS."""
    skill_name: str = 'lammps_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # Skill-level shell commands (skills.lammps_skill.shell of
    # pmd_env.yaml / test input; module loads, ...), copied verbatim
    # into the local/slurm run scripts. The global environment.shell
    # is reserved for the main framework (#134).
    shell: Annotated[str, BeforeValidator(_normalize_shell)] = DEFAULT['shell']
    # multitask/remotetask are dispatch-layer keywords: they are NOT
    # validated by the skill schema. The dispatch layer reads them
    # from the raw config dict when building the Dispatcher. Backend
    # initialization ignores them (not in the schema).
    program: str = 'lammps'

    # LAMMPS-specific runtime
    lammps_project: str = DEFAULT['lammps_project']
    lammps_path: str = DEFAULT['lammps_path']
    # LAMMPS units style of the user-provided input template:
    # 'real' (kcal/mol, Angstrom) or 'metal' (eV, Angstrom).
    lammps_units: str = DEFAULT['lammps_units']
    # The accuracy token this skill puts on the `kspace_style` line it
    # re-issues after read_data, keeping the style name the template chose.
    # See schema/reference.py DEFAULT for why a loose value is not merely
    # slower: the periodic field probe's V comes from a difference of two
    # reciprocally unconverged runs.
    ewald_accuracy: Annotated[str, BeforeValidator(
        _normalize_accuracy)] = DEFAULT['ewald_accuracy']

    # Resident execution switch (see schema/reference.py DEFAULT).
    # python_enabled routes the single-point profiles to a LAMMPS
    # instance held by a service process; validate_persistent decides
    # whether the request is honoured and where.
    python_enabled: bool = DEFAULT['python_enabled']
    python_path: str = DEFAULT['python_path']
    # debug decides whether a resident step also writes the files the
    # subprocess path writes. Inert on the subprocess path, where
    # keep_tmp already owns what survives a run.
    debug: bool = DEFAULT['debug']
    # 'main' gives the framework integrator ownership of the geometry, and then
    # the constraint keywords below are inert whatever the profile says;
    # anything else hands the jobtype to this skill, whose own constraint
    # keywords are then the only ones in force.
    driver: str = DEFAULT['driver']

    # Shake constraint switch (&lammps keywords).
    # This group owns the constraint only where LAMMPS integrates the
    # atoms (ENGINE_INTEGRATED_PROFILES); when the framework propagates,
    # the constraint belongs to its framework-side constraint block
    # block instead, and validate_shake refuses the combination rather
    # than letting it silently drop.
    # lammps_shake=0 (default, legacy behavior): the shake fix
    # defined in the input template is disabled via unfix.
    # lammps_shake=1: unfix lines are stripped so the template shake
    # fix (e.g. TIP3P water) stays active.
    lammps_shake: int = DEFAULT['lammps_shake']
    shake_fix_id: str = DEFAULT['shake_fix_id']
    shake_keywords: str = DEFAULT['shake_keywords']

    # Resource management
    keep_tmp: int = DEFAULT['keep_tmp']
    # Compute resources (#306). `thread` and `nproc` are what this skill
    # reads and what a standalone LAMMPS input writes;
    # `lammps_thread`/`lammps_nproc` are the spellings the framework
    # forwards from &lammps. The larger of each pair wins, resolved once
    # in _resolve_resources below, so neither spelling silently overrides
    # the other and a wrong value can only ever be the larger one.
    thread: int = DEFAULT['thread']
    lammps_thread: int = DEFAULT['thread']
    nproc: int = DEFAULT['nproc']
    lammps_nproc: int = DEFAULT['lammps_nproc']
    scratch_path: str = ''

    # Slurm settings
    slurm_time: str = DEFAULT['slurm_time']
    slurm_partition: str = DEFAULT['slurm_partition']
    slurm_ntasks: int = DEFAULT['slurm_ntasks']
    slurm_memory: int = DEFAULT['slurm_memory']
    slurm_command: str = DEFAULT['slurm_command']

    @model_validator(mode='after')
    def _resolve_resources(self):
        """Fold each resource pair into the name this skill reads.

        The framework spelling and the standalone spelling are two ways
        of asking for the same resource, so the larger wins (#306): a
        job that says 8 ranks anywhere gets 8, and a value can never be
        silently shrunk by the other spelling arriving later with the
        default.
        """
        def _larger(*values):
            numbers = []
            for value in values:
                try:
                    numbers.append(max(1, int(value or 1)))
                except (TypeError, ValueError):
                    continue
            return max(numbers or [1])

        self.thread = _larger(self.thread, self.lammps_thread)
        self.nproc = _larger(self.nproc, self.lammps_nproc)
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
    """LAMMPS skill runtime configuration container.

    Supports both nested access (config.calculation.method)
    and flat access (config.method) via __getattr__.
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
