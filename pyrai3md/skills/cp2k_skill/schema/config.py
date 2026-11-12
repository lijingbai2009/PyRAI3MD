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

This module provides the unified configuration schema for the CP2K skill,
including system information, calculation parameters, and runtime settings.
It does not include numpy arrays in the schema, as they are not JSON serializable.

"""

from typing import Annotated, Dict, List, Optional
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


def _normalize_pbc(value):
    """Coerce a pbc specification into a list of ints.

    Accepts a list (yaml / framework flatten) or a space-separated
    string (classic input '1 1 1'), so a skill config built from
    either source validates. The classic-input skill section reaches
    build_config as a raw string; this mirrors the framework
    MOLECULE_TYPES 'il' conversion. The annotation survives the
    flat-keyword routing into the owning sub-model, unlike a class-level field validator.
    """
    if value is None:
        return list(DEFAULT['pbc'])
    if isinstance(value, str):
        parts = value.replace(',', ' ').split()
        if not parts:
            return list(DEFAULT['pbc'])
        return [int(float(part)) for part in parts]
    return value


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
    current_state: int = 0
    state_label: List[str] = []
    state_spin: List[int] = []               # (nstate,) 2S per state (0=singlet, 2=triplet)

    npair: int = 0
    nnac: int = 0
    nsoc: int = 0
    nsing_root: int = 0                     # excited singlet count (set by normalize)
    ntrip_root: int = 0                     # triplet count (set by normalize)

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
    pbc: Annotated[List[int], BeforeValidator(_normalize_pbc)] = DEFAULT['pbc']
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
    # Whole-system stress tensor (nstate, 3, 3), the tensile-positive
    # potential stress in GPa (manifest units.stress); the framework
    # converts it centrally to Hartree/Bohr^3. The 'analytical' route
    # fills the ground-state (SCF) block from the CP2K 'STRESS TENSOR'
    # log block; the 'num' route fills every state block from the
    # strain finite differences of service/stress/.
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
    """Calculation settings: method, profile, functional, convergence."""
    project: str = DEFAULT['project']
    profile: str = DEFAULT['profile']
    method: str = DEFAULT['method']
    functional: str = DEFAULT['functional']

    # Plane-wave cutoffs (Ry): cutoff -> &MGRID CUTOFF, ref_cutoff ->
    # &MGRID REL_CUTOFF (written explicitly, never left to CP2K).
    cutoff: float = DEFAULT['cutoff']
    ref_cutoff: float = DEFAULT['ref_cutoff']

    # Basis set / pseudopotential library files (multiple allowed,
    # e.g. BASIS_MOLOPT + BASIS_ADMM_UUZ for ADMM runs).
    basis_file: List[str] = DEFAULT['basis_file']
    potential_file: List[str] = DEFAULT['potential_file']
    auxiliary_basis_file: List[str] = DEFAULT['auxiliary_basis_file']

    # Element-level basis / potential / auxiliary basis names; the
    # non-empty auxiliary_basis dict enables ADMM. normalize fills
    # defaults per atom, user entries take priority.
    basis: Dict[str, str] = DEFAULT['basis']
    potential: Dict[str, str] = DEFAULT['potential']
    auxiliary_basis: Dict[str, str] = DEFAULT['auxiliary_basis']

    # Grimme dispersion correction (d3/d3bj) -> &VDW_POTENTIAL;
    # empty or 'none' omits the section.
    dispersion: str = DEFAULT['dispersion']

    # Whole-system stress tensor request. The framework planning route
    # (md+ensemble=npt or optimize+optimize_cell) injects stress=True;
    # a standalone test yaml sets it directly. Orthogonal to the
    # profile: the input builder adds the CP2K stress keywords on top
    # of the active profile and the parser reads the tensor
    # non-blocking (zero-filled when absent).
    stress: bool = DEFAULT['stress']
    # Stress evaluation method (schema/reference.py STRESS_METHODS):
    # 'analytical' asks CP2K for STRESS_TENSOR ANALYTICAL and is the
    # ground-state route the engine prints; 'num' is the external
    # finite-difference strain route driven by this skill -- it sends no
    # STRESS_TENSOR keyword at all, which is why it is the only route
    # that can produce an excited-state tensor. (`schema/validate.py`
    # refuses 'analytical' stress under method tddft because CP2K aborts
    # with 'Stress Tensor not available for TDDFT calculations',
    # response_solver.F:943, measured in the test environment; the engine
    # offers no numerical TDDFT stress either, also measured, which is
    # why no STRESS_TENSOR NUMERICAL route is exposed.)
    stress_method: str = DEFAULT['stress_method']
    # Dimensionless strain step of the 'num' route. The central
    # difference trades truncation error (grows with the step) against
    # energy-noise amplification (shrinks with the step), so the step is
    # an input, not a constant; validate refuses a step outside
    # (0, 1e-2]. A written 0 is the schema's unset marker (as for
    # `cutoff`), so it takes the default rather than being refused.
    stress_delta: float = DEFAULT['stress_delta']
    # Strain scope of the 'num' route: 'full' samples the six
    # independent tensor components (12 perturbed single points), 'iso'
    # the uniform dilation only (2 points, reported on the diagonal).
    stress_scope: str = DEFAULT['stress_scope']

    # Slab dipole correction: &DFT SURF_DIP .TRUE. + SURF_DIP_DIR Z.
    # CP2K only implements it for a slab normal parallel to one
    # Cartesian axis, so validate refuses non-periodic and tilted cells
    # (surface_dipole.F aborts otherwise).
    dipole_correction: bool = DEFAULT['dipole_correction']

    # TDDFPT settings
    tddft_nstates: int = DEFAULT['tddft_nstates']  # computed by normalize
    tddft_dipole_form: str = DEFAULT['tddft_dipole_form']
    tddft_maxiter: int = DEFAULT['tddft_maxiter']
    tddft_converge: float = DEFAULT['tddft_converge']

    # Target states
    target_states: List[int] = DEFAULT['target_states']
    target_nac_pair: List[List[int]] = [[]]
    target_soc_pair: List[List[int]] = [[]]

    # Convergence
    scf_converge: float = DEFAULT['scf_converge']
    scf_maxit: int = DEFAULT['scf_maxit']  # &SCF MAX_SCF limit
    opt_converge: bool = DEFAULT['opt_converge']
    opt_maxit: int = DEFAULT['opt_maxit']

    # Standalone &MOTION MD settings (driver=skill md profiles only;
    # framework AIMD steps through the gradient profile instead).
    md_nstep: int = DEFAULT['md_nstep']
    md_timestep: float = DEFAULT['md_timestep']  # fs
    md_ensemble: str = DEFAULT['md_ensemble']    # NVE / NVT (CSVR)

    # Reserved internal-iteration algorithm selectors (empty =
    # program default). Consumed by the skill once it exposes its
    # own optimizer options.
    optimizer: str = ''      # internal geometry optimizer algorithm
    scf_converger: str = ''  # SCF convergence optimizer (diis/ot/...)

    verbose: int = DEFAULT['verbose']
    comment: str = ''


class Environment(BaseModel):
    """Computational environment settings for CP2K."""
    skill_name: str = 'cp2k_skill'
    skill_path: str = 'backend'
    execution_mode: str = DEFAULT['execution_mode']
    # Skill-level shell commands (skills.cp2k_skill.shell of
    # pmd_env.yaml / test input; module loads, ...), copied verbatim
    # into the local/slurm run scripts. The global environment.shell
    # is reserved for the main framework (#135).
    shell: Annotated[str, BeforeValidator(_normalize_shell)] = DEFAULT['shell']
    # Dispatch keywords of the &cp2k section, received here so the skill
    # can plan the fan-out of its own perturbed single points (the
    # 'num' strain route). They are deliberately NOT registered in
    # reference.DEFAULT: the framework merge already takes them from the
    # raw &cp2k section, and a DEFAULT entry would write them into every
    # generated classic input.
    #
    # Tri-state, because that is what actually arrives: the merged skill
    # dict carries these keys with a None value whenever the user did not
    # write them, and a plain `bool`/`int` field would reject the None and
    # break every framework-driven CP2K run. None means undeclared, which
    # the skill reads as serial -- the same aggregation the dispatch layer
    # applies to them (dispatch/planner.py::aggregate_tristate).
    multitask: Optional[bool] = None
    # Pool size of the fan-out when declared; None or 1 keeps a single
    # worker, the reading the framework's Scheduler gives it.
    multipool: Optional[int] = None
    # True asks for the remote (slurm) arm of this skill's runner; the
    # arm actually used stays the authority of execution_mode, so the
    # two spellings are cross-checked instead of silently folded.
    remotetask: Optional[bool] = None
    program: str = 'cp2k'

    # CP2K-specific runtime
    cp2k_project: str = DEFAULT['cp2k_project']
    cp2k_path: str = DEFAULT['cp2k_path']    # CP2K_HOME installation directory
    cp2k_exe: str = DEFAULT['cp2k_exe']      # empty -> $CP2K_HOME/exe/local/cp2k.psmp
    setup_script: str = DEFAULT['setup_script']  # toolchain setup script to source

    # Resource management
    keep_tmp: int = DEFAULT['keep_tmp']
    # Compute resources (#306). `thread`/`nproc` are what this skill
    # reads (nproc replaced the old `mpi` rank count, thread the old
    # run-wide num_thread); `cp2k_thread`/`cp2k_nproc` are the spellings the
    # framework forwards from &cp2k. The larger of each pair wins.
    thread: int = DEFAULT['thread']
    cp2k_thread: int = DEFAULT['thread']
    nproc: int = DEFAULT['nproc']
    cp2k_nproc: int = DEFAULT['nproc']

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

        Both spellings ask for the same resource, so the larger wins:
        neither one can silently shrink a request made in the other.
        """
        def _larger(*values):
            numbers = []
            for value in values:
                try:
                    numbers.append(max(1, int(value or 1)))
                except (TypeError, ValueError):
                    continue
            return max(numbers or [1])

        self.thread = _larger(self.thread, self.cp2k_thread)
        self.nproc = _larger(self.nproc, self.cp2k_nproc)
        return self

    @model_validator(mode='after')
    def _check_dispatch_keywords(self):
        """Cross-check the dispatch keywords against the run arm.

        `remotetask` asks for the remote arm while `execution_mode` is
        the spelling that actually selects it, so the two can disagree.
        Neither one outranks the other: a mismatch is reported instead of
        folded, because silently switching a local run to sbatch (or
        ignoring the request) would both oversubmit the cluster and make
        the reported arm a lie.
        """
        if self.multipool is not None and self.multipool < 1:
            raise ValueError(
                f"cp2k_skill: multipool must be >= 1, got {self.multipool}.")
        if self.remotetask and self.execution_mode != 'slurm':
            raise ValueError(
                "cp2k_skill: remotetask is set but the run arm is "
                f"execution_mode='{self.execution_mode}'; write "
                "`execution_mode: slurm` (&cp2k: cp2k_execution_mode) "
                "to run the calculation through sbatch, or drop "
                "remotetask to keep this arm local.")
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
    """CP2K skill runtime configuration container.

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
