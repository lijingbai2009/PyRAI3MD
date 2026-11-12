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

This module assembles the complete CP2K input file from the compiled
workflow and configuration. Coordinates are written as a separate
xyz file referenced via &TOPOLOGY COORD_FILE_NAME (Angstrom, CP2K
native unit), following the cp2k_demo layout.

"""

import os

from ...schema.reference import DISPERSION_KEYWORD
# The one place that knows how a framework state row is numbered by CP2K, so the
# LIST written here and the headers read back by the parser cannot drift apart.
from ..parser import state_map

# Vacuum padding (Angstrom) added around the molecule in each
# direction when no periodic cell is provided.
_VACUUM_PADDING = 10.0


def write_input_file(config, workflow, calcdir: str) -> str:
    """Build and write the CP2K input file to the calculation directory.

    Combines build_input() with file writing. The input is named
    $CP2K_PROJECT.inp and the coordinates $CP2K_PROJECT.xyz, both
    placed in calcdir.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.

    Returns
    -------
    str
        Path to the written input file.
    """
    project = config.project

    _write_xyz_file(config, calcdir)

    input_content = build_input(config, workflow)
    input_path = os.path.join(calcdir, f'{project}.inp')
    with open(input_path, 'w') as f:
        f.write(input_content)
    return input_path


def build_input(config, workflow) -> str:
    """Build the complete CP2K input file content.

    Assembles the input from the workflow steps:
    - &GLOBAL: PROJECT + RUN_TYPE chosen by the profile
      (ENERGY / ENERGY_FORCE / GEO_OPT / MD)
    - &FORCE_EVAL: &SUBSYS (&CELL / &TOPOLOGY / &KIND), &DFT
      (&QS / &POISSON / &MGRID / &SCF / &XC, optional ADMM),
      &PRINT &FORCES for gradients and &PROPERTIES &TDDFPT as
      siblings of &DFT (demo azo-1.inp layout)
    - &MOTION for GEO_OPT and standalone MD runs

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        Complete CP2K input file content.
    """
    blocks = []

    blocks.append(_build_global(config, workflow))
    blocks.append(_build_force_eval(config, workflow))

    motion_block = _build_motion(config, workflow)
    if motion_block:
        blocks.append(motion_block)

    return '\n\n'.join(blocks) + '\n'


def _build_global(config, workflow) -> str:
    """Build the &GLOBAL section; RUN_TYPE follows the profile."""
    run_type = _resolve_run_type(workflow)

    # EXTENDED_FFT_LENGTHS lifts the FFT grid factorization constraint;
    # without it arbitrary cell sizes can fail with
    # "Index to radix array not found."
    lines = [
        '&GLOBAL',
        f'  PROJECT {config.project}',
        '  PRINT_LEVEL LOW',
        f'  RUN_TYPE {run_type}',
        '  EXTENDED_FFT_LENGTHS TRUE',
        '&END GLOBAL',
    ]
    return '\n'.join(lines)


def _resolve_run_type(workflow) -> str:
    """Map the workflow onto a CP2K RUN_TYPE keyword.

    Gradient single points use ENERGY_FORCE (forces come from the
    &PRINT &FORCES file); optimizations and standalone MD runs use
    the &MOTION machinery.
    """
    if workflow.is_optimization:
        return 'GEO_OPT'
    if workflow.is_md:
        return 'MD'
    if workflow.has_step('gradient'):
        return 'ENERGY_FORCE'
    return 'ENERGY'


def _build_force_eval(config, workflow) -> str:
    """Build the &FORCE_EVAL section (SUBSYS + DFT + PRINT + PROPERTIES).

    &PROPERTIES is a sibling of &DFT inside &FORCE_EVAL (CP2K
    rejects it nested in &DFT: 'unknown subsection PROPERTIES of
    section DFT'); the demo azo-1.inp ordering is kept.
    """
    lines = [
        '&FORCE_EVAL',
        '  METHOD Quickstep',
    ]

    # Stress evaluation keyword (STRESS_TENSOR ANALYTICAL/NUMERICAL)
    # merged into the existing &FORCE_EVAL, never a second block
    # (dev/skills/STRESS_DEVELOPMENT.md section 2.1). Absent for the num
    # route: that one asks CP2K for no tensor at all and builds the
    # stress from the strain single points (service/stress/).
    stress_keyword = _stress_keyword(config)
    if stress_keyword:
        lines.append(f'  STRESS_TENSOR {stress_keyword}')

    lines += [
        _indent(_build_subsys(config), 2),
        _indent(_build_dft(config, workflow), 2),
    ]

    print_block = _build_print(config, workflow)
    if print_block:
        lines.append(_indent(print_block, 2))

    if workflow.is_tddft:
        lines.append(_indent(_build_tddfpt(config, workflow), 2))

    lines.append('&END FORCE_EVAL')
    return '\n'.join(lines)


def _stress_keyword(config):
    """Return the CP2K STRESS_TENSOR keyword for the stress method.

    'analytical' -> ANALYTICAL (the route CP2K prints a tensor for,
    ground state); 'num' -> None on purpose, because the external strain
    route asks the engine for nothing beyond an energy and builds the
    tensor from the +/- strain single points. Any other spelling is
    refused earlier, by schema/validate.py::validate_stress. None when
    stress is off.
    """
    if not getattr(config, 'stress', False):
        return None
    if str(getattr(config, 'stress_method', 'analytical')).lower() \
            == 'analytical':
        return 'ANALYTICAL'
    return None


def _is_periodic(config) -> bool:
    """True when periodicity is explicitly enabled through pbc.

    pbc is the single normalized periodicity authority
    (dev/methods/PBC_DEVELOPMENT.md section 3.2): a non-zero lattice
    alone does NOT make a system periodic (a molecular run may carry a
    cell for geometry only), and the default cell sentinel must never
    be mistaken for a periodic box. Only the framework/skill validation
    decides whether the pbc/lattice combination is legal; this helper
    just reads the authority.
    """
    return bool(config.pbc and any(int(p) for p in config.pbc))


def _build_cell_block(config) -> str:
    """Build the &CELL section.

    Periodic systems use the framework lattice vectors; molecular
    systems get an automatic vacuum box around the coordinates. Only
    full 3D periodicity is supported in this stage: partial
    periodicity (mixed pbc) is rejected earlier by the framework and
    skill validation, because the low-dimensional CP2K electrostatics
    solver mapping is not defined yet (PBC_DEVELOPMENT.md stage E).
    """
    if _is_periodic(config):
        lattice = config.lattice_vector or []
        if len(lattice) != 3 or not any(
                abs(v) > 1e-8 for row in lattice for v in row):
            raise ValueError(
                "CP2K periodic run (pbc=%r) requires a valid "
                "lattice_vector; got %r" % (config.pbc, lattice))
        a, b, c = lattice
        lines = [
            '&CELL',
            f'  A {a[0]:.8f} {a[1]:.8f} {a[2]:.8f}',
            f'  B {b[0]:.8f} {b[1]:.8f} {b[2]:.8f}',
            f'  C {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}',
            '  PERIODIC XYZ',
            '&END CELL',
        ]
        return '\n'.join(lines)

    # Molecular mode: vacuum box from the coordinate extent
    coord = config.coordinates or [[0.0, 0.0, 0.0]]
    span = []
    for i in range(3):
        values = [row[i] for row in coord]
        span.append(max(values) - min(values) + 2.0 * _VACUUM_PADDING)
    box = [max(length, 20.0) for length in span]

    lines = [
        '&CELL',
        f'  A {box[0]:.4f} 0.0 0.0',
        f'  B 0.0 {box[1]:.4f} 0.0',
        f'  C 0.0 0.0 {box[2]:.4f}',
        '  PERIODIC NONE',
        '&END CELL',
    ]
    return '\n'.join(lines)


def _build_subsys(config) -> str:
    """Build the &SUBSYS section (&CELL, &TOPOLOGY, &KIND blocks)."""
    lines = ['&SUBSYS']
    lines.append(_indent(_build_cell_block(config), 2))

    topology = [
        '&TOPOLOGY',
        f'  COORD_FILE_NAME {config.project}.xyz',
        '  COORD_FILE_FORMAT XYZ',
        '&END TOPOLOGY',
    ]
    lines.append(_indent('\n'.join(topology), 2))

    for block in _build_kind_blocks(config):
        lines.append(_indent(block, 2))

    lines.append('&END SUBSYS')
    return '\n'.join(lines)


def _build_kind_blocks(config) -> list:
    """Build one &KIND block per element present in the system.

    Element-level basis / potential names come from normalize
    (user overrides take priority over the qN-suffixed defaults);
    ADMM runs add a second BASIS_SET AUX_FIT line.
    """
    elements = sorted(set(config.atoms))
    basis = config.basis or {}
    potential = config.potential or {}
    auxiliary_basis = config.auxiliary_basis or {}

    blocks = []
    for element in elements:
        lines = [f'&KIND {element}', f'  ELEMENT {element}']
        if element in basis:
            lines.append(f'  BASIS_SET {basis[element]}')
        if element in auxiliary_basis:
            lines.append(f'  BASIS_SET AUX_FIT {auxiliary_basis[element]}')
        if element in potential:
            lines.append(f'  POTENTIAL {potential[element]}')
        lines.append('&END KIND')
        blocks.append('\n'.join(lines))

    return blocks


def _build_dft(config, workflow) -> str:
    """Build the &DFT section."""
    lines = ['&DFT']

    for basis_file in config.basis_file or []:
        lines.append(f'  BASIS_SET_FILE_NAME {basis_file}')
    for potential_file in config.potential_file or []:
        lines.append(f'  POTENTIAL_FILE_NAME {potential_file}')

    lines.append(f'  CHARGE {config.charges}')
    lines.append(f'  MULTIPLICITY {config.multiplicity}')

    # &QS: ASPC extrapolation stabilizes the SCF across MD steps
    qs_lines = ['&QS', '  EPS_DEFAULT 1.0E-12']
    if workflow.is_md:
        qs_lines.append('  EXTRAPOLATION ASPC')
    qs_lines.append('&END QS')
    lines.append(_indent('\n'.join(qs_lines), 2))

    lines.append(_indent(_build_poisson(config), 2))

    # ADMM: enabled by a non-empty auxiliary_basis dict
    if config.auxiliary_basis:
        lines.append(_indent(_build_admm(workflow), 2))

    lines.append(_indent(_build_xc(config), 2))

    mgrid = [
        '&MGRID',
        f'  CUTOFF {config.cutoff:g}',
        f'  REL_CUTOFF {config.ref_cutoff:g}',
        '&END MGRID',
    ]
    lines.append(_indent('\n'.join(mgrid), 2))

    lines.append(_indent(_build_scf(config), 2))

    if getattr(config, 'dipole_correction', False):
        # SURF_DIP is CP2K's slab correction. The direction is fixed to
        # Z because validate refuses a cell whose c vector is not on z
        # (surface_dipole.F aborts on a tilted cell).
        lines.append('  SURF_DIP .TRUE.')
        lines.append('  SURF_DIP_DIR Z')

    lines.append('&END DFT')
    return '\n'.join(lines)


def _build_poisson(config) -> str:
    """Build the &POISSON section (periodic vs molecular solver)."""
    if _is_periodic(config):
        lines = [
            '&POISSON',
            '  PERIODIC XYZ',
            '  PSOLVER PERIODIC',
            '&END POISSON',
        ]
    else:
        lines = [
            '&POISSON',
            '  PERIODIC NONE',
            '  PSOLVER WAVELET',
            '&END POISSON',
        ]
    return '\n'.join(lines)


def _build_admm(workflow) -> str:
    """Build the &AUXILIARY_DENSITY_MATRIX_METHOD section.

    ADMM_PURIFICATION_METHOD NONE is the only safe choice for
    TDDFPT runs (demo verified).
    """
    lines = [
        '&AUXILIARY_DENSITY_MATRIX_METHOD',
        '  METHOD BASIS_PROJECTION',
    ]
    if workflow.is_tddft:
        lines.append('  ADMM_PURIFICATION_METHOD NONE')
    lines.append('&END AUXILIARY_DENSITY_MATRIX_METHOD')
    return '\n'.join(lines)


def _build_xc(config) -> str:
    """Build the &XC section.

    pbe -> &XC_FUNCTIONAL PBE; pbe0 follows the verified bulk.inp
    layout: &PBE SCALE_X 0.75 + &HF FRACTION 0.25 (screened by a
    truncated interaction potential). Dispersion corrections attach
    as &VDW_POTENTIAL with REFERENCE_FUNCTIONAL following the
    calculation functional.
    """
    functional = config.functional
    lines = ['&XC']

    if functional == 'pbe':
        lines.append('  &XC_FUNCTIONAL PBE')
        lines.append('  &END XC_FUNCTIONAL')
    elif functional == 'pbe0':
        lines.append('  &XC_FUNCTIONAL')
        lines.append('    &PBE')
        lines.append('      SCALE_X 0.75')
        lines.append('      SCALE_C 1.0')
        lines.append('    &END PBE')
        lines.append('  &END XC_FUNCTIONAL')
        lines.append(_indent(_build_hf(), 2))

    vdw_block = _build_vdw(config)
    if vdw_block:
        lines.append(_indent(vdw_block, 2))

    lines.append('&END XC')
    return '\n'.join(lines)


def _build_hf() -> str:
    """Build the &HF exchange block for hybrid functionals."""
    lines = [
        '&HF',
        '  FRACTION 0.25',
        '  &SCREENING',
        '    EPS_SCHWARZ 1E-6',
        '  &END SCREENING',
        '  &INTERACTION_POTENTIAL',
        '    POTENTIAL_TYPE TRUNCATED',
        '    CUTOFF_RADIUS 6.0',
        '  &END INTERACTION_POTENTIAL',
        '  &MEMORY',
        '    MAX_MEMORY 3000',
        '    EPS_STORAGE_SCALING 0.1',
        '  &END MEMORY',
        '&END HF',
    ]
    return '\n'.join(lines)


def _build_vdw(config) -> str:
    """Build the &VDW_POTENTIAL section for d3/d3bj corrections.

    REFERENCE_FUNCTIONAL always follows the calculation functional
    so pbe0+d3 never writes PBE.
    """
    dispersion = config.dispersion
    if not dispersion:
        return ''

    potential_type = DISPERSION_KEYWORD[dispersion]
    reference = config.functional.upper()

    lines = [
        '&VDW_POTENTIAL',
        '  POTENTIAL_TYPE PAIR_POTENTIAL',
        '  &PAIR_POTENTIAL',
        '    TYPE ' + potential_type,
        f'    REFERENCE_FUNCTIONAL {reference}',
        '  &END PAIR_POTENTIAL',
        '&END VDW_POTENTIAL',
    ]
    return '\n'.join(lines)


def _build_scf(config) -> str:
    """Build the &SCF section (OT minimizer, demo default pattern)."""
    lines = [
        '&SCF',
        f'  MAX_SCF {config.scf_maxit}',
        f'  EPS_SCF {config.scf_converge:.1E}',
        '  SCF_GUESS ATOMIC',
        '  &OT',
        '    MINIMIZER DIIS',
        '    PRECONDITIONER FULL_SINGLE_INVERSE',
        '  &END OT',
        '&END SCF',
    ]
    return '\n'.join(lines)


def _build_tddfpt(config, workflow) -> str:
    """Build the &PROPERTIES &TDDFPT section.

    DIPOLE_FORM is a keyword of the &DIPOLE_MOMENTS subsection, not
    of &TDDFPT itself. Excited-state gradients print one force block
    per requested excited state into the td_forces tdfrc file (LIST
    order matches the target states).

    RKS_TRIPLETS follows the spin content of the request rather than a fixed
    letter: CP2K solves the triplet block only when it is asked for, and the
    count of triplet roots the framework holds is already normalized into
    `ntrip_root` from `state_spin`. With it off, a declared triplet row would be
    answered by a second singlet root wearing that row's name and the same
    number of table rows - which is why an all-singlet request must keep writing
    `F`, unchanged down to the byte, so the existing cases stay the identity test
    for this branch.
    """
    nstates = config.tddft_nstates
    triplets = 'T' if int(getattr(config, 'ntrip_root', 0) or 0) > 0 else 'F'

    lines = [
        '&PROPERTIES',
        '  &TDDFPT',
        f'    NSTATES {nstates}',
        f'    RKS_TRIPLETS {triplets}',
        f'    CONVERGENCE {config.tddft_converge:.1E}',
        f'    MAX_ITER {config.tddft_maxiter}',
        '    MIN_AMPLITUDE 0.01',
        '    &DIPOLE_MOMENTS',
        f'      DIPOLE_FORM {config.tddft_dipole_form.upper()}',
        '    &END DIPOLE_MOMENTS',
    ]

    if workflow.has_step('gradient'):
        state_map.require_addressable(config)
        numbers = state_map.list_numbers(config)
        if numbers:
            state_list = ' '.join(str(s) for s in numbers)
            lines.append('    &PRINT')
            lines.append('      &FORCES')
            lines.append(f'        LIST {state_list}')
            lines.append('        filename td_forces')
            lines.append('      &END FORCES')
            lines.append('    &END PRINT')

    lines.append('  &END TDDFPT')
    lines.append('&END PROPERTIES')
    return '\n'.join(lines)


def _build_print(config, workflow) -> str:
    """Build the &FORCE_EVAL &PRINT section.

    Ground-state gradients are read from {project}-force-1_0.xyz
    (ATOMIC FORCES in [a.u.]); force_unit hartree/bohr keeps the
    framework unit contract. The whole-system stress tensor is written
    to a dedicated file {project}_stress_tensor.dat via the
    &STRESS_TENSOR FILENAME keyword (the leading '=' locks the exact
    name so a multi-task calcdir never collides), mirroring the force
    file pattern so the parser reads a small dedicated file instead of
    scanning the main log (dev/skills/STRESS_DEVELOPMENT.md section
    2.1/2.2). The two subsections share one &PRINT block.
    """
    needs_forces = workflow.has_step('gradient')
    needs_stress = _stress_keyword(config) is not None
    if not needs_forces and not needs_stress:
        return ''

    lines = ['&PRINT']
    if needs_forces:
        lines += [
            '  &FORCES ON',
            '    force_unit hartree/bohr',
            '    filename force',
            '  &END FORCES',
        ]
    if needs_stress:
        project = getattr(config, 'project', 'cp2k')
        lines += [
            '  &STRESS_TENSOR ON',
            f'    FILENAME ={project}_stress_tensor.dat',
            '  &END STRESS_TENSOR',
        ]
    lines.append('&END PRINT')
    return '\n'.join(lines)


def _build_motion(config, workflow) -> str:
    """Build the &MOTION section for GEO_OPT and standalone MD."""
    if not (workflow.is_optimization or workflow.is_md):
        return ''

    lines = ['&MOTION']

    if workflow.is_optimization:
        lines.append(_indent(_build_geo_opt(config, workflow), 2))
    else:
        lines.append(_indent(_build_md(config), 2))

    # Trajectory output: {project}-pos-1.xyz carries the relaxed
    # structure (optimize) or the MD frame history (standalone md).
    trajectory = [
        '&PRINT',
        '  &TRAJECTORY',
        '    FORMAT xyz',
        '  &END TRAJECTORY',
        '&END PRINT',
    ]
    lines.append(_indent('\n'.join(trajectory), 2))

    lines.append('&END MOTION')
    return '\n'.join(lines)


def _build_geo_opt(config, workflow) -> str:
    """Build the &GEO_OPT block (BFGS, demo threshold pattern)."""
    lines = [
        '&GEO_OPT',
        '  TYPE MINIMIZATION',
        '  OPTIMIZER BFGS',
        f'  MAX_ITER {config.opt_maxit}',
        '  MAX_DR 3E-3',
        '  RMS_DR 1.5E-3',
        '  MAX_FORCE 4.5E-4',
        '  RMS_FORCE 3E-4',
    ]

    # Excited-state optimization tracks the active state through the
    # TDDFPT optimize machinery.
    if workflow.is_tddft and config.current_state > 0:
        lines.append('  &OPTIMIZE')
        lines.append(f'    STATE {config.current_state}')
        lines.append('  &END OPTIMIZE')

    lines.append('&END GEO_OPT')
    return '\n'.join(lines)


def _build_md(config) -> str:
    """Build the standalone &MD block (NVT via CSVR thermostat)."""
    ensemble = str(config.md_ensemble).upper()
    temperature = config.temperature

    lines = [
        '&MD',
        f'  ENSEMBLE {ensemble}',
        f'  STEPS {config.md_nstep}',
        f'  TIMESTEP {config.md_timestep:g}',
        f'  TEMPERATURE {temperature:g}',
    ]

    if ensemble == 'NVT':
        lines += [
            '  &THERMOSTAT',
            '    TYPE CSVR',
            '    &CSVR',
            '      TIMECON 100',
            '    &END CSVR',
            '  &END THERMOSTAT',
        ]

    lines.append('&END MD')
    return '\n'.join(lines)


def _write_xyz_file(config, calcdir: str):
    """Write the current coordinates as {project}.xyz (Angstrom)."""
    atoms = config.atoms
    coordinates = config.coordinates

    lines = [str(len(atoms)), config.project]
    for atom, (x, y, z) in zip(atoms, coordinates):
        lines.append(f'{atom} {x:.8f} {y:.8f} {z:.8f}')

    xyz_path = os.path.join(calcdir, f'{config.project}.xyz')
    with open(xyz_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return xyz_path


def _indent(text: str, spaces: int) -> str:
    """Indent every line of a section block."""
    pad = ' ' * spaces
    return '\n'.join(pad + line if line else line
                     for line in text.splitlines())
