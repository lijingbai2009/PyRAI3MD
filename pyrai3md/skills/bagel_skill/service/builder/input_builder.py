"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module builds BAGEL JSON input files from configuration and workflow.
It constructs each JSON block (molecule, casscf, forces, caspt2, etc.)
and assembles the complete BAGEL input as a JSON file.

Key design decisions:
- Coordinates are converted from Angstrom to Bohr (BAGEL uses Bohr).
- Background point charges are embedded as Q atoms in the geometry list.
- Archive files (.archive) manage orbital data between calculations.
- The converged archive from a previous step serves as the guess
  for the next step.

"""

import os
import re
import json
import shutil


# Internal-coordinate step cap for the optimize block, in Bohr. This is a
# template constant rather than a user keyword: its job is to keep one bad
# gradient from ending the run, and every profile that optimizes wants the
# same protection. See _build_optimize().
_OPT_MAX_STEP = 0.05


# BAGEL density-fitting basis sets (cluster-validated). An empty
# df_basis config value is resolved against this list by similarity
# matching; see _match_df_basis().
DF_BASIS_SETS = [
    'svp-jkfit',
    'tzvpp-jkfit',
    'qzvpp-jkfit',
    'cc-pvdz-jkfit',
    'cc-pvtz-jkfit',
    'cc-pvqz-jkfit',
    'cc-pv5z-jkfit',
]


def _match_df_basis(basis: str) -> str:
    """Resolve the density-fitting basis by similarity matching.

    BAGEL ships only seven jkfit auxiliary sets. An orbital basis of
    the svp/tzvpp/qzvpp family maps to its same-name jkfit set; a
    Dunning basis maps to the cc-pv{X}z-jkfit set carrying the same
    cardinal number X (augmentation, core-valence and -dk prefixes
    are ignored; 6z downgrades to the closest 5z set). Bases without
    any match (sto-3g, 3-21g, 6-31g, ano-rcc, ...) fall back to the
    default svp-jkfit.

    Parameters
    ----------
    basis : str
        Orbital basis name (lowercase).

    Returns
    -------
    str
        Matched density-fitting basis name.
    """
    if basis in ('svp', 'tzvpp', 'qzvpp'):
        return f'{basis}-jkfit'

    match = re.search(r'cc-pw?c?v?([dtq56])z', basis)
    if match:
        cardinal = match.group(1)
        # No 6z jkfit set exists; use the closest 5z auxiliary set
        cardinal = '5' if cardinal == '6' else cardinal
        candidate = f'cc-pv{cardinal}z-jkfit'
        if candidate in DF_BASIS_SETS:
            return candidate

    return 'svp-jkfit'


def write_input_file(config, workflow, calcdir: str) -> str:
    """Build and write the BAGEL JSON input file to the calculation directory.

    Combines build_input() with file writing. The file is named
    $BAGEL_PROJECT.json and placed in calcdir.

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
    input_content = build_input(config, workflow, calcdir)
    project = config.project
    input_path = os.path.join(calcdir, f'{project}.json')
    with open(input_path, 'w') as f:
        f.write(input_content)
    return input_path


def build_input(config, workflow, calcdir: str = '') -> str:
    """Build the complete BAGEL JSON input file content.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str, optional
        Calculation directory. Used to decide whether the guess
        archive exists before writing a load_ref block; empty
        string disables the existence check.

    Returns
    -------
    str
        Complete BAGEL JSON input content.
    """
    bagel_blocks = []

    for step in workflow.steps:
        step_lower = step.lower()

        if step_lower == 'molecule':
            bagel_blocks.append(_build_molecule(config))
        elif step_lower == 'casscf':
            bagel_blocks.append(_build_casscf(config))
        elif step_lower == 'save_ref':
            bagel_blocks.append(_build_save_ref(config))
        elif step_lower == 'load_ref':
            # A load_ref block without an existing guess archive
            # aborts BAGEL with 'File not found'. The archive is
            # staged into calcdir by manage_archives before this
            # point; when neither the converged archive from a
            # previous step nor a user guess is available the block
            # is skipped (same policy as the molcas orbital guess).
            guess_path = os.path.join(
                calcdir, f'{config.project}_guess.archive'
            ) if calcdir else ''
            if not calcdir or os.path.exists(guess_path):
                bagel_blocks.append(_build_load_ref(config))
        elif step_lower == 'forces':
            bagel_blocks.append(_build_forces(config, nac=False))
        elif step_lower == 'forces_nacme':
            bagel_blocks.append(_build_forces(config, nac=True))
        elif step_lower == 'hessian':
            bagel_blocks.append(_build_hessian_casscf(config))
        elif step_lower == 'smith_caspt2':
            bagel_blocks.append(_build_smith_caspt2(config))
        elif step_lower == 'forces_caspt2':
            bagel_blocks.append(_build_forces_caspt2(config, nac=False))
        elif step_lower == 'forces_caspt2_nacme':
            bagel_blocks.append(_build_forces_caspt2(config, nac=True))
        elif step_lower == 'hessian_caspt2':
            bagel_blocks.append(_build_hessian_caspt2(config))
        elif step_lower == 'optimize':
            bagel_blocks.append(_build_optimize(config, workflow))
        elif step_lower == 'forces_dipole':
            bagel_blocks.append(_build_forces_dipole(config, workflow))

    bagel_input = {'bagel': bagel_blocks}
    return json.dumps(bagel_input, indent=2)


def _build_molecule(config) -> dict:
    """Build the molecule block with geometry and basis set.

    Coordinates are converted from Angstrom to Bohr.
    Background point charges are embedded as Q atoms.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Molecule JSON block.
    """
    atoms = config.atoms
    coordinates = config.coordinates
    point_charges = config.point_charges
    basis = config.basis
    df_basis = str(config.df_basis).strip().lower()
    # BAGEL always uses a DF basis. An empty df_basis is resolved by
    # similarity matching against the supported jkfit sets
    # (svp-jkfit when no match exists).
    df_basis = df_basis or _match_df_basis(basis)

    # Build geometry list. PyRAI3MD stores coordinates in Angstrom;
    # the molecule block carries "angstrom": "true" so BAGEL reads
    # them as-is (without this flag BAGEL assumes Bohr).
    geometry = []
    for sym, coord in zip(atoms, coordinates):
        geometry.append({
            'atom': sym,
            'xyz': [
                float(coord[0]),
                float(coord[1]),
                float(coord[2]),
            ]
        })

    # Add point charges as Q atoms. The framework hands over either a
    # list of rows or an (n, 4) array, so the emptiness test cannot ask
    # for the truth value of the object itself.
    if (point_charges is not None and len(point_charges) > 0
            and len(point_charges[0]) > 0):
        for pc in point_charges:
            if len(pc) >= 4:
                geometry.append({
                    'atom': 'Q',
                    'xyz': [
                        float(pc[1]),
                        float(pc[2]),
                        float(pc[3]),
                    ],
                    'charge': float(pc[0]),
                })

    # Resolve basis set paths relative to BAGEL installation
    bagel_path = config.bagel_path
    basis_path = basis
    df_basis_path = df_basis
    if bagel_path:
        basis_path = f'{bagel_path}/share/{basis}.json'
        df_basis_path = f'{bagel_path}/share/{df_basis}.json'

    molecule = {
        'title': 'molecule',
        'basis': basis_path,
        'df_basis': df_basis_path,
        'angstrom': 'true',
        'geometry': geometry,
    }

    return molecule


def _build_casscf(config) -> dict:
    """Build the CASSCF computation block.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        CASSCF JSON block.
    """
    nstate = config.nstate
    nact = config.nact_orb
    nclosed = config.nclosed
    active = config.active
    maxiter = config.maxit_scf

    casscf = {
        'title': 'casscf',
        'natocc': True,
        'nstate': nstate,
        'nact': nact,
        'nclosed': nclosed,
        'maxiter': maxiter,
    }

    if active:
        # Internal indices are 0-based; the BAGEL JSON expects
        # 1-based orbital indices.
        casscf['active'] = [orb + 1 for orb in active]

    return casscf


def _build_casscf_method(config) -> dict:
    """Build the CASSCF method definition embedded in other blocks.

    The forces and optimize blocks carry the method definition in a
    'method' array; the CASSCF definition mirrors the standalone
    casscf block.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        CASSCF method definition block.
    """
    # The standalone casscf block is itself the method definition.
    return _build_casscf(config)


def _build_caspt2_method(config) -> dict:
    """Build the XMS-CASPT2 method definition embedded in other blocks.

    Follows the official BAGEL forces/optimize template: the method
    carries the smith sub-block (ms/xms/sssr/shift/frozen) plus the
    active space. The smith maxiter and the method-level maxiter both
    align with maxit_scf; smith 'shift' comes from pt2_shift and
    'thresh' from scf_converge.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        XMS-CASPT2 method definition block.
    """
    nstate = config.nstate
    nact = config.nact_orb
    nclosed = config.nclosed
    active = config.active
    maxiter = config.maxit_scf

    method = {
        'title': 'caspt2',
        'smith': {
            'method': 'caspt2',
            'ms': True,
            'xms': True,
            'sssr': True,
            'shift': config.pt2_shift,
            'thresh': config.scf_converge,
            'maxiter': maxiter,
            'frozen': True,
        },
        'nstate': nstate,
        'nact': nact,
        'nclosed': nclosed,
        'maxiter': maxiter,
    }

    if active:
        # Internal indices are 0-based; the BAGEL JSON expects
        # 1-based orbital indices.
        method['active'] = [orb + 1 for orb in active]

    return method


def _build_save_ref(config) -> dict:
    """Build the save_ref block to save orbital archive.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        save_ref JSON block.
    """
    project = config.project
    return {
        'title': 'save_ref',
        'file': f'{project}_converged',
    }


def _build_load_ref(config) -> dict:
    """Build the load_ref block to load orbital archive.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        load_ref JSON block.
    """
    project = config.project
    return {
        'title': 'load_ref',
        'file': f'{project}_guess',
    }


def _build_forces(config, nac: bool = False) -> dict:
    """Build the forces block for gradient and/or NAC computation.

    BAGEL requires a 'method' keyword in the forces block defining
    the electronic-structure method; it must be an *array* of method
    definition blocks (a bare dict aborts BAGEL with 'unknown
    method'). The CASSCF definition mirrors the standalone casscf
    block.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    nac : bool
        Whether to include NACME (non-adiabatic coupling) entries.

    Returns
    -------
    dict
        Forces JSON block.
    """
    nstate = config.nstate
    target_states = config.target_states
    target_nac_pair = config.target_nac_pair

    grads = []

    # Build gradient targets (the workflow selects per-step states
    # through calculate(target_states=...))
    for s in target_states:
        if isinstance(s, int) and 0 <= s < nstate:
            grads.append({'title': 'force', 'target': s})

    # Build NACME targets
    if nac:
        nac_pairs = target_nac_pair
        if not nac_pairs or nac_pairs == [[]]:
            # Default: all consecutive state pairs
            nac_pairs = [[i, i + 1] for i in range(nstate - 1)]

        for pair in nac_pairs:
            if len(pair) == 2:
                grads.append({
                    'title': 'nacme',
                    'target': pair[0],
                    'target2': pair[1],
                    'nacmtype': 'interstate',
                })

    forces = {
        'title': 'forces',
        'grads': grads,
        'export': True,
        'method': [_build_casscf_method(config)],
    }

    return forces


def _build_hessian_casscf(config) -> dict:
    """Build the hessian block for CASSCF frequency computation.

    BAGEL uses the 'hessian' block with a 'method' array containing
    the CASSCF definition block. The 'target' field specifies which
    state to compute frequencies for.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Hessian JSON block with CASSCF method.
    """
    target = config.target_states[0]

    hessian = {
        'title': 'hessian',
        'target': target,
        'method': [_build_casscf_method(config)],
    }

    return hessian


def _build_smith_caspt2(config) -> dict:
    """Build the standalone smith block for XMS-CASPT2 energies.

    BAGEL does not support a standalone caspt2-titled block; a
    single-point XMS-CASPT2 energy runs a casscf block followed by a
    smith block that names the caspt2 algorithm. Follows the
    official template: method/ms/xms/sssr flags plus the level
    shift only (no active-space or iteration keywords; those live in
    the preceding casscf block).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Smith JSON block running the CASPT2 algorithm.
    """
    return {
        'title': 'smith',
        'method': 'caspt2',
        'ms': True,
        'xms': True,
        'sssr': True,
        'shift': config.pt2_shift,
    }


def _build_forces_caspt2(config, nac: bool = False) -> dict:
    """Build the forces block with XMS-CASPT2 method.

    Follows the official BAGEL gradient template: the method keyword
    is an array holding the caspt2/smith method definition.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    nac : bool
        Whether to include NACME entries.

    Returns
    -------
    dict
        Forces JSON block with CASPT2 method.
    """
    nstate = config.nstate
    target_states = config.target_states
    target_nac_pair = config.target_nac_pair

    grads = []

    # Build gradient targets (the workflow selects per-step states
    # through calculate(target_states=...))
    for s in target_states:
        if isinstance(s, int) and 0 <= s < nstate:
            grads.append({'title': 'force', 'target': s})

    # Build NACME targets
    if nac:
        nac_pairs = target_nac_pair
        if not nac_pairs or nac_pairs == [[]]:
            nac_pairs = [[i, i + 1] for i in range(nstate - 1)]

        for pair in nac_pairs:
            if len(pair) == 2:
                grads.append({
                    'title': 'nacme',
                    'target': pair[0],
                    'target2': pair[1],
                    'nacmtype': 'interstate',
                })

    forces = {
        'title': 'forces',
        'grads': grads,
        'export': True,
        'method': [_build_caspt2_method(config)],
    }

    return forces


def _build_forces_dipole(config, workflow) -> dict:
    """Build the forces block requesting dipole properties.

    Follows the validated BAGEL dipole workflow: the forces block
    carries the dipole keyword (on/off switch for dipole output) and
    a grads list whose entries disable derivative evaluation via
    ciderivative. Force entries yield state dipoles, nacme entries
    yield transition dipoles; both sets may be requested in the same
    block. Gradients/NAC are obtained by dropping ciderivative and
    appending the needed grads entries.

    Which grads entries are written is driven by the req_dipole /
    req_transition_dipole request flags; unset flags derive from the
    profile ('_dipole' -> force entries, '_transition_dipole' ->
    nacme entries).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph (selects the method level).

    Returns
    -------
    dict
        Forces JSON block with the dipole keyword.
    """
    if workflow.is_caspt2:
        method = _build_caspt2_method(config)
    else:
        method = _build_casscf_method(config)

    want_dipole = config.req_dipole
    want_transition = config.req_transition_dipole
    if want_dipole is None and want_transition is None:
        # Derive from the profile (one capability per profile)
        want_transition = 'transition_dipole' in config.profile
        want_dipole = not want_transition

    grads = []

    # State dipoles: one force entry per state with derivatives off
    if want_dipole:
        for s in range(config.nstate):
            grads.append({
                'title': 'force',
                'target': s,
                'ciderivative': 'false',
            })

    # Transition dipoles: one nacme entry per transition pair with
    # derivatives off (interstate coupling type)
    if want_transition:
        pairs = [pair for pair in config.transition_pair if len(pair) == 2]
        # A config that never went through normalize still carries the unset
        # sentinel; the standalone convention it falls back to starts at
        # state 1, because [0, 0] is a state coupled to itself and BAGEL
        # computes no transition dipole for it (#448).
        if not pairs:
            pairs = [[0, i] for i in range(1, config.nstate)]
        for pair in pairs:
            grads.append({
                'title': 'nacme',
                'target': pair[0],
                'target2': pair[1],
                'nacmtype': 'interstate',
                'ciderivative': 'false',
            })

    forces = {
        'title': 'forces',
        'dipole': 'true',
        'grads': grads,
        'method': [method],
    }

    return forces


def _build_hessian_caspt2(config) -> dict:
    """Build the hessian block with XMS-CASPT2 method for frequency computation.

    BAGEL uses the 'hessian' block with a 'method' array containing
    the CASPT2/smith definition block. The 'target' field specifies which
    state to compute frequencies for.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Hessian JSON block with CASPT2 method.
    """
    target = config.target_states[0]

    hessian = {
        'title': 'hessian',
        'target': target,
        'method': [_build_caspt2_method(config)],
    }

    return hessian


def _build_optimize(config, workflow) -> dict:
    """Build the optimize block for geometry optimization.

    Follows the official BAGEL optimize template: the block carries
    the target state, the optimization type, the optimization step
    budget (maxit_opt) and the method definition array. The target
    state comes from opt_state_i (0-based); when opt_state_j differs,
    the block optimizes the minimum-energy conical intersection
    (opttype meci) and target2 carries opt_state_j.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph (selects the method level).

    Returns
    -------
    dict
        Optimize JSON block.
    """
    state_i = config.opt_state_i
    state_j = config.opt_state_j
    opttype = 'meci' if state_i != state_j else 'energy'

    if workflow.is_caspt2:
        method = _build_caspt2_method(config)
    else:
        method = _build_casscf_method(config)

    optimize = {
        'title': 'optimize',
        'opttype': opttype,
        'target': state_i,
        'maxiter': config.maxit_opt,
        # Step cap, spelled as BAGEL's own optimize template spells it
        # (test/ch2_sto3g_meci_opt.json carries "maxstep": 0.01). Without
        # it an oversized first step is unbounded: the measured XMS-CASPT2
        # gradient at the symmetric ethylene test geometry is ~1e3 a.u.,
        # and the unchecked step scattered the atoms over 1e3 Bohr, which
        # made BAGEL see six molecules and abort the internal-coordinate
        # transform. With the cap the same run takes four steps and
        # converges (-77.0810 -> -77.1325 Hartree, grad rms 0.041 -> 0.018).
        'maxstep': _OPT_MAX_STEP,
        'method': [method],
    }

    if opttype == 'meci':
        optimize['target2'] = state_j

    return optimize


def manage_archives(config, calcdir: str):
    """Manage archive file transfer between calculations.

    Copies .archive files from previous calculations to the current
    calculation directory, handling guess/converged archive naming.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.project
    guess = config.guess_orbital

    # Determine source for guess archive
    guess_src = guess if guess else f'{project}.archive'
    guess_dst = os.path.join(calcdir, f'{project}_guess.archive')

    # If converged archive exists in calcdir from previous step, use it as guess
    converged_in_calcdir = os.path.join(calcdir, f'{project}_converged.archive')
    if os.path.exists(converged_in_calcdir):
        shutil.copy2(converged_in_calcdir, guess_dst)
    elif os.path.exists(guess_src) and not os.path.exists(guess_dst):
        shutil.copy2(guess_src, guess_dst)
