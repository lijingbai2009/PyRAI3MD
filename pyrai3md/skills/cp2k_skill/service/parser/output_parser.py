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

This module is the main parsing entry point for CP2K results. It selects
the appropriate parser based on the calculation profile and keeps the
state-resolved shape contract shared with the other skills. A row that the
engine did not answer for is a failure, not a zero: a zero-filled row has the
right shape, costs a step of nothing, and a search reading it converges on a
surface that was never computed.

CP2K prints atomic forces; the framework contract wants gradients,
so every parsed force block is negated here (gradient = -force).

"""

import os
import warnings

import numpy as np

from .log_parser import (parse_log, parse_force_xyz, parse_tdfrc,
                         parse_xyz, parse_xyz_cell,
                         parse_stress_tensor)
from . import state_map

# 1 eV in Hartree (used to convert TDDFPT excitation energies)
EV_TO_HARTREE = 1.0 / 27.211386245988

# Conversion factors from a CP2K stress unit label to GPa, the unit
# declared by the skill manifest (units.stress). The framework applies
# the single GPa -> Hartree/Bohr^3 factor centrally, so the parser only
# normalizes to GPa. An empty/unknown label is treated as GPa (the
# CP2K default).
_STRESS_LABEL_TO_GPA = {
    '': 1.0,
    'gpa': 1.0,
    'bar': 1.0e-4,
    'atm': 1.01325e-4,
    'pa': 1.0e-9,
    'ev/angstrom^3': 160.2176634,
    'ev/a^3': 160.2176634,
    'hartree/bohr^3': 29421.0265,
}


def _extract_velocities(config, calcdir):
    """Extract the final velocities (reserved fin_velocity channel).

    The cp2k skill performs no skill-internal MD propagation that
    reports final velocities, so the fin_velocity return channel is
    reserved with an empty value; the uniform channel keeps every
    skill result contract aligned.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.

    Returns
    -------
    list
        Empty list (no velocity output).
    """
    return []


def parse_output(config, calcdir: str, workflow) -> dict:
    """Parse CP2K output files based on the calculation profile.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary with numpy arrays:
        energy, gradient, nac, soc, dipole, transition_dipole,
        excitation, osc_strength, and optional fin_coord /
        opt_converge fields.
    """
    project = config.project
    nstate = config.nstate
    natom = config.natom
    target_states = sorted(config.target_states)

    # Transition properties are extracted in transition_pair order
    # (npair, 3), keeping the pair-driven contract shared with the
    # other skills; no row at all when no pair was declared.
    transition_pairs = [list(pair) for pair in config.transition_pair
                        if len(pair) == 2]
    # No row for a pair nobody declared: the `else 1` used to hand back one
    # zero transition dipole / oscillator strength for a run that asked for no
    # transition property at all (#448).
    npair = len(transition_pairs)

    results = {
        'energy': np.zeros(nstate),
        'gradient': np.zeros((nstate, natom, 3)),
        # This skill declares no coupling channel, so the two carry no rows
        # at all; a zero row would read as "uncoupled" to a caller that never
        # asked and cannot tell the difference.
        'nac': np.empty((0, natom, 3)),
        'soc': np.empty(0),
        'dipole': np.zeros((nstate, 3)),
        'transition_dipole': np.zeros((npair, 3)),
        'excitation': np.zeros(nstate),
        'osc_strength': np.zeros(npair),
        # Whole-system stress tensor (nstate, 3, 3); zero-filled and
        # only populated when config.stress requests it (non-blocking).
        'stress_tensor': np.zeros((nstate, 3, 3)),
        # Reserved fin_velocity return channel (no velocity output)
        'fin_velocity': _extract_velocities(config, calcdir),
    }

    profile = workflow.profile
    log_path = os.path.join(calcdir, f'{project}.log')

    # Parse the log for all profiles (missing sections stay zero)
    data = parse_log(log_path, config)

    # Ground-state energy: the last 'Total energy:' of the run
    if data['s0_energy'] is not None and workflow.method == 'dft':
        results['energy'][0] = data['s0_energy']
    elif data['s0_energy'] is None and workflow.has_step('energy'):
        raise RuntimeError(
            'CP2K: the run asked for an energy but the completed log '
            'carries no "Total energy:" line.')

    # Ground-state gradient: force file {project}-force-1_0.xyz,
    # gradient = -force (forces come in hartree/bohr). A TDDFT run asks for
    # this file too, and CP2K writes it beside the excited-state blocks in the
    # same calculation (cp2k_demo/td_grad/azo-1.inp:125-129 and :146-148, whose
    # directory holds both equil-force-1_0.xyz and equil-td_forces-1_0.tdfrc):
    # a crossing between a ground state and an excited one needs a gradient for
    # both, and leaving row 0 at zero would hand the search a flat surface it
    # cannot tell apart from a converged one.
    if workflow.has_step('gradient'):
        force_path = os.path.join(calcdir, f'{project}-force-1_0.xyz')
        forces = parse_force_xyz(force_path, natom)
        if forces.size == 0:
            raise RuntimeError(
                f'CP2K {profile} finished but no atomic forces were '
                f'found in {force_path} (calcdir: {calcdir}); check '
                f'the calculation output.'
            )
        results['gradient'][0] = -forces

    # Excited states: energies and spectral data from the TDDFPT
    # table, gradients from the tdfrc force blocks.
    if workflow.method == 'tddft':
        _pack_tddft_results(results, data, config, workflow, calcdir)

    # Geometry optimization: the relaxed structure comes from the
    # last frame of {project}-pos-1.xyz (fin_coord channel,
    # Angstrom); missing final geometry is a contract violation.
    if workflow.is_optimization:
        _pack_final_geometry(results, data, config, workflow, calcdir)

    # Standalone MD: the last trajectory frame becomes fin_coord
    # and the final log energy fills the active state.
    if workflow.is_md:
        _pack_final_geometry(results, data, config, workflow, calcdir)

    # Whole-system stress tensor (non-blocking): read the CP2K
    # 'STRESS TENSOR' block when the stress route is active; a missing
    # block leaves the zero-filled contract intact. The num route
    # asks the engine for no tensor at all -- its stress is assembled
    # from the strain single points by service/stress/, which owns the
    # result array from here on.
    if (getattr(config, 'stress', False)
            and str(getattr(config, 'stress_method',
                            'analytical')).lower() != 'num'):
        _pack_stress(results, data, config, workflow, calcdir)

    return results


# Channels a two-manifold split re-answers, and how each is stitched.
_ROW_KEYED = ('energy', 'excitation', 'gradient', 'dipole', 'stress_tensor')
_PAIR_KEYED = ('transition_dipole', 'osc_strength')

# The ground state is computed by every job that runs (each needs its own SCF
# reference before it can answer excited states), so the two runs report it
# twice. They must agree: measured on an A/B of the same input differing only in
# `RKS_TRIPLETS`, both printed `Total energy: -13.56482679723119`. The tolerance
# is the printed precision, not a physical allowance - a real difference here
# means the two runs did not share a reference after all.
_GROUND_TOLERANCE = 1.0e-8


def merge_job_results(jobs: list, results: list, config) -> dict:
    """Stitch the per-job results of one evaluation into the framework's arrays.

    A request that names one spin manifold is answered by one job and comes back
    untouched, which is what keeps every shipped case the identity test for this
    path.

    With two jobs the rows are owned by spin: `energy`, `excitation`, `gradient`,
    `dipole` and `stress_tensor` are `(nstate, ...)` and each row is taken from the
    job that solved its manifold, so no value is ever averaged across two runs and
    no row can be filled from the wrong surface. Row 0 belongs to no manifold -
    both jobs computed it - so the two readings are compared at the printed
    precision and refused if they differ, rather than one being quietly preferred.

    The pair-keyed channels (`transition_dipole`, `osc_strength`) run in
    `transition_pair` order, and a job prints a value only for the pairs its own
    manifold is in, so the two arrays are disjoint and are summed; a position both
    jobs filled with different numbers means the split was not disjoint and is
    refused. Everything else (the returned geometry, the convergence verdict, the
    frequencies) has one owner per run and the jobs must agree on it.

    Parameters
    ----------
    jobs : list of Job
        The jobs as planned by `service/jobs.py`, each carrying the manifold it
        solved (`two_s`) or None when no split happened.
    results : list of dict
        The parsed result dict of each job, in the same order.
    config : SkillConfig
        The ORIGINAL configuration (not a variant): the arrays are sized and the
        rows are labelled by the request the framework made.

    Returns
    -------
    dict
        One result dict per the framework's contract.

    Raises
    ------
    RuntimeError
        When two jobs disagree on the ground state or on a single-owner channel,
        or when both filled the same transition-pair slot with different values.
    """
    if len(results) == 1:
        return results[0]

    merged = dict(results[0])
    owner = {job.two_s: index for index, job in enumerate(jobs)
             if job.two_s is not None}

    for key in _ROW_KEYED:
        if key not in merged:
            continue
        base = np.asarray(merged[key], dtype=np.float64)
        rows = base.shape[0]
        for row in range(1, rows):
            two_s = state_map.spin_of_row(config, row)
            source = results[owner[two_s]][key]
            base[row] = np.asarray(source, dtype=np.float64)[row]
        merged[key] = base

    # The ground row is answered by every job; keep the first, after making the
    # rest say the same thing.
    for key in _ROW_KEYED:
        if key not in merged:
            continue
        stacked = np.stack([np.asarray(job_result[key], dtype=np.float64)[0]
                           for job_result in results])
        spread = float(np.max(np.abs(stacked - stacked[0])))
        if spread > _GROUND_TOLERANCE:
            # Each run's own reading of row 0, printed as it is shaped: a gradient
            # row is an array, and formatting it as a scalar is what broke this
            # branch the first time a split disagreed on the atomic half.
            readings = ', '.join(
                f'run {index}: '
                + (f'{float(row):.10f}' if np.ndim(row) == 0
                   else f'{np.asarray(row).ravel()[:3].tolist()}...')
                for index, row in enumerate(stacked))
            raise RuntimeError(
                f'CP2K {config.calculation.profile}: the {len(results)} runs that '
                f'solve this request together disagree on row 0 of {key!r} by '
                f'{spread:.3e} (their own readings - {readings}). '
                f'Row 0 is the ground state, which both runs compute from the same '
                f'reference, so a difference is not a choice between two surfaces - '
                f'the runs are not describing one geometry and one SCF')

    for key in _PAIR_KEYED:
        if key not in merged:
            continue
        arrays = [np.asarray(job_result[key], dtype=np.float64)
                  for job_result in results]
        total = np.zeros_like(arrays[0])
        for array in arrays:
            both = (np.abs(total) > 0.0) & (np.abs(array) > 0.0)
            clash = both & (np.abs(total - array) > 0.0)
            if np.any(clash):
                raise RuntimeError(
                    f'CP2K {config.calculation.profile}: {key!r} was filled by two '
                    f'runs at {int(np.count_nonzero(clash))} position(s) with '
                    f'different values, so the manifolds were not disjoint - each '
                    f'pair belongs to one manifold and one run should print it')
            total = total + array
        merged[key] = total

    for key in sorted(set(merged) - set(_ROW_KEYED) - set(_PAIR_KEYED)):
        first = merged[key]
        for index, job_result in enumerate(results[1:], start=1):
            if key not in job_result:
                continue
            other = job_result[key]
            if _same_value(first, other):
                continue
            raise RuntimeError(
                f'CP2K {config.calculation.profile}: channel {key!r} is produced by '
                f'one run per evaluation, but run {index} printed a different value '
                f'({other!r} against {first!r}); a two-manifold request may only '
                f'split channels that are labelled by state rows or pairs')

    return merged


def _same_value(one, other) -> bool:
    """Whether two channel values say the same thing, array or scalar."""
    try:
        return bool(np.allclose(np.asarray(one, dtype=np.float64),
                                np.asarray(other, dtype=np.float64),
                                rtol=0.0, atol=_GROUND_TOLERANCE))
    except (TypeError, ValueError):
        return one == other


def _pack_stress(results: dict, data: dict, config, workflow,
                 calcdir: str):
    """Pack the whole-system stress tensor from the CP2K stress file.

    The ground-state (SCF) stress is read from the dedicated print file
    ``{project}_stress_tensor.dat`` written by the builder's
    ``&STRESS_TENSOR ... FILENAME`` keyword, falling back to the main
    ``{project}.log`` when that file is absent (e.g. a CP2K build that
    ignored FILENAME). The tensor is flipped to the tensile-positive
    convention (CP2K prints compressive-positive) and normalized to GPa
    (the manifest-declared units.stress); the framework converts
    GPa -> Hartree/Bohr^3 centrally. It fills the S0 block; the states
    above it stay zero-filled, because the engine prints no tensor for
    an excited state (TDDFPT has no analytical stress route, and CP2K
    offers no numerical one for a TDDFT run either -- measured on the
    deployment). The populated excited-state tensor is the num route's
    output, assembled from the
    strain single points by service/stress/ and written over this array
    before the results leave the backend.

    Parameters
    ----------
    results : dict
        Results dictionary to fill in place.
    data : dict
        Raw parsed log data.
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.
    """
    project = config.project
    stress_path = os.path.join(calcdir, f'{project}_stress_tensor.dat')
    if not os.path.exists(stress_path):
        # Fallback: scan the main log block (older runs / no FILENAME).
        stress_path = os.path.join(calcdir, f'{project}.log')
    matrix, unit = parse_stress_tensor(stress_path)
    if matrix is None:
        if bool(getattr(config, 'stress', False)):
            raise RuntimeError(
                'CP2K: the run asked for the stress tensor but the completed '
                'calculation carries no stress block; a zero tensor is not '
                'a pressure.')
        return

    factor = _STRESS_LABEL_TO_GPA.get(str(unit or '').lower().strip(), 1.0)
    # CP2K native stress is compressive-positive; the framework contract
    # wants the tensile-positive potential stress (sigma = -P).
    results['stress_tensor'][0] = -matrix * factor


def _pack_tddft_results(results: dict, data: dict, config, workflow,
                        calcdir: str):
    """Pack TDDFPT energies, spectral data, and excited-state forces.

    Parameters
    ----------
    results : dict
        Results dictionary to fill in place.
    data : dict
        Raw parsed log data.
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.
    """
    nstate = config.nstate
    natom = config.natom
    project = config.project

    if data['s0_energy'] is None:
        if workflow.has_step('energy'):
            raise RuntimeError(
                'CP2K: the run asked for an energy but the completed log '
                'carries no "Total energy:" line.')
        # No energy was requested: keep the historic zero baseline the
        # excited-state rows are referenced to.
        s0 = 0.0
    else:
        s0 = data['s0_energy']
        results['energy'][0] = s0

    # Excitation table rows carry CP2K's own (manifold, number) pair. The
    # number is not a framework row: with RKS_TRIPLETS on, CP2K prints one table
    # per spin manifold and restarts the numbering in each, so the number 1
    # exists twice over two different surfaces. `state_map` turns the pair into a
    # row, which is what makes the collision unnameable here: a triplet row can no
    # longer land on a singlet's row, because the banner decides.
    #
    # A row seen twice is a later pass over the same state and overwrites. An
    # optimization re-solves and re-prints once per geometry, and the numbers
    # change between passes (measured: the framework-mode tddft_gradient case of
    # round tmp_20261014_142736 printed its table twice), so a disagreement is not
    # evidence of a mislabel here - only a row nobody answered is.
    filled = {}
    manifold = state_map.job_manifold(config)
    for row in data['tddft_states']:
        state = state_map.resolve_log_row(config, row['multiplicity'],
                                          row['state'])
        if state is None:
            continue
        # A run that was planned for one manifold may not fill a row of the
        # other. This is checked where the claim is made rather than left for the
        # merge to overwrite: if the two runs' logs were staged under swapped
        # projects, the singlet run would happily print a triplet's energy into
        # the triplet's row and the merge would take the row from the run that
        # owns it - a wrong number that no channel shows, which is the shape of
        # the defect this file exists to close.
        if (manifold is not None
                and state_map.spin_of_row(config, state) != manifold):
            raise RuntimeError(
                f'CP2K {workflow.profile}: this run solves the 2S={manifold} '
                f'manifold (nsing_root {config.nsing_root}, ntrip_root '
                f'{config.ntrip_root}), but its log carries a '
                f'multiplicity-{row["multiplicity"]} table whose row maps onto '
                f'state {state}, a 2S={state_map.spin_of_row(config, state)} '
                f'surface. Two runs of one split request answer different rows; '
                f'the table that got here belongs to the other run')
        filled[state] = row
    for state, row in sorted(filled.items()):
        exc = row['excitation'] * EV_TO_HARTREE
        results['energy'][state] = s0 + exc
        results['excitation'][state] = row['excitation']

    # Every excited state the run declared has to have been answered. A row left
    # at zero satisfies the shape contract and describes no surface, and the
    # crossing search reading a pair of those would converge on a place that was
    # never computed.
    unanswered = [state for state in state_map.requested_rows(config)
                  if state not in filled]
    if unanswered:
        raise RuntimeError(
            f'CP2K {workflow.profile} printed no excitation-table row for state '
            f'row(s) {unanswered} of the {config.nstate} this run declares '
            f'(state_spin {list(config.state_spin)}); their energies would '
            f'otherwise be packed as zeros'
        )

    # Transition dipoles / oscillator strengths in transition_pair
    # order; only ground-to-state pairs carry computed values.
    pairs = [pair for pair in config.transition_pair if len(pair) == 2]
    for k, pair in enumerate(pairs):
        if k >= results['transition_dipole'].shape[0]:
            break
        a, b = pair
        if a != 0:
            continue
        row = filled.get(b)
        if row is not None:
            results['transition_dipole'][k] = row['transition_dipole']
            results['osc_strength'][k] = row['osc_strength']

    # Excited-state gradients: every tdfrc block is attributed to the row it
    # names, by the (number, energy) pair the header prints, because the header
    # carries no spin and a manifold restart makes a bare number ambiguous.
    if not workflow.has_step('gradient'):
        return

    tdfrc_path = os.path.join(
        calcdir, f'{project}-td_forces-1_0.tdfrc')
    blocks = parse_tdfrc(tdfrc_path, natom)
    requested = state_map.requested_rows(config)
    if blocks and not requested:
        raise RuntimeError(
            f'CP2K {workflow.profile} printed {len(blocks)} excited-state force '
            f'blocks while this run requests no excited state '
            f'(state_spin {list(config.state_spin)}); the input and the output '
            f'disagree about how many surfaces exist')
    assignment, by_order, order = state_map.attribute_blocks(
        config, data['tddft_states'], blocks)
    if order != state_map.LIST_ORDER:
        # The header numbers the engine printed follow a different counting
        # than the LIST this skill writes, so a run that still lined up did so
        # because the energies carried it. Say which one the engine uses: that
        # is the reading the test round came for.
        warnings.warn(
            f'CP2K {workflow.profile}: the tdfrc headers count excited states '
            f'per spin manifold (`{order}`) while the &PRINT &FORCES LIST this '
            f'skill writes continues across manifolds '
            f'(`{state_map.LIST_ORDER}`); the blocks were tied to their roots by '
            f'energy, and `state_map.LIST_ORDER` should be set to the printed '
            f'reading.',
            stacklevel=2,
        )
    if by_order:
        # Degenerate roots were separated only by request order; say so where a
        # reader of the run can see it, rather than letting the row look labelled.
        warnings.warn(
            f'CP2K {workflow.profile}: state row(s) {by_order} were separated '
            f'from a degenerate partner by the order the roots were requested, '
            f'because the force-block header prints a number and an energy but '
            f'no spin, and at this geometry those agree for both surfaces. '
            f'Diagnostic only; the assignment is not contradicted by the '
            f'excitation table.',
            stacklevel=2,
        )
    for row, index in assignment.items():
        if 0 < row < nstate:
            results['gradient'][row] = -blocks[index]['forces']


def _pack_final_geometry(results: dict, data: dict, config, workflow,
                         calcdir: str):
    """Pack the final geometry from the trajectory file.

    GEO_OPT and standalone MD both write {project}-pos-1.xyz; the
    last frame goes through the framework fin_coord channel
    (natom, 3) in Angstrom.

    Parameters
    ----------
    results : dict
        Results dictionary to fill in place.
    data : dict
        Raw parsed log data.
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.
    """
    natom = config.natom
    nstate = config.nstate
    target_states = sorted(config.target_states)

    pos_path = os.path.join(calcdir, f'{config.project}-pos-1.xyz')
    final_coordinates = parse_xyz(pos_path)
    if not final_coordinates:
        raise RuntimeError(
            f'CP2K {workflow.profile} finished but no final geometry '
            f'was found in {pos_path} (calcdir: {calcdir}).'
        )
    fin_coord = np.array(final_coordinates)
    if len(fin_coord) != natom:
        raise ValueError(
            f'CP2K final geometry in {pos_path} has {len(fin_coord)} '
            f'atoms, but the system has {natom}.'
        )
    results['fin_coord'] = fin_coord

    # The same frame that gave the coordinates gives the cell, when the
    # run was periodic: 'fix'-style cell evolution aside, this is how a
    # CP2K-internal iteration tells the framework what box its returned
    # coordinates actually live in. A fixed-cell run reports the cell it
    # was given, which the framework notices and leaves alone.
    lattice = parse_xyz_cell(pos_path)
    if lattice:
        results['fin_cell'] = [float(value) for row in lattice
                               for value in row]

    if data['s0_energy'] is not None:
        istate = target_states[0] if target_states else 0
        if 0 <= istate < nstate:
            results['energy'][istate] = data['s0_energy']

    if workflow.is_optimization:
        results['opt_converge'] = data['opt_converged']


def format_results(results: dict, config) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Formatted results with numpy arrays converted to lists.
    """
    formatted = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        else:
            formatted[key] = value

    return formatted
