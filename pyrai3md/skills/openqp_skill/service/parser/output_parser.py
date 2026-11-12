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

This module is the main parsing entry point for OpenQP results.

The skill always enables [properties]export=true with an empty
title, so OpenQP dumps results into fixed-name text files in the
calculation directory:
- energies          : state energies (one value per line)
- grad_<i>          : gradient of state index i (natom x 3)
- nacme / dcme      : NACME / DCME matrices (nac_type=dcm)
- nac_<i>_<j>       : NAC vector for the 1-based pair (i, j)
- opt.xyz           : optimized geometry (runtype=optimize)
- opt_status.txt    : optimization iteration table
- <project>.hess.json : hessian, frequencies, normal modes (hess)

The parser reads these files directly; the log is never parsed for
numerical data.

"""

import os
import json
import numpy as np

from .log_parser import parse_log


def _extract_velocities(config, calcdir):
    """Extract the final velocities (reserved fin_velocity channel).

    The openqp skill performs no skill-internal MD propagation that
    reports final velocities, so the fin_velocity return channel is
    reserved with an empty value; the uniform channel keeps every
    skill result contract aligned.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.

    Returns
    -------
    list
        Empty list (no velocity output).
    """
    return []


def _as_norm_modes(modes, natom):
    """Fold the engine's normal modes into the shape this skill declares.

    `norm_mode` is a 4-D ``(state, mode, atom, xyz)`` array in this skill's
    schema, but OpenQP's ``<project>.hess.json`` writes each mode as one flat
    row of ``3 * natom`` numbers -- measured in the test environment for
    `openqp_skill hf_frequency` (acetylene, 6 atoms): ``modes`` is
    ``(12, 18)`` while ``hessian`` is ``(18, 18)``. Handed on unfolded, that
    2-D array was read by the result printer as "12 states of 18 modes of a
    scalar", and `utilities/print_results._format_normal_modes` died on
    ``mode.shape[0]`` -- an ``IndexError: tuple index out of range`` that
    failed a run the engine had finished successfully (round
    tmp_20261011_085800, `hf_frequency_input` and `mrsf_frequency_input`).

    A shape that is not one of the recognized layouts yields an empty array:
    the printer then says nothing about modes rather than guessing a layout
    this engine does not write.
    """
    array = np.asarray(modes, dtype=float)
    if array.size == 0:
        return np.zeros((0,))
    if array.ndim == 4 and array.shape[2] == natom and array.shape[3] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 3 * natom:
        # (state, mode, 3 * atom)
        return array.reshape(array.shape[0], array.shape[1], natom, 3)
    if array.ndim == 3 and array.shape[1] == natom and array.shape[2] == 3:
        # (mode, atom, xyz) of one state
        return array.reshape(1, *array.shape)
    if array.ndim == 2 and array.shape[1] == 3 * natom:
        # (mode, 3 * atom), the layout the measured file uses
        return array.reshape(1, array.shape[0], natom, 3)
    return np.zeros((0,))


def parse_output(config, calcdir: str, workflow) -> dict:
    """Parse OpenQP output files based on the calculation profile.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary.

    Raises
    ------
    RuntimeError
        If the log check reports a fatal failure.
    """
    project = config.project
    log_path = os.path.join(calcdir, f'{project}.log')

    # Verify completion markers in the log
    status = parse_log(log_path, workflow)
    if not status['completed']:
        raise RuntimeError(status['error'])

    # Read the exported result files
    results = parse_export_files(config, calcdir, workflow)
    # The framework settles a skill-driven optimization from `opt_converge`
    # (`workflow/single.py::_finalize_optimize`); this skill's parser names
    # the flag `opt_converged`. Carrying only the latter made a truncated
    # OpenQP optimization read as completed in every framework leg (round
    # tmp_20261110_093321, `tddft_geometry_optimization`: the engine wrote
    # "Has Not Converged" and the run log closed `Status: completed`).
    results['opt_converge'] = status['opt_converged']
    results['opt_converged'] = status['opt_converged']

    return results


def parse_export_files(config, calcdir: str, workflow) -> dict:
    """Read the fixed-name export files produced by OpenQP.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Results dictionary with numpy arrays:
        energy, gradient, nac, soc, coordinates, and optional
        nacme/dcme matrices, hessian, frequencies, norm_mode.
    """
    nstate = config.nstate or 1
    natom = config.natom or 1
    method = config.method
    target_states = sorted(config.target_states)
    runtype = workflow.runtype

    results = {
        'energy': np.zeros(nstate),
        'gradient': np.zeros((nstate, natom, 3)),
        # Coupling channels start with no rows: a row exists only for a pair
        # this run exported (see the NAC/SOC readers below).
        'nac': np.empty((0, natom, 3)),
        'soc': np.empty(0),
        'coordinates': [],
        # Reserved fin_velocity return channel (no velocity output)
        'fin_velocity': _extract_velocities(config, calcdir),
    }

    # A run that solved both spin ladders publishes each manifold under its
    # own label and leaves the unlabelled names out entirely, so its
    # surfaces, gradients and couplings are read from a different set of
    # files than every other profile.  Nothing else applies to it: it takes
    # no optimization, frequency or NAC job (validate refuses a NAC
    # alongside spin-orbit coupling).
    if workflow.two_manifold:
        _fill_manifold_energies(config, calcdir, results)
        _fill_manifold_gradients(config, calcdir, workflow, results)
        if workflow.is_soc:
            _fill_socme(config, calcdir, results)
        return results

    # State energies: the export file "energies" lists one energy
    # per line.  For plain TDDFT / HF, energies[0] is the ground
    # state (S0); for MRSF-TDDFT, energies[0] is the internal
    # triplet reference state, so the physical S0 starts at index 1.
    # The file is mandatory: a missing energies export means the
    # OpenQP run aborted (e.g. a missing file2 orbital json on the
    # prop path), and a silent all-zero result would propagate into
    # the framework as a frozen trajectory with exit code 0.
    energy_file = os.path.join(calcdir, 'energies')
    if not os.path.exists(energy_file):
        raise RuntimeError(
            "OpenQP export file 'energies' not found in %s: the "
            "calculation did not complete (check the OpenQP log; a "
            "runtype=prop dcm run also requires the previous-step "
            "orbital file2)" % calcdir)
    energies = np.loadtxt(energy_file).reshape(-1)
    offset = 1 if method == 'mrsf' else 0
    # The file must carry a line for every state the run declares, past the
    # reference row the spin-flip convention puts in front. Clamping with min()
    # instead left the trailing rows at zero: the shape contract was satisfied,
    # and a crossing search pairing a real surface with one of those rows saw a
    # flat place rather than a missing calculation.
    needed = nstate + offset
    if len(energies) < needed:
        raise RuntimeError(
            "OpenQP export file 'energies' in %s carries %d line(s) while this "
            "run declares %d state(s)%s, so state(s) %d..%d have no energy to "
            "be read into and a zero row would be taken for a surface"
            % (calcdir, len(energies), nstate,
               ' plus the triplet reference row the mrsf convention skips'
               if offset else '',
               len(energies) - offset, nstate - 1))
    results['energy'][0] = energies[offset]
    if method != 'hf':
        results['energy'][1:] = energies[1 + offset:needed]

    # Gradients: hf ground state = grad_0; otherwise state k -> grad_{k+1}.
    # runtype=data (compute_data, the dynamics energy+gradient path)
    # exports the same grad_<i> files as runtype=grad.
    if runtype in ('grad', 'data'):
        if method == 'hf':
            grad = _load_matrix(calcdir, 'grad_0')
            if grad is None:
                raise RuntimeError(
                    "OpenQP export file 'grad_0' is missing or unreadable in "
                    "%s while the run asked for a gradient; the ground-state "
                    "row would otherwise stay zero and read as a flat surface"
                    % calcdir)
            results['gradient'][0] = grad
        else:
            for k in target_states:
                if not 0 <= k < nstate:
                    continue
                grad = _load_matrix(calcdir, f'grad_{k + 1}')
                if grad is None:
                    raise RuntimeError(
                        "OpenQP export file 'grad_%d' is missing or unreadable "
                        "in %s, so state %d of the pair this run follows has no "
                        "gradient; a zero-filled row keeps the shape contract "
                        "and would be stepped on as if it were a surface"
                        % (k + 1, calcdir, k))
                results['gradient'][k] = grad

    # Nonadiabatic couplings (runtype=prop)
    if runtype == 'prop':
        nac_type = config.nac_type
        if nac_type == 'dcm':
            # dcme matrix from MO overlap with the previous-step orbitals
            dcme = _load_matrix(calcdir, 'dcme')
            if dcme is not None:
                results['dcme'] = dcme
            nacme = _load_matrix(calcdir, 'nacme')
            if nacme is not None:
                results['nacme'] = nacme
        else:
            # NAC vectors per requested pair: framework pair [a, b]
            # (0-based) maps to the export file nac_{a+1}_{b+1}. A missing
            # export is refused by name: a zero row would read as "these two
            # surfaces are uncoupled", a claim the run never made.
            nac_list = _load_nac_pairs(
                calcdir, _selected_nac_pairs(config),
                lambda pair, name: (
                    'OpenQP: NAC pair %r was requested but no %s export '
                    'was written for it.' % (list(pair), name)))
            if nac_list:
                results['nac'] = np.array(nac_list)

    # NAC vector single point (runtype=nac): OpenQP supports only one
    # state pair per run, so only the first requested pair is read
    # from the export file nac_{a+1}_{b+1}.
    if runtype == 'nac':
        pairs = _selected_nac_pairs(config)
        if pairs:
            vec = _load_nac_pairs(
                calcdir, pairs[:1],
                lambda pair, name: (
                    'OpenQP: runtype nac asks for pair %r but no %s export '
                    'was written for it.' % (list(pair), name)))[0]
            results['nac'] = np.array([vec])

    # Geometry optimization: read the optimized geometry and the
    # final energy from the optimization status table.
    if runtype == 'optimize':
        opt_xyz = os.path.join(calcdir, 'opt.xyz')
        if os.path.exists(opt_xyz):
            optimized = _read_xyz(opt_xyz)
            results['coordinates'] = optimized
            # The framework's fin_coord channel is what updates the
            # system geometry after a skill-internal iteration. Without
            # it the run's opt.xyz/log keep the input structure while
            # the skill's own calc directory holds the optimized one
            # (round tmp_20261110_093321: the framework legs of
            # `tddft_geometry_optimization` recorded the initial
            # geometry as Step 1). `coordinates` stays because the
            # skill-level result contract still answers it.
            results['fin_coord'] = optimized

        final_energy = _read_opt_energy(os.path.join(calcdir, 'opt_status.txt'))
        if final_energy is not None:
            istate = target_states[0]
            if 0 <= istate < nstate:
                results['energy'][istate] = final_energy

    # Hessian / frequencies: read <project>.hess.json
    if runtype == 'hess':
        project = config.project
        hess_file = os.path.join(calcdir, f'{project}.hess.json')
        if os.path.exists(hess_file):
            with open(hess_file, 'r') as f:
                hess_data = json.load(f)

            results['hessian'] = np.array(hess_data.get('hessian', []))
            results['frequencies'] = np.array(hess_data.get('freqs', []))
            results['norm_mode'] = _as_norm_modes(
                hess_data.get('modes', []), natom)

            energy = hess_data.get('energy', None)
            istate = target_states[0]
            if energy is not None and 0 <= istate < nstate:
                results['energy'][istate] = energy

    return results


def _selected_nac_pairs(config):
    """The state pairs a run asks its couplings for.

    The declared target pairs win; a run that names none falls back to
    the pair map. Only well-formed two-entry pairs count.
    """
    target = [pair for pair in config.target_nac_pair if len(pair) == 2]
    mapped = [pair for pair in config.nac_pair_map if len(pair) == 2]
    return target if target else mapped


def _load_nac_pairs(calcdir, pairs, missing):
    """Load one `nac_{a+1}_{b+1}` export per requested pair.

    Parameters
    ----------
    calcdir : str
        Calculation directory.
    pairs : list of pair
        Framework pairs (0-based).
    missing : callable
        ``missing(pair, name)`` builds the refusal for an absent export;
        the two runtypes word it differently and both keep their words.

    Returns
    -------
    list of np.ndarray
        One vector per pair, in the requested order.

    Raises
    ------
    RuntimeError
        When a requested export is missing.
    """
    vectors = []
    for pair in pairs:
        a, b = sorted(pair)
        name = f'nac_{a + 1}_{b + 1}'
        vec = _load_matrix(calcdir, name)
        if vec is None:
            raise RuntimeError(missing(pair, name))
        vectors.append(vec)
    return vectors


def _ladder_counts(config):
    """State counts of the singlet and triplet ladders of a two-manifold run.

    Returns the sizes OpenQP converged the ladders to (what `socme`
    enumerates) and the number of states the input declares in each (what
    the returned arrays carry).  A ladder whose nstate_* is unset falls back
    on the declared rows, which is the same rule the input builder uses.

    Parameters
    ----------
    config : SkillConfig
        Configuration container (system submodel fields).

    Returns
    -------
    tuple of (int, int, int, int)
        (declared_singlets, declared_triplets, ladder_singlets, ladder_triplets)
    """
    spins = [int(value) for value in (config.state_spin or [])]
    declared_s = spins.count(0) if spins else config.nstate
    declared_t = spins.count(2) if spins else 0
    ladder_s = int(config.nstate_s or declared_s)
    ladder_t = int(config.nstate_t or declared_t)
    return declared_s, declared_t, ladder_s, ladder_t


def _read_ladder_rows(calcdir, name, needed, ladder):
    """Read one `energies_s` / `energies_t` file and check it is complete.

    Row 0 of each file is the ladder's internal MRSF reference, not a
    surface, so the states start at row 1.  A file that stops short leaves
    a state with no energy: the row would stay zero, a crossing search
    would pair a real surface with that zero, and nothing downstream could
    tell the missing calculation from a flat place.

    Parameters
    ----------
    calcdir : str
        Calculation directory.
    name : str
        Export file name.
    needed : int
        Rows required past the reference row.
    ladder : str
        Ladder name for the message ('singlet' or 'triplet').

    Returns
    -------
    np.ndarray
        Flat energy rows including the reference at index 0.
    """
    rows = _load_matrix(calcdir, name)
    if rows is None:
        raise RuntimeError(
            "OpenQP export file '%s' not found in %s: a run that solves both "
            "spin ladders publishes each one under its own label, so this "
            "calculation did not complete (check the OpenQP log)" % (name, calcdir))
    rows = np.asarray(rows).reshape(-1)
    if len(rows) < needed + 1:
        raise RuntimeError(
            "OpenQP export file '%s' in %s carries %d line(s) while this run "
            "declares %d %s state(s) plus the reference row, so state(s) "
            "%d..%d have no energy to be read into and a zero row would be "
            "taken for a surface"
            % (name, calcdir, len(rows), needed, ladder,
               max(len(rows) - 1, 0), needed - 1))
    return rows


def _fill_manifold_energies(config, calcdir, results) -> None:
    """Fill state energies from the labelled ladder energy exports.

    The framework's state list is ordered singlet-then-triplet (state_spin
    is sorted), so rows 0..n-1 of the singlet ladder land in the first n
    slots and the triplet ladder follows them.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    results : dict
        Results dictionary to fill in place.
    """
    declared_s, declared_t, _, _ = _ladder_counts(config)
    if declared_s:
        rows = _read_ladder_rows(calcdir, 'energies_s', declared_s, 'singlet')
        results['energy'][0:declared_s] = rows[1:1 + declared_s]
    if declared_t:
        rows = _read_ladder_rows(calcdir, 'energies_t', declared_t, 'triplet')
        results['energy'][declared_s:declared_s + declared_t] = \
            rows[1:1 + declared_t]


def _fill_manifold_gradients(config, calcdir, workflow, results) -> None:
    """Fill per-state gradients from the labelled ladder gradient exports.

    Each state k of the flat list maps to root k+1 of its own ladder, which
    is the name OpenQP wrote it under (`grad_s_2`, `grad_t_1`).  The
    dedicated runtype=soc driver computes couplings and energies only, so it
    has no gradient to read and the rows stay zero-filled.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.
    results : dict
        Results dictionary to fill in place.
    """
    if workflow.runtype == 'soc':
        return

    declared_s, _, _, _ = _ladder_counts(config)
    for state in sorted(config.target_states):
        if state < declared_s:
            name, label = f'grad_s_{state + 1}', 'singlet'
            index = state
        else:
            name, label = f'grad_t_{state - declared_s + 1}', 'triplet'
            index = state
        grad = _load_matrix(calcdir, name)
        if grad is None:
            raise RuntimeError(
                "OpenQP export file '%s' is missing or unreadable in %s, so "
                "state %d of the %s ladder this run follows has no gradient; "
                "a zero-filled row keeps the shape contract and would be "
                "stepped on as if it were a surface"
                % (name, calcdir, state, label))
        results['gradient'][index] = grad


def _fill_socme(config, calcdir, results) -> None:
    """Read the `socme` table and collapse it to one value per declared pair.

    The table holds one row per (singlet, triplet, Ms) triple with Ms over
    -1, 0, +1, and its first two columns index *within* each ladder from 0.
    Re and Im carry the phase of the Davidson amplitudes, which is a gauge
    freedom that flips between runs; the last column is the invariant
    magnitude, already in cm-1 (the header names the unit).  Following the
    molcas skill and PyRAI2MD, the three components of one state pair are
    combined into a single magnitude, and the rows are returned in
    `soc_pair_map` order because the integrators index them positionally.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    results : dict
        Results dictionary to fill in place.
    """
    if not config.soc_pair_map:
        return

    _, _, ladder_s, ladder_t = _ladder_counts(config)
    table = _load_matrix(calcdir, 'socme')
    if table is None:
        raise RuntimeError(
            "OpenQP export file 'socme' not found in %s while the run asked "
            "for spin-orbit coupling: the couplings would otherwise be read "
            "as zeros and a surface hop would decide on a coupling that was "
            "never computed (check the OpenQP log)" % calcdir)
    table = np.atleast_2d(table)
    expected = ladder_s * ladder_t * 3
    if len(table) != expected:
        raise RuntimeError(
            "OpenQP export file 'socme' in %s carries %d row(s) but the run "
            "solved %d singlet x %d triplet x 3 Ms = %d, so at least one "
            "coupling pair is missing or the ladders are not the sizes this "
            "input declares"
            % (calcdir, len(table), ladder_s, ladder_t, expected))

    collapsed = {}
    for row in table:
        ist, jst = int(row[0]), int(row[1])
        key = (ist, ladder_s + jst)
        collapsed[key] = collapsed.get(key, 0.0) + float(row[5]) ** 2

    values = []
    for pair in config.soc_pair_map:
        if len(pair) < 2:
            continue
        # The pair may be declared either way round; the singlet is the
        # index that falls inside the singlet ladder.
        first, second = int(pair[0]), int(pair[1])
        singlet, triplet = (first, second) if first < ladder_s else (second, first)
        key = (singlet, triplet)
        if key not in collapsed:
            raise RuntimeError(
                "OpenQP export file 'socme' in %s holds no row for the "
                "singlet-triplet pair %s this input declares (singlet index "
                "%d, triplet index %d of a %d x %d ladder), so that coupling "
                "would be recorded as a zero and used to decide a hop"
                % (calcdir, pair, singlet, triplet - ladder_s,
                   ladder_s, ladder_t))
        values.append(float(np.sqrt(collapsed[key])))

    results['soc'] = np.array(values, dtype=float)


def _load_matrix(calcdir: str, filename: str):
    """Load a whitespace-delimited numeric export file.

    Parameters
    ----------
    calcdir : str
        Calculation directory.
    filename : str
        Export file name (e.g. 'grad_1', 'dcme').

    Returns
    -------
    np.ndarray or None
        Parsed array, None if the file does not exist or is empty.
    """
    file_path = os.path.join(calcdir, filename)
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path)
    except ValueError:
        return None


def _read_xyz(xyz_path: str) -> list:
    """Read coordinates from an XYZ file.

    Parameters
    ----------
    xyz_path : str
        Path to the XYZ file.

    Returns
    -------
    list of list of float
        Parsed coordinates.
    """
    coord = []
    with open(xyz_path, 'r') as f:
        lines = f.read().splitlines()

    try:
        natom = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return coord

    for line in lines[2: 2 + natom]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                coord.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                pass

    return coord


def _read_opt_energy(status_path: str):
    """Read the final energy from the optimization status table.

    Parameters
    ----------
    status_path : str
        Path to opt_status.txt.

    Returns
    -------
    float or None
        Energy of the last optimization step, None if unavailable.
    """
    if not os.path.exists(status_path):
        return None

    with open(status_path, 'r') as f:
        lines = f.read().splitlines()

    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                continue

    return None


def format_results(results: dict, config) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : SkillConfig
        Configuration container.

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
