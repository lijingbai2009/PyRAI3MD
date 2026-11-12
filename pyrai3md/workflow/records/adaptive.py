"""
======================================================================
This module provides the adaptive-sampling selection and labelling
functions: read the retained history frames of finished trajectories,
judge their prediction deviations and physical validity, select the
geometries that go to the reference calculation, and turn the reference
results into training frames.

The functions are pure: they never write to a state or a file. Every
row they return carries the trajectory id it came from, the frame index
inside the screening window and the absolute MD step, so an error value
can always be traced back to the trajectory that produced it. A global
sort that loses that identity is not a reading this module produces.

Aug 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...kernel.logger import get_logger
from ...training_data import TrainingData
from ...utilities.bonds import bond_lib
from .deviation import DEVIATION_KEYS

_LOGGER = get_logger(__name__)

# (deviation key, &adaptive maximum keyword, &adaptive minimum keyword).
SCREEN_RULES = (
    ('err_energy', 'maxenergy', 'minenergy'),
    ('err_grad', 'maxgrad', 'mingrad'),
    ('err_nac', 'maxnac', 'minnac'),
    ('err_soc', 'maxsoc', 'minsoc'),
)

# The label convention of a training frame. The reference backend reports
# the framework's own units, so no conversion is applied here.
TRAINING_UNITS = {
    'energy': 'hartree',
    'gradient': 'hartree/bohr',
    'nac': 'hartree/bohr',
    'soc': 'cm-1',
}


def screen_thresholds(adaptive):
    """Read the screening thresholds from an &adaptive section.

    Parameters
    ----------
    adaptive : object or None
        Section carrying max*/min* keywords.

    Returns
    -------
    tuple(dict, dict)
        (maxima, minima), keyed by the deviation key.
    """
    maxima, minima = {}, {}
    if adaptive is None:
        return maxima, minima
    for key, maximum_key, minimum_key in SCREEN_RULES:
        maximum = getattr(adaptive, maximum_key, None)
        minimum = getattr(adaptive, minimum_key, None)
        if maximum is not None:
            maxima[key] = float(maximum)
        if minimum is not None:
            minima[key] = float(minimum)
    return maxima, minima


def history_frames(outcome, window=None):
    """Read the retained history frames of one trajectory outcome.

    A local outcome is a PyRAIState and exposes get_history(); a remote
    outcome is a Result whose 'history' came back from result.json. The
    latter is an object ndarray after Result's construction, so it is
    converted before slicing (an ndarray never answers a truth test).

    Parameters
    ----------
    outcome : object
        PyRAIState, Result, plain mapping or failure marker.
    window : int or None
        Keep only the newest N frames; None keeps all stored frames.

    Returns
    -------
    list of dict
        Chronological history frames; [] for a failed or empty outcome.
    """
    if outcome is None:
        return []
    if isinstance(outcome, dict) and outcome.get('adaptive_status') == 'error':
        return []

    getter = getattr(outcome, 'get_history', None)
    if callable(getter):
        try:
            frames = getter(window)
        except (AttributeError, TypeError, ValueError) as error:
            _LOGGER.warning(
                'Adaptive history read from get_history(%s) failed (%s); '
                'this iteration proceeds with no history frames', window,
                error)
            return []
    else:
        history = None
        if isinstance(outcome, dict):
            history = outcome.get('history')
        else:
            try:
                history = outcome.history
            except AttributeError:
                history = None
        if history is None:
            return []
        if isinstance(history, np.ndarray):
            history = history.tolist()
        try:
            frames = list(history)
        except TypeError:
            return []
        if window is not None and int(window) > 0:
            frames = frames[-int(window):]

    return [frame for frame in frames if isinstance(frame, dict)]


def frame_deviations(frame):
    """Read the deviation scalars of one history frame.

    A missing or non-numeric value is reported as 0.0, the tolerant
    reception rule of records/deviation.py. An equality with a threshold
    never triggers.
    """
    results = frame.get('results') or {}
    deviations = {}
    for key in DEVIATION_KEYS:
        value = results.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        deviations[key] = number if np.isfinite(number) else 0.0
    return deviations


def frame_state_spins(frame):
    """The 2S value of every electronic state in one history frame.

    ``state_spin`` is the primary source. When it is absent, the flat
    ``ci``/``spin`` pair expands to the same per-state list; a nested
    pair (multi-center state) uses its first center.

    Parameters
    ----------
    frame : dict
        History frame.

    Returns
    -------
    list of int or None
        One 2S value per state, in state order.
    """
    system = frame.get('system') or {}
    nstate = int(system.get('nstate') or 0)
    raw = system.get('state_spin') or []
    try:
        spins = [int(value) for value in raw]
    except (TypeError, ValueError):
        spins = []
    if nstate and len(spins) == nstate:
        return spins

    counts = system.get('ci') or []
    values = system.get('spin') or []
    if not counts or not values or len(counts) != len(values):
        return None
    if any(isinstance(value, (list, tuple)) for value in counts):
        counts = counts[0]
        values = values[0]
    expanded = []
    try:
        for count, spin in zip(counts, values):
            expanded.extend([int(spin)] * int(count))
    except (TypeError, ValueError):
        return None
    if nstate and len(expanded) != nstate:
        return None
    return expanded


def energy_order_bad(energies, state_spins, tolerance):
    """Whether a same-spin level sequence is not ascending.

    States are grouped by their 2S value and checked in state-index
    order inside each group: a later state lower than an earlier one by
    more than ``tolerance`` (Hartree) is a negative gap.

    Parameters
    ----------
    energies : sequence or None
        State energies in Hartree.
    state_spins : sequence or None
        One 2S value per state, from frame_state_spins().
    tolerance : float
        Allowed negative gap in Hartree.

    Returns
    -------
    bool
        True when a same-spin group carries a negative gap.
    """
    if energies is None or state_spins is None:
        return False
    values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if values.size != len(state_spins):
        return False
    groups = {}
    order = []
    for index, spin in enumerate(state_spins):
        if spin not in groups:
            groups[spin] = []
            order.append(spin)
        groups[spin].append(index)
    for spin in order:
        indices = groups[spin]
        for left, right in zip(indices, indices[1:]):
            if values[right] - values[left] < -float(tolerance):
                return True
    return False


def distance_ok(atoms, coordinates, bond_scale=0.7):
    """Legacy physicality filter: no atom pair closer than 0.7 bond length.

    The lookup keeps the i<j order because bond_lib is not a symmetric
    table (H-B and B-H are separate entries).

    Parameters
    ----------
    atoms : sequence or None
        Element labels.
    coordinates : array-like or None
        (natom, 3) positions in Angstrom.
    bond_scale : float
        Fraction of the typical bond length below which a pair is
        rejected.

    Returns
    -------
    bool
        False when a pair is too close; True otherwise.
    """
    if atoms is None or coordinates is None:
        return True
    labels = list(atoms)
    positions = np.asarray(coordinates, dtype=np.float64)
    if positions.shape != (len(labels), 3):
        return True
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            distance = float(np.linalg.norm(
                positions[left] - positions[right]))
            if distance < float(bond_scale) * bond_lib(
                    labels[left], labels[right]):
                return False
    return True


def _alternatives(results):
    """The comparison-model channels of one frame, or None.

    The serving skill writes `alternative_<channel>` plus
    `alternative_model_ids` only when predict_mode>0. mode=0 leaves the
    keys at None, which this helper reads as "no comparison data". The
    judgment of not_requested/available/missing belongs to the adaptive
    workflow, not to this reader.
    """
    channels = {}
    for key, value in (results or {}).items():
        if not str(key).startswith('alternative_'):
            continue
        name = str(key)[len('alternative_'):]
        channels[name] = value
    if not channels:
        return None
    if all(value is None for value in channels.values()):
        return None
    model_ids = channels.pop('model_ids', None)
    payload = dict(channels)
    payload['model_ids'] = model_ids
    return payload


def _frame_row(traj_id, frame_index, frame, maxima, minima,
               check_energy_order, gap_tolerance,
               include_analysis=False):
    """Build one traceable screening row from a history frame."""
    system = frame.get('system') or {}
    results = frame.get('results') or {}
    errors = frame_deviations(frame)
    order_bad = bool(check_energy_order and energy_order_bad(
        results.get('energy'), frame_state_spins(frame), gap_tolerance))

    triggered = order_bad
    candidate = triggered
    for key, _maximum_key, _minimum_key in SCREEN_RULES:
        if key in maxima and errors.get(key, 0.0) > maxima[key]:
            triggered = True
        if key in minima and errors.get(key, 0.0) > minima[key]:
            candidate = True

    try:
        step = int(frame.get('step')) if frame.get('step') is not None else None
    except (TypeError, ValueError):
        step = None
    cell = system.get('cell')
    pbc = system.get('pbc')
    row = {
        'traj_id': int(traj_id),
        'frame_index': int(frame_index),
        'step': step,
        'coordinates': system.get('coordinates'),
        'image': system.get('image'),
        'cell': list(cell) if cell is not None else None,
        'pbc': [bool(value) for value in pbc] if pbc is not None else None,
        'errors': {key: errors.get(key, 0.0) for key in DEVIATION_KEYS},
        'flags': {'population_stop': False, 'order_bad': order_bad,
                  'geometry_bad': False,
                  'trigger': bool(triggered),
                  'candidate': bool(candidate)},
        'selected': False,
        'reason': '',
    }
    if include_analysis:
        # The analysis layer needs the unbiased energies, the active state
        # and the velocities of the frame. They are added only when an
        # analysis switch is on, so a default run's rows and status_iter
        # stay byte-identical to the pre-analysis build.
        row['energy'] = (results.get('unbiased_energy')
                         if results.get('unbiased_energy') is not None
                         else results.get('energy'))
        row['current_state'] = system.get('current_state')
        row['velocity'] = system.get('velocities')
        row['bias_status'] = results.get('bias_status')
        for key in ('udd_energy', 'gap_bias_energy', 'gap_excess',
                    'gap_velocity_scale', 'gap_force_switched', 'bias_work',
                    'udd_force_ratio', 'udd_state_switches'):
            if results.get(key) is not None:
                row[key] = results[key]
        alternatives = _alternatives(results)
        if alternatives is not None:
            row['alternatives'] = alternatives
    return row


def _has_order_data(frame):
    """Whether a frame carries what the same-spin order check reads."""
    results = frame.get('results') or {}
    return (results.get('energy') is not None
            and frame_state_spins(frame) is not None)


def screen_trajectory(traj_id, frames, maxima, minima, maxsample,
                      check_population=True, stop_code=0,
                      check_energy_order=True, gap_tolerance=1e-6,
                      atoms=None, bond_scale=0.7, mdstep=None,
                      include_analysis=False, exploration=False,
                      population_expected=True):
    """Screen one trajectory's retained frames.

    The trigger is a frame above a maximum threshold, a physically bad
    frame, or the structure the md loop stopped on. The candidate pool is
    the union of the frames above the minimum thresholds, the bad frames,
    and - when the run ended with the ill-population code and the caller
    collects it - the stopping frame. Legacy ordering is preserved: the
    pool is ordered by frame index from newest to oldest, then the
    distance filter and the per-trajectory maxsample cap are applied in
    that order.

    Parameters
    ----------
    traj_id : int
        Planning-order trajectory index; stamped on every row.
    frames : list of dict
        Chronological history window.
    maxima, minima : dict
        Deviation key to threshold.
    maxsample : int
        Maximum selected geometries for this trajectory.
    check_population : bool
        Whether the trajectory's stopping structure is collected when md
        ended with the ill-population code. Population is md's judgment,
        not the screen's: this switch only decides whether that structure
        goes to the reference calculation.
    stop_code : int
        The md stop code of this trajectory (0 normal, 1 ill population,
        2 dissociation, 3 prediction error).
    check_energy_order, gap_tolerance : bool, float
        Same-spin level-order rule.
    atoms : sequence or None
        Element labels for the distance filter.
    bond_scale : float
        Distance filter scale (legacy 0.7).
    mdstep : int or None
        Planned trajectory length, used for the completed flag.

    Returns
    -------
    dict
        Traceable per-trajectory screening record.
    """
    rows = [
        _frame_row(traj_id, index, frame, maxima, minima,
                   check_energy_order, gap_tolerance,
                   include_analysis=include_analysis)
        for index, frame in enumerate(frames)
    ]

    # The structure the run stopped on is the last frame it recorded. md
    # judged its population and ended the trajectory there; the screen only
    # decides whether that structure is collected, so no frame is re-judged.
    population_stop = False
    if rows and int(stop_code) == 1 and check_population:
        row = rows[-1]
        row['flags']['population_stop'] = True
        row['flags']['trigger'] = True
        row['flags']['candidate'] = True
        population_stop = True

    triggered_any = any(row['flags']['trigger'] for row in rows)
    candidate_indices = set()
    if triggered_any:
        for _key, _maximum_key, _minimum_key in SCREEN_RULES:
            key = _key
            if key not in minima:
                continue
            ranked = [row for row in rows
                      if row['errors'].get(key, 0.0) > minima[key]]
            ranked.sort(key=lambda row: (-row['errors'][key],
                                         row['frame_index']))
            candidate_indices.update(row['frame_index'] for row in ranked)
        candidate_indices.update(
            row['frame_index'] for row in rows
            if row['flags']['population_stop'] or row['flags']['order_bad'])

    ordered = sorted(candidate_indices, reverse=True)
    # The physical filter is applied to the whole candidate pool before the
    # per-trajectory cap: a bad geometry stays a bad geometry even when the
    # cap is already full, and the Bad count must not depend on the order
    # the cap happens to fill in.
    for frame_index in ordered:
        row = rows[frame_index]
        if not distance_ok(atoms, row.get('coordinates'), bond_scale):
            row['flags']['geometry_bad'] = True
    selected = []
    for frame_index in ordered:
        row = rows[frame_index]
        if row['flags'].get('geometry_bad'):
            row['reason'] = 'distance filter'
            continue
        if int(maxsample) >= 0 and len(selected) >= int(maxsample):
            row['reason'] = 'maxsample cap'
            continue
        row['selected'] = True
        row['reason'] = 'selected'
        selected.append(row)

    if not triggered_any:
        for row in rows:
            row['reason'] = 'not triggered'

    last_step = None
    last_state = None
    for row in reversed(rows):
        if row.get('step') is not None:
            last_step = row['step']
            frame = frames[row['frame_index']]
            system = frame.get('system') or {}
            last_state = system.get('current_state')
            if last_state is not None:
                last_state = int(last_state)
            break
    completed = bool(
        mdstep is not None and last_step is not None
        and int(last_step) >= int(mdstep))

    max_errors = {}
    for key in DEVIATION_KEYS:
        values = [row['errors'].get(key, 0.0) for row in rows]
        max_errors[key] = float(max(values)) if values else 0.0
    if not frames:
        population_status = ('off' if not check_population
                             else 'not_applicable')
        order_status = 'off' if not check_energy_order else 'not_applicable'
    else:
        if not check_population:
            population_status = 'off'
        elif not population_expected:
            population_status = 'not_applicable'
        else:
            population_status = ('stopped' if population_stop
                                 else 'not_stopped')
        if not check_energy_order:
            order_status = 'off'
        elif any(not _has_order_data(frames[row['frame_index']])
                 for row in rows):
            order_status = 'missing'
        else:
            order_status = 'ok'
    n_selected = sum(1 for row in rows if row['selected'])
    n_candidates = sum(
        1 for row in rows if row['flags'].get('candidate'))
    n_discarded = sum(
        1 for row in rows
        if row['flags'].get('candidate') and not row['selected'])
    n_bad = sum(
        1 for row in rows if row['flags'].get('geometry_bad'))
    return {
        'traj_id': int(traj_id),
        'completed': completed,
        'exploration': bool(exploration),
        'population_status': population_status,
        'order_status': order_status,
        'last_step': last_step,
        'last_state': last_state,
        'triggered': triggered_any,
        'max_errors': max_errors,
        'n_sampled': n_candidates,
        'n_window': len(rows),
        'n_candidates': n_candidates,
        'n_selected': n_selected,
        'n_discarded': n_discarded,
        'n_bad': n_bad,
        'frames': rows,
        'selected': selected,
    }


def outcome_stop_code(outcome):
    """The md stop code a trajectory outcome carries (0 when absent).

    Both spellings of an outcome are read: a local PyRAIState exposes
    `runtime`, a remote one is the mapping `result.json` carried. An error
    marker has no trajectory to ask, but the child entry stamps the code it
    saw on the marker, so a failed calculation still reports 4.
    """
    if outcome is None:
        return 0
    if isinstance(outcome, dict):
        if outcome.get('adaptive_status') == 'error':
            value = outcome.get('stop_code', 0)
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0
        runtime = outcome.get('runtime')
    else:
        runtime = getattr(outcome, 'runtime', None)
    if runtime is None:
        return 0
    if isinstance(runtime, dict):
        value = runtime.get('stop_code', 0)
    else:
        value = getattr(runtime, 'stop_code', 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def screen_iteration(outcomes, settings):
    """Screen every trajectory of one iteration.

    Parameters
    ----------
    outcomes : list
        One outcome per trajectory, in planning order (PyRAIState,
        Result, plain mapping or failure marker).
    settings : dict
        'record_step', 'maxima', 'minima', 'maxsample',
        'check_population', 'check_energy_order', 'gap_tolerance',
        'atoms', 'bond_scale', 'mdstep'.

    Returns
    -------
    dict
        'per_trajectory', 'selected' (still carrying traj_id),
        'all_completed', 'triggered', the per-iteration 'max_errors' of every
        deviation key, and the sampled/selected/discarded frame counts.
    """
    per_trajectory = []
    selected = []
    for traj_id, outcome in enumerate(outcomes):
        frames = history_frames(outcome, settings.get('record_step'))
        result = screen_trajectory(
            traj_id, frames,
            settings.get('maxima') or {},
            settings.get('minima') or {},
            int(settings.get('maxsample', 1)),
            check_population=bool(settings.get('check_population', 1)),
            stop_code=outcome_stop_code(outcome),
            check_energy_order=bool(settings.get('check_energy_order', 1)),
            gap_tolerance=float(settings.get('gap_tolerance', 1e-6)),
            atoms=settings.get('atoms'),
            bond_scale=float(settings.get('bond_scale', 0.7)),
            mdstep=settings.get('mdstep'),
            include_analysis=bool(settings.get('analysis_active')),
            exploration=bool(settings.get('exploration')),
            population_expected=bool(
                settings.get('population_expected', True)),
        )
        per_trajectory.append(result)
        selected.extend(result['selected'])
    max_errors = {}
    for key in DEVIATION_KEYS:
        values = [result['max_errors'][key] for result in per_trajectory
                  if key in result.get('max_errors', {})]
        if values:
            max_errors[key] = max(values)
    return {
        'per_trajectory': per_trajectory,
        'selected': selected,
        'exploration': bool(settings.get('exploration')),
        'population_expected': bool(
            settings.get('population_expected', True)),
        'order_missing': [
            result['traj_id'] for result in per_trajectory
            if result.get('order_status') == 'missing'],
        'all_completed': bool(per_trajectory) and all(
            result['completed'] for result in per_trajectory),
        'triggered': any(result['triggered'] for result in per_trajectory),
        'n_trajectory': len(per_trajectory),
        'n_completed': sum(1 for result in per_trajectory
                           if result['completed']),
        'n_passed': sum(1 for result in per_trajectory
                        if not result['triggered']),
        'n_sampled': sum(result['n_sampled'] for result in per_trajectory),
        'n_selected': sum(result['n_selected'] for result in per_trajectory),
        'n_discarded': sum(result['n_discarded']
                           for result in per_trajectory),
        'n_bad': sum(result['n_bad'] for result in per_trajectory),
        'max_errors': max_errors,
    }


def frame_contract(dataset):
    """Describe the frame layout a new frame must match.

    Parameters
    ----------
    dataset : TrainingData
        Accumulated (unexpanded) training dataset.

    Returns
    -------
    dict
        Counts, channel presence and auxiliary frame fields.
    """
    return {
        'natom': int(dataset.natom),
        'nframe': int(dataset.nframe),
        'nstate': int(dataset.nstate),
        'nnac': int(dataset.nnac),
        'nsoc': int(dataset.nsoc),
        'npair': int(dataset.npair),
        'has_energy': dataset.energy is not None,
        'has_grad': dataset.grad is not None,
        'has_nac': dataset.nac is not None,
        'has_soc': dataset.soc is not None,
        'has_cell': dataset.cell is not None,
        'has_pbc': dataset.pbc is not None,
        'has_charge': dataset.charge is not None,
        'has_transition_dipole': (
            dataset.transition_dipole is not None
            or dataset.transition_dipole_norm is not None
            or dataset.osc_strength is not None),
        'has_group_id': dataset.group_id is not None,
        'has_permutation_id': dataset.permutation_id is not None,
        'symbols': (dataset.symbols[0] if dataset.symbols is not None
                    else None),
        'unit': dict(dataset.unit or {}),
    }


def _result_mapping(outcome):
    """Read one reference outcome as a plain result mapping."""
    if outcome is None:
        return None
    if isinstance(outcome, dict):
        if outcome.get('adaptive_status') == 'error':
            return None
        return outcome
    result = getattr(outcome, 'result', None)
    if result is not None:
        try:
            return dict(result)
        except (TypeError, ValueError):
            pass
    pmd = getattr(outcome, 'pmd', None)
    if pmd is not None:
        results = getattr(pmd, 'results', None)
        if results is not None:
            try:
                return dict(results)
            except (TypeError, ValueError):
                pass
    try:
        return dict(outcome)
    except (TypeError, ValueError):
        return None


def _as_array(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.size else None


def _take_region(array, indices, contract, label):
    """Project a whole-system per-atom array onto the ML region.

    A composite reference runs on the full system while the dataset
    frame holds only the trainable atoms. Arrays already sized to the
    region pass through; a whole-system array is sliced by the same
    indices the dataset uses.
    """
    if array is None:
        return None
    indices = [int(index) for index in (indices or [])]
    if not indices:
        return array
    if array.ndim < 3:
        return array
    if array.shape[1] == contract['natom']:
        return array
    if array.shape[1] == len(indices):
        return array
    try:
        return array[:, indices, :]
    except (IndexError, TypeError):
        raise ValueError(
            "%s carries %d atoms but the region names %d."
            % (label, array.shape[1], len(indices)))


def _labels_from_channels(channels, contract, indices=None):
    """Read the planned label channels carried by a composite result."""
    energy = _as_array(channels.get('energy'))
    if energy is None:
        return None, 'reference result carries no labelled energy'
    energy = energy.reshape(-1)
    if energy.size != contract['nstate']:
        return None, ('energy carries %d states but the dataset declares %d'
                      % (energy.size, contract['nstate']))

    grad = _as_array(channels.get('gradient'))
    if grad is None:
        return None, 'reference result carries no labelled gradient'
    if grad.ndim == 4 and grad.shape[0] == 1:
        grad = grad[0]
    if grad.ndim == 2 and grad.shape == (contract['natom'], 3):
        grad = grad.reshape(1, *grad.shape)
    if grad.ndim != 3 or grad.shape[-1] != 3:
        return None, 'labelled gradient is not (nstate, natom, 3)'
    try:
        grad = _take_region(grad, indices, contract, 'labelled gradient')
    except ValueError as error:
        return None, str(error)
    if grad.shape[0] != contract['nstate'] or grad.shape[1] != contract['natom']:
        return None, ('gradient shape %r does not match nstate=%d, natom=%d'
                      % (grad.shape, contract['nstate'], contract['natom']))

    nac = None
    if contract['has_nac']:
        nac = _as_array(channels.get('nac'))
        if nac is None:
            return None, 'dataset carries NAC labels but the result has none'
        if nac.ndim == 4 and nac.shape[0] == 1:
            nac = nac[0]
        if nac.ndim != 3 or nac.shape[-1] != 3:
            return None, 'nac is not (nnac, natom, 3)'
        try:
            nac = _take_region(nac, indices, contract, 'labelled nac')
        except ValueError as error:
            return None, str(error)
        if nac.shape[0] != contract['nnac'] or nac.shape[1] != contract['natom']:
            return None, ('nac shape %r does not match nnac=%d, natom=%d'
                          % (nac.shape, contract['nnac'], contract['natom']))

    soc = None
    if contract['has_soc']:
        soc = _as_array(channels.get('soc'))
        if soc is None:
            return None, 'dataset carries SOC labels but the result has none'
        soc = soc.reshape(-1)
        if soc.size != contract['nsoc']:
            return None, ('soc carries %d values but the dataset declares %d'
                          % (soc.size, contract['nsoc']))

    return {'energy': energy, 'grad': grad, 'nac': nac, 'soc': soc}, ''


def labels_from_outcome(outcome, contract, indices=None):
    """Turn one reference outcome into one training-frame label set.

    Parameters
    ----------
    outcome : object
        Reference child outcome (PyRAIState, Result or failure marker).
    contract : dict
        Frame contract from frame_contract().
    indices : sequence of int or None
        Trainable atom indices. A composite reference returns
        whole-system arrays; the label channels are sliced to these
        atoms before the frame contract is checked. None keeps the
        historical whole-result behaviour.

    Returns
    -------
    tuple(dict or None, str)
        (labels, reason). labels carries 'energy' (nstate,), 'grad'
        (nstate, natom, 3) and, when the dataset carries them, 'nac'
        (nnac, natom, 3) and 'soc' (nsoc,). reason is empty on success.
    """
    result = _result_mapping(outcome)
    if result is None:
        return None, 'reference outcome carries no result'

    channels = result.get('label_channels')
    if isinstance(channels, dict) and channels:
        return _labels_from_channels(channels, contract, indices)

    energy = _as_array(result.get('energy'))
    if energy is None:
        return None, 'reference result carries no energy'
    energy = energy.reshape(-1)
    if energy.size != contract['nstate']:
        return None, ('energy carries %d states but the dataset declares %d'
                      % (energy.size, contract['nstate']))

    grad_value = result.get('gradient')
    if grad_value is None:
        grad_value = result.get('grad')
    grad = _as_array(grad_value)
    if grad is None:
        return None, 'reference result carries no gradient'
    if grad.ndim == 4 and grad.shape[0] == 1:
        grad = grad[0]
    if grad.ndim == 2 and grad.shape == (contract['natom'], 3):
        grad = grad.reshape(1, *grad.shape)
    if grad.ndim != 3 or grad.shape[-1] != 3:
        return None, 'gradient is not (nstate, natom, 3)'
    try:
        grad = _take_region(grad, indices, contract, 'gradient')
    except ValueError as error:
        return None, str(error)
    if grad.shape[0] != contract['nstate'] or grad.shape[1] != contract['natom']:
        return None, ('gradient shape %r does not match nstate=%d, natom=%d'
                      % (grad.shape, contract['nstate'], contract['natom']))

    nac = None
    if contract['has_nac']:
        nac_value = result.get('nac')
        nac = _as_array(nac_value)
        if nac is None:
            return None, 'dataset carries NAC labels but the result has none'
        if nac.ndim == 3 and nac.shape[-1] == 3:
            pass
        elif nac.ndim == 4 and nac.shape[0] == 1:
            nac = nac[0]
        else:
            return None, 'nac is not (nnac, natom, 3)'
        try:
            nac = _take_region(nac, indices, contract, 'nac')
        except ValueError as error:
            return None, str(error)
        if nac.shape[0] != contract['nnac'] or nac.shape[1] != contract['natom']:
            return None, ('nac shape %r does not match nnac=%d, natom=%d'
                          % (nac.shape, contract['nnac'], contract['natom']))

    soc = None
    if contract['has_soc']:
        soc_value = result.get('soc')
        soc = _as_array(soc_value)
        if soc is None:
            return None, 'dataset carries SOC labels but the result has none'
        soc = soc.reshape(-1)
        if soc.size != contract['nsoc']:
            return None, ('soc carries %d values but the dataset declares %d'
                          % (soc.size, contract['nsoc']))

    return {'energy': energy, 'grad': grad, 'nac': nac, 'soc': soc}, ''


def build_frame_dataset(rows, atoms, unit=None, source='adaptive'):
    """Build one TrainingData batch from labelled selected rows.

    Parameters
    ----------
    rows : list of dict
        One row per frame: 'coordinates', 'energy', 'grad', optional
        'nac', 'soc', 'cell', 'pbc', 'charge', 'group_id',
        'permutation_id'.
    atoms : sequence or None
        Element labels of the system.
    unit : dict or None
        Unit block to carry; the canonical block is used when None.
    source : str
        Provenance label.

    Returns
    -------
    TrainingData
        The new frames.
    """
    if not rows:
        raise ValueError("adaptive dataset needs at least one frame")

    def stack(key, dtype=np.float64):
        if any(row.get(key) is None for row in rows):
            return None
        return np.asarray([row[key] for row in rows], dtype=dtype)

    xyz = np.asarray([row['coordinates'] for row in rows], dtype=np.float64)
    symbols = None
    if atoms is not None:
        labels = list(atoms)
        symbols = np.asarray([labels for _ in rows], dtype=object)
    energy = np.asarray([row['energy'] for row in rows], dtype=np.float64)
    grad = np.asarray([row['grad'] for row in rows], dtype=np.float64)
    nac = stack('nac')
    soc = stack('soc')
    charge = stack('charge')
    cell = stack('cell')
    pbc = stack('pbc', dtype=bool)
    group_id = stack('group_id', dtype=int)
    permutation_id = stack('permutation_id', dtype=int)
    return TrainingData(
        xyz=xyz,
        symbols=symbols,
        energy=energy,
        grad=grad,
        nac=nac,
        soc=soc,
        charge=charge,
        cell=cell,
        pbc=pbc,
        group_id=group_id,
        permutation_id=permutation_id,
        unit=dict(unit or TRAINING_UNITS),
        source=source,
    )


# Robust z-score cutoff of the training-data quality gate. The scale is
# the median absolute deviation, so a heavy tail of ordinary frames does
# not widen the band the way a standard deviation would.
Z_OUTLIER = 3.5

# Consistency constant that puts the MAD on the scale of a normal sigma.
MAD_SCALE = 1.4826


def energy_outlier_flags(reference, new, threshold=Z_OUTLIER):
    """Flag new frames whose QM energy lies outside the dataset's band.

    One electronic state at a time: the reference step and scale come
    from the frames the dataset held before this round, and each new
    frame is judged by its worst state. A frame is flagged when
    ``|E - median| / (1.4826 * MAD)`` exceeds the threshold.

    This reads the labels, not a prediction: a reference run that
    failed (an SCF that did not converge, a swapped state) writes an
    energy the surface never had, and the rest of its labels are wrong
    with it. A degenerate state is handled by widening the scale to its
    standard deviation, and by equality when even that is zero.

    Parameters
    ----------
    reference : array or None
        ``(n_ref, nstate)`` energies of the frames that were in the
        dataset before this round, in Hartree.
    new : array or None
        ``(n_new, nstate)`` energies of the frames this round added.
    threshold : float
        Robust z above which a frame counts as an outlier.

    Returns
    -------
    tuple
        ``(flags, detail)``: a boolean array of length ``n_new``, and
        the audit record (medians, scales, per-frame z, its kind of
        scale). ``detail['testable']`` is False with a reason when the
        two sets cannot be compared at all.
    """
    detail = {
        'testable': False,
        'reason': '',
        'threshold': float(threshold),
        'n_reference': 0,
        'n_new': 0,
        'median': [],
        'mad': [],
        'scale': [],
        'scale_kind': [],
        'z': [],
        'max_abs_z': [],
        'flagged': [],
    }
    if reference is None or new is None:
        detail['reason'] = 'the dataset carries no energy labels'
        return np.zeros(0, dtype=bool), detail
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(new, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cand.ndim == 1:
        cand = cand.reshape(-1, 1)
    detail['n_reference'] = int(ref.shape[0])
    detail['n_new'] = int(cand.shape[0])
    empty = np.zeros(int(cand.shape[0]) if cand.ndim else 0, dtype=bool)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape[1] != cand.shape[1]:
        detail['reason'] = (
            'the reference energies are %r and the new frames are %r'
            % (ref.shape, cand.shape))
        return empty, detail
    if ref.shape[0] < 3:
        detail['reason'] = (
            '%d reference frame(s) are too few for a robust energy scale'
            % ref.shape[0])
        return empty, detail
    if not np.all(np.isfinite(ref)):
        detail['reason'] = 'a reference energy is not finite'
        return empty, detail
    if not np.all(np.isfinite(cand)):
        detail['reason'] = 'a new frame carries a non-finite energy'
        return empty, detail

    median = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - median), axis=0)
    scale = MAD_SCALE * mad
    kinds = []
    for state in range(ref.shape[1]):
        if scale[state] > 0:
            kinds.append('mad')
            continue
        spread = float(np.std(ref[:, state]))
        if spread > 0:
            scale[state] = spread
            kinds.append('std')
        else:
            scale[state] = 0.0
            kinds.append('zero')

    z = np.zeros_like(cand)
    for state in range(ref.shape[1]):
        if scale[state] > 0:
            z[:, state] = np.abs(cand[:, state] - median[state]) / scale[state]
        else:
            z[:, state] = np.where(cand[:, state] != median[state],
                                   np.inf, 0.0)
    worst = z.max(axis=1)
    flagged = [int(index) for index, value in enumerate(worst)
               if value > float(threshold)]

    # A degenerate state has an unbounded z; the record keeps it as null
    # because the summary is written with allow_nan=False.
    def recorded(value):
        value = float(value)
        return value if np.isfinite(value) else None

    detail.update({
        'testable': True,
        'median': [float(value) for value in median],
        'mad': [float(value) for value in mad],
        'scale': [float(value) for value in scale],
        'scale_kind': kinds,
        'z': [[recorded(value) for value in row] for row in z],
        'max_abs_z': [recorded(value) for value in worst],
        'flagged': flagged,
    })
    flags = np.zeros(int(cand.shape[0]), dtype=bool)
    flags[flagged] = True
    return flags, detail