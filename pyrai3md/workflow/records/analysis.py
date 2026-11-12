"""
======================================================================
This module provides the adaptive-sampling analysis layer: the switches,
the shared frame container, and the pure analysis functions that turn a
round's screening rows into a report.

The layer is default-closed. `analysis_plan()` reads the `&adaptive`
switches and `analysis_active()` is the single predicate that gates every
new key and every new file, so an adaptive run with the switches off is
bit-for-bit the run it was before this module existed.

Every function here is pure: rows in, rows or a report out, no state and
no file writes. The adaptive workflow owns the one report file of a round
(`analysis_iter-<k>.json`) and the log block.

Three outcomes are distinguished, because they mean different things:

- `ok`: the analysis ran and produced a value.
- `not_applicable`: the data exists but the method's precondition is not
  met (too few neighbours, an unsupported scope, a periodic dataset).
  The record carries a countable `reason` and is skipped, never fatal.
- `missing`: the run asked for the quantity but the trajectory did not
  produce it. This is a defect; the caller warns and, when every
  trajectory is missing, stops the round.

Comparison-model data (predict_mode>0) adds `not_requested`: the run did
not ask for it, so its absence is expected and only informational.

Nov 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.bonds import bond_lib
from ...utilities.units import ANGSTROM_TO_BOHR

ANALYSIS_VERSION = 1


class AnalysisFrame(dict):
    """Versioned payload of one round's analysis.

    The report file is one frame: identity and plan at the top, the
    per-trajectory rows under `trajectories`, and the round-level
    summaries under `summary`. The version travels with the data so a
    reader can refuse a layout it does not know.
    """

    version = ANALYSIS_VERSION

    def __init__(self, **sections):
        super().__init__(version=ANALYSIS_VERSION, **sections)

OK = 'ok'
NOT_APPLICABLE = 'not_applicable'
MISSING = 'missing'
NOT_REQUESTED = 'not_requested'

# A descriptor of pairwise distances is O(natom^2); above this size the
# coverage/PES analyses report not_applicable instead of allocating it.
MAX_DESCRIPTOR_ATOMS = 200

# Cap the reference frames sampled from a growing dataset. The analyses
# are neighbourhood queries, so a bounded sample keeps a round's cost flat.
MAX_DATASET_FRAMES = 200


def analysis_plan(adaptive):
    """Read the `&adaptive` analysis switches into one plan mapping."""
    return {
        'pes': int(getattr(adaptive, 'analyze_pes', 0) or 0) == 1,
        'gap': str(getattr(adaptive, 'analyze_gap', 'off') or 'off').lower(),
        'trace': int(getattr(adaptive, 'analyze_trace', 0) or 0) == 1,
        'coverage': int(getattr(adaptive, 'analyze_coverage', 0) or 0) == 1,
        'window': int(getattr(adaptive, 'analyze_window', 3) or 3),
        'local_tolerance': float(
            getattr(adaptive, 'analyze_local_tolerance', 0.0) or 0.0),
        'global_tolerance': float(
            getattr(adaptive, 'analyze_global_tolerance', 0.0) or 0.0),
        'explore_udd': int(getattr(adaptive, 'explore_udd', 0) or 0) == 1,
        'explore_gap': int(getattr(adaptive, 'explore_gap', 0) or 0) == 1,
        'predict_mode': int(getattr(adaptive, 'predict_mode', 0) or 0),
    }


def analysis_active(plan):
    """The single predicate that gates every new key and file."""
    return bool(plan.get('pes') or plan.get('trace')
                or plan.get('coverage') or plan.get('gap', 'off') != 'off'
                or plan.get('explore_udd') or plan.get('explore_gap')
                or plan.get('predict_mode', 0) > 0)


def trace_spec(plan):
    """What the md child should collect for the analyses that are on.

    `gap` is the scope the gap analysis needs; the boolean entries are the
    scalar traces the trajectory statistics need. Exploration alone (UDD
    or gap driving) does not order a trace: it reads the prediction rows,
    not the scalar history. A plan with nothing to collect returns None.
    """
    if not (plan.get('trace') or plan.get('gap', 'off') != 'off'):
        return None
    gap = plan.get('gap', 'off')
    return {
        'gap': None if gap == 'off' else gap,
        'force': bool(plan.get('trace') or gap != 'off'),
        'population': bool(plan.get('trace')),
        'energy_drift': bool(plan.get('trace')),
        'distance': bool(plan.get('trace')),
    }


def trace_payload(outcome):
    """The analysis_trace mapping of one trajectory outcome, or None.

    A local outcome is a PyRAIState carrying RuntimeState.analysis_trace; a
    remote outcome is a Result whose result.json carried the same field
    under `runtime`. Both are accepted here so the caller does not branch
    on the dispatch route.
    """
    if outcome is None:
        return None
    if isinstance(outcome, dict) and outcome.get('adaptive_status') == 'error':
        return None
    runtime = getattr(outcome, 'runtime', None)
    if runtime is not None:
        payload = getattr(runtime, 'analysis_trace', None)
        if payload is not None:
            return payload
    getter = getattr(outcome, 'get', None)
    if callable(getter):
        try:
            runtime = getter('runtime')
        except (TypeError, ValueError):
            runtime = None
        if isinstance(runtime, dict):
            return runtime.get('analysis_trace')
    return None


def trace_rows(outcome):
    """The trace rows of one outcome; [] when the trajectory has none."""
    payload = trace_payload(outcome)
    if not isinstance(payload, dict):
        return []
    rows = payload.get('rows')
    return list(rows) if isinstance(rows, list) else []


def _status(status, reason, **extra):
    record = {'status': status, 'reason': reason}
    record.update(extra)
    return record


def _state_energies(row):
    value = row.get('energy')
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array if array.size else None


def gap_pairs(nstate, scope):
    """The (lower, upper) state pairs a scope names, 0-based."""
    nstate = int(nstate)
    if scope == 's0s1':
        return [(0, 1)] if nstate >= 2 else []
    if scope == 'adjacent':
        return [(index, index + 1) for index in range(nstate - 1)]
    if scope == 'all':
        return [(lower, upper) for lower in range(nstate)
                for upper in range(lower + 1, nstate)]
    return []


def summarize_gap(rows, scope, threshold):
    """Minimum gap, first crossing and the fraction below the threshold.

    The energies come from the unbiased prediction channel stored on the
    screening row. A row whose frame carried no state energies is counted
    as missing data; a round with no rows at all is not applicable.
    """
    if scope == 'off':
        return _status(NOT_APPLICABLE, 'analyze_gap is off')
    if not rows:
        return _status(NOT_APPLICABLE, 'the trajectory retained no frames')
    nstate = 0
    for row in rows:
        energies = _state_energies(row)
        if energies is not None:
            nstate = max(nstate, int(energies.size))
    pairs = gap_pairs(nstate, scope)
    if not pairs:
        if nstate == 0:
            return _status(
                MISSING,
                'no frame carried state energies; the gap analysis needs them')
        return _status(
            NOT_APPLICABLE,
            'scope %r needs at least two states; the frames carry %d'
            % (scope, nstate))
    values = []
    missing = 0
    for row in rows:
        energies = _state_energies(row)
        if energies is None or energies.size < nstate:
            missing += 1
            continue
        for lower, upper in pairs:
            values.append((row.get('step'), lower, upper,
                           float(energies[upper] - energies[lower])))
    if not values:
        status = MISSING if missing else NOT_APPLICABLE
        return _status(
            status,
            'no frame carried %d state energies (missing=%d)'
            % (nstate, missing))
    lowest = min(values, key=lambda item: item[3])
    crossing = next((item for item in values if item[3] <= threshold), None)
    below = sum(1 for item in values if item[3] <= threshold)
    return _status(
        OK, '',
        scope=scope, threshold=float(threshold), n_values=len(values),
        n_missing_frames=missing,
        min_gap=lowest[3], min_pair=[lowest[1], lowest[2]],
        min_step=lowest[0],
        first_below_threshold=(
            None if crossing is None else
            {'step': crossing[0], 'pair': [crossing[1], crossing[2]],
             'gap': crossing[3]}),
        fraction_below_threshold=float(below) / len(values))


def summarize_trace(rows):
    """Scalar trajectory statistics from the md-side trace rows."""
    if not rows:
        return _status(NOT_APPLICABLE, 'the trajectory retained no trace rows')
    gaps = [row['gap'] for row in rows
            if isinstance(row.get('gap'), dict) and row['gap'].get('min') is not None]
    forces = [float(row['max_force']) for row in rows
              if row.get('max_force') is not None]
    drifts = [float(row['energy_drift']) for row in rows
              if row.get('energy_drift') is not None]
    populations = [value for row in rows
                   for value in (row.get('population') or [])
                   if isinstance(value, (int, float))]
    if not gaps and not forces and not drifts:
        return _status(MISSING, 'trace rows carry no scalar trace')
    report = _status(OK, '', n_rows=len(rows))
    if gaps:
        lowest = min(gaps, key=lambda item: item['min'])
        report['min_gap'] = float(lowest['min'])
        report['min_gap_pair'] = list(lowest.get('pair') or [])
        report['min_gap_step'] = lowest.get('step')
    if forces:
        report['max_force'] = max(forces)
    if drifts:
        report['max_energy_drift'] = max(abs(value) for value in drifts)
    distances = [float(row['min_distance']) for row in rows
                 if row.get('min_distance') is not None]
    if distances:
        report['min_distance'] = min(distances)
    statuses = [row.get('min_distance_status') for row in rows
                if row.get('min_distance_status')]
    if statuses and not distances:
        report['min_distance_status'] = statuses[-1]
    if populations:
        report['min_population'] = min(populations)
    return report


def _descriptor(coordinates):
    """Rotation/translation-invariant pairwise-distance descriptor."""
    positions = np.asarray(coordinates, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        return None
    natom = positions.shape[0]
    if natom > MAX_DESCRIPTOR_ATOMS:
        return None
    left, right = np.triu_indices(natom, 1)
    return np.linalg.norm(positions[left] - positions[right], axis=1)


def _sample_frames(count):
    if count <= MAX_DATASET_FRAMES:
        return np.arange(count)
    return np.linspace(0, count - 1, MAX_DATASET_FRAMES).astype(int)


def summarize_coverage(rows, dataset, average_length=None, mdstep=None):
    """Descriptor coverage of this round's candidates against the QM data."""
    if dataset is None or getattr(dataset, 'xyz', None) is None:
        return _status(NOT_APPLICABLE, 'the dataset carries no geometries')
    candidates = [row for row in rows if row.get('coordinates') is not None]
    if not candidates:
        return _status(NOT_APPLICABLE, 'the round selected no candidates')
    try:
        index = _sample_frames(int(dataset.nframe))
        reference = np.asarray([_descriptor(dataset.xyz[position])
                                for position in index], dtype=object)
    except (AttributeError, TypeError, ValueError) as error:
        return _status(NOT_APPLICABLE,
                       'the dataset descriptor failed: %s' % error)
    if any(item is None for item in reference):
        return _status(
            NOT_APPLICABLE,
            'the descriptor needs at most %d atoms; dataset has %d'
            % (MAX_DESCRIPTOR_ATOMS, int(getattr(dataset, 'natom', 0) or 0)))
    reference = np.asarray(list(reference), dtype=np.float64)
    distances = []
    for row in candidates:
        descriptor = _descriptor(row.get('coordinates'))
        if descriptor is None or descriptor.size != reference.shape[1]:
            continue
        distances.append(float(np.min(np.linalg.norm(
            reference - descriptor, axis=1))))
    if not distances:
        return _status(
            NOT_APPLICABLE,
            'candidate descriptors needed at most %d atoms; none matched'
            % MAX_DESCRIPTOR_ATOMS)
    distances = np.asarray(distances, dtype=np.float64)
    # Use the median nearest-neighbour spacing of the reference as the
    # "covered" radius, so the fraction is not tied to an arbitrary unit.
    if reference.shape[0] > 1:
        pairwise = np.linalg.norm(
            reference[:, None, :] - reference[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        radius = float(np.median(np.min(pairwise, axis=1)))
    else:
        radius = 0.0
    report = _status(
        OK, '', n_candidates=len(distances),
        n_dataset=int(getattr(dataset, 'nframe', 0) or 0),
        n_reference=int(reference.shape[0]),
        coverage_radius=radius,
        covered_fraction=(float(np.mean(distances <= radius))
                          if radius > 0 else None),
        nearest_distance={'min': float(distances.min()),
                          'median': float(np.median(distances)),
                          'max': float(distances.max())})
    if average_length is not None:
        report['average_length'] = float(average_length)
    if mdstep:
        report['length_fraction'] = float(average_length or 0.0) / float(mdstep)
    return report


def _ker(displacement, grad_left, grad_right):
    displacement = np.asarray(displacement, dtype=np.float64).reshape(-1)
    delta_gradient = (np.asarray(grad_right, dtype=np.float64)
                      - np.asarray(grad_left, dtype=np.float64)).reshape(-1)
    norm = float(np.dot(displacement, displacement))
    if norm <= 0.0:
        return None
    kappa = float(np.dot(displacement, delta_gradient)) / norm
    slope = float(np.linalg.norm(delta_gradient)) / np.sqrt(norm)
    return kappa, slope


def _reconstruction(displacements, gradient_changes, rank=3, ridge=1e-8):
    """Low-rank local gradient reconstruction with leave-one-out error.

    A linear map H is fitted in the leading right-singular directions of
    the displacement matrix; each neighbour is predicted from a fit that
    excludes it, so the reported residual is not a training residual.
    """
    displacement = np.asarray(displacements, dtype=np.float64)
    change = np.asarray(gradient_changes, dtype=np.float64)
    count = displacement.shape[0]
    if count < 3:
        return None
    _u, singular, vt = np.linalg.svd(displacement, full_matrices=False)
    if singular.size:
        tolerance = max(float(singular[0]), 1.0) * 1e-10
        numerical_rank = int(np.sum(singular > tolerance))
    else:
        numerical_rank = 0
    usable = int(min(rank, count - 1, numerical_rank))
    if usable < 1:
        return None
    basis = vt[:usable].T
    projected_d = displacement @ basis
    projected_g = change @ basis
    residuals = []
    condition = None
    for position in range(count):
        mask = np.arange(count) != position
        if int(mask.sum()) < 2:
            continue
        train_d = projected_d[mask]
        train_g = projected_g[mask]
        normal = train_d.T @ train_d + ridge * np.eye(usable)
        condition = float(np.linalg.cond(normal))
        try:
            mapping = np.linalg.solve(normal, train_d.T @ train_g)
        except np.linalg.LinAlgError:
            continue
        predicted = projected_d[position] @ mapping
        residuals.append(float(np.linalg.norm(
            predicted - projected_g[position])) / np.sqrt(change.shape[1]))
    if not residuals:
        return None
    return {
        'rank': usable,
        'uncovered_directions': int(change.shape[1] - usable),
        'condition_number': condition,
        'rF': float(np.sqrt(np.mean(np.square(residuals)))),
        'n_validation': len(residuals),
    }


def summarize_pes(rows, dataset, state=0):
    """Local point-pair curvature, the energy-gradient residual and the
    low-rank reconstruction of the gradient change.

    Each candidate is referred to its nearest QM frames; the pair gives
    kappa and r^E, and the neighbourhood gives the leave-one-out r^F. This
    is the PES-resolution line: no new QM call, and a thin neighbourhood
    is `not_applicable` with a countable reason rather than a number.
    """
    if dataset is None:
        return _status(NOT_APPLICABLE, 'no dataset is available')
    if getattr(dataset, 'energy', None) is None or getattr(dataset, 'grad', None) is None:
        return _status(NOT_APPLICABLE,
                       'the dataset carries no energy/gradient labels')
    if int(getattr(dataset, 'nframe', 0)) < 2:
        return _status(
            NOT_APPLICABLE,
            'local pairs need at least 2 QM frames; the dataset has %d'
            % int(getattr(dataset, 'nframe', 0)))
    if getattr(dataset, 'pbc', None) is not None:
        try:
            periodic = bool(np.any(np.asarray(dataset.pbc)))
        except (TypeError, ValueError):
            periodic = False
        if periodic:
            return _status(
                NOT_APPLICABLE,
                'periodic datasets are not covered by the local pair '
                'analysis in this round')
    index = _sample_frames(int(dataset.nframe))
    reference = []
    for position in index:
        descriptor = _descriptor(dataset.xyz[position])
        if descriptor is None:
            return _status(
                NOT_APPLICABLE,
                'the descriptor needs at most %d atoms; dataset has %d'
                % (MAX_DESCRIPTOR_ATOMS, int(getattr(dataset, 'natom', 0) or 0)))
        reference.append(descriptor)
    reference = np.asarray(reference, dtype=np.float64)
    rows_out = []
    for row in rows:
        coordinates = row.get('coordinates')
        descriptor = _descriptor(coordinates)
        if descriptor is None or descriptor.size != reference.shape[1]:
            continue
        distances = np.linalg.norm(reference - descriptor, axis=1)
        order = np.argsort(distances)
        if order.size < 2:
            continue
        left, right = int(index[order[0]]), int(index[order[1]])
        energies = np.asarray(dataset.energy, dtype=np.float64)
        gradients = np.asarray(dataset.grad, dtype=np.float64)
        if energies.ndim != 2 or gradients.ndim != 4:
            return _status(NOT_APPLICABLE,
                           'dataset labels do not have the expected layout')
        if state >= energies.shape[1] or state >= gradients.shape[1]:
            return _status(
                NOT_APPLICABLE,
                'state %d is outside the dataset state axis (%d)'
                % (state, energies.shape[1]))
        displacement = (np.asarray(dataset.xyz[right], dtype=np.float64)
                        - np.asarray(dataset.xyz[left], dtype=np.float64))
        displacement = displacement * ANGSTROM_TO_BOHR
        pair = _ker(displacement, gradients[left, state],
                    gradients[right, state])
        if pair is None:
            continue
        kappa, slope = pair
        residual = float(
            energies[right, state] - energies[left, state]
            - 0.5 * np.dot(
                (gradients[left, state]
                 + gradients[right, state]).reshape(-1),
                displacement.reshape(-1)))
        neighbours = [int(index[position])
                      for position in order[:min(4, order.size)]]
        reference_position = neighbours[0]
        displacements = []
        changes = []
        for position in neighbours[1:]:
            step = (np.asarray(dataset.xyz[position], dtype=np.float64)
                    - np.asarray(dataset.xyz[reference_position],
                                 dtype=np.float64)) * ANGSTROM_TO_BOHR
            displacements.append(step.reshape(-1))
            changes.append((gradients[position, state]
                            - gradients[reference_position, state]
                            ).reshape(-1))
        reconstruction = _reconstruction(displacements, changes)
        rows_out.append({
            'traj_id': row.get('traj_id'), 'step': row.get('step'),
            'neighbours': [int(left), int(right)],
            'neighbour_set': neighbours,
            'nearest_distance': float(distances[order[0]]),
            'curvature': kappa, 'gradient_change': slope,
            'energy_residual': residual,
            'reconstruction': reconstruction})
    if not rows_out:
        return _status(
            NOT_APPLICABLE,
            'no candidate had two QM neighbours among the %d sampled frames'
            % reference.shape[0])
    resolution = sorted(
        (float(item['nearest_distance']),
         float(item['reconstruction']['rF']))
        for item in rows_out if item.get('reconstruction'))
    summary = _status(OK, '', state=int(state), candidates=rows_out)
    summary['resolution_pairs'] = resolution
    summary['reconstruction_candidates'] = int(len(resolution))
    return summary


def predictor_status(rows, requested_mode):
    """`not_requested` / `available` / `missing` for the comparison data.

    The skill only supplies the arrays; the judgment lives here. A row
    carries `alternatives` when the serving skill returned them; a
    partially returned set is missing, not available.
    """
    mode = int(requested_mode or 0)
    if mode == 0:
        return _status(
            NOT_REQUESTED,
            'predict_mode=0 returns the member mean; no comparison model '
            'was requested')
    if not rows:
        return _status(NOT_APPLICABLE, 'the trajectory retained no frames')
    missing = []
    for row in rows:
        alternatives = row.get('alternatives')
        if not isinstance(alternatives, dict) or not alternatives:
            missing.append('alternatives')
            continue
        for channel, value in alternatives.items():
            if channel == 'model_ids':
                continue
            if value is None:
                missing.append(channel)
    if missing:
        return _status(
            MISSING,
            'predict_mode=%d requested comparison data; missing channels: %s'
            % (mode, ', '.join(sorted(set(missing)))))
    return _status(OK, '', model_ids=rows[0]['alternatives'].get('model_ids'))


def saturation(history, window, global_tolerance, local_tolerance=0.0):
    """MFML-style moving-average stagnation over the validation history.

    `local_tolerance` is the per-round improvement the fit is still
    allowed to call progress; while the improvement is above it the round
    is not saturated even if the global tolerance is loose. The global
    tolerance decides the stop once the local one no longer sees news.
    """
    values = [float(value) for value in history if value is not None]
    window = max(1, int(window or 1))
    if len(values) < window + 1:
        return _status(
            NOT_APPLICABLE,
            'saturation needs %d rounds of validation MAE; have %d'
            % (window + 1, len(values)))
    if global_tolerance <= 0 and local_tolerance <= 0:
        return _status(
            NOT_APPLICABLE,
            'both analysis tolerances are 0; saturation stop is disabled')
    def average(end):
        segment = values[max(0, end - window):end]
        return float(np.mean(segment)) if segment else 0.0
    previous = average(len(values) - 1)
    current = average(len(values))
    improvement = abs(previous - current)
    local_progress = bool(local_tolerance > 0
                          and improvement > local_tolerance)
    saturated = bool(not local_progress
                     and global_tolerance > 0
                     and improvement <= global_tolerance)
    return _status(
        OK, '', window=window, moving_average=current,
        improvement=improvement,
        local_tolerance=float(local_tolerance),
        global_tolerance=float(global_tolerance),
        local_progress=local_progress,
        saturated=saturated)
