"""
======================================================================
This module provides the adaptive-private propagation bias: the UDD
uncertainty potential and the gap-CV driving term.

Both are state-resolved corrections to one evaluation result, applied
before the constraint potentials and only when the adaptive workflow
stamped a spec on RuntimeState.analysis_bias. With no spec the function
returns the caller's mapping untouched, which keeps every non-adaptive
run byte-identical.

Two bookkeeping rules the caller relies on:

- The unbiased energies and gradients are kept under `unbiased_energy` /
  `unbiased_gradient`. The hopping integrator and the analysis layer read
  those, so a biased nuclear force never changes a hopping probability or
  a reported gap.
- The bias terms are reported separately (`bias_energy`, `gap_bias_energy`)
  and never enter the training labels, which come from the reference
  calculation.

Nov 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# Fallbacks used when the run leaves the amplitude/width at 0 ("auto").
# They are order-of-magnitude starting points, not calibrated constants:
# the paper's own values are context dependent and are selected from the
# barrier scale and the near-minimum uncertainty.
DEFAULT_UDD_AMPLITUDE = 5.0e-3      # Hartree
DEFAULT_GAP_HEIGHT = 1.0e-3         # Hartree
MIN_WIDTH = 1.0e-6                  # Hartree


def _array(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.size else None


def _udd(state, data, spec, energies, gradients):
    """Apply the UDD bias to the current active state.

    Returns (bias_energy, applied). The current state is read from the
    system, so a hop moves the bias target on the next step.
    """
    mode_id = int(data.get('predict_model_id', 0) or 0)
    alternative_energy = _array(data.get('alternative_energy'))
    alternative_gradient = _array(data.get('alternative_gradient'))
    if mode_id <= 0 or alternative_energy is None or alternative_gradient is None:
        return 0.0, False
    system = getattr(getattr(state, 'pmd', None), 'system', None)
    current = int(getattr(system, 'current_state', 0) or 0)
    nstate = energies.shape[0]
    if not 0 <= current < nstate:
        return 0.0, False
    last_state = spec.get('last_state')
    if last_state is not None and int(last_state) != current:
        spec['state_switches'] = int(spec.get('state_switches', 0)) + 1
    spec['last_state'] = current
    data['udd_state_switches'] = int(spec.get('state_switches', 0))
    if alternative_energy.ndim != 2 or alternative_gradient.ndim != 4:
        return 0.0, False
    member_energy = np.concatenate(
        [np.asarray([energies[current]]),
         alternative_energy[:, current]], axis=0)
    member_gradient = np.concatenate(
        [gradients[current:current + 1],
         alternative_gradient[:, current, :, :]], axis=0)
    members = member_energy.shape[0]
    mean = float(np.mean(member_energy))
    variance = 0.5 * float(np.sum((member_energy - mean) ** 2))
    if variance <= 0.0 or members < 2:
        return 0.0, False
    natom = gradients.shape[1]
    amplitude = float(spec.get('udd_amplitude') or 0.0)
    width = float(spec.get('udd_width') or 0.0)
    if amplitude <= 0.0:
        amplitude = DEFAULT_UDD_AMPLITUDE
    if width <= 0.0:
        width = max(0.1 * np.sqrt(variance), MIN_WIDTH)
    scale = members * max(natom, 1) * width * width
    exponent = -variance / scale
    bias_energy = amplitude * (np.exp(exponent) - 1.0)
    derivative = -amplitude / scale * np.exp(exponent)
    # d(sigma^2)/dR with gradients (dU/dR) instead of forces:
    # d(sigma^2)/dR = sum_i (E_i - mean) * g_i.
    displaced = (member_energy - mean)[:, None, None] * member_gradient
    gradient_bias = derivative * np.sum(displaced, axis=0)
    true_force = -gradients[current]
    bias_force = -gradient_bias
    true_norm = np.linalg.norm(true_force, axis=1)
    bias_norm = np.linalg.norm(bias_force, axis=1)
    mask = true_norm > 1e-12
    if np.any(mask):
        data['udd_force_ratio'] = float(
            np.mean(bias_norm[mask] / true_norm[mask]))
    coordinates = getattr(system, 'coordinates', None)
    if coordinates is not None:
        current_coordinates = np.asarray(coordinates, dtype=np.float64)
        previous = spec.get('last_coordinates')
        if previous is not None:
            previous = np.asarray(previous, dtype=np.float64)
            if previous.shape == current_coordinates.shape:
                # Force is -gradient; work = F . dR over the step.
                work = float(np.sum(
                    -gradient_bias * (current_coordinates - previous)))
                spec['bias_work'] = float(
                    spec.get('bias_work', 0.0)) + work
        spec['last_coordinates'] = current_coordinates.tolist()
    data['bias_work'] = float(spec.get('bias_work', 0.0))
    if str(spec.get('udd_mode', 'force')).lower() == 'potential':
        energies[current] = energies[current] + bias_energy
        data['bias_energy'] = float(bias_energy)
    else:
        data['bias_energy'] = 0.0
    gradients[current] = gradients[current] + gradient_bias
    data['udd_energy'] = float(bias_energy)
    return float(bias_energy), True


def _gap_paper(state, data, spec, gradients, unbiased_energy,
               unbiased_gradient, lower, upper, gap, threshold):
    """The paper's gap-driven propagation: replace the active gradient.

    Above the threshold the active state's gradient is the gap gradient
    (upper minus lower); below it the active surface's own gradient is
    kept. The switch is not conservative, so the excess energy is removed
    by rescaling the velocities, and when there is not enough kinetic
    energy to pay for it the trajectory stays on the electronic surface.
    """
    if gap <= threshold:
        return 0.0, False
    active = int(getattr(
        getattr(getattr(state, 'pmd', None), 'system', None),
        'current_state', 0) or 0)
    if active not in (lower, upper):
        active = lower
    kinetic = 0.0
    if callable(getattr(state, 'kinetic_energy', None)):
        kinetic = float(state.kinetic_energy() or 0.0)
    reference = spec.get('reference_total')
    active_energy = float(unbiased_energy[active])
    if reference is None:
        spec['reference_total'] = kinetic + active_energy
        reference = spec['reference_total']
    excess = kinetic + active_energy - float(reference)
    if kinetic <= 0.0 or excess >= kinetic:
        return 0.0, False
    if excess > 0.0:
        scale = float(np.sqrt(max(0.0, 1.0 - excess / kinetic)))
        velocities = getattr(state, 'velocities', None)
        velocities = (np.asarray(velocities, dtype=np.float64)
                      if velocities is not None else np.empty(0))
        if velocities.size:
            state.update_system({
                'velocities': (velocities * scale).tolist()})
        data['gap_velocity_scale'] = scale
    gradients[active] = (unbiased_gradient[upper]
                         - unbiased_gradient[lower])
    data['gap_force_switched'] = True
    data['gap_excess'] = float(excess)
    return 0.0, True


def _gap(data, spec, energies, gradients, unbiased_energy, unbiased_gradient,
         state=None):
    """Apply the threshold-gated gap bias to the selected pair.

    The CV method's correction is antisymmetric (`+/-V/2`), so the pair's
    mean energy is unchanged and only the gap moves. The paper method
    replaces the active state's propagation gradient instead. Below the
    threshold (including equality) both leave the result untouched.
    """
    if unbiased_energy is None or unbiased_gradient is None:
        return 0.0, False
    pair = list(spec.get('gap_pair') or [])
    if len(pair) != 2:
        pair = [0, 1]
    lower, upper = int(pair[0]), int(pair[1])
    nstate = energies.shape[0]
    if not (0 <= lower < nstate and 0 <= upper < nstate and lower != upper):
        return 0.0, False
    threshold = float(spec.get('gap_threshold', 0.03) or 0.03)
    gap = float(unbiased_energy[upper] - unbiased_energy[lower])
    if gap <= threshold:
        return 0.0, False
    if str(spec.get('gap_method', 'cv')).lower() == 'paper' and state is not None:
        return _gap_paper(
            state, data, spec, gradients, unbiased_energy,
            unbiased_gradient, lower, upper, gap, threshold)
    height = float(spec.get('gap_height') or 0.0)
    width = float(spec.get('gap_width') or 0.0)
    if height <= 0.0:
        height = DEFAULT_GAP_HEIGHT
    if width <= 0.0:
        # The auto width follows the gap being driven, so a large gap does
        # not turn the same stiffness into a huge penalty.
        width = max(gap / 3.0, MIN_WIDTH)
    offset = gap - threshold
    # One-sided saturating potential: zero at the threshold, bounded by
    # `height` far above it, and monotone, so it pushes every large gap
    # down without the unbounded growth of a quadratic.
    decay = float(np.exp(-offset / width))
    penalty = height * (1.0 - decay)
    derivative = height / width * decay
    direction = (unbiased_gradient[upper]
                 - unbiased_gradient[lower])
    energies[lower] = energies[lower] - 0.5 * penalty
    energies[upper] = energies[upper] + 0.5 * penalty
    gradients[lower] = gradients[lower] - 0.5 * derivative * direction
    gradients[upper] = gradients[upper] + 0.5 * derivative * direction
    data['gap_bias_energy'] = float(penalty)
    return float(penalty), True


def finalize(state, data):
    """Apply the adaptive bias spec to one result mapping.

    Parameters
    ----------
    state : PyRAIState
        State carrying RuntimeState.analysis_bias.
    data : dict
        Assembled result mapping (energy, gradient, optional alternatives).

    Returns
    -------
    dict
        The same mapping, corrected when a spec is present.
    """
    if not isinstance(data, dict):
        return data
    runtime = getattr(state, 'runtime', None)
    spec = getattr(runtime, 'analysis_bias', None) if runtime is not None else None
    if not isinstance(spec, dict) or not spec:
        return data
    energies = _array(data.get('energy'))
    gradients = _array(data.get('gradient'))
    if energies is None or gradients is None:
        return data
    energies = energies.copy()
    gradients = gradients.copy()
    original_energy = energies.copy()
    original_gradient = gradients.copy()
    applied = False
    udd_energy = 0.0
    gap_energy = 0.0
    if spec.get('udd'):
        udd_energy, changed = _udd(
            state, data, spec['udd'], energies, gradients)
        applied = applied or changed
    if spec.get('gap'):
        gap_energy, changed = _gap(
            data, spec['gap'], energies, gradients,
            original_energy, original_gradient, state=state)
        applied = applied or changed
    if not applied:
        return data
    data['energy'] = energies
    data['gradient'] = gradients
    data['unbiased_energy'] = original_energy
    data['unbiased_gradient'] = original_gradient
    data['bias_status'] = 'applied'
    data.setdefault('udd_energy', float(udd_energy))
    data.setdefault('gap_bias_energy', float(gap_energy))
    return data
