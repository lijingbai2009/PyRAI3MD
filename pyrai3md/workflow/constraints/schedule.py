"""
======================================================================
Time-dependent parameters of the constraint subsystem.

A schedule ramps one declared parameter of one action across accepted steps.
The first version implements a linear ramp and nothing else: the endpoints are
either multipliers of the declared value ('scale') or absolute values
('value'),
the ramp is keyed on the absolute accepted step number, and outside the window
the parameter holds the nearest endpoint instead of extrapolating.

Two rules come from the design document (appendix section 7.5) and shape this
module:

- Repeated evaluation of the same step must give the same answer, so nothing
  here keeps a counter. Progress comes from the EvaluationContext only, which
  is why a retried or logged step cannot advance a ramp.
- A parameter that has no implementation cannot be ramped. The caller passes
  the names it accepts, and anything else is a refusal rather than a no-op.

Strictly positive parameters stay strictly positive: a ramp that would drive a
radius to zero or below fails at the first step where it does, instead of
producing a degenerate geometry that the signed distance cannot resolve.

The module must stay a leaf: numpy and the lab records only.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

What this is not
----------------
`workflow/dispatch/scheduler.py` runs *jobs*: it takes a task
queue and decides when a process starts. This module ramps *parameters* against
the accepted step of one run. The two share a word and nothing else, and the boundary
is stated here so a search for either one does not stop at the other (dev/methods/
CONSTRAINT_DEVELOPMENT.md section 18, D-13).
"""

import numpy as np


class ScheduleError(ValueError):
    """A schedule cannot be resolved as declared."""


POSITIVE_PARAMETERS = ('radius', 'force_constant', 'height', 'semiaxes',
                       'lengths')


def ramp_fraction(record, context):
    """Where the accepted step sits inside the window, clamped to [0, 1].

    Parameters
    ----------
    record : ScheduleRecord
    context : EvaluationContext

    Returns
    -------
    float
        0.0 before the window, 1.0 at or after its end, and the linear
        interpolation between.
    """
    return _fraction(record, int(context.step))


def _fraction(record, step):
    """The same fraction, from a step number rather than a context."""
    start = int(record.start_step)
    end = int(record.end_step)
    if end <= start:
        raise ScheduleError(
            "schedule '%s' needs end_step beyond start_step" % record.id)
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return float(step - start) / float(end - start)


# The ramped quantities that decide how much space a wall encloses. A force
# constant is not one of them: it moves the height of a barrier, never where
# the barrier stands.
GEOMETRY_PARAMETERS = ('radius', 'semiaxes', 'lengths', 'cell')


def endpoint_variants(spec, declared_values, names=GEOMETRY_PARAMETERS):
    """Every value a geometry ramp reaches, one parameter dict per extreme.

    A linear ramp is monotone in each of its components, so its endpoints
    bound everything in between: a periodic wall that fits the cell at both of
    them fits it at every step of the ramp. `delay` and `stop` say *when* a
    value is reached, not *which* values exist, which is why the bound is
    written on the endpoints and not on the window.

    Returns
    -------
    list of dict
        Empty when the action ramps no geometry, so a caller that loops over
        this checks the declared values exactly once and no more.
    """
    ramps = [record for record in spec.all_schedules()
             if record.parameter in names]
    if not ramps:
        return []
    steps = set()
    for record in ramps:
        steps.add(int(record.start_step))
        steps.add(int(record.end_step))
    variants = []
    for step in sorted(steps):
        values = dict(declared_values)
        for record in ramps:
            fraction = _fraction(record, step)
            if record.scale is not None:
                low, high = record.scale
                values[record.parameter] = _multiply(
                    declared_values[record.parameter],
                    low + (high - low) * fraction)
            else:
                low, high = record.value
                values[record.parameter] = _blend(low, high, fraction)
        variants.append(values)
    return variants


def resolve_parameters(spec, context, declared_values):
    """Apply the action's schedule, if any, to its declared parameters.

    Parameters
    ----------
    spec : ConstraintSpec
        The action. Its `schedule` and `schedules` fields may be empty, hold
        one ramp, or hold several - each declared parameter moves on its own
        window, and the same step always yields the same values.
    context : EvaluationContext
        Supplies the accepted step.
    declared_values : dict
        Parameter name to value, already in internal units. The dict is not
        modified.

    Returns
    -------
    dict
        A copy with the ramped values substituted.
    """
    values = dict(declared_values)
    for record in getattr(spec, 'all_schedules', lambda: ())():
        name = record.parameter
        if name not in values:
            raise ScheduleError(
                "action '%s' has no parameter '%s' for schedule '%s' to ramp"
                % (spec.id, name, record.id))
        fraction = ramp_fraction(record, context)
        if record.scale is not None:
            low, high = record.scale
            values[name] = _multiply(values[name], low + (high - low)
                                     * fraction)
        else:
            low, high = record.value
            values[name] = _blend(low, high, fraction)
        if _violates_positivity(name, values[name]):
            raise ScheduleError(
                "schedule '%s' drives '%s' of action '%s' to a non-positive "
                "value (%s)" % (record.id, name, spec.id, values[name]))
    return values


def _blend(low, high, fraction):
    """Interpolate scalars and lists alike."""
    if isinstance(low, (list, tuple)) or isinstance(high, (list, tuple)):
        left = np.atleast_1d(np.asarray(low, dtype=np.float64))
        right = np.atleast_1d(np.asarray(high, dtype=np.float64))
        if left.shape != right.shape:
            raise ScheduleError(
                'schedule endpoints differ in shape: %s against %s'
                % (left.shape, right.shape))
        return (left + (right - left) * fraction).tolist()
    return float(low) + (float(high) - float(low)) * fraction


def _multiply(declared, factor):
    """Scale a scalar or a list of declared values by one factor."""
    if isinstance(declared, (list, tuple, np.ndarray)):
        return [float(value) * factor
                for value in np.atleast_1d(declared).tolist()]
    return float(declared) * factor


def _violates_positivity(name, value):
    """True when a parameter that must stay positive does not."""
    if name not in POSITIVE_PARAMETERS:
        return False
    numbers = np.atleast_1d(np.asarray(value, dtype=np.float64))
    return bool(np.any(numbers <= 0.0))
