"""
======================================================================
Harmonic bias potentials of the constraint subsystem.

A bias adds energy and force without moving the structure: the declaration is
V = 1/2 k delta^2, where delta is the difference between one scalar collective
variable and its target, taken one-sided when the region says so.

The single-CV rule is deliberate. One bias binds exactly one variable, so
several restraints mean several blocks, each with its own id and its own line
in the energy breakdown. A bias that tried to bind two variables at once would
have to define a combined metric, and that definition is not free.

Regions
-------
'all' takes delta as written, 'upper' clamps it at zero from below so only
moving past the target costs anything, and 'lower' clamps from above. At the
boundary the potential is continuous and its derivative is zero on both sides,
which is what makes the clamped form usable in a Verlet loop. A dihedral is
restricted to 'all' by the schema: a one-sided periodic region needs a declared
origin and direction, and inventing those silently would change the physics.

A periodic variable differs through atan2, so a target of 170 degrees and a
value of -170 degrees are 20 degrees apart, not 340. The legacy code folded by
hand with '360 - abs(dr)' and took the sign from the target, which lands on the
wrong side of the branch for some crossings; this form cannot.

Units
-----
Coordinates in Bohr, CV values in Bohr or radian, energy in Hartree, force
constant in Hartree per Bohr squared or per radian squared, all converted once
by schema normalization. The returned gradient is dV/dR in Hartree per Bohr.

The module must stay a leaf: numpy, the lab CV layer and the lab records.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.utilities import cv, geometric
from pyrai3md.workflow.constraints import schedule


class BiasError(ValueError):
    """A bias cannot be evaluated as declared."""


def variable_value(record, coords):
    """The value of one collective variable at a geometry.

    Parameters
    ----------
    record : CVRecord
    coords : numpy.ndarray
        (natom, 3) coordinates in Bohr.

    Returns
    -------
    float
        Distance in Bohr, or angle in radians.
    """
    return _dispatch(record, coords, grad=False)


def evaluate(spec, plan, coords, state, context):
    """Evaluate one harmonic bias at a geometry.

    Parameters
    ----------
    spec : ConstraintSpec
        The bias, with `kind == 'bias'` and `family == 'harmonic'`.
    plan : ConstraintPlan
        Supplies the CV the bias binds.
    coords : numpy.ndarray
        (natom, 3) coordinates in Bohr, the geometry being evaluated.
    state : ConstraintState
        Supplies the target when the declaration said 'initial'.
    context : EvaluationContext
        Supplies the accepted step for a schedule.

    Returns
    -------
    dict
        'kind', 'family', 'cv', 'cv_value', 'target', 'parameters', 'energy'
        and 'gradient' with shape (natom, 3) in Hartree per Bohr.
    """
    if spec.kind != 'bias':
        raise BiasError(
            "evaluate() got a '%s' action, not a bias" % spec.kind)
    if spec.family != 'harmonic':
        raise BiasError(
            "bias '%s' of family %r is not implemented" % (spec.id,
                                                           spec.family))
    if spec.cv is None or spec.cv not in plan.cvs:
        raise BiasError(
            "bias '%s' has no resolvable cv" % spec.id)
    record = plan.cvs[spec.cv]
    natom = int(np.asarray(coords).shape[0])
    value, jacobian = _dispatch(record, coords, grad=True)

    target = _target_of(spec, state, record)
    parameters = schedule.resolve_parameters(spec, context, spec.parameters)
    try:
        constant = float(parameters['force_constant'])
    except (KeyError, TypeError, ValueError):
        raise BiasError(
            "bias '%s' has no usable force_constant" % spec.id)

    delta, chain = _difference(value, target, record, spec.region)
    energy = 0.5 * constant * delta * delta
    gradient = np.zeros((natom, 3), dtype=np.float64)
    if chain != 0.0 and constant != 0.0:
        derivative = constant * delta * chain
        gradient = cv.scatter_jacobian(
            jacobian * derivative, record.indices, natom)
    return {'kind': 'bias', 'family': spec.family, 'id': spec.id,
            'cv': spec.cv, 'cv_value': float(value), 'target': float(target),
            'parameters': {'force_constant': constant},
            'region': spec.region, 'energy': float(energy),
            'gradient': gradient}


def hessian(spec, plan, coords, state, context=None):
    """The exact `d2V/dR2` of one harmonic bias, as a (3N, 3N) block.

    With the half-factor this subsystem uses, `V = 1/2 * k * e^2` for the
    one-sided or periodic-wrapped difference `e`,

        d2V/dR2 = k * (J J^T + chain * e * Q)

    where `J` is the CV Jacobian the evaluation already uses and `Q` is the CV's
    own second derivative (`utilities/geometric.bond_hessian_core`,
    `angle_hessian_core`). The first term is what a numerical Hessian of the
    displaced gradients also captures; the second is why that route has a step
    size in it at all - it is the curvature of the variable, and it vanishes
    exactly at the target, which is where a converged run sits.

    Parameters
    ----------
    spec : ConstraintSpec
        A `kind == 'bias'` action.
    plan : ConstraintPlan
    coords : numpy.ndarray
        (natom, 3) in Bohr.
    state : ConstraintState
        Supplies a target bound at the reference geometry.
    context : EvaluationContext, optional
        Only a schedule would need it; no bias parameter is scheduled here.

    Returns
    -------
    dict
        'kind', 'id', 'cv', 'cv_value', 'parameters', 'size' and 'hessian'
        shaped (3 * natom, 3 * natom) in Hartree per Bohr squared.

    Raises
    ------
    BiasError
        An unknown CV kind, or a family other than 'harmonic'.
    """
    if spec.kind != 'bias':
        raise BiasError("hessian() got a '%s' action, not a bias" % spec.kind)
    if spec.family != 'harmonic':
        raise BiasError("bias '%s' of family %r has no analytic curvature"
                        % (spec.id, spec.family))
    record = plan.cvs[spec.cv] if spec.cv in getattr(plan, 'cvs', {}) else None
    if record is None:
        raise BiasError("bias '%s' names unknown cv '%s'" % (spec.id, spec.cv))
    array = np.asarray(coords, dtype=np.float64)
    natom = array.shape[0]
    parameters = schedule.resolve_parameters(spec, context, spec.parameters) \
        if context is not None else dict(spec.parameters or {})
    constant = float(parameters.get('force_constant', 0.0))
    target = _target_of(spec, state, record)
    value, jacobian = _dispatch(record, array, True)
    delta, chain = _difference(value, target, record, spec.region)
    side = _side_factor(value, target, record, spec.region)
    # `cv` hands back the block in the CV's own atom order; the row the Hessian
    # needs is indexed by atom number.
    row = np.zeros(natom * 3, dtype=np.float64)
    for position, atom in enumerate(record.indices):
        index = int(atom) * 3
        row[index:index + 3] = np.asarray(jacobian, dtype=np.float64)[position]
    # Inside a one-sided region the clamped difference is zero AND its slope is
    # zero, so the whole curvature vanishes there: `V = 1/2 k max(delta, 0)^2`
    # is flat, not merely force-free. `chain` is d(raw delta)/d(value) and stays
    # 1.0 through the clamp, so it cannot carry this factor - the activity test
    # below is what does, and leaving it out was measured as a nonzero block
    # where the energy and the gradient were both zero.
    block = constant * side * side * np.outer(row, row)
    second = geometric.cv_hessian_block(record.kind, array, record.indices)
    if second is None:
        raise BiasError(
            "bias '%s' over the cv '%s' of kind %r has no second derivative to "
            'take, and none is assumed: an unwritten term is not a vanishing '
            'one' % (spec.id, record.id, record.kind))
    core, atoms = second
    if delta != 0.0 and constant != 0.0:
        for left in range(len(atoms)):
            for right in range(len(atoms)):
                li = int(atoms[left]) * 3
                ri = int(atoms[right]) * 3
                block[li:li + 3, ri:ri + 3] += (constant * chain * side
                                                * delta * core[left, right])
    return {'kind': 'bias', 'id': spec.id, 'cv': spec.cv,
            'cv_value': float(value), 'target': float(target),
            'parameters': {'force_constant': constant},
            'region': spec.region, 'size': natom * 3, 'hessian': block}


def _side_factor(value, target, record, region):
    """1.0 where the bias is active, 0.0 inside the flat side of a clamp.

    `region` decides which sign of the raw difference is biased. The returned
    factor multiplies both terms of the curvature, because `max(delta, 0)^2`
    has a vanishing first AND second derivative where it is clamped.
    """
    if region in (None, 'all'):
        return 1.0
    raw = (cv.periodic_difference(value, target) if record.periodic
           else float(value) - float(target))
    if region == 'upper':
        return 1.0 if raw > 0.0 else 0.0
    if region == 'lower':
        return 1.0 if raw < 0.0 else 0.0
    raise BiasError("bias has unsupported region %r" % (region,))


def _dispatch(record, coords, grad):
    """Evaluate one CV record, hiding which kind it is.

    A clamped acos at an exactly collinear angle and a zero-length bond are
    both refusals rather than nans, and cv raises for them.
    """
    indices = [int(position) for position in record.indices]
    if record.kind == 'bond':
        return cv.distance(coords, indices[0], indices[1], grad=grad)
    if record.kind == 'angle':
        return cv.angle(coords, indices[0], indices[1], indices[2],
                        grad=grad)
    if record.kind == 'dihedral':
        return cv.dihedral(coords, indices[0], indices[1], indices[2],
                           indices[3], grad=grad)
    raise BiasError("unknown cv kind %r for cv '%s'"
                    % (record.kind, record.id))


def _target_of(spec, state, record):
    """The declared target, or the one bound at the reference geometry."""
    if spec.target is not None:
        return float(spec.target)
    stored = getattr(state, 'targets', {}).get(spec.id)
    if stored is None:
        raise BiasError(
            "bias '%s' has no target: the declaration left it to be captured "
            "at the reference geometry and nothing bound one" % spec.id)
    return float(stored)


def _difference(value, target, record, region):
    """The biased difference and its chain factor against the raw variable.

    Returns
    -------
    tuple
        (delta, chain) where `chain` is d(delta)/d(value): 1.0 for the plain
        and clamped forms, and still 1.0 for a periodic variable because the
        atan2 difference has unit derivative away from its own branch.
    """
    if record.periodic:
        delta = cv.periodic_difference(value, target)
        return _clamp(delta, region, record.id), 1.0
    delta = float(value) - float(target)
    return _clamp(delta, region, record.id), 1.0


def _clamp(delta, region, name):
    """Apply the one-sided region rule."""
    if region in (None, 'all'):
        return delta
    if region == 'upper':
        return delta if delta > 0.0 else 0.0
    if region == 'lower':
        return delta if delta < 0.0 else 0.0
    raise BiasError(
        "bias '%s' has unsupported region %r" % (name, region))
