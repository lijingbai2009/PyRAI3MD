"""
======================================================================
Wall potentials of the constraint subsystem.

A wall adds energy and force when geometry crosses a declared boundary, and it
never moves the structure. The form is fixed:

    V = k * max(d, 0) ** p,   p >= 2,

with no factor of one half, where d is an explicitly declared length-like
overshoot. Inside the boundary d is negative, the potential is zero and so is
its derivative; at the boundary both are zero as well, which is what keeps the
form usable inside a Verlet loop.

Geometry, target and potential shape are three separate choices, so this
module composes them rather than enumerating combinations.

Signed distances
----------------
- sphere:      d = |r - center| - radius
- ellipsoid:   rho = sqrt(sum((x_i / a_i)^2)), d = min(a) * (rho - 1)
- box:         d_i = |x_i - center_i| - lengths_i / 2, summed one-sided
- triclinic:   s = (r - center) @ B^-1 with B the row-vector lattice,
               d_i = (|s_i| - 1/2) / |B^-1 column i|

The ellipsoid distance is a scaled implicit distance, not the shortest
Euclidean
one; it is continuous, vanishes exactly on the surface and has a bounded
gradient, which is what the potential needs, and the design document says so
rather than letting the name imply more. The box is the diagonal special
case of
the triclinic form, and tests/test_wall.py asserts the two agree, because a
disagreement there would mean one of them is wrong. The triclinic
denominator is
the interplanar spacing, so the same declared k keeps the same physical meaning
as the cell changes shape instead of silently scaling with the cell.

Targets
-------
'atom' evaluates and sums per atom. 'com' and 'cog' evaluate once at the center
of each group and back-propagate with that center's own weights (m_i over the
total mass, or one over the count), so the energy of a centroid wall is not
multiplied by the number of atoms behind it. `wall_groups` cuts the selection
into molecules and gives one evaluation point per molecule; without it the whole
selection is the one group. The grouping says what is confined, the target says
how each group is weighted - the two are separate choices, which is why an
atom target has the grouping taken away rather than obeyed.

Two laws
--------
`hard` is the form above: nothing until the surface is crossed. `soft` is
the archived polynomial cavity, `V = k * Q^(p/2)` over the dimensionless squared
axis ratio `Q` - the same function as `f * (r/r0)^a` with `k = f`, `p = a` and
`r0` per axis - which pulls toward the center as strongly as it pushes away,
and its constant therefore carries no inverse length. A box of the second law uses half
edges as its axis scales, so the number a cavity spelled as a radius is the
number a wall spells as a box edge divided by two.

The wall center
---------------
The center is not a coordinate convention: `wall_center` gives the point,
`wall_ref` gives atoms whose center becomes it, and both are captured once at the
reference geometry and fixed for the run. Neither defaults to the origin - a wall
whose input says nothing about a center is confined around the center of the
atoms it acts on.

The wall is an external boundary
--------------------------------
Its geometry - radius, semiaxes, edge lengths, triclinic matrix - is a declared
parameter of the wall and never a view of the simulation cell. Deforming the
run's box therefore moves the atoms *relative to* the wall instead of carrying
the wall along, and a molecule that filled half its box keeps filling half its
box only if nothing moved. `tests/test_cv_cell.py` asserts this, because a
confinement meant to ride the cell would have to re-derive its boundary from
the lattice every step, which is a different object and not what any geometry
here claims.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.utilities import geometric
from pyrai3md.workflow.constraints import schedule


class WallError(ValueError):
    """A wall cannot be evaluated as declared."""


def evaluate(spec, plan, coords, state, context, masses=None):
    """Evaluate one wall at a geometry.

    Parameters
    ----------
    spec : ConstraintSpec
        The wall, with `kind == 'wall'` and a geometry as its family.
    plan : ConstraintPlan
    coords : numpy.ndarray
        (natom, 3) coordinates in Bohr.
    state : ConstraintState
        Supplies a center bound at the reference geometry, when declared.
    context : EvaluationContext
        Supplies the accepted step for a schedule.
    masses : array-like, optional
        Per-atom masses in atomic units of mass. Required by target 'com'.

    Returns
    -------
    dict
        'kind', 'family', 'id', 'parameters', 'energy', 'gradient' shaped
        (natom, 3) in Hartree per Bohr, and 'extent' carrying the largest
        overshoot actually seen, which is the number a run should be
        watched on.
    """
    if spec.kind != 'wall':
        raise WallError(
            "evaluate() got a '%s' action, not a wall" % spec.kind)
    natom = int(np.asarray(coords).shape[0])
    parameters = schedule.resolve_parameters(spec, context, spec.parameters)
    geometry = str(parameters.get('geometry', spec.family))
    power = int(parameters.get('power', 2))
    if power < 2:
        raise WallError(
            "wall '%s' power must be at least 2 so the force is continuous"
            % spec.id)
    constant = float(parameters.get('force_constant', 0.0))
    law = str(spec.parameters.get('law', 'hard'))
    one_sided = law != 'soft'
    center = _center_of(spec, parameters, state, coords, natom)

    gradient = np.zeros((natom, 3), dtype=np.float64)
    energy = 0.0
    largest = None
    objects = _objects(spec, plan, coords, masses)
    for members, weights, point in objects:
        offset = np.asarray(point, dtype=np.float64) - center
        terms = (_ratio_terms(geometry, offset, parameters, spec.id)
                 if not one_sided else
                 _signed(geometry, offset, parameters, spec.id))
        total, push, extent = _terms_energy(terms, constant, power,
                                            one_sided=one_sided)
        energy += total
        if push is not None and np.any(push != 0.0):
            gradient[np.asarray(members, dtype=np.int64)] += (
                push * np.asarray(weights, dtype=np.float64)[:, None])
        largest = extent if largest is None else max(largest, extent)
    return {'kind': 'wall', 'family': geometry, 'id': spec.id,
            'parameters': {'force_constant': constant, 'power': power},
            'energy': float(energy), 'gradient': gradient,
            'extent': float(largest) if largest is not None else None,
            'target_kind': spec.parameters.get('target_kind', 'atom'),
            'law': law, 'groups': len(objects)}


def _objects(spec, plan, coords, masses=None):
    """The points one wall is evaluated at, with the atoms behind each one.

    Returns
    -------
    list of tuple
        ``(members, weights, point)`` per evaluated object: one entry per atom
        when the target is an atom, one per molecule when the wall groups its
        atoms, and one for the whole selection when it does not. `weights` sums
        to one over `members`, which is what spreads a group's force back onto
        its atoms, and it is mass-weighted only for `com` because
        `geometric.group_center` is the one implementation of a center here.

    The grouping is data rather than a mode. An atom-target wall has it taken
    away by the schema (with a note in the run record), so by the time this runs
    one atom per entry is all an atom target can mean.
    """
    indices = [int(position) for position in _members(spec, plan, coords)]
    target = spec.parameters.get('target_kind', 'atom')
    array = np.asarray(coords, dtype=np.float64)
    if target == 'atom':
        return [([position], np.ones(1), array[position]) for position in
                indices]
    groups = spec.parameters.get('molecules') or [indices]
    weighted = masses if target == 'com' else None
    objects = []
    for group in groups:
        members = [int(position) for position in group]
        point, _, weights = geometric.group_center(array, members, masses=weighted)
        objects.append((members, weights, point))
    return objects


def _axis_scales(geometry, parameters, name):
    """The soft law's length per axis, in Bohr.

    A sphere has one radius in every direction, an ellipsoid its semiaxes, and a
    box half of each edge - the extent an atom is allowed before the face. The
    cavity spelled that half edge as a radius, so a wall whose `lengths` are full
    edges springs at twice the intended distance if the halving is forgotten
    here. A triclinic cell is refused because its faces are sheared, which gives
    no axis scale to divide by.
    """
    if geometry == 'sphere':
        return np.full(3, float(parameters['radius']), dtype=np.float64)
    if geometry == 'ellipsoid':
        return np.asarray(parameters['semiaxes'], dtype=np.float64)
    if geometry == 'box':
        return 0.5 * np.asarray(parameters['lengths'], dtype=np.float64)
    raise WallError(
        "wall '%s' has geometry %r, which the 'soft' law does not define: the "
        'law needs an axis scale per direction, and a triclinic cell gives '
        'sheared faces instead' % (name, geometry))


def _ratio_terms(geometry, offset, parameters, name):
    """The soft law's argument at one point, with its derivative.

    `V = k * Q^(p/2)` with `Q` the squared distance in axis ratios: a sphere and
    an ellipsoid sum the axes into a single `Q`, which makes the potential one
    polynomial of the elliptical distance, while a box keeps them apart so each
    axis carries its own term. This is the archived cavity's law, and `Q` being
    squared is what makes it two-sided - a point pulled through the center pays
    the same energy on either side of it, which is the difference from a
    confining wall and the reason this law's constant carries no inverse length.
    """
    axes = _axis_scales(geometry, parameters, name)
    if np.any(axes <= 0.0):
        raise WallError("wall '%s' needs positive axis scales for the 'soft' "
                        'law' % name)
    scaled = np.asarray(offset, dtype=np.float64) / axes
    if geometry == 'box':
        terms = []
        for axis in range(3):
            derivative = np.zeros(3)
            derivative[axis] = 2.0 * float(scaled[axis]) / float(axes[axis])
            terms.append((float(scaled[axis] * scaled[axis]), derivative))
        return terms
    return [(float(np.sum(scaled * scaled)), 2.0 * scaled / axes)]


def _terms_energy(terms, constant, power, one_sided=True):
    """Sum the contributions of every face of one wall.

    A box and a triclinic cell meet an atom at up to three faces at once, and
    the potential is the sum over the faces that are crossed, so the corners
    are
    not special cases. Sphere and ellipsoid contribute one term.

    `one_sided` is the difference between the two laws, not a flag on the same
    one: a confining wall contributes nothing until its surface is crossed, and
    a centered spring contributes at every point but the center itself.

    Returns
    -------
    tuple
        (energy, gradient against the evaluated point, largest signed
        overshoot). `gradient` is None when nothing was crossed.
    """
    energy = 0.0
    gradient = None
    extent = None
    for distance, derivative in terms:
        extent = distance if extent is None else max(extent, distance)
        item, slope = _energy_of(distance, constant, power, one_sided)
        if item == 0.0:
            continue
        energy += item
        contribution = slope * np.asarray(derivative, dtype=np.float64)
        gradient = (contribution if gradient is None
                    else gradient + contribution)
    return energy, gradient, extent


def _energy_of(quantity, constant, power, one_sided=True):
    """The potential against one term and its derivative against that term.

    For the confining law the quantity is a signed overshoot in Bohr, clamped at
    zero, and the exponent is the power itself. For the soft law it is already
    a squared ratio and dimensionless, so nothing is clamped - the two-sidedness
    is what the square carries - and the exponent is half the power because
    `Q^(p/2)` is what the archived form wrote.
    """
    if constant == 0.0:
        return 0.0, 0.0
    if one_sided:
        overshoot = quantity if quantity > 0.0 else 0.0
        if overshoot == 0.0:
            return 0.0, 0.0
        return (float(constant * overshoot ** power),
                float(constant * power * overshoot ** (power - 1)))
    if quantity <= 0.0:
        # The center itself: no energy, and no force either for p > 2, while for
        # p = 2 the slope is the finite k rather than a 0/0 to be guessed at.
        if power == 2:
            return 0.0, float(constant)
        return 0.0, 0.0
    half = power / 2.0
    return (float(constant * quantity ** half),
            float(constant * half * quantity ** (half - 1.0)))


def _signed(geometry, offset, parameters, name):
    """Signed overshoots at one point, each with its derivative.

    Parameters
    ----------
    geometry : str
        'sphere', 'ellipsoid', 'box' or 'triclinic'.
    offset : numpy.ndarray
        Position relative to the wall center, shape (3,), Bohr.
    parameters : dict
        The converted geometry numbers.
    name : str
        Wall id, for error messages.

    Returns
    -------
    list of tuple
        (distance in Bohr, gradient of that distance against the offset) for
        every face the geometry has: one for a sphere or an ellipsoid, three
        for a box and three for a triclinic cell.
    """
    if geometry == 'sphere':
        radius = float(parameters['radius'])
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            # The center is a stable branch: the point is deep inside, the
            # potential is flat there, and any direction would be arbitrary.
            return [(-radius, np.zeros(3))]
        return [(length - radius, offset / length)]
    if geometry == 'ellipsoid':
        axes = np.asarray(parameters['semiaxes'], dtype=np.float64)
        scaled = offset / axes
        rho = float(np.linalg.norm(scaled))
        distance = float(axes.min()) * (rho - 1.0)
        if rho == 0.0:
            return [(distance, np.zeros(3))]
        derivative = scaled / rho / axes * float(axes.min())
        return [(distance, derivative)]
    if geometry == 'box':
        lengths = np.asarray(parameters['lengths'], dtype=np.float64)
        half = 0.5 * lengths
        terms = []
        for axis in range(3):
            distance = abs(float(offset[axis])) - float(half[axis])
            derivative = np.zeros(3)
            if offset[axis] != 0.0:
                derivative[axis] = 1.0 if offset[axis] > 0.0 else -1.0
            terms.append((distance, derivative))
        return terms
    if geometry == 'triclinic':
        return _triclinic(offset, parameters, name)
    raise WallError("wall '%s' has unknown geometry %r" % (name, geometry))


def _triclinic(offset, parameters, name):
    """Smallest overshoot past any face of a row-vector cell, and its gradient.

    Each of the three faces contributes, matching the box form so the two
    agree on a diagonal cell.
    """
    rows = parameters.get('cell')
    if rows is None:
        raise WallError("wall '%s' is triclinic but carries no cell matrix"
                        % name)
    lattice = np.asarray(rows, dtype=np.float64)
    try:
        inverse = np.linalg.inv(lattice)
    except np.linalg.LinAlgError:
        raise WallError("wall '%s' has a singular cell" % name)
    fractional = offset @ inverse
    terms = []
    for axis in range(3):
        normal = inverse[:, axis]
        # 1 / |column of B^-1| is the distance between the two faces the axis
        # is normal to, so the declared k keeps its physical meaning when the
        # cell shears instead of silently scaling with it.
        spacing = 1.0 / float(np.linalg.norm(normal))
        coordinate = float(fractional[axis])
        distance = (abs(coordinate) - 0.5) * spacing
        derivative = (1.0 if coordinate > 0.0 else
                      (-1.0 if coordinate < 0.0 else 0.0)) * normal * spacing
        terms.append((distance, derivative))
    return terms


def _curvature_terms(geometry, offset, parameters, name, one_sided=True):
    """Per face or per term: `(value, gradient, Hessian)` of that quantity.

    The gradient is `_signed`'s or `_ratio_terms`' own, so a face cannot report
    a force that is not the derivative of the quantity being differentiated twice
    here. A sphere and an ellipsoid curve; a box face and a triclinic face are
    absolute components of a linear map, so their second derivative is zero -
    genuinely zero, not unwritten.
    """
    if not one_sided:
        return _ratio_curvatures(geometry, offset, parameters, name)
    out = []
    for distance, gradient in _signed(geometry, offset, parameters, name):
        curve = np.zeros((3, 3), dtype=np.float64)
        if geometry == 'sphere':
            length = float(np.linalg.norm(offset))
            if length > 0.0:
                unit = offset / length
                curve = (np.eye(3) - np.outer(unit, unit)) / length
        elif geometry == 'ellipsoid':
            axes = np.asarray(parameters['semiaxes'], dtype=np.float64)
            scaled = offset / axes
            rho = float(np.linalg.norm(scaled))
            if rho > 0.0:
                direction = scaled / rho
                inverse = np.diag(1.0 / axes)
                # 1/rho is the derivative of the unit vector in the scaled
                # space: without it the ellipsoid's curvature is too large by
                # exactly that factor, which a crossed face shows at once.
                curve = (float(axes.min())
                         * (inverse @ (np.eye(3) - np.outer(direction,
                                                            direction))
                            @ inverse) / rho)
        elif geometry not in ('box', 'triclinic'):
            raise WallError("wall '%s' has unknown geometry %r"
                            % (name, geometry))
        out.append((distance, np.asarray(gradient, dtype=np.float64), curve))
    return out


def _ratio_curvatures(geometry, offset, parameters, name):
    """Per term of the soft law: `(Q, dQ, d2Q)`.

    Every term here is a squared axis ratio, so its Hessian is the constant
    `2 / a^2` down the diagonal and does not care where the point sits. That is
    what makes this law a spring rather than a boundary, and it is why a
    two-sided force needs no clamp to stay continuous.
    """
    axes = _axis_scales(geometry, parameters, name)
    if np.any(axes <= 0.0):
        raise WallError("wall '%s' needs positive axis scales for the 'soft' "
                        'law' % name)
    scaled = np.asarray(offset, dtype=np.float64) / axes
    # The gradient is `_ratio_terms`' own expression, written the same way rather
    # than folded into `2 / a^2 * scaled`, which is `2 d / a^3` and a different
    # thing: the folded form only shows up at p > 2, where the `J J^T` term is
    # not multiplied by a zero curvature.
    gradient_of = 2.0 * scaled / axes
    inverse_square = 2.0 / (axes * axes)
    if geometry == 'box':
        out = []
        for axis in range(3):
            gradient = np.zeros(3)
            gradient[axis] = float(gradient_of[axis])
            curve = np.zeros((3, 3))
            curve[axis, axis] = float(inverse_square[axis])
            out.append((float(scaled[axis] * scaled[axis]), gradient, curve))
        return out
    return [(float(np.sum(scaled * scaled)), gradient_of,
             np.diag(inverse_square))]


def _face_powers(quantities, constant, power, one_sided=True):
    """Per term, the `(slope, curvature)` of the potential against that term.

    For the confining law those are `k p d^(p-1)` and `k p (p-1) d^(p-2)` of a
    clamped overshoot; the soft law's are in `_soft_powers`. The names are
    deliberate and so is their use: the slope is what multiplies the term's own
    second derivative, the curvature what multiplies `J J^T`. Swapping them
    leaves a wall that is flat inside and wrong outside.

    The potential sums `max(d, 0)^p` over the faces, so an uncrossed one carries
    neither force nor curvature; raising a negative `d` to a power instead would
    invent curvature where the wall is flat, and for even `p` even the sign.
    """
    if not one_sided:
        return _soft_powers(quantities, constant, power)
    out = []
    for quantity in quantities:
        overshoot = float(quantity) if quantity > 0.0 else 0.0
        if overshoot <= 0.0 or constant == 0.0:
            out.append((0.0, 0.0))
            continue
        first = constant * power * overshoot ** (power - 1)
        second = (constant * power * (power - 1) * overshoot ** (power - 2)
                  if power > 1 else 0.0)
        out.append((float(first), float(second)))
    return out


def _soft_powers(quantities, constant, power):
    """Per term, `(slope, curvature)` of `k Q^(p/2)` against `Q`.

    `dV/dQ = (p/2) k Q^(p/2-1)` and `d2V/dQ2 = (p/2)(p/2-1) k Q^(p/2-2)`, with no
    clamp because the law contributes everywhere but at the center itself. At
    `Q = 0` the general formula would divide by zero, so the two cases are
    written out: `p = 2` keeps its finite slope `k` and has no curvature, and any
    higher power has neither - which is the truth of the potential, not an
    approximation of it, since `V = k (r/a)^(2m)` has both `dV/dr` and `d2V/dr2`
    vanishing at the center for `m > 1`.
    """
    exponent = power / 2.0
    out = []
    for quantity in quantities:
        value = float(quantity)
        if constant == 0.0:
            out.append((0.0, 0.0))
        elif value <= 0.0:
            out.append((float(constant) if power == 2 else 0.0, 0.0))
        else:
            first = constant * exponent * value ** (exponent - 1.0)
            second = (constant * exponent * (exponent - 1.0)
                      * value ** (exponent - 2.0) if exponent > 1.0 else 0.0)
            out.append((float(first), float(second)))
    return out


def hessian(spec, plan, coords, state, context=None, masses=None):
    """The exact `d2V/dR2` of one wall, as a dense (3N, 3N) block.

    For `V = k * max(d, 0)^p` on a face's signed distance `d` (no half factor -
    the form this module fixes), a crossed face contributes

        k p (p-1) d^(p-2) J J^T  +  k p d^(p-1) Q

    where `J` is that distance's gradient chained to the atoms and `Q` its own
    second derivative. The soft law is the same expression with `max(d, 0)`
    replaced by the dimensionless `Q` of `_ratio_terms` and the exponents halved.
    A centroid wall is one more evaluation rather than a second implementation:
    the chain puts the group weight on every row and column, and the centroid's
    second derivative is the weight outer product times the face's curvature.
    """
    if spec.kind != 'wall':
        raise WallError("hessian() got a '%s' action, not a wall" % spec.kind)
    parameters = schedule.resolve_parameters(spec, context, spec.parameters)
    geometry = str(parameters.get('geometry', spec.family))
    power = int(parameters.get('power', 2))
    if power < 2:
        raise WallError(
            "wall '%s' power must be at least 2 so the force is continuous"
            % spec.id)
    constant = float(parameters.get('force_constant', 0.0))
    law = str(spec.parameters.get('law', 'hard'))
    one_sided = law != 'soft'
    array = np.asarray(coords, dtype=np.float64)
    natom = array.shape[0]
    block = np.zeros((natom * 3, natom * 3), dtype=np.float64)
    center = np.asarray(_center_of(spec, parameters, state, array, natom),
                        dtype=np.float64)
    for members, weights, point in _objects(spec, plan, array, masses):
        offset = np.asarray(point, dtype=np.float64) - center
        terms = _curvature_terms(geometry, offset, parameters, spec.id,
                                 one_sided=one_sided)
        powers = _face_powers([term[0] for term in terms], constant, power,
                              one_sided=one_sided)
        shares = {int(atom): float(weight)
                  for atom, weight in zip(members, weights)}
        for (_distance, gradient, curve), (slope, curvature) in zip(terms,
                                                                    powers):
            if slope == 0.0 and curvature == 0.0:
                continue
            flat = np.zeros(natom * 3, dtype=np.float64)
            for atom, weight in shares.items():
                flat[atom * 3:atom * 3 + 3] = weight * gradient
            block += curvature * np.outer(flat, flat)
            if slope != 0.0:
                for left, left_weight in shares.items():
                    for right, right_weight in shares.items():
                        block[left * 3:left * 3 + 3,
                              right * 3:right * 3 + 3] += (
                                  slope * left_weight * right_weight * curve)
    return {'kind': 'wall', 'family': geometry, 'id': spec.id,
            'target_kind': spec.parameters.get('target_kind', 'atom'),
            'law': law, 'size': natom * 3, 'hessian': block}


def _members(spec, plan, coords):
    """0-based atoms the wall acts on.

    A wall that names nothing acts on the whole system, which is what the
    archived cavity did and what `wall_groups` then subdivides. A record cannot
    carry an empty selection at all - `ConstraintSpec` refuses one - so the only
    way to nothing is to name nothing.
    """
    if spec.atoms is not None:
        return [int(position) for position in spec.atoms]
    if spec.group is not None:
        return [int(position) for position in plan.groups.members(spec.group)]
    return list(range(int(np.asarray(coords).shape[0])))


def _center_atoms(spec, plan, coords):
    """The atoms a wall's center is taken from: its `wall_ref`, else its own."""
    anchor = spec.parameters.get('anchor')
    if anchor:
        return [int(position) for position in anchor]
    return _members(spec, plan, coords)


def bind_center(spec, plan, coords, masses=None):
    """The center this wall is fixed to, captured at the reference geometry.

    In order of priority: the declared coordinates of `wall_center`, else the
    center of the atoms `wall_ref` names, else the center of the wall's own
    selection - the whole system when it names none. `com` weights that center by
    mass and every other target by count, which is the same rule `_objects` uses
    for the points being confined, so a wall and its center never disagree about
    what a center is.

    The value is bound once and carried in the run state: recomputing it from the
    live geometry would let the boundary follow the molecules it confines, and a
    restart would read its own drift back as the wall's anchor.
    """
    parameters = spec.parameters or {}
    center = parameters.get('center')
    if center is not None:
        return np.asarray(center, dtype=np.float64)
    array = np.asarray(coords, dtype=np.float64)
    indices = _center_atoms(spec, plan, array)
    weighted = parameters.get('target_kind') == 'com'
    values = (np.asarray(masses, dtype=np.float64).reshape(-1)
              if weighted else None)
    if weighted and values is None:
        raise WallError("wall '%s' targets 'com' but no masses were given"
                        % spec.id)
    point, _members_of_group, _weights = geometric.group_center(
        array, indices, masses=values)
    return point


def _center_of(spec, parameters, state, coords, natom):
    """Wall center: the declared point, or the one bound at the reference.

    One center per wall, captured at binding time and kept under 'center:<id>'
    in the state anchors, so two walls over different groups do not fight over
    one slot. What gets captured is `bind_center`'s rule; no case here is the
    coordinate origin, which is what made an atom-target wall mean "confine
    around (0,0,0)" while its input said nothing about a center.
    """
    center = parameters.get('center')
    if center is not None:
        return np.asarray(center, dtype=np.float64)
    anchors = getattr(state, 'anchors', None) or {}
    stored = anchors.get('center:%s' % spec.id)
    if stored is not None:
        return np.asarray(stored, dtype=np.float64)
    raise WallError(
        "wall '%s' has no center: bind the reference geometry first, or "
        "declare 'wall_center' / 'wall_ref'" % spec.id)



# ---------------------------------------------------------------------------
# A wall inside a periodic cell
#
# A wall is an external boundary and a periodic cell is an instruction to
# replicate it. Confining a molecule between two walls that repeat is only
# meaningful while the wall surface and its own periodic copy are distinct
# objects: a wall that touches or crosses a cell face has no shell between it and
# that face, so an atom that leaves the wall enters the shell and comes out
# already inside the wall's copy on the far side, with the barrier it was supposed
# to climb removed by the periodicity itself.
#
# When the wall surface lies strictly inside the cell and the shell (surface to
# face) has a nonzero thickness, an atom cannot escape: leaving the wall means
# entering the shell, and inside the shell the force points back toward the wall.
# The one way left open is a single step that jumps the whole shell and lands
# inside the wall's copy, which is what the thickness lower bound closes - the
# bound is expressed in the largest displacement one step can carry, because
# `dt`, the velocities and the masses are all available.
#
# The four checks below are input-time refusals, each naming the axis and the
# number that failed. They need the cell, the center and the size: no minimum
# image, no bond graph. A centroid wall is the exception, and its center is the
# one quantity a wrapped coordinate cannot supply (an average of 0.01 and L-0.01
# is L/2), so it goes through the unfolded geometry like any other multi-atom
# variable.
# ---------------------------------------------------------------------------

# How many times the largest single-step displacement a shell must be to stop a
# one-step jump across it. Two leaves room for a step that lands at the outer
# edge and the next one heading in from there.
SHELL_STEPS = 2.0


def _spread_along_axis(parameters, geometry, direction, name, spacing):
    """The wall's extent along one cell normal, in the same length unit."""
    unit = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise WallError("wall '%s' has a degenerate cell axis" % name)
    unit = unit / norm
    if geometry == 'sphere':
        return float(parameters['radius'])
    if geometry == 'ellipsoid':
        axes = np.asarray(parameters['semiaxes'], dtype=np.float64)
        return float(np.sqrt(np.sum((axes * unit) ** 2)))
    if geometry == 'box':
        half = 0.5 * np.asarray(parameters['lengths'], dtype=np.float64)
        return float(np.sum(half * np.abs(unit)))
    if geometry == 'triclinic':
        # The wall's own basis: a point of the wall is center + sum(t_k w_k) with
        # each |t_k| at most a half, so its reach along a direction is the sum of
        # the half-edges projected on that direction.
        rows = parameters.get('cell')
        if rows is None:
            raise WallError("wall '%s' is triclinic but carries no cell matrix"
                            % name)
        matrix = np.asarray(rows, dtype=np.float64).reshape(3, 3)
        return 0.5 * float(np.sum(np.abs(matrix @ unit)))
    raise WallError("wall '%s' has unknown geometry %r" % (name, geometry))


def periodic_check(spec, plan, coords, cell, center=None, dt=0.0,
                   velocities=None, parameters=None, check_members=True):
    """Refuse a periodic wall that cannot confine anything.

    Parameters
    ----------
    spec : ConstraintSpec
        The wall declaration.
    plan : ConstraintPlan
    coords : numpy.ndarray
        (natom, 3) in Bohr; the members' positions feed the "starts inside" check.
    cell : numpy.ndarray
        (3, 3) row-vector lattice in Bohr.
    center : numpy.ndarray, optional
        The wall center in Bohr. When it is not given the check reads the
        declared one, and a wall with no declared center cannot be placed in a
        cell, which is itself a refusal.
    dt : float
        Timestep in atomic units, for the shell thickness bound.
    velocities : numpy.ndarray, optional
        (natom, 3) Bohr per atomic time. The largest speed the run carries sets
        how far one step can push an atom; without velocities only the geometric
        containment is checked.
    parameters : dict, optional
        The geometry to measure instead of the declared one. A wall whose radius
        or cell is ramped has to fit the cell at every value the ramp reaches,
        not just at the one the run starts with; `delay` and `stop` say when a
        value is reached, not which values exist, so the caller walks the
        endpoints and this argument is how one arrives.

    Raises
    ------
    WallError
        Any of the four checks fails. The message names the check, the axis and
        the difference.
    """
    parameters = dict(spec.parameters or {}) if parameters is None \
        else dict(parameters)
    geometry = str(parameters.get('geometry', spec.family))
    if str(spec.parameters.get('law', 'hard')) == 'soft':
        raise WallError(
            "wall '%s' uses the 'soft' law inside a periodic cell: it is a "
            'spring toward one point rather than a barrier, so there is no shell '
            'to check and nothing confines. Worse, a wrapped coordinate jumps its '
            'offset by a whole lattice vector every time an atom crosses a face, '
            'which is a discontinuous force; the minimum image of a spring is a '
            'different object and is not implemented here')
    matrix = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise WallError("wall '%s': the cell is singular" % spec.id)
    if center is None:
        center = parameters.get('center')
    if center is None:
        raise WallError(
            "wall '%s' runs in a periodic cell without a declared center; "
            'the cell cannot tell where an external boundary belongs' % spec.id)
    point = np.asarray(center, dtype=np.float64).reshape(3)
    fractional = point @ inverse
    spacing = []
    spread = []
    for axis in range(3):
        column = inverse[:, axis]
        length = float(np.linalg.norm(column))
        spacing.append(1.0 / length)
        spread.append(_spread_along_axis(parameters, geometry, column, spec.id,
                                         spacing[-1]) * length)
    reach = 0.0
    if velocities is not None and dt:
        speeds = np.linalg.norm(np.asarray(velocities, dtype=np.float64),
                                axis=1)
        reach = float(dt * speeds.max()) * SHELL_STEPS if speeds.size else 0.0
    gaps = []
    for axis in range(3):
        offset = abs(float(fractional[axis]) - 0.5)
        margin = 0.5 - offset - spread[axis]
        if margin < 0.0:
            raise WallError(
                "wall '%s' does not fit the cell: along axis %s the wall "
                'surface reaches %.4f fractional units past the cell edge '
                '(center offset %.4f plus extent %.4f against a half cell of '
                '0.5); a periodic wall must lie strictly inside its cell'
                % (spec.id, 'abc'[axis], offset + spread[axis] - 0.5,
                   offset, spread[axis]))
        gap = margin * spacing[axis]
        if gap <= 0.0:
            raise WallError(
                "wall '%s' touches the cell face along axis %s: the shell "
                'between wall and face has zero thickness, so an atom that '
                'leaves the wall appears inside its periodic copy and the '
                'barrier is removed by the periodicity'
                % (spec.id, 'abc'[axis]))
        if reach > 0.0 and gap < reach:
            raise WallError(
                "wall '%s': the shell along axis %s is %.4f Bohr, thinner than "
                '%.4f Bohr, which is what one step at dt=%.4f and the largest '
                'speed in the run can carry (x%.1f); a step could jump the '
                'shell and land inside the wall copy'
                % (spec.id, 'abc'[axis], gap, reach, float(dt), SHELL_STEPS))
        gaps.append(gap)
    if not check_members:
        # A wall whose surface moves during a ramp is checked against every
        # value it reaches for fitting the cell, but an atom being outside is
        # only a statement about the geometry the run starts with: an atom
        # outside the *closed* size is what closing the wall means, not an
        # external field.
        return {'gaps': gaps, 'spacing': spacing, 'spread': spread}
    members = _members(spec, plan, coords)
    if spec.parameters.get('target_kind', 'atom') == 'atom':
        outside = []
        for position in members:
            offset = np.asarray(coords[position], dtype=np.float64) - point
            extents = [value for value, _row in
                       _signed(geometry, offset, parameters, spec.id)]
            if max(extents) >= 0.0:
                outside.append((position, max(extents)))
        if outside:
            position, amount = outside[0]
            raise WallError(
                "wall '%s' binds atom %d while it sits %.4f Bohr outside the "
                'surface; a wall in a periodic cell is confinement, and an atom '
                'that starts outside is being held out of the wall, which is an '
                'external field and needs that axis pbc=0'
                % (spec.id, position + 1, amount))
    return {'gaps': gaps, 'spacing': spacing, 'spread': spread}
