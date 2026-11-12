"""
======================================================================
Frozen components, condition rank and free degrees of freedom.

Three things happen here, and they are kept apart on purpose.

Fixing. A fix record freezes chosen Cartesian components at their anchor. The
anchor is captured once at the reference geometry and stored in the state, so a
restored run freezes the same position rather than wherever the atom happened
to be. Freezing is done by removing the component from the active set and
restoring the anchor value, never by inflating the mass.

Ranking. The independent linear velocity conditions of a structure are the
frozen components plus the Jacobian rows of every hard bond constraint, scaled
by the inverse square root of the masses, plus the center-of-mass rows if and
only if the caller says a center-of-mass operation is actually performed. The
rank of that matrix is the number of degrees of freedom that are not free.
Counting constraint lines instead of the rank double-counts a redundant
constraint: two atoms both frozen and held at their own distance add nothing.

Temperature. The free kinetic energy is computed from the tangential part of
the velocity, so the reported temperature belongs to the degrees of freedom
that exist. That projection is a diagnostic: it never writes back. Velocity
constraint inside the integrator is RATTLE, and RATTLE is out of scope for this
build, which is why the position solve in shake.py leaves velocities alone.

The current framework formula is 3 * (natom - n_frozen) - 3 with the result
clamped to at least one. This module reproduces it exactly when only a fix is
declared and the center of mass is removed, and then drops the clamp, because a
fully frozen system has no temperature to report rather than one degree of
freedom worth of noise.

The module must stay a leaf: numpy and the lab records.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.data.constraint import ConstraintStatistics, build_fix_mask

# Numerical rank tolerance for the condition matrix, relative to its largest
# singular value. A constraint that is redundant to this depth is a duplicate,
# not a nearly-independent condition.
RANK_TOLERANCE = 1.0e-8


def fix_anchors(coords, mask):
    """Anchor array holding the reference value wherever the mask is set."""
    array = np.asarray(coords, dtype=np.float64)
    anchors = np.zeros_like(array)
    anchors[mask] = array[mask]
    return anchors


def apply_fix_positions(coords, anchors, mask):
    """Restore every frozen component to its anchor.

    Returns
    -------
    tuple
        (corrected coordinates as a fresh array, number of components moved).
        A component already at its anchor is not counted, which is what makes
        the count a drift measure rather than a size measure.
    """
    array = np.asarray(coords, dtype=np.float64)
    corrected = array.copy()
    moved = int(np.count_nonzero(mask & (corrected != np.asarray(anchors))))
    corrected[mask] = np.asarray(anchors)[mask]
    return corrected, moved


def zero_fixed_velocities(velocities, mask):
    """Zero the velocity of every frozen component.

    A frozen coordinate with a nonzero velocity is a discontinuity the next
    position restore would have to paper over, so this runs whenever the
    velocities are produced or changed.
    """
    array = np.asarray(velocities, dtype=np.float64)
    corrected = array.copy()
    corrected[mask] = 0.0
    return corrected


def tangent_split(gradient, coords, mask=None, conditions=None, pairs=None,
                  tolerance=RANK_TOLERANCE):
    """Split one gradient into the part a constrained run is judged on.

    A strict condition holds the geometry on a curved manifold. At a point of
    that manifold the gradient of the objective need not vanish along the
    condition's own direction: the condition supplies that force, and a threshold
    applied to it is unreachable by construction. The previous reading of this
    removed every atom sitting in a condition from the count, which threw away
    the tangential half of those atoms too. The directional split is what the
    statement actually means: the component of the gradient along each condition
    normal is that condition's Lagrange multiplier, and what is left is the
    gradient of the objective the run is minimizing.

    The rows here are the *unweighted* Jacobians. A gradient is a covector, and
    the metric it must be split with is the Cartesian one the step is measured
    in; the mass weighting `condition_matrix` applies is a per-component column
    scale, which is right for velocities, where the metric is the kinetic one,
    and would rotate the normal space of a gradient.

    Parameters
    ----------
    gradient : array-like
        (natom, 3) or (3 * natom,) gradient in Hartree per Bohr, taken at the
        geometry the evaluation describes.
    coords : array-like
        (natom, 3) that same geometry, in Bohr.
    mask : array-like, optional
        (natom, 3) frozen components. They come back zeroed and carry no row:
        a frozen component is not a degree of freedom the run is judged on.
    conditions : list, optional
        (kind, atoms) per strict condition in plan order, the same shape
        `condition_matrix` takes. It takes precedence over `pairs`, because a
        plan that holds a length and an angle is one manifold.
    pairs : array-like, optional
        (n, 2) bond pairs, for a plan that holds lengths only.
    tolerance : float
        Relative singular-value cut, shared with `condition_rank`.

    Returns
    -------
    tuple
        (tangent, multipliers, rank). `tangent` is the flat (3 * natom,) gradient
        with the frozen components zeroed and every condition normal component
        removed. `multipliers` carries one entry per condition row in the order
        the rows were given: Hartree per Bohr for a length, Hartree per radian
        for an angle. Where rows are not independent the multipliers are the
        minimum-norm solution, which is the honest reading of a redundant set,
        and `rank` says how many independent conditions the set really holds.
    """
    vector = np.asarray(gradient, dtype=np.float64).reshape(-1)
    array = np.asarray(coords, dtype=np.float64)
    natom = array.shape[0]
    if vector.size != 3 * natom:
        raise ValueError(
            'gradient holds %d components for %d atoms'
            % (vector.size, natom))
    flat = vector.copy()

    flags = np.zeros((natom, 3), dtype=bool)
    if mask is not None:
        flags = np.asarray(mask, dtype=bool)
        if flags.shape != (natom, 3):
            raise ValueError(
                'fix mask must be (%d, 3), got %s' % (natom, flags.shape))
        flat[_frozen_components(flags)] = 0.0
    components = np.column_stack(
        (np.repeat(np.arange(natom), 3), np.tile(np.arange(3), natom)))
    mobile = components[~flags.reshape(-1)]

    rows = _unweighted_condition_rows(array, mobile, conditions=conditions,
                                      pairs=pairs)
    positions = mobile[:, 0] * 3 + mobile[:, 1]
    if rows.shape[0] == 0:
        return flat, np.zeros(0), 0

    # One SVD carries the split and the multipliers: `right` is an orthonormal
    # basis of the condition normals, so the normal component of the gradient is
    # its own projection, and lambda = U diag(1/sigma) V^T g solves rows^T lambda
    # = normal with the smallest norm, which is what a redundant set means.
    left, values, right = np.linalg.svd(rows, full_matrices=False)
    if values.size == 0 or float(values[0]) <= 0.0:
        return flat, np.zeros(rows.shape[0]), 0
    keep = values > tolerance * float(values[0])
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return flat, np.zeros(rows.shape[0]), 0

    mobile_gradient = flat[positions]
    coefficients = right[:rank] @ mobile_gradient
    tangent = flat.copy()
    tangent[positions] = mobile_gradient - right[:rank].T @ coefficients
    multipliers = left[:, :rank] @ (coefficients / values[:rank])
    return tangent, multipliers, rank


def _unweighted_condition_rows(coords, mobile, conditions=None, pairs=None):
    """(rows, 3 * mobile) Jacobian of the declared conditions, unscaled."""
    rows = []
    if conditions is not None and len(conditions):
        for item in conditions:
            rows.append(condition_row_for(coords, item[0], item[1], mobile))
    elif pairs is not None and len(pairs):
        for pair in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
            rows.append(_bond_row(coords, pair, mobile))
    if not rows:
        return np.zeros((0, mobile.shape[0]))
    return np.vstack(rows)


def condition_row_for(array, kind, atoms, mobile):
    """One angle or torsion condition row over the mobile components.

    The Jacobian comes from `utilities.cv`, the same single implementation the
    position solve uses; this only drops the columns the freeze removed, exactly
    as `_bond_row` does for a pair.
    """
    from pyrai3md.utilities import cv

    indices = [int(atom) for atom in atoms]
    if kind == 'bond':
        _value, rows = cv.distance(array, indices[0], indices[1], grad=True)
    elif kind == 'angle':
        _value, rows = cv.angle(array, indices[0], indices[1], indices[2],
                                grad=True)
    elif kind == 'dihedral':
        _value, rows = cv.dihedral(array, indices[0], indices[1], indices[2],
                                   indices[3], grad=True)
    else:
        raise ValueError("unknown condition kind %r" % kind)
    by_atom = {}
    for position, (atom, direction) in enumerate(mobile):
        by_atom.setdefault(int(atom), {})[int(direction)] = position
    row = np.zeros(mobile.shape[0])
    for position, atom in enumerate(indices):
        places = by_atom.get(int(atom), {})
        for direction in range(3):
            slot = places.get(direction)
            if slot is not None:
                row[slot] = rows[position, direction]
    return row


def cell_virial(coords, fractional, multipliers, mask=None, conditions=None,
                pairs=None):
    """The (3, 3) cell force the declared conditions supply.

    A condition holds a function of the geometry, and the geometry depends on the
    box: under the convention a variable-cell step takes - atoms ride the box, so
    the fractional coordinates are what is held (`optimizer/base.py::_cell_ride`)
    - a lattice entry moves atom i by `s_i` times that entry,

        d r_i[b] / d H[a, b] = s_i[a],

    and a condition therefore has a lattice derivative of its own,

        d g / d H[a, b] = sum_i (d g / d r_i[b]) s_i[a].

    This is that sum, weighted by the Lagrange multiplier each condition absorbs,
    and it is the term the cell force is missing when a run declares a condition:
    the engine reports the stress of the potential, which knows nothing about a
    length something else is holding. Which sign it enters the cell force with is
    a question the finite-difference criterion answers
    (`qualification/verification/test_constraint_cell_coupling.py`), not one this
    function decides.

    Only the components a step can actually move are summed. A frozen atom does
    not ride - the position solve pulls it back to its anchor, which is stored in
    absolute Cartesian and does not move with the box (#276 / D-25 on the dynamics
    side) - so its share of the condition's lattice derivative is not a force on
    the box. That is the same mobile set `tangent_split` splits the gradient on,
    which is why the rows here and the multipliers handed in describe one another.

    Parameters
    ----------
    coords : array-like
        (natom, 3) geometry the conditions were measured on, in Bohr, unfolded
        when the run is periodic.
    fractional : array-like
        (natom, 3) fractional coordinates of that same geometry in that same box.
    multipliers : array-like
        One Lagrange multiplier per condition row, in plan order, as
        `tangent_split` returns them: Hartree per Bohr for a length, Hartree per
        radian for an angle or a torsion.
    mask : array-like, optional
        (natom, 3) frozen components, the same mask `tangent_split` was given.
    conditions : list, optional
        (kind, atoms) per condition in plan order.
    pairs : array-like, optional
        (n, 2) bond pairs, for a plan that holds lengths only.

    Returns
    -------
    numpy.ndarray
        (3, 3) tensor in Hartree per Bohr of lattice entry, zeros when the plan
        declares no condition.
    """
    output = np.zeros((3, 3), dtype=np.float64)
    lambdas = np.asarray(multipliers, dtype=np.float64).reshape(-1)
    if lambdas.size == 0:
        return output
    array = np.asarray(coords, dtype=np.float64)
    natom = array.shape[0]
    frac = np.asarray(fractional, dtype=np.float64)
    if frac.shape != (natom, 3):
        raise ValueError(
            'fractional coordinates must be (%d, 3), got %s'
            % (natom, frac.shape))

    flags = np.zeros((natom, 3), dtype=bool)
    if mask is not None:
        flags = np.asarray(mask, dtype=bool)
        if flags.shape != (natom, 3):
            raise ValueError(
                'fix mask must be (%d, 3), got %s' % (natom, flags.shape))
    components = np.column_stack(
        (np.repeat(np.arange(natom), 3), np.tile(np.arange(3), natom)))
    mobile = components[~flags.reshape(-1)]

    rows = _unweighted_condition_rows(array, mobile, conditions=conditions,
                                      pairs=pairs)
    if rows.shape[0] == 0:
        return output
    if rows.shape[0] != lambdas.size:
        raise ValueError(
            '%d multipliers for %d condition rows'
            % (lambdas.size, rows.shape[0]))

    # Contract each row with the fractional coordinate of the component it
    # belongs to: column k of a row is d g / d r_{i(k), b(k)}, which feeds the
    # lattice entry (a, b(k)) weighted by s_{i(k), a}. Both the conditions and the
    # components sharing an axis are summed, so one axis of the output is a scalar
    # per lattice row.
    atoms = mobile[:, 0]
    axes = mobile[:, 1]
    scale = frac[atoms]                       # (n_mobile, 3), the a index
    weighted = rows * lambdas[:, None]        # lambda_c d g_c / d r
    for axis in range(3):
        columns = axes == axis
        if not np.any(columns):
            continue
        output[:, axis] = (weighted[:, columns]
                           @ scale[columns, :]).sum(axis=0)
    return output


def condition_matrix(masses, mask=None, pairs=None, coords=None,
                     remove_com=False, conditions=None):
    """Mass-weighted matrix of the independent linear velocity conditions.

    Parameters
    ----------
    masses : array-like
        Per-atom masses, positive, shape (natom,).
    mask : array-like, optional
        (natom, 3) frozen-component mask. Frozen columns are dropped from the
        matrix, so they contribute through the mobile-component count instead
        of as rows here.
    pairs : array-like, optional
        (n, 2) 0-based atom pairs of hard bond constraints. Each adds one row
        of the bond Jacobian, which is the velocity condition |r_ij| = const.
    coords : numpy.ndarray, optional
        Required with `pairs`, since the Jacobian depends on the geometry.
    remove_com : bool
        Add the three center-of-mass rows. Only set when the workflow really
        removes the center-of-mass drift; an external wall or an anchor breaks
        that symmetry, and then removing the degrees of freedom would be a
        guess.

    Returns
    -------
    tuple
        (matrix, mobile component indices). The matrix has as many columns as
        there are entries in the index array, so the caller can map a null
        space back to Cartesian components.
    """
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    natom = values.size
    if natom == 0:
        raise ValueError('condition matrix needs at least one atom')
    if np.any(values <= 0.0):
        raise ValueError(
            'every mass must be positive before a velocity condition can be '
            'weighted')
    flags = np.zeros((natom, 3), dtype=bool)
    if mask is not None:
        flags = np.asarray(mask, dtype=bool)
        if flags.shape != (natom, 3):
            raise ValueError(
                'fix mask must be (%d, 3), got %s' % (natom, flags.shape))
    components = np.column_stack(
        (np.repeat(np.arange(natom), 3), np.tile(np.arange(3), natom)))
    mobile = components[~flags.reshape(-1)]
    scale = 1.0 / np.sqrt(values[mobile[:, 0]])
    rows = []
    if conditions is not None and len(conditions):
        if coords is None:
            raise ValueError('a condition row needs the geometry')
        array = np.asarray(coords, dtype=np.float64)
        for item in conditions:
            kind, atoms = item[0], item[1]
            rows.append(condition_row_for(array, kind, atoms, mobile) * scale)
    elif pairs is not None and len(pairs):
        if coords is None:
            raise ValueError('a bond condition needs the geometry')
        array = np.asarray(coords, dtype=np.float64)
        for pair in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
            row = _bond_row(array, pair, mobile)
            rows.append(row * scale)
    if remove_com:
        for axis in range(3):
            row = np.zeros(mobile.shape[0])
            for position, (atom, direction) in enumerate(mobile):
                if direction == axis:
                    row[position] = np.sqrt(values[atom])
            rows.append(row * scale)
    if not rows:
        return np.zeros((0, mobile.shape[0])), mobile
    matrix = np.vstack(rows)
    keep = np.linalg.norm(matrix, axis=1) > 0.0
    return matrix[keep], mobile


def condition_rank(matrix, tolerance=RANK_TOLERANCE):
    """Numerical rank of a condition matrix from its singular values."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.size == 0 or array.shape[0] == 0:
        return 0
    values = np.linalg.svd(array, compute_uv=False)
    top = float(values[0]) if values.size else 0.0
    if top <= 0.0:
        return 0
    return int(np.count_nonzero(values > tolerance * top))


def degrees_of_freedom(masses, mask=None, pairs=None, coords=None,
                       remove_com=False, tolerance=RANK_TOLERANCE,
                       conditions=None):
    """Free degrees of freedom of the mobile, unconstrained components.

    Returns
    -------
    ConstraintStatistics
        `mobile_components`, `rank`, `com_removed` and `free_dof`, with
        `free_dof` allowed to be zero.
    """
    matrix, mobile = condition_matrix(masses, mask=mask, pairs=pairs,
                                      coords=coords, remove_com=remove_com,
                                      conditions=conditions)
    rank = condition_rank(matrix, tolerance=tolerance)
    com_rows = 3 if remove_com else 0
    statistics = ConstraintStatistics(
        mobile_components=int(mobile.shape[0]), rank=rank,
        com_removed=min(com_rows, max(0, int(mobile.shape[0]) - rank)))
    statistics.free_dof = int(mobile.shape[0] - rank)
    return statistics


def tangent_velocities(velocities, masses, matrix, mobile, mask=None):
    """Velocity with the frozen and constrained directions projected out.

    The projection runs in mass-weighted coordinates, where the kinetic energy
    is a plain squared norm and the condition matrix acts on the same basis.
    Frozen components are zeroed first, then the mobile part is projected onto
    the null space of the conditions.

    Diagnostic only: it does not enforce a velocity constraint inside the
    integrator, which is RATTLE and out of scope here.
    """
    array = np.asarray(velocities, dtype=np.float64)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    flat = array.reshape(-1).copy()
    if mask is not None:
        flat[_frozen_components(mask)] = 0.0
    if not matrix.size or not matrix.shape[0]:
        return flat.reshape(array.shape)
    positions = mobile[:, 0] * 3 + mobile[:, 1]
    weights = np.sqrt(values[mobile[:, 0]])
    weighted = flat[positions] / weights
    projected = _null_space_projector(matrix, weighted)
    flat[positions] = projected * weights
    return flat.reshape(array.shape)


def free_kinetic_energy(velocities, masses, mask=None, pairs=None,
                        coords=None, remove_com=False, conditions=None):
    """Kinetic energy over the free degrees of freedom, in Hartree.

    Falls back to the plain sum over mobile components when there is no hard
    constraint, since then the tangential part is the whole thing.
    """
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    array = np.asarray(velocities, dtype=np.float64)
    if array.shape != (values.size, 3):
        raise ValueError(
            'velocity array must be (%d, 3), got %s'
            % (values.size, array.shape))
    matrix, mobile = condition_matrix(values, mask=mask, pairs=pairs,
                                      coords=coords, remove_com=remove_com,
                                      conditions=conditions)
    kinetic = 0.0
    weights = values[:, None] * array ** 2
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), 0.0, weights)
    kinetic = 0.5 * float(np.sum(weights))
    if not matrix.size or not matrix.shape[0]:
        return kinetic, mobile
    tangent = tangent_velocities(array, values, matrix, mobile, mask=mask)
    return 0.5 * float(np.sum(values[:, None] * tangent ** 2)), mobile


def statistics(masses, velocities=None, mask=None, pairs=None, coords=None,
               remove_com=False, tolerance=RANK_TOLERANCE, conditions=None):
    """Free degrees of freedom and free kinetic energy in one record."""
    result = degrees_of_freedom(masses, mask=mask, pairs=pairs, coords=coords,
                                remove_com=remove_com, tolerance=tolerance,
                                conditions=conditions)
    if velocities is not None:
        kinetic, _ = free_kinetic_energy(
            velocities, masses, mask=mask, pairs=pairs, coords=coords,
            remove_com=remove_com, conditions=conditions)
        result.free_kinetic = float(kinetic)
    return result




def _bond_row(coords, pair, mobile):
    """One bond Jacobian row over the mobile components."""
    left, right = int(pair[0]), int(pair[1])
    vector = coords[left] - coords[right]
    length = float(np.linalg.norm(vector))
    row = np.zeros(mobile.shape[0])
    if length == 0.0:
        return row
    unit = vector / length
    lookup = {(int(atom), int(axis)): position
              for position, (atom, axis) in enumerate(mobile)}
    for axis in range(3):
        position = lookup.get((left, axis))
        if position is not None:
            row[position] += unit[axis]
        position = lookup.get((right, axis))
        if position is not None:
            row[position] -= unit[axis]
    return row


def _frozen_components(mask):
    """Flat Cartesian component indices where a (natom, 3) mask is set."""
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    return np.flatnonzero(flags)


def _null_space_projector(matrix, vector):
    """Project a mass-weighted velocity onto the null space of the conditions.

    The conditions span a row space; the component of the velocity inside that
    row space is what a constraint forbids. Taking the singular vectors of the
    condition matrix makes the split explicit and survives a rank-deficient
    matrix, which is a duplicate condition rather than an error, and which
    `condition_rank` has already reported.
    """
    if not np.count_nonzero(vector):
        return vector
    _, values, vectors = np.linalg.svd(np.asarray(matrix, dtype=np.float64),
                                       full_matrices=False)
    top = float(values[0]) if values.size else 0.0
    if top <= 0.0:
        return vector
    rank = int(np.count_nonzero(values > RANK_TOLERANCE * top))
    if rank == 0:
        return vector
    directions = vectors[:rank]
    return vector - directions.T.dot(directions.dot(vector))
