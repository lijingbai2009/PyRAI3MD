"""
======================================================================
Strict bond constraints solved on positions, with an opt-in velocity
projection.

A hard bond constraint removes degrees of freedom by making the geometry
satisfy |r_ij| = target. This module does that in the only way that is honest
without a velocity constraint: it iterates the positions until every declared
distance is met within a normalized tolerance, distributing each correction
over the two atoms in inverse proportion to their masses so no net displacement
is invented.

What this is not
----------------
It is not a time-reversible integrator. Reversible SHAKE derives the position
correction from the previous geometry and the half-step velocity through
discrete Lagrange impulses, and full RATTLE keeps `J v = 0` on the velocity as
part of the same impulse scheme. Neither is what the default path does here:
`solve_positions` leaves the velocities exactly as they arrived, and the caller
must not read that as "the velocity constraint was satisfied". The consequence
is now measured, not predicted
(`tests/test_shake_dynamics_cost.py`, design document section 1.3): the position
solve holds the bond length to 1e-12 relative at every timestep, while the total
energy of the constrained run drifts by over a hundred times the depth of the
potential driving it and does not fall when the step is halved, with `J v`
sitting near 1.2e-2 Bohr per atomic time instead of shrinking.
`solve_velocities` below is the half that was missing, reached only when a
constraint record declares `velocity_constraint: rattle`; the default `none`
takes no branch there, so what the measurement was made on has not moved. Degrees
of freedom are still counted from the position manifold, which is exactly what a
position constraint defines.

Failure is a failure
--------------------
When the iteration does not converge, the largest residual, its constraint and
the iteration count are reported. The solver never falls back to a soft spring,
because a silently softened constraint changes the physics while looking like
the intended run.

Redundancy
----------
Two atoms that are both frozen and held at their own anchor distance add no
condition: the constraint is already satisfied by construction and must not
cost degrees of freedom. The same pair frozen at a *different* target is an
incompatible declaration and fails at once.

Units: coordinates and targets in Bohr, masses in atomic units of mass, the
tolerance dimensionless (relative residual) and the physical residual reported
alongside it in Bohr.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.utilities import cv

# A residual below this fraction of the target distance is satisfied to double
# precision; asking for less is a declaration error, not a solver failure.
MIN_TOLERANCE = 1.0e-14


class ShakeError(ValueError):
    """A strict constraint cannot be built or solved as declared."""


class ShakeNotConverged(ShakeError):
    """The position iteration stopped without meeting the tolerance.

    Attributes
    ----------
    residual : float
        Largest normalized residual reached.
    physical_residual : float
        Same, expressed in Bohr, for the atom that carried it.
    constraint : tuple
        The 0-based atom pair at fault.
    iterations : int
        Passes made over the constraint set.
    description : str, optional
        How the failing condition should be named. The bond solver reports a
        pair; the multi-condition solver reports a kind over a list of atoms,
        because "constraint 2-3" of an angle says the wrong thing about a
        condition that has three atoms.
    """

    def __init__(self, residual, physical_residual, constraint, iterations,
                 description=None):
        self.residual = float(residual)
        self.physical_residual = float(physical_residual)
        self.constraint = tuple(int(atom) for atom in constraint)
        self.iterations = int(iterations)
        if description is None:
            message = ('bond constraint %d-%d did not converge: normalized '
                       'residual %.3e after %d passes (tolerance not met)'
                       % (self.constraint[0], self.constraint[1], self.residual,
                          self.iterations))
        else:
            message = ('%s did not converge: normalized residual %.3e after '
                       '%d passes (tolerance not met)'
                       % (description, self.residual, self.iterations))
        super(ShakeNotConverged, self).__init__(message)


def build_bond_jacobian(coords, pairs, natom=None):
    """Rows of d|r_ij|/dR for a set of bond pairs.

    Parameters
    ----------
    coords : numpy.ndarray
        (natom, 3) coordinates in Bohr.
    pairs : array-like
        (n, 2) 0-based atom pairs.
    natom : int, optional
        Atom count; defaults to the number of rows in `coords`.

    Returns
    -------
    numpy.ndarray
        Shape (n, 3 * natom). A zero row marks a constraint whose geometry is
        degenerate, which `normalized_residual` reports rather than hiding.
    """
    array = np.asarray(coords, dtype=np.float64)
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    total = int(natom if natom is not None else array.shape[0])
    jacobian = np.zeros((rows.shape[0], total * 3), dtype=np.float64)
    for position, (left, right) in enumerate(rows):
        vector = array[left] - array[right]
        length = float(np.linalg.norm(vector))
        if length <= cv.SINGULAR_LENGTH:
            continue
        unit = vector / length
        for axis in range(3):
            jacobian[position, left * 3 + axis] = unit[axis]
            jacobian[position, right * 3 + axis] = -unit[axis]
    return jacobian


def distances(coords, pairs):
    """Length of each declared bond at a geometry, in Bohr."""
    array = np.asarray(coords, dtype=np.float64)
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if rows.size == 0:
        return np.zeros(0)
    vectors = array[rows[:, 0]] - array[rows[:, 1]]
    return np.linalg.norm(vectors, axis=1)


def normalized_residual(coords, pairs, targets, tolerance=None):
    """Relative violation of each constraint, and the largest one.

    Parameters
    ----------
    coords : numpy.ndarray
        (natom, 3) coordinates in Bohr.
    pairs : array-like
        (n, 2) 0-based atom pairs.
    targets : array-like or float
        Target length per pair in Bohr, or one length for all of them.
    tolerance : float, optional
        When given, checked against the result for the caller's convenience.

    Returns
    -------
    tuple
        (per-constraint normalized residuals, largest of them, index of the
        worst, physical residual of the worst in Bohr).
    """
    array = np.asarray(coords, dtype=np.float64)
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    values = distances(array, rows)
    wanted = _targets(rows, targets)
    if rows.shape[0] == 0:
        return np.zeros(0), 0.0, -1, 0.0
    scale = np.maximum(np.abs(wanted), 1.0e-12)
    residual = np.abs(values - wanted) / scale
    worst = int(np.argmax(residual))
    largest = float(residual[worst])
    physical = float(abs(values[worst] - wanted[worst]))
    if tolerance is not None and largest > float(tolerance):
        raise ShakeError(
            'constraint residual %.3e exceeds the declared tolerance %.3e'
            % (largest, float(tolerance)))
    return residual, largest, worst, physical


def detect_redundancy(pairs, targets, anchors, fix_mask=None,
                      tolerance=1.0e-8):
    """Split declared bonds into effective, redundant and impossible.

    Parameters
    ----------
    pairs : array-like
        (n, 2) 0-based atom pairs.
    targets : array-like or float
        Target lengths in Bohr.
    anchors : numpy.ndarray
        Reference geometry the targets were or may be bound to, in Bohr.
    fix_mask : numpy.ndarray, optional
        (natom, 3) frozen-component mask.
    tolerance : float
        Relative agreement that counts as "the same distance".

    Returns
    -------
    dict
        'effective' pairs that add a condition, 'redundant' pairs already
        satisfied by the frozen geometry, and 'conflict' pairs frozen at both
        ends yet asked to hold a different length.
    """
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    wanted = _targets(rows, targets)
    array = np.asarray(anchors, dtype=np.float64)
    values = distances(array, rows)
    mobile = np.ones(rows.shape[0], dtype=bool)
    if fix_mask is not None:
        mask = np.asarray(fix_mask, dtype=bool)
        both = np.array([bool(mask[left].all() and mask[right].all())
                         for left, right in rows])
        agreed = np.abs(values - wanted) <= tolerance * np.maximum(
            np.abs(wanted), 1.0e-12)
        redundant = np.flatnonzero(both & agreed)
        conflict = np.flatnonzero(both & ~agreed)
        mobile[both] = False
    else:
        redundant = np.flatnonzero(np.abs(values - wanted) <= 0.0)
        conflict = np.flatnonzero([])
        mobile[redundant] = False
    return {'effective': rows[mobile],
            'redundant': rows[redundant],
            'conflict': rows[conflict]}


def solve_positions(coords, pairs, targets, masses, tolerance=1.0e-8,
                    max_iterations=200, fix_mask=None, anchors=None):
    """Move positions until every declared bond length is met.

    Corrections run along each bond, shared between the two atoms in inverse
    proportion to their masses, and frozen components are held at their anchor
    by taking them out of the correction. This is the position-level part of a
    strict constraint: see the module docstring for what it does not do.

    Parameters
    ----------
    coords : array-like
        (natom, 3) candidate coordinates in Bohr. Not modified.
    pairs : array-like
        (n, 2) 0-based atom pairs of the effective constraints.
    targets : array-like or float
        Target length per pair in Bohr.
    masses : array-like
        Positive per-atom masses in atomic units.
    tolerance : float
        Dimensionless relative residual that counts as satisfied.
    max_iterations : int
        Passes over the whole constraint set.
    fix_mask : numpy.ndarray, optional
        (natom, 3) frozen components; they are restored after every pass.
    anchors : numpy.ndarray, optional
        Reference geometry to restore frozen components to.

    Returns
    -------
    dict
        'coordinates' (fresh array), 'converged', 'iterations', 'residual'
        (per constraint), 'max_residual', 'worst' pair and 'shift' (the total
        distance each atom was moved, Bohr).

    Raises
    ------
    ShakeError
        Degenerate declaration: no atoms, non-positive mass, zero-length bond,
        or a target that is not positive.
    ShakeNotConverged
        The iteration stopped short of the tolerance.
    """
    array = np.array(coords, dtype=np.float64, copy=True)
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    if values.size != array.shape[0]:
        raise ShakeError(
            'mass array has %d entries for %d atoms'
            % (values.size, array.shape[0]))
    if np.any(values <= 0.0):
        raise ShakeError('every mass must be positive')
    if rows.size == 0:
        return {'coordinates': array, 'converged': True, 'iterations': 0,
                'residual': np.zeros(0), 'max_residual': 0.0, 'worst': None,
                'shift': np.zeros_like(array)}
    wanted = _targets(rows, targets)
    if np.any(wanted <= 0.0):
        raise ShakeError('every constraint target length must be positive')
    tolerance = float(tolerance)
    if tolerance < MIN_TOLERANCE:
        raise ShakeError(
            'tolerance %.3e is below double precision; the smallest usable '
            'value is %.1e' % (tolerance, MIN_TOLERANCE))
    iterations = int(max_iterations)
    if iterations < 1:
        raise ShakeError('max_iterations must be at least one')

    frozen = (np.zeros_like(array, dtype=bool) if fix_mask is None
              else np.asarray(fix_mask, dtype=bool))
    if anchors is None:
        anchors = array
    reference = np.array(array, dtype=np.float64, copy=True)

    # Inverse-mass sharing weights, so a heavy atom barely moves.
    weight = 1.0 / values
    for step in range(iterations):
        for position, (left, right) in enumerate(rows):
            vector = array[left] - array[right]
            length = float(np.linalg.norm(vector))
            if length <= cv.SINGULAR_LENGTH:
                raise ShakeError(
                    'bond %d-%d collapsed to zero length; the constraint '
                    'cannot be restored from a degenerate geometry'
                    % (int(left), int(right)))
            error = length - float(wanted[position])
            if abs(error) <= tolerance * float(wanted[position]):
                continue
            share = weight[left] + weight[right]
            scale = error / (length * share)
            array[left] -= weight[left] * scale * vector
            array[right] += weight[right] * scale * vector
        if frozen.any():
            array[frozen] = np.asarray(anchors)[frozen]
        residual, largest, worst, _ = normalized_residual(array, rows, wanted)
        if largest <= tolerance:
            return {'coordinates': array, 'converged': True,
                    'iterations': step + 1, 'residual': residual,
                    'max_residual': largest,
                    'worst': None if worst < 0 else tuple(rows[worst]),
                    'shift': np.linalg.norm(array - reference, axis=1)}
    residual, largest, worst, physical = normalized_residual(array, rows,
                                                            wanted)
    raise ShakeNotConverged(largest, physical,
                            rows[worst] if worst >= 0 else rows[0], iterations)


def _targets(rows, targets):
    """Broadcast one target to every pair, or check the list length."""
    array = np.asarray(targets, dtype=np.float64).reshape(-1)
    count = int(np.asarray(rows).reshape(-1, 2).shape[0])
    if array.size == 1:
        return np.full(count, float(array[0]))
    if array.size != count:
        raise ShakeError(
            'got %d targets for %d constraints' % (array.size, count))
    return array


# ---------------------------------------------------------------------------
# Multi-condition SHAKE: the same iteration over angles and torsions.
#
# `solve_positions` is the single-row case of what follows. A bond's Jacobian
# row is (+u, -u), and correcting along the row with the inverse-mass metric is
# the atom-pair formula, line by line. An angle and a torsion have the same
# structure with three or four touched atoms, and `utilities.cv` already hands
# out the row: the merge that brought the constraint subsystem in (CHANGELOG
# #255) deliberately moved that mathematics to one place, so this adds a solver,
# not a second derivative.
#
# Two things a bond solver never had to answer do have to be answered here:
#
# - Redundancy. Three angles of a planar group add two conditions, not three;
#   the third row is a linear combination of the others and a Gauss-Seidel sweep
#   on a dependent set oscillates instead of converging. This is decided by the
#   rank of the condition matrix, which is also the number of degrees of freedom
#   the set removes, so the count and the solve read the same fact.
# - Degeneracy. A collinear angle has no derivative at all, and `cv` raises on
#   it. Iterating through such a geometry is not a slow convergence, it is a
#   wrong answer, so the failure is named with the condition that caused it.
# ---------------------------------------------------------------------------

CONDITION_WIDTH = {'bond': 2, 'angle': 3, 'dihedral': 4}

# The unit each kind is measured in, for the messages: lengths in Bohr, every
# angle in radians (design document section 3.8 keeps radians end to end).
CONDITION_UNIT = {'bond': 'Bohr', 'angle': 'radian', 'dihedral': 'radian'}


def condition_value(coords, kind, indices):
    """One constrained variable at one geometry, in its own unit."""
    atoms = [int(atom) for atom in indices]
    if kind == 'bond':
        return float(cv.distance(coords, atoms[0], atoms[1]))
    if kind == 'angle':
        return float(cv.angle(coords, atoms[0], atoms[1], atoms[2]))
    if kind == 'dihedral':
        return float(cv.dihedral(coords, atoms[0], atoms[1], atoms[2],
                                 atoms[3]))
    raise ShakeError("unknown constraint kind %r; expected one of %s"
                     % (kind, ', '.join(sorted(CONDITION_WIDTH))))


def condition_row(coords, kind, indices, natom=None):
    """`dq/dR` for one condition, flattened to (3 * natom,).

    The gradient comes from the shared core, so an angle row here is the same
    numbers `cv.angle(..., grad=True)` returns.
    """
    atoms = [int(atom) for atom in indices]
    total = int(natom if natom is not None else
                np.asarray(coords, dtype=np.float64).shape[0])
    if kind == 'bond':
        _value, rows = cv.distance(coords, atoms[0], atoms[1], grad=True)
    elif kind == 'angle':
        _value, rows = cv.angle(coords, atoms[0], atoms[1], atoms[2],
                                grad=True)
    elif kind == 'dihedral':
        _value, rows = cv.dihedral(coords, atoms[0], atoms[1], atoms[2],
                                   atoms[3], grad=True)
    else:
        raise ShakeError("unknown constraint kind %r" % kind)
    # `cv` hands back the block in the atom order it was called with; the
    # solver's row is indexed by atom number, so the rows are scattered here
    # and nowhere else in this module.
    out = np.zeros(total * 3, dtype=np.float64)
    for position, atom in enumerate(atoms):
        out[atom * 3:atom * 3 + 3] = rows[position]
    return out


def normalize_conditions(conditions):
    """Check the shape of a condition list and return it as tuples.

    A condition is `(kind, indices, target)`, in the units the module docstring
    declares: Bohr for a bond length, radians for an angle or a torsion.
    """
    out = []
    for position, item in enumerate(conditions):
        try:
            kind, indices, target = item
        except (TypeError, ValueError):
            raise ShakeError(
                'condition %d is not a (kind, indices, target) triple'
                % (position + 1))
        kind = str(kind)
        if kind not in CONDITION_WIDTH:
            raise ShakeError(
                "condition %d has unknown kind %r; expected one of %s"
                % (position + 1, kind, ', '.join(sorted(CONDITION_WIDTH))))
        atoms = [int(atom) for atom in indices]
        if len(atoms) != CONDITION_WIDTH[kind]:
            raise ShakeError(
                'condition %d of kind %s names %d atoms, needs %d'
                % (position + 1, kind, len(atoms), CONDITION_WIDTH[kind]))
        value = float(target)
        if kind == 'bond' and value <= 0.0:
            raise ShakeError(
                'condition %d: a bond target length must be positive'
                % (position + 1))
        out.append((kind, atoms, value))
    return out


def build_condition_jacobian(coords, conditions, natom=None):
    """The (n, 3 * natom) condition matrix at one geometry.

    Unlike `build_bond_jacobian`, a degenerate geometry does not produce a zero
    row here: it raises, because a zero row in a rank test reads as redundancy
    and the honest reading is that this geometry cannot be constrained at all.
    """
    array = np.asarray(coords, dtype=np.float64)
    total = int(natom if natom is not None else array.shape[0])
    rows = np.zeros((len(conditions), total * 3), dtype=np.float64)
    for position, (kind, atoms, _target) in enumerate(conditions):
        try:
            rows[position] = condition_row(array, kind, atoms, natom=total)
        except cv.CVSingularError as error:
            raise ShakeError(
                'condition %d (%s over atoms %s) cannot be differentiated at '
                'this geometry: %s'
                % (position + 1, kind, [atom + 1 for atom in atoms],
                   error)) from error
    return rows


def condition_redundancy(jacobian, metric=None, tolerance=1.0e-8):
    """Split conditions into independent and linearly dependent.

    Dependence is tested in the mass metric the solver works in: a row the
    weighted Gram-Schmidt sweep cannot place outside the span of the earlier
    ones adds no condition, so it must not cost a degree of freedom either. That
    is why the count comes from here and not from `len(conditions)`.

    Returns
    -------
    dict
        'independent' positions kept, 'redundant' positions dropped, 'rank' the
        number of independent conditions.
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return {'independent': [], 'redundant': [], 'rank': 0}
    weight = (np.ones(matrix.shape[1], dtype=np.float64) if metric is None
              else np.asarray(metric, dtype=np.float64).reshape(-1))
    basis = []
    independent = []
    redundant = []
    for position in range(matrix.shape[0]):
        row = matrix[position] * weight
        scale = float(np.linalg.norm(row))
        if scale <= 0.0:
            redundant.append(position)
            continue
        for base in basis:
            row = row - float(np.dot(row, base)) * base
        residual = float(np.linalg.norm(row))
        if residual <= tolerance * scale:
            redundant.append(position)
            continue
        basis.append(row / residual)
        independent.append(position)
    return {'independent': independent, 'redundant': redundant,
            'rank': len(independent)}


def condition_residual(coords, conditions, targets=None):
    """Signed violation of each condition, relative to its own scale."""
    values = np.array([condition_value(coords, kind, atoms)
                       for kind, atoms, _ in conditions], dtype=np.float64)
    wanted = (np.array([target for _k, _a, target in conditions],
                       dtype=np.float64) if targets is None
              else np.asarray(targets, dtype=np.float64).reshape(-1))
    scale = np.maximum(np.abs(wanted), 1.0e-12)
    residual = values - wanted
    return residual, residual / scale


def solve_conditions(coords, conditions, masses, tolerance=1.0e-8,
                    max_iterations=200, fix_mask=None, anchors=None,
                    redundancy_tolerance=1.0e-8):
    """Move positions until every declared condition is met.

    Each sweep corrects one condition along its own Jacobian row, shared over
    the atoms it touches in inverse proportion to their masses:

        dR = -(q - q0) * W J / (J W J)

    with W the inverse-mass matrix. For a bond that algebra is exactly the pair
    formula `solve_positions` uses, so this is the same method with more shapes,
    not a different one. Dependent conditions are taken out of the sweep by the
    rank test, and a geometry the core cannot differentiate raises with the
    condition responsible instead of oscillating to the iteration cap.

    Parameters
    ----------
    coords : array-like
        (natom, 3) candidate coordinates in Bohr. Not modified.
    conditions : list
        (kind, indices, target) triples; see `normalize_conditions`.
    masses : array-like
        Positive per-atom masses in atomic units.
    tolerance : float
        Relative residual that counts as satisfied.
    max_iterations : int
        Passes over the whole condition set.
    fix_mask : numpy.ndarray, optional
        (natom, 3) frozen components, restored after every pass.
    anchors : numpy.ndarray, optional
        Geometry the frozen components return to.

    Returns
    -------
    dict
        'coordinates', 'converged', 'iterations', 'residual' (per condition
        kept), 'max_residual', 'worst' condition index, 'shift' per atom, and
        'redundant' with the dropped positions and the rank that set the
        degrees-of-freedom count.

    Raises
    ------
    ShakeError
        A declaration or geometry that cannot be solved: bad mass array, a
        condition that names an atom the structure does not have, a degenerate
        (collinear or eclipsed) geometry, or a tolerance below double
        precision.
    ShakeNotConverged
        The iteration stopped short of the tolerance.
    """
    array = np.array(coords, dtype=np.float64, copy=True)
    declared = normalize_conditions(conditions)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    natom = array.shape[0]
    if values.size != natom:
        raise ShakeError('mass array has %d entries for %d atoms'
                         % (values.size, natom))
    if np.any(values <= 0.0):
        raise ShakeError('every mass must be positive')
    if not declared:
        return {'coordinates': array, 'converged': True, 'iterations': 0,
                'residual': np.zeros(0), 'max_residual': 0.0, 'worst': None,
                'shift': np.zeros_like(array),
                'redundant': {'independent': [], 'redundant': [], 'rank': 0}}
    for position, (_kind, atoms, _t) in enumerate(declared):
        for atom in atoms:
            if atom < 0 or atom >= natom:
                raise ShakeError(
                    'condition %d names atom %d, which the structure of %d '
                    'atoms does not have' % (position + 1, atom + 1, natom))
    tolerance = float(tolerance)
    if tolerance < MIN_TOLERANCE:
        raise ShakeError(
            'tolerance %.3e is below double precision; the smallest usable '
            'value is %.1e' % (tolerance, MIN_TOLERANCE))
    iterations = int(max_iterations)
    if iterations < 1:
        raise ShakeError('max_iterations must be at least one')

    weight = np.repeat(1.0 / values, 3)
    frozen = (np.zeros_like(array, dtype=bool) if fix_mask is None
              else np.asarray(fix_mask, dtype=bool))
    start = np.array(array, dtype=np.float64, copy=True)
    restore = start if anchors is None else np.asarray(anchors,
                                                      dtype=np.float64)
    targets = np.array([target for _k, _a, target in declared],
                       dtype=np.float64)

    # Rank first: a dependent row must not be swept, and the count of what
    # survives is the degrees-of-freedom answer the caller reports.
    matrix = build_condition_jacobian(array, declared, natom=natom)
    split = condition_redundancy(matrix, metric=weight,
                                 tolerance=redundancy_tolerance)
    kept = list(split['independent'])
    if not kept:
        return {'coordinates': array, 'converged': True, 'iterations': 0,
                'residual': np.zeros(0), 'max_residual': 0.0, 'worst': None,
                'shift': np.zeros_like(array),
                'redundant': dict(split, rank=0)}

    def _degenerate(position, error):
        kind, atoms, _ = declared[position]
        raise ShakeError(
            'condition %d (%s over atoms %s) cannot be differentiated at this '
            'geometry: %s' % (position + 1, kind,
                              [atom + 1 for atom in atoms], error))

    for step in range(iterations):
        for position in list(kept):
            kind, atoms, target = declared[position]
            try:
                current = condition_value(array, kind, atoms)
                row = condition_row(array, kind, atoms, natom=natom)
            except cv.CVSingularError as error:
                _degenerate(position, error)
            error_value = current - target
            if abs(error_value) <= tolerance * max(abs(target), 1.0e-12):
                continue
            denominator = float(np.dot(row, weight * row))
            if denominator <= 1.0e-30:
                _degenerate(position, 'the condition row vanished, so no '
                                      'displacement changes this variable')
            shift = -error_value * weight * row / denominator
            array += shift.reshape(natom, 3)
        if frozen.any():
            array[frozen] = restore[frozen]
        active = [declared[position] for position in kept]
        signed, relative = condition_residual(array, active, targets[kept])
        largest = float(np.abs(relative).max()) if relative.size else 0.0
        if largest <= tolerance:
            worst = int(np.argmax(np.abs(relative))) if relative.size else -1
            return {'coordinates': array, 'converged': True,
                    'iterations': step + 1,
                    'residual': signed, 'max_residual': largest,
                    'worst': None if worst < 0 else kept[worst],
                    'shift': np.linalg.norm(array - start, axis=1),
                    'redundant': split}
    active = [declared[position] for position in kept]
    signed, relative = condition_residual(array, active, targets[kept])
    worst = int(np.argmax(np.abs(relative))) if relative.size else 0
    kind, atoms, _target = active[worst] if relative.size else declared[0]
    raise ShakeNotConverged(
        float(relative[worst]) if relative.size else 0.0,
        float(abs(signed[worst])) if relative.size else 0.0,
        atoms, iterations,
        description='constraint %s over atoms %s'
                    % (kind, [atom + 1 for atom in atoms]))


def relative_speeds(coords, pairs, velocities):
    """`d|r_ij|/dt` for each declared bond, Bohr per atomic time.

    The row is `u_ij . (v_i - v_j)`, the same Jacobian the position solve uses
    applied to the velocities. A bond whose geometry is degenerate reports zero,
    because there is no direction to differentiate along; `solve_velocities`
    raises on that geometry instead of projecting nothing.
    """
    array = np.asarray(coords, dtype=np.float64)
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    speeds = np.asarray(velocities, dtype=np.float64)
    if rows.size == 0:
        return np.zeros(0)
    vectors = array[rows[:, 0]] - array[rows[:, 1]]
    lengths = np.linalg.norm(vectors, axis=1)
    differences = speeds[rows[:, 0]] - speeds[rows[:, 1]]
    rates = np.einsum('ij,ij->i', vectors, differences)
    usable = lengths > cv.SINGULAR_LENGTH
    out = np.zeros(rows.shape[0])
    out[usable] = rates[usable] / lengths[usable]
    return out


def velocity_scale(velocities):
    """The largest atomic speed the run carries, Bohr per atomic time.

    A rate has units of speed, so a relative tolerance has to divide by a speed.
    This is the one the velocities themselves supply; it is never zero unless the
    whole velocity array is, and the floor keeps a stationary structure from
    turning the threshold into zero.
    """
    array = np.asarray(velocities, dtype=np.float64)
    if array.size == 0:
        return 1.0e-12
    return max(float(np.abs(array).max()), 1.0e-12)


def solve_velocities(coords, pairs, velocities, masses,
                     tolerance=1.0e-8, max_iterations=200, fix_mask=None):
    """Project velocities so every declared bond length stops changing.

    This is the velocity half of the same condition: with the geometry already
    satisfying `|r_ij| = d`, require `u_ij . (v_i - v_j) = 0`. The correction is
    Gauss-Seidel over the constraint set, shared over the two atoms in inverse
    proportion to their masses exactly as `solve_positions` shares its position
    correction, so the momentum each pair gains is equal and opposite and the
    projection invents no net momentum.

    It is reached only when a constraint record declares
    `velocity_constraint: rattle`, so a run that does not ask for it takes no
    branch here. Frozen components are held at zero through every pass, which is
    what keeps a freeze and a bond constraint from disagreeing about an atom.

    Parameters
    ----------
    coords : array-like
        (natom, 3) coordinates in Bohr, at the geometry the positions were
        projected to. The Jacobian is read here, not from the velocities.
    pairs : array-like
        (n, 2) 0-based atom pairs of the effective constraints.
    velocities : array-like
        (natom, 3) in Bohr per atomic time. Not modified.
    masses : array-like
        Positive per-atom masses in atomic units.
    tolerance : float
        Relative residual that counts as satisfied, against `velocity_scale`.
    max_iterations : int
        Passes over the whole constraint set.
    fix_mask : numpy.ndarray, optional
        (natom, 3) frozen components; their velocities stay zero.

    Returns
    -------
    dict
        'velocities' (fresh array), 'converged', 'iterations', 'residual' (per
        constraint, Bohr per atomic time), 'max_residual' (same units),
        'normalized_residual' (largest residual over the scale), 'worst' pair.

    Raises
    ------
    ShakeError
        Degenerate declaration: mass array mismatch, non-positive mass, or a
        bond whose geometry has collapsed so no tangent exists.
    ShakeNotConverged
        The iteration stopped short of the tolerance. `residual` and
        `physical_residual` on the exception are in Bohr per atomic time.
    """
    array = np.asarray(coords, dtype=np.float64)
    rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    speeds = np.array(velocities, dtype=np.float64, copy=True)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ShakeError('coordinates must be (natom, 3), got shape %s'
                         % (array.shape,))
    if values.size != array.shape[0]:
        raise ShakeError(
            'mass array has %d entries for %d atoms'
            % (values.size, array.shape[0]))
    if np.any(values <= 0.0):
        raise ShakeError('every mass must be positive')
    if speeds.shape != array.shape:
        raise ShakeError(
            'velocity shape %s does not match coordinates %s'
            % (speeds.shape, array.shape))
    if rows.size == 0:
        return {'velocities': speeds, 'converged': True, 'iterations': 0,
                'residual': np.zeros(0), 'max_residual': 0.0,
                'normalized_residual': 0.0, 'worst': None}
    for left, right in rows:
        length = float(np.linalg.norm(array[left] - array[right]))
        if length <= cv.SINGULAR_LENGTH:
            raise ShakeError(
                'bond %d-%d has no direction to project along; the position '
                'solve must deliver a geometry before the velocities can'
                % (int(left), int(right)))
    tolerance = float(tolerance)
    if tolerance < MIN_TOLERANCE:
        raise ShakeError(
            'tolerance %.3e is below double precision; the smallest usable '
            'value is %.1e' % (tolerance, MIN_TOLERANCE))
    iterations = int(max_iterations)
    if iterations < 1:
        raise ShakeError('max_iterations must be at least one')

    frozen = (np.zeros_like(array, dtype=bool) if fix_mask is None
              else np.asarray(fix_mask, dtype=bool))
    scale = velocity_scale(speeds)
    threshold = tolerance * scale
    weight = 1.0 / values
    for step in range(iterations):
        for left, right in rows:
            vector = array[left] - array[right]
            length = float(np.linalg.norm(vector))
            unit = vector / length
            rate = float(np.dot(unit, speeds[left] - speeds[right]))
            if abs(rate) <= threshold:
                continue
            share = weight[left] + weight[right]
            impulse = rate / share
            speeds[left] -= weight[left] * impulse * unit
            speeds[right] += weight[right] * impulse * unit
        if frozen.any():
            speeds[frozen] = 0.0
        residual = relative_speeds(array, rows, speeds)
        largest = float(np.abs(residual).max())
        if largest <= threshold:
            worst = int(np.argmax(np.abs(residual)))
            return {'velocities': speeds, 'converged': True,
                    'iterations': step + 1, 'residual': residual,
                    'max_residual': largest,
                    'normalized_residual': largest / scale,
                    'worst': None if largest == 0.0 else tuple(rows[worst])}
    residual = relative_speeds(array, rows, speeds)
    worst = int(np.argmax(np.abs(residual)))
    raise ShakeNotConverged(float(np.abs(residual[worst])) / scale,
                            float(abs(residual[worst])), rows[worst],
                            iterations)


def condition_rates(coords, conditions, velocities, natom=None):
    """`J v` for each condition: how fast the constrained variable is moving.

    For a bond this is the `d|r_ij|/dt` row `relative_speeds` returns; for an
    angle or a torsion it is the same contraction with the row the shared core
    already hands out, so no second derivative enters the build.
    """
    array = np.asarray(coords, dtype=np.float64)
    total = int(natom if natom is not None else array.shape[0])
    out = np.zeros(len(conditions), dtype=np.float64)
    flat = np.asarray(velocities, dtype=np.float64).reshape(total * 3)
    for position, (kind, atoms, _target) in enumerate(conditions):
        out[position] = float(np.dot(condition_row(array, kind, atoms,
                                                   natom=total), flat))
    return out


def solve_velocity_conditions(coords, conditions, velocities, masses,
                              tolerance=1.0e-8, max_iterations=200,
                              fix_mask=None, redundancy_tolerance=1.0e-8):
    """Project velocities so every declared condition stops changing.

    This is the velocity half of `solve_conditions`, the same impulse algebra:

        dv = -(J v) * W J / (J W J)

    summed over the conditions in one Gauss-Seidel sweep. It exists because the
    position solve alone leaves a constrained water carrying kinetic energy in
    the coordinate it was just forbidden to move: measured on a real periodic
    LAMMPS run, the unconstrained box drifts 5.1e-5 Hartree in `Etot` over 600
    steps, the same box with a position-only constraint drifts 0.090863, and the
    bonds projected case drifts 0.000320 (dev/methods/
    CONSTRAINT_DEVELOPMENT.md section 20; the products lived in
    evidence_20260930/ before the evidence directories were retired).

    A dependent condition is taken out by the rank test rather than swept, and a
    geometry with no tangent (a collinear angle) fails by name instead of
    projecting against a zero row.

    Parameters
    ----------
    coords : array-like
        (natom, 3) coordinates in Bohr at the projected geometry.
    conditions : list
        (kind, indices, target) triples; the target is unused here, the shape
        and the geometry are what build the rows.
    velocities : array-like
        (natom, 3) Bohr per atomic time. Not modified.
    masses : array-like
        Positive per-atom masses in atomic units.
    tolerance : float
        Relative threshold against `velocity_scale`.
    max_iterations : int
        Passes over the condition set.
    fix_mask : numpy.ndarray, optional
        (natom, 3) frozen components; their velocities stay zero.
    redundancy_tolerance : float
        Relative residual under which a condition row is dependent.

    Returns
    -------
    dict
        'velocities' (fresh array), 'converged', 'iterations', 'residual' (per
        kept condition, in the variable's own unit per atomic time),
        'max_residual', 'normalized_residual', 'worst' condition index and
        'redundant' with the dropped positions and the rank.

    Raises
    ------
    ShakeError
        Shape or mass mismatch, or a geometry whose condition has no tangent.
    ShakeNotConverged
        The sweep stopped short of the threshold.
    """
    array = np.asarray(coords, dtype=np.float64)
    declared = normalize_conditions(conditions)
    speeds = np.array(velocities, dtype=np.float64, copy=True)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    natom = array.shape[0]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ShakeError('coordinates must be (natom, 3), got shape %s'
                         % (array.shape,))
    if values.size != natom:
        raise ShakeError('mass array has %d entries for %d atoms'
                         % (values.size, natom))
    if np.any(values <= 0.0):
        raise ShakeError('every mass must be positive')
    if speeds.shape != array.shape:
        raise ShakeError('velocity shape %s does not match coordinates %s'
                         % (speeds.shape, array.shape))
    if not declared:
        return {'velocities': speeds, 'converged': True, 'iterations': 0,
                'residual': np.zeros(0), 'max_residual': 0.0,
                'normalized_residual': 0.0, 'worst': None,
                'redundant': {'independent': [], 'redundant': [], 'rank': 0}}
    for position, (_kind, atoms, _t) in enumerate(declared):
        for atom in atoms:
            if atom < 0 or atom >= natom:
                raise ShakeError(
                    'condition %d names atom %d, which the structure of %d '
                    'atoms does not have'
                    % (position + 1, atom + 1, natom))
    tolerance = float(tolerance)
    if tolerance < MIN_TOLERANCE:
        raise ShakeError(
            'tolerance %.3e is below double precision; the smallest usable '
            'value is %.1e' % (tolerance, MIN_TOLERANCE))
    iterations = int(max_iterations)
    if iterations < 1:
        raise ShakeError('max_iterations must be at least one')

    frozen = (np.zeros_like(array, dtype=bool) if fix_mask is None
              else np.asarray(fix_mask, dtype=bool))
    weight = np.repeat(1.0 / values, 3)
    scale = velocity_scale(speeds)
    threshold = tolerance * scale
    flat = speeds.reshape(natom * 3)
    try:
        matrix = build_condition_jacobian(array, declared, natom=natom)
    except ShakeError as error:
        raise ShakeError('the velocity projection cannot read the geometry it '
                         'projects: %s' % error) from error
    split = condition_redundancy(matrix, metric=weight,
                                 tolerance=redundancy_tolerance)
    kept = list(split['independent'])
    if not kept:
        return {'velocities': speeds, 'converged': True, 'iterations': 0,
                'residual': np.zeros(0), 'max_residual': 0.0,
                'normalized_residual': 0.0, 'worst': None,
                'redundant': dict(split, rank=0)}

    for step in range(iterations):
        for position in list(kept):
            kind, atoms, _target = declared[position]
            row = condition_row(array, kind, atoms, natom=natom)
            rate = float(np.dot(row, flat))
            if abs(rate) <= threshold:
                continue
            weighted = weight * row
            denominator = float(np.dot(row, weighted))
            if denominator <= 1.0e-30:
                raise ShakeError(
                    'condition %d (%s over atoms %s) has no direction to '
                    'project along at this geometry: the condition row vanished'
                    % (position + 1, kind, [atom + 1 for atom in atoms]))
            flat -= rate * weighted / denominator
        if frozen.any():
            speeds[frozen] = 0.0
        residual = condition_rates(array, [declared[p] for p in kept], speeds,
                                   natom=natom)
        largest = float(np.abs(residual).max()) if residual.size else 0.0
        if largest <= threshold:
            worst = int(np.argmax(np.abs(residual))) if residual.size else -1
            return {'velocities': speeds, 'converged': True,
                    'iterations': step + 1, 'residual': residual,
                    'max_residual': largest,
                    'normalized_residual': largest / scale,
                    'worst': None if worst < 0 else kept[worst],
                    'redundant': split}
    residual = condition_rates(array, [declared[p] for p in kept], speeds,
                               natom=natom)
    worst = int(np.argmax(np.abs(residual))) if residual.size else 0
    kind, atoms, _t = declared[kept[worst]] if residual.size else declared[0]
    raise ShakeNotConverged(
        float(abs(residual[worst])) / scale if residual.size else 0.0,
        float(abs(residual[worst])) if residual.size else 0.0, atoms,
        iterations,
        description='velocity projection of the %s over atoms %s'
                    % (kind, [atom + 1 for atom in atoms]))
