"""
======================================================================
This module provides geometric internal coordinate computation tools
for PyRAI3MD: bond distances, bond angles and dihedral angles with
their exact first and second derivatives, the dense BND/AGL/DHD
adapters, and the mass-weighted group center. Angles are radians;
degenerate geometry raises CVSingularError. It is numpy-only.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np



# ---------------------------------------------------------------------------
# The one torsion, angle and bond implementation in the package.
#
# These three helpers used to carry a closed-form cross-product torsion whose
# gradient was the derivative of neither its own value nor any torsion: on a
# generic geometry only the first atom's row was right, the second and third
# were wrong in sign and magnitude at once, and the fourth was exactly negated
# (dev/methods/CONSTRAINT_DEVELOPMENT.md section 3.9, measured 2026-09-29).
# Degenerate geometry also returned nan rather than failing, at the strict cis
# and trans conformations that occur most often in real molecules. The merge of
# the constraint subsystem (CHANGELOG #255, closing D-01, D-02 and D-03) moves
# the atan2-invariant form here, together with its guards, and the constraint
# layer now calls out to this module instead of carrying its own copy.
#
# Angles are radians and their derivatives are radians per length, so a
# Hartree/radian^2 force constant multiplies straight through with no hidden
# factor of 57.2958 anywhere. `utilities/units.py` owns the conversion for
# callers that want degrees.
# ---------------------------------------------------------------------------

# Geometry below this extent is not a direction, so the derivative of any
# variable that needs one is undefined.
SINGULAR_LENGTH = 1.0e-8

# A sine below this magnitude leaves the angle or torsion derivative with a
# division by zero, exactly the collinear and eclipsed-normal cases.
SINGULAR_SINE = 1.0e-8

# Tolerance on the acos domain: rounding can push a unit cosine a few ul past
# +-1, which is clamped; anything further is a degenerate geometry.
COS_TOLERANCE = 1.0e-8


class CVSingularError(ValueError):
    """Raised when a geometry is too degenerate to differentiate."""


def _acos(cosine, label):
    """Arccos with a rounding-tolerant domain check."""
    value = float(cosine)
    if value > 1.0:
        if value <= 1.0 + COS_TOLERANCE:
            value = 1.0
        else:
            raise CVSingularError(
                '%s cosine %.10f is above 1 by more than rounding'
                % (label, value))
    elif value < -1.0:
        if value >= -1.0 - COS_TOLERANCE:
            value = -1.0
        else:
            raise CVSingularError(
                '%s cosine %.10f is below -1 by more than rounding'
                % (label, value))
    return float(np.arccos(value))


def _as_coords(coords):
    """Return a float64 (natom, 3) view of the given coordinates."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise CVSingularError(
            'coordinates must be (natom, 3), got shape %s' % (array.shape,))
    return array


def group_center(coords, indices, masses=None):
    """Center of a group of atoms, as a position and a set of weights.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape (natom, 3).
    indices : array-like
        0-based member indices, at least one atom.
    masses : array-like, optional
        Positive per-atom masses. Without them the geometric center is
        returned, which is the center of gravity convention of the design
        document; with them the center of mass.

    Returns
    -------
    tuple
        (center, member_indices, weights). `weights` sums to one and is the
        derivative of the center with respect to the member positions, so a
        wall acting on the center back-propagates its gradient by
        dV/dR_a = weight_a * dV/d(center).
    """
    xyz = _as_coords(coords)
    members = np.asarray(indices, dtype=np.int64).reshape(-1)
    if members.size == 0:
        raise CVSingularError('a group center needs at least one atom')
    if np.any(members < 0) or np.any(members >= xyz.shape[0]):
        raise CVSingularError('group center index out of range')
    if masses is None:
        weights = np.full(members.size, 1.0 / members.size)
    else:
        values = np.asarray(masses, dtype=np.float64).reshape(-1)
        if values.size != xyz.shape[0]:
            raise CVSingularError(
                'mass array has %d entries for %d atoms'
                % (values.size, xyz.shape[0]))
        picked = values[members]
        total = float(picked.sum())
        if total <= 0.0:
            raise CVSingularError(
                'a group center needs a positive total mass, got %.3e'
                % total)
        weights = picked / total
    # The center is the weighted sum divided once, which is the arithmetic
    # the integrator's center-of-mass removal has always used; sharing this
    # helper must not move that number by a unit in the last place (design
    # document section 18, D-04). `weights` is the derivative of the center
    # with respect to the member positions and stays normalized to one.
    if masses is None:
        center = np.sum(xyz[members] * weights[:, None], axis=0)
    else:
        center = np.sum(xyz[members] * picked[:, None], axis=0) / total
    return center, members, weights


def bond_core(xyz, i, j):
    """Bond distance and the two-row derivative, `dq/dR` shaped (2, 3)."""
    vector = xyz[i] - xyz[j]
    value = float(np.linalg.norm(vector))
    if value < SINGULAR_LENGTH:
        raise CVSingularError(
            'zero-length bond between atoms %d and %d (%.3e)' % (i, j, value))
    unit = vector / value
    return value, np.vstack((unit, -unit))


def angle_core(xyz, i, j, k):
    """Angle i-j-k at the central atom j, in radians, with its (3, 3) Jacobian."""
    r1 = xyz[i] - xyz[j]
    r2 = xyz[k] - xyz[j]
    l1 = float(np.linalg.norm(r1))
    l2 = float(np.linalg.norm(r2))
    if l1 < SINGULAR_LENGTH or l2 < SINGULAR_LENGTH:
        raise CVSingularError(
            'angle %d-%d-%d has a zero-length arm (%.3e, %.3e)'
            % (i, j, k, l1, l2))
    cosine = float(np.dot(r1, r2)) / l1 / l2
    value = _acos(cosine, 'angle %d-%d-%d' % (i, j, k))
    sine = float(np.sin(value))
    if abs(sine) < SINGULAR_SINE:
        raise CVSingularError(
            'angle %d-%d-%d is collinear (%.6f rad); its derivative is '
            'undefined' % (i, j, k, value))
    factor = -1.0 / (sine * l1 * l2)
    d1 = factor * (r2 - r1 * cosine * l2 / l1)
    d2 = factor * (r1 - r2 * cosine * l1 / l2)
    return value, np.vstack((d1, -d1 - d2, d2))


def torsion_core(xyz, i, j, k, m):
    """Dihedral i-j-k-m in radians, in (-pi, pi], with its exact (4, 3) Jacobian.

    With b1 = x_j - x_i, b2 = x_k - x_j, b3 = x_m - x_k and the plane normals
    n1 = b1 x b2, n2 = b2 x b3, the two scalar invariants of the torsion are

        x = n1 . n2                  (= |n1| |n2| cos phi)
        y = |b2| b1 . (b2 x b3)      (= |n1| |n2| sin phi)

    by the Lagrange identity, so phi = atan2(y, x) is defined everywhere the
    two planes exist and x*x + y*y = |n1|^2 |n2|^2 never vanishes on a
    non-degenerate geometry. That removes both failure modes of an arccos form:
    the domain breach, and the sign read off a separate test.

    The Jacobian is the exact directional derivative of the same two invariants,
    d phi = (x dy - y dx) / (x^2 + y^2), with the bond vectors differentiated
    atom by atom, so it is the derivative of the value this function returns by
    construction. `tests/test_geometric.py` checks it against central
    differences at three step sizes and pins cis, trans and gauche.
    """
    b1 = xyz[j] - xyz[i]
    b2 = xyz[k] - xyz[j]
    b3 = xyz[m] - xyz[k]
    length2 = float(np.linalg.norm(b2))
    if length2 < SINGULAR_LENGTH:
        raise CVSingularError(
            'dihedral %d-%d-%d-%d has a zero-length central bond (%.3e)'
            % (i, j, k, m, length2))
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    s1 = float(np.linalg.norm(n1))
    s2 = float(np.linalg.norm(n2))
    if s1 < SINGULAR_LENGTH or s2 < SINGULAR_LENGTH:
        raise CVSingularError(
            'dihedral %d-%d-%d-%d is collinear at one end (%.3e, %.3e); the '
            'rotation plane is undefined' % (i, j, k, m, s1, s2))
    x = float(np.dot(n1, n2))
    det = float(np.dot(b1, np.cross(b2, b3)))
    y = length2 * det
    value = float(np.arctan2(y, x))
    scale = x * x + y * y
    if scale < SINGULAR_SINE ** 2:
        raise CVSingularError(
            'dihedral %d-%d-%d-%d is numerically undefined (|n1 n2| = %.3e)'
            % (i, j, k, m, scale))

    # Each atom moves the bond vectors it is an endpoint of: i pulls b1,
    # j pulls b1 and b2 in opposite senses, and so on along the chain.
    touches = ((-1, 0, 0), (1, -1, 0), (0, 1, -1), (0, 0, 1))
    jacobian = np.zeros((4, 3), dtype=np.float64)
    for row, weights in enumerate(touches):
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = 1.0
            db1 = weights[0] * step
            db2 = weights[1] * step
            db3 = weights[2] * step
            dn1 = np.cross(db1, b2) + np.cross(b1, db2)
            dn2 = np.cross(db2, b3) + np.cross(b2, db3)
            dx = float(np.dot(dn1, n2)) + float(np.dot(n1, dn2))
            ddet = (float(np.dot(db1, np.cross(b2, b3)))
                    + float(np.dot(b1, np.cross(db2, b3)))
                    + float(np.dot(b1, np.cross(b2, db3))))
            dlength = float(np.dot(b2, db2)) / length2
            dy = dlength * det + length2 * ddet
            jacobian[row, axis] = (x * dy - y * dx) / scale
    return value, jacobian


# ---------------------------------------------------------------------------
# Second derivatives of the same three variables.
#
# A harmonic bias on a variable q is V = k (q - q0)^2, whose curvature is
#
#     d2V/dR2 = k (2 J J^T + (q - q0) Q),      Q = d2q/dR2
#
# so the exact Hessian of a biased run needs the Jacobian (already here) and Q.
# `NumericalHessian` reaches the same number by differencing displaced gradients,
# each of which carries the bias force, so this is not the only route to the
# curvature - it is the route that is exact and costs no evaluation. Q is
# therefore derived here in closed form, for the two variables whose second
# derivative is settled; the torsion is refused by name rather than guessed, and
# the refusal carries the derivation debt (Appendix B section 10 of
# `dev/methods/CONSTRAINT_DEVELOPMENT.md`).
#
# Angle. With r1 = x_i - x_j, r2 = x_k - x_j, unit vectors u1, u2, m = u1 . u2,
# s = sin(theta), P(r) = (I - u u^T)/|r| and the projected pairs
# a = P(r1) u2, b = P(r2) u1, the first derivatives are
#
#     grad_1 theta = -a/s,      grad_2 theta = -b/s
#
# Differentiating once more, in the (r1, r2) variables, gives
#
#     d a / d r1 = -(m P1 + a u1^T + u1 a^T)/l1   (symmetric, as it must be)
#     d b / d r2 = -(m P2 + b u2^T + u2 b^T)/l2
#     d a / d r2 = (d b / d r1)^T = -P1 P2
#
# and each block then carries the derivative of the 1/s prefactor,
# -m (outer of the two gradients)/s^3. Verified against central differences of
# the analytic gradient in `tests/test_geometric.py`.
# ---------------------------------------------------------------------------


def bond_hessian_core(xyz, i, j):
    """`d2r_ij/dR2` for one bond, as a (2, 2, 3, 3) block in the order (i, j).

    The distance's second derivative is the perpendicular projector over the
    length, so the four blocks are A, -A, -A, A with A = (I - u u^T)/r.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    vector = xyz[i] - xyz[j]
    length = float(np.linalg.norm(vector))
    if length < SINGULAR_LENGTH:
        raise CVSingularError(
            'zero-length bond between atoms %d and %d (%.3e)' % (i, j, length))
    unit = vector / length
    block = (np.eye(3) - np.outer(unit, unit)) / length
    out = np.zeros((2, 2, 3, 3), dtype=np.float64)
    out[0, 0] = block
    out[1, 1] = block
    out[0, 1] = -block
    out[1, 0] = -block
    return out


def angle_hessian_core(xyz, i, j, k):
    """`d2 theta_ijk / dR2` as a (3, 3, 3, 3) block in the order (i, j, k)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    r1 = xyz[i] - xyz[j]
    r2 = xyz[k] - xyz[j]
    l1 = float(np.linalg.norm(r1))
    l2 = float(np.linalg.norm(r2))
    if l1 < SINGULAR_LENGTH or l2 < SINGULAR_LENGTH:
        raise CVSingularError(
            'angle %d-%d-%d has a zero-length arm (%.3e, %.3e)'
            % (i, j, k, l1, l2))
    u1 = r1 / l1
    u2 = r2 / l2
    cosine = float(np.dot(u1, u2))
    value = _acos(cosine, 'angle %d-%d-%d' % (i, j, k))
    sine = float(np.sin(value))
    if abs(sine) < SINGULAR_SINE:
        raise CVSingularError(
            'angle %d-%d-%d is collinear (%.6f rad); its second derivative is '
            'undefined' % (i, j, k, value))
    p1 = (np.eye(3) - np.outer(u1, u1)) / l1
    p2 = (np.eye(3) - np.outer(u2, u2)) / l2
    a = p1 @ u2
    b = p2 @ u1
    s3 = sine ** 3
    h11 = ((cosine * p1 + np.outer(a, u1) + np.outer(u1, a)) / l1 / sine
           - cosine * np.outer(a, a) / s3)
    h22 = ((cosine * p2 + np.outer(b, u2) + np.outer(u2, b)) / l2 / sine
           - cosine * np.outer(b, b) / s3)
    h12 = (-(p1 @ p2) / sine - cosine * np.outer(a, b) / s3)
    h21 = (-(p2 @ p1) / sine - cosine * np.outer(b, a) / s3)
    # r1 and r2 both point away from the vertex, so a derivative with respect to
    # the vertex is minus the sum of the two arm derivatives.
    signs = {0: (1.0, 0.0), 1: (-1.0, -1.0), 2: (0.0, 1.0)}
    out = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for left in range(3):
        sa, sb = signs[left]
        for right in range(3):
            ca, cb = signs[right]
            out[left, right] = (sa * ca * h11 + sa * cb * h12
                                + sb * ca * h21 + sb * cb * h22)
    return out


def torsion_hessian_core(xyz, i, j, k, m):
    """`d2 phi_ijkl / dR2` as a (4, 4, 3, 3) block in the order (i, j, k, m).

    The torsion is atan2(y, x) with x = n1 . n2, y = |b2| det and the bond
    vectors b1, b2, b3 linear in the coordinates, so every second derivative of
    a `b` vanishes and the whole Hessian is built from slot insertions:

        D_v D_w x = (D2n1_vw . n2) + (D_v n1 . D_w n2)
                    + (D_w n1 . D_v n2) + (n1 . D2n2_vw)
        D_v D_w det = the six terms with one b inserted in two of the three slots
        D_v D_w y   = (C2 . P2 B2) det + (u2 . B2) T_w
                      + (u2 . C2) T_v + |b2| D_v D_w det

    with P2 = (I - u2 u2^T)/|b2| the projector that differentiates a unit vector,
    and then the two quotient derivatives of atan2. Written this way - one
    bilinear evaluation per pair of coordinate directions - it is symmetric and
    translation invariant by construction, which is also the cheapest way to see
    that no term was dropped.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    b1 = xyz[j] - xyz[i]
    b2 = xyz[k] - xyz[j]
    b3 = xyz[m] - xyz[k]
    length = float(np.linalg.norm(b2))
    if length < SINGULAR_LENGTH:
        raise CVSingularError(
            'dihedral %d-%d-%d-%d has a zero-length central bond (%.3e)'
            % (i, j, k, m, length))
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    if float(np.linalg.norm(n1)) < SINGULAR_LENGTH \
            or float(np.linalg.norm(n2)) < SINGULAR_LENGTH:
        raise CVSingularError(
            'dihedral %d-%d-%d-%d is collinear at one end; its second '
            'derivative is undefined' % (i, j, k, m))
    x = float(np.dot(n1, n2))
    det = float(np.dot(b1, np.cross(b2, b3)))
    y = length * det
    scale = x * x + y * y
    if scale < SINGULAR_SINE ** 2:
        raise CVSingularError(
            'dihedral %d-%d-%d-%d is numerically undefined (|n1 n2| = %.3e); '
            'no curvature can be taken' % (i, j, k, m, scale))
    unit2 = b2 / length
    projector = (np.eye(3) - np.outer(unit2, unit2)) / length
    touches = ((-1, 0, 0), (1, -1, 0), (0, 1, -1), (0, 0, 1))

    def direction(atom, axis):
        step = np.zeros(3)
        step[axis] = 1.0
        weights = touches[atom]
        db1 = float(weights[0]) * step
        db2 = float(weights[1]) * step
        db3 = float(weights[2]) * step
        dn1 = np.cross(db1, b2) + np.cross(b1, db2)
        dn2 = np.cross(db2, b3) + np.cross(b2, db3)
        t = (float(np.dot(db1, np.cross(b2, b3)))
             + float(np.dot(b1, np.cross(db2, b3)))
             + float(np.dot(b1, np.cross(b2, db3))))
        lx = float(np.dot(unit2, db2))
        return {'db1': db1, 'db2': db2, 'db3': db3, 'dn1': dn1, 'dn2': dn2,
                'x': float(np.dot(dn1, n2) + np.dot(n1, dn2)), 't': t, 'l': lx,
                'y': lx * det + length * t}

    dirs = [direction(atom, axis) for atom in range(4) for axis in range(3)]

    def mixed(dv, dw):
        b1v, b2v, b3v = dv['db1'], dv['db2'], dv['db3']
        c1w, c2w, c3w = dw['db1'], dw['db2'], dw['db3']
        d2n1 = np.cross(b1v, c2w) + np.cross(c1w, b2v)
        d2n2 = np.cross(b2v, c3w) + np.cross(c2w, b3v)
        xx = (float(np.dot(d2n1, n2)) + float(np.dot(dv['dn1'], dw['dn2']))
              + float(np.dot(dw['dn1'], dv['dn2'])) + float(np.dot(n1, d2n2)))
        tt = (float(np.dot(b1v, np.cross(c2w, b3)))
              + float(np.dot(b1v, np.cross(b2, c3w)))
              + float(np.dot(c1w, np.cross(b2v, b3)))
              + float(np.dot(b1, np.cross(b2v, c3w)))
              + float(np.dot(c1w, np.cross(b2, b3v)))
              + float(np.dot(b1, np.cross(c2w, b3v))))
        ll = float(np.dot(c2w, projector @ b2v))
        yy = (ll * det + dv['l'] * dw['t'] + dw['l'] * dv['t'] + length * tt)
        numerator = (dw['x'] * dv['y'] + x * yy
                     - dw['y'] * dv['x'] - y * xx)
        first = x * dv['y'] - y * dv['x']
        return (numerator / scale
                - first * (2.0 * x * dw['x'] + 2.0 * y * dw['y']) / scale ** 2)

    out = np.zeros((4, 4, 3, 3), dtype=np.float64)
    for left in range(4):
        for axis_left in range(3):
            dv = dirs[left * 3 + axis_left]
            for right in range(4):
                for axis_right in range(3):
                    out[left, right, axis_left, axis_right] = mixed(
                        dv, dirs[right * 3 + axis_right])
    return out


def cv_hessian_block(kind, xyz, indices):
    """`d2q/dR2` of one variable as a (k, k, 3, 3) block, or None.

    The three kinds the CV layer knows are all served. Anything else returns
    None, which callers treat as a refusal by name rather than as zero
    curvature: an unwritten term and a vanishing one are not the same claim.
    """
    atoms = [int(atom) for atom in indices]
    if kind == 'bond':
        return bond_hessian_core(xyz, atoms[0], atoms[1]), atoms
    if kind == 'angle':
        return angle_hessian_core(xyz, atoms[0], atoms[1], atoms[2]), atoms
    if kind == 'dihedral':
        return (torsion_hessian_core(xyz, atoms[0], atoms[1], atoms[2],
                                    atoms[3]), atoms)
    return None


def _scatter(xyz, rows, indices):
    """Place a per-atom Jacobian block into a full (natom, 3) array."""
    full = np.zeros_like(xyz, dtype=np.float64)
    for position, atom in enumerate(indices):
        full[int(atom)] = rows[position]
    return full


def BND(xyz, var, grad=False):
    """Bond length between two atoms, in the length unit of the input.

    Parameters
    ----------
    xyz : np.ndarray
        Coordinates array, shape (natom, 3).
    var : list
        Atom indices [a, b].
    grad : bool
        If True, also return dq/dR as a full (natom, 3) array.

    Raises
    ------
    CVSingularError
        The two atoms coincide, so the bond has no direction.
    """
    a, b = int(var[0]), int(var[1])
    value, rows = bond_core(xyz, a, b)
    return value if not grad else (value, _scatter(xyz, rows, (a, b)))


def AGL(xyz, var, grad=False):
    """Angle a<-b>c in radians, with the derivative in radians per length.

    Parameters
    ----------
    xyz : np.ndarray
        Coordinates array, shape (natom, 3).
    var : list
        Atom indices [a, b, c], the central atom last-but-one.
    grad : bool
        If True, also return d(angle)/dR as a full (natom, 3) array.

    Raises
    ------
    CVSingularError
        An arm has zero length, or the three atoms are collinear, where the
        derivative of an angle is undefined. The old form returned nan for
        both, which is the failure mode this replaces.
    """
    a, b, c = int(var[0]), int(var[1]), int(var[2])
    value, rows = angle_core(xyz, a, b, c)
    return value if not grad else (value, _scatter(xyz, rows, (a, b, c)))


def DHD(xyz, var, grad=False):
    """Dihedral a-b-c-d in radians, signed and in (-pi, pi].

    Parameters
    ----------
    xyz : np.ndarray
        Coordinates array, shape (natom, 3).
    var : list
        Atom indices [a, b, c, d].
    grad : bool
        If True, also return d(dihedral)/dR as a full (natom, 3) array.

    Raises
    ------
    CVSingularError
        The central bond vanishes, one end is collinear, or the two rotation
        planes are parallel (an exactly eclipsed geometry). Each of these used
        to produce a nan value and a nan gradient together.
    """
    a, b, c, d = (int(var[0]), int(var[1]), int(var[2]), int(var[3]))
    value, rows = torsion_core(xyz, a, b, c, d)
    return value if not grad else (value, _scatter(xyz, rows, (a, b, c, d)))
