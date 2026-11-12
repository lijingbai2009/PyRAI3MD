"""
======================================================================
Optimal-superposition RMSD as a collective variable.

The variable is the squared root-mean-square deviation after the best rigid
superposition of a reference geometry:

    D^2 = (1/W) sum_i w_i |q_i^c - R p_i^c|^2

with `c` the weighted-centroid removal, `w_i` the weights (masses, or one for
an unweighted RMSD) and `W = sum_i w_i`. `R` is the rotation that minimizes the
sum, so `D^2` is invariant under a rigid motion of either geometry and is zero
only when the two differ by nothing but a rotation and a translation.

Two properties make this form worth using as a bias variable:

- `D^2` is single-valued even where the optimal rotation is not. Every optimal
  rotation gives the same sum, because the minimum equals an expression in the
  singular values of the cross-covariance matrix alone.
- the per-atom derivative is then NOT always single-valued, because it is
  `2 w_i` times the residual `q_i^c - R p_i^c`, which moves with the choice of
  `R`. So a value request succeeds wherever the singular-value formula makes
  sense, and a gradient request refuses when the rotation is ambiguous.

The ambiguity test is sharper than "a singular value is small", and it is worth
the extra line because the loose version refuses geometry that is perfectly
usable. A free direction costs nothing only when the *reference* block has
extent along it: if `sigma_i` vanishes while the reference has spread along the
matching right singular vector, two equally good rotations put that spread in
different places and the per-atom residual differs. A collinear group and a
planar group fitted to a planar reference both have vanishing singular values
and are still fine, because in each case the undetermined rotation acts on
directions the reference does not occupy. What genuinely fails is a geometry
that has collapsed relative to its own reference: a flat current block against a
three-dimensional reference, or a block whose atoms all coincide.

The gradient needs no derivative of the rotation. At the optimal `R` the
stationarity condition of the Procrustes problem kills that term (the envelope
theorem), and because both geometries are centered with the same weights the
residual sums to zero, so the derivative with respect to the *uncentered* input
coordinates is exactly `2 w_i r_i`. tests/test_rmsd.py checks that against
central differences rather than trusting the argument.

Degenerate geometry raises instead of returning nan, for the reason given in
utilities/cv.py: a nan here flows into a hill centre and a bias gradient and the
run continues with a structure that is quietly wrong.

Why this file owns the arithmetic rather than the framework helper: this is the
package's single Kabsch implementation, and
`workflow/integrator/operations.py::kabsch` is a scalar wrapper over
`rmsd_squared` (defect D-10 of the design document, section 18, closed by the
merge round). The deduplication could not run the other way: this module is a
leaf that must not import `workflow` (the rule restated at the end of this
header, and the import discipline the shadow build was checked with in section
17), while `operations.py` already imports from `utilities` - so the higher
layer consumes the lower one. The uncentred, unweighted cross-covariance SVD
that used to be duplicated there is deleted, along with its claim to return a
rotation; the centring, weighting and rotation below are what it now calls. The
degeneracy guard travels with the arithmetic rather than with either caller: a
gradient request refuses where the optimal rotation is not unique, and the
wrapper exposes no derivative, so it cannot bypass the refusal.

Units: lengths in the unit of the coordinates passed in (the core uses Bohr),
so `D^2` is in that unit squared and its derivative in that unit.

The module must stay a leaf: numpy only, no imports from schema, data, workflow
or skills.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# A singular value below this fraction of the largest one leaves its rotation
# direction undetermined; whether that freedom matters is then decided by how
# much reference weight the direction carries, not by the ratio alone.
AMBIGUITY_RATIO = 1.0e-8


class RmsdError(ValueError):
    """The superposition is not defined, or not unique, for this geometry."""


def weights_of(count, masses=None):
    """Normalized weights for a group: masses over their sum, or uniform.

    Returns
    -------
    numpy.ndarray
        (count,) float64 summing to one.
    """
    if masses is None:
        return np.full(int(count), 1.0 / count)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    if values.size != int(count):
        raise RmsdError(
            'the group holds %d atoms but the weights hold %d'
            % (count, values.size))
    total = float(values.sum())
    if total <= 0.0:
        raise RmsdError(
            'a mass-weighted RMSD needs a positive total mass, got %.3e'
            % total)
    if np.any(values <= 0.0):
        raise RmsdError('a mass-weighted RMSD needs positive masses')
    return values / total


def centered(coords, weights):
    """The geometry with its weighted centroid moved to the origin."""
    array = _pairs(coords)
    return array - np.sum(array * weights[:, None], axis=0)


def rmsd_squared(coords, reference, weights=None, grad=False):
    """Squared RMSD between two geometry blocks after optimal superposition.

    The single superposition entry point of the package: the RMSD history
    potential calls it for the variable and its Jacobian, and
    `workflow/integrator/operations.py::kabsch` calls it for the square root of
    the value alone.

    Parameters
    ----------
    coords : array-like
        (k, 3) current positions, in any consistent length unit.
    reference : array-like
        (k, 3) reference positions, same unit, same atom order.
    weights : array-like, optional
        Per-atom weights, usually masses. Normalized here, so any consistent
        scale works. Without them the plain mean over the group is used.
    grad : bool
        Also return d(D^2)/d(coords), shaped (k, 3).

    Returns
    -------
    float or tuple
        `D^2`, or `(D^2, jacobian)`. The jacobian differentiates the squared
        deviation, in the length unit of the inputs.

    Raises
    ------
    RmsdError
        The blocks disagree in shape, the weights are unusable, or `grad` was
        asked for where the optimal rotation is not unique.
    """
    current = _pairs(coords)
    target = _pairs(reference)
    if current.shape != target.shape:
        raise RmsdError(
            'the reference block holds %d atoms but the geometry holds %d'
            % (target.shape[0], current.shape[0]))
    if current.shape[0] < 2:
        raise RmsdError(
            'an aligned RMSD needs at least two atoms; one atom has no '
            'orientation, so its RMSD is a translation, not a variable')
    w = weights_of(current.shape[0], weights)
    q = centered(current, w)
    p = centered(target, w)
    if np.array_equal(q, p):
        # Two identical blocks have exactly zero deviation by definition, while
        # the trace form would return the round-off of a difference between two
        # large nearly equal sums - about 3e-8 after the square root, on the
        # scale of the coordinates. A caller asking whether a geometry has
        # moved at all must get zero, and a gradient of zero is the right
        # gradient there, because the residual of the superposition is zero.
        return 0.0 if not grad else (0.0, np.zeros_like(q))
    rotation, singular, determinant, free = _kabsch(q, p, w)
    value = _trace_form(q, p, w, singular, determinant)
    if not grad:
        return value
    if rotation is None:
        raise RmsdError(_ambiguous(free, singular))
    residual = q - rotation.apply(p)
    jacobian = 2.0 * residual * w[:, None]
    # The residual of the optimal superposition of two equally weighted,
    # equally centered blocks sums to zero by construction, which is what makes
    # this derivative carry no centroid term. It is checked, not assumed.
    total = np.sum(jacobian, axis=0)
    if float(np.max(np.abs(total))) > 1.0e-6 * max(
            1.0, float(np.max(np.abs(jacobian)))):
        raise RmsdError(
            'the aligned RMSD gradient does not cancel in translation; the '
            'superposition and the weights disagree')
    return value, jacobian


class _Rotation(object):
    """The optimal rotation, applied row-wise to a geometry block."""

    __slots__ = ('matrix',)

    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, block):
        return block @ self.matrix.T


def _kabsch(q, p, w):
    """The optimal rotation taking the reference onto the current geometry.

    This is the only Kabsch rotation in the package: `data/geometry.py::kabsch`
    reaches it by calling `rmsd_squared` above rather than by carrying its own
    SVD, so the framework helper and the collective variable cannot drift apart.

    Maximizing `sum_i w_i q_i . R p_i` over rotations is the Procrustes
    problem: with the weighted cross-covariance `C = sum_i w_i outer(q_i, p_i)`
    and its singular value decomposition `C = U s Vt`, the maximizer is
    `R = U diag(1, 1, d) Vt` with `d = sign(det(U Vt))`, which is the correction
    that keeps a reflected solution out of a mirror-image reference.

    Returns `(rotation_or_None, singular_values, determinant_sign, free)`. The
    rotation is None exactly when an undetermined direction still carries the
    reference, and `free` names the direction that said so, so the refusal can
    point at the geometry instead of at the arithmetic. The caller must treat
    that None as "no gradient": `rmsd_squared` raises on it, and nothing else in
    the package is allowed to hand back a residual built from a rotation that
    was not unique.
    """
    cross = q.T @ (p * w[:, None])
    left, singular, right = np.linalg.svd(cross)
    if float(singular[0]) <= 0.0:
        return None, singular, 1.0, None
    sign = _reflection_sign(left, right)
    corrected = np.diag([1.0, 1.0, sign])
    matrix = left @ corrected @ right
    free = _free_direction_holding_reference(singular, right, p, w)
    if free is not None:
        return None, singular, sign, free
    return _Rotation(matrix), singular, sign, None


def _reflection_sign(left, right):
    """+1 or -1: the determinant correction of a reflected superposition."""
    value = float(np.linalg.det(left @ right))
    if abs(value) < 1.0e-6:
        # A determinant this small means the smallest singular value is zero,
        # which is the planar case: the sign is decided by the next direction
        # and any choice gives the same residual, so call it +1.
        return 1.0
    return 1.0 if value > 0.0 else -1.0


def _free_direction_holding_reference(singular, right, p, w):
    """The index of an undetermined rotation direction the reference occupies.

    A direction whose singular value is numerically zero leaves the rotation
    about it free. That freedom costs nothing when the reference block has no
    spread along the direction, because `R p_i` does not depend on the choice;
    it is fatal when the reference does have spread there, because two equally
    good rotations then disagree about every atom's residual and the gradient
    stops being a function of the geometry.

    Returns the offending index, or None when the rotation is effectively
    unique. Collinear and planar groups land on None, which is the point of
    comparing against the reference scatter rather than the singular values
    alone: those groups are usable, and a test says so.
    """
    largest_sigma = float(singular[0])
    scatter = p.T @ (p * w[:, None])
    extent = np.einsum('ij,jk,ik->i', right, scatter, right)
    largest_extent = float(np.max(extent))
    for index in range(3):
        if float(singular[index]) > AMBIGUITY_RATIO * largest_sigma:
            continue
        if largest_extent > 0.0 and float(extent[index]) \
                > AMBIGUITY_RATIO * largest_extent:
            return index
    return None


def _trace_form(q, p, w, singular, sign):
    """`D^2` from the singular values, without needing a unique rotation.

    `sum_i w_i |q_i - R p_i|^2` at the optimum is
    `sum_i w_i |q_i|^2 + sum_i w_i |p_i|^2 - 2 (s1 + s2 + d s3)`, and the
    weighted norms of the centered blocks are the first two terms. Since the
    blocks are centered with the same weights, the cross terms vanish, which is
    what makes this identity exact rather than approximate.
    """
    current = float(np.sum(w * np.sum(q * q, axis=1)))
    target = float(np.sum(w * np.sum(p * p, axis=1)))
    best = float(singular[0] + singular[1] + sign * singular[2])
    value = current + target - 2.0 * best
    # Cancellation at near-perfect superposition can leave a small negative
    # number; the quantity is a sum of squares, so clamp rather than report it.
    return max(value, 0.0)


def _ambiguous(free, singular):
    """The refusal, named by the geometry that caused it."""
    if free is None:
        return ('the group has no extent: every atom of one block sits at the '
                'same point, so no rotation is preferred and the gradient of '
                'the aligned RMSD does not exist')
    return ('the current geometry has collapsed relative to its reference: '
            'singular direction %d is free (%.3e against a largest of %.3e) '
            'while the reference still carries spread along it, so two equally '
            'good superpositions disagree on the per-atom residual and the '
            'gradient of the aligned RMSD is not a function of the geometry'
            % (free, float(singular[free]), float(singular[0])))


def _pairs(block):
    """A float64 (k, 3) array, checked."""
    array = np.asarray(block, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise RmsdError(
            'an RMSD block must be (k, 3), got shape %s' % (array.shape,))
    if not np.all(np.isfinite(array)):
        raise RmsdError('an RMSD block holds a non-finite number')
    return array
