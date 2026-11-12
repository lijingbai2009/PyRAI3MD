"""
======================================================================
Tests for the aligned RMSD collective variable.

The variable is the squared deviation after the best rigid superposition, so
the tests have to establish three separate things and cannot borrow evidence
between them: the number, the rotation that produces it, and the derivative of
the number with respect to the coordinates.

The number is checked against a second implementation written along a
different route (explicit centroids, the cross-covariance of the *reference*
onto the current block rather than the reverse, and a mean of squared
residuals rather than a singular-value identity). The rotation is checked by
rigid motion: an RMSD that changes when the molecule is turned is a bug, not a
variable. The derivative is checked by central difference, which is the only
test here that can catch the one non-obvious step in the derivation - the claim
that the optimal rotation itself needs no derivative.

The degeneracy cases are the part most likely to be got wrong in either
direction. A planar or a collinear group is *usable*, because the rotation
those groups leave free acts on directions the reference does not occupy; a
group that has collapsed against a three-dimensional reference is not, because
two equally good superpositions then disagree atom by atom. Both are asserted.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities import rmsd

REFERENCE = np.array([[0.00, 0.00, 0.00],
                      [1.90, 0.30, 0.10],
                      [0.40, 2.00, -0.20],
                      [2.10, 1.80, 1.30],
                      [-1.20, 0.70, 1.60],
                      [0.80, -1.50, 0.40]])
MASSES = np.array([12.0, 1.0, 1.0, 16.0, 1.0, 14.0])

# A real change of shape, not a translation. Every test that needs a nonzero
# RMSD adds this: displacing all atoms by the same vector is a rigid motion and
# would leave the variable at zero, which makes a broken alignment look correct.
DISTORTION = np.array([[0.05, -0.02, 0.01],
                       [-0.03, 0.06, 0.02],
                       [0.01, 0.02, -0.05],
                       [-0.02, -0.04, 0.03],
                       [0.04, 0.01, -0.02],
                       [-0.01, 0.03, 0.04]])


def _quaternion_rotation(seed):
    """A proper rotation from a deterministic unit quaternion."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _independent_rmsd_squared(coords, reference, weights=None):
    """Squared RMSD by explicit superposition, not by the trace identity.

    Deliberately written the other way round: the cross term is formed from the
    reference onto the current block, the rotation is assembled as V D U^T
    instead of U D V^T, and the returned number is the weighted mean of the
    squared residuals rather than a combination of singular values.
    """
    if weights is None:
        current = coords - coords.mean(axis=0)
        target = reference - reference.mean(axis=0)
        cross = target.T @ current
        left, _, right = np.linalg.svd(cross)
        sign = np.sign(np.linalg.det(right.T @ left.T))
        rotation = right.T @ np.diag([1.0, 1.0, sign]) @ left.T
        residual = current - (rotation @ target.T).T
        return float(np.mean(np.sum(residual * residual, axis=1)))
    scale = np.asarray(weights, dtype=np.float64)
    scale = scale / scale.sum()
    current = coords - np.sum(coords * scale[:, None], axis=0)
    target = reference - np.sum(reference * scale[:, None], axis=0)
    cross = (target * scale[:, None]).T @ current
    left, _, right = np.linalg.svd(cross)
    sign = np.sign(np.linalg.det(right.T @ left.T))
    rotation = right.T @ np.diag([1.0, 1.0, sign]) @ left.T
    residual = current - (rotation @ target.T).T
    return float(np.sum(scale * np.sum(residual * residual, axis=1)))


def _moved(block, seed, shift=2.5):
    """The same geometry, rotated and translated."""
    return block @ _quaternion_rotation(seed).T + shift


class TestValue(unittest.TestCase):

    def test_the_number_matches_a_second_implementation(self):
        current = _moved(REFERENCE, 1) + DISTORTION
        value = rmsd.rmsd_squared(current, REFERENCE)
        self.assertAlmostEqual(value,
                               _independent_rmsd_squared(current, REFERENCE),
                               places=12)

    def test_the_weighted_number_matches_too(self):
        current = _moved(REFERENCE, 2) + DISTORTION
        value = rmsd.rmsd_squared(current, REFERENCE, weights=MASSES)
        self.assertAlmostEqual(
            value, _independent_rmsd_squared(current, REFERENCE, MASSES),
            places=12)

    def test_a_rigid_motion_of_the_same_geometry_is_zero(self):
        """Negative control for the alignment itself.

        Without the optimal rotation a turned molecule has a large RMSD; the
        whole point of the variable is that it does not.
        """
        for seed in range(6):
            value = rmsd.rmsd_squared(_moved(REFERENCE, seed), REFERENCE)
            self.assertAlmostEqual(value, 0.0, places=14)

    def test_the_value_does_not_depend_on_which_block_is_the_reference(self):
        """D^2(q,p) == D^2(p,q): the superposition is symmetric."""
        current = _moved(REFERENCE, 3) + DISTORTION
        forward = rmsd.rmsd_squared(current, REFERENCE)
        backward = rmsd.rmsd_squared(REFERENCE, current)
        self.assertAlmostEqual(forward, backward, places=13)

    def test_a_mirror_image_is_not_superimposed_onto_itself(self):
        """The determinant correction, asserted by its consequence.

        The negative control is the same superposition with the correction
        dropped: it reports zero deviation for a mirror image, which is not a
        rigid motion, so the guard is load-bearing rather than decorative.
        """
        mirrored = REFERENCE.copy()
        mirrored[:, 2] *= -1.0
        value = rmsd.rmsd_squared(mirrored, REFERENCE)
        self.assertGreater(value, 1.0e-3)
        current = mirrored - mirrored.mean(axis=0)
        target = REFERENCE - REFERENCE.mean(axis=0)
        left, _, right = np.linalg.svd(current.T @ target)
        naive = right.T @ np.diag([1.0, 1.0, 1.0]) @ left.T
        residual = target - (naive @ current.T).T
        self.assertLess(float(np.mean(np.sum(residual * residual, axis=1))),
                        1.0e-14)


class TestGradient(unittest.TestCase):

    def test_the_jacobian_matches_central_differences(self):
        current = _moved(REFERENCE, 4) + DISTORTION
        _, jacobian = rmsd.rmsd_squared(current, REFERENCE, grad=True)
        step = 1.0e-6
        finite = np.zeros_like(jacobian)
        for atom in range(current.shape[0]):
            for axis in range(3):
                up = current.copy()
                up[atom, axis] += step
                down = current.copy()
                down[atom, axis] -= step
                finite[atom, axis] = (
                    rmsd.rmsd_squared(up, REFERENCE)
                    - rmsd.rmsd_squared(down, REFERENCE)) / (2 * step)
        # The agreement is the evidence that the rotation needs no derivative
        # term at the optimum: a wrong envelope step would show up here as a
        # first-order error, not as noise.
        self.assertLess(float(np.max(np.abs(finite - jacobian))), 1.0e-8)

    def test_the_jacobian_matches_central_differences_when_weighted(self):
        current = _moved(REFERENCE, 5) + DISTORTION
        _, jacobian = rmsd.rmsd_squared(current, REFERENCE, weights=MASSES,
                                        grad=True)
        step = 1.0e-6
        finite = np.zeros_like(jacobian)
        for atom in range(current.shape[0]):
            for axis in range(3):
                up = current.copy()
                up[atom, axis] += step
                down = current.copy()
                down[atom, axis] -= step
                finite[atom, axis] = (
                    rmsd.rmsd_squared(up, REFERENCE, weights=MASSES)
                    - rmsd.rmsd_squared(down, REFERENCE,
                                        weights=MASSES)) / (2 * step)
        self.assertLess(float(np.max(np.abs(finite - jacobian))), 1.0e-8)

    def test_the_gradient_translates_nothing(self):
        """The variable is translation invariant, so its gradient sums to zero.

        This is the property that lets the derivative be `2 w_i r_i` with no
        centroid term, and it is the cheapest guard against having dropped one.
        """
        current = _moved(REFERENCE, 6) + DISTORTION
        _, jacobian = rmsd.rmsd_squared(current, REFERENCE, grad=True)
        self.assertLess(float(np.max(np.abs(np.sum(jacobian, axis=0)))),
                        1.0e-12)

    def test_the_gradient_is_zero_at_perfect_superposition(self):
        value, jacobian = rmsd.rmsd_squared(_moved(REFERENCE, 7), REFERENCE,
                                            grad=True)
        self.assertAlmostEqual(value, 0.0, places=14)
        self.assertLess(float(np.max(np.abs(jacobian))), 1.0e-7)


class TestDegeneracy(unittest.TestCase):
    """What the alignment can and cannot resolve."""

    def test_a_planar_group_still_has_a_gradient(self):
        """The case a singular-value-only test would wrongly refuse.

        A flat molecule has one vanishing singular value, but the free rotation
        acts on the plane normal, which the reference does not occupy, so every
        equally good rotation leaves each atom's residual unchanged. The
        distortion below is in-plane, so the geometry stays flat: refusing here
        would mean an RMSD restraint on any planar fragment never works.
        """
        planar = np.array([[0.0, 0.0, 0.0],
                           [1.5, 0.0, 0.0],
                           [0.7, 1.2, 0.0],
                           [-0.9, 0.6, 0.0]])
        in_plane = np.array([[0.05, -0.02, 0.0],
                             [0.01, 0.04, 0.0],
                             [-0.03, 0.02, 0.0],
                             [0.02, -0.01, 0.0]])
        current = _moved(planar, 8) + in_plane
        value, jacobian = rmsd.rmsd_squared(current, planar, grad=True)
        self.assertGreater(value, 0.0)
        self.assertEqual(jacobian.shape, planar.shape)

    def test_a_collinear_group_still_has_a_gradient(self):
        line = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
        current = _moved(line, 9) + DISTORTION[:3]
        _, jacobian = rmsd.rmsd_squared(current, line, grad=True)
        self.assertEqual(jacobian.shape, line.shape)

    def test_a_geometry_flat_against_a_solid_reference_refuses(self):
        """The gradient really is not a function here, so say so."""
        reference = REFERENCE + np.array([0.0, 0.0, 3.0])
        flattened = REFERENCE.copy()
        flattened[:, 2] = float(np.mean(flattened[:, 2]))
        with self.assertRaises(rmsd.RmsdError) as caught:
            rmsd.rmsd_squared(flattened, reference, grad=True)
        self.assertIn('collapsed', str(caught.exception))

    def test_the_same_collapsed_geometry_still_has_a_value(self):
        """The refusal is about the derivative, not about the number.

        Every optimal superposition gives the same sum of squares, so a value
        request has an answer even where a gradient request does not.
        """
        reference = REFERENCE + np.array([0.0, 0.0, 3.0])
        flattened = REFERENCE.copy()
        flattened[:, 2] = float(np.mean(flattened[:, 2]))
        value = rmsd.rmsd_squared(flattened, reference)
        self.assertGreater(value, 0.0)

    def test_a_block_of_one_point_has_no_preferred_rotation(self):
        point = np.zeros((REFERENCE.shape[0], 3))
        value = rmsd.rmsd_squared(point, REFERENCE)
        self.assertAlmostEqual(
            value, _independent_rmsd_squared(point, REFERENCE), places=13)
        with self.assertRaises(rmsd.RmsdError) as caught:
            rmsd.rmsd_squared(point, REFERENCE, grad=True)
        self.assertIn('no extent', str(caught.exception))


class TestRefusals(unittest.TestCase):

    def test_blocks_of_different_size_are_refused(self):
        with self.assertRaises(rmsd.RmsdError) as caught:
            rmsd.rmsd_squared(REFERENCE[:4], REFERENCE)
        self.assertIn('the reference block holds 6 atoms but the geometry '
                      'holds 4', str(caught.exception))

    def test_one_atom_is_not_a_variable(self):
        with self.assertRaises(rmsd.RmsdError) as caught:
            rmsd.rmsd_squared(REFERENCE[:1], REFERENCE[:1])
        self.assertIn('at least two atoms', str(caught.exception))

    def test_weights_must_cover_the_group(self):
        with self.assertRaises(rmsd.RmsdError):
            rmsd.weights_of(REFERENCE.shape[0], MASSES[:3])

    def test_a_non_positive_mass_is_refused_before_it_divides(self):
        broken = MASSES.copy()
        broken[2] = 0.0
        with self.assertRaises(rmsd.RmsdError):
            rmsd.weights_of(REFERENCE.shape[0], broken)

    def test_a_non_finite_coordinate_is_refused(self):
        broken = REFERENCE.copy()
        broken[1, 0] = np.nan
        with self.assertRaises(rmsd.RmsdError):
            rmsd.rmsd_squared(broken, REFERENCE)

    def test_a_ragged_block_is_refused(self):
        with self.assertRaises(rmsd.RmsdError):
            rmsd.rmsd_squared(np.zeros((6, 2)), REFERENCE)


class TestWeights(unittest.TestCase):

    def test_uniform_weights_recover_the_plain_mean(self):
        self.assertTrue(np.allclose(rmsd.weights_of(4), np.full(4, 0.25)))

    def test_the_scale_of_the_masses_does_not_matter(self):
        first = rmsd.rmsd_squared(_moved(REFERENCE, 10) + 0.05, REFERENCE,
                                 weights=MASSES)
        second = rmsd.rmsd_squared(_moved(REFERENCE, 10) + 0.05, REFERENCE,
                                   weights=3.7 * MASSES)
        self.assertAlmostEqual(first, second, places=14)

    def test_centered_means_the_reference_sits_at_the_origin(self):
        block = REFERENCE + 17.0
        centered = rmsd.centered(block, rmsd.weights_of(block.shape[0]))
        self.assertLess(float(np.max(np.abs(np.mean(centered, axis=0)))),
                        1.0e-12)


if __name__ == '__main__':
    unittest.main()
