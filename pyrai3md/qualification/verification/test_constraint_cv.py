"""
======================================================================
Tests for the collective-variable layer of the constraint shadow build.

Every analytical Jacobian here is checked against a central difference at
three step sizes, which is the acceptance gate the design document sets:
1e-7 Hartree per Bohr absolute on a smooth, non-singular geometry. Degenerate
geometry is checked to raise rather than to return nan, because a nan in a
constraint gradient is silently absorbed by a solver.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities import cv
from pyrai3md.utilities import bonds
from pyrai3md.utilities import geometric

# A non-singular water-like cluster in Bohr, with no collinear angle and no
# near-planar dihedral, so every derivative below is well conditioned.
COORDS = np.array([[0.0000, 0.0000, 0.0000],
                   [1.7900, 0.3100, 0.1200],
                   [0.4200, 1.8800, -0.2100],
                   [2.3100, 1.9500, 1.4400]])
STEPS = (1.0e-4, 1.0e-5, 1.0e-6)
GRADIENT_ATOL = 1.0e-7

# Rotation about z by 0.7 rad plus a shift, used for the invariance checks.
ROTATION = np.array([[np.cos(0.7), -np.sin(0.7), 0.0],
                     [np.sin(0.7), np.cos(0.7), 0.0],
                     [0.0, 0.0, 1.0]])
SHIFT = np.array([1.25, -0.4, 2.1])


def _move_to(coords, transform):
    """Apply a rotation and translation, for invariance checks."""
    return coords @ transform.T + SHIFT


class TestBondDistance(unittest.TestCase):
    """d|r_ij|/dR against finite differences."""

    def test_value_matches_vector_norm(self):
        expected = float(np.linalg.norm(COORDS[0] - COORDS[1]))
        self.assertAlmostEqual(cv.distance(COORDS, 0, 1), expected, places=12)

    def test_jacobian_matches_central_difference(self):
        _, jacobian = cv.distance(COORDS, 0, 1, grad=True)
        self._check(cv.distance, (0, 1), jacobian, [0, 1])

    def test_symmetry_of_pair_order(self):
        forward, first = cv.distance(COORDS, 0, 1, grad=True)
        backward, second = cv.distance(COORDS, 1, 0, grad=True)
        self.assertAlmostEqual(forward, backward, places=14)
        np.testing.assert_allclose(first, -second, atol=1.0e-14)

    def test_translation_and_rotation_invariance(self):
        moved = _move_to(COORDS, ROTATION)
        self.assertAlmostEqual(cv.distance(COORDS, 0, 2),
                               cv.distance(moved, 0, 2), places=12)

    def test_zero_length_bond_raises(self):
        with self.assertRaises(cv.CVSingularError):
            cv.distance(np.zeros((2, 3)), 0, 1, grad=True)

    def _check(self, function, indices, jacobian, atoms):
        for step in STEPS:
            for row, atom in enumerate(atoms):
                for axis in range(3):
                    forward = COORDS.copy()
                    forward[atom, axis] += step
                    back = COORDS.copy()
                    back[atom, axis] -= step
                    numeric = (function(forward, *indices)
                               - function(back, *indices)) / (2.0 * step)
                    self.assertLessEqual(
                        abs(numeric - jacobian[row, axis]),
                        GRADIENT_ATOL,
                        msg='step %g atom %d axis %d' % (step, atom, axis))


class TestBondAngle(unittest.TestCase):
    """Angles in radians, unlike utilities.geometric which returns degrees."""

    def test_value_is_radians(self):
        value = cv.angle(COORDS, 0, 1, 2)
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, np.pi)

    def test_jacobian_matches_central_difference(self):
        _, jacobian = cv.angle(COORDS, 0, 1, 2, grad=True)
        for step in STEPS:
            for row, atom in enumerate((0, 1, 2)):
                for axis in range(3):
                    forward = COORDS.copy()
                    forward[atom, axis] += step
                    back = COORDS.copy()
                    back[atom, axis] -= step
                    numeric = (cv.angle(forward, 0, 1, 2)
                               - cv.angle(back, 0, 1, 2)) / (2.0 * step)
                    self.assertLessEqual(
                        abs(numeric - jacobian[row, axis]), GRADIENT_ATOL)

    def test_gradient_sums_to_zero_over_translation(self):
        _, jacobian = cv.angle(COORDS, 0, 1, 2, grad=True)
        np.testing.assert_allclose(jacobian.sum(axis=0), np.zeros(3),
                                   atol=1.0e-12)

    def test_collinear_angle_raises_instead_of_nan(self):
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaises(cv.CVSingularError):
            cv.angle(line, 0, 1, 2, grad=True)

    def test_zero_arm_raises(self):
        collapsed = COORDS.copy()
        collapsed[2] = collapsed[1]
        with self.assertRaises(cv.CVSingularError):
            cv.angle(collapsed, 0, 1, 2)


class TestDihedral(unittest.TestCase):
    """Sign convention pinned to the textbook geometries."""

    def test_cis_is_zero_and_trans_is_pi(self):
        cis = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0], [1.5, 1.0, 0.0]])
        trans = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                          [1.5, 0.0, 0.0], [1.5, -1.0, 0.0]])
        self.assertAlmostEqual(cv.dihedral(cis, 0, 1, 2, 3), 0.0, places=12)
        self.assertAlmostEqual(abs(cv.dihedral(trans, 0, 1, 2, 3)), np.pi,
                               places=12)

    def test_gauche_is_sixty_degrees(self):
        angle = np.pi / 3.0
        gauche = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                           [1.5, 0.0, 0.0],
                           [1.5, np.cos(angle), np.sin(angle)]])
        self.assertAlmostEqual(np.degrees(cv.dihedral(gauche, 0, 1, 2, 3)),
                               60.0, places=8)

    def test_jacobian_matches_central_difference(self):
        _, jacobian = cv.dihedral(COORDS, 0, 1, 2, 3, grad=True)
        for step in STEPS:
            for row, atom in enumerate((0, 1, 2, 3)):
                for axis in range(3):
                    forward = COORDS.copy()
                    forward[atom, axis] += step
                    back = COORDS.copy()
                    back[atom, axis] -= step
                    numeric = (cv.dihedral(forward, 0, 1, 2, 3)
                               - cv.dihedral(back, 0, 1, 2, 3)) / (2.0 * step)
                    self.assertLessEqual(
                        abs(numeric - jacobian[row, axis]), GRADIENT_ATOL,
                        msg='step %g atom %d axis %d' % (step, atom, axis))

    def test_gradient_is_translation_invariant(self):
        _, jacobian = cv.dihedral(COORDS, 0, 1, 2, 3, grad=True)
        np.testing.assert_allclose(jacobian.sum(axis=0), np.zeros(3),
                                   atol=1.0e-12)

    def test_rotated_geometry_rotates_the_gradient(self):
        _, first = cv.dihedral(COORDS, 0, 1, 2, 3, grad=True)
        _, second = cv.dihedral(_move_to(COORDS, ROTATION), 0, 1, 2, 3,
                                grad=True)
        np.testing.assert_allclose(second, first @ ROTATION.T, atol=1.0e-10)

    def test_collinear_end_raises(self):
        straight = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [2.0, 0.0, 0.0], [3.0, 0.1, 0.0]])
        with self.assertRaises(cv.CVSingularError):
            cv.dihedral(straight, 0, 1, 2, 3, grad=True)


class TestCenters(unittest.TestCase):
    """Center of mass, center of gravity and their back-propagation weights."""

    def test_mass_weighted_center(self):
        masses = np.array([16.0, 1.0, 1.0, 12.0])
        center, members, weights = geometric.group_center(COORDS, [0, 1],
                                                   masses=masses)
        expected = ((COORDS[0] * 16.0 + COORDS[1] * 1.0) / 17.0)
        np.testing.assert_allclose(center, expected, atol=1.0e-14)
        np.testing.assert_allclose(weights, [16.0 / 17.0, 1.0 / 17.0],
                                   atol=1.0e-14)

    def test_unweighted_center_shares_each_atom_equally(self):
        center, _, weights = geometric.group_center(COORDS, [0, 1, 2])
        np.testing.assert_allclose(center, COORDS[[0, 1, 2]].mean(axis=0),
                                   atol=1.0e-14)
        np.testing.assert_allclose(weights, [1.0 / 3.0] * 3, atol=1.0e-14)

    def test_weights_sum_to_one_so_energy_is_counted_once(self):
        _, _, weights = geometric.group_center(COORDS, [0, 1, 3],
                                        masses=np.array([1., 2., 3., 4.]))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)

    def test_non_positive_total_mass_raises(self):
        with self.assertRaises(cv.CVSingularError):
            geometric.group_center(COORDS, [0, 1],
                            masses=np.array([0.0, 0.0, 1., 1.]))

    def test_empty_group_raises(self):
        with self.assertRaises(cv.CVSingularError):
            geometric.group_center(COORDS, [])


class TestPeriodicDifference(unittest.TestCase):

    def test_across_the_branch_cut(self):
        first = np.radians(170.0)
        second = np.radians(-170.0)
        self.assertAlmostEqual(cv.periodic_difference(first, second),
                               np.radians(-20.0), places=10)

    def test_away_from_the_branch_cut(self):
        self.assertAlmostEqual(cv.periodic_difference(np.radians(10.0),
                                                      np.radians(350.0)),
                               np.radians(20.0), places=10)

    def test_result_stays_in_the_principal_branch(self):
        for value in (-7.0, -2.0, 0.0, 1.0, 3.0, 6.5, 12.0):
            delta = cv.periodic_difference(value, 0.5)
            self.assertLessEqual(abs(delta), np.pi + 1.0e-12)


class TestScatter(unittest.TestCase):

    def test_scatter_places_rows_and_sums_duplicates(self):
        jacobian = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dense = cv.scatter_jacobian(jacobian, [2, 2], 4)
        np.testing.assert_allclose(dense[2], [5.0, 7.0, 9.0], atol=1.0e-14)
        self.assertEqual(float(np.abs(dense[[0, 1, 3]]).sum()), 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(cv.CVSingularError):
            cv.scatter_jacobian(np.ones((2, 3)), [0], 4)

    def test_zero_gradient_is_the_right_shape(self):
        self.assertEqual(cv.zero_gradient(5).shape, (5, 3))


class TestBadInput(unittest.TestCase):

    def test_malformed_coordinate_array_raises(self):
        with self.assertRaises(cv.CVSingularError):
            cv.distance(np.zeros(3), 0, 1)


class TestImageAwareEvaluation(unittest.TestCase):
    """A CV over a chain is evaluated once the component is unfolded."""

    def setUp(self):
        from pyrai3md.qualification.verification import test_geometric
        self.shared = test_geometric

    def test_angle_survives_the_face(self):
        expanded, _, _ = bonds.unfold_components(
            self.shared.TORN_WATER, self.shared.LATTICE,
            self.shared.TORN_BONDS)
        value, gradient = cv.evaluate_on_images(
            expanded, [1, 0, 2], 'angle', grad=True)
        naive, _ = geometric.AGL(self.shared.TORN_WATER, [1, 0, 2], grad=True)
        self.assertAlmostEqual(value, self.shared.ANGLE_TARGET, places=6)
        self.assertGreater(abs(value - naive), 0.5)
        self.assertEqual(gradient.shape, self.shared.TORN_WATER.shape)

    def test_no_crossing_reproduces_the_stored_result_bitwise(self):
        """DC-1 equivalence: an in-cell molecule is untouched by this layer."""
        for var, kind, core in (
                ((0, 1), 'bond', geometric.BND),
                ((1, 0, 2), 'angle', geometric.AGL),
                ((0, 1, 2, 3), 'dihedral', geometric.DHD)):
            expanded, _, _ = bonds.unfold_components(
                self.shared.GEOMETRY, self.shared.LATTICE,
                [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0)), (2, 3, (0, 0, 0))])
            np.testing.assert_array_equal(expanded, self.shared.GEOMETRY)
            value, gradient = cv.evaluate_on_images(
                expanded, list(var), kind, grad=True)
            expected, rows = core(self.shared.GEOMETRY, list(var), grad=True)
            self.assertEqual(value, expected)
            np.testing.assert_array_equal(gradient, rows)

    def test_translation_of_the_whole_molecule_is_invisible(self):
        """Half a cell along x: the pair stops being torn, the angle is the same."""
        expanded, offsets, _ = bonds.unfold_components(
            self.shared.TORN_WATER, self.shared.LATTICE,
            self.shared._perceived(self.shared.TORN_WATER))
        self.assertTrue(np.any(offsets != 0))
        moved = (self.shared.TORN_WATER
                 + np.array([self.shared.BOX / 2.0, 0.0, 0.0])) % self.shared.BOX
        unfolded, whole, _ = bonds.unfold_components(
            moved, self.shared.LATTICE, self.shared._perceived(moved))
        self.assertEqual(int(np.abs(whole).max()), 0)
        value, gradient = cv.evaluate_on_images(
            unfolded, [1, 0, 2], 'angle', grad=True)
        base, rows = cv.evaluate_on_images(
            expanded, [1, 0, 2], 'angle', grad=True)
        self.assertAlmostEqual(value, base, places=12)
        np.testing.assert_allclose(gradient, rows, atol=1e-12)

    def test_gradient_is_the_derivative_of_the_unfolded_value(self):
        expanded, _, _ = bonds.unfold_components(
            self.shared.TORN_WATER, self.shared.LATTICE,
            self.shared.TORN_BONDS)
        var = [1, 0, 2]
        value, gradient = cv.evaluate_on_images(
            expanded, var, 'angle', grad=True)
        for atom in var:
            for axis in range(3):
                plus = np.array(expanded, dtype=np.float64, copy=True)
                minus = np.array(expanded, dtype=np.float64, copy=True)
                plus[atom, axis] += 1e-6
                minus[atom, axis] -= 1e-6
                step = (cv.evaluate_on_images(plus, var, 'angle')
                        - cv.evaluate_on_images(minus, var, 'angle'))
                np.testing.assert_allclose(gradient[atom, axis], step / 2e-6,
                                           rtol=1e-5, atol=1e-8)

    def test_unknown_kind_and_wrong_arity_are_refused(self):
        with self.assertRaises(ValueError):
            cv.evaluate_on_images(self.shared.GEOMETRY, [0, 1], 'distance')
        with self.assertRaises(ValueError):
            cv.evaluate_on_images(self.shared.GEOMETRY, [0, 1], 'angle')


if __name__ == '__main__':
    unittest.main()
