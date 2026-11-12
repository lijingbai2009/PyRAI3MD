"""
======================================================================
The shared bond, angle and torsion helpers.

These tests arrived with the merge of the constraint subsystem, and they are
the closure proof of three registered defects in
dev/methods/CONSTRAINT_DEVELOPMENT.md section 18:

- D-01: the dihedral gradient used to be the derivative of neither its own
  value nor any torsion (only the first atom's row was right; the second and
  third were wrong in sign and magnitude, the fourth was exactly negated). The
  characterization file that pinned that shape, `constraint_lab/tests/
  test_geometric_defect_evidence.py`, was deleted rather than repaired, as the
  register required, and the pins it carried move here.
- D-02: a zero bond, a collinear angle and - worst - a strict cis or trans
  dihedral returned nan for the value and the gradient together, at the
  conformations real molecules spend most of their time in. They now raise.
- D-03: `57.2958` was written into the source instead of coming from
  `utilities/units.py`. Angles are radians here, so there is nothing to
  convert and the literal is gone.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import os
import unittest

import numpy as np

from pyrai3md.utilities import geometric
from pyrai3md.utilities import bonds as topology

MODULE_SOURCE = os.path.abspath(geometric.__file__)

GEOMETRY = np.array([[0.90, 0.40, 0.10],
                     [0.00, 1.00, 0.00],
                     [0.20, 1.80, 0.30],
                     [1.20, 2.30, -0.40],
                     [-0.60, 1.40, 1.10]])
STEPS = (1.0e-4, 1.0e-5, 1.0e-6)
TOLERANCE = 1.0e-6


def _numeric(function, xyz, var, atoms, h):
    """Central difference over the atoms the variable actually touches."""
    out = np.zeros((len(atoms), 3))
    for row, atom in enumerate(atoms):
        for axis in range(3):
            plus = np.array(xyz, dtype=np.float64, copy=True)
            minus = np.array(xyz, dtype=np.float64, copy=True)
            plus[atom, axis] += h
            minus[atom, axis] -= h
            out[row, axis] = (function(plus, var) - function(minus, var)) \
                / (2.0 * h)
    return out


class TestGradients(unittest.TestCase):

    def test_bond_gradient_is_the_unit_vector_pair(self):
        value, grad = geometric.BND(GEOMETRY, [0, 1], grad=True)
        vector = GEOMETRY[0] - GEOMETRY[1]
        unit = vector / np.linalg.norm(vector)
        np.testing.assert_allclose(grad[0], unit, atol=0.0)
        np.testing.assert_allclose(grad[1], -unit, atol=0.0)
        self.assertLess(float(np.abs(grad[2:]).max()), 1.0e-15)

    def test_angle_gradient_matches_central_differences(self):
        _, analytic = geometric.AGL(GEOMETRY, [0, 1, 2], grad=True)
        for h in STEPS:
            numeric = _numeric(geometric.AGL, GEOMETRY, [0, 1, 2], (0, 1, 2), h)
            self.assertLess(float(np.abs(analytic[[0, 1, 2]] - numeric).max()),
                            TOLERANCE)

    def test_dihedral_gradient_matches_central_differences(self):
        """The row that used to be wrong is now the derivative of the value."""
        _, analytic = geometric.DHD(GEOMETRY, [0, 1, 2, 3], grad=True)
        for h in STEPS:
            numeric = _numeric(geometric.DHD, GEOMETRY, [0, 1, 2, 3],
                               (0, 1, 2, 3), h)
            self.assertLess(float(np.abs(analytic[[0, 1, 2, 3]] - numeric).max(
            )), TOLERANCE)

    def test_every_row_of_the_dihedral_is_a_derivative(self):
        """Asserted row by row, which is the shape the old defect had.

        The pre-merge form was right on the first atom only, so a single
        norm-over-the-array check could hide a negated row by spending its
        error on the other three.
        """
        _, analytic = geometric.DHD(GEOMETRY, [0, 1, 2, 3], grad=True)
        numeric = _numeric(geometric.DHD, GEOMETRY, [0, 1, 2, 3],
                           (0, 1, 2, 3), 1.0e-5)
        for row in range(4):
            self.assertLess(
                float(np.abs(analytic[row + 0 * row] - numeric[row]).max())
                if False else
                float(np.abs(analytic[[0, 1, 2, 3]][row] - numeric[row]).max()),
                TOLERANCE, 'row %d' % row)



class TestSymmetry(unittest.TestCase):
    """A torsion is a scalar of the configuration, so a rotation changes nothing."""

    def _rotate(self, xyz, theta):
        """Rodrigues rotation about a fixed unit axis, applied to rows."""
        axis = np.array([0.37, -0.62, 0.71])
        axis = axis / np.linalg.norm(axis)
        k = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        matrix = (np.eye(3) + np.sin(theta) * k
                  + (1.0 - np.cos(theta)) * (k @ k))
        return (matrix @ xyz.T).T

    def test_the_value_and_the_gradient_row_norms_are_rotation_invariant(self):
        var = [0, 1, 2, 3]
        value, grad = geometric.DHD(GEOMETRY, var, grad=True)
        for theta in (0.3, 1.1, 2.4):
            rotated = self._rotate(GEOMETRY, theta)
            got, grad_rot = geometric.DHD(rotated, var, grad=True)
            # the comparison is wrapped, because a torsion near +-pi is the same
            # angle reported from either side of the branch cut
            self.assertLess(abs(float((got - value + np.pi)
                                      % (2.0 * np.pi) - np.pi)), 1.0e-12)
            # the row norms rotate with the frame, so compare magnitudes
            np.testing.assert_allclose(np.linalg.norm(grad_rot[var], axis=1),
                                       np.linalg.norm(grad[var], axis=1),
                                       atol=1.0e-10)

    def test_the_gradient_of_a_translated_geometry_is_unchanged(self):
        var = [0, 1, 2, 3]
        shifted = GEOMETRY + np.array([3.5, -2.0, 0.75])
        _, a = geometric.DHD(GEOMETRY, var, grad=True)
        _, b = geometric.DHD(shifted, var, grad=True)
        np.testing.assert_allclose(a, b, atol=1.0e-12)


class TestConformations(unittest.TestCase):
    """cis, trans and gauche are numbers, not nan."""

    def test_trans_is_pi_and_cis_is_zero_as_numbers_not_nan(self):
        planar_trans = np.array([[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [1.5, 0.9, 0.0],
                                 [2.5, 0.9, 0.0]])
        planar_cis = np.array([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [1.5, 0.9, 0.0],
                               [1.0, 1.8, 0.0]])
        gauche = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.5, 0.9, 0.0],
                           [1.9, 1.5, 0.6]])
        self.assertAlmostEqual(geometric.DHD(planar_trans, [0, 1, 2, 3]),
                               np.pi, places=6)
        self.assertAlmostEqual(abs(geometric.DHD(planar_cis, [0, 1, 2, 3])),
                               0.0, places=6)
        value = geometric.DHD(gauche, [0, 1, 2, 3])
        self.assertTrue(1.0 < abs(value) < 2.1, value)
        for geometry in (planar_trans, planar_cis, gauche):
            got, grad = geometric.DHD(geometry, [0, 1, 2, 3], grad=True)
            self.assertTrue(np.all(np.isfinite(grad)),
                            'a strict cis or trans used to return nan here')

    def test_the_sign_follows_the_right_handed_rotation(self):
        base = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [1.5, 0.9, 0.0],
                         [2.0, 1.5, 0.5]])
        mirror = base.copy()
        mirror[:, 2] *= -1.0
        self.assertAlmostEqual(geometric.DHD(base, [0, 1, 2, 3]),
                               -geometric.DHD(mirror, [0, 1, 2, 3]),
                               places=12)


class TestDegenerateGeometry(unittest.TestCase):
    """D-02: these raise, and they never return a nan pair."""

    def test_a_zero_length_bond_raises(self):
        xyz = GEOMETRY.copy()
        xyz[1] = xyz[0]
        with self.assertRaises(geometric.CVSingularError):
            geometric.BND(xyz, [0, 1], grad=True)

    def test_a_collinear_angle_raises(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaises(geometric.CVSingularError):
            geometric.AGL(xyz, [0, 1, 2], grad=True)

    def test_a_collinear_dihedral_raises(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        with self.assertRaises(geometric.CVSingularError):
            geometric.DHD(xyz, [0, 1, 2, 3], grad=True)

    def test_a_strict_trans_dihedral_still_gives_a_finite_gradient(self):
        """The case the arccos form could not survive.

        `arccos` reaches its domain edge and the `1 / sqrt(1 - cos^2)` factor
        divides by zero at the same moment, so the old code returned nan for
        the value and the gradient together exactly where a molecule spends
        most of its time.
        """
        trans = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [1.5, 0.9, 0.0],
                          [2.5, 0.9, 0.0]])
        value, grad = geometric.DHD(trans, [0, 1, 2, 3], grad=True)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.all(np.isfinite(grad)))
        # The value sits on the branch cut at +-pi, so a raw difference
        # between two geometries can jump by 2 pi for reasons that belong to
        # the angle's representation and not to its derivative. Each step is
        # therefore wrapped back into (-pi, pi] before it is divided.
        h = 1.0e-6
        reference = geometric.DHD(trans, [0, 1, 2, 3])
        for atom in (0, 1, 2, 3):
            for axis in range(3):
                plus = trans.copy()
                plus[atom, axis] += h
                delta = geometric.DHD(plus, [0, 1, 2, 3]) - reference
                delta = float((delta + np.pi) % (2.0 * np.pi) - np.pi)
                self.assertLess(abs(grad[atom, axis] - delta / h), 5.0e-3,
                                'atom %d axis %d' % (atom, axis))


class TestNoMagicConstants(unittest.TestCase):
    """D-03: the degree factor is gone from the source of the shared module."""

    def test_no_hardcoded_degree_factor(self):
        """Checked on the parsed numbers, not on the text.

        The module explains the old defect by naming the factor, so a substring
        search would indict the prose that documents its removal. A unit
        conversion is a number in the code; only those are forbidden here.
        """
        with open(MODULE_SOURCE, encoding='utf-8') as handle:
            source = handle.read()
        tree = ast.parse(source, filename=MODULE_SOURCE)
        numbers = [node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Constant)
                   and isinstance(node.value, float)]
        for value in numbers:
            for forbidden in (57.2958, 180.0 / np.pi, np.pi / 180.0):
                self.assertFalse(abs(value - forbidden) < 1.0e-4,
                                 '%r is a unit conversion and belongs in '
                                 'utilities/units.py' % value)

    def test_angles_are_radians_by_contract(self):
        square = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(geometric.AGL(square, [0, 1, 2]),
                               np.pi / 2.0, places=12)


class TestOneImplementationOnly(unittest.TestCase):
    """D-01's other half: the constraint layer shares this, it does not copy it."""

    def test_cv_has_no_second_torsion(self):
        from pyrai3md.utilities import cv
        self.assertIs(cv.CVSingularError, geometric.CVSingularError)
        self.assertIs(cv.torsion_core, geometric.torsion_core)
        self.assertIs(cv.angle_core, geometric.angle_core)
        self.assertIs(cv.bond_core, geometric.bond_core)

    def test_cv_dihedral_and_geometric_dhd_agree(self):
        from pyrai3md.utilities import cv
        var = [0, 1, 2, 3]
        a, ga = cv.dihedral(GEOMETRY, *var, grad=True)
        b, gb = geometric.DHD(GEOMETRY, var, grad=True)
        self.assertAlmostEqual(a, b, places=14)
        np.testing.assert_allclose(ga, gb[var], atol=0.0)


# ---------------------------------------------------------------------------
# M0-2 and M0-3: the image-aware layer (`dev/methods/CONSTRAINT_DEVELOPMENT.md` Appendix B section 2).
#
# The cell is 12 Angstrom cubic and the water is placed so that its oxygen sits
# a hair inside the origin face while one hydrogen sits a hair inside the far
# face. That is the configuration the framework stores after a wrap, and the one
# a pair-wise reduction reads as a 11.5 Angstrom bond.
# ---------------------------------------------------------------------------

BOX = 12.0
LATTICE = np.diag([BOX, BOX, BOX])
# O-H = 0.96 Angstrom with a 104.5 degree angle, placed so the oxygen sits just
# inside the origin face, one hydrogen just inside the far face (so that pair is
# torn), and the other hydrogen well inside the cell.
TORN_WATER = np.array([[0.06, 6.00, 6.00],
                       [11.10, 6.00, 6.00],
                       [0.30, 6.9295160625, 6.00]])
OH_TARGET = 0.96
ANGLE_TARGET = float(np.arccos(-0.25))
TORN_BONDS = [(0, 1, (-1, 0, 0)), (0, 2, (0, 0, 0))]


def _perceived(coordinates):
    """The bond table utilities.bonds derives from the wrapped geometry."""
    return topology.perceive_bonds(['O', 'H', 'H'], coordinates, LATTICE,
                                   pbc=[1, 1, 1])


class TestComponentExpansion(unittest.TestCase):
    """Unfolding is a property of the component, never of one constraint."""

    def test_cross_face_bond_is_the_negative_control(self):
        """The torn pair comes out at the bond length, not at L - r."""
        expanded, offsets, _ = topology.unfold_components(
            TORN_WATER, LATTICE, TORN_BONDS)
        naive = float(np.linalg.norm(TORN_WATER[0] - TORN_WATER[1]))
        unfolded = float(np.linalg.norm(expanded[0] - expanded[1]))
        self.assertAlmostEqual(naive, BOX - OH_TARGET, places=9)
        self.assertAlmostEqual(unfolded, OH_TARGET, places=9)
        self.assertEqual(int(offsets[1][0]), -1)
        self.assertEqual(int(offsets[2].max()), 0)

    def test_perceived_table_drives_the_same_expansion(self):
        """The bond table is the shape utilities.bonds already returns."""
        bonds = _perceived(TORN_WATER)
        self.assertEqual(len(bonds), 2)
        expanded, _, _ = topology.unfold_components(TORN_WATER, LATTICE, bonds)
        self.assertAlmostEqual(float(np.linalg.norm(expanded[0] - expanded[1])),
                               OH_TARGET, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(expanded[0] - expanded[2])),
                               OH_TARGET, places=6)

    def test_write_back_is_the_offset_table(self):
        expanded, offsets, _ = topology.unfold_components(
            TORN_WATER, LATTICE, TORN_BONDS)
        np.testing.assert_allclose(expanded, TORN_WATER + offsets @ LATTICE,
                                  atol=0.0)

    def test_anchor_and_row_order_do_not_change_the_answer(self):
        expanded, offsets, components = topology.unfold_components(
            TORN_WATER, LATTICE, TORN_BONDS)
        shuffled, soffset, scomponents = topology.unfold_components(
            TORN_WATER, LATTICE, list(reversed(TORN_BONDS)))
        np.testing.assert_array_equal(expanded, shuffled)
        np.testing.assert_array_equal(offsets, soffset)
        np.testing.assert_array_equal(components, scomponents)
        np.testing.assert_array_equal(offsets[0], np.zeros(3))

    def test_no_cell_is_the_identity_path(self):
        """DC-1: with no lattice nothing is moved, to the last bit."""
        expanded, offsets, components = topology.unfold_components(
            GEOMETRY, None, [(0, 1, (0, 0, 0))])
        np.testing.assert_array_equal(expanded, GEOMETRY)
        self.assertEqual(int(np.abs(offsets).max()), 0)
        np.testing.assert_array_equal(components, np.arange(GEOMETRY.shape[0]))

    def test_a_non_periodic_axis_never_shifts(self):
        bonds = [(0, 1, (-1, 1, 0)), (0, 2, (0, 0, 0))]
        expanded, offsets, _ = topology.unfold_components(
            TORN_WATER, LATTICE, bonds, pbc=[1, 0, 1])
        np.testing.assert_array_equal(offsets[:, 1], np.zeros(3))
        self.assertEqual(int(offsets[1][0]), -1)

    def test_out_of_range_bond_is_refused(self):
        with self.assertRaises(topology.ComponentError) as caught:
            topology.unfold_components(TORN_WATER, LATTICE, [(0, 7, (0, 0, 0))])
        self.assertIn('outside the structure', str(caught.exception))


class TestSecondDerivatives(unittest.TestCase):
    """The exact curvature of a bond and an angle, checked by differencing
    the analytic first derivative this module already returns. A second
    derivative is only trusted when it belongs to the same formula: D-01 was a
    gradient that was the derivative of nothing.
    """

    STEPS = (1.0e-3, 1.0e-4, 1.0e-5)

    def _fd(self, core, atoms, xyz, h):
        rows = len(atoms)
        out = np.zeros((rows, rows, 3, 3))
        for right, atom_b in enumerate(atoms):
            for axis in range(3):
                plus = np.array(xyz, dtype=np.float64, copy=True)
                minus = np.array(xyz, dtype=np.float64, copy=True)
                plus[atom_b, axis] += h
                minus[atom_b, axis] -= h
                gp = np.asarray(core(plus, *atoms)[1], dtype=np.float64)
                gm = np.asarray(core(minus, *atoms)[1], dtype=np.float64)
                out[:, right, :, axis] = (gp - gm) / (2 * h)
        return out

    def _check(self, core, analytic, atoms, xyz):
        """Agreement plus convergence: the budget is the step\'s own error.

        A central difference of an analytic gradient is O(step^2) away from the
        true curvature, measured at 2.5 * step^2 relative on this fixture, so
        the tolerance is declared per step instead of one flat number that a
        coarse step would break and a fine one would not prove. The shrinking
        assertion is the part that says the two are the same object: matching at
        one step size could be an accident, matching while converging at
        second order cannot.
        """
        dense = analytic[np.ix_(atoms, atoms)]
        scale = float(np.abs(dense).max())
        self.assertGreater(scale, 0.0)
        previous = None
        for step in self.STEPS:
            numeric = self._fd(core, atoms, xyz, step)
            error = float(np.abs(dense - numeric).max()) / scale
            self.assertLess(error, 4.0 * step * step)
            if previous is not None:
                self.assertLess(error, previous)
            previous = error

    def test_bond_second_derivative_matches_its_gradient(self):
        analytic = geometric.bond_hessian_core(GEOMETRY, 0, 1)
        np.testing.assert_allclose(analytic,
                                   np.transpose(analytic, (0, 1, 3, 2)),
                                   atol=0.0)
        self._check(geometric.bond_core, analytic, (0, 1), GEOMETRY)

    def test_angle_second_derivative_matches_its_gradient(self):
        analytic = geometric.angle_hessian_core(GEOMETRY, 0, 1, 2)
        np.testing.assert_allclose(analytic,
                                   np.transpose(analytic, (0, 1, 3, 2)),
                                   atol=1.0e-14)
        self._check(geometric.angle_core, analytic, (0, 1, 2), GEOMETRY)

    def test_both_are_invariant_to_a_rigid_shift(self):
        """The curvature of a relative variable cannot see the frame."""
        moved = GEOMETRY + np.array([1.7, -2.3, 0.4])
        for name, atoms in (('bond_hessian_core', (0, 1)),
                            ('angle_hessian_core', (0, 1, 2))):
            core = getattr(geometric, name)
            np.testing.assert_allclose(core(GEOMETRY, *atoms),
                                       core(moved, *atoms), atol=1.0e-12)
            blocks = core(GEOMETRY, *atoms)
            np.testing.assert_allclose(blocks.sum(axis=1), 0.0, atol=1.0e-12)

    def test_a_degenerate_geometry_refuses_its_curvature(self):
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaises(geometric.CVSingularError):
            geometric.angle_hessian_core(line, 0, 1, 2)
        collapsed = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaises(geometric.CVSingularError):
            geometric.bond_hessian_core(collapsed, 0, 1)

    def test_torsion_second_derivative_matches_its_gradient(self):
        """The four-atom case, by the same rule as the other two.

        A torsion's curvature is the hardest of the three to write down, and the
        bilinear construction is chosen precisely so that symmetry and
        translation invariance come out by construction rather than by
        cancellation in a long formula.
        """
        analytic = geometric.torsion_hessian_core(GEOMETRY, 0, 1, 2, 3)
        np.testing.assert_allclose(analytic,
                                   np.transpose(analytic, (1, 0, 3, 2)),
                                   atol=1.0e-14)
        moved = GEOMETRY + np.array([1.7, -2.3, 0.4])
        np.testing.assert_allclose(analytic,
                                   geometric.torsion_hessian_core(
                                       moved, 0, 1, 2, 3), atol=1.0e-12)
        self._check(geometric.torsion_core, analytic, (0, 1, 2, 3), GEOMETRY)

    def test_a_collinear_torsion_refuses_its_curvature(self):
        straight = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        with self.assertRaises(geometric.CVSingularError):
            geometric.torsion_hessian_core(straight, 0, 1, 2, 3)

    def test_an_unknown_kind_is_reported_as_absent(self):
        """None, not zero: a caller must refuse rather than omit a term."""
        self.assertIsNone(geometric.cv_hessian_block('distance', GEOMETRY,
                                                    [0, 1]))
        for kind, atoms in (('bond', [0, 1]), ('angle', [0, 1, 2]),
                            ('dihedral', [0, 1, 2, 3])):
            block, indices = geometric.cv_hessian_block(kind, GEOMETRY, atoms)
            self.assertEqual(list(indices), atoms)
            self.assertEqual(block.shape,
                             (len(atoms), len(atoms), 3, 3))


if __name__ == '__main__':
    unittest.main()
