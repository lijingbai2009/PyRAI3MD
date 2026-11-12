"""
======================================================================
Tests for the strict bond constraint solver.

The scope is positions only: no velocity is projected, so nothing here asserts
J v = 0 and the module says why. What is pinned instead is the property the
position solve actually owns: the normalized residual falls below the declared
tolerance, a failure is reported rather than softened, mass decides who moves,
and a constraint that adds nothing is recognized as adding nothing.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.workflow.constraints import shake

MASSES = np.array([16.0, 1.0, 1.0, 12.0])
STRETCHED = np.array([[0.00, 0.00, 0.00],
                      [2.40, 0.30, 0.10],
                      [0.40, 1.70, -0.20],
                      [2.10, 1.80, 1.30]])
TARGETS = [1.80, 1.75]
PAIRS = [[0, 1], [0, 2]]
# A pair whose atoms are pulled apart far enough that the solver must work.
HARD_PAIRS = [[0, 1], [1, 3]]
HARD_TARGETS = [1.00, 2.00]


def _lengths(coords, pairs):
    return shake.distances(coords, pairs)


class TestDistanceHelpers(unittest.TestCase):

    def test_distances_match_the_norm(self):
        got = _lengths(STRETCHED, PAIRS)
        expected = [float(np.linalg.norm(STRETCHED[0] - STRETCHED[1])),
                    float(np.linalg.norm(STRETCHED[0] - STRETCHED[2]))]
        np.testing.assert_allclose(got, expected, atol=1.0e-14)

    def test_empty_pair_list_is_a_no_op(self):
        self.assertEqual(_lengths(STRETCHED, []).__len__(), 0)

    def test_jacobian_rows_match_a_finite_difference(self):
        jacobian = shake.build_bond_jacobian(STRETCHED, PAIRS, natom=4)
        step = 1.0e-6
        for row, pair in enumerate(PAIRS):
            for atom in pair:
                for axis in range(3):
                    forward = STRETCHED.copy()
                    forward[atom, axis] += step
                    back = STRETCHED.copy()
                    back[atom, axis] -= step
                    numeric = (_lengths(forward, [pair])[0]
                               - _lengths(back, [pair])[0]) / (2.0 * step)
                    self.assertLessEqual(
                        abs(numeric - jacobian[row, atom * 3 + axis]), 1.0e-7)

    def test_jacobian_sums_to_zero_per_row(self):
        jacobian = shake.build_bond_jacobian(STRETCHED, PAIRS, natom=4)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(2),
                                   atol=1.0e-14)


class TestResidual(unittest.TestCase):

    def test_satisfied_constraints_report_zero(self):
        coords = STRETCHED.copy()
        coords[1] = coords[0] + np.array([TARGETS[0], 0.0, 0.0])
        _, largest, worst, _ = shake.normalized_residual(coords, PAIRS,
                                                         [TARGETS[0],
                                                          np.linalg.norm(
                                                              coords[0]
                                                              - coords[2])])
        self.assertAlmostEqual(largest, 0.0, places=12)
        self.assertEqual(worst, 0)

    def test_relative_residual_scales_with_the_target(self):
        """The same absolute error is far worse on a shorter bond."""
        long_pair = [[0, 1]]
        short_pair = [[1, 2]]
        base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
        # The same 0.1 Bohr absolute error, on a 1.9 Bohr target against a
        # 0.2 Bohr target: relatively 0.0526 against 0.5, i.e. 9.5 times worse.
        _, first, _, _ = shake.normalized_residual(base, long_pair, [1.9])
        _, second, _, _ = shake.normalized_residual(base, short_pair, [0.2])
        self.assertAlmostEqual(second / first, 9.5, places=6)

    def test_broadcast_target_covers_every_pair(self):
        residual, _, _, _ = shake.normalized_residual(STRETCHED, PAIRS, 1.0)
        self.assertEqual(residual.size, 2)

    def test_a_lone_target_broadcasts_to_every_pair(self):
        residual, _, _, _ = shake.normalized_residual(STRETCHED, PAIRS, [1.0])
        self.assertEqual(residual.size, 2)

    def test_a_mismatched_target_count_is_refused(self):
        with self.assertRaises(shake.ShakeError):
            shake.normalized_residual(STRETCHED, PAIRS, [1.0, 2.0, 3.0])

    def test_a_tolerance_violation_can_be_raised_on_the_spot(self):
        with self.assertRaises(shake.ShakeError):
            shake.normalized_residual(STRETCHED, PAIRS, [0.5, 0.5],
                                      tolerance=1.0e-6)


class TestSolvePositions(unittest.TestCase):

    def test_converged_solution_satisfies_every_bond(self):
        solution = shake.solve_positions(STRETCHED, HARD_PAIRS, HARD_TARGETS,
                                         MASSES, tolerance=1.0e-10,
                                         max_iterations=500)
        self.assertTrue(solution['converged'])
        np.testing.assert_allclose(_lengths(solution['coordinates'],
                                            HARD_PAIRS),
                                   HARD_TARGETS, atol=1.0e-10)
        self.assertLessEqual(solution['max_residual'], 1.0e-10)

    def test_a_looser_tolerance_needs_fewer_passes(self):
        tight = shake.solve_positions(STRETCHED, HARD_PAIRS, HARD_TARGETS,
                                      MASSES, tolerance=1.0e-12,
                                      max_iterations=2000)
        loose = shake.solve_positions(STRETCHED, HARD_PAIRS, HARD_TARGETS,
                                      MASSES, tolerance=1.0e-4,
                                      max_iterations=2000)
        self.assertLess(loose['iterations'], tight['iterations'] + 1)

    def test_the_input_geometry_is_not_modified(self):
        before = STRETCHED.copy()
        shake.solve_positions(STRETCHED, HARD_PAIRS, HARD_TARGETS, MASSES,
                              tolerance=1.0e-10, max_iterations=200)
        np.testing.assert_allclose(STRETCHED, before, atol=0.0)

    def test_the_heavy_atom_moves_less(self):
        solution = shake.solve_positions(STRETCHED, [[0, 1]], [1.0], MASSES,
                                         tolerance=1.0e-12,
                                         max_iterations=500)
        shift = solution['shift']
        self.assertLess(shift[0], shift[1])
        self.assertAlmostEqual(shift[0] / shift[1], MASSES[1] / MASSES[0],
                               places=6)

    def test_an_already_satisfied_structure_is_untouched(self):
        coords = STRETCHED.copy()
        coords[1] = coords[0] + np.array([1.8, 0.0, 0.0])
        coords[2] = coords[0] + np.array([0.0, 1.75, 0.0])
        solution = shake.solve_positions(coords, PAIRS, TARGETS, MASSES,
                                         tolerance=1.0e-12,
                                         max_iterations=100)
        np.testing.assert_allclose(solution['coordinates'], coords,
                                   atol=1.0e-14)
        self.assertLessEqual(solution['max_residual'], 1.0e-12)

    def test_total_displacement_is_minimal_for_one_bond(self):
        """With a single constraint the correction is along the bond, so the
        center of mass must not move."""
        before = (MASSES[:, None] * STRETCHED).sum(axis=0) / MASSES.sum()
        solution = shake.solve_positions(STRETCHED, [[0, 1]], [1.0], MASSES,
                                         tolerance=1.0e-12,
                                         max_iterations=500)
        after = ((MASSES[:, None] * solution['coordinates']).sum(0)
                 / MASSES.sum())
        np.testing.assert_allclose(after, before, atol=1.0e-12)

    def test_a_coincident_pair_reports_the_degenerate_geometry(self):
        """No bond direction exists, so no correction can be aimed."""
        collapsed = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                              [3.0, 1.0, 1.0], [4.0, 1.0, 1.0]])
        with self.assertRaises(shake.ShakeError) as caught:
            shake.solve_positions(collapsed, [[0, 1]], [2.0], MASSES,
                                  tolerance=1.0e-10, max_iterations=20)
        self.assertIn('collapsed', str(caught.exception))

    def test_non_convergence_is_reported_not_softened(self):
        """Two atoms pinned apart cannot be brought together by this solver."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[0] = True
        mask[1] = True
        with self.assertRaises(shake.ShakeNotConverged) as caught:
            shake.solve_positions(STRETCHED, [[0, 1]], [0.5], MASSES,
                                  tolerance=1.0e-14, max_iterations=3,
                                  fix_mask=mask, anchors=STRETCHED)
        self.assertGreater(caught.exception.residual, 0.0)
        self.assertGreater(caught.exception.physical_residual, 0.0)
        self.assertEqual(caught.exception.constraint, (0, 1))
        self.assertEqual(caught.exception.iterations, 3)

    def test_a_degenerate_pair_that_cannot_be_resolved_fails(self):
        """Two coinciding atoms have no bond direction to push along."""
        same = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1., 1., 1.],
                         [2., 2., 2.]])
        with self.assertRaises(shake.ShakeError):
            shake.solve_positions(same, [[0, 1]], [1.5], MASSES,
                                  tolerance=1.0e-10, max_iterations=50)

    def test_a_non_positive_target_is_refused(self):
        with self.assertRaises(shake.ShakeError):
            shake.solve_positions(STRETCHED, [[0, 1]], [0.0], MASSES)

    def test_a_zero_target_that_is_satisfiable_is_allowed(self):
        """Two atoms asked to coincide is a legal, if odd, declaration."""
        solution = shake.solve_positions(STRETCHED, [[0, 1]], [1.0e-3],
                                         MASSES, tolerance=1.0e-6,
                                         max_iterations=500)
        self.assertTrue(solution['converged'])

    def test_a_tolerance_below_double_precision_is_refused(self):
        with self.assertRaises(shake.ShakeError):
            shake.solve_positions(STRETCHED, [[0, 1]], [1.0], MASSES,
                                  tolerance=1.0e-20)

    def test_mass_array_must_match_the_atom_count(self):
        with self.assertRaises(shake.ShakeError):
            shake.solve_positions(STRETCHED, [[0, 1]], [1.0], np.array([16.0,
                                                                        1.0]))

    def test_no_constraints_returns_the_input(self):
        solution = shake.solve_positions(STRETCHED, [], [], MASSES)
        self.assertTrue(solution['converged'])
        np.testing.assert_allclose(solution['coordinates'], STRETCHED)

    def test_a_zero_iteration_budget_is_refused(self):
        with self.assertRaises(shake.ShakeError):
            shake.solve_positions(STRETCHED, [[0, 1]], [1.0], MASSES,
                                  max_iterations=0)


class TestFixInteraction(unittest.TestCase):

    def test_a_frozen_atom_stays_put_while_its_partner_is_corrected(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0] = True
        anchors = STRETCHED.copy()
        solution = shake.solve_positions(STRETCHED, [[0, 1]], [1.0], MASSES,
                                         tolerance=1.0e-12,
                                         max_iterations=500, fix_mask=mask,
                                         anchors=anchors)
        np.testing.assert_allclose(solution['coordinates'][0], STRETCHED[0],
                                   atol=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(
            solution['coordinates'][0] - solution['coordinates'][1])), 1.0,
            places=10)

    def test_two_frozen_atoms_at_the_wrong_distance_cannot_converge(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0] = True
        mask[1] = True
        with self.assertRaises(shake.ShakeNotConverged):
            shake.solve_positions(STRETCHED, [[0, 1]], [0.5], MASSES,
                                  tolerance=1.0e-12, max_iterations=20,
                                  fix_mask=mask, anchors=STRETCHED)


class TestRedundancy(unittest.TestCase):

    def setUp(self):
        self.mask = np.zeros((4, 3), dtype=bool)
        self.mask[0] = True
        self.mask[1] = True
        self.anchor_length = float(np.linalg.norm(STRETCHED[0] - STRETCHED[1]))

    def test_a_frozen_pair_at_its_own_length_is_redundant(self):
        got = shake.detect_redundancy([[0, 1]], [self.anchor_length],
                                      STRETCHED, fix_mask=self.mask)
        self.assertEqual(len(got['redundant']), 1)
        self.assertEqual(len(got['effective']), 0)
        self.assertEqual(len(got['conflict']), 0)

    def test_a_frozen_pair_at_another_length_is_a_conflict(self):
        got = shake.detect_redundancy([[0, 1]], [self.anchor_length + 0.2],
                                      STRETCHED, fix_mask=self.mask)
        self.assertEqual(len(got['conflict']), 1)
        self.assertEqual(len(got['effective']), 0)

    def test_a_mobile_pair_is_effective(self):
        got = shake.detect_redundancy([[0, 1]], [1.5], STRETCHED,
                                      fix_mask=self.mask)
        self.assertEqual(len(got['effective']), 0)
        self.assertEqual(len(got['conflict']), 1)
        free = np.zeros((4, 3), dtype=bool)
        got = shake.detect_redundancy([[0, 1]], [1.5], STRETCHED,
                                      fix_mask=free)
        self.assertEqual(len(got['effective']), 1)

    def test_without_a_mask_nothing_is_redundant_by_assumption(self):
        got = shake.detect_redundancy([[0, 1]], [1.0], STRETCHED)
        self.assertEqual(len(got['effective']), 1)


class TestMultiConditionSolve(unittest.TestCase):
    """A4: the same sweep over angles and torsions, with a rank test."""

    ANGLE = np.radians(104.5)
    CONDITIONS = [('bond', [0, 1], 1.80),
                  ('bond', [0, 2], 1.75),
                  ('angle', [1, 0, 2], ANGLE)]

    def test_bond_only_conditions_reproduce_the_pair_solver(self):
        """The new sweep on a bond declaration is the old method to round-off.

        Both solvers share the mass-weighted row correction. The pair solver
        divides the bond vector by its length inside the step scale, the row
        solver carries a unit row and divides by the Gram scalar, so the two
        agree to within a unit in the last place rather than to the bit. The
        bit-level promise (DC-1) lives on the call path a bond-only plan takes,
        which still goes to `solve_positions` and is pinned elsewhere.
        """
        bonds = [('bond', [0, 1], 1.0), ('bond', [1, 3], 2.0)]
        pair_solve = shake.solve_positions(STRETCHED, HARD_PAIRS, HARD_TARGETS,
                                           MASSES, tolerance=1.0e-12)
        sweep = shake.solve_conditions(STRETCHED, bonds, MASSES,
                                       tolerance=1.0e-12)
        np.testing.assert_allclose(pair_solve['coordinates'],
                                   sweep['coordinates'], atol=1.0e-14)
        self.assertEqual(pair_solve['iterations'], sweep['iterations'])

    def test_angle_and_bonds_converge_together(self):
        solution = shake.solve_conditions(STRETCHED, self.CONDITIONS, MASSES,
                                          tolerance=1.0e-12,
                                          max_iterations=500)
        self.assertTrue(solution['converged'])
        self.assertLessEqual(solution['max_residual'], 1.0e-12)
        coords = solution['coordinates']
        self.assertAlmostEqual(shake.condition_value(coords, 'bond', [0, 1]),
                               1.80, places=11)
        self.assertAlmostEqual(shake.condition_value(coords, 'bond', [0, 2]),
                               1.75, places=11)
        self.assertAlmostEqual(shake.condition_value(coords, 'angle',
                                                     [1, 0, 2]),
                               self.ANGLE, places=11)
        self.assertEqual(solution['redundant']['rank'], 3)

    def test_the_sweep_invents_no_momentum(self):
        """Sum m_i dR_i = 0: the angle rows cancel, so the shift has no net push.

        This is the property the pair solver has by construction, and a
        three-atom row could have broken it silently.
        """
        before = STRETCHED.copy()
        solution = shake.solve_conditions(before, self.CONDITIONS, MASSES,
                                          tolerance=1.0e-12,
                                          max_iterations=500)
        shift = solution['coordinates'] - before
        total = np.sum(MASSES[:, None] * shift, axis=0)
        np.testing.assert_allclose(total, np.zeros(3), atol=1.0e-11)

    def test_a_dependent_condition_costs_no_degree_of_freedom(self):
        """Three angles of a planar ring: two conditions, not three."""
        ring = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]])
        masses = np.array([12.0, 16.0, 1.0])
        planar = np.pi / 3.0
        conditions = [('angle', [0, 1, 2], planar),
                      ('angle', [1, 2, 0], planar),
                      ('angle', [2, 0, 1], planar)]
        matrix = shake.build_condition_jacobian(
            ring, shake.normalize_conditions(conditions), natom=3)
        split = shake.condition_redundancy(matrix, metric=np.repeat(1 / masses,
                                                                    3))
        self.assertEqual(split['rank'], 2)
        self.assertEqual(split['redundant'], [2])
        solution = shake.solve_conditions(ring, conditions, masses,
                                          tolerance=1.0e-10,
                                          max_iterations=400)
        self.assertTrue(solution['converged'])
        self.assertEqual(solution['redundant']['rank'], 2)

    def test_a_collinear_reference_fails_by_name(self):
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaises(shake.ShakeError) as caught:
            shake.solve_conditions(line, [('angle', [0, 1, 2], self.ANGLE)],
                                   MASSES[:3])
        message = str(caught.exception)
        self.assertIn('angle', message)
        self.assertIn('cannot be differentiated', message)
        self.assertIn('collinear', message)

    def test_a_frozen_atom_stays_at_its_anchor(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0] = True
        anchors = STRETCHED.copy()
        solution = shake.solve_conditions(STRETCHED, self.CONDITIONS, MASSES,
                                          tolerance=1.0e-10,
                                          max_iterations=400, fix_mask=mask,
                                          anchors=anchors)
        np.testing.assert_array_equal(solution['coordinates'][0], anchors[0])
        self.assertTrue(solution['converged'])

    def test_a_dihedral_condition_converges_with_its_chain(self):
        """The four-atom case: a torsion held together with its arms."""
        chain = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                          [2.25, 1.3, 0.0], [3.75, 1.3, 0.0]])
        masses = np.array([12.0, 12.0, 1.008, 1.008])
        conditions = [('bond', [1, 2], 1.54),
                      ('angle', [0, 1, 2], np.radians(111.0)),
                      ('angle', [1, 2, 3], np.radians(111.0)),
                      ('dihedral', [0, 1, 2, 3], np.radians(60.0))]
        solution = shake.solve_conditions(chain, conditions, masses,
                                          tolerance=1.0e-12,
                                          max_iterations=800)
        coords = solution['coordinates']
        self.assertTrue(solution['converged'])
        self.assertLessEqual(solution['max_residual'], 1.0e-12)
        self.assertEqual(solution['redundant']['rank'], 4)
        self.assertAlmostEqual(
            shake.condition_value(coords, 'dihedral', [0, 1, 2, 3]),
            np.radians(60.0), places=11)
        for kind, atoms, target in conditions:
            self.assertAlmostEqual(
                shake.condition_value(coords, kind, atoms), target, places=11)

    def test_an_eclipsed_torsion_reference_fails_by_name(self):
        """A torsion whose two rotation planes coincide has no derivative."""
        flat = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                         [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
        with self.assertRaises(shake.ShakeError) as caught:
            shake.solve_conditions(flat, [('dihedral', [0, 1, 2, 3], 1.0)],
                                   np.array([12.0, 12.0, 1.0, 1.0]))
        message = str(caught.exception)
        self.assertIn('dihedral', message)
        self.assertIn('collinear', message)

    def test_declarations_are_checked_before_the_sweep(self):
        masses = np.array([16.0, 1.0, 1.0, 12.0])
        with self.assertRaises(shake.ShakeError):
            shake.solve_conditions(STRETCHED, [('torsion', [0, 1, 2, 3], 1.0)],
                                   masses)
        with self.assertRaises(shake.ShakeError):
            shake.solve_conditions(STRETCHED, [('angle', [0, 1], 1.0)], masses)
        with self.assertRaises(shake.ShakeError):
            shake.solve_conditions(STRETCHED, [('bond', [0, 9], 1.0)], masses)
        with self.assertRaises(shake.ShakeError):
            shake.solve_conditions(STRETCHED, [('bond', [0, 1], 0.0)], masses)
        with self.assertRaises(shake.ShakeError):
            shake.solve_conditions(STRETCHED, self.CONDITIONS,
                                   np.array([16.0, 1.0]))


class TestMultiConditionVelocity(unittest.TestCase):
    """The velocity half of an angle or torsion: the same rows, projected."""

    ANGLE = np.radians(104.5)
    CONDITIONS = [('bond', [0, 1], 1.80),
                  ('bond', [0, 2], 1.75),
                  ('angle', [1, 0, 2], ANGLE)]
    VELOCITIES = np.array([[0.001, 0.002, -0.0005],
                           [-0.003, 0.001, 0.002],
                           [0.002, -0.004, 0.001],
                           [0.0, 0.0, 0.0]])

    def test_every_condition_rate_goes_to_zero(self):
        before = shake.condition_rates(STRETCHED, self.CONDITIONS,
                                       self.VELOCITIES)
        self.assertGreater(float(np.abs(before).max()), 1.0e-4)
        solution = shake.solve_velocity_conditions(
            STRETCHED, self.CONDITIONS, self.VELOCITIES, MASSES,
            tolerance=1.0e-12, max_iterations=500)
        self.assertTrue(solution['converged'])
        self.assertLessEqual(solution['max_residual'], 1.0e-12)
        self.assertEqual(solution['redundant']['rank'], 3)

    def test_the_projection_invents_no_momentum(self):
        solution = shake.solve_velocity_conditions(
            STRETCHED, self.CONDITIONS, self.VELOCITIES, MASSES,
            tolerance=1.0e-12, max_iterations=500)
        change = solution['velocities'] - self.VELOCITIES
        total = np.sum(MASSES[:, None] * change, axis=0)
        np.testing.assert_allclose(total, np.zeros(3), atol=1.0e-12)

    def test_bond_only_projection_reproduces_the_pair_solver(self):
        bonds = [('bond', [0, 1], 1.0), ('bond', [1, 3], 2.0)]
        pairs = np.array([[0, 1], [1, 3]], dtype=np.int64)
        old = shake.solve_velocities(STRETCHED, pairs, self.VELOCITIES, MASSES,
                                     tolerance=1.0e-10, max_iterations=400)
        new = shake.solve_velocity_conditions(
            STRETCHED, bonds, self.VELOCITIES, MASSES, tolerance=1.0e-10,
            max_iterations=400)
        np.testing.assert_allclose(old['velocities'], new['velocities'],
                                   atol=1.0e-14)
        self.assertEqual(old['iterations'], new['iterations'])

    def test_the_input_velocity_array_is_not_modified(self):
        original = self.VELOCITIES.copy()
        shake.solve_velocity_conditions(STRETCHED, self.CONDITIONS,
                                        self.VELOCITIES, MASSES,
                                        tolerance=1.0e-10)
        np.testing.assert_array_equal(self.VELOCITIES, original)

    def test_a_dependent_condition_is_dropped_not_swept(self):
        ring = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                         [0.75, 1.3, 0.0]])
        masses = np.array([12.0, 16.0, 1.0])
        spins = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.02],
                          [0.0, 0.0, 0.005]])
        planar = np.pi / 3.0
        conditions = [('angle', [0, 1, 2], planar),
                      ('angle', [1, 2, 0], planar),
                      ('angle', [2, 0, 1], planar)]
        solution = shake.solve_velocity_conditions(
            ring, conditions, spins, masses, tolerance=1.0e-10,
            max_iterations=200)
        self.assertEqual(solution['redundant']['rank'], 2)
        self.assertEqual(solution['redundant']['redundant'], [2])

    def test_a_collinear_geometry_has_no_tangent_to_project(self):
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaises(shake.ShakeError) as caught:
            shake.solve_velocity_conditions(
                line, [('angle', [0, 1, 2], self.ANGLE)],
                np.zeros((3, 3)), MASSES[:3])
        self.assertIn('cannot be differentiated', str(caught.exception))

    def test_frozen_components_keep_zero_velocity(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[3] = True
        solution = shake.solve_velocity_conditions(
            STRETCHED, self.CONDITIONS, self.VELOCITIES, MASSES,
            tolerance=1.0e-10, max_iterations=400, fix_mask=mask)
        np.testing.assert_array_equal(solution['velocities'][3], np.zeros(3))



if __name__ == '__main__':
    unittest.main()
