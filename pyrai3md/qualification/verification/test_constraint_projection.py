"""
======================================================================
Tests for frozen components, condition rank and free degrees of freedom.

Two things are pinned here that the framework currently gets wrong in
opposite directions. `ensemble._dof` subtracts three per frozen atom while
nothing keeps those atoms still, and it clamps a physical zero to one. The
rank-based count must reproduce the old number in the case the old number was
right about, and must stop where the old number lied.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import textwrap

import numpy as np

from pyrai3md.data import constraint as records
from pyrai3md.workflow.constraints import projection

MASSES = np.array([16.0, 1.0, 1.0])
WATER = np.array([[0.00, 0.00, 0.00],
                  [1.80, 0.20, 0.10],
                  [0.30, 1.75, -0.15]])
WATER_PAIRS = [[0, 1], [0, 2], [1, 2]]


class TestFixMask(unittest.TestCase):

    def setUp(self):
        self.plan = records.ConstraintPlan(
            actions=[records.ConstraintSpec('f1', 'fix', 'mask', atoms=[0, 1],
                                            parameters={'axes': [0, 2]}),
                     # One component already frozen by f1, one new: overlap
                     # merges per component instead of conflicting.
                     records.ConstraintSpec('f2', 'fix', 'mask', atoms=[1],
                                            parameters={'axes': [1, 2]})])

    def test_axes_land_in_the_mask(self):
        mask = projection.build_fix_mask(self.plan, 3)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[0, 2])
        self.assertTrue(mask[1, 0] and mask[1, 2])

    def test_overlapping_fixes_merge_without_conflict(self):
        """f1 freezes 0x,0z,1x,1z and f2 adds 1y, so five components."""
        mask = projection.build_fix_mask(self.plan, 3)
        self.assertTrue(mask[1, 1])
        self.assertEqual(int(mask.sum()), 5)

    def test_an_empty_plan_freezes_nothing(self):
        mask = projection.build_fix_mask(records.ConstraintPlan.empty(), 3)
        self.assertEqual(int(mask.sum()), 0)

    def test_a_group_fix_freezes_the_group_only(self):
        from pyrai3md.data.topology import GroupTable
        plan = records.ConstraintPlan(
            groups=GroupTable({'g': [2]}),
            actions=[records.ConstraintSpec('f', 'fix', 'mask', group='g',
                                            parameters={'axes': [0, 1, 2]})])
        mask = projection.build_fix_mask(plan, 3)
        self.assertEqual(np.flatnonzero(mask.any(axis=1)).tolist(), [2])


class TestFixRestore(unittest.TestCase):

    def test_anchors_hold_and_drift_is_counted(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0] = True
        anchors = projection.fix_anchors(WATER, mask)
        drifted = WATER.copy()
        drifted[0] += 0.5
        restored, moved = projection.apply_fix_positions(drifted, anchors,
                                                         mask)
        np.testing.assert_allclose(restored[0], WATER[0], atol=0.0)
        self.assertEqual(moved, 3)

    def test_restoring_a_still_structure_moves_nothing(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 1] = True
        anchors = projection.fix_anchors(WATER, mask)
        restored, moved = projection.apply_fix_positions(WATER, anchors, mask)
        self.assertEqual(moved, 0)
        np.testing.assert_allclose(restored, WATER)

    def test_mobile_atoms_are_left_alone(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0] = True
        anchors = projection.fix_anchors(WATER, mask)
        moved = WATER.copy()
        moved[1] += 0.7
        restored, _ = projection.apply_fix_positions(moved, anchors, mask)
        np.testing.assert_allclose(restored[1], moved[1])

    def test_the_input_array_is_not_modified(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0] = True
        anchors = projection.fix_anchors(WATER, mask)
        drifted = WATER.copy()
        drifted[0] = 5.0
        projection.apply_fix_positions(drifted, anchors, mask)
        self.assertAlmostEqual(float(drifted[0, 0]), 5.0, places=12)

    def test_only_frozen_velocities_are_zeroed(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 0] = True
        velocities = np.ones((3, 3))
        zeroed = projection.zero_fixed_velocities(velocities, mask)
        self.assertEqual(zeroed[1, 0], 0.0)
        self.assertEqual(float(np.abs(np.delete(zeroed.ravel(), 3)).max()),
                         1.0)
        np.testing.assert_allclose(velocities, np.ones((3, 3)))


class TestDegreesOfFreedom(unittest.TestCase):
    """The counting rules of the design document, section 10."""

    def test_no_conditions_gives_three_per_atom(self):
        self.assertEqual(projection.degrees_of_freedom(MASSES).free_dof, 9)

    def test_a_rigid_water_loses_three(self):
        got = projection.degrees_of_freedom(MASSES, pairs=WATER_PAIRS,
                                            coords=WATER)
        self.assertEqual((got.rank, got.free_dof), (3, 6))

    def test_a_removed_center_of_mass_takes_three_more(self):
        got = projection.degrees_of_freedom(MASSES, pairs=WATER_PAIRS,
                                            coords=WATER, remove_com=True)
        self.assertEqual(got.free_dof, 3)

    def test_a_duplicate_constraint_is_not_counted_twice(self):
        got = projection.degrees_of_freedom(
            MASSES, pairs=[[0, 1], [0, 1], [0, 2]], coords=WATER)
        self.assertEqual((got.rank, got.free_dof), (2, 7))

    def test_a_constraint_between_two_frozen_atoms_adds_nothing(self):
        mask = np.ones((3, 3), dtype=bool)
        got = projection.degrees_of_freedom(MASSES, mask=mask,
                                            pairs=[[0, 1]], coords=WATER)
        self.assertEqual((got.mobile_components, got.rank, got.free_dof),
                         (0, 0, 0))

    def test_fully_frozen_is_zero_and_not_one(self):
        mask = np.ones((3, 3), dtype=bool)
        got = projection.degrees_of_freedom(MASSES, mask=mask,
                                            remove_com=True)
        self.assertEqual(got.free_dof, 0)
        self.assertIsNone(got.temperature(1.0e-3))

    def test_the_freeze_only_case_reproduces_the_framework_formula(self):
        """3 * (natom - n_frozen) - 3, which is what ensemble._dof returns."""
        for frozen in range(0, 3):
            mask = np.zeros((3, 3), dtype=bool)
            mask[:frozen] = True
            got = projection.degrees_of_freedom(MASSES, mask=mask,
                                                remove_com=True)
            self.assertEqual(got.free_dof, 3 * (3 - frozen) - 3)

    def test_without_com_removal_the_count_is_three_higher(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0] = True
        with_com = projection.degrees_of_freedom(MASSES, mask=mask,
                                                 remove_com=True)
        without = projection.degrees_of_freedom(MASSES, mask=mask,
                                                remove_com=False)
        self.assertEqual(without.free_dof, with_com.free_dof + 3)

    def test_a_non_positive_mass_is_refused(self):
        with self.assertRaises(ValueError):
            projection.degrees_of_freedom(np.array([16.0, 0.0, 1.0]))


class TestKineticEnergy(unittest.TestCase):

    def test_kinetic_energy_uses_only_mobile_components(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0] = True
        velocities = np.full((3, 3), 0.1)
        total = 0.5 * float(np.sum(MASSES[:, None] * velocities ** 2))
        expected = 0.5 * float(np.sum(MASSES[[1, 2], None]
                                      * velocities[[1, 2]] ** 2))
        free, _ = projection.free_kinetic_energy(velocities, MASSES, mask=mask)
        self.assertLess(free, total)
        self.assertAlmostEqual(free, expected, places=12)

    @staticmethod
    def _bond_frame():
        """Unit vectors along and perpendicular to the 0-1 bond of WATER."""
        vector = WATER[0] - WATER[1]
        along = vector / np.linalg.norm(vector)
        guess = np.array([0.0, 0.0, 1.0])
        across = guess - np.dot(guess, along) * along
        across /= np.linalg.norm(across)
        return along, across

    def test_a_bond_stretch_velocity_is_not_free(self):
        along, _ = self._bond_frame()
        stretch = np.zeros((3, 3))
        stretch[0] = -along
        stretch[1] = along
        plain, _ = projection.free_kinetic_energy(stretch, MASSES)
        constrained, _ = projection.free_kinetic_energy(
            stretch, MASSES, pairs=[[0, 1]], coords=WATER)
        self.assertGreater(plain, 0.0)
        self.assertAlmostEqual(constrained, 0.0, places=12)

    def test_a_perpendicular_velocity_survives_the_projection(self):
        _, across = self._bond_frame()
        twist = np.zeros((3, 3))
        twist[0] = -across
        twist[1] = across
        constrained, _ = projection.free_kinetic_energy(
            twist, MASSES, pairs=[[0, 1]], coords=WATER)
        plain, _ = projection.free_kinetic_energy(twist, MASSES)
        self.assertGreater(plain, 0.0)
        self.assertAlmostEqual(constrained, plain, places=12)

    def test_a_constraint_on_a_third_pair_still_kills_its_stretch(self):
        vector = WATER[1] - WATER[2]
        unit = vector / np.linalg.norm(vector)
        stretch = np.zeros((3, 3))
        stretch[1] = -unit
        stretch[2] = unit
        constrained, _ = projection.free_kinetic_energy(
            stretch, MASSES, pairs=[[1, 2]], coords=WATER)
        self.assertAlmostEqual(constrained, 0.0, places=12)

    def test_statistics_reports_both_numbers(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2] = True
        statistics = projection.statistics(MASSES, velocities=np.ones((3, 3)),
                                           mask=mask)
        self.assertEqual(statistics.mobile_components, 6)
        self.assertEqual(statistics.free_dof, 6)
        self.assertGreater(statistics.free_kinetic, 0.0)

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            projection.free_kinetic_energy(np.ones((2, 3)), MASSES)


class TestConditionMatrix(unittest.TestCase):

    def test_bond_row_norm_is_the_inverse_mass_sum(self):
        """A mass-weighted bond row has norm sqrt(1/m_i + 1/m_j).

        The unit direction contributes its full squared length at each end, so
        what is left over is the inverse-mass factor. A row of norm one would
        mean the same correction was applied to both atoms regardless of mass.
        """
        matrix, mobile = projection.condition_matrix(
            MASSES, pairs=[[0, 1]], coords=WATER)
        self.assertEqual(matrix.shape[0], 1)
        expected = float(np.sqrt(1.0 / MASSES[0] + 1.0 / MASSES[1]))
        self.assertAlmostEqual(float(np.linalg.norm(matrix[0])), expected,
                               places=12)

    def test_a_degenerate_geometry_contributes_no_row(self):
        collapsed = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0]])
        matrix, _ = projection.condition_matrix(MASSES, pairs=[[0, 1]],
                                                coords=collapsed)
        self.assertEqual(matrix.shape[0], 0)

    def test_com_rows_span_the_translation(self):
        matrix, mobile = projection.condition_matrix(MASSES, remove_com=True)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(projection.condition_rank(matrix), 3)
        self.assertEqual(mobile.shape[0], 9)

    def test_a_pair_without_geometry_is_refused(self):
        with self.assertRaises(ValueError):
            projection.condition_matrix(MASSES, pairs=[[0, 1]])

    def test_a_mismatched_mask_is_refused(self):
        with self.assertRaises(ValueError):
            projection.condition_matrix(MASSES, mask=np.ones((2, 3), bool))


class TestTemperature(unittest.TestCase):

    def test_equipment_for_the_reported_temperature(self):
        statistics = projection.statistics(MASSES, velocities=np.ones((3, 3)),
                                           pairs=WATER_PAIRS, coords=WATER)
        self.assertEqual(statistics.free_dof, 6)
        temperature = statistics.temperature(1.0e-3)
        self.assertAlmostEqual(temperature,
                               2.0 * statistics.free_kinetic / 6.0e-3,
                               places=9)


class TestOneCenterHelper(unittest.TestCase):
    """D-04 closed: the walls and the COM removal read one implementation."""

    def test_the_shared_center_reproduces_the_inline_com_bitwise(self):
        """The integrator must not move by one unit in the last place."""
        from pyrai3md.utilities import geometric
        rng = np.random.default_rng(20260930)
        masses = np.array([15.999, 1.008, 1.008, 12.011, 1.008]) * 1822.888
        velocities = rng.normal(0.0, 1.0e-3, size=(5, 3))
        inline = np.sum(masses[:, None] * velocities, axis=0) / float(
            np.sum(masses))
        shared, _members, weights = geometric.group_center(
            velocities, np.arange(5), masses=masses)
        np.testing.assert_array_equal(inline, shared)
        np.testing.assert_allclose(weights, masses / float(np.sum(masses)),
                                   atol=0.0, rtol=1.0e-15)

    def test_operations_reads_the_helper_not_its_own_copy(self):
        import ast
        import inspect
        import os
        from pyrai3md.workflow.integrator import operations
        source = inspect.getsource(operations.velocity_correction)
        tree = ast.parse(textwrap.dedent(source))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func,
                                                         ast.Attribute):
                names.add(node.func.attr)
        self.assertIn('group_center', names)
        # The duplicate this row removes is the weighted sum itself, not the
        # guard that keeps a zero total mass out of the kinetic rescale.
        self.assertNotIn('np.sum(mass_au[:, None] * velocities, axis=0)',
                         source)


class TestTangentSplit(unittest.TestCase):
    """The gradient-side split: what a condition absorbs, and what is left.

    The reference numbers come from the Jacobian the same `utilities.cv` core
    returns, so an instrument that asks for a gradient along a condition normal
    is asking for the analytic vector, not for whatever the split happened to
    produce.
    """

    def setUp(self):
        from pyrai3md.utilities import cv
        self.coords = WATER
        _value, rows = cv.distance(self.coords, 0, 1, grad=True)
        self.bond_row = np.zeros(9)
        self.bond_row[0:3] = rows[0]
        self.bond_row[3:6] = rows[1]
        _value, arows = cv.angle(self.coords, 0, 1, 2, grad=True)
        self.angle_row = np.zeros(9)
        for position, row in enumerate(arows):
            self.angle_row[3 * position:3 * position + 3] = row

    def _split(self, gradient, **kwargs):
        return projection.tangent_split(gradient, self.coords, **kwargs)

    def test_a_gradient_along_one_normal_leaves_nothing_to_judge(self):
        wanted = 0.02 * self.bond_row
        tangent, multipliers, rank = self._split(
            wanted, conditions=[('bond', [0, 1])])
        self.assertLess(float(np.max(np.abs(tangent))), 1.0e-14)
        self.assertAlmostEqual(float(multipliers[0]), 0.02, places=12)
        self.assertEqual(rank, 1)

    def test_a_gradient_across_the_condition_is_kept_whole(self):
        # Perpendicular to the bond's own direction, taken on the pair: the
        # direction the condition holds nothing against.
        # The row's first block is the gradient of the length with
        # respect to the first atom, which is the bond's own direction.
        direction = self.bond_row[0:3]
        unit = direction / float(np.linalg.norm(direction))
        arm = np.array([0.0, 1.0, 0.0])
        arm = arm - float(arm @ unit) * unit
        arm = arm / float(np.linalg.norm(arm))
        vector = np.zeros(9)
        vector[0:3] = 0.03 * arm
        vector[3:6] = -0.03 * arm
        tangent, multipliers, _rank = self._split(
            vector, conditions=[('bond', [0, 1])])
        self.assertAlmostEqual(float(multipliers[0]), 0.0, places=14)
        np.testing.assert_allclose(tangent, vector, atol=1.0e-14)

    def test_two_conditions_recover_their_own_multipliers(self):
        wanted = 0.02 * self.bond_row + 0.015 * self.angle_row
        tangent, multipliers, rank = self._split(
            wanted, conditions=[('bond', [0, 1]), ('angle', [0, 1, 2])])
        self.assertEqual(rank, 2)
        np.testing.assert_allclose(multipliers, [0.02, 0.015], atol=1.0e-12)
        self.assertLess(float(np.max(np.abs(tangent))), 1.0e-14)

    def test_a_repeated_condition_is_one_condition(self):
        wanted = 0.02 * self.bond_row
        tangent, multipliers, rank = self._split(
            wanted, conditions=[('bond', [0, 1]), ('bond', [0, 1])])
        self.assertEqual(rank, 1, 'a duplicate row adds no degree of restraint')
        self.assertAlmostEqual(float(np.sum(multipliers)), 0.02, places=12)
        self.assertLess(float(np.max(np.abs(tangent))), 1.0e-14)

    def test_frozen_components_come_back_without_a_share(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2] = True
        wanted = 0.02 * self.bond_row + np.ones(9)
        tangent, _multipliers, _rank = self._split(
            wanted, mask=mask, conditions=[('bond', [0, 1])])
        self.assertTrue(np.all(tangent.reshape(3, 3)[2] == 0.0))

    def test_a_plan_with_no_condition_judges_the_whole_gradient(self):
        wanted = 0.02 * self.bond_row
        tangent, multipliers, rank = self._split(wanted)
        np.testing.assert_allclose(tangent, wanted, atol=0.0)
        self.assertEqual(multipliers.size, 0)
        self.assertEqual(rank, 0)

    def test_the_bond_only_spelling_matches_the_condition_spelling(self):
        wanted = 0.02 * self.bond_row
        by_pairs = self._split(wanted, pairs=[[0, 1]])
        by_conditions = self._split(wanted, conditions=[('bond', [0, 1])])
        np.testing.assert_allclose(by_pairs[0], by_conditions[0], atol=0.0)
        np.testing.assert_allclose(by_pairs[1], by_conditions[1], atol=1.0e-15)


if __name__ == '__main__':
    unittest.main()
