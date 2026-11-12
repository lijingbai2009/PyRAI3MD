"""
======================================================================
Tests for the constraint records: round trip, emptiness and isolation.

A plan is a value, so it must survive a dict round trip, a pickle round trip
and a fingerprint comparison. A state is per trajectory, so a copy must not
share a hill list or a token set. A result of an empty plan must not allocate
a coordinate array, because that array would be written into every history
frame and every dispatch payload of every unconstrained run in the project.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import pickle
import unittest

import numpy as np

from pyrai3md.data import constraint as records
from pyrai3md.data import topology as topo


def _plan():
    """A plan with one of each action kind."""
    return records.ConstraintPlan(
        groups=topo.GroupTable({'base': [0, 1]}),
        cvs={'r': records.CVRecord('r', 'bond', [0, 1])},
        actions=[
            records.ConstraintSpec('b', 'bias', 'harmonic', cv='r',
                                   target=2.0, region='upper',
                                   parameters={'force_constant': 0.006}),
            records.ConstraintSpec('f', 'fix', 'mask', group='base',
                                   parameters={'axes': [0, 1, 2]},
                                   reference='initial'),
            records.ConstraintSpec('x', 'constraint', 'shake',
                                   pairs=[[0, 1]], reference='initial',
                                   diagnostics={'tolerance': 1.0e-8,
                                                'max_iterations': 40}),
        ],
        units={'length': 'bohr'})


class TestPlan(unittest.TestCase):

    def test_empty_plan_is_the_sentinel(self):
        plan = records.ConstraintPlan.empty()
        self.assertTrue(plan.is_empty)
        self.assertEqual(plan.ids(), [])

    def test_a_declared_plan_is_not_empty(self):
        self.assertFalse(_plan().is_empty)

    def test_dict_round_trip_keeps_the_fingerprint(self):
        plan = _plan()
        restored = records.ConstraintPlan.from_dict(plan.to_dict())
        self.assertEqual(restored.fingerprint, plan.fingerprint)
        self.assertEqual(restored, plan)

    def test_pickle_round_trip(self):
        plan = _plan()
        self.assertEqual(pickle.loads(pickle.dumps(plan)), plan)

    def test_a_parameter_change_moves_the_fingerprint(self):
        first = _plan()
        second = _plan()
        actions = list(second.actions)
        actions[0] = records.ConstraintSpec(
            'b', 'bias', 'harmonic', cv='r', target=2.0, region='upper',
            parameters={'force_constant': 0.007})
        second = records.ConstraintPlan(groups=second.groups,
                                        cvs=second.cvs, actions=actions,
                                        units=second.units)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_group_order_is_part_of_the_fingerprint(self):
        first = records.ConstraintPlan(
            groups=topo.GroupTable({'a': [0, 1]}),
            actions=[records.ConstraintSpec('f', 'fix', 'mask', group='a',
                                            parameters={'axes': [0]})])
        second = records.ConstraintPlan(
            groups=topo.GroupTable({'a': [1, 0]}),
            actions=[records.ConstraintSpec('f', 'fix', 'mask', group='a',
                                            parameters={'axes': [0]})])
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_unknown_cv_reference_is_refused(self):
        with self.assertRaises(KeyError):
            records.ConstraintPlan(
                actions=[records.ConstraintSpec('b', 'bias', 'harmonic',
                                                cv='ghost')])

    def test_unknown_group_reference_is_refused(self):
        with self.assertRaises(KeyError):
            records.ConstraintPlan(
                actions=[records.ConstraintSpec('f', 'fix', 'mask',
                                                group='ghost')])

    def test_ids_can_be_filtered_by_kind(self):
        plan = _plan()
        self.assertEqual(plan.ids('bias'), ['b'])
        self.assertEqual(plan.ids('constraint'), ['x'])
        self.assertEqual(len(plan.ids()), 3)

    def test_negative_index_is_refused(self):
        with self.assertRaises(ValueError):
            records.ConstraintSpec('f', 'fix', 'mask', atoms=[-1, 2])

    def test_self_pair_is_refused(self):
        with self.assertRaises(ValueError):
            records.ConstraintSpec('x', 'constraint', 'shake', pairs=[[2, 2]])

    def test_unsupported_region_is_refused(self):
        with self.assertRaises(ValueError):
            records.ConstraintSpec('b', 'bias', 'harmonic', region='sideways')


class TestCVRecord(unittest.TestCase):

    def test_atom_count_must_match_kind(self):
        with self.assertRaises(ValueError):
            records.CVRecord('a', 'angle', [0, 1])

    def test_dihedral_is_always_periodic(self):
        self.assertTrue(records.CVRecord('d', 'dihedral', [0, 1, 2, 3],
                                         unit='radian').periodic)

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            records.CVRecord('c', 'coordination', [0, 1])

    def test_round_trip(self):
        record = records.CVRecord('r', 'bond', [2, 3])
        self.assertEqual(records.CVRecord.from_dict(record.to_dict()).indices
                         .tolist(), [2, 3])


class TestScheduleRecord(unittest.TestCase):

    def test_scale_and_value_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=0, end_step=5, scale=[0, 1],
                                   value=[0, 2])

    def test_neither_endpoint_is_refused(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=0, end_step=5)

    def test_window_must_open(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=10, end_step=10,
                                   scale=[0, 1])

    def test_a_time_keyed_schedule_is_refused(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=0, end_step=5, scale=[0, 1],
                                   time_dependent=True)


class TestState(unittest.TestCase):

    def test_copy_isolates_hills_and_tokens(self):
        state = records.ConstraintState()
        clone = state.copy()
        clone.hills.append(records.HillRecord([1.0], [0.1], 1.0e-4))
        self.assertEqual(len(state.hills), 0)
        self.assertEqual(len(clone.hills), 1)

    def test_with_hill_returns_a_new_state_and_bumps_history(self):
        state = records.ConstraintState()
        grown = state.with_hill(records.HillRecord([1.0], [0.1], 1.0e-4))
        self.assertEqual(state.history_version, 0)
        self.assertEqual(grown.history_version, 1)

    def test_deposition_token_is_idempotent_across_dict_round_trip(self):
        state = records.ConstraintState().mark(('t', 4, 'm'))
        self.assertTrue(state.spent(('t', 4, 'm')))
        restored = records.ConstraintState.from_dict(state.to_dict())
        self.assertTrue(restored.spent(('t', 4, 'm')))

    def test_a_token_is_per_step_and_per_action(self):
        state = records.ConstraintState().mark(('t', 4, 'm'))
        self.assertFalse(state.spent(('t', 5, 'm')))
        self.assertFalse(state.spent(('t', 4, 'other')))

    def test_anchors_survive_the_round_trip_as_arrays(self):
        state = records.ConstraintState(anchors={'f': np.zeros((2, 3))})
        restored = records.ConstraintState.from_dict(state.to_dict())
        np.testing.assert_allclose(restored.anchors['f'], np.zeros((2, 3)))


class TestHillRecord(unittest.TestCase):

    def test_height_must_be_positive(self):
        with self.assertRaises(ValueError):
            records.HillRecord([1.0], [0.1], 0.0)

    def test_widths_must_all_be_positive(self):
        with self.assertRaises(ValueError):
            records.HillRecord([1.0, 2.0], [0.1, -0.2], 1.0e-4)

    def test_width_count_must_match_cv_count(self):
        with self.assertRaises(ValueError):
            records.HillRecord([1.0, 2.0], [0.1], 1.0e-4)


class TestResult(unittest.TestCase):

    def test_empty_result_allocates_no_arrays(self):
        result = records.ConstraintResult.empty()
        for field in ('coordinates', 'velocities', 'gradient_correction',
                      'gradient_total', 'gradient_projected',
                      'gradient_for_integrator', 'constraint_impulse'):
            self.assertIsNone(getattr(result, field), field)
        self.assertEqual(result.components, {})
        self.assertTrue(result.is_empty)

    def test_empty_result_records_why_it_is_empty(self):
        context = records.EvaluationContext(step=9, evaluation_id='e9')
        result = records.ConstraintResult.empty(context)
        self.assertTrue(result.provenance['plan_empty'])
        self.assertEqual(result.provenance['evaluation_id'], 'e9')

    def test_flat_contributions_are_one_row_per_item(self):
        result = records.ConstraintResult(
            components={'b': {'kind': 'bias', 'energy': -0.001},
                        'w': {'kind': 'wall', 'energy': 0.002},
                        'x': {'kind': 'constraint', 'energy': 0.0}},
            energy_correction=0.001)
        flat = result.flat_contributions()
        self.assertEqual(sorted(flat),
                         ['constraint.bias.b', 'constraint.constraint.x',
                          'constraint.wall.w'])
        self.assertAlmostEqual(sum(flat.values()), 0.001, places=12)

    def test_a_result_with_no_base_keeps_none(self):
        result = records.ConstraintResult(energy_correction=0.5)
        self.assertIsNone(result.energy_base)
        self.assertIsNone(result.gradient_base)

    def test_round_trip_leaves_no_ndarray_behind(self):
        result = records.ConstraintResult(
            coordinates=np.zeros((2, 3)), gradient_total=np.ones((2, 3)),
            components={'b': {'kind': 'bias', 'energy': 1.0,
                              'gradient': np.zeros((2, 3))}})
        payload = result.to_dict()

        def walk(value):
            if isinstance(value, np.ndarray):
                return False
            if isinstance(value, dict):
                return all(walk(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(walk(item) for item in value)
            return True

        self.assertTrue(walk(payload))
        restored = records.ConstraintResult.from_dict(payload)
        np.testing.assert_allclose(restored.coordinates, np.zeros((2, 3)))
        self.assertEqual(restored.components['b']['kind'], 'bias')

    def test_four_gradient_roles_are_distinct_fields(self):
        result = records.ConstraintResult(
            gradient_total=np.ones((1, 3)),
            gradient_projected=np.zeros((1, 3)),
            gradient_for_integrator=np.full((1, 3), 2.0),
            constraint_impulse=np.full((1, 3), 3.0))
        self.assertEqual(result.to_dict()['gradient_for_integrator'][0],
                         [2.0, 2.0, 2.0])
        self.assertEqual(result.to_dict()['constraint_impulse'][0],
                         [3.0, 3.0, 3.0])


class TestStatistics(unittest.TestCase):

    def test_zero_dof_has_no_temperature(self):
        statistics = records.ConstraintStatistics(free_dof=0,
                                                  free_kinetic=0.0)
        self.assertIsNone(statistics.temperature(3.1668e-6))

    def test_temperature_uses_the_free_count(self):
        statistics = records.ConstraintStatistics(free_dof=6,
                                                 free_kinetic=3.0 * 1.0e-3)
        self.assertAlmostEqual(statistics.temperature(1.0e-3), 1.0, places=12)

    def test_round_trip(self):
        statistics = records.ConstraintStatistics(free_dof=3, rank=2,
                                                 residual=1.0e-9)
        restored = records.ConstraintStatistics.from_dict(
            statistics.to_dict())
        self.assertEqual((restored.free_dof, restored.rank), (3, 2))
        self.assertAlmostEqual(restored.residual, 1.0e-9)


class TestGroupTable(unittest.TestCase):

    def test_membership_and_lookup(self):
        table = topo.GroupTable({'a': [2, 0, 1]})
        self.assertIn('a', table)
        self.assertEqual(table.members('a').tolist(), [2, 0, 1])
        with self.assertRaises(KeyError):
            table.members('b')

    def test_empty_and_duplicate_and_negative_are_refused(self):
        for members in ([], [0, 0, 1], [-1, 2]):
            with self.assertRaises(ValueError):
                topo.GroupTable({'a': members})

    def test_overlap_is_reported_not_assumed(self):
        table = topo.GroupTable({'a': [0, 1], 'b': [1, 2], 'c': [3]})
        self.assertEqual(table.overlaps(), [('a', 'b')])

    def test_fingerprint_moves_with_order(self):
        first = topo.GroupTable({'a': [0, 1]})
        second = topo.GroupTable({'a': [1, 0]})
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_atom_count_helper(self):
        self.assertEqual(topo.atom_count(topo.GroupTable({'a': [0, 4]})), 5)


class TestContext(unittest.TestCase):

    def test_unknown_purpose_is_refused(self):
        with self.assertRaises(ValueError):
            records.EvaluationContext(purpose='guess')

    def test_round_trip(self):
        context = records.EvaluationContext(purpose='md', step=3, time=1.5,
                                            dt=20.0, evaluation_id='e',
                                            cell=np.eye(3) * 10.0,
                                            history_version=2,
                                            trajectory_id='run-a')
        restored = records.EvaluationContext.from_dict(context.to_dict())
        self.assertEqual(restored.trajectory_id, 'run-a')
        np.testing.assert_allclose(restored.cell, np.eye(3) * 10.0)


if __name__ == '__main__':
    unittest.main()
