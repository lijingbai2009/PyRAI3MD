"""
======================================================================
Test suite for the trajectory result containers
(pyrai3md/workflow/records/trajectory.py).

Tests cover:
- TrajectoryResult accessors and the tolerant history interface
  (missing state never raises)
- TrajectorySet ordering by the planning idx, index access and the
  trajectory-count-1 case of a single-trajectory run

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest


class FakeState:
    """Minimal state double exposing the history interface."""

    def __init__(self, snapshots=None):
        self._snapshots = list(snapshots or [])

    def get_history(self, n=None):
        if n is None:
            return list(self._snapshots)
        return list(self._snapshots[-n:])

    def get_history_step(self, index=-1, copy=True):
        # `copy` is part of the real interface (a hot-loop reader takes the
        # frame by reference); the double keeps the frames literal, so the
        # flag changes nothing here.
        del copy
        try:
            return self._snapshots[index]
        except IndexError:
            return None


class TestTrajectoryResult(unittest.TestCase):
    """Test one trajectory record."""

    def test_state_returns_state_like_outcome(self):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        state = FakeState()
        record = TrajectoryResult(0, state, runtime_idx='runtime_1')
        self.assertEqual(record.idx, 0)
        self.assertEqual(record.runtime_idx, 'runtime_1')
        self.assertIs(record.state, state)

    def test_state_returns_none_for_non_state_outcome(self):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        record = TrajectoryResult(2, RuntimeError('failed'))
        self.assertIsNone(record.state)
        # Plain dicts are not states either.
        self.assertIsNone(TrajectoryResult(0, {'energy': 1.0}).state)

    def test_history_reads_state_history(self):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        snapshots = [{'step': 1}, {'step': 2}, {'step': 3}]
        record = TrajectoryResult(0, FakeState(snapshots))
        self.assertEqual(len(record.history()), 3)
        self.assertEqual(record.history(2)[0], {'step': 2})
        self.assertEqual(record.history_step(-1), {'step': 3})
        self.assertEqual(record.history_step(0), {'step': 1})

    def test_history_tolerates_missing_state(self):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        record = TrajectoryResult(1, None)
        self.assertEqual(record.history(), [])
        self.assertEqual(record.history(3), [])
        self.assertIsNone(record.history_step(-1))

    def test_history_step_missing_returns_none(self):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        record = TrajectoryResult(0, FakeState([{'step': 1}]))
        self.assertIsNone(record.history_step(5))


class TestTrajectorySet(unittest.TestCase):
    """Test the ordered multi-trajectory return container."""

    def test_single_trajectory_case(self):
        from pyrai3md.workflow.records.trajectory import TrajectorySet
        state = FakeState()
        trajectories = TrajectorySet([(0, state)])
        self.assertEqual(len(trajectories), 1)
        self.assertIs(trajectories[0].state, state)
        self.assertEqual(trajectories.states, [state])

    def test_ordering_by_planning_idx(self):
        from pyrai3md.workflow.records.trajectory import TrajectorySet
        states = [FakeState() for _ in range(3)]
        # Input order scrambled: the set arranges by idx.
        trajectories = TrajectorySet([(2, states[2]),
                                      (0, states[0]),
                                      (1, states[1])])
        self.assertEqual([record.idx for record in trajectories],
                         [0, 1, 2])
        self.assertEqual(trajectories.states, states)

    def test_index_access_by_idx(self):
        from pyrai3md.workflow.records.trajectory import TrajectorySet
        states = [FakeState(), FakeState()]
        trajectories = TrajectorySet([(0, states[0]), (1, states[1])])
        self.assertIs(trajectories[1].state, states[1])
        with self.assertRaises(KeyError):
            trajectories[5]

    def test_accepts_trajectory_result_entries(self):
        from pyrai3md.workflow.records.trajectory import (
            TrajectoryResult, TrajectorySet)
        record = TrajectoryResult(0, FakeState(), runtime_idx='rt_0')
        trajectories = TrajectorySet([record])
        self.assertIs(trajectories[0], record)

    def test_empty_set(self):
        from pyrai3md.workflow.records.trajectory import TrajectorySet
        trajectories = TrajectorySet()
        self.assertEqual(len(trajectories), 0)
        self.assertEqual(trajectories.states, [])
        self.assertEqual(list(trajectories), [])


class TestTrajectorySummary(unittest.TestCase):
    """The latest recorded step, as one mapping per trajectory."""

    @staticmethod
    def _frame(step=4):
        """One frame shaped the way the history write shapes it."""
        return {
            'step': step,
            'time': step * 20.67,
            'system': {
                'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                'velocities': [[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]],
                'current_state': 0,
                'nstate': 2,
                'state_density': [[[1.0, 0.0], [0.0, 0.0]],
                                  [[0.0, 0.0], [0.0, 0.0]]],
                'image': [],
            },
            'results': {'energy': [-1.0, -0.9],
                        'err_energy': 0.001, 'err_grad': 0.002,
                        'err_nac': None},
            'nuclear_sample': {
                'K_pre': 0.0018,
                'active_gradient': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'active_config_pre': (0,),
            },
        }

    def _record(self, snapshots, idx=0):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        return TrajectoryResult(idx, FakeState(snapshots))

    def test_summary_reads_the_newest_frame(self):
        record = self._record([self._frame(step=3), self._frame(step=4)])
        summary = record.summary()
        self.assertEqual(4, summary['step'])
        self.assertEqual(4 * 20.67, summary['time'])
        self.assertEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                         summary['coordinates'])
        self.assertEqual([-1.0, -0.9], summary['energy'])
        self.assertEqual(-1.0, summary['epot'],
                         'epot is the active state row of the energies')
        self.assertEqual(0.0018, summary['kinetic'])
        self.assertEqual([1.0, 0.0], summary['population'])

    def test_deviations_carry_only_what_the_backend_reported(self):
        # A missing deviation is not a measured agreement: filling it with
        # zero would hand adaptive sampling a free pass.
        summary = self._record([self._frame()]).summary()
        self.assertEqual({'err_energy': 0.001, 'err_grad': 0.002},
                         summary['deviations'])

    def test_summary_is_empty_without_a_recorded_step(self):
        from pyrai3md.workflow.records.trajectory import TrajectoryResult
        self.assertEqual({}, TrajectoryResult(0, None).summary())
        self.assertEqual({}, self._record([]).summary())

    def test_multi_center_adds_the_active_configuration(self):
        frame = self._frame()
        frame['nuclear_sample']['active_config_pre'] = (0, 1)
        summary = self._record([frame]).summary()
        self.assertEqual([0, 1], summary['active_config'])

    def test_no_configuration_key_when_there_is_none(self):
        frame = self._frame()
        frame['nuclear_sample'].pop('active_config_pre')
        self.assertNotIn('active_config', self._record([frame]).summary())

    def test_a_set_exposes_one_summary_per_trajectory_in_idx_order(self):
        from pyrai3md.workflow.records.trajectory import TrajectorySet
        trajectories = TrajectorySet([
            (0, FakeState([self._frame(step=1)])),
            (2, FakeState([self._frame(step=3)])),
            (1, None),
        ])
        summaries = trajectories.summaries()
        self.assertEqual(3, len(summaries))
        # The set is stored sorted by idx, so the rows come back in
        # planning order whatever the input order was. A trajectory with
        # nothing recorded contributes an empty mapping, which is why its
        # position is what identifies it rather than a key inside it.
        self.assertEqual([1, None, 3],
                         [row['step'] if row else None for row in summaries])
        self.assertEqual({}, summaries[1], 'a failed trajectory reports empty')
        self.assertEqual(0, summaries[0]['idx'])
        self.assertEqual(2, summaries[2]['idx'])

    def test_a_single_trajectory_is_the_count_one_case(self):
        from pyrai3md.workflow.records.trajectory import TrajectorySet
        trajectories = TrajectorySet([(0, FakeState([self._frame(step=7)]))])
        self.assertEqual([7], [row['step'] for row in trajectories.summaries()])


if __name__ == '__main__':
    unittest.main()
