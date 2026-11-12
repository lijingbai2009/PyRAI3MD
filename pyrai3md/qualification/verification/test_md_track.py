"""
======================================================================
Test suite for the md trajectory early stops: the legacy geometry
tracker (track_type/track_index/track_thrhd/track_stop) and the state
population tolerance (check_population/pop_tolerance).

The geometry reading is pure and is tested without a workflow; the two
stop decisions are tested through AIMD with a mock calculator, because
what matters there is which step the run leaves on, whether the
offending frame was written, and whether the run-log sentence names the
parameter. The input-parity test pins the one decrement from the
external 1-based atom numbering, so a classic declaration and a yaml
declaration of the same pair name the same atoms.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.data.geometry import continuous_coordinates
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.data.state import PyRAIState
from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.schema.normalize import normalize
from pyrai3md.utilities import tracker
from pyrai3md.workflow.md import AIMD

COORDS = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                   [0.0, 3.0, 0.0], [0.0, 0.0, 0.5]])


class MockCalculator:
    """Calculator stub: fixed energies, a zero gradient and extra results."""

    def __init__(self, extra=None):
        self.state = None
        self.extra = dict(extra or {})

    def get_state(self):
        return self.state

    def run(self, target_states=None):
        del target_states
        results = {'energy': [-75.0, -74.0],
                   'gradient': np.zeros((2, 3))}
        results.update(self.extra)
        self.state.set_result(results)
        return {'energy': [-75.0, -74.0]}

    def sync(self):
        pass


class RecordingLogger:
    """Run-log stub: keeps what the stop sentence is written to."""

    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def log_step(self, step, snapshot, thresholds=None):
        del step, snapshot, thresholds


def make_state(total_steps=4, coordinates=None, density=None):
    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = (list(coordinates)
                                 if coordinates is not None
                                 else [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    config.system.natom = 2
    config.system.charges = 0
    config.system.multiplicity = 1
    config.system.nstate = 2
    if density is not None:
        config.system.state_density = density
    state = PyRAIState(runtime=RuntimeState(total_steps=total_steps))
    state.pmd = config
    state.logger = RecordingLogger()
    return state


class IsolatedCwdTestCase(unittest.TestCase):
    """Run each test in a temporary directory (the trajectory files)."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmpdir.cleanup()


class TestTrackPlan(unittest.TestCase):
    """The declaration is refused by name, never silently ignored."""

    def test_dist_plan(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[1, 2], [3, 4]],
                             'track_thrhd': [2.0, 1.0],
                             'track_stop': 1})
        self.assertEqual(spec['type'], 'dist')
        self.assertEqual(spec['groups'], [[1, 2], [3, 4]])
        self.assertEqual(spec['thresholds'], [2.0, 1.0])
        self.assertEqual(spec['stop'], 1)

    def test_no_declaration_is_none(self):
        self.assertIsNone(tracker.plan({}))
        self.assertIsNone(tracker.plan(None))

    def test_frag_needs_two_groups(self):
        with self.assertRaises(tracker.TrackError):
            tracker.plan({'track_type': 'frag', 'track_index': [[1, 2]],
                          'track_thrhd': [2.0]})

    def test_dist_entry_needs_two_atoms(self):
        with self.assertRaises(tracker.TrackError):
            tracker.plan({'track_type': 'dist', 'track_index': [[1, 2, 3]],
                          'track_thrhd': [2.0]})

    def test_unknown_type_is_refused(self):
        with self.assertRaises(tracker.TrackError):
            tracker.plan({'track_type': 'bond', 'track_index': [[1, 2]],
                          'track_thrhd': [2.0]})

    def test_index_without_type_is_refused(self):
        with self.assertRaises(tracker.TrackError):
            tracker.plan({'track_index': [[1, 2]], 'track_thrhd': [2.0]})

    def test_threshold_list_cannot_be_empty(self):
        with self.assertRaises(tracker.TrackError):
            tracker.plan({'track_type': 'dist', 'track_index': [[1, 2]],
                          'track_thrhd': []})

    def test_hop_suppression_is_not_ported(self):
        with self.assertRaises(tracker.TrackError) as caught:
            tracker.plan({'track_type': 'dist', 'track_index': [[1, 2]],
                          'track_thrhd': [2.0], 'track_stop': 2})
        self.assertIn('not implemented', str(caught.exception))

    def test_schema_refuses_a_bad_declaration(self):
        with self.assertRaises(Exception):
            build_config({'control': {'title': 't'},
                          'system': {'track_type': 'dist',
                                     'track_index': [[1, 2, 3]],
                                     'track_thrhd': [2.0]}})


class TestTrackEvaluate(unittest.TestCase):
    """One distance per tracked object, strict `>` and legacy broadcast."""

    def test_dist_rows_and_strict_comparison(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1]],
                             'track_thrhd': [1.5]})
        rows = tracker.evaluate(spec, COORDS)
        self.assertEqual(rows[0]['atoms'], [1, 2])
        self.assertAlmostEqual(rows[0]['distance'], 1.5)
        # Equality does not exceed, which is the legacy rule.
        self.assertFalse(rows[0]['exceeded'])
        self.assertFalse(tracker.exceeded(rows))

    def test_last_threshold_is_broadcast(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1], [2, 3]],
                             'track_thrhd': [2.0]})
        rows = tracker.evaluate(spec, COORDS)
        self.assertEqual([row['threshold'] for row in rows], [2.0, 2.0])
        self.assertTrue(rows[1]['exceeded'])

    def test_extra_thresholds_are_not_read(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1]],
                             'track_thrhd': [2.0, 0.5]})
        rows = tracker.evaluate(spec, COORDS)
        self.assertEqual(rows[0]['threshold'], 2.0)
        self.assertFalse(rows[0]['exceeded'])

    def test_frag_compares_group_centers(self):
        spec = tracker.plan({'track_type': 'frag',
                             'track_index': [[0, 1], [2, 3]],
                             'track_thrhd': [1.0]})
        rows = tracker.evaluate(spec, COORDS)
        first = COORDS[[0, 1]].mean(axis=0)
        second = COORDS[[2, 3]].mean(axis=0)
        self.assertAlmostEqual(rows[0]['distance'],
                               float(np.linalg.norm(first - second)))
        self.assertEqual(rows[0]['groups'], [[1, 2], [3, 4]])
        self.assertTrue(rows[0]['exceeded'])

    def test_out_of_range_atom_is_refused(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 9]],
                             'track_thrhd': [2.0]})
        with self.assertRaises(tracker.TrackError):
            tracker.evaluate(spec, COORDS)

    def test_stop_reason_names_the_parameter(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1]],
                             'track_thrhd': [0.5]})
        reason = tracker.stop_reason(tracker.evaluate(spec, COORDS))
        self.assertIn('distance 1-2', reason)
        self.assertIn('Angstrom', reason)


class TestTrackInputParity(unittest.TestCase):
    """Classic text and yaml name the same atoms for the same declaration."""

    def test_one_based_decrement_in_both_spellings(self):
        yaml_config = normalize(build_config(
            {'control': {'title': 't'},
             'system': {'track_type': 'dist',
                        'track_index': [[1, 2], [3, 4]],
                        'track_thrhd': [2.0],
                        'track_stop': 1}}))
        from pyrai3md.interface.parser import _parse_text
        text = ('&control\n'
                'title t\n'
                '&molecule\n'
                'track_type dist\n'
                'track_index 1 2, 3 4\n'
                'track_thrhd 2.0\n'
                'track_stop 1\n')
        classic_config = normalize(build_config(
            _parse_text(text.splitlines(keepends=True))))
        self.assertEqual(yaml_config.system.track_index,
                         classic_config.system.track_index)
        self.assertEqual(yaml_config.system.track_index, [[0, 1], [2, 3]])


class TestTrackInPeriodicCell(unittest.TestCase):
    """The reading uses the continuous path, not the wrapped one."""

    def test_continuous_coordinates_reduce_the_pair(self):
        coordinates = np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
        source = {'pbc': [1, 0, 0],
                  'lattice_vector': [[10.0, 0.0, 0.0],
                                     [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]],
                  'image': [[0, 0, 0], [-1, 0, 0]]}
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1]],
                             'track_thrhd': [1.0]})
        wrapped = tracker.evaluate(spec, coordinates)[0]
        continuous = tracker.evaluate(
            spec, continuous_coordinates(coordinates, source))[0]
        self.assertAlmostEqual(wrapped['distance'], 9.5)
        self.assertAlmostEqual(continuous['distance'], 0.5)
        self.assertTrue(wrapped['exceeded'])
        self.assertFalse(continuous['exceeded'])


class TestTrackWorkflow(IsolatedCwdTestCase):
    """A tracked bound ends the trajectory after the offending frame."""

    def test_exceeded_pair_stops_after_its_frame(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1]],
                             'track_thrhd': [0.5],
                             'track_stop': 1})
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=4, dt=1.0,
                        track=spec)
        state = make_state(total_steps=4)
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 1)
        self.assertEqual(state.history_size, 1)
        snapshot = state.get_history_step(-1)
        self.assertTrue(
            snapshot['results']['geometry_track'][0]['exceeded'])
        self.assertIn('MD stopped at step 1', state.logger.lines[-1])
        self.assertIn('distance 1-2', state.logger.lines[-1])
        self.assertEqual(state.runtime.stop_code, 2)

    def test_report_only_runs_to_the_end(self):
        spec = tracker.plan({'track_type': 'dist',
                             'track_index': [[0, 1]],
                             'track_thrhd': [0.5],
                             'track_stop': 0})
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0,
                        track=spec)
        state = make_state(total_steps=3)
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 3)
        self.assertFalse(state.logger.lines)
        self.assertEqual(state.runtime.stop_code, 0)

    def test_no_declaration_changes_nothing(self):
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0)
        state = make_state(total_steps=3)
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 3)
        self.assertNotIn('geometry_track',
                         state.get_history_step(-1)['results'])


class TestPopulationGuard(IsolatedCwdTestCase):
    """A population outside [-tol, 1+tol] ends the trajectory."""

    def _density(self, value):
        return [[[value, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]

    def test_out_of_range_stops(self):
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=4, dt=1.0,
                        population_tolerance=0.01)
        state = make_state(total_steps=4, density=self._density(1.2))
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 1)
        self.assertIn('outside [-0.0100, 1.0100]',
                      state.logger.lines[-1])
        self.assertEqual(state.runtime.stop_code, 1)

    def test_inside_tolerance_runs_on(self):
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0,
                        population_tolerance=0.01)
        state = make_state(total_steps=3, density=self._density(1.005))
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 3)
        self.assertFalse(state.logger.lines)
        self.assertEqual(state.runtime.stop_code, 0)

    def test_tolerance_none_disables(self):
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0,
                        population_tolerance=None)
        state = make_state(total_steps=3, density=self._density(1.2))
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 3)
        self.assertEqual(state.runtime.stop_code, 0)

    def test_prediction_error_stop_code(self):
        calculator = MockCalculator(extra={'err_grad': 1.0})
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0,
                        deviation_thresholds={'err_grad': 0.1})
        state = make_state(total_steps=3)
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 1)
        self.assertEqual(state.runtime.stop_code, 3)

    def test_non_finite_stops(self):
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0,
                        population_tolerance=0.01)
        state = make_state(total_steps=3,
                           density=self._density(float('nan')))
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 1)
        self.assertIn('not finite', state.logger.lines[-1])
        self.assertEqual(state.runtime.stop_code, 1)

    def test_no_density_is_not_judged(self):
        calculator = MockCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0,
                        population_tolerance=0.01)
        state = make_state(total_steps=3)
        calculator.state = state
        workflow.run()
        self.assertEqual(state.runtime.current_step, 3)

    def test_state_kwargs_reads_md_maxpop(self):
        state = make_state()
        state.pmd.dynamics.md_maxpop = 0.02
        self.assertAlmostEqual(
            AIMD.state_kwargs(state)['population_tolerance'], 0.02)
        state.pmd.dynamics.md_maxpop = None
        self.assertIsNone(
            AIMD.state_kwargs(state)['population_tolerance'])

    def test_md_maxpop_default_is_the_legacy_bound(self):
        state = make_state()
        self.assertAlmostEqual(
            AIMD.state_kwargs(state)['population_tolerance'], 0.01)

    def test_md_maxpop_is_an_md_keyword(self):
        from pyrai3md.interface.parser import _parse_text
        text = ('&control\n'
                'title t\n'
                '&md\n'
                'md_maxpop 0.02\n')
        config = normalize(build_config(
            _parse_text(text.splitlines(keepends=True))))
        self.assertAlmostEqual(config.dynamics.md_maxpop, 0.02)

    def test_state_kwargs_reads_the_track_declaration(self):
        state = make_state()
        state.pmd.system.track_type = 'dist'
        state.pmd.system.track_index = [[0, 1]]
        state.pmd.system.track_thrhd = [2.0]
        state.pmd.system.track_stop = 1
        spec = AIMD.state_kwargs(state)['track']
        self.assertEqual(spec['type'], 'dist')
        self.assertEqual(spec['stop'], 1)


class TestCalculationFailure(IsolatedCwdTestCase):
    """A failed calculation is code 4 and still raises."""

    def test_failure_sets_code_four_and_keeps_the_exception(self):
        class FailingCalculator(MockCalculator):
            def run(self, target_states=None):
                del target_states
                raise RuntimeError('SCF did not converge')

        calculator = FailingCalculator()
        workflow = AIMD(calculator=calculator, maxiter=3, dt=1.0)
        state = make_state(total_steps=3)
        calculator.state = state
        with self.assertRaises(RuntimeError):
            workflow.run()
        self.assertEqual(state.runtime.stop_code, 4)
        self.assertEqual(state.runtime.status, 'failed')
        self.assertIn('step 1', state.runtime.error_message)
        self.assertIn('SCF did not converge', state.runtime.error_message)


class TestStopCodeSerialization(unittest.TestCase):
    """The stop code survives the runtime document and old files default to 0."""

    def test_round_trip(self):
        runtime = RuntimeState(stop_code=1)
        restored = RuntimeState.from_dict(runtime.to_dict())
        self.assertEqual(restored.stop_code, 1)

    def test_a_document_without_the_field_reads_normal(self):
        data = RuntimeState().to_dict()
        data.pop('stop_code')
        self.assertEqual(RuntimeState.from_dict(data).stop_code, 0)


class TestStopCodeLog(unittest.TestCase):
    """The run-log tail names a non-zero code on its second-to-last line."""

    def test_note_is_the_second_to_last_line(self):
        from pyrai3md.kernel.logger import RunLogger
        state = make_state()
        with tempfile.TemporaryDirectory() as folder:
            logger = RunLogger(state, log_dir=folder)
            logger.write_start_line()
            logger.finish(
                'completed', note='MD stop code: 1 (ill state population)')
            with open(logger.log_file) as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines[-2],
                         'MD stop code: 1 (ill state population)')
        self.assertTrue(lines[-1].startswith('PyRAI3MD End:'))

    def test_zero_code_writes_no_note(self):
        from pyrai3md.kernel.logger import RunLogger, stop_code_note
        self.assertEqual(stop_code_note(0), '')
        self.assertIn('calculation failed', stop_code_note(4))
        state = make_state()
        with tempfile.TemporaryDirectory() as folder:
            logger = RunLogger(state, log_dir=folder)
            logger.finish('completed')
            with open(logger.log_file) as handle:
                lines = handle.read().splitlines()
        self.assertTrue(lines[-1].startswith('PyRAI3MD End:'))


class TestRunLogTail(IsolatedCwdTestCase):
    """The framework log leaves the code on the line before its end line."""

    def test_failed_run_prints_the_code_before_the_end_line(self):
        from unittest import mock
        from pyrai3md import pmd

        state = make_state()
        state.pmd.control.jobtype = 'md'
        state.runtime.stop_code = 4
        with mock.patch('pyrai3md.workflow.execute',
                        side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                pmd.run_with_logger(state)
        with open('pyrai3md.log') as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[-2],
                         'MD stop code: 4 (calculation failed)')
        self.assertIn('Status: error:', lines[-1])


class TestTrackSectionFormat(unittest.TestCase):
    """The run log carries the reading next to the step it describes."""

    def test_monitor_section(self):
        from pyrai3md.utilities.print_results import format_result_block
        results = {'geometry_track': [
            {'kind': 'dist', 'atoms': [1, 2], 'distance': 1.0,
             'threshold': 0.5, 'exceeded': True},
        ]}
        block = format_result_block({'atoms': ['C', 'H']}, results,
                                    step=1, runtime={}, dynamics={})
        self.assertIn('&geometric parameter monitor', block)
        self.assertIn('1-2', block)
        self.assertIn('True', block)


if __name__ == '__main__':
    unittest.main()