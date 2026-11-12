"""
======================================================================
This module locks the visibility of the six read paths that used to
swallow a broken input without a trace.

Each site answers a read it cannot complete with a documented
fallback (no stamping, a jobtype-derived name, no frames, a NaN
column, an omitted report entry). That fallback is fine; what is not
fine is reaching it because the reader itself is broken -- the run
then looks healthy while a product silently loses a field. Every
case below injects one internal defect and requires a warning that
names what was lost; the one case whose defect cannot be answered
with a meaningful fallback must raise instead.

Five of the six swallowed every exception before this module
existed, and the sixth (single.py) swallowed the conversion
failures, so each injected defect was silent; all six criteria
therefore read warning-or-raise only after the narrowing.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import tempfile
import types
import unittest

import numpy as np

from pyrai3md.workflow.adaptive import _child_log_name
from pyrai3md.workflow.dispatch.dispatcher import Dispatcher
from pyrai3md.workflow.dispatch.runtime import resolve_skill_name
from pyrai3md.workflow.records.adaptive import history_frames
from pyrai3md.workflow.records.output import MdOutputWriter
from pyrai3md.workflow.single import SingleShot


class _BrokenQueue:
    """State stand-in whose task queue raises the injected error."""

    def __init__(self, jobtype, error):
        self.jobtype = jobtype
        self.error = error
        self.runtime = None

    def iter_tasks(self):
        raise self.error


class _BrokenHistory:
    """Result stand-in whose history property raises the injected error."""

    error = None

    @property
    def history(self):
        raise self.error


class TestDegradedReadsAreVisible(unittest.TestCase):
    """One injected internal defect per site, one visible consequence."""

    def test_stamp_isolated_workdir_names_the_skipped_stamp(self):
        dispatcher = Dispatcher()
        state = _BrokenQueue('energy', TypeError('theory is not iterable'))
        with self.assertLogs('pyrai3md.dispatch', level='WARNING') as logs:
            dispatcher._stamp_isolated_workdir(state, '/tmp/unit_0')
        self.assertIn('scratch_path was not pointed at /tmp/unit_0',
                      '\n'.join(logs.output))
        self.assertIn('theory is not iterable', '\n'.join(logs.output))

    def test_skill_name_fallback_is_reported(self):
        state = _BrokenQueue('md opt', TypeError('theory is not iterable'))
        with self.assertLogs('pyrai3md.workflow.dispatch.runtime',
                             level='WARNING') as logs:
            name = resolve_skill_name(state)
        self.assertEqual(name, 'md')
        self.assertIn('falls back to the jobtype token',
                      '\n'.join(logs.output))

    def test_child_log_name_fallback_is_reported(self):
        state = _BrokenQueue('train', TypeError('theory is not iterable'))
        with self.assertLogs('pyrai3md.workflow.adaptive',
                             level='WARNING') as logs:
            name = _child_log_name(state)
        self.assertIsNone(name)
        self.assertIn('keeps the control title', '\n'.join(logs.output))

    def test_history_getter_failure_is_reported(self):
        class Outcome:
            def get_history(self, window):
                raise ValueError('history is not a list')

        with self.assertLogs('pyrai3md.workflow.records.adaptive',
                             level='WARNING') as logs:
            frames = history_frames(Outcome(), 3)
        self.assertEqual(frames, [])
        self.assertIn('no history frames', '\n'.join(logs.output))
        self.assertIn('history is not a list', '\n'.join(logs.output))

    def test_broken_history_attribute_fails_instead_of_emptying(self):
        outcome = _BrokenHistory()
        outcome.error = ValueError('history row is malformed')
        with self.assertRaises(ValueError):
            history_frames(outcome, 3)

    def test_ensemble_column_failure_is_reported_once(self):
        with tempfile.TemporaryDirectory() as workdir:
            writer = MdOutputWriter(title='frame_check', log_dir=workdir)

            def broken(ctx):
                raise ValueError('strain is not a number')

            writer._ensemble_col_cache = [('eps_xx', broken)]
            snapshot = {'dynamics': {'ensemble': 'npt'}}
            with self.assertLogs('pyrai3md.workflow.records.output',
                                 level='WARNING') as logs:
                first = writer._ensemble_row(snapshot, None, {}, 0.0)
                second = writer._ensemble_row(snapshot, None, {}, 0.0)
        names, values = first
        self.assertEqual(names, ['eps_xx'])
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(second[1][0]))
        self.assertEqual(len(logs.output), 1)
        self.assertIn('md.ensemble column eps_xx', logs.output[0])
        self.assertIn('written as NaN', logs.output[0])

    def test_unreadable_final_energy_is_reported(self):
        captured = []

        class Recorder(SingleShot):
            def _write_optimize_output(self, state, report):
                captured.append(dict(report))

        workflow = Recorder(state=None)
        state = types.SimpleNamespace(logger=None, runtime=None)
        with self.assertLogs('pyrai3md.workflow.single',
                             level='WARNING') as logs:
            workflow._finalize_optimize(state, {'energy': 'not-a-number'})
        self.assertEqual(captured, [{}])
        self.assertIn('report entry is omitted', '\n'.join(logs.output))
        self.assertIn('not-a-number', '\n'.join(logs.output))


if __name__ == '__main__':
    unittest.main()