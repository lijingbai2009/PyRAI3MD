"""
======================================================================
Test suite for the PhaseProfiler of pyrai3md/utilities/timer.py: the
per-phase wall clock a workflow attaches to its own step loop.

The table it renders is what prices a slow run, so these tests check the
measurement contract rather than any speed: that the rows add up to what
the loop actually took (a phase with no timer would otherwise shrink the
table in silence), that the wall row stays a comment instead of becoming
another phase a reader adds to the total, and that the periodic refresh
leaves the file in the run directory without waiting for a shutdown that
a killed or unclosed run never reaches.

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

from pyrai3md.utilities.timer import (PHASE_FILE, PhaseProfiler, Timer,
                                      how_long, what_is_time)


class TestTimestamps(unittest.TestCase):
    """Test the readable timestamp helpers."""

    def test_what_is_time_is_readable(self):
        self.assertRegex(what_is_time(),
                         r'^[A-Z][a-z]{2} \d{2} \d{4}  \d{2}:\d{2}:\d{2}$')

    def test_how_long_rounds_to_the_second(self):
        # Truncation used to report a sub-second run as all zeros.
        self.assertEqual(how_long(100.0, 100.6), '00:00:00:01')

    def test_how_long_never_goes_negative(self):
        self.assertEqual(how_long(100.0, 99.0), '00:00:00:00')

    def test_how_long_carries_the_days(self):
        self.assertEqual(how_long(0.0, 90061.0), '01:01:01:01')


class TestTimerClass(unittest.TestCase):
    """Test the manual and context-manager forms of Timer."""

    def test_stop_without_start_reports_nothing(self):
        self.assertEqual(Timer('idle').stop(), 0.0)

    def test_context_manager_records_elapsed(self):
        with Timer('block') as timer:
            pass
        self.assertGreaterEqual(timer.elapsed, 0.0)
        self.assertEqual(timer.name, 'block')


class TestPhaseProfiler(unittest.TestCase):
    """Test the phase rows, the wall row and the refresh cadence."""

    def test_measure_records_one_call_per_block(self):
        profiler = PhaseProfiler()
        with profiler.measure('verlet_i'):
            pass
        with profiler.measure('verlet_i'):
            pass
        self.assertEqual(profiler.stats['verlet_i'][0], 2)

    def test_rows_keep_the_order_they_were_recorded_in(self):
        profiler = PhaseProfiler()
        profiler.record('second', 0.001)
        profiler.record('first', 0.002)
        names = [line.split('phase=')[1].split()[0]
                 for line in profiler.render(1)]
        self.assertEqual(names, ['second', 'first', 'TOTAL'])

    def test_the_total_row_is_the_sum_of_the_phases(self):
        profiler = PhaseProfiler()
        profiler.record('a', 0.001)
        profiler.record('b', 0.002)
        rows = profiler.render(10)
        total = rows[-1]
        self.assertIn('total_s=0.0030', total)
        self.assertIn('ms_per_step=0.300', total)

    def test_the_wall_row_is_a_comment_on_the_total(self):
        # A reader sums the '[perf]' rows; the wall it compares them with
        # must not join that sum, which is why it renders as a comment.
        profiler = PhaseProfiler()
        profiler.record('a', 0.010)
        profiler.mark_wall(0.0125)
        rows = profiler.render(1)
        self.assertTrue(rows[-1].startswith('# wall'))
        self.assertIn('coverage=0.800', rows[-1])
        self.assertEqual([line for line in rows if line.startswith('[perf]')],
                         rows[:-1])

    def test_no_wall_means_no_wall_row(self):
        profiler = PhaseProfiler()
        profiler.record('a', 0.001)
        self.assertFalse(any(line.startswith('#') for line in
                             profiler.render(1)))

    def test_render_needs_a_step_to_divide_by(self):
        profiler = PhaseProfiler()
        profiler.record('a', 0.001)
        self.assertEqual(profiler.render(0), [])

    def test_write_lands_in_the_given_directory(self):
        profiler = PhaseProfiler()
        with profiler.measure('a'):
            pass
        profiler.mark_wall(0.001)
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler.write(tmpdir, 1)
            with open(os.path.join(tmpdir, PHASE_FILE),
                      encoding='utf-8') as handle:
                text = handle.read()
        self.assertIn('phase=a', text)
        self.assertIn('coverage=', text)

    def test_nested_rows_stay_under_their_own_header(self):
        profiler = PhaseProfiler()
        with profiler.measure('calculator_run'):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler.write(tmpdir, 1, extra=['[perf] phase=plan steps=1'])
            with open(os.path.join(tmpdir, PHASE_FILE),
                      encoding='utf-8') as handle:
                lines = handle.read().splitlines()
        self.assertIn('# nested breakdown (inside the rows above)', lines)
        self.assertLess(lines.index('[perf] phase=plan steps=1'),
                        len(lines))
        # The nested row is not part of this table's own total.
        self.assertTrue(lines[0].startswith('[perf] phase=calculator_run'))

    def test_refresh_rewrites_only_when_the_numbers_move(self):
        # The two halves are judged by file content, not by mtime: on the
        # test environment six successive writes to one file all reported
        # the same st_mtime_ns, so a timestamp proves neither half. The
        # sentinel written between the calls does prove the first one,
        # because any write would erase it.
        profiler = PhaseProfiler()
        with profiler.measure('a'):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, PHASE_FILE)
            profiler.refresh(tmpdir, 1)
            with open(path, encoding='utf-8') as handle:
                self.assertIn('steps=1', handle.read())

            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('# untouched\n')
            profiler.refresh(tmpdir, 1)
            with open(path, encoding='utf-8') as handle:
                self.assertEqual(handle.read(), '# untouched\n')

            profiler.refresh(tmpdir, 2)
            with open(path, encoding='utf-8') as handle:
                self.assertIn('steps=2', handle.read())

    def test_refresh_on_an_unwritable_directory_is_not_a_failure(self):
        profiler = PhaseProfiler()
        with profiler.measure('a'):
            pass
        profiler.refresh('/proc/this/cannot/be/written', 1)


if __name__ == '__main__':
    unittest.main()
