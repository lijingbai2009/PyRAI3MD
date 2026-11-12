"""
======================================================================
Test suite for the perf benchmark task (flat layout:
benchmark/perf.py + perf.yaml, the numpy vs numba computational
efficiency comparison of the FSSH electronic propagator).

Tests cover:
- build_legs: one leg per backend, every fssh_* kernel keyword
  pinned to the leg backend, the function-level profiler switch,
  the explicit control gl_seed (identical per-trajectory random
  stream in both legs) and the timed command format
- numba is an optional dependency: without it the leg builder
  raises and the planner reports a diagnostic skip task
- evaluate_run on a synthetic run directory: the wall-time and
  agreement metrics, the PASS/FAIL verdict protocol against the
  case tolerance, the persisted numpy.pop/numba.pop/perf.json,
  the best-effort function-level profile parsing and the ERROR
  verdict on missing leg outputs
- timing wrapper: the captured stdout file, the leg.timing wall
  time and the subprocess return-code passthrough (mocked
  pyrai3md subprocess)

Sep 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import yaml

from pyrai3md.qualification.benchmark import perf as perf_task
from pyrai3md.qualification.benchmark import planner
from pyrai3md.qualification.benchmark import timing as timing_task


BENCHMARK_DIR = os.path.dirname(planner.__file__)
SHIPPED_CASE = os.path.join(BENCHMARK_DIR, 'perf.yaml')


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class TestPerfBuildLegs(unittest.TestCase):
    """Leg builder: per-backend inputs and the timed commands."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.case = perf_task.load_case(SHIPPED_CASE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_shipped_case_declares_the_perf_contract(self):
        self.assertEqual(self.case['case'], 'perf')
        self.assertEqual(self.case['production']['method'], 'tully1')
        # The inner loop amplification that makes the 2x2 kernels
        # measurable ships with the case.
        self.assertEqual(self.case['production']['substep'], 50)
        self.assertIn('backend_population_max_dev',
                      self.case['metrics'])

    def test_build_legs_pins_every_kernel_per_backend(self):
        run_dir = os.path.join(self.tmpdir, 'run')
        os.makedirs(run_dir)
        # numba is an optional dependency: build_legs only checks the
        # availability flag (build_production_input writes the leg
        # inputs without importing numba), so pin it on to exercise
        # the build logic on hosts without the optional dependency.
        with mock.patch(
                'pyrai3md.workflow.integrator.fssh_numba.'
                'NUMBA_AVAILABLE', True):
            legs = perf_task.build_legs(self.case, run_dir)

        self.assertEqual([leg['name'] for leg in legs],
                         ['numpy', 'numba'])
        for leg in legs:
            backend = leg['name']
            self.assertEqual(
                leg['leg_dir'], os.path.join(run_dir, backend))
            self.assertEqual(leg['input_path'], os.path.join(
                run_dir, backend, 'production.yaml'))
            # The timed command cds into the leg directory before
            # launching the timing wrapper.
            self.assertIn('cd %s &&' % leg['leg_dir'], leg['command'])
            self.assertIn('pyrai3md.qualification.benchmark.timing',
                          leg['command'])
            self.assertTrue(leg['command'].rstrip().endswith(backend))

            with open(leg['input_path'], 'r',
                      encoding='utf-8') as f:
                workflow = yaml.safe_load(f)
            md = workflow['md']
            # Every FSSH kernel pinned to the leg backend and the
            # function-level profiler switched on.
            for key in perf_task.FSSH_KERNEL_KEYS:
                self.assertEqual(md[key], backend)
            self.assertEqual(md['fssh_profile'], 1)
            # The physical setup of the case survives the backend
            # stamping (the inner-loop amplification included).
            self.assertEqual(md['substep'], 50)
            self.assertEqual(md['sfhp'], 'fssh')
            # Both legs draw the identical random stream: the
            # control seed is written explicitly.
            self.assertEqual(workflow['control']['gl_seed'], 1)
            self.assertEqual(workflow['control']['title'],
                             'benchmark_perf_%s' % backend)

    def test_build_legs_without_numba_raises(self):
        run_dir = os.path.join(self.tmpdir, 'run')
        os.makedirs(run_dir)
        with mock.patch(
                'pyrai3md.workflow.integrator.fssh_numba.'
                'NUMBA_AVAILABLE', False):
            with self.assertRaises(RuntimeError) as ctx:
                perf_task.build_legs(self.case, run_dir)
        self.assertIn('numba is not installed', str(ctx.exception))


class TestPerfPlannerSkip(unittest.TestCase):
    """The planner turns a missing numba into a diagnostic skip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_numba_yields_diagnostic_skip(self):
        config = {
            'environment': {'scheduler': ''},
            'slurm': {'partition': 'test'},
            'benchmark': {'perf': {'method': 'tully1'}},
        }
        with mock.patch(
                'pyrai3md.workflow.integrator.fssh_numba.'
                'NUMBA_AVAILABLE', False):
            tasks = planner.plan_benchmark_tests(config, self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['skip'])
        self.assertEqual(tasks[0]['name'], 'benchmark_perf_tully1')
        self.assertIn('input generation failed',
                      tasks[0]['skip_reason'])
        self.assertIn('numba is not installed',
                      tasks[0]['skip_reason'])


class TestPerfEvaluateRun(unittest.TestCase):
    """Verdict protocol of evaluate_run on a synthetic run dir."""

    NSTEP = 3
    DT = 20.0

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_case(self, tolerance=1.0e-10):
        with open(os.path.join(self.tmpdir, 'perf.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'perf',
                'production': {'nstep': self.NSTEP, 'dt': self.DT},
                'metrics': {'backend_population_max_dev':
                            {'tolerance': tolerance}},
            }, f)

    def _write_leg(self, backend, wall, rows, profile=None):
        leg_dir = os.path.join(self.tmpdir, backend)
        os.makedirs(leg_dir, exist_ok=True)
        with open(os.path.join(leg_dir, 'leg.timing'), 'w',
                  encoding='utf-8') as f:
            f.write('%.6f\n' % wall)
        with open(os.path.join(leg_dir,
                               'benchmark_perf_%s.md.pop' % backend),
                  'w', encoding='utf-8') as f:
            f.write('           time            Pop1'
                    '            Pop2            Act1'
                    '            Act2\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f %.1f %.1f\n' % (
                    step * self.DT, p0, p1,
                    1.0 if p0 > p1 else 0.0,
                    0.0 if p0 > p1 else 1.0))
        if profile is not None:
            with open(os.path.join(leg_dir, '%s.stdout' % backend),
                      'w', encoding='utf-8') as f:
                f.write(profile)

    @staticmethod
    def _profile_block(func_seconds):
        """One captured 'FSSH Performance' table (profiler layout)."""
        lines = ['', 'FSSH Performance', '-' * 48,
                 '%-18s %10s %10s' % ('Function', 'Calls', 'Total')]
        total = 0.0
        for func, (calls, seconds) in func_seconds.items():
            total += seconds
            lines.append('%-18s %10d %9.2f s' % (func, calls, seconds))
        lines.append('%-18s %10s %9.2f s' % ('Total', '', total))
        lines.extend(['', 'Implementation', '-' * 48])
        return '\n'.join(lines) + '\n'

    def test_matching_legs_pass_and_persist(self):
        self._write_case()
        rows = [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)]
        self._write_leg('numpy', 10.0, rows, self._profile_block({
            'dPdt': (300, 1.20), 'RK4': (300, 2.40)}))
        self._write_leg('numba', 5.0, rows, self._profile_block({
            'dPdt': (300, 0.60), 'RK4': (300, 0.80)}))

        report = perf_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(report['case'], 'perf')
        metrics = report['metrics']
        self.assertAlmostEqual(metrics['numpy_wall'], 10.0)
        self.assertAlmostEqual(metrics['numba_wall'], 5.0)
        self.assertAlmostEqual(metrics['speedup_wall'], 2.0)
        # Identical legs: agreement at machine zero.
        self.assertAlmostEqual(
            metrics['backend_population_max_dev'], 0.0)
        # Function-level breakdown of the parsed profile tables.
        self.assertAlmostEqual(metrics['profile_dPdt'], 0.60)
        self.assertAlmostEqual(metrics['speedup_dPdt'], 2.0)
        self.assertAlmostEqual(metrics['speedup_RK4'], 3.0)

        # Persistence: both ensemble averages plus the report.
        for name in ('numpy.pop', 'numba.pop', 'perf.json'):
            self.assertTrue(
                os.path.isfile(os.path.join(self.tmpdir, name)),
                'missing %s' % name)
        import numpy as np
        saved = np.loadtxt(os.path.join(self.tmpdir, 'numpy.pop'))
        self.assertEqual(saved.shape, (self.NSTEP, 3))
        np.testing.assert_allclose(saved[:, 1:], rows, atol=1e-6)

    def test_disagreeing_legs_fail(self):
        self._write_case()
        self._write_leg('numpy', 10.0,
                        [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)])
        # 0.02 deviation blows the 1e-10 agreement tolerance.
        self._write_leg('numba', 5.0,
                        [(0.88, 0.12), (0.6, 0.4), (0.4, 0.6)])

        report = perf_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertAlmostEqual(
            report['metrics']['backend_population_max_dev'], 0.02)
        self.assertFalse(report['details']['per_metric'][
            'backend_population_max_dev'])

    def test_missing_leg_outputs_report_error(self):
        self._write_case()
        report = perf_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'ERROR')
        self.assertIn('error', report['details'])
        # A broken run never promises a persisted report.
        self.assertFalse(os.path.isfile(os.path.join(
            self.tmpdir, 'perf.json')))

    def test_parse_profile_skips_header_and_total(self):
        self._write_case()
        rows = [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)]
        block = self._profile_block({'dPdt': (100, 1.00)})
        self._write_leg('numpy', 1.0, rows, block + block)
        self._write_leg('numba', 1.0, rows, block)

        table = perf_task.parse_profile(os.path.join(
            self.tmpdir, 'numpy', 'numpy.stdout'))
        # The last block of the capture wins; the header and the
        # Total row never land in the function map.
        self.assertEqual(table, {'dPdt': (100, 1.00)})
        self.assertIsNone(perf_task.parse_profile(os.path.join(
            self.tmpdir, 'absent.stdout')))


class TestTimingWrapper(unittest.TestCase):
    """Wall-time capture and return-code passthrough of run_timed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_timed_captures_output_and_wall_time(self):
        leg_dir = os.path.join(self.tmpdir, 'leg')
        input_path = os.path.join(self.tmpdir, 'input.yaml')
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write('control: {}\n')

        def fake_call(cmd, cwd=None, stdout=None, stderr=None):
            self.assertEqual(cmd, ['pyrai3md', input_path])
            self.assertEqual(cwd, leg_dir)
            stdout.write('calculation output\n')
            return 0

        with mock.patch.object(timing_task.subprocess, 'call',
                               side_effect=fake_call):
            code = timing_task.run_timed(input_path, leg_dir, 'numpy')
        self.assertEqual(code, 0)
        self.assertEqual(
            _read(os.path.join(leg_dir, 'numpy.stdout')),
            'calculation output\n')
        wall = float(_read(os.path.join(leg_dir, 'leg.timing')))
        self.assertGreaterEqual(wall, 0.0)

    def test_run_timed_passes_the_subprocess_return_code(self):
        leg_dir = os.path.join(self.tmpdir, 'leg')
        with mock.patch.object(timing_task.subprocess, 'call',
                               return_value=3):
            code = timing_task.run_timed(
                os.path.join(self.tmpdir, 'input.yaml'),
                leg_dir, 'numba')
        self.assertEqual(code, 3)
        # The timing file is written even for failing legs so the
        # evaluation can still report the wall time of the crash.
        self.assertTrue(os.path.isfile(
            os.path.join(leg_dir, 'leg.timing')))

    def test_main_requires_three_arguments(self):
        stdout = io.StringIO()
        with mock.patch('sys.stdout', stdout):
            self.assertEqual(timing_task.main([]), 2)
        self.assertIn('usage', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
