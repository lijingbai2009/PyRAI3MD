"""
======================================================================
Test suite for the benchmark qualification pillar (flat layout:
benchmark/planner.py + per-task modules + <task>_<case>.yaml).

Tests cover:
- planner: flat case discovery pairing fssh.py with the unified
  fssh.yaml while excluding the infrastructure modules, and the
  layered case YAML resolution (dedicated <task>_<method>.yaml
  over the unified <task>.yaml)
- planner: the single compute task per case with the reference &&
  production chain written into ONE script, the
  reference.yaml/production.yaml pair in the run directory and the
  run_dir/benchmark_task handoff to the runner
- planner: slurm mode adds the .sbatch carrying the identical chain
- planner: the pmd_env benchmark subsection accepts a single
  parameter dict or a list of them (multi-method calibration, one
  independent task per entry), merges the parameter overrides into
  the run-dir case YAML and produces diagnostic skips when
  disabled/malformed/duplicated
- task module (fssh): case loading, pmd_env override merging,
  reference input (framework YAML, driver=skill selecting the skill
  public {method}_reference profile), production input (&md wiring,
  amu mass, initcond '0', the explicit md-keyword forwarding table
  with present-only semantics so the shipped rk4: '1' reaches the
  md section, the lz_a/lz_b qdm forwarding, and the
  unrecognized-production-key guard)
- lz family (fssh + gsh + lzsh): the unified final-state verdict
  shared by all four task modules (the final trajectory / state /
  reference populations, the judged state-reference difference and,
  on the lz1/lz2/lz3 routing, the analytic single-passage
  Landau-Zener probability delegation with the state-analytical
  difference; the gsh/lzsh lz cases add the deterministic kernel
  replay internal probability and the state-internal difference)
- task module (gsh): the Zhu-Nakamura calibration case reuses the
  fssh reference builder and evaluation, owns the GSH production
  input (sfhp gsh, the GSH-only keyword table rejecting fssh-only
  keywords) and plans the reference && production chain like fssh
- task module (lzsh): the Landau-Zener calibration case reuses the
  fssh reference builder and evaluation, owns the LZSH production
  input (sfhp lzsh, the LZ-only keyword table without the
  gradient-only activestate key) and plans the reference &&
  production chain like fssh
- task module (ktdc): the kTDC calibration (FSSH with the
  energy-only curvature coupling, nactype ktdc) reuses the fssh
  task module verbatim (builders, keyword table and evaluator are
  the same module objects) and ships the mirrored ktdc.yaml /
  ktdc_lz1/2/3.yaml cases
- plotting: the population_s0.png rendering of the compared
  population tables (the pinned goldenrod/royalblue/crimson
  colors, the '<method>_<model>' case-key title, the fixed
  [0, 1] probability axis with 0.2 ticks, the degraded-table
  tolerance and the matplotlib-unavailable skip) wired into the
  fssh/gsh/lzsh evaluators before the report returns
- shipped pmd_env.yaml: the full lz calibration family loads with
  duplicate-key protection and plans the dedicated lz case files
- runner: _run_benchmark_test prints the tool report's error text
  for an ERROR verdict instead of a bare {} and the metric payload
  for a FAIL verdict
- task module (fssh): population parsing, the ensemble averaging,
  the metric evaluation and the evaluate_run verdict protocol
  (PASS/ERROR report with the BENCHMARK_JSON last-line emission)
  including the production.pop/reference.pop persistence
- external tool: tools/benchmark.py (next to deploy.py) re-exports
  the fssh task module entry points and owns the CLI entry point
  (the task module itself is programmatic only)
- workflow: AIMD.state_kwargs wiring of &md step/size/sfhp into
  the AIMD constructor

Aug 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import contextlib
import copy
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from pyrai3md.qualification import manager
from pyrai3md.qualification.manager import SWITCH_CONDITION
from pyrai3md.qualification.benchmark import fssh as fssh_task
from pyrai3md.qualification.benchmark import gsh as gsh_task
from pyrai3md.qualification.benchmark import ktdc as ktdc_task
from pyrai3md.qualification.benchmark import lzsh as lzsh_task
from pyrai3md.qualification.benchmark import planner
from pyrai3md.qualification.benchmark import plotting as plotting_mod
from pyrai3md.workflow import AIMD, Optimize
from pyrai3md.data.build import build_state as build_framework_state
from pyrai3md.interface.parser import parse_input as parse_framework_input
from pyrai3md.schema.adapter import build_config as build_framework_config
from pyrai3md.schema.normalize import normalize as normalize_framework
from pyrai3md.schema.validate import validate as validate_framework


BENCHMARK_DIR = os.path.dirname(planner.__file__)
SHIPPED_CASE = os.path.join(BENCHMARK_DIR, 'fssh.yaml')
SHIPPED_GSH_CASE = os.path.join(BENCHMARK_DIR, 'gsh.yaml')
SHIPPED_GSH_LZ1_CASE = os.path.join(BENCHMARK_DIR, 'gsh_lz1.yaml')
SHIPPED_GSH_LZ2_CASE = os.path.join(BENCHMARK_DIR, 'gsh_lz2.yaml')
SHIPPED_GSH_LZ3_CASE = os.path.join(BENCHMARK_DIR, 'gsh_lz3.yaml')
SHIPPED_KTDC_CASE = os.path.join(BENCHMARK_DIR, 'ktdc.yaml')
SHIPPED_KTDC_LZ_CASES = {
    method: os.path.join(BENCHMARK_DIR, 'ktdc_%s.yaml' % method)
    for method in ('lz1', 'lz2', 'lz3')
}
SHIPPED_LZSH_CASE = os.path.join(BENCHMARK_DIR, 'lzsh.yaml')
SHIPPED_LZSH_LZ_CASES = {
    method: os.path.join(BENCHMARK_DIR, 'lzsh_%s.yaml' % method)
    for method in ('lz1', 'lz2', 'lz3')
}
SHIPPED_FSSH_LZ_CASES = {
    method: os.path.join(BENCHMARK_DIR, 'fssh_%s.yaml' % method)
    for method in ('lz1', 'lz2', 'lz3')
}
SHIPPED_PMD_ENV = os.path.join(
    os.path.dirname(BENCHMARK_DIR), 'pmd_env.yaml')


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class TestBenchmarkDiscovery(unittest.TestCase):
    """Flat layout pairing of task modules and case YAMLs."""

    def test_shipped_case_is_paired_with_its_task_module(self):
        cases = planner.discover_cases(BENCHMARK_DIR)
        self.assertIn(
            ('fssh', 'fssh', SHIPPED_CASE), cases)
        self.assertIn(
            ('gsh', 'gsh', SHIPPED_GSH_CASE), cases)
        self.assertIn(
            ('ktdc', 'ktdc', SHIPPED_KTDC_CASE), cases)
        self.assertIn(
            ('lzsh', 'lzsh', SHIPPED_LZSH_CASE), cases)

    def test_infrastructure_modules_are_not_tasks(self):
        cases = planner.discover_cases(BENCHMARK_DIR)
        task_names = {task for task, _key, _path in cases}
        for infra in planner._NON_TASK_MODULES:
            self.assertNotIn(infra, task_names)


class TestBenchmarkPlanner(unittest.TestCase):
    """Single compute task expansion of every case."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _config(self, scheduler='', benchmark=None,
                 runall=SWITCH_CONDITION):
        return {
            '_runall': runall,
            'environment': {'scheduler': scheduler},
            'slurm': {'partition': 'test'},
            'benchmark': benchmark or {},
        }

    def test_local_plan_generates_a_compute_task_per_case(self):
        tasks = planner.plan_benchmark_tests(
            self._config(''), self.tmpdir)
        # One task per case: evaluation is a qualification tool
        # called in-process by the runner, never a planned task.
        # Without a benchmark section every shipped case plans as a
        # single task under its own stem (the fssh, ktdc, gsh and
        # lzsh calibrations, the dedicated lz1/lz2/lz3 Landau-Zener
        # calibrations of all four algorithms and the perf backend
        # comparison).
        self.assertEqual(
            [task['name'] for task in tasks],
            ['benchmark_fssh', 'benchmark_fssh_lz1',
             'benchmark_fssh_lz2', 'benchmark_fssh_lz3',
             'benchmark_gsh', 'benchmark_gsh_lz1',
             'benchmark_gsh_lz2', 'benchmark_gsh_lz3',
             'benchmark_ktdc', 'benchmark_ktdc_lz1',
             'benchmark_ktdc_lz2', 'benchmark_ktdc_lz3',
             'benchmark_lzsh', 'benchmark_lzsh_lz1',
             'benchmark_lzsh_lz2', 'benchmark_lzsh_lz3',
             'benchmark_perf'])

        compute = tasks[0]
        self.assertEqual(compute['mode'], 'benchmark')
        self.assertFalse(compute['use_slurm'])
        self.assertEqual(compute['timeout'], planner.DEFAULT_TIMEOUT)
        self.assertTrue(os.path.isfile(compute['script']))

        # The run directory isolates the case: copied YAML plus the
        # two inputs generated at plan time.
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'fssh')
        self.assertEqual(compute['run_dir'], run_dir)
        self.assertEqual(compute['benchmark_task'], 'fssh')
        self.assertTrue(os.path.isfile(
            os.path.join(run_dir, 'fssh.yaml')))
        self.assertTrue(os.path.isfile(
            os.path.join(run_dir, 'reference.yaml')))
        self.assertTrue(os.path.isfile(
            os.path.join(run_dir, 'production.yaml')))
        # One script only: no standalone per-leg wrappers, no
        # evaluate script.
        sh_files = sorted(fname for fname in os.listdir(run_dir)
                          if fname.endswith('.sh'))
        self.assertEqual(sh_files, ['benchmark_fssh.sh'])
        self.assertTrue(compute['input_file'].endswith('reference.yaml'))
        self.assertTrue(os.path.isfile(compute['input_file']))

    def test_perf_case_plans_the_backend_legs(self):
        # The perf case implements the build_legs contract: one leg
        # per FSSH backend chained into the same single script, no
        # reference/production pair.
        # numba is optional: pin the availability flag so the perf case
        # plans its backend legs (a compute task) on hosts without the
        # optional dependency instead of degrading to a diagnostic skip.
        with mock.patch(
                'pyrai3md.workflow.integrator.fssh_numba.'
                'NUMBA_AVAILABLE', True):
            tasks = planner.plan_benchmark_tests(
                self._config(''), self.tmpdir)
        perf = [task for task in tasks if task['name'] == 'benchmark_perf']
        self.assertEqual(len(perf), 1)
        compute = perf[0]
        self.assertEqual(compute['benchmark_task'], 'perf')
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'perf')
        self.assertEqual(compute['run_dir'], run_dir)
        self.assertTrue(os.path.isfile(
            os.path.join(run_dir, 'perf.yaml')))
        for backend in ('numpy', 'numba'):
            self.assertTrue(os.path.isfile(os.path.join(
                run_dir, backend, 'production.yaml')))
        content = _read(compute['script'])
        # Both leg commands are chained through the timing wrapper.
        self.assertIn(
            'pyrai3md.qualification.benchmark.timing', content)
        self.assertIn(' && ', content)
        self.assertLess(
            content.index('numpy'), content.index('numba'))
        self.assertTrue(
            compute['input_file'].endswith(os.path.join(
                'numpy', 'production.yaml')))
        sh_files = sorted(fname for fname in os.listdir(run_dir)
                          if fname.endswith('.sh'))
        self.assertEqual(sh_files, ['benchmark_perf.sh'])

    def test_gsh_case_plans_the_reference_production_chain(self):
        # The gsh case reuses the two-leg reference && production
        # contract: one script, the GSH production input generated
        # from the unified gsh.yaml with the pmd_env method.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'gsh': {'method': 'tully1'}}), self.tmpdir)
        self.assertEqual(
            [task['name'] for task in tasks],
            ['benchmark_gsh_tully1'])
        compute = tasks[0]
        self.assertEqual(compute['benchmark_task'], 'gsh')
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'gsh_tully1')
        self.assertEqual(compute['run_dir'], run_dir)
        for fname in ('gsh_tully1.yaml', 'reference.yaml',
                      'production.yaml'):
            self.assertTrue(os.path.isfile(
                os.path.join(run_dir, fname)))
        with open(os.path.join(run_dir, 'production.yaml'),
                  'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        md = workflow['md']
        self.assertEqual(md['sfhp'], 'gsh')
        # The shipped case pins the velocity adjustment policy.
        self.assertEqual(md['adjust'], 1)
        self.assertEqual(md['reflect'], 1)
        # The schema default (activestate 0, all-state gradients)
        # stays unpinned; fssh-only keywords never reach the input.
        for key in ('activestate', 'deco', 'rk4', 'nactype',
                    'substep', 'integrate'):
            self.assertNotIn(key, md)
        content = _read(compute['script'])
        self.assertIn('reference.yaml', content)
        self.assertIn('production.yaml', content)
        self.assertLess(
            content.index('reference.yaml'),
            content.rindex('production.yaml'))
        self.assertIn(' && cd ', content)
        sh_files = sorted(fname for fname in os.listdir(run_dir)
                          if fname.endswith('.sh'))
        self.assertEqual(sh_files, ['benchmark_gsh_tully1.sh'])

    def test_gsh_entry_with_fssh_keywords_yields_diagnostic_skip(self):
        # fssh-only keywords (deco and friends) fail the GSH
        # production input guard and surface as a diagnostic skip
        # instead of being silently dropped.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'gsh': {'method': 'tully1', 'deco': 'off'}}),
            self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['skip'])
        self.assertEqual(tasks[0]['name'], 'benchmark_gsh_tully1')
        self.assertIn('input generation failed',
                      tasks[0]['skip_reason'])
        self.assertIn('unrecognized production keywords',
                      tasks[0]['skip_reason'])

    def test_compute_script_chains_reference_before_production(self):
        tasks = planner.plan_benchmark_tests(
            self._config(''), self.tmpdir)
        content = _read(tasks[0]['script'])
        # ONE script: both legs run through the pyrai3md entry
        # point; reference first, production only after '&&'.
        self.assertNotIn('pyrai3md_skill', content)
        self.assertIn('reference.yaml', content)
        self.assertIn('production.yaml', content)
        self.assertLess(
            content.index('reference.yaml'),
            content.rindex('production.yaml'))
        self.assertIn(' && cd ', content)

    def test_slurm_mode_keeps_the_local_script_and_names_a_job_group(self):
        # environment.scheduler is the round's mode: the case is one job,
        # so the planner writes no per-case sbatch. The compute chain and
        # its '&&' ordering stay in the local script the job runs.
        tasks = planner.plan_benchmark_tests(
            self._config('slurm'), self.tmpdir)
        # One task per shipped case file (fssh, ktdc, gsh, lzsh,
        # perf and the dedicated lz1/lz2/lz3 calibrations of all
        # four algorithms).
        self.assertEqual(len(tasks), 17)
        compute = tasks[0]
        self.assertFalse(compute['use_slurm'])
        self.assertNotIn('sbatch_script', compute)
        self.assertTrue(compute['job_group'].startswith('benchmark__'))
        content = _read(compute['script'])
        self.assertNotIn('pyrai3md_skill', content)
        self.assertIn('reference.yaml', content)
        self.assertIn(' && cd ', content)

    def test_disabled_flag_yields_diagnostic_skip(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': {'enabled': False, 'method': 'tully1'}}),
            self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['skip'])
        self.assertEqual(tasks[0]['name'], 'benchmark_fssh_tully1')

    def test_master_false_closes_an_enabled_case(self):
        # The three-state master: `false` is now "close everything", where it
        # used to mean "do not force". The case's own flag cannot reopen it,
        # and the skip reason says who closed it.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': {'method': 'tully1'}}, runall=False),
            self.tmpdir)
        closed = [task for task in tasks if task.get('skip')]
        self.assertTrue(closed)
        self.assertIn('disabled by _runall',
                      {task['skip_reason'] for task in closed})
        self.assertFalse([task for task in tasks if not task.get('skip')])

    def test_runall_overrides_disabled_flag(self):
        # The _runall master switch forces the configured entry on:
        # the same disabled case now expands into a compute task.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': {'enabled': False, 'method': 'tully1'}},
                runall=True),
            self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0].get('skip', False))
        self.assertEqual(tasks[0]['name'], 'benchmark_fssh_tully1')
        self.assertEqual(tasks[0]['mode'], 'benchmark')

    def test_configured_section_overrides_the_case_yaml(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': {'method': 'tully1', 'dt': 25, 'x0': -8,
                         'ngrid': 256}}),
            self.tmpdir)
        self.assertEqual(
            [task['name'] for task in tasks],
            ['benchmark_fssh_tully1'])
        # The merged case YAML in the run directory carries the
        # pmd_env parameters so the evaluation tool sees them too;
        # it is written under the case key even though the unified
        # fssh.yaml supplied the defaults.
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'fssh_tully1')
        with open(os.path.join(run_dir, 'fssh_tully1.yaml'),
                  'r', encoding='utf-8') as f:
            merged = yaml.safe_load(f)
        self.assertEqual(merged['production']['dt'], 25)
        self.assertEqual(merged['production']['x0'], -8)
        self.assertEqual(merged['reference']['ngrid'], 256)
        # The generated production input reflects the override.
        prod_yaml = os.path.join(run_dir, 'production.yaml')
        with open(prod_yaml, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertAlmostEqual(workflow['md']['size'], 25)

    def test_method_injects_into_the_unified_case(self):
        # No dedicated fssh_tully2.yaml ships next to the unified
        # case: the entry falls back to fssh.yaml and injects the
        # method into the merged production section.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': {'method': 'tully2'}}),
            self.tmpdir)
        self.assertEqual(
            [task['name'] for task in tasks],
            ['benchmark_fssh_tully2'])
        self.assertFalse(tasks[0].get('skip', False))
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'fssh_tully2')
        with open(os.path.join(run_dir, 'fssh_tully2.yaml'),
                  'r', encoding='utf-8') as f:
            merged = yaml.safe_load(f)
        self.assertEqual(merged['production']['method'], 'tully2')
        # The unified defaults survive the method injection.
        self.assertEqual(merged['production']['nstep'], 300)
        # Both generated inputs target the injected model.
        with open(os.path.join(run_dir, 'production.yaml'),
                  'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertEqual(workflow['qdm']['method'], 'tully2')
        with open(os.path.join(run_dir, 'reference.yaml'),
                  'r', encoding='utf-8') as f:
            ref = yaml.safe_load(f)
        self.assertEqual(ref['qdm']['method'], 'tully2')

    def test_list_format_plans_one_task_per_entry(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': [{'method': 'tully1', 'ninitcond': 100},
                         {'method': 'tully2'}]}),
            self.tmpdir)
        self.assertEqual(
            [task['name'] for task in tasks],
            ['benchmark_fssh_tully1', 'benchmark_fssh_tully2'])
        self.assertFalse(any(task.get('skip') for task in tasks))
        # Every entry lands in its own run directory with its own
        # merged case YAML and generated inputs.
        for case_key, method, ninitcond in (
                ('fssh_tully1', 'tully1', 100),
                ('fssh_tully2', 'tully2', 100)):
            run_dir = os.path.join(self.tmpdir, 'benchmark', case_key)
            self.assertTrue(os.path.isfile(
                os.path.join(run_dir, '%s.yaml' % case_key)))
            with open(os.path.join(run_dir, '%s.yaml' % case_key),
                      'r', encoding='utf-8') as f:
                merged = yaml.safe_load(f)
            self.assertEqual(merged['production']['method'], method)
            self.assertEqual(merged['production']['ninitcond'],
                             ninitcond)
            self.assertTrue(os.path.isfile(
                os.path.join(run_dir, 'reference.yaml')))
            self.assertTrue(os.path.isfile(
                os.path.join(run_dir, 'production.yaml')))
            self.assertTrue(os.path.isfile(os.path.join(
                run_dir, 'benchmark_%s.sh' % case_key)))

    def test_list_entry_without_method_yields_skip(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': [{'method': 'tully1'}, {'dt': 25}]}),
            self.tmpdir)
        # Skips are appended ahead of the planned tasks; assert by
        # kind rather than by position.
        planned = [task for task in tasks if not task.get('skip')]
        skips = [task for task in tasks if task.get('skip')]
        self.assertEqual([task['name'] for task in planned],
                         ['benchmark_fssh_tully1'])
        self.assertEqual([task['name'] for task in skips],
                         ['benchmark_fssh_1'])
        self.assertIn('method', skips[0]['skip_reason'])

    def test_non_dict_list_entry_yields_skip(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': [{'method': 'tully1'}, 'tully2']}),
            self.tmpdir)
        planned = [task for task in tasks if not task.get('skip')]
        skips = [task for task in tasks if task.get('skip')]
        self.assertEqual([task['name'] for task in planned],
                         ['benchmark_fssh_tully1'])
        self.assertEqual([task['name'] for task in skips],
                         ['benchmark_fssh_1'])
        self.assertIn('parameter mapping', skips[0]['skip_reason'])

    def test_duplicate_method_plans_only_the_first_entry(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': [{'method': 'tully1', 'dt': 25},
                         {'method': 'tully1', 'dt': 30}]}),
            self.tmpdir)
        planned = [task for task in tasks if not task.get('skip')]
        skips = [task for task in tasks if task.get('skip')]
        self.assertEqual([task['name'] for task in planned],
                         ['benchmark_fssh_tully1'])
        self.assertEqual([task['name'] for task in skips],
                         ['benchmark_fssh_tully1'])
        self.assertIn('more than once', skips[0]['skip_reason'])
        # The first entry's parameters win.
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'fssh_tully1')
        with open(os.path.join(run_dir, 'fssh_tully1.yaml'),
                  'r', encoding='utf-8') as f:
            merged = yaml.safe_load(f)
        self.assertEqual(merged['production']['dt'], 25)

    def test_resolve_case_yaml_prefers_the_dedicated_file(self):
        # A dedicated fssh_tully9.yaml beats the unified fssh.yaml
        # for the same method.
        tmp = tempfile.mkdtemp()
        try:
            for name in ('fssh.py', 'fssh.yaml', 'fssh_tully9.yaml'):
                with open(os.path.join(tmp, name), 'w',
                          encoding='utf-8') as f:
                    f.write('')
            self.assertEqual(
                planner._resolve_case_yaml(tmp, 'fssh', 'tully9'),
                os.path.join(tmp, 'fssh_tully9.yaml'))
            self.assertEqual(
                planner._resolve_case_yaml(tmp, 'fssh', 'tully1'),
                os.path.join(tmp, 'fssh.yaml'))
            self.assertEqual(
                planner._resolve_case_yaml(tmp, 'fsshx', 'tully1'), '')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_method_yields_diagnostic_skip(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={'fssh': {'dt': 25}}),
            self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['skip'])
        self.assertIn('method', tasks[0]['skip_reason'])

    def test_legacy_boolean_format_yields_diagnostic_skip(self):
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={'fssh': True}),
            self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['skip'])
        self.assertIn('no longer supported', tasks[0]['skip_reason'])

    def test_unrecognized_production_keyword_yields_skip(self):
        # The builder guard surfaces through the planner's existing
        # input-generation handler as a diagnostic skip naming the
        # offending key.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': {'method': 'tully1',
                         'fssh_debug_trace': True}}),
            self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['skip'])
        self.assertIn('input generation failed',
                      tasks[0]['skip_reason'])
        self.assertIn('fssh_debug_trace', tasks[0]['skip_reason'])

    def test_lz_family_entries_plan_with_the_dedicated_cases(self):
        # The lz entries resolve their dedicated case files (never
        # the unified defaults) and the overrides land in both legs.
        tasks = planner.plan_benchmark_tests(
            self._config('', benchmark={
                'fssh': [
                    {'method': 'lz1', 'ninitcond': 100,
                     'lz_a': 0.2712, 'lz_b': 0.0543,
                     'x0': -5, 'v0': 0.0498},
                    {'method': 'lz2', 'ninitcond': 100,
                     'lz_a': 0.2712, 'lz_b': 0.0543,
                     'x0': -5.4693, 'v0': 0.06833},
                ],
                'gsh': [
                    {'method': 'lz2', 'ninitcond': 100,
                     'lz_a': 0.2712, 'lz_b': 0.0543},
                    {'method': 'lz3', 'ninitcond': 100,
                     'lz_a': 0.2712, 'lz_b': 0.0543},
                ]}),
            self.tmpdir)
        self.assertEqual(
            [task['name'] for task in tasks],
            ['benchmark_fssh_lz1', 'benchmark_fssh_lz2',
             'benchmark_gsh_lz2', 'benchmark_gsh_lz3'])
        self.assertFalse(any(task.get('skip') for task in tasks))
        run_dir = os.path.join(self.tmpdir, 'benchmark', 'fssh_lz2')
        with open(os.path.join(run_dir, 'fssh_lz2.yaml'),
                  'r', encoding='utf-8') as f:
            merged = yaml.safe_load(f)
        # The dedicated case file wins over the unified defaults.
        self.assertEqual(merged['case'], 'fssh_lz2')
        self.assertEqual(merged['production']['ninitcond'], 100)
        self.assertEqual(merged['reference']['ngrid'], 8192)
        with open(os.path.join(run_dir, 'production.yaml'),
                  'r', encoding='utf-8') as f:
            prod_wf = yaml.safe_load(f)
        self.assertEqual(prod_wf['md']['ninitcond'], 100)
        self.assertEqual(prod_wf['qdm']['method'], 'lz2')
        self.assertAlmostEqual(prod_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(prod_wf['qdm']['lz_b'], 0.0543)
        with open(os.path.join(run_dir, 'reference.yaml'),
                  'r', encoding='utf-8') as f:
            ref_wf = yaml.safe_load(f)
        self.assertEqual(ref_wf['qdm']['method'], 'lz2')
        self.assertAlmostEqual(ref_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(ref_wf['qdm']['lz_b'], 0.0543)


class TestShippedPmdEnv(unittest.TestCase):
    """The shipped pmd_env.yaml carries the full calibration family
    (duplicate-key protected, the ktdc entries mirroring fssh with
    nactype ktdc) and plans through the dedicated lz case files."""

    def test_shipped_pmd_env_loads_the_full_lz_family(self):
        config = manager.load_config(SHIPPED_PMD_ENV)
        benchmark = config['benchmark']
        fssh_entries = benchmark['fssh']
        ktdc_entries = benchmark['ktdc']
        gsh_entries = benchmark['gsh']
        lzsh_entries = benchmark['lzsh']
        for entries in (fssh_entries, ktdc_entries, gsh_entries,
                        lzsh_entries):
            self.assertEqual([e['method'] for e in entries],
                             ['tully1', 'tully2', 'tully3',
                              'lz1', 'lz2', 'lz3'])
        # The unified ensemble standard: 100 trajectories on a
        # 4-process pool for every calibration entry.
        for entry in (fssh_entries + ktdc_entries + gsh_entries
                      + lzsh_entries):
            self.assertEqual(entry['ninitcond'], 100)
            self.assertEqual(entry['ncpus'], 4)
        # The ktdc entries mirror the fssh entries one for one:
        # the tully entries pin the kTDC NAC source as the single
        # difference (nactype ktdc replaces nactype nac) while the
        # lz entries defer nactype to the dedicated case files and
        # are byte-for-byte identical to the fssh lz entries.
        for fssh_entry, ktdc_entry in zip(fssh_entries, ktdc_entries):
            if ktdc_entry['method'] in ('lz1', 'lz2', 'lz3'):
                self.assertNotIn('nactype', ktdc_entry)
                self.assertEqual(ktdc_entry, fssh_entry)
            else:
                self.assertEqual(ktdc_entry['nactype'], 'ktdc')
                mirrored = {key: value for key, value
                            in fssh_entry.items()
                            if key != 'nactype'}
                self.assertEqual(
                    {key: value for key, value in ktdc_entry.items()
                     if key != 'nactype'}, mirrored)
        # The enabled flag is a deployment choice, not a structural
        # invariant of the shipped family: a site may disable any
        # calibration (e.g. to focus a run on multicenter). The
        # enabled->planned / disabled->skip behavior is verified
        # against the actual flags in
        # test_shipped_pmd_env_plans_the_lz_family, so no factory
        # enabled state is asserted here.
        for entry in (fssh_entries + ktdc_entries + gsh_entries
                      + lzsh_entries):
            self.assertIn('enabled', entry)

    def test_shipped_pmd_env_plans_the_lz_family(self):
        config = manager.load_config(SHIPPED_PMD_ENV)
        benchmark = config['benchmark']
        tmpdir = tempfile.mkdtemp()
        try:
            tasks = planner.plan_benchmark_tests({
                'environment': {'scheduler': ''},
                'slurm': {'partition': 'test'},
                'benchmark': benchmark,
            }, tmpdir)
            planned = sorted(task['name'] for task in tasks
                             if not task.get('skip'))
            skips = {task['name'] for task in tasks
                     if task.get('skip')}
            # The planner follows each entry's enabled flag: an
            # enabled calibration entry plans, a disabled one surfaces
            # as a diagnostic skip. The expectation is derived from
            # the shipped flags so the test never pins a
            # factory-enabled state -- which entries run is a
            # deployment choice, and this asserts the enabled/skip
            # mapping holds for whatever that choice is.
            expected_planned = []
            expected_skips = set()
            for task_name, section in benchmark.items():
                entries = ([section] if isinstance(section, dict)
                           else section)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    method = entry.get('method')
                    if not method:
                        continue
                    name = 'benchmark_%s_%s' % (task_name, method)
                    if entry.get('enabled', True):
                        expected_planned.append(name)
                    else:
                        expected_skips.add(name)
            self.assertEqual(planned, sorted(expected_planned))
            self.assertEqual(skips, expected_skips)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFsshTaskModule(unittest.TestCase):
    """Input builders, parsing and the verdict protocol of fssh.py."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.case = fssh_task.load_case(SHIPPED_CASE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_shipped_case(self):
        self.assertEqual(self.case['case'], 'fssh')
        # The unified case defaults calibrate tully1.
        self.assertEqual(self.case['production']['method'], 'tully1')
        self.assertEqual(self.case['production']['nstep'], 300)
        # The standard benchmark runs the 100-trajectory ensemble.
        self.assertEqual(self.case['production']['ninitcond'], 100)
        # The shipped case pins RK4 electronic propagation: the
        # rk4='0' forward Euler diverges on the wide-gap models.
        self.assertEqual(self.case['production']['rk4'], '1')
        self.assertIn('state_reference_difference', self.case['metrics'])

    def test_apply_overrides_merges_pmd_env_parameters(self):
        merged = fssh_task.apply_overrides(self.case, {
            'enabled': True, 'method': 'tully1', 'dt': 25, 'x0': -8,
            'ngrid': 256, 'substeps': 5,
        })
        # The shipped case stays untouched (deep merge).
        self.assertEqual(self.case['production']['dt'], 20)
        self.assertEqual(merged['production']['dt'], 25)
        self.assertEqual(merged['production']['x0'], -8)
        self.assertEqual(merged['reference']['ngrid'], 256)
        self.assertEqual(merged['reference']['substeps'], 5)
        self.assertNotIn('enabled', merged['production'])
        # Untouched case defaults survive the merge.
        self.assertEqual(merged['production']['nstep'], 300)
        self.assertEqual(merged['metrics'], self.case['metrics'])

    def test_build_reference_input_targets_skill_profile(self):
        run_dir = os.path.join(self.tmpdir, 'case')
        path = fssh_task.build_reference_input(self.case, run_dir)
        self.assertTrue(path.endswith('reference.yaml'))
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        # The reference goes through the regular pyrai3md entry
        # point: driver=skill + jobtype md selects the skill's
        # public {method}_reference profile through SingleShot.
        control = workflow['control']
        self.assertEqual(control['jobtype'], 'md')
        self.assertEqual(control['theory'], ['qdm'])
        self.assertEqual(control['driver'], 'skill')
        self.assertEqual(
            control['title'], 'benchmark_reference_tully1')
        qdm = workflow['qdm']
        self.assertEqual(qdm['method'], 'tully1')
        # Output prefix: reference_population.dat and friends.
        self.assertEqual(qdm['qdm_project'], 'reference')
        # Substepping lands the snapshots on the md time grid.
        self.assertAlmostEqual(qdm['ref_dt'], 20 / 10)
        self.assertEqual(qdm['ref_nsteps'], 300 * 10)
        self.assertEqual(qdm['ref_save_every'], 10)
        # Identical initial conditions as the production input.
        self.assertAlmostEqual(
            workflow['coordinates'][0][0],
            -20 * fssh_task.BOHR_TO_ANGSTROM)
        self.assertAlmostEqual(workflow['velocities'][0][0], 0.005)

    def test_build_production_input_wires_md_section(self):
        run_dir = os.path.join(self.tmpdir, 'case')
        path = fssh_task.build_production_input(self.case, run_dir)
        self.assertTrue(path.endswith('production.yaml'))
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        md = workflow['md']
        self.assertEqual(md['initcond'], '0')   # string keyword
        self.assertEqual(md['sfhp'], 'fssh')
        # The shipped calibration uses the exact qdm NAC vectors.
        self.assertEqual(md['nactype'], 'nac')
        # The decoherence correction collapses the residual
        # coherence of the never-hopping trajectories into pure
        # 0/1 final states (clean Bernoulli hop statistics).
        self.assertEqual(md['deco'], '0.1')
        # The shipped RK4 keyword reaches the md section.
        self.assertEqual(md['rk4'], '1')
        self.assertEqual(md['ninitcond'], 100)
        # The multitask pool for the ensemble comes from ncpus.
        self.assertTrue(md['multitask'])
        self.assertEqual(md['multipool'], 4)
        self.assertEqual(md['step'], 300)
        self.assertAlmostEqual(md['size'], 20)
        # The amu mass converts back to the au model mass exactly.
        self.assertAlmostEqual(
            workflow['mass'][0] * fssh_task.AMU_TO_AU, 2000.0)
        self.assertEqual(workflow['qdm']['method'], 'tully1')
        # The case runs locally, and it says so where the engine reads
        # it: execution_mode is a skill keyword. The framework
        # &environment section carries no such switch (#300).
        self.assertEqual(workflow['qdm']['execution_mode'], 'local')
        for retired in ('execution_mode', 'workflow_execution',
                        'workflow_workers'):
            self.assertNotIn(
                retired, workflow['environment'],
                "generated case still writes the retired &environment "
                "key '%s'" % retired)

    def test_build_production_input_forwards_propagation_keywords(self):
        # rk4/substep follow the deco "present-only" forwarding:
        # absent keys keep the md-schema defaults, present keys land
        # in the md section verbatim.
        run_dir = os.path.join(self.tmpdir, 'case')
        case = copy.deepcopy(self.case)
        case['production'].pop('rk4')
        path = fssh_task.build_production_input(case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertNotIn('rk4', workflow['md'])
        self.assertNotIn('substep', workflow['md'])
        case['production']['rk4'] = 1
        case['production']['substep'] = 40
        path = fssh_task.build_production_input(case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertEqual(workflow['md']['rk4'], '1')
        self.assertEqual(workflow['md']['substep'], 40)

    def test_build_production_input_forwards_the_keyword_table(self):
        # The explicit table spans the surface hopping + propagation
        # blocks of the md schema: keywords present in the merged
        # case land in the md section with the declared type instead
        # of being silently dropped.
        run_dir = os.path.join(self.tmpdir, 'case')
        case = copy.deepcopy(self.case)
        case['production'].update({
            'integrate': 1, 'maxh': 2, 'phasecheck': 0,
            'gap': 0.0001, 'adjust': 2, 'thermostat': 'nh',
        })
        path = fssh_task.build_production_input(case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        md = workflow['md']
        self.assertEqual(md['integrate'], 1)
        self.assertEqual(md['maxh'], 2)
        self.assertEqual(md['phasecheck'], 0)
        self.assertAlmostEqual(md['gap'], 0.0001)
        self.assertEqual(md['adjust'], 2)
        self.assertEqual(md['thermostat'], 'nh')

    def test_build_production_input_omits_absent_table_keywords(self):
        # Present-only forwarding: absent case keys keep the
        # md-schema defaults instead of pinning builder defaults.
        run_dir = os.path.join(self.tmpdir, 'case')
        path = fssh_task.build_production_input(self.case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        for key in ('integrate', 'maxh', 'phasecheck', 'gap',
                    'adjust', 'thermostat', 'reflect'):
            self.assertNotIn(key, workflow['md'])
        # The fssh_* kernel keywords share the present-only
        # semantics: the shipped calibration case never sets them,
        # so the generated input keeps the md-schema defaults.
        for key in ('fssh_dpdt', 'fssh_rk4', 'fssh_matb',
                    'fssh_ktdc', 'fssh_decoherence', 'fssh_profile'):
            self.assertNotIn(key, workflow['md'])

    def test_build_production_input_rejects_unrecognized_keywords(self):
        # A key outside the table + structural set must fail loudly:
        # the merged case YAML would otherwise promise a parameter
        # the execution input never carries.
        run_dir = os.path.join(self.tmpdir, 'case')
        case = copy.deepcopy(self.case)
        case['production']['fssh_debug_trace'] = True
        with self.assertRaises(ValueError) as ctx:
            fssh_task.build_production_input(case, run_dir)
        self.assertIn('fssh_debug_trace', str(ctx.exception))

    def test_parse_pop_skips_the_header_line(self):
        path = os.path.join(self.tmpdir, 'benchmark_x.md.pop')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('           time            Pop1            Pop2'
                    '            Act1            Act2\n')
            f.write('   20.6700000000     0.9000000000     0.1000000000'
                    '     1.0000000000     0.0000000000\n')
            f.write('   41.3400000000     0.8000000000     0.2000000000'
                    '     1.0000000000     0.0000000000\n')
        record = fssh_task.parse_pop(path)
        self.assertEqual(record.shape, (2, 5))

    def test_split_population_record_separates_the_blocks(self):
        record = np.array([[20.67, 0.9, 0.1, 1.0, 0.0],
                           [41.34, 0.8, 0.2, 1.0, 0.0]])
        population, active = fssh_task._split_population_record(
            record, 2)
        np.testing.assert_allclose(
            population, [[20.67, 0.9, 0.1], [41.34, 0.8, 0.2]])
        np.testing.assert_allclose(
            active, [[20.67, 1.0, 0.0], [41.34, 1.0, 0.0]])

    def test_split_population_record_keeps_legacy_records(self):
        # Legacy md.pop rows carry time + electronic populations
        # only; the active block degrades to None.
        record = np.array([[20.67, 0.9, 0.1], [41.34, 0.8, 0.2]])
        population, active = fssh_task._split_population_record(
            record, 2)
        np.testing.assert_allclose(population, record)
        self.assertIsNone(active)

    def test_split_population_record_rejects_unknown_widths(self):
        record = np.array([[20.67, 0.9, 0.1, 0.0],
                           [41.34, 0.8, 0.2, 0.0]])
        with self.assertRaises(RuntimeError):
            fssh_task._split_population_record(record, 2)

    def test_ensemble_average_averages_both_blocks(self):
        # Two discrete single-trajectory records (0/1) average to
        # the reference probability in both population blocks.
        populations = [(np.array([[20.67, 0.0, 1.0]]),
                        np.array([[20.67, 0.0, 1.0]])),
                       (np.array([[20.67, 1.0, 0.0]]),
                        np.array([[20.67, 1.0, 0.0]]))]
        ensemble, active_ensemble = fssh_task.ensemble_average(
            populations, 1)
        np.testing.assert_allclose(ensemble, [[0.5, 0.5]])
        np.testing.assert_allclose(active_ensemble, [[0.5, 0.5]])

    def test_ensemble_average_drops_absent_active_columns(self):
        populations = [(np.array([[20.67, 0.0, 1.0]]), None)]
        ensemble, active_ensemble = fssh_task.ensemble_average(
            populations, 1)
        np.testing.assert_allclose(ensemble, [[0.0, 1.0]])
        self.assertIsNone(active_ensemble)

    def test_ensemble_average_length_mismatch_raises(self):
        populations = [(np.array(
            [[20.67, 0.7, 0.3], [41.34, 0.5, 0.5]]), None)]
        with self.assertRaises(ValueError):
            fssh_task.ensemble_average(populations, 1)

    def _write_mock_case(self, nstep=3):
        case_path = os.path.join(self.tmpdir, 'fssh_mock.yaml')
        with open(case_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'fssh_mock',
                'production': {
                    'method': 'tully1', 'nstep': nstep, 'dt': 20.67,
                    'ninitcond': 100,
                },
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                },
            }, f)
        return case_path

    def test_evaluate_run_missing_outputs_reports_error(self):
        self._write_mock_case()
        report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'ERROR')
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = fssh_task.emit(report)
        self.assertEqual(exit_code, 2)
        self.assertIn('BENCHMARK_JSON: ', stdout.getvalue())

    def _write_mock_outputs(self, rows, with_active=True):
        """Write one reference record and one trajectory md.pop."""
        # Reference record: t=0 snapshot followed by the md grid.
        ref_dir = os.path.join(self.tmpdir, 'reference')
        os.makedirs(ref_dir)
        with open(os.path.join(ref_dir, 'mock_population.dat'),
                  'w', encoding='utf-8') as f:
            f.write('0.00 1.0 0.0\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        # Trajectory md.pop with the matching populations.
        prod_dir = os.path.join(self.tmpdir, 'production')
        os.makedirs(prod_dir)
        with open(os.path.join(prod_dir, 'benchmark_fssh_mock.md.pop'),
                  'w', encoding='utf-8') as f:
            if with_active:
                f.write('           time            Pop1'
                        '            Pop2            Act1'
                        '            Act2\n')
            else:
                # Legacy rows carry the electronic populations only.
                f.write('           time            Pop1'
                        '            Pop2\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                line = '%.2f %.4f %.4f' % (step * 20.67, p0, p1)
                if with_active:
                    line += ' %.1f %.1f' % (
                        1.0 if p0 > p1 else 0.0,
                        0.0 if p0 > p1 else 1.0)
                f.write(line + '\n')

    def test_evaluate_run_matching_records_pass(self):
        self._write_mock_case()
        rows = [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)]
        self._write_mock_outputs(rows)

        report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        # Unified metric set: the three final populations plus the
        # judged reference difference (0.0 here: both sides end on
        # S1 = 0.6); no internal or analytic target on tully cases.
        self.assertAlmostEqual(
            report['metrics']['final_state_population'], 0.6)
        self.assertAlmostEqual(
            report['metrics']['final_reference_population'], 0.6)
        # The single mock trajectory ends on state 1 (0.4 < 0.6).
        self.assertAlmostEqual(
            report['metrics']['final_trajectory_population'], 1.0)
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'], 0.0)
        for key in ('internal_hopping_probability',
                    'analytical_hopping_probability',
                    'state_internal_difference',
                    'state_analytical_difference'):
            self.assertNotIn(key, report['metrics'])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = fssh_task.emit(report)
        self.assertEqual(exit_code, 0)

        # The compared population tables are persisted into the run
        # directory: the electronic and active ensemble averages
        # plus the exact reference populations.
        for name in ('production.pop', 'production.act.pop',
                     'reference.pop'):
            self.assertTrue(
                os.path.isfile(os.path.join(self.tmpdir, name)),
                'missing %s' % name)
        saved = np.loadtxt(
            os.path.join(self.tmpdir, 'production.pop'))
        self.assertEqual(saved.shape, (3, 3))
        np.testing.assert_allclose(saved[:, 1:], rows, atol=1e-6)
        saved_act = np.loadtxt(
            os.path.join(self.tmpdir, 'production.act.pop'))
        np.testing.assert_allclose(
            saved_act[:, 1:], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def test_evaluate_run_accepts_legacy_pop_records(self):
        # Legacy md.pop rows without the active columns still
        # evaluate; the active metrics and table are simply absent.
        self._write_mock_case()
        rows = [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)]
        self._write_mock_outputs(rows, with_active=False)

        report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'], 0.0)
        # Legacy records carry no active columns: the hop fraction
        # and the active table are absent, the state metrics stay.
        self.assertIsNone(
            report['metrics']['final_trajectory_population'])
        self.assertFalse(os.path.isfile(os.path.join(
            self.tmpdir, 'production.act.pop')))

    def test_find_case_yaml_ignores_compute_inputs(self):
        self._write_mock_case()
        run_dir = self.tmpdir
        for name in ('reference.yaml', 'production.yaml'):
            with open(os.path.join(run_dir, name), 'w',
                      encoding='utf-8') as f:
                yaml.safe_dump({'control': {}}, f)
        case_path = fssh_task._find_case_yaml(run_dir)
        self.assertEqual(os.path.basename(case_path), 'fssh_mock.yaml')

    def _write_mock_lz_case(self, method='lz1', nstep=3):
        case_path = os.path.join(self.tmpdir, 'fssh_mock.yaml')
        with open(case_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'fssh_mock',
                'production': {
                    'method': method, 'nstep': nstep, 'dt': 20.67,
                    'ninitcond': 100,
                    'lz_a': 0.2712, 'lz_b': 0.0543,
                    'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0,
                },
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                    'state_analytical_difference': {'tolerance': 1.0},
                },
            }, f)
        return case_path

    def test_lz_analytic_probability_delegates_to_the_models(self):
        # The shared lz criterion delegates to the qdm lz model
        # selected by the production method (crossing velocity from
        # energy conservation on its lower adiabatic surface).
        from pyrai3md.skills.qdm_skill.service.models import get_model
        for method in fssh_task.LZ_METHODS:
            with self.subTest(method=method):
                prod = {'method': method, 'lz_a': 0.2712,
                        'lz_b': 0.0543, 'x0': -5.0, 'v0': 0.0498,
                        'model_mass': 2000.0}
                p_lz, v_c = fssh_task._lz_analytic_probability(prod)
                model = get_model(method, params={'lz_a': 0.2712,
                                                  'lz_b': 0.0543})
                self.assertAlmostEqual(
                    v_c, model.crossing_velocity(2000.0, -5.0, 0.0498))
                self.assertAlmostEqual(
                    p_lz, model.landau_zener_probability(v_c))
                self.assertTrue(0.0 < p_lz < 1.0)
        # The gsh and lzsh evaluators reuse the exact same shared
        # unified evaluator.
        self.assertIs(gsh_task.evaluate_unified,
                      fssh_task.evaluate_unified)
        self.assertIs(lzsh_task.evaluate_unified,
                      fssh_task.evaluate_unified)

    def test_evaluate_run_lz_case_scores_the_analytic_probability(self):
        # The lz family additionally compares the final state
        # population against the analytic Landau-Zener probability;
        # FSSH carries no kernel replay, so the internal target and
        # its difference never appear.
        self._write_mock_lz_case()
        rows = [(0.9, 0.1), (0.7, 0.3), (0.6, 0.4)]
        self._write_mock_outputs(rows)
        report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertIn('analytical_hopping_probability',
                      report['metrics'])
        self.assertNotIn('internal_hopping_probability',
                         report['metrics'])
        self.assertNotIn('state_internal_difference',
                         report['metrics'])
        p_lz, _ = fssh_task._lz_analytic_probability({
            'method': 'lz1', 'lz_a': 0.2712, 'lz_b': 0.0543,
            'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0})
        self.assertAlmostEqual(
            report['metrics']['analytical_hopping_probability'], p_lz)
        # p_state is the final S1 ensemble average (0.4 here).
        self.assertAlmostEqual(
            report['metrics']['state_analytical_difference'],
            abs(0.4 - p_lz))
        # Both legs end on the identical S1 probability.
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'], 0.0)
        self.assertIn('crossing_velocity', report['details'])
        # The compared population tables still persist.
        for name in ('production.pop', 'reference.pop'):
            self.assertTrue(os.path.isfile(
                os.path.join(self.tmpdir, name)), 'missing %s' % name)

    def test_evaluate_run_tully_case_scores_the_reference_difference(self):
        # The fssh tully path scores the wavepacket reference
        # difference only: FSSH carries no kernel replay (no
        # internal probability) and the tully models carry no
        # analytic hopping probability.
        self._write_mock_case()
        rows = [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)]
        self._write_mock_outputs(rows)
        report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(
            set(report['metrics']),
            {'final_trajectory_population', 'final_state_population',
             'final_reference_population',
             'state_reference_difference'})


class TestGshTaskModule(unittest.TestCase):
    """The gsh.py task module: GSH production input and the fssh
    reuse contract (reference builder and evaluation)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.case = gsh_task.load_case(SHIPPED_GSH_CASE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_shipped_case(self):
        self.assertEqual(self.case['case'], 'gsh')
        # The unified case defaults calibrate tully1 with the same
        # wavepacket parameters as the fssh case.
        self.assertEqual(self.case['production']['method'], 'tully1')
        self.assertEqual(self.case['production']['sfhp'], 'gsh')
        self.assertEqual(self.case['production']['ninitcond'], 100)
        # The shipped case pins the velocity adjustment policy.
        self.assertEqual(self.case['production']['adjust'], 1)
        self.assertEqual(self.case['production']['reflect'], 1)
        # GSH carries no fssh-only keywords.
        for key in ('deco', 'rk4', 'nactype', 'substep', 'integrate'):
            self.assertNotIn(key, self.case['production'])
        # The unified verdict judges the final-state difference
        # against the wavepacket reference and the deterministic ZN
        # kernel replay of the recorded trajectories.
        self.assertIn('state_reference_difference', self.case['metrics'])
        self.assertIn('state_internal_difference', self.case['metrics'])

    def test_reuses_the_fssh_infrastructure(self):
        # The algorithm-agnostic pieces resolve to the fssh module
        # objects themselves (never re-implemented copies).
        self.assertIs(gsh_task.load_case, fssh_task.load_case)
        self.assertIs(gsh_task.apply_overrides,
                      fssh_task.apply_overrides)
        self.assertIs(gsh_task.build_reference_input,
                      fssh_task.build_reference_input)

    def test_evaluate_run_routes_non_lz_to_the_tully_verdict(self):
        # Non-lz cases run the unified final-state verdict with the
        # ZN kernel replay internal probability; only the lz family
        # (lz1/lz2/lz3) adds the Landau-Zener comparison.
        # Missing outputs surface as ERROR on this path.
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'gsh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({'case': 'gsh',
                            'production': {'method': 'tully1'}}, f)
        report = gsh_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'ERROR')

    def test_evaluate_run_lz_family_stays_on_the_lz_path(self):
        # Every lz method runs the Landau-Zener branch of the
        # unified verdict (here surfacing the missing outputs as
        # ERROR).
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        for method in fssh_task.LZ_METHODS:
            with self.subTest(method=method):
                with open(os.path.join(run_dir, 'gsh_mock.yaml'),
                          'w', encoding='utf-8') as f:
                    yaml.safe_dump({
                        'case': 'gsh_%s' % method,
                        'production': {
                            'method': method, 'lz_a': 0.2712,
                            'lz_b': 0.0543, 'x0': -5.0,
                            'v0': 0.0498, 'model_mass': 2000.0,
                        },
                    }, f)
                report = gsh_task.evaluate_run(run_dir)
                self.assertEqual(report['verdict'], 'ERROR')

    def test_lz_analytic_probability_comes_from_the_qdm_model(self):
        # The Landau-Zener target is delegated to the qdm lz model
        # module selected by the method (crossing velocity from
        # energy conservation); the shipped lz2/lz3 calibrations
        # keep the middle-probability regime of lz1.
        from pyrai3md.skills.qdm_skill.service.models import get_model
        prod = {'method': 'lz1', 'lz_a': 0.2712, 'lz_b': 0.0543,
                'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0}
        p_lz, v_c = fssh_task._lz_analytic_probability(prod)
        model = get_model('lz1', params={'lz_a': 0.2712,
                                         'lz_b': 0.0543})
        self.assertAlmostEqual(
            v_c, model.crossing_velocity(2000.0, -5.0, 0.0498))
        self.assertAlmostEqual(p_lz,
                               model.landau_zener_probability(v_c))
        self.assertTrue(0.3 < p_lz < 0.4)
        # lz2/lz3 follow the half-linear exponent (slope difference
        # lz_a) on their shipped calibration parameters.
        for method, x0, v0 in (('lz2', -5.4693, 0.06833),
                               ('lz3', -4.4157, 0.07762)):
            prod = {'method': method, 'lz_a': 0.2712, 'lz_b': 0.0543,
                    'x0': x0, 'v0': v0, 'model_mass': 2000.0}
            p_lz, v_c = fssh_task._lz_analytic_probability(prod)
            model = get_model(method, params={'lz_a': 0.2712,
                                              'lz_b': 0.0543})
            self.assertAlmostEqual(
                p_lz, model.landau_zener_probability(v_c))
            self.assertAlmostEqual(
                p_lz, np.exp(-2.0 * np.pi * 0.0543 ** 2
                             / (0.2712 * v_c)))
            self.assertTrue(0.3 < p_lz < 0.4)

    def test_apply_overrides_merges_gsh_parameters(self):
        merged = gsh_task.apply_overrides(self.case, {
            'enabled': True, 'method': 'tully2', 'gap': 0.8,
            'maxh': 1, 'ngrid': 256,
        })
        self.assertEqual(self.case['production']['method'], 'tully1')
        self.assertEqual(merged['production']['method'], 'tully2')
        self.assertAlmostEqual(merged['production']['gap'], 0.8)
        self.assertEqual(merged['production']['maxh'], 1)
        self.assertEqual(merged['reference']['ngrid'], 256)
        self.assertNotIn('enabled', merged['production'])

    def test_build_production_input_wires_md_section(self):
        run_dir = os.path.join(self.tmpdir, 'case')
        path = gsh_task.build_production_input(self.case, run_dir)
        self.assertTrue(path.endswith('production.yaml'))
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertEqual(workflow['control']['title'], 'benchmark_gsh')
        self.assertEqual(workflow['control']['driver'], 'main')
        md = workflow['md']
        self.assertEqual(md['initcond'], '0')   # string keyword
        self.assertEqual(md['sfhp'], 'gsh')
        self.assertEqual(md['adjust'], 1)
        self.assertEqual(md['reflect'], 1)
        self.assertEqual(md['ninitcond'], 100)
        self.assertTrue(md['multitask'])
        self.assertEqual(md['multipool'], 4)
        self.assertEqual(md['step'], 300)
        self.assertAlmostEqual(md['size'], 20)
        # The amu mass converts back to the au model mass exactly.
        self.assertAlmostEqual(
            workflow['mass'][0] * gsh_task.AMU_TO_AU, 2000.0)
        self.assertEqual(workflow['qdm']['method'], 'tully1')
        # Present-only forwarding: the md-schema defaults stay in
        # charge of the gradient scope and the hopping thresholds.
        for key in ('activestate', 'gap', 'gapsoc', 'maxh'):
            self.assertNotIn(key, md)

    def test_build_production_input_rejects_fssh_keywords(self):
        # fssh-only keywords must fail loudly through the
        # unrecognized-production-key guard.
        run_dir = os.path.join(self.tmpdir, 'case')
        case = copy.deepcopy(self.case)
        case['production']['deco'] = 'off'
        case['production']['nactype'] = 'nac'
        with self.assertRaises(ValueError) as ctx:
            gsh_task.build_production_input(case, run_dir)
        message = str(ctx.exception)
        self.assertIn('unrecognized production keywords', message)
        self.assertIn('deco', message)
        self.assertIn('nactype', message)

    def test_build_production_input_omits_absent_model_parameters(self):
        # The Tully case carries no lz_a/lz_b: the qdm section keeps
        # the model class defaults (present-only forwarding).
        run_dir = os.path.join(self.tmpdir, 'case')
        path = gsh_task.build_production_input(self.case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        for key in ('lz_a', 'lz_b'):
            self.assertNotIn(key, workflow['qdm'])

    def test_shipped_lz1_case_loads(self):
        case = gsh_task.load_case(SHIPPED_GSH_LZ1_CASE)
        self.assertEqual(case['case'], 'gsh_lz1')
        prod = case['production']
        self.assertEqual(prod['method'], 'lz1')
        self.assertEqual(prod['sfhp'], 'gsh')
        # The Landau-Zener parameters and the widened gap threshold
        # are pinned by the dedicated case file.
        self.assertAlmostEqual(prod['lz_a'], 0.2712)
        self.assertAlmostEqual(prod['lz_b'], 0.0543)
        self.assertAlmostEqual(prod['gap'], 3.5)
        # The lz1 verdict judges the final-state difference against
        # the wavepacket reference, the deterministic kernel replay
        # and the analytic Landau-Zener probability.
        self.assertIn('state_reference_difference', case['metrics'])
        self.assertIn('state_internal_difference', case['metrics'])
        self.assertIn('state_analytical_difference', case['metrics'])
        # The standard benchmark runs the 100-trajectory ensemble.
        self.assertEqual(prod['ninitcond'], 100)

    def test_shipped_lz2_lz3_cases_load(self):
        # The half-linear family ships dedicated case files for both
        # algorithms with the identical model parametrization.
        for case_path, expected in (
                (SHIPPED_GSH_LZ2_CASE, 'gsh_lz2'),
                (SHIPPED_GSH_LZ3_CASE, 'gsh_lz3')):
            with self.subTest(case=expected):
                case = gsh_task.load_case(case_path)
                self.assertEqual(case['case'], expected)
                prod = case['production']
                self.assertEqual(prod['method'], expected.split('_')[1])
                self.assertEqual(prod['sfhp'], 'gsh')
                self.assertAlmostEqual(prod['lz_a'], 0.2712)
                self.assertAlmostEqual(prod['lz_b'], 0.0543)
                self.assertAlmostEqual(prod['gap'], 3.5)
                self.assertEqual(prod['ninitcond'], 100)
                self.assertIn('state_reference_difference',
                              case['metrics'])
                self.assertIn('state_internal_difference',
                              case['metrics'])
                self.assertIn('state_analytical_difference',
                              case['metrics'])
        for method, case_path in SHIPPED_FSSH_LZ_CASES.items():
            with self.subTest(case='fssh_%s' % method):
                case = fssh_task.load_case(case_path)
                self.assertEqual(case['case'], 'fssh_%s' % method)
                prod = case['production']
                self.assertEqual(prod['method'], method)
                self.assertEqual(prod['sfhp'], 'fssh')
                # The fssh lz pins follow the unified fssh verdicts.
                self.assertEqual(prod['deco'], '0.1')
                self.assertEqual(prod['rk4'], '1')
                self.assertEqual(prod['nactype'], 'nac')
                self.assertAlmostEqual(prod['gap'], 3.5)
                self.assertEqual(prod['ninitcond'], 100)
                # The RK4 coherence rotation is stable only for
                # gap*delt <= ~2.83 Hartree and the linear LZ gap
                # is unbounded: the lz cases pin 500 electronic
                # substeps (delt = 0.04 au) so the post-crossing
                # run-away never diverges the populations.
                self.assertEqual(prod['substep'], 500)
                # The fssh lz verdict judges the reference and the
                # analytic Landau-Zener differences (no kernel replay).
                self.assertEqual(
                    list(case['metrics']),
                    ['state_reference_difference',
                     'state_analytical_difference'])

    def test_lz1_parameters_forward_into_both_legs(self):
        case = gsh_task.load_case(SHIPPED_GSH_LZ1_CASE)
        run_dir = os.path.join(self.tmpdir, 'lz1')
        prod_path = gsh_task.build_production_input(case, run_dir)
        ref_path = gsh_task.build_reference_input(case, run_dir)
        with open(prod_path, 'r', encoding='utf-8') as f:
            prod_wf = yaml.safe_load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_wf = yaml.safe_load(f)
        # Both legs evaluate the identical parametrized model.
        self.assertEqual(prod_wf['qdm']['method'], 'lz1')
        self.assertAlmostEqual(prod_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(prod_wf['qdm']['lz_b'], 0.0543)
        self.assertEqual(ref_wf['qdm']['method'], 'lz1')
        self.assertAlmostEqual(ref_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(ref_wf['qdm']['lz_b'], 0.0543)
        # The explicitly pinned gap threshold reaches the md section.
        self.assertAlmostEqual(prod_wf['md']['gap'], 3.5)

    def test_lz2_parameters_forward_into_both_legs(self):
        # The qdm forwarding table covers the half-linear family
        # without any builder change.
        case = gsh_task.load_case(SHIPPED_GSH_LZ2_CASE)
        run_dir = os.path.join(self.tmpdir, 'lz2')
        prod_path = gsh_task.build_production_input(case, run_dir)
        ref_path = gsh_task.build_reference_input(case, run_dir)
        with open(prod_path, 'r', encoding='utf-8') as f:
            prod_wf = yaml.safe_load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_wf = yaml.safe_load(f)
        self.assertEqual(prod_wf['qdm']['method'], 'lz2')
        self.assertAlmostEqual(prod_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(prod_wf['qdm']['lz_b'], 0.0543)
        self.assertEqual(ref_wf['qdm']['method'], 'lz2')
        self.assertAlmostEqual(ref_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(ref_wf['qdm']['lz_b'], 0.0543)
        self.assertAlmostEqual(prod_wf['md']['gap'], 3.5)

    def test_evaluate_run_lz_case_scores_both_lz_metrics(self):
        # The gsh lz verdict judges the final-state difference
        # against the wavepacket reference, the deterministic ZN
        # kernel replay and the analytic Landau-Zener probability.
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'gsh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'gsh_lz1',
                'production': {
                    'method': 'lz1', 'nstep': 3, 'dt': 20.67,
                    'ninitcond': 100,
                    'lz_a': 0.2712, 'lz_b': 0.0543,
                    'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0,
                },
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                    'state_internal_difference': {'tolerance': 1.0},
                    'state_analytical_difference': {'tolerance': 1.0},
                },
            }, f)
        ref_dir = os.path.join(run_dir, 'reference')
        os.makedirs(ref_dir)
        rows = [(0.9, 0.1), (0.7, 0.3), (0.6, 0.4)]
        with open(os.path.join(ref_dir, 'mock_population.dat'),
                  'w', encoding='utf-8') as f:
            f.write('0.00 1.0 0.0\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        prod_dir = os.path.join(run_dir, 'production')
        os.makedirs(prod_dir)
        with open(os.path.join(prod_dir, 'benchmark_gsh_lz1.md.pop'),
                  'w', encoding='utf-8') as f:
            f.write('           time            Pop1'
                    '            Pop2\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        # The kernel replay is deterministic given the trajectory
        # record; the unit test fixes its value instead of building
        # the full md output files.
        with mock.patch.object(gsh_task, '_lz_kernel_probability',
                               return_value=0.37):
            report = gsh_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(
            set(report['metrics']),
            {'final_trajectory_population', 'final_state_population',
             'final_reference_population',
             'internal_hopping_probability',
             'analytical_hopping_probability',
             'state_reference_difference',
             'state_internal_difference',
             'state_analytical_difference'})
        p_lz, _ = fssh_task._lz_analytic_probability({
            'method': 'lz1', 'lz_a': 0.2712, 'lz_b': 0.0543,
            'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0})
        # p_state is the final state-1 ensemble average (0.4 here).
        self.assertAlmostEqual(
            report['metrics']['final_state_population'], 0.4)
        self.assertAlmostEqual(
            report['metrics']['final_reference_population'], 0.4)
        self.assertAlmostEqual(
            report['metrics']['internal_hopping_probability'], 0.37)
        self.assertAlmostEqual(
            report['metrics']['analytical_hopping_probability'], p_lz)
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'], 0.0)
        self.assertAlmostEqual(
            report['metrics']['state_internal_difference'],
            abs(0.4 - 0.37))
        self.assertAlmostEqual(
            report['metrics']['state_analytical_difference'],
            abs(0.4 - p_lz))
        self.assertIn('crossing_velocity', report['details'])

    def test_numerical_adiabatic_gradients_differentiates_the_record(self):
        # The Tully models carry no closed-form adiabatic gradient
        # interface: the replay differentiates the recorded state
        # energies against the recorded coordinates (central
        # differences in the interior, one-sided at both ends);
        # zero-displacement steps fall back to zero.
        x = np.asarray([0.0, 1.0, 3.0, 6.0])
        energies = np.stack(
            [np.asarray([0.0, 1.0, 3.0, 6.0]),
             np.asarray([0.0, -1.0, -3.0, -6.0])], axis=1)
        grads = gsh_task._numerical_adiabatic_gradients(x, energies)
        # Interior: central difference over the two-neighbour span.
        self.assertAlmostEqual(grads[1, 0], (3.0 - 0.0) / (3.0 - 0.0))
        self.assertAlmostEqual(grads[2, 0], (6.0 - 1.0) / (6.0 - 1.0))
        self.assertAlmostEqual(grads[2, 1], -1.0)
        # Ends: the one-sided difference of the adjacent step pair.
        self.assertAlmostEqual(grads[0, 0], (3.0 - 1.0) / (3.0 - 1.0))
        self.assertAlmostEqual(grads[3, 0], (6.0 - 3.0) / (6.0 - 3.0))
        self.assertAlmostEqual(grads[3, 1], -1.0)
        # A step whose two-neighbour span is frozen (equal
        # neighbour coordinates) falls back to zero.
        x_back = np.asarray([0.0, 1.0, 0.0, 2.0])
        grads = gsh_task._numerical_adiabatic_gradients(
            x_back, energies)
        self.assertAlmostEqual(grads[1, 0], 0.0)

    def test_tully_kernel_replay_takes_the_first_nonzero_gate(self):
        # The tully replay runs the production ZN kernel step by
        # step on the first recorded trajectory (numerically
        # differentiated gradients) and returns the first non-zero
        # gate probability.
        run_dir = os.path.join(self.tmpdir, 'case')
        traj_dir = os.path.join(run_dir, 'production',
                                'benchmark_gsh_0')
        os.makedirs(traj_dir)
        # Four recorded steps: the coordinate frames and the
        # energy columns of the md output files (the state energies
        # advance step by step so the numerical gradients of the
        # record are non-zero).
        xs = [0.0, 0.5, 1.0, 1.5]
        with open(os.path.join(traj_dir, 'benchmark_gsh.md.xyz'),
                  'w', encoding='utf-8') as f:
            for x in xs:
                f.write('1\ncomment\nX %.6f 0.000000 0.000000\n'
                        % (x * gsh_task.BOHR_TO_ANGSTROM))
        with open(os.path.join(traj_dir, 'benchmark_gsh.md.energies'),
                  'w', encoding='utf-8') as f:
            f.write('time Epot Ekin Etot Epot1 Epot2\n')
            for step, x in enumerate(xs, start=1):
                f.write('%.2f 0.0 0.060000 0.0 %.6f %.6f\n'
                        % (step * 20.0, 0.05 - 0.01 * x,
                           0.01 * x))
        # The kernel value itself is fixed by the mock: the test
        # scores the replay loop (the two three-step windows of the
        # four-step record, first non-zero probability wins).
        with mock.patch.object(
                gsh_task, 'internal_conversion_prob',
                side_effect=[(0.0, None), (0.31, None)]) as kernel:
            prob = gsh_task._tully_kernel_probability(
                run_dir, {'_case_name': 'gsh', 'gap': 3.5})
        self.assertAlmostEqual(prob, 0.31)
        self.assertEqual(kernel.call_count, 2)
        # The kernel consumes the numerically differentiated
        # gradients of the recorded energies.
        args = kernel.call_args_list[0][0]
        grad_n, grad_pp = args[5], args[6]
        self.assertEqual(grad_n.shape, (2, 1, 3))
        self.assertAlmostEqual(grad_pp[0, 0, 0], -0.01)
        self.assertAlmostEqual(grad_pp[1, 0, 0], 0.01)
        # Without a recorded trajectory the replay reports zero.
        empty_dir = os.path.join(self.tmpdir, 'empty')
        os.makedirs(os.path.join(empty_dir, 'production'))
        self.assertAlmostEqual(
            gsh_task._tully_kernel_probability(
                empty_dir, {'_case_name': 'gsh'}), 0.0)

    def test_evaluate_run_tully_case_scores_the_reference_difference(self):
        # The gsh tully verdict judges the final-state difference
        # against the wavepacket reference and the deterministic ZN
        # kernel replay (internal); no analytic hopping probability
        # exists for the tully models.
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'gsh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'gsh',
                'production': {
                    'method': 'tully1', 'nstep': 3, 'dt': 20.67,
                    'ninitcond': 100,
                },
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                    'state_internal_difference': {'tolerance': 1.0},
                },
            }, f)
        ref_dir = os.path.join(run_dir, 'reference')
        os.makedirs(ref_dir)
        rows = [(0.9, 0.1), (0.7, 0.3), (0.6, 0.4)]
        with open(os.path.join(ref_dir, 'mock_population.dat'),
                  'w', encoding='utf-8') as f:
            f.write('0.00 1.0 0.0\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        prod_dir = os.path.join(run_dir, 'production')
        os.makedirs(prod_dir)
        # Two trajectories with pure 0/1 final states: hop fraction
        # 0.5 (the Bernoulli statistic the verdict scores).
        for name, pop in (('benchmark_gsh_0.md.pop', [0.0, 0.0, 0.0]),
                          ('benchmark_gsh_1.md.pop', [0.0, 0.0, 1.0])):
            with open(os.path.join(prod_dir, name), 'w',
                      encoding='utf-8') as f:
                f.write('           time            Pop1'
                        '            Pop2            Act1'
                        '            Act2\n')
                for step, p1 in enumerate(pop, start=1):
                    f.write('%.2f %.4f %.4f %.1f %.1f\n'
                            % (step * 20.67, 1 - p1, p1,
                               1 - p1, p1))
        # The kernel replay is deterministic given the trajectory
        # record; the unit test fixes its value instead of building
        # the full md output files.
        with mock.patch.object(gsh_task, '_tully_kernel_probability',
                               return_value=0.45):
            report = gsh_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(
            set(report['metrics']),
            {'final_trajectory_population', 'final_state_population',
             'final_reference_population',
             'internal_hopping_probability',
             'state_reference_difference',
             'state_internal_difference'})
        # Hop fraction 0.5 (one of two trajectories ends on state
        # 1); the ensemble average lands at 0.5 against the 0.4
        # reference final probability.
        self.assertAlmostEqual(
            report['metrics']['final_trajectory_population'], 0.5)
        self.assertAlmostEqual(
            report['metrics']['final_state_population'], 0.5)
        self.assertAlmostEqual(
            report['metrics']['final_reference_population'], 0.4)
        self.assertAlmostEqual(
            report['metrics']['internal_hopping_probability'], 0.45)
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'],
            abs(0.5 - 0.4))
        self.assertAlmostEqual(
            report['metrics']['state_internal_difference'],
            abs(0.5 - 0.45))
        for key in ('analytical_hopping_probability',
                    'state_analytical_difference'):
            self.assertNotIn(key, report['metrics'])


class TestLzshTaskModule(unittest.TestCase):
    """The lzsh.py task module: LZSH production input and the fssh
    reuse contract (reference builder and evaluation)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.case = lzsh_task.load_case(SHIPPED_LZSH_CASE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_shipped_case(self):
        self.assertEqual(self.case['case'], 'lzsh')
        # The unified case defaults calibrate tully1 with the same
        # wavepacket parameters as the fssh case.
        self.assertEqual(self.case['production']['method'], 'tully1')
        self.assertEqual(self.case['production']['sfhp'], 'lzsh')
        self.assertEqual(self.case['production']['ninitcond'], 100)
        # The shipped case pins the velocity adjustment policy.
        self.assertEqual(self.case['production']['adjust'], 1)
        self.assertEqual(self.case['production']['reflect'], 1)
        # LZSH carries no fssh-only keywords and no activestate
        # (the kernels consume no gradients).
        for key in ('deco', 'rk4', 'nactype', 'substep', 'integrate',
                    'activestate'):
            self.assertNotIn(key, self.case['production'])
        # The unified verdict judges the final-state difference
        # against the wavepacket reference and the deterministic
        # LZSH kernel replay of the recorded trajectories.
        self.assertIn('state_reference_difference', self.case['metrics'])
        self.assertIn('state_internal_difference', self.case['metrics'])

    def test_reuses_the_fssh_infrastructure(self):
        # The algorithm-agnostic pieces resolve to the fssh module
        # objects themselves (never re-implemented copies).
        self.assertIs(lzsh_task.load_case, fssh_task.load_case)
        self.assertIs(lzsh_task.apply_overrides,
                      fssh_task.apply_overrides)
        self.assertIs(lzsh_task.build_reference_input,
                      fssh_task.build_reference_input)
        # The LZ-only keyword table drops the gradient-only
        # activestate escape hatch of the gsh table on purpose.
        self.assertNotIn('activestate',
                         lzsh_task.PRODUCTION_MD_KEYWORDS)

    def test_evaluate_run_routes_non_lz_to_the_tully_verdict(self):
        # Non-lz cases run the unified final-state verdict with the
        # LZSH kernel replay internal probability; only the lz
        # family (lz1/lz2/lz3) adds the Landau-Zener comparison.
        # Missing outputs surface as ERROR on this path.
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'lzsh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({'case': 'lzsh',
                            'production': {'method': 'tully1'}}, f)
        report = lzsh_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'ERROR')

    def test_evaluate_run_lz_family_stays_on_the_lz_path(self):
        # Every lz method runs the Landau-Zener branch of the
        # unified verdict (here surfacing the missing outputs as
        # ERROR).
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        for method in fssh_task.LZ_METHODS:
            with self.subTest(method=method):
                with open(os.path.join(run_dir, 'lzsh_mock.yaml'),
                          'w', encoding='utf-8') as f:
                    yaml.safe_dump({
                        'case': 'lzsh_%s' % method,
                        'production': {
                            'method': method, 'lz_a': 0.2712,
                            'lz_b': 0.0543, 'x0': -5.0,
                            'v0': 0.0498, 'model_mass': 2000.0,
                        },
                    }, f)
                report = lzsh_task.evaluate_run(run_dir)
                self.assertEqual(report['verdict'], 'ERROR')

    def test_apply_overrides_merges_lzsh_parameters(self):
        merged = lzsh_task.apply_overrides(self.case, {
            'enabled': True, 'method': 'tully2', 'gap': 0.8,
            'maxh': 1, 'ngrid': 256,
        })
        self.assertEqual(self.case['production']['method'], 'tully1')
        self.assertEqual(merged['production']['method'], 'tully2')
        self.assertAlmostEqual(merged['production']['gap'], 0.8)
        self.assertEqual(merged['production']['maxh'], 1)
        self.assertEqual(merged['reference']['ngrid'], 256)
        self.assertNotIn('enabled', merged['production'])

    def test_build_production_input_wires_md_section(self):
        run_dir = os.path.join(self.tmpdir, 'case')
        path = lzsh_task.build_production_input(self.case, run_dir)
        self.assertTrue(path.endswith('production.yaml'))
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertEqual(workflow['control']['title'], 'benchmark_lzsh')
        self.assertEqual(workflow['control']['driver'], 'main')
        md = workflow['md']
        self.assertEqual(md['initcond'], '0')   # string keyword
        self.assertEqual(md['sfhp'], 'lzsh')
        self.assertEqual(md['adjust'], 1)
        self.assertEqual(md['reflect'], 1)
        self.assertEqual(md['ninitcond'], 100)
        self.assertTrue(md['multitask'])
        self.assertEqual(md['multipool'], 4)
        self.assertEqual(md['step'], 300)
        self.assertAlmostEqual(md['size'], 20)
        # The amu mass converts back to the au model mass exactly.
        self.assertAlmostEqual(
            workflow['mass'][0] * lzsh_task.AMU_TO_AU, 2000.0)
        self.assertEqual(workflow['qdm']['method'], 'tully1')
        # Present-only forwarding: the md-schema defaults stay in
        # charge of the hopping thresholds; the gradient scope is
        # never touched (the kernels consume no gradients).
        for key in ('activestate', 'gap', 'gapsoc', 'maxh'):
            self.assertNotIn(key, md)

    def test_build_production_input_rejects_fssh_keywords(self):
        # fssh-only keywords must fail loudly through the
        # unrecognized-production-key guard.
        run_dir = os.path.join(self.tmpdir, 'case')
        case = copy.deepcopy(self.case)
        case['production']['deco'] = 'off'
        case['production']['nactype'] = 'nac'
        with self.assertRaises(ValueError) as ctx:
            lzsh_task.build_production_input(case, run_dir)
        message = str(ctx.exception)
        self.assertIn('unrecognized production keywords', message)
        self.assertIn('deco', message)
        self.assertIn('nactype', message)

    def test_build_production_input_rejects_activestate(self):
        # The GSH-only gradient escape hatch is not an LZSH keyword:
        # requesting it fails the guard instead of entering the
        # input silently.
        run_dir = os.path.join(self.tmpdir, 'case')
        case = copy.deepcopy(self.case)
        case['production']['activestate'] = 0
        with self.assertRaises(ValueError) as ctx:
            lzsh_task.build_production_input(case, run_dir)
        self.assertIn('activestate', str(ctx.exception))

    def test_build_production_input_omits_absent_model_parameters(self):
        # The Tully case carries no lz_a/lz_b: the qdm section keeps
        # the model class defaults (present-only forwarding).
        run_dir = os.path.join(self.tmpdir, 'case')
        path = lzsh_task.build_production_input(self.case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        for key in ('lz_a', 'lz_b'):
            self.assertNotIn(key, workflow['qdm'])

    def test_shipped_lz_cases_load(self):
        # The lzsh lz family ships dedicated case files with the
        # identical phase-aligned parametrization of the gsh lz
        # family (both algorithms run the identical problem).
        for method, case_path in SHIPPED_LZSH_LZ_CASES.items():
            with self.subTest(case='lzsh_%s' % method):
                case = lzsh_task.load_case(case_path)
                self.assertEqual(case['case'], 'lzsh_%s' % method)
                prod = case['production']
                self.assertEqual(prod['method'], method)
                self.assertEqual(prod['sfhp'], 'lzsh')
                self.assertAlmostEqual(prod['lz_a'], 0.2712)
                self.assertAlmostEqual(prod['lz_b'], 0.0543)
                self.assertAlmostEqual(prod['gap'], 3.5)
                self.assertEqual(prod['ninitcond'], 100)
                # The lzsh lz verdict judges the final-state
                # difference against the wavepacket reference, the
                # gradient-free kernel replay and the analytic
                # Landau-Zener probability.
                self.assertIn('state_reference_difference',
                              case['metrics'])
                self.assertIn('state_internal_difference',
                              case['metrics'])
                self.assertIn('state_analytical_difference',
                              case['metrics'])

    def test_lz_parameters_forward_into_both_legs(self):
        case = lzsh_task.load_case(SHIPPED_LZSH_LZ_CASES['lz1'])
        run_dir = os.path.join(self.tmpdir, 'lz1')
        prod_path = lzsh_task.build_production_input(case, run_dir)
        ref_path = lzsh_task.build_reference_input(case, run_dir)
        with open(prod_path, 'r', encoding='utf-8') as f:
            prod_wf = yaml.safe_load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_wf = yaml.safe_load(f)
        # Both legs evaluate the identical parametrized model.
        self.assertEqual(prod_wf['qdm']['method'], 'lz1')
        self.assertAlmostEqual(prod_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(prod_wf['qdm']['lz_b'], 0.0543)
        self.assertEqual(ref_wf['qdm']['method'], 'lz1')
        self.assertAlmostEqual(ref_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(ref_wf['qdm']['lz_b'], 0.0543)
        # The explicitly pinned gap threshold reaches the md
        # section; the production runs with the lzsh integrator.
        self.assertAlmostEqual(prod_wf['md']['gap'], 3.5)
        self.assertEqual(prod_wf['md']['sfhp'], 'lzsh')

    def test_lz_kernel_replay_reads_the_energy_series_only(self):
        # The gradient-free replay runs the production IC kernel on
        # the recorded three-step energy windows (the gate step
        # carries the minimum gap).
        run_dir = os.path.join(self.tmpdir, 'case')
        prod_dir = os.path.join(run_dir, 'production',
                                'benchmark_lzsh_lz1_0')
        os.makedirs(prod_dir)
        # Columns: time Epot Ekin Etot Epot1 Epot2; the middle row
        # carries the smallest gap (de = 0.001 Ha, convex window).
        rows = [(0.04, 0.04), (0.001, 0.04), (0.04, 0.04)]
        with open(os.path.join(
                prod_dir, 'benchmark_lzsh_lz1.md.energies'),
                'w', encoding='utf-8') as f:
            f.write('time Epot Ekin Etot Epot1 Epot2\n')
            for step, (gap, ekin) in enumerate(rows, start=1):
                f.write('%.2f 0.0 %.6f 0.0 0.0 %.6f\n'
                        % (step * 20.0, ekin, gap))
        prob = lzsh_task._gate_lzsh_kernel_probability(
            run_dir, {'dt': 20.0, 'gap': 3.5,
                      '_case_name': 'lzsh_lz1'})
        de = 0.001
        gddot = (0.04 - 2 * de + 0.04) / 20.0 ** 2
        gamma = de ** 2 / (4 * np.sqrt(de * gddot))
        self.assertAlmostEqual(prob, np.exp(-2 * np.pi * gamma))

    def test_evaluate_run_lz_case_scores_both_lz_metrics(self):
        # The lzsh lz verdict judges the final-state difference
        # against the wavepacket reference, the deterministic LZSH
        # kernel replay and the analytic Landau-Zener probability.
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'lzsh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'lzsh_lz1',
                'production': {
                    'method': 'lz1', 'nstep': 3, 'dt': 20.67,
                    'ninitcond': 100,
                    'lz_a': 0.2712, 'lz_b': 0.0543,
                    'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0,
                },
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                    'state_internal_difference': {'tolerance': 1.0},
                    'state_analytical_difference': {'tolerance': 1.0},
                },
            }, f)
        ref_dir = os.path.join(run_dir, 'reference')
        os.makedirs(ref_dir)
        rows = [(0.9, 0.1), (0.7, 0.3), (0.6, 0.4)]
        with open(os.path.join(ref_dir, 'mock_population.dat'),
                  'w', encoding='utf-8') as f:
            f.write('0.00 1.0 0.0\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        prod_dir = os.path.join(run_dir, 'production')
        os.makedirs(prod_dir)
        with open(os.path.join(prod_dir, 'benchmark_lzsh_lz1.md.pop'),
                  'w', encoding='utf-8') as f:
            f.write('           time            Pop1'
                    '            Pop2\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        # The kernel replay is deterministic given the trajectory
        # record; the unit test fixes its value instead of building
        # the full md output files.
        with mock.patch.object(
                lzsh_task, '_gate_lzsh_kernel_probability',
                return_value=0.377):
            report = lzsh_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(
            set(report['metrics']),
            {'final_trajectory_population', 'final_state_population',
             'final_reference_population',
             'internal_hopping_probability',
             'analytical_hopping_probability',
             'state_reference_difference',
             'state_internal_difference',
             'state_analytical_difference'})
        p_lz, _ = fssh_task._lz_analytic_probability({
            'method': 'lz1', 'lz_a': 0.2712, 'lz_b': 0.0543,
            'x0': -5.0, 'v0': 0.0498, 'model_mass': 2000.0})
        # p_state is the final state-1 ensemble average (0.4 here).
        self.assertAlmostEqual(
            report['metrics']['final_state_population'], 0.4)
        self.assertAlmostEqual(
            report['metrics']['final_reference_population'], 0.4)
        self.assertAlmostEqual(
            report['metrics']['internal_hopping_probability'], 0.377)
        self.assertAlmostEqual(
            report['metrics']['analytical_hopping_probability'], p_lz)
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'], 0.0)
        self.assertAlmostEqual(
            report['metrics']['state_internal_difference'],
            abs(0.4 - 0.377))
        self.assertAlmostEqual(
            report['metrics']['state_analytical_difference'],
            abs(0.4 - p_lz))
        self.assertIn('crossing_velocity', report['details'])

    def test_max_lzsh_kernel_probability_takes_the_replay_maximum(self):
        # The tully surfaces carry no single-passage crossing of a
        # fixed gate: the reported internal probability is the
        # maximum per-step probability of the replay window (an
        # empty window reports zero).
        with mock.patch.object(
                lzsh_task, '_replay_lzsh_kernel',
                return_value=np.asarray([0.0, 0.377, 0.10])):
            prob = lzsh_task._max_lzsh_kernel_probability('rd', {})
        self.assertAlmostEqual(prob, 0.377)
        with mock.patch.object(lzsh_task, '_replay_lzsh_kernel',
                               return_value=np.asarray([])):
            prob = lzsh_task._max_lzsh_kernel_probability('rd', {})
        self.assertAlmostEqual(prob, 0.0)

    def test_evaluate_run_tully_case_scores_the_reference_difference(self):
        # The lzsh tully verdict judges the final-state difference
        # against the wavepacket reference and the deterministic
        # LZSH kernel replay (internal); no analytic hopping
        # probability exists for the tully models.
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'lzsh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'lzsh',
                'production': {
                    'method': 'tully1', 'nstep': 3, 'dt': 20.67,
                    'ninitcond': 100,
                },
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                    'state_internal_difference': {'tolerance': 1.0},
                },
            }, f)
        ref_dir = os.path.join(run_dir, 'reference')
        os.makedirs(ref_dir)
        rows = [(0.9, 0.1), (0.7, 0.3), (0.6, 0.4)]
        with open(os.path.join(ref_dir, 'mock_population.dat'),
                  'w', encoding='utf-8') as f:
            f.write('0.00 1.0 0.0\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        prod_dir = os.path.join(run_dir, 'production')
        os.makedirs(prod_dir)
        # Two trajectories with pure 0/1 final states: hop fraction
        # 0.5 (the Bernoulli statistic the verdict scores).
        for name, pop in (('benchmark_lzsh_0.md.pop',
                           [0.0, 0.0, 0.0]),
                          ('benchmark_lzsh_1.md.pop',
                           [0.0, 0.0, 1.0])):
            with open(os.path.join(prod_dir, name), 'w',
                      encoding='utf-8') as f:
                f.write('           time            Pop1'
                        '            Pop2            Act1'
                        '            Act2\n')
                for step, p1 in enumerate(pop, start=1):
                    f.write('%.2f %.4f %.4f %.1f %.1f\n'
                            % (step * 20.67, 1 - p1, p1,
                               1 - p1, p1))
        # The kernel replay is deterministic given the trajectory
        # record; the unit test fixes its value instead of building
        # the full md output files.
        with mock.patch.object(
                lzsh_task, '_max_lzsh_kernel_probability',
                return_value=0.377):
            report = lzsh_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(
            set(report['metrics']),
            {'final_trajectory_population', 'final_state_population',
             'final_reference_population',
             'internal_hopping_probability',
             'state_reference_difference',
             'state_internal_difference'})
        # Hop fraction 0.5 (one of two trajectories ends on state
        # 1); the ensemble average lands at 0.5 against the 0.4
        # reference final probability.
        self.assertAlmostEqual(
            report['metrics']['final_trajectory_population'], 0.5)
        self.assertAlmostEqual(
            report['metrics']['final_state_population'], 0.5)
        self.assertAlmostEqual(
            report['metrics']['final_reference_population'], 0.4)
        self.assertAlmostEqual(
            report['metrics']['internal_hopping_probability'], 0.377)
        self.assertAlmostEqual(
            report['metrics']['state_reference_difference'],
            abs(0.5 - 0.4))
        self.assertAlmostEqual(
            report['metrics']['state_internal_difference'],
            abs(0.5 - 0.377))
        for key in ('analytical_hopping_probability',
                    'state_analytical_difference'):
            self.assertNotIn(key, report['metrics'])


class TestKtdcTaskModule(unittest.TestCase):
    """The ktdc.py task module: the kTDC calibration reuses the fssh
    task module verbatim and ships the mirrored ktdc cases pinning
    nactype ktdc as the single difference."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.case = ktdc_task.load_case(SHIPPED_KTDC_CASE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_shipped_case(self):
        self.assertEqual(self.case['case'], 'ktdc')
        # The unified case defaults calibrate tully1 with the
        # identical fssh wavepacket setup.
        self.assertEqual(self.case['production']['method'], 'tully1')
        self.assertEqual(self.case['production']['sfhp'], 'fssh')
        self.assertEqual(self.case['production']['rk4'], '1')
        self.assertEqual(self.case['production']['ninitcond'], 100)
        # The single calibration difference: the kTDC energy-only
        # curvature coupling replaces the exact qdm gradients.
        self.assertEqual(self.case['production']['nactype'], 'ktdc')
        self.assertIn('state_reference_difference', self.case['metrics'])

    def test_reuses_the_fssh_infrastructure(self):
        # Every entry point resolves to the fssh module object
        # itself (never a re-implemented copy).
        for name in ('load_case', 'apply_overrides',
                     'build_reference_input', 'build_production_input',
                     'collect_populations', 'ensemble_average',
                     'parse_reference_population', 'evaluate_unified',
                     'evaluate_run', 'emit', '_save_population',
                     '_lz_analytic_probability'):
            with self.subTest(name=name):
                self.assertIs(getattr(ktdc_task, name),
                              getattr(fssh_task, name))
        self.assertIs(ktdc_task.PRODUCTION_MD_KEYWORDS,
                      fssh_task.PRODUCTION_MD_KEYWORDS)
        self.assertIs(ktdc_task.PRODUCTION_QDM_KEYWORDS,
                      fssh_task.PRODUCTION_QDM_KEYWORDS)
        self.assertIs(ktdc_task.LZ_METHODS, fssh_task.LZ_METHODS)

    def test_build_production_input_pins_ktdc(self):
        # The production md input is the fssh setup carrying the
        # ktdc NAC source (title follows the benchmark_<case>
        # convention shared with the plot helper).
        run_dir = os.path.join(self.tmpdir, 'case')
        path = ktdc_task.build_production_input(self.case, run_dir)
        with open(path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        self.assertEqual(
            workflow['control']['title'], 'benchmark_ktdc')
        md = workflow['md']
        self.assertEqual(md['sfhp'], 'fssh')
        self.assertEqual(md['nactype'], 'ktdc')
        self.assertEqual(md['initcond'], '0')   # string keyword
        self.assertEqual(md['ninitcond'], 100)
        self.assertEqual(md['multipool'], 4)
        self.assertEqual(workflow['qdm']['method'], 'tully1')

    def test_shipped_lz_cases_load(self):
        # The ktdc lz family mirrors the fssh lz family with the
        # identical phase-aligned parametrization.
        for method, case_path in SHIPPED_KTDC_LZ_CASES.items():
            with self.subTest(case='ktdc_%s' % method):
                case = ktdc_task.load_case(case_path)
                self.assertEqual(case['case'], 'ktdc_%s' % method)
                prod = case['production']
                self.assertEqual(prod['method'], method)
                self.assertEqual(prod['sfhp'], 'fssh')
                self.assertEqual(prod['nactype'], 'ktdc')
                self.assertAlmostEqual(prod['lz_a'], 0.2712)
                self.assertAlmostEqual(prod['lz_b'], 0.0543)
                self.assertAlmostEqual(prod['gap'], 3.5)
                self.assertEqual(prod['ninitcond'], 100)
                # Same RK4 stability pin as the fssh lz family
                # (the ktdc cases mirror it entry for entry).
                self.assertEqual(prod['substep'], 500)
                # The lz verdict judges the reference and the
                # analytic Landau-Zener differences: the kernel
                # replays are gsh/lzsh specific.
                self.assertIn('state_reference_difference',
                              case['metrics'])
                self.assertIn('state_analytical_difference',
                              case['metrics'])

    def test_lz_parameters_forward_into_both_legs(self):
        case = ktdc_task.load_case(SHIPPED_KTDC_LZ_CASES['lz1'])
        run_dir = os.path.join(self.tmpdir, 'lz1')
        prod_path = ktdc_task.build_production_input(case, run_dir)
        ref_path = ktdc_task.build_reference_input(case, run_dir)
        with open(prod_path, 'r', encoding='utf-8') as f:
            prod_wf = yaml.safe_load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_wf = yaml.safe_load(f)
        # Both legs evaluate the identical parametrized model; the
        # production runs fssh with the ktdc NAC source.
        self.assertEqual(prod_wf['md']['sfhp'], 'fssh')
        self.assertEqual(prod_wf['md']['nactype'], 'ktdc')
        # The RK4 stability pin reaches the production md section.
        self.assertEqual(prod_wf['md']['substep'], 500)
        self.assertEqual(prod_wf['qdm']['method'], 'lz1')
        self.assertAlmostEqual(prod_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(prod_wf['qdm']['lz_b'], 0.0543)
        self.assertEqual(ref_wf['qdm']['method'], 'lz1')
        self.assertAlmostEqual(ref_wf['qdm']['lz_a'], 0.2712)
        self.assertAlmostEqual(ref_wf['qdm']['lz_b'], 0.0543)

    def test_evaluate_run_is_the_fssh_evaluator(self):
        # The ktdc verdict protocol is the fssh one: a run
        # directory without outputs surfaces the identical ERROR
        # report (and the shared evaluator renders the S0 plot).
        run_dir = os.path.join(self.tmpdir, 'case')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'ktdc_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({'case': 'ktdc',
                            'production': {'method': 'tully1'}}, f)
        report = ktdc_task.evaluate_run(run_dir)
        self.assertEqual(report['verdict'], 'ERROR')


class TestPopulationPlotting(unittest.TestCase):
    """plotting.plot_population_s0: the S0 overlay of the compared
    population tables (pinned colors, '<method>_<model>' case-key
    title, fixed [0, 1] probability axis, degraded-table tolerance,
    never raises)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_plot_title_follows_the_case_key_convention(self):
        # The figure title is '<method>_<model>': the sfhp
        # algorithm plus the qdm model (fssh_lz1, lzsh_tully1,
        # ...); the ktdc coupling variant reports its nactype
        # instead of the shared fssh sfhp.
        self.assertEqual(plotting_mod.plot_title(
            {'sfhp': 'fssh', 'method': 'lz1'}), 'fssh_lz1')
        self.assertEqual(plotting_mod.plot_title(
            {'sfhp': 'lzsh', 'method': 'tully1'}), 'lzsh_tully1')
        self.assertEqual(plotting_mod.plot_title(
            {'sfhp': 'gsh', 'method': 'tully3'}), 'gsh_tully3')
        self.assertEqual(plotting_mod.plot_title(
            {'sfhp': 'fssh', 'nactype': 'ktdc',
             'method': 'lz1'}), 'ktdc_lz1')
        # Legacy production sections without sfhp default to fssh.
        self.assertEqual(plotting_mod.plot_title(
            {'method': 'tully1'}), 'fssh_tully1')

    def _write_tables(self, names=('production.act.pop',
                                   'production.pop',
                                   'reference.pop')):
        rows = [(0.0, 1.0), (20.67, 0.7), (41.34, 0.4)]
        for fname in names:
            with open(os.path.join(self.tmpdir, fname), 'w',
                      encoding='utf-8') as f:
                for time, p0 in rows:
                    f.write('%.2f %.4f %.4f\n'
                            % (time, p0, 1.0 - p0))

    def test_full_tables_plot_with_the_pinned_colors_and_title(self):
        self._write_tables()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        axis = mock.MagicMock()
        figure = mock.MagicMock()
        with mock.patch.object(plt, 'subplots',
                               return_value=(figure, axis)):
            result = plotting_mod.plot_population_s0(
                self.tmpdir, 'fssh_tully1')
        self.assertIn('path', result)
        self.assertTrue(result['path'].endswith('population_s0.png'))
        # The three curves land in the pinned order and colors.
        colors = [call.kwargs['color']
                  for call in axis.plot.call_args_list]
        self.assertEqual(colors,
                         ['goldenrod', 'royalblue', 'crimson'])
        labels = [call.kwargs['label']
                  for call in axis.plot.call_args_list]
        self.assertEqual(labels, ['production active',
                                  'production ensemble', 'reference'])
        # The figure title is the '<method>_<model>' case key.
        axis.set_title.assert_called_once_with('fssh_tully1')
        axis.set_xlabel.assert_called_once_with('time (au)')
        axis.set_ylabel.assert_called_once_with('S0 population')
        # Fixed probability axis: the y range stays [0, 1] with
        # ticks every 0.2 so the figures of different
        # calibrations are directly comparable.
        axis.set_ylim.assert_called_once_with(0.0, 1.0)
        ticks = axis.set_yticks.call_args[0][0]
        self.assertTrue(np.allclose(
            ticks, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        axis.legend.assert_called_once()
        figure.savefig.assert_called_once()

    def test_missing_active_table_still_plots_the_two_legs(self):
        # A degraded table set degrades the overlay, never the plot.
        self._write_tables(names=('production.pop', 'reference.pop'))
        result = plotting_mod.plot_population_s0(
            self.tmpdir, 'ktdc_tully1')
        self.assertIn('path', result)
        self.assertTrue(os.path.isfile(result['path']))

    def test_empty_directory_returns_skipped(self):
        result = plotting_mod.plot_population_s0(
            self.tmpdir, 'fssh_tully1')
        self.assertIn('skipped', result)
        self.assertIn('no population tables', result['skipped'])

    def test_matplotlib_unavailable_returns_skipped(self):
        self._write_tables()
        with mock.patch.dict(
                'sys.modules', {'matplotlib': None,
                                'matplotlib.pyplot': None}):
            result = plotting_mod.plot_population_s0(
                self.tmpdir, 'fssh_tully1')
        self.assertIn('skipped', result)
        self.assertIn('matplotlib unavailable', result['skipped'])

    def _write_mock_fssh_run(self):
        """One mock fssh run directory: case YAML, one reference
        record and one trajectory md.pop (matching populations)."""
        with open(os.path.join(self.tmpdir, 'fssh_mock.yaml'), 'w',
                  encoding='utf-8') as f:
            yaml.safe_dump({
                'case': 'fssh_mock',
                'production': {'method': 'tully1', 'nstep': 3,
                               'dt': 20.67, 'ninitcond': 100},
                'metrics': {
                    'state_reference_difference': {'tolerance': 1.0},
                },
            }, f)
        rows = [(0.9, 0.1), (0.6, 0.4), (0.4, 0.6)]
        ref_dir = os.path.join(self.tmpdir, 'reference')
        os.makedirs(ref_dir)
        with open(os.path.join(ref_dir, 'mock_population.dat'),
                  'w', encoding='utf-8') as f:
            f.write('0.00 1.0 0.0\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))
        prod_dir = os.path.join(self.tmpdir, 'production')
        os.makedirs(prod_dir)
        with open(os.path.join(
                prod_dir, 'benchmark_fssh_mock.md.pop'),
                'w', encoding='utf-8') as f:
            f.write('           time            Pop1'
                    '            Pop2\n')
            for step, (p0, p1) in enumerate(rows, start=1):
                f.write('%.2f %.4f %.4f\n' % (step * 20.67, p0, p1))

    def test_evaluate_run_records_the_plot_in_the_details(self):
        # The evaluator renders the plot after persisting the
        # population tables and records the outcome in the report
        # details (the ktdc/gsh/lzsh evaluators share the hook).
        self._write_mock_fssh_run()
        report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        plot_info = report['details']['population_plot']
        # matplotlib is available in the test environment: the png
        # lands next to the persisted population tables.
        self.assertIn('path', plot_info)
        self.assertTrue(os.path.isfile(plot_info['path']))

    def test_plot_skip_never_flips_the_verdict(self):
        # A skipped plot (no matplotlib, unreadable tables) is
        # diagnostic only: the verdict stays PASS.
        self._write_mock_fssh_run()
        with mock.patch.object(
                fssh_task, 'plot_population_s0',
                return_value={'skipped':
                              'matplotlib unavailable: no module'}):
            report = fssh_task.evaluate_run(self.tmpdir)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertIn('skipped',
                      report['details']['population_plot'])


class TestToolsWrapper(unittest.TestCase):
    """tools/benchmark.py re-exports the fssh entry points and owns
    the command-line surface of the programmatic task module."""

    def _load_external_tool(self):
        import importlib.util
        import pyrai3md
        # The tool lives outside the package: next to the source
        # tree / deployed checkout root. An installed-only
        # environment (tests imported from site-packages, no
        # checkout around) genuinely lacks it, so the tests skip
        # there instead of failing.
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(pyrai3md.__file__)))
        candidates = [
            os.path.join(project_root, 'tools', 'benchmark.py'),
            os.path.join(os.getcwd(), 'tools', 'benchmark.py'),
        ]
        for tool_path in candidates:
            if os.path.isfile(tool_path):
                break
        else:
            self.skipTest(
                'external tool tools/benchmark.py is not available '
                'outside a PyRAI3MD checkout (tried: %s)'
                % ', '.join(candidates))
        spec = importlib.util.spec_from_file_location(
            'external_benchmark_tool', tool_path)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        return tool

    def test_wrapper_reexports_evaluate_run(self):
        tool = self._load_external_tool()
        # The wrapper's entry points resolve to the fssh task
        # module; compare by module path rather than object
        # identity: the tool inserts the checkout root into
        # sys.path and may import a different pyrai3md copy than
        # the test process (installed-only test runs).
        task_path = 'pyrai3md.qualification.benchmark.fssh'
        self.assertEqual(tool.evaluate_run.__module__, task_path)
        self.assertEqual(tool.emit.__module__, task_path)
        # The CLI wiring is no longer part of the task module.
        self.assertFalse(hasattr(fssh_task, 'main'))

    def test_main_prints_usage_without_arguments(self):
        tool = self._load_external_tool()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = tool.main([])
        self.assertEqual(exit_code, 2)
        self.assertIn('usage: python3 ./tools/benchmark.py',
                      stdout.getvalue())

    def test_main_delegates_to_evaluate_run(self):
        tool = self._load_external_tool()
        stdout = io.StringIO()
        with mock.patch.object(tool, 'evaluate_run',
                               return_value={'verdict': 'PASS'}) as mocked:
            with contextlib.redirect_stdout(stdout):
                exit_code = tool.main(['some_run_dir'])
        mocked.assert_called_once_with('some_run_dir')
        self.assertEqual(exit_code, 0)
        self.assertIn('BENCHMARK_JSON: ', stdout.getvalue())

    def test_route_evaluate_dispatches_gsh_cases(self):
        # A run directory whose case YAML names a gsh case routes
        # to the gsh task module, not the historical fssh default.
        tool = self._load_external_tool()
        run_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(run_dir, 'gsh_tully1.yaml'),
                      'w', encoding='utf-8') as handler:
                yaml.safe_dump(
                    {'case': 'gsh_tully1',
                     'production': {'nstep': 300, 'dt': 20}},
                    handler)
            with mock.patch.object(
                    tool._gsh_module, 'evaluate_run',
                    return_value={'verdict': 'PASS'}) as gsh_mock, \
                 mock.patch.object(
                    tool._fssh_module, 'evaluate_run') as fssh_mock:
                report = tool._route_evaluate(run_dir)
            gsh_mock.assert_called_once_with(run_dir)
            fssh_mock.assert_not_called()
            self.assertEqual(report['verdict'], 'PASS')
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_route_evaluate_dispatches_lzsh_cases(self):
        # A run directory whose case YAML names an lzsh case routes
        # to the lzsh task module, not the historical fssh default.
        tool = self._load_external_tool()
        run_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(run_dir, 'lzsh_lz1.yaml'),
                      'w', encoding='utf-8') as handler:
                yaml.safe_dump(
                    {'case': 'lzsh_lz1',
                     'production': {'nstep': 45, 'dt': 20}},
                    handler)
            with mock.patch.object(
                    tool._lzsh_module, 'evaluate_run',
                    return_value={'verdict': 'PASS'}) as lzsh_mock, \
                 mock.patch.object(
                    tool._fssh_module, 'evaluate_run') as fssh_mock:
                report = tool._route_evaluate(run_dir)
            lzsh_mock.assert_called_once_with(run_dir)
            fssh_mock.assert_not_called()
            self.assertEqual(report['verdict'], 'PASS')
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_route_evaluate_dispatches_ktdc_cases(self):
        # A run directory whose case YAML names a ktdc case routes
        # to the ktdc task module, not the historical fssh default.
        tool = self._load_external_tool()
        run_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(run_dir, 'ktdc_tully1.yaml'),
                      'w', encoding='utf-8') as handler:
                yaml.safe_dump(
                    {'case': 'ktdc_tully1',
                     'production': {'nstep': 300, 'dt': 20}},
                    handler)
            with mock.patch.object(
                    tool._ktdc_module, 'evaluate_run',
                    return_value={'verdict': 'PASS'}) as ktdc_mock, \
                 mock.patch.object(
                    tool._fssh_module, 'evaluate_run') as fssh_mock:
                report = tool._route_evaluate(run_dir)
            ktdc_mock.assert_called_once_with(run_dir)
            fssh_mock.assert_not_called()
            self.assertEqual(report['verdict'], 'PASS')
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


class _Dynamics:
    def __init__(self, step, size, sfhp):
        self.step = step
        self.size = size
        self.sfhp = sfhp


class _Pmd:
    def __init__(self, dynamics):
        self.dynamics = dynamics


class _State:
    def __init__(self, pmd=None):
        self.pmd = pmd


class TestWorkflowConfigKwargs(unittest.TestCase):
    """&md step/size/sfhp wiring into the AIMD constructor."""

    def test_aimd_receives_md_parameters(self):
        state = _State(_Pmd(_Dynamics(200, 20.67, 'fssh')))
        kwargs = AIMD.state_kwargs(state)
        self.assertEqual(kwargs['maxiter'], 200)
        self.assertAlmostEqual(kwargs['dt'], 20.67)
        self.assertEqual(kwargs['sfhp'], 'fssh')

    def test_non_aimd_workflow_gets_nothing(self):
        state = _State(_Pmd(_Dynamics(200, 20.67, 'fssh')))
        self.assertEqual(Optimize.state_kwargs(state), {})

    def test_state_without_dynamics_gets_nothing(self):
        self.assertEqual(AIMD.state_kwargs(_State()), {})
        self.assertEqual(AIMD.state_kwargs(_State(_Pmd(None))),
                         {})


class TestRunnerBenchmarkVerdict(unittest.TestCase):
    """runner._run_benchmark_test surfaces the tool verdict cause."""

    def _run(self, report):
        from pyrai3md.qualification import runner
        task = {'name': 'benchmark_fssh_tully1',
                'run_dir': '/nonexistent/run',
                'benchmark_task': 'fssh'}
        with mock.patch.object(runner, '_run_skill_test',
                               return_value={'name': task['name'],
                                             'status': 'pass',
                                             'details': 'ok'}), \
             mock.patch.object(fssh_task, 'evaluate_run',
                               return_value=report):
            return runner._run_benchmark_test(task, {}, {})

    def test_error_verdict_prints_the_reported_cause(self):
        # The ERROR verdict carries empty metrics; the report's
        # details.error must replace the bare {} in the output.
        report = {'verdict': 'ERROR', 'metrics': {},
                  'details': {'error': 'trajectory records 211 steps '
                                       'but the reference covers 300'}}
        result = self._run(report)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['benchmark_verdict'], 'ERROR')
        self.assertIn('[BENCHMARK ERROR]', result['details'])
        self.assertIn('trajectory records 211 steps', result['details'])
        self.assertNotIn('{}', result['details'])

    def test_error_verdict_without_details_falls_back(self):
        report = {'verdict': 'ERROR', 'metrics': {}, 'details': {}}
        result = self._run(report)
        self.assertIn('[BENCHMARK ERROR] (no details)',
                      result['details'])

    def test_fail_verdict_prints_the_metric_payload(self):
        report = {'verdict': 'FAIL',
                  'metrics': {'state_reference_difference': 0.75},
                  'details': {}}
        result = self._run(report)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('[BENCHMARK FAIL]', result['details'])
        self.assertIn('state_reference_difference', result['details'])

    def test_metric_display_separates_judged_from_reported(self):
        # Judged differences print their tolerance and per-metric
        # verdict; reported quantities (populations and the
        # probability targets, no tolerance entry) print the bare
        # value without any tolerance/verdict decoration.
        from pyrai3md.qualification import runner
        result = {
            'name': 'benchmark_gsh_lz1', 'status': 'pass',
            'mode': 'benchmark',
            'metrics': {
                'metrics': {
                    'final_state_population': 0.4,
                    'internal_hopping_probability': 0.361638,
                    'state_reference_difference': 0.1,
                    'state_internal_difference': 0.038362,
                },
                'tolerances': {
                    'state_reference_difference': 1.0,
                    'state_internal_difference': 1.0,
                },
                'runtime_seconds': 12,
            },
        }
        with contextlib.redirect_stdout(io.StringIO()) as out:
            runner._print_single_result(result, 1, 1)
        lines = out.getvalue().splitlines()
        by_name = {}
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('final_state_population:'):
                by_name['final'] = stripped
            elif stripped.startswith(
                    'internal_hopping_probability:'):
                by_name['internal'] = stripped
            elif stripped.startswith('state_reference_difference:'):
                by_name['ref'] = stripped
            elif stripped.startswith('state_internal_difference:'):
                by_name['internal_diff'] = stripped
        # Reported quantities: bare value only.
        self.assertEqual(by_name['final'],
                         'final_state_population: 0.4')
        self.assertEqual(by_name['internal'],
                         'internal_hopping_probability: 0.361638')
        # Judged differences: tolerance and verdict appended.
        self.assertEqual(
            by_name['ref'],
            'state_reference_difference: 0.1 (tolerance 1) -> PASS')
        self.assertEqual(
            by_name['internal_diff'],
            'state_internal_difference: 0.038362 '
            '(tolerance 1) -> PASS')
        self.assertIn('runtime: 12 s', out.getvalue())


class TestShippedBenchmarkInputs(unittest.TestCase):
    """Every shipped calibration input survives the real load pipeline.

    A calibration only reaches an engine after the generated YAML goes
    through parse -> build -> normalize -> validate -> build_state. The
    planner tests checked the generated text; none of them fed it to that
    chain, so a molecule section the schema refuses (the ci/spin
    length-match rule) stayed invisible locally and surfaced on the
    test environment, where all 25 benchmark tasks died at load time with
    'runtime: 0 s' before a single trajectory ran.
    """

    MODULES = ((fssh_task, 'fssh'), (gsh_task, 'gsh'),
               (ktdc_task, 'ktdc'), (lzsh_task, 'lzsh'))

    def _load(self, path):
        raw = parse_framework_input(path)
        config = normalize_framework(build_framework_config(raw))
        return build_framework_state(validate_framework(config),
                                     raw_config=raw), config

    def test_every_shipped_leg_survives_the_load_chain(self):
        legs = 0
        for module, name in self.MODULES:
            case = module.load_case(os.path.join(BENCHMARK_DIR,
                                                 f'{name}.yaml'))
            with tempfile.TemporaryDirectory() as run_dir:
                paths = (module.build_production_input(case, run_dir),
                         module.build_reference_input(case, run_dir))
                for path in paths:
                    _state, config = self._load(path)
                    legs += 1
                    # Derived, not pinned: the two Tully states of one
                    # singlet block must still come out of ci/spin.
                    self.assertEqual(len(config.state_spin), config.nstate,
                                     f'{name}: {os.path.basename(path)}')
        self.assertEqual(legs, 8)

    def test_reference_input_keeps_the_record_the_verdict_reads_back(self):
        # The evaluator parses reference/*_population.dat after the
        # reference call, and the qdm backend removes its calculation
        # directory per call when keep_tmp is 0. A reference input that
        # does not ask to keep its directory therefore completes without
        # error and leaves the verdict nothing to read.
        for module, name in self.MODULES:
            case = module.load_case(os.path.join(BENCHMARK_DIR,
                                                 f'{name}.yaml'))
            with tempfile.TemporaryDirectory() as run_dir:
                path = module.build_reference_input(case, run_dir)
                workflow = yaml.safe_load(open(path))
                self.assertEqual(workflow['qdm'].get('keep_tmp'), 1,
                                 f'{name}: reference input drops its own '
                                 f'population record')


if __name__ == '__main__':
    unittest.main()
