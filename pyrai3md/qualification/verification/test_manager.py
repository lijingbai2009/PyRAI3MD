"""
======================================================================
Test suite for the qualification manager example switches
(pyrai3md/qualification/manager.py).

Tests cover:
- _coerce_switch / _coerce_flag / _resolve_chain: the three-valued switches
  (true, false, condition) and the two-valued leaves, so a typo in
  pmd_env.yaml (e.g. 'fasle') can never silently keep a test enabled and
  `condition` can never appear where there is nothing below it to defer to
- _runall_mode: the top-level master switch (a missing key means
  'condition', which is what the bare `false` used to mean) and its override
  of the family and per-example switches in _plan_example_tests, while the
  POSTPONED_EXAMPLES policy stays untouched
- the example family switches (md_lammps_test, constraint_md_test,
  optimize_test, path_test): a family `true` opens its cases whatever they say, `false`
  closes them,
  `condition` reads them one by one, and a skipped task names the switch
  that closed it
- load_config: duplicate mapping keys fail loudly instead of
  silently collapsing to the last value (the benchmark.fssh
  multi-method defect class)
- load_config: the source-tree layer ./pyrai3md/qualification/
  pmd_env.yaml resolved from the working directory overrides the
  installed package config, keeps the empty-value fallback and the
  explicit-path chain unchanged, and raises on parse errors

Aug 18 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import json
import shutil
import tempfile
import unittest
import warnings
from unittest import mock

import yaml

from pyrai3md.qualification import manager as manager_module
from pyrai3md.qualification.manager import (EXAMPLE_CHAINS,
                                            EXAMPLE_FAMILIES,
                                                                                        _EXAMPLE_FIXTURE_SUFFIXES,
                                            _TOPOLOGY_SUFFIXES,
                                            SCRIPT_TIMEOUT_MARGIN,
                                            SWITCH_CONDITION,
                                            _coerce_example_flag,
                                            _coerce_flag,
                                            _coerce_switch,
                                            _example_family,
                                            _example_missing_requirements,
                                            _example_submits_remotely,
                                            _order_example_files,
                                            _resolve_chain,
                                            _resolve_timeout,
                                            _runall_mode,
                                            _stage_example_skill_fixtures,
                                            _write_shell_script,
                                            load_config)


class TestMultiscaleExamples(unittest.TestCase):
    """The `multiscale_test` family and its shipped cases."""

    def _examples_dir(self):
        return os.path.abspath(os.path.join(
            os.path.dirname(manager_module.__file__), os.pardir, 'examples'))

    def test_cases_exist_and_are_qualified(self):
        from pyrai3md.qualification.manager import (
            EXAMPLE_FAMILIES, POSTPONED_EXAMPLES, family_members)
        # Family membership is the prefix the switch table declares, so
        # the criterion and the planner cannot disagree about it.
        stems = family_members(EXAMPLE_FAMILIES['multiscale_test'])
        # 37 after the ONIOM true-embedding set (ten cases: base, algorithm
        # and the two periodic legs), on top of the 27 that #449 left.
        self.assertEqual(len(stems), 37)
        for stem in stems:
            self.assertNotIn(stem, POSTPONED_EXAMPLES, stem)
            self.assertTrue(stem.startswith('multiscale_oniom')
                            or stem.startswith('multiscale_qmmm'), stem)

    def test_family_switch_opens_and_closes_the_cases(self):
        from pyrai3md.qualification.manager import _example_state
        workflow = 'multiscale_qmmm_embedding_sp'
        self.assertEqual(
            _example_state(SWITCH_CONDITION,
                           {'multiscale_test': True}, workflow),
            (True, 'examples.multiscale_test'))
        self.assertEqual(
            _example_state(SWITCH_CONDITION,
                           {'multiscale_test': False}, workflow),
            (False, 'examples.multiscale_test'))
        self.assertEqual(
            _example_state(SWITCH_CONDITION,
                           {'multiscale_test': SWITCH_CONDITION}, workflow),
            (SWITCH_CONDITION, ''))

    def test_multicenter_oniom_cases_are_qualified(self):
        from pyrai3md.qualification.manager import POSTPONED_EXAMPLES
        for stem in ('multicenter_oniom2_orca_embedding_sp',
                     'multicenter_oniom2_orca_no_embedding_sp',
                     'multicenter_oniom2_orca_embedding_md',
                     'multicenter_oniom2_molcas_embedding_nac_sp',
                     'multicenter_oniom3_orca_embedding_sp',
                     'multicenter_oniom3_orca_embedding_npt',
                     'multicenter_oniom2_orca_charged_sp',
                     'multicenter_oniom2_orca_charged_no_embedding_sp',
                     'multicenter_oniom2_orca_charged_md',
                     'multicenter_oniom3_orca_charged_sp',
                     'multicenter_oniom3_orca_charged_no_embedding_sp',
                     'multicenter_oniom3_orca_charged_md'):
            self.assertNotIn(stem, POSTPONED_EXAMPLES, stem)

    def test_read_charge_cases_ship_their_charge_file(self):
        self.assertIn('.charge', _EXAMPLE_FIXTURE_SUFFIXES)
        examples_dir = self._examples_dir()
        for stem in ('multiscale_oniom2_read_charge_sp',
                     'multiscale_oniom2_read_charge_md',
                     'multiscale_oniom3_read_charge_sp',
                     'multiscale_oniom3_read_charge_md'):
            path = os.path.join(examples_dir, stem + '_example.charge')
            self.assertTrue(os.path.isfile(path), path)


class TestCoerceExampleFlag(unittest.TestCase):
    """Example switches from pmd_env.yaml become strict booleans."""

    def test_boolean_values_pass_through(self):
        self.assertTrue(_coerce_example_flag(True, 'energy'))
        self.assertFalse(_coerce_example_flag(False, 'energy'))

    def test_integer_values(self):
        self.assertTrue(_coerce_example_flag(1, 'energy'))
        self.assertFalse(_coerce_example_flag(0, 'energy'))

    def test_string_false_spellings_disable(self):
        for value in ('false', 'False', 'FALSE', 'no', 'off', '0'):
            with self.subTest(value=value):
                self.assertFalse(_coerce_example_flag(value, 'energy'))

    def test_string_true_spellings_enable(self):
        for value in ('true', 'True', 'TRUE', 'yes', 'on', '1'):
            with self.subTest(value=value):
                self.assertTrue(_coerce_example_flag(value, 'energy'))

    def test_unrecognized_value_warns_and_stays_enabled(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(_coerce_example_flag('fasle', 'gradient'))

        self.assertEqual(len(caught), 1)
        self.assertIn('examples.gradient', str(caught[0].message))
        self.assertIn("'fasle'", str(caught[0].message))


class TestLoadConfigDuplicateKeys(unittest.TestCase):
    """Duplicate mapping keys in pmd_env.yaml must fail loudly:
    yaml.safe_load silently keeps the last value, which collapsed
    the benchmark.fssh tully1/2/3 entries into a single method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, text):
        path = os.path.join(self.tmpdir, 'pmd_env.yaml')
        with open(path, 'w', encoding='utf-8') as handler:
            handler.write(text)
        return path

    def test_duplicate_key_raises_with_the_key_name(self):
        path = self._write(
            'benchmark:\n'
            '  fssh:\n'
            '    - method: tully2\n'
            '      enabled: true\n'
            '      method: tully1\n')
        with self.assertRaises(yaml.YAMLError) as ctx:
            load_config(path)
        self.assertIn('method', str(ctx.exception))
        self.assertIn('duplicate key', str(ctx.exception))

    def test_clean_mapping_loads_unchanged(self):
        path = self._write(
            'benchmark:\n'
            '  fssh:\n'
            '    - method: tully1\n'
            '      v0: 0.005\n'
            '    - method: tully2\n'
            '      v0: 0.0075\n'
            '    - method: tully3\n'
            '      v0: 0.005\n')
        config = load_config(path)
        entries = config['benchmark']['fssh']
        self.assertEqual([e['method'] for e in entries],
                         ['tully1', 'tully2', 'tully3'])
        self.assertEqual([e['v0'] for e in entries],
                         [0.005, 0.0075, 0.005])


class TestLoadConfigCwdOverride(unittest.TestCase):
    """The source-tree layer ./pyrai3md/qualification/pmd_env.yaml
    (resolved from the working directory at load time) wins over the
    installed package config, so editing the workspace config takes
    effect without a reinstall -- the stale-snapshot divergence that
    once shipped a stale deco keyword into the benchmark inputs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        # Keep the user layer out of the picture: the assertions
        # below compare package vs source-tree values only.
        self._patcher = mock.patch.object(
            manager_module, '_load_user_config', return_value=None)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_cwd_config(self, text):
        qual_dir = os.path.join(
            self.tmpdir, 'pyrai3md', 'qualification')
        os.makedirs(qual_dir, exist_ok=True)
        with open(os.path.join(qual_dir, 'pmd_env.yaml'), 'w',
                  encoding='utf-8') as handler:
            handler.write(text)
        os.chdir(self.tmpdir)

    def test_cwd_layer_overrides_package_values(self):
        self._write_cwd_config(
            'examples:\n'
            '  energy: true\n'
            'deploy:\n'
            '  host: source-tree-override\n')
        config = load_config()
        self.assertTrue(config['examples']['energy'])
        self.assertEqual(config['deploy']['host'],
                         'source-tree-override')
        # Untouched package values survive the deep merge. Compare with
        # the package layer itself: a qualification round narrows
        # pmd_env.yaml (a skill or example round flips these switches),
        # and hardcoding `gradient: false` made this test fail on every
        # narrowed round even though the merge was correct.
        package = load_config(manager_module.DEFAULT_CONFIG)
        self.assertEqual(config['examples']['gradient'],
                         package['examples']['gradient'])

    def test_cwd_empty_value_falls_back_to_the_package(self):
        self._write_cwd_config('deploy:\n  host: ""\n')
        config = load_config()
        self.assertEqual(
            config, load_config(manager_module.DEFAULT_CONFIG))

    def test_no_cwd_file_keeps_the_existing_chain(self):
        os.chdir(self.tmpdir)  # not a PyRAI3MD checkout
        self.assertEqual(
            load_config(),
            load_config(manager_module.DEFAULT_CONFIG))

    def test_explicit_path_ignores_the_cwd_layer(self):
        self._write_cwd_config('deploy:\n  host: must-not-win\n')
        config = load_config(manager_module.DEFAULT_CONFIG)
        self.assertNotEqual(config['deploy']['host'],
                            'must-not-win')

    def test_cwd_duplicate_key_raises(self):
        self._write_cwd_config(
            'deploy:\n'
            '  host: a\n'
            '  host: b\n')
        with self.assertRaises(yaml.YAMLError) as ctx:
            load_config()
        self.assertIn('duplicate key', str(ctx.exception))


class TestRunallMode(unittest.TestCase):
    """The top-level _runall master switch takes three states."""

    def test_missing_key_defers_to_the_switches_below(self):
        # `condition` is what an absent key means. It used to be `false`, and
        # `false` used to mean "do not force anything" - so a file written
        # before the three-state switch existed keeps its effective behaviour
        # only if the default defers.
        self.assertEqual(_runall_mode({}), SWITCH_CONDITION)

    def test_boolean_values_pass_through(self):
        self.assertIs(_runall_mode({'_runall': True}), True)
        self.assertIs(_runall_mode({'_runall': False}), False)

    def test_integer_values(self):
        self.assertIs(_runall_mode({'_runall': 1}), True)
        self.assertIs(_runall_mode({'_runall': 0}), False)

    def test_string_spellings(self):
        for value in ('true', 'yes', 'on', '1'):
            with self.subTest(value=value):
                self.assertIs(_runall_mode({'_runall': value}), True)
        for value in ('false', 'no', 'off', '0'):
            with self.subTest(value=value):
                self.assertIs(_runall_mode({'_runall': value}), False)
        self.assertEqual(_runall_mode({'_runall': 'Condition'}),
                         SWITCH_CONDITION)

    def test_unrecognized_value_warns_and_defers(self):
        # The fallback is the state that keeps the most tests under the
        # operator's own switches, never the one that silently closes them.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(_runall_mode({'_runall': 'fasle'}),
                             SWITCH_CONDITION)
        self.assertEqual(len(caught), 1)
        self.assertIn('_runall', str(caught[0].message))


class TestIntegrationSwitch(unittest.TestCase):
    """The integration suite sits under `test_integration` and `_runall`."""

    @staticmethod
    def _enabled(config):
        from pyrai3md.qualification.integration.planner import \
            integration_enabled
        return integration_enabled(config)

    def test_missing_or_false_is_off(self):
        self.assertFalse(self._enabled({}))
        self.assertFalse(self._enabled({'test_integration': False}))

    def test_true_and_string_true_are_on(self):
        self.assertTrue(self._enabled({'test_integration': True}))
        self.assertTrue(self._enabled({'test_integration': 'true'}))

    def test_runall_false_closes_even_a_true_integration_key(self):
        self.assertFalse(self._enabled(
            {'_runall': False, 'test_integration': True}))

    def test_runall_true_opens_even_a_false_integration_key(self):
        self.assertTrue(self._enabled(
            {'_runall': True, 'test_integration': False}))


class TestSwitchCoercion(unittest.TestCase):
    """Three-valued switches, two-valued leaves, and the chain order."""

    def test_leaf_rejects_condition_and_says_why(self):
        # A leaf is the end of the chain: `condition` there is a misplaced
        # switch, not a third answer, and the case stays enabled.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(_coerce_example_flag(
                SWITCH_CONDITION, 'constraint_md_shake'))
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn('examples.constraint_md_shake', message)
        self.assertIn('per-example switch at the end of the chain', message)

    def test_leaf_default_controls_the_fallback_wording(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(_coerce_flag('fasle', 'skills.x.tests.y', True,
                                         'case'))
        self.assertIn('case stays enabled', str(caught[0].message))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertIs(
                _coerce_flag('fasle', 'skills.x.tests.y', False, 'case'),
                False)
        self.assertIn('stays disabled', str(caught[-1].message))

    def test_chain_reports_the_first_decisive_switch(self):
        label = 'skills.molcas_skill.test'
        state, closer = _resolve_chain([
            (SWITCH_CONDITION, '_runall'), (False, label)])
        self.assertIs(state, False)
        self.assertEqual(closer, label)
        # All-deferred chain: the per-test switch decides, and there is no
        # closer to name yet.
        self.assertEqual(
            _resolve_chain([(SWITCH_CONDITION, '_runall'),
                            (SWITCH_CONDITION, label)]),
            (SWITCH_CONDITION, ''))
        # The master speaks first: a `true` above a `false` below is `true`.
        self.assertEqual(
            _resolve_chain([(True, '_runall'), (False, label)]),
            (True, '_runall'))

    def test_family_table_is_the_shipped_families_and_no_more(self):
        # By ruling: the LAMMPS MD set, the constraint MD set, the optimize set,
        # the path set, the multiscale set and the adaptive set get a family
        # switch; energy/gradient/multicenter and the two constraint cases that
        # also relax a geometry stay individual.
        # The equality is the point - a new family is a decision, not a
        # refactor: the path family landed so one key opens the whole chain
        # without opening the single-structure searches with it, and the
        # adaptive family so a probe -> reference -> train round is one key
        # away the way the multiscale band already is.
        self.assertEqual(EXAMPLE_FAMILIES,
                         {'md_lammps_test': 'md_lammps_',
                          'constraint_md_test': 'constraint_md_',
                          'optimize_test': 'optimize_',
                          'path_test': 'path_',
                          'multiscale_test': 'multiscale_',
                          'adaptive_test': 'adaptive_'})


class TestAdaptiveFamilySwitch(unittest.TestCase):
    """The `adaptive_test` family and its shipped carriers."""

    def _examples_dir(self):
        return os.path.abspath(os.path.join(
            os.path.dirname(manager_module.__file__), os.pardir, 'examples'))

    def test_cases_exist_and_are_qualified(self):
        from pyrai3md.qualification.manager import POSTPONED_EXAMPLES
        stems = sorted(
            os.path.splitext(fname)[0]
            for fname in os.listdir(self._examples_dir())
            if fname.startswith('adaptive_') and fname.endswith('.yaml'))
        # Three systems (single method, ONIOM2, QMMM) x two carriers
        # (local, slurm), plus one local carrier per analysis/exploration
        # feature.
        self.assertEqual(stems, [
            'adaptive_analysis_coverage_local',
            'adaptive_analysis_gap_local',
            'adaptive_analysis_pes_local',
            'adaptive_analysis_trace_local',
            'adaptive_checks_nosh_local',
            'adaptive_gap_cv_local',
            'adaptive_gap_paper_local',
            'adaptive_md_max_local',
            'adaptive_oniom2_local',
            'adaptive_oniom2_slurm',
            'adaptive_predict_primary_local',
            'adaptive_qmmm_local',
            'adaptive_qmmm_slurm',
            'adaptive_quality_gate_local',
            'adaptive_single_local',
            'adaptive_single_slurm',
            'adaptive_udd_force_local',
            'adaptive_udd_potential_local',
        ])
        for stem in stems:
            self.assertNotIn(stem, POSTPONED_EXAMPLES, stem)

    def test_family_switch_opens_and_closes_the_cases(self):
        from pyrai3md.qualification.manager import _example_state
        workflow = 'adaptive_qmmm_slurm'
        for state in (True, False):
            self.assertEqual(
                _example_state(SWITCH_CONDITION, {'adaptive_test': state},
                               workflow),
                (state, 'examples.adaptive_test'))
        # Deferring is what lets the per-case keys answer.
        self.assertEqual(
            _example_state(SWITCH_CONDITION,
                           {'adaptive_test': SWITCH_CONDITION}, workflow),
            (SWITCH_CONDITION, ''))










class TestShellScriptTimeout(unittest.TestCase):
    """The execution script embeds a `timeout` guard so a timed-out
    calculation is killed at the process-group level, leaving no
    residual processes (which would otherwise slow every later test).
    The script timeout is the task subprocess timeout minus a margin so
    the script fires just before the runner's backstop."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read(self, name):
        with open(os.path.join(self.tmpdir, f'{name}.sh'),
                  encoding='utf-8') as f:
            return f.read()

    def test_resolve_timeout_default_and_explicit(self):
        self.assertEqual(_resolve_timeout({}), 300)
        self.assertEqual(_resolve_timeout({'_timeout': 600}), 600)

    def test_resolve_timeout_bad_value_falls_back(self):
        self.assertEqual(_resolve_timeout({'_timeout': None}), 300)
        self.assertEqual(_resolve_timeout({'_timeout': 'abc'}), 300)

    def test_timeout_wraps_command_with_margin(self):
        _write_shell_script(
            self.tmpdir, 'wrap', 'cd /a && pyrai3md_skill c.config',
            timeout=300)
        content = self._read('wrap')
        expected = 300 - SCRIPT_TIMEOUT_MARGIN
        self.assertIn(f'timeout -k 10s {expected}s bash -c', content)
        # The command survives verbatim inside the quoted bash -c.
        self.assertIn("bash -c 'cd /a && pyrai3md_skill c.config'", content)

    def test_shell_and_export_stay_outside_the_timeout(self):
        _write_shell_script(
            self.tmpdir, 'env', 'pyrai3md x.input',
            export_vars={'MPI_NP': '1'},
            shell_commands=['module load gcc'], timeout=300)
        lines = self._read('env').splitlines()
        self.assertIn('module load gcc', lines)
        self.assertIn('export MPI_NP=1', lines)
        timeout_idx = next(i for i, l in enumerate(lines)
                           if l.startswith('timeout'))
        self.assertLess(lines.index('module load gcc'), timeout_idx)

    def test_no_timeout_writes_command_verbatim(self):
        _write_shell_script(self.tmpdir, 'plain',
                            'cd /b && pyrai3md y.input')
        content = self._read('plain')
        self.assertNotIn('timeout', content)
        self.assertIn('cd /b && pyrai3md y.input', content)

















class TestPlanningWorkers(unittest.TestCase):
    """environment.planning_workers: 0 = auto, 1 = serial."""

    def test_zero_uses_at_most_eight_and_never_more_than_the_host(self):
        with mock.patch.object(manager_module.os, 'cpu_count', return_value=32):
            self.assertEqual(
                manager_module._planning_worker_count(
                    {'environment': {'planning_workers': 0}}), 8)
        with mock.patch.object(manager_module.os, 'cpu_count', return_value=3):
            self.assertEqual(
                manager_module._planning_worker_count(
                    {'environment': {'planning_workers': 0}}), 3)

    def test_explicit_value_wins_and_bad_values_fall_back_to_auto(self):
        with mock.patch.object(manager_module.os, 'cpu_count', return_value=16):
            self.assertEqual(
                manager_module._planning_worker_count(
                    {'environment': {'planning_workers': 2}}), 2)
            self.assertEqual(
                manager_module._planning_worker_count(
                    {'environment': {'planning_workers': '3'}}), 3)
            self.assertEqual(
                manager_module._planning_worker_count(
                    {'environment': {'planning_workers': 'bad'}}), 8)
        self.assertEqual(
            manager_module._planning_worker_count({}),
            min(8, os.cpu_count() or 1))


if __name__ == '__main__':
    unittest.main()
