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

try:
    import pytest
except ImportError:  # unittest-only hosts do not need pytest
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.integration

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


def _fake_yaml_to_config(yaml_path, config_path=None):
    """Stand-in for yaml_to_config that writes the file it promises.

    Planning renders the YAML carrier's flat config once and copies it
    for the classic carrier (#473). A mock without a side effect leaves
    that copy without a source, so the stand-in writes the empty flat
    config the mocked tests never read.
    """
    config = {}
    if config_path:
        with open(config_path, 'w', encoding='utf-8') as handle:
            json.dump(config, handle)
    return config













def _path_examples():
    """The shipped `path_*` stems, read from disk (one family, one source)."""
    from pyrai3md.qualification.manager import family_members
    return family_members('path_')


class TestExampleRunallOverride(unittest.TestCase):
    """_runall forces the configured example switches on but never
    lifts the POSTPONED_EXAMPLES policy."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Isolate the gating logic from the converter pipeline: the
        # assertions below only inspect the planned task descriptors.
        self._patchers = [
            mock.patch.object(manager_module, 'yaml_to_config',
                              side_effect=_fake_yaml_to_config),
            mock.patch.object(manager_module, '_planning_worker_count',
                              return_value=1),
            mock.patch.object(
                manager_module, 'convert_to_classic_input'),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self._patchers:
            patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disabled_example_is_skipped_under_condition(self):
        # `condition` is the setting under which a per-example switch speaks.
        # With the master explicitly `false` the case closes from the master,
        # which the test below pins separately.
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'energy': False}}
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        names = {task['name'] for task in tasks}
        skipped = {task['name'] for task in tasks if task.get('skip')}
        self.assertIn('example_energy', skipped)
        self.assertNotIn('example_energy_yaml', names)
        reason = [task['skip_reason'] for task in tasks
                  if task['name'] == 'example_energy'][0]
        # The reason names the key to edit, not just "disabled": three
        # different switches can close one case.
        self.assertEqual(reason, 'disabled by examples.energy')

    def test_master_false_closes_an_enabled_example(self):
        # The inversion this switch set introduces: `false` on the master is
        # now "close everything", where it used to mean "do not force".
        config = {'_runall': False, 'examples': {'energy': True}}
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        energy = [task for task in tasks if task['name'] == 'example_energy']
        self.assertTrue(energy[0].get('skip'))
        self.assertEqual(energy[0]['skip_reason'],
                         'disabled by _runall')

    def test_family_switch_overrides_its_cases(self):
        # `md_lammps_test: true` opens the family even though every
        # md_lammps_* case is individually switched off in pmd_env.yaml.
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'md_lammps_test': True,
                               'md_lammps_nve': False,
                               'md_lammps_nvt_nh': False}}
        skipped = {task['name'] for task
                   in manager_module._plan_example_tests(config, self.tmpdir)
                   if task.get('skip')}
        self.assertNotIn('example_md_lammps_nve', skipped)
        self.assertNotIn('example_md_lammps_nvt_nh', skipped)

    def test_family_switch_false_closes_its_cases(self):
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'md_lammps_test': False,
                               'md_lammps_nve': True}}
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        closed = [task for task in tasks
                  if task['name'] == 'example_md_lammps_nve']
        self.assertTrue(closed[0].get('skip'))
        self.assertEqual(closed[0]['skip_reason'],
                         'disabled by examples.md_lammps_test')
        # A case outside the family keeps its own switch.
        self.assertNotIn('example_energy',
                         {task['name'] for task in tasks if task.get('skip')})

    def test_family_condition_defers_to_each_case(self):
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'constraint_md_test': SWITCH_CONDITION,
                                'constraint_md_shake': False,
                                'constraint_md_bias': True}}
        skipped = {task['name'] for task
                   in manager_module._plan_example_tests(config, self.tmpdir)
                   if task.get('skip')}
        self.assertIn('example_constraint_md_shake', skipped)
        self.assertNotIn('example_constraint_md_bias', skipped)

    def test_absent_family_key_defers(self):
        # Backward compatibility: a file that names no family switch keeps
        # every per-example switch in charge, as it did before they existed.
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'md_lammps_nve': False}}
        skipped = {task['name'] for task
                   in manager_module._plan_example_tests(config, self.tmpdir)
                   if task.get('skip')}
        self.assertIn('example_md_lammps_nve', skipped)

    def test_unknown_examples_key_is_reported(self):
        # A family switch is read by name, so `md_lammps_tes: true` would
        # otherwise sit in the file doing nothing while the operator believes
        # the family is on.
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'md_lammps_tes': True, 'energy': True}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            manager_module._plan_example_tests(config, self.tmpdir)
        messages = [str(item.message) for item in caught]
        # The offending key is quoted: `'md_lammps_tes'` cannot be satisfied by
        # the table printing `md_lammps_test`, which is what a substring check
        # would accept without the typo having been named at all.
        self.assertTrue(any("'md_lammps_tes'" in message
                            for message in messages), messages)
        for family in EXAMPLE_FAMILIES:
            self.assertTrue(any(family in message for message in messages),
                            (family, messages))

    def test_runall_forces_disabled_example_on(self):
        config = {'_runall': True, 'examples': {'energy': False}}
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        names = {task['name'] for task in tasks}
        skipped = {task['name'] for task in tasks if task.get('skip')}
        self.assertNotIn('example_energy', skipped)
        for suffix in ('_yaml', '_input'):
            self.assertIn('example_energy' + suffix, names)
        # Agent mode is not planned for examples at all.
        self.assertNotIn('example_energy_agent', names)

    def test_example_tasks_carry_their_artifact_anchors(self):
        # The runner locates an example's log and trajectory by the run
        # directory and control.title, both of which differ from the
        # source YAML path and stem (energy.yaml titles itself
        # energy_example). Without these the post-run checks are a
        # silent no-op.
        config = {'_runall': True, 'examples': {}}
        tasks = {task['name']: task for task in
                 manager_module._plan_example_tests(config, self.tmpdir)
                 if not task.get('skip')}
        for suffix in ('_yaml', '_input'):
            task = tasks['example_energy' + suffix]
            self.assertEqual(task['run_dir'], os.path.join(
                self.tmpdir, 'examples', 'energy', suffix.lstrip('_')))
            self.assertEqual(task['title'], 'energy_example')
            # Both modes declare their ensemble in the same source YAML;
            # the input mode's own input_file is the generated classic
            # input, which no post-run check can parse. Without this
            # anchor an entire mode is judged by nothing at all -- the
            # 22:30 run is what that looked like: 12 input-mode examples
            # reported pass with no verdict line of any kind.
            self.assertTrue(task['spec_file'].endswith('.yaml'),
                            '%s: %s' % (task['name'], task['spec_file']))
            self.assertEqual(
                os.path.basename(task['spec_file']), 'energy.yaml')

    def test_no_two_example_tasks_share_a_run_directory(self):
        # The 2026-09-28 suite ran 24 example tasks in one directory, so
        # every backend wrote its engine log into the same tmp_LAMMPS and
        # the last task of the suite left the eight failed ones with the
        # evidence of a run that passed. Uniqueness of run_dir is what
        # makes a failure keep its own log; the titles are allowed to
        # repeat only across the two input modes of one example, which
        # now sit in different directories.
        config = {'_runall': True, 'examples': {}}
        tasks = [task for task in
                 manager_module._plan_example_tests(config, self.tmpdir)
                 if not task.get('skip')]
        directories = [task['run_dir'] for task in tasks]
        # One exception is declared rather than tolerated: a chain shares
        # its directory within one mode precisely so its later member can
        # read what its earlier member wrote (nn_predict evaluating the
        # model nn_train fitted). Anything else colliding is the bug this
        # assertion was written for.
        shared = set(d for d in directories if directories.count(d) > 1)
        for directory in shared:
            grouped = [t for t in tasks if t['run_dir'] == directory]
            self.assertEqual(2, len(grouped),
                             '%s holds %d tasks' % (directory, len(grouped)))
            names = {os.path.splitext(os.path.basename(t['spec_file']))[0]
                     for t in grouped}
            modes = {t['mode'] for t in grouped}
            self.assertEqual(1, len(modes),
                             '%s mixes modes: %s' % (directory, modes))
            # The set is compared, not the sequence: the plan order is the
            # chain's own business and is asserted where the chain is.
            self.assertIn(names, [set(members)
                                  for members in EXAMPLE_CHAINS.values()],
                          '%s is shared by %s, which no chain declares'
                          % (directory, sorted(names)))
        for example in ('energy', 'gradient', 'nn_grid_search'):
            own = [t['run_dir'] for t in tasks
                   if os.path.basename(t['spec_file']) == '%s.yaml' % example]
            self.assertEqual(len(own), len(set(own)),
                             '%s shares its directory with another example'
                             % example)
        for task in tasks:
            self.assertTrue(os.path.isdir(task['run_dir']),
                            '%s was not created' % task['run_dir'])
            self.assertTrue(task['script'].startswith(task['run_dir']),
                            'the execution script of %s escapes its own '
                            'directory: %s' % (task['name'], task['script']))
            # The flattened config and the classic input are written into
            # the task's own directory (this class mocks the converter,
            # so their paths are what is checked, not their contents).
            for key in ('input_file',):
                if task['mode'] == 'pyrai3md_input':
                    self.assertEqual(os.path.dirname(task[key]),
                                     task['run_dir'])

    def test_every_shipped_example_is_qualified(self):
        # Nothing the package ships sits in the postponement branch any more:
        # the three dispatch topologies (md_local serial, md_parallel local
        # pool, md_batch Slurm fan-out) were the last ones, and this round
        # took them in. A new example that is not registered would fail here
        # rather than disappear from the round silently.
        config = {'_runall': True, 'examples': {}}
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        postponed = [task for task in tasks
                     if 'postponed' in str(
                         task.get('skip_reason', '')).lower()]
        self.assertEqual(postponed, [],
                         'no shipped example may be postponed')
        planned = {task['name'] for task in tasks}
        by_name = {task['name']: task for task in tasks}
        for name in ('md_local', 'md_parallel', 'md_batch'):
            for mode in ('yaml', 'input'):
                task_name = 'example_%s_%s' % (name, mode)
                if task_name in planned:
                    continue
                # The alternative is a single skip record saying why the
                # example is not running here -- an example that submits its
                # children to Slurm on a host with no client. Both modes
                # collapse into it, so the example is still reported on, with
                # a reason a reader can act on.
                skipped = by_name.get('example_%s' % name)
                self.assertIsNotNone(
                    skipped,
                    '%s disappeared from the plan without a task of its own'
                    % name)
                self.assertTrue(skipped.get('skip'), name)
                self.assertIn('sbatch', str(skipped.get('skip_reason', '')),
                              '%s was skipped without naming the client it '
                              'needs' % name)
        # The local-pool arm of the same topology set needs no cluster client,
        # so on a host that can run the nn stack it keeps both of its modes.
        # TensorFlow is a different requirement: a host without it reports the
        # chain as a named skip, and that record is what the plan says here.
        missing_nn = _skipped_for(by_name, 'nn_grid_search_local', 'tensorflow')
        if missing_nn is not None:
            self.assertIn('tensorflow', str(missing_nn['skip_reason']))
        else:
            for mode in ('yaml', 'input'):
                self.assertIn('example_nn_grid_search_local_%s' % mode, planned)

    def test_a_new_example_runs_and_a_named_one_is_postponed(self):
        # The plan is the directory (item 2.11): a YAML the package ships
        # is planned as runnable under _runall, and only a case named in
        # POSTPONED_EXAMPLES is reported as a skip carrying its reason --
        # which keeps that branch reachable and makes "ship a file, get a
        # plan" the contract.
        with tempfile.TemporaryDirectory() as examples:
            with open(os.path.join(examples, 'planted.yaml'), 'w',
                      encoding='utf-8') as handle:
                handle.write('control:\n  jobtype: energy\n'
                             '  theory: [pyscf]\n'
                             '  title: planted_example\n')
            config = {'_runall': True, 'examples': {}}
            tasks = manager_module._plan_example_tests(
                config, self.tmpdir, examples_dir=examples)
            self.assertEqual([task['name'] for task in tasks],
                             ['example_planted_yaml',
                              'example_planted_input'])
            self.assertTrue(all(not task.get('skip') for task in tasks))

            postponed = manager_module.POSTPONED_EXAMPLES
            manager_module.POSTPONED_EXAMPLES = {
                'planted': 'waiting for the runtime nobody has stabilized'}
            try:
                tasks = manager_module._plan_example_tests(
                    config, self.tmpdir, examples_dir=examples)
            finally:
                manager_module.POSTPONED_EXAMPLES = postponed
        self.assertTrue(all(task.get('skip') for task in tasks))
        self.assertIn('postponed', tasks[0]['skip_reason'].lower())
        self.assertIn('runtime nobody has stabilized',
                      tasks[0]['skip_reason'])


class TestPathFamilySwitch(unittest.TestCase):
    """`path_test` opens the thirty-one `path_*` examples, and `optimize_test`
    does not reach them.

    Both halves are decisions rather than incidental behaviour, which is why
    they are pinned together. One key has to open the whole path chain (the
    twenty-one cases ship with every per-case key `false`, so a family key that
    did not outrank them would plan a round of nothing), and it has to open
    *only* that chain: the two prefixes are disjoint so that an optimize round
    does not run 42 path rows with it, which is the separation the shipped
    posture reads the other way - a path round closes all twenty-five
    `optimize_*` cases. `test_manager`'s family-table test pins the membership
    list against the disk; these two assert what the switch does to it.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # The converter is mocked as in TestExampleRunallOverride: the
        # assertions below read the planned descriptors, not the files.
        self._patchers = [
            mock.patch.object(manager_module, 'yaml_to_config',
                              side_effect=_fake_yaml_to_config),
            mock.patch.object(manager_module, '_planning_worker_count',
                              return_value=1),
            mock.patch.object(
                manager_module, 'convert_to_classic_input'),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self._patchers:
            patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _planned(self, examples):
        return manager_module._plan_example_tests(
            {'_runall': SWITCH_CONDITION, 'examples': examples}, self.tmpdir)

    def test_path_family_switch_opens_every_path_case(self):
        examples = {name: False for name in _path_examples()}
        examples['path_test'] = True
        examples['optimize_test'] = False
        tasks = self._planned(examples)
        closed = {task['name'] for task in tasks if task.get('skip')}
        for name in _path_examples():
            for mode in ('yaml', 'input'):
                planned = 'example_%s_%s' % (name, mode)
                self.assertNotIn(planned, closed,
                                 '%s was closed although path_test is true'
                                 % planned)
        self.assertEqual(
            len([task for task in tasks
                 if task['name'].startswith('example_path_')]),
            2 * len(_path_examples()),
            'the path family planned a different number of rows than the two '
            'carriers of its own list')

    def test_optimize_family_switch_does_not_reach_the_path_cases(self):
        # The separation read from the other side: an optimize round, with the
        # path cases switched *on* individually, still does not run them - and
        # the reason it gives names the path switch, not the optimize one. A
        # closed case folds into one row named after the case alone, which is
        # why the two carriers are not looked for here.
        examples = {name: True for name in _path_examples()}
        examples['path_test'] = False
        examples['optimize_test'] = True
        tasks = self._planned(examples)
        closed = {task['name']: task for task in tasks if task.get('skip')}
        for name in _path_examples():
            folded = 'example_%s' % name
            self.assertIn(folded, closed,
                          '%s ran although path_test is false' % name)
            self.assertEqual(closed[folded]['skip_reason'],
                             'disabled by examples.path_test')
            for mode in ('yaml', 'input'):
                self.assertNotIn('%s_%s' % (folded, mode), closed,
                                 '%s planned a carrier row while closed'
                                 % name)
        self.assertFalse([task for task in tasks
                          if task['name'].startswith('example_optimize_')
                          and 'examples.optimize_test' in str(
                              task.get('skip_reason') or '')],
                         'optimize_test is true, so it may not close an '
                         'optimize case')


class TestExampleConverterImport(unittest.TestCase):
    """Regression: yaml_to_config imports the Interface parser through
    the correct package depth (...interface.parser). A wrong depth
    (..interface.parser) stayed latent while every example switch was
    off -- _plan_example_tests skipped before reaching the converter
    -- and only surfaced once _runall forced example planning on,
    crashing generate_plan and the remote pyrai3md test run."""

    def test_yaml_to_config_runs_on_a_shipped_example(self):
        from pyrai3md.qualification.certification.converter import (
            yaml_to_config)

        examples_dir = os.path.abspath(os.path.join(
            os.path.dirname(manager_module.__file__),
            os.pardir, 'examples'))
        yaml_path = os.path.join(examples_dir, 'energy.yaml')
        self.assertTrue(os.path.isfile(yaml_path))

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = os.path.join(tmpdir, 'energy.json')

        config = yaml_to_config(yaml_path, config_path)

        self.assertIsInstance(config, dict)
        self.assertIn('jobtype', config)
        self.assertIn('theory', config)
        self.assertTrue(os.path.isfile(config_path))


class TestExampleSkillEnvInjection(unittest.TestCase):
    """_inject_example_skill_env shares the pmd_env.yaml skill settings
    into an example flat config so the generated .input/.config carry
    each skill section's machine parameters (path/nproc/shell/
    execution_mode/...). The skills.<name>.test switch only gates skill
    certification; an already-configured section is shared with the
    example tasks regardless (regression: example conversion dropped
    every skill setting, leaving the skill path empty)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _write_config(self, config):
        path = os.path.join(self.tmpdir, 'example.json')
        with open(path, 'w', encoding='utf-8') as handler:
            json.dump(config, handler)
        return path

    def _read(self, path):
        with open(path, encoding='utf-8') as handler:
            return json.load(handler)

    def test_multi_skill_sections_receive_their_own_env(self):
        config = {
            'theory': ['openqp', 'xtb'],
            'skill_name': 'openqp_skill',
            'skill_sections': {
                'openqp': {'method': 'mrsf'},
                'xtb': {'method': 'gfn2'},
            },
        }
        path = self._write_config(config)
        skills = {
            'openqp_skill': {
                'openqp_path': '/opt/openqp', 'openqp_nproc': 2,
                'execution_mode': 'local', 'keep_tmp': 1, 'test': False,
            },
            'xtb_skill': {
                'xtb_path': '/opt/xtb', 'xtb_mem': 4000, 'test': False,
            },
        }
        manager_module._inject_example_skill_env(path, skills)
        out = self._read(path)
        oq = out['skill_sections']['openqp']
        xt = out['skill_sections']['xtb']
        self.assertEqual(oq['openqp_path'], '/opt/openqp')
        self.assertEqual(oq['openqp_nproc'], 2)
        self.assertEqual(oq['execution_mode'], 'local')
        self.assertEqual(oq['keep_tmp'], 1)
        self.assertEqual(oq['method'], 'mrsf')  # preserved
        self.assertEqual(xt['xtb_path'], '/opt/xtb')
        self.assertEqual(xt['xtb_mem'], 4000)
        # 'test' is a qualification-only key and never injected.
        self.assertNotIn('test', oq)
        self.assertNotIn('test', xt)
        # One skill's settings never leak into the other's block.
        self.assertNotIn('openqp_path', xt)

    def test_single_skill_injects_flat(self):
        config = {'theory': ['pyscf'], 'skill_name': 'pyscf_skill',
                  'method': 'dft'}
        path = self._write_config(config)
        skills = {'pyscf_skill': {'pyscf_path': 'import',
                                  'execution_mode': 'local',
                                  'test': True}}
        manager_module._inject_example_skill_env(path, skills)
        out = self._read(path)
        self.assertEqual(out['pyscf_path'], 'import')
        self.assertEqual(out['execution_mode'], 'local')
        self.assertEqual(out['method'], 'dft')
        self.assertNotIn('test', out)

    def test_fill_vs_override_semantics(self):
        config = {'theory': ['pyscf'], 'skill_name': 'pyscf_skill',
                  'scratch_path': '/user/scratch',
                  'execution_mode': 'slurm'}
        path = self._write_config(config)
        skills = {'pyscf_skill': {'scratch_path': '/env/scratch',
                                  'execution_mode': 'local'}}
        manager_module._inject_example_skill_env(path, skills)
        out = self._read(path)
        # Fill semantics keep the existing non-empty user value.
        self.assertEqual(out['scratch_path'], '/user/scratch')
        # Execution-control keys use override semantics.
        self.assertEqual(out['execution_mode'], 'local')

    def test_no_skills_is_a_noop(self):
        config = {'theory': ['pyscf'], 'skill_name': 'pyscf_skill'}
        path = self._write_config(config)
        manager_module._inject_example_skill_env(path, {})
        self.assertEqual(self._read(path), config)




class TestExampleFixtureStaging(unittest.TestCase):
    """Example runs stage shared skill topology fixtures (the LAMMPS
    h2o water box) from the skill's tests/ directory into the example
    run directory, under both the fixture name and the injected
    '<stem>_project' name the backend resolves."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmpdir, 'examples')
        os.makedirs(self.run_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_config(self, config):
        path = os.path.join(self.tmpdir, 'example.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        return path

    def _stage_shipped_example(self, stem):
        """Stage exactly as the runner does for a shipped example."""
        examples_dir = os.path.abspath(os.path.join(
            os.path.dirname(manager_module.__file__), os.pardir, 'examples'))
        example_path = os.path.join(examples_dir, stem + '.yaml')
        config_path = os.path.join(self.tmpdir, stem + '.json')
        manager_module.yaml_to_config(example_path, config_path)
        run_dir = os.path.join(self.tmpdir, stem + '_run')
        os.makedirs(run_dir)
        _stage_example_skill_fixtures(config_path, run_dir, example_path)
        return run_dir

    def test_a_leg_that_reads_no_force_field_stages_no_input_deck(self):
        # Round tmp_20261016_075328 ran `examples/optimize_cell_xtb.yaml` in a
        # directory holding three LAMMPS input decks copied in from the shared
        # examples/ folder, and the runner's minimum-image check then read a
        # pair cutoff out of a deck this engine never opened and printed
        # `[PBC] pair cutoff 10.000 A exceeds half of the 10.000 A cell`
        # inside a passing leg. Staging follows the names this example
        # resolves, so an xtb leg holds no force-field topology at all.
        run_dir = self._stage_shipped_example('optimize_cell_xtb')
        staged = os.listdir(run_dir)
        decks = [name for name in staged
                 if name.endswith(('.data', '.in.init', '.in.settings'))]
        self.assertEqual(decks, [],
                         'an xtb leg was staged a LAMMPS deck: %s' % decks)

    def test_a_declared_project_topology_still_lands(self):
        # The filter must not cut the case it exists for: the variable-cell
        # LAMMPS example names `lammps_project: water25`, and the box that
        # round 5 and round 6 measured (volume 15721.575 to 15774.652 A^3)
        # lives in the examples/ directory, not in the skill's tests/.
        run_dir = self._stage_shipped_example('optimize_cell_lammps')
        for name in ('water25.data', 'water25.in.init', 'water25.in.settings'):
            self.assertIn(name, os.listdir(run_dir), name)

    def test_lammps_fixtures_staged_under_project_name(self):
        config_path = self._write_config({
            'title': 'md_lammps_nvt_nh_example',
            'theory': ['lammps'],
            'lammps_project': 'h2o',
        })
        _stage_example_skill_fixtures(config_path, self.run_dir)
        staged = set(os.listdir(self.run_dir))
        self.assertIn('h2o.data', staged)
        self.assertIn('h2o.in.init', staged)
        self.assertIn('h2o.in.settings', staged)

    def test_project_override_renames_the_staged_copies(self):
        config_path = self._write_config({
            'title': 'md_example',
            'theory': ['lammps'],
            'lammps_project': 'waterbox',
        })
        _stage_example_skill_fixtures(config_path, self.run_dir)
        staged = set(os.listdir(self.run_dir))
        # Original fixture name plus the project-named copy.
        self.assertIn('h2o.data', staged)
        self.assertIn('waterbox.data', staged)
        self.assertIn('waterbox.in.init', staged)
        self.assertIn('waterbox.in.settings', staged)

    def test_skill_without_fixtures_is_a_noop(self):
        config_path = self._write_config({
            'title': 'md_local_example',
            'theory': ['pyscf'],
        })
        _stage_example_skill_fixtures(config_path, self.run_dir)
        self.assertEqual(os.listdir(self.run_dir), [])

    def test_unreadable_config_is_tolerated(self):
        missing = os.path.join(self.tmpdir, 'absent.json')
        _stage_example_skill_fixtures(missing, self.run_dir)
        self.assertEqual(os.listdir(self.run_dir), [])

    def test_unknown_skill_is_tolerated(self):
        config_path = self._write_config({
            'title': 'x',
            'theory': ['nosuchskill'],
        })
        _stage_example_skill_fixtures(config_path, self.run_dir)
        self.assertEqual(os.listdir(self.run_dir), [])

    def test_example_fixtures_are_not_overwritten_by_the_rename(self):
        """A project the skill does not ship keeps its own topology.

        The ethanol constraint examples name `lammps_project: ethanol`, and the
        only ethanol topology the package carries ships beside those YAMLs. The
        project-renamed copy exists to give a fixture the backend's name; run
        over a skill that also ships a different system, it wrote the water box
        content into `ethanol.data`, and a 9-atom molecule was then read against
        a 60-atom box. The renamed copy is therefore a fallback, not an
        overwrite.
        """
        example_dir = os.path.join(self.tmpdir, 'examples_src')
        os.makedirs(example_dir)
        for name in ('ethanol.data', 'ethanol.in.init', 'ethanol.in.settings'):
            with open(os.path.join(example_dir, name), 'w') as f:
                f.write('ETHANOL\n')
        example_path = os.path.join(example_dir, 'case.yaml')
        with open(example_path, 'w') as f:
            # Name the skill's own fixture so the two-source rule stages
            # it: a run that never mentions h2o would not resolve it.
            f.write('control:\n  title: case\n  note: h2o fixture\n')

        config_path = self._write_config({
            'title': 'constraint_md_periodic_xh_ethanol_example',
            'theory': ['lammps'],
            'lammps_project': 'ethanol',
        })
        _stage_example_skill_fixtures(config_path, self.run_dir, example_path)

        staged = set(os.listdir(self.run_dir))
        self.assertIn('h2o.data', staged)
        for name in ('ethanol.data', 'ethanol.in.init', 'ethanol.in.settings'):
            path = os.path.join(self.run_dir, name)
            self.assertIn(name, staged)
            with open(path, 'r') as f:
                self.assertEqual(f.read(), 'ETHANOL\n', name)


def _skipped_for(by_name, stem, requirement):
    """The named-skip record of `stem`, when the host lacks `requirement`.

    A host without TensorFlow plans the whole nn chain as one skip record per
    example (a skip has no carrier, so it carries the bare `example_<stem>`
    name and no run directory). The chain tests below assert how the carriers
    are laid out, which only exists when the chain is planned; this helper is
    how they tell the two situations apart instead of indexing a missing key.
    """
    task = by_name.get('example_%s' % stem)
    if (task and task.get('skip')
            and requirement in str(task.get('skip_reason', ''))):
        return task
    return None


class TestExampleChains(unittest.TestCase):
    def _require_nn_carriers(self, plan):
        """Skip when this host cannot plan the nn chain's carriers.

        A host without TensorFlow reports every nn example as a named skip,
        which is the correct plan (the requirement gate names the package);
        what these tests assert -- two carriers sharing or not sharing one
        directory -- has nothing to observe there. Skipping with the reason is
        the honest report; asserting the skip instead would claim coverage the
        host never had.
        """
        by_name = {task['name']: task for task in plan['example_tests']}
        skipped = _skipped_for(by_name, 'nn_train', 'tensorflow')
        if skipped is not None:
            self.skipTest('tensorflow is not importable on this host: %s'
                          % skipped['skip_reason'])

    """`nn_train` and `nn_predict` are one chain: one directory, one order.

    Every other example gets a directory per (example, mode), which is
    what keeps a failure's own log. A chain is the deliberate exception:
    its second member reads the model folder its first member wrote, and
    an alphabetical listing would run the reader first.
    """

    def test_the_shipped_ml_pair_is_a_declared_chain(self):
        self.assertEqual(('nn_train', 'nn_predict'), EXAMPLE_CHAINS['nn'])
        self.assertEqual('nn', _example_family('nn_train'))
        self.assertEqual('nn', _example_family('nn_predict'))
        self.assertIsNone(_example_family('nn_grid_search'))
        self.assertIsNone(_example_family('energy'))

    def test_a_chain_runs_in_the_declared_order_not_alphabetical(self):
        listing = ['nn_predict.yaml', 'nn_grid_search.yaml', 'energy.yaml',
                   'nn_train.yaml']
        ordered = [os.path.splitext(f)[0] for f in _order_example_files(listing)]
        self.assertLess(ordered.index('nn_train'),
                        ordered.index('nn_predict'),
                        'the evaluation was planned before the fit that '
                        'writes what it reads')

    def test_everything_else_keeps_alphabetical_order(self):
        listing = ['gradient.yaml', 'energy.yaml', 'nn_train.yaml',
                   'nn_predict.yaml']
        self.assertEqual(['energy.yaml', 'gradient.yaml', 'nn_train.yaml',
                          'nn_predict.yaml'], _order_example_files(listing))

    def test_a_chain_member_that_is_absent_is_simply_skipped(self):
        # Planning must not invent a file that is not on disk: the order is
        # built from what was discovered.
        self.assertEqual(['nn_predict.yaml'],
                         _order_example_files(['nn_predict.yaml']))

    def test_the_shipped_examples_are_planned_into_the_shared_directory(
            self):
        config = load_config(os.path.join(os.path.dirname(
            manager_module.__file__), 'pmd_env.yaml'))
        # The chain's directory layout is a mechanism, not a switchboard
        # policy: the shipped file now closes the nn examples on purpose for
        # the optimize round, and a switch-closed task carries no run_dir.
        config['_runall'] = True
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work, True)
        plan = manager_module.generate_plan(config, work)
        self._require_nn_carriers(plan)
        cases = {t['name']: t for t in plan['example_tests']
                 if t['name'].startswith('example_nn')}
        runs = {name: t['run_dir'] for name, t in cases.items()
                if 'run_dir' in t}
        self.assertEqual(runs['example_nn_train_yaml'],
                         runs['example_nn_predict_yaml'])
        self.assertNotEqual(runs['example_nn_train_yaml'],
                            runs['example_nn_grid_search_local_yaml'])
        self.assertTrue(runs['example_nn_train_yaml'].endswith(os.path.join(
            'examples', 'nn', 'yaml')))

    def test_the_remote_arm_is_a_named_skip_without_a_scheduler(self):
        # The scanning example is shipped remote-first: it asks Slurm for its
        # child runtimes, so on a host with no cluster client it is skipped by
        # name and no directory is claimed for it. A skip record that still
        # carried a run directory would be executed by any runner that keys on
        # the directory rather than on the flag.
        config = load_config(os.path.join(os.path.dirname(
            manager_module.__file__), 'pmd_env.yaml'))
        # Forced for the same reason as the test above: this one judges the
        # host gate, not whether the round happens to want the nn examples.
        config['_runall'] = True
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work, True)
        plan = manager_module.generate_plan(config, work)
        self._require_nn_carriers(plan)
        cases = {t['name']: t for t in plan['example_tests']
                 if t['name'].startswith('example_nn_grid_search')}
        local = cases['example_nn_grid_search_local_yaml']
        remote = cases.get('example_nn_grid_search')
        if remote is None:
            remote = cases['example_nn_grid_search_yaml']
        # The example the gate judged is the YAML it read: a skip record names
        # it in 'input_file', a planned record keeps it in 'spec_file' beside
        # the generated classic input.
        remote_yaml = remote.get('spec_file', remote['input_file'])
        self.assertTrue(_example_submits_remotely(remote_yaml))
        self.assertFalse(_example_submits_remotely(local['spec_file']))
        if shutil.which('sbatch') is None:
            # The gate fires before the mode loop, so the remote stem is one
            # named skip rather than two: no classic input is generated for a
            # run this host cannot submit.
            self.assertTrue(remote['skip'])
            self.assertIn('sbatch', remote['skip_reason'])
            self.assertNotIn('run_dir', remote)
            self.assertNotIn('example_nn_grid_search_input', cases)
            self.assertNotIn('skip', local)
        else:
            self.assertNotIn('skip', remote)
            self.assertTrue(os.path.isdir(remote['run_dir']))

    def test_a_chain_shares_only_within_one_mode(self):
        # The yaml carrier and the classic carrier of the same example must
        # still not overwrite each other's products: the chain is per
        # (family, mode).
        config = load_config(os.path.join(os.path.dirname(
            manager_module.__file__), 'pmd_env.yaml'))
        config['_runall'] = True   # mechanism, not round policy
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work, True)
        plan = manager_module.generate_plan(config, work)
        self._require_nn_carriers(plan)
        runs = {t['name']: t['run_dir'] for t in plan['example_tests']
                if t['name'].startswith('example_nn_train') and 'run_dir' in t}
        self.assertNotEqual(runs['example_nn_train_yaml'],
                            runs['example_nn_train_input'])


class TestExampleRequirements(unittest.TestCase):
    """An engine stack the host lacks is a named skip, not a red run."""

    def test_a_missing_package_is_named_by_the_example_that_needs_it(self):
        registry = {'nn': {'name': 'nn', 'skill_name': 'nn_pes_skill',
                           'requires': ['tensorflow']}}
        with mock.patch('importlib.util.find_spec', return_value=None):
            missing = _example_missing_requirements('nn', registry)
        self.assertEqual(['tensorflow'], missing)

    def test_an_available_stack_reports_nothing_missing(self):
        registry = {'nn': {'name': 'nn', 'skill_name': 'nn_pes_skill',
                           'requires': ['numpy']}}
        self.assertEqual([], _example_missing_requirements('nn', registry))

    def test_an_unknown_label_is_not_a_missing_requirement(self):
        # An example whose theory string names no installed skill is the
        # skill layer's problem to report, not this gate's to invent.
        self.assertEqual([], _example_missing_requirements('nosuch', {'nn': {}}))


class TestEveryExampleAnswersToOneBudget(unittest.TestCase):
    """The per-example timeout map is gone; `_timeout` is the only budget."""

    def test_the_cp2k_strain_example_is_planned_with_the_shared_budget(self):
        # The per-example `_example_timeout` map was removed on 2026-10-20.
        # It existed for `optimize_cell_gap_cp2k` (thirteen CP2K calls a step)
        # and round tmp_20261019_200652 measured both of its legs inside the
        # shared 1200 s backstop - 735 s and 728 s - once each of its files
        # declared two workers. The criterion reads the planned task's own
        # `timeout`, which is the number the script embeds and deploy sizes
        # its poll from, and it refuses a map that comes back: a second table
        # is a decision, not something to reintroduce silently.
        import warnings
        shipped = os.path.join(os.path.dirname(
            manager_module.__file__), 'pmd_env.yaml')
        config = load_config(shipped)
        self.assertNotIn('_example_timeout', config)
        # The budget's own number is not pinned here: `pmd_env.yaml` states that
        # a case which needs more room gets `_timeout` raised, and #458 raised it
        # from 1200 to 2400 for the four single-center multi-step periodic arms
        # (round 18 measured those two MD legs at about 20 min). What this
        # criterion holds is the single source - the plan resolves to the file's
        # one number, and no per-example table may come back.
        self.assertEqual(int(config['_timeout']), _resolve_timeout(config))
        shared = int(config['_timeout'])
        config['_runall'] = True

        examples = os.path.abspath(os.path.join(
            os.path.dirname(manager_module.__file__), os.pardir, 'examples'))
        work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work, True)
        only = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, only, True)
        shutil.copy(os.path.join(examples, 'optimize_cell_gap_cp2k.yaml'),
                    only)
        # The one-file directory makes the shipped example switches unknown;
        # that warning is about this fixture, not about the budget.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tasks = manager_module._plan_example_tests(
                config, work, examples_dir=only)
        planned = {task['name']: task['timeout'] for task in tasks}
        self.assertEqual(
            {'example_optimize_cell_gap_cp2k_yaml': shared,
             'example_optimize_cell_gap_cp2k_input': shared}, planned)


class TestExampleJsonFixtures(unittest.TestCase):
    """A training file beside an example is staged; a generated one is not."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.example_dir = os.path.join(self.tmp, 'examples')
        self.run_dir = os.path.join(self.tmp, 'run')
        os.makedirs(self.example_dir)
        os.makedirs(self.run_dir)

    def test_the_label_fixture_ships_in_the_examples_directory(self):
        examples = os.path.join(os.path.dirname(manager_module.__file__),
                                os.pardir, 'examples')
        self.assertTrue(os.path.isfile(os.path.join(examples,
                                                    'egs_small.json')))

    def test_a_json_beside_the_example_is_staged_under_its_own_name(self):
        with open(os.path.join(self.example_dir, 'egs_small.json'), 'w') as h:
            h.write('{}')
        yaml_path = os.path.join(self.example_dir, 'nn_train.yaml')
        # Staging follows a name this example can resolve, and a training file
        # is named by its own input rather than by a project - which is why the
        # text, not only the project keys, is consulted.
        with open(yaml_path, 'w', encoding='utf-8') as h:
            h.write('nn:\n  train_data: egs_small.json\n')
        config_path = os.path.join(self.tmp, 'case.json')
        with open(config_path, 'w') as handle:
            json.dump({'title': 'nn_train', 'theory': ['nn'],
                       'nn_project': 'NN_EG'}, handle)
        _stage_example_skill_fixtures(config_path, self.run_dir, yaml_path)
        staged = os.listdir(self.run_dir)
        self.assertIn('egs_small.json', staged)
        # The project-renamed copy exists for engine topologies, where the
        # backend resolves `<project>.data`. A training file is named by
        # the input itself, so renaming it would duplicate 80 KB of labels
        # into a second name nothing reads.
        self.assertNotIn('NN_EG.json', staged)

    def test_the_json_suffix_belongs_to_examples_only(self):
        self.assertIn('.json', _EXAMPLE_FIXTURE_SUFFIXES)
        self.assertNotIn('.json', _TOPOLOGY_SUFFIXES)



class TestOptimizeFamilySwitch(unittest.TestCase):
    """The optimize family decides by name, and by nothing else."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patchers = [
            mock.patch.object(manager_module, 'yaml_to_config',
                              side_effect=_fake_yaml_to_config),
            mock.patch.object(manager_module, '_planning_worker_count',
                              return_value=1),
            mock.patch.object(manager_module, 'convert_to_classic_input'),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self._patchers:
            patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _examples_dir():
        # Where the planner itself looks: the package's examples sit beside the
        # qualification package, not inside it.
        return os.path.abspath(os.path.join(
            os.path.dirname(manager_module.__file__), os.pardir, 'examples'))

    @classmethod
    def _optimize_stems(cls):
        """Every shipped example the family is meant to cover."""
        from pyrai3md.qualification.manager import (
            EXAMPLE_FAMILIES, family_members)
        return set(family_members(EXAMPLE_FAMILIES['optimize_test']))

    @staticmethod
    def _stem(task):
        # A switch-skipped task never reaches the converter, so it carries the
        # YAML only; the stem has to come from the file either way. Stripping
        # the carrier suffix off `task['name']` mis-reads the single-row skips.
        source = task.get('spec_file') or task['input_file']
        return os.path.splitext(os.path.basename(source))[0]

    def _skipped(self, config):
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        return {task['name']: task.get('skip_reason', '')
                for task in tasks if task.get('skip')}, tasks

    def test_optimize_family_opens_cases_the_file_disabled(self):
        # `optimize_test: true` is decisive over the per-example keys, the way
        # the two older families already are.
        stems = sorted(self._optimize_stems())
        self.assertTrue(stems, 'no shipped example starts with optimize_, so '
                               'this family would be switching an empty set')
        three = stems[:3]
        flags = {'optimize_test': True}
        flags.update({stem: False for stem in three})
        config = {'_runall': SWITCH_CONDITION, 'examples': flags}
        skipped, _tasks = self._skipped(config)
        for stem in three:
            self.assertNotIn('example_' + stem, skipped,
                             'the family opened everything except %s: %s'
                             % (stem, sorted(skipped)))

    def test_optimize_family_false_closes_every_case_by_its_own_name(self):
        # Every `optimize_cell_*` case now reads `true` on its own key, and the
        # family still closes all of them: the family is consulted before the
        # per-example switch, so the reason has to name `examples.optimize_test`
        # rather than the case key that did not get a vote. A stage that lost
        # that ordering would report "disabled by examples.optimize_cell_xtb"
        # for a switch the operator never touched.
        config = {'_runall': SWITCH_CONDITION,
                  'examples': {'optimize_test': False}}
        skipped, _tasks = self._skipped(config)
        stems = self._optimize_stems()
        self.assertIn('optimize_cell_xtb', stems,
                      'the shipped set no longer carries the case this '
                      'assertion is about')
        for stem in sorted(stems):
            name = 'example_' + stem
            self.assertIn(name, skipped, name)
            self.assertEqual(skipped[name],
                             'disabled by examples.optimize_test', name)

    def test_optimize_prefix_shadows_no_other_family(self):
        # Membership is `startswith` and the first match wins, so the table has
        # to stay pairwise disjoint - a `constraint_` prefix would swallow the
        # 33 `constraint_md_*` cases and the two individual `constraint_opt_*`
        # ones with it.
        prefixes = list(EXAMPLE_FAMILIES.values())
        for first in prefixes:
            for second in prefixes:
                if first != second:
                    self.assertFalse(second.startswith(first),
                                     '%s shadows %s' % (first, second))
        stems = {os.path.splitext(fname)[0]
                 for fname in os.listdir(self._examples_dir())
                 if fname.endswith('.yaml')}
        for stem in stems:
            hits = [name for name, prefix in EXAMPLE_FAMILIES.items()
                    if stem.startswith(prefix)]
            self.assertLessEqual(len(hits), 1, (stem, hits))
        # The two constraint relaxations stay individual, whatever the family
        # says about the optimize name.
        families = {name: True for name in EXAMPLE_FAMILIES}
        for stem in ('constraint_opt_periodic', 'constraint_opt_surface'):
            self.assertEqual(
                manager_module._example_state(SWITCH_CONDITION, families,
                                              stem),
                (SWITCH_CONDITION, ''), stem)
        # And the decision does not depend on the row order of the table.
        import itertools
        for permutation in itertools.permutations(EXAMPLE_FAMILIES.items()):
            ordered = dict(permutation)
            for stem in stems:
                self.assertEqual(
                    manager_module._example_state(SWITCH_CONDITION, ordered,
                                                  stem),
                    manager_module._example_state(SWITCH_CONDITION,
                                                  EXAMPLE_FAMILIES, stem),
                    (stem, permutation))

    def test_optimize_family_is_not_a_run_directory_chain(self):
        # Two words called "family" live in this module: the switch table above
        # and EXAMPLE_CHAINS, which shares one run directory between chained
        # examples. Putting the optimize set in the second would push 24 tasks
        # into one directory.
        for stem in self._optimize_stems():
            self.assertIsNone(_example_family(stem), stem)
        members = set()
        for chain in EXAMPLE_CHAINS.values():
            members.update(chain)
        self.assertFalse(self._optimize_stems() & members,
                         'an optimize example is a chain member: %s'
                         % sorted(self._optimize_stems() & members))


class TestShippedOptimizeRoundStage(unittest.TestCase):
    """That the shipped pmd_env.yaml's switches still mean what they say.

    Which switches are open is a working state, not a property of the code:
    a round closes what it does not run and reopens it later. Pinning a
    particular posture here made every scope change look like a regression
    (the openqp-only round of 2026-10-23 tripped four of these assertions
    while the code was fine), so the tests below read the file's own switches
    by the documented order and require the plan to agree with them - in
    whatever posture the file is in. The two ends stay pinned: `_runall`
    overrides the lot in either direction, and every shipped example carries
    a key so a new case cannot join a round by omission.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patchers = [
            mock.patch.object(manager_module, 'yaml_to_config',
                              side_effect=_fake_yaml_to_config),
            mock.patch.object(manager_module, '_planning_worker_count',
                              return_value=1),
            mock.patch.object(manager_module, 'convert_to_classic_input'),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self._patchers:
            patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _shipped_config():
        # The explicit path keeps the ~/.pyrai3md and working-directory layers
        # out: this is a claim about the file that ships, not about this host.
        return load_config(manager_module.DEFAULT_CONFIG)

    @staticmethod
    def _stem(task):
        source = task.get('spec_file') or task['input_file']
        return os.path.splitext(os.path.basename(source))[0]

    def _shipped_stems(self):
        examples_dir = TestOptimizeFamilySwitch._examples_dir()
        return {os.path.splitext(fname)[0]
                for fname in os.listdir(examples_dir)
                if fname.endswith('.yaml')}

    def _switch_closed(self, tasks):
        return {self._stem(task) for task in tasks
                if str(task.get('skip_reason', '')).startswith('disabled by')}

    def _opened_stems(self, config):
        shipped = self._shipped_stems()
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        return shipped - self._switch_closed(tasks), tasks

    def test_every_switch_in_the_shipped_file_is_a_legal_value(self):
        # The file's posture is a working state: a round closes what it does
        # not run and reopens it later, so nothing here pins which switches are
        # on. What has to hold in every posture is that each value is one the
        # loader understands - a typo is coerced to a default and then reads as
        # "the round decided so", which is the failure this catches.
        config = self._shipped_config()
        self.assertIn(_runall_mode(config), (True, False, SWITCH_CONDITION),
                      '_runall is not one of the three states')
        flags = config.get('examples') or {}
        for name in EXAMPLE_FAMILIES:
            if name in flags:
                self.assertIn(_coerce_switch(flags[name], name, SWITCH_CONDITION),
                              (True, False, SWITCH_CONDITION), name)
        for stem in self._shipped_stems():
            if stem in flags:
                self.assertIsInstance(_coerce_example_flag(flags[stem], stem),
                                      bool, stem)
        self.assertIsInstance(_coerce_example_flag(True, 'toggle'), bool)

    def test_the_example_plan_opens_exactly_what_the_example_switches_ask_for(self):
        # Independent of which round the file is set to: read the switches by
        # the documented order (master, then family, then the example's own
        # key) and require the plan to agree. A round that closes everything
        # and a round that opens thirty examples both satisfy this; a planner
        # that ignores a switch does not.
        config = self._shipped_config()
        flags = config.get('examples') or {}
        master = _runall_mode(config)

        def wants_open(stem):
            if master != SWITCH_CONDITION:
                return master is True
            for name, prefix in EXAMPLE_FAMILIES.items():
                if stem.startswith(prefix):
                    state = _coerce_switch(flags.get(name, SWITCH_CONDITION),
                                           name, SWITCH_CONDITION)
                    if state != SWITCH_CONDITION:
                        return state is True
                    break
            return _coerce_example_flag(flags.get(stem, True), stem) is True

        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        by_stem = {self._stem(task): task for task in tasks}
        shipped = self._shipped_stems()
        self.assertEqual(set(by_stem), shipped,
                         'examples without a planned row: %s'
                         % sorted(shipped - set(by_stem)))
        # Only the switch-driven closures are compared: an example may still
        # be skipped for a reason of its own (md_batch wants sbatch on this
        # host, a postponed case names its policy), and that is not the switch
        # table's business.
        closed = {stem for stem, task in by_stem.items()
                  if str(task.get('skip_reason', '')).startswith('disabled by')}
        expected_closed = {stem for stem in shipped if not wants_open(stem)}
        self.assertEqual(closed, expected_closed,
                         'the switch closure disagrees with the switches: '
                         'closed but wanted %s, wanted but not closed %s'
                         % (sorted(closed - expected_closed),
                            sorted(expected_closed - closed)))
        opened = {stem for stem, task in by_stem.items()
                  if not task.get('skip')}
        self.assertTrue(opened <= {stem for stem in shipped if wants_open(stem)},
                        'the plan ran a switched-off example: %s'
                        % sorted(opened - {stem for stem in shipped
                                           if wants_open(stem)}))

    def test_every_shipped_example_has_a_switch(self):
        # A per-case key that is absent defaults to `true`
        # (`manager._coerce_example_flag`), so an example added without one
        # silently joins every round. The reverse direction already warns at
        # plan time (`_warn_unknown_example_keys`); this is the direction that
        # would ship unnoticed, and it is what adding ten path examples at once
        # would have got wrong.
        flags = self._shipped_config().get('examples') or {}
        missing = sorted(stem for stem in self._shipped_stems()
                         if stem not in flags)
        self.assertEqual(missing, [],
                         'shipped examples with no examples: key: %s'
                         % missing)

    def test_the_skill_plan_opens_exactly_what_the_skill_switches_ask_for(self):
        # Same bargain one level down: `skills.<name>.test` decides the case
        # rows (the master overriding it), its `tests:` map speaks only when
        # `test` defers, and `runtime_tests` alone speaks for the pytest phase.
        # Read that off the file and require the plan to agree, so certifying
        # an engine for a round is a switch edit rather than a test edit.
        from pyrai3md.qualification.certification.planner import (
            plan_skill_tests,
        )
        config = self._shipped_config()
        master = _runall_mode(config)
        skills = config.get('skills') or {}
        self.assertTrue(skills, 'the shipped file declares no skill')
        tasks = plan_skill_tests(config, self.tmpdir)
        for name, section in sorted(skills.items()):
            self.assertIn('test', section,
                          '%s states no `test` switch' % name)
            self.assertIn('runtime_tests', section,
                          '%s states no `runtime_tests` switch' % name)
            rows = [task for task in tasks if task.get('skill') == name]
            if not rows:
                continue
            state = master if master != SWITCH_CONDITION else _coerce_switch(
                section.get('test', False), 'skills.%s.test' % name, False)
            force_all = state is True
            case_flags = section.get('tests') or {}
            for task in rows:
                mode = task.get('mode')
                if mode == 'skill_case':
                    stem = self._stem(task)
                    if force_all:
                        want = True
                    elif state is False:
                        want = False
                    else:
                        want = _coerce_flag(case_flags.get(stem, True),
                                            stem, True, 'case') is True
                elif mode == 'skill_runtime':
                    want = force_all or _coerce_flag(
                        section.get('runtime_tests', True),
                        'skills.%s.runtime_tests' % name, True,
                        'case') is True
                else:
                    continue
                self.assertEqual(
                    not task.get('skip'), want,
                    '%s is %s where its switches say %s: %s'
                    % (task.get('name'),
                       'skipped' if task.get('skip') else 'open',
                       'open' if want else 'skipped',
                       task.get('skip_reason')))

    def test_the_stage_config_plans_without_an_unknown_key_warning(self):
        # The switch table and the file have to land together: a family named
        # in the YAML that the table does not know is ignored, warns, and the
        # round still plans the whole 168-row example set.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            manager_module._plan_example_tests(self._shipped_config(),
                                               self.tmpdir)
        messages = [str(item.message) for item in caught]
        self.assertFalse([message for message in messages
                          if 'optimize_test' in message], messages)

    def test_runall_false_defeats_the_optimize_family(self):
        # The trap the three-level order carries: `false` on the master closes
        # everything and is consulted before the families, so a round switched
        # off this way runs no example however the family key reads.
        config = self._shipped_config()
        config['_runall'] = False
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        closed = self._switch_closed(tasks)
        self.assertEqual(closed, self._shipped_stems())
        self.assertFalse([task for task in tasks
                          if 'optimize_test' in str(task.get('skip_reason'))],
                         'a family spoke under a closed master')

    def test_runall_true_restores_the_full_round_from_the_stage_file(self):
        # "One key back" made executable: the stage values are inert under the
        # master, so nothing has to be undone in the file to run everything.
        config = self._shipped_config()
        config['_runall'] = True
        tasks = manager_module._plan_example_tests(config, self.tmpdir)
        self.assertEqual(self._switch_closed(tasks), set(),
                         'a switch still closed an example under the master')
        opened, _tasks = self._opened_stems(config)
        self.assertEqual(opened, self._shipped_stems())


if __name__ == '__main__':
    unittest.main()
