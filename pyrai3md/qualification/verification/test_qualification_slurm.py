"""
======================================================================
Test suite for the skill slurm self-submission qualification path
(scheduler x execution_mode test matrix) and the certification chain
(planner -> handler -> converter).

Tests cover:
- planner: override-semantics injection of execution_mode/slurm_*
  (the test YAMLs hardcode execution_mode=local, so fill-only
  injection could never flip the switch)
- planner: plan-time sbatch precheck producing a diagnostic skip
  when execution_mode=slurm but sbatch is unavailable
- planner: skill_slurm marking of all three skill test modes
- handler: test file discovery of each skill's tests directory
- converter: unified test YAML parsing (the former per-skill
  dev_tool): yaml name -> title/project, current_state nesting,
  lattice_vector survival
- converter: execution_mode/slurm_* written into the classic input
  skill section (&MOLCAS/&BAGEL/&PYSCF), never &ENVIRONMENT
- converter: dynamic schema key sync through the skill query
  interface
- runner: artifact authenticity assertion for skill_slurm tasks and
  harness sbatch script exclusion in the artifact scan
- runner: truncated optimize tests ending unconverged are
  PASS[NOT CONVERGED]
- runner: the run-log status check resolves the log by the task's
  run_dir and control.title for examples, by the input stem for
  skill tests
- framework converter rendering (&MD root/activestate), jobtype
  vocabulary and the normalize current_state derivation

The runtime test scripts stay in each skill's own tests/ directory;
they are discovered by the handler and executed in place by pytest.

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import unittest
from unittest import mock

from pyrai3md.qualification import runner
from pyrai3md.qualification.manager import SWITCH_CONDITION
from pyrai3md.qualification.certification import handler, planner
from pyrai3md.qualification.certification.converter import (
    _MD_ECHO_KEYS, _build_control_section, _build_md_section,
    _build_molecule_section,
    _collapse_shell_line, _derive_project_name, _resolve_jobtype,
    convert, convert_to_pyrai_config, parse_test_yaml,
)
from pyrai3md.qualification.certification.handler import (
    _EXECUTION_KEYS, _build_skill_section, _skill_defaults,
)
from pyrai3md.schema.validate import parse_jobtype
from pyrai3md.workflow import JOBTYPE_MAP, WORKFLOW_REGISTRY

class TestExecutionControlInjection(unittest.TestCase):
    """Injection of skills.<name> values into generated configs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_config(self, name, data):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return path

    def _read_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_execution_control_keys_override_existing_values(self):
        """execution_mode/slurm_* must win over the test YAML values."""
        path = self._write_config('casscf_energy.json', {
            'execution_mode': 'local',      # hardcoded by the test YAML
            'slurm_partition': 'old_part',
            'molcas_path': '',
        })
        skill_env = {
            'test': True,                    # never injected
            'molcas_path': '/opt/openmolcas',
            'execution_mode': 'slurm',
            'slurm_partition': 'gpu',
            'slurm_time': '01:00:00',
            'slurm_ntasks': '',              # empty values never inject
        }

        planner._inject_env_into_configs({'casscf_energy': path}, skill_env)

        config = self._read_config(path)
        self.assertEqual(config['execution_mode'], 'slurm')
        self.assertEqual(config['slurm_partition'], 'gpu')
        self.assertEqual(config['slurm_time'], '01:00:00')
        self.assertEqual(config['molcas_path'], '/opt/openmolcas')
        self.assertNotIn('slurm_ntasks', config)
        self.assertNotIn('test', config)

    def test_non_control_keys_keep_fill_semantics(self):
        """Existing non-empty machine parameters are never overwritten."""
        path = self._write_config('casscf_energy.json', {
            'molcas_path': '/existing/molcas',
            'molcas_mem': 8000,
        })
        skill_env = {
            'molcas_path': '/opt/openmolcas',
            'molcas_mem': 4000,
        }

        planner._inject_env_into_configs({'casscf_energy': path}, skill_env)

        config = self._read_config(path)
        self.assertEqual(config['molcas_path'], '/existing/molcas')
        self.assertEqual(config['molcas_mem'], 8000)


class TestSkillDriverForcing(unittest.TestCase):
    """Skill test configs are forced to driver='skill'; only the main
    framework examples run driver=main."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_config(self, name, data):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return path

    def _read_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_missing_driver_is_injected(self):
        path = self._write_config('casscf_energy.json',
                                  {'jobtype': 'energy'})
        planner._force_skill_driver({'casscf_energy': path})
        self.assertEqual(self._read_config(path)['driver'], 'skill')

    def test_existing_driver_is_overridden(self):
        # Unconditional override: a test YAML carrying a driver value
        # can never opt out of the skill-driven execution model.
        path = self._write_config('casscf_energy.json',
                                  {'driver': 'main'})
        planner._force_skill_driver({'casscf_energy': path})
        self.assertEqual(self._read_config(path)['driver'], 'skill')

    def test_converter_writes_driver_into_control(self):
        section = _build_control_section(
            {'jobtype': 'energy', 'driver': 'skill'}, 'casscf_energy')
        self.assertIn('driver        skill', section)

    def test_examples_without_driver_stay_main(self):
        section = _build_control_section(
            {'jobtype': 'energy'}, 'energy_example')
        self.assertNotIn('driver', section)


class TestSkillSlurmPlanning(unittest.TestCase):
    """Plan-time sbatch precheck and skill_slurm task marking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patches = [
            mock.patch.object(planner, 'discover_all', return_value={}),
            mock.patch.object(planner, 'list_skills',
                              return_value=['molcas_skill']),
            mock.patch.object(planner.handler, 'list_skill_runtime_tests',
                              return_value=[]),
            # The skills layer's own tests are not this skill's rows: they are
            # planned once for the round, and
            # TestSharedSkillLayerTestsArePlanned covers that group. This class
            # plans one skill's switch matrix, so the group stays out of its
            # task lists.
            mock.patch.object(planner.handler, 'list_shared_skill_tests',
                              return_value=[]),
            # The closed-case listing reads the skill's YAML names through the
            # handler, without generating anything for them. Pinned to the same
            # one case the fake converter returns, so a plan of the same skill
            # differs only by which switch closed it.
            mock.patch.object(planner.handler, 'list_skill_test_yamls',
                              side_effect=lambda name: [os.path.join(
                                  self.tmpdir, 'casscf_energy.yaml')]),
            mock.patch.object(planner, 'generate_test_configs',
                              side_effect=self._fake_generate),
            mock.patch.object(planner, 'convert_to_classic_input',
                              side_effect=self._fake_convert),
            mock.patch.object(planner, 'convert_to_pyrai_config',
                              side_effect=self._fake_pyrai_convert),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _fake_generate(skill_name, skill_dir):
        path = os.path.join(skill_dir, 'casscf_energy.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'profile': 'casscf_energy',
                'title': 'casscf_energy',
                'execution_mode': 'local',
            }, f)
        return {'casscf_energy': path}

    @staticmethod
    def _write_case_config(path, title=None):
        """Create the generated config a mocked converter names."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'title': title or os.path.splitext(
                os.path.basename(path))[0]}, f)

    @staticmethod
    def _fake_convert(config_path, input_path):
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write('&CONTROL\ntitle fake\n')

    @staticmethod
    def _fake_pyrai_convert(config_path, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'control': {'title': 'fake'}}, f)

    def _plan(self, skill_env, sbatch_path, runall=SWITCH_CONDITION,
              scheduler=''):
        with mock.patch.object(planner.shutil, 'which',
                               return_value=sbatch_path):
            config = {
                '_runall': runall,
                'environment': {'scheduler': scheduler},
                'skills': {'molcas_skill': skill_env},
            }
            return planner.plan_skill_tests(config, self.tmpdir)

    def test_test_false_skill_is_listed_as_skipped_cases(self):
        # A closed skill still shows what it closed: a suite that runs 0 of
        # the certified cases of a skill cannot be told apart from one that
        # certified none, and the reason has to name the key to edit.
        tasks = self._plan({'test': False}, sbatch_path=None)
        self.assertEqual([task['name'] for task in tasks],
                         ['molcas_skill_casscf_energy'])
        self.assertTrue(all(task.get('skip') for task in tasks))
        self.assertEqual(tasks[0]['skip_reason'],
                         'disabled by skills.molcas_skill.test')
        # Nothing is generated for a case that will not run.
        self.assertNotIn('script', tasks[0])

    def test_a_closed_skill_still_runs_its_engine_free_pytest_phase(self):
        # #359: `skills.<name>.test` certifies the YAML cases, which need the
        # engine. The skill's own tests/test_*.py files are code checks that
        # need none, so closing the skill must not close them - that is what
        # keeps "local only" from being the standing state of the package.
        with mock.patch.object(planner.handler, 'list_skill_runtime_tests',
                               return_value=[os.path.join(
                                   self.tmpdir, 'test_skill.py')]):
            tasks = self._plan({'test': False,
                                'runtime_tests': True}, sbatch_path=None)
        by_mode = {task['mode']: task for task in tasks}
        self.assertIn('skill_case', by_mode, sorted(by_mode))
        self.assertTrue(by_mode['skill_case'].get('skip'))
        self.assertEqual(by_mode['skill_case']['skip_reason'],
                         'disabled by skills.molcas_skill.test')
        runtime = by_mode['skill_runtime']
        self.assertNotIn(True, [runtime.get('skip')],
                         'the pytest phase was closed by the certification '
                         'switch it does not answer to')
        self.assertIn('molcas_skill_test_skill', [runtime['name']])
        # A closed runtime phase still names its own key, not the test switch.
        with mock.patch.object(planner.handler, 'list_skill_runtime_tests',
                               return_value=[os.path.join(
                                   self.tmpdir, 'test_skill.py')]):
            closed = self._plan({'test': False,
                                 'runtime_tests': False}, sbatch_path=None)
        rows = [task for task in closed if task['mode'] == 'skill_runtime']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].get('skip'))
        self.assertEqual(rows[0]['skip_reason'],
                         'disabled by skills.molcas_skill.runtime_tests')

    def test_master_false_closes_the_skill_from_above(self):
        tasks = self._plan({'test': True}, sbatch_path=None, runall=False)
        self.assertTrue(all(task.get('skip') for task in tasks))
        self.assertEqual(tasks[0]['skip_reason'], 'disabled by _runall')

    def test_runtime_tests_switch_lists_the_scripts_it_closed(self):
        # `runtime_tests` closes the pytest phase only: the YAML cases of the
        # same skill are a separate switch and stay planned.
        with mock.patch.object(planner.handler, 'list_skill_runtime_tests',
                               return_value=[os.path.join(
                                   self.tmpdir, 'test_skill.py')]):
            tasks = self._plan({'test': 'condition',
                                'runtime_tests': False}, sbatch_path=None)
        runtime = [task for task in tasks
                   if task['mode'] == 'skill_runtime']
        self.assertEqual([task['name'] for task in runtime],
                         ['molcas_skill_test_skill'])
        self.assertTrue(runtime[0].get('skip'))
        self.assertEqual(runtime[0]['skip_reason'],
                         'disabled by skills.molcas_skill.runtime_tests')
        self.assertTrue(any(not task.get('skip') for task in tasks),
                        'the YAML cases were closed along with the scripts')

    def test_per_case_switch_closes_one_case_of_two(self):
        # `test: condition` is the only setting where the per-case map speaks;
        # an open case keeps its three modes, a closed one is listed once.
        cases = {'casscf_energy': os.path.join(self.tmpdir,
                                               'casscf_energy.json'),
                 'casscf_gradient': os.path.join(self.tmpdir,
                                                 'casscf_gradient.json')}
        for path in cases.values():
            self._write_case_config(path)
        with mock.patch.object(planner, 'generate_test_configs',
                               return_value=dict(cases)):
            tasks = self._plan({'test': 'condition',
                                'tests': {'casscf_gradient': False}},
                               sbatch_path=None)
        skipped = [task for task in tasks if task.get('skip')]
        self.assertEqual([task['name'] for task in skipped],
                         ['molcas_skill_casscf_gradient'])
        self.assertEqual(skipped[0]['skip_reason'],
                         'disabled by skills.molcas_skill.tests.'
                         'casscf_gradient')
        planned = [task for task in tasks if not task.get('skip')]
        self.assertEqual(len(planned), 3)
        self.assertTrue(all(task['name'].startswith('molcas_skill_casscf_energy')
                            for task in planned),
                        [task['name'] for task in planned])

    def test_skill_true_overrides_a_closed_case(self):
        # `true` at the skill level is the promise that every case runs, so
        # the per-case map is not read at all.
        cases = {'casscf_energy': os.path.join(self.tmpdir,
                                               'casscf_energy.json')}
        for path in cases.values():
            self._write_case_config(path)
        with mock.patch.object(planner, 'generate_test_configs',
                               return_value=dict(cases)):
            tasks = self._plan({'test': True,
                                'tests': {'casscf_energy': False}},
                               sbatch_path=None)
        self.assertEqual(len(tasks), 3)
        self.assertFalse(any(task.get('skip') for task in tasks))

    def test_tests_map_that_is_not_a_mapping_is_reported(self):
        with mock.patch.object(planner, 'generate_test_configs',
                               return_value={}) as generated:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self._plan({'test': True, 'tests': True}, sbatch_path=None)
        messages = [str(item.message) for item in caught]
        self.assertTrue(any('skills.molcas_skill.tests is a mapping' in message
                            for message in messages), messages)
        del generated

    def test_runall_forces_test_false_skill_on(self):
        # The _runall master switch certifies every listed skill even
        # when its own test switch is off.
        tasks = self._plan({'test': False}, sbatch_path=None,
                           runall=True)
        self.assertEqual(len(tasks), 3)
        modes = sorted(task['mode'] for task in tasks)
        self.assertEqual(
            modes, ['pyrai3md_agent', 'pyrai3md_input', 'skill_agent'])

    def test_skill_tasks_anchor_at_their_own_leg_directory(self):
        # A skill case keeps its generated config in the skill directory and
        # runs in a directory of its own per carrier leg, under the case
        # group's directory. That split is what keeps the three legs of one
        # case from colliding, and the group directory is what keeps
        # independent cases from colliding when a slurm round runs them as
        # separate jobs (`00` trains what `01` serves inside one group). The
        # anchor is therefore the group/leg directory named by the task's own
        # mode -- not the example's case/title tree, and `title` stays out of
        # the skill layer.
        tasks = self._plan({'test': True}, sbatch_path=None, runall=True)
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertEqual(
                task['run_dir'],
                os.path.join(os.path.dirname(task['input_file']),
                             'g00', task['mode']))
            self.assertEqual(
                task['job_group'],
                'molcas_skill__g00_casscf_energy__%s' % task['mode'])
            # The case's own declared title, which is the name the framework
            # run log carries in the leg directory. This fixture's generated
            # config does not exist on disk, so the planner falls back to the
            # case stem -- the value a real config's title would replace.
            self.assertEqual(task['title'], 'casscf_energy')

    def test_a_case_title_comes_from_its_generated_config(self):
        # `certification/converter.py` writes the case YAML's `name` into the
        # flat config as `title`, and the framework run names its log after
        # that title; the task has to carry it or the post-run log check looks
        # for a file the run never wrote.
        skill_dir = os.path.join(self.tmpdir, 'molcas_skill')
        os.makedirs(skill_dir, exist_ok=True)
        config_path = os.path.join(skill_dir, 'casscf_energy.json')
        with open(config_path, 'w', encoding='utf-8') as handle:
            json.dump({'title': 'casscf_energy_test'}, handle)
        with mock.patch.object(planner, 'generate_test_configs',
                               return_value={'casscf_energy': config_path}):
            tasks = self._plan({'test': True}, sbatch_path=None, runall=True)
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertEqual(task['title'], 'casscf_energy_test')

    def test_each_leg_runs_its_own_copy_of_the_config(self):
        # A skill agent names its log after the config file it is handed, so
        # three legs launched from one shared config wrote one shared log and
        # overwrote each other's status. Every leg now runs the copy sitting in
        # its own directory, and the launcher has to say so: the per-leg path
        # present, the shared path absent.
        suffix = {'skill_agent': '.json', 'pyrai3md_agent': '.config',
                  'pyrai3md_input': '.input'}
        tasks = self._plan({'test': True}, sbatch_path=None, runall=True)
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            with open(task['script'], encoding='utf-8') as handle:
                script = handle.read()
            leg = task['run_dir']
            shared = os.path.dirname(task['input_file'])
            stem = os.path.splitext(os.path.basename(task['input_file']))[0]
            own = os.path.join(leg, stem + suffix[task['mode']])
            self.assertTrue(os.path.isfile(own), own)
            self.assertIn('cd %s &&' % leg, script, task['name'])
            self.assertIn(own, script, task['name'])
            self.assertNotIn(own.replace(leg, shared), script, task['name'])

    def test_a_leg_starts_with_the_inputs_a_launch_directory_held(self):
        # Moving the working directory must not move any lookup: a backend
        # resolves its topology by name from the launch directory
        # (`<project>.data`, `<project>.in.init`), and the converter stages
        # those beside the configs in the skill directory. The copy rule is
        # what the round tmp_20261010_133424 caught by its absence -- 24
        # LAMMPS cases died on "topology file not found" the moment the legs
        # were split. Launch scripts are the parent's, so they stay out.
        skill_dir = os.path.join(self.tmpdir, 'molcas_skill')
        os.makedirs(skill_dir, exist_ok=True)
        for name in ('casscf_energy.data', 'casscf_energy.in.init'):
            with open(os.path.join(skill_dir, name), 'w') as handle:
                handle.write('staged\n')
        cases = {'casscf_energy': os.path.join(skill_dir,
                                               'casscf_energy.json')}
        for path in cases.values():
            self._write_case_config(path)
        with mock.patch.object(planner, 'generate_test_configs',
                               return_value=dict(cases)):
            tasks = self._plan({'test': True}, sbatch_path=None, runall=True)
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            leg = task['run_dir']
            for name in ('casscf_energy.data', 'casscf_energy.in.init'):
                self.assertTrue(os.path.isfile(os.path.join(leg, name)),
                                 (leg, name))
            self.assertFalse([entry for entry in os.listdir(leg)
                              if entry.endswith(('.sh', '.sbatch'))],
                             'the launcher belongs to the parent')

    def test_missing_sbatch_produces_diagnostic_skip(self):
        tasks = self._plan(
            {'test': True, 'execution_mode': 'slurm'}, sbatch_path=None)

        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].get('skip'))
        self.assertIn('sbatch', tasks[0]['skip_reason'])

    def test_slurm_mode_marks_all_modes_and_overrides_config(self):
        tasks = self._plan(
            {'test': True, 'execution_mode': 'slurm',
             'slurm_partition': 'gpu'},
            sbatch_path='/usr/bin/sbatch')

        modes = sorted(task['mode'] for task in tasks)
        self.assertEqual(
            modes, ['pyrai3md_agent', 'pyrai3md_input', 'skill_agent'])
        for task in tasks:
            self.assertTrue(task['skill_slurm'])
            # The harness scheduler stays local in this scenario.
            self.assertFalse(task['use_slurm'])

        config_path = os.path.join(
            self.tmpdir, 'molcas_skill', 'casscf_energy.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            injected = json.load(f)
        self.assertEqual(injected['execution_mode'], 'slurm')
        self.assertEqual(injected['slurm_partition'], 'gpu')

    def test_slurm_round_keeps_the_legs_local_and_names_one_group_per_leg(self):
        # environment.scheduler is the round's execution mode: the case
        # becomes one job per carrier leg, so the planner writes no
        # per-leg sbatch. The legs share no files, so they are separate
        # jobs; the skill's own execution_mode is the other switch and
        # is not involved in this plan.
        tasks = self._plan({'test': True}, sbatch_path='/usr/bin/sbatch',
                           runall=True, scheduler='slurm')
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertFalse(task['use_slurm'])
            self.assertNotIn('sbatch_script', task)
            self.assertEqual(
                task['job_group'],
                'molcas_skill__g00_casscf_energy__%s' % task['mode'])
        self.assertEqual(
            len({task['job_group'] for task in tasks}), 3)
        self.assertEqual(
            {os.path.dirname(task['run_dir']) for task in tasks},
            {os.path.join(self.tmpdir, 'molcas_skill',
                          'g00')})

    def test_cases_naming_one_modeldir_share_a_group(self):
        # A case that trains a model writes the directory a later case
        # serves, so the cases naming one modeldir must share a job and its
        # directories; every other case is independent and gets its own.
        skill_dir = os.path.join(self.tmpdir, 'molcas_skill')
        os.makedirs(skill_dir, exist_ok=True)
        cases = {}
        for stem in ('case_train', 'case_serve', 'case_solo'):
            path = os.path.join(skill_dir, stem + '.json')
            payload = {'title': stem}
            if stem != 'case_solo':
                payload['modeldir'] = './nn_eg'
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle)
            cases[stem] = path
        with mock.patch.object(planner, 'generate_test_configs',
                               return_value=cases):
            tasks = self._plan({'test': True}, sbatch_path=None,
                               runall=True)
        self.assertEqual(len(tasks), 9)
        train = next(task for task in tasks
                     if task['name'] == 'molcas_skill_case_train_skill')
        serve = next(task for task in tasks
                     if task['name'] == 'molcas_skill_case_serve_skill')
        solo = next(task for task in tasks
                    if task['name'] == 'molcas_skill_case_solo_skill')
        self.assertEqual(train['job_group'],
                         'molcas_skill__g00_nn_eg__skill_agent')
        self.assertEqual(serve['job_group'], train['job_group'])
        self.assertEqual(train['run_dir'], serve['run_dir'])
        self.assertEqual(os.path.dirname(train['run_dir']),
                         os.path.join(skill_dir, 'g00'))
        self.assertEqual(solo['job_group'],
                         'molcas_skill__g01_case_solo__skill_agent')
        self.assertEqual(os.path.dirname(solo['run_dir']),
                         os.path.join(skill_dir, 'g01'))

    def test_local_mode_keeps_fill_only_behavior(self):
        tasks = self._plan({'test': True}, sbatch_path=None)

        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertFalse(task['skill_slurm'])
        config_path = os.path.join(
            self.tmpdir, 'molcas_skill', 'casscf_energy.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            injected = json.load(f)
        # No execution_mode in pmd_env.yaml: the YAML value survives.
        self.assertEqual(injected['execution_mode'], 'local')

    def test_three_modes_use_suffix_aligned_input_files(self):
        # Contract: skill_agent consumes the flat .json written by the
        # converter, pyrai3md_agent consumes the converter's sectioned
        # .config, and pyrai3md_input consumes the rendered .input.
        tasks = self._plan({'test': True}, sbatch_path=None)

        by_mode = {task['mode']: task for task in tasks}
        self.assertTrue(
            by_mode['skill_agent']['input_file'].endswith(
                'casscf_energy.json'))
        self.assertTrue(
            by_mode['pyrai3md_agent']['input_file'].endswith(
                'casscf_energy.config'))
        self.assertTrue(
            by_mode['pyrai3md_input']['input_file'].endswith(
                'casscf_energy.input'))
        for task in tasks:
            self.assertTrue(os.path.isfile(task['input_file']))


class TestConverterExecutionKeys(unittest.TestCase):
    """execution_mode/slurm_* land in the skill section, not elsewhere."""

    def test_molcas_section_carries_execution_keys(self):
        config = {
            'profile': 'casscf_energy',
            'execution_mode': 'slurm',
            'slurm_partition': 'gpu',
            'slurm_time': '01:00:00',
            'slurm_ntasks': 2,
            'keep_tmp': 1,
        }
        section = _build_skill_section(config, 'molcas_skill')

        self.assertTrue(section.startswith('&MOLCAS'))
        self.assertIn('execution_mode slurm', section)
        self.assertIn('slurm_partition gpu', section)
        self.assertIn('slurm_time     01:00:00', section)
        self.assertIn('slurm_ntasks   2', section)
        self.assertIn('keep_tmp       1', section)
        self.assertNotIn('&ENVIRONMENT', section)

    def test_bagel_and_pyscf_sections_share_the_behavior(self):
        config = {'profile': 'p', 'execution_mode': 'slurm'}
        self.assertIn('execution_mode slurm', _build_skill_section(config, 'bagel_skill'))
        self.assertIn('execution_mode slurm', _build_skill_section(config, 'pyscf_skill'))

    def test_empty_execution_keys_are_omitted(self):
        section = _build_skill_section({'profile': 'casscf_energy'}, 'molcas_skill')
        self.assertNotIn('execution_mode', section)
        self.assertNotIn('slurm_', section)


class TestConverterSchemaKeySync(unittest.TestCase):
    """Config keys matching the skill schema Calculation keywords are
    synced into the skill section dynamically (no hardcoded key
    list), so skill test YAML settings like maxit_md survive the
    config -> input conversion (#108)."""

    def test_skill_defaults_interface_is_available(self):
        defaults = _skill_defaults('molcas_skill')
        self.assertIn('maxit_md', defaults)
        self.assertIn('temperature', defaults)
        self.assertIn('size', defaults)

    def test_an_unknown_skill_is_not_an_empty_table(self):
        # A name with no query module must not read as "this skill declares
        # no keywords": the section would render without any of them and the
        # round would stay green. The planner turns the error into a
        # diagnostic row (see test_handler_defaults_failure).
        with self.assertRaises(ImportError):
            _skill_defaults('no_such_skill')

    def test_dynamics_scalar_keys_land_in_molcas_section(self):
        config = {
            'profile': 'casscf_abinitio_dynamics',
            'method': 'casscf',
            'maxit_md': 10,
            'temperature': 300,
            'size': 20,
        }
        section = _build_skill_section(config, 'molcas_skill')
        self.assertIn('maxit_md       10', section)
        self.assertIn('temperature    300', section)
        self.assertIn('size           20', section)

    def test_list_keys_are_not_written_to_skill_section(self):
        # Molecule-level lists belong to &MOLECULE; text input values
        # are raw strings and would fail skill list validation.
        config = {
            'profile': 'casscf_soc',
            'ci_roots': [2, 1],
            'ci_spin': [0, 2],
            'state_spin': [0, 0],
            'coupling': [[0, 1]],
        }
        section = _build_skill_section(config, 'molcas_skill')
        self.assertNotIn('ci_roots', section)
        self.assertNotIn('ci_spin', section)
        self.assertNotIn('state_spin', section)
        self.assertNotIn('coupling', section)

    def test_skill_owned_int_lists_are_written_as_tokens(self):
        # Skill-owned flat int lists (e.g. target_states) must reach
        # the classic input; the schema adapter converts the tokens
        # back to lists on the way in.
        config = {'profile': 'casscf_prop', 'target_states': [2]}
        section = _build_skill_section(config, 'molcas_skill')
        self.assertIn('target_states  2', section)

    def test_non_schema_keys_stay_out_of_skill_section(self):
        config = {'profile': 'casscf_energy', 'some_agent_key': 5}
        section = _build_skill_section(config, 'molcas_skill')
        self.assertNotIn('some_agent_key', section)

    def test_shell_collapses_to_a_single_semicolon_line(self):
        # Classic input keywords are single-line: a yaml list or a
        # newline-joined shell must land as one ';'-separated line,
        # which the kernel/skill normalizers split back identically.
        commands = ['module load gcc/11.3.0',
                    'module load intel/mkl-mpi']
        expected = ('shell          module load gcc/11.3.0'
                    '; module load intel/mkl-mpi')
        for shell in (commands, '\n'.join(commands)):
            section = _build_skill_section(
                {'profile': 'casscf_energy', 'shell': shell}, 'molcas_skill')
            self.assertIn(expected, section)
            self.assertEqual(section.count('module load'), 2)

    def test_shell_collapse_accepts_all_input_forms(self):
        joined = 'module load a; module load b'
        self.assertEqual(
            _collapse_shell_line(['module load a', 'module load b']),
            joined)
        self.assertEqual(
            _collapse_shell_line('module load a\nmodule load b'), joined)
        self.assertEqual(
            _collapse_shell_line('module load a; module load b'), joined)
        self.assertEqual(_collapse_shell_line(''), '')
        self.assertEqual(_collapse_shell_line([]), '')

    def test_empty_shell_stays_out_of_skill_section(self):
        section = _build_skill_section(
            {'profile': 'casscf_energy', 'shell': ''}, 'molcas_skill')
        self.assertNotIn('shell', section)

    def test_bagel_and_pyscf_share_the_schema_sync(self):
        bagel_defaults = _skill_defaults('bagel_skill')
        pyscf_defaults = _skill_defaults('pyscf_skill')
        self.assertTrue(bagel_defaults)
        self.assertTrue(pyscf_defaults)

        bagel_key = next(k for k, v in bagel_defaults.items()
                         if isinstance(v, int) and k not in (
                             'method', 'basis', 'nact_elec', 'nact_orb',
                             'ras1_orb', 'ras2_orb', 'ras3_orb',
                             'inactive_orb', 'bagel_project',
                             'scratch_path')
                         and k not in _EXECUTION_KEYS)
        section = _build_skill_section(
            {'profile': 'p', bagel_key: 42}, 'bagel_skill')
        self.assertIn(f'{bagel_key:<14} 42', section)

    def test_convert_writes_velocity_file_for_dynamics(self):
        # Inline velocities from the test YAML must be persisted as a
        # same-basename .velo (plain numeric text, the framework
        # title.velo naming convention) so read_coord picks them up.
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        config = {
            'profile': 'casscf_abinitio_dynamics',
            'skill_name': 'molcas_skill',
            'atoms': ['C', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            'velocities': [[1e-05, -2e-05, 1.5e-05],
                           [-1.2e-05, 1.8e-05, -1e-05]],
            'maxit_md': 10,
        }
        config_path = os.path.join(tmpdir, 'dynamics.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        input_path = os.path.join(tmpdir, 'dynamics.input')
        convert(config_path, input_path)

        velocity_path = os.path.join(tmpdir, 'dynamics.velo')
        self.assertTrue(os.path.isfile(velocity_path))
        with open(velocity_path, 'r', encoding='utf-8') as f:
            content = f.read().splitlines()
        self.assertEqual(len(content), 2)
        self.assertEqual(len(content[0].split()), 3)

        with open(input_path, 'r', encoding='utf-8') as f:
            section = f.read()
        self.assertIn('maxit_md       10', section)

    def test_convert_without_velocities_writes_no_velocity_file(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        config = {
            'profile': 'casscf_energy',
            'skill_name': 'molcas_skill',
            'atoms': ['C'],
            'coordinates': [[0.0, 0.0, 0.0]],
        }
        config_path = os.path.join(tmpdir, 'energy.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        input_path = os.path.join(tmpdir, 'energy.input')
        convert(config_path, input_path)

        velocity_path = os.path.join(tmpdir, 'energy.velo')
        self.assertFalse(os.path.exists(velocity_path))


class TestRunnerSkillSlurm(unittest.TestCase):
    """Timeout policy and artifact authenticity assertion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_ok_script(self):
        script = os.path.join(self.tmpdir, 'run.sh')
        with open(script, 'w', encoding='utf-8') as f:
            f.write('#!/bin/bash\nexit 0\n')
        return script

    def _base_task(self, script):
        return {
            'name': 'molcas_skill_casscf_energy_agent',
            'mode': 'pyrai3md_agent',
            'input_file': os.path.join(self.tmpdir, 'casscf_energy.config'),
            'script': script,
            'use_slurm': False,
            'skill_slurm': True,
        }

    def test_missing_artifacts_fail_a_passing_return_code(self):
        task = self._base_task(self._write_ok_script())
        result = runner._run_skill_test(task, {}, {})

        self.assertEqual(result['status'], 'fail')
        self.assertIn('slurm artifacts', result['details'])

    def test_artifacts_present_lets_the_task_pass(self):
        script = self._write_ok_script()
        calcdir = os.path.join(self.tmpdir, 'qm_layer0_molcas')
        os.makedirs(calcdir)
        for name in ('MOLCAS.sbatch', 'MOLCAS_slurm.out'):
            with open(os.path.join(calcdir, name), 'w') as f:
                f.write('fake\n')

        task = self._base_task(script)
        result = runner._run_skill_test(task, {}, {})

        self.assertEqual(result['status'], 'pass')

    def test_timeout_exit_code_maps_to_error(self):
        # The execution script's embedded `timeout` exits 124 (SIGTERM)
        # or 137 (SIGKILL escalation); the runner reports a timeout
        # error rather than a calculation failure. The embedded timeout
        # has already killed the engine process group, so no residual
        # processes survive to slow the next test.
        for code in (124, 137):
            script = os.path.join(self.tmpdir, f'to_{code}.sh')
            with open(script, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\nexit {code}\n')
            task = self._base_task(script)
            task['skill_slurm'] = False
            result = runner._run_skill_test(task, {}, {'_timeout': 300})
            self.assertEqual(result['status'], 'error')
            self.assertIn('timed out', result['details'].lower())

    def test_find_slurm_artifacts_excludes_harness_script(self):
        harness = os.path.join(self.tmpdir, 'task_agent.sbatch')
        with open(harness, 'w') as f:
            f.write('#!/bin/bash\n')
        calcdir = os.path.join(self.tmpdir, 'calc')
        os.makedirs(calcdir)
        skill_script = os.path.join(calcdir, 'MOLCAS.sbatch')
        skill_out = os.path.join(calcdir, 'MOLCAS_slurm.out')
        for path in (skill_script, skill_out):
            with open(path, 'w') as f:
                f.write('fake\n')

        sbatch_files, out_files = runner._find_slurm_artifacts(
            self.tmpdir, exclude_script=harness)

        self.assertEqual(sbatch_files, [skill_script])
        self.assertEqual(out_files, [skill_out])


class TestClosedSwitchRowsFoldIntoOneRow(unittest.TestCase):
    """What a round prints is bounded by what it ran, not by what it listed.

    tmp_20261016_135553 planned 390 tasks and did 123: the other 267 rows each
    said only which switch closed them, so the log's signal-to-noise was 1 in
    3 and the 267 lines a reader had to skip were byte-identical in groups of
    up to 26. The plan still lists each closed task (a per-example switch is
    worth nothing if what it closed is not enumerable), but the run now folds a
    bucket's closed rows into one row carrying the keys and their counts.

    The fold keys on the reason text, not on skipping in general: a postponed
    example or a missing sbatch client is one task's own news and keeps its own
    row.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _closed(self, name, key):
        return {'name': name, 'mode': 'pyrai3md_yaml', 'skill': 'examples',
                'input_file': '/dev/null', 'skip': True,
                'skip_reason': 'disabled by %s' % key}

    def _postponed(self, name):
        return {'name': name, 'mode': 'pyrai3md_yaml', 'skill': 'examples',
                'input_file': '/dev/null', 'skip': True,
                'skip_reason': ('Qualification postponed: waiting for the '
                                'Dispatcher and Workflow runtime to stabilize')}

    def _run(self, **buckets):
        plan = {'work_dir': self.tmpdir,
                'main_tests': [], 'example_tests': [],
                'skill_tests': [], 'benchmark_tests': []}
        plan.update(buckets)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            results = runner.run_tests(plan, {})
        return results, buffer.getvalue()

    def _report(self, results):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            runner.print_report(results)
        return buffer.getvalue()

    def test_a_closed_bucket_prints_one_row_worth_all_its_tasks(self):
        results, out = self._run(example_tests=[
            self._closed('example_energy', 'examples.energy'),
            self._postponed('example_nn_train'),
            self._closed('example_gradient', 'examples.gradient'),
        ])
        rows = results['example_results']
        self.assertEqual(len(rows), 2,
                         'msg=expected the live row plus one folded row, got '
                         '%s' % [row['name'] for row in rows])
        self.assertEqual(rows[0]['name'], 'example_nn_train',
                         'msg=the postponed task is not a closed switch and '
                         'must keep its own row')
        folded = rows[1]
        self.assertEqual(folded['name'], 'example_closed')
        self.assertEqual(folded['skip_count'], 2)
        self.assertEqual(sorted(folded['closed_reasons']),
                         ['disabled by examples.energy',
                          'disabled by examples.gradient'])
        # What the fold buys in the live stream: the closed names are gone, and
        # the row says how much it stands for. The keys themselves are not
        # printed here either - 72 `disabled by examples.*` tokens in the live
        # log is the wall of text the fold removed, just rearranged.
        self.assertNotIn('example_energy', out)
        self.assertNotIn('disabled by examples.energy', out)
        self.assertIn('[FOLD] 2 tasks closed by 2 switch(es)', out)
        # The keys are still answerable, in the report the round ends with.
        report = self._report(results)
        self.assertIn('SKIP x2: example_closed (2 switches closed)', report)
        self.assertIn('disabled by examples.energy x1', report)
        self.assertIn('disabled by examples.gradient x1', report)

    def test_the_header_counts_rows_not_plan_entries(self):
        results, out = self._run(example_tests=[
            self._closed('example_a', 'examples.a'),
            self._closed('example_b', 'examples.b'),
            self._postponed('example_c'),
        ])
        # Two rows print, so the counter runs to 2: a denominator of 3 would
        # leave the last line reading [2/3] and the poller waiting on a task
        # that never arrives.
        self.assertIn('Total tests: 2', out)
        self.assertIn('[2/2]', out)
        self.assertNotIn('[3/3]', out)

    def test_a_round_with_everything_closed_still_reports_a_verdict(self):
        results, _out = self._run(
            example_tests=[self._closed('example_a', 'examples.a'),
                           self._closed('example_b', 'examples.b')],
            skill_tests=[self._closed('cp2k_skill_energy_case',
                                      'skills.cp2k_skill.test')],
            benchmark_tests=[self._closed('benchmark_fssh_tully1',
                                          'benchmark.fssh.enabled')])
        self.assertEqual(results['status'], 'pass',
                         'msg=a fully closed round folds to three rows and '
                         'must still carry a verdict, got %r' % results)
        self.assertEqual(
            [results[key][-1]['name'] for key in
             ('example_results', 'skill_results', 'benchmark_results')],
            ['example_closed', 'skill_closed', 'benchmark_closed'])
        # The skip total reads what the rows are worth, not how many printed:
        # three rows, four closed tasks. Keeping the two buckets at one closed
        # task each would make this assertion pass with the row count.
        self.assertEqual(results['example_results'][-1]['skip_count'], 2)
        report = self._report(results)
        self.assertIn('Total rows: 3', report)
        self.assertIn('Skip rows: 3 (4 skipped tasks)', report)

    def test_the_fold_keeps_the_bucket_boundary(self):
        # One row per bucket, never one row for the whole round: 'which switch'
        # is only answerable if the example, skill and benchmark closures stay
        # in separate rows.
        results, _out = self._run(
            example_tests=[self._closed('example_a', 'examples.a')],
            skill_tests=[self._closed('a_skill_case', 'skills.a.test')])
        self.assertEqual(results['example_results'][-1]['name'],
                         'example_closed')
        self.assertEqual(results['skill_results'][-1]['name'], 'skill_closed')
        self.assertEqual(len(results['skill_results']), 1)


class TestSharedSkillLayerTestsArePlanned(unittest.TestCase):
    """pyrai3md/skills/tests/ must reach the round as pytest rows.

    The three files there check the layer every skill shares -- the agent's
    calculate() forwarding contract, the wrapper's skill-name resolution, and
    since #375 the anchor every shell-launching runner hands its deck to bash
    with -- and need no engine, so the rule #359 states for a skill's own pytest
    files says they belong in a remote round. Until #360 the round planned none
    of them: they sit beside no skill package, so no skill's directory discovery
    reached them, and the phase that would have picked them up by name
    discovers with unittest, which collects zero cases from a pytest module and
    would have called that a pass.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_the_skills_layer_files_are_discovered_as_a_group(self):
        discovered = handler.list_shared_skill_tests()
        self.assertEqual(
            [os.path.basename(path) for path in discovered],
            ['test_local_runner_anchor.py', 'test_skill_agents.py',
             'test_wrapper.py'],
            'msg=the layer\'s own tests are the shipped set, in name order')

    def test_the_shipped_stage_plans_one_row_per_shared_file(self):
        from pyrai3md.qualification.manager import load_config

        discovered = handler.list_shared_skill_tests()
        tasks = planner.plan_skill_tests(load_config(), self.tmpdir)
        shared = [task for task in tasks
                  if task['skill'] == planner.SHARED_SKILL_TESTS_LABEL]
        self.assertEqual(len(shared), len(discovered),
                         'msg=%d rows for %d files, planned %s'
                         % (len(shared), len(discovered),
                            [task['name'] for task in shared]))
        self.assertEqual({task['mode'] for task in shared},
                         {'skill_runtime'})
        # Once for the round, not once per skill: a duplicate would print the
        # same 22 cases ten times over.
        self.assertEqual(len(shared), len({task['name'] for task in shared}))
        # The shipped master forces the skills' pytest phase open, so the
        # shared group is open too: it follows the aggregate
        # skills.<name>.runtime_tests phase and has no key of its own.
        self.assertTrue(all(not task.get('skip') for task in shared),
                        'msg=the shipped master opens the runtime phase, so '
                        'the shared group must be open: %s'
                        % [(task['name'], task.get('skip_reason'))
                           for task in shared if task.get('skip')])
        for task in shared:
            self.assertTrue(os.path.isfile(task['script']),
                            'msg=%s names a script that was never written'
                            % task['name'])

    def test_the_planned_files_collect_cases_under_pytest(self):
        # A row that runs and collects nothing is the fake green this phase
        # exists to prevent, so the count is read from pytest's own line.
        for path in handler.list_shared_skill_tests():
            proc = subprocess.run(
                [sys.executable, '-m', 'pytest', path, '-q', '--collect-only',
                 '-p', 'no:cacheprovider'],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0,
                             'msg=%s would not even collect: %s'
                             % (path, proc.stdout[-300:]))
            collected = re.search(r'(\d+) tests? collected', proc.stdout)
            self.assertIsNotNone(collected,
                                 'msg=no collection line for %s: %s'
                                 % (path, proc.stdout[-300:]))
            self.assertGreater(int(collected.group(1)), 0,
                               'msg=%s collects zero cases' % path)

    def test_the_unittest_phase_would_collect_nothing_from_them(self):
        # Why these rows are pytest rows: the main-test phase discovers with
        # unittest, and these files are pytest functions. Adding them there
        # would report a green that ran nothing.
        import pyrai3md

        skills_tests = os.path.join(os.path.dirname(pyrai3md.__file__),
                                    'skills', 'tests')
        suite = unittest.TestLoader().discover(
            start_dir=skills_tests, pattern='test_*.py',
            top_level_dir=os.path.dirname(pyrai3md.__file__))
        self.assertEqual(suite.countTestCases(), 0,
                         'msg=these files are not unittest cases; the '
                         'pytest row is the only honest carrier')

    def test_the_shared_group_closes_with_the_skill_pytest_phase(self):
        from pyrai3md.qualification.manager import load_config

        config = load_config()
        config['_runall'] = SWITCH_CONDITION
        config['skills'] = {
            name: {**env, 'test': False, 'runtime_tests': False}
            for name, env in config['skills'].items()}
        tasks = planner.plan_skill_tests(config, self.tmpdir)
        shared = [task for task in tasks
                  if task['skill'] == planner.SHARED_SKILL_TESTS_LABEL]
        self.assertTrue(shared)
        self.assertTrue(all(task.get('skip') for task in shared),
                        'msg=the shared pytest group must close with the '
                        'skills.<name>.runtime_tests phase')
        self.assertEqual({task.get('skip_reason') for task in shared},
                         {'disabled by skills.*.runtime_tests'})

    def test_runall_true_opens_the_skill_pytest_phase(self):
        from pyrai3md.qualification.manager import load_config

        config = load_config()
        config['_runall'] = True
        config['skills'] = {
            name: {**env, 'runtime_tests': False}
            for name, env in config['skills'].items()}
        work_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work_dir, ignore_errors=True)
        tasks = planner.plan_skill_tests(config, work_dir)
        runtime = [task for task in tasks
                   if task.get('mode') == 'skill_runtime'
                   and task.get('skill') != planner.SHARED_SKILL_TESTS_LABEL]
        self.assertTrue(runtime)
        self.assertFalse(any(task.get('skip') for task in runtime),
                         'msg=_runall: true must open the runtime phase')


class TestSurfaceHoppingJobtype(unittest.TestCase):
    """surface_hopping is a first-class advanced jobtype (#110): the
    framework vocabulary, workflow registry and converter must all
    accept it so casscf_surface_hopping configs no longer raise
    ValueError: Unknown jobtype."""

    def test_resolve_jobtype_from_profile(self):
        self.assertEqual(
            _resolve_jobtype({'profile': 'casscf_surface_hopping'}),
            'surface_hopping')

    def test_validate_accepts_surface_hopping(self):
        tokens, is_basic, is_advanced = parse_jobtype('surface_hopping')
        self.assertEqual(tokens, ['surface_hopping'])
        self.assertFalse(is_basic)
        self.assertTrue(is_advanced)

    def test_workflow_registry_maps_surface_hopping(self):
        self.assertIn('surface_hopping', WORKFLOW_REGISTRY)
        self.assertIn('surface_hopping', JOBTYPE_MAP)

    def test_convert_writes_md_section_for_surface_hopping(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        config = {
            'profile': 'casscf_surface_hopping',
            'skill_name': 'molcas_skill',
            'atoms': ['C', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            'system': {'current_state': 1},
        }
        config_path = os.path.join(tmpdir, 'hop.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        input_path = os.path.join(tmpdir, 'hop.input')
        convert(config_path, input_path)

        with open(input_path, 'r', encoding='utf-8') as f:
            section = f.read()
        self.assertIn('jobtype       surface_hopping', section)
        self.assertIn('&MD', section)
        self.assertIn('current_state   1', section)


class TestCurrentStatePipeline(unittest.TestCase):
    """current_state travels yaml -> converter system section ->
    converter &MOLECULE line (#110)."""

    def test_molecule_section_reads_nested_system(self):
        section = _build_molecule_section(
            {'system': {'current_state': 1}})
        self.assertIn('current_state   1', section)

    def test_molecule_section_reads_top_level(self):
        section = _build_molecule_section({'current_state': 2})
        self.assertIn('current_state   2', section)

    def test_molecule_section_omits_when_absent(self):
        section = _build_molecule_section({})
        self.assertNotIn('current_state', section)

    def test_handler_nests_current_state(self):
        import pyrai3md.skills.molcas_skill as molcas_skill

        yaml_path = os.path.join(
            os.path.dirname(molcas_skill.__file__),
            'tests', 'casscf_surface_hopping.yaml')
        config = parse_test_yaml(yaml_path, 'molcas_skill')

        self.assertNotIn('current_state', config)
        self.assertEqual(config['system']['current_state'], 1)

    def test_yaml_to_config_convert_roundtrip(self):
        # End-to-end: real surface hopping YAML -> converter config ->
        # classic input with the advanced jobtype and current_state.
        import pyrai3md.skills.molcas_skill as molcas_skill

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        yaml_path = os.path.join(
            os.path.dirname(molcas_skill.__file__),
            'tests', 'casscf_surface_hopping.yaml')
        config = parse_test_yaml(yaml_path, 'molcas_skill')
        config_path = os.path.join(tmpdir, 'surface_hopping.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        input_path = os.path.join(tmpdir, 'surface_hopping.input')
        convert(config_path, input_path)

        with open(input_path, 'r', encoding='utf-8') as f:
            section = f.read()
        self.assertIn('jobtype       surface_hopping', section)
        self.assertIn('&MD', section)
        self.assertIn('current_state   1', section)


class TestHessianSectionCarrier(unittest.TestCase):
    """The `&HESSIAN` keywords one carrier can write are the ones the other
    carrier can read.

    #323 registered the `hess_*` words in the reader's type table
    (`schema/adapter.HESSIAN_TYPES`), the workflow section table and the state
    whitelist, but the classic-input writer kept its own older list of just the
    three dispatch keywords. The result was carrier-dependent: a yaml input
    declaring `hess_temperature` converted and ran, while the same input's
    `.input` carrier was refused at conversion time ("&HESSIAN carries
    keyword(s) hess_pressure, hess_temperature the node has no field for") --
    found locally the moment `examples/hessian_num.yaml` was planned in both
    carriers. The writer now derives its list from the reader's table, so the
    two cannot drift again, and a keyword neither side knows still raises.
    """

    _HESS = {'hess_temperature': 298.15, 'hess_pressure': 1.0,
             'hess_step': 0.005, 'hess_mode': 'num', 'hess_symmetry': 2,
             'multitask': True}

    def test_every_reader_keyword_is_writable(self):
        from pyrai3md.qualification.certification.converter import (
            WORKFLOW_SECTION_KEYS)
        from pyrai3md.schema.adapter import HESSIAN_TYPES

        self.assertEqual(
            set(HESSIAN_TYPES) - {'multitask', 'remotetask', 'multipool'},
            set(WORKFLOW_SECTION_KEYS['hessian'])
            - {'multitask', 'remotetask', 'multipool'})

    def test_the_classic_section_round_trips_the_values(self):
        from pyrai3md.qualification.certification.converter import (
            _build_workflow_sections)

        import re

        text = '\n'.join(_build_workflow_sections({'hessian':
                                                  dict(self._HESS)}))
        self.assertIn('&HESSIAN', text)
        for key, value in self._HESS.items():
            spelled = ('true' if value else 'false') if isinstance(
                value, bool) else str(value)
            self.assertRegex(text, r'(?m)^%s\s+%s$' % (
                re.escape(key), re.escape(spelled)))

    def test_an_unknown_keyword_is_still_refused(self):
        from pyrai3md.qualification.certification.converter import (
            _build_workflow_sections)

        with self.assertRaises(ValueError) as ctx:
            _build_workflow_sections({'hessian': {'hess_not_a_keyword': 1}})
        self.assertIn('hess_not_a_keyword', str(ctx.exception))


class TestPeriodicCellPipeline(unittest.TestCase):
    """lattice_vector travels yaml -> converter -> converter &MOLECULE
    'cell' line -> framework normalize -> skill System submodel."""

    _CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

    def test_molecule_section_flattens_lattice_vector(self):
        section = _build_molecule_section(
            {'lattice_vector': self._CELL})
        self.assertIn(
            'cell          10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0',
            section)

    def test_molecule_section_reads_nested_system(self):
        section = _build_molecule_section(
            {'system': {'lattice_vector': self._CELL}})
        self.assertIn('cell          10.0 0.0 0.0', section)

    def test_molecule_section_omits_zero_or_absent_cell(self):
        section = _build_molecule_section({})
        self.assertNotIn('cell', section)
        zero = [[0.0, 0.0, 0.0]] * 3
        section = _build_molecule_section({'lattice_vector': zero})
        self.assertNotIn('cell', section)

    def test_cp2k_yaml_cell_reaches_framework_system(self):
        # End-to-end: cp2k test YAML -> converter -> converter ->
        # classic .input cell line -> framework normalize rebuilds
        # the 3x3 lattice_vector in the System submodel.
        import pyrai3md.skills.cp2k_skill as cp2k_skill

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        yaml_path = os.path.join(
            os.path.dirname(cp2k_skill.__file__),
            'tests', 'dft_energy.yaml')
        config = parse_test_yaml(yaml_path, 'cp2k_skill')
        config_path = os.path.join(tmpdir, 'dft_energy.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        input_path = os.path.join(tmpdir, 'dft_energy.input')
        convert(config_path, input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            self.assertIn(
                'cell          10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0',
                f.read())

        config_path_agent = os.path.join(tmpdir, 'dft_energy.config')
        sectioned = convert_to_pyrai_config(config_path, config_path_agent)
        self.assertEqual(sectioned['molecule']['cell'],
                         [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0])

        # Text-mode round trip: cell line -> System.lattice_vector
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        raw = parse_input(input_path)
        fw_config = normalize(build_config(raw))
        self.assertEqual(fw_config.system.lattice_vector, self._CELL)


class TestOrcaXtbResolveProfile(unittest.TestCase):
    """orca/xtb query modules expose resolve_profile.

    The framework derives the agent-mode profile through
    skill_query.resolve_profile; a missing function silently keeps
    the schema default profile, which clashes with non-default
    methods (#128 regression: tddft agent tests hit 'Profile
    dft_energy requires method dft' remotely).
    """

    def test_orca_query_exposes_resolve_profile(self):
        from pyrai3md.skills.orca_skill import query
        self.assertTrue(hasattr(query, 'resolve_profile'))

    def test_orca_agent_jobtypes_round_trip(self):
        from pyrai3md.skills.orca_skill import query
        cases = {
            ('tddft', 'energy gradient nac'): 'tddft_nac',
            ('tddft', 'energy frequency'): 'tddft_frequency',
            ('tddft', 'energy gradient'): 'tddft_gradient',
            ('tddft', 'optimize'): 'tddft_opt_min',
            ('tddft', 'energy transition'): 'tddft_transition_dipole',
            ('tddft', 'energy soc'): 'tddft_soc',
            ('dft', 'energy dipole'): 'dft_dipole',
            ('dft', 'energy'): 'dft_energy',
        }
        for (method, jobtype), expected in cases.items():
            with self.subTest(method=method, jobtype=jobtype):
                self.assertEqual(
                    query.resolve_profile(method, jobtype), expected)

    def test_orca_token_order_is_canonicalized(self):
        from pyrai3md.skills.orca_skill import query
        # Permuted/duplicated tokens collapse onto the same profile
        self.assertEqual(
            query.resolve_profile('tddft', 'nac gradient energy'),
            'tddft_nac')
        self.assertEqual(
            query.resolve_profile('tddft', 'energy energy gradient'),
            'tddft_gradient')

    def test_xtb_query_exposes_resolve_profile(self):
        from pyrai3md.skills.xtb_skill import query
        self.assertTrue(hasattr(query, 'resolve_profile'))
        self.assertEqual(
            query.resolve_profile('gfn2', 'energy gradient'),
            'gfn2_gradient')


class TestProjectNamingSource(unittest.TestCase):
    """The yaml top-level name (converter -> config title) is the
    single naming source for the classic input title and the skill
    project."""

    def test_handler_injects_yaml_name_for_molcas_modes(self):
        import pyrai3md.skills.molcas_skill as molcas_skill
        tests_dir = os.path.join(
            os.path.dirname(molcas_skill.__file__), 'tests')
        titles = {}
        for stem in ('casscf_geometry_optimization', 'casscf_meci',
                     'casscf_mecp'):
            cfg = parse_test_yaml(
                os.path.join(tests_dir, stem + '.yaml'), 'molcas_skill')
            titles[stem] = cfg['title']
            # Project shares the single naming source with the title
            self.assertEqual(cfg['molcas_project'], cfg['title'])
        self.assertEqual(
            titles['casscf_geometry_optimization'],
            'casscf_geometry_optimization_test')
        self.assertEqual(titles['casscf_meci'], 'casscf_meci_test')
        self.assertEqual(titles['casscf_mecp'], 'casscf_mecp_test')
        # The three modes sharing one profile stays distinguishable
        self.assertEqual(len(set(titles.values())), 3)

    def test_handler_injects_yaml_name_for_all_skills(self):
        import pyrai3md.skills as skills_pkg
        import yaml as yaml_module
        for skill in ('molcas', 'pyscf', 'bagel', 'orca',
                      'xtb', 'openqp', 'lammps', 'cp2k'):
            skill_name = f'{skill}_skill'
            skill_pkg = getattr(skills_pkg, skill_name, None)
            if skill_pkg is None:
                import importlib
                skill_pkg = importlib.import_module(
                    f'pyrai3md.skills.{skill_name}')
            tests_dir = os.path.join(
                os.path.dirname(skill_pkg.__file__), 'tests')
            yaml_name = sorted(
                f for f in os.listdir(tests_dir)
                if f.endswith('.yaml'))[0]
            yaml_path = os.path.join(tests_dir, yaml_name)
            with open(yaml_path, 'r', encoding='utf-8') as f:
                expected = yaml_module.safe_load(f).get('name')
            cfg = parse_test_yaml(yaml_path, skill_name)
            self.assertEqual(cfg['title'], expected, msg=skill)
            self.assertEqual(cfg[f'{skill}_project'], expected,
                             msg=skill)

    def test_derive_project_name_title_priority(self):
        config = {'title': 'casscf_meci',
                  'profile': 'casscf_geometry_optimization'}
        self.assertEqual(
            _derive_project_name(config), 'casscf_meci')
        self.assertEqual(
            _derive_project_name({'profile': 'casscf_energy'}),
            'casscf_energy')
        self.assertEqual(_derive_project_name({}), 'pyrai3md')

    def test_convert_control_title_matches_skill_project(self):
        config = {
            'title': 'casscf_meci',
            'profile': 'casscf_geometry_optimization',
            'skill_name': 'molcas_skill',
            'atoms': ['C'],
            'coordinates': [[0.0, 0.0, 0.0]],
        }
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = os.path.join(tmpdir, 'casscf_meci.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        input_path = os.path.join(tmpdir, 'casscf_meci.input')
        convert(config_path, input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            section = f.read()
        self.assertIn('title         casscf_meci', section)
        self.assertIn('molcas_project casscf_meci', section)


class TestPyraiConfigOutput(unittest.TestCase):
    """convert_to_pyrai_config serializes the same sectioned dict the
    classic input is rendered from (#141): pyrai3md_agent consumes a
    .config whose skill parameters live in the full-name skill section
    and whose System fields (structure data + state keywords) sit in
    the molecule section per the System submodel convention."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_flat_config(self, name='gfn2_energy.json'):
        config = {
            'profile': 'gfn2_energy',
            'skill_name': 'xtb_skill',
            'title': 'gfn2_energy',
            'method': 'gfn2',
            'atoms': ['C', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            'charges': 0,
            'multiplicity': 1,
            'nstate': 1,
            'state_spin': [0],
            'scratch_path': '/scratch/xtb',
        }
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        return path

    def test_sectioned_dict_carries_structure_and_skill_section(self):
        sectioned = convert_to_pyrai_config(self._write_flat_config())

        # System fields live in the molecule section (#142)
        molecule = sectioned['molecule']
        self.assertEqual(molecule['atoms'], ['C', 'H'])
        self.assertEqual(molecule['coordinates'],
                         [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        self.assertEqual(molecule['charges'], 0)
        self.assertEqual(molecule['multiplicity'], 1)
        self.assertEqual(molecule['nstate'], 1)
        self.assertEqual(molecule['state_spin'], [0])
        self.assertNotIn('atoms', sectioned)
        self.assertNotIn('coordinates', sectioned)
        self.assertIn('control', sectioned)
        self.assertIn('xtb_skill', sectioned)
        self.assertEqual(sectioned['control']['title'],
                         'gfn2_energy')
        self.assertEqual(sectioned['control']['jobtype'], 'energy')
        # scratch_path must reach the skill through its own section
        self.assertEqual(sectioned['xtb_skill']['scratch_path'],
                         '/scratch/xtb')
        self.assertEqual(sectioned['xtb_skill']['method'], 'gfn2')

    def test_config_output_matches_returned_dict(self):
        config_path = self._write_flat_config()
        output_path = os.path.join(self.tmpdir, 'gfn2_energy.config')

        sectioned = convert_to_pyrai_config(config_path, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            self.assertEqual(json.load(f), sectioned)


class TestBuildMdSection(unittest.TestCase):
    """&MD: root is rendered in its external 1-based spelling (derived
    from the internal 0-based current_state + 1); activestate is
    passed through (default 1)."""

    def test_root_from_nested_system_current_state(self):
        config = {'system': {'current_state': 1}, 'activestate': 0}
        section = _build_md_section(config)
        self.assertIn('root          2', section)
        self.assertIn('activestate   0', section)

    def test_root_from_flat_current_state(self):
        section = _build_md_section({'current_state': 2})
        self.assertIn('root          3', section)

    def test_defaults_without_current_state(self):
        section = _build_md_section({'nstate': 4})
        self.assertIn('root          1', section)
        self.assertIn('activestate   1', section)


class TestMdSectionEnsembleEcho(unittest.TestCase):
    """&MD echoes the dynamics/ensemble keywords that deviate from the
    framework defaults, so the classic-input and agent-config modes run
    the same dynamics as the direct-YAML mode."""

    def test_nondefault_dynamics_keywords_are_echoed(self):
        config = {'sfhp': 'fssh', 'ensemble': 'npt',
                  'thermostat': 'nose-hoover-chain', 'barostat': 'mttk',
                  'thermostat_chain': 3, 'temperature': 350}
        section = _build_md_section(config)
        self.assertIn('sfhp          fssh', section)
        self.assertIn('ensemble      npt', section)
        self.assertIn('thermostat    nose-hoover-chain', section)
        self.assertIn('barostat      mttk', section)
        # Key beyond the 14-char column keeps a separating space.
        self.assertIn('thermostat_chain 3', section)
        # temperature renders with the classic 'temp' spelling.
        self.assertIn('temp          350', section)

    def test_output_throttling_keywords_are_echoed(self):
        # Without the echo the classic-input mode silently records the
        # schema defaults (direct 2000 / buffer 500) instead of the
        # frames the YAML asked for.
        section = _build_md_section({'step': 10000, 'size': 41.34,
                                     'direct': 1, 'buffer': 1000})
        self.assertIn('direct        1', section)
        self.assertIn('buffer        1000', section)

    def test_velocity_control_keywords_are_echoed(self):
        # `md_velocity_randvelo` declared a velocity draw and the `.input`
        # carrier ran none at all (first-frame Ekin 0.0 against 0.00588 in
        # the yaml carrier, round tmp_20261011_184226); `md_velocity_excess`
        # differed by exactly its declared 5.0e-4. The family lived only in
        # the schema and the yaml, not in the renderer's echo list -- the
        # same one-list-per-side slip as the &HESSIAN keywords (#325).
        section = _build_md_section({
            'step': 3, 'size': 20.67, 'temperature': 300,
            'randvelo': 1, 'excess': 0.0005, 'scale': 1.02, 'target': 300.0,
            'reset': 1, 'resetstep': 0, 'dissipate': 1, 'thermodelay': 3,
        })
        for line in ('randvelo      1', 'excess        0.0005',
                     'scale         1.02', 'target        300.0',
                     'reset         1', 'resetstep     0',
                     'dissipate     1', 'thermodelay   3'):
            self.assertIn(line, section)

    def test_every_echoed_md_keyword_is_readable_by_the_classic_parser(self):
        # Writing a name the classic reader does not know would put the
        # declaration in the `.input` file and nowhere in the parsed config
        # -- a divergence that shows up only as two carriers computing
        # different dynamics. The reader's table is the single authority.
        from pyrai3md.schema.adapter import MD_TYPES

        for name in _MD_ECHO_KEYS:
            self.assertIn(name[1], MD_TYPES, name[1])

    def test_absent_throttling_keywords_are_not_invented(self):
        section = _build_md_section({'step': 10})
        self.assertNotIn('direct', section)
        self.assertNotIn('buffer', section)

    def test_present_defaults_are_echoed_absent_keys_are_not(self):
        # Presence-based echo: a key the source YAML wrote is echoed
        # even when it equals the framework default (nactype ktdc);
        # a key absent from the flat config is not invented.
        config = {'sfhp': 'nosh', 'nactype': 'ktdc', 'step': 10,
                  'size': 20.67, 'temperature': 300, 'initcond': 0,
                  'ninitcond': 1, 'ensemble': 'nve'}
        section = _build_md_section(config)
        keys = {line.split()[0] for line in section.splitlines()
                if line.strip() and not line.startswith('&')}
        for token in ('sfhp', 'nactype', 'step', 'size', 'temp',
                      'initcond', 'ninitcond', 'ensemble'):
            self.assertIn(token, keys)
        for absent in ('barostat', 'thermostat', 'pressure',
                       'compressibility'):
            self.assertNotIn(absent, keys)

    def test_explicit_root_wins_over_current_state(self):
        section = _build_md_section({'root': '2', 'current_state': 0})
        self.assertIn('root          2', section)

    def test_multicenter_branch_echoes_dynamics(self):
        config = {'root': '1; 2', 'highlevel': '1-6; 7-12',
                  'sfhp': 'fssh', 'step': 10}
        section = _build_md_section(config)
        self.assertIn('root          1; 2', section)
        self.assertIn('sfhp          fssh', section)
        self.assertIn('step          10', section)
        self.assertNotIn('barostat', section)


class TestMoleculeSectionPeriodicEcho(unittest.TestCase):
    """&MOLECULE echoes pbc flags and the six-parameter cell spelling
    (lengths Angstrom, angles degrees) used by the example YAMLs."""

    def test_pbc_and_cell6_echoed(self):
        config = {'pbc': [1, 1, 1],
                  'cell': [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]}
        section = _build_molecule_section(config)
        self.assertIn('pbc           1 1 1', section)
        self.assertIn('cell          10.0 10.0 10.0 90.0 90.0 90.0',
                      section)

    def test_default_pbc_and_sentinel_cell_omitted(self):
        section = _build_molecule_section(
            {'pbc': [0, 0, 0],
             'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0]})
        self.assertNotIn('pbc', section)
        self.assertNotIn('cell', section)

    def test_lattice_vector_wins_over_cell6(self):
        config = {'lattice_vector': [[10.0, 0.0, 0.0],
                                     [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]],
                  'cell': [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]}
        section = _build_molecule_section(config)
        self.assertIn(
            'cell          10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0',
            section)


class TestMoleculeExplicitCiSpin(unittest.TestCase):
    """Single-center ci/spin/coupling in the classic string or typed
    list form pass through instead of the nstate derivation."""

    def test_classic_strings_pass_through(self):
        config = {'ci': '3', 'spin': '0', 'coupling': '1 2, 2 3, 1 3'}
        section = _build_molecule_section(config)
        self.assertIn('ci            3', section)
        self.assertIn('spin          0', section)
        self.assertIn('coupling      1 2, 2 3, 1 3', section)

    def test_typed_lists_render(self):
        config = {'ci': [2], 'spin': [0]}
        section = _build_molecule_section(config)
        self.assertIn('ci            2', section)
        self.assertIn('spin          0', section)

    def test_typed_coupling_still_shifts_to_one_based(self):
        config = {'ci': [3], 'spin': [0], 'coupling': [[0, 1], [1, 2]]}
        section = _build_molecule_section(config)
        self.assertIn('coupling      1 2, 2 3', section)


class TestRunLogStatus(unittest.TestCase):
    """The run-log status check follows the task's artifact anchors.

    An example writes <title>.log into its run directory, and its title
    differs from the source YAML stem, so an input-relative lookup finds
    nothing and a 'Status: error' run passes silently. Skill tests stage
    their log beside their config under the config stem and must keep
    working unchanged.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _log(self, directory, name, status):
        # One `Status:` line whose value may itself be multi-line, exactly
        # as an agent writes a safe_run traceback into it.
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('... running\nStatus: %s\n' % status)
        return path

    def test_example_log_is_found_by_title_and_run_dir(self):
        run_dir = os.path.join(self.dir, 'run')
        self._log(run_dir, 'case_example.log',
                  'failed: calculation failed with return code 1')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.yaml'),
            base_dir=run_dir, title='case_example')
        self.assertEqual(result['status'], 'fail')
        self.assertIn('return code 1', result['details'])

    def test_unanchored_lookup_blindly_passes_an_example(self):
        # Regression witness: without the anchors the very same run
        # reports a clean pass.
        run_dir = os.path.join(self.dir, 'run')
        self._log(run_dir, 'case_example.log',
                  'failed: calculation failed with return code 1')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.yaml'))
        self.assertEqual(result['status'], 'pass')

    def test_skill_test_log_keeps_the_input_stem_anchor(self):
        input_path = os.path.join(self.dir, 'case.json')
        with open(input_path, 'w', encoding='utf-8') as handle:
            handle.write('{}\n')
        self._log(self.dir, 'case.log', 'error')
        result = {'status': 'pass'}
        runner._check_skill_log_status(result, input_path)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('[LOG STATUS]', result['details'])

    def test_a_traceback_status_fails_the_task(self):
        # What `safe_run` writes when the engine raised, and the shape the
        # old word list ('failed', 'error') let through: six cp2k and molcas
        # cases rode that false green in round tmp_20261010_133424 with
        # byte-identical engine inputs and an engine that had stopped on a
        # user input error.
        run_dir = os.path.join(self.dir, 'tb')
        self._log(run_dir, 'case.log',
                  'Traceback (most recent call last):\n'
                  '  File "agent.py", line 51, in _run_calculation\n'
                  'RuntimeError: Molcas calculation failed with return '
                  'code 112\n')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.json'), base_dir=run_dir,
            title='case')
        self.assertEqual(result['status'], 'fail')
        self.assertIn('[LOG STATUS] Traceback', result['details'])

    def test_a_skill_agent_log_beside_the_config_is_still_found(self):
        # Two naming schemes, one check: the skill agent names its log after
        # the config it was handed and writes it beside that config, while a
        # framework leg writes `<title>.log` into its run directory. Looking
        # only in the run directory for the input stem -- what this did before
        # #325 -- matched nothing for skill cases, and an agent that swallows
        # its exception exits 0, so a failed run was recorded as a pass.
        skill_dir = os.path.join(self.dir, 'skill')
        os.makedirs(skill_dir, exist_ok=True)
        input_path = os.path.join(skill_dir, 'case.json')
        with open(input_path, 'w', encoding='utf-8') as handle:
            handle.write('{}\n')
        self._log(skill_dir, 'case.log',
                  'Traceback (most recent call last):')
        run_dir = os.path.join(self.dir, 'agent_leg')
        os.makedirs(run_dir, exist_ok=True)
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, input_path, base_dir=run_dir, title='case_test')
        self.assertEqual(result['status'], 'fail')
        self.assertIn('[LOG STATUS]', result['details'])

    def test_a_leg_log_named_by_the_case_title_is_found(self):
        # The framework leg's file is `<title>.log`, not `<stem>.log`.
        run_dir = os.path.join(self.dir, 'input_leg')
        self._log(run_dir, 'case_test.log', 'Traceback (most recent call last):')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.input'), base_dir=run_dir,
            title='case_test')
        self.assertEqual(result['status'], 'fail')

    def test_an_unconverged_optimization_status_is_still_tolerated(self):
        # Qualification runs truncated optimizations on purpose, and the run
        # log records that as `Status: failed: Maximum optimization steps
        # reached`. The closing line is read since #328, so this exception has
        # to come from the same table the exit-code path consults -- without it
        # four constraint-optimize examples went red in round
        # tmp_20261011_152909 for a state the suite has always accepted.
        run_dir = os.path.join(self.dir, 'unconv')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case.log'), 'w',
                  encoding='utf-8') as handle:
            handle.write('PyRAI3MD End: Oct 11 2026 15:43:17 Total: 00:00:00:03 '
                         'Status: failed: Maximum optimization steps reached\n')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.yaml'), base_dir=run_dir,
            title='case')
        self.assertEqual(result['status'], 'pass')

    def test_every_tolerated_status_text_found_on_the_cluster_passes(self):
        # The two `Status: failed` texts present in the green round
        # tmp_20261011_121148 (scanned: 10 logs, 4 x 'Maximum optimization
        # steps reached', 6 x 'Skill internal optimization did not converge').
        # Both must be tolerated by the same table the exit-code path uses --
        # if either stopped being, #328 would turn those ten tasks red in the
        # next round and the exemption would be incomplete in a way the
        # invented sample above cannot show.
        run_dir = os.path.join(self.dir, 'cluster_vocab')
        for name, text in (
                ('opt_periodic.log',
                 'failed: Maximum optimization steps reached'),
                ('opt_surface.log',
                 'failed: Skill internal optimization did not converge')):
            with self.subTest(status=text):
                self._log(run_dir, name, text)
                result = {'status': 'pass'}
                runner._check_skill_log_status(
                    result, os.path.join(self.dir, 'case.yaml'),
                    base_dir=run_dir, title=name[:-len('.log')])
                self.assertEqual(result['status'], 'pass')

    def test_only_an_optimization_budget_stop_is_tolerated(self):
        # The tolerance is scoped to a `failed:` status that carries the
        # budget wording itself. A nested engine failure reaches the closing
        # line as `error: Composite evaluation failed (1 term(s)): task
        # 'qm_task1' (evaluation 2): SCF calculation did not converge` -- a
        # centre whose SCF never converged -- and the old anywhere-in-the-text
        # match excused it: that task was recorded PASS in round
        # tmp_20261011_184226 with a single frame written. Texts below are the
        # ones actually observed on the cluster.
        tolerated = ('failed: Maximum optimization steps reached',
                     'failed: Skill internal optimization did not converge')
        refused = ('error: Composite evaluation failed (1 term(s)): task '
                   "'qm_task1' (evaluation 2): SCF calculation did not "
                   'converge',
                   'failed: calculation failed with return code 1')
        for index, (text, expected) in enumerate(
                [(entry, 'pass') for entry in tolerated]
                + [(entry, 'fail') for entry in refused]):
            with self.subTest(status=text):
                # Its own directory and title: sharing one would let a later
                # case read an earlier subtest's log.
                run_dir = os.path.join(self.dir, 'scoped%d' % index)
                stem = 'case'
                self._log(run_dir, stem + '.log', text)
                result = {'status': 'pass'}
                runner._check_skill_log_status(
                    result, os.path.join(self.dir, 'case.json'),
                    base_dir=run_dir, title=stem)
                self.assertEqual(result['status'], expected, text)

    def test_any_other_failed_status_still_downgrades(self):
        # The exemption is the unconverged vocabulary only (`_NOT_CONVERGED_
        # SIGNATURES`, shared with the exit-code path): a real failure written
        # the same way stays a failure. 'SCF not converged' cannot serve here
        # -- 'not converge' is in that table, so that text is excused by
        # design.
        run_dir = os.path.join(self.dir, 'realfail')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case.log'), 'w',
                  encoding='utf-8') as handle:
            handle.write('PyRAI3MD End: Oct 11 2026 15:43:17 Total: 00:00:00:03 '
                         'Status: failed: calculation failed with return code 1\n')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.yaml'), base_dir=run_dir,
            title='case')
        self.assertEqual(result['status'], 'fail')

    def test_a_framework_closing_line_is_a_pass(self):
        # The framework's status does not start a line: RunLogger closes with
        # `PyRAI3MD End: <time> Total: <dt> Status: completed`, the shape every
        # example leg's log carries (a real one from round
        # tmp_20261011_085800 ends `Status: completed`). A rule that only
        # looked at a leading `Status:` would call these unjudged.
        run_dir = os.path.join(self.dir, 'end_ok')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case.log'), 'w',
                  encoding='utf-8') as handle:
            handle.write('step 1\n\nPyRAI3MD End: Oct 11 2026 09:34:10 '
                         'Total: 00:00:00:17 Status: completed\n')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.yaml'), base_dir=run_dir,
            title='case')
        self.assertEqual(result['status'], 'pass')

    def test_a_log_without_a_closing_line_is_an_unfinished_run(self):
        # The openqp frequency failures stopped inside the result printing,
        # after the banner and before any status line -- the log exists, the
        # run did not finish. Real artifact, round tmp_20261011_085800.
        run_dir = os.path.join(self.dir, 'end_missing')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case.log'), 'w',
                  encoding='utf-8') as handle:
            handle.write(' *---------------------------------------------------*\n'
                         ' |          Vibrational Frequency Analysis            |\n'
                         ' *---------------------------------------------------*\n')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.json'), base_dir=run_dir,
            title='case')
        self.assertEqual(result['status'], 'fail')
        self.assertIn('no closing status', result['details'])

    def test_lines_written_after_a_success_status_are_not_the_status(self):
        # `nn_pes_skill/agent.py` appends its training table below the
        # Status line, so the value must be read off that line alone:
        # swallowing the tail would fail a run that had just succeeded.
        run_dir = os.path.join(self.dir, 'tail')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case.log'), 'w',
                  encoding='utf-8') as handle:
            handle.write('Config: case.json\nMode: local\nStatus: success\n'
                         'Training summary: eg 2 members, 200 epochs\n')
        result = {'status': 'pass'}
        runner._check_skill_log_status(
            result, os.path.join(self.dir, 'case.json'), base_dir=run_dir,
            title='case')
        self.assertEqual(result['status'], 'pass')
        self.assertNotIn('[LOG STATUS]', result.get('details') or '')

    def test_the_positive_word_is_the_only_pass(self):
        # Two vocabularies, both read: 'success' is what a skill agent's
        # `safe_run` answers, 'completed' is the word on the framework's
        # closing line. Anything else -- an empty status, a word neither
        # writer produces -- is a run that never claimed to finish.
        for status, expected in (('success', 'pass'),
                                 ('completed', 'pass'),
                                 ('', 'fail'),
                                 ('unknown', 'fail')):
            with self.subTest(status=status):
                run_dir = os.path.join(self.dir, 'vocab',
                                       status or 'empty')
                self._log(run_dir, 'case.log', status)
                result = {'status': 'pass'}
                runner._check_skill_log_status(
                    result, os.path.join(self.dir, 'case.json'),
                    base_dir=run_dir, title='case')
                self.assertEqual(result['status'], expected)


class TestMulticenterPeriodicCarrier(unittest.TestCase):
    """A multi-centre example keeps its periodicity in the classic carrier.

    `_build_molecule_pairs` returned early in its multi-centre branch, so the
    pbc/cell emission that follows it never ran: in round
    tmp_20261011_184226 `multicenter_pbc_nve`'s `.input` carrier resolved to
    `pbc [0, 0, 0]` and the 99 A unset cell while the yaml carrier carried
    `[1, 1, 1]` and the 8.399091 box, and the centre SCF then failed because
    it was evaluated as an isolated non-periodic cluster. The periodic keys
    are now emitted from a shared helper both branches call.
    """

    _CONFIG = {'charges': 0, 'multiplicity': 1,
               'highlevel': [[10, 11, 12], [22, 23, 24]],
               'pbc': [1, 1, 1],
               'cell': [8.399091, 8.399091, 8.399091, 90.0, 90.0, 90.0]}

    def test_the_multicentre_section_carries_periodicity(self):
        section = _build_molecule_section(dict(self._CONFIG))
        self.assertIn('highlevel     10 11 12; 22 23 24', section)
        self.assertIn('pbc           1 1 1', section)
        self.assertIn('cell          8.399091 8.399091 8.399091', section)

    def test_a_non_periodic_multicentre_config_stays_unchanged(self):
        # The periodic keys are emitted from the declaration, not invented:
        # a molecular multi-centre input must not grow a 99 A default box.
        config = dict(self._CONFIG)
        del config['pbc']
        del config['cell']
        section = _build_molecule_section(config)
        self.assertNotIn('pbc', section)
        self.assertNotIn('cell', section)


class TestRunnerNotConvergedTolerance(unittest.TestCase):
    """Truncated optimize tests ending unconverged are PASS[NOT CONVERGED]."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.tmpdir, 'optimize.json')
        with open(self.input_file, 'w', encoding='utf-8') as f:
            f.write('{}\n')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _task(self):
        return {
            'name': 'example_optimize_minima_agent',
            'mode': 'pyrai3md_agent',
            'input_file': self.input_file,
            'script': os.path.join(self.tmpdir, 'run.sh'),
            'use_slurm': False,
            'skill_slurm': False,
        }

    def _run_with_proc(self, returncode, stdout='', stderr=''):
        fake_proc = mock.Mock(returncode=returncode,
                              stdout=stdout, stderr=stderr)
        with mock.patch.object(runner.subprocess, 'run',
                               return_value=fake_proc):
            return runner._run_skill_test(self._task(), {}, {})

    def test_a_signature_only_in_stderr_does_not_excuse(self):
        # A traceback is the failure's own words. Round tmp_20261104_162054:
        # the yaml carrier's input_file is the installed example, so scanning
        # beside it found no log at all and this exact phrase - quoted out of
        # the composite guard - reported four failing legs as PASS.
        stderr = ('Traceback (most recent call last):\n'
                  'RuntimeError: Maximum optimization steps reached\n')
        result = self._run_with_proc(1, stderr=stderr)

        self.assertEqual(result['status'], 'fail')
        self.assertNotIn('[NOT CONVERGED]', result['details'])
        self.assertFalse(result.get('not_converged'))

    def test_a_title_anchored_log_decides_beyond_the_input_directory(self):
        # The example carrier: the input lives in the installed package, the
        # log in the staging directory under `control.title`. The budget stop
        # is read from there, which is the same anchor the zero-exit status
        # check uses.
        run_dir = os.path.join(self.tmpdir, 'staged')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case_example.log'), 'w',
                  encoding='utf-8') as f:
            f.write('PyRAI3MD End: Nov 04 2026 19:00:00 Total: 00:00:00:09 '
                    'Status: failed: Maximum optimization steps reached\n')
        task = self._task()
        task['run_dir'] = run_dir
        task['title'] = 'case_example'
        fake_proc = mock.Mock(returncode=1, stdout='', stderr='boom\n')
        with mock.patch.object(runner.subprocess, 'run',
                               return_value=fake_proc):
            result = runner._run_skill_test(task, {}, {})

        self.assertEqual(result['status'], 'pass')
        self.assertTrue(result.get('not_converged'))

    def test_a_hard_failure_in_the_title_anchored_log_is_not_excused(self):
        run_dir = os.path.join(self.tmpdir, 'staged2')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case_example.log'), 'w',
                  encoding='utf-8') as f:
            f.write('PyRAI3MD End: Nov 04 2026 19:00:00 Total: 00:00:00:09 '
                    'Status: error: Composite evaluation failed (1 term(s)): '
                    'the two probes did not converge to the same reciprocal '
                    'sum\n')
        task = self._task()
        task['run_dir'] = run_dir
        task['title'] = 'case_example'
        fake_proc = mock.Mock(returncode=1, stdout='',
                              stderr='RuntimeError: Maximum optimization '
                                     'steps reached\n')
        with mock.patch.object(runner.subprocess, 'run',
                               return_value=fake_proc):
            result = runner._run_skill_test(task, {}, {})

        self.assertEqual(result['status'], 'fail')
        self.assertFalse(result.get('not_converged'))

    def test_nonzero_with_run_log_signature_passes_not_converged(self):
        with open(os.path.join(self.tmpdir, 'optimize.log'), 'w',
                  encoding='utf-8') as f:
            f.write('PyRAI3MD End: ... Status: failed: '
                    'Maximum optimization steps reached\n')
        result = self._run_with_proc(1, stderr='some non-zero exit\n')

        self.assertEqual(result['status'], 'pass')
        self.assertIn('[NOT CONVERGED]', result['details'])
        self.assertTrue(result.get('not_converged'))

    def test_nonzero_without_signature_stays_fail(self):
        result = self._run_with_proc(
            1, stderr='RuntimeError: Molcas calculation failed '
                      'with return code 127\n')

        self.assertEqual(result['status'], 'fail')
        self.assertNotIn('[NOT CONVERGED]', result['details'])
        self.assertFalse(result.get('not_converged'))

    def test_an_error_status_cannot_excuse_itself(self):
        # The wording of a hard failure is not a budget stop. #451's round
        # tmp_20261104_124614: the composite field probe failed with an error
        # status whose sentence contains `did not converge`, and four legs
        # were recorded PASS off that phrase.
        stderr = ('PyRAI3MD End: Nov 04 2026 12:50:04 Total: 00:00:00:45 '
                  'Status: error: Composite evaluation failed (1 term(s)): '
                  "task 'field' (evaluation 2): qmmm_field V does not sum to "
                  'the group/group Q-M energy: the two probes did not '
                  'converge to the same reciprocal sum\n')
        result = self._run_with_proc(1, stderr=stderr)

        self.assertEqual(result['status'], 'fail')
        self.assertNotIn('[NOT CONVERGED]', result['details'])
        self.assertFalse(result.get('not_converged'))

    def test_an_error_status_in_a_log_outweighs_a_signature_elsewhere(self):
        with open(os.path.join(self.tmpdir, 'optimize.log'), 'w',
                  encoding='utf-8') as f:
            f.write('PyRAI3MD End: ... Status: error: Composite evaluation '
                    "failed (1 term(s)): task 'field' (evaluation 2): "
                    'the two probes did not converge\n')
        result = self._run_with_proc(
            1, stderr='RuntimeError: Maximum optimization steps reached\n')

        self.assertEqual(result['status'], 'fail')

    def test_zero_exit_behaviour_unchanged(self):
        result = self._run_with_proc(0, stdout='calculation done\n')

        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['details'], 'calculation done\n')

    def test_zero_exit_with_the_budget_status_is_marked_too(self):
        # The exemption covered both routes and marked only one of them: a leg
        # that stops on its step budget, exits 0 and says so in its log was
        # recorded as a plain pass, which is how every one of the 32
        # variable-cell legs of round `tmp_20261019_200652` read. The verdict
        # stays `pass` - a truncated test optimization is expected to stop
        # unconverged - and what #399 added is that the row says which kind of
        # pass it is, on this route as on the other.
        with open(os.path.join(self.tmpdir, 'optimize.log'), 'w',
                  encoding='utf-8') as f:
            f.write('PyRAI3MD End: Oct 19 2026 21:04:16 Total: 00:00:12:08 '
                    'Status: failed: Maximum optimization steps reached\n')
        result = self._run_with_proc(0, stdout='calculation done\n')

        self.assertEqual(result['status'], 'pass')
        self.assertIn('[NOT CONVERGED]', result['details'])
        self.assertTrue(result.get('not_converged'))
        self.assertIn('calculation done', result['details'])

    def test_zero_exit_with_a_converged_status_carries_no_marker(self):
        # The other side of the same rule: a run whose own log says it finished
        # must not be marked, or the marker stops meaning anything. This is the
        # arm that keeps the change from marking every zero-exit leg.
        with open(os.path.join(self.tmpdir, 'optimize.log'), 'w',
                  encoding='utf-8') as f:
            f.write('PyRAI3MD End: Oct 19 2026 21:19:33 Total: 00:00:00:18 '
                    'Status: completed\n')
        result = self._run_with_proc(0, stdout='calculation done\n')

        self.assertEqual(result['status'], 'pass')
        self.assertNotIn('[NOT CONVERGED]', result['details'])
        self.assertFalse(result.get('not_converged'))


class TestEmbeddedCompositeIsNotJudgedForEtot(unittest.TestCase):
    """A multi-center embedded composite is not a closed potential to judge.

    The criterion compares a recorded total against a band; an embedded
    composite re-computes its environment charges every evaluation and answers
    the gradient at fixed charges, so with several centers the response term
    that couples them never enters the force. `multicenter_qmmm_md.yaml`
    measured 8.4e-3 / 9.7e-3 Hartree over three frames against a 1e-3 budget,
    while its own kinetic energy swings 6.5e-3 over the same frames. ONIOM's
    outer (two-layer) or mid (three-layer) charges are re-computed the same
    way, so the same exemption applies.
    """

    def _spec(self, highlevel, embedding=True, kind='qmmm', theory=None,
              read_charge=False):
        return {'control': {'multiscale_type': kind,
                            'theory': theory or ['orca', 'pyscf']},
                'molecule': {'embedding': embedding,
                             'read_charge': read_charge,
                             'highlevel': highlevel}}

    def test_two_centers_are_exempt_with_a_stated_reason(self):
        reason = runner._embedding_breaks_conservation(
            self._spec([[1, 2, 3], [10, 11, 12]]))
        self.assertTrue(reason)
        self.assertIn('response term', reason)

    def test_one_center_keeps_the_criterion(self):
        self.assertIsNone(
            runner._embedding_breaks_conservation(self._spec([1, 2, 3])))
        self.assertIsNone(
            runner._embedding_breaks_conservation(self._spec([[1, 2, 3]])))

    def test_without_embedding_there_is_nothing_to_excuse(self):
        self.assertIsNone(runner._embedding_breaks_conservation(
            self._spec([[1, 2], [3, 4]], embedding=False)))

    def test_oniom_multi_center_is_also_exempt(self):
        reason = runner._embedding_breaks_conservation(
            self._spec([[1, 2], [3, 4]], kind='oniom'))
        self.assertTrue(reason)
        self.assertIn('outer/mid charges', reason)

    def test_oniom_single_center_embedded_is_exempt(self):
        reason = runner._embedding_breaks_conservation(
            self._spec([1, 2, 3], kind='oniom'))
        self.assertTrue(reason)
        self.assertIn('outer/mid charges', reason)

    def test_oniom_read_charge_stays_judged(self):
        self.assertIsNone(runner._embedding_breaks_conservation(
            self._spec([1, 2, 3], kind='oniom', read_charge=True)))

    def test_oniom_openqp_stays_judged(self):
        # openqp writes no point charges, so its run is a closed potential.
        self.assertIsNone(runner._embedding_breaks_conservation(
            self._spec([1, 2, 3], kind='oniom',
                       theory=['openqp', 'xtb'])))

    def test_another_composite_scheme_is_not_covered(self):
        self.assertIsNone(runner._embedding_breaks_conservation(
            self._spec([[1, 2], [3, 4]], kind='fragment')))


class TestSixPropertyJobtypeTokens(unittest.TestCase):
    """energy/gradient/nac/soc/dipole/transition are basic freely
    composable jobtype tokens; the planner maps them onto the
    property capabilities (transition -> transition_dipole)."""

    def test_dipole_is_basic_and_composable(self):
        tokens, is_basic, is_advanced = parse_jobtype(
            'energy gradient dipole')
        self.assertEqual(tokens, ['energy', 'gradient', 'dipole'])
        self.assertTrue(is_basic)
        self.assertFalse(is_advanced)

    def test_transition_is_basic_and_composable(self):
        tokens, is_basic, is_advanced = parse_jobtype(
            'energy transition')
        self.assertEqual(tokens, ['energy', 'transition'])
        self.assertTrue(is_basic)
        self.assertFalse(is_advanced)

    def test_full_six_property_combination(self):
        tokens, is_basic, _ = parse_jobtype(
            'energy gradient nac soc dipole transition')
        self.assertEqual(
            tokens,
            ['energy', 'gradient', 'nac', 'soc', 'dipole', 'transition'])
        self.assertTrue(is_basic)

    def test_planner_capabilities_for_dipole_combinations(self):
        from pyrai3md.planning import resolve_capabilities
        tokens, _, _ = parse_jobtype('energy gradient dipole')
        result = resolve_capabilities(tokens)
        names = result['property_names']
        self.assertIn('dipole', names)
        self.assertIn('energy', names)
        self.assertIn('gradient', names)

    def test_planner_capabilities_for_transition(self):
        from pyrai3md.planning import resolve_capabilities
        tokens, _, _ = parse_jobtype('transition')
        result = resolve_capabilities(tokens)
        names = result['property_names']
        self.assertIn('transition_dipole', names)
        self.assertIn('energy', names)  # dependency

    def test_planner_capabilities_from_dynamics_triggers(self):
        # dodipole/radiative in the dynamics section trigger the
        # dipole / transition_dipole capabilities at planning time.
        from pyrai3md.planning import resolve_capabilities
        tokens, _, _ = parse_jobtype('energy gradient')
        result = resolve_capabilities(
            tokens, planning_context={'dodipole': 1, 'radiative': 'true'})
        names = result['property_names']
        self.assertIn('dipole', names)
        self.assertIn('transition_dipole', names)


class TestSpellSkillJobtype(unittest.TestCase):
    """Skill jobtype is spelled from planning capabilities in the
    fixed canonical order with capability-to-token back-mapping."""

    def test_canonical_order(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        # capabilities arrive sorted from planning; the spelled
        # jobtype follows the fixed token order regardless.
        caps = sorted(['energy', 'gradient', 'nac', 'soc',
                       'dipole', 'transition_dipole'])
        self.assertEqual(
            spell_skill_jobtype(caps, 'energy'),
            'energy gradient nac soc dipole transition')

    def test_capability_back_mapping(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        # A force-constant request carries its own energy evaluation, and its
        # dependency list adds `energy`/`gradient` to the capability set for
        # every backend. Spelling those as extra tokens handed the skill
        # `energy gradient frequency`, a jobtype no skill has a profile for
        # (`skills/pyscf_skill/schema/method.py` offers `energy frequency`), so
        # the implied capabilities are absorbed rather than repeated.
        self.assertEqual(
            spell_skill_jobtype(
                ['energy', 'gradient', 'hessian'], 'freq'),
            'frequency')
        # The absorption is specific to `hessian`: a plain gradient job keeps
        # the two-token form every property profile expects.
        self.assertEqual(
            spell_skill_jobtype(['energy', 'gradient'], 'gradient'),
            'energy gradient')
        self.assertEqual(
            spell_skill_jobtype(['energy', 'transition_dipole'], 'sp'),
            'energy transition')

    def test_overlap_capability_is_skipped(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        self.assertEqual(
            spell_skill_jobtype(['energy', 'gradient', 'overlap'], 'md'),
            'energy gradient')

    def test_empty_capabilities_fall_back(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        self.assertEqual(spell_skill_jobtype([], 'energy'), 'energy')
        self.assertEqual(spell_skill_jobtype(None, 'md'), 'md')


class TestConverterDipoleJobtype(unittest.TestCase):
    """Dipole skill profiles convert into framework jobtype tokens.

    'transition_dipole' and 'dipole' are not jobtype words on their
    own: the converter spells them with the basic 'dipole' and
    'transition' tokens so parse_jobtype accepts the result (#122
    regression: 'transition_dipole' hit Unknown jobtype remotely).
    """

    def test_transition_dipole_profile_spells_transition_token(self):
        self.assertEqual(
            _resolve_jobtype({'profile': 'tddft_transition_dipole'}),
            'energy transition')
        tokens, is_basic, is_advanced = parse_jobtype(
            _resolve_jobtype({'profile': 'tddft_transition_dipole'}))
        self.assertEqual(tokens, ['energy', 'transition'])
        self.assertTrue(is_basic)
        self.assertFalse(is_advanced)

    def test_dipole_profile_spells_dipole_token(self):
        self.assertEqual(
            _resolve_jobtype({'profile': 'dft_dipole'}),
            'energy dipole')
        tokens, is_basic, _ = parse_jobtype(
            _resolve_jobtype({'profile': 'dft_dipole'}))
        self.assertEqual(tokens, ['energy', 'dipole'])
        self.assertTrue(is_basic)

    def test_explicit_jobtype_still_passes_through(self):
        self.assertEqual(
            _resolve_jobtype({'jobtype': 'energy gradient'}),
            'energy gradient')


class TestDeriveCurrentState(unittest.TestCase):
    """normalize fills system.current_state from the dynamics root
    only while current_state stays at its default (0)."""

    def test_nondefault_root_is_derived(self):
        from pyrai3md.schema.normalize import derive_current_state
        data = derive_current_state({'root': 2, 'current_state': 0})
        self.assertEqual(data['current_state'], 2)

    def test_explicit_current_state_is_not_overridden(self):
        from pyrai3md.schema.normalize import derive_current_state
        data = derive_current_state({'root': 0, 'current_state': 1})
        self.assertEqual(data['current_state'], 1)

    def test_double_default_is_identity(self):
        from pyrai3md.schema.normalize import derive_current_state
        data = derive_current_state({'root': 0, 'current_state': 0})
        self.assertEqual(data['current_state'], 0)
        data = derive_current_state({})
        self.assertEqual(data['current_state'], 0)



if __name__ == "__main__":
    unittest.main()
