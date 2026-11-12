"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Tests of the training launcher: one subprocess per ensemble member, the
same command line under both arms, a package prefix that is computed at
runtime, and the single-source rule for the vendored tree.

These tests need no TensorFlow.

"""

import ast
import os
import pathlib
import shutil
import subprocess
import sys
import unittest

from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
from pyrai3md.skills.nn_pes_skill.service.nn_pes.nn_pes_src.fit import (
    child_environment, get_module_for_fit_script, train_command)
from pyrai3md.skills.nn_pes_skill.service.runner import slurm_runner
from pyrai3md.skills.nn_pes_skill.service.runner.train_runner import (
    build_jobs, failed_runs)

SKILL_ROOT = pathlib.Path(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG = {
    'profile': 'mlp_eg_train', 'method': 'mlp_eg',
    'modeldir': '/tmp/models',
    'mult_nn': 3, 'nstate': 2, 'atoms': ['C', 'H'],
    'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    'train_data': '/tmp/data.json', 'scratch_path': '',
    'execution_mode': 'local', 'target_states': [0, 1],
}


def make_config(**over):
    raw = dict(CFG)
    raw.update(over)
    return normalize(build_config(raw))


class TestModulePrefix(unittest.TestCase):

    def test_prefix_is_computed_from_the_package(self):
        module = get_module_for_fit_script('mlp_eg')
        self.assertTrue(module.endswith('.nn_pes_src.training.training_mlp_eg'),
                        module)
        # The prefix is this package's own import name, not a hard-coded
        # path: the same module string works for a standalone checkout.
        self.assertNotIn('NNsMD', module)

    def test_every_model_type_has_a_training_module(self):
        for model_type in ('mlp_e', 'mlp_eg', 'mlp_g2', 'mlp_nac', 'mlp_nac2'):
            module = get_module_for_fit_script(model_type)
            name = module.rsplit('.', 1)[1]
            path = SKILL_ROOT / 'service' / 'nn_pes' / 'nn_pes_src' / 'training'
            self.assertTrue((path / (name + '.py')).exists(), module)

    def test_unknown_model_type_raises(self):
        with self.assertRaises(KeyError):
            get_module_for_fit_script('mlp_nothing')


class TestCommand(unittest.TestCase):

    def test_command_carries_the_launcher_arguments(self):
        command = train_command('mlp_eg', 2, '/tmp/models/eg', -1, 'training')
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], '-m')
        self.assertEqual(command[3:5], ['-i', '2'])
        self.assertIn('/tmp/models/eg', command)
        self.assertIn('training', command)

    def test_child_environment_carries_sys_path(self):
        import sys
        environment = child_environment()
        merged = environment['PYTHONPATH']
        for entry in sys.path:
            if entry and os.path.isabs(entry):
                self.assertIn(entry, merged.split(os.pathsep))


class TestOneProcessPerMember(unittest.TestCase):

    def test_jobs_expand_over_members(self):
        config = make_config(mult_nn=3)
        targets = [('energy_grad', 'mlp_eg', '/tmp/models/energy_grad'),
                   ('nac', 'mlp_nac2', '/tmp/models/nac')]
        jobs = build_jobs(config, targets)
        self.assertEqual(len(jobs), 6)
        self.assertEqual({job['index'] for job in jobs}, {0, 1, 2})
        self.assertEqual({job['name'] for job in jobs},
                         {'energy_grad', 'nac'})

    def test_each_job_targets_one_folder(self):
        jobs = build_jobs(make_config(mult_nn=2),
                          [('energy_grad', 'mlp_eg', '/tmp/models/energy_grad')])
        self.assertTrue(all(job['folder'] == '/tmp/models/energy_grad'
                            for job in jobs))

    def test_the_device_of_each_member_reaches_its_command(self):
        # ';' on a shared keyword is the same member axis the hyper
        # sections use, so two members can train on two devices.
        jobs = build_jobs(make_config(mult_nn=2, gpu='0;1'),
                          [('energy_grad', 'mlp_eg', '/tmp/models/eg')])
        devices = [job['command'][job['command'].index('-g') + 1]
                   for job in jobs]
        self.assertEqual(devices, ['0', '1'])

    def test_no_device_is_the_default(self):
        jobs = build_jobs(make_config(mult_nn=2),
                          [('energy_grad', 'mlp_eg', '/tmp/models/eg')])
        for job in jobs:
            self.assertEqual(job['command'][job['command'].index('-g') + 1],
                             '-1')


class TestArmsShareOneCommand(unittest.TestCase):

    def setUp(self):
        self.workdir = '/tmp/nn_pes_launcher_test'
        shutil.rmtree(self.workdir, ignore_errors=True)
        os.makedirs(self.workdir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_slurm_script_runs_the_local_command(self):
        config = make_config(execution_mode='slurm', slurm_partition='gpu',
                             slurm_time='02:00:00', slurm_memory=8000,
                             slurm_ntasks=4)
        job = build_jobs(config,
                         [('energy_grad', 'mlp_eg', self.workdir)])[0]
        script = slurm_runner.write_job_script(job['command'], self.workdir,
                                               config, 'eg_v0')
        with open(script) as handle:
            text = handle.read()
        self.assertIn('#!/bin/bash', text)
        self.assertIn('#SBATCH -p gpu', text)
        self.assertIn('#SBATCH -t 02:00:00', text)
        self.assertIn('#SBATCH --mem 8000', text)
        self.assertIn('#SBATCH -n 4', text)
        # The body is the local arm's command line, unchanged
        self.assertIn(' '.join(job['command']), text)

    def test_the_job_script_carries_the_thread_count(self):
        """A submitted member is told its training threads (#306)."""
        for thread, expected in ((4, 'export OMP_NUM_THREADS=4'),
                                 (1, 'export OMP_NUM_THREADS=1')):
            config = make_config(execution_mode='slurm', thread=thread)
            job = build_jobs(config, [('energy_grad', 'mlp_eg',
                                       self.workdir)])[0]
            script = slurm_runner.write_job_script(job['command'],
                                                   self.workdir, config,
                                                   'eg_v0')
            with open(script) as handle:
                text = handle.read()
            self.assertIn(expected, text,
                          'the job runs in its own process, so the count '
                          'it uses is whatever the script exports')

    def test_train_thread_overrides_the_serving_thread(self):
        """The job script exports train_thread, not predict_thread."""
        config = make_config(execution_mode='slurm', thread=4,
                             train_thread=2, predict_thread=6)
        job = build_jobs(config, [('energy_grad', 'mlp_eg',
                                   self.workdir)])[0]
        script = slurm_runner.write_job_script(job['command'], self.workdir,
                                               config, 'eg_v0')
        with open(script) as handle:
            text = handle.read()
        self.assertIn('export OMP_NUM_THREADS=2', text)
        self.assertIn('export TF_NUM_INTRAOP_THREADS=2', text)
        self.assertIn('export TF_NUM_INTEROP_THREADS=2', text)

    def test_empty_train_and_predict_threads_inherit_thread(self):
        from pyrai3md.skills.nn_pes_skill.schema.config import Environment
        env = Environment(thread=5, train_thread=0, predict_thread=0)
        self.assertEqual(env.train_thread, 5)
        self.assertEqual(env.predict_thread, 5)
        env = Environment(thread=5, train_thread=2, predict_thread=3)
        self.assertEqual(env.train_thread, 2)
        self.assertEqual(env.predict_thread, 3)

    def test_child_environment_carries_the_training_budget(self):
        from pyrai3md.skills.nn_pes_skill.service.nn_pes.nn_pes_src.fit import (
            child_environment,
        )
        env = child_environment(2)
        self.assertEqual(env['OMP_NUM_THREADS'], '2')
        self.assertEqual(env['TF_NUM_INTRAOP_THREADS'], '2')
        self.assertEqual(env['TF_NUM_INTEROP_THREADS'], '2')

    def test_missing_sbatch_is_reported_not_faked(self):
        if shutil.which('sbatch'):
            self.skipTest('sbatch is present; the absence path cannot be '
                          'exercised here')
        with self.assertRaises(RuntimeError) as caught:
            slurm_runner.submit_blocking('/tmp/whatever.sbatch', self.workdir)
        self.assertIn("sbatch", str(caught.exception))

    def test_failures_are_collected_by_return_code(self):
        records = [{'name': 'eg', 'index': 0, 'returncode': 0, 'log': 'a'},
                   {'name': 'eg', 'index': 1, 'returncode': 1, 'log': 'b'}]
        broken = failed_runs(records)
        self.assertEqual(len(broken), 1)
        self.assertEqual(broken[0]['index'], 1)


class TestSingleSource(unittest.TestCase):
    """The vendored copy must not reach back to the frozen project."""

    def test_no_import_of_the_original_package(self):
        offenders = []
        for path in (SKILL_ROOT / 'service' / 'nn_pes').rglob('*.py'):
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                if ('import NNsMD' in stripped
                        or 'from NNsMD' in stripped
                        or '"NNsMD.' in stripped
                        or "'NNsMD." in stripped):
                    offenders.append('%s: %s' % (path.name, stripped))
        self.assertEqual(offenders, [])

    def test_no_framework_import_in_skill_code(self):
        # The contract is that a skill never imports the framework. This
        # reads the syntax tree, so a usage example in a docstring (which
        # legitimately names the installed path) is not mistaken for an
        # import.
        offenders = []
        for path in SKILL_ROOT.rglob('*.py'):
            # The tests themselves are framework-side code and import the
            # package by name; the rule covers the skill and its vendored
            # tree.
            if (SKILL_ROOT / 'tests') in path.parents:
                continue
            tree = ast.parse(path.read_text())
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value,
                                                             ast.Constant) \
                        and isinstance(node.value.value, str):
                    docstrings.add(id(node.value))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                for name in names:
                    if name.split('.')[0] == 'pyrai3md':
                        offenders.append('%s: import %s' % (path.name, name))
            # A launch path must be computed at runtime, never written in
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and node.value.startswith(tuple(
                            ['pyrai3md.', 'pyrai3md'])):
                    offenders.append('%s: literal module %s'
                                     % (path.name, node.value[:60]))
        self.assertEqual(offenders, [])



class TestThreadSetting(unittest.TestCase):
    """The first writer wins inside a process, and says so outside it."""

    _CODE = (
        "import os\n"
        "from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config\n"
        "from pyrai3md.skills.nn_pes_skill.service.runner.env_check "
        "import configure_threads\n"
        "cfg = build_config(dict(%s, **%s))\n"
        "configure_threads(cfg)\n"
        "print('EFFECTIVE=' + str(os.environ.get('OMP_NUM_THREADS')))\n")

    def _probe(self, preset, **keywords):
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
                    'TF_NUM_INTEROP_THREADS'):
            env.pop(key, None)
        if preset is not None:
            env['OMP_NUM_THREADS'] = str(preset)
        code = self._CODE % (repr(CFG), repr(keywords))
        return subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True, env=env)

    def test_an_unset_process_takes_the_requested_count(self):
        proc = self._probe(None, thread=4)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('EFFECTIVE=4', proc.stdout)
        self.assertEqual(proc.stderr.strip(), '',
                         'nothing was overridden, so nothing is reported')

    def test_an_already_latched_process_keeps_the_first_count(self):
        # OpenMP reads OMP_NUM_THREADS once, at the first parallel region,
        # so a later term cannot retune it for the others (#302).
        proc = self._probe(2, thread=5)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('EFFECTIVE=2', proc.stdout)
        self.assertIn('already 2', proc.stderr)
        self.assertIn('predict_thread=5', proc.stderr)

    def test_the_framework_spelling_reaches_the_same_place(self):
        # `nn_pes_thread` is what the framework forwards; the skill folds
        # the pair to its larger value and exports that one (#306).
        proc = self._probe(None, nn_pes_thread=3)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('EFFECTIVE=3', proc.stdout)

    def test_the_larger_of_a_written_pair_wins(self):
        proc = self._probe(None, thread=2, nn_pes_thread=6)
        self.assertIn('EFFECTIVE=6', proc.stdout)


if __name__ == '__main__':
    unittest.main()
