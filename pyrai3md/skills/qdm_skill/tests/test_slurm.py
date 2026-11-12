"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests for the SLURM execution path with a mocked sbatch
submission: config serialization into the calculation directory,
sbatch script generation, and results collection through the
'{project}_results.json' convention after the blocking submission
returns.

"""

import json
import os
import tempfile
import unittest
from unittest import mock

from pyrai3md.skills.qdm_skill.backend import QDMBackend
from pyrai3md.skills.qdm_skill.service.runner import results_file_name

_SLURM_PATCH = ('pyrai3md.skills.qdm_skill.service.runner.'
                'slurm_runner.subprocess.run')


def _slurm_task_config(scratch_path):
    """Minimal framework-style task config dict in slurm mode."""
    return {
        'skill_config': {
            'atoms': ['X'],
            'coordinates': [[-7.9376581, 0.0, 0.0]],
            'nstate': 2,
            'state_spin': [0, 0],
            'profile': 'tully1_energy',
            'method': 'tully1',
            'scratch_path': scratch_path,
            'target_states': [0, 1],
            'execution_mode': 'slurm',
            'qdm_project': 'MockSlurm',
            # These cases read back what the run wrote into the
            # calculation directory, so the directory has to survive it:
            # keep_tmp=0 (the default since #307) would clear it at the
            # end of calculate().
            'keep_tmp': 1,
        }
    }


class TestResultsFileName(unittest.TestCase):

    def test_naming_convention(self):
        self.assertEqual(
            results_file_name('/tmp/calc/MockSlurm.json'),
            '/tmp/calc/MockSlurm_results.json')


class TestSlurmSerialization(unittest.TestCase):

    def test_backend_slurm_roundtrip(self):
        with tempfile.TemporaryDirectory() as workdir:
            task_config = _slurm_task_config(workdir)
            calcdir = os.path.join(workdir, 'calc')
            backend = QDMBackend(task_config, calcdir=calcdir)

            agent_results = {'energy': [-0.01, 0.01],
                             'fin_coord': [], 'fin_velocity': []}
            config_file = os.path.join(calcdir, 'MockSlurm.json')
            results_file = results_file_name(config_file)

            def fake_sbatch(cmd, **kwargs):
                # Emulate the cluster-side agent: it runs the same
                # config json and drops the results next to it.
                self.assertTrue(os.path.isfile(config_file))
                with open(results_file, 'w') as f:
                    json.dump(agent_results, f)
                return mock.Mock(returncode=0)

            with mock.patch(_SLURM_PATCH, side_effect=fake_sbatch):
                results = backend.calculate(['X'],
                                            [[-7.9376581, 0.0, 0.0]])

            # Collected results match the agent output
            self.assertEqual(results['energy'], [-0.01, 0.01])
            self.assertEqual(results['fin_coord'], [])
            self.assertEqual(results['fin_velocity'], [])

            # Serialized config carries the nested skill settings;
            # the cluster-side agent rebuilds its own SkillConfig
            # from this structure.
            with open(config_file) as f:
                serialized = json.load(f)
            self.assertEqual(serialized['calculation']['method'],
                             'tully1')
            self.assertEqual(serialized['calculation']['profile'],
                             'tully1_energy')
            self.assertEqual(serialized['calculation']['model_mass'],
                             2000.0)

            # sbatch script invokes the skill agent module with the
            # serialized config file as its argument
            script_path = os.path.join(calcdir, 'MockSlurm.sbatch')
            with open(script_path) as f:
                script = f.read()
            self.assertIn('pyrai3md.skills.qdm_skill.agent', script)
            self.assertIn(config_file, script)

            # Results were also written back into the task config
            self.assertEqual(
                task_config['results']['energy'], [-0.01, 0.01])

    def test_sbatch_submit_command(self):
        # The runner submits the script by absolute path and anchors the
        # submission on the calculation directory with cwd=.
        with tempfile.TemporaryDirectory() as workdir:
            task_config = _slurm_task_config(workdir)
            calcdir = os.path.join(workdir, 'calc')
            backend = QDMBackend(task_config, calcdir=calcdir)

            with mock.patch(_SLURM_PATCH) as patched:
                patched.return_value = mock.Mock(returncode=0)
                os.makedirs(calcdir, exist_ok=True)
                config_file = os.path.join(calcdir, 'MockSlurm.json')
                with open(results_file_name(config_file), 'w') as f:
                    json.dump({'energy': [0.0, 0.0]}, f)
                backend.calculate(['X'], [[-7.9376581, 0.0, 0.0]])

            patched.assert_called_once()
            cmd = patched.call_args[0][0]
            self.assertEqual(cmd, ['sbatch', '-W',
                                   os.path.join(calcdir,
                                                'MockSlurm.sbatch')])
            self.assertEqual(patched.call_args[1]['cwd'], calcdir)

    def test_process_cwd_never_moves(self):
        """The interpreter's cwd stays put (#301).

        Anchoring the child with cwd= instead of os.chdir() is what lets
        two terms of one skill run in threads of the same process: a
        chdir would resolve the other term's script name in the wrong
        directory. os.chdir is patched to fail, so the guard has teeth.
        """
        with tempfile.TemporaryDirectory() as workdir:
            task_config = _slurm_task_config(workdir)
            calcdir = os.path.join(workdir, 'calc')
            backend = QDMBackend(task_config, calcdir=calcdir)
            maindir = os.getcwd()

            with mock.patch(_SLURM_PATCH) as patched, \
                    mock.patch('os.chdir',
                               side_effect=AssertionError('chdir used')):
                patched.return_value = mock.Mock(returncode=0)
                os.makedirs(calcdir, exist_ok=True)
                config_file = os.path.join(calcdir, 'MockSlurm.json')
                with open(results_file_name(config_file), 'w') as f:
                    json.dump({'energy': [0.0, 0.0]}, f)
                backend.calculate(['X'], [[-7.9376581, 0.0, 0.0]])

            self.assertEqual(os.getcwd(), maindir)
            patched.assert_called_once()


if __name__ == '__main__':
    unittest.main()
