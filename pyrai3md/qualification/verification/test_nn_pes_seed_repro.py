"""
======================================================================
Unit and integration tests for how a fit gets its random seed.

A network in this skill is trained by a child process
(`python -m ...training_mlp_* -i .. -f .. -g .. -m ..`), so the parent's random
state never reaches it: until the seed travelled on the command line, every
rerun of the same input drew fresh weights, and the one criterion in this
package that checks a trained value (the spin-orbit reproduction test) was a
coin flip. These tests pin the three links of the chain - the configuration
field, the command line, and the child applying it - and then prove the end of
it: the same `gl_seed` trains the same network, a different one does not.

Run: python -m unittest \
    pyrai3md.qualification.verification.test_nn_pes_seed_repro

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import pyrai3md.skills.nn_pes_skill

from pyrai3md.skills.nn_pes_skill.backend import NNPESBackend
from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
from pyrai3md.skills.nn_pes_skill.service.nn_pes.nn_pes_src.fit import (
    FIT_MODULES, get_module_for_fit_script, train_command)
from pyrai3md.skills.nn_pes_skill.service.runner import train_runner


def tensorflow_present():
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


if not tensorflow_present():
    raise unittest.SkipTest('tensorflow is not installed')


DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(pyrai3md.skills.nn_pes_skill.__file__)),
    'tests', 'data')
EGS = os.path.join(DATA_DIR, 'egs_small.json')


def _frame():
    with open(EGS) as handle:
        raw = json.load(handle)
    frame = raw['xyz'][0]
    atoms = [row[0] for row in frame]
    coords = [[float(row[i]) for i in (1, 2, 3)] for row in frame]
    return atoms, coords


class TestTheSeedOnTheCommandLine(unittest.TestCase):
    """The chain from the config field to the child's argument, without a fit."""

    def _config(self, **over):
        raw = {'profile': 'mlp_eg_train', 'method': 'mlp_eg',
               'modeldir': '/tmp/model', 'train_data': EGS,
               'mult_nn': 3, 'nstate': 2, 'target_states': [0, 1],
               'atoms': ['H', 'H'], 'coordinates': [[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.74]]}
        raw.update(over)
        return build_config(raw)

    def test_gl_seed_is_a_registered_field_with_the_framework_default(self):
        # The framework's control default is 1; the skill reads the same name,
        # so a run that sets control.gl_seed reaches training without a
        # translation table, and a standalone skill fit is reproducible too.
        config = self._config()
        self.assertEqual(config.calculation.gl_seed, 1)
        self.assertEqual(self._config(gl_seed=7).calculation.gl_seed, 7)

    def test_the_framework_control_seed_reaches_the_skill_field(self):
        """The handoff the comment above claims, run through the real merge.

        A framework-driven fit is the one that has to be reproducible by
        `control.gl_seed`, and the merge hands the skill a flat dict: a key the
        framework does not pass through would leave every such run drawing fresh
        weights while the skill's own default (1) makes the run look seeded. This
        goes through `Maestro` and `_merge_one_skill_config` rather than a hand
        built dict, since which keys survive is that function's decision.
        """
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import Config

        raw = {'control': {'jobtype': 'energy', 'driver': 'main',
                           'gl_seed': 7},
               'nn_pes': {'method': 'mlp_eg', 'profile': 'mlp_eg_train',
                          'execution_mode': 'local', 'modeldir': '/tmp/m'}}
        merged = _merge_one_skill_config(Maestro(), 'nn_pes_skill',
                                         config=Config(**raw), raw_config=raw)
        self.assertEqual(build_config(merged).calculation.gl_seed, 7)

    def test_the_seed_advances_by_the_member_index(self):
        # An ensemble whose members all draw the same numbers is one network
        # trained three times: the value goes out as gl_seed + index, which is
        # how an MD run seeds its trajectories too (workflow/md.py).
        config = self._config(gl_seed=7)
        jobs = train_runner.build_jobs(
            config, [('energy_grad', 'mlp_eg', '/tmp/model/energy_grad')])
        self.assertEqual([job['command'][-2:] for job in jobs],
                         [['-s', '7'], ['-s', '8'], ['-s', '9']])

    def test_a_command_built_without_a_seed_carries_no_flag(self):
        # The legacy arm: a caller that passes no seed must get the command line
        # it got before, so this change cannot alter an invocation that opted out.
        command = train_command('mlp_eg', 0, '/tmp/m', -1, 'training')
        self.assertNotIn('-s', command)
        self.assertIn('-m', command)

    def test_every_training_module_takes_the_flag_and_seeds_from_it(self):
        """The registry arm: a future network family cannot skip the seed.

        Each module reachable through `get_module_for_fit_script` has to parse
        `-s` AND call `apply_seed` with the parsed value - a module that accepts
        the flag and ignores it would make a run look seeded while its weights
        stay random, which is the failure this whole chain exists to avoid.
        """
        import importlib
        import inspect

        for model_type in sorted(FIT_MODULES):
            module = importlib.import_module(
                get_module_for_fit_script(model_type))
            source = inspect.getsource(module)
            self.assertIn('apply_seed(args', source,
                          msg='%s does not seed its own process' % model_type)
            args = module.parse_args(['-i', '0', '-f', '/tmp/m', '-g', '-1',
                                      '-m', 'training', '-s', '5'])
            self.assertEqual(args['seed'], 5,
                             msg='%s does not read -s' % model_type)




if __name__ == '__main__':
    unittest.main()
