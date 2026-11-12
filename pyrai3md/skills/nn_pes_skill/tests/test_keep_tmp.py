"""
======================================================================
This is a standalone machine learning force field skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

What survives a run (CHANGELOG #307), tested as the directory question
it actually is.

``keep_tmp`` was declared in this skill's schema and read by nobody, so
every run left its calculation directory behind whatever the input said. It
now clears it, but at the end of the backend's life rather than at the end
of a step, and that difference is the point of these cases:

The calculation directory holds files a serving run *accumulates*
(``{project}_{profile}.log`` and the uncertainty dump), and every writer
opens them in append mode without recreating the directory. Clearing the
directory inside ``calculate()`` would therefore keep only the last step
and then stop recording silently, because the ``OSError`` of a missing
parent is swallowed by the writers. The teardown is bound to the backend
object through ``weakref.finalize``, the lifecycle the resident LAMMPS
service already uses.

Nothing above is about neural networks, so no neural network is started:
which directories a backend creates, keeps and deletes is file management,
and the container plus the environment guard are replaced by the stand-ins
in ``_fake_engine``. The real route through a trained folder is certified
by ``00_mlp_eg_train.yaml`` and the serving cases that follow it, which the
qualification suite runs through the three entry points.

Run: pytest skills/nn_pes_skill/tests/test_keep_tmp.py -v
"""

import gc
import os
import shutil
import tempfile
import unittest
from unittest import mock

from pyrai3md.skills.nn_pes_skill.backend import NNPESBackend
from pyrai3md.skills.nn_pes_skill.tests._fake_engine import FakeContainer

ATOMS = ['C', 'H']
COORDS = [[0.5, 0.25, -1.0], [0.75, -0.125, 0.0]]


class TestKeepTmp(unittest.TestCase):
    """The directory is cleared when the run is done with it, not before."""

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix='nn_pes_keep_tmp_')
        self.addCleanup(shutil.rmtree, self.work, True)
        self.model_dir = os.path.join(self.work, 'model')
        os.makedirs(self.model_dir)
        # Three seams, each a name in the backend's own namespace: the guard
        # that says whether TensorFlow can be imported (a claim
        # test_discovery_without_tensorflow.py owns), and the two calls that
        # would otherwise build and load real networks.
        patches = [
            mock.patch(
                'pyrai3md.skills.nn_pes_skill.backend.validate_environment'),
            mock.patch('pyrai3md.skills.nn_pes_skill.backend.import_container',
                       return_value=FakeContainer),
            mock.patch('pyrai3md.skills.nn_pes_skill.backend.open_container',
                       side_effect=lambda config, _container: FakeContainer(
                           nstate=2, natom=len(ATOMS))),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _backend(self, tag, **over):
        cfg = {
            'profile': 'mlp_eg_gradient',
            'method': 'mlp_eg',
            'modeldir': self.model_dir,
            'mult_nn': 2,
            'nstate': 2,
            'target_states': [0, 1],
            'write_step_log': 1,
            'atoms': ATOMS,
            'coordinates': COORDS,
            'scratch_path': os.path.join(self.work, 'run'),
            'project': 'KTMP',
            'execution_mode': 'local',
        }
        cfg.update(over)
        # A distinct execution namespace per case: backends that share one
        # would share the directory the assertions are about.
        return NNPESBackend(cfg, execution_id='tmp_NNPES_%s' % tag)

    @staticmethod
    def _steps(backend):
        log = os.path.join(backend.calcdir, 'KTMP_mlp_eg_gradient.log')
        if not os.path.isfile(log):
            return 0
        with open(log) as handle:
            return handle.read().count('step  profile=')

    def test_the_directory_survives_the_steps_it_records(self):
        backend = self._backend('steps', keep_tmp=0)
        backend.calculate(ATOMS, COORDS)
        backend.calculate(ATOMS, COORDS)
        self.assertTrue(os.path.isdir(backend.calcdir),
                        'a stepped run would lose its history if the '
                        'cleanup ran per step')
        self.assertEqual(self._steps(backend), 2,
                         'both steps are recorded while the run lives')

    def test_dropping_the_backend_clears_it_when_keep_tmp_is_off(self):
        backend = self._backend('clear', keep_tmp=0)
        backend.calculate(ATOMS, COORDS)
        calcdir = backend.calcdir
        del backend
        gc.collect()
        self.assertFalse(os.path.exists(calcdir),
                         'the keyword was declared and read by nobody')

    def test_dropping_the_backend_keeps_it_when_keep_tmp_is_on(self):
        backend = self._backend('keep', keep_tmp=1)
        backend.calculate(ATOMS, COORDS)
        calcdir = backend.calcdir
        del backend
        gc.collect()
        self.assertTrue(os.path.isdir(calcdir))
        with open(os.path.join(calcdir, 'KTMP_mlp_eg_gradient.log')) as handle:
            self.assertIn('step  profile=', handle.read())

    def test_the_trained_ensemble_is_not_keep_tmp_scratch(self):
        # modeldir holds what the training run produced; no keyword governs
        # it, so a cleared calculation directory cannot lose the surface.
        folder = os.path.join(self.model_dir, 'energy_grad')
        os.makedirs(folder)
        for name in ('weights_v0.weights.h5', 'fiterr_valid_v0.npy'):
            with open(os.path.join(folder, name), 'w') as handle:
                handle.write('placeholder')
        backend = self._backend('weights', keep_tmp=0)
        backend.calculate(ATOMS, COORDS)
        calcdir = backend.calcdir
        del backend
        gc.collect()
        self.assertFalse(os.path.exists(calcdir))
        for name in ('weights_v0.weights.h5', 'fiterr_valid_v0.npy'):
            self.assertTrue(os.path.isfile(os.path.join(folder, name)))

    def test_the_uncertainty_dump_accumulates_like_the_log(self):
        # The same append-without-recreate contract the teardown exists to
        # protect: one line per served step, not one file per step.
        backend = self._backend('uncertainty', keep_tmp=1)
        for _step in range(3):
            backend.calculate(ATOMS, COORDS)
        path = os.path.join(backend.calcdir,
                            'KTMP_mlp_eg_gradient.uncertainty.txt')
        with open(path) as handle:
            lines = [line for line in handle.read().splitlines() if line]
        self.assertEqual(3, len(lines))


if __name__ == '__main__':
    unittest.main()
