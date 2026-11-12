"""
======================================================================
Unit tests for the exit code of the five member-fit scripts
(service/nn_pes/nn_pes_src/training/training_mlp_*.py).

The framework's trainer reads nothing but the exit code of a spawned fit,
so a child that cannot find its data, its hyper file or its scaler used
to write an error line into the model folder's log and still leave with
status 0. The round then reported a trained ensemble that had never been
trained. Each of the five scripts now answers with 1 whenever the fit
produced no validation error, and the handlers that swallow the load
failure return instead of falling through.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.skills.nn_pes_skill.service.nn_pes.nn_pes_src.training import (
    training_mlp_e,
    training_mlp_eg,
    training_mlp_g2,
    training_mlp_nac,
    training_mlp_nac2,
)

SCRIPTS = sorted(name for name in os.listdir(
    os.path.dirname(training_mlp_eg.__file__))
    if name.startswith('training_mlp_') and name.endswith('.py'))


class TestFitExitCode(unittest.TestCase):
    """A fit that loaded nothing must not report success."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fit_exit_')
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _run(self, module):
        return module.main(['--index', '0',
                            '--filepath', self.dir,
                            '--gpus', '-1',
                            '--mode', 'training',
                            '--seed', '7'])

    def test_a_folder_without_training_data_exits_nonzero(self):
        # The five scripts share the shape of load_data/train_model/main, so
        # the reading is recorded once per module: exit 0 here is the defect.
        for module in (training_mlp_e, training_mlp_eg, training_mlp_g2,
                       training_mlp_nac, training_mlp_nac2):
            with self.subTest(script=module.__name__.rsplit('.', 1)[-1]):
                self.assertEqual(self._run(module), 1)

    def test_the_five_scripts_are_the_ones_covered(self):
        self.assertEqual(len(SCRIPTS), 5,
                         'a new member-fit script must be added to the '
                         'exit-code coverage above')


class TestFitHistoryFile(unittest.TestCase):
    """Every model version writes its own training-history file."""

    def test_the_history_file_carries_the_version_index(self):
        # v0..vN train into one directory. A shared `history_.json` let
        # one version's write land on another's (round tmp_20261111_195518
        # kept 768 bytes of mixed JSON), so the name follows weights_v%i.
        for module in (training_mlp_e, training_mlp_eg, training_mlp_g2,
                       training_mlp_nac, training_mlp_nac2):
            with self.subTest(script=module.__name__.rsplit('.', 1)[-1]):
                with open(module.__file__, encoding='utf-8') as handle:
                    source = handle.read()
                self.assertIn('"history_v%i.json" % i', source)
                self.assertNotIn('"history_" + ".json"', source)


if __name__ == '__main__':
    unittest.main()
