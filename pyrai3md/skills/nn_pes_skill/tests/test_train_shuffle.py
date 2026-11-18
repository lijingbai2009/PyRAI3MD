"""
======================================================================
Unit tests of the nn_pes training-data shuffle.

`shuffle: true` permutes the frames once before they are staged to the
model folder, because the fit holds out the head of the dataset for
validation. The tests assert that frames and labels move together, that
the switch is off by default, and that the permutation is drawn from the
caller's seed so a rerun and a retraining continuation stage the same
order.

These tests need no TensorFlow.

Nov 18 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.skills.nn_pes_skill.service import trainer_data


class StagedData(unittest.TestCase):
    """A small energy/gradient set staged the way the trainer stages it."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='nn_shuffle_')
        self.addCleanup(shutil.rmtree, self.dir, True)
        frames = 20
        # Distinct rows per frame, so a permutation is visible in every
        # channel and an accidental identity order cannot pass as one.
        self.arrays = {
            'xyz': np.arange(frames * 2 * 3, dtype=np.float64)
                     .reshape(frames, 2, 3),
            'energy': np.arange(frames, dtype=np.float64)
                        .reshape(frames, 1) * -0.1,
            'grad': np.arange(frames * 1 * 2 * 3, dtype=np.float64)
                      .reshape(frames, 1, 2, 3) * 1e-3,
        }

    def _stage(self, shuffle=False, seed=None):
        trainer_data.write_to_folder(
            None, 'energy_grad', self.dir, 'energy_grad', 'mlp_eg',
            self.arrays, shuffle=shuffle, seed=seed)

    def _read(self):
        with open(os.path.join(self.dir, 'data_x'), 'rb') as handle:
            x = pickle.load(handle)
        with open(os.path.join(self.dir, 'data_y'), 'rb') as handle:
            y = pickle.load(handle)
        return x, y

    def test_the_default_stages_the_producer_order(self):
        self._stage()
        x, y = self._read()
        np.testing.assert_array_equal(x, self.arrays['xyz'])
        np.testing.assert_array_equal(y[0], self.arrays['energy'])
        np.testing.assert_array_equal(y[1], self.arrays['grad'])
        self.assertFalse(
            os.path.exists(os.path.join(self.dir, 'shuffle_index.npy')))

    def test_the_switch_permutes_frames_and_labels_together(self):
        seed = 7
        self._stage(shuffle=True, seed=seed)
        x, y = self._read()
        index = np.load(os.path.join(self.dir, 'shuffle_index.npy'))
        np.testing.assert_array_equal(
            index, np.random.default_rng(seed).permutation(len(x)))
        # Every frame moves with its energy and its gradient, so a row of
        # staged data is still one dataset entry.
        np.testing.assert_array_equal(x, self.arrays['xyz'][index])
        np.testing.assert_array_equal(y[0], self.arrays['energy'][index])
        np.testing.assert_array_equal(y[1], self.arrays['grad'][index])

    def test_the_same_seed_stages_the_same_order(self):
        self._stage(shuffle=True, seed=11)
        first = self._read()
        self._stage(shuffle=True, seed=11)
        second = self._read()
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1][0], second[1][0])
        np.testing.assert_array_equal(first[1][1], second[1][1])

    def test_another_seed_stages_another_order(self):
        self._stage(shuffle=True, seed=11)
        first = self._read()[0]
        self._stage(shuffle=True, seed=12)
        second = self._read()[0]
        self.assertFalse(np.array_equal(first, second))


if __name__ == '__main__':
    unittest.main()