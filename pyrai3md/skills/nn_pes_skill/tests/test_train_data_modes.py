"""
======================================================================
Unit tests of the nn_pes training-data input modes.

The skill accepts the same dataset either as a file name (train_data)
or as an inline mapping (train_dataset). Both go through one reader, so
the tests assert that the two modes produce the same arrays, that the
schema refuses zero or two sources, and that the error messages name the
source that was actually used.

These tests need no TensorFlow.

Oct 23 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import json
import os
import unittest
from unittest import mock

import numpy as np

from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
from pyrai3md.skills.nn_pes_skill.schema.config import DEFAULT
from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
from pyrai3md.skills.nn_pes_skill.schema.validate import validate
from pyrai3md.skills.nn_pes_skill.service import trainer, trainer_data

HERE = os.path.dirname(os.path.abspath(__file__))
EGS = os.path.join(HERE, 'data', 'egs_small.json')
NAC = os.path.join(HERE, 'data', 'nac_small.json')


def _raw(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def _base(raw, **extra):
    frame = raw['xyz'][0]
    config = {
        'profile': 'mlp_eg_train',
        'method': 'mlp_eg',
        'modeldir': './models',
        'mult_nn': 2,
        'nstate': 2,
        'atoms': [str(row[0]) for row in frame],
        'coordinates': [row[-3:] for row in frame],
        'natom': len(frame),
        'eg': {'depth': 1},
    }
    config.update(extra)
    return build_config(config)


class TestTwoModes(unittest.TestCase):

    def test_file_and_inline_agree(self):
        raw = _raw(EGS)
        from_file = trainer_data.load(EGS)
        inline = trainer_data.from_payload(raw, source='<inline>')
        for name in ('xyz', 'energy', 'grad', 'nac', 'soc'):
            self.assertTrue(np.array_equal(from_file[name], inline[name]),
                            name)

    def test_match_labels_agrees(self):
        raw = _raw(EGS)
        from_file = trainer_data.match_labels(trainer_data.load(EGS))
        inline = trainer_data.match_labels(trainer_data.from_payload(raw))
        for name in from_file:
            self.assertTrue(np.array_equal(from_file[name], inline[name]),
                            name)

    def test_inline_payload_is_json_safe(self):
        payload = trainer_data.load(EGS)
        payload = {key: (value.tolist() if hasattr(value, 'tolist')
                         else value)
                   for key, value in payload.items()}
        again = trainer_data.from_payload(payload, source='<inline>')
        self.assertEqual(again['nframe'], 24)
        json.dumps(payload)


class TestNewChannels(unittest.TestCase):

    def test_transition_dipole_norm_is_derived(self):
        raw = _raw(EGS)
        raw['transition_dipole'] = np.ones((24, 2, 3)).tolist()
        dataset = trainer_data.from_payload(raw)
        self.assertEqual(dataset['npair'], 2)
        self.assertTrue(np.allclose(dataset['transition_dipole_norm'], 3 ** 0.5))

    def test_declared_pairs_must_match_arrays(self):
        raw = _raw(EGS)
        raw['osc_strength'] = np.zeros((24, 2)).tolist()
        raw['npair'] = 3
        with self.assertRaises(ValueError):
            trainer_data.from_payload(raw)

    def test_match_labels_passes_new_channels(self):
        raw = _raw(EGS)
        raw['transition_dipole'] = np.ones((24, 2, 3)).tolist()
        raw['osc_strength'] = np.zeros((24, 2)).tolist()
        arrays = trainer_data.match_labels(trainer_data.from_payload(raw))
        self.assertEqual(arrays['transition_dipole'].shape, (24, 2, 3))
        self.assertEqual(arrays['transition_dipole_norm'].shape, (24, 2))
        self.assertEqual(arrays['osc_strength'].shape, (24, 2))
        # The vector is not a derivative, so no length factor applies.
        self.assertTrue(np.allclose(arrays['transition_dipole'], 1.0))


class TestSchema(unittest.TestCase):

    def test_default_registers_the_inline_key(self):
        self.assertIn('train_dataset', DEFAULT)
        self.assertEqual(DEFAULT['train_dataset'], {})

    def test_inline_only_is_accepted(self):
        config = validate(normalize(_base(_raw(EGS),
                                          train_dataset=_raw(EGS))))
        self.assertTrue(config.calculation.train_dataset)
        self.assertEqual(config.calculation.train_data, '')

    def test_file_only_is_accepted(self):
        config = validate(normalize(_base(_raw(EGS), train_data=EGS)))
        self.assertEqual(config.calculation.train_data, EGS)
        self.assertEqual(config.calculation.train_dataset, {})

    def test_neither_source_is_refused(self):
        with self.assertRaises(ValueError):
            validate(normalize(_base(_raw(EGS))))

    def test_two_sources_are_refused(self):
        with self.assertRaises(ValueError):
            validate(normalize(_base(_raw(EGS), train_data=EGS,
                                     train_dataset=_raw(EGS))))

    def test_inline_must_be_a_mapping(self):
        with self.assertRaises(ValueError):
            validate(normalize(_base(_raw(EGS), train_dataset='data.json')))


class TestSourceLabels(unittest.TestCase):

    def _nac_config(self, **extra):
        raw = _raw(NAC)
        frame = raw['xyz'][0]
        config = {
            'profile': 'mlp_nac2_train',
            'method': 'mlp_nac2',
            'modeldir': './models',
            'mult_nn': 2,
            'nstate': 3,
            'atoms': [str(row[0]) for row in frame],
            'coordinates': [row[-3:] for row in frame],
            'natom': len(frame),
            'nnac': 3,
            'coupling': [[0, 1], [0, 2], [1, 2]],
            'nac': {'depth': 1},
        }
        config.update(extra)
        return validate(normalize(build_config(config)))

    def test_file_error_names_the_path(self):
        config = self._nac_config(train_data=NAC)
        dataset = trainer_data.load(NAC)
        dataset = dict(dataset, nnac=2)
        with self.assertRaises(ValueError) as caught:
            trainer_data.check_channels(config, dataset)
        self.assertIn(NAC, str(caught.exception))

    def test_inline_error_names_inline(self):
        raw = _raw(NAC)
        config = self._nac_config(train_dataset=raw)
        dataset = dict(trainer_data.from_payload(raw, source='<inline>'),
                       nnac=2)
        with self.assertRaises(ValueError) as caught:
            trainer_data.check_channels(config, dataset)
        self.assertIn('<inline>', str(caught.exception))


class TestPreparedSource(unittest.TestCase):

    def test_inline_needs_no_file_check(self):
        config = validate(normalize(_base(_raw(EGS),
                                          train_dataset=_raw(EGS))))
        sentinel = (('energy_grad', 'mlp_eg', './models/energy_grad'),)
        with mock.patch.object(trainer, 'targets_with_data',
                               return_value=sentinel) as called:
            result = trainer.prepared_or_raise(config, object(), './models')
        self.assertIs(result, sentinel)
        called.assert_called_once()

    def test_missing_file_still_fails(self):
        config = validate(normalize(_base(_raw(EGS),
                                          train_data='/no/such/file.json')))
        with self.assertRaises(FileNotFoundError):
            trainer.prepared_or_raise(config, object(), './models')


if __name__ == '__main__':
    unittest.main()