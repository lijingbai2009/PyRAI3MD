"""
======================================================================
Test suite for the framework training-data layer
(pyrai3md/training_data/) and the train workflow's pre-initialization
permutation expansion.

Tests cover:
- Legacy and canonical JSON reading, and the save/load round trip
- append: concatenation plus the refusals (atom order, channel counts,
  field presence)
- Permutation expansion: one-based map reading, atom-axis moves,
  invariant channels, element-preservation checks
- The inline payload path and the derived transition-dipole norm
- &train.permutation_map reaching the config and expanding a task's
  skill config before any backend exists, idempotently

Oct 23 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pyrai3md.training_data import (TrainingData, expand_permutations,
                                    from_payload, load,
                                    read_permutation_map, save)


def _raw(nframe=2):
    """A small labelled dataset with every channel the layer supports."""
    base = [8, 1, 1]
    xyz = []
    energy = []
    grad = []
    nac = []
    soc = []
    transition_dipole = []
    osc = []
    for frame in range(nframe):
        shift = 0.1 * frame
        xyz.append([[base[0], shift, 0.0, 0.0],
                    [base[1], 1.0 + shift, 0.0, 0.0],
                    [base[2], 0.0, 1.0 + shift, 0.0]])
        energy.append([1.0 + shift, 2.0 + shift])
        grad.append(np.full((2, 3, 3), 0.1 * (frame + 1)).tolist())
        nac.append(np.full((1, 3, 3), 0.2 * (frame + 1)).tolist())
        soc.append([10.0 + shift, 20.0 + shift])
        transition_dipole.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        osc.append([0.1 + shift, 0.2 + shift])
    return {
        'natom': 3, 'nstate': 2, 'nnac': 1, 'nsoc': 2, 'npair': 2,
        'xyz': xyz, 'energy': energy, 'grad': grad, 'nac': nac, 'soc': soc,
        'transition_dipole': transition_dipole, 'osc_strength': osc,
        'unit': {'energy': 'hartree'},
    }


class TestLoadSave(unittest.TestCase):

    def test_legacy_four_column_rows_are_read(self):
        data = TrainingData.from_mapping(_raw(), source='legacy')
        self.assertEqual(data.nframe, 2)
        self.assertEqual(data.natom, 3)
        self.assertEqual(data.nstate, 2)
        self.assertEqual(data.nnac, 1)
        self.assertEqual(data.nsoc, 2)
        self.assertEqual(data.npair, 2)
        self.assertEqual(data.symbols.tolist()[0], ['8', '1', '1'])
        self.assertEqual(data.transition_dipole.shape, (2, 2, 3))

    def test_transition_dipole_norm_is_derived(self):
        data = TrainingData.from_mapping(_raw(), source='derived')
        self.assertAlmostEqual(float(data.transition_dipole_norm[0, 0]), 1.0)
        self.assertAlmostEqual(float(data.transition_dipole_norm[0, 1]), 1.0)

    def test_canonical_round_trip(self):
        data = TrainingData.from_mapping(_raw(), source='roundtrip')
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'data.json')
            save(data, path)
            loaded = load(path)
        for name in ('xyz', 'symbols', 'energy', 'grad', 'nac', 'soc',
                     'transition_dipole', 'transition_dipole_norm',
                     'osc_strength'):
            self.assertTrue(np.array_equal(getattr(data, name),
                                           getattr(loaded, name)), name)
        self.assertEqual(loaded.unit, {'energy': 'hartree'})

    def test_written_geometries_are_numeric(self):
        data = TrainingData.from_mapping(_raw(), source='numeric')
        payload = data.to_payload()
        self.assertTrue(all(isinstance(row[-1], float)
                            for frame in payload['xyz'] for row in frame))
        self.assertEqual(payload['symbols'][0], ['8', '1', '1'])

    def test_from_payload_matches_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'data.json')
            with open(path, 'w') as handle:
                json.dump(_raw(), handle)
            loaded = load(path)
            inline = from_payload(_raw(), source='<inline>')
        for name in ('xyz', 'energy', 'grad', 'nac', 'soc',
                     'transition_dipole', 'osc_strength'):
            self.assertTrue(np.array_equal(getattr(loaded, name),
                                           getattr(inline, name)), name)


class TestAppend(unittest.TestCase):

    def test_append_concatenates_every_channel(self):
        data = TrainingData.from_mapping(_raw(1), source='a')
        data.append(TrainingData.from_mapping(_raw(1), source='b'))
        self.assertEqual(data.nframe, 2)
        self.assertEqual(data.xyz.shape[0], 2)
        self.assertEqual(data.energy.shape[0], 2)
        self.assertEqual(data.group_id, None)
        self.assertEqual(data.transition_dipole.shape[0], 2)

    def test_append_refuses_a_different_atom_order(self):
        data = TrainingData.from_mapping(_raw(1), source='a')
        raw = _raw(1)
        raw['xyz'][0][0][0], raw['xyz'][0][1][0] = (
            raw['xyz'][0][1][0], raw['xyz'][0][0][0])
        with self.assertRaises(ValueError):
            data.append(TrainingData.from_mapping(raw, source='b'))

    def test_append_refuses_a_different_channel_count(self):
        data = TrainingData.from_mapping(_raw(1), source='a')
        raw = _raw(1)
        raw['nnac'] = 0
        raw['nac'] = []
        with self.assertRaises(ValueError):
            data.append(TrainingData.from_mapping(raw, source='b'))

    def test_append_refuses_a_missing_field(self):
        data = TrainingData.from_mapping(_raw(1), source='a')
        raw = _raw(1)
        raw['soc'] = []
        raw['nsoc'] = 0
        with self.assertRaises(ValueError):
            data.append(TrainingData.from_mapping(raw, source='b'))

    def test_append_refuses_a_different_unit_block(self):
        data = TrainingData.from_mapping(_raw(1), source='a')
        raw = _raw(1)
        raw['unit'] = {'energy': 'ev'}
        with self.assertRaises(ValueError):
            data.append(TrainingData.from_mapping(raw, source='b'))


class TestPermutation(unittest.TestCase):

    def _map_file(self, folder, text):
        path = os.path.join(folder, 'map.txt')
        with open(path, 'w') as handle:
            handle.write(text)
        return path

    def test_one_based_map_is_read_and_expanded(self):
        data = TrainingData.from_mapping(_raw(2), source='source')
        with tempfile.TemporaryDirectory() as folder:
            path = self._map_file(folder, '1 3 2\n')
            maps = read_permutation_map(path, 3, data.symbols[0])
        self.assertEqual(maps.tolist(), [[0, 2, 1]])

        expanded = expand_permutations(data, maps)
        self.assertEqual(expanded.nframe, 4)
        self.assertEqual(expanded.group_id.tolist(), [0, 0, 1, 1])
        self.assertEqual(expanded.permutation_id.tolist(), [0, 1, 0, 1])

        # The atom-axis channels move; the electronic channels do not.
        self.assertTrue(np.array_equal(expanded.xyz[1],
                                       data.xyz[0][[0, 2, 1]]))
        self.assertTrue(np.array_equal(expanded.grad[1],
                                       data.grad[0][:, [0, 2, 1], :]))
        self.assertTrue(np.array_equal(expanded.nac[1],
                                       data.nac[0][:, [0, 2, 1], :]))
        for name in ('energy', 'soc', 'transition_dipole',
                     'transition_dipole_norm', 'osc_strength'):
            self.assertTrue(np.array_equal(getattr(expanded, name)[1],
                                           getattr(data, name)[0]), name)

    def test_two_maps_produce_three_copies_per_frame(self):
        data = TrainingData.from_mapping(_raw(2), source='source')
        maps = np.asarray([[0, 2, 1], [0, 1, 2]])
        expanded = expand_permutations(data, maps)
        self.assertEqual(expanded.nframe, 6)
        self.assertEqual(expanded.permutation_id.tolist(),
                         [0, 1, 2, 0, 1, 2])

    def test_no_map_is_a_no_op(self):
        data = TrainingData.from_mapping(_raw(1), source='source')
        self.assertIs(expand_permutations(data, None), data)
        self.assertIs(expand_permutations(data, np.empty((0, 3), int)), data)

    def test_map_length_must_match_the_atom_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._map_file(folder, '1 3 2\n')
            with self.assertRaises(ValueError):
                read_permutation_map(path, 4)

    def test_a_row_must_be_a_permutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._map_file(folder, '1 1 2\n')
            with self.assertRaises(ValueError):
                read_permutation_map(path, 3)

    def test_element_preservation_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._map_file(folder, '2 1 3\n')
            with self.assertRaises(ValueError):
                read_permutation_map(path, 3, ['O', 'H', 'H'])

    def test_an_expanded_dataset_round_trips(self):
        data = TrainingData.from_mapping(_raw(1), source='source')
        expanded = expand_permutations(data, np.asarray([[0, 2, 1]]))
        again = TrainingData.from_mapping(expanded.to_payload())
        for name in ('xyz', 'grad', 'nac', 'group_id', 'permutation_id'):
            self.assertTrue(np.array_equal(getattr(expanded, name),
                                           getattr(again, name)), name)


class TestTrainConfig(unittest.TestCase):

    def test_permutation_map_survives_config_build(self):
        from pyrai3md.schema.adapter import build_config
        config = build_config({'control': {'jobtype': 'train'},
                               'train': {'permutation_map': './map.txt'}})
        self.assertEqual(config.train.permutation_map, './map.txt')

    def test_default_is_disabled(self):
        from pyrai3md.schema.adapter import build_config
        config = build_config({'control': {'jobtype': 'train'}})
        self.assertEqual(config.train.permutation_map, '')


class TestPrepare(unittest.TestCase):

    def _state(self, map_path, task):
        return SimpleNamespace(
            pmd=SimpleNamespace(
                train=SimpleNamespace(permutation_map=map_path)),
            atoms=['O', 'H', 'H'],
            iter_tasks=lambda: [('theory', task)])

    def test_prepare_injects_inline_and_is_idempotent(self):
        from pyrai3md.workflow.train import prepare_training_data
        with tempfile.TemporaryDirectory() as folder:
            data_path = os.path.join(folder, 'data.json')
            with open(data_path, 'w') as handle:
                json.dump(_raw(2), handle)
            map_path = os.path.join(folder, 'map.txt')
            with open(map_path, 'w') as handle:
                handle.write('1 3 2\n')
            task = {'config': {'skill_config': {
                'profile': 'mlp_eg_train',
                'train_data': data_path,
                'train_dataset': {},
            }}}
            state = self._state(map_path, task)
            prepare_training_data(state)
            skill = task['config']['skill_config']
            self.assertEqual(skill['train_data'], '')
            self.assertEqual(len(skill['train_dataset']['xyz']), 4)
            self.assertEqual(skill['train_dataset']['source'], data_path)
            self.assertEqual(skill['train_dataset']['group_id'], [0, 0, 1, 1])

            prepare_training_data(state)
            self.assertEqual(len(skill['train_dataset']['xyz']), 4)

    def test_prepare_without_a_map_touches_nothing(self):
        from pyrai3md.workflow.train import prepare_training_data
        task = {'config': {'skill_config': {
            'profile': 'mlp_eg_train', 'train_data': '/not/read.json',
            'train_dataset': {},
        }}}
        state = self._state('', task)
        prepare_training_data(state)
        self.assertEqual(task['config']['skill_config']['train_data'],
                         '/not/read.json')
        self.assertEqual(task['config']['skill_config']['train_dataset'], {})

    def test_prepare_skips_non_training_tasks(self):
        from pyrai3md.workflow.train import prepare_training_data
        task = {'config': {'skill_config': {
            'profile': 'mlp_eg_gradient', 'train_data': '/not/read.json',
        }}}
        state = self._state('No', task)
        prepare_training_data(state)
        self.assertEqual(task['config']['skill_config']['train_data'],
                         '/not/read.json')

    def test_single_shot_prepare_only_for_the_train_jobtype(self):
        from pyrai3md.workflow.single import SingleShot
        with tempfile.TemporaryDirectory() as folder:
            data_path = os.path.join(folder, 'data.json')
            with open(data_path, 'w') as handle:
                json.dump(_raw(1), handle)
            map_path = os.path.join(folder, 'map.txt')
            with open(map_path, 'w') as handle:
                handle.write('1 3 2\n')
            task = {'config': {'skill_config': {
                'profile': 'mlp_eg_train',
                'train_data': data_path,
                'train_dataset': {},
            }}}
            state = self._state(map_path, task)
            state.jobtype = 'energy'
            SingleShot.prepare(state)
            self.assertEqual(task['config']['skill_config']['train_data'],
                             data_path)
            state.jobtype = 'train'
            SingleShot.prepare(state)
            self.assertEqual(
                len(task['config']['skill_config']['train_dataset']['xyz']), 2)


if __name__ == '__main__':
    unittest.main()