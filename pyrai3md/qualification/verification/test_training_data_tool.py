"""
======================================================================
Guard for the merged training-data tool.

The tool owns what the read jobtype deliberately does not: channel
filtering, statistics and persistence. These checks load the script by
path (tools/ is not a package) and exercise the packing and the four
data operations on inline data.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[3] / 'tools' / 'training_data.py'


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        'training_data_tool', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(gradient=True):
    frame = {
        'path': '/tmp/calc', 'project': 'mol',
        'symbols': ['H', 'H'],
        'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        'energy': [-1.137],
        'gradient': [[[0.0, 0.0, 0.001], [0.0, 0.0, -0.001]]],
        'nac': None,
        'soc': None,
        'unit': {'energy': 'hartree', 'gradient': 'hartree/bohr',
                 'nac': 'hartree/bohr', 'soc': 'cm-1'},
        'missing': [] if gradient else ['gradient'],
    }
    if not gradient:
        frame['gradient'] = None
    return frame


class TestTrainingDataTool(unittest.TestCase):

    def setUp(self):
        self.tool = _load_tool()
        self.work = tempfile.mkdtemp(prefix='training_tool_')
        self.addCleanup(shutil.rmtree, self.work)
        self._old_cwd = os.getcwd()
        os.chdir(self.work)

    def tearDown(self):
        os.chdir(self._old_cwd)

    def test_indices_are_one_based_and_inclusive(self):
        self.assertEqual(self.tool._parse_indices('1-2,4', 4), [0, 1, 3])
        with self.assertRaises(SystemExit):
            self.tool._parse_indices('0', 4)
        with self.assertRaises(SystemExit):
            self.tool._parse_indices('5', 4)

    def test_payload_packs_energy_and_gradient(self):
        payload = {'title': 'x', 'requested': ['energy', 'gradient'],
                   'frames': [_frame(), _frame()]}
        dataset = self.tool._dataset_from_payload(payload)
        self.assertEqual(dataset.natom, 2)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.energy.shape, (2, 1))
        self.assertEqual(dataset.grad.shape, (2, 1, 2, 3))
        self.assertEqual(dataset.unit['soc'], 'cm-1')

    def test_partial_channel_is_omitted(self):
        payload = {'title': 'x', 'requested': ['energy', 'gradient'],
                   'frames': [_frame(), _frame(gradient=False)]}
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = self.tool._dataset_from_payload(payload)
        self.assertIsNotNone(dataset.energy)
        self.assertIsNone(dataset.grad)

    def test_atom_order_mismatch_is_refused(self):
        second = _frame()
        second['symbols'] = ['H', 'C']
        payload = {'title': 'x', 'requested': ['energy'],
                   'frames': [_frame(), second]}
        with self.assertRaises(SystemExit):
            self.tool._dataset_from_payload(payload)

    def test_legacy_json_is_read(self):
        legacy = {
            'natom': 2, 'ncharge': 0, 'nstate': 1, 'nnac': 0, 'nsoc': 0,
            'xyz': [[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.74]],
                    [['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.75]]],
            'cell': [[], []], 'pbc': [[], []], 'charge': [0.0, 0.0],
            'energy': [[-1.137], [-1.136]],
            'grad': [[[[0.0, 0.0, 0.001], [0.0, 0.0, -0.001]]],
                     [[[0.0, 0.0, 0.002], [0.0, 0.0, -0.002]]]],
            'nac': [[], []], 'soc': [[], []],
        }
        with open('legacy.json', 'w') as handle:
            json.dump(legacy, handle)
        dataset = self.tool._load('legacy.json')
        self.assertEqual((dataset.natom, len(dataset)), (2, 2))
        self.assertIsNotNone(dataset.grad)

    def _base_dataset(self):
        from pyrai3md.training_data import TrainingData
        unit = {'energy': 'hartree', 'gradient': 'hartree/bohr',
                'nac': 'hartree/bohr', 'soc': 'cm-1'}
        return TrainingData(
            xyz=np.zeros((4, 2, 3)),
            symbols=np.array([['H', 'H']] * 4),
            energy=np.array([[-1.0], [-1.1], [-1.2], [-1.3]]),
            grad=np.zeros((4, 1, 2, 3)),
            unit=unit,
            source='test')

    def test_shuffle_is_seeded_and_grouped(self):
        self._base_dataset().save('data.json')
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool._shuffle('data.json', 1234, 0)
        from pyrai3md.training_data import load
        first = load('data-shuffled.json').energy.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool._shuffle('data.json', 1234, 0)
        self.assertTrue(np.array_equal(first,
                                       load('data-shuffled.json').energy))
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool._shuffle('data.json', 1234, 2)
        grouped = load('data-shuffled.json')
        self.assertEqual(len(grouped), 4)

    def test_split_merge_and_select(self):
        from pyrai3md.training_data import load
        self._base_dataset().save('data.json')
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool._split('data.json', 1, False, 1)
        self.assertEqual(len(load('data-set-1.json')), 1)
        self.assertEqual(len(load('data-set-2.json')), 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool._merge('data.json', ['data-set-1.json'], False, 1)
        self.assertEqual(len(load('data-merged.json')), 5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool._select('data.json', '1-2')
        self.assertEqual(len(load('data-selected.json')), 2)

    def test_describe_prints_axis_names(self):
        self._base_dataset().save('data.json')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.tool._describe_file('data.json')
        text = buffer.getvalue()
        self.assertIn('nmol:', text)
        self.assertIn('(nmol, nstate, natom, 3)', text)
        self.assertIn('grad:', text)


if __name__ == '__main__':
    unittest.main()