"""
======================================================================
Guard for the read-only execution flow.

A read run must return frames, never construct a backend and never write
a data or log file. These checks build a finished xTB directory and run
the real pipeline through `pmd.run`.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import glob
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml

from pyrai3md.pmd import run
from pyrai3md.workflow.read import ReadExtractionError


def _write_xtb_calc(folder, project='mol'):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, '%s.xyz' % project), 'w') as handle:
        handle.write('2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n')
    lines = ['The current total energy in Eh', '----', '   -1.1372838',
             'The current gradient in Eh/bohr', '----']
    lines += ['   %20.12f' % value
              for value in (0.0, 0.0, 0.001, 0.0, 0.0, -0.001)]
    with open(os.path.join(folder, '%s.engrad' % project), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def _read_input(path, calc_path, **read):
    payload = {
        'control': {'title': 'read_flow', 'jobtype': 'read',
                    'theory': ['xtb']},
        'read': dict({'path': calc_path}, **read),
        'molecule': {'charges': 0, 'multiplicity': 1,
                     'ci': [1], 'spin': [0]},
        'atoms': ['H', 'H'],
        'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        'xtb': {'method': 'gfn2'},
    }
    with open(path, 'w') as handle:
        yaml.safe_dump(payload, handle)


class TestWorkflowRead(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix='read_flow_')
        self.addCleanup(shutil.rmtree, self.work)
        self.calc = os.path.join(self.work, 'calc')
        _write_xtb_calc(self.calc)
        self.input = os.path.join(self.work, 'read.yaml')

    def test_payload_and_no_artifact(self):
        before = sorted(os.listdir(self.calc))
        _read_input(self.input, self.calc)
        payload = run(self.input)
        frame = payload['frames'][0]
        self.assertEqual(payload['skill'], 'xtb_skill')
        self.assertEqual(payload['requested'], ['energy', 'gradient'])
        self.assertEqual(frame['project'], 'mol')
        self.assertEqual(frame['symbols'], ['H', 'H'])
        self.assertEqual(frame['missing'], [])
        self.assertAlmostEqual(frame['energy'][0], -1.1372838)
        self.assertEqual(len(frame['gradient']), 1)
        self.assertEqual(len(frame['gradient'][0]), 2)
        self.assertEqual(sorted(os.listdir(self.calc)), before)
        self.assertEqual(sorted(os.listdir(self.work)),
                         ['calc', 'read.yaml'])
        self.assertFalse(glob.glob(os.path.join(self.work, '*.log')))

    def test_backend_is_never_constructed(self):
        from pyrai3md.skills.xtb_skill.backend import XtbBackend
        _read_input(self.input, self.calc)
        with patch.object(XtbBackend, '__init__',
                          side_effect=AssertionError('backend constructed')):
            payload = run(self.input)
        self.assertEqual(len(payload['frames']), 1)

    def test_path_list_file(self):
        list_path = os.path.join(self.work, 'dirs.txt')
        with open(list_path, 'w') as handle:
            handle.write(self.calc + '\n')
        _read_input(self.input, list_path)
        payload = run(self.input)
        self.assertEqual(len(payload['frames']), 1)

    def test_project_override(self):
        other = os.path.join(self.work, 'other')
        _write_xtb_calc(other, project='guess')
        _read_input(self.input, other, project='guess')
        payload = run(self.input)
        self.assertEqual(payload['frames'][0]['project'], 'guess')

    def test_profile_reaches_the_skill_entry(self):
        import pyrai3md.skills.xtb_skill.service.read as skill_read
        seen = []
        original = skill_read.read_calculation

        def spy(config, calcdir, project=None, profile=None):
            seen.append(profile)
            return original(config, calcdir, project=project,
                            profile=profile)

        _read_input(self.input, self.calc, profile='gfn2_energy')
        with patch('pyrai3md.skills.xtb_skill.backend.read_calculation',
                   spy):
            run(self.input)
        self.assertEqual(seen, ['gfn2_energy'])

    def test_missing_path_is_reported(self):
        _read_input(self.input, os.path.join(self.work, 'not_there'))
        with self.assertRaises(ReadExtractionError):
            run(self.input)


if __name__ == '__main__':
    unittest.main()