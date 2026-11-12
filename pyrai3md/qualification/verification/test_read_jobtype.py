"""
======================================================================
Guard for the read-only jobtype and its input section.

The read run is the one framework flow that must work without any
engine installed and must never write a data artifact. These checks pin
the vocabulary, both input carriers, the refusals and the shape of the
skill config planning hands to the reader.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

import yaml

from pyrai3md.data.build import build_state
from pyrai3md.interface.parser import parse_input
from pyrai3md.schema import adapter, config
from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.normalize import normalize
from pyrai3md.schema.validate import (ADVANCED_JOBTYPES, parse_jobtype,
                                      validate)


def _base_payload(jobtype='read', theory=('xtb',), read=None):
    return {
        'control': {'title': 'read_test', 'jobtype': jobtype,
                    'theory': list(theory)},
        'read': read if read is not None else {'path': ['/tmp/run_1']},
        'molecule': {'charges': 0, 'multiplicity': 1,
                     'ci': [1], 'spin': [0]},
        'atoms': ['H', 'H'],
        'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        'xtb': {'method': 'gfn2'},
    }


class TestReadJobtype(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix='read_jobtype_')
        self.addCleanup(shutil.rmtree, self.work)

    def _pipeline(self, payload):
        path = os.path.join(self.work, 'input.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(payload, handle)
        raw = parse_input(path)
        return validate(normalize(build_config(raw))), raw

    def test_read_is_an_advanced_jobtype(self):
        self.assertIn('read', ADVANCED_JOBTYPES)
        self.assertEqual(parse_jobtype('read'), (['read'], False, True))

    def test_read_cannot_mix_with_a_property_token(self):
        with self.assertRaises(ValueError):
            parse_jobtype('read energy')

    def test_yaml_read_section_builds_the_planned_config(self):
        config_, raw = self._pipeline(_base_payload())
        self.assertEqual(config_.jobtype, 'read')
        self.assertEqual(config_.read.path, ['/tmp/run_1'])
        self.assertTrue(config_.read.energy)
        self.assertTrue(config_.read.gradient)
        self.assertFalse(config_.read.nac)
        state = build_state(config_, raw_config=raw)
        skill = state.theory[0]['config']['skill_config']
        self.assertEqual(skill['jobtype'], 'energy gradient')
        self.assertEqual(state.theory[0]['capabilities'], [])

    def test_classic_read_section_has_no_end_marker(self):
        xyz = os.path.join(self.work, 'classic.xyz')
        with open(xyz, 'w') as handle:
            handle.write('2\nH2\nH 0 0 0\nH 0 0 0.74\n')
        text = '\n'.join((
            '&control',
            'title classic_read',
            'jobtype read',
            'theory xtb',
            '',
            '&read',
            'path /tmp/run_1',
            'energy 1',
            'gradient 1',
            'nac 0',
            'soc 0',
            '',
            '&molecule',
            'charges 0',
            'multiplicity 1',
            'ci 1',
            'spin 0',
            '',
            '&xtb',
            'method gfn2',
            '',
        ))
        path = os.path.join(self.work, 'classic.input')
        with open(path, 'w') as handle:
            handle.write(text)
        raw = parse_input(path)
        config_ = validate(normalize(build_config(raw)))
        self.assertEqual(config_.jobtype, 'read')
        self.assertEqual(config_.read.path, '/tmp/run_1')
        self.assertTrue(config_.read.energy)
        self.assertTrue(config_.read.gradient)

    def test_read_section_without_read_jobtype_is_refused(self):
        payload = _base_payload(jobtype='energy')
        with self.assertRaises(ValueError):
            self._pipeline(payload)

    def test_more_than_one_theory_is_refused(self):
        payload = _base_payload(theory=('xtb', 'xtb'))
        with self.assertRaises(ValueError):
            self._pipeline(payload)

    def test_a_reference_is_refused(self):
        payload = _base_payload()
        payload['control']['reference'] = ['orca']
        with self.assertRaises(ValueError):
            self._pipeline(payload)

    def test_no_label_is_refused(self):
        payload = _base_payload(read={
            'path': ['/tmp/run_1'], 'energy': False, 'gradient': False,
            'nac': False, 'soc': False})
        with self.assertRaises(ValueError):
            self._pipeline(payload)

    def test_empty_path_is_refused(self):
        payload = _base_payload(read={'path': ''})
        with self.assertRaises(ValueError):
            self._pipeline(payload)

    def test_manifest_section_lists_only_accepted_keywords(self):
        import pyrai3md
        manifest_path = os.path.join(
            os.path.dirname(pyrai3md.__file__), 'manifest.yaml')
        with open(manifest_path) as handle:
            manifest = yaml.safe_load(handle)
        spec = manifest['input_schema']['sections']['read']
        model = getattr(config, spec['model'].rsplit('.', 1)[-1])
        table = getattr(adapter, spec['types'].rsplit('.', 1)[-1])
        for keyword in spec['keywords']:
            self.assertTrue(
                keyword in model.model_fields or keyword in table,
                'read.%s is listed but not accepted' % keyword)


if __name__ == '__main__':
    unittest.main()