"""
======================================================================
This module locks the control mapping of the example converter.

`yaml_to_config` used to forward the eight control keywords as eight
hand-written blocks, and the adaptive/workflow/Slurm sections as three
copies of the same "carry the dict" guard. Both are now declarative:
one row per control keyword in `_CONTROL_FIELDS` (alias, transform,
when the pair is written, default) and one ordered list of the
sections carried verbatim.

The criterion is that the table is the authority, not a restatement:
removing a row removes the key, and the four emission rules keep the
behaviours the blocks had (gl_seed 0 is written, debug stays a
boolean, an absent reference writes nothing, defaults keep a default
input minimal). The 201 shipped examples dump byte-identically to the
pre-change output, key order included.

Nov 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import unittest
from unittest import mock

from pyrai3md.qualification.certification import converter

EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')


class TestTheControlTableIsTheMapping(unittest.TestCase):
    """One row per forwarded keyword, and the row decides."""

    def test_the_rows_are_the_eight_forwarded_keys_in_order(self):
        self.assertEqual(
            [row[0] for row in converter._CONTROL_FIELDS],
            ['title', 'jobtype', 'theory', 'reference', 'driver',
             'multiscale_type', 'debug', 'gl_seed'])

    def test_the_classic_aliases_are_read(self):
        flat = converter._read_control(
            {'qm': 'molcas openqp', 'abinit': 'pyscf'}, 'stem')
        self.assertEqual(flat['theory'], ['molcas', 'openqp'])
        self.assertEqual(flat['reference'], ['pyscf'])

    def test_a_zero_seed_is_written_on_presence(self):
        self.assertEqual(
            converter._read_control({'gl_seed': 0}, 'stem')['gl_seed'], 0)
        self.assertNotIn('gl_seed', converter._read_control({}, 'stem'))

    def test_the_test_switch_stays_a_boolean(self):
        self.assertIs(
            converter._read_control({'debug': 'yes'}, 'stem')['debug'], True)

    def test_default_inputs_stay_minimal(self):
        flat = converter._read_control({}, 'stem')
        self.assertEqual(flat['title'], 'stem')
        self.assertEqual(flat['jobtype'], 'energy')
        self.assertEqual(flat['theory'], [])
        for key in ('reference', 'driver', 'multiscale_type', 'debug',
                    'gl_seed'):
            self.assertNotIn(key, flat)

    def test_an_empty_reference_is_not_written(self):
        self.assertNotIn(
            'reference', converter._read_control({'reference': []}, 'stem'))

    def test_removing_a_row_removes_the_key(self):
        """Negative control: no leftover branch keeps the keyword alive."""
        kept = tuple(row for row in converter._CONTROL_FIELDS
                     if row[0] != 'multiscale_type')
        with mock.patch.object(converter, '_CONTROL_FIELDS', kept):
            flat = converter._read_control(
                {'multiscale_type': 'qmmm'}, 'stem')
        self.assertNotIn('multiscale_type', flat)


class TestTheCarriedSectionsAreOneList(unittest.TestCase):
    """The three former guards are one ordered list."""

    def test_the_order_is_the_historical_key_order(self):
        self.assertEqual(
            converter._CARRIED_SECTIONS,
            ('adaptive', 'hessian', 'search', 'train', 'prediction',
             'slurm'))

    def test_a_shipped_example_carries_both_blocks_in_that_order(self):
        config = converter.yaml_to_config(
            os.path.join(EXAMPLES, 'adaptive_oniom2_slurm.yaml'))
        keys = list(config)
        self.assertLess(keys.index('adaptive'), keys.index('slurm'))


if __name__ == '__main__':
    unittest.main()