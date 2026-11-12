"""
======================================================================
This module locks the training-data geometry contract between the
framework reader and the nn_pes skill reader.

The two readers are deliberately separate (a skill may not import
framework internals), so the agreement cannot be structural; it is a
format contract: the same fixture must produce the same field set
through both, and a malformed file must be refused by both at the same
location. Before this lock the skill accepted a top-level `symbols`
list beside numeric rows while the framework dropped it, so one file
loaded with labels and the other without; and the mixed-row refusal
named no row at all on either side.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.skills.nn_pes_skill.service.trainer_data import (
    _read_xyz as skill_read_xyz)
from pyrai3md.training_data.dataset import _read_xyz as framework_read_xyz

LABELLED = {'xyz': [[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.74]]]}
NUMERIC = {'xyz': [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]],
           'symbols': [['H', 'H']]}
BOTH = dict(NUMERIC, symbols=[['H', 'O']])
MIXED = {'xyz': [[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.74]],
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]]}
SHORT_SYMBOLS = {'xyz': [[[0.0, 0.0, 0.0]]],
                 'symbols': [['H'], ['H']]}


class TestBothReadersSayTheSameThing(unittest.TestCase):
    """One fixture, two readers, one answer."""

    def test_a_labelled_file_reads_identically(self):
        for reader in (framework_read_xyz, skill_read_xyz):
            with self.subTest(reader=reader.__module__):
                xyz, symbols = reader(LABELLED, 'fixture')
                self.assertEqual(xyz.shape, (1, 2, 3))
                self.assertEqual(symbols.tolist(), [['H', 'H']])

    def test_a_symbols_list_reads_identically(self):
        for reader in (framework_read_xyz, skill_read_xyz):
            with self.subTest(reader=reader.__module__):
                xyz, symbols = reader(NUMERIC, 'fixture')
                self.assertEqual(xyz.shape, (1, 2, 3))
                self.assertIsNotNone(symbols)
                self.assertEqual(np.asarray(symbols).tolist(), [['H', 'H']])

    def test_a_numeric_file_without_symbols_has_none(self):
        raw = {'xyz': NUMERIC['xyz']}
        for reader in (framework_read_xyz, skill_read_xyz):
            with self.subTest(reader=reader.__module__):
                self.assertIsNone(reader(raw, 'fixture')[1])

    def test_a_mixed_file_is_refused_at_the_same_row(self):
        messages = []
        for reader in (framework_read_xyz, skill_read_xyz):
            with self.subTest(reader=reader.__module__):
                with self.assertRaises(ValueError) as raised:
                    reader(MIXED, 'fixture')
                messages.append(str(raised.exception))
                self.assertIn('frame 1 row 0', str(raised.exception))
        self.assertTrue(all('mixes labelled' in message
                            for message in messages))

    def test_a_short_symbols_list_is_refused_by_both(self):
        for reader in (framework_read_xyz, skill_read_xyz):
            with self.subTest(reader=reader.__module__):
                with self.assertRaises(ValueError) as raised:
                    reader(SHORT_SYMBOLS, 'fixture')
                self.assertIn('symbols must be (nframe, natom)',
                              str(raised.exception))

    def test_labelled_rows_win_over_a_symbols_list(self):
        # Both readers treat the inline labels as the authority; the
        # separate list is only for numeric rows.
        for reader in (framework_read_xyz, skill_read_xyz):
            with self.subTest(reader=reader.__module__):
                _xyz, symbols = reader(BOTH, 'fixture')
                self.assertEqual(np.asarray(symbols).tolist(), [['H', 'O']])


if __name__ == '__main__':
    unittest.main()