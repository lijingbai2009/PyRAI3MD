"""
======================================================================
Test suite for the run-log stress tensor section
(pyrai3md/utilities/print_results.py: _format_stress).

Tests cover:
- Present (1, 3, 3) and (nstate, 3, 3) stress records: section title,
  3x3 component formatting and the derived isotropic pressure lines
  (Hartree/Bohr^3 and bar)
- Active-state projection (current_state selects the matching block;
  None / out-of-range falls back to index 0)
- Missing stress (None / empty / all-zero default): 'Not computed'
  marker at verbose >= 3 and an empty string at lower verbosity
- Shape guards for malformed records

Sep 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities.print_results import _format_stress
from pyrai3md.utilities.units import au_pressure_to_bar

TITLE = '&stress tensor in Hartree/Bohr^3'
STRESS = [[-0.001, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, -0.003]]
STRESS_S1 = [[-0.010, 0.0, 0.0], [0.0, -0.010, 0.0], [0.0, 0.0, -0.010]]


def _pressure_lines(tensor):
    arr = np.asarray(tensor, dtype=float).reshape(3, 3)
    pressure_au = -float(np.trace(arr)) / 3.0
    au_line = '%20.12f Hartree/Bohr^3' % pressure_au
    bar_line = '%20.4f bar' % au_pressure_to_bar(pressure_au)
    return au_line, bar_line


class TestFormatStressPresent(unittest.TestCase):
    """A filled stress record renders the tensor and pressure."""

    def test_single_state(self):
        text = _format_stress([STRESS], 0)
        self.assertIn(TITLE, text)
        for row in STRESS:
            self.assertIn(''.join('%20.12f' % v for v in row), text)
        au_line, bar_line = _pressure_lines(STRESS)
        self.assertIn(au_line, text)
        self.assertIn(bar_line, text)

    def test_2d_record_reshaped(self):
        # A bare (3, 3) record is treated as a single state.
        text = _format_stress(STRESS, 0)
        self.assertIn(TITLE, text)
        au_line, _ = _pressure_lines(STRESS)
        self.assertIn(au_line, text)

    def test_active_state_projection(self):
        text = _format_stress([STRESS, STRESS_S1], 1)
        au_line, _ = _pressure_lines(STRESS_S1)
        self.assertIn(au_line, text)
        # The state-0 block must not leak into the state-1 section.
        other_line, _ = _pressure_lines(STRESS)
        self.assertNotIn(other_line, text)

    def test_none_state_falls_back_to_zero(self):
        text = _format_stress([STRESS, STRESS_S1], None)
        au_line, _ = _pressure_lines(STRESS)
        self.assertIn(au_line, text)

    def test_out_of_range_state_falls_back_to_zero(self):
        text = _format_stress([STRESS, STRESS_S1], 9)
        au_line, _ = _pressure_lines(STRESS)
        self.assertIn(au_line, text)


class TestFormatStressMissing(unittest.TestCase):
    """Absent or unfilled stress obeys the verbose gate."""

    def _check_missing(self, value):
        shown = _format_stress(value, 0, show_missing=True)
        hidden = _format_stress(value, 0, show_missing=False)
        self.assertIn(TITLE, shown)
        self.assertIn('Not computed', shown)
        self.assertEqual(hidden, '')

    def test_none(self):
        self._check_missing(None)

    def test_empty_list(self):
        self._check_missing([])

    def test_all_zero_default(self):
        zero = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        self._check_missing(zero)

    def test_malformed_shape(self):
        self._check_missing([[1.0, 2.0, 3.0]])


if __name__ == '__main__':
    unittest.main()
