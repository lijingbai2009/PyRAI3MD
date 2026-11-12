"""
======================================================================
Verification of `pyrai3md/utilities/structure_formatter.py`, the single
home of the structure-table layouts the run log and the trajectory files
share.

The two coordinate layouts are byte contracts: the run log indents every
row by two spaces, a trajectory frame starts with the element symbol.
The charges, matrix, cell and TurboMole blocks are pinned on the same
fixture the pre-refactor readers used.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities import units
from pyrai3md.utilities.structure_formatter import (print_cell, print_charges,
                                                   print_coord, print_matrix,
                                                   print_pbc,
                                                   print_turbocoord)

ATOMS = ['C', 'H']
COORDS = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
CHARGES = np.array([[-0.1, 0.0, 0.0, 0.0], [0.1, 1.09, 0.0, 0.0]])
CELL = np.eye(3) * 5.0
PBC = np.array([1, 1, 1])

LOG_ROWS = (
    '  C          0.0000000000000000      0.0000000000000000'
    '      0.0000000000000000\n'
    '  H          1.0900000000000001      0.0000000000000000'
    '      0.0000000000000000\n')
FRAME_ROWS = (
    'C          0.0000000000000000      0.0000000000000000'
    '      0.0000000000000000\n'
    'H          1.0900000000000001      0.0000000000000000'
    '      0.0000000000000000\n')


class TestTheCoordinateLayouts(unittest.TestCase):
    """Two indents, one row template."""

    def test_the_run_log_row_keeps_two_spaces(self):
        self.assertEqual(print_coord(ATOMS, COORDS), LOG_ROWS)

    def test_the_trajectory_row_has_no_indent(self):
        self.assertEqual(print_coord(ATOMS, COORDS, indent=''), FRAME_ROWS)

    def test_the_charge_table_keeps_its_two_spaces(self):
        expected = (
            '       -0.1000000000000000      0.0000000000000000'
            '      0.0000000000000000      0.0000000000000000\n'
            '        0.1000000000000000      1.0900000000000001'
            '      0.0000000000000000      0.0000000000000000\n')
        self.assertEqual(print_charges(CHARGES), expected)


class TestTheCellLayouts(unittest.TestCase):
    """Matrix, cell, PBC and TurboMole blocks keep their old bytes."""

    def test_the_matrix_is_the_old_layout(self):
        expected = (
            '      5.0000000000000000       0.0000000000000000'
            '       0.0000000000000000\n'
            '      0.0000000000000000       5.0000000000000000'
            '       0.0000000000000000\n'
            '      0.0000000000000000       0.0000000000000000'
            '       5.0000000000000000\n')
        self.assertEqual(print_matrix(CELL), expected)

    def test_the_cell_and_pbc_are_the_old_layout(self):
        self.assertEqual(
            print_cell(CELL),
            '\n        5.00000000      0.00000000      0.00000000\n'
            '        0.00000000      5.00000000      0.00000000\n'
            '        0.00000000      0.00000000      5.00000000\n')
        self.assertEqual(print_pbc(PBC), '1 1 1')
        self.assertEqual(print_cell([]), 'None')
        self.assertEqual(print_pbc([]), 'None')

    def test_the_turbomole_block_is_the_old_layout(self):
        # The layout is what this pins: 24-wide 16-decimal fields, the
        # element in a left-padded 5-column field, and the $coord /
        # $periodic / $lattice / $end order. The values come from the
        # framework's Bohr factor now, not from the hand-typed
        # 0.529177249 this module used to divide by: 1.09 Angstrom read
        # out as 2.0598013275510265 Bohr under that copy and as
        # 2.0598014758420895 Bohr under `utilities.units`, a 7.2e-08
        # relative shift in the eighth digit. The expectation is
        # computed, so a fourth copy of the constant cannot come back.
        def bohr(value):
            return value * units.ANGSTROM_TO_BOHR

        expected = (
            '$coord\n'
            '%24.16f%24.16f%24.16f %-5s\n'
            % (bohr(0.0), bohr(0.0), bohr(0.0), 'C')
            + '%24.16f%24.16f%24.16f %-5s\n'
            % (bohr(1.09), bohr(0.0), bohr(0.0), 'H')
            + '$periodic 3\n$lattice angstrom\n'
            # The lattice block stays in Angstrom: the header says so.
            + '%24.16f%24.16f%24.16f\n' % (5.0, 0.0, 0.0)
            + '%24.16f%24.16f%24.16f\n' % (0.0, 5.0, 0.0)
            + '%24.16f%24.16f%24.16f\n' % (0.0, 0.0, 5.0)
            + '$end\n')
        block = print_turbocoord(
            [[atom] + list(vector) for atom, vector in zip(ATOMS, COORDS)],
            CELL, PBC)
        self.assertEqual(block, expected)
        self.assertEqual(len(block.splitlines()[1]), 78,
                         'three 24-column fields plus the 5-column element '
                         'field: the layout, not the digits')


if __name__ == '__main__':
    unittest.main()