"""
======================================================================
This module formats the framework's structure data for internal data
communication: atom/vector rows, background-charge rows, matrices,
lattice cells, PBC flags, and the TurboMole $coord block.

It is the single home of these layouts. `print_results` composes them
into the run-log sections, `kernel/logger` writes the input echo, and
`workflow/records/output` writes trajectory frames; `io.py` parses input
only and owns no printer. `print_coord` takes an explicit indent because
the run log indents a table by two spaces while a trajectory frame does
not, and both layouts are byte contracts.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .units import ANGSTROM_TO_BOHR


def print_coord(atoms, vectors, indent='  '):
    """Format element symbols and per-atom vectors into a text table.

    Parameters
    ----------
    atoms : list of str
        Element symbols of shape (natom,).
    vectors : array-like
        Per-atom vectors of shape (natom, 3).
    indent : str, optional
        Prefix of every row. Two spaces for the run log (the default);
        an empty string for a trajectory frame, whose rows start with the
        element symbol.

    Returns
    -------
    str
        Formatted table string, one line per atom.
    """
    coord = ''
    for atom, vector in zip(atoms, vectors):
        x, y, z = (float(value) for value in vector)
        coord += '%s%-5s%24.16f%24.16f%24.16f\n' % (
            indent, atom, x, y, z)

    return coord


def print_charges(charges):
    """Format background point charges into a text table.

    Parameters
    ----------
    charges : array-like
        Point charges of shape (ncharge, 4) as [q, x, y, z].

    Returns
    -------
    str
        Formatted table string, one line per charge.
    """
    charge_info = ''
    for charge in charges:
        q, x, y, z = (float(value) for value in charge)
        charge_info += '  %24.16f%24.16f%24.16f%24.16f\n' % (q, x, y, z)

    return charge_info


def print_turbocoord(xyz, cell, pbc):
    """Convert coordinates to TurboMole format string.

    Parameters
    ----------
    xyz : np.ndarray
        Coordinates array with element symbols.
    cell : np.ndarray
        Lattice vectors.
    pbc : np.ndarray
        Periodicity flags.

    Returns
    -------
    str
        Formatted TurboMole coordinate string.
    """
    coord = '$coord\n'
    for line in xyz:
        e, x, y, z = line
        coord += '%24.16f%24.16f%24.16f %-5s\n' % (
            float(x) * ANGSTROM_TO_BOHR, float(y) * ANGSTROM_TO_BOHR,
            float(z) * ANGSTROM_TO_BOHR, e
        )

    if len(cell) > 0:
        coord += '$periodic %s\n$lattice angstrom\n' % (np.sum(pbc))
        for line in cell[np.where(pbc != 0)]:
            x, y, z = line
            coord += '%24.16f%24.16f%24.16f\n' % (x, y, z)

    coord += '$end\n'

    return coord


def print_matrix(mat):
    """Convert a numpy array to a formatted string.

    Parameters
    ----------
    mat : np.ndarray
        Input matrix.

    Returns
    -------
    str
        Formatted matrix string.
    """
    matrix = ''
    for row in mat:
        matrix += ' '.join(['%24.16f' % x for x in row]) + '\n'

    return matrix


def print_cell(cell):
    """Convert a lattice vector array to a formatted string.

    Parameters
    ----------
    cell : np.ndarray
        Lattice vectors, shape (3, 3).

    Returns
    -------
    str
        Formatted cell string.
    """
    if len(cell) == 0:
        return 'None'

    a, b, c = cell
    out = '\n  %16.8f%16.8f%16.8f\n  %16.8f%16.8f%16.8f\n  %16.8f%16.8f%16.8f\n' % (
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
    )

    return out


def print_pbc(pbc):
    """Convert periodicity flags to a formatted string.

    Parameters
    ----------
    pbc : np.ndarray
        Periodicity flags, shape (3,).

    Returns
    -------
    str
        Formatted PBC string.
    """
    if len(pbc) == 0:
        return 'None'

    out = '%s %s %s' % (pbc[0], pbc[1], pbc[2])

    return out


__all__ = [
    'print_coord',
    'print_charges',
    'print_turbocoord',
    'print_matrix',
    'print_cell',
    'print_pbc',
]