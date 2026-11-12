"""
======================================================================
This module reads the trajectory products of the Landau-Zener
benchmark cases.

The lz cases replay a hopping kernel over a recorded trajectory, and
the gsh and lzsh cases each grew their own reader for the same
`<title>.md.energies` table. The column meanings live here once: the
file is `time Epot Ekin Etot Epot1 Epot2 ...`, the lz cases carry two
adiabatic states, and every consumer wants the kinetic energy beside
those two state energies. A case that starts writing a different
column layout changes this file, not each reader.

The GSH replay additionally needs the coordinates of the same steps;
that read stays with its one consumer (`gsh.py`).

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from ...utilities.units import BOHR_TO_ANGSTROM


def read_energies(traj_dir, title):
    """Read the kinetic and two state energies of one trajectory.

    Parameters
    ----------
    traj_dir : str
        Trajectory working directory holding the md output files.
    title : str
        Case title prefix of the md output file names.

    Returns
    -------
    tuple
        (ekin, state_energies): per-step kinetic energy (Hartree) and
        the two adiabatic state energies (Hartree).
    """
    path = os.path.join(traj_dir, '%s.md.energies' % title)
    table = np.loadtxt(path, ndmin=2, skiprows=1)
    # Columns: time Epot Ekin Etot Epot1 Epot2 ...
    return table[:, 2], table[:, 4:6]


def read_coordinates(traj_dir, title):
    """Read the per-step coordinates of one trajectory, in Bohr.

    Parameters
    ----------
    traj_dir : str
        Trajectory working directory holding the md output files.
    title : str
        Case title prefix of the md output file names.

    Returns
    -------
    numpy.ndarray
        Atom coordinates of every recorded step, in Bohr, in file order.
    """
    path = os.path.join(traj_dir, '%s.md.xyz' % title)
    with open(path, 'r', encoding='utf-8') as handle:
        lines = [line.strip() for line in handle if line.strip()]
    positions = []
    index = 0
    while index < len(lines):
        natom = int(lines[index])
        for atom in range(natom):
            parts = lines[index + 2 + atom].split()
            positions.append(float(parts[1]) / BOHR_TO_ANGSTROM)
        index += 2 + natom
    return np.asarray(positions)