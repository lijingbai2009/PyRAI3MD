"""
======================================================================
This is a standalone ORCA calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module parses the ORCA .engrad file produced by EnGrad
calculations (ground-state energy and gradient).

"""

import os
import numpy as np


def parse_engrad(engrad_path: str, natom: int):
    """Parse the ORCA .engrad file for energy and gradient.

    The engrad file stores the total energy in Eh and the nuclear
    gradient in Eh/bohr (flat 3*natom vector).

    Parameters
    ----------
    engrad_path : str
        Path to the .engrad file.
    natom : int
        Number of atoms.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Energy array (1,) in Eh and gradient array (natom, 3)
        in Eh/bohr. Zero-filled when the file is missing.
    """
    energy = np.zeros(1)
    gradient = np.zeros((natom, 3))

    if not os.path.exists(engrad_path):
        return energy, gradient

    with open(engrad_path, 'r') as f:
        log = f.read().splitlines()

    for n, line in enumerate(log):
        if 'The current total energy in Eh' in line:
            try:
                energy = np.array([float(log[n + 2])])
            except (ValueError, IndexError):
                pass

        if 'The current gradient in Eh/bohr' in line:
            try:
                g = [float(x) for x in log[n + 2: n + 2 + natom * 3]]
                gradient = np.array(g).reshape((natom, 3))
            except (ValueError, IndexError):
                pass

    return energy, gradient
