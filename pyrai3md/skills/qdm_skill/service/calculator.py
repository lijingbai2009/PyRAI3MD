"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Single-point property calculator for the analytic Tully models.

Diagonalizes the diabatic Hamiltonian to obtain adiabatic energies
and eigenvectors, then evaluates adiabatic gradients and
nonadiabatic coupling (NAC) vectors through the Hellmann-Feynman
relations:

    dE_i/dx = <i| dH/dx |i>
    d_ij    = <i| d/dx |j> = <i| dH/dx |j> / (E_j - E_i)

The reported NAC vector is <i| d/dx |j> * (E_j - E_i), the
derivative coupling multiplied by the signed gap, which equals
<i| dH/dx |j> through the Hellmann-Feynman relation: this is the
framework-wide pyrai3md CI-derivative data convention shared with
the molcas and bagel skills. The FSSH integrator builds the
coupling matrix D by dividing the stored vector by (E_j - E_i)
again when projecting onto the nuclear velocity.

All model formulas are evaluated in atomic units (Bohr / Hartree);
the framework coordinates arrive in Angstrom and are converted here
through the skill-local conversion constant.

"""

import warnings

import numpy as np

from .utils import ANGSTROM_TO_BOHR, NAC_ENERGY_GAP_EPS


def model_coordinate(coordinates):
    """Extract the one-dimensional model coordinate.

    Parameters
    ----------
    coordinates : array-like
        Framework coordinates in Angstrom, shape (natom, 3). Only
        the x component of the first (dummy) atom enters the model.

    Returns
    -------
    float
        Model coordinate x in Bohr.
    """
    return float(coordinates[0][0]) * ANGSTROM_TO_BOHR


def diagonalize_hamiltonian(model, x):
    """Diagonalize the diabatic Hamiltonian at x.

    Parameters
    ----------
    model : AnalyticModel
        Model providing hamiltonian(x).
    x : float
        Model coordinate in Bohr.

    Returns
    -------
    energies : numpy.ndarray
        Adiabatic energies sorted ascending, shape (nstate,).
    eigvecs : numpy.ndarray
        Eigenvectors as columns, shape (nstate, nstate); column i
        is the adiabatic state i.
    """
    ham = model.hamiltonian(x)
    energies, eigvecs = np.linalg.eigh(ham)
    return energies, eigvecs


def _default_nac_pairs(nstate):
    """Build the default NAC pair list (all adjacent pairs).

    Parameters
    ----------
    nstate : int
        Number of adiabatic states.

    Returns
    -------
    list of [int, int]
        Pairs [[0, 1], [1, 2], ...] covering adjacent states.
    """
    return [[i, i + 1] for i in range(nstate - 1)]


def calculate_properties(model, coordinates,
                         target_states=None, target_nac_pair=None,
                         with_gradient=True, with_nac=True):
    """Evaluate adiabatic single-point properties at the geometry.

    Parameters
    ----------
    model : AnalyticModel
        Analytic model defining the diabatic Hamiltonian.
    coordinates : array-like
        Framework coordinates in Angstrom, shape (natom, 3).
    target_states : list of int, optional
        States to report (0-based). None reports every state.
    target_nac_pair : list of [int, int], optional
        NAC pairs to report in the given order. None or empty falls
        back to all adjacent pairs.
    with_gradient : bool
        Whether to evaluate adiabatic gradients.
    with_nac : bool
        Whether to evaluate NAC vectors.

    Returns
    -------
    dict
        'energy' : numpy.ndarray, shape (ns,)
        'gradient' : numpy.ndarray, shape (ns, natom, 3), x only
        'nac' : numpy.ndarray, shape (nnac, natom, 3), x only;
            stored as <i| d/dx |j> * (E_j - E_i) = <i| dH/dx |j>
            (Hartree/Bohr)
        gradient / nac are omitted when their flag is False.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    natom = coordinates.shape[0]
    x = model_coordinate(coordinates)

    energies, eigvecs = diagonalize_hamiltonian(model, x)
    nstate = energies.size
    if target_states:
        states = [i for i in target_states if 0 <= i < nstate]
        if not states:
            raise ValueError(
                'target_states %s outside the %d model states'
                % (list(target_states), nstate))
    else:
        states = list(range(nstate))

    properties = {'energy': energies[states].copy()}

    if not (with_gradient or with_nac):
        return properties

    dham = model.hamiltonian_gradient(x)
    # dE_i/dx = <i|dH/dx|i> with eigenvectors stored as columns
    d_energy = np.einsum('ji,jk,ki->i', eigvecs, dham, eigvecs)

    if with_gradient:
        gradient = np.zeros((len(states), natom, 3))
        for row, state in enumerate(states):
            gradient[row, 0, 0] = d_energy[state]
        properties['gradient'] = gradient

    if with_nac:
        pairs = [pair for pair in (target_nac_pair or [])
                 if len(pair) == 2]
        if not pairs:
            pairs = _default_nac_pairs(nstate)
        nac = np.zeros((len(pairs), natom, 3))
        for row, (i, j) in enumerate(pairs):
            gap = energies[j] - energies[i]
            if abs(gap) < NAC_ENERGY_GAP_EPS:
                warnings.warn(
                    'QDM: adiabatic gap |E%d - E%d| = %.3e below '
                    'NAC_ENERGY_GAP_EPS (%.1e); NAC component set '
                    'to zero.' % (j, i, abs(gap), NAC_ENERGY_GAP_EPS))
                continue
            # Framework CI-derivative convention: store d_ij *
            # (E_j - E_i), which equals <i| dH/dx |j>; the FSSH
            # integrator divides by (E_j - E_i) again to recover
            # the standard derivative coupling d_ij. The product
            # stays finite at a crossing, and a degenerate pair is
            # naturally zeroed.
            coupling = np.einsum('j,jk,k', eigvecs[:, i],
                                 dham, eigvecs[:, j])
            nac[row, 0, 0] = coupling
        properties['nac'] = nac

    return properties
