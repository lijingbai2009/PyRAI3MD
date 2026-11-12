"""
======================================================================
This module provides the shared velocity adjustment layer of the
surface hopping algorithms of the NAMD workflow
(dev/methods/FSSH_DEVELOPMENT.md,
dev/methods/GSH_DEVELOPMENT.md). It mirrors the legacy
PyRAI2MD tsh_helper.py: every hopping algorithm (FSSH, GSH and any
future method) consumes the same adjust/reflect semantics from this
single module instead of carrying private copies.

Atomic units throughout: Hartree energies, Bohr/au velocities, au
masses. The algorithms decide which adjustment applies (NAC-vector
direction vs isotropic rescaling); this module only implements the
mechanics.

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np


def reflect_velo(velo, nac, reflect):
    """Reflect the velocity at a frustrated hopping.

    Parameters
    ----------
    velo : np.ndarray
        Nuclear velocity (natom, 3), Bohr/au.
    nac : np.ndarray
        NAC vector direction (natom, 3).
    reflect : int
        Reflection mode: 1 reverses the full velocity, 2 reflects
        along the NAC direction.

    Returns
    -------
    np.ndarray
        Reflected velocity.
    """
    if reflect == 1:
        return -velo
    if reflect == 2:
        return velo - 2 * np.sum(velo * nac) / np.sum(nac * nac) * nac
    return velo


def adjust_velo(energy_old, energy_new, velo, mass_au, nac, adjust,
                reflect):
    """Adjust the velocity along the NAC vector after a hop.

    Ported from the legacy adjust_velo (atomic units throughout:
    Hartree energies, Bohr/au velocities, au masses).

    Parameters
    ----------
    energy_old : float
        Energy of the departure state (Hartree).
    energy_new : float
        Energy of the arrival state (Hartree).
    velo : np.ndarray
        Nuclear velocity (natom, 3), Bohr/au.
    mass_au : np.ndarray
        Atomic masses (natom,) in atomic units.
    nac : np.ndarray
        NAC vector (natom, 3).
    adjust : int
        Adjustment mode (0: no rescale, 1: global rescale,
        2: rescale along the NAC).
    reflect : int
        Frustrated-hop reflection mode.

    Returns
    -------
    (np.ndarray, int)
        Adjusted velocity and the frustrated flag (1 when the hop
        lacked the kinetic energy).
    """
    kinetic = np.sum(0.5 * mass_au[:, None] * velo ** 2)
    frustrated = 0

    if adjust == 0:
        del_kinetic = energy_old - energy_new + kinetic
        if del_kinetic < 0:
            return reflect_velo(velo, nac, reflect), 1
        return velo, 0

    if adjust == 1:
        del_kinetic = energy_old - energy_new + kinetic
        if del_kinetic >= 0 and kinetic > 0:
            return velo * (del_kinetic / kinetic) ** 0.5, 0
        return reflect_velo(velo, nac, reflect), 1

    if adjust == 2:
        mass_matrix = mass_au[:, None]
        a = np.sum(nac * nac / (2 * mass_matrix))
        b = np.sum(velo * nac)
        discriminant = 4 * a * (energy_old - energy_new) + b ** 2
        if discriminant >= 0:
            if b < 0:
                f = (b + discriminant ** 0.5) / (2 * a)
            else:
                f = (b - discriminant ** 0.5) / (2 * a)
            return velo - f * nac / mass_matrix, 0
        return reflect_velo(velo, nac, reflect), 1

    return velo, 0


def adjust_velo_isotropic(energy_old, energy_new, velo, mass_au):
    """Rescale the velocity isotropically when no NAC vector exists.

    The explicit fallback policy for NAC-unavailable hops (design
    document section 18): kappa-TDC carries no NAC vector and GSH
    intersystem crossings carry no approximate NAC vector, so the
    hop is energy-balanced by a global velocity rescaling instead of
    the legacy ones((natom, 3)) placeholder direction. Insufficient
    kinetic energy marks the hop frustrated (velocity kept, no hop).

    Parameters
    ----------
    energy_old : float
        Energy of the departure state (Hartree).
    energy_new : float
        Energy of the arrival state (Hartree).
    velo : np.ndarray
        Nuclear velocity (natom, 3), Bohr/au.
    mass_au : np.ndarray
        Atomic masses (natom,) in atomic units.

    Returns
    -------
    (np.ndarray, int)
        Adjusted velocity and the frustrated flag.
    """
    kinetic = np.sum(0.5 * mass_au[:, None] * velo ** 2)
    del_kinetic = energy_old - energy_new + kinetic
    if del_kinetic >= 0 and kinetic > 0:
        return velo * (del_kinetic / kinetic) ** 0.5, 0
    return velo, 1


def energy_of(results, nstate):
    """Read the state energies of a results dict, or None.

    The adaptive exploration bias keeps the unbiased energies under
    `unbiased_energy`; the hopping decision reads those.
    """
    energy = results.get('unbiased_energy')
    if energy is None:
        energy = results.get('energy')
    if energy is None:
        return None
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    if energy.size < nstate:
        return None
    return energy[:nstate]


def soc_value(handler, i, active, soc_values):
    """SOC value (cm-1) of a pair, or None (undeclared/missing)."""
    key = tuple(sorted((i, active)))
    if key not in handler.soc_index:
        return None
    if soc_values is None:
        return None
    soc_array = np.asarray(soc_values, dtype=np.float64).reshape(-1)
    idx = handler.soc_index[key]
    if idx >= soc_array.size:
        return None
    return float(soc_array[idx])


def write_pure(handler, state, active):
    """Write the pure-state density of one state and return."""
    nstate = handler.nstate
    density = np.zeros((nstate, nstate), dtype=np.complex128)
    density[active, active] = 1.0
    state.update_system({
        'state_density': np.stack(
            [np.real(density), np.imag(density)],
            axis=-1).tolist(),
    })
    population = [0.0] * nstate
    population[active] = 1.0
    state.update_dynamics({'state_population': population})
    return state
