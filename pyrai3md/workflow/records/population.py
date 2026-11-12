"""
======================================================================
Population readings of one electronic dynamics step.

Two pure helpers shared by the md guard and by anything that reports a
population: the real diagonal of the density matrix a frame carries, and
the judgment of whether a population left the physical band. They read
data and never write a state.

The band is `[0 - tolerance, 1 + tolerance]`, the legacy rule: a
population slightly outside the interval is an integration artifact the
run reports, and a non-finite value is a defect. Which of the two ends
of that interval a caller acts on is the caller's decision; this module
only answers whether the band was left.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np


def state_population(system):
    """Real diagonal of the electronic density matrix.

    Parameters
    ----------
    system : dict or None
        A system section (a history frame's dump, or a mapping carrying
        `state_density`).

    Returns
    -------
    list of float or None
        Populations, or None when the section carries no readable
        density. Both stored shapes are read: the (nstate, nstate, 2)
        complex pair list and a plain (nstate, nstate) real matrix.
    """
    system = system or {}
    density = system.get('state_density') if isinstance(system, dict) \
        else getattr(system, 'state_density', None)
    if density is None:
        return None
    array = np.asarray(density, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 2:
        array = array[..., 0]
    if array.ndim == 2 and array.shape[0] == array.shape[1] and array.size:
        return [float(value) for value in np.diag(array)]
    return None


def population_bad(populations, tolerance):
    """Whether any population is non-finite or outside [0-tol, 1+tol]."""
    if populations is None:
        return False
    for value in populations:
        if not np.isfinite(value):
            return True
        if value < -float(tolerance) or value > 1.0 + float(tolerance):
            return True
    return False