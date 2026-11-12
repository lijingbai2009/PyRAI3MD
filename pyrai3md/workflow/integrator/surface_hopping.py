"""
======================================================================
This module provides the surface hopping entry of the NAMD workflow
(dev/methods/MD_DEVELOPMENT.md,
dev/methods/FSSH_DEVELOPMENT.md,
dev/methods/GSH_DEVELOPMENT.md,
dev/methods/LZSH_DEVELOPMENT.md). The factory initializes
the selected algorithm once per trajectory from the input 'sfhp'
keyword and returns the callable state -> state integrator used in
the MD loop:

    nosh    no surface hopping (default)
    gsh     global surface hopping
    fssh    fewest switches surface hopping
    lzsh    Landau-Zener surface hopping

Each algorithm is a callable state -> state object: it evaluates
the hopping condition from the recorded trajectory (history API)
and, when a hop occurs, updates the authoritative active-state
record itself through state.update_system({'current_state': hop}).
The workflow only chains the returned state and never judges the
hop afterwards.

For a multi-center run the gsh/fssh/lzsh selector returns an
ElectronicStepper (integrator/electronic.py): a dispatch layer that
hands each center its own data view and runs the UNCHANGED algorithm
core per center, so the multi-center control stays decoupled from the
hopping physics. C=1 runs a transparent binding (the view IS the real
state), so the single-center path is bit-for-bit the legacy one.

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .electronic import ElectronicStepper

# Methods dispatched through the per-center ElectronicStepper.
_DISPATCH_METHODS = ('gsh', 'fssh', 'lzsh')


def surface_hopping(method, state):
    """Initialize the selected surface hopping algorithm once.

    Parses the trajectory configuration a single time and returns
    the callable integrator used on every timestep (the workflow
    calls it as state = surfacehop(state)).

    Parameters
    ----------
    method : str
        Algorithm selector from the input 'sfhp' keyword: 'nosh',
        'gsh', 'fssh' or 'lzsh' (case-insensitive).
    state : PyRAIState
        State carrying the trajectory configuration; consumed at
        initialization only (fssh parses the dynamics keywords and
        the coupling pair maps here; gsh parses the hopping
        thresholds, spin list and SOC pair map; lzsh parses the
        same plus the timestep). For C>1 the bound partition plan
        sizes one algorithm instance per center.

    Returns
    -------
    callable
        A state -> state integrator: _nosh (pass-through) or the
        ElectronicStepper dispatching the selected algorithm across
        the centers (a single transparent center for C=1).

    Raises
    ------
    ValueError
        If the method is unknown.
    """
    method = str(method).lower()
    if method == 'nosh':
        return _nosh
    if method in _DISPATCH_METHODS:
        return ElectronicStepper(method, state)
    raise ValueError("Unknown surface hopping method: %s" % method)


def _nosh(state):
    """No surface hopping: return the state unchanged."""
    return state
