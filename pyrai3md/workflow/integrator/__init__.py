"""
======================================================================
This package provides the nuclear propagation integrators and the
independent physics operations of the NAMD workflow.

The module layout follows the design document
(dev/methods/MD_DEVELOPMENT.md):

    integrator/
        verlet.py           velocity-Verlet integrator functions
        operations.py       independent workflow operations (phase
                            correction, constraints, the
                            initial-velocity control, velocity
                            correction, ensemble control, and the
                            constraint reading of a recorded frame);
                            the periodic wrap lives in data/geometry.py
        surface_hopping.py  surface hopping factory (nosh/gsh/fssh,
                            initializes the algorithm once)
        fssh.py             FSSH integrator class (state management,
                            hopping decision, velocity adjustment)
        fssh_numpy.py       FSSH NumPy reference kernels
        fssh_numba.py       FSSH optional Numba kernels
        fssh_profile.py     FSSH function-level timing
        gsh.py              GSH (Zhu-Nakamura) integrator class
                            (internal conversion and intersystem
                            crossing probabilities)
        tsh_helper.py       shared surface hopping velocity
                            adjustment layer (adjust/reflect)

Integrators only perform the time integration of nuclear coordinates
and velocities; every other physical effect is an independent
operation inserted into the NAMD timestep data flow.

Aug 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .verlet import CellDivergenceError, verlet_i, verlet_ii
from .operations import (
    phase_correction,
    apply_constrain,
    velocity_initialize,
    velocity_correction,
    apply_ensemble,
    verify_recorded,
)
from .surface_hopping import surface_hopping

__all__ = [
    'CellDivergenceError',
    'verlet_i',
    'verlet_ii',
    'phase_correction',
    'apply_constrain',
    'velocity_initialize',
    'velocity_correction',
    'apply_ensemble',
    'verify_recorded',
    'surface_hopping',
]
