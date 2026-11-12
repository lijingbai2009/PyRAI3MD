"""
======================================================================
Planning layer for PyRAI3MD.

This package provides the capability resolution logic for the planning
layer. It resolves jobtype tokens into capability name strings,
derives additional capabilities from dynamics keywords, enforces
multi-backend constraints, and distributes per-skill capabilities.

The Planning layer determines WHAT to compute. It never executes
calculations, imports Skills, or instantiates backends. Planning
terminates at PyRAIState construction.

Available capabilities:
- energy: Total electronic energy
- gradient: Atomic gradients (dE/dR)
- hessian: Hessian matrix (d2E/dR2)
- nac: Nonadiabatic coupling vectors
- soc: Spin-orbit coupling matrix elements
- dipole: Electric dipole moment
- transition_dipole: Transition dipole moments
- overlap: Wavefunction overlap
- stress: Whole-system stress tensor (derived from NPT ensemble or
  optimize_cell; consumed by the barostats / cell optimization)
- field: QM/MM field duty (the probe that measures the environment's
  potential and field; not a user jobtype token)
- qmmm: QM/MM energy duty (the classical side of an additive run; not a
  user jobtype token)

The list is `resolver.CAPABILITY_NAMES`; a criterion compares the two so
this paragraph cannot go stale.

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .resolver import resolve_capabilities, CAPABILITY_NAMES
from .task import resolve_task_queue, resolve_composite_type
