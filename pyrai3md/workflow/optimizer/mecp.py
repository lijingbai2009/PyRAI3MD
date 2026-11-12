"""
======================================================================
This module provides the MECPOptimizer for geometry optimization towards
a minimum energy crossing point (MECP) between surfaces of different
spin, which is what a singlet-triplet crossing point is.

The search is the crossing machinery of `meci.py` with the derivative
coupling dropped from the branching plane, and the dropping is physics
rather than a missing input: two states of different spin are not coupled
by nuclear motion in the non-relativistic Hamiltonian, so there is no
coupling vector to span a plane with. What is left is the gradient
difference, which is real and is the direction the gap closes along, and
the average gradient outside it.

The consequence a reader should know: the crossing of two such surfaces is
not forced to be a point the way two same-spin surfaces avoid each other, so
this search finds the lowest geometry where the two surfaces meet, and the
spin-orbit coupling that turns that meeting into an actual transition rate is
a separate calculation on the geometry it returns.

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .meci import CrossingOptimizer


class MECPOptimizer(CrossingOptimizer):
    """Optimizer for minimum energy crossing point search.

    A singlet and a triplet (or any two states of different spin) have no
    derivative coupling between them, so the plane is one direction wide and
    the run reports one direction.
    """

    uses_derivative_coupling = False
