"""
======================================================================
This module provides the MEPOptimizer for minimum energy path (MEP)
calculations.

A minimum energy path is walked downhill from a starting geometry, one
path point at a time: each point is a constrained minimization on the
sphere of radius `optimize.path_radius` centered on the point before it
(arc length in mass-weighted coordinates), and the sphere is re-centred
on every accepted point. Two constraint models do that minimization and
`optimize.path_constraint` picks between them
(`workflow/optimizer/pathconstraint.py`): the classic mass-weighted
hypersphere step of Gonzalez & Schlegel (J. Phys. Chem. 94, 5523 (1990)),
which holds the point exactly on the sphere and rotates it towards the
tangential descent direction, and the one-sided ReLU-gated constraint
potential, which is an ordinary minimization of `E + phi` that the
penalty pushes out to the sphere and holds there.

The seed is the geometry the run was given, and the first step leaves it
along the gradient. A starting geometry with no gradient to leave on - a
transition state, or a minimum - has no direction at all, and the walk
says so by name instead of walking nowhere: an intrinsic reaction
coordinate is the method that seeds a direction, from the imaginary mode
of the saddle (`optimizer/irc.py`).

The walk is driven by the Path workflow, not by `Optimize.run`: see
`optimizer/pathbase.py` for the joint vector, the mass weights, the
macro/micro state machine and the reuse of the cell half of a
variable-cell search.

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .pathbase import WalkerPathOptimizer


class MEPOptimizer(WalkerPathOptimizer):
    """Minimum energy path, walked downhill from the geometry the run was given.

    Parameters
    ----------
    path_radius : float
        The arc length of one accepted path point, in Bohr, on the normalized
        mass-weighted coordinates (`optimize.path_radius`).
    inner_maxit : int
        How many probes one path point may spend before the walk gives up on
        finding its constrained minimum.
    inner_tol : float
        The tolerance of the model's own convergence reading.
    constraint_model : str
        'sphere' or 'relu'.

    Notes
    -----
    What a run reports is in `dev/methods/PATH_DEVELOPMENT.md`: `path.xyz`
    (the latest accepted point), `path_geom.xyz` (every accepted point, which is
    the walked path), `path_profile.txt` (one row per point: energy, arc length,
    radius, residual) and `path_status.txt` (one row per evaluation). The state
    this run leaves behind holds the last accepted point, because a path has no
    single optimized geometry.
    """

    # A path is driven by the Path workflow, not by this class's own `run`; the
    # statement is here because every search carries its own (pathbase's
    # `_compute_step` is what refuses a mis-wired caller by name).
    is_implemented = True
    path_kind = 'walker'


__all__ = ['MEPOptimizer']