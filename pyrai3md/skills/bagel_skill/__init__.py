"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Public API:
    BagelBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.bagel_skill import BagelBackend
    backend = BagelBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['BagelBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'BagelBackend':
        from .backend import BagelBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
