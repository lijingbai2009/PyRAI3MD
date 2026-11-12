"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Public API:
    Cp2kBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.cp2k_skill import Cp2kBackend
    backend = Cp2kBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['Cp2kBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'Cp2kBackend':
        from .backend import Cp2kBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
