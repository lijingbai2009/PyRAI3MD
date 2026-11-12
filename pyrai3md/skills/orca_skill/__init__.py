"""
======================================================================
This is a standalone ORCA calculation skill
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
    OrcaBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.orca_skill import OrcaBackend
    backend = OrcaBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['OrcaBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'OrcaBackend':
        from .backend import OrcaBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
