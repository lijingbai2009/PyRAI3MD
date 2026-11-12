"""
======================================================================
This is a standalone LAMMPS calculation skill
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
    LammpsBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.lammps_skill import LammpsBackend
    backend = LammpsBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['LammpsBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'LammpsBackend':
        from .backend import LammpsBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
