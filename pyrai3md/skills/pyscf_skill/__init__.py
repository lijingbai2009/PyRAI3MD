"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Public API:
    PySCFBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.pyscf_skill import PySCFBackend
    backend = PySCFBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['PySCFBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'PySCFBackend':
        from .backend import PySCFBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
