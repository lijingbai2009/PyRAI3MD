"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Public API:
    OpenQPBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.openqp_skill import OpenQPBackend
    backend = OpenQPBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['OpenQPBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'OpenQPBackend':
        from .backend import OpenQPBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
