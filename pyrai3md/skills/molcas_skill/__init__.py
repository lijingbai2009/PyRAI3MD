"""
======================================================================
This is a standalone OpenMolcas calculation skill
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
    MolcasBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.molcas_skill import MolcasBackend
    backend = MolcasBackend(config)
    results = backend.calculate(atoms, coordinates)

"""

__all__ = ['MolcasBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'MolcasBackend':
        from .backend import MolcasBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
