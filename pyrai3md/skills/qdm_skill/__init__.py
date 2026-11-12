"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Public API:
    QDMBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.qdm_skill import QDMBackend

"""

__all__ = ['QDMBackend']


def __getattr__(name):
    """Import the backend only when the public class is requested."""
    if name == 'QDMBackend':
        from .backend import QDMBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
