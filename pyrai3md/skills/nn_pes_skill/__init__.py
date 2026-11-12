"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Public API:
    NNPESBackend — main backend class, provides calculate()

Usage:
    from pyrai3md.skills.nn_pes_skill import NNPESBackend

The backend is resolved on first attribute access rather than at import
time, because importing it pulls in TensorFlow. Framework discovery walks
the skills by importing only each skill's query module, and a submodule
import runs this file first: an eager import here would make the whole
skill disappear on a machine that has no optional neural network stack,
and disappear silently.

"""

__all__ = ['NNPESBackend']


def __getattr__(name):
    if name == 'NNPESBackend':
        from .backend import NNPESBackend as _backend
        return _backend
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}; the nn_pes skill "
        "exports NNPESBackend, which needs tensorflow "
        "(install 'tensorflow>=2.20')")
