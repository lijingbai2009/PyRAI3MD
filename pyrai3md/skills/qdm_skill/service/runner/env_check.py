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

This module provides environment validation for QDM execution.

"""


def validate_environment(config):
    """Check that the QDM runtime dependencies are importable.

    QDM is a pure Python/numpy service and has no external program
    dependency; only the numpy import is verified.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If numpy cannot be imported in the current environment.
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise RuntimeError(
            'numpy is not installed or cannot be imported. '
            'Install it with: pip install numpy'
        )
