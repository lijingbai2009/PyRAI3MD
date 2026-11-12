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

This module provides environment validation for PySCF execution.

"""


def validate_environment(config):
    """Check that PySCF is importable.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If PySCF cannot be imported in the current environment.
    """
    try:
        import pyscf  # noqa: F401
    except ImportError:
        raise RuntimeError(
            'PySCF is not installed or cannot be imported. '
            'Install it with: pip install pyscf'
        )
