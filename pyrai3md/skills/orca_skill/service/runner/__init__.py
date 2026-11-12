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

This module provides ORCA execution runners.

The get_runner function selects the appropriate runner (local or slurm)
based on the execution_mode in the configuration dictionary.

"""

from .local_runner import run_local
from .slurm_runner import run_slurm
from .env_check import validate_environment


def get_runner(config):
    """Select the appropriate runner based on execution_mode.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with 'execution_mode'.
        'slurm' selects run_slurm; any other value selects run_local.

    Returns
    -------
    callable
        run_local or run_slurm function.
    """
    execution_mode = config.environment.execution_mode
    if execution_mode == 'slurm':
        return run_slurm
    return run_local


__all__ = [
    'get_runner',
    'run_local',
    'run_slurm',
    'validate_environment',
]
