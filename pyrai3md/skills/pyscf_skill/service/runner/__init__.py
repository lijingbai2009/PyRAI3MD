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

This module provides PySCF execution runners.

The get_runner function selects the appropriate runner (local or slurm)
based on the execution_mode in the configuration dictionary.

"""

import os

from .local_runner import run_local, to_list
from .slurm_runner import run_slurm
from .env_check import validate_environment


def get_runner(config):
    """Select the appropriate runner based on execution_mode.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with environment.execution_mode.
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


def results_file_name(config_file):
    """Return the agent results file path for one config file.

    The cluster-side agent writes the formatted results next to the
    serialized config; the backend collects them through the same
    naming convention after the blocking submission returns.

    Parameters
    ----------
    config_file : str
        Path of the serialized config json.

    Returns
    -------
    str
        '{config basename without extension}_results.json'.
    """
    return os.path.splitext(config_file)[0] + '_results.json'


__all__ = [
    'get_runner',
    'results_file_name',
    'run_local',
    'run_slurm',
    'to_list',
    'validate_environment',
]
