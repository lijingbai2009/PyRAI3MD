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

This module provides environment validation for ORCA execution.

"""

import os


def validate_environment(config):
    """Check that the ORCA installation is accessible.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If orca_path is not set, does not exist, or the orca
        executable cannot be found.
    """
    orca_path = config.orca_path
    if not orca_path:
        raise RuntimeError(
            'orca_path is not set. '
            'Set it in pmd_env.yaml (e.g. orca_path: /opt/orca)'
        )
    if not os.path.isdir(orca_path):
        raise RuntimeError(
            f'orca_path directory does not exist: {orca_path}'
        )
    orca_exe = os.path.join(orca_path, 'orca')
    if not os.path.isfile(orca_exe):
        raise RuntimeError(
            f'orca executable not found at: {orca_exe}'
        )
