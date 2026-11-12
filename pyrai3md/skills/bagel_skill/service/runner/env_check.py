"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides environment validation for BAGEL execution.

"""

import os


def validate_environment(config):
    """Check that the BAGEL installation is accessible.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If bagel_path is not set, does not exist, or the
        BAGEL executable cannot be found.
    """
    bagel_path = config.bagel_path
    if not bagel_path:
        raise RuntimeError(
            'bagel_path is not set. '
            'Set it in pmd_env.yaml (e.g. bagel_path: /opt/bagel)'
        )
    if not os.path.isdir(bagel_path):
        raise RuntimeError(
            f'bagel_path directory does not exist: {bagel_path}'
        )
    bagel_exec = os.path.join(bagel_path, 'bin', 'BAGEL')
    if not os.path.isfile(bagel_exec):
        raise RuntimeError(
            f'BAGEL executable not found at: {bagel_exec}'
        )
