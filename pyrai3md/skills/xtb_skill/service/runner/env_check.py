"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides environment validation for xTB execution.

"""

import os
import shutil


def validate_environment(config):
    """Check that the xTB installation is accessible.

    The xtb_path may point to the installation directory (the
    executable is then expected at '$xtb_path/bin/xtb') or directly
    to the xTB binary. When xtb_path is empty, the 'xtb' executable
    must be available in the system PATH.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with xTB runtime settings.

    Raises
    ------
    RuntimeError
        If the xTB executable cannot be located.
    """
    xtb_path = config.xtb_path

    if not xtb_path:
        if shutil.which('xtb') is None:
            raise RuntimeError(
                'xtb executable not found in PATH. '
                'Set xtb_path in pmd_env.yaml '
                '(e.g. xtb_path: /opt/xtb)'
            )
        return

    if os.path.isdir(xtb_path):
        binary = os.path.join(xtb_path, 'bin', 'xtb')
        if not os.path.isfile(binary):
            raise RuntimeError(
                f'xtb executable not found at: {binary}'
            )
    elif not os.path.isfile(xtb_path):
        raise RuntimeError(
            f'xtb path does not exist: {xtb_path}'
        )
