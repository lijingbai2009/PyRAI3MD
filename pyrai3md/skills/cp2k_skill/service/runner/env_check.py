"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides environment validation for CP2K execution.

"""

import os


def validate_environment(config):
    """Check that the CP2K installation is accessible.

    cp2k_path is the CP2K_HOME installation directory; the
    executable defaults to $CP2K_HOME/exe/local/cp2k.psmp when
    cp2k_exe is empty. Relative or $CP2K_HOME-based cp2k_exe values
    are resolved against cp2k_path before the existence check.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If cp2k_path is not set, does not exist, or the CP2K
        executable cannot be found.
    """
    cp2k_path = config.cp2k_path
    if not cp2k_path:
        raise RuntimeError(
            'cp2k_path is not set. '
            'Set it in pmd_env.yaml (e.g. cp2k_path: /opt/cp2k)'
        )
    if not os.path.isdir(cp2k_path):
        raise RuntimeError(
            f'cp2k_path directory does not exist: {cp2k_path}'
        )
    cp2k_exe = config.cp2k_exe or 'exe/local/cp2k.psmp'
    cp2k_exe = cp2k_exe.replace('$CP2K_HOME', cp2k_path)
    if not os.path.isabs(cp2k_exe):
        cp2k_exe = os.path.join(cp2k_path, cp2k_exe)
    if not os.path.isfile(cp2k_exe):
        raise RuntimeError(
            f'CP2K executable not found at: {cp2k_exe}'
        )
