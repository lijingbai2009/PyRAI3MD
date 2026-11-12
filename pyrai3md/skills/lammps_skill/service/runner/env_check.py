"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides environment validation for LAMMPS execution.

"""

import os
import shutil


def validate_environment(config):
    """Check that the LAMMPS installation is accessible.

    The lammps_path may point to the installation directory (the
    executable is then expected at '$lammps_path/bin/lmp', legacy
    convention) or directly to the LAMMPS binary. When lammps_path
    is empty, the 'lmp' executable must be available in the system
    PATH.

    Parameters
    ----------
    config : SkillConfig
        Validated skill configuration containing lammps_path.

    Raises
    ------
    RuntimeError
        If the LAMMPS executable cannot be located.
    """
    lammps_path = config.lammps_path

    if not lammps_path:
        if shutil.which('lmp') is None:
            raise RuntimeError(
                'lmp executable not found in PATH. '
                'Set lammps_path in pmd_env.yaml '
                '(e.g. lammps_path: /opt/lammps)'
            )
        return

    if os.path.isdir(lammps_path):
        binary = os.path.join(lammps_path, 'bin', 'lmp')
        if not os.path.isfile(binary):
            raise RuntimeError(
                f'lmp executable not found at: {binary}'
            )
    elif not os.path.isfile(lammps_path):
        raise RuntimeError(
            f'lammps path does not exist: {lammps_path}'
        )
