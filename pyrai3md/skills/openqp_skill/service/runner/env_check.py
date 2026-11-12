"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides environment validation for OpenQP execution.

"""

import os
import shutil


def validate_environment(config):
    """Check that the OpenQP installation is accessible.

    The 'openqp' command must be available in PATH (provided by a
    proper OpenQP installation). When openqp_path is set, it is only
    used to export OPENQP_ROOT in the run script; its directory must
    exist. In MPI mode (use_mpi=True) an MPI launcher must also be
    available.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If the openqp executable (or the MPI launcher) cannot
        be found.
    """
    openqp_path = config.openqp_path
    if openqp_path and not os.path.isdir(openqp_path):
        raise RuntimeError(
            f'openqp_path directory does not exist: {openqp_path}'
        )
    if shutil.which('openqp') is None:
        raise RuntimeError(
            'openqp executable not found in PATH. '
            'Install OpenQP properly so that the openqp command '
            'is available in PATH.'
        )

    if config.use_mpi:
        mpi_command = str(config.mpi_command).strip()
        launcher = mpi_command.split()[0] if mpi_command else 'mpirun'
        if shutil.which(launcher) is None:
            raise RuntimeError(
                f'use_mpi is enabled but the MPI launcher '
                f'({launcher}) was not found in PATH.'
            )
