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

This module provides OpenQP input and run script templates.

"""

from .openqp_templates import (
    INPUT_TEMPLATE,
    GUESS_TEMPLATE,
    SCF_TEMPLATE,
    TDHF_TEMPLATE,
    PROPERTIES_TEMPLATE,
    NAC_TEMPLATE,
    OPTIMIZE_TEMPLATE,
    HESS_TEMPLATE,
)

from .run_templates import (
    build_run_command,
    build_local_script,
    build_slurm_script,
)

__all__ = [
    'INPUT_TEMPLATE',
    'GUESS_TEMPLATE',
    'SCF_TEMPLATE',
    'TDHF_TEMPLATE',
    'PROPERTIES_TEMPLATE',
    'NAC_TEMPLATE',
    'OPTIMIZE_TEMPLATE',
    'HESS_TEMPLATE',
    'build_run_command',
    'build_local_script',
    'build_slurm_script',
]
