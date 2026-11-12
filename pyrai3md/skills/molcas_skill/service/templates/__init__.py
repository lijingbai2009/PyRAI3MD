"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides Molcas input and run script templates.

"""

from .molcas_templates import (
    GATEWAY_TEMPLATE,
    XFIELD_TEMPLATE,
    SEWARD_TEMPLATE,
    RASSCF_TEMPLATE,
    RASSCF_ALTER_LINE,
    ALASKA_TEMPLATE,
    ALASKA_ROOT_TEMPLATE,
    RASSI_SOC_TEMPLATE,
    CASPT2_TEMPLATE,
    SLAPAF_TEMPLATE,
)

from .run_templates import (
    build_local_script,
    build_slurm_script,
)

__all__ = [
    'GATEWAY_TEMPLATE',
    'XFIELD_TEMPLATE',
    'SEWARD_TEMPLATE',
    'RASSCF_TEMPLATE',
    'RASSCF_ALTER_LINE',
    'ALASKA_TEMPLATE',
    'ALASKA_ROOT_TEMPLATE',
    'RASSI_SOC_TEMPLATE',
    'CASPT2_TEMPLATE',
    'SLAPAF_TEMPLATE',
    'build_local_script',
    'build_slurm_script',
]
