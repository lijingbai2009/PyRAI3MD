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

This module provides SLURM and execution script templates.

"""

from .run_templates import SLURM_SCRIPT_TEMPLATE

__all__ = [
    'SLURM_SCRIPT_TEMPLATE',
]
