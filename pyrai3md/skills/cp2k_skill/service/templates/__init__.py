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

This module provides input and run script templates for CP2K.

"""

from .run_templates import build_local_script, build_slurm_script

__all__ = [
    'build_local_script',
    'build_slurm_script',
]
