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

This module provides PySCF input building capabilities.
For PySCF, building means constructing the Mole and mean-field objects.

"""

from .input_builder import build_runtime, dispatch_calculation

__all__ = [
    'build_runtime',
    'dispatch_calculation',
]
