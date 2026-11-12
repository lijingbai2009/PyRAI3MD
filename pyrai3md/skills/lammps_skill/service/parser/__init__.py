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

This module provides LAMMPS output parsing capabilities.

"""

from .log_parser import parse_log
from .output_parser import (format_persistent_results, format_results,
                            parse_output, pressure6_to_stress_tensor,
                            to_native_gradient)

__all__ = [
    'parse_log',
    'parse_output',
    'format_results',
    'format_persistent_results',
    'pressure6_to_stress_tensor',
    'to_native_gradient',
]
