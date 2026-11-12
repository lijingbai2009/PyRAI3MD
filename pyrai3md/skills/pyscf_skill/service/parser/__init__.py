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

This module provides PySCF output parsing capabilities.
For PySCF (Python-native), parsing means extracting results from
the runtime dictionary.

"""

from .output_parser import parse_output
from .result_formatter import format_results

__all__ = [
    'parse_output',
    'format_results',
]
