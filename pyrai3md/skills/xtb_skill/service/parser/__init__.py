"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides xTB output parsing capabilities.

"""

from .output_parser import parse_output, format_results

__all__ = [
    'parse_output',
    'format_results',
]
