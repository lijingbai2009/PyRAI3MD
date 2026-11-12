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

This module provides output parsing for CP2K calculations.

"""

from .output_parser import parse_output, merge_job_results, format_results

__all__ = [
    'parse_output',
    'merge_job_results',
    'format_results',
]
