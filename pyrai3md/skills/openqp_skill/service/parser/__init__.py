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

This module provides OpenQP output parsing capabilities.

"""

from .output_parser import parse_output, format_results, parse_export_files
from .log_parser import parse_log

__all__ = [
    'parse_output',
    'format_results',
    'parse_export_files',
    'parse_log',
]
