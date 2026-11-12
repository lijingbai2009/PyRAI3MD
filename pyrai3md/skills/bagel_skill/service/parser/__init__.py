"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides BAGEL output parsing capabilities.

"""

from .output_parser import parse_output, format_results
from .log_parser import parse_log

__all__ = [
    'parse_output',
    'format_results',
    'parse_log',
]
