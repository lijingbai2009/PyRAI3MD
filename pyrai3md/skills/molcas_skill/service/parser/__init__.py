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

This module provides Molcas output parsing capabilities.

"""

from .output_parser import parse_output, format_results
from .log_parser import parse_log
from .orbital_parser import manage_orbital_files, save_orbital_files

__all__ = [
    'parse_output',
    'format_results',
    'parse_log',
    'manage_orbital_files',
    'save_orbital_files',
]
