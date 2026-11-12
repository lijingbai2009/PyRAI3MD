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

This module provides BAGEL input file building capabilities.

"""

from .input_builder import write_input_file, build_input, manage_archives

__all__ = [
    'write_input_file',
    'build_input',
    'manage_archives',
]
