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

This module defines the core service layer of the OpenQP skill.
It provides workflow compilation, input building, execution, and parsing.

"""

from .workflow import compile_workflow, WorkflowGraph
from .builder import build_input, write_input_file, write_xyz_file
from .runner import run_local, run_slurm
from .parser import parse_output, format_results

__all__ = [
    'compile_workflow',
    'WorkflowGraph',
    'build_input',
    'write_input_file',
    'write_xyz_file',
    'run_local',
    'run_slurm',
    'parse_output',
    'format_results',
]
