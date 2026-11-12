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

This module defines the core service layer of the PySCF skill.
It provides workflow compilation, input building, execution, and parsing.

"""

from .workflow import compile_workflow, WorkflowGraph
from .builder import build_runtime, dispatch_calculation
from .runner import run_local, run_slurm
from .parser import parse_output, format_results

__all__ = [
    'compile_workflow',
    'WorkflowGraph',
    'build_runtime',
    'dispatch_calculation',
    'run_local',
    'run_slurm',
    'parse_output',
    'format_results',
]
