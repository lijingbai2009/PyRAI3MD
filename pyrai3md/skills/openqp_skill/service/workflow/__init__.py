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

This module provides OpenQP workflow compilation.

"""

from .compiler import compile_workflow
from .graph import WorkflowGraph

__all__ = [
    'compile_workflow',
    'WorkflowGraph',
]
