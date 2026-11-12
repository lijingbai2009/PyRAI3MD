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

This module provides workflow compilation from profiles to BAGEL step chains.

"""

from .compiler import compile_workflow
from .graph import WorkflowGraph

__all__ = [
    'compile_workflow',
    'WorkflowGraph',
]
