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

This module provides xTB input file building capabilities.

"""

from .input_builder import (
    write_input_file,
    write_xyz_file,
    write_coord_file,
    write_embedding_file,
    build_xtb_args,
)

__all__ = [
    'write_input_file',
    'write_xyz_file',
    'write_coord_file',
    'write_embedding_file',
    'build_xtb_args',
]
