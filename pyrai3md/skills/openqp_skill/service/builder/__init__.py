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

This module provides OpenQP input file building capabilities.

"""

from .input_builder import write_input_file, build_input, write_xyz_file
from .input_builder import (
    build_input_section,
    build_guess_section,
    build_scf_section,
    build_tdhf_section,
    build_properties_section,
    build_nac_section,
    build_optimize_section,
    build_hess_section,
)

__all__ = [
    'write_input_file',
    'build_input',
    'write_xyz_file',
    'build_input_section',
    'build_guess_section',
    'build_scf_section',
    'build_tdhf_section',
    'build_properties_section',
    'build_nac_section',
    'build_optimize_section',
    'build_hess_section',
]
