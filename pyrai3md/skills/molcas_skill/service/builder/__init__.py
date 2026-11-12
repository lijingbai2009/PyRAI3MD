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

This module provides Molcas input file building capabilities.

"""

from .input_builder import (write_input_file, build_input, write_xyz_file,
                            write_velocity_file, manage_orbitals)
from .module_builders import (
    build_gateway,
    build_seward,
    build_rasscf,
    build_scf,
    build_alaska,
    build_rassi,
    build_caspt2,
    build_slapaf,
    build_dynamix,
    build_surfacehop,
    build_mckinley,
)

__all__ = [
    'write_input_file',
    'build_input',
    'write_xyz_file',
    'write_velocity_file',
    'manage_orbitals',
    'build_gateway',
    'build_seward',
    'build_rasscf',
    'build_scf',
    'build_alaska',
    'build_rassi',
    'build_caspt2',
    'build_slapaf',
    'build_dynamix',
    'build_surfacehop',
    'build_mckinley',
]
