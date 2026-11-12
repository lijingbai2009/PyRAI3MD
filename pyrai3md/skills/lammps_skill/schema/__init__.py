"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Schema modules for the LAMMPS skill.

"""

from .config import SkillConfig, Config, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES

__all__ = [
    'SkillConfig',
    'Config',
    'FIELD_TO_SUBMODEL',
    'SUBMODEL_CLASSES',
]
