"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the core service layer of the QDM skill.
It provides analytic model evaluation, single-point property
calculation, and the split-operator reference dynamics solver.

"""

from .models import MODEL_REGISTRY, get_model
from .calculator import calculate_properties
from .reference import run_reference

__all__ = [
    'MODEL_REGISTRY',
    'get_model',
    'calculate_properties',
    'run_reference',
]
