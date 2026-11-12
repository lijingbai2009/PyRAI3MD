"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This subpackage owns the external strain finite-difference stress
route ('num'): the strain mathematics, the perturbed single-point
runner, and the driver that fans the points out and assembles the
stress tensor.

The route exists because CP2K prints no stress tensor for a TDDFT run
(response_solver.F aborts when STRESS_TENSOR is requested under
TDDFPT), so the excited-state potential stress has to come from energy
differences over deformed cells instead.

"""

from .num_stress import build_payloads, compute_num_stress, execute_points
from .point_runner import run_energy_point
from .strain import (FULL_DIRECTIONS, STRESS_SCOPES, HARTREE_A3_TO_GPA,
                     assemble_stress, cell_volume, difference_component,
                     direction_matrices, perturb)

__all__ = [
    'FULL_DIRECTIONS',
    'STRESS_SCOPES',
    'HARTREE_A3_TO_GPA',
    'assemble_stress',
    'build_payloads',
    'cell_volume',
    'compute_num_stress',
    'difference_component',
    'direction_matrices',
    'execute_points',
    'perturb',
    'run_energy_point',
]
