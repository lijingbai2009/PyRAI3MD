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

This module provides LAMMPS input file building capabilities.

"""

from .input_builder import (
    write_input_file,
    write_data_file,
    write_input_init_file,
    write_input_settings_file,
    write_input_package,
    write_static_input_files,
    build_script_head,
    build_step_tail,
    build_thermo_section,
    build_run_section,
    box_to_bound_numbers,
    resolve_live_box,
)
from .ff_model import (
    FFModelUnsupported,
    build_ff_model,
    load_cached_model,
    model_path,
    read_templates,
    render_data,
    render_init,
    render_settings,
    save_model,
)

__all__ = [
    'write_input_file',
    'write_data_file',
    'write_input_init_file',
    'write_input_settings_file',
    'write_input_package',
    'write_static_input_files',
    'build_script_head',
    'build_step_tail',
    'build_thermo_section',
    'build_run_section',
    'box_to_bound_numbers',
    'resolve_live_box',
    'FFModelUnsupported',
    'build_ff_model',
    'load_cached_model',
    'model_path',
    'read_templates',
    'render_data',
    'render_init',
    'render_settings',
    'save_model',
]
