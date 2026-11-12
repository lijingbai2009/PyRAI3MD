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

This module provides shared constants and utility functions for the
QDM skill. Following the skill independence convention the skill
defines its own conversion and protection constants and never imports
framework modules (same precedent as the lammps skill self-contained
constants).

"""

import os
from ..schema.config import DEFAULT
import traceback
import warnings

# Length conversion: the framework stores coordinates in Angstrom;
# all QDM model formulas are evaluated in Bohr (atomic units).
# Value kept identical to pyrai3md.utilities.units.ANGSTROM_TO_BOHR.
BOHR_TO_ANGSTROM = 0.52917721090299996
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# NAC energy-gap protection threshold (Hartree). When the adiabatic
# energy gap |E_j - E_i| falls below this value the corresponding NAC
# component is set to zero: at a degenerate crossing the adiabatic
# eigenvectors lose their meaning, so the transition gradient is not
# reported.
# Same idea and same value as pyrai3md.utilities.units.NAC_ENERGY_GAP_EPS
# (framework side), kept as an independent skill-side constant so the
# skill stays self-contained; the two sides never import each other.
NAC_ENERGY_GAP_EPS = 1e-12


def build_calcdir(base_name='tmp_QDM', execution_id='', base_path=None,
                  project=''):
    """Build the calculation directory path.

    The directory name is controlled by the caller (upper-level
    functionality) to avoid conflicts when multiple backend
    instances are created simultaneously:
      - execution_id='': falls back to the base_name (tmp_QDM)
      - execution_id='tmp_QDM' (single task) or 'tmp_high_qm_task1_QDM':
        the execution namespace is used directly as the directory name

    Parameters
    ----------
    base_name : str
        Fallback directory name for standalone mode.
    execution_id : str
        Execution namespace identifier assigned by the Calculator.
        Empty string means standalone mode (base_name fallback).
    project : str, optional
        Project this calculation runs under. When it is set, the engine
        token of the directory name is replaced by it and every other part
        stays: `tmp_QDM` becomes `tmp_<project>`, and a framework
        namespace keeps its region, role and runtime parts. Empty (the
        default) leaves the naming above untouched.
    base_path : str, optional
        Base directory (e.g. environment scratch_path).
        Defaults to the current working directory.

    Returns
    -------
    str
        Absolute calculation directory path.
    """
    dir_name = execution_id if execution_id else base_name
    if project and project != DEFAULT['project']:
        # The schema's own default is how "no project was declared" arrives
        # here, and it must not rename anything: a framework run names its
        # directory by its execution namespace, which is already unique per
        # task, and renaming it would move the artifacts of a run whose owner
        # never asked for a project.
        if project != os.path.basename(project) or '..' in project:
            raise ValueError(
                'project %r cannot be used as a directory name' % project)
        token = (base_name[len('tmp_'):] if base_name.startswith('tmp_')
                 else base_name)
        parts = dir_name.split('_')
        # Replace the window of segments that spells this engine's token, so
        # every other part of the namespace - region, role, layer, runtime
        # index - stays where it was. The token can span segments (`NN_PES`)
        # and the framework writes it in its own case (`PySCF`), hence the
        # join and the case fold. A name that carries no such window is left
        # alone: the execution namespace distinguishes it already, and
        # inventing a name here would collapse two tasks that share a project
        # into one directory.
        token_parts = token.split('_')
        width = len(token_parts)
        found = -1
        for index in range(len(parts) - width + 1):
            candidate = '_'.join(parts[index:index + width])
            if candidate.lower() == token.lower():
                found = index
        if found >= 0:
            parts[found:found + width] = [project]
            dir_name = '_'.join(parts)
    base = base_path or os.getcwd()
    return os.path.join(base, dir_name)


# Submodel flattening order.  Later submodels override earlier ones
# on name collision; environment last so its scratch_path (file
# placement) wins over calculation.scratch_path.  The framework
# job-control and adaptive-sampling sections are skipped: they drive
# the framework only (jobtype dispatch, tri-state multitask/
# remotetask graph flags) and must not reach the skill.
FLATTEN_ORDER = ('system', 'properties', 'calculation', 'environment')


def flatten_extended_config(config):
    """Flatten a nested extended PyRAIConfig into a legacy flat dict.

    .. deprecated::
        Use SkillConfig nested model instead. This function is retained
        only for backward compatibility during the migration period.

    The nested config built by PyRAIState (framework submodels plus
    skill calculation/environment/properties) is flattened so that
    existing services keep consuming a flat dictionary.

    Framework-only sections (control, adaptive) are skipped.

    Parameters
    ----------
    config : BaseModel
        Nested extended PyRAIConfig instance.

    Returns
    -------
    dict
        Flat configuration dictionary.
    """
    warnings.warn(
        "flatten_extended_config is deprecated; "
        "use SkillConfig nested model directly",
        DeprecationWarning, stacklevel=2,
    )
    nested = config.model_dump()
    flat = {}
    for sub_name in FLATTEN_ORDER:
        sub_data = nested.get(sub_name)
        if not isinstance(sub_data, dict):
            continue
        flat.update(sub_data)
    return flat


def safe_run(func, *args, **kwargs):
    """Execute a function without interrupting the main process.

    Parameters
    ----------
    func : callable
        Function to execute.

    Returns
    -------
    str
        'success' or traceback string on failure.
    """
    try:
        func(*args, **kwargs)
        status = 'success'
    except Exception:
        status = traceback.format_exc()

    return status
