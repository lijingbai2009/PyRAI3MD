"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides shared helpers for the nn_pes skill services:
calculation directory naming, legacy config flattening and the
JSON-safe conversion of result arrays.

"""

import os
from ..schema.config import DEFAULT

# Submodel flattening order. Later submodels override earlier ones on
# name collision; environment last so its scratch_path (file placement)
# wins over calculation.scratch_path. The framework job-control and
# adaptive-sampling sections are skipped: they drive the framework only
# and must not reach the skill.
FLATTEN_ORDER = ('system', 'properties', 'calculation', 'environment')


# The token is the folder stem with '_skill' dropped, upper-cased, which
# is what the framework's execution namespace uses for this skill
# ('nn_pes_skill' -> tmp_NN_PES); spelling the fallback differently
# would give one case two calculation directories depending on which
# entry point ran it.
def build_calcdir(base_name='tmp_NN_PES', execution_id='', base_path=None,
                  project=''):
    """Build the calculation directory path.

    The directory name is controlled by the caller (upper-level
    functionality) to avoid conflicts when multiple backend instances
    are created simultaneously:
      - execution_id='': falls back to the base_name (tmp_NN_PES)
      - execution_id='tmp_NN_PES' (single task) or 'tmp_high_qm_task1_NN_PES':
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
        stays: `tmp_NN_PES` becomes `tmp_<project>`, and a framework
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


def flatten_extended_config(config):
    """Flatten a nested extended config model into a legacy flat dict.

    Parameters
    ----------
    config : BaseModel
        Extended configuration carrying submodels.

    Returns
    -------
    dict
        Flat dictionary of the fields the skill understands.
    """
    flat = {}
    for name in FLATTEN_ORDER:
        section = getattr(config, name, None)
        if section is None:
            continue
        fields = getattr(type(section), 'model_fields', None)
        if not fields:
            continue
        for field in fields:
            flat[field] = getattr(section, field)
    return flat


def safe_run(func, *args, **kwargs):
    """Run a callable and report failure as text instead of raising.

    The agent entry point uses this so a crashed calculation still writes
    a log file and a non-zero exit status.

    Parameters
    ----------
    func : callable
        Zero-argument work function.

    Returns
    -------
    str
        'success' or the captured traceback.
    """
    import traceback
    try:
        func(*args, **kwargs)
    except Exception:
        return traceback.format_exc()
    return 'success'


def to_list(value):
    """Convert numpy and TensorFlow arrays in a result tree to lists.

    Parameters
    ----------
    value : dict
        Result mapping that may carry array values.

    Returns
    -------
    dict
        JSON-safe copy of the mapping.
    """
    converted = {}
    for key, item in value.items():
        if hasattr(item, 'tolist'):
            converted[key] = item.tolist()
        elif isinstance(item, (list, tuple)):
            converted[key] = [to_list({'v': x})['v'] if hasattr(x, 'tolist')
                              else x for x in item]
        else:
            converted[key] = item
    return converted
