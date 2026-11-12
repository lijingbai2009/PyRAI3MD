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

This module provides the validation functions for the xTB skill,
which validate the input config for xTB computation,
including the calculation method settings and computing environment settings.

"""

import os


from .config import SkillConfig, Calculation, Environment
from .method import AVAILABLE_METHOD
from .method import PROFILE_METHOD_MAP, method_supports_pbc
from .reference import AVAILABLE_PROFILES


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for xTB computation.

    Bridges between the nested SkillConfig and the flat-dict
    validation functions: extracts a work dict from submodels,
    runs validation, then writes results back.

    Parameters
    ----------
    config : SkillConfig
        Normalized config model.

    Returns
    -------
    SkillConfig
        Validated config model.
    """
    # Build a flat work dict from the submodels that validators touch
    work = {}
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    # Validate calculation method settings
    work = validate_method(work)

    # Validate system settings (embedding vs pbc compatibility)
    work = validate_system(work)

    # Validate computing environment settings
    work = validate_environment(work)

    # Write results back to the appropriate submodels
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'calculation': config.calculation.model_copy(update=calc_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


def validate_method(data: dict) -> dict:
    """Verify that the profile and method are valid and compatible.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with validated fields.
    """
    profile = data['profile'].lower()
    method = data['method'].lower()

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    data['profile'] = profile
    data['method'] = method

    # Check profile-method compatibility
    expected_method = PROFILE_METHOD_MAP.get(profile)
    if expected_method and expected_method != method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but method '{method}' was specified."
        )

    # Check method capability for the profile's task
    task = _profile_to_task(profile)
    if not AVAILABLE_METHOD[method].get(task, False):
        raise ValueError(
            f"Method '{method}' does not support task '{task}' "
            f"(required by profile '{profile}')."
        )

    return data


def validate_system(data: dict) -> dict:
    """Verify embedding and periodic boundary condition compatibility.

    Point-charge embedding and PBC cannot work together, and only
    GFN-FF supports periodic boundary conditions.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration.
    """
    method = data['method']
    point_charges = data.get('point_charges', [])
    pbc = data.get('pbc', [])

    has_pbc = any(int(x) != 0 for x in pbc)

    if has_pbc and len(point_charges) > 0:
        raise ValueError("xTB: charge embedding and pbc cannot work together!")

    if has_pbc and not method_supports_pbc(method):
        raise ValueError(f"xTB: method '{method}' does not support pbc!")

    return data


def validate_environment(data: dict) -> dict:
    """Verify the computing environment settings.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with validated environment.
    """
    xtb_path = data.get('xtb_path', '')
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')

    # Check xtb executable availability (directory or binary path)
    if xtb_path:
        if os.path.isdir(xtb_path):
            binary = os.path.join(xtb_path, 'bin', 'xtb')
            if not os.path.isfile(binary):
                raise ValueError(f"xtb executable not found at: {binary}")
        elif not os.path.isfile(xtb_path):
            raise ValueError(f"xtb path does not exist: {xtb_path}")

    # The thread count is not checked here (#306): the machine this run
    # lands on is not the one that parsed the input, and oversubscription
    # is allowed, so the framework does the arithmetic and warns instead
    # of failing the job (workflow/dispatch/scheduler.py).

    # Check scratch path
    if scratch_path:
        scratch_path = os.path.expanduser(scratch_path)
        if not os.path.exists(scratch_path):
            try:
                os.makedirs(scratch_path)
            except Exception as e:
                raise ValueError(f"Failed to create scratch path {scratch_path}: {e}")

    return data


def _profile_to_task(profile: str) -> str:
    """Extract the task name from a profile string.

    The method part may contain underscores (e.g. gfn_ff), so the
    task is always the last underscore-separated token.

    Parameters
    ----------
    profile : str
        Profile name like 'gfn2_energy' or 'gfn_ff_gradient'.

    Returns
    -------
    str
        Task name like 'energy' or 'gradient'.
    """
    parts = profile.rsplit('_', 1)
    if len(parts) == 2:
        return parts[1]
    return profile
