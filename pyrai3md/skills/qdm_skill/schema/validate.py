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

This module provides the validation functions for the QDM skill,
which validate the input config for the analytic model computation,
including the calculation method settings and computing environment settings.

"""


import os

from ..service.models import MODEL_REGISTRY
from .config import SkillConfig, Calculation, Environment
from .method import PROFILE_METHOD_MAP


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for QDM computation.

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
    # Machine-learning potential backend: periodic support is not
    # declared until periodic neighborhoods and labels are validated
    # separately, so reject periodic runs explicitly instead of
    # silently ignoring the cell.
    _pbc = [int(x) for x in (getattr(config.system, 'pbc', None) or [])]
    if any(_pbc):
        raise ValueError(
            "QDM does not support periodic boundary conditions in this "
            "version (pbc=%r); disable pbc or use a periodic backend "
            "(cp2k / xtb / lammps)." % (_pbc,))

    # Build a flat work dict from the submodels that validators touch
    work = {}
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    # Validate calculation method settings
    work = validate_method(work)

    # Validate computing environment settings
    work = validate_environment(work)

    # Write results back to the corresponding submodels
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'calculation': config.calculation.model_copy(update=calc_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


def validate_method(data: dict) -> dict:
    # Verify that the profile-specific requirements are met
    profile = data['profile'].lower()
    method = data['method'].lower()

    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(f"Unsupported profile: {profile}")

    # Verify that the method matches the profile
    expected_method = PROFILE_METHOD_MAP[profile]
    if method != expected_method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but got '{method}'"
        )

    if method not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported method: {method}; "
            f"available: {sorted(MODEL_REGISTRY)}")

    # Verify the model and propagation parameters
    if data['model_mass'] <= 0:
        raise ValueError("model_mass must be positive.")

    for name in ('tully_a', 'tully_b', 'tully_c', 'tully_d',
                 'tully_e'):
        value = data[name]
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided.")

    if profile.endswith('_reference'):
        if data['ref_ngrid'] < 2:
            raise ValueError("ref_ngrid must be at least 2.")
        if data['ref_dt'] <= 0:
            raise ValueError("ref_dt must be positive.")
        if data['ref_nsteps'] < 1:
            raise ValueError("ref_nsteps must be at least 1.")
        if data['ref_save_every'] < 1:
            raise ValueError("ref_save_every must be at least 1.")
        if data['wavepacket_sigma'] <= 0:
            raise ValueError("wavepacket_sigma must be positive.")
        xmin, xmax = data['ref_xmin'], data['ref_xmax']
        if xmin is not None and xmax is not None and xmin >= xmax:
            raise ValueError("ref_xmin must be smaller than ref_xmax.")

    data['profile'] = profile
    data['method'] = method

    return data


def validate_environment(data: dict) -> dict:
    # Verify that the computing environment settings are valid
    scratch_path = os.path.expanduser(data['scratch_path'])
    # The thread count is not checked here (#306): the machine this run
    # lands on is not the one that parsed the input, and oversubscription
    # is allowed, so the framework does the arithmetic and warns instead
    # of failing the job (workflow/dispatch/scheduler.py).


    # check scratch path
    if scratch_path and not os.path.exists(scratch_path):
        try:
            os.makedirs(scratch_path)
        except Exception as e:
            raise ValueError(f"Failed to create scratch path {scratch_path}: {e}")

    return data
