"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the validation functions for the PySCF skill,
which validate the input config for pyscf computation,
including the calculation method settings and computing environment settings.

"""


import os


from .config import SkillConfig, System, Calculation, Environment
from .method import PROFILE_METHOD_MAP, AVAILABLE_METHOD
from .reference import AVAILABLE_FUNCTIONAL
from .reference import AVAILABLE_BASIS
from .reference import AVAILABLE_SOLVENT


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for PySCF computation.

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
    # Molecular (non-periodic) backend: reject periodic runs explicitly
    # instead of silently ignoring the cell. The framework capability
    # gate (schema.validate) also refuses these; this is the
    # standalone/agent-entry backstop.
    _pbc = [int(x) for x in (getattr(config.system, 'pbc', None) or [])]
    if any(_pbc):
        raise ValueError(
            "PySCF is a molecular (non-periodic) backend and does not "
            "support periodic boundary conditions (pbc=%r); disable pbc "
            "or use a periodic backend (cp2k / xtb / lammps)." % (_pbc,))

    # Build a flat work dict from the submodels that validators touch
    work = {}
    work.update(config.system.model_dump())      # solvent lives in System
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    # Validate calculation method settings
    work = validate_method(work)

    # Validate computing environment settings
    work = validate_environment(work)

    # Write results back to the corresponding submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


def validate_method(data: dict) -> dict:
    # Verify that the profile-specific requirements are met
    profile = data['profile'].lower()
    method = data['method'].lower()
    functional = data['functional'].lower()
    basis = data['basis'].lower().replace('-', '')
    solvent = data['solvent'].lower()

    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(f"Unsupported profile: {profile}")

    # Verify that the method matches the profile
    expected_method = PROFILE_METHOD_MAP[profile]
    if method != expected_method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but got '{method}'"
        )

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    if functional not in AVAILABLE_FUNCTIONAL:
        raise ValueError(f"Unsupported functional: {functional}")

    if basis not in AVAILABLE_BASIS:
        raise ValueError(f"Unsupported basis: {basis}")

    if solvent not in AVAILABLE_SOLVENT:
        raise ValueError(f"Unsupported solvent: {solvent}")

    data['profile'] = profile
    data['method'] = method
    data['functional'] = functional
    data['basis'] = basis
    data['solvent'] = solvent

    return data


def validate_environment(data: dict) -> dict:
    # Verify that the computing environment settings are valid
    scratch_path = os.path.expanduser(data['scratch_path'])
    # The thread count is not checked here (#306): the machine this run
    # lands on is not the one that parsed the input, and oversubscription
    # is allowed, so the framework does the arithmetic and warns instead
    # of failing the job (workflow/dispatch/scheduler.py).

    if scratch_path and not os.path.exists(scratch_path):
        try:
            os.makedirs(scratch_path)
        except Exception as e:
            raise ValueError(f"Failed to create scratch path {scratch_path}: {e}")

    return data
