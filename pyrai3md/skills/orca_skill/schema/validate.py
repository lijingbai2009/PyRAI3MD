"""
======================================================================
This is a standalone ORCA calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the validation functions for the ORCA skill,
which validate the input config for ORCA computation,
including the calculation method settings and computing environment settings.

"""

import os


from .config import SkillConfig, Calculation, Environment
from .method import AVAILABLE_METHOD
from .method import PROFILE_METHOD_MAP
from .reference import AVAILABLE_BASIS
from .reference import AVAILABLE_PROFILES
from .reference import DISPERSION_KEYWORD


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for ORCA computation.

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
            "ORCA is a molecular (non-periodic) backend and does not "
            "support periodic boundary conditions (pbc=%r); disable pbc "
            "or use a periodic backend (cp2k / xtb / lammps)." % (_pbc,))

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
    """Verify that the profile, method, basis, and keywords are valid.

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
    functional = data['functional'].lower()
    basis = data['basis'].lower().replace(' ', '')
    num_freq = data['num_freq']
    density_fitting = data['density_fitting']
    aux_basis = str(data['aux_basis']).strip()
    dispersion_correction = str(
        data['dispersion_correction']).strip().lower()
    grid_level = data['grid_level']

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    if basis not in AVAILABLE_BASIS:
        raise ValueError(f"Unsupported basis: {basis}")

    # num_freq must be a boolean keyword (False = AnFreq, True = NumFreq)
    if not isinstance(num_freq, bool):
        raise ValueError(f"num_freq must be a boolean (False=AnFreq, True=NumFreq), got {num_freq!r}")

    # Density fitting writes RIJCOSX plus the auxiliary basis after
    # the basis set; dispersion and grid keywords are optional.
    if not isinstance(density_fitting, bool):
        raise ValueError(
            f"density_fitting must be a boolean, got {density_fitting!r}")

    if dispersion_correction in ('', 'none'):
        dispersion_correction = ''
    elif dispersion_correction not in DISPERSION_KEYWORD:
        raise ValueError(
            f"Unsupported dispersion correction: "
            f"{dispersion_correction!r}. Supported: "
            f"{sorted(DISPERSION_KEYWORD)} or 'none'.")

    if grid_level not in (0, 1, 2, 3):
        raise ValueError(
            f"grid_level must be 0 (ORCA default) or 1-3 (DEFGRID1-3), "
            f"got {grid_level!r}")

    data['profile'] = profile
    data['method'] = method
    data['functional'] = functional
    data['basis'] = basis
    data['aux_basis'] = aux_basis
    data['dispersion_correction'] = dispersion_correction
    data['grid_level'] = grid_level

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

    # A TDDFT run that names no excited state is a contradiction between
    # the method and the state manifold, and ORCA refuses the input that
    # follows from it at scan time ("IROOT must be smaller than or equal
    # to NROOTS", main_input_check.cpp) because `%tddft nroots 0` asks for
    # a root list that cannot exist. Name it here instead: degrading to a
    # ground-state run would answer a question the caller did not ask.
    if method == 'tddft' and int(data.get('tddft_nroots', 0)) < 1:
        raise ValueError(
            "method 'tddft' was given a state manifold with no excited "
            "state (tddft_nroots=%d); raise molecule.ci to name the "
            "excited roots, or use method 'dft' for a ground-state run."
            % int(data.get('tddft_nroots', 0)))

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
    orca_path = data.get('orca_path', '')
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')

    # Check ORCA installation directory availability
    if orca_path and not os.path.isdir(orca_path):
        raise ValueError(f"ORCA path does not exist: {orca_path}")

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


# Profile suffix to AVAILABLE_METHOD task key mapping.
# Most profiles use simple suffix extraction (e.g. dft_energy -> energy).
# Profiles where the suffix does not match the task key need explicit mapping.
_PROFILE_TASK_MAP = {
    'dft_soc': 'soc_coupling',
    'tddft_soc': 'soc_coupling',
    'dft_opt_min': 'geometry_optimization',
    'dft_opt_ts': 'geometry_optimization',
    'tddft_opt_min': 'geometry_optimization',
    'tddft_opt_ts': 'geometry_optimization',
}


def _profile_to_task(profile: str) -> str:
    """Extract the task name from a profile string.

    Parameters
    ----------
    profile : str
        Profile name like 'dft_energy'.

    Returns
    -------
    str
        Task name like 'energy'.
    """
    if profile in _PROFILE_TASK_MAP:
        return _PROFILE_TASK_MAP[profile]
    parts = profile.split('_', 1)
    if len(parts) == 2:
        return parts[1]
    return profile
