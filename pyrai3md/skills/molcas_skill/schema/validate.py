"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the validation functions for the Molcas skill,
which validate the input config for Molcas computation,
including the calculation method settings and computing environment settings.

"""

import os


from .config import SkillConfig, Calculation, Environment
from .method import AVAILABLE_METHOD
from .method import PROFILE_METHOD_MAP
from .reference import AVAILABLE_BASIS
from .reference import AVAILABLE_PROFILES


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for Molcas computation.

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
            "Molcas is a molecular (non-periodic) backend and does not "
            "support periodic boundary conditions (pbc=%r); disable pbc "
            "or use a periodic backend (cp2k / xtb / lammps)." % (_pbc,))

    # Build a flat work dict from the submodels that validators touch
    work = {}
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())
    work.update(config.system.model_dump())

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
    """Verify that the profile, method, and basis are valid.

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
    basis = data['basis'].lower().replace(' ', '')

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    if '.' in basis:
        # Per-element combination, e.g. 'H.ano-rcc-vdzp, C.ano-rcc-vdzp':
        # element and basis joined by '.', elements separated by ','.
        # Validate the basis part of every component individually.
        for component in basis.split(','):
            element, sep, component_basis = component.partition('.')
            if not sep or not element or component_basis not in AVAILABLE_BASIS:
                raise ValueError(f"Unsupported basis component: {component}")
    elif basis not in AVAILABLE_BASIS:
        raise ValueError(f"Unsupported basis: {basis}")

    data['profile'] = profile
    data['method'] = method
    data['basis'] = basis

    # Check profile-method compatibility
    expected_method = PROFILE_METHOD_MAP.get(profile)
    if expected_method and expected_method != method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but method '{method}' was specified."
        )

    # Check method capability for the profile's task. Generic property
# profiles combine several tokens, so validate each requested token.
    if profile.endswith('_properties'):
        requested = set(str(data.get('jobtype', '') or '').lower().split())
        # transition/dipole are request modifiers, not capabilities.
        requested.discard('transition')
        requested.discard('dipole')
        task_key = {
            'energy': 'energy',
            'gradient': 'gradient',
            'nac': 'nac',
            'soc': 'soc_coupling',
            'absorption': 'absorption_spectrum',
        }
        unknown = requested - set(task_key)
        if unknown:
            raise ValueError(
                "Generic profile '%s' does not accept task(s) %s"
                % (profile, sorted(unknown)))
        for token in sorted(requested):
            if not AVAILABLE_METHOD[method].get(task_key[token], False):
                raise ValueError(
                    f"Method '{method}' does not support task "
                    f"'{token}' (requested by '{profile}').")
    else:
        task = _profile_to_task(profile)
        if not AVAILABLE_METHOD[method].get(task, False):
            raise ValueError(
                f"Method '{method}' does not support task '{task}' "
                f"(required by profile '{profile}')."
            )

    _validate_embedding_for_profile(data)
    return data


def _validate_embedding_for_profile(data: dict) -> None:
    """Validate ESPF embedding/read_charge choices for this profile."""
    embedding = str(data.get('embedding', 'none') or 'none').strip().lower()
    read_charge = str(data.get('read_charge', 'none') or 'none').strip()
    if embedding not in ('none', 'charge', 'field'):
        raise ValueError(
            "embedding must be none, charge, or field; got %r"
            % (data.get('embedding'),))
    # A file stands on its own: its declaration carries the representation,
    # so `read_charge` needs no `embedding` next to it. That is the
    # single-theory ESPF case, where the file is the whole switch.
    if embedding == 'none' and read_charge == 'none':
        return
    method = str(data.get('method', '') or '').lower()
    if method not in ('casscf', 'caspt2'):
        raise ValueError(
            "ESPF embedding is supported only for CASSCF/CASPT2; "
            "method is %r." % (method,))
    iterative = {
        'casscf_geometry_optimization',
        'casscf_abinitio_dynamics',
        'casscf_surface_hopping',
        'casscf_frequency',
        'caspt2_frequency',
    }
    profile = str(data.get('profile', '') or '').lower()
    if profile in iterative:
        raise ValueError(
            "ESPF embedding is not supported for iterative profile "
            "'%s' in this round." % (profile,))


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
    molcas_path = data.get('molcas_path', '')
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')

    # Check molcas executable availability
    if molcas_path and not os.path.isdir(molcas_path):
        raise ValueError(f"Molcas path does not exist: {molcas_path}")

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
# Most profiles use simple suffix extraction (e.g. casscf_energy -> energy).
# Profiles where the suffix does not match the task key need explicit mapping.
_PROFILE_TASK_MAP = {
    'casscf_soc': 'soc_coupling',
    'casscf_absorption': 'absorption_spectrum',
    'caspt2_absorption': 'absorption_spectrum',
}


def _profile_to_task(profile: str) -> str:
    """Extract the task name from a profile string.

    Parameters
    ----------
    profile : str
        Profile name like 'casscf_energy'.

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
