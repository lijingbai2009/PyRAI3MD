"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the validation functions for the BAGEL skill,
which validate the input config for BAGEL computation,
including the calculation method settings and computing environment settings.

"""

import os


from .config import SkillConfig, Calculation, Environment
from .method import AVAILABLE_METHOD
from .method import PROFILE_METHOD_MAP
from .reference import AVAILABLE_BASIS, AVAILABLE_DF_BASIS
from .reference import AVAILABLE_PROFILES


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for BAGEL computation.

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
            "BAGEL is a molecular (non-periodic) backend and does not "
            "support periodic boundary conditions (pbc=%r); disable pbc "
            "or use a periodic backend (cp2k / xtb / lammps)." % (_pbc,))

    # A triplet row is refused before the input is written, because the input has
    # no line that could ask for one. `state_spin` labels a row per state while
    # the writer emits no spin or multiplicity keyword at all - neither in
    # &molecule nor in the &mcscf/casscf block - so every root BAGEL solves comes
    # from one spin manifold however the list was written. A run that paired a
    # singlet row with a triplet row would be handed two roots of one spin, and
    # the row count it checks would still agree.
    _spins = [int(x) for x in (getattr(config.system, 'state_spin', None)
                               or [])]
    if len(set(_spins)) > 1:
        raise ValueError(
            "bagel: state_spin %r names more than one spin, and the input this "
            "skill writes carries no spin keyword, so all the roots BAGEL solves "
            "come from one manifold and a row labelled otherwise would be a "
            "second root of the spin that was asked for. Keep one spin per run "
            "for BAGEL, or take a singlet/triplet pair from a backend that can "
            "request both (cp2k's TDDFPT with RKS_TRIPLETS, Molcas' dual "
            "RASSCF)." % (_spins,))

    # Build a flat work dict from the submodels that validators touch
    work = {}
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    # Validate calculation method settings
    work = validate_method(work)

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
    df_basis = str(data.get('df_basis') or '').strip().lower()

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    if basis not in AVAILABLE_BASIS:
        raise ValueError(f"Unsupported basis: {basis}")

    # An empty df_basis is resolved later by similarity matching in
    # the input builder; an explicit value must be a supported set
    if df_basis and df_basis not in AVAILABLE_DF_BASIS:
        raise ValueError(f"Unsupported df_basis: {df_basis}")

    data['profile'] = profile
    data['method'] = method
    data['basis'] = basis
    data['df_basis'] = df_basis

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
    bagel_path = data.get('bagel_path', '')
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')

    # Normalize the MPI launcher executable; an empty value falls back to
    # the default launcher, which is the one name every MPI-capable skill
    # in this package uses (#305).
    mpiexe = str(data.get('mpiexe') or '').strip().lower()
    data['mpiexe'] = mpiexe or 'mpirun'

    # Check bagel executable availability
    if bagel_path and not os.path.isdir(bagel_path):
        raise ValueError(f"BAGEL path does not exist: {bagel_path}")

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

    Parameters
    ----------
    profile : str
        Profile name like 'casscf_energy'.

    Returns
    -------
    str
        Task name like 'energy'.
    """
    parts = profile.split('_', 1)
    if len(parts) == 2:
        return parts[1]
    return profile
