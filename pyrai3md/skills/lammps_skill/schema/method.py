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

This module defines the available methods and their capabilities
for the LAMMPS skill. This is the single source of truth for
method-task compatibility.

"""

# Supported methods and their capabilities.
# LAMMPS is a classical molecular mechanics program. The force field
# itself is supplied by the user through the LAMMPS input template
# and topology (data) files, so a single generic 'ff' method covers
# all force fields.
AVAILABLE_METHOD = {
    'ff': {
        'energy': True,
        'gradient': True,
        'geometry_optimization': True,
        'classical_dynamics': True,
        'roles': ['mm'],
        'supports_pbc': True,
        'description': 'Classical force field (user-provided LAMMPS input and topology)',
    },
}

# Method to multi-scale calculation role mapping.
# LAMMPS serves as the mm layer only.
METHOD_ROLE_MAP = {
    'ff': ['mm'],
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'ff_energy': 'ff',
    'ff_gradient': 'ff',
    'ff_optimize': 'ff',
    'ff_optimize_cell': 'ff',
    'ff_nve': 'ff',
    'ff_nvt': 'ff',
    'ff_npt': 'ff',
    'qmmm_field': 'ff',
    'qmmm_energy': 'ff',
}

# Profile to LAMMPS step chain mapping.
# The energy step performs a single point evaluation (run 0);
# the gradient step additionally dumps per-atom forces.
# The optimize step runs an internal minimization (minimize);
# the md step runs an internal dynamics propagation (run N).
# Both internal-iteration steps return the final geometry through
# the fin_coord channel.
PROFILE_STEP_MAP = {
    'ff_energy': ['energy'],
    'ff_gradient': ['energy', 'gradient'],
    'ff_optimize': ['optimize'],
    'ff_optimize_cell': ['optimize'],
    'ff_nve': ['md'],
    'ff_nvt': ['md'],
    'ff_npt': ['md'],
    # qmmm_field is a probe task: it still passes the single-point
    # validation gate, but its usable result is the qmmm_field block and
    # its energy/gradient arrays are probe-only zeros.
    'qmmm_field': ['energy', 'gradient'],
    'qmmm_energy': ['energy', 'gradient'],
}

# Profiles where LAMMPS moves the atoms itself (minimize / run N). A
# geometric constraint has to be applied by whatever integrates the
# positions, so the shake fix is only consistent here; the remaining
# profiles hand one force evaluation per step to the framework
# integrator, which propagates and constrains on its own account
# through the framework constraint block.
ENGINE_INTEGRATED_PROFILES = ('ff_optimize', 'ff_optimize_cell',
                                'ff_nve', 'ff_nvt', 'ff_npt')

# Mapping from multi-token jobtype string to profile task suffix.
# Used by resolve_profile to derive the profile from method + jobtype
# when profile is not explicitly provided (e.g. classic input flow).
# Note: 'md' keeps the '_gradient' suffix because framework-driven
# MD queries the mm layer per step for energy and gradient; the
# standalone MD profiles (ff_nve/ff_nvt/ff_npt) are selected via an
# explicit profile.
_JOBBERTYPE_SUFFIX_MAP = {
    'energy':           '_energy',
    'gradient':         '_gradient',
    'energy gradient':  '_gradient',
    'md':               '_gradient',
    'optimize':         '_optimize',
    # The QM/MM tasks are selected by the field/qmmm capability tokens
    # the plan appends to the skill jobtype (e.g. 'energy gradient field'
    # and 'energy gradient qmmm'); a bare energy/gradient jobtype stays
    # on the ordinary ff_* profiles.
}


def resolve_profile(method, jobtype):
    """Derive the profile from method and jobtype.

    When the profile is not explicitly provided (e.g. the classic
    input flow writes jobtype but not profile), this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype.

    Parameters
    ----------
    method : str
        Computational method ('ff').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient').

    Returns
    -------
    str
        Profile name (e.g. 'ff_energy', 'ff_gradient').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized.
    """
    from .reference import AVAILABLE_PROFILES

    method = method.lower()
    jobtype = jobtype.lower().strip()

    if method not in AVAILABLE_METHOD:
        raise ValueError(
            f"Cannot resolve profile: unknown method '{method}'"
        )

    # QM/MM task duties travel as capability tokens in the skill jobtype.
    # The framework's plan adds one of them; the bare energy/gradient
    # jobtype keeps the ordinary ff_* profiles. Field wins over qmmm so
    # a malformed jobtype carrying both is deterministic.
    tokens = set(jobtype.split())
    for token, qmmm_profile in (('field', 'qmmm_field'),
                                ('qmmm', 'qmmm_energy')):
        if token in tokens:
            if qmmm_profile not in AVAILABLE_PROFILES:
                raise ValueError(
                    f"Resolved QM/MM profile '{qmmm_profile}' (from "
                    f"jobtype token '{token}') is not available.")
            return qmmm_profile

    suffix = _JOBBERTYPE_SUFFIX_MAP.get(jobtype)
    if suffix is None:
        raise ValueError(
            f"Cannot resolve profile for jobtype '{jobtype}'. "
            f"Known mappings: {sorted(_JOBBERTYPE_SUFFIX_MAP)}"
        )

    profile = f'{method}{suffix}'

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(
            f"Resolved profile '{profile}' (from method='{method}', "
            f"jobtype='{jobtype}') is not available. "
            f"Available profiles: {AVAILABLE_PROFILES}"
        )

    return profile
