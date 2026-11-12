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

This module defines the available methods and their capabilities
for the xTB skill. This is the single source of truth for
method-task compatibility and xTB command-line flags.

"""

# Supported methods and their capabilities.
# GFN2/GFN1/GFN0 are semiempirical tight-binding QM methods,
# GFN-FF is a general force field (MM) method.
# Each method defines which tasks it supports and the xTB CLI flag.
AVAILABLE_METHOD = {
    'gfn2': {
        'energy': True,
        'gradient': True,
        'cli_flag': '--gfn 2',
        'roles': ['qm', 'qm2'],
        'supports_pbc': False,
        'description': 'Geometry, Frequency, Noncovalent interactions xTB (2nd parameterization)',
    },
    'gfn1': {
        'energy': True,
        'gradient': True,
        'cli_flag': '--gfn 1',
        'roles': ['qm', 'qm2'],
        'supports_pbc': False,
        'description': 'Geometry, Frequency, Noncovalent interactions xTB (1st parameterization)',
    },
    'gfn0': {
        'energy': True,
        'gradient': True,
        'cli_flag': '--gfn 0',
        'roles': ['qm', 'qm2'],
        'supports_pbc': False,
        'description': 'Geometry, Frequency, Noncovalent interactions xTB (EEQ charges only)',
    },
    'gfn_ff': {
        'energy': True,
        'gradient': True,
        'cli_flag': '--gfnff',
        'roles': ['mm'],
        'supports_pbc': True,
        'description': 'General Force Field of the GFN family',
    },
}

# Single source for the periodic-cell capability question, so the schema
# check and the geometry writer cannot drift apart. The check has to be
# consulted at the writer as well: the schema sees `pbc` only when an input
# declares it on the skill itself, while a framework run carries the cell on
# the system and hands it to the backend per evaluation (backend.py:205-215),
# so nothing else re-asks the question once the cell arrives.
def method_supports_pbc(method: str) -> bool:
    """Return whether this method can run inside a periodic cell.

    Parameters
    ----------
    method : str
        Method name (gfn2, gfn1, gfn0, gfn_ff).

    Returns
    -------
    bool
        True for the methods `AVAILABLE_METHOD` marks periodic.
    """
    return bool(AVAILABLE_METHOD.get(method, {}).get('supports_pbc', False))


# Method to multi-scale calculation role mapping.
# GFN-x methods serve as qm/qm2; GFN-FF serves as mm.
METHOD_ROLE_MAP = {
    'gfn2': ['qm', 'qm2'],
    'gfn1': ['qm', 'qm2'],
    'gfn0': ['qm', 'qm2'],
    'gfn_ff': ['mm'],
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'gfn2_energy': 'gfn2',
    'gfn2_gradient': 'gfn2',
    'gfn1_energy': 'gfn1',
    'gfn1_gradient': 'gfn1',
    'gfn0_energy': 'gfn0',
    'gfn0_gradient': 'gfn0',
    'gfn_ff_energy': 'gfn_ff',
    'gfn_ff_gradient': 'gfn_ff',
}

# Profile to xTB step chain mapping.
# The energy step performs a single point calculation;
# the gradient step additionally writes the engrad file (--grad).
PROFILE_STEP_MAP = {
    'gfn2_energy': ['energy'],
    'gfn2_gradient': ['energy', 'gradient'],
    'gfn1_energy': ['energy'],
    'gfn1_gradient': ['energy', 'gradient'],
    'gfn0_energy': ['energy'],
    'gfn0_gradient': ['energy', 'gradient'],
    'gfn_ff_energy': ['energy'],
    'gfn_ff_gradient': ['energy', 'gradient'],
}

# Mapping from multi-token jobtype string to profile task suffix.
# Used by resolve_profile to derive the profile from method + jobtype
# when profile is not explicitly provided (e.g. classic input flow).
_JOBBERTYPE_SUFFIX_MAP = {
    'energy':           '_energy',
    'gradient':         '_gradient',
    'energy gradient':  '_gradient',
    'md':               '_gradient',
}

# Jobtypes that ask whoever answers them to own the iteration. The map above
# spells 'md' as a gradient, which is the framework's per-step ask; xtb
# propagates nothing itself, so on the route that hands the steps over
# (driver != 'main') not one of these can be served and the profile has to be
# refused by name rather than fall back to a single point.
_ITERATION_JOBTYPES = frozenset({'md', 'optimize', 'surface_hopping'})


def resolve_profile(method, jobtype, driver='main'):
    """Derive the profile from method and jobtype.

    When the profile is not explicitly provided (e.g. the classic
    input flow writes jobtype but not profile), this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype.

    Parameters
    ----------
    method : str
        Computational method ('gfn2', 'gfn1', 'gfn0', or 'gfn_ff').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient').
    driver : str, optional
        Execution driver (framework `control.driver`). A value other
        than 'main' hands the steps to this skill, which serves
        per-step properties only, so an iteration jobtype is refused.

    Returns
    -------
    str
        Profile name (e.g. 'gfn2_energy', 'gfn_ff_gradient').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized, or if an iteration
        jobtype is asked under a driver that would hand over the steps.
    """
    from .reference import AVAILABLE_PROFILES

    method = method.lower()
    jobtype = jobtype.lower().strip()
    driver = (driver or 'main').lower().strip()

    if method not in AVAILABLE_METHOD:
        raise ValueError(
            f"Cannot resolve profile: unknown method '{method}'"
        )

    if driver != 'main' and _ITERATION_JOBTYPES & set(jobtype.split()):
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' under "
            f"driver='{driver}' asks this skill to own the iteration, and "
            "xtb answers per-step properties only. Run the iteration on the "
            "framework (driver: main), which resolves the workflow from the "
            "jobtype and asks for one step at a time, or ask this skill for a "
            "single evaluation (jobtype 'energy' or 'gradient')"
        )

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
