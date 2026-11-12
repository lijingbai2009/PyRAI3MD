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

This module defines the available methods, tasks, and profile mappings.
This is the single source of truth for method-task compatibility.

QDM methods select one of the analytic models (the Tully set or the
Landau-Zener linear crossing model); each method exposes four
profiles:

    {method}_energy     energy only
    {method}_gradient   energy + gradient (no NAC vectors)
    {method}_nac        energy + gradient + NAC (AIMD superset)
    {method}_reference  full split-operator quantum dynamics

"""

_METHODS = ('tully1', 'tully2', 'tully3', 'lz1', 'lz2', 'lz3')
_SUFFIXES = ('_energy', '_gradient', '_nac', '_reference')

# Profile to method mapping
PROFILE_METHOD_MAP = {
    f'{method}{suffix}': method
    for method in _METHODS for suffix in _SUFFIXES
}

# Profile to step chain mapping (documentation/diagnostics only; the
# QDM backend dispatches directly to the calculator or the reference
# solver without a compiled workflow graph).
PROFILE_STEP_MAP = {
    f'{method}{suffix}': (
        ['model', 'properties', 'results']
        if suffix != '_reference'
        else ['model', 'reference', 'results']
    )
    for method in _METHODS for suffix in _SUFFIXES
}

# Mapping from the normalized jobtype token set to profile suffix.
# The framework spells capability tokens in a fixed order, but agent
# flows may hand over subsets or permuted forms; resolve_profile
# sorts and deduplicates the tokens before this lookup so the
# mapping is order-independent (same convention as the orca skill).
_JOBERTYPE_SUFFIX_MAP = {
    'energy':                    '_energy',
    'energy gradient':           '_gradient',
    'nac':                       '_nac',
    'energy nac':                '_nac',
    'energy gradient nac':       '_nac',
    'gradient nac':              '_nac',
}

# Jobtypes that would ask this skill to own the iteration. These analytic
# models propagate nothing themselves under a skill driver except 'md',
# which is one backend call ('{method}_reference'); an optimize or
# surface-hopping request has no profile here at all, so it is refused by
# name instead of falling back to a single point.
_UNSERVED_ITERATION_JOBTYPES = frozenset({'optimize', 'surface_hopping'})


def resolve_profile(method, jobtype, driver='main'):
    """Derive the profile from method, jobtype and driver.

    When the profile is not explicitly provided, this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype tokens.

    Driver semantics (aligned with framework control.driver):
      - driver='main': the main framework performs the dynamics and
        asks the skill for per-step properties. jobtype 'md' maps to
        the '{method}_nac' superset profile (E/G/NAC).
      - driver='skill': the skill performs the dynamics internally.
        jobtype 'md' maps to '{method}_reference' (full split-operator
        quantum propagation in one backend call).

    Parameters
    ----------
    method : str
        Analytic model name ('tully1', 'tully2', 'tully3', 'lz1',
        'lz2', 'lz3').
    jobtype : str
        Job type token string (e.g. 'energy', 'energy gradient',
        'energy gradient nac', 'md').
    driver : str, optional
        Dynamics driver ('main' or 'skill'); affects 'md' and refuses
        an iteration jobtype this skill has no profile for.

    Returns
    -------
    str
        Profile name (e.g. 'tully1_nac', 'tully1_reference').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized, or if an iteration
        jobtype other than 'md' is asked under a skill driver.
    """
    method = method.lower()
    jobtype = jobtype.lower().strip()
    driver = (driver or 'main').lower().strip()

    if method not in _METHODS:
        raise ValueError(
            f"Cannot resolve profile: unknown method '{method}'; "
            f"available: {sorted(_METHODS)}")

    if driver != 'main' and _UNSERVED_ITERATION_JOBTYPES & set(jobtype.split()):
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' under "
            f"driver='{driver}' asks this skill to own the iteration, and "
            "these analytic models do not. Run the iteration on the "
            "framework (driver: main), which resolves the workflow from the "
            "jobtype and asks for one step at a time, or ask this skill for a "
            "single evaluation (jobtype 'energy' or 'energy gradient')"
        )

    # Canonicalize composable capability tokens (order/repeat safe).
    if jobtype == 'md':
        suffix = '_reference' if driver == 'skill' else '_nac'
    else:
        if ' ' in jobtype:
            jobtype = ' '.join(sorted(set(jobtype.split())))
        suffix = _JOBERTYPE_SUFFIX_MAP.get(jobtype)

    if suffix is None:
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' does not map "
            f"to a known profile suffix. "
            f"Available: {sorted(_JOBERTYPE_SUFFIX_MAP) + ['md']}"
        )

    profile = f'{method}{suffix}'
    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(
            f"Resolved profile '{profile}' (method={method}, "
            f"jobtype={jobtype}) is not defined in PROFILE_METHOD_MAP"
        )

    return profile
