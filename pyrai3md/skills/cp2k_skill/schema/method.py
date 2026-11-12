"""
======================================================================
This is a standalone CP2K calculation skill
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
for the CP2K skill. This is the single source of truth for
method-task compatibility and required CP2K input steps.

Only dft and tddft methods are supported. The legacy sf_tddft
method is intentionally not supported.

"""

# Supported electronic structure methods and their capabilities.
# Each method defines which tasks it supports and which CP2K input
# steps are needed.
AVAILABLE_METHOD = {
    'dft': {
        'energy': True,
        'gradient': True,
        'dipole': False,
        'transition_dipole': False,
        'nac': False,
        'soc_coupling': False,
        'geometry_optimization': True,
        'abinitio_dynamics': True,
        'frequency': False,
        'supports_pbc': True,
        'steps': ['energy', 'gradient', 'optimize', 'md'],
        'description': 'Density Functional Theory (ground state)',
    },
    'tddft': {
        'energy': True,
        'gradient': True,
        'dipole': False,
        'transition_dipole': True,
        'nac': False,
        'soc_coupling': False,
        'geometry_optimization': True,
        'abinitio_dynamics': True,
        'frequency': False,
        'supports_pbc': True,
        'steps': ['energy', 'tddft', 'gradient', 'transition_dipole',
                  'optimize', 'md'],
        'description': 'Time-Dependent Density Functional Theory (excited states)',
    },
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'dft_energy': 'dft',
    'dft_gradient': 'dft',
    'dft_opt_min': 'dft',
    'dft_md': 'dft',
    'tddft_energy': 'tddft',
    'tddft_gradient': 'tddft',
    'tddft_transition_dipole': 'tddft',
    'tddft_opt_min': 'tddft',
    'tddft_md': 'tddft',
}

# Profile to CP2K input step chain mapping
PROFILE_STEP_MAP = {
    'dft_energy': ['energy'],
    'dft_gradient': ['energy', 'gradient'],
    'dft_opt_min': ['optimize'],
    'dft_md': ['md'],
    'tddft_energy': ['energy', 'tddft'],
    'tddft_gradient': ['energy', 'tddft', 'gradient'],
    'tddft_transition_dipole': ['energy', 'tddft', 'transition_dipole'],
    'tddft_opt_min': ['tddft', 'optimize'],
    'tddft_md': ['tddft', 'md'],
}

# Mapping from jobtype string to profile suffix.
# Used to derive the profile from method + jobtype when profile is
# not explicitly provided (e.g. classic input flow).
#
# Multi-token entries mirror the framework's canonical jobtype
# spelling (JOBTYPE_TOKEN_ORDER: energy, gradient, nac, soc,
# dipole, transition, frequency) emitted by the planner and the
# certification converter. resolve_profile sorts the tokens before
# the lookup, so every multi-word key is stored in alphabetical
# token order.
_JOBERTYPE_SUFFIX_MAP = {
    'energy':              '_energy',
    'gradient':            '_gradient',
    'energy gradient':     '_gradient',
    'optimize':            '_opt_min',
    'absorption':          '_transition_dipole',
    'transition':          '_transition_dipole',
    'energy transition':   '_transition_dipole',
    'absorption energy':   '_transition_dipole',
    # Molecular dynamics: framework AIMD (driver=main) steps through
    # the gradient profile; the standalone &MOTION MD profiles
    # (dft_md/tddft_md) are selected via an explicit profile when
    # the skill drives the run (driver=skill).
    'md':                  '_gradient',
}


def resolve_profile(method, jobtype):
    """Derive the profile from method and jobtype.

    When the profile is not explicitly provided, this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype.

    Parameters
    ----------
    method : str
        Computational method ('dft' or 'tddft').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient', 'optimize').
        The framework normalizes aliases (hess/hessian/freq/frequence
        -> frequency) before reaching this function.

    Returns
    -------
    str
        Profile name (e.g. 'dft_energy', 'tddft_gradient').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized, or the resolved
        profile is incompatible with the method (e.g. dft + soc).
    """
    method = method.lower()
    # Canonicalize the jobtype tokens: the framework spells them in a
    # fixed order (energy, gradient, nac, soc, dipole, transition,
    # frequency), but agent flows may hand over subsets or permuted
    # forms. Sorting makes the lookup order-independent; 'optimize'
    # stays untouched since it carries no composable tokens.
    jobtype = jobtype.lower().strip()
    if ' ' in jobtype:
        jobtype = ' '.join(sorted(set(jobtype.split())))

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Cannot resolve profile: unknown method '{method}'")

    suffix = _JOBERTYPE_SUFFIX_MAP.get(jobtype)
    if suffix is None:
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' does not map "
            f"to a known profile suffix. Available: {sorted(_JOBERTYPE_SUFFIX_MAP)}"
        )

    profile = f'{method}{suffix}'
    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(
            f"Resolved profile '{profile}' (method={method}, "
            f"jobtype={jobtype}) is not supported by this method"
        )

    return profile
