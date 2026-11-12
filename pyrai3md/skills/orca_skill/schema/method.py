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

This module defines the available methods and their capabilities
for the ORCA skill. This is the single source of truth for
method-task compatibility and required ORCA input steps.

Only dft and tddft methods are supported. The legacy sf_tddft
method is intentionally not supported.

"""

# Supported electronic structure methods and their capabilities.
# Each method defines which tasks it supports and which ORCA input
# steps are needed.
AVAILABLE_METHOD = {
    'dft': {
        'energy': True,
        'gradient': True,
        'dipole': True,
        'transition_dipole': False,
        'nac': False,
        'soc_coupling': False,
        'geometry_optimization': True,
        'frequency': True,
        'steps': ['energy', 'gradient', 'dipole', 'optimize', 'optimize_ts', 'hessian'],
        'description': 'Density Functional Theory (ground state)',
    },
    'tddft': {
        'energy': True,
        'gradient': True,
        'dipole': False,
        'transition_dipole': True,
        'nac': True,
        'soc_coupling': True,
        'geometry_optimization': True,
        'frequency': True,
        'steps': ['energy', 'tddft', 'gradient', 'transition_dipole', 'nac', 'soc',
                  'optimize', 'optimize_ts', 'hessian'],
        'description': 'Time-Dependent Density Functional Theory (excited states)',
    },
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'dft_energy': 'dft',
    'dft_gradient': 'dft',
    'dft_dipole': 'dft',
    'dft_opt_min': 'dft',
    'dft_opt_ts': 'dft',
    'dft_frequency': 'dft',
    'tddft_energy': 'tddft',
    'tddft_gradient': 'tddft',
    'tddft_transition_dipole': 'tddft',
    'tddft_nac': 'tddft',
    'tddft_soc': 'tddft',
    'tddft_opt_min': 'tddft',
    'tddft_opt_ts': 'tddft',
    'tddft_frequency': 'tddft',
}

# Profile to ORCA input step chain mapping
PROFILE_STEP_MAP = {
    'dft_energy': ['energy'],
    'dft_gradient': ['energy', 'gradient'],
    'dft_dipole': ['energy', 'dipole'],
    'dft_opt_min': ['optimize'],
    'dft_opt_ts': ['optimize_ts'],
    # No `gradient` step: ORCA will not put EnGrad in a deck that carries a
    # Hessian keyword, so the step would be inert (see
    # service/utils.py::writes_engrad) - the frequency run answers with
    # the Hessian and nothing else.
    'dft_frequency': ['energy', 'hessian'],
    'tddft_energy': ['energy', 'tddft'],
    'tddft_gradient': ['energy', 'tddft', 'gradient'],
    'tddft_transition_dipole': ['energy', 'tddft', 'transition_dipole'],
    # A crossing search asks for `energy gradient nac`, and the state
# gradients are half of what it steps on: without the gradient step the
# deck carried neither `EnGrad` nor `irootlist`, ORCA printed no
# `CARTESIAN GRADIENT`, and the gap's gradient difference came back exactly
# zero while the search reported it as a converged-on-nothing direction.
    'tddft_nac': ['energy', 'tddft', 'gradient', 'nac'],
    # Same shape `tddft_nac` had: the compound jobtype `energy gradient soc`
# resolves here, so a request that carries `gradient` has to keep it. A
# profile whose steps drop a requested step silently removes an evaluation
# the caller is already waiting on (#446 measured exactly that on `nac`: the
# deck lost `EnGrad`/`irootlist` and the state gradients came back identical).
    'tddft_soc': ['energy', 'tddft', 'gradient', 'soc'],
    'tddft_opt_min': ['tddft', 'optimize'],
    'tddft_opt_ts': ['tddft', 'optimize_ts'],
    'tddft_frequency': ['energy', 'tddft', 'hessian'],
}

# Mapping from jobtype string to profile suffix.
# Used to derive the profile from method + jobtype when profile is
# not explicitly provided (e.g. classic input flow).
#
# Multi-token entries mirror the framework's canonical jobtype
# spelling (JOBTYPE_TOKEN_ORDER: energy, gradient, nac, soc,
# dipole, transition, frequency) emitted by the planner and the
# qualification converter. resolve_profile sorts the tokens before
# the lookup, so every multi-word key is stored in alphabetical
# token order.
_JOBERTYPE_SUFFIX_MAP = {
    'energy':              '_energy',
    'gradient':            '_gradient',
    'energy gradient':     '_gradient',
    'frequency':           '_frequency',
    'energy frequency':    '_frequency',
    'optimize':            '_opt_min',
    'optimize_ts':         '_opt_ts',
    'dipole':              '_dipole',
    'dipole energy':       '_dipole',
    'absorption':          '_transition_dipole',
    'transition':          '_transition_dipole',
    'energy transition':   '_transition_dipole',
    'absorption energy':   '_transition_dipole',
    'nac':                 '_nac',
    'energy nac':          '_nac',
    'energy gradient nac': '_nac',
    'soc':                 '_soc',
    'energy soc':          '_soc',
    # Molecular dynamics propagates on energy/gradient surfaces
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
