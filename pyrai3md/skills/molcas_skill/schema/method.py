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

This module defines the available methods and their capabilities
for the Molcas skill. This is the single source of truth for
method-task compatibility and required Molcas modules.

"""

# Supported electronic structure methods and their capabilities.
# Each method defines which profiles it supports and which Molcas modules are needed.
AVAILABLE_METHOD = {
    'hf': {
        'energy': True,
        'gradient': False,
        'geometry_optimization': False,
        'absorption_spectrum': False,
        'abinitio_dynamics': False,
        'soc_coupling': False,
        'frequency': True,
        'surface_hopping': False,
        'modules': ['GATEWAY', 'SEWARD', 'SCF'],
        'description': 'Hartree-Fock / SCF',
    },
    'casscf': {
        'energy': True,
        'gradient': True,
        'nac': True,
        'geometry_optimization': True,
        'abinitio_dynamics': True,
        'surface_hopping': True,
        'absorption_spectrum': True,
        'soc_coupling': True,
        'frequency': True,
        'modules': ['GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA', 'RASSI', 'SLAPAF', 'DYNAMIX', 'SURFACEHOP', 'MCKINLEY'],
        'description': 'Complete Active Space SCF',
    },
    'caspt2': {
        'energy': True,
        'gradient': True,
        'nac': True,
        'geometry_optimization': False,
        'abinitio_dynamics': False,
        'surface_hopping': False,
        'absorption_spectrum': True,
        'soc_coupling': True,
        'frequency': True,
        'modules': ['GATEWAY', 'SEWARD', 'RASSCF', 'CASPT2',
                    'ALASKA', 'RASSI', 'MCKINLEY'],
        'description': 'CAS Second-Order Perturbation Theory',
    },
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'hf_energy': 'hf',
    'casscf_energy': 'casscf',
    'casscf_gradient': 'casscf',
    'casscf_nac': 'casscf',
    'casscf_properties': 'casscf',
    'casscf_geometry_optimization': 'casscf',
    'casscf_abinitio_dynamics': 'casscf',
    'casscf_surface_hopping': 'casscf',
    'casscf_soc': 'casscf',
    'casscf_absorption': 'casscf',
    'caspt2_energy': 'caspt2',
    'caspt2_properties': 'caspt2',
    'caspt2_absorption': 'caspt2',
    'casscf_frequency': 'casscf',
    'caspt2_frequency': 'caspt2',
}

# Profile to Molcas module chain mapping
PROFILE_MODULE_MAP = {
    'hf_energy': ['GATEWAY', 'SEWARD', 'SCF'],
    'casscf_energy': ['GATEWAY', 'SEWARD', 'RASSCF'],
    'casscf_gradient': ['GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA'],
    'casscf_nac': ['GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA'],
    'casscf_properties': ['GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA', 'RASSI'],
    'casscf_geometry_optimization': ['GATEWAY', 'SEWARD', 'RASSCF', 'SLAPAF'],
    'casscf_abinitio_dynamics': ['GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA', 'DYNAMIX'],
    'casscf_surface_hopping': ['GATEWAY', 'SEWARD', 'RASSCF', 'SURFACEHOP', 'ALASKA', 'DYNAMIX'],
    'casscf_soc': ['GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA', 'RASSI'],
    'casscf_absorption': ['GATEWAY', 'SEWARD', 'RASSCF', 'RASSI'],
    'caspt2_energy': ['GATEWAY', 'SEWARD', 'RASSCF', 'CASPT2'],
    'caspt2_properties': ['GATEWAY', 'SEWARD', 'RASSCF', 'CASPT2',
                          'ALASKA', 'RASSI'],
    'caspt2_absorption': ['GATEWAY', 'SEWARD', 'RASSCF', 'CASPT2', 'RASSI'],
    'casscf_frequency': ['GATEWAY', 'SEWARD', 'RASSCF', 'MCKINLEY'],
    'caspt2_frequency': ['GATEWAY', 'SEWARD', 'RASSCF', 'CASPT2', 'MCKINLEY'],
}

# Profiles that require dual RASSCF (singlet + triplet) before RASSI
DUAL_RASSCF_PROFILES = [
    'casscf_soc',
]

# Mapping from multi-token jobtype string to profile task suffix.
# Used by resolve_profile to derive the profile from method + jobtype
# when profile is not explicitly provided (e.g. classic input flow).
# Compound jobtypes map to the most specific matching profile:
# 'energy gradient soc' -> soc (not gradient or energy).
# Keys are written in sorted token order; resolve_profile sorts the
# incoming tokens before lookup so order-independent forms such as
# 'nac energy' hit the same entry as 'energy nac'.
_JOBBERTYPE_SUFFIX_MAP = {
    'energy':           '_energy',
    'gradient':         '_gradient',
    'energy gradient':  '_gradient',
    'nac':              '_nac',
    'energy nac':       '_nac',
    'energy gradient nac': '_nac',
    'soc':              '_soc',
    'energy soc':       '_soc',
    'energy nac soc':   '_soc',
    'energy gradient soc': '_soc',
    'energy gradient nac soc': '_soc',
    'absorption':       '_absorption',
    'frequency':        '_frequency',
    'energy frequency': '_frequency',
    'optimize':         '_geometry_optimization',
    'abinitio_dynamics': '_abinitio_dynamics',
    'nonadiabatic_dynamics': '_abinitio_dynamics',
    'surface_hopping':  '_surface_hopping',
    'md':               '_abinitio_dynamics',
}

# Synonymous task token: a dipole moment is a by-product of the
# energy evaluation, so it translates to energy before the suffix
# lookup (deduplicated with an existing energy request).
_SYNONYM_TOKENS = {
    'dipole': 'energy',
}

# The transition token requests transition dipoles on top of the
# remaining tasks: it is stripped before the suffix lookup (recorded
# as a flag) and satisfied by appending a RASSI step to the module
# chain, i.e. upgrading the energy template to the absorption one.
_TRANSITION_TOKEN = 'transition'

# Property tokens that the generic *_properties profile can combine
# freely. Legacy specific profiles keep their existing behaviour.
_PROPERTY_TOKENS = frozenset({'energy', 'gradient', 'nac', 'soc'})


def has_transition_request(jobtype):
    """Whether a jobtype string carries a transition-dipole request.

    Parameters
    ----------
    jobtype : str or None
        Job type string (e.g. 'energy gradient transition').

    Returns
    -------
    bool
        True if the 'transition' token is present.
    """
    if not jobtype:
        return False
    return _TRANSITION_TOKEN in jobtype.lower().split()


def canonical_jobtype_tokens(jobtype):
    """Normalize jobtype tokens for the suffix-table lookup.

    Translates synonymous tokens (dipole -> energy), strips the
    transition token (handled by the workflow, not the profile) and
    returns the remaining tokens sorted and deduplicated.

    Parameters
    ----------
    jobtype : str
        Job type string (e.g. 'energy gradient transition').

    Returns
    -------
    tuple of (str, bool)
        canonical: sorted deduplicated token string for the suffix
        lookup (may be empty when only transition was requested).
        transition: whether the transition token was present.
    """
    tokens = jobtype.lower().strip().split()
    transition = _TRANSITION_TOKEN in tokens
    tokens = [t for t in tokens if t != _TRANSITION_TOKEN]
    tokens = [_SYNONYM_TOKENS.get(t, t) for t in tokens]
    return ' '.join(sorted(set(tokens))), transition


def resolve_profile(method, jobtype):
    """Derive the profile from method and jobtype.

    When the profile is not explicitly provided (e.g. the classic
    input flow writes jobtype but not profile), this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype.

    Synonymous tasks are translated first: 'dipole' is a by-product
    of the energy evaluation and merges with the energy request.
    'transition' does not change the profile: it is stripped before
    the lookup and satisfied by appending a RASSI step to the module
    chain, so all remaining tasks (gradient, nac, soc, ...) keep
    their normal collapse behaviour.

    Parameters
    ----------
    method : str
        Computational method (e.g. 'hf', 'casscf', 'caspt2').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient',
        'energy gradient soc').

    Returns
    -------
    tuple of (str, bool)
        profile: profile name (e.g. 'casscf_energy', 'casscf_soc').
        transition: whether the jobtype requests transition dipoles
        (a RASSI step is appended to the module chain).

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

    # Jobtype tokens are freely composable and order-independent;
    # canonicalize by sorting before lookup.
    canonical, transition = canonical_jobtype_tokens(jobtype)

    # A bare transition request collapses to the energy task: the
    # transition dipoles come from the appended RASSI step.
    if not canonical:
        canonical = 'energy'

    suffix = _JOBBERTYPE_SUFFIX_MAP.get(canonical)
    if suffix is None:
        tokens = set(canonical.split())
        generic = f'{method}_properties'
        if tokens <= _PROPERTY_TOKENS and generic in AVAILABLE_PROFILES:
            return generic, transition
        raise ValueError(
            f"Cannot resolve profile for jobtype '{jobtype}'. "
            f"Known mappings: {sorted(_JOBBERTYPE_SUFFIX_MAP)}"
        )

    profile = f'{method}{suffix}'

    if profile not in AVAILABLE_PROFILES:
        tokens = set(canonical.split())
        generic = f'{method}_properties'
        if tokens <= _PROPERTY_TOKENS and generic in AVAILABLE_PROFILES:
            return generic, transition
        raise ValueError(
            f"Resolved profile '{profile}' (from method='{method}', "
            f"jobtype='{jobtype}') is not available. "
            f"Available profiles: {AVAILABLE_PROFILES}"
        )

    return profile, transition
