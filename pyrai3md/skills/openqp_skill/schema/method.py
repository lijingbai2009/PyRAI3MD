"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the available methods and their capabilities
for the OpenQP skill. This is the single source of truth for
method-task compatibility, OpenQP runtype mapping, and required
input file sections.

"""

# Supported electronic structure methods and their capabilities.
# Each method defines which tasks it supports and which OpenQP
# input sections may be needed.
AVAILABLE_METHOD = {
    'hf': {
        'energy': True,
        'gradient': True,
        'nac': False,
        'soc': False,
        'geometry_optimization': True,
        'nonadiabatic_dynamics': False,
        'frequency': True,
        'sections': ['input', 'guess', 'scf', 'properties'],
        'description': 'Hartree-Fock (ground state)',
    },
    'tddft': {
        'energy': True,
        'gradient': True,
        'nac': False,
        'soc': False,
        'geometry_optimization': True,
        'nonadiabatic_dynamics': True,
        'frequency': True,
        'sections': ['input', 'guess', 'scf', 'tdhf', 'properties'],
        'description': 'TD-DFT / TD-HF (RPA or TDA)',
    },
    'mrsf': {
        'energy': True,
        'gradient': True,
        'nac': True,
        'soc': True,
        'geometry_optimization': True,
        'nonadiabatic_dynamics': True,
        'frequency': True,
        'sections': ['input', 'guess', 'scf', 'tdhf', 'properties'],
        'description': 'Spin-Flip MRSF-TDDFT (ROHF reference, multiplicity=3)',
    },
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'hf_energy': 'hf',
    'hf_gradient': 'hf',
    'hf_geometry_optimization': 'hf',
    'hf_frequency': 'hf',
    'tddft_energy': 'tddft',
    'tddft_gradient': 'tddft',
    'tddft_geometry_optimization': 'tddft',
    'tddft_nonadiabatic_dynamics': 'tddft',
    'tddft_frequency': 'tddft',
    'mrsf_energy': 'mrsf',
    'mrsf_gradient': 'mrsf',
    'mrsf_nac': 'mrsf',
    'mrsf_soc': 'mrsf',
    'mrsf_soc_gradient': 'mrsf',
    'mrsf_geometry_optimization': 'mrsf',
    'mrsf_nonadiabatic_dynamics': 'mrsf',
    'mrsf_frequency': 'mrsf',
}

# Profile to OpenQP runtype mapping.
# *_nac profiles use runtype=nac (numerical NAC vector, single state
# pair, single-point PES analysis); with nac_type=dcm the compiler
# switches to runtype=prop (nacme via MO overlap).
# *_nonadiabatic_dynamics default to runtype=data (compute_data:
# per-step energies + gradients for the framework kTDC integrator);
# with nac_type=dcm the compiler switches to runtype=prop (nacme via
# MO overlap, which compute_data does not support), and with a
# numerical NAC vector request it stays on data and adds
# [properties] nac=nac.  Adding spin-orbit coupling to a data run is
# legal, but then no NAC of any flavour may be requested (OpenQP
# refuses both: the analytic vector route answers the singlet fold
# only, and runtype=data keeps no overlap with the previous step).
# mrsf_soc and mrsf_soc_gradient split the same way the other methods
# split _energy from _gradient: runtype=soc solves both ladders and couples
# them but computes no gradient at all, while runtype=data takes each
# ladder's gradient and then couples them.  runtype=prop is not the answer
# to "gradients and couplings": its driver opens every run by loading the
# previous step's checkpoint, so standalone it stops with "molecule loads
# previous step data cannot find [guess] file2 or [input] system2"
# (measured).  It is reachable only through the nac_type=dcm dynamics path,
# which writes the step before it.
PROFILE_RUNTYPE_MAP = {
    'hf_energy': 'energy',
    'hf_gradient': 'grad',
    'hf_geometry_optimization': 'optimize',
    'hf_frequency': 'hess',
    'tddft_energy': 'energy',
    'tddft_gradient': 'grad',
    'tddft_geometry_optimization': 'optimize',
    'tddft_nonadiabatic_dynamics': 'data',
    'tddft_frequency': 'hess',
    'mrsf_energy': 'energy',
    'mrsf_gradient': 'grad',
    'mrsf_nac': 'nac',
    'mrsf_soc': 'soc',
    'mrsf_soc_gradient': 'data',
    'mrsf_geometry_optimization': 'optimize',
    'mrsf_nonadiabatic_dynamics': 'data',
    'mrsf_frequency': 'hess',
}

# Profile to OpenQP input section chain mapping.
# The section order matches the OpenQP INI input file layout.
PROFILE_SECTION_MAP = {
    'hf_energy': ['input', 'guess', 'scf', 'properties'],
    'hf_gradient': ['input', 'guess', 'scf', 'properties'],
    'hf_geometry_optimization': ['input', 'guess', 'scf', 'properties', 'optimize'],
    'hf_frequency': ['input', 'guess', 'scf', 'properties', 'hess'],
    'tddft_energy': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'tddft_gradient': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'tddft_geometry_optimization': ['input', 'guess', 'scf', 'tdhf', 'properties', 'optimize'],
    'tddft_nonadiabatic_dynamics': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'tddft_frequency': ['input', 'guess', 'scf', 'tdhf', 'properties', 'hess'],
    'mrsf_energy': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'mrsf_gradient': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'mrsf_nac': ['input', 'guess', 'scf', 'tdhf', 'properties', 'nac'],
    # No `nac` section: a spin-orbit run cannot request a coupling of any
    # NAC flavour alongside SOC, so there is nothing for it to carry.
    'mrsf_soc': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'mrsf_soc_gradient': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'mrsf_geometry_optimization': ['input', 'guess', 'scf', 'tdhf', 'properties', 'optimize'],
    'mrsf_nonadiabatic_dynamics': ['input', 'guess', 'scf', 'tdhf', 'properties'],
    'mrsf_frequency': ['input', 'guess', 'scf', 'tdhf', 'properties', 'hess'],
}

# Profiles that involve nonadiabatic coupling computation.
# *_nac: runtype=nac NAC vector (nac_type=nac/ktdc) or runtype=prop
# nacme (nac_type=dcm). *_nonadiabatic_dynamics: runtype=data
# (energies + gradients, nac_type=ktdc) or runtype=prop nacme
# (nac_type=dcm). Only the runtype=prop path may request nacme, which
# needs the previous-step orbital json via [guess]file2 for MO overlap.
NAC_PROFILES = [
    'tddft_nonadiabatic_dynamics',
    'mrsf_nac',
    'mrsf_nonadiabatic_dynamics',
]

# Profiles that may request spin-orbit coupling.  'mrsf_soc' always does;
# 'mrsf_nonadiabatic_dynamics' does only when the framework's &md dosoc flag
# is on.  Whether a compiled run is a two-manifold one is decided by the
# compiler and carried on WorkflowGraph.soc, which is what the parser reads
# to choose between labelled and unlabelled export names.
SOC_PROFILES = [
    'mrsf_soc',
    'mrsf_soc_gradient',
    'mrsf_nonadiabatic_dynamics',
]

# Mapping from multi-token jobtype string to profile task suffix.
# Used by resolve_profile to derive the profile from method + jobtype
# when profile is not explicitly provided (e.g. classic input flow).
# Keys are the sorted, deduplicated token sets that
# data/core.spell_skill_jobtype emits.
_JOBBERTYPE_SUFFIX_MAP = {
    'energy':           '_energy',
    'gradient':         '_gradient',
    'energy gradient':  '_gradient',
    'nac':              '_nac',
    'energy gradient nac': '_nac',
    'soc':              '_soc',
    'energy soc':       '_soc',
    'gradient soc':     '_soc_gradient',
    'energy gradient soc': '_soc_gradient',
    # A run asking for both couplings lands on the SOC profile so that
    # validate_soc can name the engine rule it breaks, instead of the
    # generic "jobtype not recognized" from this lookup.
    'energy gradient nac soc': '_soc_gradient',
    'frequency':        '_frequency',
    'energy frequency': '_frequency',
    'optimize':         '_geometry_optimization',
    'nonadiabatic_dynamics': '_nonadiabatic_dynamics',
    'md':               '_nonadiabatic_dynamics',
}


def canonical_jobtype(jobtype):
    """Sort and deduplicate jobtype tokens for the suffix-table lookup.

    The framework spells a jobtype in its own capability order, which for
    the energy/gradient/nac/soc tokens is alphabetical, but a hand-written
    input may list the same tokens in any order.  Normalizing here keeps
    one request spelling from resolving to a different profile than
    another order of the same tokens.

    Parameters
    ----------
    jobtype : str
        Job type string, any token order.

    Returns
    -------
    str
        Space-joined sorted deduplicated tokens.
    """
    return ' '.join(sorted({token for token in jobtype.lower().split()
                            if token}))


def resolve_profile(method, jobtype):
    """Derive the profile from method and jobtype.

    When the profile is not explicitly provided (e.g. the classic
    input flow writes jobtype but not profile), this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype.

    Parameters
    ----------
    method : str
        Computational method ('hf', 'tddft', or 'mrsf').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient',
        'energy gradient nac', 'optimize').

    Returns
    -------
    str
        Profile name (e.g. 'hf_energy', 'mrsf_nac').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized.
    """
    from .reference import AVAILABLE_PROFILES

    method = method.lower()
    jobtype = canonical_jobtype(jobtype)

    if method not in AVAILABLE_METHOD:
        raise ValueError(
            f"Cannot resolve profile: unknown method '{method}'"
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
