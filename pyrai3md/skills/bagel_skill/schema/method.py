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

This module defines the available methods and their capabilities
for the BAGEL skill. This is the single source of truth for
method-task compatibility and required BAGEL computation steps.

"""

# Supported electronic structure methods and their capabilities.
# Each method defines which profiles it supports and which BAGEL steps are needed.
AVAILABLE_METHOD = {
    'casscf': {
        'energy': True,
        'gradient': True,
        'nac': True,
        'frequency': True,
        'geometry_optimization': True,
        'dipole': True,
        'transition_dipole': True,
        'description': 'Complete Active Space SCF',
    },
    'xmscaspt2': {
        'energy': True,
        'gradient': True,
        'nac': True,
        'frequency': True,
        'geometry_optimization': True,
        'dipole': True,
        'transition_dipole': True,
        'description': 'XMS-CASPT2 (Extended Multi-State CASPT2)',
    },
}

# Profile to method mapping for quick lookup
PROFILE_METHOD_MAP = {
    'casscf_energy': 'casscf',
    'casscf_gradient': 'casscf',
    'casscf_nac': 'casscf',
    'casscf_frequency': 'casscf',
    'casscf_geometry_optimization': 'casscf',
    'casscf_dipole': 'casscf',
    'casscf_transition_dipole': 'casscf',
    'xmscaspt2_energy': 'xmscaspt2',
    'xmscaspt2_gradient': 'xmscaspt2',
    'xmscaspt2_nac': 'xmscaspt2',
    'xmscaspt2_frequency': 'xmscaspt2',
    'xmscaspt2_geometry_optimization': 'xmscaspt2',
    'xmscaspt2_dipole': 'xmscaspt2',
    'xmscaspt2_transition_dipole': 'xmscaspt2',
}

# Profile to BAGEL computation step chain mapping.
# Steps correspond to JSON blocks in the BAGEL input file.
PROFILE_STEP_MAP = {
    'casscf_energy': ['molecule', 'load_ref', 'casscf', 'save_ref'],
    'casscf_gradient': ['molecule', 'load_ref', 'casscf', 'forces'],
    'casscf_nac': ['molecule', 'load_ref', 'casscf', 'forces_nacme'],
    'casscf_frequency': ['molecule', 'load_ref', 'casscf', 'hessian'],
    'casscf_geometry_optimization': ['molecule', 'load_ref', 'casscf', 'optimize'],
    'casscf_dipole': ['molecule', 'load_ref', 'casscf', 'forces_dipole'],
    'casscf_transition_dipole': ['molecule', 'load_ref', 'casscf', 'forces_dipole'],
    'xmscaspt2_energy': ['molecule', 'load_ref', 'casscf', 'smith_caspt2'],
    'xmscaspt2_gradient': ['molecule', 'load_ref', 'forces_caspt2'],
    'xmscaspt2_nac': ['molecule', 'load_ref', 'forces_caspt2_nacme'],
    # The reference single point is part of this chain, not an assumption:
    # 'hessian_caspt2' alone solves internally but answers in a sibling file, and
    # its own log carries nothing a reader of this job can use - measured on the
    # deployment, `xmscaspt2_frequency_test.log` of round tmp_20261015_100320
    # (392 lines) holds no line matching mrci|caspt2|energy|total and no number of
    # the -77.xxxxxx magnitude at all, while the same directory's `freq.log` does
    # print one per displaced point (round tmp_20261015_172519, line 201:
    # `-77.0713120468`). The parser reads the job log and ENERGY.out, not those
    # per-executor sections, so the surface this run is scored on has to be in the
    # job's own document. Before the calculation directory was named per project
    # (#354) the parser filled that gap from a sibling case's ENERGY.out and the
    # case passed; round tmp_20261015_172519 reported the absence honestly as three
    # failures. The chain is therefore made a prefix of the one that does print -
    # `xmscaspt2_energy`'s `casscf` + `smith_caspt2` -
    # and the Hessian follows it. Mirroring that pair rather than inserting the
    # energy block alone matters: `load_ref` is skipped when no
    # `{project}_guess.archive` sits in the calculation directory
    # (`input_builder.py:151-161`), so the correction needs the reference solve in
    # the same joblist rather than an archive borrowed from elsewhere. Round
    # tmp_20261015_211219 ran this chain and the solve is there - that job's own log
    # now prints `* METHOD: CASSCF` and `* CASPT2 energy : state  0
    # -77.0713120427` - but a run that declares one state leaves SMITH with no
    # multistate Hamiltonian to diagonalise and it prints no MS row at all, which is
    # what the parser's one-state rule reads.
    # One CASPT2 energy evaluation against the eighteen displaced-gradient jobs
    # the Hessian already runs is not where this job's cost is.
    'xmscaspt2_frequency': ['molecule', 'load_ref', 'casscf',
                            'smith_caspt2', 'hessian_caspt2'],
    'xmscaspt2_geometry_optimization': ['molecule', 'load_ref', 'optimize'],
    'xmscaspt2_dipole': ['molecule', 'load_ref', 'forces_dipole'],
    'xmscaspt2_transition_dipole': ['molecule', 'load_ref', 'forces_dipole'],
}

# Mapping from multi-token jobtype string to profile task suffix.
# Used by resolve_profile to derive the profile from method + jobtype
# when profile is not explicitly provided (e.g. classic input flow).
_JOBBERTYPE_SUFFIX_MAP = {
    'energy':           '_energy',
    'gradient':         '_gradient',
    'energy gradient':  '_gradient',
    'nac':              '_nac',
    'energy gradient nac': '_nac',
    'frequency':        '_frequency',
    'energy frequency': '_frequency',
    'optimize':         '_geometry_optimization',
    'dipole':           '_dipole',
    'dipole energy':    '_dipole',
    'transition':       '_transition_dipole',
    'absorption':       '_transition_dipole',
    'energy transition': '_transition_dipole',
    'absorption energy': '_transition_dipole',
    'md':               '_gradient',
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
        Computational method ('casscf' or 'xmscaspt2').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient', 'frequency').

    Returns
    -------
    str
        Profile name (e.g. 'casscf_energy', 'xmscaspt2_gradient').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized.
    """
    from .reference import AVAILABLE_PROFILES

    method = method.lower()
    jobtype = jobtype.lower().strip()

    # Normalize token order and remove duplicates so that non-canonical
    # jobtype strings (e.g. 'gradient energy') resolve to the same profile.
    jobtype = ' '.join(sorted(set(jobtype.split())))

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
