"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the method and profile vocabulary of the nn_pes
skill.

A method names the network family that serves the request. The five
primary methods are the model types of the container; the two composite
methods load the energy ensemble and the coupling ensemble together,
which is what a dynamics step asks for. Each method exposes the
profiles:

    {method}_energy     energy only
    {method}_gradient   energy + gradient
    {method}_nac        energy + gradient + couplings (dynamics superset)
    {method}_train      fit the ensemble from data (jobtype 'train')

"""

# Model families of the container plus the two composite spellings
_METHODS = ('mlp_e', 'mlp_eg', 'mlp_g2', 'mlp_nac', 'mlp_nac2',
            'eg_nac', 'eg_nac2')

# The role an ensemble folder serves. It decides what the network's output
# channels mean and how wide they are: 'energy_grad' holds one channel per
# electronic state, 'nac' one per declared coupling pair, and 'soc' one per
# declared spin-orbit pair.
ROLE_ENERGY = 'energy_grad'
ROLE_NAC = 'nac'
ROLE_SOC = 'soc'

# Public spelling of the method vocabulary
METHODS = _METHODS

_SUFFIXES = ('_energy', '_gradient', '_nac', '_soc', '_train')

# Which ensembles a profile loads, as (model folder inside modeldir,
# container model type) pairs. The folder names are the ones the original
# project wrote: the energy family lives in 'energy_grad', the coupling
# family in 'nac'. The composite methods load both from one model
# directory, which is what a dynamics step asks for.
PROFILE_ENSEMBLES = {
    'mlp_e_energy': ((ROLE_ENERGY, 'mlp_e'),),
    'mlp_e_train': ((ROLE_ENERGY, 'mlp_e'),),
    # A spin-orbit value is a scalar that depends on geometry, not a
    # derivative, so the request is served by the energy family: one output
    # channel per declared spin-orbit pair, reported in cm-1 with no gradient
    # alongside it.
    'mlp_e_soc': ((ROLE_SOC, 'mlp_e'),),
    'mlp_e_soc_train': ((ROLE_SOC, 'mlp_e'),),
    'mlp_eg_energy': ((ROLE_ENERGY, 'mlp_eg'),),
    'mlp_eg_gradient': ((ROLE_ENERGY, 'mlp_eg'),),
    'mlp_eg_train': ((ROLE_ENERGY, 'mlp_eg'),),
    'mlp_g2_gradient': ((ROLE_ENERGY, 'mlp_g2'),),
    'mlp_g2_train': ((ROLE_ENERGY, 'mlp_g2'),),
    'mlp_nac_nac': ((ROLE_NAC, 'mlp_nac'),),
    'mlp_nac_train': ((ROLE_NAC, 'mlp_nac'),),
    'mlp_nac2_nac': ((ROLE_NAC, 'mlp_nac2'),),
    'mlp_nac2_train': ((ROLE_NAC, 'mlp_nac2'),),
    'eg_nac_energy': ((ROLE_ENERGY, 'mlp_eg'),),
    'eg_nac_gradient': ((ROLE_ENERGY, 'mlp_eg'),),
    'eg_nac_nac': ((ROLE_ENERGY, 'mlp_eg'),
                   (ROLE_NAC, 'mlp_nac')),
    'eg_nac_train': ((ROLE_ENERGY, 'mlp_eg'),
                     (ROLE_NAC, 'mlp_nac')),
    'eg_nac_soc': ((ROLE_ENERGY, 'mlp_eg'),
                   (ROLE_NAC, 'mlp_nac'),
                   (ROLE_SOC, 'mlp_e')),
    'eg_nac_soc_train': ((ROLE_ENERGY, 'mlp_eg'),
                         (ROLE_NAC, 'mlp_nac'),
                         (ROLE_SOC, 'mlp_e')),
    'eg_nac2_energy': ((ROLE_ENERGY, 'mlp_eg'),),
    'eg_nac2_gradient': ((ROLE_ENERGY, 'mlp_eg'),),
    'eg_nac2_nac': ((ROLE_ENERGY, 'mlp_eg'),
                    (ROLE_NAC, 'mlp_nac2')),
    'eg_nac2_soc': ((ROLE_ENERGY, 'mlp_eg'),
                    (ROLE_NAC, 'mlp_nac2'),
                    (ROLE_SOC, 'mlp_e')),
    'eg_nac2_soc_train': ((ROLE_ENERGY, 'mlp_eg'),
                          (ROLE_NAC, 'mlp_nac2'),
                          (ROLE_SOC, 'mlp_e')),
    'eg_nac2_train': ((ROLE_ENERGY, 'mlp_eg'),
                      (ROLE_NAC, 'mlp_nac2')),
}

# Which hyperparameter section trains which ensemble. A role section is
# named after the network family it describes, not after the folder the
# trained members land in, because the two spellings differ for the
# energy family.
ROLE_SECTIONS = {
    ROLE_ENERGY: 'eg',
    ROLE_NAC: 'nac',
    ROLE_SOC: 'soc',
}

# How wide a network of each model type is, as the role that decides it.
# An energy or coupling family spans the electronic states or the declared
# coupling pairs; the spin-orbit family spans the declared spin-orbit pairs,
# because one SOC matrix element is one scalar output.
CHANNEL_COUNT = {
    'mlp_e': {ROLE_ENERGY: 'nstate', ROLE_SOC: 'nsoc'},
    'mlp_eg': {ROLE_ENERGY: 'nstate'},
    'mlp_g2': {ROLE_ENERGY: 'nstate'},
    'mlp_nac': {ROLE_NAC: 'nstate'},
    'mlp_nac2': {ROLE_NAC: 'nnac'},
}

# Model families that size their output by a coupling channel count. The
# count is the number of pairs the caller declared, never a pair set
# derived here.
CHANNEL_COUNT_MODELS = ('mlp_nac', 'mlp_nac2')

# Profile to method mapping, read off the declared profiles rather than
# generated from them, because a role profile carries two suffixes:
# 'mlp_e_soc_train' is the method 'mlp_e' with the spin-orbit role and the
# training step. The longest suffix is stripped first so the remainder is
# the method name and not another role.
_PROFILE_SUFFIXES = ('_soc_train', '_train', '_energy', '_gradient',
                     '_nac', '_soc')
PROFILE_METHOD_MAP = {}
for _profile in PROFILE_ENSEMBLES:
    for _suffix in _PROFILE_SUFFIXES:
        if _profile.endswith(_suffix):
            _method = _profile[:-len(_suffix)]
            if _method in _METHODS:
                PROFILE_METHOD_MAP[_profile] = _method
            break

# Profiles each method actually exposes, kept in step order for
# diagnostics (the container runs one fit or one serving pass).
PROFILE_STEP_MAP = {
    profile: (['train', 'results'] if profile.endswith('_train')
              else ['load', 'properties', 'results'])
    for profile in PROFILE_ENSEMBLES
}

# Mapping from the normalized jobtype token set to a profile suffix.
# The framework spells capability tokens in a fixed order, but agent
# flows may hand over subsets or permuted forms; resolve_profile sorts
# and deduplicates the tokens before this lookup so the mapping is
# order-independent (same convention as the qdm skill).
_JOBERTYPE_SUFFIX_MAP = {
    'energy':                    '_energy',
    'gradient':                  '_gradient',
    'energy gradient':           '_gradient',
    'nac':                       '_nac',
    'energy nac':                '_nac',
    'energy gradient nac':       '_nac',
    'gradient nac':              '_nac',
    'soc':                       '_soc',
    # A dynamics run that switches spin-orbit coupling on asks for the three
    # channels at once, and the composite profile is the one that loads all
    # three ensembles.
    'energy soc':                '_soc',
    'nac soc':                   '_soc',
    'energy nac soc':            '_soc',
    'gradient nac soc':          '_soc',
    'energy gradient nac soc':   '_soc',
    'soc train':                 '_soc_train',
    'energy nac':                '_nac',
    'energy gradient nac':       '_nac',
    'gradient nac':              '_nac',
    'train':                     '_train',
    # A hyperparameter grid scan asks this skill for trainings: every
    # point fits one ensemble with one set of hyperparameters. The run
    # that plans the scan therefore carries a training profile, which is
    # also what the scan reads back to name the points it dispatches.
    # Resolving it to a serving profile would make the parent try to load
    # a surface that no point has trained yet.
    'search':                    '_train',
}

# A prediction request names no property of its own: it asks a model family
# to serve the channel it was trained for, and the networks are loaded once
# while the backend initializes, exactly like a property step. The suffix
# therefore belongs to the method, not to the token.
_PREDICTION_TOKENS = ('predict', 'prediction')
_METHOD_SERVING_SUFFIX = {
    'mlp_e': '_energy',
    'mlp_eg': '_gradient',
    'mlp_g2': '_gradient',
    'mlp_nac': '_nac',
    'mlp_nac2': '_nac',
    'eg_nac': '_nac',
    'eg_nac2': '_nac',
}

# Suffixes a method can serve, used to reject a jobtype the network
# family cannot answer (an energy-only network has no gradient head).
_METHOD_SUFFIXES = {
    'mlp_e': ('_energy', '_soc', '_soc_train', '_train'),
    'mlp_eg': ('_energy', '_gradient', '_train'),
    'mlp_g2': ('_gradient', '_train'),
    'mlp_nac': ('_nac', '_train'),
    'mlp_nac2': ('_nac', '_train'),
    'eg_nac': _SUFFIXES + ('_soc', '_soc_train'),
    'eg_nac2': _SUFFIXES + ('_soc', '_soc_train'),
}

# Jobtypes that would ask this skill to own the iteration and have no
# profile here ('md' has its own driver-aware refusal below, because the
# framework does serve it). A network potential never moves the nuclei and
# never searches a surface, so these are refused by name rather than
# resolved to a single point.
_UNSERVED_ITERATION_JOBTYPES = frozenset({'optimize', 'surface_hopping'})


def resolve_profile(method, jobtype, driver='main'):
    """Derive the profile from method, jobtype and driver.

    When the profile is not explicitly provided, this function
    determines the correct profile by combining the method name with
    the task implied by the jobtype tokens.

    Driver semantics (aligned with framework control.driver):
      - driver='main': the framework performs the dynamics and asks the
        skill for per-step properties. jobtype 'md' maps to the
        '{method}_nac' superset, which answers energy, gradient and
        couplings in one call.
      - driver='skill' or any other value: rejected for a dynamics request
        only. That driver asks the backend to propagate the nuclei, which a
        neural network potential never does. A 'train', 'prediction' or
        property request carries no propagation at all and is resolved as
        usual, whichever driver the run was planned under.
      - an 'optimize' or 'surface_hopping' request is rejected under any
        driver other than 'main' for the same reason: it would ask the
        network to own the iteration, and there is no profile for it here.

    Parameters
    ----------
    method : str
        Model family ('mlp_eg', 'mlp_nac2', 'eg_nac', ...).
    jobtype : str
        Job type token string (e.g. 'energy', 'energy gradient', 'train',
        'prediction').
    driver : str, optional
        Dynamics driver; consulted for a dynamics or iteration request.

    Returns
    -------
    str
        Profile name (e.g. 'mlp_eg_gradient', 'eg_nac_nac').

    Raises
    ------
    ValueError
        If the method, the jobtype, or the combination is not supported.
    """
    method = (method or '').lower().strip()
    jobtype = (jobtype or '').lower().strip()
    driver = (driver or 'main').lower().strip()

    if method not in _METHODS:
        raise ValueError(
            f"Cannot resolve profile: unknown method '{method}'; "
            f"available: {sorted(_METHODS)}")

    if driver != 'main' and _UNSERVED_ITERATION_JOBTYPES & set(jobtype.split()):
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' under "
            f"driver='{driver}' asks this skill to own the iteration, and a "
            "neural network potential serves per-step properties only. Run "
            "the iteration on the framework (driver: main), which resolves "
            "the workflow from the jobtype and asks for one step at a time, "
            "or ask this skill for a single evaluation "
            "(jobtype 'energy' or 'energy gradient')"
        )

    if jobtype in _PREDICTION_TOKENS:
        suffix = _METHOD_SERVING_SUFFIX[method]
    elif 'md' in jobtype.split():
        # Only a dynamics request makes the driver a question: driver='skill'
        # means the backend propagates the nuclei, which a neural network
        # potential never does. A training or property request carries no
        # propagation at all, so any driver value is answered there.
        if driver != 'main':
            raise ValueError(
                "Cannot resolve profile: a dynamics run under "
                f"driver='{driver}' would ask this skill to propagate the "
                "nuclei; it serves per-step properties only")
        suffix = '_nac'
    else:
        if ' ' in jobtype:
            jobtype = ' '.join(sorted(set(jobtype.split())))
        suffix = _JOBERTYPE_SUFFIX_MAP.get(jobtype)

    if suffix is None:
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' does not map "
            f"to a known profile suffix. "
            f"Available: {sorted(_JOBERTYPE_SUFFIX_MAP) + list(_PREDICTION_TOKENS) + ['md']}"
        )

    if suffix not in _METHOD_SUFFIXES[method]:
        raise ValueError(
            f"Cannot resolve profile: method '{method}' has no "
            f"'{suffix}' profile; available for this method: "
            f"{sorted(_METHOD_SUFFIXES[method])}"
        )

    profile = f'{method}{suffix}'
    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(
            f"Resolved profile '{profile}' (method={method}, "
            f"jobtype={jobtype}) is not declared in the manifest")
    return profile
