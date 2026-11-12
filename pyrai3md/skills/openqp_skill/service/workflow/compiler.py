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

This module compiles a workflow graph from a profile name.
It resolves the profile into an ordered list of OpenQP input
sections and determines the runtype and NAC-related execution flags.

"""

from ...schema.method import PROFILE_SECTION_MAP
from ...schema.method import PROFILE_RUNTYPE_MAP
from ...schema.method import NAC_PROFILES
from .graph import WorkflowGraph


def compile_workflow(profile: str, nac_type: str = 'ktdc',
                     is_soc: bool = False, has_grad_t: bool = False) -> WorkflowGraph:
    """Compile a workflow graph from a profile name.

    Parameters
    ----------
    profile : str
        Calculation profile name (e.g. 'tddft_gradient').
    nac_type : str
        NAC calculation type ('ktdc', 'dcm' or 'nac'). For *_nac
        profiles, nac_type=dcm switches runtype=nac (numerical NAC
        vector, single state pair, PES analysis) to runtype=prop
        (nacme via MO overlap). For *_nonadiabatic_dynamics profiles,
        nac_type=dcm switches the default runtype=data (per-step
        energies + gradients for the framework kTDC integrator) to
        runtype=prop, and nac_type=nac keeps it on data while adding
        the numerical NAC vectors that path answers. Other profiles
        ignore this flag.
    is_soc : bool
        Whether spin-orbit coupling is requested, either by the profile
        (mrsf_soc) or by the framework's &md dosoc flag.
    has_grad_t : bool
        Whether an explicit gradient list names the other ladder. It
        makes the run two-manifold on its own, and OpenQP then labels
        every export with the manifold holding it.

    Returns
    -------
    WorkflowGraph
        Compiled workflow with section chain, runtype, and flags.

    Raises
    ------
    ValueError
        If the profile is not recognized.
    """
    profile = profile.lower()

    if profile not in PROFILE_SECTION_MAP:
        raise ValueError(f"Unknown profile: {profile}")

    sections = PROFILE_SECTION_MAP[profile]
    runtype = PROFILE_RUNTYPE_MAP[profile]
    is_nac_profile = profile in NAC_PROFILES

    # nacme (dcm) needs the MO-overlap path: inside a *_nac profile it
    # switches runtype=nac to prop; inside a *_nonadiabatic_dynamics
    # profile it switches the default data path (per-step energies +
    # gradients for the framework kTDC integrator) to prop.
    if is_nac_profile and nac_type == 'dcm':
        runtype = 'prop'

    # Numerical NAC vectors on the data path: compute_data calls the NAC
    # driver when [properties] asks for them, and the driver reads the
    # displacement control from the [nac] section, which the dynamics
    # profile does not carry by default.
    nac_vectors = (is_nac_profile and nac_type == 'nac'
                   and runtype == 'data' and not is_soc)
    if nac_vectors and 'nac' not in sections:
        sections = list(sections) + ['nac']

    # The data path carries no NAC keyword or [nac] section unless it is
    # taking vectors.
    if runtype == 'data' and not nac_vectors and 'nac' in sections:
        sections = [s for s in sections if s != 'nac']

    return WorkflowGraph(
        profile=profile,
        sections=sections,
        runtype=runtype,
        is_nac_profile=is_nac_profile,
        is_soc=is_soc,
        two_manifold=is_soc or has_grad_t,
        nac_vectors=nac_vectors,
    )
