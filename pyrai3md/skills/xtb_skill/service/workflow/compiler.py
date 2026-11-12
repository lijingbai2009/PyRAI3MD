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

This module compiles a workflow graph from a profile name.
It resolves the profile into an ordered list of xTB steps
and determines special execution requirements (e.g. gradient output).

"""

from ...schema.method import PROFILE_STEP_MAP
from .graph import WorkflowGraph


def compile_workflow(profile: str) -> WorkflowGraph:
    """Compile a workflow graph from a profile name.

    Parameters
    ----------
    profile : str
        Calculation profile name (e.g. 'gfn2_gradient').

    Returns
    -------
    WorkflowGraph
        Compiled workflow with step chain and execution flags.

    Raises
    ------
    ValueError
        If the profile is not recognized.
    """
    profile = profile.lower()

    if profile not in PROFILE_STEP_MAP:
        raise ValueError(f"Unknown profile: {profile}")

    steps = PROFILE_STEP_MAP[profile]

    return WorkflowGraph(
        profile=profile,
        steps=steps,
    )
