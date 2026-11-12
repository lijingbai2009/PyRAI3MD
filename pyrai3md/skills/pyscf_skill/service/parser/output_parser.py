"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module extracts raw results from PySCF runtime dictionaries.
For PySCF (Python-native), the runtime dict already contains
the computed properties; this module organizes them for output.

"""


def parse_output(runtime: dict, config, workflow=None) -> dict:
    """Extract results from the PySCF runtime dictionary.

    Parameters
    ----------
    runtime : dict
        Runtime dictionary containing mol, mf, post, and results.
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph, optional
        Compiled workflow graph.

    Returns
    -------
    dict
        Extracted results dictionary.
    """
    results = runtime.get('results', {})

    return results
