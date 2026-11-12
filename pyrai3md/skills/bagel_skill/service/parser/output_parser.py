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

This module is the main parsing entry point.
It selects the appropriate parser based on the calculation profile.

"""

import os
import numpy as np

from .log_parser import parse_log


def parse_output(config, calcdir: str, workflow) -> dict:
    """Parse BAGEL output files based on the calculation profile.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary.
    """
    project = config.project
    log_path = os.path.join(calcdir, f'{project}.log')

    # Parse the log file
    results = parse_log(log_path, config)

    return results


def format_results(results: dict, config) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Formatted results with numpy arrays converted to lists.
    """
    formatted = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        else:
            formatted[key] = value

    return formatted
