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

This module formats PySCF results into standardized output
aligned with PyRAI3MD data type conventions.

"""

import numpy as np


def format_results(results: dict, config) -> dict:
    """Format raw results into standardized output.

    Converts numpy arrays to lists for JSON serialization
    and ensures consistent output format.

    Parameters
    ----------
    results : dict
        Raw results from parse_output.
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
        elif isinstance(value, dict):
            formatted[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, list):
            formatted[key] = [
                v.tolist() if isinstance(v, np.ndarray) else v
                for v in value
            ]
        else:
            formatted[key] = value

    return formatted
