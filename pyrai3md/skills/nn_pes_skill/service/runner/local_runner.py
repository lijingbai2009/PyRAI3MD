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

This module provides the local serving path of the nn_pes skill.

Serving is always in-process. A dynamics step costs about a millisecond
of network evaluation, so submitting it to a scheduler would add seconds
per step; execution_mode therefore governs the training run only.

"""

from ..calculator import evaluate


def serve(config, calcdir, container):
    """Evaluate the current geometry through the loaded ensembles.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory (kept for the service signature; serving
        writes no files).
    container : object
        Container opened during backend initialization.

    Returns
    -------
    dict
        Results keyed by the Properties field names.
    dict
        Ensemble spread, for the log and the uncertainty file.
    """
    coordinates = config.system.coordinates
    return evaluate(config, coordinates, container=container)
