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

Runner selection for the nn_pes skill.

Serving is in-process by definition (see local_runner); execution_mode
only decides which arm starts the training jobs.

The training launcher is imported by the trainer rather than here: it
reaches the vendored container, which pulls in TensorFlow. Keeping it out
of this package's import graph is what lets a machine without the
optional neural network stack still import the backend and be told, by
name, why it cannot run.

"""

from .env_check import (configure_threads, configure_tensorflow_threads,
                        import_container, tensorflow_available,
                        validate_environment)  # noqa: F401
from .local_runner import serve


def get_serving():
    """Return the per-step serving callable.

    Returns
    -------
    callable
        serve(config, calcdir, container).
    """
    return serve


def get_train_arm(config):
    """Return the training arm selected by execution_mode.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with environment.execution_mode.

    Returns
    -------
    str
        'local' or 'slurm'.
    """
    mode = (config.environment.execution_mode or 'local').lower()
    return 'slurm' if mode == 'slurm' else 'local'


def results_file_name(config_file):
    """Return the agent results file path for one config file.

    The cluster-side agent writes the formatted results next to the
    serialized config; the backend collects them through the same naming
    convention.

    Parameters
    ----------
    config_file : str
        Path of the serialized config JSON.

    Returns
    -------
    str
        Path of the results JSON.
    """
    base, _extension = config_file.rsplit('.', 1)
    return f'{base}.results.json'
