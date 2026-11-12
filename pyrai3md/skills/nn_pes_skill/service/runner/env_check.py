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

This module provides environment validation and the lazy import of the
neural network container.

TensorFlow is an optional dependency of this skill (install with
'pip install pyrai3md[nn]'). It is imported here, after the
environment has been checked, and never at module import time of the
skill or its query layer, so a machine without TensorFlow still
discovers the skill and is told why it cannot run.

"""

import os
import sys


def configure_threads(config):
    """Publish the serving thread count into the process environment once.

    nn_pes serves in-process TensorFlow. ``predict_thread`` is the serving
    budget (empty/0 has already been resolved to ``thread`` by the schema);
    training children get their own ``train_thread`` through
    ``child_environment`` instead. The environment variables are published
    before TensorFlow is imported so both its OpenMP-backed kernels and its
    intra/inter-op pools read the same number.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration carrying ``environment.predict_thread``.
    """
    requested = str(config.environment.predict_thread)
    for key in ('OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
                'TF_NUM_INTEROP_THREADS'):
        effective = os.environ.setdefault(key, requested)
        if effective != requested:
            print('[NN-PES] %s is already %s for this process, so '
                  'predict_thread=%s is ignored; the thread count is fixed '
                  'at the first parallel region.' % (key, effective, requested),
                  file=sys.stderr)


def configure_tensorflow_threads(config):
    """Apply the serving thread count to the imported TensorFlow runtime.

    Called after ``import tensorflow`` and before the first op. The
    environment variables set by configure_threads() are the primary
    control; this call makes the setting explicit when a host process has
    already imported TensorFlow with different defaults.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration carrying ``environment.predict_thread``.
    """
    try:
        import tensorflow as tf
    except Exception:
        return
    value = max(1, int(config.environment.predict_thread))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(value)
        tf.config.threading.set_inter_op_parallelism_threads(value)
    except RuntimeError:
        # TensorFlow has already created its pools; the environment values
        # remain the only honest answer in that process.
        return


def tensorflow_available():
    """Report whether TensorFlow can be imported in this interpreter.

    Returns
    -------
    bool
        True when 'import tensorflow' succeeds.
    """
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


def import_container():
    """Import the vendored neural network container.

    Returns
    -------
    class
        NeuralNetPes container class.

    Raises
    ------
    RuntimeError
        If TensorFlow (or the vendored container itself) cannot be
        imported. The message names the optional extra to install.
    """
    if sys.version_info[:2] < (3, 11):
        raise RuntimeError(
            "nn_pes requires Python 3.11 or newer (Keras 3 does not "
            f"support {sys.version.split()[0]})")
    try:
        from ..nn_pes.nn_pes import NeuralNetPes
    except ImportError as error:
        raise RuntimeError(
            "tensorflow is not installed or cannot be imported; the nn_pes "
            "skill needs it. Install it with 'pip install \"tensorflow>=2.20\"' "
            "or with the neural network extra of the host package. "
            "Underlying error: %s" % error)
    return NeuralNetPes


def validate_environment(config):
    """Check that this skill can run in the current environment.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If TensorFlow cannot be imported in this environment.
    """
    if not tensorflow_available():
        raise RuntimeError(
            "tensorflow is not installed; the nn_pes skill cannot serve or "
            "train networks without it. Install 'tensorflow>=2.20' (the "
            "neural network extra of the host package).")
