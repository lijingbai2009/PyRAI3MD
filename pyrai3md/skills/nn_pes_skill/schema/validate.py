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

This module provides the validation functions for the nn_pes skill.

The framework swallows a ValueError raised by resolve_profile while it
fills defaults, so anything the profile cannot serve is rejected here
instead of being resolved into a wrong calculation.

"""

import os

from .config import SkillConfig, Calculation, Environment
from .hyper import (HYPER_KEYWORDS, broadcast, check_value, member_groups,
                    member_list, parse_value)
from .method import (PROFILE_ENSEMBLES, PROFILE_METHOD_MAP, METHODS,
                     ROLE_SECTIONS)


def check_hyper_section(data, section):
    """Check one role section's keywords against the member axis.

    Parameters
    ----------
    data : dict
        Flat config dict.
    section : str
        Role section name ('eg' or 'nac').

    Raises
    ------
    ValueError
        For an unknown keyword or a value its kind rejects, and for
        member groups that do not cover the ensemble.
    """
    values = data.get(section) or {}
    if not isinstance(values, dict):
        raise ValueError(
            f"&{section} must be a set of keyword: value pairs, got "
            f"{type(values).__name__}")
    unknown = sorted(set(values) - set(HYPER_KEYWORDS))
    if unknown:
        raise ValueError(
            f"unknown hyperparameter in &{section}: {', '.join(unknown)}; "
            f"available: {', '.join(sorted(HYPER_KEYWORDS))}")
    members = data['mult_nn']
    for keyword, raw in values.items():
        kind = HYPER_KEYWORDS[keyword][0]
        groups = [parse_value(group, kind)
                  for group in member_groups(raw, kind)]
        for value in groups:
            check_value(keyword, value)
        broadcast(groups, members, keyword)


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for the neural network potential.

    Parameters
    ----------
    config : SkillConfig
        Normalized config.

    Returns
    -------
    SkillConfig
        Validated config with normalized string fields written back.
    """
    work = {}
    work.update(config.system.model_dump())
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    work = validate_method(work)
    work = validate_environment(work)

    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'calculation': config.calculation.model_copy(update=calc_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


def validate_method(data: dict) -> dict:
    # Verify that the profile-specific requirements are met
    profile = data['profile'].lower()
    method = data['method'].lower()

    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(
            f"Unsupported profile: {profile}; "
            f"available: {sorted(PROFILE_METHOD_MAP)}")

    expected_method = PROFILE_METHOD_MAP[profile]
    if method != expected_method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but got '{method}'")

    if method not in METHODS:
        raise ValueError(
            f"Unsupported method: {method}; available: {sorted(METHODS)}")

    # A periodic run cannot be served: the geometric features and the
    # index weights were built and measured for molecules only.
    if data.get('pbc') and any(data['pbc']):
        raise ValueError(
            "nn_pes is a molecular backend and does not support periodic "
            "boundaries; disable pbc or use a periodic skill.")

    if data['mult_nn'] < 1:
        raise ValueError("mult_nn must be at least 1.")
    mode = int(data.get('nn_predict_mode', 0) or 0)
    if mode < 0:
        raise ValueError(
            "nn_predict_mode must be >= 0 (0 = member mean), got %d" % mode)
    if mode > data['mult_nn']:
        raise ValueError(
            "nn_predict_mode=%d names a member outside the %d-member "
            "ensemble" % (mode, data['mult_nn']))
    if mode > 0 and data['mult_nn'] < 2:
        raise ValueError(
            "nn_predict_mode>0 needs at least 2 ensemble members to "
            "provide a comparison model")
    if data['nstate'] < 1:
        raise ValueError("nstate must be at least 1.")

    if profile.endswith('_train'):
        # A fit and a test-set evaluation are two different calls: the
        # first consumes train_data and writes a folder, the second reads
        # that folder back. Carrying both in one request means neither
        # statement was meant, so it is refused rather than guessed.
        if data['pred_data']:
            raise ValueError(
                f"Profile '{profile}' is a training profile and evaluates "
                "nothing; pred_data belongs to a serving profile (jobtype "
                "'prediction'), so remove one of the two")
        if not isinstance(data.get('train_dataset') or {}, dict):
            raise ValueError(
                "train_dataset must be a training-data mapping, got "
                f"{type(data['train_dataset']).__name__}")
        if bool(data['train_data']) == bool(data['train_dataset']):
            raise ValueError(
                f"Profile '{profile}' needs exactly one training data "
                "source: train_data (a file name) or train_dataset "
                "(inline data); set one and leave the other empty")
        if data['train_mode'] not in ('training', 'retraining'):
            raise ValueError(
                "train_mode must be 'training' or 'retraining', got "
                f"{data['train_mode']!r}")
        if not data['modeldir']:
            raise ValueError(
                f"Profile '{profile}' needs modeldir: a training run "
                "writes one folder per ensemble inside it")
        # Read the role sections the way the trainer will, so a value the
        # container would reject is reported before any folder is made.
        for folder, _model_type in PROFILE_ENSEMBLES[profile]:
            check_hyper_section(data, ROLE_SECTIONS[folder])

        # A device index per member; -1 asks for no device at all.
        for device in member_list(data['gpu'], data['mult_nn'], 'gpu'):
            if int(device) < -1:
                raise ValueError(
                    f"gpu must be a device index or -1, got {device!r}")
    else:
        # Serving profiles read a trained ensemble from disk
        if not data['modeldir']:
            raise ValueError(
                f"Profile '{profile}' needs modeldir pointing at a "
                "trained model folder")

    data['profile'] = profile
    data['method'] = method
    return data


def validate_environment(data: dict) -> dict:
    # Verify that the computing environment settings are valid
    scratch_path = os.path.expanduser(data['scratch_path'])
    # The thread count is not checked here (#306): the machine this run
    # lands on is not the one that parsed the input, and oversubscription
    # is allowed, so the framework does the arithmetic and warns instead
    # of failing the job (workflow/dispatch/scheduler.py).


    if data['execution_mode'] not in ('local', 'slurm'):
        raise ValueError(
            f"execution_mode must be 'local' or 'slurm', got "
            f"{data['execution_mode']!r}")

    if data['slurm_ntasks'] < 1:
        raise ValueError("slurm_ntasks must be at least 1.")
    if data['slurm_memory'] < 1:
        raise ValueError("slurm_memory must be positive.")

    if scratch_path and not os.path.exists(scratch_path):
        try:
            os.makedirs(scratch_path)
        except Exception as e:
            raise ValueError(
                f"Failed to create scratch path {scratch_path}: {e}")

    return data
