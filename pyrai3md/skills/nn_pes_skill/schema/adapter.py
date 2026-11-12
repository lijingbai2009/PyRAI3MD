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

This module provides the adapter functions for the nn_pes skill,
which include the functions to build and process the config.

"""

import json
from typing import Any

from pydantic import BaseModel

from .config import (SkillConfig, Config,
                     FIELD_TO_SUBMODEL, SUBMODEL_CLASSES)


# Retired compute-resource spellings (#306).
RETIRED_RESOURCES = {
    'num_thread': ('`thread` (standalone) or `nn_pes_thread` (the nn_pes skill '
                   'section); the framework keeps no run-wide thread setting'),
}


def reject_retired_resources(raw_dict):
    """Refuse a retired compute-resource spelling.

    Parameters
    ----------
    raw_dict : dict
        Raw skill configuration before the schema filter.

    Raises
    ------
    ValueError
        When a retired keyword carries a value.
    """
    for retired, replacement in sorted(RETIRED_RESOURCES.items()):
        if raw_dict.get(retired) is not None:
            raise ValueError(
                "nn_pes_skill: `%s` is retired; write %s instead"
                % (retired, replacement))


def read_json(filename: str) -> dict:
    """Read configuration from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def build_config(raw_config: Any) -> SkillConfig:
    """Build nested SkillConfig from raw input dict or model.

    Flat dict inputs are auto-routed to the appropriate submodels via
    FIELD_TO_SUBMODEL (aligned with the PyRAIConfig pattern).

    Parameters
    ----------
    raw_config : Any
        Raw configuration input, can be a dict or BaseModel.

    Returns
    -------
    SkillConfig
        Nested skill configuration model.
    """
    if isinstance(raw_config, BaseModel):
        raw_dict = raw_config.model_dump()
    elif isinstance(raw_config, dict):
        raw_dict = raw_config
    else:
        raise TypeError("Unsupported raw_config type")

    # Agent JSON configs may carry nested submodel sections (written by
    # the certification handler); flatten them so the flat-field routing
    # below picks them up.
    for _section_name in SUBMODEL_CLASSES:
        _section = raw_dict.get(_section_name)
        if isinstance(_section, dict):
            raw_dict = {**raw_dict, **_section}

    # Retired spellings are refused instead of dropped (#306).
    reject_retired_resources(raw_dict)

    # Filter to known schema fields only. Unknown keys are dropped here,
    # which is why every new skill keyword must be registered in config
    # and covered by query.get_defaults().
    valid_fields = FIELD_TO_SUBMODEL.keys()
    filtered = {k: v for k, v in raw_dict.items() if k in valid_fields}

    # Map the skill-specific project name to the canonical
    # Calculation.project field. The classic input writes nn_project into
    # the Environment section, which the framework does not forward to the
    # skill config, so it is copied here.
    if 'nn_project' in raw_dict and raw_dict['nn_project']:
        filtered['project'] = raw_dict['nn_project']

    # Fill empty/missing fields with schema non-empty defaults. An
    # explicit False on a bool switch is kept so it is not silently
    # overwritten by its default.
    for field_name, parent_name in FIELD_TO_SUBMODEL.items():
        sub_cls = SUBMODEL_CLASSES[parent_name]
        field_info = sub_cls.model_fields[field_name]
        # Only a missing or genuinely empty entry is filled. An explicit
        # 0 is a value, not an absence: treating it as unset would let a
        # caller's mult_nn=0 or an explicit zero in a role section
        # silently become the
        # default instead of being rejected by validate.
        present = field_name in filtered
        value = filtered.get(field_name)
        empty = (not present or value is None or value == ''
                 or (isinstance(value, (list, dict, tuple)) and len(value) == 0))
        if empty:
            default = field_info.default
            if default is not None and default != '' and default != []:
                filtered[field_name] = default

    # Keep Environment.nn_project in sync with Calculation.project
    # (single source of truth).
    filtered['nn_project'] = filtered['project']

    # SkillConfig.__init__ auto-routes flat kwargs to submodels
    return SkillConfig(**filtered)


