"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

QDM skill schema module.

"""

from .config import Config, SkillConfig, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES
from .method import PROFILE_METHOD_MAP, PROFILE_STEP_MAP, resolve_profile
from .adapter import build_config, read_json
from .normalize import normalize
from .validate import validate
from .reference import STATE_LABEL, MODEL_NSTATE

__all__ = [
    'Config',
    'SkillConfig',
    'FIELD_TO_SUBMODEL',
    'SUBMODEL_CLASSES',
    'PROFILE_METHOD_MAP',
    'PROFILE_STEP_MAP',
    'resolve_profile',
    'build_config',
    'read_json',
    'normalize',
    'validate',
    'STATE_LABEL',
    'MODEL_NSTATE',
]
