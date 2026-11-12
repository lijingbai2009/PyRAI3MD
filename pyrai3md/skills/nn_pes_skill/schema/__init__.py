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

nn_pes skill schema module.

"""

from .config import Config, SkillConfig, DEFAULT, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES
from .hyper import HYPER_KEYWORDS, build_member_hyper, member_list
from .method import (METHODS, PROFILE_ENSEMBLES, PROFILE_METHOD_MAP,
                     PROFILE_STEP_MAP, ROLE_SECTIONS, resolve_profile)
from .adapter import build_config, read_json
from .normalize import normalize, coupling_pairs
from .validate import validate
from .reference import (STATE_LABEL, LABEL_ENERGY_UNIT, LABEL_GRADIENT_UNIT,
                        LABEL_SOC_UNIT, METRIC_ENERGY_UNIT,
                        METRIC_GRADIENT_UNIT, METRIC_SOC_UNIT,
                        ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM,
                        OUTPUT_GRADIENT_UNIT)

__all__ = [
    'Config',
    'SkillConfig',
    'DEFAULT',
    'FIELD_TO_SUBMODEL',
    'SUBMODEL_CLASSES',
    'HYPER_KEYWORDS',
    'build_member_hyper',
    'member_list',
    'METHODS',
    'PROFILE_METHOD_MAP',
    'PROFILE_ENSEMBLES',
    'PROFILE_STEP_MAP',
    'ROLE_SECTIONS',
    'resolve_profile',
    'build_config',
    'read_json',
    'normalize',
    'coupling_pairs',
    'validate',
    'STATE_LABEL',
    'LABEL_ENERGY_UNIT',
    'LABEL_GRADIENT_UNIT',
    'LABEL_SOC_UNIT',
    'METRIC_ENERGY_UNIT',
    'METRIC_GRADIENT_UNIT',
    'METRIC_SOC_UNIT',
    'ANGSTROM_TO_BOHR',
    'BOHR_TO_ANGSTROM',
    'OUTPUT_GRADIENT_UNIT',
]
