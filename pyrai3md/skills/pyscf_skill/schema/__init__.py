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

PySCF skill schema module.

"""

from .config import Config, SkillConfig, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES
from .method import AVAILABLE_METHOD, PROFILE_METHOD_MAP, PROFILE_STEP_MAP, initialize_method_classes
from .adapter import build_config, read_json
from .normalize import normalize
from .validate import validate
from .reference import (
    STATE_LABEL,
    PERIODIC_TABLE,
    AVAILABLE_FUNCTIONAL,
    AVAILABLE_BASIS,
    AVAILABLE_SOLVENT,
)
