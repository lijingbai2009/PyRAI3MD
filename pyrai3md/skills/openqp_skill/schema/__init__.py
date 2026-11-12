"""
======================================================================
Public API of the openqp skill schema package.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .config import SkillConfig, Config, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES

__all__ = [
    'SkillConfig',
    'Config',
    'FIELD_TO_SUBMODEL',
    'SUBMODEL_CLASSES',
]
