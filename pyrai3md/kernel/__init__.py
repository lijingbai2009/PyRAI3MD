"""
======================================================================
This module is the public API for the PyRAI3MD kernel package.
It exposes the Maestro, Environment,
logging utilities, and the unified exception hierarchy.

Jul 31 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .maestro import Maestro
from .environment import Environment
from .logger import get_logger
from .exceptions import (
    PyRAI3MDError,
    SkillNotFoundError,
    CapabilityNotFoundError,
    ValidationError,
    PyRAI3MDRuntimeError,
    InputError,
    CalculationError,
    ConvergenceError,
    CapabilityError,
)
