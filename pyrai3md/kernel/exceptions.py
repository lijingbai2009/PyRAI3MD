"""
======================================================================
This module provides the unified exception hierarchy for PyRAI3MD.
All custom exceptions used across kernel, schema, skills, and
workflows are defined here to ensure consistent error handling.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""


class PyRAI3MDError(Exception):
    """Base exception for all PyRAI3MD errors."""
    pass


class SkillNotFoundError(PyRAI3MDError):
    """Raised when a requested skill cannot be found in the registry."""
    pass


class CapabilityNotFoundError(PyRAI3MDError):
    """Raised when a requested capability is not registered."""
    pass


class ValidationError(PyRAI3MDError):
    """Raised when configuration validation fails."""
    pass


class PyRAI3MDRuntimeError(PyRAI3MDError):
    """Raised when a runtime error occurs during execution."""
    pass


class InputError(PyRAI3MDError):
    """Raised when input file format is invalid or unreadable."""
    pass


class CalculationError(PyRAI3MDError):
    """Raised when a calculation fails (e.g. skill backend error)."""
    pass


class ConvergenceError(PyRAI3MDError):
    """Raised when convergence criteria are not met (optimization, SCF, etc.)."""
    pass


class CapabilityError(PyRAI3MDError):
    """Raised when a requested property is not supported by the current skill."""
    pass
