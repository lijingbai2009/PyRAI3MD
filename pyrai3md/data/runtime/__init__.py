"""
======================================================================
This module is the public API for the PyRAI3MD runtime sub-package.
It exposes RuntimeState, Result, serialization utilities, and
state copy functions.

Aug 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .state import RuntimeState
from .result import Result
from .serializer import (
    save_state, load_state, prepare_state,
    shallow_copy_state, deep_copy_state,
)
