"""
======================================================================
This module is the public API for the PyRAI3MD data core package.
It exposes the central state object, runtime components, and
serialization utilities.

Aug 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# Runtime sub-package
from .runtime.state import RuntimeState
from .runtime.result import Result
from .runtime.serializer import (
    save_state, load_state, prepare_state,
    shallow_copy_state, deep_copy_state,
)

# Central state objects (PyRAIConfig defined in schema, not re-exported here)
from .state import PyRAIState
from .build import build_state

# Per-atom partial-charge store for ONIOM electrostatic embedding
from .charges import ChargeStore, ChargeStoreError
