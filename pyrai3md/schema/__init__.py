"""
======================================================================
This module is the public API for the PyRAI3MD schema package.
It exposes configuration models, validation/normalization functions,
adapter functions, and reference data used throughout the framework.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .config import (
    Control,
    Adaptive,
    System,
    Dynamics,
    Properties,
    Calculation,
    Environment,
    PyRAIConfig,
    Config,  # Backward compatibility alias
    # The system constraint block. Its models are declared with `System`, the
    # pass that turns one into a plan is `normalize.build_plan`, and the classic
    # `&constraint` text is read by `adapter`. One import reaches all three,
    # which is what the framework and its tests ask for.
    SystemConstraintConfig,
    UnitsInput,
    GroupInput,
    CVInput,
    BiasInput,
    WallInput,
    FixInput,
    HardConstraintInput,
    MetadynamicsInput,
    ConstraintInputError,
    UnsupportedCapabilityError,
)
from .validate import (validate, validate_task, validate_system,
                       validate_electronic, validate_dynamics)
from .normalize import (normalize, compute_derived_quantities, build_plan,
                        normalize_config, config_from_mapping)
from .adapter import (build_config, read_json)
from .reference import (
    PERIODIC_TABLE,
    ELEMENT_SYMBOLS,
    FULL_NAMES,
    ATOMIC_MASS,
    VDW_RADII,
    COVALENT_RADII,
    STATE_LABEL,
)
# One set of unit constants, imported where they are defined. `reference.py`
# stays a pure-data module with no import statement, and this package keeps
# the public `pyrai3md.schema.<NAME>` spelling.
from ..utilities.units import (
    BOHR_TO_ANGSTROM,
    ANGSTROM_TO_BOHR,
    AMU_TO_AU,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    HARTREE_TO_KCAL,
    KCAL_TO_HARTREE,
    CM_TO_HARTREE,
    HARTREE_TO_CM,
)
