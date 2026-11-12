"""
======================================================================
Pure declarations of the geometry-search algorithms.

The planning layer has to answer three questions about an
`optimize.method` before any optimizer object exists:

- is the name registered and implemented;
- does the search need an initial Hessian;
- does it span its branching plane on the derivative coupling.

Those answers used to be read from the optimizer classes themselves,
which made planning import the workflow layer. They are plain data, so
they live here; `workflow/optimizer/__init__.py::check_declarations()`
holds the classes to this table and the planning layer reads only the
table.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

_DEFAULTS = {
    'implemented': True,
    'is_path': False,
    'needs_initial_hessian': False,
    'uses_derivative_coupling': False,
    'uses_state_pair': False,
}

# Registry order is the workflow-side registry order. A partial entry
# inherits every field from _DEFAULTS.
_DECLARATIONS = {
    'minima': {},
    'gdiis': {},
    'ts': {'needs_initial_hessian': True},
    'meci': {'uses_derivative_coupling': True, 'uses_state_pair': True},
    'mecp': {'uses_derivative_coupling': False, 'uses_state_pair': True},
    'mep': {'is_path': True},
    'irc': {'is_path': True, 'needs_initial_hessian': True},
    'neb': {'is_path': True},
}

OPTIMIZER_DECLARATIONS = {
    name: dict(_DEFAULTS, **spec)
    for name, spec in _DECLARATIONS.items()
}


def implemented_optimizers():
    """The optimizer names a run may ask for, in registry order."""
    return [name for name, spec in OPTIMIZER_DECLARATIONS.items()
            if spec['implemented']]
