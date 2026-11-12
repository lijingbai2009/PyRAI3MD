"""
======================================================================
Optimizer package for geometry optimization algorithms.

This package provides optimization algorithms independent of the
workflow and calculator systems. The optimizer determines the next
geometry based on optimization algorithms and convergence criteria,
operating exclusively on PyRAIState.

Available optimizers:
- MinimaOptimizer: BFGS + RFO + Trust Radius (energy minima)
- GDIISOptimizer: Gradient DIIS (alternative strategy)
- TransitionStateOptimizer: first-order saddle point by mode following
- MECIOptimizer: minimum energy conical intersection
- MECPOptimizer: minimum energy crossing point between spin surfaces
- MEPOptimizer: minimum energy path (registered, not implemented: a path
  walks several structures)
- IRCOptimizer: intrinsic reaction coordinate (the same walk, seeded one
  radius along the imaginary mode of its saddle)
- NEBOptimizer: nudged elastic band (registered, not implemented, same
  reason)

The registry keeps every name the framework is shaped for, because it is
also the list a reader consults to find out what is coming. What a run may
ask for by name is the smaller set `implemented_optimizers()` returns: a
registered placeholder is refused while the input is still cheap to change,
instead of after the first evaluation of a run that had nothing to step on.

The path family (mep, irc, neb) is registered here and implemented outside
this registry's driving contract: `resolve_workflow` routes those names to
the Path workflow (`workflow/path.py`), which owns the multi-structure loop
and drives the optimizer through `optimizer/pathbase.py` instead of through
`Optimizer.run`. Every path class says `is_path = True`, which is how
`Optimize` refuses one that reaches it by mistake.

Skill-internal optimization (control.driver != 'main') never
reaches this registry: the planning layer keeps the jobtype on the
skill and resolve_workflow returns the SingleShot workflow instead.

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .base import Optimizer
from .minima import MinimaOptimizer
from .gdiis import GDIISOptimizer
from .ts import TransitionStateOptimizer
from .meci import MECIOptimizer
from .mecp import MECPOptimizer
from .mep import MEPOptimizer
from .irc import IRCOptimizer
from .neb import NEBOptimizer

# Registry of all available optimizers (for external queries).
OPTIMIZER_REGISTRY = {
    'minima': MinimaOptimizer,
    'gdiis': GDIISOptimizer,
    'ts': TransitionStateOptimizer,
    'meci': MECIOptimizer,
    'mecp': MECPOptimizer,
    'mep': MEPOptimizer,
    'irc': IRCOptimizer,
    'neb': NEBOptimizer,
}


def implemented_optimizers():
    """The optimizer names a run may ask for, in registry order.

    A placeholder class answers False on `is_implemented`; the check lives on
    the class so a new optimizer carries its own statement rather than leaving
    a second list to remember it.
    """
    return [name for name, cls in OPTIMIZER_REGISTRY.items()
            if getattr(cls, 'is_implemented', True)]


def check_declarations():
    """Hold the optimizer classes to `schema/optimizer_spec.py`.

    The planning layer reads the pure table; the classes carry the same
    five declarations for the workflow side. This is the guard that keeps
    the two from drifting.
    """
    from ...schema.optimizer_spec import OPTIMIZER_DECLARATIONS
    for name, cls in OPTIMIZER_REGISTRY.items():
        declared = OPTIMIZER_DECLARATIONS.get(name)
        if declared is None:
            raise AssertionError(
                "optimizer %r has no entry in OPTIMIZER_DECLARATIONS" % name)
        actual = {
            'implemented': bool(getattr(cls, 'is_implemented', True)),
            'is_path': bool(getattr(cls, 'is_path', False)),
            'needs_initial_hessian': bool(
                getattr(cls, 'needs_initial_hessian', False)),
            'uses_derivative_coupling': bool(
                getattr(cls, 'uses_derivative_coupling', False)),
            'uses_state_pair': bool(
                getattr(cls, 'uses_state_pair', False)),
        }
        if actual != declared:
            raise AssertionError(
                "optimizer %r declares %r but the table says %r"
                % (name, actual, declared))


def get_optimizer(name, **kwargs):
    """Create an optimizer instance by name.

    Framework optimizers are resolved through OPTIMIZER_REGISTRY.
    Skill-internal optimization (control.driver != 'main') is
    handled by the SingleShot workflow and never reaches this
    factory.

    Parameters
    ----------
    name : str
        Optimizer name (e.g. 'minima', 'gdiis').
    **kwargs
        Keyword arguments passed to the optimizer constructor.

    Returns
    -------
    Optimizer
        A new optimizer instance.

    Raises
    ------
    ValueError
        If the optimizer name is not recognized.
    """
    if name in OPTIMIZER_REGISTRY:
        return OPTIMIZER_REGISTRY[name](**kwargs)
    raise ValueError(
        "Unknown optimizer: %s. Available: %s"
        % (name, list(OPTIMIZER_REGISTRY.keys())))


__all__ = [
    'Optimizer',
    'MinimaOptimizer',
    'GDIISOptimizer',
    'TransitionStateOptimizer',
    'MECIOptimizer',
    'MECPOptimizer',
    'MEPOptimizer',
    'IRCOptimizer',
    'NEBOptimizer',
    'get_optimizer',
    'implemented_optimizers',
    'check_declarations',
    'OPTIMIZER_REGISTRY',
]
