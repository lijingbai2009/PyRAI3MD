"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Analytic model registry: maps method names to analytic model classes.

"""

from .base import AnalyticModel
from .lz1 import Lz1
from .lz2 import Lz2
from .lz3 import Lz3
from .tully1 import Tully1
from .tully2 import Tully2
from .tully3 import Tully3

MODEL_REGISTRY = {
    'tully1': Tully1,
    'tully2': Tully2,
    'tully3': Tully3,
    'lz1': Lz1,
    'lz2': Lz2,
    'lz3': Lz3,
}

__all__ = [
    'AnalyticModel',
    'Lz1',
    'Lz2',
    'Lz3',
    'Tully1',
    'Tully2',
    'Tully3',
    'MODEL_REGISTRY',
    'get_model',
]


def get_model(name, params=None):
    """Instantiate an analytic model by method name.

    Parameters
    ----------
    name : str
        Model / method name, one of tully1, tully2, tully3, lz1,
        lz2, lz3.
    params : dict, optional
        Parameter overrides keyed by the model parameter names
        (tully_a / tully_b / tully_c / tully_d / tully_e / lz_a /
        lz_b); None values keep the standard parameters. Entries not
        declared by the model (e.g. tully_e for tully1/tully3, lz_a
        for the Tully set) are silently ignored: the config schema
        forwards every key uniformly for every method.

    Returns
    -------
    AnalyticModel
        Model instance ready for hamiltonian / hamiltonian_gradient
        evaluation.

    Raises
    ------
    ValueError
        If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            "Unknown QDM model '%s'; available: %s"
            % (name, sorted(MODEL_REGISTRY)))
    model_cls = MODEL_REGISTRY[name]
    declared = set(model_cls._parameter_names())
    filtered = {key: value for key, value in (params or {}).items()
                if key in declared}
    return model_cls(params=filtered)
