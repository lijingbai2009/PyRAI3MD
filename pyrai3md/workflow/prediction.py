"""
======================================================================
This module provides the Prediction workflow: a test-set evaluation.

A prediction request serves an already trained model over a file of
geometries that carry reference values and reports how far the surface
was from them. The record it returns is the skill's: the ensemble spread,
the scaler and the label-unit rule all live with the model folder, so
every error number is computed there and this workflow only receives and
settles it (`records/ml.py`), while `utilities/print_results.py` prints
what came back.

The jobtype is served by the skill's normal path -- `resolve_profile`
turns `prediction` into the family's serving profile -- and the request
is distinguished by `pred_data`: a serving call that carries a test file
evaluates that file instead of answering the one geometry in the state.
The framework therefore asks for no property of its own here, which is
why the run reports no state, velocity or kinetic energy either.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .train import MLSingleShot


class Prediction(MLSingleShot):
    """Model-prediction workflow: one test-set pass, settled by its record.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator executing the single evaluation call.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask, multipool,
        local_executor, executor_spec).
    """

    name = 'prediction'

    def __init__(self, calculator=None, **kwargs):
        super().__init__(calculator=calculator, **kwargs)
