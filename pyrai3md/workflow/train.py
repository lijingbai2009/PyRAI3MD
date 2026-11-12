"""
======================================================================
This module provides the Train workflow: one model-training run.

A training request is one call of the skill backend under a profile
whose name ends in `_train`. The network is fitted inside the skill; the
framework's part is to make that call once, keep the record it returns,
and say honestly whether the record is there -- which is what
`records/ml.py` settles and `utilities/print_results.py` prints.

What a training run is not: it is not an iteration the framework drives
(no geometry is evaluated, no convergence test is applied to the fit),
and its validation error is not a pass/fail threshold. The comparative
question -- which of several trainings was best -- belongs to the grid
scan (`workflow/search.py`), and continuing an existing surface belongs
to the skill's own `train_mode` switch.

`&train` carries this node's dispatch keywords, so a long fit can be
handed to the cluster with `remotetask` the way any other workflow node
can; before this class existed the jobtype rode `SingleShot` and could
only take that class's defaults.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from ..kernel.logger import get_logger
from .base import Workflow
from .records.ml import settle_ml_outcome

_DISABLED_MAPS = frozenset(('', 'no', 'none'))


def _state_symbols(state, natom):
    """The state's atom labels, when they describe the same atom count."""
    atoms = getattr(state, 'atoms', None)
    if atoms is None:
        return None
    atoms = list(atoms)
    return atoms if len(atoms) == int(natom) else None


def prepare_training_data(state):
    """Expand each training task's dataset by the ``&train`` permutation map.

    Runs before the Calculator initializes a skill backend, so the skill
    receives frames and never a map. A request without a map is left
    exactly as it was; an already-expanded inline dataset is left alone,
    which keeps repeated calls idempotent.

    Parameters
    ----------
    state : PyRAIState
        State whose theory/reference task queue still carries the raw
        skill configuration.
    """
    pmd = getattr(state, 'pmd', None)
    settings = getattr(pmd, 'train', None) if pmd is not None else None
    map_path = getattr(settings, 'permutation_map', '') if settings is not None else ''
    if str(map_path).strip().lower() in _DISABLED_MAPS:
        return

    from ..training_data import (TrainingData, expand_permutations, load,
                                 read_permutation_map)

    logger = get_logger(__name__)
    for _category, task in state.iter_tasks():
        skill = (task.get('config') or {}).get('skill_config') or {}
        if not isinstance(skill, dict):
            continue
        profile = str(skill.get('profile', '')).lower()
        if not profile.endswith('_train'):
            continue

        inline = skill.get('train_dataset')
        if isinstance(inline, dict) and inline:
            source = str(skill.get('train_data') or '<inline>')
            dataset = TrainingData.from_mapping(inline, source=source)
            if dataset.permutation_id is not None:
                continue
        else:
            source = str(skill.get('train_data') or '')
            if not source:
                # The skill schema reports the missing source; there is
                # nothing here to expand.
                continue
            dataset = load(source)

        symbols = (dataset.symbols[0] if dataset.symbols is not None
                   else _state_symbols(state, dataset.natom))
        maps = read_permutation_map(map_path, dataset.natom, symbols)
        expanded = expand_permutations(dataset, maps)
        if expanded is dataset:
            continue

        payload = expanded.to_payload()
        payload['source'] = source
        skill['train_dataset'] = payload
        skill['train_data'] = ''
        logger.info(
            "Permutation map %s expanded the training set for '%s': %d "
            "frame(s) from %d source frame(s).",
            map_path, skill.get('profile'), expanded.nframe, dataset.nframe)


class MLSingleShot(Workflow):
    """Single calculator call whose outcome is settled from its record.

    Shared body of the two machine-learning single-call nodes
    (Train and Prediction, which declare their jobtype by setting
    `name`). The jobtype is what picks the required record channels
    (`records/ml.py:REQUIRED_CHANNELS`), so the subclasses differ only
    in that token and in their documentation.

    This is deliberately not a SingleShot subclass: `driver != 'main'`
    routes every jobtype, ML ones included, to SingleShot, and that
    path must keep its plain completion instead of applying an ML
    verdict the skill-driven run did not ask for.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator executing the single skill call.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask, multipool,
        local_executor, executor_spec).
    """

    workflow_kind = 'single'

    def __init__(self, calculator=None, **kwargs):
        super().__init__(name=self.name, calculator=calculator, **kwargs)

    def run(self):
        """Execute the calculator once and settle the outcome.

        Returns
        -------
        PyRAIState or Result
            The state when a calculator state exists, otherwise the raw
            result.
        """
        if self.local_executor is None:
            calculator = self.calculator

            def _run_call(unit_state, runtime_idx=None):
                del unit_state, runtime_idx  # calculator owns the live state
                result = calculator.run()
                state = calculator.get_state()
                if state is None:
                    return result

                # Standard step bookkeeping (shared with the single-shot
                # and iterative paths): record the result. The run-log
                # result block is written once by the pmd tail for all
                # non-iterative workflows, so it is not duplicated here.
                state.record_step_result()
                settle_ml_outcome(state, result)
                return state

            self.local_executor = _run_call
        outcomes = self.dispatch()
        return outcomes[0]

    def __repr__(self):
        return "%s(single call, outcome settled from its record)" % (
            type(self).__name__,)


class Train(MLSingleShot):
    """Model training workflow: one fit, settled by its record.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator executing the single training call.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask, multipool,
        local_executor, executor_spec).
    """

    name = 'train'

    def __init__(self, calculator=None, **kwargs):
        super().__init__(calculator=calculator, **kwargs)

    @classmethod
    def prepare(cls, state):
        """Expand the training data before any skill backend exists."""
        prepare_training_data(state)
