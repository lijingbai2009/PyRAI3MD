"""
======================================================================
This module provides the Workflow base class for the workflow layer.

The inheritance chain is unique and simple:

    Dispatcher <- Workflow (this module) <- concrete workflows
    Dispatcher <- Calculator

Workflow is a thin Dispatcher wrapper around a Calculator: its run()
executes the calculator once through self.dispatch() (unified
execution). No iteration, no state evolution in the base class.
Iterative loops (AIMD, optimize) and batch algorithms
(generate_batch/collect/postprocess) are implemented by the concrete
workflow modules themselves, which forward their units to
self.dispatch() and stay unaware of the execution mode.

Workflows communicate with the calculator exclusively through
calculator.get_state() and calculator.update(); the calculator is
the sole owner of the PyRAIState.

Workflows do NOT call external programs directly; they only use
Calculators for computation, following the design rule that all
external program access goes through the Skill Python API.

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .dispatch.dispatcher import Dispatcher


class Workflow(Dispatcher):
    """Simple workflow base: wraps a calculator and runs it once.

    run() executes the received calculator through self.dispatch()
    (unified execution logic shared with Calculator). No iteration,
    no evolution in the base class. Concrete workflows requiring
    iterative loops or batch fan-out override run() and forward
    their units to self.dispatch() through the inherited pipeline.

    Class attributes declare the execution defaults consumed by the
    ExecutionGraphBuilder:
    - multitask / remotetask: inherited from Dispatcher.
    - workflow_kind: 'single', 'iterative' or 'batch' (metadata).

    Parameters
    ----------
    name : str
        Workflow identifier (e.g. 'optimize', 'md').
    calculator : Calculator or None
        The calculator used for single-request computations.
    maxiter : int
        Maximum number of iterations/steps (consumed by concrete
        iterative workflows).
    state : PyRAIState or None
        Runtime state; defaults to the calculator state.
    multitask : bool or None
        Execution flag override (graph keywords).
    remotetask : bool or None
        Execution flag override (graph keywords).
    multipool : int or None
        Multitask worker pool size (the multipool keyword of the
        workflow input section); None keeps the class default.
    local_executor : callable or None
        Injected in-process execution entry (runs the calculator).
    executor_spec : tuple or None
        Importable spec of the execution entry for multiprocessing
        workers.
    """

    # Execution-graph node name: the token the planner keys
    # WORKFLOW_SECTIONS, WORKFLOW_NODE_DECLARATIONS and
    # WORKFLOW_BASE_CHAINS on. Every concrete workflow declares it
    # explicitly (a name derived from the class would silently miss
    # the tables whenever the two spellings differ); the verification
    # suite guards that no registry class is left unset.
    name = ''

    workflow_kind = 'single'

    # Whether the root execute() must initialize the calculator backends
    # before run(). A workflow that only orchestrates child runtimes
    # (adaptive sampling) sets this False: its own process never evaluates
    # a term, and the first-round model directory may not exist yet.
    needs_backends = True

    @classmethod
    def prepare(cls, state):
        """Pre-initialization hook for a workflow's input data.

        Called by :func:`workflow.execute` after the execution graph is
        frozen and before the Calculator initializes any skill backend.
        A workflow that has to rewrite the task configuration (the train
        workflow expands its dataset here) overrides this; the default
        does nothing.

        Parameters
        ----------
        state : PyRAIState
            State whose task queue is about to be handed to backends.
        """
        return None

    @classmethod
    def state_kwargs(cls, state):
        """Constructor kwargs this workflow takes from the input state.

        A workflow whose loop parameters come from its input section
        overrides this (AIMD reads ``&md``, AdaptiveSampling reads
        ``&adaptive``). The runner calls it once, so adding such a
        workflow does not add a branch to the runner. The default is
        no extra kwargs.

        Parameters
        ----------
        state : PyRAIState
            State carrying the parsed configuration.

        Returns
        -------
        dict
            Extra constructor keyword arguments.
        """
        return {}

    def __init__(self, name='', calculator=None, maxiter=100,
                 state=None, multitask=None, remotetask=None,
                 multipool=None,
                 local_executor=None, executor_spec=None):
        if state is None and calculator is not None:
            state = calculator.get_state()
        super().__init__(state=state, multitask=multitask,
                         remotetask=remotetask,
                         multipool=multipool,
                         local_executor=local_executor,
                         executor_spec=executor_spec)
        self.name = str(name)
        self.calculator = calculator
        self.maxiter = int(maxiter)

    def run(self):
        """Run the received calculator once through self.dispatch().

        Unified execution: the workflow dispatches the calculator
        as a single unit through the Launcher/Scheduler pipeline.
        The local_executor runs the calculator directly (no
        recursive re-entry into execute()).

        Returns
        -------
        PyRAIState or Result
            The state when a calculator state exists (iterative
            convention), otherwise the raw Result.
        """
        if self.local_executor is None:
            calculator = self.calculator

            def _run_calc(unit_state, runtime_idx=None):
                del unit_state, runtime_idx  # calculator owns the live state
                result = calculator.run()
                state = calculator.get_state()
                if (state is not None
                        and getattr(state, 'runtime', None) is not None):
                    state.runtime.mark_completed()
                    return state
                return result

            self.local_executor = _run_calc
        outcomes = self.dispatch()
        return outcomes[0]

    def _check_convergence(self, result):
        """Check whether the workflow has converged.

        Concrete iterative workflows override this.

        Parameters
        ----------
        result : Result
            The latest computation result.

        Returns
        -------
        bool
            True if convergence criteria are met.
        """
        return False

    def __repr__(self):
        return "Workflow(name='%s', maxiter=%d)" % (self.name, self.maxiter)
