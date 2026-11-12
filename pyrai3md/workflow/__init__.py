"""
======================================================================
Workflow layer for PyRAI3MD.

This package provides high-level workflow orchestrators that manage
multi-step simulation processes. Workflows call Calculator instances
for computation and update PyRAIState for state progression.

The Workflow layer determines HOW to execute. It never performs
planning operations (resolving input, deciding capabilities,
or querying Registry).

execute(state) is the single entry point: it builds (root runtime)
or verifies (worker runtime) the execution graph, freezes it into
state.execution_graph, constructs the Calculator and the resolved
Workflow, and runs it. The execution mode of every node is carried
by the graph (multitask/remotetask) and applied at construction;
the workflow creates its own local_executor from self.calculator.

PyRAIState is the sole communication object throughout.

Available workflows (one module per class at this level):
- Optimize: Geometry optimization
- AIMD: Ab initio molecular dynamics
- Scan: Potential energy surface scan
- AdaptiveSampling: round-based adaptive sampling (data/train/MD/screen/reference)
- GridSearch: hyperparameter grid scan (batch of train calls)
- Train: one model-training run, settled by the record it returned
- Prediction: one test-set evaluation, settled by the record it returned
- Hessian (registered as NumericalHessian): force constants, by finite
  difference or from the backend, plus the vibrational and thermodynamic
  reading of them; workflow/frequency.py is the analysis library it calls
- SingleShot: single-shot execution. It serves the skill-driven jobs
  (control.driver != 'main', where the skill plans the whole jobtype
  internally) and the plain property jobtypes.

The execution components are not workflows and live in named
subpackages: records/ (what a run leaves behind: trajectory/deviation/
ml/output/diagnostics), grid/ (the plan and the read-back of a
hyperparameter scan), multiscale/ (regions/composite/charges) and
dispatch/ (the runtime and term axes).

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .base import Workflow
from .records.trajectory import TrajectoryResult, TrajectorySet
from .records.deviation import (
    DEVIATION_KEYS, fetch_deviations, merge_deviations, judge_deviations,
)
from .optimize import Optimize
from .path import Path
from .md import AIMD
from .scan import Scan
from .hessian import Hessian, NumericalHessian
from .adaptive import AdaptiveSampling
from .search import GridSearch
from .single import SingleShot
from .train import Train
from .prediction import Prediction
from .calculator import Calculator, adapt_skill_result, resolve_calculator
from .multiscale.composite import (
    aggregate_results, validate_plan, assemble_results, TaskOutcome,
    invalidate_task_result,
)
from .multiscale.regions import (
    materialize_region, project_gradient, validate_region)
from .dispatch.terms import TaskExecutor

# Registry of all available workflow types (for external queries and
# graph construction; injected into the ExecutionGraphBuilder so the
# dispatch package never imports the workflow package).
#
# 'train' and 'prediction' are single-call jobtypes with their own
# classes (workflow/train.py, workflow/prediction.py): the skill owns
# the procedure (data, hyperparameters, model loading), while the node
# name exists so the run can steer that call's dispatch keywords from
# &train / &prediction the way any other workflow node can.
WORKFLOW_REGISTRY = {
    'optimize': Optimize,
    # The path family is asked for as `optimize.method`, and this is the loop
    # `resolve_workflow` hands those three names to. It is registered under its
    # own node name because the flag tables (planner) are keyed on
    # `workflow_cls.name`, and a band's fan-out reads the &optimize section.
    'path': Path,
    'md': AIMD,
    'surface_hopping': AIMD,
    'frequency': NumericalHessian,
    'scan': Scan,
    'train': Train,
    'prediction': Prediction,
    'hessian': NumericalHessian,
    'adaptive': AdaptiveSampling,
    'search': GridSearch,
    # All four force-constant tokens reach the one Hessian workflow; which
    # route the matrix comes from, and whether it is read into modes, is read
    # from the tokens by the workflow itself (workflow/hessian.py).
    'num_hess': NumericalHessian,
    'num_freq': NumericalHessian,
}

# Jobtype token -> Workflow class. Derived from the registry so a new
# workflow is registered once; `path` is reachable only through
# `optimize.method`, not as a jobtype token.
JOBTYPE_MAP = {
    name: workflow_cls
    for name, workflow_cls in WORKFLOW_REGISTRY.items()
    if name != 'path'
}



def resolve_workflow(state):
    """Resolve the workflow class from the execution driver and
    state.jobtype.

    Execution model (fixed at planning through control.driver):
    - driver != 'main': every jobtype runs as a single shot; the
      skill receives the jobtype as-is and plans the job
      internally (SingleShot).
    - driver == 'main': the framework resolves iterative workflows
      from jobtype (JOBTYPE_MAP) or None for single-shot
      calculations.

    Parameters
    ----------
    state : PyRAIState
        The sole communication object.

    Returns
    -------
    type or None
        Workflow class (not instance), or None for framework-side
        single-shot calculations (driver == 'main' only).

    Raises
    ------
    ValueError
        When the input declares the lattice as a variable while the engine
        owns the steps. See the message: the two are mutually exclusive
        routes to a relaxed box, not two switches of one route.
    """
    from ..schema.config import PATH_METHODS
    from ..schema.validate import JOBTYPE_ALIASES

    params = getattr(state, 'optimize_params', None) or {}
    method = str(params.get('opt_method', '') or '').strip().lower()
    if getattr(state, 'driver', 'main') != 'main':
        if method in PATH_METHODS:
            # A path is a sequence of geometries the framework commits, one at
            # a time, and a band is a chain the framework moves as a whole. With
            # the engine owning every step there is nobody to commit them to:
            # the skill would be handed one geometry and asked for one
            # evaluation, which is a single point, not a path.
            raise ValueError(
                "optimize.method %r with driver=%r reaches no consumer: the "
                "engine owns the iteration here, so the framework never commits "
                "a path point. Walk the path with driver: main, or ask the "
                "engine for its own path keyword (a LAMMPS neb or an ORCA IRC "
                "is the engine's grammar, not this one)"
                % (params.get('opt_method'), getattr(state, 'driver', 'main')))
        if params.get('optimize_cell'):
            # `driver` says who iterates, and the skill that answers owns every
            # step: the framework issues one evaluation and takes back the
            # geometry the engine stopped on. The lattice is a variable only for
            # whoever commits each step (the six slots ride on the framework's
            # acceptance test, `workflow/optimize.py`), so a run that hands the
            # steps over and keeps the keyword asks for a box nobody moves - and
            # an engine that relaxes a box does it with its own keyword, in its
            # own grammar, on its own schedule (for LAMMPS, `box_relax` beside
            # `minimize`; CHANGELOG.md #362 and #366).
            raise ValueError(
                "optimize_cell with driver=%r reaches no consumer: the engine "
                "owns the iteration here, so the framework never commits a "
                "lattice. Relax the box on the framework (driver: main), which "
                "differs the energy against its own six lattice rows, or on the "
                "engine, by naming the skill's own cell-relaxing profile and "
                "its box keyword; not by both switches at once"
                % getattr(state, 'driver', 'main'))
        return SingleShot

    from ..schema.validate import JOBTYPE_ALIASES
    jobtype = state.jobtype
    for token in str(jobtype or '').split():
        # A retired alias maps to None and resolves to nothing, which is this
        # function's answer for any token it has no workflow for.
        canonical = JOBTYPE_ALIASES.get(token.lower(), token.lower())
        if canonical == 'optimize' and method in PATH_METHODS:
            # One jobtype, two loops: `opt_method` is what says whether this is
            # a single-structure search or a path, and the input surface stays
            # the one every optimize run already writes.
            return Path
        if canonical in JOBTYPE_MAP:
            return JOBTYPE_MAP[canonical]
    return None


def _graph_node_flags(graph, node_type):
    """Extract the execution flags of one node type from the graph.

    Parameters
    ----------
    graph : list of dict
        Frozen execution graph.
    node_type : str
        'workflow' or 'base'.

    Returns
    -------
    tuple(bool, bool, int)
        (multitask, remotetask, multipool) of the node (local
        defaults when the node is absent).
    """
    for node in graph or []:
        if node.get('node') == node_type:
            return (bool(node.get('multitask')),
                    bool(node.get('remotetask')),
                    max(1, int(node.get('multipool') or 1)))
    return False, False, 1


def _iter_graph_nodes(graph):
    """Yield every node of a frozen graph, including stage sub-nodes.

    A single-stage workflow keeps its base nodes at the top level; a
    multi-stage workflow (AdaptiveSampling) nests them under
    node['stages']. Both shapes are one graph, so the flag collectors
    below walk the same tree instead of assuming the first shape.

    Parameters
    ----------
    graph : list of dict or None
        Frozen execution graph (top level or one stage list).

    Yields
    ------
    dict
        Node descriptors in depth-first order.
    """
    for node in graph or []:
        if not isinstance(node, dict):
            continue
        yield node
        stages = node.get('stages')
        if stages:
            for child in _iter_graph_nodes(stages):
                yield child


def _graph_base_nodes(graph):
    """Collect the base node of every backend role.

    Parameters
    ----------
    graph : list of dict
        Frozen execution graph.

    Returns
    -------
    dict
        {role: base node descriptor}. Merging is by role only: two
        roles never share a node and their flags are never combined.
        A role that appears twice keeps the later node (the planner
        emits one shape per graph; the rule only guards a malformed
        graph from silently borrowing another role's numbers).
    """
    nodes = {}
    for node in _iter_graph_nodes(graph):
        if node.get('node') == 'base' and node.get('role') is not None:
            nodes[node.get('role')] = node
    return nodes


def _graph_base_flags(graph):
    """Collect the base-node flags of every backend role.

    Parameters
    ----------
    graph : list of dict
        Frozen execution graph.

    Returns
    -------
    dict
        {role: (multitask, remotetask, multipool)}; roles without a
        base node get the local defaults.
    """
    flags = {}
    for role, node in _graph_base_nodes(graph).items():
        flags[role] = (
            bool(node.get('multitask')),
            bool(node.get('remotetask')),
            max(1, int(node.get('multipool') or 1)))
    return flags


def _graph_backend_execution(graph):
    """Collect the backend-terms execution descriptor of every role.

    The descriptor (r3b) is the finer-grained execution contract of
    one backend role: how its individual terms run (serial in-process,
    local thread pool, or remote per-term workers). It is stamped on
    the role base node and travels with the frozen graph.

    Parameters
    ----------
    graph : list of dict
        Frozen execution graph.

    Returns
    -------
    dict
        {role: backend_execution descriptor}; roles without a base
        node descriptor are absent (the Calculator then keeps serial
        in-process execution). Worker graphs stamp a single
        node='calculator' serial descriptor and are intentionally not
        collected here: a worker never fans its terms out.
    """
    descriptors = {}
    for role, node in _graph_base_nodes(graph).items():
        descriptor = node.get('backend_execution')
        if descriptor is not None:
            descriptors[role] = descriptor
    return descriptors



def execute(state, runtime_idx=None):
    """Unified execution entry point.

    Execution stages (Draft v3):
    1. Graph construction: the root runtime builds the execution
       graph (jobtype -> primary workflow -> modularized keyword
       overrides) and freezes it into state.execution_graph. Worker
       runtimes (resolve_execution_graph=False) never re-plan; they
       only verify the parent-frozen graph against the universal
       rules.
    2. Construction: the Calculator and the resolved Workflow are
       created with the node flags carried by the frozen graph.
    3. Execution: workflow.run() (or calculator.run() for
       single-shot jobs) calls self.dispatch(); the workflow
       creates its own local_executor from self.calculator.

    PyRAIState is the sole communication object.

    Parameters
    ----------
    state : PyRAIState
        Fully initialized state with the static task queue (no
        backend instances yet).
    runtime_idx : str or None
        Runtime execution suffix provided by the Workflow layer
        (e.g. 'replica_3', 'traj_8'); None for single-shot runs.

    Returns
    -------
    Result or PyRAIState
        Calculation results.
    """
    from .dispatch.planner import ExecutionGraphBuilder

    # A read run extracts labels from finished artifacts and returns
    # them. It builds no calculator, no graph and no backend, and the
    # caller owns every product (this module writes none).
    from .read import is_read_jobtype, read_calculations
    if is_read_jobtype(state):
        return read_calculations(state)

    workflow_cls = resolve_workflow(state)

    # 1. Graph construction (root) or verification (worker).
    graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
        state, workflow_cls)

    # 1b. Pre-initialization data preparation. A workflow may rewrite
    # its task configuration before any backend exists; the train
    # workflow expands a permutation map here, so the skill is handed
    # frames rather than a map.
    if workflow_cls is not None:
        workflow_cls.prepare(state)

    # 2. Construction with the frozen node flags (multitask,
    # remotetask and the node's own multipool pool size). The
    # calculator consumes the base node of its active role; the
    # per-role base flags travel with it so switch_role() applies
    # the other role's node flags.
    base_flags = _graph_base_flags(graph)
    backend_execution = _graph_backend_execution(graph)
    # A child runtime may carry the backend role it was dispatched for
    # (RuntimeState.active_role); roots and legacy checkpoints default to
    # theory. The role's own base node supplies the flags, so the
    # calculator starts with the triple switch_role() would apply.
    runtime = getattr(state, 'runtime', None)
    initial_role = getattr(runtime, 'active_role', None) or 'theory'
    (calc_multitask, calc_remotetask,
     calc_multipool) = base_flags.get(initial_role, (False, False, 1))
    calculator = Calculator(state, runtime_idx=runtime_idx,
                            initial_role=initial_role,
                            multitask=calc_multitask,
                            remotetask=calc_remotetask,
                            multipool=calc_multipool,
                            base_flags=base_flags,
                            backend_execution=backend_execution)
    # A workflow that never evaluates in this process (needs_backends
    # False) skips initialize(); the root runtime still needs a runtime
    # record for its own progress, so create the default one here.
    #
    # What this call built, this call releases. A backend can own a
    # resident engine process and a calculation directory, and a batch
    # runs its units one after another in this process whenever the pool
    # has a single worker, so a unit that leaves them behind hands the
    # next unit a machine still holding the previous one's engines. The
    # MD loop already released its own backends before returning; close()
    # is idempotent, so this is that rule generalized to every unit rather
    # than a second release.
    try:
        if getattr(workflow_cls, 'needs_backends', True):
            calculator.initialize()
        elif getattr(state, 'runtime', None) is None:
            from ..data.runtime.state import RuntimeState
            state.runtime = RuntimeState()

        if workflow_cls is not None:
            (wf_multitask, wf_remotetask,
             wf_multipool) = _graph_node_flags(graph, 'workflow')
            # Workflow.run() creates the local_executor from
            # self.calculator when needed; execute() does not
            # inject it. executor_spec still points to execute
            # for remote/multiprocessing child processes.
            workflow = workflow_cls(calculator=calculator,
                                    multitask=wf_multitask,
                                    remotetask=wf_remotetask,
                                    multipool=wf_multipool,
                                    **workflow_cls.state_kwargs(state))
            return workflow.run()

        # Single-shot calculation: a unit list of length 1.
        return calculator.run()
    finally:
        calculator.close()
