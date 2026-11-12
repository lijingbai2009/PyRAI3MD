# PyRAI3MD Workflow Development Guide

Last updated: 2026-11-12 19:20 (13.1 records the term-script shell replay and the Slurm poll deadline. Earlier: 2026-11-12 15:50 (the planning→workflow exceptions were removed: planning reads `schema/optimizer_spec.py` and `data/constraint.py`; Sections 13 and Rule 8 updated; the composite registry is marked retired in Sections 8.1 and Rule 10. Earlier: 2026-11-09 14:20 (MLSingleShot moved from single.py to train.py; the registered workflow→planning import exception recorded in Section 13; the constructor kwargs hook is now `Workflow.state_kwargs`. The rest was not re-audited. Earlier: 2026-10-24 07:40 (ml_single.py retired: MLSingleShot moved into single.py as a separate Workflow subclass; Train/Prediction keep their node files, class identity and section routing. Earlier 2026-10-23 23:52 (the dispatch section now documents the `active_role` unit channel: `RuntimeState.active_role`, single-role child materialization and inactive-queue pruning; the graph section records that the base-node collectors recurse into `stage` sub-nodes. Earlier 2026-10-22 22:00 (two module paths in the Result and dispatch sections now read `data/runtime/result.py` and `workflow/records/ml.py`; nothing else was re-read. Earlier, 19:10 (the workflow tree and the class table now carry the path family: `path.py` / `Path`, routed by `optimize.method` with a walker driven in place and a band through `dispatch(units=batch)`. Before it, 2026-10-15 16:29 (section 18.5 was rewritten against `service/utils.py::build_calcdir` for the `project` substitution; the MD step list was checked against the code for the `apply_pbc` move to `data/geometry.py`; the rest of the file was not re-audited)

---

> **Multi-center unification (refactor R2-R6, 2026-09-08).** The MD loop keeps
> its single-center execution strategy and data computation/save/read order;
> R6 only extends it functionally to multiple centers (CHANGELOG #204,
> `dev/MULTICENTER_DEVELOPMENT.md` §10 covers the history / nuclear
> integration / energy-ledger design).
> Surface hopping dispatches through `workflow/integrator/electronic.py`
> (`ElectronicStepper`: one `ElectronicBinding` per center running the
> UNCHANGED fssh/gsh/lzsh core on a `CenterView`; C=1 is a transparent binding,
> bit-for-bit the legacy path; each center's velocity write-back is masked to
> its own atoms). verlet reads the assembled active-combination gradient
> (`state.active_gradient()` and the frame's `nuclear_sample.active_gradient`),
> never a recorded state index. Per-step target states are center-aware
> (`md.resolve_target_states_centered` -> a per-center map routed by
> `calculator._task_target_states`; `sync()` skips the electronic-state scalars
> for C>1). Trajectory output (`workflow/records/output.py`) gains a center protocol:
> with an environment partition (`plan.n_layer>=2`) the global md.*/sh.* carry
> the whole system while each center writes md.#.*/sh.#.* (# = 1-based) scoped
> to its region; md.energies columns are
> `time|Epot|Ekin|Etot|per-center active states|per-center per-state energies`
> and md.pop expands per center; a plain single-center run (no region) stays
> byte-identical. `workflow/records/trajectory.py` adds `center_history_step`. See
> `methods/MD_DEVELOPMENT.md` for the loop/output detail.

## 1. Design Objective

PyRAI3MD provides a unified calculation framework for molecular and periodic ab initio dynamics. The calculation architecture separates:

- **Planning** — Determines what should be computed (capability resolution, jobtype parsing). See `PLANNING_DEVELOPMENT.md`.
- **Workflow** — Defines high-level simulation procedures (optimization, AIMD, frequency, scan).
- **Calculator** — High-level functional object that owns state, backend access, and result processing.
- **Skill** — Provides actual computational capability through external programs.

The key design principle is:

> For a single calculation step, PyRAI3MD sends only one calculation request to the same skill and receives a complete result bundle, avoiding repeated external program calls.

For example, an AIMD step requiring Energy, Gradient, NAC, SOC should generate:

```
AIMD workflow
        │
        ▼
Calculator(state) → iterates task queue from state.theory/reference
        │
        ▼
backend.calculate(atoms, coordinates)   (per task, region geometry derived live)
        │
        ▼
Result (all properties in one bundle)
```

Instead of four separate backend calls.

---

## 2. Overall Architecture

### 2.1 Runtime Pipeline

The complete execution pipeline is:

```
Input
    │
    ▼
Interface
    │
    ▼
Schema
    │
    ▼
Planning ──── resolve capabilities, expand task queue
    │
    ▼
build_state(config) ─── creates PyRAIState with task queues
────────────────────────────────────────────
Framework / Skill Boundary
────────────────────────────────────────────
execute(state, runtime_idx=None) ←── unified entry point
    │
    ├── ExecutionGraphBuilder.build() → freezes the execution graph
    │   into state.execution_graph (workers only verify the
    │   parent-frozen graph, resolve_execution_graph=False)
    ├── Creates Calculator(state, runtime_idx) + initialize()
    │   with the base.theory node flags from the graph (the per-role
    │   base flags travel with it so switch_role() applies them)
    ├── resolve_workflow(state) → Workflow class or None
    ├── If workflow: Workflow(calculator=calc).run()
    │   (Workflow creates its own local_executor from calculator)
    └── If single-shot: calculator.run()
    │
    ▼
Result
```

The boundary is defined at PyRAIState:

- Planning terminates after PyRAIState construction, distributing per-skill capabilities via Maestro and expanding composite workflows into task queues.
- `build_state(config)` creates PyRAIState and builds all task queues internally.
- Workflow execution always begins after `build_state()` returns.
- PyRAIState is the only communication protocol between the Framework and every Skill.
- Each backend receives a task-specific config dict (`{'skill_config': <skill flat dict>, 'results': {}}`) as input.

### 2.2 Layer Responsibilities

| Layer | Responsibility |
|-------|---------------|
| Planning | Decide **what** needs to be computed (capabilities + task queue) |
| PyRAIState | Runtime data container (theory/reference task queues) |
| ExecutionGraphBuilder | Decide **how** the runtime executes (freeze the execution graph once) |
| Dispatcher/Launcher/Scheduler | Execute units (multitask x remotetask composition) |
| Workflow | Decide **when** computation occurs (algorithm only) |
| Calculator | Decide **who** performs computation (iterate task queue) |
| Skill Backend | Decide **how** computation is performed |

Every layer has a single responsibility. Workflow controls execution order, never scientific computation or resource scheduling. Calculator is the sole communication layer between Workflow and Skill Backend. `execute(state, runtime_idx=None)` is the unified entry point for all calculations. See Section 11 (Dispatcher and Dispatch Subsystem).
### 2.3 Source Structure

```
pyrai3md/

├── planning/                  # Planning layer (WHAT to compute)
│   ├── resolver.py            #   Capability resolution
│   └── task.py                #   resolve_composite_type + resolve_task_queue
│
├── workflow/                  # Workflow execution layer (HOW to compute)
│   ├── __init__.py            #   Exports Workflow classes, Calculator,
│   │                          #   resolve_workflow, execute,
│   │                          #   WORKFLOW_REGISTRY, JOBTYPE_MAP
│   ├── base.py                #   Workflow(Dispatcher) base class; declares the
│   │                          #   class-level `name` used as the graph node name
│   ├── calculator.py          #   Calculator(Dispatcher): initialize/run/sync,
│   │                          #   state ownership (get_state/update),
│   │                          #   backend role grouping + switching
│   ├── optimize.py            #   Geometry optimization (iterative)
│   ├── path.py                #   Path family: MEP / IRC / NEB behind
│   │                          #     `optimize.method`; a walker driven in
│   │                          #     place, a band through dispatch(units=batch)
│   ├── md.py                  #   Ab initio MD, class AIMD (iterative)
│   ├── hessian.py             #   Hessian: batch of displaced gradients,
│   │                          #     analytical or numerical, plus modes
│   ├── frequency.py           #   vibration/thermo/molden functions it calls
│   ├── scan.py                #   PES scan workflow
│   ├── adaptive.py            #   Adaptive sampling (round-based loop)
│   ├── search.py              #   GridSearch: hyperparameter grid scan (batch)
│   ├── grid/                  #   Scan components: plan / read-back (no class)
│   │   ├── plan.py            #     Skill declaration, points, per-unit stamp
│   │   └── record.py          #     Records, ranking, tables, CSV
│   ├── train.py               #   Train: one model fit, settled by its record
│   ├── prediction.py          #   Prediction: one test-set pass, ditto
│   ├── single.py              #   Single-shot workflow (skill-driven jobs and
│   │                          #   the plain property jobtypes) + the shared
│   │                          #   ML single-call body (MLSingleShot)
│   ├── records/               #   Run records of the MD chain (components)
│   │   ├── trajectory.py      #     TrajectoryResult / TrajectorySet
│   │   ├── deviation.py       #     DEVIATION_KEYS + fetch/merge/judge
│   │   ├── ml.py              #     ML outcome channel (required record channels)
│   │   ├── output.py          #     MdOutputWriter + record_due cadence
│   │   └── diagnostics.py     #     Run-health gates and summary
│   ├── multiscale/            #   Multiscale execution components
│   │   ├── composite.py       #     pure aggregation (aggregate_results)
│   │   │                      #     (aggregate_results) and projection (project)
│   │   ├── regions.py         #     Pure geometric region operations
│   │   └── charges.py         #     Embedding-charge sources and updates
│   └── dispatch/              #   Dispatcher subsystem
│       ├── dispatcher.py      #     Dispatcher base (dispatch() tool method,
│       │                      #     unit materialization, universal rules)
│       ├── launcher.py        #     Launcher: remotetask -> 'local' or slurm_run
│       ├── scheduler.py       #     Scheduler: multitask -> multiprocessing Pool
│       ├── terms.py           #     Term axis: serial/thread executor,
│       │                      #     the skill-owned thread check,
│       │                      #     work_item/outcome
│       ├── term_worker.py     #     Remote per-term worker execution
│       ├── planner.py         #     ExecutionGraphBuilder + tri-state aggregation
│       ├── slurm_run.py       #     Slurm single-task runner
│       └── runtime.py         #     RuntimeDirectoryManager + helpers
│
├── data/                      # Data core (runtime state)
│   ├── core.py                #   PyRAIState (Task Queue) + build_pyrai_config (deprecated) + build_state
│   ├── runtime/               #   RuntimeState, Result, serializer
│   ├── structure/             #   System, Atom, Geometry, Cell, etc.
│   └── dynamics/              #   Dynamics, Ensemble, Constraints, etc.
│
└── skills/                    # Skill plugins (10 skills)
    ├── pyscf_skill/
    ├── molcas_skill/
    ├── bagel_skill/
    ├── orca_skill/
    ├── xtb_skill/
    ├── openqp_skill/
    ├── lammps_skill/
    └── cp2k_skill/
```

---

## 3. Workflow Base Classes

`workflow/base.py` provides a single base class. The inheritance chain is
unique and simple:

```
Dispatcher <- Workflow (base.py) <- concrete workflows
Dispatcher <- Calculator
```

`Workflow(Dispatcher)` is a thin wrapper around a Calculator: its `run()`
runs the received calculator once (no iteration, no evolution). Iterative
loops (AIMD, optimize) and batch algorithms
(`generate_batch/collect/postprocess`) are implemented by the concrete
workflow modules themselves, which forward their units to
`Dispatcher.run()` and stay unaware of the execution mode. Class attributes
declare the execution defaults consumed by the ExecutionGraphBuilder:

- `workflow_kind`: `'single'`, `'iterative'` or `'batch'` (metadata).
- `multitask` / `remotetask`: inherited from Dispatcher (class defaults,
  instance overrides, graph keyword overrides).

### 3.1 Interface

```python
class Workflow(Dispatcher):
    workflow_kind = 'single'

    def __init__(self, name='', calculator=None, maxiter=100,
                 state=None, multitask=None, remotetask=None,
                 local_executor=None, executor_spec=None):
        ...

    def run(self):
        """Run the received calculator once (no iteration)."""

    def _check_convergence(self, result):
        """Check whether the workflow has converged."""
        return False
```

| Class | Kind | Behavior |
|-------|------|----------|
| `Workflow` | single | Base class; `run()` runs the calculator once + `mark_completed()` |
| `AIMD` (md.py) | iterative | Self-owned MD loop (`run -> propagate -> hop -> sync`) |
| `Optimize` | iterative | Self-owned optimization loop |
| `Path` (path.py) | iterative | MEP / IRC / NEB; reached by `resolve_workflow` when `optimize.method` is one of `PATH_METHODS` (no `jobtype` token of its own). A walker is driven in place like `Optimize`; a band is a batch node whose images go through `dispatch(units=batch)` |
| `Train` / `Prediction` (train.py / prediction.py) | single | One calculator call whose outcome is settled from the record the skill returned (see Section 7.6); both share `single.py:MLSingleShot` |
| `SingleShot` (single.py) | single | Skill-driven single shot (`workflow_kind='single'`); returned by the driver branch of `resolve_workflow` for every jobtype of a skill-driven run, and reached by the property jobtypes (see Section 7.7) |
| `Hessian` (registered as `NumericalHessian`) / `GridSearch` | batch | `generate_batch -> self.dispatch(units, child_jobtype) -> collect -> postprocess` |
| `AdaptiveSampling` | sampling | Three-stage chain `md -> base -> train` |

Batch workflows forward their task descriptors to `self.dispatch()`, which
materializes every task into an independent child unit (deep-copied state,
`child_jobtype` applied, worker label stamped, frozen child execution
graph) and executes them through the composed Launcher/Scheduler. The
former abstract base classes (`SingleWorkflow` / `IterativeWorkflow` /
`BatchWorkflow`) were removed; every algorithm lives in its concrete
module.

### 3.2 Workflow Lifecycle

Every iterative workflow follows the same pattern:

1. Access PyRAIState via `self.calculator.get_state()`.
2. Iteratively call `self.calculator.run()` for computation (no arguments).
3. Evolve the framework System through the calculator update interface
   `self.calculator.update({'system': {...}})` (geometry, velocities,
   electronic state fields — any system field).
4. Call `self.calculator.sync()` to propagate the updated system state to
   the task configurations before the next computation.
5. Check convergence or advance step.
6. Return the final PyRAIState.

Result write-back is handled by Calculator (`state.set_result()` → `state.pmd.results`),
not by workflows directly. See `CALCULATOR_DEVELOPMENT.md` for the full loop contract.

---

## 4. Workflow Registry and Resolution

`workflow/__init__.py` provides the workflow resolution mechanism.

### 4.1 WORKFLOW_REGISTRY

Maps workflow names to classes (for external queries):

```python
WORKFLOW_REGISTRY = {
    'optimize': Optimize,
    'md': AIMD,
    'surface_hopping': AIMD,
    'frequency': NumericalHessian,
    'hessian': NumericalHessian,
    'num_freq': NumericalHessian,
    'num_hess': NumericalHessian,
    'scan': Scan,
    'train': SingleShot,
    'prediction': SingleShot,
    'hessian': NumericalHessian,
    'adaptive': AdaptiveSampling,
    'search': Search,
}
```

The registry is injected into the ExecutionGraphBuilder so the dispatch
package never imports the workflow package. `SingleShot` (single.py) is
registered under the `train` and `prediction` tokens: those jobtypes are
single-shot whatever the driver says, because the skill owns the procedure
(§7.6). A driver other than `'main'` reaches the same class through the
driver branch of `resolve_workflow`.

### 4.2 JOBTYPE_MAP

Mapping from jobtype token to Workflow class:

```python
JOBTYPE_MAP = {
    'optimize': Optimize,
    'md': AIMD,
    'surface_hopping': AIMD,
    'frequency': NumericalHessian,
    'hessian': NumericalHessian,
    'num_freq': NumericalHessian,
    'num_hess': NumericalHessian,
    'scan': Scan,
    'train': SingleShot,
    'prediction': SingleShot,
    'adaptive': AdaptiveSampling,
    'search': Search,
}
```

`resolve_workflow(state)` checks `control.driver` first and then uses this
mapping to resolve the workflow class from `state.jobtype`. If no matching
token is found, returns `None` (single-shot calculation). `SingleShot` is
never resolved from `JOBTYPE_MAP`; it is only returned by the driver branch.

### 4.3 resolve_workflow

`resolve_workflow(state)` is the workflow class resolution mechanism:

```python
def resolve_workflow(state):
    """Resolve the workflow class from the execution driver and
    state.jobtype.

    - driver != 'main': every jobtype runs as a single shot; the
      skill receives the jobtype as-is and plans the job
      internally (SingleShot).
    - driver == 'main': the framework resolves iterative workflows
      from jobtype (JOBTYPE_MAP) or None for single-shot
      calculations.
    """
    if getattr(state, 'driver', 'main') != 'main':
        return SingleShot

    jobtype = state.jobtype
    tokens = jobtype.split()
    for token in tokens:
        if token in JOBTYPE_MAP:
            return JOBTYPE_MAP[token]
    return None
```

Key design points:

- `resolve_workflow` returns a Workflow **class** (not an instance).
- The driver branch comes first: `control.driver != 'main'` (`'skill'`
  or a skill name, validated by `_validate_driver` in `data/core.py`)
  always returns `SingleShot`, and the jobtype is handed to the skill
  as-is.
- Returns `None` for single-shot calculations (no matching iterative jobtype, `driver == 'main'` only).
- The caller (`execute()`) is responsible for instantiation.
- No dependency on `parse_jobtype`, `is_basic`, or `is_advanced`.

### 4.4 execute(state, runtime_idx=None)

`execute(state, runtime_idx=None)` is the unified execution entry point with
three execution stages:

```python
def execute(state, runtime_idx=None):
    """Unified execution entry point.

    1. Graph construction: the root runtime builds the execution
       graph and freezes it into state.execution_graph; worker
       runtimes (resolve_execution_graph=False) only verify the
       parent-frozen graph against the universal rules.
    2. Construction: the Calculator and the resolved Workflow are
       created with the node flags carried by the frozen graph.
    3. Execution: workflow.run() (or calculator.run() for
       single-shot jobs) calls self.dispatch(); the workflow
       creates its own local_executor from self.calculator.
    """
    workflow_cls = resolve_workflow(state)
    graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
        state, workflow_cls)
    base_flags = _graph_base_flags(graph)
    backend_execution = _graph_backend_execution(graph)
    initial_role = (getattr(getattr(state, 'runtime', None),
                            'active_role', None) or 'theory')
    (calc_multitask, calc_remotetask,
     calc_multipool) = base_flags.get(initial_role, (False, False, 1))
    calculator = Calculator(state, runtime_idx=runtime_idx,
                            initial_role=initial_role,
                            multitask=calc_multitask,
                            remotetask=calc_remotetask,
                            multipool=calc_multipool,
                            base_flags=base_flags,
                            backend_execution=backend_execution)
    calculator.initialize()
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
                                **_workflow_config_kwargs(
                                    workflow_cls, state))
        return workflow.run()
    return calculator.run()
```

Key design points:

- `execute(state, runtime_idx=None)` is the single entry point for all
  calculations; the signature and external behavior are unchanged.
- Root runtimes build the execution graph exactly once; worker runtimes
  never re-plan (see Section 11 Dispatcher and Dispatch Subsystem).
- `execute()` does not inject `local_executor` into workflows or
  calculators; both create their own in-process executor closures from
  their owned calculator/self in `run()`. The dispatch package never
  imports workflow.
- `runtime_idx` (e.g. `'runtime_3'`) is the backend directory suffix
  provided by the Workflow layer for dispatched child runtimes.
- PyRAIState is the sole communication object throughout.

---

## 5. Calculator

`workflow/calculator.py` contains the Calculator that executes the PyRAIState task queue.

### 5.1 Components

- **Calculator** — High-level functional object executing the task queue (sole runtime state manager)
- **adapt_skill_result(skill_dict)** — converts skill output dict to dict-like Result
- **task_config_results(task)** / **store_task_result(task, result_dict)** — read / fallback-write task config results
- **write_task_results(task, result_dict)** — overwrite task config results (used after projection)
- **SYNC_SYSTEM_FIELDS** — system fields pushed to task configs by `Calculator.sync()`

The Calculator/Composite design (ownership rules, sync mechanism, contribution
registry, result write-back options) is documented in `CALCULATOR_DEVELOPMENT.md`.

### 5.2 Calculator Class

```python
class Calculator(Dispatcher):
    """High-level calculator executing the PyRAIState task queue.

    Inherits Dispatcher: the execution mode is declared through the
    multitask/remotetask booleans and resolved by the unified
    self.dispatch() pipeline. Universal rule 3 keeps the calculator
    always local: the constructor rejects remotetask=True.

    Iterates the active-role task queue from state.iter_tasks(),
    derives each task's atoms/coordinates from the live structure
    using the task region slice indices, executes
    backend.calculate(), and collects the results. Composite tasks
    are aggregated by contribution.

    Parameters
    ----------
    state : PyRAIState or None
        Runtime state (primary API). Tasks come from state.iter_tasks().
    runtime_idx : str or None
        Runtime execution suffix (e.g. 'replica_3', 'traj_8'); None for
        single-shot runs.
    dummy_backend : object or None
        Dummy backend for testing; replaces task backends.
    multitask : bool or None
        Execution flag override (graph keywords); None keeps the class
        default.
    remotetask : bool or None
        Must stay False (universal rule 3); True raises RuntimeError.
    local_executor : callable or None
        Injected in-process execution entry (optional).
    executor_spec : tuple or None
        Importable spec of the execution entry for multiprocessing
        workers.
    """

    def __init__(self, state=None, runtime_idx=None, dummy_backend=None,
                 multitask=None, remotetask=None,
                 local_executor=None, executor_spec=None):
        if remotetask:
            raise RuntimeError(...)   # universal rule 3
        ...

    def run(self, target_states=None):
        """Execute the active role task queue once via self.dispatch().

        A single-point computation is a unit list of length 1, fully
        unified with batch execution. Runs every task of the active
        role in queue order. Each backend writes its results into the
        task config (skill schema); the returned dict is used as
        fallback. The primary task result is projected (cap atoms),
        then all task results are aggregated by the composite module
        (active role) and written back to state.pmd.results through
        state.set_result().
        """

    def sync(self):
        """Synchronize state.pmd.system to every task configuration.

        Pushes SYNC_SYSTEM_FIELDS (atoms, coordinates, velocities,
        charges, electronic state fields, current_state...) into each
        task config. Never modifies calculation, method, or control
        sections.
        """
```

Calculator:

- Inherits `Dispatcher`; `run()` calls `self.dispatch()` with a unit list of
  length 1, and the constructor rejects `remotetask=True` (universal rule 3).
- Iterates the active-role task queue (`state.iter_tasks()`, theory first) — it never inspects the workflow type.
- `_resolve_task_geometry(task)` reads the **live** system data through the state
  pass-through properties (`state.atoms` / `state.coordinates` → `state.pmd.system`)
  and applies the pre-stored `region_indices`, so iterative workflows always pass
  up-to-date geometry. Task configs never store sliced atoms/coordinates/charges.
- `run(target_states=None)` stages the per-step target states forwarded to every
  `backend.calculate()` of that run; existing workflows may call it without arguments.
- `sync()` propagates the workflow-evolved `state.pmd.system` to the task configs.
- Serves as both the single-shot runner and the computation engine for iterative workflows.

### 5.3 Execution Flow

```
Calculator(state)
        │
        ▼
for each task in state.iter_tasks():
    ├── _resolve_task_geometry(task)   ← live pmd.system + region_indices
    ├── backend.calculate(atoms, coordinates)   ← one call per task
    ├── backend writes results → task['config'].results (skill schema)
    └── store_task_result(task, result_dict)    ← fallback if backend didn't write back
        │
        ▼
project(task_config_results(primary_task), primary_task)   ← cap atom screening
write_task_results(primary_task, projected)
        │
        ▼
aggregate_results(state, category=active_role)   ← composite pure function
    ├── aggregate: Σ contribution × property (energy, gradient)
    └── other properties taken from the primary task
        │
        ▼
state.set_result(result.to_dict())     → state.pmd.results (framework schema)
```

`category` restricts the aggregation to one backend role (the calculator's
active role, `'theory'` or `'reference'`); `None` aggregates both queues.

### 5.4 adapt_skill_result

```python
def adapt_skill_result(skill_dict):
    """Convert a skill backend result dict to a Result object."""
    if not isinstance(skill_dict, dict):
        raise TypeError(...)
    return Result(data=skill_dict)
```

Lists and tuples in the dict are automatically converted to numpy arrays by the Result constructor.

### 5.5 Dual Result Write-back

Result write-back is bidirectional:

1. **Skill → task config**: after `calculate()`, the skill backend stores its raw
   results into `task['config'].results` following the skill schema.
2. **Calculator → framework config**: the Calculator calls the composite pure
   function `aggregate_results(state, category=active_role)` (`AGGREGATE_KEYS = ('energy', 'gradient')`
   are summed with contribution coefficients; `category` restricts the aggregation
   to the active backend role, theory/reference) and writes the combined result into
   `state.pmd.results` via `state.set_result()` following the framework schema.
   Composite never writes state itself (option B); `state.result` is a read-only
   Result view of `state.pmd.results`.

### 5.6 Design Notes

Calculator is the sole runtime state manager and a self-contained functional object:

- Owns the state reference (shared mutable `state.pmd`).
- Reads task information from `state.theory/reference` task queues.
- Derives per-task geometry from `state.pmd.system` at execution time.
- Handles result collection and writes results back through `state.set_result()`.
- Synchronizes the evolved framework System to task configs via `sync()`.
- Serves as the default single-shot runner.

Iterative workflows follow the loop contract:
`calculator.run()` → `state.update_system({...})` → `calculator.sync()`.
See `CALCULATOR_DEVELOPMENT.md` for details.

The `resolve_calculator()` factory function is retained for convenience but is not used by the main pipeline. Calculator is now created directly by `execute()`.

---

## 6. Result

`data/runtime/result.py` provides `Result`, a dict-like container that stores all computation outputs uniformly.

### 6.1 Access Modes

- Dict-style access: `result['energy']`, `result['gradient']`
- Attribute-style access (backward compatible): `result.energy`, `result.gradient`
- Standard dict operations: `keys()`, `values()`, `items()`, `get()`, `update()`, `__contains__`, `__len__`, `__iter__`

### 6.2 Automatic Conversion

Lists and tuples are automatically converted to numpy arrays upon storage:

```python
result['gradient'] = [1.0, 2.0, 3.0]  # stored as np.ndarray
```

### 6.3 Serialization

- `to_dict()` — serialize to plain dictionary (numpy arrays → lists)
- `from_dict(data)` — construct from dictionary
- `copy()` — create a deep copy (numpy arrays are independently copied)

---

## 7. Workflow Types

### 7.1 Geometry Optimization (`optimize.py`)

```
Optimize workflow

loop (bounded by min(runtime.total_steps, maxiter, opt_maxit)):
    calculator.run() → state-resolved energy (ns,) + gradient (ns, na, 3)
    optimizer.run(state) → active-state extraction, geometry step,
                           trust radius, convergence check
    calculator.update(state) → propagate new coordinates
```

The optimization algorithm lives in `workflow/optimizer/` and is
selected by the optimize section `method` keyword:

| Method | Algorithm |
|--------|-----------|
| `minima` | Inverse-Hessian BFGS + fixed trust radius (optional RFO via `opt_use_rfo`) |
| `gdiis`  | DIIS extrapolation with BFGS-preconditioned step |

Control keywords (stored with the `opt_` prefix in the schema):

| Keyword | Default | Meaning |
|---------|---------|---------|
| `method` | minima | Optimizer algorithm (minima / gdiis) |
| `maxit` | 100 | Maximum optimization steps (caps the loop) |
| `state_i` | 0 | Electronic state to optimize (0-based) |
| `opt_energy` | 1e-6 | Energy change convergence threshold |
| `opt_grms` | 3e-4 | Gradient RMS convergence threshold |
| `opt_gmax` | 4.5e-4 | Gradient max convergence threshold |
| `opt_drms` | 1.2e-3 | Displacement RMS convergence threshold |
| `opt_dmax` | 1.8e-3 | Displacement max convergence threshold |

When `method` is minima/gdiis the framework expands the task jobtype
to `energy gradient` and runs its own optimization loop; skills only
provide energies and gradients. For excited-state optimization
(`state_i > 0`), the skill method must provide excited-state
gradients (e.g. `tda`/`tddft` for the PySCF skill).

Convergence: ΔE, gradient (max/rms) and displacement (max/rms)
thresholds satisfied simultaneously.

**Skill-level convergence flag**: a skill backend may set `opt_converge`
in its result dict (`True` = converged, `False` = internal optimizer
exhausted steps without converging). The framework consumes this flag
in `workflow/single.py::_finalize_optimize`: `opt_converge is False`
triggers `mark_failed('skill internal optimization did not converge')`
but the process still exits 0 — the runner treats the exit code as
success and records the failed status in the result. This keeps
unconverged optimization as a *result state* rather than a hard error.

For the optimizer internals (Hessian updates, trust radius, RFO,
state selection), see [methods/OPTIMIZE_DEVELOPMENT.md](methods/OPTIMIZE_DEVELOPMENT.md).

### 7.2 AIMD (`md.py`)

The ab initio molecular dynamics workflow (`class AIMD`, jobtype `md` or `surface_hopping`)
owns its iterative loop. Each MD step follows this sequence
(`workflow/md.py::AIMD.run`, operations from `workflow/integrator/`):

1. `verlet_i(state, dt)` — propagate coordinates to the current structure
   (velocity-Verlet first half; internally a no-op on the first step).
2. `apply_pbc(state)` — periodic boundary condition handling (defined in
   `data/geometry.py`, because a geometry search wraps between its steps too).
3. `calculator.run(target_states=...)` — E + G + NAC + SOC in one request.
   `target_states` is derived each step by `resolve_target_states(activestate, current_state, nstate)`:
   `activestate=0` → all states (`list(range(nstate))`), otherwise `[current_state]`.
   After a surface hop, `current_state` is updated and the derived value follows.
4. `verlet_ii(state, dt)` — complete the current-step velocity (no-op on the
   first step, no previous gradient).
5. `phase_correction(state)` — NAC/coupling phase correction.
6. `apply_constrain(state)` — geometry constraints.
7. `velocity_correction(state)` — centre-of-mass correction (returns the state
   untouched below two atoms: one pseudo-atom has no internal degree of
   freedom, and removing its centre deletes its motion).
8. `apply_ensemble(state, stage='post')` — ensemble thermostat/barostat post
   half step.
9. `apply_constrain(state)` again, as the `constrain_after_cell` phase: a
   barostat rescale is not a rigid motion and takes the geometry back off the
   constraint manifold, so the geometry the step records is re-projected.
10. `state.kinetic_energy()` — sampled **once**, here, from the velocities this
    step ends with. The run log's kinetic section, the frame record's Ekin
    column, the history frame's `K_pre` and the surface-hopping acceptance test
    all read that one value (#310).
11. `state.update_history()` — save the fully computed step (R, V, G and the
    pre-hop active state) to the history.
11. `surfacehop(state)` — surface hopping updates the active state itself
    (nosh/gsh/fssh); the algorithm is initialized once before the loop via
    `surfacehop = surface_hopping(sfhp, state)`.
12. `verification(state)` — trajectory verification.
13. `calculator.sync()` — push the updated state to the task configs.
14. `checkpoint(state)` — trajectory output files (`<title>.md.*` per step
    under the direct/buffer control, `<title>.sh.*` on hopping events,
    via `MdOutputWriter` in `workflow/records/output.py`), the run-log step block
    from the recorded snapshot, and `state.record_step_result()`
    (state serialization is still TODO).
15. `state.runtime.advance_step()` — advance the step counter.

Parameters:

| Parameter | Default | Meaning                                    |
|-----------|---------|--------------------------------------------|
| `maxiter` | 100     | Maximum MD steps                           |
| `dt`      | 20.67   | Time step in atomic units                  |
| `sfhp`    | `'nosh'`| Surface hopping method: `'nosh'`, `'gsh'`, `'fssh'` |

AIMD requires different properties depending on the dynamics model:

- Full surface hopping: `Energy + Gradient + NAC + SOC`
- Other models: `E + G` or `E + G + SOC`

The initial state (`current_state`, 0-based) is filled during the normalize
pipeline by `derive_current_state()` (sets `current_state = root` only when
the default is 0; explicit non-zero values are never overwritten). The
`activestate` keyword (0 = all-state gradients, non-zero = current-state
only) is registered in `schema/config.py` Dynamics and consumed by
`resolve_target_states()` at each MD step.

#### Multi-trajectory batch mode (`initcond` / `ninitcond`)

`AIMD.run()` plans the trajectory batch from the dynamics section
(`workflow/initcond.py::resolve_initial_conditions`):

1. Worker guard: a dispatched child state (`state.runtime.worker`) runs
   `_run_trajectory(state)` directly (its starting point and sub working
   directory were injected at materialization); no re-resolution, no
   re-dispatch.
2. `ninitcond == 1`: `read`/`load` updates the state coordinates/velocities from `<title>.init.xyz` (title from `control.title`, shipped as `examples/*.init.xyz` and staged by `qualification/manager._TOPOLOGY_SUFFIXES`)
   in place; the single trajectory runs exactly like the classic path
   (no identity, no sub directory). A single-trajectory task is the
   trajectory-count-1 case of the multi-trajectory task, so both paths
   return the universal `TrajectorySet` (`workflow/records/trajectory.py`): the
   per-trajectory outcomes arranged by the planning-order idx (length 1
   here).
3. `ninitcond > 1`: the resolved `{'coordinates', 'velocities'}` batch is
   forwarded to `self.dispatch(units=batch, child_jobtype='md')`; each
   child runs the `md` jobtype behind the frozen worker graph, and
   `collect(state, outcomes)` wraps the N outcomes into a
   `TrajectorySet` in planning order (no aggregation in this pass).
   Every `TrajectoryResult` record exposes the final state and the
   recorded history (`history()` / `history_step()`) for adaptive
   sampling analysis. Every step additionally receives the NN
   prediction deviations (`err_*` keys, tolerant fetch filling absent
   keys with 0.0, `workflow/records/deviation.py`) into `pmd.results`, and the
   run-log step block prints them with the `&adaptive` thresholds and
   the exceeded TRUE/FALSE judgment.

Batch dispatch identity planning (`Dispatcher.dispatch()`): any batch with
more than one unit gets one runtime identity per unit regardless of the
execution location — `allocate_runtime_identities()` uses
`RuntimeDirectoryManager.allocate_indices` (existing directories reused in
ascending order, so restarts map back onto previous runtimes), creates
`runtime/{skill}_runtime_{idx}/`, stamps `unit['runtime_idx'] =
'runtime_{idx}'` and writes the path into `state.runtime.workdir`. The
local runners forward the identity into `execute(state, runtime_idx=...)`
(`scheduler.invoke_local` / `dispatch_worker`; legacy single-argument
executors stay compatible for identity-less units), `slurm_run` consumes a
stamped index instead of self-allocating, and AIMD writes the trajectory
output files (`MdOutputWriter`) and the run log into the unit workdir.
Single-unit batches keep today's behavior (a remote single task still
self-allocates in `slurm_run`).

### 7.3 Force constants and vibrations (`hessian.py`)

One workflow, `Hessian` (`name='hessian'`, registered also as
`NumericalHessian`), owns `hessian`/`hess`/`frequency`/`freq`/`num_hess`/
`num_freq`. It takes the force-constant matrix from the backend or assembles
one from a batch of displaced gradient evaluations, then reads it through
`workflow/frequency.py`: modes and frequencies, thermochemical corrections, and
the molden record. The `Frequency` class it used to be reached through was a
placeholder that diagonalized nothing; see
[methods/NUM_HESS_DEVELOPMENT.md](methods/NUM_HESS_DEVELOPMENT.md) for the
routes, the keywords and the measurements.

Parameters: single-step workflow (`maxiter=1`), batch fan-out for the
numerical route.

### 7.4 PES Scan (`scan.py`)

Potential energy surface scanning:

1. Generate scan coordinates (placeholder: linear interpolation)
2. For each scan point:
   - `state.update_system({'coordinates': ...})` + `calculator.sync()`
   - Compute energy via `calculator.run()`
   - Record energy
3. Store scan energies in the result via `state.set_result()` (`scan_energies`)

Parameters:

| Parameter           | Default | Meaning                              |
|---------------------|---------|--------------------------------------|
| `maxiter`           | 10      | Number of scan points                |
| `scan_coordinates`  | None    | Predefined coordinates; None = auto-generate |

### 7.5 Batch and Sampling Workflows (`hessian.py` / `adaptive.py` / `search.py`)

Batch workflows inherit `Workflow` and implement `generate_batch()`,
`collect()` and `postprocess()`. Their `run()` forwards the batch task
descriptors to `self.dispatch(units, child_jobtype)`; the Dispatcher
materializes every task into an independent child unit and executes them
through the composed Launcher/Scheduler (Section 11).

| Workflow | Kind | child_jobtype | Notes |
|----------|------|---------------|-------|
| `NumericalHessian` | batch | energy gradient | 6N displacement fan-out |
| `GridSearch` | batch | train | Hyperparameter grid fan-out: one materialized child per point, each training through the `Train` node. It materializes its own units because a point is a configuration, not a geometry, and `materialize_unit` forwards only coordinates and velocities (`dev/methods/SEARCH_DEVELOPMENT.md` section 4) |
| `AdaptiveSampling` | sampling | md / energy / train | Round-based loop; every stage is a materialized child runtime (MD per trajectory, reference per selected frame, train one child). Round 1 is the external start; round k>=2 labels the previous round's selection, trains the round's network, and runs that network's MD. The parent never initializes a backend (`needs_backends=False`). Stage flags come from `&adaptive` (`md_*`, `reference_*`, `remote_train`); the loop, screening, labels and dataset are in `dev/methods/ADAPTIVE_DEVELOPMENT.md` |

### 7.6 Training and Prediction (`train` / `prediction` jobtypes)

Both are one `calculator.run()` with the skill owning the task semantics,
and each is a workflow node of its own: `workflow/train.py::Train` and
`workflow/prediction.py::Prediction`, the two `JOBTYPE_MAP` tokens
resolving to them under `control.driver: main`. Their shared body is
`workflow/single.py:MLSingleShot` (one call, the standard step
bookkeeping, the outcome settled from the returned record), so neither
re-implements the other.

What being a node buys is a section of its own: `WORKFLOW_SECTIONS` maps
`train` and `prediction` to the `&train` / `&prediction` keyword blocks
(`schema/config.py:class Train` / `class Prediction`), which carry the
dispatch layer only. That is the point -- a fit is the longest call in the
package, and `&train remotetask: true` is what hands it to the cluster
instead of inheriting a class default. `pyrai3md/examples/nn_train.yaml`
is the qualified example that reads the section.

A skill-driven run (`control.driver != 'main'`) still resolves every
jobtype, these included, to `SingleShot`: the driver branch says the skill
plans the whole job, so the ML verdict is then reached through
`SingleShot`'s plain completion for a jobtype the framework does not
interpret. Under either driver the record channels and the printed report
are identical.

`AdaptiveSampling` is the one workflow whose parent process does not evaluate.
It sets `needs_backends = False`, so `execute()` skips
`Calculator.initialize()` (the first-round model may not exist yet) and the
parent runtime only orchestrates. Each stage is materialized from the parent
state and dispatched with that stage's own flags; the MD stage keeps the full
theory queue (PBC and multiscale included), the reference stage keeps the full
reference queue, and training keeps only the trainable theory task. The
stage-level `workflow_kind` remains `sampling`; the round stops are listed in
`dev/methods/ADAPTIVE_DEVELOPMENT.md`.

The request reaches the skill as a jobtype string, and the profile is the
skill's decision, not the framework's. `control.driver` is only relevant
when the request carries dynamics: a driver that hands the propagation to
the backend says nothing about a training or prediction run that propagates
nothing. If a skill refuses a combination, the framework keeps that skill's
default profile **and reports the substitution** -- a request served under
another profile still prints as a completed run, so silence here would hide
the only symptom.

A training result carries no property channel; it carries `train_errors`,
`train_metric_units`, `train_wall_seconds`, `train_arm` and `train_summary`
(the names are the skill's contract, documented with the skill). The run
log renders that record as the `&nn training` and
`&nn validation mean absolute error` sections
(`utilities/print_results.py`) and skips the per-step property sections,
because the run observed no geometry.

A prediction result is the same shape of answer to a different question:
`pred_data` names a test file, and the record comes back as
`predict_errors` (one dict of statistics per served channel),
`predict_metric_units`, `predict_frames`, `predict_wall_seconds`,
`predict_raw_files` and `predict_summary`. The framework again computes
nothing: the statistics are the skill's (`service/predictor.py`), the raw
comparison files are written by the skill into the run directory so that
`keep_tmp` does not decide whether a reader gets them, and what the workflow
layer does is ask whether the record arrived (`workflow/records/ml.py`).

The outcome of both jobtypes is no longer an unconditional completion:
`MLSingleShot.run()` calls `workflow/records/ml.py:settle_ml_outcome`, which
settles the run by whether the call came back with the channels its jobtype
is contracted to carry and prints a verdict banner. There is deliberately no
error threshold: a validation MAE is not a convergence criterion, and the
comparative judgement of several trainings belongs to the grid scan
(`dev/methods/SEARCH_DEVELOPMENT.md` section 5.1).

### 7.7 Skill-Driven Single Shot (`single.py`)

`SingleShot` (`workflow_kind='single'`, node name `single`) is the
single-shot workflow. It is reached by the `control.driver != 'main'`
branch of `resolve_workflow`, which hands every jobtype of a skill-driven
run to it; the property jobtypes under `driver: main` and the
`train`/`prediction` jobtypes are not its business any more -- the latter
pair have their own nodes (§7.6). The skill receives the jobtype as-is and
plans the whole job internally; the framework never iterates and never
retries:

1. `run()` executes exactly one `calculator.run()`, then applies the
   standard step bookkeeping (`state.record_step_result()` +
   `logger.log_step`).
2. The outcome is finalized from `state.jobtype`:
   - `'optimize'`: writes `opt.xyz` / `opt_geom.xyz` from the final
     structure (the skill-reported geometry already applied through the
     `fin_coord` channel) and settles from the skill's
     `opt_converge` flag (`opt_converge is False` → `mark_failed`,
     otherwise `mark_completed`).
   - `'md'`: settles on the call having completed (`mark_completed`) and
     writes one run-log line naming the judgement it could not make -- the
     skill-side md convergence flag is not defined yet, so a bare
     `completed` would read as a verdict (#321).
   - anything else: plain `mark_completed` (single-shot semantics). A
     skill-driven `train` reaches the same line, because under that driver
     the framework does not interpret the jobtype; a run under
     `driver: main` is settled by `Train`/`Prediction` through
     `workflow/records/ml.py` instead (§7.6).

`opt_status.txt` is not among these products and is not written here. That
table is the framework optimizer's per-step record (step, energy, shift,
displacement and gradient norms), and on this path no framework step was
taken and no gradient was measured: the engine iterated internally and
reported only its final structure. Writing a single row of zeros for the
columns this workflow cannot know produced a table that claimed a
convergence it never observed. The convergence verdict lives in the run log,
which `log_optimize_summary` already fills with the `Skill-driven geometry
optimization` label.

---

## 8. Composite Module

Composite aggregation lives in `workflow/multiscale/composite.py` as **pure functions**
(no classes, no state mutation). The dependency is one-directional:
calculator imports composite; composite never imports calculator.
Multi-center ONIOM (N non-overlapping centers, gradient scatter-add,
strict result validation) is specified in
[COMPOSITE_DEVELOPMENT.md](COMPOSITE_DEVELOPMENT.md); this section covers
the legacy single-center aggregation API.

### 8.1 Contribution Pattern Registry

> **Retired:** `COMPOSITE_PATTERNS`, `queue_contributions`,
> `validate_contributions` and `project()` were removed. `validate_plan` checks
> the queue per scheme and structurally; `aggregate_results` is the live pure
> function. The block below is kept as the historical description.

The purpose of composite schemes is to establish the contribution pattern of
the task queue (decided in the planning step). Once the pattern is confirmed,
the composite result is just the sum of contributions multiplied by the task
results, so the schemes are expressed as a registry instead of dedicated
classes:

```python
COMPOSITE_PATTERNS = {
    'oniom2': [1, 1, -1],           # E = E(high, model) + E(low, real) - E(low, model)
    'oniom3': [1, 1, -1, 1, -1],    # three-layer ONIOM with intermediate region
    'qmmm':   [1, 1],               # QM/MM coupling
}
```

### 8.2 Aggregation and Projection

The Calculator already executed every task; composite only validates the queue
shape and aggregates:

- `queue_contributions(state, category=None)` / `validate_contributions(state, pattern_name, category=None)` —
  pattern validation (raises `ValueError` on mismatch); `category` restricts
  to one backend role (active role matching).
- `aggregate_results(state, category=None)` — pure function returning the combined dict
  (`category` restricts aggregation to one backend role, the calculator's
  active role; `None` aggregates both queues):

```
energy   = Σ contribution × task energy
gradient = Σ contribution × task gradient
other properties inherited from the primary task
```

- `project(results, task)` — gradient projection to screen out cap atoms;
  applies only to the primary task; placeholder until the cap atom format is
  decided.

The Calculator writes the aggregated dict back to `state.pmd.results` via
`state.set_result()` (option B). Jacobian transformation and QM/MM coupling
math (for future QMMM) belong to `workflow/multiscale/composite.py`, not to base
workflows and not to skills.

### 8.3 Current Status

ONIOM2/ONIOM3/QMMM contribution patterns are registered and aggregated through
the same pure pipeline. Multi-center ONIOM (1+2N / 1+4N terms) adds strict
aggregation with per-task gradient scatter-add, result validation, and refused
key enforcement; see [COMPOSITE_DEVELOPMENT.md](COMPOSITE_DEVELOPMENT.md).
QMMM-specific coupling math and the cap atom projection are placeholders;
task expansion already exists in `planning/task.py`.
See `CALCULATOR_DEVELOPMENT.md` Section 4.

---

## 9. PyRAIState and the Framework/Skill Boundary

PyRAIState is the central object and the **only** communication protocol between the Framework and every Skill.

### 9.1 Construction

The framework builds PyRAIState via `build_state(config)`:

```
build_state(config)
    │
    ▼
Environment() → get_maestro()
    │
    ▼
maestro.resolve_skills(config) → skills dict
    │
    ▼
state.pmd = config                     ← framework PyRAIConfig (sole data source)
_build_task_queue(state, config, skills, maestro)
    → _validate_driver(state)                   (control.driver check)
    → resolve_task_queue(config, skills)   (planning/task.py)
    → For each task:
        → Merge skill config values (cached per skill)
        → task config dict {'skill_config': <skill flat dict>, 'results': {}}
          (built directly by _build_task_queue; build_pyrai_config() is deprecated)
        → record the backend api path (importable string)
        → state.add_task(category, task)
    │
    ▼
Fully initialized PyRAIState (pmd + static task queues, no backend
instances yet; jobtype/system/results are read-only properties of pmd)
```

### 9.2 Task Initialization

Backend instances are created in the runtime initialization stage,
`Calculator.initialize()` (invoked by `execute()`):

```python
backend = resolve_backend(task)   # api path string → query → Backend(task_config)
```

The task config dict is passed directly into the Skill constructor. During backend initialization, the Skill automatically performs:

- adapter
- normalize
- validate
- execution profile compilation
- runtime preparation

These operations are entirely owned by the Skill. The Framework does not explicitly invoke normalization or validation after backend creation.

### 9.3 Design Rules

- Skills never receive raw input dictionaries. Skills always receive the task config dict (`{'skill_config': <skill flat dict>, 'results': {}}`).
- Skill initialization owns Skill-specific preparation.
- The Framework constructs PyRAIState and the task configs; the Skill constructor handles everything else.

---

## 10. Runtime State Separation

The previous architecture used the word "Runtime" for both execution algorithms and runtime state. These concepts are explicitly separated.

### 10.1 Data Layer: RuntimeState

`data/runtime/state.py` contains only `RuntimeState`:

- status / current step / iteration / timing / history / checkpoint
- `worker`: whether this runtime is a dispatched child worker
  (root=False by construction; old `.rt` files carrying the legacy
  `dispatchable` field deserialize as `worker = not dispatchable`)

It stores runtime information only. RuntimeState never contains workflow
logic. The frozen child execution graph travels with the PyRAIState
(`execution_graph` / `resolve_execution_graph`), not with RuntimeState.

### 10.2 StateSerializer (`data/runtime/serializer.py`)

A runtime file (`.rt`) is not another type of input file; it is another
source of PyRAIState. Restart and remote dispatch bypass the
Interface/Schema/Planning pipeline entirely:

```
New calculation:  Input → Interface → Schema → Planning → build_state() → PyRAIState
Restart/remote:   Runtime (.rt) → StateSerializer.load() → PyRAIState
```

- `StateSerializer` is the serialization layer for *state* hand-off; result documents go through `utilities/serialization.py` (`json_safe` walking dict/list/tuple and numpy scalars, plus an atomic `write_json` that renames into place). It is the only
  module that knows the storage format (currently pickle; future: HDF5
  or other formats). PyRAIState, Workflow, Calculator and Skill never
  know the format.
- `save_state(state, path, runtime_idx=None)` /
  `load_state(path)`: convenience wrappers around
  `StateSerializer.save()` / `StateSerializer.load()`.
- `prepare_state(state)`: public API returning a pickle-safe copy of
  PyRAIState (used by the Scheduler for multiprocessing payloads).
- Task configs are plain dicts (`{'skill_config': ..., 'results': {}}`) by
  construction and stay pickle-safe through serialization
  (`_prepare_state` / `_flatten_config` keep the flattening for any
  remaining structured config); skill backends in standalone
  mode natively accept dict configs.
- Backend instances, loggers, file handles and caches are excluded;
  `Calculator.initialize()` reconstructs them.

Dependency direction:

```
PyRAIState  ──(不知道格式)──>  StateSerializer  ──(知道格式)──>  pickle
Workflow    ──(不知道格式)──/
Calculator  ──(不知道格式)──/
Skill       ──(不知道格式)──/
```

When the format changes, only `serializer.py` changes.

### 10.3 Workflow Layer: Execution Logic

Execution logic belongs exclusively to `workflow/`.

- Workflow modifies RuntimeState.
- RuntimeState never contains workflow logic.

---

## 11. Dispatcher and Dispatch Subsystem

`workflow/dispatch/` implements the unified runtime execution layer.
The former mode enums (`single` / `local` / `parallel` / `dispatch`) and
the RuntimeEngine/policy/node contracts are removed: every workflow and
the calculator inherit `Dispatcher`, and the execution mode is fully
determined by two orthogonal booleans:

- `multitask`  : create multiple tasks through multiprocessing
- `remotetask` : execute tasks remotely through a launch tool

### 11.1 Dispatcher (`dispatcher.py`)

`Dispatcher.dispatch(units, child_jobtype, active_role)` is the unified
execution tool method (not a public entry point; subclasses call it
from their own `run()` methods):

1. **Unit materialization**: batch task descriptors are converted into
   independent child PyRAIState units (`materialize_unit`): deep copy of
   the parent state, `child_jobtype` applied to the state and every task
   config (pinned profile cleared), worker label stamped, child execution
   graph frozen by the parent (`resolve_execution_graph=False`).
   With an explicit `active_role` (e.g. `dispatch(..., active_role=
   'reference')`) the child is single-role instead: the role is stamped
   on `RuntimeState.active_role` and read back by `execute()`, the task
   configs keep the capabilities/profile planning gave them (the planned
   jobtype is not realigned), and the inactive role's queue is dropped so
   its backend is never constructed. `child_jobtype` is then required:
   it is what makes the child resolve a single-shot calculation rather
   than re-entering the parent workflow. The role must travel on the
   state because the multitask branch rebuilds its execution units from a
   fixed key whitelist and would drop a field carried on the unit dict.
2. **Execution composition**: `Scheduler(multitask)(Launcher(remotetask))`;
   workflow and calculator never inspect how the units are executed.
3. Results are returned in unit order. A single-point computation is a
   unit list of length 1, fully unified with batch execution.

Universal child task rules:

- **Rule 1**: every dispatched unit state is labeled as a worker
  (`state.runtime.worker = True`).
- **Rule 2**: a worker execution graph never carries `remotetask=True`
  (verified at the worker entry; violations raise).
- **Rule 3**: base nodes never carry `remotetask=True`; planning
  completely ignores the skill-section `remotetask` keyword and leaves it
  to the skill itself, which uses it to decide its own local run or slurm
  run. The Calculator constructor rejects `remotetask=True`.

### 11.2 Launcher (`launcher.py`)

Converts `remotetask` into a concrete unit execution function:

- `remotetask=False`: returns the `'local'` sentinel; the Scheduler
  executes units inside the current process through the injected
  `local_executor`.
- `remotetask=True`: returns `slurm_run`, the sole remote runner.
  Units are materialized as independent child runtimes submitted to
  Slurm via `sbatch`.

The Launcher no longer selects among multiple backends; the former
`dispatch_backend` field and `LaunchTool` abstraction were removed
in #82. `remotetask=True` maps unambiguously to `slurm_run`.

### 11.3 Scheduler (`scheduler.py`)

Converts `multitask` into an execution wrapper:

- `multitask=False`: sequential wrapper (the launch function is called
  per unit; the resource check of #306 runs here too, at one unit).
- `multitask=True`: wrapper executing the units through an explicit
  spawn `multiprocessing.Pool` over `prepare_state`-flattened child
  states (sequential fallback when workers == 1 or one unit). Both
  local and remote runners go through the Pool — the former
  serial-fallback branch for `runner=='local'` was removed in #82.

The worker count comes from the `multipool` keyword of the
dispatching workflow's input section (`&md` for AIMD, `&adaptive` for
AdaptiveSampling; carried through the Dispatcher into
`Scheduler._worker_count()`, defaulting to 1 for workflows without a
keyword section); no extra environment variables are introduced.
`validate_resources()` compares the CPU demand with the CPUs this run may
occupy: warning only, oversubscription is allowed, nothing is resized on
the user's behalf, and no environment variable is invented. Since #306 the
demand is assembled from what the skills declare rather than from any table
in the framework: a skill's manifest `resources` block names the keywords
carrying its thread and rank counts together with the value to assume when
an input omits them, and the check reads those keywords from the task queue.
The arithmetic is `units x ranks x threads`, where `units` is the pool size
capped by how many units are queued (one on the sequential path, which is
checked as well), multiplied by `composite max_workers` when the term axis
runs its backends in a thread pool. The reference is the local core count,
or `slurm.ntasks x slurm.cpus_per_task` from the environment snapshot when
the run submits, since a submitted job does not live on the machine that
parsed the input. Before #303 the demand came from
`state.pmd.environment.num_thread`, a field the environment snapshot never
populated, so the check always computed workers x 1 and could never fire;
#303 read it from the queue, and #306 cancelled the run-wide floor that
resolution depended on. Measured on a 10-core machine, `multipool: 4` with
orca at `thread: 3`, `nproc: 2` logs `Parallel request 4 unit(s) x 6 CPU(s)
per unit of orca_skill = 24 exceeds the 10 CPUs available to this run
(oversubscribed).` Criteria:
`qualification/verification/test_resource_keywords.py`.

### 11.4 Execution Graph Builder (`planner.py`)

The single planning stage of the runtime, invoked once by `execute()`:

- **Root** (`resolve_execution_graph=True`): jobtype → primary workflow
  → node chain assembled from the node-chain templates
  (`WORKFLOW_NODE_DECLARATIONS` / `WORKFLOW_BASE_CHAINS`). Every node
  resolves its multitask/remotetask/multipool triple through the same
  priority chain: declaration dict (highest priority, a key that
  appears forces its value) > input keywords > class defaults.
  - workflow nodes: read from the workflow's own keyword section
    (`&adaptive` → `state.pmd.adaptive`, `&md` → `state.pmd.dynamics`,
    `&optimize`, `&hessian`, `&search`).
  - base nodes: one per backend role (`base.theory` / `base.reference`,
    roles without tasks are skipped), read from the dispatch-layer
    keywords of the skill sections carried by the role task queue,
    combined by the aggregation rules.
  The frozen graph is stored into `state.execution_graph`. Multi-stage
  workflows (`stages` + `stage_defaults`) carry one sub-node per stage
  (`base.<role>` entries become base nodes inside the stage list).
  The two consumers in `workflow/__init__.py` (`_graph_base_flags` and
  `_graph_backend_execution`) walk the whole tree through
  `_graph_base_nodes`, so a staged `base.reference` reaches the
  Calculator exactly like a top-level one; merging is by role only.
- **Worker** (`resolve_execution_graph=False`): never builds a graph;
  only verifies the parent-frozen graph against the universal rules
  (graph missing or carrying `remotetask=True` raises RuntimeError).

Tri-state aggregation (multitask aggregated across the skills of one
backend role; multipool follows the matching rule: all unset → 1,
consistent → the declared value, conflicting → KeyError):

| Values | Result |
|--------|--------|
| all None | False |
| {True, None} | True |
| {False, None} | False |
| {True, False} | KeyError (conflicting configuration) |

Universal rule 3: base nodes are always local; the skill-section
`remotetask` keyword is never read or aggregated by planning (it is
reserved for the skill's own local/slurm decision), so it can never
conflict at a base node.

### 11.5 Remote Runner and Runtime Directory Manager

The sole remote runner is `slurm_run` (`slurm_run.py`), which handles
single-task Slurm execution: generates `slurm.sh`, submits via
`sbatch`, polls with `sacct`, and collects results. `subprocess_run`
(the former same-machine child-process runner) has been removed from
the execution chain (its module was deleted): the Launcher only chooses
between `'local'` (in-process) and `slurm_run`, and `slurm_run` is the
only remote runner; its former functionality is fully covered by the
Pool direct-call path.

`RuntimeDirectoryManager` (`runtime.py`) owns the runtime directory layout
(`<skill>_runtime_<idx>`, `.rt` and `result.json` paths), unique index
allocation (reused on restart) and the `is_completed()` restart-skip flag.
Slurm configuration (sbatch/shell/export) is read from
`state.pmd.environment`.

Dependency rule: the dispatch package **never imports workflow**; the
in-process entry is injected (`local_executor` callable + `executor_spec`
module string resolved by multiprocessing workers via importlib), and the
workflow registry is injected into the ExecutionGraphBuilder.

### 11.6 Reserved Extensions

- Backends `mpi` / `ssh` / `pbs` are reserved.
- Skill explicit `local_run()` / `slurm_run()` decisions are driven by
  the skill's own `remotetask`/`execution_mode` keywords (the framework
  never dispatches single points).

---

## 12. Registry Responsibility

Registry is no longer part of the runtime execution pipeline. Its responsibilities are limited to framework initialization.

Registry provides:

- Skill discovery
- Registry constructs
- Skill metadata
- Capability metadata
- Default configuration
- Control parameter lookup

Registry is queried **before** backend instantiation. After Skill objects have been created, workflow execution never interacts with Registry again.

---

## 13. Import Isolation

`planning/` and `workflow/` have **one registered import edge, pointing from
workflow to planning**: `workflow/calculator.py` and
`workflow/multiscale/composite.py` read three pure task-descriptor selectors
(`resolve_primary_task`, `resolve_system_task`, `region_label`) from
`planning/task.py`. They select from the queue planning already built and never
re-plan. The other direction has no edge: the optimizer declarations planning
needs live in `schema/optimizer_spec.py`, and the fix mask in
`data/constraint.py`. A new edge is a defect, not a precedent. All other communication happens through
`state.theory/reference` (task queues + task config dicts) and `state.jobtype`:

- Planning resolves tokens to capability name strings and expands composite workflows into task queues (`resolve_task_queue`).
- Skill backends read their task config dict during instantiation to build the internal CalculationRequest.
- Workflow reads `state.jobtype` to select the workflow function.
- Calculator iterates `state.iter_tasks()` to find backend instances and region metadata.

Similarly, `workflow/dispatch/` never imports the workflow package: only
`workflow/__init__.execute()` imports dispatch (one-directional); the
in-process execution entry is created by each workflow/calculator in
its own `run()` method, and batch workflows consume the dispatch
service by duck typing.

```
planning/ ──── resolves ────→ task queues in state.theory/reference
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                Skill backend                   Workflow layer
                (reads task capabilities,       (reads state.jobtype,
                 builds CalculationRequest       creates Calculator,
                 internally)                     selects workflow)
```

---

## 13.1 Dispatch contracts (2026-11-12)

- A remote term script is built by `slurm_run._build_script(..., entry='.rtt')`;
  the term worker no longer carries its own copy. The user `shell` section
  (module loads) is therefore replayed in the child, like a whole-runtime job.
- The Slurm poll has a wall-clock deadline and a consecutive-`UNKNOWN` cap; a
  scheduler that never settles raises instead of hanging the parent.

## 14. Design Rules

### Rule 1: Workflow determines HOW; Planning determines WHAT

Workflow determines how the already planned computation is executed. It never performs planning, resolves jobtype, or determines capabilities. See `PLANNING_DEVELOPMENT.md`.

### Rule 2: Workflow operates only on instantiated Backend objects

Workflow never queries Registry. Workflow only orchestrates backend execution via Calculator.

### Rule 3: Workflow cannot directly call external programs

```
Wrong:
    AIMD → BAGEL

Correct:
    AIMD → Calculator → Skill
```

### Rule 4: One calculation step generates one skill request

```
Wrong:
    energy() → gradient() → soc()  (three separate calls)

Correct:
    Calculator one request: energy + gradient + soc (one call)
```

### Rule 5: Skill only provides capability

Skill is responsible for: input generation, execution, output parsing.

Skill is NOT responsible for: optimization loop, AIMD propagation, ONIOM combination, workflow control.

### Rule 6: PyRAIState is the only communication protocol

Skills never receive raw input dictionaries. Each backend receives the task config dict (`{'skill_config': <skill flat dict>, 'results': {}}`) for initialization.

### Rule 7: Skill initialization owns Skill-specific preparation

The Framework constructs PyRAIState and the task configs. The Skill constructor performs adapter, normalize, validate, and execution profile compilation automatically.

### Rule 8: Import isolation

`planning/` and `workflow/` have one registered import edge, from workflow to
planning (the three task-descriptor selectors in `planning/task.py`, documented
in Section 13). Planning imports neither workflow nor a skill backend.
Communication happens through `state.theory/reference` (task queues + task
config dicts) and `state.jobtype`.

### Rule 9: Task queues are immutable from the main pipeline after construction

The workflow layer never modifies the structure of `state.theory/reference` task queues.
Calculator reads tasks and writes calculation results into task configs and the main
config, but never adds, removes, or reorders tasks.

### Rule 10: Composite aggregates by contribution as pure functions

Composite schemes are contribution patterns expanded in `planning/task.py`.
`aggregate_results()` is a pure function returning the combined dict; the
Calculator writes it back to `state.pmd.results`. Future Jacobian
transformation, cap atom projection, and QM/MM coupling belong to
`workflow/multiscale/composite.py`, not to base workflows and not to skills.

### Rule 11: execute(state, runtime_idx=None) is the single entry point

`execute(state, runtime_idx=None)` builds (root) or verifies (worker) the
execution graph, creates Calculator(state, runtime_idx) + initialize()
with the graph node flags, resolves the workflow class via
`resolve_workflow(state)`, and runs it. The caller never needs to manage
the resolution/instantiation logic. `execute()` does not inject
`local_executor`; both Workflow and Calculator create their own
in-process executor closures in `run()`.

### Rule 12: Planning once, execute everywhere (runtime)

The ExecutionGraphBuilder builds the execution graph exactly once per
root runtime and freezes it into `state.execution_graph`. Worker runtimes
(`resolve_execution_graph=False`) never re-plan; they only verify the
parent-frozen graph against the universal rules (rule 2: workers never
carry remotetask=True).

### Rule 13: Dispatch never imports workflow

`workflow/dispatch/` is consumed one-directionally by
`workflow/__init__.execute()`. Each Workflow and Calculator creates
its own in-process executor closure in `run()`; the
ExecutionGraphBuilder receives the workflow registry by injection.
Workflows and Calculators stay algorithm-pure: no Launch Tool /
Parallel / Slurm / Resource Validation knowledge.

### Rule 14: Runtime parameters travel through PyRAIState

All runtime execution parameters (multitask/remotetask keywords, the
dispatch pool size `multipool` keyword of the workflow input section
read by `Scheduler._worker_count()`,
slurm/shell/export) are carried by the `state.pmd` snapshot
(control/environment sections) and the modularized keyword sections.
No extra environment variables are introduced; `pmd_env.yaml` and the
kernel Environment singleton gain no keys. The former
`dispatch_backend` field was removed in #82; `remotetask=True` now
maps unambiguously to `slurm_run`.

---

## 15. Summary

PyRAI3MD workflow architecture:

```
PyRAIState (from build_state, carries task queues in theory/reference)
    │
    ▼
execute(state, runtime_idx=None)
    ├── ExecutionGraphBuilder.build() → freezes state.execution_graph
    ├── Calculator(state, runtime_idx).initialize() → backend instantiation
    ├── resolve_workflow(state) → Workflow class or None
    └── workflow.run() / calculator.run()
    │
    ▼
Workflow / Calculator (all inheriting Dispatcher)
    │
    ├── Optimize:  loop { calculator.run() → gradient → update({'system': ...}) → sync() }
    ├── AIMD:      loop { calculator.run() → E+G+NAC+SOC → update({'system': ...}) → sync() → hop }
    ├── Hessian:   (backend matrix | batch of displaced gradients) → vibration → thermo → set_result()
    ├── Scan:      loop { update({'system': ...}) → sync() → calculator.run() → record energy }
    ├── Batch:     generate_batch → self.dispatch(units, child_jobtype)
    │              → collect → postprocess  (multitask x remotetask)
    ├── Adaptive:  md (theory) → base (reference role) → train, one calculator run per stage
    ├── Composite: calculator.run() → per-task calculate → contribution aggregation
    ├── Single-shot: calculator.run() → Result
    └── SingleShot:  (control.driver != 'main') one calculator.run() →
                     finalize by state.jobtype (optimize outputs + opt_converge flag)
    │
    ▼
Result (raw or composite-aggregated; always written to state.pmd.results)
```

This separation enables:

- Efficient external program usage (one request, one call)
- Flexible AIMD/property combinations
- Multiple backend support
- ONIOM/QM-MM extension
- Complete backend independence
- Unified runtime execution (multitask x remotetask) with worker safety
- AI-driven workflow planning
- Future periodic and ML potential integration

---

## 16. Future Extensibility

Under this design, adding a new workflow only requires implementing a new Workflow class and registering it in `JOBTYPE_MAP` / `WORKFLOW_REGISTRY`. No modification of `resolve_workflow()` or the runtime pipeline is needed. Batch workflows automatically gain the full multitask x remotetask execution matrix through self.dispatch().

Examples:

```
AdaptiveWorkflow (multi-stage chain via `stages` + `stage_defaults`)
    AIMD → Sampling → Reference → DataOperation → Training

MetadynamicsWorkflow
    calculator.run()
    update bias potential
    propagate dynamics
```

Reserved dispatch backends: mpi / ssh / pbs.

---

## 17. Historical Notes

### 17.1 Removal of Runtime Planning from pmd.py

The original `pmd.py` `execute()` function contained runtime planning logic:

```
parse_jobtype() → is_basic / is_advanced → _execute_advanced() → resolve_calculator()
```

These planning decisions were moved to the Planning stage (before PyRAIState construction). The `execute()` function was simplified to a unified entry point in `workflow/__init__.py`. The `resolve_calculator()` factory function is retained for convenience but is not used by the main pipeline.

Runtime never re-analyzes the input. All scientific planning is complete before PyRAIState is built.

### 17.2 Runtime Engine Refactor (2026-08-05)

The Dispatcher was upgraded from a workflow component to the RuntimeEngine
(the sole execution engine), and the RuntimePlanner became the single
runtime Planning stage. `BatchWorkflow.dispatcher` was removed in favor of
the injected engine service; Slurm/Subprocess configuration moved from the
kernel Environment singleton to `state.pmd.environment`; `resolve_dispatcher`
was replaced by the `create_engine(state)` factory. The `execute()`
signature and the `execution_mode=local` default behavior are unchanged.

### 17.3 Dispatcher Unification Refactor (2026-08-06)

The single/local/parallel/dispatch execution modes and the
RuntimeEngine/RuntimePlanner/ExecutionPolicy/RuntimePlan/ExecutionNode
contracts were removed. Every workflow and the calculator now inherit
`Dispatcher`; execution is composed from two orthogonal booleans
(multitask/remotetask) through `Scheduler(multitask)(Launcher(remotetask))`.
The abstract base classes (SingleWorkflow/IterativeWorkflow/BatchWorkflow)
were removed; `namd.py`/`NAMD` were renamed to `md.py`/`AIMD`.
Planning became the ExecutionGraphBuilder freezing the execution graph
into `state.execution_graph` (modularized keywords, tri-state
aggregation); workers carry a parent-frozen graph
(`resolve_execution_graph=False`) and are labeled through
`state.runtime.worker`. Three universal rules were introduced (worker
label; worker graphs reject remotetask; calculator always local with the
skill-section remotetask keyword reserved for the skill's own local/slurm
decision). The `workflow_execution`/`calculator_execution` environment
keywords were removed; `slurm` is retained. (The dispatch pool size is
today read from the `multipool` keyword of the workflow input section
by `Scheduler._worker_count()`,
not from `environment.workflow_workers`.)

Follow-up (#300): the framework environment section now declares **no**
execution switch at all. `execution_mode` is a skill keyword; the retired
`execution_mode`/`workflow_execution`/`workflow_workers` keys are refused
on input (`schema.config.RETIRE_ENVIRONMENT_KEYS`). When the framework's
dispatch says remote, `data.core._build_task_queue` derives
`execution_mode=local` into each task's skill config, so exactly one
component submits the job; `Calculator._task_execution_mode` reads that
same field for its dispatch log and the thread-safety contract it hands
the backend.)

### 17.4 Dispatcher.run() → dispatch() and StateSerializer Extraction (2026-08-07)

`Dispatcher.run()` was renamed to `dispatch()` and downgraded from a
public entry point to a tool method that subclasses call from their own
`run()` methods. Both Workflow and Calculator create their own
in-process executor closures in `run()` (Workflow from
`self.calculator`, Calculator from `self.run_in_process()`);
`execute()` no longer injects `local_executor`. `multitask=True`
creates deep copies of the current subject via `_create_copy()`,
distributes units through `broadcast()`, and executes copies in
parallel through the Launcher/Scheduler pipeline.

The serialization functionality from `data/runtime/loader.py` were
consolidated into `data/runtime/serializer.py` as the
`StateSerializer` class — the sole module that knows the storage
format (currently pickle). `loader.py` was deleted. API:
`save_state` / `load_state` / `prepare_state` / `shallow_copy_state` /
`deep_copy_state`. The dependency direction is:
PyRAIState → StateSerializer → pickle. PyRAIState, Workflow,
Calculator and Skill never know the format.

### 17.5 Dispatch Dual-Axis Orthogonal Refactor (2026-08-13)

The `multitask`/`remotetask` two dimensions were made fully orthogonal.
`subprocess_run` (same-machine child-process relaunch) was removed from
the execution chain (its module was deleted); its functionality is fully
covered by Pool direct skill-API calls. The Launcher only chooses between
`'local'` (in-process) and `slurm_run`, and `slurm_run` is the only
remote runner — the former `dispatch_backend` field selection was removed. Launcher was simplified
to `remotetask=False` → `'local'`, `remotetask=True` → `slurm_run`.
Scheduler `_wrap_parallel` removed the `runner=='local'` serial
degradation branch; local runners also execute through the Pool for
parallelism. The `Environment.dispatch_backend` field,
`DEFAULT['dispatch_backend']`, and all related selection logic were
removed from `schema/config.py`, `schema/adapter.py`,
`kernel/environment.py`, and `qualification/pmd_env.yaml`.

---

## 18. ONIOM Charge Management, Two-Phase Execution and Output

This section covers the framework side of ONIOM electrostatic embedding;
the charge routing rules and the `&composite` breakdown live in
[methods/ONIOM_DEVELOPMENT.md](methods/ONIOM_DEVELOPMENT.md) section 3 and
[COMPOSITE_DEVELOPMENT.md](COMPOSITE_DEVELOPMENT.md) section 16, and
the per-center view in
[MULTICENTER_DEVELOPMENT.md](MULTICENTER_DEVELOPMENT.md) section 15.

### 18.1 ChargeStore binding (build_state)

`data/core.py::_init_charge_store` binds a `ChargeStore` when the run is
ONIOM3 (`partition_plan.n_layer == 3`) and `embedding` or `read_charge`
is set, and for QMMM (managed region = highlevel, holding the
`q_qm = point_charges[highlevel]` values). The managed region (highlevel
union midlevel) is taken from the bound partition plan's per-center
`high_indices`/`mid_indices`, so the indices are already normalized to
0-based globals. A supplied whole-system `system.point_charges` seeds the
store first. With `read_charge`, the midlevel charges are initialized
from the legacy `{title}.charge` file in the run working directory and
never updated afterwards. `PyRAIState.point_charges` is a read-only
property deriving the whole-system `[q, x, y, z]` rows at the current
geometry (falling back to `pmd.system.point_charges` when no store is
bound); it is distinct from the scalar `PyRAIState.charges` (the system
total charge) and from the arbitrary `system.external_charges` file.

### 18.2 Two-phase execution

`Calculator.run_in_process` splits an embedded evaluation into two
phases when `two_phase = (charge_store is not None and embedding and not
read_charge)`:

1. phase 1 runs every non-provider term (the L(real) baseline, the
   mid-level inter source M(I_c) and all low/mid corrections) with no
   embedding; `update_from_outcomes` then writes the midlevel charges.
2. phase 2 runs the high-level providers H(c), each embedded in its own
   center's freshly updated midlevel; `update_from_outcomes` then writes
   the highlevel charges.

Both phases share one `evaluation_id`. The routed outcomes are merged
back into the original task order, so `assemble_results`,
`collect_center_electronic` and the index-aligned gradient scatter are
unchanged. A non-multi-center ONIOM3 run degrades naturally to
low+mid then high. `read_charge` stays single-batch (the static field is
already loaded) but still embeds the high level; a run without a store
keeps the single-batch path and delivers `point_charges=None`, so it is
byte-for-byte unchanged. The remote per-term path
(`execute_remote_terms`) is split across the two phases the same way.

### 18.3 Embedding delivery channel

The embedding field travels as a `point_charges` `[q, x, y, z]` array
(Angstrom; the backend converts to Bohr):
`Calculator._build_work_items` calls `embedding_for_task` for each task
(non-None only for a high-level provider with a bound store),
`make_work_item` carries the `point_charges` field, `_run_one` passes it
to `backend.calculate(...)`, and the remote term payload forwards it in
`request`. Whether the field is actually used is decided by each skill
backend (xTB writes a `.pc` file and an `$embedding` block; a backend
without support simply ignores it).

### 18.4 Charge sync, verbose log and MD charge files

After assembly and before `set_result`, `Calculator._sync_point_charges`
writes the managed-region charges into `pmd.system.point_charges` (only
when a store is bound), so the history snapshot carries them. The
verbose run log then prints the `&background charges in Angstrom` section
through the existing `_format_charges`. For MD,
`workflow/records/output.py::MdOutputWriter` takes a `charge_output` flag (set
from `state.charge_store is not None` in `workflow/md.py`); when active it
adds `md.charges`/`sh.charges` (and, for a partitioned run, the
per-center `md.#.charges`/`sh.#.charges`) to the file matrix and writes
each frame in the legacy `{ncharge}\n{cmmt}\n{q x y z}` format from the
snapshot's `system.point_charges` (per-center rows are sliced to the
center's high region by coordinate match). With `charge_output=False`
the file matrix is byte-identical to a run without charge management.

### 18.5 Region-based static_id naming

`planning/task.py::_compose_static_id` names a composite task's working
directory by region instead of the former opaque layer number:

```
tmp_{SKILL}                              (single-shot job)
tmp_{region}_{role}[_task{N}]_{SKILL}    (composite/multitask)
```

The engine token in either form is what a declared `project` replaces, and
nothing else (`service/utils.py::build_calcdir`): `tmp_high_qm_task1_BAGEL_3`
with `project: casemt_test` becomes `tmp_high_qm_task1_casemt_test_3`, so the
region, role, ordinal and runtime index keep distinguishing the tasks that
share a run. A run that does not declare a project - which is how the schema's
own default reaches the builder - keeps the name above unchanged. The reason
the token is replaceable at all: engines write results under fixed names
inside the calculation directory (BAGEL's `ENERGY.out`, `FORCE_*.out`,
`NACME_*_*.out`; OpenQP's `OQP.json`), so two runs in one directory cannot
attribute such a file - the reader finds whichever run wrote it last and
reports it as its own.

`region` is the human-facing layer name from the shared `region_label`
helper (model to high, inter to mid, real to low; other regions such as
`all`/`fragment` pass through). The `_task{N}` ordinal is a 1-based
position assigned only when several tasks share the same
`(region, role, skill)` group (a multi-center ONIOM level or a
multi-group fragment), ordered by center then queue slot, so ONIOM center
0 is `task1`; a lone task in its group (a single-center run, the
center-less L(real) baseline) omits it. Examples:
`tmp_high_qm_task1_OPENQP`, `tmp_mid_qm2_XTB`, `tmp_low_qm3_XTB`.
`compose_execution_id` still appends the runtime suffix unchanged. This
is a naming change for new runs; an in-flight restart keeps matching its
persisted `static_id`.
