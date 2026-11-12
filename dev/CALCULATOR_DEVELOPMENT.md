# PyRAI3MD Calculator Development Guide

Last updated: 2026-11-12 19:20 (composite registry references updated to validate_plan/aggregate_results. Earlier: 2026-11-12 15:50 (Sections 4.2-4.4 mark the retired composite registry, queue/validate helpers and `project()`; `validate_plan` and `aggregate_results` are the live API. Earlier: 2026-11-04 11:10 (#451: 4.2 now says QMMM is validated structurally rather than by a contribution pattern, because its queue length depends on `embedding`, the center count and `sc_charge`. What `_run_qmmm` does per evaluation - the cold-start probe, the charge rounds with one Jacobi snapshot each, the accepted-charge re-probe, then the property round and the single classical surface - and the `qmmm_charge_iteration` report it attaches to the result are documented in methods/QMMM_DEVELOPMENT.md sections 4, 5 and 8. The rest of this file was not re-audited this round. Earlier: 2026-09-08

---

> **Multi-center unification (refactor R2-R6, 2026-09-08).** CHANGELOG
> #200-#204, `dev/MULTICENTER_DEVELOPMENT.md`. R3 wired the term executor:
> `run_in_process` freezes the step geometry/active snapshot, builds one
> unified `make_work_item` per task_slot, runs them through `_resolve_executor`
> (serial default; thread only for an audited `(backend_name, runner,
> execution_mode, profile_semantics)` contract) or `execute_remote_terms`
> (remotetask), validates completeness/no-duplicate/no-stale by
> task_id+evaluation_id, then routes each outcome to per-center electronic vs
> composite math and sums in a fixed order. `execute_with_failure_handling` /
> `apply_failure_policy` fail the whole evaluation on any term failure after
> collecting diagnostics. R4: for C>1 the calculator stages the per-center
> `electronic` arrays + the assembled `active_surface` and calls
> `state.commit_step()` (the atomic writeback); C>1 builds no global
> `energy[nstate]` axis. R6: `target_states` accepts a flat list (C=1) or a
> per-center dict routed by `_task_target_states` (by `task.center_index`);
> `sync()` skips the `_ELECTRONIC_STATE_FIELDS` scalars for C>1 so the global
> `nstate=1` never overwrites the planning-derived per-center task configs.

## 1. Design Objective

This document describes the design of the Calculator and Composite modules in
`workflow/calculator.py` and `workflow/composite.py`. The overall architecture is
defined in [ARCHITECTURE.md](ARCHITECTURE.md); the workflow layer context is in
[WORKFLOW_DEVELOPMENT.md](WORKFLOW_DEVELOPMENT.md); the state data model is in
[DATA_CORE_DEVELOPMENT.md](DATA_CORE_DEVELOPMENT.md).

Calculator is the **sole runtime state manager** of the execution pipeline:

- It executes the task queue from PyRAIState.
- It synchronizes task configurations with the framework state.
- It writes aggregated results back into the framework PyRAIConfig.

The other layers are deliberately narrow:

- **Workflow** only evolves the system (coordinates, velocities, electronic state).
- **Composite** only performs mathematical aggregation and projection.
- **Backend** only updates its own PyRAIConfig.

The design goal is a **self-contained Calculator** with no dependence on any
external iterative workflow: iterative workflows simply call `run()` and `sync()`
repeatedly.

---

## 2. Shared Mutable State and Runtime Ownership

Calculator, Workflow and future integration code intentionally operate on the
**same** framework PyRAIConfig object (`state.pmd`) during runtime. This shared
mutable state avoids unnecessary object reconstruction while keeping state
transitions explicit through `run()` and `sync()`.

`state.pmd` is the sole runtime data source (see DATA_CORE_DEVELOPMENT.md):
jobtype, structure and results are exposed through read-only pass-through
properties, and every modification goes through `state.update_system()` /
`state.set_result()`.

The ownership of runtime data is strictly defined:

| Layer | Owns / Modifies |
|-------|-----------------|
| Workflow | Framework System only (via `state.update_system()`) |
| Calculator | Task properties; synchronizes runtime state via `sync()` |
| Composite | Nothing (pure functions: aggregation + projection) |
| Backend | Its own task PyRAIConfig only |

No module may modify objects owned by another layer except through the defined
synchronization interfaces.

---

## 3. Calculator

### 3.1 Class Overview

```python
class Calculator(Dispatcher):
    """High-level calculator executing the PyRAIState task queue."""

    multitask = False
    remotetask = False

    def __init__(self, state=None, runtime_idx=None, dummy_backend=None,
                 multitask=None, remotetask=None,
                 local_executor=None, executor_spec=None):
        if remotetask:
            raise RuntimeError(...)   # universal rule 3
        ...
```

- `Calculator` inherits `Dispatcher`: `run()` calls `self.dispatch()` with a
  unit list of length 1, fully unified with batch execution. Universal rule 3
  keeps the calculator always local — the constructor rejects
  `remotetask=True` (remote single-point execution is controlled by the skill
  itself through its own execution_mode).
- `state` — PyRAIState (primary API). Tasks come from `state.iter_tasks()`.
- `runtime_idx` — runtime execution suffix provided by the Workflow layer
  (e.g. `'replica_3'`, `'traj_8'`); None for single-shot runs.
- `dummy_backend` — test-only backend injection. The parameter, attribute and all
  internal references are uniformly named `dummy_backend` to avoid confusion
  with real backends.
- `multitask` / `remotetask` — execution flag overrides from the frozen
  execution graph; None keeps the class default.
- `local_executor` / `executor_spec` — Dispatcher execution entry keywords.

### 3.2 Live Geometry Derivation

`_resolve_task_geometry(task)` derives each task's atoms/coordinates at
execution time:

- Reads the **live** geometry through the state pass-through properties
  (`state.atoms` / `state.coordinates`, which pass `state.pmd.system` data
  through directly without rebuilding view objects).
- Applies the pre-stored `region_indices` slice (region `'all'` or empty
  indices passes the full system).

Task configs never store sliced atoms/coordinates/charges, so iterative
workflows always pass up-to-date geometry to every backend.

### 3.3 run(): Three Steps Per Task Queue Execution

`run()` takes no arguments; all information comes from `self.state`:

```
for each task in state.iter_tasks():
    1. run the backend over the live task geometry
    2. sync the received skill results into the task config
3. project + aggregate through the composite module, write back to state.pmd
```

Detailed flow:

```
Calculator.run()
    │
    ▼
for each task:
    ├── _resolve_task_geometry(task)      ← live pmd.system + region_indices
    ├── backend.calculate(atoms, coordinates, target_states=...)
    └── store_task_result(task, result_dict)   ← fallback write-back
    │
    ▼
project(task_config_results(primary_task), primary_task)   ← cap atom screening
write_task_results(primary_task, projected)
    │
    ▼
aggregate_results(state, category=active role)   ← composite pure function (option B)
    │
    ▼
state.set_result(result.to_dict())      ← write into state.pmd.results
```

Result write-back follows **option B**: the composite aggregation function
returns the combined result dict (never writes state itself), and the
Calculator writes it into `state.pmd.results` through the state update
interface. This keeps Composite a pure module and keeps all state mutation in
one place.

### 3.3.1 The fin_coord Channel (Skill-Reported Final Geometry)

Skills performing internal iteration (geometry optimization, molecular
dynamics, ...) report their final structure through the `fin_coord` key of
the returned result dict (`(natom, 3)`, Angstrom). The Calculator consumes
this channel in `_apply_skill_geometry()` after every `backend.calculate()`:

- The key is a transport channel, not a physical property: it is **popped**
  from the result dict (and from the already stored task config results) so
  it never leaks into `state.pmd.results`.
- When present and different from the current structure, the Calculator
  applies it to the system coordinates via `state.update_system()`, keeping
  the authoritative geometry in sync with the skill-internal iteration
  (subsequent workflow logging and output files reflect the correct
  structure).
- Missing key or empty value: no internal iteration happened; nothing is
  changed.
- Length mismatch (`len(fin_coord) != natom`) is a fail-fast
  `ValueError` (skill contract violation): architecturally a mismatch is
  impossible, since framework-driven multiscale optimization keeps the skill
  at single-point/gradient level and skill-internal multiscale optimization
  always iterates the full system.

### 3.4 sync(): Framework State → Task Configs

After a Workflow evolves `state.pmd` (e.g. new coordinates), the task configs
must be realigned before the next `run()`:

```python
calculator.sync()
```

- Pushes `state.pmd.system` fields into every task configuration.
- Synchronized fields are declared in `SYNC_SYSTEM_FIELDS`:

```python
SYNC_SYSTEM_FIELDS = (
    'atoms', 'mass', 'coordinates', 'velocities',
    'charges', 'multiplicity', 'natom', 'nstate',
    'ci', 'spin', 'coupling', 'state_label', 'state_spin',
    'current_state',
)
```

- For Pydantic task configs, only fields declared by the config's system
  submodel are written.
- For plain dict task configs, `sync()` pushes into `config['skill_config']`
  and only updates fields **already present** in the flat skill dict:
  framework-only fields (ci, coupling, ...) are never injected into skill
  configs.
- `sync()` **never** modifies the calculation, method, or control sections.

### 3.5 Iterative Workflow Loop Contract

Every iterative workflow follows the same three-step loop:

```python
for step in range(maxiter):
    result = self.calculator.run()          # 1. compute (results → state.pmd)
    # ... future integration code ...
    state.update_system({...})              # 2. evolve framework System
    self.calculator.sync()                  # 3. propagate to task configs
```

`state.update_system(updates)` is a generalized update interface: it accepts a
dict of changed values (coordinates, velocities, electronic state fields, or
any future system field) and merges them into `pmd.system` with Pydantic
re-validation. It is not specialized to geometry, so future integrators can
pass any combination of system fields.

### 3.6 initialize(): Backend Lifecycle and Execution Namespace

Calculator introduces an explicit initialization stage before execution:

```python
calculator = Calculator(state, runtime_idx=None)
calculator.initialize()  # Instantiate all backends
calculator.run()         # Execute task queue
```

#### Backend Ownership

**Old design**: PyRAIState owned backend instances (created during `build_state()`).

**New design**: Calculator owns backend instances (created during `initialize()`).

```python
# PyRAIState stores only static task descriptors
task = {
    'api': 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend',  # string path
    'config': {'skill_config': <skill flat dict>, 'results': {}},  # plain dict
    'static_id': 'qm_layer0_pyscf',
    ...
}

# Calculator owns runtime backend instances, grouped by backend role
calculator.backends = {
    'theory':    [backend1, ...],
    'reference': [backend2, ...],
}                                                # created in initialize()
calculator.execution_ids = {'theory': [...], 'reference': [...]}
```

Backends are grouped by role following `BACKEND_ROLES = ('theory', 'reference')`.
The active role (`_active_role`, default `'theory'`) selects which queue
`run()` executes; `switch_role('reference')` switches it, but the two roles
are never executed simultaneously. Composite aggregation and projection
match the active role.

This separation ensures:
- PyRAIState is a pure data object (no Python class objects or instances)
- Backend lifetime is explicitly managed by Calculator
- Multiple Calculator instances can coexist (replica MD, adaptive sampling)

#### Execution Namespace

Each backend receives a unique `execution_id` that combines static planning information with runtime workflow context:

```
execution_id = static_id + runtime_suffix

Examples:
  qm_layer0_pyscf                    # single task, no runtime suffix
  qm_layer0_pyscf_replica_3          # replica MD
  qm2_layer1_molcas_walker_5         # adaptive sampling
  qm_layer0_pyscf_traj_8             # trajectory index
```

**Stage 1 (Planning)**: Static identifier

```python
static_id = f"{task_id}_layer{layer_id}_{skill_short}"

# Examples:
# qm_layer0_pyscf
# qm2_task1_layer1_xtb
# mm_layer0_molcas
```

Depends only on task order and selected skill; never changes during execution.

**Stage 2 (Workflow)**: Runtime suffix (optional)

```python
runtime_idx = 'replica_3'  # or 'walker_5', 'traj_8', etc.
execution_id = f"{static_id}_{runtime_idx}" if runtime_idx else static_id
```

Determined by Workflow at runtime; unknown during planning.

#### Backend Working Directory

Backend working directories are derived exclusively from `execution_id`:

```
scratch/
├── qm_layer0_pyscf/
├── qm_layer0_pyscf_replica_3/
├── qm2_layer1_xtb_replica_3/
└── ...
```

Backend never generates directory names itself. It receives only `execution_id` and constructs scratch, log, restart, and checkpoint paths using the same namespace. This eliminates manual directory management and avoids name collisions across concurrent executions.

#### initialize() Responsibilities

`Calculator.initialize()` performs:

1. **Backend instantiation**: For each task in `state.iter_tasks()`:
   ```python
   backend_cls = resolve_api(task['api'])  # dynamic import
   backend = backend_cls(config=task['config'], execution_id=execution_id)
   self.backends[category].append(backend)   # category: theory/reference
   ```

2. **Execution namespace assignment**: Combines `static_id` with optional `runtime_idx`

3. **Runtime preparation**: Initializes scratch directories, logging, and runtime objects

4. **Idempotent rebuild**: Repeated calls to `initialize()` clear and rebuild
   both role dicts (`backends` and `execution_ids` for theory and reference)

#### Lazy Import Mechanism

Planning stores backend API as a string import path:

```python
task['api'] = 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend'
```

No backend module is imported until `Calculator.initialize()` calls `resolve_api()`:

```python
def resolve_api(api_path):
    """Resolve backend class from import path string."""
    if isinstance(api_path, type):  # backward compatibility for tests
        return api_path
    module_name, cls_name = api_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)
```

This minimizes startup overhead while preserving dynamic plugin loading. The mechanism is identical to plugin loading used in NequIP, Hydra, PyTorch Lightning, and MMEngine.

#### Dispatcher Forwarding and Self-Created Executor

`Calculator(Dispatcher)` executes its single unit through
`self.dispatch()` — a single-point computation is a unit list of length
1, fully unified with batch execution:

```python
def run(self, target_states=None):
    # Stage the per-step target states for run_in_process(); the
    # local_executor closure is created once, so the value travels
    # through an instance attribute.
    self._step_target_states = target_states
    if self.local_executor is None:
        calculator = self

        def _in_process_executor(unit_state):
            del unit_state  # the calculator owns the live state
            return calculator.run_in_process()

        self.local_executor = _in_process_executor
    outcomes = self.dispatch(
        units=[{'state': self._state, 'task_index': 0}])
    return outcomes[0]
```

- `local_executor` is created by the Calculator itself in `run()`;
  `execute()` does not inject it. Calculator never imports the dispatch
  package.
- `run(target_states=None)` accepts the per-step target states chosen
  by the workflow (e.g. the NAMD workflow after a surface hop). The
  value is staged on `self._step_target_states`, consumed and reset by
  `run_in_process()`, and forwarded to every `backend.calculate()` of
  that run (task queue and `_run_single()` paths). None keeps each
  backend's initialization `calculation.target_states`; existing
  workflows call `run()` without arguments, so their behavior is
  unchanged.
- Universal rule 3: the calculator node never carries `remotetask=True`
  (the constructor rejects it); planning completely ignores the
  skill-section `remotetask` keyword and leaves it to the skill itself
  to decide its own local run or slurm run.
- See `WORKFLOW_DEVELOPMENT.md` Section 11 (Dispatcher and Dispatch
  Subsystem).

---

## 4. Composite

### 4.1 One-Direction Dependency

The dependency between calculator and composite is strictly one-directional:

- `calculator.py` imports and uses functions from `composite.py`.
- `composite.py` **never** imports or uses the calculator.
- The calculator does not need to know how results are aggregated.
- Composite does not need to know how task results are obtained.

All composite functions are **pure**: they never write to any state, config or
result container. State mutation is owned by the Calculator (option B).

### 4.2 Contribution Pattern Registry

> **Retired:** `COMPOSITE_PATTERNS`, `queue_contributions`,
> `validate_contributions` and `project()` were removed. `validate_plan` and
> `aggregate_results` are the live API. The block below is historical.

The purpose of ONIOM2/ONIOM3/QMMM is to establish the correct contribution
pattern for the task queue. Once the pattern is confirmed, the composite
result is just the sum of contributions multiplied by the task results. The
patterns are decided in the planning step (`resolve_task_queue`) and carried by
PyRAIState, so they are expressed as a registry instead of dedicated classes:

```python
COMPOSITE_PATTERNS = {
    'oniom2': [1, 1, -1],           # 3 tasks
    'oniom3': [1, 1, -1, 1, -1],    # 5 tasks
    'qmmm':   [1, 1],               # 2 tasks
}

AGGREGATE_KEYS = ('energy', 'gradient')
```

QMMM is **not** validated against that registry: its queue length depends on
`embedding`, on the center count and on `sc_charge`, so `validate_plan` checks
the duties structurally instead (one `energy`, field/QM counts equal when
embedded, and the charge duty carrying contribution 0 beside the field's). The
pattern table above stands for the fixed-length ONIOM and fragment shapes.

Former ONIOM2/ONIOM3 classes degenerated into simple validation + delegation
and were replaced by this registry plus pure functions.

### 4.3 Functions

| Function | Responsibility |
|----------|----------------|
| `queue_contributions(state, category=None)` | Collect contribution coefficients in execution order (`category` restricts to one backend role) |
| `validate_contributions(state, pattern_name, category=None)` | Validate the queue against a registered pattern (raises `ValueError` on missing state, unknown pattern or mismatch) |
| `aggregate_results(state, category=None)` | Pure aggregation: `Σ contribution × task_result` for `AGGREGATE_KEYS`, grouped by backend role (`category='theory'`/`'reference'`; `None` aggregates both queues); all other properties inherited from the primary task |
| `project(results, task)` | Gradient projection to screen out cap atoms (placeholder) |

### 4.4 Cap Atom Projection

`project()` screens cap atoms out of the gradient:

- Applies only to the **primary task** (the first task in the queue, i.e. the
  qm task); the Calculator calls it before aggregation.
- Without cap information it returns the results unchanged.
- The cap atom format is not decided yet, so the current implementation is a
  placeholder reserved for future integration.

---

## 5. Dual Result Write-back

Result write-back remains bidirectional, with clear responsibilities:

1. **Skill → task config**: the backend stores its raw results into
   `task['config'].results` (skill schema) at the end of `calculate()`.
   `store_task_result(task, result_dict)` is the fallback for backends that
   only return results (e.g. mock backends in tests): it writes only when the
   task config carries no results yet. `write_task_results(task, result_dict)`
   is the overwrite variant used after `project()`.
2. **Calculator → framework config**: the Calculator aggregates task results
   through `aggregate_results()` and writes the combined result into
   `state.pmd.results` (framework schema) via `state.set_result()`.

`state.result` is a read-only property returning a Result view of
`state.pmd.results`, so the aggregated result is always consistent with the
framework config.

---

## 6. Design Rules

### Rule 1: Calculator is self-contained

Calculator does not depend on any external iterative workflow. Iterative
workflows call `run()` and `sync()`; they never re-implement task execution.

### Rule 2: One-direction imports

`calculator.py → composite.py` only. Composite never imports calculator and
never mutates state.

### Rule 3: All state mutation goes through state.pmd

jobtype / structure / results are read-only pass-through properties of
`state.pmd`. Modifications only happen through `state.update_system()` and
`state.set_result()`; Workflow evolves the system, Calculator syncs it to task
configs.

### Rule 4: sync() only touches the system section

`Calculator.sync()` pushes `SYNC_SYSTEM_FIELDS` from `state.pmd.system` to
every task config. Calculation, method and control sections are never touched.

### Rule 5: Composite patterns are data, not classes

New composite schemes are added by expanding the task queue in
`planning/task.py` and by the pure `aggregate_results` / `validate_plan`
functions in `workflow/multiscale/composite.py`; no dedicated calculator
class and no fixed-length pattern table are needed.

### Rule 6: Test backends are named dummy_backend

Injected test backends in Calculator (constructor parameter, attribute,
`__repr__`) are uniformly named `dummy_backend` to avoid confusion with real
skill backends.

---

## 7. Verification

Calculator and Composite behavior is covered by
`qualification/verification/test_calculator.py` (see
[QUALIFICATION_DEVELOPMENT.md](QUALIFICATION_DEVELOPMENT.md)), including:

- Task queue execution and dummy_backend injection.
- Live geometry derivation via region_indices.
- Composite aggregation math (ONIOM2-style contribution patterns).
- `sync()` propagation into Pydantic and dict task configs.
- PyRAIState single-source-of-truth behavior (read-only properties,
  `update_system`, `set_result`).
- One-direction dependency assertion (composite module source contains no
  calculator import).
