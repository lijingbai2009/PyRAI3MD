# PyRAI3MD Data Core Development Guide

Last updated: 2026-11-09 16:40 (split into data/state.py + data/build.py; task assembly moved to planning/assembly.py; task persistence fields collected in data/task.py. The rest was not re-audited. Earlier: 2026-10-22 22:20 (both config trees now list the ten nested submodels plus `results`, `FIELD_TO_SUBMODEL` is described with `composite` registered and Hessian/Search/Train/Prediction routed by section name, the `static_id` examples use the `tmp_...` form, `fin_velocity`/`fin_cell` are recorded as consumed, the `api` field is a serialized import path, and section 7 no longer calls the constraint chain a placeholder - `operations.apply_constrain` delegates to the constraint runtime and the optimizer projects through the same plan. The rest was not re-audited. Before it, 2026-10-13 14:22: the `apply_constrain` location was checked against the code)

---

> **Multi-center unification (refactor R2-R6, 2026-09-08).** The data core is
> the partition axis of the unified multi-center base
> (`dev/MULTICENTER_DEVELOPMENT.md`; CHANGELOG #200-#204). `PyRAIState` persists
> `partition_plan` (`data/center.py` PartitionPlan/CenterSpec/ElectronicSpec,
> C=1 included; since CENTER_PLAN_VERSION 1.1 each CenterSpec carries its region
> `charge`/`multiplicity` and PartitionPlan carries the aggregated full-system
> `system_charge`/`system_multiplicity`), per-center `electronic[c]` carriers (independent
> energy/gradient/nac/soc/density arrays in each center's LOCAL framing, never
> padded into a global sum(S_c) axis), `active_surface`
> (active_configuration/active_energy/active_gradient/evaluation_id) and
> `continuation`. Public center interface: `center(c)`, `center_electronic(c)`,
> `set_center_electronic(c, ...)`, `get_active_configuration()` (C=1 returns
> `(s0,)`), `get_active_energy()`, `get_active_gradient()`, `active_gradient()`
> (R6: the assembled active-combination gradient the integrator reads; C=1
> degenerates to `gradient[current_state]`, C>1 returns the committed
> active_gradient and None when uncommitted), `stage_evaluation()` and
> `commit_step()` (the atomic writeback). `nstate` and `current_state` raise
> `CenterPlanError` for C>1 (read `center(c).nstate` /
> `get_active_configuration()[c]` instead). `update_history()` writes a
> structured `HistoryFrame` (nuclear_sample R/V_pre/K_pre/active_config_pre/
> active_gradient + centers[c] + electronic_outcome[c] + continuation +
> warmup); `get_history_step(index, copy=False)` keeps the read-only reference
> contract. The serializer carries a file-level version block
> (SERIALIZER_FORMAT_VERSION + partition/history/electronic semantic versions),
> covers the new fields, and migrates a version-less legacy `.rt` to a C=1
> record (a plan_fingerprint mismatch is refused as stale).
> The `K_pre` in that frame is the kinetic energy of exactly the `V_pre` stored beside it: the loop samples `kinetic_energy()` once, after the last action that writes velocities (#310). Before that the pair could disagree and the run log printed a third value -- which is the kind of thing a passing round never shows.

# 1. Purpose

This document defines the development principles for the PyRAI3MD data core.

The data core is the foundation of PyRAI3MD. It provides a unified representation of molecular and periodic systems, computational settings, calculation states, and simulation results.

The design goal is to create a data model that supports:

* Molecular and periodic systems.
* Quantum chemistry calculations.
* Machine learning models.
* Nonadiabatic molecular dynamics.
* Enhanced sampling methods.
* Workflow execution.
* Agent-based interaction.
* Checkpoint and restart.

---

# 2. Core Design Philosophy

The PyRAI3MD data core should follow:

```
Data-oriented design
+
Object-oriented interface
```

Objects define the organization and meaning of data.

Arrays provide efficient numerical computation.

The data core should not sacrifice computational performance for excessive object abstraction.

**Core Principle:**

```
Data Core describes the state of the computational world.

Skills determine how the world evolves.
```

---

# 3. State-Oriented Architecture (Task Queue)

The PyRAI3MD computational state is organized as two core objects.

**PyRAIConfig** — Unified configuration model (Pydantic model defined in `schema/config.py`):

```
PyRAIConfig

├── control: Control          ← job control and resource allocation
├── system: System            ← atomic structure and electronic settings
├── dynamics: Dynamics        ← MD simulation settings
├── adaptive: Adaptive        ← adaptive sampling parameters
├── optimize: Optimize        ← geometry optimization parameters (&optimize)
├── hessian: Hessian          ← numerical Hessian / vibrational analysis (&hessian)
├── search: Search            ← hyperparameter grid scan (&search)
├── train: Train              ← training request (&train)
├── prediction: Prediction    ← prediction request (&prediction)
├── composite: Composite      ← multi-layer / multi-center plan (&composite)
├── environment: Environment  ← computational environment snapshot (dispatch policy)
└── results: dict             ← calculation results written back by skills or Calculator
```

**PyRAIState** — Global state manager organized as a **Task Queue**:

```
PyRAIState

├── runtime: RuntimeState   ← runtime progress tracking
├── theory: list            ← theory task queue (list of task dicts)
├── reference: list         ← reference task queue (same structure)
└── pmd: PyRAIConfig        ← framework config instance (SOLE runtime data source)
```

`pmd` holds the framework PyRAIConfig directly (no dict wrapper). jobtype,
structure and result information are **read-only properties** derived from
`pmd` (e.g. `state.jobtype → pmd.control.jobtype`, `state.coordinates →
pmd.system.coordinates`, `state.result → Result view of pmd.results`); they
pass the pmd data through directly without rebuilding view objects, and
cannot be assigned from outside. All modifications go through
`state.update_system()` / `state.set_result()`, which write into pmd.

Access patterns:
- `state.theory[0]['api']` — backend API path string of the first task
- `state['theory']` — dict-like access via `__getitem__`
- `state.get_task('theory', 'qm', 0)` — locate a task by role and execution index
- `state.iter_tasks()` — iterate over all tasks (yields (category, task) tuples)

### 3.1 Task: The Minimal Running Unit

PyRAI3MD originally organized skills by **Role** (e.g. `qm`, `qm2`, `mm`), assuming
one Role == one Backend call. Composite workflows (ONIOM, Fragment, Embedding) break
this assumption. For example, ONIOM2 requires the low-level method to run twice:

```
E = E_high(model) + E_low(real) - E_low(model)
```

A single role (`qm2`) therefore maps to multiple backend invocations, so Role can no
longer uniquely identify one calculation. The state is now organized around **Tasks**:

- A **Task** is the minimal running unit: one independent backend invocation.
- Each task owns one task config (a plain dict); the backend instance is
  owned by the Calculator and created during `Calculator.initialize()`.
- The **Task Queue** order is exactly the Calculator execution order.
- **Role** remains only as a logical label (qm, qm2, mm) used for composite
  classification, not as the runtime organization.

Each task dict contains the complete key set produced by
`build_task_queue()` (planning/assembly.py) on top of the descriptors from
`_new_task()` (planning/task.py):

```
task = {
    'role': 'qm2',             # logical role in the workflow
    'task': 1,                 # execution index within the role
    'task_id': 'qm2_task1',    # '{role}' for the first invocation,
                               # '{role}_task{idx}' for repeats
    'layer_id': 1,             # layer number within the composite workflow
    'static_id': 'tmp_mid_qm2_task1_XTB',     # static execution namespace
    'contribution': -1,        # coefficient for composite aggregation
    'region': 'model',         # all / real / model / inter / fragment
    'region_indices': [...],   # pre-stored atom slice indices (geometry-free config)
    'region_charge': None,     # region charge (None = derived at runtime)
    'region_multiplicity': None,  # region spin multiplicity (None = runtime)
    'skill_name': 'xtb_skill',
    'api': 'pyrai3md.skills.xtb_skill.backend.XTBBackend',  # backend class
    'capabilities': [...],     # properties this task may compute
    'config': {                # plain dict task config (no PyRAIConfig)
        'skill_config': {...}, # merged flat skill dict
        'results': {},         # raw results written back by the skill
    },
    'multitask': None,         # dispatch-layer tri-state keyword
    'remotetask': None,        # dispatch-layer tri-state keyword
}
```

**Note:** Tasks no longer store backend instances. Backend instances are owned by Calculator and created during `Calculator.initialize()` (see [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md) Section 3.6).

Design constraints:

- **Geometry-free config**: task configs do NOT store sliced atoms/coordinates/charges.
  A config only initializes the backend. Region slice indices are pre-stored in the
  task metadata; the Calculator derives the actual atoms/coordinates/charges from the
  live `state.pmd.system` (through the state pass-through properties) at execution
  time, so iterative workflows always see updated geometry.
- **Capability restriction**: planning automatically limits non-primary composite
  tasks to `energy` + `gradient`.
- **Dual result write-back**: after execution, the skill backend writes raw results
  into its task config (`task['config']['results']`); the Calculator aggregates
  task results via the composite pure function and writes the combined result into
  the framework config (`state.pmd.results`, framework schema) through
  `state.set_result()`.

### 3.2 Task Queue Construction

Task queues are built inside `build_state()` by `build_task_queue()`:

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
resolve_task_queue(config, skills)      ← planning/task.py
    │   expands composite workflows into task descriptors
    ▼
For each task in the queue:
    ├── Merge skill config values into a flat skill dict (cached per skill)
    ├── Spell the skill jobtype from planning capabilities
    │   (driver=main + optimize: force 'energy gradient' and validate
    │    the optimize method against OPTIMIZER_REGISTRY)
    ├── Build the plain dict task config:
    │   {'skill_config': <flat skill dict>, 'results': {}}
    ├── compose static_id (e.g. 'tmp_mid_qm2_task1_XTB')
    └── state.add_task(category, task)
```

Every task receives its own task config dict, even when several tasks share the
same skill, because region metadata differs per task. `build_pyrai_config()` is
**deprecated** and no longer used by this path (kept only for backward
compatibility with external callers); the dynamic skill-submodel merging is gone.
Backend instances are **not** created here; they are instantiated later in
`Calculator.initialize()` (see [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md) Section 3.6).

This design ensures:
- The task config dict is the pure configuration input for each backend task
- PyRAIState manages all task metadata (role, task_id, layer_id, contribution,
  region, api, static_id, capabilities, multitask/remotetask)
- Task queues are built at a well-defined boundary (build_state)
- Backend instances are deferred to Calculator.initialize() (pure runtime objects)

### 3.3 Task Queue Architecture Principles

The refactoring establishes the following principles:

1. Task is the minimal running unit of PyRAIState.
2. The Task Queue is the only execution object of the Calculator.
3. Role only labels the logical position in a workflow; it is no longer the runtime organization.
4. Planning (`resolve_task_queue`) generates the Task Queue.
5. The Calculator always iterates the Task Queue and never inspects the workflow type.
6. Composite methods always aggregate results by contribution, independent of workflow type.
7. Each task owns an independent PyRAIConfig, preventing data pollution between tasks.
8. Backend instances are runtime objects and never participate in serialization.
9. PyRAIState saves the complete Task Queue, so complex composite workflows can be fully restored.
10. The framework config (`state.pmd`) is the sole runtime data source: it stores
    jobtype, system state and composite-aggregated results, while each skill config
    stores only the raw results of its own backend.

### 3.4 Skill Jobtype Translation

`planning/assembly.py` provides the reverse translation from planning capabilities back to a skill jobtype string:

- `JOBTYPE_TOKEN_ORDER`: fixed canonical ordering of basic jobtype tokens (`energy, gradient, nac, soc, dipole, transition, frequency, absorption`). The skill jobtype is always spelled in this order regardless of input order.
- `CAPABILITY_TO_TOKEN`: reverse mapping from capability name to jobtype token (`transition_dipole → transition`, `hessian → frequency`, etc.).
- `spell_skill_jobtype(capabilities, fallback)`: receives per-skill capabilities from planning, translates them to jobtype tokens via `CAPABILITY_TO_TOKEN`, and joins them in `JOBTYPE_TOKEN_ORDER`. When capabilities are unavailable the `fallback` (typically `state.jobtype`) is returned unchanged.

The driver=main + optimize special case (forcing the skill jobtype to
`energy gradient` and validating the optimize method against
`OPTIMIZER_REGISTRY`) is applied by `build_task_queue()` in `planning/assembly.py`,
not by `spell_skill_jobtype()`. Advanced jobtypes (md/surface_hopping/...)
pass through as-is since the skill plans the job internally.

---

# 4. Directory Structure

The Data Core follows a hybrid structure:

- Complex physical domains use directories.
- Simple state containers remain single files.

```
data/
├── core.py                  # PyRAIConfig + PyRAIState + build_state
├── center.py                # Multi-center partition plan (build_center_plan)
├── charges.py               # ChargeStore (ONIOM electrostatic embedding)
│
└── runtime/                 # Runtime State + Serialization
    ├── __init__.py          #   Public API re-exports
    ├── state.py             #   RuntimeState
    ├── result.py            #   Result (dict-like computation output)
    └── serializer.py        #   StateSerializer (sole format-knowing layer)
```

> The former `data/structure/` and `data/dynamics/` sub-packages were
> removed on Sep 25 2026 (CHANGELOG #216); see Section 5.

---

# 5. Structure State (removed)

The former `data/structure/` and `data/dynamics/` sub-packages were
**removed on Sep 25 2026** (CHANGELOG #216):

- `structure/`: AtomCollection, AtomView, Geometry, Cell, Topology,
  System.
- `dynamics/`: Dynamics, Ensemble, Thermostat, Barostat, Constraints,
  BondConstraint, AngleConstraint, DihedralConstraint, CavityConstraint,
  SHAKEParameters, Compressor, CollectiveVariable, Metadynamics,
  EnhancedSampling.

Both were deprecated on Aug 03 2026 ("parameters will be moved to a
derivation module") and had **zero consumers** outside `data/__init__.py`;
their unconditional import also forced every `import pyrai3md.data` to
emit a DeprecationWarning (now gone).

**Why removed rather than wired in:**

- The structure containers duplicated the authoritative runtime source
  `pmd.system` (the schema `System` model): atoms, coordinates,
  velocities, masses, point charges and the periodic cell
  (pbc/cell/lattice_vector/volume) all live there and are read through
  the PyRAIState pass-through properties. `image` (the per-atom count of cell
  lengths the coordinate wrap made an atom cross) joins that block as runtime
  state: persisted with the history frame and the checkpoint, never an input
  keyword, never synced to a backend. Continuous coordinates are a derived
  view of `coordinates + image + lattice_vector`, not a second copy of the
  geometry. `System.from_xyz_file`
  duplicated `utilities/io.read_coord`; the element tables duplicated
  `schema/reference.py`; `Topology` was an empty container (it stored
  bonds/angles but had no builder).
- The dynamics containers were pure data holders with zero algorithms.
  The live ensemble implementation is `workflow/integrator/ensemble.py`
  (NHC / CSVR / Berendsen / MTTK-Parrinello-Rahman) driven by the
  `schema/config.py` dynamics keywords — a different, already-wired
  parameter model that the containers never connected to.

**Where the functionality lives now:**

- Periodic-cell math (6-parameter <-> matrix, volume, wrapping,
  cart/frac, validation, boundary_condition) → `utilities/lattice.py`
  (pure functions; the runtime cell state stays in `pmd.system`). See
  `dev/methods/PBC_DEVELOPMENT.md`.
- Constraints / SHAKE / walls and CV / metadynamics are implemented in
  the runtime chain: `workflow/constraints/` carries the subsystem
  (`runtime.py::constrain_state` is reached from
  `operations.apply_constrain`, `workflow/integrator/operations.py:156-187`,
  and the optimizer projects frozen directions out of the trial step at
  `workflow/optimizer/base.py:266`). Their design and field lists are
  owned by `dev/methods/CONSTRAINT_DEVELOPMENT.md`; what remains in
  `dev/methods/ENSEMBLE_DEVELOPMENT.md` §4 is the historical field memo.

## 5.1 ElectronicState (removed earlier)

The `ElectronicState` class (`electronic.py`) was **removed on
Aug 09 2026**. It duplicated information already owned by the
authoritative data source `pmd.system` (charge, multiplicity,
nstate, current_state, state labels/spins) and was unused outside
the deprecated structure sub-package.

The active electronic state now lives exclusively in
`pmd.system.current_state` (see Section 6.6).

---

# 6. PyRAIConfig and PyRAIState (core.py)

`PyRAIConfig` and `PyRAIState` are the two core objects of PyRAI3MD.

## 6.1 PyRAIConfig

`PyRAIConfig` is a Pydantic model defined in `schema/config.py`. It nests ten
submodels plus the `results` write-back dict and supports both nested access
(`config.control.jobtype`) and flat access (`config.jobtype`):

```
PyRAIConfig (BaseModel)

├── control: Control
├── system: System
├── dynamics: Dynamics
├── adaptive: Adaptive
├── optimize: Optimize
├── hessian: Hessian
├── search: Search
├── train: Train
├── prediction: Prediction
├── composite: Composite
├── environment: Environment
└── results: Dict[str, Any]   ← write-back channel for calculation results
```

- `Optimize` (framework-level `&optimize` section; field names carry the
  `opt_` prefix to avoid collisions with skill sections):
  `opt_method` (minima, gdiis, ...), `opt_maxit`, `opt_energy`, `opt_grms`,
  `opt_gmax`, `opt_drms`, `opt_dmax`, `opt_use_rfo`, `opt_state_i`,
  `opt_state_j`.
- `Environment` (computational environment snapshot, populated by the
  adapter and travelling with PyRAIState so the dispatch subsystem never
  reads the kernel Environment singleton): `shell`, the export map
  `environment`, the merged `slurm` section and, since #306,
  `scratch_path`. The snapshot copies exactly those four; the model's
  remaining fields (`program`/`version`/`device`, `program_path`,
  `slurm_*`) keep their schema defaults even when the input's
  `&environment` section spells them -- they carry no live consumer, and
  keeping them in the schema is what aligns the framework model with the
  skill schemas. The four copied ones do have consumers:
  `workflow/dispatch/slurm_run.py` builds the child runtime script from
  `shell`/`environment`/`slurm`, and `pmd.py` opens the run log in
  `scratch_path` (the current directory when it is unset).
  `Scheduler.validate_resources()` reads the Slurm allocation from the
  same `slurm` section. It carries no execution switch: `execution_mode`
  is a skill keyword, and when the framework's own dispatch says remote,
  `build_task_queue` writes the derived `execution_mode` into each
  task's skill config (#300).
- Compute-resource counts are not carried here, and the run-wide
  `num_thread` floor #303 imposed on skills is cancelled with the field
  (#306). A thread count is a property of the calculation, so it lives in
  the skill section as `thread`/`nproc` or, in the framework's spelling,
  `<skill>_thread`/`<skill>_nproc`; each skill folds a pair to its larger
  value in its own schema, and the framework only reads what the queue
  already carries in order to warn about oversubscription
  (`dev/SKILL_APP_DEVELOPMENT.md` section 5.7).
- Flat access is routed through `FIELD_TO_SUBMODEL`, which registers the
  `control` / `system` / `dynamics` / `adaptive` / `optimize` / `composite`
  sections; `environment` and `results` are nested-only, and
  `hessian` / `search` / `train` / `prediction` stay out of the flat map on
  purpose because their keyword names collide with other workflow sections -
  they are routed by section name through `NESTED_FRAMEWORK_SECTIONS`
  (`schema/config.py:2130-2141`).

`build_pyrai_config()` has been **removed**: the framework creates a
dict-based task config directly in `planning/assembly.py`, and the helper
had no caller left. A caller holding the old attribute should read the task
config dict instead.
The former dynamic skill-submodel merging (`create_model`) is gone:
`Properties` / `Calculation` belong to the skill schema vocabulary, not to
the framework submodel structure.

The `results` field is the dual write-back channel:

- Skill backends write their raw calculation results into the **task config**
  (`task['config'].results`, skill schema) at the end of `calculate()`.
- The Calculator writes the composite-aggregated result into the **framework
  config** (`state.pmd.results`, framework schema) via `state.set_result()`.
  The composite aggregation (`aggregate_results()` in `workflow/multiscale/composite.py`)
  is a pure function returning the combined dict; it never writes state itself.

## 6.2 PyRAIState

Global state manager organized as a Task Queue. `pmd` (the framework
PyRAIConfig) is the **sole runtime data source**:

```python
class PyRAIState:
    def __init__(self, runtime=None):
        self.runtime = runtime       # RuntimeState
        self.theory = []             # task queue (list of task dicts)
        self.reference = []          # task queue (list of task dicts)
        self.pmd = None              # PyRAIConfig instance (sole data source)
        self.execution_graph = None  # frozen execution graph (root builds,
                                     # worker verifies the parent-frozen one)
        self.resolve_execution_graph = True  # root runtimes resolve the
                                     # graph; dispatched workers skip planning
        self._history = []           # bounded snapshot list (internal only)
```

Read-only pass-through properties (no setters; derived from pmd):

| Property | Source |
|----------|--------|
| `jobtype` | `pmd.control.jobtype` |
| `driver` | `pmd.control.driver` (falls back to `'main'`) |
| `atoms` / `coordinates` / `velocities` / `mass` | `pmd.system.<field>` |
| `mass_au` | `pmd.system.mass` converted to atomic units through `utilities.units.amu_to_au`; `None` when masses are missing, their count differs from `system.natom` or any value is non-positive (the single mass-unit access point used by verlet/fssh/`kinetic_energy()`) |
| `charges` / `multiplicity` / `nstate` / `current_state` | `pmd.system.<field>` (for C>1 `charges`/`multiplicity` keep the per-center list; the scalar each task actually runs with is the injected `region_charge`/`region_multiplicity`, and the full-system aggregate lives on `partition_plan.system_charge`/`system_multiplicity`) |
| `timestep` | `pmd.dynamics.size` (falls back to 20.67 au) |
| `root` / `activestate` | `pmd.dynamics.<field>` |
| `optimize_method` / `optimize_maxit` / `optimize_params` | `pmd.optimize.opt_*` |
| `result` | `Result` view of `pmd.results` (None when empty) |
| `history_limit` | `pmd.dynamics.history` (falls back to 4 without pmd) |
| `history_size` / `has_history` | current snapshot count / non-empty |

Step-level methods:

- `kinetic_energy()` — computes, stores (`runtime.kinetic_energy`) and
  returns the nuclear kinetic energy in Hartree
  (`0.5 * sum(mass[au] * velocity^2)` with masses read from the
  `mass_au` property; 0.0 when data is unavailable).
- `record_step_result()` — records the runtime metadata of the current step
  through the official API (property data stays in pmd).

Update interface (the sole legal modification path):

- `update_system(updates)` — generalized system update: accepts a dict of
  changed values (coordinates, velocities, electronic state fields, or any
  future system field) and merges them into `pmd.system`. Implemented through
  `_update_section(section, updates)`, which merges 'current values + updates'
  and rebuilds the submodel via Pydantic `model_validate`; extensible to other
  sections (e.g. dynamics).
- `set_result(result_dict)` — writes calculation results into `pmd.results`
  (numpy arrays converted to lists for serialization).

Task queue methods:
- `add_task(category, task)` — append a task to a category queue
- `get_task(category, role, task_idx=0)` — locate a task by role and execution index
- `iter_tasks()` — iterate (category, task) pairs in execution order (theory first)
- `n_tasks()` — total task count across both queues
- `to_dict()` — serialize the state into a complete dictionary (see Section 13)
- `__getitem__(key)` / `items()` — dict-like access

History API (bounded snapshot list, full design in Section 18):

- `_history` is a plain list in chronological order (index 0 oldest, -1
  newest). Snapshots are deep copies of the full `pmd.model_dump()` taken
  when a timestep completes; the current state never enters the history.
- `update_history()` — append a snapshot of the current pmd; when the list
  exceeds `history_limit` only the last N snapshots are kept
  (`self._history = self._history[-limit:]`).
- `get_history(n=None)` — deep copies of all (or the last n) snapshots;
  requesting more than stored returns what exists.
- `get_history_step(index=-1, copy=True)` — one snapshot; returns None
  when the step does not exist (workflows decide how to handle it).
  The default deep-copies the snapshot; `copy=False` returns a
  read-only reference into the internal snapshot list for hot-loop
  readers (e.g. the FSSH integrator), where the caller must not
  mutate the returned dict or anything it holds.
- `clear_history()` — remove all snapshots.
- The capacity comes from the input keyword `&MD history N`
  (`pmd.dynamics.history`); the schema pipeline normalizes values below 4
  to 4 (`normalize_history` in schema/normalize.py). PyRAIState only knows
  the bound, never its physical meaning; workflows interpret the
  snapshots (NAMD/FSSH/Verlet/Adaptive). External code never accesses
  `_history` directly; all getters return copies unless the read-only
  view (`copy=False`) is explicitly requested.

Workflow, Calculator and future integration code intentionally share the same
mutable `pmd` object during runtime; see `CALCULATOR_DEVELOPMENT.md`.

## 6.3 build_state

Creates fully initialized PyRAIState from validated Config:

```python
def build_state(config, raw_config=None):
    from ..kernel.environment import Environment
    from ..planning.assembly import build_task_queue
    env = Environment()
    maestro = env.get_maestro()

    state = PyRAIState()
    state.pmd = config   # framework PyRAIConfig as the sole runtime data source

    # Resolve skills and build the task queue
    skills = maestro.resolve_skills(config)
    build_task_queue(state, config, skills, maestro, raw_config=raw_config)

    return state
```

`build_state()` lives in `data/build.py` (the state container is
`data/state.py`); the task assembly it calls lives in
`planning/assembly.py`. Internally:
1. Gets Maestro from Environment (singleton)
2. Creates PyRAIState and attaches the framework PyRAIConfig as `state.pmd`
   (jobtype/system/results are derived from it via read-only properties; no
   separate jobtype assignment)
3. Calls `maestro.resolve_skills(config)` to resolve all skills and capabilities
4. Calls `build_task_queue()`, which runs `resolve_task_queue()` (planning/task.py)
   to expand composite workflows, then builds one task-specific PyRAIConfig per
   task (with API path string and static_id) and registers them via
   `state.add_task()`. Backend instances are **not** created here; they are
   instantiated later in `Calculator.initialize()`.

Returns a fully initialized PyRAIState whose task queues are ready to execute.

## 6.4 Data Flow Summary

```
build_state(config)
    → state = PyRAIState (pmd attached, task queues built, no backend instances yet)

execute(state)
    → ExecutionGraphBuilder.build() → freezes state.execution_graph
    → calculator = Calculator(state).initialize() → backend instantiation
    → workflow_cls = resolve_workflow(state)
    → if workflow_cls: Workflow(calculator=calc).run()
      (Workflow creates its own local_executor from calculator)
    → else: calculator.run()
```

Example (ONIOM2 queue):

```
state.theory = [
    {'role': 'qm',  'task': 0, 'contribution': +1, 'region': 'model', ...},
    {'role': 'qm2', 'task': 0, 'contribution': +1, 'region': 'real',  ...},
    {'role': 'qm2', 'task': 1, 'contribution': -1, 'region': 'model', ...},
]
state.jobtype == 'optimize'   (read-only property of state.pmd)

execute(state) → Calculator(state) → Optimize(calculator).run()
```

## 6.5 Method (Deprecated)

The `Method` class is deprecated. Its functionality has been merged into PyRAIState,
which now manages task metadata directly via the theory/reference task queues.

New code should access `state.theory` and `state.reference` directly.

## 6.6 Electronic State Data Flow

The active electronic state has a **single authoritative source**:
`pmd.system.current_state` (0-based).

```
Initialization
    normalize pipeline:  derive_current_state(data) fills current_state = root
                         only when current_state is the default 0;
                         explicit non-zero values are never overwritten
    optimize workflow:   state.update_system({'current_state': state_i})

NAMD surface hopping
    hop >= 0 → state.update_system({'current_state': hop})
               (official write path only; direct writes elsewhere
               are prohibited)

Propagation to backends
    calculator.sync()
        → SYNC_SYSTEM_FIELDS includes 'current_state'
        → pushed to every task config's system.current_state

Read shortcuts
    state.current_state  (read-only property of PyRAIState)
    state.root           (read-only property, initial state from
                          pmd.dynamics.root)
```

Notes:

- `RuntimeState` no longer tracks the electronic state; the former
  `current_electronic_state` field was removed (legacy serialized
  runtimes carrying it are silently ignored by `from_dict`).
- `calculation.target_states` (List[int]) defines which states a
  skill computes; it is an input of the task config, not mutated by
  the framework.
- Per-step dynamic target_states: `workflow/md.py::resolve_target_states(activestate, current_state, nstate)`
  derives the list each MD step (`activestate=0` → all states, otherwise `[current_state]`)
  and forwards it through `calculator.run(target_states=...)` → `backend.calculate()`.
  The framework never mutates a skill's calculation section directly.

## 6.7 Result Channels (fin_coord / fin_velocity / fin_cell / computed_charges)

Besides the standard property channels (energy, gradient, nac, ...), the
result vocabulary carries three structure-returning channels plus the charge
channel:

- `fin_coord` (#101): final structure `(natom, 3)` in Angstrom reported by
  the skill's internal iteration (optimize/md/...). Empty means the skill
  performed no internal iteration; skill-driven geometry optimization
  (driver != 'main') returns its converged structure through this channel.
- `fin_velocity` (#153, consumed since #235): final velocities `(natom, 3)` in
  Bohr/au reported by skill-internal MD propagation. `Calculator` consumes it
  through the same contract as `fin_coord`
  (`workflow/calculator.py:1494 _apply_skill_velocities`), stores it as
  `system.velocities` and recomputes the kinetic energy; the key is popped so it
  never reaches `state.results`.
- `fin_cell`: the box the returned coordinates actually live in, as the six cell
  parameters or the nine flattened lattice components; consumed by
  `workflow/calculator.py:1558 _apply_skill_cell`, which derives `cell`,
  `lattice_vector` and `volume` together and refuses a value for a system
  declared non-periodic.
- `computed_charges`: `(nstate, natom)` atom charges computed by the skill,
  extracted to update the background charges. This is a *computed*
  quantity and distinct from `system.point_charges`, the `(ncharge, 4)`
  background embedding charges `[q, x, y, z]` supplied by the input.

---

# 7. Dynamics State (removed)

The former `data/dynamics/` sub-package (Dynamics, Ensemble, Thermostat,
Barostat, Constraints, CollectiveVariable, EnhancedSampling,
Metadynamics, ...) was **removed on Sep 25 2026** (CHANGELOG #216); see
Section 5 for the rationale (pure data containers, zero algorithms, zero
consumers, deprecated since Aug 03 2026).

**How the system evolves now:**

- The live ensemble/integrator implementation is
  `workflow/integrator/ensemble.py` (NVE / NVT thermostats NHC + CSVR /
  NPT barostats Berendsen + MTTK-Parrinello-Rahman) and
  `workflow/integrator/verlet.py`, driven by the `schema/config.py`
  dynamics keywords (`ensemble`, `thermostat`, `barostat`,
  `temperature`, `pressure`, `thermostat_tau`, `barostat_tau`, ...).
  See `dev/methods/ENSEMBLE_DEVELOPMENT.md` and
  `dev/methods/MD_DEVELOPMENT.md`.
- Constraints / SHAKE / walls, collective variables and metadynamics are
  **implemented in the runtime chain**: `operations.apply_constrain` delegates to
  `workflow/constraints/runtime.constrain_state`
  (`workflow/integrator/operations.py:156-187`) and the optimizer's
  `apply_constraint` builds its projection from the same plan
  (`workflow/optimizer/base.py:608-655`). The subsystem is
  `pyrai3md/workflow/constraints/` and its design document is
  `dev/methods/CONSTRAINT_DEVELOPMENT.md`; the section above this one, last
  re-read on 2026-10-13, already describes it. The retired cavity family and the
  archived field names (CavityConstraint / Compressor) survive only as the memo in
  `dev/methods/ENSEMBLE_DEVELOPMENT.md` §4.

---

# 8. Runtime State (runtime/state.py)

Tracks **current execution status**.

```
RuntimeState

├── status (initialized, running, completed, failed)
├── current_step
├── total_steps
├── simulation_time
├── iteration
├── random_seed
├── checkpoint_file
├── wall_time
├── error_message
├── worker                  ← root runtime False; dispatch worker True
├── active_role             ← backend role the runtime executes
└── step_results            ← per-step result tracking
```

Supports checkpoint and restart.

The active electronic state is **not** tracked here; it lives in
`pmd.system.current_state` (sole authoritative source, see
Section 6.6).

Dispatch semantics (see `WORKFLOW_DEVELOPMENT.md` Section 11):

- `worker=False` (constructor default) marks a **root** runtime that may
  plan and dispatch. `from_dict()` defaults missing fields to `False`,
  so old serialized runtimes with the legacy `dispatchable` field are
  treated as workers (`worker = not dispatchable`) and never re-plan.
- The frozen child execution graph travels with the PyRAIState
  (`execution_graph` / `resolve_execution_graph`), not with RuntimeState.
- `active_role` ('theory' or 'reference', default 'theory') is stamped by
  `Dispatcher.materialize_unit` when a unit is dispatched for an explicit
  backend role, and read back by `workflow.execute()` to construct the
  Calculator on that role. It is a record, not a validator (the
  Calculator owns the role vocabulary). A `.rt` written before the field
  existed is unpickled without `__init__`, so the class-level default is
  what keeps it readable.

The StateSerializer (`data/runtime/serializer.py`) is the sole
serialization layer for PyRAIState. See Section 13.

---

# 9. Result State (result.py)

Stores **computed properties**.

```
Result

├── energy
├── gradient
├── nac
├── soc
├── dipole
├── populations
├── trajectory_files
└── properties
```

Result objects are transferable between Skills.

---

# 10. Central State Objects (core.py)

`PyRAIConfig` and `PyRAIState` are the two core objects:

**PyRAIConfig** — Unified configuration model:

```
PyRAIConfig

├── control / system / dynamics / adaptive / optimize /
│   hessian / search / train / prediction / composite     ← nested framework submodels
├── environment: Environment ← environment snapshot (slurm, shell — the
│                              dispatch subsystem reads only
│                              state.pmd.environment; no execution switch)
└── results: dict            ← calculation result write-back channel
```

(Properties / Calculation belong to the skill schema vocabulary used in
skill configs, not to the framework submodel structure.)

**PyRAIState** — Global state manager (Task Queue):

```
PyRAIState

├── runtime: RuntimeState   ← progress tracking
├── theory: list            ← task queue (one dict per backend invocation)
├── reference: list         ← task queue (same structure)
└── pmd: PyRAIConfig        ← sole runtime data source (jobtype, system,
                               results are read-only properties of it)
```

Each task dict contains:
- `role`: Logical workflow role (qm, qm2, mm, ref1...)
- `task`: Execution index within the role
- `task_id`: Semantic identifier (`{role}`, or `{role}_task{idx}` for repeats)
- `layer_id`: Layer number within the composite workflow
- `contribution`: Coefficient for composite aggregation (+1 / -1)
- `region`, `region_indices`, `region_charge`, `region_multiplicity`: Region slice metadata
- `skill_name`: Skill identifier
- `api`: the backend's **import path as a string**; it is serialized unchanged and
  rebuilt by `Calculator.initialize()` (the live backend instance is what stays out
  of the payload)
- `static_id`: Static execution namespace
  (`tmp_{SKILL_SHORT}` for a single task, `tmp_{region}_{role}[_task{N}]_{SKILL_SHORT}`
  for a composite; the former `tmp_{task_id}_layer{layer_id}_{SKILL}` form is retired,
  `planning/task.py:125-147`)
- `capabilities`: Properties this task may compute
- `config`: Plain dict task config `{'skill_config': flat, 'results': {}}`
  (holds raw skill results)
- `multitask` / `remotetask`: Dispatch-layer tri-state keywords

The state object should be:

* Transferable between modules.
* Copyable.
* Serializable.
* Restartable.
* Independent of specific computational backends.

---

# 11. Array-based Atomic Storage

Atomic properties should be stored using array-based structures.

```
AtomCollection

├── atomic_numbers: (natom,) int
├── symbols: (natom,) str
├── masses: (natom,) float64
├── charges: (natom,) float64
├── atom_types: (natom,) str
└── properties: dict
```

Advantages:

* Efficient numerical operations.
* Compatible with NumPy.
* Compatible with ML frameworks.
* Suitable for large systems.

---

# 12. Atom Object Interface

Object-oriented access should still be provided.

```
AtomView

        ↓

AtomCollection

        ↓

Underlying arrays
```

AtomView is used for:

* User interaction.
* Agent queries.
* Chemical analysis.

It should not be the primary storage format.

---

# 13. Serialization Strategy

The data core supports persistent storage through the
`StateSerializer` (`data/runtime/serializer.py`).

```
PyRAIState

↓

StateSerializer.save()  →  Runtime file (.rt, currently pickle)
```

**Core principle: serialization decoupled from data model.**
`StateSerializer` is the **sole** module that knows the storage
format. PyRAIState, Workflow, Calculator and Skill never know whether
the format is pickle, HDF5 or anything else.

```
PyRAIState  ──(format-agnostic)──>  StateSerializer  ──(format-aware)──>  pickle
```

Dependency direction: PyRAIState → StateSerializer → storage format.
The format adapts to the state; the state never adapts to the format.
When the format changes, only `serializer.py` changes.

The `.rt` file stores the complete PyRAIState plus the workflow
`runtime_idx`: the task queues with full task metadata (`role`, `task`,
`task_id`, `layer_id`, `static_id`, `contribution`, `region`, skill name,
capabilities, dispatch flags `multitask` / `remotetask` and the dict task
configs), the pmd runtime data source, the runtime variables and previous
results, the frozen `execution_graph` with `resolve_execution_graph`, and
the bounded `history` snapshot list. Backend classes (`api`) and instances
are runtime objects and are **excluded**; on restart, backends are rebuilt
from the saved task queue by `Calculator.initialize()`:

```
Restart:

Load State → StateSerializer.load() → PyRAIState
           → Calculator.initialize() → Instantiate Backend → Continue Workflow
```

Because PyRAIState saves the full task queue, even complex composite
workflows can be restored completely.

Task configs are plain dictionaries and pickle-safe by construction;
`_prepare_state` / `_prepare_task` additionally flatten any legacy
BaseModel configs into plain dicts, since skill backends in standalone
mode natively accept dict configs.

Public API:

| Function | Responsibility |
|----------|----------------|
| `save_state(state, path, runtime_idx)` | Serialize PyRAIState to file |
| `load_state(path)` | Restore PyRAIState from file |
| `prepare_state(state)` | Return pickle-safe state copy (for multiprocessing) |
| `shallow_copy_state(state)` | Shallow copy (shared numpy arrays) |
| `deep_copy_state(state)` | Deep copy (independent arrays) |

Current format: pickle (protocol `HIGHEST_PROTOCOL`). Future versions
may switch to HDF5 or other formats without changing any code outside
`serializer.py`.

---

# 14. Copy and Data Transfer

The state object should support controlled copying.

## Shallow copy

For:

* Workflow transfer.
* Temporary modification.

## Deep copy

For:

* Branch calculations.
* Independent simulations.
* Optimization trials.

Copy behavior should be explicitly defined.

---

# 15. Performance Requirements

The data core should prioritize:

* Contiguous numerical storage.
* Vectorized operations.
* Minimal object traversal.
* Efficient memory usage.

Avoid:

* Per-atom Python loops.
* Repeated data conversion.
* Unnecessary serialization.
* Duplicate large arrays.

---

# 16. Development Rules

When modifying the data core:

1. Preserve backward compatibility whenever possible.
2. Avoid introducing backend-specific information.
3. Avoid mixing workflow logic into data classes.
4. Add serialization tests for new fields.
5. Add performance tests for large systems.
6. Validate with representative molecular and periodic systems.
7. Follow the separation: data stores state, skills execute algorithms.

---

# 17. Long-term Goal

The PyRAI3MD data core should become a universal scientific state representation.

It should provide a common language between:

```
Quantum Chemistry

↓

Machine Learning

↓

Nonadiabatic Dynamics

↓

Enhanced Sampling

↓

Workflow Engine

↓

AI Agent
```

The data core is not only a storage structure.

It is the foundation for the future PyRAI3MD scientific computing ecosystem.

---

# 18. PyRAIState History Design (Design Specification)

This section is the merged design specification for the PyRAIState history
feature (formerly `dev/data_core_patch.md`). It defines the rationale,
semantics and API rules summarized in Section 6.2.

## 18.1 Design Goal

Give PyRAIState a general history-management capability that keeps the most
recent few `pmd.state` snapshots produced while a workflow runs.

Main application scenarios:

- NAMD / FSSH need the energy, gradient, NAC and SOC of the last few steps;
- Verlet-style time integrators need the states of previous steps;
- Adaptive Sampling needs the last N trajectory states;
- Any future workflow or algorithm that requires time-history information.

History is runtime data of PyRAIState, but it is **not** part of PyRAIConfig.

Overall structure:

```
PyRAIState
├── pmd
│   ├── configuration
│   └── state                 # current state
├── theory
├── reference
├── execution_graph
├── resolve_execution_graph
└── history                   # historical state snapshots
```

where:

```
pmd.state      = current state
history        = the most recent few historical states
```

History stores snapshots of `pmd.state`, not complete PyRAIState objects.

## 18.2 Core Design Principles

### 18.2.1 History Must Not Couple to Specific Workflows

PyRAIState must not know about NAMD, FSSH, Adaptive Sampling, Verlet,
Surface Hopping or any other concrete algorithm. History only provides:

- saving historical states;
- retrieving historical states;
- controlling the history length;
- clearing the history.

The concrete workflow decides how to interpret the historical data.

### 18.2.2 History Stores a Complete State Snapshot

Every history update copies and stores the current `state.pmd.state`:

```python
# conceptually
history.append(copy(state.pmd.state))
```

instead of maintaining separate `energy history`, `gradient history`,
`NAC history`, `SOC history`, `coordinate history`, ... lists. This keeps a
unified data interface for all workflows. For example, NAMD can obtain
energy, gradient, NAC, SOC, coordinates and velocities from the historical
states, while Adaptive Sampling can obtain coordinates and energy, without
any change to History itself.

## 18.3 History Length

The history length is controlled by an input parameter, e.g.:

```
&MD
history 10
```

meaning at most the last 10 historical states are kept.

For NAMD the history length must not be smaller than 4, because some
electronic dynamics algorithms need at least the last 4 timesteps:

```
history < 4       → invalid
history >= 4      → valid
```

This constraint is enforced by the input/schema pipeline (the
`normalize_history` step), not by PyRAIState itself — PyRAIState must never
understand the physical meaning of "4 steps". History only knows its
`max_length`.

## 18.4 History Update Mechanism

History follows FIFO / bounded-queue behavior. With `history = 4`:

```
initial:          []
after 1st save:   [t0]
after 2nd:        [t0, t1]
after 3rd:        [t0, t1, t2]
after 4th:        [t0, t1, t2, t3]
after 5th:        [t1, t2, t3, t4]
```

Only the most recent N snapshots are kept.

## 18.5 History Update API

Workflows are forbidden from operating on the history directly:

```python
state.history.append(...)   # forbidden
state.history.pop(...)      # forbidden
state.history.clear()       # forbidden
state.__dict__["history"]   # forbidden
```

All modifications must go through the official PyRAIState API:

```python
state.update_history()
```

Semantics: deep-copy the current `pmd.state` into a snapshot and append it
to the history. Conceptually:

```python
def update_history(self):
    snapshot = deepcopy(self.pmd.state)

    if len(self._history) >= self._history_size:
        self._history.pop(0)

    self._history.append(snapshot)
```

`_history` is an internal implementation detail; external code must never
access it.

## 18.6 History Retrieval API

A unified official retrieval function:

```python
state.get_history(n=None)
```

- `get_history()` returns the whole current history `[t0, t1, t2, ...]`;
- `get_history(n)` returns the most recent n snapshots, e.g.
  `state.get_history(3)` → `[t2, t3, t4]`.

If `history = 10` but only 5 steps exist, `state.get_history(10)` returns
the 5 existing snapshots — it never fabricates data.

## 18.7 Single-Step History Access

```python
state.get_history_step(index)
```

- `state.get_history_step(-1)` — the most recent historical state;
- `state.get_history_step(-2)` — the state two timesteps ago.

Workflows do not need to know the internal list implementation:

```python
previous  = state.get_history_step(-1)
previous2 = state.get_history_step(-2)
previous3 = state.get_history_step(-3)
```

Especially suitable for FSSH / Verlet style algorithms.

Hot-loop readers that only read the snapshot may request the read-only
view to skip the per-step deep copy:

```python
previous = state.get_history_step(-1, copy=False)
```

The caller must treat the returned dict (and everything it holds) as
read-only; any write must go through the official update interface.
The default (`copy=True`) keeps the safe deep copy for all other
callers.

## 18.8 Property API

Frequently used information may be exposed as read-only properties:

```python
@property
def history_size(self):
    return len(self._history)

@property
def has_history(self):
    return len(self._history) > 0
```

If needed, a convenience property may be offered:

```python
@property
def previous_state(self):
    return self.get_history_step(-1)
```

But properties like `previous_energy`, `previous_gradient`, `previous_nac`,
`previous_soc`, ... are discouraged — they should be read from the
historical snapshots, keeping the PyRAIState API free of algorithm-specific
pollution.

## 18.9 Mutability of Returned History

Returned history data must not let external code mutate the internal
history. `get_history()` must return copies (or immutable views), never the
internal `_history` itself — otherwise `history[-1].energy = ...` could
modify PyRAIState's internal history.

## 18.10 Time Semantics: Current State vs. History

It must be explicit that:

```
state.pmd.state  = the current timestep
state.history    = snapshots of completed timesteps
```

So `history[-1]` is the most recent state **before** the current one:

```
history: [t(n-3), t(n-2), t(n-1)]
current: t(n)
```

never `[t(n-3), t(n-2), t(n-1), t(n)]`. This keeps the semantics of

```python
current  = state.pmd.state
previous = state.get_history_step(-1)
```

unambiguous at all times.

## 18.11 Usage in NAMD

The NAMD workflow does not maintain history itself:

```python
state = calculator.get_state()
state = verlet(state)
state = surface_hopping(state)
state.update_history()
calculator.update(state)
```

(or `update_history()` is placed at the point where the current timestep is
considered complete, depending on the exact timestep definition).

Inside FSSH:

```python
current   = state.pmd.state
previous  = state.get_history_step(-1)
previous2 = state.get_history_step(-2)
previous3 = state.get_history_step(-3)
```

providing the data of t, t-1, t-2 and t-3.

## 18.12 Usage in Adaptive Sampling

Adaptive Sampling can simply call:

```python
samples = state.get_history(n=10)
```

and extract coordinates, energy, forces, ... according to its own algorithm.
No extra `history` argument is needed (`adaptive_sampling(state, history)`
is unnecessary); the unified `adaptive_sampling(state)` interface is kept.

## 18.13 Relationship with Calculator

The Calculator computes the current state; History is **not** maintained by
the Calculator. Once a workflow decides a timestep is complete, it calls
`state.update_history()` to store the snapshot. The data flow stays:

```
Calculator
    ↓
PyRAIState.current state
    ↓
Workflow / Algorithm
    ↓
PyRAIState.update_history()
    ↓
PyRAIState.history
```

The Calculator never needs to know the history requirements of NAMD or
Adaptive Sampling.

## 18.14 Serialization

History is official runtime data of PyRAIState, so it must be serialized
together with it:

```
PyRAIState
├── pmd
├── theory
├── reference
├── execution_graph
└── history
```

After saving and restoring, `restored_state.get_history()` must return the
same historical data as before saving. This is especially important for NAMD
restarts, because algorithms like FSSH may depend on the previous few
timesteps.

## 18.15 Forbidden Access Patterns

None of the following belongs to the official API:

```python
state._history
state.__dict__["history"]
state.history.append(...)
state.history.pop(...)
```

Workflows, Calculators and algorithms may only access history through:

```python
state.update_history()
state.get_history()
state.get_history(n)
state.get_history_step(index)
state.get_history_step(index, copy=False)   # read-only view
state.history_size
state.has_history
```

With this rule, even if the underlying implementation changes from `list` to
`deque` or a ring buffer, workflows and algorithms are unaffected.

## 18.16 Recommended Minimal API

The first version does not need a complex History class; PyRAIState itself
provides the minimal interface:

```python
class PyRAIState:

    def update_history(self):
        ...

    def get_history(self, n=None):
        ...

    def get_history_step(self, index=-1):
        ...

    @property
    def history_size(self):
        ...

    @property
    def has_history(self):
        ...
```

Internally:

```python
self._history = []
self._history_size = 4
```

Only if the history logic grows complex later should it be split into
`pyrai3md/data/history.py`. At this stage, do not introduce a separate
HistoryManager, HistoryRecord or RingBuffer abstraction in advance.

## 18.17 Summary of Design Principles

1. History belongs to PyRAIState, not to PyRAIConfig.
2. History stores snapshots of `pmd.state`, not complete PyRAIState objects.
3. History does not understand NAMD, FSSH or Adaptive Sampling.
4. Workflows operate history only through the official PyRAIState API.
5. The history length is input-controlled and uses a bounded FIFO.
6. Missing historical data is handled by the concrete algorithm, never
   fabricated by History.
7. History must be serialized and restored together with PyRAIState.
8. The first version stays simple; no separate History Framework in advance.

The final picture:

```
                  ┌─────────────────────┐
                  │     PyRAIState      │
                  │                     │
                  │  current state      │
                  │       │             │
                  │       ↓             │
                  │    history          │
                  │  [t-4,t-3,t-2,t-1] │
                  └─────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
           FSSH          Verlet       Adaptive
```

Core idea: PyRAIState is not only the sole data carrier passing the
"current state" between workflows, but also the sole carrier of the bounded
runtime history. Workflows interpret history; PyRAIState manages history.
