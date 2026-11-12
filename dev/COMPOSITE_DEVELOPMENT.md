# PyRAI3MD Composite Development Guide

Last updated: 2026-11-09 14:20 (the planning/workflow dependency note now records the one registered edge: three task-descriptor selectors from planning.task. The rest was not re-audited. Earlier: 2026-11-04 11:10 (#451: 4.1 resolves a nested `highlevel` under `qmmm` to `multi_center_qmmm` and no longer lists QMMM as a multi-center conflict; 4.3 gained the QMMM queue's per-center shape, what `assemble_qmmm` keeps single, and the `active_configuration` echo the MD loop stages from. Sections 5 onward were not re-audited this round. Earlier: 2026-10-30 22:47 (#444: the ONIOM formulas moved to
methods/ONIOM_DEVELOPMENT.md and the QMMM paragraph to
methods/QMMM_DEVELOPMENT.md; this document keeps the generic multiscale
design, the framework cooperation and the `&composite` breakdown, and both
moved sections are now pointers that keep the old section numbers). Before
it, 2026-10-30 17:05 (#443: the QMMM paragraph, re-read after the reduction
moved into the LAMMPS skill). Before that, 2026-10-30 12:20 (the QMMM queue
paragraph only: `System.qm_contains_field_gradient` was added and `assemble_qmmm` re-read against it — the switch removes `F_el` from the classical Q-row correction when the QM provider already contains the 0-order field term, and keeps it otherwise; pinned by `test_qmmm_composite.py`). Before it, 2026-10-22 22:20 (re-read against the code: multi-center detection reads `system.center_groups`, the queue builder is `_expand_oniom_centers`, Rule 7 now separates the per-center keys (`nac`/`soc`/`dipole`/`transition_dipole`) from the refused ones (`overlap`/`hessian`), and periodic region cutting is no longer listed as unimplemented (#317). The rest was not re-audited. Before it, 2026-10-04 16:20)

---

## 1. Design Objective

This document describes the multi-center ONIOM composite system. The composite
system extends the existing single-center ONIOM2/ONIOM3 and QM/MM schemes to
support N non-overlapping centers, with strict gradient scatter-add, per-task
configuration isolation, and frozen geometry snapshots.

The overall architecture is defined in [ARCHITECTURE.md](ARCHITECTURE.md); the
multi-center unified base design is in
[MULTICENTER_DEVELOPMENT.md](MULTICENTER_DEVELOPMENT.md) (this document is the
composite-module detail under it); the
workflow layer context is in [WORKFLOW_DEVELOPMENT.md](WORKFLOW_DEVELOPMENT.md);
the Calculator ownership model is in
[CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md); the planning layer is in
[PLANNING_DEVELOPMENT.md](PLANNING_DEVELOPMENT.md).

The composite system follows four fundamental principles:

1. **Composite is a pure mathematical layer.** It never imports Calculator,
   state, or backend. It receives outcomes and returns assembled results.
2. **Planning terminates before execution.** Multi-center expansion happens in
   `planning/task.py` during `build_state()`, not at each MD step.
3. **Each task owns its configuration.** Deep copy isolation prevents
   cross-task contamination; region slicing keeps local data consistent.
4. **Default serial, conditional thread.** Thread execution is declined by
   the skill, not granted by the framework: a backend may refuse the
   contract it is handed.

### 1.1 Scope

First version covers:

- Non-overlapping centers; two-layer (1+2N) and three-layer (1+4N) ONIOM.
- Per-center multi-state energy and gradient (each center keeps its own
  state axis; R4 archives them per center, never a fabricated global
  sum(S_c) axis).
- Serial execution by default; thread execution when every backend allows it.
- Full backward compatibility with existing single-center inputs and `.rt`
  files.

Not included in the first version:

- Cross-center NAC/SOC and multi-center excited-state combinations (each
  center's OWN NAC/SOC is normalized and archived per center since R4; only
  undefined cross-center coupling stays out).
- Overlapping centers, dynamic partitioning, explicit charge embedding.
- General QM/MM, link atoms, boundary Jacobian (Phase 2).
- Periodic boundary conditions (separate acceptance gate).

---

## 2. Module Architecture

The composite system spans four modules, each with a single responsibility:

```
schema/config.py       Center definitions (highlevel/midlevel) + execution config
schema/adapter.py      Semicolon group parsing (_parse_center_indices)
        │
        ▼
planning/task.py       Multi-center detection, validation, expansion (1+2N/1+4N)
        │
        ▼
data/core.py           Task config construction, plan_version stamping
data/runtime/          plan_version compatibility, .rt restart defaults
  serializer.py
        │
        ▼
workflow/calculator.py Geometry freeze, local materialization, execution, assembly
workflow/multiscale/regions.py    Pure geometric operations (validate, slice, scatter-add)
workflow/multiscale/composite.py  Strict aggregation (validate, normalize, assemble)
workflow/dispatch/terms.py  Serial/thread execution with the skill thread check
```

Dependency direction is one-directional:

```
regions.py  ←  composite.py  ←  calculator.py
                   ↑                  ↑
             (never imports       (imports both)
              calculator)
```

`planning/task.py` never imports `workflow/`. `workflow/` imports three pure
task-descriptor selectors from `planning/task.py` (`resolve_primary_task`,
`resolve_system_task`, `region_label`); this is the one registered edge.
Communication otherwise happens through `state.theory/reference` task queues.
See [WORKFLOW_DEVELOPMENT.md](WORKFLOW_DEVELOPMENT.md) Section 13.

---

## 3. Configuration and Input

### 3.1 Center Definition

Centers are defined through `highlevel` and `midlevel` in the System
configuration. No separate `centers` keyword is introduced. ASCII semicolons
(`;`) separate center groups:

```text
highlevel 1-3 5; 11-14
midlevel  1-8; 11-20
```

Two-layer uses only `highlevel`; three-layer requires matching group counts
with H_c and I_c disjoint for each center. The framework derives
`region_h = highlevel`, `region_m = highlevel ∪ midlevel` (the mid-level
model region) and `region_l = the whole system`; the mid-level task runs
over `region_m`, the high-level task over `region_h`, and the low-level
baseline over `region_l`.

The `System` model accepts `Union[List[int], List[List[int]]]` for
`highlevel` and `midlevel`, supporting both flat lists (single center,
backward compatible) and nested lists (multi-center).

### 3.2 Adapter Parsing

`schema/adapter.py` provides `_parse_center_indices` for semicolon group
parsing. It handles three input formats:

- Semicolon string: `'1-3 5; 11-14'` returns `[[0,1,2,4], [10,11,12,13]]`
  (0-based).
- Flat list: `[1,2,3]` returns `[[0,1,2]]` (single center).
- Nested list: `[[1,2],[3,4]]` returns `[[0,1],[2,3]]`.

Empty groups (consecutive or leading/trailing semicolons) raise `ValueError`.
Input indices are 1-based; the function subtracts the offset. Group-internal
sorting and deduplication follow existing `ReadIndex('s')` semantics.

The `highlevel` and `midlevel` fields in `MOLECULE_TYPES` remain `'si'` for
type declaration, but `_convert_types` routes them through
`_parse_center_indices` instead of `ReadIndex('s')`.

### 3.3 Execution Configuration

A new `Composite` submodel in `schema/config.py` controls execution:

```python
class Composite(BaseModel):
    executor: str = 'serial'       # 'serial' or 'thread'
    max_workers: int = Field(default=1, ge=1)
```

This submodel is registered in `FIELD_TO_SUBMODEL`, `SUBMODEL_CLASSES`, and
`NESTED_FRAMEWORK_SECTIONS` under the key `'composite'`. The top-level
`PyRAIConfig` carries `composite: Composite = Field(default_factory=Composite)`.

The `composite` section controls only execution strategy; it does not redefine
centers or force a scheme keyword.

---

## 4. Planning Layer

### 4.1 Composite Type Resolution

`resolve_composite_type` in `planning/task.py` detects multi-center ONIOM
before the existing groups/qmmm/reference branches:

```
multi-center highlevel/midlevel    -> multi_center_oniom{2,3}
system.center_groups non-empty      -> fragment
2 theory + multiscale_type='qmmm'   -> qmmm, flat highlevel
                                   -> multi_center_qmmm, nested highlevel
                                   (field -> [qm_charge] -> QM -> energy; field and
                                    qm_charge contribute 0, QM and energy +1)
two theory skills                   -> oniom2
three theory skills                 -> oniom3
otherwise                           -> None (single-shot)
```

A two-method queue is selected explicitly by `control.multiscale_type`:
`'qmmm'` is the additive QM/MM scheme, while an empty value or `'oniom'`
keeps the historical ONIOM2 meaning. `system.qmmm_key` no longer selects
QMMM (it is refused with the new spelling), and a theory+reference pair no
longer implies QMMM — the reference queue is not executed by the calculator.

QMMM is the other additive scheme. Its two-flow energy decomposition, the
field/charge key surface, the `qm_contains_field_gradient` ownership switch
and the classical reduction that now lives in the LAMMPS skill are documented
in [methods/QMMM_DEVELOPMENT.md](methods/QMMM_DEVELOPMENT.md); this document
only covers the generic machinery both schemes share.

Multi-center detection conflicts with `center_groups` and raises `ValueError`
(fragments are a different partition model); QMMM no longer conflicts - a
nested `highlevel` there selects `multi_center_qmmm`. The number of theory
skills determines layer count (2 or 3); center count comes from the highlevel
grouping, not from the skill count. Because three call sites count centers
(`_count_centers`, `count_geometry_centers` and the renderer's own test), the
flat-versus-nested rule is one function shared by them, and the QMMM
single-center identifiers stay under the flat spelling so an existing run's
directory names do not move.

### 4.2 Multi-Center Validation

`validate_multi_center_input` checks:

- No empty center groups.
- All indices in range `[0, natom)`.
- No duplicate indices within a group.
- No cross-center overlap: H_i intersection H_j = empty.
- Three-layer: matching group counts, I_i intersection I_j = empty,
  H_c intersection I_c = empty (the two input regions are disjoint; the
  framework builds region_m = H_c union I_c for the mid-level task).

### 4.3 Task Expansion

`_expand_oniom_centers` generates the task queue:

Two-layer N-center (1+2N terms):

```
L(real)                    contribution +1
Per center c:
    +H(M_c)                contribution +1, region_indices = highlevel[c]
    -L(M_c)                contribution -1, region_indices = highlevel[c]
```

Three-layer N-center (1+4N terms):

```
L(real)                    contribution +1
Per center c:
    +M(I_c)                contribution +1, region_indices = midlevel[c]
    -L(I_c)                contribution -1, region_indices = midlevel[c]
    +H(M_c)                contribution +1, region_indices = highlevel[c]
    -M(M_c)                contribution -1, region_indices = highlevel[c]
```

Each task carries `center_index` (contiguous 0-based integer, renamed from
the former `center_id` in R2) and `region_id` (string identifier). The
whole-system baseline task L(real) has `center_index = None`; a plain
single-center high-level task is explicitly bound to 0.

The QMMM queue is the other multi-center shape and it does not use a whole-system
baseline: one probe per center (its `probe_indices` narrowed to that center while
`partition_indices` stays the union), an optional energy-only charge round per
center, one property solve per center, and exactly one classical task over the
union with `qm_centers` tiling the QM set. Its assembly is `assemble_qmmm`, which
keeps the per-center energies and gradient rows while the classical surface, the
two removed gradients and the identity guard stay single; the multi-center charge
duty is ignored there, and unlike the generic path its `computed_charges` are
never dropped by `IGNORED_KEYS`. `active_configuration` is echoed back from the
request so the MD loop can stage the surface the next step reads.

### 4.4 Plan Version

Every task receives `plan_version` at queue construction:

- `1`: multi-center aware composite (current).
- `0`: single-task or legacy non-composite queue.

This version travels through serialization and enables backward-compatible
restart.

---

## 5. Execution Pipeline

### 5.1 Geometry Freeze

`Calculator.run_in_process` freezes the geometry snapshot at the start of
each step:

```python
snapshot_atoms = list(self._state.atoms)
snapshot_coords = np.asarray(self._state.coordinates, dtype=np.float64)
```

All tasks derive their local geometry from this snapshot through
`materialize_region`. No task reads live state geometry during execution.
This prevents mid-step geometry changes from corrupting the composite
sum.

### 5.2 Local Materialization

For each task, `materialize_region(region_indices, atoms, coordinates)`
slices the frozen snapshot into local atoms and coordinates. Empty
`region_indices` or `region='all'` passes the full system. The returned
arrays are fresh copies, not views.

### 5.3 Result Invalidation

Before each `backend.calculate()` call, `invalidate_task_result(task)`
clears stale results in the task config. This ensures the new call is the
sole data source; no fallback to historical non-empty results.

### 5.4 fin_coord Rejection

Multi-task composite rejects `fin_coord` in backend results. Sub-calculations
must not update the global geometry; only the framework optimizer/MD updates
coordinates. Single-task execution preserves the existing `fin_coord`
channel for skill-internal iteration.

### 5.5 Assembly

After all tasks complete, `assemble_results(tasks, outcomes, request)`
assembles the composite result:

1. Check all outcomes succeeded.
2. For each outcome, normalize the raw result (shape/state/finite checks).
3. Scatter the local gradient to the full system via `project_gradient`.
4. Multiply by the contribution coefficient and accumulate.
5. Return the combined dict with `energy` (nstate,) and `gradient`
   (nstate, natom, 3).

The primary task provides non-aggregated keys (status, metadata). Single-state
energy from a backend is broadcast to match the global `nstate` when needed.

---

## 6. Mathematical Contract

### 6.1 Two-Layer ONIOM

The queue is `L(real) + E_H(M_c) - E_L(M_c)` per center (coefficients
`+1, +1, -1`); the gradient scatters each term's local gradient to its
`region_indices` before multiplying by the coefficient. Derivation and the
region semantics: [methods/ONIOM_DEVELOPMENT.md](methods/ONIOM_DEVELOPMENT.md)
section 2.1.

### 6.2 Three-Layer ONIOM

Five terms per center (`+1, +1, -1, +1, -1`), same scatter-add form. See
[methods/ONIOM_DEVELOPMENT.md](methods/ONIOM_DEVELOPMENT.md) section 2.2.

### 6.3 Shape Contract

The global state-resolved axis below is the C=1 (single-center) view. Since
R4 a C>1 run builds NO global `energy[nstate]` axis: each center keeps its
own `(S_c,)` / `(S_c, A_c, 3)` arrays in `state.electronic[c]`, and the total
result carries the assembled `active_energy` (scalar) and `active_gradient`
`(natom, 3)` at the active configuration.

- Global energy: `(nstate,)`.
- Global gradient: `(nstate, natom, 3)`.
- Local gradient atom axis must equal the transform output atom count.
- Single-state scalar energy is promoted to `(1,)` then broadcast to
  `(nstate,)` when the global state requires it.
- Shape mismatches raise `ValueError`; numpy broadcasting is never used to
  guess state count or atom axis.

### 6.4 Non-Aggregate and Per-Center Properties

NAC, SOC, overlap, dipole, transition_dipole, computed_charges, and Hessian
have no defined ONIOM sum rule, so they never enter the cross-center linear
energy/gradient aggregation. R4 split the former single `_REFUSED_KEYS` set
into two classes:

- Cross-center linear aggregation is still refused (raising `ValueError` if
  such a key reaches the ONIOM sum).
- Per-center collection is allowed: `collect_center_electronic` archives each
  center's own NAC/SOC/dipole as-is into `state.electronic[c]` (the center's
  LOCAL framing), never inherited from the primary task and never broadcast
  into a global state axis. C>1 builds no `energy[nstate]` common axis; the
  total result keeps `active_configuration`/`active_energy`/`active_gradient`.

---

## 7. Gradient Scatter-Add

`workflow/multiscale/regions.py` provides the pure geometric operations:

- `validate_region(indices, natom)`: range and uniqueness check.
- `materialize_region(indices, atoms, coordinates)`: local slice from global.
- `project_gradient(local_gradient, indices, natom)`: scatter-add local
  gradient to full system.
- `slice_field(field_value, indices)`: per-atom field slicing for sync.

The scatter-add algorithm:

```
global_grad = zeros((nstate, natom, 3))
for j, i in enumerate(region_indices):
    global_grad[:, i, :] += local_grad[:, j, :]
```

Each local atom j maps to exactly one global atom i. Multiple caps targeting
the same global atom accumulate (future Phase 2); the first version has no
caps, so each global atom receives at most one contribution per task.

---

## 8. Task Executor

`workflow/dispatch/terms.py` provides serial and thread execution:

### 8.1 The Skill Decides Thread Safety

The framework keeps no capability table. Before a thread pool starts it
hands each term's backend the contract it is about to run under, and the
backend answers (`workflow/dispatch/terms.py::check_thread_safe`):

```python
    def check_thread_safe(self, runner, execution_mode, profile_semantics):
        """Raise to decline a shared thread pool; return to allow it."""
```

A backend that does not define the method runs concurrently; one that
raises names its own reason and the evaluation fails there, never
silently serialized. #301 replaced the framework-side registry that no
skill could see or fill; see `dev/SKILL_APP_DEVELOPMENT.md` section 5.6
for what makes a refusal necessary (process-wide cwd, environment,
stdout, shared artifact names).

### 8.2 TaskExecutor

```python
class TaskExecutor:
    def __init__(self, mode='serial', max_workers=1): ...
    def execute(self, work_items) -> list of dict: ...
```

- `serial`: sequential `_run_one(item)` calls.
- `thread`: `ThreadPoolExecutor` that asks each backend
  (`check_thread_safe()`) before submission. Each work item carries an
  independent backend instance and config.

### 8.3 Failure Handling

`execute_with_failure_handling` raises `RuntimeError` if any required task
fails, listing all failed task IDs and error messages. Thread scheduling
stops new tasks on failure; unstarted futures are cancelled.

---

## 9. Serialization and Restart

### 9.1 plan_version Compatibility

`data/runtime/serializer.py` handles backward compatibility:

- `_prepare_task(task)`: ensures `center_index`, `region_id`, `plan_version`
  have defaults before saving. New tasks get `plan_version=1`. Old `.rt`
  files carrying the former `center_id` key are migrated to `center_index`.
- `_restore_task(task)`: fills defaults for old `.rt` files. Restored tasks
  get `plan_version=0` (legacy).
- `StateSerializer.load()`: after loading, every task in the queue receives
  compatible defaults if the fields are absent.

Unknown `plan_version` values raise `ValueError` at the load boundary.

### 9.2 Submodel Flatting

The `Composite` submodel is added to `_SUBMODEL_NAMES` in the serializer,
so its fields (`executor`, `max_workers`) are flattened into the serialized
config alongside existing submodels (control, system, dynamics, adaptive).

### 9.3 Restart Identity

Save/load round-trip produces the same task descriptors, region indices,
center indices, and directory identities. Backend instances are excluded;
`Calculator.initialize()` reconstructs them.

---

## 10. Calculator Integration

### 10.1 Multi-Center Detection

Since R2 `run_in_process` detects multi-center by counting distinct
`center_index` values (not by a non-null `center_id`, which mis-fired once
the queue was reordered real-first):

```python
distinct_centers = {t.get('center_index') for t in tasks
                    if t.get('center_index') is not None}
is_multi_center = len(distinct_centers) > 1
```

This is distinct from multi-task (which includes single-center ONIOM). For
C>1 the calculator stages each center's electronic arrays and the assembled
active surface, then `state.commit_step()` writes them atomically (R4); C>1
builds no global `energy[nstate]` axis (the assembly request carries nstate 1
and the active configuration drives per-center selection).

### 10.2 Sync with Region Slicing

`Calculator.sync()` pushes `SYNC_SYSTEM_FIELDS` to every task config.
Per-atom fields (`atoms`, `mass`, `coordinates`, `velocities`) are sliced
by the task `region_indices` when the region is not `'all'`. Scalar fields
(`charges`, `multiplicity`, `nstate`, `current_state`) are synced as-is for
a single-center run. The `region_charge` is never overwritten by the global
sync.

Since R6, a C>1 run skips the electronic-state scalars (`nstate`, `ci`,
`spin`, `coupling`, `state_label`, `state_spin`, `current_state` -- the
`_ELECTRONIC_STATE_FIELDS` set): the global `pmd.system` carries `nstate=1`
and no per-center state axis, so syncing them would overwrite the
planning-derived per-center task configs. C>1 syncs geometry/charge only; the
per-step active state travels through `target_states` (the `calculate()`
parameter channel, routed per center by `_task_target_states`).

### 10.3 Aggregation Path

Multi-task composite uses `assemble_results` (strict scatter-add path).
Single-task uses `aggregate_results` (legacy contribution sum). The
Calculator selects the path based on `len(tasks) > 1`.

---

## 11. Validation and Error Handling

### 11.1 Plan Validation

`validate_plan(tasks, natom)` checks before execution:

- Unique `task_id` across the plan.
- Non-zero integer `contribution` coefficients.
- Region indices in range with no duplicates (via `validate_region`).

### 11.2 Result Validation

`normalize_task_result(raw, task, request)` checks per task:

- Energy present and finite (scalar or `(nstate,)` array).
- Gradient, when required, has correct atom count.
- No NaN/Inf in numerical arrays.
- Refused keys absent in multi-center mode.

### 11.3 Assembly Validation

`assemble_results` checks:

- All outcomes succeeded.
- Energy state count matches global `nstate`.
- Gradient shape is consistent.
- Failed tasks raise `ValueError` with task ID and error text.

### 11.4 Input Validation Summary

| Check | Location | Error |
|-------|----------|-------|
| Empty center group | `_parse_center_indices` | ValueError |
| Index out of range | `validate_multi_center_input` | ValueError |
| Cross-center overlap | `validate_multi_center_input` | ValueError |
| H/M group count mismatch | `validate_multi_center_input` | ValueError |
| H intersects I (3-layer) | `validate_multi_center_input` | ValueError |
| Multi-center + groups/qmmm | `resolve_composite_type` | ValueError |
| Missing energy | `normalize_task_result` | ValueError |
| Gradient shape mismatch | `normalize_task_result` | ValueError |
| Refused key in multi-center | `normalize_task_result` | ValueError |
| fin_coord in multi-task | `run_in_process` | ValueError |
| Backend declines a thread pool | `TaskExecutor._execute_thread` | the skill's own exception |

---

## 12. Design Rules

### Rule 1: Composite is a pure mathematical layer

`composite.py` and `regions.py` never import Calculator, state, or backend.
They receive outcomes and return assembled results. The Calculator calls
them; they never call the Calculator.

### Rule 2: Planning terminates before execution

Multi-center expansion happens in `planning/task.py` during `build_state()`.
The MD step only materializes real-time geometry from the frozen snapshot;
it does not re-plan.

### Rule 3: Each task owns its configuration

`Calculator.initialize()` deep-copies each task config. Sub-calculations
cannot contaminate each other through shared references. The state task
keeps the canonical reference; the backend works on its own copy.

### Rule 4: Geometry is frozen per step

All tasks in one composite evaluation derive from the same geometry
snapshot. No sub-calculation updates the global geometry. The framework
optimizer/MD is the sole geometry updater.

### Rule 5: Default serial, conditional thread

The default executor is serial with `max_workers=1`. Thread execution
hands the `(backend_name, runner, execution_mode, profile_semantics)`
contract to each backend, which allows it or refuses it in its own words.
A refusal stops the evaluation; it is never silently serialized (#301
retired the framework-side registry that no skill could fill).

### Rule 6: Old inputs remain valid

Flat list `highlevel`/`midlevel` inputs are normalized to single-center
nested lists. Old `.rt` files load with `plan_version=0` and compatible
defaults. No historical input format is broken.

### Rule 7: Non-linear properties are refused, not inherited

None of these properties has an ONIOM sum rule, and the composite splits them by
whether a center can own one. `nac`, `soc`, `dipole` and `transition_dipole` are
**per-center** keys (`PER_CENTER_KEYS`): each center's high-level provider is the
sole source, the value is archived on that center's own electronic record and never
enters the energy/gradient sum. Only `overlap` and `hessian` - no aggregation rule and
no per-center slot - are refused outright (`REFUSED_KEYS`). The primary task's
non-aggregated keys are passed through for metadata only.

### Rule 8: Import isolation

`planning/` and `workflow/` have zero imports between them. Communication
happens through `state.theory/reference` task queues and `state.jobtype`.
See [WORKFLOW_DEVELOPMENT.md](WORKFLOW_DEVELOPMENT.md) Section 13.

---

## 13. Phase 2 Roadmap

The following capabilities are designed but not yet implemented:

- **Boundary atoms and link atoms**: `BoundarySpec` with inside/outside
  atom indices, cap elements, fixed-distance or fixed-ratio Jacobian.
  `regions.py` will gain `RegionTransform` for Jacobian back-propagation.
- **Overlapping centers**: requires charge embedding and overlap correction.
- **Multi-center excited states**: needs local-to-global state mapping and
  a Hamiltonian model; not a simple broadcast.
- ~~Periodic boundary conditions~~ **implemented** (2026-10-09, #317): the region
  cut uses the continuous coordinates and the crossing ledger, C>1 is certified on a
  LAMMPS low layer, and the frame guard measures every recorded box
  (`records/diagnostics.py::_check_region_reconstruction`; see
  `dev/MULTICENTER_DEVELOPMENT.md` §11.3 and `dev/methods/PBC_DEVELOPMENT.md` §4.2).
- ~~Process-based parallelism~~ **implemented** as the term worker:
  `workflow/dispatch/term_worker.py` runs one evaluation's term group in a child process
  through the standalone `.rtt` payload (`TERM_WORKER_SUFFIX`, `TERM_PAYLOAD_VERSION='1.1'`)
  and returns `term_outcome.json`; the recursion ban holds - a worker graph never carries
  `remotetask=True` (`dev/MULTICENTER_DEVELOPMENT.md` §7.4).

---

## 14. Cross-References

- [ARCHITECTURE.md](ARCHITECTURE.md): overall module structure and
  dependency rules.
- [WORKFLOW_DEVELOPMENT.md](WORKFLOW_DEVELOPMENT.md): Calculator lifecycle,
  Dispatcher subsystem, and import isolation.
- [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md): Calculator
  ownership model, sync mechanism, and result write-back.
- [PLANNING_DEVELOPMENT.md](PLANNING_DEVELOPMENT.md): capability resolution
  and task queue expansion.
- [DATA_CORE_DEVELOPMENT.md](DATA_CORE_DEVELOPMENT.md): PyRAIState task
  queue and serialization.

---

## 15. ONIOM Charge Management and Electrostatic Embedding

Moved to [methods/ONIOM_DEVELOPMENT.md](methods/ONIOM_DEVELOPMENT.md)
section 3: charge routing rules, the ChargeStore and its managed region, and
the second meaning of `read_charge` without a partition. QMMM does not use
charge embedding at all; its field channel is described in
[methods/QMMM_DEVELOPMENT.md](methods/QMMM_DEVELOPMENT.md) section 3.

---

## 16. Composite Energy Breakdown (&composite)

Beyond the aggregated total, a composite run records the per-term energy
 breakdown so the log can show how each layer and center contributes.

### 16.1 Collection

Both assembly paths (`assemble_results` for C=1 and
 `_assemble_multi_center` for C>1) collect, per term, the RAW per-state
 energy array the backend computed (never multiplied by the
 contribution) plus the contribution coefficient. `build_composite_terms`
 orders the terms high to mid to low and, within a region, by center and
 queue slot, then stores them as parallel lists:

```
composite_terms = {
    'labels':        ['high[qm]', 'high[qm2]', 'mid[qm2]', ...],
    'energies':      [[E_s1, E_s2, ...], [E_ref], ...],
    'contributions': [1, -1, 1, ...],
    'state_indices': [1, None, 2, ...],
}
composite_total = <per-state ONIOM total (C=1) | active U (C>1)>
```

The parallel-list shape (not a list of tuples) is deliberate: the
 `Result` container converts any list/tuple value with `np.asarray`,
 which would mangle a ragged `(label, energies, contribution)` tuple.

The energy rows stay RAW because the broadcast happens on the way to the
 total, not on the way in: `assemble_results` repeats a single-state row
 over the state axis only when it sums (`composite.py`), so a reader that
 wants one number per term has to be told which state that number belongs
 to. `state_indices` is that statement: an integer local state for a term
 of a multi-center evaluation (`_task_state_index`, docs 8), or `None`
 where the term owns no state and the state the run is driven on decides.
 It is produced inside `build_composite_terms` after the high→mid→low
 ordering, never handed in from the queue, so an index cannot arrive beside
 another term's row. The term columns of `opt_status.txt` read with
 `_select_state_energy`, which is the same rule the total applies; an
 out-of-range state is a named error there rather than a silent first row.

### 16.2 Labels

`term_label(task, multi_center)` yields `<region>[<role>]`, where region
 is the geometric layer name from the shared `region_label` helper
 (model to high, inter to mid, real to low) and role is the theory
 method. A multi-center run appends the 1-based center ordinal
 `[<center+1>]`; the center-less baseline L(real) and a single-center run
 omit it. The same `region_label` taxonomy names the static_id
 directories (WORKFLOW_DEVELOPMENT.md section 18), so the log and the
 working directories speak one language.

### 16.3 Rendering

`utilities/print_results.py::_format_composite` renders the
 `&composite energy decomposition in Hartree` section right after
 `&electronic energy`. Each term occupies one line listing that term's
 per-state energies in order (a multi-state provider prints every state
 on its line); a final `total` line lists the per-state ONIOM total. The
 `&electronic energy` section keeps its one-state-per-line (vertical)
 form. A single-task run carries no `composite_terms`, so the section is
 omitted entirely (never a 'Not computed' placeholder).
