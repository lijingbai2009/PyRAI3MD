# PyRAI3MD Multi-Center Unified Computation Foundation Design

Last updated: 2026-11-05 11:40 (#460 closes three former later-extension entries in section 14 - charged QM region, multi-center multi-state models and MD checkpoint/resume - each now has shipped carriers and readings; the per-center state spelling (`state_i: [0, 0]`, `root: "2; 2"`) is what the input layer gained. Sections 1-13 were not re-audited this round. Earlier: 2026-11-04 15:20 (#452 puts the five QMMM cases in three charge tiers, and #451 is restated with the re-measured numbers: section 14's later-extension list no longer carries general QM/MM, section 15's opening names the additive route beside the ONIOM one, and 15.6 states the four QM/MM invariants (one snapshot per round, cold start, union array computes while ids address, no fallback inside a charge round) with the measured 1.23e-4 Hartree that one fitted round buys on that box and the 8.7e-7 the fixed point adds on top (re-measured under #452). Sections 1-13 were not re-audited this round. Earlier: 2026-11-02 10:40 (#448: 4.4 gained item 7: a skill's own ordering must be identity on the declaration the framework forwarded, or it is refused - two sorts of one list are two index spaces. Items 1-6 and the rest of the file were not re-audited. Before it, 2026-11-01 11:00 (section 7.2 records that the work_item's `nac_pairs` field now has a consumer; the rest was not re-read). Before it, 2026-10-22 22:20 (section 7.2 no longer claims a framework `THREAD_AUDIT_REGISTRY`: thread safety is the backend's own `check_thread_safe(runner, execution_mode, profile_semantics)` and an undeclared backend runs concurrently; `TERM_PAYLOAD_VERSION` is '1.1'. The rest was not re-audited). Before it, 2026-10-14 11:40 (only section 11.3 was recalibrated against the
working tree since 2026-10-09: a worked example of naming a center by its template
row numbers was added on a box built outside this package. The rest of the document
is at its 2026-10-09 calibration and is not claimed to track the tree since))

This document is the single authoritative design reference for the PyRAI3MD
multi-center unified computation foundation, written against the current code's
design and state. Module-level implementation details live in the per-module
development documents: data core `DATA_CORE_DEVELOPMENT.md`, the composite math
layer `COMPOSITE_DEVELOPMENT.md`, the calculator
`CALCULATOR_DEVELOPMENT.md`, planning `PLANNING_DEVELOPMENT.md`, workflow and MD
`WORKFLOW_DEVELOPMENT.md` and `methods/MD_DEVELOPMENT.md`, surface hopping
`methods/FSSH_DEVELOPMENT.md`/`GSH_DEVELOPMENT.md`/`LZSH_DEVELOPMENT.md`, input
`INPUT_SYSTEM_DEVELOPMENT.md`, logging `LOG_DEVELOPMENT.md`, acceptance
`QUALIFICATION_DEVELOPMENT.md`, overall architecture `ARCHITECTURE.md`.

---

## 1. Core decisions and invariants

The center is the unified partitioning axis of the underlying computation data:
every calculation holds a center plan of length C, and an ordinary single-center
run is simply C=1. The backend and the surface hopping consume the same center
identity, region mapping and electronic definition, and each compiles its
execution rules once, at the right point in its life cycle.

- backend: generates the complete task queue and execution strategy during the
  planning phase of build_state; at run time it only assembles the local input of
  this step and executes it.
- surface hopping: binds the per-center algorithm instance, the electronic
  property read positions, the history time window and the write-back positions
  when each MD trajectory starts, at `surface_hopping` initialization; it does not
  re-derive the center structure at every step.
- `PyRAIState`: the only global state owner; it holds the global nuclear data, the
  per-center electronic data, the static plan and the recoverable history. A center
  owns neither an independent nuclear trajectory nor a nested PyRAIState.
- `Calculator`: the only backend life-cycle and result-collection entry; one
  independent configuration, backend instance and working directory per task.
- `Composite`: a pure mathematical composition layer that projects per task and
  builds the active surface; it does not execute a backend and does not propagate
  the electronic state.
- MD: one nuclear integrator, one electronic step call, one commit, one output and
  one checkpoint main loop; SingleMD/MultiMD are not two implementations.

Invariants that run through the whole design:

- Single-center, single-center multi-task and multi-center multi-task go through
  the same task materialization, executor, result routing and electronic binding
  interfaces. The difference between C=1 and C>1 is data length and physical
  strategy, not data source or code path.
- The four identities are strictly separated: trajectory_index / center_index
  (contiguous 0-based) / task_slot+task_id / local_state_index (inside
  [0, nstate[c])).
- External region / coupling / root numbers are 1-based; internal ones are
  0-based.
- There is exactly one copy of the global coordinates/velocities/masses; the atom
  mapping comes from the static plan; this step's electronic output and history
  values may not modify a committed state through a writable alias.
- A configuration error fails before any backend directory is created, any thread
  starts, or any Slurm submission happens.
- The numeric cores never write the parent PyRAIState directly and never call the
  global velocity update on their own; the only official write-back point is
  `commit_step`.

## 2. Architecture and module responsibilities

```text
schema/config.py · adapter.py · normalize.py · validate.py
    input dimension extension, per-center normalization, strict pre-launch checks
        |
planning/task.py
    unified ONIOM expansion (real-first), task identity, electronic binding,
    molecule flags and backend_execution_descriptor
        |
data/center.py · data/core.py
    canonical center plan (PartitionPlan/CenterSpec/ElectronicSpec),
    PyRAIState center data / active surface / history / commit interfaces
        |
workflow/calculator.py
    freeze geometry -> work_items -> execute -> validate -> route -> compose
    -> stage/commit
    |- workflow/dispatch/terms.py       serial or audited threads, unified work_item/outcome
    |- workflow/dispatch/term_worker.py remotetask remote term worker (.rtt)
    |- workflow/multiscale/regions.py   pure geometric projection (materialize/project/slice)
    |- workflow/multiscale/composite.py ONIOM math composition + per-center electronic collection
        |
workflow/integrator/electronic.py
    ElectronicStepper: one binding per center, running the untouched fssh/gsh/lzsh
    cores against a CenterView proxy; C=1 is a transparent binding
        |
workflow/md.py · verlet.py · output.py · trajectory.py
    nuclear integration reads the combined active gradient; center-protocol
    output and trajectory queries
```

`regions.py` does not depend on Calculator; `dispatch/terms.py` holds no main
state and only executes the injected work items; planning never imports the
workflow dispatcher in return; composite is a pure math layer that imports no
Calculator/state/backend.

## 3. Data axes, ownership and the unified interface

### 3.1 The four identities

| Identity | Example | Meaning |
| --- | --- | --- |
| trajectory_index | the 7th initial condition | an independent nuclear trajectory owning its own PyRAIState |
| center_index | c=0, 1 | an electronic center inside the same nuclear trajectory; contiguous 0-based, stable in declaration order |
| task_slot / task_id | L(real), H(c0), L(c0) | one backend term; not the same thing as a center |
| local_state_index | the i-th state of center c | inside [0, nstate[c]); never numbered across centers |

task_id is the persistent identity, task_slot is the fast position inside the
frozen queue, and persistent recovery checks both. The whole-system baseline task
has `center_index=None`; an ordinary single-center high-level task binds 0
explicitly. Whether a run is multi-center is decided by counting distinct
`center_index` values (never by testing whether some field is non-empty).

### 3.2 The PyRAIState center data structures

```text
PyRAIState
  pmd                  global input, nuclear structure, shared dynamics options
  partition_plan       version, center order, regions and electronic definitions (data/center.py)
    centers[c]         high_indices / mid_indices / electronic_spec
  theory / reference   static backend task queue
  electronic[c]        this step's properties per center: energy/gradient/nac/soc/density
  active_surface       active_configuration / active_energy / active_gradient / evaluation_id
  continuation         the committed R/V/s/F and next_step the next nuclear step needs
  history              synchronized global frame sequence, each frame carrying centers[c] and electronic_outcome[c]
```

`data/center.py` provides three frozen dataclasses: `ElectronicSpec`
(ci/spin/nstate/state_spin/coupling/nac_pair_map/soc_pair_map/initial_root),
`CenterSpec` (center_index/high_indices/mid_indices/electronic_spec/charge/
multiplicity) and `PartitionPlan` (version/plan_fingerprint/n_center/n_layer/
centers/supplied/system_charge/system_multiplicity plus the bounded `center(c)`),
built from the normalized configuration by `build_center_plan`, with
`CENTER_PLAN_VERSION='1.1'` (since 1.1 CenterSpec carries the per-center
`charge`/`multiplicity` and PartitionPlan carries the whole-system aggregates
`system_charge`/`system_multiplicity`); an inconsistent structure raises
`CenterPlanError`.

The static `electronic_spec` contains static definitions only; mutable
current_state/density are never written back into the static plan. Per center,
`energy` is `(S_c,)`, `gradient` is `(S_c, A_c, 3)`, NAC is `(P_nac,c, A_c, 3)`,
SOC follows the pair layout after backend adaptation, and density is a
serializable real/imaginary `(S_c, S_c, 2)`. The C centers may have different S_c,
A_c and pair counts; independent arrays are held in a list -- no padding, no
requirement of a regular high-dimensional ndarray, and never an invented
`sum(S_c)` global electronic-state axis.

### 3.3 The public protocol facing consumers

```python
state.center(c)                    # bounded center read interface (CenterSpec)
state.center_electronic(c)         # the mutable electronic carrier of that center
state.set_center_electronic(c, **arrays)
state.get_active_configuration()   # tuple; C=1 still returns (s0,)
state.get_active_energy()          # scalar
state.get_active_gradient()        # (natom, 3) committed combined active gradient
state.active_gradient()            # the combined active gradient the integrator reads (see section 10)
state.stage_evaluation(evaluation) # staging only; does not advance the committed nuclear trajectory
state.commit_step(outcome)         # the only official write-back point (atomic)
```

The global scalar interfaces `state.nstate` / `state.current_state` and their
kinetics project at C=1 and raise `CenterPlanError` at C>1 (asking the reader to
use `center(c).nstate` and `get_active_configuration()[c]`); they never return the
first center or a total state count. `stage_evaluation` stages one evaluation
(per-center electronic results + active surface + continuation) and `commit_step`
writes it back atomically in one go: a failure before the commit leaves the
committed state unchanged.

### 3.4 History frames

`update_history()` writes a structured `HistoryFrame`
(`HISTORY_FRAME_VERSION='1.0'`):

```text
HistoryFrame
  step / time / evaluation_id / plan_fingerprint
  nuclear_sample: R, V_pre, K_pre, active_config_pre, active_gradient
  centers[c]:     this frame's center definition (high_indices / electronic_spec / charge / multiplicity)
  electronic_outcome[c]: energy/gradient/nac/soc/density (LOCAL framing)
  continuation:   V_post, active_config_post, active_gradient_post
  warmup:         true while the continuation is not available yet
```

The distinction between the pre-hop sample and the post-hop continuation is
kept; all centers share the same step/time and are trimmed together; a window
that is not full has an explicit warmup. `get_history_step(index, copy=False)`
preserves the read-only reference contract (the FSSH hot loop depends on it).

## 4. Input format: the fields stay, only the center dimension is added

### 4.1 Field conventions

| Field | Single-center shape | Multi-center shape | Canonical internal form |
| --- | --- | --- | --- |
| highlevel/midlevel | atom list | list[center][atom] | list[center][atom], atoms 0-based |
| ci | list[spin block] | list[center][spin block] | an independent state list per center |
| spin | list[spin block] | list[center][spin block] | S values (currently integer S) |
| coupling | list[pair][2] | list[center][pair][2] | local state pairs per center, 0-based |
| root | int | list[center] | initial state per center, external 1-based / internal 0-based |
| charges | int | list[center] | region charge per center (may be negative), a per-center scalar |
| multiplicity | int | list[center] | region spin multiplicity per center (>=1), a per-center scalar |

No new user-facing `centers` block is added, and neither &molecule/&md nor the
YAML/JSON hierarchy is replaced: center_spec is an internal derived record. The
shared public keys (sfhp/dt/decoherence and their kin) stay shared first and are
not automatically given a per-center dimension. In `schema/config.py`,
`System.ci/spin/coupling` accept both the flat and the nested form
(`Union[List[int], List[List[int]]]`, coupling nested one level more),
`System.charges/multiplicity` accept `Union[int, List[int]]` (flat = C=1, nested
or `;` = a scalar per center), and `Dynamics.root` accepts `Union[int,
List[int]]`; the molecule namespace's `multitask/remotetask/multipool` are
three-valued (an absent None is distinguished from an explicit False/1).

### 4.2 The classic input and its equivalent YAML/JSON

```text
&molecule
charges 0; 0
multiplicity 1; 1
highlevel 1-3; 8-11
ci 2 2; 2 2
spin 0 1; 0 1
coupling 1 2, 3 4, 1 3; 1 2, 3 4, 2 4
multitask 1
remotetask 0
multipool 4
&md
root 1; 3
sfhp fssh
```

The ASCII `;` separates centers only; inside one center, coupling keeps using
commas to separate pairs: split the centers first, then hand each group to the
existing single-group parser, so the pair axis of coupling is never mistaken for
the center axis. An empty group produced by a leading/trailing/double semicolon is
an error; a center that needs no coupling uses `[]` in structured input and an
explicit `none` group in the classic input (e.g. `coupling none; 1 2`) -- an empty
semicolon may not carry two meanings, and a full-width semicolon U+FF1B gets a
message naming the replacement. YAML/JSON use the nested list of the same fields
and are equivalent to the example above.

### 4.3 Index base conventions

External raw atom/coupling arrays and text are uniformly 1-based; internal
config/state/restart are uniformly 0-based. `schema/adapter.py` separates the two
entry points with `_parse_center_indices(raw=...)`: raw=True subtracts 1 from a
typed list (a 1-based external index), raw=False passes an already normalized
0-based entry through without subtracting. `_parse_center_int_lists` (ci/spin),
`_parse_center_pairs` (coupling) and `_parse_center_scalars` handle the rest --
root subtracts one with start=1 so external 1-based and string/typed entries agree;
charges/multiplicity are semantic scalars and are not shifted, and charges may be
negative. Program entry points that historically use normalized numbers go through
the dedicated normalized API instead of the raw parser again; an old raw coupling
array goes through an explicit legacy import or a format-version migration, never
through guessing the base from the numeric range. root: external 1-based, internal
0-based.

### 4.4 Center count, defaults and broadcasting

1. The geometric center count C comes from highlevel
   (`count_geometry_centers`); highlevel is the **only source** of C. An ordinary
   calculation with no region definition materializes as one center covering real.
   Several ci/root groups without highlevel fail immediately (the atom partition is
   not guessed).
2. An explicit region field carrying the center axis (highlevel/midlevel) must have
   exactly C groups and is never broadcast: defining midlevel at all means a
   three-layer ONIOM, and its number of partitions must **equal** the highlevel
   center count C (one to one), with anything else reported as an error (not "less
   than or equal"). Two-layer runs do not define midlevel.
3. Legacy flat fields at C=1 are wrapped in one layer; for the
   **electronic/state fields** (ci/spin/coupling/root/charges/multiplicity) at C>1,
   anything not fully given -- one flat value, or k<C groups -- makes the missing
   centers **copy the values of center 0 directly**
   (`as_center_groups(..., broadcast=True)` and `as_center_scalars`); k>C is an
   error. This item **reverses** the older decision ("a flat value at C>1 is an
   error, no implicit broadcast"); the region fields highlevel/midlevel stay strict
   and are still never broadcast.
4. An absent field is materialized per center from its default with the default's
   origin recorded, and `missing` must be distinguished from a user's explicit
   `[]/0` (`PartitionPlan.supplied`).
5. A coupling is never invented. An absent or empty `coupling` leaves that center
   with **no** pairs, and a pair a center does not list is **shielded** — the
   framework does not ask the engine for it and the readers do not fill it with
   zeros. `nac_pair_map`/`soc_pair_map` are derived from `coupling` by
   `data/center.py::derive_pair_declaration`, the one implementation shared by the
   flat single-center path (`schema/normalize.py`) and the per-center plan; it is
   also where a nested input that names only the pair maps fills `coupling` back
   in, and where a disagreement between the two sides raises instead of preferring
   one. The rule this item used to state — an absent coupling auto-generates the
   adjacent pairs — is withdrawn (#447): it made every multistate run ask the
   engine for couplings nobody named, and it kept the "neither side declares
   anything" case unreachable.
6. A state_spin/nstate given explicitly by the user or the API must agree with
   ci/spin, or the conflict is reported; one source is never silently preferred.
7. **The declaration survives the skill untouched (#448).** `schema/adapter.py`
   forwards `coupling`, `nac_pair_map` and `soc_pair_map` into the skill section,
   so a skill receives the framework's own state numbering and pair list. A skill
   may order its states for its engine (Molcas and ORCA both want ascending 2S),
   but that ordering must be **identical** to the input it was given: when a
   declaration is present, the order is checked and a non-conforming ladder is
   refused by name rather than permuted, because two sorts of one list are two
   index spaces and the framework addresses the returned rows by the
   declaration's order. A skill running standalone - validating what an agent
   typed directly, with nothing declared - still sorts and still enumerates its
   own pairs, and that is the only case where it does. Pinned by
   `qualification/verification/test_state_declaration.py::TestSkillCannotPermuteADeclaredLadder`
   and `TestDeclaredPairsSurviveTheSkillEnumeration`.

#### 4.4.1 The region charge / spin multiplicity model

`charges`/`multiplicity` define the **electronic structure of each center's
region** (independent per center), and they are assigned as follows:

- Center c's high-level region task H(c) and its low-level correction L(c)
  (region = high_indices[c]) use `charge_c`/`mult_c` of center c (written into
  `CenterSpec.charge/multiplicity`, with `mult_c >= 1`).
- In three layers the mid region I_c **inherits one-to-one** the
  `charge_c`/`mult_c` of the corresponding center c: even where I_c
  geometrically contains atoms of other centers, only the currently **active**
  center c's charge/spin counts, and an **inactive center assigned into another
  center's environment contributes nothing** (consistent with the inheritance).
- The whole-system low-level baseline L(real) uses the aggregates:
  `system_charge = Σ_c charge_c`; `system_multiplicity` follows the
  **unpaired-electron cancellation** rule `total_unpaired = (Σ_c (mult_c − 1)) mod
  2`, `system_multiplicity = total_unpaired + 1` (`aggregate_system_multiplicity`;
  e.g. triplet+triplet → (2+2) mod 2 = 0 → singlet, doublet+doublet → singlet,
  triplet+doublet → doublet), written into
  `PartitionPlan.system_charge/system_multiplicity`.
- Environment atoms (belonging to no center) default to electrically neutral and
  contribute no spin; L(real)'s electron count is the total atomic number minus the
  aggregate charge.
- For a non-neutral or open-shell system this aggregation may be ambiguous about
  charge redistribution, and there is no better scheme: the log therefore prints
  every multiscale region and the whole system's charge and spin multiplicity so
  the user can judge (see section 11.2).

## 5. Schema normalize/validate: fail before launching

Pipeline order: read the raw input, keeping the origin and the set of explicit
fields → the adapter's field-level center split and the one-time conversion of
external indices → the schema's structural pre-check (depth, C, ci/spin lengths,
basic types) → normalize builds the single center table and derives
nstate/state_spin/pair maps/root per center → validate checks regions, electronic
definitions and cross-field constraints → capability planning → the frozen task
and dispatch/backend execution plan → backend initialization / MD start.

The structural checks run **before any `sum/zip/index`**, so a TypeError from
normalize is never the thing that has to explain itself without context.
`normalize.compute_derived_quantities` takes the strict per-center derivation via
`build_center_plan` for a nested ci; the C=1 path errors on any ci/spin length
mismatch (no zip silently truncating). When `charges`/`multiplicity` are lists
(C>1), `total_electron`/`nalpha`/`nbeta` are computed from the **aggregates**
(`sum(charges)`, the paired multiplicity from `aggregate_system_multiplicity`);
System.charges/multiplicity keep the per-center values (like a nested ci keeps
its own) and the aggregate is used only for the derived scalars.
`validate.validate_electronic` checks `mult_c >= 1` per center for a list
multiplicity and uses the aggregate for the global total_electron/unpaired checks.
`validate.validate_centers` runs the strict per-center checks and fails before any
external computation is submitted (the explicit "midlevel group count != C" error
comes from `build_center_plan`/`_region_indices`, naming highlevel as the only
source of C and midlevel as needing a one-to-one correspondence):

- highlevel non-empty, integral (bool and truncated float refused), in range; in
  three layers H_c ⊆ I_c and I_c non-empty; no overlap between the H's of a
  two-layer run or between the I's of a three-layer run. The same checks run at C=1.
- ci non-empty, every entry a non-negative integer, sum greater than 0 (a zero-state
  spin block is allowed; negatives and all-zero are not); spin the same length as ci
  (currently a non-negative integer S).
- root inside `[0, S_c)`; every coupling entry exactly two distinct integers inside
  its own center, with a duplicated unordered pair, a self-coupling or an
  out-of-range index reported as an error rather than falling back to the NAC
  default; pairs of equal spin classify as NAC and different spin as SOC
  (`classify_pairs`), with npair/nnac/nsoc consistent against the pair maps; the
  normalized pair direction is recorded explicitly, NAC flipping sign when reversed
  and SOC taking the conjugate.
- C>1 combined with fragment/groups, the old qmmm/reference special semantics and
  dynamic regions are refused by the capability table. Periodicity is not on that
  list any more: since #315 the rule is that the outermost, whole-system layer must
  carry a box and a center may be molecular, and the boundary a periodic center plan
  still has to respect is described in section 11.3.
- The executor enum is legal, the worker ceiling a positive integer, and an
  explicit conflict is reported.

An error message contains the full field path, the center's human number (1-based)
and internal number (0-based), the actual value and the expected condition, for
instance `md.root[1]: center 2 (c=1) has nstate=4; the input root=5 (1-based) is
invalid; expected 1..4.` A center resolution error is not swallowed into a generic
except and then echoed back; pure configuration errors may be collected in one
pass and reported together, but after reporting, no task directory is created, no
thread starts and nothing is submitted to Slurm.

## 6. Backend planning: unified terms and electronic binding

### 6.1 One expander

`planning/task.py` expands with a single `_expand_oniom_centers` built on the
centers: 1+2C for two layers, 1+4C for three, which naturally gives 3 or 5 items
at C=1; an ordinary single-method whole-system calculation generates one task bound
to c0. The order is real-first (L(real) first, then per center the +H/-L
corrections). Mathematical order and property source are kept apart:
`resolve_electronic_providers`/`resolve_primary_task` make `electronic_provider[c]`
point explicitly at H(c), so NAC/SOC and the active-state properties come from the
high-level provider and not from tasks[0].

### 6.2 Static task fields

```text
plan_version / plan_fingerprint
role / task_id / task_slot / layer_id / contribution
center_index / region_id / region_indices / transform_spec
region_charge / region_multiplicity
electronic_binding_id / state_policy / reference_state / provides_electronic
property_request_spec / expected_shapes / units / state_map
skill_binding_id / api / config / static_id
```

Task configuration follows the PyRAIConfig/skill_config construction path: a deep
copy of the template at initialization, then per task the local atoms / atom count /
masses / region electronic settings / nstate / state_spin / pair maps / initial
state are overridden, and the backend is created only afterwards. `_build_task_queue`
in `data/core.py` holds the `center_plan` and computes and writes
`task['region_charge']`/`task['region_multiplicity']` from the descriptor:
`center_index is None` (L(real)) → `plan.system_charge`/`system_multiplicity`;
otherwise → `plan.center(c).charge`/`.multiplicity` (H(c), L(c) and mid tasks all
take the center they belong to), and after the deep copy of
`task_config['skill_config']` the `charges`/`multiplicity` there are overridden
with that region scalar (so the backend gets the correct scalar at its very first
calculation instead of a global list). The injection only happens at C>1; C=1 keeps
the original merge/default path unchanged. Tasks of the same skill share no
results, no mutable method dict and no electronic arrays; different method
configurations inside one program are distinguished by binding_id, and a run that
would need several bindings but does not have them is refused explicitly (rather
than de-duplicating first and then inferring the layer count wrongly).

### 6.3 Per-task requests

A request is no longer one global target_states: each center's high-level task
reads the local states and couplings from its own c, a reference correction task
reads its own `reference_state`, and neither ever receives another center's active
state. At run time `md.resolve_target_states_centered` produces a flat list at C=1
(bit-identical to the old single-center request) and `{center_index: [states]}` at
C>1, which `calculator._task_target_states` routes to each work_item by
`task.center_index` (the baseline task with center_index=None receives None).
`sync()`'s field sources fall into three kinds: global shared, sliced by region
(atoms/mass/coordinates/velocities), and derived by electronic binding. At C>1 the
electronic state scalars are skipped (`_ELECTRONIC_STATE_FIELDS`: nstate/ci/spin/
coupling/state_label/state_spin/current_state), because the global
`pmd.system.nstate=1` would overwrite the per-center planned value; the active
state goes through the target_states parameter channel instead. At C>1
`charges`/`multiplicity` must also not be overwritten by the global value: `sync()`
skips the global synchronization of those fields for any task whose
`task['region_charge']`/`task['region_multiplicity']` is non-empty (protecting the
injected region charge/spin); at C=1 nothing changes bit for bit.

## 7. Dispatch and backend concurrency

### 7.1 Two scopes

The outer dispatch's multitask/multipool/remotetask control the process-pool
fan-out and launch location of a complete unit (trajectory/structure), and the
base node's remotetask is always False. Each role additionally carries a
serializable `backend_execution_descriptor` (`planner.backend_execution_descriptor`)
that controls the execution of the backend terms inside that role:

```text
backend_execution
  scope = backend_terms
  task_slots / executor(serial|thread) / max_workers
  failure_policy = atomic_evaluation
  multitask / remotetask = molecule explicit value or inherited by rule
  source = molecule | skill inherited | composite compatibility | default
```

`molecule.multitask/remotetask/multipool` are three-valued (Optional; an absent
None distinguished from an explicit False/1); the molecule namespace is preserved
through adapter/normalize/serializer and never round-trips through a flat
namespace where a skill/md field of the same name could overwrite it.
`resolve_molecule_flags` resolves: an explicit molecule value wins; on None it
reads the same-named three-valued declarations of the skills involved (consistent
→ inherited, True/False conflict → error, all None → False), and an explicit
molecule value can dissolve a skill conflict. It does not reuse `read_base_flags`,
which ignores remotetask. `composite.executor/max_workers` remain as the
mechanism/capacity compatibility entry and error at launch when they contradict an
explicit molecule setting. Even at C=1 with a single backend, the same backend
task-group execution protocol is used; the compatibility interpretation happens once
in planning, not as a center-count test inside the executor.

Scope precedence: the workflow's md/search multitask/remotetask control the
complete unit and take no part in the molecule's term-level overrides; a C>1 term
group prefers the explicit molecule value and otherwise inherits by skill; at C=1
without explicit molecule fields the existing skill/base scheduling behaviour is
kept; once a term control is consumed by molecule, that skill's dispatch flags no
longer fan out recursively. General safety constraints cannot be overridden by
input: a remote term worker never submits a further remote term group, and a
complete base node still has remotetask=False; `multitask=False` on the MD
base.theory restricts only the complete base unit and does not veto the molecule's
concurrency over the backend terms inside it.

| Outer unit strategy | Backend strategy | Result |
| --- | --- | --- |
| single trajectory, local | serial | all terms run in order |
| single trajectory, local | thread W | at most W of this step's centers' terms run at the same time |
| several trajectories, multipool=P | thread W | each worker's independent trajectory runs W terms per its own frozen plan |
| remote MD runtime | thread W | the backend thread pool sits inside the remote runtime, and the worker does not resubmit a remote MD |
| skill execution_mode=slurm | thread W | W independent backend runners submit and wait on their jobs concurrently, within the resource limits |

### 7.2 One work_item / outcome and the failure semantics

`dispatch/terms.make_work_item` defines the unified work_item (identity/routing
task_id/task_slot/center_index/evaluation_id/identity plus the payload
backend/local_atoms/local_coords/target_states/nac_pairs/soc_pairs, plus the audit
keys; `nac_pairs` has a consumer since #446 - a backend that declares
`supports_nac_request` is handed it per step and names the pair in its own deck,
and one that does not is simply not sent it, because a skill that names its
couplings in its own input needs nothing here); the outcome is `{task_id, evaluation_id, raw, success, error}`. Serial and
threaded paths use the same format and are index-aligned, so an out-of-order
completion changes neither the archive position nor the summation order.
`TaskExecutor` submits under control: in thread mode, after the first term fails it
cancels the futures that have not started, drains and collects diagnostics from the
running ones, and returns all outcomes;
`execute_with_failure_handling`/`apply_failure_policy` unify this into "any term
failing fails the whole evaluation" and raise, having collected the diagnostics of
the completed terms first. Thread safety is the backend's own answer, not a
framework register: `dispatch/terms.py::check_thread_safe` asks the instance
`check(runner, execution_mode, profile_semantics)` and propagates whatever it raises
(`terms.py:85-117`). A backend that declares no `check_thread_safe` runs concurrently -
that is the design default, since every skill offers both a local and a slurm mode and
opts out by name when one cannot be reentered - and the executor never degrades to
serial on the skill's behalf.

Execution flow: freeze this step's geometry and active-state snapshot → generate the
local work_items by task_slot → clear this evaluation's staging, initialize/update
the task-local backend input → `executor.execute(work_items)` → validate
completeness/no duplicates/no stale entries by task_id+evaluation_id → normalize and
route the per-center electronic properties versus the composite math terms → sum in
a deterministic order into a staged evaluation → MD commits everything after the
electronic step. A failed evaluation advances neither the electronic state, the
history, the RNG, nor the global results.

### 7.3 Thread isolation and resource limits

One persistent backend per task is reused across MD steps for its wave-function
cache, and the same instance never executes twice at once; config/scratch/log/output
file names are isolated per task, with the directory identity carrying
trajectory+task_slot+evaluation_id; the input snapshot is explicitly copied or made
read-only (`np.asarray` alone is not treated as ownership isolation); threads
neither read nor write the live state and do not touch the center electronic data
from a completion callback; the runner forbids process-level chdir, `os.environ`
modification, global stdout redirection and uncontrolled shared-library state. The
upper bound is roughly P trajectories × W tasks × T CPU threads, with GPU, licenses
and Slurm concurrency adding further limits; planning reports the budget and the
effective W instead of growing the workers without limit by center count.

### 7.4 The remote term worker (molecule.remotetask=True)

`molecule.remotetask` controls the isolation location of the backend terms; it is
not the Slurm setting of a skill's internal electronic-structure program, and it
does not launch a whole MD remotely. `workflow/dispatch/term_worker.py` provides
the lightweight term worker: `execute_remote_terms` submits each term,
`run_term_worker` executes one term and returns the unified TaskOutcome, and the
payload is only the task descriptor + the independent task config + this step's
geometry/electronic request + the execution identity; it performs no MD, no
composite sum, no parent-state commit and no full task queue. The entry is the
standalone `.rtt` (`python3 -m pyrai3md {name}.rtt` → a dedicated branch of
`pmd.run`, not `_execute_runtime_file`), with `TERM_WORKER_SUFFIX='.rtt'`,
`TERM_PAYLOAD_VERSION='1.1'` (bumped for the cell/image payload) and the result returned
as `term_outcome.json`; the
per-term identity goes through `RuntimeDirectoryManager` naming to avoid contention.
The term group's flags are frozen to False (one remote boundary), and the first
version refuses an undeclared nested remote submission. Remote failure, timeout and
cancellation use the same evaluation_id/task_id collection protocol, and a missing
remote configuration errors immediately.

| molecule.multitask | molecule.remotetask | Backend term group execution |
| --- | --- | --- |
| False | False | terms execute in the current runtime, one by one |
| True | False | the current runtime's thread pool runs all terms concurrently, under W and the audit |
| False | True | isolated/remote backend term workers are started in order and awaited |
| True | True | up to W isolated/remote backend term workers run concurrently |

## 8. Composite and the center electronic model

The first version uses independent per-center electronic propagation plus a fixed
reference correction. With R the single nuclear coordinate set, s=(s0,…,s_{C-1}),
and P the local gradient return mapping:

```text
two layers  U(R;s) = E_L(real) + Σ_c [E_H,c(R_Hc;s_c) − E_L,c(R_Hc;ref_c)]
three layers U(R;s) = E_L(real) + Σ_c [E_M(I_c;ref) − E_L(I_c;ref) + E_H(H_c;s_c) − E_M(H_c;ref)]
G(R;s) = each term's gradient in its own state, projected term by term, scaled, then summed
F(R;s) = −G(R;s)
```

The reference terms do not change with a center's active state, so a hop at center
c alone makes ΔU equal that center's high-level gap. The source of the
low/mid-layer reference state, charge and multiplicity is resolved and validated
explicitly rather than assumed state-independent by broadcasting a result onto
nstate.

`workflow/multiscale/composite.py` separates two jobs: `assemble_results` performs
only the ONIOM mathematical composition of energy/gradient
(`STRICT_AGGREGATE_KEYS`), and `collect_center_electronic` files each center's
electronic properties as they are into `state.electronic[c]`. Non-aggregate keys
fall into four classes: `STRICT_AGGREGATE_KEYS` (energy/gradient) sum linearly by
contribution coefficient; `PER_CENTER_KEYS` (nac/soc/dipole/transition_dipole) are
filed per center unchanged (stored in that center's LOCAL framing, kept out of the
ONIOM sum, never inherited through a single primary task, never broadcast into a
global state axis); `REFUSED_KEYS` (overlap/hessian, expensive, produced on demand,
with no global aggregation rule) raise `ValueError` as soon as multi-center is
asked for them; `IGNORED_KEYS` (computed_charges and other products attached
**unconditionally**, with no multi-center charge contract and no framework
consumer today) are **silently dropped at C>1** instead of raising -- otherwise xtb,
which writes that file whenever it produces charges, would hard-fail every
openqp/xtb ONIOM run. C=1 still passes them through as before. Should a whole-system
charge consumer appear later, a per-region scatter contract can be defined for such
a key and moved out of `IGNORED_KEYS`. At C>1 no common `energy[nstate]` state axis
is built; the total result keeps
`active_configuration`/`active_energy`/`active_gradient`. C=1 may still export the
traditional all-state view, but the computation foundation keeps selecting states
through the same active-surface evaluator.

### 8.1 Shapes, units and electronic states

The global energy is `(nstate,)` and the global gradient `(nstate, natom, 3)` (that
is the C=1 view; at C>1 there is no global state axis and each center keeps
`(S_c,)`/`(S_c, A_c, 3)`); a local gradient's atom axis must equal the number of
atoms the transform outputs. A single-state scalar energy and an `(nlocal,3)`
gradient are promoted only by an explicit single-state adapter; every other shape
error is reported, and numpy broadcasting is never allowed to guess a state count
or an atom axis. Every term required for this step demands an energy, and every
gradient job demands a gradient; a missing entry, an empty array, NaN/Inf, a wrong
state label or count, or a non-success status blocks assembly. Coordinates in Å,
energy in Hartree, gradient in Hartree/Bohr; a finite difference displaced in Å
must convert its derivative to Hartree/Bohr before comparison.

### 8.2 Gradient scatter-add

`workflow/multiscale/regions.py` provides the pure geometric functions (importing
no Calculator/composite/state): `validate_region`, `materialize_region` (the local
slice), `project_gradient` (scatter-add into the whole system),
`project_energy`/`project_nac` (including the NAC sign flip)/`project_soc`
(including the SOC conjugate)/`project_dipole`, and `slice_field`. With no cap,
each local atom j maps to global i as `G[:, i, :] += coefficient * g[:, j, :]`;
every term is validated, mapped and only then added -- local gradients of different
sizes are never subtracted first, and a result is never written into every atom.
Where several contributions reach the same global atom they scatter-add rather than
overwrite.

## 9. Surface hopping dispatch: reading compiled, writing back

### 9.1 Factory and binding

`surface_hopping(method, state)` in
`workflow/integrator/surface_hopping.py` runs once per trajectory and, for
gsh/fssh/lzsh, returns the `ElectronicStepper` from
`workflow/integrator/electronic.py` (nosh returns a pass-through). The
`ElectronicStepper` builds one `ElectronicBinding` per center (at C=1 exactly one
transparent binding) and runs the centers in the fixed declaration order
0..C-1 every step. Algorithm instances are independent per center (one instance
with a cached nstate is never reused across different centers); instances,
accessors and futures never enter a checkpoint.

### 9.2 The CenterView proxy and zero change to the cores

The fssh/gsh/lzsh numeric cores are untouched: they still read a PyRAIState-shaped
object and still write back through update_system/update_dynamics. A non-transparent
binding runs the core against a `CenterView` proxy, which translates that center's
per-center data into the single-center interface the core expects: nstate→S_c,
current_state→`get_active_configuration()[c]`, result→the projected per-center
arrays, history→a reshape of the structured frame, and coordinates/gradient/nac
zero-masked to high_indices. C=1 uses the transparent binding (the view is the real
state), so it is bit-identical to the old single-center path.

Data shapes: energy/soc stay in the center's LOCAL framing (`(S_c,)`/`(P_soc,)`);
gradient/nac are scattered onto the global atom axis through
`regions.project_gradient`/`project_nac` and zero-masked outside the region;
coordinates are zero-masked over the same high_indices (GSH uses coordinates only as
differences, a displacement outside the region is always zero, and the coupling is
naturally local). The masked region is uniformly `high_indices`, the same projection
region the electronic provider uses.

### 9.3 Write-back and velocity confinement

The core's write-back is captured by the binding and lands on its own center only:
density persists through `set_center_electronic(c, density=...)` (the population is
derived from the diagonal of the density's real part and is never stored separately);
the post-hop active local state writes back to `active_surface.active_configuration[c]`;
the velocity write-back is masked to that center's atoms (`_apply_velocity`), so
even when adjust=1/isotropic is a global proportional scaling, no other center's
velocities change. Centers do not overlap, and each center's state and velocity
adjustment is independent.

### 9.4 The ordered event ledger and the RNG

The binding runs the centers in the order 0..C-1, and each center's write-back lands
in the state before the next center refreshes and reads, so a later center sees the
V/s after the earlier center's events (the ordered event ledger). Each center reuses
`tsh_helper` (reflect_velo/adjust_velo with adjust==2 along the NAC; adjust==1 and
adjust_velo_isotropic as a global scaling) but confines its write-back to its own
center. The RNG is not unified: the core keeps using the global `np.random`, and
determinism comes from the plan's seed (`md._seed_trajectory`) plus the fixed center
order; no shared random stream is introduced.

## 10. History, nuclear integration and the energy ledger

### 10.1 The nuclear integrator reads the combined active gradient

verlet integrates only the global R/V/F and reads the "combined active composition
gradient"; it no longer picks a gradient row by a historical state number:

- current step: `state.active_gradient()` -- at C>1 it returns the committed
  `active_surface['active_gradient']`, at C=1 it degrades to
  `gradient[current_state]` (bit-identical to the old read), and at C>1 it returns
  None when nothing is committed (never falling back to the global current_state,
  which would raise).
- previous step: the history frame's `nuclear_sample.active_gradient` (the combined
  active gradient captured at that `update_history` point).

The existing force-sampling discretization is preserved: the coordinate advance uses
the post-hop velocity and the historical (previous step's active composition)
gradient, and the completed velocity update combines the historical (previous step's
active composition) with the current combined active gradient. Center atoms never
overlap and each center's state and velocity adjustments are independent, so the
combined total-system velocity/force/energy is transparent to the verlet
integration. The two half-step formulas, the unit conversion (`BOHR_TO_ANGSTROM`),
the internal first-step skip and the shape/mass guards are unchanged.

### 10.2 The MD main loop

Per-step order: verlet_i (reads the previous frame's combined active gradient,
produces R_next) → calculator.run (runs every task against R_next, staging and
committing each center's electronic data and the active surface) → verlet_ii (reads
the current combined active gradient plus the previous frame's, produces V) →
standalone nuclear operations (phase_correction / constraints / kinetic energy /
velocity correction / ensemble) → update_history (writes the structured frame,
including the pre-hop sample and the combined active gradient) → surface hopping
(the ElectronicStepper propagates per center and handles the events) → verification
→ sync → checkpoint (output plus runtime metadata) → advance_step.
phase_correction handles only the NAC phase and is a safe no-op at C>1 when the
global result carries no nac.

### 10.3 The energy ledger

When only center c hops, ΔU equals that center's high-level gap and ΔK=−ΔU is
implemented by `tsh_helper`'s velocity adjustment (solving the quadratic along the
NAC direction and taking the valid real root with the smaller velocity change, or
the declared global proportional scaling); a frustrated hop keeps the active state
and a reflection conserves the kinetic energy. Under multi-center, each center's
velocity adjustment stays confined to its own atoms and one center's global scaling
never crosses into another; a center-local kinetic energy pool is a different
physical policy and is not the default. Hopping residuals are accepted against the
numeric solver's tolerance, Verlet against step-size convergence and long-term
drift, and the exchanges with a thermostat/constraint/external field are booked
separately.

## 11. Persistence, copying, output and the capability boundary

### 11.1 Serialization and recovery

The payload of `data/runtime/serializer.py` carries a file-level version block
(`SERIALIZER_FORMAT_VERSION='1.0'` plus the partition/history/electronic semantic
versions, the latter two read from `data.center.CENTER_PLAN_VERSION` (now `'1.1'`)
and `data.core.HISTORY_FRAME_VERSION`). `to_dict`, prepare/load,
deep_copy/shallow_copy and the dispatch sub-trajectory materialization all cover
partition_plan/electronic/active_surface/continuation/history; the partition_plan's
`to_dict` includes each center's `charge`/`multiplicity` and the whole-system
`system_charge`/`system_multiplicity`. A read-only static plan may be shared, but
mutable center data, history, RNG and task config must be isolated. A versionless
legacy `.rt` migrates into a C=1 record (a canonical plan rebuilt through
`build_center_plan`, including the new charge/multiplicity fields, with the center
carrier filled in and center_id → center_index); an old file carrying a
`partition='1.0'` version block is rejected outright by `_check_version` because 1.1
is incompatible with it (it cannot be resumed as it stood). On recovery a mismatching
plan_fingerprint is rejected as stale. A restart bypasses raw input and first-pass
planning: the checkpoint carries the center order, regions, electronic definitions
(including region charge/spin), task identities, the execution plan and the model
fingerprint, and after the recovery checks the backend and electronic algorithm
instances are rebuilt.

The MD production-line checkpoint serialization and resume are wired (#314/#316): the
RNG stream travels as `RuntimeState.random_state`, stamped with the runtime file at
`workflow/md.py:868` and restored by `workflow/md.py::_seed_trajectory` instead of
replaying the seed at the seam, the main loop runs on the absolute
`runtime.current_step`, and the frame
records are append-mode files, so a restart adds to the trajectory rather than
overwriting it (`workflow/records/output.py:264-294`). What is still open is
engine-side evidence, not code: the same split-versus-continuous pair has to be
repeated on LAMMPS' persistent and subprocess channels in the test environment machine
(dev/TODO.md section 2).

### 11.2 The trajectory output center protocol

`MdOutputWriter` in `workflow/records/output.py` takes one of two paths depending on
whether an environment partition exists (`partition_plan.n_layer >= 2`, i.e. a
highlevel region is present), decided by the `center_regions` (per-center
high_indices) passed at construction:

- No environment partition (C=1, empty high_indices): only the global
  `md.xyz/md.velo/md.energies/md.pop` plus the hop-point `sh.*` are written,
  byte-identical to the old single-centre behaviour.
- With an environment partition (C=1 with a region, or C>1):
  - The global files carry system totals: `md.xyz`/`md.velo` are the system's
    coordinates/velocities; `md.energies` has columns
    `time | Epot | Ekin | Etot | each center's active state number | each center's per-state energies`
    (concatenated in center order); `md.pop` writes one row per time step with the
    columns expanded per center's states (the populations then each center's 0/1
    active flag); `sh.*` holds the hop-point global data (post-hop active
    configuration and post-hop velocity).
  - Per-center files (where # is the 1-based center number, restricted to that
    center's high_indices region, same format as the global ones):
    `md.#.xyz`/`md.#.velo`/`md.#.energies`/`md.#.pop` and
    `sh.#.xyz`/`sh.#.velo`/`sh.#.energies`/`sh.#.pop`.
  - At C=1 with an environment partition the global files collapse to the
    single-center result and `md.1.*`/`sh.1.*` are written additionally.

Data sources: each center's per-state energies come from the frame's
`electronic_outcome[c]['energy']`; each center's active state from the pre-hop
`nuclear_sample.active_config_pre[c]` (md.*) or the post-hop
`get_active_configuration()[c]` (sh.*); each center's population from the diagonal
of the real part of that center's density (at C=1 without a partition,
`dynamics.state_population` is still read); the system Epot from `active_energy`
(with a partition) or `energies[pre_state]` (without), and Ekin from the snapshot's
masses and velocities. Output never writes back a compatibility field in a way that
affects numbers. `TrajectoryResult.center_history_step(center, index)` in
`workflow/records/trajectory.py` provides the read-only per-center history query.

The run log (`log_step` in `kernel/logger.py` →
`utilities/print_results.format_result_block`) follows the center protocol for a
multi-center frame: `snapshot_centers(snapshot)` builds one record per center from
the frame's centers/electronic_outcome/active_config_pre (including each region's
`charge`/`multiplicity`), and the `&electronic state` and `&state population`
sections list each center's active state and per-state populations in center order
(1-based); the `&region charge & spin multiplicity` section lists each center's
(1-based) region charge and spin multiplicity and adds one line with the
whole-system L(real) aggregates (`sum(charge_c)` and the paired multiplicity from
`aggregate_system_multiplicity`) so the user can judge a charge-redistribution
ambiguity. A single-center frame (including C=1 with an environment partition)
keeps the original global output and does not print that section. The region
charge/spin data source is the PartitionPlan bound to the state (the frame's
`centers[c]` being `CenterSpec.to_dict`).

### 11.3 Capability boundary

adaptive/deviation/training, the optimizer, Hessian and scan open up according to
their own capability declarations: with no center training labels and no
multi-center higher-derivative contract, the corresponding combinations are refused
explicitly (the optimizer reads the global `state.nstate`, so C>1 raises
`CenterPlanError`), while the already supported single-state / C=1 paths stay as
they are. "No bond breaking, no cap" is the first version's model boundary;
non-overlapping indices do not by themselves prove no bond is broken. Where trusted
topology exists, cross-region bonds are checked; where there is no topology, the run
does not claim to have verified this automatically.

Periodicity adds a boundary of a different kind, and it belongs to the layers rather
than to the partition. Since #315 the rule is that the OUTERMOST task - the whole-system
provider - must be able to carry a box, and a center may be any molecular method
(`dev/methods/PBC_DEVELOPMENT.md` section 5.4). Since #316 a second condition is
measured rather than assumed: a 2-layer plan hands the LOW layer every center slice
(`H(c) - L(c) + L(real)`), so the low layer has to be able to evaluate a cluster. #316
worked around that by making the center cover the whole system, which is not the feature;
since #317 it is implemented. A skill whose topology comes from a fixed template now cuts
that template to the rows of the region it was handed (`atom_selection`,
`dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` section 5.6a), and the framework's whole
contribution is those row numbers plus the region geometry in the same order.

**A center is a set of atoms, so its indices are canonicalised.** Input does not
have to declare them in any order, and nothing downstream has to defend one:
`schema/adapter.py::_parse_center_indices` sorts each group ascending, for every
carrier - the classic string form already did it through `ReadIndex(sort=True)`, and
the typed list branches now do the same, which is what makes a yaml nested list, a
classic keyword and a programmatic assignment describe one center
(`TestTheRegionTaskIsHandedItsRows.test_every_input_carrier_describes_the_same_centers`
declares `[[3,1,2],[6,4,5]]` and reads back `[[0,1,2],[3,4,5]]` all three ways).
Duplicates and out-of-range indices are refused by `schema/validate.py`, not
repaired here. The reason the order matters at all, rather than being cosmetic, sits
one layer down: `regions.py::materialize_region` and the static slice in `data/core.py`
index in that order, the row numbers handed to a classical low layer are
`region_indices + 1` in that order, and the skill renumbers its template rows by
*position* because its per-step writer puts coordinate row `i` on the row numbered
`i + 1`. So the rule is: framework canonicalises, skill obeys. A skill that took it
into its own hands would be the one that mis-assigns atoms to coordinates -
`skills/lammps_skill/tests/test_ff_model.py::TestSelection` therefore asserts that a
handed `[3,1,2]` produces a different subsystem than `[1,2,3]`, and that the skill
never sorts.

**Worked example of those row numbers, on a box built outside this package.** The
variable-cell crossing vehicle is a 25 A TIP3P box with one ethylene in it (1581
atoms), built by the local `ffauto` and equilibrated by its own NPT template. Packmol
put the solute in last, so in the engine's data file the ethylene is molecule 526 and
carries atom ids 1576-1581, which means the high-layer region is named by those
**template row numbers** - `highlevel: 1576-1581` - and the framework's contribution
stays what the paragraph above says: those numbers plus the region geometry in the
same order. The rows are found by reading the data file's own mass column back into
elements, not by trusting the build order: the two `write_data` passes of one and the
same system print the tip3p oxygen mass as `15.9994` before the run and `15.999`
after it, so a lookup keyed on the printed text resolves one atom type one way before
the run and another way after it. Two post-processing steps on the produced file are
mandatory and both were measured, not assumed: its `Velocities` section has to be
dropped whole (the skill's force-field compressor refuses a data file carrying
velocities, which costs the resident channel, and dropping only the section header
leaves 1575 velocity rows sitting between Atoms and Bonds, which corrupts the file),
and the box has to be translated so its origin is zero and wrapped into [0, L), since
the framework's cell is three lengths plus three angles with a zero origin. Nothing
else about the box changes, and the solute's charges come from a RESP fit at
HF/6-31G* over `ffauto`'s own van der Waals sampling surface (13518 points, RMS 1.37
kcal/mol against the quantum potential), because an invented charge set would put the
invention inside the vehicle every later reading is taken on.

A subsystem is then evaluated **non-periodically**, because the difference term is supposed
to remove the center's self-interaction and nothing else: keeping the box adds the cluster's
Ewald self-image energy to what gets subtracted (4.464e-4 Ha per center measured on the
shipped box). This is the boundary condition of section 5's "no bond breaking, no cap" read
at a face: an entity the cut breaks is dropped from the subsystem and recorded in the model
audit, because the crossing interaction stays in `L(real)`. The same channel serves a
three-layer plan's middle shell (`L(I_c)` is `region='inter'`, one shell wider than the
center), which is certified at plan level here because a middle-layer QM engine is what this
host lacks. `examples/multicenter_pbc_nve.yaml` and `examples/multicenter_pbc_npt.yaml` are
now that shape - two disjoint three-row centers of one 60-row topology - and
`dev/methods/PBC_DEVELOPMENT.md` section 4.2 carries the readings, including the barostat's
missing virial, which is a separate routing defect and not a property of the cut.

## 12. Implementation file map

| Layer | File | Responsibility |
| --- | --- | --- |
| center contract | `data/center.py` | PartitionPlan/CenterSpec/ElectronicSpec, build_center_plan, CenterPlanError |
| state core | `data/core.py` | center data / active surface / history / commit interfaces, active_gradient(), HistoryFrame |
| serialization | `data/runtime/serializer.py` | version block, center field coverage, legacy .rt migration |
| input | `schema/{config,adapter,normalize,validate}.py` | flat/nested dual form, per-center normalization and strict checks |
| planning | `planning/task.py` | _expand_oniom_centers, identity fields, electronic binding, molecule flags, descriptor |
| execution | `workflow/dispatch/terms.py`, `workflow/dispatch/term_worker.py` | work_item/outcome, controlled submission, failure protocol, remote term worker |
| calculation | `workflow/calculator.py` | run_in_process, target_states routing, sync guards, stage/commit |
| composition | `workflow/multiscale/composite.py`, `workflow/multiscale/regions.py` | ONIOM math, per-center collection, pure geometric projection |
| electronic dispatch | `workflow/integrator/electronic.py`, `surface_hopping.py` | ElectronicStepper/CenterView/ElectronicBinding, the factory |
| nuclear integration and MD | `workflow/integrator/verlet.py`, `workflow/md.py` | combined active gradient, center-aware target_states, the main loop |
| output | `workflow/records/output.py`, `workflow/records/trajectory.py`, `utilities/print_results.py`, `kernel/logger.py` | the center-protocol output file matrix, center_history, the run log |

## 13. Acceptance test matrix

The multi-center foundation is accepted against the following matrix (all of it in
`qualification/verification/`, using mock/analytic backends; the per-item coverage
map to test modules is in `QUALIFICATION_DEVELOPMENT.md` §11):

Input and schema: the three input carriers give the same canonical plan;
`ci 2 2; 2 2`/`spin 0 1; 0 1` yields two centers with nstate=4; differing state
counts / differing atom counts / an empty coupling / a zero-state block / a defaulted
field; an empty semicolon group, mixed depth, an inconsistent C, a spin length
mismatch, a negative or all-zero ci, a bool or non-integer, a root out of range, a
duplicated/self/cross-center pair, an out-of-range or overlapping region: all fail
before launch; a raw → normalized → serialized → recovered round trip never
subtracts 1 twice.

Backend and scheduling: an ordinary single task, C=1 two/three layers and C=2
two/three layers through the same executor (item counts 1, 3/5, 5/9, with real
computed once); every task reads its own local data without cross-talk, and two
consecutive steps do not read a stale result; a Barrier proves real entry into the
thread pool with a concurrency peak >1 and ≤W, and an out-of-order completion
matching the serial result; the inner thread still works when the MD base
multitask=False; a skill flag conflict / a backend refusing threads by itself / an
invalid W / a directory collision / a failed-stale-duplicate-missing outcome are
each handled explicitly; both molecule flags in the four combinations, and a remote
worker executing exactly one task with recursion forbidden.

Electronic propagation and physical consistency: at C=1 the unified coordinator and
the single-center reference are compared bit for bit over the same random input for
density/population/events/velocities/forces/recovered state; a hop at center 0 does
not change center 1's density or state numbering, and nosh still selects the right
active surface; sentinel frames verify the FSSH endpoints and the GSH/LZSH
three-frame window, warmup, phase correction and pair mapping, with each step
reading fresh arrays rather than an initialization-time reference; two centers
competing for insufficient kinetic energy never both succeed; the finite difference
of a multi-active-composition total energy matches the global gradient, and
heterogeneous state axes are never broadcast; verlet's two half-steps read the
combined active gradient, with zero direction / zero kinetic energy / unequal masses,
the event residual and NVE step-size convergence; after a failure the committed R/V/
electronic state/history/RNG/next_step are unchanged.

Recovery and compatibility: a committed state save/load round-trips completely
(including partition_plan/electronic/active_surface/continuation/history/RNG);
changing task completion order or trajectory scheduling order never crosses center
data, random streams or working directories; the existing regressions stay green and
new schema/center/plan/electronic integration cases are added; toggling only the
long-term probability output does not change the numeric path, a C=1 exported
compatibility view is not a second data source, and a legacy scalar access at C>1
errors immediately.

Periodicity is the one part of this matrix that is not mock-only:
`test_multicenter_periodic.py` reads the real skill manifests for the capability rule,
cuts regions out of a frame a `build_state` actually produced and drives the live
`Calculator` with a recording backend, and the two `examples/multicenter_pbc_*.yaml`
inputs are run end to end against PySCF and LAMMPS (section 11.3 says what that pair
certifies and what a fixed-topology low layer keeps out of reach).

## 14. Capability boundary and later extensions

The first version's physical model: independent per-center electronic propagation
plus a fixed reference correction, no bond breaking and no cap. State-dependent
low-level differences, cross-center NAC/SOC, energy transfer, entanglement and a
complete direct-product Hamiltonian are not implemented just because "the
partitioning is unified", and an unaccepted combination is refused during planning.
Half-integer S is not quietly opened through the input wrapper.

Later extensions each need their own physical and software contract; the current
design reserves only reference/mapping/version fields and does not imply support:

- Bond-breaking RegionTransform / link atom: BoundarySpec (inside/outside global
  indices, cap element, a distance or a fixed ratio, and the periodic image
  information when needed), with the local order fixed to real atoms followed by
  caps, and the context holding this step's mapping/Jacobian. The fixed-distance
  model `r_L=r_A+l·u`, `J_B=(l/|d|)(I−u uᵀ)`, `J_A=I−J_B`, `g_A+=J_Aᵀ g_L`,
  `g_B+=J_Bᵀ g_L`; the fixed-ratio model `J_A=(1−λ)I`, `J_B=λI`; the two are never
  mixed. A zero-length bond is refused and several caps accumulate onto the same
  global atom; the main composite sees only the global projected results and the
  coefficients.
- Periodic boundaries: an explicit cell/image contract or already-unwrapped bond
  vectors are required, with cross-cell bonding and the non-differentiable points of
  the minimum-image switch handled separately; where unsupported the entry point
  refuses rather than producing a wrong gradient.
- Several methods bound to one program, overlapping centers and dynamic regions:
  each still needs its own physical and software contract. Three former entries
  are delivered and are no longer on this list (#453/#454/#457/#460): a
  **charged QM region** (the centre's own declared charge, the MM charges from
  the topology, and the per-centre gas removal taken with that centre's fitted
  set), **multi-center multi-state models** (per-centre NAC/SOC blocks, per-centre
  crossing pairs summed into one gap, one FSSH ledger per centre), and **MD
  checkpoint serialization and resume** (the four state objects plus the RNG
  stream, with a two-process run reproducing the one-process frames digit for
  digit). General (additive) QM/MM is likewise delivered, with its contract in
  methods/QMMM_DEVELOPMENT.md section 8 and its invariants in 15.6 below; what
  remains deferred there (link atoms across a partition, mixed QM providers) is
  stated there.

## 15. Charge management and electrostatic embedding (multi-center)

Section 14 once listed "charge embedding" as a later extension; the ONIOM3 midlevel
electrostatic embedding is now implemented (framework side in
WORKFLOW_DEVELOPMENT.md §18, routing and &composite in
methods/ONIOM_DEVELOPMENT.md section 3 and COMPOSITE_DEVELOPMENT.md section 16), and so
are the two-layer outer-charge form and the additive QM/MM form, where the
embedding data is a field rather than a charge list
(methods/QMMM_DEVELOPMENT.md section 8). This section records the
invariants specific to multi-center: the ONIOM ones below, and the QM/MM ones
in 15.6.

Per-center embedding: each center's high provider H(c) embeds only that center's own
disjoint midlevel I_c and never embeds another center's charges or the lowlevel
region. `embedding_for_task` reads `embedding_region_by_center[c]` from the task's
`center_index`, which gives the inter-center isolation by construction.

Two-layer multi-center is the other shape. There is no midlevel, so center c's
environment is every atom outside its own highlevel, including another center's
high-level atoms. The charges are that evaluation's whole-system `L(real)` row
(`capture_outer_reference`), handed to phase two through a temporary ChargeStore:
one low-level row fills every environment, and no cross-center charge iteration
is performed. The persistent store still keeps the #450 rule - its highlevel rows
are written only by the providers - so the capture is a separate per-evaluation
array, not a store update. A static `read_charge` file under several ONIOM2
centers is refused (one file has no per-center binding), and a classical
(LAMMPS) outer is refused with a pointer to QMMM.

Charged centers: `CenterSpec.charge`/`multiplicity` is the center's electronic
definition, `PartitionPlan.system_charge` is their sum and
`system_multiplicity` pairs the unpaired electrons. `data/core.py` hands the
center charge to every region task of that center and the aggregate to the
full-system task. A three-layer center's midlevel must be neutral, because the
`H ∪ I` mid region inherits the center's charge one-to-one; the net charge
belongs in the high region. The shipped charged carriers are
`multicenter_oniom2_orca_charged_*` (ammonium +1 beside a neutral water) and
`multicenter_oniom3_orca_charged_*` (ORCA high/low, PySCF midlevel).

Per-center charge routing: highlevel charges come only from the H(c) entries of
theory[0] (each H_c fully updated); midlevel charges come only from the inter terms
M(I_c) of theory[1] (the disjoint I_c only, the H_c part ignored); theory[2] and every
correction term update no charges. When a center's backend produces no
`computed_charges` (openqp, for instance) that region skips the update and logs an
info line: no error, and no other center is blocked.

The row written is the one for the state that task ran (#450):
`workflow/multiscale/charges.py::_charge_row` takes `arr[state]`, where the state
comes from `Calculator._charge_embedding_state` handed in per task at both
writeback sites (the phased path and the single path). A payload with fewer rows
than the requested state writes **no** row and logs a warning - row 0 is not a
stand-in, which is the same rule the QM/MM stage follows.

Storage and two phases: `ChargeStore` is a single global per-atom array
(`atom_charges (natom,)`) whose managed region is highlevel ∪ midlevel
(= ∪_c (H_c ∪ I_c), centers being non-overlapping), and `point_charges` are emitted
as whole-system rows (row i = atom i), so multi-center is covered naturally. Execution
splits into two phases by `provides_electronic`: phase 1 runs every non-provider term
(each center's M(I_c)/L terms included) and updates the midlevel, phase 2 runs each
center's H(c) with that center's updated embedding and then updates the highlevel. Both
phases share one evaluation_id, and the routed results are merged back in the
original tasks order; the per-center collection (collect_center_electronic) and the
active-surface commit are unchanged.

Output: the per-term labels of the &composite energy breakdown gain `[<center+1>]`
under multi-center (L(real) has no center and omits it), ordered high→mid→low and
then by center; at C>1 there is no global state axis, so `composite_total` is the
single value of the active composition U. When charge management is active, MD
writes, besides the global `md.charges`/`sh.charges`, per-center
`md.#.charges`/`sh.#.charges` for a partitioned run (sliced to that center's
highlevel region by coordinate match). For a run without charge management the file
matrix and the numeric path are bit/byte unchanged.

### 15.6 The additive QM/MM route: four invariants that the reading depends on

QM/MM embeds through a field the classical stage measures, so its multi-center
contract is not the ONIOM one restated. Four rules hold it together; each is a
decision about what may be read, not about how the arithmetic is arranged.

**One snapshot per round (Jacobi).** Every center's charge round reads the same
charge array, the one the previous round left, and writes only into its own
region of it. Consequence worth testing: the answer cannot depend on the order
the centers were declared in. A Gauss-Seidel sweep would make it depend on
exactly that.

**Cold start.** Each evaluation's iteration begins with every QM center
uncharged, and the intermediate rounds live only inside the evaluation; the
store is written once, with the accepted values. A charge carried over from the
previous frame would make this frame's energy and forces answer to the
trajectory they are supposed to predict, and the run would stop being
reproducible frame by frame.

**The union array computes, the ids address.** The whole QM set in global atom
order is what the field is assembled over and what the classical stage's Q-row
corrections are reduced against - one pass, no loop over centers. Which row
belongs to which center is decided by atom id, in the `.esfp` handoff and in
`qmmm_components['centers'][c]['qm_atoms']`, never by concatenation order: two
centers reading each other's field by an accident of ordering is the failure
this rule exists to prevent. The only per-center arithmetic left is the in-cell
Q-Q removal (`U_gas`), because that term is defined inside a region.

**No fallback inside a charge round.** A round whose provider returns no
`computed_charges` stops the run. Outside a round the fallback chain (readback →
store → `molecule.q_qm`) is legitimate, because the value only feeds a report;
inside a round the number is about to become another center's Hamiltonian, so a
stored or pinned stand-in would iterate on a value nothing computed.

Measured on the shipped `multicenter_qmmm_sc_charge.yaml` (Molcas + LAMMPS, the
20-water box, two face-straddling waters): two rounds, max-norm move
9.140667e-1 then 7.000000e-4 against a 1e-3 tolerance, `stopped by tolerance`,
total energy -152.5288369992 Hartree. The same input asked one round instead of
a fixed point (`sc_charge_maxiter: 1`, which is what
`multicenter_qmmm_sp.yaml` ships) answers -152.5288361253, and with no charge
round at all - the setting `md`, `mep` and `num_freq` carry, where the probes see
uncharged centers - the first frame of `md` answers -152.5287129798. So on this
geometry one fitted round buys 1.23e-4 Hartree of inter-center polarization and
the fixed point adds 8.7e-7 more on top of it. The reciprocity and per-center
identities behind these are in methods/QMMM_DEVELOPMENT.md section 9.
