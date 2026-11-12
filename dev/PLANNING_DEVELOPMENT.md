# PyRAI3MD Planning Development Guide

Last updated: 2026-11-12 15:50 (Rule 10 now records zero imports plus the pure declaration and fix-mask locations; the `data/core.py` references point at `planning/assembly.py`. Earlier: 2026-11-09 16:40 (task assembly moved from data/core.py to planning/assembly.py as `build_task_queue`; `resolve_electronic_providers` was removed and its provider map is read from the descriptors. Only that move was checked. Earlier: 2026-11-06 13:40 (section 5.7 is new - the adaptive reference queue is expanded from `control.reference` and `molecule.highref/midref` on its own terms, the label source is a region-set comparison carried as `label_plan`, and a composite reference task's `static_id` takes the `ref_` prefix so it cannot share a calculation directory with the theory task of the same region, role and skill; the 6.1 descriptor block now lists `label_plan`. Only those two places were checked, against `planning/task.py` and against the six `examples/adaptive_*.yaml` plans. Sections 1-4 and the rest of 6 were not re-audited. Earlier: 2026-11-04 11:10 (#451: 4.1 distinguishes `'qmmm'` from `'multi_center_qmmm'` and the queue block gains the per-center QMMM expansion (field xC with a narrowed probe, the optional `qm_charge` xC, `qm` xC, one classical `energy` with `qm_centers`); 5.2 states which keys the multi-center duties add and why `probe_indices` has to survive the task whitelist. Sections 1-4 and 6 were not re-audited this round. Earlier: 2026-11-02 10:40 (#448: 4.2 gained the forwarding reading measured on the loading chain (#448): `coupling` and both pair maps reach the skill section through the ordinary merge, the state-key overlay re-asserts four state fields, and the defect was the receiving side overwriting them - no key list changed. The rest was not re-audited. Before it, 2026-11-01 20:50 (section 4.2 and the step list of 5.2 now carry #447: the coupling need is scanned into `resolve_capabilities` with every other requirement keyword instead of being read at the jobtype-spelling site, the pair it asks for must be one `molecule.coupling` declares, and the line anchors of the resolver and of `data/center.py` were re-read. Before it, 2026-11-01 11:00 (section 4.2 gained the two rules #446 put on this channel: the coupling capability follows the optimizer's `uses_derivative_coupling`, and the crossing pair travels per step; sections 1-3 were not re-read). Before it, 2026-10-22 22:20 (the data tree no longer lists the deleted `structure/` and `dynamics/` sub-packages; pair information is attributed to the center plan's `ElectronicSpec`/`classify_pairs` and reaches terms as `property_request_spec`; the token map carries `hessian`/`num_freq`/`num_hess`/`stress`; the resolution list gained the `_derive_stress` step; fragment detection reads `system.center_groups`; rule 9 is narrowed the same way as CALCULATOR. The rest was not re-audited. Before it, 2026-09-08))

---

> **Multi-center unification (refactor R2-R3, 2026-09-08).** CHANGELOG
> #200-#201, `dev/MULTICENTER_DEVELOPMENT.md`. R2 merged the single- and
> multi-center ONIOM expanders into one `_expand_oniom_centers` (real-first
> order: L(real), then the H(c)/-L(c) corrections), added the task identity
> fields (task_slot / center_index / task_id / region_id / provides_electronic
> / state_policy / reference_state) and `resolve_electronic_providers` /
> `resolve_primary_task` (NAC/SOC and active-state properties route to the
> high-level provider, not tasks[0]). R3 added `resolve_molecule_flags`
> (molecule multitask/remotetask/multipool, tri-state, skill inheritance with
> conflict detection) and `backend_execution_descriptor` (the serial/thread +
> local/remote execution strategy stamped per role, serializable). The canonical
> center table (`build_center_plan` in `data/center.py`) is the single source of
> the partition axis (C=1 included); planning terminates before execution.

## 1. Design Objective

The `planning/` package determines **what** should be computed. It transforms a scientific task description (jobtype, dynamics keywords, optional property flags) into a concrete list of capability name strings that are distributed to each skill backend.

The key design principles are:

- Users should not care about the detailed capabilities needed for their calculation. Users only input scientific-relevant information.
- Planning should not use hard-coded profiles to map user input to capability lists. Instead, it derives capabilities from jobtype tokens and workflow-related keywords.
- Planning keeps the original jobtype information for subsequent workflow resolution, while resolving the complete capability list independently.
- Planning never differentiates basic and advanced workflows. It provides a complete capability list stored in PyRAIState; the workflow decides which function to call.

> PyRAI3MD never validates skill-related config and capabilities at the planning stage.

---

## 2. Planning in the Runtime Pipeline

The complete execution pipeline shows where planning operates:

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
Planning ──── resolve capabilities, distribute per-skill capabilities
    │
    ▼
build_state(config) ─── creates PyRAIState with task queue
────────────────────────────────────────────
Framework / Skill Boundary
────────────────────────────────────────────
PyRAIState stores task descriptors (no backend instances)
    │
    ├── task['api'] = 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend'
    ├── task['config'] = {'skill_config': {...}, 'results': {}}   # plain dict
    ├── task['static_id'] = 'tmp_PYSCF'          # single-task format
    ├── task['task_id'] / task['layer_id']        # multi-task identity
    ├── task['multitask'] / task['remotetask'] / task['multipool']  # dispatch flags
    └── task['capabilities'] = ['energy', 'gradient']
    │
    ▼
execute(state) ─── unified entry point
    │
    ├── Creates Calculator(state)
    ├── Calculator.initialize() → resolve_api() → instantiate backends
    ├── resolve_workflow(state) → Workflow class or None
    ├── If iterative: Workflow(calculator=calc).run()
    └── If single-shot: calculator.run()
    │
    ▼
Result
```

The boundary is defined at PyRAIState:

- Planning terminates after PyRAIState construction, distributing per-skill capabilities via Maestro.
- PyRAIState stores only task descriptors (API paths as strings, configs, static IDs); no backend instances exist yet.
- Backend instantiation is deferred to `Calculator.initialize()` (see [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md) Section 3.6).
- PyRAIState is the only communication protocol between the Framework and every Skill.

### 2.1 Detailed Runtime Steps

```
Step 0: Environment.initialize() → Maestro.populate()
    ↓
Step 1: Interface — parse_input() (parse + coord + validate)
    ↓
Step 2: Adapter — build_config(raw_config) → Config
    ↓
Step 3: Normalize — normalize(config)
    ↓
Step 4: Validate — validate(config)
    ↓
Step 5: Build PyRAIState — build_state(config)
    ↓   PyRAIState (task queue with API paths, jobtype extracted)
    ↓   maestro.resolve_skills(config) → skills
    ↓   per-skill: construct task descriptors (api path + config + static_id)
    ↓   NO backend instantiation
    ↓
Step 6: Execute — execute(state)
    → creates Calculator(state)
    → Calculator.initialize() → resolve_api() → instantiate backends
    → resolve_workflow(state) → Workflow class or None
    → returns Workflow instance (iterative) or calculator.run() (single-shot)
    → runner.run() → Calculator.run() → backend.calculate() → Result
```

This runtime model no longer decides basic vs. advanced workflow — that is resolved by planning.
Per-skill capabilities flow through Maestro.resolve_skills() into build_state() and end up in state.theory/reference.

### 2.2 Execution Driver

`control.driver` selects the execution model at planning time. Valid values are `main`, `skill`, or any discovered skill name; `_validate_driver()` (`planning/assembly.py`) checks non-main drivers against `skills/query.discover_all()`.

- `driver = main` — the framework drives the job: `resolve_workflow(state)` maps the jobtype to a workflow class (or `None` for single-shot).
- `driver = skill` or a named skill — execution is handed to the skill itself: `resolve_workflow(state)` returns `SingleShot` regardless of the jobtype, and the jobtype string is passed to the skill untouched. Additional constraints:
  - `theory` must contain exactly one token.
  - A named driver must match `theory[0]` (the driver name and the method agree).

The driver is frozen at planning time into `PyRAIState.driver`; `resolve_workflow()` (`workflow/__init__.py`) is driver-aware and never re-decides the execution model.

---

## 3. Source Structure

```
pyrai3md/

├── planning/                  # Planning layer (WHAT to compute)
│   ├── __init__.py            #   Exports resolve_capabilities, CAPABILITY_NAMES,
│   │                          #   resolve_task_queue, resolve_composite_type
│   ├── resolver.py            #   Token → capability name strings + multi-backend constraints + dynamics derivation
│   └── task.py                #   Composite detection + Task Queue expansion (ONIOM/fragment/qmmm)
│
├── data/                      # Data core (runtime state)
│   ├── core.py                #   PyRAIState (Task Queue) + build_state (build_pyrai_config deprecated)
│   ├── center.py              #   Center plan: ElectronicSpec, classify_pairs
│   ├── charges.py / geometry.py / topology.py / constraint.py
│   │                          #   embedding charges, PBC geometry, topology,
│   │                          #   constraint records
│   └── runtime/               #   RuntimeState, Result, serializer
│
└── workflow/                  # Workflow execution layer (HOW to compute)
    └── ...
```

---

## 4. Capability Names

Planning resolves jobtype tokens into **capability name strings**. No intermediate objects are created.

### 4.1 Available Capabilities

| jobtype token      | capability name       | skill return format         |
|--------------------|-----------------------|-----------------------------|
| energy             | `'energy'`            | float (Hartree)             |
| gradient           | `'gradient'`          | ndarray (natom, 3)          |
| nac                | `'nac'`               | ndarray (nnac, natom, 3)    |
| soc                | `'soc'`               | ndarray (nsoc,)             |
| frequency          | `'hessian'`           | ndarray (3*natom, 3*natom)  |
| absorption         | `'transition_dipole'` | ndarray (nstate, 3)         |
| dipole             | `'dipole'`            | ndarray (3,)                |
| transition         | `'transition_dipole'` | ndarray (nstate, 3)         |

### 4.2 Capability Dependencies

Some capabilities implicitly require additional electronic information:

- `nac` requires the center's `nac_pairs`, classified from the coupling and state
  manifold by the one derivation `data/center.py::derive_pair_declaration`
  (`data/center.py:356`, which calls `classify_pairs` at `:299`) on the
  `ElectronicSpec` of the center plan (`data/center.py:139`). The flat single-center
  path (`schema/normalize.py:237`) and the per-center plan both call it, so the
  declaration cannot be derived two ways.
- Who asks for that capability is the optimizer's declaration, not the method name:
  `uses_derivative_coupling` (`workflow/optimizer/meci.py:80`) is what makes a
  main-driver `optimize` carry the `nac` capability at all. Since #447 the ask is
  made **here** rather than at the jobtype-spelling site: `optimize.opt_method`,
  `opt_state_i`/`opt_state_j` are scanned with every other requirement keyword into
  the planning context (`kernel/maestro.py::_extract_planning_context`, table
  `planning/resolver.py::_REQUIREMENT_KEYWORDS`), and
  `crossing_coupling_pair` + `_derive_from_requests` add `nac` inside
  `resolve_capabilities` - the sole expander. A need read only by whichever layer
  built the task is invisible to capability distribution, to the multi-backend ban
  and to the declaration check, which is how the first version of this rule came to
  sit in `planning/assembly.py` while the plan still spelled `energy gradient` (#33, #446).
- What the pair means is the declaration, not the chase: `opt_state_i`/`opt_state_j`
  say which two surfaces a crossing search follows, and `molecule.coupling` says
  which couplings are computed. `planning/assembly.py::_task_property_request` therefore
  *selects* the declared pair matching the chased one and refuses the run when it is
  not declared (`_declared_pair`); a backend that offers the channel but has no
  declaration behind the request is refused by `_require_declared_couplings`. A pair
  nobody declared is shielded, so nothing is asked of the engine behind the
  declaration and no row arrives as a fabricated zero.
- A backend whose manifest declares no coupling never sees the token
  (`planning/assembly.py::_declares_a_channel`, which reads both spellings: `nac` /
  `nonadiabatic_coupling`, `soc` / `soc_coupling`).
- The same request now carries the state a leg embeds with for embedding charges:
  `Calculator._charge_embedding_state` (named `_qmmm_embedding_state` before #450) reads
  the step's `target_states` and indexes the backend's `(nstate, natom)` charge rows with
  it; the QM/MM stage and the ONIOM charge scatter both call it.
- `soc` requires the same pass's `soc_pairs`.
- `transition_dipole` requires the transition pair list of the same spec.
The pairs reach the terms as `task['property_request_spec']`
(`workflow/calculator.py:883`); the former `ElectronicState` object no longer exists.
- What the skill side is handed is that same declaration. The state-key overlay in
  `schema/adapter.py` (`_SKILL_STATE_KEYS`, line 3030) re-asserts `nstate`,
  `state_spin`, `target_states` and `target_nac_pair` after the framework's own
  normalization - the keys it used to overwrite with framework defaults - and
  `coupling`, `nac_pair_map` and `soc_pair_map` reach the skill through the
  ordinary section merge. Measured on the loading chain (`parse_test_yaml ->
  convert -> parse_input -> normalize -> validate -> build_state`) on this
  checkout: a framework run arrives at the skill with all three already filled,
  which is why #448 changed no key list here - the defect was on the receiving
  side, where five skills overwrote what they had been given. The consequence for
  the planner is that the framework's numbering is the only one a deck may be
  built against; a skill that re-derives or re-orders it is refused, per
  `dev/MULTICENTER_DEVELOPMENT.md` 4.4 item 7.

Dependency relationships (e.g. gradient requires energy) are managed via string mapping in the resolver, not via objects:

```python
_DEPENDENCIES = {
    'gradient': ['energy'],
    'hessian': ['energy', 'gradient'],
    'nac': ['energy'],
    'soc': ['energy'],
    'overlap': ['energy'],
    'dipole': ['energy'],
    'transition_dipole': ['energy'],
}
```

---

## 5. Capability Resolution

`resolver.py` transforms jobtype tokens into capability name strings.

### 5.1 Token-to-Capability Mapping

```python
_TOKEN_CAPABILITY_MAP = {
    'energy': 'energy',
    'gradient': 'gradient',
    'nac': 'nac',
    'soc': 'soc',
    'frequency': 'hessian',
    # The analytical matrix only: the same 'hessian' capability, so a
    # multi-backend run is refused here exactly as it is for 'frequency'.
    'hessian': 'hessian',
    # The numerical route asks for no second derivative from anyone: it
    # differentiates the gradients it is already entitled to.
    'num_freq': 'gradient',
    'num_hess': 'gradient',
    'absorption': 'transition_dipole',
    'dipole': 'dipole',
    'transition': 'transition_dipole',
    # 'stress' is not a user-facing jobtype token: it is derived from the
    # ensemble (NPT) / optimize_cell switches by _derive_stress. The entry
    # keeps an explicit token resolvable, but it is deliberately absent
    # from CAPABILITY_TO_TOKEN so it never changes a skill's jobtype
    # spelling.
    'stress': 'stress',
}
```
(`planning/resolver.py:21-46`; the requirement keywords that feed it are listed by
section at `planning/resolver.py:68`)

### 5.2 Advanced Jobtype Expansion

Advanced jobtypes are expanded to their required basic properties:

```python
_ADVANCED_DEFAULTS = {
    'optimize': ['energy', 'gradient'],
    'md': ['energy', 'gradient'],
    'adaptive': ['energy', 'gradient'],
    'search': ['energy', 'gradient'],
}
```

`train` and `prediction` are intentionally not expanded: ML tasks are dispatched through the skill's own API rather than the framework capability tokens.

`surface_hopping` is accepted as an advanced jobtype (`ADVANCED_JOBTYPES` in `schema/validate.py`) and maps to the AIMD workflow in `WORKFLOW_REGISTRY` / `JOBTYPE_MAP` (`workflow/__init__.py`).

### 5.3 Resolution Logic

`resolve_capabilities(tokens, dynamics_config=None, skills=None)` performs:

1. Iterate over input tokens.
2. If a token matches an advanced jobtype, expand it to basic capability tokens.
3. Map each basic token to its capability name string via `_TOKEN_CAPABILITY_MAP`.
4. Add dynamics-derived capabilities via `_derive_from_dynamics(dynamics_config)`:
   - `sfhp=fssh` + `nactype=dcm` → `overlap`
   - `sfhp=fssh` + `nactype=nac` → `nac`
   - `dosoc=true` → `soc`
   - `dodipole=true` → `dipole`
   - `radiative=true` → `transition_dipole`
4c. Add the needs the other workflow sections state, via
   `_derive_from_requests(planning_context)`: a main-driver `optimize` whose method
   declares `uses_derivative_coupling` and chases two different states → `nac`
   (`crossing_coupling_pair`). The keywords this step reads are listed by section in
   `_REQUIREMENT_KEYWORDS`, and `test_state_declaration.py` asserts every one of them
   arrives in the planning context - that table is what keeps a need added by a
   future workflow section from being discovered only at task-build time.
5. Resolve dependencies via `_DEPENDENCIES` (e.g. gradient → add energy;
   `stress` asks `energy`, `resolver.py:55`).
6. Derive the lattice ask via `_derive_stress` (`resolver.py:291-333`): `md` with an
   active barostat, or `optimize` with `optimize_cell: true`, appends `stress` - which
   is what makes the engine compute a lattice derivative at all.
7. Deduplicate.

### 5.4 Skill Jobtype Spelling

The framework translates planning capabilities back to a skill jobtype string via `spell_skill_jobtype()` (in `data/core.py`). The six-capability jobtype spelling is based on two lookup tables:

- `JOBTYPE_TOKEN_ORDER`: fixed canonical order for skill jobtype tokens (`energy, gradient, nac, soc, dipole, transition, frequency, absorption`).
- `CAPABILITY_TO_TOKEN`: reverse mapping from capability name to jobtype token (`transition_dipole → transition`, `hessian → frequency`).

The function receives per-skill capabilities from planning and produces a canonical jobtype string spelling the capability tokens in the canonical order. When capabilities are unavailable, the fallback jobtype (e.g. `state.jobtype`) is returned as-is. Note: the driver=main + optimize forcing (`'energy gradient'` + `OPTIMIZER_REGISTRY` validation) does **not** belong to `spell_skill_jobtype()`; it is applied in `_build_task_queue()` (`data/core.py`).

### 5.5 Capability Derivation Design

The planning derivation follows these rules:

| Condition | Capability |
|-----------|-----------|
| `jobtype=[energy, sp]` | `[energy]` |
| `jobtype=[gradient, grad, engrad, optimize]` | `[energy, gradient]` |
| `jobtype=[num_hess, num_freq]` | `[energy, gradient]` (the framework differentiates
  them itself, so no second derivative is asked of a backend; this is what keeps a
  multi-backend run able to use the numerical route) |
| `jobtype=[hessian, hess, frequency, freq]` | `[energy, gradient, hessian]` |
| `jobtype=[nac]` | `[energy, nac]` |
| `jobtype=[soc]` | `[energy, soc]` |
| `jobtype=[dipole]` | `[energy, dipole]` |
| `jobtype=[transition]` | `[energy, transition_dipole]` |
| `jobtype=md` | `[energy, gradient]`, then depends on `&md` section |

Dynamics-dependent additions:

| Condition | Capability addition |
|-----------|---------------------|
| `sfhp=none/no/nohop/lzsh/znsh/gsh` | no change |
| `sfhp=fssh`, `nactype=ktdc` | no change |
| `sfhp=fssh`, `nactype=dcm` | add `overlap` |
| `sfhp=fssh`, `nactype=nac` | add `nac` |
| `dosoc=true` | add `soc` |
| `dodipole=true` | add `dipole` |
| `radiative=true` | add `transition_dipole` |

When NAC and/or SOC is activated in capability, the pair information is resolved from
the center plan's `ElectronicSpec` and handed to the term through
`property_request_spec`.

### 5.6 Multi-Skill Backend Capability Constraints

When multiple theory or reference methods are configured (≥2 theory or ≥2 reference), planning must apply additional constraints to capability resolution.

#### Constraint 1: Hessian Forbidden with Multiple Backends
When `n_theory >= 2` or `n_reference >= 2`, `jobtype=hessian/frequency` is **forbidden**. Multi-backend Hessian computation produces incorrect results because the Hessian requires a consistent electronic structure across all atoms, which cannot be guaranteed when split across different methods.

```
theory = [molcas, xtb], jobtype = frequency  → ERROR (forbidden)
theory = [orca], reference = [xtb, lammps], jobtype = hess  → ERROR (forbidden, n_reference >= 2)
theory = [orca], jobtype = frequency  → OK (single theory)
```

Planning must raise an error during capability resolution, not defer to backend execution.

#### Constraint 2: MD Advanced Properties Scoped to First Backend Pair

When `n_theory >= 2` or `n_reference >= 2` in MD workflows, advanced properties (beyond energy and gradient) apply **only** to the first theory + first reference backend pair. All other backends are restricted to `[energy, gradient]`.

```
theory = [molcas, xtb], jobtype = md, sfhp = fssh

First theory (molcas):  [energy, gradient, nac, soc]   ← full capability
Second theory (xtb):    [energy, gradient]              ← restricted
```

Advanced properties subject to this scoping:

- `nac`
- `soc`
- `overlap`
- `dipole`
- `transition_dipole`
- `hessian` (if allowed by Constraint 1)

Properties NOT subject to scoping (all backends compute them):

- `energy`
- `gradient`

Planning must generate per-backend capability lists and store them in each task's dict:

```
# Per-task capabilities are stored in each task's dict:
state.theory[0]['capabilities'] = ['energy', 'gradient', 'nac', 'soc']   # primary task
state.theory[1]['capabilities'] = ['energy', 'gradient']                 # restricted secondary task
```

### 5.7 The Adaptive Reference Plan Is Expanded on Its Own

`jobtype: adaptive` gives `control.reference` a second consumer: it is the
labelling queue, and planning treats it as a layered plan in its own right
rather than as a positional mirror of `control.theory`.

```text
resolve_reference_type(config, skills)   # 1 -> None (single), 2 -> oniom2/qmmm
                                         #     (control.multiscale_type decides),
                                         #     3 -> oniom3
_reference_regions(config, ...)          # molecule.highref/midref, inheriting
                                         #     highlevel/midlevel only when the
                                         #     theory queue is layered
_expand_reference_plan(...)              # the same region-based expander as a
                                         #     theory composite, over the ref roles
resolve_label_plan(...)                  # the label channel each reference
                                         #     task publishes
```

Two consequences belong to planning, not to the workflow. The label source is a
set comparison against the theory electronic provider's region - exactly one
reference layer covering it yields `source: layer` with that task id, no single
layer covering it while the composite spans the system yields `source: total`,
and anything else is refused with the reference layers named - and it travels as
the plain `label_plan` dict on every reference task so a remote result carries
it unchanged. Composite reference tasks also keep their own directory names:
`_compose_static_id` prefixes `ref_` onto the usual `tmp_{region}_{role}_{SKILL}`
form, because a reference high layer and a theory high layer are the same skill
in the same role over the same region, and one shared `static_id` would put both
stages in one calculation directory.

A three-layer reference whose mid task is the matching region gets its capability
set widened to the run's requested channels, so a mid-surface fit does not lose
`nac`/`soc` silently.

---

## 6. Planning Output and the Skill Backend Boundary

Planning's output is distributed as per-task capabilities in the `state.theory/reference` task queues. From this point, the Skill backend and the Workflow layer take over independently.

### 6.1 Planning → state.theory/reference

Planning resolves capabilities via `Maestro.resolve_skills(config)`, which returns a `skills` dict. `build_state()` then expands the task queue and stores it in state.theory/reference:

```
Maestro.resolve_skills(config) resolves:
    jobtype tokens → capability name strings
    planning context (sfhp, nactype, dosoc, dodipole, radiative, ensemble,
                      optimize_cell, opt_method/opt_state_i/opt_state_j,
                      hess_driver, the state declaration) → additional capabilities

Returns:
    skills = {'theory': {...}, 'reference': {...}}
    Each skill dict contains 'capabilities' key with per-skill capability list

build_state(config):
    resolve_task_queue(config, skills) → task descriptors (composite expansion)
    For each task: build plain dict task config
        ({'skill_config': flat, 'results': {}}) → state.add_task()
    state.theory/reference = [ {role, task, task_id, layer_id, static_id,
                                contribution, region, region_indices,
                                skill_name, api, capabilities, config,
                                multitask, remotetask, multipool,
                                [label_plan (adaptive reference tasks only)]}, ... ]
```

Theory/reference method names are resolved against a dynamically discovered whitelist (#150): the builtin methods `{'nn', 'e2n2'}` ∪ the methods reported by `skills/query.discover_all()`, cached after first computation, with `FALLBACK_THEORY_METHODS` as the fallback when discovery yields nothing (`qm.` / `abinit.` prefixes are stripped before matching).

Planning terminates here. It never builds CalculationRequest, never instantiates backends, and never enters workflow logic.

### 6.2 Skill Backend Instantiation (Skill-internal)

When the Skill backend is instantiated with PyRAIConfig:

```
backend = SkillBackend(pyrai_config)
    │
    ▼
Read PyRAIConfig → extract capabilities and method configuration
    │
    ▼
Build CalculationRequest (skill-internal concept)
    │
    ▼
Compile execution profile → fixed for all subsequent calls
```

`CalculationRequest` is a **skill-internal** concept. The Skill backend reads PyRAIConfig to determine what to compute, how to compute it, and fixes the execution logic during `__init__()`. Subsequent `backend.calculate()` calls use this pre-compiled profile.

### 6.3 Three-Layer Isolation

```
Planning Layer (WHAT to compute, via Maestro)
    │
    │  resolve_skills() → skills with per-skill capabilities
    │  build_state() → register to state
    │
    ▼
Skill Backend (HOW to execute)
    │
    │  reads PyRAIConfig → builds CalculationRequest internally
    │  compiles execution profile → fixed for subsequent calls
    │
    ▼
Workflow Layer (HOW to orchestrate)
    │
    │  execute(state) → creates Calculator(state), resolve_workflow(state)
    │  returns Workflow instance or calculator.run()
    │  runner.run() → Calculator.run() → backend.calculate() → Result → workflow
    │
    ▼
Result collected, PyRAIState updated
```

The three layers are **completely isolated**:

- Planning never knows about Skill execution or Workflow orchestration.
- Skill backend never knows about Workflow orchestration or Planning resolution.
- Workflow never rebuilds CalculationRequest or resolves capabilities. It reads `state.jobtype` and backend capabilities to select the appropriate workflow function.

---

## 7. Relationship with PyRAIState

The `data/core.py` module provides `PyRAIState` and `build_state` (the former `build_pyrai_config` helper is deprecated).

### 7.1 PyRAIState Structure

```python
class PyRAIState:
    jobtype          # Global job type (read-only property)
    driver           # Execution driver ('main' / 'skill' / skill name)
    pmd              # PyRAIConfig (single data source, read-only property access)
    theory           # task queue: list of {role, task, task_id, layer_id,
                     #   static_id, contribution, region, region_indices,
                     #   skill_name, api, capabilities, config (plain dict),
                     #   multitask, remotetask, multipool}
    reference        # same structure as theory
    runtime          # RuntimeState (step metadata, kinetic energy, etc.)
    execution_graph  # frozen execution graph (or None)
```

### 7.2 Planning Distributes Per-Task Capabilities

The planning layer resolves capabilities and distributes them to each task's dict. The PyRAIState structure is aligned with the task queue:

```
PyRAIState fields (from build_state):
    theory = [{'role': 'qm', 'task': 0, 'task_id': 'task0', 'layer_id': 0,
               'contribution': 1, 'capabilities': [...],
               'config': {'skill_config': {...}, 'results': {}}, ...}]
    reference = [...]
```

PyRAIState wraps the task metadata. Planning output belongs in each task's `capabilities` list because it describes the computational task — what to compute — which is a method-level concern.

### 7.3 Separation of Concerns

```
Method describes calculation requirements.

Skill implements calculation execution.
```

Example:

```
state.theory[0] = {'role': 'qm', 'task': 0, 'skill_name': 'molcas_skill',
                   'api': MolcasBackend, 'capabilities': [...],
                   'config': {'skill_config': {...}, 'results': {}}, ...}
state.jobtype = 'md'

skill.molcas executes Molcas calculation.
```

Planning determines WHAT capabilities are needed. Per-task capabilities store the resolved task. PyRAIState fields (theory, reference) store the computational approach. Skill implements the actual execution.

### 7.4 Task Queue Expansion (planning/task.py)

Planning also decides **how many backend calls** a calculation needs. `planning/task.py`
provides:

- `resolve_composite_type(config, skills)` — detects the composite scheme from the
  backend combination and system keywords:
  - `system.center_groups` non-empty → `'fragment'`
  - 2 theory methods + `multiscale_type='qmmm'` → `'qmmm'` for a flat
    `highlevel`, or `'multi_center_qmmm'` when it is written one region per
    center. The two identifiers are kept apart so a single-center run's task
    ids and calculation directories do not move when the nested spelling
    exists.
  - 3 theory methods → `'oniom3'`, 2 theory methods (empty/`'oniom'`) → `'oniom2'`
  - otherwise → `None` (single-shot)
  - `system.qmmm_key` no longer selects QMMM; it is refused with the
    `multiscale_type qmmm` spelling, and theory+reference no longer implies QMMM.
- `resolve_task_queue(config, skills)` — expands the resolved skills into ordered task
  descriptors (`{'theory': [...], 'reference': [...]}`):

```
single:    qm → task0
QMMM(emb T): field  → task0 (all,        0)   # LAMMPS qmmm_field V/E (or MM charges)
             qm     → task1 (model=highlevel, +1)   # Molcas ESPF external field
             energy → task2 (all,        +1)   # LAMMPS qmmm_energy E_classic
QMMM(emb F): qm     → task0 (model=highlevel, +1)
             energy → task1 (all,        +1)   # E_classic keeps the QM-MM electrostatics
QMMM(C>1):   field xC (probe=center c, 0) → qm_charge xC (0, only with sc_charge)
             → qm xC (model=H_c, +1) → energy x1 (all, +1)   # qm_centers = the tiling

ONIOM2:    qm  → task0 (model, +1)
           qm2 → task0 (real,  +1)
           qm2 → task1 (model, -1)
ONIOM3:    five tasks with contributions [+1, +1, -1, +1, -1]
fragment:  qm → task0..taskN (one per fragment, all +1)
```

Each descriptor carries `region` and `region_indices` (atom slice) but never sliced
geometry. Under several centers the QMMM duties add `probe_indices` (the center a
field stage measures) and `partition_indices` (the union of the centers, in global
atom order, which the classical stage addresses its charges and its Q-row
corrections by); both travel through `data/core.py`'s task whitelist, and a dropped
`probe_indices` silently makes every center measure the whole QM set - which is what
the shipped-input test `qualification/verification/test_multicenter_qmmm.py` reads.
A `qm_charge` duty asks `['energy']` only, so the skill spells an energy-only jobtype
and the deck keeps its `&ESPF` block without `&ALASKA`; it declares
`provides_charges` and never `provides_electronic`, so it stays invisible to the
electronic collection.
The QMMM field/energy tasks carry the `field`/`qmmm` capability tokens:
`data/core.py::spell_skill_jobtype` spells the skill jobtype (`energy gradient field`
/ `energy gradient qmmm`) and the skill resolves its own profile; the framework never
writes a skill profile name. `system.read_charge` is an ONIOM2/ONIOM3 keyword and is
ignored for QMMM. For a multi-center run (C>1) `data/core.py._build_task_queue` additionally fills each
task's `region_charge`/`region_multiplicity` from the bound center plan (L(real) → the
aggregated `system_charge`/`system_multiplicity`; H(c)/L(c)/mid → center c's
`charge`/`multiplicity`) and overrides the task's `skill_config` `charges`/`multiplicity`
with that region scalar; `planning/task.py._new_task` seeds both keys as `None`.
`build_state()` (data/core.py) consumes these descriptors to build one plain dict
task config (`{'skill_config': ..., 'results': {}}`) per task.
Backend instances are created later by `Calculator.initialize()`.

---

## 8. Planning Constraints

Planning **never**:

- Executes calculations
- Imports Skills
- Instantiates backends
- Calls `backend.calculate()`
- Queries Maestro at runtime
- Imports the workflow layer
- Performs workflow logic

Planning ends when per-skill capabilities have been distributed to each skill's dict. After that:

- The Skill backend reads PyRAIConfig to build its internal CalculationRequest and compile execution profile.
- The Workflow layer reads `state.jobtype` and backend capabilities to select the appropriate workflow function.
- Planning never participates in either process.

---

## 9. Import Isolation

`planning/` and `workflow/` have **zero imports** between them. Communication happens through `state.theory/reference` (skill info + PyRAIConfig) and `state.jobtype`.

Planning resolves tokens to capability name strings via Maestro, stored in state.theory/reference by `build_state()`. The Skill backend reads PyRAIConfig during instantiation. The Workflow layer reads `state.jobtype` and backend capabilities to select the workflow function.

```
Maestro.resolve_skills() ──── resolves ────→ per-skill capabilities
    │                                          │
    │  build_state                             │
    ▼                                          ▼
state.theory/reference (per-skill capabilities)
    │
    └───────────────┬───────────────┐
    ▼                               ▼
Skill backend                   Workflow layer
(reads capabilities,            (reads state.jobtype
 builds CalculationRequest       + backend capabilities,
 internally)                     selects workflow function)
```

---

## 10. Skill Capability Model

A skill provides computational capabilities. The workflow does not directly know program details. Capability matching is handled by the planning layer (resolver).

Example:

```yaml
BAGEL:
    capabilities:
        energy: true
        gradient: true
        nac: true
        soc: true
        dipole: true
        transition_dipole: true

ORCA:
    capabilities:
        energy: true
        gradient: true
        nac: false
        soc: true
```

### 10.1 Property Handling Variations

Different electronic-structure programs may provide different outputs:

**Case 1**: Energy calculation automatically provides dipole — one calculation, both results.

**Case 2**: Dipole requires additional keywords — the skill adapter modifies input generation.

**Case 3**: Property requires additional calculation — the skill internally handles it. The upper layer always sees `backend.calculate(request)`.

---

## 11. Design Rules

### Rule 1: Planning determines WHAT; Workflow determines HOW

Planning determines what should be computed. Workflow determines how the already planned computation is executed.

### Rule 2: Planning terminates at PyRAIState

PyRAIState is the final product of the Framework planning stage. Planning distributes per-skill capabilities via Maestro.

### Rule 3: PyRAIState is the only communication protocol

Skills never receive raw input dictionaries. Skills always receive PyRAIState.

### Rule 4: Skill initialization owns Skill-specific preparation

The Framework constructs PyRAIState. The Skill constructor performs adapter, normalize, validate, and execution profile compilation automatically.

### Rule 5: Workflow operates only on instantiated Backend objects

Workflow never performs planning. Workflow never queries Maestro. Workflow only orchestrates backend execution and updates PyRAIState.

### Rule 6: Workflow cannot directly call external programs

```
Wrong:
    AIMD → BAGEL

Correct:
    AIMD → Calculator → Skill
```

### Rule 7: Capability names are plain strings

```
Wrong:
    Gradient.compute() → BAGEL

Correct:
    'gradient' → state.theory/reference → backend.calculate()
```

Planning resolves capability names as plain strings. No intermediate objects are created.

### Rule 8: One calculation step generates one skill request

```
Wrong:
    energy() → gradient() → soc()  (three separate calls)

Correct:
    SinglePoint one request: energy + gradient + soc (one call)
```

### Rule 9: Skill only provides capability

Skill is responsible for: input generation, execution, output parsing.

Skill is NOT responsible for: optimization loop, AIMD propagation, ONIOM combination, workflow control.

Note: the skill-section `remotetask` keyword is ignored for the *base node* (universal
rule 3 pins that node local), but it is not discarded:
`workflow/dispatch/planner.py::resolve_molecule_flags` aggregates the skills' tri-state
`remotetask` for the finer-grained `backend_terms` scope, where an individual term may
run as a remote worker.

### Rule 10: Import isolation

`planning/` and `workflow/` have zero imports between them. The optimizer
declarations planning needs are pure data in `schema/optimizer_spec.py`, and the
fix mask is in `data/constraint.py`. Communication happens through
`state.theory/reference` (skill info + PyRAIConfig) and `state.jobtype`.

### Rule 11: Multi-backend capability constraints

When multiple theory or reference methods are configured:

- Hessian/frequency is forbidden (produces incorrect results).
- MD advanced properties (nac, soc, overlap, etc.) apply only to the first theory + first reference pair. Other backends are restricted to energy + gradient.

Planning must enforce these constraints during capability resolution, not defer to backend execution.

---

## 12. Summary

PyRAI3MD planning architecture:

```
User Input (scientific task)
    │
    ▼
Maestro.resolve_skills(config)
    ├── resolve_skill() → theory + reference backends
    ├── resolve_capabilities(jobtype_tokens) → capability name strings
    ├── derive additional capabilities from dynamics keywords
    └── distribute per-backend capabilities (backend_capabilities)
    │
    ▼
build_state(config)
    │
    ▼
PyRAIState.theory/reference (per-skill capabilities in skill info dicts)
    │
    ▼
Framework / Skill Boundary
    │
    ▼
Skill Backend Instantiation (skill-internal)
    ├── read PyRAIConfig → build CalculationRequest
    └── compile execution profile (fixed for subsequent calls)
    │
    ▼
execute(state)
    ├── Creates Calculator(state)
    ├── resolve_workflow(state) → Workflow class or None
    ├── If iterative: Workflow(calculator=calc).run()
    └── If single-shot: calculator.run()
```

This separation enables:

- Users describe scientific intent; planning derives computational requirements
- Per-skill capabilities stored in state.theory/reference as the single source of truth
- Skill backend owns its execution logic (CalculationRequest is skill-internal)
- Workflow owns orchestration logic (never rebuilds calculation requests)
- Three-layer isolation: planning, workflow, and skill backend are completely independent
- Efficient external program usage (one request, one call)
- Multiple backend support and ONIOM/QM-MM extension
- Complete backend independence
- AI-driven workflow planning
- Future periodic and ML potential integration
