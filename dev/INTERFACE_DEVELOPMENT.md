# PyRAI3MD Interface and Entry Architecture Design

Last updated: 2026-10-22 22:20 (validate is now the nine checks it actually runs, the config model is the eleven nested submodels plus `results`, the flat-routing sentence matches `FIELD_TO_SUBMODEL`, and the maestro parameters the layer table credited normalize/validate with are removed - both take the config only and discover skills through `skills/query`. The rest was not re-audited. Before it, 2026-10-01)

---

## 1. Overview

PyRAI3MD is designed as an extensible computational platform that supports both traditional quantum chemistry workflow execution and future agentic scientific computing.

To support different user interaction modes while maintaining a unified internal execution framework, PyRAI3MD separates:

- User entry points
- Input/configuration conversion
- Type conversion and configuration building
- Parameter normalization and validation
- Capability discovery
- Runtime simulation state
- Workflow execution
- Computational skills

The central design principle is:

> All external representations must eventually be converted into a unified `PyRAIState`, which serves as the only runtime state representation for PyRAI3MD.

---

# 2. Scope

The complete project directory structure is defined in [ARCHITECTURE.md](ARCHITECTURE.md).

This document focuses on the interface-related modules and the schema pipeline:

- **interface/** — Input conversion layer (parser, loader)
- **schema/** — Configuration building and validation (adapter, normalize, validate)
- **Entry points** — pmd.py and agent.py orchestration

---

# 3. Entry Point Design

PyRAI3MD provides two independent, parallel entry points:

1. `pmd.py`
   - Traditional file-based computational workflow
   - Uses `interface.parser` for input parsing

2. `agent.py`
   - Agentic task-driven workflow
   - Uses `interface.loader` for config loading

Both share the same orchestration pipeline:

```text
Interface (parse + coord + validate)
    → Adapter (build_config: raw dict → Config)
    → Normalize (decoupled from maestro)
    → Validate (decoupled from maestro)
    → Build PyRAIState (build_state: static task descriptors)
    → Execute (execute: Calculator + Workflow)
```

Key design: `build_state(config)` creates PyRAIState, resolves skills via Maestro and registers **static task descriptors** (skill name, api class, capabilities, plain dict task config) to the state queues. Backend instantiation is deferred to `Calculator.initialize()`.

After construction of `PyRAIState`, the execution path is fully unified.

---

# 4. Traditional Entry: pmd.py

## 4.1 Purpose

`pmd.py` is the traditional command-line entry point of PyRAI3MD.

Example:

```bash
python -m dev.PyRAI3MD input
python -m dev.PyRAI3MD input --info
```

---

## 4.2 Responsibilities

`pmd.py` handles two roles:

1. **CLI entry function** — prints logo, processes `sys.argv`, dispatches to `run()`
2. **Orchestration pipeline** — defines the full execution flow:
   - `parse_input()` — Interface: parse + load coordinates + validate structure
   - `build_config()` — Adapter: raw dict → Config (includes Environment init)
   - `normalize()` — Schema: compute derived quantities (decoupled from maestro)
   - `validate()` — Schema: consistency checks (decoupled from maestro)
   - `build_state()` — Build PyRAIState (static task descriptors registered; backend instantiation deferred to `Calculator.initialize()`)
   - `execute(state)` — Workflow execution (creates Calculator, returns Workflow or calculator.run())

---

## 4.3 Non-responsibilities

`pmd.py` does NOT handle:

- Input format parsing (delegated to `interface.parser`)
- Coordinate file loading (delegated to `interface.parser`)
- Structure validation (delegated to `interface.parser`)
- Type conversion (delegated to `schema.adapter`)
- Skill discovery and resolution (delegated to `kernel.maestro`)
- Skill config building (delegated to `schema.adapter`)
- Skill-specific parameter validation (delegated to each Skill)
- Computational method implementation

---

## 4.4 Execution Flow

```text
pmd.py
    │
    ↓
kernel.environment.Environment().initialize()
    │   Load package + user pmd_env.yaml
    │   Maestro.populate() via query mechanism
    │
    ↓
interface.parser.parse_input()
    │   Parse text/JSON → raw dict
    │   Load coordinate file (if external)
    │   Validate structure (atoms + coordinates resolved)
    │
    ↓
Raw Configuration (dict, values as raw strings, structure validated)
    │
    ↓
schema.adapter.build_config(raw_config)
    │   Initialize Environment + build Config
    │   Returns config = Config
    │
    ↓
schema.normalize.normalize(config)
    │   Compute derived quantities (decoupled from maestro)
    │
    ↓
schema.validate.validate(config)
    │   Full consistency checks (decoupled from maestro)
    │
    ↓
data.core.build_state(config)
    │   PyRAIState construction (static task descriptors)
    │   maestro.resolve_skills(config) → skills
    │   resolve_task_queue: register static task descriptors
    │   (skill_name / api class / capabilities / plain dict config)
    │   backend instantiation is deferred to Calculator.initialize()
    │
    ↓
workflow.execute(state) — Workflow execution
    │   Creates Calculator(state), resolve_workflow(state)
    │   Returns Workflow instance or calculator.run()
    │
    ↓
Skill Calculation
```

---

# 5. Agent Entry: agent.py

## 5.1 Purpose

`agent.py` is the entry point for agentic scientific workflows.

Example:

```bash
python -m dev.PyRAI3MD.agent config.json
```

---

## 5.2 Responsibilities

`agent.py` provides the `PyRAI3Agent` class:

- `from_file(config_file)` — Create agent from JSON config via `interface.loader`
- `from_dict(config)` — Create agent from dictionary
- `initialize()` — Run the full pipeline (see flow below)
- `run()` — Execute the task and return results

The `initialize()` pipeline:
```
Adapter (build_config: raw dict → Config)
→ Normalize (decoupled from maestro)
→ Validate (decoupled from maestro)
→ Build PyRAIState (build_state: static task descriptors)
```

Orchestration function `execute` is **imported from `workflow/__init__.py`**, avoiding code duplication.

---

## 5.3 Execution Flow

```text
agent.py (PyRAI3Agent)
    │
    ↓
interface.loader.load_config()
    │
    ↓
Raw Configuration (dict)
    │
    ↓
kernel.environment.Environment().initialize()
    │   Load package + user pmd_env.yaml
    │   Maestro.populate() via query mechanism
    │
    ↓
schema.adapter.build_config(raw_config)
    │   Initialize Environment + build Config
    │
    ↓
schema.normalize.normalize(config)
    │
    ↓
schema.validate.validate(config)
    │
    ↓
data.core.build_state(config)
    │   PyRAIState (static task descriptors registered)
    │
    ↓
workflow.execute(state) — imported from workflow
```

---

# 6. Interface Layer Design

## 6.1 Purpose

The interface layer provides the boundary between external user expressions and PyRAI3MD internal objects.

Its responsibility is:

```text
External Representation → Raw Configuration (dict)
```

For `parser.py`, this includes coordinate file loading and basic structure validation.
For `loader.py`, only minimal normalization is performed.

**No type conversion, no completeness check, no default filling.**

---

## 6.2 Interface Components

```text
interface/

├── __init__.py       # Exports parser, loader
├── parser.py         # Traditional input file → raw dict + coordinate loading + structure validation
└── loader.py         # Agent JSON config → raw dict
```

> Note: Named `loader.py` to avoid conflict with `schema/config.py` (Pydantic model definitions).

---

# 7. interface.parser

## 7.1 Core Principle

**Parse, load coordinates, and validate structure. No type conversion, no completeness check, no default filling.**

All values remain as raw strings. Type conversion is deferred to `schema.adapter`.

`parse_input()` internally performs the following steps:
1. Parse input by file extension → raw dict: legacy text (default), JSON (`.json`) or YAML (`.yaml` / `.yml` via `_parse_yaml`)
2. Load coordinate file if external reference (`_load_coordinate_file`)
3. Detect a companion StrOrb file (`_detect_strorb_file`): first `{title}.StrOrb` next to the input file, then a same-basename `.StrOrb`; when found the absolute path is recorded in `raw_config['strorb_file']`
4. Validate atoms and coordinates are resolved (`_validate_structure`)

---

## 7.2 Classic Input Processing

Example input:

```text
&CONTROL
title         atod
jobtype       hybrid
qm            nn

&Molecule
ci       2 2
spin     0 1
coupling 1 3, 1 4, 2 3, 2 4
```

Parsed output:

```python
{
    'control': {
        'title': 'atod',
        'jobtype': 'hybrid',
        'qm': 'nn',
    },
    'molecule': {
        'ci': '2 2',            # raw string, NOT [2, 2]
        'spin': '0 1',          # raw string
        'coupling': '1 3, 1 4, 2 3, 2 4',
    },
}
```

Three invariants of that loop, and the reason they are invariants:

- **Every section is read the same way.** `line.split()` gives one keyword and the rest of the line
  its raw value, and a value is kept as a string. There is no per-section special case in
  `parser.py` and there must not be one: a section whose text needs more than `keyword value` is
  interpreted one layer down, where `&molecule`'s center groups already are (§9.1). The constraint
  block used to be the exception, with its own parser under `interface/`; #271 removed the nested
  grammar it existed for, so the file is gone and the loop is generic again.
- **`&end` closes the current section.** It used to be filed as a section named `end`, and every
  following key landed there - a silent loss of the rest of the file. It now just stops the loop
  owning keys, and the sections after it parse as written.
- **One keyword per line, and a repeated keyword is last-wins.** That is the flat format's price for
  having no nesting, and the constraint section pays it like every other one: several objects of one
  kind are the `;` segments of one line, not the line written several times.

The section that does use the extra separator reads, in text, as flat lines - and none of the
parsing below is in this layer:

```text
&constraint
  shake_bond 1 2; 1 3        # two objects, one line
  shake_bond_ref 0.96        # one value, broadcast over both (recorded as a note)
  wall_type sphere
  wall_sphere_radius 4.5
  wall_atoms 1-6
  wall_fc 0.002
```

`_parse_text` hands `{'constraint': {'shake_bond': '1 2; 1 3', ...}}` over unchanged; the meaning of
`;`, of the broadcast, of `wall_type` gating and of the ramp keywords belongs to
`schema.adapter._read_constraint_section`, and its inverse `write_constraint_section` generates the
same shape from a structured block (knowledge base Q47-Q49,
`dev/methods/CONSTRAINT_DEVELOPMENT.md` §4).

---

# 8. interface.loader

## 8.1 Purpose

`loader.py` loads agent-generated or structured task descriptions from JSON files.

It performs minimal normalization:

- Read JSON file
- Normalize section names to lowercase

---

## 8.2 Difference Between parser.py and loader.py

| Module | Input | Processing | Output |
|---|---|---|---|
| parser.py | Traditional text/JSON files | Section splitting + raw extraction | Raw dict |
| loader.py | Agent JSON configs | JSON loading + lowercase normalization | Raw dict |

Both produce the same output format and enter the same downstream pipeline.

---

# 9. Schema Layer: Adapter + Normalize + Validate

## 9.1 Adapter (schema/adapter.py)

The adapter bridges Interface (raw strings) and Schema (typed values):

```text
Raw dict (sectioned, raw strings)
    │
    ↓
_apply_env_defaults(env_config)
    │   Merge package + user config as base
    │   Runtime input takes priority
    │   Empty higher-priority values (None, '', [], {}) fall back
    │   to the next lower-level non-empty value (False and 0 never
    │   fall back)
    │
    ↓
_read_constraint_section()  (per-section text semantics)
    │   '&constraint' lines -> the structured mapping of system.constraints,
    │   in the same place as the '&molecule' centre-group parsers.
    │   write_constraint_section() is its inverse, used by the converter.
    ↓
_flatten_sections()
    │   Merge control/molecule/md sections into flat dict
    │   Apply field name mapping (e.g. temp → temperature)
    │   Sections listed in NESTED_SECTION_FIELDS ('constraint', and the
    │   retired 'system_constraint') keep their shape and route to
    │   ('system', 'constraints') instead of flattening
    │
    ↓
_convert_types()
    │   Use ReadVal/ReadIndex for type conversion
    │   '2 2' → [2, 2], '300' → 300.0, '1 3, 2 4' → [[1,3],[2,4]]
    │
    ↓
Config.model_validate()
    │   Pydantic validation → typed Config object
    │
    ↓
Config (fully typed)
```

## 9.2 Task Queue Building (build_state)

After normalize/validate, `build_state(config)` creates PyRAIState, expands the task
queue, and registers one static task descriptor per task:

```text
build_state(config):
    state = PyRAIState(...)
        state.jobtype is a read-only property (derived from config.control.jobtype)

    maestro.resolve_skills(config) → skills dict
    resolve_task_queue(config, skills) → task descriptors
        (composite expansion: ONIOM/fragment/qmmm; contribution + region indices)
            │
            ↓
    _build_task_queue: for each task in the queue:
        Merge skill config values into a flat dict (cached per skill)
            │
            ↓
        task['config'] = {'skill_config': flat, 'results': {}}
            (plain dict; build_pyrai_config() is deprecated)
            │
            ↓
        StrOrb injection: raw_config['strorb_file'] + molcas_skill
            + 'guess_orbital' not set → merged['guess_orbital']
            │
            ↓
        optimize special case (driver == 'main' and jobtype == 'optimize'):
            validate opt_method against OPTIMIZER_REGISTRY
            force merged['jobtype'] = 'energy gradient'
            opt_state_i > 0 and no 'target_states' →
                merged['target_states'] = [opt_state_i]
            │
            ↓
        state.add_task(category, task)
```

Backend instantiation is deferred to `Calculator.initialize()` (Runtime phase),
not performed during `build_state()` (Planning phase). This ensures complete
separation between Planning and Runtime.

This ensures each task gets its own plain dict config, and all tasks are registered
in the PyRAIState task queues.

Key improvement: task configs never fix geometry — region slice indices stay in the
task metadata, and the Calculator derives atoms/coordinates/charges from the live
structure at execution time.

## 9.3 Normalize (schema/normalize.py)

Computes derived quantities from the typed Config:

- `natom` from atoms list
- `total_electron` from atomic numbers and charge
- `nalpha`/`nbeta` from total_electron and multiplicity
- `nstate` from state_spin (if set) or ci
- `npair`/`nnac`/`nsoc` from coupling and state_spin
- `normalize_skill_name()` — resolve skill_name from qm[0]
- `derive_current_state()` — fill `system.current_state` from `dynamics.root` only when `current_state` is left at its default 0 (fill-if-default)
- `normalize_history()` — raise `history` values below 4 to 4
- lattice reconstruction — rebuild the 3×3 `lattice_vector` from lattice/cell parameters

The normalize module also provides `normalize_jobtype()` for
alias replacement and deduplication.

## 9.4 Validate (schema/validate.py)

Nine consistency checks run in sequence by `validate(configs)` (`schema/validate.py:1354-1362`; no maestro/registry parameter):

- `validate_task()`: jobtype recognized, title non-empty, ncpu > 0, theory/reference methods checked against the dynamic whitelist
- `validate_system()`: natom/coordinates/mass dimension consistency
- `validate_electronic()`: charge/multiplicity physically reasonable
- `validate_centers()`: the center plan - `state_i`/`state_j` inside the declared manifold, region indices, multi-center partitions
- `validate_dynamics()`: step >= 0, size > 0, temperature >= 0, root >= 0
- `validate_adaptive()`: threshold ranges (max >= min), refinement params
- `validate_skill_name()`: skill_name exists among the discovered skills
- `validate_dispatch_sections()`: the `multitask`/`remotetask`/`multipool` triple as each section states it
- `validate_search_section()`: the `&search` grid vocabulary and its type rules

The theory/reference method whitelist is discovered dynamically (#150): `_valid_theory_methods()` = `BUILTIN_THEORY_METHODS` (`{'nn', 'e2n2'}`) ∪ the methods reported by `skills/query.discover_all()`; the result is cached after first computation, and `FALLBACK_THEORY_METHODS` is used when discovery yields nothing. `_strip_skill_prefix()` accepts the `qm.` / `abinit.` prefixes before matching.

Also contains the `Task` model (merged from former task.py).

---

# 10. Config Model Structure (schema/config.py)

The core Config model nests eleven sub-models plus a `results` dict:

| Sub-model | Responsibility |
|---|---|
| Control | Job control, resources, skill identification |
| System | Atoms, coordinates, charges, electronic structure, state info |
| Dynamics | Ensemble, temperature, MD propagation (velocities live in System) |
| Adaptive | Adaptive sampling thresholds, refinement |
| Optimize | Geometry optimization control (optimizer, convergence thresholds, target states) |
| Hessian | Numerical Hessian / vibrational analysis request (&hessian) |
| Search | Hyperparameter grid scan (&search) |
| Train | Training request (&train) |
| Prediction | Prediction request (&prediction) |
| Composite | Multi-layer / multi-center plan (&composite) |
| Environment | Execution mode, program paths, slurm settings, runtime execution parameters |
| Results | Computed results (energy, gradient, nac, soc, spectra) |

`Properties` and `Calculation` are **not** framework sub-models: they belong to the skill schemas. Flat routing registers Control / System / Dynamics / Adaptive / Optimize / Composite in `FIELD_TO_SUBMODEL`; Hessian / Search / Train / Prediction stay out of it on purpose (their keyword names collide with other sections) and are routed by section name through `NESTED_FRAMEWORK_SECTIONS` (`schema/config.py:2130-2141`).

Each Skill maintains its own extended config (e.g. `pyscf_skill/schema/config.py`) with skill-specific fields such as `nact_elec`, `nact_orb`, `ci_roots`, `state_spin`, `target_states`, `target_nac_pair`, `target_soc_pair`, and `shielding_tensors`.

Schema validates only the skill-independent fields. Skill-specific fields are validated by individual Skills.

---

# 11. Module Responsibilities

| Module | Responsibility | Does NOT |
|---|---|---|
| Interface (parser/loader) | External input → raw dict (+ coord + validate for parser) | No type conversion |
| Adapter | Raw dict → typed Config (env defaults + flatten + convert) | No scientific validation |
| build_state | Create PyRAIState + resolve skills + register static task descriptors (backend instantiation deferred) | No computation |
| Normalize | Compute derived quantities + resolve skill_name (`normalize(configs)`, skills discovered through `skills/query`) | No validation |
| Validate | Consistency checks (task/system/electronic/centers/dynamics/adaptive/skill_name/dispatch/search; `validate(configs)`) | No skill-specific parameters |
| Kernel + Maestro | Skill discovery (via query), resolution, config store | No scientific knowledge |
| Skill | Parameter parsing, method validation, computation | No core modification |
| PyRAIState | Unified runtime state (after all validation) | Not constructed before validation |

---

# 12. Design Advantages

## Unified Runtime State

Regardless of input source (classic text, JSON, agent config), all become `PyRAIState`.

## Layered Processing

1. **Interface**: Pure extraction — no knowledge
2. **Adapter**: Type conversion — no validation
3. **Normalize**: Derived quantities — no judgment
4. **Validate**: Consistency checks — no skill knowledge
5. **Skill**: Domain-specific validation and execution

---

# 13. Implementation Status

| File | Status | Description |
|---|---|---|
| `interface/parser.py` | Implemented | Classic text + JSON + YAML parsing + coordinate loading + StrOrb detection + structure validation |
| `interface/loader.py` | Implemented | JSON config loading + lowercase normalization |
| `schema/config.py` | Implemented | Core Config model (Control/System/Dynamics/Adaptive/Optimize/Hessian/Search/Train/Prediction/Composite/Environment/Results) |
| `schema/adapter.py` | Implemented | Env defaults + section flattening + ReadVal/ReadIndex type conversion |
| `schema/normalize.py` | Implemented | Derived quantity computation + skill_name resolution (`normalize_skill_name(data)`) + jobtype normalization |
| `schema/validate.py` | Implemented | Nine consistency checks (dynamic theory whitelist) + Task model |
| `schema/reference.py` | Implemented | Periodic table and physical constants |
| `kernel/environment.py` | Implemented | Runtime env bootstrap: config loading + Maestro population |
| `kernel/maestro.py` | Implemented | Skill coordination: discovery, resolution, config store, skill class registry |
| `kernel/logger.py` | Implemented | Run log writer (start/end lines, heading, per-step and optimize blocks) |
| `pmd.py` | Implemented | Pure orchestration (parse → build_config → normalize → validate → build_state → execute) |
| `agent.py` | Implemented | PyRAI3Agent class with same pipeline as pmd.py |
| `__main__.py` | Implemented | python -m entry point |
| `skills/query.py` | Implemented | Dynamic skill discovery via query mechanism (discover_all, resolve_label) |
| `skills/wrapper.py` | Implemented | pyrai3md_skill console_script entry point |
| `utilities/logo.py` | Implemented | Startup logo and version |
| `utilities/timer.py` | Implemented | Timestamp (`what_is_time`) and elapsed-time formatting (`how_long`) |
| `utilities/structure_formatter.py` | Implemented | Coordinate and point-charge table formatting for the run log and trajectory frames |
| `utilities/print_results.py` | Implemented | Per-step / final result block formatting (`format_result_block`) and the `opt_status.txt` record reader |
| `utilities/opt_formatter.py` | Implemented | The `opt_status.txt` header/row format |
