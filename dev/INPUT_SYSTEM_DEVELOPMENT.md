# PyRAI3MD Input System Development

Last updated: 2026-11-12 00:50 - section 5.5 added: `&read` / `read:` (jobtype `read`, `path`/`project`/`profile`, four label switches, one theory, no `&end`, nested routing, `workflow/read.py`, skill `service/read.py`), checked against `schema/config.py`, `schema/adapter.py`, `schema/validate.py`, `planning/assembly.py`, `workflow/__init__.py` and `workflow/read.py`. Earlier: 2026-11-06 13:45 - section 4 gained the `highref`/`midref` entry (the reference partition of an adaptive run: the same `si` type, the same 1-based center-group parsing as `highlevel`/`midlevel`, the five rules `validate_reference_partition` actually enforces, and the fact that the two fields are inert outside `jobtype: adaptive`), checked against `schema/adapter.py` and `schema/validate.py`. Nothing else was re-read. Earlier: 2026-11-02 10:40 - #448: the front matter gained the pair-keyword block (#448): `transition_pair` defaults to ground-to-state starting at state 1, a declared self pair is refused at normalization, and nothing is filled in behind the input. The rest was not re-audited. Before it, 2026-10-20 08:10 - section 7.2 now records that both Slurm script builders render the same `slurm:` section through `kernel/sbatch.py` (canonical hyphenated directives, raw `extra`, unknown keys refused), so the two paths no longer keep two dialects. The rest was not re-audited. Before it, 2026-10-19 19:52 - #395 wrote the dispatch trio into the classic carrier, so section 5.2 gained the rule of where `multitask`/`remotetask`/`multipool` get stated and the measured reading of what their absence cost, checked against `qualification/certification/converter.py` (`_MD_ECHO_KEYS` and the primary-skill flatten loop), `certification/handler.py:_build_skill_pairs`, and `qualification/verification/test_example_dispatch.py::TestTheDispatchPoolReachesTheClassicCarrier` on this checkout. The section 5.3 `skill_sections` paragraph (#393, re-measured for #394) stays as written. The rest of this file is as it stood on 2026-10-10 12:13 and was not re-audited for this change.

---

> **Multi-center input (refactor R1, 2026-09-08).** CHANGELOG #199,
> `dev/MULTICENTER_DEVELOPMENT.md` §4. The electronic keywords `ci` / `spin` /
> `coupling` accept both the flat single-center form and a nested per-center
> form, and `root` accepts a scalar (C=1) or a per-center list; centers are
> separated by `;` (classic strings) or nested lists (typed JSON/YAML), the
> pairs within a center by `,`. External region/coupling indices are 1-based and
> normalized to 0-based by the adapter (a full-width semicolon U+FF1B is
> rejected with a hint); `root` is externally 1-based like every other
> external index and is decremented once by the adapter. `validate_centers` fails any
> per-center inconsistency (ci/spin length mismatch, negative or all-zero ci,
> out-of-range/duplicate/self coupling pair, root outside [1, nstate[c]] (external),
> overlapping regions) before any backend directory, thread or Slurm submission.
> A plain single-center run keeps the flat form and is unchanged.

> **State pairs the input names (`molecule.coupling`,
> `molecule.transition_pair`).** `coupling` is the authority for a coupling
> request and `nac_pair_map`/`soc_pair_map` are its derived internal labels, an
> extra entry the nested formats may also carry - never a second source of truth
> (`dev/MULTICENTER_DEVELOPMENT.md` 4.4 item 5). `transition_pair` follows the
> same shape rule, and its default when nothing is declared is
> `[[0, i] for i in 1..nstate-1]`: the state-to-ground convention, **without**
> the `(0, 0)` self pair the earlier `range(nstate)` spelling produced. A
> transition is between two states, so a declared pair naming one state twice is
> refused at normalization (`schema/normalize.py::compute_derived_quantities`)
> rather than dropped, and the row it used to occupy was read as a transition
> dipole of zero (#448). Nothing else is filled in behind the input: no adjacent
> pair, no S0-Sn expansion inside a skill.

## 1. Design Philosophy

PyRAI3MD adopts an environment-driven configuration system rather than a traditional input-file-driven architecture.

Traditional quantum chemistry:

```text
Input file → Parser → Calculation
```

PyRAI3MD:

```text
Environment Configuration (pmd_env.yaml)
        +
User Overrides
        +
Task Input (classic / agent / API)
        |
        v
Unified Config (schema)
        |
        v
Workflow Execution
```

Core principle:

> Input files are only user-facing representations of calculation requests.
> The internal protocol of PyRAI3MD is the validated configuration model generated from environment and task overrides.

The complete directory structure is defined in [ARCHITECTURE.md](ARCHITECTURE.md).

The interface and schema pipeline details are defined in [INTERFACE_DEVELOPMENT.md](INTERFACE_DEVELOPMENT.md).

This document focuses on the environment configuration system, the configuration priority mechanism, and the connection between the input system and the test/skill infrastructure.

---

## 2. Configuration Layers

PyRAI3MD uses a three-tier layered configuration system with clear priority ordering.

### 2.1 Package Environment

Location:

```text
pyrai3md/qualification/pmd_env.yaml    (source in repository, installed with package)
```

The file lives inside the package (`pyrai3md/qualification/`) and is distributed with the installed package via `setup.py` `package_data`, so that it is available after `pip install`.

Purpose:

- Official default configuration
- Core keyword definitions and default values
- Skill capability registry template
- SLURM scheduler defaults

Properties:

- Managed by developers
- Updated through `pip installation`
- Read-only during runtime
- Never modified by users

Example content:

```yaml
environment:
  python: "python3"
  conda: ""
  scheduler: ""
  export:
    OMP_NUM_THREADS: [4]
    MPI_NP: [4]

slurm:
  time: "24:00:00"
  job_name: pyrai3md

skills:
  bagel_skill:
    capability:
      energy: true
      gradient: true
      nac: true
      soc: true
```

### 2.2 User Environment

Location:

```text
~/.pyrai3md/pmd_env.yaml
```

Purpose:

Persistent user-level customization that overrides package defaults.

User environment stores:

- Executable paths (e.g. skill backend binaries)
- Resource settings (CPU count, memory limits)
- Personal defaults (preferred partition, account)
- Local environment options (scratch directories)

It does not redefine:

- Workflow architecture
- Skill capability registry
- Schema definition

Example:

```yaml
environment:
  scratch: /scratch/user
  export:
    OMP_NUM_THREADS: [64]

skills:
  molcas_skill:
    molcas_path: /opt/openmolcas
    molcas_mem: 8000
  bagel_skill:
    bagel_path: /home/user/BAGEL/bin
```

### 2.3 Runtime Input

Runtime input is task-specific and has the highest priority.

Sources:

- Classic input file (`input`)
- Agent configuration (`input.config.json`)
- Python API dictionary

Runtime input overrides both user environment and package environment.

Companion files next to the input file follow the `{title}` naming convention (the same family as `title.coord` / `title.velo`):

| File | Purpose |
|------|---------|
| `{title}.coord` | Coordinates (a same-basename `.xyz` is also accepted) |
| `{title}.velo` | Initial velocities |
| `{title}.StrOrb` | Initial-guess orbitals (OpenMolcas StrOrb format); auto-detected by `_detect_strorb_file()` in `interface/parser.py` — first `{title}.StrOrb`, then a same-basename `.StrOrb` — and recorded in `raw_config['strorb_file']` |

### 2.4 Runtime Environment Module

The environment module (`kernel/environment.py`) initializes the PyRAI3MD runtime environment before task execution.

Responsibilities:

- Locate package-level `pmd_env.yaml`
- Locate user-level `~/.pyrai3md/pmd_env.yaml`
- Initialize the environment configuration
- Trigger skill synchronization through `kernel.maestro`
- Provide runtime environment information to the kernel

The environment module does not:

- Parse user input files
- Validate task parameters
- Generate workflow configurations

Configuration resolution is completed by the schema module.

### 2.5 Skill Synchronization

Skill plugins are dynamically extensible. When a new skill package is installed (e.g. `pyrai3md-orca`), PyRAI3MD must automatically discover its capabilities.

Each skill provides a `manifest.yaml` containing:

- Supported properties
- Methods
- Parameters
- Executable requirements

Synchronization happens automatically during startup:

```text
PyRAI3MD startup
    |
    v
Read package pmd_env.yaml
    |
    v
Read ~/.pyrai3md/pmd_env.yaml
    |
    v
Compare skill registry
    |
    v
Add missing skill definitions
    |
    v
Create runtime environment
```

Synchronization rule:

- Only update `skills.capability` entries
- Do not overwrite `skills.settings` (user-configured paths and values)

Example:

```yaml
# Before synchronization:
skills:
  bagel_skill:
    capability:
      energy: true
    settings:
      bagel_path: /home/user/bagel

# New package adds gradient capability:
# After synchronization:
skills:
  bagel_skill:
    capability:
      energy: true
      gradient: true
    settings:
      bagel_path: /home/user/bagel    # preserved
```

---

## 3. Configuration Priority

Final configuration is resolved according to:

```text
Runtime Input  (highest priority)
      >
User Environment  (~/.pyrai3md/pmd_env.yaml)
      >
Package Environment  (pyrai3md/pmd_env.yaml)
```

Example:

```yaml
# Package (pyrai3md/pmd_env.yaml):
environment:
  export:
    OMP_NUM_THREADS: [4]

# User (~/.pyrai3md/pmd_env.yaml):
environment:
  export:
    OMP_NUM_THREADS: [64]

# Runtime input (agent config):
# Task-specific value overrides both

# Final resolved: OMP_NUM_THREADS = task-specific value
```

The environment module (`kernel/environment.py`) locates and loads the package and user configurations. The schema module (`schema/adapter.py`) completes the configuration resolution by merging environment defaults with runtime input. The normalize module (`schema/normalize.py`) computes derived quantities from the resolved configuration.

---

## 4. Schema Config Model

The input system ultimately produces a `Config` object through the schema pipeline.

The Config model is defined in `schema/config.py` as the nesting of seven sub-models:

| Sub-model | Responsibility |
|-----------|---------------|
| Control | Job control, resources, skill identification |
| System | Atoms, coordinates, charges, electronic structure, state info |
| Dynamics | Ensemble, temperature, MD propagation (velocities live in System) |
| Adaptive | Adaptive sampling thresholds, refinement |
| Optimize | Geometry optimization control (optimizer, convergence thresholds, target states) |
| Environment | Program paths, slurm settings, shell commands (the framework dispatch layer's machine section; it declares **no** execution switch — see the retired keys below) |
| Results | Computed results (energy, gradient, nac, soc, spectra) |

`Properties` and `Calculation` are **not** framework sub-models: they belong to the skill schemas and are managed by individual Skills through their own Config models.

The framework Config contains ~160 fields. Only the five routing sections Control / System / Dynamics / Adaptive / Optimize are registered in `FIELD_TO_SUBMODEL` (the field → sub-model routing table); Environment and Results are attached to the state through separate mechanisms.

`control.debug` is the one field that deliberately crosses into the skill layer, and the routing table is why it needs a rule of its own. `_merge_one_skill_config` overlays framework fields onto a skill's config **by field name**, and `lammps_skill` already owns the spelling `debug` for its per-step artifact files. Left to the name match, the ladder would put a skill's own `debug` above the switch that is supposed to include it, and would let an unset framework default (False) silence a skill test that asked for True. It is therefore applied as layer 0 of that merge, after every other layer: `control.debug` true forces every module under test into debug, and an unset one leaves a skill's own setting alone. Both directions are pinned in `qualification/verification/test_diagnostics.py`. A new cross-layer field has to be placed the same way -- by name it is invisible to the layering question.

Configuration-layer keywords:

- **`optimize` section `opt_` prefix contract**: every Optimize field carries the `opt_` prefix (`opt_method`, `opt_maxit`, `opt_energy`, `opt_grms`, `opt_gmax`, `opt_drms`, `opt_dmax`, `opt_use_rfo`, `opt_state_i`, `opt_state_j`); `PyRAIState` re-exposes them as the read-only attributes `optimize_method`, `optimize_maxit` and `optimize_params`.
- **`derive_current_state`**: `schema/normalize.py` fills `system.current_state` from `dynamics.root` only when `current_state` is left at its default 0 (fill-if-default).

Shell keyword two-level semantics (#134, see Section 7.4): the global `environment.shell` is consumed only by the main framework dispatch scripts; skills consume only their own skill-level shell keyword.

Periodic-input keywords (#216, see `dev/methods/PBC_DEVELOPMENT.md`):

- **`lattice`** (System): the classic six parameters `[a, b, c, alpha, beta,
  gamma]` (lengths Angstrom, angles degrees), nine components, or a 3×3
  matrix. `schema/adapter.py::_parse_lattice_value` converts all three forms
  to the `lattice_vector` matrix (`FIELD_MAP` renames `lattice` →
  `lattice_vector`); an unsupported shape raises at the input layer. Both the
  classic string (`lattice 10 10 10 90 90 90`) and the typed YAML/JSON list
  go through this conversion.
- **`cell`** (System): six parameters or nine components; `normalize`
  derives `lattice_vector` from it. The default `[99, 99, 99, 90, 90, 90]` is
  the **"not filled" sentinel** — no `lattice_vector` is derived from it, so
  a non-periodic run never carries a fake 99 Å cubic cell. A `cell` that
  conflicts with an explicit `lattice_vector` raises.
- **`pbc`** (System): normalized to exactly three `0/1` flags and is the
  **single periodicity authority** (a non-zero cell alone never makes a run
  periodic). Structural rules (`System` field validators): exactly 3 elements,
  each `0/1`; `cell` length 6/9 and finite; `lattice_vector` 3×3 and finite.
- **`volume`** (System): derived from a valid `lattice_vector` when not given.
- **`highref` / `midref`** (System): the reference partition of an adaptive run,
  spelled and parsed exactly like `highlevel` / `midlevel` - `si` type, 1-based
  in the classic and YAML forms, normalized to 0-based, flat list or
  `;`-separated center groups (`_CENTER_INDEX_FIELDS` and the same pair-keyword
  block `highlevel` uses). They are read only for the reference queue and never
  enter the theory centre plan; outside an adaptive jobtype the two fields are
  inert and pass through unchanged. Inside one,
  `validate_reference_partition` requires them single-center, duplicate-free, in
  range and disjoint from each other, requires a layered reference to have an
  effective high region and at least one atom left for the low layer (a
  three-layer spelling also needs its mid region), and requires a single
  reference layer that states `highref` to cover exactly the neural-network
  surface.
- **validate semantics** (`schema/validate.py::validate_system`): `any(pbc)`
  requires a valid, non-singular cell; **partial periodicity (mixed 0/1) is
  rejected in this stage** (only full 3D or non-periodic); and a periodic run
  whose **outermost** method - the whole-system provider, read as the theory tail
  at configuration time - declares `supports_pbc=False` is refused, while the same
  molecular method is accepted as a center region of a multi-center run, because a
  center is handed no periodicity at all (#315,
  `dev/methods/PBC_DEVELOPMENT.md` section 5.4). "Not
  filled" (`pbc=[0,0,0]`) vs. "explicitly off" are the same non-periodic
  state; enabling `pbc` without a cell is an error, not a silent no-op.

For the detailed interface → adapter → merge → normalize → validate pipeline, see [INTERFACE_DEVELOPMENT.md](INTERFACE_DEVELOPMENT.md).

---

## 5. Input File Formats and Generation

### 5.1 Input Sources

All input formats are converted into the same internal representation through the interface layer:

```text
                 Text Input
                     |
                     v
              interface.parser
                     |
                     +
                 JSON Config
                     |
                     v
              interface.loader
                     |
                     v
                  raw dict
                     |
                     v
              schema.adapter
                     |
                     v
              Config Object
```

### 5.2 Input File Generation by Qualification

The qualification framework (`qualification/manager.py`) generates input files for qualification execution. This is the key connection between the input system and the qualification infrastructure.

For each skill test case, the manager generates:

| File | Format | Target Command | Mode |
|------|--------|---------------|------|
| `<test>.config` | JSON | `pyrai3md_agent` / `pyrai3md_skill` | Agent and skill modes |
| `<test>.input` | Legacy text | `pyrai3md` | CLI mode (via converter) |
| `<test>.xyz` | Coordinates | `pyrai3md` | Same-basename coordinates for CLI mode |

The `.config` files are generated by skill test generators (see Section 6). The `.input` files are produced by `qualification/certification/converter.py` from the `.config` JSON.

For each qualified example (`pyrai3md/examples/*.yaml`), the manager converts the YAML to a flat JSON config (via `qualification/certification/converter.py: yaml_to_config`) and then to `.input` + `.xyz`, so every qualified example is exercised through the two entry points that take a user-authored input file: `pyrai3md <yaml>` and `pyrai3md <input>`. No agent `.config` is rendered for examples — the planner deliberately skips `pyrai3md_agent` there because its input is a machine artifact rather than user input and it adds no parser coverage; the agent `interface.loader` pipeline stays qualified through the skill tests, which run all three modes.

All generated files are stored in `tmp_YYYYMMDD_HHMMSS/` working directory.

**The dispatch trio has to be written where it was declared, not somewhere in the flat namespace.**
`multitask` / `remotetask` / `multipool` ask one question of one section - "how many workers does
THIS fan-out get" - and two different sections legitimately ask it: `&md`/`&search`/`&train`/
`&prediction` size the framework's fan-out, while `&cp2k` sizes its own strain route, which
differentiates a box through twelve perturbed single points. The nested workflow sections already
carried theirs (`WORKFLOW_SECTION_KEYS`), and nothing else did: a skill's three fields stay out of
its `reference.DEFAULT` on purpose, because a DEFAULT entry would put them in every generated
input, and `&MD`'s keys are flattened to the top level where the echo table had no entry for them.
The result was one example running parallel in yaml mode and serial in `.input` mode, priced on the
deployment: `optimize_cell_cp2k` took 589 s in yaml against its own 821 s serial reading, and 815 s
in `.input` against 817 s, because the carrier stated no pool at all; locally `md_parallel.yaml`
gives the workflow node `multitask=True multipool=3` from yaml and `False/1` from its own generated
text. Neither reader was the obstacle - hand-inserting the lines returns `True/3` for `&MD` and two
workers for `&CP2K` - so the fix is on the writer: `&MD` echoes the trio presence-style like every
other dynamics keyword it carries, a skill section states the trio present in its own block, and a
skill's trio is no longer copied into the flat top level, because the flat namespace has one
spelling and two owners. That last clause is what a synthetic carrier with `md.multipool: 3` beside
`cp2k.multipool: 2` prices: flatten the engine's pair instead and `&MD` renders cp2k's number.
Over the shipped set the blast radius is measured, not argued: of 96 generated classic inputs 92 are
byte-identical and 4 change - the two `&MD` carriers that declare a pool and the two `&CP2K` strain
legs (`qualification/verification/test_example_dispatch.py::TestTheDispatchPoolReachesTheClassicCarrier`).
An earlier cut tried to make a skill block fall back to the shared flat namespace for keywords its
block lacked, and the same byte comparison showed it moving `ensemble`, `thermostat`, `barostat`,
`temperature`, `pressure` and `debug` out of `&MD` and into `&XTB`/`&PYSCF` of the multicenter
carriers; it was dropped. A block is fed by its own section, and `&MOLECULE`/`&MD` keep what they
own - which is also why no criterion claims "an engine block never states a dynamics keyword": 56
shipped examples' engine blocks legitimately carry `temperature`, `pressure` or `debug` (the last
being `lammps_skill`'s own spelling of its artifact switch).

### 5.2a One rule of the classic section reader

`interface.parser._parse_text` knows nothing about what any keyword means: a
section is `key value...` lines and the typing belongs to the section's own
reader in `schema.adapter`. One rule of that reader changed on 2026-10-09,
because it was silently discarding input:

- **A keyword repeated inside one section keeps both values** (as a list), in
  the order they appeared. It used to keep only the last, so
  `grid depth 1 2` followed by `grid nn_size 8 16` planned a scan over one
  axis and nothing downstream could tell the other line had ever existed --
  the silent-drop class this document already records for `#299`/`#300`. A
  reader that cannot use several values for one key now says so by name.
- No shipped input duplicates a key, and the converter emits at most one pair
  per keyword, so the rule is a guard rather than a behaviour change.

### 5.3 Agent Config Format

The agent entry point accepts JSON configuration files:

```json
{
  "control": {
    "title": "test_energy",
    "jobtype": "sp"
  },
  "system": {
    "atoms": ["O", "H", "H"],
    "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]],
    "charges": 0,
    "multiplicity": 1
  },
  "calculation": {
    "method": "casscf",
    "task": "energy"
  }
}
```

The loader (`interface/loader.py`) reads this JSON and passes it through the same schema pipeline.

**Dotted flat format (recommended agent medium).** Instead of nested sections, the agent may submit a single-level dict whose keys are `section.keyword` pairs:

```json
{
  "control.jobtype": "adaptive",
  "system.atoms": ["O", "H", "H"],
  "calculation.method": "casscf",
  "md.multipool": 4,
  "adaptive.multipool": 2
}
```

`schema/adapter.py` expands dotted keys at the first dot into sections (section names are lowercased; keys without a dot stay top-level) before the rest of the pipeline runs. Because each keyword carries its section explicitly, cross-section keywords with the same name (e.g. `multipool` in `&md` vs `&adaptive`) stay distinct — information that the legacy fully-flat dict lost.

Rules:

- Malformed keys (empty section or keyword around the dot) and collisions between a dotted section and an existing top-level entry raise `ValueError`.
- A **fully flat** keyword dict (no nested section at all, with top-level keys hitting framework keywords) is rejected with `TypeError`; use sectioned config or the dotted flat format.
- Nested sectioned config remains fully supported and is unchanged for classic/user inputs.

**Adaptive stage and screening keywords.** `&adaptive` owns the round loop controls (`maxiter`, `maxsample`, `record_step`, `keep_md`, `load`), the physical screening switches (`check_population`, `check_energy_order`/`gap_tolerance`), the property plan (`sample_energy`/`sample_grad` default on; `sample_nac`/`sample_soc`/`sample_dipole`/`sample_transition_dipole`/`sample_oscillator_strength` default off), and the per-stage dispatch triple (`md_multitask`/`md_remotetask`/`md_multipool`, `reference_multitask`/`reference_remotetask`/`reference_multipool`, `remote_train`). `check_population` decides whether the structure the md run stopped on for an ill population (md stop code 1) is collected; the population tolerance itself is `&md md_maxpop`, and the retired `&adaptive pop_tolerance` is refused by name. The prefixed keys override the generic `multitask`/`remotetask`/`multipool`; `reference_remotetask` switches the reference skill to its own slurm execution mode rather than submitting the framework runtime. The property switches are reconciled with `&md` before planning: adaptive on forces the matching md keyword on, adaptive off with md on warns and enables the adaptive switch, and both off skips the channel. The dynamic-threshold and refine keys remain parseable for input compatibility but a non-zero value is refused by the workflow. The `multipool` values are concurrency counts, not core counts: a skill's own `thread`/`ncpu` still decides how many cores one calculation uses.

**Classic adaptive input traps.** A skill section's keys are flattened into the framework namespace. Do not write `state_spin` in a skill section (it collides with `System.state_spin`; derive it from `&MOLECULE ci/spin`), do not use the retired `&md method` (use `initcond`), and keep the language of the input to framework keys. An adaptive run's `&CONTROL theory ...` + `reference ...` pair is a pair of workflow roles; the planner does not infer QMMM from it.

**The example converter's `skill_sections` key is a fourth source of a skill's config, and it is read.** `yaml_to_config` keeps every skill's parameter block under `skill_sections`, keyed by the skill stem, because two skills that spell the same keyword (`method`) cannot share one flat namespace; it then flattens only the first `control.theory` entry to the top level so the single-skill builder path stays byte-identical. `_merge_one_skill_config` (section 3) read `skills.<name>`, `&<STEM>` and `&<STEM>_skill` and did not read that key, so the flat intermediate the qualification harness writes was a file whose second skill had no way to say what it is: measured on round 18's own `examples/optimize_cell_meci_lammps/input/optimize_cell_meci_lammps.json`, the LAMMPS task arrived as `method='tda'`, its profile resolution was refused, and it answered `Keeping profile 'ff_energy'` on a task whose jobtype is `energy gradient` - a gradient request served by a profile that reports no gradient. `skill_sections[<stem>]` now enters that ladder where the input file's own `&<STEM>` block enters, so each bound skill reads its own block and the primary's flattened top level decides nothing about anyone else. `qualification/verification/test_multicenter_converter.py` pins the consumer side: one criterion asserts the two skills really declare different methods (otherwise "each reads its own" would be an identity), one pins the whole-system LAMMPS task to `('ff', 'ff_gradient', 'energy gradient')`, and one walks every shipped example with two or more theory entries and compares the flat mode's per-binding method/profile/jobtype table against the yaml mode's. That last criterion reads nine agreements and one named refusal: `optimize_ts_multiscale.yaml` declares `theory: [pyscf, pyscf]` - one skill bound twice, so only one section exists, no `skill_sections` key is written, and the config falls into the fully-flat rejection above. The refusal is pinned by its message rather than worked around.

### 5.4 Sections that own one field instead of merging their keys

Most framework sections flatten: `&md step 1000` becomes the flat field `step` on the Dynamics
submodel. One section cannot, because its inner words are the schema's own vocabulary and would
collide with fields that already mean something else (`System.groups`, `System.constraints`): the
constraint block. `NESTED_SECTION_FIELDS` (`pyrai3md/schema/config.py`) registers it under both of
its names as `section name -> (submodel, field)`, and
`_flatten_sections` hands the section over untouched so neither `FIELD_MAP` renaming nor the
`ReadVal` type conversion runs on its inner values:

```python
NESTED_SECTION_FIELDS = {
    'constraint': ('system', 'constraints'),
    'system_constraint': ('system', 'constraints'),   # retired spelling, still routed
}
```

`&constraint` is the classic-input spelling of `system.constraints`, and it is flat
`keyword value` lines like every other section - what it adds is one separator: a `;` splits
**segments**, an object line's segments being its objects and a parameter line's segments being the
per-object values. Inside a segment `,` groups fields and a space separates tokens. That grammar, its
keyword table, its broadcast/padding rules and its parameter-ramp keys are specified in
`dev/methods/CONSTRAINT_DEVELOPMENT.md` §4.3-§4.4, and both directions of it live in
`schema/adapter.py` (`_read_constraint_section`, `write_constraint_section`) - the interface layer
still only splits lines (see `dev/INTERFACE_DEVELOPMENT.md`, and knowledge base Q47).

**The same keywords are the only spelling in all three carriers** (ruling of 2026-10-02, recorded in
`dev/methods/CONSTRAINT_DEVELOPMENT.md` §32). A YAML or JSON block is the same line list, with the
list structure carrying what `;` carries in text: one element is one object, and that object's
tokens are either a scalar or their own list.

```yaml
system_constraint:
  wall_type: [sphere, box]        # the switch, exactly as `wall_type sphere box`
  wall_sphere_radius: 2.8         # one object, one scalar
  wall_atoms: [[28, 29, 30]]      # one wall on three atoms
  wall_center: [[4.199546, 4.199546, 4.199546], null]   # null is the `none` segment
  wall_groups: [[1, 2, 3, 4, 5, 6, 3], [7, 8, 9, 10, 2]]  # atoms + trailing molecule size
  shake_bond: [[1, 2, 1, 3]]      # one block of two bonds (arity x rows)
  shake_tol: 1.0e-10              # one value broadcast over the family
```

The structured records the reader emits (`units`, `groups`, `cvs`, `biases`, `walls`, `fixes`,
`constraints`, `metadynamics`, `notes`) are the schema's input and the code's convenience, not a way
to write a file: at input each of those keys, plus `version`, `schedules` and `fix_ref`, is refused by
name with the flat keyword that states the same family. The retired *nested block* grammar
(`&system_constraint` with `wall sphere` / indented attributes / `end`) was already unread, and an
input written that way still reaches the adapter as flat lines and is refused there.

---

### 5.5 The read-only extraction section (`&READ`)

`&read` / `read:` describes a read of finished calculation artifacts,
not a new calculation. The section owns one path list and four label
switches; the framework's `jobtype read` binds exactly one theory skill,
asks that skill's parser for the enabled labels, and returns the frames
to its caller. It starts no engine, creates no scratch directory,
validates no engine installation and writes no data file. The
training-data tool (`tools/training_data.py extract`) is the consumer.

```yaml
read:
  path: [run_1, run_2]      # directories, files, or a list file
  project: ''               # empty: discovered from the artifacts
  profile: ''               # empty: derived from the labels
  energy: true
  gradient: true
  nac: false
  soc: false
```

Classic input uses the same keywords. Sections are delimited by the next
`&SECTION` header; there is no `&end`:

```text
&read
path dirs.txt
energy 1
gradient 1
nac 0
soc 0

&molecule
charges 0
multiplicity 1
ci 1
spin 0
```

The section travels nested (`NESTED_FRAMEWORK_SECTIONS`) because its
label names (`energy`, `gradient`, `nac`, `soc`) are also skill
vocabulary; `_flatten_nested` excludes it from the skill merge so it can
never overwrite a skill's own `nac`/`soc` setting. `READ_TYPES` in
`schema/adapter.py` types the switches and the two strings; `path` is
deliberately absent because it is a string or a list, the same rule that
keeps `atoms` out of `HESSIAN_TYPES`. `schema/validate.py` refuses a read
section without `jobtype read`, an empty path, no enabled label, more
than one theory, a reference, a non-main driver or a multiscale
partition.

`planning/assembly.py` spells the skill jobtype from the switches
(`energy gradient nac soc` in canonical order) instead of from the empty
capability list, and writes `read.profile` before profile derivation.
The execution branch is `workflow/__init__.py::execute` ->
`workflow/read.py::read_calculations`; `pmd.run` returns that payload
without constructing a run logger. Each skill's `service/read.py`
rebuilds its validated config from the directory geometry, parses the
labels and returns one frame; `read_calculation` is re-exported by the
skill backend module and is deliberately not the Backend class, which
would validate the engine installation.

---

## 6. Connection to Skill Test System

### 6.1 Skill Test Generation (certification converter)

Skill test generation is skill-agnostic: the certification handler discovers the test files under each skill's `tests/` directory, and the certification converter parses the skill-owned test YAML definitions into flat skill JSON configs:

```text
<skill>/tests/*.yaml
        |
        v
qualification/certification/handler.py
(list_skill_test_yamls discovery)
        |
        v
qualification/certification/converter.py
(parse_test_yaml / generate_test_configs)
        |
        v
test_case.json (flat skill JSON config)
        |
        +----------------------------+
        |                            |
        v                            v
pyrai3md_skill (skill_agent)   converter -> {.input, .config}
        |                            |
        v                            v
   skill.agent                  pyrai3md / pyrai3md_agent
        |                            |
        v                            v
   skill backend               skill execution
```

The certification planner calls `converter.generate_test_configs(skill_name, output_dir)` to write config files into the test working directory. No per-skill generator is needed: `parse_test_yaml(yaml_path, skill_name)` applies unified rules keyed by the skill identity (name → title/project, nested `system` section for `current_state`, and the coupling a case declares rendered into the classic `coupling` line - nothing is invented for a case that declares none, #447).

### 6.2 Three Test Modes and Input Files

The test system verifies the input system at three levels:

| Mode | Command | Input File | What It Verifies |
|------|---------|------------|-----------------|
| `pyrai3md_yaml` | `pyrai3md <example>.yaml` | Example YAML | YAML input pipeline (example qualification) |
| `pyrai3md_input` | `pyrai3md <test>.input` | Legacy text (converter) | Full pipeline: converter → schema → kernel → skill |
| `pyrai3md_agent` | `pyrai3md_agent <test>.config` | Sectioned dict JSON (converter) | Agent interface → schema → kernel → skill |
| `skill_agent` | `pyrai3md_skill <test>.json` | Flat skill JSON (converter) | Skill standalone execution |

For detailed qualification mode design, see [QUALIFICATION_DEVELOPMENT.md](QUALIFICATION_DEVELOPMENT.md) and [SKILL_TEST_DEVELOPMENT.md](SKILL_TEST_DEVELOPMENT.md).

The first three modes share one determinism contract on the skill boundary: whichever entry point a run takes, the framework hands every skill backend a key-by-key identically typed flat dict for the shared vocabulary (classic-text spellings such as `ci 3` / `coupling '1 2, 2 3'` are unified by `schema/adapter.py::unify_skill_flat_dict`; see [SKILL_APP_DEVELOPMENT.md](SKILL_APP_DEVELOPMENT.md) Section 6.2.5). The contract is locked by `qualification/verification/test_unification.py` over the whole official example set. The `skill_agent` mode bypasses the framework entirely: the flat JSON is the agent's responsibility and the skill's own validation errors are the agent's feedback channel.

### 6.3 Converter Role

`qualification/certification/converter.py` bridges the example YAML / agent JSON config formats and the legacy text input format:

```text
example YAML                    config.json (from skill generator)
        |                                |
        v                                |
qualification/certification/converter.py |
(yaml_to_config)                         |
        |                                v
        +---------------------> qualification/certification/converter.py
                                         |
                                         v
                            legacy text input (for pyrai3md CLI)
```

The converter does not contain any backend-specific keywords. It only maps JSON fields to the legacy `&section` / `key value` text format. The one section whose text is not a plain keyword-to-value list is `&constraint`, and the converter does not spell it out either: it calls `schema.adapter.write_constraint_section`, which is the inverse of the section's reader and shares that file with it. A constraint block the flat grammar cannot state (an id on only some objects of a family, a shared `schedules` list, molecules of unequal size in one `wall_groups`) raises there rather than being rendered into a file that reads back as a different run.

### 6.4 Skill Schema Extension

Each skill maintains its own extended config with skill-specific fields:

```text
Core Config (schema/config.py)
    +
Skill Config (<skill>/schema/config.py)
    |
    v
Complete Skill Config
```

Example skill-specific fields: `nact_elec`, `nact_orb`, `ci_roots`, `state_spin`, `target_states`, `target_nac_pair`, `target_soc_pair`.

The skill generator produces config JSON that includes both core and skill-specific fields. The skill's own agent validates the skill-specific fields independently.

---

## 7. Environment Configuration and SLURM

### 7.1 pmd_env.yaml SLURM Section

The top-level `slurm` section in `qualification/pmd_env.yaml` states the
allocation a Slurm job asks for. It is a framework-level section: not
`&environment` (whose keys describe the shell a process starts in) and not
the skills' own `slurm_*` keywords (which belong to each skill's
self-submission path). The shape of the section:

```yaml
slurm:
  partition: default       # Queue/partition name
  nodes: 1                 # Number of compute nodes
  ntasks: 1                # Total MPI tasks
  cpus_per_task: 1         # CPU cores per task (OpenMP)
  mem: 8G                  # Memory per node
  mem_per_cpu: ""          # Memory per CPU; mutually exclusive with mem
  time: '4:00:00'          # Wall time limit, per job
  job_name: pyrai3md       # Job name in squeue
  output: slurm-%j.out     # Stdout file (%j = job ID)
  error: slurm-%j.err      # Stderr file
  account: ""              # Billing account
  qos: ""                  # Quality of service
  exclusive: false         # Exclusive node access
  constraint: ""           # Node feature constraint
  gres: ""                 # Generic resources (e.g. gpu:2)
  mail_type: ""            # Email notification type
  mail_user: ""            # Email address
  array: ""                # Job array specification
  extra: ""                # Additional raw sbatch options
```

A key left `''` writes no directive, and `exclusive: false` writes none
either. The merged section is read by `kernel/environment.Environment.get_slurm`.

### 7.2 The Two Scripts This Section Generates

Two builders turn this section into a submission script, for two different
jobs, but they read it through **one dialect** (`kernel/sbatch.py`):

| Builder | Job it wraps |
|---------|--------------|
| `qualification/manager.py::build_slurm_script` | a whole `pyrai3md` test invocation (`pyrai3md_input`/`pyrai3md_agent` modes; main framework tests run locally) |
| `workflow/dispatch/slurm_run.py::_build_script` (and `term_worker._build_term_script`) | one dispatched **child runtime**, or one backend term, of a run that asked for `remotetask` |

`SBATCH_OPTIONS` maps the snake_case keys onto the canonical hyphenated
directives (`cpus_per_task` becomes `--cpus-per-task`, `job_name`
`--job-name`, `mem_per_cpu` `--mem-per-cpu`), so the same section asks the
cluster for the same allocation whichever path submits it. An empty value
writes no directive, `exclusive: true` writes a bare `--exclusive` and
`false` writes none, and `extra` is copied verbatim as one or more lines.

Both builders **refuse** a key that maps to nothing (`memmory: 8G`) and name
the known set in the message - on the qualification path while the run is
being planned, on the dispatch path while the child's `slurm.sh` is written.
Use `extra` for a directive the mapping does not name; it is the one
pass-through key. Criteria in `qualification/verification/test_slurm_dialect.py`
read both builders' own scripts and pin the parity, the hyphenated spellings,
the boolean handling, `extra`, and the refusal.

### 7.3 Environment Variable Export

The `environment.export` section contains variables merged by the runner before test execution:

```yaml
export:
  OMP_NUM_THREADS: [4]
  MPI_NP: [4]
```

For SLURM jobs, variables are exported in the batch script. For local tests, variables are set in the subprocess environment.

### 7.4 Runtime Execution Keywords

Execution is driven by two orthogonal tri-state keywords
(`multitask` / `remotetask`, values None / True / False) plus the
environment fields below (see `WORKFLOW_DEVELOPMENT.md` Section 11).

Modularized keyword sources (not read from a global environment):

- **Workflow level**: the workflow's own keyword section (`&adaptive`,
  `&md`/dynamics) carries `multitask` / `remotetask`.
- **Base level**: the skill keyword sections of the skills listed
  in `&control` theory/reference (`&pyscf`, `&molcas`, `&bagel`, ...)
  carry `multitask` / `remotetask` / `multipool`. Each base node
  (one per role: `base.theory`, `base.reference`) aggregates
  `multitask` and `multipool` across the tasks of that role through
  the tri-state rule (all None → default; {True, None} → True;
  {False, None} → False; {True, False} → conflict error; conflicting
  non-default `multipool` values → `KeyError`). Roles are independent,
  so theory and reference flags never conflict with each other.
  Universal rule 3: planning completely ignores the skill-section
  `remotetask` keyword (base nodes are always local); the
  keyword is reserved for the skill itself to decide its own local run
  or slurm run.

Priority chain: declaration dictionaries (stage defaults / node-chain
templates) rank highest — a key present there forces the value;
otherwise the input keyword wins; otherwise the class default applies.
Workflow sections include `&adaptive`, `&md`, `&hessian`, `&search`,
`&optimize`, `&scan` (see `WORKFLOW_DEVELOPMENT.md` Section 11).

`&search` is the one workflow section whose keyword *names* belong to a
skill. Its `grid:` mapping keys hyperparameters by the skill's own vocabulary
(`nn`'s `depth`, `nn_size`, `learning_rate`, ...) and its values are the
candidate lists, so the framework routes the section nested and never
interprets it -- which is also why `Search` sits outside
`FIELD_TO_SUBMODEL` alongside `Hessian` (one flat name could otherwise
collide with a keyword of another skill).

| Keyword | Type | Default | Meaning |
|---------|------|---------|---------|
| `grid` | mapping of skill keyword to a list | empty (refused) | the candidate values; the scan is their Cartesian product |
| `retrieve` | int | `0` | rank the folders a previous scan left, submit nothing (legacy `retrieve`) |
| `sort_by` | str | the trained profile's first channel | channel the points are ranked on |
| `sort_stat` | `max` / `mean` | `max` | worst member or average member (a misspelling is refused, never defaulted) |
| `sort_order` | `asc` / `desc` | `asc` | smaller error first by default |
| `grid_root` | path | `./grid-search` | root of the point directories |
| `max_grid` | int | `0` (no limit) | refuse a scan larger than this many points |
| `multitask` / `remotetask` / `multipool` | tri-state / int | class defaults | the dispatch axis; legacy `use_hpc 0` is multitask without remotetask, `use_hpc 1` is both, and legacy `ml_ncpu` is `multipool` |

Classic spelling puts the whole grid on one keyword and separates the entries
with `;` (`grid depth 1 2; nn_size 8 16`); the same keywords written on
separate `grid` lines are equivalent now that a repeated key keeps both values
(section 5.2a). A name written twice or an entry without candidates is refused
by `schema.adapter._parse_search_grid` rather than resolved by order; a
candidate whose *type* is wrong for its keyword is read as text and refused by
the skill that declared the keyword, because that is where the type table lives.

`&train` and `&prediction` carry the dispatch layer of the two
machine-learning nodes and nothing else, and are rendered by the same
converter path (`_build_workflow_section_pairs`). Every framework workflow
section the classic text can state is refused rather than dropped when it
carries a keyword the node has no field for.

Whether a keyword may appear in `grid:` at all is the skill's declaration
(`manifest.yaml: searchable`), read through its query interface, and a
candidate is checked by the skill that would have to train with it. See
`dev/methods/SEARCH_DEVELOPMENT.md` sections 2 and 3.1.

Environment snapshot fields (travel with `state.pmd.environment`):

| Field | Default | Meaning |
|-------|---------|---------|
| `shell` | `''` | Raw shell commands (module loads, ...) copied verbatim into the main framework dispatch scripts; the global `environment.shell` serves only the main framework scripts, while skills consume only their own skill-level shell keyword (#134). Classic input `&environment shell:` is single-line: separate multiple commands with `;` |
| `slurm` | merged section | Slurm resource section, used when `remotetask=True` |
| `scratch_path` | `''` | Where the run works: `pmd.py` opens the framework's `<title>.log` here, and a skill without its own scratch makes its calculation directories under it. Empty means the launch directory (#306) |

Machine resources are not `&environment` keywords. A thread count and a
rank count belong to the calculation that spends them, so they are written
in the skill section as `<skill>_thread`/`<skill>_nproc` (the framework's
spelling) or plain `thread`/`nproc` (the skill's own), and the skill keeps
the larger of a pair -- see `SKILL_APP_DEVELOPMENT.md` section 5.7 (#306).
Both spellings are **refused in a workflow section**
(`schema.adapter._reject_workflow_resource_keywords`): the
`md`/`optimize`/`adaptive`/`hessian`/`search`/`composite` sections
parallelise with processes (`multitask`/`multipool`), never with threads or
ranks inside one engine, so `md: thread 5` would otherwise be a setting
that reads as configured and is dropped -- the silent-drop class of #299
and #300.

Retired keys of this section — refused on input rather than absorbed
(`schema.config.RETIRE_ENVIRONMENT_KEYS`, #300):

| Key | Where it belongs instead |
|-------|---------|
| `execution_mode` | the **skill** section (`&pyscf`, `&lammps`, ...). The framework spells its own decision in `multitask`/`remotetask`/`multipool`; `data.core._build_task_queue` hands the resulting mode to each skill so exactly one component submits |
| `workflow_execution` | `multitask`/`remotetask`/`multipool` on the workflow or skill section (#299) |
| `workflow_workers` | `multipool` on the workflow section, read by `Scheduler._worker_count()` (#66, #300) |
| `num_thread` | the skill section, as `<skill>_thread` (or plain `thread`); the run-wide floor it served (#303) is cancelled, so the framework keeps no thread setting at all (#306) |

The root runtime builds the execution graph once from these sources and
freezes it into `state.execution_graph`; dispatched worker runtimes only
verify their parent-frozen graph.

### 7.5 A Run Restates the Allocation: the `slurm` Carriers

A run can override any directive of section 7.1 without editing a config
file on disk. Three carriers state the same section, and all three reach the
one snapshot the dispatch layer reads:

| Carrier | Spelling | Route |
|---------|----------|-------|
| YAML example / mapping | top-level `slurm:` block | `interface/parser._parse_yaml` -> `schema.adapter._apply_environment_overrides` |
| Classic input | `&SLURM` section of `keyword value` lines | `interface/parser._parse_text` -> the same adapter hook |
| Agent config | `slurm` entry of the `.config` JSON | `convert_to_pyrai_config` -> the same adapter hook |

`Environment.apply_slurm_overrides` merges **per key** onto the section the
disk files state, in the documented priority order (package < `~/.pyrai3md` <
input), so an input that pins one directive keeps the rest. Three rules of
this section are worth their own sentence because each was once broken:

- **The merge is read even with no `&environment` block.** It sits at the top
  of `_apply_environment_overrides`, not after the guard that returns when the
  input declares no environment section, because an input that states a scan
  and no shell is the normal case. A merge placed behind that guard applied to
  no run that needed it.
- **The merge is for that run alone.** `Environment` is a process singleton
  and its merged config is not rebuilt per input, so the hook rolls back to
  the on-disk state (`reset_input_overrides`) before applying what the current
  input declares. Without the rollback, a second input built in the same
  process would read the first input's directives as its host defaults, and
  `package < user < input` would become `package < user < previous input < input`.
- **The classic text has no boolean type**, so `exclusive false` arrives as the
  word `false` and is read back as `False`; a reader that took any non-empty
  string for truth would reserve a whole node per grid point for a scan that
  asked to share one.

The converter renders the section (`_build_slurm_pairs` / `_build_slurm_section`)
and `slurm` is registered in `_FRAMEWORK_SECTIONS`, so a `slurm:` block survives
`yaml -> json -> .input` instead of being treated as a skill name and dropped.
`qualification/verification/test_environment_snapshot.py::TestSlurmSectionCarriers`
builds the three carriers from one case and asserts the cluster is asked the
same thing by each.

---

## 8. Startup Workflow

The complete execution pipeline integrating three-tier configuration and input processing:

```text
pyrai3md <input>  /  pyrai3md_agent <config>
    |
    v
kernel/environment.py
    |
    +--- Locate package pmd_env.yaml (pyrai3md/qualification/pmd_env.yaml)
    |
    +--- Locate user pmd_env.yaml (~/.pyrai3md/pmd_env.yaml)
    |
    +--- Combine: user overrides package
    |
    +--- Maestro.populate() via skills/query.py discover_all()
    |
    +--- Store skill runtime configs in Maestro
    |
    v
interface.parser / interface.loader
    |   (parser: parse + load coordinates + validate structure)
    |
    v
Raw dict (raw strings, structure validated)
    |
    v
schema.adapter.build_config(raw_config)
    |   Initialize Environment + build Config
    |   Returns config = Config
    |
    v
schema.normalize(config)
    |   Compute derived quantities (decoupled from maestro)
    |
    v
schema.validate(config)
    |   Full consistency checks (decoupled from maestro)
    |
    v
data.core.build_state(config)
    |   PyRAIState construction (static task descriptors)
    |   maestro.resolve_skills(config) → skills
    |   resolve_task_queue: register static task descriptors
    |   (skill_name / api class / capabilities / plain dict config)
    |   backend instantiation is deferred to Calculator.initialize()
    |
    v
workflow.execute(state) — Workflow execution
    |   Creates Calculator(state), resolve_workflow(state)
    |   Returns Workflow instance or calculator.run()
    |
    v
Skill Calculation
```

Key design: `build_state(config)` creates PyRAIState, resolves skills via Maestro and registers **static task descriptors** (skill name, api class, capabilities, plain dict task config) to the state queues. No backend instance is created here; backend instantiation is deferred to `Calculator.initialize()`.

After construction of `PyRAIState`, the execution path is fully unified regardless of input source.

For the detailed orchestration pipeline, see [INTERFACE_DEVELOPMENT.md](INTERFACE_DEVELOPMENT.md).

---

## 9. Directory Structure

The input system spans multiple modules:

```text
pyrai3md/
├── interface/
│   ├── parser.py          # Text/JSON input → raw dict + coordinate loading + structure validation
│   └── loader.py          # Agent JSON config → raw dict
│
├── schema/
│   ├── config.py          # Pydantic models (7 nested sub-models → Config)
│   ├── adapter.py         # Raw dict → typed Config + build_skill_config
│   ├── normalize.py       # Derived quantity computation (maestro parameter)
│   └── validate.py        # Consistency validation (maestro parameter) + Task model
│
├── kernel/
│   ├── environment.py     # Runtime environment initialization
│   └── maestro.py         # Skill coordination: discovery, resolution, config store
│
└── skills/
    ├── query.py           # Dynamic skill discovery querying
    └── wrapper.py         # pyrai3md_skill command entry

qualification/
├── pmd_env.yaml           # Source configuration (distributed with package)
├── manager.py             # Qualification plan + input file generation + SLURM script
├── runner.py              # Qualification execution + result collection
└── converter.py           # Example YAML → config, config.json → legacy text input

~/.pyrai3md/
└── pmd_env.yaml           # User-level customization (to implement)
```

---

## 10. Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| `qualification/pmd_env.yaml` | Implemented | Source configuration, distributed with package |
| `kernel/environment.py` | Implemented | Runtime env bootstrap: config loading + Maestro population |
| `kernel/maestro.py` | Implemented | Skill coordination: discovery, resolution, config store |
| `~/.pyrai3md/pmd_env.yaml` | Implemented | User-level persistent customization (auto-created on first run) |
| `interface/parser.py` | Implemented | Classic text + JSON parsing + coordinate loading + structure validation |
| `interface/loader.py` | Implemented | JSON config loading |
| `schema/config.py` | Implemented | 9 nested sub-models (control/system/dynamics/adaptive/optimize/hessian/search/environment/results) into Config (~160 fields) |
| `schema/adapter.py` | Implemented | Dotted-key expansion + flat-config rejection + env defaults + section flattening + type conversion |
| `schema/normalize.py` | Implemented | Derived quantities + skill_name resolution + jobtype normalization |
| `schema/validate.py` | Implemented | 6-level consistency validation + Task model |
| `pmd.py` | Implemented | Pure orchestration pipeline (338 lines) |
| `agent.py` | Implemented | PyRAI3Agent class with same pipeline as pmd.py |
| `qualification/manager.py` | Implemented | Qualification plan + input generation + SLURM |
| `qualification/certification/converter.py` | Implemented | Example YAML → config, JSON → legacy text conversion |

---

## 11. Summary

The PyRAI3MD input system is:

- **Three-tier configuration** — Package → User → Runtime, with clear priority ordering
- **Environment-driven** — Configuration comes from `pmd_env.yaml` layers with runtime overrides
- **Schema-driven** — All input formats converge to the same typed Config model
- **build_state combines skill registration** — `build_state(config)` creates PyRAIState, resolves skills via Maestro and registers static task descriptors to the state; backend instantiation is deferred to `Calculator.initialize()`
- **Skill-extensible** — Skills own their parameter validation and schema extensions; Maestro auto-populates via query mechanism on startup
- **Agent-compatible** — JSON config is the universal task description format
- **Backward-compatible** — Legacy text input supported through converter

Configuration resolution:

```text
Package Environment  (pyrai3md/qualification/pmd_env.yaml)
        +
User Environment     (~/.pyrai3md/pmd_env.yaml)
        +
Runtime Input        (input file / agent config)
        |
        v
kernel/environment.py (load configs + Maestro.populate)
        |
        v
schema/adapter.py (env defaults + flatten + type conversion)
        |
        v
schema/normalize.py (compute derived quantities, decoupled from maestro)
        |
        v
schema/validate.py (full consistency checks, decoupled from maestro)
        |
        v
data/core.build_state (PyRAIState + static task descriptors)
        |   (backend instantiation deferred to Calculator.initialize())
        v
Skill Execution
```

The connection to the test system:

```text
pmd_env.yaml (configuration)
        |
        v
manager.py (generates input files)
        |
        +--- skill generator → config.json
        |
        +--- converter.py → legacy .input
        |
        v
runner.py (executes tests via 3 modes)
        |
        v
pyrai3md / pyrai3md_agent / pyrai3md_skill
```

This design avoids duplicated input mechanisms and allows PyRAI3MD to evolve from a traditional simulation program into an AI-native computational framework.
