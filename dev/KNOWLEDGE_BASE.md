# PyRAI3MD Knowledge Base

Last updated: 2026-11-12 19:20. Part 17 is new (Q85-Q94): the pre-commit review contracts - lazy skill export, per-skill read_support, the pure optimizer/fix-mask declarations, the requested-block raise, truthful qualification verdicts, the shared pytest switch, the remote plan coverage, test placement, dispatch/wrapper hardening and the import-free schema/reference. Earlier: 2026-11-06 13:15. Part 16 is new (Q79-Q84): the reference partition is a plan of its own and the label source is derived from atom-set equality rather than from list lengths; what the framework offers versus what a skill must consume; why the MD stage and the reference stage are submitted by two different owners, including the recursion that turned one adaptive round into hundreds of PySCF jobs and where the fix lives; when `aux-labels.json` exists and who reads it; the three classes of parent-sized carrier that must be stripped before an adaptive child runs; and the sixth example family switch. The readings quoted there are the local carriers of `examples/adaptive_*.yaml` and round `tmp_20261106_102745`, plus `tools/verify.py --selftest`. Parts 1-15 were not re-audited. Earlier: 2026-11-02 22:42 (#449/#450: Part 14 gained Q72 - how many RASSCF passes a deck holds is one function asked by builder and reader, with `&ESPF` per pass - and Q73 - a `(nstate, natom)` charge row is the state the task ran, and an out-of-range state writes no row. Q71 and earlier were not re-read. Before it, 2026-11-02 10:55 (#448: Part 14 gained Q71 - five skills sorted a state list the framework had already numbered, which is two index spaces with nothing between them, and the reading that falsified this file's own premise: the declaration does reach the skill, so the defect was entirely on the receiving side. Q69 and Q70 were not re-read. Before it, 2026-11-01 21:20 (Part 14 gained Q69 and Q70 from #447: one derivation for the coupling axis - five implementations, no consistency check, and an adjacent-pair default that made "nothing declared" unreachable - and the zero-fill family cleared from six readers, where a row now exists only where the engine printed one. Q68 and the rest of Part 14 were not re-read. Before it, 2026-11-01 11:00 (Part 14 gained Q68 from #446: the per-step `nac_pairs` field had no consumer in any skill backend, a crossing search never spelled the `nac` token, and the Molcas reader answered a missing coupling table with a row of zeros - so `meci` reported a measured `Coupling Norm: 0.00000000` for a quantity the engine was never asked to compute. The rule recorded there is that an absent quantity travels as absence, and a new payload on a shared contract names its consumer in the same change. Q65 and Q66 in the same Part were rewritten: the ESPF readback now returns one row per state and the framework picks the row. Earlier, 2026-10-31 23:20 (Part 14 added from #445, checked against the code and against round `tmp_20261031_220625`: Q64 - charge ownership is derived from the whole task queue, and the two-phase defect was invisible in every energy because the embedding is resolved by a different predicate; Q65 - the ESPF readback needed three separate corrections (the gate named a value the schema refuses, block selection is bound to the `rasscf` section and the root index, then the fit residual is averaged), with the deployed printout `-0.5994 / 0.2997 / 0.2997` traced back to its `Charge on O1 = -0.5990`, `Total ESPF charge = 0.0012` block; Q66 - `ewald_accuracy` and the four mesh routes falsified on the deployed build; Q67 - a resident deck must not hold a dump it never reads. The guard note is measured, not argued: 3.9e-6 absolute residual on h2o/pppm against a 1.3e-4 budget, while the ewald box leaves 2.2e-4, so `1e-6 * scale` fails by 1.3x and the guard stays at `1e-4 * scale`. Parts 1-13 were not re-audited)。此前 2026-10-24 07:40 (the ML single-call body moved from `workflow/ml_single.py` to `workflow/single.py::MLSingleShot`; it remains a separate `Workflow` subclass, not a `SingleShot` subclass. Earlier 2026-10-22 21:20 (the cell and periodic design material now lives in `CELL_OPT_DEVELOPMENT.md`; the optimizer document points there, the PBC document names the division of labour, and A5/D-14 reads as closed for bond, angle and torsion; Q39's serialization API now names `save_state`/`load_state` in `data/runtime/serializer.py`). Earlier, 19:10 (documentation audit: the switchboard entry now carries the thirty-two-case path round and its measured postures - 500 planned / 213 closed / 290 printed here, 291 in the test environment, `_runall: true` 745 rows here and 784 on the repository `.venv`, `_runall: false` 435 / 265 / 173; the "`optimize_cell`, unwired for every search" claim was corrected to the two refusals the cell gradient actually makes; and a reference to a deleted local evidence script now says so). Before it, 2026-10-19 16:10 (#393 gave the agent-config question the fourth merge source of `_merge_one_skill_config` - the converter's `skill_sections` key, which had no consumer - checked against `schema/adapter.py:3054-3079`, `qualification/certification/converter.py:343-375` and round `tmp_20261019_122659`'s own `input/optimize_cell_meci_lammps.json` on this checkout. The pass before it (2026-10-19 12:10) had the qualification switchboard re-checked against `manager.EXAMPLE_FAMILIES`, `certification/planner.py`, `runner.run_tests` and the shipped `pmd_env.yaml` for #392 (noted 2026-10-20: the `_example_timeout` map and `manager._resolve_example_timeout` audited there were deleted by ruling, leaving `_timeout` as the only budget): the three postures, measured through the probe `tools/deploy.py` sends to the server, are **445 planned / 197 closed / 248 working / 251 printed** for the optimize round, **676 rows with 670 working** under `_runall: true` on this TensorFlow-less interpreter (**715 printed** on the repository `.venv`, example_tests 190 and skill_tests 400), and **399 / 232 / 167**, printing 170, under `_runall: false`. The family table now reads twenty-four `optimize_*` cases with sixteen `optimize_cell_*` among them over 96 qualified YAMLs (`md_lammps_` 12, `constraint_md_` 33, no overlap), the LAMMPS certification is eleven cases, and the `examples` section ships 21 `true` keys against 76 `false` and two `condition`. The 439/197/242/245, 670 and 396/229/167 readings over 93 YAMLs and 18 `true` belong to the file as #391 left it. Rounds 15, 16 and 17 are collected with their readings in CHANGELOG.md #387-#391; #392 ran no round (the deployment is offline) and instead landed the isotropic zero-pressure twin, two crossing legs whose force-field environment carries the box - both executed here, 28 s and 27 s for five steps - the periodic-engine Hessian refusal, the deploy poll that no longer reads an outage as silence, and the three `verify.py` gaps. **A full-round figure is a capability reading, not a constant**: on this interpreter the four `nn_*` examples plan as named skips because TensorFlow is absent, which is also why four `test_manager` criteria about the NN chain fail under that interpreter and pass under `.venv`. The rest was not re-audited))

This document provides a comprehensive Q&A reference for PyRAI3MD architecture, runtime modes, skill design, testing strategy, engine ownership channels and resident execution, the machine-learning workflow nodes and the grid scan, and common development pitfalls.

---

## Part 1: Architecture

### Q1: What is PyRAI3MD?

PyRAI3MD is a modular scientific computing framework for molecular dynamics and quantum chemistry. It orchestrates computational tasks across multiple backend software (PySCF, OpenMolcas, BAGEL, etc.) without embedding any backend-specific knowledge in its core.

### Q2: Why a modular architecture instead of a monolithic program?

**Answer:**
1. **Backend independence**: Quantum chemistry software evolves independently. A modular architecture allows adding/removing backends without modifying the core.
2. **Capability discovery**: Different backends support different methods (DFT, CASSCF, CASPT2, etc.). Maestro-based discovery enables dynamic matching of tasks to capabilities.
3. **Parallel development**: Skills can be developed, tested, and deployed independently by different teams.
4. **AI agent compatibility**: External AI agents can discover and invoke capabilities through standardized interfaces.

### Q3: What are the five core design principles?

**Answer:**

1. **Core owns protocols, not scientific knowledge** — The core framework defines data types, communication patterns, and workflow orchestration. Scientific knowledge (method parameters, basis sets, active spaces) belongs to skills.

2. **Skills own domain knowledge and execution** — Each skill independently manages its backend-specific logic, from input generation to output parsing.

3. **Maestro is the only capability discovery mechanism** — No module should directly import a specific skill. All capabilities are obtained through the Maestro.

4. **Workflow composes capabilities, not software** — Workflows request abstract capabilities (energy, gradient, NAC) rather than specific software commands.

5. **Internal execution uses Python objects, external interaction uses schemas** — Python objects remain in memory during computation. Serialization happens only at boundaries (input/output, inter-process communication).

### Q4: Why is qualification inside pyrai3md/ instead of at the root level?

**Answer:**
The qualification infrastructure is placed inside `pyrai3md/qualification/` to ensure it is included in the Python package during `pip install`. This allows `pyrai3md test` to work correctly after installation, as the manager, runner, and verification tests are all accessible as part of the installed package.

### Q5: What is the role of each top-level module?

**Answer:**

| Module | Responsibility |
|--------|----------------|
| `interface/` | Input parsing (text/JSON → raw dict). No computation, and no section semantics: it splits lines into `&section` and `keyword value` pairs for every section alike (Q47) |
| `schema/` | Configuration building and validation (raw dict → typed Config). |
| `planning/` | Capability resolution (jobtype → capability name strings). |
| `workflow/` | Workflow orchestrators (AIMD, optimize, the path family MEP / IRC / NEB, frequency) and Calculator. |
| `data/` | Runtime state objects (PyRAIConfig, PyRAIState, Result, build_state). |
| `kernel/` | Maestro, plugin management, runtime context, logging. |
| `skills/` | Skill plugins (pyscf, molcas, bagel, orca, xtb, openqp, lammps, cp2k, qdm, nn_pes -- ten in all, the last two model engines rather than external codes) with independent implementations. |
| `qualification/` | Qualification planning, execution, and reporting infrastructure. |
| `utilities/` | Shared generic functions (IO, filesystem, units, timer). |

---

## Part 2: Runtime Modes

### Q6: What runtime modes does PyRAI3MD support?

**Answer:**
PyRAI3MD supports three command-line entry points:

| Mode | Command | Input Format | Description |
|------|---------|--------------|-------------|
| `pyrai3md_input` | `pyrai3md <file>.input` | Classic text or YAML | Traditional CLI through full schema pipeline |
| `pyrai3md_agent` | `pyrai3md_agent <file>.config` | Sectioned dict JSON (molecule section carries System fields + full-name skill section) | Agent-oriented CLI, bypasses classic parser |
| `skill_agent` | `pyrai3md_skill <file>.json` | Flat skill JSON (all parameters skill-owned) | Skill-independent execution via wrapper |

### Q7: Why three modes instead of one?

**Answer:**

Each mode serves a distinct purpose in the development and deployment lifecycle:

1. **pyrai3md_input (Classic mode)**:
   - **Purpose**: Validates the full pipeline from text input to result.
   - **Tests**: Converter → Schema → State construction → Workflow → Backend.
   - **Use case**: End-to-end integration testing, legacy compatibility.

2. **pyrai3md_agent (Agent mode)**:
   - **Purpose**: Validates the agent interface and JSON config processing.
   - **Tests**: JSON loader → Schema → State construction → Backend.
   - **Use case**: AI agent orchestration, modern workflow integration.

3. **skill_agent (Skill mode)**:
   - **Purpose**: Validates skill independence — can the skill run without PyRAI3MD core?
   - **Tests**: wrapper.py → skill's own agent.py → backend.
   - **Use case**: Standalone skill deployment, component-level testing.

### Q8: Why does `pyrai3md_skill` need `skill_path` in the config?

**Answer:**
The `pyrai3md_skill` command invokes `wrapper.py`, which is a generic entry point that doesn't know which skill to load. The `skill_path` field tells the wrapper which skill's `agent.py` module to execute:

```bash
python -m <skill_path>.agent <config.json>
```

Without `skill_path`, the wrapper cannot locate the skill's agent module. This is why all generators must include `skill_path` in their output configs.

**Note**: The `skill_path` in the config is the full framework path (e.g., `pyrai3md.skills.pyscf_skill`). This is different from the skill-relative path in `manifest.yaml` (e.g., `backend`). The framework resolves the full path at runtime from the skill-relative manifest value.

### Q9: Why do Pydantic schemas need `skill_name` and `skill_path` fields?

**Answer:**
When a config dict passes through Pydantic validation (via `Config(**data)`), any fields not declared in the schema are silently dropped. If `skill_name` and `skill_path` are not declared in the `Environment` model, they disappear after validation, breaking downstream components that need them.

**Solution**: Each skill's `schema/config.py` must declare these fields in the `Environment` class:

```python
class Environment(BaseModel):
    skill_name: str = 'xxx_skill'
    skill_path: str = 'backend'   # Skill-relative default, no framework prefix
    # ... other fields
```

**Important**: All 10 skills declare `skill_path: str = 'backend'` as the
default (skill-relative, no framework prefix). The framework resolves the
full path at runtime (see SKILL_APP_DEVELOPMENT.md Section 6.1).

### Q10: What is the data flow from input to result?

**Answer:**

```
Input file (.input or .config)
    ↓
Interface (parser.py or loader.py)
    ↓
Raw dictionary (strings, lists)
    ↓
Adapter: build_config (flatten, type conversion, Config object)
    ↓
Normalize (derive missing quantities: electrons, state_spin, pair maps)
    ↓
Validate (profile/basis compatibility, resource checks)
    ↓
Build State: build_state (create PyRAIState, resolve skills, build task queues)
    ├── maestro.resolve_skills(config) → skills dict
    ├── resolve_task_queue(config, skills) → task descriptors (composite expansion)
    ├── For each task: create task PyRAIConfig (no geometry slicing)
    ├── Record the backend api path (task['api']); no backend instance yet
    └── Register tasks to state.theory/reference task queues
    ↓
Execute: execute(state) → ExecutionGraphBuilder.build() freezes the
    execution graph into state.execution_graph → Calculator.initialize()
    instantiates backends → Workflow(calculator).run() / calculator.run()
    → Dispatcher.run() → backend.calculate() per task
    ↓
Result object (energy, gradient, nac, soc, etc.; composite results aggregated)
```

---

## Part 3: Skill Design

### Q11: What is the directory structure of a skill?

**Answer:**

```
<skill_name>/
├── __init__.py
├── agent.py              # CLI entry (python -m <skill>.agent)
├── backend.py            # Public API: validate, build_state, calculate, local_run, slurm_run
├── manifest.yaml         # Lightweight discovery metadata
├── query.py              # Self-contained registry queries
├── skill_guide.md        # Skill-specific usage guide
├── schema/
│   ├── __init__.py
│   ├── config.py         # Pydantic models (System, Properties, Calculation, Environment)
│   ├── method.py         # AVAILABLE_METHOD, PROFILE_METHOD_MAP, PROFILE_STEP_MAP
│   ├── reference.py      # Constants, defaults, periodic table
│   ├── adapter.py        # External task → skill config conversion
│   ├── normalize.py      # Derive missing quantities
│   └── validate.py       # Consistency and compatibility checks
├── service/
│   ├── builder/          # Workflow graph → dispatch calculation
│   ├── workflow/         # Profile → WorkflowGraph compiler
│   ├── runner/           # Local and SLURM execution
│   ├── parser/           # Backend output → standardized results
│   └── templates/        # Script and parameter templates
└── tests/
    ├── *.yaml            # Test case definitions
    └── test_*.py         # Skill-owned offline unit tests (parser/builder)
```

### Q12: Why does each skill need its own `agent.py`?

**Answer:**
The `agent.py` module serves as the skill's independent CLI entry point. When `pyrai3md_skill` invokes a skill, it executes:

```bash
python -m <skill_path>.agent <config.json>
```

The agent module:
1. Reads the config file path from command-line arguments
2. Instantiates the backend
3. Calls `local_run()` or `slurm_run()` based on execution mode
4. Reports success/failure status

Without `agent.py`, the skill cannot be invoked independently by the wrapper.

### Q13: What methods must a backend provide?

**Answer:**

| Method | Purpose | When Called |
|--------|---------|-------------|
| `validate(config)` | Validate config against skill schema | During initialization |
| `__init__(pyrai_config)` | Initialize backend with PyRAIConfig | During Calculator.initialize() |
| `calculate(atoms, coords)` | Lightweight calculation step | During MD trajectory |
| `local_run()` | Execute full calculation locally | Single-point tests |
| `slurm_run()` | Execute via SLURM scheduler | HPC cluster tests |

**Note**: Backend `__init__` receives a task-specific `PyRAIConfig` object (framework submodels + merged skill submodels), not a PyRAIState. This ensures backends only see their own configuration.

**Input-file immutability** (#143): in standalone mode (JSON file path), the backend never writes the validated config back to the input file. The resolved config stays in memory; the local runner receives the in-memory config object and the slurm runner serializes its own copy into the calculation directory. Keeping the input read-only makes `pyrai3md_skill <file>.json` scripts idempotently re-runnable. Readers stay tolerant of the legacy nested layout the old write-back produced: `wrapper.resolve_skill_name` falls back to `environment.skill_name`, and the converter `load_config` flattens the five legacy sections (#142).

**Result return channels**: the returned result dict keys align with the `Properties` field names. Skills performing internal iteration (optimize/md) report the final structure through the `fin_coord` transport channel ((natom, 3) Angstrom); the framework consumes it, updates the system coordinates, and pops the key so it never reaches `state.results`. All 10 skills additionally reserve the `fin_velocity` channel ((natom, 3) Bohr/au, empty when there is no velocity output); the framework consumes it through the same channel contract as `fin_coord` — a non-empty value becomes `system.velocities` and the kinetic energy is recomputed — and currently only the lammps MD profiles return real final velocities. A third channel carries the cell: a skill whose engine owns the cell degrees of freedom (LAMMPS `fix npt`, a future CP2K `CELL_OPT`) reports the box its returned coordinates actually live in through `fin_cell`, as either the six cell parameters or the nine flattened lattice components -- the two shapes `schema.normalize` already accepts for `cell` itself -- and the framework derives `cell`, `lattice_vector` and `volume` from it together. A value for a system declared non-periodic is refused: the engine did run on a cell the input never declared, which is an input error rather than a state to absorb. See SKILL_APP_DEVELOPMENT.md Section 11 for the full contract.

### Q14: How should a backend `__init__(pyrai_config)` be implemented?

**Answer:**

Each backend receives a task-specific `PyRAIConfig` during `build_state`. The PyRAIConfig provides:
- Framework submodels — `control`, `system`, `dynamics`, `adaptive` (typed Pydantic models)
- Skill-specific submodels — dynamically merged from the skill schema by `build_pyrai_config()`
- `results` — dict field where the backend writes back its raw calculation results

```python
class MyBackend:
    def __init__(self, pyrai_config):
        self.config = pyrai_config       # task-specific extended PyRAIConfig
        # Initialize internal state from config...

    def calculate(self, atoms, coordinates):
        results = ...                    # run calculation
        self.config.results = results    # write back (skill schema)
        return results
```

**Skill independence rule**: The skill must NOT import from `pyrai3md` modules. All configuration arrives through the PyRAIConfig interface.

The backend initialization is handled by the framework's `build_state()`:
1. Resolve skills via Maestro.resolve_skills(config)
2. Expand composite workflows into task queues via resolve_task_queue()
3. Create one PyRAIConfig per task and instantiate backend(task_config)
4. Register each task to the state.theory/reference task queues

Calculator later iterates the task queues via `state.iter_tasks()`.

### Q15: Why does PySCF need `initialize_method_classes()`?

**Answer:**
PySCF is a Python-native library where method classes (RHF, UKS, TDDFT, etc.) must be imported at runtime. The `AVAILABLE_METHOD` dictionary stores these class references, but they start as `None` to avoid importing PySCF at module load time.

The `initialize_method_classes()` function populates these references:

```python
AVAILABLE_METHOD['dft']['reference'] = (dft.RKS, dft.UKS)
AVAILABLE_METHOD['tddft']['tddft'] = tddft.TDDFT
```

This function is called in `dispatch_calculation()` before any computation is dispatched. Without it, `set_method()` fails with `TypeError: cannot unpack non-iterable NoneType object`.

### Q16: Why are test YAML files structured with `input` and `expected` sections?

**Answer:**

```yaml
input:
  atoms: [C, C, H, H, H, H]
  coordinates: [...]
  profile: casscf_energy
  method: casscf
  # ... calculation parameters

expected:
  energy:
    value: null
    tolerance: 1e-6
  gradient:
    shape: [6, 3]
```

- **`input`**: Contains all calculation parameters passed to the backend.
- **`expected`**: Defines validation criteria for results. `value: null` means "skip value check" (used when reference values are unavailable). `shape` checks array dimensions.

This separation allows the same test case to be used for:
1. Generating config files (from `input`)
2. Validating results (against `expected`)
3. Documentation (showing what each test verifies)

### Q17: Why use C2H4 (ethylene) as the standard test molecule?

**Answer:**

C2H4 is chosen as the universal test molecule for the QM skills because:

1. **Moderate size**: 6 atoms, 16 electrons — large enough to test active space methods but small enough for fast calculations.
2. **Multi-reference character**: The π/π* orbitals require CASSCF treatment, testing active space functionality.
3. **Symmetry**: D2h symmetry provides predictable degeneracies and selection rules.
4. **Standard active space**: 2 active electrons in 2 active orbitals (π, π*), with 7 inactive orbitals — a well-defined test case.
5. **Consistency**: Using the same molecule across the QM skills (pyscf, molcas, bagel, orca, openqp, xtb, cp2k) enables cross-validation of results. The cp2k tests carry an explicit 10 Å periodic cell (lattice vectors) since CP2K is a periodic code.

The force-field skill (lammps) cannot use C2H4 with its shipped force-field inputs; it uses the `h2o.data` water box (60 atoms, periodic) instead.

### Q18: What is the purpose of `state_spin` and how does it relate to NAC/SOC?

**Answer:**

`state_spin` is a list of 2S values (spin quantum number × 2) for each electronic state:
- `0` = singlet (S=0, 2S=0)
- `2` = triplet (S=1, 2S=2)

The normalize module uses `state_spin` to automatically classify state pairs:

| Pair type | Condition | Physical meaning |
|-----------|-----------|------------------|
| NAC pair | `state_spin[i] == state_spin[j]` | Same-spin coupling (non-adiabatic) |
| SOC pair | `state_spin[i] != state_spin[j]` | Different-spin coupling (spin-orbit) |

Example: `state_spin = [0, 0, 2]` (2 singlets + 1 triplet)
- NAC pairs: `[[0, 1]]` (singlet-singlet)
- SOC pairs: `[[0, 2], [1, 2]]` (singlet-triplet)

For SOC calculations, you **must** include at least one triplet state (`state_spin` contains `2`) to generate SOC pairs.

---

## Part 4: Testing Strategy

### Q19: What is the testing architecture?

The transcript itself is part of the contract: the header is flushed line by line after stdout is switched to line buffering (a block-buffered stdout was what put the banner *below* the first task lines in a redirected log), each task owns fds 1/2 while it runs so a child's stray output cannot interleave with its own block (dropped on PASS, kept on FAIL), unit tests print one `[UNIT] ran=/failures=/errors=` line, and a PASS block may only print qualifying notes (`[ENERGY CONSERVATION]`, `[NPT STABILITY]`, ...) -- so a verdict that did not judge says so instead of passing silently.

**Answer:**

```
pyrai3md test
    ↓
qualification.runner.cli_entry()
    ↓
manager.load_config() → pmd_env.yaml
    ↓
manager.generate_plan()
    ├── Main tests (qualification/verification/test_*.py)
    ├── Example tests (pyrai3md/examples/*.yaml, two modes:
    │   pyrai3md_yaml / pyrai3md_input; no agent mode)
    └── Skill tests (per skill, per profile)
        ├── pyrai3md_input mode (.input via converter)
        ├── pyrai3md_agent mode (sectioned .config via converter)
        └── skill_agent mode (flat .json from the certification converter, pyrai3md_skill wrapper)
    ↓
runner.run_tests()
    ↓
Report (pass/fail/skip with tracebacks)
```

### Q20: Why three test modes for each skill?

**Answer:**

Each mode tests a different aspect of the system:

1. **pyrai3md_input**: Tests the full classic pipeline including converter.py. Catches issues in text input parsing and schema conversion.

2. **pyrai3md_agent**: Tests the modern agent interface. Catches issues in JSON config processing and state construction.

3. **skill_agent**: Tests skill independence. Catches issues in standalone execution (missing agent.py, incorrect skill_path).

This triple-mode approach ensures:
- Backward compatibility with classic input
- Forward compatibility with agent workflows
- Skill portability (can run without PyRAI3MD core)

### Q21: Why does the generator need to set `scratch_path`?

**Answer:**

The `scratch_path` determines where temporary calculation files are written. During testing, it should point to the test working directory (e.g., `tmp_20260730/pyscf_skill/tmp`) rather than the project root or site-packages.

If `scratch_path` is not set correctly:
- Temporary files pollute the source directory
- Permission errors occur when installed in site-packages
- Parallel tests may overwrite each other's files

The certification converter sets `scratch_path` in `generate_test_configs()`:

```python
scratch_dir = os.path.join(output_dir, 'tmp')
os.makedirs(scratch_dir, exist_ok=True)
config['scratch_path'] = scratch_dir
```

Since #140 the qualification agent config (`.config`) is the converter's sectioned dict serialized as JSON: `scratch_path` travels inside the full-name skill section (e.g. `xtb_skill`), which `_merge_one_skill_config` merges as the highest-priority source, so agent mode places temporary files under `scratch_path` exactly like input mode. Multi-backend agent configs must carry one skill section per backend to keep these parameters distinguishable. The example converter's own flat `.json` is the third dialect and it needed a fourth source: it keeps every skill's block under `skill_sections` keyed by the stem (two skills spelling `method` cannot share one flat namespace) and flattens only the first `control.theory` entry to the top level, so before #393 read `skill_sections[<stem>]` into the ladder the second backend could only ever be handed the first one's `method` - measured on round `tmp_20261019_122659`'s own `input/optimize_cell_meci_lammps.json`, where the LAMMPS task resolved to `ff_energy` on a jobtype that asks for a gradient. Since #141 the System fields (atoms/coordinates/velocities/charges/multiplicity/nstate/state_spin) live in the molecule section of the `.config` per the System submodel convention; `_flatten_sections` routes them into the System submodel and `_merge_one_skill_config` reads the state keywords from the molecule section as well.

### Q22: What is the role of `pmd_env.yaml`?

**Answer:**

`pmd_env.yaml` is the single source of configuration for qualification runs and deployments:

```yaml
deploy:           # Remote server connection
environment:      # Runtime requirements (python, conda, scheduler)
slurm:            # SLURM job parameters
skills:           # Per-skill environment settings
  pyscf_skill:
    test: true    # Enable/disable automated testing
  molcas_skill:
    molcas_path: /opt/openmolcas
    test: false
```

Key design decisions:
- **Single file**: All configuration in one place, easy to share and version.
- **No passwords**: SSH key authentication preferred. Sensitive data never stored.
- **Test switch**: `test: true/false` controls which skills are included in automated testing.

### Q23: How does remote deployment work?

**Answer:**

`tools/deploy.py` handles remote deployment and testing; `tools/verify.py` is the post-round verifier over a finished work directory (A records vs `results.json`, B step log vs frame record, C engine output vs recorded value, `--selftest` as the planted-error control):

```
Local → rsync source → rm -rf build + pip install → pyrai3md test (remote, background)
         (its log header "Total tests: N" is the only progress denominator)
                                         ↓
Local ← scp results ← poll status.json ← Remote execution
```

Key features:
- **Jump host support**: connects directly first and, when that fails with
  `deploy.jump_host`/`jump_username` configured, retries through the jump
  (bastion) host using `ssh -J` / `scp -J`.
- **Upload exclusions**: rsync never uploads `.git/`, `__pycache__/`,
  `*.egg-info/`, `.idea/`, `.vscode/`, `legacy/`, `skill_demo/`, and the
  local run artifacts `tmp_*/`, `test_*.log`, `build/`, `*.egg`.
- **Synchronized trigger**: the remote trigger runs
  `touch .deploy_trigger_marker`, then launches the tests with `nohup`,
  then waits with a `find . -newer <marker> -name 'tmp_*'` loop for the
  timestamped work directory created by this run (leftover directories
  from previous runs can never be reported); it returns
  `(success, work_dir)` and polling falls back to discovery when the
  directory does not appear within the wait window.
- **Process survival**: Uses `nohup` to prevent SSH disconnect from killing tests.
- **Idle timeout**: Detects hung processes (300s no response) without interrupting long calculations.
- **Guaranteed exit**: Always exits automatically, even under total failure.
- **Result discovery**: Finds the latest `tmp_<YYYYMMDD_HHMMSS>` directory (non-timestamped `tmp_*` dirs filtered out), locks it on first hit, polls `status.json` every 30s.

---

## Part 5: Common Pitfalls and Solutions

### Q24: Why does my config lose fields after validation?

**Problem**: Config dict has `skill_name` and `skill_path`, but after backend validation they're gone.

**Cause**: Pydantic drops fields not declared in the schema.

**Solution**: Add the fields to `Environment` in `schema/config.py`:

```python
class Environment(BaseModel):
    skill_name: str = 'xxx_skill'
    skill_path: str = 'backend'   # Skill-relative, no framework prefix
```

### Q25: Why does `set_method()` fail with `TypeError: cannot unpack non-iterable NoneType`?

**Problem**: `AVAILABLE_METHOD[method]['reference']` is `None`.

**Cause**: `initialize_method_classes()` was never called. PySCF method classes are populated at runtime, not at import time.

**Solution**: Call `initialize_method_classes()` in `dispatch_calculation()` before dispatching:

```python
def dispatch_calculation(config, workflow):
    initialize_method_classes()
    # ... dispatch logic
```

### Q26: Why does `pyrai3md_skill` fail with "ModuleNotFoundError"?

**Problem**: `pyrai3md_skill config.json` cannot find the agent module.

**Cause**: Config JSON lacks `skill_path` field, or the skill package is not installed.

**Solution**:
1. Ensure generator includes `skill_path` in output config
2. Ensure skill is installed: `pip install -e .`
3. Ensure skill has `agent.py` at the expected location

### Q27: Why does molcas validation fail with "state_spin does not match multiplicity"?

**Problem**: Molcas normalize raises ValueError for state_spin.

**Cause**: For singlet systems (multiplicity=1), `state_spin[0]` must equal `multiplicity - 1 = 0`. Using `state_spin: [1, 1, 1]` is invalid.

**Solution**: Use correct spin values:
- Singlet states: `state_spin = 0`
- Triplet states: `state_spin = 2`
- Valid example: `state_spin: [0, 0, 2]` for 2 singlets + 1 triplet

### Q28: Why are temporary files created in site-packages?

**Problem**: Calculation temporary files appear in `/usr/lib/python3.x/site-packages/`.

**Cause**: `scratch_path` is not set, defaulting to a relative path that resolves to the installation directory.

**Solution**: Always set `scratch_path` to an absolute path in the test working directory. The generator should create this directory:

```python
scratch_dir = os.path.join(output_dir, 'tmp')
os.makedirs(scratch_dir, exist_ok=True)
config['scratch_path'] = scratch_dir
```

---

## Part 6: Design Rationale

### Q29: Why separate schema into config/normalize/validate?

**Answer:**

This three-stage pipeline mirrors compiler design:

1. **Config** (parsing): Define the data structure. What fields exist? What types?
2. **Normalize** (semantic analysis): Derive missing information. Compute `natom` from `atoms`, `total_electron` from atomic numbers, `state_spin` from `ci` and `spin`.
3. **Validate** (verification): Check consistency. Is the method compatible with the profile? Is the basis set supported? Are resources available?

This separation ensures:
- Each stage has a single responsibility
- Errors are caught at the appropriate level
- Derived quantities are computed once, not repeatedly

A fourth role lives in the same layer: `adapter.py` turns a raw section into the typed shape the
models expect, which is where a classic `.input` section gets its meaning (Q47). #271 also settled
what this layer is **not**: a feature does not get its own schema module. The system constraint
block used to live in `schema/constraint.py` and `schema/legacy_adapter.py`; both are gone, and
their records sit in `config.py` beside `System`, their normalization in `normalize.py`, their
capability refusals in `validate.py` (public names, because `normalize.py` calls the same
functions), and their classic text in `adapter.py`.

One consequence is worth knowing before you look for a name: a function that the layer above calls
is not underscore-private. `validate.check_periodic` is called from `normalize.build_plan`, so the
rule has one implementation whichever pass reaches it first.

### Q30: Why use profiles instead of direct method calls?

**Answer:**

Backend computations are structured as module chains, not single API calls. A "gradient calculation" might require:
1. Build molecule
2. Run SCF
3. Run CASSCF
4. Compute gradient

The profile (`casscf_gradient`) encapsulates this chain. Users specify **what** they want (scientific task), not **how** to do it (backend modules).

Benefits:
- **Abstraction**: Users don't need to know backend internals
- **Portability**: Same profile name works across different backends
- **Extensibility**: New profiles can be added without changing the API

### Q31: Why is the workflow compiled once during initialization?

**Answer:**

Molecular dynamics involves thousands of calculation steps. If workflow compilation happened at each step, the overhead would be prohibitive.

By compiling once during `__init__`:
- Schema validation happens once
- Workflow graph is built once
- Method classes are loaded once
- Each `calculate()` call only updates dynamic information (coordinates, charges)

This design enables efficient nonadiabatic dynamics simulations.

### Q32: Why does each skill need its own `query.py`?

**Answer:**

The query mechanism is the **sole communication channel** between a skill and a frontend framework. It follows a strict entry/exit protocol:

```
Framework (pyrai3md)                    Skill (<skill_name>)
──────────────────                      ────────────────────
skills/query.py                         <skill>/query.py
  │                                       │
  ├─ discover_all() ──────────────────> get_metadata()
  ├─ get_registry() ──────────────────> get_registry()
  ├─ resolve_label() ─────────────────> (via discover_all)
  │                                       │
  └─ adds framework prefix                └─ returns skill-local paths
     at runtime                              (no framework knowledge)
```

**Rules:**
1. `skills/query.py` is the framework's **sole entry point** for skill discovery
2. `<skill>/query.py` is the skill's **sole exit point** for skill information
3. Skills never import framework modules; the framework imports skills
4. All paths in skill query returns are skill-relative
5. The framework prepends its package prefix at runtime

A skill's `query.py` must provide:

```python
def get_metadata():
    """Return {skill_name, label, skill_path, backend_class}.
    All paths are skill-relative."""

def get_registry():
    """Return skill-local registry with relative paths."""
```

The global `pyrai3md.skills.query` delegates to each skill's `query.py` and adds the framework prefix (`pyrai3md.skills.<skill>`) to all paths before exposing them to the rest of the framework.

### Q33: Why use YAML for test cases instead of JSON?

**Answer:**

YAML is preferred for test case definitions because:
1. **Readability**: Comments explain what each test verifies
2. **Conciseness**: Less verbose than JSON for structured data
3. **Human-editable**: Easy to add/modify test cases
4. **Generator output**: The generator converts YAML to JSON configs, maintaining a clear separation between test definition (YAML) and runtime input (JSON)

---

## Part 7: Development Workflow

### Q34: How do I add a new test case for a skill?

**Answer:**

1. Create a new YAML file in `<skill>/tests/`:

```yaml
version: 1.0
name: my_new_test

input:
  atoms: [C, C, H, H, H, H]
  coordinates: [...]
  profile: xxx_profile
  method: xxx
  # ... other parameters

expected:
  energy:
    value: null
    tolerance: 1e-6
```

The test YAML carries only `version`/`name`/`input`/`expected` — there is
no top-level `skill_name` key (the skill is determined by the directory).

2. The certification planner/converter automatically picks up the new YAML file.
3. Run `pyrai3md test` to verify the new test case.

### Q35: How do I add a new skill to PyRAI3MD?

**Answer:**

1. Create the skill directory structure (see Q11)
2. Create `manifest.yaml` with **skill-relative** paths only:
   ```yaml
   name: <program_label>      # e.g. pyscf, cp2k — plain program label,
                              # no label key, no qm. prefix
   backend:
     module: backend          # NOT pyrai3md.skills.<skill>.backend
     class: <Skill>Backend
   units:
     energy: hartree
     gradient: hartree/bohr
   roles:
     qm: <Skill>Backend
     qm2: false
     mm: false
   capabilities: [...]
   profiles: {...}
   ```
3. Implement `query.py` with `get_metadata()` and `get_registry()` (see Q32)
4. Implement `backend.py` with `__init__(pyrai_config)` and `calculate()` (see Q13)
5. Define `schema/` modules (config, method, normalize, validate)
6. Implement `service/` modules (builder, workflow, runner, parser)
7. Create `tests/` with YAML test cases (parsed by the shared
   certification converter; no per-skill generator is needed)
8. Create `agent.py` for standalone CLI execution (see Q12)
9. Add `skill_name` and `skill_path` to `Environment` in `schema/config.py` (see Q9)
10. Register in `pmd_env.yaml` under `skills:`
11. Register the skill resources in `setup.py` `package_data`
    (`manifest.yaml`, `skill_guide.md`, `tests/*.yaml`); guarded by
    `qualification/verification/test_skill_packaging.py`
12. Register the agent module in `skills/tests/test_skill_agents.py` so the
    calculate() forwarding contract (target_states/point_charges) is tested
13. Run `pip install -e .` to make the skill discoverable

**Critical**: All skill code must be independent of `pyrai3md`. Zero imports from `pyrai3md` in any skill file. The framework resolves full import paths at runtime.

### Q36: What should I check when a skill test fails?

**Answer:**

Common checklist:

1. **Config generation**: Does the generator output include `skill_name` and `skill_path`?
2. **Schema fields**: Are all required fields declared in the Pydantic models? Especially `skill_name` and `skill_path` in `Environment`.
3. **Method initialization**: Is `initialize_method_classes()` called (for PySCF)?
4. **Scratch path**: Is `scratch_path` set to the test working directory?
5. **State spin**: For SOC/NAC tests, is `state_spin` correctly configured?
6. **Agent module**: Does the skill have `agent.py`?
7. **Backend init**: Does the backend accept `PyRAIConfig` in `__init__` and extract config/skill_config correctly?
8. **Test switch**: Is `test: true` set in `pmd_env.yaml`?
9. **Manifest paths**: Is `backend.module` skill-relative (e.g., `backend`), not framework-qualified?
10. **Query module**: Does `query.py` provide `get_metadata()` and `get_registry()` with skill-relative paths?

---

## Part 8: Dispatcher and Dispatch Subsystem

### Q37: What is the Dispatcher and how does execute() work?

**Answer:**
`execute(state, runtime_idx=None)` is the unified entry point:

1. **Graph construction (once)**: `ExecutionGraphBuilder.build(state,
   workflow_cls)` freezes the execution graph into
   `state.execution_graph`. Only root runtimes build the graph;
   dispatched workers only verify their parent-frozen graph
   (`resolve_execution_graph=False`).
2. **Construction**: `Calculator(state, runtime_idx)` receives the
   calculator node flags from the graph, then `resolve_workflow(state)`
   resolves the workflow class. The `execute` function itself is
   injected as the in-process executor (`local_executor` /
   `executor_spec`), so the dispatch package never imports the workflow
   module.
3. **Execution**: every workflow and the calculator inherit
   `Dispatcher`; `Dispatcher.run(units, child_jobtype)` is the single
   pipeline — unit materialization followed by
   `Scheduler(multitask)(Launcher(remotetask))`.

Batch workflows hand their task descriptors to `Dispatcher.run()`, which
materializes every task into an independent child unit (deep-copied
state, `child_jobtype` applied, worker label, parent-frozen child graph)
and executes them through the composed Launcher/Scheduler.

### Q38: What execution modes exist and where are they configured?

**Answer:**
The execution mode is fully determined by two orthogonal booleans
(`multitask` / `remotetask`), aggregated from tri-state keywords
(None/True/False). Workflow-level keywords come from the workflow's own
keyword section (`&adaptive`, `&md`/dynamics); calculator-level keywords
come from the skill keyword sections — planning aggregates only
`multitask` and completely ignores the skill-section `remotetask`
(universal rule 3, the calculator node is always local; the keyword is
reserved for the skill itself to decide its own local run or slurm run).

Environment fields travel with PyRAIState (`state.pmd.environment`) —
the dispatch subsystem reads one single data source, no extra
environment variables:

| Field | Values | Meaning |
|-------|--------|---------|
| `slurm` | dict | Resource section for `remotetask=True` (Slurm submission) |

The framework environment section declares **no** execution switch.
`execution_mode` is a skill keyword (each skill's own schema carries it);
the framework spells its decision in `multitask`/`remotetask`/`multipool`
and `data.core._build_task_queue` hands the resulting `execution_mode`
down to the skill, so exactly one component submits. `execution_mode`,
`workflow_execution` and `workflow_workers` written under
`&environment` are refused on input (`schema.config.RETIRE_ENVIRONMENT_KEYS`,
#66 / #299 / #300).

`remotetask=True` maps unambiguously to `slurm_run` — the former
`dispatch_backend` field was removed in #82. `Calculator.run()` forwards
`Dispatcher.run()` with a single-unit list; the default in-process
execution body (`run_in_process`) runs the active role backends, while
a remote launcher relaunches the unit as a child runtime.

### Q39: How do dispatched child runtimes work?

A child that dies mid-`json.dump` used to leave a half-written `result.json`, and the parent reported the reader's complaint (`Expecting value: line 1332 column 14`) as the cause. Two things changed: result documents are now written by `utilities/serialization.py` -- one recursive `json_safe` plus an atomic `write_json` (temp name, fsync, rename) that leaves no partial file -- and both collection errors quote the tail of the child's `slurm-<job>.err` (`slurm_run._child_failure`), so the first thing an operator reads is the child's own failure.

**Answer:**
Dispatched units are serialized PyRAIState runtimes (`.rt` files written
by `save_state` from `data/runtime/serializer.py`), not another input
type:

- The Dispatcher materializes each child state (`materialize_unit`):
  worker label (`state.runtime.worker=True`), `child_jobtype` applied,
  parent-frozen child execution graph, `runtime_idx` suffix — before the
  launch tool writes the `.rt` file and launches the child. Rule 2: a
  worker graph never carries `remotetask=True`.
- Child processes load the state with `load_state()` and re-enter the
  unchanged `execute()` path; backends are rebuilt by
  `Calculator.initialize()`.
- Because workers only verify the parent-frozen graph (never rebuild),
  graph construction happens exactly once at the root — no recursive
  dispatch trees.

Restart uses the same mechanism: `.rt` → `load_state()` → PyRAIState →
standard execution path, bypassing Interface/Schema/Planning.

### Q40: How does `environment.shell` propagate through the system?

**Answer:**
`environment.shell` holds raw shell commands (typically `module load`
statements) in `pmd_env.yaml`. It is reserved for the **main
framework** (#134); skills use their own skill-level shell:

1. **Main framework scripts**: the qualification manager writes
   `environment.shell` into every wrapper — the examples scripts and
   all three skill test modes (`_skill`/`_agent`/`_input`) — after
   `#SBATCH` directives and before `export` (#136).
2. **Skill scripts**: only `skills.<name>.shell` reaches skill
   configs (fill-if-empty, riding the machine-parameter injection).
   Every external-program skill writes its config `shell` into its
   own local/slurm execution scripts before the `export` block. The
   global shell is NOT injected into skill configs.
3. **Dispatch child runtimes**: `slurm_run` copies
   `state.pmd.environment.shell` into each child `slurm.sh`
   (dispatch serves the main framework).

Three input forms are accepted and normalized identically at all
points: YAML list, multi-line string, and single-line `;`-separated
string (classic input limitation). The kernel `get_shell()` performs
the normalization (`;`-split, strip, filter empty, `\n`-join).

### Q41: What is the null-value semantics for configuration merging?

**Answer:**
When merging configuration layers (package → user → runtime), the
following values are treated as **unset** and trigger backfill from
the lower-priority layer:

- `None`
- `''` (empty string)
- `[]` (empty list)
- `{}` (empty dict)

The following values are **effective** and do NOT trigger backfill:

- `False`
- `0`

This means a user setting `OMP_NUM_THREADS: 0` in their
`~/.pyrai3md/pmd_env.yaml` will NOT be overwritten by the package
default, while `OMP_NUM_THREADS: ''` will be. Execution control keys
(`execution_mode`, `slurm_*`) use **overwrite** semantics instead —
test YAML hard-coded values always take precedence (#80).

### Q42: How is the theory method whitelist discovered?

**Answer:**
The valid theory methods are discovered dynamically instead of being
hard-coded. `schema/validate.py` `_valid_theory_methods()` resolves

```
valid theory methods = BUILTIN_THEORY_METHODS ∪ skills.query.discover_all()
```

where `BUILTIN_THEORY_METHODS = {'nn', 'e2n2'}` (built-in
machine-learning potentials with no skill directory of their own) and
`discover_all()` contributes the program labels registered by the
installed skills (molcas, bagel, orca, xtb, openqp, pyscf, lammps,
cp2k, ...). The result is cached (`_THEORY_METHOD_CACHE`), and a static
`FALLBACK_THEORY_METHODS` set (kept in sync with the shipped skills)
keeps validation functional when the skill registry cannot be
discovered (e.g. a partial installation).

Consequences:

- Adding a new skill requires **no manual edit** of the framework
  whitelist — its manifest label becomes a valid theory method
  automatically.
- `_valid_reference_methods()` reuses the same set minus `nn`
  (reference recalculates QC electronic properties, never ML
  potentials).

---

## Part 9: Engine Ownership and Resident Execution

### Q43: How does a skill hand a quantity it owns back to the framework?

**Answer:**
Three transport channels, all declared on `Properties` in the skill schema and
all consumed the same way in `workflow/calculator.py`:

| Channel | Payload | Written into | Consumed by |
|---|---|---|---|
| `fin_coord` | `(natom, 3)` Angstrom | `system.coordinates` | `_apply_skill_geometry` |
| `fin_velocity` | `(natom, 3)` Bohr/au | `system.velocities` + kinetic energy recomputed | `_apply_skill_velocities` |
| `fin_cell` | six cell parameters **or** nine flattened lattice components | `system.cell` + `lattice_vector` + `volume`, derived together | `_apply_skill_cell` |

The shared contract, in order: the consumer pops the key from the returned
result and from the task config the backend stored, so none of the three ever
reaches `state.results`; an empty or absent value is a no-op (empty means the
engine did not own that quantity, which is the normal answer for a single point
or a gradient step); a malformed value raises naming it as a skill contract
violation; a value equal to what the state already holds is skipped without
writing; and a multi-task composite rejects all three, because sub-calculations
must not move the global geometry. Two refusals are deliberate rather than
defensive: `fin_cell` on a system whose `pbc` is all zero is an error, because
the engine demonstrably ran on a cell the input never declared (the framework
does not silently turn periodicity on to make the answer fit), and the cell
update writes `lattice_vector` alongside `cell` because the input builder reads
the lattice first, so updating one and not the other propagates a stale box.

Only the framework-side consumption is a contract; who produces what is a
capability. Today lammps is the only skill filling all three (its MD profiles),
CP2K carries a reader for the last-frame lattice comment ahead of any
cell-optimizing profile it has, and the other skills return empty strings.

### Q44: What does `python_enabled` actually change?

**Answer:**
It moves the expensive part of a LAMMPS calculation into backend
initialization. The compressed force field and the generated input package are
handed once to a **service process** that holds a live LAMMPS instance, and each
later `calculate()` sends only that step's coordinates and box over a Unix
socket and receives energy, forces and stress. No input file, run script, log or
dump is written per step and no `lmp` process is started per step.

- The decision is one predicate: `python_enabled and execution_mode != 'slurm'`. Under slurm every step is its own batch job, so the switch is inert there by design and the backend says so once.
- Once a step is routed resident, nothing falls back: a service that will not start, a box the domain refused, or a reply failing validation raises. A silently cold retry would report one path's numbers under the other path's name.
- Only `ff_energy` and `ff_gradient` are served. The internal-iteration profiles run a whole trajectory inside one call and gain nothing.
- The hot loop it exists for is framework-driven MD: `jobtype: md` with `method: ff` is one `ff_gradient` call per step, which is why all twelve official `md_lammps_*.yaml` examples declare the switch themselves rather than inheriting it. See skills/LAMMPS_SKILL_DEVELOPMENT.md Section 11.

### Q45: How close are the resident and subprocess paths, and where is the residue?

**Answer:**
Close by construction, not by hope: both paths get their input package from one
`write_input_package()`, their box numbers from one
`box_to_bound_numbers()`, their result file names from one `step_data_files()`,
and their tilt threshold from one `TILT_EPSILON`. The measured agreement on the
test environment is `dE = 3.469e-17` Hartree in energy and `max|dG| = 4.1e-8`
Hartree/Bohr in gradient, and the four per-step result files are byte-identical
between the two paths.

Four residues are understood and named:

- **Digits.** Forces come out of a text dump at six significant digits, so the subprocess path cannot report more; the resident path takes them from the engine interface at full precision. A trajectory judged on energy drift is therefore not measuring the same quantity on the two paths, which has to be said out loud before one compares two runs.
- **Representation.** LAMMPS converts a domain to tilt-capable and never back, so a cell that loses its tilt mid-session is computed on a triclinic domain with three zero tilts while a fresh subprocess run computes it orthogonal, and the reciprocal sum moves: measured on an untilted 11x10x10 cell, `-19.811016` against `-19.760369` kcal/mol. That one-way conversion is the asymmetry; **the round trip through it is not**. A probe that holds coordinates fixed in one live instance and walks a cube to the sheared cell an anisotropic NPT example actually reached and back, twice over each, returned `dE` `0.000e+00` kcal/mol, `dstress` `0` atm and `dbox` `0` Å against thresholds of 1e-7 Ha and ~0.3 atm: `change_box` plus a restated coordinate array is a pure re-description on this engine, so a variable-cell step cannot be an energy source through that channel (`bringup` and skills/LAMMPS_SKILL_DEVELOPMENT.md §11.12).
- **Cost, and what it prices.** Measured in the test environment: the resident step is `4.696` ms (yaml) / `4.879` ms (classic input) of framework wall over the shipped 2000-step NVE example, 12 s end to end, with the engine's own `evaluate` inside it at ~2 ms; the subprocess path was measured at `950` ms per step. That is the reason the framework-driven MD examples declare the switch at all. It is also a guard: an example whose service did not come up needs so much longer than its budget that the qualification timeout stops it, which reports a fall-back instead of passing hours later on a timetable nothing planned for. Quote the two numbers with their provenance -- `4.20 ms` elsewhere in this repository is the median of five alternating remote A/B pairs from #244, while `4.696` above is one sample per mode, and the gap between them lives inside the sampling band that round itself measured.
- **Rounding that used to be a bug.** A 90-degree angle becomes a 6.1e-16 tilt through `cos(radians(90))`. Comparing that against exact zero converted boxes that need no conversion and moved the reciprocal sum by 0.1 kcal/mol, which is why both the writer and the service now ask the same `TILT_EPSILON` question.

### Q46: What has to be true on a machine before a suite is trusted there?

**Answer:**
For a skill with a resident path, its operator-facing self-check runs first:
`python3 -m pyrai3md.skills.lammps_skill.service.persistent.bringup --dir
/path/to/inputs --project h2o --lammps-path /path/to/lammps --nproc 1 2 --stress
--periodic`. The `--lammps-path` is not a courtesy: five checks compare against
the subprocess path, and without the binary they report `lmp executable not found
in PATH` as failures while every purely resident check passes -- a run that looks
nearly green and verified less than it appears to. C11 is asked separately, with
`--perfbench N`. It answers
eleven classes of question (C1 the interface
surface, C2 the subprocess reference number, C3 resident agreement with it, C4
repeatability and sensitivity, C5 three variable-cell sequences, C6 the owner
watchdog, C7 the stress tensor, C8 rank independence, C9 where the files went,
C10 the per-step result files compared byte for byte, C11 what one resident step
costs, asked with `--perfbench N`) and exits non-zero if any
of them fails. Acceptance is a fact about one machine: `liblammps` links exactly
one MPI implementation, and an `mpirun` from another family leaves each rank its
own singleton world. Runtime probe code of any kind is not shipped in the
backend; this module is the only answerer, and it is a development-time and
operator-time tool, not a step in a calculation.

C11 is the one that turned a performance question into a measurement. Asked
about the cost of a resident step, the answer is three tables that each belong to
their own owner and are read side by side: `pyrai3md_step.perf` in the run
directory (the workflow loop, with the skill's nested breakdown under a header so
nothing is counted twice), `lammps_backend.perf` in the calculation directory
(what the step costs the framework process around the session, plus the client's
socket buckets), and `lammps_resident.perf` (what it costs inside the engine). On
the deployment the 10000-step NVE example measured 8.45 ms per step of framework
wall time against 1.47 ms inside the engine, and the rows accounted for 99.5% of
the step, which is what made the gap an owner-and-cost question instead of a
guess; #244 then took it to 4.23 ms with two byte-identical fixes. The tables are
written on a hundred-step cadence rather than at the end of a session because
**nothing in the framework calls `backend.close()`**, so any artifact held for
shutdown is an artifact that never appears.

For the suite as a whole there is one environmental hazard worth knowing: two
`pyrai3md test` runs on the same install directory clear each other's `tmp_*`
directories, which surfaces as a whole batch of
`bash: <run>/lammps_skill/<task>.sh: No such file or directory` with no
computation having failed. Confirm with the remote process list before
investigating a force field, and keep downloaded evidence directories outside the
install directory, since the next run there will remove them.

---

## Part 10: The Classic Input Format

### Q47: Where is a `&section` of a classic `.input` file interpreted?

**Answer:** In `schema/adapter.py`, never in `interface/parser.py`.

The interface layer has one job: choose a reader by format, then split text into `&section`
headers and `keyword value` lines, with `#` comments and `&end` ending the section's ownership of
keys. It does not know what any keyword means, and it must not start. `&molecule` is interpreted in
the adapter (`_parse_center_indices` and friends), so a section that needs more than "one value per
keyword" belongs there too.

The subsystem that violated this was the constraint block: its nested grammar needed a second
parser, `interface/constraint_parser.py`, dispatched by a special case inside the shared loop. That
file is gone (#271) along with the nested grammar, and `parser.py` is back to a single generic
loop. What it kept is the one real fix that round found: `&end` closes the current section instead
of opening a phantom section named `end` that swallowed every following key.

**How to apply:** adding a keyword to any section means editing the adapter's reader **and** its
writer for that section, not the parser.

### Q48: How is `&constraint` written, and what happens to a value the input left out?

**Answer:** Flat lines, one `;` segment per object, and every fill-in the reader does is printed in
the run record.

```
&constraint
  shake_bond 1 2; 1 3          # two strict bond constraints
  shake_bond_ref 0.96          # one value, broadcast over both
  bias_bond 1 4
  bias_bond_ref 2.80
  bias_bond_fc 0.020
  bias_bond_fc_start 0.002     # a ramp: from here...
  bias_bond_fc_stop 2000       # ...to the declared value, by this step
  wall_type sphere
  wall_sphere_radius 4.5
  wall_atoms 1-6
  wall_fc 0.002
  fix_atoms 6
  fix_dir z
```

A wall has three further choices of its own, and each is one flat keyword
(`CONSTRAINT_DEVELOPMENT.md` §4.3 and §22):

- `wall_groups` cuts the confined atoms into molecules and `wall_target` decides how each piece is
  weighted — they are independent. One segment holds atom numbers followed by the size of a
  molecule, so `wall_groups 1-3 7-9 3` is the two molecules [1,2,3] and [7,8,9]; `;` separates
  walls, so two segments are two walls. A group of one atom is the atom itself, and a wall with no
  groups is one object over its whole selection.
- the center is `wall_center` (coordinates), else `wall_ref` (the center of named atoms), else the
  center of the atoms the wall acts on. None of the three is the coordinate origin, and all three
  are captured once at the reference geometry, so a restart cannot move the boundary.
- `wall_law` chooses between `hard` (energy only past the surface) and `soft`
  (`V = k Q^(p/2)` over the dimensionless squared axis ratio — the archived polynomial cavity,
  which retired into the wall in #273). A cuboid's radius is a box **half** edge, and `center`
  converts its force constant as a plain energy because its argument is already a ratio.

`wall_center`, `wall_ref` and `wall_groups` also take the word `none` as a whole segment
(`wall_ref 1 2; none`), because a positional line otherwise cannot say "this object declares
nothing" without padding it with the previous object's value.

The rules that make that readable (full table in `dev/methods/CONSTRAINT_DEVELOPMENT.md` §4.3):

- `;` separates objects on an object line and per-object values on a parameter line; `,` groups the
  fields inside one segment. Atom numbers stay 1-based and ranges are `5-8`.
- A parameter line with fewer segments than the family has objects broadcasts (one segment) or pads
  (its last value); more segments is a refusal, because the objects were miscounted somewhere and
  dropping the extras drops a declaration.
- `wall_type` names which wall geometries are read at all. A geometry it does not name contributes
  nothing and its keywords are reported as ignored rather than dropped in silence.
- A parameter keyword carries the type of its variable: `bias_bond_ref`, never `bias_ref`; a strict
  constraint has no force constant, so `shake_bond_fc` is refused by name.
- Ids are minted by appearance when the input gives none (`bias1`, `wall2`), and they are what the
  log, the energy breakdown and a restart refusal call the action by.
- A ramp is five keywords around the one it ramps (`_start/_end/_scale/_delay/_stop`), and what a
  component does not say falls back to **that object's own** value, not to a neighbour's ramp.
  `delay` defaults to step 1; `stop` is closed by the schema from `md step`, with the source noted.

Everything the reader decided lands in `SystemConstraintConfig.notes` and then in the plan, and
`<title>.log` prints an `&constraint plan` section: one line per action (kind, family, id, what it
holds, its ramp windows; for a wall also its law, target, how many objects it settles and where its
center came from) plus the notes. A run with no constraints prints no such section.

### Q48b: Which constraints does a periodic cell serve, and who decides?

**Answer:** the bond graph, per variable, at binding time - not the action kind, and not the block.

- A strict constraint, a wall and a fix are served in a cell: the constraint is solved on the
  geometry unfolded along its connected components and folded back with the crossing counts it
  implies; a wall is gated by four geometric checks, a fix needs no graph at all.
- A collective variable (so a bias or a history potential) is asked one question: are its atoms in
  one connected component? A bond inside a water has a periodic image and is measured on the
  unfolded geometry - an O-H straddling a face reads 0.96 A, not the 7.6 A the stored coordinates
  show. A distance between two waters has no derivable image rule and is refused naming the action,
  the cv and the component count.
- `wall_law soft` is refused at input time: a spring toward a point has no surface to place inside
  a cell, and a wrapped coordinate would jump its offset by a lattice vector at every face.
- A ramped wall is checked twice over: cell fitting against every value the ramp reaches, the
  "members start inside" rule against the declared geometry alone - an atom outside the closed
  radius of a closing wall is what closing means.
- That containment rule is asked **only at the starting step** (#275). A one-sided penalty wall is
  crossed by thermal motion and pushes the atom back; re-asking the question at a later step aborted
  three real LAMMPS runs whose own contribution rows then went nonzero (`2.064651e-04` Ha for the
  cuboid). The other three checks keep running every step, because they are about the cell, not
  about where the atoms began.
- An aligned-RMSD history potential names no cv: it carries its block, and the image rules ask their
  question of those atoms (#275). Inside one water it is served; a block whose atoms the graph does
  not connect is refused by name.

### Q48c: How do ensembles and constraints meet, and what breaks?

**Answer:** by the degree count, and it is measured in a cell now (#276) - except that a moving
cell takes the declared tolerance away.

- The thermostat divides by `N_df`, and `ensemble._dof` gets the reduction from the plan: the rank
  of the condition matrix for a strict constraint, the frozen components from `constraints.fixes`.
  A periodic run proves the wiring delivers: NVT Nose-Hoover at 300 K with two bonds and `rattle`
  holds them to `8.131e-11` / `1.332e-15` A at a declared `1e-10`, and a `fixes` mask leaves the
  frozen atom at `0.000e+00` A under both NVE and NVT.
- `Etot` drift under NVT is **not** a criterion (+5.013150e-03 Ha here): a thermostat is not an
  energy-conserving map. The drift row belongs to NVE sentinels only.
- NPT with a strict bond was the failure, and the seam was the loop's order (D-25, closed #276):
  `_berendsen` rescales coordinates **and** cell and acts at the ensemble's post half step, while the
  projection ran before it, so the recorded step was one rescale off the manifold - `4.806e-05` /
  `7.649e-05` A against a declared `1e-10`, and `8.108e-04` A of drift on a frozen atom that sits at
  `0.000e+00` A with a fixed cell. Asking the geometry question again after the rescale
  (`md.py:580`, phase `constrain_after_cell`) restores both (`1.71178627e-11` A, `0.000e+00` A) for
  0.999 ms per step of an engine-bound 166.229 ms step, and 0.001 ms when there is no plan. LAMMPS
  moves no box here: the md profile produces no `fin_cell`. The residual is proportional to the
  cell's rate - coupling the barostat ten times more weakly shrinks it the same factor - and
  reversing the fix reproduces the old numbers bit for bit.
- The moving cell was then asked of every remaining action class (#277) and nothing refused what it
  should serve: bonds **and** an angle hold to `2.337e-11` / `5.667e-11` A and `9.216e-09` deg while
  the cell moves `0.020523` A; the four periodic wall checks stay silent while the cell moves
  `0.025197` A (the wall itself never engaged, so that arm is evidence about the checks only); a
  bond bias runs to `1.101342e-05` Ha without being refused.
- A constrained **optimization** in a periodic box holds its declaration: `1.0e-10` bonds measured
  `9.329e-11` / `6.793e-11` A over six accepted steps, and the aperiodic PySCF case holds `6.461e-11`
  A too. **D-20 (which claimed an `8e-4` A slip there) was withdrawn by #278**: the figure came from a
  measurement script whose artifact path did not exist, so it fell through to printing a directory
  listing and a number beside it was read as a criterion. Rule adopted: a bench script must abort when
  the file it measures is missing - a silent else-branch makes a claim unfalsifiable.
- What was open instead, and is now closed (#282, 口径 A), is **D-26**: the optimizer masks the frozen components of the *displacement*
  (`optimizer/base.py:168`) but takes `grad_rms`/`grad_max` over the whole raw gradient (`:213-217`)
  and compares those to `opt_grms`/`opt_gmax` (`:247-251`). A 2x2 over the declarations isolates it:
  the bias moves last `max_g` from `8.3668e-02` to `8.4217e-02`, the frozen atom moves it to
  `2.0797e-01` - the stopping criterion counts forces on coordinates the step may not move. It errs
  conservatively (convergence is harder, never easier), and masking the norms is the literature's
  convention, so the change waits for a ruling.
- The third condition kind, a dihedral, is measured on a real engine too (#277): H-O-O-H in a PySCF
  HOOH run starts at 160.000000 deg and spreads `2.625e-10` deg over five frames, while the
  undeclared O-O and O-H move `8.435e-03` and `3.823e-02` A - the controls are what make the held
  angle mean something, since a projection that froze the molecule would hold it perfectly too. #280 then closed the two cells that were
  written off as needing a new fixture. The shipped TIP3P box already contains five waters torn by a
  cell face (stored O-H separations exceed the minimum image by 6.6 to 7.2 A); on water 22-23-24,
  whose **both** arms cross, bonds-plus-angle hold to `3.647e-12` / `7.772e-16` A and `3.097e-09`
  deg on the resident channel, while the same angle from stored vectors jumps `99.8885` deg between
  frames - the wrong coordinate convention is not a small error. And a torsion now has all three
  answers on products: served inside one molecule, served across molecules with no cell
  (`1.421e-14` deg held), refused across molecules in a cell naming the components. A strict constraint's `tolerance` is a dimensionless relative
  residual (`constraints/shake.py:48`), so a degree spread is reported as measured and not compared
  against it.

### Q48d: What does a restarted run owe the reader, and who owns the constraint on a
two-engine run?

Four rulings landed in #281-#285, and three of them were about the instruments.

- **A restart must not warm up, and must not shift the record.** `md.py` counts steps absolutely now:
  `started = runtime.current_step`, the loop runs `started+1 .. started+maxiter`, and the counter
  advances before the checkpoint, the frame and the log block are written, so all three name the step
  that finished. Measured on the products: a continuous 8-step run makes 8 force evaluations, the same
  run split 4+4 makes 4 and 4 (`h2o_force.sh` invocations on LAMMPS, `converged SCF` blocks on PySCF) -
  a checkpoint already carries the potential surface of the step it was taken at.
- **The frame list is a function of the absolute step and the checkpoint cadence, and nothing else.**
  One segment end used to gain a frame the continuous run never wrote, because every process treats
  its own last step as final. The rule that removes it: a step that writes `<title>.rt` also writes a
  frame (`AIMD._runtime_file_due` decides both). Both arms then write `[1,2,4,6,8]` with the same time
  column and coordinates equal to `0.000e+00` A over every atom (periodic 60, aperiodic 3), energies
  equal row by row. A comparator that reports `0.000e+00` was checked by pairing two genuinely
  different frames of one arm: `2.564e-02` A periodic, `1.354e-03` A aperiodic.
- **`Epot` in the trajectory is the conserved quantity, and the log now shows the addition** (#283).
  D-18's registered claim (that `Epot` is the bare electronic energy) compared `md.energies` with the
  log's `&electronic energy`, which are two readers of one key. Against an independent unconstrained
  single point of the same geometry the recorded value is `origin + constraint terms`: bare
  `-74.9631631182905238`, recorded `-74.9631292844484278`, difference exactly the bias row
  `0.0000338338420919`. `runtime.finalize` therefore stores the engine's half as `energy_origin` and
  `print_results` renders `&constraint energies in Hartree` with `origin pot`, one row per constraint
  carrying the id the input wrote, and `Epot` last. An unconstrained run gets neither key nor section.
- **Ownership of the geometry is stated, never assumed** (#284, D-27). `control.driver` is handed to
  every skill by Layer 0 of `schema/adapter.py`; the framework drops its own plan when the driver is a
  skill, the LAMMPS skill drops its shake keywords when the driver is `main`, and both warn. The
  lesson is about the default: it was `main`, which silently stripped the shake section from every
  `&lammps` input built without a framework (29 failing tests, 18 of them byte comparisons). A
  cross-boundary switch must default to "this side decides"; the other side has to say so.
- **The periodic bond graph could not see an organic single bond** (#285, D-28). Perception tests
  `distance < bond_lib(pair) * 1.2`, and `schema/reference.py::BOND_LENGTHS` keyed `C-C` at `1.20` and `C-O` at
  `1.16` - the shortest members of those series - so a relaxed ethanol's `1.526` and `1.407` A bonds
  vanished and a torsion across them was refused as naming three components. The heavy-atom entries
  are single-bond values now, verified to be widening-only: the shipped TIP3P box still perceives its
  40 O-H bonds and nothing else (closest O...O `2.704` A, H...H `1.519` A against reaches of `1.776`
  and `0.888` A), and the ethanol box perceives exactly the 8 bonds its topology declares. The case it
  unblocked holds a C-C-O-H torsion to `2.842171e-14` deg over six NVE steps while an undeclared
  torsion of the same molecule moves `5.605489e-01` deg.

### Q49: Does a YAML declaration produce the same run as the classic text?

**Answer:** Yes - and since 2026-10-02 the question has a sharper form, because a YAML input is now
written with the *same keywords* as the text rather than with a second, structured spelling.

**One grammar, three carriers.** `&constraint` in a `.input`, the `constraint:` block of a YAML and
the JSON of an agent-mode request are the same flat keyword list. What `;` is to the text, a list is
to YAML/JSON: **one element is one object**, and that object's tokens are a scalar or their own list.

| text | YAML |
|---|---|
| `wall_atoms 1 2 3` | `wall_atoms: [[1, 2, 3]]` |
| `wall_atoms 1 2 3; 4 5` | `wall_atoms: [[1, 2, 3], [4, 5]]` |
| `shake_bond 1 2 1 3` (one block, two bonds) | `shake_bond: [[1, 2, 1, 3]]` |
| `group_name oxy; hyd` | `group_name: [oxy, hyd]` |
| `wall_center 0 0 0; none` | `wall_center: [[0.0, 0.0, 0.0], null]` |

Broadcast, padding and the over-long refusal are one function, so a parameter line behaves the same
however it arrived. The structured records (`walls: [...]`, `biases: [...]`, `units: {...}`,
`version`, `schedules`, `notes`) are the reader's *output* and the schema's input - at input each is
refused by name with the flat keyword that replaces it. The internal convenience is not a user-facing
dialect, and it gets no compatibility text: a manual and a template cover the human, the docs and the
code cover the agent.

`schema.adapter.write_constraint_section` is the inverse of the reader and lives in the same file,
because a grammar kept in two places drifts. `qualification/certification/converter.py` uses it to
emit `&CONSTRAINT` from a test config's `system.constraints` block, and the same call fills the
agent-mode sectioned dict, so both output formats carry the declaration.

Three properties the tests pin (`test_constraint_parser.py::TestWriter`,
`TestListsAreTheSegments`, `TestShippedInputRoundTrip`):

- **Round trip:** a block written to text, read back and normalized gives the same plan fingerprint
  as the block itself - including a config that came out of `model_dump`, where empty fields are not
  declarations.
- **Carrier equivalence:** the list spelling and the text line read to the same mapping, element for
  element (the five cases of `TestListsAreTheSegments`).
- **Refusal over approximation:** a declaration the flat grammar cannot state (an id on only some
  objects of a family, molecules of unequal size in one `wall_groups`, a segment whose atom count is
  not a whole number of rows) raises rather than being rendered into a file that reads back as a
  different run.

The round trip is not decoration: running it over the shipped drafts is what found three defects in
the writer itself - all but the first bond row dropped, every number truncated to six digits, and a
retired record key raising `KeyError` instead of a refusal (register D-29 to D-31).

### Q50: How does one switch off a hundred tests without hiding that it did?

**Answer:** Three levels, three values, and a closed test is still listed.

`pmd_env.yaml` gates the qualification plan at three depths, decided highest-first:

```
_runall                     master: true | false | condition
  └ examples.md_lammps_test      family:  true | false | condition
    examples.constraint_md_test
    examples.optimize_test
    skills.<name>.test
        └ examples.<workflow>    leaf:    true | false
          skills.<name>.tests.<case>
          benchmark.<task>[].enabled

  outside that chain:  skills.<name>.runtime_tests   (the skill's tests/ phase)
                       pyrai3md/skills/tests/        (the shared layer, no key)
```

`true` states the answer for everything below, `false` closes everything below, and `condition` is the only value that lets the next level speak. A leaf cannot be `condition` - there is nothing beneath it to defer to - and says so. A **missing** master key means `condition`, which is what the bare `false` used to mean. The shipped file says `_runall: condition` explicitly for the optimize round (#358) and `true` is what a full round is - the master decides above the families, so `false` is not "do not force", it is "close", and a round switched off that way runs no example however its family key reads. What it still runs is the code that needs no engine: the two pytest phases below answer to `runtime_tests` and to nothing else, so a master-off round executes the 100 framework tests and 65 code checks (see the measured readings below).

What each level covers is data, not inference: `manager.EXAMPLE_FAMILIES` names the four families and their prefixes - `md_lammps_`, `constraint_md_`,
`optimize_` (since #358) and `path_` (2026-10-21) - so one key can ask for an
optimization round or for the path family's (`optimize.method` mep / irc /
neb, thirty-two cases). The shipped file is set for a path round:
`path_test: true` with `optimize_test: false` and the master `_runall: condition`.
The examples bucket measures 166 planned / 91 closed / 76 printed; the whole plan
is 500 rows, 213 closed by switches (plus one by a capability) and 290 printed on
this environment, while the test environment prints 291 because it has `sbatch` and `md_batch`
therefore runs in both carriers. Round `tmp_20261022_151713` printed those 291
rows - 288 pass / 0 fail / 0 error / 213 closed - with all thirty-two `path_*`
examples green in both carriers, and `tools/verify.py` clean over its artifacts. Membership is the name, not the physics, so the two constraint examples that also relax a geometry (`constraint_opt_periodic`, `constraint_opt_surface`) stay individual along with `energy`, `gradient`, `multicenter_*` and the non-`constraint_md_` constraint cases; `test_manager.py::test_optimize_family_is_not_a_run_directory_chain` keeps the switch family from being confused with `EXAMPLE_CHAINS`, which is a shared run directory and nothing else. The skill level covers one YAML case per key, `skills.<name>.tests.<case>` named after the test file's stem; an unlisted case defaults to `true`, so the map closes cases rather than whitelisting them.

Two pytest phases sit outside that case map, and both are keyed to what they cost rather than to what they certify. `skills.<name>.runtime_tests` is the only switch that speaks for that skill's own `tests/test_*.py` files: `true` from the master or from `test` overrides it, because that setting promises everything under the skill runs, but no `false` closes the phase (#359) - those files need no engine, and a code check that only runs on the developer's laptop is a stand-in for an unreachable server, not extra content. `pyrai3md/skills/tests/` - the checks the skills share, the agent's `calculate()` forwarding and the wrapper's name resolution - is planned once for the round under the label `skills` (`planner._shared_runtime_tasks`) and carries no key at all, for the same reason. Neither phase arrives through the main-test path: that path discovers with `unittest.TestLoader`, which collects zero cases from a pytest module and would report the silence as a pass.

Three properties keep the switchboard honest:

- **A closed test stays in the plan**, as one skipped task whose `skip_reason` names the key to edit (`disabled by skills.molcas_skill.tests.casscf_gradient`). Before this, a skill with `test: false` contributed nothing at all, and "not certified" was indistinguishable from "certified and skipped".
- **What the plan lists is not what the run prints.** The plan keeps one entry per closed task, because a per-example switch is only auditable if what it closed is enumerable; `runner.run_tests` then folds a bucket's closed rows into one `<bucket>_closed` row carrying `skip_count` and the per-key counts in `closed_reasons` (#360), so a 445-row stage plan prints 251 rows and the log is bounded by the work the round did. A skip that is not a closed switch - a postponed example, a host with no `sbatch` - keeps its own row, and the final report still groups those by reason.
- **A key that names nothing warns.** `examples: {md_lammps_tes: true}` used to be ignored silently - the shape of mistake that reads as "I opened the family"; `_warn_unknown_example_keys` compares every key against the shipped example stems and the family table.

Measured on the shipped file. The 2026-10-02 reading (a local counting script from that day, since deleted; the same retirement is recorded in `dev/methods/CONSTRAINT_DEVELOPMENT.md` section 33, when the file shipped `condition` and 63 examples) planned 331 tasks with 182 skipped, opened exactly 12 examples when `md_lammps_test: true` was set over individually-closed cases, gave 38 molcas entries for one skill at `condition` with one case closed, and produced no warning in any of the five readings it took. Those numbers belong to that file. Today's file, with 128 examples, measured through the same probe
`tools/deploy.py` sends to the server (`generate_plan(load_config(), tmp)`, folded
per category by `_is_closed`/`_folded_count`): `_runall: true` plans **745 rows - 103 main + 253 example + 364 skill + 25
benchmark - with six of them named capability skips** (two want `sbatch`, four
want TensorFlow) and no switch closed, so it prints 745. That reading is
capability-bound and not a constant: on the repository `.venv` (3.11, TensorFlow
2.21.0) the same posture plans **784 rows** (103 main + 256 example + 400 skill +
25 benchmark, two capability skips) and prints 784. `_runall: false` plans **435
rows, closes 265 by switches, and still runs the 170 rows that need no engine** -
the 103 framework tests plus the 67 engine-free code checks - printing 173. The
path round the file is actually set for plans **500 rows** (103 main + 166 example
+ 206 skill + 25 benchmark), closes **213** of them by switches plus one by a
capability, executes **286**, and prints **290** here (291 in the test environment, where
`md_batch` has `sbatch` and runs both carriers). Earlier readings recorded below
belong to the files as they then stood, not to this one: the #358 posture gave 676 rows before the shared-layer pytest group existed (and the cluster round dispatched then read 678, because the two Slurm-armed examples emit one collapsed skip on this environment and two rows there), 380/217/163 before #359 decoupled the pytest phase from the certification - that reading said "only the 99", with the per-skill code checks closed together with the certification - and 392 rows / 205 closed / 187 executed / 190 printed for the optimize stage as #358 configured it, against the 390 planned / 123 executed #358 itself recorded. On the cluster the optimize stage ran **123 executed tasks in 40 minutes against the full pass's 4.5 hours** (tmp_20261016_135553, all 123 passing, 267 rows listed as named skips); the round after it, tmp_20261017_094721, reported **194 rows, 191 passing, 0 failed, 0 errored, 206 named skips** on the shape the file had before the LAMMPS certification was opened, and carries no recorded wall-clock. Planning follows host capability by design, so a host without the optional engines reports fewer executed rows and the same plan.

---

## Summary

PyRAI3MD is designed as a **modular scientific computing framework** where:

- **Core** owns protocols and orchestration
- **Skills** own domain knowledge and execution
- **Maestro** enables dynamic capability discovery
- **Workflows** compose abstract capabilities
- **Tests** verify the full execution pipeline
- **Ownership travels through channels**: when an engine propagates the nuclei, moves the box or runs its own iteration, it says so by returning `fin_coord` / `fin_cell` / `fin_velocity`, and the framework is the only writer of the state those land in (Q43)
- **Classic input is split at one layer and read at another**: `interface/` never learns what a
  keyword means; `schema/adapter.py` owns each section's text, in both directions, so a `.input`
  and the YAML that generated it cannot drift apart (Q47, Q49)
- **A fill-in the input did not state is a note the run prints**: broadcasts, padded values,
  ignored wall geometries and a ramp stop inherited from `md step` all reach `&constraint plan` in
  `<title>.log` (Q48)
- **Path agreement is constructed, not hoped for**: where an engine can run in more than one way, one function produces the input, one predicate routes the run and one table of names writes the results, so the remaining differences are measured, bounded and named rather than absorbed into a tolerance (Q45)

The three runtime modes (classic, agent, skill) ensure comprehensive coverage of the system's capabilities, from legacy compatibility to standalone skill execution.

The qualification infrastructure is independent of the runtime core, providing planning, execution, and reporting without modifying the computational framework.

---

## Part 11: Machine-Learning Workflows and Test Layers

### Q51: Where do `train`, `prediction` and `search` live, and why does it matter that they are nodes?

**Answer:** `workflow/train.py::Train`, `workflow/prediction.py::Prediction` and
`workflow/search.py::GridSearch`. The first two share
`workflow/single.py::MLSingleShot` (one `calculator.run()`, the standard
step bookkeeping, the outcome settled from the returned record by
`workflow/records/ml.py`).

Being a node, not a token in `SingleShot`'s `else` arm, buys one concrete
thing: a section of its own. `WORKFLOW_SECTIONS['train'] = 'train'` makes
`&train remotetask: true` reach the graph, so the longest call in the
package can be handed to the cluster; before it, the jobtype could only
inherit class defaults (`dispatch/planner.py:resolve_flag` puts the
declaration layer above the input keyword, and
`test_dispatch.py::test_node_name_out_of_the_tables_drops_the_section_flags`
shows what happens to a name that is in neither).

Two things do not change. Under `control.driver != 'main'` every jobtype
still resolves to `SingleShot` -- a skill-driven run says the skill plans
the job. And the printed report keys off the record's shape, not the class
(`utilities/print_results.py` branches on `train_errors` /
`predict_errors`), so moving jobtypes between classes changed no reporting.

**Rule:** the outcome of a single-call jobtype is settled by whether the
channels it is contracted to carry came back, never by a threshold on their
values. A validation mean absolute error is not a convergence criterion.

### Q52: Who declares that a hyperparameter may be scanned, and who says what a value means?

**Answer:** split by ownership. The skill's `manifest.yaml` declares which
keywords are tunable and their type; the kind (`scalar` / `list` / `pairs` /
`bool` / `reg` / `text`) is *read* from the skill's own vocabulary
(`skills/nn_pes_skill/schema/hyper.py:HYPER_KEYWORDS`), never copied into the manifest, because
the kind is exactly what tells a scan whether a list means several ensemble
members or several candidate values. The run supplies the candidate lists
(`&search grid`); the framework generates no values from a
`start/stop/step` range, because that would need type rules it is designed
not to know and would invent numbers like `0.30000000000000004`.

**Rule:** a keyword written as a list means the member axis
(`nn_pes_skill/schema/hyper.py` says so in its own header), so a grid value is never
handed over as a list of alternatives. The one place that rule is
implemented is `workflow/grid/plan.py:write_point`.

**Trap:** the framework must not keep a second copy of a skill's type table.
`schema/adapter.py:_parse_search_grid` reads text as text;
`skills/nn_pes_skill/query.py:check_searchable` refuses a candidate whose
type is wrong for the keyword, at plan time, before any directory exists.

### Q53: How is a grid point's model directory found again after a restart?

**Answer:** two naming schemes for two different jobs. The runtime directory
keeps the manager's `..._runtime_<idx>` form, because ascending reuse of
existing indices is what lets a restarted parent map its batch back onto its
previous runtimes. The **model** directory is named by the point's values
(`NN-<title>-depth=1~nn_size=8`), so it is re-derivable from the point alone
and an index reassignment can never cross-wire two points. The separator is
`~`, not `.`: a rendered float carries a `.`, and a key that cannot be split
back into pairs is not a reversible name.

**Rule:** a scan never re-trains a finished point. A complete folder is
`skipped`, its numbers read from disk and ranked with the rest. A plain
training request that would overwrite it is refused -- a continuation cannot
apply the architecture a scan varies, because the container's `update()`
merges every hyperparameter category except `model`.

### Q54: What must a batch child never carry into a worker process?

**Answer:** two things, both measured by failure rather than reasoned out.

1. **No bound executor and no parent calculator.** A pooled unit is pickled;
   a bound method drags the workflow with it, and the workflow's Calculator
   holds live backends -- for nn that is built Keras models plus a
   `WeakKeyDictionary` of compiled functions, which cannot be pickled at
   all. So a fan-out names its child entry at module level
   (`workflow/search.py:_child_entry`, `workflow/hessian.py:_child_entry`)
   and, where a copy of the workflow is made for the batch, strips the
   calculator from it.
2. **No handle on the parent's run log.** `materialize_unit` deep-copies the
   state, hence the logger, and the copy's path is the parent's absolute
   file. Opened for writing it truncated everything the parent had reported,
   and each writer then advanced its own offset, leaving holes the
   filesystem fills with NUL bytes -- a four-point scan measured 10475 of
   them in its own log. `RunLogger.__deepcopy__` and `__getstate__` hand out
   a closed logger; a child's verdict reaches the parent through its runtime
   report, and a process starting from a saved `.rt` builds its own logger in
   its own directory.

### Q55: Which tests may start an engine, and which may not?

**Answer:** the case files and one python module. The skill's
`tests/00_…07_*.yaml` certify the shipped route through the three entry
points, and `tests/test_train_predict.py` certifies the same route in
process -- including what the fit wrote: per-member hyperparameters in each
`hyper_v<i>.json`, the stored error unscaled by the folder's own scaler, the
gradient being the derivative of the reported energy. Every other
`tests/test_*.py` is a function test and uses
`tests/_fake_engine.py:FakeContainer` instead.

**Trap:** an engine run inside a function test is not merely slow. Measured
here, a fit costs ~11 s of which the epochs are nearly nothing (30/25
epochs 73.2 s vs 5/4 epochs 71.4 s for that module), because
`nn_pes.py:146` starts one subprocess per ensemble member and each imports
TensorFlow. So tuning epochs is not the lever; ownership is. Two modules
that had been training surfaces for no reason took 78.1 s and 34.7 s; on the
fake container the same assertions run in 0.06 s and 0.04 s, on a host
without TensorFlow, with **exact** expectations rather than tolerances.

**Rule:** when a fixture's `setUpClass` fails, its tests disappear from the
count -- one such failure dropped 232 to 227 while the run still reported a
single error. Compare a suite count to a baseline; `OK` alone is not
coverage.

---

## Part 12: The Slurm Section and the Remote Arm

### Q56: Where does a dispatched child runtime get the allocation it asks the cluster for?

**Answer:** one route, three carriers. The top-level `slurm` section is read
by `kernel/environment.Environment.get_slurm`, frozen onto the run by
`schema/adapter._build_environment_model` as `state.pmd.environment.slurm`,
and turned into `#SBATCH` lines by `workflow/dispatch/slurm_run`. A run can
restate any key in its own input through three carriers that all reach that
one hook: a YAML `slurm:` block, the classic `&SLURM` section, and the `slurm`
entry of an agent `.config`. The merge (`apply_slurm_overrides`) is per key,
in the priority order package < `~/.pyrai3md` < input.

**Trap:** this section is a framework-level section, and two neighbouring
things are not it: `&environment` (whose keys describe the shell a process
starts in) and each skill's own `slurm_*` keywords (its self-submission path).
Routing a `&SLURM` block through the environment section leaves the dictionary
the job-script builder reads untouched -- the input reads as configured while
every child goes out on cluster defaults.

**Rule:** a directive spelled wrong (`memmory: 8G`) maps to nothing and would
be dropped in silence; `unknown_slurm_keys` refuses it at the point where the
job is really going out, and names the known set in the message. Both script
builders - the qualification test script and the dispatch child-runtime script
- render the section through the same `kernel/sbatch.py` dialect, so the same
`slurm:` section produces the same `#SBATCH` block in either path.

### Q57: Why does a run's `slurm:` section need two rules of its own?

**Answer:** because the two things that make a config system work are
independent here, and each broke alone.

- **It must be read even with no `&environment` section.** A merge placed
  after the guard that returns when the input declares no environment block
  applies to no scan example, because a scan states a grid and an allocation
  and never a shell. The merge is now the first thing `_apply_environment_overrides`
  does.
- **It must apply to that run alone.** `Environment` is a process singleton
  and `initialize()` is idempotent, so `combined_config` outlives the run that
  merged into it. The second input built in one process would read the first
  input's directives as host defaults, turning `package < user < input` into
  `package < user < previous input < input`. `reset_input_overrides()` rolls
  back to the on-disk state before each merge; the paths that hit it are
  ordinary ones (`agent.py` builds a config per instance, `lammps_skill`'s
  bringup builds a cold and a warm config in one process and compares them).

**Trap:** the classic text has no boolean type, so `exclusive false` arrives as
the word `false`. Read as any non-empty string is truthy, it adds
`#SBATCH --exclusive` and reserves a whole node per grid point for a scan that
asked to share one.

**Rule:** judge a carrier by the artifact the consumer produces, not by the
dictionary it holds. The text carrier gives `'2'` where the YAML gives `2`; both
become `#SBATCH --cpus-per-task=2`, and that line is what the cluster reads.

### Q58: What can be verified about a remote scan on a machine with no scheduler, and what cannot?

**Answer:** everything the framework controls. Off a cluster the planner reads
the submission request from the example's own input
(`manager._example_submits_remotely` looks for a truthy `remotetask` in any
section, not at a file name) and records a skip whose reason names `sbatch`, so
the report says "the test environment has no cluster client" rather than "the framework
is broken". With `sbatch`/`sacct` shimmed on `PATH`, the run reaches its own
dispatch and job-script paths and stops where a cluster would answer, so one
`.rt` payload and one `slurm.sh` per point can be read back: which directives
arrived, which did not, and whether the model directories stayed distinct.
Queue time, an allocation the cluster declines, and a child dying on a compute
node are only knowable in the test environment.

**Trap:** an example whose default arm is skipped must still have its mechanism
gated daily, which is why the scan ships as two arms of one plan
(`nn_grid_search.yaml` remote, `nn_grid_search_local.yaml` on a local worker
pool, same grid and ranking, different project name and grid root). A skip is
not evidence; the local arm is.

**Rule:** a scan example must differ from the host's own section on the keys it
claims to override, and the test must say so. A declared value that equals the
host's own section proves nothing; `mem: 4G` and `time: 0:20:00` differ from the
host's own mem and time, so they do. A carrier test whose declared values all
equal the defaults passes with the merge deleted.

### Q59: One dispatched point dies in the parent. What does the scan owe the operator?

**Answer:** its table. A submission that fails, or a child that reaches a
terminal state without writing its result, raises in the parent and used to
propagate out of `dispatch()` before a single row was built: the fits that had
landed on disk lost their ranking, and `summary.csv` and `crashed.txt` did not
exist. `workflow/search.py:run` now records the sentence as the reason of the
points that have nothing on disk and still calls `postprocess`. The run is
marked failed when every point failed, and the qualification runner reads that
from the log's `Status:` line, so the containment cannot turn a dead scan into a
passing one.

**Rule:** the parent's sentence outranks the folder's. `build_row` prefers the
dispatch failure over "no model directory", because the absent folder is the
consequence of the death and the parent names the cause; a point whose folder is
complete keeps its numbers and its `ok` status whatever happened to the rest of
the batch.

---

## Part 13: The Curvature Source and the Saddle Search

### Q60: Why does the force-constant source need two keywords, and why are two of their pairings refused?

`&hessian hess_driver` answers who produces the matrix (`main`: the framework's own
displacement batch; `skill`: the backend, asked for the matrix it computes) and
`&hessian hess_mode` answers how it is produced (`analytic`, `numerical`, `auto`).
They are two questions, so one keyword cannot carry both without becoming a
cross-reference, but the answers are not freely combinable: the framework has no
analytical second derivative of its own, so `main` with `analytic` asks for something
no one can give, and `auto` means precisely "the producer has not been settled on",
which cannot stand beside a named one. Both are refused in the section's own
validation, where the input is still cheap to change.

`skill` with `numerical` is legal and worth naming, because it is what ORCA's
`num_freq` and Molcas' MCKINLEY do: the engine displaces its own atoms. To the
framework that is indistinguishable from an analytical answer - one evaluation that
returns a matrix in `results['hessian']` - which is why the framework never learns
which of the two the engine did. `hess_driver` also has a second, unrelated meaning
to keep straight: `control.driver` says who drives the *run*, and one of its values
(`skill`) keeps the run out of the Hessian workflow entirely.

### Q61: Why must a saddle search count negative curvatures in a projected subspace, and follow its mode in a fixed frame?

Because the two readings a search makes of its own curvature model are both about
physical vibrations, and a `(3N, 3N)` Cartesian matrix is neither.

* The count. A Cartesian model carries the whole-body motions, and their curvature is
  whatever the updates left there. Measured at the starting geometry of a real run,
  the raw nine-by-nine matrix of a three-atom system showed five negative entries
  where the rigid-projected reading showed the single imaginary frequency the engine
  itself reported. A convergence criterion that demands "exactly one downhill
  direction" and reads the raw entries therefore refuses a saddle the search has
  already found. The search now counts eigenvalues of the mass-weighted, rigid-
  projected block - the same basis `workflow/frequency.py` builds for frequencies, so
  the run's own row and a later `frequency` job cannot drift apart.
* The direction. The vibrational basis is rebuilt from the geometry at every
  evaluation, so its columns are different vectors step to step. Carrying the
  followed mode as *components in that basis* and comparing them at the next geometry
  compares one motion's shadows in two frames; stored in the fixed mass-weighted
  Cartesian frame, the followed direction is one physical motion of the atoms. The
  difference was measured on PySCF, not argued: with the per-frame storage the search
  walked from a gradient of 0.029 Hartree per Bohr to 3.3 in 60 evaluations, steadily
  uphill, while the analytic instruments all kept passing - they are one atom carrying
  three numbers, where the rigid block is the whole space and the projection never
  runs.

**Rule:** a new curvature-reading rule needs a multi-atom instrument. `test_optimizer_crossing.py::_Saddle3` is that fixture, and its first test asserts the fixture's own rigidity invariance, because the first version of it wrote the anharmonic term on Cartesian components, which left a net force on the molecule and froze the search at a gradient of 8.5e-3 for 120 evaluations - a defect in the instrument that read as a defect in the step bounds.

### Q62: What bounds a saddle search that must accept energy rises?

Not `reject_on_energy_rise`, which is off on purpose: climbing the followed direction
is the goal, so a rise proves nothing. Two rules do the bounding.

* `TransitionStateOptimizer._rejection_reason` refuses a step whose energy change has
  the *opposite sign* to the change the curvature model predicted for it - a model
  wrong about which way a step goes, not merely about how far. Predictions smaller
  than the run's own `energy_threshold` are not judged at all: read as evidence,
  numbers at that scale are the print's roundoff, and judging their sign produced a
  rejection every evaluation.
* `MinimaOptimizer._rejection_offers_a_shorter_step` skips a rejection when the trust
  radius is already at its floor, because a rejection exists to take a *shorter* step
  and there is none left to take: the geometry it restores to and re-steps onto is the
  one just refused, and the run repeats one point forever (measured: the same
  rejection line printed 26 evaluations running with the history frozen).

The radius ceiling is a saddle search's own (0.1 Bohr, against the minima relaxation's
0.5), chosen on measurement: 0.05 was the only bound tested that diverged to +2.2
Hartree from an isotropic starting curvature, and 0.1 converged from both the true and
the isotropic starting model. With those in place, both `hess_driver` producers reach
the same HCN->HNC saddle on PySCF in 32 evaluations to 1e-8 Hartree, and the engine's
own `num_freq` there reports exactly one imaginary frequency.

### Q63: Which keyword says what an optimization may move, and why does a search refuse part of that vocabulary?

`&system_constraint`'s `fix`, and nothing else. "Optimize only some atoms" is not a
second selection mechanism: `fix` removes components from the active set and restores
their anchor (`constraints/projection.py:build_fix_mask`), the criteria read the
tangential gradient that leaves, and since #336 the same mask drives two more places
that used to decide for themselves - the atom set the numerical curvature batch
displaces (`hess_atoms`, derived during a search), and the subspace the saddle search
solves and counts in (`vibrational_basis(mask=...)`, plus `Optimizer._step_gradient`
so the step is *solved* on the tangential gradient instead of cut down afterwards).

The plan being the authority makes three input shapes refusals rather than surprises:
a `hess_atoms` set that disagrees with the pinned atoms; a `hess_atoms` set with no
`fix` behind it (the matrix would hold zero rows on atoms the run can move, and zero
is the lowest curvature a model has); a variable-cell request on a system that carries no lattice, or a singular one
(the cell gradient refuses both), and one whose evaluation returns no whole-system
stress; a
multi-center partition, because a center is a region of one geometry rather than a
surface with a saddle of its own; `hess_driver: skill` on a multi-backend run; and a
second electronic state for a search that reads one (`uses_state_pair`), which would
buy a gradient every step and discard it.

The last refusal before that - a `fix` pinning less than a whole atom - is a physics
boundary. Pinning one Cartesian component of a molecule in vacuum fixes a laboratory
direction the molecule has no reason to respect: rotate the input and the constraint
is a different constraint, and the torque the pin leaves behind has no free coordinate
to be removed along. Measured on the analytic three-atom saddle, the step fell to
1.3e-7 Bohr and the model kept its one negative curvature while the criterion still
read 2.08e-2 Hartree per Bohr on *free* components. Whole-atom pins keep the rotation
inside the movable set, so the search's subspace and the surface's symmetries agree.
Where an axis is physical - a surface normal, a cell direction, a field - the
component-level case is worth having, and it is unimplemented rather than argued
away: `dev/TODO.md` carries it.

What the plan freezes then decides the second half of the subspace rule (#338).
A whole-atom `fix` does not simply shrink the vibrational space: of the six
candidate whole-body motions, only the combinations whose frozen rows are zero
are still free of restoring force, and those are the ones the frame removes.
With one atom pinned that is three of six, and the other three - a partial
translation of the free atoms against the anchor - carry curvature and force
like any internal coordinate. Removing all six instead leaves the search a
subspace it cannot converge in: measured on the six-atom saddle, fifteen movable
components, twelve of them curvature-bearing, one of those negative, and the run
holding at `max_g 1.73e-2` for 400 evaluations when the frame took out nine. On
three atoms the same rule leaves one pinned atom with three working directions
(-0.193, +0.501, +15.593), which is why a triatomic is enough to ask for a
converged partial saddle: the four-atom instrument earlier rounds said this
needed was an artifact of the over-projection, not a property of the molecule.
The real engine agrees: `optimize_partial_ts.yaml` holds a water dimer's donor
oxygen and converges in 75 evaluations to a geometry whose own frequency job
reports one imaginary mode, -3064.0 cm-1.

---

## Part 14: Multiscale Charge Ownership and Reciprocal-Space Convergence

### Q64: Whose answer is a region's point charges, and what does a phase-local derivation break?

**Answer:**
Ownership is derived from the whole task queue, once. `update_from_outcomes`
(`workflow/multiscale/charges.py:112`) takes the `sources` map produced by
`region_charge_sources` (`charges.py:40`) over *every* task in the composite, and
that map is what says which result rows may write which region. A caller that
hands it the sub-list it happens to be running gets a wrong partition, because
"which rows belong to `highlevel`" is a property of the queue's theory ordering,
not of the phase currently executing.

The defect this rule closes (#24, round `tmp_20261031_183037`) was exactly that:
`_run_two_phase` (`workflow/calculator.py:1219`) passed its `phase_tasks`, so in a
two-phase embedding the second phase's `theory[1]` claimed the high-region rows.
**The energy was right.** `embedding_for_task` only consults `provides_electronic`
and `embedding_regions[center]`, so each task was delivered the correct
environment; what came out wrong was the charge store, the `&highlevel point
charges` block in the run log, and `has_values`, which the wrong producer set true.
That is the shape of an ownership bug: invisible in every number the workflow
consumes, visible only in the audit print - which is why it surfaced in the round
that was reading charge blocks rather than energies, and not in the four rounds
before it that compared energies.

The criterion is built as a non-identity witness: the same queue is partitioned
twice, from the whole list and from one phase's sub-list, and the test asserts the
two stores differ and that only the whole-queue one puts the high rows under the
high producer. A test that merely asserted "the store has the right values" would
pass under the buggy caller too, because the buggy caller was producing correct
values for the wrong reason in single-phase cases.

### Q65: The engine already fitted the charges - why does the framework read them back, and how does it pick which set?

**Answer:**
Because a pinned `q_qm` is the framework asserting a number it did not compute, and
a QM engine that returns a wavefunction returns a charge distribution. The readback
is the 0-order ESPF multipole fit, and getting it right took three separate
corrections, each of which is a general trap:

1. **The gate vocabulary has to match what the caller actually sends.**
   `_ESPF_CHARGE_EMBEDDINGS` (`skills/molcas_skill/backend.py:59`) read
   `('charge', 'potential')`. `potential` is a value this skill's own schema
   refuses, and what QMMM sends is `field` - so the readback existed in the tree
   and executed on **no shipped case, ever**. The trap is a derived field that
   every example pins explicitly: the derivation then looks covered (the guard is
   green) while nothing exercises it. `multiscale_qmmm_ts.yaml` now deletes its
   `q_qm` so the derived path runs on the real engine; the other three QMMM cases
   keep pinned values, because pinning *every* case and pinning *no* case are both
   blind - the derived side needs a positive witness and the pinned side needs the
   override to be testable.
2. **Choose the state's set before touching the numbers.** `ESPF.DATA` is written
   into the per-job run directory that `MOLCAS.sh` deletes, so the log is the only
   surviving source - and the log prints many `++ ESPF analysis` blocks: one per
   root inside `rasscf`, plus more from `caspt2` (the shipped double-root XMS case
   prints twelve, two and ten). A block carries no state label. Selecting by
   position ("the last block") therefore returns a `caspt2` or wrong-root vector;
   that is what the first implementation did, and it accounted for all sixteen
   failures of round `tmp_20261031_195117`. `_from_log`
   (`skills/molcas_skill/service/parser/espf_charges.py:84`) tracks
   `--- Start Module: … ---`, counts blocks **inside `rasscf` only**, and takes the
   k-th as the k-th root; a wrong length or a missing block for the requested state
   raises instead of answering.
3. **Then spread what the fit leaves over.** The multipole fit is a least-squares
   projection onto a shell grid, not a constrained sum, so the returned set does not
   add up to the target charge: 1.2e-3 e for a neutral water at the default 4 shells
   / 1 Å, and denser sampling does not remove it (1.5e-3 at 56,541 points, 5e-4 at
   8 shells - the residual is the model, not the grid). `_balance_qmmm_charges`
   (`workflow/calculator.py:1610`) subtracts `deficit / n` from each value, but only
   when the target is an int (`system.charges` as a per-atom list is a different
   request, and the case is skipped with the reason logged) and only when
   `abs(deficit) > _CHARGE_BALANCE_TOLERANCE` (1e-4, `calculator.py:134`) - below
   that the adjustment would be noise dressed as a correction.

The order is load-bearing and was the ruling: bind the state first, then average
the residue. Averaging first would spread a wrong-root vector over the whole
partition and the sum would look perfect while every value was wrong.

### Q66: Why is reciprocal-space accuracy a configured key instead of a pinned mesh?

**Answer:**
`ewald_accuracy` (`skills/lammps_skill/schema/reference.py:103`, default `1e-8`)
rewrites only the accuracy token on the `kspace_style` line this skill re-issues
after `read_data`, plus the field probe's own setup line. The reciprocal **style**
stays whatever the template declared, and non-periodic runs are unaffected by
construction: a `subsystem` re-issue returns an empty line, and a template with no
`kspace_style` yields no line to rewrite - that is tested, not argued.

The key exists because the periodic QMMM field probe's two runs (A: the QM atoms
carry `q_Q = +1`, B: they carry zero) disagreed by 3.3e-4 relative on a 1581-atom
box. The reason is not "the two decks are built differently": LAMMPS estimates the
reciprocal extent from accuracy x qsqsum x volume, the two probes have different
sqsums, and so they get different meshes (9x9x9 against 7x7x7) at the same declared
accuracy. "Then pin the mesh" was falsified on the deployed build
(LAMMPS 23 Jun 2022 - Update 1, the same version in all three remote environments):

| attempt | result |
|---|---|
| `kspace_modify mesh` | silently ignored by `ewald` - two decks differing only in that line return **bit-identical** readings (`pppm` honours it, which is the positive control) |
| `kspace_modify cutoff` | `Illegal kspace_modify command (src/kspace.cpp:630)` |
| per-atom `group/group` (would remove the need for A/B at all) | unsupported: `compute_group_group.cpp:94`, `compute.cpp:133` |
| switch to `pppm`, unpinned | worse: 0.056 -> 0.217 |

And pinning would not have been correct even if it worked. At 1e-6, probe B's own
summed per-atom energy is `-274.64915` against the converged `-274.62099` - 1.0e-4
relative, single-sided. Forcing both sides into one format buys *consistency* and
leaves that error inside the delivered V. Convergence buys both: 1e-6 -> residual
0.05610 (5.3e-4 relative), 1e-8 -> 0.00022 (2.1e-6 relative), and 1e-10 / 1e-12 no
longer move the readings, so 1e-8 is at the plateau rather than chosen for looks.

The cost is priced, because a default is a bill: 1.3-2x on long classical
trajectories (h2o/pppm/60 atoms 0.78 s -> 1.01 s; water25/ewald/75 atoms
1.46 s -> 2.96 s, single process, wall including startup), while the 0-step probe
that needed the precision costs nothing at 1e-8. That asymmetry is why the default
is the converged value and the 28 shipped classical cases spell their existing
precision out (1e-4) instead of inheriting it - the number in the file is the
number the case has always used, and the deck it generates is byte-identical to the
one from before the key existed.

Three consequences are easy to miss and are covered by tests:
- the key is part of `config_hash` (`service/builder/ff_model.py:507-509`); leave
  it out and two accuracies run in one scratch directory and hand each other the
  other's input package - a silently wrong deck, which is the worst kind;
- YAML types `0.0001` as a float and `1e-4` as a string, so the field needs a
  normalizing validator before the type check, and the validator must actually
  reject `abc` / `1.0` / `0`: an unvalidated string switch is this project's
  recorded failure family;
- DEFAULT, the schema field and `FIELD_TO_SUBMODEL` must agree, which is what the
  parity test borrowed from `nn_pes_skill` now checks for this skill - a key
  declared in config but missing from DEFAULT is dropped in framework mode without
  a word.

The probe's own guard is still `1e-4 * scale`
(`service/parser/qmmm_output.py`). Tightening it to `1e-6 * scale` was decided
against, on the round's own artifacts: recomputing the h2o/pppm probe from
`h2o_qmmm_field.esfp` against its A-run group tally gives a residual under
3.9e-6 kcal/mol on a scale of 134.4 (so pppm would clear a `1e-6 * scale` budget
by 34x), while the ewald box leaves 2.2e-4 absolute against that same budget
(1.68e-4) - short by 1.3x. The two styles converge at different rates, so the
case that binds the guard is the ewald one, and the tightening is blocked on its
convergence rather than on the instrument.

### Q67: What does a resident LAMMPS deck owe a box it still has to change?

**Answer:**
It owes the box its freedom. LAMMPS only makes a per-atom compute current when
something consumes it, and the usual consumer is a `dump` - so a deck that wants to
read `c_QMMM_PEQ` per atom declares a dump file it then never opens. On the
`qmmm_field` resident package that dump also made the session refuse the box change
the same session has to perform (`change_box` errors out while the dump is bound),
which is defect #32. The fix removes the file, not the consumer: a per-atom
`variable` over the compute plus a `compute … reduce sum`
(`service/builder/qmmm_field.py:252`) refreshes the compute and holds nothing the
domain owns. The resident package therefore declares no dump at all.

Three measurement traps sit on the same code path, and each produced a false
reading before it was seen:

- `undump all` is not a command - it wants a dump id (`src/output.cpp:817`). The
  first "option one is impossible" verdict was a syntax error in my own probe, not
  an engine limit; `undump QMMM_FIELD_RES` works fine.
- `numpy.extract_compute` returns a **view of the engine's internal buffer**. Two
  reads taken before and after a displacement are the same memory, so they compare
  equal and "prove" the box did not move. `.copy()` breaks the aliasing, and the
  same probe then moved `-224.236591155` -> `-218.176596704`.
- `change_box` re-sorts atoms. Any per-atom value read across a box change must be
  keyed by `extract_atom('id')`, not by row position - which is also why the
  service is immune to the aliasing above: it never holds a raw slice.

### Q68: A request channel nobody consumes, and the zero that hid it

**Answer:**
Three layers each looked correct, and together they made a conical-intersection
search run without the quantity its own method is defined on (#446):

- The unified work item has carried `nac_pairs` since the dispatch contract was
  written (`workflow/dispatch/terms.py:201`), filled from the center plan's
  electronic spec. No skill's `calculate()` took that keyword, so every request
  arrived and stopped there. The shape was right, the routing was right, and the
  consumer did not exist - which no test could see, because nothing on either
  side was failing.
- The jobtype never asked either: the deck builders gate a coupling block on
  `'nac' in jobtype_tokens`, and planning spelled only `energy gradient` for
  every `optimize`, whatever the method. A search that needs the derivative
  coupling is the *optimizer*, so the predicate is its own class flag
  (`uses_derivative_coupling`, `workflow/optimizer/meci.py:80`), and the pair
  now travels on the per-step request with it. A `mecp` pair declares False and
  asks for nothing; a backend whose manifest declares no coupling never sees the
  token.
- The reader finished the job by inventing the answer. `log_parser` returned
  `np.zeros((1, natom, 3))` when the log held no coupling table, so the
  framework saw a coupling of exactly zero, dropped the direction below its
  floor, and printed `Coupling Norm: 0.00000000` - a *measured* reading for a
  run that was never asked. A run that names no pair and gets none prints
  `Not measured` (that is the `mecp` line), which is the honest form of the same
  sentence.

- A fourth layer, found only once the first three were fixed: a compound jobtype
  resolves to **one** profile, and the ORCA table mapped
  `energy gradient nac` to `tddft_nac`, whose step list was
  `['energy', 'tddft', 'nac']` - no gradient step. The deck then carried neither
  `EnGrad` nor `irootlist`, ORCA printed no `CARTESIAN GRADIENT`, and the parsing
  side returned two identical state gradients, so the crossing's *gap* direction
  went from `Gap Norm 0.49364348` to exactly `0.00000000` while the row still
  passed. Requesting one more property can therefore subtract one that was
  working; the criterion for a compound request is that every requested step is
  still in the deck.

The rule to keep: an absent quantity travels as absence. A zero row in a
fixed-shape array satisfies every downstream shape check and turns "the engine
did not answer" into "the physics says no", which is the one misreading a
numerical framework must not make about its own inputs. When adding a payload to
a shared contract, name the consumer in the same change, and make the criterion
two-armed - the same request refused on a state table that cannot address it,
satisfied on one that can.


### Q69: Which couplings does a run compute - and how many places decided that?

**Answer:**
Five. `coupling` was classified into `nac_pair_map`/`soc_pair_map` in
`schema/normalize.py:237` (flat single center), `data/center.py:343` (per center),
`skills/nn_pes_skill/schema/normalize.py:125`, and twice in the qualification
converter (`:287`, `:2294`) - four of them agreeing by luck and none of them
checking the user's declaration. Measured, before #447: `coupling [[0,1]]` with
`nac_pair_map [[0,2]]` was accepted and the declaration overwritten;
`soc_pair_map [[1,1]]` was silently emptied; and an empty `coupling` on a run with
two or more states was *filled* with the adjacent pairs, which is why the "neither
side declares anything" case never fired - the authority was never empty. The
adjacent default also read back as a computed coupling for a pair nobody named, and
it wrote the coupling list into `transition_pair`, which is the **transition
dipole** vocabulary, not the coupling one.

The rule now (#447): one statement, one derivation. `ci`/`spin` say how many states
exist and of which spin, `coupling` says which pairs are coupled, and the two pair
maps are its only derived labels - `data/center.py::derive_pair_declaration`,
called by the flat path and by the per-center plan, fills the authority in from a
nested input that names only the pair maps and **raises** when the two sides
disagree. Nothing is invented for a centre that declares no pair: an undeclared
pair is shielded, and the question "may this run go without the coupling it
consumes" is answered where the need is known (`data/core.py::
_require_declared_couplings`, plus `_declared_pair` for a crossing search's chased
pair). `CENTER_PLAN_VERSION` moved to 1.2 so a persisted 1.1 plan - which carries
pairs its input never declared - is not reused. The one place that deliberately
keeps its own layout is the nn_pes dataset, whose channels are laid out over
neighbouring states; that convention is named where it lives rather than left as
the framework's default.

### Q70: Which rows may a reader invent?

**Answer:**
None, and six readers did. The pattern is the same in each: a fixed-shape array is
pre-allocated and the value for a quantity the engine did not print is left at
zero. The sharpest case is ORCA's SOC reader, which looked for a header the
engine never prints (`SPIN-ORBIT COUPLING (cm**-1)`; ORCA 6.1.1 prints
`CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS`) - it matched nothing on every
real log, and the `0.0` fallback made that look like six uncoupled states until
#447 turned the fallback into a refusal (measured on round `tmp_20261101_221306`;
the NAC reader had the same shape of defect and was caught the same way on round
`tmp_20261101_224544`). Measured before #447: the Molcas SOC read caught an out-of-range pair and
appended `0.0`; ORCA's SOC packer wrote `soc_matrix.get((a, b), 0.0)` and left
unmatched NAC rows at zero; the OpenQP NAC loop filled a missing `nac_{a+1}_{b+1}`
export with `np.zeros((natom, 3))`; Molcas/ORCA/OpenQP/CP2K all opened with
`np.zeros(...)`, and three of them used `max(count, 1)` so a channel nobody
requested came back with one phantom row. A zero row passes every shape check and
says "these two states are uncoupled" - for an intersystem-crossing rate that is
the run's own input, and the ISC rate of an unevaluated pair became 0 s^-1 rather
than "not measured".

The rules now: a row exists only where the engine printed one; SOC/NAC rows are
addressed by the **declared** pair order (Molcas extracts per pair from the printed
matrix, OpenQP reads one export per pair, ORCA keys its blocks by pair and refuses
a missing one); a count that cannot be mapped onto the request is refused by name;
and a pair an engine structurally cannot answer is dropped from the *request*
(ORCA's DoSOC numbers excited roots, so ground-state pairs leave `soc_pair_map`
rather than arriving as zeros). The criterion for each reader is two-armed - the
same request fails on the artifact that lacks the quantity and passes on the one
that has it.

### Q71: Who is allowed to order a state list that someone else already numbered?

**Answer:**
Only the side that received no numbering. Five skills sorted `state_spin` into
their engine's canonical order (ascending 2S) and then rebuilt the pair tables
from the sorted ladder - and the framework indexes the returned rows by the
**declaration's** order (`workflow/integrator/fssh.py`, and the same positional
read in `data/core.py::_task_property_request`). Two sorts of one list are two
index spaces with nothing between them: the row a skill returns for pair *k* is
read by the caller as the pair it declared *k*-th, so a permutation does not
change a number, it changes which pair a number is about. Nothing raises, and
every shape check still passes.

What that looked like in the code, measured on this checkout before #448: ORCA
overwrote the incoming `nac_pair_map`/`soc_pair_map` with the full cross product
of its own ladder - nine SOC rows where an input had declared two - and its
reader then fell back onto that table when the deck named no pair; molcas sorted
`target_nac_pair` in its schema **and** grouped the deck by spin in the builder,
two reorderings in one path; openqp sorted first and classified the declared pairs
against the sorted spins, so a pair was split into NAC or SOC by the wrong states.
The declaration itself did arrive intact - `schema/adapter.py` forwards `coupling`
and both pair maps through the section merge, which the plan for #448 had assumed
it did not (measured on the loading chain before touching anything). The defect was
entirely on the receiving side, which is why no key list changed.

The rules now, per ruling D1/D2: the framework delivers the state information in
the skill's target format, and a skill's own ordering must be **identity** on it -
so with a declaration present the order is checked and a non-conforming ladder is
refused by name, while the skill's own sorting and enumeration survive for what
they were always for: validating a standalone input an agent typed with nothing
declared. Molcas keeps a grouped deck because that is how `&ALASKA` is written, but
grouping decides only where a block sits, never which pair a row is: one function
(`nac_deck_order`) states the order and the reader inverts it. A pair the engine
structurally cannot answer is refused, not filtered out, so the row count the run
reports is the count the input named. Pinned by
`test_state_declaration.py::TestSkillCannotPermuteADeclaredLadder` (all five skills
in one failure list when their guards are disabled) and
`TestDeclaredPairsSurviveTheSkillEnumeration`, plus
`test_caspt2_espf.py::test_a_deck_grouped_by_spin_still_returns_the_declared_row_order`.

### Q72: How many RASSCF passes does one Molcas deck hold, and who decides?

**Answer:**
One function, asked by both sides. `input_builder.groups_by_manifold(config,
workflow)` decides whether the deck is written per spin manifold - it is true
when the workflow is generic (any ESPF run) or when the declared `state_spin`
spans more than one manifold and the profile is a property one
(`casscf_energy`, `casscf_gradient`, `casscf_nac`, `caspt2_energy`).
`rasscf_pass_count` returns the same answer to the reader, so the builder and
`log_parser` cannot disagree about how many blocks one round holds, and
`deck_loops` separately decides whether the "keep the last round" reset applies
(only SLAPAF's `Do while` and the `FOREACH` dynamics loop reprint their blocks).

Why it matters, measured before #449: the per-step asks the main driver makes
resolve to *named* profiles (`casscf_gradient`, `casscf_nac`), which took the
legacy branch where nothing grouped - one `Spin=1` block was then asked for
`ROOT=1..4` and for a `NAC=3 4` pair whose second state lived in the manifold
the deck never built. The same rule also fixes a mixed ladder whose couplings sit
inside one manifold: the other manifold used to lose its pass entirely.

When the deck does group, the embedding block must be per pass, not per deck:
`&ESPF` is written inside the group loop with `RM -FORCE ESPF.DATA/ESPF.EXTPOT`
between groups, because the ESP integrals live in one pair of files and a second
manifold without its own `&ESPF` would read the first manifold's density.
Round 17 (`tmp_20261102_195813`) read the one-pass mixed ESPF deck on the
engine as `espf, rasscf, ..., espf, rasscf, ..., rassi`; round 21
(`tmp_20261102_213544`) read the two-pass ESPF+NAC deck as
`2 &GATEWAY / 2 &ESPF / 4 &RASSCF / 4 energy tables / Happy landing`.

### Q73: A `(nstate, natom)` charge row - which state's row is embedded?

**Answer:**
The state the task itself ran, and both multiscale paths now ask the same
function. `Calculator._charge_embedding_state` (named
`_qmmm_embedding_state` before #450) takes the task's own `target_states` when
the step names any, else the system's `current_state`. The QM/MM stage already
worked this way since #36; `workflow/multiscale/charges.py::_charge_row` did not
- it took row 0 unconditionally while its own docstring said that was not a
standing rule. Since #450 both writeback sites pass the resolver per task, and a
payload that carries fewer rows than the requested state writes **no** row and
logs a warning: row 0 is not a stand-in, because embedding one surface's charges
under another surface's name is exactly the defect this channel exists to
prevent.

The engine-side caveat is recorded rather than hidden: the shipped ONIOM band's
high level is OpenQP, which returns no `computed_charges`, so round 23's green
band proves no regression but does not exercise the row choice. An ONIOM
embedding case whose high level is molcas/pyscf/xtb is the missing carrier.


## Part 15: Multi-Center QMMM Channels and Their Carriers

### Q74: How does a multi-center run name one state per center, and where must that sequence not travel?

**Answer:**
As a list in the same field a scalar would use. `Optimize.opt_state_i` /
`opt_state_j` and `Dynamics.root` take either a scalar (read by every center) or
one entry per center, and that spelling is what a pair search
(`state_i: [0, 0]`, `state_j: [1, 1]`) and a multi-center FSSH start
(`root: "2; 2"`) need. Two boundaries came with it (#460):

- the schema had kept `opt_state_i/j` scalar even after the per-center pair axis
  existed, because the tests set the attribute directly and never went through
  validation - a YAML input could not express the feature at all;
- the framework flattens its sections into every skill's config, and a skill
  solves one region with its own states delivered through `target_states`, so a
  per-center sequence there is dead data that the skill schema rejects
  (`Input should be a valid integer`). The merge now skips sequences for
  `root` / `opt_state_i` / `opt_state_j` and keeps forwarding scalars, because a
  single-center run's root and state index are what a skill has always received.

### Q75: Why is the multi-center crossing carrier a MECP and not a MECI?

**Answer:**
Because a MECI needs a derivative coupling between the two surfaces it
minimizes over, and a composite's two surfaces are sums of different centers'
states. Each center answers its own coupling between its own states, and that
matrix element is not the coupling of the two summed surfaces; no assembly of
per-center blocks produces it. Measured, the MECI spelling over C>1 prints
`Branching Directions 1` and `Coupling Norm: Not measured` by construction -
which reads exactly like the defect #446 fixed for the single-center case. A
MECP needs only energies and gradients, both of which the composite has, so
`multicenter_qmmm_mecp.yaml` is the shipped carrier and the MECI spelling is not
shipped for C>1 (`dev/methods/QMMM_DEVELOPMENT.md` section 9).

### Q76: What does a composite MD checkpoint carry that a plain one does not?

**Answer:**
Three objects, and the framework had already been carrying them: the charge
store the step fitted (one cold-start round per center), one electronic carrier
per center, and an active configuration with one entry per center. `_prepare_state`
copies the state's attributes wholesale and the partition plan's fingerprint is
checked against every history frame on load, so the four items a checklist would
name were never missing - what was missing was a criterion. A two-process run
(two steps, resume the `<title>.rt` for two more) now reproduces the one-process
four-step baseline digit for digit in `.md.xyz`, `.md.charges`, `.md.velo`,
`.md.pop` and the energy table including both per-center `E_QM` columns, and the
engine-free round trip
(`test_core.py::TestR4eSerializerVersioning.test_the_charge_store_and_each_centers_coupling_travel`)
pins the same three objects on the save-load path. The user-level `.rt` entry has
no second fingerprint source, so "refuse a restart whose fingerprint disagrees"
has nothing to compare against there; the check that exists refuses a file whose
own history was written under another partition, and a file whose constraint
digest no longer matches its input.

### Q77: What does `sc_charge_damping` change, and what must it not change?

**Answer:**
It changes the path of the charge fixed point and never what counts as
converged. Each round fits the centers' 0-order charges in the field the previous
round produced; with two centers polarizing each other that map need not be
contractive, so `q <- (1-a) q + a q_fit` with `a` in (0, 1] damps it (default
1.0 is the undamped update every run used before the key). The convergence test
is read on the raw fit against the previous raw fit, before the mixing, because a
damped move is shorter by construction and a tolerance on it would accept every
iteration at any weight. Measured: on the shipped self-consistent case a weight
of 0.5 leaves the round count and both residuals unchanged (`9.140667e-1` then
`7.000000e-4` against 1e-3) and only adds a `damping 0.5000` log line - no case in
the tree motivates the key, which is why it is documented as an escape hatch and
not a new default. The unit criterion turns a map that diverges undamped
(`q -> 1 - 2q`, iteration matrix -2) into a convergent one at weight 0.5.

### Q78: Which multi-center reading is explicitly not trusted yet?

**Answer:**
The numerical Hessian's curvature on a composite, and every number derived from
it. Two carriers now compute one (the per-center curvature case and the C=1
paired reference), and both matrices disagree with a direct single-point
difference on the same geometry: `H_xx` of composite atom 9 reads 3.2183 from
the displacement batch against 0.4011 from +/-0.005 A single points, and a
single-center arm's two hydrogen diagonals read ~1e-4 against 0.23. The
displacements themselves are right (`generate_batch` emits +/-0.005 A and the row
order matches), the electronic surface responds (the energy moves 6e-4 Hartree
for that step), the constraint block makes no difference, and clean-directory
reruns are bit-identical - so the gap is inside the batch/assembly side, and it
is open (CHANGELOG #460, `dev/TODO.md`). Until it is closed, a multi-center
`num_freq` case is a channel carrier, not a curvature reading.

---

## Part 16: Adaptive Reference Partition and Its Two Submission Routes

### Q79: What decides which reference term becomes the training label?

**Answer:**
The atom set, never the list length. `control.reference` is an independent
layered plan: `reference[0]` is its high layer, the last entry its low layer,
and `molecule.highref`/`midref` - parsed exactly like `highlevel`/`midlevel`,
1-based, flat or center-group - say which atoms those layers cover. The
planning layer then asks one question, in `planning/task.py:resolve_label_plan`:
does one reference layer cover the same atom set as the neural-network surface
(the theory electronic provider's region, which is the whole system for one
theory method and `highlevel` for a layered queue)? If yes, the label is that
layer's raw term (`source: layer`); if no single layer does but the reference
composite spans the system, it is the composite total (`source: total`);
otherwise the input is refused and the message names the reference layers it
saw. Three spellings that look like they could stand for "no layer" were
rejected by ruling and are absent from the code: `qm1+qm2` as a composite
marker, an enforced `len(reference) == len(theory)`, and a `none`/`x`/`no`/`off`
placeholder - an unneeded layer is omitted, not padded.

`dev/methods/ADAPTIVE_DEVELOPMENT.md` section 14 is the design entry; the
derived record a run writes is `&method alignment` in the log and
`method_alignment` in `<title>.adaptive.json`. Measured on the two shipped
solution carriers (`examples/adaptive_oniom2_local.yaml`,
`examples/adaptive_qmmm_local.yaml`), both report `source: layer` from the same
`qm` task over the same six atoms, and their returned labels agree bit for bit
(five frames of two states, `max |difference| = 0.0` on energy and gradient) -
which is the same fact `schema/validate.py` states as an advisory when a
classical outer method meets `embedding: false`.

### Q80: Where does the framework stop and the skill start when a channel is optional?

**Answer:**
The framework offers data; the skill consumes what it declares and never the
other way round. Concretely: a reference that produced an external potential
hands `field` rows to the QM task, but a skill with no field input ignores them
and the run is not refused; a skill that requires data it did not receive
raises from its own boundary, and that error is the skill's sentence, not the
framework's. Two consequences that were enforced during this build: planning
has no branch keyed on a skill name (no `if skill == 'pyscf_skill'`), and a
redundant input is not an error - so the same `&adaptive` queue is certifiable
on a host whose engine lacks one of the auxiliary channels.

The negative form of the rule matters as much: an absent quantity travels as
absence. `TrainingData` carries no field/MM channel, so an auxiliary that a
frame did not produce is not written as a zero row that a later reader cannot
distinguish from a real one.

### Q81: Why do the two remote stages of an adaptive run get submitted differently?

**Answer:**
Because one is a workflow runtime and the other is a skill call, and the two
have different owners of the job script.

- **MD** (`md_remotetask: true`): the framework serializes the child state to
  `traj_*.rt`, submits that, and the child runs `python -m pyrai3md traj_*.rt`
  on the compute node. The skill inside it executes `local` - it has no job to
  submit, it is already inside one.
- **Reference** (`reference_remotetask: 1`): the framework does **not** submit a
  runtime. It flips the reference skill's own `execution_mode` to `slurm`; the
  skill writes its own sbatch script, and the job body runs the skill agent
  with the serialized config in-process.

The split is what a bug made expensive. The serialized skill config carried
`execution_mode: slurm`, so the agent that woke up inside the compute node read
slurm and submitted a second job, which submitted ... - one adaptive round
fanned out to hundreds of PySCF jobs before it was stopped (the counts are the
`squeue`/`scancel` readings of that round, not a log that survived). The fix is
in the job body, not in the caller: `skills/pyscf_skill/service/runner/slurm_runner.py`
and the qdm equivalent force `environment.execution_mode = 'local'` on the
config they serialize into the script, because the job body *is* the remote
execution. The unused trailing `local` argument the template also passed was
removed - it was being ignored, which is how the recursion stayed invisible.
`dev/SKILL_APP_DEVELOPMENT.md` section 5.8 states the rule for every skill.

### Q82: When does an adaptive iteration write `aux-labels.json`, and who reads it?

**Answer:**
Only when a frame really produced an auxiliary channel - one of
`field`/`mm_energy`/`q_qm`/`mid_charges`/`low_charges`. `region_indices` is the
region a label was taken from, so a metadata-only sidecar was the first build's
output for a mechanical-embedding ONIOM run and is now suppressed
(`workflow/adaptive.py`, the `auxiliary` gate in `_reference_iteration`). Measured
on the post-fix local reruns: `adaptive_qmmm_local` writes five rows carrying
`mm_energy` and six `q_qm`, `adaptive_oniom2_local` and `adaptive_single_local`
write no file at all.

Nothing in the calculation consumes the sidecar - neither the dataset, the
training child nor the next iteration reads it; the only reader is
`tools/verify.py` section J, which requires it for a QMMM carrier and forbids it
for the others (`unexpected_aux` is a counted problem). That asymmetry is the
point: the file is provenance for the decomposition behind a label (`E = E_QM +
E_classic`, the field the QM region saw, the charges written back), which the
energy and gradient columns alone cannot show.

### Q83: What must be stripped before an adaptive child is dispatched?

**Answer:**
Everything the parent sized for the whole system. A sliced reference or training
child is a serialized `PyRAIState`, and three classes of carrier survived slicing
and then broke on the child's smaller geometry: the parent-shaped inputs
(`highlevel`/`midlevel`/`highref`/`midref`, `q_qm`, `point_charges`,
`external_potential`, `external_charges`, `embedding`, `read_charge`), the
parent's planning and electronic state (`electronic`, `active_surface`,
`charge_store`, `partition_plan`), and the duty markers that route a task
(`composite`, `qmmm_role`, `binding_role`, `partition_indices`,
`probe_indices`) - a lone training task that still said `composite=qmmm` tried to
assemble a composite it had no queue for. The matching skill-config fields go
too, and the whole skill config is made JSON-safe recursively, because an
`ndarray` anywhere in it aborts the sbatch submission with a serialization
error rather than a scientific one. Two readings this came with: a dispatched
training child returns a serialized state document, so the verdict must unwrap
`pmd.results` before it asks whether a train record exists; and a submitted
child runs in its own workdir, so the launch-directory fixtures (`.data`,
`.in.init`, `.in.settings`, `.charge`) are copied into it by
`_stage_runtime_fixtures` or the engine fails on a topology only the parent was
given.

### Q84: How is a whole example family asked for?

**Answer:**
By one family switch above the per-case keys and below `_runall`.
`manager.EXAMPLE_FAMILIES` is now six entries, matched by `startswith` with the
first hit winning, and the membership rule is the **name**, not the physics.
Measured over the 201 shipped example YAMLs: `md_lammps_` 12, `constraint_md_`
33, `optimize_` 25, `path_` 32, `multiscale_` 37, `adaptive_` 6, with no stem
matching two prefixes - which is what makes the row order of the table
irrelevant, and `integration/test_integration_manager_planning.py::test_optimize_prefix_shadows_no_other_family`
enumerates the table to keep both properties true as it grows. `adaptive_test`
is the sixth and joins it the way the other five did: `true` opens all six
carriers whatever their keys say, `false` closes them, `condition` (the shipped
value, and the value of an absent key) leaves the six keys in charge - so the
local half of the family runs on any host and the slurm half is a named skip
rather than a failure where there is no client.

---

## Part 17: Pre-Commit Review Contracts (2026-11-12)

### Q85: Why do skill packages export their backend lazily?

**Answer:**
Because discovery must not import computational modules. Every shipped
`skills/<name>/__init__.py` now declares `__all__` and a PEP 562
`__getattr__`; the first access to `<Skill>Backend` imports `.backend`, and
nothing else does. Measured per skill in a fresh interpreter: importing
`pyrai3md.skills.<name>.query` leaves `<name>.backend` and its service tree
out of `sys.modules`. `discover_all()` catches an import failure and warns;
before, it dropped the skill from the registry with `pass`, so a broken
optional dependency could make a skill disappear.

### Q86: Why is `read_support` copied into every skill?

**Answer:**
Because a skill must be independently distributable. The seven skills with a
`service/read.py` entry (orca, bagel, molcas, lammps, openqp, cp2k, xtb) now
each own `service/read_support.py`, including the small XYZ-frame reader it
needs. The former shared `pyrai3md/skills/read_support.py` is gone, so no
skill imports another framework module on the read path. The copies are
deliberately duplicated; a contract test is the place to catch drift.

### Q87: How does planning read optimizer metadata without importing workflow?

**Answer:**
Through pure data. `schema/optimizer_spec.py::OPTIMIZER_DECLARATIONS` carries
`implemented`, `needs_initial_hessian`, `uses_derivative_coupling`,
`uses_state_pair` and `is_path` for the eight search names; planning reads
that table, and `workflow/optimizer/__init__.py::check_declarations()` holds
the classes to it. The fix mask planning needs is another pure function,
`data/constraint.py::build_fix_mask`, not `workflow.constraints.projection`.
`planning` therefore has no import edge to `workflow`; the only registered
edge points the other way (workflow reads three task-descriptor selectors from
`planning/task.py`).

### Q88: What happens when a requested parser block is missing?

**Answer:**
It raises. Molcas `parse_log` raises when the workflow requested an energy or
gradient and the completed log has no block; CP2K raises when `stress` was
requested and no stress block exists, and when the workflow asked for energy
but the log has no `Total energy:` line. A zero array is reserved for a
property nobody requested (the TDDFT excited-state rows still reference 0.0
when energy was not asked for). The rule is the same one Q70 states for
couplings: an absent quantity travels as absence, never as a zero the next
reader cannot distinguish from a measurement.

### Q89: How does qualification avoid a false green?

**Answer:**
Three changes close the known channels. The six post-run artifact checks now
run in their own `try`; an exception sets `status='error'` with the traceback
instead of leaving the earlier PASS in place. A unittest file that collects
zero cases is an error, not a pass (pytest-style modules and helper files used
to certify themselves). A planner exception produces `_make_error_task`, an
error row; only a genuinely inapplicable case is a skip. The report still has
to be read by row: a closed switch or a host gate is a skip, not evidence the
case ran.

### Q90: What controls the shared `skills/tests` pytest group, and does `_runall: true` open it?

**Answer:**
It has no key of its own. It follows the aggregate of the skills'
`runtime_tests` switches: open when at least one skill's phase is open, and
listed as skipped with `disabled by skills.*.runtime_tests` when every skill's
phase is closed. `_runall: true` is the master that forces every listed
skill's phase open (`force_all`), so the per-skill pytest rows and the shared
group are planned even when the shipped skill sections say
`runtime_tests: false`. `_runall: false` does not close the pytest phase by
itself; `runtime_tests` is the leaf switch.

### Q91: Which tests does the remote plan include?

**Answer:**
All package test files, grouped, not one monolithic pytest process. The
current plan contains 153 system tasks - 136 `qualification/verification/`
files plus 17 `qualification/integration/` tasks (one file is sharded into
eight) - and 78 `skill_runtime` rows: 73 files in the ten skills' own
`tests/` plus five in `pyrai3md/skills/tests/`. The remote run is a
file/shard-level plan: a row runs every case in its file, but a skip, host
gate, timeout or import error is not a pass. The plan does not include
`tools/` tests or code outside the package.

### Q92: Where should a test live?

**Answer:**
A pure skill-internal test lives in `<skill>/tests/` and imports only that
skill. A test that exercises the framework route or a cross-skill contract
lives in `qualification/verification/` (unittest) or
`pyrai3md/skills/tests/` (framework-level pytest). This review kept
`test_orca_correction_bound` and `test_resident_error_boundaries` in their
skills, and kept `test_flatten`, `test_lammps_persistent_test_plan` and
`test_nn_pes_seed_repro` in verification because they need the framework or
qualification route. The skill tree is scanned for framework imports; a test
that needs the framework belongs on the framework side.

### Q93: What changed in dispatch and the skill wrapper?

**Answer:**
The remote term worker no longer carries a private script builder: it calls
`slurm_run._build_script(..., entry='.rtt')`, which keeps the SBATCH
directives, the user `shell` section and the environment exports identical to
a whole-runtime job. The Slurm poll has a wall-clock deadline and a
consecutive-`UNKNOWN` cap, so a scheduler that never settles raises instead of
hanging the parent. `pyrai3md_skill` validates `skill_name` against
`list_skills()` and passes an argv list to `subprocess.run`; the former
`shell=True` string interpolation is gone.

### Q94: Why is `schema/reference.py` import-free, and where did the unit constants go?

**Answer:**
Because `utilities` must be able to import `schema.reference` without a cycle,
and the architecture rule says that file is pure data. The nine unit
constants it used to re-export are imported by `schema/__init__.py` directly
from `utilities.units`; `pyrai3md.schema.BOHR_TO_ANGSTROM` still works, while
`reference.py` contains no import statement. A guard test parses the file and
asserts that no `import`/`from ... import` node exists.

