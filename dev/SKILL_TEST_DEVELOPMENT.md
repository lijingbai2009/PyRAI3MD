# Skill Test Development Guide

Last updated: 2026-11-12 19:20 (section 10 records the test-placement rule: pure skill tests in skill/tests, framework/contract tests in verification or skills/tests. Earlier: 2026-11-02 10:40 (#448: new 4.5 and 4.6 (#448): row-shaped assertions follow the declaration, a fixture log printing fewer transitions than the case declares is a named failure, and the empty-table arm is the only thing that catches the `else 1` padding - measured by restoring it. The rest was not re-audited. Before it, 2026-10-20 08:10 (§4.3 now records the 2026-10-20 ruling that no skill test YAML uses `expected:` at this stage; values and comparator wait for the final pre-release test. Earlier, 2026-10-16 16:51: the pmd_env.yaml switch census re-counted against the file (10 skill sections / 107 case switches, all ten `test: false` for the optimize round) and re-read against `certification/planner.py` for #359/#360: `runtime_tests` is the only switch for a skill's own pytest phase, and the shared `pyrai3md/skills/tests/` group is planned once per round with no key; the discovery tree gained both. The surrounding sections were not re-audited line by line))

This document defines the test architecture for PyRAI3MD skills.
The test system verifies both independent skill functionality and
framework integration through three execution modes.

---

## 1. Design Objectives

The test system follows three principles:

### 1.1 Skill Independence

A skill is developed, tested, and maintained without requiring the PyRAI3MD framework.
All scientific test knowledge belongs to the skill; the framework only owns orchestration.

### 1.2 Unified Runtime Protocol

All three test modes start from the same flat `.json` config generated
by the certification converter. The JSON config is the common protocol
shared by skill agent, pyrai3md agent, and classic input; the converter
derives the sectioned dict consumed by the agent (`.config`) and classic
(`.input`) modes from it.

### 1.3 Framework Decoupling

PyRAI3MD does not understand skill-specific scientific parameters or input generation logic.
The framework discovers skills via the query mechanism and discovers each skill's test files through the certification handler.

---

## 2. Test Architecture Overview

```
pmd_env.yaml (skills.<name>.test: true | false | condition,
              plus tests.<case> and runtime_tests under condition)
        │
        ▼
qualification/manager.py  →  qualification/certification/planner.py
        │
        ├─ discover_all()  →  skills/query.py  →  <skill>/query.py
        │
        ├─ handler.list_skill_runtime_tests  →  <skill>/tests/test_*.py
        │                                        (pytest, planned first, and
        │                                        planned even when `test` is
        │                                        closed: `runtime_tests` is
        │                                        the only key that speaks)
        │
        ├─ handler.list_shared_skill_tests  →  skills/tests/test_*.py
        │                                        (the layer the skills share;
        │                                        one group per round, no
        │                                        switch of its own)
        │
        ├─ handler.list_skill_test_yamls  →  <skill>/tests/*.yaml
        │
        ├─ converter.generate_test_configs(skill_dir)  →  flat .json files
        │
        └─ certification/converter.py  →  .input + .config files
        │                                  (pyrai3md_input / pyrai3md_agent)
        ▼
qualification/runner.py
        │
        ├─ Mode 1: pyrai3md_skill <name>.json     (skill_agent, local)
        ├─ Mode 2: pyrai3md_agent <name>.config   (pyrai3md_agent, local/slurm)
        └─ Mode 3: pyrai3md <name>.input          (pyrai3md_input, local/slurm)
        │
        ▼
results.json + status.json
```

The runtime test scripts (`test_*.py` shipped in each skill's `tests/` directory) run ahead of the YAML task tests, executed in place by pytest; benchmark calibration tests form the fourth plan category and run after all functional tests (see QUALIFICATION_DEVELOPMENT.md Section 6.8).

### 2.1 Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| Manager | `qualification/manager.py` | Aggregate the four plan categories (main/example/skill/benchmark) + shared script writers |
| Certification planner | `qualification/certification/planner.py` | Plan skill tests (`plan_skill_tests`): runtime tests first, then the three-mode YAML task tests |
| Handler | `qualification/certification/handler.py` | Discover the test YAMLs and `test_*.py` runtime scripts under each skill's `tests/` + generic skill section builder |
| Converter | `qualification/certification/converter.py` | Test YAML → flat .json (`parse_test_yaml`/`generate_test_configs`); flat .json → (load_config + `_flatten_legacy_sections`) → sectioned dict → {.input, .config} dual output |
| Runner | `qualification/runner.py` | Execute tests via subprocess, collect results |
| Query | `<skill>/query.py` | Provide skill metadata for discovery |

---

## 3. Test Case Ownership

Test cases belong to individual skills. Each skill organizes tests under its own `tests/` directory:

```
<skill_name>/
├── tests/
│   ├── dft_energy.yaml     # Test case definition
│   ├── dft_gradient.yaml
│   ├── hf_energy.yaml
│   ├── test_*.py           # Skill-owned runtime tests (pytest)
│   └── ...
├── query.py                # Skill metadata (sole exit point)
├── manifest.yaml           # Discovery metadata
├── backend.py              # Computational API
└── agent.py                # CLI entry
```

PyRAI3MD does **not** define scientific test cases. It does not know:
- what computational methods a skill supports
- what parameters a skill requires
- how a skill generates its backend input

These are all skill-specific knowledge carried by the skill's test YAML files; the shared certification converter parses them into flat configs.

---

## 4. Test Case Format

### 4.1 YAML Structure

Each test YAML contains two sections:

```yaml
version: 1.0
name: dft_energy_test

input:
  atoms: [C, H, H, H, H]
  coordinates:
    - [0.000,  0.000,  0.000]
    - [0.629,  0.629, -0.629]
    - [-0.629, -0.629, -0.629]
    - [0.629, -0.629,  0.629]
    - [-0.629,  0.629,  0.629]
  charges: 0
  multiplicity: 1
  nstate: 1
  profile: dft_energy
  method: dft
  functional: pbe0
  basis: ccpvdz

expected:
  energy:
    value: -40.518362
    tolerance: 1e-6
```

### 4.2 Input Section

The `input` section contains parameters meaningful to the certification converter.
It does **not** need to match PyRAI3MD Config, PyRAI2MD input, or external software input.
It only needs to be sufficient for the converter's `parse_test_yaml` to produce a valid flat .json config.

### 4.3 Expected Section

The `expected` section is the reserved home of validation information. **No
code reads it, and by ruling of 2026-10-20 no skill test YAML uses it at this
stage**: the block stays as a placeholder, and reference values are set only in
the final test before release, together with the comparator that reads them.
Until then the `value:` fields stay empty or `null`; a few blocks keep only
array `shape` keys, which are equally unread. A filled value would document a
number without guarding it.

```yaml
expected:
  energy:
    value: -40.518362     # Reference value (null during development)
    tolerance: 1e-6       # Absolute tolerance

  gradient:
    rms: 1e-6             # RMS tolerance for arrays

  nac:
    shape: [10, 10]       # Expected array shape
```

Filling a value in here is only useful together with the comparison that
reads it; on its own it documents a number without guarding it. That is why
the pre-release pass is the one that fills it.

### 4.4 Naming Contract

The YAML top-level `name` is the **single naming source** for the generated
task. By convention it equals the YAML file stem plus a `_test` suffix
(e.g. `dft_energy.yaml` -> `name: dft_energy_test`), so every generated
task title/project is explicitly marked as a test artifact and stays
isolated from production run names:

- The converter `parse_test_yaml` injects it into `config['title']`
  verbatim (falling back to the YAML file stem when the `name` key is
  absent); it does NOT add or strip the `_test` suffix automatically,
  so the suffix must be carried by the YAML `name` itself;
- The converter `parse_test_yaml` also injects the same value into
  `config['<skill>_project']` (unless the YAML input section defines an
  explicit project), so every test keeps uniquely named output files even
  when several tests share one profile;
- The converter `_derive_project_name` then derives the skill project name
  with priority `title → profile → 'pyrai3md'`, and the same `title`
  becomes the classic input `&CONTROL title`.

Tests sharing one profile (e.g. two `casscf_geometry_optimization` cases)
stay distinguishable through distinct `name` values.

### 4.5 Row-shaped results are asserted against the declaration

The coupling and transition channels return **one row per declared pair, in the
declaration's order**, so the unit fixtures that assert their shapes assert the
declaration too (#448). Three consequences for whoever edits a case or a fixture:

- A shape assertion is `(len(pair_table), ...)` or `(len(pair_table), natom, 3)`,
  never `(nstate, ...)`. When the default transition table lost its self pair
  `(0, 0)`, every fixture that indexed row 1 for the first excited state moved to
  row 0 - bagel, cp2k and orca each had one, and the move is the reading, not a
  loosened expectation.
- A fixture log that prints **fewer** transitions than the case declares is now a
  failure with a name in it (bagel refuses the pair, orca refuses the SOC element,
  molcas refuses a block set that differs from the requested pairs). The honest
  fix is to declare what the log answers or to print the missing block; padding
  the expectation with a zero row is the fabrication these readers were cleared
  of.
- A case that declares no transition pair must return an **empty axis**, and each
  reader has an arm for it (`test_no_declared_pair_leaves_the_channel_empty` and
  its twins). This is the only arm that catches the `npair = len(pairs) if pairs
  else 1` padding: with every shipped fixture declaring at least one pair, that
  branch is unreachable, so a mutation restoring it passes the rest of the suite
  silently.

### 4.6 What the skill may not do to a forwarded declaration

A skill test case that names `coupling`, `state_spin` or a pair table is a
framework run's input in miniature, so the skill's own canonical ordering must be
identity on it. Sorting `state_spin` and re-enumerating the pairs are allowed only
when the case declares nothing - which is precisely the standalone agent path this
document describes - and a case that declares an unsorted ladder is refused by name
(`TestSkillCannotPermuteADeclaredLadder`). Pairs the engine cannot answer are
refused the same way instead of being filtered out, so a case that asks ORCA for a
ground-state SOC element or a non-S0 NAC element fails at normalization rather than
returning one row fewer.

---

## 5. Test Config Generation (certification converter)

### 5.1 Purpose

The certification converter (`qualification/certification/converter.py`)
converts the test YAML files of every skill into flat `.json` configs —
the unified runtime protocol source. The parsing is skill-agnostic:
all skill-specific rules (project naming, coupling defaults,
`current_state` routing) follow generic rules keyed by the skill
identity through the skill's public query interface, so skills ship
only their `tests/` directory and no per-skill generator.

### 5.2 Entry Functions

```python
def parse_test_yaml(yaml_path, skill_name):
    """Parse one skill test YAML into a flat skill config dict.

    Copies every key of the YAML 'input' section and injects the
    skill metadata (skill_name, skill_path, theory) through the
    skill's query.get_metadata(). The yaml top-level name becomes
    config['title'] (file stem fallback) and the '{stem}_project'
    default; a case that declares pair maps gets them rendered into
    `coupling`, and one that declares neither keeps no coupling (#447);
    'current_state' is nested into a 'system' section.
    """

def generate_test_configs(skill_name, output_dir):
    """Generate flat .json configs for all test cases of one skill.

    Discovers every test YAML through the handler, points
    scratch_path at an isolated output_dir/tmp, writes
    {profile}.json files and copies the test data files next to
    the configs so the launch directory is self-contained.

    Returns
    -------
    dict
        Mapping of {profile_name: config_path}; empty when the
        skill ships no test YAML files.
    """
```

The `expected` section is **not read by any code path today**. `parse_test_yaml`
copies the YAML's `input` section only, the runner judges a skill task by its
exit status plus the artifact checks it performs itself, and nothing anywhere
compares a result against `expected.energy.value`. The block is kept in the
YAMLs as the place a reference value will go, and `value: null` is the honest
state of every one of them; the tolerance numbers sitting beside it do
nothing. Writing a measured value here without implementing the comparison
would create a gate that looks armed and judges nothing — which is the failure
mode section 18 of `OPTIMIZE_DEVELOPMENT.md` had to remove from
`opt_status.txt`. Result validation for a skill case therefore means: the
qualification suite proves the run completes and the shapes are right, while
the numbers are judged by the skill's own acceptance tool (for lammps,
`bringup`'s cold-versus-resident comparisons) and by the golden text digests in
`tests/`.

### 5.3 Unified Parsing Pattern

The converter obtains skill metadata from the skill's own `query.py`:

```python
import importlib

def parse_test_yaml(yaml_path, skill_name):
    test_case = yaml.safe_load(open(yaml_path)) or {}
    input_section = test_case.get('input', {})

    # Build config: copy all input fields, then inject metadata
    config = dict(input_section)

    # Single naming source: yaml top-level name (carries the _test
    # suffix by convention), file stem fallback; passed through verbatim
    config['title'] = test_case.get('name') or os.path.splitext(
        os.path.basename(yaml_path))[0]

    # Skill metadata through the public query interface
    query = importlib.import_module(
        f'pyrai3md.skills.{skill_name}.query')
    meta = query.get_metadata()
    config['skill_name'] = skill_name
    config['skill_path'] = meta['skill_path']
    config['theory'] = [meta['name']]

    # Project shares the naming source with the title
    stem = skill_name[:-6] if skill_name.endswith('_skill') else skill_name
    config.setdefault(f'{stem}_project', config['title'])
    return config
```

### 5.4 Output Format

The generated `.json` file is a flat dictionary containing all fields
needed by the skill backend's `build_config()`:

```json
{
  "atoms": ["C", "H", "H", "H", "H"],
  "coordinates": [[0.0, 0.0, 0.0], ...],
  "charges": 0,
  "multiplicity": 1,
  "nstate": 1,
  "skill_name": "pyscf_skill",
  "skill_path": "backend",
  "qm": ["pyscf"],
  "profile": "dft_energy",
  "method": "dft",
  "functional": "pbe0",
  "basis": "ccpvdz"
}
```

This flat JSON config is the **common source** for all three test modes:
Mode 1 consumes it directly, while the converter renders the same
sectioned dict into `.input` and `.config` for Modes 3 and 2.

### 5.5 Test Data File Distribution

Skills whose test cases depend on auxiliary data files ship them in the
same `tests/` directory. The lammps skill is the current example
(`h2o.data`, `h2o.in.init`, `h2o.in.settings`).

`generate_test_configs()` copies every file in the skill's `tests/`
directory that is neither a test YAML nor a python module next to the
generated configs in the qualification working directory, so the tests
find them in the launch directory independent of the installation
path. The data files themselves are shipped through `setup.py`
`package_data` (`pyrai3md.skills.lammps_skill: tests/h2o.data,
tests/h2o.in.init, tests/h2o.in.settings`), and
`test_skill_packaging.py` guards the packaging declarations.

### 5.6 Where an engine run belongs: the case files, not the unit tests

A claim can only be measured against the thing it is about. Two different
owners split those claims in this skill:

| Owner | Certifies | Runs a real engine? |
| --- | --- | --- |
| the case files `tests/00_mlp_eg_train.yaml` … `07_mlp_eg_predict_eval.yaml`, through `pyrai3md_skill` / `pyrai3md_agent` / `pyrai3md <input>` | the shipped route end to end: a fit in subprocesses writes a folder the next case loads and serves | yes, and that is their subject |
| `tests/test_train_predict.py` | that same route inside one process, plus what the fit wrote: per-member hyperparameters in each `hyper_v<i>.json`, the stored error unscaled by the folder's own scaler, the gradient being the derivative of the reported energy | yes |
| every other `tests/test_*.py` in the package | functions: which channels a record carries, what a statistic reduces, which directory survives, how a request is routed, what a schema refuses | no |

The middle row is the only unit-test module that trains, and it does so
because the subprocess fan-out *is* what it certifies: `nn_pes.py:146` starts
one fit per ensemble member, and each child imports TensorFlow (4.6 s
measured for a bare import here), so a run of that module costs 71 s of
which the epochs are almost nothing (measured: 30/25 epochs → 73.2 s,
5/4 epochs → 71.4 s).

What was wrong before, and twice in a row:

1. `test_predict_products.py` and `test_keep_tmp.py` each trained a small
   surface just to have a folder to serve from. Neither module asserts
   anything about a fit: the first is the prediction record, its statistics
   and its files; the second is directory lifecycle. They cost 78.1 s and
   34.7 s for it.
2. The first fix -- a per-process shared fit (`_shared_fit.py`) -- made those
   modules cheaper by reusing someone else's training, but it kept the wrong
   dependency: an engine run still sat inside function tests, and the suite
   only went 188.1 s → 153.7 s. It also hid a trap: when that cache's
   `setUpClass` failed, five tests vanished from the count (232 → 227) while
   the run still reported one error -- a fixture failure silently deletes
   coverage, so a suite count has to be compared to a baseline, not just
   grepped for `OK`.

The current shape replaces the fit with the container's own contract
(`tests/_fake_engine.py`): a `FakeContainer.call()` answering with a surface
whose derivative is its own energy, and label files written as that surface
plus a known shift. Everything downstream of the container stays real code,
and the assertions get *stronger* because the expected numbers are computed
by the test: the energy error must be `shift * HARTREE_TO_EV`, and the
gradient one distinguishes the per-Angstrom axis from the per-Bohr axis by
three significant figures rather than by tolerance. Geometries are chosen
dyadic because the serving path casts coordinates to float32 -- a coordinate
like `0.7` would inject a 6e-8 rounding into a claim that is otherwise exact.

Measured: `test_predict_products` 78.1 s → 0.06 s (19 → 20 tests),
`test_keep_tmp` 34.7 s → 0.04 s (4 → 5 tests), and both now run on a host
without TensorFlow. Suite totals are recorded in CHANGELOG #319.
---

## 6. Three Test Modes

For each generated flat `.json`, the certification planner creates three test tasks:

### 6.1 Mode 1: Skill Agent Test (`skill_agent`)

**Purpose:** Verify the skill works independently without PyRAI3MD orchestration.

```bash
pyrai3md_skill <name>.json
```

**Workflow:**

```
flat .json → pyrai3md_skill → skill.agent → backend → result
```

- Always runs locally (no slurm)
- The skill manages its own schema pipeline: `build_config() → normalize() → validate()`
- Does not require PyRAI3MD framework modules
- Performs **no pmd_env.yaml merge** at runtime. Machine-specific
  parameters (e.g. `molcas_path`, `cp2k_path`) are injected into
  the flat `.json` by the certification planner at plan time from the
  merged `skills.<name>` section of pmd_env.yaml. A skill-level
  `shell` key (skills.<name>.shell) rides along the same injection so
  the skill run scripts write the module loads; the global
  `environment.shell` is reserved for the main framework and never
  reaches skill configs (#134). Existing non-empty values from the
  skill test YAML are never overwritten; the `test` switch itself is
  never injected.
- Execution control keys (`execution_mode`, `slurm_*`) use overwrite
  semantics in the injection (#80); all other keys use fill-if-empty.
- After the environment injection the planner forces `driver: skill`
  into every skill-test config (`_force_skill_driver`, unconditional
  override): skill test cases are skill-driven by definition, so the
  `pyrai3md_input` mode gains a `driver skill` line in `&CONTROL`
  (written by the converter) and `pyrai3md_agent` routes through the
  skill driver; only the main framework examples run `driver=main`
  (#107).
- The input `.json` is **never modified** by execution (#143): the
  backend no longer writes the validated config back to the input file,
  so Mode 1 wrapper scripts are idempotently re-runnable. The wrapper
  resolves `skill_name` top-level first with an `environment` section
  fallback, and the converter flattens the legacy five-section layout
  (system/properties/dynamics/calculation/environment), so files already
  rewritten by the old write-back remain consumable (#142).
- Every skill provides `resolve_profile(method, jobtype)` in
  `schema/method.py`, mapping method + jobtype combinations to the
  correct profile name. This is required for the classic input flow
  (`.input` files) where the framework must derive the profile from
  the parsed method and jobtype tokens (#84).

### 6.2 Mode 2: Agent CLI Integration Test (`pyrai3md_agent`)

**Purpose:** Verify PyRAI3MD agent communication and skill loading.

```bash
pyrai3md_agent <name>.config
```

Mode 2 consumes the converter's **sectioned dict `.config` file** — not
the raw flat `.json`. The converter builds one shared sectioned
dict from the flat `.json` and serializes it as the `.config` JSON (the
same dict the `.input` text is rendered from), so agent mode consumes
exactly the same effective configuration as input mode.

**Workflow:**

```
flat .json → converter sectioned dict → <name>.config
    → pyrai3md_agent → agent.initialize()
    → resolve_skill() → load_skill()
    → _merge_skill_config() (4-layer merge)
    → backend → result
```

- Can run locally or under slurm
- Tests the full framework pipeline: skill discovery, registry loading,
  config merging (merge-before-normalize), backend initialization
- Verifies: skill discovery, registry loading, agent protocol, runtime initialization

### 6.3 Mode 3: Traditional CLI Test (`pyrai3md_input`)

**Purpose:** Verify compatibility with the PyRAI3MD legacy input workflow.

**Workflow:**

```
flat .json → qualification/certification/converter.py → test.input + test.xyz [+ test.velo]
    → pyrai3md test.input → classic parser → schema pipeline → result
```

- Can run locally or under slurm
- Tests the full classic pipeline: text input parsing, schema validation,
  config merging, backend execution
- The converter generates a same-basename `.xyz` file for coordinates
  (`_write_xyz_file`) and a same-basename `.velo` file when the config
  carries inline velocities (`_write_velocity_file`, dynamics tests;
  read by the framework through the `title.velo` naming convention)

### 6.4 Mode Comparison

| Aspect | skill_agent | pyrai3md_agent | pyrai3md_input |
|---|---|---|---|
| Command | `pyrai3md_skill` | `pyrai3md_agent` | `pyrai3md` |
| Input format | flat .json (converter) | sectioned dict .config (converter) | .input text + .xyz [+ .velo] (converter) |
| Schema pipeline | Skill-owned | Framework merge-before-normalize | Classic parser + framework |
| pmd_env.yaml merge | None (injected at plan time) | Runtime | Runtime |
| Slurm support | No | Yes | Yes |
| Framework dependency | None | Full | Full |

---

## 7. Qualification Architecture

### 7.1 Manager (`qualification/manager.py`)

The manager aggregates the test plan from `pmd_env.yaml` through the
three planners:

```python
def generate_plan(config, work_dir=None):
    plan = {
        'main_tests': plan_main_tests(),               # verification/planner.py
        'example_tests': _plan_example_tests(config, work_dir),
        'skill_tests': plan_skill_tests(config, work_dir),  # certification/planner.py
        'benchmark_tests': plan_benchmark_tests(config, work_dir),  # benchmark/planner.py
        'slurm': config.get('slurm', {}),
        'export': _merge_exports(config),
        'work_dir': work_dir,
    }
    return plan
```

**Key responsibilities (skill side, certification/planner.py):**

1. **Skill discovery**: Calls `discover_all()` from `skills/query.py` to find registered skills
2. **Switch resolution**: `_runall` (master) then `skills.<name>.test`, both three-valued; `true` certifies every case of the skill, `false` closes them all, `condition` reads `runtime_tests` and the per-case `tests.<case>` map. A closed skill or case is planned as a skipped task naming the key that closed it, never silently omitted. "Them" is the YAML cases: the `tests/test_*.py` phase answers to `runtime_tests` alone and is planned under all three settings (#359), which is why an optimize round still executes 64 engine-free code checks
3. **Runtime tests first**: Plans each skill's `test_*.py` runtime scripts (pytest in place) ahead of the YAML task tests
4. **Config generation**: Calls `converter.generate_test_configs()` to write the flat .json configs
5. **Script generation**: Creates `.sh` (local) and `.sbatch` (slurm) scripts for each task (shared writers in manager.py)
6. **Legacy conversion**: Calls `converter.convert()` for pyrai3md_input mode and serializes the same sectioned dict as `.config` for pyrai3md_agent mode
7. **Driver forcing**: `_force_skill_driver` injects `driver: skill` into every skill-test config (unconditional override)
8. **Diagnostic tasks**: Creates skip tasks when test configs cannot be generated

**Working directory structure:**

```
tmp_YYYYMMDD_HHMMSS/
├── pyscf_skill/
│   ├── dft_energy.json         # Flat skill config generated by the certification converter
│   ├── dft_energy.input        # Generated by converter.py (sectioned dict text)
│   ├── dft_energy.config       # Generated by converter.py (sectioned dict JSON)
│   ├── dft_energy.xyz          # Coordinate file for classic input
│   ├── dft_energy.velo         # Velocity file for classic input (dynamics tests only)
│   ├── dft_energy_skill.sh     # Local script for skill_agent mode
│   ├── dft_energy_agent.sh     # Local script for pyrai3md_agent mode
│   ├── dft_energy_agent.sbatch # Slurm script for pyrai3md_agent mode
│   ├── dft_energy_input.sh     # Local script for pyrai3md_input mode
│   ├── dft_energy_input.sbatch # Slurm script for pyrai3md_input mode
│   └── tmp/                    # Scratch directory
├── molcas_skill/
│   └── ...
├── results.json
└── status.json
```

### 7.2 Runner (`qualification/runner.py`)

The runner executes all tests and collects results:

```python
def run_tests(plan, config):
    results = {                       # four categories, and the aggregate the
        'main_results': [],           # report line sums is exactly these four
        'example_results': [],
        'skill_results': [],
        'benchmark_results': [],
        'timestamp': '...',
        'status': 'running',
    }
    _stdout_line_buffered()           # ordering: header must precede tasks
    _print_run_header(results, total) # every line flushed
    for task in main_tests:           # unittest, captured per task
        result = _run_main_test(task)
        _print_single_result(result, idx, total)
    for task in example_tests:        # the generic subprocess path as well
        result = _run_skill_test(task, plan, config)
        _print_single_result(result, idx, total)
    for task in skill_tests:          # subprocess, plus artifact read-back
        result = _run_skill_test(task, plan, config)
        _print_single_result(result, idx, total)
    for task in benchmark_tests:      # compute leg then in-process evaluate_run
        result = _run_benchmark_test(task, plan, config)
        _print_single_result(result, idx, total)
    return results
```

**Execution modes:**
- Main tests: `unittest.TextTestRunner` writing into a buffer, and the task owns
  fds 1/2 for its duration (`_OwnedStream`), so a child process' stray output
  cannot land between the counter line and its details: dropped on PASS, kept
  with the traceback on FAIL
- Skill tests: `subprocess.run()` with pre-generated scripts
  - `.sh` → `bash script.sh`
  - `.sbatch` → `sbatch script.sbatch`
- Timeout: `_timeout` from pmd_env.yaml (1200 s as shipped), and `deploy.py`'s polling idle budget is its own base plus that value so a task using its full budget is not mistaken for a hung remote

**Result collection:**
- `results.json`: Full test results with status, details, tracebacks
- `status.json`: Overall status summary

**Unconverged optimization tolerance (#109, #117):**

Qualification runs often truncate the optimization step budget to keep
tests cheap. Two tolerance paths handle the resulting non-zero exits:

- *Agent mode* (non-zero subprocess exit): the runner scans stderr/stdout
  and the test directory `<title>.log` tail for unconverged signatures
  (`Maximum optimization steps reached`, `not converge`, `No convergence`,
  `_RC_ITERATION_LIMIT_`). A match reports `status='pass'` with a
  `[NOT CONVERGED]` detail annotation; no match keeps `fail` (real
  crashes are never masked).
- *Input/config mode* (zero exit, framework records unconverged
  internally): the framework optimize workflow exits 0 even when
  `opt_converge=False`, and since #399 that route carries the **same**
  `[NOT CONVERGED]` annotation and `not_converged` flag, read from the run's
  own closing line by the runner's single `closing_status()` reader. The verdict
  is still `pass` for both routes - one budget rule, no third outcome - and
  `tools/verify.py` section C2 counts those legs as `budget_stopped`.

The skill backend `opt_converge` flag (set `False` when the skill
internal optimizer exhausts steps) is consumed by the framework
(`workflow/single.py::_finalize_optimize`) to call `mark_failed` but
still exit 0. The runner sees exit 0 and reports `pass`.

### 7.3 Converter (`qualification/certification/converter.py`)

The converter transforms the flat `.json` (via `load_config` +
`_flatten_legacy_sections`) into one shared sectioned dict, and renders
both outputs from it:

```
flat .json → sectioned dict → &CONTROL + &{STEM} + &MOLECULE [+ &OPTIMIZE | &MD] → test.input
                             → sectioned dict JSON                              → test.config
```

**Generated sections:**

| Section | Content |
|---|---|
| `&CONTROL` | title, jobtype, theory method, driver |
| `&{STEM}` | Generic skill section dynamically derived by `_build_skill_section` from the skill name (stem = skill name without the `_skill` suffix), covering all 10 skills (`&MOLCAS`/`&BAGEL`/`&PYSCF`/`&CP2K`/`&LAMMPS`/`&XTB`/`&ORCA`/`&OPENQP`); skill-specific path, project settings, and skill-owned flat int lists (e.g. `target_states`) as space-separated tokens |
| `&MOLECULE` | ci, spin (derived from state_spin), current_state, and `cell` (periodic box) |
| `&OPTIMIZE` | Optimization parameters (`_build_optimize_pairs`), written when the resolved jobtype is `optimize` |
| `&MD` | root, activestate (advanced jobtypes other than optimize) |

The converter also writes a same-basename `.xyz` file for atomic coordinates
(`_write_xyz_file`, auto-loaded by the classic pipeline via
`_load_coordinate_file`), and a same-basename `.velo` file when the config
carries inline velocities (`_write_velocity_file`).

**Periodic cell transfer:** `_build_molecule_pairs` flattens a non-zero
3×3 `lattice_vector` (top-level or nested `system` section) into the
classic `cell` keyword (9 floats, row-major) inside `&MOLECULE`; the
framework normalize step rebuilds `lattice_vector` from it. An all-zero
or missing box writes nothing, leaving molecular systems unaffected.

**Jobtype resolution:** The converter maps profile/capability names to
classic jobtype strings (`_resolve_jobtype`):

| Profile/Capability | Jobtype |
|---|---|
| energy | `energy` |
| gradient | `energy gradient` |
| frequency | `energy frequency` |
| geometry_optimization | `optimize` |
| surface_hopping | `surface_hopping` |
| transition_dipole | `energy transition` |
| dipole | `energy dipole` |
| nonadiabatic_dynamics | `md` |

---

## 8. Diagnostic Skip Tasks

When a skill's test configs cannot be generated, the certification
planner creates a diagnostic skip task (`_make_diagnostic_task`)
instead of silently ignoring the issue:

```python
{
    'name': 'molcas_skill_diagnostic',
    'skill': 'molcas_skill',
    'mode': 'local',
    'skip': True,
    'skip_reason': 'no test cases generated: no test YAML files found in ...',
}
```

**Diagnosis checks (certification/planner.py):**

1. Skill registered? → If not: "skill '<name>' not found in skill registry. Available: ..."
2. Skill `execution_mode: slurm` with sbatch missing? → "skill execution_mode is 'slurm' but sbatch is not available on this environment"
3. Runtime tests present but pytest missing? → "<n> runtime test scripts found but pytest is not installed"
4. `generate_test_configs()` raised? → "test config generation failed: ..."
5. `_diagnose_missing_tests` when no configs came back:
   - Skill package importable? → If not: "skill package pyrai3md.skills.<name> is not installed; rerun pip install to include skill resources"
   - `tests/` directory exists? → If not: "tests/ directory not found in ..."
   - YAML files present? → If not: "no test YAML files found in ..."
   - Otherwise: "test YAML parsing produced no configs"

This ensures users see clear messages in the test report rather than
silent failures.

---

## 9. Configuration for Testing

### 9.1 pmd_env.yaml

Skills are configured for testing in `qualification/pmd_env.yaml`:

```yaml
skills:
  pyscf_skill:
    test: condition         # true | false | condition (see the switch levels)
    runtime_tests: true     # the tests/test_*.py phase
    tests:                  # one key per case YAML, by file stem
      scf_energy: true
      scf_gradient: false   # closes that case alone; the rest still run
    pyscf_path: import      # Use Python import (no external binary)

  molcas_skill:
    test: false             # Disabled (e.g., molcas not installed)
    runtime_tests: true
    tests:
      casscf_energy: true
      ...
    molcas_path: /path/to/molcas
    execution_mode: ''      # 'slurm' = skill self-submits via sbatch -W
    slurm_partition: ''     # skill slurm resource keys (see skill schema)
    slurm_time: ''

environment:
  scheduler: ''             # '' = one local round; 'slurm' = one job per test case group

slurm:
  partition: default
  time: '01:00:00'
  ntasks: 1
```

Switch levels, highest first: `_runall`, then `skills.<name>.test`, then
`runtime_tests` / `tests.<case>`. The first two take three values, and only
`condition` lets the level below speak - `test: true` certifies the whole skill
and no per-case key is read, `test: false` closes it. An unlisted case defaults
to `true`, so `tests:` closes cases rather than whitelisting them.
`test`, `tests` and `runtime_tests` are planner-only keys and are never injected
into the generated skill config.

`pmd_env.yaml` currently carries 10 skill sections (openqp, bagel, cp2k, lammps,
molcas, orca, pyscf, xtb, qdm, nn_pes) with 132 case switches between them.
Three sections are open for a certification round - lammps (`test: true`), cp2k
and molcas (`test: condition` with a per-case map); the other seven keep
`test: false`. On a remote host the certification is driven by `_runall` (or by
flipping one skill to `true`/`condition`), not by an always-enabled skill.

Closing the certification that way does not close the pytest phase: since #359
each skill's `tests/test_*.py` files are planned and run on the strength of
`runtime_tests` alone, so an optimize round executes 62 of them. The shared
layer's 2 files are planned once for the round and carry no key of their own,
for the same reason - they need no engine: 64 code checks run in a round whose
certification is fully closed.

`environment.scheduler` and `skills.<name>.execution_mode` answer two
different questions and are deliberately independent:

- `scheduler` is the **main framework's round mode**: `''` runs the
  plan in one local process, `slurm` groups the test cases into
  independent jobs (one per example, one per skill case group, one per
  benchmark case, one for the unit suite and one for the integration
  suite) and submits them. Each skill case group gets its own working
  directories and scratch, so groups cannot overwrite each other;
  cases naming the same `modeldir` are a chain and share a job.
- `execution_mode` is the **skill's own policy** for executing a
  single calculation (`local` shell vs the skill's own `sbatch -W`
  self-submission). It applies to all three skill test modes
  (`pyrai3md_skill`, `pyrai3md_agent`, `pyrai3md_input`).

The target scenario "pyrai3md main process local, skill remote" is
`scheduler: ''` + `execution_mode: slurm`. `scheduler: slurm` +
`execution_mode: slurm` means the framework submits the case as a job
and the skill submits a nested job from that compute node; sites that
forbid compute-node submission must keep `execution_mode: local` for
such a round (the planner prints a reminder, it does not change the
switch). Execution policy keys
(`execution_mode`, `slurm_*`) use override semantics during injection:
pmd_env.yaml values win over the `execution_mode: local` hardcoded in
the skill test YAMLs. sbatch availability is checked at plan time
(diagnostic skip when missing), skill_slurm tasks run without the
300 s timeout, and passing results are asserted against the skill
slurm artifacts (`{project}.sbatch`, `{project}_slurm.out`) under the
task's own leg directory so a silent `run_local` fallback cannot pass
as a slurm test. In the
`pyrai3md_input` mode the converter writes these keys into the
classic input skill section (`&MOLCAS`/...), never `&ENVIRONMENT`
(which serves the framework dispatch layer only).

### 9.2 Environment Validation

Environment validation (e.g., checking `molcas_path`, `cp2k_path`) is the
responsibility of each skill's backend `__init__`, **not** the certification planner.
This ensures runtime errors are captured as skill test failures rather
than being silently skipped.

### 9.3 Framework-Level Skill Integration Tests

Besides the per-skill qualification modes, the framework keeps a small
set of skill integration tests under `pyrai3md/skills/tests/`:

- `test_wrapper.py` — the `pyrai3md_skill` wrapper's `resolve_skill_name`
  config resolution (top-level `skill_name` wins, `environment` section
  fallback, missing values return '');
- `test_skill_agents.py` — parametrized over all 10 skill agent modules
  (bagel/cp2k/lammps/molcas/openqp/orca/pyscf/xtb), verifying that
  `_run_calculation` forwards the dynamic calculation inputs
  (`target_states`, `point_charges`) to `backend.calculate()` and that
  empty config values become `None`.

These run as part of the full pytest suite, not of the qualification
plan.

---

## 10. Dependency Rules

The dependency direction must remain:

```
Skill (self-contained)
├── tests/
│   ├── *.yaml          # Skill-owned test cases
│   └── test_*.py       # Skill-owned runtime tests
├── query.py            # Sole exit point for skill information
├── backend.py
└── agent.py

PyRAI3MD (framework)
├── skills/query.py     # Sole entry point for skill discovery
├── skills/tests/       # Framework-level skill integration tests
├── qualification/
│   ├── manager.py      # Aggregates the four plan categories + shared script writers
│   ├── runner.py       # Executes via subprocess
│   └── certification/
│       ├── planner.py  # Skill test planning (runtime tests first, three modes)
│       ├── handler.py  # Test file discovery + generic skill section builder
│       └── converter.py # Test YAML → flat .json → sectioned dict → {.input, .config}
└── schema/             # Framework config pipeline
```

**Forbidden:**

```
skill → pyrai3md qualification (or any pyrai3md module)
```

A skill must be independently distributable. It must not import any pyrai3md module.

**Test placement (2026-11-12).** A pure skill-internal test lives in
`<skill>/tests/` and imports only that skill. A test that exercises the
framework route or a cross-skill contract lives in
`qualification/verification/` (unittest) or `pyrai3md/skills/tests/`
(framework-level pytest). `test_orca_correction_bound` and
`test_resident_error_boundaries` stay in their skills;
`test_flatten`, `test_lammps_persistent_test_plan` and
`test_nn_pes_seed_repro` stay in verification because they need the
framework/qualification route.

The **query mechanism** is the sole communication channel:
- Skill side: `<skill>/query.py` returns skill-local information (no framework prefix)
- Framework side: `skills/query.py` discovers skills and adds framework paths at runtime
- See [SKILL_APP_DEVELOPMENT.md](SKILL_APP_DEVELOPMENT.md) Section 4.4 for the full query protocol

---

## 11. Adding Tests for a New Skill

### Step 1: Create test YAML files

```yaml
# <skill>/tests/my_test.yaml
version: 1.0
name: my_test

input:
  atoms: [H, H]
  coordinates:
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.74]
  charges: 0
  multiplicity: 1
  profile: <profile_name>
  method: <method_name>

expected:
  energy:
    value: null
    tolerance: 1e-6
```

### Step 2: Nothing else to implement — parsing is unified

No per-skill generator is needed. The certification handler discovers
every `*.yaml` under `<skill>/tests/`, and the certification converter
(`parse_test_yaml`/`generate_test_configs`) turns them into flat `.json`
configs using generic rules keyed by the skill identity (name →
title/project, nested `system` section for `current_state`, declared
pair maps rendered into `coupling` (#447); skill metadata from the skill's own
`query.py`).

Optional: ship skill-owned runtime tests as `test_*.py` files in the
same `tests/` directory; the certification planner runs them through
pytest ahead of the YAML task tests. Remember to register the test
resources in `setup.py` `package_data`.

### Step 3: Enable testing in pmd_env.yaml

```yaml
skills:
  <skill_name>:
    test: true
```

### Step 4: Run tests

```bash
pyrai3md test
```

---

## 12. Summary

```
                test.yaml (skill-owned)
                     │
           certification converter
                     │
               flat .json (common source)
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
  pyrai3md_skill  converter.py    converter.py
  (flat .json)    → .config       → .input
  (skill_agent)   (pyrai3md_agent)    │
        │            │            pyrai3md CLI
        │            │           (pyrai3md_input)
        ▼            ▼            ▼
    skill backend  framework     classic parser
                   pipeline      + framework
```

This design keeps:
- **Skill independence**: scientific knowledge stays in the skill
- **Unified agent protocol**: the flat .json is the common source format
- **Minimal test complexity**: three modes share the same generated configs
- **Compatibility**: legacy input workflow remains supported via converter

The skill owns scientific knowledge.
PyRAI3MD owns orchestration.
Qualification owns integration testing.
