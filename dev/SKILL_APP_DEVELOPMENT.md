# Skill Application Development Guide

Last updated: 2026-11-12 19:20 (4.1 now records the implemented PEP 562 export; 11.2.1 records the requested-block raise and the per-skill read_support. Earlier: 2026-11-06 13:25 (5.8 was re-read against the two skills that own a slurm job body and now names where its rule is enforced - `pyscf_skill` and `qdm_skill` force `environment.execution_mode = 'local'` into the config they serialize, because a template argument cannot do it and the argument that used to be passed was read by nothing. 5.7 and the rest were not re-audited. Earlier: 2026-10-27 19:02 (5.7 records the one exception to the two-spelling rule: openqp_skill merged `nproc` into `openqp_nproc` (#427) and declares the prefixed keyword in its manifest, so the drift guard only demands the prefixed companion when a skill declares the bare name. Only 5.7 was re-checked this round; the rest keeps the previous audit. Before it, 2026-10-22 22:20 (framework mode is the per-task config dict, with the PyRAIConfig path marked legacy and `build_pyrai_config` recorded as deprecated; the task sketch shows `tmp_PYSCF` and the dict config; skill counts move 8 to 10 where the sentence is about schemas and stay 8 for the subprocess-agent guard; the cell snapshot keys gain `image`; `fin_velocity` is now consumed. The rest was not re-audited. Before it, 2026-10-07 13:05)

This document defines the abstract specification for developing official skills for PyRAI3MD.
All skills must conform to this specification to ensure compatibility with the PyRAI3MD schema,
qualification test system, and orchestration framework.

---

## 1. Overview

A **skill** is an autonomous computational component that provides independent implementation
of a quantum chemistry or machine learning backend. Skills are not wrappers around external
commands — they are self-contained modules with their own configuration schema, method definitions,
workflow construction, execution service, result parser, and documentation.

**Design principle:**

> PyRAI3MD orchestrates scientific tasks, while skills provide independent computational implementations.

PyRAI3MD discovers and schedules skills through capability information. Each skill independently
manages its backend-specific computational logic.

---

## 2. Design Philosophy

### 2.1 Skill Independence

A skill must be independently usable **without importing any PyRAI3MD modules**.
All skill code — including `backend.py`, `schema/`, `service/`, `query.py`, and `tests/` —
must contain zero references to `pyrai3md` or any frontend framework package path.

The skill only provides information about itself. The framework (PyRAI3MD or any other frontend)
resolves full import paths at runtime by placing the skill in its search path.

```
<skill_name>/          # Self-contained, no framework dependency
├── manifest.yaml      # Only skill-local paths
├── query.py           # Self-contained registry queries
├── backend.py         # Public computational API
├── schema/
└── service/
```

**Coupling rule:** The only communication channel between a skill and a frontend framework
is the **query mechanism** (see Section 4.4). A skill never imports framework modules;
the framework imports the skill.

### 2.2 Configuration Philosophy

PyRAI3MD Config and Skill Config are **independent but structurally similar**.

They must not form a dependency chain:

```
Wrong:
  SkillConfig → depends on → PyRAI3MD Config   (prevents standalone usage)

Correct:
  PyRAI3MD Config → task selection → Skill Config → workflow compilation → Backend execution
```

The two configurations communicate through a common capability protocol rather than direct inheritance.

---

## 3. Directory Structure

```
<skill_name>/
├── __init__.py
├── agent.py                # CLI entry (python -m <skill>.agent config.json)
├── backend.py              # Public computational API entry
├── manifest.yaml           # Lightweight discovery metadata
├── query.py                # Self-contained registry queries (optional)
├── skill_guide.md          # Skill-specific usage guide
│
├── schema/
│   ├── __init__.py
│   ├── config.py           # Skill configuration (Pydantic models)
│   ├── method.py           # Supported methods and capabilities
│   ├── reference.py        # Constants, defaults, periodic table
│   ├── adapter.py          # External task → Skill config conversion
│   ├── normalize.py        # Derive missing quantities
│   └── validate.py         # Consistency and compatibility checks
│
├── service/
│   ├── __init__.py
│   ├── workflow/           # Profile → workflow graph compilation
│   ├── builder/            # Workflow graph → backend input files
│   ├── runner/             # Local and SLURM execution
│   ├── parser/             # Backend output → standardized results
│   └── templates/          # Script and parameter templates
│
└── tests/
    ├── *.yaml              # Test case definitions
    └── test_*.py           # Skill-owned offline unit tests (parser/builder)
```

---

## 4. Runtime Loading Architecture

### 4.1 Lazy Import Principle

PyRAI3MD startup must never import skill computational modules.

**Old design**: Backend modules imported and instantiated during Planning

```
PyRAI3MD startup
    → Maestro skill discovery
    → Read manifest.yaml
    → Import backend.py          ← eagerly imports backend module
    → Initialize <Skill>Backend  ← creates backend instance
    → Compile workflow
    → Execute calculation
```

**New design**: Backend API stored as string path, imported only at execution time

```
PyRAI3MD startup
    → Maestro skill discovery
    → Read manifest.yaml
    → Store API path string    ← 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend'
    → Build PyRAIState (pure data, no backend instances)
    → Calculator.initialize()
        → resolve_api()        ← dynamic import via importlib
        → Instantiate backend
    → Execute calculation
```

Only registry information is loaded during discovery. Backend modules are imported only when `Calculator.initialize()` calls `resolve_api()`.

**Implementation (2026-11-12).** All ten shipped skills use PEP 562
`__getattr__` in their package `__init__.py`; importing `<skill>.query` does
not load `<skill>.backend` or its service tree (verified per skill in a fresh
interpreter). `skills/query.py::discover_all()` warns when a skill cannot be
imported, rather than deleting it from the registry silently.

### 4.2 manifest.yaml

The manifest provides lightweight discovery information only.

**Must include:**
- Skill name (plain program label, e.g. `pyscf`, `cp2k` — no `label` key, no `qm.` prefix)
- API entry (backend module and class)
- Result units (`units`: energy/gradient units of the returned results)
- Supported capabilities (energy, gradient, nac, etc.)
- Supported calculation profiles
- Method roles (qm, qm2, mm)
- Compute resources (`resources`): the keywords this skill reads for its
  thread and rank counts and the value to assume when an input omits one
  (section 5.7). `query.get_metadata()` surfaces the block unchanged, which
  is how the framework sizes a run without naming any engine keyword
  itself (#306)

**Optional:**
- Control parameters (user-configurable keywords) — currently only
  molcas/bagel/openqp declare them in the manifest
- `supports_pbc` (top-level bool, default `false`): whether the backend
  supports periodic boundary conditions. It is a **standalone key**, never an
  entry of `capabilities` (that list feeds the certification planner and the
  resolver's profile spelling; mixing a non-property flag in would corrupt
  them). `query.get_metadata()` surfaces it, so the framework can refuse a
  periodic run at config time (`schema/validate.py`) and each skill's own
  `validate()` refuses `pbc` when it is `false`. cp2k / lammps / xtb declare
  `true`; the molecular backends declare `false` (CHANGELOG #216). The framework
  gate reads only the whole-system layer (#315), and a center is handed no
  periodicity in its `skill_config`, so a molecular backend's own refusal stays as
  the standalone-entry backstop rather than blocking the center path.

**Must not contain:**
- Workflow implementation
- Framework package paths (e.g. `pyrai3md.skills.*`)

All paths in manifest are **skill-relative**. The framework prepends its own package prefix at runtime.

Example (aligned with the shipped manifests, e.g. molcas):

```yaml
name: molcas                 # Program label, no label key, no qm. prefix

backend:
  module: backend            # Skill-relative: resolved to <framework>.skills.<skill>.backend
  class: MolcasBackend

units:
  energy: hartree
  gradient: hartree/bohr

roles:
  qm: MolcasBackend
  qm2: false
  mm: false

capabilities:
  - energy
  - gradient
  - nonadiabatic_coupling
  - geometry_optimization

# Control parameters (optional): user-configurable keywords for this skill.
# The framework extracts these via get_metadata() and caches in the Maestro.
# They define which keywords users can set in input files.
control_parameters:
  method:
    type: str
    default: casscf
  basis:
    type: str
    default: ano-rcc-vdzp
  nact_elec:
    type: int
    default: 2
  nact_orb:
    type: int
    default: 2

# Compute resources (required): the keywords this skill reads, with the
# value assumed when an input omits one. `ranks` is written only by skills
# that start an MPI job (#306).
resources:
  threads:
    keyword: thread
    default: 1
  ranks:
    keyword: nproc
    default: 1

profiles:
  <method>_energy:
    capability: energy
    method: <method>
  <method>_gradient:
    capability: gradient
    method: <method>
```

The `units` block declares the units of the result arrays (energy,
gradient) the skill's parser returns, so consumers can interpret the
values without inspecting the skill internals.

Each control parameter entry specifies:
- `type`: data type (`str`, `int`, `float`, `bool`)
- `default`: default value (used when user does not specify)

The framework uses control parameters to:
1. Filter skill-specific fields from user input during config merging
2. Provide normalize reference for user input validation
3. Generate input templates in the qualification framework

### 4.3 Self-Contained Query Module

Each skill **must** provide its own `query.py` that reads the local `manifest.yaml`.
The skill's `query.py` is the **sole exit point** for skill information — it never
imports any framework module.

Key functions a skill's `query.py` must provide:

```python
def get_metadata():
    """Return skill metadata (skill_name, label, skill_path, backend_class,
    control_parameters).
    All paths are skill-relative (no framework prefix).
    control_parameters is extracted from manifest.yaml."""

def get_registry():
    """Return skill-local registry with relative paths."""

def get_capabilities():
    """Return supported capabilities."""

def get_profiles():
    """Return available calculation profiles."""

def get_defaults():
    """Return schema default values for skill-specific fields."""
```

### 4.4 Query Mechanism — Sole Communication Channel

The query mechanism is the **only** communication path between a skill and a frontend framework.

```
Framework side (pyrai3md)           Skill side (<skill_name>)
─────────────────────────           ─────────────────────────
skills/query.py                     <skill>/query.py
  │                                   │
  ├─ discover_all()  ──────────────> get_metadata()
  ├─ get_registry()  ──────────────> get_registry()
  ├─ resolve_label() ──────────────> (via discover_all)
  │                                   │
  └─ adds framework prefix            └─ returns skill-local paths
     at runtime                          (no framework knowledge)
```

**Rules:**
1. `skills/query.py` is the framework's **sole entry point** for skill discovery
2. `<skill>/query.py` is the skill's **sole exit point** for skill information
3. Skills never import framework modules; the framework imports skills
4. All paths in skill query returns are skill-relative
5. The framework prepends its package prefix (`pyrai3md.skills.<skill>`) at runtime

### 4.5 String-Based API Reference

Planning stores backend APIs as **serializable string import paths**, not Python class objects:

```python
# Task descriptor in PyRAIState
task = {
    'skill': 'pyscf_skill',
    'api': 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend',  # string path
    'config': {'skill_config': <skill flat dict>, 'results': {}},
    'static_id': 'tmp_PYSCF',
    'capability': ['energy', 'gradient'],
}
```

**Benefits:**

1. **Pure Data State**: PyRAIState contains no Python class objects or backend instances
2. **True Lazy Import**: Backend modules are imported only when `Calculator.initialize()` calls `resolve_api()`
3. **Serialization Friendly**: `pickle(state)` or future JSON serialization becomes straightforward
4. **Plugin Friendly**: Future third-party skills only need to expose an import path; no framework-specific registration required

**API Resolution:**

The framework provides a common resolver (see [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md) Section 3.6):

```python
def resolve_api(api_path):
    """Resolve backend class from import path string."""
    if isinstance(api_path, type):  # backward compatibility for tests
        return api_path
    module_name, cls_name = api_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)
```

This mechanism is identical to plugin loading used in NequIP, Hydra, PyTorch Lightning, and MMEngine.

**Future Extension:**

The same mechanism can later be generalized to other plugin components:

```yaml
backend:
  api: pyrai3md.skills.pyscf_skill.backend.PySCFBackend

schema:
  api: pyrai3md.skills.pyscf_skill.schema.Config

compiler:
  api: pyrai3md.skills.pyscf_skill.compiler.Compiler

generator:
  api: pyrai3md.skills.pyscf_skill.testing.Generator
```

All executable components become lazily resolved through the same API resolver, providing a unified plugin architecture across the entire PyRAI3MD framework.

---

## 5. Backend Architecture

### 5.1 Backend Responsibility

`backend.py` is the only public computational API entry.

Responsibilities:
- Initialize skill configuration (dual-mode: PyRAIState or dict)
- Compile calculation workflow
- Manage execution
- Return standardized results

Public API (from the skill's own package):

```python
from <skill_name>.backend import <Skill>Backend

# Standalone mode: dict or JSON path
backend = <Skill>Backend(config_dict)
result = backend.calculate(atoms, coords)  # Lightweight calculation step
result = backend.local_run()               # Full local calculation
result = backend.slurm_run()               # Full SLURM calculation

# Framework mode: receives the plain task-config dict
backend = <Skill>Backend({'skill_config': <flat dict>, 'results': {}})
result = backend.calculate(atoms, coords,
                           point_charges=...,
                           target_states=...)
```

`calculate()` accepts the dynamic calculation inputs beyond geometry:
`point_charges` embeds background charges, `target_states` selects the
evaluation states of multi-state calculations (e.g. gradients). All 10
skills implement this signature (`nn_pes` and `qdm`, the two newest, run
in-process), and every skill `agent.py` subprocess
entry forwards the config values through a module-level
`_run_calculation(backend)` helper that converts empty lists to `None`
so no empty parameter leaks into the backend:

```python
return backend.calculate(
    atoms, coordinates,
    point_charges=config.point_charges or None,
    target_states=config.target_states or None)
```

`skills/tests/test_skill_agents.py` guards this forwarding contract for
the 8 subprocess agents in its module list (`bagel`, `cp2k`, `lammps`,
`molcas`, `openqp`, `orca`, `pyscf`, `xtb`); `nn_pes` and `qdm` also ship an
`agent.py` but run in-process and are not covered by that guard.

When used within PyRAI3MD, the framework imports the skill via the full package path:

```python
# Framework-side (pyrai3md only):
from pyrai3md.skills.<skill_name>.backend import <Skill>Backend
# All backends are instantiated with PyRAIConfig:
instance = <Skill>Backend(pyrai_config)
```

### 5.2 Dual-Mode Initialization

The backend `__init__` must support two initialization modes:

**Framework mode** (receives the per-task config dict):
```
Framework: build_state → task queue → task['config']
    ↓
Backend.__init__({'skill_config': <flat dict>, 'results': {}})
    → dict carrying 'skill_config' → task-config mode
    → skill adapter → normalize → validate
    → compile_workflow()
    → raw results written back to task['config']['results']
```
The PyRAIConfig-carrying path still resolves (a config model with `_skill_config`
attached) but is the **legacy state mode**: `_build_task_queue` creates the dict-based
task config directly, so new code does not ask for the PyRAIConfig mode
(`skills/xtb_skill/backend.py:85-105`).

**Standalone mode** (receives dict or JSON path):
```
Backend.__init__(config_dict_or_path)
    → _is_pyraiconfig(config) → False
    → read_json() if str
    → skill adapter → normalize → validate
    → compile_workflow()
```

**Input file is read-only** (#143): the backend never writes the validated
config back to the input JSON. The resolved config stays in memory; slurm
node hand-off is covered by the runner serializing the config into the
calculation directory itself. This restores idempotent re-runs of
`pyrai3md_skill <file>.json` scripts (the previous write-back overwrote the
flat generated config with the nested SkillConfig dump, breaking re-runs).

Readers keep defense-in-depth compatibility with the legacy nested
layout the old write-back produced:

- `wrapper.resolve_skill_name` resolves `skill_name` top-level first and
  falls back to `environment.skill_name`.
- The converter `load_config` flattens the five legacy sections
  (system/properties/dynamics/calculation/environment) without
  overwriting existing top-level keys, so files already corrupted by the
  old write-back remain consumable.

The detection uses `type(obj).__name__ == 'PyRAIConfig'` to avoid importing
framework modules, preserving skill independence.

### 5.3 Skill Config Access from PyRAIConfig

When receiving a PyRAIConfig, the backend accesses framework and skill-specific fields directly:

```python
# PyRAIConfig provides unified access (nested and flat):
pyrai_config.control.jobtype   # framework fields (nested submodel)
pyrai_config.jobtype           # framework fields (flat access)
pyrai_config.nact_elec         # skill-specific fields (merged skill submodels)
```

The framework does **not** build one PyRAIConfig per task any more:
`data/core.py::build_pyrai_config` is deprecated ("No longer used by the framework
since _build_task_queue now creates a dict-based task_config directly") and stays only
for external callers. Each task carries its own plain dict
`{'skill_config': <flat skill dict>, 'results': {}}`, so the backend reaches
skill-specific fields (nstate, basis, ...) through that dict without importing
framework modules. After calculation, the backend writes its raw results back to
`config['results']`.

### 5.4 Initialization Stage

All expensive operations happen once during `__init__`:

```
<Skill>Backend(pyrai_config_or_config)
    → read PyRAIConfig (framework mode) or dict (standalone mode)
    → build_config()      # skill adapter
    → normalize()         # skill normalize
    → validate()          # skill validate
    → compile_workflow()  # Profile → WorkflowGraph
    → prepare runtime environment
```

After initialization: method, workflow, input generation logic, and runtime environment are all fixed.

### 5.5 Calculation Stage

`calculate()` must be lightweight. It should only update dynamic information:
- Atomic coordinates
- Point charges
- Electronic state
- Trajectory step
- Periodic cell (`cell=` keyword: the frozen per-evaluation snapshot
  `{lattice_vector, cell, pbc, volume, image}` or `None`; a periodic backend applies
  it to `config.system` before writing its input so a variable-cell step uses
  its own deformed cell — see Section 3.3.2 of CALCULATOR_DEVELOPMENT.md)
- Working directory

No repeated schema construction, semantic analysis, or workflow generation.

**`keep_tmp` semantics.** `keep_tmp` answers one question: does the skill's
temporary directory survive the run? The keyword itself is the same
everywhere; the point at which it takes effect differs with what the
directory holds, and both lifetimes below are the contract (#307):

- **per call.** `orca_skill`, `bagel_skill`, `molcas_skill`, `cp2k_skill`,
  `openqp_skill`, `pyscf_skill` and `qdm_skill` clear the calculation
  directory at the end of `calculate()`, once the results are collected.
  Safe because every writer into that directory makes it again
  (`np.savetxt`/`open(...,'w')` behind an `os.makedirs(..., exist_ok=True)`)
  and nothing reads a previous step's file back; a stepped run therefore
  rebuilds the directory each step and leaves the last step's files behind
  when `keep_tmp` is set.
- **at the end of the backend's life.** `nn_pes_skill` and the resident
  `lammps_skill` service cannot use the per-call point: their directory
  holds files the run *accumulates* (`{project}_{profile}.log`, the
  uncertainty dump, the service package), written in append mode by code
  that does not recreate the directory, so clearing mid-run keeps only the
  last step and then stops recording at all -- and the `OSError` of the
  missing parent is swallowed. Both bind the cleanup to the backend object
  through `weakref.finalize` (`nn_pes_skill/backend.py:_discard_calcdir`,
  `lammps_skill/backend.py:_finalize_service`), which also runs at
  interpreter exit.

Anything that must survive a call regardless of the keyword belongs
outside the calculation directory: the LAMMPS force-field model is written
beside `tmp_LAMMPS` so a cleared directory is rebuilt from that JSON alone,
and nn_pes' trained ensemble lives in `modeldir`, which no keyword governs.
The consequence of `keep_tmp: 0` is then limited to what is genuinely
scratch, and a returned path into the directory (nn's `train_summary`) is
only valid while the keyword keeps it.

`calculate()` receives the dynamic inputs through the agent forwarding
contract (see Section 5.1). The signature is uniform across backends:
`calculate(atoms, coordinates, point_charges=None, target_states=None,
cell=None)`. Molecular (non-periodic) backends accept `cell` and ignore it
(their `validate()` rejects `pbc`, so `cell` is always `None` there). Skills
performing internal iteration (optimize/md) report the final structure, velocities
and -- when the engine moved it -- the simulation cell through the `fin_coord`,
`fin_velocity` and `fin_cell` return channels consumed by the framework (see
Section 11.2).

### 5.6 Thread Safety Belongs to the Skill

Two backend terms can run in threads of one interpreter (the term axis
of `workflow/dispatch/terms.py`, asked for by `executor thread` on the
`&composite` section). The framework does not decide whether a skill can
survive that; it asks. Before the pool starts it hands every term's
backend the contract it is about to run under:

```python
    def check_thread_safe(self, runner, execution_mode, profile_semantics):
        """Raise to decline a shared thread pool; return to allow it."""
```

A backend that does not define the method runs concurrently. A backend
that raises stops the evaluation with its own sentence, and the executor
never degrades to serial behind its back (#301 replaced the
framework-side registry that no skill could see or fill).

What makes a refusal necessary is anything the calculation moves for the
whole process, because two terms share it:

- `os.chdir()`. The runner anchors its child with `cwd=` and passes an
  absolute script path instead; `qualification/verification/`
  `test_thread_anchoring.py` scans every skill runner for exactly that.
- `os.environ[...]` writes and `sys.stdout` replacement (a process-wide
  setting one term sets for the other). The same scan checks both across
  every skill module.
- artifacts named after `config.project` under a shared scratch root,
  when a second term with the same settings finds and reuses them.
- interpreter-wide library configuration (a device or thread setting of
  TensorFlow/BLAS) touched from a term.

#302 retired three of those from the skills that had them: the external
runners lost their `chdir`, `pyscf_skill` lost its process-wide tee (the
log handle is bound to the thread and reaches PySCF through `mol.stdout`)
and its shared artifact root (a local run now keys files on its own
`calcdir`), and both in-process skills set `OMP_NUM_THREADS` only when the
process has not chosen a value yet, since OpenMP latches it at the first
parallel region anyway.

Current refusals, each naming the line that moves process state in the
skill's own message: `nn_pes_skill`, and only for its `*_train` profiles
(the training arm sets TensorFlow's visible devices for the whole
process). `pyscf_skill`, `qdm_skill` and `nn_pes_skill` serving now run
their terms concurrently, each on the strength of a measurement in its own
`tests/test_concurrency.py` or `test_train_predict.py`. The Molcas scratch
directory carries one more level per term (Section 1.3 of
`dev/skills/MOLCAS_SKILL_DEVELOPMENT.md`).

---

### 5.7 Who Decides How Many Threads a Calculation Uses

Three keywords, three different things: `multitask`/`multipool` size the
process pool, `remotetask` decides who submits, and `thread`/`nproc` say how
many OpenMP threads one calculation spreads over and how many MPI ranks it
starts. The counts belong to the skill that runs the calculation, and each is
accepted in two spellings (#306):

| spelling | written by | read by |
|---|---|---|
| `thread`, `nproc` | a standalone skill input (`&pyscf thread 8`) | the skill's own schema, always |
| `<skill>_thread`, `<skill>_nproc` (`pyscf_thread`, `cp2k_nproc`) | the framework, from the skill section or from `skills.<name>` of pmd_env.yaml | the skill's own schema, as a second name of the same quantity |

The pair is folded inside the skill: the larger value wins and both default
to 1 (`_resolve_resources` in `skills/<name>_skill/schema/config.py`). A
maximum rather than a precedence, because the framework fills the plain name
with its default before the task queue is built, so an unwritten `thread` and
a written `thread: 1` arrive as the same dict; only the skill can still tell
its caller meant something by the prefixed spelling.

The rank count is the one quantity a skill may keep a single name for:
`openqp_skill` merged `nproc` into `openqp_nproc` (#427), so that skill has no
bare spelling and refuses `nproc` by name. Its manifest then declares the
prefixed name as the keyword it reads, and the two guards in
`qualification/verification/test_resource_keywords.py` follow that shape --
a declared keyword must exist in the skill's schema, and the prefixed companion
is only demanded when the declaration uses the bare name. Threads stayed a
pair.

The run-wide floor `&environment num_thread` of #303 is cancelled along with
the field: a thread count is a property of the calculation, and the framework
cannot know what an engine does with one. The name is now refused wherever it
used to be read -- `&environment` through `schema.config
.RETIRE_ENVIRONMENT_KEYS`, a skill section through `RETIRED_RESOURCES` in each
`schema/adapter.py` -- and each refusal names the replacement. cp2k's rank
keyword `mpi`, openqp's `mpi_nproc` and xtb's `xtb_nproc` (a process name for
what was a thread count, on a skill with no MPI arm) are retired the same way.

`&environment` keeps one duty of this kind: `scratch_path`, the directory the
run log and the calculation directories are made under. It travels in the
environment snapshot, which is also what lets the framework find the machine's
allocation for the check below.

A workflow section may not carry any of these keywords
(`schema/adapter.py::_reject_workflow_resource_keywords`): the
md/optimize/adaptive/hessian/search/composite sections parallelise with
processes, never with threads or ranks inside one engine, so `md: thread 5`
would read as a setting while being dropped -- the silent-drop class #299 and
#300 removed elsewhere.

Ranks and threads stay separate on purpose: `nproc` (or `<skill>_nproc`, and
`slurm_ntasks`, which SLURM may override through `$SLURM_NTASKS`) answers how
many ranks run, `thread` answers how many threads each rank gets. Neither may
stand in for the other.

A skill that starts an MPI job names the launcher `mpirun` (#305): lammps
builds `mpirun -np` in both its arms, cp2k and openqp build `mpirun -n`, and
bagel's `mpiexe` defaulted to the other spelling until now. A site that runs a
different launcher sets it explicitly (`mpiexe`, `mpi_command`); no default and
no assembled command uses anything else, and
`qualification/verification/test_mpi_launcher.py` keeps it that way.

`Scheduler.validate_resources()` asks each skill what it will consume instead
of guessing: a skill's manifest `resources` block names the keywords it reads
and the value to assume when an input omits them,

```yaml
resources:
  threads:
    keyword: thread
    default: 1
  ranks:
    keyword: nproc
    default: 1
```

and `query.get_metadata()` surfaces it, so a new skill needs no change in the
framework. The demand is `units x ranks x threads`, where `units` is the
concurrency the Scheduler actually starts -- the pool size capped by how many
units are queued, one on the sequential path -- multiplied by
`composite max_workers` when the term axis runs its backends in a thread pool.
The reference is what this run may occupy: the local cores, or
`slurm.ntasks x slurm.cpus_per_task` from the environment snapshot when the
run submits. Measured on a 10-core machine, an orca run of `thread: 3`,
`nproc: 2` and `multipool: 4` logs `Parallel request 4 unit(s) x 6 CPU(s) per
unit of orca_skill = 24 exceeds the 10 CPUs available to this run
(oversubscribed).` It only warns: oversubscription is allowed, the framework
resizes nothing on the user's behalf, and no environment variable is invented
here. The per-skill check that aborted when a thread count exceeded
`os.cpu_count()` is gone for the same reason the run-wide floor was: the
machine that parsed the input is not the machine that runs the job. Criteria
in `qualification/verification/test_resource_keywords.py`.

---

## 6. Schema Design

The `schema/` directory defines the skill's computational knowledge. All skills must conform
to the PyRAI3MD data type conventions described below.

### 6.1 Skill Schema Modules

Each skill maintains a complete schema/ directory:

```
<skill>/schema/
├── config.py      # Skill Config (Pydantic models: System, Properties, Calculation, Environment)
├── method.py      # AVAILABLE_METHOD, PROFILE_METHOD_MAP, PROFILE_STEP_MAP
├── reference.py   # AVAILABLE_PROFILES, AVAILABLE_BASIS, DEFAULT, PERIODIC_TABLE
├── adapter.py     # External task → Skill Config conversion (build_config)
├── normalize.py   # Skill-specific derived quantities
└── validate.py    # Skill-specific consistency checks (profile, method, basis, path)
```

The skill Config is **independent** from the framework Config.
Both share common fields (atoms, coordinates, charges, etc.)
but the skill Config adds skill-specific fields.

#### config.py

Defines the skill's complete configuration using Pydantic models.

Recommended sub-models: System, Properties, Dynamics, Calculation, Environment.
Each skill nests them in one `SkillConfig` and routes flat keywords to the sub-model
that owns them - the dynamic `create_model` field merge is gone (every skill's
`schema/config.py` now reads like `skills/bagel_skill/schema/config.py:342-352`):

```python
class SkillConfig(BaseModel):
    system: System = Field(default_factory=System)
    properties: Properties = Field(default_factory=Properties)
    dynamics: Dynamics = Field(default_factory=Dynamics)
    calculation: Calculation = Field(default_factory=Calculation)
    environment: Environment = Field(default_factory=Environment)
    # __init__ accepts flat kwargs and routes each to its owning sub-model,
    # mirroring the framework's PyRAIConfig pattern.
```

**Required Environment fields:**

The `Environment` model must include `skill_name` and `skill_path` to ensure
the config survives Pydantic validation and remains available to downstream components:

```python
class Environment(BaseModel):
    skill_name: str = '<skill_name>'
    skill_path: str = 'backend'   # Skill-relative, no framework prefix
    execution_mode: str = DEFAULT['execution_mode']
    # ... other fields
```

#### method.py

Defines supported electronic structure methods and their capabilities.
This is the single source of truth for method-task compatibility.

```python
AVAILABLE_METHOD = {
    '<method_name>': {
        'energy': True, 'gradient': True,
        'nac': True, 'frequency': True,
    },
}

# Profile → method mapping
PROFILE_METHOD_MAP = {
    '<method>_energy': '<method>',
    '<method>_gradient': '<method>',
}

# Profile → step chain mapping
PROFILE_STEP_MAP = {
    '<method>_energy': ['scf'],
    '<method>_gradient': ['scf', 'grad'],
}
```

#### reference.py

Provides:
- `PERIODIC_TABLE`: element symbols and atomic numbers
- `AVAILABLE_PROFILES`: all supported calculation profiles
- `AVAILABLE_BASIS`: supported basis sets
- `DEFAULT_CONVERGENCE`: convergence thresholds
- `DEFAULT`: all default calculation parameters

#### adapter.py

Converts external task descriptions into skill configuration.

Supports multiple input types:
- `dict`: direct configuration dictionary
- `BaseModel`: Pydantic model instance
- `str`: path to config file

The `build_config` function must filter input fields against valid Config fields,
ensuring unknown keys are silently ignored.

#### normalize.py

Completes missing information and derives quantities:
- Atom count, total electrons, alpha/beta electron distribution
- State count from `ci` or `state_spin`
- `state_spin` from `ci` and `spin`
- NAC/SOC pair classification from `coupling` and `state_spin`
- Active orbital indices from active space parameters

#### validate.py

Checks:
- Profile exists in AVAILABLE_PROFILES
- Method exists and is compatible with profile (via AVAILABLE_METHOD)
- Basis set is supported
- Backend executable path is valid
- Resource availability (CPU threads, scratch path)
- State indices are valid (target_states < nstate, etc.)

### 6.2 PyRAI3MD Data Type Conventions

Skills receive configuration from PyRAI3MD through the unified `Config` model.
The following type conventions must be respected:

#### 6.2.1 Index Conventions

| Convention | Description |
|---|---|
| **Atom indices** | Input is **1-based** (legacy text format). Skills receive **0-based** indices after adapter conversion. |
| **State indices** | Input is **1-based**. `coupling` pairs are converted to 0-based by the adapter. |
| **Internal use** | All indices within skill schema and service layers must be **0-based**. |

Fields requiring 1-based → 0-based conversion in the adapter:
`highlevel`, `midlevel`, `freeze`, `constrain`, `cbond`, `cangle`, `cdihedral`, `center`, `track_index`, `coupling`, `root` (the dynamics initial state; 1-based in every external spelling, typed included)

#### 6.2.2 State and Spin Conventions

| Field | Type | Semantics |
|---|---|---|
| `ci` | `List[int]` | Number of states per spin multiplicity. e.g. `[2, 2]` = 2 singlet + 2 triplet states |
| `spin` | `List[int]` | Total spin quantum number S per multiplicity. `0`=singlet, `1`=triplet |
| `nstate` | `int` | Total number of states = `sum(ci)` |
| `state_spin` | `List[int]` | **2S** per state (0-based per-state). e.g. `[0, 0, 2, 2]` for 2 singlets + 2 triplets |
| `coupling` | `List[List[int]]` | State pair indices (0-based after adapter). Each pair defines a coupling computation |
| `root` | `int` or `List[int]` | Dynamics initial state per center. **External spelling is 1-based** (the adapter decrements once); note the dual-basis caveat: `current_state` is 0-based externally, so a classic file carrying both spells the same physical state one apart. |
| `current_state` | `int` | Initial electronic state (0-based) for skill md-type tasks. Declared in the `System` submodel of all 10 skills. The certification converter (`parse_test_yaml`) writes the nested `{'system': {'current_state': N}}` section when generating configs; each skill's `build_config` flattens nested submodel sections before filtering, so the value routes automatically. In input/config mode the framework `System.current_state` reaches the skill through the sync/merge chain. |

Conversion rule: `state_spin[i] = spin[multiplicity_index] * 2`, expanded per `ci` counts.

#### 6.2.3 Core Data Types

| Field | Type | Description |
|---|---|---|
| `atoms` | `List[str]` | Element symbols |
| `coordinates` | `List[List[float]]` | (natom, 3) Cartesian coordinates in Angstrom |
| `charges` | `int` | Total molecular charge |
| `multiplicity` | `int` | Spin multiplicity (2S+1) |
| `point_charges` | `List[List[float]]` | (ncharge, 4) [q, x, y, z] |
| `pbc` | `List[int]` | (3,) Periodic boundary conditions, 0 or 1 |
| `cell` | `List[float]` | (6,) Cell parameters [a, b, c, α, β, γ] or (9,) lattice vectors |
| `lattice_vector` | `List[List[float]]` | (3, 3) Lattice vectors in Angstrom |
| `transition_pair` | `List[List[int]]` | (npair, 2) State pairs for coupling computation |
| `nac_pair_map` | `List[List[int]]` | (nnac, 2) Same-spin pairs (NAC) |
| `soc_pair_map` | `List[List[int]]` | (nsoc, 2) Different-spin pairs (SOC) |

#### 6.2.4 Adapter Type Codes

The framework adapter uses type codes for parsing classic text input:

| Code | Meaning | Example |
|---|---|---|
| `s` | Single string | `title` |
| `i` | Single integer | `factor` |
| `f` | Single float | `fbond` |
| `b` | Boolean | `embedding` |
| `sl` | String list | `qm` |
| `il` | Integer list | `ci`, `spin`, `pbc` |
| `fl` | Float list | `tbond`, `wall_center`, `cell` |
| `si` | Single index list (1-based → 0-based, flat) | `highlevel`, `freeze`, `center` |
| `g` | Group index list (1-based → 0-based, nested) | `coupling`, `cbond`, `cangle`, `track_index` |

#### 6.2.5 Flat-dict Unification (Framework Side)

The framework hands every skill backend a **neutral flat dict** (`task['config']['skill_config']`). Type *legality* belongs to the skill: its own `build_config` / `normalize` / `validate` reject bad values and report back, exactly like an AI-agent call — this is why the framework never normalizes or validates on the skill's behalf. Shape *determinism* belongs to the framework: the three input modes (yaml / classic `.input` / agent `.config`) must produce a key-by-key identically typed flat dict for the shared vocabulary, so a skill schema always sees one canonical value shape.

This guarantee is implemented by a single declarative function, `unify_skill_flat_dict` in `pyrai3md/schema/adapter.py`, called at the end of `_merge_one_skill_config` (the one runtime choke point every framework mode passes through):

- **Vocabulary**: the shared framework/skill keyword design, whose single source of truth is the classic `SECTION_TYPES` type-code tables above (plus the state-key extras `state_spin`/`target_states`/`ci_roots`/`ci_spin`/`active`/`nact_*` and the `FIELD_MAP` alias spellings). Skill-private keywords are unknown to the framework and pass through untouched — the skill `normalize` owns them.
- **Rules**: only string values convert (typed values pass through, except float-coded keys carrying a JSON int, which are cast to the canonical float shape so the modes agree on types as well); `s`-coded keys keep their canonical string form; `;`-separated multi-center strings pass through (the per-center typed projection in `data/core.py` owns them); unparsable values stay raw so the skill schema reports the original error; keys are never renamed.
- **Provenance overlay**: the framework Config overlay in `_merge_one_skill_config` treats a field as user-set when its spelling appears in a raw framework section (`_raw_user_keys`) **or** its typed value differs from the framework default. A user-written value that coincides with the framework default (e.g. an explicit `temperature: 300`) therefore survives instead of falling through to the skill default, while unwritten framework defaults still never overwrite the Maestro/skill layer.
- **Maintenance rule**: when a skill reports a type error in API mode, the fix belongs in this module only (register the keyword in the type tables / extras) — never in a new ad-hoc patch function beside it, and never in the skill.
- **Contract test**: `qualification/verification/test_unification.py` asserts the three-mode equivalence (merge level and single-center SkillConfig level) over the whole official example set.

The standalone `pyrai3md_skill` mode does not pass through the framework at all: the flat JSON is the agent's responsibility, and the skill's own validation error messages are the feedback channel the agent uses to correct it.

---

## 7. Calculation Profile Design

### 7.1 Why Profiles

Backend computations are structured as module chains or JSON blocks rather than single API calls.
The same scientific property may require multiple backend modules.

Users should specify **scientific tasks**, not backend modules:

```
Incorrect: request specific backend modules
Correct:   specify profile name (e.g. casscf_gradient)
```

The profile internally resolves to the required module/step chain.

### 7.2 Profile Structure

Each profile maps to:
- A **capability** (energy, gradient, nac, frequency, etc.)
- A **method** (casscf, caspt2, etc.)
- A **step chain** (ordered backend operations)

---

## 8. Workflow Construction

### 8.1 Principle

```
Profile name
    → compile_workflow() → WorkflowGraph
    → build_input(config, workflow) → Backend input file
    → Execute
    → Parse output
```

The workflow object is compiled once during backend initialization.

### 8.2 WorkflowGraph

Represents the ordered step chain with metadata flags:

```python
class WorkflowGraph:
    profile: str
    steps: list             # Ordered step names
    needs_gradient: bool
    needs_nac: bool
    needs_frequency: bool
```

---

## 9. Runtime Environment

### 9.1 Runtime Parameters

Skill-specific runtime parameters are configured in `qualification/pmd_env.yaml`
under the `skills.<skill_name>` section.

Common parameters:
- `<skill>_path`: Backend installation directory
- `<skill>_project`: Default project name
- `scratch_path`, `keep_tmp`, `thread`, `slurm_*`

There is no `<skill>_workdir` keyword: the calculation directory is derived per run by
the backend and its runner (`resolve_workdir`), so a working directory is never a
user-facing setting.

These defaults are also defined in `schema/reference.py` and can be overridden
by the config passed to the backend at initialization.

Scheduler parameters (Slurm partition, nodes, ntasks, time) are configured
in the `slurm` section of `pmd_env.yaml`.

### 9.2 Dynamic Execution Context

The working directory should not belong to fixed configuration.
One configuration can generate many calculations (e.g., trajectory steps).

```
Config        → fixed
RunContext    → dynamic (workdir, step index, etc.)
```

### 9.3 Test Switch

The `test` field in `pmd_env.yaml` controls whether a skill is included in automated testing:

```yaml
skills:
  <skill_name>:
    test: false    # Set to true to enable automated testing
```

Only skills with `test: true` will have test tasks generated by the manager.

---

## 10. Execution Interface

Local and SLURM execution logic resides inside the skill backend.

Both share the same pipeline:
1. Prepare input
2. Execute backend
3. Parse output

A separate generic executor layer is unnecessary at the current stage.

---

## 11. Result Interface

The backend returns a unified computational result compatible with PyRAI3MD data core.

Required properties:
- Structure (atoms, coordinates)
- Electronic state
- Energy (per state)
- Gradient (per state, per atom)
- NAC (non-adiabatic coupling, if requested)
- SOC (spin-orbit coupling, if requested)
- Dipole moment
- Transition dipole moment
- Population

### 11.1 Result Data Flow

After the task config restructuring, every task holds a task config of the
form `{'skill_config': flat_dict, 'results': {}}` (built by `data/core.py`).
Each skill's `_store_results()` writes the formatted results into
`self._task_config['results']`. The result dict keys must align with the
`Properties` schema field names so composite tasks aggregate correctly.

The Properties schema is defined in the `Properties` class of
`pyrai3md/schema/config.py`. Standard fields include `energy`, `gradient`,
`nac`, `soc`, `hessian`, `frequencies`, `dipole`, `computed_charges`,
`fin_coord`, `fin_velocity`, `fin_cell`, `transition_dipole`, `osc_strength`,
`excitation`, etc.

**Note**: `coordinates` is not a Properties field — it is system information
passed by the framework through `calculate(atoms, coordinates)` and must not
appear in results. The exception is `fin_coord`, the transport channel key
for the final structure after skill-internal iteration (contract 5 below):
it is consumed by the framework and popped, never entering `state.results`.

### 11.2 Result Return Contract

1. **Plain dicts are not validated**: skill results are plain dicts; no
   pydantic model validation is applied. Keys reference the `Properties`
   field names of each skill schema for the main framework to extract.
2. **computed_charges semantics**: `System.point_charges` strictly means
   background charges embedded into the input (framework semantics
   `(ncharge, 4)` = `[q, x, y, z]`); atom charges computed by the skill
   (extracted by the framework to update background charges) use the new
   key `computed_charges` `(nstate, natom)`, defined in Properties and
   synchronized across the framework and all 10 skill schemas.
3. **transition_pair is not part of the returned dict**: `transition_pair`
   is derived both by the framework normalize (from `coupling`) and by each
   skill's normalize; the returned dict only carries `transition_dipole` and
   `osc_strength` ordered by pair.
4. **transition_dipole shape**: `(npair, 3)`, extracted pair by pair
   according to `config.transition_pair` (in the common case the pairs are
   all ground-to-state pairs `[[0, i]]`, so `npair == nstate`).
5. **fin_coord transport channel**: after completing internal iteration
   (optimize/md/...), a skill reports the final structure `(natom, 3)`
   Angstrom under the `fin_coord` key of the returned dict; without internal
   iteration it returns an empty list (missing key equals empty, keeping
   compatibility with skills that do not implement the contract). The
   framework (`Calculator._apply_skill_geometry`) compares it with the
   current system coordinates and, when different, updates
   `state.pmd.system.coordinates` so subsequent workflow logs and `opt.xyz`
   outputs carry the correct structure; a length mismatch against `natom` is
   a contract violation and raises `ValueError` immediately (fail-fast);
   after consumption the key is popped from results and never enters
   `state.results`. The field is registered in the framework and all 10 skill
   schemas; pyscf, molcas (`parse_fin_coord`, probing `['Opt.xyz', 'opt.xyz']`
   case variants in order), orca (from `{project}.xyz`), cp2k (last frame of
   GEO_OPT/MD), lammps (ff_optimize/ff_nve/ff_nvt/ff_npt) and bagel
   (geometry_optimization) actually return it; openqp/xtb have not
   implemented it yet and must follow the same contract when they support
   internal iteration.
6. **fin_velocity transport channel**: all 10 skill schemas reserve
   `fin_velocity` `(natom, 3)` Bohr/au, and every skill parser returns the
   key uniformly (empty list when there is no velocity output), keeping the
   result contract aligned. The framework consumes it the same way it consumes
   `fin_coord`: a non-empty value becomes `system.velocities` and the kinetic
   energy is recomputed from it, an empty one leaves the state untouched.
   Currently only the lammps MD profiles (ff_nve/ff_nvt/ff_npt) return real
   final velocities. New skills must reserve the channel as well.
7. **fin_cell transport channel**: an engine that owns the cell degrees of
   freedom (LAMMPS `fix npt` with x/y/z cell terms; CP2K `CELL_OPT` once a
   profile emits it) reports the box its returned coordinates live in under the
   `fin_cell` key, as the six cell parameters `[a, b, c, alpha, beta, gamma]`
   or the nine flattened lattice components -- the two shapes
   `schema.normalize` already accepts for `cell`. The framework derives
   `cell`, `lattice_vector` and `volume` from it and writes all three in one
   update, because the builders prefer `lattice_vector` and a half-updated
   state would keep propagating the old box. An empty or absent value is the
   normal case (fixed-cell run) and leaves the state untouched; a value
   returned for a system whose `pbc` is all zero is a contract violation and
   raises, since it means the engine ran on a periodic cell the input never
   declared. New skills whose engine can move a cell must reserve the channel.

**Properties field reference** (complete definition in `schema/config.py`,
`Properties` class, line 452-531):

| Field | Type | Description |
|------|------|------|
| `energy` | `List[float]` | (nstate,) total energies |
| `kinetic_energy` | `float` | kinetic energy |
| `pot_energy` | `List[float]` | (nstate,) potential energies |
| `total_energy` | `List[float]` | (nstate,) total energies |
| `fin_coord` | `List[List[float]]` | (natom, 3) final structure transport channel, Angstrom; empty = no internal iteration |
| `fin_velocity` | `List[List[float]]` | (natom, 3) final velocity transport channel, Bohr/au; reserved in all skills, consumed by `Calculator._apply_skill_velocities` (kinetic energy recomputed, key popped before `state.results`) |
| `gradient` | `List[List[List[float]]]` | (nstate, natom, 3) gradients |
| `stress_tensor` | `List[List[List[float]]]` | (nstate, 3, 3) stress tensor |
| `hessian` | `List[List[List[float]]]` | (nstate, 3*natom, 3*natom) Hessian |
| `frequencies` | `List[List[float]]` | (nstate, 3*natom) frequencies |
| `norm_mode` | `List[List[List[List[float]]]]` | (nstate, 3*natom, natom, 3) normal modes |
| `reduced_mass` | `List[List[float]]` | (nstate, 3*natom) reduced masses |
| `dipole` | `List[List[float]]` | (nstate, 3) dipole moments |
| `computed_charges` | `List[List[float]]` | (nstate, natom) atom charges computed by the skill, extracted to update background charges |
| `transition_dipole` | `List[List[float]]` | (npair, 3) transition dipoles |
| `osc_strength` | `List[float]` | (npair,) oscillator strengths |
| `excitation` | `List[float]` | excitation energies |
| `nac` | `List[List[List[float]]]` | (nnac, natom, 3) nonadiabatic couplings |
| `soc` | `List[List[float]]` | (nsoc, 1) spin-orbit couplings |
| `ir_intensity` | `List[float]` | IR intensities |
| `uv_data` | `List[List[float]]` | UV spectrum data |

---

### 11.2.1 Missing blocks and the read entry (2026-11-12)

A requested property whose block is absent is a failure, not a zero: Molcas
raises for a requested energy/gradient block, CP2K raises for a requested
stress tensor and for a requested ground-state energy. The zero baseline is
kept only for a property the run did not request.

The read entry is self-contained: each skill carries its own
`service/read_support.py`, and the former framework module
`pyrai3md.skills.read_support` no longer exists. `read_calculation` therefore
imports only within its own skill package.

## 12. Development Workflow

The agent CLI is responsible for skill development and maintenance.

Command formats:

```
pyrai3md <skill>_<task>.yaml          (input mode)
pyrai3md_agent <skill>_<task>.config  (agent mode)
pyrai3md_skill <skill>_<task>.json    (skill mode)
```

The `pyrai3md_skill` command executes via subprocess directly within the skill context,
bypassing the main PyRAI3MD orchestration.

The skill package itself should not contain a `compiler.py`.

The qualification config suffix contract (converter-generated `.json` /
`.config` / rendered `.input`, and the System fields of `.config` living in
the molecule section) belongs to the qualification and test subsystems; see
[QUALIFICATION_DEVELOPMENT.md](QUALIFICATION_DEVELOPMENT.md) and
[SKILL_TEST_DEVELOPMENT.md](SKILL_TEST_DEVELOPMENT.md).

### 12.1 Packaging and Distribution

`setup.py` `package_data` must declare every skill's resources —
`manifest.yaml`, `skill_guide.md`, and `tests/*.yaml` (lammps additionally
ships its test data files) — so installed packages keep the skill
discoverable and testable. `qualification/verification/test_skill_packaging.py`
guards this requirement (manifest presence and resource coverage per skill).

---

## 13. Architecture Summary

```
PyRAI3MD Framework Pipeline:
    → Environment.initialize() → Maestro.populate()
    → Interface → Adapter → Config
    → build_state → PyRAIState (task descriptors, no backend instances)
    → Calculator.initialize() → backend instantiation
    → execute(state) → Calculator + Workflow → runner.run() → Result

Skill Internal:
    → manifest.yaml (discovery + control_parameters)
    → query.py (sole exit point for skill information)
    → schema (config → normalize → validate)
    → backend (<Skill>Backend)
        → dual-mode __init__: PyRAIConfig or dict
        → read pyrai_config.config + pyrai_config.skill_config
    → workflow compiler (Profile → WorkflowGraph)
    → input builder (WorkflowGraph → backend input)
    → execution service (local / slurm)
    → result parser (backend output → standardized dict)
```

**Core principle:**

Compile the computational workflow once during backend initialization,
then execute thousands of lightweight `calculate()` calls during molecular dynamics.

This architecture enables:
- Lazy loading with multi-skill resolution
- Independent skill development (no framework imports)
- Shared PyRAIState across all backends
- Control parameters declared in manifest for user input validation
- AI-assisted skill generation
- Efficient nonadiabatic dynamics
- Unified quantum chemistry backend management

---

## 14. Skill Internal Architecture Evolution

This section records the completed and planned internal architecture upgrades for skills.

### 14.1 Completed Refactoring

The following changes have been implemented as the foundation for future skill evolution:

- **Query mechanism**: All skills provide `get_defaults()` via `query.py`; framework obtains defaults and derives profiles through query instead of direct schema access.
- **Framework decoupling**: `data/core.py` no longer imports skill schema pipeline (normalize/validate); all skill-specific logic goes through query.
- **Flat dict passthrough**: `build_pyrai_config` mounts `_skill_config` attribute for skill backends to consume.
- **Backend unification**: All backends extract `config_dict` from `_skill_config`.
- **Adapter null-value protection**: All skill adapters backfill schema defaults when values are missing.
- **SkillConfig nested submodels**: molcas_skill and pyscf_skill migrated from flat dict to nested SkillConfig (System/Properties/Calculation/Environment submodels) with `FIELD_TO_SUBMODEL` mapping; service layer uses attribute access (`config.method` instead of `config['method']`).
- **Backend config_dict elimination**: molcas/pyscf backends hold `SkillConfig` instance (`self.config`) instead of flat dict.

### 14.2 Planned Evolution

**Goal**: Migrate all skills to use Pydantic Config model as runtime data container (not just validation gate), matching the framework's PyRAIConfig attribute-access pattern.

Current pattern (flat dict after validation):
```python
config_dict = config.model_dump()
method = config_dict['method']
```

Target pattern (attribute access throughout lifecycle):
```python
method = config.calculation.method
basis = config.calculation.basis
```

Migration strategy (per-skill, gradual):

1. **Phase 1** (current): Framework side complete; molcas/pyscf migrated to SkillConfig.
2. **Phase 2**: Migrate remaining skills (bagel/orca/xtb/openqp/lammps/cp2k) to nested SkillConfig.
3. **Phase 3**: Service functions accept submodel parameters instead of full config dict.
4. **Phase 4**: Deprecate `flatten_extended_config` once all skills use attribute access.

Each phase maintains backward compatibility: skills support both flat dict input and Config model input until all callers migrate.

### 14.3 skill_project and project Synchronization

Each skill's Environment submodel has a `skill_project` field (e.g. `molcas_project`, `pyscf_project`) that must stay synchronized with the generic `Calculation.project` field.

Design principle:

- `Calculation.project` is the **single source of truth**; all internal code uses `config.project`
- `Environment.skill_project` is preserved as user input field (converter output), always synced with `project`
- Mapping logic lives in skill's `build_config`, maintaining framework-skill isolation

Completed: pyscf_skill, molcas_skill. Other skills (bagel/orca/xtb/openqp/lammps/cp2k) pending.

### 14.4 Persistent LAMMPS Execution Channel (landed)

A framework MD step used to pay a full LAMMPS cold start: a new `lmp`
process, `module load`, `mpirun` launch, `read_data` and the pair-table setup,
for one `run 0`. Keeping one LAMMPS alive across steps is implemented as
`python_enabled`: a separate service process holds the instance, so PyRAI3MD
never imports `mpi4py`/`lammps` and MPI stays a deployment detail of the
skill. The design document it came from no longer exists; the surviving record
is `dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` section 11.

Two things deferred it were resolved by other means rather than proven out. No
`jump SELF` and no stdin loop is involved: the service binds a Unix socket, the
framework ships one step's geometry, and the instance is loaded once at INIT.
The repeated-`read_data` question never had to be answered on the hot path
either, because the coordinates now go in through the Python array interface and
the periodic image flags that make them mean the same positions go in beside
them (section 11.12), with `read_data` kept for the subprocess path and for the
INIT load. The force-field compression of section 5.6 landed first and is
consumed by both paths: the per-step pair mixing drops from 6102 generated
terms to 3 and the settings include from 8622 lines to a handful.

---

## 15. Appendix: Per-Skill Result Extraction Reference

Per-skill result extraction status (all paths relative to the skill root).
Every skill provides a `parse_output`-style entry under `service/parser/`
plus a `format_results` helper that converts numpy arrays to lists.

### pyscf_skill

- `service/parser/output_parser.py` — `parse_output(runtime, config, workflow)`
  extracts raw results from the runtime dict (`runtime.get('results', {})`);
  `format_results(results, config)` formats into Properties-compatible keys.
- **fin_coord**: optimize (pyscf.geomopt internal optimization) reports the
  final structure in the get_runtime() results; `backend.calculate()` applies
  `setdefault('fin_coord', [])` on the other paths.
- Open items: filter non-Properties keys (e.g. `coordinates`); confirm full
  key alignment with Properties.

### molcas_skill

- `service/parser/log_parser.py` — `parse_log(log_path, config)` parses the
  Molcas log; returned keys: `energy`, `gradient`, `nac`, `soc`, `dipole`,
  `transition_dipole`, `osc_strength`, `zero_point_energy`, `enthalpy`,
  `entropy`, `free_energy`, `frequencies`.
- `dipole` (nstate, 3) Debye from the CASSCF/CASPT2 `1-st cartesian moments`
  Total rows (CASPT2 blocks preferred); `osc_strength` from the RASSI
  `Dipole transition strengths (spin-free states)` table; `transition_dipole`
  (npair, 3) Debye from the RASSI `MEES` `MLTPL 1` matrices extracted by
  `transition_pair` (zero when the block is absent in old logs).
- Thermochemistry from the fixed MCKINLEY 298.15 K block; `frequencies`
  (3N-6, cm^-1) from `Harmonic frequencies in cm-1` (imaginary frequencies
  negative; mode vectors live in `freq.molden`).
- **opt_converge**: only `casscf_geometry_optimization` reports it;
  `parse_output` maps the log completion status to `opt_converge`
  (`not log_not_converged`), so unconverged runs return `False`.
- **fin_coord**: `parse_fin_coord()` (`service/parser/output_parser.py`)
  collects the final structure — `casscf_geometry_optimization` reads
  `$Project.Opt.xyz` (probing `['Opt.xyz', 'opt.xyz']` case variants in
  order); `casscf_abinitio_dynamics`/`casscf_surface_hopping` read the last
  frame of `$Project.md.xyz`. Angstrom, atom-count validated, missing file or
  mismatch fails fast; non-iterative profiles return nothing.
- `casscf_geometry_optimization` supports MECI/MECP via
  `opt_state_i`/`opt_state_j`; dynamics run inside a single Molcas execution
  (`>> FOREACH ITER` loop, initial velocities mandatory).
- Initial orbital auto-loading via `title.StrOrb` (`guess_orbital`
  fill-if-empty injection, molcas only).

### bagel_skill

- `service/parser/log_parser.py` — `parse_log(config, calcdir, workflow)`
  parses the BAGEL log; returned keys: `energy`, `gradient`, `nac`,
  `coordinates`, `hessian`, `frequencies`.
- **fin_coord channel**: the geometry_optimization profile returns the final
  structure (last `*** Geometry ***` block of the log, falling back to
  `opt.molden`) together with `opt_converge` (a run passing the completion
  check counts as converged).
- Open items: filter `coordinates`; add `soc` if BAGEL supports it.

### orca_skill

- `service/parser/output_parser.py` — `parse_output(config, calcdir, workflow)`
  parses ORCA log/engrad/hess files; returned keys (aligned with Properties):
  energy, gradient, nac, soc, dipole, transition_dipole, osc_strength,
  excitation, frequencies, hessian, fin_coord, opt_converge.
- **fin_coord**: the optimize profile reads `{project}.xyz` (ORCA updates it
  after every cycle; the input geometry only lives in the `.inp`); the log
  `CARTESIAN COORDINATES` block is a fallback only. Convergence uses the
  framework contract key `opt_converge`.
- `transition_dipole`/`osc_strength` follow `transition_pair` order (npair);
  frequency profile reads `{project}.hess` `$vibrational_frequencies` with
  the log section as fallback (dual absence fails fast).
- TDDFT: `NACME true` inside `%tddft`; `tddft_frequency` forces `NumFreq`;
  SOC requires a closed-shell singlet reference with `Triplets true` +
  `DoSOC true`; `tddft_nroots = max(nsinglet, ntriplet)` with
  `irootlist`/`trootlist` gradient-root splitting.

### openqp_skill

- `service/parser/output_parser.py` — `parse_output(config, calcdir, workflow)`
  parses OpenQP output; returned keys: energy, gradient, nac, soc, dcme,
  nacme, coordinates, opt_converged, hessian, frequencies, norm_mode
  (fin_velocity reserved empty).
- Pending alignment: `coordinates` (optimize final structure) does not use
  the `fin_coord` channel yet; `opt_converged` does not use the framework
  contract key `opt_converge` yet.

### xtb_skill

- `service/parser/output_parser.py` — `parse_output(config, calcdir, workflow)`
  parses xTB engrad/charge files; returned keys (aligned with Properties):
  energy, gradient, computed_charges.
- `computed_charges` (1, natom, state-resolved) replaces the legacy
  `atom_charges` key; the key is absent when the charge file is missing
  (schema default empty = not computed).
- engrad dependency contract: `build_xtb_args` always appends `--grad` so
  every profile writes the engrad file the parser reads.

### lammps_skill

- `service/parser/output_parser.py` — `parse_output(config, calcdir, workflow)`
  parses LAMMPS output; returned keys: energy, gradient, plus fin_coord for
  the internal-iteration profiles (ff_optimize/ff_nve/ff_nvt/ff_npt) and
  fin_velocity (Bohr/au) for the MD profiles.
- Output files follow the per-task basename: `{project}_{profile}.log/.xyz/.cond`
  (`{title}` is `{project}_force` for ff_gradient, see
  `service.utils.output_basename`); a single point additionally prints its own
  result files `{title}.energy/.coords/.force/.stress`, which are what the
  parser reads back instead of the log for those profiles.
- gradient = -force (converted to hartree/bohr per `lammps_units`); the
  internal-iteration log energy is a per-atom average
  (`thermo_modify norm yes`), multiplied by natom before the hartree
  conversion.
- fin_coord comes from the `write_dump` xyz last frame (Angstrom,
  atom-count validated); fin_velocity comes from the 7-column custom dump
  `{project}_{profile}.cond` (id x y z vx vy vz) converted back to Bohr/au.

### cp2k_skill

- `service/parser/output_parser.py` — `parse_output(config, calcdir, workflow)`
  parses CP2K log/force/tdfrc/xyz files; returned keys: energy, gradient,
  nac, soc, dipole, transition_dipole, osc_strength, excitation, plus
  fin_coord for GEO_OPT/MD and opt_converge for GEO_OPT (fin_velocity
  reserved empty).
- gradient = -force (CP2K prints atomic forces, hartree/bohr): the ground
  state from `{project}-force-1_0.xyz`; excited states from the
  `{project}-td_forces-1_0.tdfrc` force blocks in the `&PRINT &FORCES LIST`
  order (the sorted target excited states) — a block-count mismatch fails
  fast.
- Excited-state energies and spectral data come from the TDDFPT table:
  `excitation` (eV), `energy[state] = s0 + exc` (converted to hartree);
  `transition_dipole`/`osc_strength` (npair) follow `transition_pair` order,
  only ground->state pairs carry computed values.
- fin_coord: GEO_OPT and standalone MD both take the last frame of
  `{project}-pos-1.xyz` (Angstrom, atom-count validated, missing fails fast).
- Ground-state energy is the last `Total energy:` of the run.

### 5.8 `execution_mode` and the Slurm Job Body

`execution_mode` is a skill-side switch read by the backend before the
calculation runs. `local` executes in the current process. `slurm` calls
the skill's own `run_slurm`, which writes the configuration JSON and an
sbatch script and blocks on submission. The job body runs the skill agent
with that JSON on the compute node; the agent executes the calculation
in-process there. It must not select `slurm` again, or the job would
submit another job recursively.

This is a different layer from the framework's `remotetask`. A workflow
runtime (for example an MD child) can be serialized, submitted by the
framework as `python -m pyrai3md <child>.rt`, and run its skill locally
inside the node. A skill calculation that is submitted by its own
`run_slurm` keeps the framework child local and lets the skill own the
batch job. Both paths are legal; the ownership boundary is that the
framework dispatches workflow runtimes, while a skill that declares
`execution_mode=slurm` dispatches its own engine.

The rule is enforced where the job body's config is written, not where the
caller chooses the arm: a skill's `slurm_runner` serializes the config into the
script with `environment.execution_mode` forced to `local`
(`skills/pyscf_skill/service/runner/slurm_runner.py`,
`skills/qdm_skill/service/runner/slurm_runner.py`), because the serialized
parent config carries `slurm` and an agent that reads it back verbatim submits
again from inside the job. That is not hypothetical: an adaptive round whose
reference skills kept the inherited value fanned one labelled batch out into
hundreds of queued PySCF jobs before it was stopped, and the trail led to the
JSON in the job directory rather than to any framework decision. A template
argument cannot carry this rule - the trailing `local` the job body used to pass
was read by nothing, which is exactly why the recursion stayed invisible until
the queue showed it.
