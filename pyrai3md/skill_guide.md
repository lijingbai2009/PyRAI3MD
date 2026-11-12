# Skill Guide: pyrai3md

## Overview

`pyrai3md` is the framework, not a scientific engine. It owns the input
schema, the task/planning layer, the workflow loop, result logging and the
protocols every skill implements; each engine (PySCF, ORCA, OpenQP, MOLCAS,
BAGEL, CP2K, xTB, LAMMPS, QDM, NN-PES) is a skill under
`pyrai3md/skills/<name>_skill/`. The framework never imports an engine, and
skill production code never imports the framework (the one exception is
`pyrai3md.utilities`); they meet at the skill manifest, the `query` module and
the `SkillConfig` schema.

The same tree is used by three entry points:

| Entry point | Command | Input |
|---|---|---|
| classic / YAML | `pyrai3md <input>` | `&SECTION` text, example YAML, `.rt` |
| agent | `pyrai3md_agent <config>` | sectioned JSON config |
| skill | `pyrai3md_skill <config.json>` | flat skill JSON (launches the skill agent) |

`pyrai3md/manifest.yaml` is the framework's own discovery document. It lists
the accepted input sections and their keyword names, the entry points, the
pipeline functions, the workflow classes and the units - the same vocabulary a
skill's `manifest.yaml` uses for its profiles and capabilities. Types and
defaults stay in the model named next to each section
(`pyrai3md.schema.config`), so the manifest is not a second source of truth.

## Setup Requirements

- Python 3.9+ and the package installed from the checkout:

  ```bash
  pip install -e .
  ```

  The NN-PES workflows additionally need TensorFlow
  (`pip install -e ".[nn]"`). A host without it plans those cases as named
  skips, not as failures. Numba is the optional `.[numba]` extra.
- Each engine skill has its own installation (`pyscf`, `openqp`, `orca`,
  `molcas`, `bagel`, `cp2k`, `xtb`, `lammps`) and its own environment
  variables. Its `skill_guide.md` is the authority for those; the framework
  only forwards the `&ENVIRONMENT`/skill keys.
- `~/.pyrai3md/pmd_env.yaml` is created on the first `pyrai3md test` run. It
  overrides the packaged `qualification/pmd_env.yaml` for host paths,
  scheduler settings, deploy targets and the per-skill qualification
  switches. Source-tree runs also read
  `./pyrai3md/qualification/pmd_env.yaml`.

## Supported Workflows

The `jobtype` field of `&CONTROL` selects a workflow class (or, for
`read`, the read-only extraction flow):

| jobtype | Workflow | What it does |
|---|---|---|
| energy | `workflow.single.SingleShot` | one evaluation |
| optimize | `workflow.optimize.Optimize` | optimizer loop, cell and path variants |
| path | `workflow.path.Path` | MEP / IRC / NEB band |
| md | `workflow.md.AIMD` | classical / nonadiabatic dynamics |
| surface_hopping | `workflow.md.AIMD` | NAMD with hops |
| frequency / hessian / num_hess / num_freq | `workflow.hessian.Hessian` | analytic or displaced-gradient Hessian |
| scan | `workflow.scan.Scan` | one-dimensional scan |
| train / prediction | `workflow.train.Train`, `workflow.prediction.Prediction` | ML potentials |
| adaptive | `workflow.adaptive.AdaptiveSampling` | iterative MD/train/select |
| search | `workflow.search.GridSearch` | grid evaluation |
| read | `workflow.read.read_calculations` | read-only label extraction from finished artifacts; no engine and no data file |

The authoritative mapping is `workflows:` in `pyrai3md/manifest.yaml`; the
capability list there (`energy`, `gradient`, `hessian`, `nac`, `soc`,
`dipole`, `transition_dipole`, `overlap`, `stress`, `field`, `qmmm`) is the
vocabulary a skill's manifest also declares.

`&CONTROL driver` separates who owns the loop:

- `driver: main` - the framework runs the optimizer / MD loop and asks the
  skill for single-point energies and gradients;
- `driver: skill` - the skill runs its own loop and returns the final
  structure through the `fin_coord` channel and its convergence through
  `opt_converge`; the framework records one evaluation and settles the run
  from those two fields.

## Input

### Classic `&SECTION` text

```text
&CONTROL
title         water_opt
jobtype       optimize
theory        pyscf
driver        main

&MOLECULE
ci            1
spin          0
pbc           0

&OPTIMIZE
opt_method    minima
opt_maxit     100
```

The parser reads text or JSON, keeps values as raw strings, then resolves
coordinates from `atoms`/`coordinates`, a same-basename `.xyz`, or a
`&MOLECULE coord` path. `&MOLECULE` is the classic spelling of the framework's
`system` section (`molecule: system` in `input_schema.section_aliases`);
`&MD` is `dynamics`. The retired `qm`/`abinit` names are still read as
`theory`/`reference`, and a method label may carry the same prefix
(`qm.pyscf`); `schema.normalize` strips it in one place, `skills.query.method_label`.

### Example YAML

The shipped examples under `pyrai3md/examples/` use sectioned YAML with the
same section names as the classic input:

```yaml
control:
  title: water_energy
  jobtype: energy
  theory: [pyscf]

atoms: [O, H, H]
coordinates:
  - [0.0, 0.0, 0.0]
  - [0.0, 0.757, 0.587]
  - [0.0, -0.757, 0.587]

molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]

pyscf:
  method: dft
  basis: def2-svp
```

The parser resolves a same-basename `.xyz` when coordinates are not in the
file. The qualification suite runs every example through both the YAML and
the classic carrier. The `input:` form under `skills/<name>/tests/*.yaml` is
the skill-test spec; the qualification converter turns it into the flat skill
config and the two carriers above, so it is not a `pyrai3md` user input.

### Agent JSON

`pyrai3md_agent` reads the sectioned JSON form:

```json
{
  "control": {"title": "water", "jobtype": "energy", "theory": "pyscf"},
  "molecule": {"atoms": ["O", "H", "H"], "coordinates": [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]},
  "pyscf_skill": {"method": "dft", "basis": "def2-svp", "execution_mode": "local"}
}
```

The section keys are the skill names (`pyscf_skill`, `openqp_skill`, ...).
Programmatic use is `PyRAI3Agent.from_file(path).run()`; this entry writes no
log. The command line uses `run_cli()`, which goes through the framework
`RunLogger` and writes the same `<title>.log` closing status as `pyrai3md`.

### Runtime payloads

`.rt` files are serialized `PyRAIState` runtimes (restart / remote dispatch)
and bypass the input pipeline. `.rtt` files are single-term remote worker
payloads that write `term_outcome.json`. Neither is a user input format.

### Read-only extraction (`&READ`)

`jobtype read` reads labels from finished calculations. It starts no
engine, writes no data file and returns the frames to its caller; the
`tools/training_data.py extract` command owns the dataset.

```text
&control
title read_run
jobtype read
theory molcas

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

`path` names calculation directories, output files, or a file listing
one path per line. `project` overrides the output file stem (empty
discovers the unique artifact stem). `profile` selects the skill profile
used to parse an optimization/MD directory; empty derives it from the
requested labels. Sections are delimited by the next `&SECTION` header;
there is no `&end`.

`read` binds exactly one theory skill, no reference, `driver: main` and
no multiscale partition. File-based readers exist for molcas, orca,
bagel, xtb, openqp, cp2k and lammps; pyscf, qdm and nn_pes compute in
process and are refused by name. See `tools/TRAINING_DATA.md`.

## Command Reference

```bash
# classic or YAML input
pyrai3md input.yaml
pyrai3md input.input
python -m pyrai3md input.yaml

# parse and print the state, do not execute
pyrai3md input.yaml --info

# framework qualification plan (uses ~/.pyrai3md/pmd_env.yaml)
pyrai3md test

# agent CLI (same pipeline as above, sectioned JSON)
pyrai3md_agent config.json

# skill CLI: launch one skill agent as a subprocess
pyrai3md_skill skill_config.json

# post-round verification (files only; never runs an engine)
python tools/verify.py tmp_<stamp>
python tools/verify.py --selftest

# remote qualification round
python tools/deploy.py
python tools/deploy.py --force   # stop the running remote round
```

`pyrai3md test` calls the qualification runner: it loads `pmd_env.yaml`,
plans the enabled unit / example / skill / benchmark rows, runs them and
prints the report. The switches there are three levels - `_runall`, the
per-bucket/skill switch, and each case - with `false` closing, `condition`
letting the next level decide, and `true` certifying.

## Output Layout

A run names every artifact after `&CONTROL title`:

| File | Writer | Contents |
|---|---|---|
| `<title>.log` | `kernel.logger.RunLogger` | banner, input echo, per-step result blocks, `PyRAI3MD End: ... Status: ...` |
| `<title>.md.energies` | MD workflow | one row per step: time, Epot, Ekin, Etot, per-state extras |
| `<title>.md.<n>.energies` | layered/NAMD runs | one row per layer/centre |
| `<title>.md.xyz` | MD workflow | trajectory |
| `opt_status.txt`, `opt.xyz`, `opt_geom.xyz` | optimize workflow | iteration table and final/trajectory geometry |
| `<title>.json` | train / prediction / adaptive | record, model and summary products |
| `term_outcome.json` | remote term worker | one backend term's outcome |

Skill calculation directories and their engine files live under the scratch
path (`tmp_<TITLE>` or `tmp_<role>_<layer>_<TITLE>`); a post-round verifier
reads them without re-running anything.

## Python API

```python
from pyrai3md.pmd import run

result = run('energy.yaml')            # writes <title>.log
```

```python
from pyrai3md.agent import PyRAI3Agent

agent = PyRAI3Agent.from_file('config.json')
result = agent.run()                   # programmatic, no run log
# result = agent.run_cli()             # CLI semantics, writes <title>.log
```

```python
from pyrai3md.skills.openqp_skill import OpenQPBackend

backend = OpenQPBackend(config)
results = backend.calculate(atoms, coordinates, target_states=[0, 1])
```

The skill-facing contract is `<skill>_skill/query.py`
(`get_profiles`, `get_defaults`, `get_registry`, `resolve_profile`,
`list_profiles`, ...) and `backend.calculate(...)`. The framework calls only
those public names; a new skill must register every config field in its
`query.get_defaults()` table, or the framework's generated config silently
drops it.

## Verification and Qualification

`tools/verify.py` is the post-round audit. It reads a finished `tmp_<stamp>`
tree only - it never starts a skill entry point, engine or calculation, and
writes nothing into the round (section G calls the molcas skill's parser to
read its logs; that is file work). Sections A to K cover records vs results, step logs, engine output,
optimizer records, cell verdicts, example carrier parity, paths, MOLCAS legs,
field/gradient identities, adaptive products, and the three skill carriers
(engine deck and final energy). `--selftest` plants a clean round, corrupts
one number per section and requires every corruption to be named.

`tools/deploy.py` syncs the checkout, installs it on the configured host,
starts a qualification round there and downloads `test_<stamp>.log`; the
round directory stays on the host. `pmd_env.yaml` supplies the host, the jump
host, the conda environment and the test switches.

## Development Rules

- Import tree is one-way: `utilities -> schema -> data -> planning/workflow
  -> kernel/skills`. A skill may import `pyrai3md.utilities`; it may not
  import the framework's schema, planning or workflow layers.
- A skill's `manifest.yaml` declares its profiles, capabilities, backend
  class and tunables; `pyrai3md/skills/query.py` reads those declarations.
  No framework file may hard-code a skill directory name.
- Input names are declared once in `pyrai3md/manifest.yaml` (`input_schema`),
  with the type table next to the model. Add a keyword there and to the
  owning model, not to a second list.
- A result channel crosses the boundary by name: `fin_coord` is the final
  structure after a skill-internal iteration, `fin_velocity`/`fin_cell` are
  the dynamics variants, and `opt_converge` is the optimization verdict.
  A skill that writes a different spelling (`opt_converged`) is not read.
- Units are Hartree, Hartree/Bohr and Hartree/Bohr^3 at the framework
  boundary; a skill converts its native units before returning.
- New engine work goes in a skill under `pyrai3md/skills/`, not in the
  framework. See `dev/SKILL_APP_DEVELOPMENT.md` for the package layout,
  test YAML format and `pyrai3md_skill` entry point.
