# Skill Guide: qdm_skill

## Overview

Use qdm_skill to evaluate one-dimensional analytic nonadiabatic model
Hamiltonians (Tully 1990 examples I/II/III) and to propagate fully
quantum reference dynamics on them. The skill is a pure
Python + numpy service with no external program dependency.

It provides:
- Adiabatic single-point properties: energies, gradients, and
  nonadiabatic coupling (NAC) vectors via analytic Hellmann-Feynman
  relations.
- A Strang split-operator reference solver producing state
  populations, norm, and density snapshots for benchmarking.

## Setup Requirements

- Python with numpy importable in the current environment (no other
  dependency).
- The models are two-state by construction: set `nstate = 2`.

## Geometry convention (QDM-specific)

QDM models are one-dimensional. The structure is represented by a
single **dummy atom** (symbol `X`); only the x component of the first
atom enters the model, y/z are ignored:

```
1
Tully 1
X  -7.937658  0.0  0.0
```

- Framework coordinates are stored in Angstrom; the skill converts
  the model coordinate internally via `x(Bohr) = x(Angstrom) ×
  ANGSTROM_TO_BOHR` (−7.9376581 Å ≙ −15 Bohr).
- The dummy symbol `X` is never looked up in any element table by the
  skill; it only carries the model coordinate.
- Initial momentum for reference runs: reuse the framework velocity
  mechanism (`{title}.velo`, Bohr/au); the solver uses
  `k0 = model_mass × velocities[0][0]`. Without velocities it falls
  back to `wavepacket_k0`.

## Supported Profiles

Each method (`tully1`, `tully2`, `tully3`, `lz1`) exposes four
profiles:

| Profile | Capability | Output |
|---------|-----------|--------|
| {method}_energy | energy | E |
| {method}_gradient | gradient | E + G (no NAC vectors) |
| {method}_nac | nac | E + G + NAC (AIMD per-step superset) |
| {method}_reference | md | full split-operator quantum dynamics |

Driver semantics (framework `control.driver`):
- `driver=main`: the main framework integrates the dynamics (AIMD)
  and calls the skill per step; jobtype `md` resolves to
  `{method}_nac` by default (or `{method}_gradient` when the planned
  nactype does not consume backend NAC vectors).
- `driver=skill`: the skill integrates the dynamics internally via a
  single SingleShot call; jobtype `md` resolves to
  `{method}_reference`.

## Parameters

| Key | Default | Description |
|-----|---------|-------------|
| model_mass | 2000.0 | wavepacket mass (au); named `model_mass` because `mass` is reserved for framework atomic masses |
| tully_a / tully_b / tully_c / tully_d / tully_e | None | Tully model parameter overrides (None = standard Tully 1990 values) |
| lz_a / lz_b | None | Landau-Zener model parameter overrides (None = model class defaults) |
| ref_ngrid | 2048 | grid points for the reference propagation |
| ref_xmin / ref_xmax | None | grid boundaries (Bohr); None keeps the fixed default window [-30, +80] Bohr, x0 outside the window raises an error |
| ref_dt | 20.0 | time step (au) |
| ref_nsteps | 500 | number of time steps |
| ref_save_every | 1 | density snapshot stride |
| wavepacket_sigma | 0.5 | Gaussian width (Bohr) |
| wavepacket_k0 | 10.0 | fallback initial momentum (1/Bohr) when no velocities are provided |

## Reference output files (written into the calculation directory)

- `{project}_population.dat`: time vs. state populations
- `{project}_norm.dat`: time vs. total norm
- `{project}_grid.dat`: coordinate grid (Bohr)
- `{project}_density.npy`: density snapshots (nsnap, ngrid, 2)

The results dict additionally returns `final_population`,
`final_norm`, `x0`, `k0`, and the grid/time parameters.

## Workflow

1. **Discover Available Methods**: use the built-in query module to
   list profiles and capabilities.
2. **Create Configuration File**: prepare a JSON config with the
   dummy-atom geometry, profile, method, and model parameters.
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m pyrai3md.skills.qdm_skill.agent config.json [local|slurm]`
   - **Python API**: `from pyrai3md.skills.qdm_skill.backend import QDMBackend`
4. **Obtain Results**: results are returned by `calculate()` and
   written back into the task config (`results` section).

## Query Available Methods

qdm_skill provides a self-contained query module that reads its own
manifest.yaml:

```python
from pyrai3md.skills.qdm_skill import query

print(query.list_profiles())
# ['tully1_energy', 'tully1_gradient', 'tully1_nac', ...]

print(query.get_defaults())
# {'model_mass': 2000.0, 'ref_ngrid': 2048, ...}

print(query.resolve_profile('tully1', 'energy gradient'))
# 'tully1_gradient'
```

## Execution Modes

- `execution_mode: local` (default): in-process numpy evaluation.
- `execution_mode: slurm`: the config is serialized into the
  calculation directory and submitted with a blocking `sbatch -W`;
  the cluster-side agent writes `{project}_results.json` which the
  backend collects after submission. slurm keywords: `slurm_time`,
  `slurm_partition`, `slurm_ntasks`, `slurm_memory`, `slurm_command`.

- `thread`: OpenMP threads, written into `OMP_NUM_THREADS` once per process
  (first writer wins, #302); `qdm_thread` is the framework spelling and a
  written pair folds to its larger value (#306).
- `keep_tmp`: 0 removes the calculation directory at the end of
  `calculate()`, once the results are collected; 1 keeps the `.dat` dumps
  it wrote (energy, gradient, population, norm). The next step rebuilds the
  directory, so clearing cannot strand a stepped run (#307).

## Limitations

- Two-state models only (`nstate = 2`).
- One-dimensional coordinate; only the first atom x component is
  used.
- Surface hopping (FSSH) propagation on these models is not part of
  this skill; the framework MD workflow consumes the per-step E/G/NAC
  and performs its own integration.
