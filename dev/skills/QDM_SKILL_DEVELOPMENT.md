# QDM Skill Development Guide

Last updated: 2026-10-22 22:30 (section 1.4 and section 9 no longer call FSSH and the quantum-reference benchmark out of scope - `qualification/benchmark/fssh.py` runs the comparison through `SingleShot` with `qdm_project: reference`, and the hopping itself is the framework's `workflow/integrator/fssh.py`; the coupling auto-fill is attributed to the qualification converter. The rest was not re-audited. Before it, 2026-10-05 20:05)

This document covers **QDM-specific** implementation details for qdm_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).
The original physics design (from the former `QDM_DEVELOPMENT.md`, merged into
Appendix A below) covers the design goals, analytic gradient formulas, the
reference-solver validation plan and the FSSH benchmark roadmap.

---

## 1. QDM-Specific Design Considerations

### 1.1 Pure Python Analytic Models, No External Program

Unlike every other shipped skill, QDM does not wrap an electronic structure
program. The methods are **analytic two-state model Hamiltonians** (Tully
1990 models 1/2/3) evaluated with numpy only:

- No input file generation, no output parsing — properties are computed in memory.
- `env_check` only verifies the Python interpreter and numpy.
- Slurm execution still works: the skill agent is submitted as a Slurm job and
  runs the same Python code on the cluster node.

### 1.2 One-Dimensional Model Coordinate via a Dummy Atom

The Tully models are one-dimensional. To stay compatible with the framework
structure interface (xyz startup check, per-atom gradient shape), QDM uses a
**single dummy atom** `X`:

- The model coordinate is `x = coordinates[0][0] × ANGSTROM_TO_BOHR`
  (framework stores Angstrom; the conversion constant is self-contained in
  `service/utils.py` — the skill never imports framework modules).
- y/z of the dummy atom do not enter the model.
- Gradients and NAC vectors have shape `(n, 1, 3)` with only the x component
  of the first atom nonzero, matching the framework Properties contract.

Example structure file (`tully1.xyz`), x = −15 Bohr expressed in Angstrom:

```
1
Tully 1
X  -7.937658  0.0  0.0
```

### 1.3 Velocity Startup and Initial Momentum

The reference solver needs an initial nuclear momentum `k0`. QDM reuses the
framework `{title}.velo` mechanism (per-atom velocity rows in Bohr/au, read
by `read_coord` and synced into the skill through `SYNC_SYSTEM_FIELDS`):

```
k0 = model_mass × velocities[0][0]
```

When no velocity is provided, the schema default `wavepacket_k0` is used.

### 1.4 State Count and FSSH Scope

`nstate` is fixed to **2** (normalize rejects other values). The framework's FSSH /
surface-hopping path and the benchmark comparison against the quantum reference both
consume this skill today: `qualification/benchmark/fssh.py` runs the same model through
`SingleShot` with `qdm_project: 'reference'` and forwards its parameter overrides into
the `&qdm` section, while `workflow/md.py` and `workflow/integrator/fssh.py` are the
hopping side. What stays out of scope is a QDM-internal propagator - the skill supplies
per-step properties and the reference, the framework does the dynamics.

### 1.5 One Interpreter, Several Terms (#302)

QDM runs in-process on numpy, so a thread pool of two terms shares the
interpreter. The one thing it used to touch process-wide was
`os.environ['OMP_NUM_THREADS']`, written by every term even though OpenMP reads
that variable once, at the first parallel region: a later write cannot retune
what is already latched, it only changes what the next term reads. The local
runner now sets it only when the process has not chosen a value yet, and prints
a line to stderr naming both numbers when a term asks for a thread count the
process will not use. `thread` is therefore a per-process setting.

The single-point files already live in each term's `calcdir` (Section 5.1), so
no artifact is shared. `pyrai3md/skills/qdm_skill/tests/test_concurrency.py`
runs two terms through one pool and compares them with the serial numbers.

---

## 2. Methods and Profiles

### 2.1 Methods

| Method | Model | Standard parameters (au) |
|--------|-------|--------------------------|
| `tully1` | V11 = A·sign(x)(1−e^(−B\|x\|)), V12 = C·e^(−Dx²) | A=0.01, B=1.6, C=0.005, D=1.0 |
| `tully2` | V11 = 0, V22 = −A·e^(−Bx²)+E, V12 = C·e^(−Dx²) | A=0.1, B=0.28, C=0.015, D=0.06, E=0.05 |
| `tully3` | V11 = A, V12 = B·e^(Cx) (x<0) / B(2−e^(−Cx)) (x≥0) | A=0.0006, B=0.1, C=0.9 |
| `lz1` | V11 = A·x, V22 = −A·x, V12 = B (constant) | A=0.04, B=0.02 |
| `lz2` | V11 = 0, V22 = −A·x, V12 = B (constant) | A=0.04, B=0.02 |
| `lz3` | V11 = A·x, V22 = 0, V12 = B (constant) | A=0.04, B=0.02 |

Parameters can be overridden individually via `tully_a/b/c/d/e`
(Tully set) and `lz_a/lz_b` (lz1/lz2/lz3); `None` keeps the standard
value. Entries not declared by a model are silently ignored by
`get_model` (e.g. `tully_e` for tully1/tully3, which declare 4 and 3
parameters respectively, or `lz_a` for the Tully set),
since the config schema forwards every key uniformly for every method; a
`None` override is skipped before any name check. The lz family also owns
the analytic single-passage Landau-Zener physics (`crossing_velocity` /
`landau_zener_probability` / `adiabatic_gradients`), which the GSH and
FSSH lz benchmark verdicts consume. The slope difference at the crossing
is 2A for lz1 and A for the half-linear lz2/lz3, so the Landau-Zener
exponent doubles at equal crossing velocity: `P = exp(−πB²/(Av))` (lz1)
vs `P = exp(−2πB²/(Av))` (lz2/lz3).

### 2.2 Profile List (24 profiles)

| Profile | Capability | Outputs |
|---------|-----------|---------|
| `tully1_energy` … `tully3_energy`, `lz1_energy` … `lz3_energy` | energy | E |
| `tully1_gradient` … `tully3_gradient`, `lz1_gradient` … `lz3_gradient` | gradient | E + G (no NAC) |
| `tully1_nac` … `tully3_nac`, `lz1_nac` … `lz3_nac` | nac | E + G + NAC (AIMD superset) |
| `tully1_reference` … `tully3_reference`, `lz1_reference` … `lz3_reference` | md | full split-operator quantum dynamics |

`resolve_profile(method, jobtype, driver='main')` canonicalizes capability
token strings (order- and duplicate-insensitive, same convention as orca):

| jobtype | driver | profile suffix |
|---------|--------|----------------|
| `energy` | any | `_energy` |
| `energy gradient` | any | `_gradient` |
| any combination containing `nac` | any | `_nac` |
| `md` | `main` | `_nac` |
| `md` | `skill` | `_reference` |

---

## 3. Driver Semantics (control.driver)

- **driver=`main`**: the main framework performs the dynamics. The AIMD
  workflow calls the QDM backend per MD step through the `{method}_nac`
  superset profile (E/G/NAC); MD integration (verlet, future FSSH) stays in
  the framework. Profiles without NAC (`{method}_gradient`) exist for
  planning scenarios whose `&md nactype` does not need backend NAC vectors.
- **driver=`skill`**: the skill performs quantum dynamics internally.
  `resolve_workflow` maps driver≠main to SingleShot, so a single backend call
  runs the complete split-operator propagation (`{method}_reference`) and
  returns the final summary.

The skill never reads framework control fields; `driver` arrives as an
argument of `resolve_profile` from the planning layer or explicit callers.

---

## 4. Parameters

Schema defaults (`get_defaults()`); override via the classic input skill
section (`&QDM`), the agent JSON, or the test YAML `input` block.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` / `profile` | `tully1` / `tully1_energy` | model and calculation profile |
| `model_mass` | 2000.0 | nuclear mass in au (named `model_mass` because `mass` is the framework atomic-mass field) |
| `tully_a/b/c/d` | `None` | per-model parameter override; `None` keeps standard values |
| `target_states` | `[0, 1]` | states to report |
| `target_nac_pair` | `[[]]` | NAC pairs in reporting order; empty → the model's own pair (`[[0, 1]]`, `schema/normalize.py:145`) |
| `ref_ngrid` | 2048 | reference grid points |
| `ref_xmin` / `ref_xmax` | `None` | grid bounds in Bohr; `None` keeps the fixed default window [-30, +80] Bohr (`REF_XMIN_DEFAULT`/`REF_XMAX_DEFAULT`), x0 outside the window raises an error |
| `ref_dt` | 20.0 | time step in au |
| `ref_nsteps` | 500 | number of propagation steps |
| `ref_save_every` | 1 | snapshot stride |
| `wavepacket_sigma` | 0.5 | Gaussian width in Bohr |
| `wavepacket_k0` | 10.0 | fallback initial momentum (used when no velocity given) |
| `qdm_project` | `QDM` | output file prefix |
| `execution_mode` | `local` | `local` or `slurm` |
| `slurm_time/partition/ntasks/memory/command` | pyscf-aligned | slurm submission settings |

The framework no longer fills an absent `coupling` with adjacent state pairs
(#447): a pair the input does not declare is shielded, and the qualification
converter only renders a declaration that exists. What stays skill-side is the
model's own channel layout — the two-state Tully models couple one pair, so
`schema/normalize.py:145` fixes `nac_pair_map = [[0, 1]]` for them the way the
nn_pes dataset names its neighbouring-state channels; that is a statement about
the trained model, not a default for the framework.

---

## 5. Single-Point Calculator

`service/calculator.py` diagonalizes the diabatic Hamiltonian (numpy `eigh`,
eigenvectors as columns) and applies the Hellmann-Feynman relations:

```
dE_i/dx = <i| dH/dx |i>
d_ij    = <i| d/dx |j> = <i| dH/dx |j> / (E_j − E_i)
```

The reported NAC vector stores `d_ij · (E_j − E_i)` (equivalently
`<i| dH/dx |j>`), the framework-wide pyrai3md CI-derivative data
convention shared with the molcas and bagel skills: the FSSH
integrator builds the coupling matrix D by dividing the stored
vector by `(E_j − E_i)` again, recovering the standard derivative
coupling.

- **Gap guard**: when `|E_j − E_i| < NAC_ENERGY_GAP_EPS` (skill-local
  constant, 1e-12 Hartree), the NAC component is set to zero and a
  warning is emitted: at a degenerate crossing the adiabatic
  eigenvectors lose their meaning. The framework keeps a numerically
  identical constant in `pyrai3md/utilities/units.py` for future surface
  hopping code; the two are deliberately independent (no cross-import).
- **Benchmark point**: Tully1 at x = 0 gives `<0| dH/dx |1>` = −A·B =
  **−0.016** (d_01 = −A·B/(2C) = −1.6, gap E_1−E_0 = 2C; locked by
  `tests/test_calculator.py`).

### 5.1 Single-point output files (written into calcdir)

The energy/gradient/nac profiles persist their property arrays as plain
text files (`np.savetxt`, units in the header) so the skill output can be
cross-checked against the framework run log:

| File | Content |
|------|---------|
| `{project}_energy.dat` | electronic energies (Hartree), one state per row |
| `{project}_gradient.dat` | gradients (Hartree/Bohr), one row per state, columns = atoms × xyz |
| `{project}_nac.dat` | NAC vectors (Hartree/Bohr), one row per pair, columns = atoms × xyz (nac profile only) |

The results dict returned to the framework is unchanged; the files are
written by `write_single_point_files` in `service/runner/local_runner.py`
(local and slurm modes alike, since the cluster side runs the same local
runner).

---

### 5.2 What survives a run

`keep_tmp` is read at the end of `calculate()`: 0 removes the calculation
directory once the results are collected, 1 keeps the `.dat` dumps written
into it (#307). The per-call point is safe for this skill because every
writer recreates the directory (`service/runner/local_runner.py`,
`service/reference.py`, all `np.savetxt` with a fresh `os.makedirs`) and
nothing reads a previous step's file back; a stepped workflow simply rebuilds
the directory, which `tests/test_keep_tmp.py` measures by running one backend
twice. The lifecycle contract across skills is Section 5.5 of
`dev/SKILL_APP_DEVELOPMENT.md`.

## 6. Reference Solver (split-operator)

`service/reference.py` propagates a two-component wavepacket on the model
potential surfaces with Strang splitting (au, ℏ = 1):

```
U(dt) ≈ exp(−iT dt/2) · exp(−iV dt) · exp(−iT dt/2)
```

- Kinetic half-step in momentum space: `psi_k *= exp(−i k² dt / (4M))`,
  k from `2π·fftfreq(N, dx)`.
- Potential step grid-point wise: `exp(−iV dt) = U diag(e^{−iE dt}) Uᵀ`
  from batched `eigh` over all grid points (einsum).
- Initial packet: Gaussian `N·exp(−(x−x0)²/(4σ²) + ik0(x−x0))` on the
  `current_state` component, renormalized on the discrete grid; x0 comes from
  the model coordinate, k0 from velocity (Section 1.3) or `wavepacket_k0`.
- The FFT kinetic step implies periodic boundaries: the fixed default
  window [-30, +80] Bohr covers the benchmark momenta k0 ≤ 30 over
  T = 6000 au from x0 = -20, and every snapshot checks the edge
  probability mass, warning once when the packet touches the grid
  boundary (wraparound would corrupt the populations).

### 6.1 Output files (written into calcdir)

| File | Content |
|------|---------|
| `{project}_population.dat` | columns: t(au), population per state |
| `{project}_norm.dat` | columns: t(au), total norm |
| `{project}_grid.dat` | coordinate grid (Bohr) |
| `{project}_density.npy` | density snapshots (nsnap, ngrid, 2) |

The results dict returned to the framework carries the contract keys
(`fin_coord`/`fin_velocity` stay empty) plus the summary: `final_population`,
`final_norm`, `nsteps`, `dt`, `ngrid`, `xmin`, `xmax`, `x0`, `k0`,
`wavepacket_sigma`, `initial_state` and the four file paths.

The framework run log prints the final populations in an `&state population`
section (`pyrai3md/utilities/print_results.py`, `population` key with
`final_population` fallback); single-point jobs without population data omit
the section.

---

## 7. Execution Modes

- **local**: in-process execution through `run_local`; the backend creates
  the calculation directory under `scratch_path` and dispatches to the
  calculator or the reference solver by profile suffix.
- **slurm**: `run_slurm` serializes the config to `{project}.json` in the
  calcdir, renders the sbatch script (`service/templates/run_templates.py`,
  PYTHONPATH set to the package import root derived from the runner location)
  and submits with blocking `sbatch -W`. The cluster-side agent
  (`python -m pyrai3md.skills.qdm_skill.agent {project}.json`) writes
  `{project}_results.json` next to the config; the backend collects it after
  the submission returns. Reference trajectory files land in the same calcdir.

Standalone invocation:

```
python -m pyrai3md.skills.qdm_skill.agent <config.json>
```

---

## 8. Testing

- Skill tests (qualification auto-discovery): `tests/tully1_energy.yaml`,
  `tully1_gradient.yaml`, `tully1_nac.yaml`, `tully2_gradient.yaml`,
  `tully3_gradient.yaml`, `tully1_reference.yaml`. They use the single dummy
  atom `X` with Angstrom coordinates equal to the Bohr values divided by
  0.529… (e.g. −15 Bohr → −7.9376581 Å); the reference test inlines
  `velocities` to verify the k0 transfer.
- Unit tests: `test_models.py` (central-difference dH/dx, kink limits,
  parameter override), `test_calculator.py` (NAC benchmark −1.6, gradient vs
  finite difference, gap guard), `test_reference.py` (norm conservation
  <1e-6, population sum = 1, dt-halving convergence, velocity → k0),
  `test_query.py` (manifest interfaces, resolve_profile mapping),
  `test_slurm.py` (mocked sbatch: serialization / collection roundtrip).
- Note: the qualification harness forces `driver='skill'` for skill test
  YAMLs, so the YAMLs pin the profile explicitly; the driver=main chain is
  exercised through framework-level examples instead.

---

## 9. Known Limitations

- No surface hopping inside the skill: the hopping decision is the framework's
  (`workflow/integrator/fssh.py`), which consumes these `*_nac` profiles and the shared
  `utilities/units.py:128 NAC_ENERGY_GAP_EPS`.
- `nstate` is fixed to 2; extending to more states requires new models.
- The dummy atom `X` carries no physical mass in the framework tables; the
  nuclear mass used by QDM is `model_mass`.

---

## Appendix A. Design Origin and Roadmap

Merged from the former `QDM_DEVELOPMENT.md` (analytic model skill design).

### A.1 Design Goals and First-Version Scope

The QDM skill provides analytic nonadiabatic model Hamiltonians to serve:

- FSSH functionality development and robustness testing;
- Calculator energy / gradient / NAC verification;
- FSSH single-trajectory and ensemble population benchmarks;
- an independent full quantum dynamics reference.

The first version pursues quality over model count: a unified
analytic-model interface + Tully 1/2/3 + the Landau-Zener crossing
family (lz1/lz2/lz3, the GSH and FSSH calibration targets) + an
independent quantum reference solver. Demkov-Osherov, Spin-Boson,
Marcus and multi-dimensional models are deferred. The skill
deliberately introduces
no independent velocity schema, molecular schema, or FSSH implementation.

### A.2 Layer Responsibilities

```text
Model            defines the physical (diabatic) Hamiltonian
Calculator       converts H(x), dH/dx into energy / gradient / NAC
Reference Solver independent quantum nuclear dynamics (shares only the
                 Hamiltonian definition with the models, never the propagator)
PyRAI3MD Core    molecular state, coordinates and velocities
FSSH Workflow    classical nuclear dynamics, electronic propagation, hopping
Benchmark        compares FSSH populations against the quantum reference
```

### A.3 Tully 1 Analytic Gradient

With V11(x) = A·sign(x)·(1 − e^(−B|x|)) and V12(x) = C·e^(−Dx²), for x ≠ 0:

```
dH/dx = [ dV11/dx   dV12/dx ]
        [ dV12/dx  −dV11/dx ]

dV11/dx = A·B·e^(−B|x|)
dV12/dx = −2·C·D·x·e^(−Dx²)
```

The cusp at x = 0 is handled per the model definition; the kink limits are
verified against central differences by `tests/test_models.py`.

### A.4 Reference Solver Self-Validation

The reference solver must pass its own convergence tests before it can serve
as a benchmark reference:

- **Time step convergence**: compare dt, dt/2, dt/4 populations (dt-halving
  convergence is locked by `tests/test_reference.py`);
- **Spatial grid convergence**: compare N, 2N, 4N grid points;
- **Box size convergence**: compare L, 2L to exclude boundary reflection of
  the wavepacket.

Reported reference quantities: electronic populations Pᵢ(t), total norm
(|norm − 1| ≪ 1 required) and spatial densities ρᵢ(x,t) for visualization,
transmission/reflection analysis and debugging.

### A.5 FSSH Benchmark Plan (Future Task)

The benchmark combines the analytic calculator (FSSH production) and the
reference solver (exact quantum dynamics); the two propagations stay fully
independent and are combined only at the comparison step.

Comparison quantities: population curves P_FSSH(t) vs P_QM(t), final
populations, transmission and reflection probabilities, population
normalization (Σᵢ Pᵢ = 1 on both sides, small numerical error allowed).

Principle: **FSSH ≠ exact quantum dynamics**, so P_FSSH = P_QM is not a
hard regression condition; the reference provides a physical benchmark,
not an expected FSSH output. FSSH testing additionally covers timestep
convergence, trajectory-number convergence, energy conservation, hopping
probability, state switching, momentum rescaling, frustrated hopping and
population behavior.

### A.6 Test Hierarchy

```text
Level 1  Model test            H(x), dH/dx
Level 2  Calculator test       energy, gradient, NAC
Level 3  Single trajectory     verlet, electronic propagation, hopping,
                               switching, momentum rescaling, frustrated
                               hopping, energy conservation
Level 4  Ensemble              multiple trajectories, population,
                               statistical convergence
Level 5  Quantum benchmark     FSSH vs exact population, transmission,
                               reflection
```

Levels 1/2 are locked by the current unit tests; Levels 3–5 arrive with the
FSSH / benchmark delivery.

### A.7 Extension Roadmap

Future models (Demkov-Osherov, Spin-Boson, Marcus,
multi-dimensional models) add new model modules behind the same
`hamiltonian(x)` / `hamiltonian_gradient(x)` interface; no new velocity or
molecular schema is planned.
