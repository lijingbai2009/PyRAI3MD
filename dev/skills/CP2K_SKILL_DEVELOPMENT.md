# CP2K Skill Development Guide

Last updated: 2026-11-12 19:20 (1.6 records the requested-block raise for stress and energy; the zero baseline is unrequested-only. Earlier: 2026-10-22 21:20 (section 6.5's gap-term boundary now points at `CELL_OPT_DEVELOPMENT.md` section 6.3, where that design moved with the rest of the cell half). Before it, 18:00 (section 6.6: the excited-state numerical-stress design record merged in from the retired `dev/skills/cp2k_num_stress.md`, with its acceptance checklist and the test reading of item 2; the code comments that cited that file's sections 4/6/8 now cite 6.6). Before it, 2026-10-19 12:10 (new section 12: the `&VIBRATIONAL` design - capability word, profile names, input skeleton and the four questions a first real log has to answer - deliberately WITHOUT parser, profile registration, `resolve_profile` branch or any number, because no CP2K vibrational output exists in this repository and cp2k is not on this environment. Landed instead: `data/core.py::MATRIX_CAPABILITIES` refuses `hess_driver: skill` on a skill whose manifest declares no matrix, which names cp2k today (its capabilities are energy, gradient, stress, transition_dipole, geometry_optimization, abinitio_dynamics), pinned by `test_core.py::TestCurvatureProducerScope`. The 2026-10-15 00:20 pass stands for what it checked: section 5.2 rewritten against the deployment A/B on `RKS_TRIPLETS` and CHANGELOG #350: the two-surface request is planned as two runs by `service/jobs.py`, merged by row, with `&SOC` re-classified as a separate capability; the rest of the file was not re-audited line by line)

This document covers **CP2K-specific** implementation details for cp2k_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. CP2K-Specific Design Considerations

cp2k_skill wraps the **CP2K Quickstep** engine for ground-state DFT and
TDDFPT excited-state calculations, covering single points, analytic forces,
geometry optimization and ab initio molecular dynamics. It registers as the
`qm` backend (`qm: Cp2kBackend` in `manifest.yaml`) with the capabilities
`energy`, `gradient`, `transition_dipole`, `geometry_optimization` and
`abinitio_dynamics`.

Background: the skill was designed up front around a few deliberate
contracts that the implementation preserves — cutoffs are mapped explicitly
to `&MGRID` rather than hidden in presets; basis sets are configured on two
levels (global data files + element-level kind blocks); the dispersion
`REFERENCE_FUNCTIONAL` always follows the active functional; the
gradient = −force sign flip is concentrated in the parser instead of being
scattered across the stack; and the TDDFPT `NSTATES` (how many roots CP2K
computes) is kept separate from the active `target_states` (which roots the
framework consumes).

### 1.1 Section-Based Input Model

CP2K input is a nested section tree (`&GLOBAL`, `&FORCE_EVAL`, `&SUBSYS`,
`&DFT`, `&MOTION`, ...). The builder is **compositional**: one private
function per section (`_build_global`, `_build_force_eval`, `_build_subsys`,
`_build_cell_block`, `_build_kind_blocks`, `_build_dft`, `_build_poisson`,
`_build_admm`, `_build_xc`, `_build_hf`, `_build_vdw`, `_build_scf`,
`_build_tddfpt`, `_build_print`, `_build_motion`, `_build_geo_opt`,
`_build_md`), assembled by `write_input_file` according to the compiled
workflow steps of the profile.

### 1.2 Functional Support: pbe and pbe0 Only

`AVAILABLE_FUNCTIONAL = ['pbe', 'pbe0']`. Any other value raises
`Unsupported CP2K functional: {functional}` during validation. PBE0 is not
delegated to CP2K's XC functional registry; it is assembled by hand:

```
&XC
  &XC_FUNCTIONAL
    &PBE
      SCALE_X 0.75
      SCALE_C 1.0
    &END PBE
  &END XC_FUNCTIONAL
  &HF
    FRACTION 0.25
    &SCREENING
      EPS_SCHWARZ 1E-6
    &END SCREENING
    &INTERACTION_POTENTIAL
      TRUNCATED POTENTIAL
      CUTOFF_RADIUS 6.0
    &END INTERACTION_POTENTIAL
    &MEMORY
      MAX_MEMORY 3000
    &END MEMORY
  &END HF
&END XC
```

### 1.3 FFT Robustness Fix

`&GLOBAL` always carries `EXTENDED_FFT_LENGTHS TRUE` (with
`PRINT_LEVEL LOW`). This fixes CP2K's `Index to radix array not found` crash
on arbitrary cell sizes by allowing FFT grids whose prime factorization goes
beyond the default radix set. It is emitted unconditionally for every profile.

### 1.4 Periodic vs Molecular Systems

`_is_periodic` reads **`pbc` only** — it is the single periodicity authority
(CHANGELOG #216, `dev/methods/PBC_DEVELOPMENT.md` §3.2). A non-zero
`lattice_vector` alone no longer makes a run periodic (this previously
misclassified molecular systems whose default cell derived a fake 99 Å
lattice). Only full 3D periodicity is supported in this stage; partial
periodicity (mixed `pbc`) and a periodic run without a valid lattice are
rejected by `schema/validate.py::validate_system` before the builder runs.

- **Periodic** (`any(pbc)`): `&CELL` emits the three lattice vectors as
  `A`/`B`/`C`, `PERIODIC XYZ`, and `&POISSON` uses `PERIODIC XYZ` +
  `PSOLVER PERIODIC` (periodic Poisson solver). The cell comes from the
  per-evaluation `cell=` snapshot applied to `config.system` in
  `calculate()`, so a variable-cell (NPT) step writes its own deformed cell.
  The framework converter flattens the test YAML `lattice_vector` into the
  `&MOLECULE` cell keyword before it reaches the skill (framework adapter
  maps `lattice` → `lattice_vector`). With `dipole_correction: true` the
  `&DFT` block also writes `SURF_DIP .TRUE.` and `SURF_DIP_DIR Z`.
- **Molecular** (`pbc` all zero): an automatic vacuum box is built around the
  molecule (10 Å padding per side, minimum box edge 20 Å), with
  `PERIODIC NONE` and Poisson `PSOLVER WAVELET`.

### 1.5 Coordinates via xyz File

Coordinates are never inlined in the input. The builder writes
`{project}.xyz` into the calcdir (`_write_xyz_file`) and `&TOPOLOGY`
references it with `COORD_FILE_NAME`. Between dynamics steps only the xyz
file needs rewriting.

### 1.6 Failure Propagation

The run script exits with the CP2K return code; `backend.calculate()` raises
a `RuntimeError` on any non-zero code before parsing, so a crashed CP2K
module is never masked by zero-filled results.

A requested block that is absent is a failure, not a zero: a run that asked
for `stress` raises when no stress block exists, and a run whose workflow asks
for energy raises when the log has no `Total energy:` line. The zero baseline
is kept only for a property the run did not request; TDDFT excited-state rows
then reference 0.0 as before.

---

## 2. Calculation Profiles

Nine profiles over two methods (`schema/method.py` is the single source of
truth; the legacy `sf_tddft` method is intentionally not supported):

| Profile | Method | Input steps | Description |
|---|---|---|---|
| dft_energy | dft | energy | Ground-state single point |
| dft_gradient | dft | energy, gradient | Ground-state forces |
| dft_opt_min | dft | optimize | BFGS geometry optimization |
| dft_md | dft | md | Standalone `&MOTION` MD (driver=skill) |
| tddft_energy | tddft | energy, tddft | TDDFPT excitation energies |
| tddft_gradient | tddft | energy, tddft, gradient | State-resolved excited-state forces |
| tddft_transition_dipole | tddft | energy, tddft, transition_dipole | Transition dipoles + oscillator strengths |
| tddft_opt_min | tddft | tddft, optimize | Excited-state optimization (`&OPTIMIZE STATE n`) |
| tddft_md | tddft | tddft, md | Standalone excited-state MD (driver=skill) |

`resolve_profile(method, jobtype)` derives the profile from method + jobtype
for classic input flows (jobtype tokens are order-independent; `md` maps to
the `_gradient` suffix, see section 5.3).

---

## 3. Schema Details

### 3.1 Default Parameters (`schema/reference.py` DEFAULT)

| Parameter | Default | Meaning |
|---|---|---|
| cutoff | 400.0 | Plane-wave cutoff (Ry) → `&MGRID CUTOFF` |
| ref_cutoff | 50.0 | Relative cutoff → `&MGRID REL_CUTOFF` |
| functional | pbe | Only `pbe` / `pbe0` supported |
| basis_file | ['BASIS_MOLOPT'] | Global basis data file list |
| potential_file | ['POTENTIAL'] | Global pseudopotential data file list |
| auxiliary_basis_file | [] | ADMM aux basis files; empty disables ADMM |
| dispersion | '' | '' / 'none' off; `d3`, `d3bj` enable `&VDW_POTENTIAL` |
| nproc, cp2k_nproc | 1 | MPI ranks (`mpirun -n`); the plain name is what a standalone input writes, `cp2k_nproc` what the framework forwards from `&cp2k`, and the larger of a pair wins (#306, which retires the old `mpi` rank keyword) |
| thread, cp2k_thread | 1 | OpenMP threads per rank (`OMP_NUM_THREADS`) |
| cp2k_path | '' | CP2K installation root (`$CP2K_HOME`) |
| cp2k_exe | '' | Explicit executable; default `$CP2K_HOME/exe/local/cp2k.psmp` |
| setup_script | '' | Optional shell script sourced before launch |
| md_nstep | 100 | MD steps |
| md_timestep | 0.5 | MD timestep (fs) |
| md_ensemble | NVT | Supported ensemble (NPT not implemented) |
| scf_maxit | 200 | `MAX_SCF` |
| scf_converge | 1e-6 | `EPS_SCF` |
| tddft_maxiter | 100 | TDDFPT `MAX_ITER` |
| tddft_converge | 1e-5 | TDDFPT `CONVERGENCE` |
| tddft_dipole_form | length | `&DIPOLE_MOMENTS DIPOLE_FORM` |
| opt_maxit | 100 | GEO_OPT `MAX_ITER` |
| stress | False | Whole-system stress request (orthogonal to the profile) |
| stress_method | analytical | `analytical` (engine tensor) or `num` (skill-driven strain finite difference), see section 6.5 |
| stress_delta | 1e-3 | Dimensionless strain step of the `num` route; validate keeps it in (0, 1e-2] |
| stress_scope | full | `full` = six strain directions (12 points), `iso` = uniform dilation (2 points) |
| dipole_correction | False | `&DFT SURF_DIP .TRUE.` + `SURF_DIP_DIR Z`; requires a periodic z-normal slab cell |

Element defaults used by normalize: `DEFAULT_BASIS = 'DZVP-MOLOPT-SR-GTH'`,
`DEFAULT_AUX_BASIS = 'admm-dzp'`, `DEFAULT_POTENTIAL = 'GTH-PBE'`, plus a
valence-electron table that appends the `-q<N>` charge suffix.

### 3.2 Validation Rules (`schema/validate.py`)

- functional must be in `AVAILABLE_FUNCTIONAL`, else
  `Unsupported CP2K functional: {functional}`.
- `cutoff` and `ref_cutoff` must be positive.
- dispersion accepts `''`/`'none'` (section disabled), `d3`, `d3bj`
  (mapped to `DFTD3` / `DFTD3(BJ)`); anything else raises.
- TDDFPT profiles require `tddft_nstates > 0` and
  `current_state <= tddft_nstates`.
- `stress_method` must be one of `STRESS_METHODS`
  (`analytical`/`num`); a misspelling would otherwise fall
  through `_stress_keyword` and emit no CP2K keyword at all, leaving a
  zero-filled tensor that reads like a result.
- `stress` with the route that asks the engine for a tensor
  (`analytical`) is refused under `method: tddft`: CP2K computes the TDDFT
  forces and then aborts in `response_solver.F:943` with "Stress Tensor
  not available for TDDFT calculations" (measured in the test environment, two
  rounds of the shipped case). The refusal is per route, not per request:
  `stress_method: num` sends CP2K no stress keyword and is the route that
  may carry an excited state (section 6.5). **CP2K offers no numerical
  stress for a TDDFT run either (measured)**, so the engine-side
  `STRESS_TENSOR NUMERICAL` spelling is not part of the vocabulary at all
  -- `num1` is retired, and an input that still asks for it is refused by
  name rather than reinterpreted. For DFT the analytical tensor is the
  route to use; the numerical one exists there only as the verification
  carrier of section 6.5. Ground-state stress is certified by
  `tests/dft_stress.yaml`, the periodic TDDFT gradient by
  `tests/tddft_gradient_water.yaml`.
- `num` additionally requires full 3D periodicity and a non-singular
  `lattice_vector` (the route deforms the cell), a `stress_scope` in
  `STRESS_SCOPES`, and `0 < stress_delta <= 1e-2`. A written `stress_delta: 0`
  is the schema's unset marker, like `cutoff: 0`, and takes the default.
- `dipole_correction` requires a periodic, z-normal slab cell
  (`a_z=b_z=c_x=c_y=0`) and refuses `stress_method: num` with
  `stress_scope: full`: the shear strain points tilt the cell, which
  CP2K's correction aborts on (`surface_dipole.F`). `iso` keeps the cell
  diagonal and is allowed.
- `remotetask` is cross-checked against `execution_mode` instead of
  folded into it: setting `remotetask` while the arm is `local` raises,
  because silently submitting jobs the input did not ask for (or ignoring
  the request) would both oversubscribe the cluster and make the reported
  arm untrue. The three keywords are tri-state (`None` = undeclared =
  serial), because the framework's merged skill dict does not carry them
  unless the user wrote them and a strict `bool`/`int` field would reject
  a `None`; an explicitly written `multipool` below 1 is refused, while a
  written `0` is not the unset marker here (the schema default is `None`,
  so the empty-fill step leaves the value for the validator to see).

---

## 4. Normalize Workflow

- **Two-level basis fill** (`verify_basis`): element-level dicts are
  completed from the deduplicated `atoms` list — each element receives
  `DZVP-MOLOPT-SR-GTH-q<N>` / `GTH-PBE` (and `admm-dzp-q<N>` when ADMM is
  active), where `<N>` comes from the valence-electron table. Explicit user
  values always win. The global level stays the data-file lists
  (`BASIS_MOLOPT`, `POTENTIAL`).
- **tddft_nstates derivation** (`verify_state_num`): when not explicitly set,
  `tddft_nstates = max(nstate - 1, max(target_states))`; an explicit positive
  value is kept. This keeps CP2K's computed root count independent of the
  framework's active-state selection.

---

## 5. Input Builder Details

### 5.1 &GLOBAL and RUN_TYPE

`&GLOBAL` sets `PROJECT`, `RUN_TYPE` and `PRINT_LEVEL LOW` plus the
`EXTENDED_FFT_LENGTHS TRUE` fix (section 1.3). RUN_TYPE mapping:
energy → `ENERGY`, gradient → `ENERGY_FORCE`, optimize → `GEO_OPT`,
md → `MD`.

### 5.2 Core Sections

- `&MGRID`: `CUTOFF {cutoff}`, `REL_CUTOFF {ref_cutoff}`.
- `&QS`: `EPS_DEFAULT 1.0E-12`; MD profiles add `EXTRAPOLATION ASPC`.
- `&SCF`: `MAX_SCF`, `EPS_SCF`, `SCF_GUESS ATOMIC`, and the `&OT` minimizer
  `DIIS` with preconditioner `FULL_SINGLE_INVERSE`.
- Dispersion: `&VDW_POTENTIAL` with the mapped type and
  `REFERENCE_FUNCTIONAL {functional.upper()}` — the reference functional is
  forced to follow the active functional (PBE0+d3 keeps the `PBE0`
  reference).
- ADMM (non-empty `auxiliary_basis_file`):
  `&AUXILIARY_DENSITY_MATRIX_METHOD METHOD BASIS_PROJECTION`, with
  `ADMM_PURIFICATION_METHOD NONE` added for TDDFPT, and each `&KIND` block
  carrying `BASIS_SET AUX_FIT`.
- TDDFPT: `&PROPERTIES &TDDFPT` is emitted as a **sibling of `&DFT`
  inside `&FORCE_EVAL`** (CP2K rejects it nested in `&DFT`: `unknown
  subsection PROPERTIES of section DFT`): `NSTATES {tddft_nstates}`,
  `RKS_TRIPLETS {T|F}`, `CONVERGENCE`, `MAX_ITER`, `MIN_AMPLITUDE 0.01`,
  and `&DIPOLE_MOMENTS DIPOLE_FORM {tddft_dipole_form}`.
- `NSTATES` on the shared request is the larger of the two manifold counts
  (through `nsing_root`, excited singlet roots with the ground row excluded, and
  `ntrip_root`, both already normalized) because that is the most any one run
  needs. When `service/jobs.py` splits the request, each run rewrites it to the
  count of its OWN manifold - `NSTATES` is per run, and a run holding only
  triplets has no use for the singlet roots - except a value bigger than every
  declared root put together, which can only be a deliberate override and travels
  to both runs. An all-singlet request is never split, so it keeps `RKS_TRIPLETS F`
  and the same `NSTATES` as before, which is what holds the shipped cases
  byte-identical.
- **`RKS_TRIPLETS` replaces the singlet solve, it does not add a triplet one.**
  Measured twice in the test environment round `tmp_20261014_185533`: the job that wrote
  `NSTATES 2` with `RKS_TRIPLETS T` and no `&SOC` printed ONE table, headed
  `R-TDDFPT states of multiplicity 3` (`tddft_st_energy_test.log:263`), with no
  singlet block at all; and an A/B that took the same input and changed only that
  letter printed `multiplicity 3` from `T` and `multiplicity 1` from `F`, with the
  word `singlet` appearing zero times in the `T` log. The engine source says the
  same thing (`qs_tddfpt2_methods.F`: `mult` comes from that single boolean at
  294-297, one `tddfpt_energies` call at 421, one printed table at 1190). The
  reference input that shows BOTH tables (`cp2k_demo/sp/bulk.log:276` and `:1583`)
  carries `&SOC` inside `&TDDFPT` (`sp/bulk.inp:144`), and the two-manifold solve
  exists only inside `tddfpt_soc_energies` (its `! First multiplicity` /
  `! Second multiplicity` blocks) because the spin-orbit matrix needs both - which
  is a different task, and one that still hands back forces for a single manifold
  (line 1255: "only the values of one multiplicity will be passed back for force
  calculations"). `&SOC` is therefore NOT how this skill gets two surfaces.
- **A request naming both singlet and triplet excited states becomes TWO runs of
  one reference**, planned by `service/jobs.py`, which is the only place that
  answers "how many CP2K runs does this evaluation need". Each run carries its own
  `PROJECT` (`<project>_S`, `<project>_T`, which is what keeps the two sets of
  `.log`, `-force-1_0.xyz`, `-td_forces-1_0.tdfrc` and `-RESTART.wfn` apart inside
  one calculation directory), its own `RKS_TRIPLETS` letter, and its own `NSTATES`
  = the roots of its own manifold. The other manifold's root count is zeroed on
  the variant, and two existing consumers then follow without a new keyword: the
  builder's letter reads `ntrip_root`, and `state_map.job_manifold` reads both
  counts to restrict `&PRINT &FORCES LIST` and the rows the parse is held to
  answer. A request that names one manifold gets ONE job carrying the caller's own
  config object, so every shipped case is the identity arm for this branch, input
  bytes included.
- The merge (`output_parser.merge_job_results`) takes each row of a row-keyed
  channel (`energy`, `excitation`, `gradient`, `dipole`, `stress_tensor`) from the
  run that solved its manifold, and it never picks a winner for the ground state:
  both runs compute it, so the readings must agree at the printed precision
  (`1.0e-8` Ha) or the merge refuses - the A/B reproduced
  `Total energy: -13.56482679723119` in both logs, which is what makes one axis
  out of two runs safe. Pair-keyed channels (`transition_dipole`, `osc_strength`)
  are summed, and a position both runs filled with different numbers means the
  split was not disjoint and is refused. A channel with one owner per run (the
  returned geometry, the convergence verdict, the frequencies) must match too.
  And a table is refused **where the claim is made**: a run planned for 2S=0 that
  finds a multiplicity-3 table in its own log raises, because a merge that only
  took rows by owner would hide exactly that swap.
- **Two-manifold requests are refused for profiles whose channels have one owner
  per run** (`tddft_opt_min`, `tddft_md`): the returned geometry, its cell and the
  convergence verdict come out of a single run's documents, so splitting them
  would leave those channels unowned. Stress never reaches that decision -
  `validate_stress` already refuses both of its methods against a TDDFT run, the
  engine declining an excited-state tensor (measured twice,
  `response_solver.F:943`).
- Still open: the second run recomputes its SCF instead of restarting the first's.
  `WFN_RESTART_FILE_NAME` would skip that work, but the saving is only real if
  CP2K's own log says it read the wavefunction, and that marker has not been read
  off a run yet, so nothing is claimed (see `dev/TODO.md`).
- A non-singlet **reference** is refused by name in `schema/validate.py`: the
  `&SCF` block this skill writes is restricted (no `UKS`), so `MULTIPLICITY 2` or
  `3` would produce an input the engine itself rejects. A triplet **row** under a
  singlet reference is the supported shape (RKS linear response solves the triplet
  manifold of a closed-shell reference), and it needs no unrestricted reference.
  A triplet row under `method: dft` is refused too, since plain DFT solves no
  excited states at all and the row would name nothing.

### 5.3 Print Sections

- Ground-state forces: `&PRINT &FORCES ON`, `force_unit hartree/bohr`,
  `filename force` → `{project}-force-1_0.xyz`. Emitted for **any** run with a
  gradient step, including a TDDFT one: a crossing between a ground state and an
  excited state needs a force on each, and CP2K writes both files in the same
  calculation (`cp2k_demo/td_grad/azo-1.inp:125-129` asks for both and its
  directory holds both products). Row 0 left at zero would satisfy the shape
  contract while describing a surface nobody computed.
- Excited-state forces (tddft_gradient): `&PRINT &FORCES` with
  `LIST {the requested excited states}` and `filename td_forces` →
  `{project}-td_forces-1_0.tdfrc`. The LIST is written by
  `service/parser/state_map.py`, not by the framework row numbers blindly.
  What the keyword accepts was settled by an abort, not by reading: asking for the
  first singlet and the first triplet as `LIST 1 3` made CP2K stop mid-print with
  `State List contains invalid state` / `TDDFPT Print Forces: Invalid State`
  (`qs_tddfpt2_fprint.F:395-396`, round `tmp_20261014_142736`, the run's own
  `tddft_st_forces_test.log`, line 674), so the accepted numbers
  restart inside each manifold. A request spanning two manifolds therefore has no
  number `LIST` can use, and `state_map.require_addressable` refuses it at the
  input rather than buying that abort. Whether a triplet root can be given forces
  at all in CP2K 2025.2 is **open**, and it is what stands between this channel and
  an MECP search: the energies of both manifolds are already available from one run,
  the gradients of both are not.
- Reading the manifolds apart is `state_map`'s job and the reason it exists. The
  excitation tables restart their numbering per spin, so a row is identified by
  the pair (banner multiplicity, number), which the parser gets from the log
  lines themselves - no hypothesis involved. A tdfrc header prints a number and
  an excitation energy but no spin, so a block is tied to a row by BOTH, and the
  two possible countings are tried against the printed energies: whichever
  explains every block is reported as the convention the engine used, and a run
  whose blocks fit neither reading fails by name instead of dropping a surface.
  When two surfaces are degenerate - which is where a crossing search is going -
  neither number nor energy separates them, the tie-break falls to the order the
  roots were requested, and the run warns that it did.
- GEO_OPT/MD trajectories: `&PRINT &TRAJECTORY FORMAT xyz` →
  `{project}-pos-1.xyz`.

### 5.4 &MOTION Sections

- `&GEO_OPT`: `TYPE MINIMIZATION`, `OPTIMIZER BFGS`, `MAX_ITER {opt_maxit}`,
  convergence thresholds `MAX_DR 3E-3`, `RMS_DR 1.5E-3`, `MAX_FORCE 4.5E-4`,
  `RMS_FORCE 3E-4`; TDDFPT with `current_state > 0` adds
  `&OPTIMIZE STATE {current_state}`.
- `&MD`: `ENSEMBLE {md_ensemble}`, `STEPS {md_nstep}`,
  `TIMESTEP {md_timestep}`, `TEMPERATURE {temperature}`; NVT adds a
  `&THERMOSTAT` of `TYPE CSVR` with `TIMECON 100`.

---

## 6. Result Parsing

### 6.1 Output Files

| File | Parsed for |
|---|---|
| `{project}.log` | Total energy, SCF convergence, GEO_OPT completion, TDDFPT excitation table |
| `{project}-force-1_0.xyz` | Ground-state forces (6 columns: index, kind, element, fx, fy, fz) |
| `{project}-td_forces-1_0.tdfrc` | Excited-state force blocks |
| `{project}-pos-1.xyz` | Final geometry (GEO_OPT / MD) |

### 6.2 Log Parsing (`service/parser/log_parser.py`)

- `Total energy:` (last occurrence) → S0 energy in hartree.
- `*** SCF run converged ***` → SCF convergence flag.
- `GEOMETRY OPTIMIZATION COMPLETED` → `opt_converge`.
- `TDDFPT|` table rows → per-state excitation energy (eV), x/y/z and total
  transition dipole components (a.u.) and oscillator strength. State numbers
  in the table are 1-based and match framework state indices directly.
  Excited-state energies are reconstructed as
  `energy[state] = energy[0] + excitation * EV_TO_HARTREE`; `excitation`
  stays in eV. `transition_dipole` / `osc_strength` follow the
  `transition_pair` order (only 0 → b pairs).

### 6.3 gradient = −force, Centralized in the Parser

The sign flip is applied once, in `output_parser.py`, for every channel:

- **Ground state**: forces from `{project}-force-1_0.xyz`
  (`parse_force_xyz`), `results['gradient'][0] = -forces`.
- **Excited states**: force blocks from the tdfrc file are consumed in
  `sorted(target_states)` order (excited states only); each block's forces
  are negated into `results['gradient'][state]`. A block-count mismatch
  raises immediately (**fail fast**) instead of silently returning partial
  gradients.
- Missing force files raise `RuntimeError` (no zero-fill fallback).

### 6.4 fin_coord and fin_cell Channels

GEO_OPT and standalone MD share `_pack_final_geometry`: the **last frame**
of `{project}-pos-1.xyz` is returned as `fin_coord` (Angstrom). An atom-count
mismatch raises `ValueError`; a missing trajectory raises `RuntimeError`.
`fin_velocity` is reserved and always returns `[]` for this skill.

The same frame's `Lattice=` comment, if the run was periodic, is read as
`fin_cell` (nine flattened lattice components, Å): `parse_xyz_cell` walks the
frames the way `parse_xyz` does and keeps the last cell it saw, so the cell and
the coordinates cannot come from different steps. No profile of this skill
emits `CELL_OPT`/`CELL_REF` today, which means the reported cell equals the cell
the framework supplied and the framework leaves the state untouched -- the
channel is wired so a cell optimization needs no new plumbing when one lands,
not because one exists now. A molecular (non-periodic) run prints no lattice,
and the key is then absent rather than filled in from the input.

### 6.5 Whole-System Stress: Three Routes

`config.stress` is orthogonal to the profile: it neither changes the run
type nor replaces the forces. Three evaluation routes share that one
request (`schema/reference.py STRESS_METHODS`):

| Route | What CP2K is asked for | What fills `stress_tensor` | States |
|---|---|---|---|
| analytical | `STRESS_TENSOR ANALYTICAL` + `&PRINT &STRESS_TENSOR ON` | `_pack_stress` reads `{project}_stress_tensor.dat` (log fallback), negates, normalizes to GPa | S0 only |
| num | nothing beyond an energy | `service/stress/num_stress.py` combines the strain single points | every state the run solved |

**The `num` route is skill-driven, not framework-driven.**
`Cp2kBackend.calculate()` parses the central run as usual, then -- and
only when `stress_method == 'num'` -- replaces the whole
`(nstate, 3, 3)` array with the driver's output before the results are
stored, so the framework sees the same channel in the same unit
(manifest `units: stress: gpa`) and its central
`convert_stress_to_atomic` continues to do the single GPa →
Hartree/Bohr³ conversion. No framework batch workflow is involved: the
number of engine evaluations belongs to the stress method, which is a
skill property, and the framework rule that remote single-point
execution is controlled by the skill already puts that decision here.

The driver (`service/stress/`) is three pieces:

- `strain.py` -- the mathematics only: the six direction matrices
  (diagonal `B[a,a]=1`, shear `B[a,b]=B[b,a]=1/2`), the row-vector
  deformation `H± = H @ (I ± delta*B).T`, `R± = R @ (I ± delta*B).T`
  which keeps the fractional coordinates unchanged, the central
  difference `[E(+) - E(-)]/(2*delta*V0)` with the **central** volume,
  and the isotropic contraction `tr(sigma)/3` onto the diagonal. The
  external difference is tensile-positive as it stands, so no negation
  is applied (the parser's negation belongs to CP2K's own print).
  `HARTREE_A3_TO_GPA` is derived from the parser's existing
  `EV/A^3` factor and `EV_TO_HARTREE`, not copied.
- `point_runner.py` -- executes one perturbed point. Module-level and
  taking a plain dict, because the point may run in another process and
  a bound method would drag the parent backend into the payload. It
  rebuilds its own `Cp2kBackend` in its own directory.
- `num_stress.py` -- plans the points, runs them, guards the state
  identity, assembles. Every point gets a distinct `{project}` stem and
  a sibling directory `<parent>_num_<dir><p|m>` (the central directory
  is never reused: all CP2K artefacts of a run are pinned to `{project}`);
  each point's profile is derived by the skill's own planner,
  `resolve_profile(method, 'energy')`, so a ground-state run fans out to
  `dft_energy` and an excited-state run to `tddft_energy` -- energy-only,
  never relaxed, which is what makes the difference an instantaneous
  stress. A point inherits `stress: False` (no keyword) and
  `multitask/multipool` cleared, so it cannot recurse.

**The two dispatch axes, read inside the skill.** `&cp2k` now forwards
its `multitask` / `multipool` / `remotetask` keywords into the skill
schema (they are declared in `Environment`, and the framework merge
already keeps the raw section, so no DEFAULT entry was added -- one would
have written these keys into every generated classic input). `multitask`
selects `mp.get_context('spawn').Pool(processes=min(multipool, n_points))`
over the points; `remotetask` is the request for the slurm arm, whose
actual authority stays `execution_mode` and which is therefore
cross-checked rather than folded. Under the slurm arm every point is a
blocking `sbatch -W` job, so the pool size is the number of concurrent
jobs, and the user's `{project}.slurm` template is copied into each point
directory under the point's own stem (`slurm_runner` now looks inside the
calculation directory first, then the working directory).
Oversubscription is warned, never refused, in the same spirit as
`validate_environment`.

**State identity.** States are treated as the energy-sorted adiabatic
roots of each point. Before any difference is taken the driver requires
that every point returned the same number of roots as the central run,
that a root the central run reported is not missing from a point (the
parser zero-fills a missing row, which would otherwise enter the
difference as an energy), and that no root moved by more than half the
smallest central gap; a degenerate central spectrum is refused outright,
because at a crossing no energy-only criterion can say which state is
which. NTO / transition-density character tracking is out of scope. Each
failure raises rather than emitting a tensor.

**Measured locally, and what is not yet measured.** The strain
mathematics and the whole driver are covered by tests that replace only
the CP2K executable (`tests/test_strain_math.py` 28,
`tests/test_num_stress_driver.py` 25): the engine stand-in evaluates a
closed-form model and the tests compare the assembled tensor against a
hand-derived analytic stress. Along the `full` scope that model is
quadratic in the strain, so the central difference is exact -- measured
maximum relative deviation `1.093e-09` over twelve strain points at
`delta = 1e-3`, the residual coming from the eight-decimal coordinate and
lattice print amplified by the `1/delta` division, not from truncation.
For `iso` the volume enters as `(1+delta)^3` and the truncation is real:
measured relative error `1.8333e-05` at `delta = 1e-3`, with each halving
of the step dividing it by 4 (asserted O(delta^2)). One TDDFPT batch of
twelve points returned tensors for all three states, with the S1
diagonal shifted by `0.224263` GPa against a closed form of `0.224263`
GPa. Serial and pooled runs agree bitwise, and the slurm arm reproduces
the local numbers through a shimmed `sbatch`. **None of this is evidence
about a real CP2K run**: the sign, transpose, volume and shear-factor
conventions still have to be cross-checked against CP2K's own analytical
tensor in the test environment, which is what `tests/dft_stress_num.yaml`
against `tests/dft_stress.yaml` is for (section 6.6, acceptance item 2).

### 6.6 TDDFT/TDA Excited-State Numerical Stress: the Design Record and Its Checklist

The `num` route is general; its excited-state use carries a record of its own - what the
engine supports, what energy is being differenced, and which conventions have to be
validated before the tensor is a physical claim. It was a separate document
(`cp2k_num_stress.md`) until 2026-10-22; it lives here now, and the acceptance checklist
keeps its numbering because comments in `service/stress/` and the certification tests
cite it.

**Scope.** TDDFPT/TDA excited states of a three-dimensional periodic system only.
Delta-SCF, constrained-DFT and SOC-mixed-state methods each define their energy
differently, so nothing here is a support statement for them.

**The built-in route is withdrawn.** `STRESS_TENSOR NUMERICAL` was once designed as a
second route: CP2K's `force_env_calc_num_pressure` perturbs the cell at fixed fractional
coordinates, and an excited-state potential adds `Omega_I` to the energy, which *looked*
like a numerical stress of `E_I`. In the test environment CP2K offers **no numerical stress
for a TDDFT run**, and for a DFT run the analytical tensor is exact at one SCF cycle
instead of twelve. The vocabulary therefore has two members (`analytical`, `num`) and
the retired `num1` spelling is refused by name, so an old input is not silently
reinterpreted. Nothing of that route is implemented or tested.

**What is differenced.** The total energy of target state I is

```text
E_I(R, H) = E_0(R, H) + Omega_I(R, H)
```

with both terms in one unit (CP2K's excitation table may be eV while the total energy is
Hartree). The stress is the potential-energy stress, tension positive:

```text
sigma_I = (1/V) dE_I/depsilon = sigma_0 + (1/V) dOmega_I/depsilon
```

at fixed fractional coordinates, so the cell and the atoms deform together:

- differencing `Omega_I` alone gives the excited state's correction relative to the
  ground state; differencing `E_I` gives the complete excited-state potential stress;
- if the CP2K total energy already contains `Omega_I`, do not add it again;
- the external difference below is tension-positive as it stands; CP2K's own print uses
  the opposite convention and the parser negates it (section 6.5);
- `1 eV/A^3 = 160.2176634 GPa` and `1 Hartree/Bohr^3 = 29421.0265 GPa`.

**The external difference.** The six direction matrices, the `1/2` shear factor, the
central volume and the row-vector deformation are `service/stress/strain.py`'s (section
6.5). The formulation adds four things:

- both perturbed points reference the **central** configuration's target state, never
  each other, and neither point is relaxed - a relaxation would not be the instantaneous
  stress of the configuration the dynamics is on;
- a full tensor costs twelve perturbed single points plus the central calculation, which
  the step already has; one TDDFPT batch that solves several states yields several
  tensors at once;
- isotropic only: `p_I ≈ -[E_I(+) - E_I(-)]/(6 delta V0)`, positive under compression
  and the trace alone - not an input for a flexible-cell barostat;
- ground-state analytic plus an excitation correction
  (`sigma_0_analytic + dOmega_I/V`) avoids differencing an absolute energy but removes
  neither the excitation's noise nor the state-tracking problem.

**State tracking.** The implemented guard is section 6.5's "State identity":
energy-sorted adiabatic roots, the same number of roots as the central run, no root
moving more than half the smallest central gap, a degenerate central spectrum refused.
Character (NTO / transition-density) tracking is out of scope; where it would be
needed, the run reports rather than emitting a fabricated zero. Near a true degeneracy
the single-state derivative is neither unique nor smooth, which is why the framework's
crossing searches read a gap term instead (the boundary
[CELL_OPT_DEVELOPMENT.md](../methods/CELL_OPT_DEVELOPMENT.md) section 6.3 records).

**Step size.** Start at `delta = 1e-4 .. 1e-3` and scan `delta/2`, `delta`, `2 delta`
(`stress_delta`, section 3.1; its accepted range is section 3.2). The central difference
truncates at O(delta^2) while energy noise enters as `noise/(V0 delta)`, so smaller is
not automatically better; section 6.5's local measurements are the two limits (exact for
the quadratic `full` model, `1.8333e-05` relative for `iso` at `1e-3`, quartering with
each halving). The check list: SCF and TDDFPT convergence, enough printed digits, the
consistency of basis/CUTOFF/REL_CUTOFF/grid, grid-change noise from the perturbation,
one physical model at every point, and one output path per point.

**Coupling to an MD/NPT driver.** The returned tensor is the potential stress of the
current configuration and target state; a classical driver assembles
`P_total = P_kin - sigma_I_pot + P_constraint` with kinetic energy, stress and volume in
one unit at one instant. Removing the centre-of-mass motion zeroes the momentum, not the
kinetic pressure; CP2K computes single points and runs no second integrator; a surface
hopping run uses the active state's tensor, consistent with that step's forces and hop.
Whether a photoexcited non-equilibrium trajectory should use NPT at all is a modelling
question, and a full tensor costs several excited-state single points per step, so
updating it less often changes the integrator and must be validated on its own.

**Acceptance checklist.** Kept in the original numbering, with its measured standing.

1. Record the CP2K version, build, full input and target-state definition. **Partly:**
   the test rounds keep their inputs and logs; no separate version/build record is
   kept for the stress cases.
2. Validate the external difference against CP2K's ground-state **analytical** tensor,
   which fixes sign, transpose, volume and the shear factor. **Done** in the test environment
   (round `tmp_20261011_085800`, the same acetylene box through the same carrier,
   `analytical` against `num` at `stress_delta = 1e-3`): largest element-wise difference
   **2.309e-08 Hartree/Bohr^3**, diagonal relative gaps **1.15e-05 / 1.50e-04 /
   2.83e-05**, trace pressure **1.94623930e-04** against **1.94616631e-04** (relative
   **3.75e-05**). A flipped sign or a transposed shear would have shown at O(1e-4), not
   at the eighth decimal.
3. Validate `E_I = E_0 + Omega_I` so the excitation is neither missed nor added twice.
   **Owed for the excited-state case.**
4. Compare the full `E_I` difference against ground-state stress plus an `Omega_I`
   difference. **Owed.**
5. ~~Compare the built-in NUMERICAL path against the external difference~~ - dropped
   with that route; the external difference is checked against the analytical tensor
   instead (item 2), which tests the same conventions.
6. Scan strain steps and electronic convergence thresholds. **Owed, and first**: item 2
   holds at one `delta`, so it says the conventions match, not that the step is small
   enough.
7. Check the state match at every perturbed point, including neighbours and
   configurations prone to reordering. **Owed** (the guard exists; no engine reading).
8. Check a triclinic cell and the tensor's rotation. **No measurement.**
9. Validate fixed-cell dynamics first, then add pressure control and check the
   timestep, the averages and the fluctuations. **No measurement.**

**Official documentation.** The CP2K manual's FORCE_EVAL numerical-stress settings,
`EXCITED_STATES`/`STATE` and TDDFPT pages; the Doxygen source of
`force_env_calc_num_pressure`; `excited_states.F`, `qs_energy.F` and `qs_force.F`; and
ASE's CP2K interface for the sign conversion. trunk/master and Doxygen move with
upstream - the installed version's own documentation plus the readings above are what
count.

---

## 7. Molecular Dynamics: Two Driver Modes

- **driver=main (framework AIMD)**: jobtype `md` maps to the `_gradient`
  profile suffix (`_JOBERTYPE_SUFFIX_MAP` in `schema/method.py`), i.e. the
  framework steps through the trajectory itself and calls the gradient
  profile one step at a time. No `&MOTION` section is involved.
- **driver=skill (standalone)**: the explicit `dft_md` / `tddft_md` profiles
  run an in-engine `&MOTION &MD` block (section 5.4) with the NVT CSVR
  thermostat and `&QS EXTRAPOLATION ASPC` for SCF stability along the
  trajectory. The final frame is returned via `fin_coord`.
- **NPT is not implemented**: `md_ensemble` supports NVT; NPT support is not
  part of the current implementation.

---

## 8. Runtime Parameters and Environment

| Parameter | Effect |
|---|---|
| cp2k_path | CP2K root exported as `$CP2K_HOME`; validated by env_check |
| cp2k_exe | Executable override; default `$CP2K_HOME/exe/local/cp2k.psmp` |
| setup_script | Sourced before launch (toolchain environments) |
| nproc, cp2k_nproc | `mpirun -n {nproc}` rank count; `thread`/`cp2k_thread` set `OMP_NUM_THREADS` (#306) |
| shell | Extra shell commands injected into the run script |
| execution_mode | `local` or `slurm` script generation |

`service/runner/env_check.py` verifies the environment **before** any
calculation: an unset `cp2k_path`, a non-existent path, or a missing
executable raises `RuntimeError`. `$CP2K_HOME` placeholders inside
`cp2k_exe` are expanded, and relative executable paths are resolved against
`cp2k_path`.

---

## 9. Execution Scripts

`service/templates/run_templates.py` reproduces the proven `cp2k_demo/run.sh`
pattern — environment exports, optional setup sourcing, then mpirun with
stderr merged into the log:

```bash
#!/bin/bash
export CP2K_PROJECT={project}
export CP2K_HOME={cp2k_path}
export CP2K_EXE=$CP2K_HOME/exe/local/cp2k.psmp   # or the explicit cp2k_exe
export OMP_NUM_THREADS={thread}
# optional: source {setup_script}
mpirun -n {nproc} $CP2K_EXE -i $CP2K_PROJECT.inp -o $CP2K_PROJECT.log 2>&1
```

The SLURM variant prepends the `#SBATCH` header (partition, time, ntasks,
job-name, optional user template) to the same command block. `2>&1` merges
stderr into `{project}.log` so CP2K crash banners are captured in the parsed
log rather than lost to the scheduler's stderr stream.

---

## 10. Test Cases

### 10.1 Qualification YAML Cases (`tests/`)

Ten cases, all with an explicit periodic box (`pbc: [1,1,1]` with
`lattice_vector` written out); six of them use **ethylene (C2H4, 6
atoms)** in a 10x10x10 A box, one a four-water 6.3 A box (12 atoms), two
an eight-water 6.3 A box (24 atoms), and they cover both PBE and PBE0:

| File | Profile | Functional | Atoms | Notable settings | Expected |
|---|---|---|---|---|---|
| dft_energy.yaml | dft_energy | pbe | 6 | cutoff 400 / ref_cutoff 50 | energy shape [1] |
| dft_gradient.yaml | dft_gradient | pbe | 6 | defaults | energy + gradient shape [1, 6, 3] |
| tddft_energy.yaml | tddft_energy | pbe | 6 | nstate 3, target_states | energy shape [3] |
| tddft_gradient.yaml | tddft_gradient | pbe0 | 6 | nstate 3, target_states [0,1], current_state 1 | energy [3], gradient [3, 6, 3] |
| tddft_gradient_water.yaml | tddft_gradient | pbe | 12 | four waters, 6.3 A cell, nstate 3 | energy [3], gradient [3, 12, 3] |
| dft_opt_min.yaml | dft_opt_min | pbe | 6 | opt_maxit 500 | fin_coord shape [6, 3] |
| dft_md.yaml | dft_md | pbe | 6 | md_nstep 5, md_timestep 0.5, NVT, 300 K | energy [1], fin_coord shape [6, 3] |
| dft_stress.yaml | dft_gradient | pbe | 24 | stress true (analytical) | energy [1], gradient [1, 24, 3], stress_tensor [1, 3, 3] |
| dft_stress_num.yaml | dft_gradient | pbe | 24 | stress_method num, multitask, multipool 4; scope and delta left to their defaults | energy [1], gradient [1, 24, 3], stress_tensor [1, 3, 3] |
| tddft_stress_num.yaml | tddft_gradient | pbe0 | 6 | nstate 3, stress_method num, stress_scope iso, stress_delta 1e-3 (serial arm) | energy [3], gradient [3, 6, 3], stress_tensor [3, 3, 3] |

The two numerical cases are the ones that can only be certified on a
machine with CP2K: the analytical/numerical tensor comparison, the
`E_I = E_0 + Omega_I` accounting and the step-size convergence of
section 6.5 are read from their products. One case pins both strain
settings and one leaves them to the schema defaults, so the default fill
is exercised rather than overwritten everywhere.

### 10.2 Unit Tests (186 tests)

- `test_input_builder.py` (34 tests) — cutoff pass-through and validation,
  PBE/PBE0 functional mapping, unsupported functional rejection, dispersion
  types with `REFERENCE_FUNCTIONAL` tracking, element KIND blocks with the
  valence `-q<N>` suffix and user overrides, basis file lists, ADMM section
  + `AUX_FIT`, RUN_TYPE/`&PRINT` emission, cell/Poisson periodic-molecular
  branches, and TDDFPT section placement.
- `test_output_parser.py` (23 tests) — total energy from the log, the
  gradient = −force contract, missing-file `RuntimeError`s, excited-state
  energies from the `TDDFPT|` table, tdfrc excited-state forces, tdfrc
  block-count fail-fast, the `fin_coord` channel for GEO_OPT and MD, and
  the stress-tensor sign/unit/file-priority contract.
  Mock outputs follow the formats of real CP2K products from the
  `cp2k_demo` reference run.
- `test_run_script.py` (8 tests) — environment exports, stderr-merged
  mpirun line, custom exe / setup script, shell command pass-through, SLURM
  header, and user template respect.
- `test_strain_math.py` (28 tests) — the strain direction conventions
  (including the shear factor-of-two negative control), fractional
  coordinate preservation, the central-volume denominator, the
  tensile-positive sign, the isotropic trace contraction, the triclinic
  case, the unit factor against an SI-derived value, and the measured
  O(delta^2) convergence.
- `test_stress_schema.py` (24 tests) — the stress route defaults, the
  `num` field routing, the `&cp2k` dispatch keywords reaching the skill
  (including the tri-state read: they are absent from the framework's
  merged dict unless the user writes them, and a strict `bool` field
  would reject a `None` hand-off), and every refusal by name (unknown
  route or scope, out-of-range step, non-periodic or singular cell,
  `remotetask` against a local arm, excited state under an engine-tensor
  route).
- `test_num_stress_driver.py` (26 tests) — the whole route against a
  stand-in engine that replaces only CP2K: tensor agreement with a
  closed-form analytic stress for dft and for every state of a tddft run,
  both strain scopes, point planning (energy-only profile, no stress
  keyword, no recursion, unique stems, sibling directories, strained
  `&CELL` in the written input), failure and state-swap propagation
  through the real parser, serial-versus-pool equality, pool capping and
  ordering, the slurm arm with a user template, and the oversubscription
  warning, and the `python -m ...agent <config.json>` subprocess entry for the
  whole route. Controls over the harness itself assert that the engine really
  ran and that the strain really moved the lattice.
- `test_skill_cases.py` (43 tests) — every shipped case loaded through the
  real schema pipeline, with its keywords checked against the schema
  (an undeclared keyword would be dropped in silence), its `expected`
  shapes checked against its own atoms and nstate, and the stress-route
  rules checked per case.

None of these tests requires a CP2K installation.

---

## 11. Key Differences from Other Skills

- **Section-tree input, compositional builder**: every CP2K section has its
  own builder function; profiles compose them through the workflow graph.
- **Only two functionals, PBE0 hand-assembled**: hybrid exchange is built
  from scaled PBE + explicit `&HF` rather than a registry lookup.
- **TDDFPT support with state-resolved gradients**: excited-state forces via
  tdfrc blocks, fail-fast on block-count mismatch — no other QM skill in the
  project parses this format.
- **Dual MD driver modes**: framework-driven AIMD reuses the gradient
  profile; standalone `*_md` profiles carry their own `&MOTION` block
  (NVT only; NPT not implemented).
- **`fin_velocity` reserved**: unlike lammps_skill, final velocities are not
  returned (`fin_velocity` is always empty).
- **Periodic/molecular cell handled automatically**: explicit lattice
  vectors give periodic Poisson; molecules get a padded vacuum box with the
  wavelet solver.

---

## 12. Design: An Engine-Side Hessian (`&VIBRATIONAL`), Not Implemented This Round

**Ruled 2026-10-20: this is not an owed variable-cell item.** A formal skill
backend cannot supply the lattice rows a variable-cell TS needs; the framework's
strained batch is the supported route (`optimize_cell_ts`, `hess_driver: main`).
The design below stays as a future note for the atomic half, not as a blocker
for the cell search.

The CP2K manifest declares `capabilities: [energy, gradient, stress,
transition_dipole, geometry_optimization, abinitio_dynamics]` and its profile list
is `dft_energy dft_gradient dft_opt_min dft_md tddft_energy tddft_gradient
tddft_transition_dipole tddft_opt_min tddft_md` - no force constants at either
level. Section 6 therefore has nothing to parse for a Hessian, and this section
records what would be built, in the same design-only state as
`dev/skills/XTB_SKILL_DEVELOPMENT.md` section 8.

**Why design-only.** No CP2K vibrational output is in this repository, and cp2k is
not installed on this environment: the only CP2K products the tooling has been written
against are the run logs of the strain route, whose reader (`tools/verify.py::_engine_cp2k`)
was written after round 15 put a real log on disk (#385). A `&VIBRATIONAL` parser
written before its first log is the mistake that rule exists to prevent.

**The manifest chain to add.**

| Layer | What a Hessian profile needs | Today |
| --- | --- | --- |
| `manifest.yaml:capabilities` | `vibrational_frequency`, the spelling `data/core.py::MATRIX_CAPABILITIES` accepts and the one ORCA and MOLCAS already use | absent |
| `manifest.yaml:profiles` | `dft_hessian` beside `dft_energy`/`dft_gradient`, `capability: vibrational_frequency`, steps `[energy, gradient, hessian]`; a `tddft_hessian` only once an excited-state matrix has a producer, which it does not (see below) | absent |
| `schema/method.py` | the method-to-profile row | absent |
| `query.resolve_profile` | jobtype token `hessian`/`frequency` → `dft_hessian` | falls through to `dft_energy` |
| `service/builder/*` | the `&VIBRATIONAL` subsection and its `&PRINT` block | absent |

**The input skeleton, as a shape rather than a claim.** The skill writes one
`&GLOBAL`/`&FORCE_EVAL` tree per job; a vibrational request adds, inside
`&FORCE_EVAL`:

```
&SUBROUTINE
  VIBRATIONAL
&END SUBROUTINE
&MOTION
  &VIBRATIONAL
    ANALYTICAL   # or EMPIRICAL, and this is one of the owed answers
    &PRINT
      &NORMAL_MODES
      &END NORMAL_MODES
      &MAT_R
      &END MAT_R
    &END PRINT
  &END VIBRATIONAL
&END MOTION
```

What the first real run must settle before any of that becomes a template:

1. whether `&VIBRATIONAL` in this build takes `ANALYTICAL`, `EMPIRICAL`, or a
   `&SUBROUTINE` switch - the skill's ownQP/semi-empirical paths differ here and
   the input is wrong in a way the log reports only as a stopped run;
2. where the matrix is printed: `&MAT_R` gives the derivative-matrix file, whose
   name and unit convention (atomic units, unmixed Cartesian versus mass-weighted)
   the parser depends on;
3. whether the frequencies themselves are read from the log's `&NORMAL_MODES`
   print or derived by the framework from the matrix - the framework already
   derives them for its own numerical batch (`workflow/hessian.py`), and a run
   that has two producers of one number has to say which one the record shows,
   the way the `Cell Gap Term` row says whose stress it printed (#389);
4. the cell question, which is the one this package's variable-cell work is about:
   `&VIBRATIONAL` relaxes nothing, so its matrix is the geometry it was handed,
   and a run with `optimize_cell: true` must not let the engine's own cell
   optimizer compete with the framework's.

**The `optimize_cell` pairing, half of which is already enforced.** A variable-cell
run under `hess_driver: skill` asks two different things of one backend: force
constants for the atoms, and a lattice derivative. The framework measures the
lattice half itself - twelve strained evaluations differenced and joined to the
atomic block (`workflow/optimize.py:196-223`, #365) - and refuses the pairing
outright when the state is excited, because a backend's matrix is its ground
state's (`workflow/optimize.py:205-223`, and now `data/core.py`'s capability gate,
which refuses by name before an evaluation is paid for, tested in
`test_core.py::TestCurvatureProducerScope`). A CP2K Hessian profile inherits that
structure unchanged: it supplies the atomic block, the framework keeps the six
lattice rows, and the excited-state refusal stays, since `tddft_*` profiles carry
no matrix either.

**Not done this round, so that it is not read as done:** no capability entry, no
profile, no `resolve_profile` branch, no `&VIBRATIONAL` template, no parser, no
number. The route to those is a test run that leaves one vibrational log
behind; the four questions above are what to read off it.
