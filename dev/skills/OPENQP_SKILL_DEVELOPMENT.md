# OpenQP Skill Development Guide

Last updated: 2026-10-27 19:02 (#427 按用户裁定把 nproc 并入 openqp_nproc：Environment 不再有 nproc 字段，裸名经 RETIRED_RESOURCES 按名拒绝；run_templates 的 mpirun -n 与 slurm --ntasks、validate 的 use_mpi 检查都改读 openqp_nproc；manifest 的 control_parameters 与 resources.ranks.keyword 同步改名，框架侧资源声明闸允许 ranks 由技能用自己的前缀名声明。本轮校订 3.1 的 mpi_command 行、3.2 的 openqp_nproc/mpi_command/hess_thread 行、6 与 6a，其余小节沿上次口径未重新校订。此前 2026-10-27 17:40 (#426 按用户裁定收口数值差分的并行度口径：删除 hess_thread/nac_thread 两个键（它们写的 [hess]nproc/[nac]nproc 只在非 MPI 分支被读），rank 数与 OMP 线程分别由 nproc/openqp_nproc 与 thread/openqp_thread 决定且都不进 inp；6a 改写为两键一表，3.2 的两行改为退役说明，4 的 [nac] 行去掉 nproc，mrsf_nac 用例删掉 nac_thread，adapter 的 RETIRED_RESOURCES 按名拒绝两个旧键。此前 2026-10-27 17:25 (新增 6a 说明数值 Hessian/NAC 差分池的并行度由谁决定：MPI 下 rank 即 worker、`[hess]nproc`/`[nac]nproc` 只约束非 MPI 线程池、rank 数没有输入关键词只由启动器决定；3.2 的 hess_thread/nac_thread 两行同改为点明只在非 MPI 路径生效。本轮只改这两处，其余小节未重新校订。此前 2026-10-23 07:40 (OpenQP 1.3 spin-orbit support: section 1.3 is rewritten from 'No SOC Support' to what the engine answers and refuses, with a new 1.3a for the `socme` table; 1.5/1.6 report the ladder-labelled export names, the per-ladder index arithmetic and the conditional triplet-row rule; 2.1 lists 17 profiles; 3.2 registers dosoc/nstate_s/nstate_t/grad_s/grad_t/soc_2e/scf_converger/scal_rel/escalation; 4 says which lines each section now writes; 5.1/5.2 cover socme and its refusals; 7 counts 16 case files and names test_soc_routing.py; 8 replaces 'Not supported'. Every number quoted there was measured on OpenQP 1.3.1 on the test environment. Sections 1.1-1.2, 1.4, 2.2-2.3, 3.1, 3.3 and 6 were not re-audited. Before it, 2026-10-22 22:30 (section 1.3 now says the SOC field shapes exist and no SOC capability is registered; the parser section reports the `RuntimeError` on a missing `energies`/`grad_<i>` export instead of a silent zero, with NAC vectors the only zero-filled case; the Calculation/Environment field lists move `nstate` and `openqp_project` to their real models; the case list is 13 with `mrsf_nonadiabatic_dynamics`; the profile contract is cited at `query.py::resolve_profile`; config section and generator naming updated as in MOLCAS. The rest was not re-audited. Before it, 2026-10-11 16:25)

This document covers **OpenQP-specific** implementation details for openqp_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. OpenQP-Specific Design Considerations

### 1.1 External Command-Line Execution

OpenQP is an external command-line program executed via subprocess. The skill
invokes `openqp ...` (or `mpirun -n N openqp ...` with MPI). Unlike PySCF, it
generates input files and parses output files.

### 1.2 Method Scope

OpenQP supports three electronic structure methods:

| Method | Description | Roles |
|--------|-------------|-------|
| `hf` | Hartree-Fock (ground state) | qm |
| `tddft` | TDDFT (rpa/tda excited states) | qm |
| `mrsf` | Multi-Reference Spin-Flip | qm |

### 1.3 Spin-Orbit Coupling (OpenQP 1.3)

SOC is computed on OpenQP 1.3's MRSF response: the singlet and triplet
ladders are answered from one triplet ROHF reference and coupled through the
one-electron Breit-Pauli term plus, by default, the two-electron mean-field
correction (`soc_2e=1`; the correction is not available with
spherical-harmonic functions, so a spin-orbit deck pins `ispher=false`).
The manifest registers `soc_coupling` beside the capabilities above, and two
profiles carry it: `mrsf_soc` (runtype=soc, energies plus couplings) and
`mrsf_soc_gradient` (runtype=data, gradients with them). The framework's
`&md dosoc` flag reaches the same runtype=data path, so a NAMD step returns
energies, gradients and couplings in one call - measured end to end with
`examples/md_openqp_fssh_soc.yaml` (5 FSSH steps, C2H4, MRSF/sto-3g).

What the engine will not answer, the skill refuses at validation with the
engine's reason rather than a bare "unsupported":

| Request | Verdict | Reason OpenQP gives |
|---|---|---|
| SOC + NAC vector (`nac_type=nac`) | refused | the analytic route implements the singlet fold only; the numerical route would sweep 3N displacements once per ladder into scratch with no ladder slot |
| SOC + `nacme` on runtype=data | unreachable | `compute_data` keeps no overlap with the previous step; `nacme` lives on runtype=prop, which needs a previous step |
| SOC + legacy `[properties]grad` | never written | one index cannot say which ladder it counts; `grad_s` / `grad_t` are the selectors |
| runtype=soc + `grad_s`/`grad_t` | never written | that driver couples, it does not differentiate |
| SOC with `hf` or `tddft` | refused | no second ladder to couple |

`runtype=prop` is not the answer to "gradients and couplings together": its
driver opens every run by loading the previous step's checkpoint, and
standalone it stops with `molecule loads previous step data cannot find
[guess] file2 or [input] system2` (measured). Hence `mrsf_soc_gradient` is
mapped to runtype=data, and prop stays reserved to the `nac_type=dcm`
dynamics route, which writes the step before it.

`singlet/triplet_ci_*` remain another skill's vocabulary; here the ladder
sizes are `nstate_s` / `nstate_t`.

### 1.4 Fixed-File-Name Orbital Continuation

OpenQP uses a fixed-file-name mechanism for orbital continuation across steps:

- `[guess]save_mol=true` is always enabled
- Orbital JSON is stored as `${OPENQP_PROJECT}.json`
- `[guess]file` with `type=auto` enables automatic continuation (first step
  falls back to huckel if file missing)
- `[guess]file2` is only written for NAC profiles on the runtype=prop path
  (nac_type=dcm) to compute MO overlap (NACME)
- The backend protects this JSON file during calcdir cleanup

### 1.5 Export Mechanism

All profiles set `[properties]export=true` with empty title. The parser reads
fixed export files rather than parsing the main log:

- `energies` — state energies
- `grad_<i>` — gradient for state i
- `nacme/dcme` — NACME coupling vectors (runtype=prop, nac_type=dcm)
- `nac_<i>_<j>` — NAC vector for a single state pair (runtype=nac)
- `opt.xyz` / `opt_status.txt` — optimization trajectory and convergence
- `<project>.hess.json` — Hessian matrix
- `socme` — spin-orbit table (see 1.3a)

The log file is used only for completion verification (failure marker detection).

**A two-manifold run leaves every name above unlabelled-out.** When both
spin ladders are solved, each publishes under its own label - `energies_s`,
`energies_t`, `grad_s_<i>`, `grad_t_<i>`, `nacme_s`, `dcme_t` - and the
unlabelled stems are not written at all: they are the channel a numerical
Hessian or numerical-NAC parent uses to hand results to its displaced
workers, so a reader that expects `energies` there finds nothing. Measured
on the skill's own `mrsf_soc` deck (C2H4, sto-3g, 2 singlets + 2 triplets):
the run writes `energies_s`, `energies_t` and `socme` only. `title=` keeps
its usual meaning, prefixing every name with `<title>.` when set.

### 1.3a The socme table

`socme` holds one row per (singlet, triplet, Ms): six columns
`S_index T_index Ms Re Im Absolute`, the first two counting **from 0 within
their own ladder**, Ms over -1/0/+1, rows in `ist → jst → Ms` order and the
values in cm-1 (the header names the unit; OpenQP converts with
`0.5·α²·E_h`, about 5.8438 cm-1 per atomic unit). Re and Im carry the phase
of the Davidson amplitudes, a gauge freedom that flips between runs;
`Absolute` is the invariant and equals `hypot(Re, Im)`.

Only the singlet-triplet block is exported - the kernel prints a triplet-
triplet block in the log, but the file does not carry it, which matches the
framework's own pair split (same-spin pairs are NAC pairs, cross-spin pairs
are SOC pairs).

The skill returns **one magnitude per declared pair**, `sqrt(sum of the
three Ms Absolute²)`, in `soc_pair_map` order, because
`workflow/integrator/fssh.py` indexes `results['soc']` positionally against
that map. This is the same convention the molcas skill uses, and the unit
matches what the integrator expects (it applies its own cm-1 → hartree
factor). Row-for-row agreement with the log's printed table was measured on
OpenQP 1.3.1: labels exact, `Absolute` to 4.1e-13, `Re`/`Im` to the four
decimals the log prints.

### 1.6 State Index Convention

- HF: ground state, grad=0, energies[0]
- TDDFT/MRSF: framework state k → OpenQP state k+1
- MRSF: `grad=1` corresponds to S0 (spin-flip convention)
- The export files are read as a promise, not a suggestion: `energies` must carry
  a line for every declared state past the reference row the spin-flip convention
  skips, and every state whose gradient the run follows must have its `grad_<i>`.
  The clamp that used to shorten `nstate` to the lines available, and the
  `if grad is not None` that skipped an absent file, both left the remaining rows
  at zero, which satisfies the shape and describes no surface; each is now a
  refusal naming the row and the file.
- Two-manifold runs split that one list by ladder instead: a state `k` below
  `nstate_s` is root `k+1` of the singlet ladder (`grad_s_{k+1}`,
  `energies_s[k+1]`), a state at or above it is root `k-nstate_s+1` of the
  triplet ladder (`grad_t_{k-nstate_s+1}`, `energies_t[k-nstate_s+1]`). Root 0
  of either ladder is that ladder's internal MRSF reference, not a surface.
  The arithmetic is measured, not inferred: a deck with `nstate_s=3`,
  `nstate_t=2` writes `energies_s` with 4 rows, `energies_t` with 3, and a
  `socme` of 3 x 2 x 3 = 18 rows whose column maxima are 2 and 1.
- **A triplet row needs a two-manifold run.** One calculation used to be able
  to ask OpenQP for one spin block only (`[tdhf]multiplicity=1` for mrsf, no
  multiplicity line otherwise), so a `state_spin` naming a triplet would have
  been answered by a second singlet root wearing that row's name - a crossing
  search between surfaces that were never sampled, invisible downstream.
  OpenQP 1.3 answers both blocks from one deck when the run is two-manifold
  (the driver switches `[tdhf]multiplicity` to reach the other ladder and
  sizes each from `nstate_s` / `nstate_t`), so `validate.py` refuses a
  triplet row only on the single-block profiles: profile `mrsf_soc`,
  `mrsf_soc_gradient`, an explicit `grad_t` list or `&md dosoc` earn it.
- **No nonadiabatic coupling is needed for a singlet-triplet crossing**, and none
  is requested here: `opt_method: mecp` runs with the gradient difference as its
  only branching direction and reports `Branching Directions 1.0`. The
  `runtype: nac` profile therefore stays as it is - a NAC-vector single point that
  exports no gradients - and is not the profile a crossing search runs on.

---

## 2. OpenQP Calculation Profiles

### 2.1 Complete Profile List (17 profiles)

| Profile | Capability | Method | Runtype |
|---------|-----------|--------|---------|
| `hf_energy` | energy | hf | energy |
| `hf_gradient` | gradient | hf | grad |
| `hf_geometry_optimization` | geometry_optimization | hf | optimize |
| `hf_frequency` | vibrational_frequency | hf | hess |
| `tddft_energy` | energy | tddft | energy |
| `tddft_gradient` | gradient | tddft | grad |
| `tddft_geometry_optimization` | geometry_optimization | tddft | optimize |
| `tddft_nonadiabatic_dynamics` | nonadiabatic_dynamics | tddft | data (ktdc, nac) / prop (dcm) |
| `tddft_frequency` | vibrational_frequency | tddft | hess |
| `mrsf_energy` | energy | mrsf | energy |
| `mrsf_gradient` | gradient | mrsf | grad |
| `mrsf_nac` | nonadiabatic_coupling | mrsf | nac |
| `mrsf_soc` | soc_coupling | mrsf | soc |
| `mrsf_soc_gradient` | soc_coupling | mrsf | data |
| `mrsf_geometry_optimization` | geometry_optimization | mrsf | optimize |
| `mrsf_nonadiabatic_dynamics` | nonadiabatic_dynamics | mrsf | data (ktdc, nac; plus `soc=true` under `&md dosoc`) / prop (dcm) |
| `mrsf_frequency` | vibrational_frequency | mrsf | hess |

`tddft_nac` was removed: OpenQP's `runtype=nac` (NAC vector) only supports
MRSF (`tdhf cannot compute nac for non-mrsf calculations`).

`mrsf_nac` computes the **NAC vector** (`runtype=nac`) for single-point PES
analysis, not the dynamics NACME. OpenQP does not support batch NAC vector
calculations — only the **first** `nac_pair` from `coupling` is used.

### 2.2 Profile Resolution

`query.py::resolve_profile` (implemented at `schema/method.py:154`) derives the profile:

- `energy` → `_energy`, `gradient` → `_gradient`
- `optimize` / `geometry_optimization` → `_geometry_optimization`
- `nac` / `nonadiabatic_coupling` → `_nac`
- `dynamics` / `nonadiabatic_dynamics` → `_nonadiabatic_dynamics`
- `frequency` / `hessian` → `_frequency`

### 2.3 Input Section Requirements

Each profile requires specific OpenQP input sections:

| Profile | Required sections |
|---------|-------------------|
| *_energy | input, guess, scf, [tdhf for tddft/mrsf], properties |
| *_gradient | + properties (export) |
| *_nac (mrsf) | + nac section (runtype=nac NAC vector; nac_type=dcm switches back to runtype=prop with file2 for MO overlap) |
| *_geometry_optimization | + optimize section |
| *_frequency | + hess section |
| *_nonadiabatic_dynamics | no nac section on the default runtype=data path (ktdc: energies + gradients only); nac_type=dcm switches to runtype=prop and takes the nacme from [properties]nac=nacme with [guess]file2 |

---

## 3. OpenQP Schema Details

### 3.1 Configuration Models

The config is composed of 5 Pydantic sub-models nested in one `SkillConfig`, with flat keyword access routed by the owning sub-model - the former `create_model` field merge is gone:

- **System**: atoms, coordinates, charges, multiplicity, nstate, state_spin,
  nac_pair_map
- **Properties**: energy, gradient, nac, hessian, frequencies
- **Dynamics**: state_population, temperature
- **Calculation**: profile, method, basis, functional, target_states,
  target_nac_pair, target_soc_pair, project, td_type (rpa/tda); `nstate` lives in
  **System** (`schema/config.py:70`) and `openqp_project` in **Environment** (`:301`)
- **Environment**: openqp_project, openqp_path, openqp_nproc, use_mpi,
  mpi_command, keep_tmp, thread, scratch_path, slurm_*, shell

### 3.2 Key Parameters

| Field | Default | Description |
|-------|---------|-------------|
| functional | bhhlyp | DFT functional (for tddft/mrsf) |
| basis | 6-31g* | Basis set |
| td_type | tda | 'rpa' or 'tda' for TDDFT (default tda) |
| nac_type | ktdc | 'ktdc' = runtype=data energies+gradients for the framework kTDC integrator (default; the framework spelling 'nactype' is accepted as an alias); 'dcm' = runtype=prop nacme/MO overlap; 'nac' = runtype=nac NAC vector (single pair, PES analysis, *_nac profiles) and, inside *_nonadiabatic_dynamics, runtype=data with `[properties]nac=nac` (the numerical vectors that driver answers). Refused together with SOC (1.3) |
| dosoc | 0 | `&md dosoc` spelling; requests spin-orbit coupling on a dynamics step. The `soc` jobtype token resolves the same thing for a single point |
| nstate_s / nstate_t | 0 | Ladder sizes (`[tdhf]nstate_s/nstate_t`); 0 derives them from the declared `state_spin` rows. A given value must cover the declared rows and every `grad_s`/`grad_t` root |
| grad_s / grad_t | [] | Explicit per-ladder gradient roots. Empty derives them from `target_states`, which is how a step follows its current state |
| soc_2e | 1 | `[input]soc_2e`: two-electron mean-field SOC term on; 0 keeps one-electron only. Unset keeps OpenQP's default, and a spin-orbit deck pins `ispher=false` because the term aborts on spherical functions |
| scf_converger | '' | `[scf]converger_type` (diis/soscf/trah); accepted under OpenQP's own spelling `converger_type`. Empty writes nothing, leaving OpenQP's default |
| scal_rel | 0 | `[scf]scal_rel` scalar-relativistic order; 0 is OpenQP's default and is not written |
| escalation | '' | `[scf]escalation` reactive fallback ladder (e.g. 'soscf,trah'). Left unset on purpose: naming a stage *replaces* OpenQP's diis→soscf→trah chain instead of extending it |
| nac_dt | 20.67 | `[nac]dt` numerical derivative time step (fs) |
| nac_dx | 0.0001 | `[nac]dx` displacement step (bohr) |
| nac_bp | true | `[nac]bp` Berry phase correction |
| ~~nac_thread~~ / ~~hess_thread~~ | — | retired (#426): they wrote `[nac]nproc` / `[hess]nproc`, which OpenQP reads only off the MPI path. Writing either name now raises by name; the displacement batch is set by `openqp_nproc` (ranks) and `thread`/`openqp_thread` (OMP per rank), neither of which enters the deck (6a) |
| openqp_path | '' | Installation root, exported as `OPENQP_ROOT` in the run script and `$OPENQP_ROOT/lib` is prepended to `LD_LIBRARY_PATH`; the `openqp` command itself comes from PATH |
| openqp_nproc | 1 | Number of MPI ranks (MPI mode; mapped to `mpirun -n` and SLURM `--ntasks`). The only rank spelling this skill takes: #427 merged the old `nproc`/`openqp_nproc` pair into it, so a bare `nproc` is refused by name (`mpi_nproc` was already refused since #306) |
| thread | 1 | OpenMP threads per rank (`OMP_NUM_THREADS`), with `openqp_thread` as its framework spelling (#306, which retires `num_thread`) |
| use_mpi | false | Use MPI parallelization |
| mpi_command | '' | MPI launcher prefix; empty assembles `mpirun -n {openqp_nproc}`, a non-empty value overrides the whole prefix |

### 3.3 Normalize Workflow

1. `verify_electronic_structure`: electron count / multiplicity parity.
2. `verify_state_num`: target_states validation, NAC pair validation,
   zero-filled state arrays.
3. For MRSF: multiplicity must be 3 (triplet reference, spin-flip convention).
4. `derive_functional`: hf calculations force an empty functional (the
   schema default bhhlyp must not leak into hf).
5. `derive_nac_type`: `*_nonadiabatic_dynamics` profiles normalize `nac_type` to `dcm` (prop/nacme) or `ktdc` (data, energies + gradients; also the fallback for `nac` and missing values)
   (dynamics needs nacme/MO overlap, not the NAC vector).

---

## 4. Input Builder Details

`service/builder/input_builder.py` assembles OpenQP input files:

- `[input]` section: method, charge, multiplicity, coordinates; a
  spin-orbit run adds `ispher=false` and `soc_2e`
- `[guess]` section: save_mol, file (auto continuation), file2 (NAC profiles
  on the runtype=prop path only)
- `[scf]` section: type (rohf for mrsf), multiplicity, and the optional
  `converger_type` / `scal_rel` / `escalation` lines, each written only
  when it differs from OpenQP's own default
- `[tdhf]` section: type (rpa/tda), functional, nroots; a two-manifold run
  adds `nstate_s` / `nstate_t` and keeps `nstate` at the configured
  ladder's count, not the total across both
- `[properties]` section: export=true, plus the gradient selector - `grad`
  for a single-block run, `grad_s` / `grad_t` for a two-manifold one - the
  `soc=true` switch on the runtype=data and runtype=prop paths (never on
  runtype=soc, which takes no gradient), and the `nac` keyword
- `[nac]` section: numerical NAC vector control (`dt/dx/states/bp`; no worker
  count -- see 6a).
  runtype=nac takes the first pair only; runtype=data with a vector request
  lists every declared pair as `1 2,2 3` because that driver writes one
  export file per pair
- `[optimize]` section: optimization parameters
- `[hess]` section: Hessian computation parameters

---

## 5. Result Parsing

### 5.1 Output Files

| File | Content | Parser |
|------|---------|--------|
| `energies` | State energies (Hartree) | output_parser |
| `grad_<i>` | Gradient for state i | output_parser |
| `nacme/dcme` | NACME coupling vectors (runtype=prop) | output_parser |
| `nac_<i>_<j>` | NAC vector for a single pair (runtype=nac) | output_parser |
| `opt.xyz` | Optimization trajectory | output_parser |
| `opt_status.txt` | Convergence status | output_parser |
| `<project>.hess.json` | Hessian matrix | output_parser |
| `socme` | spin-orbit table, one row per (singlet, triplet, Ms), cm-1 (1.3a) | output_parser |
| `energies_s` / `energies_t` | per-ladder state energies (two-manifold runs only) | output_parser |
| `grad_s_<i>` / `grad_t_<i>` | per-ladder gradient of root i (two-manifold runs only) | output_parser |

A two-manifold run writes the labelled pair of rows and **no** unlabelled
`energies` / `grad_<i>` at all, so the reader must pick by the compiled
workflow's flag rather than trying one name and falling back.

### 5.2 Result Format

Results follow the standard shape contract:

- `energy`: np.ndarray (nstate,)
- `gradient`: np.ndarray (nstate, natom, 3)
- `nac`: np.ndarray (nnac, natom, 3)
- `soc`: np.ndarray (nsoc,) in cm-1, one magnitude per `soc_pair_map` row
- `hessian`: np.ndarray (3*natom, 3*natom)
- `frequencies`: np.ndarray (3*natom,)

A short or missing `energies` export, or a missing `grad_0` / `grad_<i>` file, raises
`RuntimeError` rather than reporting a zero (`service/parser/output_parser.py:195-233`);
only the NAC-vector exports still zero-fill (`:255-256`). The two-manifold
readers hold themselves to the same rule: a missing `energies_s` /
`energies_t`, a missing labelled gradient for a state the step follows, a
`socme` whose row count is not `nstate_s x nstate_t x 3`, and a declared pair
with no row in it are each a refusal. That last one matters more than it
looks: a zero is a legitimate coupling - the measured water deck at 6-31G*
returns 0.0000 cm-1 for its S1/T0 pair (and 35.34 for S0/T0), and the
C2H4/sto-3g case file returns 0.0000 for S0/T0 with 8.6359 for S1/T1 - so a
zero is both a real answer and the shape a missing calculation would take,
and only the row-presence check tells them apart.

**Normal modes arrive folded, not as the engine wrote them.** `norm_mode` is
declared 4-D `(state, mode, atom, xyz)` in `schema/config.py`, while OpenQP's
`<project>.hess.json` writes one mode per flat row: measured on the
deployment, `modes` is `(12, 18)` for a 6-atom system and `hessian` is
`(18, 18)`. `output_parser._as_norm_modes` performs that fold; a layout it
does not recognize folds to an empty array rather than being guessed at, and
`utilities/print_results._format_normal_modes` prints the shape instead of
raising, because handing the unfolded 2-D array to the printer aborted runs
the engine had finished (round tmp_20261011_085800, `hf_frequency_input` and
`mrsf_frequency_input`).

---

## 6. Runtime Parameters and Execution

Local script (`{project}.sh`) and SLURM script (`{project}.sbatch`) both export
environment variables (`OPENQP_PROJECT`, `OMP_NUM_THREADS`; when `openqp_path`
is set additionally `OPENQP_ROOT` and
`LD_LIBRARY_PATH=$OPENQP_ROOT/lib:$MKL_LIB:$GCC_LIB:$LD_LIBRARY_PATH`) and run
the `openqp` command from PATH:

```bash
# No MPI (default, use_mpi=false):
openqp $OPENQP_PROJECT.inp --nompi

# With MPI (use_mpi=true, no --nompi):
{mpi_command} openqp $OPENQP_PROJECT.inp
```

In MPI mode an empty `mpi_command` is assembled to
`mpirun -n {openqp_nproc}`; a non-empty `mpi_command` overrides the
launcher prefix as a whole (e.g. `srun -n 4`).

SLURM headers use `slurm_partition/slurm_time/slurm_ntasks/slurm_memory`;
a user-provided `{project}.slurm` template overrides the headers.

### 6a. What bounds the numerical-displacement batch

`numerical_hess` and the numerical NAC-vector driver share OpenQP's
`_displacement_pool` (`library/single_point.py`). The batch runs on as many MPI
ranks as the launcher starts -- one rank per displaced structure -- and each
rank carries its own OpenMP threads:

| Setting | Decided by | Reaches OpenQP through |
|---|---|---|
| ranks = workers | `openqp_nproc` | the launcher: `mpirun -n N`, `#SBATCH --ntasks` in slurm mode. **Not a deck key** -- OpenQP's input schema has none for it |
| OpenMP threads per rank | `thread` / `openqp_thread` | `export OMP_NUM_THREADS` (OpenQP's own spelling is `[input] omp_threads`, also `--omp N`). Not written into the deck either |

`hess_thread` / `nac_thread` were removed here (#426). They wrote
`[hess]nproc` / `[nac]nproc`, and `_displacement_pool` reads that value only on
its non-MPI branch (`min(ntask, nproc)`); on the MPI branch it takes
`min(ntask, comm.size)`. Under a rank-launching scheduler the deck number
therefore silently was not the worker count -- and since the framework forwards
an `&openqp` section without validating its keys against any table, the skill
is the only place that can still report the retired name, so
`schema/adapter.py::RETIRED_RESOURCES` refuses both by name instead of letting
the schema filter drop them.

The rank count kept one spelling only (#427): `openqp_nproc`. The bare `nproc`
is no longer a field of this skill's Environment, and writing it raises rather
than being dropped by the schema filter -- a rank count that a run sets and the
launcher never reads is the same silence #426 removed.

What a run actually used is readable from the engine's own `num_hess` /
`num_nacv` log block, which prints the worker count and `OMP_NUM_THREADS` side
by side (`single_point.py:2838,4020`), and OpenQP's CPU accounting follows the
same split (`ranks x omp` under MPI, the deck's own `nproc x omp` otherwise --
`utils/input_checker.py::_add_cpu_info`).

---

## 7. Test Cases

`openqp_skill/tests/` ships 16 YAML cases (ethylene):

`hf_energy`, `hf_gradient`, `hf_frequency`, `hf_geometry_optimization`,
`tddft_energy`, `tddft_gradient`, `tddft_geometry_optimization`,
`mrsf_energy`, `mrsf_gradient`, `mrsf_nac`, `mrsf_frequency`,
`mrsf_geometry_optimization`, `mrsf_nonadiabatic_dynamics`,
`mrsf_soc`, `mrsf_soc_gradient`, `mrsf_nonadiabatic_dynamics_soc`.

The three SOC cases carry the measured numbers of their own deck in the
header comment; they need OpenQP 1.3, which writes `socme` and the
ladder-labelled exports, and are switched off in `pmd_env.yaml` so a 1.2
deployment does not run them into a refusal. `tests/test_soc_routing.py` is
the no-engine companion: deck text, runtype flags, the refusals, and the
`socme` reader against a synthetic table.

Generate agent configs with:

```bash
# Agent configs are generated by the qualification converter
# (qualification/certification/converter.py: parse_test_yaml / generate_test_configs),
# which replaced the former per-skill dev_tool.py; the test modes run through
# `pyrai3md test`.
```

---

## 8. Key Differences from Other Backends

| Aspect | molcas_skill | openqp_skill |
|--------|--------------|--------------|
| Input style | Text module chain | INI-style sections ([input], [scf], etc.) |
| Coordinate unit | Bohr (Molcas convention) | Angstrom (framework unit) |
| SOC support | Dual RASSCF + RASSI | MRSF 1e Breit-Pauli + 2e mean-field (OpenQP 1.3), singlet-triplet block only |
| NAC mechanism | RASSI | NAC vector (runtype=nac / runtype=data with `nac=nac`) / NACME dcme (runtype=prop); never alongside SOC |
| Orbital continuation | File-based (JobIph) | Fixed-file-name JSON |
| Excited states | CASSCF/CASPT2 | TDDFT (rpa/tda) / MRSF |
| Result extraction | Log file parsing | Fixed export files; SOC read from the `socme` table, collapsed over Ms to one cm-1 magnitude per pair |
| MPI support | Via environment | Explicit use_mpi flag |
| MRSF method | Not available | Triplet reference spin-flip |
