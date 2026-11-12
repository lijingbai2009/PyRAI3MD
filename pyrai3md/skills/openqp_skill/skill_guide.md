# Skill Guide: openqp_skill

## Overview

openqp_skill runs OpenQP (open-source quantum chemistry package) via subprocess,
providing HF, TD-DFT (RPA/TDA) and MRSF-TDDFT calculations through the unified
PyRAI3MD skill interface. It never imports pyoqp; all interaction goes through
the `openqp` CLI and fixed-name text files.

## Setup Requirements

- OpenQP installed: the `openqp` command must be available in PATH
  (provided by a proper OpenQP installation). Optionally set
  `openqp_path` to the installation root; it is exported as the
  `OPENQP_ROOT` environment variable in the run script, and
  `$OPENQP_ROOT/lib` is prepended to `LD_LIBRARY_PATH`.
- `use_mpi: true` additionally requires an MPI launcher (`mpirun` in PATH or a
  custom `mpi_command`).

## Supported Profiles

| Profile | Capability | runtype |
|---|---|---|
| hf_energy / tddft_energy / mrsf_energy | energy | energy |
| hf_gradient / tddft_gradient / mrsf_gradient | gradient | grad |
| mrsf_nac | nonadiabatic_coupling | nac |
| mrsf_soc | soc_coupling | soc |
| mrsf_soc_gradient | soc_coupling | data |
| hf_geometry_optimization / tddft_geometry_optimization / mrsf_geometry_optimization | geometry_optimization | optimize |
| tddft_nonadiabatic_dynamics / mrsf_nonadiabatic_dynamics | nonadiabatic_dynamics | data (ktdc) / prop (dcm) |
| hf_frequency / tddft_frequency / mrsf_frequency | vibrational_frequency | hess |

`mrsf_nac` computes the **NAC vector** (runtype=nac) for single-point PES
analysis, not the dynamics NACME. OpenQP only supports one state pair per
NAC vector calculation (no batch): the first `nac_pair` from `coupling` is
used (`[nac]states`), and only the mrsf method is accepted (no `tddft_nac`).
With `nac_type=dcm` the mrsf_nac profile switches to runtype=prop (nacme /
MO overlap via `[guess]file2`).

Method mapping to OpenQP input:
- `hf`   -> `[input]method=hf`
- `tddft`-> `[input]method=tdhf` + `[tdhf]type=tda/rpa` (default tda) + `functional`
- `mrsf` -> `[input]method=tdhf` + `[tdhf]type=mrsf` + `[scf]type=rohf, multiplicity=3`

### Spin-orbit coupling (OpenQP 1.3)

`mrsf_soc` computes the singlet-triplet spin-orbit couplings and the two
ladder energies (runtype=soc). `mrsf_soc_gradient` adds the state gradients
to the same call (runtype=data). The framework's `&md dosoc` flag routes a
dynamics step to that same runtype=data path, so a NAMD step gets energies,
gradients and couplings in one call.

OpenQP answers SOC only under a specific shape, and the skill refuses
anything outside it instead of letting the job die remotely:

- reference must be triplet ROHF with an MRSF response (`method=mrsf` derives
  all of it);
- both ladders are addressed by their own selectors, `[properties]grad_s` and
  `grad_t`. The legacy combined `grad` is refused, because one index cannot
  say which ladder it counts;
- **no NAC of any flavour can join a spin-orbit run.** The analytic vector
  route answers the singlet fold only, and the numerical route would run one
  3N displacement sweep per ladder into scratch with no ladder slot. A
  `nac_type=nac` request alongside SOC is refused by name;
- the two-electron mean-field term (`soc_2e`, OpenQP's default 1) is not
  available with spherical-harmonic functions, so a spin-orbit deck pins
  `ispher=false`.

`runtype=prop` is not the profile for "gradients plus couplings": its driver
opens every run by loading the previous step's checkpoint, and standalone it
stops with "molecule loads previous step data cannot find [guess]file2 or
[input]system2" (measured). It remains reachable only through the
`nac_type=dcm` dynamics path, which writes the step before it.

State list convention for a two-manifold run: `state_spin` is sorted, so
singlets come first and triplets follow, `[tdhf]nstate_s` / `nstate_t` size
each ladder (0 means derive from the declared rows), and `[tdhf]nstate`
stays the configured ladder's count rather than the total.

## Workflow

1. Build config (schema build -> normalize -> validate)
2. Compile workflow: profile -> runtype + input section chain
3. Write `$OPENQP_PROJECT.xyz` and `$OPENQP_PROJECT.inp` into the calcdir
4. Run `openqp $OPENQP_PROJECT.inp` via bash script (local) or sbatch (slurm)
5. Parse results from the exported fixed-name text files (see below)

## Query Available Methods

```python
from pyrai3md.skills.openqp_skill import query

query.list_profiles()          # all profile names
query.get_profiles()           # profiles with capability/method/sections
query.get_profile('tddft_gradient')
query.get_sections('tddft_gradient')
# ['input', 'guess', 'scf', 'tdhf', 'properties']
query.get_capabilities()
query.get_registry()           # standardized registry for the framework
```

## Python API

```python
from pyrai3md.skills.openqp_skill import OpenQPBackend

backend = OpenQPBackend(config)                    # default calcdir: tmp_OPENQP
backend = OpenQPBackend(config, execution_id='tmp_OPENQP')
backend = OpenQPBackend(config, calcdir='/abs/path/calcdir')

results = backend.calculate(atoms, coordinates)

# Multi-step dynamics: reuse the same backend instance (or the same
# execution_id across instances). The orbital json is automatically
# continued through the fixed file name mechanism.
results = backend.calculate(atoms, new_coordinates, target_states=[1])
```

## Configuration Schema

Key calculation fields (see `schema/config.py` for the full list):

- `profile`, `method` (hf/tddft/mrsf), `basis`, `functional`
- `td_type` (tda/rpa, default tda; forced to mrsf for the mrsf method)
- `scf_type` (auto-derived: rhf/rohf; forced rohf for mrsf), `scf_maxit`, `scf_converge`
- `td_maxit`, `td_converge`, `nstate`, `target_states`
- Spin-orbit run controls: `dosoc` (the framework `&md` spelling; the
  `soc` jobtype token resolves it for a single point), `nstate_s` /
  `nstate_t` (ladder sizes, 0 = derive from the declared states),
  `grad_s` / `grad_t` (explicit per-ladder gradient roots; empty means
  derive from `target_states` each step), `soc_2e` (1 = include the
  two-electron mean-field term, 0 = one-electron only; unset keeps
  OpenQP's default), `scal_rel`, `scf_converger` (accepted as
  `converger_type`, OpenQP's own spelling) and `escalation` (a
  comma-separated fallback ladder; leave unset to keep OpenQP's
  `diis -> soscf -> trah`, since naming one stage replaces the chain).
- `guess_type` (default auto), `nac_type` (ktdc, dcm or nac; default
  ktdc; the framework spelling `nactype` is accepted as an alias).
  In nonadiabatic_dynamics profiles: `ktdc` = runtype=data, per-step
  energies + gradients only (the framework kTDC integrator estimates
  the coupling from the energy curvature); `dcm` = runtype=prop
  nacme/MO overlap; `nac` = runtype=data plus `[properties]nac=nac`,
  the numerical NAC vectors that driver answers (they cannot be
  combined with `dosoc`, which refuses any NAC). In the `mrsf_nac`
  single-point profile: `nac` = runtype=nac NAC vector for one state
  pair (PES analysis), `dcm` = runtype=prop nacme.
- NAC vector settings (runtype=nac only): `nac_dt` (default 20.67),
  `nac_dx` (default 0.0001), `nac_bp` (default true)
- `opt_maxit`, `opt_optimizer`, `opt_rmsd_grad`, `opt_max_grad`, `hess_dx`

### Numerical differentiation parallelism: which knob decides what

The displaced structures of a numerical Hessian or a numerical NAC vector run
on as many **MPI ranks as the launcher starts**, one rank per displacement,
each rank using its own OpenMP threads. Two knobs cover it, and **neither is
written into the deck**:

- `openqp_nproc` -- the rank count (`mpirun -n N`; in slurm mode
  also `#SBATCH --ntasks`). OpenQP has no input keyword for this at all.
  One spelling only: the bare `nproc` other skills also accept is retired
  here, so the number a run writes is the number the launcher starts.
- `thread` / `openqp_thread` -- `OMP_NUM_THREADS`, threads per process/rank
  (OpenQP's own spelling is `[input] omp_threads`, also `--omp N`).

`hess_thread` and `nac_thread` are retired: they wrote `[hess]nproc` /
`[nac]nproc`, which OpenQP reads only when the run is not under MPI
(`_displacement_pool` takes `min(displacements, comm.size)` on the MPI branch),
so under a rank-launching scheduler the deck value silently was not the worker
count. Because the framework forwards an `&openqp` section without checking its
keys against any table, this skill is the only place that can still report a
retired keyword: writing either name raises with the replacement spelled out,
rather than being filtered away. `nproc` and `mpi_nproc` are refused the same
way; `openqp_nproc` is the only rank spelling this skill takes.

What a run actually used is readable from the engine's own `num_hess` /
`num_nacv` log block, which prints the worker count and `OMP_NUM_THREADS` side
by side; the CPU footprint OpenQP accounts for follows the same split
(`ranks x omp` under MPI, the deck's own `[hess]/[nac] nproc x omp`
otherwise -- this skill writes neither line, so there the engine uses its
default).

Environment fields:

- `openqp_path` (exported as `OPENQP_ROOT` in the run script; its
  `lib` directory is prepended to `LD_LIBRARY_PATH`),
  `openqp_project` (default OQP; fixes all file names)
- `use_mpi` (default false), `openqp_nproc`, `mpi_command`
- `thread` (OpenMP threads via OMP_NUM_THREADS; `openqp_thread` in the
  framework spelling), `keep_tmp`
- `execution_mode` (local/slurm), `slurm_time`, `slurm_partition`,
  `slurm_ntasks`, `slurm_memory`, `slurm_command`

## Parallel Execution

- Default no-MPI mode (OpenMP only): the run command is
  `openqp $OPENQP_PROJECT.inp --nompi`; threads come from `thread`.
- MPI mode (`use_mpi: true`): the run command becomes
  `mpirun -n <openqp_nproc> openqp $OPENQP_PROJECT.inp` (no `--nompi`);
  `mpi_command` overrides the launcher prefix (e.g. `mpiexec -n 4`).
  In slurm mode `openqp_nproc` is also mapped to `#SBATCH --ntasks`.

## Orbital Continuation (fixed file name mechanism)

Every calculation writes `[guess]save_mol=true`, so OpenQP stores the orbital
json next to the log file. Since the log is always `$OPENQP_PROJECT.log`, the
orbital json is always `$OPENQP_PROJECT.json`. `[guess]file` declares the same
file with `type=auto`: OpenQP reads it when present and falls back to the
huckel guess otherwise (first step).

For `nac_type=dcm` in NAC profiles (nac / nonadiabatic_dynamics, runtype=prop),
the builder additionally writes `[guess]file2=$OPENQP_PROJECT.json`: OpenQP
loads the previous-step orbitals from file2 to compute the MO overlap needed
for NACME. Other workflows (including the runtype=data ktdc dynamics path)
never write file2, so OpenQP cannot fail on a missing json. In the namd
workflow all steps share the same calcdir, so the json saved by the previous
step is found in place automatically. On the very first prop/dcm step the
json does not exist yet and OpenQP aborts the run: the backend now surfaces
this through the non-zero return code / missing-energies errors instead of
silently returning zero results, so a dcm dynamics run must be seeded with
an orbital json (or use ktdc, which needs no orbital continuation).

## Result Reading (export mechanism)

All profiles set `[properties]export=true` with an empty title, so OpenQP
dumps results into fixed-name text files in the calcdir:

- `energies` — state energies (one per line)
- `grad_<i>` — gradient of state index i (hf ground state = `grad_0`,
  other methods: framework state k -> `grad_{k+1}`)
- `nacme` / `dcme` — NACME / DCME matrices (`nac_type=dcm`, runtype=prop)
- `nac_<i>_<j>` — NAC vector for a single 1-based state pair
  (`nac_type=nac`, runtype=nac)
- `opt.xyz`, `opt_status.txt` — optimized geometry and iteration table
- `$OPENQP_PROJECT.hess.json` — hessian, frequencies, normal modes
- `socme` — spin-orbit table: one row per (singlet, triplet, Ms) with
  `S_index T_index Ms Re Im Absolute`, indices counting **from 0 within each
  ladder**, Ms over -1/0/+1, values in cm-1 (the header names the unit). The
  skill returns one magnitude per declared pair, `sqrt(sum Ms Absolute^2)`,
  in `soc_pair_map` order.

A run that solves both spin ladders does **not** write the unlabelled names
above. Each manifold is published under its own label - `energies_s`,
`energies_t`, `grad_s_<i>`, `grad_t_<i>`, `nacme_s`, `dcme_t` - and the
unlabelled stem is left for the numerical-differentiation parent to hand
results to its displaced workers. Row 0 of each energy file is that ladder's
internal MRSF reference, not a surface. Measured on a C2H4/sto-3g MRSF run
with two singlets and two triplets: `socme` holds 2 x 2 x 3 = 12 rows, its
(S, T, Ms) labels match the log's own singlet-triplet table row for row, and
`Absolute` agrees with the printed value to 4.1e-13.

The parser reads these files only; the log is checked solely for
completion/convergence markers.

## Important Notes

- State index convention: hf ground state uses index 0 (`grad=0`,
  `energies[0]`); for tddft/mrsf the framework state k maps to OpenQP
  index k+1 (`energies[k+1]`), and in MRSF grad=1 is S0. A two-manifold run
  splits that one list by ladder instead: state k below `nstate_s` is
  `grad_s_{k+1}` / `energies_s[k+1]`, a state at or above it is
  `grad_t_{k-nstate_s+1}` / `energies_t[k-nstate_s+1]`.
- Advanced runtypes (MECI/MECP/MEP/TS) are not part of the current
  profile set; they can be added later by extending `method.py` maps.
  SOC is part of it as of OpenQP 1.3 (see the spin-orbit section above).
- Units: energies in hartree, gradients in hartree/bohr, spin-orbit
  couplings in cm-1 (OpenQP 1.3 writes `socme` already converted; the
  framework's FSSH integrator applies its own cm-1 to hartree factor).
