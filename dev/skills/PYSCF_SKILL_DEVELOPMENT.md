# PySCF Skill Development Guide

Last updated: 2026-10-22 22:30 (section 1.3: the pair fields exist, no SOC/NAC capability is registered; the key-parameter table gives the real defaults and drops the non-existent `tddft_nroots`/`tda`/`xc_grids` keywords - `grid_level` is the grid knob, TDA is a method value, and the TD root count is derived in `service/energy.py:274-288`; the Environment list loses `pyscf_workdir`; the profile contract and the generator naming are updated as in the other skills. The rest was not re-audited. Before it, 2026-10-07 13:05)

This document covers **PySCF-specific** implementation details for pyscf_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. PySCF-Specific Design Considerations

### 1.1 Pure Python In-Process Execution

Unlike external-command-line skills (molcas, bagel, orca, xtb, openqp, lammps),
PySCF is a **Python library** executed via `import pyscf`. There is no external
binary to invoke. This has fundamental design consequences:

- No input file generation: the service layer calls PySCF Python API directly.
- No output parsing: results are extracted from PySCF objects in memory.
- `pyscf_path` in the environment config is `import` (use Python import), not a
  filesystem path to a binary.
- Slurm execution still works: the entire Python process is submitted as a Slurm
  job, and PySCF runs inside it.

### 1.2 Method Scope

PySCF supports five electronic structure methods:

| Method | Description |
|--------|-------------|
| `hf` | Hartree-Fock |
| `dft` | Kohn-Sham DFT (with functional selection) |
| `mp2` | Møller-Plesset second-order perturbation |
| `tddft` | Time-dependent DFT (excited states) |
| `tda` | Tamm-Dancoff approximation (excited states) |

### 1.3 No SOC or NAC Support

PySCF registers no SOC (spin-orbit coupling) or NAC (non-adiabatic coupling)
capability - its manifest lists energy, gradient, frequency, geometry_optimization,
ir, uv, nmr and analysis. The schema nevertheless carries the pair-field shapes for
contract parity (`nac_pair_map` at `schema/config.py:129`, `soc_pair_map` at `:130`,
`target_soc_pair` at `:259`, the last one validated by `schema/normalize.py:113-180`);
nothing in this skill fills them.

### 1.4 One Interpreter, Several Terms (#302)

Because PySCF runs in-process, two backend terms of this skill share Python and
library state. Three things used to be process-wide and are not any more:

- **Artifact root.** `log`, `chk`, `mo.molden The frequency record written by `save_freq` (`<project>.freq.molden`, only when `save_norm_mode: true`) is *not* PySCF's own convention: `workflow/sampling.py`'s `read_molden` follows Molcas and takes the modes as read, so `save_freq` divides PySCF's mass-weighted normalized `norm_mode` by sqrt(atomic mass) before writing, through the pure helper `format_freq_molden`. There is no PySCF entry in `FREQ_READERS` -- a sampled initial-condition ensemble reaches the MD run as `<title>.init.xyz`, not as a live frequency read.`, `opt.xyz` and the NTO/analysis
  files are named after `config.project` under `config.scratch_path`. A local
  run now points that field at its own `calcdir` (the directory the
  `execution_id` already names), so the names stay the same while the directory
  becomes per-term; this also removes the case where `set_checkpoint` fed one
  term's checkpoint to the next as its initial guess. The slurm arm already
  wrote everything under `calcdir` and is unchanged -- and it must keep
  reading `scratch_path` as the parent, because its config travels to the
  child as a json file and a `calcdir`-valued scratch would nest a second
  calculation directory inside the first.
  Consequence for a caller: `{scratch}/PYSCF.log` became
  `{scratch}/tmp_PYSCF/PYSCF.log`.
- **Log capture.** The runner used to install a tee on `sys.stdout` for the
  whole process, which was not stack safe and let one term's trace into
  another's log. The tee is now bound to the calling thread
  (`service/utils.set_stdout` / `current_stdout`) and reaches PySCF through
  `mol.stdout` (`service/energy.py`), which is where PySCF writes everything.
- **Thread count.** `OMP_NUM_THREADS` is read once by OpenMP, at the first
  parallel region of the process, so a per-term write cannot retune what is
  already latched. The runner sets it only if the process has not chosen a
  value yet and prints a line to stderr when that value is not the one the
  term asked for; `thread` is therefore a per-process setting.

With those three changes the skill no longer declines a thread pool; the
measurements that established it are in
`pyrai3md/skills/pyscf_skill/tests/test_concurrency.py` (two real terms in one
pool reproduce the serial energy bitwise, each logs into its own file, and
`sys.stdout` is never replaced).

---

## 2. PySCF Calculation Profiles

### 2.1 Complete Profile List (39 profiles)

| Profile | Capability | Method |
|---------|-----------|--------|
| `hf_energy` | energy | hf |
| `hf_gradient` | gradient | hf |
| `hf_hessian` | frequency | hf |
| `hf_opt_min` | geometry_optimization | hf |
| `hf_opt_ts` | geometry_optimization | hf |
| `hf_mo` | analysis | hf |
| `hf_esp` | analysis | hf |
| `hf_population` | analysis | hf |
| `hf_ir` | ir_spectrum | hf |
| `hf_nmr` | nmr_spectrum | hf |
| `dft_energy` | energy | dft |
| `dft_gradient` | gradient | dft |
| `dft_hessian` | frequency | dft |
| `dft_opt_min` | geometry_optimization | dft |
| `dft_opt_ts` | geometry_optimization | dft |
| `dft_mo` | analysis | dft |
| `dft_esp` | analysis | dft |
| `dft_population` | analysis | dft |
| `dft_ir` | ir_spectrum | dft |
| `dft_nmr` | nmr_spectrum | dft |
| `mp2_energy` | energy | mp2 |
| `mp2_gradient` | gradient | mp2 |
| `mp2_mo` | analysis | mp2 |
| `mp2_esp` | analysis | mp2 |
| `mp2_population` | analysis | mp2 |
| `tddft_energy` | energy | tddft |
| `tddft_gradient` | gradient | tddft |
| `tddft_opt_min` | geometry_optimization | tddft |
| `tddft_uv_vis` | uv_spectrum | tddft |
| `tddft_mo` | analysis | tddft |
| `tddft_esp` | analysis | tddft |
| `tddft_population` | analysis | tddft |
| `tda_energy` | energy | tda |
| `tda_gradient` | gradient | tda |
| `tda_opt_min` | geometry_optimization | tda |
| `tda_uv_vis` | uv_spectrum | tda |
| `tda_mo` | analysis | tda |
| `tda_esp` | analysis | tda |
| `tda_population` | analysis | tda |

### 2.2 Profile Resolution

`query.py::resolve_profile` (implemented at `schema/method.py:296`) derives the profile
when not explicitly given; the framework reaches it through the skill's public query
interface (`data/core.py:2498`):

- `energy` → `_energy`, `gradient` / `energy gradient` → `_gradient`
- `frequency` / `hessian` → `_hessian`, `optimize` → `_opt_min`
- `absorption` / `transition` → `_uv_vis` (tddft/tda) or `_esp` (hf/dft/mp2)
- `mo` → `_mo`, `population` → `_population`, `esp` → `_esp`
- `ir` → `_ir`, `nmr` → `_nmr`

---

## 3. PySCF Schema Details

### 3.1 Configuration Models

The config is composed of 5 Pydantic sub-models nested in one `SkillConfig`, with flat keyword access routed by the owning sub-model - the former `create_model` field merge is gone:

- **System**: atoms, coordinates, charges, multiplicity, nstate, state_spin
- **Properties**: energy, gradient, hessian, frequencies, dipole, transition_dipole,
  osc_strength, excitation, fin_coord, fin_velocity, computed_charges
- **Dynamics**: state_population, temperature
- **Calculation**: profile, method, basis, functional, grid_level, target_states,
  scf_converge, td_converge, maxit_scf/maxit_opt, save_mo, save_norm_mode, chk_file;
  `nstate` lives in **System**
- **Environment**: pyscf_project, pyscf_path (always 'import'),
  keep_tmp, scratch_path, slurm_*, thread, verbose

### 3.2 Key Parameters

| Field | Default | Description |
|-------|---------|-------------|
| functional | pbe0 | DFT exchange-correlation functional (`schema/config.py:32`) |
| basis | ccpvdz | Basis set |
| grid_level | 3 | PySCF DFT grid level (`schema/config.py:61`, `:267`) |
| (TD root count) | derived | not a keyword: `service/energy.py:274-288` splits `nstate` against `state_spin` and sets `td.nstates` from the singlet/triplet budget |
| (TDA) | method value | `tda` is a method name, not a flag - `method: tda` selects the Tamm-Dancoff service path |
| verbose | 4 | PySCF log verbosity (4=INFO) |

### 3.3 Normalize Workflow

1. `verify_electronic_structure`: electron count / multiplicity parity.
2. `verify_state_num`: target_states validation and zero-filled state arrays; the
   excited-state root count is not a schema field - it is derived at run time from
   `nstate` and `state_spin` in `service/energy.py:274-288`.
3. TDA is selected by the method value `tda`, never by a boolean flag.

---

## 4. Service Layer Architecture

### 4.1 Workflow Graph Compiler

PySCF uses a `WorkflowGraph` compiler (`service/builder/`) that maps each profile
to a sequence of computation steps:

```
Profile → WorkflowGraph → dispatch_calculation() → PySCF API calls
```

Each step function in `service/` (energy.py, gradient.py, hessian.py, optimize.py,
analysis.py, spectra.py) handles one computation type.

### 4.2 Result Extraction

Results are extracted from PySCF objects in memory (no file parsing):

| Property | Source |
|----------|--------|
| energy | `mf.e_tot` or `mc.e_tot` or `td.e` |
| gradient | `mol.gradient` (analytical) or numerical differentiation |
| hessian | `mol.hessian()` (numerical) |
| dipole | `mf.dip_moment()` |
| transition_dipole | TDDFT transition properties |
| excitation | TDDFT/TDA excitation energies |
| frequencies | Hessian eigenvalue decomposition |

---

## 5. Runtime Parameters and Execution

PySCF runs in-process (no external binary). The `pyscf_path` config is always
`'import'`. Execution modes:

- **Local**: PySCF runs in the same Python process as the skill backend.
- **Slurm**: The entire Python process is submitted via `sbatch -W`; PySCF runs
  inside the Slurm job.

The SLURM script (`{project}.sbatch`) sets `OMP_NUM_THREADS` and `PYTHONPATH`
and runs the skill agent as a module (`service/templates/run_templates.py`):

```bash
{python_exec} -m {module} {config_file} local
```

where `{module}` is `pyrai3md.skills.pyscf_skill.agent` and `{python_exec}` is
`sys.executable`. Local execution stays in-process: the local runner calls
`dispatch_calculation()` directly and tees PySCF's stdout to the log file.

---

`keep_tmp` is applied at the same point on both arms: 0 removes the
calculation directory once the results are collected, 1 keeps the log, the
checkpoint and the molden file (#306, and the lifecycle contract of
`dev/SKILL_APP_DEVELOPMENT.md` section 5.5). The next local step rebuilds
the directory before it writes, so clearing mid-run cannot strand a stepped
workflow; `tests/test_keep_tmp.py` measures both directions.

## 6. Test Cases

`pyscf_skill/tests/` ships 9 YAML cases (ethylene C2H4):

`dft_energy`, `dft_gradient`, `dft_hessian`, `dft_ir`, `dft_nmr`,
`hf_energy`, `tda_energy`, `tda_gradient`, `tda_uv_vis`.

Generate agent configs with:

```bash
# Agent configs come from the qualification converter
# (qualification/certification/converter.py: parse_test_yaml / generate_test_configs),
# which replaced the former per-skill dev_tool.py; the test modes run through
# `pyrai3md test`.
```

---

## 6.1 Final Structure / Velocity Channels

Skills that iterate internally (geometry optimization, MD) report their outcome
through dedicated result channels:

- **fin_coord** `(natom, 3)`, Angstrom: final coordinates after skill-internal
  iteration (optimize/md/...); empty means no internal iteration happened.
  The framework `Calculator._apply_skill_geometry()` consumes this channel and
  immediately **pops** the key from both the returned result dict and the
  task-config results stored by the backend, so it never leaks into downstream
  workflow data.
- **opt_converge**: boolean convergence flag reported by optimize profiles;
  the finalize step settles the optimization outcome from it.
- **computed_charges** `(ns, na)`: atom charges computed by the skill (from
  `mf.analyze()` population analysis, per state), extracted by the framework
  to update background charges.
- **fin_velocity** `(natom, 3)`, Bohr/au: final velocities after skill-internal
  MD propagation. Reserved channel — currently always empty (no velocity
  output).

---

## 7. Key Differences from Other Backends

| Aspect | molcas_skill | pyscf_skill |
|--------|--------------|-------------|
| Execution | External subprocess | In-process Python import |
| Input format | Text module chain | No input files (API calls) |
| Output parsing | Log file parsing | In-memory object extraction |
| SOC support | Dual RASSCF + RASSI | Not supported |
| NAC support | RASSI | Not supported |
| Analysis | Limited | MO, ESP, population, IR, NMR |
| Excited states | CASSCF/CASPT2 | TDDFT/TDA |
| Orbital management | File-based (JobIph) | In-memory (no files) |
