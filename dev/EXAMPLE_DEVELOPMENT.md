# PyRAI3MD Examples Framework Design

Last updated: 2026-11-06 13:05. Section 7.10 is new: the six `adaptive_*` inputs - single method, ONIOM2 mechanical and QMMM mechanical, each in a local and a slurm carrier - with their stage-flag difference, their shared fixtures (`nn_ethylene_init/` as a directory, `egs_ethylene_100.json`, the two `.init.xyz`, the `c2h4_water25` box) and their measured local readings: all three local carriers finish at `n_iterations: 2` with five trajectories, five labels and `data_iter-2_105.json`, and only the QMMM one writes an `aux-labels.json`. Section 8's family list and its posture were re-measured rather than inherited: 201 YAMLs on disk, six families (12/33/25/32/37/6 stems), `47 true` against `154 false` case keys, and **610 planned rows - 111 main + 248 example + 226 skill + 25 benchmark - 384 closed by switches, 230 printed** on this checkout with the repository `.venv`. The six new YAMLs also gained the three-line topology header section 6 requires - 177 of 201 carried it and these six were among the 24 that did not; each was re-parsed through `parse_input -> build_config -> normalize -> validate` after the header was added. Two integration criteria the new examples legitimately broke were corrected instead of bypassed, in `qualification/integration/test_integration_example_dispatch.py`: the shipped-ensemble check no longer hardcodes a six-atom frame (it now takes the count from the example declaring that title, and a per-frame shape variant was dropped after measurement showed `read_init_xyz` already refuses a disagreeing header), and the dispatch-pool membership list went from eight sections to eleven with the three `adaptive_*_slurm` carriers. Both suites then read green: verification 3332 OK (skipped=1) in 68 s, integration 123 tests with 7218 subtests in 1578 s. Earlier: 2026-11-05 15:00 (#461 adds the ONIOM true-embedding set: ten `multiscale_*` inputs (base sp/md, the Molcas sp/nac pair, meci/neb, and the two ONIOM3 periodic legs) and six `multicenter_oniom*` inputs, all with a real point-charge consumer; the openqp embedding cases stay bookkeeping controls. Counts re-measured: 195 YAMLs, 30 `multicenter_*` of which 14 are QMMM and 12 are ONIOM (six neutral, six charged), 37 `multiscale_*`. Earlier: 2026-11-05 11:30 (#460 brings three more: `multicenter_qmmm_mecp.yaml` (each center's own pair under a crossing search), `multicenter_qmmm_md_fssh.yaml` (one FSSH ledger per center, engine couplings) and `multicenter_qmmm_ex_opt.yaml` (excited-state minimization on the per-center ladder) - 20 s / 42 s / 18 s locally, the pair search reading `cross_gap 0.57950573` and the minimization `Energy -152.45222204`. Counts re-measured: 173 YAMLs, 18 `multicenter_*` of which 14 are QMMM, 27 `multiscale_*`. Earlier: 2026-11-05 07:55 (#454/#455/#456/#457/#458 bring six more inputs into the multi-center QMMM family - `multicenter_qmmm_nac_sp.yaml` (each center answers its own derivative coupling), `multicenter_qmmm_npt.yaml` (the box under a barostat whose stress comes from the whole-system duty alone), `multicenter_qmmm_cell.yaml` (the framework optimizer relaxing those six entries) and the checkpoint pair `multicenter_qmmm_md_restart.yaml` / `multicenter_qmmm_md_restart_base.yaml` (two processes against one, same seed and same cadence), `multicenter_qmmm_neb.yaml` (the band over the two centers the endpoints actually move) - with their local cold-stage readings (19 s completed / 41 s completed / 34 s ending at `maxit` / 22 s + 56 s for the pair, whose four frames, per-frame charges, velocities, populations and energy table are equal digit for digit / 2 min 14 s for the band, which ends at `maxit: 2` with a five-point profile and a 0.1253435285 Hartree barrier) and their engine-free queue checks in `test_multicenter_qmmm.py` and `test_core.py`. Counts re-measured off the disk (at that time): 170 YAMLs, 15 `multicenter_*` of which 11 are QMMM, 27 `multiscale_*`. No test reading for those six yet. Earlier: 2026-11-04 15:20 (#452 divides those cases three ways by charge tier - converged fixed point, one fitted round, no inter-center polarization - and the readings are re-measured; #451: three `multicenter_qmmm_*` cases join the directory and the qualified list - the QM/MM ports of `multicenter_pbc_nve.yaml`, carrying the multi-center charge iteration - with their input shape, engine status and local readings (26 s / 29 s / 30 s, and the 1.23e-4 Hartree one fitted round plus the 8.7e-7 the fixed point adds, on that box). Counts re-measured off the disk: 164 YAMLs, 9 `multicenter_*` of which 5 are QMMM (three points/trajectory plus the band and the curvature carrier), 27 `multiscale_*`. Section 7.9's per-case table and the wall-time rows were not re-measured this round. Earlier: 2026-11-02 22:42 (#449/#450: two more examples join - `md_molcas_fssh_twospin.yaml` (the framework-driven 10-step FSSH on a ladder that spans two spin manifolds, whose every step asks the grouped deck; round 18 green in both carriers) and `multiscale_qmmm_embedding_nac.yaml` (the QMMM single point that asks for embedding and NAC together, i.e. the two-pass deck; round 21 green in both carriers, engine `2 &ESPF / 4 &RASSCF / Happy landing`). The directory now holds 158 YAMLs and the multi-scale family 27. Before it, 2026-11-02 12:05 (#448: `md_molcas_fssh_nac.yaml` joins the directory - 157 YAMLs on disk then - and the FSSH section gained its paragraph: same molecule as `md_molcas_fssh`, `nactype: nac` instead of kTDC, three pairs declared in a non-sorted order, `profile: casscf_properties` with `jobtype: energy gradient nac`. Only that section and the count were touched; the multi-scale section and the wall-time rows stand as previously measured. Before it, 2026-10-31 23:20 (round `tmp_20261031_220625` ran the whole multi-scale band green in the test environment, 328 PASS / 0 FAIL, `multiscale_qmmm_neb` included; the recorded wall times are on its row. Before it, 2026-10-31 18:20 (`multiscale_qmmm_neb` joins the family, making it twenty-six cases, and every shipped LAMMPS example now declares `ewald_accuracy` explicitly - what that key decides is in `skills/LAMMPS_SKILL_DEVELOPMENT.md` §12.2, and the reason the classical cases spell `1e-4` rather than inherit is the 1.3-2x it costs a periodic trajectory. Earlier the same day, 03:15: section 7.9 was rewritten against the shipped directory and re-checked: twenty-five `multiscale_*` cases, the family switch `examples.multiscale_test` on `condition` rather than `true` so the per-case keys still decide, and one table row per case. Three corrections today, all read off the test round `tmp_20261031_005246` rather than inferred: `multiscale_qmmm_mecp_cluster` names two singlets of one CAS(2,2) space and not a singlet/triplet pair (Molcas under ESPF returns one state per spin, see `methods/QMMM_DEVELOPMENT.md` §10); `multiscale_oniom2_orca_charge_mep` runs on `method: dft`, split out of the row it shared with the meci case, because a single-state tddft run derives `%tddft nroots 0` and ORCA refuses that while scanning the input; and the new periodicity paragraph states the declaration rule together with its gate `test_example_periodicity_declaration.py`, keeping the two defects of that round separate. The rest of the document was not re-audited.) Before it, 2026-10-22 19:10 (the round-shape paragraph now describes the shipped path posture: `path_test: true` with `optimize_test: false`, thirty-two path cases, twenty-five optimize cases; the previous paragraph described the optimize posture). Before it, 2026-10-20 07:50 (the two per-example-budget asides now record that the `_example_timeout` mechanism was deleted on 2026-10-20; the rest was not re-audited. Earlier, 2026-10-19 12:10: #392 added three inputs to this directory - `optimize_cell_iso.yaml`, `optimize_cell_meci_lammps.yaml`, `optimize_cell_mecp_lammps.yaml` - and one carrier asset trio, `c2h4_water25.data` with its `.in.init` and `.in.settings`; counts re-measured off the disk: 96 YAMLs, 24 `optimize_*`, 16 of them variable-cell, and the `examples` section shipping 21 `true` keys (the family key plus its twenty named cases) against 76 `false` and two `condition`. The line this one replaces carried a broken sentence (`...not re-audited line by linegainst #338;...`); its two claims - the third family switch, and the `md_lammps_nve: true` exception no longer being in the file - are kept, because section 7.3 and section 8 still state them. Section 7.5's two scan arms and Section 8 were reconciled against the code and the local-arm runs of #319 item 十; the multi-center sections, the two-center periodic runs of #317 and the 60-frame remote round `tmp_20261009_180119` that closed #320 stand as previously checked; the rest of the document follows as written)

---

## 1. Purpose

The `pyrai3md/examples/` directory stores the official YAML workflow examples of PyRAI3MD.

The examples live inside the pyrai3md package (`pyrai3md/examples/`) rather than at the repository root. This keeps them shipped together with the installed package (`setup.py` `package_data`), so qualification can always discover them through a package-relative path, both in source trees and pip installations.

Unlike ordinary software examples, these YAML files serve multiple purposes simultaneously. They are:

- runnable workflow examples for users;
- reference YAML configurations demonstrating the new input format;
- qualification inputs used by the Qualification framework;
- regression cases for future software development.

Therefore, the `pyrai3md/examples/` directory becomes the single source of truth for official workflow configurations.

---

## 2. Design Philosophy

The examples are **workflow-oriented**, not documentation-oriented.

Each YAML represents one complete runnable workflow. The examples are intentionally minimal. Detailed explanations of all input options belong to the User Manual rather than the example files.

Users are expected to:

```text
pyrai3md/examples/*.yaml
        ↓
  modify parameters
        ↓
    run pyrai3md
```

instead of writing input files from scratch.

---

## 3. YAML Input Support

The Interface layer (`pyrai3md/interface/parser.py`) detects `.yaml`/`.yml` files by extension and parses them with `yaml.safe_load`.

YAML examples use the nested section format aligned with the classic input sections:

```text
control / molecule / md / environment / <skill short name>
```

Top-level section names are normalized to lowercase, identical to the JSON handling.

Atomic coordinates are written inline in the YAML (top-level `atoms` and `coordinates`), so every example stays a single self-contained file.

---

## 4. Relationship with Other Modules

```text
pyrai3md/examples/
        │
        │ Official workflow YAMLs
        ▼

qualification/
        │
        │ Manager discovers YAML files
        ▼

    Execute
        │
        ▼

  Verification
```

The Qualification framework never maintains its own input files. It directly discovers and executes YAML files in `pyrai3md/examples/` through the package-relative path `pyrai3md/qualification` → `pyrai3md/examples`.

For the complete qualification pipeline and example test policy, see [QUALIFICATION_DEVELOPMENT.md](QUALIFICATION_DEVELOPMENT.md) Section 6.6.

---

## 5. Relationship with Runtime

The user only interacts with YAML. The remaining conversion is performed internally:

```text
YAML
 ↓
PyRAIConfig
 ↓
PyRAIState
 ↓
Runtime
```

Users are not expected to understand PyRAIConfig, runtime topology, or the internal task graph.

For the complete data flow and state construction, see [ARCHITECTURE.md](ARCHITECTURE.md) Section 3.4.5 and [DATA_CORE_DEVELOPMENT.md](DATA_CORE_DEVELOPMENT.md).

---

## 6. Directory Layout and Naming

The directory intentionally keeps a flat structure. Subdirectories are avoided. Each filename directly represents one workflow. This keeps browsing simple and consistent with Skill test YAMLs.

The full directory position within the project is defined in [ARCHITECTURE.md](ARCHITECTURE.md).

Examples are named by **workflow** rather than by software module.

Every example starts with a three-line header comment recording its execution topology, e.g.:

```text
# Topology: Dispatcher local / Workflow single / Calculator local
# Run: pyrai3md md_local.yaml
```

The topology keywords (Dispatcher local/slurm, Workflow single/parallel/dispatch/iterative, Calculator local) mirror the execution model described in [ARCHITECTURE.md](ARCHITECTURE.md).

Good names:

- `energy.yaml`
- `gradient.yaml`
- `hessian_num.yaml`
- `optimize.yaml`
- `frequency.yaml`
- `md_local.yaml`
- `md_parallel.yaml`
- `md_batch.yaml`
- `adaptive.yaml`
- `train.yaml`

Avoid names such as `example_energy.yaml` or `test_energy.yaml` because the directory itself already indicates they are examples.

---

## 7. Initial Example Set

### 7.1 Basic Workflows

#### energy.yaml

**Purpose**: Single-point energy calculation.

**Qualification**: Runtime execution, Calculator initialization, Backend execution.

#### gradient.yaml

**Purpose**: Gradient calculation.

**Qualification**: Runtime execution, Gradient backend.

### 7.2 Molecular Dynamics Workflows

#### md_local.yaml

**Purpose**: Serial local molecular dynamics.

**Example** (converted to pyrai3md yaml format):

```yaml
control:
  title: md_local_example
  jobtype: md
  theory: [pyscf]

md:
  initcond: read
  ninitcond: 1
  sfhp: nosh
  step: 10
  size: 20.67
  temperature: 300
```

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | single |
| Calculator | local |

**Current status**: YAML generated (converted to pyrai3md yaml format) and qualified: it is in `QUALIFIED_EXAMPLES`, so both entry modes are planned and executed.

#### md_parallel.yaml

**Starting states come from a shipped ensemble.** These three cases read `<title>.init.xyz` beside the YAML (`initcond: read`), so `ninitcond: 3` takes the first three of five Boltzmann-sampled frames and the trajectories are genuinely different runs. Before this they sat on the default `initcond: 0`, which repeats one loaded state `ninitcond` times: a qualification round found the three trajectories' energy records byte-identical, which made the "worst case over conditions" reading vacuous. The examples' own `coordinates:` were first moved to the HF/STO-3G equilibrium structure (the shipped one sat 12.4 mHa above it, C=C 1.3302 A against 1.3058 A, so a sampled frame could come out *lower* in energy than the centre it was drawn around); the frames were then drawn at 300 K from an HF/STO-3G harmonic analysis of that equilibrium, recorded through `pyscf_skill/service/hessian.py` and sampled through `workflow/sampling.py`, and each sampled structure was re-evaluated before shipping so a wrong mode convention (mass weighting, normalization) would show up as an unphysical distortion energy rather than pass silently. The runtime looks the file up under the run title, so the three examples ship three copies of the same five frames; the planner stages them into each run directory the way it stages the LAMMPS topology files, and `setup.py` ships `examples/*.init.xyz` -- missing either registration is invisible locally and fatal once installed. Measured remotely (round `tmp_20261007_093843`, 593/593 PASS): the three fan-out trajectories write three different records (md5 `7b2ee118 / c89499af / 65144ca5`) with `etot_drift` 8.553e-05 / 2.814e-05 / 2.635e-05 against the declared 1e-3, and each of the six modes reports the worst of them -- the same number the same ensemble produces on a laptop, which is what says the shipped file is the one that ran.

**Purpose**: Multiple trajectories executed using local multiprocessing.

**Example** (converted to pyrai3md yaml format):

```yaml
control:
  title: md_parallel_example
  jobtype: md
  theory: [pyscf]

md:
  initcond: read
  ninitcond: 3
  sfhp: nosh
  step: 10
  size: 20.67
  temperature: 300
  multitask: true
  multipool: 3
```

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | multitask + multipool (in-process pool) |
| Calculator | local |

The fan-out keywords live in the **workflow** section: `&md multitask`
opens the pool, `&md multipool` sizes it and `&md remotetask` ships each
trajectory as a remote child runtime. The framework `&environment`
section declares no execution switch at all: `execution_mode` is a
**skill** keyword (see above), and `workflow_execution` /
`workflow_workers` are refused on input
(`schema.config.RETIRE_ENVIRONMENT_KEYS`, #299 and #300) because they
read as a setting while nothing consumed them. A test that needs the
engine off-process writes `execution_mode` in the skill section; a run
that asks the framework to submit gets `execution_mode=local` handed to
the skill by `data.core._build_task_queue` so exactly one component
submits. `qualification/verification/test_example_dispatch.py` pins the
graph flags, the refusal and the repo-wide absence of the retired keys.

**Current status**: YAML generated (converted to pyrai3md yaml format) and qualified: it is in `QUALIFIED_EXAMPLES`, so both entry modes are planned and executed.

#### md_batch.yaml

**Purpose**: Multiple trajectories executed through Slurm dispatch.

**Example** (converted to pyrai3md yaml format):

```yaml
control:
  title: md_batch_example
  jobtype: md
  theory: [pyscf]

md:
  initcond: read
  ninitcond: 3
  sfhp: nosh
  step: 10
  size: 20.67
  temperature: 300
  multitask: true
  remotetask: true
  multipool: 3

environment:
  slurm_partition: default
  slurm_time: '01:00:00'
```

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | slurm |
| Workflow | remotetask + multitask + multipool (remote child runtimes) |
| Calculator | local (worker) |

**Current status**: YAML generated (converted to pyrai3md yaml format) and qualified: it is in `QUALIFIED_EXAMPLES`, so both entry modes are planned and executed.

#### md_bagel_fssh.yaml / md_molcas_fssh.yaml / md_molcas_fssh_nac.yaml / md_openqp_fssh.yaml / md_openqp_fssh_soc.yaml / md_orca_fssh.yaml

**Purpose**: Framework-driven nonadiabatic molecular dynamics (FSSH surface hopping with kTDC energy-curvature coupling) on the C2H4 molecule, one example per electronic-structure skill: BAGEL XMS-CASPT2, Molcas CASSCF, OpenQP MRSF, and ORCA TDDFT. `md_openqp_fssh_soc.yaml` is the fifth of these and the only one that adds `&md dosoc`: the same OpenQP MRSF engine, but two singlets and one triplet (`ci "2 1"`, `spin "0 1"`) so the plan carries two singlet-triplet coupling pairs, and every step returns energies, gradients and couplings from one two-manifold `runtype=data` job.

**Example** (md section shared by the first four; skill sections carry method/basis and the CI parameters):

```yaml
control:
  title: md_bagel_fssh_example
  jobtype: md
  theory: [bagel]

molecule:
  charges: 0
  multiplicity: 1
  ci: "3"
  spin: "0"
  coupling: "1 2, 2 3, 1 3"

md:
  root: "2"
  initcond: 0
  ninitcond: 1
  sfhp: fssh
  nactype: ktdc
  step: 10
  size: 20.67
```

The default `control.driver: main` keeps the propagation in the framework AIMD loop, which calls the skill for per-step energies and gradients; kTDC estimates the nonadiabatic coupling from the energy curvature. `md.root` is 1-based in every external spelling, so `root: "2"` starts the trajectory on S1. All ethylene MD examples share one geometry and one Maxwell-Boltzmann 298.15 K velocity set (fixed seed, center-of-mass translation removed, Bohr/au). The Molcas example pins `profile: casscf_gradient` in its skill section so jobtype `md` resolves to the per-step property profile instead of the internal DYNAMIX propagation profile (`casscf_abinitio_dynamics`), which would double-advance the geometry against the framework integrator.

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | single |
| Calculator | local |

**Current status**: Qualified in both modes (yaml + classic input); disabled by default in pmd_env.yaml (requires bagel / molcas / openqp / orca on the execution host).

**md_molcas_fssh_nac.yaml** (#448) is the molcas sibling with the engine answering
the couplings: `md.nactype: nac` instead of kTDC, three pairs of one CAS space
declared as `coupling: "2 3, 1 2, 1 3"` - deliberately **not** the sorted ladder -
and `profile: casscf_properties` with `jobtype: energy gradient nac` so each step
returns energies, gradients and the derivative couplings. The point of the odd
declaration order is the reading it forces: the framework indexes the returned NAC
rows by the declaration's order, so a skill that answered in its own print order
would silently hop on the wrong surface. A sibling with three pairs in sorted order
cannot tell a remap from an identity map, which is why the kTDC case above does not
cover this. What it takes to serve it - `molcas_skill` on a two-manifold-free
CASSCF table, one `&ALASKA` pass per requested pair - is the shape round 15
(`tmp_20261102_114641`) reads; the reading itself is recorded in the CHANGELOG
entry, not here.

**md_openqp_fssh_soc.yaml** is qualified by a measured 5-step run on OpenQP 1.3.1 with the shipped geometry and velocities: framework and skill both record `nsoc 2` with `soc_pair_map [[1, 2], [0, 2]]`, each step's deck is `runtype=data` carrying `ispher=false`, `nstate_s=2` / `nstate_t=1`, `grad_s=2` and `soc=true` with no `nac` keyword, and the last step's 6-row `socme` (2 singlets x 1 triplet x 3 Ms) returns `results['soc'] = [0.0943, 0.8181]` cm-1 in that pair order. Only the followed state gets a gradient (norm 0.110462 hartree/bohr); the other two rows stay zero because no selector asked for them. It is switched off in `pmd_env.yaml` by default because OpenQP 1.2 writes neither `socme` nor the labelled ladder exports, so a 1.2 host cannot serve it.

#### md_lammps_nve.yaml / md_lammps_nvt_*.yaml / md_lammps_npt_*.yaml

**Purpose**: Framework-driven classical ensembles on a 60-atom TIP3P water box with LAMMPS as the force backend (ff_gradient provides energy/gradient, plus stress for NPT). The 12 examples are exactly the legal matrix of `schema/validate.py` (`BAROSTAT_THERMOSTATS`): NVE (no coupling at all), NVT with the nose-hoover, nose-hoover-chain, and csvr thermostats, and NPT with each barostat combined with the thermostats its own derivation allows -- berendsen and scr with any of the three, parrinello-rahman and mttk with nose-hoover or nose-hoover-chain only. That last restriction is not a preference: it is what Martyna-Tobias-Klein (JCP 101, 4177 (1994)) assume, and Bernetti & Bussi (JCP 153, 114107 (2020)) state the opposite freedom for their scheme outright. So csvr×mttk and csvr×parrinello-rahman do not exist here, and `md_lammps_npt_csvr_scr.yaml` is the stochastic thermostat's variable-cell slot.

`md_lammps_nve.yaml` is the sentinel of the set: it is the only LAMMPS example whose total energy is a constant of the motion, so a broken force handoff shows up as a frozen box with a flat Etot instead of being masked by a thermostat that injects energy regardless (the zero-gradient defect that the other eleven examples passed silently). Its Etot drift is judged by the qualification runner like every other NVE example.

**Example** (md section of md_lammps_npt_nhc_mttk.yaml):

```yaml
molecule:
  charges: 0
  multiplicity: 1
  pbc: [1, 1, 1]
  cell: [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]

md:
  initcond: 0
  ninitcond: 1
  sfhp: nosh
  step: 2000
  size: 41.34
  direct: 1
  buffer: 100
  ensemble: npt
  thermostat: nose-hoover-chain
  barostat: mttk
  temperature: 300
  thermostat_tau: 100
  thermostat_chain: 3
  pressure: 1.0
  barostat_tau: 10000
  compressibility: 4.5e-5
  barostat_mode: anisotropic

lammps:
  method: ff
  lammps_shake: 0
  python_enabled: true
```

Every `md_lammps_*` example also carries `control: debug: true`, and that
switch belongs to the example rather than to the physics. It says "a bound this
run crossed should end the run", which is what a qualification round wants and
what a production trajectory does not: a default-off run reports the same
violation and keeps integrating, and the split between what may stop anyone and
what only stops a test is `dev/methods/ENSEMBLE_DEVELOPMENT.md` section 8.11.
The classic `.input` that the qualification's second mode runs is generated from
the same field, so both modes interrupt at the same step -- a keyword missing
from `_build_control_pairs` would make yaml mode abort while input mode silently
continued, the same class of gap `_MD_ECHO_KEYS` documents for the MD section.

The `barostat_tau` above is 10000 fs because this sample is anisotropic. A
liquid cell has no shear modulus, so the ~10 kbar instantaneous deviatoric
stress of a 20-molecule box walks the shape degrees of freedom; measured against
the stress a run that finishes actually recorded, tau_p 1000 drifts the cell
axes 51-80% and trips the strain gate at steps 875-1479, while 10000 drifts them
21-29% and the run completes. The three berendsen examples keep 1000
deliberately: weak coupling is a contraction with no cell momentum to
accumulate, those already finish, and `_berendsen` does read `barostat_tau`, so
raising it would only slow something that works.

All twelve share one inline Maxwell-Boltzmann velocity block (300 K, seed 20260926, center-of-mass translation removed, Bohr/au, instantaneous temperature 300.5 K over N_df = 177), so the qualification modes start from identical conditions and the thermostats are measured against an already-equilibrated kinetic energy rather than heating a frozen box.

**Length, and what it costs the judgements.** `step: 2000` at `size: 41.34` au (1 fs) is a 2 ps run recording **21 frames** (`direct 1` writes step 1, `buffer 100` then every 100th step, and the last step is forced: steps 1, 101, …, 1901, 2000 at t = 0, 10, …, 1900, 1999 fs). In the test environment the whole example takes **12 s** of wall time (`PyRAI3MD End: … Total: 00:00:00:12`, measured on `tmp_20260928_184646`), which is the 4.20 ms per step of #244 over the loop plus startup, so every example fits the global `_timeout` backstop and the per-example budgets these twelve used to carry are gone (the mechanism itself was deleted on 2026-10-20). A run that silently falls back to the subprocess path costs 950 ms per step, so the same 2000 steps need ~1900 s and are stopped at 1195 s: no resident service reports a timeout rather than passing on a timetable nothing planned for.

That budget was re-checked on 2026-09-29 against the only example the gate lets through (`md_lammps_nve`, the other eleven `md_lammps_*` being switched off in pmd_env.yaml): both modes finished in **12 s** with the step measured at **4.696 ms** (yaml) and **4.879 ms** (classic input), the run's own products recording `hello` and `init` once each against `evaluate` twice thousand, and the last-frame `Epot/Ekin/Etot/T_inst` identical between the two modes to the printed digit. The 12% distance from #244's 4.20 ms is a single sample against a five-pair median and is recorded as that, not as a trend.

21 frames is below `RATE_MIN_FRAMES` (50), and that is a consequence worth stating rather than discovering. The NVE sentinel can only separate a systematic leak from the bounded kinetic-potential exchange by fitting a slope, and 21 points is not enough for one, so `md_lammps_nve` is judged on the **band** (max minus min): the measured `etot_drift` 9.740e-04 against the 1e-03 tolerance leaves 2.6% margin, and a band is a looser test than a slope. Since #246 the verdict says which test it used - `[ENERGY CONSERVATION] judged on 21 frames by band: … (rate needs 50 frames)` - because a pass that reads identically whether it was judged strictly or loosely is not a verdict. Raising `step` to 10000 would give the rate judgement its 101 frames back at 46 s per example; the current length is the user's decision (#245), and the trace exists so the trade-off is visible in the product instead of in a footnote.

`lammps_shake: 0` is stated explicitly rather than left to the default: these examples run under `driver: main`, so the framework integrator moves the atoms and a constraint has to be declared on that side, in the `constraint` block (`dev/methods/CONSTRAINT_DEVELOPMENT.md` §3.7). A shake fix in the LAMMPS input would have no propagation to hold, and `schema/validate.py` refuses the combination rather than letting it pass as if constrained. Without the constraint the O–H stretch (period ≈ 14 fs for the fixture's `k = 450 kcal/mol/A^2`) stays resolved at 1 fs.

`python_enabled: true` is in all twelve examples for the same reason the examples are worth running at all: a framework-driven MD step is one `ff_gradient` call per step, and the subprocess path pays a full LAMMPS launch (input deck, `read_data`, kspace setup) 2000 times per run to evaluate one geometry. The resident service starts once and takes the new coordinates over its socket each step. Two consequences are recorded rather than assumed. First, the switch is declared in the example, not inherited from `pmd_env.yaml`, so `cd pyrai3md/examples && pyrai3md md_lammps_nvt_nh.yaml` behaves the same way as the qualification task does; under `execution_mode: slurm` the skill ignores it, because there each step is its own batch job. Second, the two execution paths are not bit-identical: the subprocess path reads forces out of a text dump at six significant digits, while the resident path takes them straight from the engine interface, so a resident step carries strictly more of the number. `tests/test_persistent_test_plan.py` pins that the keyword reaches the skill as a resident request through both example routes (the YAML and the generated classic input), since a key the framework owns no field for can be dropped by either hand.

The LAMMPS backend resolves its launch-directory topology as `<project>.data`; the shared h2o fixtures (`h2o.data`, `h2o.in.init`, `h2o.in.settings`) ship inside `pyrai3md/examples/` (declared in setup.py package_data) so a direct `cd pyrai3md/examples && pyrai3md md_lammps_nvt_nh.yaml` finds them next to the YAML, and qualification additionally stages project-renamed copies from `skills/lammps_skill/tests/` into the example run directory (`manager._stage_example_skill_fixtures`, honoring an overridden `lammps_project`). The fixtures are never modified: the backend compresses them into a JSON force-field model at initialization (`skills/LAMMPS_SKILL_DEVELOPMENT.md` §5.6), which is what made a loop that relaunches LAMMPS once per step affordable before the resident path above existed, and what still lets a resident session load the same model once at `INIT`.

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | single |
| Calculator | local |

**Current status**: Qualified in both modes (yaml + classic input); requires LAMMPS on the execution host. `md_lammps_nve` is switched on in pmd_env.yaml, the other eleven are switched off -- see Section 8 for what that gate does and does not mean.

### 7.3 Geometry Optimization Workflows

#### optimize_minima.yaml

**Purpose**: Ground-state geometry optimization with the Minima optimizer (BFGS + fixed trust radius).

**Qualification**: Runtime execution, iterative workflow loop, Calculator synchronization, optimizer convergence.

**Current status**: Qualified in both modes (yaml + classic input).

#### optimize_gdiis.yaml

**Purpose**: Ground-state geometry optimization with the GDIIS optimizer (DIIS extrapolation + BFGS Hessian).

**Qualification**: Same scope as `optimize_minima.yaml`, covering the alternative optimizer.

**Current status**: Qualified in both modes (yaml + classic input).

#### optimize_tda.yaml

**Purpose**: Excited-state (S1) geometry optimization of ethylene using TDA gradients, demonstrating the state-aware optimization flow (`state_i: 1`).

Note: the ethylene S1 (ππ\*) surface is repulsive along the C-C stretch and has no bound minimum; this example exercises the state selection and gradient-extraction flow with a bounded step limit (`maxit: 30`) rather than asserting a physical minimum.

**Qualification**: State-resolved result arrays, `state_i` → `target_states` forwarding, excited-state gradient extraction.

**Current status**: Qualified in both modes (yaml + classic input).

#### optimize_ts.yaml

**Purpose**: Transition-state search (`optimize.method: ts`) taking its initial curvature from the framework's own displaced-gradient batch, the producer every backend can serve.

**Qualification**: The partitioned-RFO step and its mode following, the curvature request issued before the first step, the negative-curvature count read in the rigid-projected subspace, and the saddle itself: 32 evaluations to -91.56509828334428 Hartree (HF/sto-3g, bridging HCN/HNC start), one imaginary mode (-1248.42 cm-1) on the endpoint's own frequency job.

**Current status**: Qualified, YAML mode; needs PySCF only (`pmd_env.yaml` `optimize_ts: true`).

#### optimize_ts_skill_hess.yaml

**Purpose**: The same search with the other producer, `&hessian hess_driver: skill`, which asks the backend for the matrix it computes.

**Qualification**: The dispatched single `frequency` child job (its own mode pinned so an engine without a Hessian profile fails by name instead of starting a displacement batch in the worker), and the equivalence of the two producers: same 32 evaluations, -91.56509828335417 against -91.56509828334428, one imaginary mode at -1248.60 cm-1.

**Current status**: Qualified, YAML mode; needs PySCF only.

#### optimize_ts_multiscale.yaml

**Purpose**: A saddle search over a single-center composite: one program on two layers (`theory: [pyscf, pyscf]`), three tasks per evaluation (`qm`, `qm2`, `qm2_task1`).

**Qualification**: The binding identity that allows one skill at two positions (`pyscf_skill` plus `pyscf_skill@qm2`), the numerical batch differencing the *assembled* total gradient with its per-term contribution signs, and the reduction check - with the model region covering the whole system the composite reaches the single-layer number to the printed digit, in the same 32 evaluations, one imaginary mode at -1248.69 cm-1.

**Current status**: Qualified, YAML mode; needs PySCF only. Refuses a multi-center partition, and that refusal is the other half of what this file exists to keep honest.

#### optimize_partial_minima.yaml

**Purpose**: A ground-state relaxation with a fragment held rigid, declared through the `fix` constraint the dynamics path already uses - the framework keeps one statement of "what may move", not a second selector.

**Qualification**: The step, the gradient history the curvature update is fitted to, and the trust radius all read the tangential part the criteria judge on (#337). 27 evaluations at -91.6424015 Hartree, against 23 at -91.6446781 with the constraint block removed, and the pinned atoms bit-identical from the first frame to the last. Before #337 the pinned run never converged (101 evaluations, `max_g 6.2e-2` on the free atom).

**Current status**: Qualified, YAML mode; needs PySCF only.

#### optimize_partial_ts.yaml

**Purpose**: A saddle search with one atom held by the anchor, on a six-atom system where the pinned fragment is a real half of the reaction rather than a spectator.

**Qualification**: The frame the freeze produces. Fifteen components are movable, three of the six whole-body motions are absorbed by the anchor and cost nothing, and the twelve that remain - including the partial translations the anchor opposes - are what the search steps through; taking out all six instead leaves nine directions and no way to reach the stationary point (#338). Measured on this input: 75 evaluations to -149.8362656686 Hartree, `Negative Modes (model)` at 1 from the fourteenth step, the held oxygen returned with components identical to the input, and one imaginary frequency (-3064.0 cm-1) of twelve on the endpoint's own `num_freq` job. The same dimer searched with nothing pinned needs 116 evaluations and lands 7.2e-8 Hartree away.

**Current status**: Qualified, YAML mode; needs PySCF only. Switched off by default (`pmd_env.yaml` `optimize_partial_ts: false`) on cost - six atoms over seventy-five evaluations plus a curvature batch on fifteen components - not on doubt.

For the optimizer algorithms and the state selection design, see [methods/OPTIMIZE_DEVELOPMENT.md](methods/OPTIMIZE_DEVELOPMENT.md).

### 7.4 Multi-Center Workflows

#### multicenter_gradient.yaml

**Purpose**: Two-center ONIOM energy + gradient. Two high-level centers (atoms 1-6 and 7-12, each a C2H4) are treated with openqp MRSF (3 singlets S0/S1/S2 per center, all three NAC pairs 1-2/2-3/1-3, initial state S1, no SOC); the full 60-atom system (the two centers plus a water environment) is the low-level ONIOM baseline computed with xtb (GFN2).

**Multi-center input convention**: the center-dimensioned molecule/md keywords use the classic `;`-separated string form (docs [MULTICENTER_DEVELOPMENT.md](MULTICENTER_DEVELOPMENT.md) section 4.2), which keeps every external index, `root` included, 1-based unambiguously across both qualification entry points:

```yaml
control:
  jobtype: gradient
  theory: [openqp, xtb]      # layer order: 1st = high, 2nd = low
molecule:
  highlevel: "1-6; 7-12"      # two centers (1-based atoms)
  ci: "3; 3"                  # 3 states per center
  spin: "0; 0"                # singlet per center
  coupling: "1 2, 2 3, 1 3; 1 2, 2 3, 1 3"   # per-center NAC pairs (1-based)
md:
  root: "1; 1"                # per-center initial state S0 (root is 1-based)
```

A two-layer multi-center ONIOM requires two distinct theory skills (`resolve_composite_type` rejects a single-skill multi-center input); the same program at two methods is not yet supported. The string form is used instead of nested YAML lists because a typed nested `coupling` list is interpreted as already-normalized 0-based by the schema adapter, whereas the classic string is parsed as external 1-based (docs [MULTICENTER_DEVELOPMENT.md](MULTICENTER_DEVELOPMENT.md) section 4.3).

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | single |
| Calculator | local (multi-center ONIOM2) |

**Qualification**: Multi-center schema/normalize/validate, ONIOM2 task expansion (1 + 2C terms), per-center electronic binding, composite energy/gradient aggregation, and the converter round-trip of the center fields plus the two skill sections.

**Current status**: Qualified in both modes (yaml + classic input). Requires openqp and xtb installed on the execution host (the classic-input mode is generated by the certification converter, which preserves the center strings and both skill sections; see [QUALIFICATION_DEVELOPMENT.md](QUALIFICATION_DEVELOPMENT.md) Section 6.3/6.6).

#### multicenter_md.yaml

**Purpose**: Multi-center ONIOM nonadiabatic molecular dynamics: the same two openqp MRSF C2H4 centers in the xtb water environment as multicenter_gradient.yaml, propagated by the framework AIMD loop while the FSSH electronic stepper hops each center independently (kTDC coupling from the per-center energy curvature).

**Example** (center-dimensioned md keywords; root is 1-based per center):

```yaml
control:
  title: multicenter_md_example
  jobtype: md
  theory: [openqp, xtb]

md:
  root: "1; 2"
  initcond: 0
  ninitcond: 1
  sfhp: fssh
  nactype: ktdc
  step: 10
  size: 20.67
```

`root: "1; 2"` starts center 1 on its S1 and center 2 on its S2; with C > 1 there is no global current_state, and the per-center active configuration drives the ONIOM task routing.

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | single |
| Calculator | local (multi-center ONIOM2) |

**Qualification**: The C=2 AIMD loop with the FSSH ElectronicStepper dispatch is covered end to end by `qualification/verification/test_multicenter_md.py`; the example additionally qualifies the converter round-trip of the center strings together with the non-default dynamics keywords (sfhp/nactype/root).

**Current status**: Qualified in both modes (yaml + classic input); disabled by default in pmd_env.yaml. Requires openqp and xtb installed on the execution host.

#### multicenter_pbc_nve.yaml / multicenter_pbc_npt.yaml

**Purpose**: Two centers inside a periodic box, propagated by the framework AIMD loop with two real engines: PySCF at the centers (HF/STO-3G on one water each) and LAMMPS as the whole-system provider *and* as the low layer of each center's correction (TIP3P, the shipped 60-atom `h2o.data`). Both centers straddle a box face, so the region each is cut from comes out of the continuous view rebuilt from the frame's crossing counts rather than out of the main image, and each center's `L(c)` is a **subsystem** the skill cuts out of the user's system-level templates and evaluates non-periodically (dev/methods/PBC_DEVELOPMENT.md section 4.2, dev/skills/LAMMPS_SKILL_DEVELOPMENT.md section 5.6a).

**Example** (the two differ only in the ensemble block; the NVE one fixes the box):

```yaml
control:
  title: multicenter_pbc_nve_example
  jobtype: md
  theory: [pyscf, lammps]
  gl_seed: 20261008
  debug: true

atoms: [O, H, H, ... 60 symbols, 20 waters]
coordinates:               # the shipped box, atom order identical to h2o.data
  - [0.3668257, 4.0129757, 7.0641557]
  - ...                    # 60 rows; rows 10-12 and 22-24 are the centers

molecule:
  charges: 0
  multiplicity: 1
  highlevel: [[10, 11, 12], [22, 23, 24]]
  pbc: [1, 1, 1]
  cell: [8.399091, 8.399091, 8.399091, 90.0, 90.0, 90.0]

md:
  step: 60         # 50 in the NPT example
  size: 41.34
  direct: 1
  buffer: 1        # buffer 10 in the NPT example
  randvelo: 1
  temperature: 300
  ensemble: nve    # npt + nose-hoover-chain/berendsen in the sibling

lammps:
  lammps_project: h2o    # one system-level topology, cut per task
```

**Why two centers can now sit in one LAMMPS box.** A 2-layer center plan is
`H(c) - L(c) + L(real)`, so the low layer is asked for every center slice as well as the
whole box. LAMMPS reads its topology from a `<project>.data` template and does not generate
one, and until #317 a slice smaller than that template was refused where the template is
read (`LAMMPS: atom id 4 in h2o.data is out of range [1, 3]`, which is why these two inputs
were originally built on a single water instead). The skill now cuts and compresses the
user's templates per task from the row numbers the framework hands it, and the cut-out
subsystem is evaluated non-periodically: `boundary f f f`, no `kspace_style`, `…/coul/long`
pair styles replaced by their real-space siblings at the template's own cutoffs. What that
buys, in the number this section exists for: keeping the periodic treatment would have
over-subtracted 0.280090 kcal/mol (**4.464e-4 Ha**) per center.

Order is not something this input has to get right: `schema/adapter.py` canonicalises
each center's indices to ascending for every input carrier, the framework hands the row
numbers in that order, and the skill renumbers its template rows by position (never by
value, which is what keeps atom id `i + 1` on coordinate row `i`). Both the
canonicalisation and the skill's obedience are pinned in tests -
`TestTheRegionTaskIsHandedItsRows` (plan level, three carriers) and
`skills/lammps_skill/tests/test_ff_model.py::TestSelection` (the positional rule).

**Qualification** and the readings each run leaves (2026-10-09; the local run
and the test round `tmp_20261009_180119` agree to every digit the round
prints, except that the slope differs in its fifth digit -- 2.90637e-06 Hartree
per frame here against 2.90623e-06 there, a per-step trajectory difference
between the two builds' engines, three orders of magnitude under the 1e-05
tolerance it is judged against):

| Reading | NVE (60 steps, 11 s) | NPT (50 steps, 8 s) |
|---|---|---|
| plan | 1 + 2C = 5 tasks: 1 LAMMPS whole-system, 2 PySCF centers, 2 LAMMPS center corrections | same |
| rows handed to the low layer | `[]`, `[10, 11, 12]`, `[22, 23, 24]` | same |
| `L(c)` against three independent re-evaluations of the same cluster | 0.003072046652927 and 0.001408396300691 Ha, agreement 1.6e-16 Ha | same mechanism |
| what the isolated term contains | the analytic bond-and-angle sum of that water, to 2e-15 kcal/mol | same |
| container a center is evaluated in | one fixed +/-50 A box with `boundary f f f`, written by `input_builder._subsystem_box` | same |
| energy conservation | judged on the trajectory total over 60 recorded frames: slope **2.906e-06** Ha/frame against the 1e-05 tolerance, band 1.512e-03 Ha. The widest per-center band, **8.822657e-03** Ha, is reported as `etot_center_drift` and does not decide | n/a (a barostat does work on the box) |
| `[DIAG region_reconstruction]` lines | none (C>1, real engines, fixed cell) | none (C>1, real engines, deforming cell) |
| cell edges, first to last | fixed at 8.398804 A | 8.398804 to 8.371907 A; V **-0.96%** |
| barostat pressure | n/a | `Pvir_bar` -6050.718 bar at step 1, +275.981 at the last; `Pint_bar` -2277.314 to +4701.684 bar; `T_inst` 274.5 to 318.7 K |
| `calculator_run` per step | 187-250 ms (two repeats) | 107-122 ms (two repeats) |

**Two of these numbers are the record of a falsified rule, so they read as
decisions.** The container: #317 fitted a subsystem's box to the template's own
edge lengths and slid it under the cluster, which is a statement about where the
cluster is *at load time*. In the test environment the straddling center had walked out
of it at the framework's step 2 and multi-rank domain decomposition lost the atom
outright (`ERROR on proc 0: Bond atoms 1 3 missing on proc 0`, `src/ntopo_bond_all.cpp:59`,
against a request the framework logged as `step=2 natom=3 box=None`) -- not a
refusal, an MPI_ABORT. A fixed box around the origin is not a guess, so nothing
walks out of it; the only geometry that is refused is a coordinate beyond
+/-50 A, which is no longer a cluster of this cell. The length: 4 recorded frames
is below `RATE_MIN_FRAMES` (50), so the run was judged on the band and the worst
band in the merge came from a center's own record. 60 frames put the strict slope
in charge, and once it was in charge it showed what the band had been hiding --
the composite drifts at 2.906e-06 Ha per frame while a center exchanges 8.822657e-03
Ha with the box, which is what mechanical embedding is supposed to do. That is why
`runner.py` now judges the trajectory total only (dev/QUALIFICATION_DEVELOPMENT.md
section 6.6); the tolerance itself was not touched.

The pressure rows are the point of the NPT sibling, and they moved with #318: the
`stress` request used to reach the PySCF center tasks (which have no cell) and not the
whole-system LAMMPS task, so the barostat was integrating its kinetic term alone and the
same input inflated by +0.97% instead of contracting. A cell that drifts the wrong way
still drifts, which is why this reading, not the earlier one, is what the run has to carry.
Per-step cost is quoted as a range because two repeats of one input differ by more than the
two ensembles differ from each other - the QM SCF dominates it. What stayed put: the
60-atom box now supplies a real virial, the region is re-cut from the cell of each
instant, and both runs are clean under `control.debug`.

**Execution topology**:

| Component | Mode |
|-----------|------|
| Dispatcher | local |
| Workflow | single |
| Calculator | local (multi-center ONIOM2, periodic outer task) |

**Current status**: Qualified in both modes (yaml + classic input); disabled by default in pmd_env.yaml, because the host has to carry PySCF **and** LAMMPS. The shape of what they run is pinned by `TestTheShippedPeriodicMultiCenterExample` and `TestTheRegionTaskIsHandedItsRows` in `qualification/verification/test_multicenter_periodic.py`, and the topology fixture they read (`h2o.data`, `h2o.in.init`, `h2o.in.settings` -- the same system-level files the standalone LAMMPS examples use) is declared in `setup.py` and guarded by `qualification/verification/test_skill_packaging.py`.

---

### 7.5 Machine-Learning Workflows

Three examples cover the three ML jobtypes. They are the first examples
whose input file is a *label file* rather than a structure, and the first
pair that has to run in a declared order.

#### nn_train.yaml and nn_predict.yaml -- one chain

**Purpose**: fit a small energy+gradient ensemble from the shipped label
file, then evaluate a trained surface against reference values.

**Why two files, and why they share a directory**: `nn_predict` reads the
model folder `nn_train` writes. Every other example gets a run directory
of its own per (example, mode) precisely so a failure keeps its own log;
a chain is the declared exception -- `manager.py:EXAMPLE_CHAINS` maps the
family `nn` to the ordered members `('nn_train', 'nn_predict')`, and the
planner gives both the same directory within a mode
(`examples/nn/yaml`, `examples/nn/input`) and plans the fit first. An
alphabetical listing would put `nn_predict` first, which is why the order
is declared rather than derived.

**No dependency is propagated at run time.** If the fit fails, the
evaluation fails on its missing `modeldir`, and the skill's own error
already names the fix ("Train the model with jobtype 'train' first"). A
skip invented here would hide that sentence behind a second opinion.

**Qualification**: the `Train` and `Prediction` nodes, the `&train` /
`&prediction` sections, the training record and the verdict banner, the
test-set record and its raw files, chain planning and shared-directory
staging.

#### nn_grid_search.yaml / nn_grid_search_local.yaml

**Purpose**: vary two declared hyperparameters over a 2 x 2 grid, train one
ensemble per point, rank the points by their validation error.

The scan ships as **two arms of one plan**, because a point is a model fit
and a fit is what a cluster is for:

| Arm | Dispatch | Named for |
|-----|----------|-----------|
| `nn_grid_search.yaml` | `multitask: true`, `remotetask: true`, `multipool: 4` | the default: each point becomes its own child runtime and is submitted with `sbatch` |
| `nn_grid_search_local.yaml` | `multitask: true`, no `remotetask`, `multipool: 2` | the daily gate on a machine with no scheduler |

Both state the same `grid`, `sort_by` and `max_grid`, and differ in the
project name and grid root so their products cannot collide. Only the remote
arm carries a `slurm:` section: it is the allocation those child runtimes ask
the cluster for, and the same section is rendered into the classic input as
`&SLURM` (see [INPUT_SYSTEM_DEVELOPMENT.md](INPUT_SYSTEM_DEVELOPMENT.md)
section 7.5).

**Qualification**: the skill's `searchable:` declaration read through the
query interface, `&search grid` through both carriers (a dropped section
would plan an empty scan and still report a completed run), per-point model
directories named by their own values, `summary.csv` and the ranked table.

**The remote arm is a named skip, not a failure, off-cluster.** The planner
reads the submission request from the example's own input
(`manager._example_submits_remotely`, which looks for a truthy `remotetask`
in any section rather than at a file-name convention) and skips the task with
a reason naming `sbatch` when the host has no cluster client. An example that
never says it submits is a local run whether it fails or is skipped. The arm
is still checked here down to the artifact a submission leaves: with `sbatch`
and `sacct` shimmed, the four points produce four runtime payloads and four
scripts carrying the example's directives (`--mem=4G`, `--time=0:20:00` from
the input over the host's own mem/time, and no `--exclusive` line), and end
to end only in the test environment.

**Cost, measured on this environment** (Python 3.11.16, TensorFlow 2.21.0, the
24-frame / 18-atom `egs_small.json`): the fit 18 s wall, the test-set
evaluation 8 s, and the 4-point scan on the local arm's worker pool of 2 at
**31.5 s / 32.4 s / 35.9 s** wall over three runs of the same input (12.4-12.7 s
per point, four rows `ok`), 58-62 s serial. The same scan reached the same
table through the generated classic input as well: the plan's
`example_nn_grid_search_local_input` run ends `Status: completed` in 34 s with
`Number of search: 4` and four `ok` rows in `summary.csv`. All three modes of
the train/predict pair and both modes of both scan arms were run against the
YAML and the classic text.

**No per-example budget, and no mechanism left to carry one.** The global
1200 s backstop is more than two orders above the slowest of them, so a
per-example budget would be a number nothing measured. The `_example_timeout`
map was deleted on 2026-10-20; `_timeout` is the only budget and every example
answers to it.

**Fixture**: `egs_small.json`, shipped beside the examples and staged into
each run directory (`.json` was added to the example-side staging suffixes,
and deliberately not to the skill-side ones, where a generated
`<workflow>.json` intermediate would be swept into project renaming).
`setup.py` declares `examples/*.json`, and
`test_skill_packaging.py::test_every_shipped_example_sidecar_is_declared`
now checks the rule for every non-YAML file in the directory rather than a
named list, so the next fixture cannot be shipped and forgotten.

For the training design see [methods/TRAIN_DEVELOPMENT.md](methods/TRAIN_DEVELOPMENT.md);
for the scan see [methods/SEARCH_DEVELOPMENT.md](methods/SEARCH_DEVELOPMENT.md).

### 7.9 Multiscale composite cases (`multiscale_*`)

Thirty-seven cases carry the `multiscale_` prefix. The family switch
`examples.multiscale_test` is `condition`, because the ONIOM
charge-embedding copies need ORCA/PySCF (and LAMMPS for the three-layer
ones) on the host and a single `true` would run them there regardless.
Each case runs in both the YAML and the classic-input carrier:

| case stem | scheme | embedding / charge source |
|---|---|---|
| `multiscale_oniom2_no_embedding_{sp,md}` | ONIOM2 openqp+xtb, gas-phase ethylene + water | none |
| `multiscale_oniom2_embedding_{sp,md}` | same | openqp writes no point charges; layered-bookkeeping control, not an electrostatic measurement |
| `multiscale_oniom2_read_charge_{sp,md}` | same | static `{title}.charge` (3 outer atoms) |
| `multiscale_oniom3_no_embedding_{sp,md}` | ONIOM3 openqp+xtb+lammps, 60-atom water box | none |
| `multiscale_oniom3_embedding_{sp,md}` | same | openqp writes no point charges; layered-bookkeeping control |
| `multiscale_oniom3_read_charge_{sp,md}` | same | static `{title}.charge` (3 midlevel atoms) |
| `multiscale_qmmm_embedding_{sp,md}` | QMMM molcas XMS-CASPT2 + lammps, 60-atom water box | field V/E (Molcas ESPF potential) |
| `multiscale_qmmm_no_embedding_{sp,md}` | same | isolated QM, `E_classic = U_main - U_gas` |
| `multiscale_qmmm_meci` / `_mecp` | QMMM optimize, molcas CAS-SCF + lammps on `c2h4_water25` (1581 atoms, ethylene centre) | field V/E; crossing search, `maxit: 2` |
| `multiscale_qmmm_mecp_cluster` | QMMM optimize, molcas CAS-SCF + lammps on the six-atom cluster | field V/E; two singlets of one CAS(2,2) space |
| `multiscale_qmmm_ts` | QMMM optimize, molcas CAS-SCF + lammps on the six-atom cluster | field V/E; MM atoms held by `fix`, numerical composite Hessian |
| `multiscale_qmmm_mep` | QMMM path, molcas CAS-SCF + lammps on the 60-atom box | field V/E; three-image band, `maxit: 2` |
| `multiscale_qmmm_neb` | QMMM band, molcas CAS-SCF + lammps on the 60-atom box | 3 images between `path_h2o_reactant.xyz` and `path_h2o_product.xyz`, `maxit: 2`; subprocess mode on purpose, a band dispatches every image of a round while the resident service is keyed one per project. Ran on round `tmp_20261031_220625` in both carriers, 1:14 and 1:12 wall, ending on `Maximum optimization steps reached` - the expected stop for `maxit: 2`, and the row is judged on its structure |
| `multiscale_oniom2_orca_charge_meci` | ONIOM2 orca TDDFT + xtb GFN-FF on the 60-atom box | computed outer charges (`embedding: true`) |
| `multiscale_oniom2_orca_charge_mep` | ONIOM2 orca **DFT** + xtb GFN-FF on the 60-atom box | computed outer charges; one surface, so `method: dft` (see the `nroots` rule in [ORCA_SKILL_DEVELOPMENT.md](skills/ORCA_SKILL_DEVELOPMENT.md) §4) |
| `multiscale_oniom2_bagel_charge_{mecp,ts}` | ONIOM2 bagel CAS-SCF + xtb GFN-FF on the six-atom cluster | computed outer charges; `fix` + numerical Hessian on the ts case |
| `multiscale_oniom2_orca_embedding_{sp,md,meci,neb}` | ONIOM2 orca high + pyscf HF outer on the six-atom cluster | true embedding: ORCA `%pointcharges`; ORCA drops NACME under charges, so `nac` is the Molcas carrier |
| `multiscale_oniom2_orca_no_embedding_sp` | same as `sp` | A/B control, `embedding: false`, no `%pointcharges` |
| `multiscale_oniom2_molcas_{embedding,no_embedding}_sp` | ONIOM2 molcas CASSCF + pyscf HF | Molcas `&ESPF External = N Angstrom`; A/B pair |
| `multiscale_oniom2_molcas_embedding_nac` | ONIOM2 molcas CASSCF + pyscf HF | two-pass ESPF + field-free NAC |
| `multiscale_oniom3_orca_embedding_{npt,cell}` | ONIOM3 orca + pyscf + lammps on the 60-atom box | mid shell into the ORCA `.pc`; stress from the whole-system LAMMPS |

A crossing case names two states, and both carriers of every crossing case
declare `state_i`/`state_j`: the optimizer refuses a pair that names the same
surface twice (`opt_state_i and opt_state_j must name two different surfaces`),
and the default leaves both at 0. The two states also have to be reachable by
one provider call, which today means one spin manifold — Molcas under ESPF
returns one state per spin, so a singlet/triplet pair arrives with a surface
missing (see [QMMM_DEVELOPMENT.md](methods/QMMM_DEVELOPMENT.md) §10).

**Periodicity is declared, not inherited from the topology.** `pbc` is the
authority on whether a system is periodic; a LAMMPS `lammps_project` file that
carries a box does not make the run periodic. Every case therefore writes `pbc`
explicitly — `[1, 1, 1]` plus the topology's own `cell` for the three periodic
QMMM boxes, `[0, 0, 0]` for the cluster cases — and
`qualification/verification/test_example_periodicity_declaration.py` is the
gate: it checks that nothing stays silent, that a declared `cell` really is the
box its topology was built for, and that the molecular cases name a topology
that does have a box, so `[0, 0, 0]` is a decision about that box rather than
an absence of one. Omitting the declaration crashes nothing — the data template
still states the box and the settings tail still re-issues `kspace_style` — so
what is lost is everything keyed on the declaration, which for a QM/MM field
probe is the pinned Ewald split and the kspace channel of its two tallies. The
A/B split mismatch measured in the test environment round `tmp_20261031_005246` had a
second, independent cause on the skill side (`_coul_cutoff` could not read a
cutoff from a plain `pair_style` line, so the pin was never written); both were
fixed in the same round, and the doc records them as two defects because a
declaration cannot substitute for a cutoff the skill failed to read.

ONIOM2 with a quantum outer method is the case that binds a store outside
the three-layer form: the outer region is every atom outside highlevel, the
`embedding` form fills it from theory[1]'s computed charges in the normal
two-phase order, and `read_charge` loads it once from the title charge
file. With several centers the outer region is each center's own complement,
and one whole-system `L(real)` row fills every environment (a per-evaluation
array, not the persistent store; dev/methods/ONIOM_DEVELOPMENT.md §7). The
read-charge cases ship a `{title}.charge` beside the YAML; the manager stages
`.charge` files, `setup.py` declares `examples/*.charge`, and
`test_skill_packaging.py` checks the sidecar rule for them too. The
true-embedding A/B pairs are the electrostatic reading: the embedded deck
carries `%pointcharges` and the control does not, and the two totals differ
by the field term (measured `-8.24013e-5` Hartree on the six-atom cluster).

The QMMM cases need Molcas + LAMMPS; the older ONIOM bookkeeping cases need
OpenQP + xTB, and the true-embedding set needs ORCA + PySCF (plus LAMMPS for
the three-layer legs; the Molcas pair is engine-gated). The multi-center
ONIOM carriers (neutral and charged) are the `multicenter_oniom*` stems; their
per-center charges and readings are in
[methods/ONIOM_DEVELOPMENT.md](methods/ONIOM_DEVELOPMENT.md) sections 7-8.
They are listed in `QUALIFIED_EXAMPLES` and run in both carriers when the
family switch is open.

#### multicenter_qmmm_sp.yaml / multicenter_qmmm_sc_charge.yaml / multicenter_qmmm_md.yaml

**Purpose**: The QM/MM ports of `multicenter_pbc_nve.yaml` - the same 20-water
TIP3P box, the same two face-straddling water centers (rows 10-12 and 22-24),
the same `h2o` LAMMPS topology - with Molcas CASSCF for each center instead of
PySCF, and the additive QM/MM coupling of section 7.9. They divide the charge
question three ways: `sp` fits the 0-order charges once (`sc_charge_maxiter: 1`)
and solves in that polarized field, `sc_charge` iterates the field against them
to a fixed point, and `md` runs neither - the trajectory is about the geometry
walk, so its probes carry uncharged centers and its queue is five duties.

**Configuration**:

```text
  title: multicenter_qmmm_sc_charge_example
  jobtype: energy gradient
  theory: [molcas, lammps]
  multiscale_type: qmmm
  molecule:
    highlevel: [[10, 11, 12], [22, 23, 24]]
    ci: "1; 1"        # one root per center
    charges: "0; 0"
    pbc: [1, 1, 1]
    cell: [8.399091, 8.399091, 8.399091, 90.0, 90.0, 90.0]
    embedding: true
    sc_charge: true
    sc_charge_maxiter: 5
    sc_charge_threshold: 1e-3
    # no q_qm: under several centers the classical stage is fed by what each
    # center's own ESPF fit reported, and a pinned set would name a field
    # nobody built with it
  lammps:
    lammps_project: h2o
    ewald_accuracy: 1e-8
```

**Current status**: Qualified in both carriers in the test environment host (round
`tmp_20261104_124614`), and switched on by name in `pmd_env.yaml` rather than by
the family default, because the host has to carry Molcas **and** LAMMPS. Their
input shape - the queue the plan expands, the probe each center narrows to, the
per-center partition, the absence of `q_qm`, and the classic `;`-spelling of the
three new keys - is pinned without engines by
`qualification/verification/test_multicenter_qmmm.py` (the load pipeline plus
the coupling and search-axis refusals).

#### multicenter_qmmm_mep.yaml / multicenter_qmmm_num_freq.yaml

**Purpose**: The two channels that are built on top of one multi-center
evaluation rather than beside it. `_mep` grows a band whose every image is such
an evaluation (two probes, one fitted charge round per center, two property
solves, one classical surface) and keeps ONE fitted round, like `md`: a walk
whose forces are integrated has to name one electronic structure per image.
`_num_freq` takes the framework's numerical Hessian of the composite: each of
its gradient jobs is a full multi-center evaluation, and the rows taken are the
**mobile block** the constraint plan defines (`workflow/hessian.py::_active_rows`),
not a second keyword - holding all 54 classical atoms with `fix_atoms: 7-60`
leaves 6 atoms x 3 = 18 jobs instead of 180, which is the only shape of this
case that fits a qualification budget on a 60-atom box. The mask is global: it
is not split per center, so the matrix carries both centers' intra-block
curvature and the inter-center rows of the one surface it was taken on.

**Current status**: Qualified in both carriers, switched on by name like the
other multi-center QMMM cases (Molcas and LAMMPS on one host, and `_num_freq`
is the costlier of the pair). Their shape is pinned without engines by
`TestShippedMulticenterQmmmChannels` in
`qualification/verification/test_multicenter_qmmm.py` - the seven-duty queue
with one fitted round per band image, the five-duty queue with no charge duty
per displacement, and the 18-row mobile block read off the built state.

**Readings**: local host, same engines as above. `_num_freq` 7 min 31 s for its
18 gradient jobs on the corrected mobile block (the two centers; an earlier
revision froze `7-60` and left atoms 1-6 free, so the twelve modes recorded
there belonged to six atoms in neither QM region and are withdrawn - CHANGELOG
#458). The twelve modes now read `-2610.72 / -1717.53 / -190.04 / -87.89 /
112.35 / 142.05 / 207.24 / 329.83 / 1415.59 / 1933.79 / 2478.44 / 11388.92
cm-1` with zero-point energy 0.0410257424 Hartree; a six-atom block inside a
frozen periodic box is not a water spectrum, so the case is a curvature-assembly
carrier rather than a frequency one. In the test environment host it takes 10 min 47 s
of the 2400 s per-example budget (the old mask read 18 min 39 s of 1200 s). `_mep` runs two steps, with `path_status.txt`
carrying both rows and starting at the unpolarized single point of that
geometry (-152.5287129798) before the grown image. Its Etot is not judged:
`runner._embedding_breaks_conservation` states why a multi-center ESPF-embedded
composite re-fits its charges every image, so the recorded total is not a
constant of motion by construction.

**Readings**: measured on the local host (Molcas v26.10, LAMMPS 22 Jul 2025
Update 6, cold stage). `sp` with the single fitted round: 26 s, `E_QM
-152.3057801400` split `-76.1563741400 / -76.1494060000`, `E_classic
-0.2230559853`, total `-152.5288361253`. `sc_charge`: 29 s, two rounds
(`9.140667e-1 -> 7.000000e-4` against 1e-3, `stopped by tolerance`), total
`-152.5288369992`. The unpolarized baseline is the first frame of `md`
(`-152.5287129798`, 30 s for three frames), so on this box one fitted round
carries 1.23e-4 Hartree of inter-center polarization and the second round adds
8.7e-7. The full table with its identity checks is in
`methods/QMMM_DEVELOPMENT.md` section 9.

#### multicenter_qmmm_nac_sp.yaml / multicenter_qmmm_npt.yaml / multicenter_qmmm_cell.yaml

**Purpose**: The three channels that answer a question *per center* or *with the
box*, on the same two-center 20-water box as the five above. `_nac_sp` gives each
center its own derivative coupling on its own atoms: the composite keeps the two
blocks apart (nothing is summed across centers), which is what
`multiscale/composite.py::assemble_qmmm` now does by scattering each center's
rows into its own region instead of borrowing center 0's. `_npt` puts the box
under a barostat: `stress` rides the whole-system classical task alone
(`_SYSTEM_ONLY_CAPABILITIES`), the barostat reads it through
`state.active_stress()`, and each step hands the new cell to both probes and the
classical surface while the QM tasks stay non-periodic clusters. `_cell` is the
framework optimizer stepping the six lattice entries against that same stress -
the variable-cell half of the route, on a composite.

**Current status**: Qualified in both carriers and switched on by name, like the
rest of the family. Engine-free shape pinned in
`qualification/verification/test_multicenter_qmmm.py`; the box-reading that `_cell`
depends on is pinned separately in `test_optimizer.py::TestVariableCellStepKeeps
TheCellSpellings`, because a variable-cell step that writes only nine of the box's
spellings leaves the run's own log holding two cells at once (#456).

**Readings**: local host, cold stage. `_nac_sp` 19 s, `Status: completed`, `nac`
of shape `(2, 60, 3)` with pair 1 non-zero on rows 10-12 and pair 2 on rows 22-24
only, and `state.electronic` carrying one block per center. `_npt` 41 s,
`Status: completed`, three evaluations; the first step prints `P = -19080.66 bar`
and the box goes `8.39716243 -> 8.39429460` Angstrom. `_cell` 34 s, `maxit: 2`,
ending in the expected truncation (`failed: Maximum optimization steps reached`)
with the two energies `-152.52883613` and `-152.54156374` and the box strained to
`a: 8.395978182473 / b: 8.395517681227 / c: 8.397996023406` (volume 591.96284210
against 592.51160370 at the start).

#### multicenter_qmmm_mecp.yaml / multicenter_qmmm_md_fssh.yaml / multicenter_qmmm_ex_opt.yaml

**Purpose**: The three channels that close the matrix. `_mecp` runs a crossing
search over one pair per center (the composite returns two synthetic surfaces,
so the single-surface machinery works unchanged). `_md_fssh` runs one hopping
ledger per center, with each center answering its own coupling
(`nactype: nac`) and `root: "2; 2"` giving each its own starting state.
`_ex_opt` minimizes the first excited state of every center
(`state_i: [1, 1]`), which exercises the rule that a composite reports ONE
assembled surface and a search must read that surface instead of indexing it by
the state id.

**Current status**: All three registered and switched on. The per-center pairs
are what the input layer could not express before #460 - `opt_state_i/j` took
scalars only, and a per-center list was then forwarded into every skill config,
which rejects a list; the framework now drops those sequences from the skill
merge and keeps the scalars.

**Readings**: local host, cold stage. `_mecp` 20 s to its declared `maxit: 1`:
`cross_gap 0.57950573`, `gap_norm 0.20553932`, per-center `E_QM
-76.11922511 / -76.11188854`. `_md_fssh` 42 s, `Status: completed`, two frames,
the per-center populations and active states in `.md.pop`, no hop in those
frames (`fssh_force_hop 0`). `_ex_opt` 18 s to `maxit: 1`: `Energy
-152.45222204`, `Gradient RMS 0.01630798` against the ground-state `sp` value
`-152.52883613` on the same geometry. Why the crossing carrier is MECP and not
MECI is in `methods/QMMM_DEVELOPMENT.md` section 9: a MECI's branching plane
needs the coupling between the two summed surfaces, which is not a per-center
matrix element.

#### multicenter_qmmm_neb.yaml

**Purpose**: The band over two centers. Three images are interpolated between the
same two endpoint structures the single-center carrier `multiscale_qmmm_neb.yaml`
uses on this box, and every image is one multi-center evaluation. The two centers
are the two waters the endpoints displace (rows 1-3 and 4-6), so each center's own
`E_QM` follows the reaction coordinate instead of sitting out the band.

**Current status**: Registered and switched on. Subprocess mode, for the reason
the single-center case gives for the same carrier: a band evaluates every image of
a round at once while resident mode keys one LAMMPS service per project.

**Readings**: local host, cold stage, 2 min 14 s. Two NEB steps, ending in the
declared `maxit: 2` truncation. `path_profile.txt` carries five points
(reactant -152.5281727945, three images -152.4629558045 / -152.3878677587 /
-152.4558203436, product -152.5132112872) and `path_status.txt` two rows with
`barrier` 0.1272069139 then 0.1253435285 Hartree.

#### multicenter_qmmm_md_restart.yaml / multicenter_qmmm_md_restart_base.yaml

**Purpose**: The composite's checkpoint pair, on the same two-center box as the
cases above. `_restart` integrates two steps and writes `<title>.rt`; its second
process (the resume line is in the file's own header) adds two more. `_base` runs
those four steps in one process, with the same seed, timestep and checkpoint
cadence - the cadence is what decides which steps become frames, so it has to be
the same on both sides or the comparison is between two different recordings.
A composite has three things a plain run does not to carry across the split: the
charge store (the set fitted inside the step), one electronic carrier per center,
and an active configuration with one entry per center.

**Current status**: Both legs registered and switched on; the round runs each leg
as its own single process, and the resume half of the split stays a manual second
command, as it is for the constraint restart pair. The parity criterion is
local and reads off the products.

**Readings**: local host, cold stage. The two runs' products are equal digit for
digit once the title is normalized - `.md.xyz` (248 lines, four frames, each
carrying its own `Lattice=` row), `.md.charges`, `.md.velo`, `.md.pop`, and the
energy table including both per-center `E_QM` columns (`-152.5288361253278140`,
`-152.5314474984596700`, `-152.5271661157257199`, `-152.5135030487722929`).
Wall clock: 56 s for the four-step baseline, 22 s for the resumed process - a
checkpoint already holds the surface of the step it was taken at, so the second
process evaluates only the steps it adds.

### 7.10 Adaptive sampling cases (`adaptive_*`)

Six cases carry the `adaptive_` prefix: the same three systems in two
carriers each. `examples.adaptive_test` is the family switch above their six
keys and ships at `condition`, so the per-case keys decide (local on, slurm
off); the family and its three-valued decision are pinned by
`verification/test_manager.py::TestAdaptiveFamilySwitch`, and the shape of
each plan by `verification/test_adaptive_sampling.py::TestAdaptiveExamplePlans`.

Every case declares `maxiter: 2` (one labelled iteration), `ninitcond: 5` and
`step: 10`, `maxsample: 1` with all four thresholds at `0.0` (so the cap, not
a deviation test, decides how many frames each trajectory contributes),
`record_step: 4` and `keep_md: 1`. The two carriers differ only in the stage
flags: the local arm runs MD, reference and training inside the one process
(`md_multitask: true`, `md_multipool: 5`, `reference_multitask: true`,
`reference_remotetask: 0`, `remote_train: 0`), while the slurm arm asks for
`md_remotetask: true`, `reference_remotetask: 1` and `remote_train: 1` - the
trajectory becomes a submitted framework runtime, the reference becomes the
skill's own sbatch submission, and the training child is submitted too.

| Case | Theory queue | Reference queue | Partition | What it asks |
|---|---|---|---|---|
| `adaptive_single_{local,slurm}` | `[nn]` | `[pyscf]` | none (title `adaptive_ethylene`, six atoms) | the ordinary single-method shape: the label is the whole-system reference term |
| `adaptive_oniom2_{local,slurm}` | `[nn, lammps]`, `multiscale_type: oniom` | `[pyscf, lammps]` | `highlevel` = `highref` = 1576-1581 | a layered reference whose high layer covers exactly the NN surface, mechanical embedding (`embedding: false`) |
| `adaptive_qmmm_{local,slurm}` | `[nn, lammps]`, `multiscale_type: qmmm` | `[pyscf, lammps]` | same six-atom QM region | the QM region labelled by PySCF inside the LAMMPS box, mechanical embedding, and the QMMM auxiliary channels |

**What the two solution cases differ in.** Both declare the same six-atom
region on both sides, and their `&method alignment` records say so: the label
source is `layer` from the `qm` task in both. What separates them is the
scheme and what the scheme therefore publishes - ONIOM2 records
`reference_type: oniom2` with no auxiliary channel, QMMM records
`reference_type: qmmm`, `aux: [mm_energy, q_qm]` and the note that a reference
without a field probe produces no MM field. `schema/validate.py` warns on the
first of those ("ONIOM2 with a classical (MM) outer method and embedding=false
is the same additive combination as QMMM embedding=false"), and for the labels
that warning is literally right: the two local runs, one seed and one starting
batch, returned five frames of two states that agree bit for bit (`max
|difference| = 0.0` on both the energy and the gradient array). The case stays
because the label agreement is not what the family tests - the two spellings
are assembled by different routes, publish different channels, and record
different totals beside the same label.

**Fixtures.** All three systems share the pretrained ethylene ensemble
`nn_ethylene_init/` (named by `modeldir:`, so the staging function copies the
folder whole rather than a file) and the 100-frame initial dataset
`egs_ethylene_100.json` (named by `train_data:`). The single-method case starts
from `adaptive_ethylene.init.xyz` (six atoms with velocities); the two
solution cases start from `adaptive_c2h4water.init.xyz` and read the
1581-atom periodic box `c2h4_water25.data` with its `.in.init` /
`.in.settings`. All of these ship beside the YAMLs and are declared in
`setup.py`; `qualification/verification/test_skill_packaging.py` is the guard
that fails if one is left out of the package.

**Readings.** The three local carriers were run on this environment on 2026-11-06 as
direct `pyrai3md adaptive_<case>_local.yaml` invocations (the runner's example
wrapper is not the carrier here: its script calls `module` and `timeout`, and
neither is installed in the test environment). All three finished with exit 0 and the same
structural reading - `n_iterations: 2`, and in iteration 2 `ntraj` 5,
`selected` 5, `received` 5, `verified` 5, `new_frames` 5, `reference_failed` 0,
`reference_ran` and `trained` true, `iter-2/data_iter-2_105.json` (the 100
initial frames plus the five labels), five `ref_*` directories and a trained
`NN-<title>-iter-2/`. The sidecar is the one thing that differs, and it
differs exactly the way section 14 of `methods/ADAPTIVE_DEVELOPMENT.md` says
it must: `adaptive_oniom2_local` and `adaptive_single_local` write no
`aux-labels.json`, while `adaptive_qmmm_local` writes five rows carrying
`mm_energy` (-8.270269, -8.270695, -8.267929, -8.269680, -8.270237 Hartree)
and six `q_qm` values with no `field` row, because mechanical embedding asks
for no MM field.

The slurm carriers ran in the test environment in round `tmp_20261106_102745`, where
the file was narrowed to the adaptive band: **113 rows, 109 pass, 0 fail, 0
error, 453 skipped**, the six adaptive rows (3 cases in the yaml and the input
mode) passing, and `tools/verify.py` reporting the same six carriers over
three cases with every structural counter at 0. In every case the two carriers
of one system agree on the structural counts, and their recorded energies and
gradients match at `max |difference| = 0.0`.

Two shipped-set criteria in
`qualification/integration/test_integration_example_dispatch.py` had to be
corrected rather than worked around, and both are the kind a new example meets
again: `TestShippedInitialEnsembles` compared every `<title>.init.xyz` against
a hardcoded six-atom frame, so the 1581-atom box failed it (the criterion now
takes the count from the example that declares the title, and a per-frame shape
invariant was dropped after it was measured to be an identity - `read_init_xyz`
already refuses a row count that disagrees with its own header), and the
dispatch-pool membership list, which enumerates the sections stating a
`multitask`/`multipool`/`remotetask` trio, gained the three `_slurm` carriers
because `&adaptive` there declares the generic keys next to the per-stage ones
(the three `_local` carriers state no generic key, so they stay off that list).
The family's own criterion, `verification/test_manager.py::TestAdaptiveFamilySwitch`,
was written before those two fired; with all three in place the suites read
**3332 verification OK (skipped=1)** and **123 integration tests with 7218
subtests, both green** on this checkout.

## 8. Qualification Policy

Current qualification executes:

- `energy.yaml`
- `gradient.yaml`
- `hessian_num.yaml`
- `optimize_minima.yaml`
- `optimize_gdiis.yaml`
- `optimize_tda.yaml`
- `optimize_ts.yaml` / `optimize_ts_skill_hess.yaml` / `optimize_ts_multiscale.yaml`
- `optimize_partial_minima.yaml` / `optimize_partial_ts.yaml`
- `multicenter_gradient.yaml`
- `multicenter_md.yaml`
- `multicenter_pbc_nve.yaml` / `multicenter_pbc_npt.yaml`
- `multicenter_qmmm_sp.yaml` / `multicenter_qmmm_sc_charge.yaml` /
  `multicenter_qmmm_md.yaml`
- `multicenter_qmmm_mep.yaml` / `multicenter_qmmm_num_freq.yaml`
- `multicenter_qmmm_nac_sp.yaml`
- `multicenter_qmmm_npt.yaml` / `multicenter_qmmm_cell.yaml`
- `multicenter_qmmm_neb.yaml`
- `multicenter_qmmm_mecp.yaml`
- `multicenter_qmmm_md_fssh.yaml` / `multicenter_qmmm_ex_opt.yaml`
- `multicenter_qmmm_md_restart.yaml` / `multicenter_qmmm_md_restart_base.yaml`
- `md_bagel_fssh.yaml`
- `md_molcas_fssh.yaml`
- `md_openqp_fssh.yaml`
- `md_openqp_fssh_soc.yaml`
- `md_orca_fssh.yaml`
- `md_lammps_nve.yaml`
- `md_lammps_nvt_nh.yaml` / `md_lammps_nvt_nhc.yaml` / `md_lammps_nvt_csvr.yaml`
- `md_lammps_npt_nh_berendsen.yaml` / `md_lammps_npt_nh_pr.yaml` / `md_lammps_npt_nh_mttk.yaml`
- `md_lammps_npt_nhc_berendsen.yaml` / `md_lammps_npt_nhc_pr.yaml` / `md_lammps_npt_nhc_mttk.yaml`
- `md_lammps_npt_csvr_berendsen.yaml` / `md_lammps_npt_csvr_scr.yaml`
- `nn_train.yaml` / `nn_predict.yaml` (one chain, one directory)
- `nn_grid_search.yaml` (submits its points to Slurm) /
  `nn_grid_search_local.yaml` (the same scan on a local worker pool)
- `adaptive_single_{local,slurm}.yaml` / `adaptive_oniom2_{local,slurm}.yaml`
  / `adaptive_qmmm_{local,slurm}.yaml` (section 7.10)

The policy is implemented by `QUALIFIED_EXAMPLES` in `qualification/manager.py`.

Each qualified example is rendered through the converter and exercised
through the two PyRAI3MD entry points that take a user-authored input file:

```text
YAML → flat JSON config → classic text input + xyz
```

| Mode | Command | Description |
|------|---------|-------------|
| pyrai3md_yaml | `pyrai3md <yaml>` | YAML input pipeline |
| pyrai3md_input | `pyrai3md <input>` | Classic text input pipeline |

These are PyRAI3MD runtime qualifications (task `skill` field: `examples`). The
**task** is always started locally: the qualification runner never wraps an
example in a batch script of its own (`use_slurm: False` on every example
task). What an example does *inside* that process is its own declaration -- a
scan or an MD batch may well ask `remotetask` and submit its own child
runtimes to Slurm, which is a different submission, made by the framework
dispatch layer from the run's `slurm` section. So "executed locally" names the
wrapper, not the run, and the host needs a cluster client for exactly those
examples that say so (see section 7.5's remote arm and `manager._example_submits_remotely`). Both modes are **framework entry points** (the YAML pipeline and the classic-text pipeline): examples test PyRAI3MD computation tasks, never a simulated agent-calling-skill scenario, so neither the `pyrai3md_skill` nor the `pyrai3md_agent` mode is planned for them — the agent `.config` is a machine-rendered artifact rather than a user input, and planning it would add a third of the example suite for no extra parser coverage. The agent loader pipeline stays qualified through the skill tests, which run all three modes. The flat `.json` next to each example is only the converter intermediate used to generate `.input`, it is never executed.

Every YAML in `pyrai3md/examples/` is listed in the qualification plan; examples outside `QUALIFIED_EXAMPLES` are registered as skipped reference workflows with the reason "Qualification postponed".

The reference dispatcher examples (`md_local`, `md_parallel`, `md_batch`) are qualified now: they were the last three held out while the Dispatcher and Workflow runtime settled, and the round that closed #308 took them into `QUALIFIED_EXAMPLES`. `md_local` runs one trajectory serially on the host, `md_parallel` fans three out through a local thread pool, and `md_batch` fans them out through `remotetask`, so each trajectory is a submitted Slurm child -- that is the one example whose execution needs a working cluster client on the execution host, and the one to switch off in the pmd_env.yaml `examples` section if the submission environment misbehaves. Nothing in `pyrai3md/examples/` is postponed any more, but the branch that postpones an unlisted YAML stays live and is pinned by `test_manager.py::test_an_unqualified_example_is_still_postponed`, so the next example that needs a runtime nobody has stabilized is still reported rather than silently absent.

The qualified dynamics examples (the four `md_*_fssh` NAMD examples, the twelve `md_lammps_*` ensemble examples, `multicenter_md`, and the two `multicenter_pbc_*` periodic center examples) are listed in `QUALIFIED_EXAMPLES` but carry `false` switches in the `examples` section of pmd_env.yaml, because they require the corresponding electronic-structure / MD software on the execution host. Enable them individually, through their family switch (`md_lammps_test` for the twelve `md_lammps_*` cases, `constraint_md_test` for the thirty-three `constraint_md_*` ones, `optimize_test` for the twenty-five `optimize_*` ones, `path_test` for the thirty-two `path_*` cases, `multiscale_test` for the thirty-seven `multiscale_*` ones and `adaptive_test` for the six `adaptive_*` ones - six families in `manager.EXAMPLE_FAMILIES`; the shipped file leaves `multiscale_test` and `adaptive_test` at `condition` so their per-case keys decide, and closes the other four with `false`), or through the `_runall` master switch -- and read that last clause carefully, because `_runall: true` does not merely allow the enabled ones, it **forces the disabled ones on** (`manager._plan_example_tests`, pinned by `test_manager.py::test_runall_forces_disabled_example_on`). The family asks by name: a new `optimize_*.yaml` joins every optimize round without touching pmd_env.yaml, and `constraint_opt_periodic`/`constraint_opt_surface` stay outside it even though their input says `jobtype: optimize`.

What the file says today (re-measured on 2026-11-06 against the shipped `pmd_env.yaml`, through the runner's own counters `_is_closed` and `_folded_count` on this checkout with the repository `.venv`): `_runall: condition`, `examples.md_lammps_test`/`constraint_md_test`/`optimize_test`/`path_test` `false`, `examples.multiscale_test` and `examples.adaptive_test` `condition`, and 47 case keys `true` against 154 `false`. That posture plans **610 rows - 111 main + 248 example + 226 skill + 25 benchmark - 384 of them closed by switches, and 230 printed** (101 main + 95 example + 33 skill + 1 benchmark, each bucket folding its closed rows into one row). The adaptive six sit inside that: the three `_local` keys are `true` and the three `_slurm` keys `false`, so a host without a cluster client still runs the local half of the family and the slurm half is a named skip rather than a failure. The posture it carried before this one was the path round's (`path_test: true`, `optimize_test: false`, every non-path case closed by its own key), and before that the optimize round's; the readings those two postures produced - 478/213/268, then 500/213/290, and the test rounds `tmp_20261021_195813`, `tmp_20261022_084519` and `tmp_20261022_151713` that ran them - stay in Section 4.2 of QUALIFICATION_DEVELOPMENT.md as the record of those postures, not of this one. Earlier still the file carried one deliberate exception, `md_lammps_nve: true` against the other eleven `md_lammps_*` `false`, justified as a measurement surface: the LAMMPS examples are cheap on the resident service and expensive without it (#244 measured milliseconds against hundreds of milliseconds per step), so one example was enough to prove the hot-start path end to end in both input modes. That justification is unchanged and sits in the comment above the key, switched off for this round rather than withdrawn. The eleven switched-off examples still carry `python_enabled: true` and still ship in the package: **declaring a capability and gating its execution are two different things**, and a reader who concludes from this paragraph that twelve runs happen per round has it wrong -- what runs per round is whatever the `examples` section allows, and that section is the machine's own configuration, editable without touching an example.

Example runs also stage shared skill topology fixtures into the example run directory: `manager._stage_example_skill_fixtures` copies the `.data` / `.in.init` / `.in.settings` fixtures from each theory skill's `tests/` directory (the same contract as the skill-certification converter), additionally renaming them to the injected `<stem>_project` name that the backend resolves. Files beside an example YAML carrying one of `_EXAMPLE_FIXTURE_SUFFIXES` (`.json`, `.charge`, `.xyz`, `.freq.molden` on top of the three topology names) are staged too, under their own names and keyed on the bare name before the first dot: the four `multiscale_oniom*_read_charge_*` cases rely on that for their `{title}.charge`, the adaptive cases for their `egs_ethylene_100.json` and `<title>.init.xyz`, and a `modeldir:` value found in the example text is copied whole as a directory because a pretrained ensemble is not a file. The rule that picks only what the example can resolve by name, rather than the whole shared directory, is `QUALIFICATION_DEVELOPMENT.md` section 6.6; the copy a submitted child still needs after that is `workflow/adaptive.py::_stage_runtime_fixtures`.

For the complete qualification plan structure and execution details, see [QUALIFICATION_DEVELOPMENT.md](QUALIFICATION_DEVELOPMENT.md) Section 6.4–6.7.

---

## 9. Long-term Expansion

The machine-learning jobtypes shipped their examples on 2026-10-09 (the
`nn_train` / `nn_predict` chain and `nn_grid_search`, section 7.5), and the
scan became two qualified arms in #319 item 十: `nn_grid_search.yaml`
submits its points to Slurm, `nn_grid_search_local.yaml` runs the same scan
on a local worker pool.
Future examples will gradually include:

- `frequency.yaml`
- `soc.yaml`
- `nac.yaml`
- `adaptive.yaml`
- `qm_mm.yaml`
- `periodic.yaml`
- ...

Every newly introduced workflow should provide one official YAML under `pyrai3md/examples/`.

---

## 10. Design Principles

The `pyrai3md/examples/` directory is:

- user documentation;
- runtime reference;
- qualification input;
- regression baseline.

It should remain the unique official repository of workflow YAMLs throughout the lifetime of PyRAI3MD.
