# Molcas Skill Development Guide

Last updated: 2026-11-12 19:20 (1.4 records the requested-block raise for energy/gradient. Earlier: 2026-11-11 17:25 (section 6.4: an optimization stopped at
its step budget may finish without `{project}.Opt.xyz`; it now reports
`opt_converge=False` and no `fin_coord` instead of raising. The rest was
not re-audited.)

Before it, 2026-11-06 17:40 (the pair-addressing measurement no longer
cites a retired validation-log path; the measured statement - five CASSCF NAC
logs, one table each - is unchanged.)

Before it, 2026-11-04 11:08 (#451: section 5.4 gained the multi-center QMMM charge round - an energy-only jobtype under `embedding: 'field'` keeps its `&ESPF` block and drops `&ALASKA`, verified on the two-center run's own decks (charge round 1 ESPF / 0 ALASKA, property round 1 and 1). The rest of the file was not re-audited this round. Earlier: 2026-11-02 20:20 (#449: section 5.4 was updated from the deck the code now builds - grouping keyed on the declared ladder, one `&ESPF` per embedding pass with the ESP files cleared between groups, and `rasscf_pass_count`/`deck_loops` as the single source both the builder and the reader call. The numbers quoted are local dry runs on the two-pass ESPF+NAC shape (`&RASSCF 2/&ESPF 1/RM 1` for one manifold, `4/2/2` for two); round 17's mixed QMMM single point answered the engine question - two `espf` blocks in one input, each with its own `rasscf`, `Happy landing` - and that reading is written into 5.4. Before it, 2026-11-02 16:10 (#448 cont.: section 5.4's mixed-spin paragraph was re-read against `service/builder/input_builder.py`, `service/workflow/compiler.py`, `schema/normalize.py` and `schema/method.py`, and priced by local dry runs of the four per-step asks on `state_spin [0, 0, 2, 2]`; section 5.5 was rewritten from the same runs plus the design as the maintainer stated it, and it now documents a deck defect that the runs exposed and this round fixed - a staged `StrOrb` reached none of the two passes of a dual deck. `warn_missing_orbital_guess` replaced the legacy stdout notice, fires on both builder paths, and names the wrong-active-space hazard. Two of the four things section 5.4 used to list as owed are already built and were here being described as missing: the spin-grouped pass builder `_pass_groups`/`_build_generic_pass` emits one `&RASSCF` per manifold with group-local `ROOT=` and `NAC=` lines, and `singlet_ci_roots`/`triplet_ci_roots` are derived from `state_spin` unconditionally in `schema/normalize.py`. What the paragraph now states instead is where the reach breaks: `resolve_profile` maps the framework's per-step asks (energy gradient, energy gradient nac) onto the *named* profiles, those compile with `is_generic` False, and the legacy branch they take never calls the grouping - one `Spin=1` block then gets asked for `ROOT=1..4` and for a pair whose second state lives in the manifold the deck never requested. The two genuinely open items are per-block `&ESPF` ordering (`input_builder.py:524` emits the block outside the group loop) and the orbital-continuation ruling (which manifold owns `RasOrb` between steps). The rest was not re-audited. Before it, 2026-11-02 10:40 (#448: section 6.2's NAC paragraph was rewritten for #448: the deck is written per spin group by `nac_deck_order`, the reader inverts that order, and the guard is the pair set rather than a block count; the skill's own `sorted()` of `target_nac_pair` is gone. What a real multi-pair log prints stays a test reading and is labelled as such. The rest was not re-audited. Before it, 2026-10-31 20:40 (section 5.4's ESPF 0-order charge readback was rewritten from what rounds `tmp_20261031_183037` and `tmp_20261031_195117` measured: `ESPF.DATA` is opened with `Molcas_Open` into the per-job run directory that the generated script deletes, so no calcdir ever held one; the log's `++ ESPF analysis` blocks are the live source; a run prints one per root inside `&RASSCF` plus more from `&CASPT2` - twelve on a two-root XMS job, one on a single-root job - so the read is bound to the state through the module banner and the root order rather than to a position in the file; and the fitted set leaves a 1.2e-3 charge residual that a 63-fold denser grid does not remove, which the framework averages out. The "still unconfirmed" note below it is superseded, and the silent-zero hazard it warned of is now a loud refusal. Before it, 2026-10-31 03:20 (section 5.4 and section 10 were re-read against `service/builder/embedding.py` this round: `embedding` takes `none`/`charge`/`field` and **refuses** `potential` with a message naming the spelling to use, while a charge *file* still declares `potential au` and reads as a field through `_legacy_charge_mode` - the two were conflated here and in the section 5.1 block-assembly table. Section 5.4 also gains two limits that the test round `tmp_20261031_005246` measured rather than inferred: a crossing pair spanning two spin manifolds is not served under ESPF (one call yields one state per spin, and the parser refuses the short log instead of guessing which surface is missing), and the ESPF 0-order charge readback of CHANGELOG #438 is still unconfirmed because every shipped QMMM example pins `molecule.q_qm`, so a mis-parsed `ESPF.DATA` header would return a silent zero. The rest was not re-audited.) Before it, 2026-10-29 21:27 (section 5.4 documents the pinned `GRID` /
`PNT 4 1.889726124` step every `&ESPF` block now carries and why the
default step aborts `espf_analysis` (reported upstream 2026-10-29); the
engine patch is local, not vendored. The rest was not re-audited. Before
it, 2026-10-22 22:30 (the config section no longer claims a `create_model` field merge, `molcas_workdir` is dropped from the Environment list since no such field ships - the workdir is resolved per run by `service/runner/env_check.py::resolve_workdir` - and the test-config generator is now the qualification converter, not `dev_tool.py`. The rest was not re-audited. Before it, 2026-10-11 16:25))

This document covers **OpenMolcas-specific** implementation details for molcas_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. Molcas-Specific Design Considerations

### 1.1 Text Module Chain Architecture

OpenMolcas uses a **text-based module chain** input format, unlike JSON-based backends.

Each computation is described as a sequence of named modules in a `.input` file:

```
&GATEWAY
  coord = $MOLCAS_PROJECT.xyz
  basis = cc-pvdz
  group = c1

&SEWARD

&RASSCF
  nactel = 2 0 0
  ras2 = 2
  nrofroots = 2

&ALASKA

&RASSI
  nrofjobroots = 2 2
```

The skill's input builder assembles these module blocks programmatically based on the
workflow graph, rather than filling a single JSON structure.

### 1.2 Gateway Input Convention

Gateway input files do not contain coordinate data directly. Instead, they reference
the `$MOLCAS_PROJECT.xyz` variable, which is resolved at runtime to the actual coordinate file path.

This decouples geometry updates from input file regeneration — only the `.xyz` file needs
to be rewritten between trajectory steps.

### 1.3 Directory Model: calcdir vs workdir

Molcas keeps two separate directories with distinct lifecycles:

- **calcdir** — the calculation file directory (input, xyz, output,
  orbital files). Built under the scratch path by
  `service/utils.py::build_calcdir()` (`{scratch_path}/tmp_MOLCAS`,
  or `{scratch_path}/{execution_id}` in framework state mode).
  Whether it survives after the calculation is controlled by
  `keep_tmp` (0 = removed, 1 = kept).
- **workdir** — the temporary file directory exported as
  `MOLCAS_WORKDIR`. It is **always removed** at the end of the run
  script (`rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT`), regardless of
  `keep_tmp`. The reason is that with `scratch_path: AUTO` the
  workdir lands on `/tmp` or `/srv/tmp`, which must be cleaned up
  promptly; calcdir, which holds the input/output files, follows the
  user's `keep_tmp` decision instead.

Workdir resolution (`service/runner/env_check.py::resolve_workdir()`):
- `AUTO`: prefer `/srv/tmp/$USER/$PROJECT/<hash>`, fallback to `/tmp/$USER/$PROJECT/<hash>`, where `<hash>` is ten hex digits of the SHA-1 of the calculation directory's absolute path. The term (calcdir basename) repeats across examples and slurm jobs, so without the hash two concurrent jobs on one node shared and deleted the same runfile and the gateway aborted with an internal inconsistency (round `tmp_20261111_132628`, 63 rows); the hash is stable for one calculation directory, so a restart keeps its scratch.
- Empty: prefer the environment `scratch_path`, then the AUTO fallback
- Otherwise: use the specified path

Whichever root is resolved, the backend appends one more directory
holding the calculation directory's name (`term=` argument, #301). The
run script both creates and `rm -r`s `$MOLCAS_WORKDIR/$MOLCAS_PROJECT`,
and `$MOLCAS_PROJECT` is only the program label, so two terms of this
skill under one root would otherwise delete each other's scratch.

### 1.4 Failure Propagation

A failed Molcas run (e.g. a module crashing with a non-zero return
code) must never be reported as success. Three layers enforce this:

1. **Run script exit code**: the generated local/slurm scripts
   capture the pymolcas return code (`rc=$?`) before cleaning the
   workdir and end with `exit $rc`, so the workdir cleanup never
   masks a failure.
2. **Backend return-code check**: `backend.calculate()` raises
   `RuntimeError` when the runner returns a non-zero code, except the
   unconverged-optimization stop: `_run_not_converged()` detects the
   `Stop Module ... /rc=_RC_NOT_CONVERGED_` marker (pymolcas exit
   code 96) in the log and falls through to parsing instead.
3. **Log completion check**: `log_parser.parse_log` raises when the
   log is missing, contains a failure marker (`Non-zero return
   code`, or a `Stop Module ... /rc=N` line with N != 0), or lacks
   the `Happy landing` completion banner. This catches incomplete
   runs whose exit code alone is unreliable (e.g. slurm wait mode).
   `_check_log_completion()` returns `'not_converged'` when the only
   anomaly is a SLAPAF `_RC_NOT_CONVERGED_` stop; a genuine module
   crash always wins over the tolerance.
4. **Requested result blocks**: `parse_log` raises when the workflow asked
   for an energy or gradient but the completed log carries no such block. The
   zero fill remains only for a property the run did not request.

Unconverged geometry optimizations are a result status, not a
failure: SLAPAF still writes the final structure, so the skill
returns `opt_converge=False` with the last parsed values and the
framework settles the run as not converged (single workflow) without
aborting. Zero-filled results are never substituted for missing
output; every other failure path aborts with a non-zero exit so
qualification reports mark the task as FAIL.

---

## 2. Molcas Calculation Profiles

### 2.1 Complete Profile List (15 profiles)

| Profile | Capability | Molcas Module Chain |
|---|---|---|
| `hf_energy` | energy | GATEWAY > SEWARD > SCF |
| `casscf_energy` | energy | GATEWAY > SEWARD > RASSCF |
| `casscf_gradient` | gradient | GATEWAY > SEWARD > RASSCF > ALASKA |
| `casscf_nac` | nonadiabatic_coupling | GATEWAY > SEWARD > RASSCF > ALASKA |
| `casscf_properties` | energy/gradient/NAC/SOC | GATEWAY > SEWARD > RASSCF > ALASKA > RASSI |
| `casscf_geometry_optimization` | geometry_optimization | GATEWAY > SEWARD > RASSCF > SLAPAF |
| `casscf_abinitio_dynamics` | abinitio_dynamics | GATEWAY > SEWARD > RASSCF > ALASKA > DYNAMIX |
| `casscf_surface_hopping` | surface_hopping | GATEWAY > SEWARD > RASSCF > SURFACEHOP > ALASKA > DYNAMIX |
| `casscf_soc` | soc_coupling | GATEWAY > SEWARD > RASSCF > ALASKA > RASSI (dual RASSCF) |
| `casscf_absorption` | absorption_spectrum | GATEWAY > SEWARD > RASSCF > RASSI |
| `casscf_frequency` | vibrational_frequency | GATEWAY > SEWARD > RASSCF > MCKINLEY |
| `caspt2_energy` | energy | GATEWAY > SEWARD > RASSCF > CASPT2 |
| `caspt2_properties` | energy/gradient/NAC/SOC | GATEWAY > SEWARD > RASSCF > CASPT2 > ALASKA > RASSI |
| `caspt2_absorption` | absorption_spectrum | GATEWAY > SEWARD > RASSCF > CASPT2 > RASSI |
| `caspt2_frequency` | vibrational_frequency | GATEWAY > SEWARD > RASSCF > CASPT2 > MCKINLEY |

### 2.2 Module Dependency Notes

- **GATEWAY**: Defines geometry (via `$MOLCAS_PROJECT.xyz`), basis set, symmetry group (c1), RICD approximation
- **SEWARD**: Computes one- and two-electron integrals (required before any quantum chemistry module)
- **SCF**: Hartree-Fock calculation (for `hf_energy` profile only)
- **RASSCF**: Multiconfigurational SCF (CASSCF/RASSCF)
- **ALASKA**: Analytical gradients and non-adiabatic couplings
- **RASSI**: State interaction for SOC, NAC, and transition properties
- **CASPT2**: Second-order perturbation theory (requires prior RASSCF convergence)
- **SLAPAF**: Geometry optimization
- **DYNAMIX**: Molecular dynamics propagation
- **SURFACEHOP**: Surface hopping dynamics (Tully's method)
- **MCKINLEY**: Vibrational frequency analysis

### 2.3 SOC Dual RASSCF Workflow

The `casscf_soc` profile requires **dual RASSCF** calculations (singlet + triplet) before RASSI:

```
GATEWAY (with AMFI, Angmom)
SEWARD (with rela, R02O, relint)
RASSCF (singlet: Spin=1, CIRoot from singlet_ci_roots)
>>COPY JobMix JOB001
RASSCF (triplet: Spin=3, CIRoot from triplet_ci_roots)
>>COPY JobMix JOB002
RASSI (Nrof JobIph=2, SpinOrbit, SOCOupling=0)
```

The `requires_dual_rasscf: true` flag in `manifest.yaml` triggers this workflow.
The `DUAL_RASSCF_PROFILES` list in `method.py` defines which profiles need this.

### 2.4 Transition / Absorption Requests

A jobtype string may carry a transition-dipole request on top of the
regular tasks (classic input flow):

- `has_transition_request(jobtype)` (`schema/method.py`) detects the
  `transition` token in the jobtype string; the token is stripped
  before the profile suffix lookup and recorded as a flag.
- The workflow compiler (`service/workflow/compiler.py::compile_workflow`)
  consumes that flag: when the profile's module chain does not already
  end in RASSI, a RASSI absorption step is **appended to the end of
  the chain** and the graph flag `requires_rassi_absorption` is set so
  the RASSI builder emits the absorption mode (`DIPRint=0`).
- SOC/absorption profiles already end in RASSI; their own RASSI mode
  covers the request (`requires_rassi_absorption` is only raised for
  the absorption profiles themselves).

---

## 3. Molcas Schema Details

### 3.1 Configuration Models

The config is composed of 5 Pydantic sub-models nested in one `SkillConfig` (`schema/config.py:353-364`), with flat keyword access routed to the sub-model that owns each keyword - the former `create_model` field merge is gone:

- **System**: atoms, coordinates, velocities, charges, multiplicity, nstate, current_state, state_spin, nac_pair_map, soc_pair_map
- **Properties**: energy, gradient, nac, soc, dipole, transition_dipole, hessian, frequencies, fin_velocity, computed_charges
- **Dynamics**: state_population, temperature, size, deco, thermostat
- **Calculation**: profile, method, basis, active space (nact_elec, nact_orb, ras1_orb, ras2_orb, ras3_orb, inactive_orb, active),
  CI parameters (ci_roots, ci_spin, singlet_ci_roots, singlet_ci_spin, triplet_ci_roots, triplet_ci_spin),
  target_states, target_nac_pair, target_soc_pair, guess_orbital, maxit_md
- **Environment**: molcas_project, molcas_mem, molcas_nproc, molcas_print, molcas_path,
  keep_tmp, nproc, scratch_path, slurm_*, thread

### 3.2 Active Space Parameters

Molcas uses RAS (Restricted Active Space) partitioning:

- **nact_elec**: Number of active electrons
- **nact_orb**: Total number of active orbitals
- **ras1_orb**: Orbitals in RAS1 (hole orbitals)
- **ras2_orb**: Orbitals in RAS2 (doubly active, defaults to nact_orb if not set)
- **ras3_orb**: Orbitals in RAS3 (particle orbitals)
- **inactive_orb**: Frozen inactive orbitals (computed by normalize: `(total_electrons - nact_elec) / 2`)

#### Active Orbital Selection (ALTEr)

- **active**: User-selected active orbital indices. Empty by default,
  which lets RASSCF classify orbitals by count in the orbital-file
  order. When non-empty, the RASSCF builder emits an `ALTEr` block
  that interchanges MOs in INPORB/JOBOLD so the selected orbitals
  land in the Ras2 window.
- Index convention: **1-based at every boundary** (user yaml,
  classic input, MOLCAS input), stored **0-based internally**.
  `build_config` converts to 0-based on ingest; the derivation
  converts back to 1-based when writing ALTEr.
- Derivation (`normalize.py::derive_alter_pairs`, all 0-based):
  - Ras2 window = `inactive_orb + ras1_orb ... inactive_orb + ras1_orb + ras2 - 1`;
  - orbitals shared between `active` and the window are **locked** (no swap);
  - each selected orbital outside the window is paired with a window
    position not covered by the selection (disjoint transpositions).
- The molcas skill does not use symmetry: the symmetry species of
  every ALTEr pair is fixed to **1**.
- Example (ethylene CASSCF(2,2), `inactive_orb=7`, `ras1_orb=0`,
  `ras2=2`, window = 1-based `[8, 9]`): `active: [6, 9]` locks
  orbital 9 and swaps orbital 6 into position 8:

```text
&RASSCF
 ...
 ALTEr
 1
 1 6 8
```

### 3.3 SOC-Specific Parameters

For SOC calculations, separate CI parameters are provided:

- **singlet_ci_roots**: Number of singlet states (e.g., `[3]` for 3 singlets)
- **singlet_ci_spin**: Spin for singlet states (`[0]` for 2S=0)
- **triplet_ci_roots**: Number of triplet states (e.g., `[2]` for 2 triplets)
- **triplet_ci_spin**: Spin for triplet states (`[2]` for 2S=2)

These are automatically populated by `normalize.py::verify_state_spin()` based on the `state_spin` list.

### 3.4 Supported Methods

**hf** (Hartree-Fock):
- Capabilities: energy
- Modules: GATEWAY, SEWARD, SCF

**casscf** (Complete Active Space SCF):
- Capabilities: energy, gradient, nac, geometry_optimization, abinitio_dynamics,
  surface_hopping, absorption_spectrum, soc_coupling, frequency
- Modules: GATEWAY, SEWARD, RASSCF, ALASKA, RASSI, SLAPAF, DYNAMIX, SURFACEHOP, MCKINLEY

**caspt2** (CAS Second-Order Perturbation Theory):
- Capabilities: energy, absorption_spectrum, frequency
- Modules: GATEWAY, SEWARD, RASSCF, CASPT2, RASSI, MCKINLEY

---

## 4. Molcas Normalize Workflow

The `normalize.py` module performs three key steps:

### 4.1 verify_electronic_structure

- Validates atom/coordinate count match
- Computes total electrons from atomic numbers and charge
- Computes alpha/beta electron counts from multiplicity
- Computes `inactive_orb = (total_electrons - nact_elec) / 2`
- Sets `ras2_orb = nact_orb` if not explicitly provided

### 4.2 verify_state_num

- Validates nstate >= 1
- Validates target_states are within [0, nstate)
- Validates target_nac_pair and target_soc_pair indices
- Initializes energy, dipole, gradient arrays
- Builds transition_pair list

### 4.3 verify_state_spin

- Validates state_spin consistency (singlet system: 0 or 2; non-singlet: all identical)
- Sorts state_spin and converts to state_label (s, t, q, etc.)
- Builds **nac_pair_map** (same-spin pairs) and **soc_pair_map** (different-spin pairs)
- Auto-distributes singlet/triplet CI parameters for SOC dual RASSCF

---

## 5. Molcas Input Builder Details

### 5.1 Module Block Assembly

Each Molcas module block is generated by a dedicated builder function in `module_builders.py`:

| Module | Builder Function | Key Parameters |
|---|---|---|
| GATEWAY | `build_gateway()` | coord reference, basis set, symmetry group, SOC (AMFI/Angmom), Ediff constraint (MECI/MECP); no XField |
| SEWARD | `build_seward()` | SOC (rela, R02O, relint) |
| ESPF | `build_esfp_block()` | charge (`External = n Angstrom`) or field (`External = 0`, G=0) |
| RASSCF | `build_rasscf()` | Spin, NactEl, Inactive, Ras2, ITERATIONS (maxit_scf,100), CIRoot, fileorb (guess / optimization loop), Rlxroot, MDRLXR + RasOrb (dynamics), spin='singlet'/'triplet' for SOC/MECP |
| SCF | `build_scf()` | charge, spin (for HF) |
| ALASKA | `build_alaska()` | ground state, ROOT= (1-based), NAC=i j (1-based), NOCSF by default, driven by target_states/target_nac_pair |
| RASSI | `build_rassi()` | CASSCF SOC (Nrof JobIph, SpinOrbit, EJob), CASPT2 SOC (IphNames, HEff or EJob), absorption (DIPRint=0) |
| CASPT2 | `build_caspt2()` | ss/ms/xms/rms/xdw, GRDT, DORT, IPEA/shift; state selection is in ALASKA |
| SLAPAF | `build_slapaf()` | Iterations, thrshld |
| DYNAMIX | `build_dynamix()` | velver, dt (dynamics.size, au), velo=1, ther (thermostat: off/nve=0, nose/nose-hoover/nvt=2), temp |
| SURFACEHOP | `build_surfacehop()` | tully, decoherence (dynamics.deco), psub |
| MCKINLEY | `build_mckinley()` | (no parameters) |

### 5.2 Geometry Optimization Loop (single-state / MECI / MECP)

For `casscf_geometry_optimization`, `input_builder.py` derives the mode
from `opt_state_i`/`opt_state_j` (0-based) and `state_spin`
(`_resolve_optimize_mode`):

- **single** (`i == j`): plain relaxation; RASSCF gets `Rlxroot=i+1`
- **meci** (same-spin pair): GATEWAY `Constraints` block with
  `a = Ediff n m` (1-based roots); Rlxroot relaxes state i
- **mecp** (cross-spin pair): bare `a = Ediff` constraint; dual
  singlet/triplet RASSCF jobs, each with its own Rlxroot (i must be
  the singlet; the triplet root is renumbered by the singlet count).
  Each spin job is followed by a bare `&ALASKA` so the constrained
  gradient of its crossing state reaches SLAPAF, and the singlet
  RunFile is locked as `$WorkDir/RUNFILE2` before the triplet job
  overwrites it (`>> COPY $WorkDir/$Project.RunFile $WorkDir/RUNFILE2`)

The assembled input starts with `>> EXPORT MOLCAS_MAXITER=opt_maxit`
and wraps SEWARD/RASSCF/SLAPAF in a `>> Do while` ... `>> End Do`
loop (GATEWAY stays outside so SLAPAF-updated coordinates are re-read).
Single-state and MECI optimizations emit no `&ALASKA` before SLAPAF
(SLAPAF drives the gradients itself); only MECP inserts the ALASKA
blocks. RASSCF reads `fileorb=$MOLCAS_PROJECT.RasOrb` for orbital continuation
between iterations only when a user guess has been staged (see 5.5);
without a guess the first iteration uses default orbitals and later
iterations continue from the RasOrb written by the previous step.
Invalid requests (out-of-range states, inverted MECP spin
assignment, CI roots not covering the relaxed root) raise ValueError.

### 5.3 Dynamics Loop (ab initio MD / surface hopping)

For `casscf_abinitio_dynamics` and `casscf_surface_hopping`, Molcas
propagates the nuclei internally in a single run. The assembled input
starts with the header block:

```
>> COPY $MOLCAS_SUBMIT_DIR/$Project.xyz $WorkDir/$Project.xyz
>> COPY $MOLCAS_SUBMIT_DIR/$Project.velocity.xyz $WorkDir/$Project.velocity.xyz
>> COPY $MOLCAS_SUBMIT_DIR/$Project.StrOrb $WorkDir/$Project.RasOrb   # only when a user guess is staged
>> EXPORT MOLCAS_MAXITER={md_maxiter}
```

The optional StrOrb COPY line renames the locked user guess to RasOrb
on the way into the scratch directory (naming convention: every
orbital file referenced inside the input uses the RasOrb suffix;
StrOrb only appears as the user-supplied guess filename).

The velocity file is written by `write_velocity_file` from
`System.velocities` (Bohr/au, framework `title.velo`) and is
mandatory: a missing velocities field raises ValueError. Despite the
`.velocity.xyz` filename, the content is a plain N x 3 numeric block
(no atom count line, no comment line, no element-symbol column) --
the format Molcas DYNAMIX expects. `md_maxiter`
defaults to `maxit_md x 10` and must exceed the MD step count.

GATEWAY stays outside the `>> FOREACH ITER in (1 .. maxit_md)` loop
that wraps SEWARD..DYNAMIX (closed by `>> End do` after DYNAMIX).
Inside the loop:

- RASSCF carries `MDRLXR=current_state+1` (the initial MD state from
  `system.current_state`, 0-based) and reads
  `fileorb=$MOLCAS_PROJECT.RasOrb` for orbital continuation between
  iterations only when a user guess has been staged (see 5.5)
- ALASKA stays bare (DYNAMIX selects the gradient state via MDRLXR)
- SURFACEHOP (surface hopping only) uses `decoherence = dynamics.deco`
- DYNAMIX uses `velver`, `dt = dynamics.size`, `velo = 1` (reads the
  velocity file), `ther` mapped from `dynamics.thermostat` (off/nve
  -> 0 NVE, nose/nose-hoover/nvt -> 2 Nose-Hoover NVT; any other
  thermostat raises) and `temp`

### 5.4 ESPF Embedding

`embedding` selects the external field; XField is no longer emitted.

- `none`: no ESPF and no external charges.
- `charge`: `&ESPF` with `MultipoleOrder = 0`, `External = n Angstrom`,
  then one line per charge: `x y z q 0 0 0`.
- `field`: `&ESPF` with `MultipoleOrder = 0`, `External = 0`,
  then one line per QM atom: `atom V Fx Fy Fz 0 0 0 0 0 0`, read from
  `system.external_potential`, which must carry exactly `natom` rows of
  `[V, Fx, Fy, Fz]`. The six G components are zero.
- `potential` is **retired**: `resolve_embedding_field` raises and names
  `field` as the spelling to use. A charge file is not affected, because
  the file declares its own representation on its first line
  (`point` or `field`) and `embedding` is not consulted at all on that
  path.

Both forms end with the pinned PNT grid step:

```
GRID
PNT 4 1.889726124
```

OpenMolcas writes `DELTAR` into `ESPF.DATA` with a 9-decimal format and a
later module (`espf_analysis`, reached from SCF, RASSCF and CASPT2
properties) rebuilds the grid from it with `int(range/DeltaR)`. With the
default `1/Å = 1.8897261259…` the stored value is `1.889726126`, one
bucket boundary moves, and `MkGrid` aborts with
`inconsistency in nGrdPt: 904 883` whenever a QM direction's span is an
integer number of Å (including 0). A 9-decimal step round-trips through
the file exactly, so both calls divide by the same double and agree for
every geometry. `1.889726124` is the 9-decimal floor of `1/Å`; any
9-decimal value would do. The failure was reported upstream on 2026-10-29
(both OpenMolcas v26.10-15-gf6fa8fa1c and a site build of Molcas 26
reproduce it with the same 904/883 pair); the engine side is fixed locally
by writing `DELTAR` with 17 significant digits (`espf_write`, not
vendored here), which makes the pin unnecessary once the installed Molcas
carries it.

`read_charge='none'` consumes the per-step `point_charges` /
`external_potential` arrays. A filename is read from the calculation
directory or the launch directory. Its first line declares what it holds, so
the representation is a property of the data and not of the reader:

```
point angstrom external   # rows: q x y z, arbitrary external points
point angstrom atom       # rows: q x y z, one per mid-shell atom
field au                  # rows: V Fx Fy Fz, one per QM atom
```

`read_charge: true` expands to `<project>.charge`, and the same document is
what the framework reads (section 9.5), so a skill run and the framework legs
open one file by one name. Three older spellings are still read: `charge
angstrom|bohr` followed by `x y z q` rows, `potential au` followed by `V Fx Fy
Fz` rows, and a bare count line followed by `q x y z` rows - the last one
declares no role, which is the only reading it ever had (the ONIOM mid shell
or a standalone background, decided by the regime). A count that does not
match its rows, a row of the wrong width and an undeclarable first line are
refused rather than guessed.

A skill run refuses an `atom`-role document: only the framework's charge store
consumes atom-indexed rows, and it takes the q column while the positions come
from the live geometry, so a frozen copy of the coordinates in the file is
never the one used.

For ESPF + NAC the builder emits two passes: ESPF energy/gradient/SOC,
then a no-ESPF NAC pass. The parsed NAC carries
`nac_approximation='two_pass_no_esfp'`.

When the run is embedded (`charge` or `field`), the backend also reads the
ESPF 0-order charges into `computed_charges` (shape `(1, natom)`, QM atom
order) through `service/parser/espf_charges.py`. Both of its sources had to
be measured on a real run before either could be trusted, and one of them
turned out not to exist:

- **`ESPF.DATA` is written where the framework cannot see it.**
  `espf_write.F90` opens it with `Molcas_Open`, which places it in the
  per-job run directory under `$MOLCAS_WORKDIR/$MOLCAS_PROJECT`, and the
  generated run script removes that directory on exit. No deployed calcdir
  ever held one (checked on `tmp_20261031_183037`), which is why the reader
  returned None for every QMMM run and the framework ran on the
  hand-written `molecule.q_qm` instead. The sidecar branch is kept for runs
  that do produce it, and it is preferred when present because the log
  carries only four decimals.
- **The log is the live source, and it prints several blocks.** A run writes
  one `++ ESPF analysis` section per state inside `&RASSCF` (root order) and
  more from `&CASPT2`; counted on round `tmp_20261031_195117`, a two-root
  XMS job printed twelve blocks, a single-root job one, a three-pass RASSCF
  three. Nothing inside a block names its state, so **which block is read is
  a state question, not a position in the file**: the reader tracks the
  `--- Start Module: <name> ---` banner and reads every block inside a
  `rasscf` section, in file order: the k-th such block is the k-th root of
  that section, and sections appear in the order the input declares its spin
  manifolds, so the row sequence is the global 0-based state numbering the
  framework already uses. Reading the
  last block instead hands the classical stage a response/multi-state vector
  for another root - measured as charges summing to 0.0276 where the active
  state gives 0.0012, which is what made `qmmm_energy` refuse the run. The
  rule was then checked forward on round `tmp_20261031_220625`: that run's
  `MOLCAS.log` holds `Charge on O1 = -0.5990` / `Total ESPF charge = 0.0012`
  as the first `rasscf` block, `+0.1215` / `0.0255` as the second root, and
  `-0.5730` again inside `caspt2`, and the composite delivered the first one
  with `-0.0004` taken off each atom.
- The reader returns `(nstate_read, natom)` and chooses nothing. Which state
  embeds is the framework's question, answered in
  `workflow/calculator.py::_qmmm_embedding_state` from the step's
  `target_states` (or the system's current state), and a state index beyond
  the rows that came back is refused by name in the log - not answered with
  row 0, which is exactly how a two-state run used to embed both surfaces
  with the ground state's charges. The row count is not capped by the
  declared `nstate` either: the run's roots come from the input template this
  skill was handed, which can name more of them than that field says, and a
  discarded row is a state nobody can ask for. A pair spanning two spin
  manifolds stays unserved for the reason in the next paragraph.

The fitted charges are not charge-conserving: the projection onto the shell
grid leaves a residual (1.2e-3 for a neutral water at the default 4 shells /
1 Å) which denser sampling does not remove - measured 1.2e-3 at 898 points,
1.5e-3 at 56,541 points, 5e-4 at 8 shells. The framework therefore averages
that residual over the region against the declared `molecule.charges` before
handing the values to the classical stage; see `dev/methods/QMMM_DEVELOPMENT.md`
and CHANGELOG #445. A `molecule.q_qm` the input spells out is never rescaled,
and the reader returning nothing is a silent zero, not an error.

A crossing pair that spans two spin manifolds is not served. Under ESPF one
call yields one state per spin, so a `ci_roots [1, 1]` / `ci_spin [0, 2]`
request comes back with a single block and the parser refuses it rather than
guessing which surface is missing (`the log carries 1 state energy(ies) while
the run declares 2 states`; remote `tmp_20261031_005246`).

What #448's dry runs price is that the grouping itself is built - it is just not
reachable from the per-step ask. Measured on one 6-atom molecule with
`state_spin [0, 0, 2, 2]`, `target_states [0, 1, 2, 3]`, CAS(2,3):

- the generic profile serves two manifolds: `energy gradient nac` under
  `casscf_properties` emits `&RASSCF Spin=1` then `&RASSCF Spin=3`, each with
  its own `NAC=1 2` and its own `ROOT=1, ROOT=2` (the root offsets are already
  group-local), and `casscf_soc` emits the same two blocks plus RASSI;
- the named profiles the main driver resolves to (`casscf_gradient` for
  `energy gradient`, `casscf_nac` for `energy gradient nac`) used to take the
  legacy branch, where nothing grouped - one `Spin=1` block asked for
  `ROOT=1..4` and for a pair whose second state lived in the manifold the deck
  never requested. #449 closed that: `groups_by_manifold` sends a mixed ladder
  through the grouped builder whatever the profile is called, so the same ask
  now emits `Spin=1` (roots 1, 2) and `Spin=3` (root 1) in two blocks, and the
  NAC variant carries `NAC=1 2` only in the manifold that declares the pair.
  The routing and the pass count are one predicate, which is why the reader
  cannot disagree with the writer about the deck.

Two of the four items this paragraph used to list are met elsewhere in the same
file: the pass-group function exists, and `singlet_ci_roots` / `triplet_ci_roots`
are derived from `state_spin` in `schema/normalize.py` unconditionally (it is
`build_rasscf(spin=None)` that ignores them and reads `multiplicity` /
`ci_roots[0]` instead). The other two were fixed in #449, and the grouping rule
itself moved in the same change:

- **Grouping follows the declaration, not the tokens.** `_pass_groups` now asks
  `deck_is_dual(config)` - does the declared ladder span more than one manifold
  - instead of looking for an `nac` token and a pair that happens to cross. A
  mixed ladder whose couplings sit inside one manifold used to lose the other
  manifold's pass entirely; it now gets both, which is what the input asked for.
- **`&ESPF` is written per pass.** The field is still resolved once (every pass
  embeds the same classical charges), but the block moved inside the group loop,
  with `RM -FORCE ESPF.DATA/ESPF.EXTPOT` between groups: the ESP integrals live
  in a single pair of files, so a second manifold without its own `&ESPF` would
  read the first manifold's density. Measured on the two-pass ESPF+NAC shape:
  one manifold gives `&RASSCF 2 / &ESPF 1 / RM 1` (unchanged, byte-for-byte),
  two manifolds give `&RASSCF 4 / &ESPF 2 / RM 2`, and every `&RASSCF` of the
  embedding pass is preceded by its own block.
- **One source for the pass count.** `rasscf_pass_count(config, workflow)` is
  now what the grouped builder and `log_parser` both call; the reader no longer
  compares `profile == 'casscf_geometry_optimization'`, and the
  "keep the last round" reset is gated on `deck_loops(workflow)` so a property
  pass keeps every block it printed. A mixed property deck's second block used
  to be labelled with the first pass's spin.

The engine question is answered. Round 17 (`tmp_20261102_195813`) ran the
mixed-ladder QMMM single point (`ci: "2 1"` / `spin: "0 1"`,
`multiscale_qmmm_embedding_sp`), and its Molcas log carries the sequence
`gateway -> seward -> espf -> rasscf -> caspt2 -> alaska -> mclr -> caspt2 ->
mclr -> alaska -> espf -> rasscf -> caspt2 -> rassi`: two `espf` blocks in one
input, each followed by its own `rasscf`, two energy tables, one RASSI, `Happy
landing` in 10.77 s wall - so the second block does rebuild its grid and the
`MkGrid` abort of #363 does not recur. The mixed **two-pass** deck (ESPF pass + NAC pass, four `rasscf` blocks) has its
own carrier: `examples/multiscale_qmmm_embedding_nac.yaml` (QMMM, `embedding:
true`, `jobtype: energy gradient nac`, the same mixed ladder). Round 21
(`tmp_20261102_213544`) read it on the engine: two gateways, `espf` + `rasscf`
per manifold in the embedding pass, then the field-free NAC pass, four energy
tables and `Happy landing`, with the yaml and classic-input carriers both PASS. The same
deck also ships as a skill case (`casscf_espf_nac_twospin`), which after being rewritten
in the skill-test flow's own dialect - inline `external_charges` -> `{project}.charge` +
`read_charge: true`, not the framework's whole-system `point_charges` - reads green on
all three carriers (round 22) with `2 &ESPF / 4 &RASSCF / 2 &GATEWAY` in the framework
carrier's deck.

Two questions this paragraph used to raise were mine, not the project's, and
both are retracted. The shared `{project}.RasOrb` is the design, documented in
section 5.5 - not an open decision. And serving a mixed ladder does not need
SOC: propagation on each manifold is enough, and hops within one manifold are
the same-spin cases already shipped. SOC is what an *intersystem* hop needs,
which is a separate switch (`md.dosoc`, `planning/resolver.py:305`), and that
channel is already run per step on another skill (`examples/md_openqp_fssh_soc.yaml`);
no molcas example turns it on, which is a coverage gap, not a blocker for #21.

What the same dry runs did turn up is a defect in the deck, and it is fixed
here: the `fileorb` line only reached an unlabelled `&RASSCF`, the Rlxroot line
or the MDRLXR line, so a dual deck - both the legacy `casscf_soc` branch and the
grouped generic pass - read **no** orbital guess even when the user staged one,
and ran both manifolds on Molcas' default orbitals in silence. Measured before
and after on `casscf_soc.yaml`: `&RASSCF` blocks 2, guess lines 0 before, 2
after, and still 0 when no guess exists. That is also why the missing-guess
notice became a `warnings.warn` naming the hazard (a default guess can converge
on an active space other than the one the input declares) and why it now fires
on the generic path too, which previously printed nothing.

Until then an embedded crossing names two states of one spin.

The multi-center QMMM charge round is this embedding solve with the gradient
token withheld. `resolve_profile` spells an energy-only jobtype from the task's
capabilities (`planning/task.py::_expand_qmmm` gives the `qm_charge` duty
`['energy']`), the compiler still takes the generic single-pass branch for
`casscf_energy` under `embedding: 'field'`, `&ESPF` is still inserted, and
`&ALASKA` is gated by its own token - so the round is cheaper by a module and
still answers with a bindable charge block (the row selection of #450 applies
unchanged). Read on the two-center case's own decks: the charge round's input
carries one `&ESPF` and no `&ALASKA`, the property round's carries one of each.

### 5.5 Orbital File Management

Molcas uses multiple orbital file types:

- **StrOrb**: Starting orbital file — only used as the locked
  user-supplied guess filename (`{project}.StrOrb`)
- **RasOrb**: RASSCF orbital file; every orbital reference inside a
  generated input uses this suffix (naming convention)
- **JobIph**: JobIph file for state-averaged calculations

The `manage_orbitals()` function in `input_builder.py` handles:
- Staging the user guess (an explicit `guess_orbital` or a
  `{project}.StrOrb` probed in the working directory) into the calcdir
  twice: as `{project}.StrOrb` (the locked guess, never consumed) and
  as `{project}.RasOrb` (what RASSCF reads). Neither file overwrites
  an existing one, so multi-step runs keep the orbitals converged by
  earlier steps.
- Copying `{project}.JobIph` or `{project}.JobIph.new` for state-averaged calculations

`build_input()` probes `{project}.StrOrb` in the calcdir
(`_has_strorb`) to decide the fileorb logic: when it is present the
input gets the StrOrb->RasOrb COPY line (dynamics) and every RASSCF
block carries `fileorb=$MOLCAS_PROJECT.RasOrb`; when it is absent no
fileorb line is emitted anywhere, `warn_missing_orbital_guess()` fires
on both builder paths, and loop continuation relies on the RasOrb each
RASSCF run writes.

Sharing one `{project}.RasOrb` between the passes of a dual deck is the
design, not an oversight. The deck runs the singlet group first and the
triplet group second, so the triplet pass starts from the orbitals the
singlet pass converged; `manage_orbital_files()` then saves that last
RasOrb back as the next call's StrOrb, which is what feeds the following
step's singlet pass. When no guess was ever provided the passes read
nothing (`fileorb` is not written) and each uses Molcas' own default
orbitals - deliberately, so the shipped cases stay runnable without an
orbital file per case. In a production run the guess is what makes the
result trustworthy, so the missing guess is now a warning that names the
hazard (a default guess can converge on an active space other than the
one the input declares) instead of a bare line on stdout.

Measured on `casscf_soc.yaml` (`state_spin [0, 0, 2]`, two `&RASSCF`
blocks) before and after #448: a staged guess reached **0 of the 2
passes** - the `fileorb` line was only ever written by the unlabelled
template, the `Rlxroot` line and the `MDRLXR` line, so both labelled
passes silently ran on defaults, in the legacy `casscf_soc` branch (the
guess was not even passed to `build_rasscf`) and in the grouped generic
pass (passed, then dropped for `spin is not None`). After the fix both
passes carry one line each, and a run with no guess still carries none;
`test_a_dual_deck_guess_reaches_every_spin_pass` pins both halves.

`manage_orbital_files()` in `orbital_parser.py` maintains inter-step
continuation in multi-step runs: it stages the user guess (or a
`{project}.StrOrb` probed in the working directory) as the locked
StrOrb, then refreshes that StrOrb from the RasOrb written by the
previous RASSCF step so the guess lineage follows the converged
orbitals. It also carries `{project}.JobIph`/`JobIph.new` forward.

When a `{title}.StrOrb` file exists next to the framework input file
(title = control.title, defaulting to the input basename), the
interface layer detects it and injects it into the skill
`guess_orbital` (fill-if-empty; an explicit value takes precedence),
so single-point, optimize and dynamics RASSCF blocks all start from
the same RasOrb staging.

This enables orbital continuity across trajectory steps.

### 5.6 ALASKA Gradient Control

ALASKA supports three modes:

1. **Ground state gradient** (default): `&ALASKA`
2. **Excited state gradient**: `&ALASKA` with `ROOT=i` (1-based state index)
3. **NAC computation**: `&ALASKA` with `NAC=i j` (1-based state indices)

For CASPT2 the root/pair is selected by the ALASKA block
(`ROOT=` / `NAC=`), not by CASPT2 keywords: the CASPT2 block carries
only `GRDT`. One `GRDT` can be followed by multiple `&ALASKA ROOT=`
and `&ALASKA NAC=` blocks, so multiple gradients/NACs are evaluated
in one pass. NAC defaults to `NOCSF`, i.e. both CASSCF and CASPT2 NAC
exclude the CSF contribution.

ROOT keywords are generated strictly from `target_states`; the workflow
selects per-step states through `calculate(target_states=...)`, which
overrides `config_dict['target_states']` before the input file is
built. The log parser maps the k-th parsed gradient onto the
`target_states[k]` row of the full `(nstate, natom, 3)` array
(untargeted states stay zero), matching the bagel/pyscf shape
contract.

### 5.7 RASSI SOC Computation

For SOC calculations, RASSI uses the `Nrof JobIph` format:

```
&RASSI
Nrof JobIph=2 3 2;1 2 3;1 2
SpinOrbit
EJob
SOCOupling=0
```

Format: `Nrof JobIph=njobs n_singlet n_triplet;singlet_states;triplet_states`

CASPT2 SOC uses a separate RASSI block: each spin group copies its
CASPT2 JobIph (`JOBIPH_S` / `JOBIPH_T`), then RASSI reads them through
`Nrof JobIphs` + `IphNames` + `Spin Orbit`. Multistate CASPT2 uses
`HEff`; single-state CASPT2 uses `EJob` as an approximation. A
single-root spin block has no CASPT2 effective Hamiltonian on its
JobIph, so RASSI keeps that block's CASSCF diagonal; the parser takes
the block's `FINAL CASPT2 RESULT` energy and warns when the two differ
(measured on `xmscaspt2_soc`: the triplet block was 1.52e-01 Ha below
the RASSI diagonal). The returned `energy` is the pre-SOC CASPT2
energy; `SO-RASSI State` eigenvalues are not used for `energy`.

The SOC matrix is extracted from the RASSI log output. Each row of
the SOC matrix block contains 9 columns:
`I1 S1 MS1 I2 S2 MS2 Real Imag Absolute`. The parser reads columns
`parts[0]` (I1) and `parts[3]` (I2) as state indices (#85 fix:
the former `parts[1]` incorrectly read S1 as I2). Lines with fewer
than 9 columns are skipped; numeric conversion failures are caught
and skipped; out-of-range indices are protected.

---

### 5.8 Input Format Rules Inherited From The Example-Input Note

The templates and builders in this skill were aligned, once, against a
hand-written note of example OpenMolcas inputs (an ethylene geometry plus one
block per calculation type, with the rules the engine actually needs). That
alignment round is CHANGELOG entry #7 of 2026-07-30 and it closed 25 differences
between the code and the format. The note itself used to live in the repository
as `dev/skills/MOLCAS_TEST.md`, and several code comments named it; the file was
removed on 2026-10-07 because it was neither a development guide nor a test, and
its rules are recorded here instead, with the code site that now owns each one.
Code comments that used to point at the note point at this section.

| Rule | Where it lives in the code |
| --- | --- |
| `coord` must be the first line after `&GATEWAY`, otherwise the engine cannot locate the coordinate file | `service/templates/molcas_templates.py:25-28` (comment), template at `:31-38` |
| `RICD` in `&GATEWAY` and `doanalytic` in `&SEWARD` on **every** run, for performance | `molcas_templates.py:37` and `:63`; the SOC variants keep both at `:47` and `:72` |
| `Group=c1` always (no symmetry is assumed) | `molcas_templates.py:36`, `:46` |
| Method vocabulary is HF / CASSCF / CASPT2; OpenMolcas DFT is not offered because it is too slow for per-step use | `schema/method.py:26`, `:38`, `:51` |
| Basis vocabulary, including the per-element `Element.basis` spelling | `schema/reference.py:59` (`AVAILABLE_BASIS`), enforced at `schema/validate.py:107-114` |
| RASSCF keyword order: `Spin`, `Nactel`, `Inactive`, `Ras2`, `ITERATIONS`, `CIRoot` | `service/builder/module_builders.py:155-157` |
| State averaging through `CIRoot=n n 1` (the third value fixed at 1: no symmetry) | `module_builders.py` RASSCF builder, profile-side roots in §3.2 |
| Root tracking through `Rlxroot=k` for SLAPAF and the native dynamics loop | `molcas_templates.py:114-116` (`RASSCF_RLXROOT_LINE`) |
| Test-case flexibility: `fileorb` is **omitted** for the generated cases so the engine uses its automatic guess | `service/builder/module_builders.py:176-186` (the `fileorb` parameter is off unless a guess is available), selection at `:215-220` |
| Normal use (`pyrai3md_agent` / `pyrai3md_skill` / `pyrai3md_input`): `fileorb=$MOLCAS_PROJECT.StrOrb`, its availability checked, and a missing file is answered by telling the user to run an HF energy first (that job needs no `StrOrb` and produces the `ScfOrb` to copy) | `molcas_templates.py:91-95`, `service/parser/orbital_parser.py:44-58` |
| Optimization and dynamics: `fileorb=$MOLCAS_PROJECT.RasOrb`, so the converged RASSCF orbitals continue; on the first step, with no `RasOrb` yet, `StrOrb` is copied into its place | `molcas_templates.py:103`, `orbital_parser.py:57-58`, `backend.py:262` |
| Engine-internal loops: `>> EXPORT MOLCAS_MAXITER=n` plus `&GATEWAY` / `>> Do while` … `>> End Do`; `n` is `opt_maxit` (0 = auto, `maxit_md x 10`) | `molcas_templates.py:208-212` (`EXPORT_MAXITER_TEMPLATE`), `schema/config.py:258` |
| Module chains: `GATEWAY -> DO WHILE -> SEWARD -> RASSCF -> SLAPAF -> END DO` (optimization, §5.2), `… -> ALASKA -> DYNAMIX -> END DO` (dynamics, §5.3), `… -> SURFACEHOP -> ALASKA -> DYNAMIX` (surface hopping) | §5.2, §5.3 |
| `&SLAPAF Iterations=n thrshld=0.0 5.0D-4` | `molcas_templates.py:203-205` |
| SOC needs the relativistic block: `AMFI` / `Angmom` / `0 0 0` in `&GATEWAY`, `rela` / `R02O` / `relint` / `expert` in `&SEWARD` | `molcas_templates.py:40-50`, `:66-73`, `module_builders.py:68` |
| SOC state ladder: `>>COPY $WorkDir/$Project.JobMix $WorkDir/JOB00n` between the singlet and triplet RASSCF runs, then `&RASSI … SpinOrbit EJob` | §2.3, §5.7 |

Two rules of the note are **not** implemented, and they are kept here rather than
silently dropped:

1. **Orbital-rotation input (`flip`).** The note specifies a block under
   `&RASSCF` -- the keyword `flip`, then the number of rotations, then one
   `1 <target> <original>` line each (initial index fixed at 1, no symmetry) --
   for rotating `StrOrb` when the user supplies the pairing. Nothing in this
   skill reads such a value (`grep` over `skills/molcas_skill/` finds no `flip`
   site), so an orbital swap today has to be done by hand on the orbital file.
   The note also warns that a high-level workflow (NAMD, geometry optimization)
   must **not** carry the rotation, to avoid repeated rotation across iterative
   single points; the absence of the feature is what keeps that warning
   satisfied today.
2. **The SOC basis warning.** The note asks for an ANO-rcc-type basis for SOC and
   a warning when another one is used. Only the vocabulary is enforced
   (`schema/validate.py:107-114`); a non-ANO-rcc basis on a `casscf_soc` profile
   reaches the engine with the relativistic block and no comment from us.

---

## 6. Molcas Result Parsing

### 6.1 Output Files

| File | Content |
|---|---|
| `{project}.log` | Main log with energies, gradients, NAC, SOC matrices |
| `{project}.xyz` | Coordinate file (Angstrom) |
| `{project}.Opt.xyz` | Relaxed structure from SLAPAF (geometry optimization; lowercase `opt.xyz` probed as fallback) |
| `{project}.md.xyz` | MD trajectory from DYNAMIX (dynamics; last frame = final structure) |
| `{project}.StrOrb` | Starting orbital file |
| `{project}.RasOrb` | RASSCF output orbital file |
| `{project}.JobIph` | JobIph file for state-averaged calculations |

### 6.2 Parsing Logic (log_parser.py)

The `parse_log()` function extracts:

- **Coordinates**: From "Cartesian coordinates in Angstrom" section
- **Energies**: From "Final state energy(ies)" section (RASSCF root energies)
- **Gradients**: From "Molecular gradients" section
- **NAC**: From "CI derivative coupling" section
- **SOC**: From "Spin-orbit section" section

Block counting, which is what a cross-spin MECP run turns on (#347): the dual
singlet/triplet job prints **two** energy blocks and two gradient blocks per
optimization iteration, and the two blocks do not necessarily hold the same
number of roots.

- Each block is read with **its own** spin group's root count (`ci_roots` is that
  per-group list, `[1, 1]` for the shipped MECP case, `[3, 1]` for three singlets
  and one triplet). Reading every block with the first group's count either stops
  short or runs into the next section's text, and the parse that did so swallowed
  the resulting error and returned a state table with a surface missing.
- Only the **final** iteration is kept, for energies *and* gradients. Before the
  gradients were reset with the energies, the blocks of every iteration
  accumulated and the packing took the first `len(target_states)` of them, so an
  optimization returned its **first** geometry's forces while reporting the last
  one's energy.
- A requested state with no gradient block, or fewer state energies than declared
  states, is refused by name. Both cases used to leave a zero row that satisfied
  the `(nstate, natom, 3)` shape and read as a flat surface to anything stepping
  on it. Extra blocks are reported rather than dropped in silence.
- No real Molcas dual-job log exists on this environment, so the criteria added with
  this rule (`tests/test_skill.py::TestDualSpinBlocks`) are written to the block
  shape the parser's own branch documents. Which branch a real dual-job log takes,
  and how many gradient blocks one SLAPAF iteration prints, are readings Q1-Q3 of
  the test round.

A coupling the log never printed is reported as **no rows**, not as a zero row
(#446): the reader used to answer an absent `Total/CI derivative coupling` table
with `np.zeros((1, natom, 3))`, and `meci` read that as a measured coupling of
zero - it dropped the direction below its floor (`optimizer/meci.py:49`) and
printed `Coupling Norm: 0.00000000` with `Branching Directions 1.00000000`,
while the engine had in fact been asked for nothing (the deployed input carried
two bare `&ALASKA` blocks and the log contains no `derivative coupling` line at
all). A search whose pair the run cannot address is refused by the schema
instead: `verify_state_num` rejects `target_nac_pair` indices the state table
does not cover, and the builder refuses a `nac` request on a one-state table.
The criterion is two-armed (`tests/test_caspt2_espf.py`): one state table is
refused by name, another prints `NAC=1 2`.

SOC parsing (#85):
1. Reads the SOC matrix block from the RASSI log; each data row has
   9 columns: `I1 S1 MS1 I2 S2 MS2 Real Imag Absolute`
2. State indices are read from `parts[0]` (I1) and `parts[3]` (I2);
   lines with fewer than 9 columns are skipped
3. Computes the dimension: `soc_dim = sin_state * mult[0] + tri_state * mult[-1]`
4. Reduces to configuration state basis: `soc_mtx[s1, p2] = sqrt(sum(soc_sfs[s1, first_col:final_col]^2))`
5. Extracts one row per requested pair, in the declaration's own order, so the
   framework's `soc_index` addressing cannot drift off the rows (#447)
6. A requested pair the printed matrix does not carry is refused by name. It used
   to be caught and recorded as `0.0` ("out-of-range indices are protected"), which
   reads to a hopping integrator as "these two states do not couple" - a physics
   statement the log never made. `np.zeros(1)` for a run that read no element is
   gone with it: the channel now comes back with no rows.

The same pair-addressing rule on the NAC channel. A coupling table names no
states - measured on the deployed OpenMolcas 26.10 validation round: five
CASSCF NAC logs, each of which prints exactly one table -
so a row's identity can only come from where the deck put its block. That is
what the deck order says, and #448 made it one function on both sides:
`&ALASKA` is written **once per spin group**, so a run
whose pairs span two manifolds prints its tables in the grouped order - singlet
group first, then each remaining group, each group in request order - while a
table itself still names no states. `input_builder.py::nac_deck_order` is the one
function that says that order, and `log_parser.py` maps the read blocks back
through it, so the rows the framework receives are in the **declaration's** order
and its position-based addressing cannot drift off them. The guard is no longer a
count alone: the block set must equal the requested pair set **as pairs**, and a
mismatch is refused with both named. The skill's own `sorted()` of
`target_nac_pair` is gone with it - sorting the list a second time, in a second
place, re-numbers every declared pair and leaves two index spaces.
(`tests/test_caspt2_espf.py::test_the_coupling_rows_are_addressed_by_the_requested_pairs`,
`tests/test_caspt2_espf.py::test_a_deck_grouped_by_spin_still_returns_the_declared_row_order`
- three pairs over two manifolds with the triplet pair declared first; delete the
remap or restore the sort and row 0 comes back holding the singlet block.) Whether
a real multi-pair Molcas log prints its tables in exactly this grouped order is a
test reading, listed as such in `dev/TODO.md`; it is not claimed here.

### 6.3 Result Format

The `format_results()` function returns a dict with:
- `energy`: np.ndarray (nstate,)
- `gradient`: np.ndarray (nstate, natom, 3)
- `nac`: np.ndarray (nnac, natom, 3)
- `soc`: np.ndarray (nsoc,)
- `coordinates`: list of list of float (natom, 3) in Angstrom

### 6.4 Final Structure Collection (fin_coord channel)

Iterative profiles report their final structure through the framework
`fin_coord` channel so the Calculator updates `state.pmd.system`
coordinates after the skill-internal iteration:

- `casscf_geometry_optimization`: `parse_fin_coord()` reads the relaxed
  structure from `{project}.Opt.xyz` (the lowercase spelling is
  probed as a fallback); the profile also reports the convergence
  verdict through `opt_converge` (`False` when SLAPAF stopped with
  `_RC_NOT_CONVERGED_`). An unconverged run that exhausted its step
  budget may stop without writing the file: it then reports the
  verdict and no final structure instead of raising
- `casscf_abinitio_dynamics` / `casscf_surface_hopping`: reads the
  **last frame** of the `{project}.md.xyz` trajectory
- Other profiles perform no internal iteration and return nothing

Coordinates are in Angstrom, matching the framework contract
((natom, 3)). The key is a transport channel: the Calculator pops it
from the result dict after applying it, so it never leaks into
`state.pmd.results`. Fail-fast policy: a converged iterative run that
finished without its output file, or a frame whose atom count
disagrees with the system, raises (a finished run must have produced
its final structure); only the recorded `_RC_NOT_CONVERGED_` stop is
tolerated.

---

## 7. Molcas Runtime Parameters

Configured in `qualification/pmd_env.yaml` under `skills.molcas_skill`:

| Variable | Description | Default |
|---|---|---|
| `molcas_path` | OpenMolcas installation directory | `''` |
| `molcas_mem` | Memory in MB | `4000` |
| `nproc`, `molcas_nproc` | MPI ranks: the plain name is what a standalone input writes and what the skill reads, the prefixed name what the framework forwards from `&molcas`; the larger of a pair wins (#306) | `1` |
| `molcas_print` | Molcas print level | `3` |
| `molcas_project` | Default project name | `MOLCAS` |
| `keep_tmp` | Keep the calcdir calculation files (0=no, 1=yes); workdir is always removed | `0` |
| `thread`, `molcas_thread` | OpenMP threads per rank, exported as `OMP_NUM_THREADS` (#306 retires `num_thread`) | `1` |
| `scratch_path` | Base path for calcdir and default workdir (`AUTO` for automatic workdir) | `''` |
| `slurm_time` | SLURM time limit | `01:00:00` |
| `slurm_partition` | SLURM partition name | `default` |
| `slurm_ntasks` | SLURM number of tasks | `1` |
| `slurm_memory` | SLURM memory in MB | `4000` |
| `slurm_command` | Custom SLURM command | `''` |

These defaults are also defined in `schema/reference.py` and can be overridden by the config
passed to the backend at initialization.

---

## 8. Molcas Execution Scripts

### 8.1 Local Execution

```bash
<shell commands (module load ...)>
export MOLCAS=/opt/openmolcas
export MOLCAS_MEM=4000
export MOLCAS_NPROCS=1
rc=0
$MOLCAS/bin/pymolcas project.input > project.log || rc=$?
rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
exit $rc
```

### 8.2 SLURM Execution

```bash
#!/bin/bash
#SBATCH --partition=default
#SBATCH --ntasks=1
#SBATCH --mem=4000

<shell commands (module load ...)>
export MOLCAS=/opt/openmolcas
export MOLCAS_MEM=4000
export MOLCAS_NPROCS=$SLURM_NTASKS
rc=0
$MOLCAS/bin/pymolcas project.input > project.log || rc=$?
rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
exit $rc
```

The `<shell commands>` block comes from the config `shell` key —
the skill-level shell (`skills.molcas_skill.shell` of pmd_env.yaml,
fill-if-empty injection; the global `environment.shell` is reserved
for the main framework, #134) — and contains user-specified
`module load` statements. The exit-code propagation pattern
(`rc=$? ... exit $rc`) ensures that the workdir cleanup never masks
a pymolcas failure (#76).

---

## 9. Molcas Test Cases

Test configurations are defined in `tests/` as YAML files. The property
matrix uses the generic `casscf_properties` / `caspt2_properties`
profiles: `jobtype` selects any combination of `energy`, `gradient`,
`nac` and `soc`, and the compiler derives the module chain and the
one- or two-pass plan from those tokens.

### 9.1 Property matrix (23 cases)

CASSCF cases use ethylene (`ano-s-mb`, CAS(2e,2o), 7 inactive
orbitals) unless the states require a larger active space. CASPT2 cases
use water (`ano-l-vdzp`, CAS(6e,5o), 2 inactive orbitals). ESPF cases set
`embedding: charge` and spell their background field inline under
`external_charges` (see 9.5).

The CASPT2 matrix uses `ano-l-vdzp`, not the minimal `ano-s-mb` basis:
with `ano-s-mb` the analytic-gradient step (`IPEA=0.25`, `DORT`) aborts
in the CASPT2 `LUSBT` buffer read with rc=255, while the same input with
`ano-l-vdzp` reaches `Happy landing` (measured locally on
2026-10-27, one `ROOT` case each basis).

The cases that request an analytic CASPT2 gradient or NAC set
`caspt2_ipea_shift: 0.0`, which emits `IPEA 0.0` and no `DORT`. On the
deployment's former OpenMolcas v25.06 the `IPEA 0.25` + `DORT` path
segfaults (rc=-11) at the end of the Lambda solve; the same input with
`IPEA 0.0` lands. The local v26.10 build accepts either setting, so 0.0
is the portable choice for the matrix (measured remotely on 2026-10-27,
`caspt2_ss_gradient`: 0.25+DORT rc=245/segfault, 0.0 rc=0).

The test environment now runs OpenMolcas 26.10 (`/opt/molcas-26`,
the same release as the local build), and the default shift does land
there too: re-running `xmscaspt2_gradient` and `caspt2_ss_gradient` with
`caspt2_ipea_shift` removed gave rc=0 on 2026-10-27, with the IPEA-0.25
energies sitting ~1e-3 Hartree above the pinned 0.0 values
(-76.26014779 vs -76.26109623 for the XMS singlet pair). The pin stays,
because it is what the recorded `expected` numbers were measured under.

| File | Profile | Method | Jobtype | Embedding | States / active space |
|---|---|---|---|---|---|
| `casscf_properties_energy.yaml` | casscf_properties | casscf | energy | gas | `nstate=1`, CAS(2,2) |
| `casscf_gradient.yaml` | casscf_properties | casscf | gradient | gas | 3 singlets, target 0, CAS(2,2), `active: [6, 9]` |
| `casscf_nac.yaml` | casscf_properties | casscf | nac | gas | 3 singlets, pairs (0,1), (1,2), CAS(2,2) |
| `casscf_soc.yaml` | casscf_properties | casscf | soc | gas | 2 singlets + 1 triplet, pairs (0,2), (1,2), CAS(2,3) |
| `casscf_espf_energy.yaml` | casscf_properties | casscf | energy | ESPF charge | `nstate=1`, CAS(2,2) |
| `casscf_espf_gradient.yaml` | casscf_properties | casscf | gradient | ESPF charge | 2 singlets, target 0, CAS(2,2) |
| `casscf_espf_nac.yaml` | casscf_properties | casscf | nac | ESPF charge | 3 singlets, pair (0,1), CAS(2,2) |
| `casscf_espf_soc.yaml` | casscf_properties | casscf | soc | ESPF charge | 2 singlets + 1 triplet, pairs (0,2), (1,2), CAS(2,3) |
| `casscf_prop.yaml` | casscf_properties | casscf | energy gradient nac soc | gas | 2 singlets + 2 triplets, target 2, CAS(2,3) |
| `casscf_espf_prop.yaml` | casscf_properties | casscf | energy gradient nac soc | ESPF charge | 2 singlets + 2 triplets, target 2, CAS(2,3) |
| `xmscaspt2_energy.yaml` | caspt2_properties | caspt2 (xms) | energy | gas | 2 singlets |
| `xmscaspt2_gradient.yaml` | caspt2_properties | caspt2 (xms) | gradient | gas | 2 singlets, target 0 |
| `xmscaspt2_nac.yaml` | caspt2_properties | caspt2 (xms) | nac | gas | 2 singlets, pair (0,1) |
| `xmscaspt2_soc.yaml` | caspt2_properties | caspt2 (xms) | soc | gas | 2 singlets + 1 triplet, pairs (0,2), (1,2) |
| `xmscaspt2_prop.yaml` | caspt2_properties | caspt2 (xms) | energy gradient nac soc | gas | 2 singlets + 1 triplet, target 0 |
| `xmscaspt2_espf_energy.yaml` | caspt2_properties | caspt2 (xms) | energy | ESPF charge | 2 singlets |
| `xmscaspt2_espf_gradient.yaml` | caspt2_properties | caspt2 (xms) | gradient | ESPF charge | 2 singlets, target 0 |
| `xmscaspt2_espf_nac.yaml` | caspt2_properties | caspt2 (xms) | nac | ESPF charge | 2 singlets, pair (0,1) |
| `xmscaspt2_espf_soc.yaml` | caspt2_properties | caspt2 (xms) | soc | ESPF charge | 2 singlets + 1 triplet, pairs (0,2), (1,2) |
| `xmscaspt2_espf_prop.yaml` | caspt2_properties | caspt2 (xms) | energy gradient nac soc | ESPF charge | 2 singlets + 1 triplet, target 0 |
| `caspt2_ss_gradient.yaml` | caspt2_properties | caspt2 (ss) | gradient | gas | `nstate=1`, target 0 |
| `rmscaspt2_gradient.yaml` | caspt2_properties | caspt2 (rms) | gradient | gas | 2 singlets, target 0 |
| `xdwcaspt2_gradient.yaml` | caspt2_properties | caspt2 (xdw) | gradient | gas | 2 singlets, target 0, `dwtype=3`, `dwms_zeta=1.0` |

No charge document ships in the skill: 9.5 covers where the field comes from
and how the file is staged beside every leg.

`casscf_espf_nac`, `casscf_espf_prop`, `xmscaspt2_espf_nac` and
`xmscaspt2_espf_prop` exercise the two-pass ESPF path: pass 1 runs the
ESPF energy/gradient/SOC work, pass 2 recomputes the NAC without ESPF.
Gas-phase NAC is a single pass. NAC is requested with `NOCSF` for both
methods; CASSCF and CASPT2 couplings must not include the CSF
contribution.

### 9.5 The background embedding of the ESPF cases

A case declares its embedding inline, in either representation, and never both:

```yaml
  external_charges:                                  # arbitrary external points
  - [1.0, 0.0, 0.0, 5.0]                             # q, x, y, z in e / Angstrom
  - [-1.0, 0.0, 0.0, -5.0]

  external_potential:                                # one row per QM atom
  - [-0.135774924, 0.0173259355, 0.0, -0.0421938884] # V, Fx, Fy, Fz in au
```

`converter._stage_charge_sidecar` writes it once, at config-generation time, as
the file every carrier then reads - `<title>.charge`, behind a declaration line
(`point angstrom external` or `field au`) - and replaces the inline values with
`read_charge: true` in the generated `.json`, `.config` and `.input`. No carrier
runs on inline rows: the framework's `System.read_charge` is a bool switch that
resolves `{title}.charge`, and the skill's own `read_charge: true` alias expands
to the same `<project>.charge`, so one document feeds all three legs. Two kinds
in one case are refused by the converter, and a result never prints both (the
calculator records one and the log prints whichever arrived, or neither plus a
named conflict).

`embedding` is not part of this path. It is a multiscale switch; a
single-theory run does not read it, so these cases carry neither value and the
file is the whole embedding decision. The two are also what makes a QM
embedding single point runnable without an MM calculation.

**Two charges of opposite sign, not one.** A single-row fixture would pass a
reader that only ever consumed the first row of a file, and the declaration
line is exactly the part a permissive reader gets away with ignoring. The field
is neutral, so nothing in it depends on how a charged background is accounted.

**The field twins.** Every `*_espf_*.yaml` case has a `*_espf_*_field.yaml`
twin: the same molecule, states and jobtype, with the two external point
charges replaced by the potential and field they produce at each nucleus, in
atomic units (V_i = sum_j q_j/r_ij, E_i = sum_j q_j (R_i-R_j)/r_ij^3, r in
bohr). That is the V/E block a QMMM field stage hands over, so the field case is
the same physical environment in the other representation and needs no invented
numbers. The twins are the second of the two `read_charge` case sets.

**Measured on the local OpenMolcas 26.10** (ethylene, CAS(2,2), one root, the
two-charge field above):

| case | energy | note |
|---|---|---|
| `casscf_properties_energy` (no embedding) | -77.73157779 | reference |
| `casscf_espf_energy` (point charges) | -77.73867546 | |
| `casscf_espf_energy_field` (V,E) | -77.73867546 | same to the printed digit |
| field set to zero | -77.73157779 | equals the reference exactly |
| field negated | -77.73721850 | moves, so the block is live |

The zero-field row is the channel's no-op witness: a field silently dropped
would land on the reference number too, so the negated row is what separates
"dropped" from "used". The equality of the point-charge and V/E runs is a
measurement for this field, not an identity: the two representations differ in
general by the multipole terms above the dipole, which for a neutral density
and a field source 5 Angstrom away do not reach the eighth decimal here.

The whole set runs on all three carriers: 20 cases (charge and field), 60 engine
logs, every array equal across the carriers (`tools/verify.py` section G:
`leg diff max 0.0e+00`, problems 0), and the two framework carriers print
exactly one background block - `&external charges in e` for the charge set,
`&external field in au` for the field set, never both in one log.

### 9.2 Legacy cases

| File | Profile | Method |
|---|---|---|
| `hf_energy.yaml` | hf_energy | hf |
| `casscf_energy.yaml` | casscf_energy (absorption) | casscf |
| `casscf_frequency.yaml` | casscf_frequency | casscf |
| `casscf_geometry_optimization.yaml` | casscf_geometry_optimization | casscf |
| `casscf_meci.yaml` | casscf_geometry_optimization (MECI, 5-step functional test) | casscf |
| `casscf_mecp.yaml` | casscf_geometry_optimization (MECP, 5-step functional test) | casscf |
| `casscf_abinitio_dynamics.yaml` | casscf_abinitio_dynamics (10-step FOREACH MD) | casscf |
| `casscf_surface_hopping.yaml` | casscf_surface_hopping (10-step FOREACH MD) | casscf |
| `caspt2_energy.yaml` | caspt2_energy (absorption) | caspt2 |

The SOC fixtures use CAS(2,3), not CAS(2,2): the SOC workflow runs a
second `&RASSCF Spin=3 CIRoot=2` job and a CAS(2,2) holds exactly one
triplet CSF, so Molcas stopped with "You can't ask for more roots than
there are configurations" after a full ground-state run (measured on
the deployment, rounds `tmp_20261010_133424`/`tmp_20261010_190544`).
`tests/test_skill.py::TestCISpaceLegality` now counts the spin-adapted
CSFs of the declared space and refuses the request before an engine ever
sees it.

The former per-skill `dev_tool.py` was replaced: `qualification/certification/converter.py` owns the skill test YAML parsing (`parse_test_yaml` / `generate_test_configs`), renders the classic `.input` (`convert`) and serializes the agent `.config` (`convert_to_pyrai_config`), so these YAMLs are read by `pyrai3md test`. The `expected` block documents result shapes only; the runner does not consume it.

### 9.3 Post-round verification

The shapes above are what `tools/verify.py` section G holds a round to. The
audit reads the finished round only - it never runs a skill or an engine. It
parses each of the three carrier legs' own engine log with the skill's parser
and requires the energy, gradient, NAC and SOC arrays to agree across the legs,
then asserts the shape, pass-count and energy-source contract of each property
(measured on round `tmp_20261027_173607`, deployment OpenMolcas 26.10: 24
cases, 72 legs, worst cross-leg difference `0.0e+00`, no contract problem). The
detector was checked against itself on the same round: editing one printed
digit of `casscf_gradient`'s input-leg log by 1.9e-05 named that case and left
the other 23 cases at zero.

Two properties of the engine are worth reading before a coupling is consumed by
a dynamics step. The reported energy of a CASPT2 case is the eight-decimal
engine line, so a cross-leg agreement at 0.0 means agreement to that printed
precision and no more. And a non-adiabatic coupling vector carries the arbitrary
phase of the two CI vectors it comes from: the same geometry re-run under the
next engine release inverted one whole vector (`casscf_nac`, pair (1,2): the
first atom's component read `+10.26123` under v25.06 and `-10.26122` under
v26.10) while every energy stayed within 1.5e-6. The matrix asserts shapes, so
a phase flip stays green by design; a hop that needs a sign has to fix the
phase itself.

### 9.4 Engine portability of the CASPT2 shift

The measurement recorded in 9.1 still stands: `caspt2_ipea_shift: 0.0` is what
the `expected` numbers were taken under. It is no longer a crash workaround on
the test environment, which moved from `/opt/molcas` (v25.06) to
`/opt/molcas-26` (v26.10, the same release as the local build):
the two gradient cases re-run there with the key removed both reached `Happy
landing`, their IPEA-0.25 energies sitting about 1e-3 Hartree above the pinned
values (`xmscaspt2_gradient` singlet pair: -76.26014779 against -76.26109623).
Dropping the pin would rewrite nine cases' recorded numbers, so it stays until
that matrix is retaken.

---

## 10. Key Differences from Other Backends

1. **Text-based input**: Molcas uses `.input` files with `&MODULE` blocks, not JSON
2. **Coordinate reference**: Uses `$MOLCAS_PROJECT.xyz` variable, not inline geometry
3. **Dual-directory model**: calcdir (calculation files, kept per `keep_tmp`) is separate from workdir (temporary files, always removed; AUTO resolution to `/srv/tmp` or `/tmp`)
4. **Dual RASSCF for SOC**: Requires separate singlet and triplet RASSCF calculations before RASSI
5. **Multiple orbital files**: StrOrb, RasOrb, JobIph for different purposes
6. **ESPF embedding**: `embedding=charge/field` emits `&ESPF` after SEWARD (`potential` is retired as a keyword spelling; a charge *file* may still declare `potential au`, which reads as a field); XField is retired
7. **RICD approximation**: Required for all calculations (Resolution of Identity for Coulomb)
8. **AMFI/rela keywords**: Required for SOC calculations (relativistic effects)
