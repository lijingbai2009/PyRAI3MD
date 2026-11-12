# PyRAI3MD TODO List

Last updated: 2026-11-12 13:32 (reorganized: every closed item is now one bullet
carrying its decisive reading and the round that owns the full narrative;
open items keep their text, all entries are in English. The chronological
stamp chain that used to sit in this header is replaced by "Recent passes"
below, and the round-by-round record stays in `CHANGELOG.md`.)

Recent passes (the CHANGELOG's own `Last updated` header carries the fuller list):

- #464 whole-tree audit: three "remove it" candidates closed as keep; local
  5087 passed / 3 skipped.
- #465 QM/MM field hand-back converts the native V/E block through the skill
  manifest: the eighteen remote QMMM failures were exactly that omission; two
  rounds back at 149/0 and 139/0.
- #466 `multicenter_md`'s empty kTDC NAC fixed in `_project_carrier` (a round
  then read both carriers PASS), and `md_lammps_nve`'s `etot_drift` gate set to
  `5e-3` on measurement.
- #467 benchmark reference-arm token fix, 25/25 remote PASS; full round
  `tmp_20261108_100548` 1092 rows / 1090 PASS / 2 FAIL (the two registered
  #236 R2 probes) / 0 Error.
- #477 the `ff_npt_pbc_off` probe case retired by ruling.
- #479-#488 MD stop codes, the adaptive quality gate, `history_v%i.json`,
  `gNN` support in `verify.py`, `max_in_flight` default 0, README; full round
  `tmp_20261111_195518` 1114/1114 PASS.

This document is the **current work list**, grouped by "what state is this in
right now" rather than by a Phase progress table. Every entry points at a code
line or a CHANGELOG round so it can be checked rather than remembered. Closed
entries are kept only as one-bullet summaries; the evidence lives in
`CHANGELOG.md` and in the documents each bullet names.

---

## 1. Directions that were priced and permanently rejected (do not re-introduce them as a "quick optimization")

These are not "not tried yet"; they were tried and they got numbers
(narratives: CHANGELOG #244, #259, #263, #314, #445, #464).

- **The 2026-11-06 audit's three "remove it" candidates are closed as keep**
  (#464): `workflow/dispatch/subprocess_run.py` still has a direct-call entry of
  its own (6.10); there is no skill-to-framework callback (3.10) because a
  skill's `calculate` returns a dict the framework reads, so the "missing"
  channel is a design boundary; ORCA's `MAX_CORRECTION` (8.6) is still read at
  both 0.015 and 0.045 (`test_orca_correction_bound.py`). The fourth candidate,
  splitting `verify.py --selftest` (7.12), is partly done - six nested writers
  moved to module level and eighteen sections became `_plant_section_*`
  builders (1580→220 lines, `--selftest` line-identical) - and isolating the
  rest would rewrite the audit's oracle rather than restructure it.
- **The QMMM field probe's residual guard stays at `1e-4 * scale`** (#445,
  priced on round `tmp_20261031_220625`): h2o/pppm clears `1e-6 * scale` with
  34x to spare (residual under 3.9e-6 on a scale of 134.4), while
  c2h4_water25/ewald leaves 2.2e-4 against `1e-6 * 168.49 = 1.68e-4`, short by
  1.3x. Any future tightening must be priced on the ewald box.
- **`run 1` cannot serve as the per-step evaluation line on this path**
  (#263): it really does skip the startup (`PPPM initialization` 201 times → 1,
  step tail 1.624 → 1.135 ms) and is bit-identical once an `unfix` line is
  added, but inside a real 2000-step example the two trajectories fork (1e-15
  from step 8268) because the persistent instance keeps incremental
  neighbour-list updates while `run 0` rebuilds, and this path's contract is
  **byte**-identical `md.energies/pop/velo/xyz`. The whole step is 4.740 →
  4.249 ms = 1.116x, below the 1.15x adoption gate.
- **`run 0 pre no post no`** (#244): completes 10000 steps, changes the
  `md.energies` summary, saves no milliseconds. #263's mechanism: `run 0`'s
  single evaluation happens in the pre stage, so `pre no` deletes the
  evaluation itself - not the same thing as with `run 1`.
- **`thermo 0`** (#244): bit-identical yet 2-3% slower.
- **Removing the `deepcopy` from `_build_history_frame`** (#244): the arrays
  under `results` are the live state's own objects; without the copy the history
  frame is overwritten by the next write-back. A semantics change for 0.48
  ms/step.
- **The full-tensor `scr` branch** (#259): unstable on a real engine; removed,
  and validation now refuses `scr` + `anisotropic`.
- **A gradient-descent integrator is not on the roadmap** (#314): zeroing the
  velocities every step is a different dynamics, not a slower MD, and
  `jobtype: optimize` already owns the question. The keyword is deleted and
  `RETIRE_DYNAMICS_KEYS` refuses the spelling.
- **`popreset` was rejected on purpose** (#314): an out-of-range population is
  reported and propagated (`integrator/fssh.py::_diagnose_population`), and
  collapsing to a pure state would destroy the evidence that diagnostic shows.
  Refused by name.
- **Auto-exporting `LD_LIBRARY_PATH` in the lammps run script**: ruled out by
  the user; the environment is solved by configuration, and lammps has no
  `*_ROOT` export block by choice, not by omission.

## 2. Items that need real-machine evidence before they can be closed (the local criteria are maxed out)

Closed this cycle (one bullet each; the narrative is in CHANGELOG):

- **`multicenter_md`'s kTDC coupling came back empty on a real engine (#466) -
  fixed and re-read.** Round `tmp_20261107_161909` stopped at
  `workflow/multiscale/regions.py:288 project_nac` with
  `Local NAC must have shape (P_nac, A_c, 3) or (A_c, 3), got (0,)`;
  `_project_carrier` now reads an empty NAC as "no coupling rows" while a
  malformed non-empty shape is still refused, and round `tmp_20261107_215139`
  read both carriers PASS.
- **`md_lammps_nve`'s Etot band (#466) - measured, not widened blindly.** The
  hot and cold execution paths (`python_enabled` off/on) differ by at most
  `1.35e-7` in Etot on the same fixture (500 steps / 21 frames), so the drift is
  velocity-Verlet physics rather than a path defect; the earlier "swap the seed"
  idea was wrong because the velocities are a hard-coded equilibration frame.
  `ENERGY_TOLERANCES['etot_drift']` and `pmd_env.yaml`'s `energy.etot_drift`
  both went to `5e-3` (2.9x over 1.73e-3, still catching the 5.6e-2-7.0e-2
  failures), and the remote case was rerun.
- **#33 closed (#446, round `tmp_20261101_155943`): the branch plane is two
  directions wide.** `multiscale_qmmm_meci` prints `Branching Directions
  2.00000000` with `Coupling Norm 0.18258742 / 0.24511158` and a closing gap
  (`0.38965312 -> 0.31844597`); the three links fixed are in
  `dev/KNOWLEDGE_BASE.md` Q68. Only the convergence of that search at
  `maxit: 2` stays open, and it is a structural criterion.
- **The all-zero-pbc refusal (#236 R2) is closed by round
  `tmp_20261102_114641`; the probe case was retired by ruling on 2026-11-11
  (#477).** The refusal and its unit coverage stay; the end-to-end case was
  retired because every full round carried its two intended red rows.
- **The parser state-dependence entry (#399) is closed**: both unit tests pass
  on this tree (2 passed in 1.56 s, measured 2026-10-20) and
  `interface/parser.py::_section_path` reduces a `&molecule coord` list to one
  path (`parser.py:252-255`), so the path branch can no longer receive a list.
  The narrower claim about one directory order was not re-bisected; the
  full-suite gate is green over the default order.
- **The shared-directory collision (#399) is fixed in the planner**
  (`certification/planner.py:483-500` gives every carrier leg its own
  `work_dir/<skill>/<mode>/`). What is still missing is a round that runs the
  six `nn_pes_skill_*_train_{agent,input}` cases since the fix; the current
  posture closes them.
- **"The LAMMPS water fixture's stress is not the derivative of its energy" was
  withdrawn (2026-10-13)**: the factors of 0.02-7.9 reproduce only as the
  Cartesian path (atoms pinned while the box moves); with the atoms riding the
  box the same engine agrees with `V sigma^T H^-T` to 0.984/0.997/1.002/1.071/
  0.996/0.998 at a relative strain of 1e-3. The two paths differ by exactly
  `sum_i g_i[b] S_i[a]`. The optimizer now rides (`optimizer/base.py::_cell_ride`)
  and `dev/methods/CELL_OPT_DEVELOPMENT.md` section 5.3 carries the table.
- **The shipped water fixture cannot be the acceptance vehicle for a
  converging variable-cell relaxation - closed as a blocker (#341)**: the 25 A
  cubic TIP3P box (525 waters, 1575 atoms, cutoff 10.0 A against an 11.634 A
  half extent) converged in 799 steps. Its `E(L)` floor of `5.88e-5` Hartree
  still has to be respected by anything that prices a derivative on it, and
  two readings stay on record: the 0 K density is 1.161 g/cm^3 because the
  oxygen LJ epsilon is written positive, and the cell criteria were
  size-dependent until #363 moved them to bar/strain.
- **`bringup` C5 repriced against the equilibrated box (#367)**: the three
  sequences are now built as fractions of the box `_data_file_lattice`
  reports; re-measured rows `8.3991 / 8.3991 / 8.3991` A with `max|dG|`
  3.638e-7 / 4.098e-7 / 4.098e-7 against the unchanged 1e-6. Two side
  readings: `tilt -> untilted` is measured at `dE 6.567e-5` / `max|dG| 1.930e-5`,
  and C10 was not byte-clean on that build (last digit of one energy and two
  stress components, relative 1e-15), kept as an environment divergence.

Still open:

- **The probe's pass margin is not auditable from what a run leaves behind**
  (#445): `assemble_qmmm_field` returns `sanity.v_vs_u_gg` and
  `sanity.e_vs_f_gg`, and the guard speaks only when it refuses, so a passing
  round records no residual - the reading had to be recomputed from
  `*_qmmm_field.esfp` against the run's own `c_QMMM_GG` tally, with a 5e-7
  floor from the log's eight digits. Writing the two numbers into the per-step
  result file (or into `approximations`) turns the margin into a reading.
- **The per-step cost comparison basis for persistent versus cold paths - open,
  and not worth a gate.** #448 rebuilt the alternating-pair instrument but took
  the reading out of round and unapproved, so it is not adopted. No "before"
  tree exists, so only a level is comparable; and the level is per engine call
  on `ff_nve_test` (60 atoms, the `md_lammps_nve` box), while #244's 4.20 ms
  and #262's 4.696/4.879 ms were framework MD steps with per-step file writes
  under `debug: true`. Do not take it up again outside a round.
- **The `var(ln V)/analytical` ratio of `md_lammps_npt_csvr_scr`** (#257): the
  toy box inflates it to 1.19-1.43 at `N_df = 15`; on the 60-atom box 20 ps
  gives about 20 blocks whose variance estimate carries ~32% relative standard
  error, so the reading cannot gate at any affordable length. Keep as an
  ensemble-property question, not a qualification gate.
- **ONIOM2 `embedding: true` on the openqp high-level carrier still moves no
  energy** (#444). Round `tmp_20261102_141755`: the embedding and
  no-embedding twins print the same three state totals to every digit
  (-82.4745309685 / -82.2523427404 / -82.3817271295) and the high-level deck
  carries only `charge=0`. The electrostatic route ships on carriers that do
  consume the rows (ORCA/BAGEL/Molcas high with PySCF/xTB as source). Two
  limits recorded: openqp has no point-charge writer, and ORCA drops NACME
  under embedded charges (the embedded `nac` carrier is Molcas).
- **`multiscale_qmmm_mep` records one path point, not the band (#448 finding,
  needs its own round).** Round `tmp_20261102_114641` declared
  `opt_neb_images 8` with `opt_maxit 2` / `opt_path_inner_maxit 2`, ran 16 s,
  **passed**, and left `path_profile.txt` with one row and `path_geom.xyz`
  with one frame while printing `Converged: False` beside "converged in 2
  steps". Two questions belong to that round, in order: whether a walk with
  `opt_maxit` 2 should have written more than the start, and whether both
  lines can be true of one run (if not, the status line is the defect).
- **The `Lattice=` comment spelling and units of cp2k (#236) is a missing
  PRODUCER (#399).** The reader (`cp2k_skill/service/parser/log_parser.py::
  parse_xyz_cell`) reads the comment line of CP2K's own `{project}-pos-1.xyz`,
  but no shipped profile writes that file: round 20's cp2k points are
  single-point inputs, and the one engine-written xyz carries the bare comment
  `CP2K`. Closing it needs a cp2k skill-side geometry-optimization leg with
  `keep_tmp`; adding `&MOTION` to the single-point strain batch is a design
  change, not a fix.
- **Aligning numpy/scipy in the test environment** (#255/#258): either
  `pip install -U "numpy>=1.25"` or downgrade scipy; the user decides. Without
  alignment the pyscf skill tests and `example_energy`/`example_gradient` do not
  start.
- **The before/after-compression equivalence gate**: whether those 449
  criteria run is settled (`_runall: true` proved it in rounds 6/7/8; #358 set
  the file to the optimize round), but the "bit-equivalent before and after
  compression" comparison itself has never been made.

### QMMM RASSCF non-convergence (2026-11-12, resolved)

Round `tmp_20261112_174534` failed 22 multi-center QMMM examples because the
second center's Molcas RASSCF stopped at `_RC_NOT_CONVERGED_` after 200
iterations, so the gradient task has no `Molecular gradients` block and the
new requested-block contract raises. Evidence:

- `multicenter_qmmm_sp` center 1: charge task converges in 31 iterations,
  gradient task in 33; center 2: both the charge-only and the gradient task
  stop at 200 with no convergence.
- The two centers have separate guess files (different `guessorb.h5` sizes and
  hashes; different `GssOrb` hashes; no `fileorb` in either input), so this is
  not center 1's orbitals being reused for center 2. The early iteration
  trajectories are nearly identical; center 2 then oscillates between two
  energies (~-76.14927 / -76.14955) with QN/LS step alternation.
- The two waters have different positions/orientations and different ESPF
  external fields; the failure is in the active-space/CI solution under center
  2's field, not in the parse or the ASKAA/ALASKA part.

**Fix and evidence.** State averaging through the existing template
`CIRoot={ci_root} {ci_root} 1`:

- `ci: "1; 2"` fixed every failing example at the base geometry, but
  `multicenter_qmmm_num_freq` then failed at its first finite-difference
  displacement: center 1 (still `CIRoot=1 1 1`) stopped at 200 iterations with
  the same QN/LS oscillation.
- Rerunning the exact failing input (`runtime_2/tmp_high_qm_task1_MOLCAS`,
  same `guessorb.h5`) with `CIRoot=2 2 1` converged, `rc=_RC_ALL_IS_WELL_`.
- The shipped multicast examples now declare `ci: "2; 2"` and
  `multiscale_qmmm_neb` declares `ci: '2'`. The four family examples that were
  already declaring a pair (`multicenter_qmmm_mecp` `state_i`/`state_j`,
  `multicenter_qmmm_nac_sp` `coupling`, `multicenter_qmmm_ex_opt`,
  `multicenter_qmmm_md_fssh` `md.root`) had been shipping `ci: "1; 1"`,
  which contradicted their own declarations; `"2; 2"` aligns them.
- Targeted round `tmp_20261112_201015` (`_runall: condition`, the 11 QMMM
  examples, runtime tests open): 240 rows, 236 PASS, 0 FAIL, 0 Error, 393
  tasks skipped by the narrowed switches. Final-tree full round
  `tmp_20261112_213944`: 1129/1129 PASS and `tools/verify.py` verdict clean.
  The requested-block missing-gradient raise stays.

## 3. Built but not yet wired

### 3.1 Open

> **Module-level management boundaries (2026-11-12).** The pre-commit review
> considered splitting four long functions. They stay intact: each is the
> internal orchestration of one dedicated module - the FSSH integrator
> (`workflow/integrator/fssh.py`), the QMMM composite assembly
> (`workflow/multiscale/composite.py`), the MD trajectory loop (`workflow/md.py`)
> and planning task assembly (`planning/assembly.py`). The module is the
> management unit; no subpackage or cross-module split is introduced.

- **The numerical Hessian's batch path disagrees with a direct single-point
  difference (#460, blocking).** On the composite surface, atom 9's `H_xx` is
  3.2183 from the batch path against 0.4011 from a direct ±0.005 Å single-point
  difference; on the C=1 arm the diagonal of the two H's is ~1e-4 from the batch
  path against 0.23. Ruled out: the displacements are generated correctly
  (`generate_batch` prints ±0.005 Å and the row order aligns with
  `_batch_rows`); the electronic surface responds (the same ±0.005 Å moves
  `E_QM` by 6e-4 Hartree); the constraint block changes nothing (0.401 against
  0.4011); and a clean directory reproduces bit for bit. So the gap is on the
  **batch/assembly side**. Next criteria (already written into #460): (1) on one
  geometry, compare the 18 children's read-back coordinates with the
  coordinates `generate_batch` handed out, bit for bit, to separate "the child
  never got the displacement" from "the gradient difference is overwritten by
  another quantity"; (2) use a non-QMMM periodic box with known analytic
  curvature as the control arm. Until it closes, `multicenter_qmmm_num_freq`
  and #458's C=1 pair readings count only as "channel carriers".
- **`sc_charge_damping` has no shipped example (#459/#460).** The key, its
  range (default 1.0, (0,1]), the log line and the criteria all exist, but no
  box in the tree has an undamped iteration that diverges (the shipped
  `sc_charge` converges in two rounds), so there is no engine reading that
  would prove it useful. Either add a case when a strongly coupled box turns
  up, or treat it as a switch with no scenario and demote it to a documented
  practice rather than a framework key (pending a ruling).
- **`pyrai3md --info` prints nothing: `pmd.py::print_state_info` is the
  `# TODO` placeholder** (#448). The flag is accepted, `run()` reaches the
  `info_only` branch, calls a body that is `pass`, and returns `None`, so the
  exit code looks like success. #448 read the invariants by walking the loading
  chain instead (`property_request_spec` over 44 examples, 0 violations).
  Wiring it means printing the resolved workflow, the task queue and the
  per-step property request, plus a criterion that a `--info` run writes no
  artifacts.
- **What `opt_trust_radius` cannot buy is measured (#396), and one reading is
  still owed.** The keyword reaches the construction site (six sites), and on
  the LAMMPS `water25` box the residual floor improves 1058.4 -> 220.189 bar
  while the 0.5 Bohr ceiling is still hit inside three steps. Owed: a
  **converged** pressurised cell relaxation on the xtb 81-atom carrier that
  reached 181.527 bar between two frames - this environment has no xtb, so it
  is a round attempt.
- **The md family's own classic carrier has no parallel reading (#398).** Round
  20 opened only `examples.optimize_test`, so `md_batch`/`md_parallel` appear
  only in `disabled by` lines. The writer fix they would certify landed with
  #395 and is measured on the two cp2k carriers (581 s yaml / 584 s `.input`,
  six simultaneous strain pairs); what is unmeasured is the md family.
- **Whether a cell example may pass unconverged is a `maxit` ruling, not a
  recording question (#399).** All 32 variable-cell legs stop at their `maxit`;
  both exit routes now carry `[NOT CONVERGED]` and `not_converged`, the verdict
  stays `pass`, and `verify.py` C2 counts them as `budget_stopped` (reads 32 on
  round 20).
- **The 1700 bar readings and the isotropic twin's own drift are still owed**
  (#389/#391). The target moved from 1000 to 1700 bar by ruling - an
  extrapolation from the 1688.1 bar the box measures on its own - and 1700 bar
  is not a reading yet because the deployment is offline. #391 replaced the
  direction clause with two sign-free readings (the residual must fall from the
  first recorded step to the last; a loaded drift must be smaller than its
  zero-pressure twin's, with "no twin"/"two twins" named rather than picked).
- **Multi-center variable-cell runs (C > 1) are out of scope for now** (#388):
  the code supports them (a C>1 assembly's term columns carry a per-term state
  index), but no shipped example asks a multi-center box to move, so the shape
  is covered by criteria only.
- **A three-layer plan still needs a middle-layer engine.** The rows are
  certified at plan level
  (`TestTheRegionTaskIsHandedItsRows.test_a_three_layer_plan_hands_the_mid_shell_rows_too`),
  but this environment has no QM method that can serve as the middle layer
  (xtb and cp2k are absent), so no three-layer energy number is claimed.
- **The cell force for a bent condition is measured; the converged box is not
  (#352).** A model whose stress is exactly the held angle's own lattice
  derivative gives zero cell force with the term and that derivative without it
  (six entries; sign flip, transpose and double scaling each turn two arms
  red). The endpoint needs a self-consistent solve over the six slots, and a
  wrong sign would still leave a fixed point somewhere, so it was not bought
  with a guess.
- **CP2K's two-manifold S/T route has three open parts (#350).** (a) Whether a
  triplet root can be given forces at all in this build -
  `tests/tddft_st_forces.yaml` is written as that question, and until it comes
  back the cp2k `mecp` route cannot be planned. (b) The second run recomputes
  its SCF instead of restarting the first's: `WFN_RESTART_FILE_NAME` would skip
  the work, but CP2K's own "read the wavefunction" marker has not been read off
  a run, so nothing is claimed. (c) Spin-orbit corrected spectra as their own
  capability (`tddfpt_soc_energies` route, its own profile and output section).
- **Three multi-scale placeholders wait on one undecided thing - the cap /
  link-atom format, i.e. the ONIOM region decision.** The capped-atom gradient
  projection in `composite.project()` is wired but inert
  (`workflow/calculator.py:1328` calls it; every task ships `transform_spec:
  None`), `region_charge` for a model region is still `None`
  (`planning/task.py:316`), and QMMM-specific coupling mathematics has generic
  aggregation and ready task expansion but no format either. Nothing was
  invented in their place.
- **The reverse restart chain is closed locally and open on an engine** (#314):
  a 12 + 12 split reproduces `md.*` byte-identically on the qdm model potential
  and `test_md_restart.py` holds it, with the generator position carried in
  `RuntimeState.random_state`. What remains is the same pair on the LAMMPS
  persistent and subprocess channels on a deployment machine.
- **`nac_type: dcm` (and `nacme`'s cross-step overlap) is refused, and what is
  owed is the `system2`/previous-step route** (#329/#354). The skill has no
  field for either the orbital json or `[guess] file2`, so the demand is
  refused by name. #354 removed the directory-collision half (a calc dir is
  now named by its declared `project`); the substance remains, plus what
  `nacme` does on a first step (today it routes to `runtype=data`).
- **OpenQP's own Hessian route hands back 12 modes with a spurious imaginary
  for a linear molecule** (measured 2026-10-11 on
  `hf_frequency_test.hess.json`): acetylene is linear, so the count should be
  `3N - 5 = 13`; the first of the twelve is `-124.419 cm-1`. The framework's own
  route already handles this by rank (`workflow/hessian.py`, #323); the gap is
  the skill-side removal in `openqp_skill`, and the fix belongs with the
  linear-molecule work rather than as a special case.
- **The Hessian batch has no `remotetask` evidence** (2026-10-10): the serial
  and `multitask` pool arms agree with PySCF's analytical matrix, but no SLURM
  submission was made, so the remote arm's `collect` is only known to be
  code-identical. Verify with one two-displacement job before quoting a remote
  Hessian number.
- **`ts` under a constraint plan, on a multi-center partition, or with more
  than one electronic surface is untested rather than refused** (#335): the
  search steps on the whole system's curvature and nothing exercises those
  combinations.
- **A single Cartesian component pinned on a molecule in vacuum has no
  rotational meaning** (#336 refused the search case): the torque left on free
  components sat at 2.08e-2 Hartree per Bohr with the step already at 1.3e-7,
  so a curvature search accepts whole-atom pins only. One-axis cases (surface
  normal, cell direction, applied field) are unimplemented rather than refused.
- **Two reproducibility readings are on record and unowned.** (a) The same
  geometry and input give `E_QM(center 2) = -76.14530329` in a clean directory
  and `-76.14591099` in a directory that has run once (6.1e-4, reproducible
  within one directory) - a leftover `MOLCAS.GssOrb` gives another CASSCF
  solution, and `keep_tmp: 0` narrows the band from 7.86e-3 to 2.86e-3; when
  chasing "same input, two readings", check directory state first. (b)
  `num_freq` used 93% of `_timeout: 1200` at 18:39, so raise the budget before
  adding batches.
- **`<title>.rt`'s second segment is still the hand-run command printed in the
  header**; a qualification round runs only each segment's first process (the
  same posture as constraint restart). "Refuse a restart whose fingerprint
  disagrees" has no second fingerprint source at the `.rt` entry, so the
  internal consistency that can be refused (frame fingerprint, constraint
  digest, version block) lives in `serializer.load` and no new refusal was
  invented.
- **Scope note (not owed): C2 compares one row and can only compare one row.**
  A backend's calculation directory is rewritten at every evaluation, so a
  round leaves one engine number on disk. Per-step engine exports would need
  the directory name to carry the step, which is a product-layout change
  rather than an audit change.

### 3.2 Closed (summaries)

- **Round `tmp_20261105_112532` / `tmp_20261105_125614` (#460).** The two
  failures were the classic `.input` carrier passing `"[1, 1]"` to
  `optimize.opt_state_i`; the writer now emits space-separated values, the
  reader has an explicit branch, and `validate_centers` tolerates a list.
  Re-read: 184 items / 180 pass / 0 fail / 0 error, the 10 non-`completed` legs
  all declared `maxit` stops.
- **Dispatch keywords in the classic input (#394/#395/#398).** `multitask` /
  `remotetask` / `multipool` are now emitted presence-driven (92 of 96
  generated inputs byte-identical, 4 changed); the two cp2k `.input` legs
  finished in 584 s against 815 s before, with six simultaneous strain pairs
  visible in the engine's own `PROGRAM STARTED/ENDED` stamps. A skill's trio
  no longer enters the flat namespace `&md` owns.
- **The constraint subsystem (#264-#280).** Everything that can be written is
  written (bond-type expansion through `utilities/bonds`, well-tempered
  deposition, the velocity half of angle/dihedral conditions, D-17 at 11.37 →
  1.73 ms per step at N=108). Real-engine baselines ran for every written
  capability, including periodic NVT holding a bond to 8.131e-11 A and NPT
  degrading it to 4.806e-05 A under a moving cell. Open items are listed by
  blocker in `dev/methods/CONSTRAINT_DEVELOPMENT.md` appendix B §10: the
  analytic bias Hessian (accuracy/overhead), the `molecule` grouping spelling,
  the open-face wall shape, a fourth condition kind (refused by name until
  measured), and D-24 (a continuation changes the recorded-frame cadence).
- **One keyword vocabulary (#287) and the five rulings (#281-#285).** The flat
  classic design is shared across all three carriers; retired keys are refused
  by name with the replacement offered; list semantics are one element = one
  object. D-19/D-19b/D-24 closed (absolute step numbers; frame and `.rt` share
  one predicate), D-26 closed under wording A (`projection.declared_mask`),
  D-27's default direction corrected, D-18 withdrawn (the two readers were the
  same key) and replaced by the `&constraint energies` block, D-28 fixed in
  `bonds.py`. A5/D-14 closed for a bond, an angle and a torsion.
- **Multi-center periodic regions (#315-#317).** Every recorded frame is
  measured against half the shortest cell extent
  (`_check_region_reconstruction`), and the low layer evaluates a center slice
  (the LAMMPS skill cuts the user's templates to the region's rows):
  `examples/multicenter_pbc_*.yaml` ran in 4 s (NVE, 4 steps) and 8 s (NPT, 50
  steps), each center's `L(c)` agreeing to 1.6e-16 Ha through three routes.
- **The `stress` request reached center tasks (#317, fixed #318).** One
  capability set is now imported from the resolver and `_task_capabilities`
  drops whole-system properties from a slice task; the box force now reads
  `Pvir_bar = -6050.718 bar` at step 1 (+275.981 at the last) and the box
  contracts 0.96% instead of expanding 0.97%. The modelling half stays as
  stated: a center's motion does not contribute to the box force by ruling, and
  a periodic excited-state stress is single-shot only (CP2K `stress_method:
  num`, #322) and feeds no barostat.
- **Variable-cell coverage (#355).** Seventeen `optimize_cell_*.yaml` examples,
  and the ledger's six items are closed: (1) #363 judges the cell half on
  `opt_csrms`/`opt_csmax` (bar) and `opt_cerms`/`opt_cemax` (strain); (2) #367
  pins all six slots for a held angle or torsion; (3) C5 repriced by #367; (4)
  the gap's lattice derivative supplied by #367; (5) a variable-cell saddle's
  lattice rows supplied by #365; (6) the wall clock measured (cp2k 13 CP2K
  calls per step, <= 390 s a step, 4.1 MB per carrier) with `maxit: 2` kept by
  ruling. The `driver` refusals are by name: `optimize_cell` under a non-`main`
  driver, and a `driver: skill` optimize on a skill with no internal optimizer
  (`xtb`/`qdm`/`nn_pes` tables; `driver: main` keeps the old warning).
- **The relative-base double-apply (#375/#397).** `service_paths` anchors the
  calculation directory once for all five artifacts and the server applies a
  request `cwd` only when it is absolute (`server.py::_session_cwd`);
  `test_persistent_launcher.py` pins the launched `(cwd, script)` pair.
- **`tools/verify.py` readers.** `_engine_cp2k` claims a directory only on its
  ` CP2K|` banner and takes the last `Total energy:` (#385, three mutation
  arms); `_engine_molcas` claims on the `Happy landing` marker and reads the
  last `Total SCF energy` or `RASSCF root number N` (#397/#399), with
  `state_scoped` counting excited-state legs (#387).
- **The pressure leg (#371-#391).** #378 found the carried `gfnff_topo`
  (+0.3508098 Hartree on a sane frame) and now drops it before each
  evaluation; #381 put `pV` into the trust-region objective (pinned with
  `p = 0` bit-identical either way); #389/#391 moved the example onto the
  81-atom box and replaced the direction clause with the two sign-free
  readings. `optimizer/base.py::_enthalpy` carries the bounded statement.
- **CP2K excited-state stress (#322/#374).** `stress_method: num` strains the
  cell at single points and returns a `(nstate, 3, 3)` GPa tensor tracked by
  excitation energy; `_cell_gap_conjugate` adds the gap's own derivative when
  both state pages exist (`optimize_cell_gap_cp2k` prints `Cell Gap Term:
  0.01705835 / 0.01733804`). The exact degeneracy is a design boundary, not an
  owed producer: roots swap there, and the individual surfaces are neither
  smooth nor local minima.
- **The nn_pes SOC bound and its seed (#353).** The bound is 20.0 cm-1 (under
  the 41.35 constant-mean floor by 2.1x, over the worst sampled draw by 1.5x);
  `gl_seed` reaches every training child, and `test_seed_repro.py` fails a
  training module with no `-s` or no `apply_seed`. The seed buys
  reproducibility, not cross-machine agreement: a remote reading outside
  6.37-13.30 is a divergence to investigate, not a reason to widen.
- **BAGEL's fixed-name files are cleared before each run (#355).** The run
  script clears `ENERGY.out`/`FORCE_*.out`/`NACME_*.out` on both arms, and
  `xmscaspt2_frequency` now carries the `casscf` + `smith_caspt2` prefix so the
  energy has one writer. Inside one run the file is still rewritten per
  displacement point, and the named warning says so.
- **The adaptive sampling loop (2026-10-24).** `workflow/adaptive.py` runs the
  round loop: unified directories, staged child runtimes, traceable screening,
  reference labelling, dataset snapshots, permutation expansion, PBC/multiscale
  passthrough and the resident MD uncertainty stop. Dynamic thresholds,
  refine, transfer learning and adaptive-level restart remain out of scope by
  decision (`dev/methods/ADAPTIVE_DEVELOPMENT.md`).
- **The TS search (#335).** `opt_method: ts` takes curvature from either
  producer named by `&hessian hess_driver`; both ran to convergence against
  PySCF (32 evaluations each, energies agreeing to 1e-8 Hartree) and a
  `num_freq` job reports exactly one imaginary frequency (-1248.4 cm-1). The
  step is a partitioned RFO with mode following and Bofill updates.
- **Pinned-fragment and partial searches (#337/#338).** A minima relaxation
  with a frozen fragment converges in 27 evaluations against 23 without the
  constraint; a partial saddle with one atom pinned converges in 75
  evaluations on the engine (`optimize_partial_ts.yaml`, water dimer, one
  imaginary at -3064.0 cm-1) and in 86 on the analytic `_Saddle3`. The
  400-evaluation plateau entry written on 2026-10-12 was the nine-minus-six
  frame, not the surface, and is superseded by #338.
- **A saddle search whose model has no negative direction (#335).** Partitioned
  RFO dissolves the branch: the followed mode's upper root sits above whatever
  curvature it has, so the step climbs without inventing a direction, and the
  reported count is the model's real number of downhill directions. The 0.1
  Bohr ceiling is measured (0.05 diverged to +2.2 Hartree).
- **Mode normalization (#324).** Both sampling branches convert through one
  `_mode_scales` (`sum(m x^2) = 1`), and equipartition is asserted on the
  sampled ensemble; the shared fixture's `sum(m x^2) = 13.019` had been
  inflating amplitude measurements.
- **The multi-center QMMM channels (#451-#466).** Local readings and
  per-center identities for `sp`/`sc_charge`/`md`; the remote round's real
  count was 152 pass / 4 not-pass, and both causes were fixed: the
  `_detect_not_converged` bare-substring green (#452) and the 2-rank persistent
  probe whose per-atom kspace split no longer equals group/group (the
  `qmmm_field` session is pinned to 1 rank). `sc_charge` re-measured to
  `E_QM -152.3057825700` on both machines (the earlier "cross-machine
  divergence" compared pre-fix local products with post-fix remote ones and is
  withdrawn). The four single-center multi-step controls all `completed`, so
  the residual is not multi-center-specific; multi-center `optimize_cell` was
  verified locally and remotely, and `multicenter_qmmm_npt` completed on both
  remote carriers; checkpoint/restart closed locally (byte-identical to a
  four-step baseline) and completed on both remote carriers, and #456 confirmed
  the nine lattice entries and the six cell spellings agree. Negative results
  kept:
  multi-center embedded composites do not judge Etot
  (`runner._embedding_breaks_conservation`, band 9.7e-3 against 1e-3), and a
  zero-damping `md`/`mep` probe mixes two electronic structures, so both run
  "one fit" (`sc_charge: true` + `maxiter: 1`).
- **The 2026-10-11 pause-point handover (#399).** The owed full-tree green has
  since run (last before this pass 4805 passed / 1 failed / 1 skipped / 3139
  subtests, the single failure an untracked fixture's own gap).
- **The #460 M8 follow-up.** A multi-center excited-state minimization now has
  a shipped example (`multicenter_qmmm_ex_opt.yaml`).

## 4. Debt on the documentation and product side

Open:

- **`term_outcome.json` has never been produced by a run (#399).**
  `tools/verify.py` counts `term_outcomes` and reads 0 in every round because
  the file is written only when the Calculator dispatches terms remotely. The
  per-term path is covered by criteria (the crossing legs pin the term
  columns), never by a run.
- **`'energy gradient soc' -> '_soc'` has the shape that just broke ORCA's NAC**
  (#446): a compound jobtype resolves to one profile whose step list may drop
  `gradient`. `tddft_nac` needed exactly that - its deck lost `EnGrad` and
  `irootlist`, ORCA printed no `CARTESIAN GRADIENT`, and the gap gradient came
  back exactly zero while the row passed. No shipped case asks for SOC with
  gradients; settle it with one engine reading.
- **The zero-fill family is wider than the two parsers this round walked
  into.** `Results.nac` still defaults to `[[[0.0, 0.0, 0.0]]]` in eight skill
  schemas; molcas and orca now report an absent coupling as no rows, and which
  other skill answers "not computed" with a fabricated row is un-checked (the
  same question applies to `soc`).
- **The per-state charge row is closed on both paths (#450), but the carrier
  that exercises the choice is missing**: round 23's ONIOM band is green with
  an openqp high level, which returns no `computed_charges`. An ONIOM
  embedding case whose high level is molcas/pyscf/xtb is what turns the choice
  into a reading.

Ruled or closed (one line each):

- **`NUM_HESS_DEVELOPMENT.md` is not empty** (#399): 251 lines / 14,510 bytes,
  `Last updated: 2026-10-12 23:25`. `SEARCH_DEVELOPMENT.md` documents
  `GridSearch` over a skill's declared searchable keywords, and
  `TRAIN_DEVELOPMENT.md` documents the nn_pes training chain (not the
  single-shot chain train/prediction has taken since #299).
- **`lammps_skill`'s `restart.data` is written but never read, by design**
  (ruled 2026-10-20): it serves the agent's own standalone-call orchestration,
  not a PyRAI3MD continuation entry.
- **The `expected:` block is deliberately unused until the pre-release pass**
  (ruled 2026-10-20): no comparator reads it today; it is populated together
  with its reader before release (`SKILL_TEST_DEVELOPMENT.md` section 4.3).
- **The two Slurm script builders share one dialect (closed 2026-10-20)**:
  both render through `kernel/sbatch.py`, and `test_slurm_dialect.py` pins
  parity, spellings, boolean handling, `extra` and the named refusal.
- **The per-step file-write cost is deliberately not accounted** (ruled
  2026-10-20): `debug: true` exists to produce functional evidence, so the
  saving from turning it off is ignored rather than measured.
- **A lone `opt_state_i` is range-checked** (2026-10-20): the check lives in
  `schema/validate.py::validate_centers` against the normalized `nstate`,
  `state_j` identically, multi-center runs skipped; a mutation arm turns two
  refusal cases red.

## 5. Long-term directions (the part that still holds)

Open:

- **A deployed ORCA SOC leg whose declaration is a strict subset of the ladder**
  (#448): every shipped SOC leg today is openqp, so the read-through path has
  no engine evidence.
- **Molcas multi-pair NAC print order: the mapping is one function on both
  sides, the engine reading is still owed** (#448). The deck writes `&ALASKA`
  per spin group through `input_builder.py::nac_deck_order`, the reader inverts
  it and refuses a block set that is not the requested pair set. Every Molcas
  log here requests one pair, so a real multi-pair run is round A of #448, and
  `md_molcas_fssh` gains a `nactype: nac` variant to make it a live path.
- **A frequency deck answering gradients for real has to bring an engine
  reading first** (#448, ruling D4): `energy frequency gradient` is refused on
  both sides, and nothing measured here supports that the engine can.
- **The scan's Slurm arm is verified to the artifact a submission leaves, not
  through a queue**: one `.rt` payload and one `slurm.sh` per point are read
  back; queue time, a declined allocation and a child dying on a compute node
  are only knowable in the test environment (method document section 10).
- **The scan ranks, it does not judge or adapt**: no threshold, no sequential
  or Bayesian point selection, no per-point data file.
- **Equivariant neural networks, and promoting `frequency.yaml` / `soc.yaml` /
  `nac.yaml` / `qm_mm.yaml` / `periodic.yaml` to official examples, are
  unchanged** since the pre-#262 judgement and are no longer tracked by Phase
  number.

Closed (one line each):

- **#21, the mixed-spin energy and gradient channel (#449)**: one predicate
  (`groups_by_manifold`) drives writer and reader; `md_molcas_fssh_twospin`
  runs two carriers and `casscf_gradient_twospin` three, with the two-pass QMMM
  deck carried by `multiscale_qmmm_embedding_nac` and
  `casscf_espf_nac_twospin`.
- **The ORCA skill no longer derives its pair maps itself (#448)**:
  `classify_declared_pairs` reads what the framework forwards, the cross
  product survives only for a case that declared nothing, the two engine limits
  are refusals, and `_pack_soc` has no `soc_pair_map` fallback.
- **The transition channels carry no phantom row (#448)**: eight schemas
  dropped the `[[0, i] for i in range(nstate)]` default, BAGEL's deck builder
  dropped the same fallback, four readers dropped the `npair = ... else 1`
  padding, and the framework refuses a declared self pair.
- **OpenQP's unconditional ladder sort is fixed (#448)**: it now checks
  instead of permuting when a declaration is present and refuses a
  `coupling`/map disagreement.
- **The state-resolved placeholder arrays follow the table they answer
  (#448)**: `osc_strength`/`transition_dipole` are sized by
  `len(transition_pair)`, and ground-state-only skills carry no row.
- **The ML trio landed on 2026-10-09**: `train` is one call with an honest
  verdict and a continuation path, `prediction` is a test-set evaluation with
  its own record and files, `search` is the hyperparameter grid
  (`SEARCH_DEVELOPMENT.md`, `TRAIN_DEVELOPMENT.md` sections 8.1-8.2).

---

## Operations notes (the five that keep biting)

- **Two overlapping `pyrai3md test` runs on the same machine** clean each
  other's `tmp_*`, which shows up as batches of
  `bash: <run>/lammps_skill/<task>.sh: No such file or directory` with no
  calculation failing at all. Confirmation: run it alone after the remote
  `pgrep -af "[p]yrai3md test"` reads empty.
- **`tmp_*` under the install directory is a globally collectable resource**:
  the next local `pyrai3md test`/deploy deletes the evidence directories that
  were downloaded back (#237/#238's evidence went that way). Downloads and
  probe products always go outside the install directory (for instance
  `evidence_<date>/` at the repository root), and `results.json` is archived
  per round.
- **A package-manager cleanup takes down both LAMMPS channels at once**:
  `/opt/lammps-mpi/bin/lmp` links the MPI runtime from `open-mpi`, and the
  `liblammps` shared library in `.venv` links FFTW; once either package is
  removed or upgraded to a new SONAME, both channels fail within the same day,
  and the framework side only says `return code 134` or `Library not loaded`.
  That is what happened at 2026-09-30 15:41 (the same yaml ran in the morning
  and failed in the afternoon). Recovery = reinstall `fftw` and `open-mpi`, or
  rebuild LAMMPS against the libraries that are there; to tell whether this is
  your cause, check first that the dependency paths the loader reports exist.
- **`bringup` without `--lammps-path` tests five cases less**: C2/C3/C5/C7/C10
  fail as `lmp executable not found in PATH` while the purely persistent
  C1/C6/C8/C9 all pass, which looks like a basically green run.
- **A transport failure reads as "the round made no progress".** Round 16
  (`tmp_20261018_173454`) had to be collected by hand: `tools/deploy.py` gave
  up at 19:37 with `Error: no status.json and no log progress for 2700s` while
  an independent read at 19:32 showed the job alive at row 105 with 0
  failures. The remote job is `nohup`-detached, so a poller that quits does not
  stop the round, and with `deploy.jump_host` already set a fresh round takes
  the jump. Not changed here: separating "cannot reach the host" from "no new
  completions" needs the poll loop to tell an ssh error apart from an empty
  answer.

## Path family (2026-10-21)

Open:

- **A multi-centre partition (C > 1) with `optimize.method: mep|irc|neb` is not
  refused by name.** The design reason is recorded (a path interpolates between
  structures and a multi-centre partition has no single structure to
  interpolate), and the ruling is to leave it here rather than write a guard:
  `data/core.py::_build_task_queue` has no path branch, so the combination is
  unverified rather than turned away deliberately.
- **Not covered by an engine example: an interpolation whose endpoints really
  straddle a face.** `geodesic.consistent_pair` brings the product into the
  reactant's image and is pinned in process
  (`TestThePeriodicEndpointFrame`: a 6 A cell, 5.90 A step without the alignment
  and 0.4 A with it), and section F of `tools/verify.py` checks every shipped
  band's frames for a lattice-vector jump (10 legs, 0 on round 10). No shipped
  fixture straddles, so the step is read at unit level only.

Closed (one line each):

- **The ten new examples have engine readings**: all thirty-one ran in round
  `tmp_20261022_084519` (62 rows, 62 pass), including `path_cell_mep_relu`
  (8 points, 1000.00 -> 1011.94 A^3), `path_neb_pbc_geodesic` (5 images, 0.019
  Ha barrier) and both condition cases at 0 deviation.
- **A single-centre composite with a variable cell is shipped and read**:
  `path_cell_mep_multiscale` steps the box on the outer layer's stress
  (`state.active_stress`); round `tmp_20261022_151713` both carriers identical,
  3 points, -77.3231 -> -77.6418 Hartree, volume 1000.000 -> 1000.155 A^3.
- **The plan's last two rounds are closed**: round `tmp_20261022_151713` read
  291 rows / 288 pass / 0 fail / 213 closed / 64 path rows, `verify.py` clean
  (D 12 legs / 614 rows; F 10 band legs with no jump).
- **A periodic walk seeds in the frame its evaluations come back in**:
  `pathbase.py::_continuous_atoms` builds the seed from
  `continuous_coordinates`; the fix exposed that `path_mep_pbc`'s
  `path_radius: 0.1` was tuned against the one-box-off seed and the example is
  now 0.05 (four points, -0.2816 -> -0.3864 Hartree).
- **#406's two fixes are certified**: round `tmp_20261022_103032` read
  `path_irc`'s two saddle rows both -91.5616686633, `path_irc_pbc` walking one
  saddle plus four descending points (-0.368299 -> -0.681074), and
  `irc_origin_broken 0`.
- **The post-round reading of the path family is closed**: section D re-reads a
  leg's cell verdict against its own `Lattice=` frames and section F reads
  frozen atoms over walkable frames; the D cross-check found and fixed a
  variable-cell band writing the parent state's box into every frame
  (`Path._write_band_frames` now takes `BandPathOptimizer.lattice_of`). Shape:
  500 rows / 213 closed by switches / 290 printed.
