# PyRAI3MD Adaptive Sampling Design

Last updated: 2026-11-09 (section 16 is now the merged record of the
analysis/exploration update: the former `adatpvie_patch.md` plan and its
implementation evidence were folded in, the default-closed contract, md
threshold decoupling, trace/PES/coverage/saturation, predictor alternatives,
UDD/gap and the output contract are documented there, and the patch file was
removed. The residual V1 limits are listed in 16.10.)

Before it, 2026-11-06 17:35 (the asset sections no longer name the retired
working directory: sections 12 and 13 now point at the shipped
`pyrai3md/examples/` carriers and fixtures, and section 8 states the
product location without it.)

Before it, 2026-11-06 12:10 (section 14's auxiliary sentence was
rewritten against the code after a measured rerun: the `aux-labels.json`
sidecar now appears only when a frame produced one of
`field`/`mm_energy`/`q_qm`/`mid_charges`/`low_charges`, and the
mechanical-embedding ONIOM carrier was rerun locally to confirm it
writes no file. Section 15 gained the six `examples/adaptive_*.yaml`
qualification carriers and the `examples.adaptive_test` family switch.
Section 4 now names the two submission routes the remote round actually
took: `reference_remotetask` flips the reference skill to its own slurm
submission, while MD is submitted as a framework runtime. Sections 1-13
were not re-audited.)

---

> **Capability boundary.** Adaptive sampling reuses the task queue the
> planning layer produces. PBC and multiscale/composite queues pass through
> when the input asks for a composite; an adaptive run's theory/reference
> pair is a pair of workflow roles, not an inferred QMMM composite
> (`resolve_composite_type` returns None for `jobtype: adaptive`).

## 1. Iteration Semantics

Iteration 1 is the external start:

```text
external data + external network
    -> MD trajectories
    -> screening statistics
    -> iter-1/selected_iter-1.xyz (for iteration 2)
    no reference, no training
```

Iteration k >= 2:

```text
reference on iter-(k-1)'s selected structures
    -> labels appended to the in-memory dataset
    -> reference products in iter-<k>/reference/
save iter-<k>/data_iter-<k>_<nframe>.json (unexpanded)
train iter-<k>/NN-<title>-iter-<k> from that file
training-data quality gate (only when &adaptive maxmae > 0)
    -> either the fit stands, or suspect new frames are removed one at a
       time and the network is retrained; a removal that does not lower
       the validation energy MAE is undone and the run stops
run iter-<k>/md/ with that network
screen the retained history
    -> iter-<k>/selected_iter-<k>.xyz (for iteration k+1)
```

The terminal iteration still labels the previous selection, trains, and runs MD;
its own selection has no consumer and is not referenced.

`maxiter=1` runs iteration 1 only. `maxiter=2` adds one reference+train+MD iteration.
`maxiter=3` adds two, matching the user's description ("2nd and 3rd iterations
each issue reference, train, and run MD").

## 2. Screening and Traceability

Every trajectory outcome is read from its retained history. A local outcome
is a `PyRAIState`; a remote outcome is a `Result` whose `history` came back
from `result.json`. The window is `&adaptive record_step` (default 10; 0 is
normalized to 10); the MD child history bound is `max(record_step, 4)`.

The deviation is the legacy rule (option A): `np.std` across the two members
with `ddof=1` (for two members that is `|a-b|/sqrt(2)`), then `amax` over
states for energy and over states/atoms/components for gradient. The
`sample_*` plan does not change that formula.

Per frame:

| Condition | Trigger | Candidate pool |
|---|---|---|
| any `err_* > max*` | yes | - |
| any `err_* > min*` | - | yes |
| the md run ended with stop code 1 (ill population) and `check_population=1` | yes (the stopping frame) | yes (the stopping frame) |
| `check_energy_order=1` and a same-spin level is lower than an earlier one by more than `gap_tolerance` | yes | yes |

Population is not judged per frame: `&md md_maxpop` owns that judgment and ends
the trajectory at the offending step, and the screen only decides whether that
stopping structure is collected. A frame whose stored density looks bad but
whose run did not stop is not a candidate for that reason.

The candidate order is the legacy order: frame index from newest to oldest.
The distance filter (`distance < 0.7 * bond_lib(a1, a2)`, i<j) and the
per-trajectory `maxsample` cap follow in that order. Every row carries
`traj_id`, `frame_index`, `step`, `errors`, `flags`, `selected` and `reason`;
`status_iter-<k>.json` keeps every row.

## 3. Property Plan

`&adaptive` carries:

```text
sample_energy               default 1
sample_grad                 default 1
sample_nac                  default 0
sample_soc                  default 0
sample_dipole               default 0
sample_transition_dipole    default 0
sample_oscillator_strength  default 0
```

`reconcile_adaptive_properties` (`schema/normalize.py`) runs before planning:

- adaptive on, md off -> switch the md keyword on (`nactype=nac`, `dosoc=1`,
  `dodipole=1`, `radiative=1`);
- adaptive off, md on -> warn and switch the adaptive keyword on;
- both off -> skip the property.

Energy and gradient are the ML baseline and must stay on. A channel the
label extractor does not implement yet (dipole/transition/oscillator) is
refused by `_preflight` if enabled. NAC/SOC are accepted by the current
label extractor only when the initial dataset and the reference capability
plan carry the channel.

## 4. Execution Wiring

- The theory queue is resolved by the ordinary composite rules: a layered
  theory list is ONIOM2/3 (or QMMM when `multiscale_type=qmmm`), and the
  ML task is its electronic provider. The reference queue is planned
  independently from `control.reference`; `molecule.highref/midref`
  define its own partition. `resolve_composite_type` no longer returns
  None merely because the jobtype is adaptive.
- `_ADVANCED_DEFAULTS['md'/'adaptive']` is `energy+gradient`. The NAC vector
  is added only by `_derive_from_dynamics` for `sfhp=fssh, nactype=nac`;
  ktdc and dcm do not request it.
- `data/core._spell_advanced_jobtype` spells the skill jobtype from the
  distributed capabilities for md/surface_hopping/adaptive (and a
  main-driver optimize), instead of forwarding the raw `md`/`adaptive`.
  `mlp_eg` therefore resolves `mlp_eg_gradient` and a ktdc run needs no NAC
  profile. The NN skill's profile table is not changed.
- The reference task gets the same treatment: E+G capabilities spell
  `energy gradient`, and OpenQP resolves `mrsf_gradient` (not `mrsf_nac`,
  whose parser returns zero gradients).
- Stage flags: `md_*` and `reference_*` in `&adaptive`; `remote_train` for
  the training child. `reference_remotetask` does not submit the framework
  runtime. It flips each reference skill's `execution_mode` to slurm; the
  reference child itself stays local, the skill writes its own sbatch
  script, and the job body runs the skill agent with the serialized config
  on the compute node. The skill agent executes there in-process; it does
  not submit a second job. MD is different because it is a workflow
  runtime: `md_remotetask` serializes the child to `traj_*.rt` and the
  framework submits it, with the skill local inside.
- The parent sets `needs_backends=False`; every stage is a child runtime.
- A dispatched child runs with its allocated workdir as its cwd, while the
  LAMMPS topology (`*.data`, `*.in.init`, `*.in.settings`) and an optional
  `*.charge` were staged beside the example. `_stage_runtime_fixtures` copies
  those names into each child workdir before dispatch, because a skill
  resolves them by bare name from its launch directory. The slurm carriers
  failed on exactly this before the copy existed: the child's LAMMPS run
  aborted on the topology the parent had been given.

## 5. Warm-Up Initial Conditions

Before iteration 1, `AdaptiveSampling._prepare_initial_conditions` resolves the
starting points. For `initcond: wigner`/`boltzmann` it samples from
`<title>.freq.<format>` once and writes the same batch to `<title>.init`
(the legacy product) and `<title>.init.xyz` (the file `initcond: read`
consumes later), in the `read_init_xyz` layout (natom, comment, element +
coordinates + velocities). The batch is injected into every MD child; a
child never re-samples. `initcond: read`/`load` uses the user's
`<title>.init.xyz` directly.

The ethylene test's 100-frame `ethylene.init`/`.init.xyz` is generated by a
one-time script; the adaptive run reads the first `ninitcond` frames (6).

The parent state still carries one `&MOLECULE` geometry. That geometry is
not an adaptive sampling input: it supplies the atom list, charge/spin and
the planning surface, while every trajectory starts from the
`<title>.init.xyz` batch. This is why the run log's final
`&adaptive sampling summary` prints no coordinates, cell, electronic state,
kinetic energy or velocities: those would describe the parent's single input
geometry, not the many structures the run produced.

## 6. Labels, Dataset, Permutation

The reference outcome is read as a mapping. A composite reference
publishes `label_channels` -- the normalized source term (or the
composite total), plus any auxiliary QM/MM field and MM energy rows.
Requested dataset channels: energy (nstate,), gradient
(nstate,natom,3), nac (nnac,natom,3), soc (nsoc,). Units are Hartree /
Hartree-Bohr / cm-1. Whole-system label arrays are projected onto the
trainable atoms before the shape check. A frame whose counts, symbols
or required channels disagree with the dataset is dropped with a reason.

The accumulated dataset is the external data plus all previous reference
labels. `TrainingData.append` enforces one atom order, one channel layout,
and consistent optional frame fields. A periodic run requires cell/pbc in
the initial data and carries each selected frame's cell/pbc.

`&train permutation_map` is applied in the training child's memory; the
iteration's data file on disk is always unexpanded.

New labels are appended at the tail of the dataset. `nn_pes` takes its
validation frames from the front (`split_validation_training_index` with
the member offset), so frames produced by the current iteration cannot
enter validation. The 100 → 102-frame check is pinned by
`TestLabels.test_appended_frames_stay_out_of_validation`.

### 6.1 Training-Data Quality Gate

`&adaptive maxmae` (float, eV, default 0 = off) is a ceiling on the
validation energy MAE of the iteration's fit. The number is exactly the
energy column of `&nn validation mean absolute error` (mean over the
ensemble folders and their members, already in eV); the gate never falls
back to the gradient channel, because an eV/Å value is not an eV value.

With the gate armed and the fit above the ceiling:

1. The frames this round appended (`dataset[base_nframe:]`) are tested
   against the frames the dataset held before the round
   (`dataset[:base_nframe]`). One electronic state at a time:
   `scale = 1.4826 * MAD` of the reference energies, `z = |E - median| /
   scale`, and the frame's worst state decides. `z > 3.5` flags it. A
   degenerate state widens the scale to its standard deviation and, if
   even that is zero, flags any value that differs from the median.
   Energies are read in the dataset's Hartree labels; the z-score is
   scale-free. The test reads the QM labels, not a prediction: a
   reference run that failed writes an energy the surface never had.
2. Nothing testable (no energy labels, a non-finite value, fewer than
   three reference frames) or no flagged frame stops the run.
3. Every appended frame flagged also stops the run: there is no clean
   label from this round to train on.
4. Otherwise the flagged frame with the largest z is removed from a
   **candidate** dataset (`TrainingData.subset`, the accumulated set is
   not mutated), the candidate is saved under `iter-<k>/prune/`, and a
   fresh network is fitted in `iter-<k>/NN-<title>-iter-<k>-r<r>`.
   - Lower validation energy MAE: the removal is accepted, the candidate
     becomes the dataset, and the loop repeats on the remaining new
     frames. Reaching the ceiling ends it as passed.
   - Equal or higher MAE, or a failed retrain: the removal is rejected,
     the frame stays in the dataset (the candidate is never adopted) and
     the run stops.
5. An accepted prune is compared against a fixed reference block and a
   fixed validation split (the front frames), so before/after MAEs see
   the same validation set. The improvement test is strict `<`, a gate
   decision rather than a statistical test.

The iteration record keeps `new_frames` = labels the reference verified
and adds `kept_frames` / `pruned_frames` when the gate is armed, so
`dataset frames = initial frames + sum(kept_frames)`. The closing
`&adaptive sampling summary` prints `kept_frames` under `New` on an armed
round, because that is what entered the dataset. The full audit
(per-frame z, medians, scales, removed indices) is in
`status_iter-<k>.json` under `validation_gate`. Rejected model folders
stay on disk; `status_iter-<k>.json` and `<title>.adaptive.json` name the
serving one. With `maxmae = 0` nothing is evaluated, no gate section is
printed, and the iteration record is the same as before the gate.

Verification measured on 2026-11-12 (local `.venv`, the ethylene example
with `epo 1`):

- `maxmae: 1e-6` on the shipped 100-frame dataset: the fit's energy MAE
  was `0.004353 eV`, the five new frames had a largest robust z of 2.89,
  and the run stopped with `no outlier among the 5 new frame(s)` before
  MD. `iter-2/data_iter-2_105.json` stayed the unpruned snapshot.
- A carrier whose reference energy deviations were halved: four
  removals were accepted (`z` 5.78, 4.84, 4.74, 3.62; MAE `0.00128457 →
  0.00119449 → 0.00115129 → 0.00110652 → 0.00108069 eV`), then the run
  stopped with `no outlier among the 1 new frame(s)`. The serving
  dataset was `iter-2/prune/data_iter-2_101_r4.json`.
- `maxmae: 1.2e-3` on the same carrier: one removal was accepted, MAE
  `0.00119449 eV` passed, and MD ran with
  `NN-adaptive_ethylene-iter-2-r1`; `Stop reason: maxiter exhausted`.
- `maxmae: 0`: no gate block, no `kept_frames`/`pruned_frames`/
  `validation_gate` fields, `Stop reason: maxiter exhausted`.

The shipped carrier `adaptive_quality_gate_local` packages the passing
prune path: `egs_ethylene_gate_100.json` is the shipped 100-frame set
with every reference energy deviation halved, `epo 30`, `maxmae 0.0082`.
Measured: validation energy MAE `0.00862553 → 0.00777373 eV` after
removing frame 100 (robust z 5.78), gate passed, and iteration 2 ran MD
on `NN-adaptive_ethylene-iter-2-r1`; two runs gave identical values. A
second removal measured `0.00784605 eV`, which the gate rejects.

The retrain arm under `remote_train` / slurm was not exercised.

## 7. NN Threads

`nn_pes_skill` has three thread keywords:

```text
thread          fallback / whole-skill spelling
train_thread    training subprocess budget (0/empty -> thread)
predict_thread  serving TensorFlow budget (0/empty -> thread)
```

Training starts one subprocess per ensemble member; `child_environment`
sets `OMP_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, and
`TF_NUM_INTEROP_THREADS` to `train_thread` for each. Serving sets the same
variables and `tf.config.threading` to `predict_thread` before the first
op. The ethylene test uses 2/2: 2 members x 2 = 4 training threads, 3 MD
children x 2 = 6 serving threads.

### 7.1 NN Diagnostics and I/O Switches

Two skill keywords control the optional diagnostics:

```text
write_step_log  0/1, default 0  step log + uncertainty dump during serving
predict_std     0/1, default 0  post-fit spread on the training frames
```

- `write_step_log: 0` skips both `{project}_{profile}.log` and
  `{project}_{profile}.uncertainty.txt` in the serving path. The ensemble
  spread itself is still computed: it is the adaptive sampling signal
  (`err_*`), not a diagnostic. The ethylene test sets it to 1.
- `predict_std: 1` makes a training run serve every training frame with
  every member and report the largest member-to-member spread as
  `&nn prediction max std`. The columns are `ndata`, `energy`, `gradient`,
  `nac`, `soc`; units are Hartree, Hartree/Bohr, Hartree/Bohr, cm-1, while
  the validation MAE table stays in eV, eV/Angstrom, eV/Angstrom, cm-1.
  The serving path forces the switch off; only `_train`/`_predict` profiles
  enter the analysis. The ethylene adaptive input sets it to 1 for the
  per-iteration fit report.

The `&nn training` header reports the couplings that were actually fitted:
`Number of NAC/SOC` counts a folder only when `train_errors` carries that
role. An E+G surface therefore prints 0 even when the interface declared
coupling pairs.

### 7.2 MD Phase Profile

`&MD perf_profile` (default 0) owns `pyrai3md_step.perf`. When off, no
phase timer runs and no file is written. When on, the phases are accumulated
over the trajectory and the table is written once at its end: it is an
overall cost summary, not a per-step trace. The ethylene adaptive input sets
`perf_profile 1` so the test keeps the performance table.

## 8. OpenQP Frequency File

OpenQP writes `<calldir>/<project>.freq.molden` on the Hessian compute path,
but the skill's parser reads only `<project>.hess.json` and the default
cleanup deletes the molden. The ethylene test kept it with
`&OPENQP keep_tmp 1`, and a one-off working-tree script copied it to
`<title>.freq.molden` and inserted `[N_FREQ]`/`[NATOM]`, which the framework
reader needs. That carrier and script were not kept; the framework's
`hess_molden` route writes the same file with both headers natively and is
the current way to obtain it.

## 9. Directory Layout

```text
<title>.adaptive/
  <title>.adaptive.json
  <title>.log                  # the run log; one &adaptive sampling block per iteration
  iter-1/                     # external network/data; MD + statistics only
    md/traj_<idx>/             # includes the trajectory's <title>.log
    selected_iter-1.xyz population_iter-1.json status_iter-1.json
  iter-<k>/                   # k >= 2
    data_iter-<k>_<nframe>.json
    NN-<title>-iter-<k>/
    NN-<title>-iter-<k>.log   # the fit's run log, named after its folder
    NN-<title>-iter-<k>-r<r>/ # quality-gate retrain r, when one ran
    prune/data_iter-<k>_<nframe>_r<r>.json   # its candidate dataset
    md/traj_<idx>/
    reference/ref_<idx>/ new-samples.json
    selected_iter-<k>.xyz population_iter-<k>.json status_iter-<k>.json
```

The data file name carries the iteration number and the accumulated frame count,
so `data_iter-2_106.json` is iteration 2's 106-frame set. `data_iter-<k>_*` is
always the unpruned snapshot of that round; a gate retrain reads its candidate
from `prune/`, whose name adds the retry index. `data_iter-<k>_*` is what the
dataset audit reads, so `new_frames` and the unpruned snapshot stay one
statement; `kept_frames` in `status_iter-<k>.json` ties the serving dataset to
it. Every stage is a framework
task in its own directory: MD leaves the same log and `md.*` files an
independent run does, and the fit leaves the same log an independent training
run does, named after the model folder it produced. `keep_md=0` removes the
iteration's MD directory after its statistics; the model, dataset, reference
products and logs stay. `population_iter-<k>.json` keeps one population
list per trajectory (lengths differ; nothing is averaged). The shipped
inputs and assets live under `pyrai3md/examples/`; run products land in the
launch directory under `<title>.adaptive/`, never in the package tree.

## 9.1 Formatted Iteration Record

`<title>.adaptive/<title>.log` is the run's main-framework output: the
project directory holds the input and user assets, while the adaptive work
directory holds that log beside `iter-*`. The adaptive report is a
`&adaptive sampling` section in it, one per iteration, in the same grammar as
`&optimization progress` and the MD step blocks. The block follows the run
order:

1. For `N >= 2`, an `&data` section with `Received data: X` (structures
   handed over by the previous iteration) and `Verified data: Y` (successful
   reference labels), then the iteration's fit in the same `&nn training`
   grammar as an independent training run: atom/state/coupling counts, arm
   and wall clock, training data, model folder, member fit logs, validation
   table and `&nn prediction max std`. Iteration 1 has no handed-over data
   and no fit, so this part is absent.
2. `&nn validation gate`, when `maxmae > 0`: the ceiling, the fit's energy
   MAE and whether it passed, the new-frame counts, one `round` line per
   retrain (removed frame, its robust z, the retrained folder, the new MAE,
   whether the removal stayed) and the verdict. A run the gate stopped
   prints this block and a minimal `&adaptive sampling` block with the stop
   status, and no screening.
3. `&adaptive sampling` and `Iteration: N`, after one blank line from the
   NN result above.
4. One line per trajectory in the legacy layout:

```text
Traj(ID: ... Step: ... State: ...) => Geom(Sampled: ... Collect: ...
Discard: ... (Bad ...) Refine: ...) => MaxStd(Energy: ... Gradient: ...)
```

5. The sampling counts (trajectories, average length, ground state, passed,
   completed, sampled trajectories/geometries, collected, discarded,
   refinement), the `MaxStd`/`Threshold`/`Pass` table for the channels the
   run labels, and the sampling verdict (`PASS` or `NOT PASSED`).

`Discarded` reports the total candidates that were not selected and, in
parentheses, how many of them failed the physical distance filter (`Bad`);
the rest were beyond the per-trajectory `maxsample` cap. The channel list
comes from `sample_*`: an off channel has no row even though its threshold
keyword carries a schema default. The selected count is not repeated in the
sampling verdict. Dataset and model paths appear only in the fit's
`&nn training` part. The closing `&adaptive sampling summary` prints `Elapsed
iteration`, `Final convergence`, the stop reason, and one trend row per
iteration: `Iter`, `New` (reference-passed collected frames added to the
dataset), `Ground %` (trajectories ending in the ground state) and
`Length %` (average trajectory length divided by the configured `&MD step`).
Fields that repeat the per-iteration record (`External`, `Ntraj`,
`Triggered`, `Selected`, `Completed`) are not repeated in the summary.

## 10. Failure Semantics

| Case | Behaviour |
|---|---|
| dynamic-threshold/refine/transfer nonzero, or `load != 1` | refuse at startup |
| no unique trainable theory task, no reference task, unreadable/incompatible initial data | preflight refusal |
| iter-1 external model not servable | preflight refusal |
| one MD child fails | no frame from it; other trajectories continue |
| every MD child fails | stop, mark failed |
| reference labels empty in an iteration >= 2 | stop, mark failed; no training on the old data |
| some reference children fail | drop those frames with a reason |
| training record missing or model not servable | stop, mark failed; never run MD on a half-product |
| `maxmae > 0` and the fit's validation energy MAE is above the ceiling with every new frame flagged | stop, mark failed; no network is fitted on labels from a round that is entirely suspect |
| `maxmae > 0` and the new frames cannot be tested (no energy labels, a non-finite value, fewer than three reference frames) | stop, mark failed, reason names the untestable input |
| `maxmae > 0` and no new frame is an outlier | stop, mark failed; the target cannot be met by pruning |
| removing one flagged frame and retraining does not lower the energy MAE | the frame stays in the dataset (the candidate is never adopted), stop, mark failed |
| a gate retrain fails or its model is not servable | the candidate is not adopted, stop, mark failed; the rejected folder stays for audit |
| `maxmae = 0` | the gate is not evaluated; every behaviour above is unchanged |
| no selection | stop; no next-iteration reference |
| `maxiter` spent | stop; the terminal iteration's own selection is not referenced |

## 11. Non-Goals

- dynamic threshold update (`dynsample`, `dyn*`, `ini*`, `fwd*`, `bck*`,
  `maxdiscard`), the refine channel, transfer learning, and adaptive-level
  restart remain out of scope.
- `load=0` is refused: the initial network is trained by a separate
  `jobtype: train` run.
- The optional dipole/transition/molecular-oscillator label extractors are
  switches only in this iteration; enabling them is refused by `_preflight`.

## 12. Ethylene Test

Assets under `pyrai3md/examples/`: `adaptive_ethylene.init.xyz` (the initial
frames), `nn_ethylene_init/` (the pretrained two-member `mlp_eg` ensemble),
`egs_ethylene_100.json` (the 100-frame S0/S1 E+G training set), and the YAML
carriers `adaptive_single_local.yaml` / `adaptive_single_slurm.yaml`. The
classic `.input` carriers and the one-off prepare scripts were
working-tree only and are no longer kept; `hess_molden` now writes the
frequency file the preparation chain used to shim. The classic inputs carry
no `profile`.
Thresholds: energy 0.05/0.045, gradient 0.10/0.09; `maxiter=3`,
`maxsample=1`, `keep_md=1`; MD 6 trajectories / pool 3 / 200 steps /
0.5 fs / S1 / FSSH ktdc / RK4; reference pool 2 with OpenQP thread 2;
NN 3x100 and 4x80, batch 5, 100 epochs with 20/50/30 LR steps,
train_thread 2, predict_thread 2; the test input sets `write_step_log 1`,
`predict_std 1` and `perf_profile 1`.

Verification: parse and plan the four `.input` files (the adaptive plan must
show one theory NN task with `mlp_eg_gradient` and one reference OpenQP task
with `mrsf_gradient`), run the prepare chain, then the adaptive run, and
check the iteration directory/data flow, traceable `status_iter-<k>.json` rows, init first
six frames, reference gradient non-zero, and the CPU settings. Unit coverage:
`qualification/verification/test_adaptive_sampling.py`, the adaptive schema
tests, the ktdc capability tests, and the NN launcher thread tests.

The 2026-10-25 layout/logging check used a second title, `ethylene2` (same
initial data and network, `ethylene2_adaptive.input` and
`ethylene2.init.xyz`), so the earlier `ethylene.adaptive` products stayed
untouched. Three iterations took 1m59s and produced
`iter-2/data_iter-2_106.json`/`iter-3/data_iter-3_112.json`, `NN-ethylene2-iter-2/3`,
`reference/ref_<idx>/` in each iteration, `NN-ethylene2-iter-2.log`/`-3.log` at
the iteration root with the training data, model folder and two member fit logs,
six MD child logs per iteration, and three `&adaptive sampling` sections plus the
final summary in `ethylene2.adaptive/ethylene2.log`. Each iteration >= 2 is
preceded by `&data` and `&nn training`; the fit reports `Number of NAC: 0`
for the E+G surface and `&nn prediction max std` gives energy 0.036244 Ha
and gradient 0.041576 Ha/Bohr (nac/soc `--`). `population_iter-<k>.json`
keeps six per-trajectory lists whose lengths differ (iter 1:
31/31/31/28/23/34; iter 2: 49/39/40/34/39/47; iter 3:
52/50/60/55/51/58). `perf_profile 1` wrote one end-of-trajectory
`pyrai3md_step.perf` per MD child. All six trajectories still stopped on
`err_energy`/`err_grad` (completed 0) and each iteration selected six
geometries, the same reading as the first end-to-end run.
## 13. Ethylene-in-Water ONIOM2 Test

Assets under `pyrai3md/examples/`: `c2h4_water25.data/.in.init/.in.settings`
(the 1581-atom periodic box), `adaptive_c2h4water.init.xyz` (the 100 full-box
Wigner frames), and the YAML carriers `adaptive_qmmm_local.yaml` /
`adaptive_oniom2_local.yaml` (with their `_slurm` twins). The one-frame
`energy gradient` composite carrier, the single-trajectory smoke carrier and
the insertion/embedding check scripts were working-tree only; the shipped
`.init.xyz` is the insertion script's output.

The input declares `theory nn lammps`, `reference openqp`,
`highlevel 1576-1581`, `pbc 1 1 1` and the box cell. Planning produces
ONIOM2 `[L(real) + H(model) - L(model)]`: LAMMPS owns the periodic box,
the NN high layer serves the six ethylene atoms, and OpenQP labels the
same six atoms for S0/S1 E+G. The shipped 100-frame asset was produced once
by a working-tree insertion script that aligned the Wigner ethylene frames to
the box template with Kabsch, rotated their velocities, kept the water rows
fixed, and rejected frames that leave the box or come within 0.9 Angstrom of
a solvent atom (extra Wigner samples were drawn when the supplied 100 could
not all be placed).

The PBC gate follows the method combination, not a universal tail. A
layered theory queue's tail (LAMMPS) owns the box. The reference tail
owns it only when the reference plan reaches the whole system: a single
reference layer beside a single theory method covers every atom, and a
layered reference always owns the remaining low region. A single
reference layer beside a layered theory queue labels only the
neural-network region and may be molecular (OpenQP on the six-atom
slice). The reference list is independent of the theory list; it is not
padded or truncated, and `none/x/no/off` are not placeholders -- omit a
layer instead.

`Calculator.close()` runs at the end of each MD trajectory: it releases
the resident LAMMPS service and the skill calculation directories before
a pool worker can be reused for the next trajectory. Without it, the
second trajectory in a worker reused the first one's execution namespace
and was refused by the resident-service claim. Since 2026-10-31 that
release is the general rule rather than the MD loop's own:
`workflow/__init__.py::execute()` closes the Calculator it built in a
`finally`, so
every dispatched unit ends with its backends released, and the claim is
keyed on the socket path instead of the namespace, because an isolated
batch drops the `_runtime_{idx}` suffix on purpose and a parent
optimization loop holds a service of that namespace while its numerical
Hessian displacements run (the same refusal, met again in
`multiscale_qmmm_ts`). The LAMMPS socket
relocation directory also carries a short calcdir digest, because
isolated adaptive runtimes intentionally drop the runtime suffix and
their deep paths all relocate out of the calculation directory.

The full run (`c2h4water_adaptive.input`, 3 iterations, 6 trajectories,
200 steps) completed in 5m25s. Iteration 1 selected six geometries;
iteration 2 reference returned five of six labels (one OpenQP SCF did not
converge and was dropped), so `data_iter-2_105.json` holds 105 frames;
iteration 3 returned all six and `data_iter-3_111.json` holds 111.
`selected_iter-*.xyz` stays 1581 atoms, `new-samples.json` and the
training data are 6 atoms, and `&nn training` reports `Number of atoms:
6` with `Number of NAC: 0`. The composite energy decomposition is exact:
`E_total = E_LAMMPS(1581) + E_NN(6) - E_LAMMPS(6)` (state 1:
`-8.2320524184 - 77.4626214721 - 0.0373170730 = -85.7319909635`
Hartree). The NN high task carries `natom=6`, `pbc=[0,0,0]`; the LAMMPS
low layers carry the 1581-atom box and the `atom_selection` of the six
ethylene ids. `perf_profile 1` writes one end-of-trajectory table per
child; `population_iter-<k>.json` keeps six per-trajectory population
lists with the lengths measured in each iteration. A one-frame `energy
gradient` composite run (`c2h4water_sp.input`, a working-tree carrier) and a
one-off check of its `&composite energy decomposition` reported a 1e-10 Ha
residual on the single point and 0 on the first adaptive trajectory frame.

## 14. Reference Partition and Label Correspondence

`control.reference` is an independent layered plan, not a positional
mirror of `control.theory`. Its length is the number of reference
layers: `reference[0]` is high, `reference[1]` of a three-layer plan is
mid, and the last entry is low. `molecule.highref/midref` define the
regions those layers cover and are parsed exactly like
`highlevel/midlevel` (1-based atom indices, flat or center-group form).
They never enter the theory centre plan. Empty `highref/midref` inherit
`highlevel/midlevel` when the theory queue is layered; with one theory
method the high region is the whole system unless a proper subset is
written (a layered reference needs one).

Planning derives the training-label source from atom-set equality, never
from list lengths. The neural-network surface is the theory electronic
provider's region: every atom for one theory method, `highlevel` for a
layered queue. If exactly one reference layer covers that same atom
set, the label is that layer's raw term. If no single layer does but the
reference composite spans the whole system, the label is the composite
total. If neither holds, planning refuses the input and names the
reference layers it saw. A three-layer reference whose mid task region
(`highref ∪ midref`, the ONIOM3 model region) matches the surface labels
from the mid term; its capability set is widened to the run's requested
channels so NAC/SOC do not silently disappear from a mid-surface fit.

Auxiliary data follows the scheme, not a skill name. A QMMM reference
publishes `field` (the V/F rows handed to the QM calculation),
`mm_energy` (`E_classic`) and `q_qm`; a QMMM reference with no MM
provider records a warning and carries no field. An ONIOM reference
with electrostatic embedding publishes `mid_charges` when a mid shell
exists and `low_charges` when the low shell computed charges; it warns
when neither layer exists. Mechanical embedding carries only the
matched layer's energy and gradient. Auxiliary data is exposed in the
reference result and in the run log, and the per-iteration sidecar
`iter-<k>/reference/aux-labels.json` is written only when a frame really
produced one of `field`/`mm_energy`/`q_qm`/`mid_charges`/`low_charges`:
`region_indices` names the region a label was taken from, which is not an
auxiliary channel, so a mechanical-embedding reference writes no file
rather than a metadata-only one. Nothing but `tools/verify.py` reads the
sidecar; neither the dataset nor the next iteration does.
`TrainingData` carries no field/MM channel in this build.

The framework offers data channels and lets the skill consume what it
declares. A skill that does not support the external potential ignores
the field instead of failing the term; a skill that requires data it did
not receive raises from its own boundary. Planning never refuses a run
because a skill currently lacks an optional channel, and no branch keys
on a skill name.

The correspondence is written once in the run log as
`&method alignment` (theory surface, reference type, label source,
source task ids, auxiliary channels, warnings) and in
`<title>.adaptive.json` under `method_alignment`.

## 15. Example Coverage

The adaptive verification suite carries three configuration-level
examples:

- `test_single_method_adaptive_plan`: one NN surface and one reference
  calculator on a single water, the ordinary adaptive shape;
- `test_oniom2_mechanical_adaptive_plan`: the twenty-water box with the
  NN high layer on one water and xTB low, reference OpenQP high + xTB
  low, `embedding=false`;
- `test_qmmm_mechanical_adaptive_plan`: the same box with the NN in the
  QM role and LAMMPS low, reference Molcas QM + LAMMPS classical,
  `multiscale_type=qmmm`, `embedding=false`.

The water box and its cell are read from the shipped
`examples/md_lammps_nve.yaml`, so the examples use the same 60-atom
twenty-water model as the LAMMPS qualification cases. The QMMM case has
no field probe; its label plan records `mm_energy`/`q_qm` and the
warning that no MM field is produced.
`test_qmmm_label_channels_project_onto_the_water` carries the same
full-system channels through `labels_from_outcome` and checks that only
the three trainable atoms reach the frame. Electrostatic QMMM and
charge-embedding ONIOM are not covered by these examples.

The six `examples/adaptive_*.yaml` (the same three systems, each in a
`_local` and a `_slurm` carrier) are the qualification side of this
coverage, and they are asked for by one key: `examples.adaptive_test`,
the family switch `manager.EXAMPLE_FAMILIES` puts above the six case
keys the way `multiscale_test` sits above the `multiscale_*` set. The
shipped file leaves it at `condition`, so the six keys decide one by one
(local on, slurm off); `true` opens all six for a round that checks the
probe -> reference -> train chain end to end, `false` closes them
whatever they say. `test_manager.py::TestAdaptiveFamilySwitch` pins both
halves: the six stems on disk and their membership in
`QUALIFIED_EXAMPLES`, and the three-valued decision. Section 16.10 lists the
eleven single-feature carriers added on 2026-11-08; with them the family
contains seventeen stems and the same three-valued switch opens each one.

## 16. Analysis Layer and Exploration Options (2026-11-08)

This section is the merged record of the analysis/exploration update. It is
default-closed: with every switch off the loop writes the same files and
values it did before. The single predicate `analysis_active` gates every new
analysis key, every analysis-specific `status_iter` field and every
`analysis_iter-<k>.json`. The options were implemented and verified on
2026-11-08; this section replaces the former `adatpvie_patch.md` plan. The
wider X Sampling draft's NN-side model analysis (learning capacity, data
sensitivity and transferability) was not part of this update; its draft
document was removed, so that proposal has no design document here.

### 16.1 Scope, invariants and work-package status

- **Ownership.** md supplies trajectory data; adaptive analyzes and decides.
  Adaptive-private options are injected through `RuntimeState` and the md
  constructor, never through a user-facing md keyword. A user-requested md
  run sees only `&md` thresholds.
- **Default closed.** Existing result keys and values, files and
  stop/screening behaviour are unchanged. `alternative_*` is additive and
  `None` at `predict_mode=0`; `RuntimeState.analysis_trace/analysis_bias`
  are omitted from `to_dict`/result.json/.rt when unset. One deliberate
  extension outside the analysis gate: `status_iter` now always records
  `population_status`, `order_status` and `exploration`, because those
  describe the existing sampling checks and are needed by the log even when
  no analysis switch is on.
- **Pure analysis.** The analyzers take rows and return rows/records; they do
  not write the state or files. Only the adaptive loop writes the round
  report.
- **Three-state semantics.** `ok`, `not_applicable` (the data exists but the
  method's precondition is not met; the record carries a countable reason and
  is skipped) and `missing` (the run asked for it but the trajectory did not
  produce it; every trajectory missing stops the round). Comparison-model
  data adds `not_requested`, which is informational and never fatal. The
  population status is not a data check any more: it reports `off`,
  `not_applicable` (nosh), `not_stopped` or `stopped`, which says whether the
  md guard ended the trajectory, not whether a frame's density was readable.
- **Skill supplies data, adaptive decides.** The serving skill does not
  write status or warn; the adaptive workflow synthesizes
  `not_requested/available/missing` when it consumes the result.
- **Gap mean invariant.** The CV bias is an antisymmetric `±V/2`; it is
  applied only above `gap_threshold`, and a negative gap stops gap
  exploration.
- **UDD follows the active state.** The target state is
  `pmd.system.current_state` at each step, so a hop switches it. Gap, NAC and
  hopping probabilities read the unbiased channels. The member count is not
  hard-limited.
- **Predictor selection.** `predict_mode=0` returns the member mean with
  `alternative_* = None`; `predict_mode=k>0` (1-based) returns member k and
  writes the other members to `alternative_<channel>` plus
  `alternative_model_ids`. `M<2` is refused. The comparison channels mirror
  every prediction channel, so a new dipole/transition/oscillator channel
  follows automatically.

| WP | Scope | Status |
|---|---|---|
| WP0 | shared channels, `RuntimeState` carriers, injection points, cross-validation | done |
| WP1 | md threshold decoupling and `md_max*` | done |
| WP2 | analysis container, switches, report call point | done |
| WP3 | trajectory trace channel | done |
| WP4 | QM-side PES: kappa, `r^E`, low-rank `r^F`, resolution curve | done; periodic datasets are `not_applicable` (16.10) |
| WP5 | coverage, progress, MFML saturation stop | done |
| WP6 | predictor comparison channel | done |
| WP7 | state-resolved bias hook and unbiased channels | done |
| WP8 | UDD exploration | done; enabling is by `udd_start_iteration`, not by an automatic `Δ_k` test (16.10) |
| WP9 | CV and paper gap exploration | done |
| WP10 | reference failure task ids | done |
| WP11 | acceptance matrix and experiments | done (16.10) |

### 16.2 md threshold decoupling

`AIMD.__init__` accepts `deviation_thresholds`, `trace` and `bias`.
`Dynamics` carries `md_maxenergy/md_maxgrad/md_maxnac/md_maxsoc`, all default
`None`. An empty threshold set makes `_stop_deviations` return `[]`, so a QM
md child (no `err_*`) and an unguarded NN md never stop on a deviation. A
user NN md reads `&md md_max*`; an adaptive md child stamps all four keys
from `&adaptive max*` during `_md_iteration`. A channel the adaptive run does
not label carries `None`, which leaves the child's configured value in place;
the channel cannot fire because its `err_*` is absent. The guard log no
longer mentions `&adaptive`. This changes the already-confirmed case "direct
NN md plus `&adaptive`": the adaptive block no longer supplies md thresholds.

The population bound is not stamped: `&md md_maxpop` (default 0.01) belongs to
the md run, and an adaptive child uses its own value like any other md run.
`&adaptive check_population` only decides whether the structure the child
stopped on (stop code 1) is collected; the parent does not re-judge any
population, and `&adaptive pop_tolerance` is retired (dev/methods/
MD_DEVELOPMENT.md section 30). A failed calculation (stop code 4) comes back
as the child entry's error marker, carrying that code, and is never collected
as a structure.

### 16.3 Analysis container, trace and missing data

`workflow/records/analysis.py` defines the versioned `AnalysisFrame` and the
pure analyzers. The adaptive loop writes `iter-<k>/analysis_iter-<k>.json` and
one `&adaptive analysis` log block per round. `status_iter-<k>.json` keeps
every screening row and, per trajectory, the population-stop and order
statuses and the exploration flag.

The trajectory trace rides `RuntimeState.analysis_trace = {'spec','rows'}`.
A local outcome exposes it through the returned state; a remote outcome
through `result.json`'s `runtime`, and one accessor accepts both. The md side
records only what the requested analyses need: gap (scope from
`analyze_gap`), maximum force, population, energy drift and the minimum
inter-atomic distance (the last is `O(natom^2)` and is skipped above 200
atoms). A short trajectory with no recorded rows is `not_applicable`; a
requested trace with no rows on every trajectory is `missing` and stops the
round.

The core dynamics checks follow the same absence rule: a propagating run
(`sfhp != nosh`) that returns no density reports population `missing`; a
frame with no energy or spin labels reports order `missing`; a per-trajectory
warning names the trajectory, and every trajectory missing stops the round.
`sfhp=nosh` reports population `not_applicable` rather than a false failure.

### 16.4 PES, coverage and saturation

PES analysis reads the existing QM labels, never an NN frame:

- A1 point-pair curvature `kappa = dR^T dg / |dR|^2` and gradient-change
  magnitude `K = |dg| / |dR|`.
- A2 energy-gradient residual `r^E = E_j - E_i - 0.5 (g_i+g_j)^T dR`.
- A3 low-rank local gradient reconstruction `r^F`, with the numerical rank
  from an SVD, leave-one-out validation, condition number and uncovered
  directions; a thin neighbourhood returns `not_applicable` with the point
  count, nearest distance and requirement.
- A4 the resolution curve: nearest-distance/`r^F` pairs, so the report gives
  the geometry spacing at which the retained-point reconstruction stays
  within tolerance.

A candidate without a QM label is never given a fabricated `r^E`/`r^F`; it
gets only the neighbourhood support estimate. Periodic datasets are
`not_applicable` in V1; the local pair formulas are limited to fixed-cell or
non-periodic systems.

Coverage uses a fixed descriptor over a bounded sample of the dataset
(at most 200 frames and 200 atoms) and reports the covered fraction, radius
and nearest-distance median. Saturation is the MFML statistic on the
validation MAE (a fixed prefix of the dataset): a moving average over
`analyze_window` rounds, a local improvement, and local/global tolerances.
Crossing the tolerance stops the run with reason `saturated`, distinct from
`converged` and `maxiter exhausted`; a zero tolerance disables the stop.

### 16.5 Predictor comparison channel

The nn_pes skill gains `nn_predict_mode` (default 0). Its calculator exposes
the member outputs and, in mode k, returns member k as the primary prediction
while mirroring every other prediction channel to `alternative_<channel>` and
recording `alternative_model_ids`, `predict_model_id` and `n_models`. Mode 0
returns the mean and leaves the alternatives `None`; no status key is written.
`M<2` with `k>0` is refused in three places: skill configuration validation,
adaptive preflight, and the skill backend after the container is loaded with
the actual member count. Units conversion covers the alternative channels.
Composite aggregation drops primary-inherited alternatives only when more
than one task contributes; a single-task result keeps them.

The adaptive `predict_mode` stamps the serving config and consumes the
alternative channels. A serving skill that cannot expose members leaves the
channels absent; adaptive then warns and skips the comparison instead of
stopping the run.

### 16.6 UDD and gap exploration

Both are independent adaptive switches and both mark the trajectory
`exploration`; exploration trajectories are recorded but excluded from the
production statistics and from the population judgment.

**UDD.** `explore_udd` requires `predict_mode>0` because the covariance force
is reconstructed from the primary prediction plus the alternatives. The bias
is `E_bias(sigma^2) = A[exp(-sigma^2/(N_M N_A B^2)) - 1]`; `force` changes the
nuclear force only, `potential` changes the active state's energy and force.
The target follows `current_state`, so a hop switches it; `udd_start_iteration`
delays the bias, and `bias_work`, `udd_force_ratio` and
`udd_state_switches` are reported. `udd_amplitude`/`udd_width` default to 0
(auto). Multi-center runs (`C>1`) are refused in preflight.

**Gap CV.** `explore_gap` with `gap_method=cv` applies
`delta E_i = -V/2`, `delta E_j = +V/2` and the corresponding gradient split;
the pair mean is unchanged, the correction is exactly zero at or below
`gap_threshold`, and a negative gap stops the exploration. `gap_pair` selects
one pair; without it the scope comes from the state ladder.

**Gap paper.** `gap_method=paper` replaces the active propagation gradient
with the upper-lower gradient difference, rescales velocities when the excess
is payable, and requires `sfhp=nosh`. It cannot be combined with `explore_udd`
(preflight refuses it). Its next-round initial conditions come from the last
pre-trigger frame of the previous round, half on the lower and half on the
upper surface; when no such frame exists the configured batch is used with a
warning. CV and UDD may be combined and are accounted separately.

The bias hook is `workflow/records/analysis_bias.finalize`, called before
the constraint finalize. It keeps `unbiased_energy`, `unbiased_gradient` and
`unbiased_nac` for FSSH/GSH/LZSH, screening and analysis; only the integrator's
force/potential channels are modified. The total energy carries a potential
bias, while `bias_energy`/`gap_bias_energy`/`bias_work` are reported
separately to avoid double counting. Preflight refuses either exploration
switch when `&system_constraint` declares a metadynamics bias: both would add
a propagation potential, and the run never combines them.

### 16.7 Reference failures and exploration accounting

A reference failure row now carries `ref_index` and `source_task_ids`, and
the warning names the task id. One failed task drops that frame; every
reference task failing stops the round with `reference data collection
failed`. The iteration summary counts new labels only from verified frames.

Exploration trajectories are flagged in the screening record. `Ground %` is
computed over production trajectories only and prints `--` when there is no
production trajectory; the iteration block prints the exploration count
separately.

### 16.8 Output contract

The `&adaptive analysis` block prints `version`, the switch line
(`gap/trace/pes/coverage`, `predict=<mode>`, `explore=udd|gap`), one line per
trajectory for the checks, gap, trace, PES and predictor sections, the bias
line (UDD/gap energies, work, force ratio, state switches, and the paper
method's `gap_force_switched`/`gap_velocity_scale`), and the summary
(coverage, PES, saturation with both tolerances, cumulative labels and rounds
since improvement). The sampling block prints a `Data checks` line even when
no analysis block exists; a switched-off check is omitted, and mixed
per-trajectory statuses collapse to `value(n)`. `status_iter` carries the
same two statuses per trajectory.

### 16.9 Switch reference

```text
analyze_pes        0/1                       QM-side local PES (kappa, r^E, coverage)
analyze_gap        off|s0s1|adjacent|all      gap statistics
analyze_trace      0/1                       scalar trajectory trace (via RuntimeState)
analyze_coverage   0/1                       coverage + MFML saturation progress
analyze_window     3                         moving-average window (rounds)
analyze_local_tolerance / analyze_global_tolerance   eV; 0 disables the saturation stop

maxmae             0.1 eV                    validation energy MAE ceiling; 0 disables
                                              the training-data quality gate (section 6.1)

predict_mode       0|k                       adaptive override of nn_predict_mode
explore_udd        0/1                       UDD bias, following current_state
udd_mode           force|potential
udd_amplitude / udd_width   0 = auto
udd_start_iteration 1                        first round the UDD bias is applied
explore_gap        0/1                       gap-driven exploration
gap_method         cv|paper                  cv default; paper needs sfhp=nosh
gap_pair           i j                       selected state pair
gap_threshold      0.03 Ha                   below this no driving is applied
gap_height / gap_width      0 = auto
```

### 16.10 Example coverage, verification and V1 limits

The `adaptive_test` family now opens 18 carriers: the six original
single/ONIOM2/QMMM local+slurm examples plus one carrier per feature:

| carrier | feature |
|---|---|
| `adaptive_analysis_pes_local` | `analyze_pes` (curvature, r^E, low-rank r^F, resolution) |
| `adaptive_analysis_gap_local` | `analyze_gap` with `s0s1` |
| `adaptive_analysis_trace_local` | `analyze_trace` (force, drift, min distance; gap follows `analyze_gap`, population needs a propagating run) |
| `adaptive_analysis_coverage_local` | `analyze_coverage` + saturation stop |
| `adaptive_predict_primary_local` | `predict_mode=1` + `alternative_*` |
| `adaptive_udd_force_local` | UDD `force` mode |
| `adaptive_udd_potential_local` | UDD `potential` mode |
| `adaptive_gap_cv_local` | gap `cv` method |
| `adaptive_gap_paper_local` | gap `paper` method + per-round initial conditions |
| `adaptive_md_max_local` | `&adaptive max*` overrides `md_max*` |
| `adaptive_quality_gate_local` | `maxmae` + a narrowed reference fixture: one frame pruned, refit into `NN-<title>-iter-2-r1`, gate passes, MD continues |
| `adaptive_checks_nosh_local` | population/order checks on a `nosh` run |

Each carrier runs through both the YAML and the generated classic-input mode.
The example checker reads `maxiter`/`ninitcond` from the carrier rather than
pinning 2/5, and checks the iteration count and trajectory count; for a
carrier with at least two iterations it also checks the second-round
reference labels, data snapshot and trained model.

Verification measured on 2026-11-08:

- Local `qualification/verification`: 3571 tests OK, 1 skipped.
- Remote adaptive-only round `tmp_20261108_211603`: 34/34 example tasks pass.
- Remote adaptive-only round `tmp_20261108_222245`: overall PASS, 34/34
  example tasks pass. Its logs show
  `Data checks: population=not_applicable, order=ok` and
  `gap_velocity_scale=0.821716 gap_force_switched=True`. The `switches` line's
  `predict/explore` fields were added after that round and are covered by the
  local runs and unit tests.
- Coverage carrier, round 4: `saturation MAE 0.004421 ... SATURATED` and
  `Stop reason: saturated`.
- UDD potential, 50 steps: `max_energy_drift 3.62e-4 Ha`.
- Mean vs single-member A/B: same QM budget (2 candidates, 2 new frames),
  final validation MAE energy 0.00539801 vs 0.00539769 eV and gradient
  0.02872263 vs 0.02872156 eV/Angstrom.
- `md_max` carrier stops at step 1 on the adaptive threshold instead of
  running to step 20.

Known V1 limits (not defects in the implemented path, but not done):

- Periodic PES neighbourhoods return `not_applicable`; the local pair
  formulas are fixed-cell/non-periodic only.
- UDD enabling is the user's `udd_start_iteration`; an automatic decision
  from the validation-error trend (`Delta_k`) is not implemented.
- The UDD report prints the force ratio, work and switch count; the
  per-step active state is in the analysis rows, but a standalone state
  sequence and member count are not printed.
- FSSH hopping combined with UDD force was not measured (the example model
  has no NAC); long-trajectory UDD/gap behaviour and tolerance calibration
  remain to be studied on the target systems.

