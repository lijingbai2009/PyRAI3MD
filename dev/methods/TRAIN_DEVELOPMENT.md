# PyRAI3MD Neural Network Potential Training Design

Last updated: 2026-10-24 07:40 (the `MLSingleShot` body now lives in `workflow/single.py`; the node identity, settlement and `&train` routing are unchanged. Earlier 2026-10-23 22:15 (section 4 now states the two training-data input modes, the new optional transition-dipole channels, and the framework-side permutation expansion before backend initialization; the `trainer_data.py` line references were re-derived. Sections 1-3 and 5-11 were not re-audited.

Before it, 2026-10-22 22:20 (section 11.1 no longer claims a fit has no verdict at all: `settle_ml_outcome` judges whether the record came back, and what is absent is a numeric tolerance on the reported MAE. The rest was not re-audited.

Before it, 2026-10-15 12:10 (section 5.3/6.1 re-checked against
the code: the seed element `[-s, seed]` is new, and the `fit.py`,
`train_runner.py` and `schema/config.py` line numbers were
re-derived; the rest of the file was not re-audited line by line)

---

## 1. Scope of this document

This document describes the **training method** the package implements: how a
label set becomes a servable ensemble of neural network potentials, what the
contract of that label set is, how the fit is executed, and what has to be true
before the folder it writes may be read back by a dynamics run.

Three other documents own adjacent ground, and this one deliberately does not
duplicate them:

| Document | Owns |
| --- | --- |
| `dev/skills/NN_PES_SKILL_DEVELOPMENT.md` | the `nn_pes_skill` package as a skill: directory layout, the TensorFlow import boundary, the serving path, the results contract |
| `pyrai3md/skills/nn_pes_skill/skill_guide.md` | the user-facing contract: profiles, configuration keywords, units |
| `dev/SKILL_TEST_DEVELOPMENT.md` | how a skill's test cases are planned and staged by the qualification suite |

Everything below is stated against the code as it stands, with the file and line
of each decision so a reader can check it. Paths are relative to the package
root `pyrai3md/`.

---

## 2. How a training run is requested

A training run is not a workflow of its own. It is one call of the skill backend
under a profile whose name ends in `_train`.

| Step | Where |
| --- | --- |
| profile resolution from method + jobtype | `skills/nn_pes_skill/schema/method.py:207` (`resolve_profile`), token map at `:155` (`'train' -> '_train'`, `'soc train' -> '_soc_train'`) |
| step plan declared per profile | `schema/method.py:144` (`PROFILE_STEP_MAP`: `['train', 'results']` for a `_train` profile, `['load', 'properties', 'results']` otherwise) |
| framework execution | `workflow/train.py::Train` (node `train`), whose body is `workflow/single.py:MLSingleShot.run`: one calculator call, the standard step bookkeeping, then `workflow/records/ml.py:settle_ml_outcome` -- completed when the record its jobtype promises came back, failed naming the missing or empty channel when it did not. `&train` is the section that steers that call's dispatch (see `dev/WORKFLOW_DEVELOPMENT.md` 7.6) |
| backend dispatch | `skills/nn_pes_skill/backend.py:158` (a `_train` profile creates the model folder instead of loading one), `:362` (the same `calculate()` entry routes to the trainer and returns the training record) |
| service entry | `skills/nn_pes_skill/service/trainer.py:376` (`run`) |

Two consequences of that shape are worth stating plainly:

- **A training run evaluates no geometry.** It returns `train_*` keys and empty
  `fin_coord` / `fin_velocity` (`backend.py:363-374`), and its log section is the
  training report rather than a property block (`backend.py:243`, `agent.py:96`).
- **A training run is a serial term.** `backend.py:185` declines a thread pool
  for a `_train` profile only, because the training arm sets the TensorFlow
  device visibility for the whole process. Serving profiles stay thread-safe;
  the claim is pinned by `skills/nn_pes_skill/tests/test_train_predict.py`.

Validation refuses a training request that cannot be satisfied
(`schema/validate.py:133-155`): missing `train_data`, missing `modeldir`, a
`train_mode` outside `('training', 'retraining')`, a `gpu` index below `-1`, or a
role section the container would reject. The role sections are read
**the way the trainer will read them** (`validate.py:148`, calling
`schema/validate.py:34 check_hyper_section`), so an unknown keyword or a
per-member group that does not cover the ensemble is reported before any folder
is created.

---

## 3. What gets trained

### 3.1 Roles, folders and model families

A profile names a set of **ensembles**. Each ensemble is one folder on disk, one
model family, and one role that decides what its output channels mean
(`schema/method.py:41-43`):

| Role constant | Folder name | Section that carries its hyperparameters |
| --- | --- | --- |
| `ROLE_ENERGY` | `energy_grad` | `eg` |
| `ROLE_NAC` | `nac` | `nac` |
| `ROLE_SOC` | `soc` | `soc` |

`ROLE_SECTIONS` (`method.py:103`) is spelled out because the two names differ for
the energy family: the section is named after the **network family**, the folder
after the **ensemble the trained members land in**.

`PROFILE_ENSEMBLES` (`method.py:55`) enumerates every declared profile as a
tuple of `(role, model_type)` pairs. `targets_for()` (`trainer.py:57`) is the
trainer's only view of it -- the profile decides the work list, nothing else
does.

### 3.2 Output width is declared, never derived

`CHANNEL_COUNT` (`method.py:113`) maps `(model_type, role)` to the name of the
config field that sizes the output axis:

| Model family | Width comes from |
| --- | --- |
| `mlp_e` | `nstate` for the energy role, `nsoc` for the spin-orbit role |
| `mlp_eg`, `mlp_g2` | `nstate` |
| `mlp_nac` | `nstate` (one channel per state -- see below) |
| `mlp_nac2` | `nnac` (an arbitrary number of declared pairs) |

`channel_count()` (`trainer.py:73`) reads that field off the config. The
`soc` folder reuses the energy family with the derivative head switched off
(`trainer.py:149-151`, `base['model']['energy_only'] = True`), because a spin-orbit
matrix element is one scalar per declared pair and not a derivative of anything.

`check_channels()` (`trainer_data.py:216`) then refuses the two ways this can be
wrong before a fit starts: the label file must carry exactly the declared number
of channels, and a pair set that `mlp_nac` cannot represent is rejected with a
message that names `mlp_nac2` as the way out (`trainer_data.py:262-268`).

### 3.3 Ensemble members

`mult_nn` (`schema/config.py:62`, default 2) is the ensemble size, and it is the
axis every hyperparameter keyword is written against. It is also what makes a
trained folder a **model**, not a network: the serving path reports the
member-to-member spread as an uncertainty (`dev/skills/NN_PES_SKILL_DEVELOPMENT.md`
section 4 and 7), which is the reason the fit is one subprocess per member
(section 6 below) rather than one process that fits in series.

---

## 4. The label contract

### 4.1 The file

One JSON file, named by `train_data` (`config.py:42`), or the same
mapping handed inline as `train_dataset`; both arrive at one reader,
`trainer_data.py:146 from_payload` (`load` at `:307` is the file wrapper).
The keys are the ones the original project used:

| Key | Shape | Required |
| --- | --- | --- |
| `xyz` | `(nframe, natom, 3)`, rows of `[symbol, x, y, z]` or `[x, y, z]` | yes |
| `symbols` | `(nframe, natom)` | optional; needed when `xyz` rows carry no symbol |
| `energy` | `(nframe, nstate)` | yes for the energy role |
| `grad` | `(nframe, nstate, natom, 3)`; a single-state file may omit the state axis | yes for `mlp_eg` / `mlp_g2` |
| `nac` | `(nframe, nnac, natom, 3)`; a single-pair file may omit the channel axis | yes for the coupling roles |
| `soc` | `(nframe, nsoc)`, a flat row is read as one pair | yes for `soc` |
| `transition_dipole` | `(nframe, npair, 3)` | optional; carried, not fitted here |
| `transition_dipole_norm` | `(nframe, npair)` | optional; derived from the vector when absent |
| `osc_strength` | `(nframe, npair)` | optional; carried, not fitted here |
| `natom`, `nstate`, `nnac`, `nsoc`, `npair` | counts | optional, derived from the arrays when absent |
| `unit` | channel to unit spelling | optional, checked, never trusted |

Element symbols are not needed for the fit: the networks are built from index
lists derived from the atom count (`feature_indices`, `trainer_data.py:503`), and
the element names stay on the caller's system section. They are still read so a
permutation map can be checked for element preservation (section 4.4).

Shape disagreements are refused at load time, not corrected: a declared `natom`
that differs from the geometries, fewer than one state, or an `energy` array whose
second axis differs from the declared state count each raise before any folder
exists (`trainer_data.py:146-305`).

The transition-dipole channels are carried through `match_labels`
(`trainer_data.py:385`) unchanged: no current training profile consumes them, and
the vector is not a derivative, so no length factor applies.

### 4.2 Units: one convention, and a check instead of a switch

Labels are read as **Hartree energies and Hartree/Bohr derivatives**; spin-orbit
values are read as **cm-1** (`schema/reference.py:58-60`). The network is fed
Angstrom, so the labels are put on that axis once, in
`trainer_data.py:385 match_labels`:

```
dE/dR_angstrom = dE/dR_bohr * (1 / BOHR_TO_ANGSTROM)     # trainer_data.py:426-427
```

`BOHR_TO_ANGSTROM = 0.52917721090299996` (`schema/reference.py:27`). Energy and
gradient therefore describe one surface, which is the only rule a training file
has to satisfy. Spin-orbit labels get no factor: a value that is not a derivative
has no length unit to convert.

**There is deliberately no unit keyword.** A switch would let a caller declare a
scheme that does not match their own file, and the failure is silent: the network
learns a surface whose reported gradient is the derivative of its energy by the
wrong factor, and only an energy-conservation run hundreds of steps later notices
(`trainer_data.py:385-430`). Instead, `check_units()` (`trainer_data.py:332`)
inspects the file's optional `unit` block and refuses any declared unit that is
not one of the accepted spellings -- so a self-describing file that disagrees is
caught at the one place a mismatch can be caught cheaply.

### 4.3 What each ensemble is fitted on

`targets_for_role()` (`trainer_data.py:523`) is the single mapping from role plus
model family to the label array: `mlp_e` gets `energy`, `mlp_g2` gets `grad`,
the coupling families get `nac`, the spin-orbit folder gets `soc` (and raises if
the file carries none, `trainer_data.py:549-554`), and `mlp_eg` gets the pair
`[energy, grad]`.

The arrays are then written as the pickles the training subprocess reads
(`trainer_data.py:564 write_to_folder`, which calls the container's own
`model_save_data_to_folder` at `service/nn_pes/datasets/general.py:202`, writing
`data_x` and `data_y` -- `general.py:73-75`). Nothing here touches a prediction.

### 4.4 Two input modes and permutation expansion

A training request names exactly one source (`schema/validate.py:142-150`):

- `train_data`: a file name, read by `load` (`trainer_data.py:307`);
- `train_dataset`: the same mapping inline, read by `from_payload`
  (`trainer_data.py:146`).

Both pass the same shape, count and unit checks. The inline mode exists so the
framework can expand data and hand the result to the backend without writing an
intermediate file.

`&train permutation_map` (`schema/config.py:Train.permutation_map`) names a
legacy permutation-map file: whitespace-separated, **1-based** atom orders, one
row per additional permutation. `workflow/train.py::prepare_training_data` runs
from the `Workflow.prepare` hook (`workflow/__init__.py::execute`) **before**
`Calculator.initialize`, so the skill backend is constructed from frames and
never sees a map. `Train` and `AdaptiveSampling` call it directly, and
`SingleShot` calls it for a skill-driven `jobtype: train`; for each `_train`
task it:

1. reads the source (file or existing inline mapping),
2. reads and validates the map (`pyrai3md/training_data/permutation.py`):
   every row must be a permutation of the atom list and, when symbols are
   present, must preserve the element order,
3. expands in source-frame-major order (original, map 1, map 2, ...), writing
   `group_id` and `permutation_id`,
4. writes the result to `skill_config['train_dataset']` and clears
   `train_data`.

Only the atom-indexed channels move: `xyz`, `grad` and `nac` are re-indexed on
the atom axis; `energy`, `soc`, `transition_dipole`, `transition_dipole_norm`,
`osc_strength` and the per-frame auxiliary arrays are copied. An atom relabelling
is not a spatial rotation, so the transition-dipole vector is not rotated, and no
NAC sign is flipped: the phase is settled by the training loss. The operation is
idempotent (an inline dataset that already carries `permutation_id` is left
alone), so a grid-search child or a repeated prepare cannot expand twice.

The fallback API for adaptive sampling is `pyrai3md/training_data/`:
`TrainingData.load/save/append` and `expand_permutations`. `append` refuses a
different atom order, channel count, unit block or field set rather than
padding it.

---

## 5. The hyperparameter vocabulary

`schema/hyper.py:46 HYPER_KEYWORDS` is the whole vocabulary: keyword name to
`(kind, target path inside the container's hyper dict)`. The names are those of
the original project, so an old input file translates without renaming.

### 5.1 The member axis is always the outer one

A keyword may be written once for the ensemble or once per member. In the classic
text the member axis is the `;` separator; in YAML and JSON it is the outer list
(`member_groups`, `hyper.py:117`). Inside a group the value keeps its own shape,
which is why a keyword whose value is itself a list must be nested one level
deeper to differ between members. `kinds` are `scalar`, `list`, `pairs`, `bool`,
`text`, `reg` -- the `pairs` kind is what lets the classic text
`1 0, 2 0, 2 1` and the typed form `[[1, 0], [2, 0], [2, 1]]` mean the same index
table (`_as_groups`, `hyper.py:186`).

`broadcast()` (`hyper.py:253`) then fits the groups to `mult_nn`: one group is
repeated for every member, exactly-as-many is taken as written, more-than-members
keeps the leading ones (because `mult_nn` decides how many members are read), and
**fewer groups than members is an error**, not a silent partial fit.

### 5.2 Assembly order of one member's hyper dict

`build_member_hyper()` (`hyper.py:319`) produces one dictionary per member:

1. start from the container's defaults for the model type
   (`get_default_hyperparameters_by_modeltype`, imported inside
   `trainer.py:134` so that TensorFlow is not pulled in by the schema layer);
2. write the fields the data and the declaration decide -- `model_type`, `atoms`,
   `states`, `nnac` where the family has it, and the three feature index lists
   (`trainer.py:104-157`);
3. lay the caller's role section over that, per member;
4. assemble the two keywords that have no one-to-one spelling: the activation
   (name + alpha become the container's `{'class_name', 'config'}` form,
   `hyper.py:353-358`, defaults `leaky_softplus` / `0.03` at `:106-107`) and the
   three regularizer switches (`reg_l1` / `reg_l2` are shared by them, so they are
   resolved first -- `_regularizer`, `hyper.py:386`);
5. mirror what the user wrote into the `retraining` section as well
   (`_mirror_retraining`, `hyper.py:409`), because the container reads
   `hyperall[mode]` and `mode` is `train_mode` (`training_mlp_eg.py:107`). A
   keyword whose slot the family's retraining section does not carry is skipped
   rather than invented.

### 5.3 `train_mode`

`training` and `retraining` (`config.py:59`, default `training`) select which
section of the member hyper dict the fit subprocess reads. They do not select a
different code path in the framework: the mode is passed through unchanged as
`-m` (`train_runner.py:52`, `fit.py:82`), which is why the mirroring above
matters -- spelling `epo` once must not mean it applies to only one of the two
sections.

---

## 6. Execution: one subprocess per member, two arms

### 6.1 The work list

`build_jobs()` (`runner/train_runner.py:35`) expands the targets into one entry
per `(ensemble, member)`: for each `(folder, model_type)` in the profile, for each
`index` in `range(mult_nn)`, it asks `train_command()` for the command line. The
per-member device index comes from `gpu` (`config.py:67`, default `-1` meaning no
device) read with the same member-axis rule as a hyperparameter
(`member_list`, `hyper.py:293`).

`train_command()` (`service/nn_pes/nn_pes_src/fit.py:82`) is the single source of
the command line for both arms:

```
[sys.executable, '-m', <training module>, '-i', index, '-f', folder, '-g', device, '-m', mode, '-s', seed]
```

`-s` is the last element and the only one that can be absent. `build_jobs()` reads
`gl_seed` (`config.py:58`, default 1 - the framework's control field under its own
name, so `control.gl_seed` of a framework-driven run arrives without a translation
table) and sends `gl_seed + index`, the rule an MD run applies per trajectory
(`workflow/md.py:189`). A child process inherits no random state from its parent,
so without that flag every rerun of one input draws fresh weights; the flag is
what makes a trained value a measured constant instead of a sample. Applying it is
one call per child entry (`repro.apply_seed`, `nn_pes_src/repro.py:33`), one call
because a Keras fit consumes three streams (Python's `random`, NumPy, the framework
seed) and seeding one of them leaves the others drawing fresh numbers. A caller
that passes no seed gets the command line it got before, flag absent; and a network
family added later cannot skip the seed, because `tests/test_seed_repro.py` walks
`FIT_MODULES` and fails a training module that has no `-s` or parses it without
applying it.

The module name is built from **this module's own import name**
(`get_module_for_fit_script`, `fit.py:38`, `here = __name__.rsplit('.', 1)[0]`),
so no installation path is written into the source and the launcher works whether
the package sits inside a larger distribution or is used on its own. The child's
environment carries the absolute entries of `sys.path` through `PYTHONPATH`
(`child_environment`, `fit.py:59`) rather than a guessed project root.

### 6.2 The two arms

`run_training()` (`train_runner.py:102`) starts everything and then waits for
everything in one loop. `start()` (`train_runner.py:75`) differs only in the
launcher in front of the same command line:

| Arm | Selected by | What it does |
| --- | --- | --- |
| `local` | `environment.execution_mode != 'slurm'` (`runner/__init__.py:46`) | `subprocess.Popen(command, env=child_environment(), cwd=folder)` |
| `slurm` | `execution_mode == 'slurm'` | writes `<folder>_<index>.sbatch` and submits it with `sbatch -W` |

The slurm arm writes one independent job per member
(`slurm_runner.write_job_script`, `runner/slurm_runner.py:64`), with `#SBATCH`
lines taken from `slurm_partition`, `slurm_time`, `slurm_memory`, `slurm_ntasks`
(`SBATCH_OPTIONS`, `slurm_runner.py:32`) and a job name of `nn_pes_train`. The
submission is blocking (`submit_blocking`, `slurm_runner.py:95`), which is the
convention the shipped skills use and what keeps a single wait loop in the
parent instead of a second scheduler poller. A missing `sbatch` is an error, not
a fallback to local (`slurm_runner.py:115-119`).

Serving, by contrast, is always in-process (`runner/local_runner.py:28 serve`):
a dynamics step costs about a millisecond of network evaluation, so submitting it
to a scheduler would add seconds per step. `execution_mode` therefore governs the
training run only.

### 6.3 Threads

`configure_threads()` (`runner/env_check.py:32`) publishes
`environment.thread` into `OMP_NUM_THREADS` with `setdefault`, and prints to
stderr when a value was already latched. The reason is the OpenMP latch, not a
preference: nn_pes is in-process TensorFlow and the runtime reads the variable at
its first parallel region, so the **first term fixes the count for the whole
process** and a later term asking for something else is told rather than silently
rewiring the others (`env_check.py:48-53`, CHANGELOG #306). The slurm arm must
say the count in the script, because the job runs in its own process
(`slurm_runner.py:85-87`).

### 6.4 TensorFlow is imported late

The training launcher reaches the vendored container, which pulls in TensorFlow.
It is therefore imported **by the trainer, not by the runner package's import
graph** (`runner/__init__.py:22-27` explains the boundary; `import_container`,
`runner/env_check.py:71`). That is what lets a machine without the optional
neural-network stack still import the backend, still be discovered by the
qualification planner, and still be told by name why it cannot run
(`env_check.py:56-69`, `:100-117`). `import_container` also refuses an interpreter
older than 3.11 before trying, because Keras 3 does not support it
(`env_check.py:85-89`).

---

## 7. Completion, readback, and what a number means

### 7.1 Order of operations

`trainer.run()` (`trainer.py:376`) follows one order and the reason is recorded at
the top of the module (`trainer.py:19-24`): write the hyperparameters and the
data, start one subprocess per member, wait for all of them, then reload and read
back the fit error. Reversed, a reload would resurrect the weights of a previous
run and a finished fit could not be told from a crashed one.

Any member with a non-zero return code fails the run
(`train_runner.py:143 failed_runs`, consumed at `trainer.py:401-408`) and the
message names the folder as a partial product that must not be used for serving,
with the per-member log path (`fitlog_<index>.txt`, written by the fit script at
`nn_pes_src/training/training_mlp_eg.py:291`).

### 7.2 The fit error is the completion certificate

`_read_fit_error()` (`service/nn_pes/nn_pes.py:382`) loads
`fiterr_valid_v<i>.npy` for every member and raises `FileNotFoundError` naming the
missing files if any are absent. That check is the last thing the training writes
(`training_mlp_eg.py:269`), so its absence means the folder is a half-product: the
parent writes freshly initialized weights *before* starting a fit
(`nn_pes.py:207 _save`), so a run whose children died still leaves loadable
`weights_v<i>.weights.h5` files that would predict noise. Returning an empty list
here used to make a failed fit look finished; `open_container()`
(`service/calculator.py:146`, which calls `_read_fit_error` at `:183`) now
refuses to serve such a folder.

### 7.3 Turning the stored error into a physical number

The stored value is in the **scaled units of that member's own normalizer**, where
a number means nothing: 0.02 is neither a milli-Hartree nor a milli-eV until the
spread the data was divided by is put back. `physical_errors()`
(`trainer.py:196`) does that, reading the spread from the scaler the folder was
fitted with (`container._models_scaler[folder]`, `trainer.py:240`) rather than
re-deriving a guess, and prints in the units the container declares for its own
plots (`schema/reference.py:66-68`):

| Channel | Multiplier | Printed unit |
| --- | --- | --- |
| energy | `HARTREE_TO_EV` (27.211386245988, `reference.py:31`) | eV |
| gradient, nac | `HARTREE_TO_EV` | eV/angstrom |
| soc | `1.0` (labels are cm-1 and stay cm-1) | cm-1 |

The per-channel table is keyed by `(role, model_type)` (`trainer.py:225-239`)
because `mlp_e` and `mlp_g2` report one channel each while `mlp_eg` reports two
from the same stored row -- hence the `position` index into the array the container
saved.

### 7.4 Reporting

Two renderings of one run, deliberately in the same grammar:

- `<project>_train.summary.txt` in the calculation directory
  (`trainer.py:268 summarize`, path built at `trainer.py:422-424`): wall clock, one
  line per member with its exit code and log path, then the validation MAE per
  channel with its unit.
- the skill's own run log, `<{project}_{profile}.log` (`backend.py:231`), appended
  with `format_training()` (`trainer.py:306`) -- a titled `&nn training` section
  between rules, a counts header (atoms, states, NAC pairs, SOC pairs, arm, wall
  clock, summary path), then a `&nn validation mean absolute error` table with one
  row per ensemble member and one column per channel. A channel a folder does not
  produce prints `--`, not `0.0`, because "never trained" and "trained to a
  perfect zero" are different statements.

`format_training` takes the column order
`('energy', 'gradient', 'nac', 'soc')` (`trainer.py:53`), the same spelling the
framework's run log uses, so the two renderings of one record read alike.

The standalone cluster agent writes the same section into the log it leaves next
to the serialized config (`agent.py:96-99`), and the results it writes there are
collected by the same naming convention the backend uses
(`runner/__init__.py:63 results_file_name`).

---

## 8. What is on disk after a run

Inside `modeldir` the trainer creates one folder per ensemble role
(`trainer.py:187-189`), and the container writes per member `v<i>`
(`nn_pes.py:207-223`):

| File | Written by | Read by |
| --- | --- | --- |
| `hyper_v<i>.json` | `nn_pes.py:220` | the fit subprocess (`training_mlp_eg.py:92`) and `load` |
| `scaler_v<i>.json` | `nn_pes.py:222`, `training_mlp_eg.py:249` | the fit subprocess (`:99`) and `load`; its spread is what `physical_errors` reads |
| `weights_v<i>.weights.h5` | `nn_pes.py:218`, `training_mlp_eg.py:242` | `load` |
| `data_x`, `data_y` | `datasets/general.py:73-92` | the fit subprocess (`training_mlp_eg.py:48-65`) |
| `fiterr_valid_v<i>.npy`, `fiterr_train_v<i>.npy` | `training_mlp_eg.py:269-270` | `_read_fit_error` (valid only), the report |
| `fitlog_<i>.txt` | `training_mlp_eg.py:291` | the failure message |
| `fit_stats/` | `training_mlp_eg.py:142` | diagnostics |

The vendored tree's plotting modules were **not** brought over
(`dev/skills/NN_PES_SKILL_DEVELOPMENT.md` section 1.1); the training scripts keep
their numeric output and their per-member log. Nothing in the vendored tree names
a project root -- every former `NNsMD.*` import is package-relative, guarded by
`tests/test_train_launcher.py::TestSingleSource` rather than by convention.

One container method is worth naming because it looks like the entry point and is
not: `NeuralNetPes.fit()` (`nn_pes.py:408`) is not called by this skill's training
path. The skill goes through `service/runner/train_runner.py` so that the arm
selection, the per-member log, and the return-code check live in one place the
qualification suite can drive. It takes `seed` all the same: `fit(..., seed=)`
forwards `seed + index` down the `_fit_models` -> `fit_model_by_modeltype` ->
`train_command` chain, so a run that reaches training through the container rather
than the launcher is not silently the unseeded one.

`keep_tmp` applies when the backend goes away, not per step
(`backend.py:414 _discard_calcdir`, attached through `weakref.finalize`, CHANGELOG
#307): the calculation directory holds the summary and the skill log, which are
products of the run rather than scratch. The model folder under `modeldir` is
never touched by that cleanup -- it is the point of the run. The prediction
products of section 8.2 sit one level further out, in the run directory, for
the same reason stated differently: a test-set evaluation exists to leave its
raw comparison behind, and `keep_tmp` governs TensorFlow scratch.

### 8.1 Continuation: the order that makes a restart real

`create` builds freshly initialized networks and `save` writes their weights
unconditionally (`nn_pes.py:155`, `:218`). The preparation order therefore
decides whether a restart is a restart: create, save, and only then ask the
fit subprocess to continue meant the trained surface was replaced by random
weights *before* the subprocess started, and its `load_weights` then read back
exactly what had just overwritten them while reporting, truthfully, that it
had loaded old weights. `targets_with_data` (`trainer.py:159`) now branches:

| Request | Folder state | Calls | Result |
| --- | --- | --- | --- |
| a new fit | empty | `create` -> `save` | trains |
| a new fit | complete | none | `FileExistsError`, naming the folder and the switch to write |
| a continuation | complete | `load` -> `update` -> `save` | continues from the weights on disk |
| a continuation | incomplete | none | `FileNotFoundError`, naming the absent product |

A continuation is read from the category the fit subprocess will consult:
`train_mode` names that category and `initialize_weights` inside it says
whether to load (`training_mlp_eg.py:177-183`). `initialize_weights` is a
role-section keyword, so it is written `eg: {initialize_weights: false}` and
applies per member (`hyper.py:319 build_member_hyper` mirrors what the caller
wrote into both categories, which is why spelling it once is enough). The
container's own defaults differ between the categories, and that difference is
the usual way to ask for a continuation: `train_mode: retraining` with nothing
else written is one.

The guardrail that must be *reported* rather than silently applied: the
container's `update()` merges every category but `model` (`nn_pes.py:178-205`),
so a continuation cannot change the architecture. `model_shape_changes`
(`trainer.py:215`) compares the requested `model` block against the folder's
own and refuses, naming each key with both values.

`prepared_or_raise` (`trainer.py:452`) checks `train_data` before preparing
anything, so a `FileNotFoundError` from the model folder is no longer reported
as a problem with the data file.

Measured, this round: a continuation over a trained folder printed
`Info: Load old weights at: <modeldir>/energy_grad/weights_v0.weights.h5` and
`Info: Transferring weights...` in `fitlog_0.txt`, and re-ran in 14.3 s; the
same input with `train_mode: training` was refused before touching the folder.
The call sequences above are pinned by
`skills/nn_pes_skill/tests/test_train_resume.py`, which uses a recording
container precisely because the fit subprocess's own log cannot distinguish
loading trained weights from loading the random ones that replaced them.

### 8.2 The test-set evaluation

`jobtype prediction` with `pred_data` set is not a training run and not a
property step: the trained folder is served over every frame of a file and
compared against the labels in it (`skills/nn_pes_skill/service/predictor.py`,
`dev/skills/NN_PES_SKILL_DEVELOPMENT.md` section 7). One unit rule matters
here and it is the rule of section 4.2 applied to a *served* value: the file
carries Hartree/Bohr derivatives, `match_labels` puts them on the Angstrom
axis, and the served gradient comes back Hartree/Bohr again, so the two are
placed on one axis before they are subtracted, and only then converted to the
printed eV/Angstrom. `tests/test_predict_products.py::TestUnitChain` pins
that chain against a surface the test knows analytically: labels are the
surface plus one Hartree, so the reported error must be exactly
`HARTREE_TO_EV`, and the gradient claim separates the per-Angstrom axis
from the per-Bohr one by three significant figures.

---

## 9. Training data in the qualification suite

The skill ships three small label files under
`skills/nn_pes_skill/tests/data/` (`egs_small.json`, `nac_small.json`,
`soc_small.json`) and three training cases that use them:

| Case | Profile | Section | Members |
| --- | --- | --- | --- |
| `tests/00_mlp_eg_train.yaml` | `mlp_eg_train` | `eg` | 2 |
| `tests/02_mlp_nac2_train.yaml` | `mlp_nac2_train` | `nac` | 2 |
| `tests/04_mlp_e_soc_train.yaml` | `mlp_e_soc_train` | `soc` | 2 |

Three decisions make those cases behave:

- **Numbering is load-bearing.** The cases of one skill share a run directory and
  the plan runs them in name order, so a train case is numbered before the serving
  case that reads the folder it writes. Nothing trained is shipped with the
  package.
- **The data file is named by name, not by path.** `stage_skill_fixtures()`
  (`qualification/certification/planner.py:152`, data directory lookup at
  `:179-181`) copies every file in the skill's `tests/data` beside the generated
  configs, so `train_data: ./egs_small.json` resolves in the run directory for
  every mode.
- **`nn_project` is spelled out rather than derived**, because the flat config the
  skill entry point reads carries no project; leaving it out lets the service
  modes and the standalone mode write their products into different directories.

The epochs and widths stay small on purpose (30/25 epochs, `nn_size` 24,
`depth` 2): the certification budget is one `_timeout` per task, and what the
suite is certifying is that the training chain runs through the framework's own
entry points, not that the surface converges.

---

## 10. Which gate pins which contract

| Contract | Gate |
| --- | --- |
| command line, module prefix, child `PYTHONPATH`, per-member device, job-script content, missing `sbatch`, return-code collection | `skills/nn_pes_skill/tests/test_train_launcher.py` |
| a trained folder is complete, the read-back hyperparameters differ per member exactly as written, metric units, summary file, training section in the skill log | `skills/nn_pes_skill/tests/test_train_predict.py` |
| declared channel counts, the `mlp_nac` one-channel-per-state limit | `tests/test_schema_validate.py` |
| the label unit rule and its refusal of a disagreeing `unit` block | `tests/test_units.py`, with `tests/test_schema_validate.py::test_the_unit_convention_is_not_a_keyword` |
| the member axis of a hyperparameter keyword (broadcast, surplus, partial coverage, the two carrier spellings of an index row) | `tests/test_hyper.py` |
| the skill is discovered and skipped by name when TensorFlow is absent | `tests/test_discovery_without_tensorflow.py` |
| packaging of `tests/data` and the manifest | `qualification/verification/test_skill_packaging.py` |

`test_train_predict.py` is the one that fits and serves in a single process, which
is the route a user actually takes (jobtype `train`, then a dynamics run), and it
prints its measured label reproduction error rather than asserting an invented
tolerance.

---

## 11. Known limits

1. **A fit has no numeric convergence verdict.** The framework does judge whether the
   training job produced its record: `workflow/single.py:MLSingleShot.run` settles every ML run
   through `workflow/records/ml.py:189 settle_ml_outcome`, which marks the run completed
   only when the record its jobtype promises came back and fails it naming the missing or
   empty channel. What is absent is a *numeric* tolerance - whether the reported
   validation MAE is small enough for a given dynamics run is a judgement the caller makes
   from the summary, and nothing reads it back against a threshold.
2. **Training is molecular.** `schema/validate.py:120-124` refuses any `pbc`,
   because the geometric features and the index weights were built and measured
   for molecules only. A periodic potential is not a flag away.
3. **`mlp_nac` cannot represent an arbitrary pair set** -- one coupling channel per
   electronic state (`trainer_data.py:260-267`); `mlp_nac2` is the family for
   declared pairs.
4. **One profile per process.** A training term reconfigures TensorFlow's device
   visibility for the whole process, so two terms that train cannot share a thread
   pool (`backend.py:185-211`).
5. **The thread count is process-scoped, not term-scoped** (section 6.3). A second
   term that asks for a different count is told.
6. **No plots.** The vendored plotting modules were not ported, so a fit leaves
   numbers and logs, not figures.
7. **Spin-orbit labels are not derivatives** and are therefore never rescaled by a
   length factor (section 4.2). They are the one channel whose metric unit is not
   an energy unit.
