# PyRAI3MD Hyperparameter Grid Scan Design

Last updated: 2026-10-22 22:20 (the three `schema/hyper.py` citations now carry their real prefix `skills/nn_pes_skill/`, since the framework schema has no hyper vocabulary. The rest was not re-audited; the module, keyword and gate names in sections 2-8 were re-checked and stand. Before it, 2026-10-10 12:13)

---

## 1. Scope of this document

This document describes the **hyperparameter grid scan**: how a skill's
declared tunable keywords become a set of training jobs, how one point of
the grid reaches the training path unchanged, and how the records those
trainings left are read back, ranked and written out.

| Document | Owns |
| --- | --- |
| `dev/methods/TRAIN_DEVELOPMENT.md` | one training run: the label contract, the member axis, the two arms, the completion certificate |
| `dev/skills/NN_PES_SKILL_DEVELOPMENT.md` | the `nn_pes_skill` package as a skill: its profiles, its results contract, its folder layout |
| `dev/WORKFLOW_DEVELOPMENT.md` | the workflow layer: registration, the execution graph, the single-shot path |

Paths below are relative to the package root `pyrai3md/`.

### 1.1 What this is not

Three names in this package sit close to a grid scan and mean something
else. They are kept apart on purpose, and the separation is asserted by
`qualification/verification/test_ml_verdict.py`.

| Name | What it does |
| --- | --- |
| `jobtype search` | varies the **training** of one model family over a declared grid (`workflow/search.py`) |
| `&scan` / `jobtype scan` | steps a **geometry** along a reaction coordinate and computes one energy per point (`workflow/scan.py`) |
| `mult_nn` and the `;` groups | the **ensemble member** axis inside one training request (`skills/nn_pes_skill/schema/hyper.py:17-41`) |

Structural/conformational candidate search was the original occupant of
`workflow/search.py`. It was a placeholder with no test, no example and no
consumer, and it is documented as such in the revision history of
`dev/TODO.md`. This round replaced it. Structure search is reserved for a
token of its own (`structure_search` is the suggested name), with its own
registry entry and its own input section; nothing in this document should
be read as its design.

---

## 2. Who declares what

The split is one sentence: **the skill says which keywords may be varied
and what a candidate means; the run says which values to try.**

| Question | Answered by | Where |
| --- | --- | --- |
| which keywords are tunable | the skill's manifest | `skills/nn_pes_skill/manifest.yaml` (`searchable:`) |
| what type a keyword takes | the same declaration | `type: int / float / bool / str / choice / list` |
| what shape a value must have | the skill's own vocabulary | `skills/nn_pes_skill/schema/hyper.py:46` (`HYPER_KEYWORDS`), checked at `check_searchable` |
| which section a keyword writes | derived from the trained profile | `schema/method.py:99` (`ROLE_SECTIONS` via `get_search_targets`) |
| which values to try | the run's input | `&search grid: {depth: [1, 2]}` |

The declaration is read through the skill's query module, so a scan can be
planned on a machine that cannot run the skill it plans for
(`skills/nn_pes_skill/query.py` states that discovery imports no
TensorFlow, and `tests/test_searchable.py` re-proves it by subprocess
probe).

### 2.1 Why there is no value generator in the framework

The candidate lists come from the input, not from a `start/stop/step`
expansion. Three reasons, each of which is a defect the alternative would
have created:

- A framework-side generator would have to know the type rules of a keyword
  it is designed not to know. The manifest boundary is stated in the file
  itself: *"which keywords live in a section, and what they mean, stays
  inside the skill"*.
- `hyper.check_value` (`skills/nn_pes_skill/schema/hyper.py:221-250`) already holds the bounds
  that cannot be recovered from (`depth >= 1`, `0 < val_split < 1`). A
  second copy of them would drift.
- A float step invents values nobody compares: `0.1, 0.2, 0.30000000000000004`.

`check_value` is reached through `query.check_searchable`, so the bounds are
enforced by their owner at the moment the grid is planned, before any
directory exists. Measured refusal (`test_searchable.py`):

| Input | Refusal |
| --- | --- |
| `depth: [2, 2.5]` | `'depth' expects an integer candidate, got 2.5` |
| `dropout: [[0.1, 0.2]]` | `'dropout' takes one value per grid point ([0.1, 0.2] is what a member group of it means)` |
| `loss_weights: 10` | `'loss_weights' is itself a list, so a candidate of it must be a list` |
| `nn_size: [0]` | `nn_size must be at least 1, got 0` |
| `use_reg_weight: 'l3'` | `'use_reg_weight' accepts none / l1 / l2 / l1_l2, got 'l3'` |

### 2.2 The member axis is not the grid axis

`skills/nn_pes_skill/schema/hyper.py` reads a list along the ensemble member axis: `epo:
'30;25'` means 30 epochs for member 0 and 25 for member 1. A scan that
handed over its candidates as lists would therefore vary the members, not
the points. The rule that keeps the two apart is implemented exactly once,
in `workflow/grid/plan.py:write_point`, and is pinned case by case in
`test_grid_plan.py`:

| Declared kind | Candidate | Written into `skill_config[section][keyword]` |
| --- | --- | --- |
| `scalar` / `bool` / `reg` / `text` | `60` | `60` (broadcast across members) |
| `list` | `[1, 10]` | `[[1, 10]]` (one group, broadcast) |
| `pairs` | `[[1, 0], [2, 0]]` | `[[[1, 0], [2, 0]]]` (one group) |
| any | a list of candidates | refused at the plan, as section 2.1 |

---

## 3. How a scan is requested

| Step | Where |
| --- | --- |
| jobtype token `search` | `schema/validate.py:36` (advanced jobtype), `planning/resolver.py:50` (its capability default only sizes the task queue; a scan computes no property) |
| profile the parent carries | `skills/nn_pes_skill/schema/method.py` (`'search' -> '_train'`) |
| workflow resolution | `workflow/__init__.py` (`JOBTYPE_MAP['search'] = GridSearch`) |
| graph node and keywords | `workflow/dispatch/planner.py` (`WORKFLOW_SECTIONS['search']`, `WORKFLOW_NODE_DECLARATIONS['search']`) |
| input section | `schema/config.py:class Search` |
| plan and stamp | `workflow/grid/plan.py` |
| fan-out | `Workflow.run` → `Dispatcher.dispatch(units=…, child_jobtype='train')` |
| one point executed | `workflow/search.py:_child_entry` → `workflow.execute` → `JOBTYPE_MAP['train']` → `Train` |
| read back and rank | `workflow/grid/record.py` |
| outcome | `workflow/search.py:postprocess` |

**The parent carries a training profile on purpose.** Resolving `search`
to a serving profile made the parent open a model folder that no point had
fitted yet, and the run died in `open_container` before planning anything.
That was measured, not reasoned: the first end-to-end attempt failed with
`modeldir does not exist: .../nn_parent` (see section 8).

### 3.1 The `&search` section

The classic carrier writes the whole grid on one keyword with `;` between the
entries (`grid depth 1 2; nn_size 8 16`); writing each keyword on its own
`grid` line reaches the same mapping, because the section reader keeps both
values of a repeated key as of this round -- it used to keep only the last,
and a scan planned from the surviving half was the wrong scan. The three
carriers are pinned equal by
`qualification/verification/test_ml_verdict.py::TestSearchGridCarriers`.

| Keyword | Meaning | Default |
| --- | --- | --- |
| `grid` | keyword to its candidate list; the scan is their Cartesian product | empty, and an empty grid is refused |
| `retrieve` | read the folders a previous scan left and rank them; submit nothing | `0` |
| `sort_by` | channel to rank on | the first channel the trained profile reports |
| `sort_stat` | `max` (worst member) or `mean` | `max` |
| `sort_order` | `asc` / `desc` | `asc` |
| `grid_root` | root of the scan's directories | `./grid-search` |
| `max_grid` | refuse a scan larger than this (0 states no limit) | `0` |
| `multitask` / `remotetask` / `multipool` | the dispatch axis, as every workflow node | class defaults |

`sort_stat` and `sort_order` are checked in `schema/validate.py`
(`validate_search_section`), because a misspelled statistic would otherwise
fall back to `max` and report a ranking nobody asked for. The grid itself is
refused by `plan_points`, which is the one place that can see the skill's
vocabulary; validating it twice would put one rule in two owners.

---

## 4. How one point is stamped

`Dispatcher.materialize_unit` deep-copies the parent state and forwards only
`coordinates` and `velocities` from a raw task descriptor (`dispatch/dispatcher.py:43-67`). A grid point is
a configuration, not a geometry, so the scan materializes its units itself
and writes into each child's task config before handing them to `dispatch()`.
Units that already carry a `'state'` pass through untouched
(`dispatcher.py:312-317`), which is why neither `Dispatcher` nor
`task_system_updates` needed to change.

The object path is the one the task queue builds (`data/core.py:1769-1775`)
and the skill reads (`skills/nn_pes_skill/backend.py:84-86`):

```
unit['state'].theory[i]['config']['skill_config']
    ['eg' | 'nac' | 'soc'][keyword]   the point's value, member-axis safe
    ['profile']                       '<method>_train'
    ['modeldir']                      <grid_root>/NN-<title>-<key>, absolute
    ['train_data']                    absolute path
and on the task itself
    ['capabilities']                  cleared: a training reports no property
```

Clearing the child's capabilities matters: the parent's plan asked for energy
and gradient, and a child that left those on its task would be judged against
channels it never computed. With the list empty the child behaves exactly
like a `jobtype train` run, which is the shape the resolver deliberately
leaves unexpanded (`planning/resolver.py:50-57`).

The parent state is never mutated (`test_grid_plan.py::test_the_parent_state_is_never_touched`).

### 4.1 Point keys and directories

A point's key is its values rendered in keyword order and joined by `~`:
`depth=1~nn_size=24~dropout=0.005`. The separator is not `.` because a
rendered float carries one, and a key that cannot be split back into its
pairs is not the reversible name a restarted scan re-derives its folders
from. Two naming schemes are used, for two different jobs:

| Directory | Named by | Why |
| --- | --- | --- |
| runtime directory | `nn_pes_skill_runtime_<idx>` (the manager's default) | `RuntimeDirectoryManager` reuses existing indices in ascending order, which is what lets a restarted parent map its batch back onto its previous runtimes (`dispatch/dispatcher.py:418-463`) |
| model directory | `NN-<title>-<key>` | the durable record of a point: re-derivable from its values alone, so an index reassignment can never cross-wire two points |

`runtime_root()` points the runtimes under `grid_root`, and
`runtime_isolated_workdir()` is true, so each point's calculation directory
and skill log stay inside its own runtime directory instead of appending
into a neighbour's.

---

## 5. How results are read

The original project scraped a finished training out of its own log: it read
the last ten lines of `NN-<title>.log`, looked for the section title
`&nn validation mean absolute error` and took the numbers two and three
lines below by position (`legacy/.../remote_train.py:106-151`). A missing
marker became a row of zeros, which is indistinguishable from a fit that
reported nothing. This scan reads structured products instead, in this
preference order:

1. **The returned record** (`grid/record.py:point_record`), which handles the
   three carriers a dispatched point arrives in: the live child state, a
   result wrapped around the state dump, and a plain property result.
   `train_errors` arrives already unscaled into eV, eV/Å and cm-1
   (`skills/nn_pes_skill/service/trainer.py:physical_errors`).
2. **The folder**, through the skill's own TF-free reader
   (`skills/nn_pes_skill/service/train_record.py`), for `retrieve` and for a
   point whose parent process died: the per-member `fiterr_valid_v<i>.npy`,
   the `hyper_v<i>.json` that names the model type, and the
   `scaler_v<i>.json` spread the stored error is multiplied back by. The
   channel table and the unit names are imported from the same module the
   fit report uses, so a folder read back and a folder reported at the end
   of its own fit cannot disagree.
3. **Never** the human log. `&nn validation mean absolute error` stays a
   printed product.

A point is failed when no record came back, when `train_errors` is empty, or
when a folder lacks one of its member products. That is the same rule the
serving path applies when it refuses a folder whose fit never finished
(`service/calculator.py:176-187`), and `service/train_record.py:read_model_dir`
drops a folder that reports no channel instead of ranking an empty dict.

### 5.1 Ranking

| Choice | Decision | Reason |
| --- | --- | --- |
| rank on | one channel (`sort_by`), default the profile's first | derived from the profile's ensembles, never invented |
| statistic | worst member (`max`) by default | a folder is deployed as a whole, and its worst member is the surface that gets served -- the same reasoning as `service/deviation.py:88` |
| ties | broken on the point key | the same scan reads out the same table twice |
| a point that reports nothing | last, in either order | it is not a large error, it is no error |
| verdict | `mark_completed` with `nfailed` unless every point failed | a 200-point scan where one Keras child died is a successful scan |

Products, all under `grid_root`:

| Artifact | Content |
| --- | --- |
| run log `&grid search` | the plan table, written **before** the fan-out, so a scan that dies halfway still leaves what was asked for |
| run log `&grid search summary` | the ranked table, with a unit row per column and `--` for a channel a point does not carry |
| `summary.csv` | the same ranking, machine-readable, one header row, plus a detail block keyed by role/channel/member |
| `crashed.txt` | the points with no record, and why. Not written at all when nothing crashed |
| `state.pmd.results['grid_search']` | points, ranking, counts, the ranked channel and the CSV path |

---

## 6. Execution modes

| Legacy spelling | This framework |
| --- | --- |
| `use_hpc 0` (one pool of 1, `chdir` per point) | `&search multitask: false` (serial, in this process) |
| `use_hpc 1` (`Pool(min(nsearch, ml_ncpu))` + `sbatch -W`) | `multitask: true, remotetask: false, multipool: <n>` (spawn pool over local children) |
| cluster submission | `multitask: true, remotetask: true, multipool: <n>` (pool over `slurm_run`) |
| `retrieve 1` | `&search retrieve: 1` |
| `ml_ncpu` | `&search multipool` |

Two rules the planner already enforces make this work without touching the
dispatch package: the declaration layer beats the input keyword, so
`WORKFLOW_NODE_DECLARATIONS['search']` had to stop forcing `multitask: False`
(`dispatch/planner.py:resolve_flag`); and every dispatched child is a worker
that never re-plans and never re-dispatches, so a point resolves to
`SingleShot` and cannot recurse into another scan.

A single point's own failure is caught in `_child_entry` for the three
refusal classes the skill raises (`FileNotFoundError`, `FileExistsError`,
`ValueError`) and reported as a failed row; a defect of any other kind keeps
its traceback and stops the run, because a scan must not launder a pipeline
bug into a data point.

A point that dies **in the parent** is a third case, and it is the remote
arm's usual one: the submission itself fails, or the child reaches a terminal
state without writing its result. That exception leaves `dispatch()` and used
to take the whole batch with it -- no table, no `crashed.txt`, and the fits
that had landed on disk lost their place in the ranking, which is the scan's
only product. `workflow/search.py:run` now records the sentence as the reason
of the points that have nothing on disk (`batch dispatch stopped (...)`) and
still writes the table; `grid/record.py:build_row` prefers that sentence over
the folder's own "no model directory", because the absence is a consequence
and the parent names the cause, and a point whose fit did land keeps its
numbers whatever happened to the rest of the batch.

**The shipped default is the remote arm.** `examples/nn_grid_search.yaml`
submits each point as its own child runtime and states the allocation in a
top-level `slurm:` section, which the converter renders as `&SLURM` so the
classic carrier cannot ask for something else (see
[INPUT_SYSTEM_DEVELOPMENT.md](../INPUT_SYSTEM_DEVELOPMENT.md) section 7.5).
`examples/nn_grid_search_local.yaml` is the same scan over a local worker
pool and carries the daily gate. Off a cluster the remote arm is a named skip
in qualification rather than a failure; what is verified here and what is
verified in the test environment is the first limit of section 10.

---

## 7. Continuation, restart and skipping

A scan is long, so these three paths matter, and each is a different thing:

| Situation | Behaviour | Where |
| --- | --- | --- |
| a point's folder is already complete and this is a plain training request | the point is **skipped**, its numbers are read from the folder and ranked, and it is marked `skipped` in the table | `grid/record.py:skipped_row`, decided by `folder_complete` |
| the same request reaches the trainer anyway | **refused**: `already holds a trained ensemble, and this request fits a new one ... Write initialize_weights: false in the eg section` | `service/trainer.py:targets_with_data` |
| a scan planned under `train_mode: retraining` | **refused at the plan**, before a point is stamped: a scan fits every point into a new directory, and a continuation cannot apply the architecture the grid varies | `grid/plan.py:build_context` |
| `retrieve: 1` | nothing is submitted; every point is read from disk | `workflow/search.py:run` |
| a restart of the whole run | runtime indices are reused ascending, and model directories are re-derived from the keys, so the same input maps back onto the same work | `dispatch/runtime.py`, `grid/plan.py:point_key` |

Skipping is what the refusal forces, and it is stated here rather than
buried: running the same input twice means "fill in the points that are
missing", not "train them again". Training a *continuation* over a point
(`train_mode: retraining` plus the member's `initialize_weights: false`)
still re-fits that folder; the scan stamps a plain `training` request per
point, so the continuation switch belongs to the individual run, not to the
scan.

---

## 8. What was measured this round

All on the test environment, Python 3.11.16, TensorFlow 2.21.0, the shipped
`egs_small.json` fixture (18 atoms, 2 states, 24 frames), `mult_nn: 2`
except where noted. These numbers measure the **mechanism**; a 2-epoch fit
is not a surface anyone would deploy.

| Case | Result |
| --- | --- |
| 4 points (`depth: [1, 2]`, `nn_size: [8, 16]`), serial | all four trained, four model folders and four runtime directories created, `summary.csv` written |
| same input re-run | four rows `skipped`, ranked from the folders: best `depth=1~nn_size=16` at 0.10891268 eV (worst member), then 0.12316324, 0.13489195, 0.19245446 |
| `retrieve: 1` | identical ranking, run wall **4 s**, no submission |
| 2 points (`depth: [1, 2]`), `multitask: true`, `multipool: 2`, `mult_nn: 1` | both trained in worker processes, per-point wall 11.0 s and 11.3 s, run wall 19 s, statuses `ok` |
| 2 points, `remotetask: true` (no cluster client on this environment) | two child runtimes materialized under `grid-remote/`, each with its own `.rt` and `slurm.sh`; reading the payloads back gives `jobtype=train`, empty capabilities, `profile=mlp_eg_train`, and distinct absolute model directories (`NN-<title>-depth=1` and `-depth=2`). Submission itself stopped at `sbatch: No such file or directory`, which is the arm's verified boundary |
| the shipped remote arm, 4 points, `multipool: 4`, with `sbatch`/`sacct` shimmed on `PATH` | four runtime payloads and four `slurm.sh`, one per point, each carrying `--partition=default`, `--mem=4G`, `--time=0:20:00` (the host's own section states different mem and time, so the input's section demonstrably arrived) and no `--exclusive` line. Every point then fails for the reason a stub must give (no `result.json`), and the scan writes `summary.csv` with four `failed` rows and `crashed.txt` naming that reason: the table survives a batch whose submission died |
| the shipped local arm, 4 points, `multipool: 2` | run wall **31.5 s / 32.4 s / 35.9 s** on three runs of the same input, 12.4-12.7 s per point, four rows `ok`, no `crashed.txt`, `Status: completed` |
| the same arm through the classic carrier the planner generates | `example_nn_grid_search_local_input` ran in its own directory: **34 s**, `Number of search: 4`, four `ok` rows in `summary.csv`, four key-named model folders and four runtime directories |
| one `slurm:` section stated three ways (YAML block, `&SLURM` text, agent `.config`) | the same `#SBATCH` line set out of each; `exclusive false` read back as `False` in all three, and the text carrier demonstrably rendered the section |
| fresh single training | `mark_completed`, verdict banner written |
| the same training re-run | refused: `FileExistsError` naming the folder and the switch to write |
| continuation (`train_mode: retraining`, `initialize_weights: false`) | the fit subprocess reported `Load old weights at: .../weights_v0.weights.h5` and `Transferring weights...` |
| prediction over the 24-frame test set | run wall **2.7 s**; MAE 1.78331138 eV (energy), 1.27836250 eV/Å (gradient); every printed statistic re-derived from the raw columns |
| prediction over the coupling test set (`mlp_nac2`, 24 frames, one member) | MAE 4.480195 eV/Å, RMSE 6.951833, MAX 33.596223; the single member reports `spread_max` 0.0 rather than no line |
| `&search grid` read through all three carriers (two `grid` lines, one `;` line, the yaml mapping) | the same `{'depth': [1, 2], 'nn_size': [8, 16]}` out of each |

Three defects were found by these runs rather than by reading, and all three
are fixed above: the serving profile on the parent (section 3), the
declaration that made every scan serial (section 6), and the pooled arm
failing to pickle its own payload because the batch copy carried the parent's
calculator and a bound executor (`workflow/search.py:_create_copy`,
`_child_entry`).

Driving the remote arm to its artifacts found three more, none of which the
local arm could show: an input's `slurm:` section reached no job script at all
(the merge sat behind the `&environment` guard, and a scan states no such
block), the kernel environment let one run's directives be read by the next run
in the same process as its host defaults, and a point whose submission died in
the parent ended the whole scan instead of becoming one row's reason. All three
are judged by gates now, listed in section 9.

---

## 9. Which gate pins which contract

| Contract | Gate |
| --- | --- |
| the manifest declares only real hyperparameters, kind is read and never copied, `get_metadata` carries the surface, discovery imports no TensorFlow | `skills/nn_pes_skill/tests/test_searchable.py` |
| a train profile per method, the role sections and channels a point stamps, `search` resolves to a trainable profile | same file, `TestSearchTargets` |
| candidate refusal by type, kind, bounds and declared choice | same file, `TestCheckSearchable` |
| grid expansion order, key reversibility, the member-axis table, `write_point` onto a real materialized unit, parent state untouched, refusal of an empty grid / missing data / multi-skill plan | `qualification/verification/test_grid_plan.py` |
| the three result carriers, the folder read including per-member unscaling, half-product refusal, ranking by max/mean, ties, absent points, the two tables, `summary.csv`, `crashed.txt` | `qualification/verification/test_grid_record.py` |
| the ML verdict, the unchanged non-ML path, `GridSearch` registration, `child_jobtype`, the serial-forcing declaration gone, `&search multitask` reaching the graph node | `qualification/verification/test_ml_verdict.py` |
| call order of a continuation, the three refusals, the architecture guard | `skills/nn_pes_skill/tests/test_train_resume.py` |
| the prediction record, its raw files, and the unit chain of the gradient | `skills/nn_pes_skill/tests/test_predict_products.py` |
| the coupling channel of a test set ((npair, natom, 3), pair order, eV/Å) and the single-member spread | same file, `TestRawFiles::test_the_coupling_channel_is_compared_channel_by_channel` and `TestUnitChain::test_a_single_member_reports_no_spread_rather_than_no_line` |
| `&search grid` equal across the three carriers, and its refusals; a scan refused under `train_mode: retraining` | `qualification/verification/test_ml_verdict.py::TestSearchGridCarriers`, `::TestScanRefusesARetrainingRequest` |
| a run's `slurm:` section reaching the snapshot from all three carriers, applying with or without an `&environment` block, not being inherited by the next run in the process, and the text `false` reading back as a boolean | `qualification/verification/test_environment_snapshot.py::TestSlurmSectionCarriers` |
| the shipped scan example's directives arriving through both its carriers, and the two arms asking for the dispatch they are named for | `qualification/verification/test_ml_example_input.py::TestShippedMlExamples` |
| the remote arm planned as a named skip that claims no run directory, with the submission request read from the input rather than from a file-name convention | `qualification/verification/test_manager.py::TestExampleChains::test_the_remote_arm_is_a_named_skip_without_a_scheduler` |
| a submission that dies in the parent becoming a row's reason, outranking the folder's own sentence, and not rewriting a point whose fit landed | `qualification/verification/test_grid_record.py::TestTables::test_the_dispatch_death_outranks_the_absent_folder`, `::TestTables::test_a_completed_folder_is_not_judged_by_the_batch_death` |

---

## 10. Known limits

1. **A scan ranks, it does not judge.** Nothing here decides that an error is
   acceptable; there is no threshold keyword, because a validation MAE is not a
   convergence criterion and a number that passed a review would become a magic
   constant.
2. **Only the fit error is comparable.** `sort_by` reads what a training
   reported. The test-set statistics of a prediction run are a different
   record and are not part of a scan; a scan cannot yet rank points by how they
   generalize to data they were not fitted on.
3. **One skill family, one measured metric path.** The scan reads the records
   an `nn` training wrote; a second skill can declare its own `searchable:`
   block, but nothing has been scanned but `nn` here.
4. **No adaptive or sequential search.** The grid is planned in full before
   anything runs. Early stopping a branch, random sampling, and Bayesian
   selection are all absent, and the plan has no place to report "the next
   point depends on the last".
5. **One skill per scan.** A plan that queues more than one skill has no answer
   to which one the grid belongs to, and is refused.
6. **A point's framework log is the parent's scan table.** A dispatched
   child no longer writes the parent's run log: two independent handles on
   one file overwrite each other, and one scan left 10475 NUL bytes in its
   own log before that rule existed. The child's verdict sentence therefore
   reaches the ranking table through its runtime report
   (`grid/record.py:point_failure`) rather than a log line, and a point's
   own evidence stays in its runtime directory (skill log, engine output,
   result file).
8. **The remote arm is verified to the artifact, not through a queue.** What
   is checked on the test environment is everything the framework controls: the point
   becomes its own runtime payload, the submission script is built from the
   run's `slurm` section, one script per point with distinct model
   directories, and a submission that dies leaves a named row instead of no
   table. What is not checked here is the scheduler's own behaviour -- queue
   time, an allocation the cluster declines, a child that dies on a compute
   node -- because there is no cluster client on the host these numbers come
   from. Off a cluster the shipped example is a named skip in the plan
   (`skip_reason` naming `sbatch`), so a host without one reports "this
   machine has no scheduler" rather than "the framework is broken", and the
   arm's end-to-end verdict belongs to the test run.
7. **The Slurm arm is verified down to the per-point payload and the
   generated job script** (section 8): each remote child really carries its
   own point. Submission itself is unexercised, because the test environment has no
   cluster client.
8. **A prediction has been measured on energy, gradient and couplings.** The
   spin-orbit channel goes through the same code with a factor of 1.0 and no
   derivative axis, and has not been run against a test set here.
9. **Two evaluations of one project and profile write the same prediction
   file.** The naming is keyed by `{project}_{profile}` as every other product
   of this skill is, so a second test set overwrites the first unless the
   project differs.
10. **A point inherits the parent's data file.** There is no per-point
   `train_data`; varying the training set is a different experiment than
   varying a hyperparameter.
