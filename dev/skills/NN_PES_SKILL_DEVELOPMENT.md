# NN_PES Skill Development Guide

Last updated: 2026-11-06 17:30 (section 10 records the Keras-3 port's
numerical-equivalence and version-wall evidence, and section 1.1 no longer
names the retired standalone tree. The migration-era project directory was
removed from the working tree; the vendored copy under
`service/nn_pes/` is the single source now.)

Before it, 2026-10-23 22:15 (section 6b now names the two training-data input modes and the framework-side permutation preprocessing; section 8 gained the channel-extension rule. The rest was not re-audited.

Before it, 2026-10-22 22:30 (re-read against the code end to end; no statement needed changing - the manifest capability list, `query.py::resolve_profile`/`check_searchable`, `schema/hyper.py`, the trainer chain and the fixture paths all match. Nothing was re-measured.

Before it, 2026-10-15 12:10 (section 6's spin-orbit evidence sentence
re-read off the shipped fixture: the bound is 20 cm-1 at 40 epochs, not the
60 cm-1 at 60 epochs written there - neither number matched the fixture, which
ran 5 epochs until this round - and the fit is seeded. The rest of the file was
not re-audited line by line

This document covers **nn_pes-specific** implementation details for
`nn_pes_skill`. For the abstract skill development specification
(directory structure, schema conventions, backend architecture, profile
design), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).
User-facing contracts (units, profiles, configuration keys) are in
`pyrai3md/skills/nn_pes_skill/skill_guide.md`.

---

## 1. What this skill wraps

### 1.1 A vendored model container, not an external program

`nn_pes_skill` is the second skill in the package that owns its science
in-process (the first is `qdm_skill`). Instead of driving an executable,
it carries the neural network potential container under
`service/nn_pes/`: the model classes, the geometric feature layers, the
scalers, and the training modules. That tree is the package-maintained copy
of the standalone NNsMD project (itself ported from
`legacy/PyRAI3MD/Machine_Learning/NNsMD` to Keras 3), with two changes:

- Every `NNsMD.*` import became a package-relative import, so nothing in
  the vendored tree names a project root.
- The plotting modules were not brought over; the training scripts keep
  their numeric output and their per-member log.

The single-source rule is guarded by a test rather than by convention
(`tests/test_train_launcher.py::TestSingleSource`): no framework code may
import the legacy tree. The migration-era working copy that produced the
equivalence record in section 10 has been retired.

### 1.2 It fills a label that already existed

`schema/validate.py` defines `BUILTIN_THEORY_METHODS = {'nn', 'e2n2'}` with
the comment that these are machine-learning potentials that have **no skill
directory of their own**, and the classic input grammar already accepts an
`&nn` block with `qm = 'nn'`. The manifest therefore declares
`name: nn`: the skill is what that label was pointing at, not a new
concept. `e2n2` keeps its builtin status until something implements it.

---

## 2. The import boundary around TensorFlow

TensorFlow is a heavy optional dependency and discovery is allergic to
`ImportError`: `skills/query.py:discover_all()` wraps each skill import in
`except ImportError: pass`, and importing a submodule runs the package
`__init__` first. The shipped convention of re-exporting the backend class
at module level (`qdm_skill/__init__.py`) would therefore make this skill
**silently vanish** on a machine without the extra. Three rules keep that
from happening:

1. `nn_pes_skill/__init__.py` resolves `NNPESBackend` through PEP 562
   `__getattr__`, so importing the package touches nothing heavy.
2. `service/runner/__init__.py` does not re-export the training launcher.
   The launcher path (`nn_pes_src/fit.py` → `selection` → `models`) is what
   pulls Keras in, and it is only reached when a training run starts.
   `fit.py` itself is stdlib-only now: the module-name mapping moved out of
   `selection.py` into it, because that mapping is launcher logic.
3. `backend.py` imports the container only inside `__init__`, after
   `validate_environment()` has raised if TensorFlow is missing. The error
   is a `RuntimeError` naming the package, not an `ImportError`.

`tests/test_discovery_without_tensorflow.py` runs this in a child
interpreter with a `sys.meta_path` blocker, because blocking inside the
test process would break the numeric tests that need TensorFlow.

Qualification mirrors the same honesty: the manifest lists
`requires: [tensorflow]`, `get_metadata()` passes it through, and
`certification/planner.py:_missing_requirements()` turns an unimportable
requirement into a diagnostic skip. Measured three-state behaviour with
`skills.nn_pes_skill.test` switched on: TensorFlow present → cases and
runtime tests planned; blocked → a single skip reading
`skill needs tensorflow, which is not importable on this environment`.

---

## 3. Units: measured, then fixed

The framework integrates in atomic units with Angstrom coordinates
(`workflow/integrator/verlet.py:46-51`), and it kicks with
`v -= 0.5 * gradient * inv_mass * dt`, so the reported gradient is
`+dE/dR`, in Hartree/Bohr.

Two sources disagreed about the container: the fixture geometries are in
Angstrom (shortest C–C bond 1.3337, which as Bohr would be an impossible
0.706 Å), while the legacy caller treated the gradients as Hartree/Bohr
(`legacy/PyRAI3MD/Machine_Learning/model_pyNNsMD.py:129-131`).

### 3.1 The instrument that was rejected

A trajectory regression `dE = c · Σ g·ΔR` over consecutive frames gave
per-state slopes 0.21/1.50/1.08/1.07 and a pooled median of 1.038 with a
95% CI of [0.85, 1.19]. Its first negative control did not collapse; after
crossing the pairing (work from one pair against `dE` of another) states
1-3 fell to -0.00/-0.10/-0.12, so the estimator does see something real.
But MD steps are far too large for a first-order expansion, which
systematically attenuates the slope, and an attenuated estimate cannot
separate 1.0 from 1.8897. **That instrument was discarded and is not
evidence for anything.**

### 3.2 The instrument that settles it

Consistency inside one trained model: serve a geometry, then central-
difference the energy the same call reports, with respect to the
coordinates as fed. Sweeping `h` from 3e-5 to 3e-2, the ratio of the
container gradient to that derivative converges to **0.99998**
(p10 0.99906, p90 1.00094 at the converged step sizes). Three orders of
magnitude away from 1.8897, so:

- the network's gradient is the derivative with respect to its **input
  coordinate unit**, which for these folders is Angstrom;
- the boundary multiplies by `BOHR_TO_ANGSTROM = 0.52917721090299996` to
  hand over Hartree/Bohr;
- this is fixed behaviour, not a configuration knob. A knob here would let
  a user turn energy conservation off.

Re-verified end to end through the backend: the reported gradient against
the finite difference of the reported energy has a maximum relative
residual of 4.7e-05 to 7.8e-05 (`tests/test_train_predict.py`).

### 3.3 Inbound labels, and why there is no keyword for them

A training file carries Hartree energies and Hartree/Bohr gradients and
couplings. That is the framework's own convention, the one every engine
backend reports, so the skill reads it and asks nothing else:
`service/trainer_data.py:match_labels` moves the derivative onto the
Angstrom axis the network is actually fed, and the serving boundary moves
the result back.

The earlier port had `eg_unit` and `nac_unit` with three accepted schemes.
They are gone, for one reason: a unit switch's failure is silent. Declaring
a scheme the file does not use trains a network whose reported gradient is
the derivative of its own energy by a factor of 1.8897, and the first thing
that notices is energy drift hundreds of steps later. "Energy and gradient
describe one surface" is not a preference to configure but the definition
of a trained surface, so it is enforced rather than offered.

A file that describes itself is checked, not trusted: a `unit` block naming
any other scheme is refused when the data is read
(`service/trainer_data.py:check_units`).

What is displayed is a different unit from what is read. The fit stores its
validation error in the scaled units of its own normalizer, where a number
means neither milli-Hartree nor milli-eV, so the summary multiplies the
spread back out of the scaler the folder was fitted with and prints eV and
eV/Angstrom (`service/trainer.py:physical_errors`,
`schema/reference.py:METRIC_*`), which is also the convention the container's
own `plots` section declares. A spin-orbit folder is the exception on both
sides of that sentence: its labels are already cm-1 and its metric stays
cm-1, because a matrix element is a value rather than a derivative, so no
length factor enters it in either direction.

---

## 4. Serving: compiled once, measured

Weights are loaded once at backend initialization on every path, so the
question "save the model or not" is not what separates the designs; the
difference is whether each step re-traces the vectorized Jacobian. The
eager container path re-traces it and TensorFlow says so in a warning.

Measured on a CPU-only test environment, 18 atoms, 2 states, `mult_nn = 2`, arms
interleaved:

| path | batch=1 per step | batch=108 per geometry | agreement with eager |
|---|---|---|---|
| A eager `model(x, training=False)` | 1274 ms | 11.9 ms | baseline |
| B per-member `tf.function(input_signature=…)` | 1.43 ms | 30 µs | energy `0.000e+00`, gradient `1.13e-08` |
| C exported SavedModel `.serve()` | 1.34 ms | 33 µs | energy `0.000e+00`, gradient `1.13e-08` |

Adoption rule written before the measurement: implement B or C only if a
batch=1 speed-up of at least 2x with agreement no worse than 1e-6. Both
clear it by three orders of magnitude, so **B is the serving path** and C
is not adopted: it needs an export step at training time, it is no faster
in practice, and it carries the restored-root-object trap (dropping the
`tf.saved_model.load` object while keeping only `.serve` produced
`Called a function referencing variables which have been deleted` during
this very measurement).

Implementation notes:

- Compilation is per ensemble member, keyed by the member object in a
  `weakref.WeakKeyDictionary`. An `id()`-keyed cache was rejected because
  ids are recycled after collection and would resurrect a trace bound to a
  dead model.
- The input signature is `[None, *trailing dims]`, recorded on the member
  itself so it dies with it. A later call with different trailing
  dimensions raises instead of silently re-tracing per step.
- The container comment saying the member loop is deliberately Python is
  still true: tracing *the list of models* buys nothing. What buys 900x is
  tracing each member once.
- Skill-side per-step cost measured 1.83 to 1.94 ms; the test budget is
  100 ms so a regression back to eager fails fast.

---

## 5. Training: one subprocess per member, two arms

Granularity is the container's own: **one process per (ensemble, member)**,
all submitted before waiting (`nn_pes.py:440-462`), with the GPU index
carried per member. The two arms run the identical command line:

- `local`: `Popen([sys.executable, '-m', <prefix>.service.nn_pes.nn_pes_src.training.training_mlp_<type>, -i, …])`
- `slurm`: the same command written into a job script, submitted
  `sbatch -W`, one independent job per member, and the parent spawns one
  local process per submission so the concurrency semantics match the
  local arm. The skill does not reimplement the framework's sacct polling
  (`workflow/dispatch/slurm_run.py`).

The package prefix is computed from `__name__` at runtime and the child's
`PYTHONPATH` carries the parent's absolute `sys.path` entries, so no
installation path is written into skill code and the same module works
outside the framework.

Order of operations is fixed: write hyperparameters and pickled data →
start members → wait for all exit codes → reload → read the fit errors.
Reordering lets a reload resurrect weights from an earlier run.

Failure semantics that a test enforces:

- a member exiting non-zero makes the run raise with the exit code and the
  `fitlog_*.txt` path, and nothing is reported as a finished fit;
- serving refuses a folder whose `fiterr_valid_v*.npy` files are missing.
  This matters because the parent saves freshly initialized
  `weights_v*.weights.h5` **before** the fit: a dead run leaves loadable
  files behind, and loading them would serve noise as if it were a surface.

Measured timings: 6 epochs × 2 members, 24 geometries → 14.7 s; 20 epochs →
25.0 s.

### 5.1 Threads

The container takes its thread count from the environment, so the skill
sets it once per process. `service/runner/env_check.configure_threads`
calls `os.environ.setdefault('OMP_NUM_THREADS', <thread>)` before the
serving objects are built, and writes a line to stderr when the process
had already latched a different value -- the same first-wins contract
OpenMP imposes on `pyscf_skill` (CHANGELOG #302). The slurm arm exports
the same value in each member's job script, so a submitted member runs
with the thread count the input asked for rather than whatever the
submitting shell carried.

The keyword is `thread`, and `nn_pes_thread` is the framework's spelling of
it; the larger of a pair wins (#306). Two lifetime-specific budgets sit
beside it: `train_thread` for each training member subprocess and
`predict_thread` for the serving TensorFlow runtime. Both inherit `thread`
when empty/0. The training launcher writes `OMP_NUM_THREADS` and the two
`TF_NUM_*` variables per child; the serving path publishes them and applies
`tf.config.threading` before the first op, so an MD child can use a
different count from the fit that produced the model.

### 5.2 What survives a run

`keep_tmp` clears the calculation directory, but at the end of the backend's
life rather than at the end of a step (#307). The directory holds what a
serving run accumulates step by step -- `{project}_{profile}.log` and the
uncertainty dump -- and both writers append without recreating the directory,
so a mid-run clear would keep only the last step and then record nothing at
all, the `OSError` being swallowed. The cleanup is therefore bound to the
backend object with `weakref.finalize` (`backend.py:_discard_calcdir`), the
same lifecycle the resident LAMMPS service uses, and it also runs at
interpreter exit.

What a training run produces is not scratch and no keyword governs it: the
ensemble (weights, scalers, hyperparameters, `fiterr_valid_v*.npy`,
`fitlog_*.txt`) lives in `modeldir`, outside the calculation directory, so a
cleared directory cannot lose the surface.

The one thing inside the directory that a caller may want back is
`{project}_train.summary.txt`, whose path the training record returns as
`train_summary`. With `keep_tmp: 0` the teardown clears it and the returned
path dangles. Nothing is lost of the numbers themselves -- the framework
report prints `&nn training` and `&nn validation mean absolute error` from the
returned `train_errors`, and the read-back cases in `tests/test_train_predict.py`
declare `keep_tmp: 1` for that reason -- but a caller that wants the file has
to ask for it. `tests/test_keep_tmp.py` measures the two lifetimes apart.

---

## 6. Coupling channels: declared, never derived

`mlp_nac` sizes its output by `states` (`models/mlp_nac.py:79`), while
`mlp_nac2` sizes it by `nnac` (`models/mlp_nac2.py:68`). The data layer of
the original project counted channels as `nstate * (nstate - 1) / 2`
(`legacy/PyRAI3MD/Machine_Learning/training_data.py:125`). So the channel
**count** comes from the model, but which state pair a channel means is a
property of the training data and cannot be inferred from either.

The skill therefore does not guess, and it does not enumerate. The state
pairs are a declaration the framework already owns: `&MOLECULE`'s
`coupling` becomes `nac_pair_map` and `nnac` in
`schema/normalize.py:verify_state_num`, and the coupling network is sized
by that count. Declaring one pair trains one channel; nothing is added to
complete a ladder, and no all-pairs traversal is used to invent a width the
caller did not ask for.

Three agreements follow from reading the dimension that way, and all three
are enforced rather than assumed:

- the training file's channel count equals the declared count
  (`service/trainer_data.py:check_channels`);
- `mlp_nac`, whose virtual-potential scheme builds one output per state,
  is only accepted for a declaration of exactly `nstate` pairs; an other
  size is refused with a pointer to `mlp_nac2`, which takes any count;
- at serving the loaded ensemble's width is compared with the declaration
  (`service/calculator.py:open_container`), so a folder trained for one
  pair set cannot be reported under another's state names.

`target_nac_pair` then selects and orders what the caller wants out of the
declared channels, and a pair outside the map raises. A count declared
without any pair (`nnac` alone, no coupling section) serves the channels in
the order the model holds them and raises if a specific pair is asked for,
because there would be nothing to match it against. Zero-filling a missing
channel was rejected because a silently absent coupling changes a hopping
trajectory.

This mapping is unverified against a real trained coupling surface: the
regression fixture carries `nnac = 0` and the shipped coupling fixture is
synthetic. Treat it as a contract the caller must confirm against their own
data.

---

## 6b. Input layout: one shared block, one block per network family

Every ML method shares the settings that are not about a network: where
the models live (`modeldir`), where the data comes from (`train_data`,
a file name, or `train_dataset`, the same mapping inline), whether the
run fits or refits (`train_mode`), how many members an ensemble has
(`mult_nn`), and which device each training subprocess gets (`gpu`, a
`;` list of device indices that broadcasts from a single value).
Exactly one of `train_data` / `train_dataset` is set
(`schema/validate.py`); both modes reach one reader
(`service/trainer_data.py:from_payload`, `load`). The inline form exists
for the framework's permutation preprocessing: `&train permutation_map`
is expanded by the train workflow before backend initialization and
written to `train_dataset`, with `train_data` cleared. The skill never
reads a map and never reorders atoms
(`dev/methods/TRAIN_DEVELOPMENT.md` section 4.4).
That is the `&nn` block, and most keyword names are the ones the original
project used, so a PyRAI2MD input translates without renaming. Two are not:
`gpu` was a mode number there (1 meant "use the first device", 2 "spread
over two"), and here it is the device list it always wanted to be, so a
legacy `gpu 1` reads as device 1. The label units were `eg_unit`/`nac_unit`
there and are not keywords at all here (section 3.3).

Each ensemble then gets its own block of hyperparameters, `&eg`, `&nac` and
`&soc` (`schema/hyper.py` owns that vocabulary: 45 keywords, each with its
kind and the dotted path it fills inside the container's dictionary).
Legacy spelled the second member of an ensemble as a second block, `&eg2`
and `&nac2`; the member axis now lives *inside* one block, on the `;`
separator, so a block is read once and expanded per member:

```
&EG
depth          3               # one value: both members
nn_size        100;60          # member 0 is 100 wide, member 1 is 60
loss_weights   1 1;1 10        # a list value, one per member
```

The member axis is always the outer one, in every carrier: one `;` segment
in the classic text, one element of the outer list in yaml and JSON. That
is why `nn_size` stays a scalar. It is the width of every hidden layer, so
`nn_size: [100, 60]` would have to mean either "two layers" or "two
members", and the two are different shapes of the same word. Writing
`[[100, 60]]` to reach the second reading would put the burden of a
dimension on the reader; `nn_size: '100;60'` says what it is.

`mult_nn` decides how many groups are read: one group broadcasts to every
member, surplus groups are ignored, and a number of groups between one and
`mult_nn` raises rather than training a shorter ensemble than asked.
Ensembles larger than two have never been measured here, so nothing claims
they are useful.

The framework knows nothing about any of this vocabulary. A skill that
needs blocks beside its own names them in `manifest.yaml` under
`subsections`, exposes them through `query.get_subsections()`, and the
framework folds each one into the skill config under the same key
(`schema/adapter.py:_merge_one_skill_config`) and renders them back out of
a yaml or an agent config into `&EG` / `&NAC` blocks
(`qualification/certification/handler.py:build_subsection_blocks`). Three
carriers, one meaning, verified key by key and member by member in
`qualification/verification/test_nn_pes_input.py`.

---

## 6c. Spin-orbit: a scalar channel of the energy family

A spin-orbit matrix element is one number per state pair that depends on
geometry, and it is not the derivative of anything. So the request is
served by the network shape that already outputs scalars: `mlp_e` in the
`soc` folder, with its `states` axis widened to the declared pair count and
`energy_only` left on, which keeps the autodiff gradient head out of the
graph. `soc` in a jobtype resolves to `mlp_e_soc`, and a dynamics run that
wants energies, couplings and spin-orbit values in one call resolves to the
composite `eg_nac_soc` (or `eg_nac2_soc`), which loads three ensembles from
one `modeldir`.

Two things follow from reading it that way. The channel count comes from the
declared spin-orbit pairs (`soc_pair_map`, resolved by the framework from
`coupling` and the state spins), and the width of the loaded folder is
compared against that declaration at serving. And the unit never passes
through a length factor: the file is read as cm-1, the network is fitted on
cm-1, and the metric is printed in cm-1, while `plots.unit_gradient` is
emptied so nothing downstream mistakes the channel for a force.

The evidence is a synthetic field (`tests/data/soc_small.json`): two smooth
functions of two different interatomic distances, spanning roughly 70 and
150 cm-1 and differing in sign, so a channel mix-up cannot hide behind a
similar value. A trained two-member ensemble reproduces its labels to within
20 cm-1 at 40 epochs, reports shape `(2, 1)`, carries no gradient channel,
and its summary prints cm-1
(`tests/test_train_predict.py::TestSpinOrbitTraining`). The bound sits under
the constant-mean predictor on that fixture (41.35 cm-1), so it separates a
network that holds the field from one that prints its mean, and it is a
measured constant rather than a sample because `control.gl_seed` reaches the
fit (`gl_seed + member index` on the child's `-s`; `gl_seed 1` measures
9.74 cm-1 on this environment). No
engine-calculated spin-orbit data has been put through this path.

---

## 7. Results contract

- `energy`: all trained states, Hartree, shape `(nstate,)`.
- `gradient`: `(nstate, natom, 3)`, Hartree/Bohr.
- `nac`: `(npair, natom, 3)`, Hartree/Bohr, in the requested pair order.
- `soc`: `(npair, 1)`, cm-1, in the order the caller declared its
  spin-orbit pairs. The channel is produced by the energy family with its
  derivative head off, which is the same arrangement the original project
  used for spin-orbit values (`legacy/.../hyper_pynnsmd.py:357` builds the
  SOC hyper with `class_name: EnergyModel` and `states: nsoc`).
- `fin_coord` / `fin_velocity`: set to empty explicitly, as every other
  shipped skill does.
- `err_energy` / `err_grad` / `err_nac` / `err_soc`: one scalar per computed
  property, the largest member-to-member standard deviation of the step
  (`service/deviation.py`). The property arrays that feed composite
  aggregation stay as they were; these are additional keys. The framework
  receives them on the channel it already had
  (`workflow/records/deviation.py:DEVIATION_KEYS`), compares them against the
  `&adaptive max*` thresholds, and prints them in its `&prediction error`
  block. The full spread arrays also go to
  `{project}_{profile}.uncertainty.txt`, and the numbers the backend
  returned go to `{project}_{profile}.log`, so a reported scalar can be
  re-derived from the file rather than trusted.

A training step returns no property channel at all. It returns
`train_errors` (role to channel to one value per ensemble member),
`train_metric_units`, `train_wall_seconds`, `train_arm` and `train_summary`,
and the framework prints that record as `&nn training` plus
`&nn validation mean absolute error` (`utilities/print_results.py`) in the
same section grammar the property blocks use. The skill keeps the identical
report in `{project}_{profile}.log` inside its calculation directory
(`service/trainer.py:format_training`), and `agent.py` writes it into the
log a standalone run leaves behind. Two details of that layout are
deliberate: a channel the folder does not produce prints `--` rather than
`0.00000000`, and a training result prints no `&electronic state`,
`&velocities` or `&kinetic energy` section, because the run observed none
of them.

The framework's `resolve_profile` call is the skill's own decision made
from the request it is handed, and `control.driver` enters that decision
only when the request carries dynamics: `jobtype: train` under a
skill-driven run propagates nothing, so a driver that would hand the
propagation to the backend must not make the resolution fail. When a
combination genuinely cannot be resolved, the framework keeps the skill
default but reports the substitution, since a training run that was
actually served reads as a completed run either way.

A test-set evaluation (`pred_data` set on a serving request) returns a third
record: `predict_errors` (channel to `mae` / `rmse` / `max` / `spread_max`),
`predict_metric_units`, `predict_frames`, `predict_wall_seconds`,
`predict_raw_files` and `predict_summary`. It is the *serving* call repeated
over every frame of the file, so the channels present are exactly the ones the
profile serves, and a channel it does not produce is absent from the record and
from the files rather than reported as a zero.

Two of its rules are worth stating where they can be misread.

- **The axis before the difference.** A label file carries Hartree/Bohr
  derivatives (`service/trainer_data.py:match_labels` puts them on the Angstrom
  axis the network is fed) while a served gradient comes back Hartree/Bohr
  again. The two are placed on one axis and only then subtracted, and the
  result is converted once into the printed eV/Angstrom. The factors are held
  apart in one table, `service/predictor.py:_CHANNELS`, and the chain is
  asserted by `tests/test_predict_products.py::TestUnitChain`, which writes a label
  file equal to the answered surface plus one Hartree and then demands the exact
  eV figure the chain implies.
- **The products are not scratch.** The raw comparison
  (`{project}_{profile}.{channel}.pred.txt`), the per-frame table
  (`{project}_{profile}.frames.txt`) and the summary are written into the run
  directory rather than the calculation directory, because `keep_tmp` governs
  TensorFlow scratch and a reader asked for these files. They carry four
  blocks of columns in one unit each -- reference, prediction, |deviation|,
  ensemble spread -- in that order, so a difference rebuilt from the first two
  meets the third (up to the ten significant digits the file prints).

`spread_max` is the ensemble's disagreement, not an error: a set of members
that agrees on a wrong surface says nothing about accuracy, and the name says
which of the two was measured. No relative or percent figure is reported,
because a derivative component can be exactly zero and a floor invented to
keep that division alive would then be printed as though it had been measured.

One correction recorded because it changed behaviour: `build_config` usedto treat an explicit `0` as "unset" and refill it with the default, which
made `mult_nn: 0` and `epo: 0` impossible to express and impossible to
reject. Only a missing key, `None`, `''` or an empty container counts as
absent now.

---

## 8. How to extend

- **New model family**: add the profile suffixes to `schema/method.py`
  (`PROFILE_ENSEMBLES`, `_METHOD_SUFFIXES`), the profiles to
  `manifest.yaml`, and, if it needs a new keyword, the field to
  `schema/config.py` **and** to `DEFAULT`. The defaults dictionary is the
  keyword vocabulary the framework filters by; a field missing from it is
  dropped before the backend ever sees it.
  `tests/test_query.py::TestDefaults` asserts the two sets agree.
- **Composite profiles**: anything that must answer several capabilities
  in one dynamics call follows `eg_nac` (a profile whose ensemble list spans
  two model folders in the same directory).
- **New searchable hyperparameter**: one line in `manifest.yaml` under
  `searchable:` (`{type: ...}`) plus, if it is not already there, the keyword
  in `schema/hyper.py:HYPER_KEYWORDS`. Nothing else: the kind a scan needs is
  *read* from that vocabulary, and `tests/test_searchable.py` refuses a
  declaration the trainer does not know. Which role section the keyword writes
  is derived from the profile being trained (`query.get_search_targets`), so a
  scan never guesses where a value belongs.
- **Which candidate values are admissible**: `query.check_searchable`, which
  applies the declared type, the keyword's kind and `hyper.check_value`. A
  grid scan plans through it, so a value the container would reject inside a
  fit subprocess is refused before a directory exists.
- **New training-data channel**: extend
  `service/trainer_data.py:from_payload` (shape/count checks) and
  `match_labels` (unit handling), then the consumers: `targets_for_role`,
  `service/train_record.py:CHANNEL_TABLE`, and the training report columns.
  A channel that no current profile fits is read and carried through but
  named as such; `transition_dipole`, `transition_dipole_norm` and
  `osc_strength` are in that state now. Do not add a permutation map to the
  skill: expansion belongs to the train workflow before backend
  initialization.
- **Continuation**: `train_mode: retraining` plus the chosen category's
  `initialize_weights: false` (`service/trainer.py:wants_continuation`). The
  folder is loaded before anything is written over it, and a request that
  would replace a trained ensemble is refused rather than warned about
  (`dev/methods/TRAIN_DEVELOPMENT.md` section 8.1).
- **Never** import the legacy NNsMD tree, and never add a framework
  import to skill code; `TestSingleSource` reads the syntax tree to catch
  both.

---

## 9. Known limits

- Two evaluations of one project and profile write the same prediction file:
  the names are keyed by `{project}_{profile}` as every other product of this
  skill is, so a second test set overwrites the first unless the project
  differs.
- CPU-only verification. `nn_pes_src/device.py` and the int64 index weights
  are unexercised on this environment and unproven on CUDA.
- The shipped fixtures prove the chain, not accuracy: reproduction of a
  training label after 30 epochs measured 0.11 to 0.19 Hartree, which is
  nowhere near a converged surface.
- A microcanonical run driven **through the framework** on such a fixture
  does not hold its total energy: over the same 50 fs window the excursion
  was 0.043 Hartree at 0.2 fs and 0.026 Hartree at 0.025 fs, and it stops
  shrinking with the step size. The plateau is a property of the fixture,
  not of the unit bridge, and the two were separated as follows. Hand
  integrating the same surface, probing every 400 steps, gives
  `reported gradient / dE/dR(Angstrom) = 0.529478` against the exact
  conversion factor 0.529177 at every probe point, so energy and gradient
  stay one surface along the whole path. What breaks is the geometry: the
  closest atom-atom approach reaches 0.363 Angstrom, because an
  undertrained network has no repulsive wall. Microcanonical conservation is
  therefore a meaningful gate only on a converged surface, and no shipped
  example asserts it on these fixtures.
- The slurm arm is verified only down to the generated job script, because
  `sbatch` does not exist on the development host. Real submission needs an
  HPC check.
- Coupling channels rest on a synthetic fixture and a caller-declared pair
  map, as described in section 6.
- Member ensembles beyond two are untested. `mult_nn` will accept the
  number and the role sections will expand to it, but nothing has measured
  whether a wider ensemble buys a better uncertainty estimate here, so the
  shipped cases stay at two.

## 10. The Keras-3 port: what was compared

The container was ported from the TensorFlow 2.3 / Keras 2 era code in
`legacy/PyRAI3MD/Machine_Learning/NNsMD`, which cannot run on Python 3.12 or
later because `tf.keras` is Keras 3 there, not a Keras 2 facade. The port
targets Keras 3 with the TensorFlow backend, keeps the directory layout of
the original, and was accepted against a golden reference produced by the
untouched legacy tree. This section records the evidence; the migration
harness itself (dual virtualenvs, golden-dump and comparison tools) was
retired with the working copy, and the legacy tree remains in `legacy/`.

### 10.1 Supported stack

| | Python | TensorFlow | Keras |
|---|---|---|---|
| primary | 3.11 | 2.20 | 3.15 |
| also verified | 3.12 | 2.20 | 3.15 |
| allowed, not measured | 3.11 or 3.12 | 2.21 | 3.15 (latest on PyPI) |
| golden reference only | 3.11 | 2.15 (Keras 2) | used only as the golden reference |

The requirement is `tensorflow>=2.20`. Every gate was executed on 2.20.0, so
2.21 and later 2.x install without being a declared pass; TensorFlow 2.21
depends on `keras>=3.12`, whose latest release is 3.15.1, the same Keras the
port was accepted on, so the layer the migration lives in does not move.
Python 3.12 was accepted by re-running every gate on a 3.12 environment, and
the paired 20-epoch fit showed a zero difference against the 3.11 run in
errors, per-epoch loss and final weights. Keras and h5py arrive as
TensorFlow dependencies and are deliberately not pinned separately, because
pinning them apart is how `tf.keras` silently changes identity.

### 10.2 Numerical equivalence against the original

The recorded comparison ran 20 epochs, batch 32, learning rate 1e-3,
`loss_weights [1, 10]`, a fixed 160/20/20 split of the 200 geometries in the
migration fixture, **no shuffling**, and **the same starting weights** (the
legacy pass dumped them before fitting, the target pass loaded them).

| model | test energy MAE | test gradient MAE | per epoch loss delta | final weight drift |
|---|---|---|---|---|
| `mlp_eg` | 0.8304 eV both | 0.6996 eV/Å both | 3.1e-05 | 4.8e-05 |
| `mlp_e` | 0.2752 eV both | - | 1.2e-07 | 6.5e-07 |
| `mlp_g2` | - | 0.8155 eV/Å both | 2.4e-07 | 1.9e-07 |
| `mlp_nac` | - | 6.39 eV/Å both | 2.4e-07 | 1.0e-06 |
| `mlp_nac2` | - | 8.63 eV/Å both | 1.2e-07 | 1.8e-07 |

Worst relative deviation over all 36 reported error components:
**5.45e-05**. Targets are trained in atomic units; the conversion uses the
constants of the legacy caller, 1 Ha = 27.211396132 eV and
1 Ha/Bohr = 51.4221 eV/Å.

All 36 components to four decimals. Legacy and target share each of these
values at that precision, so one number per cell is enough; the largest
relative gap is the `mlp_e` test energy MAE at `5.45e-05`, visible only in
the fifth decimal:

| model | split | energy MAE | energy RMSE | gradient MAE | gradient RMSE |
|---|---|---|---|---|---|
| `mlp_eg` | train | 0.8793 | 1.0809 | 0.6478 | 0.9491 |
| `mlp_eg` | val | 0.8118 | 0.9963 | 0.7329 | 1.0533 |
| `mlp_eg` | test | 0.8304 | 0.9569 | 0.6996 | 1.0506 |
| `mlp_e` | train | 0.2477 | 0.3280 | - | - |
| `mlp_e` | val | 0.3575 | 0.4676 | - | - |
| `mlp_e` | test | 0.2752 | 0.3671 | - | - |
| `mlp_g2` | train | - | - | 0.7305 | 1.0773 |
| `mlp_g2` | val | - | - | 0.8340 | 1.1975 |
| `mlp_g2` | test | - | - | 0.8155 | 1.2159 |
| `mlp_nac` | train | - | - | 6.0841 | 7.8197 |
| `mlp_nac` | val | - | - | 6.2257 | 8.2231 |
| `mlp_nac` | test | - | - | 6.3904 | 8.2827 |
| `mlp_nac2` | train | - | - | 8.5241 | 12.0291 |
| `mlp_nac2` | val | - | - | 8.3919 | 12.0826 |
| `mlp_nac2` | test | - | - | 8.6270 | 12.1565 |

Energy columns are eV and gradient columns eV/Å. Re-running the same fit on
Python 3.12 instead of 3.11 moved none of these numbers: the cross
interpreter comparison came back `0.000e+00` for errors, loss history and
final weights.

Two readings this table does not support: the coupling numbers come from a
synthetic field because the real fixture has `nnac = 0`, and that pair of
runs is a single-stage mean squared error fit, not the production two-stage
fit that recompiles with the phaseless loss. They show that both stacks
compute the same thing, not how accurate a coupling surface is.

### 10.3 The version wall

PyPI wheels, measured on this environment:

| TF | Python wheel | `tf.keras` actual identity | numpy constraint |
|---|---|---|---|
| 2.13.1 | 3.9-3.11 | Keras 2.13 | `>=1.22,<=1.24.3` |
| 2.15.1 | 3.9-3.11 | Keras 2.15 | `>=1.23.5,<2.0` |
| 2.16.2 | 3.9-3.12 | **Keras 3** | `<2.0` (3.11) / `>=1.26` (3.12) |
| 2.19.0 | 3.9-3.12 | Keras 3 | `>=1.26,<2.2` |
| 2.20.0 | 3.9-3.13 | Keras 3 | `>=1.26` (including numpy 2) |
| 2.21.0 | 3.10-3.13 | Keras 3 | `>=1.26` |

Python 3.12 and later necessarily land on Keras 3; `tf.keras` is no longer
Keras 2. That, not the version number itself, is why the old code stops
working.

### 10.4 Constructs that had to change

| construct | replacement |
|---|---|
| `import tensorflow.keras as ks` | `KERAS_BACKEND=tensorflow` + `import keras` |
| `K.batch_dot(x, y, axes=)` | explicit `tf.einsum` subscripts, aligned to the golden numbers |
| `K.int_shape` | `x.shape` |
| `K.{expand_dims,sum,shape,concatenate,sqrt,square,mean,min,log,exp,tile,less,arange,reshape}` | `keras.ops.*` |
| `K.set_value` / `K.get_value` | `variable.assign(value)` / `variable.numpy()` |
| `Metric.reset_states` override | inherit Keras 3 `reset_state()` |
| `models.save_model(model, dir)`, SavedModel `model.save(dir)` | `tf.saved_model.save` / `keras.export.ExportSavedModel` |
| `save_weights/load_weights("*.h5")` | `weights_v{i}.weights.h5` (no legacy reader) |
| `fit(validation_freq=)` | segmented `fit` + `evaluate` at each validation point |
| `Adam(lr=)`, `optimizer.lr` before build | `Adam(learning_rate=)`, build the optimizer before reading lr |
| `tf.test.is_built_with_gpu_support()` | `tf.config.list_physical_devices('GPU')` |
| `@tf.function` on instance methods / list of models | eager calls (not graphable in Keras 3) |
| `Dense(name=self.name+'_dense_0')` | let Keras 3 name layers (`self.name` contains `/`) |
| `ks.activations.deserialize(..., custom_objects=)` | `activations.get` + `@register_keras_serializable` |
| `tf.lite.TFLiteConverter` | removed |
| `np.int`, `np.float`, `np.Infinity`, `np.infty`, `np.Inf`, `np.NaN`, `np.float_` | `int`, `float`, `math.inf` |

TensorFlow operators (`gather(batch_dims)`, `repeat`, `linalg.cross`,
`math.acos/atan2/divide_no_nan`, `GradientTape.batch_jacobian`, `config.*`)
continue to work on the TF backend and were kept.

### 10.5 Keras-3 semantic drift

1. `validation_freq` has no counterpart. The segmented loop validates once
   per `epostep` epochs, appends the validation metrics to one history, and
   keeps the validation points alignable with the golden run.
2. History keys changed (`MeanAbsoluteError.name` defaults to `mae`). The
   compile call passes explicit `name=` values so the old literals
   (`mean_absolute_error`, `energy_mean_absolute_error`, `energy_lr`) still
   resolve; the training scripts used to swallow these KeyErrors in `except`.
3. `optimizer.lr` cannot be read or written before the optimizer is built;
   build it first or use an lr schedule object.
4. A custom `Loss` returning a scalar zero must return a per-sample zero
   vector; `NACphaselessLoss` keeps its numeric semantics and was checked
   axis by axis against the golden run.
5. Dict outputs plus `loss={'energy':…, 'force':…}` and `loss_weights` are
   matched more strictly; both the dict (training) and list (prediction)
   modes are covered by tests.
6. Static atom-index tables moved from non-trainable `int64` weights to
   plain attributes with lazily built `tf.constant` caches; the acceptance
   test is "export, reload, predict point by point identically".
7. Models still build explicitly, but the `precomputed_features` switch
   trick was removed.

### 10.6 Real defects fixed during the port

These are version-independent defects the port exposed; all are fixed in the
vendored copy.

| # | location | defect |
|---|---|---|
| 1 | `features.py:222` | `Dihedral.get_config` returned `self.angle_shape`; the class only has `dihed_shape`, so serialization raised |
| 2 | `normalize.py:84` | `get_config` wrote `"axs"` while `__init__` takes `axis`; deserialization dropped it |
| 3 | `mlp_nac2.py:217-237` | `get_config` omitted the required `nnac`; rebuilding always failed |
| 4 | `mlp_nac.py:26-54`, `hyper_mlp_nac.py:18` | `nnac` was forwarded into `ks.Model.__init__` as an unknown kwarg |
| 5 | `hyper_mlp_g2.py:55-78` | the `retraining` block lacked `loss_weights`; `training_mlp_g2.py:103` read it unconditionally |
| 6 | all five models' `get_config` | none called `super()`, dropping `name`/`trainable` |
| 7 | five training scripts | module-level argparse/logging/stream redirection before imports; not importable and not spawn-safe |
| 8 | `fit.py:49,56` | child processes used `python3` from PATH instead of `sys.executable` |
| 9 | `nn_pes.py:186-194` | `update()` had the if/elif branches reversed; list input was raised |
| 10 | `training_mlp_g2.py:276` | assigned a non-existent `output_as_dict` on `GradientModel2` |
| 11 | `training_mlp_e.py:220-247` | the plotting `try:` was commented out, so a plot error aborted the `fiterr_valid` write |
| 12 | training scripts | bare `except:` swallowed weight/hyper/scaler load failures and continued with `None` |
| 13 | `nn_pes.py:60` | version banner hard-coded "Tested for tf-gpu= 2.3"; now prints the measured python/tf/keras versions |
| 14 | `datasets/general.py:74-94` | `data_x`/`data_y` were extension-less pickles; now `np.savez_compressed` |

### 10.7 Known limits of the recorded evidence

- Only the CPU path was executed: the shipped wheels are CPU-only, so the GPU
  branches in `nn_pes_src/device.py` (visible devices, memory growth) and the
  int64 index weights are unverified here. Verify them on a Linux/CUDA host.
- `NACModel`'s precomputed mode contracts over the feature axis and then
  takes the diagonal of the two atom axes to reach
  `(batch, states, atoms, 3)`; that step is legacy behavior and is covered by
  the training smoke case, not to be removed while shortening the expression.
- `EarlyStopping(loss_monitor='val_loss')` requires validation data and says
  so with a KeyError, which is the intended loud failure.
