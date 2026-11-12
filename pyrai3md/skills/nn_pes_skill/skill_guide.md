# Skill Guide: nn_pes_skill

## Overview

Use nn_pes_skill to evaluate a trained neural network potential energy
surface: energies, nuclear gradients and non-adiabatic coupling vectors,
each reported as the mean over an ensemble of networks. The skill owns
the model container (the vendored NNsMD tree under `service/nn_pes/`) and
the training run that produces it, so a jobtype `train` followed by a
dynamics run needs nothing outside this package.

It provides:

- Single-point properties for one geometry: `energy` (Hartree),
  `gradient` (Hartree/Bohr), `nac` (Hartree/Bohr per coupling channel), and
  `soc` (one spin-orbit matrix element per declared pair, in cm-1).
- Training of the ensembles through one subprocess per ensemble member,
  started locally or submitted to slurm.

The skill is the framework's implementation of the `nn` theory label,
which was already reserved for machine-learned potentials
(`schema/validate.py`: `BUILTIN_THEORY_METHODS`).

## Setup Requirements

- Python 3.11 or newer. Keras 3, which TensorFlow 2.16+ ships as
  `tf.keras`, does not support older interpreters.
- TensorFlow, which is **optional** for the rest of the package:
  `pip install pyrai3md[nn]` or `pip install "tensorflow>=2.20"`.
- Without it the skill is still listed by discovery, and using it fails
  with `tensorflow is not installed` rather than looking absent.
- Molecular systems only. `supports_pbc` is false and a periodic request
  is rejected by the skill validation.

## Methods and profiles

`method` selects the network family; `profile` is `{method}_{capability}`,
and a profile that fits rather than serves ends in `_train`. Twenty-five
profiles are declared. A training run of a role the method name does not
carry spells both, so `mlp_e_soc_train` means "the energy family, fitted into
the spin-orbit folder"; the role decides the channel count and the family
decides the shape.

| method | what the network holds | profiles |
|---|---|---|
| `mlp_e` | scalars: energies, or spin-orbit values per pair | `_energy`, `_soc`, `_soc_train`, `_train` |
| `mlp_eg` | energies with gradients by differentiation | `_energy`, `_gradient`, `_train` |
| `mlp_g2` | gradients directly | `_gradient`, `_train` |
| `mlp_nac` | couplings from a virtual-potential diagonal | `_nac`, `_train` |
| `mlp_nac2` | couplings by propagated derivative | `_nac`, `_train` |
| `eg_nac`, `eg_nac2` | energy ensemble plus coupling ensemble, and a spin-orbit one on request | `_energy`, `_gradient`, `_nac`, `_soc`, `_soc_train`, `_train` |

A dynamics run under `control.driver = main` asks jobtype `md`, which
resolves to the composite `_nac` profile: one call returns energies,
gradients and couplings, reading both ensembles from one model folder.

## Unit contract

Measured on a trained ensemble, not assumed:

- Coordinates enter in Angstrom. `modeldir` folders train on
  Angstrom geometries, and the network's gradient output is the
  derivative with respect to the coordinate unit it was fed. The check is
  a central difference of the energy the same call reports: ratio
  0.99998 (p10 0.99906, p90 1.00094).
- Energies are Hartree.
- The boundary multiplies the gradient and the couplings by 0.52917721090299996
  to hand over Hartree/Bohr, which is what the integrator kicks with
  (`workflow/integrator/verlet.py`). `manifest.yaml` states this in its
  `units:` block.
- A training file is read as **Hartree energies with Hartree/Bohr gradients
  and couplings**, which is the framework's own convention and what the
  engine backends report. There is no unit keyword. The trainer converts
  the derivative onto the Angstrom axis the network is fed
  (`service/trainer_data.py:match_labels`), which is the one rule the pair
  has to satisfy: the gradient must be the derivative of the energy the same
  surface reports. What a served prediction returns is fixed by that axis
  choice, and the boundary takes it back to Hartree/Bohr.
- A unit switch was removed rather than kept, because its failure mode is
  silent: declaring a scheme the file does not use trains a network whose
  reported gradient is the derivative of its own energy by a factor of
  1.8897, and only an energy conservation run notices, hundreds of steps
  later. A file that carries a `unit` block naming another scheme is
  refused at read time instead.
- Training **metrics** are printed in eV and eV/Angstrom. The fit stores its
  validation error in the scaled units of its own normalizer, where a number
  means nothing to a reader, so the summary multiplies the spread back out
  of the scaler the folder was fitted with
  (`service/trainer.py:physical_errors`). Spin-orbit values would be printed
  in cm-1; this container trains no SOC model.

## Training

`jobtype: train` (or `profile: mlp_eg_train` and friends) drives a
single-shot training run. The framework routes it exactly like any other
skill jobtype: one calculator call, and the skill decides from the request
what to do. A request with no profile is resolved by the skill, and the
resolution does not depend on `control.driver`, because a training run
propagates nothing.
The order of operations is fixed and must not be reordered:

1. Write hyperparameters and pickled data for each ensemble.
2. Start one subprocess per ensemble member, all of them before waiting.
   This is the shape of the original container, and it is how a member
   lands on its own GPU index.
3. `local` starts the command directly; `slurm` writes the same command
   into a job script and submits `sbatch -W`, one independent job per
   member. The arms differ only in the launcher in front of it.
4. After every member exits with status 0, reload the folder and read
   back the per-member fit error.

Failure semantics: a member that dies makes the run raise with the exit
code and the `fitlog_*.txt` path, and the folder is left marked by the
absence of `fiterr_valid_v*.npy`. Serving refuses such a folder, because
the parent writes freshly initialized weights **before** the fit starts,
so the files alone do not prove a finished training run.

Outputs land in `modeldir/{ensemble}/`, one folder per role:
`energy_grad` for the energy families and `nac` for the coupling family.
Each holds `hyper_v{i}.json`,
`weights_v{i}.weights.h5`, `scaler_v{i}.json`, `fiterr_{train,valid}_v{i}.npy`,
`fitlog_{i}.txt`, plus `{project}_train.summary.txt` in the calculation
directory with the per-member MAE and wall clock.

The run is also reported in the log, in the same section grammar a property
step uses:

```
  &nn training
-------------------------------------------------------------------------------
  Number of atoms:  18        Number of state:  2
  Number of NAC:    1         Number of SOC:    0
  Training arm:     local     Wall clock:       22.5 s
  Summary file:     /run/NNPES_train.summary.txt
-------------------------------------------------------------------------------

  &nn validation mean absolute error
-------------------------------------------------------------------------------
                              energy      gradient           nac           soc
                                  eV   eV/angstrom
  energy_grad_v0          0.06857300    0.00867100            --            --
  energy_grad_v1          0.09154200    0.01098500            --            --
-------------------------------------------------------------------------------
```

One row per ensemble member, the units the skill reports the fit in, and a
channel the folder does not produce printed as `--`. A zero would claim a
perfect fit for a network that never had that channel. The same record
reaches the framework as `train_errors`, `train_metric_units`,
`train_wall_seconds`, `train_arm` and `train_summary`, and the run log
prints it with this layout; a training step prints no property sections,
because it observed no geometry.

## Serving

The ensembles are loaded once, at backend initialization. Each dynamics
step then runs one compiled forward pass per member and reads nothing
from disk. A `jobtype: prediction` (alias `predict`) request is the same
warm start with no property token attached: it resolves to the channel the
family was fitted for, `mlp_eg` to `mlp_eg_gradient`, `mlp_e` to
`mlp_e_energy`, a coupling family to its `_nac` profile.

Measured on a CPU-only test environment, an 18-atom two-state ensemble with `mult_nn = 2`:

| path | per step |
|---|---|
| eager container call, re-traced each step | 1274 ms |
| per-member `tf.function` with an input signature | 1.43 ms |
| exported SavedModel `.serve()` | 1.34 ms |

The compiled path is what the skill uses (about 900 times faster, with
the gradient agreeing with the eager path to 1e-08). The SavedModel path
was measured and **not** adopted: it needs an export step at training
time, it keeps its variables alive only while the restored root object is
held, and it buys nothing over the compiled path here.

## Output channels

Three ensembles live under `modeldir`, one per role, and each reports a
different kind of number:

| folder | hyperparameter block | network family | channels | unit |
|---|---|---|---|---|
| `energy_grad` | `&eg` | `mlp_e`, `mlp_eg`, `mlp_g2` | one per electronic state | Hartree, Hartree/Bohr |
| `nac` | `&nac` | `mlp_nac`, `mlp_nac2` | one per declared coupling pair | Hartree/Bohr |
| `soc` | `&soc` | `mlp_e` | one per declared spin-orbit pair | cm-1 |

A spin-orbit value is a scalar that depends on geometry rather than a
derivative of anything, so the request is answered by the energy family with
its derivative head off (`energy_only`) and one output per pair: `soc`
arrives as `(npair, 1)` in cm-1 and never carries a gradient channel with it.
A `soc` jobtype token resolves to `mlp_e_soc`, and a dynamics run that needs
all three asks the composite `eg_nac_soc`.

## Coupling channels

A trained network carries a channel count, never a record of which state
pair each channel means: that is a property of the training data. The
framework owns the declaration. Its `&MOLECULE` spells the pairs as
`coupling 1 2, 1 3, 2 3` in 1-based state indices and splits them on the
state spins, and what reaches the skill is the resolved pair of that split:
`nac_pair_map` and `soc_pair_map`, in 0-based indices, with `nnac` and `nsoc`
counting them.

The skill reads those maps and nothing else. Taking `coupling` directly
would put non-adiabatic pairs into the spin-orbit set and shift every state
index by one. One declared pair is one channel, so nothing is added to
complete a ladder and no all-pairs traversal invents a width the caller did
not ask for. A count without a map is honoured, but only in the order the
model holds its channels; asking for a specific pair then raises rather than
guessing which channel means it.

Three agreements are enforced instead of assumed: the training file's channel
count equals the declared one; `mlp_nac`, whose virtual-potential scheme
builds one output per electronic state, is only accepted for a declaration of
exactly `nstate` pairs (wider sets need `mlp_nac2`); and at serving the loaded
ensemble's width is compared with the declaration, so a folder trained on one
pair set cannot be reported under another's state names. A requested pair
outside the map raises, and channels are never zero-filled, because a
silently missing coupling changes a surface hopping trajectory.

## Hyperparameter sections and the member axis

All ML methods share one block, `&nn`: where the models live, where the
data comes from, whether the run fits or refits, the ensemble size, and the
device of each training process. Units are not in it, because they are not a
choice (see the unit contract above). Each ensemble then gets its own block of
hyperparameters, `&eg`, `&nac` and `&soc`, named as the original project
named them, and the keywords inside keep their original spelling.

The two members of an ensemble take their parameters from **one** block.
The separator between them is `;`, and the member axis is always the
outer one:

```
&EG
depth          3               # one value: both members
nn_size        100;60          # member 0 is 100 wide, member 1 is 60
loss_weights   1 1;1 10        # a list value, one per member
angle_index    1 2 3;4 5 6     # an index table, one per member
```

In a yaml the same block nests under the skill section, and a `;` text is
interchangeable with an outer list:

```yaml
nn:
  eg:
    depth: 3
    nn_size: '100;60'
    loss_weights: [[1, 1], [1, 10]]
```

`mult_nn` decides how many groups are read. One group serves every
member; more groups than members keep the leading ones; fewer groups than
members, when more than one was written, raises, because silently
truncating an ensemble the caller sized differently would train a
different model than the one asked for. `nn_size` is the width of every
hidden layer, so per member it stays one number — a network shape is not
a list of layers here.

`schema/hyper.py` owns that vocabulary: each keyword says which section of
the container's dictionary it fills, what shape it must have, and whether
it is a number, a boolean, an index table, a regularizer selector or a
name. Unknown keywords are refused with the list of accepted ones, before
a training folder is created. A keyword written once is mirrored into the
container's `retraining` section as well, because `train_mode` decides
which of the two the fit reads; a slot the role does not carry is skipped
rather than invented.

## Ensemble spread and the prediction-error channel

`mult_nn >= 2` gives the standard deviation between the members of the
ensemble, with `ddof=1` — for two members that is their difference divided
by the square root of two. It is the only error bar a neural network
potential has, so the skill reports it: one number per **computed**
property, the largest spread over that property's components, in the
property's own unit.

| property | deviation key | unit |
|---|---|---|
| `energy` | `err_energy` | Hartree |
| `gradient` | `err_grad` | Hartree/Bohr |
| `nac` | `err_nac` | Hartree/Bohr |
| `soc` | `err_soc` | cm-1 |

A maximum, not a mean, because the framework compares it against a
threshold — `&adaptive maxenergy/maxgrad/maxnac/maxsoc` — and a guard has to
be on the worst component of the step, which is what the original project
did with `np.amax`. The property arrays the framework aggregates are
unaffected: only these scalars travel alongside them. A property the profile
did not compute gets no key, and one member gives 0.0 rather than silence,
because with a single network there is no disagreement to measure.

Two records are written per step into the calculation directory:

- `{project}_{profile}.log` — the skill's own log: the states served, each
  property with its unit, and each deviation with the unit and the statement
  that it is a member-to-member standard deviation.
- `{project}_{profile}.uncertainty.txt` — the full spread per step, so the
  reported scalars can be re-derived rather than trusted.

The framework's report then prints what the backend returned, in its
`&prediction error` block, with the threshold and whether it was exceeded
(`utilities/print_results.py:_format_prediction_error`). The key names are
a contract across the boundary, and
`qualification/verification/test_deviation.py::TestProducerChannelAgreement`
holds both sides to it: a renamed key would otherwise read as a perfectly
agreed surface.

## Configuration keys

Shared block (`&nn`): `modeldir`, `train_data` (a file name),
`train_dataset` (the same data as an inline mapping), `train_mode`,
`mult_nn`, `gpu`, `eg`, `nac`, `soc`. Exactly one of `train_data` /
`train_dataset` is set for a training request; both reach the same
reader. The inline form is what the framework writes when it expands
`&train permutation_map` before the backend is initialized, so a map is
never read here.

`gpu` is a device list: one index per ensemble member, separated by `;`,
and one value broadcasts to every member. It is not the original project's
mode number (`gpu 1` used to mean "use the first GPU"); here it means device
1. `-1` asks for no device.

State fields, owned by the framework's `&MOLECULE` and read from there:
`nstate`, `coupling` (the channel declaration), `target_states`,
`target_nac_pair`.

Serving needs `modeldir` pointing at a folder a completed `train` run
wrote. Training needs `modeldir` plus exactly one data source,
`train_data` or `train_dataset`, and reads the role section of the
ensemble it fits.

Environment: `execution_mode` (governs the training arm; serving is always
in-process), `scratch_path`, `thread`, `train_thread`, `predict_thread`,
`keep_tmp`, `slurm_*`, `nn_project`. `train_thread` and `predict_thread`
are independent budgets; empty/0 inherits `thread`. Training gives each
member subprocess its own `train_thread`; serving sets TensorFlow's
intra/inter-op pools to `predict_thread`. The sample/model-generation
keywords and the data source are documented above.

`keep_tmp` clears the calculation directory when the backend is gone, not
after each step: the directory holds the step log and the uncertainty dump,
which are appended across a run. The trained ensemble lives in `modeldir`,
outside that directory, and is not affected. `keep_tmp: 0` does remove
`{project}_train.summary.txt`, whose path the training record returns, so a
caller that wants the file asks for `keep_tmp: 1` (#307).


## Limits

- GPU code paths are present but unverified: the development host is a
  CPU-only wheel. Nothing here has been measured on a CUDA device,
  including the int64 index weights.
- 30 epochs on 24 geometries is a demonstration that the chain works. The
  measured reproduction of a training label was 0.11 to 0.19 Hartree, so
  the shipped fixtures are not converged surfaces.
- The two coupling networks rest on a synthetic coupling fixture; the real
  regression data carries `nnac = 0`.
- The spin-orbit channel rests on a synthetic fixture: two smooth functions
  of two different interatomic distances, in cm-1. A trained ensemble
  reproduces its labels to within 60 cm-1 of a roughly 70 and 150 cm-1 span
  (`tests/test_train_predict.py::TestSpinOrbitTraining`), which shows the
  routing and the scalar shape but is not a converged SOC surface, and no
  engine-calculated spin-orbit data has been put through it.
- The vendored tree is the live code. The retired standalone migration-era
  project is not imported from here.
