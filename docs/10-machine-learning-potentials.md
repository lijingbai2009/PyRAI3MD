---
title: "10. Machine learning potentials"
nav_order: 10
---

This chapter covers the machine-learning side: the label file, training and prediction, the hyperparameter scan, and the adaptive loop that uses a reference engine to improve a potential where it is weakest.

## 10.1. Training data

A training run starts from one label set: a JSON mapping of geometries and their computed values. train_data names the file and train_dataset hands the same mapping inline; both reach one reader and pass the same checks.

| Key | Shape | When it is needed |
| :--- | :--- | :--- |
| xyz | (nframe, natom, 3) | always; rows carry the symbol or not |
| symbols | (nframe, natom) | when the xyz rows carry no symbol |
| energy | (nframe, nstate) | the energy role |
| grad | (nframe, nstate, natom, 3) | the mlp_eg and mlp_g2 families |
| nac | (nframe, nnac, natom, 3) | the coupling roles |
| soc | (nframe, nsoc) | the soc role |
| transition_dipole | (nframe, npair, 3) | optional; carried, not fitted |
| osc_strength | (nframe, npair) | optional; carried, not fitted |
| natom, nstate, nnac, nsoc, npair | counts | optional; derived from the arrays |
| unit | channel to unit spelling | optional; checked |

A single-state file may omit the state axis of grad and a single-pair file the channel axis of nac. The canonical file also carries a format name, a version and any per-frame field the run produced (charge, cell, pbc).

```json
{
  "format": "pyrai3md-training-data", "version": 1,
  "natom": 3, "nstate": 2, "nnac": 0, "nsoc": 0, "npair": 0,
  "xyz": [[[0.000000, 0.000000, 0.000000],
           [0.000000, 0.000000, 1.889726],
           [1.700000, 0.000000, -0.500000]], "..."],
  "symbols": [["O", "H", "H"], "..."],
  "energy": [[-76.0201000000, -75.9011000000], "..."],
  "grad": [[[[0.0, 0.0, 1.204e-03], [0.0, 0.0, -6.02e-04],
             [0.0, 0.0, -6.02e-04]], "..."], "..."],
  "unit": {"energy": "hartree", "gradient": "hartree/bohr"}
}
```

Energies are Hartree, derivatives Hartree per Bohr, spin-orbit couplings cm-1. The reader puts every derivative on the Angstrom axis the network is fed, once, so energy and gradient describe one surface; a spin-orbit value is not a derivative and gets no length factor. There is no unit keyword, because a switch could state a scheme the numbers do not follow; the optional unit block is checked instead.

Labels come from finished calculations. A jobtype read run starts no engine: it parses the read paths with the single theory skill and returns the frames. [Section 10.6](#106-data-tools) wraps it with extract, shuffle, split, merge and select. A channel is written only when every selected frame carries it.

## 10.2. Training

A training run is one call of the nn_pes skill under a profile ending in _train: the skill writes the data and hyperparameters, runs one fit per member, waits and reads the result back. The framework makes the call once and states whether the record arrived; no geometry is evaluated, so the log carries a training report only.

modeldir names the durable folder; one subfolder per role lands inside it, and each role reads its hyperparameters from a section named after the network family.

| Role | Folder under modeldir | Hyperparameter section |
| :--- | :--- | :--- |
| energy and gradient | energy_grad | eg |
| nonadiabatic couplings | nac | nac |
| spin-orbit couplings | soc | soc |

mult_nn is the ensemble size and the axis every hyperparameter is written against: one value for the ensemble or one per member, semicolon in the classic text and outer list in YAML, and fewer groups than members is refused. gl_seed seeds the fit, each member receiving gl_seed plus its index, because a child inherits no random state.

train_mode selects the section the fit reads: training fits a new folder, retraining continues from the weights on disk and cannot change the architecture. A member folder holds hyper_v&lt;i&gt;.json, scaler_v&lt;i&gt;.json, weights_v&lt;i&gt;.weights.h5, the fiterr arrays, fitlog_&lt;i&gt;.txt and fit_stats/.

The completion certificate is fiterr_valid_v&lt;i&gt;.npy, written last: initialised weights are written before the fit starts, so without it a folder loads and predicts noise. The report multiplies the stored error by the folder's scaler spread and prints eV, eV per Angstrom and cm-1.

> **Note.** A fit has no numeric convergence verdict: the framework judges whether the record arrived, and whether the error suits a dynamics run is the caller's judgement.

## 10.3. Prediction

A prediction run serves a trained folder over a file of geometries that carry reference values. pred_data names that file, and the skill repeats its serving call over every frame; an absent channel is absent rather than zero.

The model folder computes every number. Four statistics are reported per channel: mae, rmse, the largest absolute deviation, and spread_max, the largest member disagreement. spread_max is a disagreement, not an error, and no relative figure is reported, because a derivative component can be zero.

Two raw products are written in the run directory: {project}_{profile}.{channel}.pred.txt, four column blocks in one unit each (reference, prediction, absolute deviation, spread), and {project}_{profile}.frames.txt. Printed values are eV, eV per Angstrom and cm-1.

nn_predict_mode selects the answering member: zero is the member mean, k a one-based member, and the others are mirrored into alternative channels. Fewer than two members refuses a non-zero mode.

## 10.4. Hyperparameter search

A grid search varies the training of one model family over a declared grid and ranks what came back. The skill says which keywords may be varied and what a candidate means; the run says which values to try. The grid maps keyword to candidate list and the scan is their Cartesian product, with no start, stop and step generator.

Each point is stamped on a copy of the parent state with the child jobtype train and an absolute model directory. Its key joins its values in keyword order by a tilde, as in depth=1~nn_size=24; its folder is NN-&lt;title&gt;-&lt;key&gt;.

| Keyword | Role in the scan |
| :--- | :--- |
| grid | keyword to candidate list; an empty grid is refused |
| retrieve | read the folders a previous scan left and rank them; submit nothing |
| sort_by | channel the ranking reads; default the profile's first channel |
| sort_stat | max ranks on the worst member, mean on the average |
| sort_order | ascending or descending |
| grid_root | root of the scan directories |
| max_grid | refuse a scan larger than this; zero states no limit |

A complete folder is skipped, read back and marked, so a rerun fills in what is missing; a scan planned under retraining is refused, because a continuation cannot apply the architecture the grid varies. summary.csv holds the ranking and crashed.txt the points with no record.

The scan ranks and does not judge: no threshold keyword exists, because a validation error is not a convergence criterion. One skill per scan.

## 10.5. Adaptive sampling

Adaptive sampling is a round-based loop that retrains the potential on what it got wrong. Iteration 1 is the external start: an external data set and network run the dynamics and the screens are computed, with no reference and no training. Each later round labels what the previous one selected, appends the labels, fits a network and runs the dynamics with it.

```text
iteration 1   external data + external network -> MD -> screen
               -> iter-1/selected_iter-1.xyz
 
iteration k   reference labels on iter-(k-1) selection
               -> data_iter-<k>_<nframe>.json
               -> train NN-<title>-iter-<k>
               -> MD with that network -> screen
               -> iter-<k>/selected_iter-<k>.xyz
```

The terminal round still labels, trains and runs dynamics, and its own selection has no consumer; maxiter counts the rounds. sample_energy and sample_grad are the baseline and stay on, while sample_nac, sample_soc, sample_dipole, sample_transition_dipole and sample_oscillator_strength are switches; [Section 5.7](05-07-adaptive.html) lists their defaults. A channel that is off has no row in the sampling table.

Screening reads each outcome from its history over a window of record_step steps. The deviation is the ensemble standard deviation, read over states for the energy and over states, atoms and components for the gradient. A frame above a max threshold is a trigger, one above the matching min threshold joins the pool, and candidates are taken newest first up to maxsample per trajectory. Every row is kept in status_iter-&lt;k&gt;.json.

Two checks promote a frame. check_population collects the structure a dynamics run stopped on when the population went ill; the parent only decides whether it is worth labelling. check_energy_order promotes a frame in which a same-spin level lies below an earlier one by more than the tolerance, the signature of a crossing the dynamics missed.

maxmae arms the training-data quality gate. When the fit's validation energy error is above the ceiling, the frames this round appended are tested against the frames held before it, one state at a time, with a robust z score. A flagged frame is removed from a candidate set, a fresh network is fitted, and the removal is kept only when the error strictly falls; otherwise the run stops.

Four analysis switches read the round without changing it and are closed by default: analyze_pes for local curvature and a low-rank gradient reconstruction, analyze_gap for gap statistics, analyze_trace for scalars along a trajectory, and analyze_coverage for covered descriptor space. Two exploration switches add a bias: explore_udd builds a potential or force bias from the member spread (udd_mode, udd_amplitude, udd_width, udd_start_iteration), and explore_gap pushes a chosen pair apart (gap_method, gap_pair, gap_threshold), its antisymmetric term leaving the pair mean unchanged.

Products land under &lt;title&gt;.adaptive/: md/traj_&lt;idx&gt;/ and selected_iter-1.xyz in iteration 1, then the unpruned data snapshot, the model folder with its log, reference/ref_&lt;idx&gt;/, new-samples.json, analysis_iter-&lt;k&gt;.json and prune/ for a gate retrain. [Section 5.7](05-07-adaptive.html) lists the keywords and [Section 13.5](13-examples-and-tutorials.html#135-tutorial-adaptive-sampling) walks a run through its rounds.

## 10.6. Data tools

tools/training_data.py turns finished calculations into a maintained label set: extraction is the framework feature jobtype read, and the tool adds the filtering, the statistics and the file.

| Command | What it does | Output |
| :--- | :--- | :--- |
| extract | runs jobtype read over a read input and keeps the requested channels | data.json |
| shuffle | permutes the frames with a seed; an optional group count shuffles a prefix separately | data-shuffled.json |
| split | shuffles, then writes a prefix and the remainder | data-set-1.json, data-set-2.json |
| merge | shuffles the first set and appends the others | data-merged.json |
| select | keeps one-based indices and inclusive ranges | data-selected.json |
| describe | prints counts, shapes and units | no file |

```bash
python tools/training_data.py extract read.yaml -o egs.json
python tools/training_data.py shuffle egs.json -s 1234
python tools/training_data.py split   egs.json -p 200 -r 0
python tools/training_data.py merge   egs.json extra.json
python tools/training_data.py select  egs.json --index 1-100,150
```
