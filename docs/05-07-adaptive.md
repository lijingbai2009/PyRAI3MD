---
title: "5.7. ADAPTIVE"
nav_order: 7
parent: "5. Keyword sections"
---

The adaptive section steers the iterative sampling loop of the adaptive jobtype. It fixes the iteration budget, the screening window and its thresholds, the property channels the labels carry, the analysis and exploration options, and the dispatch flags of the three stages. The classic spelling of the section is &amp;ADAPTIVE; in YAML and agent JSON it is the adaptive mapping.

Iteration 1 is the external starting point: the training data named in the input and the pretrained ensemble are taken as given, the MD trajectories run on that network, and their retained history is screened. Every iteration k of 2 and above labels the structures of iteration k-1 with the reference queue, appends those labels to the dataset, trains the iteration k network from the appended dataset, runs the MD trajectories on the new network, and screens them again. The loop stops when a round selects nothing, when the budget set by maxiter is spent, when a saturation test fires, or when a missed data check ends the round.

The parent workflow never evaluates the potential: it holds no backend, and the reference, training and MD stages are child runtimes materialized from the parent state, each with its own calculator. [Section 10](10-machine-learning-potentials.html) describes the loop and its products, and [Section 11](11-parallel-and-remote-execution.html) describes the dispatch layer the stage flags belong to.

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&ADAPTIVE**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| maxiter | 1 | Sets the number of iterations. A value of 1 runs the external start only, and 0 ends the run before the first trajectory with the reason maxiter is… |
| maxsample | 1 | Caps the number of structures one trajectory contributes to the reference calculation in one iteration. The cap follows the distance filter in the… |
| load ‡ | 1 | Chooses where iteration 1 takes its data and network. Only 1 is accepted: the run starts from the training data and the pretrained ensemble named in… |
| transfer ‡ | 0 | Requests transfer learning from the previous network. Only 0 is accepted; transfer learning is not ported in this build and any other value stops the… |
| remote_train | 0 | Sets the route of the training child. 0 runs the fit locally; a non-zero value submits it through the remote route. Each iteration trains one… |
| keep_md | 1 | Keeps the round MD directory. 1 keeps iter-&lt;k&gt;/md after the round finishes, and 0 deletes it, which leaves the screening record as the only trace of… |
| record_step | 10 | Sets the screening window: the last N retained history frames of every trajectory are screened each iteration. 0 is normalized to 10, and the MD… |
| check_population | 1 | Collects the structure the MD loop stopped on when its own population guard ended the trajectory with stop code 1. 1 collects that frame, 0 leaves it… |
| check_energy_order | 1 | Turns on the same-spin level-order check. With 1 a frame whose same-spin level lies below an earlier one by more than gap_tolerance is a trigger and… |
| gap_tolerance | 1e-06 | Sets the tolerance of the same-spin level-order check, in Hartree. An inversion larger than this value flags the frame, and a smaller one is treated… |
| sample_energy ‡ | 1 | Asks for energy labels and energy deviations. Only 1 is accepted: the energy is half of the training baseline and validation refuses 0. |
| sample_grad ‡ | 1 | Asks for gradient labels and gradient deviations. Only 1 is accepted: the gradient is half of the training baseline and validation refuses 0. |
| sample_nac | 0 | Asks for NAC labels and derivative-coupling deviations, labelled in Hartree/Bohr. 0 and 1 are accepted; turning it on switches &MD nactype to nac, a… |
| sample_soc | 0 | Asks for SOC labels, labelled in cm-1. 0 and 1 are accepted; turning it on switches &MD dosoc on, and the reference queue must plan the soc… |
| sample_dipole ‡ | 0 | Asks for dipole labels. Only 0 is accepted in this build, which labels energy, gradient, NAC and SOC only: an input that turns this key or &MD… |
| sample_transition_dipole ‡ | 0 | Asks for transition-dipole labels. Only 0 is accepted in this build: the label extractor stops at energy, gradient, NAC and SOC, so an input that… |
| sample_oscillator_strength ‡ | 0 | Asks for oscillator-strength labels. Only 0 is accepted in this build, for the same reason as sample_transition_dipole: the channel is not labelled… |
| predict_mode | 0 | Chooses the ensemble member that serves the MD trajectories. 0 returns the member mean; k above 0 returns the 1-based member k as the primary… |
| maxmae | 0 | Sets the validation energy MAE ceiling of the training-data quality gate, in eV. 0 disables the gate, so every fit stands. |
| maxenergy | 0.05 | Sets the energy deviation that triggers a frame, in Hartree, and is stamped on every MD child of the round as md_maxenergy, replacing the value &MD… |
| minenergy | 0.02 | Sets the energy deviation that puts a frame in the candidate pool, in Hartree. A frame above it is selected when the trajectory triggered anywhere in… |
| dynenergy † | 0.1 | Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. |
| inienergy † | 0.3 | Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. |
| fwdenergy † | 1 | Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. |
| bckenergy † | 1 | Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. |
| maxgrad | 0.15 | Sets the gradient deviation that triggers a frame, in Hartree/Bohr, and is stamped on every MD child of the round as md_maxgrad. Validation requires… |
| mingrad | 0.06 | Sets the gradient deviation that puts a frame in the candidate pool, in Hartree/Bohr. Validation requires it not to exceed maxgrad. |
| dyngrad † | 0.1 | Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. |
| inigrad † | 0.3 | Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. |
| fwdgrad † | 1 | Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. |
| bckgrad † | 1 | Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. |
| maxnac | 0.15 | Sets the NAC deviation that triggers a frame, in Hartree/Bohr, and is stamped on every MD child of the round as md_maxnac. Validation requires it to… |
| minnac | 0.06 | Sets the NAC deviation that puts a frame in the candidate pool, in Hartree/Bohr. Validation requires it not to exceed maxnac. |
| dynnac † | 0.1 | Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. |
| ininac † | 0.3 | Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. |
| fwdnac † | 1 | Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. |
| bcknac † | 1 | Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. |
| maxsoc | 50 | Sets the SOC deviation that triggers a frame, in cm-1, and is stamped on every MD child of the round as md_maxsoc. Validation requires it to be at… |
| minsoc | 20 | Sets the SOC deviation that puts a frame in the candidate pool, in cm-1. Validation requires it not to exceed maxsoc. |
| dynsoc † | 0.1 | Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. |
| inisoc † | 0.3 | Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. |
| fwdsoc † | 1 | Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. |
| bcksoc † | 1 | Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. |
| refine ‡ | 0 | Requests the legacy geometry refinement of the selected structures. It is not ported: only 0 is accepted, and any other value stops the run at… |
| refine_num † | 4 | Belongs to the legacy refinement family. The value is stored on the section and no adaptive stage reads it. |
| refine_start † | 0 | Belongs to the legacy refinement family. The value is stored on the section and no adaptive stage reads it. |
| refine_end † | 200 | Belongs to the legacy refinement family. The value is stored on the section and no adaptive stage reads it. |
| refine_gap † | 0.3 | Belongs to the legacy refinement family, where it named the gap criterion of a refinement window. The value is stored on the section and no adaptive… |
| explore_udd | 0 | Turns on UDD exploration, which builds a bias from the covariance of the ensemble members. It needs predict_mode above 0, at least two members and a… |
| explore_gap | 0 | Turns on gap-driven exploration, which needs at least two electronic states and marks its trajectories as exploration, outside the production… |
| gap_threshold | 0.03 | Sets the gap, in Hartree, at which gap driving starts. At or below this value the correction is exactly zero; above it the cv potential grows and… |
| udd_mode | force | Chooses what the UDD bias changes: force modifies the nuclear force only, potential adds the bias energy to the active state as well. The gradient… |
| udd_amplitude | 0 | Sets the height of the UDD bias, in Hartree. 0 selects the automatic value, 5.0e-3 Hartree. |
| udd_width | 0 | Sets the width of the UDD bias, in Hartree. 0 selects the automatic value, one tenth of the ensemble standard deviation. |
| udd_start_iteration | 1 | Delays the UDD bias to the given iteration, counted from 1. Earlier rounds run unbiased and report no bias work. |
| gap_method | cv | Chooses the gap-driving method. cv splits the selected pair with an antisymmetric potential that leaves the pair mean unchanged; paper replaces the… |
| gap_pair | [] | Names the state pair the gap bias acts on, as two indices such as 0 1. An empty value takes the pair from the state ladder, and a list that is… |
| gap_height | 0 | Sets the height of the saturating cv gap potential, in Hartree. 0 selects the automatic value, 1.0e-3 Hartree. |
| gap_width | 0 | Sets the decay width of the cv gap potential, in Hartree. 0 selects the automatic value, one third of the gap being driven. |
| analyze_pes | 0 | Turns on the QM-side PES analysis of the dataset. It reads the reference labels already in the data and never an NN frame, and reports point-pair… |
| analyze_gap | off | Selects the scope of the gap statistics: off, s0s1, adjacent or all. The analysis reads the unbiased predicted state energies and reports the minimum… |
| analyze_trace | 0 | Collects the scalar trajectory trace through the MD child: maximum force, population, energy drift and the minimum inter-atomic distance. A short… |
| analyze_coverage | 0 | Turns on the descriptor coverage report and the saturation test. Coverage is measured over a bounded sample of the dataset, at most 200 frames and… |
| analyze_window | 3 | Sets the number of rounds in the moving average of the saturation test, at least 1. The test needs one round more of validation MAE than the window. |
| analyze_local_tolerance | 0 | Sets the per-round validation-MAE improvement, in eV, that still counts as progress. An improvement above it keeps the round unsaturated even when… |
| analyze_global_tolerance | 0 | Sets the validation-MAE improvement, in eV, below which the fit counts as saturated. With both tolerances at 0 the saturation stop is disabled. |
| dynsample ‡ | 0 | Requests the dynamic threshold update of the legacy loop. It is not ported: only 0 is accepted, and any other value stops the run at start-up with a… |
| maxdiscard † | 0 | Belongs to the legacy discard budget. The value is stored on the section and no adaptive stage reads it. |
| pop_step † | 200 | Belongs to the legacy population-report cadence. The value is stored on the section and no adaptive stage reads it. |
| multitask | None | Sets the generic fan-out flag of the adaptive node. It reaches the MD and reference stages as the fallback of md_multitask and reference_multitask… |
| remotetask | None | Sets the generic remote flag of the adaptive node. It reaches the MD stage only, as the fallback of md_remotetask: training follows remote_train and… |
| multipool | 1 | Sets the size of the adaptive node own worker pool, at least 1. Each stage batch runs under its own pool, md_multipool for the trajectories and… |
| md_multitask | None | Overrides the fan-out flag of the MD stage. Unset falls back to multitask and then to the workflow default, which is serial, and one trajectory runs… |
| md_remotetask | None | Overrides the remote flag of the MD stage. Unset falls back to remotetask; a remote MD child is serialized and submitted by the framework, with the… |
| md_multipool | None | Sets the worker pool of the MD stage, at least 1, with one trajectory per worker. Unset means 1, and the generic multipool does not raise it. |
| reference_multitask | None | Overrides the fan-out flag of the reference stage. Unset falls back to multitask and then to the workflow default, which is serial, and the stage… |
| reference_remotetask | None | Switches the reference skill itself to its slurm execution mode. The reference child stays local and the skill writes its own submission script, so… |
| reference_multipool | None | Sets the worker pool of the reference stage, at least 1. Unset means 1, and the generic multipool does not raise it. |

Full descriptions for all available keywords are summarized below.

**maxiter** - Sets the number of iterations. A value of 1 runs the external start only, and 0 ends the run before the first trajectory with the reason maxiter is 0. The loop also stops with maxiter exhausted when a round selects structures and the budget is spent, so the last selection has no reference consumer. A terminal iteration still labels the previous selection, trains its network and runs MD; only its own selection goes unused.

**maxsample** - Caps the number of structures one trajectory contributes to the reference calculation in one iteration. The candidate pool is the union of the frames above the minimum thresholds, the physically bad frames and the stopping frame when it is collected, taken newest first; the distance filter runs over the whole pool and the cap is applied after it. A value of 0 therefore selects nothing, and the round then ends with the reason no sample selected.

**load** - Chooses where iteration 1 takes its data and network. Only 1 is accepted: the run starts from the training data and the pretrained ensemble named in the input, and any other value stops the run at start-up with a message that asks for a jobtype train run first. The key is kept because the legacy loop could also start from raw data, which this build does not implement.

**transfer** - Requests transfer learning from the previous network. Only 0 is accepted; transfer learning is not ported in this build and any other value stops the run at start-up.

**remote_train** - Sets the route of the training child. 0 runs the fit locally; a non-zero value submits it through the remote route. Each iteration trains one network, so the key decides the route of one child runtime and not a fan-out.

**keep_md** - Keeps the round MD directory. 1 keeps iter-&lt;k&gt;/md after the round finishes, and 0 deletes it, which leaves the screening record as the only trace of the trajectories.

**record_step** - Sets the screening window: the last N retained history frames of every trajectory are screened each iteration. 0 is normalized to 10, and the MD child keeps at least four frames of history whatever the value. A trajectory shorter than the window is screened on the frames it has, and its completed flag follows the last recorded step against the planned MD length.

**check_population** - Collects the structure the MD loop stopped on when its own population guard ended the trajectory with stop code 1. The tolerance is not adaptive: &MD md_maxpop owns the judgment, the adaptive child inherits that value like any other MD run, and this switch only decides whether the stopping frame goes to the reference calculation. A frame whose stored density looks poor but whose run did not stop is not collected for that reason, and a failed calculation (stop code 4) is never collected.

**check_energy_order** - Turns on the same-spin level-order check. With 1 a frame whose same-spin level lies below an earlier one by more than gap_tolerance is a trigger and a candidate; the check reads energy and state-spin labels, and a trajectory that returns neither is reported as missing.

**gap_tolerance** - Sets the tolerance of the same-spin level-order check, in Hartree. An inversion larger than this value flags the frame, and a smaller one is treated as numerical noise.

**sample_energy** - Asks for energy labels and energy deviations. Only 1 is accepted: the energy is half of the training baseline and validation refuses 0.

**sample_grad** - Asks for gradient labels and gradient deviations. Only 1 is accepted: the gradient is half of the training baseline and validation refuses 0.

**sample_nac** - Asks for NAC labels and derivative-coupling deviations, labelled in Hartree/Bohr. Turning it on switches &MD nactype to nac, and a run whose nactype is already nac has this key switched on by normalization, so the two declarations cannot disagree. The reference queue must plan the nac capability, and an initial dataset that already carries NAC labels must agree with it, or start-up refuses the run.

**sample_soc** - Asks for SOC labels, labelled in cm-1. 0 and 1 are accepted; turning it on switches &MD dosoc on, and the reference queue must plan the soc capability.

**sample_dipole** - Asks for dipole labels. Only 0 is accepted in this build, which labels energy, gradient, NAC and SOC only: an input that turns this key or &MD dodipole on is refused by normalization.

**sample_transition_dipole** - Asks for transition-dipole labels. Only 0 is accepted in this build: the label extractor stops at energy, gradient, NAC and SOC, so an input that turns this key or &MD radiative on is refused by normalization.

**sample_oscillator_strength** - Asks for oscillator-strength labels. Only 0 is accepted in this build, for the same reason as sample_transition_dipole: the channel is not labelled yet.

**predict_mode** - Chooses the ensemble member that serves the MD trajectories. 0 returns the member mean; k above 0 returns the 1-based member k as the primary prediction and writes the other members to the alternative channels of the predictor comparison. A value above the member count, or any positive value with a one-member ensemble, is refused.

**maxmae** - Sets the validation energy MAE ceiling of the training-data quality gate, in eV. With a positive value, a fit whose validation energy MAE exceeds it is retried after the worst outlier among the frames added in this round is removed, one frame per attempt and one retraining per removal. A removal stays only when it lowers the MAE; when it does not, or when every new frame is an outlier, the run stops before MD with the reason recorded in the iteration record. A value of 0 disables the gate, so every fit stands.

**maxenergy** - Sets the energy deviation that triggers a frame, in Hartree, and is stamped on every MD child of the round as md_maxenergy, replacing the value &MD declared. Validation requires it to be at least minenergy.

**minenergy** - Sets the energy deviation that puts a frame in the candidate pool, in Hartree. A frame above it is selected when the trajectory triggered anywhere in its window, and validation requires it not to exceed maxenergy.

**dynenergy** - Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**inienergy** - Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**fwdenergy** - Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**bckenergy** - Belongs to the legacy dynamic-threshold family of the energy channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**maxgrad** - Sets the gradient deviation that triggers a frame, in Hartree/Bohr, and is stamped on every MD child of the round as md_maxgrad. Validation requires it to be at least mingrad.

**mingrad** - Sets the gradient deviation that puts a frame in the candidate pool, in Hartree/Bohr. Validation requires it not to exceed maxgrad.

**dyngrad** - Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**inigrad** - Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**fwdgrad** - Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**bckgrad** - Belongs to the legacy dynamic-threshold family of the gradient channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**maxnac** - Sets the NAC deviation that triggers a frame, in Hartree/Bohr, and is stamped on every MD child of the round as md_maxnac. Validation requires it to be at least minnac.

**minnac** - Sets the NAC deviation that puts a frame in the candidate pool, in Hartree/Bohr. Validation requires it not to exceed maxnac.

**dynnac** - Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**ininac** - Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**fwdnac** - Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**bcknac** - Belongs to the legacy dynamic-threshold family of the NAC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**maxsoc** - Sets the SOC deviation that triggers a frame, in cm-1, and is stamped on every MD child of the round as md_maxsoc. Validation requires it to be at least minsoc.

**minsoc** - Sets the SOC deviation that puts a frame in the candidate pool, in cm-1. Validation requires it not to exceed maxsoc.

**dynsoc** - Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**inisoc** - Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**fwdsoc** - Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**bcksoc** - Belongs to the legacy dynamic-threshold family of the SOC channel. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**refine** - Requests the legacy geometry refinement of the selected structures. It is not ported: only 0 is accepted, and any other value stops the run at start-up with a message that asks for 0.

**refine_num** - Belongs to the legacy refinement family. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**refine_start** - Belongs to the legacy refinement family. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**refine_end** - Belongs to the legacy refinement family. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**refine_gap** - Belongs to the legacy refinement family, where it named the gap criterion of a refinement window. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**explore_udd** - Turns on UDD exploration, which builds a bias from the covariance of the ensemble members. It needs predict_mode above 0, at least two members and a single-center run, and it follows the active state, so a hop moves the target. The bias starts at udd_start_iteration, and exploration trajectories are recorded but kept out of the production statistics and the population judgment.

**explore_gap** - Turns on gap-driven exploration, which needs at least two electronic states and marks its trajectories as exploration, outside the production statistics and the population judgment. With gap_method cv the selected pair is split by an antisymmetric potential above gap_threshold, so the pair mean is unchanged and the correction vanishes at the threshold. With gap_method paper the active propagation gradient becomes the upper-minus-lower difference, the excess energy is paid by rescaling the velocities, and the next round starts from the last pre-trigger frame, half on the lower and half on the upper state.

**gap_threshold** - Sets the gap, in Hartree, at which gap driving starts. At or below this value the correction is exactly zero, and above it the cv potential grows monotonically and saturates at gap_height. The gap analysis counts crossings of the same value, and a negative gap stops the exploration, which keeps the bias from driving an already inverted ladder.

**udd_mode** - Chooses what the UDD bias changes: force modifies the nuclear force only, potential adds the bias energy to the active state as well. The gradient channel is modified in both modes.

**udd_amplitude** - Sets the height of the UDD bias, in Hartree. 0 selects the automatic value, 5.0e-3 Hartree.

**udd_width** - Sets the width of the UDD bias, in Hartree. 0 selects the automatic value, one tenth of the ensemble standard deviation.

**udd_start_iteration** - Delays the UDD bias to the given iteration, counted from 1. Earlier rounds run unbiased and report no bias work.

**gap_method** - Chooses the gap-driving method. cv splits the selected pair with an antisymmetric potential that leaves the pair mean unchanged; paper replaces the active propagation gradient and rescales the velocities when the excess energy is payable. paper requires sfhp nosh and is refused together with explore_udd.

**gap_pair** - Names the state pair the gap bias acts on, as two indices such as 0 1. An empty value takes the pair from the state ladder, and a list that is neither empty nor two indices long, or that names a state outside the ladder, is refused.

**gap_height** - Sets the height of the saturating cv gap potential, in Hartree. 0 selects the automatic value, 1.0e-3 Hartree.

**gap_width** - Sets the decay width of the cv gap potential, in Hartree. 0 selects the automatic value, one third of the gap being driven.

**analyze_pes** - Turns on the QM-side PES analysis of the dataset. It reads the reference labels already in the data and never an NN frame: point-pair curvature and gradient-change magnitude, the energy-gradient residual, a low-rank local gradient reconstruction with its numerical rank and leave-one-out check, and the resolution curve that gives the geometry spacing at which the reconstruction stays within tolerance. Periodic datasets are not applicable in this version, and a thin neighbourhood reports the point count and nearest distance it would have needed.

**analyze_gap** - Selects the scope of the gap statistics: off, s0s1, adjacent or all. The analysis reads the unbiased predicted state energies of the screened frames and reports the minimum gap with its pair, the first crossing below gap_threshold and the fraction of frames below it; a frame without state energies counts as missing data. The same setting tells the MD child which gap trace to record, so a switched-on scope costs per-step work in the trajectory.

**analyze_trace** - Collects the scalar trajectory trace through the MD child: maximum force, population, energy drift and the minimum inter-atomic distance. A short trajectory with no rows reports not_applicable, and a trace that every trajectory omits stops the round.

**analyze_coverage** - Turns on the descriptor coverage report and the saturation test. Coverage is measured over a bounded sample of the dataset, at most 200 frames and 200 atoms; saturation is the moving average of the validation MAE over analyze_window rounds, and crossing the tolerance stops the run with the reason saturated.

**analyze_window** - Sets the number of rounds in the moving average of the saturation test, at least 1. The test needs one round more of validation MAE than the window.

**analyze_local_tolerance** - Sets the per-round validation-MAE improvement, in eV, that still counts as progress. An improvement above it keeps the round unsaturated even when the global tolerance looks loose; 0 disables the local test.

**analyze_global_tolerance** - Sets the validation-MAE improvement, in eV, below which the fit counts as saturated. With both tolerances at 0 the saturation stop is disabled.

**dynsample** - Requests the dynamic threshold update of the legacy loop. It is not ported: only 0 is accepted, and any other value stops the run at start-up with a message that asks for 0.

**maxdiscard** - Belongs to the legacy discard budget. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**pop_step** - Belongs to the legacy population-report cadence. The value is stored on the section and no adaptive stage reads it. This keyword is accepted but has no effect in v3.0.0.

**multitask** - Sets the generic fan-out flag of the adaptive node. It reaches the MD and reference stages as the fallback of md_multitask and reference_multitask, while the training child is always one runtime. An unset value keeps the workflow default, which is serial.

**remotetask** - Sets the generic remote flag of the adaptive node. It reaches the MD stage only, as the fallback of md_remotetask: training follows remote_train and the reference stage does not inherit it. An unset value keeps the workflow default, which is local.

**multipool** - Sets the size of the adaptive node own worker pool, at least 1. Each stage batch runs under its own pool, md_multipool for the trajectories and reference_multipool for the labels, so this key does not widen a stage fan-out.

**md_multitask** - Overrides the fan-out flag of the MD stage. Unset falls back to multitask and then to the workflow default, which is serial, and one trajectory runs per worker of md_multipool.

**md_remotetask** - Overrides the remote flag of the MD stage. Unset falls back to remotetask; a remote MD child is serialized and submitted by the framework, with the serving skill running locally inside it.

**md_multipool** - Sets the worker pool of the MD stage, at least 1, with one trajectory per worker. Unset means 1, and the generic multipool does not raise it.

**reference_multitask** - Overrides the fan-out flag of the reference stage. Unset falls back to multitask and then to the workflow default, which is serial, and the stage then runs over reference_multipool workers. The stage labels one selected structure per child runtime, keeping the full reference queue and its own composite layers from the highref and midref partition. A remote reference stage is a different key: reference_remotetask switches the skill itself to its slurm mode.

**reference_remotetask** - Switches the reference skill itself to its slurm execution mode. The reference child stays local and the skill writes its own submission script, so this key does not submit the framework runtime; unset means 0 and there is no generic fallback.

**reference_multipool** - Sets the worker pool of the reference stage, at least 1. Unset means 1, and the generic multipool does not raise it.

The loop owns three of the four MD guards: the max\* energy, gradient, NAC and SOC values are stamped on every MD child of the round, so they replace the md_max\* values that &MD declared, and a channel the run does not label leaves its child value in place. The population guard is not overwritten: &MD md_maxpop keeps the value the MD section gave it, and check_population only decides what the loop does with the structure that guard stopped on.

The retired key pop_tolerance is not read: an input that still carries it is refused at validation with a message that points at &MD md_maxpop. The legacy threshold family dyn\*, ini\*, fwd\* and bck\* parses and does nothing; the screen compares against max\* and min\* only, and the analysis, exploration and dispatch keys are read through the adaptive section alone, never through &MD.

The one list-typed key of the section is gap_pair, whose absent value is the empty list, that is no pair selected; the table prints the field declaration of the model for that default.
