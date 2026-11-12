---
title: "5.24. NN_PES"
nav_order: 24
parent: "5. Keyword sections"
---

The nn_pes section configures the nn_pes skill, which serves a trained neural-network potential energy surface and also owns the training run that produces it. It is the implementation of the nn theory label and is spelled &NN_PES in the classic input, nn in YAML and agent JSON. It is read when nn appears in the theory queue of &CONTROL.

The model container is a folder tree. Three ensembles live under modeldir, one folder per role: energy_grad for energies and gradients, nac for coupling vectors and soc for spin-orbit scalars. Each ensemble carries its own hyperparameter block, &EG, &NAC or &SOC, and this section selects the data, the ensemble size, the device and the runtime. The framework parity keys at the end of the list are documented once in [Section 5.2](05-02-system-molecule.html).

The skill exposes the following profiles. A profile fixes the method and the set of properties a calculation returns.

| Profile | Capability | Method |
| :--- | :--- | :--- |
| mlp_e_energy | energy | mlp_e |
| mlp_e_train | energy | mlp_e |
| mlp_e_soc | soc | mlp_e |
| mlp_e_soc_train | soc | mlp_e |
| mlp_eg_energy | energy | mlp_eg |
| mlp_eg_gradient | gradient | mlp_eg |
| mlp_eg_train | energy | mlp_eg |
| mlp_g2_gradient | gradient | mlp_g2 |
| mlp_g2_train | gradient | mlp_g2 |
| mlp_nac_nac | nac | mlp_nac |
| mlp_nac_train | nac | mlp_nac |
| mlp_nac2_nac | nac | mlp_nac2 |
| mlp_nac2_train | nac | mlp_nac2 |
| eg_nac_energy | energy | eg_nac |
| eg_nac_gradient | gradient | eg_nac |
| eg_nac_nac | nac | eg_nac |
| eg_nac_train | energy | eg_nac |
| eg_nac2_energy | energy | eg_nac2 |
| eg_nac2_gradient | gradient | eg_nac2 |
| eg_nac2_nac | nac | eg_nac2 |
| eg_nac2_train | energy | eg_nac2 |
| eg_nac_soc | soc | eg_nac |
| eg_nac_soc_train | soc | eg_nac |
| eg_nac2_soc | soc | eg_nac2 |
| eg_nac2_soc_train | soc | eg_nac2 |

The keywords, default values, and short descriptions are listed below.

**&NN_PES**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Chooses the arm of a training run. local starts one subprocess per ensemble member in place; slurm writes the same command into a job script and… |
| project | NNPES | Sets the stem of the run artifacts, for example {project}_train.summary.txt and {project}_{profile}.log. It follows nn_project, which is the single… |
| nn_project | NNPES | Sets the project stem through the classic-input spelling. It is copied into project, which names the training summary, the per-step log and the… |
| scratch_path | '' | Sets the parent directory of the calculation directory, which holds the step log and the uncertainty dump. The trained ensemble lives in modeldir… |
| keep_tmp | 0 | Keeps the calculation directory when 1. A value of 0 clears it when the backend is gone rather than after each step, because the log and the… |
| thread | 1 | Sets the thread count of the skill. nn_pes_thread is the framework spelling and the larger value wins, and the two lifetime budgets start from this… |
| nn_pes_thread | 1 | Sets the thread count through the framework spelling. It folds into thread, the larger value being used. |
| train_thread | 0 | Sets the threads of each training member subprocess, which gets its own budget. Empty or 0 inherits thread. |
| predict_thread | 0 | Sets the intra-op and inter-op pool size of the TensorFlow instance used for serving. Empty or 0 inherits thread. |
| slurm_time | 01:00:00 | Sets the wall-clock limit of the submitted training job in hh:mm:ss. It is read only when execution_mode is slurm. |
| slurm_partition | default | Sets the scheduler partition of the submitted training job. An empty value writes no partition directive, so the cluster default answers. |
| slurm_ntasks | 1 | Sets the number of tasks of the submitted training job. The skill starts no MPI ranks, so the value describes the allocation only. |
| slurm_memory | 4000 | Sets the memory per node of the submitted training job in MB. |
| slurm_command | '' | Sets the submission command of the batch script, for example sbatch or a site wrapper. An empty value uses the default. |
| profile | mlp_eg_gradient | Selects family and capability. {method}_energy, _gradient, _nac and _soc serve properties, and {method}_train fits an ensemble. Twenty-five profiles… |
| method | mlp_eg | Selects the network family: mlp_e for scalars, mlp_eg for energies with differentiated gradients, mlp_g2 for gradients computed directly, mlp_nac and… |
| modeldir | '' | Sets the model root, the folder holding energy_grad, nac and soc. A training run writes the role folders inside it and a serving run reads one a… |
| train_data | '' | Names the training file, which is read as Hartree energies with Hartree/Bohr gradients and couplings. A training profile needs exactly one of… |
| train_dataset | {} | Carries the same training data inline as a mapping instead of by file name, which is what the framework writes when it expands a permutation map. It… |
| pred_data | '' | Names a test set. A serving request that carries it is a test-set evaluation, one pass over the frames with an error table instead of one geometry… |
| gl_seed | 1 | Sets the master random seed. Each ensemble member is seeded with this value plus its index, which is what makes a fit reproducible, because a child… |
| train_mode | training | Selects whether the fit starts from scratch, training, or refits an existing folder, retraining. A hyperparameter written once is mirrored into the… |
| mult_nn | 2 | Sets the ensemble size, which is also how many semicolon-separated member groups of a hyperparameter are read: one group serves every member, surplus… |
| nn_predict_mode | 0 | Selects the serving predictor. 0 returns the member mean, and k from 1 to mult_nn returns member k as the prediction while the other members go to… |
| gpu | -1 | Lists one device index per ensemble member, separated by semicolons; a single value broadcasts to every member and -1 asks for no device. The index… |
| eg | {} | Holds the hyperparameters of the energy ensemble in the per-member form of the container, for example depth 3 and nn_size 100;60. The vocabulary and… |
| nac | {} | Holds the hyperparameters of the coupling ensemble, in the same per-member form. The flat keyword nac means this block, never the couplings a run… |
| soc | {} | Holds the hyperparameters of the spin-orbit ensemble, which is trained by the energy family with its derivative head off, because a spin-orbit value… |
| nstate | 2 | Sets the number of electronic states, which is the channel count of the energy ensemble and the ladder the couplings are declared over. The value… |
| nnac | 0 | Sets the number of declared coupling channels, normally resolved from the framework coupling declaration. One declared pair is one channel, and the… |
| target_states | [0, 1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_nac_pair | [[]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_soc_pair | [[]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| write_step_log | 0 | Writes two per-step diagnostic files, {project}_{profile}.log and {project}_{profile}.uncertainty.txt, when 1. The ensemble spread itself is computed… |
| predict_std | 0 | Reports the post-fit prediction spread on the training frames when 1. It belongs to training and prediction only; a serving call never enters that… |
| charges | 0 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| multiplicity | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| ci | [1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| spin | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| coupling | [] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| pbc | [false, false, false] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| cell | [99, 99, 99, 90, 90, 90] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| lattice_vector | [[0, 0, 0], [0, 0, 0], [0, 0, 0]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| kpoint | [1, 1, 1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Chooses the arm of a training run. local starts one subprocess per ensemble member in place; slurm writes the same command into a job script and submits one blocking sbatch job per member. Serving is always in process and ignores this key.

**project** - Sets the stem of the run artifacts, for example {project}_train.summary.txt and {project}_{profile}.log. It follows nn_project, which is the single source of truth.

**nn_project** - Sets the project stem through the classic-input spelling. It is copied into project, which names the training summary, the per-step log and the uncertainty dump.

**scratch_path** - Sets the parent directory of the calculation directory, which holds the step log and the uncertainty dump. The trained ensemble lives in modeldir, outside it.

**keep_tmp** - Keeps the calculation directory when 1. A value of 0 clears it when the backend is gone rather than after each step, because the log and the uncertainty dump are appended across a run; it also removes {project}_train.summary.txt.

**thread** - Sets the thread count of the skill. nn_pes_thread is the framework spelling and the larger value wins, and the two lifetime budgets start from this value.

**nn_pes_thread** - Sets the thread count through the framework spelling. It folds into thread, the larger value being used.

**train_thread** - Sets the threads of each training member subprocess, which gets its own budget. Empty or 0 inherits thread.

**predict_thread** - Sets the intra-op and inter-op pool size of the TensorFlow instance used for serving. Empty or 0 inherits thread.

**slurm_time** - Sets the wall-clock limit of the submitted training job in hh:mm:ss. It is read only when execution_mode is slurm.

**slurm_partition** - Sets the scheduler partition of the submitted training job. An empty value writes no partition directive, so the cluster default answers.

**slurm_ntasks** - Sets the number of tasks of the submitted training job. The skill starts no MPI ranks, so the value describes the allocation only.

**slurm_memory** - Sets the memory per node of the submitted training job in MB.

**slurm_command** - Sets the submission command of the batch script, for example sbatch or a site wrapper. An empty value uses the default.

**profile** - Selects family and capability. {method}_energy, _gradient, _nac and _soc serve properties, and {method}_train fits an ensemble. Twenty-five profiles are declared, and a profile that does not belong to the method is refused.

**method** - Selects the network family: mlp_e for scalars, mlp_eg for energies with differentiated gradients, mlp_g2 for gradients computed directly, mlp_nac and mlp_nac2 for couplings by a virtual potential or by propagation, and eg_nac or eg_nac2 for an energy ensemble together with a coupling ensemble.

**modeldir** - Sets the model root. The folder holds one subfolder per role, energy_grad, nac and soc, and each subfolder carries the hyper_v{i}.json, weights_v{i}.weights.h5, scaler_v{i}.json, fiterr_train_v{i}.npy and fiterr_valid_v{i}.npy files of every member. A training run writes those folders; a serving run reads them and refuses a folder without a valid-error file, because the parent writes freshly initialized weights before the fit starts. The folder is also what a scaler is read from when a test-set evaluation wants physical units.

**train_data** - Names the training file for a training profile. The file is read as Hartree energies with Hartree/Bohr gradients and couplings, the convention the engine backends report; there is no unit keyword, and a file carrying a unit block for another scheme is refused at read time. The trainer converts the derivative onto the Angstrom axis the network is fed, so the one rule the pair has to satisfy is that the gradient is the derivative of the energy the same surface reports. Exactly one of train_data and train_dataset has to be set.

**train_dataset** - Carries the training set inline as a mapping, in the same shape and under the same unit rules as a file. The framework expands a permutation map into this mapping before the backend is initialized, so the skill never reads a map and never reorders atoms itself. It is mutually exclusive with train_data; setting both, or neither, is refused before any folder is created.

**pred_data** - Names a test set for a serving profile. The request then becomes a test-set evaluation: one pass over the file frames and an error table, instead of one geometry and its properties. The file is read by the same loader and unit convention as a training file, and no training file is needed for it, because the scaler a metric needs lives in the model folder. A training profile that carries pred_data is refused, since a fit and a measurement are two different calls.

**gl_seed** - Sets the master random seed. Each ensemble member is seeded with this value plus its index, which is what makes a fit reproducible, because a child process inherits no random state.

**train_mode** - Selects training, a fit from scratch, or retraining, a refit of an existing folder. The hyperparameter schema mirrors every keyword written once into the retraining section as well, because this key decides which of the two sections the fit reads. A value outside the two names is refused.

**mult_nn** - Sets the ensemble size. It decides how many member groups of a hyperparameter are read, where the separator between members is a semicolon and the member axis is always the outer one. One group serves every member and more groups than members keep the leading ones, while fewer groups than members raise when more than one was written, because silently truncating an ensemble would train a different model than the one asked for. With two or more members the skill also reports the member-to-member standard deviation as the error bar of the surface.

**nn_predict_mode** - Selects which member a serving call reports. 0 returns the member mean and is the default; k from 1 to mult_nn returns member k as the prediction and writes the other members to the alternative channels, which is what the adaptive analysis layer reads. A negative value, a value beyond mult_nn, or a value above 0 with a single-member ensemble is refused, because there is no second model to compare with.

**gpu** - Lists the device index of each training subprocess, one per ensemble member, separated by semicolons; a single value broadcasts to every member, and -1 asks for no device at all. The entry is a device number, not the original project mode number, so gpu 1 means device 1. The parameter is validated per member and must not be below -1; the GPU code paths are present but unverified, because the development host is a CPU-only wheel.

**eg** - Holds the hyperparameters of the energy ensemble, one block per member with semicolon as the member separator, for example depth 3 with nn_size 100;60 or loss_weights 1 1;1 10. The block is read when the run fits the energy family, and the container keeps one section for training and one for retraining. The vocabulary, the value shape and the broadcast rule belong to the hyperparameter schema: an unknown keyword is refused with the accepted list before a training folder is created, and a scalar broadcasts while an index table or a list needs one row per member.

**nac** - Holds the hyperparameters of the coupling ensemble in the same per-member form. The coupling family has its own constraints: mlp_nac builds one output per electronic state and is therefore accepted only for a declaration of exactly nstate pairs, while a wider declared pair set needs mlp_nac2. The training file channel count has to equal the declared count, so this block and the framework coupling declaration are read together.

**soc** - Holds the hyperparameters of the spin-orbit ensemble, in the same per-member form. The role is fitted by the energy family with the derivative head off, because a spin-orbit matrix element is a scalar that depends on geometry rather than a derivative of anything, and it is reported in cm-1 with no gradient channel. A soc jobtype token resolves to the mlp_e_soc profile, and a dynamics run that needs all three roles asks for the composite eg_nac_soc profile.

**nstate** - Sets the number of electronic states, which is the channel count of the energy ensemble and the ladder the couplings are declared over. The value must be at least 1.

**nnac** - Sets the number of declared coupling channels, normally resolved from the framework coupling declaration. One declared pair is one channel, and the width of the loaded ensemble is compared with this count at serving.

**target_states** - The framework declares which states a run reports, and the skill serves one channel of the energy ensemble per state, so a target outside the ladder is refused instead of being answered with a neighbouring channel. The state set also sizes the coupling declaration: the framework splits the declared pairs on the state spins and hands the resolved nac_pair_map and soc_pair_map over, which is what the coupling channels are compared with at serving.

**target_nac_pair** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**target_soc_pair** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**write_step_log** - Writes two per-step diagnostic files, {project}_{profile}.log and {project}_{profile}.uncertainty.txt, when 1. The ensemble spread itself is computed either way, because it is the adaptive sampling signal.

**predict_std** - Reports the post-fit prediction spread on the training frames when 1. It belongs to training and prediction only; a serving call never enters that analysis.

**charges** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**multiplicity** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**ci** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**spin** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**coupling** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**pbc** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**cell** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**lattice_vector** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**kpoint** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

TensorFlow is an optional dependency of the package. Install it as pyrai3md[nn] or tensorflow&gt;=2.20 on Python 3.11 or newer; without it the skill is still listed by discovery and any use fails with "tensorflow is not installed" instead of looking absent. The backend is molecular: supports_pbc is false and any non-zero periodicity is refused, because the geometric features and the index weights were built and measured for molecules only.

One training term runs per process. A fit reconfigures TensorFlow device visibility for the whole process, so two terms that train cannot share a thread pool, and the thread count is process-scoped rather than term-scoped: a second term asking for a different count is told so rather than silently given one.

A fit has no numeric convergence verdict. The framework judges whether the training record came back, marking a run failed when a promised channel is missing or empty, but no threshold is applied to the reported validation MAE: whether the surface is good enough for a given dynamics run is the caller's judgement from the summary. The summary prints metrics in eV and eV/Angstrom, and spin-orbit values in cm-1.

The coupling and spin-orbit channels are declared, never derived, and channels are never zero-filled, because a silently missing coupling changes a surface-hopping trajectory. The shipped fixtures are demonstrations rather than converged surfaces, the coupling and spin-orbit ensembles rest on synthetic data, and the GPU paths are present but unverified on a CPU-only development host.
