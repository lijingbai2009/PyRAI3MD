---
title: "13. Examples and tutorials"
nav_order: 13
---

This chapter works from the inputs that ship with the package. The examples directory is the reference for every workflow keyword combination that the released version supports. Reading one example beside [Section 6](06-workflows.html) is usually faster than composing an input from the keyword tables alone.

## 13.1. The shipped examples

The directory holds 213 YAML inputs and 254 files in total. Almost all of the extra files are companion data that one or more inputs read: xyz endpoints for a band, a charge file, the LAMMPS topology triple of .data, .in.init and .in.settings, an .init.xyz batch of starting points, or a JSON training set. Each YAML input is runnable as it stands, from that directory, with pyrai3md followed by the file name.

The files are grouped by the prefix before the first underscore. The table counts every file of the family, inputs and companions together, and gives the number of YAML inputs in parentheses where the two differ.

| Family | Files | What it demonstrates | Representative files |
| :--- | :--- | :--- | :--- |
| path | 48 (32 inputs) | Path searches: minimum energy path, intrinsic reaction coordinate and nudged elastic band walks, in the gas phase, in a periodic cell, and across a composite; the shipped endpoint files carry the reactant, product and transition-state guesses. | path_neb, path_irc, path_mep_pbc, path_cineb, path_neb_xtb |
| multiscale | 41 (37 inputs) | Two- and three-layer composites: ONIOM2 and ONIOM3 with mechanical or electrostatic embedding, additive QM/MM, charge management from a file, and the composite versions of the searches (mecp, meci, mep, neb, ts). | multiscale_qmmm_embedding_sp, multiscale_oniom2_orca_embedding_sp, multiscale_oniom3_orca_embedding_npt, multiscale_qmmm_neb, multiscale_qmmm_ts |
| constraint | 37 | The constraint block in every role: harmonic and flat-bottomed biases, confining walls, fixed atoms and directions, SHAKE and RATTLE, metadynamics including the RMSD form, and the same terms inside a periodic NVE, NVT or NPT trajectory. | constraint_md_bias, constraint_md_shake, constraint_md_periodic_npt_wall, constraint_md_metad_rmsd, constraint_opt_surface |
| multicenter | 30 | Several QM centers in one system: one high-level region per center, the per-center state ladders, the fitted charge rounds, and the periodic box that carries all of them. | multicenter_qmmm_sp, multicenter_oniom3_orca_embedding_npt, multicenter_pbc_nve, multicenter_qmmm_sc_charge, multicenter_qmmm_num_freq |
| md | 30 (27 inputs) | The trajectory itself: NVE, NVT and NPT with each thermostat and barostat pairing, surface hopping from four engines, velocity handling, batch and parallel dispatch, and integer-step restarts. | md_local, md_bagel_fssh, md_lammps_nvt_nhc, md_lammps_npt_nhc_mttk, md_parallel |
| optimize | 25 | Single-structure searches: minima, transition states, crossings of two surfaces, GDIIS, partial optimizations that move a subset of atoms, and variable-cell relaxation under an external pressure. | optimize_minima, optimize_ts, optimize_cell_gdiis, optimize_cell_meci, optimize_partial_ts |
| adaptive | 20 (18 inputs) | Adaptive sampling rounds over a reference and a neural network: the single-method carrier, the QM/MM and ONIOM2 carriers, the local and slurm dispatch pairs, the training-data quality gate, and one carrier per analysis switch. | adaptive_single_local, adaptive_qmmm_local, adaptive_quality_gate_local, adaptive_analysis_gap_local, adaptive_oniom2_slurm |
| nn | 4 | The machine-learning side without a trajectory: training from a dataset, prediction from a trained ensemble, and the hyperparameter grid scan that ranks the resulting models. | nn_train, nn_predict, nn_grid_search, nn_grid_search_local |
| hessian | 1 | The framework numerical Hessian and vibrational analysis on a three-atom system, sized to run inside a short qualification budget. | hessian_num |
| single point | 2 | The two baselines every other family is read against: one energy and one gradient on the same ethylene geometry, with a minimal theory section. | energy.yaml, gradient.yaml |

The path family is the largest because each walk exists in the gas phase, in a periodic cell and across a composite. path_neb and path_irc show the two walkers with framework gradients, path_neb_xtb repeats the band on the xTB skill, and path_mep_pbc shows a walk whose points carry a cell. The multiscale family covers the same ground for composites: multiscale_qmmm_embedding_sp is the smallest readable QM/MM input, multiscale_oniom3_orca_embedding_npt adds the three-layer form and a moving box, and multiscale_qmmm_neb and multiscale_qmmm_ts show that a composite is an ordinary theory queue as far as the workflow is concerned.

The constraint family is organised by term and then by ensemble: constraint_md_bias and constraint_md_metad_rmsd show the two bias forms, constraint_md_shake shows a hard constraint, and constraint_md_periodic_npt_wall and constraint_opt_surface put a wall and a bias into a periodic NPT trajectory and into a surface search. The multicenter family answers the other structural question: multicenter_qmmm_sp is two centers with one charge round, multicenter_qmmm_sc_charge iterates the charges to a fixed point, and multicenter_oniom3_orca_embedding_npt carries three centers through a barostatted run.

The md family is the one to read before writing any trajectory input. md_local is the smallest runnable trajectory, md_bagel_fssh is the surface-hopping skeleton that this chapter repeats with PySCF, and the md_lammps_nvt_nhc and md_lammps_npt_nhc_mttk pair show a legal thermostat and a legal thermostat-barostat pairing on a classical engine. The optimize family reads the same way: optimize_minima, optimize_ts, optimize_cell_gdiis and optimize_cell_meci name one search each, and optimize_partial_ts shows hess_atoms holding all but a few atoms.

The adaptive family gives each feature its own carrier. adaptive_single_local is the reference chain in its shortest form, adaptive_qmmm_local puts a neural network in the QM role of a solvent box, adaptive_quality_gate_local is the only input that sets maxmae, and adaptive_analysis_gap_local sets analyze_gap. The four nn inputs cover the model lifecycle without a trajectory, hessian_num is the numerical Hessian carrier, and energy.yaml and gradient.yaml are the two baselines.

> **Note.** The examples assume the engines they name are installed and that the scratch directory exists. A family that needs an engine the host does not have fails at its first skill call, not at input validation, so read the header comment of a file before treating that failure as an input error.

## 13.2. Tutorial: energy, optimization and frequencies

The three inputs below are one water molecule carried through the three single-structure workflows. They share the electronic structure section, so a mistake in one run is the same mistake in the next. The first input asks for the energy only.

```yaml
control:
  title: h2o_energy
  jobtype: energy
  theory: [pyscf]
 
atoms: [O, H, H]
coordinates:
  - [0.000000000, 0.000000000, 0.117300000]
  - [0.000000000, 0.757200000, -0.469200000]
  - [0.000000000, -0.757200000, -0.469200000]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]
 
pyscf:
  method: hf
  basis: sto-3g
  scratch_path: ./scratch
```

The second input relaxes the geometry. The optimize section is the one from optimize_minima.yaml, and the jobtype token selects the iterative workflow rather than the single-point one.

```yaml
control:
  title: h2o_opt
  jobtype: optimize
  theory: [pyscf]
 
optimize:
  method: minima
  maxit: 100
  opt_energy: 1e-6
  opt_grms: 3e-4
  opt_gmax: 4.5e-4
  opt_drms: 1.2e-3
  opt_dmax: 1.8e-3
 
atoms: [O, H, H]
coordinates:
  - [0.000000000, 0.000000000, 0.117300000]
  - [0.000000000, 0.757200000, -0.469200000]
  - [0.000000000, -0.757200000, -0.469200000]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]
 
pyscf:
  method: hf
  basis: sto-3g
  scratch_path: ./scratch
```

The third input asks for the frequencies at the optimized geometry, which is copied from the opt.xyz the second run leaves behind. The hessian section asks for the molden record and for the thermochemical corrections at room conditions.

```yaml
control:
  title: h2o_freq
  jobtype: frequency
  theory: [pyscf]
 
hessian:
  hess_step: 0.005
  hess_molden: true
  hess_temperature: 298.15
  hess_pressure: 1.0
 
atoms: [O, H, H]
coordinates:
  - [0.000000000, 0.000000000, 0.119400000]
  - [0.000000000, 0.761500000, -0.474200000]
  - [0.000000000, -0.761500000, -0.474200000]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]
 
pyscf:
  method: hf
  basis: sto-3g
  scratch_path: ./scratch
```

The three commands are run in this order from one directory.

```text
pyrai3md h2o_energy.yaml
pyrai3md h2o_opt.yaml
pyrai3md h2o_freq.yaml
```

The energy run writes one result block to h2o_energy.log. The block opens with the electronic energy in Hartree, and because the input asked for nothing else, the remaining sections either carry the gradient twice or say that a property was not computed.

The optimization run repeats the block once per evaluation, under a step header, and keeps four products. opt.xyz is overwritten every step and holds the structure the run would stop on. opt_geom.xyz is appended every step and is the trajectory of the search. opt_status.txt carries one row per accepted step with the columns step, energy, shift, max_d, rms_d, max_g and rms_g, in Hartree, Angstrom and Hartree/Bohr. The fourth product is the run log, whose final section states which criterion stopped the run.

The frequency run writes h2o_freq.freq.molden. The sampler looks for that exact name, so it has to keep it when the file is moved beside a later dynamics input. The log carries the vibrational frequencies in cm-1, the normal modes and the thermochemical corrections; the molden record carries the frequencies and mass-weighted modes in the layout the Wigner sampler reads. Water has three real frequencies at a minimum, and an imaginary mode would be printed as a negative one.

## 13.3. Tutorial: surface hopping

The input below runs a ten-step FSSH trajectory on ethylene with three singlet states and PySCF TDA. It follows md_bagel_fssh.yaml, with the same geometry, the same starting velocities and the same state ladder, so the two inputs differ only in the electronic structure engine. The coupling is requested as a list of pairs and the dynamics estimates each coupling from the energy curvature, so the engine only has to return state energies and gradients.

```yaml
control:
  title: md_ethylene_fssh
  jobtype: md
  theory: [pyscf]
 
atoms: [C, C, H, H, H, H]
coordinates:
  - [-1.120191050, 0.000000000, -2.661941306]
  - [-1.120191050, 0.000000000, -1.331783306]
  - [-0.590584410, 0.756356668, -3.234891306]
  - [-1.649797690, -0.756356668, -3.234891306]
  - [-1.876547718, -0.529606640, -0.758833306]
  - [-0.363834382, 0.529606640, -0.758833306]
 
velocities:
  - [0.000233177, -0.000042299, -0.000208531]
  - [-0.000217441, 0.000195315, 0.000185800]
  - [-0.000893166, -0.000054466, 0.000176273]
  - [0.001361820, -0.000285727, 0.000253295]
  - [-0.000024165, -0.000666228, 0.000131482]
  - [-0.000631993, -0.000816868, -0.000290190]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [3]
  spin: [0]
  coupling: "1 2, 2 3, 1 3"
 
md:
  ensemble: nvt
  thermostat: nose-hoover
  temperature: 300
  initcond: 0
  ninitcond: 1
  step: 10
  size: 20.67
  root: "2"
  sfhp: fssh
  nactype: ktdc
  substep: 20
  deco: "0.1"
 
pyscf:
  method: tda
  functional: pbe
  basis: sto-3g
  nstate: 3
  target_states: [0, 1, 2]
  execution_mode: local
  keep_tmp: 0
  thread: 1
```

The md block answers four separate questions. ensemble, thermostat and temperature fix the nuclear bath. step and size fix the propagation, with step counting steps and size giving the timestep in atomic units. root names the state the trajectory starts on, in the 1-based spelling the input uses everywhere, so root two is the second state. sfhp, nactype, substep and deco fix the electronic side: the fewest switches algorithm, couplings estimated from the energy curvature, twenty electronic substeps per nuclear step, and the decoherence time in atomic units. [Section 5.3](05-03-dynamics.html) gives the full vocabulary and Section B gives the index bases.

The run writes six md files and, whenever a hop happens, the matching six sh files. md.xyz holds every recorded frame in Angstrom with a comment line naming the title, the step and the active state. md.velo holds the velocities of the same frames in atomic units. md.energies holds the step, the potential energy of the active state, the kinetic energy and the per-state energies. md.pop holds the populations followed by a one-hot vector that marks the post-hop active state. md.ensemble is written only for a thermostatted or barostatted ensemble and carries the thermostat and barostat readings of the step. md.stress appears only when the engine supplied a stress tensor. The sh files carry the same columns at hopping steps only, and the writer emits them immediately rather than on the direct and buffer cadence that governs the md files.

A hop is read in three places. The md.xyz comment of the step gains a suffix of the form to 2 CI, so a frame that changed state is recognisable without the populations. The active-state block of md.pop switches its one-hot vector from the old state to the new one. The sh.xyz frame for the same step appears with the post-hop velocities in sh.velo, which is the velocity the integrator rescaled or reflected to conserve energy across the gap. In the log, the per-step header is followed by the same numbers the records carry, so a hop can be found by searching for the comment suffix once the step is known.

The run also writes md_ethylene_fssh.rt on the cadence of the md checkpoint keyword, and by default it writes none. A trajectory that was interrupted is continued by launching that file, not by editing the input: pyrai3md md_ethylene_fssh.rt. The restored state carries the step count, the history window, the electronic density and the random generator position, so the resumed segment continues the same hopping sequence instead of re-seeding it. The input is read once at the start; changing the dynamics section afterwards does not alter the state a runtime file restores.

> **Note.** The default verbosity does not write per-step amplitudes. Set verbose 4 in the control section when the amplitudes themselves have to be inspected, which writes &lt;title&gt;.fssh; the file grows with every electronic step, so it is a diagnosis tool rather than a production record.

## 13.4. Tutorial: QM/MM

A QM/MM run is a two-entry theory queue with multiscale_type qmmm. The first entry is the QM provider and the second is the classical provider, which also owns the box. The highlevel list names the QM region, and the embedding switch decides whether the QM calculation sees the classical charges as a field or only through the additive energy. The input below puts one water in the QM region and one in the classical region, on a six-atom cluster.

```yaml
control:
  title: qmmm_water_dimer
  jobtype: energy gradient
  theory: [pyscf, lammps]
  multiscale_type: qmmm
 
atoms: [O, H, H, O, H, H]
coordinates:
  - [1.000000000, 1.000000000, 1.000000000]
  - [2.000000000, 1.000000000, 1.000000000]
  - [1.000000000, 2.000000000, 1.000000000]
  - [5.000000000, 5.000000000, 5.000000000]
  - [6.000000000, 5.000000000, 5.000000000]
  - [5.000000000, 6.000000000, 5.000000000]
 
molecule:
  charges: 0
  multiplicity: 1
  pbc: [0, 0, 0]
  ci: [1]
  spin: [0]
  highlevel: [1, 2, 3]
  q_qm: [-0.83, 0.415, 0.415]
  embedding: true
 
pyscf:
  method: hf
  basis: sto-3g
  execution_mode: local
  thread: 1
 
lammps:
  method: ff
  lammps_project: qmmm_cluster_test
  lammps_shake: 0
  python_enabled: true
  execution_mode: local
```

The pbc line is not decoration. The classical topology declares a ten Angstrom box, and an input that leaves the periodicity to the default reads as a molecular run in its own log while the engine still loads the box. Stating pbc as three zeros declares this run a cluster, which is what the shipped QM/MM examples do when they reuse a periodic topology. The lammps_project value names the topology stem the classical stage reads.

A larger environment is built rather than typed. The multiscale generator packs a solute into solvent shells, writes the classical files and produces the combined starting conditions:

```text
title         waterbox
mode          create
env_type      solvent
solute        solute.xyz
solvent       solvent.xyz
nshell        2
density       1.0
solbox        60
mass          18.0
packmol       /path/to/packmol
cpus          8
```

```bash
python tools/multiscale_generator.py creation.in
```

The tool writes the packed system and the classical files beside the creation file; the merge and edit modes then adjust a starting geometry or a velocity set without rebuilding the box. [Section 10.6](10-machine-learning-potentials.html#106-data-tools) describes the modes and [Section 7.1](07-multiscale-calculations.html#71-additive-qmmm) describes the composite itself.

A QM/MM run reports the composite in its own log block, headed &composite energy decomposition in Hartree. Each term occupies one line, ordered high to low, carrying the raw energies of that term in state order, and a final total line carries the additive QM/MM energy of each state. The classical term is the full force-field evaluation of the whole system, not the MM correction alone, so the terms do not sum to the total by eye; the coefficients that make them sum are the plan's, and the block states them by order. Beside it, the &qmmm 0-order multipole charges block reports the charges the classical field was built from, and the &electronic energy block reports the assembled total that the workflow judged. A run with embedding off prints no charge block, which is the quickest way to confirm which embedding the run actually used.

## 13.5. Tutorial: adaptive sampling

Adaptive sampling alternates between running trajectories on the current neural network, labelling the selected structures with the reference method, and refitting the network. The input below runs two iterations over a pending ethylene model, with PySCF as the reference, the quality gate armed at 0.0082 eV, and the gap analysis restricted to the ground and first excited state. It is adaptive_quality_gate_local.yaml with one analysis switch added.

```yaml
control:
  title: adaptive_ethylene
  jobtype: adaptive
  theory: [nn]
  reference: [pyscf]
  gl_seed: 20261105
 
atoms: [C, C, H, H, H, H]
coordinates:
  - [-1.1201910500, 0.0000000000, -2.6619413060]
  - [-1.1201910500, 0.0000000000, -1.3317833060]
  - [-0.5905844100, 0.7563566680, -3.2348913060]
  - [-1.6497976900, -0.7563566680, -3.2348913060]
  - [-1.8765477180, -0.5296066400, -0.7588333060]
  - [-0.3638343820, 0.5296066400, -0.7588333060]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [2]
  spin: [0]
 
md:
  initcond: read
  ninitcond: 5
  step: 10
  size: 20.67
  root: 1
  sfhp: nosh
  nactype: ktdc
  rk4: 1
  format: molden
 
adaptive:
  maxiter: 2
  maxsample: 1
  record_step: 4
  maxmae: 0.0082
  keep_md: 1
  load: 1
  md_multitask: true
  md_multipool: 5
  reference_multitask: true
  reference_multipool: 5
  maxenergy: 0.0
  minenergy: 0.0
  maxgrad: 0.0
  mingrad: 0.0
  gap_tolerance: 1.0e-6
  sample_energy: 1
  sample_grad: 1
  analyze_gap: s0s1
 
nn:
  method: mlp_eg
  modeldir: ./nn_ethylene_init
  train_data: ./egs_ethylene_gate_100.json
  mult_nn: 2
  nstate: 2
  target_states: [0, 1]
  execution_mode: local
  scratch_path: ./scratch
  eg:
    depth: 2
    nn_size: 16
    batch_size: 5
    epo: 30
    learning_rate: 1.0e-3
    epostep: 1
    loss_weights: '1 10'
    val_split: 0.1
 
pyscf:
  method: tda
  functional: pbe
  basis: sto-3g
  nstate: 2
  target_states: [0, 1]
  execution_mode: local
  thread: 1
```

Four settings carry the workflow. theory names the surface the trajectories run on and reference names the method that labels them; nn in the reference queue is refused, because a network cannot label itself. maxiter bounds the rounds and maxsample bounds how many structures one round may keep. load is the statement that the round starts from the data and the network the input names rather than from a previous run. The md block is an ordinary dynamics block, because every round launches ordinary trajectories.

The products collect under a directory named after the title, not in the launch directory:

```text
adaptive_ethylene.adaptive/
  adaptive_ethylene.adaptive.json      overall round record
  adaptive_ethylene.log                run log, one sampling block per round
  iter-1/
    md/traj_<idx>/                     the round's trajectories
    selected_iter-1.xyz                the structures the round selected
    population_iter-1.json             the population record
    status_iter-1.json                 the round's counts and verdict
  iter-2/
    data_iter-2_<nframe>.json          the accumulated training set
    NN-adaptive_ethylene-iter-2/       the refitted ensemble
    NN-adaptive_ethylene-iter-2-r1/    a gate retrain, when one ran
    reference/ref_<idx>/               the labelling calculations
    reference/new-samples.json         the labels this round added
    analysis_iter-2.json               only when an analysis switch is on
    md/traj_<idx>/
    selected_iter-2.xyz
```

The first iteration is the external round: it runs trajectories on the network supplied in the input, selects structures and labels them, but has no model of its own to write. Every later iteration fits a model, stores it beside the data snapshot and the labels, and hands it to the next round. The data file name carries the frame count, so data_iter-2_106.json is the 106-frame set of the second round and the number can be checked against the two carrier counts without opening the file.

A round is judged on three things before its trajectories are trusted. The maxmae gate compares the fitted model's validation energy MAE against the ceiling in eV; a fit above the ceiling is retried after the worst new frames are removed one at a time, and the retry is kept only if it lowers the MAE. The gate writes a &nn validation gate block and records the decision, so a passing fit and a pruned fit are distinguishable from the log alone. The gap analysis, when analyze_gap is not off, reports the minimum gap, the first crossing and the fraction of frames below a threshold for the chosen state scope, which is how a round that collected only high-energy structures is noticed. The physical screens are the third check: maxenergy, maxgrad and their minima bound the accepted structures, and check_population decides whether the structure a trajectory stopped on is collected.

> **Note.** The analysis switches default off and the analysis file is written only when at least one is on. A missing analysis_iter-2.json is therefore a statement about the input, not about the round.
