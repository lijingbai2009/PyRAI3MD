---
title: "6. Workflows"
nav_order: 6
---

This chapter follows the jobtype. It states what each token selects, what a property run plans, how each iterative workflow moves the structure it is given, and what a run leaves on disk. The keyword tables behind these descriptions are in Chapter 5.

## 6.1. Job types and workflow resolution

The control keyword jobtype selects the workflow and, for a property run, the capabilities planned. Basic tokens name a property and combine freely; an advanced token names a workflow and stands alone. [Section 5.1](05-01-control.html) lists the keyword itself.

| Token | What the run returns |
| :--- | :--- |
| energy | Total energies of the state ladder, Hartree. |
| gradient | Energies and the nuclear gradient of each state, Hartree/Bohr. |
| nac | Derivative couplings of the declared pairs, Hartree/Bohr. |
| soc | Spin-orbit couplings of the declared pairs, cm-1. |
| frequency | Force constants read into frequencies, modes and thermochemistry. |
| absorption | Transition dipoles, oscillator strengths and the spectrum. |
| dipole | State dipole moments. |
| transition | Transition dipole moments of the declared pairs. |
| hessian | The force-constant matrix, without the vibrational reading. |
| num_freq | The frequency output, by the framework displacement batch. |
| num_hess | The matrix alone, by the framework displacement batch. |

| Advanced token | What the run returns |
| :--- | :--- |
| md | A trajectory: frames, velocities, energies, populations, optional hops, a runtime record. |
| surface_hopping | The same dynamics loop, spelled for a run whose subject is the hops. |
| adaptive | Rounds of data, training, dynamics, screening and reference labelling. |
| train | One fit: the training record returned and the model folder written. |
| prediction | One served pass over a test set and its error statistics. |
| search | A grid of training runs, ranked after the grid finishes. |
| optimize | A final structure, a per-step convergence record and the accepted steps. |
| read | Frames extracted from finished calculations, written nowhere. |

Eight aliases resolve to the canonical spelling: sp is energy, opt is optimize, freq and frequence are frequency, hess is hessian, numfreq is num_freq, numhess is num_hess, namd is md and predict is prediction. hop and hybrid are retired, refused with a message naming md with a hopping algorithm and a composite queue; [Appendix E](appendix-e-migrating-pyrai2md-inputs.html) lists both. Mixing a basic token with an advanced one is refused, as are two advanced tokens.

A driver other than main short-circuits resolution: every jobtype reaches the skill unchanged and the run is a single shot, one skill, one evaluation, the outcome settled from what the skill returned. A path method and a variable-cell search are refused there, and exactly one theory skill is required. With driver main the workflow comes from the token, and a basic token resolves none, so a property run is one calculator call. No path jobtype or section exists: optimize.method routes the three path names.

There is no scan section either, and jobtype scan is refused as an unknown jobtype because the token belongs to neither group. The registered Scan class is a placeholder that would repeat the input geometry and record each energy; its coordinate generation was never written.

## 6.2. Single-point and property calculations

A run holding only basic tokens plans one capability set and evaluates it once, at one geometry. The planner collects the requests, adds their dependencies and hands the set to the theory queue as one call: a gradient needs the energy of its state, a frequency needs the gradient, absorption and transition ask for transition dipoles, a numerical frequency needs only energies and gradients. The coupling and transition tokens take their pairs from the molecule section, where ci and coupling declare the state ladder and the pairs returned.

The set is spelled into the token order the skills expect (energy, gradient, nac, soc, dipole, transition, frequency), and each skill maps it onto one of its profiles. A combination the skill has no profile for is refused at planning, in a message naming the jobtype string and the profile it resolved to; a density functional run asking for soc is refused this way.

```yaml
# Gradient and nonadiabatic coupling in one evaluation, C2H4
# Run: pyrai3md ethylene_grad_nac.yaml
 
control:
  title: ethylene_grad_nac
  jobtype: gradient nac
  theory: [molcas]
  driver: main
 
atoms: [C, C, H, H, H, H]
coordinates:
  - [-1.120191050,  0.000000000, -2.661941306]
  - [-1.120191050,  0.000000000, -1.331783306]
  - [-0.590584410,  0.756356668, -3.234891306]
  - [-1.649797690, -0.756356668, -3.234891306]
  - [-1.876547718, -0.529606640, -0.758833306]
  - [-0.363834382,  0.529606640, -0.758833306]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: 2
  spin: 0
  coupling: 1 2
 
molcas:
  method: casscf
  basis: ano-s-mb
  nact_elec: 2
  nact_orb: 2
  inactive_orb: 7
```

A non-iterative run ends with one result block in the log. The energy is always present, because it is the run's progress. Verbose 1 adds the structure and the cell; verbose 2, the default, adds the gradient, couplings, dipole, transition dipole, frequency and thermochemistry sections and the velocities, each only when requested. Verbose 3 adds the Hessian and a not-computed line for every absent property. Coupling sections are titled by pair, and a multi-center run prints them per center.

## 6.3. Geometry optimization

jobtype optimize runs a search for one structure, and optimize.method selects the algorithm. [Section 5.4](05-04-optimize.html) is the keyword reference; the path names are described in [Section 6.4](#64-reaction-paths-mep-irc-neb).

| Method | Character | Requirements |
| :--- | :--- | :--- |
| minima | Inverse-Hessian BFGS step under a trust radius, RFO optional. | Energies and gradients. The default. |
| gdiis | DIIS extrapolation preconditioned by the BFGS model. | As minima. No RFO variant. |
| ts | Partitioned RFO climbing one mode and descending the rest. | An initial Hessian before the first step; Bofill updates after it. |
| meci | Two-surface crossing search, branching plane from the gradient difference and the derivative coupling. | A pair (state_i, state_j) and a coupling for both plane directions. |
| mecp | The same machinery across multiplicities. | A pair. The coupling is not used. |

A search ends when five criteria hold at the same step; the progress block prints each threshold beside its value with a True or False judgement. The defaults are repeated here as the reference a run is compared with.

| Keyword | Default | Quantity and unit |
| :--- | :--- | :--- |
| opt_energy | 1.0e-6 | Change in the electronic energy, Hartree. |
| opt_grms | 3.0e-4 | Root-mean-square gradient, Hartree/Bohr. |
| opt_gmax | 4.5e-4 | Largest gradient component, Hartree/Bohr. |
| opt_drms | 1.2e-3 | Root-mean-square atomic step, Bohr. |
| opt_dmax | 1.8e-3 | Largest atomic step component, Bohr. |

A variable-cell run is judged twice: the atomic half keeps those five criteria, and the lattice half is read on its own pair, a stress in bar and a dimensionless strain. Both must clear their thresholds, because a box under stress has not converged however quiet the atoms are.

| Keyword | Default | Quantity and unit |
| :--- | :--- | :--- |
| opt_csrms | 40.0 | Root-mean-square cell stress, bar. |
| opt_csmax | 60.0 | Largest cell stress component, bar. |
| opt_cerms | 5.0e-5 | Root-mean-square cell strain, dimensionless. |
| opt_cemax | 7.0e-5 | Largest cell strain component, dimensionless. |

The step is bounded by a trust radius that belongs to the algorithm: 0.3 Bohr to start for minima and gdiis, 0.05 for the saddle search, 0.5 or 0.1 as a ceiling and 1e-4 as a floor. The radius is resized on the ratio of predicted to returned energy change, quartered below a quarter, doubled above three quarters and otherwise left alone so a run cannot oscillate. opt_trust_radius overrides the start; opt_use_rfo takes the step from the augmented Hessian instead of the BFGS model.

The curvature model starts as the identity, is updated by the BFGS formula on accepted pairs, skips an update whose curvature product is non-positive, and resets after repeated failure. The saddle search mixes the rank-one and symmetric-Broyden updates by Bofill's rule, so its model can acquire a negative eigenvalue and keep it. A step is accepted on the actual energy: a rise clearing a small floor is rejected, the geometry restored, the radius shrunk and a shorter step taken along the same direction, and four rejections in a row stop the run. Saddle and crossing searches skip that test. A run ending on maxit is marked failed.

optimize_cell adds the six movable lattice entries to the stepped vector, with the atoms riding the box at fixed fractional coordinates and the stress conjugate to each entry as its force. opt_pressure, in bar, makes the objective the enthalpy E + pV; without it the run asks for a zero-stress box. opt_cell_coupling chooses between an isotropic scale of the box and six independent entries. Either keyword without optimize_cell is refused, and so is a variable-cell run whose backend declares no stress, or a saddle whose curvature comes from the backend.

With driver main the framework steps and writes opt.xyz (latest structure), opt_geom.xyz (a frame per accepted step) and opt_status.txt (one row per step: energy, change, displacement and gradient readings, then the cell columns). With a skill driver the engine iterates and the framework writes the two xyz files from the structure returned through fin_coord; opt_status.txt is not written, and the outcome is settled from opt_converge.

```yaml
# Minimum-energy conical intersection of two singlets, C2H4
# Run: pyrai3md ethylene_meci.yaml
 
control:
  title: ethylene_meci
  jobtype: optimize
  theory: [molcas]
  driver: main
 
optimize:
  method: meci
  state_i: 0
  state_j: 1
  maxit: 60
  opt_grms: 1.0e-3
  opt_gmax: 2.0e-3
 
molecule:
  charges: 0
  multiplicity: 1
  pbc: [0, 0, 0]
  ci: 2 1
  spin: 0 0
  coupling: 1 2
 
molcas:
  method: casscf
  basis: ano-s-mb
  nact_elec: 2
  nact_orb: 2
  inactive_orb: 7
```

## 6.4. Reaction paths (MEP, IRC, NEB)

A path is asked for with optimize.method mep, irc or neb: the keyword selects it, not the jobtype, and the run enters the path workflow while each name refuses a driver other than main for the reason given in [Section 6.1](#61-job-types-and-workflow-resolution). A walker, serving mep and irc, evaluates one probe at a time on the calculator's state, because a probe's gradient decides the next probe. A band, serving neb, evaluates every image of a round as one batch, so images run in parallel under multipool, each in its own runtime directory.

A minimum energy path walks a mass-weighted hypersphere of radius path_radius re-centred on every accepted point, so one point is one arc length. The sphere model is the Gonzalez-Schlegel construction: the probe is held on the shell and rotated towards minus the tangential gradient, and the rotation shrinks whenever the energy fails to fall. The relu model minimizes the energy plus a one-sided penalty that vanishes inside the shell. A point ends when its reading crosses path_inner_tol within path_inner_maxit probes; the walk ends at a stationary point, at an uphill point, whose predecessor is the minimum, or at its step bound.

An IRC is that walk seeded one radius along the imaginary mode of a transition state. irc_mode hessian takes the mode from the framework's Hessian, run with the vibrational analysis off, and irc_mode molden reads it from the file named by irc_mode_file. irc_direction both runs one walker per side from the one saddle, restoring it before each branch; forward or backward runs one side. The names label the ends, because a mode read from a file has an arbitrary sign.

A band relaxes a chain of images between the structures the input names. The force on an interior image is a tangent, the energy-weighted average of the two segment directions, a parallel spring of constant neb_spring that equalizes the spacing, and the physical force with only its perpendicular component kept, so the band does not slide off the path. The endpoints carry nothing, and relaxation is FIRE at the initial timestep neb_step. With neb_climb, the first round whose largest perpendicular force falls below neb_climb_threshold inverts the highest image's parallel force and drops its spring, one-way. A band ends when the interior forces are below opt_grms and opt_gmax, with a climbing image judged on its own force.

linear interpolation interpolates fractional coordinates, which keeps every image of a periodic band inside the cell; geodesic interpolates in a Morse-scaled internal-coordinate metric, the better guess when connectivity changes, and needs two interior images. The validator refuses each keyword group on a method that cannot read it, a band without both endpoints, a geodesic band with fewer than two images, a climb threshold without the climb switch, and irc_mode molden without a file.

A path writes its own records: path.xyz holds the latest structure, path_geom.xyz every accepted frame or the band once per round, path_profile.txt one row per accepted point or image, and path_status.txt one row per evaluation or round, with path_forward.xyz and path_backward.xyz for a two-sided IRC. None is opt_status.txt: that reader classifies a record by its column count, and a path table under that name would be read as another writer's. A periodic frame carries its own box, and for a band that is the image's box.

```yaml
# Nudged elastic band between the HCN and HNC minima
# Run: pyrai3md path_neb.yaml
 
control:
  title: path_neb
  jobtype: optimize
  theory: [pyscf]
  driver: main
 
optimize:
  method: neb
  maxit: 100
  neb_reactant: path_reactant.xyz
  neb_product: path_product.xyz
  neb_images: 8
  neb_interpolate: linear
  neb_spring: 0.01
  neb_step: 0.1
  neb_climb: true
  neb_climb_threshold: 1.0e-2
  opt_grms: 3.0e-3
  opt_gmax: 4.5e-3
 
atoms: [H, C, N]
coordinates:
  - [0.000000000, 0.000000000, 1.055827000]
  - [0.000000000, 0.000000000, -0.010776000]
  - [0.000000000, 0.000000000, -1.145051000]
 
molecule:
  charges: 0
  multiplicity: 1
 
pyscf:
  method: hf
  basis: sto3g
  scratch_path: ./scratch
```

## 6.5. Ab initio MD and NAMD

jobtype md and surface_hopping select the same dynamics workflow, which integrates the nuclei with velocity Verlet in two half kicks around one electronic evaluation per step. A step applies the first half of the ensemble operator, propagates and wraps the coordinates, evaluates the target states, merges deviations, completes the velocity from the previous and current gradients, corrects coupling phases, projects the constraints, corrects the velocities, applies the second half of the ensemble operator, projects again after a cell rescale, samples the kinetic energy, records the pre-hop state, attempts a hop, judges the guards, synchronizes the tasks, advances the counter and writes the checkpoint. The kinetic energy is sampled last because every operation before it may write velocities. Step 1 records the unpropagated condition, so the first propagated frame is step 2.

The ensemble keyword selects nve, nvt or npt. An nve run requires the thermostat off and refuses dissipation, nvt requires a thermostat, and npt requires both. Undefined pairings are refused at validation rather than left to drift:

| Barostat | Thermostats accepted | Note |
| :--- | :--- | :--- |
| berendsen | nose-hoover, nose-hoover-chain, csvr | Weak coupling; constrains no thermostat choice. |
| parrinello-rahman | nose-hoover, nose-hoover-chain | The extended cell is derived with a particle chain. |
| mttk | nose-hoover, nose-hoover-chain | Parrinello-Rahman plus a chain on the cell momentum. |
| scr | nose-hoover, nose-hoover-chain, csvr | First-order stochastic cell equation; isotropic only. |

The starting point comes from initcond. 0 or none uses the geometry and velocities carried by the state; read or load takes ninitcond frames from &lt;title&gt;.init.xyz, each atom line carrying element, coordinates and velocities; wigner and boltzmann sample from &lt;title&gt;.freq.&lt;format&gt;, where the format is molden, g16, orca or bagel, seeded from gl_seed and the temperature. Fewer frames than ninitcond is an error. The first step's kinetic energy is set by excess, scale or target, one request written three ways, and two at once is refused. A restarted segment applies neither the draw nor the scaling.

Hopping is selected by sfhp, and the algorithms differ in what they need:

| sfhp | Data requirement | Character |
| :--- | :--- | :--- |
| nosh | Energies and gradients of the active state. | No hopping; the default, a pass-through in the step. |
| gsh | Gradients of every state, so activestate 0. | Global hopping from a pure-state density; no decoherence or substeps. |
| fssh | Energies, gradients and a coupling, as a NAC vector or from the energy curvature. | Fewest switches, with substeps, decoherence and an acceptance test. |
| lzsh | State energies, plus the previous step couplings for a cross-spin hop. | Landau-Zener; the active state is irrelevant and every hop rescales the velocity isotropically. |

The switches refine any of them. nactype names the coupling source of an FSSH run: nac takes the derivative coupling vector, ktdc estimates it from the energy curvature, needing no vector. phasecheck corrects the coupling sign between steps, substep sets the electronic substeps of one nuclear step, and rk4 picks forward Euler or a fourth-order integrator for the amplitudes. deco sets decoherence and off disables it. adjust picks the velocity policy after an accepted hop, none, a global rescale or a rescale along the coupling; reflect picks what a frustrated hop does. maxh bounds the highest state a hop may target. gap and gapsoc are the windows in eV for internal-conversion and intersystem-crossing hops. dosoc requests the couplings, and addsoc biases every coupling of one evaluation by a constant in cm-1.

Four resident guards end a trajectory, each after the offending frame is written. md_maxpop bounds the population, and md_maxenergy, md_maxgrad, md_maxnac and md_maxsoc bound the prediction deviation of a machine-learned surface, an unset value leaving that channel unjudged. A tracked geometry, track_type with track_index and track_thrhd in Angstrom, ends the run when its bound is crossed, track_stop choosing between a report and a stop. Cell divergence and any non-finite number are always fatal. The outcome is carried as a stop code:

| Code | Meaning |
| :--- | :--- |
| 0 | Ran to the planned end. |
| 1 | Ill population: outside the bound, or not a number. |
| 2 | A tracked geometry bound was crossed. |
| 3 | A prediction error above one of the md_max thresholds. |
| 4 | The calculation failed. |

A checkpoint &lt;title&gt;.rt is written on the cadence named by checkpoint, and the last step always writes one. It carries the state, the counter, the generator state and the runtime record, and a run is continued by launching that file, which bypasses the input, planning and dispatch stages. restart and addstep are refused keywords, because step already means the steps the new process adds. With ninitcond above one each trajectory is dispatched as a child with its own runtime directory, files and log.

```yaml
# NAMD with FSSH on C2H4, Molcas CASSCF, NVT thermostat
# Run: pyrai3md ethylene_namd.yaml
 
control:
  title: ethylene_namd
  jobtype: md
  theory: [molcas]
  driver: main
  gl_seed: 20260926
 
atoms: [C, C, H, H, H, H]
coordinates:
  - [-1.120191050,  0.000000000, -2.661941306]
  - [-1.120191050,  0.000000000, -1.331783306]
  - [-0.590584410,  0.756356668, -3.234891306]
  - [-1.649797690, -0.756356668, -3.234891306]
  - [-1.876547718, -0.529606640, -0.758833306]
  - [-0.363834382,  0.529606640, -0.758833306]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: 3
  spin: 0
  coupling: 1 2, 1 3, 2 3
 
md:
  root: 2
  initcond: wigner
  ninitcond: 2
  temperature: 298.15
  ensemble: nvt
  thermostat: nose-hoover-chain
  thermostat_chain: 3
  thermostat_tau: 100.0
  sfhp: fssh
  nactype: nac
  substep: 20
  deco: 0.1
  gap: 0.5
  step: 2000
  size: 20.67
  buffer: 500
  checkpoint: 100
  md_maxenergy: 0.05
  md_maxgrad: 0.15
 
molcas:
  method: casscf
  basis: ano-s-mb
  nact_elec: 4
  nact_orb: 4
  inactive_orb: 7
```

## 6.6. Hessian, frequencies and thermochemistry

The four force-constant tokens reach one workflow that answers two questions: which matrix the run wants, spelled by the jobtype token, and who produces it, spelled by hess_driver, which is not control.driver. The token also decides whether the matrix is read.

| Jobtype | Matrix route | Vibrational reading |
| :--- | :--- | :--- |
| hessian, hess | auto: the backend matrix when the evaluation returns one, the displacement batch when it does not. | Off unless hess_vibration asks for it. |
| frequency, freq | auto, as above. | On: frequencies, modes and thermochemistry. |
| num_hess | numerical: the framework displaces and differences. | Off unless hess_vibration asks for it. |
| num_freq | numerical, as above. | On. |

hess_mode and hess_driver are cross-checked. main displaces and differences the gradients the framework already has, so main with analytic is refused, and so is main with auto, because auto is the statement that no producer has been settled on. skill with numerical is legal and means the engine displaces its own atoms. A route the run did not get is written into the log, with the number of evaluations an auto fallback will spend.

The numerical route is a batch of central differences: each Cartesian component of each mobile atom is displaced by plus and minus hess_step, 0.005 Angstrom by default, and the gradient difference divided by twice the step in Bohr is one column, so the default step costs 2 times 3N evaluations for N mobile atoms, one child run each in its own runtime directory. A displaced point without a gradient fails the run and raises, naming the atom and axis. The matrix is assembled on the mobile block and symmetrized there, the held rows zero.

hess_atoms selects a partial Hessian in the one-based input numbering. During a search the held set belongs to the constraint plan, because fix is the one statement about what may move, so a hess_atoms beside a fix must agree with it: a mismatch is refused, and so is hess_atoms with no fix behind it, since the matrix would carry zero rows on atoms the run may move. The batch works per component, and the analysis re-reads the mobile set from the displacement rows so a held atom cannot appear as a soft mode.

The analysis mass-weights the matrix and removes the rigid directions by rank, so a linear molecule keeps four modes and a single atom none. Frequencies are signed, imaginary modes negative, modes normalized. The thermochemistry is the rigid rotor harmonic oscillator model with the quasi-RRHO free-rotor blend below hess_cutoff; hess_temperature, hess_pressure, hess_scale and hess_symmetry set the standard state. The result carries zero_point_energy, internal_energy, enthalpy, entropy, free_energy, heat_capacity_cv and heat_capacity_cp. The log prints frequencies, modes, corrections and the matrix at verbose 3, and hess_molden writes &lt;title&gt;.freq.molden, the file an initial-condition sampler reads.

Two restrictions follow. A vibrational analysis cannot be combined with a variable cell, because the analysis builds frequencies on a mass-weighted basis and the six lattice entries have no mass to be weighted by; a box's curvature is read by the search that asked for it. And an analytical matrix cannot be asked of a queue holding more than one backend, because no single skill owns a composite's second derivative.

## 6.7. Reading finished calculations

jobtype read extracts labels from calculations that have already run. It starts no engine, builds no calculator and evaluates nothing: it opens the paths the read section names, hands each to the file reader of the single theory binding, and returns the frames to its caller. No training-data file, trajectory or result artifact is written, so the working directory is untouched. The dataset belongs to the training-data tool, which calls this flow; [Section 10.1](10-machine-learning-potentials.html#101-training-data) describes it.

The read section names the paths, an optional project stem overriding the one discovered from the files, an optional profile used to parse an optimization or dynamics directory, and one flag per label: energy, gradient, nac and soc. An entry may be a directory, an output file, or a file listing one path per line. A read run binds exactly one theory skill, no reference and no multiscale partition; in-process engines such as pyscf, qdm and nn_pes are refused by name. Building a label set for a later training run is the recommended use. [Section 5.10](05-10-read.html) lists the section.

```text
&control
title read_run
jobtype read
theory molcas
 
&read
path dirs.txt
energy 1
gradient 1
nac 0
soc 0
 
&molecule
charges 0
multiplicity 1
ci 1
spin 0
```

## 6.8. Artifacts and restart per workflow

Each workflow owns its file names, and this table is the single owner of them. Files are named after control.title and written in the run directory unless a row says otherwise. A dispatched child restamps its log to its own unit directory and leaves a runtime file and a result record there.

| Workflow | Files | Restart and notes |
| :--- | :--- | :--- |
| Single point or property run | &lt;title&gt;.log carries the result block. | Nothing to restart; &lt;title&gt;.rt exists only in a dispatched child. |
| optimize | opt.xyz, opt_geom.xyz, opt_status.txt, and the log. | Rewritten by every run. A skill-driven run writes the two xyz files only. |
| path (mep, irc, neb) | path.xyz, path_geom.xyz, path_profile.txt, path_status.txt, and path_forward.xyz with path_backward.xyz for a two-sided IRC. | No opt_status.txt. A resumed path walks again from its new start. |
| md and surface_hopping | &lt;title&gt;.md.xyz, .md.velo, .md.energies, .md.pop, .md.ensemble for nvt and npt, .md.stress when the backend supplied one, the &lt;title&gt;.sh.\* set on hop steps, the charge files when charge management is on, and &lt;title&gt;.rt. | Continue by launching &lt;title&gt;.rt. Each trajectory child gets its own runtime/&lt;skill&gt;_runtime_&lt;idx&gt;/ with its files, log and record. |
| hessian, frequency, num_hess, num_freq | &lt;title&gt;.freq.molden when hess_molden is true, and the matrix, modes and corrections in the log. | The molden file is the input an initial-condition sampler looks for. |
| train | The model folder named by modeldir, one subfolder per fitted role, holding hyper_v&lt;i&gt;.json, scaler_v&lt;i&gt;.json, weights_v&lt;i&gt;.weights.h5, fiterr_valid_v&lt;i&gt;.npy, fitlog_&lt;i&gt;.txt and fit_stats/, plus &lt;project&gt;_train.summary.txt. | A continuation is declared with train_mode retraining and cannot change the architecture. |
| prediction | {project}_{profile}.{channel}.pred.txt, {project}_{profile}.frames.txt and the summary, in the run directory. | Kept out of the calculation directory so a scratch cleanup cannot delete them. |
| adaptive | The &lt;title&gt;.adaptive tree with &lt;title&gt;.adaptive.json, the per-iteration data, model and selection records, and the log. | The summary carries every iteration record, so a round reads back without its trajectory. |
| search | summary.csv, crashed.txt when a point crashed, and the grid record in the log. | summary.csv is the ranking; a rerun skips complete model folders and re-ranks them. |
