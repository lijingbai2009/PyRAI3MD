---
title: "2. Features"
nav_order: 2
---

This chapter summarises what the program does. Each item is one or two sentences and points to the chapter that explains it.

## 2.1. Workflows

The jobtype selects the workflow. A property jobtype evaluates one capability set, while an advanced jobtype owns an iteration. The table lists every jobtype the validator accepts; [Section 6.1](06-workflows.html#61-job-types-and-workflow-resolution) explains when to use which.

| Jobtype | What it does |
| :--- | :--- |
| energy, gradient, nac, soc | single evaluation of the requested properties |
| dipole, transition, absorption | transition moments and spectra |
| hessian, frequency | force constants, vibrations and thermochemistry |
| num_hess, num_freq | numerical force constants from displaced gradients |
| optimize | minima, GDIIS, transition state, MECI, MECP, variable cell |
| optimize with mep/irc/neb | reaction paths and bands |
| md, surface_hopping | ab initio MD and nonadiabatic dynamics |
| adaptive | iterative reference labelling, training and screening |
| train, prediction | machine-learning potential fitting and scoring |
| search | hyperparameter grid evaluation and ranking |
| read | label extraction from finished calculations |

## 2.2. Calculation engines

Ten skills ship with the package. The capability list is the vocabulary the planner uses to decide whether a request can be served; a jobtype that needs a capability the chosen engine does not declare is refused before the run starts. The role column shows where an engine can sit in a method queue: qm is the high layer, qm2 the middle layer and mm the classical outer layer.

| Engine | Capabilities | Roles | PBC | Profiles |
| :--- | :--- | :--- | :--- | :--- |
| pyscf | energy, gradient, frequency, geometry_optimization, ir_spectrum, uv_spectrum, nmr_spectrum, analysis | qm, qm2 | no | 39 |
| molcas | energy, gradient, nonadiabatic_coupling, geometry_optimization, abinitio_dynamics, surface_hopping, absorption_spectrum, soc_coupling, vibrational_frequency | qm | no | 15 |
| bagel | energy, gradient, nonadiabatic_coupling, geometry_optimization, vibrational_frequency, dipole, transition_dipole | qm | no | 14 |
| orca | energy, gradient, dipole, transition_dipole, nonadiabatic_coupling, soc_coupling, geometry_optimization, vibrational_frequency | qm | no | 14 |
| openqp | energy, gradient, nonadiabatic_coupling, soc_coupling, geometry_optimization, nonadiabatic_dynamics, vibrational_frequency | qm | no | 17 |
| xtb | energy, gradient, stress | qm, qm2, mm | yes | 8 |
| cp2k | energy, gradient, stress, transition_dipole, geometry_optimization, abinitio_dynamics | qm | yes | 9 |
| lammps | energy, gradient, stress, geometry_optimization, classical_dynamics | mm | yes | 9 |
| qdm | energy, gradient, nac, md | qm | no | 24 |
| nn_pes | energy, gradient, nac, soc | qm | no | 25 |

The engines divide into three groups. PySCF, OpenMolcas, BAGEL, ORCA and OpenQP are molecular quantum-chemistry codes. xTB and CP2K add semi-empirical and periodic density functional work. LAMMPS is the classical engine and also supplies the QM/MM field, while QDM provides the analytic two-state models used to validate the surface-hopping code. The tenth skill, nn_pes, evaluates a trained neural-network potential. [Sections 5.15](05-15-pyscf.html) to [5.24](05-24-nn_pes.html) document each skill.

## 2.3. Machine-learning potentials

The nn_pes skill trains an ensemble of neural networks on energies, gradients and, when requested, nonadiabatic and spin-orbit couplings. It serves the ensemble inside any workflow that needs a potential, and it reports the spread between members as an uncertainty. Training, prediction and hyperparameter search are workflows of their own, and adaptive sampling closes the loop by running trajectories on the current model, selecting uncertain structures, labelling them with a reference engine and retraining. Chapter 10 covers the data format and the loop.

## 2.4. Multiscale calculations

A method queue of two entries is an ONIOM2 composite, or additive QM/MM when multiscale_type is qmmm; three entries are ONIOM3. Regions come from highlevel and midlevel, embedding charges come from point charges or a charge file, and a multi-centre run repeats the whole plan for each centre with optional charge self-consistency. Chapter 7 gives the energy expressions and the supported jobtypes.

## 2.5. Dynamics and nonadiabatic dynamics

The md workflow integrates with velocity Verlet in NVE, NVT or NPT, with Nose-Hoover, Nose-Hoover-chain or CSVR thermostats and Berendsen, Parrinello-Rahman, MTTK or stochastic-cell-rescaling barostats. Initial conditions can be read or sampled from a frequency file. Three surface-hopping algorithms are available: fewest switches with explicit or curvature-driven couplings, Granucci-Persico style hopping on all-state gradients, and Landau-Zener hopping that needs energies only. Guards stop a trajectory when a population, an energy, a gradient, a coupling or a tracked geometry leaves its window. [Section 6.5](06-workflows.html#65-ab-initio-md-and-namd) explains the step order and every switch.

## 2.6. Periodic systems and constraints

A cell is given by pbc, cell and lattice_vector. Engines that declare periodic support run minimum-image dynamics and can supply stress, which lets the optimizer relax the cell at a target pressure. Constraints are a separate block: harmonic biases, confining walls, fixed atoms, SHAKE and metadynamics act on dynamics and, where meaningful, on optimization. Chapters 8 and 9 cover both.

## 2.7. Parallel and remote execution

Two independent axes control execution. multitask runs independent units in a local process pool; remotetask runs them as child runtimes, normally through Slurm, with a generated script. multipool sets the worker count of a stage, and the adaptive workflow can override both axes per stage. Chapter 11 describes the axes, the scheduler section and the runtime files that make a run restartable.

## 2.8. Output, testing and tools

Every run writes a log named after its title and, depending on the workflow, trajectory, status, model or summary files. Chapter 12 lists them and the analysis tools that read them. The package ships a qualification suite that runs framework tests, shipped examples, skill certification cases and benchmarks through the same entry points a user uses; Chapter 14 explains how to run it locally or on a cluster and how to verify a finished round.
