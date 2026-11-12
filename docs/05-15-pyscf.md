---
title: "5.15. PYSCF"
nav_order: 15
parent: "5. Keyword sections"
---

The pyscf section configures the PySCF skill, the in-process backend for Hartree-Fock, Kohn-Sham density functional theory, MP2, TDDFT and TDA. It returns ground- and excited-state energies, analytic gradients, numerical Hessians and frequencies, optimized minima and transition states, MO, ESP and population analysis, IR and NMR spectra, and UV/vis absorption for the time-dependent methods. Solvation runs through the PySCF solvent models, so a named solvent selects IEF-PCM while gas, vacuum, none or an empty value leaves the calculation in the gas phase.

PySCF is a Python library rather than a program: no input deck is written and no log is parsed, so the numbers come straight from live PySCF objects. The section therefore has no executable to point at, and pyscf_path stays import.

The thirty-nine profiles choose the chain and each one locks its method. The hf and dft families provide energy, gradient, hessian, opt_min, opt_ts, mo, esp, population, ir and nmr; the mp2 family provides energy, gradient and the three analysis chains; the tddft and tda families provide energy, gradient, opt_min, uv_vis and the three analysis chains. The prefix of profile and the value of method must agree.

Nothing outside the Python environment is required: the interpreter that runs the framework must be able to import pyscf, and the optional pyscf/properties extension supplies the infrared and shielding modules behind the ir and nmr profiles. No auxiliary input file is staged; chk_file names an SCF checkpoint whose contents become the initial guess whenever the file already exists. The remaining keys at the end of the table are shared with the framework sections, and the skill receives them from &MOLECULE and &MD.

The skill exposes the following profiles. A profile fixes the method and the set of properties a calculation returns.

| Profile | Capability | Method |
| :--- | :--- | :--- |
| hf_energy | energy | hf |
| hf_gradient | gradient | hf |
| hf_hessian | frequency | hf |
| hf_opt_min | geometry_optimization | hf |
| hf_opt_ts | geometry_optimization | hf |
| hf_mo | analysis | hf |
| hf_esp | analysis | hf |
| hf_population | analysis | hf |
| hf_ir | ir_spectrum | hf |
| hf_nmr | nmr_spectrum | hf |
| dft_energy | energy | dft |
| dft_gradient | gradient | dft |
| dft_hessian | frequency | dft |
| dft_opt_min | geometry_optimization | dft |
| dft_opt_ts | geometry_optimization | dft |
| dft_mo | analysis | dft |
| dft_esp | analysis | dft |
| dft_population | analysis | dft |
| dft_ir | ir_spectrum | dft |
| dft_nmr | nmr_spectrum | dft |
| mp2_energy | energy | mp2 |
| mp2_gradient | gradient | mp2 |
| mp2_mo | analysis | mp2 |
| mp2_esp | analysis | mp2 |
| mp2_population | analysis | mp2 |
| tddft_energy | energy | tddft |
| tddft_gradient | gradient | tddft |
| tddft_opt_min | geometry_optimization | tddft |
| tddft_uv_vis | uv_spectrum | tddft |
| tddft_mo | analysis | tddft |
| tddft_esp | analysis | tddft |
| tddft_population | analysis | tddft |
| tda_energy | energy | tda |
| tda_gradient | gradient | tda |
| tda_opt_min | geometry_optimization | tda |
| tda_uv_vis | uv_spectrum | tda |
| tda_mo | analysis | tda |
| tda_esp | analysis | tda |
| tda_population | analysis | tda |

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&PYSCF**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Selects the run arm: local starts the calculation in a subprocess on the current machine, and slurm writes a batch script and submits it. The default… |
| project | PYSCF | Sets the artifact stem. The log, checkpoint, orbital and geometry files are named after it, for example &lt;project&gt;.log, &lt;project&gt;.chk… |
| profile | dft_energy | Selects one of the thirty-nine profiles, named &lt;method&gt;_&lt;task&gt;, for example dft_energy or tddft_gradient. The profile fixes the complete step chain… |
| method | dft | Sets the electronic-structure method: hf, dft, mp2, tddft or tda. It selects the SCF reference class, the post-SCF service and the property modules a… |
| pyscf_path | import | Names how PySCF is reached. import loads the library from the running interpreter, which is the only mode the framework uses; there is no binary… |
| pyscf_project | PYSCF | Carries the project name in the spelling the framework forwards. A non-empty value overwrites project and the two are kept in step. |
| scratch_path | '' | Sets the parent directory of the per-term calculation directory. An empty value leaves the placement to the framework scratch root. |
| thread | 1 | Sets the OpenMP thread count written to OMP_NUM_THREADS. The variable is latched the first time the process needs it, so a later term asking for… |
| pyscf_thread | 1 | Carries the thread count in the spelling the framework forwards. The larger of pyscf_thread and thread is applied, and the retired spelling… |
| slurm_time | 01:00:00 | Sets the wall-clock limit of the batch job in HH:MM:SS. The default is 01:00:00. |
| slurm_partition | '' | Names the SLURM partition. An empty value writes no --partition directive, so the cluster default answers. |
| slurm_ntasks | 1 | Sets the number of SLURM tasks of the batch job. The default is 1. |
| slurm_memory | 4000 | Sets the memory request of the batch job in megabytes. The default is 4000. |
| slurm_command | '' | Adds extra commands to the submission line of the batch job, for example an account or a constraint. |
| keep_tmp | 0 | 0 removes the calculation directory once the results are collected; 1 keeps the log, checkpoint and Molden files. |
| functional | pbe0 | Sets the exchange-correlation functional for a dft, tddft or tda run from the 140 names in the skill list, for example pbe0 or b3lyp. The hf and mp2… |
| basis | ccpvdz | Sets the orbital basis from the 362 names in the skill list. Dashes and spaces are stripped, so cc-pvdz is written ccpvdz and 6-31g\* becomes 631g\*. |
| density_fitting | true | Turns the resolution-of-the-identity approximation on or off through mf.density_fit(). It is switched off automatically for the tddft_gradient and… |
| non_local_correction | true | Turns the non-local correlation correction of the functional on through mf.nlc. Given false, mf.nlc is disabled and the functional runs without the… |
| dispersion_correction | '' | Sets the Grimme dispersion correction handed to PySCF as mf.disp: d3bj, d3zero, d3bjm, d3zerom, d3op or d4. An empty value adds no correction, and… |
| scf_converge | 1e-09 | Sets the SCF energy convergence threshold in Hartree and is handed to PySCF as conv_tol. The default is 1e-9. |
| maxit_scf | 100 | Sets the SCF cycle cap and is handed to PySCF as max_cycle. The default is 100 iterations. |
| td_converge | 1e-06 | Sets the convergence threshold of the TDDFT or TDA response iterations. The default is 1e-6. |
| chk_file | '' | Names the SCF checkpoint file. An existing file is read as the initial guess; an empty value writes… |
| save_mo | false | Writes &lt;project&gt;.mo.molden with the orbitals of the last SCF step when true. The default false skips the file. |
| save_norm_mode | false | Writes &lt;project&gt;.freq.molden with the frequencies and normal modes of a hessian, ir or nmr run when true. The default false skips the file. |
| opt_converge | false | Reports whether the geometry optimization converged. The skill writes it as a result and the framework reads it as the run verdict; it is not a… |
| opt_energy | 1e-06 | Sets the energy-change threshold of the geometry optimizer in Hartree, passed as convergence_energy. The default is 1e-6. |
| opt_grms | 0.0003 | Sets the root-mean-square gradient threshold of the optimizer in Hartree/Bohr, passed as convergence_grms. The default is 3e-4. |
| opt_gmax | 0.00045 | Sets the largest-gradient threshold of the optimizer in Hartree/Bohr, passed as convergence_gmax. The default is 4.5e-4. |
| opt_drms | 0.0012 | Sets the root-mean-square displacement threshold of the optimizer in Bohr, passed as convergence_drms. The default is 1.2e-3. |
| opt_dmax | 0.0018 | Sets the largest-displacement threshold of the optimizer in Bohr, passed as convergence_dmax. The default is 1.8e-3. |
| opt_sp | true | Runs a single-point energy after the optimization so the same record carries the final energy and dipole; given false the run returns the optimized… |
| opt_freq | false | Runs a numerical Hessian and frequency analysis after a converged ground-state optimization. It fires only for target state 0 and only when the… |
| maxit_opt | 100 | Sets the step cap of the geometry optimizer used by the opt_min and opt_ts profiles. The default is 100 steps. |
| maxit_md | 1 | Sets the step cap of a molecular-dynamics run. The skill registers no dynamics profile, so the value is recorded for the contract and no MD loop is… |
| grid_level | 3 | Selects the DFT integration grid level of the Kohn-Sham steps: 1 to 5 are the conventional values and 3 is the shipped default. |
| ir_width | 100 | Sets the half-width of the Gaussian broadening of the IR trace in cm-1. The default is 100.0. |
| ir_scale | 1 | Sets the scaling factor applied to the harmonic frequencies before the IR trace is drawn. The default 1.0 leaves them unscaled. |
| uv_width | 100 | Sets the half-width of the Gaussian broadening of the UV/vis spectrum in nm. The default is 100.0. |
| nics_ring † | [0] | Reserved NICS request: the 1-based ring atom indices whose axis would receive ghost centres. The key is declared for the NMR contract and documented… |
| nics_scan † | [0] | Reserved NICS request: the distances in Angstrom along the ring axis of nics_ring. The shipped profiles leave it at [0.0]. |
| thermostat † | nose-hoover | Selects the thermostat of a dynamics run, nose-hoover by default, with nve and berendsen as the other names. The skill registers no dynamics profile… |
| barostat † | berendsen | Selects the barostat of a constant-pressure dynamics run, berendsen by default. The skill registers no dynamics profile, so the value never reaches a… |
| ensemble † | nve | Names the thermodynamic ensemble of a dynamics run: nve, nvt or npt. The skill registers no dynamics profile, so the value never reaches a… |
| charges | 0 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| multiplicity | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| ci | [1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| spin | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| coupling | [] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| pbc | [false, false, false] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| cell | [99, 99, 99, 90, 90, 90] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| lattice_vector | [[0, 0, 0], [0, 0, 0], [0, 0, 0]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| kpoint | [1, 1, 1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_states | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_nac_pair | [[]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_soc_pair | [[]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| verbose | 4 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| temperature | 298.15 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html). |
| pressure | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html). |
| solvent | gas | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| shape | ellipsoid | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| factor | 40 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| scale | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Selects the run arm: local starts the calculation in a subprocess on the current machine, and slurm writes a batch script and submits it. The default is local.

**project** - Sets the artifact stem. The log, checkpoint, orbital and geometry files are named after it, for example &lt;project&gt;.log, &lt;project&gt;.chk, &lt;project&gt;.mo.molden and &lt;project&gt;.opt.xyz.

**profile** - Selects the computation chain. The name is &lt;method&gt;_&lt;task&gt;, and the thirty-nine registered chains are hf_energy, hf_gradient, hf_hessian, hf_opt_min, hf_opt_ts, hf_mo, hf_esp, hf_population, hf_ir, hf_nmr, the same ten for dft, mp2_energy, mp2_gradient, mp2_mo, mp2_esp, mp2_population, and the seven tddft_ and tda_ chains: energy, gradient, opt_min, uv_vis, mo, esp and population. The profile alone decides which steps run, from the molecule and reference steps through gradient, hessian, optimize, optimize_ts, the analysis chains and the spectra. A profile whose prefix does not match method is refused, and the nac, soc, dipole and stress jobtypes have no profile here.

**method** - Sets the electronic-structure level and, with it, the reference class the chain starts from. hf runs a Hartree-Fock reference; dft runs the same reference with the exchange-correlation functional from functional; mp2 adds a second-order Moller-Plesset correction on the reference; tddft and tda run a time-dependent response on it, tda within the Tamm-Dancoff approximation. The capability table behind the profiles decides what each method may ask for: hf and dft have no UV/vis chain, mp2 has no Hessian or optimization chain, and tddft and tda have neither a Hessian, a transition-state optimization, IR nor NMR.

**pyscf_path** - Names how PySCF is reached. import loads the library from the running interpreter, which is the only mode the framework uses; there is no binary whose path could be given here.

**pyscf_project** - Carries the project name in the spelling the framework forwards. A non-empty value overwrites project and the two are kept in step.

**scratch_path** - Sets the parent directory of the per-term calculation directory. An empty value leaves the placement to the framework scratch root.

**thread** - Sets the thread count of the in-process run. The value is written to OMP_NUM_THREADS the first time the process needs the variable, because PySCF reads the setting once per process; a later term that asks for a different count keeps the latched value and says so on stderr rather than silently oversubscribing the node. The two spellings thread and pyscf_thread fold to their larger value, so neither can shrink a request made in the other.

**pyscf_thread** - Carries the thread count in the spelling the framework forwards. The larger of pyscf_thread and thread is applied, and the retired spelling num_thread is refused.

**slurm_time** - Sets the wall-clock limit of the batch job in HH:MM:SS. The default is 01:00:00.

**slurm_partition** - Names the SLURM partition. An empty value writes no --partition directive, so the cluster default answers.

**slurm_ntasks** - Sets the number of SLURM tasks of the batch job. The default is 1.

**slurm_memory** - Sets the memory request of the batch job in megabytes. The default is 4000.

**slurm_command** - Adds extra commands to the submission line of the batch job, for example an account or a constraint.

**keep_tmp** - 0 removes the calculation directory once the results are collected; 1 keeps the log, checkpoint and Molden files.

**functional** - Sets the exchange-correlation functional of a dft, tddft or tda run. The skill keeps a list of 140 names covering the LDA, GGA, meta-GGA, hybrid and range-separated functionals PySCF assembles, among them pbe, pbe0, b3lyp, b3lyp5, b3pw91, camb3lyp and wb97x; the list holds decomposed exchange and correlation parts as well, which is how the composite names are spelled without dashes. The value is compared case-insensitively. hf and mp2 carry no functional flag, so the key is ignored there.

**basis** - Sets the orbital basis. The skill keeps a list of 362 names, written without dashes because PySCF spells them that way: ccpvdz, augccpvtz, ccpvtzpp, def2svp, def2tzvpp, 631g\*, 6311g\*\*, sto3g, and the auxiliary jkfit and ri sets of the same families. A dummy centre X used for a ghost atom receives the minao basis. The value is compared case-insensitively.

**density_fitting** - Turns the resolution-of-the-identity approximation on or off through mf.density_fit(). It is switched off automatically for the tddft_gradient and tda_gradient chains, whose analytic gradients PySCF does not support under density fitting.

**non_local_correction** - Turns the non-local correlation correction of the functional on through mf.nlc. Given false, mf.nlc is disabled and the functional runs without the VV10 term.

**dispersion_correction** - Sets the dispersion correction model that PySCF dispatches on mf.disp. The accepted names are the versions the dftd3 and dftd4 packages provide: d3bj and d3zero for the Becke-Johnson and zero-damping D3 forms, d3bjm and d3zerom for their three-body variants, d3op for the optimised-power form, and d4 for D4. An empty value leaves the correction off; the model must not contradict a dispersion suffix carried by the functional name.

**scf_converge** - Sets the SCF energy convergence threshold in Hartree and is handed to PySCF as conv_tol. The default is 1e-9.

**maxit_scf** - Sets the SCF cycle cap and is handed to PySCF as max_cycle. The default is 100 iterations.

**td_converge** - Sets the convergence threshold of the TDDFT or TDA response iterations. The default is 1e-6.

**chk_file** - Names the SCF checkpoint. The path is expanded for a leading tilde and handed to PySCF as mf.chkfile; when the file already exists it is read as the initial guess, which is how a converged orbital set is carried from one run to the next. An empty value lets the skill build the name from project, profile, method, functional and basis inside the scratch directory.

**save_mo** - Writes &lt;project&gt;.mo.molden with the orbitals of the last SCF step when true. The default false skips the file.

**save_norm_mode** - Writes the frequency record as &lt;project&gt;.freq.molden. The writer does not copy PySCF's own convention: PySCF reports mass-weighted normalized modes (sqrt(amu) Bohr), while the framework's Molden reader and the initial-condition sampler follow the Molcas convention of unnormalized, un-mass-weighted Cartesian displacements, so each mode vector is divided by the square root of the atomic mass before it is written. The file also serves the front-end visualization.

**opt_converge** - Reports whether the geometry optimization converged. The skill writes it as a result and the framework reads it as the run verdict; it is not a setting that steers the optimizer.

**opt_energy** - Sets the energy-change threshold of the geometry optimizer in Hartree, passed as convergence_energy. The default is 1e-6.

**opt_grms** - Sets the root-mean-square gradient threshold of the optimizer in Hartree/Bohr, passed as convergence_grms. The default is 3e-4.

**opt_gmax** - Sets the largest-gradient threshold of the optimizer in Hartree/Bohr, passed as convergence_gmax. The default is 4.5e-4.

**opt_drms** - Sets the root-mean-square displacement threshold of the optimizer in Bohr, passed as convergence_drms. The default is 1.2e-3.

**opt_dmax** - Sets the largest-displacement threshold of the optimizer in Bohr, passed as convergence_dmax. The default is 1.8e-3.

**opt_sp** - Runs a single-point energy after the optimization so the same record carries the final energy and dipole; given false the run returns the optimized structure alone.

**opt_freq** - Runs a numerical Hessian and frequency analysis after a converged ground-state optimization. It fires only for target state 0 and only when the optimization converged.

**maxit_opt** - Sets the step cap of the geometry optimizer used by the opt_min and opt_ts profiles. The default is 100 steps.

**maxit_md** - Sets the step cap of a molecular-dynamics run. The skill registers no dynamics profile, so the value is recorded for the contract and no MD loop is advanced here.

**grid_level** - Selects the DFT integration grid level of the Kohn-Sham steps: 1 to 5 are the conventional values and 3 is the shipped default.

**ir_width** - Sets the half-width of the Gaussian broadening of the IR trace in cm-1. The default is 100.0.

**ir_scale** - Sets the scaling factor applied to the harmonic frequencies before the IR trace is drawn. The default 1.0 leaves them unscaled.

**uv_width** - Sets the half-width of the Gaussian broadening of the UV/vis spectrum in nm. The default is 100.0.

**nics_ring** - Reserved NICS request: the 1-based ring atom indices whose axis would receive ghost centres. The key is declared for the NMR contract and documented here for completeness, and the shipped profiles leave it at [0]. This keyword is accepted but has no effect in v3.0.0.

**nics_scan** - Reserved NICS request: the distances in Angstrom along the ring axis of nics_ring. The shipped profiles leave it at [0.0]. This keyword is accepted but has no effect in v3.0.0.

**thermostat** - Selects the thermostat of a dynamics run, nose-hoover by default, with nve and berendsen as the other names. The skill registers no dynamics profile, so the value never reaches a calculation. This keyword is accepted but has no effect in v3.0.0.

**barostat** - Selects the barostat of a constant-pressure dynamics run, berendsen by default. The skill registers no dynamics profile, so the value never reaches a calculation. This keyword is accepted but has no effect in v3.0.0.

**ensemble** - Names the thermodynamic ensemble of a dynamics run: nve, nvt or npt. The skill registers no dynamics profile, so the value never reaches a calculation. This keyword is accepted but has no effect in v3.0.0.

**charges** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**multiplicity** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**ci** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**spin** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**coupling** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**pbc** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**cell** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**lattice_vector** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**kpoint** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**target_states** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**target_nac_pair** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**target_soc_pair** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**verbose** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**temperature** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html).

**pressure** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html).

**solvent** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**shape** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**factor** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**scale** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

The skill is molecular and non-periodic. Any true pbc is refused at validation, so cell, lattice_vector and kpoint never reach a calculation, and a periodic or extended workflow belongs to cp2k, xtb or lammps. There is no non-adiabatic coupling and no spin-orbit coupling, so target_nac_pair and target_soc_pair stay empty and the nac, soc, dipole and stress jobtypes are not registered.

The method families are narrow on purpose: mp2 has no Hessian and no optimization, hf and dft have no UV/vis profile, which needs tddft or tda, and tddft and tda have no Hessian, no transition-state optimization, no IR and no NMR. The ir and nmr profiles need the optional pyscf/properties extension and stop at the first call with a named error without it. thread is a process-wide setting rather than a per-term one, so the first term to run fixes it for the whole run.
