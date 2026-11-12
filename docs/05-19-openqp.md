---
title: "5.19. OPENQP"
nav_order: 19
parent: "5. Keyword sections"
---

The openqp section configures the OpenQP layer of a run. It selects the profile, the method, the basis set and functional, the SCF and response controls, the guess and NAC settings, and the runtime that launches openqp. The classic spelling of the section is &amp;OPENQP; in YAML and agent JSON it is the openqp mapping.

Every profile compiles to an OpenQP runtype and a chain of input sections, and the skill writes the geometry file and the deck itself, so no user-side OpenQP input file exists. [Section 5](05-keyword-sections.html) describes how a jobtype reaches a profile.

The table lists the execution and program keys first, then the method, response and optimizer keys, then the spectrum-output keys, and the framework-parity keys last.

The skill exposes the following profiles. A profile fixes the method and the set of properties a calculation returns.

| Profile | Capability | Method |
| :--- | :--- | :--- |
| hf_energy | energy | hf |
| hf_gradient | gradient | hf |
| hf_geometry_optimization | geometry_optimization | hf |
| hf_frequency | vibrational_frequency | hf |
| tddft_energy | energy | tddft |
| tddft_gradient | gradient | tddft |
| tddft_geometry_optimization | geometry_optimization | tddft |
| tddft_nonadiabatic_dynamics | nonadiabatic_dynamics | tddft |
| tddft_frequency | vibrational_frequency | tddft |
| mrsf_energy | energy | mrsf |
| mrsf_gradient | gradient | mrsf |
| mrsf_nac | nonadiabatic_coupling | mrsf |
| mrsf_soc | soc_coupling | mrsf |
| mrsf_soc_gradient | soc_coupling | mrsf |
| mrsf_geometry_optimization | geometry_optimization | mrsf |
| mrsf_nonadiabatic_dynamics | nonadiabatic_dynamics | mrsf |
| mrsf_frequency | vibrational_frequency | mrsf |

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&OPENQP**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Selects the run arm. local writes a bash script and runs it in the calculation directory, and slurm writes a batch script and submits it with sbatch… |
| shell | '' | Prepends extra shell commands, such as module loads, to the run script. A list is joined one command per line and a string is split on semicolons… |
| project | openqp_calc | Sets the file stem of the run. The skill writes &lt;project&gt;.xyz and &lt;project&gt;.inp and reads &lt;project&gt;.out and &lt;project&gt;.json from the same directory… |
| profile | tddft_energy | Selects the OpenQP profile, and with it the runtype, the input sections and the method. The 17 profiles are hf_energy, hf_gradient… |
| openqp_project | OQP | Sets the project name in the framework spelling of the section. A non-empty value takes precedence over project, is exported as OPENQP_PROJECT and… |
| openqp_path | '' | Names the OpenQP installation root. The run script exports it as OPENQP_ROOT and prepends $OPENQP_ROOT/lib to LD_LIBRARY_PATH, and a path that does… |
| use_mpi | false | Chooses the launch mode. false runs openqp with --nompi and uses OpenMP threads only, and true launches the job with mpirun and starts openqp_nproc… |
| mpi_command | '' | Overrides the launcher prefix of an MPI run, for example mpiexec -n 4. An empty value lets the script build mpirun -n &lt;openqp_nproc&gt;; the default is… |
| openqp_nproc | 1 | Sets the number of MPI ranks, passed as -n to the launcher and, in slurm mode, as #SBATCH --ntasks. It is the only rank spelling this skill accepts… |
| thread | 1 | Sets the OpenMP threads per process through OMP_NUM_THREADS. It is the spelling a standalone configuration writes, while openqp_thread is the… |
| openqp_thread | 1 | Sets the thread count in the framework spelling of the section. It folds into thread and the larger of the two values is used, so neither spelling… |
| keep_tmp | 0 | Keeps the calculation directory after the run. 1 leaves the input, the log, the orbital json and the export files in place, and 0 removes the… |
| slurm_time | 01:00:00 | Sets the batch time limit in HH:MM:SS and is written as #SBATCH --time; the default is 01:00:00. |
| slurm_partition | default | Sets the batch partition and is written as #SBATCH --partition; the default is default. |
| slurm_ntasks | 1 | Sets the number of batch tasks and is written as #SBATCH --ntasks. In MPI mode the submitted count is the larger of this value and openqp_nproc; the… |
| slurm_memory | 4000 | Sets the memory of the batch allocation in MB and is written as #SBATCH --mem; the default is 4000. |
| slurm_command | '' | Names the submission command of a slurm run. The default is empty, which leaves the sbatch line to the runner. |
| method | tddft | Sets the electronic-structure method: hf for a Hartree-Fock ground state, tddft for a TD-DFT or TD-HF response, or mrsf for spin-flip MRSF-TDDFT on a… |
| basis | 6-31g\* | Sets the orbital basis set from the OpenQP library. The accepted names are 3-21g, 4-31g, 6-31g, 6-31g\*, 6-31g\*\*, 6-31+g\*, 6-311g, 6-311g\*, 6-311g\*\*… |
| functional | bhhlyp | Names the exchange-correlation functional, written on the [input] line for tddft and mrsf. The accepted names are the libxc spellings bhhlyp, b3lyp… |
| td_type | tda | Selects the response type of a tddft run: tda for the Tamm-Dancoff approximation or rpa for the full response. The mrsf method forces mrsf regardless… |
| scf_type | '' | Holds the reference type written as [scf] type, one of rhf, rohf or uhf. Normalization derives it from the charge and multiplicity, the mrsf method… |
| scf_maxit | 100 | Limits the SCF iterations through [scf] maxit; the default is 100. |
| scf_converge | 1e-06 | Sets the SCF energy threshold through [scf] conv, in Hartree; the default is 1e-6. |
| td_maxit | 50 | Limits the response iterations through [tdhf] maxit; the default is 50. |
| td_converge | 1e-06 | Sets the response convergence threshold through [tdhf] conv, in atomic units; the default is 1e-6. |
| guess_type | auto | Sets the initial-guess type through [guess] type, for example auto for the Huckel fallback of a first step. Every run also saves the orbital json… |
| nac_type | ktdc | Selects the nonadiabatic-coupling flavour. ktdc runs the data route for the framework kTDC integrator and needs no orbital continuation, dcm asks for… |
| nac_dt | 20.67 | Sets the time step of the numerical NAC vector differentiation, in atomic units of time; the default is 20.67. |
| nac_dx | 0.0001 | Sets the displacement of the numerical NAC vector differentiation, in Bohr; the default is 0.0001. |
| nac_bp | true | Also computes the branching-plane quantities of a NAC vector run. The default is true. |
| dosoc | 0 | Requests spin-orbit coupling for the MRSF response, the request side of the framework &md dosoc flag. 1 selects the two-manifold run that solves the… |
| nstate_s | 0 | Sizes the singlet ladder of a two-manifold run through [tdhf] nstate_s. 0 means the value is derived from state_spin, and a non-zero value must cover… |
| nstate_t | 0 | Sizes the triplet ladder of a two-manifold run through [tdhf] nstate_t. 0 means the value is derived from state_spin, and a non-zero value must cover… |
| scal_rel | 0 | Sets the scalar-relativistic DKH order of the reference, written only when it differs from OpenQP's default. 0 keeps that default; the default is 0. |
| opt_maxit | 100 | Limits the geometry optimization cycles through [optimize] maxit; the default is 100. |
| opt_optimizer | bfgs | Selects the native optimizer written as [optimize] optimizer; the default is bfgs. |
| opt_rmsd_grad | 0.0001 | Sets the RMS gradient threshold of the optimization through [optimize] rmsd_grad, in Hartree/Bohr; the default is 1e-4. |
| opt_max_grad | 0.0003 | Sets the maximum gradient threshold of the optimization through [optimize] max_grad, in Hartree/Bohr; the default is 3e-4. |
| hess_dx | 0.01 | Sets the displacement of the numerical Hessian, written as [hess] dx in Bohr; the default is 0.01. |
| nstate | 1 | Sets the number of electronic states in the ladder. It is written as [tdhf] nstate and must agree with the length of state_spin; a two-manifold run… |
| ir_width † | 100 | Sets the half-width of the Lorentzian broadening of the IR spectrum in cm^-1; the default is 100.0. |
| ir_scale † | 1 | Scales the intensities of the IR spectrum. The default is 1.0. |
| uv_width † | 100 | Sets the broadening width of the UV/visible spectrum in cm^-1; the default is 100.0. |
| nics_ring † | [0] | Lists the ring indices used for an NICS scan. The default [0] names no ring. |
| nics_scan † | [0] | Lists the scan distances of an NICS calculation in Angstrom. The default is [0.0]. |
| charges | 0 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| multiplicity | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| ci | [1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| spin | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| coupling | [] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_states | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| temperature † | 298.15 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| verbose | 4 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| solvent † | gas | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| pbc | [0, 0, 0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| cell | [99, 99, 99, 90, 90, 90] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| lattice_vector | [[0, 0, 0], [0, 0, 0], [0, 0, 0]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| kpoint | [1, 1, 1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Selects the run arm. local writes a bash script and runs it in the calculation directory, and slurm writes a batch script and submits it with sbatch; in slurm mode openqp_nproc also becomes #SBATCH --ntasks. The default is local.

**shell** - Prepends extra shell commands, such as module loads, to the run script. A list is joined one command per line and a string is split on semicolons; the commands run before openqp starts. The default is empty.

**project** - Sets the file stem of the run. The skill writes &lt;project&gt;.xyz and &lt;project&gt;.inp and reads &lt;project&gt;.out and &lt;project&gt;.json from the same directory; the fixed-name export files carry no stem. The default is openqp_calc.

**profile** - Selects the OpenQP profile, and with it the runtype, the input sections and the method. The 17 profiles are hf_energy, hf_gradient, hf_geometry_optimization, hf_frequency, tddft_energy, tddft_gradient, tddft_geometry_optimization, tddft_nonadiabatic_dynamics, tddft_frequency, mrsf_energy, mrsf_gradient, mrsf_nac, mrsf_soc, mrsf_soc_gradient, mrsf_geometry_optimization, mrsf_nonadiabatic_dynamics and mrsf_frequency; the default is tddft_energy.

**openqp_project** - Sets the project name in the framework spelling of the section. A non-empty value is copied onto project and exported as OPENQP_PROJECT, and because the log is always $OPENQP_PROJECT.log, the orbital json is always $OPENQP_PROJECT.json and the automatic continuation follows that fixed name. The default is OQP.

**openqp_path** - Names the OpenQP installation root. The run script exports it as OPENQP_ROOT and prepends $OPENQP_ROOT/lib to LD_LIBRARY_PATH ahead of the MKL and GCC directories, so an installation that ships its own libraries resolves them. The openqp executable itself is looked for in PATH, not under this root, and a path that does not exist is refused at validation. The default is empty.

**use_mpi** - Chooses the launch mode. false runs openqp with --nompi and uses OpenMP threads only, and true launches the job with mpirun and starts openqp_nproc ranks; the default is false.

**mpi_command** - Overrides the launcher prefix of an MPI run, for example mpiexec -n 4. An empty value lets the script build mpirun -n &lt;openqp_nproc&gt;; the default is empty.

**openqp_nproc** - Sets the number of MPI ranks. With use_mpi true the run script launches mpirun -n &lt;openqp_nproc&gt;, and in slurm mode the value also raises #SBATCH --ntasks to match. It is the only rank spelling this skill reads: neither nproc nor mpi_nproc is accepted, because the rank count has no second name here. The count also bounds the displacement pool of a numerical Hessian or NAC vector, one displacement per rank. The default is 1.

**thread** - Sets the OpenMP threads per process through OMP_NUM_THREADS. It is the spelling a standalone configuration writes, while openqp_thread is the framework spelling, and the larger of the pair is used; the default is 1.

**openqp_thread** - Sets the thread count in the framework spelling of the section. It folds into thread and the larger of the two values is used, so neither spelling can shrink a request made in the other; the default is 1.

**keep_tmp** - Keeps the calculation directory after the run. 1 leaves the input, the log, the orbital json and the export files in place, and 0 removes the directory once the results are parsed; the default is 0.

**slurm_time** - Sets the batch time limit in HH:MM:SS and is written as #SBATCH --time; the default is 01:00:00.

**slurm_partition** - Sets the batch partition and is written as #SBATCH --partition; the default is default.

**slurm_ntasks** - Sets the number of batch tasks and is written as #SBATCH --ntasks. In MPI mode the submitted count is the larger of this value and openqp_nproc; the default is 1.

**slurm_memory** - Sets the memory of the batch allocation in MB and is written as #SBATCH --mem; the default is 4000.

**slurm_command** - Names the submission command of a slurm run. The default is empty, which leaves the sbatch invocation to the runner, and a non-empty value is carried with the job settings.

**method** - Sets the electronic-structure method and the OpenQP input it compiles to. hf writes method=hf and takes no functional. tddft writes method=tdhf with the chosen functional and td_type tda or rpa. mrsf writes method=tdhf with [tdhf] type=mrsf and a triplet ROHF reference, which is the only reference OpenQP accepts for a spin-flip response. The method must agree with the profile prefix, and a method that cannot answer the profile task is refused.

**basis** - Sets the orbital basis set, written as basis= on the [input] line. The names come from the OpenQP basis library: the Pople sets 3-21g, 4-31g, 6-31g, 6-31g\*, 6-31g\*\*, 6-31+g\*, 6-311g, 6-311g\* and 6-311g\*\*, the Dunning sets cc-pvdz, cc-pvtz, cc-pvqz, aug-cc-pvdz and aug-cc-pvtz, the Karlsruhe sets def2-svp, def2-tzvp and def2-tzvpp, and sto-3g. The value is lowercased and stripped of spaces before the check, and any other name is refused rather than passed to the engine. The default is 6-31g\*.

**functional** - Names the exchange-correlation functional, appended to the [input] line for tddft and mrsf. The list is closed and holds the libxc names bhhlyp, b3lyp, b3lyp5, pbe, pbe0, bp86, blyp, b97, b97m, m06, wb97, wb97x and cam-b3lyp; hf takes an empty value and refuses a functional, while tddft and mrsf refuse an empty one. The default is bhhlyp, and the name reaches libxc as given.

**td_type** - Selects the response type written as [tdhf] type. tda applies the Tamm-Dancoff approximation and rpa solves the full response; the mrsf method forces mrsf and ignores the request, and hf carries no response block at all, so the value is emptied. A value other than rpa or tda on a tddft method is refused. The default is tda.

**scf_type** - Holds the reference type written as [scf] type, one of rhf, rohf or uhf. Normalization derives it from the charge and multiplicity, the mrsf method forces rohf, and the default empty value means it has not been derived yet.

**scf_maxit** - Limits the SCF iterations through [scf] maxit; the default is 100.

**scf_converge** - Sets the SCF energy threshold through [scf] conv, in Hartree; the default is 1e-6.

**td_maxit** - Limits the response iterations through [tdhf] maxit; the default is 50.

**td_converge** - Sets the response convergence threshold through [tdhf] conv, in atomic units; the default is 1e-6.

**guess_type** - Sets the initial-guess type of the [guess] block. Every run also sets save_mol=true, so OpenQP stores its orbitals as &lt;project&gt;.json beside the log, and [guess] file declares the same file with type=auto, which means the json is read when present and the Huckel guess is built when it is not. The default auto therefore continues the previous step in a multi-step run and starts from Huckel on the first one.

**nac_type** - Selects the nonadiabatic-coupling flavour, and with it the runtype of a dynamics or NAC profile. ktdc keeps a dynamics run on runtype=data with per-step energies and gradients only, which is what the framework kTDC integrator needs and what needs no orbital continuation. dcm switches to runtype=prop for the nacme and MO overlap of the previous step, which a single calculation cannot supply and which is therefore refused outside the dynamics path. nac keeps a single-point mrsf_nac run on runtype=nac for one state pair, and inside a dynamics profile it adds [properties] nac=nac for numerical vectors. The framework spelling nactype is accepted as an alias, the default is ktdc, and a spin-orbit run refuses any of them.

**nac_dt** - Sets the time step of the numerical NAC vector differentiation through [nac] dt, in atomic units of time. The NAC vector is a difference quotient over a displaced geometry and a finite time step, so the two controls nac_dt and nac_dx define the approximation. The default is 20.67, and the section is written only for a runtype=nac run.

**nac_dx** - Sets the nuclear displacement of the numerical NAC vector differentiation through [nac] dx, in Bohr. It is the geometry step that the finite difference is taken over, and it is written in the same [nac] section as nac_dt. The default is 0.0001.

**nac_bp** - Asks for the branching-plane quantities of a NAC vector run through [nac] bp. The flag rides in the same section as nac_dt and nac_dx and is only written for a runtype=nac run. The default is true.

**dosoc** - Requests spin-orbit coupling, the framework spelling of &md dosoc carried into the skill section. 1 selects the two-manifold run: the singlet and triplet ladders are solved and coupled, and nstate_s and nstate_t size them. 0 leaves the run single-manifold. A spin-orbit run needs method mrsf, at least one singlet and one triplet row, and refuses every NAC flavour, because the analytic vector route answers the singlet fold only. The default is 0.

**nstate_s** - Sizes the singlet ladder of a two-manifold run, written as [tdhf] nstate_s. 0 derives the size from the declared singlet rows of state_spin, and a non-zero value must cover them. The state list is sorted by spin, so singlets come first; the default is 0.

**nstate_t** - Sizes the triplet ladder of a two-manifold run, written as [tdhf] nstate_t. 0 derives the size from the declared triplet rows of state_spin, and a non-zero value must cover them. Triplet states follow the singlets in the sorted state list; the default is 0.

**scal_rel** - Sets the scalar-relativistic DKH order of the reference, written only when it differs from OpenQP's default. 0 keeps that default; the default is 0.

**opt_maxit** - Limits the geometry optimization cycles through [optimize] maxit; the default is 100.

**opt_optimizer** - Selects the native optimizer written as [optimize] optimizer; the default is bfgs.

**opt_rmsd_grad** - Sets the RMS gradient threshold of the optimization through [optimize] rmsd_grad, in Hartree/Bohr; the default is 1e-4.

**opt_max_grad** - Sets the maximum gradient threshold of the optimization through [optimize] max_grad, in Hartree/Bohr; the default is 3e-4.

**hess_dx** - Sets the displacement of the numerical Hessian, written as [hess] dx in Bohr; the default is 0.01.

**nstate** - Sets the number of electronic states in the ladder, written as [tdhf] nstate. It must be at least 1, must agree with the length of state_spin, and bounds target_states. In a two-manifold run the value stays the configured ladder count rather than the total across both ladders, because each ladder is sized by nstate_s or nstate_t. The default is 1.

**ir_width** - Sets the half-width of the Lorentzian broadening of the IR spectrum in cm^-1; the default is 100.0. This keyword is accepted but has no effect in v3.0.0.

**ir_scale** - Scales the intensities of the IR spectrum. The default is 1.0. This keyword is accepted but has no effect in v3.0.0.

**uv_width** - Sets the broadening width of the UV/visible spectrum in cm^-1; the default is 100.0. This keyword is accepted but has no effect in v3.0.0.

**nics_ring** - Lists the ring indices used for an NICS scan. The default [0] names no ring. This keyword is accepted but has no effect in v3.0.0.

**nics_scan** - Lists the scan distances of an NICS calculation in Angstrom. The default is [0.0]. This keyword is accepted but has no effect in v3.0.0.

**charges** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**multiplicity** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**ci** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**spin** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**coupling** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**target_states** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**temperature** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). This keyword is accepted but has no effect in v3.0.0.

**verbose** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**solvent** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). This keyword is accepted but has no effect in v3.0.0.

**pbc** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**cell** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**lattice_vector** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**kpoint** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

The skill runs the openqp command and never imports pyoqp, so the installation must provide that executable in PATH; openqp_path only exports OPENQP_ROOT and its lib directory. Every run writes &lt;project&gt;.xyz and &lt;project&gt;.inp and then reads the fixed-name export files energies, grad_&lt;i&gt;, nacme, dcme, nac_&lt;i&gt;_&lt;j&gt;, opt.xyz, opt_status.txt, &lt;project&gt;.hess.json and socme.

The exchange-correlation family is closed. functional takes the libxc names bhhlyp, b3lyp, b3lyp5, pbe, pbe0, bp86, blyp, b97, b97m, m06, wb97, wb97x or cam-b3lyp, hf takes none, and any other name is refused instead of being handed to the engine. The response type is limited to TDA and RPA for tddft.

MRSF covers the spin-flip side. MRSF-TDDFT runs on a triplet ROHF reference, it is the only method that answers NAC vectors and spin-orbit couplings, and one NAC vector calculation covers one state pair. A spin-orbit run cannot carry NAC of any flavour, needs at least one singlet and one triplet row, and pins ispher=false; nac_type dcm is refused for a single calculation because it continues from the previous step orbital json. Advanced runtypes (MECI, MECP, MEP, TS) are not in the profile set, and a non-zero pbc is refused because the backend is molecular.
