---
title: "5.17. BAGEL"
nav_order: 17
parent: "5. Keyword sections"
---

The bagel section drives BAGEL for multiconfigurational chemistry. It covers CASSCF and XMS-CASPT2 energies, analytic gradients, non-adiabatic coupling elements between states of one spin manifold, Hessians and frequencies from finite differences of gradients, internal geometry optimization of a single state or of a minimum-energy conical intersection, and state and transition dipoles. Unlike the text-deck engines, BAGEL reads a JSON block list: the skill builds the molecule, casscf or smith blocks and the forces, hessian or optimize block from the section keys.

The fourteen profiles are casscf_energy, casscf_gradient, casscf_nac, casscf_frequency, casscf_geometry_optimization, casscf_dipole and casscf_transition_dipole, and the same seven with the xmscaspt2 prefix. Each profile fixes method to the prefix, and the nac chains carry the same-spin NACME of CASSCF or XMS-CASPT2 states; there is no spin-orbit chain at all.

BAGEL keeps its orbitals in .archive files rather than in a Molcas RasOrb or StrOrb. A casscf run writes &lt;project&gt;_converged.archive through its save_ref step, later steps copy that file in as &lt;project&gt;_guess.archive, and the load_ref block is written only when the archive is present, since a reference to a missing file aborts BAGEL. The installation supplies the basis data: &lt;basis&gt;.json and &lt;df_basis&gt;.json are read from &lt;bagel_path&gt;/share, and the run script exports BAGEL, BLAS, LAPACK, BOOST, MPI, OMP_NUM_THREADS, MKL_NUM_THREADS and BAGEL_NUM_THREADS before it launches the binary.

The keys at the end of the table are shared with the framework sections, and the skill receives them from &MOLECULE and &MD.

The skill exposes the following profiles. A profile fixes the method and the set of properties a calculation returns.

| Profile | Capability | Method |
| :--- | :--- | :--- |
| casscf_energy | energy | casscf |
| casscf_gradient | gradient | casscf |
| casscf_nac | nonadiabatic_coupling | casscf |
| casscf_frequency | vibrational_frequency | casscf |
| casscf_geometry_optimization | geometry_optimization | casscf |
| casscf_dipole | dipole | casscf |
| casscf_transition_dipole | transition_dipole | casscf |
| xmscaspt2_energy | energy | xmscaspt2 |
| xmscaspt2_gradient | gradient | xmscaspt2 |
| xmscaspt2_nac | nonadiabatic_coupling | xmscaspt2 |
| xmscaspt2_frequency | vibrational_frequency | xmscaspt2 |
| xmscaspt2_geometry_optimization | geometry_optimization | xmscaspt2 |
| xmscaspt2_dipole | dipole | xmscaspt2 |
| xmscaspt2_transition_dipole | transition_dipole | xmscaspt2 |

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&BAGEL**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Selects the run arm: local runs BAGEL in a subprocess on the current machine, and slurm writes a batch script and submits it. The default is local. |
| shell | '' | Adds skill-level shell commands to the run script, given as a list, a multi-line string or one line with semicolons. Module loads belong here; the… |
| project | bagel_calc | Sets the artifact stem. The JSON deck, the log and the archive files are named after it, for example &lt;project&gt;.json, &lt;project&gt;.log and… |
| profile | casscf_energy | Selects one of the fourteen profiles, named &lt;method&gt;_&lt;task&gt;, for example casscf_energy or xmscaspt2_gradient. The profile fixes the step chain and… |
| method | casscf | Sets the method: casscf or xmscaspt2. It must equal the prefix of profile and it decides whether the chain carries a smith CASPT2 block after the… |
| bagel_path | '' | Names the BAGEL installation root. The executable is &lt;path&gt;/bin/BAGEL, the basis and density-fitting JSON files are read from &lt;path&gt;/share, and a run… |
| bagel_project | BAGEL | Carries the project name in the spelling the framework forwards. A non-empty value overwrites project and the two are kept in step. |
| blas | '' | Sets the BLAS installation path, exported as BLAS and prepended to LD_LIBRARY_PATH before BAGEL starts. An empty value leaves the library search path… |
| lapack | '' | Sets the LAPACK installation path, exported as LAPACK and prepended to LD_LIBRARY_PATH. An empty value changes nothing. |
| boost | '' | Sets the Boost installation path, exported as BOOST, with &lt;path&gt;/lib joining LD_LIBRARY_PATH. An empty value changes nothing. |
| mpi | '' | Sets the MPI installation root, exported as MPI, with &lt;path&gt;/lib joining LD_LIBRARY_PATH and &lt;path&gt;/bin joining PATH so that the launcher named by… |
| mkl | '' | Names the MKL setup script sourced before the run, together with arch. An empty value sources nothing. |
| arch | '' | Sets the architecture argument of the MKL setup script named by mkl, for example intel64. It is ignored when mkl is empty. |
| mpiexe | mpirun | Names the MPI launcher: mpirun by default, an absolute path or another spelling such as mpiexec.hydra when a site needs one. no or none runs… |
| bagel_nproc | 1 | Sets the MPI rank count of a local run. Slurm runs use the granted SLURM_NTASKS instead, and the larger of bagel_nproc and nproc wins in the local… |
| nproc | 1 | Sets the rank count of a local run in the spelling the skill reads. The larger of nproc and bagel_nproc is applied, and it is independent of thread… |
| thread | 1 | Sets the OpenMP thread count per rank, exported as OMP_NUM_THREADS, MKL_NUM_THREADS and BAGEL_NUM_THREADS in both arms. The larger of thread and… |
| bagel_thread | 1 | Carries the thread count in the spelling the framework forwards; the larger of bagel_thread and thread is applied. |
| slurm_time | 01:00:00 | Sets the wall-clock limit of the batch job in HH:MM:SS. The default is 01:00:00. |
| slurm_partition | default | Names the SLURM partition of the batch job. The default is default. |
| slurm_ntasks | 1 | Sets the number of SLURM tasks, which the batch script takes as its rank count. The default is 1. |
| slurm_memory | 4000 | Sets the memory request of the batch job in megabytes. The default is 4000. |
| slurm_command | '' | Adds extra commands to the submission line of the batch job. |
| keep_tmp | 0 | 0 removes the calculation directory once the results are collected; 1 keeps the deck, the log and the archive files. |
| basis | cc-pvdz | Sets the orbital basis from the 38 names in the skill list, for example cc-pvdz, cc-pvtz, aug-cc-pvdz, svp, tzvpp, ano-rcc or sto-3g. The basis is… |
| df_basis | '' | Sets the density-fitting basis used by the CASSCF and CASPT2 blocks, one of the seven jkfit sets: svp-jkfit, tzvpp-jkfit, qzvpp-jkfit, cc-pvdz-jkfit… |
| nact_elec | 2 | Sets the number of active electrons in the casscf block, written as nactel. The default is 2. |
| nact_orb | 2 | Sets the number of active orbitals, written as nacta and derived with nclosed when active is empty. The default is 2. |
| nclosed | 0 | Sets the number of closed-shell orbitals of the casscf block. Together with nact_orb it defines the default active space, the orbitals nclosed + 1… |
| active | [] | Lists the active orbitals explicitly, 1-based as inputs spell them. An empty list takes them from nclosed and nact_orb. |
| ci_roots | [1] | Sets the number of CI roots solved in the casscf block. The default [1] solves the lowest root. |
| ci_spin | [0] | Sets 2S of the CI roots, 0 for a singlet and 2 for a triplet. The default [0] asks for singlet roots. |
| nstate | 1 | Sets the number of electronic states requested. The deck carries no spin keyword, so every root comes from one manifold whatever the state labels say. |
| scf_converge | 1e-08 | Sets the CASSCF convergence threshold and the thresh of the smith CASPT2 block. The default is 1e-8. |
| pt2_shift | 0.2 | Sets the XMS-CASPT2 level shift, written as shift in the smith block. The default is 0.2. |
| maxit_scf | 500 | Sets maxiter of every CASSCF and CASPT2 block, including the smith sub-block. The default is 500 cycles. |
| maxit_opt | 100 | Sets maxiter of the optimize block used by the two geometry_optimization profiles. The default is 100 steps. |
| opt_state_i | 0 | Sets the first optimization target state, 0-based. Equal to opt_state_j it writes opttype energy and relaxes that state alone. |
| opt_state_j | None | Sets the second optimization target state, 0-based, where None means a single-state relaxation and a distinct value requests the minimum-energy… |
| ir_width † | 100 | Sets the half-width in cm-1 recorded with an IR spectrum. The frequency profiles report wavenumbers and intensities, so the value is metadata for a… |
| ir_scale † | 1 | Sets the frequency scaling factor recorded with an IR spectrum. The default 1.0 leaves the harmonic wavenumbers unscaled, and no BAGEL block consumes… |
| uv_width † | 100 | Sets the half-width in nm recorded with an absorption spectrum. BAGEL reports energies and transition dipoles per state, and no block consumes the… |
| nics_ring † | [0] | Reserved NICS request: the ring atom indices that would define the scan axis. No bagel profile consumes it. |
| nics_scan † | [0] | Reserved NICS request: the distances in Angstrom along the ring axis. No bagel profile consumes it. |
| charges | 0 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| multiplicity | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| ci | [1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| spin | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| coupling | [] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| pbc | [0, 0, 0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| cell | [99, 99, 99, 90, 90, 90] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| lattice_vector | [[0, 0, 0], [0, 0, 0], [0, 0, 0]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| kpoint | [1, 1, 1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_states | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| target_nac_pair | [[]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| verbose | 4 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| temperature | 298.15 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html). |
| solvent | gas | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| shape | ellipsoid | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| factor | 40 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| scale | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Selects the run arm: local runs BAGEL in a subprocess on the current machine, and slurm writes a batch script and submits it. The default is local.

**shell** - Adds skill-level shell commands to the run script, given as a list, a multi-line string or one line with semicolons. Module loads belong here; the framework global shell is reserved for the main run.

**project** - Sets the artifact stem. The JSON deck, the log and the archive files are named after it, for example &lt;project&gt;.json, &lt;project&gt;.log and &lt;project&gt;_converged.archive.

**profile** - Selects the step chain, which is the sequence of blocks written into the JSON deck. casscf_energy runs molecule, load_ref, casscf and save_ref; casscf_gradient and casscf_nac run molecule, load_ref, casscf and then one forces block, with the nacme set for the coupling chain; casscf_frequency ends in a hessian block and casscf_geometry_optimization in an optimize block; and the two dipole chains end in one forces block carrying the dipole or the transition-dipole set. The xmscaspt2 chains repeat that shape around the smith CASPT2 level: xmscaspt2_energy runs molecule, load_ref, casscf and smith_caspt2; xmscaspt2_gradient, xmscaspt2_nac and the two dipole chains go from load_ref straight to a caspt2 forces block; and xmscaspt2_frequency runs molecule, load_ref, casscf, smith_caspt2 and hessian_caspt2, because a bare hessian_caspt2 job prints no energy in its own log. The framework jobtypes map onto these chains: md onto the gradient chain and dipole, transition and absorption onto the dipole chains.

**method** - Sets the level of the chain. casscf builds the casscf block and either saves the orbitals through save_ref or consumes them; xmscaspt2 runs the same casscf block and then a smith block whose flags ms, xms and sssr plus shift carry the multistate treatment. BAGEL has no standalone caspt2-titled block, which is why an XMS-CASPT2 energy is always a casscf step followed by a smith step. A profile whose prefix does not match method is refused before the JSON deck is written.

**bagel_path** - Names the BAGEL installation root. The executable is &lt;path&gt;/bin/BAGEL, the basis and density-fitting JSON files are read from &lt;path&gt;/share, and a run is refused when the directory or the binary is missing.

**bagel_project** - Carries the project name in the spelling the framework forwards. A non-empty value overwrites project and the two are kept in step.

**blas** - Sets the BLAS installation path, exported as BLAS and prepended to LD_LIBRARY_PATH before BAGEL starts. An empty value leaves the library search path as the machine set it.

**lapack** - Sets the LAPACK installation path, exported as LAPACK and prepended to LD_LIBRARY_PATH. An empty value changes nothing.

**boost** - Sets the Boost installation path, exported as BOOST, with &lt;path&gt;/lib joining LD_LIBRARY_PATH. An empty value changes nothing.

**mpi** - Sets the MPI root that the run script exports as MPI, adds to LD_LIBRARY_PATH as &lt;path&gt;/lib and to PATH as &lt;path&gt;/bin. BAGEL itself is linked against this MPI, and the launcher named by mpiexe is found through the same PATH entry, so the two must name the same MPI installation. An empty value leaves both the library and the program search paths untouched, which is correct only when the machine already provides them.

**mkl** - Names the MKL setup script that the run script sources before BAGEL starts, together with arch. Sourcing is what puts the MKL runtime on the path on a cluster whose login environment does not carry it; an empty value writes no source line, and the value is ignored by a BAGEL built against another BLAS.

**arch** - Sets the architecture argument passed to the MKL setup script, for example intel64 or intel64_and_ia32. It is only read together with mkl: when mkl is empty no setup script runs and arch has no effect, and BAGEL reports the MKL layer it actually loaded in its own log.

**mpiexe** - Names the MPI launcher: mpirun by default, an absolute path or another spelling such as mpiexec.hydra when a site needs one. no or none runs &lt;bagel_path&gt;/bin/BAGEL directly as a single process that uses threads only.

**bagel_nproc** - Sets the number of MPI ranks of a local run, handed to the launcher as mpiexe -np bagel_nproc, with the larger of bagel_nproc and nproc applied. The local arm deliberately reads this value rather than $SLURM_NTASKS, because a local job sits outside an allocation and an inherited SLURM_NTASKS would describe a different machine. The skill's own batch script takes the granted $SLURM_NTASKS instead, from the slurm_ntasks header. Ranks and threads are separate keywords: neither stands in for the other.

**nproc** - Sets the rank count of a local run in the spelling the skill reads. The larger of nproc and bagel_nproc is applied, and it is independent of thread, which counts threads rather than ranks.

**thread** - Sets the OpenMP thread count per rank, exported as OMP_NUM_THREADS, MKL_NUM_THREADS and BAGEL_NUM_THREADS in both arms. The larger of thread and bagel_thread wins, and the retired spelling num_thread is refused.

**bagel_thread** - Carries the thread count in the spelling the framework forwards; the larger of bagel_thread and thread is applied.

**slurm_time** - Sets the wall-clock limit of the batch job in HH:MM:SS. The default is 01:00:00.

**slurm_partition** - Names the SLURM partition of the batch job. The default is default.

**slurm_ntasks** - Sets the number of SLURM tasks, which the batch script takes as its rank count. The default is 1.

**slurm_memory** - Sets the memory request of the batch job in megabytes. The default is 4000.

**slurm_command** - Adds extra commands to the submission line of the batch job.

**keep_tmp** - 0 removes the calculation directory once the results are collected; 1 keeps the deck, the log and the archive files.

**basis** - Sets the orbital basis from the 38 names in the skill list, for example cc-pvdz, cc-pvtz, aug-cc-pvdz, svp, tzvpp, ano-rcc or sto-3g. The basis is read from &lt;basis&gt;.json in the BAGEL share directory, so only names shipped with the installation resolve.

**df_basis** - Sets the density-fitting basis of the correlated blocks. The seven accepted sets are the jkfit auxiliary bases shipped with BAGEL: svp-jkfit, tzvpp-jkfit, qzvpp-jkfit, cc-pvdz-jkfit, cc-pvtz-jkfit, cc-pvqz-jkfit and cc-pv5z-jkfit, each read from &lt;bagel_path&gt;/share. An empty value lets the builder match the orbital basis by similarity and fall back to svp-jkfit when no match exists, and the retired spelling no or none is no longer accepted.

**nact_elec** - Sets the number of active electrons in the casscf block, written as nactel. The default is 2.

**nact_orb** - Sets the number of active orbitals, written as nacta and derived with nclosed when active is empty. The default is 2.

**nclosed** - Sets the number of closed-shell orbitals of the casscf block. Together with nact_orb it defines the default active space, the orbitals nclosed + 1 through nclosed + nact_orb in the 1-based input numbering.

**active** - Lists the active orbitals explicitly, 1-based as inputs spell them. An empty list takes them from nclosed and nact_orb.

**ci_roots** - Sets the number of CI roots solved in the casscf block. The default [1] solves the lowest root.

**ci_spin** - Sets 2S of the CI roots, 0 for a singlet and 2 for a triplet. The default [0] asks for singlet roots.

**nstate** - Sets the number of electronic states requested. The deck carries no spin keyword, so every root comes from one manifold whatever the state labels say.

**scf_converge** - Sets the CASSCF convergence threshold and the thresh of the smith CASPT2 block. The default is 1e-8.

**pt2_shift** - Sets the XMS-CASPT2 level shift, written as shift in the smith block. The default is 0.2.

**maxit_scf** - Sets maxiter of every CASSCF and CASPT2 block, including the smith sub-block. The default is 500 cycles.

**maxit_opt** - Sets maxiter of the optimize block used by the two geometry_optimization profiles. The default is 100 steps.

**opt_state_i** - Sets the first optimization target state, 0-based. Equal to opt_state_j it writes opttype energy and relaxes that state alone.

**opt_state_j** - Sets the second optimization target state, 0-based, where None means a single-state relaxation and a distinct value requests the minimum-energy conical intersection of the pair (opttype meci).

**ir_width** - Sets the half-width in cm-1 recorded with an IR spectrum. The frequency profiles report wavenumbers and intensities, so the value is metadata for a broadened trace and no BAGEL block consumes it. This keyword is accepted but has no effect in v3.0.0.

**ir_scale** - Sets the frequency scaling factor recorded with an IR spectrum. The default 1.0 leaves the harmonic wavenumbers unscaled, and no BAGEL block consumes the value. This keyword is accepted but has no effect in v3.0.0.

**uv_width** - Sets the half-width in nm recorded with an absorption spectrum. BAGEL reports energies and transition dipoles per state, and no block consumes the width. This keyword is accepted but has no effect in v3.0.0.

**nics_ring** - Reserved NICS request: the ring atom indices that would define the scan axis. No bagel profile consumes it. This keyword is accepted but has no effect in v3.0.0.

**nics_scan** - Reserved NICS request: the distances in Angstrom along the ring axis. No bagel profile consumes it. This keyword is accepted but has no effect in v3.0.0.

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

**verbose** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**temperature** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html).

**solvent** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**shape** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**factor** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**scale** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

BAGEL supports CASSCF and XMS-CASPT2 and nothing else here: there is no coupled-cluster, no density functional and no implicit solvent, and the section carries no functional or solvent key. Spin-orbit coupling is absent at every level, so there is no SOC field to fill, and the non-adiabatic coupling is computed between states of one spin manifold only. The generated deck carries no spin keyword, so a run with more than one spin in state_spin is refused rather than misinterpreted, and every root of one job belongs to one manifold.

The backend is molecular: any true pbc is refused at validation, so a periodic or extended workflow belongs to cp2k, xtb or lammps. Optimization steps are capped at a maximum step of 0.05 Bohr because an initial analytic CASPT2 gradient can be very large, and a frequency run is a finite-difference Hessian of gradients rather than an analytic one. Orbitals continue through .archive files only, so a first CASPT2 step without an archive simply omits the load_ref block and starts from a fresh CASSCF.
