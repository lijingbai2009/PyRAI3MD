---
title: "5.16. MOLCAS"
nav_order: 16
parent: "5. Keyword sections"
---

The molcas section drives OpenMolcas through pymolcas. It covers CASSCF, with RAS1/RAS2/RAS3 partitioning or an explicit ALTEr active space, and CASPT2 in its ss, ms, xms, rms and xdw forms, together with analytic gradients and non-adiabatic couplings through ALASKA, spin-orbit coupling through a dual singlet/triplet RASSCF and RASSI, absorption and transition dipoles through RASSI, SLAPAF geometry optimization of a single state, of a same-spin crossing seam or of a singlet-triplet crossing point, engine-internal ab initio dynamics and surface hopping through DYNAMIX and SURFACEHOP, and MCKINLEY vibrational frequencies.

The fifteen profiles choose the module chain: hf_energy; casscf_energy, casscf_gradient, casscf_nac, casscf_properties, casscf_geometry_optimization, casscf_abinitio_dynamics, casscf_surface_hopping, casscf_soc, casscf_absorption and casscf_frequency; and caspt2_energy, caspt2_properties, caspt2_absorption and caspt2_frequency. Each profile fixes its method, and the capability table behind it decides what may be asked: hf serves energy and frequency only, casscf serves every task, and caspt2 serves energy, gradient, nac, absorption, soc and frequency but not geometry optimization, dynamics or surface hopping.

Four files matter beside the installation. molcas_path must name the OpenMolcas root that holds bin/pymolcas; the geometry is staged as &lt;project&gt;.xyz and referenced from the GATEWAY block rather than inlined; a &lt;title&gt;.StrOrb orbital guess sitting next to the framework input is picked up automatically and staged as &lt;project&gt;.RasOrb; and a dynamics run needs initial velocities, without which the skill raises. Background charges arrive either as per-step arrays or through read_charge, which may name a file that declares its own point-charge or field representation.

The keys at the end of the table are shared with the framework sections, and the skill receives them from &MOLECULE and &MD. The dynamics keys in the middle, size, deco, thermostat and root, are the values the adapter forwards from &MD into the generated DYNAMIX and SURFACEHOP decks.

The skill exposes the following profiles. A profile fixes the method and the set of properties a calculation returns.

| Profile | Capability | Method |
| :--- | :--- | :--- |
| hf_energy | energy | hf |
| casscf_energy | energy | casscf |
| casscf_gradient | gradient | casscf |
| casscf_nac | nonadiabatic_coupling | casscf |
| casscf_properties | energy | casscf |
| casscf_geometry_optimization | geometry_optimization | casscf |
| casscf_abinitio_dynamics | abinitio_dynamics | casscf |
| casscf_surface_hopping | surface_hopping | casscf |
| casscf_soc | soc_coupling | casscf |
| casscf_absorption | absorption_spectrum | casscf |
| caspt2_energy | energy | caspt2 |
| caspt2_properties | energy | caspt2 |
| caspt2_absorption | absorption_spectrum | caspt2 |
| casscf_frequency | vibrational_frequency | casscf |
| caspt2_frequency | vibrational_frequency | caspt2 |

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&MOLCAS**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Selects the run arm: local writes and runs the pymolcas script on the current machine, and slurm writes a batch script and submits it. The default is… |
| shell | '' | Adds skill-level shell commands to the run script, given as a list, a multi-line string or one line with semicolons. Module loads and environment… |
| project | molcas_calc | Sets the OpenMolcas project name, exported as MOLCAS_PROJECT. The deck, the geometry, the orbital, the optimized structure and the trajectory files… |
| profile | casscf_energy | Selects one of the fifteen profiles: hf_energy, the ten casscf_\* chains and the four caspt2_\* chains, for example casscf_energy or caspt2_properties… |
| method | casscf | Sets the multiconfigurational method: hf, casscf or caspt2. It must equal the prefix of profile and it decides which modules may appear in the chain. |
| molcas_path | '' | Names the OpenMolcas installation root. pymolcas is looked for at &lt;path&gt;/bin/pymolcas and then at &lt;path&gt;/pymolcas, and the directory must exist at… |
| molcas_project | MOLCAS | Carries the project name in the spelling the framework forwards. A non-empty value overwrites project and the two are kept in step. |
| molcas_mem | 4000 | Sets the memory exported as MOLCAS_MEM in megabytes. The default is 4000 MB. |
| molcas_nproc | 1 | Carries the MPI rank count in the spelling the framework forwards. The larger of molcas_nproc and nproc is exported as MOLCAS_NPROCS. |
| molcas_print | 3 | Sets the print level exported as MOLCAS_PRINT. The default 3 writes the standard module output, lower values quieten the deck and higher values add… |
| nproc | 1 | Sets the MPI rank count of a local run. The larger of nproc and molcas_nproc wins, so neither spelling can shrink a request made in the other. |
| thread | 1 | Sets the OpenMP thread count exported as OMP_NUM_THREADS, one thread count per rank. The larger of thread and molcas_thread wins, and the retired… |
| molcas_thread | 1 | Carries the thread count in the spelling the framework forwards; the larger of molcas_thread and thread is applied. |
| slurm_time | 01:00:00 | Sets the wall-clock limit of the batch job in HH:MM:SS. The default is 01:00:00. |
| slurm_partition | default | Names the SLURM partition of the batch job. The default is default. |
| slurm_ntasks | 1 | Sets the number of SLURM tasks, which becomes the rank count of the submitted job. The default is 1. |
| slurm_memory | 4000 | Sets the memory request of the batch job in megabytes. The default is 4000. |
| slurm_command | '' | Adds extra commands to the submission line of the batch job. |
| keep_tmp | 0 | 0 removes the calculation directory once the results are collected; 1 keeps the deck, the logs and the orbital files. |
| basis | ano-rcc-vdzp | Sets the orbital basis from the 24 names in the skill list: the Pople, ANO-S, ANO-L, ANO-RCC, correlation-consistent and def2 families. A per-element… |
| nact_elec | 2 | Sets the number of active electrons in the RASSCF, written as NactEl. The default is 2. |
| nact_orb | 2 | Sets the number of active orbitals in the RASSCF, written as Ras2 unless the RAS partition overrides it. The default is 2. |
| inactive_orb | 0 | Sets the number of inactive orbitals. Zero derives it from (N_elec - nact_elec)/2. |
| ras1_orb | 0 | Sets the size of the RAS1 subspace, the holes allowed above it. Zero leaves a plain CAS without RAS partitioning. |
| ras2_orb | 0 | Sets the size of the RAS2 subspace. Zero falls back to nact_orb, which is the plain CAS case. |
| ras3_orb | 0 | Sets the size of the RAS3 subspace, the particles allowed into it. Zero leaves a plain CAS without RAS partitioning. |
| active | [] | Lists the active orbitals explicitly, 1-based as the input spells them, and the skill writes them as ALTEr pairs. An empty list takes the orbitals… |
| ci_roots | [1] | Sets the number of CI roots in each spin block, written as ROOT in the RASSCF block. One block is written per ci_spin entry, and the default [1]… |
| ci_spin | [0] | Sets 2S of each spin block, written as Spin in the RASSCF block: 0 for a singlet, 2 for a triplet. The default [0] runs one singlet block. |
| state_spin | [] | Labels the spin of each requested state as 2S, 0 for a singlet and 2 for a triplet. It decides which state sits in which block and which pairs count… |
| nstate | 1 | Sets the number of electronic states requested. It bounds the state ladder the framework builds and the roots the CI blocks must cover. |
| xms | true | Legacy multistate switch for CASPT2: true, the default, resolves an empty caspt2_type to xms and false resolves it to ss. |
| caspt2_type | '' | Selects the CASPT2 variant: ss for single-state, ms for MULTISTATE all, xms for XMULTISTATE all, rms for RMULTISTATE all, and xdw for XMULTISTATE… |
| dwms_zeta | 1 | Sets the XDW-CASPT2 zeta parameter, written as DWMS for the xdw variant. It is a positive real number, 1.0 by default. |
| dwtype | 3 | Selects the exponent type of the XDW-CASPT2 reference weights, one of 1, 2 or 3, written as DWTYPE. The default is 3. |
| nocsf | true | Drops the CSF term from the non-adiabatic coupling when true, which is the default for both the CASSCF and the CASPT2 NAC. Given false the CSF… |
| caspt2_ipea_shift | None | Sets the CASPT2 IPEA shift in Hartree. None leaves the OpenMolcas default of 0.25; 0.0 emits IPEA 0.0, the portable setting for analytic gradient and… |
| caspt2_real_shift | 0 | Sets the CASPT2 real level shift in Hartree, written as REAL. The default 0.0 applies no shift. |
| caspt2_imaginary_shift | 0.2 | Sets the CASPT2 imaginary level shift in Hartree, written as IMAG. The default is 0.2. |
| singlet_ci_roots | [3] | Sets the number of roots of the singlet RASSCF block in a SOC run. The default [3] asks for three singlets. |
| singlet_ci_spin | [0] | Sets 2S of the singlet block in a SOC run. The default [0] labels it a singlet. |
| triplet_ci_roots | [2] | Sets the number of roots of the triplet RASSCF block in a SOC run. The default [2] asks for two triplets. |
| triplet_ci_spin | [2] | Sets 2S of the triplet block in a SOC run. The default [2] labels it a triplet. |
| embedding | none | Selects the ESPF embedding: none, charge for point charges or field for an electrostatic field with its gradients. The retired value potential is… |
| read_charge | none | Selects the source of the embedding data: none reads the per-step point_charges and external_potential arrays, true reads &lt;project&gt;.charge, and any… |
| scf_converge | 1e-08 | Sets the SCF density and energy threshold. The default is 1e-8. |
| maxit_scf | 200 | Sets the first number of the RASSCF ITERATIONS card, the CI iteration cap; the second number, the super-CI orbital cap, stays at 100. The default is… |
| opt_maxit | 100 | Sets the SLAPAF relaxation cap and is exported as MOLCAS_MAXITER for the optimization loop. The default is 100 iterations. |
| maxit_md | 1 | Sets the number of MD steps and bounds the FOREACH loop that wraps SEWARD through DYNAMIX. The default is 1 step. |
| md_maxiter | 0 | Sets the MOLCAS_MAXITER export limit of a dynamics run, where 0 derives ten times maxit_md. The value must exceed maxit_md. |
| size | 20.67 | Sets the DYNAMIX time step in atomic units, written as dt. The default 20.67 au is about 0.5 fs. |
| deco | 0.1 | Sets the decoherence parameter of the surface-hopping run, written as decoherence in the SURFACEHOP block. It is a string, 0.1 by default. |
| thermostat | off | Selects the DYNAMIX thermostat: off or nve runs NVE, and nose, nose-hoover or nvt run a Nose-Hoover NVT. Any other value raises. |
| root | 1 | Sets the initial state of the dynamics, spelled 1-based in the input; the MDRLXR card of the RASSCF block is this value. |
| opt_converge | false | Reports whether the optimization converged. The skill writes it as a result and the framework reads it as the run verdict. |
| opt_state_i | 0 | Sets the first optimization target state, 0-based. Equal to opt_state_j it relaxes that state alone; with a distinct same-spin partner it requests a… |
| opt_state_j | None | Sets the second optimization target state, 0-based, where None means a single-state relaxation. A same-spin pair gives a MECI and a singlet-triplet… |
| ir_width | 100 | Sets the half-width in cm-1 recorded with an IR spectrum. The parser reports band positions and intensities per mode, so the value carries the width… |
| ir_scale | 1 | Sets the frequency scaling factor recorded with an IR spectrum. The default 1.0 leaves the harmonic wavenumbers unscaled. |
| uv_width | 100 | Sets the half-width in nm recorded with an absorption spectrum. RASSI reports oscillator strengths per root, and the width carries the half-width of… |
| nics_ring † | [0] | Reserved NICS request: the ring atom indices that would define the scan axis. No molcas profile consumes it. |
| nics_scan † | [0] | Reserved NICS request: the distances in Angstrom along the ring axis. No molcas profile consumes it. |
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
| verbose | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| temperature | 298.15 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html). |
| solvent | gas | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| shape | ellipsoid | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| factor | 40 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| scale | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Selects the run arm: local writes and runs the pymolcas script on the current machine, and slurm writes a batch script and submits it. The default is local.

**shell** - Adds skill-level shell commands to the run script, given as a list, a multi-line string or one line with semicolons. Module loads and environment activations belong here; the framework global shell is reserved for the main run.

**project** - Sets the OpenMolcas project name, exported as MOLCAS_PROJECT. The deck, the geometry, the orbital, the optimized structure and the trajectory files are named after it.

**profile** - Selects the module chain. casscf_energy runs GATEWAY, SEWARD and RASSCF; casscf_gradient and casscf_nac add ALASKA for the analytic gradient and for the coupling of a same-spin pair; casscf_properties runs GATEWAY, SEWARD, RASSCF, ALASKA and RASSI for a free combination of energy, gradient, nac and soc; casscf_geometry_optimization adds SLAPAF and derives single-state, MECI or MECP from opt_state_i, opt_state_j and state_spin; casscf_abinitio_dynamics and casscf_surface_hopping propagate the nuclei inside one Molcas run through DYNAMIX, the second with SURFACEHOP; casscf_soc runs the dual singlet/triplet RASSCF and RASSI; casscf_absorption and caspt2_absorption read RASSI oscillator strengths; and the two _frequency profiles add MCKINLEY. A generic _properties chain accepts only the tokens energy, gradient, nac and soc plus the dipole and transition modifiers, and refuses anything else.

**method** - Sets the multiconfigurational level. hf runs a plain SCF and serves energy and frequency only, with no gradient, optimization, NAC, SOC or dynamics. casscf runs the complete set of profiles, from energy to surface hopping. caspt2 adds the perturbative correction on the CASSCF reference and serves energy, gradient, nac, absorption, soc and frequency, but no geometry optimization, no ab initio dynamics and no surface hopping, because the perturbative gradient path is single-shot. A profile whose prefix does not match method is refused before any deck is written.

**molcas_path** - Names the OpenMolcas installation root. pymolcas is looked for at &lt;path&gt;/bin/pymolcas and then at &lt;path&gt;/pymolcas, and the directory must exist at validation.

**molcas_project** - Carries the project name in the spelling the framework forwards. A non-empty value overwrites project and the two are kept in step.

**molcas_mem** - Sets the memory exported as MOLCAS_MEM in megabytes. The default is 4000 MB.

**molcas_nproc** - Carries the MPI rank count in the spelling the framework forwards. The larger of molcas_nproc and nproc is exported as MOLCAS_NPROCS.

**molcas_print** - Sets the print level exported as MOLCAS_PRINT. The default 3 writes the standard module output, lower values quieten the deck and higher values add the orbital and integral detail OpenMolcas offers.

**nproc** - Sets the MPI rank count of a local run. The larger of nproc and molcas_nproc wins, so neither spelling can shrink a request made in the other.

**thread** - Sets the OpenMP thread count exported as OMP_NUM_THREADS, one thread count per rank. The larger of thread and molcas_thread wins, and the retired spelling num_thread is refused.

**molcas_thread** - Carries the thread count in the spelling the framework forwards; the larger of molcas_thread and thread is applied.

**slurm_time** - Sets the wall-clock limit of the batch job in HH:MM:SS. The default is 01:00:00.

**slurm_partition** - Names the SLURM partition of the batch job. The default is default.

**slurm_ntasks** - Sets the number of SLURM tasks, which becomes the rank count of the submitted job. The default is 1.

**slurm_memory** - Sets the memory request of the batch job in megabytes. The default is 4000.

**slurm_command** - Adds extra commands to the submission line of the batch job.

**keep_tmp** - 0 removes the calculation directory once the results are collected; 1 keeps the deck, the logs and the orbital files.

**basis** - Sets the orbital basis of the SEWARD step. The 24 accepted names cover the Pople sets 3-21g, 4-31g, 5-21g, 6-31g and 6-31g\*\*, the ANO-S, ANO-L and ANO-RCC sequences, the Dunning correlation-consistent sets up to aug-cc-pvqz, and def2-svp, def2-tzvp and def2-tzvpp. A single name applies to every element; a per-element combination joined by commas, such as H.ano-rcc-vdzp, C.ano-rcc-vdzp, validates each component separately and is written element by element.

**nact_elec** - Sets the number of active electrons in the RASSCF, written as NactEl. The default is 2.

**nact_orb** - Sets the number of active orbitals in the RASSCF, written as Ras2 unless the RAS partition overrides it. The default is 2.

**inactive_orb** - Sets the number of inactive orbitals. Zero derives it from (N_elec - nact_elec)/2.

**ras1_orb** - Sets the size of the RAS1 subspace, the holes allowed above it. Zero leaves a plain CAS without RAS partitioning.

**ras2_orb** - Sets the size of the RAS2 subspace. Zero falls back to nact_orb, which is the plain CAS case.

**ras3_orb** - Sets the size of the RAS3 subspace, the particles allowed into it. Zero leaves a plain CAS without RAS partitioning.

**active** - Lists the active orbitals explicitly, 1-based as the input spells them, and the skill writes them as ALTEr pairs. An empty list takes the orbitals from inactive_orb, nact_orb and the RAS partition.

**ci_roots** - Sets the number of CI roots in each spin block, written as ROOT in the RASSCF block. One block is written per ci_spin entry, and the default [1] solves the lowest root.

**ci_spin** - Sets 2S of each spin block, written as Spin in the RASSCF block: 0 for a singlet, 2 for a triplet. The default [0] runs one singlet block.

**state_spin** - Labels the spin of each requested state as 2S, 0 for a singlet and 2 for a triplet. It decides which state sits in which block and which pairs count as same-spin for a NAC.

**nstate** - Sets the number of electronic states requested. It bounds the state ladder the framework builds and the roots the CI blocks must cover.

**xms** - Keeps the legacy multistate switch of the CASPT2 branch: it is the only source of the variant when caspt2_type is empty, resolving to xms when true and to ss when false. An explicit caspt2_type always wins, so the switch matters only for inputs written before the five names were accepted.

**caspt2_type** - Selects how the CASPT2 step treats the reference states. ss writes a single-state calculation and is therefore one state per call, which makes it unable to serve a NAC and, for SOC, limited to one state per spin manifold; ms writes MULTISTATE all, xms writes XMULTISTATE all and is the resolution of the default switch, rms writes RMULTISTATE all, and xdw writes XMULTISTATE all with the DWMS zeta and DWTYPE exponent of the XDW variant. An empty value resolves from xms.

**dwms_zeta** - Sets the zeta parameter of the XDW-CASPT2 weighting, written as DWMS only when caspt2_type is xdw. The value is a positive real number and defaults to 1.0, the vendor recommendation for the smooth interpolation between reference states.

**dwtype** - Selects the exponent form of the XDW-CASPT2 weights, written as DWTYPE next to DWMS: 1, 2 or 3 are the three forms OpenMolcas accepts and 3 is the shipped default. Like dwms_zeta it only reaches a deck whose caspt2_type is xdw.

**nocsf** - Controls the CSF contribution of the non-adiabatic coupling. With true, the default, one NOCSF card is added to the ALASKA block for both the CASSCF and the CASPT2 NAC, so the coupling carries the orbital-relaxation part only; with false the CSF term is kept.

**caspt2_ipea_shift** - Sets the CASPT2 IPEA shift in Hartree. None leaves the OpenMolcas default of 0.25; 0.0 emits IPEA 0.0, the portable setting for analytic gradient and NAC runs because OpenMolcas v25.06 segfaults on the IPEA 0.25 plus DORT path.

**caspt2_real_shift** - Sets the CASPT2 real level shift in Hartree, written as REAL. The default 0.0 applies no shift.

**caspt2_imaginary_shift** - Sets the CASPT2 imaginary level shift in Hartree, written as IMAG. The default is 0.2.

**singlet_ci_roots** - Sets the number of roots of the singlet RASSCF block in a SOC run. The default [3] asks for three singlets.

**singlet_ci_spin** - Sets 2S of the singlet block in a SOC run. The default [0] labels it a singlet.

**triplet_ci_roots** - Sets the number of roots of the triplet RASSCF block in a SOC run. The default [2] asks for two triplets.

**triplet_ci_spin** - Sets 2S of the triplet block in a SOC run. The default [2] labels it a triplet.

**embedding** - Selects the ESPF representation of the environment. charge writes an External block of point charges in Angstrom, field writes one row per atom with the potential and its three gradients in atomic units, and none writes no ESPF block unless read_charge names a file. ESPF is refused for the iterative profiles, the two MD profiles and the two frequency profiles, and for any method other than casscf or caspt2.

**read_charge** - Selects where the embedding data comes from. none takes the per-step point_charges and external_potential arrays, true reads &lt;project&gt;.charge from the calculation or launch directory, and any other value names a file that is fixed for the run. The first line of such a file declares point angstrom or point bohr with external or atom rows, or field au; older charge and potential spellings still read.

**scf_converge** - Sets the SCF density and energy threshold. The default is 1e-8.

**maxit_scf** - Sets the first number of the RASSCF ITERATIONS card, the CI iteration cap; the second number, the super-CI orbital cap, stays at 100. The default is 200.

**opt_maxit** - Sets the SLAPAF relaxation cap and is exported as MOLCAS_MAXITER for the optimization loop. The default is 100 iterations.

**maxit_md** - Sets the number of MD steps and bounds the FOREACH loop that wraps SEWARD through DYNAMIX. The default is 1 step.

**md_maxiter** - Sets the MOLCAS_MAXITER export limit of a dynamics run, where 0 derives ten times maxit_md. The value must exceed maxit_md.

**size** - Sets the DYNAMIX time step in atomic units, written as dt. The default 20.67 au is about 0.5 fs.

**deco** - Sets the decoherence parameter of the surface-hopping run, written as decoherence in the SURFACEHOP block. It is a string, 0.1 by default.

**thermostat** - Selects the DYNAMIX thermostat: off or nve runs NVE, and nose, nose-hoover or nvt run a Nose-Hoover NVT. Any other value raises.

**root** - Sets the initial state of the dynamics, spelled 1-based in the input; the MDRLXR card of the RASSCF block is this value.

**opt_converge** - Reports whether the optimization converged. The skill writes it as a result and the framework reads it as the run verdict.

**opt_state_i** - Sets the first optimization target state, 0-based. Equal to opt_state_j it relaxes that state alone; with a distinct same-spin partner it requests a minimum-energy conical intersection.

**opt_state_j** - Sets the second optimization target state, 0-based, where None means a single-state relaxation. A same-spin pair gives a MECI and a singlet-triplet pair a MECP, with opt_state_i the singlet.

**ir_width** - Sets the half-width in cm-1 recorded with an IR spectrum. The parser reports band positions and intensities per mode, so the value carries the width of a broadened trace rather than a broadening applied inside the deck.

**ir_scale** - Sets the frequency scaling factor recorded with an IR spectrum. The default 1.0 leaves the harmonic wavenumbers unscaled.

**uv_width** - Sets the half-width in nm recorded with an absorption spectrum. RASSI reports oscillator strengths per root, and the width carries the half-width of the broadened trace.

**nics_ring** - Reserved NICS request: the ring atom indices that would define the scan axis. No molcas profile consumes it. This keyword is accepted but has no effect in v3.0.0.

**nics_scan** - Reserved NICS request: the distances in Angstrom along the ring axis. No molcas profile consumes it. This keyword is accepted but has no effect in v3.0.0.

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

**verbose** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**temperature** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.3](05-03-dynamics.html).

**solvent** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**shape** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**factor** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**scale** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

OpenMolcas is a molecular code here: any true pbc is refused at validation, so cell, lattice_vector and kpoint never reach a deck, and a periodic or extended workflow belongs to cp2k, xtb or lammps. The method matrix is narrow in the same way: hf serves energy and frequency only, and CASPT2 serves no geometry optimization, no ab initio dynamics and no surface hopping, so an optimized or propagated CASPT2 trajectory is a sequence of single-point calls rather than an internal loop.

Several couplings are restricted. ss cannot serve a NAC at all, a NAC needs a same-spin pair, mixed spin manifolds need an explicit SOC request, and MECP needs opt_state_i on the singlet and opt_state_j on the triplet. ESPF is refused for the iterative profiles and omits the ESPF grid-point derivatives from its gradients, and under ESPF a CASPT2 gradient plus NAC becomes two passes with the NAC labelled a no-ESPF approximation. The returned energy is the pre-SOC CASPT2 energy, never a SO-RASSI eigenvalue, and a single-root spin block keeps its CASSCF diagonal in RASSI with a warning.
