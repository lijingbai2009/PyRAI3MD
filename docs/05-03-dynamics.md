---
title: "5.3. DYNAMICS"
nav_order: 3
parent: "5. Keyword sections"
---

The dynamics section configures the MD and surface-hopping loop: the thermodynamic ensemble, the initial conditions, the number of steps and the timestep, the hopping algorithm and its couplings, and the guards that end a trajectory. The classic spelling is &amp;MD; in YAML and agent JSON it is the dynamics mapping, and the historical name md selects the same model.

The section is read by a run whose jobtype is md or surface_hopping, and by the md stage of an adaptive run; every other workflow ignores it. [Section 6](06-workflows.html) gives the step order of the integrator and [Section 8](08-periodic-systems.html) the ensemble algorithms behind the thermostat and barostat keywords.

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&MD**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| ensemble | nve | Selects the thermodynamic ensemble: nve integrates the microcanonical equations, nvt adds a particle thermostat and npt adds a barostat as well. The… |
| thermostat | off | Selects the particle thermostat: off, the default, nose-hoover, nose-hoover-chain or csvr, with the aliases nh and nose for the first, nhc for the… |
| thermostat_chain | 1 | Sets the length of the Nose-Hoover chain, an integer of at least 1. A value of 1 is a single Nose-Hoover thermostat and every further link adds one… |
| thermostat_tau | 100 | Sets the coupling time of the particle thermostat in femtoseconds, must be positive. It fixes the thermostat mass through the number of degrees of… |
| thermostat_mass | None | Overrides the mass of the first thermostat link in atomic units. An empty value derives the mass from thermostat_tau and the number of degrees of… |
| thermodelay | 200 | Sets how many steps the thermostat stays off after the trajectory leaves an excited state. It is read together with dissipate and counts steps of the… |
| dissipate | 0 | Turns on the gated thermostat of an excited-state trajectory: 1 holds the thermostat off while the run is on an excited state and for thermodelay… |
| barostat | berendsen | Selects the cell algorithm of an npt run: off, berendsen, the default, parrinello-rahman, mttk or scr, with the aliases pr for the second and… |
| barostat_mode | isotropic | Selects the cell coupling: isotropic scales the volume with one scalar coordinate and anisotropic relaxes the full lattice tensor. The default is… |
| barostat_tau | 1000 | Sets the coupling time of the barostat in femtoseconds, must be positive. It fixes the cell mass together with the compressibility unless… |
| barostat_mass | None | Overrides the cell mass in atomic units. An empty value derives it from the instantaneous volume, the compressibility and barostat_tau. scr carries… |
| compressibility | 4.5e-05 | Sets the isothermal compressibility of the sample in inverse bar, must be positive. The default 4.5e-5 is the value of liquid water, and the barostat… |
| temperature | 300 | Sets the target temperature in kelvin and must be at least 0. It is the target of the thermostat, the temperature of a Maxwell-Boltzmann velocity… |
| pressure | 1 | Sets the target pressure of an npt run in bar. It is the reference the barostat drives the internal pressure to, and it has no effect outside npt. |
| root | 0 | Sets the initial electronic state, written 1-based so that 1 is the ground state and stored 0-based. A scalar is the single-center value and a list… |
| activestate | 0 | Selects which gradients a step requests: 0 asks for every state and 1 asks for the active state only. Global surface hopping needs the all-state… |
| initcond | 0 | Selects the source of the initial conditions: 0 or none uses the geometry and velocities the state already carries, read or load takes frames from… |
| ninitcond | 1 | Sets the number of trajectories, one per initial condition, and must be at least 1. Each trajectory is planned as its own task with its own runtime… |
| format | molden | Selects the format of the frequency record that Wigner and Boltzmann sampling read: molden, the default, g16, orca or bagel. The file is… |
| method † | wigner | Records the legacy sampling method name, wigner by default. The sampling is selected by initcond, which supersedes this keyword. |
| randvelo | 0 | Turns on a Maxwell-Boltzmann velocity draw at the start of a trajectory: 1 draws every Cartesian component with a width of the square root of the… |
| excess | 0 | Sets the kinetic energy to add to the first step, in Hartree. It is one of three spellings of the same request and may not be combined with scale or… |
| scale | 1 | Sets the factor applied to the kinetic energy of the first step, dimensionless, with 1.0 leaving the velocities untouched. It may not be combined… |
| target | 0 | Sets the kinetic energy that the first step is brought to, in Hartree. It may not be combined with excess or scale: the three keywords describe one… |
| reset | 0 | Turns on the removal of the rigid rotation of the velocities: 1 diagonalizes the mass-weighted inertia tensor and removes the rotational part, 0… |
| resetstep | 0 | Sets the cadence of the rotational reset in steps: 0 resets on the first step only and a non-zero value resets on every step whose number is a… |
| step | 10 | Sets the number of steps this process integrates, a non-negative integer. A resumed run adds it to the step its runtime file carries, so the same… |
| size | 20.67 | Sets the integration timestep in atomic units of time, must be positive. The default 20.67 au is about 0.5 fs, and the clock columns of the output… |
| history | 4 | Sets the number of state snapshots kept for the integrators. It must be at least 1, and any value below 4 is raised to 4 because the electronic… |
| sfhp | nosh | Selects the hopping algorithm: nosh, the default, runs no hopping, gsh selects global surface hopping, fssh fewest-switches surface hopping and lzsh… |
| nactype | ktdc | Selects the coupling source of the fewest-switches integrator: nac uses the nonadiabatic coupling vectors and ktdc the curvature form built from the… |
| phasecheck | 1 | Turns on the phase correction of the couplings, 1 by default. Each step compares the overlap of a coupling pair with the previous step and flips the… |
| substep | 20 | Sets the number of electronic substeps per nuclear step, at least 1. The electronic equations are propagated with a step of the nuclear timestep… |
| rk4 | 0 | Selects the electronic propagator: the string 0, the default, takes the first-order local step, and any other value uses fourth-order Runge-Kutta. |
| integrate | 0 | Selects how the electronic propagation treats the accumulated hopping matrix: 0, the default, resets it at every substep and a non-zero value… |
| deco | 0.1 | Sets the decoherence parameter of the fewest-switches kernel, in Hartree, and reads a number as a string. The default 0.1 enters the coherence time… |
| adjust | 1 | Selects the velocity policy after a hop: 0 leaves the velocity unchanged, 1 rescales the whole velocity so that the total energy is conserved, and 2… |
| reflect | 1 | Selects what a frustrated hop does, that is a hop that lacks the kinetic energy to reach the upper state: 1 reverses the whole velocity and 2… |
| maxh | 10 | Limits how far a hop may jump, counted in state indices: only a target whose index differs from the active one by at most this value is eligible. The… |
| gap | 0.5 | Sets the energy gap below which an internal-conversion hop is considered, in eV. The value is converted to Hartree when the integrator is built, and… |
| gapsoc | 0.5 | Sets the energy gap below which an intersystem-crossing hop is considered, in eV. It is the spin-forbidden counterpart of gap and is converted to… |
| addsoc | 0 | Adds a constant bias to every spin-orbit coupling element of one evaluation, in inverse centimetres. The bias is applied while the result is… |
| dosoc | 0 | Turns on the spin-orbit couplings: 1 adds the coupling capability to the plan of every step and 0 leaves it out. It is also the gate of addsoc, so… |
| dodipole | 0 | Turns on the dipole of every evaluation: 1 adds the dipole capability to the plan. The dipoles are records of the step and do not enter the… |
| radiative | 0 | Requests transition dipoles and oscillator strengths when its value is 1, true or yes. Any other value leaves them out of the plan. Like dodipole it… |
| fssh_rk4 | numpy | Selects the implementation of the Runge-Kutta kernel, numpy for the reference form or numba for the compiled form. It is one of five per-function… |
| fssh_dpdt | numpy | Selects the implementation of the density-derivative kernel, numpy or numba. It is the kernel that propagates the electronic density matrix of one… |
| fssh_matb | numpy | Selects the implementation of the hopping-matrix kernel, numpy or numba. It is the kernel that builds the switching probabilities of one substep. |
| fssh_ktdc | numpy | Selects the implementation of the curvature-coupling kernel, numpy or numba. It is read only by a run whose nactype is ktdc. |
| fssh_decoherence | numpy | Selects the implementation of the decoherence kernel, numpy or numba. It is read only by a run whose deco value enables decoherence. |
| fssh_force_hop | 0 | Sets the hopping random number to zero so that any positive probability hops. It is a debug switch, never used in production runs, and is isolated… |
| fssh_profile | 0 | Turns on the function-level timing of the electronic kernels and writes the implementation table beside the run log. 0, the default, times nothing. |
| perf_profile | 0 | Turns on the per-step phase table, which is written to pyrai3md_step.perf in the working directory. 0, the default, writes no table. |
| md_maxenergy | None | Sets the energy deviation above which the trajectory stops, in Hartree, when the backend reports a prediction deviation. An empty value leaves the… |
| md_maxgrad | None | Sets the gradient deviation above which the trajectory stops, in Hartree per Bohr. An empty value leaves the channel unjudged, and an adaptive md… |
| md_maxnac | None | Sets the coupling deviation above which the trajectory stops, in Hartree per Bohr. An empty value leaves the channel unjudged, and an adaptive md… |
| md_maxsoc | None | Sets the spin-orbit deviation above which the trajectory stops, in inverse centimetres. An empty value leaves the channel unjudged, and an adaptive… |
| md_maxpop | 0.01 | Sets the population tolerance: a population outside the interval from minus the tolerance to one plus the tolerance ends the trajectory after the… |
| direct | 2000 | Sets the number of steps recorded one by one from the start: up to this step every frame is written. After it buffer decides the cadence, and the… |
| buffer | 500 | Sets the recording cadence after the direct window: every buffer-th step is written, counted from the end of that window. The default is 500, the… |
| checkpoint | 0 | Sets the cadence of the runtime file &lt;title&gt;.rt in steps: a step whose absolute number is a multiple of the value writes the file, and the last step… |
| multitask | None | Turns on the multiprocessing fan-out of this workflow units: true runs the trajectories in a spawn pool and false runs them one after another in… |
| remotetask | None | Submits each trajectory of this workflow to Slurm: true writes a batch script per unit and collects its result record. It needs sbatch and sacct on… |
| multipool | 1 | Sets the number of pool workers of this workflow fan-out, at least 1. It is the width of the pool and not the number of units; the default 1 keeps… |

Full descriptions for all available keywords are summarized below.

**ensemble** - Selects the thermodynamic ensemble and therefore which of the thermostat and barostat keywords are read. nve is the default microcanonical run: it requires a thermostat of off and refuses dissipate, because there is no thermostat to delay. nvt adds a particle thermostat and ignores the barostat; npt requires both a barostat and a thermostat, and it also needs a stress from the backend, which in practice means a periodic provider. The ensemble files &lt;title&gt;.md.ensemble and &lt;title&gt;.sh.ensemble are created for nvt and npt runs only.

**thermostat** - Selects the particle thermostat of an nvt or npt run. The values are off, nose-hoover, nose-hoover-chain and csvr, with nh, nose, nhc and csnr accepted as aliases; the same vocabulary is written in barostat terms by the pairing table, which refuses a combination with no derivation. parrinello-rahman and mttk are extended-cell methods whose cell dynamics carries a Nose-Hoover chain, so they accept only the two Nose-Hoover thermostats; berendsen and scr accept any thermostat that is not off. The default off belongs to nve, and nvt or npt refuses it.

**thermostat_chain** - Sets the length of the Nose-Hoover chain, an integer of at least 1. A value of 1 is a single Nose-Hoover thermostat and every further link adds one thermostat to the chain. An MTTK run thermalizes the cell momentum with a chain of the same length.

**thermostat_tau** - Sets the coupling time of the particle thermostat in femtoseconds, must be positive. It fixes the thermostat mass through the number of degrees of freedom and the target temperature unless thermostat_mass overrides it. A chain coupled faster than the motion it thermostats pumps that motion instead of damping it.

**thermostat_mass** - Overrides the mass of the first thermostat link in atomic units. An empty value derives the mass from thermostat_tau and the number of degrees of freedom, which is what a normal run uses; the further links of a chain keep their derived masses.

**thermodelay** - Sets how many steps the thermostat stays off after the trajectory leaves an excited state. It is read together with dissipate and counts steps of the run, not steps on the excited state.

**dissipate** - Turns on the gated thermostat of an excited-state trajectory: 1 holds the thermostat off while the run is on an excited state and for thermodelay steps after it lands on the ground state, so the excess energy leaves with the nuclear motion. The barostat is not gated. nve refuses a non-zero value because it has no thermostat to hold.

**barostat** - Selects the cell algorithm of an npt run and is inert in any other ensemble. berendsen, the default, is a weak-coupling relaxation with no ensemble claim and is the right choice for equilibration. parrinello-rahman and mttk are deterministic extended-Lagrangian cell dynamics and need a Nose-Hoover thermostat on the particles; mttk adds a chain on the cell momentum. scr is a first-order stochastic cell equation with no cell momentum, derived to pair with any thermostat, and it is isotropic only.

**barostat_mode** - Selects the cell coupling: isotropic scales the volume with one scalar coordinate and anisotropic relaxes the full lattice tensor. The default is isotropic. scr supports the isotropic form only and refuses anisotropic, because its tensor form shears a liquid with no restoring modulus.

**barostat_tau** - Sets the coupling time of the barostat in femtoseconds, must be positive. It fixes the cell mass together with the compressibility unless barostat_mass overrides it. For scr it is the volume relaxation time of a first-order cell equation, so it is not an oscillation period there.

**barostat_mass** - Overrides the cell mass in atomic units. An empty value derives it from the instantaneous volume, the compressibility and barostat_tau. scr carries no cell mass at all, so the keyword does not apply to it.

**compressibility** - Sets the isothermal compressibility of the sample in inverse bar, must be positive. The default 4.5e-5 is the value of liquid water, and the barostat derives its cell mass from it.

**temperature** - Sets the target temperature of the run in kelvin, and it is the one value three mechanisms read: the thermostat drives the instantaneous temperature to it, randvelo draws velocities from a Maxwell-Boltzmann distribution at it, and the Wigner and Boltzmann initial condition samplers populate the vibrational states at it. Zero is accepted and a negative value is refused. An nve run has no thermostat to read it, so the keyword there only seeds the draws.

**pressure** - Sets the target pressure of an npt run in bar. It is the reference the barostat drives the internal pressure to, and it has no effect outside npt.

**root** - Sets the initial electronic state of the trajectory. It is written 1-based, so 1 is the ground state, and the adapter decrements it once into the 0-based index the run uses; a value below 1 in the input and a state outside the ladder built from ci are both refused. A scalar is the single-center spelling and a list gives one initial state per center of a multi-center run. The run fills the current state of the system from root when that field is left at its default. A hop rewrites that current state and never root, so this keyword records where the trajectory started while the runtime ledger records where it is.

**activestate** - Selects which gradients a step requests: 0 asks for every state and 1 asks for the active state only. Global surface hopping needs the all-state form, Landau-Zener hopping needs energies only, and fewest-switches hopping works with either.

**initcond** - Selects where the initial conditions come from. 0, the default, and none use the geometry and velocities the state already carries, and with more than one trajectory the same condition is repeated. read and load take ninitcond frames from &lt;title&gt;.init.xyz, whose atom lines are an element symbol followed by six numbers, position in Angstrom and velocity in Bohr per atomic unit; fewer frames than asked for is an error and extra frames are truncated. wigner and boltzmann sample the frequency record &lt;title&gt;.freq.&lt;format&gt; at the target temperature, seeded by the global seed of the control section, and the legacy value 1 is read as wigner.

**ninitcond** - Sets the number of trajectories, one per initial condition, and must be at least 1. Each trajectory is planned as its own task with its own runtime directory. With initcond 0 the trajectories repeat the same initial condition byte for byte.

**format** - Selects the format of the frequency record that Wigner and Boltzmann sampling read: molden, the default, g16, orca or bagel. The file is &lt;title&gt;.freq.&lt;format&gt; in the working directory, and the title comes from the control section.

**method** - Records the legacy sampling method name, wigner by default. The sampling is selected by initcond, which supersedes this keyword. This keyword is accepted but has no effect in v3.0.0.

**randvelo** - Turns on a Maxwell-Boltzmann velocity draw at the start of a trajectory: 1 draws every Cartesian component with a width of the square root of the thermal energy over the atomic mass. The draw replaces the velocities the state carries and needs no frequency record, which is what separates it from vibrational sampling.

**excess** - Sets the kinetic energy to add to the first step, in Hartree. It is one of three spellings of the same request and may not be combined with scale or target. A restarted segment applies neither this nor the velocity draw.

**scale** - Sets the factor applied to the kinetic energy of the first step, dimensionless, with 1.0 leaving the velocities untouched. It may not be combined with excess or target.

**target** - Sets the kinetic energy that the first step is brought to, in Hartree. It may not be combined with excess or scale: the three keywords describe one request and only one of them may be written.

**reset** - Turns on the removal of the rigid rotation of the velocities: 1 diagonalizes the mass-weighted inertia tensor and removes the rotational part, 0 leaves the velocities as they are. The translation is subtracted on every step of every run, and a periodic run never takes the rotational half.

**resetstep** - Sets the cadence of the rotational reset in steps: 0 resets on the first step only and a non-zero value resets on every step whose number is a multiple of it. It is read only when reset is 1.

**step** - Sets the number of steps this process integrates, a non-negative integer, and the default 10 is a short smoke run rather than a production length. It bounds the loop directly, so the number of recorded frames is a function of it together with direct and buffer. A resumed segment adds its own step count to the step the runtime file carries: step is the length of one process, not of the whole trajectory.

**size** - Sets the integration timestep in atomic units of time, which is the unit of the velocity array; 20.67 au is about half a femtosecond and is the default. The trajectory clock printed in the energies and population files is the step number minus one times this value, so step 1 sits at time zero. A value must be positive, and a timestep large enough to break the electronic substeps is a physical choice rather than something the framework rescales.

**history** - Sets the number of state snapshots kept for the integrators. It must be at least 1, and any value below 4 is raised to 4 because the electronic dynamics needs the last four steps.

**sfhp** - Selects the hopping algorithm of the trajectory. nosh, the default, propagates the nuclei on one surface and runs no hopping at all. gsh is global surface hopping, which needs all-state gradients and rebuilds a pure-state density, and it uses neither decoherence nor electronic substeps. fssh is fewest-switches surface hopping, the algorithm that reads nactype, substep, rk4, deco, adjust, reflect and maxh. lzsh is Landau-Zener surface hopping, which needs state energies only and rescales the velocity isotropically on every hop. The algorithm is a framework keyword and is never passed to the skill.

**nactype** - Selects the coupling source that the fewest-switches integrator consumes. nac uses the nonadiabatic coupling vectors, which the plan requests as their own capability; ktdc, the default, uses the curvature form built from the energy gap and the gradients, so a step that needs no coupling vectors can still hop. The integrator accepts only these two values and refuses anything else when it is initialized. The overlap-based dcm spelling is known to the planner, which then asks for an overlap capability, but no integrator implements it and the run fails at that same point.

**phasecheck** - Turns on the phase correction of the couplings, 1 by default. Each step compares the overlap of a coupling pair with the previous step and flips the sign of the pairs that came out negative, while a pair whose overlap is at the noise level is left alone.

**substep** - Sets the number of electronic substeps per nuclear step, at least 1. The electronic equations are propagated with a step of the nuclear timestep divided by this number, and the default 20 resolves the phase within one nuclear step.

**rk4** - Selects the electronic propagator: the string 0, the default, takes the first-order local step, and any other value uses fourth-order Runge-Kutta.

**integrate** - Selects how the electronic propagation treats the accumulated hopping matrix: 0, the default, resets it at every substep and a non-zero value accumulates it across the substeps of one nuclear step.

**deco** - Sets the decoherence parameter of the fewest-switches kernel, in Hartree, and it is read as a string so that the word off can turn the correction off. The parameter enters the coherence time of a pair as the inverse gap times one plus the parameter over the kinetic energy, which is the OpenMolcas form of the correction. The default 0.1 is the legacy value, and the kernel that applies it is selected separately by fssh_decoherence.

**adjust** - Selects the velocity policy after a hop: 0 leaves the velocity unchanged, 1 rescales the whole velocity so that the total energy is conserved, and 2 rescales the component along the coupling vector. The default is 1.

**reflect** - Selects what a frustrated hop does, that is a hop that lacks the kinetic energy to reach the upper state: 1 reverses the whole velocity and 2 reflects it along the coupling vector. The default is 1.

**maxh** - Limits how far a hop may jump, counted in state indices: only a target whose index differs from the active one by at most this value is eligible. The default 10 leaves every state of a small ladder reachable.

**gap** - Sets the energy gap below which an internal-conversion hop is considered, in eV. The value is converted to Hartree when the integrator is built, and the default 0.5 eV is the legacy threshold.

**gapsoc** - Sets the energy gap below which an intersystem-crossing hop is considered, in eV. It is the spin-forbidden counterpart of gap and is converted to Hartree when the integrator is built.

**addsoc** - Adds a constant bias to every spin-orbit coupling element of one evaluation, in inverse centimetres. The bias is applied while the result is finalized, so the matrix the step records is the biased one, and it is applied only to a run that asked for couplings with dosoc.

**dosoc** - Turns on the spin-orbit couplings: 1 adds the coupling capability to the plan of every step and 0 leaves it out. It is also the gate of addsoc, so without it nothing is computed and nothing is biased.

**dodipole** - Turns on the dipole of every evaluation: 1 adds the dipole capability to the plan. The dipoles are records of the step and do not enter the propagation.

**radiative** - Requests transition dipoles and oscillator strengths when its value is 1, true or yes. Any other value leaves them out of the plan. Like dodipole it extends what a step computes rather than how it propagates.

**fssh_rk4** - Selects the implementation of the Runge-Kutta kernel, numpy for the reference form or numba for the compiled form. It is one of five per-function kernel switches and changes the speed of a trajectory, not its result.

**fssh_dpdt** - Selects the implementation of the density-derivative kernel, numpy or numba. It is the kernel that propagates the electronic density matrix of one substep.

**fssh_matb** - Selects the implementation of the hopping-matrix kernel, numpy or numba. It is the kernel that builds the switching probabilities of one substep.

**fssh_ktdc** - Selects the implementation of the curvature-coupling kernel, numpy or numba. It is read only by a run whose nactype is ktdc.

**fssh_decoherence** - Selects the implementation of the decoherence kernel, numpy or numba. It is read only by a run whose deco value enables decoherence.

**fssh_force_hop** - Sets the hopping random number to zero so that any positive probability hops. It is a debug switch, never used in production runs, and is isolated from the electronic integration mode of integrate.

**fssh_profile** - Turns on the function-level timing of the electronic kernels and writes the implementation table beside the run log. 0, the default, times nothing.

**perf_profile** - Turns on the per-step phase table, which is written to pyrai3md_step.perf in the working directory. 0, the default, writes no table.

**md_maxenergy** - Sets the energy deviation above which the trajectory stops, in Hartree, when the backend reports a prediction deviation. An empty value leaves the channel unjudged, and the md stage of an adaptive run has it overwritten from the adaptive section.

**md_maxgrad** - Sets the gradient deviation above which the trajectory stops, in Hartree per Bohr. An empty value leaves the channel unjudged, and an adaptive md child reads the bound from its adaptive section instead.

**md_maxnac** - Sets the coupling deviation above which the trajectory stops, in Hartree per Bohr. An empty value leaves the channel unjudged, and an adaptive md child reads the bound from its adaptive section instead.

**md_maxsoc** - Sets the spin-orbit deviation above which the trajectory stops, in inverse centimetres. An empty value leaves the channel unjudged, and an adaptive md child reads the bound from its adaptive section instead.

**md_maxpop** - Sets the population tolerance of the resident guard, which is the one detection channel that is on by default. The interval read is minus the tolerance to one plus the tolerance, and the first population outside it ends the trajectory after the offending frame is written, with stop code 1. An empty value turns the guard off, and a negative or non-finite value is refused rather than reinterpreted, because a negative tolerance would move both ends inward and refuse every physical population. Unlike the four deviation thresholds, this bound is not overwritten in the md child of an adaptive run: the parent reads the stop code and only decides whether to collect the structure.

**direct** - Sets the number of steps recorded one by one from the start: up to this step every frame is written. After it buffer decides the cadence, and the default is 2000.

**buffer** - Sets the recording cadence after the direct window: every buffer-th step is written, counted from the end of that window. The default is 500, the last step of a run is always written, and hop frames are never throttled.

**checkpoint** - Sets the cadence of the runtime file &lt;title&gt;.rt in steps: a step whose absolute number is a multiple of the value writes the file, and the last step of a run always writes it. 0, the default, writes no file during the run.

**multitask** - Turns on the multiprocessing fan-out of this workflow units: true runs the trajectories in a spawn pool and false runs them one after another in process. An empty value keeps the workflow default. It decides how the units run, never how many; ninitcond sets the number.

**remotetask** - Submits each trajectory of this workflow to Slurm: true writes a batch script per unit and collects its result record. It needs sbatch and sacct on the path and a slurm section or the environment keys of the input. An empty value keeps the workflow default.

**multipool** - Sets the number of pool workers of this workflow fan-out, at least 1. It is the width of the pool and not the number of units; the default 1 keeps the fan-out serial.

The four deviation thresholds md_maxenergy, md_maxgrad, md_maxnac and md_maxsoc judge only the deviation keys a backend reports, so a molecular-mechanics or quantum-chemistry run whose provider reports none leaves them all unjudged. The last three keywords of the table are shared with every other workflow section: multitask and remotetask choose how the units of a run are executed and multipool chooses the width of the pool.

The retired spellings of this section, among them restart, addstep, record, thermo and popreset, are not accepted: each is refused with the keyword that replaced it, and the differences are collected in [Appendix E](appendix-e-migrating-pyrai2md-inputs.html).
