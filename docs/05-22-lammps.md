---
title: "5.22. LAMMPS"
nav_order: 22
parent: "5. Keyword sections"
---

The lammps section configures the lammps skill, the classical force-field layer of the package. It is read when lammps appears in the theory queue of &CONTROL and is spelled &LAMMPS in the classic input, lammps in YAML and agent JSON.

The skill never builds a force field. The topology file &lt;project&gt;.data and the optional input templates are supplied by the user in the launch directory, and the section selects only the profile, the QM/MM partition and the runtime. The keys are grouped as the run arm and program, the engine controls, the output and selection keys, and the framework parity keys documented in [Section 5.2](05-02-system-molecule.html).

The skill exposes the following profiles. A profile fixes the method and the set of properties a calculation returns.

| Profile | Capability | Method |
| :--- | :--- | :--- |
| ff_energy | energy | ff |
| ff_gradient | gradient | ff |
| ff_optimize | geometry_optimization | ff |
| ff_optimize_cell | geometry_optimization | ff |
| ff_nve | classical_dynamics | ff |
| ff_nvt | classical_dynamics | ff |
| ff_npt | classical_dynamics | ff |
| qmmm_field | gradient | ff |
| qmmm_energy | gradient | ff |

The keywords, default values, and short descriptions are listed below.

**&LAMMPS**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Chooses the run arm. local runs LAMMPS as a subprocess, or inside the resident service when python_enabled asks for it; slurm submits one blocking… |
| shell | '' | Sets the skill-level shell commands, one command per line, copied into the run or batch script before LAMMPS starts. The module loads that put libmpi… |
| project | lammps_calc | Sets the artifact stem of the run. It follows lammps_project, which is the single source of truth, so the two spellings always name the same files. |
| lammps_project | LAMMPS | Sets the project stem of the user files and of every artifact: the topology file &lt;project&gt;.data, the optional templates &lt;project&gt;.in… |
| lammps_path | '' | Sets the LAMMPS installation: a directory that is expected to hold bin/lmp, a path to the binary itself, or an empty value that searches lmp in PATH. |
| python_enabled | false | Asks for resident execution, where a service process holds one LAMMPS instance and receives each step of geometry over a Unix socket. It takes effect… |
| python_path | '' | Sets the interpreter of the resident service process, that is the python that can import lammps and matches the MPI family of lammps_path. An empty… |
| debug | false | Writes the cold-path result files even on the resident path, so the two trajectories can be compared file for file. It costs the per-step I/O the… |
| driver | skill | Records who owns the geometry, skill or main. The framework sets it and a skill section never writes it; a section read on its own has no framework… |
| keep_tmp | 0 | Keeps the calculation directory when 1, and clears it when the backend or the resident service retires when 0. The directory holds the compressed… |
| thread | 1 | Sets the OpenMP threads per rank, exported as OMP_NUM_THREADS. lammps_thread is the framework spelling, and the larger value of a written pair is… |
| lammps_thread | 1 | Sets the OpenMP thread count through the framework spelling. It folds into thread, the larger value winning, so a request made in either spelling is… |
| nproc | 1 | Sets the number of MPI ranks. Above 1 the run starts with mpirun -np N; lammps_nproc is the framework spelling and the larger of the pair wins. |
| lammps_nproc | 1 | Sets the MPI rank count through the framework spelling. It folds into nproc, and in resident mode it is the world of the service process. |
| slurm_time | 01:00:00 | Sets the wall-clock limit of the submitted job in hh:mm:ss. It is read only when execution_mode is slurm. |
| slurm_partition | '' | Sets the scheduler partition of the submitted job. An empty value writes no partition directive, so the cluster default answers. |
| slurm_ntasks | 1 | Sets the number of tasks of the submitted job, which the batch template forwards as the MPI rank count. |
| slurm_memory | 4000 | Sets the memory per node of the submitted job in MB. |
| slurm_command | '' | Sets the submission command of the batch script, for example sbatch or a site wrapper. An empty value uses the default. |
| profile | ff_energy | Selects the task: ff_energy, ff_gradient, ff_optimize, ff_optimize_cell, ff_nve, ff_nvt, ff_npt, qmmm_field or qmmm_energy. The profile fixes what… |
| method | ff | Selects the method. Only ff is accepted, because LAMMPS serves the classical force-field layer and never a QM layer. |
| lammps_units | real | Declares the unit style of the user input: real (kcal/mol, Angstrom, fs, atm) or metal (eV, Angstrom, ps, bar). It is not a conversion switch; the… |
| ewald_accuracy | 1e-8 | Sets the accuracy token written onto every kspace_style line the skill re-issues after read_data, in scientific notation such as 1e-8. It is not a… |
| lammps_shake | 0 | Turns the shake fix defined by the input template on (1) or off (0). It is refused for ff_energy and ff_gradient, where the framework propagates and… |
| shake_fix_id | fShakeTIP3P | Names the shake fix the input template defines, fShakeTIP3P by default. The skill renumbers or removes that fix instead of adding a second one. |
| shake_keywords | '' | Holds a raw fix ... shake ... command that is injected when lammps_shake is 1 and the templates define no such fix. An empty value in that situation… |
| box_relax | off | Selects the lattice degrees of freedom an engine-owned minimization may move: off keeps the box, iso scales it, aniso lets each axis answer for… |
| box_relax_pressure | 0 | Sets the target pressure in bar of the box-relax fix, converted to the pressure unit the run style reads at the point of writing. |
| min_style | cg | Sets the minimization algorithm written as min_style, cg by default. Any style the installed LAMMPS build accepts may be named, for example sd… |
| min_etol | 0.0001 | Sets the energy tolerance of minimize, in the native energy unit of the run. The default is 1e-4. |
| min_ftol | 1e-08 | Sets the force tolerance of minimize, in the native force unit of the run. The default is 1e-8. |
| min_maxiter | 1000 | Caps the number of minimization iterations, passed as the maxiter argument of minimize. The default is 1000. |
| min_maxeval | 1000000 | Caps the number of force evaluations, passed as the maxeval argument of minimize. The default is 1000000. |
| barostat_mode | isotropic | Selects how the NPT fix couples to the cell: isotropic scales the volume as one scalar, anisotropic lets the three axes move on their own. The… |
| thermo_damp | 100 | Sets the thermostat damping time of the NVT and NPT fixes, in the native time unit of the run. |
| pressure | 1 | Sets the target pressure of the NPT fix, read in the pressure unit of the run style: atm for real and bar for metal. |
| pressure_damp | 1000 | Sets the barostat damping time of the NPT fix, in the native time unit of the run. It is validated as positive for ff_npt. |
| md_nstep | 100 | Sets the number of dynamics steps of the MD profiles, written as run N. The value must be positive. |
| md_timestep | 1 | Sets the timestep of the MD profiles in the native time unit of the run: fs for real and ps for metal. The value must be positive. |
| md_seed | 12345 | Seeds the initial velocities through velocity all create when the system carries no velocities of its own. |
| dump_freq | 100 | Sets how often the trajectory and the thermo output are written, in steps. The value must be positive. |
| stress | false | Requests the whole-system stress tensor. The deck adds a virial-only pressure compute whose six components are returned tensile-positive in bar; the… |
| atom_selection | [] | Lists the 1-based topology rows this task evaluates; an empty list means the whole system. A multi-center run gives every task its own subsystem, and… |
| qm_atoms | [] | Lists the QM atoms as 0-based indices in the framework order. The partition must be non-empty, must not cover the whole system, and is mutually… |
| qm_probe | [] | Lists the atoms a qmmm_field probe measures, as a subset of qm_atoms. Empty measures the whole QM set, which is the single-center case. |
| qm_centers | [] | Tiles qm_atoms into centers, one group per center, so qmmm_energy removes the in-cell Q-Q pairs inside each center only. Empty means one center. |
| q_qm | [] | Gives the QM zero-order charges in electron units, in qm_atoms order; the length has to equal the QM set. qmmm_energy writes them into the data file. |
| qmmm_field_file | '' | Points at a four-column ESPF file holding precomputed V and E for a standalone qmmm_energy run. A runtime caller passes the same arrays through the… |
| charges | 0 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| multiplicity | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| nstate | 1 | Sets the number of electronic states. The force-field layer solves one surface, so the normalization sets the value to 1 and states other than the… |
| ci | [1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| spin | [0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| coupling | [] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| verbose | 4 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| solvent | gas | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| temperature | 298.15 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| shape | ellipsoid | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| factor | 40 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| scale | 1 | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| pbc | [0, 0, 0] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| cell | [99, 99, 99, 90, 90, 90] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| lattice_vector | [[0, 0, 0], [0, 0, 0], [0, 0, 0]] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |
| kpoint | [1, 1, 1] | Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html). |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Chooses the run arm. local runs LAMMPS as a subprocess, or inside the resident service when python_enabled asks for it; slurm submits one blocking sbatch job per calculation.

**shell** - Sets the skill-level shell commands, one command per line, copied into the run or batch script before LAMMPS starts. The module loads that put libmpi on the search path belong here, because the resident service inherits them.

**project** - Sets the artifact stem of the run. It follows lammps_project, which is the single source of truth, so the two spellings always name the same files.

**lammps_project** - Sets the project stem of the user files and of every artifact: the topology file &lt;project&gt;.data, the optional templates &lt;project&gt;.in, &lt;project&gt;.in.init and &lt;project&gt;.in.settings, the generated decks, the logs and the dumps.

**lammps_path** - Sets where LAMMPS lives. A directory is read as an installation whose bin/lmp is the binary, a file path is taken as the binary itself, and an empty value searches lmp in PATH. lammps_path and python_path have to name one installation when the resident path is used: liblammps links exactly one MPI implementation, and a launcher from another family leaves every rank its own singleton world, which the service detects and refuses.

**python_enabled** - Asks for resident execution, where a service process holds one LAMMPS instance and receives each step of geometry over a Unix socket. It takes effect only with a non-slurm arm, and only ff_energy, ff_gradient, qmmm_field and qmmm_energy are served that way.

**python_path** - Sets the interpreter of the resident service process, that is the python that can import lammps and matches the MPI family of lammps_path. An empty value uses the interpreter running PyRAI3MD.

**debug** - Writes the cold-path result files even on the resident path, so the two trajectories can be compared file for file. It costs the per-step I/O the resident path exists to avoid and is inert on the subprocess path.

**driver** - Records who owns the geometry, skill or main. The framework sets it and a skill section never writes it; a section read on its own has no framework in it, so the engine integrates and its own constraint keywords apply.

**keep_tmp** - Keeps the calculation directory when 1, and clears it when the backend or the resident service retires when 0. The directory holds the compressed force field, the generated decks, the log and the dumps, and both execution paths read this one key.

**thread** - Sets the OpenMP threads per rank, exported as OMP_NUM_THREADS. lammps_thread is the framework spelling, and the larger value of a written pair is used.

**lammps_thread** - Sets the OpenMP thread count through the framework spelling. It folds into thread, the larger value winning, so a request made in either spelling is honoured.

**nproc** - Sets the number of MPI ranks. Above 1 the run starts with mpirun -np N; lammps_nproc is the framework spelling and the larger of the pair wins.

**lammps_nproc** - Sets the MPI rank count through the framework spelling. It folds into nproc, and in resident mode it is the world of the service process.

**slurm_time** - Sets the wall-clock limit of the submitted job in hh:mm:ss. It is read only when execution_mode is slurm.

**slurm_partition** - Sets the scheduler partition of the submitted job. An empty value writes no partition directive, so the cluster default answers.

**slurm_ntasks** - Sets the number of tasks of the submitted job, which the batch template forwards as the MPI rank count.

**slurm_memory** - Sets the memory per node of the submitted job in MB.

**slurm_command** - Sets the submission command of the batch script, for example sbatch or a site wrapper. An empty value uses the default.

**profile** - Selects the task: ff_energy, ff_gradient, ff_optimize, ff_optimize_cell, ff_nve, ff_nvt, ff_npt, qmmm_field or qmmm_energy. The profile fixes what the generated deck does, and a profile that does not belong to the method is refused.

**method** - Selects the method. Only ff is accepted, because LAMMPS serves the classical force-field layer and never a QM layer.

**lammps_units** - Declares the unit style the user template is written in, real or metal. real reads energies in kcal/mol, distances in Angstrom, time in fs and pressure in atm; metal reads eV, Angstrom, ps and bar. Every quantity the backend returns is labelled with the style in _native_units and the framework scales it, so a wrongly declared style shifts every result instead of raising. The value also decides the time unit of md_timestep and the pressure unit of pressure.

**ewald_accuracy** - Sets the accuracy token written onto every kspace_style line the skill re-issues after read_data, in scientific notation such as 1e-8. It is not a speed dial: the periodic QM/MM field probe takes its potential from the difference of two evaluations, and at 1e-6 that difference was short by 5.3e-4 relative on a 1581-atom box.

**lammps_shake** - Turns the shake fix defined by the input template on (1) or off (0). It is refused for ff_energy and ff_gradient, where the framework propagates and the framework constraint block owns the constraint, and for both QM/MM tasks; the engine-integrated profiles may keep it.

**shake_fix_id** - Names the shake fix the input template defines, fShakeTIP3P by default. The skill renumbers or removes that fix instead of adding a second one.

**shake_keywords** - Holds a raw fix ... shake ... command that is injected when lammps_shake is 1 and the templates define no such fix. An empty value in that situation is an error.

**box_relax** - Selects which lattice degrees of freedom an engine-owned minimization may move. off keeps the box the framework handed over, iso scales the volume, aniso lets each axis relax on its own. The box is written as a fix box/relax line under minimize, so the engine owns the cell for the whole relaxation; that is why a value other than off requires the ff_optimize_cell profile, and why ff_optimize_cell with off is refused. The relaxed cell comes back through fin_cell.

**box_relax_pressure** - Sets the target pressure in bar of the box-relax fix, converted to the pressure unit the run style reads at the point of writing.

**min_style** - Sets the minimization algorithm written as min_style, cg by default. The choice is passed through to the engine, so any style the installed build understands may be named, including sd, quickmin, fire and hftn. It only affects the ff_optimize and ff_optimize_cell profiles.

**min_etol** - Sets the energy tolerance of minimize, in the native energy unit of the run. The default is 1e-4.

**min_ftol** - Sets the force tolerance of minimize, in the native force unit of the run. The default is 1e-8.

**min_maxiter** - Caps the number of minimization iterations, passed as the maxiter argument of minimize. The default is 1000.

**min_maxeval** - Caps the number of force evaluations, passed as the maxeval argument of minimize. The default is 1000000.

**barostat_mode** - Selects how the NPT fix couples to the cell: isotropic scales the volume as one scalar, anisotropic lets the three axes move on their own. The framework integrator reads the same word when it drives an NPT step.

**thermo_damp** - Sets the thermostat damping time of the NVT and NPT fixes, in the native time unit of the run.

**pressure** - Sets the target pressure of the NPT fix, read in the pressure unit of the run style: atm for real and bar for metal.

**pressure_damp** - Sets the barostat damping time of the NPT fix, in the native time unit of the run. It is validated as positive for ff_npt.

**md_nstep** - Sets the number of dynamics steps of the MD profiles, written as run N. The final structure, the final cell and the final velocities are returned, so the value is also the length of the trajectory a run hands back. It is validated as positive and read only by ff_nve, ff_nvt and ff_npt.

**md_timestep** - Sets the timestep of the MD profiles. The unit follows lammps_units: fs for real and ps for metal. The value is validated as positive. Dropping a constraint is not free, because the timestep then has to resolve the constrained motions itself; the bundled TIP3P water box has an O-H period of about 14 fs, so 1 fs keeps roughly fourteen points per period.

**md_seed** - Seeds the initial velocities through velocity all create when the system carries no velocities of its own.

**dump_freq** - Sets how often the trajectory dump and the thermo output are written, in steps. A value coarser than md_nstep leaves a run with its final dumps only. It is validated as positive and read by the MD profiles.

**stress** - Requests the whole-system stress tensor. The input gains a pressure compute with the kinetic term excluded, and its six components are written after run 0 to {title}.stress and read back from there instead of from the eight-digit thermo table. The parser negates the compressive pressure into the tensile-positive convention and normalizes the result into bar, the unit the manifest declares. A step that was asked for a tensor and wrote no file is a failed step, never a zero.

**atom_selection** - Lists the 1-based atom ids of the topology template that this task evaluates, in the order they are handed over. An empty list is the whole system, and an absent key means the same thing. The selection is what lets one multi-center run give every task its own subsystem, so it joins the force-field cache identity: a cache blind to it could answer one task with another task's rows. It is mutually exclusive with qm_atoms.

**qm_atoms** - Lists the QM atoms as 0-based indices in the framework atom order. The partition has to be non-empty, must not cover the whole system, and every index has to be inside it. Both QM/MM profiles read the key, and both refuse a template without a charge column or a bonded row that crosses the partition. It is mutually exclusive with atom_selection.

**qm_probe** - Lists the atoms a qmmm_field probe measures, as a subset of qm_atoms. An empty list measures the whole QM set, which is what every single-center run does. Under several centers the atoms a probe does not measure are environment charges on the same side of the A/B difference as the MM subsystem, so each center has to name itself.

**qm_centers** - Tiles qm_atoms into centers, one group per center, and qmmm_energy uses the tiling to remove the in-cell Q-Q pairs inside each center only, which leaves the inter-center pair in the classical surface where both centers feel its force. Every center has to be non-empty and the groups have to tile qm_atoms. An empty list is one center, the single-center behavior.

**q_qm** - Gives the QM zero-order charges in electron units, in qm_atoms order; the length has to equal the QM set. qmmm_energy writes them into the data file, evaluates the full force field once, and forms the electrostatic term from the Flow-1 arrays. A runtime caller may pass the per-step charges through the calculate call instead, which is what a dynamics run does.

**qmmm_field_file** - Points at a four-column ESPF file, one row per QM atom holding V and the three field components, for a standalone qmmm_energy run. The file has to exist when it is named. A runtime caller never reads it back per step, because the V and E arrays travel in memory.

**charges** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**multiplicity** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**nstate** - Sets the number of electronic states. The force-field layer solves one surface, so the normalization sets the value to 1 and states other than the ground state have no meaning here.

**ci** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**spin** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**coupling** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**verbose** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**solvent** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**temperature** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**shape** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**factor** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**scale** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**pbc** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**cell** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**lattice_vector** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

**kpoint** - Framework keyword; the skill receives it from &MOLECULE and &MD. See [Section 5.2](05-02-system-molecule.html).

LAMMPS is the classical mm layer and never a qm or qm2 layer: the manifest declares the mm role only, so a theory queue cannot place it in a QM slot. A force field is user data. &lt;project&gt;.data is mandatory and a missing topology file is a validation error; &lt;project&gt;.in and the legacy &lt;project&gt;.in.init and &lt;project&gt;.in.settings are optional, and without them the deck is generated from the profile. The skill analyses the templates once into &lt;project&gt;.ff.json and renders every deck from that model, and a template it cannot prove equivalent falls back to the verbatim files with a warning.

QM/MM is two standalone profiles, never a switch inside ff_gradient. qmmm_field probes the MM potential and field at charged probes through an A/B difference and writes &lt;title&gt;.esfp; qmmm_energy evaluates the full force field with the QM charges and returns the reduced classical surface U_main minus U_gas minus the electrostatic term. A v1 limit is that Q-Q non-Coulomb coefficients are not zeroed, because quantum and MM atoms commonly share LAMMPS types; the shipped water fixture simply carries zero epsilon on those rows.

SHAKE belongs to whoever integrates the positions. lammps_shake is refused for ff_energy and ff_gradient, where the framework propagates and the framework constraint block declares the constraint, and for both QM/MM profiles. For the engine-integrated profiles the constraint decision is recorded once in the force-field model, so the deck emits neither fix nor unfix and the legacy mechanism only applies to a template the model refused.

The default unit style is real, with kcal/mol, Angstrom, fs and atm. Results leave the backend in native units with _native_units set and the framework scales them from the manifest; the .esfp file is the atomic-unit exception, and .qmmm, .grad and the auxiliary gradients stay native because they are composite input and audit data. The stress tensor is the potential part only, on a single-point evaluation, in bar: the internal-iteration profiles report no tensor, and restart.data is written for sequential task chaining, never as a framework continuation entry.
