---
title: "12. Output and analysis"
nav_order: 12
---

This chapter describes what a run reports: the log and its levels, the trajectory and status files a workflow writes, and the tools that read them afterwards.

## 12.1. The run log

The run log is &lt;title&gt;.log in the run directory. A dispatched child restamps its logger into its own unit workspace, so two processes never append to one file. The log opens with the start line and the version banner, then echoes the input as it was read: the control, system and environment sections, the sections the jobtype reads, and the skills that answered.

Every step writes a block in one grammar: a step header, the energies, and then whatever the verbose level allows. The levels are cumulative. Level 0 keeps the progress and error blocks; 1 adds the structures; 2 adds the property arrays and is the default; 3 adds the Hessian and an explicit not-computed marker for every absent property; 4 or more also dumps the per-step surface-hopping amplitudes to &lt;title&gt;.fssh. silent governs the screen rather than the file: at its default the log is the only destination, and zero echoes the same blocks to the terminal.

The last line is PyRAI3MD End: &lt;timestamp&gt; Total: &lt;elapsed&gt; Status: &lt;status&gt;. A reader who only tails the file can therefore tell whether the run succeeded, what it ended with and how long it took.

An optimization writes &optimization setup once, with the method, the optimizer, the step bound and the cell settings, then &optimization progress per step with each criterion beside its threshold and whether it is met, and closes with a banner sentence: converged in n steps, or did not converge after n steps with the reason. The path family writes the same two blocks. Its setup rows name the band size, the spring, the climbing image and the IRC direction, and its progress rows carry the arc, residual and radius of the point or of the round.

A grid search writes &grid search before the fan-out, with the number of points and one row per planned point, and &grid search summary afterwards, with the ranking, a unit row per column and a dash for a channel a point does not carry. An adaptive run adds &data and &nn training per later round, &nn validation gate when the quality gate is armed, &adaptive sampling per round, and a closing &adaptive sampling summary with one trend row per iteration. A training run prints &nn training with &nn validation mean absolute error; a prediction run prints &nn prediction error analysis, headed by the test frame count and the raw comparison files, and one table per statistic after it. Both end with a banner naming the record that came back.

## 12.2. Result and trajectory files

[Section 6.8](06-workflows.html#68-artifacts-and-restart-per-workflow) lists the files each workflow leaves. The dynamics products need a column-by-column reading, because they are the ones a reader parses years later.

Every frame of &lt;title&gt;.md.xyz and &lt;title&gt;.md.velo carries the comment line &lt;title&gt; coord &lt;step&gt; state &lt;k&gt;, extended with to &lt;m&gt; CI on a step that hopped. Geometries are Angstrom and velocities atomic units. A periodic geometry frame also carries Lattice="a b c alpha beta gamma", because the coordinates are the wrapped main image and the cell is the missing fact.

&lt;title&gt;.md.energies holds one row per recorded step: the time as (step - 1) times the time step in atomic units, the potential energy of the active state, the kinetic energy, their sum, and then the energy of every electronic state. All energies are Hartree. The potential, kinetic and total columns describe the state the step ran, so they are not the sum of the state columns.

&lt;title&gt;.md.pop holds the same time column, the population of every state and a zero-or-one vector that marks the active state after the hop. A run that never propagates a density writes no population rows. &lt;title&gt;.md.stress appears only when a backend supplied a non-zero whole-system stress, with nine tensor components and the derived pressure in bar; &lt;title&gt;.md.ensemble appears only for an extended ensemble, with the per-step thermostat and barostat record.

&lt;title&gt;.md.charges carries the managed point charges [q, x, y, z] of each recorded frame and exists only when charge management is bound. A partitioned run adds one set of per-centre files, &lt;title&gt;.md.&lt;n&gt;.\* with a one-based centre index, whose frames hold the atoms of that centre only, while the global files carry the whole system. The &lt;title&gt;.sh.\* family is the same matrix for hopping steps only and is written whenever a hop happened, outside the direct and buffer cadence that thins the md.\* records.

## 12.3. Trajectory and structure tools

Four command line tools read finished products. They write in the directory they are run in and never into the trajectory they read.

| Command | Reads | Writes |
| :--- | :--- | :--- |
| traj_analyzer.py analysis.in | trajectory directories, one keyword per line in the input | &lt;title&gt;.traj.log, and the products of the mode |
| traj_generator.py sampling.in | a frequency or initial-condition file and the program templates | one directory per trajectory, runset-&lt;n&gt;.sh, runall.sh |
| multiscale_generator.py creation.in | solute, solvent and the packing parameters | the combined environment and its initial conditions |
| analyze_npt_frames.py --root &lt;tree&gt; | md.xyz cell comments and md.ensemble columns | nothing; the report goes to the terminal |

tools/traj_analyzer.py walks one stage at a time. Mode 1 counts rows per trajectory and writes the complete list; mode 2 measures energy drift and writes the conserved list with a per-step drift file; mode 3 extracts frames to JSON and snapshot files; mode 4 averages populations and writes the hop energy gaps; mode 5 classifies snapshots by geometrical parameters; mode 6 writes plot data; mode 7 fetches single trajectories; mode 8 reads initial, final and hopping snapshots. The reader for this program is the default, so the legacy programs are reached by naming them.

tools/traj_generator.py samples initial conditions and builds one calculation directory per sample, each with its own script, for a list of programs including this one; the number of trajectories is the node count times the job count, and an index continues an existing set instead of overwriting it. tools/multiscale_generator.py builds a solvent or aggregate environment, prepares the classical files and merges initial conditions, in one of seven modes from create to mm. tools/analyze_npt_frames.py recomputes the cell history of a variable cell run from the frame comments and the ensemble columns and reports the volume excursion, the accumulated rotation of the simulation frame, the strain extremes, the pressure components and the temperature; it writes nothing.

```bash
python tools/traj_analyzer.py analysis.in
python tools/traj_generator.py sampling.in
python tools/multiscale_generator.py creation.in
python3 tools/analyze_npt_frames.py --root run_tree --detail
```

A trajectory analysis is usually a chain: mode 1 keeps the complete trajectories, mode 2 removes the ones that lost energy, and the later modes read the resulting lists instead of the whole set, so every average describes the same set of trajectories.
