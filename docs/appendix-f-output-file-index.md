---
title: "F. Output file index"
nav_order: 6
parent: "Appendices"
---

Every file the framework writes is named after the run title, except the three fixed names a child runtime uses and the scheduler logs, which carry the job identifier. The table below is alphabetical by the name as it appears on disk; a title in angle brackets stands for the control title, so a run titled water writes water.md.xyz. [Section 12.2](12-output-and-analysis.html#122-result-and-trajectory-files) describes the same set by producer, and [Section 12.1](12-output-and-analysis.html#121-the-run-log) describes the log blocks that carry the same numbers.

| File | Writer | Contents |
| :--- | :--- | :--- |
| &lt;title&gt;.adaptive/ | adaptive workflow | The run directory of an adaptive run: the round record, the run log, and one iter-&lt;k&gt; directory per round with the data snapshot, the model, the reference labels, the MD folders and the round status. |
| &lt;title&gt;.charge | hand-written or certification converter | Background charges or an external field: a header naming the kind, the unit and the row role, then q x y z rows. Read by a run that sets read_charge. |
| &lt;title&gt;.freq.molden | hessian workflow | Frequencies in cm-1 and the mass-weighted modes in the Molcas layout. The initial-condition sampler looks for this exact name. |
| &lt;title&gt;.fssh | hopping integrator | Per-step FSSH amplitudes, written only at verbose 4 or above. It is a diagnosis record and grows with every electronic step. |
| &lt;title&gt;.init | sampling and adaptive warm-up | The sampled batch in the legacy layout. It is the same batch the sampler hands to the trajectories. |
| &lt;title&gt;.init.xyz | sampling and adaptive warm-up | The multi-frame starting conditions with velocities, one frame per trajectory. Read by initcond read, and it is the file to edit when a run must start from chosen structures. |
| &lt;title&gt;.json | tools/traj_analyzer.py | The arrays of a trajectory extracted for later analysis, written when save_traj is on. Modes 4 to 7 read it back instead of re-parsing the trajectory. |
| &lt;title&gt;.log | run logger | The run log: setup, one result block per evaluation under a step header, warnings, and the conclusion. Every other file carries a subset of these numbers. |
| &lt;title&gt;.md.energies | md workflow writer | One row per recorded step: step, active-state potential energy, kinetic energy and the per-state energies. |
| &lt;title&gt;.md.ensemble | md workflow writer | One row per recorded step of a thermostatted or barostatted run: temperatures, kinetic energy, volume, chain and barostat state, and the pressures in bar. An NVE run writes no such file. |
| &lt;title&gt;.md.pop | md workflow writer | State populations followed by a one-hot vector marking the post-hop active state. The two blocks together are how a hop is read from disk. |
| &lt;title&gt;.md.stress | md workflow writer | Step, the nine components of the active-state stress tensor in Hartree/Bohr^3, and the derived pressure in bar. Written only when the engine supplied a stress tensor. |
| &lt;title&gt;.md.velo | md workflow writer | Velocities of every recorded step in Bohr per atomic time, in the same atom order as the geometry. |
| &lt;title&gt;.md.xyz | md workflow writer | Structure of every recorded step in Angstrom. A periodic frame states its cell in the comment as a Lattice key, and a frame on which the active state changed carries a to &lt;n&gt; CI suffix. |
| &lt;title&gt;.rt | md checkpoint | The serialized runtime state: geometry, velocities, history, electronic density, step count and generator position. Launching it continues the trajectory; the md checkpoint keyword sets the cadence, and no file is written when it is zero. |
| &lt;title&gt;.sh.energies | md workflow writer | The energies record at hopping steps only, with the same columns as the md file. |
| &lt;title&gt;.sh.pop | md workflow writer | The population record at hopping steps only, carrying the post-hop active state. |
| &lt;title&gt;.sh.velo | md workflow writer | Post-hop velocities at hopping steps, which is where the rescaling or reflection across the gap is visible. |
| &lt;title&gt;.sh.xyz | md workflow writer | Structure at hopping steps only. The sh files are written immediately on a hop, independent of the direct and buffer cadence that governs the md files. |
| md.charges, sh.charges | md workflow writer | The managed-region point charges q x y z of each recorded frame and each hopping frame. Written only when charge management is active; a partitioned run also writes md.&lt;n&gt;.charges per center. |
| opt.xyz | optimize workflow | The latest accepted structure, overwritten every step. It is the file to copy into the next input when a run is chained. |
| opt_geom.xyz | optimize workflow | Every accepted frame appended in order, which is the trajectory of the search rather than its latest state. |
| opt_status.txt | optimize workflow | The convergence table: a fixed header, then one row per accepted step with step, energy, shift, max_d, rms_d, max_g and rms_g. A cell-relaxing run adds the cell columns after the fixed ones. |
| path.xyz | path workflow | The latest frame of a path search, rewritten as the walk advances. |
| path_geom.xyz | path workflow | Every accepted frame of the walk or every image of the band, which is the path itself. |
| path_profile.txt | path workflow | One row per accepted point or per image, carrying the profile the search reports at the end of the run. |
| path_status.txt | path workflow | One row per evaluation for a walker, or per round for a band. It is deliberately not opt_status.txt, because its columns answer different questions. |
| path_forward.xyz, path_backward.xyz | path workflow | The two sides of an IRC walk when both directions are requested. |
| pyrai3md_step.perf | step profiler | Phase timings of the md step when perf_profile is on. It is a performance record, not a trajectory record. |
| result.json | child runtime | The result payload a dispatched runtime returns to its parent: energies, gradients and the metadata the parent needs to continue. Its absence is how a failed child is detected. |
| slurm-%j.err, slurm-%j.out | scheduler | Standard error and standard output of a submitted child, named with the job identifier unless the slurm section sets explicit names. |
| summary.csv | grid search | The ranked table of a hyperparameter scan, one row per grid point with the ranking statistics of its ensemble. |
| term_outcome.json | dispatch term worker | The unified outcome of one remote term: the result the worker produced plus its status. The parent reads it instead of parsing the worker log. |

Three naming rules explain most of the surprises. A record that is rewritten every step has no step number in its name, so opt.xyz and path.xyz always describe the newest state. A record that is appended carries the step in its first column rather than in its name, so the file stays one table. And a record that is conditional is absent rather than empty: no stress tensor means no md.stress, no extended ensemble means no md.ensemble, and no checkpoint keyword means no runtime file at all.

A run directory that is reused can therefore contain a mixture of records from two runs. The framework truncates the files it owns at startup, but it does not delete a file a previous run wrote under a different title, and it does not delete the adaptive directory of an earlier run with the same title. Check the title and the modification times before reading a record whose step numbers do not start where the log says they should.
