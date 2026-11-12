---
title: "G. Troubleshooting"
nav_order: 7
parent: "Appendices"
---

This appendix pairs the failures a new user meets with the cause and the action. The messages are quoted from validation and from the run log; the sections named in the action column hold the full rule.

| Symptom | Likely cause | What to do |
| :--- | :--- | :--- |
| pyrai3md: command not found | the package is not installed in the active environment | run pip install . in the checkout, or activate the environment that has it |
| the input stops at "missing atoms or coordinates" | no atoms array, no coord path and no same-name .xyz file | add top-level atoms and coordinates, or point coord at a file, or place &lt;input stem&gt;.xyz next to the input |
| the input stops at "unknown jobtype" | a retired or unsupported token such as hop, hybrid or scan | use md with a surface-hopping algorithm, a composite method queue, or an optimize path method; see [Section 6.1](06-workflows.html#61-job-types-and-workflow-resolution) |
| "theory label is not registered" | a misspelt method label, or a skill whose directory is absent | check the label spelling, then list the installed skills with the skill query module |
| "capability is not available from the selected skill" | the jobtype asks for a property the engine does not declare | consult [Section 2.2](02-features.html#22-calculation-engines) and the skill limitations, or choose another engine |
| "executable not found" or the engine starts and exits at once | the engine path is empty or wrong, or the module environment is not loaded | set the path in ~/.pyrai3md/pmd_env.yaml under the skill section, load the site module the engine needs, and rerun |
| a periodic request is refused | the chosen engine does not declare periodic support | use cp2k, xtb with GFN-FF, or lammps; [Section 8.4](08-periodic-systems.html#84-which-engines-support-pbc) lists the support |
| "partial periodicity is not supported" | pbc was given one or two flags | give three flags, all zero or all one, and a real cell |
| npt is refused | the barostat and thermostat pair is not allowed, or one of them is off | use the pairing table in [Section 6.5](06-workflows.html#65-ab-initio-md-and-namd), for example Berendsen with Nose-Hoover |
| the Hessian request is refused at validation | hess_driver main combined with analytic or auto, or a variable cell with a vibrational analysis | use numerical for main, skill for a backend matrix, and drop the cell relaxation while analysing vibrations |
| a path method is refused | driver is not main, or keywords of another path method are present | set driver main and keep only the keywords of the chosen method; the groups are listed in [Section 6.4](06-workflows.html#64-reaction-paths-mep-irc-neb) |
| adaptive stops at startup | load is not 1, transfer is not 0, or refine or dynsample is non-zero | remove those keys; this version does not implement those branches |
| read keys are rejected | a read section is present without jobtype read | set jobtype read, or remove the read section |
| the table of contents is empty or its page numbers are stale | the contents is a Word field and has not been updated | open the file in Word and update the field, or accept the prompt on open |
| no log file appears | the run wrote into another directory, or the scratch path changed | look for &lt;title&gt;.log beside the scratch root printed by --info |
| a Slurm child fails immediately with an invalid partition | the partition in pmd_env.yaml does not exist on the host | set partition to names the scheduler accepts; an empty value writes no directive |
| a trajectory ends before the requested number of steps | a guard fired: population, energy, gradient, coupling or tracked geometry | read the stop code in the log ([Section 6.5](06-workflows.html#65-ab-initio-md-and-namd)) and widen the corresponding md_max\* or track threshold |
| the engine runs out of memory | too many ranks or threads per node, or too small a memory directive | lower thread or nproc, raise mem in the slurm section, or reduce the system |
| a model cannot be found | modeldir does not contain the profile or the trained folder | set modeldir and profile in the nn_pes section and check the model folder layout in [Section 10.2](10-machine-learning-potentials.html#102-training) |
| training does not use the GPU | gpu is -1, or the submission has no gres | set the GPU index and add gres to the slurm section |
| two runs of the same input differ slightly | different thread counts, or a seed that was not fixed | fix gl_seed, keep the thread count equal, and compare on the same build |
| a shipped example fails on this machine | the example binds an engine that is not installed here | run it with --info first to see the resolved plan, then switch the theory queue to an installed engine |

When a message is not listed here, read it literally: validation names the offending key and often the replacement, and the run log ends with a status line that separates a refusal, a failed calculation and a guard stop. [Appendix D](appendix-d-refusal-and-limitation-matrix.html) collects the refusal matrix.
