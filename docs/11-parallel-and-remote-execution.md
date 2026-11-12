---
title: "11. Parallel and remote execution"
nav_order: 11
---

This chapter describes how a workflow spreads its work across processes and machines, how a scheduler submission is configured, and how a runtime file makes a finished or interrupted run restartable.

## 11.1. The two dispatch axes

Two orthogonal switches and one pool size decide how a workflow spreads its work. multitask runs the units of one workflow as a spawn process pool on this machine; remotetask submits each unit as its own child runtime, normally with Slurm; multipool is how many workers run at once. The three are written in the workflow section that fans out, never in a calculation section.

At the workflow level all three are tri-state: an absent value leaves the decision to the skill sections and then to the workflow class default. A skill section declares its own value, and planning aggregates the declarations of every skill in one role. All unset means false, true beside unset means true, false beside unset means false, and true beside false is refused as a conflicting configuration. A node-chain declaration outranks both, so a workflow whose units are its own steps stays serial whatever the section says: the adaptive and the single-structure optimize nodes force multitask false, while the md, hessian, path, search, train and prediction nodes take the keyword.

Adaptive sampling runs its stages one after another, so the node itself is pinned serial and each stage carries its own triple. md_multitask, md_remotetask and md_multipool steer the dynamics stage and reference_multitask and reference_multipool the labelling stage; a stage value that is left empty falls back to the generic triple. The reference stage never submits a framework runtime: its remote switch turns each reference skill into its own Slurm client instead, and the training stage uses remote_train. The parent executes one stage at a time, so the stage flags change the width of a stage and never the order of the rounds.

A workflow section cannot carry thread, nproc or a skill-prefixed spelling of them. A workflow parallelises with processes, and threads belong to the calculation, so the keyword is written in the skill section where the engine actually runs; the misplaced key is refused by name and the message names the section that owns it.

| Workflow | What one unit is | Keyword that widens the batch |
| :--- | :--- | :--- |
| md | one trajectory of a multi-trajectory run | &md multitask, &md multipool |
| optimize | one evaluation of the loop, which stays serial | none; the node is pinned |
| path | one image of a band, once per round | &optimize multipool |
| hessian | one displaced geometry of the finite-difference batch | &hessian multitask, &hessian multipool |
| search | one grid point, dispatched as a training child | &search multitask, &search multipool |
| adaptive | one trajectory or one reference frame, per stage | the md_ and reference_ triples |
| train, prediction | one skill call, so the batch has one unit | the flags reach the node but have one unit to spread |

Inside one evaluation the backend terms are a second, finer axis. A multi-trajectory dynamics run and a single-structure optimization pin the theory role serial; the hessian, path, search, train and prediction nodes let the skill-section tri-state through. Per-term remote execution is selected from the system section or from a skill tri-state, never from the workflow section.

## 11.2. Slurm child runtimes

A remote unit becomes one submission. The framework creates runtime/&lt;skill&gt;_runtime_&lt;idx&gt;/ under the working directory, serializes the child state into a .rt file there, writes slurm.sh and submits it with sbatch from that directory, so the child runs with the directory as its working directory.

```text
runtime/
  nn_pes_skill_runtime_3/
    nn_pes_skill_runtime_3.rt     serialized child PyRAIState
    slurm.sh                      #SBATCH lines, shell, exports, launch
    result.json                   written by the child
    slurm-<job>.out, slurm-<job>.err
```

The script holds, in this order, the directives the slurm section renders, the environment shell commands, the exported variables with the package path prepended to PYTHONPATH, and the launch line python3 -m pyrai3md &lt;name&gt;.rt. The parent then polls sacct until the job reaches a terminal state, reads result.json and returns it. A missing or unparsable result is reported with the tail of the child's slurm-\*.err, because that file is the only view the parent has of what went wrong on the node.

The slurm section supplies the allocation: partition, nodes, task and CPU counts, memory, wall time, job name, output and error files, account, quality of service, constraint, mail events, array and gres. [Section 5.12](05-12-environment-and-slurm.html) lists the keys and their defaults, an empty value writes no directive, an unknown key is refused by the script builder, and extra is the only pass-through. gres requests generic resources such as a GPU, which is how a training job asks the scheduler for the device its gpu keyword selects.

Two submission routes share that section and differ in who owns the job. A framework runtime is submitted by the parent: the whole child state travels in the .rt file, and the child writes result.json. A self-submitting skill writes its own script and returns a result; an adaptive reference stage and a training run whose environment selects the slurm execution mode take that route, the training arm submitting one blocking job per ensemble member. Per-term remote workers are the finest route: one backend term becomes a .rtt payload and answers with term_outcome.json.

> **Note.** max_in_flight and environment.scheduler are pmd_env settings, not run input. The first bounds how many submitted jobs a qualification round keeps in the queue, the second names the scheduler that round talks to; neither is read from an input file.

## 11.3. Runtime files and restart

A .rt file is not another input format; it is another source of the runtime state. Launching pyrai3md &lt;title&gt;.rt loads the state and enters the unchanged execution path, so interface parsing, schema validation and planning are bypassed. Every dispatched child leaves its own .rt beside its result.json, which is what makes a child reproducible without its input file.

A .rtt file is the per-term analogue: one backend term, its frozen geometry and electronic request, and one term_outcome.json back. It runs one term and never the whole workflow, so it can neither sum a composite nor advance the parent state.

md.checkpoint names the cadence of the runtime file. Zero, the default, writes none; a value of n writes one every n steps and always on the last step, stamped with the generator state at the moment the geometry and the history were written. The trajectory frame set follows the same arithmetic, so a resumed run continues the file instead of repeating a frame.

A restart preserves the coordinates, velocities, electronic state, history frames, constraint plan, task queue, accumulated results and the random number state, so a resumed trajectory continues the draw the first process was making instead of reseeding. It does not preserve the backends: loggers, file handles, caches and the live skill objects are excluded and rebuilt when the calculator initializes. It also does not re-read the input file, so an edited keyword has no effect on a resumed run; changing a setting is a new input, not a restart. A file whose stored partition fingerprint no longer matches the state it carries is refused rather than resumed.
