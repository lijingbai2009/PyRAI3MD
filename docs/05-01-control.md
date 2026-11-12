---
title: "5.1. CONTROL"
nav_order: 1
parent: "5. Keyword sections"
---

The control section names the run, chooses the workflow, and queues the calculation methods. It is required in every input file. The classic spelling of the section is &amp;CONTROL; in YAML and agent JSON it is the control mapping.

The method queue is written high to low. One method is a single calculation, two methods form an ONIOM2 or QM/MM composite, and three methods form an ONIOM3 composite. [Section 7](07-multiscale-calculations.html) explains the composite rules and [Section 4](04-input-files-and-conventions.html) shows the same job in the three input carriers.

The keywords, default values, and short descriptions are listed below.

**&CONTROL**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| title | pyrai3md | Sets the name of the run. The run log, the trajectory files, the molecular charge file, the runtime file and the frequency files are all named after… |
| jobtype | energy | Sets the type of job. Basic tokens name a property and may be combined in one run: energy, gradient, nac, soc, frequency, absorption, dipole… |
| theory | [] | Sets the electronic-structure method queue, written high to low. Each entry is a skill label such as pyscf, molcas, orca, bagel, openqp, xtb, cp2k… |
| reference | [] | Sets the reference method queue used to label structures during adaptive sampling. It accepts the same labels as theory except nn. At most three… |
| driver | main | Chooses who owns the iteration. main lets the framework run the optimizer or the MD loop and ask the skill for single points; skill, or the name of… |
| multiscale_type | '' | Names the two-entry composite scheme: an empty value keeps the legacy ONIOM2 meaning, qmmm selects additive QM/MM with ESPF embedding and oniom… |
| gl_seed | 1 | Sets the global random number seed. It seeds the initial-condition sampling and, per trajectory, the ensemble draw as gl_seed plus the trajectory… |
| verbose | 2 | Sets the amount of information written to the run log. The levels are cumulative: 0 prints progress and error blocks, 1 adds structures, 2 adds… |
| silent | 1 | Controls the screen output. 1 writes the run log only and is the default; 0 also prints the log blocks to the terminal. The log file is written in… |
| debug | false | Turns on the global test switch. It enables run-health monitoring and early interruption in every module, and it is forwarded to the skill… |
| skill_name | '' | Binds the run to one skill in agent mode. When it is empty and theory is set, the first theory entry supplies it. A name that is not in the skill… |
| skill_path | '' | Records the backend module path of the bound skill. It is written by the certification converter and carried as metadata; dispatch always resolves… |

Full descriptions for all available keywords are summarized below.

**title** - Sets the name of the run. The run log, the trajectory files, the molecular charge file, the runtime file and the frequency files are all named after it, for example &lt;title&gt;.log and &lt;title&gt;.md.xyz. An empty value is reset to pyrai3md.

**jobtype** - Sets the type of job and therefore the workflow. Basic tokens name a property and may be combined in one run: energy, gradient, nac, soc, frequency, absorption, dipole, transition, hessian, num_freq and num_hess. Advanced tokens select a workflow and must be used alone: md (also spelled namd), adaptive, train, prediction, search, optimize, surface_hopping and read. The aliases sp, opt, freq, hess, numfreq, numhess, namd and predict are normalized to their canonical spelling. Mixing basic and advanced tokens, or naming two advanced tokens, is refused. The retired tokens hop and hybrid are refused with a message that names the replacement: surface hopping is md with a surface-hopping algorithm, and a hybrid calculation is a composite run.

**theory** - Sets the electronic-structure method queue, written high to low. Each entry is a skill label: pyscf, molcas, orca, bagel, openqp, xtb, cp2k, lammps, qdm or nn. The package spelling (pyscf_skill) also resolves, and the retired prefixes in qm.pyscf or abinit.orca are stripped once. One entry runs a single calculation. Two entries select an ONIOM2 composite, or additive QM/MM when multiscale_type is qmmm; for QM/MM the first entry provides the QM field and the second owns the classical surface and the cell. Three entries select ONIOM3. Unknown labels are refused at validation.

**reference** - Sets the reference method queue used to label structures during adaptive sampling. It accepts the same labels as theory except nn. At most three layers are allowed, and a run with a layered theory queue accepts one reference layer unless the jobtype is adaptive. Its regions are given by highref and midref. The legacy spelling abinit is read as reference.

**driver** - Chooses who owns the iteration. With main the framework runs the optimizer or the MD loop and asks the skill for single-point energies and gradients; this is the default and the only mode in which a path search, a variable-cell search, or a framework numerical Hessian is available. With skill, or with the label of the first theory skill, the skill runs its own loop and returns the final structure through the fin_coord channel and its convergence through opt_converge; the framework records one evaluation and settles the run from those two fields. A non-main driver requires exactly one theory skill, must name that skill when a name is given, and is refused with a path method or with optimize_cell.

**multiscale_type** - Names the two-entry composite scheme. An empty string keeps the legacy ONIOM2 meaning of a two-method queue. qmmm selects additive QM/MM: the QM story is theory[0], the classical story is theory[1], the QM region comes from highlevel and the embedding field is handed between them. oniom spells the ONIOM2 selection explicitly. The key is only meaningful with two or three theory entries: three entries are always ONIOM3, and qmmm with three entries is refused.

**gl_seed** - Sets the global random number seed. Initial-condition sampling draws from a generator seeded with this value, and each trajectory of a multi-trajectory run is seeded with gl_seed plus its index, so surface hopping and Wigner sampling are reproducible run to run. Zero is a legal explicit value and is distinguished from an absent key. A restarted runtime file carries the generator state instead of re-seeding.

**verbose** - Sets the detail level of the run log and of each result block. Levels are cumulative: 0 prints progress and error blocks only, 1 adds structures, 2 adds the property arrays and is the default, 3 adds the Hessian and an explicit not-computed marker for every absent property. A value of 4 or more additionally dumps the per-step FSSH amplitudes to &lt;title&gt;.fssh.

**silent** - Controls the screen output. 1 writes the run log only and is the default; 0 also prints the log blocks to the terminal. The log file is written in both cases.

**debug** - Turns on the global test switch and is the one control field that reaches into the skill layer: it enables run-health monitoring, gating and early interruption, and it is applied as the lowest layer of the skill-configuration merge, so it outranks a skill's own debug key. It writes extra diagnostics and can stop a run at the first inconsistent record; leave it false for production.

**skill_name** - Binds the run to one skill. It is the agent-mode field: an agent configuration that names a skill here does not need a theory queue. When it is empty and theory is set, the first theory entry supplies the name. If it is set it must be a registered skill label, otherwise validation refuses the input.

**skill_path** - Records the backend module path of the bound skill. It is written by the certification converter and carried with the configuration as metadata. Dispatch resolves the real module path from the skill manifest, so the value documents where a skill lives but does not redirect execution.

The aliases qm and abinit are accepted anywhere theory and reference are accepted; both are stored under the modern names and appear as such in the run log.
