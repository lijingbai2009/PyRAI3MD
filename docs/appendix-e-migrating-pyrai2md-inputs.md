---
title: "E. Migrating PyRAI2MD inputs"
nav_order: 5
parent: "Appendices"
---

PyRAI2MD is the predecessor of PyRAI3MD, and most of its input vocabulary survives in 3.0.0 under a modern spelling. This appendix maps the sections and keywords that moved, so that an old input can be brought forward deliberately instead of by trial and error.

The molecular section is the part that changed least. The atomic symbols, coordinates, velocities, charges, multiplicity, ci, spin, coupling, highlevel and midlevel keep their names, their meaning and their index bases; only the section name moved, and the old name is still read. What changed around it is the method queue: qm and abinit became theory and reference, and the labels inside the queue are now skill names.

| 2.x spelling | 3.0.0 spelling | What changed |
| :--- | :--- | :--- |
| &MOLECULE | &SYSTEM | SYSTEM is the new name; MOLECULE is still accepted and is the spelling most examples use. The fields inside keep their names and 1-based atom numbers. |
| &MD | &DYNAMICS | The section is read under either name. In YAML and agent JSON it is the md mapping; the dynamics spelling is the long form. |
| qm | theory | The method queue, written high to low. Each entry is a skill label; qm.pyscf and abinit.orca are stripped once and read as pyscf and orca. |
| abinit | reference | The labelling queue of adaptive sampling. It accepts the same labels as theory except nn. |
| jobtype train, adaptive, md | unchanged | The three workflow tokens keep their spelling and their meaning. |
| jobtype hop | refused | Surface hopping is md with sfhp set to fssh, gsh or lzsh and the nactype keyword beside it. There is no separate hop jobtype. |
| jobtype scan | refused | There is no scan jobtype and no scan section in 3.0.0; the Scan class is a registered placeholder. A reaction path is written as an optimize run with the path family, and a hyperparameter scan is the search jobtype. |
| NN, MLP, E2N2, SCHNET sections | nn_pes section | One skill section replaces the per-architecture sections. profile selects the computation chain and method selects the architecture inside it. |
| EG, NAC, SOC hyperparameter sections | nn_pes role blocks | The eg, nac and soc blocks of the nn_pes section carry the architecture and training settings for the energy-and-gradient, coupling, and spin-orbit roles. |
| MOLCAS, ORCA, BAGEL, XTB sections | skill sections | Each keeps its own name as a skill section and gains execution_mode and profile. The section is read only when the skill is in the theory or reference queue. |
| use_hpc | multitask and remotetask | use_hpc 0 is multitask without remotetask; use_hpc 1 is both. The two switches answer the two questions separately, which is what lets a run fan out on one host without a scheduler. |
| ml_ncpu | multipool | The pool size of the workflow section that fans out. It is a per-section keyword now, so the md fan-out and the reference fan-out can differ. |
| control ml_ncpu, control qc_ncpu | removed | Neither is a control keyword. Parallelism is declared by the section that owns the fan-out, as multipool, and a control keyword is no longer read for it. |
| initcond 1 | initcond wigner | The numerical spelling of the sampling modes is gone. initcond names the source: none, read, wigner or boltzmann. |
| &FILE and FILE extraction | jobtype read | The read section names the path, the profile and the labels to extract; a dataset built from finished artefacts is then written by tools/training_data.py. |
| restraint keys in &MOLECULE | the constraint block | The flat freeze, cbond, cavity and shape keys are refused with the block keyword that replaces each one. In YAML the block is molecule.constraints; in classic input it is &CONSTRAINT. |

Three habits carry over unchanged and are worth keeping. A title is still the stem of every product, so two runs in one directory still overwrite each other. The state ladder is still written as ci and spin, with coupling naming the pairs, and the pair maps are still derived rather than written. The highlevel and midlevel regions are still 1-based atom lists.

Two habits have to change. A method queue is now a skill queue, so a label that was a program name in 2.x is a skill name in 3.0.0, and an unknown label is refused at validation rather than at the first calculation. A workflow token is now the only thing that selects a workflow: composition of basic property tokens is allowed, and mixing a property token with a workflow token is refused, which catches an input that expected the old precedence rules.

The keyword status marks in [Section 5](05-keyword-sections.html) are the quickest way to tell a moved keyword from a retired one. A dagger means the keyword is accepted and read by nothing, a section sign means it is an alias, and a refusal means the replacement is named in the error. Section C lists the aliases, and Section D lists the refusals.
