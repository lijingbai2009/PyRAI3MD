---
title: "C. Keyword status index"
nav_order: 3
parent: "Appendices"
---

This appendix is generated from the same keyword specifications as the tables in [Section 5](05-keyword-sections.html), so a mark in a table and a row here come from one declaration. Three questions are answered: which keywords are accepted but have no effect in this version, which accept several values but honour one, and which names are aliases stored under a modern name. The marks used in the keyword tables are the same three: a dagger, a double dagger and a section sign.

### Keywords with no effect

A keyword in this table parses, validates and appears in the run log under its own name, and no workflow reads it. It is kept so that an older input still loads; a new input should not use one.

| Keyword | Section | Note |
| :--- | :--- | :--- |
| qmmm_xyz | &SYSTEM | accepted but not read |
| solvent | &SYSTEM | accepted but not read; implicit solvation is set by the solvent key of the engine section |
| method | &DYNAMICS | accepted but not read |
| pop_step | &ADAPTIVE | accepted but not read |
| maxdiscard | &ADAPTIVE | accepted but not read |
| refine_num | &ADAPTIVE | accepted but not read |
| refine_start | &ADAPTIVE | accepted but not read |
| refine_end | &ADAPTIVE | accepted but not read |
| refine_gap | &ADAPTIVE | accepted but not read |
| dynenergy | &ADAPTIVE | accepted but not read |
| inienergy | &ADAPTIVE | accepted but not read |
| fwdenergy | &ADAPTIVE | accepted but not read |
| bckenergy | &ADAPTIVE | accepted but not read |
| dyngrad | &ADAPTIVE | accepted but not read |
| inigrad | &ADAPTIVE | accepted but not read |
| fwdgrad | &ADAPTIVE | accepted but not read |
| bckgrad | &ADAPTIVE | accepted but not read |
| dynnac | &ADAPTIVE | accepted but not read |
| ininac | &ADAPTIVE | accepted but not read |
| fwdnac | &ADAPTIVE | accepted but not read |
| bcknac | &ADAPTIVE | accepted but not read |
| dynsoc | &ADAPTIVE | accepted but not read |
| inisoc | &ADAPTIVE | accepted but not read |
| fwdsoc | &ADAPTIVE | accepted but not read |
| bcksoc | &ADAPTIVE | accepted but not read |
| program | &ENVIRONMENT | accepted but not read |
| version | &ENVIRONMENT | accepted but not read |
| device | &ENVIRONMENT | accepted but not read |
| program_path | &ENVIRONMENT | accepted but not read |
| slurm_time | &ENVIRONMENT | accepted but not read |
| slurm_partition | &ENVIRONMENT | accepted but not read |
| slurm_ntasks | &ENVIRONMENT | accepted but not read |
| num_gpu | &ENVIRONMENT | accepted but not read |
| slurm_memory | &ENVIRONMENT | accepted but not read |
| slurm_command | &ENVIRONMENT | accepted but not read |
| nics_ring | &PYSCF | accepted but not read |
| nics_scan | &PYSCF | accepted but not read |
| thermostat | &PYSCF | accepted but not read |
| barostat | &PYSCF | accepted but not read |
| ensemble | &PYSCF | accepted but not read |
| nics_ring | &MOLCAS | accepted but not read |
| nics_scan | &MOLCAS | accepted but not read |
| ir_width | &BAGEL | accepted but not read |
| ir_scale | &BAGEL | accepted but not read |
| uv_width | &BAGEL | accepted but not read |
| nics_ring | &BAGEL | accepted but not read |
| nics_scan | &BAGEL | accepted but not read |
| solvent | &ORCA | accepted but not read |
| temperature | &ORCA | accepted but not read |
| shape | &ORCA | accepted but not read |
| factor | &ORCA | accepted but not read |
| scale | &ORCA | accepted but not read |
| ir_width | &ORCA | accepted but not read |
| ir_scale | &ORCA | accepted but not read |
| uv_width | &ORCA | accepted but not read |
| nics_ring | &ORCA | accepted but not read |
| nics_scan | &ORCA | accepted but not read |
| solvent | &OPENQP | accepted but not read |
| temperature | &OPENQP | accepted but not read |
| ir_width | &OPENQP | accepted but not read |
| ir_scale | &OPENQP | accepted but not read |
| uv_width | &OPENQP | accepted but not read |
| nics_ring | &OPENQP | accepted but not read |
| nics_scan | &OPENQP | accepted but not read |
| solvent | &XTB | accepted but not read |
| temperature | &XTB | accepted but not read |
| shape | &XTB | accepted but not read |
| factor | &XTB | accepted but not read |
| scale | &XTB | accepted but not read |
| ir_width | &CP2K | accepted but not read |
| ir_scale | &CP2K | accepted but not read |
| uv_width | &CP2K | accepted but not read |
| nics_ring | &CP2K | accepted but not read |
| nics_scan | &CP2K | accepted but not read |

### Keywords that honour one value

A keyword in this table accepts a list or a compound spelling but reads one member of it in v3.0.0. The table names the keyword, not the value: the description in Chapter 5 says which member survives.

| Keyword | Section | Note |
| :--- | :--- | :--- |
| load | &ADAPTIVE | several values accepted, one honoured |
| sample_energy | &ADAPTIVE | several values accepted, one honoured |
| sample_grad | &ADAPTIVE | several values accepted, one honoured |
| sample_dipole | &ADAPTIVE | several values accepted, one honoured |
| sample_transition_dipole | &ADAPTIVE | several values accepted, one honoured |
| sample_oscillator_strength | &ADAPTIVE | several values accepted, one honoured |
| dynsample | &ADAPTIVE | several values accepted, one honoured |
| refine | &ADAPTIVE | several values accepted, one honoured |
| transfer | &ADAPTIVE | several values accepted, one honoured |

### Aliases

An alias is read and stored under its modern name, and the run log prints the modern name. The table below joins the aliases the specifications declare with the legacy spellings the package accepts at the control and dynamics layer.

| Alias | Stored as | Scope |
| :--- | :--- | :--- |
| qm | theory | &CONTROL - legacy keyword spelling |
| abinit | reference | &CONTROL - legacy keyword spelling |
| sp | energy | jobtype - canonical jobtype token |
| opt | optimize | jobtype - canonical jobtype token |
| freq, frequence | frequency | jobtype - canonical jobtype token |
| hess | hessian | jobtype - canonical jobtype token |
| numfreq | num_freq | jobtype - canonical jobtype token |
| numhess | num_hess | jobtype - canonical jobtype token |
| namd | md | jobtype - canonical jobtype token |
| predict | prediction | jobtype - canonical jobtype token |
| nh, nose | nose-hoover | &MD - thermostat spelling |
| nhc, nose-hoover-chains | nose-hoover-chain | &MD - thermostat spelling |
| csnr | csvr | &MD - thermostat spelling |
| pr, parrinello | parrinello-rahman | &MD - barostat spelling |
| parrinello-rahman-npt | parrinello-rahman | &MD - barostat spelling |
| c-rescale | scr | &MD - barostat spelling |
| stochastic-cell-rescaling | scr | &MD - barostat spelling |
| MD | DYNAMICS | section name - the section is read under either name |
| MOLECULE | SYSTEM | section name - MOLECULE remains the historical spelling |

The three tables are built from the same keyword specifications as the tables in Chapter 5, so a mark in a keyword table and a row here come from one declaration. The index covers every framework section and every skill section. A keyword that appears in none of the three tables is either an active input or a retired one; retired keywords are refused rather than accepted, and [Appendix E](appendix-e-migrating-pyrai2md-inputs.html) lists their replacements.

> **Note.** A retired keyword is deliberately absent from all three tables. It is refused at validation with a message that names its replacement, and [Appendix E](appendix-e-migrating-pyrai2md-inputs.html) collects those replacements. A keyword that is merely out of date is an alias; a keyword whose meaning no longer exists is a refusal.
