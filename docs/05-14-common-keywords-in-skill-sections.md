---
title: "5.14. Common keywords in skill sections"
nav_order: 14
parent: "5. Keyword sections"
---

Every skill section accepts the same small set of keys. They are documented once here and repeated in each skill table with the same meaning, so a section can be read on its own.

The first group controls execution: where the run happens, what the calculation directory is called, and which profile and method the skill uses. The second group is the framework state the skill receives: charge, multiplicity, the state ladder of ci and spin, the requested coupling pairs, and the periodic cell. The third group is resources: threads, scratch retention and the directives of a skill self-submission.

A key written in a skill section wins over the same key in the home pmd_env.yaml. Framework state written in the skill section is also accepted but is normally written in the molecule section, where the planner can see it before the skill is bound.

**all skill sections**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| execution_mode | local | Chooses how the skill runs: local as a subprocess on this machine, or slurm as a child submitted to the scheduler. |
| project | engine-specific | Sets the file stem of the skill calculation directory; each skill has its own default such as PYSCF or MOLCAS. |
| profile | engine-specific | Selects the computation chain of the skill; the profile tables in [Sections 5.15](05-15-pyscf.html) to [5.24](05-24-nn_pes.html) list every value. |
| method | engine-specific | Selects the electronic-structure method inside the profile; an incompatible method and profile pair is refused. |
| charges | 0 | Total charge of the system, read from the molecule section. |
| multiplicity | 1 | Spin multiplicity of the system, read from the molecule section. |
| ci | [1] | Number of states per spin multiplicity, read from the molecule section. |
| spin | [0] | Total spin S per multiplicity, read from the molecule section. |
| coupling | [] | State pairs that request a coupling, read from the molecule section. |
| pbc | [0, 0, 0] | Periodic flags of the three axes, read from the molecule section. |
| cell | [99, 99, 99, 90, 90, 90] | Cell parameters, read from the molecule section. |
| lattice_vector | zero matrix | Lattice vectors, read from the molecule section. |
| kpoint | [1, 1, 1] | k-point grid, read from the molecule section. |
| thread | 1 | Number of OpenMP threads requested from the engine; a process-wide setting, so the first value written wins. |
| keep_tmp | 0 | Keeps the skill calculation directory after the run when set to 1. |
| slurm_time | 01:00:00 | Wall-clock limit of a skill self-submission. |
| slurm_partition | '' | Partition of a skill self-submission; an empty value writes no directive. |
| slurm_ntasks | 1 | Task count of a skill self-submission. |
| slurm_memory | 4000 | Memory in MB of a skill self-submission. |
| slurm_command | '' | Extra sbatch command appended to a skill self-submission. |

Full descriptions for all available keywords are summarized below.

**execution_mode** - Chooses how the skill runs: local as a subprocess on this machine, or slurm as a child submitted to the scheduler.

**project** - Sets the file stem of the skill calculation directory; each skill has its own default such as PYSCF or MOLCAS.

**profile** - Selects the computation chain of the skill; the profile tables in [Sections 5.15](05-15-pyscf.html) to [5.24](05-24-nn_pes.html) list every value.

**method** - Selects the electronic-structure method inside the profile; an incompatible method and profile pair is refused.

**charges** - Total charge of the system, read from the molecule section.

**multiplicity** - Spin multiplicity of the system, read from the molecule section.

**ci** - Number of states per spin multiplicity, read from the molecule section.

**spin** - Total spin S per multiplicity, read from the molecule section.

**coupling** - State pairs that request a coupling, read from the molecule section.

**pbc** - Periodic flags of the three axes, read from the molecule section.

**cell** - Cell parameters, read from the molecule section.

**lattice_vector** - Lattice vectors, read from the molecule section.

**kpoint** - k-point grid, read from the molecule section.

**thread** - Number of OpenMP threads requested from the engine; a process-wide setting, so the first value written wins.

**keep_tmp** - Keeps the skill calculation directory after the run when set to 1.

**slurm_time** - Wall-clock limit of a skill self-submission.

**slurm_partition** - Partition of a skill self-submission; an empty value writes no directive.

**slurm_ntasks** - Task count of a skill self-submission.

**slurm_memory** - Memory in MB of a skill self-submission.

**slurm_command** - Extra sbatch command appended to a skill self-submission.

The defaults above are the values most skills ship. A skill may override one of them, and its table in the following sections shows the value that skill actually uses; the slurm partition, for example, is engine-specific.
