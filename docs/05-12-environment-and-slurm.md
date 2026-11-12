---
title: "5.12. ENVIRONMENT and SLURM"
nav_order: 12
parent: "5. Keyword sections"
---

The environment section records the execution metadata that travels with the state: the shell lines, the free environment block, the scratch root and the scheduler fields. In v3.0.0 a calculation reads shell, environment and scratch_path; the remaining keys are carried so that a serialized runtime still describes the machine it came from, and the table marks them accordingly. The directives that a submission actually uses come from the slurm section, whose keys follow.

The environment section describes the machine a run starts on and the allocation a submitted child asks for. The engine paths, the shell setup and the scheduler settings normally live in the user file ~/.pyrai3md/pmd_env.yaml, which is created on first use and read in every session. This section overrides those values for one run, and the merge order is the package defaults, then the user file, then this input.

The overrides belong to the run that declares them: they are applied before the workflows are planned and rolled back before the next input is built in the same process, so an input can only override a setting it declares. The values travel with the run state, which is what lets the dispatch subsystem read one source instead of the process environment.

Only four of the keys below reach the framework: shell, environment, slurm and scratch_path. The remaining scalars are the historical spellings of settings that moved into the skill sections and the slurm mapping, and they are accepted so that older input files still parse.

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&ENVIRONMENT**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| shell | '' | Sets raw shell commands, one per line, that are copied verbatim into the submission scripts of the main framework dispatch. The classic carrier… |
| environment | {} | Holds the exported environment variables as a name to value mapping, for example OMP_NUM_THREADS. The values are written under the export key of this… |
| program † | '' | Names the program label of a legacy input. It is parsed for compatibility only: the engine of a calculation is the skill named in the theory queue… |
| version † | '' | Names the program version of a legacy input. It is parsed for compatibility only: a skill resolves its own executable. The default is empty. |
| device † | cpu | Names a device selector of a legacy input. It is parsed for compatibility only: a device is chosen in the skill section that runs the calculation… |
| program_path † | '' | Names an executable path of a legacy input. It is parsed for compatibility only: each skill has its own path keyword. The default is empty. |
| scratch_path | '' | Sets the directory the run works in. The framework opens &lt;title&gt;.log here, and a skill without a scratch directory of its own creates its calculation… |
| slurm_time † | 01:00:00 | Sets a wall time in the legacy scalar spelling. The submission script reads the time directive of the slurm mapping of this section, so this key… |
| slurm_partition † | default | Sets a queue name in the legacy scalar spelling. The submission script reads the partition directive of the slurm mapping. The default is default. |
| slurm_ntasks † | 1 | Sets a task count in the legacy scalar spelling. The submission script reads ntasks from the slurm mapping. The default is 1. |
| num_gpu † | 0 | Sets a GPU count in the legacy scalar spelling. It is parsed for compatibility only: a GPU is asked for through the gres directive of the slurm… |
| slurm_memory † | 4000 | Sets a memory request in the legacy scalar spelling. The submission script reads mem from the slurm mapping. The default is 4000. |
| slurm_command † | '' | Sets a submission command in the legacy scalar spelling. The framework submits through its own Slurm launcher, which builds the command itself. The… |
| slurm | {} | Merges Slurm directives into the allocation a submitted child asks for, as a mapping of directive names to values, for example partition, time, mem… |

Full descriptions for all available keywords are summarized below.

**shell** - Sets raw shell commands, one per line, that are copied verbatim into the submission scripts the main framework dispatch builds, which is where a module load or a conda activation goes. The classic carrier writes them on one line separated by semicolons. The key serves the main framework scripts only: a skill consumes its own shell keyword, so a command written here does not reach a skill subprocess. The default is empty.

**environment** - Holds the exported environment variables as a name to value mapping. The values are written under the export key of this section or of a pmd_env.yaml file, and the merged mapping is flattened into this field with one value per name. They are exported in the batch script of a remote child and set in the subprocess environment of a local one, which is how a run pins for example OMP_NUM_THREADS or MPI_NP. The default is empty.

**program** - Names the program label of a legacy input. It is parsed for compatibility only: the engine of a calculation is the skill named in the theory queue. The default is empty. This keyword is accepted but has no effect in v3.0.0.

**version** - Names the program version of a legacy input. It is parsed for compatibility only: a skill resolves its own executable. The default is empty. This keyword is accepted but has no effect in v3.0.0.

**device** - Names a device selector of a legacy input. It is parsed for compatibility only: a device is chosen in the skill section that runs the calculation. The default is cpu. This keyword is accepted but has no effect in v3.0.0.

**program_path** - Names an executable path of a legacy input. It is parsed for compatibility only: each skill has its own path keyword. The default is empty. This keyword is accepted but has no effect in v3.0.0.

**scratch_path** - Sets the directory the run works in. The framework opens &lt;title&gt;.log here, and a skill without a scratch directory of its own creates its calculation directories under it, so one keyword moves the whole run off a fast local disk or onto a shared one. A relative value is resolved against the launch directory, and an empty value keeps the historical behaviour of writing beside the launch directory.

**slurm_time** - Sets a wall time in the legacy scalar spelling. The submission script reads the time directive of the slurm mapping of this section, so this key changes no script. The default is 01:00:00. This keyword is accepted but has no effect in v3.0.0.

**slurm_partition** - Sets a queue name in the legacy scalar spelling. The submission script reads the partition directive of the slurm mapping. The default is default. This keyword is accepted but has no effect in v3.0.0.

**slurm_ntasks** - Sets a task count in the legacy scalar spelling. The submission script reads ntasks from the slurm mapping. The default is 1. This keyword is accepted but has no effect in v3.0.0.

**num_gpu** - Sets a GPU count in the legacy scalar spelling. It is parsed for compatibility only: a GPU is asked for through the gres directive of the slurm mapping. The default is 0. This keyword is accepted but has no effect in v3.0.0.

**slurm_memory** - Sets a memory request in the legacy scalar spelling. The submission script reads mem from the slurm mapping. The default is 4000. This keyword is accepted but has no effect in v3.0.0.

**slurm_command** - Sets a submission command in the legacy scalar spelling. The framework submits through its own Slurm launcher, which builds the command itself. The default is empty. This keyword is accepted but has no effect in v3.0.0.

**slurm** - Merges Slurm directives into the allocation a submitted child asks for, as a mapping of directive names to values, for example partition, nodes, ntasks, cpus_per_task, mem, time, job_name, output, error, account, qos, exclusive, constraint, gres, mail_type, mail_user, array and extra. An empty value writes no directive, exclusive false writes none either, and extra is copied verbatim as the only pass-through. Unknown names are refused by the script builder rather than silently dropped. The directives are read when remotetask is true, and the merge order is the package file, then the user file, then this section.

The retired keys of this section, execution_mode, workflow_execution, workflow_workers and num_thread, are refused at validation with a message that names where each setting belongs now. Thread and rank counts are properties of a calculation, so they are written in the skill section as &lt;skill&gt;_thread and &lt;skill&gt;_nproc, and a workflow section refuses them by name.

The slurm section supplies the directives of the submission scripts the framework writes for child runtimes and for skill self-submission. The same keys can be written in three places: the slurm block of the home pmd_env.yaml, a &SLURM section in the classic input, and the slurm mapping of YAML or agent input. The merge is per key and the input wins.

An empty value writes no directive. Keys that are not listed here are refused by the script builder, so a misspelt option is reported instead of being ignored; the extra key is the only free pass-through and is copied verbatim into the script.

The keywords, default values, and short descriptions are listed below.

**&SLURM**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| partition | long,short,express | Sets the scheduler partition or comma-separated list of partitions for the child jobs. |
| nodes | 1 | Sets the number of nodes requested by a child job. |
| ntasks | 10 | Sets the number of tasks per node; an engine that runs with MPI uses this value as its rank count. |
| cpus_per_task | 1 | Sets the number of CPUs per task requested from the scheduler. |
| mem | 16G | Sets the memory per node requested by the child job, for example 16G. |
| mem_per_cpu | '' | Sets the memory per CPU instead of the memory per node; only one of mem and mem_per_cpu should be non-empty. |
| time | 1:00:00 | Sets the wall-clock limit of the child job in hh:mm:ss. |
| job_name | pyrai3md | Sets the job name written as the --job-name directive. |
| output | slurm-%j.out | Sets the standard-output file of the child job; the default uses the scheduler job identifier. |
| error | slurm-%j.err | Sets the standard-error file of the child job; the default uses the scheduler job identifier. |
| account | '' | Sets the accounting project passed as --account. |
| qos | '' | Sets the quality-of-service string passed as --qos. |
| exclusive | false | Requests exclusive node access when true. |
| constraint | '' | Sets the scheduler node constraint string passed as --constraint. |
| gres | '' | Requests generic resources such as a GPU with --gres; it is also the trigger for the GPU submission template of the training skill. |
| mail_type | '' | Sets the --mail-type events for scheduler notifications. |
| mail_user | '' | Sets the --mail-user address for scheduler notifications. |
| array | '' | Sets a scheduler array specification; a manual run normally leaves it empty because the framework submits one job per unit. |
| extra | '' | Copies a free-form string into the script verbatim; it is the only pass-through directive and must be valid script syntax. |

Full descriptions for all available keywords are summarized below.

**partition** - Sets the scheduler partition or comma-separated list of partitions for every child job the run submits. A list lets the scheduler place each job where capacity exists. An empty value writes no --partition directive.

**nodes** - Sets the number of nodes requested by a child job.

**ntasks** - Sets the number of tasks per node. An engine that runs with MPI uses this value as its rank count, and the framework also uses it to size a grouped submission. It does not change the number of concurrent jobs, which is a dispatch setting.

**cpus_per_task** - Sets the number of CPUs per task requested from the scheduler.

**mem** - Sets the memory per node requested for a child job, for example 16G. It is written as --mem; mem_per_cpu is the alternative spelling and only one of the two should carry a value.

**mem_per_cpu** - Sets the memory per CPU instead of the memory per node; only one of mem and mem_per_cpu should be non-empty.

**time** - Sets the wall-clock limit of the child job in hh:mm:ss.

**job_name** - Sets the job name written as the --job-name directive.

**output** - Sets the standard-output file of the child job; the default uses the scheduler job identifier.

**error** - Sets the standard-error file of the child job; the default uses the scheduler job identifier.

**account** - Sets the accounting project passed as --account.

**qos** - Sets the quality-of-service string passed as --qos.

**exclusive** - Requests exclusive node access when true.

**constraint** - Sets the scheduler node constraint string passed as --constraint.

**gres** - Requests generic resources such as a GPU. It is written as --gres and doubles as the switch that makes the training skill use its GPU submission template instead of the plain one.

**mail_type** - Sets the --mail-type events for scheduler notifications.

**mail_user** - Sets the --mail-user address for scheduler notifications.

**array** - Sets a scheduler array specification; a manual run normally leaves it empty because the framework submits one job per unit.

**extra** - Copies a free-form string into the generated script verbatim. It is the only key that is not validated against the directive list, so anything written here must already be valid script syntax.
