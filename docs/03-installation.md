---
title: "3. Installation"
nav_order: 3
---

This chapter installs the package, points it at the engines it will drive, and checks the result. The order matters: the three commands appear only after the install, and an engine is found only after its section in the configuration file names it.

## 3.1. System requirements

PyRAI3MD is a pure Python package. It runs under Python 3.9 or newer and is tested with 3.11. Four pip libraries are needed by every run: numpy &gt;= 1.20 for the array core, pydantic &gt;= 2 for the configuration model and its validation, PyYAML &gt;= 6 for the YAML carrier and the environment file, and matplotlib &gt;= 3.5 for the population plots of the benchmark calibrations.

Two optional packages extend a run without being needed by it. numba &gt;= 0.56 accelerates the FSSH propagator kernels; the NumPy reference runs without it. TensorFlow &gt;= 2.20 is needed only by the neural network potential skill, stays out of the default install because of its size, and is reported when missing. PyTorch &gt;= 2.9 is optional on the same terms.

Engines are external installations, with two exceptions: PySCF runs in process and is pip-installed, and the QDM models are analytic and need only numpy. [Section 3.4](#34-setting-up-calculation-engines) lists what each engine needs on the host.

| Dependency | Required | Needed for |
| :--- | :--- | :--- |
| Python &gt;= 3.9 | Yes | Interpreter; 3.11 is tested |
| numpy &gt;= 1.20 | Yes | Array core of framework and skills |
| pydantic &gt;= 2 | Yes | Configuration model and validation |
| PyYAML &gt;= 6 | Yes | YAML carrier and environment file |
| matplotlib &gt;= 3.5 | Yes | Benchmark population plots |
| numba &gt;= 0.56 | No | Accelerated FSSH kernels |
| TensorFlow &gt;= 2.20 | No | The nn_pes skill |
| PyTorch &gt;= 2.9 | No | Neural-network calculations |
| Engine installations | No | Engine jobs; see [Section 3.4](#34-setting-up-calculation-engines) |

## 3.2. Installing the package

Download the repository and enter it.

```bash
git clone <repository-url>
cd PyRAI3MD
```

Install into the active interpreter. The install creates three commands: pyrai3md for YAML and classic text, pyrai3md_agent for sectioned JSON, and pyrai3md_skill for one skill launched from a flat JSON config.

```bash
pip install .
```

The three commands enter the same pipeline at different carriers ([Section 4](04-input-files-and-conventions.html)). A remote install by the deploy tool follows the same sequence ([Section 14](14-qualification-and-verification.html)).

## 3.3. Configuration layers

Configuration is layered, and a run resolves each key from the highest layer that sets it: the environment and slurm sections of the run input, then ~/.pyrai3md/pmd_env.yaml, then the package template pyrai3md/qualification/pmd_env.yaml. The precedence is input &gt; home &gt; package.

The package template ships the defaults and is not edited. The home file is created on the first pyrai3md test run and holds engine paths, skill execution modes and shell lines, the scheduler, the deploy target and the switches.

The merge is per key: a key in the home file replaces the package value and leaves its siblings alone. An empty value counts as unset and falls back, while false and 0 are values. An input section merges the same way for that run alone.

```yaml
environment:
  python: python3
  scheduler: local
  shell:
    - module load compiler/gcc/gcc-11.3.0
skills:
  molcas_skill:
    molcas_path: /opt/molcas-26
    execution_mode: local
deploy:
  host: cluster.example.org
  username: user
  install_path: /home/user/pyrai3md
```

The environment section holds the interpreter, the conda environment, the scheduler and the shell and export lists; slurm holds the directives a dispatched child or grouped job asks for. A user file that names only what a host changes is normal. [Section 5.12](05-12-environment-and-slurm.html) documents both.

## 3.4. Setting up calculation engines

Each engine is a skill, and the framework never imports one. The skill reads its installation path, executable and parallel settings from its own section; the framework only forwards those keys. PySCF runs from the interpreter and the QDM models are analytic, so neither needs an installation.

Engine modules and the site base environment are loaded by the skill, not the framework. A skill section carries a shell list of module load lines and exports, and the skill copies that list into every script it writes, so a host states the CP2K compiler and MPI modules once, in the cp2k_skill section.

| Engine | Must be on the host | Key that names it | Notes |
| :--- | :--- | :--- | :--- |
| pyscf | PySCF importable in the interpreter | pyscf_path | Set to import for the current environment; no launcher. |
| molcas | OpenMolcas tree with pymolcas in it | molcas_path | Needs the site module that declares the base directory. |
| bagel | BAGEL tree with $BAGEL/bin/BAGEL and its library variables | bagel_path | The library variables are exported in the shell list. |
| orca | ORCA 5.0 or newer with an orca executable | orca_path | A parallel build also names its MPI library. |
| openqp | An openqp command in PATH, or OPENQP_ROOT | openqp_path | An MPI launcher only with use_mpi. |
| xtb | An xTB tree, the binary, or xtb in PATH | xtb_path | Threads and memory are separate keys. |
| cp2k | A CP2K installation with an executable | cp2k_path | cp2k_exe overrides it; setup_script is sourced first. |
| lammps | A LAMMPS tree, the binary, or lmp in PATH | lammps_path | The launch directory must hold a topology file. |
| qdm | Python and numpy only | none | Analytic models; two states by construction. |
| nn_pes | TensorFlow &gt;= 2.20 in the interpreter | none; modeldir names the folder | Discovered without TensorFlow; refuses a run by naming it. |

OpenMolcas is the one engine whose setup is more than a path. Its launcher resolves the installation from the environment the site module exports, so the shell list loads that module and the MOLCAS variable it exports, while molcas_path names the same tree.

PySCF is the engine to start with: a pip install, no module, and pyscf_path: import means the running interpreter is the installation. The nn_pes modeldir is empty on purpose, because it names a folder a training run produced on a particular machine. [Section 5.15](05-15-pyscf.html) to [Section 5.24](05-24-nn_pes.html) document each skill section.

## 3.5. Verifying the installation

Two commands check an installation. The info switch parses the input, resolves every layer and prints the configuration without executing anything.

```text
pyrai3md energy.yaml --info
```

The qualification round then exercises the installation end to end.

```text
pyrai3md test
```

The round plans from ~/.pyrai3md/pmd_env.yaml, opens a directory named tmp_YYYYMMDD_HHMMSS in the current directory, and writes two verdict files: results.json, one record per task, and status.json, the terminal state (pass, fail, error or skip). A local round leaves one row fragment per task under jobs/; a grouped Slurm round leaves the job specs, batch scripts and fragments there.

The work directory is disposable; the next round makes its own. What to run is decided in the home file rather than on the command line, through the _runall master, the family and skill switches and the per-case keys; a host that carries PySCF alone should leave the other families closed. [Section 14](14-qualification-and-verification.html) explains the categories, switches and verifier.

## 3.6. Installing on a cluster

A cluster installation is the same install done remotely, and the deploy tool performs it: it syncs the checkout to the configured host, installs it, starts a round and collects the result. The round directory stays on the host; only the log comes back as test_&lt;stamp&gt;.log.

```bash
python tools/deploy.py
python tools/deploy.py --force
```

The deploy section names the host, the user, the install path and optionally a jump host; environment.scheduler runs the round in one local process, or as one grouped Slurm job per carrier when it reads slurm. With Slurm, environment.max_in_flight caps unfinished jobs, and 0 submits the whole round. The force form cancels the jobs a Slurm round recorded and stops the orchestrator; work a skill submitted itself follows that skill section. [Section 14](14-qualification-and-verification.html) gives the sequence and what each step leaves.
