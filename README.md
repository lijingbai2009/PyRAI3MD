# Python Rapid with Advanced Infrastructure for Artificial Intelligence Ab Initio Molecular Dynamics
<pre>

                              /\
   |\    /|                  /++\
   ||\  /||                 /++++\
   || \/ || ||             /++++++\
   ||    || ||            /PyRAI3MD\
   ||    || ||           /++++++++++\                     __
            ||          /++++++++++++\     |\ |  /\  |\/| | \
            ||__ __    *==============*    | \| /--\ |  | |_/


                      Power your Research
                            with
      Artificial Intelligence-ready Advanced Infrastructure
                            for
                  Ab Initio Molecular Dynamics

                      Author: Jingbai Li

            Hoffmann Institute of Advanced Materials
                  Shenzhen Polytechnic University

                             2026

                        version:   3.0.0


</pre>

## Features
 - Modular skill architecture: the framework owns protocols and orchestration, each scientific engine is a skill that owns its own schema, input builder, runner and output parser.
 - Ten calculation skills: OpenMolcas, BAGEL, ORCA, PySCF, OpenQP, GFN-xTB, CP2K, LAMMPS, QDM and a neural network potential (NN PES).
 - Workflows for single-point energy, geometry optimization (minimum, TS, MECI, MECP, variable cell), path search (MEP, IRC, NEB), ab initio MD / NAMD (FSSH, GSH, LZSH), frequencies, numerical Hessian, PES scan, training, prediction, adaptive sampling and grid search.
 - Multiscale calculations: additive QM/MM with ESPF embedding, ONIOM2 / ONIOM3, and multi-center plans that mix quantum, machine-learning and classical layers.
 - Periodic systems: PBC geometry and minimum-image handling, NVE / NVT / NPT ensembles, thermostat and barostat families, periodic stress.
 - System constraints: harmonic bias, walls, SHAKE, frozen atoms and metadynamics, including variable-cell runs.
 - Three user input carriers plus machine entry points: YAML, classic text input, agent JSON, and serialized `.rt` runtimes; the same pipeline serves CLI, agent, and child-runtime execution.
 - Parallel and distributed execution: multiprocessing task pools (`multitask`), Slurm child runtimes (`remotetask`), and per-term remote workers.

## Prerequisite
 - **Python >= 3.9**. PyRAI3MD is written and tested with Python 3.11.
 - **numpy >= 1.20**, **pydantic >= 2.0**, **PyYAML >= 6.0**, **matplotlib >= 3.5**. Installed by `pip install .`.
 - **numba >= 0.56** (optional): accelerated FSSH electronic propagator kernels. The NumPy reference runs without it.
 - **TensorFlow >= 2.20** (optional): required only by `nn_pes_skill`. It is kept out of the default install because of its size; without it the skill is still discovered and reports what is missing.
 - **PyTorch >= 2.9.0** (optional): neural-network calculations. Like TensorFlow it stays out of the default install; other NN packages will be added as optional dependencies as they are integrated.
 - **Calculation engines**: each skill reads the installation path and executable from its own section of `pmd_env.yaml`. PySCF runs in process and can be pip-installed; xTB, OpenMolcas, BAGEL, ORCA, OpenQP, CP2K and LAMMPS are external installations.

## Content
<pre>
 File/Folder Name                                  Description
---------------------------------------------------------------------------------------------------
 PyRAI3MD
  |--setup.py                                     Package installer (name='pyrai3md', version 3.0.0)
  |
  |--pyrai3md                                     Python package
  |   |--pmd.py                                   Traditional CLI entry (pyrai3md) + orchestration
  |   |--agent.py                                 Agent entry (pyrai3md_agent) for JSON configs
  |   |--manifest.yaml                            Framework discovery: input schema + code entry points
  |   |--skill_guide.md                           Framework usage guide: CLI, input, outputs, contracts
  |   |
  |   |--interface                                Input conversion and structural validation
  |   |--schema                                   Pydantic configs, normalization, validation, constants
  |   |--planning                                 Capability resolution, task queue, assembly
  |   |--workflow                                 Workflow orchestration and Calculator
  |   |   |--optimize.py                          Geometry optimization loop
  |   |   |--path.py                              MEP / IRC / NEB
  |   |   |--md.py                                Ab initio MD and NAMD
  |   |   |--hessian.py, frequency.py, scan.py    Hessian, frequencies, PES scan
  |   |   |--adaptive.py, train.py, prediction.py, search.py
  |   |   |--optimizer                            minima, TS, MECI, MECP, GDIIS, cell
  |   |   |--integrator                           velocity Verlet, surface hopping, MD operations
  |   |   |--constraints                          bias, walls, SHAKE, projection, metadynamics
  |   |   |--multiscale                           composite aggregation, regions, embedding charges
  |   |   |--dispatch                             task pools, Slurm runtimes, per-term workers
  |   |   |--records                              trajectory, output, diagnostics, ML outcome
  |   |   |--grid                                 hyperparameter grid planning and read-back
  |   |
  |   |--data                                     Runtime state, geometry, centers, charges
  |   |--training_data                            Training data and permutation records
  |   |--kernel                                   Environment, Maestro skill registry, logging, cache
  |   |--utilities                                Structure/result formatting, geometry, lattice, units
  |   |--examples                                 212 shipped example YAMLs + topologies
  |   |
  |   |--qualification                            Software qualification infrastructure
  |   |   |--manager.py                           Four-category plan + shared script writers
  |   |   |--runner.py                            Execution and report (local or grouped Slurm)
  |   |   |--jobs.py                              Grouped Slurm submission, polling, collection
  |   |   |--pmd_env.yaml                         Environment, deploy and switch template
  |   |   |--verification                         Framework unit tests
  |   |   |--integration                          Cross-mode and shipped-example tests
  |   |   |--certification                        Skill test planning and converters
  |   |   |--benchmark                            FSSH / GSH / LZSH / kTDC / perf calibrations
  |   |
  |   |--skills                                   Skill plugins
  |       |--wrapper.py                           pyrai3md_skill command entry
  |       |--query.py                             Dynamic skill discovery
  |       |--pyscf_skill                          PySCF
  |       |--molcas_skill                         OpenMolcas
  |       |--bagel_skill                          BAGEL
  |       |--orca_skill                           ORCA (DFT / TDDFT)
  |       |--xtb_skill                            GFN-xTB
  |       |--openqp_skill                         OpenQP (HF / TDDFT / MRSF)
  |       |--cp2k_skill                           CP2K Quickstep DFT / TDDFPT
  |       |--lammps_skill                         LAMMPS classical MM, QM/MM field and energy
  |       |--qdm_skill                            Analytic Tully / Landau-Zener model engines
  |       |--nn_pes_skill                         Neural network potential (optional TensorFlow)
  |
  |--tools
  |   |--deploy.py                                Remote deployment and test-round driver
  |   |--verify.py                                Post-round artifact verifier (A-K + selftest)
  |   |--benchmark.py                             Benchmark driver
  |   |--analyze_npt_frames.py                    Cell-history recomputation
  |   |--refresh_water_fixture.py                 Fixture refresh utility
  |   |--verify_seed_reproducibility.py           Cross-build bit-identity check
  |   |--training_data.py                         Training-data extraction and operations
  |   |--traj_analyzer.py                         Trajectory analysis
  |   |--traj_generator.py                        Trajectory generation
  |   |--multiscale_generator.py                  Solvent/aggregate environment generator
  |   |--manual_update.py                         Manual install update helper
  |
   `--dev                                         Architecture, method and skill development guides
      |--ARCHITECTURE.md                          The code architecture specification
      |--QUALIFICATION_DEVELOPMENT.md             Test framework design
      |--methods/                                 MD, optimize, path, PBC, QMMM, ONIOM, ...
       `-skills/                                  One development guide per skill
</pre>

Every skill follows the same layout: `agent.py` (CLI), `backend.py` (public API), `query.py` (manifest / profiles / defaults), `schema/` (config and validation), `service/` (builder, workflow, runner, parser, templates), and `tests/` (YAML cases plus runtime tests).

## Installation
Download the repository

    git clone <repository-url>

Install the package. This creates the commands **pyrai3md**, **pyrai3md_agent** and **pyrai3md_skill**.

    cd PyRAI3MD
    pip install .

## Run PyRAI3MD

Configuration is layered. The package template `pyrai3md/qualification/pmd_env.yaml` supplies the defaults, `~/.pyrai3md/pmd_env.yaml` (created on the first run) overrides it, and the `environment` / `slurm` sections of the runtime input override both, so the precedence is **input > home > package**. Edit the engine paths, execution mode and Slurm settings in the user file.

    pyrai3md input.yaml           # YAML input
    pyrai3md input.input          # classic text input
    pyrai3md input.yaml --info    # parse and print the resolved configuration only

    pyrai3md_agent config.json    # agent entry, JSON configuration
    pyrai3md_skill config.json    # run one skill application directly

A minimal YAML input (also shipped as `examples/energy.yaml`)

    control:
      title: energy_example
      jobtype: energy
      theory: [pyscf]

    atoms: [C, C, H, H, H, H]
    coordinates:
      - [-1.120191050, 0.000000000, -2.661941306]
      - [-1.120191050, 0.000000000, -1.331783306]
      - [-1.120191050, 0.923341000, -3.234891306]
      - [-1.120191050, -0.923341000, -3.234891306]
      - [-2.043532050, 0.000000000, -0.758833306]
      - [-0.196850050, 0.000000000, -0.758833306]

    molecule:
      charges: 0
      multiplicity: 1
      ci: [1]
      spin: [0]

    pyscf:
      method: hf
      basis: sto3g
      scratch_path: ./scratch

The run writes `<title>.log` and, for iterative workflows, the trajectory and status records in the working directory. The 212 shipped examples under `pyrai3md/examples/` cover energy, gradient, optimization, path, MD/NAMD, multiscale and adaptive calculations.

## Test PyRAI3MD
Run the qualification round from any directory

    pyrai3md test

The round plans four categories: framework unit and integration tests, shipped-example carriers, skill certification cases and benchmark calibrations. Example and skill tasks run through the same input pipelines and CLI entry points a user would use. The work directory is `tmp_YYYYMMDD_HHMMSS/`; the verdict is written to `results.json` and `status.json`, with one row fragment per task under `jobs/`.

Test selection lives in `~/.pyrai3md/pmd_env.yaml`: `_runall` (master switch), `test_integration`, the example families, `skills.<name>.test` / `.runtime_tests`, and the benchmark entries. Engine-specific settings are read from the matching skill section.

Remote testing uses the same round driver

    python tools/deploy.py         # sync, install, trigger, poll, collect
    python tools/deploy.py --force # cancel the round and its Slurm jobs

Configure the `deploy` section (`host`, `username`, `install_path`, optional `jump_host`) and `environment.scheduler` in `pmd_env.yaml`. With `environment.scheduler: slurm` the round is submitted as independent grouped jobs (one per example carrier, per skill carrier leg, per benchmark case, plus one for verification and one for integration); `environment.max_in_flight` caps unfinished jobs (`0` submits the whole round at once). Skill self-submission remains controlled by `skills.<name>.execution_mode`.

Audit a downloaded round with the post-round verifier

    python tools/verify.py <work_dir>
    python tools/verify.py --selftest

It cross-checks records against `results.json`, step logs, engine outputs, carrier parity, multiscale identities, adaptive products and skill decks / final energies. `--selftest` runs planted-error negative controls.

## Data and trajectory tools

    python tools/training_data.py extract read.yaml -o data.json
    python tools/training_data.py shuffle data.json -s 1234
    python tools/traj_analyzer.py analysis.in
    python tools/traj_generator.py sampling.in

`extract` calls the read-only `jobtype read` flow (a `read` section in
YAML or `&read` in classic input) and saves the canonical training-data
JSON. The other subcommands (`shuffle`, `split`, `merge`, `select`)
maintain that file. The trajectory tools keep the legacy Molcas /
NewtonX / SHARC / fromage readers and write/read PyRAI3MD trajectories.
See the manuals under `tools/`.

# User manual
- PDF manual: link to be added.
- Online wiki: link to be added.

# Cite us
PyRAI3MD is the successor of PyRAI2MD. The methods it implements are described in:
- Jingbai Li, Patrick Reiser, Benjamin R. Boswell, André Eberhard, Noah Z. Burns, Pascal Friederich, and Steven A. Lopez, "Automatic discovery of photoisomerization mechanisms with nanosecond machine learning photodynamics simulations", Chem. Sci. 2021, 12, 5302-5314. DOI:10.1039/D0SC05610C

- Jingbai Li, Rachel Stein, Daniel Adrion, Steven A. Lopez, "Machine-learning photodynamics simulations uncover the role of substituent effects on the photochemical formation of cubanes", J. Am. Chem. Soc. 2021, 143, 48, 20166-20175. DOI:10.1021/jacs.1c07725

- Jingbai Li, Steven A. Lopez, "Excited-state distortions promote the reactivities and regioselectivities of photochemical 4π-electrocyclizations of fluorobenzenes", Chem. A Eur J. 2022, 28, e202200651. DOI:10.1002/chem.202200651

- Jingbai Li, Steven A. Lopez, "A Look Inside the Black Box of Machine Learning Photodynamics Simulations", Acc. Chem. Res., 2022, 55, 1972-1984. DOI:10.1021/acs.accounts.2c00288

- L. Wang, C. Salguero, S.A. Lopez, J. Li, "Machine learning photodynamics uncover blocked non-radiative mechanisms in aggregation-induced emission", Chem, 2024, 10, 2295-2310. DOI: 10.1016/j.chempr.2024.04.017

- Z. Li, F. J. Hernández, C. Salguero, R. Crespo-Otero, S.A. Lopez, J. Li, "Machine learning photodynamics decode multiple singlet fission channels in pentacene crystal", Nat Commun, 2025, 16, 1194. DOI:10.1038/s41467-025-56480-y

- Z. Li, H. Fu, S. A. Lopez, J. Li, "Machine learning photodynamics reveal intersystem-crossing-driven ladderdiene ring opening", Chem. Sci. 2025, 16, 13031-13041. DOI:10.1039/D4SC07395A

- L. Wang, J. Li, "Modeling the Nonradiative Decay of Cyclooctatetrathiophene in Solution and Crystal Phases Using Machine Learning", J. Chem. Theory Comput. 2025, 21, 19, 9300-9308. DOI:10.1021/acs.jctc.5c01136

# License
PyRAI3MD is distributed under the PyRAI3MD Research License 1.0. See `LICENSE`.