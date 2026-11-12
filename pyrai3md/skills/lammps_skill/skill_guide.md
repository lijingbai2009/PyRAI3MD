# Skill Guide: lammps_skill

## Overview
Use lammps_skill to perform classical molecular mechanics calculations through LAMMPS: single point energy and gradient, internal geometry optimization (minimize), and standalone molecular dynamics in the nve/nvt/npt ensembles. This skill provides a configuration-driven, profile-based interface for user-supplied LAMMPS force field inputs. The force field itself (template and topology files) is provided by the user; the skill does not generate them (tooling to be added later).

## Setup Requirements
- LAMMPS installed (https://www.lammps.org)
- Either set `lammps_path` in config to the LAMMPS installation directory (expects `$lammps_path/bin/lmp`, legacy convention), or point `lammps_path` directly to the LAMMPS binary, or leave it empty to search `lmp` in PATH
- Parallelization has two axes: `nproc` MPI ranks (launched with `mpirun -np N`
  when above 1) and `thread` OpenMP threads inside each rank (#306)
- User-provided files in the launch directory (validated before any calculation):
  - `<project>.data`: LAMMPS topology (data) file (required)
  - Optional: `<project>.in` (LAMMPS input template; when absent the input script is fully generated from the profile), `<project>.in.init`, `<project>.in.settings` (legacy ffa/moltemplate convention; `project` = `lammps_project`, default `LAMMPS`)

## Supported Profiles

| Profile | Capability | Method | LAMMPS execution |
|---------|-----------|--------|------------------|
| ff_energy | Single point energy | ff | `run 0` + thermo (PotEng) |
| ff_gradient | Energy + gradient (= -force) | ff | `run 0` + thermo + force dump |
| ff_optimize | Geometry optimization | ff | `minimize` + final structure dump |
| ff_nve | Classical MD (NVE ensemble) | ff | `fix nve` + `run N` + final structure dump |
| ff_nvt | Classical MD (NVT ensemble) | ff | `fix nvt` + `run N` + final structure dump |
| ff_npt | Classical MD (NPT ensemble) | ff | `fix npt` + `run N` + final structure dump |
| qmmm_field | QM/MM MM potential/field probe (Flow 1) | ff | A/B `run 0` (q_Q=1), per-Q V/E |
| qmmm_energy | QM/MM classical energy (Flow 2) | ff | full FF with q_qm + Python in-cell term |

## Roles
- **mm**: LAMMPS serves as the molecular mechanics layer only
- **qm / qm2**: not supported

## Workflow
1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Prepare LAMMPS Files**: Place `<project>.data` (and optional `<project>.in` / `<project>.in.init` / `<project>.in.settings`) in the launch directory
3. **Create Configuration File**: Prepare a JSON config file with system info, profile, and method
4. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m pyrai3md.skills.lammps_skill.agent config.json [local|slurm]`
   - **Python API**: `from pyrai3md.skills.lammps_skill import LammpsBackend`
5. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods
lammps_skill provides a self-contained query module that reads its own manifest.yaml:
```python
# From the skill's parent directory:
from lammps_skill import query

# List all available profiles
print(query.list_profiles())
# ['ff_energy', 'ff_gradient', 'ff_optimize', 'ff_optimize_cell',
#  'ff_nve', 'ff_nvt', 'ff_npt', 'qmmm_field', 'qmmm_energy']

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('ff_energy')
# {'capability': 'energy', 'method': 'ff', 'steps': ['energy']}

# Get LAMMPS step chain for a profile
steps = query.get_steps('ff_gradient')
# ['energy', 'gradient']

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient', 'geometry_optimization', 'classical_dynamics']

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API
```python
# From the skill's parent directory:
from lammps_skill.backend import LammpsBackend

# Initialize backend (default calcdir: tmp_LAMMPS)
backend = LammpsBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = LammpsBackend(config_dict, execution_id='tmp_LAMMPS')  # -> scratch/tmp_LAMMPS

# Or with explicit full calcdir path:
backend = LammpsBackend(config_dict, calcdir='/path/to/calcdir')

# Execute calculation (local or slurm is selected by execution_mode in config)
results = backend.calculate(atoms, coordinates)

# For multi-step dynamics, reuse the same backend instance:
for step in trajectory:
    results = backend.calculate(atoms, coordinates)
    # coordinates are updated in the data file each step; run 0 single point
```

## Configuration Schema
Key fields in the JSON config:
- `atoms`: List of atom symbols
- `coordinates`: List of [x, y, z] coordinates in Angstrom
- `profile`: Calculation profile ('ff_energy', 'ff_gradient', 'ff_optimize', 'ff_nve', 'ff_nvt', 'ff_npt')
- `method`: Method name ('ff')
- `execution_mode`: 'local' or 'slurm'
- `lammps_project`: Project name for LAMMPS files (default: 'LAMMPS')
- `lammps_path`: Path to LAMMPS installation directory or binary ('' to search PATH)
- `nproc`: Number of MPI ranks (default: 1); `lammps_nproc` is the name the
  framework forwards it under, and a skill keeps the larger of a written pair
- `thread`: OpenMP threads per rank, exported as `OMP_NUM_THREADS` (default: 1);
  `lammps_thread` is its framework spelling. `nproc` and `thread` are two axes:
  a SLURM allocation asks for the product (#306)
- `lammps_units`: LAMMPS units style of the user input, 'real' (kcal/mol) or 'metal' (eV) (default: 'real')
- `ewald_accuracy`: the accuracy token written onto the `kspace_style` line this skill re-issues after `read_data`; the reciprocal style stays the template's. Default '1e-8'. Not a speed dial: the periodic QMMM field probe takes V from the difference of two runs whose squared charge differs by orders of magnitude, and at 1e-6 that difference was short by 3.3e-4 relative on the 1581-atom box. A classical trajectory pays 1.3-2x, which is why the shipped non-probe cases spell their template's number out instead of inheriting
- `python_enabled`: resident-service switch, bool (default: false); see Resident Execution
- `python_path`: interpreter of the service process, i.e. the python that can `import lammps` (default: '' = the interpreter running PyRAI3MD). `lammps_path` keeps meaning the lmp binary or installation directory
- `debug`: resident-path switch, bool (default: false). false, a step exchanges geometry and numbers over the socket and writes nothing but the service log; true, every step also issues the write commands the subprocess path issues, so the same `<title>.energy`/`.coords`/`.force`/`.stress` result files appear beside the input and the two paths can be compared file for file (bringup's C10 does exactly that, byte for byte). Inert on the subprocess path, where `keep_tmp` already decides what survives
- Force-field compression is not a switch: the backend always analyses the templates once into a JSON model (`<project>.ff.json`, written beside the calculation directory) holding only the referenced force-field types, and renders every input file from it — including `<project>.in.init`, whose `*_style hybrid` declarations are trimmed to the sub-styles the surviving coefficients actually use (LAMMPS aborts on a hybrid sub-style no `pair_coeff` assigns). A template the model cannot prove equivalent degrades to the verbatim templates with a warning
- `lammps_shake`: Shake constraint switch, 0 (off, default) or 1 (on). Rejected for the framework-integrated profiles (`ff_energy` / `ff_gradient`), see below
- `shake_fix_id`: Name of the shake fix defined in the input template (default: 'fShakeTIP3P', legacy ffa/moltemplate TIP3P convention)
- `shake_keywords`: Raw `fix ... shake ...` command injected when the template defines no such fix and `lammps_shake=1`
- `scratch_path`: File placement directory for calculations
- Geometry optimization parameters (`ff_optimize`):
  - `min_style`: Minimization algorithm (default: 'cg')
  - `min_etol` / `min_ftol`: Energy / force tolerance (defaults: 1e-4 / 1e-8)
  - `min_maxiter` / `min_maxeval`: Max iterations / evaluations (defaults: 1000 / 1000000)
- Standalone MD parameters (`ff_nve` / `ff_nvt` / `ff_npt`):
  - `md_nstep`: Number of MD steps (default: 100)
  - `md_timestep`: Timestep in fs for `real` units (default: 1.0)
  - `temperature`: Target temperature in K (default: 298.15)
  - `thermo_damp`: Thermostat damping parameter for nvt/npt (default: 100)
  - `pressure` / `pressure_damp`: NPT target pressure (atm) / barostat damping (defaults: 1.0 / 1000)
  - `dump_freq`: Thermo output frequency (default: 100)
  - `md_seed`: RNG seed for `velocity all create` (default: 12345)
  - `velocities`: Optional (natom, 3) velocities in Bohr/au; when provided they are written into the data file instead of `velocity all create`

## Shake Constraint Switch (&lammps keywords)
A geometric constraint must be applied by whatever integrates the positions, so `lammps_shake` and the framework's `constraint` block are two separate control sources and each owns the constraint only on its own side. For the framework-integrated profiles (`ff_energy`, `ff_gradient` — one force evaluation per step, the framework propagates) `lammps_shake=1` is rejected by `schema/validate.py`, and the refusal names the framework `constraint` block as where the declaration goes; nothing is silently dropped. For the engine-integrated profiles (`ff_optimize`, `ff_nve`, `ff_nvt`, `ff_npt`) LAMMPS moves the atoms itself, so a shake fix stays consistent and `lammps_shake=1` is allowed.

The framework side is not an equivalent substitute. Its default solve constrains positions only — the velocity constraint is taken up only by a record declaring `velocity_constraint: rattle` — see `dev/methods/CONSTRAINT_DEVELOPMENT.md` §1.3 and §3.7.

- The decision is recorded once in the force-field model: the rendered `<project>.in.settings` either carries the renumbered `group`/`fix` lines or nothing at all, and the input script emits **neither `fix` nor `unfix`**
- Only when the model refuses a template (see below) does the legacy mechanism apply: `lammps_shake=0` appends `unfix <shake_fix_id>` before `run 0` to disable a template-defined fix, and `lammps_shake=1` strips every `unfix <shake_fix_id>` line so the template fix stays active, injecting `shake_keywords` when the templates define no such fix (error if empty)

Dropping the constraint is not free: the timestep has to resolve the constrained motions itself. For the bundled TIP3P water box (`k = 450 kcal/mol/A^2`, reduced mass 0.9476 amu) the O–H period is about 14 fs, so a 1 fs timestep keeps roughly 14 points per period.

## QM/MM Tasks (`qmmm_field` / `qmmm_energy`)
QM/MM is two standalone profiles, not a switch inside `ff_gradient`:

- `qmmm_field` (Flow 1) probes the MM electrostatic potential/field at the QM atoms with `q_Q=+1`. It runs A (full system) and B (MM charges zeroed) with the same coefficients, cutoffs, special_bonds and kspace, then reports `V_i = 2(peA_i - peB_i)` and `E_i = FA_i - FB_i`. The factor 2 is the `pe/atom` half allocation. The A group tally is the guard: `sum V = U_GG` and `sum E = F_GG`, with `compute group/group Q M pair yes [kspace yes]`.
- `qmmm_energy` (Flow 2) writes the QM 0-order charges `q_qm` into the data file, removes the QM region's bonded rows, and runs the full force field once. The backend then evaluates `in_cell_qq` in Python (`U_gas`, `F_gas`), forms `qmmm_el = sum_i q_qm_i V_i` and `F_el,i = q_qm_i E_i` from the Flow-1 arrays, and returns `E_classic = U_main - U_gas - qmmm_el`.

`qm_atoms` is 0-based in the framework atom order. `q_qm` is in the same order; `qmmm_field_file` may point at a four-column ESPF file for a standalone energy run. The retired `qmmm` boolean is refused. The field task needs a neutral MM subsystem (so A/B Ewald backgrounds cancel); v1 `qmmm_energy` keeps QM and MM partitions neutral. Both tasks refuse `lammps_shake`, `atom_selection`, a template without a charge column, and a bonded row crossing the partition.

### Files
`qmmm_field` writes `<title>.esfp` (atomic units, one row per QM atom: `V Ex Ey Ez`; E is the force on a +1 probe) and returns native V/E in the `qmmm_field` result block. It does not write the production `.energy`/`.force`.

`qmmm_energy` leaves the standard `<title>.energy`/`<title>.force` as `U_main` and the full atomic forces, and adds:

```
<title>.qm_multipole   # id q, e, qm_atoms order
<title>.qmmm           # E_classic, U_main, U_gas, qmmm_el, native units
<title>.u_gas.force    # QM rows, native force
<title>.qmmm_el.force  # QM rows, q_qm*E, native force
<title>.grad           # all atoms, REDUCED classical gradient, dU/dR, native
```

The `.esfp`/`.qm_multipole` files are output/audit only. A runtime caller passes q_qm and V/E through `calculate(..., qmmm_inputs={...})`; the energy task never reads `.esfp` back per step. The standard `.force`/`.energy` are `U_main` and stay unreduced, because they are the verification record; what the skill RETURNS is the reduced classical surface (`gradient`, mirrored on disk as `.grad`): the QM rows have `dU_gas/dR` and `dqmmm_el/dR` removed, every other row is `U_main`'s own. The QM-MM electrostatic force on MM is already in `U_main`'s MM rows; it is not subtracted.

### Forces and units
The QM/MM tasks return native numbers under the framework's labels: the parser negates the engine's force into `gradient` (the sign belongs to the label) and marks the unit set with `_native_units`; the framework scales by the units this skill's manifest declares. `.esfp` is the atomic-unit exception for Molcas. `.qmmm`, its auxiliary gradients and `.grad` stay native: composite input and audit data, not Properties.

### Known limit
v1 does not zero Q-Q non-Coulomb coefficients: QM and MM atoms commonly share LAMMPS types, so type-based zeroing would remove Q-M/M-M terms too. The shipped water fixture has epsilon=0 on the Q-Q rows; a general Q-Q LJ treatment is future work.

## Resident Execution (python_enabled)
`python_enabled: true` moves the expensive part of a LAMMPS calculation into `LammpsBackend.__init__`: the compressed force field and the generated input package are handed once to a **service process** that holds a live LAMMPS instance through the LAMMPS Python module, and every later `calculate()` sends only that step's coordinates and box over a Unix socket and receives energy, forces and the configurational stress. No input file, run script, log or dump is written per step, and no `lmp` process is started per step.

| keyword | default | meaning |
|---------|---------|---------|
| `python_enabled` | `false` | Ask for the resident service. false keeps the subprocess path byte-for-byte unchanged |
| `python_path` | `''` | Interpreter of the **service** process, i.e. the one where `from lammps import lammps` works. Empty means the interpreter running PyRAI3MD |

Scope and limits, all deliberate:

- **Who asks for it**: the skill's own `ff_gradient_persistent.yaml` case, and all twelve official framework-driven MD examples (`pyrai3md/examples/md_lammps_*.yaml`), each of which declares `python_enabled: true` in its own `lammps` section rather than inheriting it, so running an example by hand takes the same path the qualification task takes.
- **Profiles**: `ff_energy`, `ff_gradient`, `qmmm_field` and `qmmm_energy` are served resident. Framework-driven dynamics, geometry optimization and variable-cell runs ask the mm layer for one force evaluation per step (`jobtype 'md'` resolves to `ff_gradient`), so that is the hot loop the service accelerates. `ff_optimize`/`ff_nve`/`ff_nvt`/`ff_npt` run their whole internal iteration inside one `calculate()` call and stay on the subprocess path.
- **QM/MM**: the field task serves A/B on the live instance and returns per-Q V/E; the energy task sets the per-step q_qm and returns U_main/forces. The V/E and q_qm travel in memory (EVALUATE carries q_qm), never through a per-step `.esfp` or `.qm_multipole` read. The backend runs the Python `in_cell_qq`/qV/qE assembly after the service reply, exactly as on the cold path.
- **Routing**: `python_enabled` is an intent; the decision is `python_enabled and execution_mode != 'slurm'`. With `execution_mode: slurm` the skill submits one blocking `sbatch -W` job per calculation, so there is nothing to keep resident, and a one-line warning says the switch was ignored. That is a designed route, not a degradation.
- **No silent fallback**: once a step is routed resident, a service that cannot start, a template the compressor could not prove equivalent, or a reply that fails validation raises. It never re-runs the calculation through the subprocess path and never returns a zero-filled result.
- **Fixed topology**: atom count, order, species, charges, masses, coefficients, units, pbc class and the constraint decision belong to the loaded instance. The backend was initialized with them; a step asking for anything else is refused, and the fix is a new backend, not a hot update.
- **MPI**: `lammps_nproc` ranks are the service's own world (`mpirun -np N` when above 1). The framework process never imports `mpi4py` or `lammps` and never learns a rank; forces and stress are gathered by atom id inside the service, so a decomposed run answers with the same global arrays a serial one does. `lmp.finalize()` is never called: it would tear down MPI under the sibling instances.
- **Readiness is answered by the service**: before it binds a socket the service imports `lammps`, constructs its instance (with a communicator when `lammps_nproc > 1`) and checks that the world it got has the ranks it was asked for. A service that cannot work has therefore already exited and written its reason into `<basename>.log`, and the launcher's error quotes the tail of that log. There is no separate pre-flight test of the interpreter: a second process asking the same questions can only produce an answer that disagrees with the one that matters.
- **`python_path` and `shell` name one installation**: the service inherits the `shell` module loads, which is what puts `libmpi` on the search path for the LAMMPS module. `liblammps` links exactly one MPI implementation (`ldd <site-packages>/lammps/liblammps.so | grep libmpi` shows which), and the `mpirun` that starts the service has to come from that same one; a launcher from another family leaves each rank its own singleton world, which the service detects and refuses to serve. If `python_path` points at a different LAMMPS build than `lammps_path` does, the resident and subprocess paths compute on two different binaries and their numbers are no longer comparable.
- **Variable cell**: every step may carry a new cell snapshot; the box travels as the true `boxlo/boxhi/xy/xz/yz` numbers the same helper writes for the data file. `change_box` gives each dimension its own keyword, so the nine numbers are issued as two commands (bounds, then tilt factors); the single-clause `change_box all final <9 numbers>` is an illegal command and would abort the process group. No `remap` is sent: it would accrue box crossings into the image flags, and the service states those flags itself (next bullet). The box is read back and compared afterwards. A mismatch ends the session instead of continuing on a box the domain did not accept.
- **Image flags travel with the coordinates**: the framework stores wrapped (main image) positions, so the crossing counts of the same step arrive in the `EVALUATE` message and are written as `set atom <idspec> image nx ny nz` after the positions and before the evaluation. `<idspec>` is one atom ID or a `lo*hi` range and nothing else: `set.cpp` parses the selection as a single token through `utils::bounds`, so a space-separated ID list is read as a stray keyword and aborts the job (`ERROR: Illegal set command`), and `ID` is not a style word at all. A target group is therefore emitted as its maximal contiguous runs, which keeps the first evaluation of a 648-atom box at one command. A live instance keeps the flags of the previous step, and a new wrapped coordinate laid over a stale flag relocates that atom by a whole box, which is what LAMMPS reports as inconsistent image flags and what makes a bond evaluate across the periodic boundary. The counts are seeded once at configuration time by `schema.normalize.seed_initial_image`, which perceives the bonds of the input geometry and propagates the crossings along each molecule, so an equilibrated frame that was already wrapped across a face does not start out read as all zero. The first step restates every atom rather than trusting what `read_data` derived from box containment; later steps send only what changed, and a step where nothing crossed sends nothing. The subprocess path is given the same numbers as the three trailing Atoms columns, so both paths hand LAMMPS the identical `(coordinate, image)` pair.
- **Result files, one per quantity**: a single point prints its own answer into `<title>.energy` (one number), `<title>.coords` and, when asked, `<title>.force` and `<title>.stress`, all four under the stem `output_basename()` gives (`h2o_force` for `ff_gradient`). `step_data_files()` is the single source of those names for writer and parser alike, and the parser reads them instead of scanning the log, so the energy keeps fifteen significant digits rather than the thermo table's eight. A missing file is an error quoted with the log's tail, never a zero. `ff_optimize` and the MD profiles keep the log as their energy channel, since LAMMPS runs a trajectory there and no per-step file exists.
- **Tilt capability follows the cell**: a domain is made tilt-capable when a cell that carries tilt arrives, not at load. Converting a box that needs no conversion changes the numbers (measured in the test environment: an untilted 11x10x10 cell gives -19.811016 kcal/mol as loaded and -19.760369 after a conversion made for nothing, because the reciprocal sum is re-framed), which would break the agreement with the subprocess path the resident service exists to keep. Untilted steps therefore also skip the tilt-factor command, which LAMMPS rejects on an orthogonal domain. Once a step has carried tilt the domain keeps that representation, so a later untilted cell of the same session is computed on a tilted domain with zero tilts.
- **Ownership**: one backend, one service, claimed under its socket path; a second resident backend on a socket a live service already answers is refused. The claim is not keyed on `execution_id`, because that name is a directory convention and an isolated-workdir batch shares it across units on purpose (each unit has its own calculation directory, and the parent optimization loop holds one service of that namespace while its numerical Hessian runs). `close()` (also reached by `with` and by a finalizer at interpreter exit) releases it. A hard `kill`/`SIGTERM` of the owning process is caught by the service's own `--owner-pid` watchdog, then by the batch job's walltime. On the normal path the unit that built a Calculator releases it: `workflow.execute()` closes in a `finally`, and the MD loop closes at the end of a trajectory, so a resident session is asked to `CLOSE` when its run ends. What no `close()` reaches is a hard `kill`/`SIGTERM` of the owning process, which is the watchdog's case. Anything the service wants to leave behind must still be written during the run, not at close: the per-phase table is refreshed into `<calcdir>/lammps_resident.perf` every 100 evaluated steps for exactly that reason, and a table written only on `close()` appears in no downloaded example run. Three tables describe one resident step and each is written by its own owner: `pyrai3md_step.perf` in the run directory (the workflow loop, framework side), `<calcdir>/lammps_backend.perf` (this skill's backend around the session, with the client's socket buckets appended under their own header), and `<calcdir>/lammps_resident.perf` (inside the engine). Measured in the test environment, the 10000-step NVE example runs 4.20 ms per step of framework wall time against 1.47 ms inside the engine, and the rows account for 99.3% of the step; the ranking, the two fixes that took it out of 8.45 ms and the candidates that measurement ruled out are in `dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` §11.12 and CHANGELOG #244.
- **Everything in the calculation directory, under the subprocess names**: the directory holds the one-time package (`<project>.data`, `<project>.in.init`, `<project>.in.settings`, `<project>.ff.json`, `<basename>.in`), the launch script `<basename>.sh` (which starts the service instead of `lmp`), the log `<basename>.log` (the service's own output plus everything LAMMPS prints to the screen), and while the session is live the endpoint `svc.sock` and the owner record `owner.json`. No private subdirectory is created, so a resident run's directory contains exactly the kinds of files a cold run's does, and a failure turns up where the subprocess path already taught you to look; one `keep_tmp` decides the fate of all of them. The two process files carry fixed short names because the directory is already the namespace, and both are unlinked when the service retires. Only a directory path too long for an AF_UNIX socket name leaves the socket in a per-user runtime directory under the system temp root (mode 0700), and the log's first line then names where it went.
- **`keep_tmp`** means the same thing on both paths: 0 clears the calculation directory when the service closes, anything else leaves the package, the script, the log and the dumps for inspection.

Verify a deployment before trusting it:

```bash
python -m pyrai3md.skills.lammps_skill.service.persistent.bringup \
    --dir /path/to/inputs --project h2o --nproc 1 2 --stress --periodic
```

It reports PASS/FAIL/SKIP per check (interface surface, subprocess-versus-resident numbers, variable cell, rank independence, owner watchdog, and where every file landed: the directory contents of a plain and a `debug` session, nothing left in the runtime directory, the four result files byte-compared between the two paths) and exits non-zero on a failure. The unit tests run the same service loop against a stub, so a PASS there says nothing about real LAMMPS numerics.

## Results
- `qmmm_field`: Present on the `qmmm_field` task. Carries native `v_mm_native`, `e_mm_native`, `u_gg_native`, `f_gg_native`, the probe charge and the group-check residuals. Its `energy`/`gradient` are probe-only zeros.
- `qmmm`: Present on the `qmmm_energy` task. Carries native `u_main_native`, `u_gas_native`, `qmmm_el_native` and `e_classic_native`, the received `q_qm`, the QM-row gradients `u_gas_gradient_native`/`qmmm_el_gradient_native`, and `mm_gradient_native` - the reduced classical gradient the framework consumes as `gradient`. See "QM/MM Tasks" above.
- `energy`: List of one potential energy in Hartree (converted from kcal/mol for `real` or eV for `metal`); always the total energy (the per-atom normalization of the internal-iteration thermo output is reversed by the parser)
- `gradient`: Gradient array of shape (1, natom, 3) in Hartree/Bohr (gradient = -force, converted from kcal/mol/Angstrom or eV/Angstrom)
- `fin_coord`: Final structure of shape (natom, 3) in Angstrom, returned by the internal-iteration profiles (ff_optimize and the MD profiles) through the framework fin_coord channel; the framework updates the system coordinates (keeping its own element list) and drops the key
- `fin_cell`: The cell the run ended on, as the nine flattened lattice components (row-major, Angstrom), returned by the MD profiles when their dump carries a box. `fix npt` holds the cell lengths as degrees of freedom, so the run can finish in a box the input never declared while its coordinates already belong to that box; the framework derives `cell`, `lattice_vector` and `volume` from these nine numbers and refuses the value outright if the system was declared non-periodic. A fixed-cell run reports the cell it was given, which the framework notices and leaves alone
- `fin_velocity`: Final velocities of shape (natom, 3) in Bohr/au, returned by the MD profiles (ff_nve/ff_nvt/ff_npt) through the fin_velocity channel. The framework applies it to the system state exactly as it applies `fin_coord` — an engine that propagated the nuclei owns their velocities too — and recomputes the kinetic energy from the pair, so the run log's `&velocities` and `&kinetic energy` describe the run that happened rather than the zeros the step started from. Every skill reserves this channel; skills without MD output keep it empty, and an empty value leaves the state untouched

## Important Notes
- Coordinates are injected into the Atoms section of `<project>.data` by replacing the last 3 columns of each atom line (all other columns such as id/type/mol/charge are preserved as-is)
- The generated input script injects `include <project>.in.init` / `read_data <project>.data` / `include <project>.in.settings` at the top (legacy order), then appends the workflow tail:
  - single point: `thermo_style custom step pe etotal`, `thermo 1`, `run 0`, then `variable pyrai_pe equal pe` with `print` into `<title>.energy`, a `<title>.coords` dump, the optional force dump `<title>.force`, and the optional `<title>.stress` print
  - ff_optimize: `min_style` + `minimize etol ftol maxiter maxeval`, final `write_dump xyz` (`<project>_<profile>.xyz`) and `write_data restart.data`
  - ff_nve/ff_nvt/ff_npt: thermo control (`thermo_style multi` + `thermo_modify norm yes`), velocity initialization, `timestep`, the ensemble fix, `run N`, final `write_dump xyz` (`<project>_<profile>.xyz`), final `write_dump custom` (`<project>_<profile>.cond` with `id x y z vx vy vz`) and `write_data restart.data`
- `restart.data` is written for the agent's own sequential-task orchestration, not as a PyRAI3MD continuation entry: the framework never reads it, and `read_data` always points at `<project>.data`
- All skill-generated dumps are element-independent: no `modify element` clause is emitted, because the atom types of the data file may be redundant; element symbols are owned by the user-provided data/framework. Every dump carries `modify sort id`, so the row order is the atom id order
- Per-task file naming: the generated input script shares the dump output basename (`<project>_<profile>.in`, `<project>_force.in` for ff_gradient, e.g. `h2o_ff_npt.in`), so different tasks placed in the same directory never overwrite each other; the user template `<project>.in` stays untouched. The run script and the LAMMPS log follow the same basename (`<basename>.sh`/`<basename>.sbatch`/`<basename>.log`)
- `<project>.xyz` (without the profile suffix) is reserved for the framework-provided initial structure file (element + coordinates) used by read_dump templates; the skill never writes it
- For the internal-iteration profiles the execution commands of the user template body (thermo/dump/velocity/timestep/fix/run/minimize/...) are dropped so the generated sections stay the single source of execution control; force-field related custom content is preserved
- MD initial velocities: the system `velocities` (Bohr/au) are converted and written into the data file Velocities section when provided; otherwise `velocity all create <temperature> <md_seed>` is emitted
- A single point reads the files it printed for itself: energy from `<title>.energy`, forces from the `<title>.force` dump sorted by atom id (the id column is optional), stress from `<title>.stress`; its log is read for diagnosis only, because the thermo table quantizes to eight significant digits and a missing result file is a failed step rather than a zero. The internal-iteration profiles keep the log as their energy channel (`<project>_<profile>.log`, `<project>_force.log` for ff_gradient; table format with PotEng/TotEng columns, or multi-style `PotEng = ...` entries), and their final structure comes from the `<project>_<profile>.xyz` write_dump output with the MD profiles' final velocities from `<project>_<profile>.cond`
- Every LAMMPS profile returns native numbers labelled as what they are: `gradient` is dU/dR, negated by the parser, and the run's unit set travels in `_native_units`. The framework does the unit scaling from this skill's manifest and imports nothing from the skill. `.esfp` is the exception: atomic units for Molcas. `.qmmm`, its auxiliary gradients and `.grad` stay native.
- Missing `<project>.data` in the launch directory raises a clear validation error
- The calculation directory (calcdir) is fixed per backend instance
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization: `LammpsBackend(config, execution_id='tmp_LAMMPS')` -> `scratch/tmp_LAMMPS`
- lammps_skill.agent is the CLI entry point for subprocess execution
- With `python_enabled=false` (the default) both local and slurm execution run via subprocess; with `python_enabled=true` and `execution_mode=local` the calculation runs inside a LAMMPS instance held by a service process this backend started
