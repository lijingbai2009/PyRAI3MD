# Skill Guide: cp2k_skill

## Overview
Use cp2k_skill to perform DFT and TDDFT calculations through CP2K (Quickstep). This skill provides a configuration-driven, profile-based interface for ground-state and excited-state computations, including energies, gradients, transition dipoles, geometry optimizations, and standalone ab initio molecular dynamics. Periodic and molecular systems are both supported.

## Setup Requirements
- CP2K installed (tested against the cp2k_demo toolchain builds)
- Set `cp2k_path` in config to the CP2K installation directory (CP2K_HOME)
- The executable defaults to `$CP2K_HOME/exe/local/cp2k.psmp`; override with `cp2k_exe`
- Set `nproc` to the number of MPI ranks and `thread` for OpenMP threads
  (the framework can also steer them as `cp2k_nproc`/`cp2k_thread`; a skill
  keeps the larger of a written pair, #306)
- Optional `setup_script` is sourced before execution (e.g. the toolchain `install/setup`), and `shell` carries extra commands such as `module load`

## Supported Profiles

| Profile | Capability | CP2K Input |
|---------|-----------|------------|
| dft_energy | Ground-state DFT single point | `RUN_TYPE ENERGY` |
| dft_gradient | DFT energy + gradient | `RUN_TYPE ENERGY_FORCE` + `&PRINT &FORCES` |
| dft_opt_min | DFT geometry optimization (minimum) | `RUN_TYPE GEO_OPT` + `&MOTION &GEO_OPT` |
| dft_md | Standalone DFT molecular dynamics | `RUN_TYPE MD` + `&MOTION &MD` |
| tddft_energy | TDDFPT excited-state energies | `&PROPERTIES &TDDFPT` |
| tddft_gradient | TDDFPT excited-state gradients | TDDFPT + `&PRINT &FORCES LIST` (tdfrc) |
| tddft_transition_dipole | TDDFPT transition dipoles | TDDFPT excitation table |
| tddft_opt_min | TDDFPT excited-state optimization | `GEO_OPT` + `&OPTIMIZE STATE` |
| tddft_md | Standalone TDDFPT molecular dynamics | `RUN_TYPE MD` + TDDFPT |

Only `dft` and `tddft` methods are supported. Supported functionals: `pbe` and `pbe0` (anything else raises `Unsupported CP2K functional`).

In framework-driven dynamics (driver=main) MD runs through the `gradient` profile one step at a time; the standalone `*_md` profiles are for driver=skill single-shot runs.

## Workflow
1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Create Configuration File**: Prepare a JSON config file with system info, profile, method, functional, and cutoffs
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m cp2k_skill.agent config.json [local|slurm]`
   - **Python API**: `from cp2k_skill.backend import Cp2kBackend`
4. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods
cp2k_skill provides a self-contained query module that reads its own manifest.yaml:
```python
# From the skill's parent directory:
from cp2k_skill import query

# List all available profiles
print(query.list_profiles())
# ['dft_energy', 'dft_gradient', 'dft_md', ...]

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('tddft_gradient')
# {'capability': 'gradient', 'method': 'tddft', 'steps': ['energy', 'tddft', 'gradient']}

# Get CP2K input step chain for a profile
steps = query.get_steps('tddft_gradient')
# ['energy', 'tddft', 'gradient']

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient', 'transition_dipole', ...]

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API
```python
# From the skill's parent directory:
from cp2k_skill.backend import Cp2kBackend

# Initialize backend (default calcdir: tmp_CP2K)
backend = Cp2kBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = Cp2kBackend(config_dict, execution_id='tmp_CP2K')  # -> scratch/tmp_CP2K
backend = Cp2kBackend(config_dict, execution_id='tmp_mid_qm2_task1_CP2K')  # -> scratch/tmp_mid_qm2_task1_CP2K

# Or with explicit full calcdir path:
backend = Cp2kBackend(config_dict, calcdir='/path/to/calcdir')

# Execute calculation
results = backend.calculate(atoms, coordinates)

# For multi-step dynamics, reuse the same backend instance:
for step in trajectory:
    results = backend.calculate(atoms, coordinates, point_charges)
    # calcdir remains fixed across steps

# Optional per-step target states (e.g. chosen by a NAMD workflow);
# None keeps the target_states fixed at initialization:
results = backend.calculate(atoms, coordinates, target_states=[1])
```

## Configuration Schema
Key fields in the JSON config:
- `atoms`: List of atom symbols
- `coordinates`: List of [x, y, z] coordinates in Angstrom
- `charges`: Total charge (default: 0)
- `multiplicity`: Spin multiplicity (default: 1)
- `nstate`: Number of electronic states (default: 1)
- `profile`: Calculation profile (e.g. 'tddft_gradient')
- `method`: Electronic structure method ('dft' or 'tddft')
- `functional`: Exchange-correlation functional, 'pbe' or 'pbe0' (default: 'pbe'); pbe0 maps to `&XC_FUNCTIONAL &PBE SCALE_X 0.75` + `&HF FRACTION 0.25`
- `cutoff`: Plane-wave cutoff in Ry -> `&MGRID CUTOFF` (default: 400)
- `ref_cutoff`: Refined grid cutoff in Ry -> `&MGRID REL_CUTOFF` (default: 50)
- `basis_file` / `potential_file` / `auxiliary_basis_file`: Global data file lists (defaults: `BASIS_MOLOPT`, `POTENTIAL`)
- `basis` / `potential` / `auxiliary_basis`: Element-level dicts overriding the per-kind defaults; normalization fills missing elements with `DZVP-MOLOPT-SR-GTH-q<N>` / `GTH-PBE` (and `admm-dzp-q<N>` when ADMM is active)
- `dispersion`: 'd3'/'d3bj' -> `&VDW_POTENTIAL` with `REFERENCE_FUNCTIONAL` following the active functional; '' disables (default: '')
- `lattice_vector` / `pbc`: Periodic cell; without a cell the skill builds a vacuum box with the WAVELET Poisson solver
- `dipole_correction`: Slab dipole correction (`&DFT SURF_DIP .TRUE.` + `SURF_DIP_DIR Z`; default false). Requires a periodic, z-normal slab cell (`pbc: [1,1,1]` with vacuum along z); refused for non-periodic runs, tilted cells, and `stress_method: num` with `stress_scope: full`
- `tddft_nstates`: Number of TDDFPT roots (derived from `nstate`/`target_states` when unset)
- `tddft_dipole_form`: Transition dipole gauge, 'length' or 'velocity' (default: 'length')
- `md_nstep` / `md_timestep` / `md_ensemble` / `temperature`: Standalone MD settings (NVT uses the CSVR thermostat)
- `target_states`: States to compute gradients for
- `stress`: Request the whole-system stress tensor (default: false); the
  result arrives in `stress_tensor` as `(nstate, 3, 3)`, tensile-positive,
  in GPa
- `stress_method`: `analytical` (default; CP2K's `STRESS_TENSOR
  ANALYTICAL`, ground state -- the route to prefer for DFT, since the
  engine already reports this tensor) or `num` -- the skill's own external
  strain finite difference, which asks CP2K for no tensor at all and is
  therefore the only route available to a TDDFT excited state (CP2K
  provides neither an analytical nor a numerical stress tensor for a TDDFT
  run)
- `stress_scope` / `stress_delta`: for `num`, which strain directions to
  sample (`full` = six, 12 perturbed single points; `iso` = uniform
  dilation, 2 points reported on the diagonal) and the dimensionless step
  (default 1e-3; start at 1e-4~1e-3 and compare delta/2, delta, 2*delta)
- `multitask` / `multipool` / `remotetask`: the `&cp2k` dispatch keywords,
  read by the skill to plan the `num` fan-out. `multitask: true` spreads
  the strain points over a pool of `multipool` worker processes (default
  1); under `execution_mode: slurm` (`remotetask: true` asks for it) each
  point is its own blocking `sbatch` job, so the pool size is the number
  of concurrent jobs. The node budget is
  `multipool x nproc x thread`; the skill warns when it exceeds the
  detected CPU count.
- `execution_mode`: 'local' or 'slurm'
- `cp2k_project`: Project name for CP2K files (default: 'cp2k_calc')
- `cp2k_path`: CP2K installation directory (CP2K_HOME)
- `cp2k_exe`: Executable path (default: `$CP2K_HOME/exe/local/cp2k.psmp`)
- `nproc`: Number of MPI ranks (default: 1), `cp2k_nproc` its framework spelling
- `thread`: OpenMP threads (OMP_NUM_THREADS), `cp2k_thread` its framework spelling
- `setup_script`: Optional toolchain setup script sourced before execution
- `scratch_path`: File placement directory for calculations ('AUTO' for automatic scratch selection)

## Results
Result dict keys follow the Properties schema field names:
- `energy`: State-resolved energies (nstate,) in Hartree
- `gradient`: State-resolved gradient (nstate, natom, 3) in Hartree/Bohr, computed as the negative of the CP2K atomic forces; untargeted states stay zero
- `transition_dipole`: Transition dipole moments (npair, 3) extracted in `transition_pair` order; only ground-to-state pairs carry computed values
- `osc_strength`: Oscillator strengths (npair,) in `transition_pair` order
- `excitation`: Excitation energies (nstate,) in eV (state-resolved)
- `fin_coord`: Final structure (natom, 3) in Angstrom returned through the framework fin_coord channel (optimize and standalone MD profiles); the framework updates the system coordinates and drops the key
- `fin_cell`: The last frame's `Lattice=` comment in `{project}-pos-1.xyz`, flattened to nine lattice components (Angstrom), returned by the same profiles. It equals the input cell for the fixed-cell runs this skill performs today -- no profile emits `CELL_OPT`/`CELL_REF` yet -- and the framework derives `cell`, `lattice_vector` and `volume` from it, so a cell optimization needs no new channel when it arrives
- `opt_converge`: Optimization convergence flag (optimize profiles); False means the internal optimization stopped before convergence

## Important Notes
- Coordinates are written as `{project}.xyz` and referenced via `&TOPOLOGY COORD_FILE_NAME` in Angstrom (CP2K native unit)
- CP2K prints atomic forces; the skill negates them to produce framework gradients (gradient = -force), with all conversions centralized in the parser
- Ground-state gradients come from `{project}-force-1_0.xyz` (`force_unit hartree/bohr`); excited-state gradients come from `{project}-td_forces-1_0.tdfrc` blocks in the sorted `target_states` order — a block-count mismatch fails fast
- Excitation table rows (`TDDFPT|`) carry the excitation energy (eV), transition dipole components, and oscillator strength; excited-state energies are `E(S0) + excitation` converted to Hartree
- When `auxiliary_basis` is set, ADMM is enabled (`&AUXILIARY_DENSITY_MATRIX_METHOD BASIS_PROJECTION` + `BASIS_SET AUX_FIT` per kind); TDDFPT runs add `ADMM_PURIFICATION_METHOD NONE`
- Dispersion `REFERENCE_FUNCTIONAL` always follows the active functional (pbe0+d3 writes `REFERENCE_FUNCTIONAL PBE0`)
- MD runs switch `&QS EXTRAPOLATION` to ASPC for density extrapolation between steps
- The run script mirrors cp2k_demo/run.sh: `export CP2K_HOME/CP2K_EXE/OMP_NUM_THREADS`, optional `source setup_script`, then `mpirun -n N $CP2K_EXE -i $CP2K_PROJECT.inp -o $CP2K_PROJECT.log 2>&1` (stderr merged into the log so input errors are never lost)
- A non-zero CP2K return code raises immediately instead of masking the failure with zero-filled results
- CP2K prints its stress tensor compressive-positive; the parser negates it into the framework's tensile-positive convention and normalizes the printed unit label to GPa. The `num` strain difference is tensile-positive as it stands and is not negated
- `num` asks CP2K for ordinary single points only, so it is the route for an excited-state tensor: the engine aborts when a stress tensor is requested from a TDDFT run, and offers no numerical TDDFT stress either, so `analytical` under `method: tddft` is refused by validation rather than paid for in an engine run
- The `num` points are never relaxed and keep the fractional coordinates: the result is the instantaneous stress of the configuration handed in, not a relaxed-cell stress. One batch serves every state the run solved, so three states cost the same twelve points as one
- States are treated as the energy-sorted adiabatic roots of each point. A point that drops a root, moves a root past half the smallest central gap, or a central spectrum with a degenerate root fails the stress by name; character-based state tracking (NTO/transition-density overlap) is not implemented
- `num` needs full 3D periodicity and a non-singular `lattice_vector`; the strain points are written beside the central calculation directory under their own `{project}` stems, and their directories follow `keep_tmp`
- NAC/SOC/frequency are not supported in this phase
- WORKDIR 'AUTO' mode uses `/srv/tmp` if available, otherwise `/tmp`, for fast I/O
- The calculation directory (calcdir) is fixed per backend instance
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization: `Cp2kBackend(config, execution_id='tmp_CP2K')` -> `scratch/tmp_CP2K`
- cp2k_skill.agent is the CLI entry point for subprocess execution
- Both local and slurm execution run via subprocess
