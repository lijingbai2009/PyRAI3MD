# Skill Guide: bagel_skill

## Overview

Use bagel_skill to perform multiconfigurational quantum chemistry calculations through BAGEL. This skill provides a configuration-driven, profile-based interface for CASSCF and XMS-CASPT2 computations.

## Setup Requirements

- BAGEL installed and accessible via the `BAGEL` environment variable
- Required environment variables:
  - `BAGEL`: BAGEL installation root
  - `BLAS`, `LAPACK`, `BOOST`, `MPI`, `MKL`, `ARCH`: library paths used by BAGEL
- Set `bagel_path` in config to the BAGEL installation directory
- Ensure the BAGEL executable is available at `$BAGEL/bin/BAGEL`

## Supported Profiles

| Profile | Capability | BAGEL Steps |
|---------|-----------|-------------|
| casscf_energy | Single point CASSCF energy | molecule, casscf, save_ref |
| casscf_gradient | CASSCF analytical gradient | molecule, casscf, forces |
| casscf_nac | CASSCF non-adiabatic coupling | molecule, casscf, forces(nacme) |
| casscf_frequency | CASSCF vibrational frequency | molecule, casscf, hessian |
| xmscaspt2_energy | XMS-CASPT2 single point energy | molecule, load_ref, caspt2 |
| xmscaspt2_gradient | XMS-CASPT2 analytical gradient | molecule, load_ref, forces(caspt2) |
| xmscaspt2_nac | XMS-CASPT2 non-adiabatic coupling | molecule, load_ref, forces(caspt2/nacme) |
| xmscaspt2_frequency | XMS-CASPT2 vibrational frequency | molecule, load_ref, hessian(caspt2) |

## Workflow

1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Create Configuration File**: Prepare a JSON config file with system info, profile, method, basis set, and active space settings
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m bagel_skill.backend config.json [local|slurm]`
   - **Python API**: `from bagel_skill.backend import BagelBackend`
4. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods

bagel_skill provides a self-contained query module that reads its own manifest.yaml:

```python
# From the skill's parent directory:
from bagel_skill import query

# List all available profiles
print(query.list_profiles())
# ['casscf_energy', 'casscf_gradient', 'casscf_nac', ...]

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('casscf_energy')
# {'capability': 'energy', 'method': 'casscf'}

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient', 'nac', 'frequency']

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API

```python
# From the skill's parent directory:
from bagel_skill.backend import BagelBackend

# Initialize backend (default calcdir: tmp_BAGEL)
backend = BagelBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = BagelBackend(config_dict, execution_id='tmp_BAGEL')  # -> scratch/tmp_BAGEL
backend = BagelBackend(config_dict, execution_id='tmp_mid_qm2_task1_BAGEL')  # -> scratch/tmp_mid_qm2_task1_BAGEL

# Or with explicit full calcdir path:
backend = BagelBackend(config_dict, calcdir='/path/to/calcdir')

# Execute calculation
results = backend.local_run()   # local subprocess
results = backend.slurm_run()   # SLURM subprocess

# For multi-step dynamics, reuse the same backend instance:
for step in trajectory:
    results = backend.calculate(atoms, coordinates, point_charges)
    # calcdir remains fixed, archive files are reused as initial guesses

# Optional per-step target states (e.g. chosen by a NAMD workflow);
# None keeps the target_states fixed at initialization:
results = backend.calculate(atoms, coordinates, target_states=[1])
```

## Configuration Schema

Key fields in the JSON config:
- `atoms`: List of atom symbols
- `coordinates`: List of [x, y, z] coordinates in Angstrom
- `point_charges`: Optional background charges [charge, x, y, z]
- `charges`: Total charge (default: 0)
- `multiplicity`: Spin multiplicity (default: 1)
- `nstate`: Number of electronic states (default: 1)
- `profile`: Calculation profile (e.g. 'casscf_energy')
- `method`: Electronic structure method ('casscf' or 'xmscaspt2')
- `basis`: Basis set (e.g. 'cc-pvdz')
- `df_basis`: Density-fitting basis set (one of the seven *-jkfit sets; leave empty to auto-match `basis`, defaulting to 'svp-jkfit' when no match exists)
- `nact_elec`: Number of active electrons
- `nact_orb`: Number of active orbitals
- `nclosed`: Number of closed-shell orbitals (BAGEL-specific)
- `active`: List of active orbital indices (BAGEL-specific)
- `target_states`: States for gradient/frequency calculations
- `target_nac_pair`: Non-adiabatic coupling pairs
- `execution_mode`: 'local' or 'slurm'
- `mpiexe`: MPI launcher executable (default 'mpirun', #305; 'no'/'none' runs
  BAGEL directly without a launcher)
- `nproc`: MPI rank count (`bagel_nproc` is the framework spelling); fallback
  for $SLURM_NTASKS in local mode
- `thread`: OpenMP threads per rank, exported as `OMP_NUM_THREADS`/
  `MKL_NUM_THREADS`/`BAGEL_NUM_THREADS` in both modes; `bagel_thread` is the
  framework spelling. Ranks and threads are separate keywords: neither one
  stands in for the other any more (#305, #306)
- `bagel_project`: Project name for archive files
- `scratch_path`: File placement directory for calculations

## Important Notes

- BAGEL input is in JSON format, not a text module chain
- Coordinates are passed in Angstrom; the molecule block sets "angstrom": "true" so BAGEL reads them as-is
- BAGEL manages orbitals via `.archive` files instead of RasOrb/StrOrb/JobIph
- The `save_ref` step writes a `*_converged.archive` file; later steps load it as `*_guess.archive`
- Frequency calculations use the BAGEL `hessian` block:
  - CASSCF level: `method` array contains the CASSCF definition block
  - CASPT2 level: `method` array contains the CASPT2/smith definition block
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization, not auto-detected: `BagelBackend(config, execution_id='tmp_BAGEL')` -> `scratch/tmp_BAGEL`
- bagel_skill.backend is the CLI entry point for subprocess execution
- Python API: directly obtain `BagelBackend` and call `local_run()` / `slurm_run()` / `calculate()`
- Both local and slurm execution run via subprocess
