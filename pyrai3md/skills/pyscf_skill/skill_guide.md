# Skill Guide: pyscf_skill

## Overview

Use pyscf_skill to perform quantum chemistry calculations through PySCF. This skill provides a configuration-driven, profile-based interface for DFT, HF, MP2, TDDFT, and TDA computations.

## Setup Requirements

- PySCF installed and importable in the current Python environment
- Optional: `pip install git+https://github.com/pyscf/properties` for property calculations
- Set `pyscf_path` in config: `'import'` to use the current environment, or an explicit path
- Note: This is NOT the MLAtom pyscf module

## Supported Profiles

| Profile | Capability | Step Chain |
|---------|-----------|------------|
| hf_energy | energy | molecule, reference, results |
| hf_gradient | gradient | molecule, reference, gradient, results |
| hf_hessian | frequency | molecule, reference, hessian, results |
| hf_opt_min | geometry_optimization | molecule, reference, optimize, results |
| hf_opt_ts | geometry_optimization | molecule, reference, optimize_ts, results |
| hf_mo | analysis | molecule, reference, mo_analysis, results |
| hf_esp | analysis | molecule, reference, esp_analysis, results |
| hf_population | analysis | molecule, reference, population_analysis, results |
| hf_ir | ir_spectrum | molecule, reference, ir_analysis, results |
| hf_nmr | nmr_spectrum | molecule, reference, nmr_analysis, results |
| dft_energy | energy | molecule, reference, results |
| dft_gradient | gradient | molecule, reference, gradient, results |
| dft_hessian | frequency | molecule, reference, hessian, results |
| dft_opt_min | geometry_optimization | molecule, reference, optimize, results |
| dft_opt_ts | geometry_optimization | molecule, reference, optimize_ts, results |
| dft_mo | analysis | molecule, reference, mo_analysis, results |
| dft_esp | analysis | molecule, reference, esp_analysis, results |
| dft_population | analysis | molecule, reference, population_analysis, results |
| dft_ir | ir_spectrum | molecule, reference, ir_analysis, results |
| dft_nmr | nmr_spectrum | molecule, reference, nmr_analysis, results |
| mp2_energy | energy | molecule, reference, mp2, results |
| mp2_gradient | gradient | molecule, reference, mp2, gradient, results |
| mp2_mo | analysis | molecule, reference, mp2, mo_analysis, results |
| mp2_esp | analysis | molecule, reference, mp2, esp_analysis, results |
| mp2_population | analysis | molecule, reference, mp2, population_analysis, results |
| tddft_energy | energy | molecule, reference, tddft, results |
| tddft_gradient | gradient | molecule, reference, tddft, gradient, results |
| tddft_opt_min | geometry_optimization | molecule, reference, tddft, optimize, results |
| tddft_uv_vis | uv_spectrum | molecule, reference, tddft, uv_analysis, results |
| tddft_mo | analysis | molecule, reference, tddft, mo_analysis, results |
| tddft_esp | analysis | molecule, reference, tddft, esp_analysis, results |
| tddft_population | analysis | molecule, reference, tddft, population_analysis, results |
| tda_energy | energy | molecule, reference, tda, results |
| tda_gradient | gradient | molecule, reference, tda, gradient, results |
| tda_opt_min | geometry_optimization | molecule, reference, tda, optimize, results |
| tda_uv_vis | uv_spectrum | molecule, reference, tda, uv_analysis, results |
| tda_mo | analysis | molecule, reference, tda, mo_analysis, results |
| tda_esp | analysis | molecule, reference, tda, esp_analysis, results |
| tda_population | analysis | molecule, reference, tda, population_analysis, results |

## Workflow

1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Create Configuration File**: Prepare a JSON config file with system info, profile, method, basis set, and functional
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m pyscf_skill.agent config.json [local|slurm]`
   - **Python API**: `from pyscf_skill.backend import PySCFBackend`
4. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods

pyscf_skill provides a self-contained query module that reads its own manifest.yaml:

```python
# From the skill's parent directory:
from pyscf_skill import query

# List all available profiles
print(query.list_profiles())
# ['hf_energy', 'dft_energy', 'mp2_energy', 'tddft_energy', ...]

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('dft_energy')
# {'capability': 'energy', 'method': 'dft'}

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient', 'frequency', 'geometry_optimization', ...]

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API

```python
# From the skill's parent directory:
from pyscf_skill.backend import PySCFBackend

# Initialize backend
backend = PySCFBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = PySCFBackend(config_dict, execution_id='tmp_PYSCF')  # -> scratch/tmp_PYSCF

# Or with explicit full calcdir path:
backend = PySCFBackend(config_dict, calcdir='/path/to/calcdir')

# Execute calculation
results = backend.local_run()   # local subprocess
results = backend.slurm_run()   # SLURM subprocess

# For multi-step dynamics, reuse the same backend instance:
for step in trajectory:
    results = backend.calculate(atoms, coordinates, point_charges)

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
- `state_spin`: List of spin values for each state
- `profile`: Calculation profile (e.g. 'dft_energy')
- `method`: Electronic structure method ('hf', 'dft', 'mp2', 'tddft', 'tda')
- `functional`: DFT functional (e.g. 'pbe0', 'b3lyp')
- `basis`: Basis set (e.g. 'cc-pvdz', '6-31g*')
- `solvent`: Solvent model ('gas', 'pcm', 'cosmo')
- `target_states`: States for gradient/excited-state calculations
- `execution_mode`: 'local' or 'slurm'
- `pyscf_project`: Project name
- `pyscf_path`: PySCF path ('import' or explicit path)
- `scratch_path`: Working directory for scratch files
- `verbose`: Verbosity level
- `thread`: OpenMP threads for the in-process run. It is written into
  `OMP_NUM_THREADS` the first time the process needs the variable; a later
  term that asks for something else keeps the latched value and says so on
  stderr (#302). `pyscf_thread` is the name the framework forwards it under,
  and a written pair folds to its larger value (#306)
- `keep_tmp`: 0 removes the calculation directory once the results are
  collected, 1 keeps the log, checkpoint and molden files (#306)

## Important Notes

- PySCF is a Python-native library: no text input files or external executables are needed
- The builder constructs PySCF Mole and mean-field objects directly
- Coordinates are passed in Angstrom in the config
- The profile determines the complete computation workflow (step chain)
- Each profile enforces a specific method (e.g. 'dft_energy' requires method='dft')
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization: `PySCFBackend(config, execution_id='tmp_PYSCF')` -> `scratch/tmp_PYSCF`
- pyscf_skill.agent is the CLI entry point for subprocess execution
- Python API: directly obtain `PySCFBackend` and call `local_run()` / `slurm_run()` / `calculate()`
- Both local and slurm execution run via subprocess
