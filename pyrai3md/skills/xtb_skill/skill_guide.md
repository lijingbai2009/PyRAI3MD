# Skill Guide: xtb_skill

## Overview
Use xtb_skill to perform semiempirical quantum chemistry and force field calculations through xTB. This skill provides a configuration-driven, profile-based interface for GFN2-xTB, GFN1-xTB, GFN0-xTB, and GFN-FF single point energy and gradient computations.

## Setup Requirements
- xTB installed (https://github.com/grimme-lab/xtb)
- Either set `xtb_path` in config to the xTB installation directory (expects `$xtb_path/bin/xtb`), or point `xtb_path` directly to the xTB binary, or leave it empty to search `xtb` in PATH
- Parallelization is controlled by `thread` (OMP_NUM_THREADS; the framework
  spells it `xtb_thread`), memory by `xtb_mem` (MB)

## Supported Profiles

| Profile | Capability | Method | xTB CLI |
|---------|-----------|--------|---------|
| gfn2_energy | Single point energy | GFN2-xTB | `xtb --gfn 2` |
| gfn2_gradient | Energy + analytical gradient | GFN2-xTB | `xtb --gfn 2 --grad` |
| gfn1_energy | Single point energy | GFN1-xTB | `xtb --gfn 1` |
| gfn1_gradient | Energy + analytical gradient | GFN1-xTB | `xtb --gfn 1 --grad` |
| gfn0_energy | Single point energy | GFN0-xTB | `xtb --gfn 0` |
| gfn0_gradient | Energy + analytical gradient | GFN0-xTB | `xtb --gfn 0 --grad` |
| gfn_ff_energy | Single point energy | GFN-FF | `xtb --gfnff` |
| gfn_ff_gradient | Energy + analytical gradient | GFN-FF | `xtb --gfnff --grad` |

## Roles
- **qm / qm2**: GFN2-xTB, GFN1-xTB, GFN0-xTB serve as quantum mechanics layers
- **mm**: GFN-FF serves as the molecular mechanics layer

## Workflow
1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Create Configuration File**: Prepare a JSON config file with system info, profile, and method
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m pyrai3md.skills.xtb_skill.agent config.json [local|slurm]`
   - **Python API**: `from pyrai3md.skills.xtb_skill import XtbBackend`
4. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods
xtb_skill provides a self-contained query module that reads its own manifest.yaml:
```python
# From the skill's parent directory:
from xtb_skill import query

# List all available profiles
print(query.list_profiles())
# ['gfn0_energy', 'gfn0_gradient', 'gfn1_energy', ...]

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('gfn2_energy')
# {'capability': 'energy', 'method': 'gfn2', 'steps': ['energy']}

# Get xTB step chain for a profile
steps = query.get_steps('gfn2_gradient')
# ['energy', 'gradient']

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient']

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API
```python
# From the skill's parent directory:
from xtb_skill.backend import XtbBackend

# Initialize backend (default calcdir: tmp_XTB)
backend = XtbBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = XtbBackend(config_dict, execution_id='tmp_mid_qm2_task1_XTB')  # -> scratch/tmp_mid_qm2_task1_XTB

# Or with explicit full calcdir path:
backend = XtbBackend(config_dict, calcdir='/path/to/calcdir')

# Execute calculation (local or slurm is selected by execution_mode in config)
results = backend.calculate(atoms, coordinates)

# With background point charges (qm/qm2 embedding):
results = backend.calculate(atoms, coordinates, point_charges)

# For multi-step dynamics, reuse the same backend instance:
for step in trajectory:
    results = backend.calculate(atoms, coordinates)
    # calcdir remains fixed, GFN-FF topology (gfnff_topo) is reused
```

## Configuration Schema
Key fields in the JSON config:
- `atoms`: List of atom symbols
- `coordinates`: List of [x, y, z] coordinates in Angstrom
- `point_charges`: Optional background charges [charge, x, y, z] (mutually exclusive with cell/pbc)
- `cell`: Unit cell vectors in Angstrom (only with gfn_ff)
- `pbc`: Periodic boundary flags [x, y, z], e.g. [1, 1, 1] (only with gfn_ff)
- `charges`: Total charge (default: 0)
- `multiplicity`: Spin multiplicity (default: 1)
- `profile`: Calculation profile (e.g. 'gfn2_energy')
- `method`: Method name ('gfn2', 'gfn1', 'gfn0', or 'gfn_ff')
- `execution_mode`: 'local' or 'slurm'
- `xtb_project`: Project name for xTB files (default: 'XTB')
- `xtb_path`: Path to xTB installation directory or binary ('' to search PATH)
- `scratch_path`: File placement directory for calculations
- `xtb_mem`: Memory in MB
- `thread`: Number of OpenMP threads (xtb has no MPI arm, so the old
  process-count name for it is retired, #306)

## Results
Result dict keys follow the Properties schema field names:
- `energy`: List of one total energy in Hartree (ground state method)
- `gradient`: Gradient array of shape (1, natom, 3) in Hartree/Bohr
- `computed_charges`: Atomic partial charges of shape (1, natom) from `charges` (GFN-x) or `gfnff_charges` (GFN-FF); the key is omitted when xTB wrote no charge file

## Important Notes
- Geometry is written to `<project>.xyz` (or Turbomole coord format when cell/pbc is set for gfn_ff)
- Background point charges are embedded via the `$embedding` block and a `<project>.pc` file (coordinates in Bohr); embedding is mutually exclusive with periodic boundary conditions
- Periodic boundary conditions are only supported by GFN-FF
- Energy and gradient are parsed from `<project>.engrad`; failure to parse raises an error pointing to `<project>.out` for diagnostics
- A user-provided `<project>.inp` template in the current working directory is merged into the generated input
- The calculation directory (calcdir) is fixed per backend instance, enabling GFN-FF topology reuse across dynamics steps
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization: `XtbBackend(config, execution_id='tmp_XTB')` -> `scratch/tmp_XTB`
- xtb_skill.agent is the CLI entry point for subprocess execution
- Both local and slurm execution run via subprocess
