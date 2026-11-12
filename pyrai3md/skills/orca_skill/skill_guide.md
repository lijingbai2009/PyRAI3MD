# Skill Guide: orca_skill

## Overview
Use orca_skill to perform DFT and TDDFT quantum chemistry calculations through ORCA. This skill provides a configuration-driven, profile-based interface for ground-state and excited-state computations, including energies, gradients, dipoles, transition dipoles, spin-orbit couplings, nonadiabatic couplings (S0-SN only), geometry optimizations, and vibrational frequencies.

## Setup Requirements
- ORCA 5.0+ installed and accessible
- Set `orca_path` in config to the ORCA installation directory
- Ensure the `orca` executable is available at `$ORCA/orca`
- For parallel ORCA builds, set `mpi` to the MPI library path

## Supported Profiles

| Profile | Capability | ORCA Keywords |
|---------|-----------|---------------|
| dft_energy | Ground-state DFT single point | `! functional basis TightSCF` |
| dft_gradient | DFT analytical gradient | `EnGrad` |
| dft_dipole | DFT dipole moment (SP) | dipole from output |
| dft_opt_min | DFT geometry optimization (minimum) | `Opt` |
| dft_opt_ts | DFT transition state optimization | `OptTS` |
| dft_frequency | DFT vibrational frequencies/Hessian | `AnFreq` or `NumFreq` |
| tddft_energy | TDDFT excited-state energies | `%tddft nroots` |
| tddft_gradient | TDDFT excited-state gradients | `EnGrad` + `irootlist` |
| tddft_transition_dipole | TDDFT transition dipoles (SP) | absorption spectrum |
| tddft_nac | Nonadiabatic couplings (S0-SN only) | `NACME true` inside `%tddft`; not written when the run embeds point charges (ORCA aborts that combination, `main_input_check.cpp:10261`) |
| tddft_soc | Spin-orbit coupling | `Triplets true` + `DoSOC true` in `%tddft` |
| tddft_opt_min | TDDFT excited-state optimization (minimum) | `Opt` |
| tddft_opt_ts | TDDFT transition state optimization | `OptTS` |
| tddft_frequency | TDDFT vibrational frequencies | `NumFreq` (always numerical) |

Only `dft` and `tddft` methods are supported. The legacy `sf_tddft` method is not supported.

All profiles share the accuracy/acceleration keywords on the main
keyword line: `RIJCOSX` + auxiliary basis (density fitting) and the
Grimme dispersion keyword follow the basis set, while the grid level
rides on a dedicated `! DEFGRID<n>` line (see Configuration Schema).

## Workflow
1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Create Configuration File**: Prepare a JSON config file with system info, profile, method, functional, and basis set
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m orca_skill.agent config.json [local|slurm]`
   - **Python API**: `from orca_skill.backend import OrcaBackend`
4. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods
orca_skill provides a self-contained query module that reads its own manifest.yaml:
```python
# From the skill's parent directory:
from orca_skill import query

# List all available profiles
print(query.list_profiles())
# ['dft_dipole', 'dft_energy', 'dft_frequency', ...]

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('tddft_gradient')
# {'capability': 'gradient', 'method': 'tddft', 'steps': ['energy', 'tddft', 'gradient']}

# Get ORCA input step chain for a profile
steps = query.get_steps('tddft_soc')
# ['energy', 'tddft', 'soc']

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient', 'dipole', ...]

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API
```python
# From the skill's parent directory:
from orca_skill.backend import OrcaBackend

# Initialize backend (default calcdir: tmp_ORCA)
backend = OrcaBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = OrcaBackend(config_dict, execution_id='tmp_ORCA')  # -> scratch/tmp_ORCA
backend = OrcaBackend(config_dict, execution_id='tmp_mid_qm2_task1_ORCA')  # -> scratch/tmp_mid_qm2_task1_ORCA

# Or with explicit full calcdir path:
backend = OrcaBackend(config_dict, calcdir='/path/to/calcdir')

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
- `point_charges`: Optional background charges [charge, x, y, z]
- `charges`: Total charge (default: 0)
- `multiplicity`: Spin multiplicity (default: 1)
- `nstate`: Number of electronic states (default: 1)
- `state_spin`: Spin quantum number (2S) per state (needed for SOC with mixed singlet/triplet)
- `profile`: Calculation profile (e.g. 'tddft_gradient')
- `method`: Electronic structure method ('dft' or 'tddft')
- `functional`: DFT functional (default: 'pbe0')
- `basis`: Basis set (default: 'def2-svp')
- `density_fitting`: Density fitting switch (default: true); writes `RIJCOSX` plus the auxiliary basis after the basis set, false omits both
- `aux_basis`: Auxiliary basis for density fitting (default: 'def2/J')
- `dispersion_correction`: Grimme dispersion correction, 'd3'/'d3zero'/'d4' -> D3BJ/D3ZERO/D4 (default: 'd3'); '' or 'none' disables
- `grid_level`: Integration grid level, 1-3 -> dedicated `! DEFGRID1-3` line, 0 keeps the ORCA default (default: 2)
- `num_freq`: Frequency keyword switch for ground-state DFT (false = AnFreq analytical, true = NumFreq numerical); TDDFT frequency runs always use `NumFreq` since ORCA has no analytical TDDFT second derivatives
- `target_states`: States to compute gradients for (TDDFT)
- `target_nac_pair`: NAC state pairs; only pairs including state 0 are allowed
- `target_soc_pair`: SOC state pairs (usually derived from state_spin)
- `execution_mode`: 'local' or 'slurm'
- `orca_project`: Project name for ORCA files (default: 'ORCA')
- `orca_path`: ORCA installation directory
- `mpi`: MPI library path for parallel ORCA (optional)
- `scratch_path`: File placement directory for calculations ('AUTO' for automatic scratch selection, e.g. /srv/tmp)
- `orca_mem`: Memory per core in MB (%maxcore)
- `nproc`: Number of `%pal nprocs` ranks (`orca_nproc` is the framework
  spelling)
- `thread`: OpenMP threads per rank (`OMP_NUM_THREADS`), `orca_thread` in the
  framework spelling; a written pair folds to its larger value (#306)

## Results
Result dict keys follow the Properties schema field names:
- `energy`: State-resolved energies (nstate,) in Hartree; index 0 is the ground state and every entry shares one reference so it is comparable with `gradient[k]`. ORCA prints only the energy the computed gradient belongs to (identical in the log's `FINAL SINGLE POINT ENERGY` and the `.engrad` `$energy`, and an excited state whenever excited-state gradients were requested), so the reference is reconstructed from `property.txt` / the `CIS/TD-DFT TOTAL ENERGY` block rather than read off directly
- `gradient`: State-resolved gradient (nstate, natom, 3) in Hartree/Bohr; untargeted states stay zero
- `dipole`: Ground-state dipole moment (nstate, 3) in e*bohr
- `transition_dipole`: Transition dipole moments (npair, 3) extracted in `transition_pair` order; only ground-to-state pairs carry computed values
- `osc_strength`: Oscillator strengths (npair,) in `transition_pair` order
- `excitation`: Excitation energies (nstate,) in eV (state-resolved)
- `nac`: Nonadiabatic coupling vectors (nnac, natom, 3) in `target_nac_pair` order (S0-SN only)
- `soc`: Spin-orbit couplings (nsoc,) in `target_soc_pair` order, cm^-1
- `frequencies`: Harmonic vibrational frequencies in cm^-1 (frequency profiles)
- `hessian`: Cartesian Hessian matrix (frequency profiles)
- `fin_coord`: Final structure (natom, 3) in Angstrom returned through the framework fin_coord channel (optimize profiles); the framework updates the system coordinates and drops the key
- `opt_converge`: Optimization convergence flag (optimize profiles); False means the internal optimization stopped before convergence

## Important Notes
- Coordinates are inlined in the geometry block (`* xyz charge mult`) in Angstrom
- Background point charges are embedded through a side file: the input carries `%pointcharges "{project}.pc"` and that file, written beside the input, opens with the charge count followed by one `q x y z` row per charge (charge in atomic units, position in Angstrom). ORCA rejects the rows inline
- SOC calculations require a closed-shell singlet (RHF) reference (multiplicity 1); triplet target states enter through `Triplets true` + `DoSOC true` in the `%tddft` block; SOC values are returned in cm^-1 (ORCA native unit)
- NAC supports only couplings between S0 and excited states (SN); pairs between two excited states are rejected
- `tddft_nroots` is derived during normalization: max(excited singlets, triplets) from `state_spin` (pure-singlet runs keep nstate - 1); ORCA computes that many roots of each multiplicity with `Triplets true`, and the parser maps the combined root numbering back onto the framework state list
- Gradient roots are split between `irootlist` (singlet roots, root 0 = ground state, only there) and `trootlist` (triplet roots, 1-based), both ascending; the parser collects `CARTESIAN GRADIENT` blocks in log order and distributes them along the print order `irootlist[1:] + trootlist + [0]` (ground state requested) or `irootlist + trootlist` otherwise, with triplet entries carrying the state offset `nsing_root + root`; a block-count mismatch fails fast. `Triplets true` is added whenever the state list contains triplet roots, not only for SOC
- `num_freq` controls the ground-state frequency keyword: false -> `AnFreq`, true -> `NumFreq`; tddft_frequency always runs with `NumFreq` (no analytical TDDFT second derivatives in ORCA), and NACME is written inside the `%tddft` block (ORCA rejects it on the main keyword line)
- Every input carries `PrintBasis` on the keyword line and `Print[P_MOs] 1` in the `%scf` block; frequency runs write `{project}.hess` automatically (no `%output Print[P_Hessian]` block - ORCA rejects it)
- jobtype aliases `hess`, `hessian`, `freq`, `frequence` all resolve to frequency calculations
- WORKDIR 'AUTO' mode uses `/srv/tmp` if available, otherwise `/tmp`, for fast I/O
- The calculation directory (calcdir) is fixed per backend instance
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization: `OrcaBackend(config, execution_id='tmp_ORCA')` -> `scratch/tmp_ORCA`
- orca_skill.agent is the CLI entry point for subprocess execution
- Both local and slurm execution run via subprocess
