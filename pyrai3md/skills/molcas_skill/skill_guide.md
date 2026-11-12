# Skill Guide: molcas_skill

## Overview
Use molcas_skill to perform multiconfigurational quantum chemistry calculations through OpenMolcas. This skill provides a configuration-driven, profile-based interface for CASSCF and CASPT2 computations.

## Setup Requirements
- OpenMolcas installed and accessible via MOLCAS environment variable
- Set `molcas_path` in config to the OpenMolcas installation directory
- Ensure `pymolcas` is available at `$MOLCAS/pymolcas` (recent installs) or `$MOLCAS/bin/pymolcas` (older installs)

## Supported Profiles

| Profile | Capability | Molcas Modules |
|---------|-----------|----------------|
| casscf_energy | Single point CASSCF energy | GATEWAY, SEWARD, RASSCF |
| casscf_gradient | CASSCF analytical gradient | GATEWAY, SEWARD, RASSCF, ALASKA |
| casscf_nac | CASSCF analytical NAC | GATEWAY, SEWARD, RASSCF, ALASKA |
| casscf_properties | Free energy/gradient/NAC/SOC combination | GATEWAY, SEWARD, RASSCF, ALASKA, RASSI |
| casscf_geometry_optimization | Geometry optimization (single-state / MECI / MECP) | GATEWAY, SEWARD, RASSCF, SLAPAF |
| casscf_abinitio_dynamics | Ab initio MD (energy + gradient, no NAC/hopping) | GATEWAY, SEWARD, RASSCF, ALASKA, DYNAMIX |
| casscf_surface_hopping | Surface hopping dynamics | GATEWAY, SEWARD, RASSCF, SURFACEHOP, ALASKA, DYNAMIX |
| casscf_soc | Spin-orbit coupling | GATEWAY, SEWARD, RASSCF (dual), RASSI |
| casscf_absorption | Absorption spectrum (RASSI) | GATEWAY, SEWARD, RASSCF, RASSI |
| caspt2_energy | CASPT2 single point energy | GATEWAY, SEWARD, RASSCF, CASPT2 |
| caspt2_properties | CASPT2 energy/gradient/NAC/SOC | GATEWAY, SEWARD, RASSCF, CASPT2, ALASKA, RASSI |
| caspt2_absorption | CASPT2 absorption spectrum | GATEWAY, SEWARD, RASSCF, CASPT2, RASSI |
| casscf_frequency | CASSCF vibrational frequency | GATEWAY, SEWARD, RASSCF, MCKINLEY |
| caspt2_frequency | CASPT2 vibrational frequency | GATEWAY, SEWARD, RASSCF, CASPT2, MCKINLEY |

## Workflow
1. **Discover Available Methods**: Use the built-in query module to list profiles and capabilities
2. **Create Configuration File**: Prepare a JSON config file with system info, profile, method, basis set, and active space settings
3. **Execute Calculation**:
   - **CLI (subprocess)**: `python -m molcas_skill.agent config.json [local|slurm]`
   - **Python API**: `from molcas_skill.backend import MolcasBackend`
4. **Obtain Results**: Load the config file to find the calculation results

## Query Available Methods
molcas_skill provides a self-contained query module that reads its own manifest.yaml:
```python
# From the skill's parent directory:
from molcas_skill import query

# List all available profiles
print(query.list_profiles())
# ['casscf_absorption', 'casscf_energy', 'casscf_gradient', ...]

# Get all profiles with details
profiles = query.get_profiles()

# Get a specific profile
info = query.get_profile('casscf_energy')
# {'capability': 'energy', 'method': 'casscf', 'modules': ['GATEWAY', 'SEWARD', 'RASSCF']}

# Get Molcas module chain for a profile
modules = query.get_modules('casscf_soc')
# ['GATEWAY', 'SEWARD', 'RASSCF', 'RASSI']

# Get all capabilities
caps = query.get_capabilities()
# ['energy', 'gradient', 'geometry_optimization', ...]

# Get full registry (standardized format for framework query aggregator)
registry = query.get_registry()
```

## Python API
```python
# From the skill's parent directory:
from molcas_skill.backend import MolcasBackend

# Initialize backend (default calcdir: tmp_MOLCAS)
backend = MolcasBackend(config_dict)

# With execution namespace (caller-controlled, for multiple backends):
backend = MolcasBackend(config_dict, execution_id='tmp_MOLCAS')  # -> scratch/tmp_MOLCAS
backend = MolcasBackend(config_dict, execution_id='tmp_mid_qm2_task1_MOLCAS')  # -> scratch/tmp_mid_qm2_task1_MOLCAS

# Or with explicit full calcdir path:
backend = MolcasBackend(config_dict, calcdir='/path/to/calcdir')

# Execute calculation
results = backend.local_run()   # local subprocess
results = backend.slurm_run()   # SLURM subprocess

# For multi-step dynamics, reuse the same backend instance:
for step in trajectory:
    results = backend.calculate(atoms, coordinates, point_charges)
    # calcdir remains fixed, orbital files (RasOrb) are reused as initial guesses

# Optional per-step target states (e.g. chosen by a NAMD workflow);
# None keeps the target_states fixed at initialization:
results = backend.calculate(atoms, coordinates, target_states=[1])
```

## Configuration Schema
Key fields in the JSON config:
- `atoms`: List of atom symbols
- `coordinates`: List of [x, y, z] coordinates in Angstrom
- `point_charges`: Optional ESPF background charges [charge, x, y, z] in Angstrom
- `embedding`: `none`, `charge`, or `field` (`potential` is retired; ONIOM supports only none/charge, QMMM only none/field). Read only when the data arrives as an array - a `read_charge` file declares its own representation
- `read_charge`: `none` (use the per-step point_charges/external_potential arrays), `true` (the project's own `<project>.charge`), or a filename; the file's first line says whether it holds point charges or a field
- `external_potential`: (natom, 4) [V, Fx, Fy, Fz] in atomic units, the field representation
- `caspt2_type`: `ss`, `ms`, `xms`, `rms`, or `xdw` (empty resolves from the legacy `xms` switch)
- `nocsf`: omit the CSF term from NAC; default `true` for both CASSCF and CASPT2
- `caspt2_ipea_shift` / `caspt2_real_shift` / `caspt2_imaginary_shift`: CASPT2 IPEA, real shift, imaginary shift
- `dwtype` / `dwms_zeta`: XDW-CASPT2 exponent type and zeta
- `charges`: Total charge (default: 0)
- `multiplicity`: Spin multiplicity (default: 1)
- `nstate`: Number of electronic states (default: 1)
- `profile`: Calculation profile (e.g. 'casscf_energy')
- `method`: Electronic structure method ('casscf' or 'caspt2')
- `basis`: Basis set (e.g. 'ano-rcc-vdzp')
- `nact_elec`: Number of active electrons
- `nact_orb`: Number of active orbitals
- `ci_roots`: Number of CI roots per spin block
- `ci_spin`: Spin quantum number (2S) per spin block
- `opt_state_i` / `opt_state_j`: Optimization target states (0-based) for geometry optimization
- `opt_maxit`: Maximum SLAPAF relaxation iterations (also exported as MOLCAS_MAXITER)
- `maxit_scf`: Maximum RASSCF CI iterations (first number of ITERATIONS, default 200)
- `maxit_md`: Number of MD steps for dynamics (FOREACH loop bound); in framework-driven runs the value is forwarded from the framework input sections (e.g. `&md`), while an explicit `&molcas` entry keeps the highest priority
- `md_maxiter`: MOLCAS_MAXITER export limit for dynamics (0 = auto, maxit_md x 10; must exceed maxit_md)
- `velocities`: Initial velocities in Bohr/au, required for dynamics (written as `$MOLCAS_PROJECT.velocity.xyz`)
- `root`: Initial MD state (internal 0-based; framework external inputs spell it 1-based and the adapter decrements once); MDRLXR is derived from `current_state + 1`
- `size` / `deco` / `thermostat`: MD timestep in au, surface-hopping decoherence, thermostat ('off'/'nve' -> NVE, 'nose'/'nose-hoover'/'nvt' -> Nose-Hoover NVT)
- `execution_mode`: 'local' or 'slurm'
- `molcas_project`: Project name for Molcas files
- `scratch_path`: File placement directory for calculations ('AUTO' for automatic scratch selection, e.g. /srv/tmp)
- `molcas_mem`: Memory in MB
- `nproc`: Number of MPI ranks (`mpirun -np`); `molcas_nproc` is the framework
  spelling of the same count
- `thread`: OpenMP threads per rank (`OMP_NUM_THREADS`); `molcas_thread` is the
  framework spelling. A written pair folds to its larger value (#306)

## CASPT2 Variants and Analytic Properties
- `caspt2_type='ss'`: single-state CASPT2; one state per call (SOC may use one state per spin manifold).
- `caspt2_type='ms'`: `MULTISTATE all`.
- `caspt2_type='xms'`: `XMULTISTATE all` (default).
- `caspt2_type='rms'`: `RMULTISTATE all`.
- `caspt2_type='xdw'`: `XMULTISTATE all` + `DWMS <zeta>` + `DWTYPE <1|2|3>`.
- Analytic gradient: one `GRDT` + one `&ALASKA ROOT=n` per target state; RI/CD is mandatory.
- For a portable analytic-gradient/NAC run set `caspt2_ipea_shift: 0.0` (emits `IPEA 0.0`, no `DORT`): OpenMolcas v25.06 segfaults on the `IPEA=0.25` + `DORT` path, while v26.10 (local build and the deployment) accepts either.
- Analytic NAC: one `GRDT` in CASPT2, then one `&ALASKA NAC=i j` per pair; `NOCSF` is added by default (both CASSCF and CASPT2 NAC exclude CSF). ms/xms/rms/xdw only (ss+NAC is refused).
- Multiple gradients/NACs: one `GRDT`, then multiple `&ALASKA ROOT=n` and `&ALASKA NAC=i j` blocks. No `RLXROOT` or `&MCLR` block is needed.
- SOC: multistate CASPT2 uses RASSI `HEff`; a single-root spin block has no CASPT2 effective Hamiltonian on its JobIph, so RASSI keeps that block's CASSCF diagonal. The parser still takes the block's `FINAL CASPT2 RESULT` as the pre-SOC energy and warns when it differs from the RASSI diagonal. `ss` uses `EJob` as an approximation.
- The returned energy is the pre-SOC CASPT2 energy; `SO-RASSI State` eigenvalues are never used as `energy`.
- Gas-phase CASPT2 gradient+NAC is one pass: `GRDT`, then the mixed ALASKA ROOT/NAC blocks.

## ESPF Embedding
- `embedding='none'`: no ESPF and no XField (unless `read_charge` names a file - the file is then the whole switch).
- `embedding='charge'`: `&ESPF` with `External = n Angstrom`, rows `x y z q 0 0 0`.
- `embedding='field'`: `&ESPF` with `External = 0`, rows `atom V Fx Fy Fz 0 0 0 0 0 0`; all G components are zero.
- `read_charge='none'`: charges/potential come from the per-step `point_charges` / `external_potential` arrays.
- `read_charge=<file>`: the file is fixed for the run, read from the calculation directory or the launch directory, and its first line declares what it holds:
  - `point angstrom|bohr external`: rows `q x y z`, arbitrary external points;
  - `point angstrom|bohr atom`: atom-indexed - refused here, only the framework's charge store consumes it (q column, positions from the live geometry);
  - `field au`: rows `V Fx Fy Fz`, one per QM atom.
  Older spellings still read: `charge angstrom|bohr` + `x y z q`, `potential au` + `V Fx Fy Fz`, and a bare count line + `q x y z`.
- ESPF is emitted immediately after SEWARD. ESPF + NAC is executed as two passes: ESPF energy/gradient/SOC first, then a no-ESPF NAC pass. The NAC is labelled as a no-ESPF approximation.
- ESPF gradients omit the ESPF grid-point derivatives (OpenMolcas limitation).

## Geometry Optimization Modes
The `casscf_geometry_optimization` profile derives its mode from `opt_state_i`/`opt_state_j` and `state_spin`:
- **Single-state** (`opt_state_i == opt_state_j`): plain relaxation of the target state; RASSCF gets `Rlxroot=n`
- **MECI** (same-spin pair): GATEWAY adds a `Constraints` block with `a = Ediff n m` (1-based crossing roots); Rlxroot relaxes the lower state
- **MECP** (cross-spin pair): GATEWAY adds a bare `a = Ediff` constraint; `opt_state_i` must be the singlet and `opt_state_j` the triplet, controlled by dual singlet/triplet RASSCF jobs with per-job `Rlxroot` (triplet roots renumbered by the singlet count)

All optimizations export `MOLCAS_MAXITER=opt_maxit` on the first line and wrap SEWARD/RASSCF/SLAPAF in a `>> Do while` loop (GATEWAY stays outside so SLAPAF-updated coordinates are re-read); RASSCF reads `fileorb` for orbital continuation between iterations.

## Molecular Dynamics (ab initio / surface hopping)
For `casscf_abinitio_dynamics` and `casscf_surface_hopping`, Molcas propagates the nuclei internally in a single run:
- The input header copies `$Project.xyz` and `$Project.velocity.xyz` from the submit directory into the working directory, then exports `MOLCAS_MAXITER` (md_maxiter, default maxit_md x 10; must exceed the MD step count)
- `>> FOREACH ITER in (1 .. maxit_md)` wraps SEWARD..DYNAMIX (GATEWAY stays outside), closed by `>> End do` after DYNAMIX
- RASSCF carries `MDRLXR=current_state+1` (the initial MD state; dynamics.root seeds current_state) and reads `fileorb=$MOLCAS_PROJECT.RasOrb` for orbital continuation; ALASKA stays bare
- DYNAMIX uses `velver`, `dt=size`, `velo=1` (reads the velocity file), `ther` from thermostat (0 = NVE, 2 = Nose-Hoover NVT; anything else raises) and `temp`
- Initial velocities are mandatory (from the framework title.velo); a missing velocities field raises

## Final Structure Reporting (fin_coord)
Iterative profiles return their final structure through the framework `fin_coord` channel, and the Calculator applies it to the system coordinates:
- `casscf_geometry_optimization`: the relaxed structure is read from `$MOLCAS_PROJECT.opt.xyz`
- `casscf_abinitio_dynamics` / `casscf_surface_hopping`: the last frame of the `$MOLCAS_PROJECT.md.xyz` trajectory is read

Coordinates are in Angstrom. A missing output file or an atom-count mismatch raises (fail-fast).

## Important Notes
- Coordinates are written to a separate .xyz file and referenced via `$MOLCAS_PROJECT.xyz` in the GATEWAY input, not inlined
- Background charges/potential are embedded via ESPF (`&ESPF` immediately after SEWARD); XField is no longer emitted
- CASSCF SOC performs dual RASSCF (singlet + triplet) before RASSI; CASPT2 SOC copies the CASPT2 JobIph per spin block and uses RASSI HEff (or EJob for single-state CASPT2). A single-root spin block contributes its CASSCF diagonal to RASSI; the pre-SOC energy comes from that block's FINAL CASPT2 RESULT.
- WORKDIR 'AUTO' mode uses `/srv/tmp` if available, otherwise `/tmp`, for fast I/O
- The calculation directory (calcdir) is fixed per backend instance, enabling orbital file reuse (RasOrb -> StrOrb) as initial guesses across multiple steps
- If a `{title}.StrOrb` file exists next to the framework input file (title = control.title, defaulting to the input basename), it is auto-detected and injected as `guess_orbital` (an explicit value takes precedence); the skill stages it into calcdir as `$MOLCAS_PROJECT.RasOrb` (never overwriting an existing file) and all RASSCF blocks then read `fileorb=$MOLCAS_PROJECT.RasOrb`; optimize without any guess keeps the historical `fileorb=$MOLCAS_PROJECT.StrOrb` line
- Execution namespace (`execution_id`) is assigned by the Calculator at initialization, not auto-detected: `MolcasBackend(config, execution_id='tmp_MOLCAS')` -> `scratch/tmp_MOLCAS`
- molcas_skill.agent is the CLI entry point for subprocess execution
- Python API: directly obtain `MolcasBackend` and call `local_run()` / `slurm_run()` / `calculate()`
- Both local and slurm execution run via subprocess
