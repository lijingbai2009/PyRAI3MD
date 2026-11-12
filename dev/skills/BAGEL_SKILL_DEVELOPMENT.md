# BAGEL Skill Development Guide

Last updated: 2026-11-02 10:40 (#448: the deck-builder paragraph now names the declared-pair rule (the self pair `(0, 0)` is out of the fallback), and the returned-channel list reports `transition_dipole`/`osc_strength` as (npair,...) rows with an empty table giving an empty axis. The rest was not re-audited. Before it, 2026-10-22 22:30 (the config section no longer claims a `create_model` field merge, `bagel_workdir` is dropped from both the Environment list and the parameter table since no such field ships, and the test-config generator is the qualification converter rather than `dev_tool.py`. The rest was not re-audited.)

Before it, 2026-10-14 22:25 (the `=== FCI iteration ===` row format re-read against 17 logs of test round `tmp_20261014_185533` - the state number is the second column, the first counts CI solves, and the last print of a state is the converged one - plus the `ENERGY.out` attribution caveat in the constraints list; the rest of the file was not re-audited line by line)

This document covers **BAGEL-specific** implementation details for bagel_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. BAGEL-Specific Design Considerations

### 1.1 JSON Input Format

BAGEL uses **JSON format** for input files, unlike text-based module chains.

Each computation is described as a set of JSON blocks assembled into a `{"bagel": [...]}` structure:

```json
{
  "bagel": [
    {
      "molecule": {
        "geometry": [
          {"atom": "C", "xyz": [0.0, 0.0, 0.0]},
          {"atom": "H", "xyz": [1.0, 0.0, 0.0]}
        ],
        "basis": "cc-pvdz",
        "df_basis": "cc-pvdz-jkfit"
      }
    },
    {
      "casscf": {
        "natocc": true,
        "nstate": 2,
        "nact": 2,
        "nclosed": 0,
        "active": [0, 1]
      }
    },
    {
      "forces": {
        "title": "forces",
        "grads": [
          {"title": "force", "target": 0}
        ],
        "export": true
      }
    }
  ]
}
```

The skill's input builder constructs these JSON blocks programmatically as Python dicts
and serializes them to JSON, rather than assembling text-based module blocks.

### 1.2 Archive-Based Orbital Management

BAGEL uses `.archive` files to store and transfer orbital information between calculations.

- Prior CASSCF calculations produce a `{project}_converged.archive` file
- This file is copied as `{project}_guess.archive` for subsequent calculations (e.g., CASPT2)
- The `file` field in JSON blocks references the archive filename (without extension)

The `manage_archives()` function in `input_builder.py` handles:
- Copying `guess_orbital` to `{project}_guess.archive` if provided
- Copying `{project}_converged.archive` to `{project}_guess.archive` (orbital reuse from previous step)

The `load_ref` block is written **conditionally**: `build_input()` checks whether
`{project}_guess.archive` exists in the calculation directory (after
`manage_archives()` has staged it) and skips the block when it is absent. A
`load_ref` block pointing at a missing archive aborts BAGEL with
`File not found`, so first-run CASPT2 calculations without a guess archive simply
omit the block (same skip-if-absent policy as the molcas orbital guess). When an
archive is available it is copied into the calculation directory under the scratch
path and the block references its name.

This mechanism replaces the text-based orbital restart approach used by other backends.

### 1.3 Coordinate Unit Convention

BAGEL defaults to **Bohr**, but the molecule block sets
`"angstrom": "true"` so coordinates are passed in **Angstrom** (the
PyRAI3MD standard) without any conversion:

```python
molecule['angstrom'] = 'true'
geometry.append({
    'atom': element,
    'xyz': [x, y, z]
})
```

Point charges use the same convention, with the key `"atom": "Q"` and an additional
`"charge"` field. The log parser reads coordinates back as Angstrom
(no unit conversion in either direction).

### 1.4 Density Fitting Basis

BAGEL requires a **density fitting (DF) basis** for Coulomb and exchange integrals;
the `df_basis` keyword is always written into the molecule block. BAGEL ships only
seven auxiliary sets: `svp-jkfit`, `tzvpp-jkfit`, `qzvpp-jkfit`, `cc-pvdz-jkfit`,
`cc-pvtz-jkfit`, `cc-pvqz-jkfit`, `cc-pv5z-jkfit`.

Resolution rules:
- An explicit `df_basis` must be one of the seven sets (validated, case-insensitive)
- An empty/omitted `df_basis` is resolved by similarity matching
  (`input_builder._match_df_basis`):
  - `svp`/`tzvpp`/`qzvpp` → same-name `-jkfit`
  - Dunning bases → `cc-pv{X}z-jkfit` with the same cardinal number X
    (augmentation, core-valence, weighted and `-dk` variants are ignored,
    e.g. `aug-cc-pwcv5z` → `cc-pv5z-jkfit`; 6z downgrades to `cc-pv5z-jkfit`)
  - No match (`sto-3g`, `3-21g`, `6-31g`, `ano-rcc`, ...) → default `svp-jkfit`

The DF basis path is resolved relative to the BAGEL installation directory:
`{bagel_path}/share/{df_basis}.json`.

---

## 2. BAGEL Calculation Profiles

### 2.1 Complete Profile List (14 profiles)

| Profile | Capability | BAGEL Step Chain |
|---|---|---|
| `casscf_energy` | energy | molecule > load_ref > casscf > save_ref |
| `casscf_gradient` | gradient | molecule > load_ref > casscf > forces |
| `casscf_nac` | nac | molecule > load_ref > casscf > forces_nacme |
| `casscf_frequency` | frequency | molecule > load_ref > casscf > hessian |
| `casscf_geometry_optimization` | geometry_optimization | molecule > load_ref > casscf > `maxstep = 0.05` in the optimize section (the vendor template value; an initial analytic CASPT2 gradient measured `1.27e3` a.u. and otherwise throws the first internal-coordinate step off the surface), optimize |
| `casscf_dipole` | dipole | molecule > load_ref > casscf > forces_dipole |
| `casscf_transition_dipole` | transition_dipole | molecule > load_ref > casscf > forces_dipole |
| `xmscaspt2_energy` | energy | molecule > load_ref > casscf > smith_caspt2 |
| `xmscaspt2_gradient` | gradient | molecule > load_ref > forces_caspt2 |
| `xmscaspt2_nac` | nac | molecule > load_ref > forces_caspt2_nacme |
| `xmscaspt2_frequency` | frequency | molecule > load_ref > hessian_caspt2 |
| `xmscaspt2_geometry_optimization` | geometry_optimization | molecule > load_ref > optimize |
| `xmscaspt2_dipole` | dipole | molecule > load_ref > forces_dipole |
| `xmscaspt2_transition_dipole` | transition_dipole | molecule > load_ref > forces_dipole |

### 2.2 Step Dependency Notes

- **molecule**: Defines geometry (Angstrom via `"angstrom": "true"`), basis set, DF basis, point charges
- **casscf**: Runs CASSCF with `natocc=True`, `nstate`, `nact`, `nclosed`, `active` orbitals
- **save_ref**: Saves CASSCF orbitals to `{project}_converged.archive`
- **load_ref**: Loads previously saved CASSCF orbitals from `{project}_guess.archive`; the block is only written when that archive exists in the calculation directory (applies to every casscf and xmscaspt2 chain)
- **forces**: Computes analytical gradients for specified target states; requires an embedded CASSCF `method` definition array
- **forces_nacme**: Computes non-adiabatic coupling matrix elements (NACME) between state pairs; requires an embedded CASSCF `method` definition array
- **forces_dipole**: Forces block with the `dipole` keyword and `ciderivative: "false"` grads entries (validated BAGEL dipole workflow); force entries yield state dipoles, nacme entries yield transition dipoles; requires an embedded method definition array (CASSCF or CASPT2 depending on the profile level)
- **hessian**: Computes analytical Hessian via finite differences of gradients (CASSCF level)
- **smith_caspt2**: Standalone `smith` block running the CASPT2 algorithm after the preceding casscf block; BAGEL does not support a standalone caspt2-titled block, so single-point XMS-CASPT2 energies always use this casscf + smith pair (flags `ms`/`xms`/`sssr` plus the `shift` level shift only)
- **forces_caspt2**: Computes gradients with XMS-CASPT2 method
- **forces_caspt2_nacme**: Computes NACME with XMS-CASPT2 method
- **hessian_caspt2**: Computes Hessian with XMS-CASPT2 method (uses `frozen=True`)
- **optimize**: Runs a geometry optimization on the target state; the block carries its own `method` definition array (CASSCF or XMS-CASPT2 depending on the profile level)

---

## 3. BAGEL Schema Details

### 3.1 Configuration Models

The config is composed of 5 Pydantic sub-models nested in one `SkillConfig` (`schema/config.py:342-352`), with flat keyword access routed to the sub-model that owns each keyword - the former `create_model` field merge is gone:

- **System**: atoms, coordinates, charges, multiplicity, nstate, state_spin, nac_pair_map
- **Properties**: energy, gradient, nac, dipole, transition_dipole, hessian, frequencies
- **Dynamics**: state_population, temperature
- **Calculation**: profile, method, basis, df_basis, active space (nact_elec, nact_orb, nclosed, active),
  CI parameters (ci_roots, ci_spin), target_states, target_nac_pair, opt_state_i/opt_state_j,
  scf_converge, pt2_shift, maxit_scf/maxit_opt, guess_orbital
- **Environment**: bagel_project, bagel_path, bagel_nproc,
  library paths (blas, lapack, boost, mpi, mkl, arch), mpiexe,
  keep_tmp, nproc, scratch_path, slurm_*, thread

### 3.2 Active Space Parameters

BAGEL uses a simpler active space model compared to Molcas:

- **nact_elec**: Number of active electrons
- **nact_orb**: Total number of active orbitals
- **nclosed**: Number of closed (frozen) orbitals below the active space
- **active**: List of active orbital indices (auto-generated from `nclosed` if empty)

Orbital index convention: `active` is **1-based at every boundary**
(user yaml, classic input, BAGEL JSON) and stored **0-based
internally** for list operations. `build_config` converts the
user-provided 1-based indices to 0-based on ingest, and the input
builder converts them back to 1-based when writing the BAGEL JSON.

The `build_active_orbitals()` function in `normalize.py` generates default active orbital
indices (0-based, internal) if not provided:

```python
if not active and nact_orb > 0:
    data['active'] = list(range(nclosed, nclosed + nact_orb))
```

### 3.3 Density Fitting Basis

- **df_basis**: Density fitting basis set; defaults to empty so the
  builder resolves it by similarity to `basis` (see §1.4). An explicit
  value must be one of the seven supported jkfit sets (validated in
  `validate.py`); `no`/`none` is no longer accepted
- Always written to the molecule block; resolved to full path:
  `{bagel_path}/share/{df_basis}.json`
- Required for Coulomb and exchange integral approximation

### 3.4 Supported Methods

**casscf** (Complete Active Space SCF):
- Capabilities: energy, gradient, nac, frequency, geometry_optimization, dipole, transition_dipole
- Steps: molecule, load_ref, casscf, save_ref, forces, forces_nacme, forces_dipole, hessian, optimize

**xmscaspt2** (XMS-CASPT2):
- Capabilities: energy, gradient, nac, frequency, geometry_optimization, dipole, transition_dipole
- Steps: molecule, load_ref, casscf, smith_caspt2, forces_caspt2, forces_caspt2_nacme, forces_dipole, hessian_caspt2, optimize

### 3.5 No SOC Support

BAGEL does not currently support SOC (spin-orbit coupling) calculations.
The schema has no `target_soc_pair`, `soc_pair_map`, `nsoc`, or `singlet/triplet_ci_*` fields.
Only NAC between same-spin states is supported.

### 3.6 Iteration Budgets and CASPT2 Convergence Keywords

The iteration-budget keywords map onto the BAGEL `maxiter` keyword:

- **maxit_scf** (default `500`): feeds the `maxiter` keyword of **every
  casscf and caspt2 keyword region** — the standalone `casscf` block
  and the embedded caspt2 method definitions inside `forces`,
  `forces_nacme`, `forces_dipole`, `hessian`, and `optimize`-level
  methods, including the `maxiter` inside the caspt2 `smith` sub-block
  and the method-level `maxiter` beside `smith`
- **maxit_opt** (default `100`): feeds the `maxiter` keyword of the
  `optimize` block itself

Two further keywords feed the caspt2 `smith` sub-block:

- **pt2_shift** (default `0.2`): feeds the smith `shift` keyword
  (XMS-CASPT2 level shift) in every caspt2 method definition and in
  the standalone `smith_caspt2` energy block
- **scf_converge** (default `1e-8`): feeds the smith `thresh` keyword
  in every caspt2 block; the test yamls set `scf_converge: 1e-6`

The former CASPT2-specific constants (`caspt2_maxiter`/`caspt2_shift`/
`caspt2_thresh` in `DEFAULT_CONVERGENCE`) are retired; all budgets and
thresholds now come from user-facing keywords declared in `DEFAULT`
(`schema/reference.py`) and exposed as `Calculation` fields
(`schema/config.py`); the framework adapter passes them through as
plain scalars.

---

## 4. BAGEL Normalize Workflow

The `normalize.py` module performs three key steps:

### 4.1 verify_electronic_structure

- Validates atom/coordinate count match
- Computes total electrons from atomic numbers and charge
- Computes alpha/beta electron counts from multiplicity

### 4.2 verify_state_num

- Validates nstate >= 1
- Validates target_states are within [0, nstate)
- Validates target_nac_pair indices
- Initializes energy, dipole, gradient arrays
- Builds transition_pair list
- Builds **nac_pair_map** (same-spin pairs only, no SOC)

### 4.3 build_active_orbitals

- If `active` list is empty and `nact_orb > 0`, generates default active orbital indices
- Default (0-based, internal): `[nclosed, nclosed+1, ..., nclosed+nact_orb-1]`
- The input builder adds +1 when writing to the BAGEL JSON (1-based)

---

## 5. BAGEL Input Builder Details

### 5.1 JSON Block Assembly

Each BAGEL JSON block is generated by a dedicated builder function in `input_builder.py`:

| Block | Builder Function | Key Parameters |
|---|---|---|
| molecule | `_build_molecule()` | geometry (Angstrom, `angstrom: true`), basis set, df_basis, charge, point charges (Q atoms) |
| casscf | `_build_casscf()` | natocc=True, nstate, nact, nclosed, active, maxiter |
| save_ref | `_build_save_ref()` | file: `{project}_converged` |
| load_ref | `_build_load_ref()` | file: `{project}_guess` |
| forces | `_build_forces()` | grads list (force/nacme), target states, nacmtype='interstate', export=True |
| forces_nacme | `_build_forces(nac=True)` | nacme targets from target_nac_pair |
| hessian | `_build_hessian_casscf()` | target state, method array with casscf block |
| smith_caspt2 | `_build_smith_caspt2()` | title 'smith', method 'caspt2', ms/xms/sssr true, shift(pt2_shift); standalone smith block for XMS-CASPT2 energies (BAGEL has no standalone caspt2-titled block) |
| forces_caspt2 | `_build_forces_caspt2()` | forces block with caspt2 method |
| forces_caspt2_nacme | `_build_forces_caspt2(nac=True)` | forces_caspt2 with nacme entries |
| hessian_caspt2 | `_build_hessian_caspt2()` | hessian block with caspt2 method, frozen=True |

### 5.2 Molecule Block Details

The molecule block includes:

```json
{
  "title": "molecule",
  "basis": "/path/to/bagel/share/cc-pvdz.json",
  "df_basis": "/path/to/bagel/share/cc-pvdz-jkfit.json",
  "angstrom": "true",
  "geometry": [
    {"atom": "C", "xyz": [0.0, 0.0, 0.0]},
    {"atom": "Q", "xyz": [1.0, 0.0, 0.0], "charge": -0.5}
  ]
}
```

- Basis set paths are resolved relative to `bagel_path`
- Point charges are embedded as `"atom": "Q"` with `"charge"` field
- Coordinates are written in Angstrom; `"angstrom": "true"` tells
  BAGEL to read them as-is (without the flag BAGEL assumes Bohr)

### 5.3 CASSCF Block Details

```json
{
  "title": "casscf",
  "natocc": true,
  "nstate": 3,
  "nact": 2,
  "nclosed": 4,
  "active": [5, 6],
  "maxiter": 500
}
```

- `natocc=True`: Uses natural orbital occupation numbers
- `active`: Optional list of active orbital indices in the BAGEL JSON
  (1-based; the skill stores them 0-based internally and converts
  on write). Example: `nclosed=4`, `nact=2` -> `[5, 6]`
- If `active` is empty, BAGEL uses default orbital ordering

### 5.4 Forces Block Details

The forces block uses a `grads` list to specify multiple gradient/NACME computations.
BAGEL **requires** a `method` keyword defining the electronic-structure method inside
the forces block; omitting it aborts with
`A required keyword is missing from the input for forces: method`.

**Critical**: `method` must be an **array of method definition blocks**, not a bare
dict. A bare dict aborts BAGEL with `ERROR: EXCEPTION RAISED: : unknown method`.

**CASSCF forces** (follows the official BAGEL gradient template):
```json
{
  "title": "forces",
  "grads": [
    {"title": "force", "target": 0},
    {"title": "force", "target": 1},
    {"title": "nacme", "target": 0, "target2": 1, "nacmtype": "interstate"}
  ],
  "export": true,
  "method": [
    {
      "title": "casscf",
      "natocc": true,
      "nstate": 3,
      "nact": 2,
      "nclosed": 7,
      "maxiter": 500,
      "active": [8, 9]
    }
  ]
}
```

**XMS-CASPT2 forces** (the method block carries `smith` with
`frozen=true`; `shift` comes from `pt2_shift`, `thresh` from
`scf_converge`, and both the smith `maxiter` and the method-level
`maxiter` align with `maxit_scf`; `natocc`/`davidson_subspace` are
omitted from the method block):
```json
{
  "title": "forces",
  "grads": [
    {"title": "force", "target": 0},
    {"title": "nacme", "target": 0, "target2": 1, "nacmtype": "interstate"}
  ],
  "export": true,
  "method": [
    {
      "title": "caspt2",
      "smith": {
        "method": "caspt2",
        "ms": true,
        "xms": true,
        "sssr": true,
        "shift": 0.2,
        "thresh": 1e-6,
        "maxiter": 500,
        "frozen": true
      },
      "nstate": 3,
      "nact": 2,
      "nclosed": 4,
      "maxiter": 500
    }
  ]
}
```

- `force`: Computes gradient for specified target state
- `nacme`: Computes NACME between `target` and `target2` states
- `nacmtype='interstate'`: Fixed pairing for interstate NACME computation
- `export=True`: Exports results to separate output files
- `method`: Array of embedded method definitions (CASSCF for
  `forces`/`forces_nacme`, XMS-CASPT2/smith for
  `forces_caspt2`/`forces_caspt2_nacme`); the CASSCF definition mirrors the
  standalone casscf block, including 1-based `active` orbital indices and
  `maxiter` sourced from `maxit_scf`

Force targets are generated strictly from `target_states`; the workflow
selects per-step states through `calculate(target_states=...)`, which
overrides `config_dict['target_states']` before the input file is
built. The parser reads `FORCE_i.out` per state and zero-fills
untargeted states, so the gradient is always a full `(nstate, natom,
3)` array.

### 5.5 XMS-CASPT2 Block Details

BAGEL does not support a standalone `title: caspt2` block — that form
only appears as a method definition inside higher-level tasks
(forces/hessian/optimize). A single-point XMS-CASPT2 energy instead
runs a casscf block followed by a standalone `smith` block naming the
caspt2 algorithm:

```json
{
  "title": "casscf",
  "natocc": true,
  "nstate": 3,
  "nact": 2,
  "nclosed": 4,
  "maxiter": 500,
  "active": [5, 6]
}
```
```json
{
  "title": "smith",
  "method": "caspt2",
  "ms": true,
  "xms": true,
  "sssr": true,
  "shift": 0.2
}
```

- `method`: The Smith algorithm name (`"caspt2"` as a plain string, not
  a method definition array)
- `ms=True`: Multi-state CASPT2
- `xms=True`: Extended multi-state (XMS-CASPT2)
- `sssr=True`: State-specific state average
- `shift`: sourced from `pt2_shift` (default `0.2`)
- The active-space and iteration keywords live in the preceding
  casscf block; the smith block carries only the flags and the level
  shift, exactly as in the official template

### 5.6 Hessian Block Details

Frequency calculations use the `hessian` block with a `method` array;
the embedded method definition is reused verbatim from the method
builders, so it carries the full keyword set (natocc/maxiter for
CASSCF, smith thresh/maxiter for CASPT2):

**CASSCF Hessian:**
```json
{
  "title": "hessian",
  "target": 0,
  "method": [
    {
      "title": "casscf",
      "natocc": true,
      "nstate": 3,
      "nclosed": 4,
      "nact": 2,
      "maxiter": 500,
      "active": [5, 6]
    }
  ]
}
```

**XMS-CASPT2 Hessian:**
```json
{
  "title": "hessian",
  "target": 0,
  "method": [
    {
      "title": "caspt2",
      "smith": {
        "method": "caspt2",
        "ms": true,
        "xms": true,
        "sssr": true,
        "shift": 0.2,
        "thresh": 1e-6,
        "maxiter": 500,
        "frozen": true
      },
      "nstate": 3,
      "nact": 2,
      "nclosed": 4,
      "maxiter": 500
    }
  ]
}
```

Note: CASPT2 Hessian uses `frozen=True` to freeze certain orbitals during the calculation.

### 5.7 Optimize Block Details

Geometry optimization uses the `optimize` block (profiles
`casscf_geometry_optimization` / `xmscaspt2_geometry_optimization`).
Following the official BAGEL template:

**CASSCF optimize (local minimum, opttype energy):**
```json
{
  "title": "optimize",
  "opttype": "energy",
  "target": 0,
  "maxiter": 100,
  "method": [
    {
      "title": "casscf",
      "natocc": true,
      "nstate": 3,
      "nact": 2,
      "nclosed": 7,
      "maxiter": 500,
      "active": [8, 9]
    }
  ]
}
```

**Conical intersection (opttype meci):** when `opt_state_i != opt_state_j`
the block optimizes the minimum-energy conical intersection and carries
`"target2": opt_state_j` in addition to `"target": opt_state_i`.

**XMS-CASPT2 optimize:** the `method` array carries the same
`caspt2`/`smith` block used by CASPT2 forces (with `frozen=true`).

- `opttype`: `energy` (local minimum, BAGEL default) or `meci`
  (minimum-energy conical intersection); derived from the state pair —
  `opt_state_i == opt_state_j` stays `energy`, a distinct pair becomes
  `meci`
- `target`: `opt_state_i` (0-based optimization target state)
- `target2`: `opt_state_j`, written only for `opttype=meci`
- `maxiter`: sourced from `maxit_opt` (default `100`)
- `method`: array of method definitions (same rules as the forces block);
  CASSCF or XMS-CASPT2 depending on the profile level
- `normalize.py` validates `0 <= opt_state_i/opt_state_j < nstate`

### 5.8 Dipole Forces Block Details

State dipoles and transition dipoles come from a `forces` block
carrying the `dipole` keyword (profiles `*_dipole` /
`*_transition_dipole`). Following the cluster-validated BAGEL dipole
workflow: the `dipole` keyword switches dipole output on, and every
`grads` entry disables derivative evaluation via
`"ciderivative": "false"` — force entries yield state dipoles, nacme
entries yield transition dipoles. An empty grads list (the older
official template) does **not** work.

**CASSCF dipole (state dipoles via force entries):**
```json
{
  "title": "forces",
  "dipole": "true",
  "grads": [
    { "title": "force", "target": 0, "ciderivative": "false" },
    { "title": "force", "target": 1, "ciderivative": "false" }
  ],
  "method": [
    {
      "title": "casscf",
      "natocc": true,
      "nstate": 3,
      "nact": 2,
      "nclosed": 7,
      "maxiter": 500,
      "active": [8, 9]
    }
  ]
}
```

**Transition dipoles via nacme entries** (`*_transition_dipole`
profiles): one entry per `transition_pair`. The table is the framework's
declaration, read through; when nothing was declared the skill's own
convention is `[[0, i] for i in 1..nstate-1]`, and the self pair `(0, 0)` is
not in it - the deck builder used to open its request with one (#448):
```json
{ "title": "nacme", "target": 0, "target2": 1, "nacmtype": "interstate", "ciderivative": "false" }
```

**XMS-CASPT2 dipole:** the `method` array carries the `caspt2`/`smith`
block (with `frozen=true`) used by CASPT2 forces.

- `dipole`: written as the string `"true"`; decides whether dipole
  properties are computed at all
- `grads`: force entries correspond to state dipoles, nacme entries
  to transition dipoles; `ciderivative: "false"` disables the
  gradient/NAC evaluation (drop it to compute gradients/NAC alongside)
- Request selection: defaults derive from the profile (`_dipole` →
  force entries, `_transition_dipole` → nacme entries); the
  `req_dipole` / `req_transition_dipole` flags override this and may
  be set together to request both property sets in one block
- `excitation` energies and `osc_strength` are derived during parsing (see §6.3)

---

## 6. BAGEL Result Parsing

### 6.1 Output Files

| File | Content |
|---|---|
| `{project}.log` | Main log with JSON-formatted geometry (Angstrom), Hessian matrix, vibrational frequencies, NACME energy gaps, dipole-moments sections |
| `ENERGY.out` | State energies in Hartree (one per line) |
| `FORCE_{i}.out` | Gradient for state i (header line + natom rows with a leading row-index column; columns 1:4 are x/y/z, Hartree/Bohr) |
| `NACME_{i}_{j}.out` | Derivative coupling vector between states i and j (same layout as FORCE); BAGEL already divides by the energy gap |
| `opt.molden` | Optimization geometry in Molden format (`[Atoms] Angs`); fin_coord fallback when the log carries no complete geometry block |

BAGEL does **not** write HESSIAN.out / FREQUENCY.out / dipole files:
the Hessian and frequencies are printed in the log (`++ Hessian ++`
section and `Freq (cm-1)` lines), and state dipoles, transition
dipoles and oscillator strengths appear in the `* CASSCF dipole
moments` / `* CASPT2 dipole moments` sections.

### 6.2 Parsing Logic (log_parser.py)

Before any result extraction, `parse_log()` runs a completion check
(`_check_log_completion`, same policy as the molcas skill) so that failed runs are
never masked by zero-filled results:

- Missing `{project}.log` → `FileNotFoundError` (the calculation produced no output)
- Any log line containing `ERROR` (BAGEL prints `ERROR: EXCEPTION RAISED: ...` on
  crashes) → `RuntimeError` carrying the offending line

The backend additionally raises `RuntimeError` when the runner returns a non-zero
return code. `parse_log()` then extracts:

- **Coordinates**: From `.log` file (JSON geometry, Angstrom — no conversion)
- **Energies**: From `ENERGY.out` (flat array, reshaped to nstate). For a CASSCF
  run the log's `=== FCI iteration ===` block is the primary source. A row prints
  `[ci solve] [state] [*] energy residual overlap`: the **state** number is the
  second column, and the first column counts the CI solves of that block, so a
  single root reprinted once per solve makes several rows that all carry state 0
  (both shapes measured across 17 logs of test round `tmp_20261014_185533`;
  which states the block answered for is confirmed by the `* ci vector, state N`
  banners that follow it). One row per state is kept - the **last** print, which is
  the converged one and, in every block measured, the one carrying the `*`. State
  numbers that are not `0, 1, 2, ...` are refused, because packing by row position
  would hand an energy to the wrong state while the row count still agreed. When
  neither the log section nor `ENERGY.out` carries `nstate` values the parse
  raises instead of returning zeros - a zero-filled array satisfies the shape
  contract and describes no surface.
- **Mixed spin is refused at the input**: `state_spin` labels a row per state, but
  the input this skill writes carries no spin or multiplicity keyword at all (in
  `&molecule` or the `&mcscf` block), so every root BAGEL solves comes from one
  manifold. A request naming both a singlet and a triplet - the shape a MECP
  search makes - passes that label straight through to a pair of same-spin roots,
  with the row count agreeing, so `schema/validate.py` refuses it by name and
  points at the backends that can serve it (cp2k's TDDFPT with `RKS_TRIPLETS`,
  Molcas' dual RASSCF). Same-spin multi-state work, including MECI with NAC, is
  unaffected.
- **Gradients**: From `FORCE_{i}.out` for each state (skips header line and the leading row-index column)
- **NAC**: From `NACME_{i}_{j}.out` for each coupling pair, then multiplied by (E_b − E_a) so every skill returns the same CI-derivative convention as the molcas skill (BAGEL exports d_ij already divided by the gap). The gap printed in the log (`Energy gap is:` right after `NACME Target states:`) takes precedence; `ENERGY.out` states only serve as a fallback
- **Hessian**: From the log `++ Hessian ++` section (6-column blocks with a row-index column)
- **Frequencies**: From the log `Freq (cm-1)` lines (blocks of six, concatenated)
- **Dipoles**: From the log dipole-moments section, `State` lines → (nstate, 3) in a.u.
- **Transition dipoles**: From the log dipole-moments section, `Transition a - b` lines, ordered by `transition_pair` → (npair, 3) in a.u.
- **Excitation energies**: `energy - energy[0]` (derived from `ENERGY.out`)
- **Oscillator strengths**: Read from the log `Oscillator strength :` lines (no derivation); the CASPT2-level dipole section takes precedence over the CASSCF one when both are printed

Coordinate parsing:
- Searches for `"atom"` lines in the log file
- Extracts x, y, z coordinates in Angstrom (matches the `angstrom: true` input flag)
- Returns first `natom` coordinates

### 6.3 Result Format

The `format_results()` function returns a dict with:
- `energy`: np.ndarray (nstate,)
- `gradient`: np.ndarray (nstate, natom, 3)
- `nac`: np.ndarray (nnac, natom, 3)
- `hessian`: np.ndarray (3*natom, 3*natom)
- `frequencies`: np.ndarray (3*natom,)
- `coordinates`: list of list of float (natom, 3) in Angstrom

For `*_dipole` / `*_transition_dipole` profiles, `parse_log()` additionally returns:

- `dipole`: np.ndarray (nstate, 3) — state dipole moments
- `transition_dipole`: np.ndarray (npair, 3) — transition dipoles, one row per
  declared `transition_pair` (the framework's table; with nothing declared the
  skill's convention is `[[0, i] for i in 1..nstate-1]`, no self pair). A
  declared pair whose transition the log does not print is refused by name, not
  zero-filled (#448).
- `excitation`: np.ndarray (nstate,) — excitation energies relative to state 0
- `osc_strength`: np.ndarray (npair,) — oscillator strengths printed in the same
  dipole section, keyed by the same pair order; an empty table gives an empty
  axis, not a row of zeros

Missing individual data files (e.g. a `FORCE_i.out` for an untargeted state) are
treated as zero-valued results; a missing or failed log file instead raises, so a
crashed calculation can never be reported as successful.

The `fin_velocity` return channel is reserved with an empty value: the bagel
skill performs no skill-internal MD propagation that reports final velocities,
so the channel stays empty to keep every skill result contract aligned.

For `geometry_optimization` profiles, `parse_log()` additionally returns:

- `fin_coord`: np.ndarray (natom, 3) in Angstrom — the final optimized geometry.
  A converged run prints the final structure in a `*** Geometry ***` block at
  the end of the log (all `"atom"` blocks are collected in groups of `natom`,
  the last full group is kept); when the log carries no complete geometry
  block the parser falls back to `opt.molden` (`[Atoms] Angs` section,
  coordinates already in Angstrom). Runs interrupted by an `ERROR` line are
  rejected by the completion check before any extraction, so a partially
  optimized structure in `opt.molden` from a crashed run is never parsed
  (strict failure policy)
- `opt_converge`: bool — BAGEL prints no distinct non-convergence marker, so a
  run that passes the completion check (no `ERROR` lines) is reported as
  converged; pending refinement after cluster-side validation

---

## 7. BAGEL Runtime Parameters

Configured in `qualification/pmd_env.yaml` under `skills.bagel_skill`:

| Variable | Description | Default |
|---|---|---|
| `bagel_path` | BAGEL installation directory | `''` |
| `bagel_project` | Default project name | `BAGEL` |
| `bagel_nproc` | Number of MPI processes for a local run; the skill's own slurm script uses the granted `$SLURM_NTASKS` instead (its header states `slurm_ntasks`) | `1` |
| `blas` | BLAS library path | `''` |
| `lapack` | LAPACK library path | `''` |
| `boost` | Boost library path | `''` |
| `mpi` | MPI library path | `''` |
| `mpiexe` | MPI launcher executable. The repository standardizes on `mpirun` (#305); a site that needs another spelling (`mpiexec.hydra`, an absolute path) sets it explicitly, and `no`/`none` runs BAGEL directly without MPI | `mpirun` |
| `mkl` | MKL library path | `''` |
| `arch` | Architecture for MKL source | `''` |
| `keep_tmp` | Keep temporary files (0=no, 1=yes) | `0` |
| `thread`, `bagel_thread` | Threads per rank, exported as `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`BAGEL_NUM_THREADS` in both execution modes (#305); `SLURM --cpus-per-task` is allocated from `nproc x thread`. The plain name is what a standalone input writes, `bagel_thread` what the framework forwards from `&bagel`; the larger of a pair wins (#306, which retires `num_thread`) | `1` |
| `scratch_path` | Scratch directory path | `''` |
| `slurm_time` | SLURM time limit | `01:00:00` |
| `slurm_partition` | SLURM partition name | `default` |
| `slurm_ntasks` | SLURM number of tasks | `1` |
| `slurm_memory` | SLURM memory in MB | `4000` |
| `slurm_command` | Custom SLURM command | `''` |

These defaults are also defined in `schema/reference.py` and can be overridden by the config
passed to the backend at initialization.

---

## 8. BAGEL Execution Scripts

Both local and SLURM scripts share the same environment block and
run line (`run_templates._bagel_run_commands`). Two separate keywords
drive the two parallelism axes: `mpiexe` plus the process count decide
how many ranks run, and `thread` decides how many threads each rank
gets (#305). The process count follows the allocation's owner: the
skill's own slurm script uses the granted `$SLURM_NTASKS` (its header
states `slurm_ntasks`), while a local script uses `bagel_nproc`
directly. The local script must not read `$SLURM_NTASKS`: inside a
harness job (the qualification's grouped submission) that variable is
the harness allocation, and reading it launched ten ranks for a
two-rank skill case and timed the BAGEL cases out (round
`tmp_20261111_132628`). Before #305 the thread variables were not
keyword-driven at all: MPI mode pinned them to 1 while the SLURM
header already allocated `--cpus-per-task` from the same keyword, and
direct mode read them off the rank expression -- a process keyword
answering a thread question.

### 8.1 MPI Mode (default, `mpiexe: mpirun`)

BAGEL starts under the MPI launcher, one rank per process, and each rank
gets `thread` threads (1 is the keyword default, which is why the
block below reads 1):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BAGEL_NUM_THREADS=1
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BLAS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH
mpirun -np 1 $BAGEL/bin/BAGEL project.json > project.log
```

That block is the local script (`bagel_nproc`). The skill's own slurm
script writes the same line with `${SLURM_NTASKS:-<bagel_nproc>}`
because its header states that task count.

Any launcher executable can be selected via `mpiexe` (the default is
`mpirun`; alternatives such as `mpiexec.hydra` are opt-in); only the
executable name is given, the
`-np` rank count is appended by the template.

### 8.2 Direct Mode (`mpiexe: no/none`)

Without an MPI launcher BAGEL runs directly -- one process, no ranks to
spread -- and the thread variables still come from `thread` (#305):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BAGEL_NUM_THREADS=1
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BLAS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH
$BAGEL/bin/BAGEL project.json > project.log
```

The block above shows the keyword default. A direct run that wants N
threads says `thread: N` (or `bagel_thread: N` through the framework, the
larger of the two winning, #306); it no longer inherits its thread count from
`bagel_nproc` or `$SLURM_NTASKS`, which is what made a process keyword
answer a thread question. In SLURM mode the scripts additionally carry the
`#SBATCH` header (partition, time, ntasks, cpus-per-task, mem, job name,
output/error files); `--cpus-per-task` is allocated from `nproc x thread`, so
the header and the exports agree.

---

## 9. BAGEL Test Cases

Test configurations are defined in `tests/` as YAML files:

| File | Profile | Method |
|---|---|---|
| `casscf_energy.yaml` | casscf_energy | casscf |
| `casscf_gradient.yaml` | casscf_gradient | casscf |
| `casscf_nac.yaml` | casscf_nac | casscf |
| `casscf_frequency.yaml` | casscf_frequency | casscf |
| `xmscaspt2_energy.yaml` | xmscaspt2_energy | xmscaspt2 |
| `xmscaspt2_gradient.yaml` | xmscaspt2_gradient | xmscaspt2 |
| `xmscaspt2_nac.yaml` | xmscaspt2_nac | xmscaspt2 |
| `xmscaspt2_frequency.yaml` | xmscaspt2_frequency | xmscaspt2 |
| `casscf_geometry_optimization.yaml` | casscf_geometry_optimization | casscf |
| `casscf_meci.yaml` | casscf_geometry_optimization (opttype meci) | casscf |
| `xmscaspt2_geometry_optimization.yaml` | xmscaspt2_geometry_optimization | xmscaspt2 |
| `xmscaspt2_meci.yaml` | xmscaspt2_geometry_optimization (opttype meci) | xmscaspt2 |
| `casscf_dipole.yaml` | casscf_dipole | casscf |
| `casscf_transition_dipole.yaml` | casscf_transition_dipole | casscf |
| `xmscaspt2_dipole.yaml` | xmscaspt2_dipole | xmscaspt2 |
| `xmscaspt2_transition_dipole.yaml` | xmscaspt2_transition_dipole | xmscaspt2 |

The former per-skill `dev_tool.py` was replaced: `qualification/certification/converter.py` owns the skill test YAML parsing (`parse_test_yaml` / `generate_test_configs`), renders the classic `.input` (`convert`) and serializes the agent `.config` (`convert_to_pyrai_config`), so these YAMLs are read by `pyrai3md test`.

---

## 10. Key Differences from Other Backends

1. **JSON-based input**: BAGEL uses `.json` files with `{"bagel": [...]}` structure
2. **Angstrom I/O**: The molecule block sets `"angstrom": "true"`, so coordinates go in and come back in Angstrom (no conversion)
3. **Density fitting**: Requires `df_basis` parameter for Coulomb/exchange approximation
4. **Archive files**: Uses `.archive` files for orbital management (guess/converged naming)
5. **nclosed + active**: Simpler active space model (no RAS partitioning like Molcas)
6. **No SOC support**: Only NAC between same-spin states (no spin-orbit coupling)
7. **Smith algorithm**: XMS-CASPT2 uses the Smith algorithm with `ms`, `xms`, `sssr` flags
8. **Frozen orbitals**: CASPT2 Hessian and CASPT2 forces/optimize use `frozen=True` for orbital freezing
9. **Method arrays**: forces/hessian/optimize blocks embed the method as an array of definitions; a bare dict triggers `unknown method`
10. **Separate output files**: ENERGY.out, FORCE_*.out, NACME_*.out, opt.molden (optimization geometry); Hessian, frequencies and dipole moments are printed in the log, not exported as files
11. **Those file names are fixed, so their contents are not attributable to one run**: every job writes `ENERGY.out` into the same working directory, and a finite-difference job (the XMS-CASPT2 Hessian case, `xmscaspt2_frequency_test.log` of round `tmp_20261014_185533`) prints no energy line at all in its log, leaving whatever the last displaced point wrote. `log_parser._parse_energy` therefore warns by name when it takes energies from `ENERGY.out` instead of the log, rather than passing the value off as this run's.
12. **Library paths**: Requires explicit BLAS, LAPACK, Boost, MPI, MKL paths
13. **Dipole via forces**: dipoles/transition dipoles come from a `forces` block with the `dipole` keyword plus `ciderivative: "false"` grads entries (force → state dipole, nacme → transition dipole), not a dedicated property block
