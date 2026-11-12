# ORCA Skill Development Guide

Last updated: 2026-11-02 10:40 (#448: section 1.3 gained the read-through rule (the declaration arrives already filled and the skill classifies it instead of enumerating its own sorted ladder), the sort-is-a-check rule, the two engine limits becoming refusals, and why a frequency profile carries no `gradient` step; section 5.2's SOC and NAC header lines were corrected against the readings recorded in 1.3 (they still named the headers #447 disproved) and the transition-row rule added. Sections 2-4 and 6-8 were not re-read. Before it, 2026-11-01 21:05 (#447: section 1.3 gained the SOC mirror of the NAC ground-state restriction, and the rule that a requested pair the log prints no element for is refused instead of recorded as 0.0 - both the `soc_pair_map` filter and the reader changed; the profile table's `tddft_nac`/`tddft_soc` rows now list the `gradient` step #446 restored, with the `irootlist`/`trootlist` it buys. Before it, 2026-11-01 15:50 (#446: section 1.3 gained the engine's refusal of NACME beside point charges, quoted with its return code and source line, and the rule that the switch is then not written; the parser no longer reports a zero coupling for a log that printed none. Sections 2-5 were not re-read). Sections 2-5 were not re-read again. Before it, 2026-10-31 03:12 (§4's `%tddft` bullet gained the `nroots` authoring rule: the derivation `max(nsinglet - 1, ntriplet)` yields 0 for a single-state run and ORCA refuses `%tddft nroots 0` while scanning the input, exit code 3, before any calculation starts — read off the remote refusal of `multiscale_oniom2_orca_charge_mep`, whose centre is now `method: dft`. Only that bullet was re-checked. Before it, 2026-10-31 00:47 (the point-charge carrier is a `%pointcharges "{project}.pc"` file reference, not an inline block: §1.1 item 4 and the §4 embedding bullet were checked against `service/builder/input_builder.py` and `service/templates/orca_templates.py`, and against the remote ORCA refusal `Error: Quotes expected after scanning %pointcharges`. The rest was not re-audited. Before it, 2026-10-22 22:30: the jobtype-alias sentence now matches `JOBTYPE_ALIASES`: `freq`/`frequence` to `frequency` while `hess` maps to the distinct `hessian` jobtype; the profile contract is cited at `query.py::resolve_profile` with the framework call site; the generator is the qualification converter. Before it, 2026-10-05 01:05))

This document covers **ORCA-specific** implementation details for orca_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. ORCA-Specific Design Considerations

### 1.1 Text Keyword + Block Input Format

ORCA uses a single text input file (`{project}.inp`) composed of:

1. A main keyword line starting with `!` (functional, basis, job keywords)
2. Simple blocks (`%pal`, `%maxcore`, `%tddft`)
3. An inline geometry block terminated by `*`
4. A one-line `%pointcharges "{project}.pc"` reference when the run is
   embedded (ORCA refuses the charge rows inline; see §4)

```
! pbe0 def2-svp EnGrad TightSCF

%pal
  nprocs 4
end
%maxcore 1000
%tddft
  nroots 3
  irootlist 1, 2
end
* xyz 0 1
  C      0.153069584    0.000000000   -3.629753391
  ...
*
```

Unlike molcas_skill (separate `.xyz` file), orca_skill **inlines** the geometry.
Coordinates are written in **Angstrom** (framework unit, ORCA default), so no
unit conversion is applied.

### 1.2 Method Scope

Only `dft` and `tddft` are supported. The legacy `sf_tddft` workflow from
qc_orca is intentionally **not** ported.

### 1.3 NAC Restriction (S0 -> SN only)

ORCA TDDFT NACME supports couplings between the ground state and excited
states. The schema enforces this:

- `target_nac_pair` entries that do not include state 0 raise a `ValueError`
  during normalization.
- When `target_nac_pair` is not provided, normalize derives
  `[[0, i] for i in 1..nstate-1]`.

- **The declaration the framework hands this skill is read, not rebuilt
  (#448).** `schema/adapter.py` forwards `coupling`, `nac_pair_map` and
  `soc_pair_map` into the skill section (measured on the loading chain: a
  framework run arrives with all three already filled). `verify_state_spin`
  therefore classifies the declared pairs with its own
  `classify_declared_pairs` - the same rule as
  `data/center.py::derive_pair_declaration`, kept local because a skill must
  not import framework internals - and refuses a disagreement between
  `coupling` and the two maps. What it used to do was call
  `build_spin_pair_maps(state_spin)` unconditionally, which is the full cross
  product of its own sorted ladder: nine SOC pairs where the input named two,
  re-numbered by an order the framework never chose. The enumeration stays for
  what it always was - a standalone input that declared nothing.
- **A declared ladder is checked, never permuted.** ORCA's canonical order is
  ascending 2S, and sorting a list that comes with declared pairs re-numbers
  every pair while the framework indexes the returned rows by the
  declaration's order (`workflow/integrator/fssh.py`). Two sorts of one list
  are two index spaces. So `state_spin` is sorted only when nothing was
  declared; with a declaration present, a non-ascending list is refused by
  name. The same rule now holds on molcas, cp2k, pyscf and openqp.
- The two engine limits are refusals rather than filters now. A declared NAC
  pair that does not touch state 0, and a declared SOC pair that does, are
  both rejected in `verify_state_spin`; the old code dropped them silently, so
  the run reported fewer rows than the input asked for.
- A frequency profile carries no `gradient` step (#448). `dft_frequency` used
  to list `['energy', 'hessian', 'gradient']` while `writes_engrad` returns
  False for any frequency workflow - the engine will not put `EnGrad` beside a
  Hessian keyword - so the step was inert and the parser's `is_frequency`
  guard left the pre-allocated zeros in place: the run reported flat surfaces
  it had never differentiated. The compound jobtype that claimed otherwise,
  `energy frequency gradient`, is out of the vocabulary
  (`schema/method.py::_JOBERTYPE_SUFFIX_MAP`), so writing it by hand now fails
  at profile resolution. A run that wants a Hessian and a gradient together
  goes through `num_freq`, where the framework takes the gradients itself by
  finite difference.
- `nac_pair_map` keeps only same-spin pairs containing state 0.
- A single charge-free deck cannot carry both a coupling and a gradient:
  measured in the test environment (round `tmp_20261101_221306`, ORCA 6.1.1) a deck
  with `EnGrad` **and** `NACME true` finishes the TD-DFT part and then aborts
  with `!!! CANNOT OPEN FILE Filename: <project>.nacme.tmp !!!`, return code 64.
  `service/utils.py::writes_engrad` therefore suppresses `EnGrad` /
  `irootlist` / `trootlist` exactly when `NACME` is written, and
  `writes_nacme` suppresses `NACME` when the run carries charges - one
  predicate pair that the builder, the TDDFT result packer and the gradient
  gate all read, so a deck and its parser cannot disagree about whether a
  gradient block exists. With charges the coupling is the missing one (that is
  the crossing-search deck, and its gap gradient is real); without charges the
  gradient is, and the run reports no gradient rows rather than a zero surface.
- The SOC reader's header was wrong, and the zeros hid it. The packer used to
  read `soc_matrix.get((a, b), 0.0)`, so a log it could not parse came back as
  six zero couplings instead of an error. The table ORCA 6.1.1 actually prints
  is `CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS`, one row per
  (triplet root, singlet root) with `(Re, Im)` for the Z, X and Y components
  (measured on the same round); the reader now parses that layout, keeps the
  norm over the three components, and maps the columns to framework indices
  with `_map_orca_socme_pair` (the triplet column is 1-based, the singlet
  column 0-based with 0 = S0) instead of the assumed combined numbering.
- The NACME vectors are read from the `CARTESIAN NON-ADIABATIC COUPLINGS`
  block (per-atom `<GS|d/dx|ES>` rows, measured on round
  `tmp_20261101_224544`), one block per requested root **in deck order**. The
  reader used to look for `Coupling vector between states i and j`, a string
  ORCA does not print, so it matched nothing on a real log; the packer now
  refuses a block count that differs from the requested pair count, because a
  block names no root and its position is its only identity. `irootlist` is
  part of that request: without it ORCA computes the coupling of root 1 only,
  which the same round measured (one block for a two-pair request).
- The mirror rule on the SOC channel (#447): DoSOC numbers **excited** roots, so a
  pair through state 0 has no element to read. `soc_pair_map` therefore keeps only
  pairs that exclude state 0 (`schema/normalize.py`, beside the NAC filter), and a
  requested pair the log prints no element for is refused by name. Both sides of
  that used to be answered with `0.0` — the pair list carried ground-state pairs
  whose value was fabricated, and `0.0` reads to an ISC rate as "these two states do
  not couple". The placeholder arrays the parser starts from hold no rows either:
  a row exists only where the engine printed one.
- NACME is also incompatible with embedded charges, and the engine says so at input check: `Error (ORCA_CIS): The NACMEs are currently not compatible with point charges!` (`main_input_check.cpp:10261`), return code 25. A run that carries background charges therefore does not get the switch written, and the builder warns in the run log; the step then reports no coupling row. Measured in the test environment, where an ONIOM2 `meci` leg with computed outer charges asked for `energy gradient nac`, wrote `NACME true`, and the engine aborted it (#446).

### 1.4 num_freq Keyword

`num_freq` (boolean) selects the frequency keyword for ground-state
`dft_frequency` runs:

| num_freq | ORCA keyword | meaning |
|----------|-------------|---------|
| false (default) | `AnFreq` | analytical frequencies |
| true | `NumFreq` | numerical frequencies |

TDDFT has no analytical second derivatives in ORCA, so `tddft_frequency`
always emits `NumFreq` regardless of `num_freq`.

The framework additionally maps the jobtype aliases `freq` and `frequence` to
`frequency`, while `hess` maps to `hessian` - a distinct basic jobtype, not a synonym of
`frequency` (see `pyrai3md/schema/validate.py:47-55` `JOBTYPE_ALIASES`).

---

## 2. ORCA Calculation Profiles

### 2.1 Complete Profile List (14 profiles)

| Profile | Capability | Steps | Extra ORCA keywords |
|---------|-----------|-------|---------------------|
| dft_energy | energy | energy | TightSCF |
| dft_gradient | gradient | energy, gradient | EnGrad |
| dft_dipole | dipole | energy, dipole | (dipole from output) |
| dft_opt_min | geometry_optimization | optimize | Opt |
| dft_opt_ts | geometry_optimization | optimize_ts | OptTS |
| dft_frequency | vibrational_frequency | energy, gradient, hessian | AnFreq / NumFreq |
| tddft_energy | energy | energy, tddft | %tddft nroots |
| tddft_gradient | gradient | energy, tddft, gradient | EnGrad + irootlist/trootlist |
| tddft_transition_dipole | transition_dipole | energy, tddft, transition_dipole | %tddft nroots |
| tddft_nac | nonadiabatic_coupling | energy, tddft, gradient, nac | `NACME true` inside %tddft, `irootlist`/`trootlist` from the gradient step |
| tddft_soc | soc_coupling | energy, tddft, gradient, soc | Triplets true + DoSOC true (RHF singlet reference), `irootlist`/`trootlist` from the gradient step |
| tddft_opt_min | geometry_optimization | tddft, optimize | Opt |
| tddft_opt_ts | geometry_optimization | tddft, optimize_ts | OptTS |
| tddft_frequency | vibrational_frequency | energy, tddft, hessian | NumFreq (always numerical) |

### 2.2 Profile Resolution

`query.py::resolve_profile` (implemented at `schema/method.py:128`) derives the profile
when it is not explicitly given; the framework calls it through the skill's public query
interface (`data/core.py:2498`, which falls back from the 3-arg to the 2-arg form):

- `energy` -> `_energy`, `gradient` / `energy gradient` -> `_gradient`
- `frequency` -> `_frequency`, `optimize` -> `_opt_min`
- `dipole` -> `_dipole`, `absorption` / `transition` -> `_transition_dipole`
- `nac` -> `_nac`, `soc` -> `_soc`, `md` -> `_gradient`

Incompatible combinations (e.g. `dft` + `soc`) raise `ValueError`.

---

## 3. ORCA Schema Details

### 3.1 Calculation-Specific Fields

| Field | Default | Description |
|-------|---------|-------------|
| functional | pbe0 | DFT functional |
| basis | def2-svp | Basis set (validated against AVAILABLE_BASIS) |
| tddft_nroots | derived | max(excited singlets, triplets); pure-singlet runs keep `nstate - 1` |
| num_freq | false | ground-state switch: false=AnFreq, true=NumFreq; TDDFT always NumFreq |
| density_fitting | true | Enable density fitting (RIJCOSX + aux_basis) |
| aux_basis | def2/J | Auxiliary basis for density fitting |
| dispersion_correction | d3 | Grimme dispersion correction (d3/d3zero/d4; empty or 'none' disables) |
| grid_level | 2 | Integration grid level (0 = ORCA default, 1-3 -> DEFGRID1-3) |
| tda | false | Use TDA instead of full TDDFT |
| scf_maxit | 200 | SCF MaxIter (`%scf MaxIter`) |
| target_states | [0] | States computed in TDDFT gradients |
| target_nac_pair | derived | Only pairs including state 0 are allowed |
| target_soc_pair | [] | Usually derived from state_spin via soc_pair_map |

### 3.2 Environment-Specific Fields

| Field | Default | Description |
|-------|---------|-------------|
| orca_path | '' | ORCA installation directory (must contain `orca`) |
| orca_project | ORCA | Input/output file basename |
| nproc, orca_nproc | 1 | `%pal nprocs` ranks; the plain name is what a standalone input writes, the prefixed name what the framework forwards, and the larger of a pair wins (#306) |
| thread, orca_thread | 1 | OpenMP threads per rank (`OMP_NUM_THREADS`); the SLURM header asks `--cpus-per-task` for `nproc x thread` |
| orca_mem | 1000 | `%maxcore` memory per core in MB |
| mpi | '' | MPI library path for parallel ORCA |
| scratch_path | '' | 'AUTO' -> /srv/tmp or /tmp |

### 3.3 Normalize Workflow

1. `verify_electronic_structure`: electron count / multiplicity parity.
2. `verify_state_num`: target_states validation, NAC pair restriction (S0 only),
   `tddft_nroots = max(nsinglet - 1, ntriplet)` (ORCA computes that
   many roots of each multiplicity; pure-singlet runs keep `nstate - 1`),
   `nsing_root`/`ntrip_root` counts, zero-filled state arrays.
3. `verify_state_spin`: singlet systems allow singlet+triplet mixing (SOC);
   builds `nac_pair_map` (same-spin, S0 only) and `soc_pair_map` (different spin).

---

## 4. Input Builder Details

`service/builder/input_builder.py` assembles `{project}.inp`:

- Keyword line: `functional basis [RIJCOSX aux_basis] [D3BJ] [EnGrad] [Opt|OptTS] [AnFreq|NumFreq] PrintBasis TightSCF`
  - `EnGrad` is added for gradient profiles but skipped for frequency and optimization profiles: EnGrad is mutually exclusive with Opt/OptTS (ORCA sees EnGrad and overrides the optimization task); optimizations drive the gradient machinery internally
  - `PrintBasis` is always emitted (basis function table for inspection)
  - SOC keeps the closed-shell singlet (RHF) reference; ORCA rejects any
    other reference for DoSOC (`main_input_check.cpp`). Triplet target
    states enter through `Triplets true` in the `%tddft` block
  - `NACME` is not a main-line keyword; it is written inside the `%tddft`
    block as `NACME true` (ORCA rejects it on the keyword line), and is not
    written at all when the run embeds point charges (section 1.3)
- `%scf` block: `MaxIter {scf_maxit}` and always `Print[P_MOs] 1`
  (MO coefficient dump).
- `%tddft` block: `nroots {tddft_nroots}`; gradient roots are split
  between `irootlist` (singlet roots, root 0 = ground state) and `trootlist`
  (triplet roots, 1-based within the triplet roots),
  both ascending; ORCA prints one gradient block per requested root
  in the order `irootlist[1:] + trootlist + [0]` when the ground
  state is requested, otherwise `irootlist + trootlist`; `Triplets
  true` is emitted whenever the state list contains triplet roots
  (SOC adds `DoSOC true`).
  - `nroots` must be at least 1, and the derivation gives 0 for a
    single-state run: a case that walks one surface (an mep band, a
    ground-state optimization) therefore has to ask for `method: dft`.
    ORCA refuses `%tddft nroots 0` while scanning the input, before any
    calculation starts, and the process exits with return code 3.
- Frequency runs write `{project}.hess` automatically (no `%output
  Print[P_Hessian]` block: ORCA rejects it on the keyword/block level).
- Embedding charges are read from a side file: the input carries
  `%pointcharges "{project}.pc"` and `write_input_file` writes that file
  beside the input. The file opens with the charge count, then one
  `q x y z` row per charge (charge in atomic units, position in
  Angstrom, the frame the geometry block uses). ORCA rejects the rows
  inline (`Error: Quotes expected after scanning %pointcharges`,
  `orca_main/main_input_master.cpp`), which is what a remote round
  measured before this form was adopted. Charges taken from a file keep
  their mutual interaction out of the nuclear energy, matching the
  embedding split: the classical provider owns that term.

---

## 5. Result Parsing

### 5.1 Output Files

| File | Content | Parser |
|------|---------|--------|
| `{project}.engrad` | Energy (Eh) and gradient (Eh/bohr) of the state whose gradient was computed | engrad_parser |
| `{project}.out` | Main log (all sections) | log_parser |
| `{project}.property.txt` | Ground-state SCF energy, per-root total energies, additive corrections (all dispersion-free) | property_parser |
| `{project}.hess` | Hessian matrix (frequency profiles) | log_parser.parse_hessian |

### 5.2 Log Parsing Rules (log_parser.py, resolved by output_parser)

- **Reference energy:** `FINAL SINGLE POINT ENERGY` (and the `.engrad`
  `$energy`, which carries the same number) is the total energy of the
  state whose nuclear gradient was printed last. That is the ground
  state for a DFT gradient, but an excited root for an `EnGrad` +
  `irootlist n` run, so it must never be used as the S0 reference of a
  TDDFT profile: doing so double-counts the excitation energy and
  makes an NVE trajectory look like it loses energy at exactly the
  rate the state gap varies. TDDFT energies resolve in this order:
  1. `{project}.property.txt`: `$CIS_Energies &e0` plus
     `$VdW_Correction &vdW` gives the ground state, and `&totalEnergy`
     labelled by `&Root` gives each excited state (all values are
     dispersion-free, so the corrections are added back). Accepted only
     when one corrected root reproduces `FINAL SINGLE POINT ENERGY`,
     which proves no unenumerated correction (solvation, relativity)
     is in play.
  2. The log's `CIS/TD-DFT TOTAL ENERGY` block: `E(SCF)` plus the
     labelled `DE(...) (Root n)` lines, with the corrections recovered
     by subtracting the matching root's excitation from the final
     single point energy.
  3. Neither usable: `STATE ... <S` lines positionally on top of the
     final single point energy (historic behaviour).
  A recovered correction larger than `MAX_CORRECTION` (0.02 Eh) is
  treated as a mis-read and step 3 applies instead.
- `E(SCF)` and the `DE(...) (Root n)` lines are reported separately as
  `escf_energy` and `root_excitations`; `Dispersion correction` becomes
  `dispersion`. The `STATE ... <S` lines stay the fallback ladder and
  are collected by their printed root label, not by position.
- TDDFT gradients: one `CARTESIAN GRADIENT` block per requested root,
  collected in log order and distributed along the print order
  `irootlist[1:] + trootlist + [0]` (ground state requested) or
  `irootlist + trootlist` otherwise; triplet entries carry the state
  offset `nsing_root + root`; a block-count mismatch fails fast
- Dipole: `Total Dipole Moment` line (e*bohr)
- Transition dipoles / oscillator strengths / excitation energies:
  `ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS` section. The two
  transition channels are extracted **per declared pair**, in
  `transition_pair` order, and a run with no declared pair has no row at all:
  the packer used to size them `len(pairs) if pairs else 1`, which answered a
  run that asked for no transition property with one zero dipole and one zero
  oscillator strength (#448). The default table the framework derives is
  `[[0, i] for i in 1..nstate-1]` — the state-to-ground convention, without
  the self pair `(0, 0)` the old `range(nstate)` spelling opened with.
- SOC: `CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS` table (one row per
  triplet/singlet root pair, `(Re, Im)` for Z, X and Y; the triplet column is
  1-based and the singlet column 0-based, mapped onto framework indices by
  `_map_orca_socme_pair`); values kept in cm^-1 (ORCA native unit, no
  conversion). A requested pair the table prints no element for is refused.
- NAC: `CARTESIAN NON-ADIABATIC COUPLINGS` blocks (`<GS|d/dx|ES>` per atom), one
  block per requested root in deck order; the block count must equal the
  requested pair count, since a block names no states and its position is its
  only identity. The old `... between states a and b` header never appeared on
  a real log, and the label shift built on it (#447).
- Frequencies: `$vibrational_frequencies` block of `{project}.hess`
  (mode count line, then `<index> <cm^-1>` rows, the layout the legacy
  wigner sampler read), falling back to the log's `VIBRATIONAL FREQUENCIES`
  section; both missing fails fast
- Optimization: final geometry from `{project}.xyz` (ORCA updates it every
  cycle; the input geometry only lives in `{project}.inp`), falling back to
  the last `CARTESIAN COORDINATES (ANGSTROEM)` log block; convergence from
  `OPTIMIZATION RUN DONE`

Missing sections stay zero-filled, keeping the `(nstate, ...)` shape contract
shared with the other skills.

### 5.3 Return Channels (optimize profiles)

Optimization profiles (`dft_opt_min`, `dft_opt_ts`, `tddft_opt_min`,
`tddft_opt_ts`) report their outcome through the framework channels:

- **fin_coord** `(natom, 3)`, Angstrom: the relaxed structure read from
  `{project}.xyz` (ORCA updates it every cycle), falling back to the
  last `CARTESIAN COORDINATES (ANGSTROEM)` log block; a missing final
  geometry or an atom-count mismatch fails fast.
- **opt_converge**: bool, settled from the `OPTIMIZATION RUN DONE`
  marker in the log.
- **fin_velocity**: reserved return channel, always empty (the orca
  skill performs no skill-internal MD propagation); kept so every
  skill result contract stays aligned.

---

## 6. Runtime Parameters and Execution

Local script (`{project}.sh`) and SLURM script (`{project}.sbatch`) both export
only `ORCA_PROJECT`, `ORCA` and `MPI`, prepend `$MPI/lib` and `$MPI/bin`,
then run (`service/templates/run_templates.py`):

```bash
$ORCA/orca $ORCA_PROJECT.inp > $ORCA_PROJECT.out 2>&1
```

The `2>&1` merge is deliberate: ORCA reports input/keyword errors on
stderr, and losing them would leave an empty failure log.

SLURM headers use `slurm_partition/slurm_time/slurm_ntasks/slurm_memory`;
a user-provided `{project}.slurm` template overrides the headers.

---

## 7. Test Cases

`orca_skill/tests/` ships 14 YAML cases (ethylene geometry, default pbe0/sto-3g):

`dft_energy`, `dft_gradient`, `dft_dipole`, `dft_frequency`, `dft_opt_min`,
`dft_opt_ts`, `tddft_energy`, `tddft_gradient`, `tddft_transition_dipole`,
`tddft_nac`, `tddft_soc` (singlet+triplet state_spin for DoSOC),
`tddft_opt_min`, `tddft_opt_ts`, `tddft_frequency`.

Generate agent configs with:

```bash
# Agent configs come from the qualification converter
# (qualification/certification/converter.py: parse_test_yaml / generate_test_configs),
# which replaced the former per-skill dev_tool.py; the test modes run through
# `pyrai3md test`.
```

---

## 8. Key Differences from Other Backends

| Aspect | molcas_skill | orca_skill |
|--------|--------------|------------|
| Input style | Module chain + separate .xyz | Keyword line + blocks, inline geometry |
| Coordinate unit in input | Bohr (Molcas) | Angstrom (ORCA default) |
| SOC mechanism | Dual RASSCF + RASSI | RHF singlet reference + Triplets/DoSOC in %tddft |
| NAC scope | Same-spin pairs (RASSI) | S0-SN pairs only (NACME) |
| Gradient source | ALASKA | EnGrad file / CARTESIAN GRADIENT blocks |
| Hessian source | MCKINLEY | .hess file (written automatically by Freq/NumFreq) |
