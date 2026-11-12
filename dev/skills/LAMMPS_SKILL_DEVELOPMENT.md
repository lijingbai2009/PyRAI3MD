# LAMMPS Skill Development Guide


Last updated: 2026-10-31 23:20 (#445 收口). Section 12.2's guard paragraph was replaced with the deployed round's own numbers: recomputing `h2o_qmmm_field.esfp` against that run's `c_QMMM_GG[1]` gives a residual under 3.9e-6 kcal/mol on a scale of 134.4 (2.9e-8 relative, 34x inside a `1e-6 * scale` budget), while the ewald box leaves 2.2e-4 against that same budget, short by 1.3x, so tightening is decided against on measurement and the guard stays at `1e-4 * scale`; the same round's decks carry `kspace_style ewald 1e-8` and `kspace_style pppm 1e-8`. The whole-directory count is now 539 passed. Before it, 2026-10-31 18:20:
`ewald_accuracy` was added as the skill's own knob for the reciprocal-sum
accuracy — §3.1 gains its row, §3.2 its validation rule, and §12.2 was
rewritten from what the deployed engine (23 Jun 2022 Update 1) actually does:
the pin's job is converging the two probes, not equalising their grids, ewald
silently ignores `kspace_modify mesh`, and the key now reaches every
`kspace_style` line the builders write, classical decks included.  Before it, the same day at 02:55 (#444): Sections 11.5 and 11.7, the resident
file table and Section 12.7 were re-read against
`service/persistent/launcher.py`, `backend.py` and the two persistent test
files: the live-service claim is keyed on the socket path, not on the
execution namespace, because an isolated-workdir batch gives every unit its
own calculation directory under one shared namespace and the parent
optimization loop holds a service of that namespace while its numerical
Hessian runs. The counts in 12.7 and in the test table are the measured ones
(537 passed over the whole directory). Before it, 2026-10-30 17:05 (#443). Sections 12.3, 12.4 and 12.8, the file
table and the profile table were re-read against `backend.py`,
`service/parser/qmmm_output.py`, `service/parser/output_parser.py` and
`workflow/records/conversion.py`: the skill now returns the REDUCED classical
surface as `gradient` (and as `{title}.grad`), negates the engine's force into
that label itself, and declares only its units to the framework; the composite
no longer subtracts anything, it adds `qmmm_el_gradient` back when
`System.qm_contains_field_gradient` is false. Before it, 2026-10-30 12:20:
Sections 12.3 and 12.8 recorded the new switch. Before that, 2026-10-29 21:27: Section 12.2 documents the alpha-only pin
(`kspace_modify gewald 3.2/rc` plus a probe accuracy of 1e-6) that keeps
the A/B probes on one Ewald split (the unpinned pair used
`0.29013089/6³` against `0.2567432/4³` and disagreed with `group/group`
by 0.13 kcal/mol), the 17-digit probe dumps, and `tools/verify.py`
section H. Before it,
2026-10-28 23:30. Section 12, the profile table and the output-file table were re-read against the tree for the two standalone QM/MM tasks (`qmmm_field`/`qmmm_energy`), their separate builders/parsers/writers, native force output and the framework conversion, protocol-v2 resident A/B and per-step q_qm, and the 14+4 QM/MM tests (full LAMMPS skill suite 509 passed). The old single-evaluation QM/MM contract and its `.espf`/TOTAL labels are superseded.

---

## 1. LAMMPS-Specific Design Considerations

### 1.1 Classical Force Field Skill

lammps_skill is the classical molecular mechanics (`mm` role) skill of PyRAI3MD.
It drives LAMMPS through a plain text input script (`{project}.in`-style), a
LAMMPS data file for the topology/coordinates, and auxiliary include files.
All results are converted to the framework contract units (energy in hartree,
gradient in hartree/bohr) regardless of the LAMMPS unit system (`real` or `metal`).

The skill covers the full classical lifecycle: single-point energies, analytic
gradients (parsed from atomic forces), in-engine geometry optimization and
three-ensemble MD (NVE / NVT / NPT), the latter two returning final coordinates
**and** velocities through the `fin_coord` / `fin_velocity` channels.

### 1.2 Optional Main Template, Mandatory Data File

The input file layout splits static system description from per-task execution:

- `{project}.data` — **mandatory**. Topology, masses, atom types, coordinates
  and (optionally) velocities. Loaded with `read_data`. Validated by
  `schema/validate.py` with an explicit error message when missing.
- `{project}.in` — **optional main template**. May carry user force field
  definitions (pair_style, pair_coeff, bond/angle coefficients, kspace, etc.)
  and any user-defined fix definitions. When absent, the builder generates the
  complete input script from the jobtype alone (`input_builder.write_input_file`).
- `{project}.in.init` / `{project}.in.settings` — optional include files,
  included before/after the template body when present. For the bundled h2o
  test system they are required data files (staged by the qualification
  converter, see section 9.2), but the builder treats them as optional includes.

This split lets the skill reuse one static system description across all seven
profiles (`schema/reference.py::AVAILABLE_PROFILES`: `ff_energy`, `ff_gradient`,
`ff_optimize`, `ff_optimize_cell`, `ff_nve`, `ff_nvt`, `ff_npt`) while generating
per-task execution sections programmatically.

Since the templates are libraries rather than per-system files, the backend
first compresses them into a JSON force-field model holding only the
referenced types (§5.6); the three files above stay the user-facing input and
are never modified (contract #142).

### 1.3 Per-Task Output Naming

`service/utils.py::output_basename(project, profile)` derives one basename per
task; the `.in`, `.log`, `.sh`, `.sbatch` and all dump files of a task share it:

| Profile | basename |
|---|---|
| ff_energy | `{project}_ff_energy` |
| ff_gradient | `{project}_force` |
| ff_optimize | `{project}_ff_optimize` |
| ff_nve | `{project}_ff_nve` |
| ff_nvt | `{project}_ff_nvt` |
| ff_npt | `{project}_ff_npt` |

ff_gradient deliberately uses the `_force` suffix (matching the QM skills'
force-file convention); all other profiles use `{project}_{profile}`.

### 1.4 Element-Free Dump Contract

All trajectory/force dumps are written with `write_dump` in `custom`/`xyz`
style **without** `modify element` and with `modify sort id`, i.e. atoms are
ordered by atom index and carry no element symbols. The parser therefore
ignores the first column of dump frames (atom type number) and relies on the
system `atoms` list for element identity. This keeps dumps deterministic and
parser-stable across LAMMPS builds that differ in element-name handling.

### 1.5 Per-Atom Thermo Energy and the norm Restore

The generated scripts emit `thermo_modify norm yes`, so LAMMPS reports
**per-atom** potential/total energies in the log. The parser restores the
extensive value by multiplying with the atom count before the unit conversion
(see section 6). Any hand-written template that changes this behaviour must be
kept consistent with the parser expectation.

### 1.6 Failure Propagation

A non-zero LAMMPS return code is propagated as a `RuntimeError` by the backend
before parsing, so a crashed run can never be masked by zero-filled results.

---

## 2. Calculation Profiles

| Profile | Capability | Driver | Description |
|---|---|---|---|
| ff_energy | energy | skill | Single-point energy (`run 0`) |
| ff_gradient | energy, gradient | skill | Single-point energy + forces (`run 0` + force dump) |
| ff_optimize | geometry_optimization | skill | In-engine `minimize` |
| ff_optimize_cell | geometry_optimization | skill | In-engine `minimize` with `fix box/relax` beside it: the lattice is the engine's variable and the answer box comes back on the `.cell` channel |
| ff_nve | classical_dynamics | skill | MD with `fix nve` |
| ff_nvt | classical_dynamics | skill | MD with `fix nvt` (Nosé–Hoover thermostat) |
| ff_npt | classical_dynamics | skill | MD with `fix npt` (thermostat + barostat) |
| qmmm_field | energy, gradient (probe) | skill | A/B `run 0` Coulomb probe at q_Q=+1; per-Q V/E |
| qmmm_energy | energy, gradient | skill | One full-FF `run 0` with q_qm; Python in-cell/gas and qV/qE assembly |

Declared capabilities: `energy`, `gradient`, `stress`, `geometry_optimization`,
`classical_dynamics` (`manifest.yaml:16-22`). Role registration in `manifest.yaml`:
`mm: LammpsBackend` (`qm`/`qm2` are false). Contract units: energy `hartree`,
gradient `hartree/bohr`, stress `bar` (`manifest.yaml:12`), which the parser
normalizes from the engine's atm/bar reading through `PRESSURE_TO_BAR`
(`service/utils.py:82`). Ground state only (`nstate = 1`).

**Framework profile resolution (`query.resolve_profile`).** The framework only
upgrades a skill's profile away from its schema default when the skill's
`query` module exposes `resolve_profile` (gate at `data/core.py`). Without it a
framework-driven dynamics step (jobtype `md`) would stay at the default
`ff_energy` and never request forces, so the gradient comes back all zero and
the trajectory freezes. `query.resolve_profile(method, jobtype)` therefore
delegates to `schema/method.py::resolve_profile`, whose `_JOBBERTYPE_SUFFIX_MAP`
maps `md` and `energy gradient` to `ff_gradient`. Note that framework-driven MD
uses the single-point `ff_gradient` profile (the framework integrator owns the
ensemble propagation and queries LAMMPS for energy + forces each step); the
standalone `ff_nve`/`ff_nvt`/`ff_npt` profiles are only selected for skill-internal
dynamics. Every skill must expose this hook.


---

## 3. Schema Details

### 3.1 Default Parameters (`schema/reference.py` DEFAULT)

Minimization parameters:

| Parameter | Default | Meaning |
|---|---|---|
| min_style | cg | LAMMPS minimization style (`cg`, `sd`, `fire`, `hftn`) |
| min_etol | 1e-4 | Energy tolerance for `minimize` |
| min_ftol | 1e-8 | Force tolerance for `minimize` |
| min_maxiter | 1000 | Max minimization iterations |
| min_maxeval | 1000000 | Max force/energy evaluations |

MD parameters:

| Parameter | Default | Meaning |
|---|---|---|
| md_nstep | 100 | Number of MD steps |
| md_timestep | 1.0 | Timestep in fs (real) / ps-scale via units |
| md_seed | 12345 | RNG seed for `velocity all create` |
| dump_freq | 100 | Thermo output frequency (`thermo {dump_freq}`) |
| thermo_damp | 100.0 | Temperature damping parameter (fs), used by nvt/npt |
| pressure | 1.0 | Target pressure (atm) for npt |
| pressure_damp | 1000.0 | Pressure damping parameter (fs) for npt |

LAMMPS-specific parameters:

| Parameter | Default | Meaning |
|---|---|---|
| lammps_units | real | LAMMPS unit style (`real` or `metal` only) |
| lammps_shake | 0 | 0 = no constraint, 1 = keep/inject shake fix (rejected for framework-integrated profiles, §5.5) |
| shake_fix_id | fShakeTIP3P | Fix id used by a template-defined shake fix |
| shake_keywords | (empty) | Raw `fix ... shake ...` command injected when no template fix exists |
| lammps_project | LAMMPS | Default project name |
| nproc, lammps_nproc | 1 | MPI ranks for `mpirun -np`: the plain name is what a standalone input writes and what the skill reads, `lammps_nproc` what the framework forwards from `&lammps`; the larger of a pair wins (#306) |
| thread, lammps_thread | 1 | OpenMP threads per rank, exported as `OMP_NUM_THREADS`; same two spellings (#306) |
| lammps_path | (empty) | Empty = `lmp` from PATH; directory = `{path}/bin/lmp`; else used directly |
| ewald_accuracy | 1e-8 | The accuracy token this skill writes onto the `kspace_style` line it re-issues after `read_data`; the reciprocal **style** stays whatever the template declared. See §12.2 - this is not a performance dial, it is what makes the periodic QMMM field probe's two runs agree |

### 3.2 Validation Rules (`schema/validate.py`)

- `{project}.data` must exist in the working directory; the error message
  points at the missing data file.
- `{project}.in` is explicitly documented as optional in the validator.
- `lammps_units` must be `real` or `metal`.
- `ewald_accuracy` must parse as a number strictly inside (0,1), after
  `strip().lower()`; a non-numeric token or a value outside that range is
  refused here rather than at the engine, because it is written verbatim into
  `kspace_style`. `schema/config.py:_normalize_accuracy` first folds whatever
  YAML produced into that token: `1e-4` arrives as a string and `0.0001` as a
  float, and they are the same request. There is no "keep the template's
  value" spelling - an empty value is re-filled with the default before the
  checker sees it, so a case wanting a loose sum writes the loose number.
- `lammps_path`, when pointing at a directory, must contain `bin/lmp`.
- `lammps_shake = 1` requires `shake_fix_id` or `shake_keywords`, **and**
  an engine-integrated profile: a constraint applied by LAMMPS while the
  framework propagates the atoms would not match the integration
  (§5.5).

---

## 4. Normalize Workflow

LAMMPS is a ground-state-only skill. `schema/normalize.py` pins the state
model before validation:

- `nstate = 1`, `target_states = [0]`, `state_label = ['s0']`
- NAC and SOC coupling requests are cleared (not available for classical FF)

---

## 5. Input Builder Details

`service/builder/input_builder.py` assembles one input script per task.

### 5.1 File Assembly (`write_input_file`)

Header order (each include line is emitted only when the file exists):

```
include "{project}.in.init"
{main template body, if {project}.in exists}
read_data "{project}.data"
kspace_style ...        # replayed from {project}.in.init, if it defines one
include "{project}.in.settings"
```

When `{project}.in` is absent, `_read_template` returns an empty string and the
builder relies on the generated execution section plus the data file to define
the system (the h2o test system keeps its force field in the include files).

**kspace re-issue after `read_data`.** LAMMPS invalidates a `kspace_style`
that was defined before `read_data` once the simulation box becomes triclinic,
aborting with `Must redefine kspace_style after changing to triclinic box`.
A variable-cell (NPT) step rewrites the data-file box with tilt factors
(section 5.4), so `_extract_kspace_line(project)` pulls the user's
`kspace_style` command verbatim from `{project}.in.init` and the header replays
it immediately after `read_data`. The replay is harmless for an orthorhombic
box (NVT / single point) and is shared by both the single-point and the
internal-iteration write paths.

### 5.2 Single-Point Profiles (ff_energy / ff_gradient)

The template body is kept and the execution tail is appended:

```
thermo_style custom step pe etotal
thermo 1
thermo_modify norm yes
run 0
variable pyrai_pe equal pe
print "${pyrai_pe}" file {title}.energy
write_dump all custom {title}.coords x y z modify sort id
```

ff_gradient adds the force dump, a `stress: true` request the six stress
variables plus one `print` line (section 11.7a). Every name comes from
`step_data_files(title, …)`, so the four per-step result files share their
stem with `{title}.log`, `{title}.sh` and `{title}.in`:

```
write_dump all custom {title}.force fx fy fz modify sort id
```

The parser reads these files and never the thermo table, which is why the
energy is a `print`ed variable rather than a `thermo` row: the table carries
8 significant digits, `print` carries 15 (section 11.7a).

### 5.3 Internal-Iteration Profiles (ff_optimize / ff_nve / ff_nvt / ff_npt)

For profiles that iterate inside LAMMPS, `_strip_execution_lines` first removes
template-provided execution commands (`thermo*`, `dump`, `velocity`,
`timestep`, `fix`, `unfix`, `run`, `minimize`, `min_style`, ...) so the
generated section is the single source of execution control.

Optimization section:

```
min_style {min_style}
minimize {min_etol} {min_ftol} {min_maxiter} {min_maxeval}
write_dump all xyz {project}_ff_optimize.xyz modify sort id
write_data restart.data
```

MD section (shared skeleton, ensemble-specific fix):

```
thermo_style custom ...
thermo {dump_freq}
thermo_modify norm yes
velocity all create {temperature} {md_seed}     # only when the system has no velocities
timestep {md_timestep}
fix PYRAI_MD all nve                            # ff_nve
fix PYRAI_MD all nvt temp {T} {T} {thermo_damp} # ff_nvt
fix PYRAI_MD all npt temp {T} {T} {thermo_damp} iso {P} {P} {pressure_damp} ...
                                                # ff_npt: x/y/z pressure control, drag 2.0,
                                                # plus fix PYRAI_RECENTER all recenter INIT INIT INIT
run {md_nstep}
unfix PYRAI_MD
write_dump all custom {project}_{profile}.xyz  id x y z modify sort id
write_dump all custom {project}_{profile}.cond id x y z vx vy vz modify sort id
write_data restart.data
```

`restart.data` is **not** a PyRAI3MD continuation entry. It is the node data
file the agent writes when it orchestrates a sequence of standalone LAMMPS
calls; the framework never reads it, and `read_data` always points at
`<project>.data` (ruled 2026-10-20). The `.cond` dump above, not this file, is
the source of the returned velocity and cell channels.

The `.cond` dump carries positions **and** velocities and is the source of the
`fin_velocity` channel (section 6.4). It also carries the box: every custom
dump opens its frames with `ITEM: BOX BOUNDS`, in the **bounding plane** frame
and at `%.16e`, so it is the source of `fin_cell` too (section 6.4). The
`fix npt` spelling above carries the three cell lengths as degrees of freedom,
which means the run can end in a box the input never named while its returned
coordinates already belong to that box; until this channel existed the number
sat unread in the file and the state kept propagating the old cell.

`bound_and_tilt_to_lattice` in `service/builder` is the exact inverse of
`lattice_to_true_box` followed by `box_to_bound_numbers`, and `tests/test_fin_cell.py`
holds the round trip against four lattices (orthorhombic, right-angle float
noise, positive and negative shear) plus the box a real test run printed.
The reduction from a lattice to six cell parameters is deliberately **not**
implemented here: the skill reports the nine flattened components, and
`utilities/lattice.matrix_to_cell6` on the framework side is the one place that
formula lives -- a skill may not import it, so repeating it per skill would have
put three copies of an angle convention in play for no gain.

`restart.data` is written by the same tail and is still read by nothing; the
cell comes from the dump, which is the frame the coordinates came in.

### 5.4 Data File Update (`write_data_file`)

Each call reads the user template `{project}.data` (never modified in place,
contract #142) and writes the per-step copy into the calcdir:

- **Simulation box** (`_rewrite_data_box`): when the run is periodic (`pbc`
  on, resolved by `_resolve_periodic_lattice`), the box is rebuilt from the
  framework `lattice_vector` — the template's `lo` origin is kept and
  `hi = lo + length`, so a variable-cell (NPT) step resizes the box instead
  of keeping the stale template size (CHANGELOG #216, PBC §3.3); the box
  therefore always matches the coordinates of the same evaluation (the
  `calculate(cell=)` frozen snapshot channel). The
  lower-triangular framework convention maps to `lx/ly/lz` plus the tilt
  factors `xy = B[1][0]`, `xz = B[2][0]`, `yz = B[2][1]`, written to the
  LAMMPS bounding-box convention; the tilt line is written self-labelled
  (`'<xy> <xz> <yz> xy xz yz'`, CHANGELOG #223) and both the labelled and
  the bare three-float spellings are detected on read, so a triclinic
  template is rewritten instead of getting a second tilt line. The line is
  dropped when the cell becomes orthorhombic. An upper-triangular lattice
  component above tolerance **raises** instead of being silently dropped:
  the framework re-triangularizes variable-cell commits
  (`verlet._scale_cell`), so such a lattice means the convention was
  violated upstream. A non-periodic run
  keeps the template box. The shared h2o fixtures use `kspace_style pppm`
  (triclinic-capable; `ewald` rejects tilted boxes).
- **Failure diagnostics**: a LAMMPS run whose log carries no thermo energy
  line raises with the last ~20 log lines embedded, so a remote abort is
  diagnosable without fetching the calculation directory.
- **Coordinates**: each `Atoms` line is updated by **atom id** (`id-1` is the
  framework index), replacing only the coordinate columns — no longer the
  blind last-three-columns, which corrupted lines carrying trailing image
  flags. The atom_style is read from the `Atoms # <style>` header comment
  when present; otherwise a heuristic resolves the coordinate columns
  (trailing integer tokens are image flags, preserved) and fails fast on an
  ambiguous line. id/mol/type/charge columns and image flags are preserved.
  An id set that does not map one-to-one onto `1..natom` raises.
- **Velocities**: the `Velocities` section is refreshed when the system
  carries velocities.

### 5.5 Shake Handling (`_build_shake_section`)

A geometric constraint has to be applied by whatever integrates the
positions, so there are two separate control sources and each owns the
constraint only on its own side: `lammps_shake` while LAMMPS propagates,
the framework's `constraint` block while the framework propagates
(`dev/methods/CONSTRAINT_DEVELOPMENT.md` §3.7). For the
framework-integrated profiles (`ff_energy`, `ff_gradient`, which hand one
force evaluation per step to the framework) `lammps_shake = 1` is
rejected by `schema/validate.py`, and the message points at the
framework `constraint` block rather than leaving the request nowhere to
go. The
engine-integrated profiles (`ff_optimize`, `ff_optimize_cell`,
`ff_nve`, `ff_nvt`, `ff_npt`, listed in `schema/method.py:80-81`
`ENGINE_INTEGRATED_PROFILES`)
let LAMMPS move the atoms, so a shake fix stays consistent there.

The framework side is not a drop-in replacement for a LAMMPS shake fix.
Its default solve is on positions only: the bond length is held to
1e-12 relative, while the velocity constraint is taken up only by a
record that declares `velocity_constraint: rattle`, and the measured cost
of the position-only path on the bundled water box is recorded in
`qualification/verification/test_constraint_shake_dynamics_cost.py` and §1.3 of the
constraint document.

- **With a force-field model** (the default): the decision is recorded
  once in `model['constraints']` and the rendered settings either carry
  the renumbered `group`/`fix` lines or nothing at all. The deck emits
  **neither `fix` nor `unfix`**, which removes the historical split
  where the `fix` lived in `{project}.in.settings` while the matching
  `unfix` was appended by the builder.
- **Without a model** (a template shape the model refuses, §5.6): the
  legacy mechanism applies — `lammps_shake = 0` emits
  `unfix {shake_fix_id}` when the templates define such a fix, and
  `lammps_shake = 1` keeps it, injecting `shake_keywords` when no fix
  is defined.

Note that dropping the constraint is not free: the TIP3P O–H stretch
then has to be resolved by the timestep itself. With the fixture's
`k = 450 kcal/mol/A^2` and a reduced mass of 0.9476 amu the period is
about 14 fs (≈2360 cm⁻¹), so `md.size = 41.34` au (1 fs) gives roughly
14 points per period. Halve the timestep rather than re-enabling the
constraint if the NVE total-energy drift grows.

### 5.6 Force-Field Compression (`service/builder/ff_model.py`)

A shipped force-field template is a *library*: the bundled water box
declares 111 atom / 952 bond / 5355 angle / 1004 dihedral / 38 improper
types while 60 atoms reference only 2 atom, 1 bond and 1 angle type.
LAMMPS materialises its pair table from the **declared** type count —
that template generates 6102 of 6105 mixed `pair_coeff` terms per call —
and echoes the 8622-line settings file into its log on every invocation,
so a framework MD loop pays that fixed cost once per step (measured
0.9–1.2 s/step against a LAMMPS `Total wall time` of 0:00:00).

`build_ff_model(config, profile)` therefore analyses the three templates
once, at backend initialization, into a single JSON model
(`{project}.ff.json`, written **beside** `tmp_{SKILL}` so `keep_tmp`
cannot remove it):

- the referenced type sets and an order-preserving renumbering onto
  `1..N` (LAMMPS requires contiguous type ids);
- the topology rows (id / molecule / type / charge, bonds, angles) and
  the masses of the referenced types only — coordinates are *not*
  stored, they are live state;
- the coefficients, **deduplicated last-write-wins over their sub-style**.
  This is not a cleanup: `{project}.in.settings` appends the re-fitted
  block (`pair_coeff 76 76 lj/charmm/coul/long 0.102 3.188`, the Price &
  Brooks 2004 TIP3P parameters matched to `kspace_style pppm`) over
  generated placeholders (`0.0 1.0 # (default parameters)`), so the
  later line is the one that must survive. Only the *same* sub-style
  re-fitted for the same pair shadows anything: under a hybrid wrapper
  two rows of one type pair carrying different sub-styles are two
  parameters, so the sub-style name is part of the dedup key. Every
  superseded line is recorded in `audit.dropped_duplicates` with both
  line numbers, and each rendered coefficient keeps a `# src L<n>`
  provenance comment;
- the constraint state (§5.5);
- `audit.dropped_unused_coeffs` for the library entries no atom
  references.

`write_input_files` renders `{project}.data`, `{project}.in.init`,
`{project}.in.settings` and the deck from the model on every step, so a
deleted calcdir is rebuilt from the JSON alone. The per-step data writer
(`write_data_file`, §5.4) is unchanged: it only replaces the coordinate
columns and preserves id/mol/type/charge, so the renumbering rides in
the rendered template and needs no mapping logic of its own.

**Refusal over guessing.** Anything the model cannot prove equivalent
raises `FFModelUnsupported` and the backend falls back to the verbatim
templates with a warning: a wildcard or range `pair_coeff`, a used type
with no coefficient or no mass entry, a command outside the census
(`pair/bond/angle/dihedral/improper_coeff`, `group`, `fix`), a
`Velocities` section, a missing count line, or an unrecognised
data-file line. Dihedral and improper entity rows are in scope: only
their type column renumbers and the atom ids of a row are copied as
written, so they ride the same path as bonds and angles. A cross
`pair_coeff` row counts as present when the init template asks LAMMPS
to derive it (`pair_modify mix arithmetic|geometric`) and both types
carry a diagonal row of the same sub-style; without that statement, or
with a type whose two diagonal rows name different sub-styles, the pair
is still required explicitly. Both branches are asserted in
`tests/test_ff_model.py`.

**The model owns the style declarations.** LAMMPS requires every
sub-style a `*_style hybrid` wrapper lists to be assigned to at least one
type pair, and aborts otherwise ("Pair hybrid sub-style is not used").
Trimming the coefficients of unused types therefore starves the wrapper:
the reference template declares `buck/coul/long` (its nine rows belong to
library types 101-106), `dihedral_style hybrid fourier` and
`improper_style hybrid cvff`, none of which survive a 60-atom TIP3P box.
So `{project}.in.init` is rendered from the model rather than copied: the
line is split into `[sub-style, arguments]` segments using the names the
coefficient rows of that family actually carry (the declaration alone is
ambiguous, because each sub-style brings its own cutoffs), and segments
with no surviving row are dropped. A family left with no sub-style and no
type to evaluate becomes `*_style none`, which is LAMMPS' own default; a
family that still has types to evaluate is refused instead. Every other
init line, and the whole file when nothing is starved, is byte-identical
to the template, and the wrapper keeps its exact spelling
(`hybrid/overlay` is never rewritten as `hybrid`).

Because the compression only removes unreferenced library entries and
makes the existing override order explicit, it is **numerically
transparent**: the same geometry must give identical energy, forces and
stress with and without it. Compression is a mandatory stage rather than a
switch; the verbatim templates remain only as the degradation path a
refused model falls back to.

### 5.6a Subsystem Selection (`atom_selection`)

A 2-layer multi-center plan is `L(real) + Σ_c [H(c) − L(c)]`, so with
`theory: [pyscf, lammps]` each `L(c)` is a classical calculation on **one
center's atoms** while `L(real)` is one on the whole system. The user ships a
single system-level `{project}.data`; the skill therefore has to be able to say
what its templates look like when only some of the rows survive. That is this
key's whole job, and the framework's whole contribution is the list of row
numbers (dev/MULTICENTER_DEVELOPMENT.md section 11.3, dev/methods/
PBC_DEVELOPMENT.md section 4.2).

- **`atom_selection: List[int]`** (`schema/config.py::System`, registered in
  `schema/reference.py::DEFAULT`). 1-based ids of the topology template this
  task computes, in the order the framework hands the sliced geometry.
  Empty means the whole system, which is every path that existed before the
  key. `is_subsystem(model)` is the single reader of the fact.
- **Position, not value, is what renumbering follows.** The contract is
  `write_data_file`'s mapping — coordinate row `i` goes to the row whose id is
  `i + 1` — so `_apply_selection` numbers the template as
  `new_id = position + 1`, `old_id = selection[position]` and adds no sorting
  of its own. Whether the incoming list is sorted is not the skill's business:
  a center is a SET of atoms, so the **framework** adapter
  (`pyrai3md/schema/adapter.py:257` `_parse_center_indices`, sorting at `:309-316`)
  canonicalises its indices to ascending at input time for every carrier (yaml
  nested list, classic string, programmatic assignment) - the skill's own
  `schema/adapter.py` has no such helper and never sorts, and
  `ff_model.py:419-437` `_selection_of` keeps the order it is handed - and
  `regions.py::materialize_region` and the static slice in `data/core.py` both
  index in that one order. The skill therefore always receives ascending rows
  in practice, and stays faithful to whatever it is given — which is what
  `test_the_new_numbering_follows_the_handed_order` pins with a deliberately
  non-ascending selection.
- **What a cut keeps.** `Atoms` rows of the selected ids in the handed order
  (only the id column changes; molecule, type and charge travel verbatim);
  an entity row in `Bonds`/`Angles`/`Dihedrals`/`Impropers` survives only when
  **all** its members were selected, with its member columns renumbered and
  its row id renumbered. A row the cut breaks is *dropped and recorded* in
  `model['audit']['torn_entities']` with the ids that fell outside: an
  interaction crossing the region boundary belongs to `L(real)` and would be
  subtracted twice if the subsystem carried it too. The declared counts are
  rewritten to the surviving row counts, and the existing
  `_build_type_maps` / `_prune_styles` / `_renumber_coeffs` / `render_data`
  chain re-derives the type census, the Masses section and the coefficient
  table from those rows, so compression is reused rather than duplicated.
- **Cache safety.** A selection is part of the model identity:
  `_hash_config` appends the ordered row list when there is one, and
  `load_cached_model` additionally refuses a cache whose recorded
  `model['selection']` differs from the request. Two centers of the same size
  and the same atom types have identical counts and an identical type map, so
  the selection is the only field that says which subsystem a session is, and
  `compute_fingerprint` therefore hashes it too.
- **The key is absent for a whole-system run.** `build_ff_model` writes
  `selection` only when one was asked for, so `{project}.ff.json` of an
  unchanged run is byte-identical to the generation before the key existed;
  this is why the 24 golden input digests of
  `tests/test_input_builder_golden.py` did not move, and it is verified rather
  than assumed.

**A subsystem is computed non-periodically.** The difference term must leave
the center's *self* interaction in place: kept periodic, `L(c)` carries the
cluster's Ewald self-image energy as well, and subtracting it removes something
`L(real)` never contained. Measured on the shipped 60-atom water box, for one
isolated water evaluated three ways at the same geometry — the composite's own
term, the skill's generated input, and a hand-cut 3-atom file — the periodic
treatment reads 0.280090 kcal/mol (**4.464e-4 Ha**) below the isolated one per
center, and the isolated reading equals the analytic bond-and-angle energy of
that water to 2e-15 kcal/mol. So for a subsystem the skill emits
`boundary f f f` (before `read_data`, which LAMMPS requires — it refuses the
command once the box exists), drops the `kspace_style` line from the rendered
init, and suppresses the per-step kspace re-issue through one producer
(`_reissue_kspace`) so the subprocess header and the resident package cannot
drift.

**Coulomb degradation is a measured table, not a guess.** Every `…/coul/long`
sub-style needs a kspace command, and LAMMPS refuses it outright without one
("Pair style requires a Kspace style"), so the swap is mandatory. What it maps
to, and how many style arguments the sibling takes, was read off the binary
(this build is 22Jul2025):

| declared | real-space sibling | style arguments |
|---|---|---|
| `lj/charmm/coul/long rmin cutoff [inner]` | `lj/cut/coul/cut rmin cutoff` | first two kept, `inner` dropped |
| `buck/coul/long cutoff` | `buck/coul/cut cutoff` | kept |
| `coul/long cutoff` | `coul/cut cutoff` | kept |

The numbers are copied from the template verbatim, because the template's own
cutoff is what the difference term must reproduce. Two measured facts decide
where they go: at a pair distance of 9.5 Å, `lj/cut/coul/cut 6.0 12.0` reads
the bare Coulomb while `12.0 6.0` reads exactly zero (with and without
`hybrid`), so the *last* style argument given governs the interaction range and
keeping charmm's first two numbers puts the declared cutoff in charge; and
`lj/cut/coul/cut` takes at most two style arguments, so charmm's three cannot
be copied through. Per-pair rows are renamed, never re-shaped: `lj` keeps
(eps, sigma), `buck/coul/cut` keeps (A, rho, C) exactly as its long-range twin
(a two-number row is refused by the engine), `coul/cut` takes none.

Anything not in the table is **refused by name** with `FFModelUnsupported`,
which leaves the verbatim templates and stops the run rather than silently
changing an energy form — including `lj/cut/coul/long`, whose argument order
this build has not been measured to agree with its `_cut` sibling. Refusal is
not cosmetic here: `lj/charmm/coul/long` has **no same-family real-space
sibling** in this build (`lj/charmm/coul`, `lj/charmmfsw`, `lj/charmmfsh` are
all absent), so the swap changes the combining rule as well — CHARMM always
mixes Rmin arithmetically and epsilon geometrically, while `lj/cut/coul/cut`
honours the template's `pair_modify`. For the shipped water that is inert to
the last digit (every LJ epsilon the template declares is 0.0, and `coul/cut`,
`lj/cut/coul/cut` and `buck/coul/cut` all read the identical
1.2124000038669500 kcal/mol for one molecule); for a force field with real
epsilon it is not, which is why the change is written into
`model['audit']['coulomb_degradation']` and named here instead of being
buried in the rendered text.

**The container is fixed, the atoms move.** A subsystem's geometry is the
continuous view of its region, so a molecule torn by a box face arrives with an
atom beyond the template's bounds, and a non-periodic LAMMPS run refuses atoms
outside its box (measured: "Did not assign all atoms correctly" at
`read_data`). `_subsystem_box` therefore writes one fixed container —
`±_SUBSYSTEM_BOX_HALF` (50 Å) about the origin on all three axes, tilt line
blanked — instead of a box fitted to where the cluster happened to be. The fit
was the first implementation and the deployment falsified it: a resident
instance carries the box it was initialised with while the integrator keeps
moving the atoms, so at a later step the center was outside its own container,
and with more than one MPI rank an escaped atom is not refused but lost from the
domain decomposition (measured on `examples/multicenter_pbc_nve.yaml`:
"ERROR on proc 0: Bond atoms 1 3 missing on proc 0"). Nothing in what the engine
computes depends on the number — the model carries `boundary f f f` and no
kspace command — so it is chosen to be far larger than any cell this framework
carries rather than to be tight. A coordinate outside even that container is
refused: it means the region rebuild is no longer handing a molecular cluster
of this cell, which is a geometry defect, not a box to be grown.

**Verification of the whole channel, end to end.** Driving
`examples/multicenter_pbc_nve.yaml` for one step with PySCF at two centers and
LAMMPS as the low layer, and re-evaluating the coordinates the backend was
actually handed: the composite's `L(c)` for the two three-row centers agrees
with the skill's own generated input, with a hand-cut file and with the
analytic bonded energy to 1.6e-16 Ha. The positional rule is pinned beside the
refusals — `test_the_new_numbering_follows_the_handed_order` here, and
`TestTheRegionTaskIsHandedItsRows` at plan level, which declares a center
`[[3,1,2],[6,4,5]]` and checks it comes back `[[0,1,2],[3,4,5]]` through all
three input carriers.

---

## 6. Result Parsing

### 6.1 Output Files

| File | Source | Parsed for |
|---|---|---|
| `{title}.energy` | `print` of the `pe` variable, single point | energy (extensive, fifteen digits) |
| `{title}.force` | ff_gradient force dump | gradient |
| `{title}.stress` | `print` of six stress variables | stress_tensor |
| `{title}.coords` | per-step geometry dump | record only |
| `{basename}.log` | LAMMPS stdout (run script redirect) | energies of the internal-iteration profiles |
| `{project}_{profile}.xyz` | optimize/MD coordinate dump | fin_coord |
| `{project}_{profile}.cond` | MD coordinate+velocity dump | fin_velocity, fin_cell |
| `{project}_{profile}.cell` | `ff_optimize_cell` only: the `print "CELL …"` line the deck writes | the box the engine stopped on, returned as `fin_cell` (`output_parser.py:603`, written by `input_builder.py:1272-1284`) |
| `{title}.esfp` | `qmmm_field` writer | V/E in atomic units (Molcas ESPF), rows `id V Ex Ey Ez` |
| `{title}.qm_multipole` | `qmmm_energy` writer | received q_qm (`id q`, e) |
| `{title}.qmmm` | `qmmm_energy` writer | `E_classic`, `U_main`, `U_gas`, `qmmm_el` (native) |
| `{title}.u_gas.force` | `qmmm_energy` writer | QM rows of `in_cell_qq` (native) |
| `{title}.qmmm_el.force` | `qmmm_energy` writer | QM rows of `q_qm*E` (native) |
| `{title}.grad` | `qmmm_energy` writer | the REDUCED classical gradient, all atoms, `dU/dR` (native) |

All LAMMPS profiles return native numbers under the framework's labels:
`energy` in the run's native energy unit and `gradient` in the native
gradient unit, already negated (`gradient = dU/dR`, never a force). The
framework scales them by the units its manifest declares. The standard
`.energy`/`.force` and auxiliary files stay in LAMMPS units; `.esfp` is the atomic-unit
handoff to Molcas.

`{basename}` and the `{title}` of the four result files both follow
`output_basename(project, profile)` (section 1.3); section 11.7a records which
profile reads which file, and why the single point does not read the log.

### 6.2 Dual-Format Log Parsing (`service/parser/log_parser.py`)

LAMMPS prints thermo output either as a column table
(`Step ... PotEng ... TotEng` header followed by data rows) or in multi-line
form (`PotEng = value`). The parser supports **both** formats and, when a log
contains both (e.g. a template switching `thermo_style` mid-run), takes the
record that appears **last** in the file. The last data row of the selected
format provides the final potential/total energies.

### 6.3 Energy norm Restore

Because of `thermo_modify norm yes` (section 1.5), a thermo record is
per-atom. The parser restores the extensive energy:

```
energy = parsed_value * natom * ENERGY_TO_HARTREE
```

where `ENERGY_TO_HARTREE` converts kcal/mol (real) or eV (metal) to hartree.
This multiplication belongs to the log channel and therefore to the
internal-iteration profiles only: `{title}.energy` holds LAMMPS's own total
potential energy, so the single point converts units and never scales by
`natom`.

### 6.4 fin_coord, fin_velocity and fin_cell Channels

- **fin_coord**: the last frame of the xyz dump (`_parse_xyz_dump`). The first
  frame column (atom type number) is ignored per the element-free dump
  contract; frame atom count is checked against the system.
- **fin_velocity** (MD profiles only): the `.cond` dump carries
  `vx vy vz` in LAMMPS units. The parser converts them back to the framework
  Bohr/au with `LAMMPS_VELOCITY_TO_AU` (inverse of `VELOCITY_TO_LAMMPS`:
  real → Å/fs ≈ 21.877 Bohr/au; metal → Å/ps).
- **Engine-side cell relaxation (`fix box/relax`) is written, and it landed on an engine.** The
  grammar is the one the test build answers to, taken from the engine rather than from
  current online documentation (LAMMPS 23 Jun 2022 - Update 1, the version string in the
  eighth round's own `water25_force.log`; probe records in `CHANGELOG.md` #362): the
  styles that parse are `iso P`, `aniso P`, `x P`, `y P`, `z P` - and `aniso` takes
  **one** value there, `aniso P P` and `aniso P P P` are refused (`fix_box_relax.cpp:215`).
  `xy/xz/yz` and `tri` parse but are refused at fix setup on an orthorhombic box
  (`Can not specify Pxy/Pxz/Pyz ... with non-triclinic box`, :287); `axis`, `aaxis`,
  `2x2`, `triclinic` and `pre` do not exist in this build. `couple` accepts
  `none/xy/xz/yz` only (`couple iso|aniso` refused at :179) and `dilate` accepts `all`
  only (`none`/`x` refused at :186); `nreset` parses. **`vmax` has to be written out
  loud**: its default caps a box step at 1e-4 relative, and measured on the 25 A water
  box, a 100 kbar target moves an edge 0.01% with the default and 3.98% with
  `vmax 0.1`. Under `run` the fix is accepted and the box does not move at all:
  `box/relax` belongs to `minimize`.
- **What the profile writes, measured on a real engine on this checkout.** The case is
  `tests/ff_optimize_cell.yaml` (the shipped h2o topology, `box_relax: aniso`,
  `box_relax_pressure: 0.0`, budget 200 steps / 2000 evaluations) and the engine is
  `lmp_serial` on the test environment. The builder emits
  `fix PYRAI_BOX all box/relax aniso 0 vmax 0.01` above `minimize 0.0001 1e-08 200 2000`,
  the run stops at timestep 121 with exit 0, and the answer box is
  `CELL 8.13313960982016 7.89538115742456 8.36566666513793 0 0 0` against the 8.399091 A
  cube the data file was written with - the three edges move independently, which is what
  `aniso` is for. The same number comes back from the engine's own `restart.data`
  (8.133139609820162 / 7.89538115742456 / 8.365666665137926), so the `print` line and the
  written box are two independent products of one state, and they agree to the last digit
  either of them carries. The control arm is `ff_optimize` on the same fixture: its deck
  carries no `fix` line, it stops at timestep 54, and it writes **no `.cell` file** - the
  profile that does not own the lattice reports no lattice. What this case does **not**
  certify is the physics of the moved box: the h2o fixture declares a 9.0/10.0 A pair
  cutoff inside a 4.2 A half-extent, so its energy is not a smooth function of its own
  cell (the header of `pyrai3md/examples/optimize_cell_lammps.yaml`), and the convergence
  vehicle for a relaxed box stays the framework-driven water25 run.
- **A mixed topology can carry a centre, and one now does (#392).**
  `pyrai3md/examples/c2h4_water25.data` (155,434 B) is one ethylene solvated by 25
  TIP3P waters: 1581 atoms, ids contiguous 1..1581, bonds, angles and dihedrals in
  the same file, and the solute as the LAST molecule - id 526 of 526, rows
  1576-1581 - which is why `optimize_cell_meci_lammps.yaml` and
  `optimize_cell_mecp_lammps.yaml` name those rows in `molecule.highlevel` while
  every other gap example puts its high region at rows 1-6. The two crossing legs
  ran on this environment (`lmp_serial` 22Jul2025): five steps in 28 s and 27 s, a box of
  25.237532 x 24.433851 x 25.611381 A whose smallest half extent, 12.22 A, stays
  above the 9.0/10.0 A cutoff, and the framework driving the lattice while LAMMPS
  answers `compute Pconf … pressure NULL virial` for it. Two things a reader has
  to get right about the trio. First, only the `.data` is the carrier's own:
  `.in.init` and `.in.settings` are byte-identical to `water25`'s (md5
  `546aa016ce6075de353bd9edb4835bc9`, `b2947b1f58c0f1311a243c48fc268456`) because
  ffauto writes one 98-type mass dictionary and one diagonal `pair_coeff i i` row
  per type, and `pair_modify mix arithmetic` supplies the cross terms - so the
  duplicate is load-bearing (the solute's types 4 and 25 are priced by a template
  `water25` never exercises), and `test_multicenter_periodic.py::
  TestTheForceFieldCarrierOfTheCrossingPair` pins both halves. Second, these two
  examples use `ff_optimize`, **not** `ff_optimize_cell`: the profile that names
  the engine as the box owner requires `box_relax`
  (`schema/validate.py:203-211` refuses either mismatch in its own direction), and
  here the lattice is the framework's variable, so the deck is the static
  single-evaluation one above.
- **A driver=main stress is the configurational part only.** When the framework drives the
  iterations and asks the engine for one evaluation per step, the deck the skill writes is
  `compute Pconf all pressure NULL virial` (`input_builder.py:1537`), not LAMMPS's own
  per-atom `stress/atom` on a fix. The `NULL` group is the point: it removes the kinetic
  term, so the six numbers the parser turns into a tensor are the potential virial, which
  is what a lattice derivative is the conjugate of. A cell criterion built on the total
  pressure would answer to a temperature the static evaluation does not have.
- **fin_cell** (MD profiles only): the box of that same `.cond` frame, as nine
  flattened lattice components in Å. `fix npt` carries the three cell lengths
  as degrees of freedom, so a run can end in a box the input never named while
  its returned coordinates already belong to it; the framework derives `cell`,
  `lattice_vector` and `volume` from those nine numbers and refuses the value
  outright for a system whose `pbc` says non-periodic, because that is an input
  error rather than a state to absorb (the `pbc` authority #235 settled). A
  fixed-cell run reports the cell it was given, which the framework notices and
  leaves alone. `ff_optimize` reports no cell at all: `minimize` has no cell
  degree of freedom, so a cell there would be an invention. The resident
  service never reports one either: it answers single-point and gradient
  profiles only, the cell is always the one the client pushed down, and
  `_verify_box` already reads it back and fails loudly if the engine holds a
  different one, so a `box` field in the RESULT would only echo a number the
  caller supplied and a mismatch the caller already learns about. `ff_optimize`
  reports none and `ff_optimize_cell` reports exactly one line: the box the
  engine stopped on, which is the distinction between the two profiles and is
  measured above.

lammps_skill is the **only skill in the project that fully implements the
`fin_velocity` channel**: velocities leave the skill in the same Bohr/au
convention the framework uses to propagate dynamics, with an exact round-trip
conversion on both the builder side (Bohr/au → LAMMPS units) and the parser
side (LAMMPS units → Bohr/au).

Both keys are transport channels, not physical properties, and the framework
consumes them the same way: `Calculator._apply_skill_geometry` and
`Calculator._apply_skill_velocities` pop the key from the result (and from the
task config the backend stored), refuse a shape that does not match `natom`,
skip a value that equals what the state already holds, and otherwise write it
into `pmd.system`. The velocity half also recomputes the kinetic energy,
because a state that carries the engine's velocities but reports the kinetic
energy of the zeros it started from contradicts itself. Applying velocities was
the missing consumer of this channel for as long as the skill produced it: a
standalone `ff_nve` run at 86 K logged zero velocities and zero kinetic energy
while its own `.cond` dump held the nonzero ones.

### 6.5 Gradient

ff_gradient parses `{title}.force` (`fx fy fz` columns, with or without
the id column), negates the forces and converts them:

```
gradient = -force * FORCE_TO_HARTREE_PER_BOHR   # shape (1, natom, 3)
```

---

## 7. Runtime Parameters

| Parameter | Effect |
|---|---|
| lammps_path | Binary resolution: empty → `lmp` from PATH, directory → `{path}/bin/lmp`, otherwise the value is used as the binary |
| nproc, lammps_nproc | `mpirun -np {nproc}` prefix; 1 rank runs the binary directly |
| thread, lammps_thread | `OMP_NUM_THREADS` in every arm: the subprocess script, the resident service launcher, and the SLURM allocation of `nproc x thread` (#304) |
| lammps_units | Selects the unit conversion constants (real/metal) and the velocity conversion scale |
| execution_mode | `local` or `slurm` run script generation |
| keep_tmp | 0 removes the calcdir after parsing, 1 keeps it |

---

## 8. Execution Scripts

`service/templates/run_templates.py` generates the local/SLURM scripts. The
command block exports the shared basename and redirects LAMMPS stdout into the
per-task log:

```bash
#!/bin/bash
export LAMMPS_INPUT={basename}
export OMP_NUM_THREADS=1
{binary} -in $LAMMPS_INPUT.in > $LAMMPS_INPUT.log          # lammps_nproc = 1
mpirun -np {nproc} {binary} -in $LAMMPS_INPUT.in > $LAMMPS_INPUT.log  # lammps_nproc > 1
```

`{binary}` comes from `_resolve_lammps_binary` (section 7, `lammps_path`).
The SLURM variant prepends the scheduler header (`#SBATCH ...`) to the same
command block. `{basename}` is `output_basename(project, profile)`, so the
input script, log, dump files and run script all share one per-task name.

---

## 9. Test Cases

### 9.1 Qualification YAML Cases (`tests/`)

Thirteen YAML cases. Eleven run the **same 60-atom periodic system** (20 TIP3P
water molecules) with coordinates embedded in each file; the two QM/MM cases add
one non-periodic 6-atom water cluster beside the periodic system. Nine of the
eleven are in the table below; the two grammar witnesses of `fix box/relax`
(`ff_box_relax_iso.yaml`, `ff_box_relax_aniso_pressure.yaml`, both
`profile: ff_optimize_cell`) follow the table and are described in section 6.4
and CELL_OPT_DEVELOPMENT.md section 8.4:

| File | Profile | Notable settings | Expected |
|---|---|---|---|
| ff_energy.yaml | ff_energy | defaults | energy shape [1] |
| ff_gradient.yaml | ff_gradient | defaults | energy + gradient shape [1, 60, 3] |
| ff_gradient_persistent.yaml | ff_gradient | `python_enabled: true`, `debug: true` | energy + gradient shape [1, 60, 3] |
| ff_stress.yaml | ff_gradient | `stress: true` | energy + gradient + stress_tensor shape [1, 3, 3] |
| ff_optimize.yaml | ff_optimize | min_* parameters | fin_coord shape [60, 3] |
| ff_optimize_cell.yaml | ff_optimize_cell | `box_relax: aniso`, `box_relax_pressure: 0.0`, min budget 200/2000 | fin_coord shape [60, 3]; the engine-side answer box is the `.cell` line, not an `expected` field |
| ff_nve.yaml | ff_nve | md_nstep, md_seed | fin_coord / fin_velocity shape [60, 3] |
| ff_nvt.yaml | ff_nvt | thermo_damp | fin_coord / fin_velocity shape [60, 3] |
| ff_npt.yaml | ff_npt | md_nstep 10, dump_freq 5, pressure | fin_coord / fin_velocity shape [60, 3] |
| qmmm_cluster.yaml | qmmm_field | `qm_atoms: [0, 1, 2]`, non-periodic cluster fixture `qmmm_cluster_test.{data,in.init,in.settings}` | probe-only energy shape [1] + gradient shape [1, 6, 3]; `.esfp` written |
| qmmm_cluster_energy.yaml | qmmm_energy | the same cluster, `q_qm: [-0.83, 0.415, 0.415]`, `embedding: true`, and a shipped Flow-1 `.esfp` named in `qmmm_field_file` (fixture `qmmm_cluster_energy_test.*`) | energy + gradient shape [1, 6, 3]; `.qmmm` decomposition, `.qm_multipole` and the reduced `.grad` written |
| qmmm_periodic.yaml | qmmm_field | `qm_atoms: [0, 1, 2]`, the 60-atom periodic box | probe-only energy + gradient shape [1, 60, 3]; `.esfp` written |

`ff_gradient_persistent.yaml` is the resident twin of `ff_gradient.yaml`:
its `input` section differs by the two resident keywords and nothing else,
so the subprocess and service paths are compared on identical inputs.

The three QM/MM cases follow the same component-energy/gradient contract
(section 12): their `expected` block is shape-only, because a QM/MM total
depends on the LAMMPS build's Ewald split. `qmmm_cluster_energy.yaml` is the
only Flow-2 case: it reads the probe's V/E from a shipped `.esfp`
(`qmmm_field_file`) because a case YAML has no in-memory field result, and
it sets `embedding: true` so the run subtracts `qmmm_el = sum q_qm.V` from
the classical side instead of reading the file and ignoring it. Their
project names are the YAML `name` values (`qmmm_cluster_test`,
`qmmm_cluster_energy_test`, `qmmm_periodic_test`); the converter stages the
shared `h2o.*` fixtures under each project name and copies each case's own
`<project>.*` topology verbatim, which is why the energy case ships
`qmmm_cluster_energy_test.{data,in.init,in.settings}` as copies of the
cluster's. The staging loop now leaves an
existing target alone, so a case-specific fixture is not overwritten by a
shared one that sorts later (the reason this guard exists is exactly two
topologies in one skill).

The pbc-declaration sweep (`test_pbc_declaration.py::_CASES`) walks eight of these
names; `ff_optimize_cell` and the two `box_relax` cases are outside that parametrized
list, and the cell they declare is pinned against the `h2o.data` fixture box by
`qualification/verification/test_unification.py` instead.

Every one of the eleven periodic cases declares `pbc: [1, 1, 1]` and
`cell: [8.399091, 8.399091, 8.399091, 90.0, 90.0, 90.0]`, and those values have to keep
agreeing with the box written in `h2o.data` (`0.0 8.39909100 xlo xhi`). The
non-periodic `qmmm_cluster` case is the deliberate exception: `pbc: [0, 0, 0]`,
`boundary f f f` and a real-space Coulomb style in its own fixture, so the
in-cell Q-Q correction has a literal minimum image. The
reason is not decoration. The framework cell is what `write_data_file` applies
when the run is periodic, and `pbc` is the single switch that says whether it
applies at all: with an undeclared (all-zero) `pbc` the builder leaves the
template's box alone, so the case silently computed a periodic Ewald sum on a
topology whose config claimed no cell and a 99 Å sentinel — and its log, its
`.json`, its `.input` and its `.config` all repeated that claim. `method=skill`
delegates the *task* to the engine, not the parameters, so a periodic topology
has to say so. `tests/test_pbc_declaration.py` pins both halves: the declared
cell equals the fixture's box, and a declared cell is what reaches the generated
data file (while an undeclared one provokes no box invention at all).
`debug: true` is what makes that comparison a file comparison: the resident
case leaves the same log and dumps as the cold one, in the same names. The
keywords must survive the planner's three artifacts (flat `.json`, classic
`.input`, agent `.config`) and the framework's skill-section merge from all
four input routes, which `test_persistent_test_plan.py` pins case by case.

### 9.2 Distributed Data Files

The former per-skill `dev_tool.py` (its `TEST_DATA_FILES` tuple and
`generate_all`) no longer exists. The staging is now the qualification
converter's: `qualification/certification/converter.py::generate_test_configs`
(`:2279`) copies every `tests/*{.data,.in.init,.in.settings}` file next to the
configs it writes, keyed by those suffixes at `:2238`, and the module's own
header calls itself "the unified replacement of the former per-skill
dev_tool.generate_all". `h2o.data` supplies the 60-atom coordinates/topology
referenced by the YAML cases' `read_data` step; the operator entry points today
are `tools/deploy.py` (submit) and `tools/verify.py` (re-read the products).

### 9.3 Unit Tests

Nineteen `test_*.py` modules, none of which needs a LAMMPS installation
(`_fake_lammps_service.py` is the stub that speaks the real interface shapes:
ctypes arrays for gather/scatter, integer style/type codes, `c_double` and
pointer returns, the seven-element `extract_box()`). The counts below are the
A-layer inventory, grouped by what a failure would actually tell you.

Input generation (what the engine is asked to do):

| File | Tests | Pins |
|---|---|---|
| `test_input_builder.py` | 62 | header assembly, optimize/MD sections, per-task input naming, template-free generation, the kspace re-issue after `read_data`, the stress file, and the shake invariants (framework-driven profiles reject a constraint the integrator cannot honour, engine profiles accept it, and a model-driven fix travels with the settings file) |
| `test_input_builder_box.py` | 18 | lattice to true box, the bounding-plane lines a deck and a data file write, the template origin lookup, the live box a step resolves, the static package |
| `test_input_builder_package.py` | 3 | one writer for both paths: cold package and resident `INIT` package are file-for-file equal under the same model |
| `test_input_builder_golden.py` | 24 | digests of the generated text. This is the over-reach detector: any edit that changes what LAMMPS is told moves it, and a round that claims not to touch the deck must leave it byte-identical |
| `test_ff_model.py` | 46 | force-field compression, hybrid sub-style takeover, the render-declaration vs coefficient self-check |

Cell arithmetic (does one box mean one thing on both sides):

| File | Tests | Pins |
|---|---|---|
| `test_box_roundtrip_parity.py` | 19 | `lattice → true box → bounding planes → lattice` is an inverse and not an approximation, on the cells the qualification runs actually reached (including the sheared and negative-tilt endpoints of the anisotropic NPT examples), the plane span is not the axis length under shear, the resident box producer and the data-file box writer are the same arithmetic, and the re-triangularized representative is the one that travels |
| `test_box_relax.py` | 11 | the `fix box/relax` line the deck carries is the grammar this build accepts and names `vmax` out loud, the style/profile pairing is refused by name in both directions (`box_relax` on `ff_optimize`, `ff_optimize_cell` with the style off), and the box the engine stopped on comes back rather than the one that was asked for |

Result reading (what comes back):

| File | Tests | Pins |
|---|---|---|
| `test_output_parser.py` | 24 | table/multi log formats, per-atom norm restore, `fin_coord` from dumps, `fin_velocity` unit conversion, force dumps with and without the id column |
| `test_fin_cell.py` | 24 | the plane+tilt to lattice inverse (round-trip on five lattices and in the test environment box at full precision), header-driven tilt names, last frame wins, no box block means no cell reported, MD reports nine numbers while `ff_optimize` reports none |
| `test_pbc_declaration.py` | 34 | every shipped case declares `pbc`/`cell`, the declared cell equals the fixture box, a declared cell is what reaches the generated data file, an undeclared one provokes no box invention |

Execution placement (where and how a step runs):

| File | Tests | Pins |
|---|---|---|
| `test_calcdir_placement.py` | 5 | directory name follows the caller's namespace, the five resident artifacts live in the calcdir, result file names come from the same stem function the writer uses |
| `test_query.py` | 7 | the `resolve_profile` hook being exposed, the jobtype to profile mapping, an unknown method raising |

Resident service, one file per module:

| File | Tests | Pins |
|---|---|---|
| `test_persistent_protocol.py` | 43 | framing (big-endian length prefix, oversize refused on both ends before allocating, truncated header or body meaning a dead service, a split frame reassembled, deadline as timeout, a real socket pair round trip), the field contract per operation (unknown operation, unknown field, missing field named, wrong version, the closed error vocabulary), the image field being optional with its rows validated (`evaluate` without image counts sends none, a malformed image row is refused), and the session fingerprint being stable, hex, insensitive to key order and sensitive to every component including the protocol version |
| `test_persistent_client.py` | 30 | handshake and build check before anything loads, the INIT ack, native numbers back from `evaluate`, coordinates and image flags going out together as the arrays the framework holds, reply identity (wrong request id, generation gap, missing required field), short / missing / unrequested force arrays refused rather than zero-filled, non-finite values refused, recoverable errors keeping the session while dead-service and timeout errors poison it, peer shutdown reported dead, `close()` idempotent, evaluate after close raising, parallel callers serialized |
| `test_persistent_launcher.py` | 26 | argv and script assembly (no `lmp` binary, `exec` so the group stays killable, the shell prologue first), endpoint naming and the AF_UNIX relocation that moves only the socket, the private runtime directory, socket claim / release / collision and one namespace holding two sockets, a socket file alone not counting as readiness, a dead service quoted from its own log, `terminate` unlinking socket and owner record |
| `test_persistent_server.py` | 45 | the INIT execution order, world-size refusal, load not converting the domain, tilt converting it before the tilt line, `TILT_EPSILON` parity with the writer, box step sequence and read-back mismatch being fatal, **a cell that has not moved being applied once rather than re-boxed every step**, image flags travelling with the coordinates, the engine-side phase table staying with the run, gather/scatter interface shapes, non-finite numbers ending the session, owner death and close without `finalize()` |
| `test_persistent_backend.py` | 27 | the routing decisions (default cold, slurm cold with the reason said once, internal-iteration profiles never resident, an uncompressed template refusing residency), no per-step file on the hot path, the cell snapshot becoming the step box, an atom-count change refused before it is sent, `close()` / `with` / an exception inside the block all releasing, one socket per live backend, two directories of one namespace both live and each releasing only its own, a rebuilt backend reclaiming its socket, `keep_tmp` both ways, a start failure not being allowed to look like a signature problem, and the import-boundary checks that only the server module may touch the native stack |
| `test_persistent_test_plan.py` | 19 | the keyword survives every hand the planner makes (flat `.json`, classic `.input`, agent `.config`), the read-back of the string `'true'`, and that all twelve official MD examples route resident in both example modes |
| `test_resources.py` | 6 (#306, def-level) | the manifest `resources` block: the thread axis is what a run sets, and no LAMMPS route claims an MPI rank it does not start |

The counts are also the history of this line, taken from the rounds' own
entries (lammps-only figures; where a round recorded a combined
`lammps + utilities` count it is not spliced into this chain): 118 before the
resident work, 323 after #233 (the service itself), 330 after #234 (result
files and placement), 366 after #235 (`pbc` declarations and the tilt
threshold), 390 after #236 (`fin_cell`), 393 after #237 (the examples, and
unchanged through the #238 documentation round), 406 after #239 (the cold and
resident sides closed on identical numbers, the equilibrated water box
shipped), 407 after #240 (the `set atom` grammar), 411 after #242 (image flags
seeded from the geometry), 422 after #243 (a cell that has not moved stops
being re-boxed every step, and the phase table lands in the products), and 523
measured now (`pytest -q` over the whole directory, 2026-10-31). What grew
since #243 is #244's phase-table judgments, the constraint invariants #254 put
on the builder, #256/#260's box round-trip parity file, the QM/MM file (26
tests now, 14 when this chain was last counted) and #443/#444's reduced-surface,
periodicity and socket-claim pins. The intermediate rounds were not counted
here, so the chain is a floor and the measured total is the figure to quote.
Framework-side consumption of the three channels is judged where the consumer
lives, in `qualification/verification/test_calculator.py`:
`TestFinCoordConsumption` 7, `TestFinVelocityConsumption` 9,
`TestSkillFinCellConsumption` 13, plus `test_optimizer.py`/`test_workflow.py`
for the `opt_status.txt` ruling.

Run them with `python3 -m pytest pyrai3md/skills/lammps_skill/tests -q
-p no:cacheprovider`; `unittest discover` collects 0 from this directory,
because the tests are pytest-style classes.

---

## 10. Key Differences from Other Skills

- **Ground state only**: no `nstate`/`target_states` flexibility; the
  normalize step pins `nstate = 1` and clears NAC/SOC.
- **Only complete `fin_velocity` implementation**: the `.cond` dump +
  `VELOCITY_TO_LAMMPS` / `LAMMPS_VELOCITY_TO_AU` pair provide an exact
  Bohr/au round trip; other skills leave `fin_velocity` reserved.
- **Optional main template**: most skills generate the entire input; here the
  user template (when present) contributes force field definitions and is only
  stripped of execution commands for internal-iteration profiles.


- **Per-atom thermo with parser-side restore**: `thermo_modify norm yes` is a
  LAMMPS-specific contract matched by the `* natom` restore in the parser.
- **Element-free dumps**: no `modify element`; atom identity comes from the
  system atom list, not from dump content.

---

## 11. Resident Service (`python_enabled`)

### 11.1 Why a second process

The subprocess path pays, per framework step, for writing three input files,
starting `lmp`, re-parsing the topology and force field, and parsing a log. The
force-field compressor (#230) made that content available as one model built
once, but the model still only fed a per-step file writer. `python_enabled`
keeps the loaded LAMMPS instance alive instead, in a process of its own.

The process boundary is not an optimization, it is the isolation contract: the
service interpreter is the one that can `import lammps`, so the framework
interpreter never links liblammps, never imports `mpi4py`, never holds a
communicator and never asks what rank it is. Even a single-rank world runs as a
service for that reason.

### 11.2 Modules

| file | side | responsibility |
|------|------|----------------|
| `service/persistent/protocol.py` | shared | framing, message field lists, session fingerprint, error classes; pure data |
| `service/persistent/client.py` | framework | `PersistentSession`: one request in flight, four reply gates, poison semantics |
| `service/persistent/launcher.py` | framework | argv/script building, `Popen` as a new session group, readiness handshake, owner record, socket claim registry |
| `service/persistent/server.py` | service | the only module importing LAMMPS (and `mpi4py` when parallel): rank 0 IPC, `comm.bcast`, box application, scatter/gather, extraction, close |
| `service/persistent/bringup.py` | service | operator-run acceptance report (interface surface, numbers against the subprocess path, variable cell, ranks, watchdog, where every file landed) |
| `service/persistent/profile.py` | framework | the step profiler that writes the middle table of the three perf files (`PROFILE_FILE = 'lammps_backend.perf'`, `:39`); kept skill-side because a skill must not import framework internals |

`service/persistent/__init__.py` exports only the framework-side modules, and an
AST check in `test_persistent_backend.py` fails if any file outside `server.py`
and `bringup.py` imports `lammps` or `mpi4py`.

The service is launched as a **file** (`python .../persistent/server.py`), not
through `python -m`, and `server.py` falls back to importing `protocol.py` by
path. Consequence: the service interpreter needs no PyRAI3MD install, no
pydantic, and nothing but the standard library and LAMMPS.

### 11.3 Shared producers

Two extractions keep the paths comparable, and both are load-bearing:

- `lattice_to_true_box()` / `box_to_bound_numbers()` / `box_to_bound_lines()`: the
  data-file writer and the resident box payload come from one function, so a
  variable-cell step cannot evaluate in a box that differs from the one the cold
  path would have written. What crosses the socket is the **bounding planes plus
  the tilt factors** — the numbers a data file states — because that is the frame
  both `read_data` and `change_box all <axis> final` take, and `extract_box()`
  echoes them back unchanged. The true bounds would have put a tilted cell one
  shear off the subprocess path's domain.
- `build_script_head()` / `build_step_tail(files=)`: INIT loads the same script
  text the subprocess path writes; `files=False` is the same sequence minus the
  lines naming an output file. `tests/test_input_builder_golden.py` freezes the
  whole generated text of every profile so a refactor cannot drift the cold path
  one space at a time.

### 11.4 Ordering and error classes inside the service

Rank 0 reads a request, validates it, and only then broadcasts; every rank
executes the same collective sequence, so rank 0 broadcasts exactly once per
served operation and a pre-collective rejection is answered without one. A
post-collective failure is not repaired: LAMMPS reports input errors through
`MPI_Abort`, which ends the process group, so the service reports what it can
and dies, and the `--owner-pid` watchdog plus the batch walltime are what keep
nothing behind. `lmp.close()` releases the instance; `lmp.finalize()` would shut
MPI down under any sibling instance and is never called.

Per step the order is box, positions, `change_box all remap` (only where the
domain carries tilt factors), `run 0`, extract. A domain becomes tilt-capable
when a cell that carries tilt arrives, never at load: conversion re-frames the
reciprocal sum, and the deployment measures -19.811016 kcal/mol for an untilted
cell against -19.760369 for the same cell on a domain converted for nothing. The
conversion is one-way — LAMMPS has no path back to an orthogonal domain — so a
cell that loses its tilt mid-session is computed triclinic-with-zero-tilts while
the subprocess path would compute it orthogonal; the deployment puts that at
8.1e-5 in the framework's energy units, and `bringup` reports it without
claiming it (11.9).

"carries tilt" is decided against `protocol.TILT_EPSILON`, not against zero, and
it has to be the same number on both paths. A cell declared with three 90-degree
angles reaches this code as `[[10, 0, 0], [6.1e-16, 10, 0], ...]` because
`cos(radians(90))` is not zero, so an exact comparison calls such a box sheared
and converts the domain — and the test environment measured the consequence on the
qualification water box: PPPM went from a 6x6x6 grid to 8x8x8 and the resident
energy read -20.5581 where the subprocess read -20.6642, a 0.106 kcal/mol split
no tolerance in this skill covers. The data-file writer had always used 1e-10;
the service compared to 0.0, and `bringup` never caught it because its cells are
built with literal zeros rather than from angles. The threshold now lives in the
one module both sides import.
Forces come from `gather_atoms_subset` (which resolves through `atom->map()` and
cannot match a ghost, because LAMMPS stores ghost tags negated);
`gather_atoms` and `extract_atom('f')` are banned because they expose
rank-local or ghost-polluted order.

### 11.5 Lifecycle

`close()` is idempotent and is what `with` blocks and the `weakref.finalize`
hook call. It covers ordinary exit and garbage collection; it cannot cover
`SIGTERM`/`SIGKILL`, which is what `multiprocessing.Pool.__exit__` sends to a
worker (scheduler.py `_run_pool`), so the service carries its own owner-pid
watchdog and an idle timeout. `skill_guide.md` states this as a user-visible
limit rather than implying it is solved.

Who calls it is the framework's decision, and since #444 the rule is that the
unit which built a Calculator releases it: `workflow/__init__.py::execute()`
closes in a `finally`, so a batch whose pool has one worker (units run one
after another in the dispatching process) does not accumulate the engines of
every unit before it. The MD loop's own release at the end of a trajectory is
the same rule stated early; `close()` being idempotent makes the second call a
no-op.

The live-service claim is keyed on the socket path, not on the execution
namespace. The namespace is a directory convention and is not unique per
service: an isolated-workdir batch (`Hessian`, `AIMD`, `Search`, adaptive)
drops the `_runtime_{idx}` suffix on purpose because the directory is already
unique, so an optimization loop and every displacement of its numerical
Hessian share one namespace while owning one directory and one socket each.
Keying the claim on the namespace refused the second displacement and with it
the whole batch; keying it on the socket refuses only what would actually
collide, a second service on a socket a live one already answers, and lets
`close()` release exactly its own.

### 11.6 How a run decides, and how the plan is tested

Routing has one predicate, `is_persistent_enabled()`: `python_enabled` on and
`execution_mode` other than `slurm`. `execution_mode` is already the field the
skill uses to pick its runner, so residency needed no new channel from the
framework. A slurm step submits and waits once per call, which leaves nothing to
keep resident, so the switch is inert there by design rather than an error.

Qualification covers the switch with `ff_gradient_persistent.yaml`, and
`test_persistent_test_plan.py` checks the planning chain end to end:
`generate_test_configs` discovers the case and stages its topology under the
project name, `parse_test_yaml` carries `python_enabled` as a native bool into
the flat `.json`, `convert()` emits `python_enabled true` inside `&LAMMPS`, and
`convert_to_pyrai_config()` keeps the bool through the agent `.config`. The
read-back half matters just as much: a classic input delivers the value as a
string, so the test asserts `'true'` reaches `environment.python_enabled is
True` and routes resident, and that the cold twin (no keyword) routes to the
subprocess path. `python_path` is deliberately absent from the shipped case:
which interpreter holds `lammps` and `mpi4py` belongs to the machine running the
test, `validate_persistent()` fills it from `sys.executable`, and users who need
a different one set it in the `skills.lammps_skill` section of their own
`pmd_env.yaml`.

The switch is not confined to the skill's own cases. All twelve official
framework-driven MD examples (`pyrai3md/examples/md_lammps_*.yaml`) declare
`python_enabled: true` in their `lammps` section, because a framework MD step is
one `ff_gradient` call per step and the subprocess path was paying a LAMMPS
launch 2000 times per 2 ps run. They declare it rather than inherit it from
`skills.lammps_skill` so a direct `pyrai3md md_lammps_nvt_nh.yaml` from the
examples directory behaves like the qualification task. One asymmetry follows
from the two paths, not from the switch: a cold step reads forces out of a text
dump at six significant digits, while a resident step takes them from the engine
interface at full precision, so the resident number is the more accurate of the
two and the two are not interchangeable when a trajectory is being judged on
energy drift. `TestOfficialExamplesRouteResident` pins that the keyword survives
both example routes, the YAML and the classic input the planner generates from
it, through the merge and the skill's own `build_config()`.

### 11.7 What a resident run writes, and in what names

Both paths get their input package from one function,
`input_builder.write_input_package()`, which calls the three writers with the
same model argument. That is not a convenience: the resident path used to list
the writers itself and passed the model to two of them, so it loaded a verbatim
`in.init` (still declaring `buck/coul/long`) together with a *pruned*
`in.settings` (whose rows for that sub-style had been trimmed away), and LAMMPS
aborted at the first run with "Pair hybrid sub-style is not used". A missing
keyword argument had silently changed the force field on one path only.
`tests/test_input_builder_package.py` now hashes both packages and requires the
three files to be identical, and `ff_model._validate()` refuses any model whose
hybrid declaration lists a sub-style no surviving `pair_coeff` row uses, so the
inconsistency cannot reach a writer at all.

Naming follows the subprocess path, because the naming is how a failure is
found. `launcher.service_paths()` puts every artifact in the calculation
directory: the package, the compressed model `<project>.ff.json`, the script
`<basename>.sh` (starting the service instead of `lmp`), the log
`<basename>.log` (the service's output plus LAMMPS' screen text), the socket
`svc.sock` and the owner record `owner.json`. Nothing is written outside the
directory a person already opened, and the two process files carry fixed short
names on purpose: the directory is the namespace, so repeating the stem in the
name only spends the AF_UNIX path budget, and a relocated socket still has to
be findable by someone who knows what they are looking for. `owner.json` and
`svc.sock` are unlinked together when the service is reaped. The single
exception is a calculation path so long that even `svc.sock` would exceed the
108-byte AF_UNIX limit: the socket then binds under `_runtime_dir()` (a 0700
directory named after the execution namespace plus a digest of the calculation
directory) and `start_service` says so in
the first line of the log rather than leaving the relocation to be guessed.
The digest is load-bearing, not decorative: an isolated-workdir batch shares
one namespace across all of its units, so without the digest every unit of the
batch would relocate onto one socket and the claim of 11.5 would refuse all of
them but the first.
The service runs with the calculation directory as its working directory, so a
step that writes files writes them beside its input.
The launcher's readiness error quotes the end of `<basename>.log` through the
same `utils.read_log_tail()` the cold parser uses, so "LAMMPS log tail:" reads
the same whichever path failed. There is no pre-flight interpreter test: the
service imports `lammps`, builds its instance and proves its world before it
binds, so a service that cannot work has already exited with its reason in that
log.

`debug` decides how much of the cold path's output a resident step reproduces.
Off (default), the loaded script carries the thermo setup only and each step
sends back numbers over the socket. On, each step also issues the cold path's
write commands, so the four result files below appear as they do for a
subprocess run and the two paths can be compared file for file — which is what
bringup's C10 does, byte for byte.
The per-step command sequence is `build_run_section()`, delivered once with
INIT and repeated verbatim for the smoke evaluation and every step after it; it
is part of the session fingerprint, because a session that started without
dumps must not grow them half way through. Splitting setup from execution also
fixed the load order: the script no longer contains a `run`, so nothing is
evaluated before the box conversion and the kspace re-issue that must precede
the first number.

### 11.7a The four result files, and the digits they carry

A single point states its own result as text, one file per quantity, all four
under the run stem that `output_basename()` decides and
`step_data_files()` renders into names:

| file | written by | content |
|---|---|---|
| `<title>.energy` | `variable pyrai_pe equal pe` + `print "${pyrai_pe}"` | one number, fifteen significant digits |
| `<title>.coords` | `write_dump all custom … x y z modify sort id` | LAMMPS text dump, six significant digits |
| `<title>.force` | `write_dump all custom … fx fy fz modify sort id` | same, gradient profiles only |
| `<title>.stress` | `print "STRESS ${pyrai_sxx} …"` | six pressure components, fifteen digits, `stress` switch only |

The parser reads these and nothing else for the single-point profiles: it no
longer lifts the energy out of the log's thermo table, because that table
quantizes to eight digits and would leave the resident path (which returns
doubles over the socket) disagreeing with the subprocess path in the seventh
decimal. `parse_log()` stays the energy source for `ff_optimize` and the MD
profiles, where LAMMPS runs a trajectory of its own and the log is the only
channel for it; a missing energy or stress file is an error, quoted with the
end of the log, not a zero to be quietly filled in.

Six digits on the force file is the engine's ceiling, not a choice this skill
made. Measured on 23Jun2022u1, `dump_modify … format` and
`write_dump … modify format` are `Illegal dump_modify command
(src/dump.cpp:1175)`, a per-column format list reaches a different rejection
(`Expected integer parameter (src/dump_custom.cpp:1711)`), and the `cdm` and
`json` dump styles do not exist in that build (`Unrecognized dump style
(src/output.cpp:771)`). So a text force table is written with `%g`: the
deployment writes `19.748 -14.436 4.69371` for one force row while the energy
file holds `-20.6642423474568`. Cold-versus-resident force agreement sits at
`max|dG|=4.1e-8` in framework units; energy agreement, which had been limited
by the same kind of truncation, is now `dE=3.5e-17` (it was `5.5e-10` while
the energy came out of the thermo table). Widening that would mean a binary dump, which
contradicts the point of these files being readable, or a split high/low column
pair, which makes the file unreadable to a person.

### 11.8 The interface adapter, because two Python interfaces exist

The service talks to LAMMPS through `_ApiAdapter`, not through the raw
module, because the two published Python interfaces disagree in ways that
are invisible until a real build answers back:

| question | SWIG wrapper | pybind11 wrapper (deployed) |
|---|---|---|
| `gather_atoms_subset` | `(name, dtype, count, ids)`, ids a list | `(name, dtype, count, ndata, ids)`, ids a ctypes int array |
| `scatter_atoms` data | list accepted | ctypes array required |
| `extract_compute` style/type | names | C integers; None for an unknown pair |
| a global vector | sequence | pointer, no length attached |
| `extract_box()` | flat bounds, origin, tilts | `boxlo(3), boxhi(3), xy, yz, xz, periodicity, box_change` |
| box change | (whichever build spells it) | `x final a b y final c d z final e f`, then `xy final g xz final h yz final i`; `change_box all final` with nine numbers and `remap matrix` are both illegal |
| `gather_state` | present | absent |
| `has_mpi_support` | module | class |
| version | `LAMMPS_VERSION` | `__version__` |

Each disagreement is resolved by trying the known shapes in order and
recording the one that worked, never by detecting the package: the
candidate call is a pure Python-level construct, so a wrong shape raises
`TypeError`/`ArgumentError` before LAMMPS is touched, which makes the trial
free and repeatable. Two consequences worth remembering. `extract_box()`
must be read by shape: flattening the pybind11 answer puts
`periodicity` where the tilt factors are, and a box read-back is the gate
that decides whether a variable-cell step is trusted. And a vector compute
has to be asked for with its length stated by the caller, since a pointer
carries none; `pressure6()` says six because that is the contract of the
method, not because the object could tell it.

The test stub answers the way the deployed interface does (ctypes arrays,
integer constants, the seven-part box), so a shape mistake fails in the unit
tests. `TestInterfaceShapes` covers the SWIG-shaped answers against the same
adapter, which is what keeps the fallbacks real rather than aspirational.

The box-change row is a grammar, not a call shape, so no Python-level trial
can discover it: an illegal `change_box` is answered by `MPI_Abort`, which
takes the resident group down at the first variable-cell step. It was settled
by running each candidate spelling in the test environment, one per process, and
`TestBoxSpelling` now pins the two surviving strings verbatim so a future
"cleanup" cannot quietly reunite them into the form the documentation of the
data file suggests.

### 11.9 Acceptance in the test environment

`service/persistent/bringup.py` was run on the machine this path targets
(testcluster, LAMMPS 23Jun2022u1 through the pybind11 `lammps` package, OpenMPI
4.1.6, `--nproc 1 2 --stress --periodic`) and reports **14 PASS, 0 FAIL,
1 SKIP** in 22.7 s: the interface surface, the subprocess reference energy
(`-0.03293056631265918` Hartree), resident agreement with it
(`dE=3.469e-17`, `max|dG|=4.066e-08`), repeatability and sensitivity
(`0.000e+00` against `1.898e-06` for a displaced step), three variable-cell
sequences (untilted resize `4.642e-15`, triclinic throughout `7.841e-16`,
untilted then tilted `7.841e-16`), the owner watchdog (0.2 s after `kill -9`),
the stress tensor (`4.108e-16` relative), rank independence between 1 and 2
ranks (`8.334e-15`), and where the run put its files.

The file checks are the ones #234 added. A plain resident session leaves
exactly seven names in its calculation directory (the input package, the deck,
the launch pair, the log) and the per-user runtime directory is not created at
all, because `svc.sock` fits: the qualification case's directory is 74
characters and the endpoint 82, against an AF_UNIX limit of 108, so no path was
relocated (`grep -c relocated` on the log reads 0). Under `debug: true` the
same directory holds eleven names after the service retired, which is the
seven plus the four result files and is also the proof that `_retire()`
unlinked socket and owner record; auditing it while the session was still live
showed thirteen, and that interim failure is why the audit sits in the
`finally`. `C10` then compares the four result files byte for byte between the
subprocess run and the resident one: identical. The names it deliberately
excludes differ by design, not by defect — the resident deck stops before the
`run 0` tail that the subprocess deck ends with, because that tail travels over
the socket, and the launch script and log describe two different ways of
running.

The one SKIP is a measurement that is deliberately not claimed: a cell that
loses its tilt after a tilted step. LAMMPS converts a domain to tilt-capable
and never back, so the resident instance computes that cell as
triclinic-with-zero-tilts while a subprocess start computes it orthogonal, and
the reciprocal sum differs by `8.071e-05` in the framework's energy units. The
check prints the number and the reason rather than passing or failing on it,
so the effect stays visible in the report instead of being hidden by a
tolerance.

`keep_tmp: 0` was verified separately on the same machine with a resident
one-step run: the session directory was gone when `close()` returned, the
scratch root held nothing, and no per-user runtime directory existed to hold
anything either. While a session is live its endpoint answers
`stat` as mode `0600` — `_bind()` narrows it before `listen()`, because a
calculation directory under a shared submission path can well be
group-writable, and on AF_UNIX that mode is what decides who may connect. The
owner record sits beside it, and both are gone once `close()` returns. That is
the whole of what a resident step writes, and it all sits in one directory the
caller owns.

**Re-run on 2026-09-29 (`tmp_20260929_185407`, same flags) says 11 PASS, 3 FAIL,
1 SKIP, and the three failures are the story.** The passing set is the same list
with fresh numbers: C1 interface surface (20220623, `has_mpi_support` on the
class, missing none), C2 subprocess reference `-0.3180548544071746` Ha, C3
resident agreement `dE = 2.776e-16` with `max|dG| = 4.216e-08`, C4 repeatability
`0.000e+00` against a displaced `1.713e-06`, C6 owner watchdog 0.2 s, C7 stress
`1.826e-15`, C8 rank independence `4.496e-15`, C9 seven names in a plain session
and nothing left in the runtime directory, C10 four result files byte-identical
and eleven names after the debug session retired. Two operator facts came out of
the first invocation, which was run without `--lammps-path`: C2, C3, C5, C7 and
C10 all came back `lmp executable not found in PATH` while C1, C6, C8 and C9
passed. The resident legs need no binary because they never start one; every
check that compares against the subprocess path does. `bringup` reports that as a
failure instead of skipping it quietly, which is the right behavior but surprises
whoever reads `PASS 11` as twelve checks having been possible.

What failed is the three C5 variable-cell sequences, and each failed on the
gradient leg only: `dE` was `7.816e-14`, `8.171e-14` and `8.171e-14` (bit-clean)
against `max|dG|` of `4.139e-06`, `4.175e-06` and `4.175e-06` with a
`FORCE_TOLERANCE` of `1e-6`. The products price what those sequences actually
are: the cold legs' `h2o_force.energy` reads `8636.197`, `16910.530` and
`18215.052` kcal/mol and the force dump peaks at `|F| ~ 46` kcal/mol/Å, against
C3's total of `-199.6` kcal/mol. C5's cells (11x10x10, 12x10x10 and the sheared
11/10/10) were written for the 10 Å cube the fixture used to be; the shipped
fixture has been the equilibrated 8.399 Å box since #239/#258, so those cells now
drive a badly off-equilibrium configuration. An energy that agrees to 1e-13 while
forces differ by ~1e-4 relative is the signature of a second-order (curvature)
sensitivity, not of two geometries: the cold path re-reads positions from
six-digit text while the resident path scatters the framework's doubles, and in a
strained configuration a 1e-6 Å position difference moves a force by ~5e-3
kcal/mol/Å while moving the energy by ~1e-13. **That is a hypothesis with the
right shape and unproven provenance.** The measured half of the story is that C5
passed with `max|dG| = 3.952e-08` back in #233, on the pre-#239 fixture. Until
C5's cells are re-priced against the current fixture, this section's headline is
`11 PASS / 3 FAIL`, not the #235-era `14 PASS / 0 FAIL`, and nothing should quote
the old line as today's acceptance.

The qualification suite says the same thing from the other side. Its eight
LAMMPS cases (section 9.1) share one calculation directory, and it now holds,
per case, the input package (`<project>.data`, `.in.init`, `.in.settings`,
`.ff.json`) plus
`<title>.in/.sh/.log` and the result files the profile produces —
`ff_gradient_test_force.{energy,coords,force}` for the subprocess case and
`ff_gradient_persistent_test_force.{energy,coords,force}` for the resident one
that ran with `debug: true`. Those three files are byte-identical between the
two cases, the energy reads `-20.6642423474516`, and a `find` over the whole
run directory turns up no `*_force.xyz`, no `*_stress.dat` and no runtime
directory: the retired names are gone from a real run, not just from the
builder. The directory is not free of other names, and is not meant to be: the
internal-iteration profiles still leave `{title}.xyz`, `{title}.cond` and
`restart.data` there, because for them the dump is a trajectory the framework
consumes rather than one step's result, and #234's renaming stopped at that
boundary by the user's ruling.

### 11.10 The resident line, round by round (#233 to #261)

Five rounds built and then corrected this path, and each one's lesson is a
test rather than a paragraph, so the table below is the index to them.

| Round | What the machine said | What changed | What now proves it |
|---|---|---|---|
| #233 | `ERROR: Pair hybrid sub-style is not used`, `Last command: run 0` on the resident path's first real run | One missing `model` argument made the hot path load a verbatim `.in.init` while `.in.settings` was the trimmed render. `write_input_package()` became the one writer both paths call, the model gained a self-check that every declared hybrid sub-style has coefficients, artifact names stopped being private to the session, `debug` appeared, and every runtime probe was deleted | `test_input_builder_package.py` (3): cold and hot packages equal file for file; `test_ff_model.py` (46) including the render-vs-coefficient invariant; `ff_gradient_persistent.yaml` as the resident twin; 323 unit tests, deploy `79 pass / 0 fail`, `bringup` 11 PASS |
| #234 | Cold and resident energies differed in the sixth decimal | The subprocess path started writing four result files per step (`<title>.energy/.coords/.force/.stress`, one stem function for writer and parser) and reads its results from them instead of the log: energy 8 digits to 15, `dE` `5.5e-10` to `3.469e-17`. Forces stayed at six because a text dump has no legal way to widen them, so the largest gradient disagreement is `4.1e-8` Hartree/Bohr and that is the engine's ceiling, not a bug to chase. Socket, owner record and `.ff.json` moved into the calculation directory under fixed short names | `test_calcdir_placement.py` (5), golden digests re-recorded with the three drift classes accounted for, `bringup` C9/C10 (11 names after retire, four result files byte-identical between paths), 330 unit tests |
| #235 | A `ff_nve` log said `Temp=86.51` while the framework logged zero velocities and zero kinetic energy; the resident energy read `-20.5581` against the cold `-20.6642` | `fin_velocity` got its missing consumer in the framework (`_apply_skill_velocities`, kinetic energy recomputed: `Ekin = 0.0242458786045029` Ha, 86.51 K, matching the engine). All eight cases had to declare `pbc`/`cell`, because with an undeclared `pbc` the builder leaves the template's box alone and the case silently computes periodic Ewald on a config claiming no cell. The float noise of `cos(radians(90))` = 6.1e-16 was converting the domain, so the tilt threshold became one shared constant, `TILT_EPSILON = 1e-10`. `opt_status.txt` stopped being fabricated on the engine path | `test_pbc_declaration.py` (34), `test_persistent_server.py`'s two tilt-threshold judgments plus the writer-parity one, `TestFinVelocityConsumption` (9) framework-side, deploy `81 pass / 0 fail`, `bringup` 14 PASS / 0 FAIL / 1 SKIP |
| #236 | The `.cond` recorded a box the state never learned about | The third transport channel: `fin_cell`, six parameters or nine flattened components, framework-derived into `cell` + `lattice_vector` + `volume` together, refused outright for a system that declared no periodicity. `parse_cond_cell` reads the box LAMMPS already writes, and `bound_and_tilt_to_lattice` is the exact inverse of the writer's transform. CP2K got the reader ahead of any cell-optimizing profile it has. The resident `RESULT` deliberately does not carry a box back | `test_fin_cell.py` (24) including the deployment box at full precision, `TestSkillFinCellConsumption` (13), golden unchanged as the over-reach detector, deploy `82 pass` with the `ff_npt` cell recovered as `10.000108541 / 10.001080950 / 9.999448296` and the `.cond` byte-identical (`md5 7c90562…`) to the #235 run that had no channel |
| #237 | Framework MD paid a LAMMPS launch per step | All twelve official MD examples declare `python_enabled: true` in their own `lammps` section, so a hand-run example takes the same path the qualification task takes | `TestOfficialExamplesRouteResident` (3 of the 19 in `test_persistent_test_plan.py`): the count, the declarations, and both example routes reaching a validated resident config; 393 unit tests |
| #238 | Nothing new from the machine: this was the round that wrote the accounting down | Documentation only. The test inventory, this table's existence, the knowledge base's ownership chapter and the architecture doc's channels entry | the golden digests, untouched, which is what made a doc-only round checkable |
| #239 | Two remote runs of the same example, `python_enabled` false against true, frame by frame | Numeric parity closed (identical initial conditions, `Epot` differing by at most `1.37e-05` Ha over 11 frames with no growth by frame number -- the cold path's six-digit ceiling, now measured instead of argued), the first efficiency number recorded (cold `95 s` per 100 steps against resident `5 s`; the `~19x` extrapolation from that short run was later corrected to `~113x` at 10000 steps), the equilibrated 8.399 Å water box shipped through `tools/refresh_water_fixture.py`, and image counts delivered in the same `update_system` as the wrapped coordinates | per-frame comparison in the round's own products; `System.image` persistence outside `SYNC_SYSTEM_FIELDS`; the `[PBC]` minimum-image note the runner records without judging |
| #240 / #242 | `ERROR: Illegal set atom command` on the first remote image write, then an alarm that no grammar explained | `set atom` accepts exactly one selector token (intervals in the `utils::bounds` sense, so a run ships contiguous crossing segments rather than one command per atom), and the initial crossing counts are seeded from the geometry instead of assumed zero: the reported mismatch was an input frame that never declared it had crossed a box face | the rewritten stub grammar, `test_input_builder`'s image rows, and #242's seed-from-geometry judgments |
| #243 / #244 | A step cost 8.39 ms and the ranking table's top row was not where I had guessed | A cell that has not moved is no longer re-boxed every step, the framework's own phase table ships in the products (`pyrai3md_step.perf`, every 100 steps), and reading the manifest once per run instead of once per step took the step from **8.39 to 4.20 ms** (median of five alternating remote A/B pairs, 1.996x). Two of my three cost hypotheses were falsified by the table, including the one I ranked first | `TestMdPhaseProfile` holding the measurement to its own contract (rows do not nest, coverage is checked), the resident-side `lammps_resident.perf` / `lammps_backend.perf`, and the ranking table in `dev/methods/MD_DEVELOPMENT.md` section 27 |
| #245 | The examples' length had been described three different ways | The twelve examples were unified on one length and the per-example `_example_timeout` budgets were retired whole, with the test environment's own `PyRAI3MD End: … Total:` as the budget argument; the step count itself settled at 2000 (2 ps, 21 frames) by the user's ruling, and the consequence was written down rather than discovered: 21 frames is below `RATE_MIN_FRAMES`, so the NVE sentinel is judged on the band and says so | the verdict line `[ENERGY CONSERVATION] judged on 21 frames by band: … (rate needs 50 frames)`; the retired block no longer resolves at all: `manager._resolve_example_timeout` and the `_example_timeout` map were deleted on 2026-10-20, leaving `_timeout` as the only budget |
| #253 | Fourteen failures that turned out to be the framework naming its own stop, not a crash | Gating, monitoring and early interruption moved under `control.debug`, splitting "numerically impossible to continue" from "walked too far for a test", the layered diagnostics module landed, and the anisotropic NPT examples moved to `barostat_tau: 10000` on measured drift | `workflow/records/diagnostics.py`, the `[DIAG …]` lines the runner reads back into `results.json`, and `test_manager`'s left-column alignment |
| #256 / #259 / #260 | An illegal `change_box` answers with `MPI_Abort`, and a full-tensor `scr` destabilized on the real engine | The hysteresis probe became one self-contained file riding on the service's own commands and read-back channel; the tensor `scr` branch was deleted after measurement and `scr` + `anisotropic` is refused at validation; the probe then ran and the box round trip came back **bit-identical** (`dE` `0.000e+00` kcal/mol, `dstress` 0 atm, `dbox` 0 Å over six legs, two returns each) | `probe_box_hysteresis.sh` with `hysteresis.json` beside it, `test_box_roundtrip_parity.py` (19) for the framework half, and the validation rejection for the branch that was removed |
| #261 | A 109-item round that had never been recomputed from its own products | Nothing in the engine: the round was the audit. 70 pass / 0 fail / 0 error / 39 skip, the two input modes bit-identical at the last frame, `scr`'s off-diagonal `mu` exactly zero for all 21 frames (isotropic as intended), `scr_sigma` measured `2.422e-3`–`2.546e-3` against the analytic `2.51e-3` | recomputation from `md.energies` / `.ensemble` / `.md.xyz` rather than from status bits; the residual on the energy ledger is the `buffer: 1` per-step record, still open |
| #262 | `bringup` answered `11 PASS / 3 FAIL / 1 SKIP` where the doc said `14 PASS / 0 FAIL`, and the engine refused `run 1` six ways | No code. The hot-start path was measured on the example面 (`tmp_20260929_185407`: 2000 steps in 12 s wall, `4.696` / `4.879` ms per step for the yaml / input modes, `hello` and `init` once each against `evaluate` 2000 times, `box_skipped=1999 of 2000`), the three C5 failures were recorded with their products instead of smoothed over, and `probe_run1_gate.sh` asked whether `run 1` could carry the single-point evaluation cheaper - it cannot (section 11.12) | section 11.9's re-run paragraph, section 11.12's cost re-read and the six-spelling table, and the documentation alignment of section 9.3 (17 files / 449 tests), KNOWLEDGE_BASE Q43-Q46, `ARCHITECTURE.md`'s variable-cell section and `OPTIMIZE_DEVELOPMENT.md` section 15.1 |

Four invariants held across every one of these rounds and are the reason the
corrections were cheap: one function produces the deck for both paths, one
predicate decides residency, a resident step never falls back silently, and the
golden digests detect anyone reaching outside the stated scope.

Open, recorded rather than fixed: `restart.data` is still written and never read
back; CP2K's `Lattice=` spelling and units are unconfirmed against real CP2K
output (the reader is plumbing ahead of a profile that does not exist); the
`pbc`-off-plus-engine-cell rejection has unit coverage but no deployment-machine
counterexample; the framework's own `optimize_cell` cell degrees of freedom
remain future work; and the shipped force field asks for a real-space cutoff
larger than half the example box can give it, which is a force-field decision
the qualification suite records and does not judge (11.12). Added by #262, in
the same spirit: **`bringup` C5 currently fails on the test environment** -- all three
variable-cell sequences report bit-clean energies (`dE` ~8e-14) with gradients
`4.14e-06` to `4.17e-06` against a `1e-6` tolerance, because C5's synthetic cells
were written for the 10 Å cube the fixture used to be and the shipped fixture is
the equilibrated 8.399 Å box (#239/#258), which makes those sequences strained
rather than merely large; re-pricing C5's cells, or the tolerance, against the
current fixture is its own round and this round did not touch it. The other thing
#262 recorded rather than fixed: with every `skills.*.test` switched off in
pmd_env.yaml, a hot-start acceptance round gets **zero** machine coverage of the
skill面 (0 skill tasks planned this round), so a green example run is not
evidence that the 449 unit judgments ran anywhere but the development machine.



### 11.11 Where the design came from, and what it became

This section is the surviving record of two planning documents,
`lammps_update.md` (2026-09-26, the isolation-by-service-process design) and
`lammp_update_preview.md` (the `python_enabled` build plan with its 2026-09-27
as-built write-back). Both were deleted once their content was folded here, so
a CHANGELOG entry that cites them is pointing at decisions, not at files. What
follows is the part of them that is still a decision rather than a description:
the design ideas that were tried in writing and then rejected, because the next
person to touch this path will otherwise propose them again.

- **Config surface: two fields, not seven.** The design proposed
  `lammps_backend_mode`, `lammps_service_mode`, `lammps_service_python`,
  `lammps_service_endpoint`, `lammps_service_launcher` and three timeouts. What
  shipped is `python_enabled` and `python_path`; the timeouts are constants in
  `protocol.py`. Every extra field is a knob somebody has to understand before
  a run, and only one of them had a real decision behind it.
- **One service ownership, no `external` mode.** A backend starts and reclaims
  its own service. Sharing a service across backends solves none of this
  problem and imports leases, BUSY replies and cross-process session identity.
  LAMMPS here is a continuous numerical computation, not a concurrent service.
- **Two resident profiles, not six.** `ff_energy` and `ff_gradient` (with
  stress) are resident. `ff_optimize`, `ff_nve`, `ff_nvt` and `ff_npt` stay
  subprocess because they finish a whole propagation inside one `calculate()`
  call, where a resident instance buys nothing and would turn "run from the
  top" into "continue".
- **Six modules, no `engine.py`.** `protocol`/`client`/`launcher`/`server` plus
  `bringup` and `profile`; the command sequence comes from the same `input_builder` helpers
  the cold path uses, so equivalence is guaranteed by one implementation rather
  than by cross-checking two.
- **No probes in shipped code.** The interpreter probe, `validate_persistent_
  environment` and the `PROBE_TIMEOUT_S` budget were all deleted. The service
  imports `lammps`, builds its instance and proves its world before it binds a
  socket, so "cannot work" is already an exit code plus a reason in the log; a
  probe only asks the same question in a second process.
- **The service starts as a file, not `python -m`.** Its interpreter usually
  has no PyRAI3MD installed and should not need pydantic to answer a socket.
- **The deployment's interface, not SWIG's.** `gather_atoms_subset` takes
  `(name, dtype, count, ndata, ids)`, and `gather_state()` and
  `extract_modify()` do not exist in the deployed build; the box is read back
  through `extract_box()` alone. Anything written against the manual's older
  shape fails with a `TypeError` on the machine, which is why these are stated
  here rather than left in a design note.
- **Energy leaves as a printed variable, not a thermo column.** A thermo table
  quantizes to eight significant digits and `print` of an equal-style variable
  keeps fifteen; that difference was the whole of the cold/resident energy gap
  before it was fixed. The six-significant-digit text dumps for forces are the
  engine's limit (`dump_modify format` is refused, `cdm`/`json` styles do not
  exist) and are left alone, which is why a resident force is the more accurate
  of the two paths and the two are not interchangeable under an energy-drift
  gate.

### 11.12 Periodic images, the cell coupling, and what a step costs

The framework integrates and the skill answers one single point per step, so the
geometry that arrives at each step is the framework's own: wrapped into the main
image, with the cell that belongs to it. Three things follow from that, all
settled in this round.

**Image flags travel with the coordinates.** A live LAMMPS instance keys its
unwrapped positions on `x + image * box`, and it keeps the flags of the previous
step. Overwriting `x` with a new wrapped coordinate while those flags stay is
what LAMMPS reports as `WARNING: Inconsistent image flags`, and it evaluates a
bond across the whole box when it does. So:

- `EVALUATE` carries the `image` rows beside the coordinates (optional on the
  wire, so an older message still validates; absence reads as "no crossings"),
  and `server.evaluate` applies them after the scatter and before the step tail
  with `set atom <idspec> image nx ny nz`. The selection is one token, a single
  ID or a `lo*hi` range (`set.cpp` hands it to `utils::bounds`), so a target
  group is emitted as its maximal contiguous runs and the first evaluation of a
  648-atom box is one command, not 648. `ID` is not a style word in any LAMMPS
  this skill targets, and a space-separated list of IDs is parsed as a second
  keyword, so the first spelling of this line aborted the job with
  `ERROR: Illegal set command (src/set.cpp:74)` and took the MPI group with it.
  The first step restates every atom rather than trusting what `read_data`
  derived from box containment; later steps send only what changed, so a step
  where nothing crossed sends no command at all;
- no `change_box all remap` is sent any more, including on a tilted cell: remap
  would accrue crossings into the flags, and doing that as well as stating them
  counts every crossing twice;
- the subprocess path is handed the same numbers as three trailing columns of
  its Atoms section, which `read_data` reads automatically. That is not
  decoration: making one path image-aware alone would have broken the
  byte-for-byte cold/resident gate, and on an orthorhombic cell it would have
  been invisible anyway, because LAMMPS takes the minimum image for bonds and
  pairs there. Under tilt it evaluates them from the unwrapped positions, which
  is where a wrong count becomes a wrong number.

**Where the initial counts come from.** Counting cannot recover a crossing that
happened before the counting started, and the shipped water box is the final
frame of an NPT equilibration wrapped into the main image: seven of its forty
O-H bonds sit on opposite faces. Handed over with all-zero counts, those bonds
are continuous only in appearance, and the log says so on every run. So the
counts are now derived once at configuration time
(`schema.normalize.seed_initial_image` calling
`utilities/bonds.derive_initial_image`): bonds are perceived by
minimum-image distance against the repository's own `bond_lib` table, each
connected component is walked as a spanning tree whose children inherit the
image step of the bond that reached them, and the wrapped coordinates are never
touched - only the integers are read off. On the deployed box this yields five
non-zero rows, each in {-1, 0, +1}, reproduces the reference implementation of
the same algorithm row for row, and satisfies LAMMPS' own criterion (the longest
unwrapped bond becomes 1.028 A against a half-box of 4.200 A). A structure the
derivation cannot satisfy - a ring that winds around the cell, a bond too long
for the box, more atoms than `PERCEPTION_MAX_ATOMS` - is refused with a warning
rather than answered, because a wrong count is worse than the missing one it
would replace. A user who states `image:` in the input is never overwritten.

**The cell coupling is the framework's word.** `md.barostat_mode` already
existed and already drove the framework integrator, but the skill never saw it,
so `fix npt` was hard-coded to three same-pressure axis clauses, which is
LAMMPS' `aniso`. An example declaring `isotropic` still got an anisotropic box,
which is how a 20-water equilibration came out with a 1.4 axis ratio instead of
a cube. `barostat_mode` is now registered in the skill's own `reference.py`
(which is what stops the merge dropping it), validated against the framework's
two-word vocabulary, and rendered as `iso` for the isotropic case. The
anisotropic case keeps being written as three axis clauses rather than the
equivalent `aniso` keyword, so the input text of a run that asked for
anisotropic is reproduced byte for byte.

**The fixture is an equilibrated box, and it is too small for its own
cutoff.** The shipped water box is now the final frame of a LAMMPS `ff_npt`
equilibration (`tools/refresh_water_fixture.py` turns a real-units `.cond` into
the repository's representation: translated to a zero origin, wrapped with its
crossing counts recorded, momenta de-drifted and rescaled to 298.15 K). It
lands at 8.399 Angstrom cubic and 1.010 g/cc, which is liquid water. It is also
smaller than twice the 10.0 Angstrom real-space cutoff `h2o.in.init` declares,
so a pair can interact through more than one periodic image and the neighbour
list changes size as atoms cross the half box: energy that nothing the
integrator did is responsible for. `total potential energy` of the box being
about a quarter of what liquid TIP3P should measure is the same fact seen
through numbers. Re-equilibrating cannot fix that inequality, so the
qualification run records it in the example's details (`[PBC] pair cutoff ...
exceeds half of the ... cell`) and leaves the verdict to the force-field
decision that owns it; `runner._check_minimum_image`.

**Where the NPT energy gain stands, now that the records are readable.** The
09:19 round finally allowed the ledger to be added up, because #251's gutter
fix made `md.stress`, `md.ensemble` and `md.xyz` agree with each other
(`cell_work_ledger.py`, output `CELL_WORK_LEDGER.txt`). Per
recorded frame it compares the particle energy change with the mechanical work
the cell could have done, in both of its parts: the isotropic `-(P dV)` and the
deviatoric `-V sigma_dev : d epsilon`, using the engine's own returned stress.
Three findings, in the order that matters:

- the shape term is 3-10x the volume term everywhere, and dominant in the runs
  whose box collapses. Anyone reasoning about an NPT energy budget from
  `P dV` alone is missing the larger half -- which is exactly what an earlier
  entry of this changelog did when it called the gap "two orders of magnitude
  unexplained". That claim is withdrawn.
- the ledger does **not** close uniformly. `nh_pr` balances (0.115 Ha gained
  against 0.092 Ha of cell work), while the two berendsen examples that finish
  show only 0.017-0.020 Ha of cell work against 0.174-0.226 Ha gained. So the
  quasi-static work of the box explains the collapsing runs and not the
  well-behaved ones.
- the residual is not yet evidence of anything, and must not be written as if
  it were. Frames are 100 steps apart, so each term is a trapezoid over a
  signal that varies on a few steps; a correlated integrand undersamples that
  way, and in one direction only. Two measurements settle it: a per-step record
  (`buffer: 1`), which removes the truncation and is still outstanding, and the
  box hysteresis probe below, which has now been run in the test environment and
  closes the one channel where the engine itself could have been handing back a
  configuration it was not given.

**The remaining channel: is `change_box` a re-description or a deformation?**
The handshake's structural guarantees are already in place and were re-checked
rather than assumed: the reply carries the engine's own box
(`protocol.py` `OP_RESULT`), `server._verify_box` compares all nine fields with
what was sent to `BOX_TOLERANCE = 1e-8` and kills the session on a mismatch, and
the box is re-derived from the just-committed lattice every step. What none of
those can see is a box change that returns the same *numbers* while leaving the
*atoms* somewhere else -- 11.4's measured conversion gap of
-19.811016 against -19.760369 kcal/mol is the trace of that. `test_box_roundtrip_parity.py`
closes the framework half of the question with no engine in the loop: lattice,
true box, bounding planes, lattice again, on negative-tilt and sheared cells
including the one `nh_pr` reached, to 1e-10 A, plus the byte-parity between the
resident box producer and the data-file writer that 11.3 claims. The engine half
needs an engine, and was run on 2026-09-29 with
`probe_box_hysteresis.sh`, one self-contained file (the
driver rides inside it as a heredoc, and the water fixture is read out of the
installed package): it holds coordinates fixed in one live instance and walks
A (the fixture cube) to B (the sheared cell `nh_pr` last logged in the 09:19
round) back to A, twice over each, so a per-trip loss separates from a drift.
The cell numbers go out through `box_to_bound_numbers(lattice_to_true_box(...))`
— the arithmetic `resolve_live_box` is — and through the service's own
`_change_box_lines`, and they come back through `_ApiAdapter` and the same
compute call-shape aliases `_extract_compute` settled on, so the probe cannot be
clean on a cell a real step would have mishandled.

The answer is **clean, to the last bit**: over two returns each, both cells give
`dE` 0.000e+00 kcal/mol, `dstress` 0 atm and `dbox` 0 A against thresholds of
1e-7 Ha and ~0.3 atm (LAMMPS 23 Jun 2022 Update 1, `testcluster-next`, six legs,
`hysteresis.json` beside the driver). `change_box` plus a restated coordinate
array is a pure re-description on this engine, so no variable-cell step can be
an energy source through this channel, and the 0.045-0.226 Ha the NPT examples
gain does not come from here. Two readings of that, both wanted:
the ledger's residual is left to the two candidates it already named -- the
100-step frame spacing, and the engine's non-conservative virial -- and the
untilted return leg (three zero tilts on a triclinic domain) is accepted by this
build, so a cell that passes through an untilted shape has no wall in front of
it.

Getting the probe to run cost three facts about this build's Python interface,
none of which is a probe defect and all of which the service already knew:
the pybind11 constructor spells its arguments `cmdargs=`, and a positional
list lands in `name`, which it then formats into `liblammps_<name>.so`;
`-in` in `cmdargs` opens the log but does **not** process the script, so the
box is undefined until `file()` is called, the same order the service's `LOAD`
uses; and `scatter_atoms` takes a ctypes array, not a list, which is why the
probe asks `_ApiAdapter` rather than re-deriving the call shape.

**What a resident step costs, and where.** Three tables describe one step, each
written by its own owner on a hundred-step cadence, and read together they price
the whole of it: `pyrai3md_step.perf` in the run directory (the workflow loop),
`lammps_backend.perf` in the calculation directory (this skill's backend around
the session, with the client's socket buckets under their own header), and
`lammps_resident.perf` in the same directory (the engine). Measured on the
deployed cluster over a 10000-step NVE example: **8.4 ms per step of framework
wall time, of which the engine takes 1.47 ms** (the median of five alternating
pre-fix samples, which spread over 2.3%), the phase rows accounting for 99.5% of
the step so the ranking is a measurement rather than an inference. The table
below is the pre-fix baseline.

| ms per step | phase | where it lives |
|---|---|---|
| 2.42 | `convert` | framework: the skill's `manifest.yaml` was read and YAML-parsed **once per step** |
| 2.34 | `execute_terms` | framework, of which the session round trip 2.22 (client: send 0.15, wait 1.88, parse 0.09) |
| 1.63 | `verlet_i` + `verlet_ii` | framework: each began by deep-copying the previous history frame |
| 0.67 | `update_history` | framework: building the frame (its `deepcopy(model_dump())` is 0.48 of it) |
| 0.51 | `snapshot_lookup` | framework: the checkpoint deep-copied the frame a third time |
| 0.24 | `apply_pbc` | framework |
| 0.17 | `velocity_correction` | framework |
| 0.12 | `output_record` | framework |
| 1.27 | `step_tail` | engine, inside `evaluate`: the `run 0` tail |

Two of those rows were fixed in #244, and both fixes are byte-for-byte silent:
`resolve_stress_unit_factor` now resolves a skill's declared unit once per
process (an unresolvable unit still warns every step, because that is a defect
and not a computation), and the four read-only consumers of the previous history
frame - both Verlet half steps, the barostat's previous stress, the checkpoint
snapshot - take it by reference instead of copying it. The step went from
**8.39 to 4.20 ms** (the median of five alternating remote A/B pairs; the spread within a
variant was 2.3% for A and 0.9% for B), and `md.energies`, `md.pop`, `md.velo`, `md.xyz` came out
byte-identical to the unfixed run, with `etot_drift_rate 3.360e-08` unchanged.

**Re-read on 2026-09-29 over the shipped 2000-step NVE example**
(`tmp_20260929_185407`, both input modes, all three tables present): the step
costs **4.696 ms** in the yaml mode and **4.879 ms** in the classic-input mode,
the phase rows covering 99.4% of each, and the whole example finishes in 12 s of
wall time (`PyRAI3MD End: … Total: 00:00:00:12`). Inside it the backend answers
`2.806` / `2.965` ms per step, of which the socket wait is `2.398` / `2.553` and
the engine's own `evaluate` is `1.976` / `2.116` -- `step_tail` (the `run 0`
tail) being `1.675` / `1.773` of that. The residency claims are in the same
files rather than in prose: `op=hello calls=1`, `op=init calls=1`,
`op=evaluate calls=2000`, and `box_skipped=1999 of 2000 steps` -- one load, one
box, two thousand geometries. Against #244's `4.20 ms` the gap is ~12%, and it is
**not** recorded as a regression, because #244's number is the median of five
alternating A/B pairs and this is one sample per mode; the same round showed a
variant spreading 2.3% and a sampling-order effect of 0.96x against 1.20x, which
is the band this difference lives in. Calling it either "fine" or "a slowdown"
would need the same five pairs, and this round did not spend them.

Three guesses died on this table and should stay dead. The `50 ms per step` of
#239 was startup, not steady state. The unconditional `change_box` was not where
the milliseconds were: it is now skipped when the cell repeats (10000
applications down to 1, `box 0.005 ms/step`), which buys correctness while the
per-step `PPPM initialization` stays at 10001 because `run 0` itself rebuilds the
reciprocal sum. And the calculator's per-step rebuild is worth 0.08 ms, not the
"expected first" it was booked as: `plan 0.004`, `work_items 0.006`, `snapshot
0.042`, `assemble 0.019`. What remains is the engine's `run 0` tail (1.27 ms per
step) and the 0.75 ms per step of transport around it, and both of the engine-side
knobs worth trying have now been tried in scratch trees and refused by numbers, so
the repository never carried either: `run 0 pre no post no` completes all 10000
steps but changes the energy record (a different `md.energies` digest) and buys no
milliseconds, and `thermo 0` is byte-parity clean - the per-step thermo block is
genuinely not what the parser reads - while coming out 2-3% *slower*
(`step_tail 1.282` against `1.247-1.267`). The second of those is the informative
one: it says the step tail's cost is the evaluation itself, not output
formatting, so `run 0` is not going to be made cheaper by removing what it prints.

**How long one LAMMPS step actually is, and how long one `run 0` is.** The
numbers below were measured in the test environment (60-atom TIP3P, 2 ranks, `real`,
`pppm 1e-4`, 8.399 A cell) and are the cost basis for every engine-side decision
in this section, so they live here rather than in a scratch file.

`step_probe.py` pattern comparison, geometry displaced between samples so neither
side is timed on a static box, three repeats with the order alternated:

```
framework's pattern  'run 0' repeated       1.084 ms/step   [1.110, 1.084, 1.084]
engine's pattern     'run 2000' + fix nve     0.241 ms/step  [0.244, 0.238, 0.241]
ratio                                            4.5x
```

So a real LAMMPS step costs 0.24 ms, and what the framework buys per step is not a
step but an evaluation costing 1.08 ms - which is the bulk of the measured
`step_tail 1.25-1.62 ms`, the rest being the `print`/`write_dump` commands around
it. `run_cost.py` splits the same price into opening and steps by timing `run N`
for growing N in a fresh engine each time (`fix nve`, one process per N because a
second instance in a process that closed one aborts):

```
run 1     1.872 ms   run 5    3.894 ms   run 50   14.822 ms   run 1000 234.785 ms
run 2     2.522 ms   run 10   4.950 ms   run 200  48.153 ms
=> marginal step ~0.233 ms, opening ~1.1-1.6 ms
```

That is the whole engine-side story: pay the opening once and take 0.233 ms per
step, or pay it once per framework step. It also bounds what any substitution for
`run 0` can save, which is how the `run 1` candidate below was priced before it was
tried. Two further measurements from the same round: silencing the per-run text
(`to_logfile` versus `-log none`) moved the repeated-`run 0` number from 1.100 to
1.077 ms/step, so **the per-run banner costs 0.023 ms, 2%** - an earlier suspicion
that 18 MB of per-step log text was part of the cost was wrong by a factor of ~37
and is recorded as corrected. And the ceiling if the engine owned propagation
instead of the framework was `step_tail 1.25 -> ~0.25`, i.e. whole step 4.20 ->
~3.2 ms (1.31x) - an integrator-ownership decision, not an optimization.

Probe hygiene, because each of these cost a wasted round before it was known: the
data file must be the **generated** `h2o.data` (2 atom types, 9936 B), not the
template (5659 B) - with the template every `run 0` dies with `All pair coeffs are
not set`, which looks exactly like an engine limitation and is nothing of the sort;
`rsync -aq` push fails silently on this node while `scp` and `rsync -avz` work,
which produced a 1.01x "verdict" from a switch that had never been thrown;
`pgrep -c -f 'pyrai3md test'` matches the checking shell's own command line, so use
`pyrai3md[ ]test`; and one LAMMPS instance per process.

**`run 1` as an evaluation carrier was asked of the engine twice, and the second
asking reversed the first** (#262 measured six spellings and booked them as
refused; #263 re-measured, `run1_fixstate_verdicts.jsonl` and
``). The #262 numbers are real and reproduce to the
digit - what was wrong is what they were taken to be about. That rig loaded the
**unrendered** template settings (8622 lines, 111 atom types), whose line 8620 is
a live `fix fShakeTIP3P tip3p shake 0.0001 10 100 b 952 a 5355` with its `unfix`
commented out on 8622. The package the resident service loads comes from
`write_static_input_files()`: 6 settings lines,
`constraints = {'enabled': False, 'engine': 'framework'}`, and no fix line at all.
So #262 measured `run 1` on an instance that ships in no configuration - and a
constraint fix is exactly the thing that acts inside a step.

Three decks, one spelling per process, references taken in the same process:

| deck | what was loaded | the `run 1` family |
|---|---|---|
| A `tmpl_shakeon` | #262's deck verbatim, SHAKE live | `run 1` dE=`31.189`; `run 1 pre no` dF=`301.54`; `run 1 pre no post no` dF=`301.54` at `timestep 1.0` and `3.0147e+04` at `0.1` |
| B `tmpl_shakeoff` | same force field, one added `unfix fShakeTIP3P` | **all five spellings bit-identical to `run 0`** |
| C `ship` | the production-rendered package, no fix | **all five bit-identical**, and independent of `timestep` |

The dt column is the mechanism, and it is not a matter of reading: a force
evaluation at fixed positions cannot depend on the step size, so an answer that
changes by two orders when the step size drops tenfold is not an evaluation. The
same warning LAMMPS prints (`WARNING: No fixes with time integration, atoms won't
move (src/verlet.cpp:60)`) is also why #262's drift gate read `motion_vs_run0` =
`0.0` on every row: `Shake::initial_integrate` moves the atoms inside the step and
`undo_integrate` puts them back before the run ends, and a difference-of-drifts
test cannot see motion that is undone.

**Skipping the opening is therefore real, and still not adopted.** Issuing
`run 1 pre no` per step measures `step_tail 1.624 -> 1.135 ms` on the shipped
NVE example, and the engine's own account says why: 200 repetitions of `run 0`
draw 201 `PPPM initialization` blocks, 200 of `run 1 pre no` draw 1. It is refused
on two independent grounds, both measured:

1. **Byte parity is lost.** Over 2000 steps the two sides leave step 0 identical,
   first differ at 1e-15 around step 8268, and end 2 ps apart by `1.26e-05` in
   potential energy (`md.energies` digests differ per side and are stable within a
   side, so this is systematic). The resident instance renews its neighbour list
   incrementally across steps where every `run 0` rebuilds it: a different order of
   additions, amplified by a chaotic trajectory. Both sides conserve equally well -
   `Etot span 9.740e-04` on each - so this is a reproducibility question, not a
   physics one, and reproducibility is the contract this path is accepted on
   (#244's adoption stood on ten samples of one digest).
2. **The gain is under the gate.** Median whole step 4.740 -> 4.249 ms = **1.116x**
   against the 1.15x bar, and only 1498 of 2000 steps qualify: 502 steps restate
   image counts or apply the cell and must keep the opening, which caps the
   achievable ratio near 1.16x anyway.

Two failures of my own rig are recorded because they nearly produced a confident
wrong answer twice over. The first A/B driver switched sides by matching the
shipped text of the constant, so after side A rewrote it side B's assertion
failed while the function still returned the status of a later `grep` - ten
samples, both sides, same code, and identical digests that looked like parity
passing. What exposed it was a row this round added to the engine's phase table,
`[perf] per_step_line='...'`, which is kept precisely for that reason: an A/B of
engine-side candidates is unreadable unless every run names the command it
answered with. The second was running two drivers at once against one installed
tree and one sample directory; the wall clocks, the digests and the sample
directory itself all mixed, so that pass was discarded and re-run serially behind
a lock. `tools/deploy.py` also clears non-source files from the install tree, so
probes live in a scratch directory in the test environment and raw verdicts are fetched
back before the next deploy.

The state shipped by #263 is the original: `run 0` per step on both paths, with
the `cold_tail` field, the INIT cold-versus-hot comparison, the per-step tail
selection rule and the `TIMESTEP` exemption in `bringup` C10 all removed rather
than left unused. `build_run_section` carries the measured reason, and
`test_input_builder_box.py` pins that `pre no` appears in neither the loaded script
nor the per-step sequence, so re-adopting it has to be a decision rather than an
accident. The scratch directories these measurements ran in are not part of the delivered
tree: every number that decides anything is written here, a probe's file name is a
label for what was run rather than a path to find, and a rerun is rebuilt from the
builders this section names.

One cheaper-looking change was measured and refused: dropping the `deepcopy`
inside `_build_history_frame` looks redundant because `model_dump()` already
returns fresh lists, but the arrays under `results` are handed out as the very
same objects the live state holds, so the frame would alias the next step's
write-back and stop being a record of its own step. That is a state-semantics
change, not an optimization, and 0.48 ms per step does not buy it.

Timing is always on because two clock reads per phase are nanoseconds against
what they measure, and it is written during the run rather than at the end
because nothing in the framework closes a resident backend (Q46, and
`skill_guide.md`): an artifact held back for a shutdown that never happens is an
artifact nobody collects.

## 12. QM/MM Tasks (`qmmm_field` / `qmmm_energy`)

### 12.1 Two profiles, not a switch
The old `System.qmmm` boolean and the `ff_gradient + qmmm` branch are
retired. QM/MM is two standalone profiles, each with its own input builder,
so `ff_energy`/`ff_gradient` carry no QM/MM conditional:

| Profile | Flow | Engine work | Contract |
|---|---|---|---|
| `qmmm_field` | 1 | A/B `run 0` with q_Q=+1, Coulomb-only | `<title>.esfp` (a.u.), native V/E in `qmmm_field` |
| `qmmm_energy` | 2 | one full-FF `run 0` with q_qm | standard `.energy`/`.force`, `.qm_multipole`, `.qmmm`, `.u_gas.force`, `.qmmm_el.force`, `.grad` |

`System.qm_atoms` is required for both (0-based framework order, sorted and
unique). `qmmm_field` sets Q=+1 at run time and ignores `q_qm` and
`qmmm_field_file`; those are `qmmm_energy` inputs and are not refused on
the field profile (a composite shares one LAMMPS config across both
tasks). The field result carries only V/E: QMMM uses the ESPF potential
representation, so there is no MM-charge handoff (`request_charge` was
removed in CHANGELOG #438). `System.q_qm` and `qmmm_field_file` are the
static fallbacks for a standalone energy run; a runtime caller passes
`q_qm`, `v_mm`, `e_mm` (or `.esfp` content) through
`calculate(..., qmmm_inputs=...)`. The `qmmm_energy` task's `embedding`
bool decides whether the QM part owns the QM-MM electrostatics at all: true
subtracts `qmmm_el` from the classical energy and its gradient from the QM
rows, false leaves them in `E_classic = U_main - U_gas` and reports
`qmmm_el` and its gradient as zero, which makes the QM rows a bit-for-bit
copy of the main gradient. Whether the removed field gradient is handed back
is a separate question, answered by the framework's
`System.qm_contains_field_gradient` (see section 12.3). The field
task needs a neutral MM subsystem; v1 energy keeps QM and MM partitions
neutral (charged-QM background/self treatment is future work). The retired
`qmmm` key, `lammps_shake`, `atom_selection`, a missing charge column and a
cross-boundary bonded row are refused.

### 12.2 Flow 1: qmmm_field
The field model is the energy model with every non-Coulomb pair parameter
zeroed (LJ epsilon and Buck A/B/C); pair style, hybrid sub-style assignment,
cutoffs, `special_bonds` and kspace declarations are untouched, so the
Coulomb surface is identical to Flow 2 by construction. The script declares
the QM/MM groups, `compute QMMM_PEQ QMMM_QM pe/atom pair [kspace]` and
`compute QMMM_GG QMMM_QM group/group QMMM_MM pair yes [kspace yes]`, then
runs A (`q_Q=+1`, MM actual) and B (`q_Q=+1`, MM zeroed) with the same
box/alpha/total charge. The result is

```text
V_i = 2 (peA_i - peB_i)
E_i = FA_i - FB_i
sum V = U_GG, sum E = F_GG
```

The factor 2 is the measured half allocation of `compute pe/atom`; the
`sum V = U_GG` check is a hard guard, not advisory. `kspace yes` is emitted
whenever a kspace style exists (the old single-evaluation computes omitted
it; measured Q-M moved from -11.85 to -38.16 kcal/mol when it was added).

**The two runs must converge to one reciprocal sum.** LAMMPS re-derives
`gewald` *and* the extent of the reciprocal sum from `qsqsum` at every init,
and zeroing the MM charges changes `qsqsum` (22.63 → 3 on the 20-water
fixture), so A and B used to run with `alpha = 0.29013089 / mesh 6³` and
`alpha = 0.2567432 / mesh 4³`. With alpha pinned but the extent left loose the
two runs still sum over different sets of k vectors - measured on the
1581-atom `c2h4_water25` box at `1e-6`, `9 9 9` against `7 7 7` at the same
`G vector = 0.32` - and `V = 2(peA - peB)` is then short by exactly that
truncation while `group/group`, computed inside one run, is not. Production
read that as `residual = 0.056107194227891455 (scale = 168.4866548205339)`,
3.3e-4 relative, and the guard refused the run. The deck therefore pins the
split parameter, alpha, right after the computes, and carries the accuracy
from `ewald_accuracy`:

```text
kspace_style <user style> <ewald_accuracy>
kspace_modify gewald 3.2/rc_coul
```

`rc_coul` is the smallest Coulomb cutoff of the pair styles (last numeric
argument of each segment, `model['init']['styles']['pair']`). The extent is
**not** settable, and that is the reason the accuracy carries the load:
`kspace_modify mesh` is accepted and silently ignored by ewald - two decks
differing only by that line return bit-identical `pe`, `sum pe/atom` and
`U_GG`, and the second probe keeps its own estimate - while pppm honours it
(36 32 36 → 9 9 9 on the same build, the positive control that the command
itself works). `kspace_modify cutoff` is refused outright
(`Illegal kspace_modify command`, `src/kspace.cpp:630`). Measured on that box,
with the reconstruction checked against the deployed logs (same grids, same
residual): `1e-6` leaves 0.05610 absolute, `1e-8` leaves 0.00022, and `1e-10`
and `1e-12` do not move the readings, so `1e-8` is where it converges. The
0-step probe costs the same either way (0.80 s wall).

Convergence, not agreement, is the property worth buying. At `1e-6` probe B's
own tally is `-274.64915` against the converged `-274.62099`, one part in
1e-4 short on its own account, so a scheme that only made the two extents
equal would have passed the guard and still handed the QM engine a V carrying
that error. The same measurement prices the two variants that looked
available: switching the probe to pppm without pinning is *worse*
(0.0561 → 0.217, grids `36 32 36` against `18 16 18`), and pinning a common
pppm grid coarser than the full-charge estimate buys the identity but moves
`U_GG` by 5.9e-5 relative (`36 32 36` pinned against `45 45 45` estimated).
Per-atom `group/group`, which would remove the A/B construction altogether,
is unsupported on the deployed build (23 Jun 2022 Update 1, all three
interpreter environments on it): `Illegal compute group/group command`
(`compute_group_group.cpp:94`) and `Illegal compute_modify command`
(`compute.cpp:133`).

The key is not confined to the probe: it is the accuracy on every
`kspace_style` line this skill writes, the classical decks' re-issue included,
and that costs a periodic trajectory 1.3-2x (50-step NVE, one rank, wall
including engine start: h2o/pppm 0.78 → 1.01 s at `1e-4` → `1e-8`, grid
`5³` → `36³`; water25/ewald 1.46 → 2.96 s, 437 → 4801 k vectors). The shipped
classical and non-probe composite cases therefore spell their template's own
number into the YAML instead of inheriting `1e-8`, and only the QMMM field
cases ask for the converged value; a declared value equal to the template's
spelling leaves the generated deck byte-for-byte unchanged.

The guard stays at `1e-4` of the scale, and round `tmp_20261031_220625` is what
says why `1e-6` would be wrong rather than merely unmeasurable. Its own artifacts
give the pppm side: `h2o_qmmm_field.esfp` sums to `-0.02973654051016026` Hartree,
`-18.659960895989895` kcal/mol, against that run's `c_QMMM_GG[1] = -18.659957` -
so the residual is under `3.9e-6` kcal/mol on a scale of `134.4`, about `3e-8`
relative. (The log prints the tally to eight significant digits, so that reading
has a `5e-7` floor: it resolves a `1.3e-4` budget three orders up. The engine
agrees with itself - at `1e-8` it reports `estimated relative force accuracy`
between `7.19e-7` and `2.95e-6` for these probe runs.) A `1e-6 * scale` guard
would therefore pass h2o with 34x to spare. It would **fail** the ewald box:
`1e-8` leaves 0.00022 absolute there against a `1e-6 * 168.49 = 1.68e-4` budget,
short by 1.3x. The two probes converge at different rates on the two styles, so
the case that decides the guard is the ewald one, and tightening is blocked on
its convergence, not on the instrument.

**A declared dump forbids changing the box type, so the resident package
holds a `reduce` instead.** The loaded field script used to end with a
`dump QMMM_FIELD_RES ... id c_QMMM_PEQ` whose file nobody read: it existed
because LAMMPS refreshes a per-atom compute when something consumes it. A
declared dump is also what makes `change_box all triclinic` illegal
(`Cannot change box ortho/triclinic with dumps defined`,
`src/change_box.cpp:308`), so the first tilted step of a variable-cell
search aborted the resident session with the A/B probe half written. The
package now consumes the same compute with
`variable QMMM_PEQV atom c_QMMM_PEQ` plus
`compute QMMM_PEQS all reduce sum v_QMMM_PEQV`, which holds no file. Both
halves were checked on the deployed build with that deck: with the dump the
conversion errors, with the reduce it goes through, and the per-atom values
the service extracts track the geometry (before a displacement the reduce
and the array agree at `-224.236591155`, after one at `-218.176596704`).
`undump all` is not an alternative: this build answers
`Could not find undump ID: all` (`src/output.cpp:817`) - it takes a dump id,
and a fix built on un-declaring then re-declaring would leave every future
box-type change dependent on remembering to do so.

The per-`run 0` dumps carry `format float %20.17g`: the cold path reads V
from `probe_{A,B}.pe`, and the default `%g` keeps six significant digits,
which quantised V at ~5e-4 and hid everything the guard could see. The
resident path extracts the compute directly and was never quantised.

`tools/verify.py` section H re-derives this from a finished round: it
checks `sum V` against `U_GG` at the production guard (1e-4 of the scale)
and against a converged Ewald sum at 1e-3 relative, so a deck that lost
the pinned line is named instead of passing a guard it never exercised.

The field result carries native V/E and the group tallies; `.esfp` converts
V to Hartree and E to Hartree/(Bohr e) for Molcas. E is the force on a +1
probe; if the Molcas ESPF reader defines the field with the opposite sign,
flip it only in the writer. The field task does not write the production
`.energy`/`.force`.

### 12.3 Flow 2: qmmm_energy
The QM zeros-order charges are written into the data file, the QM region's
bonded rows are removed, and one full-FF `run 0` evaluates U_main and the
full atomic forces. After the service/cold call and before result assembly,
the backend evaluates `in_cell_qq(q_qm, QM positions, lattice, units)`
(U_gas/F_gas), then

```text
qmmm_el = sum_i q_qm_i V_i
F_el,i  = q_qm_i E_i
E_classic = U_main - U_gas - qmmm_el
```

The V/E arrays come from the field task in memory; the energy task does not
re-run Flow 1, read `.esfp` per step, or read `.qm_multipole` back. The
engine's own `.force` and `.energy` files remain U_main's full force and
energy, including the Q-M Coulomb on MM, because they are the verification
record. What the skill RETURNS is the reduced classical surface:
`gradient = d(U_main - U_gas - qmmm_el)/dR` on the QM rows and `dU_main/dR`
on every other row, written beside them as `{title}.grad`. Both removed
terms are reported in the block (`u_gas_gradient_native`,
`qmmm_el_gradient_native`) because one of them is reattachable: the
framework switch `System.qm_contains_field_gradient` (default `true`, the
Molcas ESPF contract, whose analytic gradient already contains `-q_i*E_i`)
decides whether `qmmm_el_gradient` is added back to the QM rows. A QM
provider that reports only the fixed-potential derivative (a polarizable NN)
sets it `false`. Either branch holds exactly one copy of the term.

v1 does not zero Q-Q non-Coulomb coefficients: QM and MM atoms commonly
share LAMMPS types, so type-based zeroing would also remove Q-M/M-M terms.
The shipped water fixture has epsilon=0 on its Q-Q rows; a general Q-Q LJ
treatment is recorded as future work.

### 12.4 Native output and the unit contract
Every LAMMPS profile returns its numbers in the run's native unit, under the
framework's own labels: the parser negates the engine's force into
`gradient` (the sign is what the label means, so it is applied where the
force is read) and marks the result with `_native_units: real|metal`. The
framework scales `energy` and `gradient` by the unit names this skill's
manifest declares (`conversion.native_energy` / `native_gradient`) and never
imports a module of this skill to do it. This is what every other skill
already did: CP2K negates its parsed forces in its own parser
(`cp2k_skill/service/parser/output_parser.py`, `results['gradient'][0] =
-forces`); LAMMPS was the only one that shipped forces under a `forces`
label and had the framework call back into the skill to fix it. `.qmmm`, its aux gradients and the
`.grad` file stay native: they are composite input and audit data, not
framework Properties. `.esfp` is the atomic-unit exception. The direct
backend result is native as well; the public framework result is
Hartree/Hartree-Bohr.

### 12.5 Resident path
The resident protocol is version 2. INIT carries a `qmmm` descriptor
(`profile`, `qm_ids`, and for field `peq`/`gg`/`restore_groups`); EVALUATE
carries `q_qm` for the energy task. The field service:
sets QM=+1, runs A, extracts pe/forces and the A group tally, sets MM=0,
runs B, extracts pe/forces, restores the MM charges by distinct-charge
groups, and returns `qmmm_raw` with `pe_a/pe_b/forces_a/forces_b/u_gg/f_gg`.
The energy service sets each QM atom's charge from `q_qm`, runs the standard
step tail, and returns U_main/forces. The backend runs `in_cell_qq` and the
qV/qE assembly after the reply; V/E never travel over the socket. Measured
on the 6-atom water fixture: field cold/hot V and E agree to dump precision;
energy heat of formation path `U_main`, `U_gas`, `E_classic` agree to 1e-8
kcal/mol, and `qmmm_el` to the cold dump's 6-digit V precision. The resident
energy path also writes the standard native `.energy`/`.force` and the
qmmm auxiliaries each step.

### 12.6 Refusals
A QM/MM input on `ff_energy`/`ff_gradient`; the retired `qmmm` boolean;
missing/ill-formed `qm_atoms`; wrong-length or non-finite `q_qm` on
`qmmm_energy`; a missing
`qmmm_field_file` on `qmmm_energy`; `lammps_shake`;
`atom_selection`; a non-neutral system; a cross-boundary bonded row; a
periodic run with no cell; `qm_probe` outside `qmmm_field` or `qm_centers`
outside `qmmm_energy`; a probe whose ids are not inside `qm_atoms`; a
`qm_probe` equal to `qm_atoms` (there would be no unprobed environment - leave
the key empty and the probe keeps the whole QM set); centers that do not tile
the QM set or that share an atom; a bonded row spanning two centers; a probe
whose environment (MM plus the other centers) is not net neutral. A QM/MM run never falls back to the verbatim
template: `_resolve_ff_model` turns `FFModelUnsupported` into an error.

### 12.7 Tests
`tests/test_qmmm.py` (47 tests, no LAMMPS) covers the profile/validation gate,
partition charge rules, `in_cell_qq`, both assemblers, `.esfp` round-trip,
the energy/force writers, the native-to-framework conversion, the Coulomb
cutoff read off a hybrid *and* a plain `pair_style`, and the assembled field
deck following the declared periodicity (a periodic run pins the Ewald split
and tallies the kspace channel on both computes; a molecular run carries
neither).
`tests/test_qmmm_engine.py` (9 tests, real engine) covers field cold, energy
cold, field hot/cold parity, energy hot/cold parity, the `.grad` file
holding the reduced surface, and the two-center readings of section 12.9 -
probe reciprocity, probe narrowing, the per-center in-cell removal and the
same three through a resident session - on the 6-atom water fixture and a
nine-atom three-water cluster. The full LAMMPS skill suite is 539 passed locally (run from `tests/`, which is where `h2o.data` is shipped); the engine harness
resolves `lammps_path` from `qualification/pmd_env.yaml` with a verified PATH
`lmp` fallback, and a skip is never a pass.

### 12.8 Composite scope
The framework composite was rewritten to this contract on 2026-10-29:
`planning/task.py::_expand_qmmm` plans `qmmm_field -> QM -> qmmm_energy`
with contributions `[0, +1, +1]` and appends the `field`/`qmmm` capability
tokens to the two LAMMPS tasks; the skill resolves its own
`qmmm_field`/`qmmm_energy` profile from the skill jobtype
(`energy gradient field` / `energy gradient qmmm`), so the framework never
writes a skill profile name. `Calculator._run_qmmm` runs the three
stages, reads the field task's atomic-unit `.esfp` for the Molcas
`external_potential`, resolves `q_qm` from the QM `computed_charges` or
`molecule.q_qm`, and `workflow/multiscale/composite.py::assemble_qmmm`
returns `E = E_QM + E_classic` on the classical surface the skill already
reduced: the Q rows arrive as `d(U_main - U_gas - qmmm_el)/dR`, and
`System.qm_contains_field_gradient: false` adds `qmmm_el_gradient` back to
them (see section 12.3). The framework applies the manifest's declared units
before aggregation, so a mixed-skill sum is in Hartree/Hartree-Bohr. The retired `qmmm`/`mm_mm` keys are no longer read.

### 12.9 Several QM centers: `qm_probe` and `qm_centers`

A framework run with more than one QM region hands each duty its own center:

- `qm_probe` (0-based, a subset of `qm_atoms`, `qmmm_field` only) narrows the
  A/B probe to one center. `QMMM_MM` is `all - probe`, so the **other centers
  land on the environment side of the difference**: `V` at a probe atom is the
  potential of the MM charges plus the other centers' current charges, which is
  why one electrostatic route covers the inter-center field and no third group
  is needed. The environment charges are the whole QM array in `qm_atoms` order
  (`calculate(qmmm_inputs={'q_qm': ...})`): the cold path rewrites them into the
  data file after the package is written, the resident path sets them per atom
  and restores them after leg B. The A/B neutrality gate is therefore a
  run-time question per round, not a model-build question.
- `qm_centers` (list of groups tiling `qm_atoms`, `qmmm_energy` only) splits the
  in-cell Q-Q removal per center: `u_gas_centers_native` holds one number per
  group and their sum is `u_gas_native`. The inter-center pair stays on the
  classical surface, which is what makes the composite's `qmmm_el` count it
  exactly once.
- `{title}.esfp` gained a leading atom-id column. Rows are bound by id in the
  framework (`Calculator._qmmm_external_potential`), and a file with the old
  four-token rows still reads, but only for the one-center deck that wrote every
  QM atom in ascending order.
- The model cache is per center: `{project}.ff.json` would otherwise let two
  centers in one directory read each other's partition, so the cache name and
  `_hash_config` carry the `qm_probe`/`qm_centers` fingerprint, and one center
  writes neither (byte-identical to before).

Measured on the cluster fixture (`test_qmmm_engine.py`), where the classical
water is deliberately uncharged so any probe potential can only come from the
other center: `q_A . V(A) = q_B . V(B) = 0.01200838` Hartree (the same
inter-center pair from either side); center A's rows `0.552490 / 0.627721 /
0.608354` Hartree per element charge against `0.000000` measured over the union
partition; per-center gas `-9.17466068 / -5.38138711` summing to the total, the
first equal to the standalone one-center run of that water to relative 1e-9.
The same three readings hold with `python_enabled: true`.
