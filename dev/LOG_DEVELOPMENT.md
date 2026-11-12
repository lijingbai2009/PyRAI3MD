# PyRAI3MD Log Design Specification

Last updated: 2026-11-06 13:50 (the adaptive paragraph gained the `&method alignment` block, transcribed from the three shipped local carriers of `examples/adaptive_*.yaml`: one occurrence per run, immediately before the final summary, five lines, with `reference type` reading `single` / `oniom2` / `qmmm` and only the QMMM run adding a `warnings:` line. The rest was not re-audited. Earlier: 2026-11-04 11:08 (#451: section 8 owns the three multi-center QMMM blocks - the per-center rows of `&qmmm energy decomposition`, the `: center N` suffix on the external-field block, and `&qmmm charge iteration` with its per-round and per-center residuals - and states the ownership rule they follow. The rest of the file was not re-audited this round. Earlier: 2026-10-24 08:45 (sections 14 and 17 now state the 0-4 content levels: 0 progress/error, 1 structure, 2 properties, 3 hessian and missing markers, 4 debug; the input header is exempt. Section 17 also records the new `&cell` and `&optimization setup` blocks and the merged method rows in `&optimization progress`. Earlier 2026-10-22 22:20 (the start banner's jobtype-section list now includes `search`, and `format_result_block` is quoted with its full current signature. Before it, 2026-10-09 16:10)

---

> **Stress output (2026-09-24)**: see CHANGELOG #213/#214. `format_result_block`
> gained the `&stress tensor in Hartree/Bohr^3` section (`_format_stress`): it
> projects the current state's 3×3 stress tensor and appends the derived
> instantaneous pressure P=−Tr/3 (in Hartree/Bohr^3 and in bar); when the backend
> filled nothing (missing or all zero) the section shows 'Not computed' behind the
> verbose gate, and a non-NPT run is unaffected. The data comes from each skill's
> stress-specific file (CP2K/LAMMPS) or data file (xTB gradlatt).

> **Multi-center unification (refactor R6, 2026-09-08)**: see CHANGELOG #204. The
> per-step result block of the run log gained the center protocol:
> `format_result_block` in `utilities/print_results.py` takes an optional `centers`
> argument (built by `snapshot_centers(snapshot)` from the history frame's
> `centers` / `electronic_outcome[c]['density']` /
> `nuclear_sample.active_config_pre`). On a multi-center frame the
> `&electronic state` and `&state population` sections list, in center order
> (1-based), each center's active state and its per-state populations (the
> population taken from the diagonal of that center's density's real part); a
> single-center frame (including C=1 with an environment partition) keeps the
> original global state/population output unchanged. `log_step` in
> `kernel/logger.py` builds `centers` from the snapshot and passes it through.

> **Multi-center region charge/spin (2026-09-09)**: each per-center record built by
> `snapshot_centers` additionally carries the region `charge`/`multiplicity` (read
> from the frame's `centers[c]`, i.e. `CenterSpec.to_dict`), and a multi-center
> frame's result block gained the `&region charge & spin multiplicity` section
> (`print_results._format_center_regions`): it lists every center's (1-based) region
> charge and spin multiplicity and appends one line with the whole-system L(real)
> aggregates (`sum(charge_c)` and the unpaired-electron cancellation multiplicity
> from `aggregate_system_multiplicity`) so the user can check the charge
> redistribution ambiguity of a non-neutral or open-shell system. A single-center
> frame does not print that section. The data source is the PartitionPlan bound to
> the state (see `dev/MULTICENTER_DEVELOPMENT.md` §4.4.1 / §11.2).

## 1. Overall design principles

The PyRAI3MD log serves three kinds of reader at once:

- Human users: quickly understand the calculation type, the run status and the
  result;
- CLI / shell: able to find things by keyword;
- LLM / Agent: able to locate information through stable sections and keywords.

Hence: **Structured, Hierarchical, Searchable, Human-readable**.

Core principles:

- plain text;
- one unified `&section` syntax;
- section names stable and unique;
- inside a section, one unified `key : value` style;
- numbers, units and status expressed explicitly;
- no reliance on colour, emoji or elaborate ASCII art to carry meaning;
- verbose changes the amount of information only, never the basic grammar;
- the same kind of information uses the same expression across workflows.

## 2. Section markers

Use:

```text
&section
```

and not:

```text
[section]
```

For example:

```text
&run

&system

&method

&step

&result
```

Rules:

- a section marker occupies its own line;
- `&` is at the start of the line;
- section names are lowercase;
- section names are snake_case;
- section names contain no spaces;
- section names are not abbreviated unless the abbreviation is already formal
  PyRAI3MD terminology;
- one blank line is left after a section ends, for visual readability.

For example:

```text
&electronic_energy
...

&gradient
...
```

and not:

```text
&Electronic Energy
```

or:

```text
&E
```

## 3. Inside a section: `key : value`

This is the single most important rule of the whole log.

Preferred:

```text
Energy       : -77.0742106388 Eh
Gradient RMS : 1.42e-05 Eh/Bohr
Status       : CONVERGED
Steps        : 13
Walltime     : 82.4 s
```

and not:

```text
Energy = -77.0742106388
```

nor:

```text
Energy
    -77.0742106388
```

Why: `key : value` has three advantages -- it is easy for the eye to scan, easy to
handle with grep / awk / sed, and easy for an LLM to recognize as a key-value pair.

## 4. Key naming conventions

All keywords:

- in English;
- lowercase;
- snake_case when they contain several words;
- one physical quantity uses one keyword globally;
- the keyword does not change because the workflow differs.

So in principle:

```text
energy
gradient_rms
gradient_max
status
steps
walltime
```

and never several spellings for one concept:

```text
Energy
E
electronic_energy
total_energy
```

Keyword consistency matters more than visual design.

## 5. Key alignment

Within one section use fixed-width alignment:

```text
&result
status       : SUCCESS
energy       : -77.0742106388 Eh
steps        : 13
walltime     : 82.4 s
```

rather than:

```text
&result
status : SUCCESS
energy : -77.0742106388 Eh
steps : 13
walltime : 82.4 s
```

Alignment noticeably improves the readability of numeric output. Table box-drawing
for the sake of alignment is not recommended.

## 6. Units

The unit follows the number directly:

```text
energy : -77.0742106388 Eh
gradient : 1.42e-05 Eh/Bohr
dipole : 2.31 Debye
time : 82.4 s
temperature : 300 K
```

rather than:

```text
energy : -77.0742106388
unit   : Hartree
```

For a section containing many values in the same unit, the unit may be declared in
the section header or section metadata and then omitted from the rows. For example:

```text
&gradient
unit : Eh/Bohr

1  C  ...
2  C  ...
```

Which of the two is used may be decided per data type, but one class of data must
stay consistent.

## 7. Data blocks versus key-value

Not everything fits `key : value`.

Scalar information uses:

```text
energy : -77.0742106388 Eh
status : CONVERGED
```

Vectors / matrices / coordinates use a dedicated data block:

```text
&coordinates
unit : Angstrom

1  C  0.000000  0.000000  0.000000
2  C  1.340000  0.000000  0.000000
...
```

That keeps both readability and machine parsing.

## 8. Order inside a section

Follow a fixed information order rather than letting each skill decide. The general
sequence is:

```text
context -> configuration -> status -> primary result -> secondary result -> detailed data
```

For example, a calculation step:

```text
&step
step   : 12
status : RUNNING

energy       : ...
delta_energy : ...
gradient_rms : ...
gradient_max : ...

...
```

This way an LLM always knows where the main information of a new section lives.

The multi-center QMMM blocks follow that order and one rule of ownership: what
belongs to an evaluation is printed once per evaluation, not once per task.

- `&qmmm energy decomposition in Hartree` keeps its global rows (`E_QM`,
  `E_classic`, `U_main`, `U_gas`, `qmmm_el`, `total`) and gains
  `center N E_QM state S` / `center N U_gas` rows only when there is more than
  one center. A single center prints exactly what it printed before.
- `&qmmm external potential/field in a.u. (atom, V, Ex, Ey, Ez)` is one block
  for one center, and gains the suffix `: center N` when there are several,
  because with two probes the union view would hide which probe sampled which
  row.
- `&qmmm charge iteration` is new with the iteration: the mode, the rounds it
  ran, its bound and tolerance, `max |dq| round K` per round, one line per
  center with that center's residual sequence, and `stopped by` naming
  tolerance / single round / maxiter. It is a trajectory variable, so a reader
  comparing two frames cannot tell `sc_charge: false` from a run that converged
  on its first round without it; the block is absent (not empty) when the
  evaluation ran no charge rounds.

## 9. Iterative workflow steps must share one format

For iterative calculations (optimization, MD, NAMD):

```text
&step 1
...

&step 2
...

&step 3
...
```

or a uniform step keyword:

```text
&step
index  : 1
...
```

Which of the two is used can be decided later, but all of PyRAI3MD must use one.

Current implementation (`kernel/logger.py` + `utilities/print_results.py`):

- MD and the other per-step workflows call `log_step(step, snapshot)` for every
  step, which builds the result block with `format_result_block()` (coordinates,
  background charges, electronic states, velocities, energies, kinetic energy,
  gradients, NAC, SOC, dipoles, thermodynamic quantities, the stress tensor and so
  on). The section names are module constants in `print_results.py`
  (`SECTION_ENERGIES` / `SECTION_KINETIC` / `SECTION_VELOCITIES` /
  `SECTION_CONTRIBUTIONS`), and the post-run verification script imports those
  constants instead of copying the names out again: a miscopied section name raises
  no error, it only reports "nothing comparable" on a log full of numbers. A
  property that is absent prints its title with 'Not computed'.
- The stress section (`&stress tensor in Hartree/Bohr^3`) prints the current
  state's 3×3 tensor and the derived instantaneous pressure P=−Tr/3 (with the bar
  conversion); when the backend filled nothing (missing or all zero) it shows 'Not
  computed' behind the verbose gate, and non-NPT runs are unaffected.
- optimize prints an `&optimization progress` block per step (`log_optimize_step`):
  numbers uniformly at 8 decimals, right-aligned; each of the five convergence
  criteria (Delta E, Gradient RMS, Gradient Max, Displacement RMS, Displacement Max)
  shows its current value, its threshold and the True/False verdict on one line; the
  last line prints Converged.

What must never appear is several styles side by side:

```text
* Step 1
STEP 2
---- Step 3 ----
Iteration 4
```

## 10. Sections and visual hierarchy

Plain text should still have a visual hierarchy. Recommended:

```text
================================================================================
PyRAI3MD
================================================================================

&run
...

&system
...

--------------------------------------------------------------------------------
&step
...
--------------------------------------------------------------------------------

&step
...

================================================================================
&result
...
================================================================================
```

That is: `&section` is the semantic level, blank lines are the reading level, and
`---` / `===` are the visual level. The rules themselves must never carry meaning.

## 11. No excessive decoration

Not recommended:

```text
╔════════════════════════════╗
║      OPTIMIZATION          ║
╚════════════════════════════╝
```

and neither is heavy use of:

```text
🚀
✓✓✓
✨
████████████
```

A very limited amount of visual furniture is allowed, but no calculation status may
depend on it. The core statement must always be:

```text
&result
status : SUCCESS
```

and not:

```text
✓ Optimization finished
```

## 12. Status must be machine-searchable

Statuses use a stable keyword:

```text
status : RUNNING
status : SUCCESS
status : FAILED
status : CANCELLED
status : TIMEOUT
```

rather than natural language:

```text
status : Calculation completed successfully.
```

so that this alone gives the status:

```text
grep "status :" run.log
```

The outcome banners a workflow writes at the end of its run
(`log_optimize_summary`, `log_ml_summary`) are prose in this same log and are
deliberately not status fields: the machine-readable statement is the footer's
`Status:`, and a banner is the sentence that explains it -- which channel was
empty, which step count was reached, which record the run came back with.
grep for a status; read the banner for a reason.

A dispatched child writes no run log at all. The handle a batch copy
carries points at the parent's absolute `<title>.log`, and two independent
handles on one file overwrite each other -- a four-point scan left 10475
NUL bytes in its own log before this rule existed. `RunLogger.__deepcopy__`
and `__getstate__` therefore hand out a closed logger: the child's report
reaches the parent in its result, the parent prints the batch, and the
child keeps its own evidence in its runtime directory (the skill's log, the
engine output, `result.json`). A process that starts from a saved `.rt`
state is not a copy: it builds its own logger in its own working directory.

## 13. Warnings and errors get their own sections

Do not mix errors into ordinary prose. For example:

```text
&warning
type    : convergence
message : ...
```

and:

```text
&error
type    : backend
message : ...
```

so they can be located quickly:

```text
grep "^&error" run.log
grep "^&warning" run.log
```

The detailed backend message stays inside the section.

## 14. The core rule of verbose

Verbose does not change the format; it changes how detailed the sections are. This
rule is the important one.

The levels are cumulative. The **input header echo is exempt**: `&control`,
`&system`, its coordinate/charge tables and the full keyword dump are written in
full at every level, because they describe what the run was asked to do rather
than what it found.

| Level | Result-block content |
|---|---|
| 0 | step header; electronic and kinetic energy; constraint energies and geometry; energy contributions/composite; prediction error; `&optimization setup` and `&optimization progress`; ML records; outcome banners |
| 1 | + coordinates, `&cell`, background charges, electronic state, region charge/spin |
| 2 | + population, velocities, gradient, NAC, SOC, dipole/transition dipole, thermodynamics, frequencies, normal modes, thermo corrections, stress, ensemble |
| 3 | + hessian, every `Not computed` marker, rejected path probes |
| 4 | debug artifacts (the FSSH intermediate dump), unchanged |

Every row of `&optimization progress` is a progress/convergence/error value, so
that block and `&optimization setup` are level 0 in full. The verbose level
controls the property block that accompanies an optimization step (structure at
1, properties at 2) and whether rejected path probes are printed (3). A path's
accepted points write the same level-gated property block as a relaxation step;
band rounds write the progress block only.

That is: **the same keyword keeps exactly the same name and the same semantics at
every verbose level**, so an LLM does not have to learn a different language per
verbose mode.

## 15. Input and log speak one language

Since the PyRAI3MD input already uses:

```text
&control
&system
&environment
...
```

the log continuing with:

```text
&run
&system
...
```

is the natural choice. PyRAI3MD then forms one unified text language:

```text
Input
    &section
        key = value

Log
    &section
        key : value
```

with a natural division:

```text
=  a user declaration / configuration
:  a program report / result
```

That difference is deliberate and fits the architecture.

## 16. Log header

The header may be visually strengthened a little but stays simple. Current
implementation (`kernel/logger.py`, `write_heading`):

- the first line is `PyRAI3MD Start: <what_is_time()>` (a timestamp formatted like
  "Aug 21 2026  10:30:00");
- then the logo and the version information;
- then the `&control`, `&system` and `&environment` sections in order;
- then the jobtype sections from `JOBTYPE_SECTIONS` (md → `&dynamics`; adaptive →
  `&dynamics` + `&adaptive`; optimize → `&optimize`, which is also the section the path
  family reads; search → `&search`; `kernel/logger.py:95-102`);
- finally the `&skills` section.

An adaptive run adds two blocks. `format_adaptive_round` writes one
`&adaptive round` block per round (trajectories, selected frames, reference
counts, dataset/model paths) at the point the round's statistics are written.
`format_adaptive_summary` writes the final `&adaptive sampling` block from
`results['adaptive']` (round table, stop reason, final dataset and model). The
final block goes through the ordinary `log_result` path
(`utilities/print_results.py::format_result_block`), so an adaptive run keeps
the same tail as every other non-iterative workflow.

A multiscale adaptive run adds one block of its own, written once per run
immediately before the final `&adaptive sampling` summary (measured: one
occurrence in each of the three shipped local carriers):

```text
  &method alignment
--------------------------------------------------------------------------------
  theory surface           6 atom(s)
  reference type           oniom2
  label source             layer (high)
  source task(s)           qm
  auxiliary channels       none
```

It is the correspondence the planning layer derived (`dev/methods/ADAPTIVE_DEVELOPMENT.md`
section 14), stated once rather than left to the reader: which atom set the
network replaces, what the reference queue was classified as (`single`,
`oniom2` or `qmmm`), which reference term or total supplies the label, and
which auxiliary channels - if any - the frame produced. `source task(s)` names
the reference task ids; a QMMM reference prints `auxiliary channels  mm_energy,
q_qm` and adds a `warnings:` line saying the fieldless reference produces no MM
field, while both mechanical spellings print `none`. The same object is written
machine-readably into `<title>.adaptive.json` as `method_alignment`, which is
what `tools/verify.py` section J reads. The block is printed from the plan, not
from a guess, so the five lines above are the same five lines in all three
adaptive shapes.

No elaborate banner is needed.

### 16.1 The `&constraint plan` section (2026-10-01, CHANGELOG #271, #273)

`_write_input_section(state, 'system')` prints the input as given, and there the
constraint block is only one `constraints: {...}` dump line -- the
auto-numbered ids are empty in it, so a reader cannot see that six O-H bonds became
several actions. `write_heading` therefore prints one more section after the
jobtype sections and before `&skills`:

```text
  &constraint plan
-------------------------------------------------------------------------------
  input units angstrom length, degree angle, hartree energy; the numbers below are internal (bohr, radian, hartree)
  bias       harmonic     bias1          cv cv_of_bias1 ramp force_constant(1-20)
  wall       sphere       wall1          6 atoms (1 2 3 4 5 6) law hard, target atom, 6 objects, center from declared
  fix        mask         fix1           1 atom (6)
  constraint shake        constraint1    1 rows
  constraint shake        constraint2    1 rows
  group      oh1            2 atoms (1 2)
  cv         cv_of_bias1    bond       2 atoms (1 4)
  cv         r1             bond       2 atoms (1 2)

  notes:
    'shake_bond_ref' broadcast over 2 objects
    'wall_box_lengths' ignored: 'wall_type' does not name that geometry
    bias 'bias1' ramp 'schedule_of_bias1' stops at the last step of the run (md.step = 20)
```

Four conventions (the block above is taken from a real run, it is not a mock-up):

- **One line per action** (kind / family / id / what it acts on / the ramp window),
  then the two tables for groups and cvs. The id is minted by the schema and does
  not exist in the input, yet every later energy-decomposition line and every
  restart refusal names it. When an action covers more than 6 atoms only the count
  is reported, not the numbers -- the numbers are in the input, and what this
  section has to say is which action holds how many atoms.
- **The wall line writes out all three independent choices**: `law` (hard: zero
  force inside the boundary; soft: pulls toward the center throughout), `target`
  (an atom or a center of mass / geometry), and `objects` (how many objects this
  wall actually settles: without grouping an `atom` target means the atom count, a
  `com`/`cog` means one), plus where the center came from -- `declared`
  (`wall_center`), `wall_ref` or `selection` (the center of the chosen group). The
  center default is no longer the coordinate origin, so the log must say where it
  came from; otherwise a reader who looks only at the input assumes an unwritten
  center means none.
- **`notes` is the list of "not written in the input, decided by the system"**:
  broadcasts, completions, keywords blocked by `wall_type`, and a ramp stop closed
  by `md.step`. These decisions are not part of the fingerprint
  (`ConstraintPlan.to_dict` excludes notes), so rewording them never makes an old
  checkpoint's restart be refused.
- **Default-off extends into the log**: with `plan.is_empty` the whole section is
  omitted rather than printed as an empty title
  (`test_constraint_integration.py::TestRunRecord::test_a_run_with_no_constraints_prints_no_section`).

The criteria live in `test_constraint_integration.py::TestRunRecord` (five items:
entities and ids present, a broadcast becoming a note, an ignored geometry becoming
a note, the wall line's law/target/objects/center source, and no such section at all
when there are no constraints).

## 17. Log footer

The final result needs an explicit ending. Current implementation
(`kernel/logger.py`, `finish`) ends with one line:

```text
PyRAI3MD End: <what_is_time()> Total: <how_long> Status: <status>
```

so that even a reader who only does:

```text
tail -50 run.log
```

can judge: whether it succeeded, the final result, how long the calculation took,
and whether an error needs to be read.

Screen output: by default the log content goes into the log file only; it is printed
to the screen as well only when `control.silent = 0` (silent defaults to 1, i.e.
file only).

Content level: the content level of the result block is controlled by
`control.verbose` (default 2):

- at `verbose 0` only the progress/error blocks are printed: step header, energies,
  constraint decomposition, energy contributions, prediction error, and the
  optimization setup/progress tables;
- at `verbose 1` the structure is added: coordinates, `&cell`, background charges,
  electronic state and region charge/spin;
- at `verbose 2` (the default) the property arrays are added: population,
  velocities, gradient, NAC, SOC, moments, thermodynamics, frequencies, normal
  modes, thermo corrections, stress and ensemble diagnostics;
- at `verbose 3` the hessian, every "Not computed" marker and rejected path probes
  are added;
- at `verbose 4` the FSSH integrator additionally writes each step's intermediate
  data (the A/D/B matrices and the per-substep g probabilities) to `{title}.fssh`
  for debugging, in the same directory as `{title}.md.pop`; when it is off there is
  no extra cost at all.

## 18. In short: the PyRAI3MD log grammar

These are the rules to develop against:

```text
SECTION
    &<section_name>

KEY-VALUE
    <key> : <value>

CONFIGURATION
    <key> = <value>

DATA BLOCK
    &<section_name>
    ...
    <tabular data>

NAMING
    lowercase snake_case

STATUS
    status : <ENUM>

UNIT
    value + unit

SEPARATOR
    "-" for local separation
    "=" for major boundaries

VERBOSE
    controls information density only
    never changes semantic names
```

## 19. Implementation map

How the clauses of this specification map onto modules:

- Log writing and section organization: `kernel/logger.py`. `get_logger()` is the
  logging entry; `RunLogger(state, log_dir=None, start_time=None)` owns the run-level
  log (the start line, the heading, the per-step blocks, the optimize progress block,
  the end line).
- Timing and timestamps: `utilities/timer.py`. `what_is_time()` produces the
  timestamp; `how_long()` formats the elapsed run time.
- Result-block formatting: `utilities/print_results.py`.
  `format_result_block(system, results, step=None, runtime=None, thresholds=None,
  verbose=2, centers=None, dynamics=None)` builds the per-step and final result blocks,
  including the per-center tables and the threshold verdicts (`utilities/print_results.py:134-136`).
- Coordinate and charge tables: `utilities/structure_formatter.py`. `print_coord()` /
  `print_charges()` produce the aligned table text; `print_coord(..., indent='')` is the
  trajectory-frame layout, whose rows start with the element symbol.

## 20. Timing conventions

`how_long()` rounds first and then formats as DD:HH:MM:SS, so a sub-second run is
never displayed as 00:00:00:00.

`what_is_time()` uses the `"%b %d %Y  %H:%M:%S"` format.

`RunLogger` accepts a `start_time` argument (the timestamp at the `run()` entry), and
the footer's Total is computed from it; without one, the construction time is used.
