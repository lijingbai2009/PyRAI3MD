# PyRAI3MD System Constraint, Frozen Atom and Bias Potential Design

Last updated: 2026-11-12 19:20 (composite registry references updated to validate_plan/aggregate_results. Earlier: 2026-11-09 14:20 (the classic constraint writer moved out of `schema/adapter.py` to `qualification/certification/constraint_writer.py`; the rest was not re-audited. Earlier: 2026-10-22 22:00 (D-14 now reads as closed for a bond, an angle and a torsion - the angle/torsion cell force is measured in `test_constraint_cell_coupling.py` and a fourth kind is refused by name; the header note and the round-status line follow, §15's test map now names the merged files, and the draft banner no longer pins a count. Before it, 2026-10-13 14:22: section 5.2 rewritten and the `apply_constrain` / `kabsch` locations checked against the code for the `data/geometry.py` move and the single-writer ruling on `System.image`; the rest of the file was not re-audited)

> **Test location, 2026-10-04 (CHANGELOG #299).** `pyrai3md/utilities/tests/` was merged into
> `qualification/verification/` (142 tests: geometric, image-flag, lattice, print-results, timer),
> which is now the single home of the framework unit tests. The dated gate readings below keep the
> path they were taken at: wherever this document says `utilities/tests`, it means those same five
> files as they were located on that day.

---

> **Status (2026-09-29, CHANGELOG #241, #249, #252, #254, #264, #271 and #273).** Design document, and the
> subsystem is **merged into the framework**. The modules of C0, C1, part of C2 and part of C4 now
> live at their final paths (`data/constraint.py`, `utilities/{cv,selection,
> rmsd}.py`, `workflow/constraints/*`, and since #271 the constraint records in `schema/config.py`
> with their normalization in `schema/normalize.py`, their refusals in `schema/validate.py` and
> their classic text reader in `schema/adapter.py` and writer in
> `qualification/certification/constraint_writer.py`) and are called by the
> Calculator, the integrator, the ensemble thermostat and the serializer (§19, §20). They are covered
> by 782 tests selected with `-k constraint` under `qualification/verification/` (789 repository-wide
> by the same selection; `python3 -m unittest discover -p 'test_constraint*.py'` runs 775), plus the
> 132 of `utilities/tests/` whose geometric, image-flag and lattice cases belong to this subsystem,
> and the repository-wide run is 3410 passed with 6 skips (all four re-measured on 2026-10-01 after
> #280 added three torsion tests on top of D-25 four; the six skips are the `test_fssh*` cases
> numba gates, none of them constraint. They read 792 / 792 / 785 / 3425 before #273 retired the
> cavity family). **Still default
> closed** in the sense that matters: with no `&constraint` block and no classic constraint
> field carrying intent, the plan is empty, every seam returns immediately, and an unconstrained run
> reproduces its own numbers exactly. What remains out is stated by what is missing rather than
> by a list of refusals: a condition kind outside bond, angle and torsion has no priced cell
> force and is refused by name (§18.1 D-14, CELL_OPT_DEVELOPMENT.md section 4.3), and a
> collective variable whose atoms the bond graph does not
> connect is refused by design in a periodic cell (§5.2 layer 4). The analytical curvature of a
> bias (bond, angle, torsion) and of every wall geometry is in, and the curvature of a history
> potential over several variables is refused by name with its reason measured (§18.3, #268).
> Angular constraints (§18 A4), periodic image-aware geometry (#266, §18.2),
> well-tempered deposition, the velocity half of an angle or torsion, and `bond_type`
> expansion (#267, §18.3) are implemented and tested.
> closed** (§1.1) in the sense the design requires: every seam is guarded by `plan.is_empty`, and
> the milestone gates in §14 and the suite in §15 are what a later round verifies against.
> Potential definitions, CV mathematics and the original DSL sample are kept in this file as
> 附录 A, and the round-by-round plan and reconciliation as 附录 B; sections 1-32 stay the
> authority for design decisions, code integration points and milestones. Where an appendix
> describes current code, the sections are correct.
>
> Three findings from the 2026-09-28 code audit changed the shape of the original draft and are
> carried through the whole document:
>
> - Additive potentials cannot be applied at `apply_constrain`; that call site runs after the
>   second velocity kick, so bias and wall forces must enter through the result finalizer inside
>   `Calculator.run_in_process` (see §7).
> - Constraints belong to whoever moves the atoms. The skill side already enforces this rule at
>   validation time; the framework side must mirror it instead of inventing a second protocol (§3.7).
> - The framework kept no topology object in the PBC stage A+B refactor, and the one bond
>   perception routine (`utilities/bonds.py::perceive_bonds`) was periodic-only and
>   image-specific when the first version was drafted; it has since gained the non-periodic
>   branch and the topology service. The first version therefore takes explicit atom selection
>   and drops the draft's plan to add a covalent radii table (§3.8).

## 1. Design Goals and Overall Judgment

The goal is one backend-independent constraint interface that takes geometry, constraint
parameters and the runtime context, and returns corrected geometry, per-item additive potential,
additive gradient, and the total energy and gradient synthesized from an existing electronic
structure result. MD, optimization, single point and later sampling share one numerical
implementation.

Five kinds of action are organized together but implemented separately.

| Kind | Modifies coordinates | Defines added potential | Runtime state needed |
| --- | --- | --- | --- |
| `constraint` | yes, projection onto the constraint manifold | no | targets, solver diagnostics, MD half-step information |
| `fix` | yes, restore frozen components | no | initial anchors, degree-of-freedom mask |
| `bias` | no | yes | static parameters or schedule |
| `wall` | no | yes | geometry, target, schedule |
| `metadynamics` | no | yes | CV or reference-structure history, deposition progress |

The distinctions that must survive implementation are recorded in the appendix §1 and are not
repeated here. Two of them decide where code goes, so they are restated as requirements:

1. If a hard constraint changes `R` into `R*`, the total energy is `E_base(R*)`. Adding a
   "correction" to the energy of the old geometry and labelling it with the new geometry is not
   allowed, so projection must happen before the electronic evaluation that produces the reported
   `E/G` pair, not after.
2. `fix` and SHAKE carry no potential of their own. Their per-item energy is defined as zero, and
   constraint reaction, impulse, projected displacement and initialization energy change are
   separate diagnostics, never disguised as potential terms.

Overall judgment: the framework is missing a constraint engine, not a configuration surface. The
`System` model already carries nineteen flat constraint and external-potential fields, and several modules reserve
named slots for this work, but nothing reads them at runtime. The productive order is therefore to
fix the wiring (§4, §5), then ship static soft potentials end to end (§7, §8, C1), and only then
build the strict-constraint dynamics that need a correct commit point (§3.4, C2, C4).

(The sentence above is the 2026-09-28 audit judgement and is kept as written because the order it
argued for is the order that was executed. It no longer describes the tree: the engine now exists at
`workflow/constraints/`, `apply_constrain` delegates to it, and the §2 inventory rows and the §3 gap
sentences below were corrected on 2026-10-01 to say so.)

Out of scope for the first two milestones, stated positively so that silence is not read as
support: PBC-affected constraints of any kind, variable-cell freeze, `bond_type` automatic
expansion, well-tempered metadynamics, constrained surface hopping, constrained NPT, analytical
bias Hessians, and any constraint applied while a skill integrates the system.

### 1.1 Default-closed contract

Other module development is in flight, so this subsystem ships closed and is verified later.
The current tree already satisfies most of that, and the audit result is recorded here rather than
assumed: of the nineteen `System` constraint and external-potential fields, eighteen have **zero**
runtime consumers (`constrain`, `cbond`, `cangle`, `cdihedral`, `tbond`, `tangle`, `tdihedral`,
`fbond`, `fangle`, `fdihedral`, `cavity`, `center`, `center_type`, `groups`, `shape`, `factor`,
`scale`, `compress`), `apply_constrain` returns its input unchanged, and no constraint module
exists. The single exception is §1.2 below.

The contract every milestone inherits:

| ID | Requirement | How it is checked |
| --- | --- | --- |
| DC-1 | An absent or empty `system_constraint` means no code path: no controller construction, no finalizer registered, an empty plan, and no new work in the timestep | unconstrained input reproduces current output byte-for-byte, part of the C0 and C1 gates |
| DC-2 | A consumer tests the plan for emptiness before acting; an empty list is never allowed to produce side effects through iteration, allocation or shape checks | code review plus a DC-1 test on every phase |
| DC-3 | History frames, results keys and the `.rt` payload do not grow when the plan is empty: constraint keys are omitted, not written as zeros or empty arrays | frame and payload size compared against the unconstrained baseline (`test_constraint_composition.py`) |
| DC-4 | Nothing is enabled by a default value. Only non-empty user selection activates a path, which is why the legacy `fbond=10.0`, `shape`, `scale` defaults are not evidence of intent (§13) | `test_constraint_schema.py` asserts a defaults-only config yields an empty plan |
| DC-5 | Each phase lands with the existing suite unchanged in count and outcome, and the new tests are the only added behaviour | full-suite comparison before and after, as recorded in `CHANGELOG.md` |

### 1.2 The one field that is already not closed, and what to do about it

`freeze` is read by exactly one consumer, the degrees-of-freedom counter
(`pyrai3md/workflow/integrator/ensemble.py:223-226`), so an input that sets `freeze` today lowers
`N_df` used for temperature while nothing prevents those atoms from moving. This is a defect on the
thermostat path, not a missing feature, and it is the reason §3.1 asks for a regression test rather
than a new test.

Verified blast radius: no shipped example, qualification or skill test yaml sets `freeze` (the one
textual match, `pyrai3md/examples/md_lammps_nve.yaml:180`, is a comment about the box), so no live
input is affected.

Handling decision, consistent with shipping closed: this document does **not** change `_dof`
behaviour as a drive-by, because either removing the subtraction or making the atoms actually
frozen would silently shift NVT and NPT temperatures for that input class. The two are made
consistent together at C1-4, when the freeze mask becomes real and the rank-based DOF of §10
replaces the formula. Until then C0 adds only a test that records the current behaviour so the C1
change is deliberate and visible.

### 1.3 Scope decisions taken during the shadow build

Two decisions were made while the code was being written and are recorded here because they
narrow what "implemented" means.

**SHAKE by default, with the velocity half behind a keyword.** Positions satisfy the declared bond
lengths after `shake.py`. The velocities are left alone unless the constraint record says
`velocity_constraint: rattle`, in which case `shake.solve_velocities` projects them onto the tangent
space of the same bond lengths after the position solve. The consequences of the default are
accepted, not hidden:

| Matters | With the switch on (`rattle`) | In the default build (`none`) |
| --- | --- | --- |
| bond length | held to the tolerance each step | held to the tolerance each step, unchanged |
| bond velocity | `J v` driven to `tolerance * max|v|` by the same record | not enforced; drift is suppressed only indirectly by the position solve |
| verification | asserts the residual and the solver's failure report, and additionally that the projection reaches its tolerance | asserts the normalized residual and the solver's failure report; the convergence-in-`dt` claim for the total energy was measured on 2026-09-29 and does **not** hold, which is the next table |
| DOF counting | rank is taken from the position manifold, which is what a position constraint defines | rank is taken from the position manifold, which is what a position constraint defines |
| surface hopping with constraints | still refused | still refused, for an additional reason: nothing in the default path keeps a rescaled velocity feasible |

`fix` velocity zeroing is not RATTLE and stays in: a frozen component with a nonzero velocity is a
discontinuity the next restore would paper over.

**Measured, not argued (2026-09-29).** The first row of that table is true and was never in doubt.
What the default actually costs is an energy question, and a position solve cannot reach it, so the
cost was measured instead of argued. `qualification/verification/test_constraint_shake_dynamics_cost.py` integrates a
two-bond water against an analytic harmonic base that pulls each hydrogen 0.2 Angstrom inside its
constrained position, so the solver fights the force at every step. The timestep comes from the
repository's own O-H oscillator (450 kcal/mol/Angstrom^2 over a reduced mass of 0.9476 amu, a
14.096 fs period) at twenty points per period, which is dt = 0.7048 fs, and is then halved twice.
The criteria were written down before the run and are not loosened after it.

| Criterion | Pass form, fixed beforehand | Measured, `none` | Measured, `rattle` | Verdict |
| --- | --- | --- | --- | --- |
| bond length | `max abs(d(t) - d0) / d0 < 1e-6` | 1.0e-12 at dt, dt/2 and dt/4 | 1.0e-12, same numbers | passes on both |
| total energy | `drift(dt/2) < 0.7 * drift(dt)` | 139, 145, 147 times the depth of the driving well | 2.48, 0.97, 0.43 | **fails on `none`**, passes on `rattle` |
| radial velocity | no threshold; `J v` reported at every dt | 1.48e-2, 1.53e-2, 1.53e-2 Bohr per atomic time | 8.7e-5, 4.2e-5, 2.1e-5 | does not converge on `none` |

Two controls keep the reading honest. An unconstrained run of the same integrator drifts by 4.4e-3,
1.1e-3 and 2.7e-4 of the same scale, so the leak belongs to the projection and not to the Verlet
loop. And both constrained schemes start from the same velocity, because the initial projection is
the same call the run uses: writing a one-pass cancellation inside the test file instead would have
made the two branches differ before the first step, which is exactly the kind of difference that
makes a comparison mean nothing. (Those numbers are from 2026-09-29 and supersede the first pass of
this measurement, which used a single-pass helper in the test file and read 117, 155 and 187; the
shape of the finding did not move, and the `|J v|` of the default branch is the same order.)

**Implemented, in the branch this document reserved for it.** Criterion 2 failing together with a
`J v` that refuses to shrink is the velocity-side-injection shape, and that combination is the
condition under which the user ruled a velocity projection should arrive as a switch rather than as a
replacement: one keyword on the constraint record, `velocity_constraint: none | rattle`, default
`none`, which adds a step after the position solve and changes no line of it. As of 2026-09-29 that
keyword exists in the lab, on both spellings of the record, and the table above is measured through
`controller.project_velocities` rather than through a stand-in. RATTLE does not override SHAKE and
nothing enables it by default: an unset record returns the caller's own velocity array, object for
object, and the projected coordinates are byte-identical with the switch on or off
(`tests/test_shake_velocity_projection.py`). What the measurement still cannot decide is the cost
inside the real integrator, which needs the MD loop of §7 and therefore belongs to the merge round;
so does the reading of D-15, because a velocity projection that has to be iterated against a solver
the timestep does not own is not a one-line addition.

**A single-position constraint solver, not an integrator.** `shake.solve_positions` iterates the
correction along each bond, shared by inverse mass. It converges the positions and reports the
residual. It is not the time-reversible SHAKE of a velocity-Verlet loop, which needs the previous
geometry and the half-step velocity through discrete impulses; `initial_projection` in the same
module is where a trial-geometry projection belongs. Both names say which they are, because the
distinction decides whether long-run energy behaviour can be claimed.

## 2. Existing Code and Reserved Hooks

All paths are relative to the project root. The following inventory is what a reader of the
current tree can verify.

| Item | Actual state | Location |
| --- | --- | --- |
| `apply_constrain(state)` | implemented: delegates to `constraints/runtime.py::constrain_state`, which restores frozen anchors, solves the bond constraints and removes frozen velocity components (the docstring's promise, now kept) | `pyrai3md/workflow/integrator/operations.py:156-187` |
| `Optimizer.apply_constraint` | empty reserved hook; its TODO lists harmonic restraints, collective variables and metadynamics bias | `pyrai3md/workflow/optimizer/base.py:248-265` |
| `energy_contributions` | read and formatted, never written; `_format_energy` requires a flat name-to-number mapping and the printed header already says `&energy contributions (constraint/multiscale)` | `pyrai3md/utilities/print_results.py` (header constant `SECTION_CONTRIBUTIONS`, read in `_format_energy`) |
| `kabsch` | implemented Kabsch alignment helper, not consumed by any dynamics path; directly reusable by the RMSD history potential | `pyrai3md/data/geometry.py::kabsch` (moved out of `integrator/operations.py`: it reads no state and belongs to no timestep) |
| Geometry derivatives | `BND`, `AGL`, `DHD` with coordinate derivatives exist; angles are in degrees | `pyrai3md/utilities/geometric.py` |
| Cell convention | row-vector cell, `R = S @ B` and `S = R @ inv(B)`; must not be inverted when transcribing the appendix formulas | `pyrai3md/utilities/lattice.py` |
| Constants | single source for unit conversions | `pyrai3md/utilities/units.py` |
| Bond length lookup | `bond_lib(a1, a2)` reads `schema.reference.BOND_LENGTHS` and returns `1.2` Angstrom for an unknown pair; builds no adjacency by itself | `pyrai3md/utilities/bonds.py::bond_lib`, `pyrai3md/schema/reference.py::BOND_LENGTHS` |
| Bond perception | `perceive_bonds(atoms, coordinates, lattice_vector, pbc, scale)` returns `(i, j, delta)` bond triples with the image step each bond needs; requires a lattice, all-pairs cost, capped at 4096 atoms; its consumers are image seeding, `bond_graph` and unfolding | `pyrai3md/utilities/bonds.py::perceive_bonds`, `pyrai3md/schema/normalize.py::seed_initial_image` |
| Radii table | `VDW_RADII` is the van der Waals set with `2.00` for unknown entries; `COVALENT_RADII` is the Cordero single-bond set (H-Cm) and is consumed only by the geodesic metric's Morse rest lengths. Neither is a bond-perception criterion: perception still uses `bond_lib` | `pyrai3md/schema/reference.py:135-160`, `:170-191` |
| Constraint and external-potential config fields | nineteen flat fields on `System`, in two commented blocks: eleven constraint keys (`freeze`, `constrain`, `cbond`, `cangle`, `cdihedral`, `tbond`, `tangle`, `tdihedral`, `fbond`, `fangle`, `fdihedral`, `:353-361`) and eight external-potential keys (`shape`, `factor`, `scale`, `cavity`, `center`, `center_type`, `groups`, `compress`, `:363-371`). Force-constant defaults are `fbond=10.0`, `fangle=0.005`, `fdihedral=1e-6` | `pyrai3md/schema/config.py:352-371` |
| `System.image` | runtime ledger of crossed cell lengths per atom, written by the coordinate wrap in the same `update_system` call as the wrapped coordinates | `pyrai3md/schema/config.py:388-394` |
| Skill-owned integration channels | `fin_coord`, `fin_velocity` and `fin_cell` are consumed in the main thread and popped before results are stored | `pyrai3md/workflow/calculator.py:961-981`, `:1371`, `:1431`, `:1495` |
| Driver routing | `control.driver != 'main'` resolves every jobtype to a single shot, so the skill iterates | `pyrai3md/workflow/__init__.py:115-139` |
| Deleted placeholders | `data/structure`, `data/dynamics`, `data/cv.py` no longer exist; their field design survives only in documents | `pyrai3md/data/`, `dev/DATA_CORE_DEVELOPMENT.md:293-297`, `:318`, `:332`, `:643-647` |
| `kernel/` | contains `cache.py`, `environment.py`, `exceptions.py`, `logger.py`, `maestro.py`; no constraint or CV code, and it stays a coordination layer | `pyrai3md/kernel/` |

`ARCHITECTURE.md` remains the only maintainer of the module tree, so this document describes
module boundaries in prose and links to it rather than duplicating a tree.

## 3. Confirmed Gaps

### 3.1 `freeze` is bookkeeping only

`System.freeze` reaches exactly one consumer: the DOF counter
(`pyrai3md/workflow/integrator/ensemble.py:223-226`). Nothing in the package zeroes the
velocity, the coordinate or the gradient of a frozen atom. `velocity_correction` removes center
of mass translation for all atoms (`operations.py:319`, `:359-365`) and its docstring at
`:331-333` states that frozen velocities "are re-zeroed by `apply_constrain` (the second-phase
constraint engine), not here". That engine does not exist.

(Corrected 2026-10-01: the last sentence no longer holds. `apply_constrain` now delegates to
`constraints/runtime.py::constrain_state`, whose `fix` actions return frozen components to their
anchor (`controller.py:799-807`) and drop their velocity components, so the hold-out that paragraph
found is where the re-zeroing happens. What this section did its work on — treating a DOF count that
subtracts frozen atoms as a wrong-temperature bug rather than a missing feature, and the ownership
question in the second bullet — is unchanged, and is why the paragraph above is left in place.)

Consequences for the design:

- A DOF count that subtracts frozen atoms while the integrator still moves them is currently a
  wrong-temperature bug, not a missing feature. Enabling the freeze mask in C1 is a behaviour
  change and needs its own regression test, not just a new test.
- Each skill schema also declares `freeze: List[int] = []` (xtb, lammps, openqp, molcas, pyscf,
  bagel, cp2k, orca). A framework field with the same name does not mean the skill honours it;
  ownership has to be stated per path, see §3.7.

### 3.2 The text parser cannot express nested blocks, and `&end` opens a section

`_parse_text` (`pyrai3md/interface/parser.py:77-137`) splits a `&name` line into a new section
(`:114-119`), takes token[0] as the key and joins the rest of the line into one raw string
(`:127-133`), skips lines with fewer than two tokens (`:124-125`), and lets a repeated key
overwrite the earlier value (`:135`). There is no block stack.

The important detail: a literal `&end` line is not a closer. It is parsed as the start of a
section named `end`, so every following key is silently filed under that phantom section. The
appendix §4.2 sample, pasted into a `.input` file today, would lose most of its content without
any error.

There is also no per-section hook. `parse_input` (`:24-74`) dispatches by file extension to
`_parse_json` (`:140`), `_parse_yaml` (`:164`) or `_parse_text` (`:63`), and no existing section
(`adaptive`, `theory`, `regions`, `partition`) parses sub-blocks; the nesting that exists in the
project today is produced downstream in the adapter from `;` separators on a single line. So the
DSL work is a change to the core tokenizer loop, and the plan must treat that as the risk it is
(see the milestone gates in §14 and the mitigation in §4.2).

### 3.3 A nested section has four registration points

`NESTED_FRAMEWORK_SECTIONS` is defined in `pyrai3md/schema/config.py:931-939`, not in the adapter,
which only imports it (`pyrai3md/schema/adapter.py:25-27`). Membership is what keeps a section
out of the flat namespace: `_flatten_sections` keeps it nested and type-converts it in place
(`:805-866`, nested branch `:841-857`), and `_partition_fields` merges it into its submodel
(`:642-645`). A new nested section additionally needs a type map and a `SECTION_TYPES` entry
(`:154-163`), an index-conversion slot in `INDEX_START` (`:83-95`, applied by `ReadIndex` at
`:1046-1055`), and default registration in the flat `DEFAULT` keyword map
(`pyrai3md/schema/config.py:29-188`). Missing any one of the four produces a silently dropped
field rather than an error, which is the failure mode already recorded for skill config fields in
this project.

`System` currently mounts no nested submodel at all; every constraint-ish field is a scalar or a
list. `PyRAIConfig` (`:942-1013`) accepts flat kwargs through `FIELD_TO_SUBMODEL` (`:972-997`) and
exposes flat attribute access through `__getattr__` (`:999-1012`). Registering
`System.constraints` therefore makes the constraint model the first nested submodel on `System`,
and its acceptance criteria must include a full regression over every existing section.

### 3.4 The evaluation commit point sits inside the calculator

`run_in_process` (`pyrai3md/workflow/calculator.py:1089`) freezes the geometry and cell snapshot
(`:1133-1144`), builds and executes work items (`:1196-1215`), validates and routes outcomes
(`:1217-1221`), assembles results (`:1255`), and in the multi-center branch calls
`stage_evaluation` (`:1264`) immediately followed by `commit_step()` (`:1274`). The comment at
`:1257-1260` records the intent to move that commit into the MD loop, which the appendix already
flagged as not being current fact.

This is a hard dependency, not a style question. Metadynamics deposition, schedule progression and
history versioning are defined over *accepted* steps. While the calculator commits, a step that a
workflow later rejects has already advanced the constraint history. C4 must not start before that
move lands. The mitigation that keeps an early mistake detectable is in §6: every evaluation
carries an `evaluation_id` and a deposition token, so double deposition is observable instead of
silent.

### 3.5 `set_result` normalizes only top-level arrays

`PyRAIState.set_result` (`pyrai3md/data/core.py:1118`) converts values with
`value.tolist() if isinstance(value, np.ndarray)` (`:1138`), so nested arrays inside a dict of
dicts are stored untouched. `to_dict` (`:1211`) deep-copies history and electronic blocks and lets
nested ndarrays survive (`:1235-1247`). That is safe for the `.rt` file, which is a pickle
(`pyrai3md/data/runtime/serializer.py:421-428`), and unsafe for the cross-process paths, which is
exactly where per-item components would be transported.

So the boundary to fix is one function, and the design must say which of the two chains it
protects: pickle for local restart, payload for subprocess and Slurm dispatch.

### 3.6 Restart and remote dispatch bypass normalization

`serializer.py:14-24` states that restart and remote dispatch bypass the Interface, Schema and
Planning pipeline entirely. `_check_version` (`:303-335`) raises on any semantic version mismatch,
`_check_plan_fingerprint` (`:338-377`) refuses a stale plan, and legacy migration is done by
`_restore_task` (`:208`) and `_restore_center_model` (`:238`).

For constraints this means the bound state, not the input text, is the thing that must be
persisted: bound targets, freeze anchors, reference centers, image state, hill list, history
version and absolute step counter. Rebuilding them from the current structure on restore would
re-capture an `initial` target from a drifted geometry.

### 3.7 The integrator owns the constraint, and that rule is already enforced on the skill side

A decision taken on 2026-09-26 and recorded in the LAMMPS skill says: constraints can only be
applied by the party that moves the atoms. A skill driven step by step as a single-point gradient
source therefore must not constrain the geometry itself: nothing in that input propagates, so the
fix would either do nothing or constrain a configuration the framework then moves on its own
account. The framework now does have an engine to take the constraint in (`apply_constrain` at
`pyrai3md/workflow/integrator/operations.py:156` delegating to
`pyrai3md/workflow/constraints/runtime.py:188`), which is why the refusal below points at the
`constraint` block rather than claiming the request has nowhere to go. Until 2026-10-01 this section
asserted the opposite, and the assertion had propagated into the skill's error message and twelve
shipped examples.

This is implemented, not aspirational: `lammps_shake` defaults to `0`
(`pyrai3md/skills/lammps_skill/schema/reference.py:103`), and
`pyrai3md/skills/lammps_skill/schema/validate.py:222-271` rejects `lammps_shake=1` for any profile
outside the engine-integrated set with a message naming the inconsistency and the place the
declaration belongs.

The framework side must mirror the same predicate rather than invent a second protocol. The mirror
is **not in place yet**: no module under `pyrai3md/schema/` or `pyrai3md/workflow/constraints/`
reads a skill's shake switch or profile. The three signals below are what the mirror is specified
against, which is not the same as what runs today (open item, 2026-10-01):

| Signal | Meaning | Constraint behaviour |
| --- | --- | --- |
| `control.driver != 'main'` | every jobtype is a single shot; the skill iterates (`workflow/__init__.py:115-139`) | reject framework constraints |
| skill returns `fin_coord`, `fin_velocity` or `fin_cell` | the skill propagated the state (`calculator.py:961-981`) | reject framework hard constraints; additive static potentials only if the evaluation geometry is the reported one |
| `driver == 'main'` and no ownership channel used | the framework Verlet moves the atoms | framework applies projection and potentials |

Silent partial application is the failure this section exists to prevent. A wall potential added by
the framework on a geometry the engine moved internally pairs a bias at `R_engine` with an
electronic energy at `R_framework`.

Naming follows the same boundary: the framework block is `constraint` (since #271; the retired
`system_constraint` spelling still routes to the same field so archived inputs run, but no new input
writes it), and its inner words stay inside that block, so they never collide in the flattened
namespace with `lammps_shake`, `shake_fix_id` or `shake_keywords`
(`pyrai3md/skills/lammps_skill/schema/config.py:271-273`). Declaring framework bond constraints
together with `lammps_shake=1` on an engine-integrated profile is the combination that mirror
covers, and as of 2026-10-01 no validation refuses it: the framework gate is the missing half
described above, not an existing check this sentence can cite.

### 3.8 There is one bond perception routine, and it is not a topology service

After `data/structure` was deleted in the PBC stage A+B refactor the framework kept no topology
object and `schema/config.py` has no bonds field, so `bond_type X-H` expansion, per-molecule groups
and molecular unwrapping all lack a bond graph. What does exist, since the image work of #239 and
#240, is a single distance-based perception routine:

| Property | Actual state | Location |
| --- | --- | --- |
| `perceive_bonds(atoms, coordinates, lattice_vector, pbc, scale)` | returns `(i, j, delta)` triples where `delta` is the image step the winning nearest shift needs | `pyrai3md/utilities/bonds.py::perceive_bonds` |
| Criterion | `bond_lib(a, b) * BOND_SCALE` with `BOND_SCALE = 1.2`; `bond_lib` reads `schema/reference.py::BOND_LENGTHS` and returns 1.2 Angstrom for an unlisted pair, so an unknown pair gets a 1.44 Angstrom cutoff rather than an error | `pyrai3md/utilities/bonds.py::BOND_SCALE`, `pyrai3md/utilities/bonds.py::bond_lib`, `pyrai3md/schema/reference.py::BOND_LENGTHS` |
| Non-periodic behaviour | since #267 a structure with no cell is perceived too, with every image step zero, so a bond table has one shape either way; `derive_initial_image` keeps its own early return, so seeding is untouched | `pyrai3md/utilities/bonds.py::perceive_bonds` |
| Cost and guard | all-pairs, `O(N^2)`, refused above `PERCEPTION_MAX_ATOMS = 4096` | `pyrai3md/utilities/bonds.py::PERCEPTION_MAX_ATOMS` |
| Consumers | `schema/normalize.py::seed_initial_image`, the image tests, `constraints/images.py` (unfold for a periodic constraint), and `bond_graph` / `bond_pairs_by_type` behind a `bond_type` selector (`schema/normalize.py::_Normalizer.bond_type_pairs`) | `pyrai3md/utilities/bonds.py::bond_graph`, `pyrai3md/utilities/bonds.py::bond_pairs_by_type` |

Two design consequences, one of them a revision of the draft:

1. No second radius table. The draft's plan to add `utilities/covalent_radii.py` is dropped. The
   covalent table later added at `schema/reference.py::COVALENT_RADII` does not reopen this: it is
   the geodesic metric's rest-length data, and `bond_lib` remains the only perception criterion. The
   project has already rejected a competing element-radius graph in the framework on the ground
   that it fights the force field's own bond table at boundary cases, and that reasoning stands,
   but the stronger point now is that a perception routine already exists in-repo with a documented
   criterion. If `bond_type` expansion is ever implemented it must extend and reuse
   `perceive_bonds` (including a non-periodic branch and an explicit unknown-pair failure instead of
   the 1.2 Angstrom fallback), and per-molecule groups must use the same result, so that one
   geometry produces one bond list for images, constraints and regions alike. **Done in #267:**
   `bond_graph()` is that single result, and the unknown-pair failure is placed in the topology
   service rather than in `perceive_bonds` itself, because a missed bond when seeding image flags
   costs a zero count while a missed bond when selecting a constraint decides physics. The two
   risks are not the same, so they are not given the same rule.
2. The first version therefore took explicit atoms, explicit `bond i j` pairs and named groups
   whose members are listed explicitly, and deferred `bond_type` to §3.8 item 1. That deferral
   is **closed for `bond_type` in #267**: a constraint record may now say `bond_type: O-H` and the
   pairs come from the plan's own graph, which is perceived from the structure the plan was built
   with (`build_plan(..., coordinates=, lattice=)`). Two refusals remain, both narrow and both
   named: a bond type cannot feed a selector that needs *atoms* (a group, a fix, a wall) because
   it resolves to pairs, and it cannot feed an angle or torsion constraint for the same reason; and
   `molecule` grouping still waits, because what is missing is the **spelling** of a molecule (a
   composition? the atom that starts a component?), not the connected components, which
   `bond_graph()['components']` already returns.

### 3.9 The dihedral gradient in `utilities/geometric.py` is wrong on three of four atoms

Found while writing the CV layer, and re-measured on 2026-09-29 because the first version of this
section described it wrongly. `utilities/geometric.py::DHD` returns a value whose convention differs
from the constraint CV's by exactly 180 degrees (its zero is at trans, not at cis), and its
analytic gradient is not the gradient of that value. Compared against a central difference **of its
own return value**, so no unit convention and no second implementation can enter the comparison:

| Row | Status on generic geometry | Measured |
| --- | --- | --- |
| `a` | correct | agrees with the central difference |
| `b` | wrong in sign and magnitude | no constant factor relates it to the derivative |
| `c` | wrong in sign and magnitude | likewise |
| `d` | exactly negated | `g[d] == -dq/dR[d]` |

That pattern held on all 400 geometries sampled before the merge (the characterization file was deleted when `DHD` was fixed, as D-01 required; the surviving pins are in `qualification/verification/test_geometric.py`).
The same file shows why the earlier wording said "the two inner atoms": on the one right-angled
geometry that section first quoted, `[[1,0,0],[0,1,0],[0,1,1],[1,2,1]]`, rows `a` and `b` are
correct and rows `c` and `d` are negated, and generalizing from that symmetric case named the wrong
pair. It also shows where the "57.3" came from: comparing `DHD`'s degrees-per-Angstrom rows with a
radian-based implementation multiplies every row by 57.2958, which looks like a single unit slip in
the wrong place. After that factor is applied, `a` agrees, `d` is exactly negated, and `b` and `c`
match neither, so the residue is a wrong expression and not a conversion.

Two structural statements come with the comparison, and they are the ones a fix has to satisfy
rather than the row-by-row list: the four rows are not the gradient of a rotation-invariant function
(the Noether residual `sum_i r_i x g_i` is of order one against its own scale, where the constraint
CV's own Jacobian gives machine precision), and they *are* translation invariant (`sum_i g_i`
vanishes), which is exactly why a partial check passes and the defect survives.

Three degenerate inputs make it worse rather than louder, and they are register row D-02: at a
strict trans or cis dihedral the `arccos` argument leaves the domain and the derivative divides by
`sqrt(1 - cos^2)`, so **the value and the gradient both come back nan** on the most common real
conformations; three collinear atoms give a nan gradient the same way; `BND` on a zero-length bond
and `AGL` on a collinear angle return nan rows as well.

The consequence is bounded: no caller exists at all today. `pyrai3md/utilities/__init__.py:38`
re-exports the three names and nothing in the package calls them, so this is not a live numeric
bug in MD. The consequence for this document is that any torsion bias, CV or restraint built on it
would inherit a wrong force direction, which is why the constraint subsystem implements its own
torsion, now the *shared* implementation in `utilities/geometric.py` with `cv.py` delegating to it, pinned to the textbook cis, trans and gauche values.
The defect belongs to whoever picks up the existing module, and it is recorded here so the next
reader does not "reuse" it by accident - and so that the merge round does not fix two signs when
three rows are wrong.

## 4. Input Model and the Flat Text Format

### 4.1 One semantic model

The canonical configuration entry is `config.system.constraints`, a `SystemConstraintConfig`
submodel; `&constraint` in text is a spelling of it, and JSON and YAML express it as
`system: {constraints: {...}}`. No new ambiguous flat top-level keywords are added. The structure
(`version`, `units`, `groups[]`, `cvs[]`, `constraints[]`, `biases[]`, `walls[]`, `fixes[]`,
`metadynamics[]`, and since #271 `notes[]`) and the rules for stable ids, list order, strict
unknown-field rejection and inline CV promotion are defined in appendix §4.1 and adopted unchanged.

Every input form must reach one model: text, JSON, YAML and dotted-key JSON all converge on
`SystemConstraintConfig`, and the round trip runs through the certification converter
(`pyrai3md/qualification/certification/converter.py`), which since #271 renders the block with
`qualification/certification/constraint_writer.write_constraint_section` and keeps the grammar in one file. The near-duplicate
legacy copy at `pyrai3md/qualification/converter.py` was not extended, and #301 removed it: nothing
in the package imported it, and the duplicate `_EXECUTION_KEYS` it carried made the single source of
that vocabulary ambiguous.

### 4.2 Why the block DSL was dropped, and what replaced it

The first version shipped a nested block grammar (`&system_constraint` with `wall sphere` /
indented attributes / `end`). It bought the section a syntax the framework uses nowhere else, and
it cost two things: `_parse_text` needed a special-case dispatch to a second parser (which is why
`interface/constraint_parser.py` existed at all), and a declaration took fifteen lines to say what
one `&md step` line says. Nesting is free in YAML and JSON, where the structured spelling already
lives; in a text file it is only cost.

The replacement is the flat form the rest of the input already uses, with exactly one addition.
`&` opens a section, a section runs to the next `&` line, and `&end` closes it - the same rule
`&molecule` obeys, which is why `interface/parser.py` now holds no constraint code whatsoever: it
splits lines into `keyword value` pairs and the schema adapter gives them meaning, again exactly as
it does for `&molecule`.

The one addition is `;`:

- `;` separates **segments**. On an object line a segment is one object (`bias_bond 1 2; 3 4`
  declares two biases); on a parameter line a segment is the value for one object
  (`bias_bond_ref 1.50; 1.42`).
- `,` groups the fields of one segment and a space separates the tokens inside a field
  (`meta_cv r13, r34` is one history over two variables; `wall_center 0 0 0` is one vector).
- Atom numbers are 1-based and a range is written `5-8`. A full-width `；` is refused with the
  replacement named, because it reads as whitespace and would silently merge two objects.
- `#` or `!` at the start of a line is a comment, and an inline `#` ends the line, as everywhere.

**The three rules that keep a long declaration readable.** An object-count anchor is fixed per
family by the object lines it writes, **in the order those lines appear in the file** (for walls,
in the order `wall_type` names its geometries). A parameter line may carry fewer segments than the
family has objects: one segment applies to every object, and a shorter list pads with its last
value - both written into `notes`.
More segments than objects is a refusal, because somewhere the objects were miscounted and dropping
the extras drops a declaration. `wall_type` decides which wall keywords are read at all, and a
keyword it does not read is reported in `notes` rather than dropped in silence. `notes` reaches the
run record: `&constraint plan` in `<title>.log` prints the minted ids, what each action holds, and
every note beside them.

One limitation comes with the flat shape and is stated rather than worked around: a keyword is
given **once** per section, and a file that repeats it is last-wins, exactly as `&molecule` behaves.
That is the interface layer's rule for every section, and this section does not get an exception for
it; the multiple objects a block grammar would have spelled as repeated records are the `;`
segments of one line here, which is also why the grammar has no need of a second parser.

### 4.3 Keyword table

Everything the section reads, and nothing it does not. A keyword is given once - the `;` segments
are how one line carries several objects.

Units, all three optional:

| Keyword | Values | Notes |
| --- | --- | --- |
| `unit_length` | `angstrom` (default), `bohr` | applies to every length-shaped number in the section |
| `unit_angle` | `degree` (default), `radian` | angles, torsions and cell angles |
| `unit_energy` | `hartree` (default), `ev`, `kcal`, `cm` | heights and force constants |

Named groups (`group_name` is the object line):

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `group_name` | one group name | `name` |
| `group_atoms` | 1-based atoms or ranges | `atoms` |
| `group_element` | one or more element symbols | `element` |
| `group_group` | one already-declared group name | `group` |

Collective variables (`cv_bond`, `cv_angle`, `cv_dihedral` are the object lines, concatenated in
that order):

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `cv_bond` / `cv_angle` / `cv_dihedral` | 2 / 3 / 4 atom numbers | `kind`, `atoms` |
| `cv_name` | one name | `name` |

Strict constraints, solved by projection (object lines concatenated in the order
`shake_bond`, `shake_bond_type`, `shake_angle`, `shake_dihedral`):

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `shake_bond` | one atom pair | `kind: bond`, `bonds` |
| `shake_bond_type` | one element combination (`OH`, `CH`) | `kind: bond`, `bond_type` |
| `shake_angle` / `shake_dihedral` | one atom triple / quadruple | `kind`, `bonds` |
| `shake_bond_ref`, `shake_angle_ref`, `shake_dihedral_ref` | the held value | `target` |
| `shake_tol` | solver tolerance, default `1e-8` | `tolerance` |
| `shake_iter` | solver iteration cap, default 200 | `max_iterations` |
| `shake_vel` | `none` (default) or `rattle` | `velocity_constraint` |
| `shake_id` | one name | `id`, else `constraint1..n` |

`shake_tol` is a solver stopping rule, and the qualification record of 2026-10-02 shows what that
costs as a judged bound: the three periodic shake cases that declare `1e-10` land at 9.29e-11
(`constraint_md_periodic_shake`), 8.68e-11 (`..._npt_angle`) and 8.32e-11 (`..._nvt_shake`) relative
drift from the reference frame, i.e. 83-93% of the tolerance they asked for. The pass is real - the
solver held what it promised - but the margin is set by the solver's own threshold, not by a
separately derived bound, so a different build can cross it without anything being wrong. Read a
shake pass as "the solver converged", not as "the geometry is 100x inside spec".

A strict constraint has no potential, so it has no force constant: `shake_fc` and its typed forms
are refused by name, with `bias_*_fc` pointed at. `shake_bond_type` perceives the bond graph, so it
needs the structure (coordinates and, for a cell, the lattice) and names the reason when it is built
without one.

Harmonic biases (one scalar variable each; `bias_bond`, `bias_angle`, `bias_dihedral`, `bias_cv`
are the object lines):

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `bias_bond` / `bias_angle` / `bias_dihedral` | 2 / 3 / 4 atom numbers | `atoms` |
| `bias_cv` | one declared cv name | `cv` |
| `bias_bond_ref`, `bias_angle_ref`, `bias_dihedral_ref`, `bias_cv_ref` | the target value | `target` |
| `bias_bond_fc`, `bias_angle_fc`, `bias_dihedral_fc`, `bias_cv_fc` | force constant, positive | `force_constant` |
| `bias_bond_region`, `bias_angle_region`, `bias_dihedral_region`, `bias_cv_region` | `all` (default), `upper`, `lower` | `region` |
| `bias_id` | one name | `id`, else `bias1..n` |

The parameter keywords are typed on purpose: there is no `bias_ref`, because a value must say which
variable it belongs to, and a name that could only be guessed at is a name that can be guessed
wrong. `inside`/`outside` and a one-sided region on a dihedral remain named refusals (§5.2, §14).

Walls. `wall_type` is read first and decides what the rest of the block means:

| Keyword | Values |
| --- | --- |
| `wall_type` | `none` (default: the block reads no wall), or any of `sphere ellipsoid box triclinic`. Multiple instances of one type are the `;` segments of that type's geometry line. |

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `wall_sphere_radius` | one radius | `geometry: sphere`, `radius` |
| `wall_ellipsoid_semiaxes` | three semiaxes | `geometry: ellipsoid`, `semiaxes` |
| `wall_box_lengths` | three full edge lengths | `geometry: box`, `lengths` |
| `wall_triclinic_cell` | six cell parameters or nine matrix components | `geometry: triclinic`, `cell` |
| `wall_atoms` / `wall_element` / `wall_group` | the confined atoms | `atoms` / `element` / `group` |
| `wall_groups` | atom numbers, **then** the size of a molecule | `molecules`; `1-3 7-9 3` is the two molecules [1,2,3] and [7,8,9], and `;` separates the walls, so one segment belongs to one wall |
| `wall_center` | three numbers | `center` - the point, in the declared length unit |
| `wall_ref` | the anchor atoms | `ref_atoms`; their center becomes the wall's center, captured at the reference geometry |
| `wall_power` | exponent, at least 2 | `power` |
| `wall_fc` | force constant, positive | `force_constant` |
| `wall_target` | `atom` (default), `com`, `cog` | `target` |
| `wall_law` | `hard` (default) or `soft` | `law` |
| `wall_id` | one name | `id`, else `wall1..n` |

Objects are numbered in the order `wall_type` names its geometries, so `wall_type box sphere` with
two box segments and one sphere segment puts the sphere third in every positional parameter line.
A geometry that `wall_type` does not name contributes nothing and is echoed in `notes`.

Three rules about the wall's own keys, each of which is a decision rather than a detail:

- **the selection is implied, not repeated.** A wall with `wall_atoms` confines those atoms; a wall
  that writes only `wall_groups` confines exactly the atoms its groups name (the alternative is to
  type one atom list twice, once to select and once to partition, and let the two disagree); a wall
  that writes neither confines every atom. Each fallback is written into `notes`, and groups that
  do not tile the selection are refused rather than left partly unrestrained.
- **the grouping says what is confined, the target says how it is weighted.** They are independent
  keywords: `wall_target com` with `wall_groups 1-3 4-6 3` is two molecules evaluated at their own
  mass centers, a singleton group is the atom itself (so `com` with a trailing size of 1 and
  `wall_target atom` are the same run), and a wall with no groups is one object over the whole
  selection. A grouping whose molecules hold more than one atom means nothing to an `atom` target,
  so it is dropped with a note.
- **the center has a priority, and none of its readings is the origin.** `wall_center` (coordinates)
  wins over `wall_ref` (the center of named atoms), which wins over the center of the atoms the wall
  acts on; `com` weights that center by mass and every other target by count. All three are captured
  once at the reference geometry and fixed for the run, so a restart cannot move the boundary it
  resumed under. Where a wall declares none of these and one is needed, the run record says
  `center from selection`.

`wall_center`, `wall_ref` and `wall_groups` are also the three keywords that accept the word `none`
as one segment (`wall_ref 1 2; none`): a positional line has to be able to say "this wall declares
nothing" instead of padding it with the previous wall's value, which would move a boundary the input
never placed.

Fixes, which freeze coordinate components (§5.2):

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `fix_atoms` / `fix_element` / `fix_group` | the frozen selection | `atoms` / `element` / `group` |
| `fix_dir` | `x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz` (default) | `direction` |
| `fix_ref` | `initial` only | `reference` |
| `fix_id` | one name | `id`, else `fix1..n` |

History potentials (metadynamics); `meta_cv` and `meta_rmsd` are the object lines:

| Keyword | One segment holds | Fields |
| --- | --- | --- |
| `meta_cv` | one or more cv names | `kind: gaussian`, `cvs` |
| `meta_rmsd` | the aligned selection | `kind: rmsd`, `atoms` |
| `meta_height` | hill height, positive | `height` |
| `meta_width` | one width per cv, or one width for all | `width` |
| `meta_pace` | steps between depositions | `pace` |
| `meta_gamma` | well-tempered ratio, above 1 | `bias_factor` |
| `meta_temp` | temperature in Kelvin | `temperature` |
| `meta_weight` | `uniform` (default), `mass` | `weight`, rmsd only |
| `meta_rmsd_ref` | `initial` (default) or 3 numbers per atom | `reference`, rmsd only |
| `meta_id` | one name | `id`, else `metad1..n` |

`meta_gamma` is a ratio of temperatures, so it needs `meta_temp`; one without the other is refused
at input time rather than deposited at plain height.

### 4.4 Parameter ramps

A ramp is five keywords around the keyword it ramps - `P_start`, `P_end`, `P_scale`, `P_delay`,
`P_stop` - and it is declared on the action, not in a schedule section. The rampable bases are
`bias_bond_fc`, `bias_angle_fc`, `bias_dihedral_fc`, `bias_cv_fc`, `wall_fc`, the four wall geometry
lines and `meta_height`; anything else that asks for one is refused, because a parameter with no
implementation must not be smoothed over by a ramp that looks like it works.

- `start` and `end` are exact values. Where a key is absent the **object's own value** is used, so
  omitting `bias_angle_fc_start` does not inherit the bond ramp: each object falls back to its own
  base, which is what makes a one-segment ramp mean "the same ramp everywhere".
- `scale` is a pair of factors applied to the base and is mutually exclusive with `start`/`end`. A
  scale on a converted parameter (a radius, a semiaxis) is refused rather than guessed, because the
  two endpoints would be multiplied before and after a unit change; `start`/`end` is the
  unambiguous spelling there.
- `delay` defaults to step 1. `stop` is left open by the reader and closed by the schema with the
  run's own `md step`, with the source written into `notes` - the length of the run is not
  something this section states in numbers.
- `delay` and `stop` bound *when* a value is reached, not *which* values exist. A geometry ramp is
  described by its endpoints, so the periodic wall checks (§5.2) are evaluated at every value the
  ramp produces, not at the one the run starts with.
- A geometry ramp may be written per component instead: `wall_x_start`, `wall_y_end`,
  `wall_gamma_stop` and so on, where `x/y/z` are the three semiaxes, the three box edges or the
  three cell lengths and `alpha/beta/gamma` are the cell angles. Components of one vector share one
  schedule, so they share the window and the writer prints the whole endpoint vector back.
- The endpoints are checked where they are declared: both must be positive for a quantity that
  has to stay positive, and a vector endpoint must have the same number of components as the value
  it replaces. The positive list is the runtime's own (`schedule.POSITIVE_PARAMETERS`), and it
  counts a force constant and a hill height among the lengths: the subsystem refuses a zero or
  negative value at *every* step, so a ramp that turns a term on starts above zero (say `1e-8`)
  rather than at zero, and the refusal says so at input time instead of at the first ramped step.

### 4.5 Selection sources

Given §3.8, the text accepts explicit atom lists, element symbols and named groups everywhere a
selection is owed, plus `shake_bond_type` for a strict constraint. `element` is a set union and is
not a chemical molecule. `molecule` has no spelling in this grammar: it is recognized by the schema
and refused at execution until the grouping convention of §3.8 item 1 is settled. Recognizing a
syntax is never reported as supporting the feature.

Index convention: user input is 1-based in all formats, the normalized plan is 0-based, and a typed
plan or a `.rt` file is never decremented again. Units: the defaults above, with the conversion rule
`q_internal = c * q_input`, `k_internal = k_input / c**2` and the wall exponent convention of
appendix §4.3 and §7.2 adopted unchanged. The row-vector cell convention of `utilities/lattice.py`
overrides the column-vector form written in the appendix sample.

### 4.6 One section, read whole

A water dimer whose two OH bonds are held, one stretch biased, both molecules kept inside a
sphere, one hydrogen frozen in `z`, and a history deposited along the donor OH:

```
&constraint
  unit_length angstrom
  unit_energy hartree
  shake_bond 1 2; 1 3; 4 5; 4 6
  shake_bond_ref 0.96
  cv_bond 1 2
  cv_name rOHa
  bias_bond 1 4
  bias_bond_ref 2.80
  bias_bond_fc 0.020
  bias_bond_fc_start 0.002
  bias_bond_fc_stop 2000
  wall_type sphere
  wall_sphere_radius 4.5
  wall_center 0.0 0.0 0.0
  wall_atoms 1-6
  wall_fc 0.002
  fix_atoms 6
  fix_dir z
  meta_cv rOHa
  meta_height 0.005
  meta_width 0.15
  meta_pace 100
```

Two objects of one family, one ramp that starts at a stated value and ends at the value the line
declares, and one geometry `wall_type` does read. The same declaration as YAML is the `system:
constraints:` records of §4.1, and the two reach the same plan fingerprint - which is the property
`test_constraint_parser.py::TestWriter` pins, in both directions.

## 5. Normalization and Reference Binding

Two phases, and the second one does not re-interpret the DSL, exactly as in appendix §5:

```
parse -> typed schema -> static normalize -> ConstraintPlan
                                  |
                    trajectory initial geometry / masses / image
                                  |
                          bind_reference -> ConstraintState
```

The hook point is concrete. `normalize()` (`pyrai3md/schema/normalize.py:512-562`) runs
`model_dump` (`:534`), `_flatten_config_dump` (`:535`), `normalize_jobtype` (`:536`),
`compute_derived_quantities` (`:537`, the step that produces masses at `:113-126`),
`seed_initial_image` (`:536`), `derive_current_state` (`:537`), `normalize_history` (`:538`),
`normalize_skill_name` (`:541`) and then `_partition_to_submodels` (`:542`) before
`Config.model_validate` (`:557`), with a nested restore loop at `:543-556`. A new
`normalize_constraints()` is inserted between `:537` and `:542`, so it sees geometry, masses and
cell already derived, and its output must be added to the nested restore loop so the content
survives the round trip. `normalize_history` (`:372`) is the closest existing precedent for the
shape of such a step.

Reference binding is per trajectory and happens after the geometry is known, before the first
step: sampling-derived trajectories that start from a different geometry must bind their own
targets and anchors instead of inheriting the parent structure, and restart skips this phase
entirely (§3.6). `_convert_lattice` (`:263-333`) and `normalize_pbc` remain the only producers of
the runtime cell; the constraint module reads it and never writes it.

Validation follows the two patterns already in `schema/validate.py`: capability refusal copies
`_reject_unsupported_pbc_backends` (`:398-450`, reading `meta.get('supports_pbc') is False` at
`:446`), and combination refusal copies the C>1 refusals in `validate_centers` (`:513`, `:624-639`)
and the ensemble refusals in `validate_ensemble` (`:829`, `:872-918`). New checks are called from
`validate()` (`:1027-1077`) and must fire during planning, before any backend is constructed. The
set to reject in C0 and C1 is: a `wall_law soft` wall in a cell (§4.3, §22), the archive's flat
constraint keys written at all (§4.2), `driver != 'main'` for a framework-constraint run, and any
unimplemented capability name. A collective variable is **not** in that list: the question is asked
per variable at binding time, where the bond graph exists (§5.2).

#### What a `fix` freezes, and what a one-sided `region` means

Stated here because both are routinely mis-read from the keyword alone, and both
are asserted by tests rather than by this prose:

- A **fix** constrains *coordinate components*, not atoms and not distances.
  `direction` is a per-component mask over `x y z / xy / xz / yz / xyz`, the anchor
  is the initial geometry (`reference` accepts only `initial`), and the effect is
  three separate ones: the projected positions return to the anchor every step, the
  frozen components of the velocity are zeroed, and the frozen components are
  removed in `gradient_projected` - which is what an optimizer or a diagnostic
  reads, while a Verlet kick uses the unmasked `gradient_total` and is pulled back
  by the projection in the same step. A fix carries no potential: it contributes
  nothing to `energy_correction` and nothing to `hessian_block` (its ids list
  stays empty), and its degrees-of-freedom effect enters through the mobile
  component count, not through a condition row.
- A **one-sided region** is a clamp on the biased difference, not a wall. With
  `V = 1/2 k max(delta, 0)^2`, `upper` acts only where the variable is above its
  target and `lower` only where it is below; inside the flat side the energy, the
  gradient and the analytic curvature are all exactly zero (a curvature reported
  there was fixed in #270: the clamp zeroes `delta` but leaves
  `d(raw delta)/d(value) = 1`, so the activity has to be carried separately from
  `chain`). At `delta = 0` with `region: all` the force vanishes and the curvature
  does not - that is the harmonic rest point, and it is pinned as its own case.
  Exactly on the kink of a one-sided region nothing is pinned: the sign there is
  decided by the last bits of the distance evaluation, and an assertion would
  freeze a rounding artifact rather than a fact. `inside` and `outside`, and any
  one-sided region on a dihedral (whose periodic difference needs a declared
  origin), stay refused by name until their conventions exist.

**Section 5.2, as of the periodic round (2026-09-30, CHANGELOG #266).** The PBC clause of that list
is no longer "any constraint on a periodic system"; it is per capability, and both halves are
implemented rather than planned:

- a **strict constraint** is measured on the geometry unfolded along its bond graph. Perception is
  `utilities/bonds.perceive_bonds` (the framework's only such routine, so no second radius
  table is introduced - section 3.8 stands), the unfolding is
  `utilities/geometric.expand_components` keyed on the *connected component* rather than on one
  constraint, and the write-back is `constraints/images.fold_back`, which emits the wrapped
  coordinates together with the crossing counts they imply. A plan that constrains an angle or a
  torsion goes to `shake.solve_conditions`; a bond-only plan still goes to `solve_positions`.
- a **wall** and a **fix** need no bond graph at all, because neither measures a separation between
  two stored atoms. A periodic wall is instead gated by four geometric checks
  (`wall.periodic_check`): the surface strictly inside the cell, a shell between surface and face
  that a single step cannot jump, every member starting inside, and a center that is either declared
  or taken from the unfolded group. A `wall_law soft` wall is refused by name before those four
  run, because it has no surface to place inside anything. A wall whose surface reaches a face is
  refused, because
  periodicity then removes the barrier it is supposed to be.
- a **ramped wall** is checked twice over, and the two halves answer different
  questions. That the wall fits the cell and that its shell is thick enough are asked of
  **every value the ramp reaches** (`schedule.endpoint_variants`), because `delay` and `stop`
  bound when a value is reached, not which values exist. That every member starts inside is asked
  of **the declared geometry alone**: an atom outside the *closed* radius of a wall that is
  closing on it is what closing means, and checking the end of the ramp there refused the
  compression case outright (`binds atom 29 while it sits 0.0031 Bohr outside`, measured before
  the split).
- a **collective variable** is gated per variable, not per kind. The first version refused
  `bias` and `metadynamics` outright in a cell, and `check_periodic` refused their blocks at
  input time; both were throwing away the served half to avoid the unservable one. Whether a cv
  has a periodic image is a bond-graph fact, so the question is asked where the graph is known:
  `_validate_images` runs `require_same_component` on each cv of each action (plus the half-box
  rule for a bond cv with a declared target) and measures the connected ones on the unfolded
  geometry. A distance between two waters is still refused, by name: `bias 'pull_two_waters' cv
  'oo' names atoms [1, 4], which lie in 2 separate connected components`.

- a **wall in a cell** is confined by four geometric checks (it fits, the shell to the face is
  not zero, the shell is thicker than one step can carry, its declared center is not a cell face),
  and the fourth question - is every member inside at the start - is asked only while the run has
  not moved. A one-sided penalty surface is crossed by thermal motion and pushes the atom back, so
  re-asking it mid-run aborts a wall for confining (#275, D-22).

The guard lives in `constraints/state_adapter.require_usable` (`:212`), which reads the expansion
once per evaluation and leaves it on the view; `solve_geometry` (`:435`) is the only way the rest of
the core reaches geometry, so a non-periodic run cannot diverge from its own numbers.

Reference resolution errors (unknown group, cyclic group reference, empty expansion, atom index
out of range, non-positive mass, incompatible anchors) are initialization-time failures that list
the source line or input path, per appendix §5.1.

## 6. Data Contracts

Three persistent records, with the ownership split of appendix §6.1 adjusted to what the code
actually offers:

| Object | Owner | Contents |
| --- | --- | --- |
| `ConstraintPlan` | immutable, derived snapshot of `config.system.constraints`, stored on the state and versioned | explicit atom indices, CVs, canonical parameters, group references, unit and semantic version, fingerprint |
| `ConstraintState` | per `PyRAIState` | bound `initial` targets, freeze anchors, reference center, hill list or reference structures, last committed step, history version, image state |
| `ConstraintResult` | one evaluation, committed into results | corrected geometry, per-item energy and gradient, totals, projected quantities, rank and residual diagnostics, geometry and history versions |

`PyRAIState` (`pyrai3md/data/core.py:284-314`) has no `cell` attribute and no constraint slot;
coordinates, velocities and masses are read-only pass-throughs of `pmd.system` (`:343-358`,
`mass_au` at `:358`) and the result property is `result` (`:532-533`). So the plan enters the state as
a new read-only accessor alongside `partition_plan`, and the geometry and cell snapshot the
constraint module consumes is the one `run_in_process` already freezes at `:1128-1138` for the PBC
consistency contract. Reusing that snapshot is deliberate: it is the same object the backends see,
which is what makes the `E(R*)` rule of §1 satisfiable.

The field list, shapes and units of `ConstraintResult` are appendix §6.2 and are adopted with the
naming of the four gradient kinds kept exactly as there: `gradient_total` is the true derivative of
the physical potential, `gradient_projected` is what an optimizer or diagnostic consumes,
`gradient_for_integrator` is what Verlet kicks with, and `constraint_impulse` is an MD Lagrange
quantity that must never be added to a potential gradient. A base value that was never computed
stays `None`; zero arrays are not manufactured to look like results.

The nested components produced by the per-item breakdown cross the `set_result` boundary described
in §3.5, so C1 defines the recursive normalization there and keeps the flat
`constraint.<kind>.<id>` keys for `energy_contributions`.

Deposition idempotency is designed in from C0 rather than bolted on at C4: `ConstraintState` carries
`evaluation_id` and the deposition token `(trajectory_id, accepted_step, item_id)`, and the commit
API returns a new state instead of mutating. This makes a premature commit detectable while the
commit point is still wrong (§3.4).

## 7. Runtime Integration Points

The actual MD loop (`pyrai3md/workflow/md.py:496-591`) is:

```
497 apply_ensemble(pre)
503 verlet_i
505 apply_pbc
517 calculator.run(target_states=...)
526 merge_deviations
531 verlet_ii
533 phase_correction
535 apply_constrain
    velocity_correction
    apply_ensemble(post)
    apply_constrain (constrain_after_cell)
    state.kinetic_energy()   <- sampled once, on the step-final velocities
    update_history
568 surfacehop
570 verification
572 calculator.sync()
579 checkpoint
591 advance_step
```

(The line numbers move every time the loop gains a profiled step, as it did for `target_states`;
they are given once and every symbol below is cited by name, so a drift here is a cosmetic defect
and not a wrong design claim.)

The appendix proposed adding bias gradients through this loop. That is not possible at `:535`,
because `verlet_i` performs the first kick from the previous frame's
`snapshot['nuclear_sample']['active_gradient']` (`pyrai3md/workflow/integrator/verlet.py:121-146`,
kick at `:286`) and `verlet_ii` performs the second kick from `state.active_gradient()`
(`:606`, kick at `:643` and `:664`). A force attached after `:533` has already missed both kicks of the
current step, and adding it to the stored result would double it on the next step.

The correct division of labour:

| Job | Where | Why there |
| --- | --- | --- |
| additive potential `V_add` and its true gradient `G_add` | a new finalizer inside `run_in_process`, after `convert_stress_to_atomic` (`calculator.py:1297`) and before `adapt_skill_result` and `set_result` (`:1301-1303`) | it is the single point where the assembled global result is final, for both C=1 and C>1, and it runs once per evaluation |
| hard-constraint position projection and half-step velocity correction | between `verlet_i` and the calculator, i.e. a new explicit call, not `:486` | the electronic evaluation must see the projected geometry `R*` (§1 rule 1) |
| freeze masking, velocity zeroing, and (a later decision) the RATTLE velocity constraint | `apply_constrain` at `:486` | this is the one position in the loop where a velocity-level constraint belongs; the call site keeps its name and its thinness. **RATTLE is out of scope for the current build (§17)**: positions are constrained, velocities are not projected |
| DOF, free kinetic energy, temperature | `ConstraintStatistics` consumed by `ensemble._dof`, `state.kinetic_energy()` and the writers (§10) | today each of them guesses |

No finalizer or hook registry exists in `Calculator` today, so C1 creates the mechanism, not the
call site only. It must be a single call site: adding the potential inside `assemble_results`
(`:1238`) would cover the multi-center path and miss `C=1`, whose
`active_gradient()` (`data/core.py:700`) has a fallback to the result gradient through
`_active_gradient` (`:73`). Both paths have to yield the same total, and a test asserts it (§15).

Step 0 semantics are preserved: `verlet_i` returns unchanged when no previous frame is recorded
(`verlet.py:256-259`) and `verlet_ii` skips when the previous frame carries no gradient
(`:454-459`), so the first evaluation happens at the initial geometry with the initial velocities.
Reference binding and the initial projection run before that evaluation, and the energy change
caused by the initial projection is reported as a diagnostic, not folded into the first step.

## 8. Public Controller API

The staged interface of appendix §6.3 is adopted as the public surface, with the `context`
requirement stated explicitly. `SystemConstraintController(plan)` is rebuilt from plan and state;
`bind_reference` produces a `ConstraintState`; `project_positions`, `evaluate_potentials`,
`combine`, `project_velocities`, `statistics` and `commit` are the phase entry points, and
`apply_structure` is the one-shot convenience path that projects, optionally calls an evaluator for
the projected geometry, and combines.

Binding rules that the implementation must hold:

- `context` carries evaluation purpose (`single`, `md`, `optimization_trial`, `hessian`), physical
  time, accepted step, timestep, evaluation id, cell, and history version. No function reads a
  module-level step counter.
- `evaluate_potentials` is pure: it deposits nothing. Only an accepted step may `commit`.
- Repeated evaluation of the same geometry, plan, schedule step and history version returns the
  same numbers. Every total is synthesized from the stored base, never by `+=` on a previous total.
- Failure leaves no partial state: no coordinates, no results, no history. A projection that does
  not converge within tolerance reports the largest residual, the offending item id and the
  iteration count, and does not continue integrating, and does not silently fall back to a soft
  spring.
- If `apply_structure` is handed a base result, its geometry id is checked; when the geometry
  changed, the module refuses to combine and asks for re-evaluation.
- `project_velocities(velocities, state, context, masses=, coords=)` zeroes frozen components in
  every case, and projects the bond tangent space only for a record that declares
  `velocity_constraint: rattle`. That projection reads the Jacobian at the geometry the positions
  were projected to, so a record asking for it without `coords` or without `masses` is a refusal,
  not a guess. With no such record the call returns the array the caller handed in, object for
  object.

The controller lives in `workflow/constraints/` as a composition of small components (`bias`,
`wall`, `shake`, `projection`, `schedule`, `metadynamics`), not as one class holding every
algorithm, and numerical helpers stay in `utilities/` with leaf dependencies so region selection,
scanning and analysis can reuse them. `kernel/` stays a coordination layer and holds no numerical
algorithm, per appendix §1 and §3. The first milestone creates only the files it needs.

## 9. Energy, Gradient and Electronic-State Isolation

First version: every added potential is state independent. For `C=1`, `E_total[s] =
E_base[s] + V_add` over the actually computed states, and `G_add` is added only to gradient rows
that exist; a missing state gradient stays missing. Adding the same scalar to every state does not
change gaps. For `C>1` no Cartesian product tensor is manufactured; the active scalar and the
global `(N,3)` gradient carry the addition. Composite contributions (`validate_plan` and
`aggregate_results` in `pyrai3md/workflow/multiscale/composite.py`, and the strict assembler at
`:577` with `_scatter_gradient` from `pyrai3md/workflow/multiscale/regions.py:91-152`) keep their task
coefficients and their link-atom handling untouched; the constraint term is added once, after
assembly, and is not copied into per-center local electronic arrays. Note that
`composite.project` (`:187-210`) is still an explicit placeholder for cap-atom back-mapping, which
is one more reason global terms must not be distributed per center.

`results['energy']` and `results['gradient']` express the physical total; `base_energy` and
`base_gradient` and the original electronic carrier are retained, and
`active_surface` carries base, total and integrator drive in one consistent commit. `C=1`
electronic propagation must keep reading the base electronic view so that GSH gradient semantics
do not change because a corrected result is now available. FSSH, GSH and LZSH couplings, NAC and
SOC are not rewritten by this module; velocity rescaling after a hop has to stay inside the
feasible velocity space, which is why hard constraints combined with surface hopping stay rejected
until that adaptation exists.

Idempotency and caching follow appendix §8.2, with one unit fact added from the audit: gradient
unit conversion for the optimizer path happens in the workflow, not the optimizer.
`Optimize._resolve_skill_units` (`pyrai3md/workflow/optimize.py:240-326`) derives
`gradient_factor` from the skill manifest (Hartree per Angstrom mapping to `BOHR_TO_ANGSTROM` at
`:312-316`) and applies it at `:170-174`, while `optimizer/base.py` converts coordinates only
(`:89-90`, `:151`, `:171`). Any gradient the finalizer injects must therefore already be Hartree
per Bohr, and the ordering constraint on the finalizer (after unit normalization, before
`set_result`) exists as much to guarantee that as to guarantee single application.

## 10. Degrees of Freedom and Thermostats

`_dof` (`pyrai3md/workflow/integrator/ensemble.py:204-226`) computes
`3 * (natom - n_frozen) - N_COM_DOF` with `N_COM_DOF = 3` (`:116`) and clamps the result to at
least one (`:226`); the comments at `:80-81` and `:207` note that the SHAKE count is left for the
second phase. Counting constraints by formula and clamping a physical zero to one are both
wrong once real constraints exist: a fully frozen system has zero degrees of freedom, and static
evaluation is legitimate while temperature control is not, and the temperature is `N/A` rather
than one degree of freedom worth of noise.

The replacement is rank based, using the formula and the connectivity-decomposition guidance of
appendix §9.2: build the mass-weighted linear velocity conditions on the free Cartesian components
after the fix mask, take the rank, and let `ConstraintStatistics` be the single source of
`N_free`, `K_free` and temperature for `_dof`, `PyRAIState.kinetic_energy()`
(`data/core.py:965`), the run log and `MdOutputWriter` (`pyrai3md/workflow/records/output.py:307`).
Bias, wall and metadynamics do not reduce DOF. Removing center-of-mass translation or rotation is
counted only when the pipeline actually performs that operation, which today is
`velocity_correction` (`operations.py:359-365`) for translation only.

Every operation that changes velocities must keep the velocity in the feasible space: thermostat
scaling (`:553-602` NHC, `:668-669` CSVR), COM correction, hop rescaling and initial sampling.
Uniform scalar scaling preserves the tangent space of static homogeneous velocity constraints;
non-uniform operations need an explicit check or re-projection. The sequence "freeze, remove COM,
freeze" must not be followed by thermostat action computed from the old kinetic energy -- and since #310 it cannot be: the loop samples `kinetic_energy()` once, after the post half step and the second projection, so the scalar the thermostat, the log and the frame record share is the one the recorded velocities reproduce. The NPT
path is out of scope for these milestones: the barostat rescales coordinates and cell
(`:735-744`, `:948-953`) and MTTK with constraints is refused upstream
(`validate_ensemble`, `:872-918`), so constrained variable-cell work belongs to C5.

## 11. Optimization, Single Point, Hessian and Learning Data

The per-workflow support table is appendix §10 and is adopted unchanged in content. The
implementation notes that come out of this audit:

- Geometry handling API: with no backend, return projected geometry plus additive energy and
  gradient; with an evaluator, return totals for that same geometry.
- Framework single point: `Calculator.run()` is already a history-free evaluation
  (`pyrai3md/workflow/single.py:77`), so the finalizer of §7 serves it without new plumbing.
- Minima optimization: the loop is evaluate (`optimize.py:165`), move (`optimizer.run(state)`,
  `:188`), sync (`:191`), then check convergence (`:194`), with the break at `:218-219`. Reported
  energy, gradient and geometry must belong to the same evaluated structure, so the accept and
  output order is adjusted at integration time. `Optimizer.apply_constraint` (`base.py:248-265`)
  is the reuse point for displacement projection; zeroing a gradient is not sufficient because BFGS
  couples the components, and the trust-radius scaling at `:329-345` is norm based and cannot act
  as a mask.
- Numerical Hessian (`pyrai3md/workflow/hessian.py:150-152`, `:172-173`) displaces Cartesian
  coordinates and differentiates gradients, so with hard constraints it no longer measures a
  Cartesian Hessian. First version: static potentials only, no hard constraints, fixed schedule and
  history; reduced Hessians are a separate later definition.
- Scan (`pyrai3md/workflow/scan.py:71-78`) and search (`pyrai3md/workflow/search.py:120-121`)
  reuse the stateless evaluation but must bind a reference per candidate or inherit it explicitly,
  instead of re-capturing an `initial` target on every trial.
- Adaptive, prediction and train (`pyrai3md/workflow/adaptive.py:89-113`) keep their labels from
  the base electronic result; a wall or bias energy must not enter a training set.
- MECI, MECP, TS and NEB are not claimed. Capability is added one method at a time.

## 12. Restart, Parallel Dispatch and Output

Persisted set: plan and semantic version, bound targets, anchors and reference center, topology and
atom-order fingerprint, the `System.image` ledger, hills or reference structures, history version,
absolute step and time, last deposition token, and the next-force continuation of §7 in the
appendix. Restore never re-infers bonding or re-captures `initial` targets (§3.6); a run that asks
to continue metadynamics without history fails, and an old constraint-free runtime migrates to an
empty plan. The version block (`serializer.py:70-105`) gains a constraint semantic version, and
migration follows the `_restore_task` (`:208`) pattern.

State copies deep-copy mutable history; trajectories share only the immutable plan, never hills,
step counters or writable NumPy buffers. Batch trial evaluation reads a frozen history and a worker
never deposits. Checkpoints are written after a complete accepted step.

Output: every evaluation reports the corrected geometry, `E_base`, per-item additive energies and
`E_total`; detailed mode adds the four gradient kinds. MD adds DOF, free kinetic energy, constraint
error, schedule work and deposition jump. A biased energy is never printed as an electronic energy.
Two concrete surfaces: the flat `constraint.<kind>.<id>` keys feed the existing
`energy_contributions` reader (`print_results._format_energy`, header constant `SECTION_CONTRIBUTIONS`), which used to print nothing
because no writer exists; and `MdOutputWriter.record` (`output.py:307`, row writer at `:722`) gains
the per-step columns, reading them from the history snapshot the frame builder already produces
(`data/core.py:861-940`) rather than from a second source. Frames store a history version and
per-item summaries, not the full hill list.

## 13. Legacy Migration

The reference implementation is `legacy/PyRAI3MD/Molecule/constraint.py` at the repository root
(464 lines, importing under the name `PyRAI2MD`). The migration table of appendix §12 is adopted,
with the mapping entries re-derived from the actual code so the conversions are not guessed:

| Legacy behaviour (verified) | New design |
| --- | --- |
| `V = f * d^2` with no factor of one half, gradient `2*f*d*g` (`:273-274`, `:302-303`, `:333-334`) | harmonic convention `V = 1/2 k d^2`, so `k = 2f` before unit conversion |
| hardcoded `0.529177249` (`:277`, `:306`, `:337`, `:367`) | single source in `utilities/units.py` |
| empty target captured from the first geometry, stored on the trajectory object (`:255-263`, `:284-292`, `:313-321`, `:344-362`) | explicit `reference initial` / `target initial`, bound once and persisted; restart does not re-capture |
| `_match_list` repeats the last value or truncates silently (`:147-158`) | strict length checking; a compatibility adapter that broadcasts must record the expansion |
| `freeze` zeroes gradients only, `traj.grad[:, frozen, :] = 0` (`:372-374`) | `fix`: coordinates, velocities, displacement mask and DOF, see §3.1 |
| `constrain` selects the atoms a cavity potential acts on, falling back to all atoms (`:42`, `:105-109`) | not a SHAKE list; it must never be migrated by name. A wall given nothing takes every atom, and a wall given `wall_groups` takes exactly the atoms its groups name (§4.3) |
| `cavity` is a full-space polynomial potential with nonzero interior values, `V_i = F_i^(alpha/2)`, `alpha = max(factor, 2)` (`:85`, `:193-232`) | `wall_law soft`, one of the wall's two laws: the same `Q^(p/2)` over the dimensionless axis ratio, and no fifth family. It was a `legacy_polynomial` family first (#264) and retired into the wall once §22.1 measured the two against each other; the energy/gradient inconsistency of D-15 is resolved in favour of the derivative |
| group centroid quantities duplicated per molecule, reduced mass `m_i / group_mass` (`:120`, `:186-187`) | `wall_groups` names the molecules by their atom numbers, with the size last, and `wall_target` decides the weighting; the archived arithmetic is pinned in `test_constraint_legacy_equivalence.py` and the wall evaluates once per group, so the energy no longer carries the atom count |
| `center_type` with first-capture-then-fix center (`:344-362`) | fixed reference center; `record_center` is not a field of the new schema, only `center` and `center_type` exist (`config.py:352-371`) |
| dihedral wrap `if abs(dr) > 180: dr = (360 - abs(dr)) * sign(target)` (`:331-332`) | periodic difference through `atan2`, with the branch documented |
| `exit('...')` on bad input (`:169`, `:425`, `:444`) | typed exceptions and diagnostics; the interpreter stays alive |
| `compress` linear radius schedule from iteration (`:138-145`, `:212-216`) | radius schedule with explicit endpoints |
| `groups` as `[[nmol, natom], ...]` (`:160-186`) | expanded to explicit atom order with total count checked |
| `GeomTracker` stop criterion (`:379-464`) | geometry monitoring built on the shared CV code; not a potential, not a hard constraint |

Legacy and new-style definitions coexisting is refused in the first version, with a message asking
for migration, because otherwise a bias is applied twice. Non-empty selection or action definitions
alone activate the legacy path; the `System` defaults `fbond`, `shape` and `scale` are not evidence
of user intent. Line-for-line porting is not the goal: the point of the comparison is to decide
which behaviours are preserved and which numerical problems are fixed. Since the current runtime
entry point is a no-op, numeric compatibility against the legacy file needs its own golden cases
rather than an assumption that existing trajectories already carry these constraints.

Attribution: the legacy file is `PyRAI2MD` module code authored by Jingbai Li (header dated
Sep 27 2022). It is consulted as a behaviour reference and its logic is not copied into new files;
where a formula is transcribed the source is named in the file header, per the project convention
of keeping authorship information with borrowed code.

> **Path note (2026-10-01, #271).** The file names in §14, §15, §17, §18 and §19 are the plan as it was
> executed on 2026-09-29, kept as the record of that move. Where they name `schema/constraint.py`,
> `schema/legacy_adapter.py` or `interface/constraint_parser.py`, those modules no longer exist:
> their content lives in `schema/config.py`, `schema/normalize.py`, `schema/validate.py` and
> `schema/adapter.py` respectively, and the classic `&constraint` text is flat and is read, in both
> directions, in the adapter (§4 and §21).

> **Retirement note (2026-10-01, #273).** Where §13, §14, §18, §19 and §21 name the cavity as a live
> family (`legacy_polynomial`, `workflow/constraints/legacy_cavity.py`, a `cavities` block, the
> `test_constraint_legacy_cavity.py` cases), that is the record of how the archived potential was
> first reproduced. It is retired now: the same law is `wall_law soft` (§4.3, §22.1), the schema
> family and the module are gone, and the archived arithmetic is transcribed into
> `test_constraint_legacy_equivalence.py` instead of imported. The D-11 and D-15 rows keep their
> original wording because they record the decision that made the retirement possible.


The staging of appendix §13 is kept, re-scoped by §3.2, §3.7 and §3.8 and by the decision that the
text DSL ships with the first contract milestone.

| Phase | Deliverables | Gate to enter the next phase |
| --- | --- | --- |
| C0: contracts, input (closed by default, DC-1 to DC-5 apply) | explicit selection, CV math, data records, `SystemConstraintConfig` nested on `System`, `&system_constraint` block parser, normalize and validate, certification round trip, empty plan, legacy adapter spec | text, JSON and YAML produce one identical plan; repeated blocks are not lost; an unclosed block or a stray `end` fails with a line number; indices convert exactly once; unsupported capability and unsupported combination fail loudly; the existing section flatten regression is unchanged |
| C1: static soft potentials in use (still DC-1: no plan, no behaviour change) | `fix`, harmonic bias, four wall geometries, COM and COG targets, schedule, per-item energy and gradient, calculator finalizer, `energy_contributions` writer, freeze velocity and coordinate masking, single-structure API, restart, optimizer displacement projection | central-difference gradients pass at the stated tolerance; base plus components equals total for C=1 and C>1; structure and its `E/G` come from one geometry; repeated evaluation does not accumulate; an unconstrained input reproduces current output exactly |
| C2: strict constraints at fixed cell | `bond_type` and connectivity groups (only after `perceive_bonds` gains the non-periodic branch and the unknown-pair failure of §3.8 item 1), bond SHAKE positions (RATTLE velocities deferred, and proposed as a keyword switch after the §1.3 measurement came out against them), projection call site in the MD loop, DOF from constraint rank, thermostat and COM adaptation, restart of solver state | non-periodic NVE and NVT long-run residuals and conservation pass; initialization, restart and two-trajectory isolation pass |
| C3: PBC and advanced projection | image-aware CV and group, angular constraints, constrained optimization, preset schedules | triclinic cross-boundary continuity passes; unsupported combinations keep failing |

> C3 status after #266: the image-aware geometry (searched nearest image, component unfolding, CV evaluation on unfolded coordinates) and the angular constraints both landed, together with the periodic wall guards and the crossing-count write-back. What C3 still owes is the constrained virial and cell-force assembly (A5, D-14), the velocity half of an angle or torsion, and the non-periodic bond perception (D-05) that a multi-atom CV over an unconnected group would need before it can be served rather than refused.
| C4: history potentials | CV Gaussian metadynamics, Cartesian and internal RMSD (reusing `kabsch`), commit and next-force continuation, finite history | one step never deposits twice; the new-history force for the next step is right at the same endpoint; restart equivalence passes. Hard dependency: the commit-point move of §3.4 must have landed |
| C5: cell coupling and extensions | cell and strain CVs, constraint virial, NPT, variable-cell freeze, well-tempered | cell and strain differentials, pressure sign and extended-energy coupling pass |

Phases are implementation dependencies, not alternative input systems. A user who needs a soft
restraint should not have to wait for metadynamics.

### C0 task list

| Task | Files | Change | Acceptance |
| --- | --- | --- | --- |
| C0-1 | `pyrai3md/utilities/selection.py` (new) | explicit index, range, element and named group resolution; cycle and empty-expansion and out-of-range failure listing the source | unit tests over a synthetic atom list; no import of schema or workflow |
| C0-2 | `pyrai3md/utilities/cv.py` (new) | distance, angle, torsion and coordination values plus coordinate Jacobians, wrapping `geometric.BND/AGL/DHD` with degree-to-radian conversion at the boundary, row-vector cell convention for cell-dependent values | analytic Jacobian against central difference on random non-singular geometries; zero-length bond and collinear angle refuse |
| C0-3 | `pyrai3md/data/constraint.py`, `pyrai3md/data/topology.py` (new) | `ConstraintSpec`, `ConstraintPlan`, `ConstraintState`, `ConstraintResult`, `GroupTable` as plain records with dict round trip and `evaluation_id`; no algorithms | `to_dict` and `from_dict` equality; pickle round trip; data records import nothing from workflow |
| C0-4 | `pyrai3md/schema/constraint.py` (new), `pyrai3md/schema/config.py` | `SystemConstraintConfig` plus mount as `System.constraints`; register in `NESTED_FRAMEWORK_SECTIONS:931-939`, `SECTION_TYPES` and type maps (`adapter.py:154-163`), `DEFAULT` (`config.py:29-188`), `FIELD_TO_SUBMODEL` (`:972-997`) | two walls and three bond biases in one YAML keep list order and unique ids; a flat kwargs config still routes correctly; every existing section regression unchanged |
| C0-5 | `pyrai3md/interface/constraint_parser.py` (new), `pyrai3md/interface/parser.py` | block state machine with stack, line numbers, accumulating lists; dispatch branch inside `_parse_text` and nothing else changed in the shared loop | the appendix §4.2 sample parses to the intended records; the same text with a missing `end` fails with a line number; `&end` closes only the outer block |
| C0-6 | `pyrai3md/schema/normalize.py`, `pyrai3md/schema/validate.py`, `pyrai3md/schema/adapter.py` | `normalize_constraints()` inserted between `:537` and `:542` and covered by the nested restore loop; `INDEX_START` slot; unit conversion through `ReadVal`; combination and capability refusals in the style of `validate.py:398-450` and `:624-639` | identical plan from every input form; index decremented exactly once; periodic, `driver != 'main'`, engine-owned plus hard constraint, and legacy-plus-new combinations each raise before any backend is built |
| C0-7 | `pyrai3md/qualification/certification/converter.py` | nested section support at the seven points of §4.1, including new rendering code beside `_render_molecule_section:785` | YAML to `.input` to config round trip preserves every item; the legacy copy has since been deleted (#301) |

### C1 task list

| Task | Files | Change | Acceptance |
| --- | --- | --- | --- |
| C1-1 | `pyrai3md/workflow/constraints/{controller,bias,wall,projection,schedule}.py` (new) | staged API of §8; fix mask projection and COM and COG targets; schedule resolved per evaluation, no module clock | same geometry twice gives same numbers; wall is zero inside, zero gradient at the boundary, positive outside; group centroid energy counted once |
| C1-2 | `pyrai3md/workflow/calculator.py` | new finalizer called after `convert_stress_to_atomic:1297` and before `set_result:1303`; writes totals, base copies and flat `energy_contributions` | one application per evaluation for C=1 and C>1; `print_results` shows the rows; unconstrained runs byte-identical |
| C1-3 | `pyrai3md/data/core.py` | read-only constraint accessors, atomic state commit, recursive ndarray normalization at the `set_result:1138` boundary | nested components survive local store and a subprocess payload round trip |
| C1-4 | `pyrai3md/workflow/integrator/operations.py` | `apply_constrain` becomes a thin adapter for freeze masking and velocity zeroing, with a comment stating that additive `E/G` do not belong here; `velocity_correction:331-333` promise made real | frozen atom coordinates and velocities invariant over a 30-step run; COM removal still correct |
| C1-5 | `pyrai3md/data/runtime/serializer.py` | plan and state into the payload, constraint semantic version, fingerprint and atom-order staleness | continuous run equals save plus restore; a plan built for another atom count raises |
| C1-6 | `pyrai3md/workflow/optimize.py`, `pyrai3md/workflow/optimizer/base.py` | implement `apply_constraint:248-265` as displacement projection plus constrained convergence report, consuming gradients already in Hartree per Bohr | relaxed geometry satisfies the constraint; reported `E/G` pair the final structure |
| C1-7 | `pyrai3md/qualification/verification/test_constraint_*.py` (new) | the suite of §15 | discovery by the existing planner, no external quantum chemistry software |

## 15. Verification Plan

Tests live in `pyrai3md/qualification/verification/test_constraint_*.py`, are auto-discovered by
the existing planner, and use analytic toy systems and a dummy backend, so the constraint
mathematics is verified without any external program. Errors keep full tracebacks and fixtures are
inline rather than read from disk, per the project test conventions.

The behaviour matrix of appendix §14 is adopted, with three groups added by this audit and two
corrected. **The file names below are the plan's own map, kept because each row states what an
area was written to prove; the tree has since merged several of them.** `test_constraint_potential.py`
is now `test_constraint_potentials.py`, `test_constraint_composition.py` lives in
`test_constraint_controller.py` (`TestComposition`), and `test_constraint_legacy.py` in
`test_constraint_legacy_equivalence.py`. The rest of the current set is whatever the
`test_constraint_*.py` glob finds; a name below that no longer matches a file has its coverage in
that set, not in a deleted file of its own:

| File | Proves |
| --- | --- |
| `test_constraint_parser.py` | the flat `&constraint` grammar, both directions: segments and their object order, broadcast/pad/overlong and the note each one leaves, `wall_type` gating with the ignored keywords echoed, the five ramp keys with their per-object fallback and component form, the refused spellings (`shake_*_fc`, `bias_ref`, the legacy `cbond` family), and that the interface layer hands the section over as raw strings. It replaced the block-stack cases of the first version when the grammar went flat (#271), and it ends with the writer: a structured block written to text and read back gives the same plan fingerprint. The wall's own keys (`wall_groups`, `wall_ref`, `wall_center`, `wall_law`, and the `none` segment that lets one wall declare no anchor while its neighbour does) are pinned here in both directions |
| `test_constraint_schema.py` | the block is mounted as one `System` field whose default enables nothing (`constraints` is `None` unless the input writes the section, and the section is registered in `NESTED_SECTION_FIELDS` under both its current and its retired name); index converted once; units equivalent across spellings; combinations refused |
| `test_constraint_cv.py` | Jacobians by central difference at three step sizes; translation and rotation invariance; degenerate geometry refusals; dihedral branch on both sides |
| `test_constraint_potential.py` | wall interior, boundary and exterior; box equal to diagonal triclinic under the row-vector convention; COM with unequal masses; COG equal sharing; multi-wall addition; group energy not duplicated |
| `test_constraint_composition.py` | base plus components equals total for C=1 and C>1; the `set_result` nested-array boundary; repeated evaluation and theory or reference switching do not accumulate; bias never reaches `electronic` |
| `test_constraint_integration.py` | a regression case that fails if the additive force is attached at `md.py:486`: the first kick of the step must already carry it; and both kicks agree with one constraint version of the force |
| `test_constraint_projection.py` | freeze mask invariance including velocities and gradients (the §3.1 behaviour change), compatible and conflicting targets, redundant constraint rank, and the `velocity_constraint` switch on both branches: the unset record returns the caller's own array untouched, the set one reaches `J v` inside the tolerance while the projected coordinates stay byte-identical |
| `qualification/verification/test_bonds.py::TestFocalPerception` | the focused bond scan returns exactly the rows the all-pairs probe returns inside the reached component, ignores what it does not reach, and gives the same row whichever end of the pair scanned first
| `test_constraint_metadynamics.py::TestWellTempered` and `::TestCommitWithAFullBias` | the height recurrence against the accumulated bias (the third height is what separates it from scaling the previous height), plain deposition unchanged, the underflow case spending its token without growing the history
| `test_constraint_shake.py::TestMultiConditionVelocity` | every `J v` driven to zero, no net momentum from the impulse, bond-only agreement with `solve_velocities` to 1e-14, rank dropping the dependent row, no in-place mutation of the caller's velocities
| `test_constraint_projection.py::TestOneCenterHelper` | the shared center reproduces the inline COM bit for bit, and `operations` reads the helper (AST) instead of a second weighted sum
| `test_constraint_integration.py::TestOneRestorePathOnly` | the dead second restart path stays deleted; the live refusal is the serializer's
| `qualification/verification/test_geometric.py::TestSecondDerivatives` | the bond and angle curvature against a central difference of this module's own analytic gradient at three step sizes, with the tolerance declared per step and a convergence clause; symmetry, rigid-shift invariance, degenerate refusals, and the torsion answered as absent rather than zero |
| `test_constraint_controller.py::TestAnalyticHessian` | the analytic block against a four-point double difference of the additive **energy** - the instrument is validated first on an angle bias off target, then used on the hills; five hill configurations (single bond variable at one and two sigma, an angle variable, two variables at mixed offsets, three hills at once) agree elementwise to 2e-4 relative, with the step chosen from a measured ladder (worst gap 5.3e-3 at h=1e-2, 4.8e-4 at 3e-3, 5.3e-5 at 1e-3, 4.8e-6 at 3e-4) and stated in the helper's docstring; bias cases at and off target, all four wall geometries under atom/COG/COM targets at p=2 and p=3, an uncrossed wall contributing exactly zero, a fix or strict constraint contributing nothing, and the named refusals of an aligned-RMSD history and an unknown variable kind |

| `test_constraint_periodic.py` (added by #266) | the periodic seam on a real cell: capability-granular guard (a wall and a fix pass, a collective variable is refused by name), the four wall input checks, the half-box gate with its axis and its negative control, the unfolded centroid against the wrapped average, and `fold_back` paired with `unwrap_coordinates` so the counts written beside a wrapped coordinate are the ones it implies |
| `qualification/verification/test_lattice.py::TestMinImageCandidateSearch` | searched nearest image equals the component-wise rule bit for bit on an orthogonal cell, beats it by over 2 Angstrom on a 60 degree hexagonal cell, agrees with a brute-force shell, and refuses a cell that is not size-reduced |
| `qualification/verification/test_geometric.py::TestComponentExpansion` and `::TestImageAwareEvaluation` | the cross-face bond negative control (11.54 becomes 0.96 Angstrom), the offset table as the write-back map, anchor and row-order independence, the no-cell identity path, unchanged numbers to the bit when nothing crosses a face, and a half-cell translation that moves the value and the Jacobian by nothing |
| `test_constraint_shake.py::TestMultiConditionSolve` | the multi-condition sweep: bonds reproduced to 1e-14 against the pair solver with the same iteration count, an angle and a torsion converging to 1e-12 relative, no net momentum invented by the correction, redundancy decided by rank (three planar angles cost two degrees), and a collinear geometry failing by name |
| `test_constraint_statistics.py` | rigid three-atom water 9 minus 3 equals 6, joint rank of mask and hard conditions, zero DOF refusing temperature control, COM deduction counted only when actually performed, log and thermostat reading one source |
| `test_constraint_restart.py` | continuous equals save plus restore, plan and atom-order mismatch raises, subprocess and Slurm payload carries constraints, a worker does not double-add |
| `test_constraint_workflow.py` | optimize pairing of structure with `E/G`, numerical Hessian with frozen history, learning labels free of bias, refused combinations reported early rather than ignored |
| `test_constraint_legacy.py` | `k = 2f`, explicit initial target, fixed center, polynomial nonzero interior, group weighting difference against the new COM semantics |

Smooth non-singular test points use `atol = 1e-7` Hartree per Bohr and `rtol = 1e-5`, checked over
several differentiation step sizes; projection tests are judged by the user tolerance and the
normalized residual. For a long MD run the two claims are separate, and the second one is now
measured rather than assumed: the constraint residual must converge in `dt` and tolerance, and it
does (§1.3, `1e-12` relative, on both branches), but the total energy of the **default** branch does
**not** converge in `dt` and must not be certified by a convergence-in-`dt` test; the branch that
declares `velocity_constraint: rattle` does converge, first order, and its drift assertion belongs to
the same table (§1.3). A single absolute energy threshold is not used in either case.

Every group above additionally carries the DC-1 check: the same input with no constraint block
must reproduce the pre-change output exactly, which is what makes shipping this subsystem closed
verifiable rather than assumed.

Smallest end-to-end demonstration, and the C1 exit criterion: a dummy backend single point on four
atoms with one bond harmonic bias, one sphere wall and one frozen atom, asserting that the total
equals base plus the two components, that the total gradient matches the central difference of the
total, that `energy_contributions` prints the named rows, that the frozen atom's gradient and
velocity stay exactly zero, and that removing the constraint block reproduces the current output
unchanged.

## 16. Document Update Requirements

This document is the authority for constraint, CV and bias-potential design. The following sites
currently hold competing or stale content; each is edited in the milestone named, and none keeps a
rival field list afterwards.

| Document | Required change | Phase |
| --- | --- | --- |
| `dev/methods/ENSEMBLE_DEVELOPMENT.md` §4 (`:239-248`) | delete the competing constraint field memo (`BondConstraint`, `AngleConstraint`, `CavityConstraint`, `SHAKEParameters`, `CollectiveVariable`, `Metadynamics` parameter lists) and leave a pointer here; answer the framework-versus-skill ownership question at `:217` with §3.7; mark the bias-into-virial point at `:229` as C5; keep `N_df = 3N - N_c - N_com` at `:159` with `N_c` sourced from §10 | C0 |
| `dev/methods/MD_DEVELOPMENT.md` §11, §12 and the loop-order text (`:586`, `:627`, `:761`, `:959`) | replace "the first version only reserves the interface" with a pointer; move frozen-atom ownership here; correct the claim that external potential and constraint are unified in `apply_constrain` into the split of §7 | C0, C1 |
| `dev/methods/PBC_DEVELOPMENT.md` §4.2, §6 | done (2026-10-07 rewrite): §4.2 now states that `apply_constrain` delegates to `constraint_runtime.constrain_state` and lists the periodic constraint geometry that landed (`workflow/constraints/images.py`), with the region/embedding half left at stage E; the document was re-audited line by line against the code and its stage letters are cited from `schema/validate.py:406`, `:418` | C3 |
| `dev/methods/OPTIMIZE_DEVELOPMENT.md` §14 (`:354-374`) | point at the existing `Optimizer.apply_constraint` hook and §11 | C1 |
| `dev/WORKFLOW_DEVELOPMENT.md:678`, `:1380`, `:165` | describe `apply_constrain` per §7; drop `MetadynamicsWorkflow` as a workflow class in favour of the controller; fix the stale `dynamics/` tree line | C4 |
| `dev/DATA_CORE_DEVELOPMENT.md:293-297`, `:318`, `:332`, `:643-647` | redirect the preserved deleted-package field design to §6 | C0 |
| `dev/TODO.md` Phase V memo | repoint the "constraint, CV and metadynamics field design memo" reference from `ENSEMBLE_DEVELOPMENT.md` §4 to this document | done with the shadow build (§17) |
| `dev/ARCHITECTURE.md:105` and the module tree | add the `workflow/constraints/` and `utilities/` entries when the files exist; `ARCHITECTURE.md` stays the only tree maintainer | C1 |
| `dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` shake section | record that framework-side SHAKE does not replace the engine-side `fix shake`, per the existing skill validation guard | C2 |

`CHANGELOG.md` (project root) gets one entry per implementation milestone, following the existing numbering
and date conventions.

## 17. Build Status, at the Framework Paths

The numeric core of C0 and C1 exists **inside the package** - `pyrai3md/data/constraint.py`,
`pyrai3md/schema/constraint.py`, `pyrai3md/utilities/{selection,cv,rmsd}.py`,
`pyrai3md/interface/constraint_parser.py` and `pyrai3md/workflow/constraints/*` - and is wired into
the Calculator, the integrator, the ensemble thermostat and the serializer (§19). Before the merge
these same files lived under `pyrai3md/constraint_lab/` behind an import discipline that let no
framework module see them; that discipline, its README file map and its two guard tests existed only
to keep the move mechanical, and they expired when the move ran (§19.0). What survives of the rule
is narrower and is enforced in `test_constraint_state_adapter.py`: the state adapter must not import
the state class, the config model, the parser or the calculator it is called by, because it is handed
those objects rather than owning them.

| Phase | Built in the lab | Still deferred to integration |
| --- | --- | --- |
| C0 | explicit selection, CV values and Jacobians, plan and state records, `SystemConstraintConfig`, `build_plan`, the `&system_constraint` block parser, unit and index normalization, capability refusals | mounting on `System`, the four registration points, `normalize_constraints`, `validate.py` refusals, the `_parse_text` dispatch branch, certification round trip |
| C1 | harmonic bias, four wall geometries with `atom`/`com`/`cog` targets, the `legacy_polynomial` cavity as a fifth family with its own conversion rule (§13), fix mask and anchors, schedule resolution, rank-based DOF and free kinetic energy, the staged controller with `base + components == total`, the result-shape contract of §9 (scalar plus `(natom, 3)`, `(ns,)` plus `(ns, natom, 3)`, energy with no gradient), empty-plan short circuit | the Calculator finalizer, the `apply_constrain` adapter, the `set_result` recursion boundary, serializer version and restore, `energy_contributions` writer, MD loop call sites, optimizer displacement projection |
| C2 part | bond SHAKE position solve with normalized residual, non-reversible `initial_projection`, redundancy and conflict detection, mass-weighted correction; the measured cost of leaving velocities alone (§1.3); **and the velocity half behind its keyword**: `shake.solve_velocities` plus `velocity_constraint: none \| rattle` on the record, default `none`, routed by `controller.project_velocities` and measured on both branches through that route | the projection call site in the timestep, thermostat and COM adaptation, long-run NVE and NVT gates, and whether any run should ever turn the keyword on by default (it should not, per §1.3) |
| C4 part | fixed-history Gaussian CV potential, deposition tokens keyed by `(trajectory_id, accepted_step, item_id)`, hill commit returning a new state, next-step force discipline, **and the second backend**: the aligned `D^2` variable of `utilities/rmsd.py` running through the same Gaussian kernel | the `kabsch` fix that lets the lab's duplicate alignment be deleted (D-10), the commit-point move that must precede any wiring |
| C5 math, not dynamics | cell axis length, axis-pair angle, volume and the six Green-St. Venant strain components with exact `dq/dB`, under the row-vector convention; the statement that a cell variable has no atomic gradient at all and that a wall boundary does not deform with the cell | constraint virial, variable-cell freeze, NPT coupling, and the reuse of `verlet._retriangularize` instead of a second Cholesky (D-14) |
| integration rehearsal, not integration | `state_adapter.read_state` and `require_usable`, tested against a stub that carries the real accessors' quirks (`mass_au` returning None, the 99-sentinel `cell`, `current_state` raising for C>1, a non-zero `System.image`), **and one test that runs the merge round's whole call order on that stub** (`tests/test_wiring_rehearsal.py`): read, refuse-if-unusable, bind, project, evaluate, combine over the three result shapes, commit, hand the payload through pickle and JSON, rebuild it, flatten it and print it with the framework's own `_format_energy` | the adapter's two call sites: `apply_constrain` and the Calculator finalizer. Nothing in the framework imports this file; the stub is the only thing it has ever read |
| §13 migration, one function deep | `schema/legacy_adapter.adapt_legacy_fields`: the nineteen flat classic-input keys to a mapping that feeds `build_plan`, with `k = 2 f`, `alpha = max(factor, 2)`, the defaults-are-not-intent gate of §13, the refusal to migrate `constrain` by name, and a coexistence error when the old keys and a `&system_constraint` block both speak | the call site in the config normalization, and deleting the archived cavity arithmetic once the plan path owns it |

Verification status. `python3 -m unittest discover -s pyrai3md/qualification/verification -t . -p
"test_constraint*.py"` runs 785 tests (792 when the selection is made by `-k constraint`, which also catches the constraint cases in shared files), and `pyrai3md/utilities/tests/` adds 132 more,
covering
the acceptance items of §15 that do not need a wired framework: analytic Jacobians against
central differences at three step sizes, `atol = 1e-7` Hartree per Bohr; dihedral sign pinned to
cis, trans and gauche; box equal to diagonal triclinic; interior, boundary and exterior wall
values; COM gradient sharing by mass and COG sharing equally; rank of a redundant constraint;
zero DOF not clamped; the freeze-only case reproducing `3 * (natom - n_frozen) - 3` exactly;
SHAKE residual convergence and its failure report; no double deposition on a replayed step;
`base + sum(components) == total` and its idempotence; the empty plan allocating nothing. To those
this round adds: the three result shapes of §9 with a missing gradient staying missing and no
invented zero row; the aligned `D^2` against an independently written superposition, its rigid-
motion invariance, its central-difference gradient and the refusal where the optimal rotation is
not unique; the same variable driven through the existing Gaussian hill kernel with its reference
bound once, not re-read from the live geometry; the legacy cavity against the archived arithmetic,
including the per-molecule duplication and the gradient inconsistency recorded as D-15; the cell
variables against central differences of the lattice, a rotation of the frame proving to be not
strain, and the statement that an energy written on the cell has no atomic gradient at all; and the
state adapter against a stub that carries the real accessors' quirks, including the two that must
fail loudly (`mass_au` returning None, a non-zero `System.image`) and the property that must never
be read (`current_state` on a multi-center run). The 2026-09-29 round adds four more: a
characterization of the §3.9 dihedral defect measured row by row against central differences of the
function's own return value, so the register cannot drift back to the wording that misled an earlier
round; the legacy field adapter, with its defaults-are-not-intent gate, its `k = 2 f` arithmetic
checked against the archived potential and its coexistence refusal; the wiring rehearsal above,
which reaches the framework's print reader and asserts the rendered rows are countable; and the
`velocity_constraint` switch on both branches, including the strictest form of the default-closed
check (the unset record returns the caller's own array, `assertIs`) and a plan fingerprint that
changes with the keyword so a restart cannot flip it silently. One file integrates rather than
unit-tests: the measured cost of position-only SHAKE in §1.3, marked skippable because it is a
dynamics run, and since this round every number in it comes from the controller rather than from a
helper written in the test file.
Everything in §15 that needs the framework (composition through the Calculator, restart, remote
dispatch, MD integration, workflow and print-out paths) is not tested yet because it is not
reachable yet.

Default-closed status, after the merge. DC-1 is no longer a construction property - the code is
imported by the framework now - so it is checked as behaviour: `test_constraint_end_to_end.py` and
`test_constraint_integration.py` assert that the same input with no constraint block reproduces the
unconstrained numbers exactly, that the finalizer returns before it touches a number on an empty
plan, that a projection call returns the caller's own velocity array, and that an empty plan writes
no key into `energy_contributions` and no block into the runtime file. DC-3 (no allocation for an
empty plan) and DC-4 (defaults are not intent) are asserted in the same files. The subsystem is
therefore closed by test, which is the stronger of the two forms.

## 18. Merge Duties: Fix the Recorded Defects, Then Delete the Duplicates

Standing rule from the user (2026-09-28): a defect found on the way is recorded, and the round
that merges the lab into the framework must **fix it and remove the duplicate**. Neither half is
optional. Fixing without de-duplicating leaves two implementations that can disagree later;
de-duplicating without fixing preserves the wrong one. The merge is not done while any row below
is open - which is why section 18.1 names the rows this merge did not close rather than
leaving the reader to count. (While the lab existed, `constraint_lab/tests/test_integration_manifest.py`
checked that this register and that directory's README checklist listed the same item ids; the
guard expired with the move, its two operands having become one document.)

A shadow build necessarily duplicates some arithmetic. That duplication is the price of not
touching files another line is editing, and it expires at the merge: the lab copy is deleted, the
surviving implementation gains the lab's guards and the lab's tests.

| ID | Defect or duplicate today | Merge action | Proof it is closed |
| --- | --- | --- | --- |
| D-01 | `utilities/geometric.py::DHD` gradient is the derivative of neither its own value nor any torsion: on generic geometry only row `a` is right, `b` and `c` are wrong in sign *and* magnitude, `d` is exactly negated, and the four rows fail rotational invariance (§3.9, `constraint_lab/tests/test_geometric_defect_evidence.py`); `constraint_lab/utilities/cv.py` carries a correct implementation | fix the shared implementation once, in place, and delete the lab torsion | the characterization file above is **deleted, not repaired**; the central-difference, Noether-residual and cis/trans/gauche pins move to `qualification/verification/test_geometric.py`; only one torsion function remains in the package |
| D-02 | `geometric.{BND,AGL,DHD}` return nan instead of failing: a zero bond length, a collinear angle, and - worst - a strict trans or cis dihedral, where `DHD` returns nan **value and gradient** together (`arccos` domain plus a `sqrt(1 - cos^2)` division by zero at the most common real conformations) (§3.9) | adopt the `CVSingularError` policy in the shared module; callers that relied on nan become explicit failures | degenerate-geometry tests live in `utilities/tests/`; the lab wrapper is gone |
| D-03 | degree handling: `geometric.py` hardcodes `57.2958` while `utilities/units.py` is the constant source | one conversion, from `units.py`, at the module boundary | the literal appears in no source file; grep is part of the merge checklist |
| D-04 | two center-of-mass notions: `operations.velocity_correction` computes a mass-weighted COM **velocity**, `constraint_lab/utilities/cv.py::group_center` computes a center of positions | one mass-weighted center helper, used by both the constraint walls and the COM drift removal | the wall gradient-sharing test and the COM test read the same helper |
| D-05 | bond perception exists once (`utilities/bonds.py::perceive_bonds`) but only for a periodic cell (§3.8); the first version therefore takes explicit pairs and refuses `bond_type` | extend that one routine with a non-periodic branch and an unknown-pair failure instead of a second table, then implement `bond_type` on it | image seeding, constraint groups and `bond_type` all resolve through it and agree on one bond list for one geometry |
| D-06 | `ensemble._dof` counts frozen atoms while nothing freezes them, and clamps a physical zero to one (§3.1) | replace the formula with the rank-based `ConstraintStatistics`; this is a defect fix that moves NVT and NPT temperatures, so it gets its own changelog entry | the freeze-only equivalence test proves the two agree where the old one was right; a zero-DOF test proves no clamping |
| D-07 | `apply_constrain` is a no-op and `velocity_correction`'s docstring already promises frozen-atom re-zeroing to it (§3.1, §3.7) | wire the adapter; either honour the promise or correct the docstring, never leave both | a run with `freeze` keeps those atoms' coordinates and velocities invariant |
| D-08 | `set_result` converts only top-level ndarrays (§3.5) | recursive conversion at that one boundary | nested component arrays survive a subprocess payload round trip |
| D-09 | the multi-center commit happens inside the calculator (§3.4), which is wrong for anything keyed on accepted steps | land the announced commit move, then wire metadynamics | a rejected step deposits nothing; a replayed step deposits once |
| D-10 | `operations.kabsch` neither centers nor mass-weights and its docstring claims a rotation it does not return (§2) | correct it and reuse it for the RMSD history potential; no second RMSD implementation | `utilities/rmsd.py`'s own centering, weighting and Kabsch rotation are deleted once `kabsch` does the same job; until then its value is compared against an independently written superposition, so the duplication is measured and not trusted |
| D-11 | the legacy cavity is a full-space polynomial that is nonzero inside (§13), while `constraint_lab/.../wall.py` is outside-only | implement `legacy_polynomial` as its own mode in the constraint module; the legacy module stays a reference, not a second engine | done in the lab (`workflow/constraints/legacy_cavity.py`, its schema family and the `cavity` DSL block); at merge the legacy `System` fields map onto it through the adapter, and the archived module is never imported |
| D-12 | `energy_contributions` has a reader and no writer (§2) | add the writer in the finalizer; do not open a parallel breakdown channel next to `composite_terms` | the existing print path shows `constraint.<kind>.<id>` rows and the composite section is unchanged |
| D-13 | two things are called a schedule: `workflow/dispatch/scheduler.py` runs jobs, `constraint_lab/.../schedule.py` ramps parameters | no de-duplication; name the boundary in both docstrings so neither is searched for the other | grep for "schedule" in the architecture notes shows the distinction |
| D-14 | a strain or cell CV differentiated with respect to the lattice produces a deformed cell that leaves the framework's lower-triangular convention, and `verlet._retriangularize` already owns that Cholesky-and-rotation step (§2, dev/methods/MD_DEVELOPMENT.md) | reuse `verlet._retriangularize` for any re-triangulation the cell driver needs; do not write a second Cholesky | one re-triangularization path in the tree, and a cell-constrained run rotates its frame the same way the barostat already does |
| D-15 | the archived cavity's energy counts each molecule's centroid term once per atom of that molecule, while its gradient carries no such factor (`legacy/PyRAI3MD/Molecule/constraint.py:193-237`, §13), so the archived force is not the derivative of the archived energy | keep the archived energy (a migrated run's cavity energy must not move) and take its own derivative, as the lab does; state the atom-count difference in the migration note rather than reproducing the inconsistency | the golden case asserts the energy equality and the per-atom gradient ratio, so the divergence is on record; if a legacy trajectory is ever reproduced force-for-force, this row is the reason it cannot be |
| D-16 | `constraints/runtime.py::restore` is dead code **and** disagrees with the writer: it reads `payload['plan_fingerprint']` / `payload['state']`, while `_constraint_payload` (`data/runtime/serializer.py:554`) writes `{'semantic_version', 'fingerprint', 'state'}` nested under `payload['constraint']`. No caller exists anywhere (`grep '\.restore('` over the package: zero hits). The same duty has a live, stricter implementation in `serializer.py:496 _check_constraint_payload`, which also compares the semantic version | delete `runtime.restore`, do not repair it: two restart checks are two things that can disagree, and the serializer's is the one on the path that actually runs. If a future need puts the check below the serializer, move it rather than keep a copy | `test_constraint_integration.py` gains a row asserting `runtime` exposes no restart-refusal function, alongside the existing fingerprint and semantic-version refusals that go through `StateSerializer.load` |
| D-17 | `workflow/integrator/ensemble.py::_dof` rebuilds a `SystemConstraintController` and re-assembles the condition matrix on every call, and `_dof` is called from four sites on the thermostat and barostat paths (`:678`, `:733`, `:934`, `:1005`), so a long constrained NVT/NPT run pays that per step. The plan itself is already cached (`data/core.py:1136`), so the repeated work is controller construction plus Jacobian assembly | **measure before proposing**: the cost of the constraint step per MD step and its share of the whole step, on a constrained system of realistic size. Only then decide whether anything may be cached, and with which key - a cache keyed without `coords` would be silently wrong, because the rank of the condition matrix is a function of the geometry (a collinear angle costs a rank row that a bent one does not) | the measurement goes in section 20 with its numbers and its machine, and any cache that comes out of it is guarded by a test that changes geometry and expects a changed count |

Anything found after this round appends a row here and, when the row describes framework work, an
item to the wiring checklist in section 19.3 in the same change, so the register cannot drift from
the work it describes. A defect that someone else fixes
first keeps its row and is marked closed with the changelog number that closed it: deleting a row
would erase the evidence that the merge owed a fix in the first place. While the lab existed its
own README carried the checklist and `test_integration_manifest.py` failed if the two id lists
stopped matching; the move retired both, and the divergence it guarded against is now prevented by
one document holding both halves.

### 18.1 Disposition at the merge (2026-09-29, CHANGELOG #264)

The register above is left intact on purpose: a row that someone else closes first keeps its text,
and this table is where the outcome is recorded. "Closed" means both halves of the standing rule -
the fix landed *and* the duplicate removed.

Six rows are not closed by this merge: D-04 (the shared center helper, whose other half lives in
the file the integrator line was editing), D-05 (`bond_type`, open by design until `perceive_bonds`
grows a non-periodic branch), D-09's second half (the calculator's multi-center commit move, which
is R6's change to make) and D-14 (no cell driver exists yet, so there is nothing to de-duplicate), plus the two rows the dependency-structure audit added afterwards (D-16, D-17), neither of which was found by a failing test.
The standing rule's exit is met for every row this round could own and is explicitly outstanding for
those six: they are debt carried past the merge, recorded here with the reason, not a checkbox
marked complete.

| ID | State | Evidence or reason it stays open |
| --- | --- | --- |
| D-01 | closed | `utilities/geometric.py` holds the atan2-invariant torsion; `utilities/cv.py` delegates to it and its private torsion is deleted; the characterization file was deleted, not repaired, and its pins live in `qualification/verification/test_geometric.py` |
| D-02 | closed | `CVSingularError` raised for a zero bond, a collinear angle and a collinear or numerically undefined torsion; the strict cis/trans case is asserted to give a **finite** value and gradient where the old form gave a nan pair |
| D-03 | closed | no degree-conversion literal remains in the module's parsed numbers (`TestNoMagicConstants` reads the AST, so the prose that names the old factor does not indict itself); angles are radians, derivatives radians per length |
| D-04 | **open** | one mass-weighted center helper shared by the walls and the COM drift removal is still two implementations. It did not happen in this round because `operations.velocity_correction` is the integrator line's file and was being edited concurrently; the constraint side reads only `utilities/cv.py::group_center` today |
| D-05 | open by design | `bond_type` and connectivity groups still refuse; `perceive_bonds` has no non-periodic branch, and inventing a second radius table is prohibited (section 3.8) |
| D-06 | closed on the constrained path, deliberately partial elsewhere; the periodic side of it is a measured run now (#276, sections 26) - NVT Nose-Hoover with two held bonds keeps `8.131e-11` A at a declared `1e-10`, and a `fixes` mask holds `0.000e+00` A under both NVE and NVT in the cell. NPT is the exception and is D-25 | `_dof` takes the rank of the condition matrix and refuses a zero count instead of clamping. The `max(..., 1)` clamp stays for runs with no plan, because removing it changes every existing NVT/NPT temperature and that behaviour belongs to the ensemble line |
| D-07 | closed | `apply_constrain` delegates to `constraints/runtime.constrain_state`; the freeze is real, and a run with one frozen atom shows its position unchanged to `0.0e+00` in every trajectory frame |
| D-08 | closed | `PyRAIState._to_plain` converts recursively at `set_result`, so nested component arrays survive a subprocess payload |
| D-09 | partial | the *constraint* history now commits at the accepted step (`md.py` calls `constraints/runtime.commit`), which is the deposit-once behaviour this row asks for. The calculator's own multi-center commit move (R6) is the ensemble line's change and is not done here |
| D-10 | closed | see W-12: one Kabsch implementation, `kabsch == sqrt(rmsd_squared)` exactly, degeneracy guard in the shared code |
| D-11 | closed | `legacy_polynomial` is a family of the constraint module; the archived module is never imported; the flat fields migrate through `schema/legacy_adapter.py` and the draft case `constraint_legacy_cavity` shows the migrated action producing a row in a real run |
| D-12 | closed | the finalizer writes flat `constraint.<kind>.<id>` keys into `energy_contributions`, printed by the existing reader; no second breakdown channel |
| D-13 | closed | both docstrings state the job-scheduler / parameter-ramp boundary |
| D-14 | closed for a held length, an angle and a torsion (2026-10-13; the last two measured later) | the cell driver exists - `optimizer/base.py::_cell_ride` moves the box and carries the atoms with it, and `_cell_gradient` is the strain conjugate - and the condition's own lattice derivative is applied beside it (`constraints/projection.py::cell_virial`, priced in `verification/test_constraint_cell_coupling.py`, including `TestAnAngleAndATorsionHaveTheSameCellForce`: worst entry 1.13e-11 rad per Bohr for the angle and 2.73e-12 for the torsion). `workflow/optimize.py::CELL_MEASURED_CONDITIONS` is the three-kind set; a fourth kind is refused by name; `verlet._retriangularize` remains the only allowed re-triangulation |
| D-15 | closed as designed | the archived cavity energy is kept and its own derivative taken; the atom-count factor is stated in the test that pins both halves |
| D-16 | **open** | found by the dependency-structure audit after the merge, not by a failing test: `constraints/runtime.restore` has no caller anywhere and reads payload keys (`plan_fingerprint`) that the writer never produces, while the live refusal is `serializer._check_constraint_payload`. The deletion is the fix, so nothing is repaired until the row closes |
| D-17 | **open, deliberately unmeasured** | `_dof` rebuilds the controller and the condition matrix on every call from four thermostat/barostat sites. No number exists yet for the per-step cost, so no cache is proposed - and a cache keyed without `coords` would be wrong, because the rank of the condition matrix depends on the geometry |

### 18.2 Second round: the periodic path and multi-condition SHAKE (2026-09-30, CHANGELOG #266)

Ordered by the user's ruling in 附录 B section 0 - the bond / angle / torsion
functions first, then the wall, then real measurement - and recorded here in the same order.

| Item | State | Evidence |
| --- | --- | --- |
| A4 multi-constraint SHAKE (angle, torsion) | closed | `shake.solve_conditions` with the rank test in `condition_redundancy`; `schema/constraint.py:1554` no longer refuses `kind: angle` / `dihedral`; degrees of freedom come from the same rank (`controller.statistics` passes the condition list); `test_constraint_shake.py::TestMultiConditionSolve` and `test_constraint_end_to_end.py::test_an_angle_constraint_is_solved_end_to_end` |
| A7 periodic constraints | closed for what the bond graph connects | `utilities/lattice.py:468` searched nearest image with a certificate; `utilities/geometric.py:286` component unfolding; `constraints/images.py`; the guard at `state_adapter.py:212` is per variable (#274). Measured in sections 23 and 24 |
| A7's remaining half (D-05) | open by design | `perceive_bonds` still has no non-periodic branch, so a non-periodic structure has no bond table and a multi-atom CV over it is refused. The geometry core now *consumes* a `(i, j, delta)` table (M2 ruling: (a)), which is the half that did not need the perception to exist |
| Wall periodic handling (M3, ruling A) | closed | `wall.periodic_check` (four input-time refusals) plus the capability-granular guard; `test_constraint_periodic.py::TestWallInputChecks` |
| Constraint virial / cell coupling (A5, D-14) | untouched | deliberately out of this round's scope, as recorded in 附录 B section 7 |
| Angle / torsion velocity projection | open, refused by name | position path only; a record asking `velocity_constraint: rattle` on a non-bond constraint raises instead of projecting just its bonds |

### 18.3 Third round: the capabilities that did not need a machine (2026-09-30, CHANGELOG #267)

Ordered by what blocks them, not by size. Everything here is closed by offline tests; the
residual list is sorted into three buckets in 附录 B section 10.

| Item | State | Evidence |
| --- | --- | --- |
| D-05 non-periodic perception, then `bond_type` | closed (was: open by design) | `bonds.bond_graph`, `bond_pairs_by_type`, `perceive_bonds_focus`; `_Normalizer.bond_type_pairs` (written as `schema/constraint.py` at the time, now `schema/normalize.py:545`); 4 schema cases plus 4 focal-perception cases (`test_bonds.py::TestFocalPerception` pins row-for-row equality with the all-pairs probe and the orientation invariant) |
| A6 well-tempered metadynamics | closed | `MetadynamicsInput.bias_factor` / `temperature` with `dT = (gamma - 1) T` computed at input; height scaled by `exp(-V(s)/k_B dT)` at deposition; a flattened bias writes no hill and counts `zero_hills:<id>`; 7 cases in `test_constraint_metadynamics.py` |
| Angle / torsion velocity half (RATTLE) | closed | `shake.solve_velocity_conditions`, `controller._tangent_conditions`; 8 cases; the plan-wide projection rule is stated where it is enforced |
| Six-parameter cell in a wall's triclinic geometry | closed | `cell_to_internal` through the one converter `molecule.cell` uses; equivalence plus a bad-length refusal |
| D-04 one mass-weighted center | closed | `cv.group_center` is the single implementation, with the weighted sum divided once so the integrator's number does not move by a bit; 2 guard cases |
| D-16 dead `runtime.restore` | closed by deletion | no caller, payload keys that the writer never produced; the guard asserts the entry is gone |
| D-17 cost of the constrained DOF path | measured, then fixed at the source | per step (read + guard + statistics): 11.37 ms to 1.73 ms at 108 atoms, 85.59 ms to 4.05 ms at 300; perception alone 71 ms (300) and 295 ms (600) against 1.6 and 3.1 ms focused. No cache was added: a key without the coordinates would be a silent wrong rank |
| D-18 recorded total energy of a constrained run | **withdrawn (#283)**, and the `&constraint` decomposition the ruling asked for is implemented | The registered sentence was "`Epot` = the bare electronic energy (`-74.9629801723133937`)". Re-measured, the two things compared were **the same key**: the trajectory writer and the log both read `results['energy']`, so they agree by construction and the agreement says nothing. The test that can decide it needs an independent bare energy - the starting geometry of `constraint_md_bias` handed to an unconstrained PySCF single point gives **`-74.9631631182905238`**, the step's `Epot` is **`-74.9631292844484278`**, and the difference `3.3833842e-05` Ha is the bias row `0.0000338338420919` to 4e-18. **`Epot` already carried the constraint potentials**, so `Etot` is the quantity the integrator conserves. What was missing was the decomposition, which 口径 A asked for: `runtime.finalize` now also stores the engine's own half as `energy_origin`, and `print_results` renders `&constraint energies in Hartree` as `origin pot`, one row per constraint **under the id the input wrote**, and `Epot`. Verified on six products (the 2026-10-01 D-18 evidence set, deleted on 2026-10-02; §33): 5 steps of the PySCF bias run, 5 steps of a geometric-only SHAKE run (no rows, the two numbers equal), 5 steps of three non-spherical walls (`cuboid`/`ellipsoid`/`sheared`), 7 steps of the periodic triclinic wall under LAMMPS (worst sum residual `1.110e-16`, rest `0.000e+00`), and 7 blocks of the constrained optimization (`constraint_opt_surface`, row `pull`) - the "MD and later iterative workflows" the ruling named. An unconstrained run gets no key and no section |
| D-19 a resumed run wiped the trajectory | closed (#274) | `md.py` called `output.reset()` unconditionally, so the second segment deleted the first segment's frames. Now gated on `runtime.current_step > 0`; the criterion got stronger as a result - 4+4 steps agrees with a continuous 8-step run at `0.000e+00` A per frame, where the old claim was `6.5e-12` A |
| D-19b the resumed segment's time column | **closed (#281)** | coordinates and energies agree frame by frame, but the appended rows carry the previous step's time (`0.00/20.67/41.34/62.01/62.01/82.68/103.35/124.02`). Adding a writer-side offset does **not** fix it: `step` after a restore is already the absolute number (4,5,6,7), just one lower than the state it carries, so an offset turns "one behind" into a two-step gap (`...62.01/124.02/144.69/165.36/186.03`, measured). The fix belongs in the loop's starting step, not in `output.py`. #279 found the line: `md.py` writes the runtime file at phase 6 (`self.checkpoint(state)`, :619) and advances the counter at phase 7 (`state.runtime.advance_step()`, :630), so every checkpoint carries `current_step = steps done - 1` (read out of a real `prestart.rt`: `current_step` 3 after four steps, `simulation_time` 0.0, `total_steps` 0), and the resume's `step = current_step + 1` (:668) re-uses the step number the previous process already recorded. Closed with D-19's renumbering: the counter now advances before anything is written and the loop runs on absolute numbers, so the periodic 4+4 split's time column is `0.00 / 12.40 / 37.20 / 62.00 / 86.80` - the same five labels the continuous run writes, no repeats |
| D-20 a constrained optimization can accept an infeasible point | **withdrawn (#278)** | Registered in #274 as `1.0855` A against a declared `1.0863` A at `tolerance: 1e-10` - a slip of `8e-4` A. Re-measured against the product that holds the accepted geometries, the slip is `6.461e-11` A, and three independent readings agree: the archived artifact itself (the 2026-10-01 aperiodic set's 7-frame `opt_geom.xyz`, deleted; §33), a rerun of the shipped `constraint_opt_surface.yaml` on the current build, and a 2x2 of arms that add or remove the `fixes` and the bias one at a time (`7.224e-11` / `1.104e-10` / `9.920e-11` / `6.461e-11` A). The origin of the bad number is a bench bug, not the run: `constraint_aperiodic/measure.py:145` opened `<title>.opt.xyz`, which the optimizer never writes (it writes `opt_geom.xyz` and `opt.xyz`, unprefixed), the `if os.path.exists` guard fell through to printing a directory listing, and a number beside that listing was read as a criterion. The lesson is kept as a rule: a measurement script must fail when its artifact is missing - a silent else-branch turns a missing file into an unfalsifiable report |
| D-21 the periodic guard asked per kind | closed (#274) | `bias`/`metadynamics` were refused outright in a cell, and `check_periodic` refused their blocks at input time; the rule is now `require_same_component` per cv at binding time (`_validate_images`), which serves an intramolecular bond and names the intermolecular one. The campaign's decisive number: a cv across a face reads `0.959168` A on the minimum image and `7.618022` A in storage, and the bias row stays at `5.0e-6` Ha |
| D-22 the periodic wall re-asked "does a member start outside" at every step | closed (#275) | `_validate_images` called the declared-geometry `periodic_check` without `check_members`, so the containment rule that says *starts* outside ran on the live geometry from `runtime.finalize` at every evaluation. All three non-spherical walls died the same way once the confined water reached the surface - `pwall_box` reported `binds atom 29 while it sits 0.0011 Bohr outside` after the first printed step, in a run whose own contribution row then went to `2.064651e-04` Ha. A one-sided penalty wall is crossed by thermal motion and pushes the atom back, which is the job; the guard is now tied to `view.step == 0` and the ramp endpoints keep their separate `check_members=False` pass. Criteria: the same out-of-size wall still refuses at step 0 and is served at step 1 |
| D-23 an aligned-RMSD history potential read an attribute that does not exist | closed (#275) | `_variable_names` fell back to `spec.cvs` when the action names no cv, and `ConstraintSpec` has no such field - the reference is that one line in the whole repository. Since `kind: rmsd` is exactly the action that carries its block instead of a cv name, the first periodic rmsd run died at its first evaluation with `AttributeError: 'ConstraintSpec' object has no attribute 'cvs'` rather than with a refusal. The fallback is deleted; rmsd returns no names and `_variable_atoms`' own `spec.atoms` path hands the block to `require_same_component`. Criteria: a whole water's rmsd potential is served in a cell, and a two-atom block the graph does not connect is refused as `metadynamics 'shape' rmsd block` |
| D-24 a resumed run records a different frame cadence | **closed (#281)**, 口径 confirmed by the ruling | The same cell, the same two held bonds and the same timestep: one process of 8 steps records **2** frames (time column `0.00 / 86.80`), the same run split 4+4 records **6** (`0.00 / 37.20 / 37.20 / 49.60 / 62.00 / 74.40`) - one frame per step after the seam, and the seam step twice. The constraint criterion passes on both halves (bond spread `1.839e-11` A continuous, `2.150e-11` A in two processes), but #274's frame-by-frame identity criterion cannot be applied to a periodic run until the frame lists agree, and the repeated `37.20` is D-19b reproducing on the cell engine. Cause: one variable with two jobs. `md.py:439-441` takes `requested = runtime.total_steps`, which is **0** in a checkpoint, so `total_steps` becomes `self.maxiter`, i.e. the *remaining* iterations; `:686` then uses that same number as an *absolute* end (`final = step >= self._total_steps`) while `step` continues 4,5,6,7. So `final` holds on every remaining step and each of them writes a frame. A candidate was excluded by experiment rather than argument: unpickling a real `prestart.rt` and rewriting its `total_steps` from 0 to 8 changed nothing (both arms produced the identical six values), which rules out the checkpoint field and leaves `maxiter`-as-end-index. The fix is the renumbering plus one rule - **a step that writes a checkpoint also writes a frame**, both decided by the single predicate `_runtime_file_due`, so the frame set is a function of the absolute step and the checkpoint cadence and nothing else. Measured on the pair with the same cadence: both arms write frames `[1,2,4,6,8]` with the same time column, and every shared frame agrees to **`0.000e+00` A over all 60 atoms**, the energies columns equal row by row (the 2026-10-01 D-19/D-24 comparison output, deleted; §33); the aperiodic PySCF pair does the same over 8 frames (`0.000e+00` over 3 atoms, energy rows equal). The comparator was controlled: two different steps of one arm differ by `2.564e-02` A (periodic) and `1.354e-03` A (aperiodic), so the zeros are a measurement and not a bench that cannot fail |
| D-25 a moving cell took the strict tolerance away | closed (#276, attributed and measured) | NPT with an anisotropic Berendsen barostat moves and shears the recorded lattice by up to `0.025420` A per entry across three frames, and the same `oh_bonds` block that holds to `8.131e-11` A in a fixed cell (the NVT sibling) holds only to **`4.806e-05` / `7.649e-05` A** in the moving one - five orders against the declared `1.0e-10`. The framework neither refuses the combination nor meets what it declares. The measurement instrument was controlled first: the bonds read `0.981527 / 0.981539 / 0.981479` A whether the minimum image is searched with each frame's own lattice, with the frame-0 lattice, or with the next frame's, so the residual is not a lattice-choice artifact; and the affine hypothesis fails per interval rather than in total: the recorded a-axis steps of `+0.002105` / `-0.000733` A would drag the bond by `+2.46e-04` / `-8.57e-05` A against the measured `+1.23e-05` / `-6.04e-05` A - `5%` and `70%` of affine, so neither a full ride nor a constant lag fits, and what is left is a projection that runs without compensating the cell's motion inside the interval. The cause is then a sentence in the framework's own code plus a line in the loop: `integrator/ensemble.py::_berendsen` documents that it "rescales the coordinates and the cell" and is "applied at post", while `md.py` called `apply_constrain` before that post half step - so every recorded step carried the rescale of a geometry the solver had already fixed, and LAMMPS is not involved because the md profile produces no `fin_cell`. One frozen atom showed the same thing as the bonds (`8.108e-04` A of drift under NPT against `0.000e+00` under NVE), which is how the seam, rather than the bond arithmetic, was suspected. The fix is one more geometry question after the rescale (the `md.py` phase `constrain_after_cell`): the same block now holds to `1.71178627e-11` / `4.72044626e-12` A in the fast arm and `1.14519505e-12` / `3.75688369e-11` A in the slow one, the frozen atom returns to `0.000e+00` A, and the fixed-cell arms are unchanged (`8.13138e-11` → `8.13127e-11` A). The negative control was run rather than argued: moving the new block back above the post half step reproduces `4.80572812e-05` / `7.64921032e-05` A bit for bit, so the causal chain closes on the ordering. Cost is 0.999 ms per step in a 166.229 ms engine-bound step (0.60%) and 0.001 ms for a run with no plan, since the entry still returns the same object |
| D-26 the convergence test counts directions the step may not move | **closed (#282)**, 口径 A as ruled | A constrained optimization masks the frozen components of the *displacement* (`optimizer/base.py:168` via `apply_constraint`), which is why the declared bonds hold to `1e-10` above. The **gradient** metrics were not masked: `base.py:213-217` took `grad_rms`/`grad_max` over the whole raw gradient and `:247-251` compared those against `opt_grms`/`opt_gmax`. The last-step `max_g` separated on exactly one factor of the 2x2: without `fixes` `8.3668e-02` (`b_only`) and `8.4217e-02` (`b_bias`), with `fixes` `2.0797e-01` (`b_fix`) and `1.9529e-01` (`b_all`) - the bias makes no difference, the frozen atom roughly doubles it, because its residual force is counted in a stopping criterion the step is forbidden to answer. Fixed by `constraints/projection.declared_mask` (freeze mask ∪ every atom named by a strict condition's rows, read through `state_adapter._condition_rows`, the same source the solver uses) and asking the gradient metrics only outside it. Measured on the product that raised it, at the same geometry and the same step of `constraint_opt_surface`: `Gradient RMS` **`0.04358507` → `0.00873618`**, `Gradient Max` **`0.08206846` → `0.01490334`**, while `Energy` (`-74.95614281`) and both displacement columns are unchanged to the printed digit, so the difference is attributable to the removed components alone. Boundaries are pinned by `TestConvergenceGradientCriterion`: an empty plan reproduces the old all-components arithmetic exactly, and a run with nothing free reads **0.0 rather than nan**. **Superseded on 2026-10-10** (see section 35): the mask removed whole atoms, and with them the tangential half of those atoms, so a run could be declared converged over the largest force in its structure. The criterion is now directional (tangential part judged, absorbed part printed as the multiplier) and `projection.declared_mask` is gone with its only consumer. The two figures quoted above (`0.04358507 → 0.00873618`, `0.08206846 → 0.01490334`) are readings of the withdrawn mask and must not be cited as the current criterion's behaviour |
| D-27 two constraint engines can own one trajectory, unrefused | **closed (#276 implementation, #284 default)** | `md.py` propagates and constrains when the profile is framework-driven; the skill's own SHAKE switch is already restricted to `ENGINE_INTEGRATED_PROFILES` (`skills/lammps_skill/schema/method.py:78`), which is the direction that *is* gated. The reverse was not: declaring `profile: ff_nvt` together with a framework `constraints` block ran to completion with no refusal or warning, the engine log reporting `Finding SHAKE clusters ... 20 = # of frozen angles` while the framework's `apply_constrain` ran 1.331 ms per step and the record kept the framework's targets `0.981532` / `1.003004` A. Implemented as the ruling put it - ownership by `control.driver`: Layer 0 of `schema/adapter.py` hands every skill the driver, `data/core.py::constraint_plan` drops the block and warns when the driver is not `main`, `skills/lammps_skill/schema/validate.py` clears the skill's own constraint keywords and warns when it is `main`, and the run log prints who owns the geometry (`ownership framework (control.driver='main')`). #284 then corrected the **direction of the default**: it was `main`, which silently stripped the shake keywords from every `&lammps` input built without a framework; it is now `skill`, and only a driver the framework states takes ownership away. Verified live on all three runs of the restart pair (one warning each) and pinned by `TestConstraintOwnership` |
| D-28 the bond-length table could not see an organic single bond | found and closed (#285) | `utilities/bonds.py::bond_lib` answered `C-C` with `1.20` and `C-O` with `1.16` - the shortest members of those series - while periodic perception tests `distance < bond_lib * 1.2`, so a relaxed ethanol's `1.526` Å C-C and `1.407` Å C-O fell outside their own reach and the torsion across them was refused as naming "atoms [1, 2, 3, 4], which lie in 3 separate connected components", on a molecule that is one molecule. The table's own docstring promises *typical equilibrium* distances, so the heavy-atom entries now carry the single-bond values (`C-C 1.53`, `C-N 1.47`, `C-O 1.43`, `N-N 1.45`, `N-O 1.40`, `O-O 1.48`). The change only widens a reach, never narrows one, and that is measured rather than argued: the shipped 60-atom TIP3P box still perceives **40 bonds, all O-H, none other** (the number the ported perception was calibrated on), because its closest O···O and H···H contacts are `2.704` and `1.519` Å against reaches of `1.776` and `0.888` Å; and the ethanol box now perceives exactly the **8 bonds** its LAMMPS topology declares. Full suite with the new table: **3427 passed / 6 skipped / 916 subtests** |
| D-29 the writer silently dropped every bond row but the first | found and closed (#287) | `write_constraint_section` is documented as refusing rather than approximating, and `_constraint_write_object_value` broke that promise for a strict constraint: `rows = value[0] if isinstance(value[0], (list, tuple)) else value` took the **first row only**, so a record declaring the water's two O-H bonds (`bonds: [[1, 2], [1, 3]]`) wrote `shake_bond 1 2` and read back as a run that holds one bond. Nothing in the suite could see it, because no shipped input had ever been written through that path with two rows in one record. Two changes: the flat grammar states several rows of one object as one segment of arity*k* tokens (`shake_bond 1 2 1 3` is one named block of two bonds, `shake_bond 1 2; 1 3` is two blocks of one), which the reader cuts on the width the keyword owes and refuses when the count is not a multiple; and the writer joins every row. Found by the round trip of section 32, which is the only instrument that can see it: the criterion compares the plan fingerprint of the block before the rewrite with the plan of the keywords after, and this draft failed it with `pairs [[0,1],[0,2]] != [[0,1]]` |
| D-30 the written `.input` truncated every number to six digits | found and closed (#287) | the writer formatted numbers with `%g`, so a wall center stored at `4.199546` Å came out as `4.19955` and an rmsd reference block lost its last three digits per coordinate: the file the run wrote for itself was a **five-parts-per-million different run**. The same `%g` sites also composed the ramp endpoints of a component ramp. One formatter now (`_constraint_number`, `repr` of the float, the shortest text that re-reads as the same double), used by object lines, parameter lines, ramp endpoints and scale factors. The criterion is the same fingerprint round trip: six drafts failed it before the change (the wall family and the rmsd history) and none after |
| D-31 a record field the model does not have raised a KeyError in the writer | found and closed (#287) | `_constraint_is_default` asked `model_fields[field]` for whatever key the block carried, so a declaration still written in a retired spelling (`reference:` on a constraint) died as `KeyError: 'reference'` from inside the writer instead of being refused. The lookup now checks the field list and refuses by name, saying that neither the record model nor the flat grammar has that key. This is the failure mode of a *helper* reached from a diagnostic path, which is why it is on the register: the same input through the reader is refused correctly, so only the emit-a-.input side showed it |
| Analytical curvature: bias and wall | closed | `geometric.bond_hessian_core`, `angle_hessian_core`, `torsion_hessian_core`, `cv_hessian_block`, `bias.hessian`, `wall._curvature_terms`, `wall.hessian`, `controller.hessian_block`. The torsion is written as a bilinear directional form - the bond vectors are linear in the coordinates, so every `d2b` vanishes and what is left is slot insertion - which makes symmetry and translational invariance structural rather than a cancellation in a long expression (measured residuals 1.8e-15 and 2.3e-15; the finite-difference gap follows `2.7 * step^2` down to the rounding floor). Walls: four geometries under an atom, center-of-geometry and center-of-mass target at p=2 and p=3, agreed to 1e-6 relative. Two coefficient errors were caught by that yardstick: an ellipsoid projector missing its `1/rho`, and the slope and curvature of `V = k d^p` attached to each other's term. |
| Analytical curvature: history potential | closed, after one wrong power of `sigma` | `metadynamics.hessian` assembles `H = sum_jk V_jk J_j J_k^T + sum_j V_j Q_j` over the declared variables. The defect that made an earlier draft refuse this row was algebraic, not structural: with `z_j = (s_j - c_j)/sigma_j` dimensionless, one differentiation brings down `1/sigma_j`, so `dV/ds_j = -V z_j/sigma_j` and `d2V/ds_jdk = V (z_j z_k/(sigma_j sigma_k) - delta_jk/sigma_j^2)`; writing `1/sigma^2` in the first factor agrees at `z = 0` - which is exactly where a hill deposited at the current value sits - and diverges everywhere else, by a `z`-dependent amount (ratios 0.46 to 8.7 on one fixture), which is why it first read as a broken multi-variable coupling. |
| Analytical curvature: aligned-RMSD history | refused by name | its optimal rotation moves with the geometry, so the curvature is not this sum; `cv_hessian_block` still returns None for an unknown variable kind and the callers refuse rather than treat a missing term as a vanishing one |
| `molecule` grouping, `region: inside/outside`, one-sided torsion region, open-planar walls, constraint virial | open, blocked on a convention or an upstream driver | each refuses by name at input; see 附录 B section 10 bucket three. Note that `region: inside/outside` here is a **bias** region (which side of a collective variable a harmonic restraint acts on), not atom selection: optimizing only a selected region is settled and needs no convention, because it is the complement of a `fix` (`dev/methods/OPTIMIZE_DEVELOPMENT.md` section 14, ruled 2026-10-13) |

## 19. Merge Route (executed 2026-09-29)

The user authorized the merge on 2026-09-29: *"将约束功能接入 pyrai3md 主框架，要准确记录合并路线，保证可回溯"*. This section is that record. It is written as a checklist that was filled in as each step ran, so a later reader can reconstruct not only where every file went but in what order and with what evidence.

### 19.0 Reproducibility anchor

| Item | Value |
| --- | --- |
| Pre-move snapshot | a local tarball from 2026-09-29 (since deleted), sha256 prefix `d9ccc1a7d9c76c55` |
| Move driver | the `File map` table of `constraint_lab/README.md` (the same table `test_integration_manifest.py` had been guarding). The move script read that table and refused to move anything unless every row resolved to an existing source and a **free** target |
| Prefix rewrite | `pyrai3md.constraint_lab.` -> `pyrai3md.` in 33 files; post-move grep for `constraint_lab` in any framework `.py`: **0 hits** |
| Suite at the move | 565 tests in the new locations (651 minus the three shadow-build guard files retired below). End state after the wiring: the counts live in the status header (669 constraint tests, 104 in `utilities/tests/`, repository-wide 2497 OK / 6 skipped), on the repository's `.venv` |

Retired with the move, because their premise expired rather than because they failed:

| File | Why it stops existing | Where its rule went |
| --- | --- | --- |
| `tests/test_import_discipline.py` | it forbade the lab importing framework modules; the lab is now the framework | the surviving narrower rule (the state adapter must not import the state class) lives in `test_constraint_state_adapter.py::TestNoFrameworkImports` |
| `tests/test_integration_manifest.py` | it guarded a file map whose purpose was the move; the move is done | the register id cross-check is this section's table of D-xx dispositions |
| `tests/test_geometric_defect_evidence.py` | **deleted, not repaired**, exactly as D-01's proof column required once `geometric.DHD` is fixed and `cv.py` stops being a second torsion | its central-difference, Noether and cis/trans/gauche pins move to `qualification/verification/test_geometric.py` |

### 19.1 File moves (all 17 library modules, from the documented map)

`utilities/{selection,cv,rmsd}.py`, `data/{constraint,topology}.py`, `schema/{constraint,legacy_adapter}.py`, `interface/constraint_parser.py`, `workflow/constraints/{schedule,bias,wall,legacy_cavity,projection,shake,metadynamics,controller,state_adapter}.py` plus the new `workflow/constraints/__init__.py`, each to the same-named path under `pyrai3md/`. `setup.py` uses `find_packages()` (verified: line 39), so `workflow/constraints` needs no packaging edit, and the lab carried no resource files, so `package_data` is untouched.

### 19.2 Test placement (20 files, renamed to the §15 discovery pattern)

All to `pyrai3md/qualification/verification/`, which the qualification planner globs:
`test_constraint_{parser,controller,shapes,cv,cv_cell,records,legacy_equivalence,metadynamics,potentials,projection,rmsd,rmsd_history,schedule,schema,selection,shake,shake_dynamics_cost,shake_velocity,state_adapter,wiring_rehearsal}.py`, plus `test_constraint_end_to_end.py` and `test_constraint_integration.py`. (`test_constraint_legacy_adapter.py` went with #271 and `test_constraint_legacy_cavity.py` with #273; the file roster above is the one the tree carries now.)
The §15 groups map onto these names: *parser* -> `test_constraint_parser`; *schema* -> `test_constraint_schema` + `test_constraint_selection`; *cv* -> `test_constraint_cv` + `test_constraint_cv_cell`; *potential* -> `test_constraint_potentials` (whose `TestCenterLaw` class is what the retired cavity cases became); *composition* -> `test_constraint_shapes` + `test_constraint_controller`; *projection* -> `test_constraint_projection` + `test_constraint_shake` + `test_constraint_shake_velocity`; *statistics* -> inside `test_constraint_projection`; *restart* -> `test_constraint_records` plus the new wiring tests; *legacy* -> `test_constraint_legacy_{adapter,cavity,equivalence}`; *integration* and *workflow* -> `test_constraint_wiring_rehearsal` and the new end-to-end cases of §19.4. `test_constraint_shake_dynamics_cost.py` stays marked by `PYRAI3MD_SKIP_SLOW=1` because it is a dynamics run, not a unit test.

### 19.3 Wiring checklist (each row is one framework edit; ticked when it runs)

| # | Framework edit, by symbol | Status |
| --- | --- | --- |
| W-1 | `interface/parser.py::_parse_text` dispatch branch for `&system_constraint`; `&end` closes only the outer block | done, and `&end` no longer opens a section named `end` at all |
| W-2 | `schema/config.py::System.constraints` field + a dedicated nested-section route (`system_constraint` -> `system.constraints`, *not* a merge of its inner keys into `System`) | done - `Optional[SystemConstraintConfig] = None`, and `None` is the default-closed signal |
| W-3 | `schema/adapter.py`: `SECTION_TYPES`, `_partition_fields` routing, type-map registration so the block survives flattening untouched | done - `SECTION_TYPES['system_constraint']` with an empty type map, and a dedicated route in `_partition_fields` and `_flatten_sections` |
| W-4 | `schema/normalize.py::normalize` step `normalize_constraints` between `normalize_skill_name` and `_partition_to_submodels`; index conversion exactly once | done - `normalize_constraints` plus the classic-field migration, both writing `data['constraints']` back |
| W-5 | `schema/validate.py`: capability and combination refusals in the local refusal style | partial - the refusals fire, but through `schema.constraint` rather than a new `validate.py` branch; see section 20 |
| W-6 | `data/core.py::PyRAIState`: read-only plan/state accessors and recursive ndarray conversion at `set_result` (D-08) | done - `constraint_plan()`, `constraint_state`, and the recursive `_to_plain` at `set_result` (D-08) |
| W-7 | `workflow/calculator.py::run_in_process`: finalizer after `convert_stress_to_atomic` and before `set_result`, gated on an empty plan, writing flat `energy_contributions` (D-12) | done - `Calculator._finalize_constraints`, called on both the single-task and the in-process path |
| W-8 | `workflow/integrator/operations.py::apply_constrain`: freeze masking, SHAKE positions, velocity zeroing and the `rattle` branch (D-07) | done - delegates to `constraints/runtime.constrain_state` |
| W-9 | `data/runtime/serializer.py`: plan and state into the payload, fingerprint refusal on a mismatched restart | done - `constraint` payload block with semantic version and fingerprint, history rebuilt from the payload |
| W-10 | `workflow/integrator/ensemble.py::_dof`: rank-based count replacing `3*(natom-n_frozen)-3` and the clamp (D-06, own changelog entry) | done on the constrained path (rank-based, never clamped); the `max(...,1)` clamp stays on the unconstrained path, stated in section 20 |
| W-11 | `utilities/geometric.py::{DHD,BND,AGL}`: gradient fixed once, nan replaced by an explicit failure, `57.2958` replaced by `units.py` (D-01, D-02, D-03), and the duplicate torsion in `utilities/cv.py` deleted | done - the atan2-invariant form moved into the shared module, `cv.py` now delegates, the defect-characterization file is deleted, and the pins live in `qualification/verification/test_geometric.py` |
| W-12 | `workflow/integrator/operations.py::kabsch` (now `data/geometry.py::kabsch`): center and mass-weight, correct the over-claiming docstring, delete the duplicate alignment (D-10) | done - the direction went **upward-consumes-downward**: `utilities/rmsd.py` is a leaf and already held the correct weighted superposition, so `kabsch` became a scalar wrapper over `rmsd_squared` rather than the reverse; one cross-covariance SVD remains in the package, and `test_constraint_kabsch.py` pins it against an independently written Horn quaternion superposition |
| W-13 | the boundary statement between the two "schedule" words (D-13) | done in both docstrings. The shared center helper (D-04) is **not** done: see section 18.1 |
| W-15 | C1-6: `optimizer/base.py::apply_constraint` was a docstring of TODOs with no caller | done - it now projects the frozen components out of the trial step *after* the trust radius and *before* the step is recorded, `optimize.py` binds the reference before its loop and calls the position solve after each step, and the converged run evaluates its final structure once |
| W-14 | `skills/pyscf_skill/schema/method.py`: `from pyscf.prop import ...` ran unconditionally at initialization, so every PySCF job failed with "No module named 'pyscf.prop'" on a wheel that does not ship the optional properties extension | done - the import is optional now, and only the NMR-shielding path reports its absence, at the call that needs it |
| W-16 | `workflow/md.py::checkpoint` carried `# TODO: write the PyRAIState serialization file.`: without a written file there is nothing for §12's restart contract to resume from, so the constraint history could never be exercised through the CLI | done - `_write_runtime_file` writes `<title>.rt` on the `md.checkpoint` cadence and once at the end of the run (a `checkpoint` of 0 still writes nothing), and the payload no longer carries the entry point's `RunLogger` file handle (defect 9 above) |

### 19.4 Two corrections this merge found in its own document

- `DEFAULT` lives in `schema/config.py:29`, not in `reference.py`; `reference.py` holds the periodic table, masses and unit constants. The four registration points are therefore `config.py` (`DEFAULT`, `System`, `SUBMODEL_CLASSES`, `NESTED_FRAMEWORK_SECTIONS`, `PyRAIConfig`) and `adapter.py` (`SECTION_TYPES`, `_partition_fields`, `_convert_types`, `_raw_user_keys`).
- There is no `System` mirror class in `data/core.py`. The flat legacy constraint fields enumerated by the adapter are `schema/config.py::System` members, so the legacy adapter's input is the config model, not a runtime dataclass.

### 19.5 What the merge found, in the order it found it

Nine things the shadow build could not have told me, all of them now fixed and all
of them visible only once the code was inside the framework:

1. **The installed package is a copy, not the source tree.** `python3 -m pyrai3md`
   with the system interpreter imports a stale user-site install of the same
   package (Python 3.9), which
   does not contain this merge. My first "real run" of the day executed the old
   code and reported an energy with no constraint row; the row appeared only
   after I pointed `PYTHONPATH` at the working tree. Every number in this section
   is from the working tree, and the environment that should be used from now on
   is the repository's `.venv` (python 3.11, PySCF 2.14, LAMMPS
   with MPI), which imports the source tree in place.
2. **`_parse_text` had no section closer.** A literal `&end` opened a section
   named `end` and filed every following key there, which section 3.2 recorded as
   the reason the constraint block needed its own parser. The flat loop now stops
   owning keys at `&end`; the phantom section is gone, and a stray key after a
   closer is dropped instead of being attributed to a section that was never
   written.
3. **The classic flat constraint fields are only read where the classic input
   put them.** `cavity`, `factor`, `scale`, `shape` and `center_type` written at
   the top level of a YAML are overwritten by the merged environment defaults;
   under `molecule:` they survive. The migration therefore reads them from the
   config model, and the draft case file writes them the way a real classic input
   does. This is an input-routing fact about the existing framework, not about
   constraints, and it is why the migrated values were checked end to end rather
   than only against the adapter.
4. **A derived cache must not survive a restore.** `PyRAIState` caches the
   constraint plan, pickle carries instance attributes, so a resumed run restored
   from a file whose input had been edited would have kept the *old* cached plan
   and the fingerprint check would have compared the old plan with itself. The
   load path now drops the cache before it derives, which is the only way that
   check can see a changed input.
5. **The degrees-of-freedom count had to keep the framework's center-of-mass
   convention.** `statistics.free_dof` is the mobile-component count minus the
   constraint rank; the thermal count the ensemble expects is that minus three.
   Returning the raw figure would have quietly raised every constrained run's
   temperature by a factor of `ndof / (ndof - 3)`. The rank-based count is the
   fix D-06 asked for; the subtraction is the convention it must respect.
6. **A `for` loop can afford an early `continue`; a `while` loop cannot.** Skipping the consumed
constraint block meant indexing, so the flat loop became `while index < total` - and its
`if len(tokens) < 2: continue` branch then never advanced the counter. A classic input with one
valueless key hung forever. The symptom was not a wrong parse but a suite that never finished, which
is why the guard is written as a termination test (`test_a_key_without_a_value_is_skipped_not_looped`)
rather than as an assertion about the returned mapping. This defect was introduced by this merge and
found by the repository-wide run, not by any test written before it.
7. **A trace-form RMSD cannot report exact identity.** Written as `tr(P-P)`-style
sums of nearly equal magnitudes, `rmsd_squared` returned about 3e-8 for two *identical*
blocks instead of 0 - the first question a caller asks of a superposition, and the one
that decides whether a convergence test fires on the first iteration. Identical blocks
now return exactly 0.0 with a zero gradient; the trace form keeps the general case with
its non-negative clamp. This defect predates constraints in the sense that the function
did, but nothing in the repository compared a structure with itself until the constraint
subsystem needed to.
8. **The thermostat must refuse, not clamp.** With constraints honoured, a
   fully frozen or fully constrained system is reachable for the first time. It
   has zero thermal degrees of freedom, and `max(ndof, 1)` would report a
   temperature for a system that has none. On the constrained path the count
   raises instead. The clamp still stands on the unconstrained path, because
   removing it there changes behaviour this merge does not own - that half of
   D-06 is therefore recorded as deliberately partial rather than silently skipped.
9. **A live state is not picklable, because it is also a printing state.** `pmd.py`
attaches the `RunLogger` to `state.logger`, and that object holds the open `<title>.log`
file handle. The first `md.checkpoint` tick therefore died inside `pickle.dump` with
`TypeError: cannot pickle '_io.TextIOWrapper' object`, and the half-written `.rt` made the
resuming process fail with `EOFError: Ran out of input` - two errors, one defect, and the
second one hides the first. `_prepare_state` had already been dropping the transient
`_staged_evaluation` slot for the same reason (a checkpoint stores committed state, not
whatever the writing process happened to be mid-way through); the logger now goes the same
way, and the restored state comes back with no logger at all because the resuming process
opens its own log file. This is the one defect in this list that a unit test of the
serializer could never have produced on its own: it needs a state built by the entry point,
so `test_a_state_holding_the_run_logger_still_writes_a_file` builds a real `RunLogger` and
the negative control (removing the drop) reproduces the production `TypeError` exactly.

## 20. Functional Cases Run Against Real Engines

> Where these inputs live now: they were drafted in `draft_constraint_examples/`
> and folded into the official example set, so today they ship in
> `pyrai3md/examples/` - `TestShippedConstraintExamples` discovers them by
> their `system_constraint`/`constraint` section rather than by filename, so
> no count is pinned here. The dated sentences below keep the draft
> path because that is the directory each measurement was made in; §33 records
> the retirement of the evidence and draft directories.

Ten draft inputs, one per declared capability, in `draft_constraint_examples/`
(the user folds them into the official example campaign in the next phase). Nine
of them ran to completion on PySCF 2.14 / HF / STO-3G water in the local venv,
and the table records the number each case was written to produce, read off the
framework's own products - `<title>.md.xyz` and `<title>.log` - not off a test
fixture. The first seven are in the table; the eighth (`constraint_opt_surface`,
constrained optimization) and the ninth (`constraint_md_restart`, checkpoint plus
resume) are measured in the paragraphs after it.

| Case | What it declares | Evidence from the run |
| --- | --- | --- |
| `constraint_sp_bias` | bond CV + harmonic bias + sphere wall, one gradient | two rows in `&energy contributions`; total = unconstrained electronic energy + the printed bias row, to 1e-10 |
| `constraint_md_shake` | both O-H bonds held at their initial length, 5 steps | bond length spread over the trajectory 6.5e-12 and 3.3e-16 Angstrom, while the oxygen still moves 8.8e-4 |
| `constraint_md_rattle` | the same with `velocity_constraint: rattle` | spread 1.7e-12 and 9.1e-11 Angstrom on the identical position path |
| `constraint_md_freeze` | the oxygen frozen | its position is unchanged to `0.0e+00` in every frame, while both hydrogens move 2.9e-2 |
| `constraint_md_bias` | a bias holding one O-H at 0.90 Angstrom | a `constraint.bias.hold_oh` row in all five steps, growing as the bond stretches |
| `constraint_md_metad` | Gaussian history along the O-H CV, `pace: 2` | a `constraint.metadynamics.explore` row every step, arriving from the deposited hill |
| `constraint_legacy_cavity` | only the classic flat fields, no block | `constraint.wall.legacy_cavity` rows, i.e. the migration produced a real `legacy_polynomial` action |

The eighth case, `constraint_opt_surface`, is the constrained optimization C1-6 asks for: an
oxygen frozen, one O-H held by SHAKE and a bias pulling the other towards 0.95 Angstrom, six GDIIS
steps on HF/STO-3G. Read off its own products, the frozen atom's position is unchanged to
**0.0e+00** across all seven written structures, the constrained bond's spread over them is
**7.0e-11 Angstrom**, and the biased bond moves **0.985 -> 0.955 Angstrom** towards the declared
target - the constraint force is what the optimizer is following, not a term the run ignores. That
case also exposed a pairing hazard the loop had before this merge and that constraints make visible:
the iteration order is evaluate, move, project, so the structure written last was one the run had
never evaluated. A constrained optimization now performs one evaluation after its last projection,
which is why the log carries seven energy blocks for six steps; an unconstrained optimization gains
nothing - the extra evaluation is gated on the plan not being empty, and `test_integrator` plus the
existing optimization tests are unchanged.

The tenth case is the periodic one, and it is the case section 5.2 used to make impossible:
`draft_constraint_examples/constraint_md_periodic_shake.yaml` runs the 60-atom TIP3P box of
`md_lammps_nve.yaml` (8.399 Angstrom cell, `pbc: [1, 1, 1]`, LAMMPS as the force provider with its
own SHAKE switched off) with water 1's two O-H lengths and its H-O-H angle held by the framework
solver, 600 steps, 7 recorded frames. Read off its own products (the 2026-09-30 periodic evidence
set, deleted on 2026-10-02; §33):

- the targets bound at the reference frame are 0.981532, 1.003004 Angstrom and 104.183450 degrees,
  and over the seven frames the largest deviations are **9.117e-11**, **2.473e-11 Angstrom** and
  **3.239e-09 degrees** (declared tolerance 1e-10 relative);
- frames 3, 4 and 6 have that molecule *torn across a cell face*, and the held length is still the
  continuous one - which is the unfolding doing its job, because the same pair measured off the
  stored coordinates reads close to 8 Angstrom;
- locality: an unconstrained water in the same box moves normally (O-H 0.954922 to 0.952726
  Angstrom, angle 110.766 to 99.285 degrees), so the correction lands only where it was declared;
- the energy half is attributed by A/B on the same box rather than asserted. Unconstrained: `Etot`
  drift **5.1e-5 Hartree**. Position-only constraint: **0.090863 Hartree** with `Ekin` rising
  0.084976 to 0.142765. Bonds with `velocity_constraint: rattle`: **0.000320 Hartree**, and the
  bonds are still held to 8.24e-11 Angstrom. The middle row is the cost section 1.3 records for a
  position-only projection, seen here in a real engine run at 1.8e3 times the baseline and 284 times
  the projected case; it is not a regression this round introduced, but it is now measurable rather
  than predicted;
- a locally generated angle-plus-`rattle` case (since deleted, like the other drafts) exits with
  the named refusal,
  because the velocity half of an angle or torsion is not built: the run is not allowed to report a
  RATTLE while projecting only the bonds.

Two things this table does not claim. **The periodic case is not in the official example campaign**:
shipping it under `pyrai3md/examples/` hits the hardcoded LAMMPS example count in
`skills/lammps_skill/tests/test_persistent_test_plan.py` and the remote-targeted
`qualification/pmd_env.yaml` switches, both of which are the user's call, so it sits with the other
draft constraint cases. What *was* verified is that the merge did not disturb the
LAMMPS provider chain: `pytest pyrai3md/skills/lammps_skill/tests pyrai3md/utilities/tests` gives
**555 passed** on the local venv (its own LAMMPS build, `lammps_path` supplied through the skill
config rather than by editing the remote-targeted `qualification/pmd_env.yaml`, which stays the
user's call). And **planner discovery is evidenced
indirectly**: the files sit in `qualification/verification/`, which is the only directory the
qualification runner globs, and the repository-wide `unittest discover -s pyrai3md` counts them
(2497 including 669 `test_constraint*`); the campaign itself was not launched, because that would
submit remote jobs.

The ninth case is the one section 12 asks for and the only one that needs two processes.
`constraint_md_restart.yaml` runs four steps with `md.checkpoint: 2` and both O-H bonds held
at their initial length; the run writes `constraint_md_restart.rt`, and
`pyrai3md constraint_md_restart.rt` resumes it for four more steps. Measured on the two
trajectory files the runs leave in their own directories (the resuming process writes its
own `<title>.md.xyz`, so the two halves were run side by side rather than appended):

- the target is `0.958166865217` Angstrom for both bonds, taken from the *first* structure of
  the first run;
- across all eight frames in the two processes the largest deviation from that target is
  **6.5e-12 Angstrom** (first run: 0, 1.7e-12, 4.5e-12, 6.0e-12; resumed run: 6.5e-12,
  6.1e-12, 5.3e-12, 4.3e-12). Every resumed frame is measured against the baseline the file
  carries, not against the geometry the file resumed at;
- the resume is a continuation, not a restart: the atom-wise distance between the last frame
  of the first run and the first frame of the resumed run is 4e-4 to 3.9e-3 Angstrom, one
  integration step, and the engine header of the resumed process echoes the moved coordinates
  (O at z 0.119229 rather than the input's 0.117340), which is the stored state reaching the
  backend rather than the input being re-read.

This is the case that exposed defect 9 above, so the two halves of it are covered at two
levels on purpose: the CLI run above shows the seam works end to end, and
`test_a_state_holding_the_run_logger_still_writes_a_file` shows why the file could not be
written at all before the logger was dropped from the payload.

The checkpoint writer got its own unconstrained control, because DC-1 and DC-3 are now claims
about a file that is written inside the step loop. The same input, run twice - once with no
`md.checkpoint` and once with `checkpoint: 2` - produces a `.md.energies` that is **byte
identical** and a `.md.xyz` whose coordinate blocks are identical (only the title on the comment
line differs, which is the two runs' names, not their physics); the payload of the file the
second run writes is `{'constraint': {}, 'runtime_idx': ..., 'state': ..., 'version': ...}`, and
resuming that file exits 0 with no `constraint.` row in any of the three logs. So the cadence
costs nothing it does not declare, and an empty plan allocates nothing in the file.

A note on why the CLI numbers alone cannot settle the retargeting question. A position
constraint holds the bond to its solver tolerance, so a resumed structure is never far from
the target it carries - 6.5e-12 Angstrom here - and a run cannot distinguish "target from the
file" from "target re-read off the resumed geometry" by distance alone. The decisive version is
`test_a_restarted_run_does_not_retarget_its_own_drift`: it saves a bound state, loads it, moves
the restored coordinates 0.25 Angstrom away and re-binds, then asserts the targets are still the
stored ones. The CLI case proves the file reaches a real run; that test proves which value the
run obeys.

Restart, refusal and DOF behaviour are covered by
`qualification/verification/test_constraint_integration.py` (26 tests), because
those need a state built to order rather than a calculation: the checkpoint is
refused when the block is edited between save and load, a file carrying a
history with no declared block is refused, the same bond pair given two targets
is refused, and the DOF arithmetic is pinned at four atoms with a chain of three
bonds and with a bond whose two ends are both frozen.


## 21. Fourth round: the flat text format, and the schema modules dissolved (2026-10-01, #271)

Two restructurings landed together because they are the same statement: the constraint subsystem
is not a separate module that the framework visits, it is part of the framework's input and schema
layers.

| Item | State | Where it went, and why there |
| --- | --- | --- |
| The block DSL (`&system_constraint` with nested `wall sphere` records) | dropped, replaced by flat `&constraint` | §4.2 states the reason: nesting is free in YAML and JSON and only cost in text, and the shape it forced a second parser on `interface/parser.py`. The old name still routes: `system_constraint` is kept in `NESTED_SECTION_FIELDS` so an archived input runs rather than failing on a rename |
| The classic `&constraint` reader | `schema/adapter.py::_read_constraint_section` | the interface layer's duty is splitting lines; `&molecule` is interpreted in the adapter, so a section that needs `;` semantics belongs beside `_parse_center_indices`, which is the same grammar one family older. `interface/constraint_parser.py` is deleted and `parser.py` keeps only its `&end` fix |
| The writer, for `.input` generation | `schema/adapter.py::write_constraint_section` | one grammar, two directions, one file. The certification converter calls it; a block the flat text cannot state (an id on only some objects, a shared schedule list, molecules of unequal size in one `wall_groups`) is a conversion error, not a silently dropped declaration. Where a positional line would need a gap, the writer emits `none` (§4.3) |
| `SystemConstraintConfig` and its record models | `schema/config.py`, above `System` | the user's instruction: the constraint keywords belong to `class system`. `System.constraints` lost its forward reference and the bottom-of-file `model_rebuild()`, because the class is now defined before it |
| `normalize_*` and `build_plan` | `schema/normalize.py` | the pipeline step of that name already lived there; the record builder that used to be called `normalize_constraints` is now `normalize_hard_constraints`, which is also the name that says what it builds |
| `_check_periodic`, `_check_ramp_shape`, `_check_ramp_positive` | `schema/validate.py`, public | refusals are the validation layer's vocabulary. They are public because `normalize.py` imports them: the plan build calls the same functions, so a rule cannot mean two things depending on which pass reached it first |
| `schema/legacy_adapter.py` | merged into `schema/adapter.py` | the nineteen flat keys are another classic-input spelling, which is what that file owns. Its docstring says what it is for and that new inputs should not use it |
| `md.step` as the default ramp stop | `build_plan(md_step=)`, fed by `schema/normalize.py` (`data.step`) and `data/core.py::constraint_plan` | the reader cannot see the length of the run; the schema can, and it writes the source into `notes` |
| Ramped walls in a periodic cell | `workflow/constraints/schedule.py::endpoint_variants` + `state_adapter._wall_geometry_variants` | `delay`/`stop` bound when a value is reached, not which values exist, so the four cell checks run against every endpoint the ramp produces |
| `notes` | `SystemConstraintConfig.notes` to `ConstraintPlan.notes`, printed by `kernel/logger.py::_write_constraint_plan` | a broadcast, a padded line, an ignored wall keyword and a stop taken from the run are decisions the input did not state in numbers; they belong in the record of the run. `notes` stays out of `to_dict`, so a reworded note cannot break a restart |

Refusals kept from the first version, unchanged in substance: the block DSL's nested records,
`molecule` grouping (§3.8), `region: inside/outside`, a one-sided dihedral region, the legacy
cavity's flat keywords inside `&constraint`, and a shared `schedules` list from the text form.



## 22. Fifth round: the wall absorbs the cavity, and the schema stops being half-and-half (2026-10-01, #273)

The user's objection that started this round was structural, not numerical: `System` carried
nineteen flat restraint keywords *and* a `constraints` block, so a declaration could land in either
place and a reader could not tell which one the run obeyed. The ruling was to drop the compatibility
rather than layer it, and the retirement only became safe once the wall could be shown to do
everything the cavity did - which is what §22.1 measures.

### 22.1 The cavity is a law of the wall, and the relation is measured

`wall_law soft` is the archived polynomial potential: `V = k Q^(p/2)` over the dimensionless
squared axis ratio, with a box's axis scale its half edge. The comparison is done in
`test_constraint_legacy_equivalence.py::TestCavityLawIsTheSoftLaw` against the archived arithmetic
**transcribed into the test file**, not against the module being retired, so the judgement survives
the deletion of the thing it judged. Measured on two molecules of three atoms, both shapes, p = 2,
4 and 6:

| Quantity | Measured |
| --- | --- |
| Energy ratio, archived over wall | 3.000000000000 in every case, which is the molecule size |
| Force, archived minus wall | largest difference 1.1e-19 |
| Center captured by each | identical to 0.0e+00 |
| Wall Hessian against a central difference of the wall gradient | worst 6.9e-10, at the coarsest of four step sizes, and it shrinks as the step shrinks |

The factor of three is the one deliberate difference and it is defect D-15: the archived code
counted each molecule's centroid term once per atom of that molecule while its gradient carried no
such factor, so the archived force was never the derivative of the archived energy. The wall keeps
the force and drops the factor, which means a migrated run's **energy is the old number divided by
the molecule size** and its forces are unchanged. §13's migration table now states that, and the
draft case `draft_constraint_examples/constraint_wall_center.yaml` (which replaces
`constraint_legacy_cavity.yaml`) says it in its own header.

What the wall had to gain for that claim to be honest, and what the sweep found on the way:

- the `J J^T` term of the soft law's Hessian used `2 / a^2 * (d / a)` where `_ratio_terms` uses
  `2 d / a^2`. The two differ by one factor of the axis scale, and only at p > 2 - at p = 2 the
  curvature multiplier is zero, so the first four powers of the step sweep are what exposed it
  (a step-independent gap of 6.4e-4, which no single-step check would have distinguished from
  truncation error).
- a cuboid's radius is a box half edge. Writing the same number as a full edge at p = 4 is worth
  sixteen times the energy, and `test_a_cuboid_radius_is_a_box_half_edge` pins that factor instead
  of asserting "different".
- `wall_law soft` is refused on a periodic system by name, at input time: it is a spring toward
  a point rather than a barrier, so the four containment checks have nothing to measure, and a
  wrapped coordinate jumps its offset by a whole lattice vector at every face.

### 22.2 The four wall keywords, and the grammar decision each one forced

| Keyword | Rule | Why it is a rule and not a detail |
| --- | --- | --- |
| `wall_groups` | atom numbers, then the size of a molecule; `;` separates walls | the input says how to group, the target says how to weight, so one line covers `1-3 7-9 3` and `55-57 60 61 62 3` without a second spelling for "a molecule of one" |
| `wall_ref` | the anchor atoms | the point a wall is measured from should be nameable by atoms, not only by coordinates that the user has to compute |
| `wall_center` | the point | outranks `wall_ref`, which outranks the center of the selection; none of the three is the origin |
| `wall_law` | `hard` (default) or `soft` | two potentials over one geometry, so the choice belongs on the wall and not in a fifth family. The names are the physics ones the user set: a hard wall does nothing until you are in it, a soft wall pulls from wherever you are |

`wall_center`, `wall_ref` and `wall_groups` accept `none` as a whole segment. A positional line with
a gap in it is the one thing the flat grammar could not state, and padding it with the previous
wall's value would move a boundary the input never placed; `none` says "this wall declares nothing"
and is what the writer emits back for such a record, which is why a mixed block still round-trips to
the same plan fingerprint.

### 22.3 What was deleted, and what replaced each piece

| Removed | Replaced by |
| --- | --- |
| `workflow/constraints/legacy_cavity.py` and its schema family `cavities` / `LegacyCavityInput` | `wall.py`'s soft law; the controller's cavity branch, the state adapter's mass rule and the logger's family guard are gone with it |
| `normalize_cavities` and its five helpers (`_cavity_radii`, `_cavity_powers`, `_cavity_molecules`, `_cavity_compress`, `_broadcast`) | `normalize_walls` plus `_confined_atoms`; the ramp that `compress` was replaces the whole schedule machinery for that job |
| Nineteen flat `System` constraint fields, mirrored in all eight skill schemas | nothing: `RETIRE_CONSTRAINT_KEYS` turns each one into a named refusal that quotes its new spelling, so an archived input fails loudly instead of running unconstrained |
| `System.groups` as the fragment-group name | `center_groups`, because the constraint block's own `groups` table and the cavity partition both wanted the same word (§22.4) |
| `ensemble._dof`'s read of `system.freeze` | the plan's fix mask, which is the same set of components the integrator actually restores (measured identical at 9 / 3 / 3 / 8 degrees of freedom on the four fixtures) |

The archived `center_type` and `compress` readings are not lost: `mass` is `wall_target com`,
`geometric` is `wall_target cog`, `origin` is `wall_center 0 0 0`, and a compression is a geometry
ramp whose window opens at step 0 and closes at `steps`, which is the same number for every step of
a measured trajectory (`test_a_radius_ramp_is_the_archived_compression`).

One difference the wall does not hide: the archived cavity, given no `groups`, treated every atom of
its selection as a molecule of one. A wall given no `wall_groups` makes the whole selection one
group, which with one molecule is the center itself and therefore holds nothing. The migration
therefore writes the trailing `1`, and `TestCenterLaw` states the degeneracy rather than papering
over it.

### 22.4 Open, measured, not fixed here

**D-18.** On the migrated draft run, the MD trajectory's `Epot` equals the step's bare electronic
energy and `Etot = Epot + Ekin` to the printed precision, while the same step's recorded gradient
does carry the wall force (the run's H gradient is -0.02301741 where the provider's is +0.00654127,
the difference being the wall). So a constrained run's recorded total energy is not the quantity its
integrator conserves. The additive path itself is not in question - `controller.combine` sets
`energy_total = base + correction`, the finalizer writes the breakdown beside it, and the single
point case of §19.4 checks the row sum against the total. Where the MD record chooses the base over
the total is a question for the output layer, and it changes every constrained run's reported
energies, so it is registered for the user's ruling rather than decided inside this milestone.


## 23. Sixth round: both engines, both boundary conditions (2026-10-01, #274)

The campaign the user asked for: PySCF as the aperiodic skill, LAMMPS as the periodic one, after
repairing the local LAMMPS install. Everything below is read back out of what the runs wrote
(the 2026-10-01 aperiodic and periodic evidence sets, deleted on 2026-10-02; §33), by scripts
that open only framework products.

Aperiodic, water at HF/STO-3G, 5 steps:

| Case | Measured |
| --- | --- |
| `constraint_md_shake` | both O-H held to `6.482e-12` and `2.220e-16` A over the trajectory while the oxygen itself moves `8.745e-04` A |
| `constraint_md_rattle` | same position path, `1.712e-12` / `9.076e-11` A |
| `constraint_md_freeze` | frozen oxygen drifts `0.000e+00` A; a hydrogen still moves `2.933e-02` A |
| `constraint_md_bias` / `constraint_sp_bias` | bond pulled `3.054e-02` A off its start; single point carries rows `2.925018e-04` and `0` |
| `constraint_wall_center` (the migrated cavity) | `constraint.wall.water_cage` grows `1.829459771461e-4` -> `2.066109240232e-4` Ha |
| `constraint_md_restart` | the 4+4-step process now agrees with the 8-step continuous run **frame by frame at `0.000e+00` A** (D-19 fixed; the old criterion was a nonzero 6.5e-12 A) |

Periodic, 60-atom TIP3P box, 8.399091 A cell, 600 steps, 7 frames - the same box for all variants,
so the only difference is what the block declares:

| Case | Bonds / angle held | `Etot` drift |
| --- | --- | --- |
| unconstrained baseline | bond moves `3.744e-02` A | `+1.637560e-04` Ha |
| position-only shake, 2 bonds + angle | `8.934e-11` / `2.421e-11` A, `3.238e-09` deg | `+9.101845e-02` Ha |
| bonds + `rattle` | `8.206e-11` / `1.845e-11` A | `+4.705014e-04` Ha |
| bonds + angle, both with `rattle` | `8.002e-12` / `1.110e-15` A, `6.336e-09` deg | `+1.080206e-04` Ha |
| bias on one O-H (a cv inside a molecule) | rows `2.1e-15` -> `1.368028e-05` Ha | served |
| bias on an O...O pair across waters | refused: `2 separate connected components` | - |
| wall closing 2.8 -> 2.0 A over steps 50-300 | rows `0` -> `5.474741754228e-04` Ha | served |

Three things this campaign settled rather than assumed:

1. **The angle case with velocity projection used to be a named refusal** (the last row of the
   #264 batch); multi-condition RATTLE has since landed, so the refusal no longer fires and the row
   became a measurement instead. The old criterion was rewritten, not deleted.
2. **A cv has a periodic image exactly when the bond graph connects its atoms**, so the guard is
   per variable (§5.2). The decisive number is in the bias row: frames 4-6 have that O-H straddling
   a face, where the stored coordinates read `7.618 / 7.624 / 7.694` A and the minimum image reads
   `0.959168 / 0.950403 / 0.944545` A. The bias energy stayed at `5.0e-6 -> 1.37e-05` Ha; had it
   measured the stored pair it would have been `0.45` Ha, five orders away.
3. **A wall that closes is not an external field**, which is what the ramped-wall split of the
   member check is for: the cell-fitting checks walk every endpoint, the "starts inside" check reads
   the declared geometry once.

Open, measured, not fixed here (so they are on the record rather than in the dark): **D-19b** puts
the resumed segment's time column one step behind (`0.00/20.67/41.34/62.01/62.01/82.68/...`) while
every coordinate and energy agrees; **D-20** was listed here as a constrained optimization
accepting a point outside its own declared tolerance, and #278 **withdrew** it - re-read from the
accepted-geometry product the same case holds `6.461e-11` A, and the `8e-4` A figure came from a
measurement script that fell through to a directory listing when its artifact path did not exist
(section 28); what replaced it is **D-26**, the unmasked gradient norms in the stopping criterion;
**D-18** still records
`Epot` as the bare electronic energy for a run whose forces carry the constraint correction, and
the candidate site is where the writer picks `results['energy']` (`workflow/records/output.py:359, :381`)
rather than the total the finalizer assembled. **D-22** and **D-23** were found and closed by the
second periodic batch, and **D-24** (a resumed run records a different frame cadence, so the seam's
time label repeats and the two products cannot be compared frame by frame) joins the open list;
section 24 carries their numbers.

> **Superseded by section 31 (2026-10-02, #281-#285).** Everything this paragraph leaves open is now
> closed: **D-19b** and **D-24** by the absolute step numbering plus one cadence rule, **D-26** by
> the masked gradient criteria, **D-27** by driver ownership (with its default corrected), and
> **D-18** is withdrawn - its number compared the trajectory column against the log, which read the
> same key. The paragraph stays as the record of what was believed at the sixth round.

The local install note, because it decides which engine is testable on the test environment:
a locally built MPI LAMMPS under the user's home was unloadable (the MPI runtime was
gone with the package manager's open-mpi), and reinstalling LAMMPS restored open-mpi
5.0.11 and fftw, and also brought the same 22 Jul 2025 - Update 6 build as
`lmp_serial` / `lmp_mpi`. The old prefix is what the draft cases point at, so a
repaired LAMMPS is part of the reproducibility of these numbers.


## 24. Seventh round: the rest of the periodic matrix, on the other LAMMPS channel (2026-10-01, #275)

§23 covered one wall geometry (sphere) and one additive action (bias), and it did not say which
LAMMPS channel ran it. This round closes the remaining combinations the ruling asked for - the
three non-spherical walls, the two history potentials, a restart inside a cell - and the channel
attribution gets corrected on the way, because two of the seven cases died on the first try.

Channel attribution, corrected from the products rather than from intention. The §23 batch set
`python_enabled: true` and its `tmp_LAMMPS/h2o_force.sh` execs
`skills/lammps_skill/service/persistent/server.py --endpoint .../svc.sock`, with
`lammps service listening on ... (world 1)` as the first line of the LAMMPS log: **that batch was
the resident service**, not the subprocess. Every case below sets `python_enabled: false` and its
`h2o_force.sh` is `lmp -in $LAMMPS_INPUT.in`: **the subprocess channel**. Both routes now have a
real constrained campaign behind them, and `#274`'s closing "resident channel still to be verified"
is void.

Periodic, TIP3P 60 atoms in an 8.399091 A cubic cell, subprocess channel, read from
`<title>.log` contribution rows and `<title>.md.xyz`:

| Case | Steps / frames | Measured |
| --- | --- | --- |
| `pwall_ellipsoid` | 150 / 3 | semiaxes from the member extents (reduced norm kept at or below 0.95 at the start): start `0.9347`, worst over the recorded frames `1.1276`; `constraint.wall.vessel` printed in 3 rows, 2 nonzero, max `4.482278e-05` Ha |
| `pwall_box` | 150 / 3 | full edge lengths at 1.05 twice the per-axis extent: per-face fraction start `0.9524`, worst `1.2891`; rows 3 printed, 2 nonzero, max `2.064651e-04` Ha |
| `pwall_triclinic` | 150 / 3 | wall lattice sheared 5% and scaled `2.3441` so its tightest face leaves the same margin: face fraction start `0.4762`, worst `0.5016`; rows 3 printed, 1 nonzero, max `9.738379e-08` Ha |
| `pmetad_gaussian` | 60 / 2 | one intramolecular bond as the cv, hills every 2 steps: `constraint.metadynamics.explore` `0.000000e+00 -> 1.684309e-03` Ha, and the pushed O-H moves `0.9815 -> 1.1241` A, so the history potential changed the path rather than sitting at zero |
| `pmetad_rmsd` | 60 / 2 | an aligned-RMSD block over one whole water: rows `0.000000e+00 -> 4.841365e-04` Ha, same pair `0.9815 -> 0.9875` A |
| `prestart_base` / `prestart` | 8 in one process, 4+4 in two | held-bond spread against the first frame `1.839e-11` A continuous, `2.150e-11` A in two processes; the frame lists differ (2 frames `0.00 / 86.80` against 6 frames `0.00 / 37.20 / 37.20 / 49.60 / 62.00 / 74.40`), which is D-24 |

"A wall that is confining is a wall that gets crossed" is the finding of the first three rows. Each
geometry reached its surface during the run and its contribution row went nonzero, which is the
intended behaviour of a one-sided penalty - and it is exactly what D-22 turned into a `WallError`
before the fix. The three sizes were derived from the members' own extents (5% margin) rather than
picked, so a future run of the same fixture declares the same thing.

What the measurement had to get right first, recorded because the wrong version reads plausible:
folding a member's *stored coordinate* into the cell and then subtracting the wall centre is not the
minimum-image offset. `y = 5.25` in an `8.40` cell folds to `-3.15`, and the water that sits
comfortably inside then measures a fraction of `8.47` - the bench's first `measure_new.py` said so,
and the fix is to fold the *difference from the centre*. The containment numbers above are the
corrected ones.

Two of the seven cases also carry the criteria for the two fixes:
`test_constraint_periodic.py::TestWallInputChecks.test_the_containment_rule_is_about_the_start_not_about_every_step`
(D-22, refuses at step 0 and serves at step 1) and
`TestCapabilityGranularity.test_an_rmsd_history_potential_names_no_cv_and_is_still_served` with
`test_an_rmsd_block_the_graph_does_not_connect_is_refused` (D-23, served in the same component,
refused by name across components).

Gates after §26.1: repository `pytest pyrai3md` 3407 passed / 6 skipped (the six are the
numba-gated `test_fssh*` cases, none of them constraint), `-k constraint` 786 repository-wide and
779 under `qualification/verification/`, `-k "constraint or periodic"` 816 under that directory,
`unittest discover -p 'test_constraint*.py'` 772 OK, `utilities/tests` 132 passed.


## 26. Eighth round: the ensemble axis, and the cell that moves under a strict bond (2026-10-01, #276)

Two axes were still unmeasured. The first is the ensemble: `NVT` and `NPT` had never met a
framework constraint on a periodic system, even though the degree count the thermostat divides by
is the thing `ensemble._dof` takes from the plan (D-06). The second is the aperiodic side of the
capabilities the periodic batch had just exercised - the three non-spherical wall geometries and
the aligned-RMSD history potential had only ever been measured in a cell.

Periodic (LAMMPS TIP3P 60 atoms, 8.399091 A cell, subprocess channel, 150 steps, 3 frames):

| Case | Measured |
| --- | --- |
| `pnvt_shake` (NVT Nose-Hoover 300 K, two bonds with `rattle`) | bonds held to `8.131e-11` and `1.332e-15` A against the first frame at a declared `1.0e-10`; `Ekin 8.479741e-02 -> 9.454183e-02` Ha; `Etot` drift `+5.013150e-03` Ha, which is not this case's criterion - a thermostat is not an energy-conserving map, and saying so is what keeps the number from being read as a conservation claim |
| `pfreeze` (NVE, `fixes` freezing atom 1 in xyz) | frozen atom max displacement `0.000e+00` A on the shortest image, other 59 atoms move up to `8.772e-01` A; `Etot` drift `+2.2e-03` Ha against the `+1.6e-04` Ha unconstrained baseline, which is the pinned atom breaking the local structure while the fluid relaxes - recorded without a mechanism |
| `pnvt_freeze` (same mask under NVT) | frozen atom `0.000e+00` A, others up to `8.942e-01` A, `Ekin 8.328868e-02 -> 9.462430e-02` Ha |
| `pnpt_shake` (NPT, anisotropic Berendsen, same bond block) | the recorded lattice moves and shears by up to `0.025420` A per entry across the three frames, and the bonds hold only to **`4.806e-05` / `7.649e-05` A** - five orders against the tolerance the same block keeps to `8.131e-11` in the fixed cell. D-25 |

The NPT row is the finding of the round, so its instrument was tested before its number was
believed. Searching the minimum image with each frame's own lattice, with the frame-0 lattice, and
with the next frame's all give the same series - `0.981527 / 0.981539 / 0.981479` A, spread
`4.806e-05` - so the residual is not produced by which cell the measurement substitutes. The
affine ride was then tested per interval rather than in total: the a axis moves `+0.002105` and
`-0.000733` A between the three frames, which a bond riding the cell would answer with `+2.46e-04`
and `-8.57e-05` A, while the measured bond answers `+1.23e-05` and `-6.04e-05` A - `5%` and `70%`
of the affine figure. A fraction that is not constant kills both "the bond rides the cell" and "the
projection is a fixed number of steps behind", and what is left is that the projection does run and
does not compensate the cell's motion inside the interval - an ordering or ownership question
between the barostat and the solve. Choosing among those candidates would change the published
accuracy of every constrained NPT run; the framework neither refuses the combination nor delivers the declared
tolerance. It is registered as D-25 and left unfixed for the cell-ownership ruling (the A5 / D-14
line), which is the same ruling that decides who moves the box at all.

Aperiodic (PySCF water at HF/STO-3G, 5 steps):

| Case | Measured |
| --- | --- |
| `awall_nonsphere` (three `hard` walls, ellipsoid + cuboid + triclinic in one run) | all three contribution rows nonzero at every printed step, last values `5.505403e-06` / `2.217332e-05` / `3.674277e-05` Ha, so each geometry's own arithmetic ran on the engine. Sizes are derived from the same water (hydrogen at reduced norm `1.0077`, y fraction `0.5411` against a face at `0.5`), not chosen |
| `ametad_rmsd` (aligned-RMSD history, no cell) | history rows `0, 0, 0, 2.617433e-04, 2.725871e-07` Ha. The value at a deposition step is below the `5e-4` hill height, which is the documented next-step force discipline showing up in a product rather than only in a test; the drop after it is the narrow `width: 0.01` chosen for the case |

Shipped-input check: the six cases were renamed into `draft_constraint_examples/` (26 drafts now,
title equal to the file stem) and three were re-run after the rename. `awall_nonsphere` reproduces
the three wall rows digit for digit, `pnpt_shake` still gives `4.80572812e-05` / `7.64921032e-05` A
with the `0.025420` A cell change, and `pnvt_shake` echoes `tolerance 1e-10, velocity_constraint
rattle`. `--info` prints 28 lines that contain no constraint section, so it is not evidence of a
load and was not used as one.

Gates: no `.py` changed this round, and the same tree re-runs at 3403 passed / 6 skipped with 878
subtests, identical to #275 - the new drafts touch no test surface.

### 26.1 D-25, from a number to a seam

The NPT row above is the only case in this document where a framework constraint silently missed a
tolerance the input declared, so the record is the whole chain rather than the headline:

1. **A/B on the mechanism.** The same `oh_bonds` block holds to `8.131e-11` A when the cell is fixed
   (the NVT arm) and to `4.806e-05` / `7.649e-05` A when the cell moves. Making the barostat couple
   ten times more weakly (`barostat_tau: 1000` → `10000`) moves the cell 9.85 times slower and
   shrinks the residual 9.8x / 10.7x, so the error is proportional to the cell's rate - one step's
   worth of rescale, not an accumulated drift.
2. **The instrument before the conclusion.** Searching the minimum image with each frame's own
   lattice, with the frame-0 lattice, or with the next frame's gives the identical series, and the
   per-interval comparison against a full affine ride gives 5% and 70% - so neither the measurement
   nor "the bond rides the cell" explains it.
3. **A second symptom, different action class.** Under the same NPT run a `fixes`-frozen atom drifts
   `8.108e-04` A while it is `0.000e+00` A with a fixed cell. Bonds and anchors failing together
   points at the geometry as a whole leaving the manifold after the solve.
4. **Where, in the framework's own words.** `integrator/ensemble.py::_berendsen` says it "rescales
   the coordinates and the cell" and is "applied at post"; the loop in `md.py` projected
   (`apply_constrain`) before that post half step and recorded after it. LAMMPS does not move the
   box in these runs - the md profile produces no `fin_cell`.
5. **Fix.** Ask the geometry question again after the rescale (the `md.py` phase
   `constrain_after_cell`). It is one line because `constrain_state` is idempotent on a state that
   was already projected, and it returns the same object for a run with no plan.
6. **Measured on the motivating products, both ways.** Fast arm `4.80572812e-05` / `7.64921032e-05`
   → `1.71178627e-11` / `4.72044626e-12` A; slow arm `4.90837326e-06` / `7.16202361e-06` →
   `1.14519505e-12` / `3.75688369e-11` A; the NPT frozen atom `8.108e-04` → `0.000e+00` A; the
   fixed-cell arms unchanged (`8.13138e-11` → `8.13127e-11` A, and `0.000e+00` → `0.000e+00` A).
   Moving the new block back above the post half step reproduces the pre-fix numbers **bit for bit**,
   which is the negative control that ties the fix to the ordering rather than to a story.
7. **Cost, in the D-17 idiom.** 0.999 ms per step against a 166.229 ms engine-bound step (0.60%),
   and 0.001 ms for an unconstrained run, where the phase exists but does nothing.
8. **What the fix does *not* re-measure.** The fixed-cell arms were re-run only for the two cases
   above - NVT `8.13138e-11` → `8.13127e-11` A, NVE frozen `0.000e+00` → `0.000e+00` A - so the
   §23 and §24 numbers of the earlier rounds were not all produced by this build. The ordering can
   only matter when the barostat moves the cell, and those runs declare `barostat: "off"`; that is
   the argument for leaving them as they are, and it is stated here rather than hidden behind a
   claim that every number in this document comes from the current code.

Criteria now in the suite (`test_constraint_integration.py::TestBarostatOrdering`): the affine
rescale really does take a fixed bond to 1.001x and a frozen anchor `|anchor| * 0.001` away (the
negative control inside the test), the second projection brings both back to `<= 1e-9` A and
`<= 1e-12` A, the loop's phase order is pinned from `AIMD._run_trajectory`'s source
(`ensemble_pre < apply_constrain < ensemble_post < constrain_after_cell < update_history`, and the
reversed construction fails an assertion rather than a name lookup), and an empty plan still returns
the same object at the new seam.


## 27. Ninth round: the moving cell asked of every remaining action class (2026-10-01, #277)

Four cells of the matrix had no product behind them, all of them about what happens when the cell
itself moves: a multi-condition solve (the D-25 fix re-asks the geometry question, and an angle row
costs a rank and a Jacobian block that bonds do not), the four periodic wall checks under a
deforming cell, a force-side bias under the same, and a constrained geometry optimization on the
periodic engine, which at the time was D-20's question asked where it had never been asked -
D-20 has since been withdrawn (section 28), and this row's numbers survive as what a constrained
optimization inside a periodic box does.

Periodic (TIP3P 60 atoms, subprocess channel, 150 steps, 3 frames; bonds measured on the shortest
image with each frame's own lattice):

| Case | Measured |
| --- | --- |
| `pnpt_angle` (two bonds + the 2-1-3 angle, `rattle`) | bonds held to `2.337e-11` and `5.667e-11` A, angle to `9.216e-09` deg, cell moved `0.020523` A per entry - the rescale-and-reproject seam of §26.1 covers the multi-condition branch, not just the pair branch |
| `pnpt_wall` (the sphere wall, radius 2.8 A) | no refusal from any of the four periodic checks while the cell moves `0.025197` A per entry, and the run completes; all three `constraint.wall.vessel` rows are `0.000000e+00` Ha because the confined water never reached the surface - this case is evidence about the checks, not about force delivery |
| `pnpt_bias` (a bond bias) | `constraint.bias.hold_oh` `5.3341544973e-06 → 1.101342e-05` Ha, biased O-H `0.981527 → 1.014647` A (a bias is a force, so it should wander), cell `0.025232` A; the per-variable image rule and the wall checks refused nothing |
| `pnpt_opt` (constrained optimization, LAMMPS, fixed cell, 6 accepted steps) | both declared bonds held to `9.329e-11` and `6.793e-11` A against a declared `1.0e-10` |

The last row was written while D-20 was still on the register as a real slip; #278 withdrew that
reading, so the row's own number (`9.329e-11` / `6.793e-11` A) stands as what a constrained
optimization inside a periodic box does, and no comparison to a "failing" aperiodic case is implied.
See section 28.

Two input-shape facts worth keeping, both learned by being refused:

- A strict constraint's rows travel in the `bonds` field whatever the kind is. Writing
  `atoms: [2, 1, 3]` for an angle is refused with `constraint 'hoh' needs an explicit list of 3
  atom rows` (`schema/normalize.py:1268`) - a correct named refusal, and one every draft now warns
  about in its header because it is easy to hit.
- A length-returning minimum-image function is fine for bonds and silently wrong for angles. The
  first `measure_batch4.py` did exactly that and reported **every angle as `0.0000` deg**, which
  reads as perfect conservation. Returning the vector, and using one helper per purpose, gave the
  real `9.216e-09` deg. A shared helper across two different questions is the hazard, not the
  arithmetic.

Shipped inputs: the four cases were renamed into `draft_constraint_examples/` (32 drafts now -
`constraint_md_periodic_npt_{angle,wall,bias}.yaml` and `constraint_opt_periodic.yaml`) and re-run
from those files; every number above reproduces digit for digit. The first attempt failed all four
at once for a reason that had nothing to do with the inputs - the fixture had not been copied into
the per-case directories - and the framework said so by name (`LAMMPS topology file not found:
h2o.data`).

### 27.1 The third branch of the multi-condition solver, on an engine (2026-10-01)

A dihedral is the last condition kind that had never been priced by a real engine: the periodic
drafts all use the TIP3P water box, whose force field carries no torsion term to hold, and the
aperiodic drafts used a 3-atom water, which has no torsion to define. `constraint_md_torsion.yaml`
is hydrogen peroxide at HF/STO-3G with one declared row, atoms 3-1-2-4 (H-O-O-H).

Measured over the five recorded frames: the torsion starts at `160.000000` deg and its spread
against that first frame is `2.625e-10` deg, while the two things nobody declared move by
`8.435e-03` A (the O-O bond) and `3.823e-02` A (an O-H). The controls are not decoration - a
projection that froze the molecule would report a perfectly held torsion, so the only meaningful
statement is "this angle stayed and these lengths did not".

Units, so the number is not over-read: a strict constraint's `tolerance` is a dimensionless relative
residual (`constraints/shake.py:48`), and `2.625e-10` deg is a physical spread. They are not the same
kind of quantity and are not compared here.

Two bench errors are part of this record because both could have produced a clean-looking wrong
answer:

- The generator printed the starting torsion as `24.474568` deg by feeding the atom order
  `(O, O, H, H)` to a function that expects a chain. The construction was right; the *question* was
  asked of the wrong four points. The framework's own plan echo (`constraint dihedral hooh 4 atoms
  (3 1 2 4)`) is what put the chain back in order, and the true start is `160` deg.
- `measure_batch4.py` computed angles through a minimum-image helper that returns a **length**, and
  reported every angle as `0.0000` deg - a silently perfect conservation. Returning the vector gave
  the real `9.216e-09` deg for the periodic angle case. One helper serving two different questions is
  the hazard.

Still uncovered, stated plainly: the periodic side of a dihedral needs a fixture whose force field
has a torsion term, and none of the periodic drafts' TIP3P box does. The torsion branch is therefore
measured on PySCF only.


## 28. Tenth round: D-20 withdrawn, D-26 measured in its place (2026-10-01, #278)

The plan for this round was the equal-conditions A/B that §27 said was needed: same molecule, same
optimize block, same starting geometry, one engine. The first arm of it did not need an engine swap
at all, because the thing that had to be re-measured was the aperiodic case itself.

| Arm | Declares | Slip of the declared bond | Undeclared 1-3 moves | Last `max_g` |
| --- | --- | --- | --- | --- |
| `d20_b_only` | the strict bond | `7.224e-11` A | `1.252e-02` A | `8.3668e-02` |
| `d20_b_fix` | bond + frozen oxygen | `1.104e-10` A | `9.976e-02` A | `2.0797e-01` |
| `d20_b_bias` | bond + harmonic pull on the other O-H | `9.920e-11` A | `1.269e-02` A | `8.4217e-02` |
| `d20_b_all` | all three, i.e. the shipped case | `6.461e-11` A | `9.222e-02` A | `1.9529e-01` |
| archived #274 artifact | all three | `6.461e-11` A | `9.222e-02` A | `1.9529e-01` |
| shipped draft, rerun | all three | `6.461e-11` A | `9.222e-02` A | `1.9529e-01` |

Every arm holds the declared bond at the declared tolerance, including the original case read from
its own accepted-geometry product. **D-20 is therefore withdrawn.** It is not a narrowed defect, a
conditional defect, or a defect on one engine: the `8e-4` A never appeared in any file the run wrote.

Where the number came from, recorded because the failure mode is reusable. The measurement script
for the aperiodic batch opened `constraint_opt_surface/constraint_opt_surface.opt.xyz`. The optimizer
writes `opt_geom.xyz` and `opt.xyz` in the run directory, without the title prefix, so the path never
existed; `if os.path.exists(xyz):` was false, and the `else` branch printed the directory listing
instead. The bond length I then reported was read off a print next to that listing rather than from
the artifact the claim was about. The rule that comes out of this is short: **a measurement script
must abort when the file it measures is missing** - an `else` that reports something else turns a
missing artifact into an unfalsifiable statement, and it looks like output. `measure_d20.py` reads
`opt_geom.xyz` directly, with no fallback branch, and it found the missing-prefix case on its own
first run.

What replaced the defect is D-26, and it is visible in the last column of the table above. The
displacement of a constrained optimization is masked at its frozen components
(`optimizer/base.py:168`, which is why the bonds hold), but the gradient norms are taken over the
whole raw gradient (`:213-217`) and compared against `opt_grms`/`opt_gmax` (`:247-251`). The 2x2
isolates the cause cleanly: the bias changes `max_g` by `5e-4` (`8.3668e-02` → `8.4217e-02`), the
frozen atom doubles it (`8.3668e-02` → `2.0797e-01`). So the stopping criterion counts forces on the
three coordinates the step is forbidden to move. The direction of the error is conservative - it
makes convergence harder, not easier, which is why no published structure is wrong - and the fix
(masking the norms the way the displacement is masked, which is the standard convention for
constrained optimizations) would change which runs report converged. That is a ruling, not a
milestone edit, so it is registered and left.


## 29. Eleventh round: two clock defects attributed, and a gate that is missing (2026-10-01, #279)

Nothing here changes shipped behaviour; the point of the round was to finish the attribution the
register owed on the two resumed-run defects so that a ruling could be given with numbers in front of
it, and to record one combination that no layer refuses.

**D-19b and D-24 are one ordering plus one overloaded variable.** Reading a real four-step
checkpoint back out of its pickle gives `current_step = 3`, `simulation_time = 0.0`,
`total_steps = 0`. The first two follow from `md.py` writing the runtime record at phase 6 and
calling `advance_step()` at phase 7: every checkpoint therefore carries one fewer step than the
process actually completed, and `advance_step` accumulates no time by design
(`data/runtime/state.py:111-121`), so the time column inherits the step number and the resumed loop
starts by repeating a step already recorded. The frame cadence follows from `total_steps` being 0 in
that same record: `md.py:439-441` then sets the budget to `self.maxiter` - a *remaining* count -
while `:686` compares `final` against it as though it were an *absolute* end, so on the remaining
steps `step >= self._total_steps` is always true and each one writes a frame.

An experiment excluded the more convenient hypothesis. Rewriting the checkpoint's `total_steps` from
0 to 8 and resuming produced byte-identical time columns (`0.00 37.20 37.20 49.60 62.00 74.40` in
both arms), so the resumed run does not read that field at all and the conflation is `maxiter` versus
absolute step. That matters because the heavier fix (changing checkpoint semantics) is thereby ruled
out, leaving two local edits: advance before writing the record, and compare `final` against
`restored step + remaining`. Both alter published products (time labels, frame lists) and neither
touches geometry or energy, which is why they wait for a ruling rather than being applied inside a
testing milestone.

**D-27: two constraint engines can hold one trajectory and nothing refuses it.** The framework's
integrator owns the geometry when the profile is framework-driven, and the skill already refuses its
own SHAKE outside `ENGINE_INTEGRATED_PROFILES` (`skills/lammps_skill/schema/method.py:78`) - that
direction is gated. The reverse is not. Declaring `profile: ff_nvt` with `lammps_shake: 1` *and* a
framework `constraints` block runs to completion: the engine log says `Finding SHAKE clusters ... 20
= # of frozen angles` and `SHAKE stats Bond 1 0.957199 delta 4.67149e-06` against its own TIP3P
target, while `apply_constrain` spends 1.331 ms per step pulling the bonds back to the bound
`0.981532` / `1.003004` A, which is what the record carries. Two engines, two different targets, no
warning, and the answer to "which length is held" comes from whoever wrote last. Where a gate
belongs is a design question the register does not decide: the skill cannot see the framework's plan,
the framework must not import skill config, and the one place that already reads both the plan and
`lammps_shake` is `schema/validate`, which could refuse by naming the combination without adding a
channel.


## 30. Twelfth round: the two cells that looked like fixture work and were not (2026-10-01, #280)

Two cells of the periodic matrix had been left as "needs a new fixture": an angle constraint on a
molecule that a cell face tears in two, and the dihedral branch inside a cell. Both turned out to be
answerable with the fixture that already ships, and the reason is that the fixture's own starting
geometry is torn.

**The torn water.** Reading the shipped coordinates against their minimum images finds five waters
whose stored O-H separations differ from the physical bond by 6.6 to 7.2 A. Water 8 (atoms 22-23-24)
has **both** arms across a face: stored 7.7295 and 8.1616 A, minimum image 0.9868 and 0.9895 A.
Declaring its two bonds and its angle is therefore the strongest available test of the coordinate
convention, because nothing about that molecule can be evaluated on stored numbers at all.

| Quantity | Measured on the record (`constraint_md_periodic_torn_angle`, 150 steps, 3 frames) |
| --- | --- |
| bond 22-23 | held to `3.647e-12` A against a declared `1.0e-10` |
| bond 22-24 | held to `7.772e-16` A |
| angle 23-22-24 | held to `3.097e-09` deg |
| control: the same angle from stored vectors | `10.7611` deg in one frame, `103.4039` in another, a spread of `99.8885` deg |
| channel | resident service (`h2o_force.sh` execs `service/persistent/server.py`, log says `lammps service listening`), which is also the first multi-condition solve on that channel |

The control row is the point: an evaluation that used stored coordinates would not be slightly wrong,
it would chase a variable that jumps by nearly a right angle between frames. The same-component and
half-box guards served the molecule because, unfolded, it is one molecule.

**The dihedral branch, both sides.** `constraint_md_periodic_torn_dihedral.yaml` (retired 2026-10-02;
the artifact it produced is what this paragraph reads, the rule is now witnessed by
`test_constraint_periodic.TestDihedralBranch`) declares a torsion
23-22-25-26 - H-O of one water, O-H of the next - inside the cell, and the run stops at the guard
that now asks per variable:

    ImageError: constraint 'hbond_twist' names atoms [23, 22, 25, 26], which lie in
    2 separate connected components of the perceived bond graph; ...

`constraint_md_torsion_intermolecular.yaml` is the identical declaration with no cell, on PySCF, and
it is served: over 4 frames the torsion holds to `1.421e-14` deg while the undeclared O...O across
the hydrogen bond moves `6.503e-03` A. Together with §27.1's intramolecular HOOH case (`160.000000`
deg held to `2.625e-10` deg), the three possible answers for a torsion - served inside one molecule,
served across molecules with no cell, refused across molecules in a cell - each have a product.

Three offline criteria were added for the same branch (`TestDihedralBranch` in
`test_constraint_periodic.py`), including the negative control that the refusal is a periodic
statement: the same row with `pbc=[0,0,0]` builds a plan.

The "needs a new fixture" note in the register was wrong in one respect worth keeping: it assumed
the missing thing was a force field with a torsion term, when what the cell actually needed was a
molecule already torn by a face, and the shipped box had five of them.

Gates after section 30: repository `pytest pyrai3md` 3410 passed / 6 skipped, `-k constraint` 789
repository-wide and 782 under `qualification/verification/`, `-k "constraint or periodic"` 819 under
that directory, `unittest discover -p 'test_constraint*.py'` 775 OK, `utilities/tests` 132 passed.

---

## 31. Thirteenth round: the five rulings land, and three of them turn out to be about the instruments that read them (2026-10-02, #281-#285)

The user answered all five open items in one message. Four were口径 choices to implement; the fifth
(D-18) was a reading that did not survive being re-taken against something independent, and it
changed what had to be built.

**The restart pair, with a criterion the ruling made stronger (#281).** Step numbers are absolute now
(`started = runtime.current_step`, `total_steps = started + maxiter`, the loop runs
`started+1 .. total_steps`, and `advance_step()` happens before anything is written, so the
checkpoint, the trajectory frame and the run-log block all describe the step that just finished).
That alone removes D-19b's repeated label but not D-24's frame list: each process's last step is
`final`, and `final` forces a record, so a segmented run still gains one frame per seam. The rule
that closes the gap is one sentence - **a step that writes a checkpoint also writes a frame** - and
both decisions now read the same predicate (`AIMD._runtime_file_due`), because the checkpoint
cadence is the one piece of information a resumed process is guaranteed to share with the run it
continues. Measured on both engines: the periodic pair writes `[1,2,4,6,8]` in both arms with
identical time columns and `0.000e+00` A per frame over all 60 atoms, the aperiodic pair writes all
eight frames with `0.000e+00` over 3 atoms and equal energy rows. Force evaluations count 8 against
4+4 (`h2o_force.sh` invocations and `converged SCF` blocks), which is D-19's "no warm-up on resume"
read off the products rather than asserted.

**D-26 (#282)** became `projection.declared_mask` - freeze mask union every atom named by a strict
condition's rows - and the gradient criteria are asked only outside it. The number that shows the
change is from the very case that raised it: `constraint_opt_surface` step 1, same geometry,
`Gradient RMS 0.04358507 -> 0.00873618`, `Gradient Max 0.08206846 -> 0.01490334`, energy and both
displacement columns unchanged to the printed digit.

**D-27 (#284) taught the default-direction lesson.** Ownership travels on `control.driver`, the
framework drops its own plan when the driver is a skill and the LAMMPS skill drops its shake
keywords when the driver is `main`, and each side warns. But the skill's *default* was written as
`main`, meaning "the framework owns this unless told", and that is backwards for a section that can
be built with no framework in sight: 29 red tests, 18 of them byte-comparisons of generated LAMMPS
input, all of them correct complaints. A cross-boundary switch must default to "this side decides",
and only the other side's explicit statement may take it away.

**D-18 (#283) is a same-key comparison, the third instrument error of the campaign.** The
registered claim was that `md.energies`' `Epot` equals the step's bare electronic energy. Both
numbers in that sentence came from `results['energy']` - the writer and the log are two readers of
one key - so the equality was a tautology. Comparing against something independent (the same
geometry through an unconstrained single point) gives `-74.9631631182905238` bare against
`-74.9631292844484278` recorded, and the `3.3833842e-05` Ha difference is the bias row. `Epot` was
already the conserved quantity. The deliverable the ruling asked for was the decomposition, which
now prints as `&constraint energies in Hartree`: the engine's own half as `origin pot` (stored by
`runtime.finalize` as `energy_origin`), one row per constraint carrying **the id the input wrote**,
and `Epot` last, so a reader can do the addition on the page. Six products verify it, including a
geometric-only run whose two numbers are equal and a periodic LAMMPS wall run.

**The ethanol cell (#285) and D-28.** A real torsion constraint needed a real force field, and
getting one onto the framework surfaced a support-library defect: `utilities/bonds.py` answers
`C-C` with `1.20` and `C-O` with `1.16` - the shortest members of those series - while periodic
perception tests `distance < bond_lib * 1.2`. Every ordinary single bond in an organic molecule is
therefore invisible to the bond graph, and the constraint was refused as naming three components of
a molecule that is one molecule. The heavy-atom entries now carry single-bond values, and the
widening is bounded by measurement rather than by argument: the shipped TIP3P box still perceives
exactly its 40 O-H bonds and nothing else, because its closest O...O and H...H contacts (2.704 and
1.519 A) sit well outside the new reaches (1.776 and 0.888 A). The ethanol box then perceives
exactly the 8 bonds its topology declares. Two deviations from the instruction are on the record in
CHANGELOG #285: ORCA's analytic and numerical gradient helpers (`orca_scfgrad`, `orca_numfreq`) do
not run in this shell - they fork, hold 0.01 s of CPU and never return, under both `orca` directly
and `mpirun -n 1 orca`, for HF as for PBE0, while the energy-only path and the `orca_vpot` ESP step
do complete - so the relaxation was done by this repository's PySCF skill at PBE0/def2-svp and ORCA
supplied the single point on the relaxed structure (`-154.736415620280` Ha).

Gates after section 31: repository `pytest pyrai3md` **3427 passed / 6 skipped / 916 subtests**,
`-k constraint` **806** repository-wide and **799** under `qualification/verification/`,
`-k "constraint or periodic"` **836** under that directory, `unittest discover -p
'test_constraint*.py'` **792 OK**, `utilities/tests` **132 passed**. Shipped drafts: **38**, and #286 aligned them to the examples format, so every product is named `<file stem>_example.*` now (`constraint_opt_surface_example.log`, `constraint_md_periodic_restart_example.rt`, ...) - the figures in sections 23-30 were measured under the old titles and the restart, ethanol, D-18 and D-26 ones were re-taken under the new ones and come back identical
(`constraint_md_restart_base.yaml` and `constraint_md_periodic_dihedral_ethanol.yaml` added; the
restart pair's headers now state frame-sequence identity as the criterion), all loaded through
`parse_input -> build_config -> normalize -> validate -> build_plan` by `TestShippedDraftsLoad`, and
both new ones executed for real on their engines.

## 32. Fourteenth round: one keyword grammar for three carriers (2026-10-02, #287)

The ruling that opened this round was about a diagram, not about code: the user read a draft's
`system_constraint:` block, saw `walls: [- id: vessel, geometry: sphere, schedules: [...]]`, and
said the design had drifted too far from the classic input - *yaml and json should keep the flat
design of the classic input, only written in yaml or json; the keywords of input, yaml and json
must be the same*. Two further rulings made it operational: the structured records are a
convenience of the code and must not be exposed at all (no special guidance text either - a
non-conforming input is a main-framework error, users have manuals and templates, an agent reads
the docs and the code), and lists exist precisely to replace the text-separator tricks, so
**one list element is one object** and the nested list is one object's row of tokens.

**The reader has one path now.** `_constraint_declaration` is `_read_constraint_section`; the
routing function that recognised "keys that are the schema's field names" and handed them to
pydantic untouched is gone, along with `_constraint_structured` (which had enforced "one family,
one spelling") and `CONSTRAINT_STRUCTURED_KEYS`. A record list at input is now refused *by name*,
each refusal naming the flat line that writes the same family (`'biases' is refused: write
bias_bond, bias_angle, bias_dihedral or bias_cv for the biases and bias_*_ref, bias_*_fc,
bias_*_region for their parameters`). The keywords themselves are unchanged: 21 object lines, 46
parameter lines, 3 unit keys, `wall_type`, and the 50 ramp plus 30 component keys the suffixes
derive.

**The list rule, stated once and enforced everywhere.** `wall_atoms: [[1, 2, 3]]` is one wall on
three atoms and `wall_atoms: [[1, 2], [3, 4]]` is two walls, exactly as `wall_atoms 1 2 3` and
`wall_atoms 1 2 3; 4 5` read in text; a scalar-valued keyword takes a scalar (`shake_tol: 1.0e-10`,
which is the same one-segment broadcast as the text line); the placeholder segment `none` is the
native `null`, because `_constraint_is_none` already matches it (adapter.py:1008). The three
broadcast rules (one segment over all objects, pad with the last value, refuse over-long) are the
same function on both carriers, which is what makes the equivalence provable rather than hoped
for. `wall_type` keeps its keyword spelling - it is the switch that makes the block default-closed
in text, and a structure that could infer it would be a second rule about which geometries exist.

**A segment may carry several rows, and only whole ones.** `shake_bond 1 2 1 3` is one named block
holding two bonds; `shake_bond 1 2; 1 3` is two blocks of one bond each. The width comes from the
keyword (`row2`, `row3`, `row4`), so a segment whose count is not a multiple is refused as a
miscount. This is what closed D-29: without it the flat grammar cannot state one constraint record
with two rows, which is how every shipped water case had been written.

**`wall_groups` keeps the molecule dimension exactly as text states it** (the ruling: *write it in
the established format and just check that the selected atom count divides by the trailing molecule
size*). One element is one wall, its content is atom numbers with the per-molecule count last, so
`[[1, 2, 3, 4, 5, 6, 3], [7, 8, 9, 10, 2]]` is the two-molecule wall and the two-atom wall of
`wall_groups 1 2 3 4 5 6 3; 7 8 9 10 2`. Molecules of unequal length have no nesting: declare
another wall with the same parameters. The divisibility refusal was already live (`'wall_groups'
lists %d atoms with %d per molecule, which does not divide out; a partial molecule would leave %d
atom(s) unconfined`), so requirement 2 needed confirmation, not code.

**The audit of what the flat grammar cannot state** (the ruling: if equivalent support exists,
delete the redundant keyword or field; if it is a real hole, add it to the flat grammar and sync
input/yaml/json). Deleted, with the reason in each case: `version` (only 1 was ever accepted), the
shared `schedules` section and the per-record `schedule` name reference (a ramp is written on the
parameter it smooths), `walls.lattice` (an alias of `cell`, and `wall_triclinic_cell` states it),
`groups[].id` (never read; `group_name` is the handle), `constraints[].reference` and
`fixes[].reference` plus the keyword `fix_ref` (normalize derives `initial`/`explicit` from whether
a target or a center was given, and the fix validator refused anything but `initial` anyway), the
`bonds`-as-`atoms` aliases on `_Selector` and `CVInput` (a text shape feeding the atom field),
`molecule` (a named refusal with no spelling), the singular `cv` alias on metadynamics, and
`BiasInput.kind` (harmonic was the only law). `bias_element`/`bias_group`/`meta_element`/
`meta_group` went the same way for a structural reason worth keeping: the classic grammar never had
them, because a bias's and a history potential's **object line is the selection** - its segments
count the objects - so an element list of unknown length has nothing to align against, while a wall
and a fix, whose objects are counted by their geometry, do have `wall_element`/`wall_group` and
`fix_element`/`fix_group`. The one real hole was the wildcard, and it is a value range rather than
a keyword: `shake_bond_type X-H`, X meaning any element that is not hydrogen (an `X-C` or `X-X`
label is refused saying so; case and order are free).

**The instrument that made three of this round's defects visible** is `verify.py` beside these
drafts: for each shipped input it strips the retired fields (only where the reader derives the same
value anyway), writes the block through `write_constraint_section`, reads the emitted keywords back
through the reader, and compares the two **plan fingerprints**. 38 drafts had to agree. They did not
at first: seven failed on D-29's dropped rows, six on D-30's six-digit `%g`, and one crashed in the
writer as D-31's `KeyError`. After the three fixes the run prints `0 of 39 drafts differ` (the 39th
is the new X-H case, written flat from the start and measured on its engine instead).

**Engine re-runs on the rewritten inputs** (the 2026-10-01 flat-draft evidence set,
deleted on 2026-10-02; §33):

| case | criterion | reading |
|---|---|---|
| periodic restart, 8 steps against 4+4 | frame list, time column, per-frame coordinates | both arms `[1, 2, 4, 6, 8]`, times `0.0/12.4/37.2/62.0/86.8`, `max \|dx\| = 0.000e+00` on all five shared frames, control (step 2 against step 4 of one arm) `2.564e-02` |
| aperiodic restart, same pair | same | both arms 8 frames, times `0.0 ... 144.69`, `0.000e+00` on all eight, control `1.354e-03` |
| held bonds across the restart | the two bound O-H lengths | `0.9815315425` / `1.0030037255` A, identical on every frame of both arms |
| constrained-energy block (D-18) | `origin pot` + rows = `Epot` = the trajectory column | 5 steps, both identities `0.000e+00`, the row named `hold_oh` |
| ethanol torsion (the real force field) | declared torsion against an undeclared one | spread `2.842171e-14` deg declared, `5.605489e-01` deg undeclared |
| constrained optimization (D-26) | held bond, frozen atom, masked gradient | held O-H spread `7.037104e-11` A against `1e-10` relative, frozen atom `0.000e+00`, last-step Gradient Max `1.1848e-01` against the `2.0797e-01` the register recorded before the mask |
| X-H wildcard (new draft) | six perceived rows held, non-X-H bonds free | `[[1,5],[1,6],[1,7],[2,8],[2,9],[3,4]]` = 5 C-H + 1 O-H, worst relative drift `4.989761e-11`, control C-C/C-O `1.687058e-03` |

Shipped drafts are **39**, each one now a flat block, and `TestShippedDraftsLoad` still pushes every
one through `parse_input -> build_config -> normalize -> validate -> build_plan` - which is the only
test that can see an input the reader would refuse. New sentinel classes: `TestListsAreTheSegments`
(five cases: three objects as three elements, one multi-token object nested, several rows in one
object, `null` as the placeholder, and the retired spellings refused by name with their replacement)
and `TestBondPairsByType` (six cases for the wildcard, including that an H-H pair is not a match).

Gates after section 32: repository `pytest pyrai3md` **3440 passed / 6 skipped / 917 subtests**,
`-k constraint` repository-wide **813 passed / 93 subtests** and **806 / 93** under
`qualification/verification/` (93 is every subTest in that selection; the drafts' own criterion,
`TestShippedDraftsLoad`, reports **39 subtests** - one per shipped input, 38 before this round),
`unittest discover -p 'test_constraint*.py'` **799 OK**, `utilities/tests`
**138 passed** (six of them the new wildcard cases, against the 132 recorded in section 31). The
`# Fixture:` header lines added to the 25 periodic drafts came after the repository run, so that
criterion was re-taken on its own: **1 passed / 39 subtests**. Two
measurement-environment notes belong with them, because both first read as a defect and were not: a
cold bytecode/assertion-rewrite cache made one directory's collection take **224.91 s** where the same
warm collection takes **2.91 s**, and another user's process on the test environment stretched the 125 s full
suite and the 10 s constraint subset to minutes - so `ps -o etime,time,state` (is CPU time
accumulating?) is checked before anything is called a hang, and long gate chains are run one stage at
a time.

---

## 33. 证据目录的退役（2026-10-02，#289）

`evidence_20260928/` … `evidence_20261002/`（仓库根，五个）已从工作树删除。它们是这个子系统
各轮的台架与原始产物：逐帧读数、A/B 两臂的 `.md.xyz`/`.log`、比对与量测脚本、以及乙醇拓扑的
中间文件。删除的前提是**判据级读数已经在文档里**，这一条把它们的位置说清楚：

- 每个缺陷行的读数在 §18.3 的 disposition 表与 §20-§32 的轮次记述里（D-18/D-19b/D-24/D-25/
  D-26/D-27/D-28/D-29/D-30/D-31 各带自己的数字与复算条件）；
- 每轮的数字与偏离裁定在 `CHANGELOG.md` 的 #265-#289；
- 出货输入的写侧一致性判据（原 `flat_drafts/verify.py`）已升为常驻测试
  `TestShippedConstraintExamples::test_writing_a_shipped_block_out_and_back_in_keeps_the_plan`，
  对 `pyrai3md/examples/` 里全部 39 份约束输入跑 `写出去 → 读回来 → 比 plan 指纹`；
- 乙醇拓扑**再生所需的输入**（`ethanol_ffa.yaml`、`ethanol.lt`、`run_moltemplate.sh`、
  `ethanol.xyz`、`ethanol_opt.xyz`、`ethanol_sp.inp`）留在
  配方文本已逐字并入本文 **§34**，`dev/methods/fixtures/` 随之删除（生成物一律不入库）。

正文与 CHANGELOG 里仍会出现 `evidence_2026xxxx/…` 这样的路径，那是**当时产物所在的位置**，
不是今天还能打开的文件。本轮删除前已整体快照到一份本地 tarball（此后也已删除）
（含五个目录、被合并的两份文档与被退役的草稿目录），需要复核旧读数时从那里取，不要改文档。

同一批收尾里，`dev/constraint_next.md` 并入本文**附录 B**、
`dev/methods/CONSTRAINT_DEVELOPMENT_APPENDIX.md` 并入**附录 A**，两份文件删除；
`draft_constraint_examples/` 退役（其 39 份输入已是 `pyrai3md/examples/` 的一部分）；
包目录里那份 `constraint_end_to_end.log` 删掉。



## 34. 乙醇拓扑的再生链路与配方（原 `dev/methods/fixtures/ethanol/`，#290 并入后该目录删除）

`pyrai3md/examples/constraint_md_periodic_dihedral_ethanol.yaml` 与 `constraint_md_periodic_xh_ethanol.yaml` 用的 `ethanol.data` 不是手写也不是别处拷贝的。本节留着**再生它所需的输入**；生成物（`.gbw`、`.densities`、ffa 中间文件、moltemplate 的 `output_ttree`、ORCA 输出）不入库，按下表重跑即得。算例判据在 §31/§32，链路各步的数字在 CHANGELOG #285 与 #287，本节不重复记数，只保管配方。

| 文件 | 是什么 | 怎么来的 |
|---|---|---|
| `ethanol.xyz` | 起点几何，与两份算例 `coordinates:` 同一坐标（盒内摆放） | 由下面的内部坐标搭出后平移入盒 |
| `ethanol_opt.xyz` | 松弛后的几何（分子坐标系摆放） | 本仓库 PySCF 技能，PBE0/def2-svp，33 步 GDIIS 收敛，末步 `Gradient Max 1.6132e-04` |
| `ethanol_sp.inp` | ORCA 单点输入 | 手写；在同一松弛结构上出 `-154.736415620280` Ha |
| `ethanol_ffa.yaml` | FFautomator 作业描述（GAFF2 + RESP） | `ffa ethanol_ffa.yaml` 产出拓扑与电荷 |
| `ethanol.lt` | moltemplate 规则（FFautomator 内部调用的那份） | ffa 生成 |
| `run_moltemplate.sh` | 出 `.data` 的命令 | ffa 生成 |

起点几何的内部坐标：C-C `1.512`、C-O `1.427`、O-H `0.955` Å（手工构造，不是测量值）。

链路产物规模：`ethanol.data` 为 9 原子 / 8 键 / 13 角 / **12 二面角**、10×10×10 Å 盒；RESP 电荷拟合的就是那份 ORCA 波函数，9 个电荷代数和为 0。

**ethanol_ffa.yaml**

```yaml
--- # ffa config: one ethanol in a 10 x 10 x 10 Angstrom cell.
#
# Consumed by `ffa ethanol_ffa.yaml` from Force Field Automator
# (the Force Field Automator checkout used at the time, installed into the project
# venv), which is
# what turns the ORCA-optimized geometry into the LAMMPS topology the periodic
# dihedral draft uses. `file` points at the optimized structure, not at the
# internal-coordinate guess, so the topology and the constrained dihedral come
# from the same relaxed molecule the single point was computed on.
project_name: ethanol

forcefield: gaff2

ethanol:
  name: ethanol
  file: ethanol_opt.xyz
  charges: 0
  num: 1
  resp:
    # The RESP step reads an ORCA wavefunction job, not a bare geometry: `file`
    # is that job's stem and `path` where its .gbw/.densities/.densitiesinfo
    # live. Those are the products of the single point above (PBE0 / def2-SVP /
    # def2/J / RIJCOSX on the relaxed structure), so the charges are fitted to
    # the same level the fixture's energies are reported at.
    file: ethanol_sp
    path: sp

ethanol_layer:
  type: molecule
  build:
    - ethanol
  mix:
  thickness: 0.0

system:
  box: [10.0, 10.0, 10.0]
  build:
    - ethanol_layer
  vacuum: 0.0
  space: 0.0
  noz: false

job:
  - equilibration

equilibration:
  ensemble: NVT
  temperature: 300
  initialization: true
  time: 1

resp:
  orca_path: /path/to/orca
  vdw_point_density: 10.0
  ihfree: False
  toler: 1.0e-8
  maxit: 25

packmol:
  seed: null
  packmol_path: /path/to/packmol

moltemplate:
  forcedrewrite: False
  moltemplate_path: /path/to/moltemplate/bin
```

**moltemplate/ethanol.lt**

```text
import "sub_l0_m0_ethanol.lt"

# Periodic boundary conditions:
write_once("Data Boundary") {
    0.0           10.00000000 xlo xhi
    0.0           10.00000000 ylo yhi
    0.0           10.00000000 zlo zhi
}

layer_1_sub_1 = new sub_l0_m0_ethanol[1]
```

**run_moltemplate.sh（绝对路径改为 `<repo>` 相对形式）**

```bash
export PATH=/path/to/moltemplate/bin:$PATH
moltemplate.sh -xyz <repo>/ethanol/ethanol.xyz <repo>/ethanol/moltemplate/ethanol.lt > <repo>/ethanol/moltemplate/ethanol.log
```

**sp/ethanol_sp.inp**

```text
! PBE0 DEF2-SVP DEF2/J RIJCOSX
%maxcore 2000
* xyzfile 0 1 ethanol_opt.xyz
```

**ethanol_opt.xyz（松弛几何，PySCF 输出原样）**

```text
ethanol relaxed geometry (33 frames in the trajectory, last one taken)
C          0.0832193644     0.0000000000    -0.0493137151
C          1.5728861274     0.0000000000     0.2824507948
O          1.7244180749     0.0000000000     1.6811420572
H          2.6651622790     0.0000000000     1.8821375882
H         -0.2195007516     0.8882419468    -0.6219328380
H         -0.4795152052     0.0000000000     0.8956361435
H         -0.2195007516    -0.8882419468    -0.6219328380
H          2.0608207187     0.8865380023    -0.1703710055
H          2.0608207187    -0.8865380023    -0.1703710055
```

**ethanol.xyz（盒内摆放，与两份算例的 coordinates: 一致）**

```text
C       1.58488900       2.21504500       0.77140300
C       0.87576400       0.88553400       1.01372200
O       0.44558800       0.84016400       2.35244900
H       0.00000000       0.00000000       2.49727600
H       2.63423000       2.08278400       0.47157500
H       1.57202000       2.78774800       1.71025700
H       1.08879700       2.82112700       0.00000000
H       1.56201300       0.04462600       0.78786300
H       0.01954500       0.78155200       0.31719300
```

一句话：**这些是"没有它就重跑不出乙醇拓扑"的部分，所以从目录逐字搬进了本文**；运行算例用的仍是 `pyrai3md/examples/ethanol.data`（连同 `.in.init` / `.in.settings`），`dev/` 不进包，`setup.py` 的 `package_data` 与它无关。
## 附录 A：势能定义与 DSL 草案（原 `CONSTRAINT_DEVELOPMENT_APPENDIX.md`，2026-09-27）

并入于 2026-10-02（CHANGELOG #289）。原文件是独立文档，其开头声明"设计决策、代码接入点与里程碑的权威表述以主文档为准；两者冲突时以主文档为准"，并入后这条优先级不变：以下 A.x 节是 2026-09-27 的设计推导与 legacy 行为细节，与正文 §1-§32 冲突处以正文为准。其中若干条已被正文 §3 的实测结论更正（文本解析器与 `&end` 行为、`freeze` 的实际作用范围、MD 阶段顺序与附加势接入位置、`System.image` 与 `fin_*` 所有权通道）。

日期：2026-09-27。状态：设计推导附录，由 `constraint_draft.md`（架构调研稿）更名而来。

本文现在是主文档的**附录 A**（下方），只保留势能数学定义、CV 与严格约束的推导约定、DSL 草案语法和 legacy 行为细节。设计决策、代码接入点与里程碑的权威表述以主文档为准；两者冲突时以主文档为准。本文对当前代码状态的描述基于 2026-09-27 的源码阅读，其中若干条已被主文档 §3 的实测结论更正（文本解析器与 `&end` 行为、`freeze` 的实际作用范围、MD 阶段顺序与附加势接入位置、`System.image` 与 `fin_*` 所有权通道）。开发完成后再与主文档合并。

本方案依据用户提供的 constraint DSL 设计、当前工作区源代码，以及仓库根的 `../../../legacy/PyRAI3MD/Molecule/constraint.py` 制定。代码事实优先于已有设计文档中的历史描述。

### 1. 目标与核心决定

建立一套后端无关的系统约束控制接口：接收几何、约束参数和必要的运行上下文，返回修正几何、分项附加势能、附加梯度，以及在已有电子结构结果基础上合成的总能量和总梯度。MD、优化、单点和后续采样共用同一套数值实现。

用户描述“选择什么、如何作用、参数和时间规则”；schema/normalize 完成选择展开、索引和单位转换；运行时只使用显式索引、数值参数和类型化对象。五种作用统一组织，算法分开实现：

| 类别 | 直接修改坐标 | 定义附加势能 | 所需运行状态 |
|---|---|---|---|
| `constraint` | 是，投影到约束流形 | 否 | 目标值、求解器诊断、MD 半步信息 |
| `fix` | 是，恢复冻结分量 | 否 | 初始锚点、自由度掩码 |
| `bias` | 否 | 是 | 静态参数或 schedule |
| `wall` | 否 | 是 | 几何、作用对象、schedule |
| `metadynamics` | 否 | 是 | CV/参考结构历史、沉积进度 |

必须保留以下区别：

1. soft bias/wall 不等于结构投影。输入结构在 wall 外时，一次能量评估只返回势能与梯度；需要结构松弛时由优化器迭代移动。
2. 若严格约束将 `R` 改成 `R*`，总能量必须使用 `E_base(R*)`。不能给旧几何的能量加一个“约束修正值”便声称它属于新几何。
3. `fix` 与 SHAKE 没有独立势能。其分项能量定义为零；约束反力、冲量、投影位移和初始化能量变化是独立诊断量。
4. 能量的真实导数、优化使用的投影梯度、积分器使用的驱动梯度分别命名。投影后的梯度一般不再是总势能对所有笛卡尔坐标的导数。
5. 全局约束在 composite/multi-center 结果组装后只加一次，不能分别加到 ONIOM 正负项或各中心电子能量上。
6. `kernel/` 是协调层，不放 SHAKE、wall 等数值算法；不恢复旧版 `Molecule.Constraint` 大类，不要求每个 skill 实现约束。

### 2. 当前架构调研与实际接入点

#### 2.1 输入与归一化

| 现有代码 | 已确认的行为 | 对新模块的要求 |
|---|---|---|
| [`interface/parser.py`](../../pyrai3md/interface/parser.py) `_parse_text` | `&section` 下按 `key value` 写入字典；单 token 行跳过；重复 key 覆盖；`&end` 没有专门的闭块语义 | 新 DSL 不能直接交给现有循环；必须增加专用块解析器 |
| [`schema/adapter.py`](../../pyrai3md/schema/adapter.py) | dotted keys 展开、分节 flatten、类型转换、索引转换；`NESTED_FRAMEWORK_SECTIONS` 保留部分分节 | 约束必须保留嵌套对象，不能让 `target/scale/groups/center` 污染扁平命名空间 |
| [`schema/config.py`](../../pyrai3md/schema/config.py) `System` | 归档里的 `freeze/constrain/cbond/cangle/cdihedral/tbond/cavity/groups/compress/...` 扁平字段 | 这些字段已从模型删除：写出来会被 `RETIRE_CONSTRAINT_KEYS` 按名拒绝并指明新拼法，功能一律由 `system.constraints` 块承担 |
| [`schema/normalize.py`](../../pyrai3md/schema/normalize.py) | `model_dump → flatten → derived quantities → partition → model_validate`；派生质量与晶胞 | 在几何、质量、晶胞就绪后执行约束归一化，并保护嵌套内容往返 |
| [`schema/validate.py`](../../pyrai3md/schema/validate.py) | 一致性与任务合法性检查 | 增加对象引用、能力、流程组合和约束冲突检查 |

坐标加载发生在 parser 构建 raw config 时，因而显式原子/元素选择可以在 normalize 中解析。采样派生的轨迹可能使用不同初始几何，`target initial`、冻结锚点等必须在每条轨迹第一次运行前绑定，不能沿用母结构坐标。restart 已经是运行态，不能重走输入归一化并重新捕获目标。

#### 2.2 计算与运行态

| 现有代码 | 已确认的行为 | 设计影响 |
|---|---|---|
| [`data/core.py`](../../pyrai3md/data/core.py) `PyRAIState` | 共享一套坐标/速度/质量；维护 `partition_plan/electronic/active_surface`；通过 `update_system/set_result` 更新 | 约束以全局原子为对象，不能每中心复制一套核状态 |
| `PyRAIState.active_gradient()` | 优先读取 `active_surface.active_gradient`，C=1 可回退到结果梯度 | 仅修改 `pmd.results['gradient']` 会漏掉多中心实际受力路径 |
| [`workflow/calculator.py`](../../pyrai3md/workflow/calculator.py) `run_in_process` | 执行各 task，组装/投影到全局原子，转换 stress，最后 `set_result` | 最后增加一次静态附加势评估；保留原始组装结果作为重算基准 |
| 同上，多中心分支 | 当前在 Calculator 内 `stage_evaluation` 后立即 `commit_step` | 注释中的“以后移至 MD commit”不是当前事实；约束接入必须调整此提交点或把同一事务扩展至修正结果 |
| [`workflow/multiscale/composite.py`](../../pyrai3md/workflow/multiscale/composite.py)、[`workflow/multiscale/regions.py`](../../pyrai3md/workflow/multiscale/regions.py) | composite 聚合、区域与 link-atom 梯度回映射 | 在这些步骤完成后施加全系统约束，不改变任务系数 |
| [`data/runtime/result.py`](../../pyrai3md/data/runtime/result.py) | 字典式 Result | 可承载结构化分项输出，但须明确键与数组 shape |
| [`data/runtime/serializer.py`](../../pyrai3md/data/runtime/serializer.py) | `.rt` 保存 PyRAIState；restart/remote 跳过输入管线；有版本与 plan 校验 | 保存已绑定目标、拓扑指纹和 MetaD 历史；不能只序列化输入参数 |

#### 2.3 MD、温控、输出与其他工作流

当前 [`workflow/md.py`](../../pyrai3md/workflow/md.py) 的实际顺序为：

```text
ensemble(pre) → verlet_i → apply_pbc → calculator.run
→ verlet_ii → phase_correction → apply_constrain（空操作）
→ velocity_correction → ensemble(post)
→ apply_constrain（constrain_after_cell）→ kinetic_energy
→ update_history → surface_hopping → verification → sync → checkpoint
```

关键接入问题：

- [`integrator/operations.py`](../../pyrai3md/workflow/integrator/operations.py) 的 `apply_constrain` 直接返回输入；在这个位置追加 bias 梯度已经晚于当前步第二次速度 kick。
- [`integrator/verlet.py`](../../pyrai3md/workflow/integrator/verlet.py) 的第一次 kick 读取历史帧 `nuclear_sample.active_gradient`，第二次读取当前 `state.active_gradient()`；两处都必须使用同一约束版本的受力。
- `velocity_correction` 对所有原子去 COM 平动并重缩放；原子数少于 2 时原样返回状态（单粒子没有内部自由度，去质心等于抹掉它的全部运动 —— Tully/Landau-Zener 模型轨迹正是这种形状）。它可能重新赋予冻结原子速度，不能简单放在一次 freeze 后继续使用。
- [`integrator/ensemble.py`](../../pyrai3md/workflow/integrator/ensemble.py) `_dof` 当前按 `3*(N-N_frozen)-3` 计算并钳制到至少 1；尚未扣除严格约束的独立秩，也未按实际操作判断是否应扣 COM。
- `PyRAIState.kinetic_energy()` 对全部质量/速度求和；加入约束后需明确自由速度与动能统计来源。
- [`workflow/records/output.py`](../../pyrai3md/workflow/records/output.py) 读取历史和中心结果输出能量，不能只改当前 state；[`utilities/print_results.py`](../../pyrai3md/utilities/print_results.py) 已有 `energy_contributions` 格式化入口，但目前期望扁平的数值字典。
- [`workflow/optimize.py`](../../pyrai3md/workflow/optimize.py) 与 [`optimizer/base.py`](../../pyrai3md/workflow/optimizer/base.py) 通过结果 E/G 更新几何；只把梯度清零不能保证 BFGS 耦合位移为零，必须约束位移和收敛判据。
- [`workflow/hessian.py`](../../pyrai3md/workflow/hessian.py) 对笛卡尔坐标位移后差分梯度；若自动投影位移点，算出的便不是普通笛卡尔 Hessian。
- [`workflow/scan.py`](../../pyrai3md/workflow/scan.py)、[`workflow/search.py`](../../pyrai3md/workflow/search.py) 可复用无历史势能评估；[`workflow/adaptive.py`](../../pyrai3md/workflow/adaptive.py)、prediction/train 的数据链要保留未加 bias 的训练标签。
- driver 不是 `main` 时，skill 自己做优化/动力学，框架无法在内部每一步插入约束；没有明确能力协议时应拒绝该组合。

#### 2.4 可复用工具及限制

- [`utilities/geometric.py`](../../pyrai3md/utilities/geometric.py) 有 `BND/AGL/DHD` 及导数，但角度使用 degree，且存在零键长、共线和 `acos` 边界处理不足。新 CV 接口用稳定的 radians 实现或有完整转换与检查的包装；不能直接假定已有函数满足新协议。
- [`utilities/lattice.py`](../../pyrai3md/utilities/lattice.py) 明确采用**行向量晶胞**：`R=S@B`，`S=R@inv(B)`。用户草案的列向量公式需转写，不能直接复制 `H^-1 r`。
- [`utilities/units.py`](../../pyrai3md/utilities/units.py) 提供统一常数。全局结构是 Å，质量 amu，MD 速度 Bohr/atomic-time，梯度 Hartree/Bohr；约束数值核采用原子单位并在边界转换。
- [`utilities/bonds.py`](../../pyrai3md/utilities/bonds.py) 提供 `bond_lib` 查询与未知键的 1.2 Å 兜底；键长表在 [`schema/reference.py::BOND_LENGTHS`](../../pyrai3md/schema/reference.py)，同处还有 `VDW_RADII` 与 `COVALENT_RADII`。这些是数据，不是一份完整的共价半径拓扑识别表。
- 当前源码 `data/` 不包含早期文档提到的 `data/structure`、`data/dynamics` 对象树，不以其作为新代码的接入基础。

### 3. 推荐代码布局与依赖方向

> **本节是 2026-09 草案的原始布局，已被实现取代。** 现行路径见主文档 §21（#271）：
> `interface/constraint_parser.py` 已删除，经典 `&constraint` 文本的读写并入 `schema/adapter.py`；
> `schema/constraint.py` 与 `schema/legacy_adapter.py` 已解散——记录模型并入 `schema/config.py`
> 的 `System` 区块，归一化与 `build_plan` 并入 `schema/normalize.py`，拒绝项并入
> `schema/validate.py`，legacy 适配并入 `schema/adapter.py`。下面的树保留原样，仅作设计过程记录。

新增模块采用现有目录层级，不创建第二套计算引擎：

```text
pyrai3md/
├── interface/constraint_parser.py   # 新增：DSL 块结构 → raw ordered records（#271 已删除该文件）
├── schema/
│   ├── constraint.py               # 新增：输入模型、规则、normalize 与 legacy 适配
│   ├── config.py                   # 扩展：System.constraints 嵌套配置
│   ├── adapter.py / normalize.py   # 扩展：保留嵌套对象、执行归一化
│   └── validate.py                 # 扩展：workflow/capability 校验
├── utilities/
│   ├── selection.py                # 新增：显式索引、范围、元素、group 解析
│   ├── topology.py                 # 不新建：邻接与连通分量复用 utilities/bonds.py::perceive_bonds
│   ├── cv.py                       # 新增：几何变量及坐标/晶胞 Jacobian
│   ├── (covalent_radii.py)         # 撤销：不新建第二张半径表（见 §5.1 更新）
│   └── lattice.py / units.py       # 复用，PBC 几何辅助函数按需扩展
├── data/
│   ├── topology.py                 # 新增：TopologyPlan/GroupTable 数值记录
│   ├── constraint.py               # 新增：ConstraintPlan/State/Result 数据记录
│   ├── core.py                     # 扩展：只读访问、原子化提交、历史快照
│   └── runtime/serializer.py       # 扩展：版本、恢复、复制隔离
├── workflow/
│   ├── constraints/
│   │   ├── controller.py          # SystemConstraintController，阶段化公共接口
│   │   ├── bias.py                # harmonic 与单侧 restraint
│   │   ├── wall.py                # sphere/ellipsoid/box/triclinic
│   │   ├── shake.py               # 严格位置约束与 RATTLE 速度约束
│   │   ├── projection.py          # fix 掩码、切空间投影、秩与 DOF
│   │   ├── schedule.py            # step/time → 数值参数，无全局时钟
│   │   └── metadynamics.py        # CV 与 RMSD 两类历史势
│   ├── calculator.py              # 电子结果组装后的一次性修正
│   ├── md.py / optimize.py        # 生命周期、接受/拒绝步、坐标提交
│   └── integrator/operations.py   # 薄适配器，委托阶段接口
└── qualification/verification/
    └── test_constraint_*.py        # 框架验证，无外部量化软件依赖
```

数值工具 `utilities/{selection,topology,cv,lattice}` 保持叶子依赖，仅依赖数值库和同层常数；不导入 schema、workflow、Maestro。数据记录不执行算法。schema 构造声明和静态计划；workflow 的 controller 消费数值计划。可执行 controller/缓存不存入 config 或 restart。

group、topology、CV 的解析与数学不私有化到 `workflow/constraints/`，以后区域选择、扫描和分析可直接复用。但普通 group 不等于 ONIOM center，也不自动改变 `highlevel/midlevel/partition_plan`。

首期仅创建实际需要的文件；不要为了最终目录树一次性添加空壳。`SystemConstraintController` 负责组合小组件，不包含所有算法实现。

### 4. 输入模型与 DSL

#### 4.1 唯一语义模型

正式配置入口推荐 `config.system.constraints: SystemConstraintConfig`，text 的 `&system_constraint` 是它的输入别名。JSON/YAML 对应 `system: {constraints: {...}}`，不新增含糊的顶层扁平关键字。

```text
SystemConstraintConfig
  version
  units
  groups[]
  cvs[]
  constraints[]
  biases[]
  walls[]
  fixes[]
  metadynamics[]
```

列表保留次序，每个作用项有唯一 `id`；未给 id 时按类型与出现次序产生稳定名称。严格模型拒绝未知字段。`cv` 记录引用解析后的对象 ID，不在每次积分中再次处理字符串。bias 可引用命名 CV，也可定义一个内联 CV，后者在 normalize 中转为同类内部对象。

#### 4.2 代表性输入（拟定语法，非当前可执行示例）

```text
&system_constraint
  units
    length angstrom
    angle degree
    energy hartree
  end

  group substrate
    atoms 1-12
  end

  fix
    id fixed_base
    group substrate
    direction xyz
    reference initial
  end

  constraint shake
    id xh
    bond_type X-H
    target initial
    tolerance 1e-8
    max_iterations 200
  end

  bias harmonic
    id stretch_13_14
    bond 13 14
    target 1.50
    force_constant 0.02
    region upper
  end

  wall sphere
    id confinement
    center 0 0 0
    radius 15.0
    target atom
    atoms 13-30
    power 2
    force_constant 0.001
    schedule linear
      parameter force_constant
      start_step 0
      end_step 10000
      scale 0 1
    end
  end

  cv r13_14
    bond 13 14
  end
  metadynamics gaussian
    id explore_bond
    cv r13_14
    height 0.0001
    width 0.10
    pace 100
  end
&end
```

实际结构须包含对应原子且避免固定自由度与 `bond_type` 展开产生不相容目标。数值仅演示语法，不是通用物理参数建议。首期未完成的块应报“不支持该能力”，不能解析后静默忽略。

嵌套 cell 示例使用项目的行向量约定：

```text
wall triclinic
  id solvent_boundary
  center 0 0 0
  cell
    20 0 0
    3 18 0
    1 2 20
  end
  target com
  group molecule
  power 4
  force_constant 0.001
end
```

`cell` 与 `lattice a b c alpha beta gamma` 二选一；此处是 wall 自身的边界，不自动开启系统 PBC，也不自动成为 NPT 晶胞。`group molecule` 表示拓扑连通分量的逐分子目标，不能把所有分子的原子合为单个 COM。

#### 4.3 解析与单位约定

- 新 parser 只解析语法结构、保留 token 与来源行号；数值、单位和物理检查在 schema 完成。
- 使用块栈匹配 `end`；`&end` 仅关闭最外层；未闭块、重复非列表属性、非法嵌套必须报告行号。多个 `wall/bias/bond` 记录不能被字典覆盖。
- 为新 section 单独分派解析，其余传统 section 保持原逻辑；新 section 内遇到下一个 `&section` 前必须闭合。禁止 `eval` 表达式。
- text、JSON、YAML、dotted JSON 都进入相同模型；更新 certification converter 的输入生成/反向输出，验证 round-trip，不把整个列表编码成不可解析字符串。
- 外部用户输入的原子索引统一 1-based，包括 JSON/YAML；归一化计划是 0-based。已类型化计划及 `.rt` 不再次减 1。
- 默认长度 Å、角度 degree、能量 Hartree；力常数的单位由对应势能公式推导。harmonic 的距离常数为 Hartree/Å²、角常数为 Hartree/degree²；Gaussian width 使用各 CV 单位。
- 转换采用 `q_internal=c*q_input`：`target_internal=c*target_input`，`width_internal=c*width_input`，`k_internal=k_input/c²`。例如 angle 的 `c=π/180`，距离的 `c=ANGSTROM_TO_BOHR`。
- wall 首版采用长度型越界量，`k` 的单位为 Hartree/Å^power；转为 Bohr 后除以 `c^power`。legacy 无量纲全空间势另有 convention 标记。

### 5. normalize、拓扑与参考绑定

归一化分两步，第二步不重新解释 DSL：

```text
parse → typed schema → 静态 normalize → ConstraintPlan
                             ↓
                    轨迹初始几何/质量/晶胞
                             ↓
                    bind_reference → ConstraintState
```

静态步骤依次为：单位与索引转换、解析 group 依赖、拓扑构建、`bond_type` 展开、CV 引用解析、参数/冲突检查、稳定排序和 fingerprint。运行绑定只捕获 `initial` 目标/锚点并建立每条轨迹独立历史。

#### 5.1 group 和拓扑

1. 选择支持 `atoms 1-20`、`element O H`、命名 group。一个选择器中元素集合表示并集，不意味着化学上完整的“水分子”；定义为 water 的 group 仍需拓扑或显式分子编号。
2. 明确区分单个 group 与 group 集合。首版 `atoms/element/group` 选择来源互斥；多来源组合以后以明确的 union/intersection 扩展，不能默认猜测。
3. `bond_type C-H/O-H/X-H` 仅匹配拓扑中已有的键。显式 `bond i j` 则允许指定不在拓扑内的几何距离。
4. 用户显式 bonds/molecule IDs 优先于距离推断；无拓扑时才用 `d_ij < scale*(Rcov_i+Rcov_j)` 推断，记录半径表版本、scale、来源与展开清单。未知元素缺少共价半径时要求显式拓扑，不沿用默认 1.2 Å。
   > **裁定更新（2026-09-28）**：本节设想的「新增 `utilities/covalent_radii.py` 共价半径表」已撤销。PBC 阶段 A+B 删除 `data/structure` 后框架不再有拓扑对象（`schema/config.py` 无 bonds 字段），但 `utilities/bonds.py::perceive_bonds`（`:76-129`）已提供一套按 `bond_lib × 1.2` 判键的感知函数，返回 `(i, j, delta)` 并携带该键所需的 image 跨越数；它只在有晶胞时工作、代价 O(N^2) 且 4096 原子以上拒绝，目前仅被 `schema/normalize.py::seed_initial_image` 消费。因此不再新建第二张半径表：将来实现 `bond_type X-H` 或按连通分量分组时，必须扩展并复用这同一个函数（含非周期分支与未知键元素显式报错，而不是沿用 1.2 Å 兜底），使 image、约束与区域对同一几何得到同一份键表。首版选择来源只有显式原子索引、显式 `bond i j` 与显式成员列表的命名 group。详见主文档 §3.8、§5。
5. 输出 bonds、neighbors、connected components；angles/dihedrals 按需生成，避免大系统无条件全枚举。标准化无向键并去重；重复约束相同目标合并，冲突目标报错。
6. group 成员保持稳定顺序。允许不同势项作用于重叠 group，其能量按定义相加；同一个逐分子目标内部不允许重复计原子。
7. 非反应性约束使用初始化拓扑，运行中不随瞬时键长重新识别 X-H；主动学习筛选出的新结构作为新任务时可以重新 normalize。只改变速度或刚体旋转的派生轨迹复用静态拓扑。
8. group 引用未知名称、环依赖、空展开、原子越界、非正质量和不相容锚点在初始化阶段失败，并列出来源。

#### 5.2 PBC 边界

`apply_pbc` 现在带账本：包裹时把每原子跨越的胞数累加进 `System.image`，坐标与计数由
`data/geometry.py::record_wrapped_geometry` 一次提交（`data/geometry.py::apply_pbc`，原
`workflow/integrator/operations.py`）。跨边界键、COM、RMSD 因此都从 `image` 重建连续坐标后再算
（`utilities/lattice.py::unwrap_coordinates` / `continuous_view`）。

**`System.image` 的写权只有一个入口，收的是「本批坐标的绝对计数」，不是增量。** 两个调用者各自
知道自己那份绝对值：`apply_pbc` 的包裹报的是相对当前帧的增量，所以它自己加上账本再提交；约束折叠
`images.fold_back` 报的已经是绝对值（其 docstring 明写重复相加会把每个跨过的面计两次）。把账本加到
折叠的绝对值上是错的，已实测：账本逐原子、折叠计数按连通分量一致，相加会让同一分子内的原子互相错开
一个胞，周期 + 严格键约束的 LAMMPS 跑在第 197/600 次求值死于 `Bond atom missing in image check`。
判据 `qualification/verification/test_constraint_periodic.py::TestTheLedgerHasOneWriter`（含把写入口
退回相加的负对照）。仍未裁定的是**账本的绝对水平**：折叠会把水平改写到它自己那一帧，跨帧差（GSH 曲率、
轨迹 unwrap）读到的分子位置因此可能整体平移整数个胞。

后续公共 PBC 工具应提供 image-aware displacement 和 molecular unwrapping；部分 PBC 只沿周期方向取像。一般三斜晶胞的最近像不能无条件等同于 fractional 分量逐个 `round`，要采用经验证的最近格点/邻像搜索，并拒绝无法唯一恢复分子的拓扑。

固定晶胞的局域 bond/angle 可使用一致的键像；COM/COG 与 Cartesian RMSD 必须先重建完整分子。跨越整个周期盒的无限连通网络不定义普通“分子 COM”，应要求显式局域 group 或禁止该作用项。初始化确定像关系并在轨迹中连续维护；restart 保存 image 状态。

首个版本没有这些实现时，只开放非周期体系；遇到 PBC 约束明确拒绝，不能称为“已支持三斜 PBC”。非周期空间的 triclinic wall 可以独立实现。

### 6. 数据契约与公共接口

#### 6.1 三类持久数据

| 对象 | 所有者 | 内容 |
|---|---|---|
| `ConstraintPlan` | `PyRAIState.constraint_plan`，不可变 | 显式原子、CV、规范参数、group/topology 引用、单位/语义版本、fingerprint |
| `ConstraintState` | 每个 PyRAIState 独有 | initial targets、fix anchors、reference cell、MetaD hills/结构、last committed step、history version、image 状态 |
| `ConstraintResult` | 单次求值返回，提交到 results | 几何、逐项 E/G、总修正、投影量、DOF/残差等诊断、几何与历史版本 |

`config.system.constraints` 保存规范化配置；运行计划是其派生快照，不可独立编辑。计划变化必须显式重建并改变 fingerprint。schema 不存可执行对象；控制器可由 plan+state 重建。

#### 6.2 数值返回值

建议 `ConstraintResult` 至少包含：

```text
coordinates                 (N,3), Å，修正后的框架几何
velocities                  (N,3), Bohr/atomic-time，可选
delta_coordinates           (N,3), Å
energy_correction           scalar, Hartree
gradient_correction         (N,3), Hartree/Bohr，附加势的真实导数
components[id]              {kind, energy, gradient?, cv_values, parameters}
energy_base / energy_total  (S,) 或全局 active scalar；未给 base 时为 None
gradient_base / gradient_total
                            (S,N,3) 或全局 active (N,3)
gradient_projected          同梯度 shape，可选，优化/诊断使用
gradient_for_integrator     (N,3)，见第 9 节
constraint_impulse          (N,3)，MD 时可选，不冒充势能梯度
stress_correction           (3,3), Hartree/Bohr^3，可选
diagnostics                 converged, iterations, residual, rank, free_dof
provenance                  geometry_id, evaluation_id, plan_fingerprint,
                            step, time, history_version, schedule_version
```

所有已启用项保留唯一 id。常规记录每项能量/CV/参数，逐项梯度按详细输出请求保存，避免 `项数×N×3` 的历史内存膨胀；总梯度始终可得。没有能量/梯度的 base 不制造零数组冒充已计算结果。

物理定义为：

\[
V_{add}=V_{bias}+V_{wall}+V_{metad},\quad
G_{add}=\nabla_R V_{add},\quad
E_{total}=E_{base}+V_{add},\quad G_{total}=G_{base}+G_{add}.
\]

base 已包含电子计算及 composite 贡献；constraint breakdown 不重复包含这些项。`energy_contributions` 输出扁平键（如 `constraint.bias.stretch_13_14`），详细结构写入 `results['constraint']`。若同时展示小计和总计，应标记层级，禁止消费者把所有行再次求和。

#### 6.3 分阶段 API（提案）

```python
controller = SystemConstraintController(plan)
runtime = controller.bind_reference(initial_geometry, masses, cell)

projection = controller.project_positions(candidate, runtime, context)
potential = controller.evaluate_potentials(projection.geometry, runtime, context)
corrected = controller.combine(base_result, potential, projection, context)
velocity = controller.project_velocities(geometry, velocity, runtime, context)
stats = controller.statistics(geometry, velocity, runtime, context)
runtime_next = controller.commit(accepted_frame, runtime, context)
```

`context` 至少包含求值目的（single/md/optimization_trial/hessian）、物理时间、接受步号、dt、evaluation ID、晶胞及历史版本；函数不得读取模块级步数。`evaluate_potentials` 是纯求值，不沉积 hill；只有接受步可 `commit`。

为用户要求的一次性处理提供 `apply_structure(...)` 便捷接口：先投影几何，再调用可选 evaluator 获取该几何的 base E/G，最后合成。未提供 evaluator 时只返回几何与约束附加 E/G；提供旧几何 base 时必须检查 geometry ID，一旦几何改变即拒绝合成或重新求值。

同一个几何、plan、schedule、history 版本的重复评估得到相同结果。每次从保存的 base 合成，不在上一次 total 上使用 `+=`。失败时不提交部分坐标、结果或历史；大位移投影超限应报告而非继续积分。

### 7. 各类势能和严格约束的数学定义

以下是拟采用的数值约定；它们不要求调用外部约束软件。

#### 7.1 Harmonic bias 与区域

统一使用 `V=1/2*k*δ²`。`δ=q-q0`；`upper` 用 `max(δ,0)`，`lower` 用 `min(δ,0)`，`all` 用原 δ；梯度通过 `dV/dq * dq/dR` 计算。边界 q=q0 时梯度为零。

二面角差使用 `atan2(sin(q-q0), cos(q-q0))`，按周期最短差处理。恰好相差 π 是此形式的分支不可微点，应记录并避免在该点做差分验收；需要全局光滑周期势时另设 `periodic_cosine`，不悄悄改变 harmonic 定义。首版 dihedral 仅支持 `region all`，单侧周期区间必须以后明确定义原点和方向。

每个 bias 只绑定一个标量 CV；多个 CV 独立 restraint 使用多个块。`inside/outside` interval 留待扩展，首版不接受未定义含义的输入。

#### 7.2 Wall：几何、目标、势形分开

首版 wall 采用外侧才生效的 `V=k*max(d,0)^p`，`p≥2`；无 `1/2` 因子。`d` 是明确声明的长度型越界量：

| 几何 | 定义 |
|---|---|
| sphere | `d=norm(r-center)-radius` |
| ellipsoid | 半轴 `a`；`ρ=sqrt(sum((x/a)^2))`，`d=min(a)*(ρ-1)`，这是一种尺度化隐式距离，并非精确欧氏最短距离 |
| box | `length` 是完整边长，半边长 `h=length/2`；对各轴 `d_i=abs(x_i)-h_i` 的单侧势求和 |
| triclinic | 行晶胞 B，`s=x@inv(B)`，以中心为原点、边界 `s_i=±1/2`；`d_i=(abs(s_i)-1/2)/norm(inv(B)[:,i])`，对面越界项求和 |

box 是对角 B 的特例；triclinic 的分母保证 `d_i` 是到相应面的法向距离，避免不同晶胞尺度改变 k 的物理意义。棱角处多个面的势相加。轴长度/半径必须正，cell 必须有限且可逆。中心点与边界点设置稳定导数分支。

`target atom` 逐原子求和；`target com` 对每个 group 的 COM 求值，再以 `m_i/sum(m)` 回传梯度；`target cog` 以 `1/N` 回传。不能按 group 原子数重复计算同一质心能量。

首版 center 为固定 Cartesian 点，或初始化 group 的固定 COM/COG。后续若支持跟随 group 移动的 center，需要同时对 wall 目标与 center group 求导，包括重叠 group，不能将动态 center 当常数。

#### 7.3 Fix

规范化得到 `(N,3)` 布尔掩码；`direction xyz/x/y/z/xy/...` 选择冻结分量。reference 初次绑定后固定。每次 drift 或优化位移后恢复锚点，速度相应分量为零；数值求解中把冻结分量移出活动变量，不通过设质量为无穷大实现。

保留完整 `gradient_total` 供能量差分和锚点反力分析。另返回清零冻结分量的驱动/投影梯度。重叠 fix 按分量去重，锚点不相容报错。首版锚点在 Cartesian 空间，variable-cell 下的 fractional freeze 是另一种能力。

#### 7.4 SHAKE/RATTLE

> **范围更新（2026-09-28，代价已于 2026-09-29 实测）。** 影子构建只实现 SHAKE 的位置求解，RATTLE 的速度约束不在本轮范围。本节对可逆性与 `Jv = 0` 的论述保留为后续集成阶段的要求，不是当前实现的描述。实测结论见主文档 §1.3：位置求解把键长守到 1e-12，但位置-only 方案的总能量漂移不随 dt 下降（dt、dt/2、dt/4 分别约为驱动势阱深度的 117、155、187 倍），`Jv` 也停在 1.2e-2 量级不收敛。所以本节"第二次 kick 后令 `J(x)v=0`"的要求不是留待以后再议，而是已被量化为必须补上的一步；其接入形状定为约束记录上的关键词切换 `velocity_constraint: none | rattle`（默认 `none`，位置通路一行不改）。

首批严格约束支持 bond；angle 作为后续能力，dihedral 再后续。schema 可认识这些类型，但未实现算法时必须拒绝执行，不能把“语法预留”称为支持。

对于活动坐标 x 和约束 `c(x)=0`，令 `J=dc/dx`，`M` 为活动质量矩阵。初始化几何投影可迭代使用：

\[
\Delta x=-M^{-1}J^T(JM^{-1}J^T)^+c(x).
\]

这是初始结构/试探结构的局部投影，不可直接宣称它就是可逆 SHAKE 积分。MD 位置校正必须使用上一步几何、半步速度和离散约束冲量的 SHAKE/RATTLE 推导，并同步校正半步速度；第二次 kick 后令 `J(x)v=0`。velocity-Verlet 的速度约束需求见 [GROMACS constraint algorithms](https://manual.gromacs.org/2024.2/reference-manual/algorithms/constraint-algorithms.html)。

冻结分量先移出 J。独立秩不足时识别重复/冗余约束，报告不相容目标；非线性几何退化不能仅靠伪逆掩盖。不同类型的残差按其自然尺度归一化，`tolerance` 表示无量纲相对残差，另记录 Å/radian 下的物理残差。

位置约束失败返回最大残差、项 id、迭代数；禁止自动改成软弹簧。固定两原子的 bond 若目标等于锚点距离则是冗余约束，不再扣 DOF；不同则立即失败。

#### 7.5 Schedule

首版 `linear` 采用绝对接受步号，端点外夹持到端点值；`end_step>start_step`。`scale s0 s1` 乘初始参数，`value v0 v1` 是绝对参数值，两者互斥。重复评估同一步不会推进 schedule。

只开放实际有实现的参数（首先 force_constant、radius）；半径全程必须正。时间型 schedule 显式声明 time 单位并用物理时间，不混用 timestep 计数。严格约束目标移动涉及 `Jv+∂c/∂t=0`，不纳入首版 schedule。

时间依赖势记录 `∂V/∂t` 或按明确离散协议积分的外功；此时 `K+E_total` 不作为静态 NVE 守恒量。schedule 改变势参数造成的能量变化与数值漂移分开输出。

#### 7.6 Conventional CV MetaD

固定历史下：

\[
V(s)=\sum_h w_h\exp\left[-\frac12\sum_k
  \left(\frac{\Delta(s_k,s_{h,k})}{\sigma_{h,k}}\right)^2\right].
\]

每个 width 对应一个 CV，`height>0`、`width>0`、`pace` 为正整数；周期 CV 用周期差并处理分支。以 `dV/ds` 乘 CV Jacobian 回传梯度。CV 与 Gaussian 分离的参数组织参考 [PLUMED METAD](https://www.plumed.org/doc-v2.9/user-doc/html/_m_e_t_a_d.html)。

首版用普通 Gaussian hills；well-tempered 是后续独立模式。沉积在接受步末尾执行，初始 step 0 默认不沉积，首个 hill 在完成 pace 步后产生。力评估只使用提交前历史 `H_n`，新 hill 从下一步生效。

由于 Verlet 第一半步会读取历史力，提交新 hill 后应在同一终点几何用 `H_(n+1)` 重新评估附加力，生成独立 `next_force` continuation；不要重写描述本步积分的 pre-deposition 力。若使用历史截断/淘汰，也按同一规则重建下一步力。记录本次沉积造成的势能跳变。

#### 7.7 RMSD/内部坐标结构历史势

`metadynamics rmsd` 是独立 history backend，保存参考结构，不硬塞成“一个固定参考 RMSD 的 conventional MetaD”。

Cartesian 模式明确选择质量权重或等权，去平移及最优刚体旋转；使用最优对齐的 RMSD 定义，参考 [PLUMED RMSD](https://www.plumed.org/doc-v2.9/user-doc/html/_r_m_s_d.html)。势可直接使用平方距离 `D²` 构建 `w*exp(-D²/(2σ²))`，避免在 D=0 对 RMSD 开方求导。原子一一对应，不默认对相同元素做排列最小化；对齐退化和镜像处理需要专门验证。没有对照验证时不声称与 xTB/Grimme 实现数值等价。

内部坐标模式定义无量纲 `D²=sum_k w_k*(Δq_k/scale_k)²`，长度和角度必须各有尺度，二面角用周期差。多 group 必须明确是多个独立历史势，还是一个组合距离；第一版默认多个独立项，不隐式拼接。

`history all` 保存全部参考；有限历史使用明确字段 `history_stride/max_references`，不沿用含义不明的 `history 100 5000`。截断会改变势能，记录淘汰事件。评估复杂度随参考数增长，后续才考虑 neighbor list/压缩。

#### 7.8 Lattice CV 与 stress

保留 `lattice a/b/c/alpha/beta/gamma` 与明确的 strain CV 类型；`a,b,c,角度` 与六个 strain 分量之间不是简单更名转换。保留 CV 定义及其 `dq/dB` 链式导数。`lattice xy` 本身有歧义，应要求 `cell_component xy` 或 `strain xy`，并规定 reference cell、坐标系和 shear convention。

`dq/dB` 与 Cartesian gradient 是不同通道；若 CV 仅依赖 cell，其原子梯度可为零，但晶胞驱动力不能丢失。stress 沿用现有拉伸为正、`P_int=P_kin-stress` 约定，在固定 fractional 坐标的形变路径上验证 `dV/dstrain / volume`；外墙是否随 cell 变形也必须明确。

当前已有 NPT 实现不意味着约束 NPT 自动成立。hard-constraint impulse virial、wall/CV cell 导数、barostat 子步骤后的再约束以及晶胞重新定向都通过验证前，拒绝相应 NPT/optimize_cell 组合。固定外部 wall 的 B 与系统 cell 不得混用。

### 8. 全局能量、梯度与电子态隔离

#### 8.1 默认态无关附加势

首版所有附加势均 state-independent。C=1 的可用态：`E_total[s]=E_base[s]+V_add`；仅对实际计算出的梯度行加 `G_add`，缺失态梯度仍缺失。各态同加标量不改变能隙。

C>1 不制造笛卡尔乘积态张量；`active_energy_total=active_energy_base+V_add`，全局 `(N,3)` 梯度同理。ONIOM 展开中的 baseline、high/low correction 和 link-atom 贡献保持原样，约束不乘 task 系数，不复制进各中心局域电子阵列。

`results['energy'] / ['gradient']` 对外表达已合成的物理势能和真实梯度；同时保留 `base_energy/base_gradient` 及原电子 carrier。`active_surface` 保留 base、total 与积分器驱动量，并通过一次提交保持一致。C=1 电子传播也必须明确读取原 electronic/base 视图，不能因回退到 corrected result 而改变 GSH 的梯度语义。

FSSH/GSH/LZSH 的 NAC、SOC、密度矩阵不被约束模块改写。外力会改变真实核轨迹，但不能把 projected gradient 当成电子耦合算法的原电子梯度。跃迁重标速度的方向必须限定在可行速度切空间，再求满足能隙的幅度；事后投影可能破坏能量守恒，所以未完成该适配前禁止 hard constraints + surface hopping。

#### 8.2 合成位置、幂等与缓存

推荐所有框架自管的 Calculator 路径使用共同的结果 finalizer：

```text
已投影并锁定的几何/晶胞快照
→ backend tasks → global scatter/composite
→ base 单位统一 → 保存 base
→ evaluate_potentials(snapshot, frozen_history, context)
→ combine → 一次性提交 results + active_surface
```

static single-point、MD、优化共享此 finalizer；MD/优化负责调用前的位置投影。普通单点由最外层全局 Calculator 入口执行一次 prepare；内部 term worker 不投影、不加偏置。每条执行路径携带“全局评估/子任务”标识，避免本地、subprocess、Slurm 路径重复作用。

当前 Optimize 仍有 manifest 驱动的梯度转换。实施时先把全局 base E/G 明确转换到统一单位再加约束，调整已有消费者防止二次转换；不能把 Bohr 梯度加到尚未转换的 skill 梯度上。该单位审计是首期必要工作，不由每个势函数自行猜测。

base 缓存只保存物理计算结果；附加势缓存键包含 geometry/cell、plan fingerprint、history version、schedule step、求值目的。相同坐标而时间/历史不同不等价。任务重试、重复日志、theory/reference 切换不能触发重复加势或 hill 沉积。

### 9. MD 与统计接入

#### 9.1 无历史静态约束的阶段顺序

初始化时绑定 reference，投影初始几何及速度，计算首个 E/G；当前“无历史首步不推进坐标”的语义保留。其余步骤建议为：

```text
1. ensemble(pre)，保证输入速度位于可行空间
2. verlet_i：旧的受力 kick + drift
3. fix + SHAKE 位置校正，并同步半步速度/冲量
4. PBC 表示更新（保持 image 连续性），冻结本步几何/晶胞快照
5. Calculator：base E/G + 本步附加势，提交 total/drive 梯度
6. verlet_ii：当前力 kick
7. RATTLE + fix 速度约束；phase correction 保持电子职责
8. 约束兼容的 COM 操作、ensemble(post)，必要时再次速度投影
9. 重新计算 K_free、DOF、温度和本步诊断；保存 pre-hop frame
10. surface hopping（仅在兼容能力通过后），核对可行速度和能量
11. 接受步提交：MetaD/schedule continuation、sync、checkpoint
```

严格 bond 不用 `gradient_projected` 替代 RATTLE 的拉格朗日冲量；积分器驱动力是总势能梯度，冻结分量可屏蔽，严格约束由位置/速度算法处理。历史 `active_gradient` 存该积分器驱动力，另外保存物理 `gradient_total` 供分析。

凡改动速度的操作均需遵守可行空间：温控、COM 修正、跃迁重标及初始采样。均匀标量缩放在静态齐次速度约束下可保持切空间；非均匀操作必须检查/投影。不能沿用“冻住 → 去 COM → 冻住”却仍按旧动能进行温控。

若本步最终坐标发生超出约定的改变，之前计算的 E/G 作废；不能只更新输出坐标。未来 cell 缩放与再三角化还需一致变换锚点/reference frame，暂未实现时在 validation 拒绝。

#### 9.2 自由度和温度

对自由 Cartesian 分量建立质量加权约束矩阵，将 hard constraints 和实际启用的动量/转动移除条件联合计秩：

\[
N_{free}=N_{mobile\ components}-rank(A),\quad
K_{free}=\frac12 v_{free}^T M v_{free},\quad
T=\frac{2K_{free}}{N_{free}k_B}.
\]

其中 A 是在 fix 掩码后构造的独立线性速度条件（包含 J）。不能简单用“约束条数 + 3”重复扣减；bias/wall/MetaD 不直接减少 DOF。去 COM/转动仅在流程明确启用且物理上合适时计入；外部锚点/外墙破坏相应对称性时不自动假定这些自由度可删除。

例：三原子水 9 DOF，三条独立刚性几何约束后为 6；若另实际去掉整体平动再为 3。完全冻结的体系为 0：静态求值允许，温控 MD 拒绝，温度输出 N/A，不能 clamp 成 1。

`ConstraintStatistics` 成为统一 DOF/自由动能来源；替换 ensemble `_dof` 并供 `kinetic_energy`、日志、MdOutputWriter 使用，禁止每处单独猜测 SHAKE/fix 数量。大体系按约束连通分量计算秩，不要求每步对全系统稠密矩阵 SVD。

### 10. 优化、单点、Hessian 与学习数据

| 工作流 | 支持规则与接入 |
|---|---|
| 几何处理 API | 无 backend 时提供投影几何及附加 E/G；有 evaluator 时提供同几何总 E/G |
| framework 单点 | 先 prepare，再锁定几何并评估；输出 geometry ID 与完整分项 |
| minima 优化 | 用 corrected energy、可行方向梯度；fix 位移掩码；strict constraint 后续采用投影步/retraction 与约束收敛判据 |
| scan/search | 每个候选独立绑定或显式继承 reference；不能每次试探重新定义 initial target |
| numerical Hessian | 固定 schedule/MetaD 历史，差分真实 `gradient_total`；首期仅无 hard constraints 的静态势；有冻结/流形约束时另定义 reduced Hessian |
| analytic Hessian/frequency | 必须提供附加势二阶导数或明确转数值路线，否则拒绝“带偏置的总频率”请求 |
| adaptive/prediction/train | 训练标签与模型偏差来自 base 电子结果，不能把 wall/MetaD 污染为电子结构学习目标 |
| skill 内部 MD/优化 | 未有经验证的委托协议时拒绝框架约束；不能仅在终点补修正 |

优化的能量、梯度、坐标和收敛报告必须属于同一已评估几何。当前循环是先评估再移动几何、随后检查收敛；接入时调整接受步与最终输出顺序，避免把旧 E/G 与新结构配对。初期不声称支持全部 MECI/MECP/TS/NEB，按方法逐一增加能力。

### 11. 重启、并行、结果输出

- restart 保存 plan 与语义版本、reference targets/anchors、拓扑/原子顺序 fingerprint、MetaD 完整 hills 或参考、history version、绝对步号/时间、last deposition token、PBC images 与 next-force continuation。
- `.rt` 恢复不得根据当前结构重新推断成键或 initial targets。缺少历史而输入要求继续 MetaD 时报错；旧无约束 runtime 可迁移为空计划。
- 状态复制必须深复制可变历史；多条轨迹只能共享不可变 plan，不共享 hills、step counter 或 NumPy 可写缓冲区。批量试探求值读取冻结历史，worker 不沉积。
- 沉积 token 采用 `(trajectory_id, accepted_step, item_id)` 等稳定标识，重试与恢复不会重复沉积；checkpoint 在完整接受步提交后保存。控制器实例和临时求解缓存不持久化。
- 不把全部 MetaD 历史复制到每个 HistoryFrame。帧保存 history version、分项能量、CV、残差与受力摘要；完整历史只由 ConstraintState 保存。
- 为每次评估输出 `R*`、`E_base`、分项附加能量、`E_total`；详细模式输出 G_base/G_add/G_total/G_projected。MD 输出额外包含 DOF、K_free、约束误差、schedule work、deposition jump；不把偏置能量标作纯 electronic energy。
- 实现时扩展 `data/core.py::to_dict`、serializer prepare/restore/copy、HistoryFrame 和输出消费者。`set_result` 当前仅对顶层 ndarray 转 list，新嵌套 components 需递归序列化，不能留下不可 JSON 化的数组。

### 12. Legacy 行为迁移与不可照搬之处

参考文件实际导入名为 `PyRAI2MD`，使用 `traj.coord/energy/grad` 就地更新；它提供 bond/angle/dihedral harmonic、polynomial cavity、freeze gradient 和 GeomTracker，没有 SHAKE/RATTLE 或 MetaD。

| legacy 输入/行为 | 新设计 |
|---|---|
| `cbond/cangle/cdihedral + t* + f*` | 转成显式 harmonic bias；legacy `V=f*δ²` 转为新 convention `k=2f` 后再换单位 |
| target 为空则第一次调用捕获几何值 | 改为显式 `reference/target initial` 并持久化；重启不重捕获 |
| `_match_list` 最后值复制、过长截断 | 新输入严格检查长度；legacy 适配若保留广播，必须记录展开结果；不静默丢弃多余目标 |
| `freeze` 仅清梯度 | 转为 fix，同时维护坐标、速度、位移掩码和 DOF |
| `constrain` | 是 cavity 的作用原子，不是 SHAKE 列表，不能按名称误迁移 |
| `cavity/shape/factor/scale` | legacy 是全空间 polynomial，内部也有非零势；迁移为明确的 `legacy_polynomial` 模式，不能换成 outside-only wall |
| `groups [[nmol,natom],...]` | 按旧选择原子顺序展开显式组，检查总原子数；新 API 不使用压缩计数形式 |
| cavity group 能量 | 旧实现中质心中间量按成员重复存储，应测试其按组大小加权的实际行为；兼容模式显式保留该权重，新 COM wall 每组只计一次 |
| `center_type` 与 `record_center` | 旧 center 首次捕获后固定；迁移为 fixed reference center，不误解释为每步追踪 COM |
| `compress` | 转为 radius/axis schedule，明确初始因子、端点及步号含义 |
| hardcoded `0.529177249` | 使用 `utilities.units` 唯一常数 |
| 二面角差手工折返、`exit()` | 周期差 + 类型化异常/诊断；失败不终止解释器进程 |
| `GeomTracker` | 属于几何监测/停止条件，复用公共 CV；不混入势能或 hard constraint |

legacy 与新式约束若同时提供，首版拒绝混用，提示先迁移，避免重复加势；不能将 System 的默认 `fbond/shape/scale` 误当成用户启用了旧约束。只有非空选择/作用定义激活 legacy 路径。

与 legacy 对照的目标是明确哪些行为保留、哪些数值问题修正，不以逐行移植为目标。当前 PyRAI3MD 的约束运行入口本来是 no-op，因此旧文件的数值兼容需要独立 golden cases，而不是假设已有轨迹已实现同样的约束。

### 13. 实施阶段与验收门槛

| 阶段 | 交付范围 | 进入下一阶段前的门槛 |
|---|---|---|
| C0：契约与输入 | DSL parser、严格 schema、nested routing、units、显式 groups/CV、空计划、legacy 适配规范 | text/JSON/YAML 一致；重复块不丢失；未支持能力明确失败 |
| C1：可独立使用的静态模块 | fix、bond/angle/dihedral harmonic、四种 wall、COM/COG、分项 E/G、单结构 API、Calculator 合成、单中心/多中心 | 单位和梯度差分通过；结构与 E/G 同源；重复调用不累加；无约束行为不变 |
| C2：固定晶胞严格约束 | topology/bond_type、bond SHAKE/RATTLE、MD stage 接入、DOF、温控、fix 优化支持 | 非周期 NVE/NVT 长程残差和守恒通过；初始化/restart/双轨迹隔离通过 |
| C3：PBC 与高级投影 | image-aware CV/group、角约束、约束优化、受约束跃迁重标、预设 schedule | 三斜跨边界连续性；可行空间与跃迁能量同时成立；不支持组合继续拒绝 |
| C4：历史势 | CV Gaussian MetaD、Cartesian/internal RMSD、history commit/continuation、恢复与有限历史 | 同一步不重复沉积；终点新历史的下一步力正确；重启等价 |
| C5：晶胞耦合与扩展 | cell/strain CV、constraint virial、NPT、variable-cell freeze、well-tempered 等 | cell/strain 差分、压力符号、extended-energy 与分步耦合验证通过 |

阶段是实施依赖，不是不同输入体系。C1 解决“给定几何和参数得到结构、修正能量各组分、修正梯度”的直接需求；C2 起扩展严格约束的动力学一致性。若用户仅需要 soft restraint，不应被迫等待 MetaD/NPT。

实施中的主要改动清单：

1. `parser + adapter + config + normalize + validate + certification/converter`：语法到唯一配置。
2. 公共选择/拓扑/CV 与 constraint 数据模型：规范参数到数值计划。
3. controller 和势函数：纯求值到可检查的分项结果。
4. Calculator finalizer 与 state 提交：全局只合成一次，覆盖 C=1/C>1、远程执行和 base/total 区分。
5. MD/optimizer/ensemble/output/serializer：阶段顺序、DOF、接受步和恢复。
6. 同步更新 `dev/ARCHITECTURE.md`、`dev/INPUT_SYSTEM_DEVELOPMENT.md`、`dev/methods/MD_DEVELOPMENT.md`、`dev/methods/ENSEMBLE_DEVELOPMENT.md`、`dev/methods/PBC_DEVELOPMENT.md`、`dev/methods/OPTIMIZE_DEVELOPMENT.md`，消除与旧 placeholder 顺序的冲突。具体删改项与所属里程碑见主文档 [CONSTRAINT_DEVELOPMENT.md](./CONSTRAINT_DEVELOPMENT.md) 的 Document Update Requirements 一节。

### 14. 验证计划

测试放在 `qualification/verification/test_constraint_*.py`，现有 planner 自动发现。以解析式小模型/dummy backend 为主，无需真实量化软件验证纯约束数学。

| 测试组 | 必须覆盖的行为 |
|---|---|
| parser/schema | 重复 wall、多个 bond、嵌套 cell/schedule、未闭块、重复 id、拼写错误、未知/循环 group、文本与结构化输入往返 |
| index/unit | 1→0 仅一次；Å/Bohr、degree/radian、Hartree/其他显式能量单位的等价输入；k 的二次/幂次单位换算 |
| CV/bias | 随机非奇异结构中心差分；键零长度、角共线明确失败；二面角分支两侧；upper/lower 边界；整体平移/旋转性质 |
| wall | 内部零值、边界零梯度、外部值；box 与对角 triclinic 等价；COM 非等质量梯度；COG 均分；多 wall 相加；group 能量不重复 |
| projection | freeze 分量坐标/速度不动；相容/冲突目标；重复约束秩；bond 残差和 RATTLE `Jv`；无质量或非正质量报错 |
| composition | base+sum(components)=total；C=1/C>1、ONIOM 正负项、局域到全局 scatter；缺失态梯度；theory/reference 与重复求值不累加 |
| integration | 原 apply_constrain 位置无法产生正确 kick 的回归例；历史/当前两次 kick 均含附加力；COM/温控不解冻；无约束路径不变 |
| statistics | 刚性三原子 9−3=6、固定掩码与 hard constraint 联合秩、零 DOF、按实际 COM 操作扣减；日志与 thermostat 同源 |
| MD physics | 静态势 NVE 能量误差随 dt 减小改善；约束误差受 tolerance 控制；时间势外功/沉积能量跳变单独统计 |
| MetaD | 固定历史 E/G 差分；pace 与 step0；拒绝步不沉积；重复 commit 幂等；新 hill/淘汰后 next_force；多轨迹隔离 |
| restart/remote | 连续跑与中途保存恢复一致；plan/atom order 不匹配失败；subprocess/Slurm payload 保留约束；worker 不重复加势 |
| workflow | 优化最终结构与 E/G 配对；数值 Hessian 固定历史；学习标签不含 bias；不支持 skill 内部优化/NPT 组合提前报错 |
| legacy | `k=2f`、initial target、固定 center、压缩端点、polynomial 内部非零、group multiplicity 与新语义的明确差异 |

平滑、非奇异测试点的初始梯度门槛建议 `atol=1e-7 Hartree/Bohr, rtol=1e-5`，配合多个差分步长检查收敛；刚性投影测试以用户 tolerance 和规范化残差判断。长程 MD 不用单一绝对阈值替代 dt/tolerance 收敛验证。

本节列出的测试均未实现，也不声称任何一项已通过。实施与验收判据以主文档的里程碑表和验证矩阵为准。


---

## 附录 B：轮次计划与对账（原 `dev/constraint_next.md`）

并入于 2026-10-02。B.x 的编号沿用原文件的 §x，因为正文、代码注释与出货样例的头部注释里有多处以 "constraint_next section N" 形式指向它；指向本文时写作 `dev/methods/CONSTRAINT_DEVELOPMENT.md` 附录 B section N。原文件里 §11 出现两次（#271 一轮与 #281-#285 一轮），编号照原样保留，以免历史引用错位。

**原文件的标题**：约束子系统下一阶段：bond / angle / torsion 的像感知功能，再到 wall

（Last updated: 2026-10-01 22:40，即 #286/#287 两轮之前的状态）

**事实底稿在 `dev/ARCHITECTURE.md` §3.7**（接入后的十个调用点、三条运行通路、默认闭合的门在哪、
以及这张图暴露的 D-16 / D-17）。动某个函数前先去那张表找它的上游与调用频率，再决定要不要改。

本文是**开工前的顺序与出口判据**，不是设计权威。设计权威仍是
`dev/methods/CONSTRAINT_DEVELOPMENT.md`（下称主文档）；本文每条都指回它的章节或代码行。
现状基线：本文的 M0 / M1 / M3-A / M4 已全部落地（CHANGELOG #266）；#274 修好本机 LAMMPS 并把两套引擎各跑一遍（非周期 9 例 + 周期 7 例），周期放行改为按变量、ramp 壁的成员检查改为只看声明几何；
#275 又补齐周期矩阵剩下的组合（ellipsoid / box / triclinic 三种 wall 几何、gaussian 与 rmsd 两种历史势、
带胞的两段进程续跑，共 7 例走子进程通道），并修好 D-22（壁内在规则每步重问）与 D-23（rmsd 读不存在的
`spec.cvs`）、登记 D-24（续跑改变记录的帧节奏，未修）；通道归属按产物更正：#274 那批其实是常驻服务；第二轮又补齐了扫描出来的拓扑键图、well-tempered、多条件 RATTLE、六参数晶胞与 D-04 / D-16 / D-17（#267）。剩下什么、分别被什么挡住，见本文 §10。本文保留作为开工顺序与出口判据的记录，设计权威仍是 `dev/methods/CONSTRAINT_DEVELOPMENT.md`。

---

### 0. 裁定顺序与理由

用户裁定（2026-09-30）：**先补全 bond / angle / torsion 需要的功能函数，然后改进 wall，最后全面展开实际测试。**

这个顺序比"wall 更简单所以先做"更强，理由记下来免得后来人重排：
wall 在周期体系下剩下的唯一难点是**质心锚定**（`workflow/constraints/wall.py:596 bind_center` 的
`state.reference_center` 与 `state.anchors['center:<id>']` 两条来源），而对 wrapped 坐标求质心会把
`0.01` 与 `L−0.01` 平均到 `L/2`。正确质心本来就得在**展开坐标**上求，而"沿键图展开"正是
bond/angle/torsion 需要的那件函数。所以 M0 落地后，wall 那一档只剩四条几何检查，不是再造一套机制。

---

### 1. 现状：框架有什么、缺什么（实测，非推断）

| 已有 | 位置 | 能做什么 | 不能做什么 |
| --- | --- | --- | --- |
| wrap / unwrap / 跨格计数 | `utilities/lattice.py:327 wrap_fractional_with_image`、`:382 wrap_fractional`、`:406 unwrap_coordinates` | 折叠坐标、记录每原子跨格数 | 只是视图，不替算法决定约束绑哪个像 |
| 最近像向量归约 | `utilities/lattice.py:468 min_image_displacement` | 把成对差向量归到最短等价向量：先做逐分量归约，再按 `_MIN_IMAGE_SHELL`（`:444`）搜索候选像 | 搜索半径上限 `_MIN_IMAGE_MAX_SHELL = 3`（`:456`）；需要更宽搜索的**非 size-reduced 胞**具名 `ValueError`（`:546-554`），不静默交出一个更长的向量（PBC_DEVELOPMENT §5.2） |
| 键感知 | `utilities/bonds.py::perceive_bonds` 返回 `(i, j, delta)`；`derive_initial_image` 由 `schema/normalize.py::seed_initial_image` 调用 | 周期与非周期体系下找键并给出像位移 | D-05 已关闭：两条分支都在；`bond_graph` 提供拓扑服务 |
| 引擎侧的像对 | `workflow/calculator.py:557-589` 把 `{坐标, image}` 一起交付，docstring 明写"positions alone do not say which image an atom sits in" | 还原绝对位置 | 谁改写了 wrapped 坐标而不更新 `image`，交出去的就是自相矛盾的一对 |
| 约束的几何核 | `utilities/geometric.py:77 bond_core`、`:88 angle_core`、`:111 torsion_core` | 直接对传入坐标求值与求导 | **核本身没有像概念**：展开由 `state_adapter.py:212 require_usable` 交给 `constraints/images.py`（几何核只吃展开后的坐标） |

缺的四层（M0/M1 要补的正是 1、2 层，3 层是规矩不是数据，4 层留后）：

1. **多原子 CV 的像一致性**：键长可逐对取最近像；角要三原子、二面角要四原子绕同一参考原子展开。
   两两独立归约会拼出物理上不存在的构型。
2. **求解器与 wrap 的耦合**：SHAKE 修正会把原子推过胞面，必须定"在展开坐标上解、解完统一折叠并同步 `image`"。
3. **半胞有效性**：被约束距离接近 `L_min/2` 时最近像出现平手，逐步就不连续。判据前移到输入期拒绝。
4. **范围**：非键接的距离 CV、RMSD、壁中心在周期下**没有路径可走**，第一轮继续拒绝，不为它们发明第二条像规则。
5. **约束 virial**（跨胞约束键对应力张量的贡献）属 A5/D-14，本阶段不做。梯度侧无需额外机制：delta 固定时
   `d(展开坐标)/d(存储坐标)` 是单位阵，力只需散射到正确原子。

---

### 2. M0：image-aware 几何核（不动主框架运行路径）

| 交付 | 落点 | 出口判据 |
| --- | --- | --- |
| 最近像归约补成**候选像搜索版** | `utilities/lattice.py::min_image_displacement` 就地升级，**不在约束侧写第二份** | 非倾斜胞与现实现逐位一致；倾斜胞给出真最短向量；把 `lattice.py` docstring 里那句自认缺口删掉 |
| 按**连通分量**展开：anchor 取存储坐标，其余沿键边累加像位移，返回展开坐标与回写映射 | `utilities/geometric.py` 新函数（与 `bond_core/angle_core/torsion_core` 同族） | **跨面断键负对照**：水分子 O 与一个 H 分居胞两侧；正确实现残差 ≈ 0，"直接相减"得 ≈ `L−r` |
| 角/二面角在展开坐标上求值，Jacobian 行散射回 wrapped 原子 | 复用 `utilities/cv.py::angle/dihedral`，不复制数学 | 整分子平移半格：值与 Jacobian 逐位不变；**不跨面时必须与今天的结果逐位相同**（DC-1 等价性） |

设计约束（写死，防走偏）：

- 展开的单位是**连通分量**，不是每条约束。一条 O–H 键与一根 H–O–H 角共享 O；若每条约束各自锚定自己的
  第一个原子，同一个 O 在一次 SHAKE 解里就有两套局部坐标，而 Gauss-Seidel 跨所有约束迭代，结果会依赖
  约束枚举顺序。
- 键感知判据继续只有 `utilities/bonds.py::bond_lib` 一份，表在 `schema/reference.py::BOND_LENGTHS`（主文档 §3.8 禁止第二份竞争表）。
- 半胞检查（第 1 层判据）在 validate 阶段具名拒绝并给到具体轴与差值；`bonds.py` 派生初始计数时
  已有同形状警告，复用其口径。
- 上一轮我提的"delta 必须进 plan 与 fingerprint"这条**在半导体拒绝先落地后不再需要**（delta 变成从当前
  几何确定性导出，不是存储状态）。这条收回必须写在这里，免得后来人以为我忘了。

---

### 3. M1：多条件 SHAKE（A4，angle / torsion 硬约束）

约束 Jacobian 已存在（`cv.angle` / `cv.dihedral` 带退化拒绝），缺的是求解侧：多条件 Newton/Gauss-Seidel、
共线与退化几何具名失败、冗余由秩检测判。

出口：残差 `1e-12` 相对收敛；退化几何具名失败而非发散；DOF 按条件矩阵秩计（沿用 `ensemble.py::_dof`
的受约束分支，含减 3 的质心约定）；**位置通路照旧，`velocity_constraint` 默认仍 `none`**。

主文档 §14 C1 与 §18 的 A4 行是这一段的两半账：实现 + 删除拒绝语句（当年写在 `schema/constraint.py:1449`，该模块已随 #271 解散，现在这句在 `schema/normalize.py`）
`%s constraint is not implemented`）。

---

### 4. M2：拓扑输入（**接口归属待定**）

`perceive_bonds` 缺的是非周期那一半（D-05）。产物形态应是 `(i, j, delta)` 的表，delta 在非周期下恒为 0，
这样 M0 的展开函数只有一种输入形状。

待用户裁定（**开工前必答**）：

- **(a)** 用户在 `schema/normalize.py` 侧产出挂在 `System` 上的键/角/二面角表；M0/M1 先写成"消费一份表"
  的函数，用测试夹具喂表。
- **(b)** 本轮由约束侧补 `perceive_bonds` 的非周期分支，normalize 侧只做接线。

我倾向 (a)：M0/M1 的判据不需要真感知，先把几何核钉死，接口耦合留给一次做完。
若选 (a)，新字段要走四个注册点并登记 `reference.py`/`config.py` 的 DEFAULT（项目既有规范），且默认不计算、
不写入（DC-1）。

---

### 5. M3：wall 改进（四条输入期具名拒绝）

物理前提（用户补充，2026-09-30）：**wall 要么在非周期体系用，要么在周期体系里不能超出盒子**，否则没有物理意义。

机制版：壁面严格落在胞内且壳层（壁面到胞面）厚度非零时，原子逃离壁必须先离开壁内进入壳层，而**壳层里力指向壁内**，
所以到不了胞面；唯一漏洞是一步位移直接跨过整个壳层落在对侧壁内，用壳层厚度下限堵住。反之壁面触到或越过胞面时壳层厚度为 0，
原子跨面后直接出现在壁内，势垒被周期性打穿。

四条检查：

| 检查 | 内容 | 失败即 |
| --- | --- | --- |
| 包含性 | 分数坐标下壁表面每条轴严格在 `(0,1)` 内。sphere：逐轴 `|c_frac − 0.5| + R/L_axis ≤ 0.5`；box / triclinic 用半轴与基向量在分数空间同判 | 具名拒绝，给到轴与差值 |
| 壳层厚度 | 壁面到最近胞面间距 ≥ 若干倍最大单步位移（`dt`、`temperature`、质量都可得） | 拒绝 |
| 成员初始在壁内 | 绑定时刻每个成员的带号超出量为负（`wall.py:302 _signed` 已返回该量） | 拒绝，理由写"这是外场用法，要求该轴 pbc=0" |
| 中心来源 | 质心锚定 + 任一轴周期 | 拒绝；**M0 落地后升级为"用展开坐标求质心并放行"** |

有利事实（实测）：`wall.py` 现有四种几何 **全部是闭合包围形**（`sphere:209`、`ellipsoid:217`、`box:226`、
`triclinic:237`），没有开口平面，所以不需要按形状定策略，一条包含判据覆盖全部；且该判据只用 cell + center +
尺寸，**不需要 min image、不需要键表**。

**待裁定的 A/B（我上一轮提的，仍未答）**：
- **A**：把周期拒绝的粒度从"体系是否周期"改成"该能力是否需要像一致性"，即 wall（显式中心 + 过包含检查）放行，
  其余照旧拒绝。`workflow/constraints/state_adapter.py:212 require_usable` 是主框架运行路径上的守卫（`:209 IMAGE_INDEPENDENT = ('wall', 'fix')` 划出不需要像规则的那半），动它需要点头。
- **B**：与 M0/M1 一起开放，本轮只把四条判据写进主文档 §5.2 / A5 当设计条款，不动代码。

---

### 6. M4：全面实测

- PySCF 一侧没有周期体系，所以**周期算例只能走 LAMMPS**（本机 `.venv` 里自编译 OpenMPI 那份可用）。
  `qualification/pmd_env.yaml` 照旧不碰（远程口径，属用户裁定），继续用 skill 配置喂 `lammps_path`。
- 这一档是任务 #24「LAMMPS + 框架约束的真实算例」第一次成案。若选 A，wall 算例不依赖 D-05 即可跑；
  若 M0/M1 完成，bond/angle 的周期算例才能跑。
- 判据一律读框架自己的产物（`<title>.md.xyz`、`<title>.log`），与主文档 §20 同一口径，不读测试夹具。
- 收口时逐条更新：主文档 §5.2 / §14 / §15 / §18.1 disposition、`dev/TODO.md` §3、CHANGELOG 新条目。

---

### 7. 本阶段不做的事

- 非键接距离 CV、RMSD、复合体组心在周期体系下的像规则（第 4 层：没有键图路径就没有确定答案，宁可继续拒绝）。
- 约束 virial / 变胞约束（A5、D-14）。
- 开口平面型 wall（当前四种几何都不是，等真有需求再定形状策略）。
- 未获 A/B 裁定前不放宽 `require_usable`；未获 M2 裁定前不在 normalize 侧新增字段。
- 不提交 git（本项目无上传历史，可回溯靠文档与快照）。

---

### 8. 进度对账（开工时逐条打勾）

（表内行号随当前工作树维护，截至 #267。）

| # | 里程碑 | 依赖 | 状态 |
| --- | --- | --- | --- |
| M0-1 | `min_image_displacement` 候选像搜索版 | — | **done #266**（`lattice.py:468`；正交胞逐位一致 + 六方胞实测增益 >2 Å + 非简约胞具名拒绝） |
| M0-2 | 连通分量展开 + 回写映射 | M0-1 | **done #266**（`geometric.py:286`；跨面断键负对照 11.54→0.96 Å） |
| M0-3 | 角/二面角在展开坐标求值 + Jacobian 散射 | M0-2 | **done #266**（`geometric.py:379`；不跨面逐位相同，平移半格 1e-12 内不变） |
| M0-4 | 半胞有效性在 validate 硬拦 | M0-2 | **done #266**（`images.py:185`；含显式 target，给到失效轴。附注：首版因重复换算 Å→Bohr 而假通过，已配成 6 Å 过 / 11 Å 拒的正负对照） |
| M1-1 | 多条件 SHAKE（angle / torsion） | M0-3 | **done #266**（`shake.py:537`；1e-12 相对收敛，键-only 与旧解一致到 1e-14） |
| M1-2 | 退化与共线具名失败 + 秩冗余检测复用 | M1-1 | **done #266**（`condition_redundancy:485`；平面三角形 rank=2） |
| M1-3 | 删除 `schema/constraint.py` 的 not-implemented 拒绝 | M1-1 | **done #266**（原 `:1449`；#271 之后该模块解散，函数现名 `normalize_hard_constraints`，在 `schema/normalize.py`） |
| M2 | 拓扑表归属 | 用户裁定 (a)/(b) | **裁定 (a)（2026-09-30）**：约束侧只消费 `(i, j, delta)` 表，normalize 侧不新增字段；`perceive_bonds` 的非周期分支（D-05）留待接线那一次做完 |
| M3-1 | 四条 wall 检查 | — | **done #266**（`wall.py:720 periodic_check`） |
| M3-2 | 质心锚定壁在周期下放行 | M0-2 + A/B 裁定 | **裁定 A（2026-09-30）**：`require_usable:211` 改为按能力判定；质心走展开坐标（`state_adapter.py:309 _wall_geometry_variants`） |
| M4 | LAMMPS 周期约束算例 | M3-1 / M1 | **done #266**：`draft_constraint_examples/constraint_md_periodic_shake.yaml`，产物与数字来自 2026-09-30 的周期证据集（已删，§33）。未并入 `pyrai3md/examples/`（撞官方 LAMMPS 例数硬编码与 `pmd_env.yaml` 远程口径，属用户裁定） |

### 9. 收口时记下的三条，供下一轮开工前读

- **展开坐标必须用偏移表生成，不能沿键累加向量。** 两者给同一个分子，只有
  `coords + offsets @ lattice` 让"没跨面的原子"保持逐位不变，而 DC-1 的判据就是逐位。
  沿键累加会把坐标过一遍 cart→frac→cart，末位就变了。
- **`fold_back` 的 image 计数是绝对值，不是增量。** crossings 由展开坐标的绝对分数位置导出，
  已经含展开自己那份偏移；再 `offsets + crossings` 就把一次跨面报成两次。这类"补偿被乘/加两遍"
  的错不在失败测试里出现，是在写回写判据时出现的。
- **约束的速度半边只到键为止。** 角度与二面角声明 `velocity_constraint: rattle` 现在具名拒绝，
  下一轮若补 `solve_velocities` 的多条件版本，判据要用同一盒 A/B：本轮实测位置-only 在真实
  LAMMPS 跑里 `Etot` 漂移 `0.090863 Ha`，而基线是 `5.1e-5 Ha`、rattle 是 `0.000320 Ha`。
  位置通路保持不动是裁定，不是遗漏。

### 10. 第二轮之后剩下的清单（2026-09-30，CHANGELOG #267）

第 8 节的 M0–M4 已全部落地。这一节是"下一步开工前该看什么"的单一入口：
把剩下的事情按**挡住它的东西**分成三桶，而不是按大小或难度。混成一锅是上一轮我犯的错
（当时口头列的清单漏了 well-tempered、解析 bias Hessian、六参数晶胞三项）。

#### 桶一：缺实现（不依赖实机，也不缺裁定，可以直接开工）

**已清空**（#269）。本轮关掉的是多变量 history potential 的解析曲率；根因是一处 `sigma` 的幂次（`z` 无量纲 ⇒ 一次求导带 `1/sigma`，不是 `1/sigma^2`），它只在 hill 正好落在当前值上时看不出来，所以之前被读成"多变量耦合写错"。判据也换成了能量的四角双阶差分，并给出步长阶梯与仪器自校（细节见主文档 §18.3 与 2026-09-30 第二轮证据说明的 §6–§7，均已删，§33）。

| 曾列在此格、实际不该在此 | 归属 |
| --- | --- |
| 约束 virial / 变胞耦合（A5、D-14） | **已实现（2026-10-13 键条件；角与二面角随后实测，见 `TestAnAngleAndATorsionHaveTheSameCellForce`；第四类按名拒绝）**：`projection.cell_virial` 按分数坐标口径给出条件自己的晶格导数，乘子取自 `tangent_split`，由 `runtime.constraint_split` 的 `cell_virial` 键传出、`optimizer/base._cell_gradient` 从 `V sigma^T H^-T` 里减去；`ConstraintResult.stress_correction` 仍是零生产者零消费者的声明字段（本轮没有用它承载，位力走 split 的键，因为乘子与 Jacobian 都在那一层）。归属裁定已给出：框架移盒，引擎报应力（§18.1 D-14、桶三末行），不是可独立完成的一格。**#276 把这个裁定从抽象变成有数字的**：移动胞下的严格键实测只剩 `4.806e-05` Å（= D-25），同一条块在固定胞是 `8.131e-11` Å |

> **MD-side ruling recorded 2026-10-08 (#314).** The ownership question this row is waiting on
> has been decided for the dynamics: the framework moves the cell (the barostat in
> `workflow/integrator/ensemble.py`, with `verlet._retriangularize` the only re-triangulation),
> the engine's reported box is not a second authority for an `md` run, and the constraint virial
> is **not** built for this round - a constrained bond under a moving cell is conserved by
> re-projecting after the rescale (`md.py` phase `constrain_after_cell`, D-25/D-26), while the
> missing reaction term in `P_int` stays a stated limitation of the pressure the barostat reads.
> Variable-cell **optimization** is out of the MD workflow by the same ruling and stays with the
> optimization workflow, where `optimize_cell` still has no cell degrees of freedom.

#### 桶二：缺实机基线（代码已在，判据未取）

| 项 | 已有什么 | 还欠什么 |
| --- | --- | --- |
| 多条件 RATTLE 在真实引擎上的能量判据 | **#274 已取**：键-only rattle 漂移 `+4.705014e-04` Ha、键+角都 rattle `+1.080206e-04` Ha（锁到 `8.002e-12 / 1.110e-15` Å、角 `6.336e-09` °），位置-only 那一档 `+9.101845e-02`、无约束基线 `+1.637560e-04` | LAMMPS 通道已修（`a package-manager install of LAMMPS` 补回 open-mpi 5.0.11 与 fftw，旧前缀 `/opt/lammps-mpi/bin/lmp` 重新可加载）。仍欠：长跑（>5 ps）下角速度投影的能量守恒形状与质量漂移 |
| 周期下的 cv 供给（bias / metadynamics） | **#274 已放行并按变量设闸**：`_validate_images` 对每个 cv 走 `require_same_component`；同盒 O–H 的 bias 行 `2.1e-15 -> 1.368028e-05` Ha，跨面帧存储量 `7.618 Å` 对最小像 `0.959168 Å` | 跨两分子的 cv 仍是具名拒绝（正确，没有像规则可推）。**#275 已取**：周期下 gaussian 历史势行 `0 -> 1.684309e-03` Ha（被推的 O–H `0.9815 -> 1.1241` Å），同批的 RMSD 历史势行 `0 -> 4.841365e-04` Ha 也跑通（rmsd 那一条正是炸出 D-23 的算例）；仍欠长跑的收敛形状 |
| 周期 wall 的真实算例 | **#274 已取**：半径 2.8→2.0 Å 在 50-300 步收拢的壁供能，行由 0 增至 `5.474741754228e-04` Ha；顺带修正了"成员起始在壁内"被对 ramp 每个端点各查一次的错误 | **#275 已取**：ellipsoid 约化模 `0.9347 -> 1.1276` 行最大 `4.482278e-05` Ha，box 逐面分数 `0.9524 -> 1.2891` 行最大 `2.064651e-04` Ha，triclinic 面分数 `0.4762 -> 0.5016` 行最大 `9.738379e-08` Ha——三种几何都在真实周期运行里越过了面并供出非零罚项（这正是 D-22 被修的原因） |
| 系综 × 约束在周期侧的真跑 | 离线判据齐（`_dof` 取条件矩阵的秩、fixes 供冻结分量）| **#276 已取**：NVT Nose-Hoover + 两根 rattle 键锁到 `8.131e-11` / `1.332e-15` Å；`fixes` 冻结原子在 NVE 与 NVT 下都是 `0.000e+00` Å（最小像上量）。NVT 下 `Etot` 漂移 `+5.013150e-03` Ha **不作判据**（恒温器不是保能映射）|
| 移动胞 × 严格键（NPT）| **D-25 已修（#276）**：接缝在循环次序——`_berendsen` 缩放坐标与胞、且作用在 post，而 `apply_constrain` 在其之前，故记录的是投影后被缩放的几何；post 之后再问一次几何（`md.py` 相位 `constrain_after_cell`）后快档 `1.71178627e-11` Å、慢档 `1.14519505e-12` Å、NPT 冻结原子 `0.000e+00` Å；反序构造逐位复现原缺陷 ⇒ 因果闭合在次序。代价 0.999 ms/步（整步 166.229 ms 的 0.60%），空 plan 0.001 ms。原口径：同一条约束块在固定胞锁到 `8.131e-11` Å，在移动胞（逐元素最多变 `0.025420` Å）只剩 `4.806e-05` / `7.649e-05` Å，比声明的 `1e-10` 差 5 个数量级；框架既不拒收也不兑现。量具已做负对照（三种 lattice 取法同结果；仿射信号 `1.603e-04` Å 比观测大 3 倍）。机制三候选未归因，等胞所有权裁定 |
| 非周期侧的非球 wall 与 rmsd 历史势 | 单测覆盖四种几何与两种历史势 | **#276 已取**：PySCF 一条 run 里 ellipsoid/cuboid/sheared 三条 wall 行每步非零（`5.505403e-06` / `2.217332e-05` / `3.674277e-05` Ha）；rmsd 历史势行 `0, 0, 0, 2.617433e-04, 2.725871e-07`，沉积步低于 hill 高度正是"本步提交的 hill 不在本步受力"的次序纪律在产物里现形 |
| 移动胞下的多条件求解、wall 检查、力侧 bias、周期受约束优化 | 四类各自的离线判据齐 | **#277 已取**：键+角在胞动 `0.020523` Å 下锁到 `2.337e-11`/`5.667e-11` Å 与 `9.216e-09` °（D-25 的修法覆盖多条件分支）；周期 wall 的四条检查在胞动 `0.025197` Å 下不拒收（但壁未接触、三行全 0，故只证"不误伤"）；bias 行 `5.3341544973e-06 → 1.101342e-05` Ha 且不被误拒；受约束优化在周期引擎上把两根键守到 `9.329e-11`/`6.793e-11` Å ⇒ **D-20 不是所有受约束优化的共性**；但**原因未隔离**（两例步数/初值/收敛不同），要的是等条件 A/B（只换引擎）——**该 A/B 已在 #278 做完，结论是 D-20 不成立并撤回**：包内那份非周期算例的接受几何同样守到 `6.461e-11` Å，那个 `8e-4` Å 是台架 bug 造出来的（测量脚本去开 `constraint_opt_surface.opt.xyz`，而优化器只写不带前缀的 `opt_geom.xyz`/`opt.xyz`，路径不存在于是走进 else 分支打目录清单，清单旁边的数字被当判据读了）。新规矩：**测量脚本缺产物必须抛错，不许 else 兜底**。取代 D-20 的是 **D-26（实测、待裁定）**：`optimizer/base.py:168` 只掩码位移的冻结分量，`:213-217` 的 `grad_rms/grad_max` 取自整条原始梯度，`:247-251` 又用它比 `opt_grms/opt_gmax`；2×2 实测末步 `max_g` 无 fix 时 `8.3668e-02`/`8.4217e-02`、有 fix 时 `2.0797e-01`/`1.9529e-01`，即停止判据在数一些本步骤禁止响应的力。偏差方向保守（更难判收敛），掩码会改变哪些 run 报收敛，故等裁定 |
| 被胞面撕开的分子上做角约束（真实盒）| 只有离线判据 | **#280 已取，且不必新造 fixture**：现成水盒里 5 个水分子的存储 O–H 与最小像差 6.6~7.2 Å，其中 22-23-24 两条臂都跨面。常驻通道实测键守到 `3.647e-12`/`7.772e-16` Å、角守到 `3.097e-09` °；对照是"按存储向量算同一个角会在两帧之间跳 `99.8885` °" ⇒ 坐标约定用错不是差一点。多条件求解也第一次上了常驻服务 |
| 二面角（dihedral）条件在真引擎上 | 离线判据齐（符号、解析雅可比、多条件秩）| **#277 已取（PySCF HOOH）**：H-O-O-H 起始 160.000000°，5 帧散布 `2.625e-10` °，同时未声明的 O–O 动 `8.435e-03` Å、O–H 动 `3.823e-02` Å（对照成立才算数）。**周期侧 #280 补齐（当时判断"需要带 torsion 的 fixture"是错的）**：跨分子二面角在真实盒里以具名拒绝收口（`constraint_md_periodic_torn_dihedral.yaml`：`2 separate connected components`；该 example 已于 #291 退役，同一拒绝现由 `test_constraint_periodic.TestDihedralBranch` 常驻覆盖），无胞跨分子被服务并守到 `1.421e-14` °（`constraint_md_torsion_intermolecular.yaml`，对照 O···O 动 `6.503e-03` Å）⇒ 三种答案各有产物 |
| 拓扑键图在真实体系上的判据 | `bond_graph` + `bond_type` 的 9 项离线判据 | 一个真实 `.data`/真实盒上展开 `bond_type` 的算例（含未知键对触发的具名拒绝） |
| well-tempered 的收敛形状 | 7 项判据（递推式、下溢为无、commit 侧计数） | 一条真实 CV 上的长跑：偏置压平后 `zero_hills:<id>` 是否如预期出现 |
| wall 的 ellipsoid / box / triclinic 周期判据 | `periodic_check` 四种几何的 support 长度都实现了 | **#275 已测**（见上一行数字）。剩下的是开口 wall 形状，属桶三缺约定 |
| 周期下的 restart / 多轨迹 / 约束优化、两条 LAMMPS 通道 | 非周期版本都有测 | **#275 已取 restart 的约束半边**：带胞 8 步连续与 4+4 两段的被锁键散布 `1.839e-11` / `2.150e-11` Å 都合格，但两次产物记的帧数不同（2 帧对 6 帧，接缝时间标签重复 `37.20`）＝ **D-24 未修**，逐帧相等这条判据暂时用不上。通道归属按产物更正：#274 批的 `h2o_force.sh` exec `service/persistent/server.py`（常驻），#275 批 `python_enabled: false` exec `lmp -in`（子进程），两条各有真实批次；多轨迹与受约束的周期优化仍未测 |

#### 桶三：缺约定（要用户裁定，不是要代码）

| 项 | 缺的决定 | 现在代码的行为 |
| --- | --- | --- |
| `molecule` 分子分组 | 拼法：按化学式？按起始原子？按分量大小？ | 键图与连通分量已经有了（`bond_graph()['components']`），缺的只是**名字怎么拼**，所以输入期具名拒绝并写明"缺的是约定不是机器" |
| `region: inside / outside` | 一侧区域的原点与方向定义（二面角的单边区域同理）。**这里的 region 是 bias 的 CV 区间，不是原子选区**：「只优化选定区域」已于 2026-10-13 裁定为 fix 的补集，不需要新约定也不需要新关键词（`OPTIMIZE_DEVELOPMENT.md` §14） | 枚举接受、归一化期具名拒绝（`schema/normalize.py:964-966`） |
| 跨分子 CV 的周期像规则 | 给定一条不在同一分子里的距离/RMSD，**谁**决定取哪个像 | 继续拒绝（§1 第 4 层）。放行它等于现场发明一个约定 |
| 开口平面型 wall | 开口形状在周期下的镜像语义 | 现有四种几何全是闭合包围形，一条包含判据覆盖；开口要按形状定策略 |
| NPT 与约束的组合 | 约束反应力对压强的贡献算不算进 `stress`（与桶一第二行同源） | 引擎报盒（`fin_cell`）与框架移盒的所有权目前由 profile 路由保证不同时拥有 |

#### 复核入口

```
python3 -m pytest pyrai3md -q          # 3425 passed / 1 skipped
python3 -m pytest pyrai3md/qualification/verification -k constraint -q   # 730
```

测量与判据归属：2026-09-30 第二轮证据说明（已删，§33）；
#266 那一轮的运行产物：2026-09-30 周期证据集（已删，§33）。


### 11. 第三轮：输入格式改扁平 + schema 结构合并（2026-10-01，#271）

这一轮做的不是新能力，是**把已有能力放对地方**。判据是主文档 §21 的表，这里只记"为什么"和"复算入口"。

#### 11.1 输入格式：DSL → 扁平 `&constraint`

- 触发点：用户指出"格式与现有风格不符，DSL 的 block 在 yaml/json 里天然具备，在文本场景下输入复杂度过高"。
- 定稿规则（全部由 `test_constraint_parser.py` 的 76 项钉住）：一段 `;` 一个对象；参数行少于对象数就广播/补齐并写 `notes`，多于就拒；`wall_type` 决定读哪些 wall 几何，未点名的关键词**被忽略且回显**；参数后缀带精确类型（`bias_bond_ref`，不存在 `bias_ref`）；`shake_*_fc` 具名拒绝；id 缺省自动编号；ramp 拆成五个后缀键且**不跨对象继承**（缺的那一端用该对象自己的 base）。
- 位置纠正：`constraint_parser.py` 曾放在 `interface/`，用户指出"parser 只做输入类型分拣和文本分隔，经典输入的解析由 schema 的 adapter 完成"。于是该文件删除，读写两个方向都在 `schema/adapter.py`，`interface/parser.py` 回到通用循环。
- 反向渲染：`write_constraint_section` 让 qualification 的 YAML → `.input` 转换能写出这段，且**写不出的形状直接报错**（部分对象有 id、共享 schedules、同一 `wall_groups` 里分子长短不一），绝不产出读回不同内容的文件。往返等价由 plan 指纹判定。位置线需要「这个对象没声明」时用 `none` 段表达（#273），否则补齐会把上一条 wall 的中心搬到这一条上。
- `notes` 落到运行记录：`<title>.log` 新增 `&constraint plan` 段（实体 + 自动编号 id + ramp 窗口 + notes）。无约束的运行不打印该段（DC-1）。

#### 11.2 schema 结构：两个独立模块 → 四个框架文件

`schema/constraint.py`（1917 行）与 `schema/legacy_adapter.py`（438 行）删除。分工：记录模型 → `config.py`（`System` 区块之上，`System.constraints` 因此不再需要前向引用与 `model_rebuild()`）；`normalize_*` 与 `build_plan` → `normalize.py`（与流水线上同名步骤同处）；三条具名拒绝 → `validate.py` 并改为公开名（`normalize.py` 调同一函数，规则不可能因先到的一层而变味）；legacy 十九键适配 → `adapter.py`。

顺带补的六处真缺陷（都不是搬运副作用，全有判据；细节与数字见 CHANGELOG #271）：

- ramp 的 `stop` 由 `md.step` 关闭这一路，此前只有 config 侧写了 `end_step` 才走得通；而
  `normalize.py` 的输入期拒绝通道拿不到步数，会把"合法的开放 stop"判成错误。补
  `build_plan(md_step=)`，两处分别喂 `data.step` 与 `pmd.dynamics.step`。
- `ScheduleRecord` 端点写死 `float(value[i])`，向量端点（三条半轴、六参数晶胞）装不进去（`TypeError`）；
  改 `_endpoint_value` 保留嵌套并强制两端形状一致。
- 对象线顺序：读侧按家族表序拼接、写侧按记录序发行，混合对象线的家族（`fix_element` 写在
  `fix_atoms` 前）会让 `fix_dir` 落到错的选择上——数字全对、归属错。统一为"按文件里对象线出现的
  顺序"锚定。负对照：改回表序，往返判据的 plan 指纹由 `f3f1669005ca91ff` 变 `9724c6bc09b88772`。
- 写侧对 `model_dump()` 必带的空 `schedules: []` 报错——**空字段不是声明**，
  改为只在真有内容时拒。
- 默认值探测读 `model_fields[f].default`，对 `default_factory` 字段拿到的是 PydanticUndefined
  并被当列表迭代（`TypeError`）；改用 `get_default(call_default_factory=True)`。
- 搬运脚本的字符串替换把函数**定义**也改了名，`normalize.py` 与 `validate.py` 各存一份拒绝函数、
  其中一份是引用不到辅助名的死代码。去重后用 `is` 同一性钉住
  （`normalize.check_periodic is validate.check_periodic`）。

#### 11.3 复算入口

- 约束子集：`python3 -m pytest pyrai3md/qualification/verification -q -k constraint`
- 扁平文法与读写：`pyrai3md/qualification/verification/test_constraint_parser.py`（90 项，整包子集 `-k constraint` 777 passed；#273 为 wall 的四个关键词与 `none` 段补了读写双向判据）
- 文本→config→plan→运行记录：`test_constraint_integration.py`（26 项，含 `TestRunRecord` 5 项，其中一项钉 wall 行的 law/target/objects/中心来源）
- 桶二、桶三未变：见 §10。本轮**没有**新增实机依赖，也没有替用户裁定任何约定。


---

### 11. 收口（2026-10-02，#281-#285）：本文标"未修/待裁定"的行全部落地

本文是开工顺序与出口判据的记录，正文各节保留当时的说法不动；这一节只把结果对上号：

- **D-19b / D-24 关掉**（原文多处写"D-24 未修，逐帧相等这条判据暂时用不上"）。现在两台的 A/B 都能逐帧比：
  周期 `[1,2,4,6,8]` 与非周期 `[1..8]`，两档帧号、时间列完全一致，共享帧逐帧最大坐标差 `0.000e+00` Å
  （周期 60 原子、非周期 3 原子），能量列逐行相同；力评估次数 8 对 4+4，续跑不再 warm up。
  判据能用的前提是那条新规矩：**写 `.rt` 的步也记一帧**（`AIMD._runtime_file_due` 一个谓词同时决定文件与帧）。
- **D-26 按口径 A 关掉**（原文"等裁定"）。`projection.declared_mask` = 冻结分量 ∪ 严格条件参与原子分量，
  梯度判据只看剩下的分量；同一几何同步量 `Gradient RMS 0.04358507→0.00873618`、`Max 0.08206846→0.01490334`。
- **D-27 关掉，且默认值方向被纠正**（原文只讨论"该在哪一层拦"）。所有权随 `control.driver` 传递、两侧各自
  具名警告；skill 侧默认改回 `skill`——默认写成 `main` 会让每一个不带框架的 `&lammps` 输入静默丢掉 shake。
- **D-18 撤回**（原文按"记录能量不守恒"登记）。那条读数比较的是同一个键 `results['energy']` 的两个读者；
  独立裸能量证明 `Epot = origin + 约束行`。口径 A 要的分解块已加：`&constraint energies in Hartree`
  （`origin pot` / 每条约束用输入写的 id / `Epot`），六个真实产物验到求和残差 `0.000e+00`（周期 wall 一例
  最大 `1.110e-16` Ha）。
- **二面角在周期真引擎上的那一格补完**（原文写"需要带 torsion 的 fixture"才算数）。乙醇 CH3CH2OH 进
  10×10×10 Å 盒，拓扑由 FFautomator 出（9 原子 / 8 键 / 13 角 / 12 二面角，RESP 电荷拟合 ORCA 波函数），
  声明的 C-C-O-H 守到 `2.842171e-14` °（容差 `1e-10`），未声明的 H-C-C-H 同期走 `5.605489e-01` °。
  这一格顺带挖出支持库缺陷 **D-28**：`utilities/bonds.py` 的 C-C 记成 1.20 Å，而周期感知判
  `distance < bond_lib × 1.2`，于是任何有机单键都看不见、整分子被读成三个连通分量；改成单键值后
  水盒仍恰好 40 条 O-H（最近 O···O `2.704` Å / H···H `1.519` Å 对触及 `1.776` / `0.888` Å），
  乙醇盒感知到拓扑声明的 8 条键。
- **本文剩余未闭合项**：A5/D-14 与 `optimize_cell` 的胞自由度均已于 2026-10-13 闭合（约束位力按分数坐标口径进 `dE/dH`；键、角、二面角三类随后都已实测，见 `test_constraint_cell_coupling.py::TestAnAngleAndATorsionHaveTheSameCellForce`；第四类条件仍按名拒绝）；
  约束本身对周期与非周期两套边界条件的支持已按目标收口。

### 12. 草稿格式对齐（2026-10-02，#286）

38 份草稿已按 `dev/EXAMPLE_DEVELOPMENT.md` §6 对齐：Topology/Run 三行头 38/38、`control.title ==
文件名_example` 38/38（原来那条 title 与文件名脱钩的 `constraint_md_periodic_cv_across` 一并纠正）、
14 份补上显式 `environment: local/single`。产物名因此全部带 `_example`。

**要进 `pyrai3md/examples/` 还差三件事**（这次没做，属于另一裁定）：① 按 workflow 重命名（§6 明确
反对按功能加前缀，`constraint_` 前缀要拆成 `md_lammps_nvt_wall.yaml` 一类）；② 登记进
`qualification/manager.py:67 QUALIFIED_EXAMPLES`；③ 乙醇的 `ethanol.data` / `ethanol.in.init` /
`ethanol.in.settings` 走 `setup.py package_data` 显式声明，否则 pip 安装后拿不到（水盒那三件套现在是
从证据目录拷进运行目录的，官方样例集里没有这条通路）。

复跑口径顺手记一条：`constraint_opt_surface` 在干净目录三次跑出 `origin pot
-74.9561538149754654` / `Gradient RMS 0.00873618` 逐位相同，而在留着
`scratch/PYSCF_hf_...chk` 的旧目录里给 `0.00873616`、两次互不相同 ⇒ **PySCF 复用 scratch checkfile
会让末几位漂移**，"逐位相同"类判据必须每次换干净工作目录。

### 13. 输入只留一套关键字（2026-10-02，#287）

裁定：**yaml/json 保持经典 input 的扁平设计，只是换一种书写格式；input/yaml/json 关键词一致**，结构化
记录是代码内部便利、不暴露，也不为旧拼写写兼容指引。列表用来取代文本分隔符 ⇒ **一个元素＝一个对象**。

已落地的三步（本文原先的"下一步讨论 yaml"已走完）：

1. 读者：`_constraint_declaration` = `_read_constraint_section`，结构化直通与
   `CONSTRAINT_STRUCTURED_KEYS`/`_constraint_structured` 删除；`version`/`units`/`notes`/
   `schedules`/七族记录键/`fix_ref` 在输入处**具名拒绝并给出替代关键字**。
2. 语法补洞：一条对象线可带多行（`shake_bond 1 2 1 3`＝一个具名块两条键，非 arity 整数倍具名拒绝）；
   `wall_groups` 维持"原子序号＋末位每分子数"，判据就是整除（守卫本来就在）；`X-H` 通配进
   `shake_bond_type` 的值域（`X-C`/`X-X` 拒绝，大小写与顺序自由）。
3. schema：审计出写不出的字段后按裁定处置——等价支持已存在就删（八项＋`bias/meta` 的
   `element`/`group` 四字段），真漏洞才补（只有 X-H，且它是值域不是关键字）。

**判据**：2026-10-01 的扁平草稿证据集里的往返脚本（已删，§33）对每份草稿做
`结构化块 → write_constraint_section → 读回 → build_plan` 的指纹比对，38 份全部相同
（`0 of 39 drafts differ`，第 39 份是新的 X-H 算例）。这条往返第一次跑时就挖出写侧三条缺陷：
D-29（writer 只写 bonds 的第一行，静默丢键）、D-30（`%g` 把数字截到 6 位，写出的 `.input` 是
百万分之五不同的运行）、D-31（退役键在 writer 里 `KeyError` 而不是具名拒绝）。

**引擎复跑**（扁平草稿、干净目录）：周期续跑 A/B 共享帧 `0.000e+00` Å×5、对照 `2.564e-02`；
非周期 `0.000e+00`×8、对照 `1.354e-03`；D-18 分解 5 步两条恒等式 `0.000e+00`（行名 `hold_oh`）；
乙醇二面角 `2.842171e-14` °/对照 `5.605489e-01` °；受约束优化持键 `7.037104e-11` Å、
冻结原子 `0.000e+00`、末步 `Gradient Max 1.1848e-01`（#282 未掩码记录 `2.0797e-01`）；
X-H 通配 6 行守到 `4.989761e-11`、对照 C-C/C-O `1.687058e-03`。

随包草稿 **39 份**。进 `pyrai3md/examples/` 仍差 §12 列的三件事（按 workflow 重命名、登记
`QUALIFIED_EXAMPLES`、乙醇三件套走 `package_data`）；乙醇两份算例在原地跑需要把
`ethanol/lammps/equilibration/ethanol.{data,in.init,in.settings}` 拷进启动目录，这点已写进草稿头注释。


## 35. 判据改为切向投影：D-26 的掩码口径被取代（2026-10-10）

**裁定**：收敛限度判定应当考虑约束的贡献。落地的口径是「切向投影判定，乘子单独报」：
严格条件（键长／键角／二面角）只判梯度在约束流形切空间上的分量，法向分量作为该条件的
Lagrange 乘子输出、不参与判负；谐振约束／CV 偏置／metadynamics 属于目标函数的一部分，
按总梯度判，不剥离。这替换 #282 的 `declared_mask` 抹除口径。

**为什么必须换**（同一份结构上的对照，本机解析梯度，非引擎）：一根沿 x  held 的键，
在两个原子上横着施加 5 Hartree/Bohr 的力。掩码口径认为「这两个原子被条件点名 ⇒ 它们的
9 个分量里 6 个不算」，于是只判 3 个分量、报 `Gradient Max = 1e-6` 并判收敛；切向口径读到
的仍是 5.0，不收敛。反过来，同一对原子上纯粹沿键方向的 5 Hartree/Bohr（正是条件吸收的那部分）
在新口径下判据读作 1e-12 量级、判收敛，同时把 λ = 5.0 打在日志里。也就是说旧口径既会漏判
（把真实可动的力当成不算）又会把该算的一起抹掉，而新口径只在条件自己的方向上让路。

**实现落点**：

- `constraints/projection.tangent_split(gradient, coords, mask, conditions|pairs)`：
  返回 (切向梯度, 每条条件的乘子, 数值秩)。用未做质量加权的 Jacobian 行 —— 梯度是余向量，
  判它的度量就是步长所在的笛卡尔度量；质量加权是逐列缩放，对速度（动能度量）是对的，
  对梯度会转法空间。乘子由同一次 SVD 给出（最小范数解，重复条件按秩读）。
  解析 Jacobian 对照下 λ 精确复原到 1e-12（键 + 角两条件的混合情形）。
- `constraints/runtime.constraint_split(state, gradient)`：本子系统对框架的第三个钩子，
  把上面的数字加上每条条件的 target/value/residual 与位置求解的诊断打包。
  `constrain_state` 也不再丢弃 `project_positions` 的 diagnostics，写进
  `ConstraintState.diagnostics['last_projection']`（该字段本来就在）。
- `workflow/optimizer/base.convergence`：判据取 `note_evaluation` 在**该梯度所属几何**上
  读到的切向分量（因此必须由 workflow 在移动原子之前调用）；`declared_mask` 随唯一消费者删除。
- `utilities/print_results.format_constraint_geometry` + `RunLogger.log_optimize_step`：
  新增 `&constraint geometry` 段，与 `&constraint energies in Hartree` 同一套
  `_section`／`%-18s %24.16f` 约定；每条声明一行（id、按输入 1-based 编号的原子、target、
  value、residual、multiplier），段尾是 rank／frozen components／restored components／
  iterations／max residual。长度 Bohr、角度 radian，标题里写明。

**真引擎回归**（随包 `constraint_opt_periodic.yaml`，LAMMPS 供力、60 原子（20 个 TIP3P 水）水盒、6 步）：
两根声明的 O-H 在新日志段里读作残差 `-6.8811623066e-12` 与 `2.2204460493e-15` Bohr，
乘子 `-1.4820317988e-03` 与 `1.2402278389e-02` Hartree/Bohr，rank 2、求解 3 迭代、
max residual `3.71e-12` Bohr。同一轮里过程表与轨迹的计数被对齐：末次求值有自己的行号
（旧实现把交付结构的帧与上一行的号码配在一起，表里出现同号重复行）。

**已经合上的**：约束与变胞同现（A5/D-14）。cell driver 就是优化器自己（`_cell_ride` +
`_cell_gradient`），条件对盒的导数由 `projection.cell_virial` 按分数坐标口径给出、乘子取自
`tangent_split` 那次 SVD，键条件已用解析模型把符号与幅度一起钉死（有位力项盒子停在盒弹簧自己的
静止长度，去掉它停在 0.227 Bohr 之外）。**仍然开着的**：角度与二面角条件是同一公式但未测，
`optimize_cell` 对其按名拒绝；分量级 fix 无乘子，其对位力的贡献记 0，与 #314 同源。

