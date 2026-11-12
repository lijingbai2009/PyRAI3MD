# Periodic Boundary Conditions (PBC) Design

Last updated: 2026-10-22 21:10 (section 1 and section 4.4 now name the division of labour with
`CELL_OPT_DEVELOPMENT.md`, which owns the geometry-search half of a variable cell; nothing else
was re-read on this pass. Before it, 2026-10-13 14:22: the wrap/ledger entries and the
position-in-the-loop rows were checked against the code for the `data/geometry.py` move; sections
4.2 and 7 were reconciled again against
the two-center real-engine runs, which now certify C>1 on LAMMPS as the low layer, and
against the barostat reading that shows no virial. Section 7.1's subsystem-input line
follows the fixed container of #320; sections 4.1, the stage letters and the earlier
status blocks follow as written)

First reviewed against the working tree on 2026-09-24. This document separates
**what the code does** from **what is still proposed**; the stage letters (A-E)
in section 6 are load-bearing, because code comments and error messages cite
them (`schema/validate.py:408-420`, `skills/cp2k_skill/schema/validate.py:207`,
`workflow/integrator/gsh.py:135`). Paths are relative to the package root
`pyrai3md/`.

> **Implementation status (2026-09-25, CHANGELOG #216).** Stage A (the data
> contract) and Stage B (the fixed-cell chain) landed. Decisions of that round,
> restated against the current code:
>
> - **`pbc`, after normalization, is the single periodicity authority**; the cell
>   describes geometry only. `_is_periodic` (CP2K) and xTB both read `pbc`, so a
>   non-zero lattice no longer implies a periodic run
>   (`skills/cp2k_skill/service/builder/input_builder.py:189-201`, `:244`,
>   `:350-351` -- `PERIODIC NONE` plus `PSOLVER WAVELET` for a molecule,
>   `PERIODIC XYZ` only for full 3D).
> - **The default cell is a sentinel.** `cell=[99,99,99,90,90,90]`
>   (`schema/config.py:93`) means "not filled in", and normalize does not derive
>   `lattice_vector` from it (`schema/normalize.py:283` `_convert_lattice`, the
>   `cell_is_sentinel` branch at `:315-317`; restated at
>   `workflow/constraints/state_adapter.py:14-17`). Behaviour change that round: a
>   molecular CP2K run went from `PERIODIC XYZ` at a fake 99 A box to
>   `PERIODIC NONE` + `PSOLVER WAVELET`.
> - **Partial periodicity (mixed 0/1) is refused explicitly**, at the framework
>   layer (`schema/validate.py:408-420`) and again in the CP2K skill
>   (`skills/cp2k_skill/schema/validate.py:203-209`); the message points at Stage E
>   of this document. The low-dimensional `PSOLVER` mapping waits until there is
>   an acceptance criterion for it.
> - **Every evaluation gets one frozen snapshot** through
>   `backend.calculate(..., cell=...)`, frozen together with the geometry
>   (`workflow/calculator.py:540` `_snapshot_cell`, consumed at `:618`, `:1146`),
>   and the same snapshot reaches a remote term through the payload
>   (`workflow/dispatch/term_worker.py:193`, `:354`; the cell dict keys are
>   documented at `workflow/dispatch/terms.py:124`, `:196`). That channel is what
>   retired the one-step-behind cell problem; `sync()` was deliberately not moved
>   earlier -- it still refreshes the configuration at step end for
>   checkpoint/restart (`workflow/calculator.py:128-138`). A region-slice task
>   does **not** inherit the global cell; only whole-system tasks do
>   (`workflow/calculator.py:814`).
> - **The shared cell module lives at `utilities/lattice.py`** (pure functions,
>   no state store). The old `data/structure/cell.py::Cell`, together with the
>   whole `data/structure` and `data/dynamics` sub-packages, is gone; its
>   volume / boundary_condition / ndim_periodic semantics are carried by
>   `utilities/lattice.py:113` (`cell_volume`) and `:188` (`boundary_condition`).
> - **Capability is declared, not inferred.** Each skill's `manifest.yaml` carries
>   a top-level `supports_pbc` (cp2k / lammps / xtb true, the rest false), read
>   through `query.get_metadata` -> `discover_all` and enforced at configuration
>   time by `schema/validate.py:441-522` (`_reject_unsupported_pbc_backends`); a
>   molecular backend also refuses `any(pbc)` in its own `validate()` as the
>   second line. The granularity reaches the method level, not only the skill
>   level (`skills/xtb_skill/schema/method.py:33`, `:41`, `:49` declare `gfn2`/
>   `gfn1`/`gfn0` false against `:57`, where `gfn_ff` is true).
> - **`apply_pbc` raises.** Enabling periodicity without a usable cell is a
>   `ValueError` now, not a silent return (`data/geometry.py::apply_pbc`). The
>   function no longer lives in `workflow/integrator/operations.py`: a geometry
>   search whose step carries the atoms with the box wraps between its own steps,
>   so the wrap is a structural operation of the framework and not a phase of one
>   timestep.

> **Implementation status (2026-09-28).** Stage C (coordinate continuity /
> image) landed. The representation conventions were decided as follows, and
> section 4.1 below describes the result rather than a gap:
>
> - **The primary structure data stays wrapped Cartesian coordinates** (the
>   main-image view). It was not changed to continuous coordinates, because
>   inter-atomic connectivity can only be judged cheaply, by the minimum image, in
>   the main image. Measured and confirmed the same round: a bond pair that
>   crosses the box face reads 9.24 A raw and 0.965 A under the minimum image --
>   the trajectory numbers were never wrong, what was missing was telling the
>   reader about the box.
> - **The cell and the per-atom integer image counts are both persisted.**
>   `apply_pbc` accumulates crossings into `System.image`
>   (`schema/config.py:1032`) while wrapping, and commits the coordinates and the
>   counts in one write (`data/geometry.py::apply_pbc`,
>   `::record_wrapped_geometry` - the field's only writer, which takes the
>   absolute counts of the coordinates it is given; see
>   `dev/methods/CONSTRAINT_DEVELOPMENT.md` section 5.2 for why a fold's counts
>   may not be added to the ledger).
>   `image` is runtime state that travels with history frames and checkpoints -- it
>   is not an input keyword, and it is deliberately **not** in
>   `SYNC_SYSTEM_FIELDS` (`workflow/calculator.py:128-138`). Continuous
>   coordinates and whole-molecule coordinates are **derived views**
>   (`utilities/lattice.py:406` `unwrap_coordinates`, `:468`
>   `min_image_displacement`), not the primary representation.
> - **Every evaluation snapshot carries the image.** No new kwarg: the
>   `calculate(cell=...)` channel of Stage B is the one channel. Both LAMMPS
>   execution paths consume the same `(coordinates, image)` -- the cold path
>   writes the three trailing columns of the Atoms section
>   (`skills/lammps_skill/service/builder/input_builder.py:801`), the persistent
>   path batches `set atom <idspec> image nx ny nz` alongside the coordinates
>   (idspec being a single ID or a `lo*hi` range, merged over contiguous runs).
>   Under an orthorhombic box both sides use the minimum image and only changing
>   one is invisible; under a triclinic box LAMMPS builds bonds from the unwrapped
>   coordinates, which is what makes the cold/persistent bit-equivalence gate able
>   to see the difference at all.
> - **The initial crossings are seeded from the geometry at configuration time.**
>   Accumulation can only record crossings the framework has *observed*, but a
>   wrapped equilibrated frame has already crossed box faces before the run starts
>   -- 7 O-H bonds of the shipped 20-water box do, on 5 atoms. So
>   `schema/normalize.py:1744-1745` (`seed_initial_image`, after
>   `compute_derived_quantities`) runs `utilities/bonds.derive_initial_image`
>   (`:431`): perceive bonds by minimum image against `bond_lib`, walk a spanning
>   tree per connected component, inherit parent count + the image step the bond
>   needs, then reduce to integer counts by fragment mean. **Not one coordinate
>   character is changed; only counts are read out.** This does not contradict the
>   rule above: no user file and no yaml declares `image`, the value comes from
>   the input's own geometry, and an explicitly written `image:` is kept with the
>   derivation standing down. A system that cannot be perceived (a ring that winds
>   around the box, a bond length that does not fit the box, more atoms than
>   `PERCEPTION_MAX_ATOMS` = 4096, `utilities/bonds.py:73`) returns empty and
>   leaves a `UserWarning`: configuration time never raises here and never hands
>   back a wrong count.
> - **Trajectory products carry the per-frame box.** `<title>.md.xyz` still writes
>   main-image coordinates, with `Lattice="..."` on the comment line in the
>   extended-XYZ convention (`workflow/records/output.py:102`, `:134`), so a
>   reader can perform minimum-image analysis.
> - **Capability plus static consistency.** The qualification suite **records
>   without downgrading** a run whose `rcut` exceeds half the cell
>   (`[PBC] pair cutoff ... exceeds half of the ... cell`,
>   `qualification/runner.py:819`, `:869`). The shipped 20-water box really is
>   smaller than twice its force-field cutoff; that is a force-field choice the
>   test suite must not overrule, but the record has to be visible -- same rule as
>   #229, "no silent pass without evidence".

> **Implementation status (2026-10-09, #316).** The step-by-step certification #315
> left open is landed: `workflow/records/diagnostics.py::_check_region_reconstruction`
> asks every recorded frame whether the bonds inside each center still rebuild shorter
> than half the cell (section 4.2), and two real-engine examples carry the reading
> (`examples/multicenter_pbc_nve.yaml`, `examples/multicenter_pbc_npt.yaml`). What that
> certification does NOT cover is stated there too: a low layer whose topology comes from
> a fixed template cannot be handed a center slice, which is measured rather than assumed,
> and it keeps a multi-center run with more than one center out of reach of the test environment's
> engine pair.

> **Implementation status (2026-10-08, #315).** Multi-center + periodic is open on the
> outermost-layer rule above; the variable-cell certification of that reconstruction is not
> (see section 4.2). Earlier status blocks follow as written.

> **Implementation status (2026-10-07, this review).** Stage D (variable cell)
> has landed since the phase table was written, and the four problems of section
> 3 are all fixed; section 4.4, 5.2 and 6 were corrected accordingly. Stage E
> (partial periodicity, region reconstruction, periodic embedding) is the stage
> still open, and the code still points at it from the two refusal sites.

---

## 1. Goals and overall judgement

The current version implements the PBC chain end to end for **full 3D periodic,
fixed-cell and variable-cell runs with a periodic-capable backend**. What a *geometry search* may
do with that cell is a separate design: [CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) owns the
gate that decides which methods may carry a box, the six-slot conjugate, the ride convention, the
pressure objective and the variable-cell examples. This file owns the cell a run carries, the
semantics of every frame and the snapshot that reaches a backend. What
remains open is not plumbing but physics definitions: partial periodicity,
molecules that straddle a box face inside a *region* (as opposed to the
whole-system case Stage C solved), and a periodic embedding model.

The three problems the design started from, and where each stands:

1. Framework and backend must interpret periodicity the same way -- settled:
   normalized `pbc` is the authority, and both layers refuse disagreement.
2. Every force evaluation must use the cell that belongs to those coordinates,
   variable-cell runs in particular -- settled by the frozen snapshot channel.
3. After an atom crosses a boundary, cross-frame differencing, region geometry
   and restart must keep the right coordinate semantics -- settled for the
   whole-system view (Stage C); region-level reconstruction is Stage E.

Sequencing judgement that still holds: accept NVE/NVT at fixed cell on one
backend for the whole system before accepting NPT, partial periodicity, and
periodic composite systems. Support for backend, method and periodicity
dimension is declared explicitly rather than discovered at runtime.

## 2. What exists now

| Link | State | Code |
| --- | --- | --- |
| Input and data fields | `pbc`, `cell`, `lattice_vector`, `volume`, `kpoint` on the system section; `image`, `fin_cell`, `barostat_mode`, `optimize_cell` added since the first review | `schema/config.py:1021-1025`, `:1032`, `:1230`, `:1363`, `:1588` |
| Cell geometry | pure functions, no state store: volume, boundary condition, wrapping helpers, unwrap, minimum image | `utilities/lattice.py:113`, `:188`, `:406`, `:468` |
| Input conversion | six-parameter and nine-component `cell` to matrix | `schema/normalize.py:283` (`_convert_lattice`), `:318` reads the six-parameter `lattice` |
| Coordinate wrapping | fractional modulo, triclinic, per-axis periodicity; raises without a usable cell; accumulates `image` | `data/geometry.py::apply_pbc`, `::record_wrapped_geometry` |
| Position in the MD loop | after position propagation, before the backend evaluation | `workflow/md.py:596-599` (`verlet_i` -> `apply_pbc` -> `calculator.run`) |
| Position in an optimization loop | after the optimizer's step, before the constraint solve and the next evaluation | `workflow/optimize.py:246-262` |
| Variable-cell propagation | cell deformation, six-parameter refresh, volume update, re-triangularization of the deformed lattice | `workflow/integrator/verlet.py:490` (`_retriangularize`), called at `:588` |
| Ensembles | pressure coupling, stress consumption, extended variables, cell algorithms | `workflow/integrator/ensemble.py:141`, `:820`, `:895`, `:1181`; stress at `:69`, `:393` |
| Synced fields | `cell`, `lattice_vector`, `pbc`, `volume` are synced; `image` and the extended DOF are not | `workflow/calculator.py:128-138` |
| State history | full system snapshot plus `nuclear_sample` | `data/core.py` (`_build_history_frame`) |
| Engine-reported cell | `fin_cell` -> `_apply_skill_cell` | `workflow/calculator.py:983`, `:1507-1554` |
| Periodic backends | CP2K cell block, xTB periodic coord, LAMMPS data template and persistent server | each skill's `service/builder/input_builder.py` |

`legacy/PyRAI3MD/Molecule/pbc_helper.py`'s basic wrapping logic was ported long
ago. Some early passages of `dev/methods/ENSEMBLE_DEVELOPMENT.md` and
`dev/methods/MD_DEVELOPMENT.md` still describe PBC as a placeholder; those
descriptions were superseded by the status notes above and by the sections of
those two documents that were rewritten since.

There is deliberately **no cell object** in the runtime chain. State flows
through the normalized `System` fields, and geometry is computed by pure
functions in `utilities/lattice.py`; a second independent cell state would be a
second thing to keep in sync.

## 3. Problems that were confirmed, and where each was fixed

This section keeps the original findings because code comments and tests still
cite the section numbers. Each entry ends with its current status.

### 3.1 Cell synchronization did not reach the actual backends

Was: `Calculator.sync()` updated the task configuration, but the CP2K backend
built its own skill configuration at initialization, and `calculate()` updated
atoms, coordinates, point charges and target states -- not the cell.
`workflow/dispatch/terms.py::_run_one` did not pass a cell either. Compounding
it, the MD loop ran `sync()` *after* the step's calculation, so an integrator
that changed the cell inside `verlet_i` could not deliver the deformed cell
before the forces were asked for. Adding sync fields or letting a backend read
the task config would not fix that timing.

**Status: fixed** by the per-evaluation snapshot of Stage B
(`workflow/calculator.py:540`, `:618`, `:1146`, `workflow/dispatch/terms.py:124`)
rather than by moving `sync()`. All ten backends now carry the `cell=None`
keyword (`skills/cp2k_skill/backend.py:158`, `skills/lammps_skill/backend.py:459`,
`skills/xtb_skill/backend.py:160`, `skills/nn_pes_skill/backend.py:318`,
`skills/pyscf_skill/backend.py:170`, `skills/orca_skill/backend.py:158`, and the
rest).

### 3.2 Periodicity was judged differently in different places

Was: the framework's wrapping and xTB judged periodicity by `pbc`, while CP2K
also treated any non-zero lattice as periodic and wrote `PERIODIC XYZ` in both
the CELL and POISSON blocks. Reproduced at the time: a non-zero lattice with
`pbc=[1,1,0]` and one with `pbc=[0,0,0]` each produced `PERIODIC XYZ`.

**Status: fixed.** `pbc` is the only authority
(`skills/cp2k_skill/service/builder/input_builder.py:189-201`), a molecular input
gets `PERIODIC NONE` + `PSOLVER WAVELET`, and mixed periodicity is refused before
a file is written (`schema/validate.py:408-420`,
`skills/cp2k_skill/schema/validate.py:203-209`). The geometry-vs-electrostatics
pairing (which solver is defined for which periodicity) is exactly what Stage E
has to define before the low-dimensional cases can open; the reference is
<https://www.cp2k.org/periodicity>.

### 3.3 LAMMPS box and atom-row handling had gaps

Was: `write_data_file` kept the template box and replaced the last three columns
of each atom row with coordinates -- but a LAMMPS atom row may carry three image
flags after the coordinates, so the trailing three columns are not necessarily
coordinates. Reproduced in a temporary directory: resizing the framework box from
10 to 20 left the generated data file at 10; a template with image flags kept its
old coordinates and wrote the new ones over the flags. See
<https://docs.lammps.org/read_data.html>.

**Status: fixed.** For a periodic run the box is rebuilt from
`lattice_vector`, keeping the template origin and following the current cell --
so a variable-cell step resizes the box instead of inheriting a stale one -- and
triclinic tilt factors are written in LAMMPS's bounding-box convention
(`skills/lammps_skill/service/builder/input_builder.py:762-776`). Rows are updated
by atom id with the atom style read from the Atoms header rather than guessed
(`:672` `_atom_coord_start`), trailing image flags are preserved (`:743`
`_with_image_columns`), and a periodic run writes this step's image counts into
those columns (`:709` `_image_rows`, `:801`). Framework-driven MD and LAMMPS-internal
MD remain different call paths and are accepted separately.

### 3.4 Input conversion and validation were insufficient

Was: `schema/adapter.py::FIELD_MAP` mapped `lattice` to `lattice_vector` early,
while `lattice_vector` demanded a matrix and `_convert_lattice` wanted the raw
six parameters; `lattice 10 10 10 90 90 90` failed validation while the same six
values through `cell` built; `pbc=[2,0]` built, because no unified three-element
0/1 check existed. `apply_pbc` returned silently when periodicity was on and the
cell was missing or singular -- a behaviour the then-current "no cell, no
operation" test accepted.

**Status: fixed.** The adapter special-cases the six-parameter form
(`schema/adapter.py:2490`, `:3038`, mapping declared at `:181`) and normalize
accepts it (`schema/normalize.py:318`); `pbc` is normalized to three values and
checked (`schema/validate.py:408-420`); `apply_pbc` raises
(`data/geometry.py::apply_pbc`). The checks were centralized as recommended: field shape, finite
values, periodicity flags, cell validity, angles and volume, and the mutual
consistency of the several cell representations.

## 4. Physical and data links that had to be added

### 4.1 Cross-boundary history coordinates and GSH (landed)

The main coordinates and the history frames store wrapped coordinates: that is
the chosen representation, not a leftover. The crossing counts wrapping discards
are accumulated by `apply_pbc` into `System.image` (`data/geometry.py`), which gives each of the three
uses a definite way to be read:

- **In-box coordinates**: for backends and display that require wrapped
  coordinates. A backend receives one snapshot of `(coordinates, image, cell)`
  from the same instant; without the image it cannot know which image it is being
  asked to record an atom in.
- **Continuous coordinates**: derived by
  `unwrap_coordinates(wrapped, image, lattice)` (`utilities/lattice.py:406`), used
  for cross-frame differencing, displacement statistics and restart. The
  `r-rpp`-style differences in `workflow/integrator/gsh.py:61`, `:161` read this
  way: when an atom crosses a box face, a genuinely small displacement is no
  longer read as a jump of nearly one box length. The regression case also pins
  the negative semantics -- the derived view guarantees **temporal** continuity
  only, and does not reassemble a molecule that straddles the face (with O at
  0.7758 and H at 9.9999 the unwrapped pair is still 19.2 A apart), so nobody can
  "helpfully fix" that later.
- **Whole-molecule coordinates**: wired for the case that needed it, a region
  cut out for a center task. `utilities/lattice.py::continuous_view` rebuilds the
  un-wrapped positions from the snapshot's own cell and crossing counts, and
  `workflow/multiscale/regions.py::materialize_region` slices from that view rather
  than from the main image (#315). The counts themselves come from the topology at
  configuration time (`schema/normalize.py::seed_initial_image` walking
  `utilities/bonds.py::derive_initial_image` over the perceived bond graph),
  which is what makes the unwrapped pair a bond and not a box diagonal. The
  certification #315 left open is now asked of the run itself: every recorded frame of a
  periodic run with more than one center is measured against the same bound
  (`section 4.2`, `records/diagnostics.py::_check_region_reconstruction`), and a frame whose
  counts cannot hold its centers together is named in the log and, under `control.debug`,
  stops the run rather than quietly handing an engine two fragments.

What is still a limit, and now measured rather than assumed: the counts stay in fractional
units, so the rebuilt bond tracks the cell it is read against. The bound above is what keeps
that from being silent, and `qualification/verification/test_multicenter_periodic.py` pins it
on a box halved under counts accrued for the old one (a 0.958 A bond rebuilding to 3.986 A
against a 2.250 A bound). Applying the same view to products nobody has asked for yet is
still open.

Exact transport of the continuous view under a deforming cell is Stage D work and
the fractionally-counted image is only exact at fixed cell, as stated above.

The trajectory file is a main-image product, so a molecule that straddles the box
looks broken in `md.xyz`; the per-frame `Lattice="..."` is what lets a reader
apply the minimum image, and connectivity is cheap to establish only under that
precondition.

### 4.2 Region splitting, embedding and geometry operations

`workflow/multiscale/regions.py` (`materialize_region`) no longer slices blind: it
takes the snapshot's `image` and `lattice` and cuts the region out of the continuous
view, which is what lets a molecular center backend receive a whole molecule
(#315). The framework also keeps the box away from that subsystem: no cell on the
work item, and `pbc`/`cell`/`lattice_vector` inerted in its static `skill_config`,
so a molecular engine's own "I am not periodic" backstop still means what it was
written to mean in standalone use.

Both paths are pinned by `qualification/verification/test_multicenter_periodic.py`,
which checks the static config and the live backend call against the same fixture,
and by its negative controls: the same methods inverted (`theory: [lammps, pyscf]`)
must still be refused, because then the molecular method is the outermost layer.

The rest of Stage E is still what it was, and what it still needs:

- periodic distances, angles, dihedrals and coordination judgements;
- reconstruction of complete molecule or region geometry;
- the direction of boundary-crossing bonds of linked atoms, and the way their
  gradients fold back;
- point-charge positions consistent with the mirror image of the local QM region;
- dynamic regions, constraints, CVs and later bias potentials.

Constraints are **no longer the placeholder this section once described**:
`workflow/integrator/operations.py:156-187` (`apply_constrain`) delegates to
`constraint_runtime.constrain_state` (frozen components, bond constraints,
RATTLE, velocity projections). Periodic constraint geometry landed with it:
`workflow/constraints/images.py` (`read_expansion:108`, `fold_back:258`,
`require_same_component:171`, `require_half_box:185`, `group_center:242`), used by
`constraints/state_adapter.py:227`, `:327`, `:331`, `:377-385`, `:429` and
`constraints/runtime.py:210`, plus the triclinic fractional wall
(`constraints/wall.py:23`) and the SHAKE counterpart (`constraints/shake.py:874`).
Multi-center runs used to refuse PBC outright; that refusal was retired on
2026-10-08 (#315) in favour of a granularity rule: multi-center means more than one method in
`theory`/`reference` and fixes no combination, and the only capability requirement is that the
OUTERMOST layer - the whole-system provider `planning/task.py::resolve_system_task` identifies,
whose input spelling is the theory tail - can carry a box. A center is a molecular subsystem:
it is handed no cell (`calculator._build_work_items` keeps `task_cell` None for it), its static
`skill_config` gets `pbc`/`cell`/`lattice_vector` inerted (`data/core.py`, the region block),
and its atoms are cut out of the continuous view rebuilt from the snapshot's own crossing counts
(`utilities/lattice.py::continuous_view`), so a molecule straddling a face arrives as one cluster
instead of as two fragments 8.4 Angstrom apart. The contract test that pins all of that, static
and live, is `qualification/verification/test_multicenter_periodic.py`.

**Who watches the cut.** Counts are stored in fractional units, so the rebuilt bond is a
function of the cell it is read against; a variable-cell run moves that ground under counts
accrued for the old one, and the failure is not an exception but a center arriving at its
backend torn. `workflow/records/diagnostics.py::_check_region_reconstruction` (check id
`region_reconstruction`, #316) asks that question of every frame the run already records, as
the fourth member of the recorded-step family in `dev/methods/MD_DEVELOPMENT.md` section 28.
A pair of atoms inside a center must rebuild to less than half of the shortest cell extent
(`utilities/lattice.py::half_box_bound`, the criterion `derive_initial_image` applies
before it will hand out counts at all), and a pair that rebuilds longer is a finding only if
the center would call it a bond, which is the nearest-image reading under the perception
reach `bond_lib(...) * BOND_SCALE`. The two questions are asked cheap-first: a box whose
centers are whole molecules answers "no pair rebuilds long" and never reaches the
nearest-image search, which is the only part of the watch that searches cells. A frame with
no counts at all is asked the same question with zero counts, because that is also what a
structure whose perception refused is left with, and the cut is just as torn there.

Measured on the 60-atom water box, per recorded frame: 0.4 us for a non-periodic frame,
0.5 us for a single-center periodic one (so every example that ships today pays that),
84 us with two centers, 107 us with twenty, and 408 us in the shape that does reach the
search - a center whose own atoms are scattered across the box, which no real center is.
Against the 4.20 ms a step costs (`dev/methods/MD_DEVELOPMENT.md` section 27) the operating
figure is 2.5% of one step, paid only on a step the trajectory writes.

Readings that say the watch has teeth, all on the straddling-water fixture: correct counts
say nothing; one atom's count moved by a cell rebuilds its bond at 15.716 A against a
4.500 A bound; the same counts read against a box halved to 4.5 A rebuild it at 3.986 A
against 2.250 A, which is the negative statement of section 4.1 measured rather than argued;
and a frame stripped of its counts entirely rebuilds it at the wrapped 8.447 A.

**What the real-engine pair certifies here.**
`examples/multicenter_pbc_nve.yaml` and `examples/multicenter_pbc_npt.yaml` run two PySCF
centers inside the shipped 60-atom TIP3P box with LAMMPS as the low layer, which is the
combination the outermost-layer rule above allows. The blocker this round removed was the
low layer's topology: a 2-layer plan asks LAMMPS for `L(real)` on all sixty atoms *and* for
`L(c)` on each center's three, and the engine reads a `<project>.data` template it does not
generate, so until #317 the run died at backend initialization on
`LAMMPS: atom id 4 in h2o.data is out of range [1, 3]` (measured in #316, which worked
around it by making the center cover the whole system — a shape that certified nothing about
the cut). The skill now cuts and compresses the user's system-level templates per task from
the row numbers the framework hands it, and a subsystem so cut is evaluated non-periodically
(section 4.2's rule, documented in dev/skills/LAMMPS_SKILL_DEVELOPMENT.md section 5.6a):
`boundary f f f`, no `kspace_style`, and every `…/coul/long` pair sub-style replaced by its
real-space sibling with the template's own cutoff numbers kept.

Why the box has to go, measured on this same fixture rather than argued: keeping it periodic
makes the subsystem carry the cluster's Ewald self-image energy, and the composite then
subtracts something `L(real)` never contained — 0.280090 kcal/mol (**4.464e-4 Ha**) per
center. Both treatments are exactly translation invariant (a 2 A shift of the molecule
changes each reading by 0.000000 kcal/mol, since LAMMPS wraps and takes the nearest image),
so position in the box is not what separates them; the self-image term is. The isolated
reading is then provably nothing but the center's own internal energy: for the water at rows
10-12 it equals the analytic bond-and-angle sum to 2e-15 kcal/mol, because `special_bonds
amber` carries into the cut (the intramolecular nonbonded pair stays excluded, exactly as in
`L(real)`) and every Lennard-Jones epsilon this template declares is 0.0.

Read back off the new runs' own products. The composite's `L(c)` for the two centers agrees
with three independent re-evaluations of the coordinates the backend was actually handed
(the skill's generated input, a hand-cut 3-atom file, the analytic bonded terms) at
0.003072046652927 Ha and 0.001408396300691 Ha, to 1.6e-16 Ha. The frame-level watch
`region_reconstruction` has now executed at C>1 against real engines, in NVE on a fixed cell
and in NPT on a deforming one, with `debug` turning any finding into an interruption; neither
run reported one. A center's indices are canonicalised to ascending at input time
(`schema/adapter.py::_parse_center_indices`), which is what lets a yaml list, a classic
keyword and a programmatic assignment describe one center — `TestTheRegionTaskIsHandedItsRows`
checks exactly that, declaring `[[3,1,2],[6,4,5]]` and reading back `[[0,1,2],[3,4,5]]`.
What the difference leaves in the low layer is what
mechanical embedding says it should: subtracting the isolated center from the whole box
(−199.582438 − (+1.212400) against the environment-only reference −177.076837 kcal/mol, same
box, water deleted, ids renumbered) leaves the environment plus a center-environment
exchange of −23.718001 kcal/mol (**3.780e-2 Ha**), while the center's own internal
1.212400 kcal/mol is what comes out.

The NPT cell now moves on a real pressure, which is what #318 fixed. A multi-center
plan's `stress` request had been landing on the molecular center tasks and not on the
whole-system one, because `planning/task.py` carried its own copy of the basic-capability
set (`{'energy','gradient'}`) while `planning/resolver.py` had already grown the
`'stress'` entry that exists precisely so the L(real) baseline keeps it. With the two
definitions unified and one rule added in `_new_task` - a task whose region is a slice
never carries a whole-system property - the flags read as the ruling says they should:
`L(real)` `['energy','gradient','stress']` with `stress=True`, both LAMMPS center
corrections `['energy','gradient']` with `stress=False`, and the PySCF centers no longer
hold a `stress` they cannot compute. The effect is in the run's own record:
`Pvir_bar` **−6050.718 bar** at the first step (+275.981 at the last), internal pressure
**−2277.314** to **+4701.684 bar** over 50 steps, `T_inst` 274.5 to 318.7 K. The box now
*contracts*, a 8.398804 → 8.371907 Å (V **−0.96%**), where the same input without a
virial had been inflating at +0.97% on the kinetic term alone - a 2% swing in the wrong
direction that no reading of the earlier run could have distinguished, because a barostat
with no virial still moves the cell. Against the one-water shape's 9.35e-4 Å worst-axis
drift, this is a box with real pressure in it.

Cost, from `pyrai3md_step.perf` with three LAMMPS evaluations and two PySCF evaluations per
step: `calculator_run` 187-250 ms/step for the 4-step NVE run (two measurements of the
same input) and 107-122 ms/step for the 50-step NPT one; whole runs take 4 s and 10 s. The
spread between repeats of one input is of the same size as the difference between the two
ensembles, because the QM SCF dominates it, so these are ranges and not rankings. Against
the one-water shape (58.9 ms per step, 55.4 ms inside the calculator) the only claim worth
making is that a step now carries three LAMMPS calls instead of one. The region watch is not
where the time goes - its per-step ceiling is measured separately in section 7.1.

What remains of Stage E is the other half: the old text follows, kept because the code
comments still cite it, and
limit should stay until the model is defined -- its check reads `pbc`, so it stays
consistent with the unified backend semantics by construction.

### 4.3 Trajectory output, restart and training data

The full state snapshot keeps the system, so cell information was never lost in
a restart. What was missing was the ordinary trajectory products, and the per-frame
cell landed with Stage C (`workflow/records/output.py:102`, `:134`). `image` and
`pbc` are still **not** written to xyz (`output.py:104-105`: main-image
coordinates plus `Lattice` only) -- a reader that needs them reads
`<title>.md.energies` / the extended record or the checkpoint.

The machine-learning interface declares its position instead of leaving it to be
inferred: `skills/nn_pes_skill/schema/validate.py:120-124` refuses any `pbc`,
because the geometric features and index weights were built and measured for
molecules only. A framework configuration that happens to carry a cell is not
evidence that a model supports periodic neighbourhoods.

### 4.4 Variable cell and supported scope

NPT had to be audited for coordinates, cell, energy, gradient and stress all
belonging to one evaluation instant, with the post-stage path that rescales
coordinates and cell examined for a history record that mixes deformed geometry
with pre-deformation results. That audit is what produced Stage B's snapshot
channel and the sync-timing decision recorded there.

The vocabulary is now declared rather than free:
`barostat in {off, berendsen, parrinello-rahman, mttk, scr}` and
`barostat_mode in {isotropic, anisotropic}` (`schema/validate.py:76-80`, `:89`,
`:116`), the thermostat x barostat combinations are a literature-defined table
enforced at validation (`schema/validate.py:104-119`; see
`dev/methods/ENSEMBLE_DEVELOPMENT.md` section 8.13 and CHANGELOG #257), an
anisotropic commit re-triangularizes the deformed lattice
(`workflow/integrator/verlet.py:490`, `:588`), a cell deformation that exceeds
what a lattice can carry raises instead of drifting into a NaN box
(`workflow/records/diagnostics.py:134` `CellDivergenceError`), and an
engine-reported cell re-enters the state through `fin_cell`
(`workflow/calculator.py:983`, `:1507-1554`).

Pressure definitions and allowed deformation directions for a **partially
periodic** system still cannot inherit the 3D bulk assumption; that combination
remains refused (Stage E). The geometry-search half of the variable cell - which method may carry
a box, the six movable entries, the stress conjugate, the pressure and the examples - is
[CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md); this section owns the dynamics side (barostats,
sync timing, the snapshot and `fin_cell`).

## 5. Design decisions (as built)

### 5.1 One cell data contract

The row-vector convention is unchanged: with cell matrix B, Cartesian coordinates
R and fractional coordinates S, `R = S @ B`. Lengths are Angstrom; conversion to
a backend's units happens at that backend's declared boundary.

- `lattice_vector` is the runtime authority for the cell matrix.
- the six parameters and `volume` are derived from it, so there are not three
  independently editable copies of the same state.
- `pbc` normalizes to three boolean/0-1 values and is the periodicity authority.
- origin convention: a non-periodic run keeps the template box, a periodic run
  keeps the template's low origin while rebuilding the extents
  (`skills/lammps_skill/service/builder/input_builder.py:762-776`).
- every state update still goes through the official get/update interfaces.

Legacy-format compatibility, defaults, and the refusal of invalid input all live
in the input layer. Runtime consumers read normalized data and do not re-guess
what an input meant.

### 5.2 One periodic geometry module

The cell conversions that used to be scattered over normalize, verlet, ensemble
and operations are collected in `utilities/lattice.py`: six-parameter/matrix
conversion and validation, volume (`:113`), boundary condition (`:188`),
Cartesian/fractional conversion, wrapping with the resulting image change,
minimum-image displacement (`:468`), and unwrapping (`:406`).

Wrapping and minimum image are different operations, and the triclinic case was
not assumed correct because wrapping was. `min_image_displacement` is now a
certified search: it widens the shell search beyond 27 cells when the cell shape
demands it, up to `_MIN_IMAGE_MAX_SHELL = 3` (`:456`), and raises `ValueError`
(`:546-554`) rather than returning a shorter vector when the certificate demands
a search beyond that bound. Targeted cases for strongly tilted cells are in
`qualification/verification/test_lattice.py`.

Whole-molecule reconstruction from topology is the part of this list still
missing (Stage E).

### 5.3 One consistent snapshot per evaluation

A calculation request carries atoms, coordinates, cell, pbc and the image counts
of the same evaluation, and the backend consumes that snapshot before writing
input. The contract as built:

1. the snapshot freezes after position and cell propagation
   (`workflow/calculator.py:540`);
2. region extraction and every backend evaluation use that snapshot (`:618`);
3. a backend's internal configuration gets the current cell through the same
   keyword (`skills/cp2k_skill/backend.py:187`, `skills/lammps_skill/backend.py:496`);
4. a parallel term never reads a mutating shared state -- the payload carries the
   snapshot (`workflow/dispatch/terms.py:124`, `:196`);
5. results correlate to an evaluation id and the geometry it used.

For a local-cluster model, whether the global periodic cell is inherited is a
decision of the scheme, and the current answer is no: only whole-system tasks
inherit it (`workflow/calculator.py:814`).

One timing note that this channel deliberately does not hide: `sync()` stays at
step end, so a backend that reads the task configuration rather than the
snapshot sees the previous step's cell
(`workflow/calculator.py:1641`; the same reason `image` is not in
`SYNC_SYSTEM_FIELDS`).

### 5.4 Capability checks and duty boundaries

Backends declare the periodicity dimensions, cell types, stress, variable cell,
embedding and composite combinations they support, and the planner refuses an
unsupported combination instead of letting a backend silently ignore a field:
`supports_pbc` in the manifest, gated at
`schema/validate.py:441-522`, with method-level granularity where the method
decides (`skills/xtb_skill/schema/method.py:33-57`).

The gate follows the theory/reference length matrix: with one theory method and
one reference, both own the box; with a layered theory queue and one reference,
only the theory tail (the environment/whole-system provider) owns the box and
the reference labels the first theory method's centre region; with one theory
and several references, the theory tail and the reference tail are checked.
More than one reference beside a layered theory queue is an illegal combination
(`validate_methods`). A centre-region backend may therefore be molecular even
when the environment layer is periodic; the runtime slices that task to its
region and clears its cell/PBC before the backend is called.

The backend owns its own periodic interactions and long-range treatment; the
framework owns the periodic geometry it uses itself, the history coordinates,
region operations, and data transport. Both need correct PBC semantics, and
neither gets to assume the other's.

### 5.5 Performance

Correctness first, then optimization: cache a cell inverse or decomposition while
the cell is fixed and invalidate it on a cell update; stop converting the six
parameters, the matrix and the volume repeatedly across modules; keep coordinate
operations vectorized instead of per-atom Python; introduce a neighbour list only
when an actual periodic distance/neighbourhood consumer needs one, with its
update condition defined. None of this was worth doing while the backend sync and
coordinate semantics were still open, and no caching state was added for matrix
operations that are not the bottleneck.

## 6. Staged implementation and acceptance

| Stage | Work | Acceptance focus | Status |
| --- | --- | --- | --- |
| A: data contract | fix lattice input, unify the periodicity decision, add the checks, centralize cell conversion | classic input and YAML/JSON equivalent; an explicit non-periodic run is not turned periodic by a backend; an invalid periodic cell errors | landed (2026-09-25, #216) |
| B: fixed-cell chain | carry the request through to backends, fix the CP2K periodicity axes and LAMMPS file generation | what the backend actually reads matches the framework's coordinates/cell/PBC; fixed-cell NVE/NVT reproducible | landed (2026-09-25, #216) |
| C: coordinate continuity | persist image counts, cross-boundary GSH differences, cell travelling with outputs and restart | cross-boundary differences agree with an equivalent continuous trajectory; restart and image round-trip; both sides of a backend get the same `(coordinates, image)` | landed (2026-09-28) |
| D: variable cell | sync timing, the new cell at the backend, stress, evaluation snapshot | every step's backend uses that step's cell; geometry pairs with E/G/stress; NPT verified on controlled systems | landed (barostat vocabulary, re-triangularization, divergence guard, `fin_cell`; CHANGELOG #223, #246, #257, remote NPT runs measured) |
| E: advanced combinations | partial periodicity, region reconstruction, periodic embedding, constraints on regions, CVs, model data | each combination gets its own physical definition and regression case, opened by capability | open |

## 7. Tests and evidence

### 7.1 What backs the statements above

- `qualification/verification/test_lattice.py` -- conversion, wrapping, image
  accumulation, unwrap round-trip, minimum image including strongly tilted cells.
  (This file moved here from the former `utilities/tests/`, which no longer
  exists.)
- `qualification/verification/test_schema_pbc.py` -- normalization, sentinel,
  partial-periodicity refusal, capability refusal at configuration time.
- `qualification/verification/test_calculator.py` -- the snapshot timing,
  including the variable-cell ordering and the region non-inheritance
  (assertion at `:509`).
- `qualification/verification/test_dispatch.py` -- the payload carries the cell.
- `qualification/verification/test_integrator.py:730`
  `TestApplyPbcWrapping`, with the missing-cell refusal at `:859`.
- `skills/{cp2k,lammps,orca,xtb}_skill/tests/test_input_builder.py` and the LAMMPS
  golden/persistent-server cases
  (`skills/lammps_skill/tests/test_input_builder_golden.py:140`,
  `test_persistent_server.py:480-492`) -- the backend side of the same contract.
  Not every skill has such a file, so these cover the three periodic-capable
  backends plus ORCA's non-periodic refusal, not "each skill".
- `qualification/verification/test_multicenter_periodic.py` -- the outermost-layer
  capability rule, the region cut out of the continuous view (static config and live
  backend call), the recorded-frame watch with its positive control, its three negative
  controls and its measured cost ceiling, the rows a region task is handed and the order
  it keeps (`TestTheRegionTaskIsHandedItsRows`, including the three-layer midlevel), and
  the shape of the two shipped periodic center examples
  (`TestTheShippedPeriodicMultiCenterExample`: two disjoint 3-row selections of one
  60-row topology, the second declared out of ascending order).
- `skills/lammps_skill/tests/test_ff_model.py::TestSelection` and
  `test_input_builder.py::TestSubsystemInput` -- the cut itself (row order, entity
  cutting and its audit, refusals, cache identity) and the subsystem's input text
  (`boundary f f f` before `read_data`, no kspace anywhere, the degraded pair styles with
  the template's own cutoffs, the fixed container a cluster cannot walk out of, and the
  resident and subprocess paths producing the same bytes).
- `examples/multicenter_pbc_nve.yaml`, `examples/multicenter_pbc_npt.yaml` -- run on
  this environment against PySCF and LAMMPS on 2026-10-09, with the readings in section 4.2.
- The full suite as measured on 2026-10-07 for the qualification round recorded
  in CHANGELOG #311: 2745 unit tests OK, 2 skipped.

An end-to-end periodic trajectory against external engines at a real cell now exists with
**two** centers inside one 60-atom LAMMPS box (the two examples above), which is the case
that could not run before: the low layer's subsystem selection is what closed it, and the
difference term was checked against three independent re-evaluations of the same cluster
(section 4.2). What still has no real-engine run is a **three-layer** plan, where the middle
shell would have to be a QM method this environment does not carry (xtb, cp2k) -- the row selection
itself is certified at plan level. Nothing here is a physics-level validation of, say, a
triclinic NPT run. The NPT example's barostat used to receive no virial at all; that
routing defect was fixed in #318, and the same input now measures `Pvir_bar` -6050.718
bar at step 1 and +275.981 bar at the last, which turns the box from +0.97% into -0.96%
(section 4.2 and dev/TODO.md).

### 7.2 What the next batch is

1. Partial periodicity: which solver is defined for which dimension (CP2K), how a
   reduced-dimension box is written (LAMMPS), and what a lower-dimensional
   pressure means -- refused today, so this is definition work, not test work.
2. Region reconstruction: straddling molecules inside a QM region, the direction
   of the crossing bonds, gradient fold-back, mirror consistency of the embedding
   charges. The connectivity question is now answered every recorded step at C>1 by
   `region_reconstruction` on real engines, and a subsystem center's own topology is cut
   from the user's template; what remains here is the *geometry* of a torn bond (a center
   that ends mid-bond needs a cap or link atom, which is `transform_spec`'s reserved job)
   and the embedding-charge mirror.
3. Periodic model data: whether a machine-learning interface declares periodic
   neighbourhoods and what its labels mean then (today it refuses `pbc`).
4. Step-by-step certification of the reconstruction under a deforming cell: closed on
   2026-10-09 (#316) for the single-center case and **opened and closed at more than one
   center on 2026-10-09 (#317)**, by the recorded-frame watch of section 4.2 running on the
   two `multicenter_pbc_*` examples with two centers inside the shipped 60-atom box, the low
   layer now able to evaluate a center slice through `atom_selection` instead of refusing it.
   What that closes is the reach of the watch, not its scope: a torn bond that ends at the
   region boundary still has no cap atom, and a three-layer plan still needs a middle-layer
   engine this environment does not carry.
5. Triclinic NPT: re-triangularization is unit-tested; its effect on a long
   trajectory (a slow global frame rotation of the recorded coordinates) should be
   measured against a fixed-frame reference before being relied on.

## 8. Document maintenance

Any PBC statement in `MD_DEVELOPMENT.md`, `ENSEMBLE_DEVELOPMENT.md`,
`DATA_CORE_DEVELOPMENT.md`, `CONSTRAINT_DEVELOPMENT.md`, `CALCULATOR_DEVELOPMENT.md`
and the skill documents must say which of the five stages it describes, and the
two obsolete claims must not come back: "adding wrapping completes PBC" and "the
sync fields exist, so a backend sees the new cell mid-step". Both were measured
false before they were fixed.
