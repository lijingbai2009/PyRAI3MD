# Variable-Cell Geometry Optimization (Cell Relaxation) Design

Last updated: 2026-10-24 08:45 (section 8.10 records the level-1 `&cell` log section and the `frozen_ride` / method columns appended to `opt_status.txt` after the cell block. Earlier 2026-10-22 21:40 (the cell and periodic material of the optimizer document moved here
in one pass: the cell material of `OPTIMIZE_DEVELOPMENT.md` sections 10, 15.1, 15.2, 17 and 18
now lives here, and every number below is the measurement that document carried, not a new one. The 2026-10-18
wrap-up ledger this file was born from is still its spine. `_example_timeout`, which one of those
records used, was deleted on 2026-10-20.)

## 1. What this document is, and what it is not

It is the design and implementation record of the optimize route's cell half: which combinations
may carry a cell and which are refused by name, the six lattice components a search steps, the
strain/stress conjugates they are judged on, the enthalpy under pressure, the coupling, the
periodic curvature frame a saddle or a crossing needs when the box is a variable, the convergence
record's cell columns, the examples that carry them, and their readings.

It is not the single-structure optimizer document.
[OPTIMIZE_DEVELOPMENT.md](OPTIMIZE_DEVELOPMENT.md) owns the algorithms, the atomic thresholds and
the atomic convergence report; this file states what a cell step adds to them. It is not the
ensemble document either: [ENSEMBLE_DEVELOPMENT.md](ENSEMBLE_DEVELOPMENT.md) owns the barostats,
the kinetic term and the NPT trajectory. And it does not own periodic geometry:
[PBC_DEVELOPMENT.md](PBC_DEVELOPMENT.md) owns the lattice data contract, the image ledger and the
snapshot that carries a cell to a backend. The cell half consumes those answers; whether they are
right is that document's question. What is specific to this file is what happens when a geometry
search is allowed to move the box, including the gate that decides which engine may carry one.

## 2. The rulings

**The user's eight items**, in their original order:

1. A 1-ethylene + water-box carrier and a `gdiis` variable-cell example (`gdiis` does not apply to
   transition-state searches).
2. The `meci`/`mecp` variable-cell examples on that same box, with the xtb environment first and
   LAMMPS in a later round.
3. An `anisotropic` test case.
4. `box_relax` cases for the LAMMPS skill, in the skill's tests and **not** in `examples/`.
5. A single-method gap case on CP2K's two-state numerical stress, through the cp2k skill's `num`
   route.
6. A CP2K reader for `tools/verify.py`.
7. A multiscale variable-cell run must record each layer's energy.
8. The bar threshold of #371 may be raised so the case can pass.

**Four rulings added in the same conversation** (hard premises of the plan):

- Carrier order: xtb first, LAMMPS next; the box may later become a configuration closer to the
  crossing for `meci`/`mecp`.
- #371: **"use the eighth step's reading"** - the threshold is set from step 8's measured residual,
  not from the last step's divergent reading.
- Item 5's cost: **"raise the budget to 2400 seconds"**. (The `_example_timeout` map that carried
  this was deleted by ruling on 2026-10-20, after round 20 measured the same two legs at 735 s and
  728 s inside the shared 1200 s budget; the gap example now runs `maxit: 2`. See section 8.5.)
- Item 7: **"add the output to `opt_status.txt`"** - the per-term energies go into the
  machine-readable table, not only the log block.

**Standing constraints** that stay in force: an optimization has exactly two outcomes, converged
or not converged; a number enters a document only after it was measured; a criterion must not be an
identity and each mutation arm must be shown to bite; a new configuration key is registered at
every consumer; a skill does not import framework internals; and no threshold is relaxed except by
item 8's ruling.

## 3. Three facts on disk that changed the plan

1. The source box is **1581 atoms** (525 waters + 1 ethylene in 25 A; the ethylene occupies rows
   1576-1581; the xyz comment line is empty, so it carries no cell). The LAMMPS mixed topology
   already existed as `c2h4_water25.data` (155,434 B). The "81-atom box" in the recipe is the
   **cut-down** carrier, not the source: one ethylene plus 25 waters.
2. `_example_timeout` existed and was idle at plan time (`manager._resolve_example_timeout` read the
   top-level map). Item 5 was the example that needed it. **It no longer exists**: the mechanism
   was deleted on 2026-10-20 (#399), and `_timeout: 1200` is the only budget.
3. The per-term energy **text** channel was already generic and independent of the cell: the
   assembly's `result['composite_terms']` reaches the log through `print_results`. What was missing
   was the **table** side: `utilities/opt_formatter.py` declared six atomic and five cell columns, its
   row writer accepted only 6 or 11 values, and the reader's header classifier
   (`utilities/print_results.py`) knew those two exact shapes.
   Item 7 was therefore an extension of those three places and every reader - not a new channel.

## 4. The supported surface and the gate

### 4.1 The shapes the gate judges

**Ruled 2026-10-17.** Three shapes, and multi-center runs are explicitly outside them:

| shape | engine spelling that works | what the gate does with it (measured, no engine started) |
|---|---|---|
| single method | `theory: [lammps]`, `[xtb]`, `[cp2k]` | plans `energy+gradient+stress`, resolves to `Optimize` |
| single method, a non-periodic program | `theory: [pyscf]` (and orca/molcas/bagel/nn_pes/qdm/openqp) | refused at validation: "the whole-system method ... does not support periodic boundary conditions but pbc is enabled; the outermost layer must carry the cell" |
| single-center multiscale, QM at the centre | `theory: [pyscf, xtb]` with `molecule.highlevel` naming the centre - **the second entry is the whole-system layer**, so the periodic engine goes last | plans the three layered tasks (`xtb` real + `pyscf` model + `xtb` correction), `energy+gradient+stress`, resolves to `Optimize` |
| single-center multiscale, periodic engine first | `theory: [xtb, pyscf]` | refused by the same message, and the message is right: it names the *outermost* layer as the one that must carry the cell |
| single-center multiscale, gap search on the centre | `theory: [pyscf, xtb]` with `molecule.highlevel` naming the centre and `molecule.ci: "3"` declaring the manifold | plans the three layered tasks, the `xtb` real layer answering `energy+gradient+stress` and the `pyscf` centre `energy+gradient`; resolves to `Optimize`, and `optimizer/meci.py::_cell_stress` takes its layered branch: the box is the outer layer's stress and `Cell Gap Term Source` prints `none: box carried by the outer layer` (#373). `examples/optimize_cell_meci_multiscale.yaml` is that run |
| gap / crossing (`mecp`, `meci`) + cell, one method | cp2k only, by ruling | accepted at the gate for cp2k and for xtb alike, so the gate is not what restricts it - see the producer note below |

### 4.2 Producers the surface still lacks

The last row is the one to read carefully, because acceptance there is not the
same claim as a reading. One of the combinations the surface allows still has no
producer, and it is engine-side, not framework-side:

- **a backend-supplied curvature** (`hess_driver: skill`) needs the same engine
  to answer a Hessian request. Among the three periodic programs none declares
  that capability today - `xtb_skill` and `cp2k_skill` list
  `energy/gradient/stress` (cp2k also `transition_dipole`, `geometry_optimization`,
  `abinitio_dynamics`; lammps also `geometry_optimization`, `classical_dynamics`),
  while the frequency capability sits with `pyscf` (`frequency`) and
  `orca`/`bagel` (`vibrational_frequency`), all of which the PBC gate refuses a
  box. So the framework's hybrid block (#365) can only be exercised by a
  criterion until one of the three gains a Hessian profile; a run that asks for
  it fails at the dispatched job by name, which is the documented behaviour for
  a skill with no Hessian profile, not a silent zero.
- **the gap's lattice derivative** (#367) reads `state_stress` for *both* named
  states (`optimizer/meci.py::_cell_gap_conjugate`: `sigma_j - sigma_i`, and it
  returns "no term" when either is missing, which the `Cell Gap Term` row prints
  as NaN). The `stress` capability the planner derives is one whole-system
  tensor, so an example for this row needs a periodic engine that returns the
  stress of a named second state - cp2k, by the ruling, over its numerical
  strain route. #382 supplies that row with `examples/optimize_cell_gap_cp2k.yaml`,
  which reuses the geometry, box and three-state manifold of the already
  certified `cp2k_skill/tests/tddft_stress_num.yaml` and adds only the optimizer
  and the gap. The qualification `num_stress._state_guard` puts on it is narrower
  than the earlier note: it refuses a run whose *centre spectrum* contains a
  vanishing gap (and a strain point that moves a root past half the smallest
  positive gap), so a search standing ON an intersection still has no producer,
  while one walking toward it does. The multiscale shape above is a different
  question with a different answer, and it has its own example.

### 4.3 Every search takes the keyword, and the three refusals

`optimize_cell: true` joins the six movable lattice entries to the optimized vector for `minima`,
`gdiis`, `ts`, `meci` and `mecp`. Three combinations are still refused by name, and each names the
half that is missing rather than banning a search:

- a saddle whose curvature comes from a backend (`hess_driver` other than `main`)
  raises when `Optimize` is built, before an evaluation is paid for: the matrix a
  backend returns covers the atoms, and six lattice rows filled with zeros are six
  directions of zero curvature, the lowest a mode-following search can be pointed
  at. Ask the framework for the matrix - its batch strains the box beside
  displacing the atoms and assembles `(3N + 6)^2` (`workflow/hessian.py`) - or drop
  the keyword.
- a saddle that also asks for a vibrational analysis raises from the same place:
  the analysis reports modes as frequencies *of masses*, and a lattice entry has
  none. The periodic frame does weight the six entries, but by the identity, to
  keep the frame square and report the rank of the rigid block
  (`workflow/frequency.py::periodic_vibrational_basis`); a box's curvature is read
  by the search that asked for it, not as a frequency.
- a crossing raises at the evaluation when the backend answers one whole-system
  `(3, 3)` tensor: its objective is the average of two surfaces, so its cell force
  is the average of the two conjugates and a single reading says what the box is
  under, not which surface put it there. The refusal names `(nstate, 3, 3)`, the
  shape that serves it (`optimizer/meci.py::_cell_stress`). It is not turned away
  for a limit its own backend never had, which is why it left the method-name list.

A strict condition is carried on the cell force for the three kinds the tree has priced - a bond,
an angle and a torsion (`workflow/optimize.py::CELL_MEASURED_CONDITIONS`). The angle and the
torsion were measured after the length: `projection.cell_virial` reproduces a finite strain of the
condition itself on every component a cell step can move, worst entry 1.13e-11 rad per Bohr for the
angle and 2.73e-12 for the torsion against derivatives of order 1e-2
(`qualification/verification/test_constraint_cell_coupling.py::TestAnAngleAndATorsionHaveTheSameCellForce`).
A fourth kind is refused by name until it is priced the same way, and `verlet._retriangularize`
remains the only allowed re-triangulation (A5/D-14).

### 4.4 The provenance of the table

The readings in it come from a gate-level probe that runs each input
through `parse_input -> build_config -> normalize -> validate -> build_state ->
resolve_workflow` and reports the planned capabilities and the per-task stress
ask. No engine was started, so the table says what the package accepts, not what
any engine answers.

## 5. The mechanics of one cell step

### 5.1 The variables: six triangular entries

`optimize_cell: true` does two things, and they were always meant to be one: it
derives the `stress` capability during planning (`planning/resolver.py`) so the
whole-system tensor is computed and unit-converted framework-side to
Hartree/Bohr^3, and it makes the six movable lattice entries part of the optimized
vector, driven by the stress conjugate to them.

The variables are the six entries of the row-vector lattice on and below the
diagonal (`CELL_COMPONENT_SLOTS`, defined in `utilities/lattice.py`), not the full
3x3. A full matrix step walked the box out of the triangular spelling within one
iteration and the LAMMPS data writer refused it outright; holding the three above
the diagonal costs nothing because those directions shear the atoms against the
box, and the 3N atomic components of the same vector already answer that question.

### 5.2 The conjugate: `V sigma^T H^-T`

The conjugate force is `dE/dH = V sigma^T H^-T`, not the volume times the stress:
stretching one axis by `dh` changes the volume by `h^2 dh`, and the reading that used the volume
alone asked 15.9 Hartree per Bohr for a force of 0.076 on the box this round measured it on.

### 5.3 Atoms ride the box

**A cell step holds the fractional coordinates.** This replaced the opposite ruling, and the
measurement that replaced it is the reason the section can be trusted at all. `V sigma^T H^-T` is
the derivative of the energy at *fixed fractional* coordinates; a run that keeps the atoms in
Cartesian space while the box moves is differentiating along another path, and the two differ by
exactly the nuclear gradient contracted with the fractional coordinates. On the equilibrated
60-atom (20-molecule) LAMMPS water box (8.399091 A, `|g| = 0.2217` Hartree/Bohr, relative strain 1e-3 per
component) central differences of the engine's own energy give, per slot
`(xx, yx, yy, zx, zz, zy)`, these ratios against `V sigma^T H^-T`:

| path | xx | yx | yy | zx | zz | zy |
| --- | --- | --- | --- | --- | --- | --- |
| atoms ride the box | 0.984 | 0.997 | 1.002 | 1.071 | 0.996 | 0.998 |
| atoms keep Cartesian | 0.053 | 5.69 | -0.47 | -9.53 | -0.75 | 36.9 |

with three sign flips on the second row. The identity between the two rows is exact and was checked
at three strain steps (1e-3, 1e-4, 1e-5): `FD_ride = FD_cart + sum_i g_i[b] S_i[a]`, and the sign of
`result['gradient']` was calibrated first with a single-atom displacement (FD `+8.859196e-3` against
the reported `+8.870976e-3` Hartree/Bohr) so the instrument could not flatter itself. The
integrator already rides (`workflow/integrator/verlet.py` scales coordinates and lattice together
through `exp(v_eps dt)`), so the alternative was a framework with two definitions of the cell
force, one per jobtype. Riding is implemented in `workflow/optimizer/base.py::_cell_ride`, on the
accepted step and on the shorter re-step after a rejection alike (`::_apply_step` is the one place
a joint step is written), because a run that rode on one and not the other would put its trial
geometries on two surfaces while claiming both describe one quadratic model.

### 5.4 The criteria: a pressure and a strain

The cell half is judged on a pressure and on a strain (`opt_csrms` and `opt_csmax` in bar,
`opt_cerms` and `opt_cemax` as the step of a lattice component over the length of its own row,
dimensionless), not on the atomic four. A lattice component and a nuclear coordinate are both
lengths, and their conjugates are both Hartree per Bohr, so the old rule was dimensionally
consistent - and size dependent, which is what a box criterion must not be: `9.807e-4` Hartree per
Bohr on the x row of the eighth round's water box (V = 106094 Bohr^3, that row 47.97 Bohr) is a
residual stress of 130.5 bar, and the very same reading on a box twice as long is 32.6 bar, three
times as long 14.5 bar. The factors are exactly 1/4 and 1/9 because `sigma = force x h / V` scales
as 1/L^2, and they are pinned by `qualification/verification/test_periodic_hessian_cell.py`.

The stress is rebuilt from the six relaxed components rather than read off the engine:
`utilities/lattice.py::cell_force_to_stress` solves the 6 by 6 map the slots define, because the
entries above the diagonal are held but not zero (one shear of the stress loads force on both
`B[1][0]` and `B[0][1]`), and because a run with a held condition adds a lattice term to the cell
force that no engine reports as stress - and that term is what decides where a constrained run
stops (#367): the ride term of the potential and the condition's ride term cancel entry by entry
at the atomic stationary point, so the box lands on the lattice's own rest shape in all six slots,
while dropping the term leaves it off by `ride / k`. Both readings are pinned slot by slot, for a
held angle and a held torsion, in `qualification/verification/test_constraint_cell_coupling.py`. On
an orthorhombic box the six-component solve and the closed form `sigma = B (dE/dB)^T / V` agree; on
a tilted one the shortcut is off by 0.59 in the same units, which the criteria also measure. The
readings stay on their own rows, because the two magnitudes are not the same event, and the run log
prints each cell row beside the threshold it was judged against.

**A missing stress is refused, not read as a zero.** `optimize_cell` on a backend
without the capability raises by name, so a run cannot report cell rows for a box
that never moved; the barostat refuses the same reading for the same reason.

### 5.5 Pressure: the objective and the coupling

**A cell that must relax under pressure is an optimizer run (#364).** The optimizer's objective
became the enthalpy when `opt_pressure` (bar) is named: `d(E + p V)/dB = V (sigma^T + p I) B^-T`,
which is `stress_to_cell_force` read on `sigma + p I`, so the same engine tensor, the same
conversion (`utilities/units.py::bar_to_au_pressure`) and the same tensile-positive sign the
barostats use carry, and the box is at rest where `sigma = -p I` rather than where it vanishes.

`opt_cell_coupling` says how many directions that pressure may move: `isotropic` projects the
lattice half of every step onto the one shape-preserving direction (the three row lengths in their
own slots) and judges the mean normal stress, leaving a deviatoric or shear residual printed but
unjudged, because no step the run can take would relieve it; `anisotropic`, the default, leaves all
six relaxed components free, which is what the three cell examples before this keyword did. What
the NPT run still owns is dynamics: the kinetic term, the temperature, the strain gate and the
trajectory record ([ENSEMBLE_DEVELOPMENT.md](ENSEMBLE_DEVELOPMENT.md) section 8). The two answers
must agree at the same pressure, and `examples/optimize_cell_pressure.yaml` is the vehicle the
criterion is written against.

### 5.6 Which direction a pressure moves the box

**It is read, not inferred (#391, #392).** The sign of `opt_pressure` is not it: the
ethylene-in-water box ends its own unloaded relaxation carrying `|sigma| = 1688.1 bar` of
compression, so its rest point `sigma = -p I` lies at a LARGER volume and a correct run expands.
The clause therefore asks two measurable things - that the residual of `sigma + p I` come down from
the first recorded step to the last, and that the loaded drift be smaller than what the same box
drifts with nothing on it. That second half needs a twin in the round, and the twin is matched on
four declared items plus one measurement: engine list, optimizer, declared coupling, starting
volume within `1e-6`, and `opt_pressure` absent. The engine list is in the key because a drift is a
property of the scoring function - an xtb arm and a cp2k arm of one YAML share everything else and
are not each other's unstressed self - and two mode directories of one example merge into one
reference while their drifts agree to the same tolerance, and count as two when they do not. When
the round holds no twin or several, the verdict says so by name instead of picking one by glob
order.

The isotropic side had no twin until `examples/optimize_cell_iso.yaml` (#392): with the anisotropic
pairing reading `paired against optimize_cell_aniso, which drifted 17.087668 A^3 unloaded: this leg
drifted 15.730721 A^3 under 1000.0 bar` on round `tmp_20261018_200317`, the isotropic leg printed
`paired against nothing` and its direction rested on the residual alone.

**The bar band and the `Converged` row are read off the same block and printed, not judged, and
that is a measurement rather than a convenience (#393).** Round `tmp_20261019_122659` is the first
run of these two files at 1700 bar and the first round that held the isotropic twin:
`optimize_cell_pressure` drifted 15.195667 A^3 against its twin's 17.614350 A^3 unloaded,
`optimize_cell_pressure_aniso` 14.748035 against 17.087668, both residuals falling step over step
(rms 10818.074 -> 1319.892 bar and 7800.131 -> 1817.053, max 11522.058 -> 2555.059), both ending
above the declared band with `Converged: False`.

Whether more budget closes that gap was tested on the one periodic-stress engine this environment has
(LAMMPS, the 1575-atom `water25` box, isotropic, 2000 bar, `maxit: 25`, a declared 100/100): the
residual fell 21076.0 -> 1058.4 bar, its lowest reading arriving at the second-to-last step before
the last stepped back up to 3730.2, with the enthalpy still dropping 0.096 Hartree a step and the
zero-residual volume extrapolated 178 A^3 beyond the last recorded frame. On the pressure carrier
itself the shape is a crossing: the round's 181.527 bar at 1168.8679 A^3 sits 0.49 A^3 from the
rest point interpolated from the two steps around it at 287.99 bar per A^3, and the step the run
then took was 3.95 A^3 - 8.07 times the distance left. A relaxation stops where its step lands, and
the step is bounded by the trust radius (`workflow/optimizer/minima.py:108`, initial 0.3 Bohr); as
of #396 that radius is an input keyword, `opt_trust_radius`, so the example therefore stays a
functional test at `maxit` 5, asks 1 and 2 decide pass or fail, and asks 3 and 4 print under
`; NOT JUDGED (functional-test budget)` so the numbers a round gives are still on the record while
nothing is claimed from them.

**The step bound is a keyword now, and what it buys is measured, not assumed (#396).**
`opt_trust_radius` is plumbed through six places (schema `DEFAULT`, both models that carry the cell
thresholds, `OPTIMIZE_TYPES` for the classic reader, the `optimize_params` whitelist, the
construction site `workflow/optimize.py`, and the classic writer), and it is unset-by-default on
purpose: `minima`/`gdiis` start at 0.3 and the transition-state search at 0.05
(`optimizer/ts.py:170`), so a framework default written into every run would have moved the TS legs
eight-fold - which is what the arm in
`test_cell_keyword_parity.py::TestTheStepBoundReachesTheSearch` catches (it reports `0.3 != 0.05`).
Measured with the keyword on the LAMMPS `water25` box (1575 atoms, isotropic, 2000 bar, 100/100, 15
steps, `trust_radius: 0.1`): the declared value does reach the run (the first progress block prints
`Trust Radius: 0.10000000`), the residual floor improves from 1058.4 bar to **220.189 bar** at step
13, the adaptive resizing still reaches its ceiling 0.5 (`minima.py:40-41`) within three steps, and
the band is still not entered - with the radius collapsed to 0.03125 the last two frames sit at
249.7 and 236.5 bar on a volume frozen at 15696.7969 A^3. That corrects the half-truth in the
paragraph above: the step bound decides how fast the search approaches, the floor belongs to the
carrier, and this box's `E(L)` is not smooth (the cutoff statement in
[dev/skills/LAMMPS_SKILL_DEVELOPMENT.md](../skills/LAMMPS_SKILL_DEVELOPMENT.md)). An engine-side
converged pressurised relaxation therefore has to be attempted on the carrier that reached 181.527
bar between two frames - the xtb 81-atom box - which this environment cannot run.

### 5.7 The barostat and the search answer one pressure

**`sigma = -p I` is one rest point for two paths, checked statically (#396).** The claim that the
NPT run and the variable-cell search agree at a given pressure was previously carried as a design
statement. What is decidable without an engine is now a criterion
(`test_npt_stability.py::TestTheBarostatAndTheCellSearchAnswerOnePressureTheSameWay`): driven
through the real `integrator/ensemble._berendsen` and `Optimizer._cell_gradient`, a box displaced
±2000 bar from its rest point is pushed back along all three row lengths by both paths, both drives
vanish at `sigma = -p I`, and the same holds at 0 and 5000 bar. The comparison is made with zero
kinetic pressure because that is the documented division of labour - the barostat relaxes the total
pressure (`p_int = p_kin - sigma`) while a cell criterion is deliberately built on the kinetic-free
stress (why the framework asks LAMMPS for `compute Pconf all pressure NULL virial`), and a fourth
criterion keeps that difference on the record by asserting the two paths DO part once a temperature
is in play. Two arms (delete the `p I` term on the search side; flip the barostat's drive) fail all
four criteria. What remains unproven is the endpoint claim - that two converged runs stop on the
same box - which the measurements above say no shipped carrier delivers at an example budget.

## 6. The box inside the searches

### 6.1 Every search

`optimize_cell: true` is askable for every search the optimizer workflow drives - `minima`,
`gdiis`, `ts`, `meci` and `mecp`; the refusals in
section 4.3 are properties of the evaluation, not of the method name. GDIIS takes the lattice as a
variable with no code of its own to make it do so (`gdiis.py:63, 65-72` hands the cell arguments to
`Optimizer.__init__`), and it is a minima search only: `gdiis` and `ts` are two classes in
`OPTIMIZER_REGISTRY`, so there is no `method: ts` + gdiis combination to ask for and
`optimize_cell_gdiis.yaml` is the cell row of that statement.

The two crossing cases on the 81-atom box drive state 1 against state 2 of a pyscf centre with xtb
carrying the box, and the centre's basis is a measurement rather than a default: TDA/sto-3g returns
its two lowest roots 11.7584 and 11.7600 eV apart, a pair 0.0016 eV wide, which would start the
search on top of the crossing it is sent to find; TDA/631gs (36 AOs, HF -78.031361 Ha) returns
8.9984, 9.9844 and 10.5589 eV, so the driven pair is 0.9860 eV apart. PySCF produces no derivative
coupling in this package (`pyscf_skill/schema/normalize.py:226` initialises `nac` to zeros and no
service module of that skill writes it), so the meci case walks the gap down with one
branching-plane direction and says so; a two-direction MECI needs a carrier that answers with the
coupling.

### 6.2 A saddle in a relaxable box

**A saddle in a relaxable box** takes the six lattice entries into the same search,
which means the curvature model must describe them too. The framework's own batch
does: with `optimize_cell` it adds six strain points, each a pair of ± displacements
of one triangular entry (`workflow/hessian.py::_cell_batch`), holds the fractional
coordinates so the atoms ride the box exactly as the optimizer's step does, and
differentiates the *cell force* rather than the energy - at the strain a finite
difference uses, the energy change is second order while the cell force is first, so
a difference of two energies would read the engine's print rather than the surface.
The atom-on-cell and cell-on-atom blocks are differenced on their own sides rather
than folded into one and symmetrized: writing only one half and letting symmetry fill
the other reported 0.052 where the independent reading is 0.026, twice the value. A
cell evaluation that returns no stress fails by name. A curvature taken from a
backend is **joined** rather than refused (#365): `hess_driver: skill` keeps the
block the engine computes, which covers the atoms, and the framework lays its own
six strained rows beside it (`workflow/hessian.py::_strain_cell_block_onto`), the
coupling half coming from the atomic gradient those same twelve evaluations return -
`d g_a / d H_b` by the same first-order reading as the cell rows, so no row is
zero-filled. A backend block that does not cover every atom is refused by name,
because the joined matrix would otherwise carry a silent hole. What stays refused is
the pairing that has no producer: an **excited** surface asked of a backend, since a
skill that answers a Hessian request answers it for its ground state (`ts` with
`opt_state_i` above zero on `hess_driver: skill`, and the same for the crossing
family), and a vibrational analysis, which reports modes as frequencies of masses
while a lattice entry has none. The frame the search projects onto is extended with
the box as well: the periodic basis (`workflow/frequency.py::periodic_vibrational_basis`)
weights the atomic half by sqrt(mass) and the six lattice entries by the identity, and
its rigid block carries the rotations' lattice part, `Omega x h` folded into the six
triangular slots, whose rank is reported rather than assumed (`Rigid Directions
Removed`). With no lattice the function returns the old basis element for element,
which is the criterion that keeps the fixed-cell runs bit-identical.

### 6.3 A crossing in a relaxable box

**A crossing in a relaxable box** steps the box on the same objective it steps the
atoms on, which is the average of two surfaces, so its cell force is the average of
the two conjugates: `CrossingOptimizer._cell_stress` reads `state_stress(i)` and
`state_stress(j)` and returns their mean, where a relaxation reads the one
whole-system tensor the base class carries. The state-resolved reading is the shape
the properties schema already declares and the numerical-strain route in the CP2K
skill already produces, `(nstate, 3, 3)`; a backend that answers a single `(3, 3)`
is refused at the evaluation with that shape named, because one tensor says what the
box is under and nothing about which surface put it there. Reading the active state's
tensor twice would not be a coarser crossing search but a run whose box and atoms sit
on two different objectives inside one generalized vector, and the criterion proves
the difference rather than arguing it: two surfaces whose box springs rest at 18 and
20 Bohr put the average's minimum at 19, so a correct reading calls the box done at
19 with a cell force of exactly zero (0, 0, 0, 0, 0, 0) while a single-surface
reading, at the same geometry and the same evaluation, reports 7.41e-3 Hartree per
Bohr on the first slot and refuses to converge. Started short of both surfaces, the
same search travels to 19.0000 Bohr in 8 steps
(`qualification/verification/test_state_stress_pair.py`).

The gap's lattice derivative is taken whenever the backend supplies a
state-resolved tensor. `optimizer/meci.py::_cell_gap_conjugate` reads
`state_stress` for both named states and adds `V (sigma_j - sigma_i)^T B^-T`,
scaled by `GAP_MIXING * gap` along the unit derivative, beside the mean in
`_cell_gradient`; `Cell Gap Term` reports its norm and its source (round 20, both
carriers: 0.01705835 / 0.01733804, `Source: state-resolved stress`). What is not
available is the same derivative at a degenerate central spectrum: the CP2K
numerical-strain route refuses to track states through a vanishing gap, because
energy-only state matching cannot follow a root crossing. The exact intersection
is not a target: at the degeneracy the roots swap, only the gap-space gradient
and stress vanish, and the individual surfaces are neither smooth nor local
minima, so they cannot serve as convergence criteria; TDDFT also carries error
at the crossing. The method therefore stops at an approximate structure near the
crossing, and the missing gap derivative there is a statement of that design
boundary, not an owed producer. A layered/multiscale plan sets the term to zero
by design, because the outer layer carries the cell.

### 6.4 When the engine owns the box

**A box a skill-driven path relaxes is the engine's business, and the framework says so by name.**
`optimize_cell` has exactly one reader - the framework's optimizer, which commits the six lattice
rows through its own acceptance test - so a run that hands the steps to a skill and keeps the
keyword is asking for a box nobody moves. `resolve_workflow` refuses that pairing and names both
routes (#367): iterate on the framework, or declare a profile that owns the lattice.

For LAMMPS that profile exists (#366): `profile: ff_optimize_cell` with `box_relax: iso|aniso` and
`box_relax_pressure` in bar writes `fix box/relax <style> <P> vmax 0.01` beside the `minimize`, and
the box the engine stopped on comes back on the `.cell` channel rather than being assumed from the
one that was asked for. Measured on the test environment with `lmp_serial` on the shipped 60-atom h2o
fixture: the `8.399091` A cube the data file declares ends as `8.1331 / 7.8954 / 8.3657` A (three
edges, three answers), the engine's own `restart.data` carries the same numbers, and the fixed-box
control (`ff_optimize`, same topology) writes no `.cell` file at all. LAMMPS `minimize` itself has
no cell degree of freedom, which is why the `ff_optimize` profile reports no `fin_cell` while a
`fix npt` run does ([LAMMPS_SKILL_DEVELOPMENT.md](../skills/LAMMPS_SKILL_DEVELOPMENT.md) section
6.4); a cell reported after an engine-side optimization would be an invention.

## 7. The fixtures the cell set runs on

### 7.1 The 81-atom ethylene-in-water cube

**Recipe** (a one-off development-host operation; the result was inlined into the YAML and no
script was shipped, following "generated intermediates do not enter the package and are deleted
after use"): cut **whole molecules** out of the 1581-atom box around the ethylene's centre of mass -
a water enters only if its oxygen is within R and all three of its atoms come with it, never half a
molecule - with R solved so that exactly 25 waters qualify. Translate the cluster into the positive
quadrant and take a cubic cell of `L = max(9.5 A, cluster span + 1.0 A)`. **Never wrap it**: GFN-FF
perceives topology and would read a face-crossing O-H bond as broken, and wrapped coordinates were
never the set the engine actually used.

**Ordering**: the ethylene's six atoms go first (rows 1-6), the waters after, so
`highlevel: [[1, 2, 3, 4, 5, 6]]` is the whole ethylene.

**Audit**: the box is 81 atoms in a 10.5 A cube, built with Packmol 21.2.1 under its periodic
constraint and a 2.3 A intermolecular exclusion. Audited with this package's own perception:
`bond_graph` finds 26 connected components under PBC, so no molecule straddles a face; the closest
intermolecular contact is 2.518 A and nothing sits under a non-bonded floor once bonded and 1-3
pairs are excluded; the density is 0.654 g/cm^3 and the half box 5.250 A, which is the number the
next round's LAMMPS carrier has to keep its pair cutoff under. A crop of the shipped 525-water box
was tried first and rejected on measurement: 0.543 g/cm^3 with O-O pairs at 1.914 A, because a
sphere cut from a periodic box presses on its own image.

**Non-bonded clash check** (the correction of an earlier pass that reported a 0.96 A O-H bond as a
clash): reuse what the package already has instead of writing a second perception - the lattice
helpers plus `perceive_bonds` and `bond_graph` for the bond pairs. The exclusion set is the bond
pairs together with their second neighbours (a water's H-H at 1.51 A and the ethylene's H...H at
about 2.2 A both sit there); only intermolecular and same-molecule non-bonded pairs are judged,
per element pair (H...H, H...C, H...O, C...O), and a failure prints the pair and its distance. The
same run prints the smallest intermolecular distance and the cutoff/half-box ratio, which the
LAMMPS carrier switch later reads directly.

### 7.2 The LAMMPS boxes

Three boxes are on record and each answers a different question:

- **the shipped 60-atom `h2o.data` fixture** (8.399091 A) is the mechanism carrier and cannot carry
  a converging relaxation. It declares `pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0` in an
  8.399091 A box, so the cutoff is 2.4x the half-box and `E(L)` carries kinks where pairs cross 9.0
  and 10.0 A and where the minimum image changes. Measured as the difference between two equivalent
  spellings of one physical structure, that floor is `5.88e-5` Hartree, while
  `MinimaOptimizer.reject_on_energy_rise` fires above `1e-6` - 58x finer - so every cell step is
  judged a rise, the radius collapses (0.3 -> 7.3e-7 Bohr) and the run stops after five
  evaluations. A relaxation acceptance therefore needs a box whose cutoff is under half its shortest
  extent, which the shipped fixture is not.
- **the 25 A cubic TIP3P box** (525 waters, 1575 atoms) from the local `ffauto`, equilibrated by
  ffauto's own NPT template (296.0 K, +42.7 bar, -9.87 kcal/mol per water, 0.9990 g/cm^3,
  orthorhombic 25.3866 x 23.2680 x 26.6154 A, no tilt, and no cutoff warning: the smallest half
  extent is 11.634 A against a 10.0 A cutoff). It is the box the ride pricing and the trust-radius
  readings in section 5 use.
- **the 1581-atom mixed topology** (`c2h4_water25.data`, 25.237532 x 24.433851 x 25.611381 A,
  smallest half extent 12.22 A over a 10.0 A cutoff, net charge 0.000000, centre = molecule 526 =
  rows 1576-1581). It is the carrier of the two force-field crossing legs in section 8.9.

### 7.3 The converged relaxation on the 25 A box

**That box was built and the relaxation converged (2026-10-13, CHANGELOG #341).** Priced on it, the
ride path reproduces `V sigma^T H^-T` to 0.9992 / 1.0181 / 0.9965 / 1.0093 / 0.9344 / 0.9991 while
the Cartesian path gives 1.2348 / 6.8765 / 0.7275 / -0.6588 / -1.6494 / 0.4762 with two sign flips -
the second independent box to separate the two. The relaxation then ran to `Converged: True` in 799
steps (118 s): E -10.910679 Hartree, cell volume 15721.6 -> 13525.1 A^3 (-14.0%), `cell_grms`
5.016e-4 and `cell_gmax` 9.807e-4 against criteria of 5e-3, every one of the six rows inside its
threshold.

Two readings that go with it, neither of them flattering. The converged density is 1.161 g/cm^3,
16% above the 300 K liquid, because that force-field table writes the oxygen LJ epsilon positive,
which makes the Lennard-Jones half repulsive-only (measured: `E_vdwl` +0.7263 kcal/mol/atom at
+0.102, -0.8469 at -0.102) and leaves the cohesion to the Coulomb term; the total is right for
TIP3P and the partition is not. The cell criteria were not scale invariant - `cell_gmax` 9.807e-4
Hartree per Bohr is a residual stress of 130.5 bar on this box (sigma = force x h / V, measured with
the package's own `au_pressure_to_bar` on the volume that box recorded), while the same number on a
box twice as long asks four times less, and on the 60-atom water box about 7.5 times more. That is
closed in #363: the cell half is judged on `opt_csrms`/`opt_csmax` (bar) and `opt_cerms`/`opt_cemax`
(strain), whose defaults are the pressures and strains the atomic four already mean on this box.

What the bar and strain readings cannot tell, and #383 added a check for: which directions the box
was allowed to move in. `opt_cell_coupling` decides exactly one thing - whether the lattice half of
a step is confined to the single direction that scales the three rows together
(`optimizer/base.py::_cell_direction`, :757-780, which returns that unit vector for `isotropic` and
None for `anisotropic`) - and both choices move the box, so the cell columns and the volume drift
are green either way. `_check_cell_shape` now reads the trajectory's own `Lattice=` comments and
compares the ratios of the row lengths against the declaration: `isotropic` with the ratios changed
is a step that was never confined, `anisotropic` with the ratios unchanged to below
`CELL_SHAPE_FLOOR` = 1e-5 is the projection still being applied. The floor is set from a product
rather than from taste: on round tmp_20261018_082139 every anisotropic leg did change its row
ratios, the smallest by 9.7e-05 (cp2k, on its two-step budget), while the declared-isotropic leg
left them at exactly zero as its three rows all moved 10.000000 -> 9.976689 A - the two readings
this separates do not overlap there. It prints the measured change whichever way the verdict goes,
on its own `[COUPLING]` line rather than inside the `[CELL]` sentence, because `tools/verify.py`
section D reconstructs that sentence from the status record and compares it byte for byte with the
round log. Until that round only `optimize_cell_pressure.yaml` named its coupling; the six other
cell cases were running `anisotropic` by inheritance, and `optimize_cell_aniso.yaml` is the first to
say so while asking for the shape change that shows it.

## 8. What the cell rounds landed

### 8.1 `optimize_cell_gdiis.yaml`

`jobtype: optimize`, `theory: [xtb]`, the cubic cell and `ci/spin` in `molecule`, `method: gdiis`,
`maxit: 5`, the five atomic thresholds copied from `optimize_cell_xtb.yaml`, `optimize_cell: true`
and the four cell thresholds, `xtb: gfn_ff` with `gfn_ff_gradient`. **Zero code changes**: the GDIIS
optimizer forwards the cell parameters to `Optimizer.__init__` unchanged, and the mechanism-level
evidence was already in `test_optimizer_crossing.py`. Two places state that **GDIIS does not apply
to a TS search**: the example header (`GDIISOptimizer` has only a minima branch; `ts` is another
class in `OPTIMIZER_REGISTRY`, so `method: ts` + gdiis does not exist) and the
target list in `OPTIMIZE_DEVELOPMENT.md` section 15. Criterion: the runner's cell
post-check (columns present, volume moved) plus an engine-free case asserting the GDIIS report
carries the four cell readings and a volume that changes with the step.

### 8.2 `optimize_cell_meci.yaml`, `optimize_cell_mecp.yaml` (xtb outer layer first)

`theory: [pyscf, xtb]` - **the centre first, the whole system second** (the #372 ruling; the reverse
order is refused by name by the PBC gate; section 4.1), `highlevel: [[1..6]]`, `ci: "3"`, `spin:
"0"`, `method: meci | mecp`, `state_i: 1`, `state_j: 2`, `maxit: 5`, `optimize_cell: true` and the
four cell thresholds, `xtb: gfn_ff/gfn_ff_gradient`, `pyscf: {method: tda, basis: sto3g}` and **no
`scratch_path`** (the #375 scoping lesson). `meci` needs a derivative coupling, `mecp` does not.

**The one cell that had to be measured before writing the files**: whether the whole ethylene's
lowest TDA/sto-3g roots are non-degenerate. The earlier example uses a water centre precisely
because a bare C-C pair is degenerate under TDA and reports `Not enough eigenvalues`. The local
PySCF measured it in one single point; had it been degenerate, the centre would have gone back to
the water form and the header would have said so rather than hiding it behind extra roots.

Criterion: the three `Cell Gap Term Source` labels (`none: box carried by the outer layer` for the
layered branch, `state-resolved stress` for the non-layered one) and a gap cell term that is
neither zero nor NaN; `test_state_stress_pair.py` walks the new examples automatically. The LAMMPS
environment was not part of this round (no mixed `.data` in the package then).

### 8.3 `optimize_cell_aniso.yaml`

`anisotropic` is the framework default, and the seven variable-cell examples at the time ran it
without saying so. The new example declares `opt_cell_coupling: anisotropic` **explicitly**, at zero
pressure, on the section 7.1 box.

The criterion must be non-identity and must contrast with the isotropic sibling: the runner's cell
post-check reads only the volume, so a clause compares the three row lengths of the first and last
`Lattice=` comments (`opt_geom.xyz`): `isotropic` requires the length ratios to be unchanged,
`anisotropic` requires them to change. Two mutation arms: replace the direction projection with the
identity, and replace the ratio test with a tautology - `test_cell_optimization_check.py` must go red
on both.

### 8.4 The LAMMPS skill's `box_relax` cases

Two YAMLs in `skills/lammps_skill/tests/`, not in `examples/`: `ff_box_relax_iso.yaml`
(`profile: ff_optimize_cell`, `box_relax: iso`, `box_relax_pressure: 0.0`) and
`ff_box_relax_aniso_pressure.yaml` (`box_relax: aniso`, `box_relax_pressure: 1000.0`). The planner
picks them up without code changes. Hard constraint: their `pbc`/`cell` must equal the `h2o.data`
fixture box (8.399091 A), because `test_unification.py` asserts each case's cell against the
fixture. Registration: the skill's case list (nine -> eleven), the `pmd_env.yaml` per-case keys and
the CHANGELOG's "nine examples" count.

The criterion is **mechanism, not physics** (the fixture's cutoff is larger than half the box, which
the existing case already declares): `fin_cell` exists, the `.cell` line parses back, and the iso and
aniso spellings differ in their syntax (iso writes one value and a vmax).

### 8.5 `optimize_cell_gap_cp2k.yaml` (two-state numerical stress on a single method)

Geometry, box and three-state manifold were taken from the certified `tddft_stress_num.yaml` case
(ethylene at 10 A, `nstate: 3`), so the cell adds only the optimizer and the gap, no new physics.
`theory: [cp2k]` (single method, no `highlevel`), `cp2k: {method: tddft, profile: tddft_gradient,
functional: pbe0, cutoff: 400, ref_cutoff: 50, execution_mode: local}`, `method: mecp`,
`state_i: 1`, `state_j: 2`, `optimize_cell: true`; `ci/spin` travel through the calculator's sync
path, the same convention `optimize_cell_cp2k.yaml` uses.

What the cell adds is the **consumer side**: the non-layered branch of `meci.py` reads two states'
tensors so `_cell_gap_conjugate` can report `state-resolved stress`, the one of the three labels
that had no real-engine reading.

**The state guard**: it refuses a central spectrum with a zero gap, uses `0.5 x` the smallest
positive gap as its tolerance, and requires every strain point to return the same number of roots
with no root moving more than that tolerance. The route therefore holds only while the search
**approaches** the crossing and never stands on it. That boundary is deliberate and is recorded in
`OPTIMIZE_DEVELOPMENT.md` and `TODO.md`: at a degeneracy the roots swap, only the gap-space gradient
and stress vanish, and an individual surface is neither smooth nor a local minimum, so it cannot
serve as a convergence criterion; the method returns an approximate structure near the crossing.

**Cost, as ruled**: `full` scope is 1 central + 12 strain points = 13 cp2k calls per evaluation, at
most 390 s per step measured, so five steps is about 1950 s - above the 1200 s budget. The ruling
raised that example's budget to 2400 s through the then-live `_example_timeout` map.
**That mechanism was deleted on 2026-10-20** (#399) and the example now runs `maxit: 2` with
`multitask: true` / `multipool: 2`, measured at 735 s and 728 s inside the shared budget. Before
that, `maxit: 4` needed about 1560 s and both of its legs returned `Test timed out (1200s)` (round
tmp_20261016_075328, CHANGELOG #359).

### 8.6 The CP2K reader in `tools/verify.py`

A new `_engine_cp2k(calcdir)` globs `calcdir/*.log` and **claims the directory only if the file
carries CP2K's own banner prefix** (` CP2K| `): the readers are tried in sorted order with `CP2K`
first, so a reader without a gate would steal another engine's log. On a claim it takes the last
`Total energy:` token (the SCF prints it repeatedly) and returns it; the unit is Hartree, so no
`ENGINE_VARIANTS` entry was added. The selftest plants a `demo_cp2k_opt` leg whose log carries two
`Total energy:` lines with only the last one equal to the record - which also pins "last, not
first". Two mutation arms: change the last line by 1e-3 (section C2 reports the mismatch) and remove
the banner (the directory falls into `dirs_unread`, `reader_missing` +1). The fixture is inline in
the selftest; no `.log` ships.

### 8.7 The per-term columns in `opt_status.txt`

**What is written**: signed **contributions** (`contribution x term energy on the driven state`),
not raw energies, so that "the row's term columns sum to its `energy` column" is a machine-checkable
identity - exactly the reconciliation the assembly legs lacked. The raw per-term energies stay in
the `&composite energy decomposition` log block, and both readings come from the same
`build_composite_terms` call rather than a second computation.

**Column names**: the `term_label` tokens (`high[qm]`, `mid[qm2][2]`, ...), which are whitespace-free;
their order follows the assembly's own sort key (high -> mid -> low -> per centre), so one table has
one reading. The number of columns varies with the layer/centre plan, so the columns are dynamic.

**Four places changed together** (missing one silently mis-columns the table): the header writer
gains a third shape (atomic / + cell / + term columns) through the one width constant; the row
writer's "6 or 11 values" becomes "6 or 11 plus the term count"; the header classifier becomes a
**prefix** test (the fixed 6/11 prefix unchanged, whatever follows is term columns), otherwise a
layered cell table classifies as `WRITER_FOREIGN` and both the runner and `verify.py` go blind; and
the optimizer's header assembly takes its column set from "cell or not **plus** the step's
`composite_terms.labels`", with a same-shape assertion beside the existing cell-column length
assertion. Readers key rows by column name, so they needed a confirmation rather than a rewrite.

Criterion (engine-free, written before the code): the existing pinned atomic and cell headers stay;
a third round-trip case requires the classifier to claim a term-column table and the reader to read
the term columns by name; an end-to-end case asserts `sum(term columns) == energy` on a layered plan
with `optimize_cell: true`. Two mutation arms: flip one term's sign, and drop `composite_terms` from
the result - both criteria must go red.

### 8.8 The pressure threshold of #371

The threshold lives in the example's own `opt_csrms`/`opt_csmax`, flows through the optimizer's
constructor and is read back by the runner's pressure check. Changing only the declared value
satisfies both "the residual is inside the band" and `Converged: True`; the schema defaults, the
four atomic thresholds and the two-outcome rule are untouched.

**The reading that set it** (the user's "use the eighth step"): round 14's trajectory went 1276.0 bar
-> 171.9 bar at step 8 -> and back up to 969.6 bar over steps 31-40 (radius pinned, no rejections).
The example therefore runs **to step 8** (`maxit: 8`), and its two thresholds are the round tens
covering step 8's rms and max readings, re-read from that round's own `opt_status.txt` - not copied
from a number this document quotes. The example header and the CHANGELOG record it as a user ruling
("the example's budget is 8 because the ruling set the threshold at step 8, not the uniform 5; #371's
verdict and #380's withdrawal of `stalled` stand"), and the runner's comment that said "no threshold
moved" was corrected to cite the ruling.

### 8.9 The force-field crossing carriers (#392)

`examples/optimize_cell_meci_lammps.yaml` and `optimize_cell_mecp_lammps.yaml` put a six-atom PySCF
TDA centre inside a 1581-atom TIP3P box whose outer layer is LAMMPS on a mixed topology - the case
neither the electronic gap examples (xtb or cp2k scoring every atom) nor the single-layer
`optimize_cell_lammps` (no centre at all) covers. This environment has `lmp_serial` 22Jul2025 and PySCF, so
both legs ran: five steps in 28 s and 27 s, volume 15793.26056503894 -> 15863.58332503204 A^3
(70.322760 of drift), `cell_gmax` ending at 7.673e-02 Hartree/Bohr, the `Cell Stress RMS` row coming
down 16175.1 -> 6105.4 bar against the declared 40/60, `Converged: False` at the budget, `[COUPLING]`
at 3.648e-04 against the declared `anisotropic`, and the gap block printing `Cell Gap Term:
0.00000000` with `Source: none: box carried by the outer layer` - the same source label
`optimize_cell_meci_multiscale` shows, now read on a classical layer. Two findings come out of those
products. The first is the identity boundary of section 8.7: over the five rows the term columns
summed 1.51e-2 to 1.28e-2 Hartree BELOW the energy column, where the non-crossing
`optimize_ts_multiscale` rows had summed to it exactly (0.0 over 64 rows, round 16) - so `sum of
columns == energy` belongs to one assembly shape and not to the table, which is what
`test_workflow.py::TestEachTermIsReadAtTheStateTheAssemblyRecords` now pins in both directions. The
second is that this carrier cannot separate the pair it was built as: the two legs' status tables
are row for row identical, because the PySCF route supplies no derivative coupling and a crossing
search without one is the same search whichever keyword asked (`meci.py:60`,
`uses_derivative_coupling` is the only difference). A leg that diverged from its sibling here would
be a finding about the branching plane, not about the force field.

### 8.10 The cell in the run log and the columns after it

The run log now prints the box itself at `verbose >= 1`: an `&cell in Angstrom`
section with the three lattice rows, the six cell parameters (`a b c` in Angstrom,
`alpha beta gamma` in degrees), the volume in A^3 and the `pbc` flags. It sits
after `&coordinates` and before `&background charges`; a non-periodic run gets no
section below level 3 (and the `Not computed` marker at level 3). The cell was
previously recoverable only from the `Lattice=` comment of `opt.xyz`/`opt_geom.xyz`
and from the volume column.

`opt_status.txt` keeps its seven (or twelve) fixed columns in their old order. A
variable-cell run appends `frozen_ride` after `volume`, because the ride's work is
a cell reading the same step produced; a saddle adds `neg_modes`/`mode_eig` (and
`rigid_removed` on a box); a crossing adds `cross_gap`/`branch_dirs`/`gap_norm`/
`coupling_norm`. `classify_header` reads the fixed prefix, so every earlier reader
keeps treating the table as the same writer's record; a row that omits a declared
method column is refused by `status_line`. String readings (`Cell Gap Term Source`,
`Direction`, `Role`) stay in the log only.

---

## 9. The examples

The variable-cell half of the route is carried, since #355, by seventeen shipped examples that
ask the box to move: `optimize_cell_aniso`, `optimize_cell_cp2k`, `optimize_cell_gap_cp2k`,
`optimize_cell_gdiis`, `optimize_cell_iso`, `optimize_cell_lammps`, `..._lammps_angle`,
`..._lammps_constrained`, `optimize_cell_meci`, `optimize_cell_meci_lammps`,
`optimize_cell_meci_multiscale`, `optimize_cell_mecp`, `optimize_cell_mecp_lammps`,
`optimize_cell_pressure`, `optimize_cell_pressure_aniso`, `optimize_cell_ts`, `optimize_cell_xtb`.
They are the seventeen `optimize_cell_*` YAMLs on disk; the eight fixed-cell `optimize_*` examples
are listed in [OPTIMIZE_DEVELOPMENT.md](OPTIMIZE_DEVELOPMENT.md) section 17, and twenty-five
`optimize_*` YAMLs are what the `optimize_test` family switch opens. One more example relaxes a
box, and the count above does not see it because it is named with the method it carries rather
than with the keyword: `multicenter_qmmm_cell.yaml` steps the six entries of a two-center QMMM
box against the composite's whole-system stress.

Four of the fixed-cell eight keep their own key at `false` in the shipped `pmd_env.yaml`
(`optimize_minima`, `optimize_gdiis`, `optimize_partial_ts`, `optimize_tda`), and they still run in
an optimize round: the family is consulted before the case, so `optimize_test: true` opens them -
which is what rounds 8 and 9 did, all twelve on two carriers each (the executed stems are listed
case by case in `CHANGELOG.md` #360), and round 10 ran thirteen with `optimize_cell_pressure` added
by #364. For the example framework and qualification policy, see
[EXAMPLE_DEVELOPMENT.md](../EXAMPLE_DEVELOPMENT.md) and
[QUALIFICATION_DEVELOPMENT.md](../QUALIFICATION_DEVELOPMENT.md).

## 10. The convergence record

A variable-cell run adds `Cell Gradient RMS` and `Cell Gradient Max` (judged against
`opt_csrms`/`opt_csmax` in bar) and `Cell Displacement RMS`/`Cell Displacement Max` (against
`opt_cerms`/`opt_cemax` as a strain), beside `Cell Volume`. The atomic and the cell half are
reported separately and each must clear its threshold, because a box still under stress has not
converged however quiet the atoms are. These rows are produced by the analytic instrument of
`test_optimizer_crossing.py` and by the seventeen examples of section 9. Reading them off a real
product is the job of `qualification/runner.py::_check_cell_optimization`, which fails a run whose
status record has no cell columns or whose volume never changed, and `_check_cell_shape` reads the
coupling from the trajectory's own lattice rows (section 7.3).

A leg that declares `opt_pressure` is asked two things more (`_check_cell_pressure`, #371): that the
run's own progress block reports the `External Pressure` the input asked for - the cell rows alone
cannot tell a pressure-driven stop from a zero-pressure one, because both are the residual of
`sigma + p I` - and that the box moved toward compression, under a positive pressure. Under a
nonzero pressure the objective those two facts are read in is `E + pV` as well (#381), so the
`Energy:` row of the run log and the `energy` column of `opt_status.txt` are an enthalpy and not any
engine's reading; `tools/verify.py` counts such a leg as an objective and leaves that column
uncompared, exactly as it leaves a composite leg's.

Those are the two facts the shipped example's header claims for itself, and until this clause
nothing in the suite checked them. The first three rounds under the clause read as one story: round
`tmp_20261017_180206` failed at 10 kbar with a 5937.0 bar residual, the target was lowered to 1000
bar, round `tmp_20261017_210205` still failed at 2803.2 bar - and both of those readings turned out
to be taken on a poisoned surface, a carried GFN-FF topology scoring sane geometries ~1.13 Hartree
high (#378). With that fixed, round `tmp_20261018_082139` compressed the box as asked (-6.976987
A^3, the direction ask passing for the first time), its worst step fell from +1.131534 to
+6.4948e-04 Hartree, and its endpoint residual is 969.6 bar. Per step the residual falls 1276.0 ->
171.9 bar by step 8 and then climbs again (196.6, 246.7, 505.4, 1164.0), so it never enters 40/60 at
any point in the run - and `maxit` is back to the example budget of 5 (#380), because the missing
thing was never steps. There are two outcomes here, converged and not converged, and this leg
reports the second. What is open is the fixture, not the machinery: a stiff 10 A ethylene cube has
no 1 kbar rest point reachable in an example-sized budget, while the compliant 1575-molecule LAMMPS
water box moves 53 A^3 and converges, so a pressure arm that is meant to pass belongs on that
carrier. The clause earned its keep on all three rounds - each time it named the residual and the
direction rather than printing a bare "did not converge".

What no run carries yet is the two quantum cases' cost: neither xtb nor cp2k is installed on the
development host, so its `maxit` was a ceiling, not a measurement, until the round that first ran
them. That reading is in: the xtb case took seconds over four steps (volume 0.131569 A^3, first step
`cell_gmax` 8.6753e-03), and the cp2k case's per-step cost and the budget that followed are section
8.5's record. The agreed response to a cost reading is to lower the ceiling or close the example,
never to widen a threshold (round tmp_20261016_075328, CHANGELOG #359). Round
`tmp_20261015_211219` read the two LAMMPS cases off real products for the first time (the flat box
moved 53.076419 A^3 over 12 steps, the held one 51.087745 over 13, and the YAML and classic carriers
reported the same numbers) and refused the two quantum ones, for reasons that belong to this
document.

A state manifold is declared where the system can read it (`molecule`'s `ci` and `spin`), not in the
optimizer section: `optimize` selects which state the run drives, while how many states exist is
what the engine is handed, and the round said `tddft_nstates=0` when the count sat in the wrong
section. A periodic box needs a method that writes one; for xtb that is GFN-FF, since xtb 6.7.1
stops a GFN2 SCF at `scf: Multipoles not available with PBC` and answers a GFN1 gradient run with a
lattice-gradient record of exact zeros (section 3.2 of
[XTB_SKILL_DEVELOPMENT.md](../skills/XTB_SKILL_DEVELOPMENT.md), #356).

## 11. Registration and verification

**Registration**: five examples joined the qualified list and the `pmd_env.yaml` switches
(`optimize_cell_gdiis`, `optimize_cell_meci`, `optimize_cell_mecp`, `optimize_cell_aniso`,
`optimize_cell_gap_cp2k`), and two skill cases joined the LAMMPS case list. The example count pins
moved in step (the manager's qualified list, the family-switch expectation, the stem-count comment),
`setup.py` needed nothing (the `examples/*.yaml` and `tests/*.yaml` globs already cover them), and
the plan-shaped numbers in the qualification documents were re-measured only **after** the examples
were added. `test_cell_keyword_parity.py` walks the new examples for real: each one's `ci` covers
the driven state and `opt_cell_coupling`/`opt_pressure` read back through both carriers.

**Local gate**: the fixture's clash check; the two-carrier chain (parse -> convert -> parse back,
reading the coupling, pressure, four cell thresholds and `ci/spin/nstate`); a PySCF single point for
the centre's degeneracy; the unit criteria named in section 8; and `tools/verify.py --selftest` with
the two CP2K arms.

**Remote round**: the engine legs of items 1/2/3/5, the LAMMPS cases on three carriers, the real
header of item 7 and step 8's rms/max of item 8, then a full `tools/verify.py` pass over the
downloaded artifacts. Per-cell verdicts: columns present and volume moved (1); the gap source label
and a finite non-zero cell gap (2); the row-length ratio changed (3); `fin_cell` and the iso/aniso
syntax (4); `state-resolved stress` inside the budget (5); section C2 matched for the CP2K leg (6);
the term columns present and summing to the energy (7); `Converged: True` (8).

## 12. Risks

The silent failures this project has been bitten by, each mapped to a guard: one place in the
classifier left unprefixed (a layered cell table read as foreign, and both the runner and the
verifier silently stop judging); a term count that disagrees with the header (every row read as
malformed, which presents as "zero steps"); Bohr/Angstrom mixing (the two conversion factors are not
exact reciprocals, so only increments are converted); wrapped versus unwrapped coordinates and the
`apply_pbc` ledger when a step is refused; the `task['config']['skill_config']` versus
`task['skill_config']` level confusion; a degenerate centre under TDA; a `highlevel` index list that
does not line up with `ci/spin`; three count pins drifting apart; a reader that steals another
engine's log by sort order; an edit anchor that swallows a neighbouring heading; and "a number
written in a comment before the example existed".

## 13. The progress ledger

**#381-#386 (the eight items).** The enthalpy objective (a new in-house instrument: converging in 4
steps and 0.4 bar with the term, 40 steps and 8795 bar without it, bit-identical at `p = 0`; three
mutation arms); the ethylene+water box and four new examples (the centre's spectrum measured at
0.0016 eV against 0.9860 eV for the water centre; PySCF has no NAC producer, written into the
header); `[COUPLING]` with its floor set from round 14's real trajectory (anisotropic minimum
9.7e-05, isotropic exactly 0); the per-term columns (four places, three arms); the CP2K reader, the
two LAMMPS `box_relax` cases, the CP2K gap example and the 2400 s budget; registration and the plan
shape re-measured at 437 / 197 / 240 / printed 243. The pressure example's 100/100 bar pair was set
by ruling.

**#387 (a framework defect the 15th round hit).** The term columns' state selection was registered by
the producer (a local state index for C > 1, `None` for C = 1, following the driven state) and the
consumer reuses `_select_state_energy`, refusing an out-of-range index by name; seven standing
criteria and three arms. The logger's row list was also missing `Cell Gap Term`, so that number never
reached the log.

**#388.** The `_example_timeout` 2400 entry got its first read-back criterion (positive parse plus a
fallback when the key is renamed), and `optimize_cell_pressure_aniso.yaml` put the 1 kbar question on
a box with all six components free, so the pressure residual's off-diagonal part was judged for the
first time.

**#389 (round 16).** 243 rows / 241 pass / 2 fail; the six assembly legs green on the #387 fix;
`verify.py` clean; `sum(term columns) == energy` exact over 64 rows of both carriers. The same round
falsified #381's causal claim about the pressure legs: they read identically to round 15 except for
the threshold (residual 987.45141829 bar, volume +0.074059 A^3).

**#391 (round 17).** 245 rows / 238 pass / 4 fail (the two pressure examples on two carriers);
`verify.py` clean. The same box expanded again (isotropic 1157.625 -> 1173.837737 A^3, residual
11518.1 -> 653.4 bar; anisotropic drifted 15.730721 A^3, residual 8286.1 -> 1669.8/2370.3 bar), while
its zero-pressure twin drifted 17.087668 A^3 on its own and its last step's |sigma| was 1688.1 bar in
compression - so the verdict was narrowed to "the residual must fall and the twin comparison must
hold", and the trust radius, not the budget, is what decides the band.

**#392.** The zero-pressure twin example; the engine-consistency pairing rule; a 1e-6 relative merge
tolerance; the 1581-atom mixed carrier in the package; the two LAMMPS-environment crossing examples;
the xtb/CP2K Hessian design sections and the `MATRIX_CAPABILITIES` refusal; the deploy poll's
four-way outcome and two clocks; and `verify.py`'s `shape_ratio_change` second source and PySCF
reader. The plan shape was 445 / 197 / 248 / printed 251, gate 3315 passed / 1 skipped / 0 failed.

**#393-#395 (rounds 18 and 19).** Round 18: 251 rows / 244 pass / 4 fail, `verify.py` clean over the
334 MB download. Round 19: 251 rows / 248 pass / 0 fail / Skip 197, clean over 332 MB. The converter
was changed to keep `skill_sections` by theory **entry** count, so `theory: [pyscf, pyscf]` reads back
binding for binding; and the three dispatch keywords were written into the generated classic input
(92 of 96 inputs byte-identical, 4 changed), measured as `optimize_cell_cp2k` at 589 s (yaml) against
815 s (`.input`) and `md_parallel`'s node going from `True/3` to `False/1`.

**#396.** `opt_trust_radius` became a keyword through six places, defaulting to `None` because
`minima`/`gdiis` start at 0.3 while a TS search starts at 0.05; the NPT-versus-cell static
consistency criteria. Two self-refutations stay on record: a proposed "backfill the classic-side
skill config" was withdrawn when the criterion was pointed at the real construction point (both modes
already delivered it), and "the step size decides the band" was only half true.

**#398-#399 (round 20 and the close-out).** Round 20 collected: both CP2K carriers at 581 s (yaml) and
584 s (`.input`), with the engine's own timestamps showing the twelve strain points as six
simultaneous pairs; the whole-tree gate 4735 passed / 1 skipped / 0 failed / 2944 subtests (482.91 s).
The budget-stopped legs now carry `[NOT CONVERGED]` and a `not_converged` flag on the zero-exit path
too, and `verify.py` section C2 counts them as `budget_stopped` (32 on round 20). `_example_timeout`
was deleted by ruling, leaving `_timeout: 1200`, after the same two legs measured 735 s and 728 s.

## 14. Current state, and what is not covered

The cell half is shipped and read: the seventeen `optimize_cell_*` examples of section 9 across the
fixed-cell and variable-cell routes, the per-term columns in `opt_status.txt` reconciled by
`verify.py` section C3, the pressure verdict narrowed to the residual and the twin comparison, and
the shape verdict (`[COUPLING]`) read off the trajectory's own lattice rows. The gate table of
section 4 is what the package accepts, measured at the gate; the readings above are what the engines
answered.

Not covered, and recorded rather than owed silently:

- **One converged pressurized variable-cell run on the xtb 81-atom box** (the one whose two frames
  bracketed 181.527 bar) - this environment has no xtb, so it is a round attempt, not a local test.
- **A multi-centre (C > 1) variable-cell search**: the code supports it and no example ships, by
  ruling, until the multi-centre design lands.
- **A LAMMPS mixed topology** in the package beyond the 1581-atom carrier's `.data`.
- **A periodic engine-side Hessian**: among the three periodic programs none declares the
  capability, so `hess_driver: skill` on a box has no producer; the framework's own batch covers it.
- **A fourth constraint kind on the cell force**: bond, angle and torsion are priced; the next one
  is refused by name until it is.