# PyRAI3MD Path Family (MEP / IRC / NEB) Design

Last updated: 2026-10-22 17:10

## 1. What this document is, and what it is not

It is the design of the path family: the three searches that walk more than one
structure, asked for as `optimize.method: mep | irc | neb` under the `optimize`
jobtype. It is not a new jobtype vocabulary: there is no `jobtype: path` and no
`&path` section, and the input surface is the one every optimize run already
writes.

It is also not `dev/methods/OPTIMIZE_DEVELOPMENT.md`, which owns the
single-structure loop, the optimizers, the cell half and the convergence
criteria. This document owns what a path adds to them and refers to that one for
everything else.

## 2. Directory structure and ownership

| Path | What it owns |
|---|---|
| `workflow/path.py` | The `Path` workflow: routing, the two execution routes, the band's construction, and the four records a run leaves |
| `workflow/optimizer/pathbase.py` | `PathOptimizer`, `WalkerPathOptimizer` (the macro/micro state machine) and `BandPathOptimizer` (the images and the FIRE state) |
| `workflow/optimizer/pathconstraint.py` | The two constraint models: the mass-weighted hypersphere step and the ReLU-gated constraint potential |
| `workflow/optimizer/mep.py`, `irc.py`, `neb.py` | The three registered searches; `irc` also owns the mode picker and the seed |
| `utilities/geodesic.py` | The port of `geodesic-interpolate` (MIT, Copyright (c) 2024 Xiaolei Zhu), with the attribution and the six adaptations in its header |
| `utilities/io.py::read_xyz_frame` | The endpoint reader, including the `Lattice="..."` key |
| `qualification/verification/test_path.py` | The in-process instrument and the assertions of section 15 |

`workflow/constraints/` is untouched: that package holds the geometric conditions
a run declares (fixes, bonds, collective variables), while the two models here
decide where a *probe* goes inside one path point.

## 3. Routing, and the two execution routes

`resolve_workflow` reads `opt_method` and returns `Path` when it is one of the
three names, so one jobtype reaches two loops; the planner's node tables are
keyed on the workflow's own name (`path`), which reads the same `&optimize`
section. `Optimize` refuses a path method by name (`is_path`) if one reaches it,
and `resolve_workflow` refuses a path under `driver != 'main'`: a path is a
sequence of geometries the framework commits, and an engine that owns every step
has nobody to commit them to.

Two routes, because the two kinds of walk evaluate differently:

- **walker (mep, irc)**: one probe at a time, in place on the calculator's own
  state, in the order `Optimize.run` uses (write the probe, wrap it into its
  cell, land a declared condition, sync, evaluate, read back). The probe's
  gradient decides the next probe in the same frame, and the sequence is serial
  by construction; a dispatched child per probe would deep-copy the parent state
  and lose the walker's history between probes.
- **band (neb)**: one round is every image, fanned out with
  `dispatch(units=batch)` exactly as the Hessian workflow fans out its displaced
  geometries, so images run in parallel (`multipool`) and each gets its own
  runtime directory. `RuntimeDirectoryManager` reuses existing indices, so the
  same N-image band maps onto the same N directories every round.

## 4. The path optimizer contract

`PathOptimizer` extends `Optimizer` for its machinery (thresholds, the joint
vector's cell half, `apply_constraint`, `note_evaluation`, `usable_curvature_product`)
and refuses `run(state)`/`_compute_step` by name. The interface the workflow
drives:

- walker: `begin(state)`, `next_trial()`, `apply_trial(state)`, `observe(state)`,
  `finished()`, `report()`, `path_points()`, `results()`
- band: `begin(state)`, `set_band(...)`, `band_batch(state)`,
  `band_collect(state, results)`, plus the same reporting four

`WalkerPathOptimizer` is a state machine over `SEED -> PROBE -> ACCEPT ->
PROBE -> ... -> DONE`. One path point is one constrained minimization on the
sphere of `path_radius` centred on the last accepted point; the sphere is
re-centred on every accepted point, and the walk ends at a stationary point, on
an uphill step (the minimum is the point before it), or at its step bound - the
last of which reports the framework's own `Maximum optimization steps reached`
phrase, so a truncated path is reported as `PASS[NOT CONVERGED]` the way a
truncated relaxation is.

## 5. Coordinates

Mass weights are `sqrt(m_i / M)` per atomic component, normalised so that
`path_radius` is a plain length in Bohr (the unit `opt_trust_radius` and the
lattice half of a joint step are written in, so no amu factor enters the input),
and 1 per lattice entry, because a box has no mass. The joint vector is
`(3N + 6)` for a variable-cell run, in the same spelling
`optimizer/base.py::_extend_to_cell` builds.

A periodic walk keeps the **continuous** frame: `System.image` is a
crossing ledger, the input pipeline can already populate it before the
first step (a molecule written across a face is stored folded), and
`_adopt` reads every evaluation back through it. The seed is therefore
built from `continuous_coordinates` rather than from the stored main image
(`pathbase.py::_continuous_atoms`); building it from the stored
coordinates put the reference one box from the geometry the first
evaluation adopted, which the sphere model absorbed but the `relu` model
read as a penalty offset of that size. `apply_trial` still writes a
*relative* step onto the stored geometry, so the backend keeps receiving
the main image.

## 6. MEP, and the two constraint models

Both models answer the same question - where is the next probe on the sphere -
so the walker holds one loop and knows nothing about which is inside it.

**sphere** (Gonzalez & Schlegel, J. Phys. Chem. 94, 5523 (1990)): the point is
held exactly on the sphere and rotated towards `-G_perp`;
`X' = rho (cos(phi) u + sin(phi) d)` has length `rho` for any `phi`, which is
what makes the arc length of a path point a property of the run. Measured on the
instrument of section 15: every shell point at exactly 1.000 of the radius, two
evaluations per point, four points for a walk that arrives at the minimum.

The rotation *shrinks* whenever the probe's energy failed to fall, down to a
floor. Measured: a fixed 0.25 rad bounced between two energies
(`+0.0051007335` then `-0.0051007335`, repeatedly) and spent every probe without
settling; letting it grow again after a fall kept the bounce alive at a finite
amplitude. Shrinking only turns the movement into a geometric decay, which is
what lets the tangential gradient fall below the tolerance.

**relu**: the minimisation of `E + phi`, with

    phi(rho) = ((rho - rho_t) / rho_t)**2   for rho > rho_t, else 0

one-sided by construction (the ReLU gate), the penalty gradient of a Morse-scaled
shell, and a BFGS step capped at the arc length and at the distance to the shell
when the probe is outside it. Measured: the shell points sit at 1.004 of the
radius (the balance between the energy's outward slope and the penalty's
restoring force, a few parts in ten thousand outside the shell), five
evaluations per point, and the same minimum as the `sphere` arm.

Two deliberate departures from the OpenQP implementation this is reimplemented
from (`pyoqp/oqp/library/libscipy.py::ConstrainOpt`, OpenQP Research License: no
code copied), both measured:

- the chain rule keeps its weight factor; the reference's `df` expression omits
  one `w`, which is only correct for a uniform mass;
- the penalty gradient is **not** projected orthogonal to the energy gradient.
  That projection removes the radial force the shell is held by whenever the
  displacement is collinear with the gradient (the first step of every path
  point); measured, the walk then drifted outward and settled at `rho = 3.05 s`
  instead of holding 1.004.

A walk from a stationary point has no direction and stops naming that fact;
`irc` is the method that supplies one.

## 7. IRC

The MEP walk seeded one `path_radius` along the imaginary mode of a saddle.
`irc_mode: hessian` asks the framework's Hessian computation (honouring
`&hessian hess_driver`/`hess_mode`) with its analysis **off**, and reads the
modes from the atomic block here: a variable-cell run refuses a vibration
analysis outright (*"a box's curvature is read by the search that asked for it,
not as a frequency"*, `workflow/hessian.py`), and the imaginary mode of a
reaction path is an atomic mode anyway. The whole matrix is handed to the `relu`
model as its starting curvature, so one computation serves two purposes.

Two things that handoff needs, both found by running a variable-cell IRC and a
`relu` IRC on this checkout rather than by reading the code:

- **The seed is atomic.** A variable-cell run walks a joint `(3N+6)` vector, and
  the imaginary mode is `3N`: `IRCOptimizer.seed` displaces the atoms on the
  **atomic** weights (which `WalkerPathOptimizer.begin` now keeps beside the
  joint ones) and leaves the six lattice components where the saddle had them,
  so the first point of the walk is one `path_radius` away in the coordinates
  the mode lives in and the box relaxes during the walk. The seed used to hand
  the joint weights to the `3N` mode - a numpy broadcast error on every
  variable-cell IRC - and now refuses a mismatched pair by name.
- **The `relu` model's starting curvature has to be positive definite.** Its
  BFGS fit minimizes `E + phi`, so `ReLUConstraint._inverse_of` uses the
  handed matrix only when it is positive definite; a saddle's is not, and is
  replaced by the identity. Measured on `path_irc_relu`'s surface (HF/STO-3G
  HCN): with `pinv(H)` the residual of every point plateaued at 0.15 - the shell
  radius - and the walk accepted one point in forty evaluations; with `|H|` it
  accepted six, still at 0.15; from the identity it walks six points down the
  basin and settles the last one at 0.0025. The atomic-block truncation of a
  `(3N+6)` matrix stays by design and is pinned to be exact.

`irc_mode: molden` reads a vibration file instead. The reader returns
frequencies **signed** with a negative value for an imaginary mode; until this
round `read_molden` took the absolute value while `write_freq_molden` wrote the
sign, so the two halves of one file disagreed about the same mode and no file
could seed a walk. The only consumer that wants a magnitude
(`sample_initial_conditions`, which has no thermal distribution for an imaginary
mode) now takes it itself and warns. Measured on the shipped fixture: the same
saddle's own Hessian, one imaginary mode at -1184.09 cm-1, and the two routes
(the file and the Hessian) seed the same displacement. The two periodic arms
(`path_irc_pbc`, `path_cell_irc`) read this file: their walk is on a force field
that has no Hessian of its own, and the mode is a direction, so the file is what
keeps them about the periodic and cell machinery rather than about a curvature
batch.

`irc_direction` picks the side: `both` runs one walker per side from the one
saddle, each with its own geometry record, and the shared result keys carry the
forward branch while each side also has its own block. The two are *labels for
the two ends*: a mode read from a file has an arbitrary sign.

Both branches start from the **one saddle**. A walker is driven on the
calculator's state in place, so the second branch would otherwise inherit the
first one's last geometry - measured on `path_irc` before the reset, the two
`saddle` rows read -91.5616686633 and -91.6662865772, and on `path_cell_irc`
the forward branch was an empty walk whose single row carried the backward
branch's final box. `Path._walker_origin` snapshots the coordinates, the box
and the crossing ledger before the first walk and `_restore_walker_origin`
puts them back before each subsequent one. Pinned in process by
`test_path.py::TestTheIrcBranchOrigin` and after the run by
`tools/verify.py` section F, which requires the two saddle rows to agree.

A mode read from a file can belong to another surface (a `molden` from a
previous calculation, for example). Then only one side of it may descend on
the surface actually walked: the shipped `path_irc_pbc` is that case, and its
header records the measurement (`irc_mode: hessian` refuses the geometry as a
gfn_ff minimum, forward climbs, backward descends).

## 8. NEB

The literature's band force, in three parts:

- **tangent**: Henkelman & Jonsson, J. Chem. Phys. 113, 9978 (2000). The
  energy-weighted average of the two segment directions, with the geometric
  secant as the fallback when the energies cannot order the three images.
- **force**: a parallel spring `k (|R_{i+1} - R_i| - |R_i - R_{i-1}|) tau` plus
  the physical force perpendicular to the tangent only, so the band does not
  slide downhill off the path; the endpoints carry nothing.
- **climb**: Henkelman, Uberuaga & Jonsson, J. Chem. Phys. 113, 9901 (2000).
  With `neb_climb`, the first round whose perpendicular force max falls below
  `neb_climb_threshold` inverts the highest image's parallel force and drops its
  spring; the latch is one-way, because a band handed back to the springs would
  leave the saddle it just found.

FIRE (Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006)) relaxes the band, which
is the right integrator for a field that is not the gradient of anything. A
variable-cell band puts the six relaxed lattice entries in the same vector, with
the stress conjugate as their force and the enthalpy as the objective under
`opt_pressure`; the two halves are judged by the two threshold pairs
`optimize_cell` already names.

Measured on the instrument of section 15: the settled band's largest gradient
perpendicular to its own tangent is 1.7e-3 (the threshold), its segments
0.398-0.417 apart, its endpoints untouched; the climbing image ends at the
saddle with its own force at 1.7e-3, and the plain band's highest image stays
slightly off it.

## 8a. Declared conditions on a path

A run that declares a strict condition (`shake_bond` / `shake_angle` /
`shake_dihedral`) or a freeze (`fix_atoms`) holds it on every path point, with
**one implementation**: `constraints/runtime.constrain_state` for the position
solve and `constraint_split` for the tangential force. What differs between the
two routes is where those two calls happen.

- **walker**: the workflow writes the probe, wraps it into its cell and lands the
  condition (`constrain_state`), then the optimizer reads the geometry the
  backend actually evaluated back out of the state and unwraps it into its own
  continuous frame (`pathbase::WalkerPathOptimizer._adopt`). Without that
  adoption a walk on a periodic system would put its sphere a whole box length
  from the structure the force belongs to, and a walk with a held bond would keep
  bookkeeping on the pre-solve geometry. Measured on `path_mep_bond`: the held
  C-N is constant to 5.3e-11 A over every written frame.
- **band**: an interpolated image satisfies no condition on its own, so the band
  lands every **interior** image on the manifold before it dispatches a round and
  again after every FIRE step - the frames a round prints are then the geometries
  the next round evaluates - and reads each image's force through the same split.
  The two endpoints are left exactly where the run was given them: they are the
  band's boundary, and a condition they violate is the input's own inconsistency,
  visible as a nonzero residual on the profile's first or last row. Measured on
  `path_neb_bond`, whose two endpoint files carry the same C-N length: every one
  of the band's written frames holds it to 2.2e-16 A.

A variable-cell path carries the condition's own lattice term as well:
`constraint_split` returns `cell_virial` from the geometry and the plan, the
walker subtracts it in `_cell_gradient` and the band does the same per image, so
neither steps the box on a force the engine's stress cannot see. The cost is
stated rather than hidden: a constrained band runs one position solve and one
split **per interior image per round**, in the parent.

---

## 9. Interpolation, and the geodesic port

`linear` interpolates in Cartesian coordinates, or in fractional coordinates at
the cell for a periodic band. Both schemes share one rule: **the two endpoints
are brought into one periodic image before anything is interpolated**
(`geodesic.consistent_pair`, the metric's own nearest-image convention). A
structure file lists the main image, so a molecule written across a cell face
has some atoms on the far side; the product of one reaction can then differ
from the reactant by a whole lattice translation even though they are the same
physical pair. Interpolating the raw arrays drags those atoms through the wall
- measured on a 6 A cell with an O-H-H straddling it: the linear chain's last
step was 5.90 A, and it is 0.4 A once the product is moved to the reactant's
image. The moved product is the same crystal, and the record's `Lattice=`
comment is what makes the frame readable; the geodesic's metric already read
nearest-image distances, but its initial guess and the linear arm did not.
Pinned by `test_path.py::TestThePeriodicEndpointFrame` (a fixture that
straddles the face, both schemes, and the raw step as the reading).

`geodesic` places the interior images by minimising the path length in a
Morse-scaled redundant-internal-coordinate metric. The port's header lists the
six adaptations: the atoms arrive from the run and their radii come from
`schema.reference.COVALENT_RADII` (Cordero single-bond, H-Cm; 1.5 A beyond);
units stay Angstrom; pairs are
read in the nearest periodic image at each image's own cell (with a direct
wrapped-distance screen, because a minimum image is not a metric a K-D tree can
index); the metric's three parameters are the caller's (and the reference's
stiffness argument is now actually used - it computes `morse_scaler(alpha=1.7)`
and ignores its own `scaler`); every draw goes through a
`numpy.random.Generator` seeded from `control.gl_seed`; and the solver is this
framework's own - the same residual, the same analytic Jacobian and the same
`soft_l1` robustness as iteratively reweighted least squares, solved densely
with a Levenberg damping instead of through scipy, which this package does not
depend on.

Two further deviations, both because this port builds a *band*: the reference's
chain alignment is kept for the interior images but neither endpoint is moved,
and it is skipped for a periodic band (a rigid rotation inside a fixed cell is a
different structure unless the cell is rotationally symmetric). The chain is
aligned *before* the smoothing and not after: this metric measures a segment
between the Cartesian midpoint of two adjacent images, so placement is part of
the length rather than a gauge choice - measured, re-aligning the smoothed path
took the internal length from 2.3920 back up to 9.2144.

What it buys, measured on the shipped HCN -> HNC pair (PySCF HF/STO-3G): the
geodesic arm's interior images all sit between -92.79 and -92.88 Ha, while a
linear interpolation's middle image sits at -79.4 Ha, because the straight line
takes the migrating hydrogen through the C-N region. The barrier a band then
relaxes is a property of its initial guess, not of the reaction.

## 10. Endpoint files

`optimize.neb_reactant`/`neb_product`/`neb_ts` name XYZ files. The atoms and
their order are checked against the system's (a path interpolated between two
different atom orders is not a path), and a variable-cell band needs both
endpoints to declare a cell in the extended-XYZ `Lattice="..."` comment line the
framework's own writer emits.

## 11. Control keywords

All in the `&optimize` section, bare spellings that take the `opt_` prefix on
their way into the schema, and each of them registered in four places
(`DEFAULT`, the `Optimize` model, `adapter.OPTIMIZE_TYPES`, `core.optimize_params`):

| Key | Default | The one question it answers |
|---|---|---|
| `path_constraint` | `sphere` | which constraint algorithm holds the step (mep/irc) |
| `path_radius` | 0.1 | how far one path point advances (Bohr) |
| `path_inner_maxit` | 20 | how many probes one point may spend |
| `path_inner_tol` | the run's `opt_grms` | the tangential gradient that ends a point |
| `irc_direction` | `both` | which side(s) of the saddle the walk takes |
| `irc_mode` | `hessian` | where the imaginary mode comes from |
| `irc_mode_file` | - | which file holds it |
| `neb_reactant`, `neb_product`, `neb_ts` | - | the band's endpoints and optional guess |
| `neb_images` | 8 | how many interior images |
| `neb_interpolate` | `linear` | how they are placed |
| `neb_geodesic_threshold` / `_scaler` / `_min_neighbors` | 3.0 / 1.7 / 4 | the metric's cutoff, stiffness and coordination |
| `neb_spring` | 0.01 | the spring constant (Hartree/Bohr^2) |
| `neb_climb`, `neb_climb_threshold` | false / 1e-2 | whether the highest image climbs, and the force that arms it |
| `neb_step` | 0.1 | FIRE's initial timestep (atomic units) |

`path_inner_tol` falling back to `opt_grms` is a rule rather than a default,
and it was measured into being: the shipped HCN walk loosens `opt_grms` to 3e-3
and left the inner tolerance at its 3e-4 default, ten times tighter, so every
point spent its whole probe budget hovering around a minimum it was never
allowed to call found.

The section validator refuses each group on a method that cannot read it (band
keys on a walker, hypersphere keys on a band, IRC keys on neither), a band
without both endpoints, `geodesic` with fewer than two interior images, and a
climb threshold without the climb switch.

## 12. Convergence and termination

A walker point ends when its model's own reading crosses `path_inner_tol`. A
point whose probe budget is spent is kept and the walk *carries on* (the
reference's micro-iteration takes the same branch), with that point's `settled`
flag false in the profile. A walk ends at a stationary point, at an uphill
point, or at its step bound. A band ends when the interior forces are below
`opt_grms`/`opt_gmax` - the climbing image judged on its own force - and, for a
variable-cell band, when the cell half is below the two `optimize_cell`
threshold pairs as well.

## 13. Outputs

`path.xyz` (the latest frame), `path_geom.xyz` (every accepted frame, or the
band once per round), `path_profile.txt` (one row per accepted point or per
image: index, direction, energy, change, arc, segment length, residual, radius,
role), `path_status.txt` (one row per evaluation or per round), and
`path_forward.xyz`/`path_backward.xyz` for the two sides of an `irc_direction: both`.
Every periodic frame carries its own box in its comment line (`Lattice="..."`),
and for a band that is **the image's** box, not the parent state's: a
variable-cell band steps each image's six lattice entries, so one box for the
whole record would disagree with the table's own volume column (that was the
defect section 15a records).
None of them is written through the framework's `opt_status.txt` table
(`utilities/opt_formatter.py` writes it, `utilities/print_results.py` reads it):
that reader classifies a record by its column count, so a path table written there would be
read as another writer's. The variable-cell columns reuse that module's cell
names (`cell_grms`, `cell_gmax`, `cell_drms`, `cell_dmax`, `volume`), so one
reading is spelled one way in both tables. What `state` ends on is the primary
path's last accepted point - a path has no single optimized geometry, and the
result block (`path_*`) is what carries the rest.

## 14. Systems, and what is refused

Isolated molecules, single-centre multiscale composites, and periodic systems
(fixed cell, or a variable cell under `opt_pressure`). Refused by name: a path
under `driver != 'main'`, a state pair (`opt_state_j`), an excited-state IRC with
a backend-produced Hessian, `neb_interpolate: geodesic` above 200 atoms (the
dense normal-equation system), and - for a periodic band - an endpoint file
without a cell comment.

**A multi-centre partition (C > 1) is out of scope and is *not* refused by
name**: the design reason is real (a path interpolates between structures, and a
multi-centre partition has no single structure to interpolate), but no guard was
written for it in this round, so an input that declares one is an unverified
combination rather than a refused one. Do not read this section as a promise
that it is turned away; the single-structure searches refuse a multi-centre
saddle by name (`data/core.py::_build_task_queue`) and a path has no equivalent
yet. Recorded in `dev/TODO.md`.

## 15. Verification and examples

`qualification/verification/test_path.py` carries the in-process instrument: a
double well on the normalised mass-weighted coordinates whose constrained
minima and saddle are known in closed form, plus a curved two-coordinate surface
for the band. It pins: both models' monotone descent and their radius (1.000 and
1.004 of the shell), the walk's evaluation bound, the band's perpendicular
gradient at the settled band and its climber at the saddle, the geodesic's
shorter internal length and its reproducibility for a seed, the periodic
interpolation, the mode picker and the seed's radius, the molden reader's sign,
the endpoint reader's three refusals, the routing and the keyword holders, the
`relu` model's positive-definite start, a variable-cell IRC seed that leaves the
box alone, the exact atomic-block truncation of a `(3N+6)` matrix, and a band
whose frozen atom keeps its anchor on every interior image.
32 tests, all passing on this checkout.

The thirty-two shipped `path_*` examples are the engine-side arms, and the
system matrix they cover is: isolated molecule (mep / irc / neb), single-centre
multiscale (`path_*_multiscale`, PySCF twice over one centre), periodic fixed
cell (`path_mep_pbc` / `path_neb_pbc` on the shipped 60-atom force-field box,
`path_neb_xtb` and `path_neb_pbc_geodesic` on xtb, `path_irc_pbc`), variable
cell (`path_cell_mep`, `path_cell_mep_relu`, `path_cell_irc`, `path_cell_cineb`,
`path_cell_neb_geodesic`), the declared conditions (`path_*_bond`,
`path_mep_angle`, `path_mep_torsion`, `path_neb_angle`, `path_neb_torsion`), the
frozen ride (`path_mep_freeze`, `path_neb_freeze`), and the two constraint models
on both loops (`path_mep_relu`, `path_irc_relu`). The first twenty-one ran in
both carriers in the test environment host in round `tmp_20261021_195813`: 42 example
rows, 42 pass, in a round of 269 rows (266 pass, 0 fail, 213 closed) under
`examples.path_test: true` with the optimize family closed. The ten added after
that round - the combinations the coverage audit found implemented with no
engine example - ran in round `tmp_20261022_084519`: **62 example rows, 62
pass**, in a round of 289 rows (286 pass, 0 fail, 213 closed). Their readings:
`path_cell_mep_relu` 8 points with the volume 1000.00 -> 1011.94 A^3 (the
penalty model on a moving box, first engine reading), `path_cell_irc` 4 points
and 1000.00 -> 996.05 A^3, `path_neb_pbc_geodesic` 5 images with a 0.019 Ha
barrier, `path_cell_neb_geodesic` 5 images and 1000.000 -> 1000.128 A^3 under
the anisotropic coupling, `path_neb_angle` / `path_neb_torsion` 5 images each
with their held value constant, the two frozen cases 0 deviation, and
`path_irc_relu` 6 points down the basin.
Round `tmp_20261022_103032` re-ran the same plan after the two defects the
first review found (section 15a), with the same totals and `tools/verify.py`
clean. Their per-case switches are
in `examples:` of `pmd_env.yaml` and the group switch is `path_test`, with
`manager.PATH_EXAMPLES` and `QUALIFIED_EXAMPLES` carrying the thirty-one names
(an example missing from the latter is postponed whatever its switches say, and
a shipped example missing an `examples:` key is now refused by
`test_manager.py::TestShippedOptimizeRoundStage::test_every_shipped_example_has_a_switch`
- an absent key defaults to `true`).
`manager`'s family key is pinned against that list and against the disk by
`qualification/verification/test_manager.py::TestPathFamilySwitch`, which also
pins the other half of the ruling: the two prefixes are disjoint, so an
optimize round does not run the path rows and a path round does not run the
`optimize_*` ones. `test_dispatch.py` pins that the `path` node reads the
`&optimize` section (`WORKFLOW_NODE_DECLARATIONS['path']` forces no flag, so
`&optimize multipool` reaches it), with a misnamed-node negative control.

> A single-centre composite with a variable cell **is** shipped:
> `path_cell_mep_multiscale` (PySCF centre + xtb/GFN-FF environment). An
> earlier draft of this document called the combination infeasible, and the
> reasoning behind that was too narrow: it tested PySCF twice over one centre,
> whose layers declare no `stress`, and stopped there. The classified
> environment is xtb, CP2K or LAMMPS, and the composite's box is stepped on the
> stress of the layer that carries the cell (`Optimizer._cell_stress` reads
> `state.active_stress`, the structural-track provider's whole-system tensor) -
> so a force-field environment is exactly the producer the cell half needs. The
> centre stays PySCF and has no stress of its own, which is why `Cell Gap Term`
> is zero there by the shape of the plan. Measured in the test environment host: 3
> points, -77.3231 -> -77.6418 Hartree, volume 1000.000 -> 1000.155 A^3.

## 15a. What the round was read against afterwards

A round's verdict says whether each example ran and stopped as declared; it
cannot say whether the physics the example claims came out. Four claims the path
family makes are recomputable from the files a run leaves, and they are
recomputed by `tools/verify.py` sections D and F (the same single-file
post-round tool that re-derives the other categories), over round
`tmp_20261021_195813`:

- A leg that declares a strict condition held it on every frame it committed.
  `path_geom.xyz` carries every accepted point (a walker) or every image of every
  round (a band), so the held internal coordinate is recomputed frame by frame
  and compared against the tolerance that leg's own flat config declares. Ten
  condition legs, 68 frames: the worst deviation was 1.001e-10 Bohr
  (`path_mep_bond`'s held C-N; the declared solve tolerance is 1e-10 Bohr, so the
  rounded reading and not the solve is the floor), 9.3e-11 Bohr for
  `path_irc_bond`, 4.4e-16 Bohr for the band `path_neb_bond`, 5.5e-11 radian for
  the held H-O-H angle and 9.0e-11 radian for the held H-O-O-H dihedral. The
  limit is ten times the declared tolerance: a hold that was not held moves by
  the walk's own arc, 0.15 Angstrom, 1.5e8 times that limit. This one reading
  covers the walker's held bond, the band's landing and the two angular
  conditions.
- A leg that freezes atoms (`fix_atoms`) kept them at the anchor the run bound.
  The atoms come from the leg's config (`constraints.fixes[*].atoms`), and the
  check runs over the walkable frames: every frame of a walker, the interior
  images of a band - the two endpoints are the band's boundary conditions and
  `_land_images` never lands them, so a frozen atom that differs there is the
  input's own choice. Measured on the two local runs the cases were written
  against: deviation exactly 0 over five frames (`path_mep_freeze`) and twenty
  (`path_neb_freeze`). The in-process arm is
  `test_path.py::TestTheFrozenBand`.
- A composite leg reproduces the surface its single-layer counterpart evaluates
  on the geometry both declare. The counterpart has to be the same system -
  the same atoms and the same starting geometry - not merely the name without
  `_multiscale`: `path_cell_mep_multiscale` shares only its suffix with
  `path_cell_mep` (a 12-atom composite at -77.32 Hartree against a 6-atom
  single layer at -0.63), and pairing them would compare two molecules. Six
  composite legs (three examples on two carriers) agree with their partners to the print of
  `path_profile.txt` (ten decimals of a Hartree) on every point the two share:
  the MEP pair over its whole profile (same start, same parameters), the IRC pair
  on the shared saddle, the NEB pair on both endpoints. A composite that
  assembles its layers wrongly shows up here and nowhere else in the round.
- A variable-cell path leg's verdict is re-read by section D from its own two
  records: the five cell columns by name (the table's header is not
  `opt_formatter.py`'s, so `utilities/print_results.py::read_status` must not be used), a volume that moved
  between the first and the last row, and the boxes matched the way each kind
  writes them - a walker's frames are the points it accepted, so every frame
  box must appear among the rows of its own direction (the rows include the
  probes it rejected, which is why `path_cell_irc`'s last row is not the last
  frame's box); a band writes its frames *after* the FIRE step while the row of
  round N is the geometry that round was **evaluated on** - the geometry round
  N-1 committed - so a row of step N >= 2 must match one of round N-1's frames.
  Measured on round `tmp_20261022_084519`: ten legs over 550 rows, all moved,
  56 frames/rounds matched, no disagreement.
- The two branches of an IRC `both` start from the one saddle, so their two
  `saddle` rows must carry the same energy. Measured on round
  `tmp_20261022_084519`: four legs read a pair that differed by 0.1 Hartree
  (`path_irc`: -91.5616686633 against -91.6662865772; `path_cell_irc`:
  -0.3453618416 against -0.6583014035), which is the defect the next paragraph
  names. Round `tmp_20261022_103032` certified the reset: both rows read
  -91.5616686633, `path_cell_irc`'s forward branch records its own saddle, and
  the clause reports `irc_origin_broken 0`.
- The two numeric models' own readings are pinned in process by
  `test_path.py` (section 15), so the post-round tool does not restate them.

The cross-check in the last clause found a defect that no verdict had: a
variable-cell **band** wrote the parent state's box into every frame, so
`path_cell_cineb`'s record read 1000.000 -> 1000.638 A^3 while every `Lattice=`
row of its trajectory read 1000.000. `Path._write_band_frames` now takes each
image's box from `BandPathOptimizer.lattice_of`, which is also where the record's
own volume column comes from; a fixed-cell band's comment is unchanged.

Found the same way, on the same round: the second branch of an IRC `both` was
walking from the first branch's endpoint. A walker is driven on the state in
place, and the two branches shared it; `Path._walker_origin` now snapshots the
coordinates, box and ledger before the first walk and
`_restore_walker_origin` puts them back before each later one. After the reset
`path_irc`'s two saddle rows read -91.5616686633 and the forward branch
commits six points where it committed none, while the backward branch's four
rows are bit-identical to the ones it produced before.

Section F's detector needed one fix of its own, found by running it on the round
rather than by reading it: the member atoms of a strict condition live under
`bonds` for every kind (`schema/config.py::ConstraintInput`), so a reader that
looked for kind-named keys answered empty for the two angle legs and audited six
of the ten legs while reporting no problem at all. The fixture in
`verify.py::selftest` now carries an angle and a dihedral leg, and reverting the
key to the kind-named form makes the selftest fail - so the control is
demonstrated, not assumed. The D and F clauses added afterwards carry their own
planted failures (a stationary path box, a volume no frame carries, dropped cell
columns; a frozen atom moved in an interior frame, and the same motion in a
`Role product` frame staying silent), and both were shown to bite by mutating the
detector.

Fixed - and it was not the crossing count but the frame: a periodic walk built
its seed from the stored main image while `_adopt` returned the continuous
reading, and the input pipeline had already put five atoms of `path_mep_pbc`'s
60-atom box across a face. The two readings differ by a whole box, so the
start row reported `radius 4.948` and `max_d 8.399` (the box edge) and the
first accepted point sat 4.86 Bohr from the start. `pathbase.py::
_continuous_atoms` now builds the seed in the same frame (`_joint_of` reads
it, `_adopt` already did), the initialisation stays untouched (the backend
receives the main image), and the first evaluation is adopted back at the
trial to 1.8e-15 Bohr. Pinned by `test_path.py::
TestThePeriodicWalkerSeed`, with the stored-coordinate version as the
mutation control (seed 0.2 instead of 10.2).

The fix moved the measured reference, which exposed a second reading: on that
strained box a `path_radius` of 0.1 Bohr takes a first step that *raises* the
energy by 0.31 Hartree, so the walk spent its budget coming back and accepted
no point at all. It is now 0.05 in `path_mep_pbc.yaml` - four points, monotone
from -0.2816 to -0.3864 Hartree over 0.20 Bohr of arc, start row at radius 0 -
with 0.01 measured as monotone but covering a fifth of the arc. The old 0.1
was tuned against the one-box-off seed, so its shell was not this walk's.

## 16. Design principles

- One section, one vocabulary: a path is an `optimize.method`, so it reads the
  `&optimize` section and reuses `opt_grms`/`optimize_cell`/`opt_pressure` rather
  than inventing a second set of thresholds for the same numbers.
- The cell half is reused, not re-implemented: `_extend_to_cell`,
  `_cell_gradient`, `_enthalpy`, `_apply_step` and `_apply_constraint` are the
  single-structure search's, so a path cannot step on a different box force than
  a relaxation does.
- One column, one reading: the two tables carry different columns because a
  row about a probe and a row about a band are not the same row.
- Numbers in this document are measured, and the ones that are not say so.