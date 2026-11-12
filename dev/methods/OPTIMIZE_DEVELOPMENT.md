Last updated: 2026-10-24 08:45 - the output section now records the run-level `&optimization setup` block, the method-specific rows merged into `&optimization progress`, and the method columns appended to `opt_status.txt` (after the cell columns, before the composite terms). Earlier 2026-10-22 21:40 - the cell half moved out: the cell and periodic material this file
carried in sections 10, 15.1, 15.2, 17 and 18 now lives in
[CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md), and the cell material in those sections here is
now pointers. No
algorithm, threshold or reading changed in the move. This pass also corrected two stale claims it
found: the path family is implemented and routed to the Path workflow (sections 2 and 15), and the
constraint-to-cell force is priced for a bond, an angle and a torsion rather than a bond only
(section 14). The audit of 2026-10-19 16:10 remains the last pass over the optimizer's own text.
---

> **Multi-center capability boundary (refactor R2-R6, 2026-09-08).** The
> optimizer has no multi-center higher-derivative (Hessian) contract yet: it
> reads the global state axis (`state.nstate`), which raises `CenterPlanError`
> for C>1, so a C>1 optimize/Hessian combination is not supported this round;
> the already-supported single-state / C=1 path is unchanged
> (`dev/MULTICENTER_DEVELOPMENT.md` §11).

## 1. Design Philosophy

Geometry optimization is treated as an algorithm module independent of the workflow and calculator systems.

Responsibilities are clearly separated:

- **Workflow** controls the optimization loop.
- **Calculator** evaluates energy, gradients, and other properties.
- **Optimizer** determines the next geometry based on optimization algorithms.

The optimizer never directly performs electronic structure calculations and only operates on `PyRAIState`.

---

## 2. Directory Structure

```text
pyrai3md/workflow/
    optimize.py              # optimization framework (loop control)
    optimizer/
        __init__.py          # OPTIMIZER_REGISTRY, get_optimizer
        base.py              # OptimizerBase: state, history, trust radius,
                             #   convergence, state-resolved extraction
        minima.py            # BFGS (+ optional RFO) + resized trust radius
        gdiis.py             # GDIIS extrapolation + BFGS Hessian
        ts.py                # Transition state: partitioned RFO, mode
                             #   following (askable since 2026-10-12)
        meci.py              # MECI: two surfaces, branching plane, gap
        mecp.py              # MECP: the same crossing machinery, no coupling
```

Whether a name may be asked for is a property of the class (`Optimizer.is_implemented`), and `implemented_optimizers()` is the list validation checks against. Two reasons land on that flag and they are not the same claim: a placeholder never took a step, and a written search has not been run against the thing it reads.

The path family (`mep`, `irc`, `neb`) is registered in the same directory but is not driven by this
workflow: `resolve_workflow` routes those names to the Path workflow (`workflow/path.py`), which
owns the multi-structure loop over `optimizer/pathbase.py`. Every path class says `is_path = True`,
which is how `Optimize` refuses one that reaches it by mistake. See
[PATH_DEVELOPMENT.md](PATH_DEVELOPMENT.md).

```text
    optimizer/
        mep.py               # minimum energy path (driven by the Path workflow)
        irc.py               # intrinsic reaction coordinate, both branches
        neb.py               # nudged elastic band, geodesic seeds
        pathbase.py          # the path base class and the joint vector
        pathconstraint.py    # sphere / ReLU path constraints
```

Only one framework workflow file (`optimize.py`) is maintained. Different optimization targets are implemented independently without introducing additional strategy/controller subpackages.

For the overall position of the workflow subsystem, see [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 3. Optimization Workflow

The optimization loop is controlled by `OptimizeWorkflow` (`workflow/optimize.py`). Actual per-step sequence:

```text
for step in range(total_steps):
    result = calculator.run()              # energy + gradient
    state = calculator.get_state()
    state.record_step_result()             # official step bookkeeping
    logger.log_step(...)                   # run-log step block (evaluated geometry)
    state = optimizer.run(state)           # geometry update only
    calculator.sync()                      # propagate new coordinates
    converged, report = optimizer.convergence(state)   # independent verdict
    logger.log_optimize_step(...)          # progress block with thresholds
    write opt.xyz / opt_geom.xyz / opt_status.txt into the run's directory
    state.runtime.advance_step()
    if converged:
        break
```

Convergence is judged after `calculator.sync()` by the independent `optimizer.convergence(state)` call, never inside `optimizer.run()`; the skill gradient is converted to Hartree/Bohr (manifest `units` section) between `calculator.run()` and `record_step_result()`.

The step limit is:

```python
total_steps = min(runtime.total_steps, workflow.maxiter, opt_maxit)
```

so the `maxit` keyword of the optimize section bounds the whole optimization, not only the optimizer object.

Responsibilities:

| Component | Duties |
|-----------|--------|
| Workflow | control iterations, call calculator, call optimizer, update calculator coordinates, log progress to the run log, write output files |
| Calculator | evaluate energy and gradients, return updated `PyRAIState` |
| Optimizer | compute next geometry, maintain history and Hessian, apply trust radius, determine convergence |

Output files written per step:

- `opt.xyz` — latest geometry
- `opt_geom.xyz` — all steps appended. Frame N and row N describe the **same structure**: the loop snapshots the geometry the engine was just scored on and writes that snapshot beside its energy/gradient pair, because the optimizer moves the atoms afterwards and the trial it proposes is what the *next* evaluation is computed on. This pairing was not held before #377: on round `tmp_20261017_210205` the shipped pressure leg's frame N evaluated to the energy in row N+1 (`-0.9291051`, `-0.9264462`, `-0.9289608`, `-0.9290993` against the following row to seven digits), so a reader checking a converged structure against the criteria that judged it was reading two different geometries. `test_workflow.py::test_every_row_describes_the_structure_of_its_own_frame` recomputes the energy from each frame's coordinates (the older assertion compared a frame's comment number with the row, and both are copies of the same report, so it could not see the shift).
- `opt_status.txt` — one line per step (step, energy, shift, max/rms displacement, max/rms gradient). Written only by this workflow: the columns are measured here, so a run the engine iterated (`driver` not `main`, section 18) does not produce the file. The table keeps the seven atomic columns; a variable-cell run appends its five cell columns plus `frozen_ride` (#355, [CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 10), the method appends its own numeric readings after the cell block (`neg_modes`/`mode_eig` and, for a box, `rigid_removed` for a saddle; `cross_gap`/`branch_dirs`/`gap_norm`/`coupling_norm` for a crossing), and a composite run appends its per-term columns after them (#384): one column per assembled term, holding `contribution x` that term's energy **at the state the assembly records for it** (`composite_terms['state_indices']`; `None` where the term owns no state and the state the run is driven on decides, which is every single-center run). Reading one global index for every row is what crashed the six composite legs of round 15 (#387), because the rows arrive as the backend's raw per-state arrays and the broadcast to the state axis happens only on the way to the total. The raw rows themselves stay in the run log's `&composite energy decomposition` block; `dev/COMPOSITE_DEVELOPMENT.md` section 16.1 owns that record's shape.
- The three files are written into the run's own directory, resolved the way the dynamics chain resolves it (`workflow/md.py` restamps the logger with a dispatched unit's workdir, and the optimization records read that value instead of deriving a directory of their own). They are opened by `run()`, not by construction: a workflow that was built and never run must not delete the previous run's trajectory.

Run log (`<title>.log`) content for optimize jobs:

- Before the first step `RunLogger.log_optimize_setup` writes one `&optimization setup` block: method, optimizer class, step bound, cell mode/coupling/pressure, band size/spring/climbing, IRC direction/mode/imaginary count, constraint conditions and initial volume. It deliberately uses no `threshold` word and none of the qualification reader's tail keys.
- Each step writes an `&optimization progress` block via `RunLogger.log_optimize_step` (energy, Delta E, gradient/displacement max+rms, trust radius, converged True/False); all values are printed with 8 decimals, right-aligned, and each judged entry (Delta E, gradient/displacement RMS+max) shows its threshold and per-criterion verdict (True/False) on the same line; stdout progress printing was removed in favor of the run log.
  The same block carries the method-specific rows: `Iteration` and `Frozen Ride Max` when measured; a saddle's `Negative Modes (model)`, `Mode Eigenvalue (model)` and `Rigid Directions Removed`; a crossing's `Crossing Gap`, `Branching Directions`, `Cell Gap Term`/`Source`, `Gap Norm`, `Coupling Norm`, `Mean Energy`; a band's `Barrier`, `Image`, `Climb`, `Spring`; a walker's `Arc`, `Length`, `Perp RMS`/`Max`, `Radius`, `Residual`, `Direction`, `Role`, `Settled`. A value the method could not measure prints `Not measured`, never a zero. New rows never carry a threshold, so the five judged atomic rows keep their one-line verdict format.
- Path searches call the same two channels: an accepted point writes the ordinary property block (level-gated) plus `&optimization progress`; a band round writes the progress block only; a rejected walker probe appears at `verbose >= 3`. Their machine record remains `path_status.txt`/`path_profile.txt`, not `opt_status.txt`.
- After the loop `RunLogger.log_optimize_summary` writes the outcome banner: `Geometry optimization converged in N steps (final energy = ...)` or `Geometry optimization did not converge after N steps (...)`; the skill-driven `SingleShot` path writes the same banner with the `Skill-driven` label from the `opt_converge` verdict

---

## 4. Optimizer Responsibilities

The optimizer internally manages:

- Optimization history
- Hessian (inverse) matrix
- Trust radius
- Iteration counter
- Failure (rejection) counter
- Coordinate shape conversion
- Active electronic state index

The optimizer only receives and returns `PyRAIState`.

---

## 5. Coordinate Representation

`PyRAIState` always stores geometries using physical representations:

- Coordinates: `(N, 3)` Angstrom
- Gradient: `(na, 3)` Hartree/Bohr for the active state

The optimizer internally converts `(N, 3)` → reshape → `(3N,)` Bohr for numerical optimization. Before returning, coordinates are converted back to `(N, 3)` Angstrom. Unit conversion factors are resolved from the skill manifest's units section (`coord_to_bohr`, `disp_to_angstrom`).

The workflow and calculator never observe flattened coordinates.

---

## 6. Optimization Algorithms

### Minima Optimization (`minima.py`)

Step computation:

```text
gradient (3N,)
    ↓
BFGS Newton step  x = -H @ g     (H = inverse Hessian)
    ↓  (or, when opt_use_rfo is set)
RFO step from the augmented Hessian
    ↓
trust radius scaling
    ↓
new coordinates
```

The default algorithm is inverse-Hessian BFGS with a trust radius resized on how well the model predicted the step it allowed (Section 8). RFO is available through `use_rfo` (Section 9).

### GDIIS (`gdiis.py`)

GDIIS maintains a sliding window (`window_size`, default 6) of previous coordinates and gradients, solves the constrained DIIS least-squares problem, and builds the extrapolated step:

```text
x_new = sum_i c_i * x_i  -  H^{-1} * sum_i c_i * g_i
```

with a BFGS-updated inverse Hessian for preconditioning (identity on the first step).

`GDIISOptimizer.__init__` accepts `use_rfo` for interface parity with `MinimaOptimizer` but ignores it: GDIIS builds its step from the DIIS extrapolation and has no RFO variant.

Both optimizers are registered in `OPTIMIZER_REGISTRY` and constructed through `get_optimizer(method, ...)`.

---

## 7. Hessian Update

The optimizer internally maintains the inverse Hessian `H`, including:

- initialization (identity matrix);
- inverse BFGS update after each accepted step:

```text
H_new = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
rho = 1 / (s^T y)
```

where `s` is the displacement and `y` the gradient difference. An update is skipped when `s^T y` is not positive, or is negligible against `|s| |y|` (`base.usable_curvature_product`, floor `1e-12` of that product). The floor is relative because a trust-region step is allowed to be small: measured on a box relaxation whose radius had shrunk to its floor, an absolute cut in the units of the problem skipped *every* remaining update, so the curvature model froze at whatever the first steps had taught it and the run spent its last fifty evaluations moving 1e-4 Bohr a time;

- reset after repeated optimization failure.

No separate Hessian manager is introduced.

---

## 8. Trust Radius

The radius bounds the step and is resized on how well the quadratic model
predicted the step it allowed: `rho = actual / predicted`, read in three bands.

```text
predicted change   g.s + 1/2 s.B.s        (B: the curvature, not its inverse)
rho < 0.25         shrink to a quarter
rho > 0.75         grow, up to 0.5 Bohr
in between         leave it alone
floor              1e-4 Bohr
initial            0.3 Bohr
```

Step acceptance policy:

```text
energy(actual new geometry) < energy(current)   → accept
otherwise                                        → reject:
    restore the previous geometry, shrink the radius, re-step under it
4 rejections in a row                            → the workflow stops the run
```

Acceptance is judged on the actual computed energy of the proposed geometry, not
on the quadratic model prediction. The rejection rule belongs to the searches
that minimize one surface: a saddle search climbs its one negative direction on
purpose and a crossing search pays average energy to close a gap, so both set
`reject_on_energy_rise = False` while keeping the resizing.

The resize is not the rejection. A step that was accepted but badly mispredicted
still shrinks the radius, which is what lets a search recover from an over-long
step without throwing away the evaluation it already paid for.

---

## 9. RFO (Rational Function Optimization)

Enabled with `opt_use_rfo: true` in the optimize section (default off).

The RFO step diagonalizes the augmented Hessian:

```text
[ B    g ] [ v ]     [ v ]
[ g^T  0 ] [ 1 ] = λ [ 1 ]
```

where `B = H^{-1}` is the **positive Hessian** (not the inverse Hessian), symmetrized before diagonalization, and the eigenvector is sign-normalized so its last component is positive. The step is `x = v / v_last`.

Properties:

- with a positive definite Hessian the RFO step is always shorter than the Newton step;
- on the test system (ethylene, HF/STO-3G) BFGS and RFO converge in the same number of steps along nearly identical trajectories.

---

## 10. Convergence

The optimizer provides:

```python
converged, report = optimizer.convergence(state)
```

Convergence requires all of:

| Criterion | Keyword | Default |
|-----------|---------|---------|
| Energy change | `opt_energy` | 1e-6 Ha |
| Gradient RMS | `opt_grms` | 3e-4 Ha/Bohr |
| Gradient Max | `opt_gmax` | 4.5e-4 Ha/Bohr |
| Displacement RMS | `opt_drms` | 1.2e-3 Bohr |
| Displacement Max | `opt_dmax` | 1.8e-3 Bohr |

Two further families of row exist for runs that need them, and both are judged
against the keywords above rather than new ones:

- **A variable-cell run** adds five cell rows and is judged on a pressure and a strain rather than
  on the atomic four; the rows, the thresholds, the runner's checks and the readings are
  [CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 10. This document keeps the four
  atomic criteria above and the two-surface rows below.

- **A two-surface run** adds `Gap (Sd-Si)` (1-based, the pair the input named)
  and, for the crossing searches, `Crossing Gap` judged against `opt_energy`
  beside `Branching Directions`.

**The gradient criteria are judged on the tangential part** when the run declares
strict conditions. On the manifold each condition's own direction carries the
force the condition absorbs, so a threshold read there is unreachable; what the
run is judged on is everything else. The absorbed part is not hidden - it is
printed as the condition's Lagrange multiplier in the `&constraint geometry`
section (Section 14). The earlier reading (D-26) removed every *atom* a condition
named from the count, which discarded the tangential half of those atoms with
them. Measured on a bond held along x with a 5 Hartree/Bohr force across it, the
masked reading judged 3 of 9 components and called the run converged over the
largest force in the structure; the directional split reads that force and does
not converge. A frozen component is still not a degree of freedom at all, so it
carries no force either way.

The report keys (exact strings, `workflow/optimizer/base.py`) are: `Iteration`,
`Energy`, `Delta E`, `Gradient RMS`, `Gradient Max`, `Displacement RMS`,
`Displacement Max`, `Trust Radius`, plus the rows above, and a constrained run
also carries the `Constraint` record. The workflow writes the report into the run
log (`RunLogger.log_optimize_step`) with the matching threshold keys.

---

## 11. Optimization Control Keywords

All optimize section keywords are stored in the schema with an `opt_` prefix to avoid clashes with skill sections (both optimize and e.g. pyscf define `method`):

```yaml
optimize:
  method: minima       # → opt_method (minima / gdiis)
  state_i: 1           # → opt_state_i
  maxit: 30            # → opt_maxit
  opt_energy: 1e-6
  opt_grms: 3e-4
  opt_gmax: 4.5e-4
  opt_drms: 1.2e-3
  opt_dmax: 1.8e-3
  opt_use_rfo: true
  optimize_cell: true   # keeps its own spelling; see CELL_OPT_DEVELOPMENT.md
```

Mapping rules:

- nested (sectioned) input: `_flatten_sections` adds the `opt_` prefix to optimize section keys automatically;
- a keyword that already names the section is **not** prefixed (`OPTIMIZE_PASSTHROUGH_KEYS`). `optimize_cell` is that case: prefixing it invented `opt_optimize_cell`, a name no model owns, so the switch was dropped between the input and the optimizer and a run that asked to relax a box relaxed only the atoms. A verification test now reads the keyword through the real flattener;
- flat agent config: `qualification/certification/converter.py yaml_to_config` forwards optimize section keywords with the `opt_` prefix (unprefixed short names are not schema fields and would be dropped);
- the workflow reads `opt_*` keys through `PyRAIState.optimize_params` and caps the loop with `opt_maxit` (Section 3).

For an askable `opt_method`, the framework expands the task jobtype to `energy gradient` and runs its own optimization loop; skills only provide energy and gradients. A two-surface method additionally requests `state_i` and `state_j` per step through `Calculator.run(target_states=...)`. A cell run asks for one thing more: `optimize_cell` is what makes `planning/resolver.py::_derive_stress` append the `stress` capability, so the engine is consulted for a lattice derivative as well as a gradient. The keyword keeps its own name on that path - `OPTIMIZE_PASSTHROUGH_KEYS` in `schema/adapter.py` - and a #355 regression test reads the shipped examples through all three carriers, because the prefixing this set prevents once invented `opt_optimize_cell`, a name no model owns, and a run relaxed its atoms while its box sat still.

---

## 12. Optimization History

The optimizer maintains history of:

- Energy
- Gradient
- Displacement
- Coordinates

for BFGS updates, GDIIS extrapolation, restart, and optimization reports. History remains internal to the optimizer.

---

## 13. Electronic State Selection (state_i / state_j)

The optimize section provides `state_i` and `state_j` keywords to select the electronic states involved in the optimization.

- `state_i` — the electronic state to optimize. Default 0 (ground state S0).
- `state_j` — the second state of a crossing search (`meci`, `mecp`), and the
  state whose energy difference a two-surface run reports. Equal to `state_i`
  means one surface: no pair is requested and no gap row is written. Both
  indices are range-checked against `nstate` at input validation
  (`schema/validate.py::validate_centers`, single-center runs, against the
  normalized `nstate` - the plan's `sum(ci)` is the fallback when the flat key
  is absent), so an out-of-range surface is
  refused at load rather than at run construction; both are read from the
  state-resolved arrays, so no index shifting happens anywhere.

Example (optimize the first excited state S1; see `pyrai3md/examples/optimize_tda.yaml`):

```yaml
optimize:
  method: minima
  state_i: 1
  maxit: 30

molecule:
  ci: [2]          # S0 + S1
  spin: [0]

pyscf:
  method: tda      # must provide excited-state gradients
  basis: sto3g
```

### State Numbering Rule

State indices are 0-based and identical at every layer. No plus/minus one conversion is applied anywhere:

```text
input state_i: 1 (S1)
framework current_state / active_state = 1
skill target_states contains 1
result arrays: energy[1], gradient[1] are the S1 values
```

0 always refers to the ground state S0.

### State-Resolved Result Arrays

Skills return state-resolved arrays:

```text
energy:   (ns,)
gradient: (ns, na, 3)
```

The optimizer reads `energy[state_i]` and `gradient[state_i]` through `_extract_energy_gradient()` using its `active_state` index. A gradient without a state axis `(na, 3)` is accepted for backward compatibility.

One exception, and it is a rule of its own: a **multi-center composite** reports a
single assembled surface - the one the per-center `active_configuration` asks
for - and records that configuration in the same result. Its energy is therefore
the requested state already, and indexing it by the state id addresses a state
the surface never carried (`index 1 is out of bounds for axis 0 with size 1` on
the first multi-center excited-state run, CHANGELOG #459). The optimizer reads
the assembled value when the result carries the marker and keeps the loud failure
for a plain run whose provider returned fewer states than was asked for; a
two-surface composite (a crossing search over the pair axis) keeps indexing its
pair, which the same criterion pins. `state_i` itself takes a scalar (read by
every center) or one index per center.

### Excited-State Optimization

To optimize an excited state, the skill method must provide excited-state gradients (e.g. `tda` or `tddft` for the PySCF skill), and `state_i` selects the target state. The framework forwards `state_i` to the skill as `target_states` when the user does not set it explicitly.

Validation of whether the chosen method supports excited states belongs to the skill itself (skill initialization/normalization), not to the framework workflow. The framework checks only that `state_i`/`state_j` lie inside the declared state manifold, and it does so once, at input validation (`schema/validate.py::validate_centers`) - there is no second range check downstream.

### Practical Notes

Excited-state potential energy surfaces often lack bound minima. For ethylene S1 (ππ\*), the surface is repulsive along the C-C stretch: a symmetry-preserving Cartesian optimizer converges to the planar stationary point, while internal-coordinate optimizers with empirical (ground-state) Hessian guesses may leave the plane and follow the dissociation valley. Endpoints on such surfaces depend on numerical details of the trajectory. The `optimize_tda.yaml` example therefore exercises the state-aware optimization flow with a bounded step limit (`maxit: 30`) rather than asserting a physical minimum.

---

## 14. Constraints during an optimization

Constraints are live on this path, not reserved: `workflow/constraints/` is the
subsystem and `dev/methods/CONSTRAINT_DEVELOPMENT.md` is its design document.
What an optimization does with a declaration:

- the trial step has its frozen components taken out before it is recorded
  (`Optimizer.apply_constraint`), so the reported displacement is the step
  actually taken;
- the trial geometry is then landed on the declared manifold by the position
  solve (`constraints.runtime.constrain_state`), because a bond length is a
  curved condition and a linear projection of a step does not reach it;
- harmonic restraints, collective variables and a metadynamics bias need no
  handling in the optimizer at all: they are forces, and the Calculator
  finalizer adds their energy and gradient to every evaluation;
- the convergence criteria read the tangential part of the gradient and the
  absorbed part is reported as multipliers (Section 10).
- a search that follows curvature also *solves* on that tangential part
  (`Optimizer._step_gradient`, which a minima relaxation leaves alone): the step
  is an eigenvector of the model, so a coordinate the plan pinned is a zero row,
  and a zero row is the lowest curvature the model can hold. The plan is
  likewise the only source of the atom set the numerical curvature batch
  displaces - `hess_atoms` is derived from it during a search, and an input that
  names a different set is refused instead of being overwritten (#336).
- a search accepts whole-atom pins. Pinning one Cartesian component of a
  molecule in vacuum fixes a laboratory direction the molecule has no reason to
  respect, and the torque that leaves behind has no free coordinate to be
  removed along (measured in `dev/TODO.md`); `direction: xyz` holds a fragment
  rigid and keeps the rotation inside the movable set.
- **optimizing only a selected region is the complement of a fix, and it has no
  keyword of its own** (ruled 2026-10-13). A run that wants one region to move
  declares the atoms it wants *held* through the keys that already exist -
  `fix_atoms`, `fix_element`, `fix_group`, each of which `utilities/selection.py`
  resolves from an explicit list, an element or a named group - and the region
  that moves is whatever is left. Nothing is added for it: no field, no index
  notation, no classic-input keyword, because a second way of naming a frozen set
  is a second answer to the question "which atoms may this step move", and the
  projection that answers it (`constraints/projection.py::build_fix_mask`) reads
  one plan. What is genuinely missing is a *grouping* convention - naming a
  molecule rather than listing its atoms - and that stays refused by name
  (`dev/methods/CONSTRAINT_DEVELOPMENT.md` 附录 B section 10), because the bond
  graph and its connected components already exist and what is undecided is the
  spelling, not the machine.

  This is not the `region: inside/outside` question in that same appendix. That
  one is a **bias** region - which side of a collective variable a harmonic
  restraint acts on (`schema/config.py::BiasInput.region`, refused during
  normalization) - and it shares nothing with atom selection beyond the word. The
  two are deliberately left unmerged: a keyword that answered both would be
  answered by neither.

The run log carries one `&constraint geometry` block per progress step, in the
shape the `&constraint energies` block of a dynamics step already uses
(`utilities/print_results.format_constraint_geometry`): one row per declared
condition with its id, its atoms as the input numbered them, its target, its
value at the geometry this gradient was computed on, the residual between them,
and the multiplier; then the position solve's own counters (condition rank,
frozen components, restored components, iterations, max residual). Lengths are
Bohr and angles radian, and the section title says so.

The subsystem's own limits apply here too and are its document's to state: a
strict condition held while the box moves is priced - the condition's own lattice
derivative goes into the cell force beside the stress conjugate
(`constraints/projection.py::cell_virial`, measured in
`verification/test_constraint_cell_coupling.py`) - for a held length, an angle and a
torsion. A fourth kind is refused by name until it is priced the same way, and A5/D-14
is closed for the three measured kinds
([CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 4.3).

---

## 15. Optimization targets

- Minima (implemented)
- GDIIS (implemented)
- Transition State (implemented: partitioned RFO with mode following, Section 15.2)
- MECI (implemented: two surfaces over one branching plane, Section 15.2)
- MECP (implemented: the same machinery with the coupling dropped, Section 15.2)
- MEP (implemented; the Path workflow owns it, [PATH_DEVELOPMENT.md](PATH_DEVELOPMENT.md))
- NEB (implemented; the Path workflow owns it, for the same reason)
- IRC (implemented; the Path workflow owns it)

All optimization targets but the path family share the one optimization workflow
and calculator interface, the five criteria, the constraint hooks and the `opt_*`
thresholds; the path family is routed to the Path workflow (section 2).

### 15.1 Cell optimization (`optimize_cell`)

`optimize_cell: true` is askable for every search this workflow drives - `minima`, `gdiis`, `ts`,
`meci` and `mecp` - and it adds the six movable lattice entries to the optimized vector. The cell half is
not an optimizer algorithm: which engine may carry a cell, the gate that refuses the rest, the
strain/stress conjugate, the ride convention, the pressure objective, the coupling, the periodic
curvature frame a saddle or a crossing needs, the seventeen variable-cell examples and their
readings are all owned by [CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md). What belongs here is
the one sentence that ties it to the algorithms: every search takes the keyword, and the three
combinations that are refused by name - a backend-supplied curvature on a saddle, a vibrational
analysis, and a crossing whose backend answers one whole-system tensor - fail at construction or at
the evaluation, not at the name. What this document still decides directly is the target list
above: GDIIS is a minima branch and `ts` is another class, so the two do not combine.

### 15.2 The saddle and the crossing

**Transition state** (`opt_method: ts`, askable since 2026-10-12). The step is a
partitioned rational-function search: the curvature model is diagonalized, and the
augmented problem is solved separately on one chosen mode and on the rest. Along
the chosen mode the *upper* root of the 2x2 problem is taken, which lies above the
curvature itself, so the denominator is negative and the coefficient sends the step
uphill; over the remaining modes the *lower* root of their augmented problem is
taken, which lies below every curvature in the block, so each coefficient descends.
That is the saddle condition read off the model, and it removed the case the earlier
version needed a hand for: a model with no negative curvature required the lowest
direction to be flipped, which also made the reported count of downhill directions a
statement about the step taken rather than about the model. The mode to climb is
chosen once (lowest curvature) and then *followed* by overlap with the direction
climbed last time, so a search does not change what it climbs when two curvatures
cross. Between evaluations the model is kept by Bofill's mixture of the rank-one and
symmetric-Broyden updates, which tolerates the negative `s . y` a saddle region hands
out constantly; a BFGS update refuses those steps, and an update that refuses them
leaves the search holding its initial curvature for the rest of the run.

Two readings of that model matter, and both are now taken in the vibrational
subspace. The count of downhill directions is a count over the rigid-projected
curvature, because a (3N, 3N) Cartesian model also carries the whole-body motions
with whatever curvature the updates left there: at the initial geometry of a
measured run the raw matrix showed five negative entries where the projected one
showed the single imaginary frequency the engine reported, so counting the raw
entries refuses a saddle the search has already found. The followed direction is
likewise stored in the fixed mass-weighted frame, not as components in the current
basis - the basis is rebuilt from the geometry at every evaluation, so components
compared across two of them are one motion's shadows in two frames, and a search
that does that walks uphill with a growing gradient (measured on PySCF: from a
gradient of 0.029 Hartree per Bohr to 3.3 in 60 evaluations).

What the frame takes out of the movable set depends on whether anything holds the
molecule in place. With nothing frozen the whole-body motions are directions the
surface offers no resistance to, and all six (five for a linear structure, three
for a lone atom) go. With a `fix` in place only the combinations the anchor
*absorbs* are still free: a motion that leaves every frozen atom where it stands
costs nothing, while one the anchor opposes has become an ordinary deformation
with an ordinary curvature and has to stay available. Measured on the six-atom
analytic saddle with atom 6 pinned: fifteen components are movable, three of them
are absorbed by the anchor, and the twelve that remain carry exactly one negative
curvature (-0.296, then +0.115 up to +89.4). Removing all six candidate motions
from those fifteen, which is what this frame did until #338, leaves nine
directions and holds the search at `max_g 1.73e-2` for 400 evaluations, because
the force along a partial translation of the free atoms is real and nothing else
in the search can answer it; the same run with the twelve converges at 0.03, 0.10
and 0.35 Bohr off the saddle in 73, 92 and 112 evaluations. The count is read on
the same frame it steps on, which is why a triatomic with one atom pinned reports
one downhill direction (-0.193, +0.501, +15.593) instead of the none the
over-projected frame showed, and reaches its saddle in 86 evaluations.

Curvature comes from either producer, named by `&hessian hess_driver`: `main`
displaces the geometry through the framework's own batch, `skill` is asked for the
matrix it computes (see [NUM_HESS_DEVELOPMENT.md](NUM_HESS_DEVELOPMENT.md) section
5). The request happens once, before the first step, and one that returns no matrix
is refused rather than stepped on without one. The step is bounded by the search's
own acceptance rule: an energy rise is not a rejected step - climbing is the point -
but a step whose energy change has the opposite sign to the one the model predicted
is, and a rejection at the radius floor is skipped rather than repeating the move it
just refused. The radius ceiling is 0.1 Bohr, set by measurement: 0.05 was the only
bound tested that diverged to +2.2 Hartree from an isotropic starting curvature,
while 0.1 reaches the saddle in 17 of 21 evaluations from the true and from the
isotropic model on an analytic three-atom saddle.

**A saddle in a relaxable box** extends the same search with six lattice entries. The
framework's curvature batch covers them, a backend block is joined rather than zero-filled, and the
projection frame weights the lattice entries by the identity; the batch, the three refusals and the
bit-identical no-lattice criterion are
[CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 6.2.

**Conical intersection and singlet-triplet crossing** (`meci`, `mecp`, sharing
`CrossingOptimizer`). The step is taken on the gradient of one objective built
from both surfaces: the average gradient with the branching plane removed, plus
the gap's own slope. At the crossing the second term vanishes and the first is
what a stationary point of the crossing condition asks to be zero, so the five
criteria keep their meaning while the gap is judged separately against
`opt_energy`. The plane is spanned by the gradient difference and the derivative
coupling; a pair of different spin has no coupling to span with, so `mecp` drops
that direction and reports `Branching Directions 1.0` where `meci` reports 2.0.
A single-surface evaluation, or one index named twice, is refused by name.

Both read the pair from the state-resolved arrays the framework already carries
(`energy (ns,)`, `gradient (ns, natom, 3)`, `nac (npairs, natom, 3)` addressed
through `system.nac_pair_map` the way the hopping integrator addresses it), and
both request it per step, so no new result channel was opened for them.

Which two surfaces a search follows (`opt_state_i`/`opt_state_j`) is not the same
statement as which couplings get computed (`molecule.coupling`). Since #447 the
planner scans `optimize.opt_method` into the capability request
(`planning/resolver.py::crossing_coupling_pair`), and the task builder selects the
declared pair that matches the chased one
(`data/core.py::_task_property_request` → `_declared_pair`). A search that spans its
plane on a coupling the input does not declare is therefore refused by name before
the first evaluation, instead of running on a plane of one direction; and a backend
whose manifest offers no coupling (LAMMPS, PySCF) is handed no `nac` token, so the
search runs on the gradient difference alone - the documented reading, not a silent
zero.

**A crossing in a relaxable box** steps the box on the average of the two surfaces' conjugates
and adds the gap's own lattice derivative. The state-resolved tensor, the degeneracy boundary and
the `Cell Gap Term` readings are [CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 6.3.

---

## 16. Design Principles

The optimizer follows several principles.

**Separation of Responsibilities**

- Workflow controls execution.
- Calculator performs property evaluation.
- Optimizer performs numerical optimization.

**Lightweight Architecture**

Avoid introducing unnecessary abstraction layers. No Strategy Manager, Controller Manager, Hessian Manager, or Observer. The optimizer remains compact and maintainable.

**Extensibility**

Additional optimization algorithms can be added by introducing new modules under `optimizer/` and registering them in `OPTIMIZER_REGISTRY`, without modifying the optimization framework.

**Runtime Independence**

The optimizer is independent of Workflow implementation, Calculator implementation, and electronic structure methods. It only consumes and produces `PyRAIState`.

---

## 17. Examples and Qualification

Official optimize examples under `pyrai3md/examples/`:

| Example | Method | Description |
|---------|--------|-------------|
| `optimize_minima.yaml` | minima (BFGS) | Ground-state optimization of ethylene (HF/STO-3G) |
| `optimize_gdiis.yaml` | gdiis | Ground-state optimization of ethylene (HF/STO-3G) |
| `optimize_tda.yaml` | minima (BFGS) | S1 excited-state optimization of ethylene (TDA/STO-3G, `state_i: 1`) |
| `optimize_ts.yaml` | ts (partitioned RFO) | HCN->HNC saddle search on the framework's displaced-gradient curvature: 32 evaluations, one imaginary mode at the endpoint (-1248.42 cm-1) |
| `optimize_ts_skill_hess.yaml` | ts | the same search with `hess_driver: skill`, one dispatched property job for the matrix: same 32 evaluations, -91.5650982834 against -91.5650982833 |
| `optimize_ts_multiscale.yaml` | ts | the same search over a single-center composite (one program on two layers, three tasks per evaluation): reproduces the single-layer endpoint to the printed digit |
| `optimize_partial_minima.yaml` | minima | HCN with the C-N fragment pinned: 27 evaluations against 23 with the constraint removed, the pinned atoms bit-identical throughout |
| `optimize_partial_ts.yaml` | ts | water dimer with the donor oxygen held: 75 evaluations to -149.8362656686, one imaginary mode (-3064.0 cm-1) on the endpoint's own frequency job, the held atom bit-identical throughout |

All of them are qualified in the YAML mode; the three ethylene relaxations are also qualified in the classic-input and agent modes. The eight rows above are the fixed-cell half of the route; the seventeen variable-cell ones are
listed with their own readings in
[CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 9, and twenty-five `optimize_*` YAMLs
are what the `optimize_test` family switch opens.

Four of these eight keep their own key at `false` in the shipped `pmd_env.yaml` (`optimize_minima`, `optimize_gdiis`, `optimize_partial_ts`, `optimize_tda`), and they still run in an optimize round: the family is consulted before the case, so `optimize_test: true` opens them - which is what rounds 8 and 9 did, all twelve on two carriers each (the executed stems are listed case by case in `CHANGELOG.md` #360), and round 10 ran thirteen with `optimize_cell_pressure` added by #364. For the example framework and qualification policy, see [EXAMPLE_DEVELOPMENT.md](../EXAMPLE_DEVELOPMENT.md) and [QUALIFICATION_DEVELOPMENT.md](../QUALIFICATION_DEVELOPMENT.md).

---

**Skill-driven optimization has its own step ceiling.** BAGEL's optimizer trusts the gradient it is handed, and an initial analytic CASPT2 gradient for a near-degenerate reference can be enormous (`1.27e3` a.u. measured on the shipped MECI case), which sent the first internal-coordinate step off the surface and made BAGEL abort. The generated optimize input therefore sets `maxstep` (`bagel_skill/service/builder/input_builder.py`, `_OPT_MAX_STEP = 0.05`), matching the vendor's own template; both optimize profiles then descend monotonically from the first step. Framework-side `opt_*` knobs are unaffected.

## 18. Skill-Internal Optimization (driver ≠ main)

When `control.driver` is not `main`, the jobtype `optimize` never reaches this workflow or the optimizer registry: the planning layer keeps the jobtype on the skill and `resolve_workflow` returns the `SingleShot` workflow (`workflow/single.py`). The skill iterates the geometry optimization internally and reports back through the `opt_converge` contract (plus the final structure through the `fin_coord` channel); `SingleShot` writes `opt.xyz`/`opt_geom.xyz` — the final structure is a real product of the run — and settles the outcome from `opt_converge` (`log_optimize_summary` with the `Skill-driven` label; `opt_converge: False` marks the runtime failed).

It writes no `opt_status.txt`. That table belongs to section 6's loop, whose steps this path does not have: the framework took no step, moved no atoms and measured no gradient, so the only row it could have written carried zeros in every column that matters and read as an optimization that converged on its first iteration. Earlier rounds wrote exactly that.

For truncated-step qualification runs that end unconverged, the qualification runner detects the `Maximum optimization steps reached` signature (run log or stderr) and reports `PASS[NOT CONVERGED]` instead of a failure (`qualification/runner.py`). Both routes carry the annotation since #399: an example leg exits 0 while the framework's own log records the unconverged state, and that route used to read as a bare `pass` - so all 32 variable-cell legs of round `tmp_20261019_200652` were indistinguishable from converged ones. The exemption is unchanged (still `pass`, no third outcome); what is new is the marker, the `not_converged` flag, and `tools/verify.py`'s `budget_stopped` count over those legs' own closing lines.

**A box relaxed by a skill's own profile is that profile's business.** `optimize_cell` has exactly
one reader - this workflow's optimizer - so `resolve_workflow` refuses the pairing of the keyword
with a `driver` that is not `main` and names the engine-side profiles instead. The LAMMPS
`ff_optimize_cell` profile and its measured box are
[CELL_OPT_DEVELOPMENT.md](CELL_OPT_DEVELOPMENT.md) section 6.4.
