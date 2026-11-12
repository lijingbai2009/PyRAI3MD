# PyRAI3MD Molecular Dynamics (NAMD) Design

Last updated: 2026-10-23 23:52 (section 25 gained the caller-owned starting-point path: `AIMD(initial_conditions=..., adopt_initial_conditions=True)`, its validation helper and the batch-length rule. Earlier 2026-10-22 22:20 (sections 1/2/3.3 now state the split half-kick form the code runs: `verlet_i` stores `V(t+dt/2)` rather than leaving `V(t)`; sections 5/9 quote `Calculator.run()` as returning a `Result`; section 15's step order gained the phase-0 `apply_ensemble(stage='pre')` half step and the deviation merge; the output section carries the writer's current signature and the charges and per-center suffix rows. The rest was not re-audited.

Before it, 2026-10-13 14:22 (the `apply_pbc` location was checked against the code; section 28 carries the fourth recorded-step
sentinel, its measured cost, and the fact that its two-center shape now runs on real engines;
the rest follows as written)

---

> **The `&md` keyword surface and the recorded step (2026-10-08).** See CHANGELOG #314. The
> velocity-control family is implemented (`excess`/`scale`/`target`/`randvelo`/`reset`/
> `resetstep`, §14 and §23), `addsoc` biases the spin-orbit couplings of one evaluation, the
> dissipation gate `dissipate` delays the thermostat until the run has been on the ground state
> for `thermodelay` steps (§28), and every archive spelling that nothing reads is now refused by
> name. `verification()` is gone: the questions it reserved a slot for are asked of each
> recorded frame in `AIMD.checkpoint` (§28). A resumed segment continues the surface-hopping
> random stream, because the generator state now travels in `<title>.rt` (§23).

> **Ensemble trajectory output (2026-09-25).** See CHANGELOG #218. `workflow/records/output.py`
> adds `md.ensemble`/`sh.ensemble`: one line = time + the per-step thermostat/barostat
> record (instantaneous and target temperature, N_df, Ekin, volume, velocity scale
> factors, Nose-Hoover chain masses/positions/momenta or the CSVR rescaling parameters,
> the kinetic/virial/internal/target pressures in bar, and the barostat matrices
> mu / g_eps / eps / p_eps / v_eps as nine row-major components in atomic units; MTTK
> adds the cell-momentum bath). The column set is planned once from the dynamics
> keywords (`_ensemble_columns`) and headed lazily on first write; **the files are
> created only for nvt/npt runs** — an nve run produces no file and stays byte-for-byte
> unchanged. The per-step intermediates come from the `runtime.ensemble` diagnostics
> recorded by `workflow/integrator/ensemble.py`; persisted extended DOF are read from
> the snapshot system fields. The run log gains a matching `&ensemble` section
> (`utilities/print_results.py::_format_ensemble`, rendered for nvt/npt dynamics in
> every workflow that reuses `format_result_block`). Partitioned runs write only the
> global ensemble files. File inventory and formats are given below.

> **Stress trajectory output (2026-09-24).** See CHANGELOG #213. `workflow/records/output.py`
> adds `md.stress`/`sh.stress` (`_write_stress`): one line = time + the 9 row-major
> components of the current-state 3×3 stress tensor (Hartree/Bohr^3) + pressure
> P=−Tr/3 (bar). **The file is created and a line is written only when the backend
> populated a non-zero stress for that step**; non-NPT runs do not produce this
> file and their output is byte-for-byte unchanged; partitioned runs write only
> the global stress and produce no per-center stress files. File inventory and
> formats are given below.

> **Multi-center unification (refactor R2-R6, 2026-09-08).** See CHANGELOG #204 and
> `dev/MULTICENTER_DEVELOPMENT.md` (§10 history, nuclear integration, and energy
> ledger). This round's R6 does not change the MD main execution strategy or the
> order in which data is computed/saved/read; it is only a single-center →
> multi-center functional extension, and C=1 without environmental partitioning is
> bit/byte unchanged. Key points: verlet_i/ii read the "assembled active combination
> gradient" (current step `state.active_gradient()`, previous-step frame
> `nuclear_sample.active_gradient`) and no longer pick gradient rows by historical
> state number (the discrete "history A state / current B state" semantics are
> preserved); surface hopping is dispatched per center through `ElectronicStepper`
> in `workflow/integrator/electronic.py` (zero changes to the cores, each center's
> velocity write-back is masked to that center's atoms); per-step target_states are
> derived per center (`resolve_target_states_centered`). Trajectory output
> (`workflow/records/output.py`) adds the center protocol: with environmental partitioning
> (`plan.n_layer>=2`) the global `md.xyz`/`md.velo` keep only the whole-system
> coordinates/velocities, `md.energies` columns are
> `time|Epot|Ekin|Etot|per-center active state|per-center per-state energies`, and
> `md.pop` has one line per time step with columns expanded over each center's
> states; region results are additionally saved per 1-based center index as
> `md.#.xyz`/`md.#.velo`/`md.#.energies`/`md.#.pop` plus hopping-point `sh.#.*`, in
> the same formats as the global files but scoped to that center's region. For a
> single center with environmental partitioning, the global files degrade to the
> single-center results and `md.1.*` is written in addition; for a single center
> without environmental partitioning, only `md.*` is written (current bytes
> unchanged). Checkpoint serialization and resume are wired: the RNG stream travels
> as `RuntimeState.random_state` (`data/runtime/state.py`, stamped at
> `workflow/md.py:868` and restored on resume), the loop runs on absolute step
> numbers, and the output appends rather than overwrites (#314/#316; the residual
> open item is the engine-side repeat of the same pair on LAMMPS' persistent and
> subprocess channels, recorded in dev/TODO.md section 2).

## 1. Design Goals

PyRAI3MD's NAMD Workflow uses velocity-Verlet integration based on Cartesian coordinates.

The first version reuses the two-stage implementation already validated in PyRAI2MD:

```text
verlet_i
    ↓
Calculator
    ↓
phase correction / external potential / constraint
    ↓
verlet_ii
```

where:

- `verlet_i()`: updates the next-moment coordinates from the current coordinates, velocity, and gradient;
- Calculator: computes electronic structure properties (energy, gradient, NAC, SOC, etc.) at the new coordinates;
- `verlet_ii()`: updates the velocity using the gradients of the previous and current moments.

This implementation is mathematically equivalent to the textbook standard velocity-Verlet. It is written as two half kicks rather than one merged update because the cell-aware drift of an NPT step has to scale the velocities it is holding mid-step: `verlet_i` computes `v(t+dt/2) = v(t) - 0.5 G(t)/m dt` and advances the coordinates with that half velocity (`integrator/verlet.py:291-300`), and `verlet_ii` completes the second half kick.

---

## 2. Standard velocity-Verlet vs. the Current Implementation

The standard velocity-Verlet can be written as:

\[
\mathbf v_{n+\frac12}=\mathbf v_n+\frac12\mathbf a_n\Delta t
\]

\[
\mathbf R_{n+1}=\mathbf R_n+\mathbf v_{n+\frac12}\Delta t
\]

Compute:

\[
\mathbf a_{n+1}=-\frac{\mathbf G_{n+1}}{m}
\]

then:

\[
\mathbf v_{n+1}=\mathbf v_{n+\frac12}+\frac12\mathbf a_{n+1}\Delta t
\]

Merging the two velocity updates gives:

\[
\mathbf R_{n+1}=\mathbf R_n+\mathbf v_n\Delta t-\frac12\frac{\mathbf G_n}{m}\Delta t^2
\]

and:

\[
\mathbf v_{n+1}=\mathbf v_n-\frac12\left(\frac{\mathbf G_n}{m}+\frac{\mathbf G_{n+1}}{m}\right)\Delta t
\]

PyRAI3MD produces the latter two equations, but through the split pair of half
kicks rather than by writing the merged velocity update: the coordinate step uses
`v(t+dt/2)`, which is algebraically `v_n dt - 0.5 (G_n/m) dt^2`.

Therefore:

```text
verlet_i:
    (Vn, Gn) → Rn+1

Calculator:
    Rn+1 → Gn+1

verlet_ii:
    (Vn, Gn, Gn+1) → Vn+1
```

This is an equivalent implementation of the standard velocity-Verlet.

---

## 3. verlet_i() Design

### 3.1 Function

`verlet_i()` is only responsible for coordinate propagation.

Mathematical expression:

\[
\mathbf R_{n+1}=\mathbf R_n+\mathbf V_n\Delta t-\frac12\frac{\mathbf G_n}{m}\Delta t^2
\]

where:

- \(\mathbf R_n\): current Cartesian coordinates;
- \(\mathbf V_n\): current velocity;
- \(\mathbf G_n=\nabla E(\mathbf R_n)\): current gradient;
- \(m\): atomic mass;
- \(\Delta t\): MD timestep.

### 3.2 Inputs

`verlet_i(state)` obtains from `PyRAIState`:

```text
coordinates
velocity
gradient
mass
timestep
current electronic state
freeze information
gradient-descent flag
```

The gradient should come from the current active electronic state.

### 3.3 Outputs

It updates only:

```text
coordinates
```

i.e.:

```python
state = verlet_i(state)
```

After execution:

```text
state.current.coordinates
    =
R(t + dt)
```

and the velocity is the half-step value:

```text
V(t + dt/2)
```

so at this point the state is in an intermediate integration state - the drift and the
following evaluation read the half-kicked velocities (`verlet_i` writes
`updates['velocities'] = half_velocities`, `integrator/verlet.py:306-316`), which is what
the reprojection after the cell rescale and `verlet_ii`'s "skip when the previous
gradient is absent" guard both depend on.

---

## 4. Special Handling of the First Timestep

In PyRAI2MD:

```python
if itr == 1:
    return traj
```

The reason is that the initial timestep does not yet have complete previous-step integration information.

PyRAI3MD should explicitly define:

```text
initialization
    ↓
initial electronic structure calculation
    ↓
initial gradient
    ↓
initial velocity
    ↓
start Verlet propagation
```

Therefore the first-step initialization logic should not be mixed into the general `verlet_i()` mathematical formula.

Current implementation (`workflow/integrator/verlet.py`): on the first step (when history contains no recorded step yet), `verlet_i()` and `verlet_ii()` return the original state directly inside the functions (no-op), so the MD main loop needs no first-step branch.

Recommendation:

```text
NAMD initialization
    ├── initialize coordinates
    ├── initialize velocity
    ├── calculate initial E/G/NAC/SOC
    └── initialize history
```

then formally enter:

```text
NAMD timestep loop
```

This way `verlet_i()` itself can stay simple.

---

## 5. Relationship Between the Calculator and Verlet

The Calculator is not part of the Verlet algorithm.

Its responsibility is:

given the current coordinates, compute electronic structure properties.

Therefore:

```python
state = verlet_i(state)

result = calculator.run(target_states=target_states)

state = verlet_ii(state)
```

is the recommended data flow. `Calculator.run()` returns the evaluation's `Result`
(`workflow/calculator.py:657-678`), not a state: the geometry and properties are written
into `state.pmd` by the Calculator itself, and what the loop takes from the return value
are the deviations (`workflow/md.py:611-620`).

More completely:

```text
               PyRAIState
                   │
                   ↓
             ┌───────────┐
             │ verlet_i  │
             └─────┬─────┘
                   │
                   ↓
             R(t + dt)
                   │
                   ↓
             ┌───────────┐
             │ Calculator│
             └─────┬─────┘
                   │
                   ↓
          E/G/NAC/SOC(t+dt)
                   │
                   ↓
       phase correction / constraint
                   │
                   ↓
             ┌───────────┐
             │ verlet_ii │
             └─────┬─────┘
                   │
                   ↓
             V(t + dt)
```

The Calculator should not be embedded in Verlet.

---

## 6. verlet_ii() Design

### 6.1 Function

`verlet_ii()` updates the velocity from the gradients of two timesteps:

\[
\mathbf V_{n+1}=\mathbf V_n-\frac12\left(\frac{\mathbf G_n}{m}+\frac{\mathbf G_{n+1}}{m}\right)\Delta t
\]

It therefore needs:

```text
G(t)
G(t+dt)
V(t)
```

### 6.2 The Role of History

PyRAI3MD does not need to maintain a dedicated `grad1`.

The previous-moment gradient should be obtained via:

```python
previous = state.get_history_step(-1)
```

The current gradient comes from:

```python
current = state.current_state
```

So conceptually:

```python
previous_gradient = state.get_history_step(-1).gradient
current_gradient = state.gradient
```

then perform:

```python
velocity -= 0.5 * (
    previous_gradient + current_gradient
) / mass * timestep
```

This way Verlet and History connect naturally.

---

## 7. Time Semantics of History and Verlet

The relationship between the current state and historical states must be explicit:

```text
history:
    t-3
    t-2
    t-1

current:
    t
```

Therefore:

```python
state.get_history_step(-1)
```

represents:

```text
t - dt
```

while:

```python
state.current
```

represents:

```text
t
```

In `verlet_ii()`:

```text
previous.gradient = G(t-dt)
current.gradient  = G(t)
```

If the current coordinates are:

\[
R(t)
\]

then the two gradients correspond exactly to the velocity-Verlet update of the velocity from:

\[
V(t-dt)\rightarrow V(t)
\]

---

## 8. When History Is Updated

History should not be updated inside `verlet_i()`.

The reason is that after:

```python
verlet_i()
```

the state is in the intermediate state:

```text
R(t+dt)
V(t)
G(t)
```

This is not a complete MD timestep.

Therefore History should be updated after all physical operations of a timestep have finished.

Recommended:

```text
verlet_i
    ↓
apply_pbc
    ↓
calculator.run(target_states)
    ↓
verlet_ii
    ↓
phase_correction
    ↓
apply_constrain
    ↓
velocity_correction
    ↓
apply_ensemble
    ↓
apply_constrain (constrain_after_cell)
    ↓
kinetic energy (one sample, step-final velocities)
    ↓
update_history() (records the pre-hop state)
    ↓
surface_hopping
    ↓
checkpoint (writes the step, then watches the frame it wrote)
    ↓
calculator.sync()
    ↓
timestep complete
```

The exact position is determined by the NAMD Workflow's state definition (see Section 15 for the current implementation), but it must be guaranteed that History stores snapshots of complete timesteps (the pre-hop active state), not Verlet intermediate states.

---

## 9. NAC Phase Correction

In PyRAI2MD:

```python
traj.phase_correction()
```

is located at:

```text
new coordinates
    ↓
new electronic properties
    ↓
phase correction
```

This position is reasonable.

Phase correction is not a component of Verlet.

It belongs to:

```text
electronic property temporal consistency correction
```

Therefore in PyRAI3MD it should remain independent:

```python
result = calculator.run(target_states=target_states)
state = phase_correction(state)
```

Its inputs can obtain the previous-moment electronic properties through History.

For example:

```text
previous:
    NAC(t-dt)
    SOC(t-dt)
    ...

current:
    NAC(t)
    SOC(t)
    ...
```

from which phase correction is performed.

In this refactor, only a placeholder function is kept. When the phase function is created in the future, it must not modify state internals directly, but go through the official get and update operations.

**Update (2026-10-08, #314).** The phase correction is no longer a placeholder. `workflow/integrator/operations.py::phase_correction` ports the legacy `trajectory.phase_correction`: it gates on `&md phasecheck`, compares each coupling pair's inner product with the previous recorded step, flips the sign of the pairs that came out negative, and leaves a pair whose overlap is at or below `PHASE_OVERLAP_EPS` alone (a coupling crossing zero leaves only noise in the vectors, so its phase is not a fact about the molecule). The corrected array is written back through `state.set_result`, and the loop calls it immediately after the calculator and before the history snapshot, so the record and the hopping integrator see the same vectors.

---

## 10. External Potential

In PyRAI2MD:

```python
traj = self.ext_pot.apply_potential(traj)
```

If the external potential changes:

\[
E_{\mathrm{electronic}}+E_{\mathrm{external}}
\]

as well as:

\[
G_{\mathrm{electronic}}+G_{\mathrm{external}}
\]

then it should be located at:

```text
Calculator
    ↓
electronic E/G
    ↓
external potential
    ↓
total E/G
    ↓
verlet_ii
```

i.e.:

```python
state = calculator.run()
state = apply_external_potential(state)
state = verlet_ii(state)
```

The external potential should not become an internal implementation detail of the Calculator.

**Update (2026-10-08, #314).** No standalone external-potential operation was built, and the
need it was written for is met elsewhere: an additive energy and gradient enters through the
constraint potential families (`bias`, `wall`, metadynamics' history potentials) and is applied
in the Calculator's result finalization (`workflow/calculator.py::_finalize_constraints` →
`workflow/constraints/runtime.py::finalize`), i.e. on the evaluation the step just did rather
than on a second pass over the state. The reason that placement, and not the one sketched
above, is the one that holds is written out in `dev/methods/CONSTRAINT_DEVELOPMENT.md` §7:
an additive term applied after `apply_constrain` misses both velocity kicks of the split step
and would be counted twice on a replayed step. A legacy `ext_pot`/`apply_potential` reader
does not exist in this package, and `apply_constrain` no longer claims to add one.

---

## 11. Constraint

PyRAI2MD's original:

```python
freeze_atom()
```

will in the future be uniformly handled by:

```python
state = constrain(state)
```

Constraint is not part of Verlet itself.

Recommended structure:

```text
Calculator
    ↓
raw gradient
    ↓
constraint
    ↓
corrected gradient
    ↓
Verlet II
```

The first version only reserves the interface:

```python
# TODO: apply constraints
state = constrain(state)
```

The concrete constraint algorithms will be developed independently later.

**Update (2026-10-08, #314).** They were, and `state = constrain(state)` is no longer a reserved
interface: `apply_constrain` delegates to `workflow/constraints/runtime.py::constrain_state`,
which restores frozen components to their anchors, solves the strict bond/angle/torsion
conditions (SHAKE, with the RATTLE velocity half), and returns the geometry the plan declares.
The authority for that subsystem is `dev/methods/CONSTRAINT_DEVELOPMENT.md`; what belongs to
this document is only its position in the step, which §15 records.

---

## 12. Frozen Atom

If certain atoms are frozen, then:

\[
V_i=0
\]

and the corresponding gradients do not participate in dynamics propagation.

Therefore frozen-atom handling should be uniformly the responsibility of the Constraint module, instead of writing separately in `verlet_i()` and `verlet_ii()`:

```python
velo[freeze] = 0
grad[freeze] = 0
```

The goal is to avoid Verlet gradually becoming a "big function" containing a large number of special rules:

```text
Verlet
    ├── freeze
    ├── PBC
    ├── constraint
    └── external potential
```

---

## 13. Kinetic Energy

PyRAI2MD:

```python
traj.kinetic = np.sum(
    0.5 * traj.mass * traj.velo ** 2
)
```

corresponds to:

\[
\frac12\sum_i m_i|\mathbf v_i|^2
\]

This is a diagnostic / system property and does not belong to Verlet.

It is therefore recommended to keep it independent:

computed directly through the State property:

`state.kinetic_energy`.

---

## 14. Velocity Correction

In PyRAI2MD:

```python
reset_velo()
```

is used for:

- removing overall translation of the system;
- removing overall rotation;
- overall translational velocity correction for periodic systems;
- other velocity normalization / correction. It returns the state untouched when the system has fewer than two atoms: one pseudo-atom has no internal degree of freedom, so removing its mass-weighted centre deletes all of its motion (this is the shape of every Tully and Landau-Zener model trajectory, and the reason a benchmark run's total energy stopped being annihilated by the correction).

None of these belong to velocity-Verlet.

It is recommended to keep it independent:

```python
state = velocity_correction(state)
```

Do not write it into:

```python
verlet_ii()
```

**Update (2026-10-08, #314).** The operation is implemented and it now has two halves with
different defaults.

- The translation half runs on **every step of every run**: the mass-weighted center-of-mass
  velocity is subtracted and the result is rescaled so the kinetic energy is what it was
  (`operations.py::velocity_correction`, the "flying ice cube" suppression ported from the
  legacy `reset_velo`). A state with fewer than two atoms returns untouched, because removing
  the center of one pseudo-atom deletes all of its motion.
- The rotation half is opt-in and periodicity-gated: `&md reset 1` asks for it,
  `&md resetstep` says how often (0 = the first step only; otherwise the absolute step number
  modulo the cadence), and a run with `any(pbc)` never takes it. The reason a box is excluded
  is the same one the translation-only default had: the flow inside a cell is the box's own
  angular momentum, and projecting it out would rewrite the momentum balance the barostat
  reads. Where it does act, the rigid part is removed about the mass-weighted principal axes
  (the inertia tensor is diagonalized with `eigh`, so the legacy Kabsch sign check on the
  eigenvectors is not needed: flipping an axis flips its lever arm and its velocity component
  together), and the velocities are rescaled back to the kinetic energy the step had.
- `velocity_initialize(state)` is the other entry, called once per trajectory from
  `_run_trajectory` and only for a process that starts at step 0. It draws
  (`randvelo: 1`) or scales (`excess`/`scale`/`target`) the initial velocities; see §23 for
  what each keyword means and why the three scaling keywords may not coexist.

---

## 15. The Implemented NAMD Timestep (Actual Order in AIMD.run)

`AIMD.run()` in `workflow/md.py` implements each timestep in the following order (the operations come from the `workflow/integrator/` subpackage):

```python
def step(state):

    # 0. Extended-ensemble pre half step (no-op for NVE)
    state = apply_ensemble(state, stage='pre')

    # 1. Nuclear propagation (internal no-op on the first step)
    state = verlet_i(state, dt=dt)
    state = apply_pbc(state)

    # 2. Electronic structure calculation
    target_states = resolve_target_states(
        state.activestate, state.current_state, state.nstate)
    result = calculator.run(target_states=target_states)
    merge_deviations(state.pmd.results, fetch_deviations(result))

    # 3. Complete velocity + independent operations
    state = verlet_ii(state, dt=dt)
    state = phase_correction(state)
    state = apply_constrain(state)
    state = constraint_commit(state)
    state = velocity_correction(state, step)
    state = apply_ensemble(state, stage='post')
    state = apply_constrain(state)     # re-project after the cell rescale
    state.kinetic_energy()             # one sample, from the final velocities

    # 3b. Tracked geometry (legacy GeomTracker), read on the final geometry
    #     and stored in the step's results so the frame carries it
    state.pmd.results['geometry_track'] = tracker.evaluate(
        track_spec, continuous_coordinates(state.coordinates, state.pmd.system))

    # 4. Record the pre-hop step into the history
    state.update_history()

The kinetic energy is sampled once per step, after every action that writes velocities (the centre-of-mass correction, the ensemble post half step, the second constraint projection). The run log's `&kinetic energy in Hartree`, the `md.energies` Ekin column, the history frame and the surface-hopping acceptance test all read that one value. Sampling it earlier left the log printing a number its own `&velocities` block could not reproduce: a round measured 524 steps where the log's Ekin exceeded the record's, always on constrained or thermostatted runs, because projection only removes velocity components.

    # 5. Hop and sync (surfacehop = surface_hopping(sfhp, state)
    #    is initialized once before the loop). What a recorded step is
    #    asked happens in step 6, where the step has a number.
    state = surfacehop(state)
    population = population guard reading (after the hop)
    calculator.sync()

    # 6. Advance the counter, assemble the stop decisions, then write
    #    everything that names this step
    state.runtime.advance_step()
    state.runtime.simulation_time = (step - 1) * dt
    checkpoint(state)          # trajectory frame, log block, runtime file, metadata
    if reasons:                # track / population / err_* guard
        stop after the offending frame
```

Three key semantics:

1. **Compute first, complete the velocity afterwards**: `verlet_i` first propagates coordinates only; after the calculator finishes E/G/NAC/SOC at the new coordinates, `verlet_ii` completes this step's velocity using "previous-step gradient + current gradient"; then `phase_correction`, `apply_constrain`, `constraint_commit`, `velocity_correction` and `apply_ensemble(stage=post)` execute in turn as independent operations, followed by a second `apply_constrain` whenever the post step has rescaled coordinates and cell together (`md.py` phase `constrain_after_cell`), so the geometry the step records is the one the constraints declare. `state.kinetic_energy()` is sampled **last**, immediately before `update_history()`: one sample per step, from the velocities that step ends with -- the same array the frame record recomputes `Ekin` from, the same value the run log prints, and the same value the surface-hopping acceptance test reads after the history is written.

2. **History records the pre-hop state**: `update_history()` is placed before `surface_hopping`; the snapshot stores the pre-hop active state and the complete properties; the checkpoint therefore reads step data from the history snapshot rather than the live config.

3. **Sync after the hop**: `calculator.sync()` is called only after `surface_hopping` has updated `current_state`, so the task configs are refreshed from the state the next step will start on; then `advance_step` and `checkpoint` finish the step. The counter is advanced BEFORE anything is written (#281), because the checkpoint, the trajectory frame and the log block all have to name the step that just finished: a checkpoint taken before the increment carried one fewer, and the resumed process then reused a step number its predecessor had already recorded.

On the first step (empty history), `verlet_i`/`verlet_ii` are no-ops inside the functions, preserving the initial geometry and initial velocity; the main loop has no branch.

During the design phase, external potential and constraint were two separate operations; this refactor unified the geometric half of them into `apply_constrain(state)`, which manages every hard constraint condition (frozen components, strict bond/angle/torsion, and their velocity counterparts). The additive half did not join it: an external-potential-style energy and gradient is applied at the Calculator's result finalization, together with the bias/wall/history potentials, for the reason recorded in §10; state and calculator updates follow PyRAI3MD's PyRAIState functional design (reads via read-only properties / `state.result` / `get_history_step`, writes via `state.update_system()`).

---

## 16. Gradient Descent Mode

In PyRAI2MD:

```python
if graddesc == 1:
    velo = np.zeros(velo.shape)
```

This mode is essentially not ordinary MD, but:

```text
gradient-descent-like nuclear propagation
```

It is therefore not recommended to keep adding large amounts of:

```python
if graddesc:
    ...
```

to the new Verlet algorithm. For example:

```text
NAMD
├── velocity_verlet
└── gradient_descent
```

instead of:

```text
verlet_i()
    └── if gradient_descent
```

**Update (2026-10-08, #314): this mode was ruled out, not deferred.** The `graddesc`
keyword and its schema field are deleted, and no gradient-descent integrator is on the MD
roadmap. Zeroing the velocities every step is not a slower dynamics but a different one, and
the package already owns the tool that question is asked with: `workflow/optimize.py` with the
optimizers under `workflow/optimizer/`. A run that wants stationary structures rather than
trajectories uses `jobtype: optimize`; the input now says so by refusing the spelling.

---

## 17. PBC

PyRAI2MD:

```python
self.traj.apply_pbc()
```

is located at:

```text
verlet_i
    ↓
PBC
    ↓
Calculator
```

This position is reasonable.

Because the Calculator should use:

```text
wrapped / valid Cartesian coordinates
```

to compute electronic structure properties.

Therefore:

```python
state = verlet_i(state)
state = apply_pbc(state)
state = calculator.run(state)
```

PBC itself does not belong to the Verlet mathematical integration.

`apply_pbc` (`data/geometry.py`, shared with the optimization loop) is now implemented
(CHANGELOG #216, PBC stage A/B): it turns the Cartesian coordinates into
fractional coordinates `s = R B^-1`, reduces them modulo 1 on every
periodic axis (`pbc != 0`) and turns them back, so the structure sent to
the backend always sits inside the (possibly deformed) cell. The cell is
built from `lattice_vector` or, failing that, the 6 classic cell
parameters. It is a no-op when the state is incomplete or no axis is
periodic; when at least one axis is periodic, an invalid cell (missing,
all-zero, non-finite or singular) **raises `ValueError`** instead of
silently skipping the wrap. Like every operation it reads and writes the
state exclusively through the official get/update interfaces
(`state.coordinates`, `state.update_system`). The wrap no longer discards
information: the count of cell lengths each atom crossed is committed beside
the wrapped coordinates as `System.image` (PBC stage C), which is what lets a
cross-frame difference, a restart and a backend that keeps periodic images all
reconstruct the same continuous path. The coordinates themselves stay wrapped
by design: bond and contact detection works on the main image through the
minimum-image convention, and a periodic engine handed wrapped coordinates
without their counts evaluates a bond across the whole box.

---

## 18. Unit Handling

PyRAI2MD contains:

```python
* 0.529177249
```

This shows that the old code contained conversions between atomic units and the internal Cartesian coordinate unit.

In PyRAI3MD, unit conversion constants should not be buried directly inside:

```python
verlet_i()
```

It is recommended to explicitly define the internal units of:

```text
coordinates
velocity
gradient
mass
timestep
```

The Verlet algorithm uses only the unified internal units.

For example:

\[
\mathbf R_{n+1}=\mathbf R_n+\mathbf V_n\Delta t-\frac12\frac{\mathbf G_n}{m}\Delta t^2
\]

should not contain:

```python
* 0.529177249
```

or similar implicit unit conversions.

Unit conversion should be done at the Calculator / State interface layer.

---

## 19. API Design

It is recommended that the first version provide only two core functions:

```python
verlet_i(state)
verlet_ii(state)
```

There is no need to create:

```text
VerletManager
VerletIntegrator
VelocityVerletEngine
NuclearPropagationManager
```

or other extra abstractions.

If the code size allows, everything can be placed in:

```text
workflow/
    md.py  main loop code
```

```text
workflow/
    integrator/
        verlet.py  integrator and related functions
```

The first version prioritizes simplicity.

---

## 20. Core Responsibility Boundaries

| Function | Owning module |
|---|---|
| Coordinate integration | `verlet_i` |
| Velocity integration | `verlet_ii` |
| Energy / Gradient | Calculator |
| NAC / SOC | Calculator |
| NAC phase correction | NAMD Workflow / electronic dynamics |
| External potential | independent operation |
| Constraint | `constrain` |
| PBC | independent operation |
| Kinetic energy | State property / utility |
| Velocity correction | independent operation |
| History | `PyRAIState` |
| Surface hopping | Surface hopping algorithm |
| Thermostat | Thermodynamic operation |

---

## 21. Final Design Philosophy

PyRAI3MD should not simply carry PyRAI2MD's `_propagate()` over as is.

PyRAI2MD's:

```python
_propagate()
```

actually took on all of these responsibilities at once:

```text
Verlet
+ PBC
+ Calculator
+ NAC correction
+ external potential
+ freeze atom
+ kinetic energy
+ velocity correction
```

PyRAI3MD should restore it to a data flow:

```text
        ┌──────────────┐
        │   PyRAIState │
        └──────┬───────┘
               │
               ↓
        ┌──────────────┐
        │  verlet_i    │
        │  Rn → Rn+1   │
        └──────┬───────┘
               │
               ↓
             PBC
               │
               ↓
        ┌──────────────┐
        │  Calculator  │
        │   E/G/NAC    │
        │     /SOC     │
        └──────┬───────┘
               │
               ↓
        phase correction
               │
               ↓
        external potential
               │
               ↓
           constraint
               │
               ↓
        ┌──────────────┐
        │  verlet_ii   │
        │  Vn → Vn+1   │
        └──────┬───────┘
               │
               ↓
        kinetic energy
               │
               ↓
       velocity correction
               │
               ↓
       surface hopping
               │
               ↓
       update_history()
```

The most core principle is:

Verlet is only responsible for the time integration of nuclear coordinates and velocities; the Calculator is only responsible for providing electronic structure properties at the current coordinates; other physical operations are inserted into the integration flow as independent Workflow operations; PyRAIState is responsible for passing and saving state between these operations.

This keeps the numerical algorithm already validated in PyRAI2MD while conforming to PyRAI3MD's current "state → operation → state" data-flow architecture, and avoids over-abstracting the NAMD Workflow.

---

## 22. Electronic State Management: root / current_state / activestate and target_states Derivation

In framework-level dynamics (driver=main) the three electronic-state keywords each have their own role. Internal values are 0-based (0..nstate-1); the single external exception is `root`, which is spelled 1-based in every input form (yaml / classic text / agent config, string or typed alike) and decremented once by the adapter:

- `root` (&md): the initial state of the dynamics (external spelling 1-based; the adapter stores the internal 0-based index). The authoritative active state is derivation-populated by `derive_current_state()` in `schema/normalize.py` during the normalize phase (only when `current_state` is the default 0 is `current_state = root` set; an explicitly declared non-default `current_state` is not overwritten); `AIMD.run()` does no runtime seeding and does not modify config parameters before the dynamics;
- `current_state` (&molecule / system): the framework's active-state ledger, recording the electronic state currently occupied. When surface hopping performs a hop, the integrator updates it via `update_system`; it is the only authoritative source;
- `activestate` (&md): the gradient-computation scope switch. 0 = compute gradients of all states; non-zero = compute only the gradient of `current_state`.

The skill backend is never aware of `current_state`: each step the framework derives the `target_states` list from `activestate` and the current `current_state`, and delivers it through the existing `calculator.run(target_states=...)` → `backend.calculate(target_states=...)` parameter channel; the skill only computes for the state list it receives. After a hop, `current_state` has changed; the derivation is recomputed in real time inside each step of the loop, following step by step.

The derivation rule is implemented by `resolve_target_states(activestate, current_state, nstate)` in `workflow/md.py`:

- `activestate == 0` → `list(range(nstate))` (all-state gradients, including all state pairs needed for NAC)
- `activestate != 0` → `[current_state]` (active-state gradient only)
- `current_state` out of range (not within [0, nstate)) → `ValueError`

In dynamics scenarios `target_states` is never None: the None semantics is "do not compute gradients", which conflicts with the requirement that dynamics must obtain a gradient every step.

Supporting conventions: schema `DEFAULT['root']=0` (internal value; the external default spelling is 1); `validate_dynamics` checks the internal root ≥ 0 and its error message reports the external 1-based spelling; in the qualification converter's &MD section, root is derived from the config `current_state + 1` (1 if missing, external spelling), and `activestate` is passed through from the config (1 by default).

Responsibility layering: skill test yaml files (e.g. `casscf_surface_hopping.yaml`) are skill-side files whose root/initial-state derivation is the responsibility of the qualification converter (its &MD root is derived from `current_state`, and &MOLECULE `current_state` is written directly); under driver=skill, the framework &MD root does not participate in execution and has no effect on skill tests.

---

## 23. Checkpoint and Output Files

`AIMD.checkpoint(state)` (`workflow/md.py`) currently executes in the following order:

1. Trajectory output files: appended to the legacy-format `md.*`/`sh.*` files via `MdOutputWriter.record(step, snapshot, state, final)` (`workflow/records/output.py`) (file inventory and formats below); `record` returns whether this step was written (the direct/buffer gating result).
2. Run-log step block: written to the run log from the latest history snapshot (`state.get_history_step(-1)`) via `logger.log_step(step, snapshot)`, subject to the same direct/buffer gating as `md.*`. The section titles it writes are the module constants `print_results.SECTION_ENERGIES` / `SECTION_KINETIC` / `SECTION_VELOCITIES` / `SECTION_CONTRIBUTIONS`, so anything reading the log back after the fact imports them instead of retyping them (a retyped title that quietly stops matching reports "nothing to compare" over a log full of values). The kinetic section is a single `%28.16f` value; a multicentre block additionally prints a `total :` row at 10 decimals, so its summed value carries fewer digits than the per-state `State` lines beside it. The snapshot is used instead of the live pmd config because surface hopping rewrites `current_state` after the step is recorded, while the computed properties of that step remain unchanged.
3. PyRAIState serialization file (`<title>.rt`) via `_write_runtime_file`, when `md.checkpoint` names a cadence; the last step of a run always writes.
4. Runtime metadata recording: written via the official `state.record_step_result()` interface (property data stays in pmd results; the runtime records metadata only).

The frame gate and the file share one predicate (`AIMD._runtime_file_due`): a step that leaves a checkpoint also leaves a trajectory frame, which is what makes the frame list of a segmented run equal to that of the same run in one process (CONSTRAINT_DEVELOPMENT.md #281).

Output file inventory (named by title, in the same directory as `<title>.log`, append mode; before `run()` starts, `MdOutputWriter(title, log_dir=None, direct=2000, buffer=500, dt=20.67, total_steps=0, center_regions=None, charge_output=False)` is created and headers are written; `md.py:533-543` passes the center regions and the charge flag):

```text
title.md.xyz        structure at each output step (Å)
title.md.velo       velocity at each output step (au)
title.md.energies   time / Epot / Ekin / Etot / per-state energies
title.md.pop        time / per-state population (new)
title.md.stress     time / current-state 3×3 stress tensor / pressure (bar) (new, only when stress is available)
title.md.ensemble   time / per-step thermostat & barostat record (new, nvt/npt runs only)
title.sh.xyz        structure only at steps where hopping occurred
title.sh.velo       velocity only at steps where hopping occurred (live post-hop values)
title.sh.energies   energies only at steps where hopping occurred
title.sh.pop        population only at steps where hopping occurred (new)
title.sh.stress     stress/pressure only at steps where hopping occurred (new, only when stress is available)
title.sh.ensemble   ensemble record only at steps where hopping occurred (new, nvt/npt runs only)
title.md.charges    per-atom charges at each output step (only when charge management is on)
title.sh.charges    charges only at steps where hopping occurred (same predicate)
title.md.<c>.*      the per-center set (xyz/velo/energies/pop, plus <c>.charges) written
                    when the run is partitioned over more than one center
```
The suffix matrix is `MdOutputWriter._output_suffixes` (`workflow/records/output.py:241-258`).

The data sources are consistent with the checkpoint specification: coordinates, velocities, energies, and kinetic energy are taken from the current step's history snapshot `state.get_history_step(-1)` (pre-hop, consistent with the configuration at compute time); population is taken from the live value after electronic propagation (the diagonal real part of the live `pmd.system.state_density`). A hop is detected when the live `current_state` differs from the snapshot `current_state`; if one occurred, `sh.*` is written unconditionally (coordinates/energies still use the snapshot — the hop occurred at that geometry and the energies are the pre-hop computed values; velocities use the live values adjusted by fssh; the xyz comment appends `to %d CI`, same as legacy).

Formats (bit-aligned with legacy PyRAI2MD aimd):

- xyz/velo: each frame `%d\n%s\n` + per atom `%-5s%24.16f%24.16f%24.16f`; the comment is `%s coord %d state %d` (title, step number, pre-hop active state 1-based); self-describing, no header.
- energies line: `%20.2f%28.16f%28.16f%28.16f` + per state `%28.16f` (time = (step−1)·dt au; Epot is the pre-hop active-state energy); header `time  Epot  Ekin  Etot  Epot1,2,3...`. **Step 1 records the unpropagated initial condition** — on the first loop iteration `verlet_i` returns unchanged because no history frame exists yet and `verlet_ii` skips with it to keep the two half kicks paired (`integrator/verlet.py`) — so its clock reads 0.00 and frame *k* sits at (k−1)·dt. The first propagated frame is step 2.
- pop line: `%20.2f` + per state `%28.16f`; header `time  Pop1,2,3...`.
- stress line: `%20.2f` (time) + the 9 row-major components of the current-state 3×3 stress tensor `%20.12f` (Hartree/Bohr^3) + pressure P=−Tr/3 `%20.4f` (bar); header `time  sxx sxy sxz syx syy syz szx szy szz  P(bar)`. **`md.stress`/`sh.stress` is created and a line is written only when the backend populated a non-zero stress for that step (lazy header write)**; non-NPT runs do not produce this file and their output is byte-for-byte unchanged; the stress is taken from the current state (for multi-center, the reference state of the whole-system provider), and partitioned runs write only the global stress and produce no per-center stress files.
- ensemble line: `%20.2f` (time) + one `%20.12f` per planned column; the column set is derived once from the dynamics keywords (`MdOutputWriter._ensemble_columns`): a fixed core block (`T_inst(K)` = 2·Ekin/(N_df·k_B), `T_target(K)`, `N_df`, `Ekin(Ha)`, `V(A^3)`), a thermostat block (nh/nhc: `scale_pre`, `scale_post`, `Q1..QM`, `xi1..xiM`, `peta1..petaM`; csvr: `alpha`, `scale_post`, `ratio`, `c`, `r1`, `s`) and a barostat block (berendsen: `Pkin_bar`, `Pvir_bar`, `Pint_bar`, `Pext_bar`, `mu_xx..mu_zz`; pr/mttk: `W`, the same four pressure columns, `geps_*`, `eps_*`, `peps_*`, `veps_*` as nine row-major components in atomic units, `scale_cell_pre/post`; mttk adds `ke_bar`, `scale_bar`, `Qbar1..M`, `xibar1..M`, `petabar1..M`). Pressures are isotropic (Tr/3) conversions of the barostat tensors with the compressive-positive convention (P_vir = −Tr(stress)/3, so P_int = P_kin + P_vir); a missing diagnostic renders as `nan` so the columns stay aligned. **Created only when `dynamics.ensemble` is nvt/npt (lazy header write)**; per-step intermediates come from the `runtime.ensemble` diagnostics (live runtime record, snapshot runtime projection as fallback) and persisted DOF from the snapshot system fields; partitioned runs write only the global files.

direct/buffer output control: `step <= direct` writes every step directly; afterwards every `buffer`-th step is written, counted from the end of the direct window, and the last step (final) is written forcibly. `MdOutputWriter.should_output(step, final)` is a **pure function of the step number** (`step <= direct or (step - direct) % buffer == 0`), not a stateful counter, so a step that produces no record (a missing snapshot) cannot shift the phase of the following ones and a restart re-derives the same frame set. With `direct 1 / buffer 1000` over 10000 steps the written steps are `{1, 1001, 2001, …, 9001, 10000}` — 11 frames at t = 0, 1, …, 9, 9.999 ps. The run-log step block is subject to the same predicate. `sh.*` is not limited by direct/buffer — it is written as soon as a hop occurs.

The `&md` keyword surface is now closed in both directions: every spelling the archive carried
is either read by something or refused by name at the input (`schema/config.py::RETIRE_DYNAMICS_KEYS`
plus the before-validator on `Dynamics`, which covers the classic text, the YAML/JSON sections,
the dotted agent form and a programmatic `Dynamics(...)` at once).
`qualification/verification/test_md_keywords.py` pins that, including against the archive
vocabulary list it carries.

What each family does today:

- output: `direct`, `buffer` (the frame cadence) and `checkpoint` (the `<title>.rt` cadence,
  #274/#281). `record` and `record_step` are refused: what a history frame stores is fixed by
  `HistoryFrame`, and how many frames are kept is `&md history`. `restart` and `addstep` are
  refused: a run is continued by launching its `<title>.rt`, and `step` already means the
  number of steps this process adds.
- initial velocity: `randvelo` draws each Cartesian component from a Maxwell-Boltzmann
  distribution at `temperature` (`sigma = sqrt(k_B T / m)`), the temperature-only start - it
  needs no frequency record, which is what separates it from `initcond: wigner/boltzmann`.
  `excess`, `scale` and `target` then set the kinetic energy of the first step in Hartree,
  measured with `state.kinetic_energy()`; they are three spellings of one request, so
  declaring two is a validation error (`validate_dynamics`). A restarted segment applies
  neither the draw nor the scaling.
- velocity reset: `reset` and `resetstep`, §14.
- ensemble: `ensemble`, `thermostat`, `barostat` and their coupling keywords, plus `dissipate`
  read together with `thermodelay`, §28. The legacy single `thermo` spelling is refused.
- spin-orbit: `dosoc` asks for the couplings and `addsoc` biases them by a constant in cm^-1
  at the Calculator's finalization, so the biased matrix is what the step records; the bias is
  applied only to a run that requested SOC.
- refused as a group: `popreset` (this port reports an out-of-range population and propagates
  the values it found, `integrator/fssh.py::_diagnose_population`), and
  `ref_energy`/`ref_grad`/`ref_nac`/`ref_soc`/`datapath` (a reference calculation is the
  `base.reference` stage of the adaptive chain, not a keyword of `&md`).

---

## 24. surface_hopping jobtype Registration

`surface_hopping` is registered as an independent jobtype in `JOBTYPE_MAP` and `WORKFLOW_REGISTRY` in `workflow/__init__.py`: `'surface_hopping'` → AIMD, mapping to the same AIMD workflow as `'md'`.

The concrete hop algorithm is selected by the `sfhp` keyword of &md (sfhp ∈ nosh/gsh/fssh/lzsh, case-insensitive). `surface_hopping(sfhp, state)` in `integrator/surface_hopping.py` is a factory function: it is initialized once before the trajectory starts and returns a callable algorithm instance, invoked each step inside the loop as `state = surfacehop(state)`:

```text
nosh    no surface hopping (default, pass-through function)
gsh     global surface hopping (GshIntegrator, see GSH_DEVELOPMENT.md)
fssh    fewest switches surface hopping (FsshIntegrator, see FSSH_DEVELOPMENT.md and GSH_DEVELOPMENT.md)
lzsh    Landau-Zener surface hopping (LzshIntegrator, see LZSH_DEVELOPMENT.md)
```

`sfhp` is a framework-exclusive keyword (a member of the qualification flatten `PROTECTED_KEYS`) and is never passed down to the skill; the planning resolver additionally derives extra capabilities from the sfhp+nactype combination (fssh+dcm → overlap, fssh+nac → nac).

---

## 25. Multi-Trajectory Planning (initcond / ninitcond)

The md workflow plans the number of trajectories from the &md `initcond`/`ninitcond` keywords (one trajectory = one initial condition = one dispatched task); `multitask` only decides whether multiprocessing is used for parallelism, and `remotetask` only decides whether jobs are submitted to Slurm (the two-axis orthogonal semantics are unchanged).

`initcond` values (string, normalized to lowercase, unrecognized values raise `ValueError`; the schema registration has been changed from int to str, adapter type `'s'`, the classic numeric 0/1 switch semantics is retired):

```text
0 / none        use the xyz/velo already loaded into PyRAIState (default); when ninitcond > 1, N trajectories repeat the same initial condition
read / load     read initial states from `<title>.init.xyz` in the current working directory according to ninitcond, pushing frames one by one to subtasks; when ninitcond = 1, update the state's xyz/velo in place and run a single trajectory
wigner          Wigner sampling (quantized harmonic oscillator: vibrational-state Boltzmann population draw + Wigner quasi-probability rejection sampling)
boltzmann       Boltzmann sampling (classical-limit harmonic oscillator: Gaussian coordinate/momentum widths)
1               legacy numeric switch compatibility: normalized to wigner
```

`ninitcond` is the number of trajectories (default 1, lowered from 20: the default behavior is fully consistent with the classic single trajectory); `ninitcond < 1` raises an error. The parsing logic is centralized in `workflow/initcond.py` (`normalize_initcond` / `read_init_xyz` / `resolve_initial_conditions`).

`<title>.init.xyz` format: a single multi-frame file, each frame in the standard xyz layout (natom line + comment line + natom atomic lines); an atomic line is element symbol + 6 float columns X Y Z VX VY VZ. Coordinates are read as Angstrom and velocities as Bohr/au, the same units `&system` uses, and `read_init_xyz` performs no conversion between them. The file is named after the run title (`control.title`), the same rule the frequency record `<title>.freq.<format>` follows, which is what lets several examples share one directory without colliding. Fewer frames than ninitcond raises an error naming the file; extra frames are truncated to the first ninitcond.

One shipped ensemble is not a copy of the default state: with `initcond: 0` and `ninitcond: 3` the three trajectories repeat one state byte for byte, which is what a qualification round measured and what made the cross-trajectory "take the worst case" aggregation vacuous. The md dispatch examples therefore carry a five-frame Boltzmann ensemble and use its first three frames.

wigner/boltzmann sampling (implementation centralized in `workflow/sampling.py`, a pure-function module that does not write to disk and does not touch global random state):

* Frequency file convention: `'<title>.freq.<format>'` is located in the current working directory (consistent with the `<title>.init.xyz` convention); title comes from `control.title`, format comes from the &md `format` keyword (value table: molden = Molcas .freq.molden; g16 = Gaussian log; orca = ORCA .hess layout; bagel = BAGEL log). The four readers return a unified freqdata dict (freqs cm-1, vib unnormalized non-mass-weighted Cartesian modes, xyz Bohr, amass g/mol). The g16/orca/bagel readers normalize and un-mass-weight what their engine prints; `read_molden` takes the modes as read because Molcas already writes them unnormalized and unmass-weighted -- which is why the PySCF skill's `save_freq` divides its mass-weighted normalized `norm_mode` by sqrt(atomic mass) before writing `<project>.freq.molden` (through the pure helper `format_freq_molden`), so that one convention serves both producers. There is no PySCF *reader*: `FREQ_READERS` is molden/g16/orca/bagel. Imaginary frequencies: molden takes absolute values with a warning; orca/bagel filter them out with a warning (via the framework logger).
* Sampling semantics: the `temperature` keyword provides the temperature (default 300.0); `control.gl_seed` seeds `np.random.Generator`; sampling is completed once in the root runtime (the ninitcond subtasks share the same sampling result), naturally insensitive to worker scheduling order. The return contract is consistent with read/load (coordinates Angstrom, velocities Bohr/au, shape (natom, 3)), distributed through the existing dispatch chain.
* Intentional deviation from legacy: the new version draws the vibrational state only once per sample (exact Boltzmann weights); legacy wignerfunc redrew the state on every rejection retry, biasing the state mixture by the acceptance rate.

Execution model (`workflow/md.py` `AIMD.run`):

1. Worker guard: when `state.runtime.worker` is true (a dispatched sub-trajectory), run `_run_trajectory(state)` directly, without re-parsing initial conditions and without re-dispatching (recursion prevention).
2. ninitcond = 1: read/load updates the state coordinates/velocities in place and runs a single trajectory in situ, with no directory and no identity, fully consistent with the classic behavior.
3. ninitcond > 1: `resolve_initial_conditions(state)` produces a batch task → `self.dispatch(units=batch, child_jobtype='md')`; each sub state is materialized by deep copy and injected with that frame's coordinates/velocities (`task_system_updates` supports velocities), frozen into a worker graph (calculator node only), with jobtype aligned to md.

Caller-owned starting points (`AIMD(initial_conditions=..., adopt_initial_conditions=True)`): a caller that already owns the starting points (the adaptive chain selecting structures from finished trajectories) bypasses all three `initcond` sources. The switch is constructor-only; no input keyword exposes it. With the switch on, `workflow/initcond.py::adopt_initial_conditions` validates the payload (non-empty; one `{'coordinates', 'velocities'}` mapping per trajectory; `(natom, 3)` shapes; nested lists returned, input untouched) and the batch length is authoritative, overriding `ninitcond`. The single-trajectory branch updates the state with `batch[0]` before the run, the same in-place replacement read/load performs; the multi-trajectory branch uses the unchanged `dispatch(units=batch, child_jobtype='md')`. With the switch off, `resolve_initial_conditions(state)` is called exactly as before; supplying a payload with the switch off raises instead of being silently ignored, and an empty payload with the switch on raises instead of falling back to `<title>.init.xyz`.

Sub working directories and identities: when the batch size > 1, they are planned uniformly by the Dispatcher via `RuntimeDirectoryManager.allocate_indices` (independent of local/remote; existing directories are reused in ascending order, restart-stable); each subtask receives a `'runtime_{idx}'` identity and a subdirectory `runtime/{skill}_runtime_{idx}/` (written to `state.runtime.workdir`). Trajectory output isolation: the directories of MdOutputWriter and the run log are taken from `state.runtime.workdir`, so each trajectory's `md.*`/`sh.*` and `<title>.log` land in its own subdirectory; backend directories carry the same suffix (e.g. `tmp_PYSCF_runtime_3`). Single-task batches (including remote single tasks) behave unchanged.

Legacy keywords: `method` (sampling method) registration is retained; its sampling semantics have been taken over by the `initcond` values; `format`/`temperature` are actually consumed by the sampling (see above). Multi-trajectory result aggregation (average population, etc.) is out of the current scope (collect wraps the N final states as-is into a TrajectorySet, see §26).

gl_seed reproducibility: the `gl_seed` in the control section seeds the global NumPy random number generator when each trajectory enters `run()` (seed = gl_seed + trajectory index; the index is parsed from the trailing digits of the workdir name `'{title}_{idx}'`, 0 for single tasks / no suffix), guaranteeing that the per-trajectory surface hopping sequence is identical across arbitrary runs of the same input; the resolved seed is written to `state.runtime.random_seed` for inspection. This seed is insensitive to worker scheduling order (each trajectory is seeded independently by its own index).

---

## 26. Result Return Mechanism and NN Prediction Deviation Reception (TrajectorySet / err_*)

Underlying generic return architecture: a single-trajectory task is a multi-trajectory task with trajectory count = 1, so both paths of `AIMD.run()` uniformly return a `TrajectorySet` from `workflow/records/trajectory.py`, i.e. an ordered collection of per-trajectory results arranged by planning-order idx.

- `TrajectoryResult`: a single-trajectory record with fields `idx` (planning-order number), `runtime_idx`, `outcome`; the `state` property is returned when `outcome` is a PyRAIState, otherwise None (tolerating Result/exception objects); `history(n)`/`history_step(index)` forward to `state.get_history*`, returning []/None without raising when state is missing.
- `TrajectorySet`: stored in ascending idx; supports `len()`, `[idx]` indexing by planning number, in-order iteration, and a `states` list view. No copying and no aggregation: history data is owned by each trajectory's state; reads are O(1) locating + the existing get_history interfaces. The single-trajectory path returns `TrajectorySet([(0, final_state)])`; the worker path still returns the single-trajectory state directly (returned inside the sub-runtime and wrapped by the parent).
- dispatch/pool.map guarantees outcomes are returned in unit (planning) order; the enumerate position in collect is the trajectory idx.

NN prediction deviation reception (`workflow/records/deviation.py`): the framework does not distinguish skills, does not choose the deviation method (per-component RMS or per-state std are both decided by the future NN skill), and does not compute deviations — it only receives them.

- `DEVIATION_KEYS = err_energy / err_grad / err_nac / err_soc / err_dipole / err_trans_dipole` (legacy naming).
- `fetch_deviations(result)`: keys that exist and are not None are passed through as-is (no shape assumptions); missing keys / None values / None input are filled with the scalar 0.0; no raising, no interruption.
- `merge_deviations(results, deviations)`: fills missing keys only; keys already provided by the backend are not overwritten. `_run_trajectory` calls it after `calculator.run()` at every step; the six keys enter each step's `update_history` snapshot along with pmd.results, so adaptive sampling can read the deviations of the latest trajectory segment.
- `judge_deviations(deviations, thresholds)`: returns `{key: (error, threshold, exceeded)}`, with semantics consistent with legacy `_chkerror` (error > threshold records TRUE; equal does not exceed); when a threshold is missing/non-numeric, exceeded is False. This round it is only for log display and does not trigger termination.

The error part of the run log: the checkpoint calls `logger.log_step(step, snapshot, thresholds=...)`, passing in the &adaptive threshold mapping (err_energy→maxenergy, err_grad→maxgrad, err_nac→maxnac, err_soc→maxsoc; err_dipole/err_trans_dipole have no input thresholds yet, so only values are listed without judgment). `format_result_block` of `print_results` gains an optional `thresholds` parameter; when any `err_*` key is present, a `&prediction error` block is output after the SOC section, each line being deviation key + value + threshold + `Exceeded: TRUE/FALSE`; when no deviation keys are present the block is omitted (non-MD single-step calculations are unaffected).

---

## 27. Per-Step Cost Accounting (the two-level phase table)

The loop above is timed, not estimated. `utilities/timer.py::PhaseProfiler`
accumulates wall time per named phase in the shape `workflow/integrator/fssh_profile.py`
already used for the FSSH propagator, and `AIMD._run_trajectory` attaches one to
its own loop: every action of §15 gets a `with profiler.measure('<phase>')`, the
checkpoint's four actions get theirs through `AIMD._phase`, and the loop's total is
stamped as a `# wall` comment row so the reader can see what fraction of the step
the rows account for. The table is refreshed every `PHASE_EVERY = 100` steps into
`pyrai3md_step.perf` beside the run log, and once more at the end of a trajectory.

Two rules keep the table honest, both of them learned the expensive way:

- **Rows never nest inside their own total.** A component that is timed inside one
  row renders its breakdown under a `# nested breakdown (inside the rows above)`
  header (`Calculator.profile_lines`, appended by `_nested_profile`), so the
  milliseconds the force evaluation spends are not added a second time to the step
  row that contains it.
- **Coverage is checked, not assumed.** `test_workflow.py::TestMdPhaseProfile`
  runs a mock 60-atom trajectory and asserts the summed phases stay within 5% of
  the loop wall, that each of the nineteen loop phase names was recorded once per step,
  that the file exists without any shutdown having happened, and that the cost per
  step at 5000 steps is not more than 1.5 times the cost at 1000 - a per-step cost
  that grows with the length of the run is an accumulation bug, and it shows up
  nowhere else.

Why the loop is timed at all: in the test environment the engine reported 1.47 ms per
step while the framework measured 8.39 ms, and the difference between those two
numbers is the whole question. Priced this way, the answer was not the integration
mathematics (the four propagator rows - `verlet_i`, `verlet_ii`, `apply_pbc`,
`velocity_correction` - came to 2.04 ms together, and 1.63 ms of that was two
deep copies of the previous history frame rather than any arithmetic) but a
`manifest.yaml` re-read and re-parsed once per step inside `Calculator.run()`
(2.42 ms) plus a third such copy in the checkpoint (0.51 ms). Removing the
per-step manifest read and letting the four read-only consumers of the frame take
it by reference took the step from 8.39 to 4.20 ms with the `md.*` records
byte-identical. See CHANGELOG #244 and
`dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` §11.12 for the measured tables and for
the candidates the same measurement ruled out.

---

## 28. What a step is asked after it is recorded (#314)

The integrator used to reserve a slot for an operation called `verification`: after surface
hopping and before the checkpoint, it would check the consistency of the recorded step and flag
anomalies without altering the dynamics. The slot was never filled, and by #310 the question
had already been split across three owners that do the job with a rule the placeholder never
had:

| Check | Where | What it watches |
|---|---|---|
| non-finite extended coordinates | `records/diagnostics.py::check_finite`, called from `integrator/verlet.py` and `integrator/ensemble.py` | a lattice, strain, or chain coordinate that stopped being a number; fatal in every mode |
| cell divergence | `integrator/verlet.py::_divergence_report` | one commit rotating the frame by >0.05 rad or stretching an axis by >1.6 (fatal), and a box that has left its starting size by e^0.405 (interrupts only under `control.debug`) |
| NN prediction deviation | `records/deviation.py::judge_deviations` | the six `err_*` keys against the `&adaptive` thresholds, printed in the run log |

Three questions were left over, all of them about a frame the run has already written, and
#314 put them where a finding already has two outlets - the runtime record and the log. The
loop no longer has a `verification` phase; `AIMD.checkpoint` asks these on the steps the
trajectory record itself keeps (`output.record_due`, the same predicate `diagnostics` uses to
decide when to speak), inside the `output_record` row of the phase table:
- `kinetic_identity`: the frame's recorded `Ekin` against the one its own velocities and masses
  give, relative tolerance `1e-12`. This is the shape the #310 defect had - a step that sampled
  the kinetic energy on either side of an action that rewrites the velocities - and a
  recomputation from the same arrays is the only check that sees it.
- `image_integrity`: the periodic crossing counts of a periodic frame must be finite, of shape
  `(natom, 3)` and integral. They are what lets the continuous view, a restart and a periodic
  engine agree on which copy of an atom is meant.
- `constraint_geometry`: `integrator/operations.py::verify_recorded` measures the recorded
  geometry against the plan's own targets, through `constraints/controller.py::measure_recorded`
  (bond sets measured with the vectorized `shake.distances` the solver uses). A periodic frame
  is asked in the continuous view rebuilt from its own image counts, because a bond straddling
  a box face otherwise reads as violated when the reader's picture is simply the main image.
- `region_reconstruction` (#316): for a periodic run with more than one center, every pair of
  atoms a center owns must rebuild to less than half the shortest cell extent from the frame's
  own coordinates and crossing counts, and a pair that rebuilds longer is a finding when the
  center would perceive it as a bond. This is the fourth question because it is the same class:
  a center handed two fragments instead of one molecule is a bookkeeping failure of the frame,
  and the failure shows up as physics in an engine rather than as an error. Its semantics, the
  order the two readings are asked in, and the measured cost are in
  `dev/methods/PBC_DEVELOPMENT.md` section 4.2.

None of the four is fatal on its own: a bad frame is still a trajectory, so the verdict goes
through `diagnostics.report` and stops the run only under `control.debug` (raised as
`RecordedStepError`, the module's convention that the caller who knows whether the state is
still consistent is the one who raises).

Measured cost, so that "on the recorded steps" is a bounded claim rather than a promise: on a
60-atom frame the two pure readings cost 33.8 us per recorded step, and the same box with 40
strict bond constraints costs 133.9 us - against the 4.20 ms/step the deployment measures for a
full step, roughly three per cent of one step, paid only on a step the run writes. A paired
bench on the mock loop (15 alternating pairs, checks installed and uninstalled) put the
per-step wall difference at 0.009 +- 0.030 ms, i.e. below what that instrument can resolve,
which is why the direct measurement is the one quoted.

The fourth reading costs what the shape of the run makes it cost, measured on the same 60-atom
box: 0.4 us on a non-periodic frame and 0.5 us on a single-center periodic one, which is every
example that shipped before #316; 84 us with two centers and 107 us with twenty on a box of
whole molecules; 408 us in the shape that has to ask the nearest-image question at all, a center
whose own atoms sit scattered across the box. The early return is not a shortcut around the
work: a run with one center cuts no region, so there is nothing for the check to certify.

The 84 us shape is no longer only a bench. Since #317 the two `multicenter_pbc_*` examples
carry two centers inside the 60-atom box with LAMMPS as the low layer, because the low layer
can now be handed a center slice at all (dev/skills/LAMMPS_SKILL_DEVELOPMENT.md section 5.6a),
so the watch has run at C>1 against real engines - 4 recorded frames on a fixed cell and 6 on a
deforming one, `control.debug` on, no finding in either. And against those runs the reference
frame for "is this cheap?" moves: `pyrai3md_step.perf` reads 186.9 ms/step for the NVE pair and
107.0 ms/step for the NPT one with three LAMMPS evaluations and two PySCF ones per step, so the
tens of microseconds the watch spends are four significant figures down the page - which is
exactly what the 4.20 ms/step comparison was already saying, now on a run that cuts regions at
more than one center.

## 14. Resident Deviation Stop

The trajectory loop carries a resident uncertainty guard, not an adaptive-only
one: after a step's counter advances and before its checkpoint, `AIMD` judges the
step's `err_*` values against the `&adaptive` `max*` thresholds
(`workflow/md.py::_stop_deviations`, reusing `records/deviation.py::judge_deviations`).
Any `error > threshold` ends that trajectory after the offending frame has been
written. Equality does not stop.

The guard reads the run's own thresholds (`&md md_max*`), not `&adaptive`.
Empty thresholds leave every channel unjudged, so a QM md run never trips it.
An adaptive md child has the four values overwritten from `&adaptive` by the
parent workflow (see section 29). A direct NN md run that declares no
`md_max*` therefore has no early stop (the #468 behavior change), while its
`err_*` values are still printed in the step log. The stop is reported in the
run log, and the stopping frame is forced into the trajectory record even when
it is not on the `direct`/`buffer` cadence.

The planned step count is still `&md step`; the trajectory is "completed" for
adaptive sampling when its last recorded step reaches that number, otherwise it
stopped early. `runtime.status` is not the criterion (the trajectory still marks
itself completed).

## 29. Detection Thresholds and the Adaptive Decoupling (2026-11-08)

`&md` carries four detection thresholds:

```text
md_maxenergy   default None   Hartree
md_maxgrad     default None   Hartree/Bohr
md_maxnac      default None   Hartree/Bohr
md_maxsoc      default None   cm-1
```

`None` (the default) leaves the channel unjudged. The three routes are:

| route | threshold source |
|---|---|
| QM md | none; the backend reports no `err_*` |
| plain NN md | `&md md_max*`; absent means no early stop |
| adaptive md child | `&adaptive` `max*`, overwriting `md_max*` on the child |

`AIMD` takes the thresholds as constructor arguments
(`deviation_thresholds`); `_stop_deviations` returns `[]` when the mapping is
empty. The two call sites (the trajectory guard and the `&prediction error`
step log) read the same instance mapping, so a run cannot stop on one set and
print another.

The exploration bias (UDD / gap, `dev/methods/ADAPTIVE_DEVELOPMENT.md section 16`) keeps the
unbiased evaluation under `unbiased_energy` / `unbiased_gradient` /
`unbiased_nac`. FSSH, GSH and LZSH read those for every hopping decision, and
the adaptive screening and trace read the unbiased energies; only the nuclear
propagation sees the biased force.

## 30. Tracked Geometry and Population Bounds (2026-11-12)

Two more residents end a trajectory on a physical reading rather than on a
surrogate prediction. Both are ported from the legacy `AIMD._chkgeom` /
`_chkpopulation`, and both share the deviation guard's exit path: the reading
is taken inside the step, the decision is assembled after `advance_step`, the
offending frame is checkpointed as the segment's final frame, and only then does
the loop break. `runtime.status` still ends as completed.

### 30.1 Geometry tracking (`&system`, legacy `&molecule`)

The four legacy keywords are read from the system section and validated by
`utilities/tracker.py::plan`, which the `System` model calls at load time:

```text
track_type    None | dist | frag
track_index   1-based atom groups
track_thrhd   one bound per tracked object, Angstrom
track_stop    0 report only, 1 end the trajectory
```

- `dist` compares each pair (`track_index` group of exactly two atoms) against
  its bound; `frag` compares the centers of exactly two groups, unweighted.
- The comparison is `distance > threshold`; equality does not trigger.
- A threshold list shorter than the object list repeats its last value, and
  extra thresholds are not read, which is the legacy reader's rule.
- **The unit is Angstrom**, the unit of the stored coordinates, not the legacy
  internal Bohr. A declaration migrated from PyRAI2MD has to convert its
  numbers once.
- Atom numbers are 1-based in every external spelling (yaml and classic) and
  are decremented once by the adapter. A typed yaml `track_index` goes through
  the same decrement as a classic `track_index 1 2, 3 4`; the parity test is
  `test_md_track.py::TestTrackInputParity`.
- PBC runs are measured on the continuous coordinates
  (`data/geometry.py::continuous_coordinates`), so a pair that straddles a face
  is not read as one cell length.
- `track_stop 2` (the legacy "suppress surface hopping while out of bounds",
  which collapsed the density to a pure state) is not implemented and is
  refused at load time.
- The per-step reading rides `results['geometry_track']` with the frame and is
  printed as `&geometric parameter monitor` on the record cadence (the first
  `direct` steps, then every `buffer`-th, and always on the stopping step).

### 30.2 Population bound (`&md md_maxpop`)

```text
md_maxpop   default 0.01   a population outside [-t, 1 + t] ends the run
            None           the guard is off
```

- The reading is the real diagonal of `system.state_density` after this step's
  electronic propagation and hop decision; a run without a density (`nosh`)
  has nothing to judge and passes.
- A non-finite population also stops the run. The predicate is
  `workflow/records/population.py::population_bad`, the same one the adaptive
  screen uses for its own reporting, so "ill population" means one thing.
- The bound is checked after the hop, while the recorded frame carries the
  pre-hop density (the existing frame convention); the stop sentence reports
  the post-hop value it judged. The stopping structure is the last recorded
  frame.
- **This changes plain FSSH/GSH/LZSH runs**: the 0.01 default was already the
  legacy bound, but the framework previously reported an out-of-range
  population and propagated it (`popreset` stays retired). A run that wants
  that behaviour back writes `md_maxpop none` (or sets it to a larger value).

### 30.3 The stop code returned to the caller

`AIMD` stamps `runtime.stop_code` on the trajectory it returns, using the
legacy early-stop categories extended with a normal code:

```text
0  ran to its planned end
1  ill population (outside md_maxpop, or non-finite)   (legacy 3 and 4)
2  dissociation: a tracked geometry bound was crossed   (legacy 5)
3  prediction error above the md_max* thresholds        (legacy 1)
4  the calculation failed                               (legacy 2)
```

A step that violates more than one guard reports the first code in that order
while the run-log sentence still names every reason. Code 4 is stamped by the
`calculator_run` phase before the exception leaves the loop: the run still
raises (a step without a property is not a completed trajectory), and the
adaptive child entry carries the code on its error marker. The code is part of
`RuntimeState.to_dict`, so it reaches a remote child's `result.json`; files
written before the field existed read as 0.

The run-log tail names a non-zero code on the second-to-last line, immediately
above the `PyRAI3MD End:` line, in every case: a returned early stop and a
raised failure both print `MD stop code: N (<meaning>)` (`kernel/logger.py::
stop_code_note`). Code 0 adds nothing, so a complete trajectory keeps the
previous last two lines. The adaptive child's own log gets the same line
through `_child_entry`.

`&adaptive check_population` is the only adaptive-side switch left: with it on
(1, the default) the screen collects the last frame of a trajectory whose code
is 1 and sends that structure to the reference calculation; with it off (0)
the structure is not collected, but the child still stops, because the guard
is md's. The screen does not judge any frame's population. `&adaptive
pop_tolerance` is retired and points at `&md md_maxpop`.

The three stop sentences are joined into one `MD stopped at step N: ...` line,
naming each violated bound with its number; a run whose only violation is an
`err_*` channel keeps the previous wording exactly.

