# PyRAI3MD LZSH (Landau-Zener) Surface Hopping Development

Last updated: 2026-10-22 22:20 (section 11's factory sentence matches `surface_hopping.py:79-80`, and section 7's mangled sentence about `tsh_helper.adjust_velo_isotropic` was repaired so the helper name, the Ekin source and the pairwise A/B reading each sit in one piece. The rest was not re-audited. Before it, 2026-10-07 13:05)

---

> **Multi-center dispatch (refactor R5, 2026-09-08).** The LZSH (Landau-Zener)
> core in `workflow/integrator/lzsh.py` is UNCHANGED; multi-center runs reach it
> through `workflow/integrator/electronic.py`'s `ElectronicStepper` (one
> `LzshIntegrator` per center on a `CenterView` proxy). LZSH rescales the
> velocity isotropically on a hop (a global rescale), so the binding's
> region-masked velocity write-back is what keeps one center's hop from touching
> another center's velocity. C=1 is a transparent binding, bit-for-bit the legacy
> path.

# 1. Objective

Add a Landau-Zener surface hopping (LZSH) algorithm to the PyRAI3MD 3.0 hopping layer, alongside FSSH and GSH:

- Replace the Zhu-Nakamura improved Landau-Zener probability of GSH with the textbook single-passage Landau-Zener formula (L. Landau, Phys. Z. Sowjetunion 1932, 2, 46; C. Zener, Proc. R. Soc. Lond. A 1932, 137, 696).
- Derive the diabatic force difference from the recorded energy gap series instead of rebuilding diabatic forces from trajectory gradients: the LZSH kernels consume the state energies and the timestep only (no gradients, no NAC vectors).
- Representations differ by process: internal conversion works in the adiabatic (NAC) representation (the recorded gap is the avoided-crossing gap — gap curvature carries the force difference); intersystem crossing works in the spin-diabatic representation (the S/T energies carry no SOC, so the recorded gap is the spin-diabatic gap itself — gap slope carries the force difference).
- Keep the GSH conventions: both probabilities are evaluated at t-dt (the nuclear propagation lags the electronic properties by one step); internal conversion hops with P_LZ, intersystem crossing hops with 1 - P_LZ.
- Pure NumPy single implementation; reuse the shared infrastructure of the hopping layer: the `surface_hopping` factory entry and the `tsh_helper.py` velocity adjustment helpers (both defined in `FSSH_DEVELOPMENT.md`).

---

# 2. Input Configuration

LZSH introduces no new keywords; it reuses the existing &md surface hopping keywords:

```
maxh / adjust / reflect / gap / gapsoc / verbose
```

- `gap` (eV) is the internal-conversion crossing threshold; `gapsoc` (eV) the intersystem-crossing threshold; both convert to Hartree at initialization.
- `adjust` / `reflect` select the velocity adjustment policy of a hop (shared with FSSH/GSH through `tsh_helper`).
- `maxh` bounds the highest state a hop may target.

Data precondition: **state energies only**. Unlike GSH, LZSH does not need all-state gradients; the kernels never read `results['gradient']` or `results['nac']`, so the `activestate` keyword is irrelevant to LZSH. The intersystem-crossing leg additionally reads the previous-step SOC values from `results['soc']` (cm-1).

---

# 3. File Structure

```
pyrai3md/
└── workflow/
    └── integrator/
        ├── lzsh.py             (LzshIntegrator main flow, Landau-Zener kernels)
        ├── tsh_helper.py       (shared velocity adjustment, see FSSH_DEVELOPMENT.md)
        └── surface_hopping.py  (factory entry, sfhp: lzsh branch)
```

lzsh.py

Responsible for:

- static configuration parsing (`nstate / maxh / adjust / reflect / gap / gapsoc / state_spin / soc_pair_map`)
- internal conversion probability (gap-curvature kernel)
- intersystem crossing probability (gap-slope kernel)
- energy-ordered cumulative sampling and the hop decision
- pure-state density rebuild
- velocity adjustment calls into `tsh_helper`

The snapshot/results readers (`_snapshot_results / _snapshot_system / _energy_of`) are small module-local copies of the gsh.py readers (identical semantics).

`run_lzsh(state)` is kept as a thin wrapper around a disposable `LzshIntegrator` instance for standalone use; the workflow uses the factory path.

---

# 4. Internal Conversion (Gap Curvature)

The adiabatic representation records the avoided-crossing gap

```
delta(q) = sqrt(df^2*q^2 + 4C^2)        (df = force difference, C = coupling)
```

with the minimum de = 2C at the crossing and the curvature delta''(0) = df^2/de. The kernel consumes the three-step gap series at t, t-dt, t-2dt:

1. Three-step gate (GSH semantics): the gap at t-dt must be the smallest of the three steps (`argmin == 1`), below the `gap` threshold, and the hop must be classically allowed (previous-step total energy above the crossing energy).
2. Gap curvature at the gate step: the three-point second time derivative `gddot = (g_t - 2*g_{t-dt} + g_{t-2dt})/dt^2 = v_c^2*df^2/de` carries the force difference times the squared crossing velocity.
3. Classical Landau-Zener with velocity cancellation: with V12 = de/2 and df = sqrt(de*gddot)/v_c the crossing velocity cancels,

```
gamma = V12^2/(v_c*df) = de^2/(4*sqrt(de*gddot))
P_hop = exp(-2*pi*gamma)
```

A non-convex gap window (`gddot <= 0`, noise or a flat gap) degrades to zero probability. For the textbook Landau-Zener model with constant diabatic slopes the quadratic gap approximation at the symmetric sampling points reproduces the analytic probability (verified numerically: the lz1 middle-probability parameter set gives P ~ 0.377, matching the analytic P_LZ exactly).

---

# 5. Intersystem Crossing (Gap Slope)

The S and T states are spin-diabatic: their computed energies carry no SOC, so the recorded gap is the spin-diabatic gap itself — a linear zero crossing for crossing surfaces, a constant for parallel ones. The gap has **no avoided curvature**: the curvature form of §4 diverges as the gap minimum goes to zero and is unusable. The kernel instead consumes the gap rate of change:

1. No minimum-gap gate: the spin-diabatic gap has no avoided minimum to detect. Only two constraints apply, both evaluated at t-dt (the same evaluation moment as the internal conversion kernel): the absolute gap below the `gapsoc` threshold and the hop classically allowed (previous-step total energy above the target-state energy, `ex = ep[i]`, the GSH convention).
2. Signed-gap central difference: `gdot = |deltaE(t) - deltaE(t-2dt)|/(2*dt)` over the signed gap (the absolute value applies to the threshold only). For a linear crossing the signed gap is linear in time, so the central difference is exact at every crossing position — including the crossing landing exactly on the t-dt sample and landing inside either sub-interval. The absolute-value gap, by contrast, folds a cusp at the crossing where the central difference collapses to zero; it must never be differenced.
3. Classical Landau-Zener with velocity cancellation: `gdot = v_c*df` and V12 = SOC (cm-1 -> Hartree, the previous-step `results['soc']` value like GSH), so

```
gamma = SOC^2/gdot
P_hop = 1 - exp(-2*pi*gamma)
```

The non-parallel PES assumption turns the Landau-Zener survival into a hop (the GSH convention); the weak-coupling limit `2*pi*gamma` recovers the perturbative result. A parallel gap (gdot == 0) carries no crossing and degrades to zero probability; an undeclared or uncomputed SOC pair degrades to zero probability (the GSH convention).

---

# 6. Pure-State Density Rebuild

LZSH keeps the GSH density semantics:

- The density matrix is rebuilt every step as the pure state of the final active state.
- There is no coherent propagation and no decoherence term.
- Consequently the electronic population block and the active-state block are trajectory-wise identical.

---

# 7. Velocity Adjustment

Shared with FSSH/GSH through `tsh_helper.py`:

- The LZSH kernels produce no coupling vector (no NAC), so every hop — internal conversion and intersystem crossing alike — takes the isotropic rescaling policy through `tsh_helper.adjust_velo_isotropic` (the missing-NAC policy of `FSSH_DEVELOPMENT.md` §18, the same policy GSH uses for its intersystem-crossing leg). For strictly one-dimensional model trajectories the isotropic rescaling lies along the single direction of motion and is exact. Its `Ekin` is `state.runtime.kinetic_energy`, sampled once per step after the ensemble post half step and the second constraint projection (#310), so the acceptance test and the recorded step read the same velocities; on the unconstrained-NVE calibration legs that move changed no population (verified pairwise against the pre-change code: fssh 20, gsh 12, lzsh 12, ktdc 12 trajectories, `md.pop` digests bit-identical).
- Insufficient kinetic energy is treated as frustrated hopping; the state and the density stay on the departure state (the GSH convention).

---

# 8. Numerical Protection (Phase 6 Style)

Every degenerate input degrades to zero probability; nothing is clipped and nothing is reset:

- Non-convex IC gap window (`gddot <= 0`): noise, a flat gap or a crossing outside the gate.
- Parallel or frozen spin-diabatic gap (`gdot <= 0`): no crossing to evaluate.
- Zero or negative gap, SOC coupling or timestep; missing energy rows.
- Undeclared or uncomputed SOC pairs.

---

# 9. Legacy Bug Fixes

LZSH is a new implementation without a legacy counterpart; it inherits the already-fixed behavior of the GSH port (hop velocity through the shared `tsh_helper` policy, frustrated-hop state/density retention) and introduces no new bug class.

---

# 10. Hopping Sampling

- All allowed pairs (internal conversion plus intersystem crossing) accumulate in energy order, mirroring the FSSH/GSH cumulative sampling gate.
- One `np.random.uniform` draw per step decides the hop.
- `maxh` bounds the target state; a hop above it is rejected.
- A successful hop updates the authoritative active-state record through `state.update_system({'current_state': hop})`; the workflow only chains the returned state.

---

# 11. PyRAI3MD Integration

LZSH follows the factory-style integrator design defined in `FSSH_DEVELOPMENT.md`:

1. **Initialization timing**: `AIMD.run()` calls `surfacehop = surface_hopping(self.sfhp, state)` once before the loop; `sfhp: lzsh` returns the dispatch layer `ElectronicStepper('lzsh', state)`, which binds one `LzshIntegrator` per center (a single transparent binding at C=1; `surface_hopping.py:79-80`), and the loop calls `state = surfacehop(state)`.
2. **Data ownership**: the instance holds only parsed static parameters (thresholds in atomic units, the spin list, the SOC pair map) and **never caches any trajectory data**. Every step reads through the official PyRAIState interfaces (`state.result`, the read-only history views, `system.state_spin`, `system.soc_pair_map`) and writes back through `update_system/update_dynamics`; restart semantics and serialization behavior are unchanged.
3. **Warm-up**: with fewer than 3 history steps the integrator only seeds the pure-state density and performs no hop decision (mirroring FSSH/GSH); the kernels read t-dt and t-2dt one index deeper than the raw offset because the md loop records the current step into the history before the hopping decision.
4. **Energy-only chain**: no all-state gradient requirement — calculators may resolve only the active state plus the hop candidates the workflow requests; the kernels degrade pairs with missing energies to zero probability.

---

# 12. Benchmark Calibration

The qualification benchmark pillar ships an LZSH calibration (`qualification/benchmark/lzsh.py` + `lzsh.yaml`, see the lzsh section of `QUALIFICATION_DEVELOPMENT.md`) and the dedicated Landau-Zener family (`lzsh_lz1.yaml`, `lzsh_lz2.yaml`, `lzsh_lz3.yaml`):

- The case reuses the algorithm-independent fssh pieces (`load_case / apply_overrides / build_reference_input / collect_populations / ensemble_average`), so LZSH, GSH and FSSH calibrate against the identical exact split-operator quantum reference.
- The LZSH production input owns a dedicated keyword forwarding table (`sfhp: lzsh` plus the present-only `gap / gapsoc / adjust / reflect / maxh` entries); FSSH-only keywords fail the unrecognized-keyword guard, and the GSH-only `activestate` entry is absent (the kernels do not consume gradients). Model parameter overrides (`lz_a / lz_b`) forward into the qdm section of both legs (present-only).
- The tully verdict is the unified final-state metric set shared with the fssh evaluator (`evaluate_unified`): the report carries the final trajectory / state / reference populations, the deterministic LZSH kernel replay of the recorded trajectories (`internal_hopping_probability`) and two judged differences — `state_reference_difference` (the ensemble-averaged final S1 population against the exact wavepacket reference final probability) and `state_internal_difference` (against the replayed probability). The hop statistic stays a clean Bernoulli count because of the pure-state rebuild (§6) and is reported as `final_trajectory_population`; the tully surfaces carry no single-passage crossing of a fixed gate, so the replay (`_max_lzsh_kernel_probability`) reports the maximum per-step probability of `_replay_lzsh_kernel` instead of a gate value. No analytic hopping probability exists for the tully models, so that comparison is skipped. The shipped tolerances are placeholders to be calibrated from the benchmark results.
- **The lz family** (lz1/lz2/lz3) calibrates against the Landau-Zener crossing models (constant diabatic slopes, constant coupling — the only model family where the gap-curvature derivation of §4 holds exactly; the Tully models violate the premise through exponentially decaying diabatic slopes). Within the unified metric set the lz cases judge three differences against the ensemble-averaged final S1 population: `state_reference_difference` (the exact wavepacket reference final probability), `state_analytical_difference` (`P_LZ` with `v_c` from energy conservation) and `state_internal_difference` (the LZSH gate probability replayed from the recorded trajectory energy snapshots through the production kernel — the first non-zero per-step probability of `_replay_lzsh_kernel` (`_gate_lzsh_kernel_probability`), reported as `internal_hopping_probability`; simpler than the GSH replay, which additionally needs the model adiabatic gradients). The slope difference at the crossing is 2·lz_a for lz1 and lz_a for the half-linear lz2/lz3, so their exponent doubles at equal velocity: `P = exp(-pi*lz_b^2/(lz_a*v_c))` vs `P = exp(-2*pi*lz_b^2/(lz_a*v_c))`. All shipped tolerances are 1.0 placeholders pending calibration.
- The lz calibrations share the gsh parametrization `lz_a = 0.2712, lz_b = 0.0543` (gap minimum 2·lz_b = 2.95 eV, the gap gate pinned at 3.5 eV) and the identical phase-aligned initial conditions, so LZSH and GSH run the identical problem with the same three-step IC gate. The pmd_env entries ship disabled until the scan confirms the verdicts; every calibration entry runs the unified 100-trajectory ensemble on a 4-process pool.

---

# 13. Testing

`qualification/verification/test_lzsh.py`:

- IC kernel hand-computed reference and the "quadratic gap reproduces the analytic Landau-Zener probability" case (the lz1 parameter set, P ~ 0.377);
- IC guards (`gddot <= 0`, missing snapshots, classically forbidden hop, gap-threshold rejection, gate-step minimum requirement);
- ISC gap-slope kernel hand-computed reference (linear crossing gap): the `1 - exp(-2*pi*SOC^2/gdot)` form, the weak-coupling perturbative limit, exactness at every crossing position (the crossing exactly at t-dt and inside either sub-interval), the t-dt constraints (`|deltaE_{t-dt}| <= gapsoc` and energy conservation) with no minimum-gap gate, and the parallel-gap `gdot == 0` degradation;
- ISC independence from the curvature form (finite and correct as the gap minimum goes to zero);
- SOC-missing degradation (undeclared pair, uncomputed value), warm-up seeding and the missing-energy pass-through;
- the history-window offset contract (t-dt / t-2dt one index deeper than the raw offset);
- pure-state density rebuild, energy-ordered cumulative sampling, the `maxh` constraint and the frustrated-hop fallback;
- factory dispatch of `sfhp: lzsh`.

The calibration evaluation (the dual-family verdict: the wavepacket reference and LZSH kernel replay differences for the tully cases, the additional LZ-analytic verdict shared with fssh/gsh for the lz family, the gate-step and maximum-value replay wrappers, the wavepacket reference plotting) is covered by `qualification/verification/test_benchmark.py`.
