# PyRAI3MD GSH (Zhu-Nakamura) Global Surface Hopping Development

Last updated: 2026-10-22 22:20 (section 11's factory sentence now returns `ElectronicStepper` with one `GshIntegrator` bound per center, and the test count is the 24 cases `test_gsh.py` declares. The rest was not re-audited. Before it, 2026-10-07 13:05)

---

> **Multi-center dispatch (refactor R5, 2026-09-08).** The GSH (Zhu-Nakamura)
> core in `workflow/integrator/gsh.py` is UNCHANGED; multi-center runs reach it
> through `workflow/integrator/electronic.py`'s `ElectronicStepper` (one
> `GshIntegrator` per center on a `CenterView` proxy). GSH reads coordinates
> strictly as differences (r - rpp, rp - r), so the center coordinate zero-mask
> makes every out-of-region displacement vanish and the coupling localizes to the
> center, exactly like the zero-masked gradient/NAC; the intersystem-crossing
> branch uses no coordinates. Each center's velocity write-back is masked to its
> own atoms. C=1 is a transparent binding, bit-for-bit the legacy path.

# 1. Objective

Port the legacy `PyRAI2MD/Dynamics/Propagators/gsh.py` global (generalized) surface hopping implementation into PyRAI3MD 3.0:

- Keep the existing Zhu-Nakamura algorithm and physical model.
- Internal conversion follows C. Zhu, Phys. Chem. Chem. Phys. 2014, 16, 25883-25895 (EQ 1-8).
- Intersystem crossing follows the 2020 adaptation, C. Zhu, Phys. Chem. Chem. Phys. 2020, 22, 11440-11451.
- Fix the two legacy bugs (hop velocity direction, frustrated-hop density) without changing any valid behavior.
- Harden numerical protection in the Phase 6 style defined by `FSSH_DEVELOPMENT.md` (degrade to zero probability, no clip, no reset).
- Pure NumPy single implementation: GSH has no electronic-substep hot loop, so the NumPy/Numba dual backend of FSSH is not introduced.
- Reuse the shared infrastructure of the hopping layer: the `surface_hopping` factory entry and the `tsh_helper.py` velocity adjustment helpers (both defined in `FSSH_DEVELOPMENT.md`).

---

# 2. Input Configuration

GSH introduces no new keywords; it reuses the existing &md surface hopping keywords:

```
maxh / adjust / reflect / gap / gapsoc / verbose
```

- `gap` (eV) is the internal-conversion crossing threshold; `gapsoc` (eV) the intersystem-crossing threshold; both convert to Hartree at initialization.
- `adjust` / `reflect` select the velocity adjustment policy of a hop (shared with FSSH through `tsh_helper`).
- `maxh` bounds the highest state a hop may target.

Data precondition: internal conversion needs **all-state gradients**. The input keeps `activestate` at the schema default 0, so the calculator resolves all target states and stores `results['gradient']` as a `(nstate, natom, 3)` array; pairs with missing gradient rows degrade gracefully to zero probability.

---

# 3. File Structure

```
pyrai3md/
└── workflow/
    └── integrator/
        ├── gsh.py              (GshIntegrator main flow, Zhu-Nakamura kernels)
        ├── tsh_helper.py       (shared velocity adjustment, see FSSH_DEVELOPMENT.md)
        └── surface_hopping.py  (factory entry, sfhp: gsh branch)
```

gsh.py

Responsible for:

- static configuration parsing (`nstate / maxh / adjust / reflect / gap / gapsoc / state_spin / soc_pair_map`)
- internal conversion probability (EQ 1-8)
- intersystem crossing probability (the 2020 adaptation)
- energy-ordered cumulative sampling and the hop decision
- pure-state density rebuild
- velocity adjustment calls into `tsh_helper`

`run_gsh(state)` is kept as a thin wrapper around a disposable `GshIntegrator` instance for standalone use; the workflow uses the factory path.

---

# 4. Internal Conversion

Formula-by-formula port of the legacy EQ 1-8:

1. Crossing detection compares the energy ordering of the current and the two previous steps (the three-step argmin criterion).
2. The gap threshold (`gap`, Hartree) gates the pair.
3. The classical allowance test checks the kinetic energy along the crossing direction. Where the kinetic energy comes from: `state.runtime.kinetic_energy`, which the MD loop samples once per step -- after the centre-of-mass correction, the ensemble post half step and the second constraint projection (#310) -- so the allowance test and the recorded step read the same velocities. Moving that sample point changed no population on the unconstrained-NVE calibration legs: checked pairwise against the pre-change code (fssh 20, gsh 12, lzsh 12, ktdc 12 trajectories, `md.pop` digests bit-identical).
4. EQ 7/8 build the approximate forces of the two states from the two previous gradient steps.
5. The approximate NAC vector follows from the force difference.
6. EQ 3 evaluates the improved Landau-Zener hopping probability.

Missing gradient rows for the involved states degrade the pair to zero probability.

---

# 5. Intersystem Crossing

Formula-by-formula port of the legacy 2020 adaptation:

- The probability consumes the previous-step gradients and the previous-step SOC values.
- The non-parallel PES assumption turns the Landau-Zener survival into a hop: `P = 1 - P_LZ`.
- An undeclared or uncomputed SOC pair yields zero probability instead of the legacy process termination (`sys.exit`); the pair map comes from `system.soc_pair_map` and the coupling values from `results['soc']` (cm-1).

---

# 6. Pure-State Density Rebuild

GSH keeps the legacy density semantics:

- The density matrix is rebuilt every step as the pure state of the final active state.
- There is no coherent propagation and no decoherence term (no FSSH-style substep propagation, no `deco` keyword).
- Consequently the electronic population block and the active-state block are trajectory-wise identical; the benchmark metrics exploit this as an internal consistency check (§12).

---

# 7. Velocity Adjustment

Shared with FSSH through `tsh_helper.py` (zero implementation change when extracted from `fssh.py`):

- Internal conversion hops rescale along the *target* state's approximate NAC vector through `tsh_helper.adjust_velo` (`adjust` / `reflect` policy applies).
- Intersystem crossing hops carry no NAC vector and take the isotropic rescaling fallback through `tsh_helper.adjust_velo_isotropic` (the missing-NAC policy of `FSSH_DEVELOPMENT.md` §18).
- Insufficient kinetic energy is treated as frustrated hopping.

---

# 8. Numerical Protection (Phase 6 Style)

Every degenerate input degrades to zero probability; nothing is clipped and nothing is reset:

- Fully frozen coordinates (the displacement over the last two steps is the zero vector). Per-component frozen motion (e.g. a strictly one-dimensional model trajectory) stays legal: the force inversion applies to the moving components only and frozen components contribute zero force instead of a 0/0 NaN — exactly the Tully / lz1 model case.
- Zero gap, zero force magnitudes and zero SOC couplings.
- Missing gradient rows or SOC pairs.

---

# 9. Legacy Bug Fixes

Two genuine legacy bugs were fixed during the port (both covered by regression tests):

1. **Hop velocity direction**: the legacy hop rescaling consumed the never-filled active-state row of the approximate NAC array (always the `ones` placeholder); the port uses the target state's approximate NAC vector, so the energy balance follows the hop direction.
2. **Frustrated-hop density**: the legacy density was written on the hop target even when the hop was frustrated; the port keeps both the active state and the density matrix on the departure state.

---

# 10. Hopping Sampling

- All allowed pairs (internal conversion plus intersystem crossing) accumulate in energy order, mirroring the FSSH cumulative sampling gate.
- One `np.random.uniform` draw per step decides the hop (the legacy stream is preserved).
- `maxh` bounds the target state; a hop above it is rejected.
- A successful hop updates the authoritative active-state record through `state.update_system({'current_state': hop})`; the workflow only chains the returned state.

---

# 11. PyRAI3MD Integration

GSH follows the factory-style integrator design defined in `FSSH_DEVELOPMENT.md`:

1. **Initialization timing**: `AIMD.run()` calls `surfacehop = surface_hopping(self.sfhp, state)` once before the loop; `sfhp: gsh` returns the dispatch layer `ElectronicStepper('gsh', state)`, which binds one `GshIntegrator` per center (a single transparent binding at C=1; `surface_hopping.py:80`, `electronic.py:560-597`), and the loop calls `state = surfacehop(state)`.
2. **Data ownership**: the instance holds only parsed static parameters (thresholds in atomic units, the spin list, the SOC pair map) and **never caches any trajectory data**. Every step reads through the official PyRAIState interfaces (`state.result`, the read-only history views, `system.state_spin`, `system.soc_pair_map`) and writes back through `update_system/update_dynamics`; restart semantics and serialization behavior are unchanged.
3. **Warm-up**: with fewer than 3 history steps the integrator only seeds the density from `state_population` and performs no hop decision (mirroring the FSSH warm-up); the kernels read t-dt and t-2dt one index deeper than the raw offset because the md loop records the current step into the history before the hopping decision.
4. **All-state gradient chain**: the schema default `activestate: 0` resolves all target states, the calculator stores the `(nstate, natom, 3)` gradient array in `results['gradient']`, and the probability kernels gate on the available rows.

---

# 12. Benchmark Calibration

The qualification benchmark pillar ships a GSH Tully-model calibration case (`qualification/benchmark/gsh.py` + `gsh.yaml`, see the gsh section of `QUALIFICATION_DEVELOPMENT.md`) and the dedicated Landau-Zener family (`gsh_lz1.yaml`, `gsh_lz2.yaml`, `gsh_lz3.yaml`):

- The case reuses the algorithm-independent fssh pieces (`load_case / apply_overrides / build_reference_input`), so GSH and FSSH calibrate against the identical exact split-operator quantum reference.
- The GSH production input owns a dedicated keyword forwarding table (`sfhp: gsh` plus the present-only `gap / gapsoc / adjust / reflect / maxh / activestate` entries); FSSH-only keywords (`deco / rk4 / substep / integrate / nactype`) fail the unrecognized-keyword guard. Model parameter overrides (`lz_a / lz_b`) forward into the qdm section of both legs (present-only).
- The tully verdict is the unified final-state metric set shared with the fssh evaluator (`evaluate_unified`): the report carries the final trajectory / state / reference populations, the deterministic ZN kernel replay of the recorded trajectories (`internal_hopping_probability`) and two judged differences — `state_reference_difference` (the ensemble-averaged final S1 population against the exact wavepacket reference final probability) and `state_internal_difference` (against the replayed gate probability). The hop statistic stays a clean Bernoulli count because of the pure-state rebuild (§6) and is reported as `final_trajectory_population`; the tully models carry no closed-form adiabatic gradient interface, so the replay (`_tully_kernel_probability`) differentiates the recorded state energies numerically (`_numerical_adiabatic_gradients`, central differences in the interior, one-sided at the ends) and takes the first non-zero gate probability. No analytic hopping probability exists for the tully models, so that comparison is skipped. The shipped tolerances are placeholders to be calibrated from the benchmark results.
- **The lz family** (lz1/lz2/lz3) calibrates against the Landau-Zener crossing models (constant diabatic slopes, constant coupling — the only model family where the EQ7/8 force-reconstruction premises hold exactly; the Tully models violate both): lz1 is the textbook linear crossing, lz2 flattens the upper surface (`V11 = 0, V22 = −lz_a·x`) and lz3 flattens the lower one (`V11 = lz_a·x, V22 = 0`). Within the unified metric set the lz cases judge three differences against the ensemble-averaged final S1 population: `state_reference_difference` (the exact wavepacket reference final probability), `state_analytical_difference` (`P_LZ` with `v_c` from energy conservation — both supplied by the qdm lz model modules) and `state_internal_difference` (the deterministic ZN gate probability replayed from the recorded trajectory snapshots through the production kernel, reported as `internal_hopping_probability`, with the adiabatic gradients supplied by each model's `adiabatic_gradients`). The slope difference at the crossing is 2·lz_a for lz1 and lz_a for the half-linear lz2/lz3, so their exponent doubles at equal velocity: `P = exp(-pi*lz_b^2/(lz_a*v_c))` vs `P = exp(-2*pi*lz_b^2/(lz_a*v_c))`. All shipped tolerances are 1.0 placeholders pending calibration.
- The lz calibrations share the parametrization `lz_a = 0.2712, lz_b = 0.0543` (gap minimum 2·lz_b = 2.95 eV, the gap gate pinned at 3.5 eV) and phase-align the crossing exactly onto the gate snapshot; the middle-probability target P_LZ ~ 0.37 fixes the crossing velocity. The far-side topology sets the propagation window: lz1 and lz2 run away on the descending lower surface (nstep = 45 and 38 bound the ensemble inside the fixed reference grid window [-30, +80] Bohr), lz3 coasts on the flat lower surface (nstep = 45 is comfortable). The lz2/lz3 wavepacket momenta (k0 ~ 137/155 au) sit above the Nyquist bound of the lz1 4096-point reference grid and double it to ngrid = 8192.
- The pmd_env entries ship disabled until the scan confirms the verdicts (exactly like the fssh entries); every calibration entry runs the unified 100-trajectory ensemble on a 4-process pool.

---

# 13. Testing

`qualification/verification/test_gsh.py` (24 cases):

- IC/ISC probability kernels against the legacy verbatim formulas;
- numerical guards (frozen coordinates — full and per-component, zero gradients, zero gap, missing SOC);
- warm-up seeding and the missing-energy pass-through;
- the history-window offset contract (t-dt / t-2dt one index deeper than the raw offset);
- IC hop with energy conservation along the target-state approximate NAC (bug 1 regression);
- frustrated hop keeps the state and the density on the departure state (bug 2 regression);
- ISC isotropic rescaling, undeclared SOC pair degradation, the `maxh` constraint and the shared `tsh_helper` layer;
- factory dispatch of `sfhp: gsh`.

The calibration evaluation (the dual-family verdict: the wavepacket reference and ZN kernel replay differences for the tully cases, the additional LZ-analytic verdict shared with fssh for the lz family, the analytic-probability and adiabatic-gradient delegation to the qdm lz model modules, the numerical gradient differentiation of the tully replay) is covered by `qualification/verification/test_benchmark.py`.
