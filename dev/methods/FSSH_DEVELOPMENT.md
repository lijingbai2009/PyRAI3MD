# PyRAI3MD FSSH NumPy + Numba Optimization Design

Last updated: 2026-11-02 10:40 (#448: the row-identity rule extended to the transition channel (#448): one row per declared `transition_pair`, no self pair, no padded zero row, and the Molcas deck grouping inverted back to declaration order. The rest was not re-audited. Before it, 2026-10-22 22:20 (sections 3/21 now spell the timing and implementation switches as the `fssh_*` keywords of `&md`, which is what shipped - there is no `&fssh` section; the file-structure block is marked as the design-time naming; the integrator factory is described as returning `ElectronicStepper`, which binds one algorithm instance per center. The rest was not re-audited. Before it, 2026-10-07 13:05))

---

> **Multi-center dispatch (refactor R5, 2026-09-08).** The FSSH numerical core
> in `workflow/integrator/fssh.py` is UNCHANGED by the multi-center base: it
> still reads one PyRAIState-shaped object and writes back through
> update_system/update_dynamics. Multi-center runs reach it through
> `workflow/integrator/electronic.py`: the `surface_hopping` factory returns an
> `ElectronicStepper` that binds one `FsshIntegrator` per center to a `CenterView`
> proxy (nstate -> the center's S_c, current_state -> the center's active local
> state, result/history -> the center's projected arrays, coordinates/gradient/
> NAC zero-masked to the center region). C=1 uses a transparent binding (the view
> IS the real state), bit-for-bit the legacy path (see `test_electronic.py`,
> `test_fssh_regression.py`). Each center's velocity write-back is masked to its
> own atoms, so one center's hop never changes another center's velocity.

# 1. Objective

Refactor the existing PyRAI2MD FSSH Cython implementation with NumPy + Numba instead of Cython:

- Keep the existing FSSH algorithm and physical model.
- NumPy serves as the reference implementation.
- Numba accelerates high-frequency numerical kernels.
- Support function-level NumPy/Numba selection.
- Support function-level timing profiling for real benchmarking.
- Reduce Python overhead and temporary array allocation in high-frequency substeps.
- Keep the FSSH main flow simple, without complex backend/registry abstractions.
- Provide a clean interface for later numerical stability fixes.

---

# 2. Overall Architecture

A strict two-level structure:

```
FSSH
├── rk4: numpy / numba
├── dPdt: numpy / numba
├── matB: numpy / numba
├── ktdc: numpy / numba
└── decoherence: numpy / numba
```

That is:

```
FSSH → function → numpy/numba
```

No additional abstractions such as:

- backend
- provider
- registry
- dispatcher
- backend manager

---

# 3. Input Configuration

The implemented spelling is the `fssh_*` block of `&md` - there is no `&fssh`
section (see "Keyword Placement" below and `schema/config.py:1405-1413`):

```
&md
    fssh_rk4: numba
    fssh_dpdt: numba
    fssh_matb: numpy
    fssh_ktdc: numba
    fssh_decoherence: numpy
```

Each function accepts only two legal values:

- numpy
- numba

Default (the same `&md` spelling; every `fssh_*` keyword defaults to `numpy`):

```
&md
    fssh_rk4: numpy
    fssh_dpdt: numpy
    fssh_matb: numpy
    fssh_ktdc: numpy
    fssh_decoherence: numpy
```

Rationale: the NumPy version is the reference implementation, so the code still runs without Numba and numerical regression stays convenient.

---

# 4. File Structure

Layout as proposed at design time; the shipped locations are the ones in the
File Mapping table below (`pyrai3md/workflow/integrator/fssh*.py`):

```
PyRAI3MD/
└── Dynamics/
    └── Propagators/
        ├── fssh.py
        ├── fssh_numpy.py
        ├── fssh_numba.py
        └── fssh_profile.py
```

fssh.py

Responsible for:

- FSSH main flow
- state management
- hopping decisions
- velocity adjustment
- decoherence calls
- NumPy/Numba function selection
- logging

fssh_numpy.py

NumPy reference kernels:

- dPdt_numpy
- rk4_numpy
- matB_numpy
- ktdc_numpy
- decoherence_numpy

fssh_numba.py

Corresponding Numba kernels:

- dPdt_numba
- rk4_numba
- matB_numba
- ktdc_numba
- decoherence_numba

Decoherence convention: the decay exponents carry a half factor
(exp(-delt/(2 tau_k))), matching OpenMolcas src/surfacehop/tully.F90,
so tau_k is the population decoherence time. Eq. 17 of Granucci &
Persico, J. Chem. Phys. 126, 134114 (2007), doi:10.1063/1.2715585,
damps amplitudes with the full exponent and would decay populations
twice as fast; OpenMolcas documents the half factor as the
correction of that equation, and this implementation follows
OpenMolcas. The kinetic-energy floor is likewise aligned to the
OpenMolcas clamp of 1e-5 au.

fssh_profile.py

Responsible for function-level timing.

---

# 5. Function Selection Mechanism

Select the functions once at initialization:

```python
dPdt = dPdt_numba if config["dPdt"] == "numba" else dPdt_numpy
rk4 = rk4_numba if config["rk4"] == "numba" else rk4_numpy
matB = matB_numba if config["matB"] == "numba" else matB_numpy
ktdc = ktdc_numba if config["ktdc"] == "numba" else ktdc_numpy
decoherence = (
    decoherence_numba
    if config["decoherence"] == "numba"
    else decoherence_numpy
)
```

Afterwards the FSSH main loop only calls:

```python
dA = dPdt(...)
A = rk4(...)
B = matB(...)
nacme = ktdc(...)
A = decoherence(...)
```

No NumPy/Numba decision is made inside the high-frequency substeps.

---

# 6. dPdt

Original algorithm:

\[ \dot{A} = KA - AK \]

where

\[ K = -iH - D \]

NumPy:

```python
def dPdt_numpy(A, H, D):
    K = -1j * H - D
    return K @ A - A @ K
```

Numba: use @njit.

For small electronic-state matrices, matrix multiplication is not guaranteed to be fastest; benchmarking is required:

- NumPy @
- Numba @
- Numba explicit loops

Key test sizes:

- nstate = 2
- nstate = 3
- nstate = 5
- nstate = 10
- nstate = 20

---

# 7. RK4

RK4 is the main optimization target because every electronic substep executes \( k_1, k_2, k_3, k_4 \).

The Numba version should merge the whole RK4 into one kernel:

```
Python
  ↓
Numba RK4
  ├── k1
  ├── k2
  ├── k3
  └── k4
```

Avoid:

```
Python → dPdt
Python → dPdt
Python → dPdt
Python → dPdt
```

Pre-allocate the workspace:

- k1
- k2
- k3
- k4
- A2
- A3
- A4

Avoid large numbers of np.zeros() per substep.

---

# 8. matB

Formula:

\[ B_{ij} = 2 \, \mathrm{Im}(A_{ij}^{*} H_{ij}) - 2 \, \mathrm{Re}(A_{ij}^{*} D_{ij}) \]

NumPy can use array operations directly:

```python
B = (
    2 * np.imag(np.conj(A) * H)
    - 2 * np.real(np.conj(A) * D)
)
```

Numba uses explicit loops.

The default implementation is likewise decided by benchmarking.

---

# 9. κTDC

Keep the existing curvature-driven TDC algorithm:

\[ \ddot{\Delta E} = \frac{\Delta E(t) - 2\Delta E(t-\Delta t) + \Delta E(t-2\Delta t)}{\Delta t^2} \]

and

\[ d_{ij} = -\frac{1}{2} \sqrt{\frac{\ddot{\Delta E}}{\Delta E}} \]

---

# 10. κTDC Gap Handling

This is an important numerical fix in the refactoring.

gap > 0

Normal computation:

```
κTDC → NACME → hopping probability
```

gap = 0

Neither perform division by gap, nor simply set NACME = 0 and continue normal hopping.

Instead: record the properties of the current step, skip the hopping decision of that pair and proceed to the next step.

Reason: when \(\Delta E(t) = 0\), \(\sqrt{\ddot{\Delta E}/\Delta E}\) is not defined in the first place.

gap < 0

For NN PES, correction via `np.sort(E)` is not allowed.

The state identity output by the NN follows the PES identity defined by the training data, not the instantaneous energy order. Therefore:

```
NN state identity  ≠  instantaneous energy order
```

When gap < 0:

- Keep the NN state identity.
- Mark the current pair invalid.
- The pair does not participate in hopping this step.
- Do not resort the PES.
- Re-evaluate on the next step.

---

# 11. Multi-State Gap Validity

The following is not allowed:

```python
if any_gap_invalid:
    skip_hopping()
```

because the situation may be:

```
S0-S1 : gap = 0
S0-S2 : valid
S1-S2 : valid
```

Pair-level validity must be used:

```
pair_valid[i, j]
```

For example:

```
(0,1) → invalid
(0,2) → valid
(1,2) → valid
```

Only the affected pair is excluded from hopping on the current step.

---

# 12. State Identity and Hopping Sampling Order

The following must be strictly separated:

PES identity

Determined by the NN output index; it must not be changed by energy sorting.

Hopping sampling order

May use

```python
stateindex = np.argsort(E)
```

for the cumulative sampling of conventional FSSH:

\[ g_1 + g_2 + \cdots > z \]

Therefore energy sorting may only be used for the sampling order, never to redefine the PES identity.

---

# 13. Singlet + Triplet Hopping

For systems containing both singlet and triplet states:

- Do not split the hopping probability by multiplicity.
- All allowed target states participate in the FSSH probability.
- Accumulate in hopping sampling order.
- SOC enters the electronic propagation through the Hamiltonian off-diagonal coupling.

When SOC is weak and does not change the PES identity, no extra S/T PES reordering is performed.

Implementation (workflow/integrator/fssh.py):

- The sampling gate uses the unified pair set `hop_pairs = nac_pairs + soc_pairs` (duplicates kept once, each pair tagged 'nac' or 'soc'). Both pair types feed the same `g` vector; the cumulative loop (`argsort(energy)` order, `gsum > z`, `maxh` limit) is shared with same-spin hopping and carries no multiplicity branching.
- Pair validity is pair-type dependent:
  - NAC pairs keep the Phase 6 identity-order rule (degenerate or energy-order-inverted gaps invalidate the pair; it protects the NAC matrix assembly, whose coupling sign follows the `(E_j - E_i)` convention).
  - SOC pairs are invalidated by degenerate gaps only (`|gap| <= GAP_CUTOFF`). A T1-below-S1 crossing is a normal intersystem-crossing region, not an identity inversion, so the energy-order inversion exclusion must not apply.
- The NAC matrix assembly keeps receiving the NAC-only validity map; SOC pairs carry no derivative coupling, so the B-matrix cross-spin flow comes solely from the `2 Im(A* H)` SOC term.
- Velocity adjustment of a cross-spin hop follows section 18: no NAC vector exists for the pair, so the hop is energy-balanced by the isotropic rescaling fallback.
- Whether SOC is active is decided by the data: `soc_pair_map` plus `results['soc']` (cm-1) supplied by the skill; the `sfhp` keyword system is unchanged. Two rules make that mapping addressable (#447): the integrator indexes the rows with `soc_pair_map`'s own order (`_build_hamiltonian`, `fssh.py:326`), so a skill must answer one row per declared pair in that order - molcas and openqp read their elements pair-by-pair, and a requested pair the engine printed no element for is now refused by name instead of recorded as `0.0`. A `0.0` there is not "no coupling": it is an ISC rate of zero for a pair the run never evaluated.
- The same identity rule governs the transition channel, which the spectrum and
  the hop-rate bookkeeping read by row (#448). `transition_dipole` and
  `osc_strength` carry **one row per declared `transition_pair`, in the
  declaration's order**, and a run with nothing declared carries no row at all.
  Two things used to break that: every skill's schema rebuilt the table as
  `[[0, i] for i in range(nstate)]`, whose first entry `(0, 0)` is a state
  coupled to itself and which therefore shifted every later row's address by
  one, and the readers padded a lone zero row when the table was empty
  (`npair = len(pairs) if pairs else 1`). Both halves are gone; the default the
  framework derives starts at state 1. Molcas needs one more step because its
  `&ALASKA` blocks are written per spin group: the deck order is a function of
  the declaration (`input_builder.py::nac_deck_order`) and the reader inverts it,
  so the rows arrive in the order the input named.

---

# 14. Decoherence

Keep the current OpenMolcas-style renormalization scheme; its theoretical behavior is not changed during the performance refactoring phase.

Main optimization:

\[ f_i = \exp(-\Delta t / \tau_i) \]

then

\[ A_{ij} \rightarrow f_i f_j A_{ij} \]

NumPy can use:

```python
factor = np.exp(-delt / tau)
A *= factor[:, None] * factor[None, :]
```

followed by the active-state population renormalization.

---

# 15. Population Numerical Error

A certain amount of floating-point error is tolerated: \( A_{ii} < 0 \) or \( A_{ii} > 1 \).

Do not directly use `np.clip(Aii, 0, 1)`, because that artificially modifies the density matrix.

Current experience: population deviations within \( 10^{-2} \) usually do not visibly affect the trajectory.

Default strategy:

- Detect.
- warning/debug output.
- Keep the original value and continue propagation.
- Determine whether it is related to timestep/RK4/decoherence.

Further numerical protection is only applied when a clear instability appears.

---

# 16. Active-State Population

Do not assume `Aii = 0` is a normal situation.

The active state itself should carry a non-zero population. Therefore \( A_{ii} \approx 0 \) is mainly used as a numerical diagnostic rather than directly triggering a state reset.

---

# 17. NAC Lookup Optimization

The existing

```python
nac_pair in nac_coupling
nac_coupling.index(nac_pair)
```

is a linear search.

Build at initialization:

```python
nac_index[state, target]
```

so that `idx = nac_index[state, target]` directly, reducing O(N) to O(1).

This optimization is independent of NumPy/Numba but worth doing at the same time.

---

# 18. Missing NAC Vector

The existing

```python
np.ones((natom, 3))
```

fallback is not explicit enough.

Distinguish:

- NAC available
- NAC unavailable

and define an explicit fallback policy for velocity adjustment.

In particular, κTDC has no NAC vector, so the velocity adjustment after a κTDC hop must be designed separately.

This is a numerical-algorithm fix and must not be mixed with performance optimization.

---

# 19. Hopping Random Number

Make an independent decision per electronic substep:

```
substep
   ↓
calculate g
   ↓
generate random z
   ↓
cumulative probability
   ↓
hopping decision
```

instead of

```
all substeps
   ↓
accumulate probability
   ↓
one random number
```

This matches the approach of the current OpenMolcas surface hopping implementation.

---

# 20. integrate

Legacy situation:

```
integrate == 99
```

was a debug mechanism.

The refactoring removes the magic number and fully isolates the debug force-hop from the electronic-step integration:

- `integrate` keeps only the electronic substep integration mode semantics (0: reset the B matrix every substep; non-zero: accumulate B across substeps);
- the debug force-hop is split into the independent &md keyword `fssh_force_hop` (0/1, default 0): the hopping random number is forced to zero so that any positive probability hops; debugging/testing only;
- both the `integrate == 99` magic number and the `INTEGRATE_DEBUG_FORCE_HOP` named constant have been removed; the value space of `integrate` is no longer reused.

---

# 21. Profiling

FSSH adds, in `&md` (an integer switch, not a `&fssh` section):

```
&md
    fssh_profile: 1
```

Timing is collected separately for:

- dPdt
- rk4
- matB
- ktdc
- decoherence
- hopping
- velocity adjustment

Output:

```
FSSH Performance
-----------------------------------------
Function          Calls      Total
dPdt              40000      1.82 s
RK4               10000      2.31 s
matB              10000      0.21 s
kTDC              30000      0.17 s
decoherence       10000      0.44 s
hopping           10000      0.05 s
-----------------------------------------
Total                         5.00 s
```

plus the implementation table:

```
Implementation
dPdt          numba
RK4           numba
matB          numpy
kTDC          numba
decoherence   numpy
```

---

# 22. Numba Warm-Up

The first Numba call includes JIT compilation time.

Benchmarks must therefore follow:

```
initialization
 ↓
Numba warm-up
 ↓
JIT compilation
 ↓
start timing
```

The first call must not be used directly as performance data.

---

# 23. Benchmark Strategy

Fix:

- trajectory
- timestep
- substep
- nstate
- CPU
- Python/NumPy/Numba versions

Test separately:

- all NumPy
- dPdt Numba
- dPdt + RK4 Numba
- dPdt + RK4 + matB Numba
- ...

Final result:

```
Function       NumPy       Numba       Speedup
dPdt           xxx us      xxx us      x.xx
RK4            xxx us      xxx us      x.xx
matB           xxx us      xxx us      x.xx
kTDC           xxx us      xxx us      x.xx
decoherence    xxx us      xxx us      x.xx
```

Adopt Numba based on actual benchmarks, not on the assumption that Numba is always faster.

---

# 24. Numerical Regression

Every Numba kernel must be compared against the NumPy reference, e.g.

\[ \max |A_{\mathrm{NumPy}} - A_{\mathrm{Numba}}| \]

covering:

- dPdt
- RK4
- matB
- κTDC
- decoherence

as well as:

- A Hermiticity
- Tr(A)
- population
- hopping probability

Ultimately add trajectory-level regression:

- same initial condition
- same random seed
- NumPy vs Numba

comparing:

- population evolution
- state history
- hopping events
- energy
- velocity

---

# 25. Workspace Optimization

Pre-allocate at FSSH initialization:

```
workspace
├── dA
├── k1
├── k2
├── k3
├── k4
├── A2
├── A3
├── A4
├── B
├── dB
├── tau
├── factor
└── g
```

Principle: create as few new ndarrays as possible inside the high-frequency substeps.

Especially avoid repeated calls of:

- np.zeros(...)
- np.ones(...)
- np.copy(...)

---

# 26. Development Order

Phase 1: NumPy reference

First convert the existing Cython FSSH to pure NumPy:

```
Cython
   ↓
NumPy reference
```

Ensure identical physical behavior.

Phase 2: structural optimization

Handle:

- workspace
- NAC lookup
- temporary arrays
- matrix operations

Phase 3: Numba kernels

Implement in order:

```
dPdt
↓
RK4
↓
matB
↓
κTDC
↓
decoherence
```

Run the regression test immediately after each kernel.

Phase 4: function-level selection

Implement (shipped as the `fssh_*` keywords of `&md`, not as a `&fssh` section):

```
&md
    fssh_dpdt: numba
    fssh_rk4: numba
    fssh_matb: numpy
    fssh_ktdc: numba
    fssh_decoherence: numpy
```

Phase 5: profiling

Add function-level timing and identify the real bottlenecks.

Phase 6: numerical stability

Handle independently:

- gap = 0
- gap < 0
- population < 0
- population > 1
- missing NAC
- frustrated hopping

Where the kinetic energy comes from: `state.runtime.kinetic_energy`, which the MD loop samples once per step -- after the centre-of-mass correction, the ensemble post half step and the second constraint projection (#310) -- so the allowance test and the recorded step read the same velocities. Moving that sample point changed no population on the unconstrained-NVE calibration legs: checked pairwise against the pre-change code (fssh 20, gsh 12, lzsh 12, ktdc 12 trajectories, `md.pop` digests bit-identical).

Phase 7: trajectory regression

Final verification:

```
legacy PyRAI2MD
      ↕
PyRAI3MD NumPy
      ↕
PyRAI3MD Numba
```

Ensure the optimization does not change the core physical behavior of FSSH.

---

# 27. Final Design Principles

The core of FSSH should stay very simple:

```
                    FSSH
                      │
       ┌──────────────┼──────────────┐
       │              │              │
      RK4            dPdt           matB
       │              │              │
    numpy/numba    numpy/numba    numpy/numba

       ┌──────────────┴──────────────┐
       │                             │
      κTDC                       decoherence
       │                             │
    numpy/numba                  numpy/numba
```

No extra backend layer.

The core principle is:

One mathematical function = one NumPy implementation + one Numba implementation; the FSSH configuration decides which one is used; NumPy is the reference, Numba provides acceleration, and the profiler decides which functions are worth accelerating.

This keeps the flat design of PyRAI3MD 3.0 while leaving enough room for further algorithm optimization, NN PES numerical protection and performance benchmarking of FSSH.

---

# PyRAI3MD Implementation Status

How this design landed in PyRAI3MD 3.0 (incremental refactoring from the legacy `PyRAI2MD/Dynamics/Propagators/fssh.pyx`, without changing the AIMD main loop or the existing PyRAIState logic).

## File Mapping (design document -> PyRAI3MD)

| Design document location | PyRAI3MD location | Notes |
|---|---|---|
| `Dynamics/Propagators/fssh.py` main flow | `pyrai3md/workflow/integrator/fssh.py` | `FsshIntegrator` class: state management, hopping decisions, velocity adjustment, numerical protection; `run_fssh(state)` kept as a thin wrapper around a disposable instance |
| `fssh_numpy.py` | `pyrai3md/workflow/integrator/fssh_numpy.py` | The 5 NumPy reference kernels (dPdt / rk4 / matB / ktdc / decoherence) |
| `fssh_numba.py` | `pyrai3md/workflow/integrator/fssh_numba.py` | Optional Numba kernels (`NUMBA_AVAILABLE` guard, automatic numpy fallback when missing) |
| `fssh_profile.py` | `pyrai3md/workflow/integrator/fssh_profile.py` | Function-level timing and implementation table (§21), Numba warm-up (§22) |
| Velocity adjustment helpers (legacy `tsh_helper.py`) | `pyrai3md/workflow/integrator/tsh_helper.py` | Shared velocity adjustment layer consumed by every hopping algorithm (`reflect_velo / adjust_velo / adjust_velo_isotropic`); mass conversion read centrally through the data-side `state.mass_au`; the GSH consumers are documented in `GSH_DEVELOPMENT.md` |
| `Dynamics/Propagators/gsh.py` main flow | `pyrai3md/workflow/integrator/gsh.py` | `GshIntegrator` class: Zhu-Nakamura internal conversion and intersystem crossing probabilities, energy-ordered cumulative sampling, pure-state density rebuild; `run_gsh(state)` kept as a thin wrapper around a disposable instance; see `GSH_DEVELOPMENT.md` |
| NAMD entry | `pyrai3md/workflow/integrator/surface_hopping.py` | `surface_hopping(method, state)` factory: initialized once per trajectory and returns the dispatch layer `ElectronicStepper(method, state)`, which binds one algorithm instance per center - one `FsshIntegrator`/`GshIntegrator`/`LzshIntegrator` each, a single transparent binding at C=1 (`surface_hopping.py:77-80`, `electronic.py:560-597`); `surface_hopping` remains the only external hopping entry |

## Keyword Placement (&md, no new &fssh section)

- All existing keyword semantics are kept unchanged: `substep / rk4 / integrate / deco / adjust / reflect / maxh / gap / nactype / sfhp`;
- Function-level implementation selection added: `fssh_rk4 / fssh_dpdt / fssh_matb / fssh_ktdc / fssh_decoherence` (values `numpy|numba`, default `numpy`);
- `fssh_profile` added (0/1, function-level timing summary switch);
- `fssh_force_hop` added (0/1, debug force-hop switch, fully isolated from the `integrate` electronic-step integration mode, see §20);
- The density matrix A is persisted in `system.state_density` (nested list of `[re, im]` pairs), written back through `update_system`, supporting serialization and restart; `state_population` stays consistent with the real diagonal of A (written back through `update_dynamics`).

## Phase 6 Numerical Strategy Implementation Notes

1. **pair-level gap validity (§10/§11)**: each step computes `gap = E[j] - E[i]` in PES identity order; a pair with `|gap| <= 1e-16` or `gap < 0` is invalid for the step (excluded from NAC matrix construction and hopping sampling), no `np.sort(E)` resorting is performed, re-evaluated on the next step.
2. **population numerical error (§15)**: out-of-range `A_ii` only triggers a warning diagnostic; the original value continues to propagate, no `np.clip`.
3. **active-state population (§16)**: near-zero values only produce diagnostic output, no state reset.
4. **missing NAC vector (§18)**: available / unavailable are distinguished explicitly, the legacy `ones((natom, 3))` fallback removed. Available → `adjust/reflect` along the NAC direction; unavailable (including κTDC hops) → isotropic velocity rescaling `f = sqrt((Ekin + E_old - E_new)/Ekin)`; insufficient kinetic energy is treated as frustrated hopping (original velocity and state kept).
5. **integrate magic number (§20)**: the `integrate == 99` magic number is removed; the debug force-hop is split into the independent `fssh_force_hop` keyword (0/1); `integrate` keeps only the electronic substep integration mode (0 resets B every substep, non-zero accumulates).
6. **kTDC previous-step D reconstruction**: the legacy propagator persists the computed `last_d`; PyRAI3MD rebuilds D(t-dt) from the history, so the kTDC curvature window reads the genuine three points E(t-dt), E(t-2dt) and E(t-3dt) (history step -3). When the third point is missing it degenerates to reusing E(t-2dt), keeping the propagation well-defined with zero-curvature data.
7. **nactype validation**: the integrator accepts only `nac` and `ktdc` (case-insensitive); any other value raises `ValueError` at initialization instead of silently running the kTDC branch.
8. **rolling active state within substeps**: as in legacy, a successful (non-frustrated) hop inside a substep immediately changes the probability reference state; the velocity is adjusted once from the original velocity only after the loop ends.
9. **warm-up**: when the history has fewer than 4 steps, the density matrix is only seeded from `state_population` (corresponding to legacy `iter < 4`).

## Class-Based Initialization (Factory Style)

1. **Initialization timing**: `AIMD.run()` calls `surfacehop = surface_hopping(self.sfhp, state)` once before the loop; inside the loop `state = surfacehop(state)`. Parameter parsing (dynamics keywords, pair maps), kernel selection (§5) and Numba warm-up (§22) all happen at initialization; an unknown `sfhp` raises `ValueError` before any work starts.
2. **Data ownership**: the `FsshIntegrator` instance holds only parsed static parameters and kernel bindings, **never caches any trajectory data**. The density matrix, previous-step energies/velocities, etc. are fetched every step through the official PyRAIState interfaces (`system.state_density`, `state.result`, read-only history views) and written back through `update_system/update_dynamics`; restart semantics and serialization behavior are unchanged.
3. **Process-level globals removed**: no module-level caches such as `_KERNELS/_PROFILER` exist anymore; multiple trajectories in the same process with different `fssh_*` configurations do not interfere; the profiler is per-instance.
4. **Efficient data access (for NN potential scenarios)**: history reads use the `state.get_history_step(index, copy=False)` read-only view, eliminating two full-snapshot deep copies per step; `state_density` write-back is vectorized (`np.stack(...).tolist()`); `nac_index` is precomputed once; random numbers are drawn once per step (same stream as per-substep scalar draws, bit-identical sequence).

## Testing

- `qualification/verification/test_fssh.py`: per-kernel regression against the legacy scalar-loop formulas, numpy/numba numerical consistency, density-matrix serialization round trip, warm-up, official hop path, Phase 6 numerical strategy cases;
- `qualification/verification/test_fssh_regression.py`: trajectory-level regression driven by synthetic models (2-state / 3-state / force-hop), numpy vs numba vs golden three-way comparison, golden references fixed offline (reproducible with the same seed, no Cython build environment required).

## GSH (Zhu-Nakamura) Implementation

The Zhu-Nakamura global surface hopping implementation (`pyrai3md/workflow/integrator/gsh.py`, the shared `tsh_helper.py` velocity adjustment layer and the Tully model calibration case) is documented separately in `GSH_DEVELOPMENT.md`.

## Benchmark Calibration (Tully + Landau-Zener family)

The qualification benchmark pillar calibrates FSSH in `qualification/benchmark/fssh.py` against the exact split-operator quantum reference of the qdm skill (identical initial conditions on both legs), with one unified case file (`fssh.yaml`) serving the Tully models and dedicated case files for the Landau-Zener family (`fssh_lz1.yaml`, `fssh_lz2.yaml`, `fssh_lz3.yaml`):

- The Tully verdict is the unified final-state metric set: the evaluator reports the final trajectory / state / reference populations and judges `state_reference_difference` (the ensemble-averaged final S1 population against the exact wavepacket reference final probability). The shipped pins carry the calibration verdicts: `deco: 'off'` (the decoherence decay overestimates the final S1 population by ~+0.08 through suppressed back-hopping) and `rk4: '1'` (the forward-Euler coherence rotation diverges on the wide-gap models).
- The lz family runs the identical physical setup of the gsh lz calibrations (same initial conditions and model parameters, dedicated `gsh_lzN.yaml` counterparts) so both hopping algorithms are compared on the same problem; the dedicated case files pin the fssh knobs (`deco: 'off'`, `rk4: '1'`, `nactype: nac`) plus the gap gate (3.5 eV, above the gap minimum 2·lz_b = 2.95 eV).
- The lz verdict stays anchored on the analytic single-passage Landau-Zener probability within the unified metric set: besides `state_reference_difference` the lz cases judge `state_analytical_difference` (the ensemble-averaged final S1 population vs `P_LZ`, with the crossing velocity `v_c` from energy conservation — both delegated to the qdm lz model modules; lz1 uses `P = exp(-pi*lz_b^2/(lz_a*v_c))`, the half-linear lz2/lz3 use `P = exp(-2*pi*lz_b^2/(lz_a*v_c))`). The wavepacket final probability is reported (`final_reference_population`) and judged through the reference difference like every other case; the deterministic kernel replay internal probability is GSH/LZSH-specific and never computed for FSSH.
- Every calibration entry runs the unified 100-trajectory ensemble on a 4-process pool (binomial noise of the final S1 fraction sigma ~ 0.043-0.048; the shipped tolerances are 1.0 placeholders pending calibration); the pmd_env lz entries ship disabled until the scan confirms the verdicts.
- The lz routing, the analytic-probability delegation and the verdict protocol are covered by `qualification/verification/test_benchmark.py`.

---

## Appendix: Tully Model Reference (Atomic Units)

The Tully 1-3 benchmark models use atomic units (a.u.) internally. The QDM skill and the FSSH benchmark calibration both adopt this convention so that the quantum reference and the trajectory results stay aligned.

### Parameter Unit Table

| Parameter | Meaning | Atomic unit |
|-----------|---------|-------------|
| x | 1-D nuclear coordinate | bohr |
| M | nuclear mass | electron mass |
| V11, V22 | diabatic potential energy | Hartree |
| V12 | diabatic coupling | Hartree |
| dV/dx | potential gradient | Hartree/bohr |
| t | time | a.u. time |
| dt | timestep | a.u. time |
| k0 | initial wavenumber | bohr^-1 |
| p0 | initial momentum | a.u. |
| sigma | initial Gaussian width | bohr |

Conversion factors:

- 1 a.u. time = 0.02418884 fs
- 1 Hartree = 27.2114 eV
- 1 bohr = 0.529177 Å

### Tully 1: Simple Avoided Crossing

The diabatic potentials and coupling:

\[ V_{11}(x) = \begin{cases} A(1 - e^{-Bx}), & x > 0 \\ -A(1 - e^{Bx}), & x < 0 \end{cases} \]

\[ V_{22}(x) = -V_{11}(x) \]

\[ V_{12}(x) = C e^{-D x^2} \]

Classical parameters:

| Parameter | Value | Unit note |
|-----------|-------|----------|
| A | 0.01 | Hartree |
| B | 1.6 | B^{1/2} ~ bohr^{-1} (makes Bx dimensionless) |
| C | 0.005 | Hartree |
| D | 1.0 | D ~ bohr^{-2} (makes Dx^2 dimensionless) |

### Mass and Time in Atomic Units

The QDM / FSSH benchmark parameters `model_mass` and `ref_dt` are in atomic units:

- `model_mass: 2000.0` means M = 2000 m_e (electron masses), approximately 1.097 amu. This is the typical reduced nuclear mass used in Tully models.
- `ref_dt: 20.0` means dt = 20 a.u. time, approximately 0.484 fs.
- `ref_nsteps: 200` gives a total propagation time of 200 x 20 = 4000 a.u., approximately 96.8 fs.
