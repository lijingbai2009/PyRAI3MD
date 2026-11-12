# PyRAI3MD Periodic Boundary, Ensemble and Thermostat/Barostat Design

Last updated: 2026-10-22 22:00 (section 8's `optimize_cell` pointer now names `CELL_OPT_DEVELOPMENT.md`, where that material moved, and section 1.4 describes the deleted `data/dynamics`/`data/structure` objects in the past tense and names their deletion; the rest was not re-read. Before it, 2026-10-13 14:22: the `kabsch` and `apply_constrain` locations were checked against the code; the rest of the file was not re-audited)

---

> **A stochastic cell barostat, and a pairing table (2026-09-29).** See
> CHANGELOG #257 and section 8.13. `barostat: scr` (stochastic cell
> rescaling, Bernetti-Bussi JCP 153, 114107 (2020); tensor form arXiv:2111.06402)
> is implemented in `workflow/integrator/ensemble.py::_scr`, and the
> `thermostat` x `barostat` combinations that mean something are now a table
> enforced at validation time rather than a free grid.

> **NPT hardening (2026-09-26).** See CHANGELOG #223. Three corrections from the
> first remote variable-cell runs: (1) the MTTK cell-momentum bath kinetic energy
> is `sum(p_eps^2) / (2W)` (the inverted `0.5 W sum(p^2)` form inflated it by W^2
> and overflowed the chain to a NaN cell within two steps at realistic stress);
> (2) `_scale_cell` now re-triangularizes the deformed lattice
> (`L = chol(B B^T)`, frame rotation `O = B^-1 L` applied to the committed
> coordinates/velocities): anisotropic deformations `B @ S` used to leave
> upper-triangular components, breaking the framework lower-triangular convention
> that backend data writers rely on — fractional coordinates, volume and all
> distances are exactly invariant under the rotation, at the cost of a slow global
> frame rotation of the trajectory (tensors committed in earlier frames stay in
> their original frame; the per-step rotation is O(shear increment), far below the
> split integrator's truncation error); (3) the `barostat_mode` default is now
> `isotropic` (anisotropic stays a fully supported explicit option).

> **Phase-1 research report (Sep 12 2026).** This document is research/collection
> material; it contains no implementation plan and is not accompanied by any code
> changes. Sections 3 and 4 are context and interface constraints prepared for
> later plan-making; they do not constitute a plan themselves.

> **Per-step ensemble diagnostics (2026-09-25).** See CHANGELOG #218. The
> thermostat/barostat intermediates that used to live and die as locals inside
> `workflow/integrator/ensemble.py` (NHC scale factors and chain masses, the CSVR
> alpha/ratio/c draws, the Berendsen mu matrix and pressure tensors, the
> extended-cell W, g_eps, alpha and particle/cell scale factors, plus explicit
> `*_skipped` reasons for the guard branches) are now collected per step into the
> transient `state.runtime.ensemble` dict (`_diag` helper, reset at the pre stage
> of every step; nve keeps it empty). The persisted extended DOF
> (`thermostat_dof`, `barostat_strain/momentum/velocity`,
> `barostat_thermostat_dof`) remain the sole authority for the chain/cell state.
> Consumers: the `md.ensemble`/`sh.ensemble` trajectory files and the run-log
> `&ensemble` section (formats in `dev/methods/MD_DEVELOPMENT.md` section 23);
> the history frames carry a light runtime projection (kinetic energy + ensemble
> diagnostics) so `RunLogger.log_step` sees them. This round also fixed the MTTK
> cell-momentum NHC to rescale p_eps with the accumulated **scale factor**
> returned by `_nhc_propagate` instead of its second return (the propagated bath
> kinetic energy), which overflowed the cell to NaN within a few steps at
> physical velocities.

## 1. Code Status Review

### 1.1 MD main loop and integrator

The new framework's AIMD main loop is in `_run_trajectory()` of `pyrai3md/workflow/md.py`; the per-step data flow is now fixed as:

```
verlet_i  ->  apply_pbc  ->  calculator.run  ->  merge_deviations
-> verlet_ii -> phase_correction -> apply_constrain -> constraint_commit
-> velocity_correction -> apply_ensemble(stage=post)
-> apply_constrain again when a barostat has just rescaled coordinates and cell
   (md.py phase `constrain_after_cell`, #276)
-> kinetic_energy()  (one sample per step, from the step-final velocities)
-> update_history
-> surfacehop -> verification -> calculator.sync() -> advance_step -> checkpoint (the counter moves before anything is written, so the frame, the log block and the runtime file all name the step that finished; #281)
```

The integrator `pyrai3md/workflow/integrator/verlet.py` is the two-stage velocity-Verlet validated in PyRAI2MD (the two half-step velocity updates are merged):

\[
R(t+\Delta t) = R(t) + V(t)\Delta t - \tfrac{1}{2}\frac{G(t)}{m}\Delta t^2,\qquad
V(t) = V(t-\Delta t) - \tfrac{1}{2}\frac{G(t-\Delta t)+G(t)}{m}\Delta t
\]

Internal unit conventions: coordinates Å, velocities Bohr/au, gradients Hartree/Bohr, masses amu (converted to au via `state.mass_au`), time au. All reads go through read-only `PyRAIState` properties; all writes go through `state.update_system()`.

### 1.2 Key landing points: the four TODO placeholders in operations.py

All of the new phase-1 physics corresponds to functions in `pyrai3md/workflow/integrator/operations.py` that are currently placeholders (returning state unchanged), and their call sites in the main loop are already arranged:

| Function | Status | Phase-1 responsibility |
|---|---|---|
| `apply_pbc` | TODO | periodic coordinate wrapping (before the calculator) |
| `apply_ensemble` | implemented | thermostats: Nose-Hoover, Nose-Hoover chain, CSVR only (`validate.VALID_THERMOSTATS` = off / nose-hoover / nose-hoover-chain / csvr); SCR and Berendsen are **barostats**, and the pre/post half-step split belongs to the extended variables. Measured in a periodic cell with constraints present: NVT holds two declared bonds to `8.131e-11` A, NPT with the post-step reprojection to `1.71178627e-11` A (sections 26 and 26.1 of `CONSTRAINT_DEVELOPMENT.md`) |
| `velocity_correction` | implemented, deliberately partial | removes the mass-weighted center-of-mass translation and rescales back to the same kinetic energy, so the temperature is untouched, and returns the state untouched for fewer than two atoms (one pseudo-atom has no internal degree of freedom -- subtracting its centre deletes its motion, which is the shape of the Tully and Landau-Zener model trajectories); the rotation half of the legacy reset is not applied, because in a periodic box it would fight the cell-internal flow |
| `apply_constrain` | implemented | frozen components to their anchor, bond/angle/torsion conditions solved on the unfolded geometry and folded back with their crossing counts, `rattle` velocity projection when declared; the additive energies and gradients are NOT applied here (they belong to the Calculator evaluation) |

All four are in. Carrying NPT did turn out to need one change to the loop order, and it is in the block above: a barostat rescale is not a rigid motion, so the geometry has to be asked again after it (D-25).

### 1.3 Runtime data model: PyRAIConfig (Pydantic) is the sole authority

The runtime `PyRAIState` (`pyrai3md/data/core.py`) is built on top of `schema.config.PyRAIConfig`, accessed via `state.pmd.system` / `state.pmd.dynamics`. The relevant fields **already exist** (`pyrai3md/schema/config.py`):

- `System`: `pbc (3,)`, `cell (6,) = [a,b,c,α,β,γ]`, `lattice_vector (3,3)`, `kpoint`, `volume`; plus constraint-related `freeze/constrain/cbond/cangle/cdihedral/tbond/...` spellings are retired by name (config.py `RETIRE_CONSTRAINT_KEYS`); a restraint is declared in `system.constraints`, whose walls carry the confinement and the cavity's old spring law.
- `Dynamics`: `temperature`, `pressure`, `thermostat='off'`, `barostat='berendsen'`, `ensemble='nve'`, `thermodelay`, `dissipate`, `size` (time step). The legacy single `thermo` spelling was retired by name in #314: it predates the `ensemble` + `thermostat` + `barostat` vocabulary and had no reader.

`PyRAIState` already has `kinetic_energy()` (writes `runtime.kinetic_energy`), `mass_au`, `update_system()`, and the history API (`update_history`/`get_history_step`).

### 1.4 Object model that exists but is **deprecated** (important)

Under `pyrai3md/data/dynamics/` and `pyrai3md/data/structure/` there **was** a full, fine-grained object-model scaffold: `Cell` (with `volume`, `boundary_condition`, `ndim_periodic`), `Ensemble`, `Thermostat`, `Barostat`, `Constraints` (with `SHAKEParameters`, `CavityConstraint`, `Compressor`), `CollectiveVariable`, `EnhancedSampling`, `Metadynamics`. Both directories were deleted on 2026-09-25 (#216, PBC stage A+B) as pure data containers with zero consumers; the field design survives in the memo below.

At the time, `pyrai3md/data/dynamics/__init__.py` stated:

> DEPRECATED: This sub-package is deprecated as of Aug 03 2026. Parameters will be moved to the derivation module.

`core.py` never imported this model, and the runtime never went through it. The classes were therefore **unwired design references** - a blueprint for field/parameter naming, not a landing vehicle. That was the architectural decision point Section 3 took up.

### 1.5 Three hard gaps (key to NPT feasibility)

1. **stress_tensor is declared but never populated.** `Properties.stress_tensor (nstate,3,3)` is declared in the main schema and in all 6 skill schemas, but `grep` confirms that **no service-layer code writes to it**. Yet every barostat needs the internal pressure

\[
P_{\text{int}} = P_{\text{kin}} - P_{\text{vir}} = \frac{1}{3V}\sum_i\Big(\tfrac{p_i^2}{2m_i} - \mathbf{r}_i\cdot\mathbf{F}_i\Big)
\]

in which the virial/stress must come from the electronic-structure backend. This is the biggest prerequisite dependency for NPT.

2. **Variable-cell information is not synced to the backend.** `SYNC_SYSTEM_FIELDS` in `pyrai3md/workflow/calculator.py` only contains `atoms/mass/coordinates/velocities/charges/multiplicity/natom/nstate/ci/spin/coupling/state_label/state_spin/current_state`, and **does not contain `cell/lattice_vector/pbc/volume`**. After NPT changes the cell each step, the new cell cannot be passed to the backend via `calculator.sync()`; even for fixed-cell PBC, the backend needs the cell in order to do periodic electronic structure.

3. **Input keywords are exposed.** The `MD_TYPES` whitelist in `pyrai3md/schema/adapter.py` carries `thermostat`/`barostat`/`ensemble`/`pressure` and the coupling keywords next to `temp`/`thermodelay`/`dissipate`, so the classic text, the YAML section and the agent's dotted form all reach the same fields; `FIELD_MAP` routes the legacy `temp` to `temperature`.

In addition, `units.py` currently has no unit-conversion constants for pressure/virial/compressibility (bar→au, atm→au, etc.).

### 1.6 Reusable legacy reference implementations

- `legacy/PyRAI3MD/Molecule/pbc_helper.py`: `compute_cell(lattice)` (builds the 3×3 primitive-cell vectors from a,b,c,α,β,γ), `apply_pbc(coord, cell, pbc)` (fractional-coordinate modulo wrapping, selective per pbc axis).
- `legacy/PyRAI3MD/Dynamics/reset_velocity.py`: removes translation/rotation, projects onto the principal inertia axes, re-scales for kinetic-energy conservation, and includes `kabsch()` (for RMSD, phase-2 interface).
- `legacy/PyRAI3MD/Dynamics/Ensembles/thermostat.py`: `nose_hoover(traj)`, actually a **dual-bath Nose-Hoover Chain** (q1,q2,v1,v2, Yoshida-type exponential splitting); it is the direct blueprint for the new NHC.
- The legacy `ensemble.py` dispatcher already has `nve/nvt/nve_nvt/mixednvt` branches and a one-line `## TODO add barostat`.

## 2. Literature and Algorithm Survey

### 2.1 Periodic boundary conditions and the minimum-image convention

- The cell is represented by three basis vectors \(\mathbf{b}_1,\mathbf{b}_2,\mathbf{b}_3\); wrapping uses fractional coordinates \(\mathbf{s}=\mathbf{r}\,\mathbf{B}^{-1}\), applies `s %= 1` on the periodic axes, then converts back to Cartesian (legacy `apply_pbc` is exactly this method).
- Minimum image: \(\mathbf{r}_{ij}\) subtracts \(\mathbf{b}_k\,\mathrm{round}(\cdot)\) axis by axis (GROMACS eq.6). For a triclinic box, 14 nearest images must be considered.
- **Key point**: the framework's forces/gradients are computed by the backend (CP2K/PySCF/BAGEL/NN potentials, etc.); the PBC "minimum image / long-range electrostatics" should be handled inside the backend. On the framework side, `apply_pbc` is only responsible for coordinate wrapping (ensuring the coordinates sent to the backend are inside the cell), and for maintaining the cell for the barostat. This is consistent with the legacy division of labor.

### 2.2 NVT Thermostat

**Nose-Hoover (single bath)**, extended Hamiltonian, friction variable \(\xi\), momentum \(p_\xi\) (GROMACS eq.33-36):

\[
\ddot{\mathbf{r}}_i = \frac{\mathbf{F}_i}{m_i} - \frac{p_\xi}{Q}\dot{\mathbf{r}}_i,\qquad
\dot{p}_\xi = (T-T_0),\qquad Q=\frac{\tau_T^2 T_0}{4\pi^2}
\]

Conserved quantity \(H=\sum\frac{p_i^2}{2m_i}+U+\frac{p_\xi^2}{2Q}+N_f kT\xi\). Drawback: insufficient ergodicity for small/rigid systems.

**Nose-Hoover Chain (Martyna-Klein-Tuckerman 1992)**: uses M coupled baths in series to solve ergodicity (GROMACS eq.37-38):

\[
\dot{p}_{\xi_1}=(T-T_0)-p_{\xi_1}\tfrac{p_{\xi_2}}{Q_2},\quad
\dot{p}_{\xi_i}=\Big(\tfrac{p_{\xi_{i-1}}^2}{Q_{i-1}}-kT\Big)-p_{\xi_i}\tfrac{p_{\xi_{i+1}}}{Q_{i+1}}
\]

Standard integration uses Trotter/Liouville operator splitting + Yoshida higher-order sub-steps (legacy `nose_hoover` is the M=2 version of this structure). Reference: Martyna, Klein, Tuckerman, *J. Chem. Phys.* **97**, 2635 (1992).

**CSVR / stochastic velocity rescaling (Bussi-Donadio-Parrinello 2007)**: Berendsen plus a stochastic term, which keeps first-order exponential relaxation while giving the correct canonical distribution (GROMACS eq.32):

\[
\mathrm{d}K = (K_0-K)\frac{\mathrm{d}t}{\tau_T} + 2\sqrt{\frac{KK_0}{N_f}}\frac{\mathrm{d}W}{\sqrt{\tau_T}}
\]

In implementation, each step (or every few steps) rescales the velocities with a stochastic factor \(\alpha\): \(K'=\alpha K\), where \(\alpha\) is determined by two Gaussian random numbers and \(\tau_T\) (NAMD reports that only 2 random numbers per ~20 steps are needed, so the cost is extremely low, and it preserves holonomic constraints, needing no SHAKE recomputation). This is exactly CP2K's CSVR. Reference: Bussi, Donadio, Parrinello, *J. Chem. Phys.* **126**, 014101 (2007).

**Degrees-of-freedom counting** (general to thermostats): \(N_{df}=3N-N_c-N_{com}\), number of constraints \(N_c\), center-of-mass removal \(N_{com}=3\) (take 6 for rotation removal in vacuum). This directly relates to the phase-2 coupling of SHAKE/constraints with frozen atoms.

### 2.3 NPT Barostat

**Berendsen (weak coupling, first order)**: each step scales the coordinates and box vectors with the matrix \(\mu\) (GROMACS eq.44-46):

\[
\mu_{ij}=\delta_{ij}-\frac{n_{PC}\Delta t}{3\tau_p}\beta_{ij}(P_{0ij}-P_{ij}(t))
\]

\(\beta\) is the isothermal compressibility (water ~\(4.6\times10^{-5}\,\mathrm{bar}^{-1}\)). The average pressure is correct but it **does not generate a strict NPT ensemble** (the volume fluctuations are wrong); it is suitable for equilibration. Velocities are not scaled.

**Parrinello-Rahman (extended ensemble, true NPT)**: the box-vector matrix obeys (GROMACS eq.51-54):

\[
\frac{\mathrm{d}\mathbf{b}^2}{\mathrm{d}t^2}=V\mathbf{W}^{-1}\mathbf{b}'^{-1}(\mathbf{P}-\mathbf{P}_{ref}),\qquad
(\mathbf{W}^{-1})_{ij}=\frac{4\pi^2\beta_{ij}}{3\tau_p^2 L}
\]

The particle equations contain a matrix friction term \(\mathbf{M}\). It is usually combined with Nose-Hoover temperature control; it gives correct fluctuations, but is unstable when the initial value is far from equilibrium. Reference: Parrinello, Rahman, *J. Appl. Phys.* **52**, 7182 (1981).

**MTTK (Martyna-Tobias-Klein 1994, velocity-Verlet variant)**: introduces \(\epsilon=\tfrac13\ln(V/V_0)\), \(v_\epsilon=p_\epsilon/W=\dot V/3V\), \(\alpha=1+3/N_{dof}\) (GROMACS eq.59-66):

\[
\dot{\mathbf{r}}_i=\frac{\mathbf{p}_i}{m_i}+\frac{p_\epsilon}{W}\mathbf{r}_i,\quad
\dot{\mathbf{p}}_i=\mathbf{F}_i-\alpha\frac{p_\epsilon}{W}\mathbf{p}_i-\frac{p_{\xi_1}}{Q_1}\mathbf{p}_i
\]

\[
\dot{p}_\epsilon = 3V(\alpha P_{\text{kin}}-P_{\text{vir}}-P)-\frac{p_{\eta_1}}{Q'_1}p_\epsilon
\]

and also attaches an NHC (\(\eta\)) to the barostat degree of freedom. It **can only be combined with Nose-Hoover temperature control**; using Trotter splitting (GROMACS eq.67-73) yields a reversible, measure-preserving integrator whose coordinate/velocity updates contain \(e^{v_\epsilon\Delta t}\) and \(\sinh\) factors (eq.71-72). Reference: Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994).

**Stochastic cell rescaling (SCR)**: a *first-order* cell equation, so it carries no cell momentum, no `W`, and no `barostat_*` tensor at all. Bernetti & Bussi, *J. Chem. Phys.* **153**, 114107 (2020) give the isotropic form, which is what `barostat_mode: isotropic` implements:

\[
\mathrm{d}\varepsilon=-\frac{\beta_T}{\tau_P}\big(P_0-P_{\text{int}}\big)\,\mathrm{d}t+\sqrt{\frac{2k_BT\beta_T}{V\tau_P}}\,\mathrm{d}W,\qquad \varepsilon=\ln V
\]

Del Tatto, arXiv:2111.06402 (published as *Applied Sciences* **12**, 1139 (2022)) gives the full-tensor form that `barostat_mode: anisotropic` implements. Its Euler step (their eq 28, with the external stress isotropic so their \(\bm{\Sigma}\) vanishes) is

\[
\bm{\mu}=\mathbf{I}-\frac{\beta_T}{3\tau_p}\Big[\big(P_0\mathbf{I}-\mathbf{P}_{\text{int}}\big)-\frac{k_BT}{V}\mathbf{I}\Big]\Delta t+\sqrt{\frac{2\beta_T k_BT\Delta t}{3V\tau_p}}\,\mathbfcal{R}
\]

with \(\mathbfcal{R}\) a \(3\times3\) matrix of **nine independent** standard Gaussians, applied as \(\mathbf{h}\leftarrow\bm{\mu}\mathbf{h}\), \(\mathbf{q}_i\leftarrow\bm{\mu}\mathbf{q}_i\), and optionally \(\mathbf{p}_i^{T}\leftarrow\mathbf{p}_i^{T}\bm{\mu}^{-1}\) (their eq 29). **The framework takes the positions-only variant and does not rescale momenta**, which then obliges it to the eq 12 internal pressure (kinetic part \(N_{df}k_BT/3V\), not the instantaneous tensor). That follows the paper's own implementations -- its chapter-3 pseudocode parenthesises the momentum line and section 2.1 says rescaling momenta is "not in general a necessary feature" -- and it is NOT a stability fix: section 8.13 records the measurement that killed the idea of using it as one. Two details are the whole method and are easy to lose:

- the \(-(k_BT/V)\mathbf{I}\) term is the Ito drift correction the target distribution requires. Drop it and the box still relaxes, still looks healthy, and has the wrong volume fluctuations;
- the paper stores the box vectors in the **columns** of \(\mathbf{h}\) and acts on them from the Cartesian side, while this framework keeps them in the **rows** of `lattice_vector` and deforms by `lattice @ S` (`verlet._scale_cell`). So `S` is \(\bm{\mu}^{\mathsf T}\) here, and the momenta take \(\bm{\mu}^{-1}\) untransposed. On a cubic cell the two sides are the same matrix, which is why the test that pins this uses a sheared one.

\(\tau_p\) here is the **volume relaxation time**, not an oscillation period: the noise amplitude goes as \(1/\sqrt{\tau_p}\) and the drift as \(1/\tau_p\), so their ratio fixes \(\langle(\Delta\ln V)^2\rangle=k_BT\beta_{\text{sys}}/\langle V\rangle\) independently of both barostat parameters. That is the quantity to judge the scheme on, and it is also why an SCR example needs no long `barostat_tau`: the 10000 fs that `parrinello-rahman` and `mttk` require is the signature of a second-order cell integrating a stress noise, not of variable-cell MD in general.

Its own text states the pairing freedom that matters for the matrix in 8.13: "Stochastic cell rescaling can in principle be combined with any thermostat to sample the isothermal-isobaric ensemble." What it does **not** survive is a finite-time-step energy diagnostic: because the noise prefactor depends on the propagated variable, no effective energy is defined for the Euler integrator (arXiv:2111.06402 section 4), so `[ENERGY CONSERVATION]` says nothing about it either way.

**Important engineering constraint**: since GROMACS 5.1, the **MTTK + constraints** combination was **removed**, because of numerical-stability issues with the iterative constraint forces. This directly relates to the phase-2 compatibility of SHAKE with NPT.

**Unified integration framework**: all thermostats/barostats can be expressed uniformly with Liouville operators + Trotter splitting (\(iL=iL_1+iL_2+iL_{\epsilon,1}+iL_{\epsilon,2}+iL_{\text{NHC-baro}}+iL_{\text{NHC}}\)). This is the theoretical basis for bringing NVE/NVT/NPT and the phase-2 metadynamics bias into the same reversible splitter (Tuckerman's reversible measure-preserving framework; Shinoda et al. combined the MTK fluid-statics equations with the Parrinello-Rahman strain energy).

### 2.4 Specifics of coupling with NAMD (surface hopping)

The framework is non-adiabatic molecular dynamics; `apply_ensemble` runs before surface hopping and, since #310, **before** the single per-step `kinetic_energy()` sample: what the hop test and the thermostat share is the kinetic energy of the velocities this step ends with. Points to note:

- Kinetic energy/temperature is defined with the velocities of the **currently active potential-energy surface**; a surface hop changes the velocities abruptly (velocity adjustment/rescaling, `adjust/reflect` fields), and the thermostat's \(N_f kT\) term must be consistent with the post-hop velocities.
- Legacy has `nve_nvt`/`mixednvt`: the excited state runs NVE, and the ground state is thermostatted only after a delay. This kind of "switch ensemble by state" strategy must also be considered under NPT.
- The extended variables (\(\xi,p_\xi,\epsilon,p_\epsilon\), NHC chains) must be persisted with history/checkpoint, and **whether a surface hop resets the bath** is a semantic decision that needs to be made explicit.

## 3. Context Groundwork for Feasibility and Implementation Approach (no plan)

**Natural landing points are clear**: the four `operations.py` placeholders + the already-arranged main-loop order let thermostat/barostat/PBC land without changing the main-loop skeleton. Fixed-cell PBC (wrapping) has almost no prerequisite dependencies and can be done first; legacy already has a portable implementation.

**The three prerequisite chains for NPT** (they determine the implementation order and difficulty):

1. **Stress source**: `stress_tensor` is declared across the whole chain but never populated. NPT needs each backend (CP2K/PySCF/BAGEL/NN potentials) to actually return stress/virial, or the framework computes it from \(\sum \mathbf{r}_i\otimes\mathbf{F}_i\) (the latter is only rigorous for classical/pair potentials; for ab initio the backend must provide it). This is a cross-skill change; the workload and risk are concentrated here.
2. **Cell sync**: `SYNC_SYSTEM_FIELDS` must be extended to `cell/lattice_vector/pbc/volume`, otherwise the variable cell cannot be passed back to the backend. **(Refined by #216.)** The sync fields alone are NOT sufficient: `sync()` runs *after* the step's force evaluation, so a variable-cell step (`verlet._scale_cell` deforms the cell inside `verlet_i`) would reach the backend one step late. The authoritative per-evaluation channel is now `backend.calculate(cell=...)`, a snapshot frozen at the same moment as the geometry (`Calculator._snapshot_cell` → work item → `term_worker` payload); `sync()` only refreshes the task configs for the next step / checkpoint. "A sync field exists" ≠ "the backend used this step's deformed cell" (see `dev/methods/PBC_DEVELOPMENT.md` §3.1/§5.3).
3. **Units and keywords**: `units.py` needs bar/atm/compressibility→au; the adapter whitelist and config need to expose `thermostat/barostat/ensemble/pressure` and each algorithm's parameters (\(\tau_T\), chain length M, \(\tau_p\), \(\beta\), W/Q, isotropic/anisotropic).

**Persistence of the extended degrees of freedom**: \(\xi,p_\xi\) (NHC chains) and \(\epsilon,p_\epsilon\) (barostat) need cross-step storage. Whether the existing history/`update_system`/checkpoint mechanism carries these auxiliary variables needs to be confirmed (currently `System`/`Dynamics` have no corresponding fields).

**Unresolved architectural ownership**: the deleted `data/dynamics` sub-package claimed "algorithms belong to Dynamics Skills", while md.py's `apply_ensemble` sits at the framework integrator layer. Whether thermostat/barostat goes in the framework or in a skill is a design orientation that must be aligned first (it also affects the phase-2 ownership of metadynamics/constraints).

**Integrator consistency**: the velocity-Verlet full-step kinetic energy can directly compute \(P_{\text{int}}\) (MTTK requires velocity-Verlet), which is compatible with the existing two-stage verlet; but the Trotter sub-steps of MTTK/NHC will require splitting the current "single full velocity update" into an operator-splitting sequence, which is in the same vein as the exponential-splitting formulation of legacy `nose_hoover`.


### 3.4 The dissipation gate: `dissipate` delays the thermostat (#314)

`&md dissipate 1` changes when the thermostat is allowed to act, not which thermostat acts:

1. while the trajectory sits on an excited electronic state, both ensemble stages return
   before the thermostat branches, so the nuclei propagate on NVE;
2. the step on which the run is first found on the ground state is stamped once on
   `RuntimeState.dissipate_step` and never moved back, so a hop up does not close the gate
   again and a restarted segment does not restart the delay;
3. the thermostat takes over once `thermodelay` steps have been counted from that stamp, on
   the same step number the loop uses (the runtime counter lags it by one until the step is
   written, and both readings use the loop's number).

Two scope rules, both of them deliberate. The barostat is not gated: the ruling is NVE to NVT,
and a cell that stopped breathing in the middle of a run would be a different ensemble rather
than a delayed one. The gate is not half-applied: the `pre` and `post` Nose-Hoover half steps
are decided by the same reading, because one half kick without the other is a different
integrator. `dissipate` defaults to 0, and the default is proved rather than asserted - the
same thermal state with the keyword absent and with `dissipate: 0` produces bit-identical
velocities (`qualification/verification/test_md_dissipate.py`), so no shipped example pays
for it or changes its bytes. The observable in a product is `md.ensemble`: on a gated step the
thermostat columns carry no number at all, and from the step the delay ends on they do; the
`Ekin` of a gated step is the number the same structure has on NVE, which is what "propagates
on NVE" means in digits (measured in `pyrai3md/examples/md_dissipate.yaml`'s header).

An NVE run that declares `dissipate` is refused by name in `validate_ensemble`: it has no
thermostat to delay.

## 4. Phase-2 Interface Points (also handled in this round)

Many phase-1 mechanisms are the direct foundation of phase 2; the survey marks the following interfaces:

1. **CollectiveVariable already covers the CV types**: `distance/angle/torsion/coordination/volume/lattice_parameter` (field design preserved in the memo below; the former `pyrai3md/data/dynamics/cv.py` container has been deleted). The CV definitions for metadynamics can be rebuilt directly from this design; when `volume` is used as a CV it is the same physical quantity as the barostat's \(\epsilon\), so phase 1 should treat "cell/volume" as a first-class citizen, and phase-2 metadynamics can then bias it.

2. **The Metadynamics scaffold design is preserved**: hill_height/hill_width/deposition_interval/**bias_factor**/history (well-tempered capable) plus EnhancedSampling (metadynamics/umbrella/replica_exchange) are recorded in the memo below; the containers themselves have been deleted. Missing are CV evaluation and the bias-potential/bias-force algorithm. RMSD as a CV needs Kabsch: `data/geometry.py::kabsch` is now that implementation and delegates to `utilities/rmsd.py`, which the constraint subsystem's RMSD history potential also uses (CHANGELOG #264, closing D-10).

3. **The bias potential enters the virial → affects NPT**: the forces from metadynamics/bias/constraint potentials contribute to the internal pressure \(P_{\text{int}}\); when the phase-1 barostat computes the virial it must reserve an interface for the "external potential / bias force" term, otherwise the pressure will be inconsistent once phase 2 adds bias. The bias potential should also be included in the conserved quantity of the extended Hamiltonian.

4. **DOF counting for SHAKE and constraints**: the SHAKE/constraint field design (bond/angle/dihedral/cavity/compressor/frozen) is preserved in the memo below; the former `pyrai3md/data/dynamics/constraint.py` containers have been deleted. The phase-1 thermostat's \(N_f\) counting and the barostat's \(\alpha=1+3/N_{dof}\) both already depend on the constraint count \(N_c\); the landing point for SHAKE/constraint potentials is `operations.apply_constrain`, which delegates to `constraints/runtime.constrain_state` (`workflow/integrator/operations.py:156-187`). **Note that GROMACS removed MTTK+constraints for stability**; phase 2 must handle this conflict head-on in the plan design.

5. **Unified operator-splitting framework**: if phase 1 uses Liouville/Trotter splitting to implement NHC+barostat, then phase-2 metadynamics bias, SHAKE (as a constraint operator), and bias/constraint potentials can all be hooked into the same splitting sequence. It is recommended that phase 1 use this framework as the skeleton, leaving operator insertion slots for phase 2.

6. **CSVR is constraint-friendly**: the literature notes that CSVR preserves holonomic constraints and needs no SHAKE recomputation. If phase 2 uses SHAKE + NVT together, CSVR is less troublesome than Langevin; this trade-off is worth including when choosing the thermostat in phase 1.

**Memo: field design of the deleted `data/dynamics` containers** (removed 2026-09-25; the whole sub-package was pure data containers with zero consumers and duplicated the live `workflow/integrator/ensemble.py` + `schema/config.py` parameter model; when phase 2 lands, rebuild these fields directly in the schema (parameters) and integrator (algorithms) layers per the current "parameters in schema, algorithms in workflow" partition):

- `BondConstraint`: atoms (nbond, 2) index pairs; target bond lengths (Angstrom); force_constant default 10.0 (classic fbond).
- `AngleConstraint`: atoms (nangle, 3); target angles (degrees); force_constant default 0.005 (classic fangle).
- `DihedralConstraint`: atoms (ndih, 4); target dihedrals (degrees); force_constant default 1e-6 (classic fdihedral).
- `CavityConstraint`: radii (3,) along x/y/z; alpha (exponential factors); scale (pre-factors); shape per term ('ellipsoid'/'cuboid'); center_atoms; center_type ('mass'/'origin'/'geometric').
- `SHAKEParameters`: tolerance 1e-6; max_iterations 1000; constrained_bonds (nbond, 2).
- `Compressor`: target_ratio; n_steps (progressive compression).
- `Constraints` aggregate: bond/angle/dihedral/cavity/shake/compressor/frozen_atoms/groups ([nmol, natom_per_mol]).
- `CollectiveVariable`: name; type ('distance'/'angle'/'torsion'/'coordination'/'volume'/'lattice_parameter'); atoms; parameters dict (e.g. coordination cutoff); value.
- `Metadynamics`: enabled; cvs (names); hill_height (Hartree, default 0.01); hill_width per CV; deposition_interval (default 100); bias_factor (well-tempered); history (deposited hills, for restart).
- `EnhancedSampling`: method ('metadynamics'/'umbrella_sampling'/'replica_exchange'); metadynamics params; extensible parameters dict.
- Deleted as pure duplication (no memo needed): `Ensemble/Thermostat/Barostat/Dynamics` (superseded by `schema/config.py` dynamics keywords + `workflow/integrator/ensemble.py`).

## 5. Core References

- Martyna, Klein, Tuckerman, *J. Chem. Phys.* **97**, 2635 (1992) (Nose-Hoover Chain)
- Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994) (MTTK / NPT)
- Parrinello, Rahman, *J. Appl. Phys.* **52**, 7182 (1981) (Parrinello-Rahman barostat)
- Bussi, Donadio, Parrinello, *J. Chem. Phys.* **126**, 014101 (2007) (CSVR)
- Berendsen et al., *J. Chem. Phys.* **81**, 3684 (1984) (Berendsen thermostat/barostat)
- Tuckerman et al. (reversible measure-preserving integrator / Trotter splitting); Shinoda et al. (MTK + Parrinello-Rahman combination, adopted by LAMMPS `fix npt`)
- Bussi, Parrinello, *Phys. Rev. E* **75**, 056707 (2007) (Langevin / sampling-accuracy comparison)
- Bernetti, Bussi, *J. Chem. Phys.* **153**, 114107 (2020) (stochastic cell rescaling, an optional extra barostat)
- Frenkel & Smit, *Understanding Molecular Simulation* (NHC/NPT algorithms and DOF counting)
- GROMACS 2021 Reference Manual, "Molecular Dynamics" (all equations and complete update algorithms for Berendsen/CSVR/NH/NHC/Parrinello-Rahman/MTTK)

## 6. Missing-Property Backlog (phase-1 landing status)

> This section records, after the phase-1 (PBC/NVE/NVT/NPT framework layer)
> implementation, the stress/variable-cell-related property gaps and their landing
> status. The unified marker in the source code is `MISSING-PHASE1-STRESS`.
> Phase 1 only fills in the framework-side read/conversion/sync channels and
> places the markers; it **does not change the skills' stress computation**. The
> backend actually providing full 3×3 stress is left to later phases.

| # | Gap | Phase-1 status | Landing point |
|---|---|---|---|
| 1 | `Properties.stress_tensor` is declared but never populated; a default of zero would silently make the barostat wrong | Marked; the read path explicitly errors on "not populated" (returns None) and does not use zero | `schema/config.py::Properties.stress_tensor`; `data/core.py::active_stress()` |
| 2 | 6 skill schemas declare `stress_tensor` but the service layer has no writes | Each marked individually "stress computation and write-back to be added in a later phase" | `skills/{cp2k,pyscf,bagel,orca,molcas,openqp}_skill/schema/config.py` |
| 3 | The framework lacks a stress read entry point | Added `PyRAIState.active_stress()` (projects the active state `(3,3)` from the generic `(nstate,3,3)`); the docstring notes it depends on the backend populating it | `data/core.py::_active_stress()/active_stress()` |
| 4 | The history frame lacks an `active_stress` slot | Added in `_build_history_frame()`; old frames without the key return None, the pre-stage barostat half-step kick skips gracefully, and the post stage errors if stress is missing | `data/core.py::_build_history_frame()` |
| 5 | `units.py` lacks pressure/compressibility/time conversion | Filled in (bar/atm→Hartree/Bohr³, compressibility bar⁻¹→au, fs→au time) | `utilities/units.py` |
| 6 | `SYNC_SYSTEM_FIELDS` lacks `cell/lattice_vector/pbc/volume` | Filled in; the extended DOF (thermostat_dof/barostat_*) are intentionally not synced | `workflow/calculator.py::SYNC_SYSTEM_FIELDS` |
| 7 | The adapter whitelist lacks `thermostat/barostat/ensemble/pressure` and their parameters | Filled in the `MD_TYPES` whitelist | `schema/adapter.py` |
| 8 | full tensor requires the backend to return the complete stress tensor (not just the scalar pressure), and in generic `(nstate,3,3)` form | Explicitly written at each skill marker: the dimensions/units/"whole-system provider only" scope | same skill schemas as #2 |

**Scope and format constraints for backends providing stress (#2/#8 details)**: according to the dual-track parsing (Section 4A) `resolve_system_task` selection rules, **only the skill corresponding to the whole-system provider task must give stress**:

- Single-center, non-multiscale: `theory[0]` (the only task, i.e. the whole system).
- Single-center, multiscale: `theory[-1]` (the whole-system low-level calculation, the L(real) baseline of ONIOM).
- Multi-center (multiscale): `theory[-1]` (the whole-system calculation, providing the single global stress of the entire system).

High-level model-region tasks need not provide stress. When writing back, the generic `(nstate,3,3)` format must be maintained (giving per-state stress when excited-state stress exists); a single state may use `(3,3)` (the framework's `normalize_task_result` broadcasts to nstate), the units are Hartree/Bohr³, and the sign convention satisfies `P_int = P_kin - stress` (the `ensemble.py` internal-pressure convention; the sign must be re-checked when integrating a backend).

**Why phase 1 does not change the skills**: the actual computation of stress is independent work spanning backends (CP2K/PySCF/BAGEL/ORCA/Molcas/OpenQP/NN potentials), and each program's stress-output conventions differ; phase 1 first opens up the read/conversion/sync/persistence channels at the framework layer and places the markers, so that NVT and PBC/NVE are not blocked by missing stress, and NPT can run end-to-end once the backends provide stress (the current NPT test case is verified with an inline mock stress, see `qualification/verification/test_integrator.py`).

## 7. Phase 2: Skill-side Stress Production and Planning Route (landed)

> This section records the status after the three skills CP2K / xTB(GFN-FF) /
> LAMMPS filled in stress computation and reading; the corresponding source-code
> marker is updated from `MISSING-PHASE1-STRESS` to an implemented description.
> The interface and unit conventions reference `dev/skills/STRESS_DEVELOPMENT.md`;
> the landing is governed by the current state of the PyRAI3MD code.

**Planning route (stress capability derivation)**: `planning/resolver.py::_derive_stress` derives the `stress` capability in two scenarios, and reads `ensemble` and `optimize.optimize_cell` via `kernel/maestro.py::_extract_dynamics_config`:

- `jobtype=md` and `ensemble=npt` (the NPT barostat needs the internal pressure).
- `jobtype=optimize` and `optimize_cell=true` (variable-cell optimization; the framework optimizer then carries the six movable lattice components in its own optimized vector - see `dev/methods/CELL_OPT_DEVELOPMENT.md` section 5).

> Two different things can move a cell, and only one of them is this
> section's subject. The framework's own barostat integrates the cell and
> writes `lattice_vector`/`cell`/`volume` itself (`verlet._scale_cell`).
> `optimize_cell` adds the six lattice components to the framework's own
> optimizer, which is a different route to the same three fields: the search
> writes them once per accepted step beside the coordinates.
> The other is an engine that integrates the cell inside its own binary
> (LAMMPS `fix npt` with x/y/z cell terms during a `driver: skill` run): the
> framework cannot follow that step, so the engine reports the box its
> returned coordinates live in through the `fin_cell` channel and the
> framework re-derives the same three fields from it. A stale cell is not a
> cosmetic error: the next single point would be evaluated on the initial box
> with the deformed structure in it.

**Passing channel (boolean flag)**: `stress` does not enter the skill jobtype spelling (`CAPABILITY_TO_TOKEN` does not contain it); instead `data/core.py::_build_task_queue` injects `skill_config['stress']=True` according to the task's capabilities. This flag is orthogonal to the profile: MD/optimize still go through the gradient profile each step, with the stress computation layered on top.

**Skill-side keywords and reading** (non-blocking by default, missing values zero-filled, shape `(nstate,3,3)`):

- CP2K: when `config.stress`, add `STRESS_TENSOR ANALYTICAL` in `&FORCE_EVAL` and `&STRESS_TENSOR ON` in `&PRINT`; the parser reads the `STRESS TENSOR` block from the log (compression positive → negate so that tension is positive), normalized to GPa by the unit label (manifest `units.stress: gpa`). Excited-state stress has one route: `stress_method: num`, the external strain finite difference the CP2K skill drives itself over perturbed single points (`service/stress/`), which fills every state the TDDFPT run solved. The engine supplies no tensor for a TDDFT run -- neither `STRESS_TENSOR ANALYTICAL` nor `STRESS_TENSOR NUMERICAL` (both measured in the test environment) -- so there is no engine-side numerical option to choose between, and the retired `num1` spelling is refused by name.
- xTB(GFN-FF): periodic `--grad` automatically produces `gradlatt`; the parser takes the last record, `stress = G.T @ H / |det(H)|` (Hartree/Bohr³, tension positive, manifest `units.stress: hartree/bohr^3`). Ground state only.
- LAMMPS: when `config.stress`, the single-point segment adds `compute Pconf all pressure NULL virial` and `thermo_style custom ... c_Pconf[1..6]`; the parser reconstructs by `xx yy zz xy xz yz`, negates, and normalizes to bar (manifest `units.stress: bar`). Ground state only.

**Centralized framework unit conversion**: `workflow/calculator.py::convert_stress_to_atomic`, before `set_result`, uses `resolve_stress_unit_factor` to uniformly convert `stress_tensor` to Hartree/Bohr³ according to the `manifest.yaml::units.stress` of the skill corresponding to the structural-track provider task (`resolve_system_task`) (for the GPa/bar/atm factors see `utilities/units.py`). Therefore a skill only needs to output its native unit and declare it; sign/tensor-reconstruction and other program-specific handling still happen inside each skill's parser.

**Tests**: each skill's `tests/*_stress.yaml` (CP2K PBE ground state, CP2K PBE0 S1 placeholder, xTB GFN-FF ground state, LAMMPS ground state) for remote testing; CP2K/xTB use a 6.3 Å cubic 8-water box, and LAMMPS reuses the shared `tests/h2o.data` (20 waters, 10 Å box, constrained by the single `.data` fixture mechanism). Parser unit tests cover extraction, zero-filling, and unit normalization with mock output.

---

## 8. Variable-Cell Formulation As Implemented, And What Was Measured About It

Sep 28 2026. The integrator docstrings cite "docs section 4 / 6 / 10" for these
formulas; that numbering predates this review document's restructure, and this
section is where the material lives now.

Everything below was measured locally against an analytic stress with no engine
in the loop (`qualification/verification/test_ensemble_stability.py`, fixtures
imported from `test_integrator.py`) and against the products of the
`tmp_20260928_184646` qualification run
(`tools/analyze_npt_frames.py --root tmp_20260928_184646`). Where a measurement
contradicted what this document or a docstring had claimed, the claim was the
thing that changed.

### 8.1 The split step, and which tensor belongs to which frame

One framework step, in the order `md._run_trajectory` runs it:

```
apply_ensemble(pre) -> verlet_i -> apply_pbc -> evaluate -> verlet_ii
  -> velocity_correction -> apply_ensemble(post)
  -> apply_constrain (constrain_after_cell) -> kinetic_energy
  -> update_history
```

The cell is committed inside `verlet_i`: the drift is `B @ exp(v_eps dt)`, and
when that product carries upper-triangular components `_retriangularize` returns
the Cholesky representative `L` of its Gram matrix plus the orthogonal factor
`O = B^-1 L` that carries `B @ exp(v_eps dt)` onto `L`. `L` is the *same cell
shape*, so applying `O` to the coordinates and velocities is a change of
description, not of physics.

That is also the whole reason the extended cell DOF have to travel with it.
`barostat_strain`, `barostat_momentum` and `barostat_velocity` are tensors of the
simulation frame, and the increment they receive next is computed in the new one:
the post kick builds `g_eps` from the kinetic pressure of the rotated velocities
and from a stress the backend evaluated after this commit. Transported as
`T' = O^T T O` (row-vector convention, preserves symmetry, preserves the
determinant, and keeps `v_eps = p_eps / W` and `Tr(eps) = ln(V/V0)` consistent),
the run stops depending on the gauge. Measured, 60 steps of a shearing run
started from the same physical state written in a frame rotated about z:

| | largest lattice difference between the two descriptions |
|---|---|
| transport in place | 1e-37 A (they commit the same numbers) |
| transport removed | 3.2e-6 A, growing 7.9e-7 A per step |

The previous frame's stress and gradient need **no** transport, and that is worth
stating because the obvious guess was that they did. Both are captured *after*
their own step's commit, and nothing rotates the frame between that moment and
the half kick that reads them at the start of the next step. Reading them in the
current basis is therefore already correct; transporting them at read time would
rotate them twice.

### 8.2 The frame does turn, and the turn is not the defect

`_scale_cell`'s old comment claimed the per-step rotation was "far below the
split integrator's own truncation error" as if it were a rounding matter. The
measured increment is 5.7e-5 rad per step for a shearing anisotropic run, the
increments do not cancel, and 400 steps turn the representative frame by 1.30
degrees. The NPT examples of the qualification run turned 3.4-28.4 degrees over
2000 steps, which is the same rate. Meanwhile NVE and all three NVT examples turn
by exactly 0.000 (the lattice in their `md.xyz` never changes) -- so the rotation
is a property of the anisotropic cell dynamics, not of a bug. Pinned as
`TestAnisotropicCellStability.test_the_frame_turns_a_little_every_step_and_never_back`.

Two consequences. First, the transport of 8.1 cannot be waived as rounding.
Second, accumulated rotation is a poor *failure* criterion, which is why
`runner.BOX_NOTES` carries `cell_rotation_rad` as a note and `BOX_TOLERANCES`
does not judge it: a healthy run would cross any fixed accumulated bound simply
by running longer (a 2 ps record turns 0.11 rad; a 10 ps record would turn five
times that with nothing wrong). Volume and temperature separate the sick runs,
and they are what the verdict uses.

### 8.3 Stored velocities are peculiar momenta

Freeze every atom, give the cell a drift, run one `verlet_i`: the coordinates
move and `state.runtime.kinetic_energy` is exactly 0.0. The stored array is the
pecular (cell-relative) momentum, so `P_kin`, `T_inst` and `md.energies`' Ekin are
thermal energies already, and `verlet.py`'s `exp(-alpha v_eps dt/2)` velocity
coupling is the MTTK peculiar-momentum form of the same fact.

No "subtract the streaming velocity" helper is owed anywhere. Adding one where
the stored velocities are already peculiar would halve `P_kin` and under-
pressurize every NPT run without raising an error, which is why this is a gate
that ran before any fix and is pinned permanently
(`TestVelocityConventionAudit::test_a_pure_cell_drift_carries_no_temperature`).

### 8.4 The barostat mass, and a factor of sqrt(3)

`W = (V/beta)(tau_p/2pi)^2` (docs section 6 of the old numbering,
`ensemble._barostat_mass`) is the mass of a **scalar** volume coordinate, and it
is attached to the 3x3 tensor `p_eps`. The isotropic mode of that tensor is three
equal diagonal components sharing one strain rate, so its restoring stiffness is
three times the scalar case and the cell oscillates `sqrt(3)` fast: with the
volume spring at `1/beta` the period is `tau_p/sqrt(3)`, not `tau_p`. Measured on
a compressed box driven only by that spring: turning points at steps
289, 577, 866, 1149, 1436, 1721 -- period 576 steps against the 577.4 the formula
predicts, for `barostat_tau: 1000` at 1 fs. Pinned as
`TestIsotropicBarostatStability.test_volume_oscillation_period_is_tau_p_over_sqrt3`.

What `tau_p` therefore sets is the cell's response time to within that factor;
nothing about it needed a remote run to confirm, and the formula itself is right.

### 8.5 Symmetrization is a definition, not a truncation

`_extended_cell` symmetrizes `g_eps` before the kick and `p_eps` after it, and
`verlet._scale_cell` drops any antisymmetric part of the strain. Measured across a
400-step anisotropic run: `|p_eps - p_eps^T|` is exactly `0.000e+00` at every
step. Nothing is being thrown away per step, because nothing is ever created --
the cell's rotation is carried entirely by the gauge choice of 8.1, not by an
antisymmetric momentum. The proposed `g_eps_skew` / `p_eps_skew` diagnostic
columns would have been permanent zeros in `md.ensemble`, so they were not added.

### 8.6 The MTTK cell thermostat pumps the volume channel (open)

The one finding that is a defect and not a documentation correction, and it is
not the frame transport: it reproduces in **isotropic** mode, where the measured
frame rotation is exactly zero.

6 atoms, 300 K, box at the target pressure and 3 A out of shape equilibrium,
400 steps, same code path as Parrinello-Rahman except for the chain on the cell
momentum:

| barostat | V/V0 excursion | T_inst range | frame rotation |
|---|---|---|---|
| parrinello-rahman, isotropic | 1.0000-1.0000 | 300.0-300.0 K | 0 |
| mttk, isotropic | 1.0000-**1.2173** | **82-723 K** | 0 |
| mttk, isotropic, cell chain neutralized | 1.0000-1.0037 | 299.1-300.0 K | 0 |

`1.0037` is the offset MTTK's own `alpha = 1 + 3/N_df` asks for; everything above
it comes from the chain. The excursion shrinks monotonically as the chain is
slowed -- its `Q_1` is built from `thermostat_tau` (100 fs in every example) by
`ensemble._thermostat_masses`, against a cell that oscillates with a 577 fs period
(8.4) -- and at 2000 steps the sweep over `thermostat_tau` is:

```
natom=6   tau_T=  10 fs  DIVERGED (lattice_vector non-finite)
natom=6   tau_T=  50 fs  V/V0 0.8227-1.3408   T 64-681 K
natom=6   tau_T= 100 fs  V/V0 0.7853-1.2173   T 82-723 K
natom=6   tau_T= 1000fs  V/V0 0.8753-1.0904   T 161-532 K
natom=6   tau_T= 3000fs  V/V0 0.9943-1.0118   T 294-304 K
natom=60  tau_T=  10 fs  DIVERGED (lattice_vector non-finite)
natom=60  tau_T= 100 fs  V/V0 0.8974-1.2644   T 219-363 K
natom=60  tau_T= 3000fs  V/V0 0.9968-1.0091   T 298-303 K
```

At `thermostat_tau = 10 fs` the harness raises the message the remote
`md_lammps_npt_nhc_mttk` failure reported, verbatim: `lattice_vector contains a
non-finite value: [[nan, 0.0, 0.0], [nan, nan, 0.0], [nan, nan, nan]]`. That
example's death and `md_lammps_npt_nh_mttk`'s collapsed box (final axis 1.66 A,
volume 0.425 of target) are therefore this mechanism's, not the frame transport's.

**Resolved in #250, and the resolution had a false start worth recording.**
The parameterization was wrong as stated: `Q_1` for the cell chain is built from
`thermostat_tau` while the cell oscillates with a period of order `tau_p`, and a
chain faster than the motion it thermostats pumps that motion. The first fix
passed the right time constant into `_extended_cell` and changed nothing --
`_nhc_propagate` recomputes `Q` from `thermostat_tau` inside itself, so the value
the caller had built was the one that only reached `_diag`. Any time constant
handed to a chain therefore has to arrive through `_nhc_propagate(mass_tau_au=...)`,
and the reason this survived review is that the printed `Qbar` and the live `Q`
are computed in two places.

With `W` held at `tau_p` so only the chain mass varies, six atoms, conservative
drive, 2000 steps, sampled at every step, against an equipartition width of
1.4% for this box:

| cell-chain tau | max \|dV/V\| | rms dV/V | reading |
|---|---|---|---|
| 100 fs (`thermostat_tau`, the old behaviour) | 0.241 | 0.115 | ~11 sigma: pumped |
| 400 fs | 0.108 | 0.047 | still over-driven |
| **1000 fs (`barostat_tau`, what it now uses)** | **0.044** | **0.011** | ~2 sigma, the sampling it should have had |
| 4000 fs | 0.0007 | 0.0003 | frozen, not sampled |
| Parrinello-Rahman, no cell chain | 0.004 | 0.0024 | reference |

The cell chain now takes `barostat_tau` -- an existing field, no new knob -- and
the `expectedFailure` on
`TestIsotropicBarostatStability.test_mttk_at_equilibrium_keeps_the_box` is
removed, with the test asserting both a bound and the equipartition band. Earlier
numbers in this section quoted a sweep that moved `barostat_tau` and therefore
changed `W` and `Q` together; the table above is the separated measurement.

### 8.7 What the chain kicks are, and are not

The kick in `ensemble._nhc_substep` adds `force(k) * tau / 4` where
`force(0) = 2K - N_df kT`, with no division by `Q_k` -- which reads like a
textbook deviation. It is not one, and the audit is on record so nobody spends a
round "fixing" it: run one chain on one initial condition (N_df 177, chains of 1
and 3, 20000 steps, started at twice the target kinetic energy) against a hand-
transcribed canonical Tuckerman chain (`nhc_chain_audit.py`, run in that round's scratch tree, which is not kept in the repo) and
compare the sampled invariant measure, which is the only thing a thermostat has to
get right:

| chain | framework `<K>` / (N_df kT / 2) | canonical transcription |
|---|---|---|
| 1 link | 1.000 | 0.877 |
| 3 links | 1.002 | 0.941 |

The framework's version is the one that samples the canonical distribution.

### 8.8 Preconditions recorded rather than fixed

Stated because they condition every number above and none of them is the
integrator's:

- All eight NPT examples start at `P_int = -1784 bar` on the first frame. With
  `compressibility: 4.5e-5` and `barostat_tau: 1000` that offset is by design an
  8% volume swing, so the first frames of those runs are a driven transient, not
  equilibrium.
- `barostat_tau` is read by the berendsen path too (`ensemble._berendsen`
  forms `mu = I - (dt/(3 tau_p)) beta (P_ext - P_int)`), so "raise tau_p" is
  not a parrinello-rahman-only change: it weakens a berendsen barostat by the
  same factor. The five anisotropic examples now run at 10000 fs and the three
  berendsen ones deliberately stay at 1000, because those finish and a slower
  weak coupling would only make them track the target less.
- The `[PBC]` trace in every example's log says the pair cutoff (10 A) exceeds
  half the shortest axis (4.2 A). The virial the barostat is fed is therefore not
  the virial of an equilibrated TIP3P box, and no cell judgement can be stricter
  than that fact.
- `tools/refresh_water_fixture.py` normalizes velocities to N_df = 180 (298.16 K)
  while the framework reports 303.2 K over its own N_df = 177. Same momenta, two
  counts; the 1.7% temperature difference is that gap, not heating.

The force field and the box are the user's decision and were left alone, per the
ruling recorded in #243 and #245; the fixture files themselves are read-only to
the backend (#142).

### 8.9 A retraction: the "deviatoric instability" of #248 was the test's, not the framework's

#248 reported that Parrinello-Rahman anisotropic grows a shape strain of ~6.2 and
trips its own gate within ~2400 steps on a **valid equation of state with zero
noise**, and concluded that the anisotropic channel amplifies. That conclusion was
wrong and is withdrawn. The drive used for it,
`test_ensemble_stability.surrogate_stress`'s deviatoric term
`K_dev * dev(B^T B / det^(1/3))`, is a function of the cell that is not the
gradient of any energy: a force with curl pumps an oscillator that is otherwise
perfect, which is exactly the growth that was measured.

Repeating the same runs against `conservative_stress` -- the central-difference
gradient of the declared scalar `cell_potential` -- removes it: eps_max 0.125
instead of 6.19, volume 1.002 instead of 1.11, no gate. The framework's own
credentials on that drive are what a Hamiltonian split step must have: the
extended energy `KE + sum(p_eps^2)/(2W) + U(B)` moves by 4e-9 (small tilt) to
2.9e-5 (tilt 1.0 A) over 400 steps, and the step is reversible -- forward 200,
flip every momentum, forward 200, flip, and the lattice and the unwrapped
positions return to within 3e-7 A. (Comparing WRAPPED coordinates shows a ~10 A
"failure" that is only the image bookkeeping; see `_unwrapped`.) Both facts are
pinned as `TestStepIsHamiltonian`.

Two consequences. First, the shape drive in these tests is now conservative by
construction, and any future claim of barostat instability has to survive that
law before it can be aimed at the integrator. Second, the anisotropic examples in
`tmp_20260928_223009` still diverge, and the honest attribution is the drive: a
virial computed with a 10 A cutoff in an 8.4 A cell is not the gradient of a cell
potential either (the neighbour list rebuilds as the box changes), and the
framework integrates it faithfully. Since the box is not being changed, the
framework's obligation is to stop on its own terms: `_divergence_report` now also
bounds the CUMULATIVE deformation against the oldest retained frame
(`MAX_CUMULATIVE_*`), so a cell that has left its own starting size by a factor of
two fails as `CellDivergenceError` naming the axis and the ratio, instead of
reaching 20238 A and being aborted by LAMMPS with `Triclinic box skew is too
large`. The per-step bounds alone could not catch a creep.

### 8.10 What the cumulative bound is measured against, and what a free shape mode costs

The bound added in #250 did not fire once in `tmp_20260929_071340`, and that was
two separate mistakes in one clause.

- **Reference.** It compared the cell against the oldest frame still in the
  output history. `history_limit` trims, so that reference sits a few steps back
  in the past, and a box creeping for two thousand steps never looked like 2x to
  it. `system.barostat_strain` is the quantity that is BY DEFINITION measured
  against the cell the run started with (Tr(eps) = ln(V/V0)), it is carried by
  the frame transport of 8.1, and it costs nothing extra to read. The bound now
  reads eps.
- **Value.** ln(2) was a guess. This round supplies the measurement: the three
  berendsen examples that FINISH 2000 steps reach 1.03x on a cell axis and 1.19x
  in volume, while every parrinello-rahman and mttk example had already passed
  1.7x on an axis by the frame before it died (nh_pr 0.568x, nh_mttk 0.488x,
  csvr_pr 5.66x). ln(1.5) sits 2.1x clear of the healthiest finished run and
  fires hundreds of steps before the death. It is also read on all nine eps
  components, not the diagonal only: in the triangular gauge a shape collapse
  accumulates off-diagonal first, and a bound the shear walks past is not a
  bound. A 10 ps anisotropic run would need this re-measured -- shape drift is
  diffusive, so the margin scales as sqrt(5).
- **What actually killed nh_pr** was not the lattice: it was
  `thermostat_dof = [-4.2e58, inf]` hitting the System validator, one step after
  a log block reading T_inst 708 K with a per-step velocity scale of 0.854. The
  chain was saturated -- it can remove at most 27% of the kinetic energy per
  step and the collapsing box was pouring heat in faster. Reading that block was
  only possible after the md.ensemble gutter fix (#250, #251 -- a value
  that fills its 20-character field used to fuse with its neighbour and shift
  every column after it), and it is why the cell bound has to fire EARLY: by
  the time the thermostat overflows the run is over, and the traceback names
  the victim, not the cause.

A free shape mode was then priced directly (`aniso_noise_replay.py`, same scratch-tree convention).
The recorded deviatoric stress of the HEALTHY nh_berendsen run -- off-diagonal
rms ~10 kbar, which is the normal statistical-mechanical size for 60 atoms in
592 A^3, sqrt(N)*(r.F)/V -- was fed to the framework's own cell loop, with only
the isotropic part replaced by the harness's equilibrium value so the volume
channel starts balanced. Everything that moved is therefore the noise's doing:

| law | mode | tau_p (fs) | worst axis drift | worst volume |
|---|---|---|---|---|
| parrinello-rahman | anisotropic | 1000 | 51-70% | ~0% |
| mttk | anisotropic | 1000 | 60-80% | 1.0% |
| berendsen | anisotropic | 1000 | 25% | 0.0% |
| parrinello-rahman | isotropic | 1000 | 0.3% | 1.0% |

The volume does not move and the shape does, which is the remote signature; and
isotropic coupling of the same law does not move at all. A liquid cell has no
shear modulus, so its shape coordinates are free particles hit by a random
force, and the displacement is the DOUBLE time-integral of that force -- drift
without any defect in the integrator. That is the strongest statement this
experiment supports, and its limit should be read with it: the harness has
`gradient = 0`, so it contains none of the liquid's instantaneous shear
restoring force, and it therefore overstates the drift. The conservative-drive
harness of 8.9, which does have a shape-restoring term, is stable over 2000
steps. Whether the remote 30-50% is *all* noise or partly drive still needs a
stress that responds to shape -- and now, at last, readable columns to see it in.

### 8.11 The cumulative bound is a test instrument, and two families of failure are kept apart

`control.debug` decides what a violated bound does to a trajectory. The rule
is not "some runs get checked and some do not" -- every run is measured every
step -- but a distinction the first cut of #250 blurred:

- **A commit that cannot be computed** stops every run, debugging or not.
  A non-positive determinant, a non-finite lattice, a non-finite extended
  coordinate (particle chain, cell chain, strain, cell momentum) are
  arithmetic: there is no successor state to propagate. They now stop at the
  writer, in `verlet._divergence_report` and `ensemble`'s `check_finite`, with
  the step, the frame `T_inst`, the velocity scale and the drive in the
  message. Before this, the same event stopped two fields later in a pydantic
  validator, which could only say `thermostat_dof contains a non-finite value`
  -- the victim, not the cause. That traceback is what tmp_20260929_071340
  reported for six NPT tasks.
- **A bound that is a judgement** stops only a run that was started as a
  test. The accumulated strain limit (`MAX_CUMULATIVE_LOG_STRAIN = ln 1.5`) is
  an opinion about how far a box has travelled from its own start; a
  production run that trips it keeps integrating and the monitor says so on
  the schedule the run already chose for itself. That schedule is
  `md.direct`/`md.buffer`, shared with the trajectory writer through
  `workflow.records.output.record_due` -- one predicate deciding which steps this run
  has records at, and therefore which steps it can be commented on at. The
  shipped examples (direct 1, buffer 100) get twenty-one monitor lines over
  2000 steps, the same twenty-one frames `md.ensemble` holds, because a line
  about the box belongs next to the numbers it was said with. A violation that
  persists between two recorded steps carries `(since step N)`, so the cadence
  never costs the reader the moment it began. `WARN_CUMULATIVE_LOG_STRAIN =
  0.22` (1.246x) is the earlier, say-it-once line: the healthiest runs that
  finish 2000 steps reach 0.174 in log volume, so 0.22 sits above all of them
  and below the bound that interrupts.

The single owner of that policy is `pyrai3md/workflow/records/diagnostics.py`: the
bounds, the two error classes (`EnsembleDivergenceError` subclasses
`CellDivergenceError`, so existing handlers keep catching both), the message
texts, the emission ladder, and a cumulative record on
`RuntimeState.diagnostics` that survives `to_dict`/`from_dict`/`copy` -- a
dispatched child has to be able to report what it watched. `md.py` closes the
run with one summary line per check, so "nothing was reported" and "nothing
was watched" are different in the log. The qualification reads those lines back
out of `<title>.log` into the record (`runner._check_run_diagnostics`), because
a passing run shows only its stdout tail and a silent monitor is no monitor.

Deliberate deviations from the plan, with reasons: the warn emission stays at
the commit site rather than moving into `operations.verification`, because the
commit is where the numbers exist and a per-step call would put new work on the
NVE path that `test_integrator` pins byte-for-byte. And the schema validator is
left as the backstop rather than being taught to raise the integrator's error,
because `data/core.py` must not import `workflow` -- the writer-side
`check_finite` catches the case one step earlier instead.

`control.debug` is also a global for skills, which is a name collision the
design had to pass through: `lammps_skill` already spells its per-step
artifact switch `debug` (`schema/config.py:259`), and the skill merge keys
framework fields into the skill config by *name*. It is therefore applied as
layer 0 of `_merge_one_skill_config`, after every other layer: a debugged run
debugs every module under test, and an undecided run leaves a skill's own
setting alone -- an unset `control.debug` must not silence
`lammps_skill/tests/ff_gradient_persistent.yaml`, which asks for its artifacts
on its own behalf. `test_diagnostics` pins both directions.

What the 09:19 round measured about the bound itself, after `md.ensemble`
became readable: the five anisotropic parrinello-rahman and mttk examples
crossed it at steps 374-855 while their **volume** was still 0.88-1.00 of the
start and only one **axis** had gone to 0.488-0.667. Shape, not size. That is
why the three berendsen examples on the same box finish: weak coupling has no
cell momentum for a deviatoric stress to accumulate into.

### 8.12 What csvr × parrinello-rahman measured, and why the fix was not "use mttk"

`tmp_20260929_121646` isolated an interaction by holding everything but one
pair fixed. Same box, same fixture, same 1 fs step, same `barostat_tau` 10000:

| thermostat | barostat | temperature_drift | volume_drift |
|---|---|---|---|
| nose-hoover | parrinello-rahman | 0.261 | 0.085 |
| nose-hoover-chain | parrinello-rahman | 0.176 | 0.180 |
| csvr | berendsen | 1.754 | 0.083 |
| csvr | parrinello-rahman | **13.466** | 0.064 |

The last row is the outlier, and the box is not the explanation: its volume
drifted 6.4% and its frame turned 0.034 rad, both smaller than the rows that
passed. What happened is thermal -- `T_inst` reached 4339 K against a 300 K
target, which the qualification reported as
`[NPT STABILITY] temperature_drift 13.466 exceeds tolerance 3.000`.

Mechanically, parrinello-rahman and mttk differ in one thing: mttk puts the
cell coordinates themselves in a Nosé-Hoover chain, parrinello-rahman leaves
them Hamiltonian. The barostat force is `V (alpha p_kin - sigma - p_ext I)`,
and `p_kin` is a quadratic form in the particle velocities -- so a thermostat
that rescales the velocities *stochastically* modulates the drive of a degree
of freedom that has no way to dump what that modulation feeds it. CSVR's own
authority makes the race unwinnable: with `c = dt/tau_T = 0.01` it removes about
one percent of the excess per step, while the cell returns what it absorbed on a
10 ps time constant. Deterministic thermostats on the same barostat sit at 0.18
and 0.26, and CSVR with berendsen (a contraction, no cell momentum) sits at
1.75, so neither factor alone explains it; the pair does.

The example was then renamed `md_lammps_npt_csvr_mttk`, **and that rename was
wrong**: `schema/validate.py` has for some time refused `mttk` with anything but
a Nose-Hoover thermostat, so the renamed file could not be loaded at all. It
passed every local test suite because nothing in them put an example through
`validate()`; on a cluster it would have died at Step 4 of `pmd.run` with
`barostat 'mttk' cannot be combined with thermostat 'csvr'`. 8.13 keeps the
diagnosis above (a stochastic thermostat feeding a cell that cannot give the
energy back is a real mechanism, and `csvr × parrinello-rahman` measured it at
4339 K) and replaces the conclusion: the formulation the pairing needs is not
`mttk` under a different name, it is the barostat that was derived for it.

### 8.13 The pairing is literature, not a free grid: scr, and what refuses what

`thermostat` and `barostat` are two keywords, which invites reading the whole
\(3\times3\) set of their values as a menu. It is not a menu. Each barostat
comes from a derivation that says what the particles' heat bath is, and the
framework now records that in one table instead of letting every combination
through to a run:

| `barostat` | thermostats it accepts | why |
|---|---|---|
| `berendsen` | `nose-hoover`, `nose-hoover-chain`, `csvr` | a relaxation scheme (JCP 81, 3684 (1984)); it claims no ensemble, so it constrains nothing |
| `parrinello-rahman` | `nose-hoover`, `nose-hoover-chain` | extended-Lagrangian cell dynamics (JAP 52, 7182 (1981)); the chain on the particle momenta is in the derivation |
| `mttk` | `nose-hoover`, `nose-hoover-chain` | MTTK *is* Parrinello-Rahman with a chain on the cell momentum as well (JCP 101, 4177 (1994)); it does not exist with a stochastic particle bath |
| `scr` | `nose-hoover`, `nose-hoover-chain`, `csvr` | first-order stochastic cell equation; "can in principle be combined with any thermostat" (JCP 153, 114107 (2020)) |

`off` × variable cell is refused by the pre-existing rule that `npt` needs a
thermostat, which matters more here than anywhere: a cell that is driven and a
system that cannot shed the work means the drift `[ENERGY CONSERVATION]`
reports (if anything reported it) would be the algorithm, not the physics -- and nothing reports it: `_check_energy_conservation` judges only an `nve` trajectory with the thermostat off, and any other leg writes a note saying it was not judged instead of printing a number that should not be read as conservation.

The table lives in `schema/validate.py` as `BAROSTAT_THERMOSTATS` with a
per-entry reason in `BAROSTAT_PAIRING_REASON`, and it is enforced at
**validation** time, not as a `debug` monitor during a run. That placement was
the user's ruling and it is the right one: a combination with no derivation is
not something a trajectory should be allowed to spend twenty thousand steps
discovering, and a warning buried in a log is how `csvr_mttk` would have come
back as "the example is flaky".

The one gap this round exposed about the framework's own gates is worth keeping
in words, because it is not an ensemble question at all: **no local test loaded
a shipped example through the config pipeline.** `parse_input → build_config →
normalize → validate` is four calls, all of them offline, and every one of the
26 examples survives them (asserted in
`qualification/verification/test_stochastic_cell_rescaling.py`). A file can be
consistent with its documentation, consistent with its neighbours, and
inconsistent with its own schema, and only that pipeline can tell.

`scr` itself is measured in the places a claim about a stochastic scheme should
be checked, all against the source rather than against the implementation:

| what | expected | measured |
|---|---|---|
| zero-noise reduction | `_berendsen`'s `mu` plus \((\beta\,\Delta t/3\tau_p)(k_BT/V)\mathbf{I}\), nothing else | ratio 1.000000, off-diagonal 0.0 |
| noise amplitude | \(\mathrm{var}(\mathrm{d}\ln V)=2k_BT\beta_T\Delta t/(V\tau_p)\) | 0.9988 (isotropic), 0.9953 ± 0.007 (tensor), 40000 draws |
| which side acts on the cell | eq 9's \(D_{ai,bj}\propto\delta_{ab}\sum_g h_{gi}h_{gj}\) on a **sheared** lattice | residual 0.036 RMS; the transposed guess 0.330 |
| stationary width on the surrogate | \(k_BT\beta_{\text{sys}}/\langle V\rangle\) | 1.19 at \(\tau_p=100\) fs, 1.43 at 1000 fs |
| reproducibility | same `gl_seed`, same lattice; different seed, different lattice | exact |

The stationary width sits *above* the analytic value because the surrogate box
holds six atoms: with \(N_{df}=15\) the thermostat's own kinetic pressure noise
is \(\sqrt{2/N_{df}}=36\%\) and it enters the cell drive through `p_kin`. That
term is \(O(1/N)\), so the number to watch in the remote `md_lammps_npt_csvr_scr`
run is this ratio on 60 atoms, not the toy's.

Why the example is `barostat_mode: isotropic`, measured rather than assumed: the
tensor form injects nine Gaussians, so it shears the cell as well as breathing
it, and a liquid has no shear modulus to put that back. On the 6-atom surrogate
with the 500 bar shape spring the anisotropic branch tripped the per-step axis
bound of `workflow/records/diagnostics.py` inside 12000 steps at every \(\tau_p\) tried
from 100 to 1000 fs, while the isotropic branch left the committed lattice
exactly proportional (its upper triangle and its frame rotation both read
0.0). That is consistent with what the tensor paper validates it on -- Lennard
Jones, ice Ih, gypsum, gold, all crystals -- and with why it recommends the
scalar form for fluids. The anisotropic branch stays implemented and unit
tested; it is not what a water box at 1 bar should run. Note the consequence
for what the remote round actually covered: the shipped example is the ONLY
isotropic-barostat run in the set (the other seven NPT examples are all
`anisotropic`), so tmp_20260929_151626 exercised the scalar branch of `_scr` on
a real engine and not the tensor one. The tensor branch's evidence is local
only -- the drift reduction to `_berendsen`, the eq 9 diffusion tensor on a
sheared lattice, and the fluctuation-dissipation ratio -- and whether it
survives the frame gate on a 60-atom box is a question this round did not
answer.

**What the tensor branch actually died of, and the diagnosis that was wrong first.**
The run (`scr_aniso_probe`, anisotropic, `direct 1 / buffer 1`, `debug` off so
the bounds only warn) stopped at step 1306 on the frame-turn bound, but the cell
was the consequence: the temperature went 311 K -> 897 K by step 200 and 3.3e4 K
by step 1250 while the volume still read ~570 A^3, and only then did the box
inflate. The first reading, written here in full because it is the one a reader
would reach for, blamed eq 29b's `p^T <- p^T mu^-1`: a random `mu` has
`E[ln s_min(mu)] < 0` at any amplitude (-2.13e-3 per step at this run's
sigma = 1.45e-3, measured from 200000 draws), so inverting it inflates the
velocity norm at +4.25e-3 per step in ln T against a measured +3.73e-3, and a
CSVR thermostat whose entire authority is `c = dt / tau_T` = 1% of the excess
cannot out-negotiate that.

**That diagnosis did not survive the next run.** Reimplementing the same scheme
without the momentum map left the failure in place: same heating from step ~150,
and a cell that then collapsed instead of inflating. Ten-fold `tau_p` (which
cuts the noise as `1/sqrt(tau_p)`) slowed it by roughly the factor the noise
amplitude predicts -- 6469 steps versus 1306 -- without removing it. So the map
is one real energy channel and not the cause.

The reading that fits all four measurements is the one the source itself
implies by its choice of test systems. The tensor form injects ~1.5e-3 of
*shear* strain per 1 fs step; a liquid has no shear modulus, so that strain is
not stored elastically but dissipated into the particles' thermal motion, and a
hotter box drives the collapse. Crystals, which is what arXiv:2111.06402
validates the anisotropic variant on (Lennard Jones, ice Ih, gypsum, gold), do
have the modulus, and that is why the same equation is sound there and not here.
The bulk channel carries no such penalty, which is why the isotropic branch of
the identical scheme measured a thermal residue of 0.141 on this same box.

Two consequences are now structural rather than editorial. `schema/validate.py`
refuses `scr` with `barostat_mode: anisotropic` with the measured reason in the
message, and `ensemble._scr` implements only the scalar form -- the branch is
removed, not disabled, because nothing reachable may configure it and a test on
unreachable code is a false assurance. And the toy surrogate is officially
useless for this class of question: its stress is an analytic function of the
cell, so it has no dissipation channel at all. Both variants ran side by side on
it to within 1 K over 4000 steps at `k_dev` of 500, 50 and 0 bar. That is worth
saying plainly because it is the local evidence that *did not* predict the
failure, and the reason the probe had to go to the cluster at all.

**What tmp_20260929_151626 said.** All eight variable-cell examples passed in
both input modes, with the new `md_lammps_npt_csvr_scr` the cleanest trace of
the set:

| example (thermostat x barostat) | volume_drift | temperature_drift | rotation |
|---|---|---|---|
| csvr x **scr** (isotropic, tau_p 1000) | **0.055** | **0.250** | 0.000 |
| csvr x berendsen (aniso, tau_p 1000) | 0.083 | 1.754 | 0.062 |
| nose-hoover x pr (aniso, tau_p 10000) | 0.080 | 0.399 | 0.027 |
| nose-hoover-chain x pr | 0.176 | 0.305 | 0.019 |
| nose-hoover x mttk | 0.113 | 0.374 | 0.038 |
| nose-hoover-chain x mttk | 0.167 | 1.422 | 0.033 |
| nose-hoover x berendsen | 0.116 | 0.597 | 0.059 |
| nose-hoover-chain x berendsen | 0.159 | 1.741 | 0.070 |

The pairing worth reading is the first two rows: same thermostat, same box,
same 2 ps, same `tau_p` = 1000 fs, only the barostat swapped -- and the
temperature drift the cell leaves behind drops from 1.754 to 0.250. For
reference the same thermostat on the un-thermostatted cell had measured 13.466
in the 12:16 round. What that comparison does NOT settle: `scr` here also runs
`barostat_mode: isotropic` where `berendsen` runs `anisotropic`, so part of the
0.055 and the exactly-zero rotation belong to the mode rather than to the
scheme, and the honest reading of the 7x is "a stochastic cell with the noise
the ensemble asks for leaves far less heat in the particles than weak coupling
does", not "scr is 7x better than berendsen".

**The shipped configuration, re-measured after the tensor branch was removed**
(`tmp_20260929_162523`, positions-only scalar `scr`): `csvr_scr`
volume_drift 0.076, temperature_drift 0.141, frame rotation 0.000, and the
round finished **PASS with 70/70 and no errors** -- including
`test_subprocess_smoke`, the one unit case that had errored in the two rounds
before it, for a reason this work does not own and cannot explain. One
observation from those three rounds is worth keeping: `nhc_pr`'s final volume
was 487.9714241792 A^3 in the 15:16 round, 485.8363330603 in the 15:57 round,
and 487.9714241792 again both in a hand-run repeat and in the 16:25 round,
while every other example matched to ten digits throughout. Three of four
measurements agree, so the odd one is a 0.4% excursion of one anisotropic
second-order run, not a code difference -- but it does bound how tight a
variable-cell tolerance may be: any `volume_drift` threshold below ~1% would be
reading that noise.

**What tmp_20260929_173945 added: the same eight, read back from the frame
artifacts rather than from the verdict lines.** The round finished 109 tasks /
70 pass / 0 fail / 0 error / 39 skip with `status.json: pass`, and its sixteen
`[NPT STABILITY]` lines are byte-identical to the 16:25 round. What that round
adds is the check that a status word cannot carry: reading `Epot/Ekin/Etot`,
`T_inst`, `V` and the per-frame `Lattice=` back out of the 24 run directories
shows the two input modes agree to 0.00e+00 on every example (21 frames and 21
cell records in each), and it puts two numbers on the shipped `scr`
configuration that had only been asserted. The recorded `mu` off-diagonals
(`mu_xy`, `mu_xz`, `mu_yx`, `mu_zy`) measure exactly 0.0 across all 21 frames,
so the isotropic branch breathes and does not shear; and `scr_sigma` runs
2.422e-3 to 2.546e-3 against the analytic single-step width of 2.51e-3 that the
example header derives from `V`, `kT`, `beta_T`, `dt` and `tau_p`. The energy
budget keeps the shape the ledger left it: +0.038 to +0.226 Ha on the
weak-coupling and second-order pairings, -0.008 on `nhc_pr`, and -0.034 on
`csvr_scr`, which loses rather than gains. The box-hysteresis channel (the last
structural suspect for that gain) was closed separately; see 11.12 of
`LAMMPS_SKILL_DEVELOPMENT.md`.

And one thing no round in this file measures: `var(ln V)` needs independent
samples of the volume, and 21 frames spaced 100 steps apart on a 1 ps
relaxation time is about two. The number 8.13 predicted (5.6% width) is
therefore still untested on a liquid; measuring it needs a dedicated run
(`direct 1, buffer 1`, tens of thousands of steps). That job is not going to be
run -- the user closed the ensemble work here with the qualification suite
green -- so 5.6% stays what it has always been in this document: a derivation,
not a measurement.
