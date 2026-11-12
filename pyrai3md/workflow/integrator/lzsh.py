"""
======================================================================
This module provides the Landau-Zener surface hopping (LZSH) main
flow of the NAMD workflow (dev/methods/LZSH_DEVELOPMENT.md).

The algorithm replaces the Zhu-Nakamura improved Landau-Zener
probability of GSH with the textbook single-passage Landau-Zener
formula (L. Landau, Phys. Z. Sowjetunion 1932, 2, 46; C. Zener,
Proc. R. Soc. Lond. A 1932, 137, 696) and derives the diabatic
force difference from the recorded energy gap series instead of
rebuilding diabatic forces from trajectory gradients:

    internal conversion   adiabatic (NAC) representation: the
                          recorded gap is the avoided-crossing gap
                          with the minimum 2V12; the curvature of
                          the gap at the gate step (three-point
                          second time derivative) carries the force
                          difference and the crossing velocity
                          cancels: gamma = de^2 / (4*sqrt(de*gddot))
    intersystem crossing  spin-diabatic representation: the S/T
                          energies and gradients carry no SOC, so
                          the recorded gap is the spin-diabatic gap
                          itself (linear zero crossing, no avoided
                          curvature); the gap rate of change from
                          the signed three-point central difference
                          carries the force difference times the
                          velocity: gamma = SOC^2 / gdot

Both kernels consume the state energies and the timestep only (no
gradients, no NAC vectors). Internal conversion keeps the GSH
three-step gate (the gap at t-dt is the smallest of the three
steps, below the threshold and classically allowed); intersystem
crossing drops the minimum-gap gate (the spin-diabatic gap has no
avoided minimum to detect) and keeps the gap threshold and the
energy conservation constraints only. Both probabilities are
evaluated at t-dt (the nuclear propagation lags the electronic
properties by one step, the ZNSH convention). The hop probability
conventions follow GSH: internal conversion hops with P_LZ,
intersystem crossing hops with 1 - P_LZ (the non-parallel PES
assumption turns the Landau-Zener survival into a hop).

Like FSSH and GSH the algorithm is a class-based integrator:
LzshIntegrator is initialized once per trajectory by the
surface_hopping factory (surface_hopping.py) and called once per
timestep. Initialization parses the static trajectory configuration
exactly once; every step reads the current data through the
PyRAIState official interfaces (state.result, the read-only history
view, system.state_spin, system.soc_pair_map) and writes back
through state.update_system() and state.update_dynamics().

LZSH keeps the GSH density semantics: the density matrix is rebuilt
as the pure state of the final active state every step (no coherent
propagation, no decoherence). Velocity adjustment uses the shared
isotropic rescaling policy of the tsh_helper module for every hop:
the kernels produce no coupling vector (design document section 18).

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.units import CM_TO_HARTREE, EV_TO_HARTREE
from .tsh_helper import (adjust_velo_isotropic, energy_of as _energy_of,
                          soc_value, write_pure)


# ---- Snapshot and results readers (shared semantics with gsh) ----

def _snapshot_results(snapshot):
    """Extract the results dict of one history snapshot."""
    if not snapshot:
        return {}
    return snapshot.get('results') or {}


def _snapshot_system(snapshot):
    """Extract the system dict of one history snapshot."""
    if not snapshot:
        return {}
    return snapshot.get('system') or {}



# ---- Landau-Zener probability kernels ----

def internal_conversion_prob(i, active, energy, ep, epp, ekinp,
                             dt, gap_au):
    """Landau-Zener internal conversion probability for one pair.

    The adiabatic avoided-crossing gap carries the coupling through
    its minimum (de = 2V12 at the crossing) and the force
    difference through its curvature: near the crossing
    delta(q) = sqrt(df^2*q^2 + 4C^2) with delta''(0) = df^2/de, so
    the three-point second time derivative of the recorded gap
    gddot = v_c^2*df^2/de and the crossing velocity cancels in
    gamma = V12^2/(v_c*df) = de^2/(4*sqrt(de*gddot)).

    Parameters
    ----------
    i : int
        Target state index (0-based, i != active).
    active : int
        Active state index (0-based).
    energy, ep, epp : np.ndarray
        State energies (nstate,) in Hartree at t, t-dt and t-2dt.
    ekinp : float
        Nuclear kinetic energy of the previous step (Hartree).
    dt : float
        MD time step (au).
    gap_au : float
        IC hopping gap threshold (Hartree, converted by the caller).

    Returns
    -------
    float
        Hopping probability (evaluated at the gate step t-dt).
    """
    # Three-step gate (GSH semantics): the gap at t-dt must be the
    # smallest of the three steps, below the threshold, and the hop
    # must be classically allowed.
    del_e = np.abs([energy[i] - energy[active],
                    ep[i] - ep[active],
                    epp[i] - epp[active]])
    etotp = ep[active] + ekinp
    ex = (ep[i] + ep[active]) / 2
    if np.argmin(del_e) != 1 or del_e[1] > gap_au or etotp - ex < 0:
        return 0.0

    de = del_e[1]
    if de <= 0 or dt <= 0:
        return 0.0

    # Gap curvature at the gate step (three-point second time
    # derivative); a non-convex window (noise, flat gap or a
    # crossing outside the gate) degrades to zero probability.
    gddot = (del_e[0] - 2 * del_e[1] + del_e[2]) / dt ** 2
    if gddot <= 0:
        return 0.0

    # Classical Landau-Zener: gamma = V12^2/(v_c*df) with
    # V12 = de/2 and df = sqrt(de*gddot)/v_c (velocity cancels).
    gamma = de ** 2 / (4 * np.sqrt(de * gddot))
    return float(np.exp(-2 * np.pi * gamma))


def intersystem_crossing_prob(i, active, energy, ep, epp, ekinp,
                              soc_value, dt, gapsoc_au):
    """Landau-Zener intersystem crossing probability for one pair.

    The S and T states are spin-diabatic: their computed energies
    carry no SOC, so the recorded gap is the spin-diabatic gap
    itself (a linear zero crossing for crossing surfaces, a
    constant for parallel ones). The gap has no avoided curvature,
    so the force difference comes from the signed three-point
    central difference gdot = |deltaE(t) - deltaE(t-2dt)|/(2*dt)
    = v_c*df (exact at every crossing position for linear gaps);
    the crossing velocity cancels in gamma = SOC^2/(v_c*df).

    No minimum-gap gate: the spin-diabatic gap has no avoided
    minimum to detect; only the gap threshold and the energy
    conservation constrain the evaluation (both at t-dt, the same
    evaluation moment as the internal conversion kernel).

    Parameters
    ----------
    i : int
        Target state index (0-based, i != active).
    active : int
        Active state index (0-based).
    energy, ep, epp : np.ndarray
        State energies (nstate,) in Hartree at t, t-dt and t-2dt.
    ekinp : float
        Nuclear kinetic energy of the previous step (Hartree).
    soc_value : float or None
        Spin-orbit coupling of the pair (cm-1); None when the pair
        is undeclared or the value was not computed (zero
        probability).
    dt : float
        MD time step (au).
    gapsoc_au : float
        ISC hopping gap threshold (Hartree, converted by the
        caller).

    Returns
    -------
    float
        Hopping probability (evaluated at t-dt).
    """
    # Constraints at t-dt only (no minimum-gap gate): the absolute
    # gap below the threshold and the hop classically allowed.
    signed = [energy[i] - energy[active],
              ep[i] - ep[active],
              epp[i] - epp[active]]
    etotp = ep[active] + ekinp
    ex = ep[i]
    if abs(signed[1]) > gapsoc_au or etotp - ex < 0:
        return 0.0
    if soc_value is None:
        return 0.0

    # V12 = SOC (hbar = 1 in Hartree atomic units).
    v12 = float(soc_value) * CM_TO_HARTREE
    if v12 <= 0 or dt <= 0:
        return 0.0

    # Signed-gap central difference (exact for linear crossings at
    # every crossing position); parallel or frozen gaps (gdot == 0)
    # carry no crossing and degrade to zero probability.
    gdot = abs(signed[0] - signed[2]) / (2 * dt)
    if gdot <= 0:
        return 0.0

    # Non-parallel PES convention (same as GSH): the Landau-Zener
    # survival becomes the hop, P = 1 - P_LZ; the weak-coupling
    # limit 2*pi*gamma recovers the perturbative result.
    gamma = v12 ** 2 / gdot
    return float(1.0 - np.exp(-2 * np.pi * gamma))


# ---- Integrator ----

class LzshIntegrator:
    """Landau-Zener surface hopping integrator.

    Callable state -> state object initialized once per trajectory
    by the surface_hopping factory. Every step evaluates the
    Landau-Zener internal-conversion and intersystem-crossing
    probabilities from the recorded trajectory (history API),
    samples one hop with an energy-ordered cumulative draw, adjusts
    the velocity through the shared isotropic policy and writes the
    pure-state density of the final active state back.

    Raises
    ------
    ValueError
        If the state lacks a pmd.dynamics configuration.
    """

    def __init__(self, state):
        dynamics = getattr(getattr(state, 'pmd', None), 'dynamics',
                           None)
        if state is None or state.pmd is None or dynamics is None:
            raise ValueError(
                "LzshIntegrator requires a state with pmd.dynamics")

        system = state.pmd.system

        # ---- Static trajectory configuration (parsed once) ----
        self.nstate = int(state.nstate)
        self.maxh = int(dynamics.maxh)
        self.adjust = int(dynamics.adjust)
        self.reflect = int(dynamics.reflect)
        self.gap_au = float(dynamics.gap) * EV_TO_HARTREE
        self.gapsoc_au = float(dynamics.gapsoc) * EV_TO_HARTREE
        self.dt = float(dynamics.size)

        # Per-state 2S spin list; an empty or short list falls back
        # to the all-same-spin behavior (every pair takes the
        # internal-conversion path), matching trajectories without
        # explicit spin annotations.
        state_spin = list(system.state_spin or [])
        self.state_spin = (state_spin
                           if len(state_spin) >= self.nstate else [])

        # Declared SOC pairs (row index lookup for results['soc']).
        self.soc_index = {}
        for idx, pair in enumerate(system.soc_pair_map or []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                key = tuple(sorted((int(pair[0]), int(pair[1]))))
                self.soc_index[key] = idx

        control = getattr(state.pmd, 'control', None)
        self.verbose = int(getattr(control, 'verbose', 2)) \
            if control is not None else 2

    def _same_spin(self, i, active):
        """True when both states share the spin multiplicity."""
        if not self.state_spin:
            return True
        return self.state_spin[i] == self.state_spin[active]

    def _soc_value(self, i, active, soc_values):
        """SOC value (cm-1) of a pair, or None (undeclared/missing)."""
        return soc_value(self, i, active, soc_values)

    def _write_pure(self, state, active):
        """Write the pure-state density of one state and return."""
        return write_pure(self, state, active)

    def __call__(self, state):
        """Run one LZSH timestep (state -> state).

        Evaluates the Landau-Zener hopping probabilities of every
        non-active state, samples one hop with an energy-ordered
        cumulative draw and writes the final active state, density
        and (adjusted) velocity back through the official
        interfaces.
        """
        if state is None or state.pmd is None:
            return state

        nstate = self.nstate
        active = int(state.current_state)
        if nstate < 1 or not 0 <= active < nstate:
            return state

        result = state.result
        results = result if result is not None else {}
        energy = _energy_of(results, nstate)
        if energy is None:
            return state

        # Warm-up: the probability kernels need the energies of
        # t-dt and t-2dt (the GSH iter > 2 condition).
        if state.history_size < 3:
            return self._write_pure(state, active)

        # The MD loop records the fully computed step into the
        # history before the surface hopping decision, so the
        # newest snapshot duplicates the current step: t-dt and
        # t-2dt live one index deeper than the raw offset.
        prev_snap = state.get_history_step(-2, copy=False)
        before_snap = state.get_history_step(-3, copy=False)
        previous = _snapshot_results(prev_snap)
        before = _snapshot_results(before_snap)
        prev_system = _snapshot_system(prev_snap)

        ep = _energy_of(previous, nstate)
        epp = _energy_of(before, nstate)
        if ep is None or epp is None:
            return self._write_pure(state, active)

        mass_au = state.mass_au
        velocities = np.asarray(state.velocities, dtype=np.float64)
        if mass_au is None or velocities.size == 0:
            return self._write_pure(state, active)
        velocities = velocities.reshape(-1, 3)

        # Previous-step kinetic energy (the history snapshots carry
        # no runtime block; recompute from the snapshot velocity).
        velo_prev = prev_system.get('velocities')
        if velo_prev is None:
            return self._write_pure(state, active)
        velo_prev = np.asarray(velo_prev,
                               dtype=np.float64).reshape(-1, 3)
        ekinp = float(np.sum(0.5 * mass_au[:, None]
                             * velo_prev ** 2))

        # ---- Per-state hopping probabilities ----
        z = np.random.uniform(0, 1)
        g = np.zeros(nstate)
        for i in range(nstate):
            if i == active:
                continue
            if self._same_spin(i, active):
                p = internal_conversion_prob(
                    i, active, energy, ep, epp, ekinp,
                    self.dt, self.gap_au)
            else:
                soc_value = self._soc_value(
                    i, active, previous.get('soc'))
                p = intersystem_crossing_prob(
                    i, active, energy, ep, epp, ekinp,
                    soc_value, self.dt, self.gapsoc_au)
            g[i] = p

        # ---- Energy-ordered cumulative sampling ----
        stateindex = np.argsort(energy)
        gsum = 0.0
        event = 0
        new_state = active
        for j in range(nstate):
            gsum += g[stateindex[j]]
            nhop = abs(stateindex[j] - active)
            if gsum > z and 0 < nhop <= self.maxh:
                new_state = int(stateindex[j])
                event = 1
                break

        # ---- Velocity adjustment of the hop ----
        frustrated = 0
        vt = velocities
        if event == 1:
            # The kernels produce no coupling vector: every hop
            # takes the isotropic rescaling policy (design document
            # section 18, the GSH intersystem-crossing leg).
            vt, frustrated = adjust_velo_isotropic(
                energy[active], energy[new_state], velocities,
                mass_au)
            if frustrated == 1:
                new_state = active

        # ---- Write-back (pure-state density every step) ----
        update = {}
        if event == 1:
            # The isotropic policy keeps the velocity on a
            # frustrated hop; written back verbatim (GSH semantics).
            update['velocities'] = np.asarray(vt).tolist()
        self._write_pure(state, new_state)
        if new_state != active:
            update['current_state'] = new_state
        if update:
            state.update_system(update)

        if self.verbose >= 2:
            stateorder = np.argsort(energy).argsort()
            lines = []
            for k in range(nstate):
                mult = (self.state_spin[k] + 1
                        if self.state_spin else 0)
                lines.append('    %-5s %-5s %-5s %12.8f'
                             % (k + 1, mult, stateorder[k] + 1,
                                g[k]))
            print('\n    Random number:           %12.8f\n'
                  '    Accumulated probability: %12.8f\n'
                  '    state mult  level   probability\n%s\n    '
                  % (z, gsum, '\n'.join(lines)))

        return state


def run_lzsh(state):
    """Run one LZSH timestep through a disposable integrator.

    Thin compatibility entry for standalone calls: builds a
    temporary LzshIntegrator from the state configuration and
    executes one step. Trajectories use the surface_hopping factory
    instead, which initializes the integrator once and reuses it.

    Parameters
    ----------
    state : PyRAIState
        State with the recorded timestep and the current electronic
        properties.

    Returns
    -------
    PyRAIState
        The updated state (unchanged when the configuration is
        missing).
    """
    if state is None or state.pmd is None:
        return state
    if getattr(state.pmd, 'dynamics', None) is None:
        return state
    return LzshIntegrator(state)(state)
