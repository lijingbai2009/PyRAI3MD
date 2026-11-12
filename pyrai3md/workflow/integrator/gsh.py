"""
======================================================================
This module provides the global (generalized) surface hopping (GSH)
main flow of the NAMD workflow
(dev/methods/GSH_DEVELOPMENT.md).

The algorithm is based on the Zhu-Nakamura theory (C. Zhu, Phys.
Chem. Chem. Phys. 2014, 16, 25883-25895 for internal conversion;
C. Zhu, Phys. Chem. Chem. Phys. 2020, 22, 11440-11451 for the
intersystem crossing adaptation). Ported from the legacy
PyRAI2MD Dynamics/Propagators/gsh.py with two bug fixes and a
numerical-protection hardening pass:

    velocity direction   the legacy hop rescaling consumed the
                         never-filled active-state row of the
                         approximate NAC array (always the ones
                         placeholder); the port uses the target
                         state's approximate NAC vector instead
    frustrated density   the legacy density was written on the hop
                         target even when the hop was frustrated;
                         the port keeps both the active state and
                         the density on the departure state
    numerical guards     frozen coordinates (r == r-2dt), zero
                         norms, zero gaps and zero SOC couplings
                         degrade to zero probability instead of
                         dividing by zero (Phase 6 style: no clip,
                         no reset); the EQ7/8 inversion masks
                         per-component frozen motion so strictly
                         one-dimensional trajectories stay
                         NaN-free; a missing SOC pair returns zero
                         probability instead of terminating the
                         process

Like FSSH the algorithm is a class-based integrator: GshIntegrator
is initialized once per trajectory by the surface_hopping factory
(surface_hopping.py) and called once per timestep. Initialization
parses the static trajectory configuration exactly once; every step
reads the current data through the PyRAIState official interfaces
(state.result, the read-only history view, system.state_spin,
system.soc_pair_map) and writes back through state.update_system()
and state.update_dynamics().

GSH keeps the legacy density semantics: the density matrix is
rebuilt as the pure state of the final active state every step (no
coherent propagation, no decoherence). Velocity adjustment is
shared with FSSH through the tsh_helper module: internal-conversion
hops rescale along the target-state approximate NAC vector,
intersystem-crossing hops carry no approximate NAC vector and use
the isotropic rescaling policy (design document section 18).

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...data.geometry import continuous_coordinates
from ...utilities.units import CM_TO_HARTREE, EV_TO_HARTREE
from .tsh_helper import (adjust_velo, adjust_velo_isotropic,
                          energy_of as _energy_of, soc_value, write_pure)


# ---- Snapshot and results readers ----

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



def _gradient_of(results):
    """Read the per-state gradient array, or None.

    The MD workflow stores results['gradient'] as a
    (nstate_computed, natom, 3) array; rows beyond the computed
    target states stay absent and gate the probability kernels.
    """
    gradient = results.get('gradient')
    if gradient is None:
        return None
    gradient = np.asarray(gradient, dtype=np.float64)
    if gradient.ndim != 3:
        return None
    return gradient


def _matrix_of(rows):
    """Assemble a (natom, 3) float matrix, or None."""
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0 or rows.size % 3 != 0:
        return None
    return rows.reshape(-1, 3)


def _continuous(matrix, system):
    """Undo the periodic wrap of a coordinate matrix, when one applies.

    The history stores the main-image coordinates a backend was evaluated
    on, so an atom that crossed a cell face between two frames differs
    from its previous position by roughly one box length instead of the
    few hundredths of an Angstrom it really moved. GSH builds its
    curvature estimate out of exactly those differences, so it has to work
    on the continuous path, rebuilt from the crossing count the wrap
    recorded (dev/methods/PBC_DEVELOPMENT.md stage C).

    The reading itself is `data/geometry.py::continuous_coordinates`, the
    same one a region cut-out and a recorded-frame check take: three fields
    off whichever spelling of the system the caller holds, one set of guards.
    """
    return continuous_coordinates(matrix, system)


# ---- Zhu-Nakamura probability kernels ----

def internal_conversion_prob(i, active, energy, ep, epp, grad, gpp,
                             r, rp, rpp, ekinp, mass_au, gap_au):
    """Zhu-Nakamura internal conversion probability for one pair.

    Ported formula-by-formula from the legacy
    internal_conversion_prob (EQ 1-8 of C. Zhu, Phys. Chem. Chem.
    Phys. 2014, 16, 25883-25895) with numerical guards: every
    degenerate input (frozen coordinates, zero gap, zero forces,
    missing gradient rows) degrades to zero probability instead of
    dividing by zero.

    Parameters
    ----------
    i : int
        Target state index (0-based, i != active).
    active : int
        Active state index (0-based).
    energy, ep, epp : np.ndarray
        State energies (nstate,) in Hartree at t, t-dt and t-2dt.
    grad : np.ndarray
        Per-state gradients (nstate, natom, 3) of the current step.
    gpp : np.ndarray
        Per-state gradients (nstate, natom, 3) two steps back.
    r, rp, rpp : np.ndarray
        Nuclear coordinates (natom, 3) at t, t-dt and t-2dt.
    ekinp : float
        Nuclear kinetic energy of the previous step (Hartree).
    mass_au : np.ndarray
        Atomic masses (natom,) in atomic units.
    gap_au : float
        IC hopping gap threshold (Hartree, converted by the caller).

    Returns
    -------
    (float, np.ndarray)
        Hopping probability and the approximate NAC vector
        (natom, 3); the vector is zero whenever the probability is
        zero.
    """
    nac_zero = np.zeros_like(r)

    # Crossing detection: the gap at t-dt must be the smallest of
    # the three steps, below the threshold, and the hop must be
    # classically allowed (legacy early-stop conditions).
    del_e = np.abs([energy[i] - energy[active],
                    ep[i] - ep[active],
                    epp[i] - epp[active]])
    etotp = ep[active] + ekinp
    ex = (ep[i] + ep[active]) / 2
    if np.argmin(del_e) != 1 or del_e[1] > gap_au or etotp - ex < 0:
        return 0.0, nac_zero

    de = del_e[1]
    if de <= 0:
        return 0.0, nac_zero
    if grad is None or gpp is None:
        return 0.0, nac_zero
    if i >= grad.shape[0] or active >= grad.shape[0]:
        return 0.0, nac_zero
    if i >= gpp.shape[0] or active >= gpp.shape[0]:
        return 0.0, nac_zero

    # EQ 7 / EQ 8 approximate forces; fully frozen coordinates
    # (r == rpp) carry no curvature information and degrade to
    # zero. Per-component frozen motion (e.g. a strictly one-
    # dimensional model trajectory) is legal: the inversion applies
    # to the moving components only, frozen components contribute
    # zero force instead of a 0/0 NaN.
    dcoord = r - rpp
    if not np.any(dcoord):
        return 0.0, nac_zero
    begin_term = np.zeros_like(dcoord)
    moving = np.abs(dcoord) > 0
    begin_term[moving] = -1.0 / dcoord[moving]

    arg_min = np.argmin([i, active])
    arg_max = np.argmax([i, active])

    f1_grad_manip_1 = grad[arg_min] * (rp - rpp)
    f1_grad_manip_2 = gpp[arg_max] * (rp - r)
    f_ia_1 = begin_term * (f1_grad_manip_1 - f1_grad_manip_2)

    f2_grad_manip_1 = grad[arg_max] * (rp - rpp)
    f2_grad_manip_2 = gpp[arg_min] * (rp - r)
    f_ia_2 = begin_term * (f2_grad_manip_1 - f2_grad_manip_2)

    # Approximate nonadiabatic (vibronic) coupling direction.
    nac = (f_ia_2 - f_ia_1) / (mass_au[:, None] ** 0.5)
    norm = np.sum(nac ** 2) ** 0.5
    if norm <= 0:
        return 0.0, nac_zero
    nac = nac / norm

    # EQ 4 / EQ 5 mass-weighted force magnitudes.
    f_a = np.sum((f_ia_2 - f_ia_1) ** 2 / mass_au[:, None]) ** 0.5
    f_b = np.abs(np.sum((f_ia_2 * f_ia_1)
                        / mass_au[:, None])) ** 0.5
    if f_a <= 0 or f_b <= 0:
        return 0.0, nac_zero

    # EQ 1 / EQ 2 (dE = 2Vx, hbar = 1 in Hartree atomic units).
    a_squared = (f_a * f_b) / (2 * de ** 3)
    b_squared = (etotp - ex) * (f_a / (f_b * de))
    sign = np.sign(np.sum(f_ia_1 * f_ia_2))
    if a_squared <= 0:
        return 0.0, nac_zero

    # EQ 3 improved Landau-Zener probability.
    denom = b_squared + (np.abs(b_squared ** 2 + sign)) ** 0.5
    if denom <= 0:
        return 0.0, nac_zero
    b_in_denom_term = 2.0 / denom
    p = float(np.exp(-(np.pi / (4 * a_squared ** 0.5))
                     * b_in_denom_term ** 0.5))
    return p, nac


def intersystem_crossing_prob(i, active, energy, ep, epp, gradp,
                              ekinp, soc_value, mass_au, gapsoc_au):
    """Zhu-Nakamura intersystem crossing probability for one pair.

    Ported formula-by-formula from the legacy
    intersystem_crossing_prob (C. Zhu, Phys. Chem. Chem. Phys.
    2020, 22, 11440-11451; equations adapted from the 2014 paper).
    The non-parallel PES assumption of the legacy code is kept (the
    legacy parallel branch was hard-coded off), so the probability
    is always 1 - P_LZ.

    Parameters
    ----------
    i : int
        Target state index (0-based, i != active).
    active : int
        Active state index (0-based).
    energy, ep, epp : np.ndarray
        State energies (nstate,) in Hartree at t, t-dt and t-2dt.
    gradp : np.ndarray
        Per-state gradients (nstate, natom, 3) of the previous step.
    ekinp : float
        Nuclear kinetic energy of the previous step (Hartree).
    soc_value : float or None
        Spin-orbit coupling of the pair (cm-1); None when the pair
        is undeclared or the value was not computed (zero
        probability, replacing the legacy process termination).
    mass_au : np.ndarray
        Atomic masses (natom,) in atomic units.
    gapsoc_au : float
        ISC hopping gap threshold (Hartree, converted by the
        caller).

    Returns
    -------
    float
        Hopping probability.
    """
    del_e = np.array([energy[i] - energy[active],
                      ep[i] - ep[active],
                      epp[i] - epp[active]])
    etotp = ep[active] + ekinp
    ex = ep[i]
    if (np.argmin(np.abs(del_e)) != 1
            or np.abs(del_e[1]) > gapsoc_au or etotp - ex < 0):
        return 0.0
    if soc_value is None:
        return 0.0

    # V12x2 = 2 * SOC (hbar = 1 in Hartree atomic units).
    v12x2 = 2.0 * float(soc_value) * CM_TO_HARTREE
    if v12x2 <= 0:
        return 0.0
    if gradp is None:
        return 0.0
    if i >= gradp.shape[0] or active >= gradp.shape[0]:
        return 0.0

    # EQ 7 / EQ 8 approximate forces from the previous-step
    # gradients.
    f_ia_1 = gradp[active]
    f_ia_2 = gradp[i]

    # EQ 4 / EQ 5 mass-weighted force magnitudes.
    f_a = np.sum((f_ia_2 - f_ia_1) ** 2 / mass_au[:, None]) ** 0.5
    f_b = np.abs(np.sum((f_ia_2 * f_ia_1)
                        / mass_au[:, None])) ** 0.5
    if f_a <= 0 or f_b <= 0:
        return 0.0

    # EQ 1 / EQ 2.
    a_squared = (f_a * f_b) / (2 * v12x2 ** 3)
    b_squared = (etotp - ex) * (f_a / (f_b * v12x2))
    sign = np.sign(np.sum(f_ia_1 * f_ia_2))
    if a_squared <= 0:
        return 0.0

    # EQ 3 improved Landau-Zener probability (non-parallel PES).
    denom = b_squared + (np.abs(b_squared ** 2 + sign)) ** 0.5
    if denom <= 0:
        return 0.0
    b_in_denom_term = 2.0 / denom
    p_lz = float(np.exp(-(np.pi / (4 * a_squared ** 0.5))
                        * b_in_denom_term ** 0.5))
    return 1.0 - p_lz


# ---- Integrator ----

class GshIntegrator:
    """Global (generalized) surface hopping integrator.

    Callable state -> state object initialized once per trajectory
    by the surface_hopping factory. Every step evaluates the
    Zhu-Nakamura internal-conversion and intersystem-crossing
    probabilities from the recorded trajectory (history API),
    samples one hop with an energy-ordered cumulative draw, adjusts
    the velocity through the shared tsh_helper layer and writes the
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
                "GshIntegrator requires a state with pmd.dynamics")

        system = state.pmd.system

        # ---- Static trajectory configuration (parsed once) ----
        self.nstate = int(state.nstate)
        self.maxh = int(dynamics.maxh)
        self.adjust = int(dynamics.adjust)
        self.reflect = int(dynamics.reflect)
        self.gap_au = float(dynamics.gap) * EV_TO_HARTREE
        self.gapsoc_au = float(dynamics.gapsoc) * EV_TO_HARTREE

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
        """Run one GSH timestep (state -> state).

        Evaluates the Zhu-Nakamura hopping probabilities of every
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

        # Warm-up: the probability kernels need the energies,
        # gradients and coordinates of t-dt and t-2dt (the legacy
        # iter > 2 condition).
        if state.history_size < 3:
            return self._write_pure(state, active)

        # The MD loop records the fully computed step into the
        # history before the surface hopping decision, so the
        # newest snapshot duplicates the current step: t-dt and
        # t-2dt live one index deeper than the raw offset (the
        # legacy energy1/coord1 and energy2/coord2 semantics).
        prev_snap = state.get_history_step(-2, copy=False)
        before_snap = state.get_history_step(-3, copy=False)
        previous = _snapshot_results(prev_snap)
        before = _snapshot_results(before_snap)
        prev_system = _snapshot_system(prev_snap)
        before_system = _snapshot_system(before_snap)

        ep = _energy_of(previous, nstate)
        epp = _energy_of(before, nstate)
        grad = _gradient_of(results)
        gradp = _gradient_of(previous)
        gpp = _gradient_of(before)
        r = _matrix_of(state.pmd.system.coordinates)
        rp = _matrix_of(prev_system.get('coordinates'))
        rpp = _matrix_of(before_system.get('coordinates'))
        if (ep is None or epp is None or r is None or rp is None
                or rpp is None):
            return self._write_pure(state, active)

        # Difference against a frame is only a displacement if both frames
        # speak the same continuity: put the three matrices on the
        # continuous path before the curvature estimate reads them.
        r = _continuous(r, state.pmd.system)
        rp = _continuous(rp, prev_system)
        rpp = _continuous(rpp, before_system)

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
        nac_by_target = {}
        for i in range(nstate):
            if i == active:
                continue
            if self._same_spin(i, active):
                p, nac = internal_conversion_prob(
                    i, active, energy, ep, epp, grad, gpp,
                    r, rp, rpp, ekinp, mass_au, self.gap_au)
                if p > 0:
                    nac_by_target[i] = nac
            else:
                soc_value = self._soc_value(
                    i, active, previous.get('soc'))
                p = intersystem_crossing_prob(
                    i, active, energy, ep, epp, gradp, ekinp,
                    soc_value, mass_au, self.gapsoc_au)
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
            # Internal-conversion hops carry the target state's
            # approximate NAC vector (legacy fix: the active-state
            # row was never filled); intersystem crossings carry
            # none and take the isotropic rescaling policy.
            nac = nac_by_target.get(new_state)
            if nac is not None:
                vt, frustrated = adjust_velo(
                    energy[active], energy[new_state], velocities,
                    mass_au, nac, self.adjust, self.reflect)
            else:
                vt, frustrated = adjust_velo_isotropic(
                    energy[active], energy[new_state], velocities,
                    mass_au)
            if frustrated == 1:
                # Legacy fix: both the active state and the density
                # revert to the departure state (the legacy density
                # stayed on the hop target).
                new_state = active

        # ---- Write-back (pure-state density every step) ----
        update = {}
        if event == 1:
            # The legacy flow returns the adjusted velocity even on
            # a frustrated hop (reflect semantics); kept as-is.
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


def run_gsh(state):
    """Run one GSH timestep through a disposable integrator.

    Thin compatibility entry for standalone calls: builds a
    temporary GshIntegrator from the state configuration and
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
    return GshIntegrator(state)(state)
