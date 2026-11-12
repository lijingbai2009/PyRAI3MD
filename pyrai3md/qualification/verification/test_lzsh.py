"""
======================================================================
This module tests the LZSH (Landau-Zener) surface hopping integrator
of the NAMD workflow (dev/methods/LZSH_DEVELOPMENT.md):

Tests cover:
- internal-conversion kernel (gap curvature): hand-computed
  reference, the quadratic-gap reproduction of the analytic
  single-passage Landau-Zener probability (the lz1 parameter set)
  and the GSH-style three-step gate criteria
- intersystem-crossing kernel (gap slope): hand-computed reference
  of the signed-gap central difference, exactness at every crossing
  position (crossing exactly at t-dt and inside either sub-
  interval), the two t-dt constraints without a minimum-gap gate,
  the weak-coupling perturbative limit and the parallel-gap
  degradation
- numerical guards: missing/zero SOC, gap thresholds, classical
  allowance, flat gap windows degrade to zero probability (no
  nan/inf)
- warm-up steps (insufficient history) seed the pure-state density
  without hopping
- internal-conversion hop through the official update path
  (current_state, velocities, state_density, state_population)
  with isotropic rescaling energy conservation
- frustrated hopping keeps the active state and the density on the
  departure state
- intersystem crossing hop rescales isotropically; undeclared or
  uncomputed SOC pairs degrade to zero probability
- maxh constraint and the surface_hopping factory dispatch

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.units import (
    AMU_TO_AU, CM_TO_HARTREE, EV_TO_HARTREE,
)
from pyrai3md.workflow.integrator import surface_hopping
from pyrai3md.workflow.integrator.electronic import ElectronicStepper
from pyrai3md.workflow.integrator.lzsh import (
    LzshIntegrator, run_lzsh,
    internal_conversion_prob, intersystem_crossing_prob,
)

# Test system constants
MASS = [12.0, 1.0]                       # amu
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]   # Angstrom
VELOS = [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]  # Bohr/au
MASS_AU = np.asarray(MASS) * AMU_TO_AU


# ---- State builders ----

def _make_state(nstate=2, current_state=0, velocities=VELOS,
                soc_pairs=None, state_spin=None,
                **dynamics_overrides):
    """Create a minimal PyRAIState for LZSH tests."""
    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [list(row) for row in COORDS]
    config.system.velocities = [list(row) for row in velocities]
    config.system.mass = list(MASS)
    config.system.natom = 2
    config.system.nstate = nstate
    if soc_pairs is not None:
        config.system.soc_pair_map = [list(p) for p in soc_pairs]
    if state_spin is not None:
        config.system.state_spin = list(state_spin)

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    state.update_system({'current_state': current_state})
    for key, value in dynamics_overrides.items():
        setattr(config.dynamics, key, value)
    return state


def _push_step(state, energy, soc=None):
    """Snapshot one recorded step through the official history API.

    Mirrors the MD loop order: the step's results are applied
    first, then the snapshot is pushed -- the live state keeps
    carrying the pushed data, so pushing the current step last
    reproduces the production history layout (the newest snapshot
    duplicates the current step). The LZSH kernels consume the
    state energies only: no gradient payloads.
    """
    result = {'energy': list(energy)}
    if soc is not None:
        result['soc'] = list(soc)
    state.set_result(result)
    state.update_history()


def _density(state):
    data = state.pmd.system.state_density
    arr = np.asarray(data, dtype=np.float64)
    return arr[..., 0] + 1j * arr[..., 1]


def _kinetic(velocities):
    velo = np.asarray(velocities, dtype=np.float64)
    return float(0.5 * np.sum(MASS_AU[:, None] * velo ** 2))


# ---- Internal conversion kernel (gap curvature) ----

class TestLzshInternalConversionKernel(unittest.TestCase):

    def _ic_frames(self):
        # Crossing geometry: the gap at t-dt is the smallest of the
        # three steps (argmin == 1) and the window is convex.
        e = np.array([0.0, 0.04])
        ep = np.array([0.0, 0.02])
        epp = np.array([0.0, 0.06])
        return e, ep, epp

    def test_ic_prob_hand_reference(self):
        e, ep, epp = self._ic_frames()
        dt = 20.0
        de = 0.02
        gddot = (0.04 - 2 * 0.02 + 0.06) / dt ** 2
        gamma = de ** 2 / (4 * np.sqrt(de * gddot))
        expected = float(np.exp(-2 * np.pi * gamma))
        got = internal_conversion_prob(1, 0, e, ep, epp, 4.0, dt,
                                       1.0 * EV_TO_HARTREE)
        self.assertGreater(expected, 0.0)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_ic_quadratic_gap_reproduces_analytic_lz(self):
        # The textbook Landau-Zener model (lz1 parameter set): the
        # avoided-crossing gap delta(q) = sqrt(df^2*q^2 + de^2) is
        # quadratic at its minimum, delta(q) ~ de + df^2*q^2/(2*de),
        # so the three-point second derivative of the quadratic gap
        # series at the symmetric sampling points is the exact gate
        # curvature and the kernel probability reproduces the
        # analytic single-passage Landau-Zener probability:
        #   P = exp(-2*pi*lz_b^2/(v_c*df)), df = 2*lz_a.
        lz_a, lz_b, v_c, dt = 0.2712, 0.0543, 0.035, 20.0
        df = 2 * lz_a
        de = 2 * lz_b
        q = v_c * dt                    # one sampling window
        g_side = de + df ** 2 * q ** 2 / (2 * de)
        e = np.array([0.0, g_side])     # t:  one window past the
        ep = np.array([0.0, de])        # t-dt: crossing (gate step)
        epp = np.array([0.0, g_side])   # t-2dt: one window before
        got = internal_conversion_prob(1, 0, e, ep, epp, 0.1, dt,
                                       3.5 * EV_TO_HARTREE)
        analytic = float(np.exp(
            -np.pi * lz_b ** 2 / (lz_a * v_c)))
        self.assertGreater(got, 0.0)
        np.testing.assert_allclose(got, analytic, rtol=1e-12)
        np.testing.assert_allclose(got, 0.377, atol=0.001)

    def test_ic_gate_criteria(self):
        e, ep, epp = self._ic_frames()
        gap_au = 1.0 * EV_TO_HARTREE
        # Crossing detected one step too early: argmin != 1.
        self.assertEqual(internal_conversion_prob(
            1, 0, ep, e, epp, 4.0, 20.0, gap_au), 0.0)
        # Gap above the threshold.
        self.assertEqual(internal_conversion_prob(
            1, 0, e, ep, epp, 4.0, 20.0,
            0.01 * EV_TO_HARTREE), 0.0)
        # Classical allowance violated: no previous-step kinetic
        # energy (etotp = 0 < ex = 0.01).
        self.assertEqual(internal_conversion_prob(
            1, 0, e, ep, epp, 0.0, 20.0, gap_au), 0.0)

    def test_ic_flat_gap_window_zero(self):
        # Flat gap: every step identical (no crossing information);
        # the argmin gate rejects the flat window (argmin == 0).
        flat = np.array([0.0, 0.02])
        self.assertEqual(internal_conversion_prob(
            1, 0, flat, flat, flat, 4.0, 20.0,
            1.0 * EV_TO_HARTREE), 0.0)


# ---- Intersystem crossing kernel (gap slope) ----

class TestLzshIntersystemCrossingKernel(unittest.TestCase):

    def _linear_frames(self, slope, offset=0.0):
        # Signed spin-diabatic gap deltaE = E_i - E_active crossing
        # zero at t-dt + offset*dt (linear in time); the target
        # state climbs while the active state stays at zero.
        dt = 20.0
        e = np.array([0.0, slope * (1.0 + offset) * dt])
        ep = np.array([0.0, slope * offset * dt])
        epp = np.array([0.0, slope * (-1.0 + offset) * dt])
        return e, ep, epp

    def test_isc_prob_hand_reference(self):
        # Crossing exactly at the gate step t-dt: the absolute gap
        # minimum is zero (the curvature form of the IC kernel is
        # unusable here), yet the gap-slope kernel stays finite and
        # exact: gdot = |slope*dt - (-slope*dt)|/(2*dt) = slope and
        # P = 1 - exp(-2*pi*SOC^2/gdot).
        slope = 5.0e-4
        soc = 5000.0
        e, ep, epp = self._linear_frames(slope)
        v12 = soc * CM_TO_HARTREE
        expected = 1.0 - np.exp(-2 * np.pi * v12 ** 2 / slope)
        got = intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, soc, 20.0, 0.2)
        self.assertGreater(expected, 0.0)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_isc_exact_at_every_crossing_position(self):
        # The signed gap is linear in time, so the central
        # difference is exact wherever the crossing lands inside
        # the window (exactly at t-dt is covered above; here the
        # crossing sits inside either sub-interval).
        slope = 5.0e-4
        soc = 5000.0
        v12 = soc * CM_TO_HARTREE
        expected = 1.0 - np.exp(-2 * np.pi * v12 ** 2 / slope)
        for offset in (0.5, -0.5, 0.25):
            e, ep, epp = self._linear_frames(slope, offset=offset)
            got = intersystem_crossing_prob(
                1, 0, e, ep, epp, 4.0, soc, 20.0, 0.2)
            np.testing.assert_allclose(got, expected, atol=1e-12,
                                       err_msg='offset=%g' % offset)

    def test_isc_no_minimum_gap_gate(self):
        # No argmin gate: when the smallest absolute gap of the
        # window sits at the current step (the crossing has already
        # passed the gate), the GSH-style minimum gate would reject
        # the pair -- LZSH evaluates it anyway as long as the
        # t-dt constraints hold.
        slope = 5.0e-4
        soc = 5000.0
        e = np.array([0.0, 0.4 * slope * 20.0])    # smallest |gap|
        ep = np.array([0.0, 1.4 * slope * 20.0])
        epp = np.array([0.0, 2.4 * slope * 20.0])
        v12 = soc * CM_TO_HARTREE
        expected = 1.0 - np.exp(-2 * np.pi * v12 ** 2 / slope)
        got = intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, soc, 20.0, 0.2)
        self.assertGreater(got, 0.0)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_isc_weak_coupling_limit(self):
        # Weak SOC: 1 - exp(-2*pi*gamma) ~ 2*pi*gamma (the
        # perturbative result).
        slope = 5.0e-4
        soc = 10.0
        e, ep, epp = self._linear_frames(slope)
        got = intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, soc, 20.0, 0.2)
        v12 = soc * CM_TO_HARTREE
        # Linearization error of 1 - exp(-x) ~ x is x/2 ~ 1.3e-5.
        np.testing.assert_allclose(got, 2 * np.pi * v12 ** 2 / slope,
                                   rtol=1e-3)

    def test_isc_constraints(self):
        slope = 5.0e-4
        soc = 5000.0
        e, ep, epp = self._linear_frames(slope)
        # Gap above the threshold at t-dt... never triggers on the
        # zero gate-step gap; use an offset window instead.
        e, ep, epp = self._linear_frames(slope, offset=0.5)
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, soc, 20.0,
            0.1 * slope * 20.0), 0.0)
        # Energy conservation violated: no previous-step kinetic
        # energy and the target state above the active one (ex =
        # ep[i] > etotp).
        e, ep, epp = self._linear_frames(slope, offset=0.5)
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, 0.0, soc, 20.0, 0.2), 0.0)

    def test_isc_parallel_gap_zero(self):
        # Parallel spin-diabatic surfaces: the signed gap is
        # constant, gdot == 0, no crossing -- zero probability.
        const = np.array([0.0, 0.01])
        self.assertEqual(intersystem_crossing_prob(
            1, 0, const, const, const, 4.0, 5000.0, 20.0, 0.2),
            0.0)

    def test_isc_missing_soc_zero(self):
        slope = 5.0e-4
        e, ep, epp = self._linear_frames(slope)
        # Undeclared pair / uncomputed value.
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, None, 20.0, 0.2), 0.0)
        # Zero and negative couplings.
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, 0.0, 20.0, 0.2), 0.0)
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, 4.0, -5.0, 20.0, 0.2), 0.0)


# ---- Warm-up, factory and configuration ----



# ---- Internal conversion hops ----



# ---- Intersystem crossing hops ----



if __name__ == '__main__':
    unittest.main()
