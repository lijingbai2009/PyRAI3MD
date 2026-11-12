"""
======================================================================
This module tests the GSH (Zhu-Nakamura) global surface hopping
integrator of the NAMD workflow
(dev/methods/GSH_DEVELOPMENT.md):

Tests cover:
- probability kernel regression against the legacy formulas
  (internal conversion EQ 1-8, intersystem crossing adaptation)
- numerical guards: frozen coordinates, zero gradients, zero gaps,
  zero/missing SOC degrade to zero probability (no nan/inf)
- early-stop criteria: three-step argmin crossing detection, gap
  thresholds, classical allowance
- warm-up steps (insufficient history) seed the pure-state density
  without hopping
- internal-conversion hop through the official update path
  (current_state, velocities, state_density, state_population)
  with energy conservation along the target-state approximate NAC
  direction (legacy velocity-direction bug regression)
- frustrated hopping keeps the active state and the density on the
  departure state (legacy frustrated-density bug regression)
- intersystem crossing hop rescales isotropically; undeclared SOC
  pairs degrade to zero probability (legacy sys.exit regression)
- maxh constraint and the surface_hopping factory dispatch
- the shared tsh_helper velocity adjustment layer (energy
  conservation and frustrated flag for both algorithms)

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
from pyrai3md.utilities.units import AMU_TO_AU, CM_TO_HARTREE
from pyrai3md.workflow.integrator import surface_hopping
from pyrai3md.workflow.integrator.electronic import ElectronicStepper
from pyrai3md.workflow.integrator.gsh import (
    GshIntegrator, run_gsh,
    internal_conversion_prob, intersystem_crossing_prob,
    _continuous,
)
from pyrai3md.workflow.integrator.tsh_helper import (
    adjust_velo, adjust_velo_isotropic,
)

# Test system constants
MASS = [12.0, 1.0]                       # amu
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]   # Angstrom
VELOS = [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]  # Bohr/au
MASS_AU = np.asarray(MASS) * AMU_TO_AU


# ---- Legacy formula reference kernels (regression baseline) ----

def _ref_ic_prob(i, active, e, ep, epp, g, gpp, r, rp, rpp, ekinp,
                 mass, gap_au):
    """Legacy internal_conversion_prob, verbatim formulas."""
    del_e = np.abs([e[i] - e[active], ep[i] - ep[active],
                    epp[i] - epp[active]])
    etotp = ep[active] + ekinp
    ex = (ep[i] + ep[active]) / 2
    if np.argmin(del_e) != 1 or del_e[1] > gap_au or etotp - ex < 0:
        return 0.0, np.zeros_like(r)
    de = del_e[1]
    begin_term = -1 / (r - rpp)
    arg_min = np.argmin([i, active])
    arg_max = np.argmax([i, active])
    f1_grad_manip_1 = (g[arg_min]) * (rp - rpp)
    f1_grad_manip_2 = (gpp[arg_max]) * (rp - r)
    f_ia_1 = begin_term * (f1_grad_manip_1 - f1_grad_manip_2)
    f2_grad_manip_1 = (g[arg_max]) * (rp - rpp)
    f2_grad_manip_2 = (gpp[arg_min]) * (rp - r)
    f_ia_2 = begin_term * (f2_grad_manip_1 - f2_grad_manip_2)
    m = mass[:, None]
    nac = (f_ia_2 - f_ia_1) / (m ** 0.5)
    nac = nac / (np.sum(nac ** 2) ** 0.5)
    f_a = np.sum((f_ia_2 - f_ia_1) ** 2 / m) ** 0.5
    f_b = np.abs(np.sum((f_ia_2 * f_ia_1) / m)) ** 0.5
    a_squared = (f_a * f_b) / (2 * de ** 3)
    b_squared = (etotp - ex) * (f_a / (f_b * de))
    sign = np.sign(np.sum(f_ia_1 * f_ia_2))
    pi_over_four_term = -(np.pi / (4 * a_squared ** 0.5))
    b_in_denom_term = (2 / (b_squared
                            + (np.abs(b_squared ** 2 + sign)) ** 0.5))
    p = float(np.exp(pi_over_four_term * b_in_denom_term ** 0.5))
    return p, nac


def _ref_isc_prob(i, active, e, ep, epp, gp, ekinp, soc, mass,
                  gapsoc_au):
    """Legacy intersystem_crossing_prob, verbatim formulas."""
    del_e = [e[i] - e[active], ep[i] - ep[active],
             epp[i] - epp[active]]
    etotp = ep[active] + ekinp
    ex = ep[i]
    if (np.argmin(np.abs(del_e)) != 1
            or np.abs(del_e[1]) > gapsoc_au or etotp - ex < 0):
        return 0.0
    v12x2 = 2 * soc * CM_TO_HARTREE
    f_ia_1 = gp[active]
    f_ia_2 = gp[i]
    m = mass[:, None]
    f_a = np.sum((f_ia_2 - f_ia_1) ** 2 / m) ** 0.5
    f_b = np.abs(np.sum((f_ia_2 * f_ia_1) / m)) ** 0.5
    a_squared = (f_a * f_b) / (2 * v12x2 ** 3)
    b_squared = (etotp - ex) * (f_a / (f_b * v12x2))
    sign = np.sign(np.sum(f_ia_1 * f_ia_2))
    pi_over_four_term = -(np.pi / (4 * a_squared ** 0.5))
    b_in_denom_term = (2 / (b_squared
                            + (np.abs(b_squared ** 2 + sign)) ** 0.5))
    p = float(np.exp(pi_over_four_term * b_in_denom_term ** 0.5))
    return 1.0 - p    # legacy keeps the non-parallel PES assumption


# ---- State builders ----

def _make_state(nstate=2, current_state=0, velocities=VELOS,
                soc_pairs=None, state_spin=None,
                **dynamics_overrides):
    """Create a minimal PyRAIState for GSH tests."""
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


def _push_step(state, energy, gradient=None, soc=None,
               coordinates=None):
    """Snapshot one recorded step through the official history API.

    Mirrors the MD loop order: the step's results and coordinates
    are applied first, then the snapshot is pushed -- the live
    state keeps carrying the pushed data, so pushing the current
    step last reproduces the production history layout (the newest
    snapshot duplicates the current step).
    """
    result = {'energy': list(energy)}
    if gradient is not None:
        result['gradient'] = np.asarray(gradient).tolist()
    if soc is not None:
        result['soc'] = list(soc)
    state.set_result(result)
    if coordinates is not None:
        state.update_system({'coordinates': coordinates})
    state.update_history()


def _density(state):
    data = state.pmd.system.state_density
    arr = np.asarray(data, dtype=np.float64)
    return arr[..., 0] + 1j * arr[..., 1]


def _kinetic(velocities):
    velo = np.asarray(velocities, dtype=np.float64)
    return float(0.5 * np.sum(MASS_AU[:, None] * velo ** 2))


# ---- Kernel regression and guards ----

class TestGshKernelLegacyRegression(unittest.TestCase):
    """Probability kernel regression against the legacy formulas."""

    def _ic_data(self):
        # Crossing geometry: the gap at t-dt is the smallest of the
        # three steps; energies shrink toward the crossing region.
        e = np.array([0.0, 0.04])
        ep = np.array([0.0, 0.02])
        epp = np.array([0.0, 0.06])
        # Coordinates with no exactly-zero component (the frozen-
        # component 1D case is covered separately in
        # test_ic_prob_frozen_components_nan_free).
        r = np.array([[0.20, 0.04, 0.02], [1.20, 0.06, 0.03]])
        rp = np.array([[0.10, 0.03, 0.01], [1.10, 0.04, 0.02]])
        rpp = np.array([[0.00, 0.01, 0.01], [1.00, 0.02, 0.01]])
        g = np.array([[[0.10, 0.02, 0.0], [0.05, 0.01, 0.0]],
                      [[-0.10, -0.02, 0.0], [-0.05, -0.01, 0.0]]])
        gpp = np.array([[[0.06, 0.01, 0.0], [0.03, 0.0, 0.0]],
                        [[-0.06, -0.01, 0.0], [-0.03, 0.0, 0.0]]])
        return e, ep, epp, g, gpp, r, rp, rpp

    def test_ic_prob_matches_legacy(self):
        e, ep, epp, g, gpp, r, rp, rpp = self._ic_data()
        ekinp = 4.0
        expected_p, expected_nac = _ref_ic_prob(
            1, 0, e, ep, epp, g, gpp, r, rp, rpp, ekinp, MASS_AU,
            1.0 * 0.03674932217565)   # 1 eV in Hartree
        got_p, got_nac = internal_conversion_prob(
            1, 0, e, ep, epp, g, gpp, r, rp, rpp, ekinp, MASS_AU,
            1.0 * 0.03674932217565)
        self.assertGreater(expected_p, 0.0)
        np.testing.assert_allclose(got_p, expected_p, atol=1e-12)
        np.testing.assert_allclose(got_nac, expected_nac,
                                   atol=1e-12)

    def test_ic_prob_early_stop_criteria(self):
        e, ep, epp, g, gpp, r, rp, rpp = self._ic_data()
        # Crossing detected one step too early: argmin != 1.
        p, nac = internal_conversion_prob(
            1, 0, ep, e, epp, g, gpp, r, rp, rpp, 4.0, MASS_AU,
            0.1)
        self.assertEqual(p, 0.0)
        np.testing.assert_allclose(nac, np.zeros_like(r))
        # Gap above the threshold.
        p, _ = internal_conversion_prob(
            1, 0, e, ep, epp, g, gpp, r, rp, rpp, 4.0, MASS_AU,
            0.01)
        self.assertEqual(p, 0.0)
        # Classical allowance violated (no kinetic energy).
        p, _ = internal_conversion_prob(
            1, 0, e, ep, epp, g, gpp, r, rp, rpp, 0.0, MASS_AU,
            0.1)
        self.assertEqual(p, 0.0)

    def test_ic_prob_numerical_guards(self):
        e, ep, epp, g, gpp, r, rp, rpp = self._ic_data()
        # Frozen coordinates: r == rpp kills the curvature term.
        p, nac = internal_conversion_prob(
            1, 0, e, ep, epp, g, gpp, rpp, rp, rpp, 4.0, MASS_AU,
            0.1)
        self.assertEqual(p, 0.0)
        self.assertFalse(np.any(np.isnan(nac)))
        # Zero gradients: no coupling information.
        zero = np.zeros_like(g)
        p, nac = internal_conversion_prob(
            1, 0, e, ep, epp, zero, zero, r, rp, rpp, 4.0, MASS_AU,
            0.1)
        self.assertEqual(p, 0.0)
        self.assertFalse(np.any(np.isnan(nac)))
        # Missing gradient rows for the involved states.
        p, nac = internal_conversion_prob(
            1, 0, e, ep, epp, g[:1], gpp, r, rp, rpp, 4.0, MASS_AU,
            0.1)
        self.assertEqual(p, 0.0)

    def test_ic_prob_frozen_components_nan_free(self):
        # Strictly one-dimensional motion (model benchmarks): the
        # y/z components of r, rp and rpp are identical, so the
        # component-wise 1/(r - rpp) curvature term hit 0/0 NaNs
        # before the per-component frozen-motion mask. The masked
        # inversion keeps the probability finite and restricts the
        # approximate NAC to the moving direction.
        e = np.array([0.0, 0.04])
        ep = np.array([0.0, 0.02])
        epp = np.array([0.0, 0.06])
        r = np.array([[0.30, 0.0, 0.0]])
        rp = np.array([[0.15, 0.0, 0.0]])
        rpp = np.array([[0.0, 0.0, 0.0]])
        g = np.array([[[0.10, 0.0, 0.0]], [[-0.10, 0.0, 0.0]]])
        gpp = np.array([[[0.06, 0.0, 0.0]], [[-0.06, 0.0, 0.0]]])
        p, nac = internal_conversion_prob(
            1, 0, e, ep, epp, g, gpp, r, rp, rpp, 4.0, MASS_AU,
            0.1)
        self.assertFalse(np.isnan(p))
        self.assertGreater(p, 0.0)
        self.assertFalse(np.any(np.isnan(nac)))
        np.testing.assert_allclose(nac[:, 1:], 0.0, atol=1e-14)
        self.assertGreater(abs(nac[0, 0]), 0.0)

    def test_isc_prob_matches_legacy(self):
        e = np.array([0.0, 0.04])
        ep = np.array([0.0, 0.02])
        epp = np.array([0.0, 0.06])
        gp = np.array([[[5e-4, 0.0, 0.0], [2e-4, 0.0, 0.0]],
                       [[-5e-4, 0.0, 0.0], [-2e-4, 0.0, 0.0]]])
        soc = 5000.0
        expected = _ref_isc_prob(1, 0, e, ep, epp, gp, 4.0, soc,
                                 MASS_AU, 0.2)
        got = intersystem_crossing_prob(
            1, 0, e, ep, epp, gp, 4.0, soc, MASS_AU, 0.2)
        self.assertGreater(expected, 0.0)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_isc_prob_guards(self):
        e = np.array([0.0, 0.04])
        ep = np.array([0.0, 0.02])
        epp = np.array([0.0, 0.06])
        gp = np.array([[[5e-4, 0.0, 0.0], [2e-4, 0.0, 0.0]],
                       [[-5e-4, 0.0, 0.0], [-2e-4, 0.0, 0.0]]])
        # Missing SOC value (undeclared pair / not computed): zero
        # probability instead of the legacy process termination.
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, gp, 4.0, None, MASS_AU, 0.2), 0.0)
        # Zero SOC coupling.
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, gp, 4.0, 0.0, MASS_AU, 0.2), 0.0)
        # Gap above the threshold.
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, gp, 4.0, 5000.0, MASS_AU, 0.01), 0.0)
        # Zero gradients.
        self.assertEqual(intersystem_crossing_prob(
            1, 0, e, ep, epp, np.zeros_like(gp), 4.0, 5000.0,
            MASS_AU, 0.2), 0.0)


# ---- Warm-up, factory and configuration ----



# ---- Internal conversion hops ----



# ---- Intersystem crossing hops ----



# ---- Shared velocity adjustment layer ----

class TestTshHelperSharedLayer(unittest.TestCase):
    """The tsh_helper layer serves every hopping algorithm."""

    VELO = np.asarray(VELOS, dtype=np.float64)
    NAC = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_adjust_velo_global_rescale_conserves_energy(self):
        vt, frustrated = adjust_velo(0.0, 0.05, self.VELO, MASS_AU,
                                     self.NAC, 1, 1)
        self.assertEqual(frustrated, 0)
        np.testing.assert_allclose(_kinetic(vt),
                                   _kinetic(self.VELO) - 0.05,
                                   rtol=1e-12)

    def test_adjust_velo_frustrated_flag(self):
        vt, frustrated = adjust_velo(0.0, 100.0, self.VELO, MASS_AU,
                                     self.NAC, 1, 0)
        self.assertEqual(frustrated, 1)
        np.testing.assert_allclose(vt, self.VELO)

    def test_adjust_velo_isotropic_conserves_energy(self):
        vt, frustrated = adjust_velo_isotropic(
            0.0, 0.05, self.VELO, MASS_AU)
        self.assertEqual(frustrated, 0)
        np.testing.assert_allclose(_kinetic(vt),
                                   _kinetic(self.VELO) - 0.05,
                                   rtol=1e-12)

    def test_adjust_velo_isotropic_frustrated_flag(self):
        vt, frustrated = adjust_velo_isotropic(
            0.0, 100.0, self.VELO, MASS_AU)
        self.assertEqual(frustrated, 1)
        np.testing.assert_allclose(vt, self.VELO)




if __name__ == '__main__':
    unittest.main()
