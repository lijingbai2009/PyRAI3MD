"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests for the analytic models (Tully set and the Landau-Zener
lz1/lz2/lz3 crossing family): central-difference verification of the
analytic dH/dx, Hamiltonian symmetry, parameter override / registry
behavior, and the lz-family analytic physics (crossing velocity,
Landau-Zener probability, adiabatic gradients).

"""

import unittest

import numpy as np

from pyrai3md.skills.qdm_skill.service.models import (
    MODEL_REGISTRY, get_model)
from pyrai3md.skills.qdm_skill.service.models.tully3 import Tully3

# Points away from the kink at x = 0 (the Tully1 V11 derivative
# takes a one-sided limit there and cannot be centered).
_SMOOTH_POINTS = [-8.0, -3.0, -0.7, 0.4, 2.5, 9.0]


class TestModelHamiltonians(unittest.TestCase):

    def test_registry_contents(self):
        self.assertEqual(sorted(MODEL_REGISTRY),
                         ['lz1', 'lz2', 'lz3',
                          'tully1', 'tully2', 'tully3'])

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_model('tully9')

    def test_hamiltonian_symmetry(self):
        for name in MODEL_REGISTRY:
            model = get_model(name)
            for x in _SMOOTH_POINTS + [0.0]:
                ham = model.hamiltonian(x)
                self.assertEqual(ham.shape, (2, 2))
                np.testing.assert_allclose(ham, ham.T, atol=1e-14,
                                           err_msg='%s x=%s' % (name, x))

    def test_gradient_matches_central_difference(self):
        dx = 1e-5
        for name in MODEL_REGISTRY:
            model = get_model(name)
            for x in _SMOOTH_POINTS:
                analytic = model.hamiltonian_gradient(x)
                numeric = model.numerical_hamiltonian_gradient(x, dx=dx)
                np.testing.assert_allclose(
                    analytic, numeric, atol=1e-6,
                    err_msg='%s x=%s' % (name, x))

    def test_tully1_kink_right_limit(self):
        # dV11/dx at x = 0 takes the right limit A*B
        model = get_model('tully1')
        grad = model.hamiltonian_gradient(0.0)
        self.assertAlmostEqual(grad[0, 0],
                               model.tully_a * model.tully_b)

    def test_tully2_form(self):
        # Tully 1990 example II: flat V11, Gaussian barrier on V22
        # shifted by E, Gaussian coupling.
        model = get_model('tully2')
        self.assertEqual(model.tully_e, 0.05)
        for x in [-2.0, 0.0, 1.5, 4.0]:
            ham = model.hamiltonian(x)
            v22 = (-model.tully_a * np.exp(-model.tully_b * x * x)
                   + model.tully_e)
            v12 = model.tully_c * np.exp(-model.tully_d * x * x)
            np.testing.assert_allclose(
                ham, [[0.0, v12], [v12, v22]], atol=1e-14)

    def test_tully3_kink_branch(self):
        # The two coupling branches meet at x = 0 with the same
        # derivative C*B, so the gradient is continuous there.
        model = get_model('tully3')
        expected = model.tully_c * model.tully_b
        grad0 = model.hamiltonian_gradient(0.0)
        self.assertAlmostEqual(grad0[0, 1], expected)
        # Left/right limits of dV12/dx agree at the junction.
        left = model.hamiltonian_gradient(-1e-8)[0, 1]
        right = model.hamiltonian_gradient(1e-8)[0, 1]
        self.assertAlmostEqual(left, right, places=8)
        # x >= 0 branch: V12 = B(2 - exp(-Cx)) approaches 2B.
        ham_far = model.hamiltonian(50.0)
        self.assertAlmostEqual(ham_far[0, 1],
                               2.0 * model.tully_b, places=6)

    def test_tully3_form_branches(self):
        model = get_model('tully3')
        for x in [-3.0, -0.4]:
            ham = model.hamiltonian(x)
            v12 = model.tully_b * np.exp(model.tully_c * x)
            np.testing.assert_allclose(
                ham, [[model.tully_a, v12],
                      [v12, -model.tully_a]], atol=1e-14)
        for x in [0.0, 0.6, 3.0]:
            ham = model.hamiltonian(x)
            v12 = model.tully_b * (2.0 - np.exp(-model.tully_c * x))
            np.testing.assert_allclose(
                ham, [[model.tully_a, v12],
                      [v12, -model.tully_a]], atol=1e-14)

    def test_parameter_override(self):
        model = get_model('tully1', params={'tully_a': 0.02})
        self.assertEqual(model.tully_a, 0.02)
        # untouched parameters keep the standard values
        self.assertEqual(model.tully_b, 1.6)
        # tully_e override applies on tully2
        model = get_model('tully2', params={'tully_e': 0.08})
        self.assertEqual(model.tully_e, 0.08)

    def test_none_override_keeps_standard(self):
        model = get_model('tully2', params={'tully_a': None})
        self.assertEqual(model.tully_a, 0.1)

    def test_undeclared_parameter_ignored(self):
        # tully3 declares no tully_e; the config schema forwards
        # all five keys uniformly, so get_model must silently drop
        # entries the model does not declare.
        model = get_model('tully3', params={'tully_e': 1.0})
        self.assertEqual(model.name, 'tully3')
        self.assertFalse(hasattr(model, 'tully_e'))
        # Declared overrides still apply next to ignored entries.
        model = get_model('tully3', params={'tully_a': 0.001,
                                            'tully_e': 1.0})
        self.assertEqual(model.tully_a, 0.001)

    def test_direct_construction_still_validates(self):
        # The constructor keeps the strict check as a defense for
        # direct programmatic misuse (get_model filters first).
        with self.assertRaises(ValueError):
            Tully3(params={'tully_e': 1.0})

    def test_tully3_ignores_tully_e_none(self):
        # None overrides are skipped before the name check
        model = get_model('tully3', params={'tully_e': None})
        self.assertEqual(model.name, 'tully3')

    def test_lz1_defaults_and_form(self):
        model = get_model('lz1')
        self.assertEqual(model.lz_a, 0.04)
        self.assertEqual(model.lz_b, 0.02)
        # V11 = lz_a*x, V22 = -lz_a*x, V12 = lz_b
        ham = model.hamiltonian(2.0)
        np.testing.assert_allclose(ham, [[0.08, 0.02], [0.02, -0.08]],
                                   atol=1e-14)
        # dH/dx is constant and traceless-diagonal
        grad = model.hamiltonian_gradient(2.0)
        np.testing.assert_allclose(grad, [[0.04, 0.0], [0.0, -0.04]],
                                   atol=1e-14)

    def test_lz1_parameter_override(self):
        model = get_model('lz1', params={'lz_a': 0.08, 'lz_b': None})
        self.assertEqual(model.lz_a, 0.08)
        # None keeps the class default
        self.assertEqual(model.lz_b, 0.02)
        # lz_a/lz_b are ignored by the Tully models
        model = get_model('tully1', params={'lz_a': 0.08,
                                            'lz_b': 0.01})
        self.assertEqual(model.name, 'tully1')
        self.assertFalse(hasattr(model, 'lz_a'))

    def test_lz1_analytic_landau_zener(self):
        # P = exp(-pi*b^2/(a*v)); at x = 0 the launch velocity is
        # conserved (E_lower(0) = -b cancels the +b term).
        model = get_model('lz1')
        v_c = model.crossing_velocity(2000.0, 0.0, 0.03)
        self.assertAlmostEqual(v_c, 0.03)
        p_lz = model.landau_zener_probability(v_c)
        self.assertAlmostEqual(
            p_lz, np.exp(-np.pi * 0.02 ** 2 / (0.04 * 0.03)))
        # A launch away from the crossing converts lower-surface
        # potential energy into kinetic energy at x = 0.
        v_c = model.crossing_velocity(2000.0, -5.0, 0.0498)
        ekin = 0.5 * 2000.0 * 0.0498 ** 2
        ekin += -np.sqrt((0.04 * -5.0) ** 2 + 0.02 ** 2) + 0.02
        self.assertAlmostEqual(v_c, np.sqrt(2.0 * ekin / 2000.0))
        # Zero velocity is guarded (no crossing probability).
        self.assertEqual(model.landau_zener_probability(0.0), 0.0)

    def test_lz2_defaults_and_form(self):
        model = get_model('lz2')
        self.assertEqual(model.lz_a, 0.04)
        self.assertEqual(model.lz_b, 0.02)
        # Flat upper surface: V11 = 0, V22 = -lz_a*x, V12 = lz_b
        ham = model.hamiltonian(2.0)
        np.testing.assert_allclose(ham, [[0.0, 0.02], [0.02, -0.08]],
                                   atol=1e-14)
        # dH/dx is constant with the slope only on the lower branch
        grad = model.hamiltonian_gradient(2.0)
        np.testing.assert_allclose(grad, [[0.0, 0.0], [0.0, -0.04]],
                                   atol=1e-14)

    def test_lz3_defaults_and_form(self):
        model = get_model('lz3')
        self.assertEqual(model.lz_a, 0.04)
        self.assertEqual(model.lz_b, 0.02)
        # Flat lower surface: V11 = lz_a*x, V22 = 0, V12 = lz_b
        ham = model.hamiltonian(2.0)
        np.testing.assert_allclose(ham, [[0.08, 0.02], [0.02, 0.0]],
                                   atol=1e-14)
        # dH/dx is constant with the slope only on the upper branch
        grad = model.hamiltonian_gradient(2.0)
        np.testing.assert_allclose(grad, [[0.04, 0.0], [0.0, 0.0]],
                                   atol=1e-14)

    def test_lz_half_linear_parameter_override(self):
        model = get_model('lz2', params={'lz_a': 0.08, 'lz_b': None})
        self.assertEqual(model.lz_a, 0.08)
        # None keeps the class default
        self.assertEqual(model.lz_b, 0.02)
        model = get_model('lz3', params={'lz_b': 0.01})
        self.assertEqual(model.lz_b, 0.01)
        self.assertEqual(model.lz_a, 0.04)

    def test_lz2_analytic_landau_zener(self):
        # Launching at the crossing conserves the velocity
        # (E_lower(0) = -lz_b cancels the +lz_b term), and the
        # exponent doubles against lz1 at equal velocity.
        model = get_model('lz2')
        v_c = model.crossing_velocity(2000.0, 0.0, 0.03)
        self.assertAlmostEqual(v_c, 0.03)
        p_lz = model.landau_zener_probability(v_c)
        self.assertAlmostEqual(
            p_lz, np.exp(-2.0 * np.pi * 0.02 ** 2 / (0.04 * 0.03)))
        # A launch away from the crossing converts lower-surface
        # potential energy into kinetic energy at x = 0.
        v_c = model.crossing_velocity(2000.0, -5.0, 0.06833)
        ekin = 0.5 * 2000.0 * 0.06833 ** 2
        e_lower_x0 = -0.04 * -5.0 / 2.0
        e_lower_x0 -= np.sqrt((0.04 * -5.0 / 2.0) ** 2 + 0.02 ** 2)
        ekin += e_lower_x0 + 0.02
        self.assertAlmostEqual(v_c, np.sqrt(2.0 * ekin / 2000.0))
        # Zero velocity is guarded (no crossing probability).
        self.assertEqual(model.landau_zener_probability(0.0), 0.0)

    def test_lz3_analytic_landau_zener(self):
        model = get_model('lz3')
        v_c = model.crossing_velocity(2000.0, 0.0, 0.03)
        self.assertAlmostEqual(v_c, 0.03)
        p_lz = model.landau_zener_probability(v_c)
        self.assertAlmostEqual(
            p_lz, np.exp(-2.0 * np.pi * 0.02 ** 2 / (0.04 * 0.03)))
        # The approach side climbs the +lz_a branch exactly like lz1
        v_c = model.crossing_velocity(2000.0, -5.0, 0.07762)
        ekin = 0.5 * 2000.0 * 0.07762 ** 2
        e_lower_x0 = 0.04 * -5.0 / 2.0
        e_lower_x0 -= np.sqrt((0.04 * -5.0 / 2.0) ** 2 + 0.02 ** 2)
        ekin += e_lower_x0 + 0.02
        self.assertAlmostEqual(v_c, np.sqrt(2.0 * ekin / 2000.0))
        self.assertEqual(model.landau_zener_probability(0.0), 0.0)

    def test_lz_adiabatic_gradients_match_eigenvalues(self):
        # Closed-form adiabatic gradients vs centered difference of
        # the sorted eigenvalues (the lz family is smooth everywhere
        # because the gap never closes: lz_b > 0).
        dx = 1e-5
        for name in ('lz1', 'lz2', 'lz3'):
            model = get_model(name)
            for x in [-3.0, -0.5, 0.0, 1.2, 4.0]:
                lower, upper = model.adiabatic_gradients(x)
                ev_plus = np.sort(
                    np.linalg.eigvalsh(model.hamiltonian(x + dx)))
                ev_minus = np.sort(
                    np.linalg.eigvalsh(model.hamiltonian(x - dx)))
                numeric = (ev_plus - ev_minus) / (2.0 * dx)
                self.assertAlmostEqual(
                    lower, numeric[0], places=6,
                    msg='%s lower x=%s' % (name, x))
                self.assertAlmostEqual(
                    upper, numeric[1], places=6,
                    msg='%s upper x=%s' % (name, x))


if __name__ == '__main__':
    unittest.main()
