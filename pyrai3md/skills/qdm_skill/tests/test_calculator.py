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

Unit tests for the single-point calculator: the analytic NAC
benchmark at the Tully1 crossing point, gradient consistency with
finite differences, energy ordering, NAC pair filtering, and the
near-zero-gap guard driven by the skill-local NAC_ENERGY_GAP_EPS.

"""

import unittest
import warnings

import numpy as np

from pyrai3md.skills.qdm_skill.service.models import get_model
from pyrai3md.skills.qdm_skill.service.calculator import (
    calculate_properties, diagonalize_hamiltonian, model_coordinate)
from pyrai3md.skills.qdm_skill.service.utils import (
    ANGSTROM_TO_BOHR, NAC_ENERGY_GAP_EPS)


class TestModelCoordinate(unittest.TestCase):

    def test_angstrom_to_bohr(self):
        coords = [[-7.9376581, 0.0, 0.0]]
        x = model_coordinate(coords)
        self.assertAlmostEqual(x, -15.0, places=5)

    def test_only_first_atom_x_counts(self):
        coords = [[1.0 / ANGSTROM_TO_BOHR, 5.0, 9.0],
                  [99.0, 99.0, 99.0]]
        self.assertAlmostEqual(model_coordinate(coords), 1.0)


class TestEnergies(unittest.TestCase):

    def test_ascending_order(self):
        model = get_model('tully1')
        for x in [-10.0, -1.0, 0.0, 3.0, 10.0]:
            energies, _ = diagonalize_hamiltonian(model, x)
            self.assertTrue(np.all(np.diff(energies) >= 0))

    def test_tully1_far_field(self):
        # Far left V11 -> -A, V12 -> 0, so E -> (-A, +A)
        model = get_model('tully1')
        properties = calculate_properties(
            model, [[-7.9376581, 0.0, 0.0]],
            with_gradient=False, with_nac=False)
        np.testing.assert_allclose(
            properties['energy'], [-0.01, 0.01], atol=1e-8)

    def test_target_states_filter(self):
        model = get_model('tully1')
        full = calculate_properties(model, [[-2.0, 0.0, 0.0]],
                                    with_gradient=False, with_nac=False)
        single = calculate_properties(model, [[-2.0, 0.0, 0.0]],
                                      target_states=[1],
                                      with_gradient=False, with_nac=False)
        self.assertEqual(single['energy'].shape, (1,))
        self.assertAlmostEqual(single['energy'][0],
                               full['energy'][1])

    def test_out_of_range_target_raises(self):
        model = get_model('tully1')
        with self.assertRaises(ValueError):
            calculate_properties(model, [[0.0, 0.0, 0.0]],
                                 target_states=[5],
                                 with_gradient=False, with_nac=False)


class TestGradient(unittest.TestCase):

    def test_matches_finite_difference(self):
        # Adiabatic gradient vs numerical diagonalization sweep
        h = 1e-4
        for name in ['tully1', 'tully2', 'tully3']:
            model = get_model(name)
            for x in [-4.0, 0.7, 5.0]:
                coords = [[x / ANGSTROM_TO_BOHR, 0.0, 0.0]]
                properties = calculate_properties(
                    model, coords, with_nac=False)
                e_minus = diagonalize_hamiltonian(model, x - h)[0]
                e_plus = diagonalize_hamiltonian(model, x + h)[0]
                numeric = (e_plus - e_minus) / (2.0 * h)
                np.testing.assert_allclose(
                    properties['gradient'][:, 0, 0], numeric,
                    atol=1e-5, err_msg='%s x=%s' % (name, x))

    def test_gradient_shape_only_x(self):
        model = get_model('tully1')
        properties = calculate_properties(
            model, [[-2.0, 0.0, 0.0]], with_nac=False)
        self.assertEqual(properties['gradient'].shape, (2, 1, 3))
        self.assertEqual(properties['gradient'][0, 0, 1], 0.0)
        self.assertEqual(properties['gradient'][0, 0, 2], 0.0)


class TestNac(unittest.TestCase):

    def test_tully1_crossing_benchmark(self):
        # At x = 0 the CI-derivative convention stores
        # <0| dH/dx |1> = d_01 * (E_1 - E_0)
        # = (-AB/(2C)) * (2C) = -A * B
        model = get_model('tully1')
        properties = calculate_properties(
            model, [[0.0, 0.0, 0.0]], with_gradient=False)
        self.assertEqual(properties['nac'].shape, (1, 1, 3))
        self.assertAlmostEqual(properties['nac'][0, 0, 0], -0.016,
                               places=8)

    def test_pair_order_respected(self):
        # The stored <i| dH/dx |j> is symmetric under swapping the
        # pair order, because dH/dx is a real symmetric matrix.
        model = get_model('tully1')
        forward = calculate_properties(
            model, [[0.5, 0.0, 0.0]], target_nac_pair=[[0, 1]],
            with_gradient=False)
        backward = calculate_properties(
            model, [[0.5, 0.0, 0.0]], target_nac_pair=[[1, 0]],
            with_gradient=False)
        np.testing.assert_allclose(
            forward['nac'][0, 0, 0], backward['nac'][0, 0, 0])

    def test_far_field_nac_vanishes(self):
        # Far from the crossing V12 -> 0, adiabatic ~ diabatic
        model = get_model('tully1')
        properties = calculate_properties(
            model, [[-15.0 / ANGSTROM_TO_BOHR, 0.0, 0.0]],
            with_gradient=False)
        self.assertAlmostEqual(properties['nac'][0, 0, 0], 0.0,
                               places=8)

    def test_near_zero_gap_sets_nac_to_zero(self):
        # tully_a = 0 + large x gives H ~ 0, so the gap falls
        # below NAC_ENERGY_GAP_EPS: NAC is zeroed with a warning
        # instead of raising a division-by-zero error.
        model = get_model('tully1', params={'tully_a': 0.0})
        coords = [[100.0, 0.0, 0.0]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            properties = calculate_properties(
                model, coords, with_gradient=False)
        self.assertEqual(properties['nac'][0, 0, 0], 0.0)
        self.assertTrue(any('NAC_ENERGY_GAP_EPS' in str(w.message)
                            for w in caught))

    def test_gap_guard_matches_skill_constant(self):
        # Guard must fire exactly at the skill-local constant
        energies, _ = diagonalize_hamiltonian(
            get_model('tully1', params={'tully_a': 0.0}), 100.0)
        self.assertLess(abs(energies[1] - energies[0]),
                        NAC_ENERGY_GAP_EPS)


if __name__ == '__main__':
    unittest.main()
