"""
======================================================================
Tests for the result shapes the controller has to absorb.

The framework hands a finalizer state-resolved results: energy as a scalar or
one value per electronic state, gradient as the active `(natom, 3)` array or
the state-resolved `(ns, natom, 3)` one. Getting this wrong is not a crash at
integration time, it is a silently wrong force, so the shapes are pinned here
against a synthetic base rather than discovered while wiring.

Absence is carried by the key, not by a row: data/state.py returns None when no
gradient was committed. A bias gradient with nothing electronic behind it is
therefore not published as `gradient_total`, only as `gradient_correction`.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data import constraint as records
from pyrai3md import schema
from pyrai3md.workflow.constraints import controller as ctl

ATOMS = ['O', 'H', 'H', 'C']
MASSES = np.array([16.0, 1.0, 1.0, 12.0])
GEOMETRY = np.array([[0.00, 0.00, 0.00],
                     [1.90, 0.30, 0.10],
                     [0.40, 2.00, -0.20],
                     [2.10, 1.80, 1.30]])
CONTEXT = records.EvaluationContext(purpose='single', step=0,
                                    evaluation_id='shape')


def _controller(fix=False):
    mapping = {'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
               'biases': [{'cv': 'r', 'target': 1.0,
                          'force_constant': 0.02, 'region': 'all'}]}
    if fix:
        mapping['fixes'] = [{'id': 'f', 'atoms': [1], 'direction': 'xyz'}]
    plan = schema.build_plan(mapping, atoms=ATOMS, natom=4, masses=MASSES)
    controller = ctl.SystemConstraintController(plan)
    return controller, controller.bind_reference(GEOMETRY, masses=MASSES)


def _potential(controller, state):
    return controller.evaluate_potentials(GEOMETRY, state, CONTEXT,
                                          masses=MASSES)


class TestScalarBase(unittest.TestCase):
    """C>1 active combination, and C=1 with a single state."""

    def setUp(self):
        self.controller, self.state = _controller()
        self.potential = _potential(self.controller, self.state)

    def test_scalar_energy_and_active_gradient_combine(self):
        base = {'energy': -1.5, 'gradient': np.zeros((4, 3))}
        totals = self.controller.combine(base, self.potential, None, CONTEXT)
        self.assertAlmostEqual(float(totals.energy_total),
                               -1.5 + self.potential.energy_correction,
                               places=14)
        np.testing.assert_allclose(totals.gradient_total,
                                   self.potential.gradient_correction,
                                   atol=0.0)
        self.assertEqual(totals.provenance['nstates'], 1)

    def test_a_3d_gradient_is_refused_rather_than_guessed(self):
        base = {'energy': -1.5, 'gradient': np.zeros((4, 3, 2))}
        with self.assertRaises(ctl.ControllerError):
            self.controller.combine(base, self.potential, None, CONTEXT)


class TestStateResolvedBase(unittest.TestCase):
    """C=1 with several states: the same added potential, every state."""

    def setUp(self):
        self.controller, self.state = _controller()
        self.potential = _potential(self.controller, self.state)
        self.base_gradient = np.stack([np.full((4, 3), 0.1 * k)
                                       for k in range(3)])
        self.base = {'energy': [-1.0, -0.9, -0.8],
                     'gradient': self.base_gradient}

    def test_energy_shifts_every_state_by_the_same_amount(self):
        totals = self.controller.combine(self.base, self.potential, None,
                                         CONTEXT)
        expected = np.array([-1.0, -0.9, -0.8]) + self.potential.energy_correction
        np.testing.assert_allclose(np.asarray(totals.energy_total), expected,
                                   atol=0.0)
        # gaps are untouched by a state-independent addition
        gaps_before = np.diff([-1.0, -0.9, -0.8])
        gaps_after = np.diff(np.asarray(totals.energy_total))
        np.testing.assert_allclose(gaps_after, gaps_before, atol=1.0e-15)

    def test_gradient_adds_to_every_row_and_keeps_shape(self):
        totals = self.controller.combine(self.base, self.potential, None,
                                         CONTEXT)
        self.assertEqual(totals.gradient_total.shape, (3, 4, 3))
        for k in range(3):
            np.testing.assert_allclose(
                totals.gradient_total[k],
                self.base_gradient[k] + self.potential.gradient_correction,
                atol=0.0)

    def test_base_array_is_not_mutated(self):
        before = self.base_gradient.copy()
        self.controller.combine(self.base, self.potential, None, CONTEXT)
        np.testing.assert_allclose(self.base_gradient, before, atol=0.0)

    def test_row_count_must_match_the_energy(self):
        mismatched = {'energy': [-1.0, -0.9], 'gradient': self.base_gradient}
        with self.assertRaises(ctl.ControllerError):
            self.controller.combine(mismatched, self.potential, None, CONTEXT)

    def test_totals_survive_the_dict_round_trip(self):
        totals = self.controller.combine(self.base, self.potential, None,
                                         CONTEXT)
        restored = records.ConstraintResult.from_dict(totals.to_dict())
        np.testing.assert_allclose(np.asarray(restored.gradient_total),
                                   totals.gradient_total, atol=0.0)
        self.assertFalse(any(isinstance(value, np.ndarray)
                             for value in totals.to_dict().values()))


class TestMissingBaseGradient(unittest.TestCase):
    """A run that computed no gradient still has no gradient afterwards."""

    def setUp(self):
        self.controller, self.state = _controller()
        self.potential = _potential(self.controller, self.state)

    def test_energy_only_base_yields_no_total_gradient(self):
        totals = self.controller.combine({'energy': -1.0}, self.potential,
                                         None, CONTEXT)
        self.assertIsNone(totals.gradient_total)
        self.assertIsNone(totals.gradient_for_integrator)
        self.assertIsNone(totals.gradient_projected)

    def test_the_explicit_none_key_behaves_the_same(self):
        totals = self.controller.combine({'energy': -1.0, 'gradient': None},
                                         self.potential, None, CONTEXT)
        self.assertIsNone(totals.gradient_total)

    def test_the_bias_gradient_is_still_reported_on_its_own(self):
        """Diagnostics keep the additive force; it just is not the run's."""
        totals = self.controller.combine({'energy': -1.0}, self.potential,
                                         None, CONTEXT)
        self.assertIsNotNone(totals.gradient_correction)
        self.assertGreater(float(np.abs(
            totals.gradient_correction).max()), 0.0)

    def test_a_missing_base_gradient_is_not_filled_with_zeros(self):
        totals = self.controller.combine({'energy': -1.0}, self.potential,
                                         None, CONTEXT)
        self.assertIsNone(totals.gradient_base)
        self.assertIsNone(totals.gradient_total)
        self.assertEqual(totals.to_dict()['gradient_total'], None)


class TestFrozenComponentsAcrossStates(unittest.TestCase):

    def test_the_mask_reaches_every_state_row(self):
        """`atoms: [1]` is 1-based, so the frozen row is index 0."""
        controller, state = _controller(fix=True)
        potential = _potential(controller, state)
        base = {'energy': [-1.0, -0.9],
                'gradient': np.stack([np.full((4, 3), 0.25) for _ in range(2)])}
        totals = controller.combine(base, potential, None, CONTEXT)
        self.assertEqual(totals.gradient_total.shape, (2, 4, 3))
        self.assertEqual(float(np.abs(totals.gradient_projected[:, 0]).max()),
                         0.0)
        self.assertGreater(float(np.abs(totals.gradient_projected[:, 1]).max()),
                           0.0)
        # rows that stay are untouched, so the difference is exactly the
        # dropped part and nothing else
        difference = np.abs(totals.gradient_total
                            - totals.gradient_projected)
        self.assertEqual(float(difference[:, 1:].max()), 0.0)
        np.testing.assert_allclose(difference[:, 0],
                                   np.abs(totals.gradient_total[:, 0]),
                                   atol=0.0)

    def test_a_state_without_its_own_row_is_not_invented(self):
        """One energy value with a two-row gradient is a contradiction."""
        controller, state = _controller(fix=True)
        potential = _potential(controller, state)
        with self.assertRaises(ctl.ControllerError):
            controller.combine({'energy': [-1.0, -0.9, -0.8],
                                'gradient': np.full((2, 4, 3), 0.2)},
                               potential, None, CONTEXT)


if __name__ == '__main__':
    unittest.main()
