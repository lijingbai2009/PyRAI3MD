"""
======================================================================
Tests for the aligned RMSD as a history-potential variable.

The hill kernel is the one the CV metadynamics already uses, so what these
tests have to establish is the parts the new backend changes: that the
declaration normalizes into the squared deviation with a width in Bohr squared,
that the reference geometry is bound once and does not follow the run, that the
gradient reaches only the atoms of the aligned group, and that a degenerate
alignment fails as a typed error rather than as a nan inside a hill.

The reference is the load-bearing one. A variable defined as "the deviation
from where this run started" silently becomes identically zero if the reference
is read from the current geometry, and every test that deposits a hill would
still pass. So the refusal to re-read is asserted directly.

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
from pyrai3md.utilities import rmsd
from pyrai3md.workflow.constraints import (
    controller, metadynamics)
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

ATOMS = ['O', 'H', 'H', 'C', 'N', 'H']
MASSES = np.array([16.0, 1.0, 1.0, 12.0, 14.0, 1.0])
GEOMETRY = np.array([[0.00, 0.00, 0.00],
                    [1.60, 0.30, 0.10],
                    [0.40, 1.70, -0.20],
                    [2.10, 1.80, 1.30],
                    [-1.20, 0.70, 1.60],
                    [0.80, -1.50, 0.40]]) * ANGSTROM_TO_BOHR
GROUP = [1, 2, 3, 4]
HEIGHT = 1.0e-4
WIDTH = 0.10

# A change of shape for the aligned group. Moving every atom of the group by
# the same vector is a translation, and an aligned RMSD is blind to it, so a
# test that expected a deviation from that would be asserting nothing.
DISTORTION = np.array([[0.05, -0.02, 0.01],
                       [-0.03, 0.06, 0.02],
                       [0.01, 0.02, -0.05],
                       [-0.02, -0.04, 0.03]]) * ANGSTROM_TO_BOHR


def _changed(coords=GEOMETRY):
    """The geometry with its aligned group distorted."""
    moved = coords.copy()
    moved[GROUP] = moved[GROUP] + DISTORTION
    return moved


def _declaration(**extra):
    record = {'id': 'md', 'kind': 'rmsd', 'atoms': [atom + 1 for atom in GROUP],
              'height': HEIGHT, 'width': WIDTH, 'pace': 2}
    record.update(extra)
    return {'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
            'metadynamics': [record]}


def _plan(**extra):
    return schema.build_plan(_declaration(**extra), atoms=ATOMS, natom=6,
                             masses=MASSES)


def _context(step=2, trajectory='t'):
    return records.EvaluationContext(purpose='md', step=step,
                                     trajectory_id=trajectory)


def _bound(plan, coords=GEOMETRY, masses=MASSES):
    ctl = controller.SystemConstraintController(plan)
    return ctl, ctl.bind_reference(coords, masses=masses)


def _spec(**extra):
    """The single history action of a declaration, as a spec."""
    plan = _plan(**extra)
    return plan.spec(plan.ids('metadynamics')[0])


class TestNormalization(unittest.TestCase):

    def test_the_width_becomes_a_squared_deviation_scale(self):
        """A width of 0.1 Angstrom is 0.1 Bohr-converted, then squared.

        The variable is D^2, so the Gaussian must be written in Bohr squared or
        the declared width means nothing.
        """
        spec = _spec()
        sigma = float(spec.parameters['widths'][0])
        self.assertAlmostEqual(sigma,
                               (WIDTH * ANGSTROM_TO_BOHR) ** 2, places=14)

    def test_the_group_is_resolved_to_explicit_atoms(self):
        spec = _spec()
        self.assertEqual(list(spec.atoms), GROUP)
        self.assertEqual(spec.family, 'rmsd')
        self.assertEqual(spec.reference, 'initial')
        self.assertIsNone(spec.target)

    def test_a_collective_variable_alongside_rmsd_is_refused(self):
        """Two ways to name the same variable is a declaration error."""
        with self.assertRaises(schema.ConstraintInputError):
            _plan(cvs=['r'])

    def test_one_atom_cannot_be_aligned(self):
        with self.assertRaises(schema.ConstraintInputError):
            schema.build_plan(
                {'metadynamics': [{'id': 'md', 'kind': 'rmsd', 'atoms': [1],
                                   'height': HEIGHT, 'width': WIDTH}]},
                atoms=ATOMS, natom=6, masses=MASSES)

    def test_two_widths_for_one_variable_are_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _plan(width=[WIDTH, WIDTH])

    def test_mass_weighting_without_masses_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            schema.build_plan(_declaration(weight='mass'), atoms=ATOMS,
                              natom=6, masses=None)

    def test_an_unknown_weighting_is_refused(self):
        with self.assertRaises(Exception):
            _plan(weight='charge')

    def test_an_explicit_reference_block_is_kept_on_the_plan(self):
        block = GEOMETRY[GROUP] / ANGSTROM_TO_BOHR
        spec = _spec(reference=block.reshape(-1).tolist())
        self.assertEqual(spec.reference, 'explicit')
        np.testing.assert_allclose(spec.target, block * ANGSTROM_TO_BOHR,
                                   rtol=1.0e-12)

    def test_an_explicit_reference_of_the_wrong_size_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _spec(reference=[0.0, 0.0, 0.0, 1.0, 1.0])

    def test_reference_initial_is_the_same_as_omitting_it(self):
        self.assertEqual(_spec(reference='initial').reference, 'initial')


class TestBoundReference(unittest.TestCase):
    """The variable is the deviation from where the run started."""

    def test_the_reference_is_captured_at_binding(self):
        plan = _plan()
        _, state = _bound(plan)
        np.testing.assert_allclose(np.asarray(state.targets['md']),
                                   GEOMETRY[GROUP], atol=1.0e-14)

    def test_a_displaced_geometry_gives_the_deviation_it_was_bound_to(self):
        plan = _plan()
        ctl, state = _bound(plan)
        result = ctl.evaluate_potentials(_changed(), state, _context(step=0),
                                         masses=MASSES)
        self.assertGreater(result.components['md']['cv_values'][0], 0.0)

    def test_binding_against_a_later_geometry_does_not_move_the_target(self):
        """Negative control for the re-read the design forbids.

        If the reference were taken from the live state, aligning the run
        against itself would make the variable identically zero and every
        deposition test would still pass.
        """
        plan = _plan()
        ctl, _ = _bound(plan)
        moved = _changed()
        state = ctl.bind_reference(moved, masses=MASSES)
        result = ctl.evaluate_potentials(moved, state, _context(step=0),
                                         masses=MASSES)
        self.assertAlmostEqual(
            result.components['md']['cv_values'][0], 0.0, places=14)
        bound = ctl.bind_reference(GEOMETRY, masses=MASSES)
        earlier = ctl.evaluate_potentials(moved, bound, _context(step=0),
                                          masses=MASSES)
        self.assertGreater(earlier.components['md']['cv_values'][0], 0.0)

    def test_an_unbound_state_is_refused(self):
        plan = _plan()
        ctl = controller.SystemConstraintController(plan)
        state = records.ConstraintState()
        state.targets = {}
        with self.assertRaises(controller.ControllerError):
            ctl.evaluate_potentials(GEOMETRY, state, _context(step=0),
                                    masses=MASSES)


class TestHills(unittest.TestCase):
    """A hill deposited at one geometry is felt at another.

    Evaluating at the geometry a hill was deposited on gives a force of exactly
    zero, because that is the top of the Gaussian, so the gradient tests deposit
    at the bound reference and read the force at the distorted geometry.
    """

    def test_a_hill_lands_at_the_current_deviation(self):
        plan = _plan()
        ctl, state = _bound(plan)
        moved = _changed()
        fresh = ctl.commit(moved, state, _context(step=2), masses=MASSES)
        self.assertEqual(len(fresh.hills), 1)
        expected = rmsd.rmsd_squared(moved[GROUP], GEOMETRY[GROUP])
        self.assertAlmostEqual(float(fresh.hills[0].center[0]), expected,
                               places=14)

    def test_a_replayed_step_deposits_once(self):
        plan = _plan()
        ctl, state = _bound(plan)
        once = ctl.commit(GEOMETRY, state, _context(step=2), masses=MASSES)
        twice = ctl.commit(GEOMETRY, once, _context(step=2), masses=MASSES)
        self.assertEqual(len(twice.hills), 1)

    def test_the_hill_gradient_reaches_only_the_aligned_group(self):
        plan = _plan()
        ctl, state = _bound(plan)
        fresh = ctl.commit(GEOMETRY, state, _context(step=2), masses=MASSES)
        result = ctl.evaluate_potentials(_changed(), fresh, _context(step=4),
                                         masses=MASSES)
        gradient = result.components['md']['gradient']
        outside = [atom for atom in range(6) if atom not in GROUP]
        self.assertEqual(float(np.max(np.abs(gradient[outside]))), 0.0)
        self.assertGreater(float(np.max(np.abs(gradient[GROUP]))), 0.0)

    def test_the_hill_gradient_matches_a_central_difference(self):
        plan = _plan()
        ctl, state = _bound(plan)
        fresh = ctl.commit(GEOMETRY, state, _context(step=2), masses=MASSES)
        moved = _changed()

        def energy(coords):
            return ctl.evaluate_potentials(coords, fresh, _context(step=4),
                                           masses=MASSES).energy_correction
        gradient = ctl.evaluate_potentials(
            moved, fresh, _context(step=4),
            masses=MASSES).components['md']['gradient']
        step = 1.0e-6
        for atom in GROUP:
            for axis in range(3):
                up = moved.copy()
                up[atom, axis] += step
                down = moved.copy()
                down[atom, axis] -= step
                finite = (energy(up) - energy(down)) / (2 * step)
                self.assertAlmostEqual(gradient[atom, axis], finite,
                                       delta=1.0e-7)

    def test_mass_weighting_moves_the_heavy_atom_more(self):
        """The point of the weighting option, asserted as an ordering.

        Atom 4 of the group is N (14 amu) and atom 1 is H (1 amu). Under a
        uniform deviation their responses are comparable; under a mass-weighted
        one the nitrogen dominates, which is the behaviour a user selects.
        """
        def gradient(weight):
            plan = _plan(weight=weight)
            ctl, state = _bound(plan)
            fresh = ctl.commit(GEOMETRY, state, _context(step=2),
                               masses=MASSES)
            return ctl.evaluate_potentials(
                _changed(), fresh, _context(step=4),
                masses=MASSES).components['md']['gradient']
        plain = gradient('uniform')
        weighted = gradient('mass')
        self.assertGreater(abs(weighted[4][0]) / abs(weighted[1][0]),
                           abs(plain[4][0]) / abs(plain[1][0]))
        self.assertLess(float(np.max(np.abs(np.sum(weighted, axis=0)))),
                        1.0e-9)

    def test_the_rmsd_hill_is_one_dimensional(self):
        plan = _plan()
        ctl, state = _bound(plan)
        fresh = ctl.commit(GEOMETRY, state, _context(step=2), masses=MASSES)
        self.assertEqual(np.asarray(fresh.hills[0].center).size, 1)
        self.assertEqual(np.asarray(fresh.hills[0].width).size, 1)


class TestRefusals(unittest.TestCase):

    def test_a_missing_reference_geometry_is_reported_as_such(self):
        plan = _plan()
        spec = plan.spec(plan.ids('metadynamics')[0])
        with self.assertRaises(metadynamics.MetadynamicsError) as caught:
            metadynamics.evaluate(spec, plan, GEOMETRY, [], _context(),
                                  reference=None)
        self.assertIn('no reference geometry', str(caught.exception))

    def test_a_reference_of_the_wrong_shape_is_refused(self):
        plan = _plan()
        spec = plan.spec(plan.ids('metadynamics')[0])
        with self.assertRaises(metadynamics.MetadynamicsError):
            metadynamics.evaluate(spec, plan, GEOMETRY, [], _context(),
                                  reference=GEOMETRY[:2])

    def test_a_collapsed_geometry_raises_a_typed_error_not_a_nan(self):
        """The alignment failure has to arrive as a refusal.

        A nan in a hill centre is worse than an exception: it enters the
        Gaussian, the exponential underflows to zero, and the run continues with
        a history potential that is quietly absent.
        """
        plan = _plan()
        spec = plan.spec(plan.ids('metadynamics')[0])
        reference = GEOMETRY[GROUP] + np.array([0.0, 0.0, 3.0])
        flat = GEOMETRY[GROUP].copy()
        flat[:, 2] = float(np.mean(flat[:, 2]))
        collapsed = GEOMETRY.copy()
        collapsed[GROUP] = flat
        with self.assertRaises(metadynamics.MetadynamicsError):
            metadynamics.evaluate(spec, plan, collapsed, [], _context(),
                                  reference=reference)

    def test_mass_weighting_is_demanded_at_evaluation_too(self):
        plan = _plan(weight='mass')
        spec = plan.spec(plan.ids('metadynamics')[0])
        with self.assertRaises(metadynamics.MetadynamicsError):
            metadynamics.evaluate(spec, plan, GEOMETRY, [], _context(),
                                  reference=GEOMETRY[GROUP], masses=None)

    def test_a_gaussian_action_still_needs_its_cv(self):
        """The plan check is scoped to the family that reads cv_names."""
        plan = schema.build_plan(
            {'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
             'metadynamics': [{'id': 'md', 'cvs': ['r'], 'height': HEIGHT,
                               'width': [0.1], 'pace': 2},
                              {'id': 'rmsd', 'kind': 'rmsd',
                               'atoms': [2, 3, 4, 5], 'height': HEIGHT,
                               'width': WIDTH, 'pace': 2}]},
            atoms=ATOMS, natom=6, masses=MASSES)
        ctl = controller.SystemConstraintController(plan)
        self.assertEqual(len(ctl.plan.actions), 2)


if __name__ == '__main__':
    unittest.main()
