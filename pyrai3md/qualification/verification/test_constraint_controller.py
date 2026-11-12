"""
======================================================================
Tests for the staged controller.

The controller owns the two promises a workflow depends on and cannot check
for itself: that a total is always rebuilt from the stored base rather than
accumulated onto a previous total, and that an empty plan costs nothing at all.

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
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

ATOMS = ['O', 'H', 'H', 'C']
MASSES = np.array([16.0, 1.0, 1.0, 12.0])
BASE_GEOMETRY = np.array([[0.00, 0.00, 0.00],
                          [1.60, 0.30, 0.10],
                          [0.40, 1.70, -0.20],
                          [2.10, 1.80, 1.30]])
BASE_RESULT = {'energy': -1.2345, 'gradient': np.arange(12,
                                                        dtype=np.float64
                                                        ).reshape(4, 3) * 0.01}

FULL = {'groups': [{'name': 'g', 'atoms': [1, 2]}],
        'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
        'biases': [{'id': 'b', 'cv': 'r', 'target': 1.6,
                    'force_constant': 0.02, 'region': 'all'}],
        'walls': [{'id': 'w', 'geometry': 'sphere', 'center': [0, 0, 0],
                   'radius': 0.5, 'atoms': [3, 4], 'power': 2,
                   'force_constant': 0.005}],
        'fixes': [{'id': 'f', 'group': 'g', 'direction': 'xyz'}],
        'constraints': [{'id': 'x', 'bonds': [[2, 3]],
                         'tolerance': 1.0e-10, 'max_iterations': 500}],
        'metadynamics': [{'id': 'm', 'cvs': ['r'], 'height': 1.0e-4,
                          'width': [0.1], 'pace': 2}]}


def _plan(mapping=FULL):
    return schema.build_plan(mapping, atoms=ATOMS, natom=4, masses=MASSES)


def _context(step=1, purpose='md'):
    return records.EvaluationContext(purpose=purpose, step=step, dt=20.67,
                                     evaluation_id='e%d' % step)


class TestEmptyPlan(unittest.TestCase):
    """DC-1 and DC-3: an empty plan must not exist as far as the run cares."""

    def setUp(self):
        self.controller = ctl.SystemConstraintController(
            records.ConstraintPlan.empty())

    def test_the_controller_reports_noop(self):
        self.assertTrue(self.controller.is_noop())

    def test_potential_evaluation_allocates_no_arrays(self):
        result = self.controller.evaluate_potentials(BASE_GEOMETRY,
                                                     records
                                                     .ConstraintState(),
                                                     _context(),
                                                     masses=MASSES)
        self.assertTrue(result.is_empty)
        self.assertIsNone(result.gradient_correction)
        self.assertIsNone(result.coordinates)
        self.assertEqual(result.components, {})
        self.assertTrue(result.provenance['plan_empty'])

    def test_projection_returns_the_same_numbers(self):
        coords, info = self.controller.project_positions(BASE_GEOMETRY,
                                                         records
                                                         .ConstraintState(),
                                                         _context(),
                                                         masses=MASSES)
        np.testing.assert_allclose(coords, BASE_GEOMETRY, atol=0.0)
        self.assertFalse(info['projected'])

    def test_velocities_are_returned_untouched(self):
        velocities = np.ones((4, 3))
        got = self.controller.project_velocities(velocities,
                                                 records.ConstraintState())
        np.testing.assert_allclose(got, velocities, atol=0.0)

    def test_statistics_count_the_plain_system(self):
        statistics = self.controller.statistics(BASE_GEOMETRY,
                                                np.ones((4, 3)), MASSES)
        self.assertEqual(statistics.free_dof, 12)
        self.assertEqual(statistics.rank, 0)

    def test_commit_only_moves_the_step_counter(self):
        state = records.ConstraintState()
        grown = self.controller.commit(BASE_GEOMETRY, state, _context(9),
                                       masses=MASSES)
        self.assertEqual(grown.hills, [])
        self.assertEqual(grown.last_committed_step, 9)
        self.assertEqual(state.hills, [])

    def test_combine_still_reports_the_base_unchanged(self):
        potential = self.controller.evaluate_potentials(
            BASE_GEOMETRY, records.ConstraintState(), _context(),
            masses=MASSES)
        totals = self.controller.combine(dict(BASE_RESULT), potential, None,
                                         _context())
        self.assertAlmostEqual(totals.energy_total, BASE_RESULT['energy'],
                               places=14)
        np.testing.assert_allclose(totals.gradient_total,
                                   BASE_RESULT['gradient'], atol=0.0)

    def test_a_factory_refuses_to_build_for_an_empty_plan(self):
        self.assertIsNone(ctl.controller_for(_plan({'groups': []})))
        self.assertIsNone(ctl.controller_for(None))


class TestComposition(unittest.TestCase):
    """base + sum(components) == total, always rebuilt from the base."""

    def setUp(self):
        mapping = dict(FULL)
        mapping.pop('constraints')
        mapping.pop('metadynamics')
        mapping.pop('fixes')
        self.plan = _plan(mapping)
        self.controller = ctl.SystemConstraintController(self.plan)
        self.state = self.controller.bind_reference(BASE_GEOMETRY,
                                                    masses=MASSES)

    def _evaluate(self, coords):
        context = _context(1, purpose='single')
        potential = self.controller.evaluate_potentials(coords, self.state,
                                                        context,
                                                        masses=MASSES)
        return self.controller.combine(dict(BASE_RESULT), potential, None,
                                       context)

    def test_component_energies_sum_into_the_total(self):
        totals = self._evaluate(BASE_GEOMETRY)
        self.assertGreater(len(totals.components), 1)
        subtotal = sum(item['energy'] for item in totals.components.values())
        self.assertAlmostEqual(totals.energy_base + subtotal,
                               totals.energy_total, places=14)
        self.assertAlmostEqual(totals.energy_correction, subtotal, places=14)

    def test_combining_twice_changes_nothing(self):
        context = _context(1, purpose='single')
        potential = self.controller.evaluate_potentials(BASE_GEOMETRY,
                                                        self.state, context,
                                                        masses=MASSES)
        first = self.controller.combine(dict(BASE_RESULT), potential, None,
                                        context)
        second = self.controller.combine(dict(BASE_RESULT), potential, None,
                                         context)
        self.assertAlmostEqual(first.energy_total, second.energy_total,
                               places=15)
        np.testing.assert_allclose(first.gradient_total, second.gradient_total,
                                   atol=0.0)
        # a second combine must not have accumulated onto the first
        self.assertAlmostEqual(second.energy_total, first.energy_total,
                               places=15)

    def test_the_potential_evaluation_is_not_mutated_by_combine(self):
        context = _context(1, purpose='single')
        potential = self.controller.evaluate_potentials(BASE_GEOMETRY,
                                                        self.state, context,
                                                        masses=MASSES)
        before = potential.energy_correction
        self.controller.combine(dict(BASE_RESULT), potential, None, context)
        self.assertAlmostEqual(potential.energy_correction, before, places=15)

    def test_gradient_total_is_the_base_plus_the_correction(self):
        totals = self._evaluate(BASE_GEOMETRY)
        expected = BASE_RESULT['gradient'] + totals.gradient_correction
        np.testing.assert_allclose(totals.gradient_total, expected, atol=0.0)

    def test_a_missing_base_stays_missing(self):
        context = _context(1, purpose='single')
        potential = self.controller.evaluate_potentials(BASE_GEOMETRY,
                                                        self.state, context,
                                                        masses=MASSES)
        totals = self.controller.combine(None, potential, None, context)
        self.assertIsNone(totals.energy_base)
        self.assertIsNone(totals.gradient_base)
        self.assertAlmostEqual(totals.energy_total, totals.energy_correction,
                               places=15)

    def test_flat_keys_are_printed_by_the_existing_reader(self):
        from pyrai3md.utilities import print_results
        totals = self._evaluate(BASE_GEOMETRY)
        text = print_results._format_energy(totals.energy_total,
                                            totals.flat_contributions())
        self.assertIsInstance(text, str)
        self.assertIn('&energy contributions (constraint/multiscale)', text)
        self.assertIn('constraint.bias.b', text)
        self.assertIn('constraint.wall.w', text)

    def test_provenance_names_the_geometry_history_and_step(self):
        totals = self._evaluate(BASE_GEOMETRY)
        self.assertEqual(totals.provenance['step'], 1)
        self.assertEqual(totals.provenance['plan_fingerprint'],
                         self.plan.fingerprint)
        self.assertEqual(totals.provenance['history_version'], 0)

    def test_the_integrator_drive_starts_as_the_total(self):
        totals = self._evaluate(BASE_GEOMETRY)
        np.testing.assert_allclose(totals.gradient_for_integrator,
                                   totals.gradient_total, atol=0.0)


class TestFixesAndProjection(unittest.TestCase):

    def setUp(self):
        self.plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                           'fixes': [{'id': 'f', 'group': 'g',
                                      'direction': 'xyz'}]})
        self.controller = ctl.SystemConstraintController(self.plan)
        self.state = self.controller.bind_reference(BASE_GEOMETRY,
                                                    masses=MASSES)

    def test_frozen_atoms_do_not_move(self):
        drifted = BASE_GEOMETRY + 0.4
        coords, info = self.controller.project_positions(drifted, self.state,
                                                         _context(),
                                                         masses=MASSES)
        np.testing.assert_allclose(coords[:2], BASE_GEOMETRY[:2], atol=0.0)
        self.assertEqual(info['restored_components'], 6)
        self.assertTrue(info['projected'])

    def test_mobile_atoms_keep_their_drift(self):
        drifted = BASE_GEOMETRY + 0.4
        coords, _ = self.controller.project_positions(drifted, self.state,
                                                      _context(),
                                                      masses=MASSES)
        np.testing.assert_allclose(coords[2:], drifted[2:], atol=0.0)

    def test_frozen_velocities_are_zeroed_every_step(self):
        velocities = np.full((4, 3), 0.2)
        got = self.controller.project_velocities(velocities, self.state)
        self.assertEqual(float(np.abs(got[:2]).max()), 0.0)
        self.assertAlmostEqual(float(got[2, 0]), 0.2, places=12)

    def test_the_projected_gradient_zeroes_frozen_components(self):
        mapping = dict(FULL)
        mapping.pop('constraints')
        mapping.pop('metadynamics')
        plan = _plan(mapping)
        controller = ctl.SystemConstraintController(plan)
        state = controller.bind_reference(BASE_GEOMETRY, masses=MASSES)
        context = _context(1, purpose='single')
        potential = controller.evaluate_potentials(BASE_GEOMETRY, state,
                                                   context, masses=MASSES)
        totals = controller.combine(dict(BASE_RESULT), potential, None,
                                    context)
        frozen = totals.gradient_projected[:2]
        self.assertEqual(float(np.abs(frozen).max()), 0.0)
        difference = np.abs(totals.gradient_projected
                            - totals.gradient_total)
        self.assertAlmostEqual(float(difference[:2].max()),
                               float(np.abs(totals.gradient_total[:2]).max()),
                               places=15)
        self.assertEqual(float(difference[2:].max()), 0.0)

    def test_statistics_lose_the_frozen_components(self):
        statistics = self.controller.statistics(BASE_GEOMETRY,
                                                np.ones((4, 3)), MASSES,
                                                state=self.state)
        self.assertEqual(statistics.free_dof, 6)
        self.assertEqual(statistics.mobile_components, 6)


class TestBondProjection(unittest.TestCase):

    def setUp(self):
        self.plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [3, 4]}],
                           'constraints': [{'id': 'x', 'bonds': [[3, 4]],
                                            'target': 1.0,
                                            'tolerance': 1.0e-10,
                                            'max_iterations': 500}]})
        self.controller = ctl.SystemConstraintController(self.plan)
        self.state = self.controller.bind_reference(BASE_GEOMETRY,
                                                    masses=MASSES)

    def test_a_stretched_bond_is_pulled_back(self):
        drifted = BASE_GEOMETRY.copy()
        drifted[3] += np.array([0.9, 0.0, 0.0])
        coords, info = self.controller.project_positions(drifted, self.state,
                                                         _context(),
                                                         masses=MASSES)
        length = float(np.linalg.norm(coords[2] - coords[3]))
        self.assertAlmostEqual(length, 1.0 * ANGSTROM_TO_BOHR, places=9)
        self.assertTrue(info['converged'])
        self.assertLessEqual(info['max_residual'], 1.0e-10)

    def test_masses_are_required_and_theabsence_is_reported(self):
        drifted = BASE_GEOMETRY.copy()
        drifted[3] += np.array([0.9, 0.0, 0.0])
        with self.assertRaises(ctl.ControllerError):
            self.controller.project_positions(drifted, self.state, _context(),
                                              masses=None)

    def test_the_bond_target_comes_from_the_reference_geometry(self):
        plan = _plan({'constraints': [{'id': 'x', 'bonds': [[1, 2]]}]})
        controller = ctl.SystemConstraintController(plan)
        state = controller.bind_reference(BASE_GEOMETRY, masses=MASSES)
        reference = float(np.linalg.norm(BASE_GEOMETRY[0]
                                         - BASE_GEOMETRY[1]))
        np.testing.assert_allclose(state.targets['x'], [reference],
                                   atol=1.0e-14)
        drifted = BASE_GEOMETRY.copy()
        drifted[1] += np.array([0.5, 0.5, 0.0])
        coords, _ = controller.project_positions(drifted, state, _context(),
                                                 masses=MASSES)
        self.assertAlmostEqual(float(np.linalg.norm(coords[0] - coords[1])),
                               reference, places=10)

    def test_a_missing_binding_is_reported_not_re_bound(self):
        """An 'initial' target must not be re-captured from a drifted frame."""
        plan = _plan({'constraints': [{'id': 'x', 'bonds': [[1, 2]]}]})
        controller = ctl.SystemConstraintController(plan)
        drifted = BASE_GEOMETRY.copy()
        drifted[1] += 0.5
        with self.assertRaises(ctl.ControllerError):
            controller.project_positions(drifted, records.ConstraintState(),
                                         _context(), masses=MASSES)


class TestCommit(unittest.TestCase):

    def setUp(self):
        self.plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'metadynamics': [{'id': 'm', 'cvs': ['r'],
                                             'height': 1.0e-4,
                                             'width': [0.1], 'pace': 2}]})
        self.controller = ctl.SystemConstraintController(self.plan)
        self.state = self.controller.bind_reference(BASE_GEOMETRY,
                                                    masses=MASSES)

    def test_deposition_happens_on_pace_and_never_twice(self):
        state = self.state
        lengths = []
        for step in range(0, 7):
            state = self.controller.commit(BASE_GEOMETRY, state,
                                           _context(step), masses=MASSES)
            lengths.append(len(state.hills))
        self.assertEqual(lengths, [0, 0, 1, 1, 2, 2, 3])

    def test_a_replayed_step_leaves_the_history_alone(self):
        once = self.controller.commit(BASE_GEOMETRY, self.state, _context(2),
                                      masses=MASSES)
        twice = self.controller.commit(BASE_GEOMETRY, once, _context(2),
                                       masses=MASSES)
        self.assertEqual(len(once.hills), len(twice.hills))
        self.assertEqual(once.history_version, twice.history_version)

    def test_the_committed_state_is_a_new_object(self):
        grown = self.controller.commit(BASE_GEOMETRY, self.state, _context(2),
                                       masses=MASSES)
        self.assertEqual(self.state.hills, [])
        self.assertEqual(len(grown.hills), 1)

    def test_the_new_hill_changes_the_next_force_not_this_one(self):
        context = _context(2)
        before = self.controller.evaluate_potentials(BASE_GEOMETRY, self.state,
                                                     context, masses=MASSES)
        grown = self.controller.commit(BASE_GEOMETRY, self.state, context,
                                       masses=MASSES)
        after = self.controller.evaluate_potentials(BASE_GEOMETRY, grown,
                                                    context, masses=MASSES)
        self.assertEqual(before.components['m']['energy'], 0.0)
        self.assertGreater(after.components['m']['energy'], 0.0)
        self.assertEqual(after.components['m']['parameters']['hills'], 1)

    def test_two_trajectories_deposit_on_their_own_clock(self):
        first = self.controller.commit(BASE_GEOMETRY, self.state,
                                       _context(2), masses=MASSES)
        second = self.controller.commit(BASE_GEOMETRY, self.state,
                                        _context(2), masses=MASSES)
        self.assertEqual(len(first.hills), 1)
        self.assertEqual(len(second.hills), 1)
        both = self.controller.commit(BASE_GEOMETRY, first, _context(4),
                                      masses=MASSES)
        self.assertEqual(len(both.hills), 2)
        self.assertEqual(len(first.hills), 1)


class TestApplyStructure(unittest.TestCase):

    def setUp(self):
        self.plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 0.02}],
                           'fixes': [{'atoms': [3], 'direction': 'xyz'}]})
        self.controller = ctl.SystemConstraintController(self.plan)

    def test_without_an_evaluator_only_the_correction_is_reported(self):
        drifted = BASE_GEOMETRY + 0.3
        result = self.controller.apply_structure(drifted, masses=MASSES,
                                                 context=_context(0, 'single'))
        self.assertIsNone(result.energy_base)
        self.assertIsNotNone(result.energy_total)
        self.assertGreater(result.energy_correction, 0.0)
        # Atom 1 is mobile, so it keeps the drift it arrived with.
        np.testing.assert_allclose(result.coordinates[0], drifted[0],
                                   atol=0.0)

    def test_the_evaluator_sees_the_projected_geometry(self):
        seen = {}

        def evaluator(coords):
            seen['coords'] = coords.copy()
            return {'energy': -2.0, 'gradient': np.zeros_like(coords)}

        drifted = BASE_GEOMETRY + 0.3
        result = self.controller.apply_structure(drifted, masses=MASSES,
                                                 evaluator=evaluator,
                                                 context=_context(0, 'single'))
        np.testing.assert_allclose(seen['coords'], result.coordinates,
                                   atol=0.0)
        self.assertAlmostEqual(result.energy_total,
                               -2.0 + result.energy_correction, places=14)

    def test_an_evaluator_and_a_base_result_together_are_refused(self):
        with self.assertRaises(ctl.ControllerError):
            self.controller.apply_structure(
                BASE_GEOMETRY, masses=MASSES,
                evaluator=lambda x: {'energy': 0.0,
                                     'gradient': np.zeros_like(x)},
                base={'energy': 0.0, 'gradient': np.zeros_like(BASE_GEOMETRY)})

    def test_a_base_that_reports_its_geometry_must_match(self):
        base = dict(BASE_RESULT)
        base['geometry'] = BASE_GEOMETRY + 1.0
        with self.assertRaises(ctl.ControllerError):
            self.controller.apply_structure(BASE_GEOMETRY + 0.3,
                                            masses=MASSES, base=base,
                                            context=_context(0, 'single'))

    def test_delta_coordinates_describe_the_correction_made(self):
        drifted = BASE_GEOMETRY + 0.3
        result = self.controller.apply_structure(drifted, masses=MASSES,
                                                 context=_context(0, 'single'))
        self.assertEqual(result.delta_coordinates.shape, (4, 3))
        shift = np.abs(result.delta_coordinates[0]).max()
        self.assertAlmostEqual(float(shift), 0.0, places=14)


class TestControllerGuards(unittest.TestCase):

    def test_a_plan_cannot_be_built_with_a_dangling_reference(self):
        """The plan refuses it, so no controller ever sees such a reference."""
        plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
                      'biases': [{'cv': 'r', 'target': 1.0,
                                  'force_constant': 0.02}]})
        with self.assertRaises(KeyError):
            records.ConstraintPlan(groups=plan.groups, cvs={},
                                   actions=list(plan.actions),
                                   units=plan.units)

    def test_an_unknown_base_type_is_refused(self):
        plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
                      'biases': [{'cv': 'r', 'target': 1.0,
                                  'force_constant': 0.02}]})
        controller = ctl.SystemConstraintController(plan)
        state = controller.bind_reference(BASE_GEOMETRY, masses=MASSES)
        context = _context(0, 'single')
        potential = controller.evaluate_potentials(BASE_GEOMETRY, state,
                                                   context, masses=MASSES)
        with self.assertRaises(ctl.ControllerError):
            controller.combine('not a result', potential, None, context)


class TestAnalyticHessian(unittest.TestCase):
    """C: the curvature of a bias, exact and without a difference step.

    `NumericalHessian` reaches the same number by differencing displaced
    gradients, each of which already carries the bias force, so this is not a
    new physical term - it is the term without a step size in it. The tests
    compare against a central difference of the controller's own correction, and
    then check that every shape it cannot produce is refused by name instead of
    quietly left out.
    """

    def _manager(self, kind, atoms, target=None, wall=False, meta=False):
        block = {'cvs': [{'name': 'v', 'kind': kind, 'atoms': atoms}],
                 'biases': [{'id': 'hold', 'cv': 'v', 'target': target,
                             'force_constant': 0.05}]}
        if wall:
            block['walls'] = [{'id': 'vessel', 'geometry': 'sphere',
                               'center': [1.0, 1.0, 1.0], 'radius': 2.0,
                               'atoms': [4], 'force_constant': 0.01}]
        if meta:
            block['metadynamics'] = [{'id': 'md', 'cvs': ['v'],
                                      'height': 1.0e-4, 'width': [0.1],
                                      'pace': 1}]
        plan = schema.build_plan(block, atoms=list(ATOMS), natom=len(ATOMS),
                                 masses=list(MASSES))
        manager = ctl.SystemConstraintController(plan)
        state = manager.bind_reference(BASE_GEOMETRY, masses=MASSES)
        return manager, state

    def _numerical(self, manager, state, masses=None):
        size = BASE_GEOMETRY.size
        step = 1.0e-5

        def gradient(array):
            correction = manager.evaluate_potentials(
                array, state, None, masses=masses).gradient_correction
            if correction is None:
                return np.zeros(size)
            return np.asarray(correction, dtype=np.float64).reshape(-1)

        out = np.zeros((size, size))
        for column in range(size):
            shift = np.zeros(size)
            shift[column] = step
            out[:, column] = (
                gradient(BASE_GEOMETRY + shift.reshape(-1, 3))
                - gradient(BASE_GEOMETRY - shift.reshape(-1, 3))) / (2 * step)
        return out

    def _assert_agrees_aggregate(self, manager, state):
        """Same comparison for a plan whose correction needs masses."""
        analytic = manager.hessian_block(BASE_GEOMETRY, state,
                                         masses=MASSES)['hessian']
        numerical = self._numerical(manager, state, masses=MASSES)
        scale = float(np.abs(numerical).max())
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(analytic, numerical, rtol=0.0,
                                   atol=scale * 1.0e-6)
        np.testing.assert_allclose(analytic, analytic.T,
                                   atol=scale * 1.0e-12)

    def _assert_agrees(self, manager, state):
        analytic = manager.hessian_block(BASE_GEOMETRY, state)['hessian']
        numerical = self._numerical(manager, state)
        np.testing.assert_allclose(analytic, numerical, rtol=0.0,
                                   atol=float(np.abs(numerical).max()) * 1e-6)
        self.assertGreater(float(np.abs(numerical).max()), 0.0)
        # Newton's third law inside the block: no net force from a shift.
        np.testing.assert_allclose(analytic.sum(axis=1), 0.0,
                                   atol=float(np.abs(numerical).max()) * 1e-12)
        np.testing.assert_allclose(analytic, analytic.T,
                                   atol=float(np.abs(numerical).max()) * 1e-12)

    def test_bond_bias_block_matches_the_differenced_gradient(self):
        manager, state = self._manager('bond', [1, 2])
        self._assert_agrees(manager, state)
        self.assertEqual(manager.hessian_block(BASE_GEOMETRY,
                                              state)['ids'], ['hold'])

    def test_angle_bias_off_target_keeps_the_variable_term(self):
        """Away from the target the curvature is not only `k J J^T`.

        The `e Q` term is the same order as the first one here (the angle sits
        ~0.2 rad from its target), so dropping it would miss by orders of
        magnitude rather than by the assertion tolerance.
        """
        manager, state = self._manager('angle', [1, 2, 3], target=1.4)
        self._assert_agrees(manager, state)

    def test_torsion_bias_block_matches_the_differenced_gradient(self):
        """The hardest curvature of the three, against the same yardstick.

        A wrapped periodic difference is in play here, so this also pins that
        the chain factor of +/-1 is applied to both terms and not forgotten in
        the `e Q` one.
        """
        manager, state = self._manager('dihedral', [1, 2, 3, 4])
        self._assert_agrees(manager, state)

    def test_every_wall_geometry_agrees_with_the_differenced_gradient(self):
        """Four geometries, two target kinds, two powers.

        A box and a triclinic face are linear in the coordinates, so their
        curvature is genuinely zero and only the `J J^T` term is left; a sphere
        and an ellipsoid carry a projector, and the ellipsoid's is the one that
        needs the scaled-space factor. All of them are checked against the same
        yardstick as the bias, which is the only way the difference shows.
        """
        geometries = [
            # Small enough that every member and every group center sits
            # outside: a wall nobody has touched has a zero Hessian on both
            # sides of the comparison, which would pass without proving
            # anything (the scale guard below is what makes that visible).
            ('sphere', {'center': [1.0, 1.0, 1.0], 'radius': 0.6}),
            ('ellipsoid', {'center': [1.0, 1.0, 1.0],
                            'semiaxes': [0.6, 0.7, 0.5]}),
            ('box', {'center': [1.0, 1.0, 1.0],
                     'lengths': [1.0, 1.0, 1.0]}),
            ('triclinic', {'center': [1.0, 1.0, 1.0],
                            'cell': [[1.0, 0.0, 0.0], [0.3, 0.9, 0.0],
                                     [0.15, 0.2, 1.1]]}),
        ]
        for geometry, extra in geometries:
            for target, atoms in ((None, [2, 3]), ('com', [1, 2, 3]),
                                  ('cog', [1, 2, 3])):
                for power in (2, 3):
                    record = {'id': 'w', 'geometry': geometry, 'atoms': atoms,
                              'force_constant': 0.02, 'power': power}
                    record.update(extra)
                    if target:
                        record['target'] = target
                    plan = schema.build_plan({'walls': [record]},
                                             atoms=list(ATOMS),
                                             natom=len(ATOMS),
                                             masses=list(MASSES))
                    manager = ctl.SystemConstraintController(plan)
                    state = manager.bind_reference(BASE_GEOMETRY,
                                                  masses=MASSES)
                    with self.subTest(geometry=geometry, target=target,
                                      power=power):
                        self._assert_agrees_aggregate(manager, state)

    def test_an_uncrossed_wall_contributes_no_curvature(self):
        plan = schema.build_plan(
            {'walls': [{'id': 'w', 'geometry': 'sphere', 'center': [0.0, 0.0,
                                                                    0.0],
                        'radius': 5.0, 'atoms': [2, 3],
                        'force_constant': 0.02}]},
            atoms=list(ATOMS), natom=len(ATOMS), masses=list(MASSES))
        manager = ctl.SystemConstraintController(plan)
        state = manager.bind_reference(BASE_GEOMETRY, masses=MASSES)
        out = manager.hessian_block(BASE_GEOMETRY, state, masses=MASSES)
        np.testing.assert_array_equal(
            out['hessian'], np.zeros((BASE_GEOMETRY.size, BASE_GEOMETRY.size)))

    def test_an_unknown_cv_kind_is_refused_not_assumed_flat(self):
        """A variable with no derivative at all must not read as zero."""
        plan = schema.build_plan(
            {'cvs': [{'name': 'v', 'kind': 'bond', 'atoms': [1, 2]}],
             'biases': [{'id': 'hold', 'cv': 'v', 'force_constant': 0.05}]},
            atoms=list(ATOMS), natom=len(ATOMS), masses=list(MASSES))
        manager = ctl.SystemConstraintController(plan)
        state = manager.bind_reference(BASE_GEOMETRY, masses=MASSES)
        # The kind is switched after binding: a CV the gradient layer does not
        # know must be refused where the curvature is taken, not read as flat.
        plan.cvs['v'].kind = 'cell_length'
        from pyrai3md.workflow.constraints import bias
        with self.assertRaises(bias.BiasError) as caught:
            bias.hessian(plan.spec('hold'), plan, BASE_GEOMETRY, state)
        self.assertIn('unknown cv kind', str(caught.exception))

    def _energy_matrix(self, manager, state, context, x, keys, step=1.0e-3):
        """Four-point double difference of the additive ENERGY.

        The step is 1e-3 Bohr. Neither the 1e-5 a first difference uses nor a
        hair larger: a second difference divides by `4 * step^2`, so too small a
        step buries the signal in the rounding of an energy of order 1e-4
        Hartree, and too large one leaves truncation behind. Measured on this
        fixture, the worst relative gap between the analytic block and this
        reference falls as `step^2` - 5.3e-03 at 1e-2, 4.8e-04 at 3e-3,
        5.3e-05 at 1e-3, 4.8e-06 at 3e-4 - which is what makes the comparison a
        test of the formula rather than of the instrument. The tolerance asserted
        below is that ladder read at the step used, rounded up.
        """
        size = x.size

        def energy(y):
            return float(manager.evaluate_potentials(y, state, context,
                                                     masses=MASSES)
                         .energy_correction)

        def shift(y, k, sign):
            other = y.copy()
            other.reshape(-1)[k] += sign * step
            return other

        out = {}
        for i, j in keys:
            out[(i, j)] = (
                energy(shift(shift(x, i, 1), j, 1))
                - energy(shift(shift(x, i, 1), j, -1))
                - energy(shift(shift(x, i, -1), j, 1))
                + energy(shift(shift(x, i, -1), j, -1))) / (4.0 * step * step)
        return out

    def test_the_energy_instrument_agrees_on_a_known_bias(self):
        """Validate the instrument before trusting it on the hills."""
        manager, state = self._manager('angle', [1, 2, 3], target=1.4)
        analytic = manager.hessian_block(BASE_GEOMETRY, state)['hessian']
        keys = [(3, 3), (0, 0), (6, 6), (1, 4)]
        truth = self._energy_matrix(manager, state, None, BASE_GEOMETRY, keys)
        for key, value in truth.items():
            self.assertGreater(abs(value), 0.0)
            self.assertLess(abs(analytic[key] - value) / abs(value), 2.0e-4)

    def test_hill_curvature_matches_the_energy_double_difference(self):
        """A6's curvature: one and two variables, single and several hills.

        The offsets are fractions of each hill's own width, which is what makes
        the test bite: a hill deposited exactly where the run is sitting has a
        vanishing `dV/ds` and a cancelling `d2V/ds2`, and a formula wrong by one
        power of sigma passes there while missing everywhere else.
        """
        from pyrai3md.data import constraint as records
        from pyrai3md.utilities import cv
        cases = (
            ('one cv, one sigma off',
             [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}], ['r'], [0.3],
             [(1.0,)]),
            ('one cv, two sigma off',
             [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}], ['r'], [0.3],
             [(2.0,)]),
            ('angle cv, one sigma off',
             [{'name': 'a', 'kind': 'angle', 'atoms': [2, 1, 3]}], ['a'],
             [20.0], [(1.0,)]),
            ('two cvs, both off',
             [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]},
              {'name': 'a', 'kind': 'angle', 'atoms': [2, 1, 3]}], ['r', 'a'],
             [0.3, 20.0], [(0.4, -0.7)]),
            ('two cvs, three hills',
             [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]},
              {'name': 'a', 'kind': 'angle', 'atoms': [2, 1, 3]}], ['r', 'a'],
             [0.3, 20.0], [(0.6, -0.4), (-1.2, 1.1), (0.0, 1.8)]),
        )
        keys = [(3, 3), (0, 0), (6, 6), (1, 4)]
        for tag, cvs, names, widths, offsets in cases:
            plan = schema.build_plan(
                {'cvs': cvs,
                 'metadynamics': [{'id': 'md', 'cvs': names,
                                   'height': 2.0e-4, 'width': widths,
                                   'pace': 1}]},
                atoms=list(ATOMS), natom=len(ATOMS), masses=list(MASSES))
            manager = ctl.SystemConstraintController(plan)
            spec = plan.spec('md')
            sigma = np.asarray(spec.parameters['widths'], dtype=np.float64)
            values = np.array([
                cv.distance(BASE_GEOMETRY, *[int(a) for a in
                                              plan.cvs[n].indices])
                if plan.cvs[n].kind == 'bond' else
                cv.angle(BASE_GEOMETRY, *[int(a) for a in
                                          plan.cvs[n].indices])
                for n in names])
            state = manager.bind_reference(BASE_GEOMETRY, masses=MASSES)
            state.hills = [records.HillRecord(
                values + np.array(offset) * sigma, sigma, 2.0e-4,
                token=('t', 5, 'md')) for offset in offsets]
            context = records.EvaluationContext(purpose='md', step=5,
                                                trajectory_id='t')
            analytic = manager.hessian_block(BASE_GEOMETRY, state, context,
                                             masses=MASSES)['hessian']
            truth = self._energy_matrix(manager, state, context,
                                        BASE_GEOMETRY, keys)
            for key, value in truth.items():
                with self.subTest(case=tag, entry=key):
                    if abs(value) < 1.0e-9:
                        self.assertLess(abs(analytic[key]), 1.0e-9)
                        continue
                    self.assertLess(abs(analytic[key] - value) / abs(value),
                                    2.0e-4)

    def test_a_one_sided_region_is_flat_in_curvature_too(self):
        """Clamped side: no energy, no force, and no curvature.

        `V = 1/2 k max(delta, 0)^2` is flat where it is clamped, so a block that
        reports the wrong zero there disagrees with the energy and the gradient
        the run already published - the trap is that the chain factor the
        gradient uses survives the clamp (`d(raw delta)/d(value)` is 1.0), so it
        cannot carry this information and the activity test has to be separate.
        """
        def assembled(region, bond):
            plan = schema.build_plan(
                {'cvs': [{'name': 'v', 'kind': 'bond', 'atoms': [1, 2]}],
                 'biases': [{'id': 'b', 'cv': 'v', 'target': 1.0,
                             'force_constant': 0.05, 'region': region}]},
                atoms=list(ATOMS), natom=len(ATOMS), masses=list(MASSES),
                coordinates=list(BASE_GEOMETRY))
            manager = ctl.SystemConstraintController(plan)
            # The pair is laid out on x so that the variable is exactly the
            # bond length: BASE_GEOMETRY puts atom 2 off-axis, and a test that
            # moved only x would be comparing against a distance that is not the
            # number in the table.
            x = np.array(BASE_GEOMETRY, dtype=np.float64)
            x[0, :] = 0.0
            x[1, :] = (bond * ANGSTROM_TO_BOHR, 0.0, 0.0)
            state = manager.bind_reference(x, masses=MASSES)
            result = manager.evaluate_potentials(x, state, None, masses=MASSES)
            block = manager.hessian_block(x, state, masses=MASSES)['hessian']
            return (float(result.energy_correction),
                    float(np.abs(np.asarray(result.gradient_correction,
                                            dtype=np.float64)).max()),
                    float(np.abs(block).max()))

        # No case sits exactly on the kink of a one-sided region: there the
        # sign of the raw difference is decided by the last bits of the distance
        # evaluation, so both answers are arithmetically defensible and neither
        # is a fact worth pinning. The two-sided case at the target is pinned
        # separately below, where the kink does not exist.
        cases = (
            ('upper', 0.90, False), ('upper', 0.99, False),
            ('upper', 1.01, True), ('upper', 1.10, True),
            ('lower', 0.90, True), ('lower', 0.99, True),
            ('lower', 1.01, False), ('lower', 1.10, False),
            ('all', 0.90, True), ('all', 1.10, True),
        )
        for region, bond, active in cases:
            energy, gradient, curvature = assembled(region, bond)
            with self.subTest(region=region, bond=bond):
                if active:
                    self.assertGreater(energy, 0.0)
                    self.assertGreater(gradient, 0.0)
                    self.assertGreater(curvature, 0.0)
                else:
                    self.assertEqual((energy, gradient, curvature),
                                     (0.0, 0.0, 0.0))
        # At the target with a two-sided region the force is zero and the
        # curvature is not: that is the harmonic rest point, and it is the case
        # where the `e Q` term drops out while `k J J^T` stays.
        energy, gradient, curvature = assembled('all', 1.00)
        self.assertEqual((energy, gradient), (0.0, 0.0))
        self.assertGreater(curvature, 0.0)

    def test_an_aligned_rmsd_history_is_refused_by_name(self):
        """Its rotation depends on the geometry, so the chain rule differs."""
        from pyrai3md.workflow.constraints import metadynamics
        plan = schema.build_plan(
            {'metadynamics': [{'id': 'md', 'kind': 'rmsd', 'atoms': [1, 2, 3],
                               'height': 2.0e-4, 'width': [0.3], 'pace': 1,
                               'reference': [0.0, 0.0, 0.0,
                                             1.6, 0.3, 0.1,
                                             0.4, 1.7, -0.2]}]},
            atoms=list(ATOMS), natom=len(ATOMS), masses=list(MASSES))
        manager = ctl.SystemConstraintController(plan)
        state = manager.bind_reference(BASE_GEOMETRY, masses=MASSES)
        with self.assertRaises(metadynamics.MetadynamicsError) as caught:
            manager.hessian_block(BASE_GEOMETRY, state, None, masses=MASSES)
        self.assertIn('aligned RMSD', str(caught.exception))

    def test_a_fix_or_constraint_contributes_no_curvature(self):
        block = {'fixes': [{'id': 'held', 'atoms': [1], 'direction': 'xyz'}],
                 'constraints': [{'id': 'b', 'kind': 'bond',
                                  'bonds': [[1, 2]]}]}
        plan = schema.build_plan(block, atoms=list(ATOMS), natom=len(ATOMS),
                                 masses=list(MASSES))
        manager = ctl.SystemConstraintController(plan)
        state = manager.bind_reference(BASE_GEOMETRY, masses=MASSES)
        out = manager.hessian_block(BASE_GEOMETRY, state)
        self.assertEqual(out['ids'], [])
        np.testing.assert_array_equal(
            out['hessian'], np.zeros((BASE_GEOMETRY.size, BASE_GEOMETRY.size)))



if __name__ == '__main__':
    unittest.main()
