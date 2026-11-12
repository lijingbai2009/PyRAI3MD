"""
======================================================================
Tests for the `velocity_constraint: none | rattle` switch.

Position-only SHAKE was measured, not assumed, to hold the bond length to
1e-12 relative while `J v` sits at a finite value and the total energy of a
driven run drifts without converging as the step is halved (design document
section 1.3, `tests/test_shake_dynamics_cost.py`). The remedy the user ruled on
is a keyword on the constraint record rather than a replacement of the position
solve, so these tests pin three things at once:

- the projection works when it is asked for: `|J v|` falls to the declared
  tolerance, the momentum of the pair is untouched, and the tangential relative
  velocity survives because the impulse runs along the bond;
- the default is the old behaviour, literally: a record that does not name the
  keyword returns the *same array object* the caller handed in, and the position
  path produces byte-identical coordinates whether the switch is on or off;
- the switch is part of the plan, not a global: two records in one plan can
  disagree, a record that opted out keeps its non-tangent velocity, and the plan
  fingerprint changes with it, which is what makes a restart that flips the
  keyword a refusal instead of a silent change of physics.

The solver is tested directly as well as through the controller, because the
controller's job is the routing and the solver's job is the arithmetic.

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
from pyrai3md.workflow.constraints import projection
from pyrai3md.workflow.constraints import shake
from pyrai3md.utilities.units import AMU_TO_AU

COORDS = np.array([[0.00, 0.00, 0.00],
                   [1.50, 0.20, 0.00],
                   [0.40, 1.90, -0.20],
                   [2.30, 1.70, 1.10]])
MASSES = np.array([16.0, 1.0, 1.0, 12.0]) * AMU_TO_AU
VELOCITIES = np.array([[0.30, -0.10, 0.05],
                       [0.10, 0.40, -0.20],
                       [-0.20, 0.05, 0.10],
                       [0.15, -0.25, 0.30]])
PAIR = np.array([[0, 1]])
CHAIN = np.array([[0, 1], [1, 2]])

TOLERANCE = 1.0e-10
# A rate is a speed, so the declared relative tolerance is against the largest
# atomic speed the array carries. Written out here so the assertions say which
# number they mean.
SCALE = shake.velocity_scale(VELOCITIES)


def _plan(mode=None):
    """One bond constraint, with the switch set or left at its default."""
    record = {'id': 'x', 'bonds': [[1, 2]], 'tolerance': TOLERANCE,
              'max_iterations': 200}
    if mode is not None:
        record['velocity_constraint'] = mode
    return schema.build_plan({'constraints': [record]},
                             atoms=['O', 'H', 'H', 'C'], natom=4,
                             masses=MASSES / AMU_TO_AU)


def _context(step=1):
    return records.EvaluationContext(purpose='md', step=step, dt=20.67,
                                     evaluation_id='e%d' % step)


class TestSolverDirect(unittest.TestCase):
    """The arithmetic, with no plan and no controller in the way."""

    def test_the_radial_velocity_is_removed_to_the_declared_tolerance(self):
        out = shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES,
                                     tolerance=TOLERANCE)
        self.assertTrue(out['converged'])
        self.assertLess(out['max_residual'], TOLERANCE * SCALE)
        left = float(np.abs(shake.relative_speeds(
            COORDS, PAIR, out['velocities'])).max())
        self.assertLess(left, TOLERANCE * SCALE)

    def test_the_projection_runs_along_the_bond_so_the_momentum_is_untouched(
            self):
        out = shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES,
                                     tolerance=TOLERANCE)
        change = (out['velocities'] - VELOCITIES) * MASSES[:, None]
        scale = float(np.abs((VELOCITIES * MASSES[:, None]).sum(axis=0)).max())
        self.assertLess(float(np.abs(change.sum(axis=0)).max()) / scale,
                        1.0e-14)

    def test_the_tangential_relative_velocity_survives(self):
        out = shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES,
                                     tolerance=TOLERANCE)
        unit = (COORDS[0] - COORDS[1])
        unit = unit / np.linalg.norm(unit)
        before = VELOCITIES[0] - VELOCITIES[1]
        after = out['velocities'][0] - out['velocities'][1]
        projected_before = before - np.dot(before, unit) * unit
        projected_after = after - np.dot(after, unit) * unit
        np.testing.assert_allclose(projected_after, projected_before,
                                   atol=1.0e-12)

    def test_the_inverse_mass_sharing_reaches_the_heavier_atom_less(self):
        out = shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES,
                                     tolerance=TOLERANCE)
        shift = np.linalg.norm(out['velocities'] - VELOCITIES, axis=1)
        self.assertLess(shift[0], shift[1])

    def test_a_chain_takes_more_than_one_pass_and_still_converges(self):
        out = shake.solve_velocities(COORDS, CHAIN, VELOCITIES, MASSES,
                                     tolerance=1.0e-9)
        self.assertTrue(out['converged'])
        self.assertGreater(out['iterations'], 1)
        threshold = 1.0e-9 * SCALE
        self.assertLess(float(np.abs(shake.relative_speeds(
            COORDS, CHAIN, out['velocities'])).max()), threshold)

    def test_a_single_pass_on_a_chain_reports_the_failure_it_reached(self):
        with self.assertRaises(shake.ShakeNotConverged) as caught:
            shake.solve_velocities(COORDS, CHAIN, VELOCITIES, MASSES,
                                   tolerance=1.0e-14, max_iterations=1)
        failure = caught.exception
        self.assertEqual(failure.iterations, 1)
        self.assertGreater(failure.residual, 0.0)
        self.assertIn(tuple(int(atom) for atom in failure.constraint),
                      [tuple(row) for row in CHAIN])

    def test_a_collapsed_bond_is_refused_before_any_projection(self):
        broken = COORDS.copy()
        broken[1] = broken[0]
        with self.assertRaises(shake.ShakeError):
            shake.solve_velocities(broken, PAIR, VELOCITIES, MASSES)

    def test_no_pairs_returns_the_same_numbers(self):
        out = shake.solve_velocities(COORDS, np.zeros((0, 2), dtype=np.int64),
                                     VELOCITIES, MASSES)
        np.testing.assert_allclose(out['velocities'], VELOCITIES, atol=0.0)
        self.assertTrue(out['converged'])
        self.assertEqual(out['iterations'], 0)

    def test_frozen_components_leave_the_solve_at_zero(self):
        mask = np.zeros_like(COORDS, dtype=bool)
        mask[3] = True
        out = shake.solve_velocities(COORDS, CHAIN, VELOCITIES, MASSES,
                                     tolerance=TOLERANCE, fix_mask=mask)
        np.testing.assert_allclose(out['velocities'][3], np.zeros(3), atol=0.0)
        self.assertLess(float(np.abs(shake.relative_speeds(
            COORDS, CHAIN, out['velocities'])).max()), TOLERANCE * SCALE)

    def test_the_shape_and_mass_checks_run_before_the_iteration(self):
        with self.assertRaises(shake.ShakeError):
            shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES[:-1])
        with self.assertRaises(shake.ShakeError):
            shake.solve_velocities(COORDS, PAIR, VELOCITIES[:-1], MASSES)
        with self.assertRaises(shake.ShakeError):
            shake.solve_velocities(COORDS, PAIR, VELOCITIES,
                                   np.array([16.0, 0.0, 1.0, 12.0]))
        with self.assertRaises(shake.ShakeError):
            shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES,
                                   tolerance=1.0e-18)
        with self.assertRaises(shake.ShakeError):
            shake.solve_velocities(COORDS, PAIR, VELOCITIES, MASSES,
                                   max_iterations=0)

    def test_relative_speeds_read_the_jacobian_of_the_given_geometry(self):
        """Central check of the condition the projection drives to zero."""
        unit = (COORDS[0] - COORDS[1]) / np.linalg.norm(COORDS[0] - COORDS[1])
        expected = float(np.dot(unit, VELOCITIES[0] - VELOCITIES[1]))
        self.assertAlmostEqual(float(shake.relative_speeds(COORDS, PAIR,
                                                           VELOCITIES)[0]),
                               expected, places=14)

    def test_the_scale_is_the_largest_speed_and_never_zero(self):
        self.assertAlmostEqual(SCALE, float(np.abs(VELOCITIES).max()),
                               places=14)
        self.assertEqual(shake.velocity_scale(np.zeros((2, 3))), 1.0e-12)


class TestControllerRouting(unittest.TestCase):
    """What the switch does to the call a workflow makes."""

    def test_the_default_returns_the_callers_own_array(self):
        """DC-1 in its strictest form: no copy, no branch, no change.

        An identical-looking array would still let a rewrite of the default path
        sneak a projection in; object identity is the assertion that says the
        code did not run.
        """
        controller = ctl.SystemConstraintController(_plan())
        velocities = np.array(VELOCITIES, dtype=np.float64, copy=True)
        state = controller.bind_reference(COORDS, masses=MASSES)
        got = controller.project_velocities(velocities, state, _context(),
                                            masses=MASSES, coords=COORDS)
        self.assertIs(got, velocities)
        self.assertGreater(float(np.abs(shake.relative_speeds(
            COORDS, PAIR, got)).max()), 1.0e-3)

    def test_rattle_projects_without_touching_the_position_path(self):
        plain = ctl.SystemConstraintController(_plan())
        tangent = ctl.SystemConstraintController(_plan('rattle'))
        context = _context()
        drifted = COORDS + np.array([[0.05, 0.01, 0.0],
                                     [-0.04, 0.02, 0.01],
                                     [0.0, 0.0, 0.02],
                                     [0.01, -0.01, 0.0]])
        positions = {}
        residual = {}
        for name, controller in (('none', plain), ('rattle', tangent)):
            state = controller.bind_reference(COORDS, masses=MASSES)
            projected, info = controller.project_positions(drifted, state,
                                                           context,
                                                           masses=MASSES)
            velocities = controller.project_velocities(
                VELOCITIES, state, context, masses=MASSES, coords=projected)
            positions[name] = projected
            residual[name] = float(np.abs(shake.relative_speeds(
                projected, PAIR, velocities)).max())
            self.assertTrue(info['converged'])
        self.assertLess(residual['rattle'],
                        TOLERANCE * shake.velocity_scale(VELOCITIES))
        self.assertGreater(residual['none'], 1.0e-2,
                           'the branch that did not ask for the projection '
                           'must not receive it')
        np.testing.assert_allclose(positions['none'], positions['rattle'],
                                   atol=0.0)

    def test_a_rattle_projection_refuses_to_guess_the_geometry(self):
        controller = ctl.SystemConstraintController(_plan('rattle'))
        state = controller.bind_reference(COORDS, masses=MASSES)
        with self.assertRaises(ctl.ControllerError):
            controller.project_velocities(VELOCITIES, state, _context(),
                                          masses=MASSES)
        with self.assertRaises(ctl.ControllerError):
            controller.project_velocities(VELOCITIES, state, _context(),
                                          coords=COORDS)

    def test_a_frozen_atom_is_still_zeroed_when_the_switch_is_on(self):
        mapping = {'constraints': [{'id': 'x', 'bonds': [[1, 2]],
                                    'velocity_constraint': 'rattle',
                                    'tolerance': TOLERANCE}],
                   'fixes': [{'id': 'f', 'atoms': [4], 'direction': 'xyz'}]}
        plan = schema.build_plan(mapping, atoms=['O', 'H', 'H', 'C'], natom=4,
                                 masses=MASSES / AMU_TO_AU)
        controller = ctl.SystemConstraintController(plan)
        state = controller.bind_reference(COORDS, masses=MASSES)
        got = controller.project_velocities(VELOCITIES, state, _context(),
                                            masses=MASSES, coords=COORDS)
        np.testing.assert_allclose(got[3], np.zeros(3), atol=0.0)
        self.assertLess(float(np.abs(shake.relative_speeds(
            COORDS, PAIR, got)).max()),
            TOLERANCE * shake.velocity_scale(VELOCITIES))

    def test_only_the_records_that_opted_in_become_tangent(self):
        """The switch belongs to a record, not to the run."""
        mapping = {'constraints': [
            {'id': 'x', 'bonds': [[1, 2]], 'velocity_constraint': 'rattle',
             'tolerance': TOLERANCE},
            {'id': 'y', 'bonds': [[3, 4]], 'tolerance': TOLERANCE}]}
        plan = schema.build_plan(mapping, atoms=['O', 'H', 'H', 'C'], natom=4,
                                 masses=MASSES / AMU_TO_AU)
        controller = ctl.SystemConstraintController(plan)
        state = controller.bind_reference(COORDS, masses=MASSES)
        pairs = np.array([[0, 1], [2, 3]])
        got = controller.project_velocities(VELOCITIES, state, _context(),
                                            masses=MASSES, coords=COORDS)
        residual = shake.relative_speeds(COORDS, pairs, got)
        threshold = TOLERANCE * shake.velocity_scale(VELOCITIES)
        self.assertLess(abs(float(residual[0])), threshold)
        self.assertGreater(abs(float(residual[1])), 1.0e-3)

    def test_an_impossible_demand_fails_rather_than_softening(self):
        """One pass cannot make a two-bond chain tangent; that is a failure.

        The solver reporting a smaller velocity instead of raising would leave
        the run believing its bonds were velocity-constrained.
        """
        mapping = {'constraints': [{'id': 'x', 'bonds': [[1, 2], [2, 3]],
                                    'velocity_constraint': 'rattle',
                                    'tolerance': 1.0e-13,
                                    'max_iterations': 1}]}
        plan = schema.build_plan(mapping, atoms=['O', 'H', 'H', 'C'], natom=4,
                                 masses=MASSES / AMU_TO_AU)
        controller = ctl.SystemConstraintController(plan)
        state = controller.bind_reference(COORDS, masses=MASSES)
        with self.assertRaises(shake.ShakeNotConverged) as caught:
            controller.project_velocities(VELOCITIES, state, _context(),
                                          masses=MASSES, coords=COORDS)
        self.assertEqual(caught.exception.iterations, 1)


class TestPlanSurface(unittest.TestCase):
    """The keyword reaches the plan, the dict, and the fingerprint."""

    def test_an_undeclared_record_carries_none(self):
        plan = _plan()
        self.assertEqual(plan.actions[0].diagnostics['velocity_constraint'],
                         'none')

    def test_the_value_travels_through_the_dict_round_trip(self):
        spec = _plan('rattle').actions[0]
        rebuilt = records.ConstraintSpec.from_dict(spec.to_dict())
        self.assertEqual(rebuilt.diagnostics['velocity_constraint'], 'rattle')
        self.assertEqual(rebuilt.to_dict(), spec.to_dict())

    def test_a_flipped_switch_is_a_different_plan(self):
        """A restart with the other half of the constraint is a refusal.

        The merge round compares this fingerprint with the one stored in the
        checkpoint; the assertion here is that the comparison can see the switch
        at all.
        """
        self.assertNotEqual(_plan('none').fingerprint,
                            _plan('rattle').fingerprint)

    def test_the_same_switch_declared_two_ways_is_the_same_plan(self):
        """Text from the block and a mapping are one declaration, not two."""
        from_text = _plan('rattle')
        from_dict = schema.build_plan(
            {'constraints': [{'id': 'x', 'bonds': [[1, 2]],
                              'velocity_constraint': 'rattle',
                              'tolerance': TOLERANCE,
                              'max_iterations': 200}]},
            atoms=['O', 'H', 'H', 'C'], natom=4, masses=MASSES / AMU_TO_AU)
        self.assertEqual(from_text.fingerprint, from_dict.fingerprint)

    def test_an_unknown_word_is_refused_not_ignored(self):
        with self.assertRaises(schema.ConstraintInputError):
            _plan('shake')

    def test_the_projection_settings_are_the_ones_the_solver_reads(self):
        """The controller takes tolerance and iteration cap from the record."""
        controller = ctl.SystemConstraintController(_plan('rattle'))
        state = controller.bind_reference(COORDS, masses=MASSES)
        pairs, tolerance, iterations = controller._tangent_requests(state)
        np.testing.assert_allclose(pairs, PAIR, atol=0.0)
        self.assertAlmostEqual(tolerance, TOLERANCE, places=18)
        self.assertEqual(iterations, 200)

    def test_a_plan_with_no_opted_in_record_asks_for_nothing(self):
        controller = ctl.SystemConstraintController(_plan())
        self.assertIsNone(controller._tangent_requests(
            controller.bind_reference(COORDS, masses=MASSES)))


class TestFrozenAndTangentAgree(unittest.TestCase):
    """The two halves of the constraint must not disagree about one atom."""

    def test_a_bond_between_two_frozen_atoms_needs_no_projection(self):
        mask = np.zeros_like(COORDS, dtype=bool)
        mask[0] = True
        mask[1] = True
        zeroed = projection.zero_fixed_velocities(VELOCITIES, mask)
        out = shake.solve_velocities(COORDS, PAIR, zeroed, MASSES,
                                     tolerance=TOLERANCE, fix_mask=mask)
        np.testing.assert_allclose(out['velocities'], zeroed, atol=0.0)


if __name__ == '__main__':
    unittest.main()
