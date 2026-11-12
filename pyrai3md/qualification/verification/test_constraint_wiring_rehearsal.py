"""
======================================================================
A rehearsal of the whole integration sequence, on a stub state.

The merge round writes three lines of framework glue: a finalizer call in
`workflow/calculator.py`, a projection call in
`workflow/integrator/operations.py`, and a print-out hand-off. Those lines are
short, and the risk in them is not their syntax but the *order* and the *shapes*
they pass along. This file runs that order once, end to end, against a fake
state, and asserts the numbers at every hand-off point. The merge round can then
copy the sequence instead of inventing it under pressure.

What is exercised, in call order:

    state_adapter.read_state      stub state  -> SystemView + EvaluationContext
    state_adapter.require_usable  view + plan -> refusal or the same view
    controller.bind_reference     reference geometry -> bound targets and anchors
    controller.project_positions  drifted geometry -> constrained geometry
    controller.evaluate_potentials pure correction at the projected geometry
    controller.project_velocities  frozen components zeroed
    controller.statistics          degrees of freedom on the constrained manifold
    controller.combine             base + correction, for three base shapes
    controller.commit              the step earns its hills
    ConstraintResult.to_dict       the cross-process payload
    pickle / json round trips      what a remote worker actually sends
    ConstraintResult.from_dict     the parent's rebuild
    flat_contributions + the framework reader
                                 the line a user reads in `<title>.log`

Three things this is deliberately strict about, because each is a way the merge
round could produce a run that looks fine and reports the wrong energy: a base
that carries no gradient must not gain one, totals must be rebuilt rather than
accumulated, and an empty plan must not leave a single key behind.

The reader at the end is a framework module, imported by a test on purpose:
`test_import_discipline` allows a test to reach a framework formatter to prove
the breakdown line is printed by the reader that already exists, while the
shipped math may not.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import pickle
import unittest

import numpy as np

from pyrai3md.data import constraint as records
from pyrai3md import schema
from pyrai3md.qualification.verification import test_constraint_state_adapter as stubs
from pyrai3md.workflow.constraints import controller as ctl
from pyrai3md.workflow.constraints import state_adapter
from pyrai3md.utilities import print_results
from pyrai3md.utilities.units import AMU_TO_AU, ANGSTROM_TO_BOHR

ATOMS = ['O', 'H', 'H', 'C', 'H']
MASSES_AMU = np.array([16.0, 1.0, 1.0, 12.0, 1.0])
MASSES = MASSES_AMU * AMU_TO_AU

# The geometry the targets and anchors are bound at. Angstrom, as the state
# carries it; the adapter converts to Bohr at the boundary.
REFERENCE = np.array([[0.00, 0.00, 0.00],
                      [1.60, 0.30, 0.10],
                      [0.40, 1.70, -0.20],
                      [2.10, 1.80, 1.30],
                      [-1.20, 0.90, 1.10]])
# One integrator step later: the constrained bond has stretched past its target
# and every atom has moved, so the projection has real work to do.
DRIFTED = np.array([[0.03, -0.02, 0.01],
                    [1.72, 0.33, 0.06],
                    [0.44, 1.66, -0.24],
                    [2.06, 1.84, 1.26],
                    [-1.16, 0.94, 1.06]])
VELOCITIES = np.arange(15, dtype=np.float64).reshape(5, 3) * 1.0e-3

# Five kinds at once, because the sequence is the thing being rehearsed: a CV,
# a group, a harmonic bias, a wall that acts through the group and also covers
# the frozen atom, a freeze, a bond constraint bound at the reference, and a
# metadynamics action whose hill arrives on the second accepted step.
FULL = {
    'groups': [{'name': 'cage', 'atoms': [2, 3, 5]}],
    'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [2, 3]}],
    'biases': [{'id': 'b', 'cv': 'r', 'target': 1.6, 'force_constant': 0.02,
                'region': 'all'}],
    'walls': [{'id': 'w', 'geometry': 'sphere', 'center': [0.0, 0.0, 0.0],
               'radius': 1.0, 'group': 'cage', 'power': 2,
               'force_constant': 0.005}],
    'fixes': [{'id': 'f', 'atoms': [5], 'direction': 'xyz'}],
    'constraints': [{'id': 'x', 'bonds': [[1, 2]],
                    'tolerance': 1.0e-10, 'max_iterations': 500}],
    'metadynamics': [{'id': 'm', 'cvs': ['r'], 'height': 1.0e-4,
                      'width': [0.1], 'pace': 2}],
}

BASE = {'energy': -76.4321,
        'gradient': np.arange(15, dtype=np.float64).reshape(5, 3) * 1.0e-3}
MULTI_STATE = {'energy': np.array([-76.4321, -76.1007]),
               'gradient': np.stack([BASE['gradient'],
                                     BASE['gradient'] * 0.5])}
NO_GRADIENT = {'energy': -76.4321}

FIXED_ATOM = 4
CONSTRAINED_PAIR = (0, 1)
SHAKE_RELATIVE = 1.0e-10


def _state(coordinates, velocities=None, multi_center=False, step=0):
    state = stubs._StubState(system=stubs._System(natom=len(ATOMS)),
                             coordinates=coordinates, velocities=velocities,
                             masses=MASSES, atoms=list(ATOMS),
                             multi_center=multi_center)
    # The clock belongs to the state, not to the caller: `read_state` takes the
    # step from `runtime.current_step`, so a rehearsal of a given step sets it
    # where the framework keeps it.
    state.runtime = stubs._Runtime(step=step, time=step * 20.67)
    return state


def _read(state, step):
    return state_adapter.read_state(state, purpose='md',
                                    evaluation_id='eval-%d' % step,
                                    trajectory_id='rehearsal')


def _plan(mapping=None):
    return schema.build_plan(FULL if mapping is None else mapping,
                             atoms=ATOMS, natom=len(ATOMS),
                             masses=MASSES_AMU)


def _distance(coords, pair):
    return float(np.linalg.norm(coords[pair[0]] - coords[pair[1]]))


class Rehearsal(object):
    """One run through the merge round's call order, keeping every stage.

    The stages are attributes rather than return values because the tests assert
    on the hand-offs between them, and a test that recomputed a stage would be
    checking its own arithmetic instead of the sequence.
    """

    def __init__(self, mapping=None, base=None, step=2):
        self.plan = _plan(mapping)
        self.controller = ctl.SystemConstraintController(self.plan)

        reference_view, reference_context = _read(_state(REFERENCE, step=0), 0)
        self.reference_view = reference_view
        self.bound = self.controller.bind_reference(
            reference_view.coordinates, masses=reference_view.masses,
            cell=reference_view.cell, context=reference_context)

        view, context = _read(_state(DRIFTED, velocities=VELOCITIES,
                                     step=step), step)
        self.view = state_adapter.require_usable(view, self.plan)
        self.context = context
        self.projected, self.projection = self.controller.project_positions(
            view.coordinates, self.bound, context, masses=view.masses)
        self.potential = self.controller.evaluate_potentials(
            self.projected, self.bound, context, masses=view.masses)
        self.velocities = self.controller.project_velocities(
            view.velocities, self.bound, context)
        self.statistics = self.controller.statistics(
            self.projected, self.velocities, view.masses, self.bound, context)
        self.totals = self.combine(base if base is not None else BASE)

    def combine(self, base):
        return self.controller.combine(base, self.potential, self.projection,
                                       self.context)

    def commit(self, state=None):
        return self.controller.commit(self.projected,
                                      self.bound if state is None else state,
                                      self.context, masses=self.view.masses)

    def component_energy(self, action_id):
        return float(self.potential.components[action_id]['energy'])


class TestStageHandoffs(unittest.TestCase):
    """Each stage's output must be usable by the next one, unchanged in kind."""

    @classmethod
    def setUpClass(cls):
        cls.trial = Rehearsal()

    def test_the_view_arrives_in_bohr_and_the_cell_stays_absent(self):
        np.testing.assert_allclose(self.trial.reference_view.coordinates,
                                   REFERENCE * ANGSTROM_TO_BOHR, atol=1.0e-14)
        self.assertIsNone(self.trial.view.cell)
        self.assertFalse(self.trial.view.periodic)

    def test_the_context_carries_the_step_the_projection_was_asked_for(self):
        self.assertEqual(self.trial.context.step, 2)
        self.assertEqual(self.trial.context.evaluation_id, 'eval-2')
        self.assertEqual(self.trial.context.trajectory_id, 'rehearsal')

    def test_the_projection_reports_that_it_actually_worked(self):
        self.assertTrue(self.trial.projection['projected'])
        self.assertTrue(self.trial.projection['converged'])
        self.assertEqual(self.trial.projection['restored_components'], 3)

    def test_the_solver_residual_sits_inside_the_declared_tolerance(self):
        self.assertLess(self.trial.projection['max_residual'], 1.0e-10)

    def test_the_constrained_bond_holds_the_length_it_was_bound_to(self):
        target = float(self.trial.bound.targets['x'][0])
        got = _distance(self.trial.projected, CONSTRAINED_PAIR)
        self.assertAlmostEqual(got / target, 1.0, delta=SHAKE_RELATIVE)

    def test_the_bound_target_is_the_reference_length_not_the_drifted_one(self):
        bound = _distance(self.trial.reference_view.coordinates,
                          CONSTRAINED_PAIR)
        drifted = _distance(self.trial.view.coordinates, CONSTRAINED_PAIR)
        self.assertAlmostEqual(float(self.trial.bound.targets['x'][0]), bound,
                               places=12)
        self.assertGreater(drifted, bound + 1.0e-3,
                           'the rehearsal geometry must actually stretch the '
                           'constrained bond or the solve proves nothing')

    def test_the_frozen_atom_returns_to_its_anchor_exactly(self):
        np.testing.assert_allclose(self.trial.projected[FIXED_ATOM],
                                   self.trial.reference_view.coordinates[
                                       FIXED_ATOM], atol=0.0)

    def test_the_atoms_the_solver_moved_are_not_the_input(self):
        for atom in CONSTRAINED_PAIR:
            self.assertGreater(np.abs(self.trial.projected[atom]
                                      - self.trial.view.coordinates[atom]).max(),
                               0.0)

    def test_the_potential_is_evaluated_at_the_projected_geometry(self):
        np.testing.assert_allclose(self.trial.potential.coordinates,
                                   self.trial.projected, atol=0.0)

    def test_the_correction_is_the_sum_of_the_components(self):
        total = sum(float(item['energy'])
                    for item in self.trial.potential.components.values())
        self.assertAlmostEqual(self.trial.potential.energy_correction, total,
                               places=14)

    def test_the_gradient_correction_is_the_sum_of_the_component_gradients(
            self):
        stacked = np.zeros((len(ATOMS), 3))
        for item in self.trial.potential.components.values():
            stacked += item['gradient']
        np.testing.assert_allclose(self.trial.potential.gradient_correction,
                                   stacked, atol=1.0e-14)

    def test_every_declared_kind_contributed(self):
        self.assertEqual(sorted(self.trial.potential.components),
                         ['b', 'm', 'w'])
        self.assertEqual(self.trial.plan.ids('constraint'), ['x'])
        self.assertEqual(self.trial.plan.ids('fix'), ['f'])

    def test_the_velocity_projection_touches_only_frozen_components(self):
        np.testing.assert_allclose(self.trial.velocities[FIXED_ATOM],
                                   np.zeros(3), atol=0.0)
        others = np.delete(np.arange(len(ATOMS)), FIXED_ATOM)
        np.testing.assert_allclose(self.trial.velocities[others],
                                   VELOCITIES[others], atol=0.0)

    def test_the_dof_count_loses_the_frozen_components_and_one_bond(self):
        """The freeze leaves the mobile set; only the bond row is a rank."""
        self.assertEqual(self.trial.statistics.mobile_components,
                         3 * len(ATOMS) - 3)
        self.assertEqual(self.trial.statistics.rank, 1)
        self.assertEqual(self.trial.statistics.free_dof,
                         3 * len(ATOMS) - 3 - 1)

    def test_the_commit_returns_a_state_the_original_never_became(self):
        before = len(self.trial.bound.hills)
        grown = self.trial.commit()
        self.assertEqual(len(grown.hills), before + 1)
        self.assertEqual(len(self.trial.bound.hills), before)
        self.assertEqual(grown.last_committed_step, 2)


class TestBaseShapes(unittest.TestCase):
    """The three shapes a framework result may arrive in."""

    @classmethod
    def setUpClass(cls):
        cls.trial = Rehearsal()

    def test_a_scalar_energy_and_a_per_atom_gradient(self):
        totals = self.trial.combine(BASE)
        self.assertAlmostEqual(totals.energy_total,
                               BASE['energy'] + self.trial.potential
                               .energy_correction, places=14)
        np.testing.assert_allclose(totals.gradient_total,
                                   BASE['gradient']
                                   + self.trial.potential.gradient_correction,
                                   atol=1.0e-14)
        self.assertEqual(totals.provenance['nstates'], 1)

    def test_a_state_resolved_gradient_gains_the_same_correction_in_every_row(
            self):
        totals = self.trial.combine(MULTI_STATE)
        correction = self.trial.potential.energy_correction
        np.testing.assert_allclose(totals.energy_total,
                                   MULTI_STATE['energy'] + correction,
                                   atol=1.0e-14)
        for row, base_row in enumerate(MULTI_STATE['gradient']):
            np.testing.assert_allclose(
                totals.gradient_total[row],
                base_row + self.trial.potential.gradient_correction,
                atol=1.0e-14, err_msg='state %d' % row)
        self.assertEqual(totals.provenance['nstates'], 2)

    def test_a_base_without_a_gradient_does_not_invent_one(self):
        """The expensive direction: a bias force with nothing behind it."""
        totals = self.trial.combine(NO_GRADIENT)
        self.assertIsNone(totals.gradient_total)
        self.assertIsNone(totals.gradient_projected)
        self.assertIsNone(totals.gradient_for_integrator)
        self.assertIsNotNone(totals.gradient_correction,
                             'the correction itself must still be reported')
        self.assertAlmostEqual(totals.energy_total,
                               NO_GRADIENT['energy']
                               + self.trial.potential.energy_correction,
                               places=14)

    def test_the_projected_gradient_zeros_the_frozen_rows_only(self):
        totals = self.trial.combine(BASE)
        np.testing.assert_allclose(totals.gradient_projected[FIXED_ATOM],
                                   np.zeros(3), atol=0.0)
        others = np.delete(np.arange(len(ATOMS)), FIXED_ATOM)
        np.testing.assert_allclose(totals.gradient_projected[others],
                                   totals.gradient_total[others], atol=0.0)

    def test_a_repeated_combine_rebuilds_rather_than_accumulates(self):
        first = self.trial.combine(BASE)
        second = self.trial.combine(BASE)
        self.assertAlmostEqual(first.energy_total, second.energy_total,
                               places=15)
        np.testing.assert_allclose(first.gradient_total, second.gradient_total,
                                   atol=0.0)
        self.assertAlmostEqual(second.energy_base, BASE['energy'], places=14)

    def test_the_evaluation_result_is_left_alone_by_combining(self):
        correction = self.trial.potential.energy_correction
        self.trial.combine(BASE)
        self.assertAlmostEqual(self.trial.potential.energy_correction, correction,
                               places=15)
        self.assertIsNone(self.trial.potential.energy_base)

    def test_a_base_computed_at_another_geometry_is_refused(self):
        """The one-call form checks the pairing; `combine` is the finalizer
        that guarantees it by construction and therefore does not re-check."""
        base = dict(BASE)
        base['geometry'] = self.trial.view.coordinates
        with self.assertRaises(ctl.ControllerError):
            self.trial.controller.apply_structure(
                self.trial.view.coordinates, state=self.trial.bound,
                context=self.trial.context, masses=self.trial.view.masses,
                base=base)

    def test_apply_structure_is_the_geometry_handling_one_call_form(self):
        result = self.trial.controller.apply_structure(
            self.trial.view.coordinates, state=self.trial.bound,
            context=self.trial.context, masses=self.trial.view.masses,
            base=dict(BASE))
        np.testing.assert_allclose(result.coordinates, self.trial.projected,
                                   atol=0.0)
        self.assertAlmostEqual(result.energy_total,
                               self.trial.combine(BASE).energy_total,
                               places=14)
        np.testing.assert_allclose(result.delta_coordinates,
                                   self.trial.projected
                                   - self.trial.view.coordinates, atol=0.0)

    def test_a_gradient_for_the_wrong_atom_count_is_refused(self):
        base = {'energy': -1.0, 'gradient': np.zeros((3, 3))}
        with self.assertRaises(ctl.ControllerError):
            self.trial.combine(base)


class TestHistoryReachesTheNextStep(unittest.TestCase):
    """A deposited hill must change the next evaluation of the same run."""

    def test_the_second_step_carries_the_hill_the_first_step_earned(self):
        run = Rehearsal(step=2)
        self.assertAlmostEqual(run.component_energy('m'), 0.0, places=15)
        grown = run.commit()
        later = Rehearsal(step=3)
        later.bound = grown
        potential = later.controller.evaluate_potentials(
            later.projected, grown, later.context, masses=later.view.masses)
        self.assertGreater(float(potential.components['m']['energy']), 0.0)
        self.assertEqual(potential.provenance['history_version'], 1)

    def test_a_replayed_step_deposits_no_second_hill(self):
        run = Rehearsal(step=2)
        once = run.commit()
        again = run.commit(once)
        self.assertEqual(len(again.hills), len(once.hills))
        self.assertEqual(again.history_version, once.history_version)


class TestCrossProcessPayload(unittest.TestCase):
    """What a remote worker sends and the parent rebuilds.

    A dispatch payload goes through pickle to the worker and a dict, usually
    JSON, back. Both paths are checked, and the arrays have to be gone in the
    middle or the parent's writer meets a type it cannot format.
    """

    @classmethod
    def setUpClass(cls):
        cls.trial = Rehearsal()
        cls.totals = cls.trial.combine(BASE)

    @staticmethod
    def _plain(value, path='result'):
        """Reject anything that is not plain JSON data, naming where."""
        offenders = []
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    offenders.append('%s: key %r' % (path, key))
                offenders.extend(TestCrossProcessPayload._plain(
                    item, '%s[%r]' % (path, key)))
        elif isinstance(value, (list, tuple)):
            for position, item in enumerate(value):
                offenders.extend(TestCrossProcessPayload._plain(
                    item, '%s[%d]' % (path, position)))
        elif isinstance(value, np.ndarray):
            offenders.append('%s holds an ndarray' % path)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            offenders.append('%s holds %s' % (path, type(value).__name__))
        return offenders

    def test_the_dict_payload_holds_no_framework_types(self):
        self.assertEqual(self._plain(self.totals.to_dict()), [])

    def test_a_json_round_trip_rebuilds_the_same_numbers(self):
        payload = json.loads(json.dumps(self.totals.to_dict()))
        rebuilt = records.ConstraintResult.from_dict(payload)
        self.assertAlmostEqual(rebuilt.energy_total, self.totals.energy_total,
                               places=14)
        self.assertAlmostEqual(rebuilt.energy_base, self.totals.energy_base,
                               places=14)
        np.testing.assert_allclose(rebuilt.gradient_total,
                                   self.totals.gradient_total, atol=1.0e-14)
        np.testing.assert_allclose(rebuilt.coordinates, self.totals.coordinates,
                                   atol=0.0)

    def test_the_plan_fingerprint_survives_the_trip(self):
        payload = json.loads(json.dumps(self.totals.to_dict()))
        rebuilt = records.ConstraintResult.from_dict(payload)
        self.assertEqual(rebuilt.provenance['plan_fingerprint'],
                         self.trial.plan.fingerprint)

    def test_a_missing_gradient_survives_as_absence(self):
        totals = self.trial.combine(NO_GRADIENT)
        payload = json.loads(json.dumps(totals.to_dict()))
        self.assertIsNone(payload['gradient_total'])
        rebuilt = records.ConstraintResult.from_dict(payload)
        self.assertIsNone(rebuilt.gradient_total)
        self.assertIsNone(rebuilt.gradient_for_integrator)

    def test_a_pickle_round_trip_rebuilds_the_object(self):
        copy = pickle.loads(pickle.dumps(self.totals))
        self.assertIsInstance(copy, records.ConstraintResult)
        self.assertAlmostEqual(copy.energy_correction,
                               self.totals.energy_correction, places=15)
        np.testing.assert_allclose(copy.gradient_correction,
                                   self.totals.gradient_correction, atol=0.0)
        self.assertEqual(sorted(copy.components), sorted(self.totals.components))

    def test_the_flat_breakdown_is_the_same_before_and_after(self):
        payload = json.loads(json.dumps(self.totals.to_dict()))
        rebuilt = records.ConstraintResult.from_dict(payload)
        self.assertEqual(rebuilt.flat_contributions(),
                         self.totals.flat_contributions())

    def test_the_flat_keys_name_the_kind_and_the_declared_id(self):
        self.assertEqual(sorted(self.totals.flat_contributions()),
                         ['constraint.bias.b', 'constraint.metadynamics.m',
                          'constraint.wall.w'])

    def test_the_sum_of_the_rows_is_the_total_without_double_counting(self):
        flat = self.totals.flat_contributions()
        self.assertAlmostEqual(sum(flat.values()),
                               self.totals.energy_correction, places=14)


class TestFrameworkReader(unittest.TestCase):
    """The last hand-off: the numbers reaching the reader that prints them.

    This is the only place the rehearsal leaves the lab, and it is the reason
    the rehearsal exists: the merge round is allowed to be a copy of this test
    because the call it copies has already been shown to render.
    """

    @classmethod
    def setUpClass(cls):
        cls.trial = Rehearsal()
        cls.totals = cls.trial.combine(BASE)
        cls.flat = cls.totals.flat_contributions()
        cls.block = print_results._format_energy(BASE['energy'], cls.flat)

    def test_the_contributions_section_is_rendered(self):
        self.assertIn('&energy contributions', self.block)

    def test_every_row_is_named_and_the_number_stays_separable(self):
        rows = {}
        for line in self.block.split('\n'):
            if not line.startswith('  constraint.'):
                continue
            tokens = line.split()
            self.assertEqual(len(tokens), 2,
                             'a name longer than the field must not swallow '
                             'its number: %r' % line)
            rows[tokens[0]] = float(tokens[1])
        self.assertEqual(sorted(rows), sorted(self.flat))
        for name, value in self.flat.items():
            self.assertAlmostEqual(rows[name], value, delta=1.0e-15)

    def test_the_electronic_section_still_reports_the_base_not_the_total(self):
        self.assertIn('%28.16f' % BASE['energy'], self.block)

    def test_no_contributions_render_no_section(self):
        block = print_results._format_energy(BASE['energy'], {})
        self.assertNotIn('&energy contributions', block)

    def test_the_guard_against_a_wrong_count_is_live(self):
        """Negative control: a dropped row would fail the test above."""
        broken = dict(self.flat)
        broken.pop('constraint.wall.w')
        block = print_results._format_energy(BASE['energy'], broken)
        rows = [line.split()[0] for line in block.split('\n')
                if line.startswith('  constraint.')]
        self.assertEqual(len(rows), len(self.flat) - 1)


class TestEmptyPlanShortCircuit(unittest.TestCase):
    """DC-1 and DC-3 across the whole sequence, not one method at a time."""

    @classmethod
    def setUpClass(cls):
        cls.plan = _plan({'units': {'length': 'angstrom', 'angle': 'degree',
                                    'energy': 'hartree'}})
        cls.controller = ctl.SystemConstraintController(cls.plan)
        view, context = _read(_state(DRIFTED, velocities=VELOCITIES), 5)
        cls.view = state_adapter.require_usable(view, cls.plan)
        cls.context = context
        cls.bound = cls.controller.bind_reference(view.coordinates,
                                                  masses=view.masses,
                                                  context=context)
        cls.projected, cls.projection = cls.controller.project_positions(
            view.coordinates, cls.bound, context, masses=view.masses)
        cls.potential = cls.controller.evaluate_potentials(
            cls.projected, cls.bound, context, masses=view.masses)
        cls.totals = cls.controller.combine(dict(BASE), cls.potential,
                                            cls.projection, context)

    def test_the_plan_is_the_empty_sentinel(self):
        self.assertTrue(self.plan.is_empty)
        self.assertTrue(self.controller.is_noop())

    def test_the_controller_factory_declines_to_exist(self):
        self.assertIsNone(ctl.controller_for(self.plan))

    def test_nothing_is_rearranged(self):
        np.testing.assert_allclose(self.projected, self.view.coordinates,
                                   atol=0.0)
        self.assertFalse(self.projection['projected'])
        self.assertEqual(self.projection, {'projected': False})

    def test_no_array_is_allocated(self):
        self.assertTrue(self.potential.is_empty)
        self.assertIsNone(self.potential.coordinates)
        self.assertIsNone(self.potential.gradient_correction)

    def test_the_numbers_pass_straight_through(self):
        self.assertAlmostEqual(self.totals.energy_total, BASE['energy'],
                               places=15)
        np.testing.assert_allclose(self.totals.gradient_total, BASE['gradient'],
                                   atol=0.0)
        self.assertEqual(self.totals.energy_base, BASE['energy'])

    def test_no_key_is_written_into_the_contributions(self):
        self.assertEqual(self.totals.flat_contributions(), {})
        block = print_results._format_energy(BASE['energy'],
                                             self.totals.flat_contributions())
        self.assertNotIn('constraint.', block)

    def test_the_payload_carries_nothing_a_worker_has_to_serialize(self):
        payload = self.totals.to_dict()
        self.assertEqual(payload['components'], {})
        self.assertIsNone(payload['gradient_correction'])
        self.assertTrue(payload['provenance']['plan_empty'])
        rebuilt = records.ConstraintResult.from_dict(
            json.loads(json.dumps(payload)))
        self.assertTrue(rebuilt.is_empty)

    def test_velocities_and_dof_are_the_plain_system(self):
        velocities = self.controller.project_velocities(
            self.view.velocities, self.bound, self.context)
        np.testing.assert_allclose(velocities, VELOCITIES, atol=0.0)
        statistics = self.controller.statistics(self.projected, velocities,
                                                self.view.masses, self.bound,
                                                self.context)
        self.assertEqual(statistics.free_dof, 3 * len(ATOMS))
        self.assertEqual(statistics.rank, 0)


if __name__ == '__main__':
    unittest.main()
