"""
======================================================================
Tests for the linear schedule.

A ramp must be a pure function of the accepted step: a retried evaluation, a
log replay or a second call at the same step must not advance it. The
endpoint rules matter for the same reason: extrapolating past the window would
let a wall radius keep shrinking after the run was supposed to have stopped
compressing.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.data import constraint as records
from pyrai3md.workflow.constraints import schedule


def _schedule(**kwargs):
    options = {'id': 's', 'start_step': 10, 'end_step': 20,
               'scale': [0.0, 1.0]}
    options.update(kwargs)
    return records.ScheduleRecord(**options)


def _context(step):
    return records.EvaluationContext(purpose='md', step=step)


def _spec(name='b', schedule_record=None, **parameters):
    values = {'force_constant': 0.02}
    values.update(parameters)
    return records.ConstraintSpec(name, 'bias', 'harmonic', cv='r',
                                  target=1.0, parameters=values,
                                  schedule=schedule_record)


class TestFraction(unittest.TestCase):

    def test_before_the_window_is_zero(self):
        self.assertEqual(schedule.ramp_fraction(_schedule(), _context(0)),
                         0.0)
        self.assertEqual(schedule.ramp_fraction(_schedule(), _context(10)),
                         0.0)

    def test_after_the_window_holds_at_one(self):
        self.assertEqual(schedule.ramp_fraction(_schedule(), _context(20)),
                         1.0)
        self.assertEqual(schedule.ramp_fraction(_schedule(), _context(999)),
                         1.0)

    def test_inside_the_window_is_linear(self):
        self.assertAlmostEqual(
            schedule.ramp_fraction(_schedule(), _context(15)), 0.5, places=12)
        self.assertAlmostEqual(
            schedule.ramp_fraction(_schedule(), _context(12)), 0.2, places=12)

    def test_the_same_step_twice_gives_the_same_fraction(self):
        record = _schedule()
        first = schedule.ramp_fraction(record, _context(14))
        second = schedule.ramp_fraction(record, _context(14))
        self.assertEqual(first, second)


class TestParameterResolution(unittest.TestCase):

    def setUp(self):
        self.record = _schedule()

    def test_scale_multiplies_the_declared_value(self):
        spec = _spec(schedule_record=self.record)
        got = schedule.resolve_parameters(spec, _context(15),
                                          {'force_constant': 0.02})
        self.assertAlmostEqual(got['force_constant'], 0.01, places=12)

    def test_absolute_value_endpoints_win_over_the_declaration(self):
        record = _schedule(scale=None, value=[1.0, 3.0])
        spec = _spec(schedule_record=record)
        got = schedule.resolve_parameters(spec, _context(15),
                                          {'force_constant': 0.02})
        self.assertAlmostEqual(got['force_constant'], 2.0, places=12)

    def test_no_schedule_passes_the_declaration_through(self):
        spec = _spec()
        values = {'force_constant': 0.02}
        self.assertEqual(schedule.resolve_parameters(spec, _context(7),
                                                     values), values)

    def test_the_declared_parameters_are_not_mutated(self):
        spec = _spec(schedule_record=self.record)
        values = {'force_constant': 0.02}
        schedule.resolve_parameters(spec, _context(15), values)
        self.assertAlmostEqual(values['force_constant'], 0.02, places=12)

    def test_an_unscheduled_action_ignores_the_step(self):
        spec = _spec()
        for step in (0, 5, 9999):
            got = schedule.resolve_parameters(spec, _context(step),
                                              {'force_constant': 0.02})
            self.assertAlmostEqual(got['force_constant'], 0.02, places=12)


class TestPositivity(unittest.TestCase):

    def test_a_ramp_through_zero_radius_is_refused(self):
        record = _schedule(parameter='radius')
        spec = records.ConstraintSpec('w', 'wall', 'sphere',
                                      parameters={'radius': 10.0},
                                      schedule=record)
        with self.assertRaises(schedule.ScheduleError):
            schedule.resolve_parameters(spec, _context(10),
                                        {'radius': 10.0})

    def test_a_ramp_that_stays_positive_is_allowed(self):
        record = _schedule(scale=[0.5, 1.0], parameter='radius')
        spec = records.ConstraintSpec('w', 'wall', 'sphere',
                                      parameters={'radius': 10.0},
                                      schedule=record)
        got = schedule.resolve_parameters(spec, _context(10),
                                          {'radius': 10.0})
        self.assertAlmostEqual(got['radius'], 5.0, places=12)

    def test_a_non_positive_endpoint_inside_the_window_is_refused(self):
        record = _schedule(scale=[2.0, -1.0])
        spec = _spec(schedule_record=record)
        with self.assertRaises(schedule.ScheduleError):
            schedule.resolve_parameters(spec, _context(20),
                                        {'force_constant': 0.02})


class TestWrongParameter(unittest.TestCase):

    def test_ramping_a_parameter_the_action_lacks_is_refused(self):
        record = _schedule(parameter='height')
        spec = _spec(schedule_record=record)
        with self.assertRaises(schedule.ScheduleError):
            schedule.resolve_parameters(spec, _context(15),
                                        {'force_constant': 0.02})


class TestRecordGuards(unittest.TestCase):

    def test_only_linear_is_implemented(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', kind='exponential', start_step=0,
                                   end_step=1, scale=[0, 1])

    def test_both_endpoint_spellings_are_refused(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=0, end_step=1,
                                   scale=[0, 1], value=[0, 1])

    def test_a_closed_window_is_refused(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=5, end_step=5, scale=[0, 1])

    def test_a_time_keyed_ramp_is_refused(self):
        with self.assertRaises(ValueError):
            records.ScheduleRecord('s', start_step=0, end_step=1, scale=[0, 1],
                                   time_dependent=True)


if __name__ == '__main__':
    unittest.main()
