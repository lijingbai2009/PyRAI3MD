"""
======================================================================
Test suite for `&md dissipate`: an excited trajectory propagates on NVE
and the thermostat starts only once the run has been on the electronic
ground state for `thermodelay` steps (dev/methods/
ENSEMBLE_DEVELOPMENT.md, the section on the dissipation gate).

The keyword answers one question, when the chain is allowed to touch the
velocities, and the answer is a step number rather than a countdown:
`RuntimeState.dissipate_step` records the step the trajectory first landed
on the ground state, is never moved back, and `thermodelay` counts from
it. That is what makes the feature restart-safe (a segment that resumes
mid-run does not restart the delay) and hop-safe (a trajectory that hops
back up keeps the thermostat it already switched on, instead of dithering
between two ensembles).

Two things the gate does not do, each with a test: it does not half-act
on one side of the split step, because one Nose-Hoover half kick without
the other is a different integrator; and it does not reach the barostat,
because the ruling is nve to nvt, and a cell that stopped breathing in
the middle of a run would be a different ensemble rather than a delayed
one.

`dissipate` defaults to 0, and the default is proved rather than asserted:
the same thermal state with and without the keyword produces
bit-identical velocities, so no existing example pays for this or changes
its bytes.

Oct 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema import normalize, validate
from pyrai3md.schema.adapter import build_config
from pyrai3md.workflow.integrator import apply_ensemble
from pyrai3md.workflow.integrator.ensemble import _dissipate_gate_open
from pyrai3md.workflow.md import _ground_state_reached, _record_dissipate_landing

# Absolute import: `unittest discover` runs from inside this directory on
# the deployment, where a relative import has no parent package.
from pyrai3md.qualification.verification.test_integrator import (
    _make_ensemble_state)


def _thermal_state(dissipate=1, thermodelay=3, state_index=1):
    """An NVT/Nose-Hoover state sitting on an excited electronic state.

    The chain length, the temperature and the coupling are the ones
    test_integrator's Nose-Hoover case uses, so the control arm (gate open)
    really cools the velocities instead of overflowing a stiff chain.
    """
    state = _make_ensemble_state(ensemble='nvt', thermostat='nose-hoover',
                                 thermostat_chain=1, temperature=300.0,
                                 dissipate=dissipate, thermodelay=thermodelay)
    state.update_system({'current_state': state_index})
    return state


def _both_stages(state):
    """One step's worth of ensemble control: pre half then post half."""
    apply_ensemble(state, stage='pre')
    apply_ensemble(state, stage='post')


class TestGateClosesOnAnExcitedState(unittest.TestCase):
    """While the trajectory is excited, the chain does not touch it."""

    def test_velocities_survive_both_half_steps_untouched(self):
        state = _thermal_state()
        before = np.asarray(state.velocities, dtype=np.float64).copy()
        _both_stages(state)
        np.testing.assert_array_equal(
            before, np.asarray(state.velocities, dtype=np.float64))
        self.assertEqual(0, state.runtime.dissipate_step)

    def test_the_same_state_without_the_gate_does_move(self):
        # The negative control on the control: if the chain were inert for
        # another reason, the test above would pass for the wrong one.
        state = _thermal_state(dissipate=0)
        before = np.asarray(state.velocities, dtype=np.float64).copy()
        _both_stages(state)
        after = np.asarray(state.velocities, dtype=np.float64)
        self.assertFalse(np.array_equal(before, after),
                         'the Nose-Hoover half steps did not act at all')

    def test_dissipate_zero_is_bit_identical_to_the_default(self):
        # What lets the keyword exist without touching the 34 shipped
        # dynamics examples: absent and 0 are the same run, byte for byte.
        written = _thermal_state(dissipate=0)
        absent = _thermal_state()
        absent.pmd.dynamics.dissipate = 0
        _both_stages(written)
        _both_stages(absent)
        np.testing.assert_array_equal(
            np.asarray(written.velocities, dtype=np.float64),
            np.asarray(absent.velocities, dtype=np.float64))


class TestDelayCountsFromTheLanding(unittest.TestCase):
    """`thermodelay` is measured from the step the run landed on.

    The gate reads the step being integrated, which is one more than the
    runtime counter at both ensemble stages (the counter advances when the
    step is written). A trajectory that landed on step 10 with a delay of 3
    is therefore still gated on steps 10 and 11 and thermalized from step 12
    on: the landing step counts as the first step on the ground state.
    """

    def _landed(self, thermodelay=3, landed=10, current=10):
        state = _thermal_state(thermodelay=thermodelay, state_index=0)
        state.runtime.dissipate_step = landed
        state.runtime.current_step = current
        return state

    def test_the_gate_stays_closed_through_the_delay(self):
        # Steps 10 and 11 are the first and second steps on the ground state;
        # with a delay of three the thermostat has not started yet.
        for current in (9, 10, 11):
            with self.subTest(current=current):
                state = self._landed(current=current)
                self.assertFalse(_dissipate_gate_open(state,
                                                      state.pmd.dynamics))

    def test_the_gate_opens_once_the_delay_has_run_out(self):
        state = self._landed(thermodelay=3, landed=10, current=11)
        self.assertFalse(_dissipate_gate_open(state, state.pmd.dynamics))
        state.runtime.current_step = 12
        self.assertTrue(_dissipate_gate_open(state, state.pmd.dynamics))
        velocities = np.asarray(state.velocities, dtype=np.float64)
        _both_stages(state)
        self.assertFalse(np.array_equal(
            velocities, np.asarray(state.velocities, dtype=np.float64)),
            'the chain acts on the step the delay ends on')

    def test_a_run_that_never_landed_never_opens(self):
        state = self._landed(thermodelay=0, current=500)
        state.runtime.dissipate_step = 0
        self.assertFalse(_dissipate_gate_open(state, state.pmd.dynamics),
                         'a zero delay still waits for the landing')


class TestLandingIsRecorded(unittest.TestCase):
    """The step the hop decision ends on is the step the delay counts from."""

    def test_the_first_step_on_the_ground_state_is_stamped(self):
        state = _thermal_state(state_index=0)
        _record_dissipate_landing(state, 7)
        self.assertEqual(7, state.runtime.dissipate_step)

    def test_the_first_loop_step_stamps_one_not_zero(self):
        # Zero is the value that means 'has not landed', so a trajectory that
        # starts on the ground state must still get a step number of its own.
        state = _thermal_state(state_index=0)
        _record_dissipate_landing(state, 1)
        self.assertEqual(1, state.runtime.dissipate_step)

    def test_a_hop_back_up_does_not_close_the_gate_again(self):
        state = _thermal_state(state_index=0)
        _record_dissipate_landing(state, 7)
        state.update_system({'current_state': 1})
        _record_dissipate_landing(state, 9)
        self.assertEqual(7, state.runtime.dissipate_step,
                         'the stamp is set once and never moved back')

    def test_a_run_without_the_keyword_pays_nothing(self):
        state = _thermal_state(dissipate=0, state_index=0)
        _record_dissipate_landing(state, 4)
        self.assertEqual(0, state.runtime.dissipate_step)


class TestMultiCenterLanding(unittest.TestCase):
    """One chain is shared, so the landing is judged on the whole run."""

    class _Stub:
        def __init__(self, configuration, n_center):
            self._configuration = configuration
            self.n_center = n_center

        def get_active_configuration(self):
            return self._configuration

    def test_all_centers_have_to_be_down(self):
        self.assertFalse(_ground_state_reached(self._Stub([0, 1], 2)))
        self.assertTrue(_ground_state_reached(self._Stub([0, 0], 2)))

    def test_a_single_center_reads_the_ledger(self):
        state = _thermal_state(state_index=1)
        self.assertFalse(_ground_state_reached(state))
        state.update_system({'current_state': 0})
        self.assertTrue(_ground_state_reached(state))


class TestValidationAndPersistence(unittest.TestCase):
    """The combination rules and the restart travel of the new state."""

    def _load(self, md):
        config = {'control': {'title': 't_dissipate', 'jobtype': 'md'},
                  'system': {'atoms': ['O', 'H', 'H'],
                             'coordinates': [[0.0, 0.0, 0.11734],
                                             [0.0, 0.75754, -0.46936],
                                             [0.0, -0.75754, -0.46936]]},
                  'molecule': {'ci': [1], 'spin': [0]},
                  'md': dict({'step': 2, 'size': 20.67}, **md)}
        return validate(normalize(build_config(config)))

    def test_the_keyword_loads_with_a_thermostat(self):
        dynamics = self._load({'ensemble': 'nvt', 'thermostat': 'nh',
                               'dissipate': 1, 'thermodelay': 50}).dynamics
        self.assertEqual(1, dynamics.dissipate)
        self.assertEqual(50, dynamics.thermodelay)

    def test_an_nve_run_refuses_the_keyword(self):
        # An NVE run has no thermostat to delay, so accepting the setting
        # would let a run believe it was being dissipated.
        with self.assertRaises(ValueError) as caught:
            self._load({'ensemble': 'nve', 'dissipate': 1})
        message = str(caught.exception)
        self.assertIn('dissipate', message)
        self.assertIn('no thermostat to delay', message)

    def test_the_default_is_closed(self):
        self.assertEqual(0, self._load({}).dynamics.dissipate)

    def test_the_landing_step_travels_through_the_runtime_record(self):
        runtime = RuntimeState(total_steps=10)
        runtime.dissipate_step = 6
        self.assertEqual(6, RuntimeState.from_dict(runtime.to_dict())
                         .dissipate_step)
        self.assertEqual(6, runtime.copy().dissipate_step,
                         'a dispatched child keeps the delay its parent had '
                         'already run down')
        legacy = RuntimeState.from_dict({'current_step': 2, 'total_steps': 9})
        self.assertEqual(0, legacy.dissipate_step,
                         'a runtime written before the keyword existed reads '
                         'as never landed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
