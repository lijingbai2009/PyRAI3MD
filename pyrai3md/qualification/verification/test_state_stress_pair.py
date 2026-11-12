"""
======================================================================
The stress of a named state, and the box a crossing search steps on it.

A search that walks two surfaces together minimizes their average, so the
force on its box is the conjugate of the average too. Reading the active
state's tensor instead is not a coarser crossing search - it steps the box
on one surface while the atoms step on the average of two, inside one
generalized vector. The instrument here is built so the two answers land in
different places: two surfaces whose box springs rest at 18 and 20 Bohr put
the average's minimum at 19 and the single surface's at 18, so the endpoint
settles both the sign and the question of whether the state axis is read at
all.

What a single whole-system tensor cannot answer is refused by name, in the
words of the shape that would: a backend that means to serve a crossing
search in a relaxable box answers with (nstate, 3, 3), which is the shape
the properties schema declares and the numerical-strain route in the CP2K
skill already produces.

Oct 13 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from pyrai3md.workflow.optimizer import get_optimizer

SURFACES = (18.0, 20.0)          # Bohr: the two box springs' rest lengths
CURVE = 0.05                     # Hartree per Bohr^2
EDGE = 17.0                      # Bohr: start short of both, so it must move


class _PeriodicCrossing:
    """Two surfaces, each a box spring, and a gap that closes at their mean."""

    nstate = 2
    nac_pairs = [(0, 1)]

    def energy(self, edge, surface):
        return 0.5 * CURVE * (edge - SURFACES[surface]) ** 2

    def stress(self, edge, surface):
        """The tensor this surface's own energy implies, Hartree per Bohr^3.

        Inverted from `dE/dH = V sigma^T H^-T` for a cubic row, so the cell
        force the framework recovers is the derivative of the energy this object
        returns and nothing else.
        """
        derivative = CURVE * (edge - SURFACES[surface]) * BOHR_TO_ANGSTROM
        return np.diag([derivative / (edge * ANGSTROM_TO_BOHR) ** 2, 0.0, 0.0])


class TestStateStressReading(unittest.TestCase):
    """`state_stress(i)` answers about a named state or says it cannot."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._directory = tempfile.mkdtemp()
        os.chdir(self._directory)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._directory, ignore_errors=True)

    def _state(self, model, edge=SURFACES[0] + 1.0):
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'crossing_cell'
        config.system.atoms = ['H']
        config.system.natom = 1
        config.system.mass = [1.0]
        config.system.coordinates = [[0.0, 0.0, 0.0]]
        config.system.nstate = model.nstate
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.nac_pair_map = [list(pair) for pair in model.nac_pairs]
        config.system.pbc = [1, 1, 1]
        rows = np.eye(3) * (edge * BOHR_TO_ANGSTROM)
        config.system.lattice_vector = rows.tolist()
        config.optimize.optimize_cell = True
        config.optimize.opt_method = 'meci'
        config.optimize.opt_state_i = 0
        config.optimize.opt_state_j = 1
        state = PyRAIState(runtime=RuntimeState(total_steps=200))
        state.pmd = config
        return state

    def _evaluate(self, state, model, gap_coupling=True):
        edge = float(np.asarray(state.pmd.system.lattice_vector)[0, 0]
                     ) * ANGSTROM_TO_BOHR
        energies = [model.energy(edge, 0), model.energy(edge, 1)]
        stresses = [model.stress(edge, 0), model.stress(edge, 1)]
        gradient = np.zeros((model.nstate, 1, 3))
        coupling = (np.zeros((1, 1, 3), dtype=np.float64).tolist()
                    if gap_coupling else None)
        results = {'energy': energies,
                   'gradient': gradient.tolist(),
                   'stress_tensor': [stress.tolist() for stress in stresses]}
        if coupling is not None:
            results['nac'] = coupling
        state.set_result(results)
        return state

    def test_the_two_pages_are_the_two_surfaces(self):
        model = _PeriodicCrossing()
        state = self._evaluate(self._state(model), model)
        first = np.asarray(state.state_stress(0), dtype=np.float64)
        second = np.asarray(state.state_stress(1), dtype=np.float64)
        self.assertEqual(first.shape, (3, 3))
        self.assertEqual(second.shape, (3, 3))
        self.assertGreater(first[0, 0], 0.0,
                           'surface 0 starts beyond its rest length')
        self.assertLess(second[0, 0], 0.0,
                        'surface 0 and surface 1 must not answer the same thing, '
                        'or this file cannot tell a per-state reading from a '
                        'repeated one')
        self.assertIsNone(state.state_stress(model.nstate),
                          'an index past the state axis returned a tensor')

    def test_a_single_whole_system_tensor_names_the_shape_that_serves(self):
        model = _PeriodicCrossing()
        state = self._evaluate(self._state(model), model)
        single = np.asarray(state.result['stress_tensor'], dtype=np.float64)[0]
        state.set_result({**state.result.to_dict(),
                          'stress_tensor': single.tolist()})
        self.assertIsNone(state.state_stress(0),
                          'a (3, 3) reading has no state axis to select from')
        optimizer = get_optimizer('meci', cell_variables=True,
                                  active_state=0, active_state_j=1)
        with self.assertRaises(ValueError) as caught:
            optimizer._cell_stress(state)
        message = str(caught.exception)
        self.assertIn('(nstate, 3, 3)', message)
        self.assertIn('optimize_cell', message)

    def test_a_short_state_axis_is_refused_by_the_page_that_is_missing(self):
        """One page of the two is not a crossing objective, and says which page.

        A backend that serves the ground state alone answers with a (1, 3, 3):
        the state axis exists, so the shape test passes, but the second surface
        has no page to read. The refusal names the half that is absent rather
        than the general shape, because the fix differs: one page short means the
        run asked for a state pair the backend never produced.
        """
        model = _PeriodicCrossing()
        state = self._evaluate(self._state(model), model)
        single_page = np.asarray(state.result['stress_tensor'],
                                 dtype=np.float64)[:1]
        state.set_result({**state.result.to_dict(),
                          'stress_tensor': single_page.tolist()})
        self.assertIsNotNone(state.state_stress(0),
                            'page 0 is present and must be readable')
        self.assertIsNone(state.state_stress(1),
                          'a (1, 3, 3) has no page for the second surface')
        optimizer = get_optimizer('meci', cell_variables=True,
                                  active_state=0, active_state_j=1)
        with self.assertRaises(ValueError) as caught:
            optimizer._cell_stress(state)
        self.assertIn("only one state's tensor", str(caught.exception))


    def _verdict_at_the_average_s_rest_length(self, optimizer, model):
        """Judge the box at the one geometry the two readings cannot agree at.

        At 19 Bohr the two surfaces' cell forces are equal and opposite, so the
        average's is exactly zero while either surface alone is not. The step
        count is taken out of it: one evaluation answers.
        """
        state = self._state(model, edge=sum(SURFACES) / 2)
        self._evaluate(state, model)
        optimizer.note_evaluation(state)
        optimizer.run(state)
        converged, report = optimizer.convergence(state)
        return converged, report



if __name__ == '__main__':
    unittest.main()


class _CompositeStressState:
    """The two attributes a single-center composite branch reads, and nothing more.

    `meci._cell_stress` asks a layered plan for exactly two things: the plan
    itself (`theory`, whose tasks carry `layer_id`) and the whole-system stress.
    A stub that holds those two and no more is therefore a statement about the
    branch - if it reached for anything else, this object would raise instead of
    the reading passing.
    """

    def __init__(self, stress, layers):
        self.theory = [{'layer_id': layer} for layer in layers]
        self._stress = None if stress is None else np.asarray(
            stress, dtype=np.float64)

    def active_stress(self):
        """The whole-system tensor, already projected onto one state."""
        return self._stress

    def state_stress(self, index):
        """A state-resolved page: present for state 0, absent for state 1.

        That is what a backend answering with a (1, 3, 3) hands over, and it is
        the condition the two branches below are told apart by.
        """
        if self._stress is None or index != 0:
            return None
        return self._stress


