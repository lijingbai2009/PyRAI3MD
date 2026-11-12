"""
======================================================================
A strict condition held while the box moves: the cell force the
condition supplies, and the box a run reaches with and without it.

The engine reports the stress of the potential. A length the plan holds is
not part of that potential - it is a reaction, and the reaction has a
lattice derivative of its own, because under the convention a variable-cell
step takes (atoms ride the box, `optimizer/base.py::_cell_ride`) a lattice
entry moves every atom by its fractional coordinate times that entry. The
term is A5/D-14 of `dev/methods/CONSTRAINT_DEVELOPMENT.md` Appendix B section 10,
and this file is the measurement that fixes both its size and its sign.

The instrument is analytic, so the answer is known before the run:

  E = 1/2 k (h - h0)^2 + 1/2 c (d - a)^2,     d = |r2 - r1|,
  with a bond condition holding d at d0.

The pair term's contribution to the fixed-fractional lattice derivative is
`c (d - a) n_b s_2a`, and the multiplier `tangent_split` solves for is exactly
`c (d - a)`, so the condition absorbs that contribution *whole*: with the term
applied the box relaxes to h0, and without it the box stops where the pair
spring pushes it, at h0 - c (d0 - a) s_2x / k. The two answers differ by a
number this file computes rather than reads back.

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
from pyrai3md.schema import SystemConstraintConfig
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.lattice import cart_to_frac
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from pyrai3md.workflow.constraints import projection
from pyrai3md.workflow.constraints import runtime as constraint_runtime
from pyrai3md.workflow.optimizer import get_optimizer
from pyrai3md.utilities.lattice import CELL_COMPONENT_SLOTS

# The model, in Bohr and Hartree. The box spring sets where the box belongs,
# the pair spring is what the condition holds, and the two are deliberately
# inconsistent (PAIR_REST != BOND_TARGET) so the multiplier is not zero: a
# condition that costs nothing supplies no cell force and would prove nothing.
BOX_REST = 10.0 * ANGSTROM_TO_BOHR
BOX_CURVE = 0.05
PAIR_CURVE = 0.5
PAIR_REST = 1.6
BOND_TARGET = 0.97 * ANGSTROM_TO_BOHR


class _HeldBox:
    """Energy, gradient and stress of the model above, at one geometry."""

    def __init__(self, pair_rest=PAIR_REST, target=BOND_TARGET):
        self.pair_rest = pair_rest
        self.target = target

    def parts(self, coords_bohr, edge_bohr):
        delta = coords_bohr[1] - coords_bohr[0]
        distance = float(np.linalg.norm(delta))
        axis = delta / distance
        stretch = distance - self.pair_rest
        energy = (0.5 * BOX_CURVE * (edge_bohr - BOX_REST) ** 2
                  + 0.5 * PAIR_CURVE * stretch ** 2)
        gradient = np.zeros((2, 3))
        gradient[1] = PAIR_CURVE * stretch * axis
        gradient[0] = -gradient[1]
        return energy, gradient, distance, axis, stretch

    def cell_derivative(self, coords_bohr, edge_bohr):
        """dE/dH at fixed fractional coordinates, the (3, 3) the stress encodes.

        The box spring contributes on the xx entry alone; the pair spring
        contributes through the ride, which is the chain term
        `sum_i g_i[b] s_i[a]` and is exactly what the condition's multiplier
        absorbs.
        """
        _energy, gradient, _distance, _axis, _stretch = self.parts(
            coords_bohr, edge_bohr)
        lattice = np.eye(3) * edge_bohr
        fractional = cart_to_frac(coords_bohr, lattice)
        total = np.zeros((3, 3))
        total[0, 0] = BOX_CURVE * (edge_bohr - BOX_REST)
        total += np.outer(fractional[1] - fractional[0], gradient[1])
        return total

    def stress(self, coords_bohr, edge_bohr):
        """The tensile-positive tensor whose conjugate is the derivative above."""
        derivative = self.cell_derivative(coords_bohr, edge_bohr)
        # For a cubic row, `V sigma^T H^-T = h^2 sigma^T`, so the tensor that
        # reproduces a given derivative is its transpose over the squared edge.
        return derivative.T / edge_bohr ** 2

    def evaluate(self, state):
        coords = np.asarray(state.coordinates, dtype=np.float64) * ANGSTROM_TO_BOHR
        rows = np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
        edge = float(rows[0, 0]) * ANGSTROM_TO_BOHR
        energy, gradient = self.parts(coords, edge)[:2]
        # The model is written in Bohr, so its gradient is already Hartree per
        # Bohr, which is the unit `state.result` carries and the optimizer reads.
        # A model written in Angstrom would owe the chain-rule factor here; this
        # one does not, and multiplying anyway scales the atomic half against the
        # stress half and puts the box in the wrong place.
        state.set_result({
            'energy': [energy],
            'gradient': gradient.reshape(1, 2, 3).tolist(),
            'stress_tensor': self.stress(coords, edge).reshape(1, 3, 3).tolist(),
        })
        return state


class TestTheConditionHasItsOwnCellForce(unittest.TestCase):

    def setUp(self):
        self._cwd = os.getcwd()
        self._directory = tempfile.mkdtemp()
        os.chdir(self._directory)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._directory, ignore_errors=True)

    def _state(self, edge_angstrom):
        """Two atoms along x at the condition's own target, in a cubic box."""
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'held_box'
        config.system.atoms = ['O', 'H']
        config.system.natom = 2
        config.system.mass = [15.999, 1.008]
        config.system.coordinates = [[0.0, 0.0, 0.0],
                                     [BOND_TARGET * BOHR_TO_ANGSTROM, 0.0, 0.0]]
        config.system.nstate = 1
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.pbc = [1, 1, 1]
        config.system.lattice_vector = [[edge_angstrom, 0.0, 0.0],
                                        [0.0, edge_angstrom, 0.0],
                                        [0.0, 0.0, edge_angstrom]]
        config.system.cell = [edge_angstrom, edge_angstrom, edge_angstrom,
                              90.0, 90.0, 90.0]
        config.system.constraints = SystemConstraintConfig(**{
            'units': {'length': 'angstrom'},
            'constraints': [{'id': 'oh', 'kind': 'bond', 'bonds': [[1, 2]]}],
        })
        state = PyRAIState(runtime=RuntimeState(total_steps=200))
        state.pmd = config
        constraint_runtime.bind(state)
        return state

    def _edge(self, state):
        return float(np.asarray(state.pmd.system.lattice_vector,
                                dtype=np.float64)[0, 0]) * ANGSTROM_TO_BOHR

    def _relax(self, box, edge_angstrom, limit=200, virial=True):
        """Run the same loop `Optimize.run` runs, on the model box."""
        state = self._state(edge_angstrom)
        # The same criteria the model-box relaxations elsewhere use. Tighter ones
        # do not make this a better test: the rejection floor is 1e-6 Hartree, so
        # a run asked to converge below the size of its own last overshoot spends
        # its budget refusing steps rather than taking them.
        optimizer = get_optimizer('minima', maxiter=limit,
                                  energy_threshold=1e-8,
                                  gradient_threshold=1e-4,
                                  gradient_max_threshold=4e-4,
                                  displacement_threshold=1e-4,
                                  displacement_max_threshold=4e-4,
                                  # The cell thresholds are stated here rather than
                                  # left to the schema defaults because this fixture
                                  # is a 9.6 A box, not the 25 A box the defaults are
                                  # calibrated on: at that edge the atomic pair above
                                  # means 89.4 / 357.6 bar (`p = g a^2 / V` in atomic
                                  # units) and 1.0417e-5 / 4.1667e-5 of strain. The
                                  # box is not what this test judges, and a default
                                  # six times tighter would make the step budget, and
                                  # so the assertion, about the criterion instead.
                                  cell_stress_rms_threshold=89.4,
                                  cell_stress_max_threshold=357.6,
                                  cell_strain_rms_threshold=1.0417e-5,
                                  cell_strain_max_threshold=4.1667e-5,
                                  cell_variables=True)
        report = {}
        for _step in range(limit):
            box.evaluate(state)
            optimizer.note_evaluation(state)
            if not virial:
                self._strip_virial(optimizer)
            optimizer.run(state)
            constraint_runtime.constrain_state(state)
            converged, report = optimizer.convergence(state)
            if converged:
                return state, report, True
        return state, report, False

    def _strip_virial(self, optimizer):
        # The mutation: a split that reports no lattice derivative, which is what
        # the framework did before this round.
        split = optimizer.constraint_split
        if split is not None:
            split.pop('cell_virial', None)

    def test_the_held_length_takes_its_share_out_of_the_cell_force(self):
        box = _HeldBox()
        state, report, _converged = self._relax(box, 9.6)
        edge = self._edge(state)
        # The tolerance is the residual the run is allowed to leave, and it is
        # not a round number: a search stops with a cell force it can no longer
        # pay for, and the offset that force implies is itself over the box
        # spring. Asserting tighter than that would grade the trust radius
        # rather than the term under test; the control below is 20x further out,
        # which is what separates the two.
        residual = report.get('Cell Gradient Max', 0.0)
        allowed = max(0.05, residual / BOX_CURVE * 2.0)
        self.assertLess(
            abs(edge - BOX_REST), allowed,
            msg='with the condition\'s lattice derivative applied the box '
                'belongs at the box spring\'s own rest length %.6f Bohr; it '
                'stopped at %.6f (offset %.2e, residual cell force %.2e, '
                'allowed %.2e)'
                % (BOX_REST, edge, abs(edge - BOX_REST), residual, allowed))
        self.assertLess(
            report['Gradient Max'], 1.0e-4,
            msg='the atomic half did not converge, so the box reading below is '
                'not the reading of a stationary geometry')

    def test_without_the_term_the_box_stops_where_the_pair_spring_pushed_it(self):
        """The negative control, with its answer computed and not read back.

        Removing the term leaves the pair spring's chain contribution in the cell
        force. At the held length the multiplier is `c (d0 - a)` and the ride
        weights it by the hydrogen's fractional coordinate, so the box stops at
        `h0 - c (d0 - a) s_x / k` - short of h0, by an amount this asserts
        rather than measures twice.
        """
        box = _HeldBox()
        state, _report, _converged = self._relax(box, 9.6, virial=False)
        edge = self._edge(state)
        coords = np.asarray(state.coordinates, dtype=np.float64) * ANGSTROM_TO_BOHR
        lattice = np.eye(3) * self._edge(state)
        fractional = cart_to_frac(coords, lattice)
        multiplier = PAIR_CURVE * (BOND_TARGET - PAIR_REST)
        predicted = BOX_REST - multiplier * fractional[1, 0] / BOX_CURVE
        self.assertLess(
            edge, BOX_REST - 0.05,
            msg='without the term the box should have been pushed short of %.6f '
                'Bohr and stopped at %.6f' % (BOX_REST, edge))
        self.assertAlmostEqual(
            edge, predicted, delta=2.0e-2,
            msg='the box stopped at %.6f Bohr; dropping the condition\'s lattice '
                'derivative puts it at %.6f' % (edge, predicted))

    def test_the_contraction_is_the_derivative_of_the_condition_itself(self):
        """The term against a finite difference of the condition, not of a model.

        `cell_virial` with a unit multiplier must be `d g / d H` along the ride,
        which is a difference of the held length itself and needs no energy, no
        spring constant and no sign convention.
        """
        coords = np.array([[0.0, 0.0, 0.0], [BOND_TARGET, 0.0, 0.0]])
        edge = BOX_REST
        lattice = np.eye(3) * edge
        fractional = cart_to_frac(coords, lattice)
        strain = 1.0e-5
        measured = np.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                if row * 3 + column not in CELL_COMPONENT_SLOTS:
                    continue
                plus = lattice.copy()
                minus = lattice.copy()
                step = strain * edge
                plus[row, column] += step
                minus[row, column] -= step
                value = []
                for candidate in (plus, minus):
                    ridden = fractional @ candidate
                    # The condition's atoms are 0-based here, as everywhere in
                    # the plan; the input spelling is the one that is 1-based.
                    separation = ridden[1] - ridden[0]
                    value.append(float(np.linalg.norm(separation)))
                measured[row, column] = (value[0] - value[1]) / (2.0 * step)
        got = projection.cell_virial(coords, fractional, [1.0],
                                     conditions=[('bond', (0, 1))])
        np.testing.assert_allclose(
            got, measured, atol=1.0e-9,
            err_msg='the contraction is not the lattice derivative of the '
                    'condition it is built from')

    def test_a_fixed_box_is_handed_no_term_it_cannot_use(self):
        """No cell, no lattice derivative: the key is absent, not zero."""
        state = self._state(9.6)
        state.pmd.system.pbc = [0, 0, 0]
        box = _HeldBox()
        box.evaluate(state)
        from pyrai3md.workflow.optimizer import get_optimizer as build
        optimizer = build('minima', cell_variables=False)
        optimizer.note_evaluation(state)
        split = optimizer.constraint_split
        self.assertIsNotNone(split, 'the plan declares a bond, so there is a split')
        self.assertNotIn('cell_virial', split)


class TestWhatAVariableCellRunRefusesAndReports(unittest.TestCase):
    """The named refusals and the observable that keeps a zero honest."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._directory = tempfile.mkdtemp()
        os.chdir(self._directory)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._directory, ignore_errors=True)

    def _state(self, block, coordinates, atoms=('O', 'H')):
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'refusal_case'
        config.system.atoms = list(atoms)
        config.system.natom = len(atoms)
        config.system.mass = [15.999] + [1.008] * (len(atoms) - 1)
        config.system.coordinates = [list(row) for row in coordinates]
        config.system.nstate = 1
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.pbc = [1, 1, 1]
        config.system.lattice_vector = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                        [0.0, 0.0, 10.0]]
        config.system.cell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        config.optimize.optimize_cell = True
        config.system.constraints = SystemConstraintConfig(**block)
        state = PyRAIState(runtime=RuntimeState(total_steps=4))
        state.pmd = config
        constraint_runtime.bind(state)
        return state

    def _optimize(self, state):
        from pyrai3md.workflow.optimize import Optimize

        class Calculator:
            def __init__(self, given):
                self.state = given

            def run(self, target_states=None):
                self.state.set_result({
                    'energy': [0.0],
                    'gradient': np.zeros(
                        (1, self.state.pmd.system.natom, 3)).tolist(),
                    'stress_tensor': np.zeros((1, 3, 3)).tolist(),
                })
                return dict(self.state.result)

            def get_state(self):
                return self.state

            def sync(self):
                pass

        return Optimize(calculator=Calculator(state), maxiter=2)

    def test_a_held_length_is_priced_and_runs(self):
        state = self._state({'units': {'length': 'angstrom'},
                             'constraints': [{'id': 'oh', 'kind': 'bond',
                                              'bonds': [[1, 2]]}]},
                            [[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
        self._optimize(state)

    def test_a_held_angle_runs_and_its_term_is_in_the_cell_force(self):
        """An angle is priced now, and the run carries the term it was refused for.

        The measurement is `TestAnAngleAndATorsionHaveTheSameCellForce`: the
        contraction against a finite strain of the condition itself, and the
        finding that a box step cannot re-choose which image the condition spans.
        This test adds the other half - that the term is actually in play when the
        plan holds a bent condition, rather than merely available.
        """
        state = self._state(
            {'units': {'length': 'angstrom', 'angle': 'degree'},
             'constraints': [{'id': 'hoh', 'kind': 'angle',
                              'bonds': [[2, 1, 3]]}]},
            [[0.0, 0.0, 0.0], [0.97, 0.0, 0.0],
             [-0.24, 0.93, 0.0]], atoms=('O', 'H', 'H'))
        self._optimize(state)          # must not raise
        from pyrai3md.workflow.optimize import Optimize

        class _Zero:
            def __init__(self, given):
                self.state = given

            def run(self, target_states=None):
                self.state.set_result({
                    'energy': [0.0],
                    'gradient': np.zeros(
                        (1, self.state.pmd.system.natom, 3)).tolist(),
                    'stress_tensor': np.zeros((1, 3, 3)).tolist()})
                return dict(self.state.result)

            def get_state(self):
                return self.state

            def sync(self):
                pass

        # Drive one evaluation through the optimizer the way `Optimize` does, and
        # read the split it hands the cell step: an angle plan must land a
        # non-zero lattice derivative there, or the gate was opened on a term no
        # run ever applies.
        from pyrai3md.workflow.optimizer import get_optimizer
        from pyrai3md.workflow import optimize as optimize_module

        state.set_result({
            'energy': [0.0],
            'gradient': np.zeros((1, state.pmd.system.natom, 3)).tolist(),
            'stress_tensor': np.zeros((1, 3, 3)).tolist()})
        optimizer = get_optimizer('minima', maxiter=2, cell_variables=True)
        optimizer.note_evaluation(state)
        split = optimizer.constraint_split
        self.assertIsNotNone(split, 'the plan declares an angle condition')
        virial = split.get('cell_virial')
        self.assertIsNotNone(virial,
                             msg='an angle plan supplied no cell term at all: '
                                 '%r' % (split,))
        # And it is *zero* here, which is the correct reading rather than a dead
        # branch: the cell term is the multiplier times the contraction, and the
        # multiplier answers the engine's gradient. This probe hands the framework
        # a zero gradient and a geometry that already satisfies the declared angle,
        # so nothing is being held against anything. A non-zero number here would
        # mean the term was built from something other than the reaction, and the
        # scaling of the term against its multiplier is measured in
        # TestAnAngleAndATorsionHaveTheSameCellForce.
        self.assertEqual(float(np.max(np.abs(virial))), 0.0)
        self.assertIn('angle', optimize_module.CELL_MEASURED_CONDITIONS)

    def test_a_condition_outside_the_priced_set_is_still_refused_by_name(self):
        """The gate is a tripwire for the next condition kind, not a museum piece.

        Nothing in the schema can declare an unpriced kind today, so the guard is
        exercised by narrowing the priced set the way a future addition would
        leave it: the refusal must name the kind and stop before any record is
        written.
        """
        from pyrai3md.workflow import optimize as optimize_module

        state = self._state(
            {'units': {'length': 'angstrom', 'angle': 'degree'},
             'constraints': [{'id': 'hoh', 'kind': 'angle',
                              'bonds': [[2, 1, 3]]}]},
            [[0.0, 0.0, 0.0], [0.97, 0.0, 0.0],
             [-0.24, 0.93, 0.0]], atoms=('O', 'H', 'H'))
        keep = optimize_module.CELL_MEASURED_CONDITIONS
        optimize_module.CELL_MEASURED_CONDITIONS = frozenset(('bond',))
        try:
            with self.assertRaises(ValueError) as caught:
                self._optimize(state)
        finally:
            optimize_module.CELL_MEASURED_CONDITIONS = keep
        message = str(caught.exception)
        self.assertIn('angle', message)
        self.assertIn('optimize_cell', message)
        self.assertFalse(os.path.exists('opt_status.txt'),
                         'the refusal came after a record was written')

    def test_the_ride_of_a_frozen_atom_is_reported_not_hidden(self):
        """A pinned atom's net motion is zero; the work undone to make it so is not.

        The cell step rides every atom, and the position solve puts a frozen one
        back on its anchor. Reporting only the net leaves a run that carried a
        pinned atom across the box reading as a constraint that held, so the ride
        is its own row.
        """
        box = _HeldBox()
        # Built with the fix in the block from the start: the plan is derived once
        # and cached on the state, so a fix added afterwards is a fix the run
        # never declared.
        state = self._state(
            {'units': {'length': 'angstrom'},
             'constraints': [{'id': 'oh', 'kind': 'bond', 'bonds': [[1, 2]]}],
             'fixes': [{'id': 'pin', 'atoms': [2]}]},
            [[0.0, 0.0, 0.0], [BOND_TARGET * BOHR_TO_ANGSTROM, 0.0, 0.0]])
        # The hydrogen sits at a non-zero fractional coordinate, so a lattice
        # step carries it. An atom at the origin would prove nothing.
        optimizer = get_optimizer('minima', maxiter=6,
                                  energy_threshold=1e-8,
                                  gradient_threshold=1e-4,
                                  gradient_max_threshold=4e-4,
                                  displacement_threshold=1e-4,
                                  displacement_max_threshold=4e-4,
                                  cell_variables=True)
        anchor = np.asarray(state.coordinates, dtype=np.float64)[1].copy()
        report = {}
        for _step in range(6):
            box.evaluate(state)
            optimizer.note_evaluation(state)
            optimizer.run(state)
            constraint_runtime.constrain_state(state)
            _converged, report = optimizer.convergence(state)
        self.assertIn('Frozen Ride Max', report)
        self.assertGreater(
            report['Frozen Ride Max'], 0.0,
            'a lattice step carried nothing the run had frozen, so the row is not '
            'reading the ride')
        moved = np.asarray(state.coordinates, dtype=np.float64)[1]
        self.assertLess(
            float(np.max(np.abs(moved - anchor))), 1.0e-9,
            'the pinned atom did not return to its anchor')


if __name__ == '__main__':
    unittest.main()


class TestAnAngleAndATorsionHaveTheSameCellForce(unittest.TestCase):
    """The term a variable-cell run applies is the condition's own lattice
    derivative, and that holds for the bent conditions as it does for a length.

    A bond was priced end to end above: with the term the box stops at the box
    spring's rest length, without it 0.227 Bohr short. What that experiment does
    NOT establish is the tensor's *content* for an angle or a torsion, and a
    variable-cell run holding one of those was refused until this class measured
    it. Three things are measured here, each for its own reason:

    - `cell_virial` at unit multiplier must equal a finite difference of the
      condition value along the ride. This is the quantity that enters the cell
      force, and it is checked against the definition rather than against another
      branch of the same function.
    - the image a condition is measured on cannot change when only the box moves.
      Under the ride convention (`optimizer/base.py::_cell_ride`) the fractional
      coordinates are what is held, and the wrapping is a function of those alone,
      so a lattice step cannot re-choose which neighbour image a three- or
      four-atom condition spans. That was the stated reason an angle or a torsion
      could not be trusted with a moving box, and it is answered by measurement
      instead of by the formula.
    - the same checks hold with atoms unfolded past a cell face, the case where a
      wrong image assignment would show up at all.
    """

    # Fractional coordinates of a four-atom chain. Atoms 2 and 3 carry an integer
    # part, i.e. they are the unfolded images of atoms sitting a full cell over,
    # which is the geometry a periodic run actually measures a condition on.
    FRAC = np.array([
        [0.05, 0.10, 0.12],
        [0.45, 0.12, 0.15],
        [1.85, 0.20, 0.10],
        [1.20, 0.35, 0.22],
    ])
    BOX = np.diag([18.0, 20.0, 22.0])          # Bohr
    STRAIN = 1.0e-5
    # Declared after the measurement, not typed as a round number: the worst
    # analytic-minus-finite-difference entry over all nine movable components is
    # 1.13e-11 rad per Bohr for the angle and 2.73e-12 for the torsion, and the
    # derivative itself is of order 1e-2, so this leaves the difference between a
    # correct contraction and a mis-scaled one (1e-2 against 1e-3, a factor of ten)
    # four orders of magnitude outside the gate.
    TOLERANCE = 1.0e-9

    def _coords(self, lattice, fractional=None):
        return (self.FRAC if fractional is None else fractional) @ lattice

    def _derivative_by_strain(self, kind, atoms):
        """d(condition)/d H[a,b] at fixed fractional coordinates, by strain."""
        from pyrai3md.utilities import cv

        def value(lattice):
            coords = self._coords(lattice)
            if kind == 'angle':
                return cv.angle(coords, *atoms)
            return cv.dihedral(coords, *atoms)

        # Only the components a cell step can move are compared. The optimizer
        # reads the six entries `CELL_COMPONENT_SLOTS` names (a lower-triangular
        # deformation), and the other three are not independent variables; the
        # tensor `cell_virial` returns is full, so a comparison over all nine
        # would grade entries no run ever consumes. The bond criterion above
        # passes either way because a pair lying along x has zero derivative in
        # those three by symmetry - a bent condition does not, which is why this
        # fixture is deliberately non-symmetric.
        slots = list(CELL_COMPONENT_SLOTS)
        measured = np.full((3, 3), np.nan)
        edge = float(self.BOX[0, 0])
        step = self.STRAIN * edge
        for row in range(3):
            for column in range(3):
                if row * 3 + column not in slots:
                    continue
                plus = self.BOX.copy()
                minus = self.BOX.copy()
                plus[row, column] += step
                minus[row, column] -= step
                high = value(plus)
                low = value(minus)
                if kind == 'dihedral':
                    # The value lives in (-pi, pi], so a branch crossing would
                    # read as a jump of 2*pi and not as a derivative. Take the
                    # wrapped difference; the sizes here are 1e-5 rad, four
                    # orders below the cut, and the image test below shows why no
                    # crossing is reachable from a box step.
                    high = low + (high - low + np.pi) % (2.0 * np.pi) - np.pi
                measured[row, column] = (high - low) / (2.0 * step)
        return measured

    def _assert_contraction_matches(self, kind, atoms):
        from pyrai3md.workflow.constraints import projection

        coords = self._coords(self.BOX)
        got = projection.cell_virial(
            coords, self.FRAC, [1.0], conditions=[(kind, atoms)])
        measured = self._derivative_by_strain(kind, atoms)
        movable = np.isfinite(measured)
        np.testing.assert_allclose(
            got[movable], measured[movable], atol=self.TOLERANCE,
            err_msg='the %s contraction is not the lattice derivative of the '
                    'condition it is built from' % kind)
        return got, measured

    def test_the_angle_contraction_is_the_derivative_of_the_condition(self):
        got, measured = self._assert_contraction_matches('angle', (0, 1, 2))
        # The arm that makes this more than a tautology: the derivative is not
        # zero, so a contraction that returned zeros for a bent condition would
        # fail here instead of passing a shape check.
        self.assertGreater(
            float(np.max(np.abs(measured[np.isfinite(measured)]))), 1.0e-4)
        self.assertEqual(got.shape, (3, 3))

    def test_the_torsion_contraction_is_the_derivative_of_the_condition(self):
        _got, measured = self._assert_contraction_matches(
            'dihedral', (0, 1, 2, 3))
        self.assertGreater(
            float(np.max(np.abs(measured[np.isfinite(measured)]))), 1.0e-4)

    def test_the_term_scales_with_the_multiplier_it_is_handed(self):
        """The contraction is linear in the reaction it is built from.

        A cell term that ignored its multiplier, or read it off the wrong row,
        would still pass a unit-multiplier comparison against a finite difference.
        Scaling the multiplier by 2.5 must scale every entry by 2.5, and a zero
        multiplier must give a zero tensor - the same shape this file insists a
        zero row is never a silent answer.
        """
        from pyrai3md.workflow.constraints import projection

        coords = self._coords(self.BOX)
        one = projection.cell_virial(coords, self.FRAC, [1.0],
                                     conditions=[('angle', (0, 1, 2))])
        scaled = projection.cell_virial(coords, self.FRAC, [2.5],
                                        conditions=[('angle', (0, 1, 2))])
        np.testing.assert_allclose(scaled, 2.5 * one, rtol=1.0e-12,
                                   err_msg='the angle term is not linear in its '
                                           'multiplier')
        self.assertGreater(float(np.max(np.abs(one))), 0.0,
                           msg='a zero unit term makes the scaling arm vacuous')
        silent = projection.cell_virial(coords, self.FRAC, [0.0],
                                       conditions=[('angle', (0, 1, 2))])
        np.testing.assert_allclose(silent, np.zeros((3, 3)), atol=0.0,
                                   err_msg='a condition that reacts against '
                                           'nothing supplied a cell force anyway')

    def test_the_condition_is_not_measured_on_the_wrong_side_of_the_box(self):
        """Atoms 2 and 3 are a full cell over: the angle they define is the one
        the run measures, and moving the box cannot move them back."""
        from pyrai3md.utilities import cv
        from pyrai3md.utilities.lattice import (wrap_fractional_with_image,
                                                unwrap_coordinates)

        coords = self._coords(self.BOX)
        straight = cv.angle(coords, 1, 2, 3)
        folded, images = wrap_fractional_with_image(self.FRAC, [1, 1, 1])
        for scale in (0.95, 1.0, 1.05, 1.12):
            lattice = self.BOX * scale
            back = unwrap_coordinates(folded @ lattice, images, lattice)
            np.testing.assert_allclose(back, coords * scale, atol=1.0e-9,
                                       err_msg='the round trip through the '
                                               'wrapped frame moved an atom by a '
                                               'lattice vector at scale %.2f'
                                               % scale)
            np.testing.assert_array_equal(
                wrap_fractional_with_image(self.FRAC, [1, 1, 1])[1], images,
                err_msg='the image assignment changed while only the box moved')
        # And the condition really is measured across the face: folding the
        # positions into one cell and taking the naive difference would be a
        # different angle, so a reader that skipped the unfolding could not pass
        # the assertions above by accident.
        wrapped_coords = folded @ self.BOX
        gap = abs(straight - cv.angle(wrapped_coords, 1, 2, 3))
        self.assertGreater(gap, 0.05,
                           msg='the wrapped and unfolded readings agree to '
                               '%.4f rad, so this geometry does not exercise '
                               'the image at all' % gap)


class TestAnAngleMovesTheCellForceByItsOwnDerivative(unittest.TestCase):
    """The end-to-end half of #351, on the quantity the optimizer consumes.

    #351 priced the tensor (`cell_virial` against a finite strain of the
    condition) and opened the gate on that. What it did not show is the step the
    bond experiment shows: that the term reaches the cell force the optimizer
    actually steps along, with the right sign, for a bent condition.

    The instrument makes that decidable without predicting a converged box. A
    bond relaxes one cell entry, so its endpoint has a closed form. An angle
    couples several entries at once (measured: its derivative is non-zero in six
    of them), so an endpoint would need a self-consistent solve over six slots,
    and a wrong sign would still produce *a* fixed point somewhere. Instead the
    model here is built so that the answer is exact: the engine's stress is
    constructed to be the angle spring's own lattice derivative and nothing else,
    so

      - with the condition's term applied, the cell force is ZERO, and
      - without it, the cell force is that derivative, entry by entry,

    and the prediction on the right-hand side is computed in this test from
    `utilities.cv` and the fractional coordinates - never from
    `projection.cell_virial`, which is the thing under test. A sign error, a
    transposed tensor, or a term scaled by the wrong cell factor moves at least
    one of the six entries and the assertions below say which.
    """

    # Bohr, cubic box. A real molecule, because the framework perceives a bond
    # graph before it will measure a condition across a periodic boundary
    # (`constraints/images.py::require_same_component` refuses the alternative
    # rather than inventing an image rule), and bent out of any plane so the
    # derivative has a number of its own in several cell entries. Atom order is
    # (H, O, H): `cv.angle` measures at its middle argument, and the 1-based
    # spelling of the condition says the same thing.
    CARTESIAN = np.array([
        [3.77, 3.63, 4.00],
        [2.00, 3.00, 4.00],
        [1.37, 4.77, 4.45],
    ])
    EDGE = 18.0
    STIFFNESS = 0.5                        # Hartree per radian squared
    OFFSET = 0.05                          # radians the condition holds away from
                                           # the spring's own rest angle, so the
                                           # multiplier is not zero
    ATOMS = ('H', 'O', 'H')
    MASSES = [1.008, 15.999, 1.008]

    def setUp(self):
        self._cwd = os.getcwd()
        self._directory = tempfile.mkdtemp()
        os.chdir(self._directory)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._directory, ignore_errors=True)

    def _model(self):
        """Energy, gradient and the stress whose conjugate is the same derivative."""
        from pyrai3md.utilities import cv
        from pyrai3md.utilities.lattice import frac_to_cart

        lattice = np.eye(3) * self.EDGE
        fractional = self.CARTESIAN / self.EDGE
        coords = frac_to_cart(fractional, lattice)
        angle, jac = cv.angle(coords, 0, 1, 2, grad=True)
        # The spring sits at `angle - OFFSET`, so holding the angle at its current
        # value costs `STIFFNESS * OFFSET` and the reaction's multiplier is exactly
        # that much per radian.
        energy = 0.5 * self.STIFFNESS * self.OFFSET ** 2
        multiplier = self.STIFFNESS * self.OFFSET
        gradient = np.zeros((3, 3))
        for position, atom in enumerate((0, 1, 2)):
            gradient[atom] += multiplier * jac[position]
        # dE/dH[a,b] = sum_i s_i[a] g_i[b] under the ride; for a cubic row the
        # tensor whose conjugate reproduces it is the transpose over h squared.
        derivative = fractional.T @ gradient
        stress = derivative.T / self.EDGE ** 2
        predicted = derivative.reshape(-1)[list(CELL_COMPONENT_SLOTS)]
        return (energy, gradient, stress, angle, multiplier, predicted)

    def _state(self, energy, gradient, stress):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema import SystemConstraintConfig
        from pyrai3md.schema.config import PyRAIConfig
        from pyrai3md.utilities.units import BOHR_TO_ANGSTROM
        from pyrai3md.workflow.constraints import runtime as constraint_runtime

        angstrom = BOHR_TO_ANGSTROM
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'held_angle'
        config.system.atoms = list(self.ATOMS)
        config.system.natom = 3
        config.system.mass = list(self.MASSES)
        config.system.coordinates = (self.CARTESIAN * angstrom).tolist()
        config.system.nstate = 1
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.pbc = [1, 1, 1]
        edge = self.EDGE * angstrom
        config.system.lattice_vector = [[edge, 0.0, 0.0],
                                        [0.0, edge, 0.0],
                                        [0.0, 0.0, edge]]
        config.system.cell = [edge, edge, edge, 90.0, 90.0, 90.0]
        config.optimize.optimize_cell = True
        config.system.constraints = SystemConstraintConfig(**{
            'units': {'length': 'angstrom', 'angle': 'radian'},
            'constraints': [{'id': 'ohh', 'kind': 'angle',
                             'bonds': [[1, 2, 3]]}]})
        state = PyRAIState(runtime=RuntimeState(total_steps=4))
        state.pmd = config
        constraint_runtime.bind(state)
        state.set_result({
            'energy': [float(energy)],
            'gradient': gradient.reshape(1, 3, 3).tolist(),
            'stress_tensor': stress.reshape(1, 3, 3).tolist(),
        })
        return state

    def _cell_force(self, state, with_virial=True):
        from pyrai3md.workflow.optimizer import get_optimizer

        optimizer = get_optimizer('minima', maxiter=2, cell_variables=True)
        optimizer.note_evaluation(state)
        split = optimizer.constraint_split
        self.assertIsNotNone(split, 'the plan declares an angle')
        self.assertIsNotNone(split.get('cell_virial'),
                             'an angle plan must supply the cell term')
        virial = np.asarray(split['cell_virial'], dtype=np.float64)
        if not with_virial:
            # The mutation the bond experiment uses: a split that reports no
            # lattice derivative, which is what the framework did before #341.
            split.pop('cell_virial', None)
        lattice = (np.asarray(state.pmd.system.lattice_vector,
                              dtype=np.float64) * ANGSTROM_TO_BOHR)
        return optimizer._cell_gradient(state, lattice), virial

    def test_the_reaction_cancels_the_models_cell_force_exactly(self):
        """With the term the cell force is zero; without it, it is the derivative.

        Both halves are checked against the prediction this file computes from
        `cv`, so the pair cannot both pass by a shared mistake: the zero comes
        from the optimizer's own subtraction, the non-zero from the model.
        """
        _energy, _gradient, _stress, angle, multiplier, predicted = self._model()
        self.assertGreater(abs(multiplier), 0.0)
        # The scale that matters is the reaction times the condition's derivative,
        # and here that is 3.3e-4 Hartree per Bohr of cell entry: the gate is set
        # far above floating-point noise and far below the number, not at a round
        # figure that would silently reject a legitimate geometry.
        self.assertGreater(float(np.max(np.abs(predicted))), 1.0e-6,
                           msg='this geometry carries no cell derivative at all, '
                               'so the comparison below would be vacuous')
        state = self._state(*self._model()[:3])
        applied, virial = self._cell_force(state, with_virial=True)
        raw, _ = self._cell_force(state, with_virial=False)

        np.testing.assert_allclose(
            applied, np.zeros(6), atol=1.0e-8,
            err_msg='with the condition\'s lattice derivative applied the cell '
                    'force must vanish, because this model\'s stress IS that '
                    'derivative; it came back %s' % (applied.tolist(),))
        np.testing.assert_allclose(
            raw, predicted, rtol=1.0e-10, atol=1.0e-10,
            err_msg='without the term the cell force should be exactly the '
                    'held angle\'s own lattice derivative (multiplier %.6f '
                    'radian^-1, angle %.6f)' % (multiplier, angle))
        np.testing.assert_allclose(
            np.asarray(virial).reshape(-1)[list(CELL_COMPONENT_SLOTS)],
            predicted, rtol=1.0e-10, atol=1.0e-10,
            err_msg='the term the split supplies is not the derivative this '
                    'test computed from cv')
        # The entries that make this a real test: the derivative is non-zero in
        # more than one slot, so a transposition or a single-axis-only term moves
        # at least one of them.
        several = int(np.sum(np.abs(predicted)
                             > 0.01 * float(np.max(np.abs(predicted)))))
        self.assertGreater(several, 2,
                           msg='this geometry should exercise several cell '
                               f'entries at a visible size; it exercises {several}'
                               f' of 6 above 1% of the largest ({predicted.tolist()})')


SLOT_ORDER = list(CELL_COMPONENT_SLOTS)


def _slots_of(lattice):
    """The six entries a variable-cell step can move, in slot order."""
    return np.array([lattice.flat[slot] for slot in SLOT_ORDER],
                    dtype=np.float64)


def _lattice_of(values):
    rows = np.zeros((3, 3), dtype=np.float64)
    for index, slot in enumerate(SLOT_ORDER):
        rows.flat[slot] = values[index]
    return rows


class _SpringBox:
    """Pair springs inside a lattice the run also moves, with one condition held.

    The point of the model is that its endpoint is provable rather than read
    back. Under the ride convention a variable-cell step holds the fractional
    coordinates, so the lattice derivative of the potential is

        dE/dH = sum_i (dE/dx_i) s_i + dE_box/dH,

    and the cell force a run with a held condition steps along is that plus the
    condition's own term, `lambda * sum_i (d theta/dx_i) s_i`. The atomic half of
    the same search stops where `dE/dx_i + lambda d theta/dx_i = 0`, so the two
    ride terms cancel *entry by entry* and the box endpoint is exactly the lattice
    spring's rest shape: `dE_box/dH = 0`, all six slots, whatever the geometry did
    on the way. Drop the condition's term and that cancellation is gone - the box
    then stops `sum_i g_i s_i / k` off its rest value, which is a number this class
    computes from `cv` and the final geometry.
    """

    # Bohr and Hartree throughout; the state speaks angstrom at its boundary.
    BOX_REST = np.array([[9.50, 0.00, 0.00],
                         [0.70, 9.80, 0.00],
                         [0.50, 0.60, 10.20]])
    PERTURB = 0.02                       # every slot starts 2% off its rest
    PAIR_STIFFNESS = 2.0e-1
    SLOT_STIFFNESS = 2.0e-2
    # A spring in the held coordinate itself, resting off the value the starting
    # geometry has. This is what makes the multiplier nonzero, and the held
    # coordinate is the only one that can be priced this way: every other internal
    # coordinate of the model (a bond length, a valence angle) leaves a torsion
    # free, so a softer handle would be relieved by the geometry rather than paid
    # for by the constraint.
    CONDITION_STIFFNESS = 2.0e-1
    CONDITION_OFFSET = 3.0e-2            # radians
    # Bend springs at the valence angles' own values. A torsion is invisible to a
    # distance and to an angle, so without these the chain bends, the terminal
    # atoms come closer, and the held torsion stops being the only coordinate that
    # could relieve anything.
    BEND_STIFFNESS = 5.0e-1
    BENDS = ()

    KIND = 'angle'
    ATOMS = ('H', 'O', 'H')
    FRACTIONAL = np.array([[0.10, 0.05, 0.05],
                           [0.20, 0.11, 0.06],
                           [0.11, 0.22, 0.12]])
    PAIRS = ((0, 1), (1, 2), (0, 2))
    # The 1-based spelling the plan uses. `cv.angle` measures at its middle
    # argument, and atom order here is (H, O, H), so the apex is the oxygen and
    # the pair the reaction spring sits on, (0, 2), is the side opposite it.
    PLAN_ATOMS = (1, 2, 3)

    def start(self):
        return self.PERTURB, self.BOX_REST * (1.0 + self.PERTURB)

    def condition(self, coords, grad=False):
        from pyrai3md.utilities import cv

        order = [index - 1 for index in self.PLAN_ATOMS]
        if self.KIND == 'angle':
            return cv.angle(coords, *order, grad=grad)
        return cv.dihedral(coords, *order, grad=grad)

    def start_geometry(self):
        lattice = self.start()[1]
        return self.FRACTIONAL @ lattice, lattice

    def rest_lengths(self):
        """Each pair rests at the distance the starting geometry has."""
        coords, _lattice = self.start_geometry()
        return np.array([float(np.linalg.norm(coords[second] - coords[first]))
                         for first, second in self.PAIRS])

    def condition_rest(self):
        """The held coordinate's spring rest: the value the plan holds, moved."""
        coords, _lattice = self.start_geometry()
        return float(self.condition(coords)) + self.CONDITION_OFFSET

    def bend_rests(self):
        """Each valence angle rests at the value the starting geometry has."""
        from pyrai3md.utilities import cv

        coords, _lattice = self.start_geometry()
        return [float(cv.angle(coords, first, middle, last))
                for first, middle, last in self.BENDS]

    def parts(self, coords, lattice):
        from pyrai3md.utilities import cv

        rests = self.rest_lengths()
        energy = 0.0
        gradient = np.zeros_like(coords)
        for (first, second), rest in zip(self.PAIRS, rests):
            delta = coords[second] - coords[first]
            length = float(np.linalg.norm(delta))
            axis = delta / length
            stretch = length - rest
            energy += 0.5 * self.PAIR_STIFFNESS * stretch ** 2
            gradient[second] += self.PAIR_STIFFNESS * stretch * axis
            gradient[first] -= self.PAIR_STIFFNESS * stretch * axis
        for (first, middle, last), rest in zip(self.BENDS, self.bend_rests()):
            value, jac = cv.angle(coords, first, middle, last, grad=True)
            bend_stretch = float(value) - rest
            energy += 0.5 * self.BEND_STIFFNESS * bend_stretch ** 2
            reaction = self.BEND_STIFFNESS * bend_stretch
            for position, atom in enumerate((first, middle, last)):
                gradient[atom] += reaction * np.asarray(jac,
                                                        dtype=np.float64)[position]
        held, held_jac = self.condition(coords, grad=True)
        held_stretch = float(held) - self.condition_rest()
        energy += 0.5 * self.CONDITION_STIFFNESS * held_stretch ** 2
        held_reaction = self.CONDITION_STIFFNESS * held_stretch
        gradient += held_reaction * np.asarray(held_jac, dtype=np.float64)
        offset = _slots_of(lattice) - _slots_of(self.BOX_REST)
        energy += 0.5 * self.SLOT_STIFFNESS * float(offset @ offset)
        box_derivative = np.zeros((3, 3))
        for index, slot in enumerate(SLOT_ORDER):
            box_derivative.flat[slot] = self.SLOT_STIFFNESS * offset[index]
        return energy, gradient, box_derivative

    def cell_force(self, coords, lattice):
        """dE/dH at fixed fractional coordinates, the (3, 3) the stress encodes."""
        from pyrai3md.utilities.lattice import frac_to_cart

        inverse = np.linalg.inv(lattice)
        fractional = coords @ inverse
        _energy, gradient, box_derivative = self.parts(coords, lattice)
        return (np.einsum('ia,ib->ab', fractional, gradient)
                + box_derivative)

    def evaluate(self, state):
        from pyrai3md.utilities.lattice import cell_force_to_stress

        coords = (np.asarray(state.coordinates, dtype=np.float64)
                  * ANGSTROM_TO_BOHR)
        lattice = (np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
                   * ANGSTROM_TO_BOHR)
        energy, gradient, _box = self.parts(coords, lattice)
        derivative = self.cell_force(coords, lattice)
        stress = cell_force_to_stress(derivative, lattice)
        state.set_result({
            'energy': [float(energy)],
            'gradient': gradient.reshape(1, len(coords), 3).tolist(),
            'stress_tensor': stress.reshape(1, 3, 3).tolist(),
        })
        return state

    def plan(self):
        units = {'length': 'angstrom', 'angle': 'radian'}
        return SystemConstraintConfig(**{
            'units': units,
            'constraints': [{'id': 'held', 'kind': self.KIND,
                             'bonds': [list(self.PLAN_ATOMS)]}],
        })

    def state(self):
        _perturb, lattice = self.start()
        angstrom = BOHR_TO_ANGSTROM
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'endpoint_%s' % self.KIND
        config.system.atoms = list(self.ATOMS)
        config.system.natom = len(self.ATOMS)
        config.system.mass = [1.008 if element == 'H' else 15.999
                              for element in self.ATOMS]
        coords = self.FRACTIONAL @ lattice
        config.system.coordinates = (coords * angstrom).tolist()
        config.system.nstate = 1
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.pbc = [1, 1, 1]
        config.system.lattice_vector = (lattice * angstrom).tolist()
        from pyrai3md.utilities.lattice import matrix_to_cell6
        config.system.cell = [float(value) for value in
                              matrix_to_cell6(lattice * angstrom)]
        config.optimize.optimize_cell = True
        config.system.constraints = self.plan()
        state = PyRAIState(runtime=RuntimeState(total_steps=300))
        state.pmd = config
        constraint_runtime.bind(state)
        return state

    def relax(self, state, limit=300, virial=True):
        optimizer = get_optimizer('minima', maxiter=limit,
                                  energy_threshold=1e-8,
                                  gradient_threshold=2.0e-6,
                                  gradient_max_threshold=4.0e-6,
                                  displacement_threshold=2.0e-5,
                                  displacement_max_threshold=8.0e-5,
                                  cell_variables=True)
        report = {}
        for _step in range(limit):
            self.evaluate(state)
            optimizer.note_evaluation(state)
            if not virial:
                split = optimizer.constraint_split
                if split is not None:
                    split.pop('cell_virial', None)
            optimizer.run(state)
            constraint_runtime.constrain_state(state)
            converged, report = optimizer.convergence(state)
            if converged:
                return report, True
        return report, False

    def slots(self, state):
        return _slots_of(np.asarray(state.pmd.system.lattice_vector,
                                    dtype=np.float64) * ANGSTROM_TO_BOHR)


class _TorsionSpringBox(_SpringBox):
    """The same instrument with a four-atom chain and a held torsion."""

    KIND = 'dihedral'
    ATOMS = ('H', 'C', 'C', 'H')
    FRACTIONAL = np.array([[0.10, 0.05, 0.05],
                           [0.20, 0.11, 0.06],
                           [0.30, 0.20, 0.10],
                           [0.22, 0.31, 0.18]])
    PAIRS = ((0, 1), (1, 2), (2, 3), (0, 3))
    PLAN_ATOMS = (1, 2, 3, 4)
    BENDS = ((0, 1, 2), (1, 2, 3))


class _EndpointInstrument:
    """The three readings every held condition has to give.

    A mixin rather than a TestCase subclass, so the driver that has no `MODEL`
    is never collected as a case of its own.
    """

    MODEL = None

    def setUp(self):
        self._cwd = os.getcwd()
        self._directory = tempfile.mkdtemp()
        os.chdir(self._directory)
        self._runs = {}

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._directory, ignore_errors=True)

    def _endpoint(self, virial=True):
        """Run the search once per arm and hand back the state it stopped on."""
        if virial not in self._runs:
            box = self.MODEL()
            state = box.state()
            report, converged = box.relax(state, virial=virial)
            self._runs[virial] = (box, state, report, converged)
        return self._runs[virial]

    def _ride(self, box, state):
        """sum_i (dE/dx_i) s_i at the endpoint, in the six movable slots."""
        coords = (np.asarray(state.coordinates, dtype=np.float64)
                  * ANGSTROM_TO_BOHR)
        lattice = (np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
                   * ANGSTROM_TO_BOHR)
        fractional = coords @ np.linalg.inv(lattice)
        gradient = box.parts(coords, lattice)[1]
        ride = np.einsum('ia,ib->ab', fractional, gradient)
        return (np.array([ride.flat[slot] for slot in SLOT_ORDER]),
                np.max(np.abs(fractional)))

    def _allowed(self, box, state, report):
        """How far off the predicted box a converged run is allowed to sit.

        Two terms, both read off the run rather than chosen. The cell residual is
        the force the search stopped carrying, and a lattice spring of stiffness
        `k` leaves it as an offset of `residual / k`. The atomic residual is the
        other half: the cancellation that puts the box at its rest shape cancels
        the ride term *entry by entry*, and what it cancels is the atomic gradient
        at the geometry the run actually reached, weighed by the fractional
        coordinate that carries it onto the box.
        """
        _ride, fractional_max = self._ride(box, state)
        cell_residual = float(report.get('Cell Gradient Max', 0.0))
        atomic_residual = float(report.get('Gradient Max', 0.0))
        return (2.0 * (cell_residual
                       + fractional_max * atomic_residual)
                / box.SLOT_STIFFNESS)

    def test_the_held_condition_carries_a_reaction(self):
        """Without a reaction there is nothing for the term to cancel.

        The witness is the ride term of the potential at the endpoint, computed
        from `cv` and the geometry: it is what the condition's own lattice
        derivative has to cancel, and a run whose potential gradient already
        vanished would put the box at its rest shape with or without the term and
        prove nothing about the term at all.
        """
        box, state, _report, _converged = self._endpoint()
        offset = box.condition_rest() - float(box.condition(
            np.asarray(state.coordinates, dtype=np.float64)
            * ANGSTROM_TO_BOHR))
        self.assertAlmostEqual(abs(offset), box.CONDITION_OFFSET, delta=1.0e-12,
                               msg='the spring that prices the held coordinate '
                                   'does not rest where the plan holds it')
        ride, _fractional_max = self._ride(box, state)
        self.assertGreater(float(np.max(np.abs(ride))), 1.0e-5,
                           msg='the potential has no lattice derivative at this '
                               'endpoint, so the cell term cancels nothing and the '
                               'endpoint reading below is vacuous (%s)'
                               % (ride.tolist(),))
        several = int(np.sum(np.abs(ride)
                             > 0.05 * float(np.max(np.abs(ride)))))
        self.assertGreater(several, 2,
                           msg='a bent condition should price several cell slots; '
                               'this one prices %d of 6 (%s)'
                               % (several, ride.tolist()))

    def test_the_held_box_stops_at_the_lattices_own_rest_shape(self):
        """The prediction: all six slots, not one edge as in the bond case."""
        box, state, report, converged = self._endpoint()
        self.assertTrue(converged, 'the search did not converge: %r'
                        % (report.get('Status'),))
        self.assertLess(float(report.get('Gradient Max', 1.0)), 4.0e-5,
                        msg='the atomic half did not converge, so the box reading '
                            'is not the reading of a stationary geometry')
        slots = box.slots(state)
        rest = _slots_of(box.BOX_REST)
        allowed = self._allowed(box, state, report)
        delta = slots - rest
        self.assertLess(float(np.max(np.abs(delta))), allowed,
                        msg='with the condition\'s lattice derivative applied the '
                            'ride term cancels entry by entry and the box belongs '
                            'at the lattice spring\'s rest shape; it stopped '
                            '%s off it, with %s Bohr allowed'
                            % (delta.tolist(), allowed))

    def test_without_the_term_the_box_stops_ride_over_stiffness(self):
        """The control, predicted slot by slot and not just seen to differ.

        Take the condition's lattice derivative away and the cancellation above
        does not happen: the box stops where the potential's own ride term pushes
        it, at `rest - ride / k`, and the run's endpoint says so entry by entry.
        """
        box, state, report, converged = self._endpoint(virial=False)
        self.assertTrue(converged, 'the control run did not converge')
        ride, _fractional_max = self._ride(box, state)
        predicted = _slots_of(box.BOX_REST) - ride / box.SLOT_STIFFNESS
        slots = box.slots(state)
        delta = slots - _slots_of(box.BOX_REST)
        allowed = self._allowed(box, state, report)
        separation = float(np.max(np.abs(delta)))
        self.assertGreater(separation, 10.0 * allowed,
                           msg='dropping the term has to move the box far past the '
                               'floor a converged run sits on; it moved %.2e '
                               'against a floor of %.2e (ratio %.1f, gate 10)'
                               % (separation, allowed, separation / allowed))
        np.testing.assert_allclose(
            slots, predicted, rtol=0.1, atol=2.0 * allowed,
            err_msg='the box stopped at %s; the potential ride term over the '
                    'lattice stiffness puts it at %s (allowed %.2e)'
                    % (slots.tolist(), predicted.tolist(), 2.0 * allowed))


class TestTheHeldAngleStopsAtAPredictedBox(_EndpointInstrument,
                                           unittest.TestCase):
    """Item (2) of the #355 ledger for a held angle: the box *position*, pinned."""

    MODEL = _SpringBox


class TestTheHeldTorsionStopsAtAPredictedBox(_EndpointInstrument,
                                             unittest.TestCase):
    """The same reading for a held torsion, which no distance can price."""

    MODEL = _TorsionSpringBox


class TestTheAngleOnlyCellExampleIsTheEndToEndCarrier(unittest.TestCase):
    """The shipped example isolates the angle; its endpoint is the unit test.

    The endpoint instrument above prices the angle's box position on a model;
    this example is the real-engine functional carrier. It must keep holding
    exactly one angle (no bond beside it), so a failure in the run names the
    angle path rather than the combined case. `maxit: 6` is deliberate: this is
    a functional test, not a convergence test.
    """

    def test_it_holds_one_angle_and_no_bond_at_a_functional_budget(self):
        import yaml
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, 'examples',
            'optimize_cell_lammps_angle.yaml'))
        with open(path, encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        optimize = data['optimize']
        condition = data['system_constraint']
        self.assertIs(optimize['optimize_cell'], True)
        self.assertEqual(optimize['maxit'], 6)
        self.assertEqual(condition['shake_angle'], [[2, 1, 3]])
        self.assertNotIn('shake_bond', condition)
        for key in ('shake_tol', 'shake_vel', 'shake_id'):
            self.assertEqual(len(condition[key]), 1, key)
        self.assertEqual(condition['fix_atoms'], 4)
