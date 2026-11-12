"""
======================================================================
The curvature batch of a run that relaxes its box.

A saddle search steps on (3N + 6) components, so the matrix it starts from
has to describe all of them: six more rows and columns, differenced through
the cell force rather than through the energy, and the couplings between a
lattice entry and an atom. This file assembles one from a model whose
derivatives are known, and checks the assembly against a finite difference
written here rather than read back from the code under test - which is a
check on the ordering, the units, the divisor and the slot map, and not on
the physics of the cell force (that is priced against an engine in
`tools/cell_stress_fd.py` and `dev/methods/CELL_OPT_DEVELOPMENT.md` section 5.4).

The joint frame gets its own criteria: the basis a periodic search projects
into carries the box's whole-body motions beside the structure's, and the
metric it uses has no mass to offer a lattice entry.

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
from pyrai3md.data.runtime.result import Result
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.lattice import (cell_force_to_stress, cart_to_frac,
                                        stress_to_cell_force)
from pyrai3md.utilities.units import (ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM,
                                      au_pressure_to_bar)
from pyrai3md.workflow.frequency import (periodic_rigid_block,
                                         periodic_vibrational_basis,
                                         vibrational_basis)
from pyrai3md.workflow.hessian import Hessian
from pyrai3md.utilities.lattice import CELL_COMPONENT_SLOTS

SLOTS = list(CELL_COMPONENT_SLOTS)
# The model: a box spring on the xx entry and an independent spring per atom
# about a fixed reference, so the two halves couple only through the ride.
BOX_CURVE = 0.05
BOX_REST = 18.0
ATOM_CURVE = 0.2
EDGE = 10.0                                    # Angstrom
REFERENCE = np.array([[0.3, 0.1, 0.2],
                      [2.1, 0.4, 1.7],
                      [1.2, 2.6, 0.8]]) * ANGSTROM_TO_BOHR
MASSES = [15.999, 1.008, 1.008]


def model(coords_bohr, lattice_bohr):
    """Energy, atomic gradient and cell force of the model, in atomic units.

    The cell force is the derivative at fixed fractional coordinates, which is
    the path a cell step takes, so it carries the box spring and the ride's own
    chain term and nothing else.
    """
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    lattice = np.asarray(lattice_bohr, dtype=np.float64).reshape(3, 3)
    edge = float(lattice[0, 0])
    displacement = coords - REFERENCE
    energy = (0.5 * BOX_CURVE * (edge - BOX_REST) ** 2
              + 0.5 * ATOM_CURVE * float(np.sum(displacement ** 2)))
    gradient = ATOM_CURVE * displacement
    fractional = cart_to_frac(coords, lattice)
    cell = np.zeros((3, 3))
    cell[0, 0] = BOX_CURVE * (edge - BOX_REST)
    # d r_i[b] / d H[a,b] at fixed fractional coordinates is s_i[a], so the
    # chain term is the contraction of the two.
    cell += fractional.T @ gradient
    return energy, gradient, cell


def stress_of(coords_bohr, lattice_bohr):
    """The tensile-positive tensor whose conjugate is the model's cell force."""
    _energy, _gradient, cell = model(coords_bohr, lattice_bohr)
    volume = float(abs(np.linalg.det(lattice_bohr)))
    # M = V sigma^T H^-T  =>  sigma = H M^T / V
    return lattice_bohr @ cell.T / volume


class _Backend:
    """The model, in the shape a backend answers in."""

    def __init__(self):
        self.calls = 0
        self.cells = []

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        del atoms, point_charges, target_states
        self.calls += 1
        rows = np.asarray((cell or {}).get('lattice_vector') or [],
                          dtype=np.float64)
        if rows.shape != (3, 3) or not np.any(rows):
            raise ValueError('a cell task arrived with no lattice')
        self.cells.append(rows.tolist())
        coords = np.asarray(coordinates, dtype=np.float64) * ANGSTROM_TO_BOHR
        lattice = rows * ANGSTROM_TO_BOHR
        energy, gradient, _cell_force = model(coords, lattice)
        # The model is written in Bohr, so its gradient is already Hartree per
        # Bohr, which is the unit a backend reports and the unit `collect`
        # divides by a Bohr step. Converting it again scales the whole atomic
        # block by 0.529 - the same mistake the harmonic harness in
        # `test_hessian_link.py` documents.
        return {'energy': energy,
                'gradient': gradient.reshape(1, -1, 3).tolist(),
                'stress_tensor': stress_of(coords, lattice).reshape(1, 3, 3)
                .tolist()}


class _Calculator:
    def __init__(self, state, backend):
        self.state = state
        self.backend = backend

    def run(self, target_states=None):
        del target_states
        data = self.backend.calculate(
            self.state.atoms, self.state.coordinates,
            cell={'lattice_vector': self.state.pmd.system.lattice_vector})
        self.state.set_result(data)
        return dict(self.state.result)

    def get_state(self):
        return self.state

    def sync(self):
        self.state = self.state

    def initialize(self):
        pass


def _executor(backend):
    def run_child(child_state, runtime_idx=None):
        del runtime_idx
        data = backend.calculate(
            child_state.atoms, child_state.coordinates,
            cell={'lattice_vector': child_state.pmd.system.lattice_vector})
        return Result(data=data)

    return run_child


def _state(edge=EDGE, optimize_cell=True):
    config = PyRAIConfig()
    config.control.jobtype = 'num_hess'
    config.control.driver = 'main'
    config.control.title = 'cell_hessian'
    config.system.atoms = ['O', 'H', 'H']
    config.system.natom = 3
    config.system.mass = list(MASSES)
    config.system.coordinates = (REFERENCE * BOHR_TO_ANGSTROM).tolist()
    config.system.nstate = 1
    config.system.charges = 0
    config.system.multiplicity = 1
    config.system.pbc = [1, 1, 1]
    config.system.lattice_vector = [[edge, 0.0, 0.0], [0.0, edge, 0.0],
                                    [0.0, 0.0, edge]]
    config.system.cell = [edge, edge, edge, 90.0, 90.0, 90.0]
    config.optimize.optimize_cell = bool(optimize_cell)
    state = PyRAIState(runtime=RuntimeState(total_steps=2))
    state.pmd = config
    return state


class _Isolated(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._directory = tempfile.mkdtemp()
        os.chdir(self._directory)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._directory, ignore_errors=True)


class TestTheCellBlockOfTheBatch(_Isolated):
    """Six more rows and columns, differenced through the cell force."""

    def _assemble(self, step=0.005):
        backend = _Backend()
        state = _state()
        workflow = Hessian(calculator=_Calculator(state, backend),
                           step_size=step, source='numerical', driver='main',
                           analysis=False)
        workflow.local_executor = _executor(backend)
        batch = workflow.generate_batch(state)
        results = []
        for task in batch:
            # An atomic task strains nothing, so it is evaluated in the box the
            # state carries; a cell task brings its own.
            rows = task.get('lattice') or state.pmd.system.lattice_vector
            results.append(Result(data=backend.calculate(
                state.atoms, task['coordinates'],
                cell={'lattice_vector': rows})))
        return workflow, state, batch, results, backend

    def test_the_batch_strains_the_box_beside_displacing_the_atoms(self):
        _workflow, state, batch, _results, _backend = self._assemble()
        atomic = [task for task in batch if 'cell_slot' not in task]
        cells = [task for task in batch if 'cell_slot' in task]
        self.assertEqual(len(atomic), 2 * 9, 'one pair per atomic component')
        self.assertEqual(len(cells), 2 * len(SLOTS),
                         'one pair per lattice entry the run moves')
        self.assertEqual(sorted(task['cell_slot'] for task in cells),
                         sorted(SLOTS * 2))
        for task in cells:
            self.assertIn('lattice', task,
                          'a strain task that carries no lattice strains nothing')
        # The atoms ride the strain: a task whose coordinates did not move with
        # the box would difference a path whose conjugate is not the stress.
        reference = np.asarray(state.coordinates, dtype=np.float64)
        base = np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
        for task in cells:
            moved = np.asarray(task['coordinates'], dtype=np.float64)
            strained = np.asarray(task['lattice'], dtype=np.float64)
            expected = cart_to_frac(reference * ANGSTROM_TO_BOHR,
                                    base * ANGSTROM_TO_BOHR)
            expected = expected @ (strained * ANGSTROM_TO_BOHR) * BOHR_TO_ANGSTROM
            np.testing.assert_allclose(moved, expected, atol=1.0e-12,
                                       err_msg='the atoms did not ride the box')

    def test_the_assembled_matrix_is_an_independent_difference_of_the_model(self):
        workflow, state, _batch, results, _backend = self._assemble()
        got = workflow.collect(state, results)
        self.assertIsNotNone(got, 'the batch returned no matrix')
        size = 3 * 3 + len(SLOTS)
        self.assertEqual(got.shape, (size, size))

        step = workflow.step_size * ANGSTROM_TO_BOHR
        coords = REFERENCE.copy()
        lattice = np.eye(3) * (EDGE * ANGSTROM_TO_BOHR)
        expected = np.zeros((size, size))
        # The atomic block: the gradient's own derivative at a fixed box.
        for index in range(9):
            plus = coords.reshape(-1).copy()
            minus = coords.reshape(-1).copy()
            plus[index] += step
            minus[index] -= step
            difference = (model(plus.reshape(-1, 3), lattice)[1].reshape(-1)
                          - model(minus.reshape(-1, 3), lattice)[1].reshape(-1))
            expected[index, :9] = difference / (2.0 * step)
        # The atom rows' coupling to the box: how the cell force answers to an
        # atomic component, at the box the atomic tasks all share. It is the same
        # mixed second derivative the cell rows carry from the other side, and the
        # assembly symmetrizes the two readings.
        for index in range(9):
            plus = coords.reshape(-1).copy()
            minus = coords.reshape(-1).copy()
            plus[index] += step
            minus[index] -= step
            forces = []
            for candidate in (plus, minus):
                forces.append(model(candidate.reshape(-1, 3),
                                    lattice)[2].reshape(-1)[SLOTS])
            expected[index, 9:] = (forces[0] - forces[1]) / (2.0 * step)
        # The cell rows: the ride path, which is what a cell step takes.
        fractional = cart_to_frac(coords, lattice)
        for position, slot in enumerate(SLOTS):
            delta = np.zeros((3, 3))
            delta[slot // 3, slot % 3] = step
            rows = []
            for sign in (1, -1):
                strained = lattice + sign * delta
                ridden = fractional @ strained
                _energy, gradient, cell_force = model(ridden, strained)
                rows.append((gradient.reshape(-1),
                             cell_force.reshape(-1)[SLOTS]))
            expected[9 + position, :9] = (rows[0][0] - rows[1][0]) / (2.0 * step)
            expected[9 + position, 9:] = (rows[0][1] - rows[1][1]) / (2.0 * step)

        mobile = list(range(9)) + [9 + index for index in range(len(SLOTS))]
        block = got[np.ix_(mobile, mobile)]
        wanted = expected[np.ix_(mobile, mobile)]
        self.assertLess(float(np.abs(block - wanted).max()), 1.0e-6,
                        'the assembled cell block is not the difference of the '
                        'model it was built from')
        self.assertLess(float(np.abs(block - block.T).max()), 1.0e-6,
                        'the joint block is not symmetric')

    def test_the_cell_rows_come_from_the_stress_and_not_from_the_energy(self):
        """A tensor the batch cannot read is a named failure, not a zero row."""
        backend = _Backend()
        state = _state()
        workflow = Hessian(calculator=_Calculator(state, backend),
                           step_size=0.005, source='numerical', driver='main',
                           analysis=False)
        batch = workflow.generate_batch(state)
        results = []
        for task in batch:
            rows = task.get('lattice') or state.pmd.system.lattice_vector
            data = backend.calculate(state.atoms, task['coordinates'],
                                     cell={'lattice_vector': rows})
            if 'cell_slot' in task:
                data = {'energy': data['energy'], 'gradient': data['gradient']}
            results.append(Result(data=data))
        self.assertIsNone(workflow.collect(state, results),
                          'a strain task with no stress was read as a zero row')

    def test_a_backend_block_is_joined_with_strained_cell_rows(self):
        """The pairing that used to be refused is measured instead.

        The atomic corner is the backend's own matrix and nothing else; the six
        lattice rows are the framework's twelve strained evaluations of this box,
        and they must be the same numbers the `driver='main'` route differences at
        the same step. That is the independent expectation: the cell half is read
        against a route that does not share this code path, not reproduced from the
        assembly under test.
        """
        step = 0.005
        main, state, _batch, results, backend = self._assemble(step)
        reference = main.collect(state, results)
        self.assertIsNotNone(reference, 'the main route stopped differencing')
        size = reference.shape[0] - len(SLOTS)

        atomic = np.arange(size * size, dtype=np.float64).reshape(size, size)
        hybrid = Hessian(calculator=_Calculator(state, backend),
                         step_size=step, source='analytic', driver='skill',
                         analysis=False)
        self.assertTrue(hybrid.hybrid_cell_block,
                        'a backend matrix beside a variable box is the hybrid case')
        hybrid.local_executor = _executor(backend)
        joined = hybrid._strain_cell_block_onto(state, atomic)
        self.assertEqual(joined.shape, reference.shape,
                         'the joined matrix lost or gained rows')
        np.testing.assert_allclose(joined[:size, :size],
                                   0.5 * (atomic + atomic.T), atol=0.0,
                                   err_msg='the backend block was altered')
        np.testing.assert_allclose(joined[size:, :], reference[size:, :],
                                   atol=1e-12,
                                   err_msg='the strained cell rows are not what the '
                                           'main route differences at this step')
        np.testing.assert_allclose(joined[:size, size:], reference[:size, size:],
                                   atol=1e-12,
                                   err_msg='the coupling columns are not mirrored '
                                           'from the same strain rows')
        self.assertGreater(np.max(np.abs(joined[size:, :])), 0.0,
                           'an all-zero cell block is the fiction this route '
                           'replaced, and a criterion that cannot see it is '
                           'indifferent to it')

    def test_a_backend_block_that_misses_atoms_is_refused_by_name(self):
        """A partial matrix would be padded with rows no evaluation produced."""
        state = _state()
        backend = _Backend()
        hybrid = Hessian(calculator=_Calculator(state, backend),
                         source='analytic', driver='skill', analysis=False)
        size = 3 * len(state.atoms)
        with self.assertRaises(ValueError) as caught:
            hybrid._strain_cell_block_onto(state, np.zeros((size - 3, size - 3)))
        message = str(caught.exception)
        self.assertIn('cover every atom', message)
        self.assertIn('%d atoms' % (size // 3), message)

    def test_a_vibrational_analysis_of_a_box_is_refused_by_name(self):
        state = _state()
        with self.assertRaises(ValueError) as caught:
            Hessian(calculator=_Calculator(state, _Backend()),
                    source='numerical', driver='main', analysis=True)
        self.assertIn('optimize_cell', str(caught.exception))

    def test_a_fixed_cell_batch_is_untouched(self):
        """No box, no cell rows: the matrix is the (3N, 3N) it always was."""
        backend = _Backend()
        state = _state(optimize_cell=False)
        workflow = Hessian(calculator=_Calculator(state, backend),
                           step_size=0.005, source='numerical', driver='main',
                           analysis=False)
        batch = workflow.generate_batch(state)
        self.assertEqual(len(batch), 2 * 9)
        self.assertFalse(any('cell_slot' in task for task in batch))


class TestTheJointFrame(unittest.TestCase):
    """The basis a periodic curvature search projects into."""

    COORDS = REFERENCE * BOHR_TO_ANGSTROM
    LATTICE = np.eye(3) * (EDGE * ANGSTROM_TO_BOHR)

    def test_no_lattice_gives_the_atomic_answer_element_for_element(self):
        """The degenerate case is the criterion that the two did not drift."""
        masses = np.asarray(MASSES, dtype=np.float64)
        plain = vibrational_basis(self.COORDS, masses)
        joint = periodic_vibrational_basis(self.COORDS, masses, None)
        for index in range(4):
            np.testing.assert_allclose(
                np.asarray(joint[index], dtype=np.float64),
                np.asarray(plain[index], dtype=np.float64), atol=0.0,
                err_msg='a call with no lattice is not the atomic reading')

    def test_the_metric_weighs_atoms_and_leaves_the_box_alone(self):
        masses = np.asarray(MASSES, dtype=np.float64)
        _basis, weights, _center, _rank = periodic_vibrational_basis(
            self.COORDS, masses, self.LATTICE)
        self.assertEqual(weights.size, 9 + len(SLOTS))
        np.testing.assert_allclose(weights[:9], np.repeat(np.sqrt(masses), 3))
        np.testing.assert_allclose(
            weights[9:], np.ones(len(SLOTS)), atol=0.0,
            err_msg='a lattice entry has no mass, so its weight is one; a '
                    'number here other than one is an invented mass, and an '
                    'invented mass drifts with the geometry')

    def test_a_rotation_carries_the_box_with_it(self):
        """Only the atoms turning is a deformation here, not a null direction."""
        block, _center = periodic_rigid_block(self.COORDS,
                                              np.asarray(MASSES), self.LATTICE)
        self.assertEqual(block.shape, (9 + len(SLOTS), 6))
        edge = EDGE * ANGSTROM_TO_BOHR
        for axis in range(3):
            cell_part = block[9:, 3 + axis]
            self.assertGreater(
                float(np.max(np.abs(cell_part))), 0.0,
                'the rotation about axis %d left the lattice half at zero, so '
                'the frame would call a deformation a null direction' % axis)
            self.assertAlmostEqual(
                float(np.max(np.abs(cell_part))), edge, places=9,
                msg='the lattice half of a rotation is one cell vector long')
        for axis in range(3):
            self.assertEqual(float(np.max(np.abs(block[9:, axis]))), 0.0,
                             'a translation moved the box')

    def test_a_periodic_structure_loses_six_and_reports_how_many(self):
        masses = np.asarray(MASSES, dtype=np.float64)
        basis, weights, _center, rank = periodic_vibrational_basis(
            self.COORDS, masses, self.LATTICE)
        self.assertEqual(rank, 6)
        self.assertEqual(basis.shape, (9 + len(SLOTS), 9 + len(SLOTS) - 6))
        self.assertEqual(weights.size, 9 + len(SLOTS))
        # Orthonormal in the metric the frame declares.
        product = basis.T @ basis
        np.testing.assert_allclose(product, np.eye(product.shape[0]),
                                   atol=1.0e-10)


class TestTheCellForceAndItsPressureReading(unittest.TestCase):
    """The inverse of the cell force, and what a cell threshold means as a pressure.

    A threshold on `dE/dB` is in Hartree per Bohr, so the same number asks
    different things of boxes of different size: on the eighth round's water box
    it is a pressure of some 130 bar, and on a box twice as long the very same
    reading is four times softer. The stress-side criteria below are what a
    scale-invariant cell criterion is calibrated against, measured on the box
    the round actually ran rather than on a made-up cell.
    """

    # tmp_20261016_165643/examples/optimize_cell_lammps: an orthorhombic TIP3P
    # box with no tilt, the row lengths the data file declares and the volume
    # its first recorded step printed.
    BOX_ROWS = (25.386649, 23.267958, 26.615364)          # Angstrom
    BOX_VOLUME = 15721.575                               # Angstrom^3
    CELL_GMAX = 9.807e-4                                 # Hartree/Bohr, last step

    @staticmethod
    def _lattice(rows_angstrom, shear=0.0):
        """A row-vector lattice in Bohr, optionally sheared in xy and xz."""
        base = np.diag(np.asarray(rows_angstrom, dtype=np.float64)
                       * ANGSTROM_TO_BOHR)
        if shear:
            base[1, 0] = shear * rows_angstrom[1] * ANGSTROM_TO_BOHR
            base[2, 0] = 0.5 * shear * rows_angstrom[2] * ANGSTROM_TO_BOHR
        return base

    def test_the_inverse_returns_the_stress_the_force_was_made_from(self):
        rng = np.random.default_rng(20261016)
        for shear in (0.0, 0.3, -0.2):
            lattice = self._lattice(self.BOX_ROWS, shear)
            half = rng.normal(size=(3, 3))
            stress = half + half.T                    # tensile-positive, symmetric
            force = stress_to_cell_force(stress, lattice)
            self.assertTrue(
                np.allclose(cell_force_to_stress(force, lattice), stress,
                            rtol=0.0, atol=1e-13 * max(1.0, np.abs(stress).max())),
                'the round trip moved the stress on a sheared box')

    def test_a_transposed_formula_is_not_the_inverse(self):
        # The criterion above has to fail for a wrong-but-plausible assembly.
        lattice = self._lattice(self.BOX_ROWS, 0.3)
        half = np.array([[1.0, 0.4, 0.2], [0.4, 2.0, 0.6], [0.2, 0.6, 3.0]])
        force = stress_to_cell_force(half, lattice)
        wrong = force.T @ lattice / abs(np.linalg.det(lattice))
        self.assertGreater(np.max(np.abs(wrong - half)), 1e-3,
                           'the transpose form would have passed a symmetric '
                           'box and is therefore not what is being pinned')

    def test_a_cell_force_reads_as_a_pressure_on_the_measured_box(self):
        lattice = self._lattice(self.BOX_ROWS)
        volume = abs(np.linalg.det(lattice))
        self.assertAlmostEqual(volume, self.BOX_VOLUME * ANGSTROM_TO_BOHR ** 3,
                               delta=1.0)
        force = np.zeros((3, 3))
        force[0, 0] = self.CELL_GMAX
        bar = au_pressure_to_bar(cell_force_to_stress(force, lattice)[0, 0])
        self.assertAlmostEqual(bar, 130.5, delta=0.5)

    def test_the_same_force_is_softer_on_a_longer_box(self):
        small = self._lattice(self.BOX_ROWS)
        force = np.zeros((3, 3))
        force[0, 0] = self.CELL_GMAX
        base = au_pressure_to_bar(cell_force_to_stress(force, small)[0, 0])
        for scale, expected in ((2.0, 0.25), (3.0, 1.0 / 9.0)):
            big = self._lattice([row * scale for row in self.BOX_ROWS])
            got = au_pressure_to_bar(cell_force_to_stress(force, big)[0, 0])
            self.assertAlmostEqual(got / base, expected, delta=1e-12)

    def test_a_lattice_without_volume_raises_by_name(self):
        force = np.zeros((3, 3))
        with self.assertRaises(ValueError) as caught:
            cell_force_to_stress(force[0, 0:1], np.zeros((3, 3)))
        self.assertIn('volume', str(caught.exception))

    def test_the_six_relaxed_components_alone_return_the_stress(self):
        # The optimizer keeps six numbers, not nine: the entries above the
        # diagonal are held, yet they are not zero, because one shear of the
        # stress loads force on both B[1][0] and B[0][1]. The six-component
        # spelling therefore solves the map the slots define.
        rng = np.random.default_rng(20261017)
        for shear in (0.0, 0.25, -0.4):
            lattice = self._lattice(self.BOX_ROWS, shear)
            half = rng.normal(size=(3, 3))
            stress = half + half.T
            slots = stress_to_cell_force(stress, lattice).reshape(-1)[SLOTS]
            back = cell_force_to_stress(slots, lattice)
            self.assertTrue(
                np.allclose(back, stress, rtol=0.0,
                            atol=1e-13 * max(1.0, np.abs(stress).max())),
                'six slots did not return the stress on a sheared box')

    def test_zero_filling_the_held_entries_is_not_the_inverse(self):
        # The plausible shortcut (read the six slots as a full matrix whose other
        # entries are zero, then apply the closed form) is exact on an
        # orthorhombic box and wrong on a tilted one. Measured here so the
        # six-component solve cannot be traded for the shortcut later.
        lattice = self._lattice(self.BOX_ROWS, 0.25)
        stress = np.array([[1.0, 0.4, 0.2], [0.4, 2.0, 0.6], [0.2, 0.6, 3.0]])
        slots = stress_to_cell_force(stress, lattice).reshape(-1)[SLOTS]
        padded = np.zeros((3, 3), dtype=np.float64)
        padded.reshape(-1)[SLOTS] = slots
        shortcut = cell_force_to_stress(padded, lattice)
        self.assertGreater(np.max(np.abs(shortcut - stress)), 1e-2,
                           'the tilted case was not discriminating: the two '
                           'forms would be interchangeable')
        self.assertLess(np.max(np.abs(cell_force_to_stress(slots, lattice)
                                      - stress)), 1e-13)

    def test_a_shape_that_is_neither_nine_nor_six_is_refused(self):
        lattice = self._lattice(self.BOX_ROWS)
        with self.assertRaises(ValueError) as caught:
            cell_force_to_stress(np.zeros(7), lattice)
        self.assertIn('relaxed components', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
