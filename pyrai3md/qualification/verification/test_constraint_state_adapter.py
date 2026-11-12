"""
======================================================================
Tests for the state adapter, exercised against a stub.

The stub carries the same attribute names as PyRAIState and the same awkward
behaviours, so the merge round does not meet them for the first time in a live
run: `mass_au` returning None instead of raising, a lattice whose default is
three zero rows, a six-parameter `cell` whose default is a 99 Angstrom
sentinel, and `current_state` raising outright for a multi-center run.

That last one is the reason the adapter is written at all: the constraint core
reads the global assembled gradient, which `active_gradient()` already provides
for C=1 and C>1 alike, so a property that raises must simply never be touched.
The multi-center stub here raises on read, and the test passes only if nothing
reaches for it.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import unittest

import numpy as np

from pyrai3md.data import constraint as records
from pyrai3md import schema
from pyrai3md.workflow.constraints import state_adapter
from pyrai3md.utilities.units import AMU_TO_AU, ANGSTROM_TO_BOHR

COORDS = np.array([[0.00, 0.00, 0.00],
                   [1.90, 0.30, 0.10],
                   [0.40, 2.00, -0.20]])
MASS_AMU = np.array([16.0, 1.0, 1.0])
MASS_AU = MASS_AMU * AMU_TO_AU


class _System(object):
    """The pmd.system surface the adapter reads."""

    def __init__(self, natom=3, lattice=None, pbc=None, image=None,
                 cell=None):
        self.natom = natom
        self.lattice_vector = lattice if lattice is not None else \
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.pbc = pbc if pbc is not None else [0, 0, 0]
        self.image = image if image is not None else []
        self.cell = cell if cell is not None else [99, 99, 99, 90, 90, 90]


class _Pmd(object):
    def __init__(self, system):
        self.system = system


class _Runtime(object):
    def __init__(self, step=7, time=144.0):
        self.current_step = step
        self.simulation_time = time


class _StubState(object):
    """Attribute-compatible with the read surface of PyRAIState."""

    def __init__(self, system=None, coordinates=COORDS, velocities=None,
                 masses=MASS_AU, atoms=('O', 'H', 'H'), multi_center=False):
        self.pmd = _Pmd(system or _System())
        self._atoms = list(atoms)
        self._coordinates = coordinates
        self._velocities = velocities
        self._masses = masses
        self.runtime = _Runtime()
        self.timestep = 20.67
        self._multi_center = multi_center

    @property
    def atoms(self):
        return self._atoms

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def velocities(self):
        return self._velocities

    @property
    def mass_au(self):
        if self._masses is None:
            return None
        array = np.asarray(self._masses, dtype=np.float64)
        if array.size != self.pmd.system.natom or np.any(array <= 0.0):
            return None
        return array

    @property
    def current_state(self):
        if self._multi_center:
            raise RuntimeError(
                'PyRAIState.current_state is undefined for a multi-center run')
        return 0

    @property
    def nstate(self):
        if self._multi_center:
            raise RuntimeError(
                'PyRAIState.nstate is undefined for a multi-center run')
        return 1


def _plan(mapping=None):
    return schema.build_plan(mapping if mapping is not None else
                             {'cvs': [{'name': 'r', 'kind': 'bond',
                                        'atoms': [1, 2]}],
                              'biases': [{'cv': 'r', 'target': 1.0,
                                          'force_constant': 0.02}]},
                             atoms=['O', 'H', 'H'], natom=3)


class TestHappyPath(unittest.TestCase):

    def setUp(self):
        self.view, self.context = state_adapter.read_state(
            _StubState(velocities=np.full((3, 3), 0.1)), purpose='md',
            evaluation_id='e1', trajectory_id='run-a')

    def test_coordinates_are_converted_to_bohr_once(self):
        np.testing.assert_allclose(self.view.coordinates,
                                   COORDS * ANGSTROM_TO_BOHR, atol=1.0e-14)
        np.testing.assert_allclose(self.view.coordinates_angstrom, COORDS,
                                   atol=0.0)

    def test_the_round_trip_back_to_angstrom_is_exact(self):
        back = state_adapter.to_angstrom(self.view.coordinates)
        np.testing.assert_allclose(back, COORDS, atol=1.0e-14)

    def test_clock_and_identity_come_from_the_state(self):
        self.assertEqual(self.context.step, 7)
        self.assertAlmostEqual(self.context.time, 144.0, places=9)
        self.assertAlmostEqual(self.context.dt, 20.67, places=9)
        self.assertEqual(self.context.trajectory_id, 'run-a')
        self.assertEqual(self.context.evaluation_id, 'e1')

    def test_no_lattice_means_no_cell(self):
        self.assertIsNone(self.view.cell)
        self.assertFalse(self.view.periodic)

    def test_masses_are_carried_through(self):
        np.testing.assert_allclose(self.view.masses, MASS_AU, atol=1.0e-9)

    def test_velocities_keep_their_shape(self):
        self.assertEqual(self.view.velocities.shape, (3, 3))

    def test_velocities_may_be_absent(self):
        view, _ = state_adapter.read_state(_StubState(velocities=None))
        self.assertIsNone(view.velocities)


class TestMultiCenterState(unittest.TestCase):

    def test_a_multi_center_state_reads_without_touching_the_raising_accessors(
            self):
        """The core works on the global gradient, so it must not ask for them."""
        view, context = state_adapter.read_state(
            _StubState(multi_center=True, velocities=np.zeros((3, 3))),
            purpose='md')
        self.assertEqual(view.natom, 3)
        self.assertEqual(context.step, 7)


class TestFailures(unittest.TestCase):

    def test_missing_masses_are_refused_when_the_plan_needs_them(self):
        view, _ = state_adapter.read_state(_StubState(masses=None))
        self.assertIsNone(view.masses)
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.require_usable(view, _plan(
                {'constraints': [{'id': 'x', 'bonds': [[1, 2]]}]}))

    def test_a_plain_bias_still_works_without_masses(self):
        """An empty-of-mass requirement must not be invented."""
        view, _ = state_adapter.read_state(_StubState(masses=None))
        self.assertIs(state_adapter.require_usable(view, _plan()), view)

    def test_a_non_positive_mass_reads_as_no_masses(self):
        view, _ = state_adapter.read_state(
            _StubState(masses=np.array([16.0, 0.0, 1.0])))
        self.assertIsNone(view.masses)

    def test_periodic_pbc_is_refused_for_a_plan_with_actions(self):
        view, _ = state_adapter.read_state(
            _StubState(system=_System(lattice=[[10, 0, 0], [0, 10, 0],
                                              [0, 0, 10]], pbc=[1, 1, 1])))
        self.assertTrue(view.periodic)
        # The stub structure has no perceived bond between atoms 1 and 2, so the
        # variable is refused as an image question, not as a kind: the same plan
        # runs the moment the graph connects its atoms.
        with self.assertRaises(state_adapter.images.ImageError):
            state_adapter.require_usable(view, _plan())

    def test_crossed_images_count_as_periodic(self):
        view, _ = state_adapter.read_state(
            _StubState(system=_System(lattice=[[10, 0, 0], [0, 10, 0],
                                               [0, 0, 10]],
                                      image=[[0, 0, 0], [1, 0, 0],
                                             [0, 0, 0]])))
        self.assertTrue(view.periodic)

    def test_pbc_without_a_lattice_is_a_contradiction(self):
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(_StubState(system=_System(pbc=[1, 0, 0])))

    def test_the_natom_must_match_the_coordinates(self):
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(
                _StubState(system=_System(natom=5), coordinates=COORDS))

    def test_a_short_atom_list_is_refused(self):
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(_StubState(atoms=('O',)))

    def test_malformed_velocities_are_refused(self):
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(
                _StubState(velocities=np.zeros((2, 3))))

    def test_non_finite_coordinates_are_refused(self):
        broken = COORDS.copy()
        broken[1, 0] = np.nan
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(_StubState(coordinates=broken))

    def test_a_plan_for_another_structure_is_refused(self):
        """A plan naming atom 10 against a 3-atom state is a stale plan."""
        view, _ = state_adapter.read_state(_StubState())
        stale = records.ConstraintPlan(
            cvs={'r': records.CVRecord('r', 'bond', [0, 9])},
            actions=[records.ConstraintSpec('b', 'bias', 'harmonic', cv='r',
                                            target=1.0,
                                            parameters={'force_constant':
                                                        0.02})])
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.require_usable(view, stale)

    def test_a_missing_state_is_refused(self):
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(None)

    def test_a_state_without_pmd_system_is_refused(self):
        class _Bare(object):
            atoms = ['O']
            coordinates = [[0.0, 0.0, 0.0]]
            velocities = None
            mass_au = None
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(_Bare())


class TestEmptyPlanBypassesEverything(unittest.TestCase):
    """DC-1: with nothing declared, an unusable state is still usable."""

    def test_a_periodic_state_passes_for_an_empty_plan(self):
        view, _ = state_adapter.read_state(
            _StubState(system=_System(lattice=[[10, 0, 0], [0, 10, 0],
                                               [0, 0, 10]], pbc=[1, 1, 1])))
        self.assertIs(state_adapter.require_usable(
            view, records.ConstraintPlan.empty()), view)

    def test_missing_masses_pass_for_an_empty_plan(self):
        view, _ = state_adapter.read_state(_StubState(masses=None))
        self.assertIs(state_adapter.require_usable(
            view, records.ConstraintPlan.empty()), view)


class TestLatticeReading(unittest.TestCase):

    def test_a_real_lattice_is_read_and_converted(self):
        view, _ = state_adapter.read_state(
            _StubState(system=_System(lattice=[[10, 0, 0], [0.5, 10, 0],
                                               [0.1, 0.2, 10]],
                                      pbc=[1, 1, 1])))
        self.assertIsNotNone(view.cell)
        np.testing.assert_allclose(view.cell,
                                   np.array([[10, 0, 0], [0.5, 10, 0],
                                             [0.1, 0.2, 10]])
                                   * ANGSTROM_TO_BOHR, atol=1.0e-12)

    def test_the_nine_nines_cell_default_never_becomes_a_lattice(self):
        """A molecule must not inherit a 99 Angstrom box from a sentinel."""
        view, _ = state_adapter.read_state(_StubState())
        self.assertIsNone(view.cell)
        self.assertFalse(view.periodic)

    def test_a_singular_lattice_is_refused(self):
        with self.assertRaises(state_adapter.StateAdapterError):
            state_adapter.read_state(
                _StubState(system=_System(
                    lattice=[[1, 0, 0], [2, 0, 0], [0, 0, 1]],
                    pbc=[1, 1, 1])))


class TestMassesOrRaise(unittest.TestCase):

    def test_the_helper_returns_what_it_demanded(self):
        view, _ = state_adapter.read_state(_StubState())
        np.testing.assert_allclose(state_adapter.masses_or_raise(view, 'x'),
                                   MASS_AU, atol=1.0e-9)

    def test_the_helper_names_the_action_when_it_cannot(self):
        view, _ = state_adapter.read_state(_StubState(masses=None))
        with self.assertRaises(state_adapter.StateAdapterError) as caught:
            state_adapter.masses_or_raise(view, 'xh')
        self.assertIn('xh', str(caught.exception))


class TestNoFrameworkImports(unittest.TestCase):
    """The adapter's own claim, checked on the import graph.

    ``test_import_discipline`` guards the whole lab against the framework files
    under edit; this guard is narrower and belongs to the adapter alone. The
    adapter takes a state object as an argument, so importing the class that
    supplies that object would defeat the reason it exists.

    A substring scan of the text is not the check: the module docstring names
    those packages on purpose, to say what it does not import.
    """

    # Read by the adapter as attributes or by a live module, never imported.
    #
    # The shadow-build form of this rule forbade the whole of `pyrai3md.data`,
    # `pyrai3md.schema` and `pyrai3md.interface`; the modules moved into those
    # packages at the merge, so that version now indicts the move. What still
    # matters is narrower and is the reason the adapter exists at all: it must
    # not import the state class it is handed, the config model it does not own,
    # or the calculator that calls it. `pyrai3md.data.constraint` is a record
    # module and is imported on purpose.
    NEVER = ('pyrai3md.data.state', 'pyrai3md.schema.config',
             'pyrai3md.interface.parser', 'pyrai3md.workflow.calculator')

    def _imported_names(self):
        path = state_adapter.__file__
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return names

    def _offenders(self, names):
        return [name for name in names
                if name in self.NEVER
                or name.startswith(tuple(part + '.' for part in self.NEVER))]

    def test_the_adapter_reaches_no_framework_module(self):
        self.assertEqual(self._offenders(self._imported_names()), [])

    def test_the_guard_would_catch_an_import(self):
        """Negative control: the predicate flags a real violation.

        Asserting only that the adapter is clean would pass a guard whose
        predicate had been written wrong; these names stand for the imports a
        careless rewrite of this module would actually add.
        """
        self.assertEqual(
            self._offenders(['numpy',
                             'pyrai3md.data.constraint',
                             'pyrai3md.data.state',
                             'pyrai3md.workflow.calculator']),
            ['pyrai3md.data.state', 'pyrai3md.workflow.calculator'],
            'the record module the adapter does import must not be flagged, '
            'the state class it must not import must be')


if __name__ == '__main__':
    unittest.main()
