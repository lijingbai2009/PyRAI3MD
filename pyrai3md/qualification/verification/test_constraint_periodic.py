"""
======================================================================
The constraint core inside a periodic cell.

`dev/methods/CONSTRAINT_DEVELOPMENT.md` Appendix B settles the order: the bond / angle / torsion functions
come first (M0), the wall guards come after (M3), and both are judged on the
framework's own arithmetic rather than on a fixture's number. What this file
pins is the seam where those meet a run:

- the periodic guard is per capability, not per cell. A wall needs no bond graph
  and is let through with four geometric checks; a constraint is measured on the
  unfolded geometry; a collective variable over atoms the graph does not connect
  is still refused, and says which one it refused and why;
- the half-box gate arrives before the solver, as an input refusal with the axis
  and the difference, because a constrained distance at half the shortest
  interplanar spacing sits where the nearest image ties;
- the solve runs unfolded and the write-back folds, so the crossing counts that
  go out with the coordinates are the ones those coordinates imply.

Units: the stub carries Angstrom, the core reads Bohr, and the assertions below
convert at the same boundary the adapter does.

Sep 30 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md import schema
from pyrai3md.data.geometry import apply_pbc
from pyrai3md.utilities.lattice import unwrap_coordinates
from pyrai3md.utilities.units import AMU_TO_AU, ANGSTROM_TO_BOHR
from pyrai3md.workflow.constraints import images
from pyrai3md.workflow.constraints import runtime as constraint_runtime
from pyrai3md.workflow.constraints import state_adapter
from pyrai3md.workflow.constraints import wall

BOX = 20.0
LATTICE = [[BOX, 0.0, 0.0], [0.0, BOX, 0.0], [0.0, 0.0, BOX]]
# A water whole inside the cell, 0.96 Angstrom O-H bonds, at the cell center.
WATER = np.array([[10.00, 10.00, 10.00],
                  [10.96, 10.00, 10.00],
                  [9.76, 10.73, 10.00]])
MASS_AMU = np.array([16.0, 1.0, 1.0])
# The same molecule moved so its bound hydrogen sits just inside the far x face:
# the stored pair straddles the cell, which is the configuration a wrapped
# coordinate cannot measure and an unfolded one can.
TORN_SHIFT = np.array([BOX / 2.0 - 0.48, 0.0, 0.0])


class _System(object):
    def __init__(self, natom=3, lattice=None, pbc=None, image=None):
        self.natom = natom
        self.lattice_vector = lattice or [[0.0] * 3, [0.0] * 3, [0.0] * 3]
        self.pbc = pbc or [0, 0, 0]
        self.image = image or []
        self.cell = [99, 99, 99, 90, 90, 90]


class _Pmd(object):
    def __init__(self, system):
        self.system = system


class _Runtime(object):
    def __init__(self, step=0):
        self.current_step = step
        self.simulation_time = 0.0


class _State(object):
    """The read surface of PyRAIState, with a cell this time.

    Atoms and masses default to the single water the module constants describe;
    a test that needs more than one molecule passes them, because the thing
    under test is which components the bond graph connects.
    """

    def __init__(self, coordinates=WATER, pbc=(1, 1, 1), lattice=LATTICE,
                 velocities=None, image=None, step=0, atoms=None,
                 masses=None):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        atoms = list(atoms) if atoms is not None else ['O', 'H', 'H']
        masses = (np.asarray(masses, dtype=np.float64) if masses is not None
                  else MASS_AMU)
        self.pmd = _Pmd(_System(natom=len(atoms), lattice=lattice,
                                pbc=list(pbc), image=image))
        self.atoms = atoms
        self.coordinates = coordinates
        self.velocities = velocities
        self.mass_au = masses * AMU_TO_AU
        self.runtime = _Runtime(step)
        self.timestep = 20.67


def _view(plan_mapping=None, **kwargs):
    """One state read, guarded by the same entry a run uses."""
    state = _State(**kwargs)
    view, _context = state_adapter.read_state(state, purpose='md')
    plan = (schema.build_plan(plan_mapping, atoms=state.atoms,
                              natom=len(state.atoms),
                              masses=list(state.mass_au / AMU_TO_AU))
            if plan_mapping is not None else None)
    return state_adapter.require_usable(view, plan), plan


def _wall_block(radius=2.5, center=None, atoms=(1, 2, 3), force=0.01):
    return {'walls': [{'id': 'vessel', 'geometry': 'sphere',
                       'center': list(center or [10.0, 10.0, 10.0]),
                       'radius': radius, 'atoms': list(atoms),
                       'force_constant': force}]}


class TestCapabilityGranularity(unittest.TestCase):
    """A: the periodic guard asks what a capability measures, not the flags."""

    def test_a_wall_needs_no_bond_graph(self):
        view, _plan = _view(_wall_block())
        self.assertTrue(view.periodic)
        self.assertIsNotNone(view.expansion)

    def test_a_centroid_wall_is_anchored_on_the_unfolded_group(self):
        """A wrapped centroid of 0.01 and L-0.01 is L/2, so it is not used."""
        block = {'walls': [{'id': 'cage', 'geometry': 'sphere',
                            'radius': 2.5, 'atoms': [1, 2, 3],
                            'target': 'com', 'force_constant': 0.01}]}
        torn = (WATER + TORN_SHIFT) % BOX
        plan = schema.build_plan(block, atoms=['O', 'H', 'H'], natom=3,
                                 masses=list(MASS_AMU))
        view, _context = state_adapter.read_state(_State(coordinates=torn))
        spec = plan.actions[0]
        self.assertIsNone(spec.parameters.get('center'))
        expansion = images.read_expansion(view)
        center = state_adapter._wall_center_from_images(
            spec, plan, view, expansion)
        bohr = ANGSTROM_TO_BOHR
        # The mass-weighted center of the continuous molecule: the hydrogen that
        # crossed the face is counted one cell out, where its bond puts it.
        np.testing.assert_allclose(
            center / bohr, [(16.0 * 19.52 + 20.48 + 19.28) / 18.0,
                            (16.0 * 10.0 + 10.0 + 10.73) / 18.0, 10.0],
            atol=1.0e-6)
        # The negative control is the wrapped average, which pulls the center
        # inward by more than a bond length and would confine a group that is
        # not there.
        wrapped_center = ((16.0 * torn[0, 0] + torn[1, 0] + torn[2, 0]) / 18.0)
        self.assertGreater(abs(float(center[0]) / bohr - wrapped_center), 1.0)
        # A molecule torn across a face has its center within a bond length of
        # that face, so the cage around it cannot also fit the cell: the
        # containment check says so, naming the axis.
        with self.assertRaises(wall.WallError) as caught:
            wall.periodic_check(spec, plan, expansion.coordinates, view.cell,
                                center=center)
        self.assertIn('axis a', str(caught.exception))

    def test_a_collective_variable_inside_one_molecule_is_served(self):
        """The O-H distance across a face is still one bond.

        The variable is measured on the unfolded geometry, so the answer is the
        bond, not the 8 Bohr gap the stored coordinates would show; what the
        guard asks is whether the bond graph connects the atoms at all.
        """
        block = {'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
                 'biases': [{'cv': 'r', 'target': 0.96,
                             'force_constant': 0.05}]}
        view, plan = _view(block)
        self.assertTrue(view.periodic)
        self.assertEqual([spec.kind for spec in plan.actions], ['bias'])

    def test_a_collective_variable_the_graph_does_not_connect_is_refused(self):
        """The refusal moved from the block to the variable that needs it.

        Two atoms the bond graph puts in separate components - here an O and an H
        6 Angstrom apart, so no bond is perceived - have no periodic image: the
        distance between them is a molecule-to-molecule separation, and an image
        rule for that would be invented rather than derived. The refusal names the
        action, the cv and how many components its atoms sit in.
        """
        block = {'cvs': [{'name': 'd', 'kind': 'bond', 'atoms': [1, 2]}],
                 'biases': [{'cv': 'd', 'target': 0.96,
                             'force_constant': 0.05}]}
        apart = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.9, 0.0]]
        with self.assertRaises(state_adapter.images.ImageError) as caught:
            _view(block, coordinates=apart)
        message = str(caught.exception)
        self.assertIn("bias 'bias1' cv 'd'", message)
        self.assertIn('connected components', message)

    def test_a_fix_needs_no_bond_graph(self):
        block = {'fixes': [{'id': 'held', 'atoms': [1], 'direction': 'xyz'}]}
        view, _plan = _view(block)
        self.assertTrue(view.periodic)

    def test_an_rmsd_history_potential_names_no_cv_and_is_still_served(self):
        """An aligned RMSD carries its block on the action, not a cv name.

        The image rules ask their question of the atoms the action holds, which
        is the one shape a `kind: rmsd` history potential has; reading it as a
        list of cv names instead is an attribute that does not exist on the
        action, and the run dies before its first step with an AttributeError
        rather than a refusal."""
        block = {'metadynamics': [{'id': 'shape', 'kind': 'rmsd',
                                   'atoms': [1, 2, 3],
                                   'reference': list(np.ravel(WATER)),
                                   'height': 0.0005, 'width': [0.01],
                                   'pace': 2}]}
        view, plan = _view(block)
        self.assertTrue(view.periodic)
        self.assertEqual(plan.actions[0].family, 'rmsd')

    def test_an_rmsd_block_the_graph_does_not_connect_is_refused(self):
        """The same question, asked of the block: two atoms no bond joins have
        no periodic image of their deviation, and the refusal names the action
        and the block rather than superposing across the seam."""
        block = {'metadynamics': [{'id': 'shape', 'kind': 'rmsd',
                                   'atoms': [1, 2],
                                   'reference': [0.0, 0.0, 0.0,
                                                0.96, 0.0, 0.0],
                                   'height': 0.0005, 'width': [0.01],
                                   'pace': 2}]}
        apart = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.9, 0.0]]
        with self.assertRaises(state_adapter.images.ImageError) as caught:
            _view(block, coordinates=apart)
        self.assertIn("metadynamics 'shape' rmsd block", str(caught.exception))


class TestWallInputChecks(unittest.TestCase):
    """M3-1: four named input-time refusals, no second image convention."""

    def test_a_wall_that_crosses_a_face_is_refused(self):
        with self.assertRaises(wall.WallError) as caught:
            _view(_wall_block(radius=12.0))
        message = str(caught.exception)
        self.assertIn('does not fit the cell', message)
        self.assertIn('axis', message)

    def test_a_wall_whose_surface_reaches_the_face_is_refused(self):
        # The limit of the shell check: a surface that reaches the face has no
        # shell left, and the refusal says so with the axis.
        with self.assertRaises(wall.WallError) as caught:
            _view(_wall_block(radius=9.9, center=[10.0, 10.0, 15.0]))
        message = str(caught.exception)
        self.assertIn('axis c', message)

    def test_a_shell_thinner_than_one_step_is_refused(self):
        velocities = np.zeros((3, 3), dtype=np.float64)
        velocities[1, 0] = 1.0
        with self.assertRaises(wall.WallError) as caught:
            _view(_wall_block(radius=5.0), velocities=velocities)
        message = str(caught.exception)
        self.assertIn('thinner than', message)
        self.assertIn('one step', message)

    def test_a_member_starting_outside_is_refused(self):
        with self.assertRaises(wall.WallError) as caught:
            _view(_wall_block(radius=0.5))
        message = str(caught.exception)
        self.assertIn('outside the surface', message)
        self.assertIn('pbc=0', message)

    def test_the_containment_rule_is_about_the_start_not_about_every_step(self):
        """The refusal above says "starts outside", and a one-sided penalty
        surface is exactly the thing thermal motion crosses and the wall pushes
        back. Re-asking the question at a later step aborts a periodic wall for
        doing the one job it is there for, which a real LAMMPS run in a 60-atom
        box did: after its first printed step a member sat 0.0011 Bohr outside
        and the run died in the guard instead of in the potential."""
        _view(_wall_block(radius=0.5), step=1)
        with self.assertRaises(wall.WallError):
            _view(_wall_block(radius=0.5), step=0)

    def test_a_periodic_wall_without_a_center_is_refused(self):
        block = {'walls': [{'id': 'vessel', 'geometry': 'sphere',
                            'radius': 2.5, 'atoms': [1, 2, 3],
                            'force_constant': 0.01}]}
        with self.assertRaises(Exception) as caught:
            _view(block)
        self.assertIn('center', str(caught.exception))

    def test_a_soft_law_wall_is_refused_before_the_first_step(self):
        """The four checks above measure a barrier against a cell face, and this
        law has no barrier: it is a spring toward one point. Worse than having
        nothing to check, a wrapped coordinate jumps its offset by a whole lattice
        vector every time an atom crosses a face, so the force would break at the
        seam. Refused at input time, naming the wall."""
        with self.assertRaises(schema.UnsupportedCapabilityError) as caught:
            schema.build_plan(
                {'walls': [{'id': 'spring', 'geometry': 'sphere',
                            'radius': 2.5, 'center': [10.0, 10.0, 10.0],
                            'atoms': [1, 2, 3], 'force_constant': 0.001,
                            'law': 'soft'}]},
                atoms=['O', 'H', 'H'], natom=3, masses=list(MASS_AMU),
                pbc=[1, 1, 1])
        message = str(caught.exception)
        self.assertIn("'wall_law soft'", message)
        self.assertIn('spring', message)

    def test_the_same_wall_is_served_when_the_cell_is_off(self):
        """The refusal is about the cell, not about the law: with pbc=0 the same
        wall is a spring and runs."""
        plan = schema.build_plan(
            {'walls': [{'id': 'spring', 'geometry': 'sphere', 'radius': 2.5,
                        'center': [10.0, 10.0, 10.0], 'atoms': [1, 2, 3],
                        'force_constant': 0.001, 'law': 'soft'}]},
            atoms=['O', 'H', 'H'], natom=3, masses=list(MASS_AMU),
            pbc=[0, 0, 0])
        self.assertEqual(plan.spec('spring').parameters['law'], 'soft')


class TestConstraintImageRules(unittest.TestCase):
    """M0-4 and the same-component rule, on the runtime guard."""

    def test_a_connected_bond_set_passes_and_unfolds_to_nothing(self):
        block = {'constraints': [{'id': 'b', 'kind': 'bond',
                                  'bonds': [[1, 2], [1, 3]]}]}
        view, _plan = _view(block)
        np.testing.assert_array_equal(view.expansion.offsets,
                                      np.zeros((3, 3), dtype=np.int64))

    def test_a_torn_molecule_unfolds_across_the_face(self):
        block = {'constraints': [{'id': 'b', 'kind': 'bond',
                                  'bonds': [[1, 2]]}]}
        torn = (WATER + TORN_SHIFT) % BOX
        view, _plan = _view(block, coordinates=torn)
        offsets = view.expansion.offsets
        self.assertNotEqual(int(np.abs(offsets).max()), 0)
        length = float(np.linalg.norm(view.expansion.coordinates[0]
                                      - view.expansion.coordinates[1]))
        self.assertAlmostEqual(length / ANGSTROM_TO_BOHR, 0.96, places=2)

    def test_a_pair_the_graph_does_not_connect_is_refused(self):
        block = {'constraints': [{'id': 'b', 'kind': 'bond',
                                  'bonds': [[1, 3]]}]}
        far = WATER.copy()
        far[2] = [15.0, 15.0, 15.0]
        with self.assertRaises(images.ImageError) as caught:
            _view(block, coordinates=far)
        message = str(caught.exception)
        self.assertIn('connected components', message)
        self.assertIn('2 separate', message)

    def test_a_target_under_half_the_box_passes(self):
        """The negative control for the gate: 6 A is fine in a 20 A cell.

        The half-box bound here is 10 A. A target of 6 A is a long bond, not a
        tie, and the run must not be stopped by it - which is also what keeps
        this gate from passing for the wrong reason if a unit ever drifts.
        """
        block = {'constraints': [{'id': 'b', 'kind': 'bond',
                                  'bonds': [[1, 2]], 'target': 6.0}]}
        view, _plan = _view(block)
        self.assertTrue(view.periodic)

    def test_a_target_beyond_half_the_box_is_refused_by_axis(self):
        block = {'constraints': [{'id': 'b', 'kind': 'bond',
                                  'bonds': [[1, 2]], 'target': 11.0}]}
        with self.assertRaises(images.ImageError) as caught:
            _view(block)
        message = str(caught.exception)
        self.assertIn('nearest image tied', message)
        self.assertIn('axis a', message)


class TestFoldBack(unittest.TestCase):
    """The solve runs unfolded; what goes back is a coordinate and its counts."""

    def _expansion(self, coordinates):
        view, _ = state_adapter.read_state(_State(coordinates=coordinates))
        return images.read_expansion(view)

    def test_folding_reproduces_the_stored_pair(self):
        torn = (WATER + TORN_SHIFT) % BOX
        expansion = self._expansion(torn)
        unfolded = expansion.coordinates
        wrapped, crossings = images.fold_back(expansion, unfolded)
        np.testing.assert_allclose(wrapped, torn * ANGSTROM_TO_BOHR,
                                   atol=1.0e-9)
        # The counts are the absolute cell offset of the continuous position:
        # this is the pair an explicit engine reads, and unwrap() of it must be
        # the geometry the solve ran on.
        self.assertEqual(crossings[1, 0], 1)
        self.assertEqual(int(np.abs(crossings).sum()), 1)
        from pyrai3md.utilities.lattice import unwrap_coordinates
        continuous = torn + crossings * BOX
        np.testing.assert_allclose(
            unwrap_coordinates(wrapped / ANGSTROM_TO_BOHR, crossings,
                               np.asarray(LATTICE, dtype=np.float64)),
            continuous, atol=1.0e-9)

    def test_a_shift_across_a_face_moves_the_count_not_the_coordinate(self):
        torn = (WATER + TORN_SHIFT) % BOX
        expansion = self._expansion(torn)
        # One whole cell, in the core's unit: the core reads Bohr and the input
        # is read Angstrom, and mixing the two here would shift the molecule by
        # half a cell and test nothing.
        cell_x = np.asarray(LATTICE, dtype=np.float64)[0] * ANGSTROM_TO_BOHR
        moved = expansion.coordinates + cell_x
        wrapped, crossings = images.fold_back(expansion, moved)
        np.testing.assert_allclose(wrapped, torn * ANGSTROM_TO_BOHR,
                                   atol=1.0e-9)
        # Every atom left the x face once, so the x count went up by one and the
        # other axes, which the shift did not touch, kept their counts.
        np.testing.assert_array_equal(
            crossings, expansion.offsets + np.array([1, 0, 0]))

    def test_a_non_periodic_fold_is_the_identity(self):
        expansion = images.read_expansion(
            state_adapter.read_state(_State(pbc=(0, 0, 0),
                                           lattice=None))[0])
        self.assertFalse(expansion.periodic)
        wrapped, crossings = images.fold_back(expansion, WATER)
        np.testing.assert_array_equal(wrapped, WATER)
        self.assertEqual(int(np.abs(crossings).max()), 0)


class TestTheLedgerHasOneWriter(unittest.TestCase):
    """The counts stored beside a geometry must be the counts of that geometry.

    `System.image` is what turns a stored wrapped coordinate back into the
    continuous position, and an engine reads it as the per-atom cell offset of
    the coordinates it was handed with it. Two writers reach that field on a
    periodic constrained step - the wrap and the constraint fold - and they know
    the counts in two different ways: a wrap reports the crossings of this wrap
    against the frame the ledger describes, a fold reports counts that are
    already absolute for the geometry it returns. Both go through
    `data/geometry.py::record_wrapped_geometry`, which commits the pair together
    and adds nothing.

    Adding the ledger to a fold's absolutes is the mutation this class exists to
    catch. The ledger is per atom (each atom accrues the faces *it* crossed)
    while a fold's counts are uniform per connected component, so the sum shifts
    the atoms of one molecule against each other: measured on a periodic LAMMPS
    run with a strict bond constraint, the engine died on it at evaluation 197
    of 600 with `Bond atom missing in image check`.
    """

    # A ledger whose atoms are not in step is the case that discriminates, and
    # the lag has to sit on an atom of the constrained pair: the mutation this
    # class controls adds the ledger to a fold's counts, which shifts a bonded
    # pair against itself only where the ledger itself disagrees across that
    # bond. A lag on the third atom leaves the pair untouched and the control
    # would pass for the wrong reason.
    LAGGING = [[3, 0, 0], [2, 0, 0], [3, 0, 0]]

    def _state(self, coordinates, image):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema import SystemConstraintConfig
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = 'md'
        config.control.driver = 'main'
        config.control.title = 'ledger_case'
        config.system.atoms = ['O', 'H', 'H']
        config.system.natom = 3
        config.system.mass = [float(value) for value in MASS_AMU]
        config.system.coordinates = [[float(value) for value in row]
                                     for row in coordinates]
        config.system.nstate = 1
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.pbc = [1, 1, 1]
        config.system.lattice_vector = [[float(value) for value in row]
                                        for row in LATTICE]
        config.system.image = [[int(value) for value in row] for row in image]
        # A non-empty plan is what makes the periodic branch of the solve run
        # at all: `state_adapter.require_usable` attaches an expansion only for
        # a periodic view with something declared.
        config.system.constraints = SystemConstraintConfig(**{
            'units': {'length': 'angstrom'},
            'constraints': [{'id': 'oh', 'kind': 'bond', 'bonds': [[1, 2]]}],
        })
        state = PyRAIState(runtime=RuntimeState(total_steps=4))
        state.pmd = config
        constraint_runtime.bind(state)
        return state

    def _step(self, image):
        """One wrap and one solve on the torn molecule, as a run does them."""
        state = self._state((WATER + TORN_SHIFT) % BOX, image)
        apply_pbc(state)
        constraint_runtime.constrain_state(state)
        coords = np.asarray(state.pmd.system.coordinates, dtype=np.float64)
        counts = np.asarray(state.pmd.system.image, dtype=np.int64)
        cell = np.asarray(LATTICE, dtype=np.float64)
        continuous = unwrap_coordinates(coords, counts, cell)
        bond = float(np.linalg.norm(continuous[0] - continuous[1]))
        span = int(np.abs(np.diff(counts, axis=0)).max())
        return bond, span, counts

    def test_the_committed_pair_rebuilds_the_geometry_the_solve_ran_on(self):
        bond, span, _counts = self._step(self.LAGGING)
        self.assertAlmostEqual(
            bond, 0.96, places=2,
            msg='the stored counts do not belong to the stored coordinates: the '
                'constrained bond reads %.4f A on the geometry they rebuild'
                % bond)
        self.assertLessEqual(
            span, 1,
            msg='a bonded pair spans %d cells, which is the reading an engine '
                'refuses as `Bond atom missing in image check`' % span)

    def test_adding_the_ledger_to_a_fold_shifts_the_molecule_against_itself(self):
        """The negative control: the criterion above has resolution.

        The mutation is the write this round replaced - commit a fold's absolute
        counts with the ledger added - and it must tear the constrained bond, or
        the passing assertion above is not reading the pair at all.
        """
        from unittest import mock

        def adding(state, coordinates, image=None, also=None):
            updates = {'coordinates': np.asarray(coordinates,
                                                dtype=np.float64).tolist()}
            if image is not None:
                previous = np.asarray(state.pmd.system.image or [],
                                      dtype=np.int64)
                counts = np.asarray(image, dtype=np.int64)
                if previous.shape == counts.shape:
                    counts = previous + counts
                updates['image'] = [[int(value) for value in row]
                                    for row in counts]
            if also:
                updates.update(also)
            state.update_system(updates)
            return state

        with mock.patch.object(constraint_runtime, 'record_wrapped_geometry',
                               adding):
            bond, _span, _counts = self._step(self.LAGGING)
        self.assertGreater(
            bond, 5.0,
            msg='adding the ledger to a fold left the bond at %.4f A, so the '
                'criterion this controls cannot fail' % bond)


if __name__ == '__main__':
    unittest.main()


class TestDihedralBranch(unittest.TestCase):
    """The third condition kind, whose rows name four atoms.

    A torsion is served where the bond graph reaches from the first atom to the
    last, and refused where it does not - and the refusal has to name the
    constraint branch, not just the bias cvs it was first written for. The
    served half still has a shipped run (`pyrai3md/examples/
    constraint_md_torsion_intermolecular.yaml`); the refused half is proven
    here only, since the periodic negative example was retired on
    2026-10-02 - the qualification harness scores a designed refusal as a
    failure, so a negative case has no verdict to report.
    """

    BOX = 20.0
    # H-O-O-H: one connected chain, each neighbour 0.97 A from the atom before
    # it and the two oxygens 1.45 A apart, so the perceived bond graph reaches
    # end to end.
    HOOH = np.array([[9.65, 10.90, 10.00],
                     [10.00, 10.00, 10.00],
                     [11.45, 10.00, 10.00],
                     [11.80, 10.90, 10.00]])
    # Two waters 2.70 A apart: each is its own component, and the row that
    # reaches across the hydrogen bond has no image to be measured on.
    TWO_WATERS = np.array([[10.00, 10.00, 10.00],
                           [10.96, 10.00, 10.00],
                           [9.76, 10.73, 10.00],
                           [12.70, 10.00, 10.00],
                           [13.66, 10.00, 10.00],
                           [12.46, 10.73, 10.00]])

    @staticmethod
    def _block(row):
        return {'constraints': [{'id': 'twist', 'kind': 'dihedral',
                                 'bonds': [list(row)],
                                 'tolerance': 1.0e-10}]}

    def test_a_torsion_the_graph_reaches_is_served_in_a_cell(self):
        view, plan = _view(self._block([3, 1, 2, 4]),
                           coordinates=self.HOOH, atoms=['H', 'O', 'O', 'H'],
                           masses=[1.008, 15.999, 15.999, 1.008])
        self.assertTrue(view.periodic)
        self.assertEqual([spec.kind for spec in plan.actions], ['constraint'])

    def test_a_torsion_across_two_molecules_is_refused_naming_the_constraint(self):
        with self.assertRaises(state_adapter.images.ImageError) as caught:
            _view(self._block([2, 1, 4, 5]),
                  coordinates=self.TWO_WATERS,
                  atoms=['O', 'H', 'H', 'O', 'H', 'H'],
                  masses=[15.999, 1.008, 1.008, 15.999, 1.008, 1.008])
        message = str(caught.exception)
        self.assertIn("constraint 'twist'", message)
        self.assertIn('2 separate connected components', message)

    def test_the_same_row_without_a_cell_is_not_refused(self):
        """The refusal is a periodic-image statement, so no cell means no
        refusal: the identical four-atom row is an ordinary condition."""
        plan = schema.build_plan(
            self._block([2, 1, 4, 5]), atoms=['O', 'H', 'H', 'O', 'H', 'H'],
            natom=6, masses=[15.999, 1.008, 1.008, 15.999, 1.008, 1.008],
            pbc=[0, 0, 0])
        self.assertEqual(plan.actions[0].family, 'dihedral')


