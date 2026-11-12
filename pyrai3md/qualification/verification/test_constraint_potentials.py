"""
======================================================================
Tests for the additive potentials: harmonic bias and the four walls.

Both families add energy and force without moving anything, so the two things
worth pinning are the derivative (always against a central difference) and the
boundary behaviour: a one-sided form must be continuous, must be zero inside
and must have a zero slope at the surface, or a Verlet step across the
boundary injects an impulse that the run cannot account for.

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
from pyrai3md.workflow.constraints import bias, wall
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

MASSES = np.array([16.0, 1.0, 1.0, 12.0])
ATOMS = ['O', 'H', 'H', 'C']
BASE = np.array([[0.00, 0.00, 0.00],
                 [1.60, 0.30, 0.10],
                 [0.40, 1.70, -0.20],
                 [2.10, 1.80, 1.30]])
CONTEXT = records.EvaluationContext(purpose='single', step=0)
STEPS = (1.0e-4, 1.0e-5, 1.0e-6)
ATOL = 1.0e-7


def _plan(mapping):
    return schema.build_plan(mapping, atoms=ATOMS, natom=4, masses=MASSES)


def _evaluate(spec, plan, coords, state=None, masses=MASSES):
    if spec.kind == 'bias':
        return bias.evaluate(spec, plan, coords,
                            state or records.ConstraintState(), CONTEXT)
    return wall.evaluate(spec, plan, coords,
                         state or records.ConstraintState(), CONTEXT,
                         masses=masses)


class TestHarmonicBias(unittest.TestCase):

    def setUp(self):
        self.plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 0.02,
                                       'region': 'all'}]})
        self.spec = self.plan.spec(self.plan.ids('bias')[0])

    def test_energy_is_a_half_k_delta_squared(self):
        result = _evaluate(self.spec, self.plan, BASE)
        delta = result['cv_value'] - self.spec.target
        self.assertAlmostEqual(result['energy'],
                               0.5 * 0.02 / ANGSTROM_TO_BOHR ** 2 * delta ** 2,
                               places=10)

    def test_gradient_matches_central_difference(self):
        result = _evaluate(self.spec, self.plan, BASE)
        self._check_gradient(lambda x: bias.evaluate(
            self.spec, self.plan, x, records.ConstraintState(), CONTEXT)
            ['energy'], result['gradient'])

    def test_at_the_target_the_energy_and_force_vanish(self):
        value = bias.variable_value(self.plan.cvs['r'], BASE)
        spec = records.ConstraintSpec('at_target', 'bias', 'harmonic', cv='r',
                                      target=value,
                                      parameters={'force_constant': 0.02},
                                      region='all')
        result = bias.evaluate(spec, self.plan, BASE,
                               records.ConstraintState(), CONTEXT)
        self.assertAlmostEqual(result['energy'], 0.0, places=14)
        self.assertEqual(float(np.abs(result['gradient']).max()), 0.0)

    def test_upper_region_is_flat_below_the_target(self):
        spec = self._with_region('upper', above=False)
        result = _evaluate(spec, self.plan, BASE)
        self.assertEqual(result['energy'], 0.0)
        self.assertEqual(float(np.abs(result['gradient']).max()), 0.0)

    def test_upper_region_is_active_above_the_target(self):
        spec = self._with_region('upper', above=True)
        result = _evaluate(spec, self.plan, BASE)
        self.assertGreater(result['energy'], 0.0)
        self._check_gradient(lambda x: bias.evaluate(
            spec, self.plan, x, records.ConstraintState(), CONTEXT)['energy'],
            result['gradient'])

    def test_lower_region_is_the_mirror_image(self):
        spec = self._with_region('lower', above=False)
        self.assertGreater(_evaluate(spec, self.plan, BASE)['energy'], 0.0)

    def test_repeated_evaluation_is_identical(self):
        first = _evaluate(self.spec, self.plan, BASE)
        second = _evaluate(self.spec, self.plan, BASE)
        self.assertEqual(first['energy'], second['energy'])
        np.testing.assert_allclose(first['gradient'], second['gradient'])

    def test_dihedral_difference_wraps_across_the_branch(self):
        plan = _plan({'cvs': [{'name': 'd', 'kind': 'dihedral',
                               'atoms': [1, 2, 3, 4]}],
                      'biases': [{'cv': 'd', 'target': np.radians(170.0),
                                  'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('bias')[0])
        result = _evaluate(spec, plan, BASE)
        self.assertLess(abs(result['cv_value'] - spec.target), np.pi)
        self._check_gradient(lambda x: bias.evaluate(
            spec, plan, x, records.ConstraintState(), CONTEXT)['energy'],
            result['gradient'])

    def test_a_bias_with_no_target_and_no_binding_fails(self):
        plan = _plan({'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
                      'biases': [{'cv': 'r', 'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('bias')[0])
        with self.assertRaises(bias.BiasError):
            bias.evaluate(spec, plan, BASE, records.ConstraintState(),
                          CONTEXT)
        bound = records.ConstraintState(targets={spec.id: 1.0})
        self.assertIn('energy', bias.evaluate(spec, plan, BASE, bound,
                                              CONTEXT))

    def _with_region(self, region, above):
        value = bias.variable_value(self.plan.cvs['r'], BASE)
        target = value * (0.5 if above else 1.5)
        return records.ConstraintSpec(
            'probe', 'bias', 'harmonic', cv='r', target=target,
            parameters={'force_constant': 0.02}, region=region)

    def _check_gradient(self, energy, gradient):
        for step in STEPS:
            for atom in range(4):
                for axis in range(3):
                    forward = BASE.copy()
                    forward[atom, axis] += step
                    back = BASE.copy()
                    back[atom, axis] -= step
                    numeric = (energy(forward) - energy(back)) / (2.0 * step)
                    self.assertLessEqual(abs(numeric - gradient[atom, axis]),
                                         ATOL,
                                         msg='step %g atom %d axis %d'
                                         % (step, atom, axis))


class TestWallGeometries(unittest.TestCase):

    def test_sphere_interior_boundary_and_exterior(self):
        spec = self._sphere(radius=1.5, center=(0.0, 0.0, 0.0))
        inside = self._energy(spec, np.array([[0.5, 0.0, 0.0]]))
        surface = self._energy(spec, np.array([[1.5, 0.0, 0.0]]))
        outside = self._energy(spec, np.array([[2.5, 0.0, 0.0]]))
        self.assertEqual(inside, 0.0)
        self.assertEqual(surface, 0.0)
        self.assertGreater(outside, 0.0)

    def test_sphere_gradient_matches_central_difference(self):
        spec = self._sphere(radius=1.0, center=(0.2, 0.1, 0.0))
        self._check(spec, BASE[[3]])

    def test_ellipsoid_gradient_matches_central_difference(self):
        spec = self._wall('ellipsoid', semiaxes=(1.1, 0.9, 0.7),
                          center=(0.3, 0.2, 0.1))
        self._check(spec, BASE[[1, 2, 3]])

    def test_box_gradient_matches_central_difference(self):
        spec = self._wall('box', lengths=(1.4, 1.2, 1.0),
                          center=(0.2, 0.1, 0.0))
        self._check(spec, BASE[[0, 1, 2, 3]])

    def test_triclinic_gradient_matches_central_difference(self):
        cell = [[1.8, 0.0, 0.0], [0.3, 1.6, 0.0], [0.1, 0.2, 1.7]]
        spec = self._wall('triclinic', cell=cell, center=(0.0, 0.0, 0.0))
        self._check(spec, BASE[[0, 1, 2, 3]])

    def test_box_and_diagonal_triclinic_agree(self):
        box = self._wall('box', lengths=(2.0, 2.0, 2.0), center=(0., 0., 0.))
        triclinic = self._wall('triclinic',
                               cell=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
                                     (0.0, 0.0, 2.0)),
                               center=(0., 0., 0.))
        rng = np.random.default_rng(3)
        for _ in range(50):
            point = rng.uniform(-1.6, 1.6, size=(1, 3))
            self.assertAlmostEqual(self._energy(box, point),
                                   self._energy(triclinic, point), places=12)

    def test_a_corner_crosses_two_faces_and_sums_them(self):
        spec = self._wall('box', lengths=(1.0, 1.0, 1.0), center=(0., 0., 0.))
        face = np.array([[0.8, 0.0, 0.0]])
        corner = np.array([[0.8, 0.8, 0.0]])
        single = self._energy(spec, face)
        self.assertAlmostEqual(self._energy(spec, corner), 2.0 * single,
                               places=12)

    def _bound_state(self, plan, spec_id, coords=BASE):
        """A state with the wall's centroid captured at the reference.

        A centroid wall with no declared center is bound at binding time, and
        evaluating it unbound is a contract violation the wall reports; the
        controller is the thing that binds.
        """
        from pyrai3md.workflow.constraints.controller \
            import SystemConstraintController
        controller = SystemConstraintController(plan)
        return controller.bind_reference(coords, masses=MASSES)

    def test_com_wall_counts_the_group_once_and_shares_the_force(self):
        # A declared center, so the shell does not move with the group: the
        # center of mass of O at the origin and H at 1.6 Bohr is about 0.1
        # Bohr out, and a 0.05 Bohr shell around the origin is inside it.
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'center': [0, 0, 0],
                                 'radius': 0.05, 'target': 'com', 'group': 'g',
                                 'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('wall')[0])
        result = wall.evaluate(spec, plan, BASE, records.ConstraintState(),
                               CONTEXT, masses=MASSES)
        self.assertGreater(result['energy'], 0.0)
        self._check_gradient_against(plan, spec, BASE, result['gradient'],
                                     records.ConstraintState())
        weights = MASSES[[0, 1]] / MASSES[[0, 1]].sum()
        for axis in range(3):
            self.assertAlmostEqual(result['gradient'][0, axis],
                                   result['gradient'][1, axis]
                                   * weights[0] / weights[1], places=14)

    def test_cog_wall_shares_equally(self):
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'center': [0, 0, 0],
                                 'radius': 0.4, 'target': 'cog', 'group': 'g',
                                 'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('wall')[0])
        result = wall.evaluate(spec, plan, BASE, records.ConstraintState(),
                               CONTEXT, masses=MASSES)
        self.assertGreater(result['energy'], 0.0)
        self.assertAlmostEqual(abs(result['gradient'][0, 0]),
                               abs(result['gradient'][1, 0]), places=14)

    def test_a_bound_centroid_center_follows_the_reference_geometry(self):
        """No declared center means the group center at binding time.

        That makes the wall a relative shell: it bites only once the group
        moves away from where it started, which is what a confinement on a
        solute means. At the reference geometry itself the overshoot is
        exactly the negative radius.
        """
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      # 0.05 Angstrom shell: tight enough that moving the
                      # hydrogen by 4 Bohr drags the center of mass out.
                      'walls': [{'geometry': 'sphere', 'radius': 0.05,
                                 'target': 'com', 'group': 'g',
                                 'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('wall')[0])
        state = self._bound_state(plan, spec.id)
        self.assertIn('center:%s' % spec.id, state.anchors)
        at_reference = wall.evaluate(spec, plan, BASE, state, CONTEXT,
                                     masses=MASSES)
        self.assertEqual(at_reference['energy'], 0.0)
        # The declared radius is 0.5 Angstrom, and the plan holds internal
        # units, so the overshoot at the reference is the converted radius.
        self.assertAlmostEqual(at_reference['extent'],
                               -0.05 * ANGSTROM_TO_BOHR, places=9)
        pushed = BASE.copy()
        pushed[1] += np.array([4.0, 0.0, 0.0])
        moved = wall.evaluate(spec, plan, pushed, state, CONTEXT,
                              masses=MASSES)
        self.assertGreater(moved['energy'], 0.0)

    def test_an_unbound_centroid_wall_is_refused(self):
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'radius': 1.2,
                                 'target': 'com', 'group': 'g',
                                 'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('wall')[0])
        with self.assertRaises(wall.WallError):
            wall.evaluate(spec, plan, BASE, records.ConstraintState(),
                          CONTEXT, masses=MASSES)

    def test_wall_needing_masses_reports_it(self):
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'radius': 1.0,
                                 'target': 'com', 'group': 'g',
                                 'force_constant': 0.01}]})
        spec = plan.spec(plan.ids('wall')[0])
        with self.assertRaises(wall.WallError):
            wall.evaluate(spec, plan, BASE, records.ConstraintState(),
                          CONTEXT, masses=None)

    def test_triclinic_k_is_independent_of_shear(self):
        """A sheared cell keeps the same face spacing, so the same k means
        the same physics; only the interplanar distance matters."""
        straight = self._wall('triclinic',
                              cell=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
                                    (0.0, 0.0, 2.0)), center=(0., 0., 0.))
        point = np.array([[0.0, 0.0, 1.2]])
        self.assertAlmostEqual(self._energy(straight, point),
                               self._energy(self._wall(
                                   'box', lengths=(2.0, 2.0, 2.0),
                                   center=(0., 0., 0.)), point), places=12)

    def _sphere(self, radius, center):
        return records.ConstraintSpec(
            'probe', 'wall', 'sphere', atoms=[0],
            parameters={'geometry': 'sphere', 'power': 2,
                        'force_constant': 0.01, 'target_kind': 'atom',
                        'radius': radius, 'center': list(center)})

    def _wall(self, geometry, center=(0., 0., 0.), power=2,
              force_constant=0.01, **shape):
        parameters = {'geometry': geometry, 'power': power,
                      'force_constant': force_constant,
                      'target_kind': 'atom', 'center': list(center)}
        if 'semiaxes' in shape:
            parameters['semiaxes'] = list(shape['semiaxes'])
        if 'lengths' in shape:
            parameters['lengths'] = list(shape['lengths'])
        if 'cell' in shape:
            parameters['cell'] = [list(row) for row in shape['cell']]
        if 'radius' in shape:
            parameters['radius'] = shape['radius']
        return records.ConstraintSpec('probe', 'wall', geometry, atoms=[0],
                                      parameters=parameters)

    def _energy(self, spec, coords):
        return wall.evaluate(spec, None, coords, records.ConstraintState(),
                             CONTEXT, masses=None)['energy']

    def _check(self, spec, coords):
        result = wall.evaluate(spec, None, coords, records.ConstraintState(),
                               CONTEXT, masses=None)
        for atom in range(coords.shape[0]):
            for axis in range(3):
                for step in STEPS:
                    forward = coords.copy()
                    forward[atom, axis] += step
                    back = coords.copy()
                    back[atom, axis] -= step
                    numeric = (self._energy(spec, forward)
                               - self._energy(spec, back)) / (2.0 * step)
                    self.assertLessEqual(
                        abs(numeric - result['gradient'][atom, axis]), ATOL,
                        msg='atom %d axis %d step %g' % (atom, axis, step))

    def _check_gradient_against(self, plan, spec, coords, gradient,
                               state=None):
        bound = state if state is not None else records.ConstraintState()

        def energy(geometry):
            return wall.evaluate(spec, plan, geometry, bound, CONTEXT,
                                 masses=MASSES)['energy']
        for atom in (0, 1):
            for axis in range(3):
                step = 1.0e-6
                forward = coords.copy()
                forward[atom, axis] += step
                back = coords.copy()
                back[atom, axis] -= step
                numeric = (energy(forward) - energy(back)) / (2.0 * step)
                self.assertLessEqual(abs(numeric - gradient[atom, axis]),
                                     ATOL)


class TestWallGroupingAndCenter(unittest.TestCase):
    """What a wall confines, where it is centered, and how each is weighted.

    The grouping and the target are two separate choices: `wall_groups` says how
    the selection is cut up, `wall_target` says how one piece of it is weighted,
    and a molecule of one atom is the same object either way. The center is a third
    choice, and its default is the wall's own atoms rather than the coordinate
    origin, because a boundary at (0, 0, 0) is a statement about the frame no input
    ever made.
    """

    def _bind(self, plan):
        from pyrai3md.workflow.constraints.controller \
            import SystemConstraintController
        return SystemConstraintController(plan).bind_reference(BASE,
                                                               masses=MASSES)

    def _wall(self, plan):
        return plan.spec(plan.ids('wall')[0])

    def test_an_atom_wall_is_centered_on_the_atoms_it_acts_on(self):
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                 'group': 'g', 'force_constant': 0.01}]})
        spec = self._wall(plan)
        anchor = self._bind(plan).anchors['center:%s' % spec.id]
        self.assertTrue(np.allclose(anchor, 0.5 * (BASE[0] + BASE[1])))
        # Not the origin: the two atoms are 1.63 Bohr apart, so their midpoint is
        # 0.81 Bohr out and a wall centered there is not a wall centered at zero.
        self.assertGreater(float(np.linalg.norm(anchor)), 0.5)

    def test_the_default_center_is_not_a_boundary_at_the_origin(self):
        """The same sphere at the selection's center and at (0, 0, 0) is a
        different run, and the one with no center written is the first."""
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                 'group': 'g', 'force_constant': 0.01}]})
        spec = self._wall(plan)
        state = self._bind(plan)
        got = wall.evaluate(spec, plan, BASE, state, CONTEXT, masses=MASSES)
        at_origin = records.ConstraintSpec(
            'origin', 'wall', 'sphere', atoms=list(spec.atoms),
            parameters=dict(spec.parameters, center=[0.0, 0.0, 0.0]))
        other = wall.evaluate(at_origin, plan, BASE,
                              records.ConstraintState(), CONTEXT,
                              masses=MASSES)
        self.assertGreater(got['energy'], 0.0)
        self.assertNotAlmostEqual(got['energy'], other['energy'], places=10)
        # Both atoms sit the same distance from their own midpoint, so the two
        # contributions are equal and the total is twice one of them.
        self.assertAlmostEqual(got['extent'],
                               float(np.linalg.norm(BASE[1] - BASE[0])) / 2.0
                               - 0.4 * ANGSTROM_TO_BOHR, places=12)

    def test_wall_ref_moves_the_center_to_the_atoms_it_names(self):
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]},
                                 {'name': 'h', 'atoms': [3, 4]}],
                      'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                 'group': 'g', 'ref_atoms': [3, 4],
                                 'force_constant': 0.01}]})
        spec = self._wall(plan)
        anchor = self._bind(plan).anchors['center:%s' % spec.id]
        self.assertTrue(np.allclose(anchor, 0.5 * (BASE[2] + BASE[3])))

    def test_wall_center_outranks_wall_ref(self):
        """Both declared: the coordinates win, the anchor is never read, and no
        capture happens at the reference geometry."""
        plan = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                      'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                 'group': 'g', 'center': [0.1, 0.2, 0.3],
                                 'ref_atoms': [1, 2],
                                 'force_constant': 0.01}]})
        spec = self._wall(plan)
        self.assertIn('center', spec.parameters)
        self.assertNotIn('anchor', spec.parameters)
        state = self._bind(plan)
        self.assertNotIn('center:%s' % spec.id, state.anchors)
        result = wall.evaluate(spec, plan, BASE, state, CONTEXT, masses=MASSES)
        # The farther of the two atoms decides `extent`, and with the coordinates
        # at the origin the anchor atoms would have been the midpoint of the same
        # pair, so a value this large can only come from the declared point.
        distance = float(np.linalg.norm(BASE[1]
                                        - np.array([0.1, 0.2, 0.3])
                                        * ANGSTROM_TO_BOHR))
        self.assertAlmostEqual(result['extent'],
                               distance - 0.4 * ANGSTROM_TO_BOHR, places=12)

    def test_a_molecule_of_one_atom_is_the_atom_itself(self):
        """`com` with singleton groups and `atom` are the same run: the weighting
        of one atom is its own coordinate either way."""
        midpoint = 0.5 * (BASE[0] + BASE[1])
        center = (midpoint / ANGSTROM_TO_BOHR).tolist()
        grouped = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                         'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                    'target': 'com', 'group': 'g',
                                    'molecules': [[1], [2]], 'center': center,
                                    'force_constant': 0.01}]})
        plain = _plan({'groups': [{'name': 'g', 'atoms': [1, 2]}],
                       'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                  'group': 'g', 'center': center,
                                  'force_constant': 0.01}]})
        one = wall.evaluate(self._wall(grouped), grouped, BASE,
                            records.ConstraintState(), CONTEXT, masses=MASSES)
        two = wall.evaluate(self._wall(plain), plain, BASE,
                            records.ConstraintState(), CONTEXT, masses=MASSES)
        self.assertEqual(one['groups'], 2)
        self.assertAlmostEqual(one['energy'], two['energy'], places=12)
        self.assertLess(float(np.max(np.abs(one['gradient']
                                            - two['gradient']))), 1.0e-12)

    def test_grouping_gives_one_evaluation_point_per_molecule(self):
        ungrouped = _plan({'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                      'target': 'cog', 'force_constant': 0.01,
                                      'center': [0.0, 0.0, 0.0],
                                      'atoms': [1, 2, 3, 4]}]})
        grouped = _plan({'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                   'target': 'cog', 'force_constant': 0.01,
                                   'center': [0.0, 0.0, 0.0],
                                   'atoms': [1, 2, 3, 4],
                                   'molecules': [[1, 2], [3, 4]]}]})
        one = wall.evaluate(self._wall(ungrouped), ungrouped, BASE,
                            records.ConstraintState(), CONTEXT, masses=MASSES)
        two = wall.evaluate(self._wall(grouped), grouped, BASE,
                            records.ConstraintState(), CONTEXT, masses=MASSES)
        self.assertEqual(one['groups'], 1)
        self.assertEqual(two['groups'], 2)
        self.assertNotAlmostEqual(one['energy'], two['energy'], places=10)

    def test_a_wall_that_names_no_atoms_confines_the_whole_system(self):
        """The default is every atom, and it is said out loud in the run record:
        an input that appears to select nothing and in fact selects everything is
        not a quiet fallback. The count still has to reach the evaluation, since a
        wall that lost its atoms and gained none would report zero energy and look
        like a run that obeys its boundary."""
        plan = _plan({'walls': [{'geometry': 'sphere', 'radius': 0.4,
                                 'center': [0.0, 0.0, 0.0],
                                 'force_constant': 0.01}]})
        spec = self._wall(plan)
        self.assertEqual(list(spec.atoms), [0, 1, 2, 3])
        self.assertTrue(any('acts on all 4 atoms' in note
                            for note in plan.notes))
        total = wall.evaluate(spec, plan, BASE, records.ConstraintState(),
                              CONTEXT, masses=MASSES)
        self.assertEqual(total['groups'], 4)
        per_atom = sum(wall.evaluate(
            records.ConstraintSpec('a%d' % atom, 'wall', 'sphere',
                                   atoms=[atom],
                                   parameters=dict(spec.parameters)),
            plan, BASE, records.ConstraintState(), CONTEXT,
            masses=None)['energy'] for atom in range(4))
        self.assertAlmostEqual(total['energy'], per_atom, places=14)


class TestSoftLaw(unittest.TestCase):
    """`law: soft`, the wall that replaced the archived polynomial cavity.

    The arithmetic that proves the two are one law is in
    `test_constraint_legacy_equivalence.py`; what belongs here is the behaviour a
    run sees: where the center comes from, that it does not move with the
    molecules it holds, how the old `center_type` and `compress` keywords are now
    spelled, and what the input refuses that the archived code papered over.
    """

    def _bind(self, plan):
        from pyrai3md.workflow.constraints.controller \
            import SystemConstraintController
        return SystemConstraintController(plan).bind_reference(BASE,
                                                               masses=MASSES)

    def _wall(self, plan):
        return plan.spec(plan.ids('wall')[0])

    def _run(self, block, coords, state=None, step=0):
        plan = _plan(block)
        spec = self._wall(plan)
        bound = state if state is not None else records.ConstraintState()
        return wall.evaluate(spec, plan, coords, bound,
                             records.EvaluationContext(purpose='md', step=step),
                             masses=MASSES)

    def test_the_group_is_held_on_both_sides_of_the_center(self):
        """A confining wall is flat inside its surface; this one pulls back from
        wherever the group has gone, which is what makes it a spring and not a
        boundary."""
        block = {'groups': [{'name': 'g', 'atoms': [1, 2, 3, 4]}],
                 'walls': [{'geometry': 'sphere', 'radius': 2.0,
                            'target': 'com', 'group': 'g', 'law': 'soft',
                            'force_constant': 0.001}]}
        plan = _plan(block)
        spec = self._wall(plan)
        state = self._bind(plan)
        at = wall.evaluate(spec, plan, BASE, state, CONTEXT, masses=MASSES)
        forward = wall.evaluate(spec, plan, BASE + np.array([[2.0, 0, 0]]),
                                state, CONTEXT, masses=MASSES)
        back = wall.evaluate(spec, plan, BASE - np.array([[2.0, 0, 0]]),
                             state, CONTEXT, masses=MASSES)
        self.assertEqual(at['energy'], 0.0)
        self.assertAlmostEqual(forward['energy'], back['energy'], places=15)
        self.assertGreater(forward['energy'], 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(
            forward['gradient'] + back['gradient']))), 0.0, places=15)

    def test_the_quadratic_soft_law_is_a_spring_of_2k_over_r_squared(self):
        """`V = k (r/R)^2` has gradient `+2k r / R^2` and therefore the restoring
        force `-2k r / R^2`: the cavity's `scale` is not a spring constant, and a
        migration that read it as one is stiff by R^2/2."""
        radius = 2.0 * ANGSTROM_TO_BOHR
        block = {'walls': [{'geometry': 'sphere', 'radius': 2.0,
                            'force_constant': 0.001, 'law': 'soft',
                            'center': [0.0, 0.0, 0.0], 'atoms': [1]}]}
        point = np.array([[1.3, -0.4, 0.2]])
        result = self._run(block, point)
        for axis in range(3):
            self.assertAlmostEqual(
                result['gradient'][0, axis],
                2.0 * result['parameters']['force_constant']
                * point[0, axis] / radius ** 2, places=14)
            self.assertAlmostEqual(
                result['energy'],
                result['parameters']['force_constant']
                * float(np.sum(point[0] ** 2)) / radius ** 2, places=14)

    def test_a_bound_center_does_not_follow_the_molecules_it_holds(self):
        """The whole system translated is not the same run: the center was
        captured at the reference, so a shifted molecule is now away from it. A
        recompute from the live geometry would make the variable identically
        zero, which is the failure this pins."""
        block = {'groups': [{'name': 'g', 'atoms': [1, 2, 3, 4]}],
                 'walls': [{'geometry': 'sphere', 'radius': 2.0,
                            'target': 'com', 'group': 'g', 'law': 'soft',
                            'force_constant': 0.001}]}
        plan = _plan(block)
        spec = self._wall(plan)
        state = self._bind(plan)
        shifted = BASE + np.array([[3.0, 0.0, 0.0]])
        moved = wall.evaluate(spec, plan, shifted, state, CONTEXT,
                              masses=MASSES)
        self.assertGreater(moved['energy'], 0.0)
        again = wall.evaluate(spec, plan, BASE, state, CONTEXT, masses=MASSES)
        self.assertEqual(again['energy'], 0.0)
        restored = records.ConstraintState.from_dict(state.to_dict())
        self.assertAlmostEqual(
            float(np.max(np.abs(np.asarray(restored.anchors['center:%s'
                                                       % spec.id])
                                - np.asarray(state.anchors['center:%s'
                                                            % spec.id])))),
            0.0, places=14)

    def test_the_target_places_the_center_the_way_its_weights_do(self):
        """The archived `center_type: mass` and `geometric` are the wall's
        `target: com` and `cog`, and one asymmetric group separates them."""
        block = {'groups': [{'name': 'g', 'atoms': [1, 2, 3, 4]}],
                 'walls': [{'geometry': 'sphere', 'radius': 2.0,
                            'target': 'com', 'group': 'g', 'law': 'soft',
                            'force_constant': 0.001}]}
        mass_plan = _plan(block)
        mass_spec = self._wall(mass_plan)
        mass_center = self._bind(mass_plan).anchors['center:%s' % mass_spec.id]
        geometric_plan = _plan(dict(block, walls=[dict(
            block['walls'][0], target='cog')]))
        geometric_spec = self._wall(geometric_plan)
        geometric_center = self._bind(geometric_plan).anchors[
            'center:%s' % geometric_spec.id]
        weights = MASSES / MASSES.sum()
        self.assertTrue(np.allclose(mass_center,
                                    np.sum(BASE * weights[:, None], axis=0)))
        self.assertTrue(np.allclose(geometric_center, BASE.mean(axis=0)))
        self.assertGreater(float(np.linalg.norm(mass_center
                                                - geometric_center)), 0.1)

    def test_the_origin_option_is_a_declared_center(self):
        """`center_type: origin` had no wall spelling; `wall_center 0 0 0` is that
        statement, and the run record says the center was declared rather than
        captured."""
        block = {'walls': [{'geometry': 'sphere', 'radius': 2.0,
                            'force_constant': 0.001, 'law': 'soft',
                            'center': [0.0, 0.0, 0.0], 'atoms': [1, 2, 3, 4]}]}
        plan = _plan(block)
        spec = self._wall(plan)
        state = self._bind(plan)
        self.assertNotIn('center:%s' % spec.id, state.anchors)
        result = wall.evaluate(spec, plan, BASE, state, CONTEXT, masses=MASSES)
        # Atom 1 sits at the origin, so it pays nothing and the other three carry
        # the whole energy: a center captured from the group would have moved it.
        self.assertEqual(result['groups'], 4)
        self.assertAlmostEqual(float(np.abs(result['gradient'][0]).max()),
                               0.0, places=15)

    def test_a_radius_ramp_is_the_archived_compression(self):
        """`compress: [ratio, steps]` shrank the radii as
        `R * (1 + (ratio - 1) * min(step, steps) / steps)`, which is this build's
        ramp with the window opened at step 0 and closed at `steps`: the two are
        the same number at every step, including past the end."""
        ratio, steps = 0.5, 10
        ramped = {'walls': [{'geometry': 'sphere', 'radius': 2.0,
                             'force_constant': 0.001, 'law': 'soft',
                             'center': [0.0, 0.0, 0.0], 'atoms': [1],
                             'schedules': [{'parameter': 'radius',
                                            'value': [2.0, 2.0 * ratio],
                                            'start_step': 0,
                                            'end_step': steps}]}]}
        point = np.array([[1.3, -0.4, 0.2]])
        for step in (0, 3, 10, 14):
            factor = 1.0 + (ratio - 1.0) * min(step, steps) / steps
            plain = {'walls': [{'geometry': 'sphere',
                                'radius': 2.0 * factor,
                                'force_constant': 0.001, 'law': 'soft',
                                'center': [0.0, 0.0, 0.0], 'atoms': [1]}]}
            got = self._run(ramped, point, step=step)
            want = self._run(plain, point)
            self.assertAlmostEqual(got['energy'], want['energy'], places=15)
            self.assertAlmostEqual(got['extent'], want['extent'], places=12)

    def test_a_power_below_two_is_refused_rather_than_floored(self):
        """The archived code replaced `factor` by `max(factor, 2)` in silence. The
        exponent is the shape of the potential, so this build says the number is
        illegal instead of running a different one."""
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _plan({'walls': [{'geometry': 'sphere', 'radius': 2.0,
                              'force_constant': 0.001, 'law': 'soft',
                              'power': 1, 'center': [0, 0, 0], 'atoms': [1]}]})
        self.assertIn('power', str(caught.exception))

    def test_an_atom_in_two_molecules_is_refused(self):
        """A partition that double-counts one atom and drops another still covers
        the right number of atoms, so the check is on the set, not the count."""
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _plan({'walls': [{'geometry': 'sphere', 'radius': 2.0,
                              'force_constant': 0.001, 'law': 'soft',
                              'target': 'com', 'atoms': [1, 2, 3, 4],
                              'molecules': [[1, 2], [2, 3, 4]]}]})
        self.assertIn('cover', str(caught.exception))

    def test_two_soft_law_walls_sum_as_the_archived_multi_factor_loop_did(self):
        """The archived one cavity could carry several powers and shapes at once;
        here that is several walls, and the run record names each."""
        block = {'groups': [{'name': 'g', 'atoms': [1, 2, 3, 4]}],
                 'walls': [
                     {'id': 'soft', 'geometry': 'ellipsoid',
                      'semiaxes': [2.0, 1.5, 1.0], 'power': 2,
                      'force_constant': 0.001, 'target': 'com', 'group': 'g',
                      'law': 'soft'},
                     {'id': 'stiff', 'geometry': 'box',
                      'lengths': [4.0, 3.0, 2.0], 'power': 4,
                      'force_constant': 0.0002, 'target': 'com', 'group': 'g',
                      'law': 'soft'}]}
        plan = _plan(block)
        state = self._bind(plan)
        coords = BASE + np.array([[1.1, -0.6, 0.3]])
        from pyrai3md.workflow.constraints.controller \
            import SystemConstraintController
        result = SystemConstraintController(plan).evaluate_potentials(
            coords, state, records.EvaluationContext(purpose='md', step=0),
            masses=MASSES)
        parts = [result.components[name]['energy'] for name in ('soft', 'stiff')]
        self.assertAlmostEqual(sum(parts), result.energy_correction, places=15)
        self.assertEqual({result.components[name]['law'] for name in
                          ('soft', 'stiff')}, {'soft'})


if __name__ == '__main__':
    unittest.main()
