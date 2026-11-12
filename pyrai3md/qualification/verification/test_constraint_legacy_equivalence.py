"""
======================================================================
Tests for the numeric relationship with the legacy constraint module.

legacy/PyRAI3MD/Molecule/constraint.py is a PyRAI2MD module in a keyword-dict
style, with exit() calls and hardcoded unit factors, so it cannot be imported
and should not be ported line for line. What has to hold is the arithmetic
mapping the migration table promises: the old force constant is twice the new
one because the old potential dropped the factor of one half, the old Bohr
factor is the repository's own constant to within its written precision, and
where the old and new behaviour genuinely differ, a test says so instead of
letting a migration assume they agree.

The legacy formulas are reimplemented inline here, from the file's own lines,
so a change in that archive cannot silently move this comparison.

Attribution: the legacy module is PyRAI2MD code by Jingbai Li (header dated
Sep 27 2022), consulted as a behaviour reference; no code is copied from it.

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
from pyrai3md.workflow.constraints import bias
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

# The literal legacy uses where it scales a gradient by the Bohr radius.
LEGACY_BOHR = 0.529177249

COORDS = np.array([[0.00, 0.00, 0.00],
                   [1.90, 0.30, 0.10],
                   [0.40, 2.00, -0.20],
                   [2.10, 1.80, 1.30]])


def _bond_plan():
    """A plan holding one bond variable and no action, for unit arithmetic."""
    return records.ConstraintPlan(
        cvs={'r': records.CVRecord('r', 'bond', [0, 1], unit='bohr')})


def _bias(force_constant, target, region='all', unit='bohr'):
    mapping = {'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
               'biases': [{'cv': 'r', 'target': target,
                           'force_constant': force_constant,
                           'region': region}]}
    plan = schema.build_plan(mapping, atoms=['O', 'H', 'H', 'C'], natom=4)
    spec = plan.spec(plan.ids('bias')[0])
    return plan, spec


class TestForceConstantConvention(unittest.TestCase):
    """V_legacy = f * d^2 against V_new = 1/2 * k * d^2, so k = 2 f."""

    def test_the_energies_agree_when_the_constant_is_doubled(self):
        """V_legacy = f d^2 and V_new = 1/2 k d^2 are the same function at
        k = 2f, checked in internal units so no unit rule interferes."""
        f_legacy = 0.02 / ANGSTROM_TO_BOHR ** 2
        plan = _bond_plan()
        delta = 0.35 * ANGSTROM_TO_BOHR
        current = float(np.linalg.norm(COORDS[0] - COORDS[1]))
        spec = records.ConstraintSpec('migrated', 'bias', 'harmonic', cv='r',
                                      target=current - delta, region='all',
                                      parameters={'force_constant':
                                                  2.0 * f_legacy})
        result = bias.evaluate(spec, plan, COORDS, records.ConstraintState(),
                               records.EvaluationContext())
        self.assertAlmostEqual(result['energy'], f_legacy * delta ** 2,
                               places=14)
        self.assertAlmostEqual(result['energy'],
                               0.5 * (2.0 * f_legacy) * delta ** 2,
                               places=14)

    def test_the_gradient_agrees_too(self):
        """dV/dq is 2 f d for the legacy form and k d for the new one.

        The spec is built directly in internal units here: routing through the
        schema would convert the declared constant once more, and this test is
        about the factor of two, not about the unit rule.
        """
        f_legacy = 0.02 / ANGSTROM_TO_BOHR ** 2
        plan = _bond_plan()
        spec = records.ConstraintSpec('migrated', 'bias', 'harmonic', cv='r',
                                      target=1.0, region='all',
                                      parameters={'force_constant':
                                                  2.0 * f_legacy})
        result = bias.evaluate(spec, plan, COORDS, records.ConstraintState(),
                               records.EvaluationContext())
        delta = result['cv_value'] - spec.target
        vector = COORDS[0] - COORDS[1]
        unit = vector / np.linalg.norm(vector)
        np.testing.assert_allclose(result['gradient'][0],
                                   2.0 * f_legacy * delta * unit, atol=1.0e-12)
        np.testing.assert_allclose(result['gradient'][1],
                                   -2.0 * f_legacy * delta * unit,
                                   atol=1.0e-12)

    def test_a_migrated_constant_halves_the_stiffness_if_not_doubled(self):
        """The trap the table exists to prevent, shown as a factor of two."""
        plan, spec = _bias(force_constant=0.02, target=1.0)
        result = bias.evaluate(spec, plan, COORDS, records.ConstraintState(),
                               records.EvaluationContext())
        doubled, again = _bias(force_constant=0.04, target=1.0)
        second = bias.evaluate(again, doubled, COORDS,
                               records.ConstraintState(),
                               records.EvaluationContext())
        self.assertAlmostEqual(second['energy'], 2.0 * result['energy'],
                               places=12)


class TestUnitConstants(unittest.TestCase):

    def test_the_repository_constant_supersedes_the_legacy_literal(self):
        """The legacy literal is right to about eight digits, no further.

        Worth pinning because a migration that kept the literal would carry a
        7e-8 relative error into every gradient the old code scaled, and the
        repository constant is the one everything else already agrees with.
        """
        relative = abs(BOHR_TO_ANGSTROM - LEGACY_BOHR) / BOHR_TO_ANGSTROM
        self.assertLess(relative, 1.0e-7)
        self.assertGreater(relative, 0.0)

    def test_the_inverse_is_the_reciprocal_not_a_second_literal(self):
        self.assertAlmostEqual(ANGSTROM_TO_BOHR * BOHR_TO_ANGSTROM, 1.0,
                               places=14)


class TestPeriodicFolding(unittest.TestCase):
    """Legacy folded by hand and took the sign from the target."""

    @staticmethod
    def _legacy_fold(value, target):
        dr = value - target
        if abs(dr) > 180.0:
            dr = (360.0 - abs(dr)) * np.sign(target)
        return dr

    def test_agreement_on_the_ordinary_case(self):
        for value, target in ((30.0, 10.0), (100.0, 90.0), (5.0, 1.0)):
            self.assertAlmostEqual(
                np.degrees(bias.cv.periodic_difference(np.radians(value),
                                                       np.radians(target))),
                self._legacy_fold(value, target), places=9)

    def test_legacy_sign_follows_the_target_not_the_crossing(self):
        """Across the branch the shortest way round can be negative while the
        legacy fold returns +|delta|, because its sign comes from the target
        number rather than from which way the torsion has to turn."""
        value, target = 350.0, 10.0
        mine = np.degrees(bias.cv.periodic_difference(np.radians(value),
                                                      np.radians(target)))
        legacy = self._legacy_fold(value, target)
        self.assertAlmostEqual(mine, -20.0, places=9)
        self.assertAlmostEqual(legacy, 20.0, places=9)

    def test_a_target_of_opposite_sign_flips_the_legacy_answer_again(self):
        value, target = 345.0, 15.0
        mine = np.degrees(bias.cv.periodic_difference(np.radians(value),
                                                      np.radians(target)))
        legacy = self._legacy_fold(value, target)
        self.assertAlmostEqual(mine, -30.0, places=9)
        self.assertAlmostEqual(legacy, 30.0, places=9)

    def test_the_difference_is_always_the_short_way_round(self):
        for value in (-180.0, -95.0, -1.0, 0.0, 1.0, 95.0, 179.0):
            for target in (-175.0, -20.0, 0.0, 40.0, 178.0):
                delta = np.degrees(bias.cv.periodic_difference(
                    np.radians(value), np.radians(target)))
                self.assertLessEqual(abs(delta), 180.0 + 1.0e-9)


class TestCavityAgainstWall(unittest.TestCase):
    """The archived cavity is nonzero inside, so it is not a *confining* wall.

    That is the statement this class made before `wall_law soft` existed. The
    law that covers the cavity is in `TestCavityLawIsTheSoftLaw` below; what
    stays true is that no exponent on the confining form produces the cavity,
    because the confining form is zero on one side of its surface by definition.
    """

    @staticmethod
    def _legacy_cavity(point, centre, radius, factor):
        """V_i = F_i ** (alpha / 2) with F_i = ((x - x0) / R) ** 2."""
        alpha = max(factor, 2)
        total = 0.0
        for axis in range(3):
            force = ((point[axis] - centre[axis]) / radius) ** 2
            total += force ** (alpha / 2.0)
        return total

    def test_the_legacy_form_has_energy_inside_the_boundary(self):
        centre = np.zeros(3)
        inside = np.array([0.1, 0.0, 0.0])
        legacy = self._legacy_cavity(inside, centre, 5.0, 40)
        self.assertGreater(legacy, 0.0)

    def test_a_confining_wall_is_exactly_zero_inside(self):
        from pyrai3md.workflow.constraints import wall
        spec = records.ConstraintSpec(
            'w', 'wall', 'sphere', atoms=[0],
            parameters={'geometry': 'sphere', 'power': 20,
                        'force_constant': 1.0, 'target_kind': 'atom',
                        'radius': 5.0, 'center': [0.0, 0.0, 0.0]})
        result = wall.evaluate(spec, None, np.array([[0.1, 0.0, 0.0]]),
                               records.ConstraintState(),
                               records.EvaluationContext(), masses=None)
        self.assertEqual(result['energy'], 0.0)

    def test_so_a_big_exponent_is_not_the_way_to_migrate_it(self):
        """The cavity is a different law, not a wall with a stiff exponent: the
        two agree on nothing inside the surface, which is where a cavity spends
        most of a trajectory."""
        inside = self._legacy_cavity(np.array([0.1, 0.0, 0.0]), np.zeros(3),
                                     5.0, 40)
        at_centre = self._legacy_cavity(np.zeros(3), np.zeros(3), 5.0, 40)
        self.assertEqual(at_centre, 0.0)
        self.assertGreater(inside, 0.0)


class TestCavityLawIsTheSoftLaw(unittest.TestCase):
    """`wall_law soft` is the archived cavity, so the cavity can retire.

    The formulas are written out here from `legacy/PyRAI3MD/Molecule/constraint.py`
    rather than called through the module, because the module is the thing being
    retired: a comparison that imports it would delete itself along with it.

    One number differs, deliberately. The archived code counted the centroid term
    once **per atom of the molecule** while its gradient counted it once (defect
    D-15 of design-document section 18), so a migrated run's energy is the new one
    times the molecule size and its force is the new one exactly. The wall keeps
    the force and drops the inconsistent factor; these tests measure the ratio
    instead of hiding it.
    """

    CAVITY = np.array([[0.00, 0.00, 0.00],
                       [1.80, 0.30, 0.10],
                       [-0.40, 1.90, -0.20],
                       [6.10, 0.20, 1.30],
                       [7.90, 0.90, 1.10],
                       [6.40, 2.20, 0.20]])
    MASSES = np.array([15.0, 1.0, 1.0, 15.0, 1.0, 1.0])
    RADII = np.array([3.0, 2.5, 2.0])
    GROUPS = ([0, 1, 2], [3, 4, 5])
    SIZE = 3
    CONSTANT = 1.0e-3

    @classmethod
    def _center(cls):
        weights = cls.MASSES / float(cls.MASSES.sum())
        return np.sum(cls.CAVITY * weights[:, None], axis=0)

    @classmethod
    def _legacy(cls, shape, power, gradient=False):
        """The archived sum, including its once-per-atom repeat.

        Each term is one `F` of the archived code, and the gradient is the
        archived chain rule `pre * alpha * F^(alpha/2 - 1) * (X - X0) / R^2`: an
        ellipsoid has one `F` for the molecule, so all three axes carry the same
        factor, while a cuboid has one per axis and each axis carries only its own.
        """
        centre = cls._center()
        alpha = max(power, 2)
        energy = 0.0
        push = np.zeros_like(cls.CAVITY)
        for group in cls.GROUPS:
            weights = cls.MASSES[group] / float(cls.MASSES[group].sum())
            point = np.sum(cls.CAVITY[group] * weights[:, None], axis=0)
            offset = point - centre
            scaled = offset / cls.RADII
            inverse_square = offset / cls.RADII ** 2
            if shape == 'ellipsoid':
                items = [(float(np.sum(scaled * scaled)), inverse_square)]
            else:
                items = [(float(scaled[axis] * scaled[axis]),
                          np.array([0.0 if other != axis else inverse_square[other]
                                    for other in range(3)]))
                         for axis in range(3)]
            for term, slope in items:
                energy += cls.CONSTANT * term ** (alpha / 2.0) * len(group)
                if gradient:
                    # The archived gradient has no atom count in it: it scales
                    # each atom by `m_i / sum(m)` alone, which is the half of
                    # defect D-15 - the energy the archived code reported is not
                    # the derivative of the force it reported.
                    scale = (cls.CONSTANT * alpha
                             * term ** (alpha / 2.0 - 1.0))
                    push[group] += scale * slope * weights[:, None]
        return (energy, push) if gradient else energy

    @classmethod
    def _wall(cls, shape, power):
        """The same run as one wall of the 'soft' law.

        A cuboid's axis scale is the wall's half edge, so the cavity radius is
        the box length divided by two; the ellipsoid keeps the radii as semiaxes.
        """
        from pyrai3md.workflow.constraints import wall
        geometry = 'ellipsoid' if shape == 'ellipsoid' else 'box'
        parameters = {'geometry': geometry, 'power': power, 'law': 'soft',
                      'target_kind': 'com',
                      'force_constant': cls.CONSTANT,
                      'molecules': [list(group) for group in cls.GROUPS],
                      'center': cls._center().tolist()}
        if shape == 'ellipsoid':
            parameters['semiaxes'] = cls.RADII.tolist()
        else:
            parameters['lengths'] = (2.0 * cls.RADII).tolist()
        spec = records.ConstraintSpec('cavity_as_wall', 'wall', geometry,
                                      atoms=[0, 1, 2, 3, 4, 5],
                                      parameters=parameters)
        plan = records.ConstraintPlan(actions=[spec])
        state = records.ConstraintState()
        result = wall.evaluate(spec, plan, cls.CAVITY, state,
                               records.EvaluationContext(),
                               masses=cls.MASSES)
        return spec, plan, result

    def test_the_energy_is_the_wall_energy_times_the_molecule_size(self):
        """Measured on two molecules of three atoms: the ratio is exactly 3."""
        for shape in ('ellipsoid', 'cuboid'):
            for power in (2, 4, 6):
                legacy = self._legacy(shape, power)
                _spec, _plan, result = self._wall(shape, power)
                self.assertAlmostEqual(
                    legacy, self.SIZE * result['energy'], places=15,
                    msg='%s p=%d' % (shape, power))

    def test_the_force_needs_no_factor_because_the_archived_one_had_none(self):
        """The archived gradient dropped the repeat its energy carried, so the
        two are consistent only once the energy drops it too: the wall's force is
        the cavity's, not a third of it."""
        for shape in ('ellipsoid', 'cuboid'):
            for power in (2, 4, 6):
                _energy, legacy = self._legacy(shape, power, gradient=True)
                _spec, _plan, result = self._wall(shape, power)
                worst = float(np.max(np.abs(legacy - result['gradient'])))
                self.assertLess(worst, 1.0e-14,
                                msg='%s p=%d differs by %.3e'
                                % (shape, power, worst))

    def test_a_cuboid_radius_is_a_box_half_edge(self):
        """The cavity's radius is the extent an axis allows, which is half of the
        box edge the wall declares. Spelling the same number as a full edge
        divides every axis ratio by two, and at p = 4 that is worth exactly
        sixteen times the energy - a slip in this conversion is not subtle."""
        _spec, _plan, right = self._wall('cuboid', 4)
        self.assertAlmostEqual(self._legacy('cuboid', 4),
                               self.SIZE * right['energy'], places=15)
        from pyrai3md.workflow.constraints import wall
        wrong = records.ConstraintSpec(
            'wrong', 'wall', 'box', atoms=[0, 1, 2, 3, 4, 5],
            parameters={'geometry': 'box', 'power': 4, 'law': 'soft',
                        'target_kind': 'com',
                        'force_constant': self.CONSTANT,
                        'molecules': [list(group) for group in self.GROUPS],
                        'center': self._center().tolist(),
                        'lengths': self.RADII.tolist()})
        got = wall.evaluate(wrong, records.ConstraintPlan(actions=[wrong]),
                            self.CAVITY, records.ConstraintState(),
                            records.EvaluationContext(), masses=self.MASSES)
        self.assertAlmostEqual(got['energy'], 16.0 * right['energy'],
                               places=12)

    def test_the_soft_law_is_two_sided(self):
        """A point pulled through the center pays what it pays on the far side of
        it, which is the difference from a boundary and the reason this law needs
        no face clamp."""
        from pyrai3md.workflow.constraints import wall
        centre = self._center()
        spec = records.ConstraintSpec(
            'spring', 'wall', 'sphere', atoms=[0],
            parameters={'geometry': 'sphere', 'power': 2, 'law': 'soft',
                        'target_kind': 'atom', 'force_constant': 1.0,
                        'radius': 2.0, 'center': centre.tolist()})
        plan = records.ConstraintPlan(actions=[spec])
        base = centre.reshape(1, 3)
        forward = wall.evaluate(spec, plan, base + np.array([[1.0, 0.0, 0.0]]),
                                records.ConstraintState(),
                                records.EvaluationContext())
        back = wall.evaluate(spec, plan, base - np.array([[1.0, 0.0, 0.0]]),
                             records.ConstraintState(),
                             records.EvaluationContext())
        self.assertAlmostEqual(forward['energy'], back['energy'], places=15)
        self.assertAlmostEqual(forward['gradient'][0, 0],
                               -back['gradient'][0, 0], places=15)

    def test_the_soft_law_hessian_matches_finite_differences(self):
        """Both shapes at p = 2, 4 and 6, checked against a central difference of
        the reported gradient at four step sizes. The worst disagreement measured
        is 6.9e-10, and it is at the coarsest step: every row shrinks as the step
        shrinks, which is what a correct analytic block looks like. A wrong `J J^T
        term instead shows up as a step-independent gap, and it was the p = 4 rows
        that found one here."""
        from pyrai3md.workflow.constraints import wall
        for shape in ('ellipsoid', 'cuboid'):
            for power in (2, 4, 6):
                spec, plan, _result = self._wall(shape, power)
                for step in (1.0e-3, 3.0e-4, 1.0e-4, 1.0e-5):
                    block = wall.hessian(
                        spec, plan, self.CAVITY, records.ConstraintState(),
                        records.EvaluationContext(),
                        masses=self.MASSES)['hessian']
                    worst = 0.0
                    for atom in range(6):
                        for axis in range(3):
                            row = np.zeros_like(self.CAVITY)
                            row[atom, axis] = step
                            forward = wall.evaluate(
                                spec, plan, self.CAVITY + row,
                                records.ConstraintState(),
                                records.EvaluationContext(),
                                masses=self.MASSES)['gradient']
                            back = wall.evaluate(
                                spec, plan, self.CAVITY - row,
                                records.ConstraintState(),
                                records.EvaluationContext(),
                                masses=self.MASSES)['gradient']
                            numeric = ((forward - back) / (2.0 * step)
                                       ).reshape(-1)
                            worst = max(worst, float(np.max(np.abs(
                                numeric - block[atom * 3 + axis]))))
                    self.assertLess(
                        worst, 1.0e-9,
                        msg='%s p=%d step=%.0e differs by %.3e'
                        % (shape, power, step, worst))

    def test_a_triclinic_wall_has_no_soft_law(self):
        """The law divides by an axis scale, and a sheared cell gives none; the
        refusal names the geometry rather than quietly reading the rows as
        lengths."""
        from pyrai3md.workflow.constraints import wall
        spec = records.ConstraintSpec(
            'shear', 'wall', 'triclinic', atoms=[0],
            parameters={'geometry': 'triclinic', 'power': 2, 'law': 'soft',
                        'target_kind': 'atom', 'force_constant': 1.0,
                        'cell': [[2.0, 0.0, 0.0], [0.4, 1.8, 0.0],
                                 [0.1, 0.2, 1.7]],
                        'center': [0.0, 0.0, 0.0]})
        with self.assertRaises(wall.WallError):
            wall.evaluate(spec, records.ConstraintPlan(actions=[spec]),
                          np.array([[0.4, 0.2, 0.1]]),
                          records.ConstraintState(),
                          records.EvaluationContext())

    def test_a_soft_law_wall_refuses_a_periodic_cell(self):
        """There is no shell to check and nothing confines, and a wrapped
        coordinate would jump the offset by a whole lattice vector at every
        face."""
        from pyrai3md.workflow.constraints import wall
        spec, plan, _result = self._wall('ellipsoid', 2)
        cell = np.diag([40.0, 40.0, 40.0])
        with self.assertRaises(wall.WallError) as caught:
            wall.periodic_check(spec, plan, self.CAVITY, cell,
                                center=self._center().tolist())
        self.assertIn('soft', str(caught.exception))


class TestMatchListBroadcast(unittest.TestCase):
    """Legacy padded a short target list by repeating its last value."""

    @staticmethod
    def _legacy_match(targets, values):
        values = list(values)
        if len(values) < len(targets):
            values += [values[-1]] * (len(targets) - len(values))
        return values[:len(targets)]

    def test_the_legacy_rule_would_have_silently_invented_targets(self):
        got = self._legacy_match([1, 2, 3], [1.5])
        self.assertEqual(got, [1.5, 1.5, 1.5])

    def test_the_new_input_refuses_a_mismatched_count(self):
        with self.assertRaises(schema.ConstraintInputError):
            schema.build_plan(
                {'cvs': [{'name': 'a', 'kind': 'bond', 'atoms': [1, 2]},
                         {'name': 'b', 'kind': 'bond', 'atoms': [2, 3]},
                         {'name': 'c', 'kind': 'bond', 'atoms': [3, 4]}],
                 'metadynamics': [{'cvs': ['a', 'b', 'c'],
                                   'height': 1.0e-4, 'width': [0.1, 0.2],
                                   'pace': 10}]},
                atoms=['O', 'H', 'H', 'C'], natom=4)


if __name__ == '__main__':
    unittest.main()
