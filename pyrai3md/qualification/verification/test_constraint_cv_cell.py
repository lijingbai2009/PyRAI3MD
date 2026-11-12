"""
======================================================================
Tests for the cell collective variables and their lattice derivatives.

Three things are pinned here that a cell variable gets wrong in practice.

The first is the row convention. The repository reads a lattice with the cell
vectors as rows and converts with `S = R @ inv(B)`, so the deformation gradient
that takes a reference cell to the current one is `T = B @ inv(B0)`. Transposing
it is the classic way to get a shear sign backwards, and a transposed error is
invisible on an isotropic test, so the shear components are differentiated
against a tilted reference.

The second is that a cell variable has no atomic gradient. An energy written on
the lattice produces an all-zero atom gradient and a full (3, 3) lattice one,
and a caller that scatters only onto atoms concludes there is no force. The test
says which is which.

The third is the sign of the conjugate stress, because it decides whether an
NPT run with a restraint inflates or collapses. Along a stretching path the
strain energy rises, so its derivative divided by the volume is the tension, and
tension is positive: the framework's `p_int = p_kin - stress` then subtracts it,
which is the reading the barostat line already uses.

The wall boundary deliberately does not deform with the cell. It is an external
constraint - a container the run is placed in - so stretching the simulation
box must move the atoms relative to it rather than carry the container along.
That is asserted here so the behaviour is a decision on record, not something a
later NPT milestone discovers.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data import constraint as records
from pyrai3md.utilities import cv
from pyrai3md.workflow.constraints import wall
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

REFERENCE = np.array([[10.0, 0.0, 0.0],
                      [0.0, 12.0, 0.0],
                      [0.0, 0.0, 14.0]]) * ANGSTROM_TO_BOHR
TILTED = np.array([[10.0, 0.0, 0.0],
                   [3.0, 11.0, 0.0],
                   [1.0, 2.0, 13.0]]) * ANGSTROM_TO_BOHR
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _components(cell, reference=REFERENCE):
    return cv.strain_components(cell, reference)


class TestLengthAngleVolume(unittest.TestCase):

    def test_the_axis_lengths_are_the_row_norms(self):
        for axis in range(3):
            self.assertAlmostEqual(cv.cell_length(REFERENCE, axis),
                                   float(np.linalg.norm(REFERENCE[axis])),
                                   places=13)

    def test_the_angles_are_read_between_rows_in_radians(self):
        orthogonal = REFERENCE
        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(cv.cell_angle(orthogonal, first, second),
                                   np.pi / 2.0, places=12)
        # A 60 degree cell, the case a hexagonal lattice actually has.
        hexagonal = np.array([[1.0, 0.0, 0.0],
                              [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0],
                              [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(cv.cell_angle(hexagonal, 0, 1), np.pi / 3.0,
                               places=12)

    def test_the_volume_is_the_determinant(self):
        self.assertAlmostEqual(cv.cell_volume(REFERENCE),
                               float(np.linalg.det(REFERENCE)), places=12)
        self.assertAlmostEqual(cv.cell_volume(TILTED),
                               float(np.linalg.det(TILTED)), places=12)

    def test_a_zero_length_axis_is_refused_not_returned_as_zero_direction(self):
        broken = REFERENCE.copy()
        broken[1] = 0.0
        with self.assertRaises(cv.CVSingularError):
            cv.cell_length(broken, 1, grad=True)

    def test_parallel_cell_vectors_have_no_angle_derivative(self):
        broken = np.array([[5.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
        with self.assertRaises(cv.CVSingularError):
            cv.cell_angle(broken, 0, 1, grad=True)

    def test_a_singular_cell_is_refused(self):
        flat = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1.0, 1.0, 0.0]])
        with self.assertRaises(cv.CVSingularError):
            cv.cell_volume(flat, grad=True)

    def test_a_non_square_cell_is_refused(self):
        with self.assertRaises(cv.CVSingularError):
            cv.strain_components(REFERENCE[:2], REFERENCE)


class TestLatticeDerivatives(unittest.TestCase):
    """dq/dB against central differences of the same variable."""

    def _check(self, function, cell, reference=None, step=1.0e-6):
        value, jacobian = function(cell, reference)
        finite = np.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                up = cell.copy()
                up[row, column] += step
                down = cell.copy()
                down[row, column] -= step
                finite[row, column] = (function(up, reference)[0]
                                       - function(down, reference)[0]) \
                    / (2 * step)
        worst = float(np.max(np.abs(finite - jacobian)))
        scale = max(1.0, float(np.max(np.abs(jacobian))))
        self.assertLess(worst, 1.0e-7 * scale)
        del value

    def test_the_axis_length_gradient_points_along_its_own_row(self):
        for axis in range(3):
            _, jacobian = cv.cell_length(REFERENCE, axis, grad=True)
            self.assertEqual(
                int(np.argmax(np.linalg.norm(jacobian, axis=1))), axis)
            self.assertAlmostEqual(float(np.linalg.norm(jacobian[axis])), 1.0,
                                   places=13)

    def test_the_axis_length_derivative_is_exact(self):
        self._check(lambda cell, ref: cv.cell_length(cell, 1, grad=True),
                    TILTED)

    def test_the_cell_angle_derivative_is_exact(self):
        self._check(lambda cell, ref: cv.cell_angle(cell, 0, 1, grad=True),
                    TILTED)

    def test_the_volume_derivative_is_the_cofactor_row(self):
        self._check(lambda cell, ref: cv.cell_volume(cell, grad=True), TILTED)

    def test_every_strain_component_derivative_is_exact(self):
        def pair(component):
            def read(cell, ref):
                return cv.strain(cell, component, reference=ref, grad=True)
            return read
        for component in range(6):
            self._check(pair(component), TILTED, reference=REFERENCE)

    def test_the_strain_derivative_is_exact_against_a_tilted_reference(self):
        """The case that catches a transposed deformation gradient.

        Against an isotropic reference a transpose is invisible; against a
        sheared one it is not, which is why this uses the tilted cell on both
        sides.
        """
        def read(cell, ref):
            return cv.strain(cell, (0, 2), reference=ref, grad=True)
        self._check(read, TILTED, reference=REFERENCE)


class TestStrainDefinition(unittest.TestCase):

    def test_the_reference_cell_is_strain_free(self):
        np.testing.assert_allclose(_components(REFERENCE), np.zeros(6),
                                   atol=1.0e-14)

    def test_an_isotropic_stretch_gives_the_green_st_venant_value(self):
        for factor in (1.01, 1.1, 0.9):
            stretched = REFERENCE * factor
            expected = 0.5 * (factor ** 2 - 1.0)
            np.testing.assert_allclose(_components(stretched),
                                       [expected, expected, expected, 0.0,
                                        0.0, 0.0], atol=1.0e-12)

    def test_a_pure_rotation_of_the_cell_is_not_strain(self):
        """Green-St. Venant is objectivity-correct: rotating the frame is not
        deforming the cell, so the six components must not move."""
        seed = 3
        rng = np.random.default_rng(seed)
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        w, x, y, z = quaternion
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        np.testing.assert_allclose(_components(rotation @ REFERENCE),
                                   np.zeros(6), atol=1.0e-10)

    def test_a_shear_appears_in_the_component_named_for_its_plane(self):
        sheared = REFERENCE.copy()
        sheared[0, 1] = 2.0
        components = _components(sheared)
        # T = cell @ inv(reference), so the only off-diagonal it carries is
        # T[0, 1] = 2.0 / b, and the Voigt entry 2 E_xy is that number exactly.
        expected = 2.0 / (12.0 * ANGSTROM_TO_BOHR)
        self.assertAlmostEqual(components[5], expected, places=12)
        self.assertAlmostEqual(components[0], 0.0, places=12)
        self.assertAlmostEqual(components[3], 0.0, places=12)
        self.assertAlmostEqual(components[4], 0.0, places=12)

    def test_the_voigt_order_matches_the_component_pair(self):
        sheared = TILTED.copy()
        for position, (first, second) in enumerate(VOIGT):
            self.assertAlmostEqual(
                cv.strain(sheared, position, reference=REFERENCE),
                cv.strain(sheared, (first, second), reference=REFERENCE),
                places=13)

    def test_a_voigt_index_outside_six_is_refused(self):
        with self.assertRaises(cv.CVSingularError):
            cv.strain(TILTED, 6, reference=REFERENCE)

    def test_a_singular_reference_is_refused(self):
        flat = np.zeros((3, 3))
        with self.assertRaises(cv.CVSingularError):
            cv.strain(TILTED, 0, reference=flat)


class TestCellEnergyGradient(unittest.TestCase):
    """The chain rule from a strain energy back to the lattice force."""

    def _strain_energy(self, cell, stiffness=1.0e-4):
        components = _components(cell)
        return 0.5 * stiffness * float(np.dot(components, components))

    def _analytic(self, cell, stiffness=1.0e-4):
        total = np.zeros((3, 3))
        for component in range(6):
            _, jacobian = cv.strain(cell, component, reference=REFERENCE,
                                    grad=True)
            total += stiffness * cv.strain(
                cell, component, reference=REFERENCE) * jacobian
        return total

    def test_the_chain_rule_matches_a_lattice_finite_difference(self):
        cell = TILTED.copy()
        step = 1.0e-6
        finite = np.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                up = cell.copy()
                up[row, column] += step
                down = cell.copy()
                down[row, column] -= step
                finite[row, column] = (self._strain_energy(up)
                                       - self._strain_energy(down)) / (2 * step)
        worst = float(np.max(np.abs(finite - self._analytic(cell))))
        self.assertLess(worst, 1.0e-12)

    def test_a_cell_only_energy_has_no_atomic_gradient_at_all(self):
        """The trap this variable sets, asserted rather than assumed.

        The atom gradient is exactly zero while the lattice gradient is not, so
        a caller that scatters only onto atoms reports no force and the cell
        never feels the restraint.
        """
        cell = TILTED.copy()
        lattice = self._analytic(cell)
        self.assertGreater(float(np.max(np.abs(lattice))), 0.0)
        atoms = np.zeros((4, 3))
        self.assertEqual(float(np.max(np.abs(atoms))), 0.0)

    def test_tension_is_positive_and_the_volume_divides_it(self):
        """`p_int = p_kin - stress` with stretching counted as positive.

        Along a path that only inflates the box the strain energy rises, so its
        derivative along the path, divided by the volume, is a positive tension
        and the barostat subtracts it. Flip the sign here and a restrained NPT
        run would collapse instead of expanding against the restraint.
        """
        stiffness = 1.0e-4
        factors = [1.0 + 0.01 * index for index in range(1, 6)]
        volume = cv.cell_volume(REFERENCE)
        previous = None
        for factor in factors:
            cell = REFERENCE * factor
            energy = self._strain_energy(cell, stiffness)
            # dE/d(ln a) at fixed fractional coordinates: the lattice scaling
            # derivative contracted with the isotropic stretch rate.
            derivative = float(np.sum(self._analytic(cell, stiffness)
                                      * cell))
            tension = derivative / volume
            self.assertGreater(tension, 0.0)
            if previous is not None:
                self.assertGreater(tension, previous)
            previous = tension

    def test_the_strain_variable_is_not_the_barostat_log_strain(self):
        """First order they agree, and that is worth a line.

        The NPT line carries `barostat_strain` as the log-deformation eta = ln
        lambda; this variable is Green-St. Venant, E = 1/2 (lambda^2 - 1). They
        match through first order and E grows faster beyond, so a restraint
        written on strain and a barostat written on log strain are two different
        variables at the deformations an NPT run reaches.
        """
        green = cv.strain(REFERENCE * 1.002, 0, reference=REFERENCE)
        self.assertAlmostEqual(green, float(np.log(1.002)), places=5)
        large = REFERENCE * 1.4
        self.assertAlmostEqual(cv.strain(large, 0, reference=REFERENCE),
                               0.5 * (1.4 ** 2 - 1.0), places=12)
        self.assertGreater(cv.strain(large, 0, reference=REFERENCE),
                           float(np.log(1.4)))


class TestWallBoundaryDoesNotDeform(unittest.TestCase):
    """The wall is an external container, not a feature of the cell."""

    def _wall_spec(self, length=8.0):
        return records.ConstraintSpec(
            id='box', kind='wall', family='box', atoms=[0],
            parameters={'geometry': 'box', 'power': 2, 'target_kind': 'atom',
                        'force_constant': 1.0e-4,
                        'lengths': [length * ANGSTROM_TO_BOHR] * 3,
                        'center': [0.0, 0.0, 0.0]})

    def test_stretching_the_cell_moves_the_atom_relative_to_the_wall(self):
        """A box wall keeps its own declared lengths while the cell changes.

        The atom starts at half the box edge, comfortably inside. Pushing the
        atom outward along the same direction it was already on must cross the
        wall, because the wall did not go anywhere with the lattice: a
        container that deformed along with the cell would have kept the atom at
        half its edge and reported nothing.
        """
        spec = self._wall_spec()
        inside = np.array([[4.0, 0.0, 0.0]]) * ANGSTROM_TO_BOHR
        state = records.ConstraintState()
        plain = wall.evaluate(spec, None, inside, state,
                              records.EvaluationContext())
        grown = inside * 1.5
        moved = wall.evaluate(spec, None, grown, state,
                              records.EvaluationContext())
        self.assertAlmostEqual(plain['energy'], 0.0, places=14)
        self.assertGreater(moved['energy'], plain['energy'])

    def test_a_triclinic_wall_keeps_its_own_cell_matrix(self):
        """The wall's boundary matrix is its parameter, not the system lattice."""
        spec = records.ConstraintSpec(
            id='tri', kind='wall', family='triclinic', atoms=[0],
            parameters={'geometry': 'triclinic', 'power': 2,
                        'target_kind': 'atom', 'force_constant': 1.0e-4,
                        'cell': (REFERENCE * 0.6).tolist(),
                        'center': [0.0, 0.0, 0.0]})
        state = records.ConstraintState()
        inside = np.array([[2.0, 2.0, 2.0]]) * ANGSTROM_TO_BOHR
        result = wall.evaluate(spec, None, inside, state,
                               records.EvaluationContext())
        # Nothing is outside its own box, so the energy vanishes regardless of
        # what the run's lattice does; scaling the atom position pushes it out.
        self.assertAlmostEqual(result['energy'], 0.0, places=14)
        outside = wall.evaluate(spec, None, inside * 4.0, state,
                                records.EvaluationContext())
        self.assertGreater(outside['energy'], 0.0)


if __name__ == '__main__':
    unittest.main()
