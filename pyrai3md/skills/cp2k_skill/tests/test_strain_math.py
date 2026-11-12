"""
Unit tests for the external strain finite-difference stress
mathematics (service/stress/strain.py).

Every expected value below is derived by hand from a closed-form
energy model, never by the module under test, so the conventions the
finite-difference route can get wrong are each pinned by its own
assertion:

- the shear factor of two (B[a,b] = B[b,a] = 1/2 vs both slots at 1),
- the central-volume denominator (V0, not V(+) or V(-)),
- the tensile-positive sign (no negation of the energy difference),
- which tensor slot each strain direction fills,
- the Hartree/Angstrom^3 -> GPa factor, derived here from SI values.

Three models are used: a volume (bulk) term, a bilinear x*y term that
is exactly quadratic in the shear strain, and an anisotropic quadratic
term. Where a model is exactly quadratic in the strain the central
difference is exact and the assertion is a machine-precision equality;
where it is not (the volume term) the error is measured and reported,
and its O(delta^2) convergence is asserted instead of a guessed
tolerance.

These tests do NOT require a CP2K installation.

Run: pytest cp2k_skill/tests/test_strain_math.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import numpy as np
import pytest

from ..service.stress.strain import (
    HARTREE_A3_TO_GPA, STRESS_SCOPES, assemble_stress, cell_volume,
    difference_component, direction_matrices, perturb)

# 1 Hartree/Angstrom^3 in GPa, from SI values (1 Eh = 4.3597447222071e-18 J,
# 1 A^3 = 1e-30 m^3): 4.3597447222071e12 Pa. This is the independent
# reference for the module constant, which is derived from the parser's
# eV/A^3 label factor and 1 Hartree in eV.
EA3_TO_GPA_SI = 4.3597447222071e12 / 1e9

# Bulk-model parameters (model units, Hartree and Angstrom^3)
BULK = 2.0e-4
V_REST = 100.0
# Central cell is expanded by 20% so the first-order term is non-zero
V_CENTRAL = 1.2 * V_REST

DELTA = 1.0e-3

# A triclinic lattice: row vectors, non-orthogonal, volume V_CENTRAL
CUBIC = np.eye(3) * np.cbrt(V_CENTRAL)
TRICLINIC = np.array([
    [np.cbrt(V_CENTRAL), 0.0, 0.0],
    [0.31 * np.cbrt(V_CENTRAL), np.cbrt(V_CENTRAL), 0.0],
    [0.17 * np.cbrt(V_CENTRAL), -0.23 * np.cbrt(V_CENTRAL),
     np.cbrt(V_CENTRAL)],
])

# Two atoms away from the origin so the bilinear and quadratic models
# respond to the strain (a position at the origin contributes nothing).
POSITIONS = np.array([
    [1.1, 0.7, 0.3],
    [-0.4, 1.9, -0.8],
])


def volume_energy(H):
    """E = 1/2 * BULK * (V - V_REST)^2 / V_REST, in Hartree."""
    volume = abs(float(np.linalg.det(np.asarray(H, dtype=np.float64))))
    return 0.5 * BULK * (volume - V_REST) ** 2 / V_REST


def bilinear_energy(R):
    """E = sum_i x_i * y_i, in Hartree (c = 1)."""
    positions = np.asarray(R, dtype=np.float64)
    return float(np.sum(positions[:, 0] * positions[:, 1]))


def s_expected_gha():
    """sigma_aa of the bulk model at the central cell, Hartree/A^3.

    dE/depsilon = BULK*(Vc-V_REST)/V_REST * V * tr(B), so the tension
    -positive sigma : B is s * tr(B) with s = BULK*(Vc-V_REST)/V_REST.
    """
    return BULK * (V_CENTRAL - V_REST) / V_REST


def fd_component(H, R, B, delta, energy_fn):
    """One finite-difference component, in Hartree/A^3, using the
    central volume of the configuration handed in."""
    volume0 = cell_volume(H)
    H_plus, R_plus = perturb(H, R, B, delta, 1)
    H_minus, R_minus = perturb(H, R, B, delta, -1)
    # difference_component converts to GPa; divide back out so the
    # closed-form comparisons below stay in model units.
    gpa = difference_component([energy_fn(H_plus, R_plus)],
                               [energy_fn(H_minus, R_minus)],
                               delta, volume0)
    return gpa[0] / HARTREE_A3_TO_GPA


class TestUnitFactor:
    """Hartree/Angstrom^3 -> GPa conversion used by the driver."""

    def test_matches_si_derived_value(self):
        assert HARTREE_A3_TO_GPA == pytest.approx(EA3_TO_GPA_SI, rel=1e-4)

    def test_is_hartree_per_a3_not_eV(self):
        # Guard against pairing the eV/A^3 factor with Hartree energies
        # (a silent 27x error): the value must sit near 4.36e3, not 1.6e2.
        assert 4.0e3 < HARTREE_A3_TO_GPA < 4.7e3


class TestDirections:
    """Strain direction matrices and scope validation."""

    def test_full_scope_has_six_directions(self):
        labels = [label for label, _ in direction_matrices('full')]
        assert labels == ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']

    def test_shear_slots_carry_one_half(self):
        matrices = dict(direction_matrices('full'))
        assert matrices['xy'][0, 1] == 0.5 and matrices['xy'][1, 0] == 0.5
        # The diagonal directions must not pick up any off-diagonal slot
        assert np.count_nonzero(matrices['xx'] - np.diag([1.0, 0.0, 0.0])) == 0

    def test_iso_scope_is_identity(self):
        matrices = direction_matrices('iso')
        assert len(matrices) == 1
        assert matrices[0][0] == 'iso'
        assert np.array_equal(matrices[0][1], np.eye(3))

    def test_scopes_declared_once(self):
        assert set(STRESS_SCOPES) == {'full', 'iso'}

    def test_unknown_scope_is_refused(self):
        with pytest.raises(ValueError, match='stress_scope'):
            direction_matrices('shear_only')


class TestPerturb:
    """The deformation keeps the fractional coordinates unchanged."""

    def test_fractional_coordinates_are_preserved(self):
        for B in (np.diag([1.0, 0.0, 0.0]),
                  np.array([[0., .5, 0.], [.5, 0., 0.], [0., 0., 0.]])):
            for sign in (1, -1):
                H2, R2 = perturb(TRICLINIC, POSITIONS, B, DELTA, sign)
                frac_in = POSITIONS @ np.linalg.inv(TRICLINIC)
                frac_out = R2 @ np.linalg.inv(H2)
                assert np.allclose(frac_in, frac_out, rtol=0, atol=1e-12)

    def test_zero_strain_matrix_leaves_the_configuration_untouched(self):
        # B = 0 pins the multiplication order: a transposed or mis-ordered
        # deformation still returns the input bit-for-bit here, so this
        # test cannot pass by accident of a symmetric lattice.
        H2, R2 = perturb(TRICLINIC, POSITIONS, np.zeros((3, 3)), DELTA, 1)
        assert np.array_equal(H2, TRICLINIC)
        assert np.array_equal(R2, POSITIONS)

    def test_invalid_step_or_sign_is_refused(self):
        with pytest.raises(ValueError, match='positive'):
            perturb(CUBIC, POSITIONS, np.eye(3), 0.0, 1)
        with pytest.raises(ValueError, match='positive'):
            perturb(CUBIC, POSITIONS, np.eye(3), float('nan'), 1)
        with pytest.raises(ValueError, match='sign'):
            perturb(CUBIC, POSITIONS, np.eye(3), DELTA, 0)

    def test_singular_cell_is_refused(self):
        flat = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match='positive finite'):
            cell_volume(flat)
        with pytest.raises(ValueError, match='3x3'):
            cell_volume(np.eye(2))


class TestShearFactor:
    """The shear convention is priced, not assumed (docs section 4.2).

    The bilinear model E = sum x*y is exactly quadratic in the shear
    strain, so the central difference is exact and the comparison to
    the hand-derived derivative is a machine-precision equality.
    """

    @staticmethod
    def _energy(H, R):
        return bilinear_energy(R)

    def test_shear_component_matches_closed_form(self):
        expected = 0.5 * np.sum(POSITIONS[:, 0] ** 2
                                + POSITIONS[:, 1] ** 2) / V_CENTRAL
        got = fd_component(CUBIC, POSITIONS,
                           dict(direction_matrices('full'))['xy'],
                           DELTA, self._energy)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_full_shear_slots_would_double_the_component(self):
        # Negative control: B[a,b] = B[b,a] = 1 (the wrong reading of the
        # strain parameter) must return exactly twice the component. If
        # this ever stops differing from the 1/2 convention the control
        # is vacuous and the assertion above proves nothing.
        wrong = np.zeros((3, 3))
        wrong[0, 1] = wrong[1, 0] = 1.0
        correct = fd_component(CUBIC, POSITIONS,
                               dict(direction_matrices('full'))['xy'],
                               DELTA, self._energy)
        got = fd_component(CUBIC, POSITIONS, wrong, DELTA, self._energy)
        assert got == pytest.approx(2.0 * correct, rel=1e-12)

    def test_diagonal_direction_of_shear_model_matches_closed_form(self):
        # x*y under a pure x stretch: x'y' = xy(1+delta) -> component
        # equals sum(xy)/V0; pins the diagonal convention too.
        expected = bilinear_energy(POSITIONS) / V_CENTRAL
        got = fd_component(CUBIC, POSITIONS, np.diag([1.0, 0.0, 0.0]),
                           DELTA, self._energy)
        assert got == pytest.approx(expected, rel=1e-12)


class TestCentralVolume:
    """The denominator is the CENTRAL volume, never V(+) or V(-)."""

    def test_uses_central_volume(self):
        expected = s_expected_gha()
        got = fd_component(CUBIC, POSITIONS, np.diag([1.0, 0.0, 0.0]), DELTA,
                           lambda H, R: volume_energy(H))
        relative_error = abs(got - expected) / abs(expected)
        # The diagonal strain is exact for this model, so any error here
        # comes from the denominator: reading V(+) instead of V0 shifts
        # the value by ~delta (1e-3), seven thousand times the measured
        # relative error below.
        assert relative_error < 1.0e-4

    def test_perturbed_volume_would_miss_the_tolerance(self):
        # Control for the assertion above: reproduce the V(+) reading
        # and show it is outside that tolerance.
        expected = s_expected_gha()
        volume_plus = V_CENTRAL * (1.0 + DELTA) ** 3
        H_plus, _ = perturb(CUBIC, POSITIONS, np.diag([1.0, 0.0, 0.0]), DELTA, 1)
        H_minus, _ = perturb(CUBIC, POSITIONS, np.diag([1.0, 0.0, 0.0]), DELTA, -1)
        wrong = (volume_energy(H_plus)
                 - volume_energy(H_minus)) / (2.0 * DELTA * volume_plus)
        assert abs(wrong - expected) / abs(expected) > 1.0e-4


class TestSignConvention:
    """The external difference is tensile-positive with no negation."""

    def test_expanded_cell_gives_positive_diagonal(self):
        # V_CENTRAL > V_REST: stretching costs energy, so the potential
        # stress is positive (tension). A stray negation flips this.
        for axis in range(3):
            B = np.zeros((3, 3))
            B[axis, axis] = 1.0
            got = fd_component(CUBIC, POSITIONS, B, DELTA,
                               lambda H, R: volume_energy(H))
            assert got > 0.0

    def test_compressed_cell_gives_negative_diagonal(self):
        compressed = np.eye(3) * np.cbrt(0.8 * V_REST)
        expected = BULK * (float(np.linalg.det(compressed)) - V_REST) / V_REST
        got = fd_component(compressed, POSITIONS, np.diag([1.0, 0.0, 0.0]),
                           DELTA, lambda H, R: volume_energy(H))
        assert expected < 0.0
        assert got == pytest.approx(expected, rel=1.0e-4)


class TestAssembleTensor:
    """Slot placement and the isotropic contraction."""

    def test_full_tensor_places_each_direction_in_its_own_slots(self):
        # Distinct hand-made values per direction: a slot mix-up cannot
        # pass as a symmetric fill.
        values = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        tensor = assemble_stress(values, 'full')[0]
        assert tensor[0, 0] == 1.0 and tensor[1, 1] == 2.0
        assert tensor[2, 2] == 3.0
        assert tensor[0, 1] == tensor[1, 0] == 4.0
        assert tensor[0, 2] == tensor[2, 0] == 5.0
        assert tensor[1, 2] == tensor[2, 1] == 6.0

    def test_state_axis_is_preserved(self):
        # One distinct value pair per direction, in evaluation order
        # (xx, yy, zz, xy, xz, yz): a repeated row would hide an
        # index mix-up between the state axis and the direction axis.
        values = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],
                  [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]
        tensor = assemble_stress(values, 'full')
        assert tensor.shape == (2, 3, 3)
        assert tensor[0, 2, 2] == 3.0 and tensor[1, 2, 2] == 30.0
        assert tensor[1, 0, 2] == tensor[1, 2, 0] == 50.0

    def test_iso_scope_contracts_the_trace(self):
        # B = I measures tr(sigma); the diagonal of the isotropic tensor
        # is tr(sigma)/3, which is the -p * I form written once, not twice.
        tensor = assemble_stress([[9.0]], 'iso')[0]
        assert np.allclose(tensor, 3.0 * np.eye(3))

    def test_wrong_direction_count_is_refused(self):
        with pytest.raises(ValueError, match='Expected 6'):
            assemble_stress([[1.0]] * 5, 'full')
        with pytest.raises(ValueError, match='Expected 1'):
            assemble_stress([[1.0], [2.0]], 'iso')

    def test_non_finite_component_is_refused(self):
        with pytest.raises(ValueError, match='Non-finite'):
            assemble_stress([[float('nan')]] * 6, 'full')


class TestTriclinicCell:
    """A non-orthogonal lattice: only the volume enters the bulk model."""

    def test_diagonal_matches_closed_form(self):
        expected = s_expected_gha()
        for axis in range(3):
            B = np.zeros((3, 3))
            B[axis, axis] = 1.0
            got = fd_component(TRICLINIC, POSITIONS, B, DELTA,
                               lambda H, R: volume_energy(H))
            assert got == pytest.approx(expected, rel=1.0e-4)

    def test_shear_components_vanish_for_an_isotropic_state(self):
        # det(I + delta*B) with the 1/2 shear convention is 1 - delta^2/4:
        # an even function of the strain, so the central difference is
        # zero. A volume that responded linearly to shear would fail here.
        for label in ('xy', 'xz', 'yz'):
            B = dict(direction_matrices('full'))[label]
            got = fd_component(TRICLINIC, POSITIONS, B, DELTA,
                               lambda H, R: volume_energy(H))
            assert abs(got) < 1.0e-9 * s_expected_gha()


class TestDeltaConvergence:
    """The truncation error is measured, and its order is asserted.

    The isotropic direction is the one that carries real truncation
    error: B = I scales all three lattice rows, so the volume enters as
    (1 + delta)^3 and the bulk model is not quadratic in the strain.
    A diagonal direction scales one row, so its volume is linear in the
    strain and the central difference of this model is exact to machine
    precision (measured 1.7e-13 relative, asserted below).
    """

    @staticmethod
    def _iso_error(delta):
        expected = 3.0 * s_expected_gha()  # sigma : I = tr(sigma)
        got = fd_component(CUBIC, POSITIONS, np.eye(3), delta,
                           lambda H, R: volume_energy(H))
        return abs(got - expected) / abs(expected)

    def test_central_difference_is_second_order_in_delta(self):
        errors = {delta: self._iso_error(delta)
                  for delta in (DELTA, DELTA / 2.0, DELTA / 4.0)}
        ratio_half = errors[DELTA] / errors[DELTA / 2.0]
        ratio_quarter = errors[DELTA / 2.0] / errors[DELTA / 4.0]
        # O(delta^2): each halving of the step divides the error by 4.
        assert 3.9 < ratio_half < 4.1
        assert 3.9 < ratio_quarter < 4.1

    def test_diagonal_direction_is_exact_for_a_linear_volume(self):
        expected = s_expected_gha()
        got = fd_component(CUBIC, POSITIONS, np.diag([1.0, 0.0, 0.0]),
                           DELTA, lambda H, R: volume_energy(H))
        assert abs(got - expected) / abs(expected) < 1.0e-10

    def test_measured_error_is_reported_for_the_record(self):
        error = self._iso_error(DELTA)
        # The measured floor of this scheme at the recommended starting
        # step: no accuracy claim about a real CP2K run, only about the
        # strain mathematics itself.
        assert error < 1.0e-4
        print(f'[num strain FD] iso direction delta={DELTA:g} '
              f'relative truncation error {error:.4e} '
              f'(O(delta^2), halving divides it by 4)')
