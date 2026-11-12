"""
======================================================================
Test suite for the shared periodic-cell math module
(pyrai3md/utilities/lattice.py).

Tests cover:
- cell6_to_matrix / matrix_to_cell6 round trips (cubic, orthorhombic,
  triclinic) and the standard orientation convention.
- cell_volume for cubic and triclinic cells and the zero-cell case.
- normalize_pbc accepted forms (None, empty, scalar, bool, float 0/1)
  and rejection of wrong length, out-of-range and non-numeric values.
- boundary_condition cluster / surface / periodic classification.
- validate_lattice failure branches (missing cell with PBC on, bad
  shape, non-finite entries, degenerate periodic axis, singular
  matrix) and the non-periodic pass-through.
- cart_to_frac / frac_to_cart round trips and wrap_fractional on
  negative coordinates, multi-box displacements and partial axes.
- wrap_fractional_with_image: the crossing count the wrap accrues, the
  edge-snap compensation, and the axes that never cross.
- unwrap_coordinates / min_image_displacement: the continuous and
  nearest-image views rebuilt from wrapped coordinates plus image.

Sep 25 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities.lattice import (
    cell6_to_matrix,
    matrix_to_cell6,
    cell_volume,
    normalize_pbc,
    boundary_condition,
    validate_lattice,
    cart_to_frac,
    frac_to_cart,
    wrap_fractional,
    wrap_fractional_with_image,
    unwrap_coordinates,
    min_image_displacement,
    half_box_bound,
)

TRICLINIC_CELL6 = [8.0, 9.0, 10.0, 80.0, 95.0, 100.0]


class TestCell6MatrixConversion(unittest.TestCase):
    """Six-parameter <-> matrix round trips and orientation."""

    def test_cubic_matrix(self):
        lattice = cell6_to_matrix([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        np.testing.assert_allclose(
            lattice, np.eye(3) * 10.0, atol=1e-12)

    def test_standard_orientation(self):
        lattice = cell6_to_matrix(TRICLINIC_CELL6)
        # a along x, b in the xy plane.
        self.assertAlmostEqual(lattice[0][1], 0.0, places=12)
        self.assertAlmostEqual(lattice[0][2], 0.0, places=12)
        self.assertAlmostEqual(lattice[1][2], 0.0, places=12)

    def test_round_trip_triclinic(self):
        back = matrix_to_cell6(cell6_to_matrix(TRICLINIC_CELL6))
        np.testing.assert_allclose(back, TRICLINIC_CELL6, atol=1e-9)

    def test_round_trip_orthorhombic(self):
        cell6 = [3.0, 4.0, 5.0, 90.0, 90.0, 90.0]
        back = matrix_to_cell6(cell6_to_matrix(cell6))
        np.testing.assert_allclose(back, cell6, atol=1e-9)


class TestCellVolume(unittest.TestCase):
    """Volume |det(B)| and the zero-cell case."""

    def test_cubic_volume(self):
        self.assertAlmostEqual(
            cell_volume(np.eye(3) * 2.0), 8.0, places=12)

    def test_triclinic_volume(self):
        lattice = cell6_to_matrix(TRICLINIC_CELL6)
        self.assertAlmostEqual(
            cell_volume(lattice),
            abs(float(np.linalg.det(lattice))), places=12)

    def test_zero_cell_volume(self):
        self.assertEqual(cell_volume(np.zeros((3, 3))), 0.0)


class TestNormalizePbc(unittest.TestCase):
    """Accepted forms and rejection matrix for pbc normalization."""

    def test_accepted_forms(self):
        self.assertEqual(normalize_pbc(None), [0, 0, 0])
        self.assertEqual(normalize_pbc([]), [0, 0, 0])
        self.assertEqual(normalize_pbc(0), [0, 0, 0])
        self.assertEqual(normalize_pbc(1), [1, 1, 1])
        self.assertEqual(normalize_pbc([1, 0, 1]), [1, 0, 1])
        self.assertEqual(normalize_pbc([True, False, True]), [1, 0, 1])
        self.assertEqual(normalize_pbc([1.0, 0.0, 1.0]), [1, 0, 1])
        self.assertEqual(normalize_pbc(np.array([1, 1, 0])), [1, 1, 0])

    def test_rejects_wrong_length(self):
        for bad in ([1, 0], [1, 0, 1, 0]):
            with self.assertRaises(ValueError):
                normalize_pbc(bad)

    def test_rejects_out_of_range(self):
        for bad in ([2, 0, 0], [1, -1, 0], [0.5, 0, 0]):
            with self.assertRaises(ValueError):
                normalize_pbc(bad)

    def test_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            normalize_pbc(['x', 0, 0])


class TestBoundaryCondition(unittest.TestCase):
    """cluster / surface / periodic classification."""

    def test_classification(self):
        self.assertEqual(boundary_condition(None), 'cluster')
        self.assertEqual(boundary_condition([0, 0, 0]), 'cluster')
        self.assertEqual(boundary_condition([1, 1, 1]), 'periodic')
        self.assertEqual(boundary_condition([1, 1, 0]), 'surface')
        self.assertEqual(boundary_condition([1, 0, 0]), 'surface')


class TestValidateLattice(unittest.TestCase):
    """Structural and periodic validation branches."""

    def test_valid_periodic_cell_returns_volume(self):
        lattice = cell6_to_matrix(TRICLINIC_CELL6)
        volume = validate_lattice(lattice, [1, 1, 1])
        self.assertAlmostEqual(
            volume, abs(float(np.linalg.det(lattice))), places=12)

    def test_non_periodic_allows_zero_cell(self):
        self.assertEqual(validate_lattice(np.zeros((3, 3)), [0, 0, 0]), 0.0)
        self.assertEqual(validate_lattice(None, None), 0.0)

    def test_missing_cell_with_pbc_raises(self):
        for empty in (None, [], np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                validate_lattice(empty, [1, 1, 1])

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            validate_lattice([[1.0, 0.0], [0.0, 1.0]], [1, 1, 1])

    def test_non_finite_raises(self):
        lattice = [[float('nan'), 0.0, 0.0],
                   [0.0, 10.0, 0.0],
                   [0.0, 0.0, 10.0]]
        with self.assertRaises(ValueError):
            validate_lattice(lattice, [1, 1, 1])

    def test_degenerate_periodic_axis_raises(self):
        lattice = [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
        with self.assertRaises(ValueError):
            validate_lattice(lattice, [1, 1, 1])
        # A rank-deficient matrix is singular on every axis setting
        # (partial periodicity is rejected in this PBC stage anyway).
        with self.assertRaises(ValueError):
            validate_lattice(lattice, [1, 0, 1])

    def test_singular_matrix_raises(self):
        lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            validate_lattice(lattice, [1, 1, 1])


class TestCoordinateConversions(unittest.TestCase):
    """cart <-> frac round trips and fractional wrapping."""

    def setUp(self):
        self.lattice = cell6_to_matrix(TRICLINIC_CELL6)

    def test_round_trip(self):
        coords = np.array([[1.0, -2.0, 3.5], [7.0, 8.0, -9.0]])
        back = frac_to_cart(cart_to_frac(coords, self.lattice), self.lattice)
        np.testing.assert_allclose(back, coords, atol=1e-9)

    def test_singular_raises(self):
        singular = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            cart_to_frac([[1.0, 0.0, 0.0]], singular)

    def test_wrap_negative_and_multi_box(self):
        frac = np.array([[-0.25, 2.75, -3.5]])
        wrapped = wrap_fractional(frac, [1, 1, 1])
        np.testing.assert_allclose(wrapped, [[0.75, 0.75, 0.5]], atol=1e-12)

    def test_wrap_partial_axes_only(self):
        frac = np.array([[-0.25, 2.75, -3.5]])
        wrapped = wrap_fractional(frac, [1, 0, 0])
        np.testing.assert_allclose(
            wrapped, [[0.75, 2.75, -3.5]], atol=1e-12)

    def test_wrap_does_not_modify_input(self):
        frac = np.array([[-0.25, 2.75, -3.5]])
        original = frac.copy()
        wrap_fractional(frac, [1, 1, 1])
        np.testing.assert_allclose(frac, original)

    def test_wrap_tiny_negative_snaps_to_zero_not_upper_edge(self):
        # numpy's modulo returns exactly 1.0 for a tiny negative input
        # (-1e-17 % 1.0 == 1.0); the half-open [0, 1) convention must
        # snap that upper-edge result back to 0.0, otherwise a boundary
        # atom flips between lo and hi on consecutive steps.
        frac = np.array([[-1e-17, 0.0, -4e-17]])
        wrapped = wrap_fractional(frac, [1, 1, 1])
        np.testing.assert_allclose(wrapped, [[0.0, 0.0, 0.0]], atol=1e-12)

    def test_wrap_genuine_interior_value_preserved(self):
        # a value genuinely just below the edge is a real position and
        # must not be snapped to 0.0
        frac = np.array([[1.0 - 1e-9, 0.5, 0.25]])
        wrapped = wrap_fractional(frac, [1, 1, 1])
        np.testing.assert_allclose(
            wrapped, [[1.0 - 1e-9, 0.5, 0.25]], atol=1e-15)


class TestWrapFractionalWithImage(unittest.TestCase):
    """Wrap that reports the cell crossings the wrapped value throws away."""

    def test_identity_holds_for_negative_and_multi_box(self):
        frac = np.array([[-0.25, 2.75, -3.5], [12.3, -1e-17, 0.0]])
        wrapped, image = wrap_fractional_with_image(frac, [1, 1, 1])
        np.testing.assert_allclose(wrapped, frac - image, atol=1e-12)
        np.testing.assert_allclose(wrapped, [[0.75, 0.75, 0.5],
                                             [0.3, 0.0, 0.0]], atol=1e-12)
        self.assertEqual(image.tolist(), [[-1, 2, -4], [12, 0, 0]])

    def test_upper_edge_snap_credits_the_extra_crossing(self):
        # The snap is what keeps frac = wrapped + image true at the edge:
        # without the compensation the atom would be credited with a
        # crossing it is no longer making.
        frac = np.array([[-1e-17, -4e-17, 0.5]])
        wrapped, image = wrap_fractional_with_image(frac, [1, 1, 1])
        np.testing.assert_allclose(wrapped, [[0.0, 0.0, 0.5]], atol=1e-12)
        self.assertEqual(image.tolist(), [[0, 0, 0]])

    def test_non_periodic_axes_never_cross(self):
        frac = np.array([[-0.25, 2.75, -3.5]])
        wrapped, image = wrap_fractional_with_image(frac, [1, 0, 0])
        np.testing.assert_allclose(wrapped, [[0.75, 2.75, -3.5]], atol=1e-12)
        self.assertEqual(image.tolist(), [[-1, 0, 0]])

    def test_integer_dtype_and_copy_semantics(self):
        frac = np.array([[-0.25, 2.75, -3.5]])
        original = frac.copy()
        _, image = wrap_fractional_with_image(frac, [1, 1, 1])
        self.assertEqual(image.dtype, np.int64)
        np.testing.assert_allclose(frac, original)

    def test_wrap_fractional_delegates_unchanged(self):
        frac = np.array([[-0.25, 2.75, -3.5], [12.3, 0.0, 1.0 - 1e-9]])
        np.testing.assert_allclose(
            wrap_fractional(frac, [1, 1, 1]),
            wrap_fractional_with_image(frac, [1, 1, 1])[0], atol=1e-15)


class TestDerivedCoordinateViews(unittest.TestCase):
    """unwrap_coordinates and min_image_displacement: views, not state."""

    def test_unwrap_rebuilds_the_continuous_path(self):
        lattice = np.diag([10.0, 10.0, 10.0])
        wrapped = np.array([[2.0, 7.0, 5.0]])
        image = np.array([[1, -1, 2]])
        np.testing.assert_allclose(
            unwrap_coordinates(wrapped, image, lattice),
            [[12.0, -3.0, 25.0]], atol=1e-9)

    def test_unwrap_without_image_is_the_identity(self):
        lattice = np.diag([10.0, 10.0, 10.0])
        wrapped = np.array([[2.0, 7.0, 5.0]])
        for empty in (None, [], np.zeros((0, 0), dtype=np.int64)):
            np.testing.assert_allclose(
                unwrap_coordinates(wrapped, empty, lattice), wrapped)

    def test_unwrap_round_trips_the_wrap(self):
        lattice = cell6_to_matrix(TRICLINIC_CELL6)
        cart = np.array([[25.0, -12.0, 7.0], [3.0, 4.0, 5.0]])
        frac = cart_to_frac(cart, lattice)
        wrapped_frac, image = wrap_fractional_with_image(frac, [1, 1, 1])
        rebuilt = unwrap_coordinates(frac_to_cart(wrapped_frac, lattice),
                                     image, lattice)
        np.testing.assert_allclose(rebuilt, cart, rtol=1e-10, atol=1e-8)

    def test_min_image_picks_the_nearest_face(self):
        lattice = np.diag([10.0, 10.0, 10.0])
        # The 9.242 Angstrom split of a boundary-straddling O-H pair is
        # 0.758 Angstrom across the other face, which is the bond length
        # that makes bond detection work on main-image coordinates.
        reduced = min_image_displacement(np.array([[9.242, 0.0, 0.0]]),
                                        lattice)
        np.testing.assert_allclose(reduced, [[-0.758, 0.0, 0.0]], atol=1e-12)

    def test_min_image_is_a_no_op_inside_the_half_box(self):
        lattice = np.diag([10.0, 10.0, 10.0])
        d = np.array([[3.0, -4.0, 0.5], [1.0, 2.0, 3.0]])
        np.testing.assert_allclose(min_image_displacement(d, lattice), d,
                                   atol=1e-12)

    def test_min_image_preserves_shape(self):
        lattice = np.diag([10.0, 10.0, 10.0])
        d = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
        self.assertEqual(min_image_displacement(d, lattice).shape, d.shape)

    def test_min_image_rejects_a_singular_cell(self):
        singular = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            min_image_displacement(np.array([[1.0, 2.0, 3.0]]), singular)


def _component_wise(displacements, lattice):
    """The reduction the search replaced, kept here as the equivalence oracle.

    M0-1's exit criterion is not "the new code is close to the old one", it is
    "on a cell where the old rule was already minimal the numbers are the same
    bits". Only the rule itself, written out, can be compared bit for bit.
    """
    flat = np.asarray(displacements, dtype=np.float64).reshape(-1, 3)
    frac = flat @ np.linalg.inv(np.asarray(lattice, dtype=np.float64))
    frac = frac - np.round(frac)
    return (frac @ np.asarray(lattice, dtype=np.float64)).reshape(
        np.asarray(displacements).shape)


class TestMinImageCandidateSearch(unittest.TestCase):
    """M0-1: the nearest image is searched, not assumed per axis."""

    ORTHO = np.diag([10.0, 12.0, 9.0])
    # A 60 degree hexagonal slab: 10 A vectors, 120 degrees between them. This
    # is an ordinary cell a 2D material or a graphene oxide sheet would be
    # declared in, not a stress test.
    HEX60 = np.array([[10.0, 0.0, 0.0],
                      [5.0, 8.660254, 0.0],
                      [0.0, 0.0, 12.0]])
    # Two cell vectors 0.6 degrees apart: the gap between the planes they span
    # is 0.1 Angstrom while the vectors are 10 Angstrom long, so no fixed-size
    # shell around the component-wise value can certify a nearest image.
    NOT_REDUCED = np.array([[10.0, 0.0, 0.0],
                            [9.99, 0.1, 0.0],
                            [0.0, 0.0, 10.0]])

    def _sample(self, seed=20260930, count=4000, span=30.0):
        rng = np.random.default_rng(seed)
        return rng.uniform(-span, span, size=(count, 3))

    def test_orthogonal_cell_returns_the_same_bits(self):
        deltas = self._sample()
        np.testing.assert_array_equal(min_image_displacement(deltas, self.ORTHO),
                                      _component_wise(deltas, self.ORTHO))

    def test_tilted_cell_finds_a_shorter_image(self):
        deltas = self._sample()
        searched = min_image_displacement(deltas, self.HEX60)
        assumed = _component_wise(deltas, self.HEX60)
        short = np.linalg.norm(searched, axis=1)
        long_ = np.linalg.norm(assumed, axis=1)
        self.assertTrue(np.all(short <= long_ + 1e-12))
        # The old rule is wrong by over 2 Angstroms on this cell; if a future
        # edit quietly restores component-wise rounding this margin goes to
        # zero.
        self.assertGreater(float((long_ - short).max()), 2.0)

    def test_searched_minimum_is_the_global_one(self):
        """Brute force over a +/-5 shell agrees with the certificate."""
        deltas = self._sample(count=60, span=25.0)
        inv = np.linalg.inv(self.HEX60)
        span = range(-5, 6)
        shell = np.array([(dx, dy, dz) for dx in span for dy in span
                          for dz in span], dtype=np.float64)
        for position, delta in enumerate(deltas):
            frac = delta @ inv
            deviation = frac - np.round(frac)
            cart = (deviation - shell) @ self.HEX60
            best = cart[np.argmin(np.einsum('ij,ij->i', cart, cart))]
            got = min_image_displacement(delta[None, :], self.HEX60)[0]
            np.testing.assert_allclose(got, best, atol=1e-12)

    def test_non_reduced_cell_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            min_image_displacement(self._sample(count=8), self.NOT_REDUCED)
        message = str(caught.exception)
        self.assertIn('size-reduced', message)
        self.assertIn('certifiable', message)

    def test_a_non_periodic_axis_is_never_shifted(self):
        cell = np.eye(3) * 10.0
        delta = np.array([[16.0, 16.0, 0.0]])
        np.testing.assert_allclose(
            min_image_displacement(delta, cell), [[-4.0, -4.0, 0.0]])
        np.testing.assert_allclose(
            min_image_displacement(delta, cell, pbc=(0, 1, 1)),
            [[16.0, -4.0, 0.0]])
        np.testing.assert_array_equal(
            min_image_displacement(delta, cell, pbc=(0, 0, 0)), delta)

    def test_none_pbc_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            min_image_displacement([[1.0, 2.0, 3.0]], self.ORTHO, pbc=None)
        self.assertIn('explicit pbc', str(caught.exception))

    def test_an_exact_half_cell_tie_keeps_the_zero_shift(self):
        cell = np.diag([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(
            min_image_displacement(np.array([[2.5, 0.0, 0.0]]), cell),
            [[2.5, 0.0, 0.0]])


class TestHalfBoxBound(unittest.TestCase):
    """The periodic bond bound is half the interplanar spacing."""

    def test_orthorhombic_bound_is_half_the_shortest_axis(self):
        lattice = [[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]]
        self.assertAlmostEqual(half_box_bound(lattice), 3.0)

    def test_tilted_bound_uses_the_interplanar_spacing(self):
        # A 60-degree cell has a 10 A vector but only 8.66 A between the
        # planes it spans, and LAMMPS measures the gap; taking the
        # vector would let a bond through that the engine still calls
        # inconsistent.
        angle = np.radians(60.0)
        lattice = [[10.0, 0.0, 0.0],
                   [10.0 * np.cos(angle), 10.0 * np.sin(angle), 0.0],
                   [0.0, 0.0, 10.0]]
        bound = half_box_bound(lattice)
        self.assertLess(bound, 5.0)
        self.assertAlmostEqual(bound, 0.5 * 10.0 * np.sin(angle))

    def test_a_degenerate_cell_has_no_bound(self):
        self.assertEqual(half_box_bound([[0.0, 0.0, 0.0],
                                         [0.0, 10.0, 0.0],
                                         [0.0, 0.0, 10.0]]), 0.0)


if __name__ == '__main__':
    unittest.main()
