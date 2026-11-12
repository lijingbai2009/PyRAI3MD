"""
======================================================================
Tests for the framework's Kabsch helper, `data/geometry.py`.

The helper used to be broken in three ways at once, and each is pinned here
separately so a regression shows up as one failing assertion rather than as a
mysterious number: it removed no centre, it applied no weights, and its
docstring described a rotation step it did not perform (defect D-10,
dev/methods/CONSTRAINT_DEVELOPMENT.md section 18). The merge round closes the
row in both required halves: the helper is corrected, and its duplicate
alignment arithmetic is deleted in favour of the single implementation in
`utilities/rmsd.py`, which the helper now wraps.

Direction of the deduplication, and why it is this direction: `utilities` is a
leaf layer that must not import `workflow` (the rule in the headers of
`utilities/rmsd.py` and `utilities/lattice.py`, and the import discipline the
shadow build was enforced with, section 17), while `data/geometry.py`
already imports from `utilities`. So the shared implementation stays in
`utilities/rmsd.py` and the higher layer consumes it; the reverse would have
made a leaf depend on the workflow above it. The last test class checks that
the arithmetic really is gone from `data/geometry.py` and that the two entry
points cannot drift apart.

The number is checked against a superposition written along a third route that
shares no line with either implementation: Horn's configuration model, the 4x4
symmetric matrix built from the cross-covariance, whose largest eigenvector is
the optimal unit quaternion. Nothing there is a singular-value decomposition of
the 3x3 cross-covariance, so agreement to 1e-12 is evidence and not a tautology.
A brute-force scan over random rotations backs it up as a lower bound: a value
some rotation beats is not an optimum.

Two honesty notes the assertions record rather than hide. First, the rigid
rotation case is asserted to be *exactly* zero on a centrosymmetric block whose
coordinates are dyadic rationals, since there the centring, the singular values
and the sum are all representable bit for bit; on a general geometry the value
comes from the singular-value identity, whose absolute error is ~1e-15 in the
square, so the deviation of a turned copy is asserted below 1e-7 and not equal
to 0.0. Second, a rotation that is not unique still has a single-valued
deviation, so the degeneracy refusal belongs to the gradient and is asserted to
be unreachable from this wrapper, which returns a float and no derivative at
all.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import inspect
import unittest

import numpy as np

from pyrai3md.utilities import rmsd
from pyrai3md.workflow.constraints import metadynamics
from pyrai3md.data.geometry import kabsch

# A general (asymmetric, non-degenerate) block: six atoms, no symmetry the
# alignment could exploit, so a wrong determinant fix or a missing centring
# changes the number.
REFERENCE = np.array([[0.00, 0.00, 0.00],
                      [1.90, 0.30, 0.10],
                      [0.40, 2.00, -0.20],
                      [2.10, 1.80, 1.30],
                      [-1.20, 0.70, 1.60],
                      [0.80, -1.50, 0.40]])
MASSES = np.array([12.0, 1.0, 1.0, 16.0, 1.0, 14.0])

# Every coordinate is a dyadic rational and so is exactly representable, and the
# block is centrosymmetric, so a signed-permutation rotation leaves no rounding
# anywhere: the rigid case can then be asserted to be exactly 0.0 instead of
# merely small. EXACT_BLOCK carries the same guarantee for the centring step.
EXACT_STAR = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])
EXACT_BLOCK = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.5, 0.0],
                        [0.0, -0.5, 2.0]])

DISTORTION = np.array([[0.05, -0.02, 0.01],
                       [-0.03, 0.06, 0.02],
                       [0.01, 0.02, -0.05],
                       [-0.02, -0.04, 0.03],
                       [0.04, 0.01, -0.02],
                       [-0.01, 0.03, 0.04]])

ROTATE_90_Z = np.array([[0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])


def _quaternion_to_matrix(unit):
    """A proper rotation from a unit quaternion, written out."""
    w, x, y, z = np.asarray(unit, dtype=np.float64) / np.linalg.norm(unit)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _weights(count, masses):
    """Normalized weights: uniform, or the given masses over their sum."""
    if masses is None:
        return np.full(int(count), 1.0 / count)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    return values / values.sum()


def _independent_rmsd(coords, reference, masses=None):
    """RMSD after the optimal superposition, by Horn's quaternion method.

    Maximize `trace(R M)` over the rotation group with `M = sum_i w_i p_i q_i^T`
    the cross-covariance of the centered blocks. Parametrizing `R` by a unit
    quaternion turns the objective into a quadratic form `u^T N u` on a
    symmetric 4x4 matrix built from the entries of `M`, and the maximizing
    quaternion is the eigenvector of the largest eigenvalue (Horn 1987, the
    configuration model). Quaternion rotations are proper by construction, so
    this route needs no determinant correction - which is exactly why it is a
    useful second opinion on an implementation that does need one.

    Returns the weighted root-mean-square deviation, and the rotation.
    """
    w = _weights(coords.shape[0], masses)
    current = np.asarray(coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    mobile = current - np.sum(current * w[:, None], axis=0)
    fixed = target - np.sum(target * w[:, None], axis=0)
    matrix = (mobile * w[:, None]).T @ fixed
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    horn = np.array([[a + e + i, f - h, g - c, b - d],
                     [f - h, a - e - i, b + d, g + c],
                     [g - c, b + d, -a + e - i, f + h],
                     [b - d, g + c, f + h, -a - e + i]])
    values, vectors = np.linalg.eigh(horn)
    rotation = _quaternion_to_matrix(vectors[:, int(np.argmax(values))])
    residual = fixed - (rotation @ mobile.T).T
    squared = float(np.sum(w * np.sum(residual * residual, axis=1)))
    return float(np.sqrt(squared)), rotation


def _best_random_rotation(coords, reference, masses=None, trials=4000,
                          seed=11):
    """Lowest deviation over random rotations: a lower bound on the optimum."""
    rng = np.random.default_rng(seed)
    w = _weights(coords.shape[0], masses)
    mobile = coords - np.sum(coords * w[:, None], axis=0)
    fixed = reference - np.sum(reference * w[:, None], axis=0)
    best = np.inf
    for _ in range(trials):
        rotation = _quaternion_to_matrix(rng.normal(size=4))
        residual = fixed - (rotation @ mobile.T).T
        best = min(best, float(np.sum(w * np.sum(residual * residual, axis=1))))
    return float(np.sqrt(best))


class TestRigidMotion(unittest.TestCase):
    """The docstring's promise: a rotation is applied, a centre is removed."""

    def test_a_turned_and_shifted_block_is_exactly_zero(self):
        """The corrected helper, at the accuracy the arithmetic can reach.

        This is the assertion D-10 owed: the old version reported the deviation
        of a turned molecule because it never removed the centre and never
        solved for the rotation that minimizes the sum. On a centrosymmetric
        block with dyadic coordinates and a signed-permutation rotation nothing
        rounds, so the number is 0.0 and not merely small.
        """
        moved = EXACT_STAR @ ROTATE_90_Z.T + 2.5
        self.assertEqual(kabsch(moved, EXACT_STAR), 0.0)
        self.assertEqual(kabsch(moved, EXACT_STAR, masses=np.ones(6)), 0.0)

    def test_a_random_rigid_motion_is_zero_to_the_form_accuracy(self):
        """The same claim on a general geometry, with the tolerance stated.

        The value comes from the singular-value identity, whose absolute error
        is a multiple of machine epsilon times the weighted norm, so the
        deviation of a rotated block is ~1e-8 rather than 0.0. The square is
        what is computed and it is zero at 1e-15, which is why the assertion is
        on the square's size and the deviation's bound, not on equality.
        """
        rng = np.random.default_rng(5)
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        moved = (REFERENCE @ _quaternion_to_matrix(quaternion).T
                 + np.array([4.25, -1.5, 2.0]))
        deviation = kabsch(moved, REFERENCE)
        self.assertLess(deviation, 1.0e-7)
        self.assertLess(rmsd.rmsd_squared(moved, REFERENCE), 1.0e-14)

    def test_centring_is_now_part_of_the_answer(self):
        """Negative control for the missing centre.

        A pure translation changes no aligned deviation. The uncentered route
        that used to live in this function reported it as a large deviation, so
        this comparison is what shows the defect is gone rather than that both
        implementations agree.
        """
        shifted = EXACT_BLOCK + 11.0
        # A shifted block must align to zero. The sum of squares is not
        # bit-zero on every CPU/BLAS (a compute node read 2.1e-8 where the
        # login node read exactly 0), so the comparison uses the roundoff
        # bound this file already uses elsewhere, not exact equality.
        self.assertLess(kabsch(shifted, EXACT_BLOCK), 1.0e-7)
        uncentered = float(np.sqrt(np.mean(np.sum(
            (shifted - EXACT_BLOCK) ** 2, axis=1))))
        self.assertGreater(uncentered, 1.0)

    def test_the_existing_caller_contract_survives(self):
        """`kabsch(p, q)` still returns one float, and zero for p == q.

        The only consumer in the tree is the integrator test, which asserts the
        identical-block case at ten places; the masses keyword defaults to None
        so no call site had to change.
        """
        same = np.asarray(REFERENCE, dtype=np.float64)
        value = kabsch(same, same.copy())
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 0.0, places=10)


class TestKnownDistortion(unittest.TestCase):
    """A real change of shape, measured against a second route."""

    def test_the_unweighted_number_matches_an_independent_superposition(self):
        moved = (REFERENCE @ ROTATE_90_Z.T + 2.5) + DISTORTION
        expected, _ = _independent_rmsd(moved, REFERENCE)
        self.assertAlmostEqual(kabsch(moved, REFERENCE), expected, places=12)

    def test_the_mass_weighted_number_matches_too(self):
        moved = (REFERENCE @ ROTATE_90_Z.T + 2.5) + DISTORTION
        expected, _ = _independent_rmsd(moved, REFERENCE, MASSES)
        self.assertAlmostEqual(kabsch(moved, REFERENCE, masses=MASSES),
                               expected, places=12)

    def test_the_scale_of_the_masses_changes_nothing(self):
        moved = REFERENCE + DISTORTION
        self.assertAlmostEqual(kabsch(moved, REFERENCE, masses=MASSES),
                               kabsch(moved, REFERENCE, masses=7.3 * MASSES),
                               places=14)

    def test_no_random_rotation_beats_the_reported_deviation(self):
        """The value is an optimum, not just a number two routes produce.

        Both implementations could in principle agree on a rotation that is not
        optimal; a scan over four thousand random rotations that never goes
        below them says otherwise. Random sampling of SO(3) is a weak optimizer,
        so the assertion is the lower bound only.
        """
        moved = (REFERENCE @ ROTATE_90_Z.T) + DISTORTION
        reported = kabsch(moved, REFERENCE)
        randomized = _best_random_rotation(moved, REFERENCE)
        self.assertLessEqual(reported, randomized + 1.0e-12)
        weighted = kabsch(moved, REFERENCE, masses=MASSES)
        randomized_w = _best_random_rotation(moved, REFERENCE, MASSES)
        self.assertLessEqual(weighted, randomized_w + 1.0e-12)

    def test_a_mirror_image_is_not_superimposed_onto_itself(self):
        """The determinant sign fix, asserted as a consequence.

        Dropping it lets a reflection count as a rotation and reports zero for
        a pair that no rotation aligns.
        """
        mirrored = REFERENCE.copy()
        mirrored[:, 2] *= -1.0
        self.assertGreater(kabsch(mirrored, REFERENCE), 1.0e-2)


class TestMassWeighting(unittest.TestCase):
    """The keyword D-10 asked for, and what it actually changes."""

    def test_the_weighted_and_unweighted_centres_differ(self):
        """A heavy atom pins the weighted centre; the centroid drifts away.

        The block has one 40 amu atom at the origin and a light atom four Bohr
        out, so the plain centroid sits well off the heavy atom while the
        mass-weighted centre stays close to it. Centring is the step the old
        helper skipped and weighting is the step it could not have done, so the
        two deviations differing is the point of the keyword.
        """
        asymmetric = np.array([[0.0, 0.0, 0.0],
                               [4.0, 0.0, 0.0],
                               [0.0, 3.0, 0.0],
                               [0.0, 0.0, 2.0],
                               [1.0, 1.0, 1.0],
                               [-1.0, 0.5, -0.5]])
        heavy = np.array([40.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        moved = asymmetric + DISTORTION
        plain = kabsch(moved, asymmetric)
        weighted = kabsch(moved, asymmetric, masses=heavy)
        self.assertGreater(abs(plain - weighted), 1.0e-3)
        centroid = asymmetric.mean(axis=0)
        com = np.sum(asymmetric * (heavy / heavy.sum())[:, None], axis=0)
        self.assertGreater(float(np.linalg.norm(centroid - com)), 0.5)
        expected, _ = _independent_rmsd(moved, asymmetric, heavy)
        self.assertAlmostEqual(weighted, expected, places=12)

    def test_equal_masses_recover_the_plain_rmsd(self):
        moved = REFERENCE + DISTORTION
        self.assertAlmostEqual(kabsch(moved, REFERENCE),
                               kabsch(moved, REFERENCE, masses=np.ones(6)),
                               places=14)

    def test_a_usable_rmsd_needs_one_positive_mass_per_atom(self):
        broken = MASSES.copy()
        broken[0] = 0.0
        with self.assertRaises(rmsd.RmsdError):
            kabsch(REFERENCE, REFERENCE, masses=broken)
        with self.assertRaises(rmsd.RmsdError):
            kabsch(REFERENCE, REFERENCE, masses=MASSES[:4])


class TestDegeneracyGuard(unittest.TestCase):
    """A rotation that is not unique must not silently produce a gradient."""

    def setUp(self):
        self.reference = REFERENCE + np.array([0.0, 0.0, 3.0])
        self.collapsed = REFERENCE.copy()
        self.collapsed[:, 2] = float(np.mean(self.collapsed[:, 2]))

    def test_a_collapsed_block_still_has_a_deviation(self):
        """The value is single-valued, so the helper answers.

        Every optimal rotation gives the same sum of squares, so refusing here
        would take a working restraint away from a geometry that is fine to
        bias on.
        """
        value = kabsch(self.collapsed, self.reference)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_the_same_block_refuses_to_yield_a_gradient(self):
        """The refusal sits with the derivative, in the shared code.

        Two equally good superpositions of this pair disagree atom by atom, so
        `rmsd.rmsd_squared(..., grad=True)` raises. The wrapper cannot be used
        to get around that: it hands back a float and no derivative, which is
        what its signature makes impossible.
        """
        value = kabsch(self.collapsed, self.reference)
        self.assertAlmostEqual(
            value ** 2, rmsd.rmsd_squared(self.collapsed, self.reference),
            places=14)
        with self.assertRaises(rmsd.RmsdError) as caught:
            rmsd.rmsd_squared(self.collapsed, self.reference, grad=True)
        self.assertIn('collapsed', str(caught.exception))

    def test_a_planar_group_is_not_refused(self):
        """The guard is not 'a singular value vanished'.

        A planar group against a planar reference has a free rotation about the
        plane normal, and the deviation of a turned copy is still zero: the
        helper must answer, not refuse, or no planar fragment can ever be
        biased on.
        """
        planar = np.array([[0.0, 0.0, 0.0],
                           [1.5, 0.0, 0.0],
                           [0.5, 1.25, 0.0],
                           [-1.0, 0.5, 0.0]])
        moved = planar @ ROTATE_90_Z.T
        self.assertEqual(kabsch(moved, planar), 0.0)
        _, jacobian = rmsd.rmsd_squared(
            moved + np.array([[0.05, -0.02, 0.0],
                              [0.01, 0.04, 0.0],
                              [-0.03, 0.02, 0.0],
                              [0.02, -0.01, 0.0]]),
            planar, grad=True)
        self.assertEqual(jacobian.shape, planar.shape)

    def test_the_refusals_of_the_shared_code_reach_the_helper(self):
        with self.assertRaises(rmsd.RmsdError):
            kabsch(REFERENCE[:3], REFERENCE)
        with self.assertRaises(rmsd.RmsdError):
            kabsch(REFERENCE[:1], REFERENCE[:1])
        with self.assertRaises(rmsd.RmsdError):
            kabsch(REFERENCE[:, :2], REFERENCE)


class TestSingleImplementation(unittest.TestCase):
    """The half of D-10 that is about deletion, not correction."""

    def test_the_helper_carries_no_alignment_arithmetic(self):
        source = inspect.getsource(inspect.getmodule(kabsch))
        self.assertNotIn('linalg.svd', source)
        self.assertNotIn('linalg.det', source)
        body = inspect.getsource(kabsch)
        self.assertIn('rmsd_squared', body)

    def test_the_wrapper_is_the_square_root_of_the_shared_variable(self):
        """One implementation, checked by identity of the number.

        If a second alignment ever reappears in `data/geometry.py`, this is the
        assertion that notices.
        """
        moved = (REFERENCE @ ROTATE_90_Z.T + 1.5) + DISTORTION
        self.assertEqual(kabsch(moved, REFERENCE),
                         float(np.sqrt(rmsd.rmsd_squared(moved, REFERENCE))))
        self.assertEqual(kabsch(moved, REFERENCE, masses=MASSES),
                         float(np.sqrt(rmsd.rmsd_squared(
                             moved, REFERENCE, weights=MASSES))))

    def test_the_history_potential_shares_the_one_path(self):
        """metadynamics reads the same implementation the helper wraps.

        D-10 asks for "no second RMSD implementation": the history potential and
        this helper must be the same arithmetic seen from two layers, so the
        workflow module that owns the second backend holds no SVD of its own.
        """
        self.assertIn('rmsd.rmsd_squared', inspect.getsource(metadynamics))
        self.assertNotIn('linalg.svd', inspect.getsource(metadynamics))
        # One cross-covariance SVD in the whole package: the shared one.
        self.assertEqual(inspect.getsource(rmsd).count('linalg.svd'), 1)


if __name__ == '__main__':
    unittest.main()
