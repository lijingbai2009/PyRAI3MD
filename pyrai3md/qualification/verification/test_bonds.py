"""
======================================================================
Test suite for the periodic image-flag derivation module
(pyrai3md/utilities/bonds.py).

Tests cover:
- perceive_bonds: the nearest-image bond set and the image step each
  bond carries, including the axis restriction a partial cell imposes.
- derive_initial_image: the crossing counts a wrapped snapshot implies,
  the canonical form that keeps a molecule in the cell it was wrapped
  into, and its determinism.
- The refusal branches: a ring that winds around the cell, a bond that
  cannot fit the cell, a non-periodic or degenerate cell, too few or
  too many atoms, and non-finite coordinates.
- unwrapped_bond_lengths as the negative control: the same structure
  with all-zero counts leaves the straddling bond as long as the box.

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest
import warnings
from unittest import mock

import numpy as np

from pyrai3md.qualification.verification._source_paths import source_read
from pyrai3md.utilities import bonds
from pyrai3md.utilities.bonds import (
    derive_initial_image,
    perceive_bonds,
    unwrapped_bond_lengths,
)
from pyrai3md.utilities.lattice import half_box_bound

BOX = 10.0
CUBIC = [[BOX, 0.0, 0.0], [0.0, BOX, 0.0], [0.0, 0.0, BOX]]

# Two waters. The first sits wholly inside the cell. The second has one
# hydrogen wrapped to the far x face: the O-H bond is 0.96 A long only
# once that hydrogen is counted one cell back.
ATOMS = ['O', 'H', 'H', 'O', 'H', 'H']
COORDS = [
    [5.00, 2.00, 2.00],
    [5.76, 2.00, 2.00],
    [4.62, 2.76, 2.00],
    [0.10, 8.00, 8.00],
    [9.14, 8.00, 8.00],
    [0.58, 8.94, 8.00],
]


def _rows(image):
    return [[int(value) for value in row] for row in image]


class TestPerceiveBonds(unittest.TestCase):

    def test_bond_set_matches_the_topology(self):
        bonds = perceive_bonds(ATOMS, COORDS, CUBIC)
        pairs = sorted((i + 1, j + 1) for i, j, _delta in bonds)
        self.assertEqual(pairs, [(1, 2), (1, 3), (4, 5), (4, 6)])

    def test_the_winning_shift_travels_with_the_bond(self):
        bonds = perceive_bonds(ATOMS, COORDS, CUBIC)
        steps = {(i + 1, j + 1): delta for i, j, delta in bonds}
        # Atom 5 is wrapped to the opposite x face, so its bond to atom
        # 4 is continuous one cell back, not across the box.
        self.assertEqual(steps[(4, 5)], (-1, 0, 0))
        self.assertEqual(steps[(1, 2)], (0, 0, 0))

    def test_a_non_periodic_axis_is_never_shifted(self):
        # The same structure without periodicity in x cannot know about
        # the crossing: the pair is 9.04 A apart and simply is not a
        # bond, which is the honest answer for a non-periodic direction.
        bonds = perceive_bonds(ATOMS, COORDS, CUBIC, pbc=[0, 1, 1])
        pairs = sorted((i + 1, j + 1) for i, j, _delta in bonds)
        self.assertNotIn((4, 5), pairs)
        self.assertIn((1, 2), pairs)

    def test_intermolecular_contacts_stay_out_of_the_bond_set(self):
        # O...O at 2.7 A and H...H at 1.8 A are hydrogen bonds, not
        # covalent ones: the 1.2 tolerance must not reach them.
        atoms = ['O', 'O']
        coords = [[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]]
        self.assertEqual(perceive_bonds(atoms, coords, CUBIC), [])


class TestDerivedInitialImage(unittest.TestCase):

    def test_a_straddling_molecule_gets_its_crossing_back(self):
        image = derive_initial_image(ATOMS, COORDS, CUBIC)
        self.assertEqual(_rows(image), [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                        [0, 0, 0], [-1, 0, 0], [0, 0, 0]])

    def test_the_derivation_is_deterministic(self):
        first = derive_initial_image(ATOMS, COORDS, CUBIC)
        second = derive_initial_image(ATOMS, COORDS, CUBIC)
        self.assertEqual(_rows(first), _rows(second))

    def test_a_molecule_wholly_inside_stays_at_zero(self):
        atoms = ['O', 'H', 'H']
        coords = [[5.00, 2.00, 2.00], [5.76, 2.00, 2.00], [4.62, 2.76, 2.00]]
        self.assertEqual(_rows(derive_initial_image(atoms, coords, CUBIC)),
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_the_whole_molecule_stays_in_its_own_cell(self):
        # Both hydrogens wrapped to the far face: the counts must land
        # on the atoms that crossed, not drag the oxygen out of the box,
        # which is what fix recenter expects to find.
        atoms = ['O', 'H', 'H']
        coords = [[0.10, 5.00, 5.00], [9.14, 5.00, 5.00],
                  [9.60, 5.80, 5.00]]
        image = _rows(derive_initial_image(atoms, coords, CUBIC))
        self.assertEqual(image, [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        # The reset moved the assignment onto the molecule's own copy of
        # the cell: rebuilt continuous, its center stays inside the box
        # (without the reset the two crossed hydrogens pull it to -0.04).
        fractional = np.asarray(coords) / BOX + np.asarray(image)
        center = fractional.mean(axis=0)
        self.assertTrue(np.all(center >= 0.0) and np.all(center <= 1.0))

    def test_a_partial_cell_counts_only_periodic_axes(self):
        image = _rows(derive_initial_image(ATOMS, COORDS, CUBIC,
                                           pbc=[1, 1, 0]))
        self.assertTrue(all(row[2] == 0 for row in image))
        self.assertEqual(image[4][:2], [-1, 0])

    def test_zero_counts_leave_the_straddling_bond_across_the_box(self):
        # Negative control on the same geometry: this is exactly what
        # LAMMPS' image_check sees when the crossings are not declared.
        bonds = perceive_bonds(ATOMS, COORDS, CUBIC)
        zeros = [[0, 0, 0] for _ in ATOMS]
        long_now = unwrapped_bond_lengths(COORDS, zeros, CUBIC, bonds)
        derived = derive_initial_image(ATOMS, COORDS, CUBIC)
        fixed = unwrapped_bond_lengths(COORDS, derived, CUBIC, bonds)
        self.assertGreaterEqual(long_now.max(), half_box_bound(CUBIC))
        self.assertLess(fixed.max(), half_box_bound(CUBIC))


class TestRefusalBranches(unittest.TestCase):

    def test_a_ring_that_winds_around_the_cell_is_refused(self):
        # Ten hydrogens 0.8 A apart in a 8 A cell close into a ring that
        # wraps the periodic direction once: no integer assignment makes
        # every bond short, so the answer must be "nothing", with a
        # warning, rather than a set of counts that breaks the ring.
        coords = [[0.8 * n, 5.0, 5.0] for n in range(10)]
        atoms = ['H'] * 10
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            image = derive_initial_image(atoms, coords,
                                         [[8.0, 0, 0], [0, 10.0, 0],
                                          [0, 0, 10.0]])
        self.assertEqual(image, [])
        self.assertEqual(len(caught), 1)
        self.assertIn('not derivable', str(caught[0].message))

    def test_a_bond_too_long_for_the_cell_is_refused(self):
        # Half of this cell apart and under the bond tolerance at the
        # same time: the structure cannot satisfy the check.
        lattice = [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]
        image = derive_initial_image(['O', 'H'], [[0.0, 0.0, 0.0],
                                                  [1.0, 0.0, 0.0]],
                                     lattice)
        self.assertEqual(image, [])

    def test_a_non_periodic_system_is_left_alone(self):
        self.assertEqual(derive_initial_image(ATOMS, COORDS, CUBIC,
                                              pbc=[0, 0, 0]), [])

    def test_a_singular_cell_is_left_alone(self):
        self.assertEqual(derive_initial_image(ATOMS, COORDS,
                                              [[0.0, 0.0, 0.0],
                                               [0.0, 10.0, 0.0],
                                               [0.0, 0.0, 10.0]]), [])

    def test_non_finite_coordinates_are_left_alone(self):
        coords = [list(row) for row in COORDS]
        coords[2][0] = float('nan')
        self.assertEqual(derive_initial_image(ATOMS, coords, CUBIC), [])

    def test_a_single_atom_derives_nothing(self):
        self.assertEqual(derive_initial_image(['O'], [[0.0, 0.0, 0.0]],
                                              CUBIC), [])

    def test_an_oversized_system_is_refused_before_it_stalls(self):
        coords = [[0.1 * n, 0.0, 0.0] for n in range(6)]
        original = bonds.PERCEPTION_MAX_ATOMS
        bonds.PERCEPTION_MAX_ATOMS = 5
        try:
            self.assertEqual(derive_initial_image(['H'] * 6, coords, CUBIC),
                             [])
        finally:
            bonds.PERCEPTION_MAX_ATOMS = original


class TestFocalPerception(unittest.TestCase):
    """The focused scan is the same rule, run from fewer atoms."""

    def setUp(self):
        rng = np.random.default_rng(20260930)
        self.box = 18.0
        self.lattice = np.diag([self.box, self.box, self.box])
        coords = []
        atoms = []
        for _ in range(12):
            center = rng.uniform(0.5, self.box - 0.5, size=3)
            coords.append(center)
            coords.append(center + np.array([0.96, 0.0, 0.0]))
            coords.append(center + np.array([-0.24, 0.93, 0.0]))
            atoms.extend(['O', 'H', 'H'])
        self.atoms = atoms
        self.coords = np.mod(np.asarray(coords), self.box)

    def test_rows_agree_with_the_full_probe_on_the_reached_component(self):
        from pyrai3md.utilities.bonds import (perceive_bonds,
                                                   perceive_bonds_focus)
        full = perceive_bonds(self.atoms, self.coords, self.lattice)
        focus = [0, 1, 2]
        focal = perceive_bonds_focus(self.atoms, self.coords, self.lattice,
                                     focus)
        members = set(focal_pair[0] for focal_pair in focal)
        members.update(focal_pair[1] for focal_pair in focal)
        members.update(focus)
        expected = sorted(pair for pair in full
                          if pair[0] in members and pair[1] in members)
        self.assertEqual(focal, expected)
        self.assertGreaterEqual(len(focal), 2)

    def test_a_bond_outside_the_reached_set_is_not_returned(self):
        from pyrai3md.utilities.bonds import (perceive_bonds,
                                                   perceive_bonds_focus)
        full = perceive_bonds(self.atoms, self.coords, self.lattice)
        focal = perceive_bonds_focus(self.atoms, self.coords, self.lattice,
                                     [0])
        self.assertTrue(all(0 in (pair[0], pair[1]) for pair in focal))
        self.assertLess(len(focal), len(full))

    def test_an_empty_focus_and_no_cell_keep_the_shape(self):
        from pyrai3md.utilities.bonds import perceive_bonds_focus
        self.assertEqual(
            perceive_bonds_focus(self.atoms, self.coords, None, [0]), [])
        self.assertEqual(
            perceive_bonds_focus(self.atoms, self.coords, self.lattice, []), [])

    def test_the_oriented_row_is_the_same_whichever_end_scanned(self):
        from pyrai3md.utilities.bonds import perceive_bonds_focus
        one = perceive_bonds_focus(self.atoms, self.coords, self.lattice, [0])
        other = perceive_bonds_focus(self.atoms, self.coords, self.lattice, [1])
        self.assertEqual(one, other)



class TestBondPairsByType(unittest.TestCase):
    """The `shake_bond_type` shortcut's own matcher, including the wildcard.

    The graph is written out rather than perceived: this class is about which
    pairs a label selects, and a hand-made graph states that without depending
    on a geometry and a radius table agreeing.
    """

    GRAPH = {
        # 1-2, 1-3 and 2-8 are O-H bonds, 4-5 an H-H contact, 6-7 a C-H bond.
        'bonds': [(0, 1, [0, 0, 0]), (0, 2, [0, 0, 0]), (3, 4, [0, 0, 0]),
                  (5, 6, [0, 0, 0]), (1, 7, [0, 0, 0])],
        'atoms': ['O', 'H', 'H', 'H', 'H', 'C', 'H', 'O'],
    }

    def _rows(self, label):
        from pyrai3md.utilities.bonds import bond_pairs_by_type
        return bond_pairs_by_type(self.GRAPH, label)

    def test_x_h_takes_every_heavy_element_bonded_to_hydrogen(self):
        self.assertEqual(self._rows('X-H'), [[1, 2], [1, 3], [2, 8], [6, 7]])

    def test_the_wildcard_is_case_and_order_free(self):
        self.assertEqual(self._rows('x-h'), self._rows('X-H'))
        self.assertEqual(self._rows('H-X'), self._rows('X-H'))

    def test_an_h_h_pair_is_not_a_wildcard_match(self):
        """X is any element that is *not* hydrogen, so 4-5 cannot appear."""
        self.assertNotIn([4, 5], self._rows('X-H'))

    def test_a_named_pair_still_names_two_elements(self):
        self.assertEqual(self._rows('O-H'), [[1, 2], [1, 3], [2, 8]])
        self.assertEqual(self._rows('C-H'), [[6, 7]])

    def test_the_wildcard_pairs_with_hydrogen_and_nothing_else(self):
        for label in ('X-C', 'X-X', 'X-O'):
            with self.assertRaises(ValueError) as caught:
                self._rows(label)
            self.assertIn('pairs with', str(caught.exception))

    def test_a_one_word_label_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._rows('OH')
        self.assertIn('two element symbols', str(caught.exception))


class TestCertifiedSearch(unittest.TestCase):
    """Perception uses the cell module's certified nearest image."""

    # Two cell vectors 0.6 degrees apart: the gap between the planes they
    # span is 0.1 Angstrom while the vectors are 10 Angstrom long, so the
    # search cannot certify a nearest image.
    NOT_REDUCED = np.array([[10.0, 0.0, 0.0],
                            [9.99, 0.1, 0.0],
                            [0.0, 0.0, 10.0]])

    def test_an_exact_half_cell_tie_keeps_the_zero_shift(self):
        cell = np.diag([1.48, 5.0, 5.0])
        rows = perceive_bonds(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
                              cell)
        self.assertEqual(rows, [(0, 1, (0, 0, 0))])

    def test_a_singular_cell_perceives_nothing(self):
        self.assertEqual(
            perceive_bonds(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]]), [])

    def test_a_non_size_reduced_cell_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            perceive_bonds(['H', 'H'],
                           [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
                           self.NOT_REDUCED)
        self.assertIn('size-reduced', str(caught.exception))


class TestUnlistedPairBoundary(unittest.TestCase):
    """What the radius table, and only the table, decides.

    Two readings these cases pin. The listed-pair set is the table's key
    set, not a probe over a hand-written symbol list: the probe and the
    derived set measured the same 20 pairs, so the switch changed no
    coverage, and a pair the table gains now reaches the service without
    any edit in bonds.py. And the refusal window is the cutoff itself: the
    epsilon that used to widen it (`1.2 * scale * 1.0000001`) turned a
    length just past the cutoff into a defect, while the listed branch
    bonds only below the cutoff. Measured on the same geometry: 1.2
    exactly raises, and 1.2 * (1 + 5e-8) raised under the epsilon but
    does not now.
    """

    def _pair_at(self, length, symbols=('Xx', 'Yy')):
        return bonds.bond_graph(list(symbols),
                                [[0.0, 0.0, 0.0], [length, 0.0, 0.0]],
                                scale=1.0)

    def test_a_new_table_key_reaches_the_topology_service_alone(self):
        listed = {tuple(sorted(key.split('-'))) for key in bonds.BOND_LENGTHS}
        self.assertEqual(bonds._listed_pairs(), listed)
        with mock.patch.dict(bonds.BOND_LENGTHS, {'Xx-Yy': 1.5}):
            self.assertIn(('Xx', 'Yy'), bonds._listed_pairs())

    def test_the_dead_string_form_of_the_pair_test_is_gone(self):
        # `listed` holds tuples, so a '%s-%s' lookup could never hit and
        # the scan compared against it anyway. The pair label still
        # appears in the refusal message, so the assertion is on the
        # comparison, not on the format.
        body = source_read('pyrai3md/utilities/bonds.py').split(
            'def bond_graph')[1].split('\ndef ')[0]
        self.assertIn('if pair in listed:', body)
        self.assertNotIn("'%s-%s' % pair", body)

    def test_a_pair_on_the_default_cutoff_is_refused(self):
        with self.assertRaises(bonds.UnknownPairError):
            self._pair_at(1.2)

    def test_a_pair_just_past_the_cutoff_is_not_a_defect(self):
        self._pair_at(1.2 * (1.0 + 5e-8))


if __name__ == '__main__':
    unittest.main()
