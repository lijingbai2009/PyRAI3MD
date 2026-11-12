"""
======================================================================
Tests for explicit atom selection.

The rules under test are the ones that keep an index from being shifted
twice or from arriving unshifted: user input is 1-based in every format, the
plan is 0-based, and exactly one subtraction happens between the two.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities import selection


class TestParseIndexList(unittest.TestCase):

    def test_single_integer(self):
        self.assertEqual(selection.parse_index_list(4).tolist(), [4])

    def test_range_and_mixed_list(self):
        got = selection.parse_index_list([1, '3-5', 7]).tolist()
        self.assertEqual(got, [1, 3, 4, 5, 7])

    def test_comma_separated_string(self):
        self.assertEqual(selection.parse_index_list('2,4-6').tolist(),
                         [2, 4, 5, 6])

    def test_order_is_preserved_before_deduplication(self):
        self.assertEqual(selection.parse_index_list([5, 1, 5]).tolist(),
                         [5, 1, 5])

    def test_zero_is_refused_as_not_one_based(self):
        with self.assertRaises(selection.SelectionError):
            selection.parse_index_list([0, 1])

    def test_negative_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.parse_index_list([-2])

    def test_inverted_range_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.parse_index_list(['9-4'])

    def test_beyond_atom_count_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.parse_index_list([1, '2-9'], natom=5)

    def test_garbage_token_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.parse_index_list(['three'])

    def test_empty_selection_is_refused(self):
        for value in ([], '', None):
            with self.assertRaises(selection.SelectionError):
                selection.parse_index_list(value)


class TestZeroBasedConversion(unittest.TestCase):

    def test_subtracts_once(self):
        self.assertEqual(selection.to_zero_based([1, 4], natom=5).tolist(),
                         [0, 3])

    def test_refuses_an_already_converted_array(self):
        with self.assertRaises(selection.SelectionError):
            selection.to_zero_based([0, 3], natom=5)

    def test_upper_bound_is_checked_after_the_shift(self):
        with self.assertRaises(selection.SelectionError):
            selection.to_zero_based([6], natom=5)


class TestResolveIndices(unittest.TestCase):

    def test_atoms_source(self):
        got = selection.resolve_indices({'atoms': [3, '1-2']}, natom=4)
        self.assertEqual(got.tolist(), [0, 1, 2])

    def test_element_union_is_case_insensitive_and_deduplicated(self):
        atoms = ['O', 'H', 'H', 'C']
        got = selection.resolve_indices({'element': ['o', 'H']}, atoms=atoms)
        self.assertEqual(got.tolist(), [0, 1, 2])

    def test_element_matching_nothing_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_indices({'element': ['Au']}, atoms=['O', 'H'])

    def test_element_without_atom_list_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_indices({'element': ['O']}, natom=3)

    def test_group_reference_resolves_to_members(self):
        groups = {'base': {'atoms': [1, '3-4']}}
        got = selection.resolve_indices({'group': 'base'}, groups=groups,
                                        natom=6)
        self.assertEqual(got.tolist(), [0, 2, 3])

    def test_nested_group_reference(self):
        groups = {'inner': {'atoms': [2]}, 'outer': {'group': 'inner'}}
        self.assertEqual(
            selection.resolve_indices({'group': 'outer'}, groups=groups,
                                      natom=5).tolist(), [1])

    def test_unknown_group_lists_what_exists(self):
        with self.assertRaises(selection.SelectionError) as caught:
            selection.resolve_indices({'group': 'ghost'},
                                      groups={'real': {'atoms': [1]}},
                                      natom=3)
        self.assertIn('real', str(caught.exception))

    def test_circular_group_reference_is_refused(self):
        groups = {'a': {'group': 'b'}, 'b': {'group': 'a'}}
        with self.assertRaises(selection.SelectionError) as caught:
            selection.resolve_indices({'group': 'a'}, groups=groups, natom=4)
        self.assertIn('circular', str(caught.exception))

    def test_two_sources_in_one_selector_are_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_indices({'atoms': [1], 'element': ['H']},
                                      atoms=['H'], natom=2)

    def test_bond_type_needs_topology_and_is_refused(self):
        with self.assertRaises(selection.UnsupportedSelectionError):
            selection.resolve_indices({'bond_type': 'X-H'}, natom=4)

    def test_molecule_selection_is_refused(self):
        with self.assertRaises(selection.UnsupportedSelectionError):
            selection.resolve_indices({'molecule': 'water'}, natom=4)

    def test_no_source_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_indices({}, natom=4)

    def test_bare_list_is_treated_as_atoms(self):
        self.assertEqual(selection.resolve_indices([2, 1], natom=3).tolist(),
                         [0, 1])


class TestResolveBondPairs(unittest.TestCase):

    def test_pairs_are_ordered_and_sorted(self):
        got = selection.resolve_bond_pairs([[4, 2], [1, 3]], natom=5)
        self.assertEqual(got.tolist(), [[0, 2], [1, 3]])

    def test_duplicates_are_kept_for_the_caller_to_merge(self):
        got = selection.resolve_bond_pairs([[2, 1], [1, 2]], natom=3)
        self.assertEqual(got.tolist(), [[0, 1], [0, 1]])

    def test_self_pair_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_bond_pairs([[3, 3]], natom=4)

    def test_wrong_arity_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_bond_pairs([[1, 2, 3]], natom=4)

    def test_empty_list_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_bond_pairs([], natom=4)

    def test_out_of_range_pair_is_refused(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_bond_pairs([[1, 9]], natom=4)


class TestLeafDiscipline(unittest.TestCase):
    """Selection must not reach up into schema, data or workflow."""

    def test_no_framework_imports(self):
        import pyrai3md.utilities.selection as module
        self.assertFalse(hasattr(module, 'ConstraintPlan'))
        self.assertEqual(sorted(name for name in dir(module)
                                if name.startswith('pyrai3md')), [])
        with open(module.__file__) as handle:
            text = handle.read()
        self.assertNotIn('from pyrai3md.schema', text)
        self.assertNotIn('from pyrai3md.data', text)
        self.assertNotIn('from pyrai3md.workflow', text)
        self.assertIn('numpy as np', text)


if __name__ == '__main__':
    unittest.main()
