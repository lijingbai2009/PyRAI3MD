"""
======================================================================
Tests for the flat `&constraint` reader of the classic input.

The section is read where every other classic section is read: the interface
layer splits a file into `&section` headers and `keyword value` lines and knows
nothing about what a line means, and `schema.adapter` gives the tokens their
shape. These tests therefore call the adapter's reader, not the parser, and one
of them pins the opposite - that the parser hands over raw strings, so the
division of duty cannot quietly drift back into the interface layer.

What is pinned here is grammar, not arithmetic:

- one ';' segment is one object, and the objects of a family keep one order;
- a parameter line may say less than the objects do, which broadcasts or pads
  and is written down, or it may say more, which is refused;
- `wall_type` decides which wall geometries are read at all, and a keyword it
  does not read is reported rather than dropped;
- a ramp is five keys around the key it ramps, and what an object does not say
  falls back to that object's own value rather than to its neighbour's;
- the wall's own keys hold one more rule: `wall_groups` is the one line whose
  segments are a list rather than a value, because the list carries the molecule
  size at its end, and `wall_center` outranks `wall_ref` rather than merging;
- an input the grammar cannot mean is refused with the keyword it stumbled on,
  because a constraint that never reaches the run is the most expensive failure
  this section can have.

The structured spelling is the reference: two tests build the same declaration
both ways and compare plan fingerprints, which is what keeps the text front-end
from inventing a capability the schema does not have.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.schema import adapter
from pyrai3md.schema import build_plan, ConstraintInputError

ATOMS = ['O', 'H', 'H', 'O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.1173], [0.0, 0.7575, -0.4694], [0.0, -0.7575, -0.4694],
          [3.0, 0.0, 0.1173], [3.0, 0.7575, -0.4694], [3.0, -0.7575, -0.4694]]


def read(**section):
    """The mapping one `&constraint` section reads to."""
    return adapter._read_constraint_section(section)


class TestReaderDuty(unittest.TestCase):
    """Where the section is read, and what the interface layer keeps."""

    def test_the_interface_layer_splits_and_types_nothing(self):
        from pyrai3md.interface.parser import _parse_text
        text = ('&constraint\n'
                '  shake_bond 1 2; 3 4\n'
                '  shake_bond_ref 0.96\n')
        sections = _parse_text(text.split('\n'))
        self.assertEqual(sections['constraint']['shake_bond'], '1 2; 3 4')
        self.assertIsInstance(sections['constraint']['shake_bond'], str)

    def test_the_interface_layer_imports_no_constraint_grammar(self):
        """`interface/parser.py` must not grow a second reader for this section.

        The reason it had one was a block grammar the flat loop could not hold;
        with the grammar flat, the only thing left to do at that layer is stop
        owning keys at `&end`, which the generic loop already does.
        """
        import os
        path = os.path.join(os.path.dirname(adapter.__file__), os.pardir,
                            'interface', 'parser.py')
        with open(path) as handle:
            source = handle.read()
        self.assertNotIn('constraint_parser', source)

    def test_the_old_block_module_is_gone(self):
        import importlib
        import pkgutil
        import pyrai3md
        # Walk the package that was imported, not a directory spelled
        # 'pyrai3md' wherever the current working directory happens to be: a
        # relative path silently walks nothing, or a different tree, and on a
        # case-insensitive filesystem it walks back into this test's own file
        # and reports the module name as a leftover.
        found = [name for _finder, name, _ispkg
                 in pkgutil.walk_packages(pyrai3md.__path__,
                                          pyrai3md.__name__ + '.')
                 if name.endswith('.constraint_parser')]
        self.assertEqual(found, [])
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module('pyrai3md.interface.constraint_parser')


class TestObjects(unittest.TestCase):
    """One ';' segment is one object."""

    def test_segments_become_records_in_line_order(self):
        block = read(bias_bond='1 2; 3 4; 5 6',
                     bias_bond_ref='1.50; 1.42; 0.98')
        self.assertEqual(len(block['biases']), 3)
        self.assertEqual([item['atoms'] for item in block['biases']],
                         [[1, 2], [3, 4], [5, 6]])
        self.assertEqual([item['target'] for item in block['biases']],
                         [1.5, 1.42, 0.98])

    def test_the_object_lines_of_one_family_concatenate_in_file_order(self):
        """A bond, an angle and a dihedral are three objects of one family.

        The parameter lines count them together, in the order the lines were
        written: a file that starts with `shake_angle` means its first
        `shake_tol` to belong to the angle, not to a bond declared below it.
        """
        block = read(shake_bond='1 2', shake_angle='1 2 3',
                     shake_dihedral='1 2 3 4', shake_tol='1e-9; 1e-8; 1e-7')
        records = block['constraints']
        self.assertEqual([item['kind'] for item in records],
                         ['bond', 'angle', 'dihedral'])
        self.assertEqual([item['tolerance'] for item in records],
                         [1e-9, 1e-8, 1e-7])
        block = read(shake_dihedral='1 2 3 4', shake_bond='1 2',
                     shake_tol='1e-9; 1e-8')
        self.assertEqual([item['kind'] for item in block['constraints']],
                         ['dihedral', 'bond'])
        self.assertEqual([item['tolerance'] for item in block['constraints']],
                         [1e-9, 1e-8])

    def test_ranges_are_expanded_and_order_is_kept(self):
        block = read(fix_atoms='1-3; 6 4')
        self.assertEqual([item['atoms'] for item in block['fixes']],
                         [[1, 2, 3], [6, 4]])

    def test_commas_group_the_fields_of_one_segment(self):
        block = read(meta_cv='r13, r34; r45')
        self.assertEqual([item['cvs'] for item in block['metadynamics']],
                         [['r13', 'r34'], ['r45']])

    def test_indices_stay_one_based_for_the_schema(self):
        """The reader keeps the input's base; the schema decrements once."""
        block = read(shake_bond='1 2')
        self.assertEqual(block['constraints'][0]['bonds'], [[1, 2]])

    def test_an_empty_segment_is_refused_not_read_as_a_gap(self):
        for value in ('1 2;;3 4', '; 1 2', '1 2;'):
            with self.assertRaises(ValueError) as caught:
                read(bias_bond=value)
            self.assertIn('empty segment', str(caught.exception))

    def test_a_full_width_semicolon_names_the_replacement(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2；3 4')
        message = str(caught.exception)
        self.assertIn('\uff1b', message)
        self.assertIn(';', message)

    def test_a_row_of_the_wrong_width_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(shake_angle='1 2')
        self.assertIn('is not a multiple of the 3', str(caught.exception))

    def test_a_non_numeric_atom_is_named_not_echoed_as_a_type_error(self):
        with self.assertRaises(ValueError) as caught:
            read(shake_bond='1 x')
        self.assertIn('1-based atom indices', str(caught.exception))


class TestBroadcast(unittest.TestCase):
    """Fewer values than objects fills in, and says so; more is refused."""

    def test_one_segment_applies_to_every_object(self):
        block = read(shake_bond='1 2; 3 4; 5 6', shake_bond_ref='0.96')
        self.assertEqual([item['target'] for item in block['constraints']],
                         [0.96, 0.96, 0.96])
        self.assertIn("'shake_bond_ref' broadcast over 3 objects",
                      block['notes'])

    def test_a_shorter_list_pads_with_its_last_value(self):
        block = read(bias_bond='1 2; 3 4; 5 6',
                     bias_bond_fc='0.02; 0.04')
        self.assertEqual([item['force_constant'] for item in block['biases']],
                         [0.02, 0.04, 0.04])
        self.assertIn("'bias_bond_fc' padded with its last value to 3 objects",
                      block['notes'])

    def test_an_overlong_list_is_refused_because_a_object_was_miscounted(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2; 3 4', bias_bond_fc='0.02; 0.04; 0.06')
        message = str(caught.exception)
        self.assertIn('3 segments for 2 objects', message)
        self.assertIn('dropping the extras', message)

    def test_a_parameter_needs_the_objects_of_its_own_type(self):
        """`bias_angle_ref` cannot land on a bond, so it has nothing to hold."""
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2', bias_angle_ref='109.5')
        self.assertIn('no objects to apply to', str(caught.exception))

    def test_a_group_without_a_name_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(group_atoms='1-3')
        self.assertIn('no objects to apply to', str(caught.exception))


class TestWallSwitch(unittest.TestCase):
    """`wall_type` is read first and decides which keywords mean anything."""

    def test_absent_switch_reads_no_wall_and_reports_what_it_skipped(self):
        block = read(wall_sphere_radius='14.0', wall_fc='0.001',
                     wall_atoms='1-6')
        self.assertNotIn('walls', block)
        joined = ' '.join(block['notes'])
        for key in ('wall_sphere_radius', 'wall_fc', 'wall_atoms'):
            self.assertIn("'%s'" % key, joined)
        self.assertIn('wall_type reads no wall', joined)

    def test_none_is_the_explicit_way_to_switch_the_block_off(self):
        block = read(wall_type='none', wall_sphere_radius='14.0')
        self.assertNotIn('walls', block)

    def test_the_named_types_are_read_in_the_order_the_switch_names(self):
        block = read(wall_type='box sphere', wall_sphere_radius='14.0',
                     wall_box_lengths='10 10 10', wall_fc='0.001')
        self.assertEqual([item['geometry'] for item in block['walls']],
                         ['box', 'sphere'])

    def test_two_segments_of_one_type_are_two_walls(self):
        block = read(wall_type='sphere', wall_sphere_radius='14.0; 12.0',
                     wall_center='0 0 0', wall_fc='0.001')
        self.assertEqual(len(block['walls']), 2)
        self.assertEqual([item['radius'] for item in block['walls']],
                         [14.0, 12.0])
        self.assertIn("'wall_center' broadcast over 2 objects",
                      block['notes'])

    def test_a_geometry_the_switch_does_not_name_is_ignored_with_a_note(self):
        block = read(wall_type='sphere', wall_sphere_radius='14.0',
                     wall_box_lengths='10 10 10', wall_fc='0.001')
        self.assertEqual(len(block['walls']), 1)
        self.assertIn(
            "'wall_box_lengths' ignored: 'wall_type' does not name that "
            'geometry', block['notes'])

    def test_a_geometry_the_switch_names_but_never_gives_is_not_a_wall(self):
        block = read(wall_type='sphere ellipsoid', wall_sphere_radius='14.0',
                     wall_fc='0.001')
        self.assertEqual(len(block['walls']), 1)
        self.assertEqual(block['walls'][0]['geometry'], 'sphere')

    def test_mixing_none_with_a_geometry_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='none sphere', wall_sphere_radius='14.0')
        self.assertIn('cannot mix', str(caught.exception))

    def test_an_unknown_geometry_name_lists_the_ones_that_exist(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='cylinder')
        message = str(caught.exception)
        self.assertIn('cylinder', message)
        self.assertIn('triclinic', message)

    def test_the_same_type_twice_is_refused_because_a_segment_is_the_way(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='sphere sphere', wall_sphere_radius='14.0; 12.0')
        self.assertIn("names sphere twice", str(caught.exception))

    def test_a_triclinic_cell_takes_six_parameters_or_nine(self):
        block = read(wall_type='triclinic', wall_fc='0.001',
                     wall_atoms='1-6',
                     wall_triclinic_cell='8 8 8 90 90 120')
        self.assertEqual(block['walls'][0]['cell'],
                         [8.0, 8.0, 8.0, 90.0, 90.0, 120.0])
        block = read(wall_type='triclinic', wall_fc='0.001', wall_atoms='1-6',
                     wall_triclinic_cell='8 0 0 0 8 0 0 0 8')
        self.assertEqual(len(block['walls'][0]['cell']), 9)

    def test_a_cell_of_seven_numbers_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='triclinic', wall_triclinic_cell='8 8 8 90 90 90 1')
        self.assertIn('six cell parameters', str(caught.exception))


class TestWallGroupingKeywords(unittest.TestCase):
    """`wall_groups`, `wall_ref`, `wall_center` and `wall_law`.

    The four share one grammar decision and one design decision. The decision is
    that `;` separates objects, so each of these gives one value per wall and
    never a list inside one wall's value - except `wall_groups`, whose segments
    are themselves a list, cut by the molecule size written at its end. The
    decision is that the input says how to group and the target says how to
    weight, so a wall over six atoms in two molecules of three is one line.
    """

    def _plan(self, mapping, **kwargs):
        return build_plan(mapping, atoms=ATOMS, natom=len(ATOMS),
                          masses=[15.999, 1.008, 1.008] * 2,
                          coordinates=COORDS, **kwargs)

    def test_the_trailing_number_cuts_the_list_into_molecules(self):
        block = read(wall_type='sphere', wall_sphere_radius='4.0',
                     wall_fc='0.001', wall_target='com',
                     wall_groups='1-3 4-6 3')
        self.assertEqual(block['walls'][0]['molecules'],
                         [[1, 2, 3], [4, 5, 6]])

    def test_ranges_are_expanded_before_the_cut(self):
        """The user's own spelling: consecutive numbers from a range and single
        atoms mix freely, because the cut happens on the expanded list."""
        block = read(wall_type='sphere', wall_sphere_radius='4.0',
                     wall_fc='0.001', wall_target='com',
                     wall_groups='1-3 4 5 6 3')
        self.assertEqual(block['walls'][0]['molecules'],
                         [[1, 2, 3], [4, 5, 6]])

    def test_a_list_that_does_not_divide_is_refused_not_left_short(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='sphere', wall_sphere_radius='4.0', wall_fc='0.001',
                 wall_target='com', wall_groups='1 2 3 4 5 2')
        self.assertIn('does not divide', str(caught.exception))

    def test_a_molecule_size_of_zero_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='sphere', wall_sphere_radius='4.0', wall_fc='0.001',
                 wall_target='com', wall_groups='1 2 0')
        self.assertIn('at least one atom', str(caught.exception))

    def test_one_segment_per_wall(self):
        """Two segments are two walls, each with its own partition, which is what
        the switch order and the alignment rule already decide for every other
        wall keyword."""
        block = read(wall_type='sphere', wall_sphere_radius='4.0; 3.5',
                     wall_fc='0.001', wall_target='com; com',
                     wall_groups='1-3 3; 4-6 3')
        self.assertEqual(len(block['walls']), 2)
        self.assertEqual(block['walls'][0]['molecules'], [[1, 2, 3]])
        self.assertEqual(block['walls'][1]['molecules'], [[4, 5, 6]])

    def test_the_groups_are_the_selection_when_nothing_else_selects(self):
        """Typing the same atom list twice, once to confine and once to group,
        would be a place for the two to disagree; the plan says in the run record
        which atoms it took."""
        plan = self._plan(read(wall_type='sphere', wall_sphere_radius='4.0',
                               wall_fc='0.001', wall_target='com',
                               wall_groups='4-6 3'))
        spec = plan.spec(plan.ids('wall')[0])
        self.assertEqual(list(spec.atoms), [3, 4, 5])
        self.assertTrue(any('confines the 3 atoms its groups name' in note
                            for note in plan.notes))

    def test_a_selection_and_groups_that_disagree_are_refused(self):
        """The groups subdivide the confined atoms, so an atom in a molecule that
        the wall does not confine means one of the two lines is not what it says."""
        with self.assertRaises(ConstraintInputError) as caught:
            self._plan(read(wall_type='sphere', wall_sphere_radius='4.0',
                            wall_fc='0.001', wall_target='com',
                            wall_atoms='1-3', wall_groups='1-3 4-6 3'))
        self.assertIn('does not contain', str(caught.exception))

    def test_wall_ref_is_the_anchor_and_wall_center_outranks_it(self):
        plan = self._plan(read(wall_type='sphere', wall_sphere_radius='4.0',
                               wall_fc='0.001', wall_atoms='1-3',
                               wall_ref='4 5 6'))
        spec = plan.spec(plan.ids('wall')[0])
        self.assertEqual(spec.parameters['anchor'], [3, 4, 5])
        self.assertNotIn('center', spec.parameters)
        both = self._plan(read(wall_type='sphere', wall_sphere_radius='4.0',
                               wall_fc='0.001', wall_atoms='1-3',
                               wall_ref='4 5 6', wall_center='1 2 3'))
        declared = both.spec(both.ids('wall')[0])
        self.assertNotIn('anchor', declared.parameters)
        self.assertEqual(len(declared.parameters['center']), 3)

    def test_none_is_the_way_to_declare_no_center_for_one_wall(self):
        """`none` in one of these three positions is not a value: it says this
        object takes the default, which the positional line cannot otherwise
        tell apart from a gap."""
        block = read(wall_type='sphere', wall_sphere_radius='4.0; 3.5',
                     wall_fc='0.001', wall_atoms='1-3',
                     wall_ref='none; 1 2', wall_groups='none; 1-3 3')
        self.assertEqual(block['walls'][0]['ref_atoms'], [])
        self.assertEqual(block['walls'][1]['ref_atoms'], [1, 2])
        self.assertEqual(block['walls'][0]['molecules'], [])
        plan = self._plan(block)
        self.assertNotIn('anchor', plan.spec(plan.ids('wall')[0]).parameters)
        self.assertEqual(plan.spec(plan.ids('wall')[1]).parameters['anchor'],
                         [0, 1])

    def test_the_law_is_one_word_per_wall_and_a_stray_word_is_refused(self):
        block = read(wall_type='sphere', wall_sphere_radius='4.0; 3.5',
                     wall_fc='0.001', wall_atoms='1-6',
                     wall_law='soft; hard')
        self.assertEqual([record['law'] for record in block['walls']],
                         ['soft', 'hard'])
        with self.assertRaises(ValueError) as caught:
            read(wall_type='sphere', wall_sphere_radius='4.0', wall_fc='0.001',
                 wall_atoms='1-6', wall_law='soft hard')
        self.assertIn('one word per segment', str(caught.exception))

    def test_an_unknown_law_names_the_two_that_exist(self):
        with self.assertRaises(Exception) as caught:
            self._plan({'walls': [{'geometry': 'sphere', 'radius': 4.0,
                                   'force_constant': 0.001, 'atoms': [1, 2, 3],
                                   'law': 'spring'}]})
        self.assertIn('hard', str(caught.exception))
        self.assertIn('soft', str(caught.exception))

    def test_the_law_changes_the_units_of_the_force_constant(self):
        """`soft` divides by a length already, so its constant is an energy and
        not an energy over a length to the power: the same declared number means
        two different internal numbers, which is why the two laws do not share a
        conversion helper."""
        hard = self._plan(read(wall_type='sphere', wall_sphere_radius='4.0',
                               wall_fc='0.001', wall_atoms='1-3'))
        soft = self._plan(read(wall_type='sphere', wall_sphere_radius='4.0',
                               wall_fc='0.001', wall_atoms='1-3',
                               wall_law='soft'))
        first = hard.spec(hard.ids('wall')[0])
        second = soft.spec(soft.ids('wall')[0])
        self.assertNotAlmostEqual(first.parameters['force_constant'],
                                  second.parameters['force_constant'],
                                  places=12)


class TestRamps(unittest.TestCase):
    """Five keys around the key they ramp, falling back per object."""

    def test_an_endpoint_family_becomes_one_inline_schedule(self):
        block = read(bias_bond='1 2', bias_bond_fc='0.02',
                     bias_bond_fc_start='0.005', bias_bond_fc_stop='5000')
        schedules = block['biases'][0]['schedules']
        self.assertEqual(schedules, [{'parameter': 'force_constant',
                                      'start_step': 1, 'end_step': 5000,
                                      'value': [0.005, 0.02]}])

    def test_an_absent_endpoint_falls_back_to_that_objects_own_value(self):
        """Two objects, one ramp: neither inherits the other's number."""
        block = read(bias_bond='1 2; 3 4', bias_bond_fc='0.02; 0.08',
                     bias_bond_fc_start='0.005')
        first, second = (item['schedules'][0] for item in block['biases'])
        self.assertEqual(first['value'], [0.005, 0.02])
        self.assertEqual(second['value'], [0.005, 0.08])

    def test_delay_defaults_to_the_first_step(self):
        block = read(wall_type='sphere', wall_sphere_radius='14.0',
                     wall_fc='0.001', wall_fc_end='0.02')
        self.assertEqual(block['walls'][0]['schedules'][0]['start_step'], 1)

    def test_stop_is_left_open_for_the_schema_to_close(self):
        """The reader cannot see the length of the run, and does not guess it."""
        block = read(wall_type='sphere', wall_sphere_radius='14.0',
                     wall_fc='0.001', wall_fc_end='0.02')
        self.assertNotIn('end_step', block['walls'][0]['schedules'][0])

    def test_scale_is_two_factors_and_cannot_mix_with_the_endpoints(self):
        block = read(meta_cv='r13', meta_width='0.1', meta_height='0.01',
                     meta_height_scale='0 1')
        schedule = block['metadynamics'][0]['schedules'][0]
        self.assertEqual(schedule['scale'], [0.0, 1.0])
        self.assertNotIn('value', schedule)

    def test_scale_with_an_endpoint_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2', bias_bond_fc='0.02', bias_bond_fc_scale='0 1',
                 bias_bond_fc_end='0.04')
        self.assertIn('never both', str(caught.exception))

    def test_a_window_without_endpoints_ramps_nothing_and_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2', bias_bond_fc='0.02', bias_bond_fc_delay='10',
                 bias_bond_fc_stop='5000')
        self.assertIn('no endpoints', str(caught.exception))

    def test_a_ramp_on_a_key_that_cannot_ramp_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2', bias_bond_ref='1.5', bias_bond_ref_start=1)
        self.assertIn('does not know', str(caught.exception))

    def test_a_ramp_needs_the_base_value_it_falls_back_to(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2', bias_bond_fc_start='0.005')
        self.assertIn('ramps a value this object does not carry',
                      str(caught.exception))

    def test_component_ramps_compose_one_vector_from_the_objects_base(self):
        block = read(wall_type='ellipsoid', wall_ellipsoid_semiaxes='3 4 5',
                     wall_fc='0.001', wall_atoms='1-6',
                     wall_x_start='2.5', wall_z_end='6.0',
                     wall_y_start='4.0', wall_y_end='4.0')
        schedule = block['walls'][0]['schedules'][0]
        self.assertEqual(schedule['parameter'], 'semiaxes')
        self.assertEqual(schedule['value'], [[2.5, 4.0, 5.0],
                                             [3.0, 4.0, 6.0]])

    def test_a_component_ramp_leaves_the_other_components_at_base(self):
        block = read(wall_type='box', wall_box_lengths='10 10 10',
                     wall_fc='0.001', wall_atoms='1-6',
                     wall_x_end='8.0')
        self.assertEqual(block['walls'][0]['schedules'][0]['value'],
                         [[10.0, 10.0, 10.0], [8.0, 10.0, 10.0]])

    def test_the_components_of_one_vector_share_their_window(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='box', wall_box_lengths='10 10 10',
                 wall_fc='0.001', wall_atoms='1-6',
                 wall_x_end='8.0', wall_x_delay='10',
                 wall_y_end='8.0', wall_y_delay='20')
        self.assertIn('share their schedule', str(caught.exception))

    def test_a_component_ramp_next_to_the_vector_ramp_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='box', wall_box_lengths='10 10 10',
                 wall_fc='0.001', wall_atoms='1-6',
                 wall_box_lengths_start='9 9 9', wall_box_lengths_end='10 10 10',
                 wall_x_end='8.0')
        self.assertIn('ramp the same lengths twice', str(caught.exception))

    def test_a_component_ramp_on_a_sphere_is_refused_with_the_right_advice(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='sphere', wall_sphere_radius='14.0',
                 wall_fc='0.001', wall_atoms='1-6', wall_x_end='8.0')
        self.assertIn('wall_sphere_radius_start', str(caught.exception))

    def test_angles_ramp_only_a_six_parameter_cell(self):
        block = read(wall_type='triclinic', wall_triclinic_cell='8 8 8 90 90 90',
                     wall_fc='0.001', wall_atoms='1-6',
                     wall_gamma_end='120.0')
        self.assertEqual(block['walls'][0]['schedules'][0]['value'],
                         [[8.0, 8.0, 8.0, 90.0, 90.0, 90.0],
                          [8.0, 8.0, 8.0, 90.0, 90.0, 120.0]])
        with self.assertRaises(ValueError) as caught:
            read(wall_type='triclinic', wall_triclinic_cell='8 0 0 0 8 0 0 0 8',
                 wall_fc='0.001', wall_atoms='1-6', wall_gamma_end='120.0')
        self.assertIn('nine-component cell', str(caught.exception))

    def test_a_geometry_line_ramp_takes_the_shapes_of_that_geometry(self):
        block = read(wall_type='ellipsoid', wall_ellipsoid_semiaxes='3 4 5',
                     wall_fc='0.001', wall_atoms='1-6',
                     wall_ellipsoid_semiaxes_start='2 3 4',
                     wall_ellipsoid_semiaxes_end='4 5 6')
        self.assertEqual(block['walls'][0]['schedules'][0]['value'],
                         [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])

    def test_an_endpoint_of_the_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(wall_type='ellipsoid', wall_ellipsoid_semiaxes='3 4 5',
                 wall_fc='0.001', wall_atoms='1-6',
                 wall_ellipsoid_semiaxes_end='4')
        self.assertIn('needs 3 numbers per segment', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            read(wall_type='ellipsoid', wall_ellipsoid_semiaxes='3 4 5',
                 wall_fc='0.001', wall_atoms='1-6',
                 wall_ellipsoid_semiaxes_end='4 5 6 7')
        self.assertIn('needs 3 numbers per segment', str(caught.exception))


class TestListsAreTheSegments(unittest.TestCase):
    """A YAML or JSON list is the ';' segmentation, keyword for keyword.

    The one rule is that a list element is one object: a scalar for an object of
    one token, its own list for an object of several. That is what makes the
    three carriers one grammar rather than three dialects, and these tests
    compare the mapping a list reads to against the mapping the same declaration
    as classic text reads to.
    """

    def test_three_objects_are_three_elements(self):
        as_lists = read(wall_type='sphere',
                        wall_sphere_radius=[1.0, 2.0, 3.0],
                        wall_atoms=[[1, 2, 3], [4], [5, 6]])
        as_text = read(wall_type='sphere',
                       wall_sphere_radius='1.0; 2.0; 3.0',
                       wall_atoms='1 2 3; 4; 5 6')
        self.assertEqual(as_lists, as_text)
        self.assertEqual(len(as_lists['walls']), 3)

    def test_one_object_of_several_tokens_is_a_nested_row(self):
        one = read(wall_type='sphere', wall_sphere_radius=2.8,
                   wall_atoms=[[1, 2, 3]])
        self.assertEqual(one['walls'][0]['atoms'], [1, 2, 3])
        self.assertEqual(one, read(wall_type='sphere', wall_sphere_radius=2.8,
                                   wall_atoms='1 2 3'))
        # The flat list is three objects, and one wall cannot hold three.
        with self.assertRaises(ValueError) as caught:
            read(wall_type='sphere', wall_sphere_radius=2.8,
                 wall_atoms=[1, 2, 3])
        self.assertIn('3 segments for 1 objects', str(caught.exception))

    def test_one_object_can_carry_several_rows(self):
        """A named block of bonds is one object, not two declarations.

        `shake_bond 1 2 1 3` is the two bonds of one block, while
        `shake_bond 1 2; 1 3` is two blocks of one bond each. The rows have a
        fixed width per keyword, so a segment that is not a whole number of them
        is a miscount rather than a second reading.
        """
        block = read(shake_bond=[[1, 2, 1, 3]], shake_bond_ref=0.96,
                     shake_id='oh_bonds')
        self.assertEqual(len(block['constraints']), 1)
        self.assertEqual(block['constraints'][0]['bonds'], [[1, 2], [1, 3]])
        self.assertEqual(block, read(shake_bond='1 2 1 3',
                                     shake_bond_ref='0.96',
                                     shake_id='oh_bonds'))
        two = read(shake_bond='1 2; 1 3', shake_bond_ref='0.96')
        self.assertEqual(len(two['constraints']), 2)
        with self.assertRaises(ValueError) as caught:
            read(shake_bond='1 2 3')
        self.assertIn('multiple of the 2', str(caught.exception))

    def test_a_null_element_is_the_placeholder_the_text_writes_as_none(self):
        as_lists = read(wall_type='sphere',
                        wall_sphere_radius=[2.8, 2.8],
                        wall_center=[[0.0, 0.0, 0.0], None],
                        wall_fc=0.002)
        as_text = read(wall_type='sphere', wall_sphere_radius='2.8; 2.8',
                       wall_center='0 0 0; none', wall_fc=0.002)
        self.assertEqual(as_lists, as_text)
        self.assertEqual(as_lists['walls'][0]['center'], [0.0, 0.0, 0.0])
        self.assertEqual(as_lists['walls'][1].get('center', []), [])

    def test_the_retired_spellings_are_answered_by_name(self):
        """Each one was a spelling the subsystem carried; none is 'unknown'.

        The structured records are what the reader emits and what the schema
        consumes, so a refusal that only said 'not a keyword' would leave a user
        of the old block guessing. Every message names the flat line to write.
        """
        replacements = {
            'version': 'no version keyword',
            'units': 'unit_length',
            'notes': 'not a declaration',
            'schedules': 'bias_bond_fc_start',
            'groups': 'group_name',
            'cvs': 'cv_bond',
            'biases': 'bias_bond',
            'walls': 'wall_type',
            'fixes': 'fix_atoms',
            'constraints': 'shake_bond',
            'metadynamics': 'meta_cv',
            'fix_ref': 'fix_dir',
        }
        for retired, replacement in sorted(replacements.items()):
            with self.assertRaises(ValueError) as caught:
                read(**{retired: [{'id': 'x'}], 'shake_bond': '1 2'})
            message = str(caught.exception)
            self.assertIn("'%s' is refused" % retired, message)
            self.assertIn(replacement, message)


class TestRefusedKeywords(unittest.TestCase):
    """A spelling the subsystem has carried is answered, not called unknown."""

    def test_a_force_constant_on_a_strict_constraint_is_refused(self):
        for key in ('shake_fc', 'shake_bond_fc'):
            with self.assertRaises(ValueError) as caught:
                read(**{key: '10.0', 'shake_bond': '1 2'})
            self.assertIn('carries no potential', str(caught.exception))

    def test_a_generic_parameter_name_is_refused_with_the_typed_one(self):
        with self.assertRaises(ValueError) as caught:
            read(bias_bond='1 2', bias_ref='1.5')
        self.assertIn('bias_bond_ref', str(caught.exception))

    def test_the_legacy_molecule_spellings_are_refused_in_this_section(self):
        with self.assertRaises(ValueError) as caught:
            read(cbond='1 2')
        message = str(caught.exception)
        self.assertIn('legacy', message)
        self.assertIn('shake_bond', message)

    def test_an_unknown_keyword_says_the_section_is_flat(self):
        """The nested block grammar is gone; a leftover line is told so.

        An archived `&system_constraint` block reaches this layer as flat lines,
        so the refusal has to name the shape the section reads rather than only
        the word it did not recognise.
        """
        with self.assertRaises(ValueError) as caught:
            read(restraint='1 2')
        message = str(caught.exception)
        self.assertIn('flat', message)
        self.assertIn('restraint', message)

    def test_a_retired_unit_section_is_refused_with_the_keys_that_replace_it(
            self):
        with self.assertRaises(ValueError) as caught:
            read(units='length angstrom')
        message = str(caught.exception)
        self.assertIn("'units' is refused", message)
        self.assertIn('unit_length', message)

    def test_a_stray_nested_key_is_refused_with_its_family_named(self):
        with self.assertRaises(ValueError) as caught:
            read(wall='sphere', wall_type='sphere',
                 wall_sphere_radius='14.0')
        self.assertIn('wall', str(caught.exception))


class TestUnitsAndStructured(unittest.TestCase):
    """The unit keys, and the refusal of the records the reader used to pass."""

    def test_unit_keywords_become_the_units_mapping(self):
        block = read(unit_length='bohr', unit_angle='radian',
                     unit_energy='kcal', bias_bond='1 2',
                     bias_bond_ref='1.5', bias_bond_fc='0.02')
        self.assertEqual(block['units'], {'length': 'bohr',
                                         'angle': 'radian',
                                         'energy': 'kcal'})

    def test_a_unit_keyword_with_two_segments_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            read(unit_length='bohr; angstrom')
        self.assertIn('one unit name', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            read(unit_length='bohr angstrom')
        self.assertIn('one word per segment', str(caught.exception))

    def test_structured_records_are_refused_with_the_flat_line_named(self):
        """The records are the reader's output, not a spelling of an input.

        One section read two ways says nothing about which reading the run
        applies, so a record list is refused by name and the refusal names the
        keyword line of the same family.
        """
        with self.assertRaises(ValueError) as caught:
            read(biases=[{'atoms': [1, 2], 'target': 1.5,
                          'force_constant': 0.02, 'id': 'hold'}])
        message = str(caught.exception)
        self.assertIn("'biases' is refused", message)
        self.assertIn('bias_bond', message)

    def test_one_family_has_no_second_spelling_to_clash_with(self):
        with self.assertRaises(ValueError) as caught:
            read(walls=[{'geometry': 'sphere', 'radius': 14.0}],
                 wall_type='sphere', wall_sphere_radius='12.0')
        self.assertIn("'walls' is refused", str(caught.exception))

    def test_the_retired_section_name_routes_to_the_same_reader(self):
        sections = {'constraint': {'wall_type': 'sphere',
                                   'wall_sphere_radius': '14.0',
                                   'wall_fc': '0.001', 'wall_atoms': '1-6'},
                    'system_constraint': {'shake_bond': '1 2'}}
        first = adapter._constraint_declaration(sections['constraint'])
        second = adapter._constraint_declaration(sections['system_constraint'])
        self.assertEqual(first['walls'][0]['radius'], 14.0)
        self.assertEqual(second['constraints'][0]['bonds'], [[1, 2]])

    def test_a_section_of_only_retired_keys_refuses_every_one_of_them(self):
        with self.assertRaises(ValueError) as caught:
            adapter._constraint_declaration(
                {'units': {'length': 'bohr'},
                 'constraints': [{'kind': 'bond', 'bonds': [[1, 2]]}]})
        message = str(caught.exception)
        self.assertIn("'units' is refused", message)
        self.assertIn("'constraints' is refused", message)


class TestTextReachesThePlan(unittest.TestCase):
    """The reader's output is the schema's input: one plan, two spellings."""

    def _plan(self, mapping, **kwargs):
        return build_plan(mapping, atoms=ATOMS, natom=len(ATOMS),
                          masses=[15.999, 1.008, 1.008] * 2,
                          coordinates=COORDS, **kwargs)

    def test_the_text_and_the_mapping_give_the_same_fingerprint(self):
        section = read(shake_bond='1 2; 4 5', shake_bond_ref='0.96',
                       bias_bond='1 3', bias_bond_ref='0.90',
                       bias_bond_fc='0.02', bias_id='hold',
                       fix_atoms='6', fix_dir='z')
        structured = {'constraints': [
            {'kind': 'bond', 'bonds': [[1, 2]], 'target': 0.96},
            {'kind': 'bond', 'bonds': [[4, 5]], 'target': 0.96}],
            'biases': [{'atoms': [1, 3], 'target': 0.90,
                        'force_constant': 0.02, 'id': 'hold'}],
            'fixes': [{'atoms': [6], 'direction': 'z'}]}
        first = self._plan(section)
        second = self._plan(structured)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_ids_are_minted_by_appearance_when_the_input_gives_none(self):
        plan = self._plan(read(bias_bond='1 2; 3 4', bias_bond_ref='1.5; 1.4',
                               bias_bond_fc='0.02'))
        self.assertEqual(plan.ids(), ['bias1', 'bias2'])

    def test_an_open_stop_closes_at_the_runs_own_step_and_says_so(self):
        plan = self._plan(read(wall_type='sphere', wall_sphere_radius='2.4',
                               wall_center='0 0 0', wall_atoms='1-3',
                               wall_fc='0.001', wall_fc_end='0.02'),
                          md_step=4000)
        record = plan.schedules['schedule_of_wall1']
        self.assertEqual((record.start_step, record.end_step), (1, 4000))
        self.assertTrue(any('md.step = 4000' in note for note in plan.notes))

    def test_an_open_stop_without_a_declared_step_count_is_refused(self):
        with self.assertRaises(ConstraintInputError) as caught:
            self._plan(read(wall_type='sphere', wall_sphere_radius='2.4',
                            wall_center='0 0 0', wall_atoms='1-3',
                            wall_fc='0.001', wall_fc_end='0.02'))
        self.assertIn('declares no step count', str(caught.exception))

    def test_the_notes_survive_into_the_plan(self):
        plan = self._plan(read(bias_bond='1 2; 3 4', bias_bond_ref='1.5',
                               bias_bond_fc='0.02'))
        self.assertTrue(any('broadcast over 2 objects' in note
                            for note in plan.notes))

    def test_a_vector_ramp_is_converted_as_a_vector_not_as_two_scalars(self):
        plan = self._plan(read(wall_type='ellipsoid',
                               wall_ellipsoid_semiaxes='3 4 5',
                               wall_center='0 0 0', wall_atoms='1-6',
                               wall_fc='0.001', wall_x_end='2.0',
                               wall_x_stop='1000'), md_step=1000)
        record = plan.schedules['schedule_of_wall1']
        from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
        self.assertAlmostEqual(record.value[1][0], 2.0 * ANGSTROM_TO_BOHR)
        self.assertEqual(len(record.value[0]), 3)

    def test_a_cell_ramp_arrives_as_endpoint_matrices_in_internal_units(self):
        plan = self._plan(read(wall_type='triclinic', wall_atoms='1-6',
                               wall_triclinic_cell='8 8 8 90 90 90',
                               wall_fc='0.001', wall_center='4 4 4',
                               wall_triclinic_cell_end='7 7 7 90 90 90',
                               wall_triclinic_cell_stop='1000'),
                          md_step=1000)
        record = plan.schedules['schedule_of_wall1']
        self.assertEqual(len(record.value), 2)
        self.assertEqual(np_shape(record.value[1]), (3, 3))
        self.assertAlmostEqual(record.value[1][0][0], 7.0 * 1.8897261246)

    def test_a_non_positive_endpoint_is_refused_at_input_time(self):
        with self.assertRaises(ConstraintInputError) as caught:
            self._plan(read(wall_type='sphere', wall_sphere_radius='2.4',
                            wall_center='0 0 0', wall_atoms='1-3',
                            wall_fc='0.001', wall_sphere_radius_end='0.0',
                            wall_sphere_radius_stop='100'), md_step=100)
        self.assertIn('non-positive endpoint', str(caught.exception))


def np_shape(value):
    """The shape of a nested endpoint list, without numpy at the top level."""
    import numpy as np
    return np.asarray(value, dtype=float).shape


class TestWriter(unittest.TestCase):
    """The same grammar read backwards: a mapping becomes text that reads back."""

    ATOMS = ['O', 'H', 'H']
    GEOMETRY = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]

    def _both(self, block):
        """(plan from the mapping, plan from the text the mapping writes)."""
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        pairs = write_constraint_section(block)
        written = dict(pairs)
        self.assertNotIn('notes', written,
                         'a generated section must not carry the reader notes')
        text = read(**written) if written else {}
        common = dict(atoms=self.ATOMS, natom=3, masses=[16.0, 1.0, 1.0],
                      coordinates=self.GEOMETRY, md_step=1000)
        return (self._plan(block, md_step=1000),
                self._plan(text, md_step=1000))

    def _plan(self, mapping, **kwargs):
        return build_plan(mapping, atoms=self.ATOMS, natom=3,
                          masses=[16.0, 1.0, 1.0],
                          coordinates=self.GEOMETRY, **kwargs)

    def test_a_full_block_writes_and_reads_back_as_the_same_plan(self):
        block = {
            'units': {'length': 'angstrom', 'angle': 'degree',
                      'energy': 'hartree'},
            'groups': [{'name': 'oh', 'atoms': [1, 2]}],
            'cvs': [{'name': 'r1', 'kind': 'bond', 'atoms': [1, 2]}],
            'biases': [{'id': 'stretch', 'cv': 'r1', 'target': 0.72,
                        'force_constant': 0.05,
                        'schedules': [{'parameter': 'force_constant',
                                       'start_step': 10, 'end_step': 500,
                                       'value': [0.01, 0.05]}]}],
            'walls': [
                {'id': 'vessel', 'geometry': 'sphere', 'center': [0, 0, 0],
                 'radius': 1.5, 'atoms': [1, 2, 3], 'force_constant': 0.01},
                {'id': 'cage', 'geometry': 'ellipsoid', 'center': [0, 0, 0],
                 'semiaxes': [1.0, 2.0, 3.0], 'atoms': [1, 2],
                 'force_constant': 0.02, 'power': 4}],
            'fixes': [{'id': 'frozen', 'atoms': [3], 'direction': 'z'}],
            'constraints': [{'id': 'ohb', 'kind': 'bond', 'bonds': [[1, 2]],
                             'velocity_constraint': 'rattle'}],
            'metadynamics': [{'id': 'hill', 'cvs': ['r1'], 'height': 0.01,
                              'width': [0.1], 'pace': 50}],
        }
        first, second = self._both(block)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_the_wall_grouping_keys_survive_the_round_trip(self):
        """`wall_law`, `wall_ref` and `wall_groups` are written back as one
        segment per wall, and a partition comes back as its atom list plus the
        size it was cut by. A wall that names no anchor is written `none`, which
        is the same word the switch uses to name nothing."""
        block = {
            'walls': [
                {'id': 'cavity', 'geometry': 'ellipsoid', 'law': 'soft',
                 'semiaxes': [1.0, 2.0, 3.0], 'atoms': [1, 2, 3],
                 'target': 'com', 'molecules': [[1], [2], [3]],
                 'force_constant': 0.001},
                {'id': 'shell', 'geometry': 'sphere', 'radius': 1.5,
                 'ref_atoms': [1, 2], 'atoms': [1, 2, 3],
                 'molecules': [[1, 2, 3]], 'target': 'com',
                 'force_constant': 0.01}]}
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        written = dict(write_constraint_section(block))
        self.assertEqual(written['wall_law'], 'soft; hard')
        self.assertEqual(written['wall_groups'], '1 2 3 1; 1 2 3 3')
        self.assertEqual(written['wall_ref'], 'none; 1 2')
        first, second = self._both(block)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_one_wall_with_a_center_and_one_without_is_not_a_positional_gap(
            self):
        """Mixed center declarations are the case the padding rule cannot help
        with: repeating the first wall's coordinates for the second would move a
        boundary the input never placed, so `none` says which object declares
        nothing."""
        block = {
            'walls': [{'id': 'a', 'geometry': 'sphere', 'radius': 1.5,
                       'center': [0, 0, 0], 'atoms': [1, 2, 3],
                       'force_constant': 0.01},
                      {'id': 'b', 'geometry': 'sphere', 'radius': 1.0,
                       'ref_atoms': [1, 2], 'atoms': [1, 2, 3],
                       'force_constant': 0.01}]}
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        written = dict(write_constraint_section(block))
        self.assertEqual(written['wall_center'], '0 0 0; none')
        self.assertEqual(written['wall_ref'], 'none; 1 2')
        plan = self._plan(read(**written), md_step=1000)
        first, second = plan.ids()
        self.assertIn('center', plan.spec(first).parameters)
        self.assertNotIn('center', plan.spec(second).parameters)
        self.assertEqual(plan.spec(second).parameters['anchor'], [0, 1])

    def test_molecules_of_unequal_size_have_no_flat_spelling(self):
        """The text form carries one size per segment, so a partition of a
        two-atom and a three-atom molecule cannot be written; the structured form
        still runs, and the refusal says which keyword is short."""
        block = {
            'walls': [{'id': 'mixed', 'geometry': 'sphere', 'radius': 1.5,
                       'atoms': [1, 2, 3], 'target': 'com',
                       'molecules': [[1, 2], [3]],
                       'force_constant': 0.01}]}
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        with self.assertRaises(ValueError) as caught:
            write_constraint_section(block)
        self.assertIn('wall_groups', str(caught.exception))
        plan = self._plan(block, md_step=1000)
        spec = plan.spec(plan.ids('wall')[0])
        self.assertEqual(spec.parameters['molecules'], [[0, 1], [2]])

    def test_a_component_ramp_writes_the_vector_it_composed(self):
        """The text form of a component ramp is the whole endpoint pair.

        `wall_x_end 2.0` on a 3-4-5 ellipsoid composes [3,4,5] to [2,4,5]; the
        writer prints those two vectors, which read back as the same schedule
        without needing to guess which components the input named.
        """
        block = read(wall_type='ellipsoid', wall_ellipsoid_semiaxes='3 4 5',
                     wall_center='0 0 0', wall_atoms='1-2', wall_fc='0.02',
                     wall_x_end='2.0', wall_x_stop='800')
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        written = dict(write_constraint_section(block))
        self.assertEqual(written['wall_ellipsoid_semiaxes_start'],
                         '3.0 4.0 5.0')
        self.assertEqual(written['wall_ellipsoid_semiaxes_end'],
                         '2.0 4.0 5.0')
        self.assertEqual(written['wall_ellipsoid_semiaxes_stop'], '800')
        first, second = self._both(block)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_a_parameter_none_of_the_objects_carries_is_not_written(self):
        """Defaults the run applies anyway stay out of the file."""
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        keys = [key for key, _value in write_constraint_section(
            {'walls': [{'geometry': 'sphere', 'radius': 1.5,
                        'atoms': [1], 'center': [0, 0, 0],
                        'force_constant': 0.01}]})]
        self.assertNotIn('wall_target', keys)
        self.assertNotIn('shake_tol', keys)

    def test_an_id_on_only_some_of_the_objects_cannot_be_written(self):
        """A positional line cannot leave a gap, and an id is positional."""
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        with self.assertRaises(ValueError) as caught:
            write_constraint_section(
                {'biases': [{'atoms': [1, 2], 'target': 1.0,
                             'force_constant': 0.01},
                            {'atoms': [1, 3], 'target': 1.0,
                             'force_constant': 0.01, 'id': 'named'}]})
        message = str(caught.exception)
        self.assertIn('bias_id', message)
        self.assertIn('name them all or none', message)

    def test_a_shared_schedule_list_is_refused_with_the_reason(self):
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        with self.assertRaises(ValueError) as caught:
            write_constraint_section(
                {'schedules': [{'id': 'ramp', 'parameter': 'force_constant',
                                'start_step': 0, 'end_step': 10,
                                'scale': [0, 1]}],
                 'biases': [{'cv': 'r1', 'target': 1.0,
                             'force_constant': 0.01, 'schedule': 'ramp'}]})
        self.assertIn("'schedules'", str(caught.exception))

    def test_the_writer_reads_the_structured_block_not_its_own_output(self):
        """The directions of the grammar are not interchangeable inputs.

        A flat keyword line is what the writer emits; feeding one back to it
        would be a `constraints` mapping with fields the schema has never had,
        so the refusal says which shape the writer reads instead of calling the
        keywords unknown and leaving the caller to guess.
        """
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        with self.assertRaises(ValueError) as caught:
            write_constraint_section({'bias_bond': '1 2',
                                      'bias_bond_ref': '0.96',
                                      'bias_bond_fc': '0.05'})
        message = str(caught.exception)
        self.assertIn('not a field of the constraint block', message)
        self.assertIn('biases', message)


    def test_object_lines_written_out_of_table_order_survive_the_round_trip(self):
        """The writer's line order is the record order, and the reader's anchor is
        the file order, so a family whose lines are written in an unusual order
        must still come back as the same run.

        The first pass of the split had the reader follow a fixed table order
        while the writer emitted records as they came, which put a `fix_dir` on
        the wrong selection without any number being wrong.
        """
        block = {'fixes': [{'element': ['H'], 'direction': 'x'},
                           {'atoms': [3], 'direction': 'z'}]}
        first, second = self._both(block)
        self.assertEqual(first.fingerprint, second.fingerprint)
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        written = dict(write_constraint_section(block))
        self.assertEqual(list(written)[:4],
                         ['fix_element', 'fix_atoms', 'fix_dir'])
        self.assertEqual(written['fix_dir'], 'x; z')

    def test_a_models_own_dump_writes_and_reads_back(self):
        """The path a `.config` JSON takes: every field present, defaults included.

        `model_dump` carries an empty inline ramp list on every action and an
        empty `notes` on every run because the model declares them, and an empty
        field is not a declaration; a writer that refused the shape its own model
        produced would turn a re-render into an input error.
        """
        from pyrai3md import schema
        structured = {
            'units': {'length': 'bohr', 'angle': 'radian', 'energy': 'ev'},
            'cvs': [{'name': 'r1', 'kind': 'bond', 'atoms': [1, 2]}],
            'biases': [{'id': 'stretch', 'cv': 'r1', 'target': 1.4,
                        'force_constant': 0.05, 'region': 'upper'}],
            'walls': [{'geometry': 'sphere', 'center': [0, 0, 0],
                       'radius': 2.8, 'atoms': [1, 2, 3],
                       'force_constant': 0.01, 'power': 6, 'target': 'com'}],
        }
        dumped = schema.SystemConstraintConfig(**structured).model_dump()
        self.assertIn('schedules', dumped['biases'][0])
        self.assertIn('notes', dumped)
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        written = dict(write_constraint_section(dumped))
        self.assertEqual(written['wall_target'], 'com')
        self.assertEqual(written['unit_length'], 'bohr')
        first, second = self._both(structured)
        self.assertEqual(first.fingerprint, second.fingerprint)


class TestRefusalsHaveOneHome(unittest.TestCase):
    """The dissolved module left one implementation of each rule, not two.

    The split was done by moving text between files, and the first pass of it
    renamed a function's *definition* as well as its calls, so `normalize.py`
    and `validate.py` each ended up with a copy - one of them dead, and the dead
    one referencing a helper that had stayed behind. A green suite cannot see
    that shape: the live copy answers and the tests pass. Identity is what a
    duplicated refusal cannot survive, so that is what is asserted.
    """

    def test_the_plan_build_calls_the_validation_layers_own_functions(self):
        import importlib
        normalize = importlib.import_module('pyrai3md.schema.normalize')
        validate = importlib.import_module('pyrai3md.schema.validate')
        for name in ('check_periodic', 'check_ramp_shape',
                     'check_ramp_positive'):
            self.assertIs(getattr(normalize, name), getattr(validate, name),
                          '%s has a second definition somewhere' % name)

    def test_a_direct_plan_build_still_refuses_two_pbc_flags(self):
        # `check_periodic`'s length check was a candidate for deletion: the
        # normalizer path hands it three flags. The public `build_plan`
        # argument does not, so the branch is reachable and the refusal is
        # what a caller with a malformed pbc actually gets.
        from pyrai3md.schema.validate import ConstraintInputError
        with self.assertRaises(ConstraintInputError) as caught:
            build_plan({'fixes': [{'atoms': '1-2', 'direction': 'xyz'}]},
                       atoms=ATOMS, natom=len(ATOMS),
                       masses=[15.999, 1.008, 1.008] * 2,
                       coordinates=COORDS, pbc=[0, 1])
        self.assertIn('pbc must hold three flags, got 2', str(caught.exception))

    def test_the_dissolved_modules_are_really_gone(self):
        import importlib
        for name in ('pyrai3md.schema.constraint',
                     'pyrai3md.schema.legacy_adapter',
                     'pyrai3md.interface.constraint_parser'):
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)


class TestShippedInputRoundTrip(unittest.TestCase):
    """A test config converted to `.input` reads back as the same run."""

    def test_convert_emits_a_section_the_pipeline_reads(self):
        """A test config's `constraints` block survives JSON -> .input -> Config.

        The qualification config carries the records the reader emits; the
        converter writes them back as the flat section, and the classic pipeline
        reads that section into the same records. A block the text could not
        state would raise inside `convert`, not quietly lose the constraint.
        """
        import json
        import os
        import shutil
        import tempfile
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.schema.adapter import build_config

        directory = tempfile.mkdtemp(prefix='constraint_convert_')
        self.addCleanup(shutil.rmtree, directory, True)
        source = os.path.join(directory, 'constraint_case.json')
        with open(source, 'w') as handle:
            json.dump({
                'title': 'constraint_case',
                'jobtype': 'sp',
                'theory': ['pyscf'],
                'atoms': ['O', 'H', 'H'],
                'coordinates': [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0],
                                [-0.24, 0.93, 0.0]],
                'constraints': {
                    'biases': [{'atoms': [1, 2], 'target': 0.96,
                                'force_constant': 0.05}],
                    'walls': [{'geometry': 'sphere', 'radius': 2.0,
                               'center': [0.0, 0.0, 0.0], 'atoms': [1, 2, 3],
                               'force_constant': 0.01}]},
            }, handle)
        from pyrai3md.qualification.certification.converter import convert
        text = convert(source)
        self.assertIn('&CONSTRAINT', text)
        section = _parse_text(text.split('\n'))['constraint']
        self.assertEqual(section['bias_bond'], '1 2')
        self.assertEqual(section['wall_sphere_radius'], '2.0')
        config = build_config(_parse_text(text.split('\n')))
        block = config.system.constraints
        self.assertEqual(len(block.biases), 1)
        self.assertEqual(block.biases[0].target, 0.96)
        self.assertEqual(block.walls[0].radius, 2.0)


if __name__ == '__main__':
    unittest.main()
