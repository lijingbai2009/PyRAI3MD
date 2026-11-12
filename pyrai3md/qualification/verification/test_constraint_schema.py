"""
======================================================================
Tests for constraint input normalization.

Three things are pinned here. Every number in the plan is in internal units,
converted by the one rule q_internal = c * q_input and k_internal = k_input /
c ** power. Indices shift from 1-based to 0-based exactly once. And a
declaration the build cannot honour fails instead of being parsed and ignored,
which is the failure mode the design document calls out as the expensive one.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md import schema
from pyrai3md.data import constraint as records
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR, KCAL_TO_HARTREE

ATOMS = ['O', 'H', 'H', 'C']
MASSES = np.array([16.0, 1.0, 1.0, 12.0])


def _normalize(mapping, **kwargs):
    options = {'atoms': ATOMS, 'natom': 4, 'masses': MASSES}
    options.update(kwargs)
    return schema.build_plan(mapping, **options)


class TestEmptiness(unittest.TestCase):

    def test_defaults_produce_the_empty_plan(self):
        self.assertTrue(_normalize({}).is_empty)

    def test_a_defaults_only_config_enables_nothing(self):
        config = schema.SystemConstraintConfig()
        self.assertTrue(config.is_empty())
        self.assertTrue(schema.normalize_config(config, atoms=ATOMS,
                                                natom=4).is_empty)

    def test_the_system_model_carries_one_constraint_field(self):
        """DC-4: no default value can enable a feature.

        The block used to register a `{'constraints': {}}` default of its own
        while it lived apart from the framework. Mounted on `System` the
        statement is the field's own default, which is None: a run that writes
        no constraint section declares nothing, and there is no second place a
        default could turn a feature on.
        """
        self.assertIn('constraints', schema.System.model_fields)
        self.assertIsNone(schema.System().constraints)

    def test_both_section_names_route_to_that_one_field(self):
        """The nested-section table is the section's only way into `System`.

        `constraint` is what a new input writes and `system_constraint` is the
        retired spelling of the same block; both must land on
        `system.constraints`, because an archived input that routes nowhere
        simply stops constraining.
        """
        from pyrai3md.schema.config import NESTED_SECTION_FIELDS
        self.assertEqual(NESTED_SECTION_FIELDS['constraint'],
                         ('system', 'constraints'))
        self.assertEqual(NESTED_SECTION_FIELDS['system_constraint'],
                         ('system', 'constraints'))


class TestUnitConversion(unittest.TestCase):

    def test_angstrom_target_becomes_bohr(self):
        plan = _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.5,
                                       'force_constant': 0.02}]})
        spec = plan.spec(plan.ids('bias')[0])
        self.assertAlmostEqual(spec.target, 1.5 * ANGSTROM_TO_BOHR, places=12)

    def test_force_constant_divides_by_the_square(self):
        plan = _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.5,
                                       'force_constant': 0.02}]})
        spec = plan.spec(plan.ids('bias')[0])
        expected = 0.02 / ANGSTROM_TO_BOHR ** 2
        self.assertAlmostEqual(spec.parameters['force_constant'], expected,
                               places=9)

    def test_bohr_input_is_not_converted_again(self):
        first = _normalize({'units': {'length': 'bohr'},
                            'cvs': [{'name': 'r', 'kind': 'bond',
                                     'atoms': [1, 2]}],
                            'biases': [{'cv': 'r', 'target': 1.0,
                                        'force_constant': 0.02}]})
        spec = first.spec(first.ids('bias')[0])
        self.assertAlmostEqual(spec.target, 1.0, places=12)
        self.assertAlmostEqual(spec.parameters['force_constant'], 0.02,
                               places=12)

    def test_degree_target_becomes_radian(self):
        plan = _normalize({'cvs': [{'name': 'a', 'kind': 'angle',
                                    'atoms': [1, 2, 3]}],
                           'biases': [{'cv': 'a', 'target': 180.0,
                                       'force_constant': 0.005}]})
        spec = plan.spec(plan.ids('bias')[0])
        self.assertAlmostEqual(spec.target, np.pi, places=12)
        self.assertAlmostEqual(spec.parameters['force_constant'],
                               0.005 / (np.pi / 180.0) ** 2, places=6)

    def test_kcal_force_constant_converts_energy_too(self):
        plan = _normalize({'units': {'length': 'bohr', 'energy': 'kcal'},
                           'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 1.0}]})
        spec = plan.spec(plan.ids('bias')[0])
        self.assertAlmostEqual(spec.parameters['force_constant'],
                               KCAL_TO_HARTREE, places=12)

    def test_wall_constant_divides_by_the_power(self):
        plan = _normalize({'walls': [{'geometry': 'sphere', 'radius': 10.0,
                                      'atoms': [1, 2], 'power': 4,
                                      'force_constant': 0.001}]})
        spec = plan.spec(plan.ids('wall')[0])
        self.assertAlmostEqual(spec.parameters['force_constant'],
                               0.001 / ANGSTROM_TO_BOHR ** 4, places=3)

    def test_metadynamics_width_follows_its_cv_unit(self):
        plan = _normalize({
            'cvs': [{'name': 'd', 'kind': 'dihedral', 'atoms': [1, 2, 3, 4]}],
            'metadynamics': [{'cvs': ['d'], 'height': 1.0e-4,
                              'width': [0.1], 'pace': 10}]})
        spec = plan.spec(plan.ids('metadynamics')[0])
        self.assertAlmostEqual(spec.parameters['widths'][0],
                               0.1 * np.pi / 180.0, places=12)
        self.assertAlmostEqual(spec.parameters['height'], 1.0e-4, places=12)

    def test_units_of_the_plan_are_recorded(self):
        plan = _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 0.01}]},
                          )
        self.assertEqual(plan.units['length'], 'angstrom')
        self.assertAlmostEqual(plan.units['length_to_bohr'], ANGSTROM_TO_BOHR)
        self.assertTrue(plan.units_converted)


class TestIndexConversion(unittest.TestCase):

    def test_one_based_input_becomes_zero_based_once(self):
        plan = _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [3, 4]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 0.01}]})
        self.assertEqual(plan.cvs['r'].indices.tolist(), [2, 3])

    def test_restoring_a_plan_does_not_shift_again(self):
        plan = _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [3, 4]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 0.01}]})
        again = records.ConstraintPlan.from_dict(plan.to_dict())
        self.assertEqual(again.cvs['r'].indices.tolist(), [2, 3])

    def test_range_spelling_matches_the_explicit_list(self):
        first = _normalize({'groups': [{'name': 'g', 'atoms': ['2-4']}],
                            'fixes': [{'group': 'g', 'direction': 'x'}]})
        second = _normalize({'groups': [{'name': 'g', 'atoms': [2, 3, 4]}],
                             'fixes': [{'group': 'g', 'direction': 'x'}]})
        self.assertEqual(first.fingerprint, second.fingerprint)


class TestIdsAndOrder(unittest.TestCase):

    def test_declaration_order_is_kept(self):
        plan = _normalize({
            'walls': [{'geometry': 'sphere', 'radius': 5.0, 'atoms': [1],
                       'force_constant': 0.001},
                      {'geometry': 'sphere', 'radius': 4.0, 'atoms': [2],
                       'force_constant': 0.001}],
            'biases': [{'atoms': [1, 2], 'target': 1.0,
                        'force_constant': 0.01}]})
        self.assertEqual(plan.ids('wall'), ['wall1', 'wall2'])
        self.assertEqual(plan.ids('bias'), ['bias1'])

    def test_declared_ids_are_kept(self):
        plan = _normalize({'walls': [{'id': 'vessel', 'geometry': 'sphere',
                                      'radius': 5.0, 'atoms': [1],
                                      'force_constant': 0.001}]})
        self.assertEqual(plan.ids('wall'), ['vessel'])

    def test_duplicate_id_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'biases': [{'id': 'x', 'atoms': [1, 2], 'target': 1.0,
                                    'force_constant': 0.01},
                                   {'id': 'x', 'atoms': [2, 3], 'target': 1.0,
                                    'force_constant': 0.01}]})

    def test_inline_cv_gets_a_derived_name_and_stays_unique(self):
        plan = _normalize({'biases': [{'atoms': [1, 2], 'target': 1.0,
                                       'force_constant': 0.01},
                                      {'atoms': [2, 3], 'target': 1.2,
                                       'force_constant': 0.01}]})
        names = [plan.spec(name).cv for name in plan.ids('bias')]
        self.assertEqual(len(set(names)), 2)


class TestRefusals(unittest.TestCase):
    """Unsupported must mean unsupported, out loud."""

    def test_bond_type_without_a_structure_names_what_is_missing(self):
        with self.assertRaises(schema.UnsupportedCapabilityError) as caught:
            _normalize({'constraints': [{'bond_type': 'O-H'}]})
        message = str(caught.exception)
        self.assertIn('without one', message)
        self.assertIn('coordinates', message)

    def test_bond_type_expands_to_the_perceived_pairs(self):
        """The graph is perceived from the structure, then the type selects."""
        water = [[0.00, 0.00, 0.00],
                 [0.96, 0.00, 0.00],
                 [-0.24, 0.93, 0.00],
                 [6.00, 6.00, 6.00]]
        plan = _normalize({'constraints': [{'id': 'oh', 'bond_type': 'O-H',
                                            'tolerance': 1.0e-10}]},
                         coordinates=water, lattice=None)
        spec = [item for item in plan.actions
                if item.kind == 'constraint'][0]
        self.assertEqual(spec.pairs.tolist(), [[0, 1], [0, 2]])

    def test_bond_type_is_refused_for_an_angle_or_torsion(self):
        water = [[0.00, 0.00, 0.00],
                 [0.96, 0.00, 0.00],
                 [-0.24, 0.93, 0.00],
                 [6.00, 6.00, 6.00]]
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _normalize({'constraints': [{'kind': 'angle',
                                         'bond_type': 'O-H'}]},
                       coordinates=water, lattice=None)
        self.assertIn('3-atom rows', str(caught.exception))

    def test_an_unlisted_element_pair_is_refused_not_guessed(self):
        """The topology service does not fall back to the default radius."""
        pair = [[0.00, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [5.00, 5.00, 5.00],
                [6.00, 5.00, 5.00]]
        with self.assertRaises(schema.UnsupportedCapabilityError) as caught:
            _normalize({'constraints': [{'bond_type': 'O-C'}]},
                       atoms=['O', 'Na', 'H', 'H'], natom=4,
                       masses=np.array([16.0, 23.0, 1.0, 1.0]),
                       coordinates=pair, lattice=None)
        message = str(caught.exception)
        self.assertIn('not in the bond-radius table', message)
        self.assertIn('schema/reference.py', message)

    def test_only_the_families_with_a_separate_object_line_select_by_element(
            self):
        """An element list needs something to align against.

        `wall_element` and `fix_element` are parameter lines: the wall's shape
        and the fix's direction say how many objects there are, so a selection of
        any length can be one per object. A bias, a cv and a history potential
        take their atoms from their own object line, whose segments are that
        count, so an element list of unknown length has nothing to line up with.
        The classic grammar never had `bias_element`; these fields came from the
        shadow build's records and are not a capability the text can state.
        """
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'biases': [{'element': ['O'], 'target': 1.0,
                                    'force_constant': 0.01}]})
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'metadynamics': [{'group': 'g', 'cvs': ['r'],
                                          'height': 1.0e-4, 'width': [0.1]}]})
        plan = _normalize({'walls': [{'geometry': 'sphere', 'radius': 2.0,
                                      'element': ['O'], 'center': [0, 0, 0],
                                      'force_constant': 0.001}]})
        self.assertEqual([spec.kind for spec in plan.actions], ['wall'])

    def test_a_group_has_no_molecule_partition(self):
        """The molecule dimension is the wall's, and only as `wall_groups`.

        A group is a set of atoms; partitioning it into molecules was a field
        that could only ever raise, so it is not a field at all now.
        """
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _normalize({'groups': [{'name': 'water', 'molecule': 'h2o'}]})
        self.assertIn('molecule', str(caught.exception))

    def test_a_periodic_axis_does_not_refuse_the_variable_block(self):
        """Whether a cv has a periodic image is a fact about the bond graph.

        The schema layer has no graph, so it no longer refuses the block; the
        binding-time guard asks per variable. What the input layer still refuses
        outright is the law no geometry can rescue (`wall_law soft`), which is
        the periodic test file's case.
        """
        plan = _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                    'atoms': [1, 2]}],
                           'biases': [{'cv': 'r', 'target': 1.0,
                                       'force_constant': 0.01}]},
                          pbc=[0, 1, 0])
        self.assertEqual([spec.kind for spec in plan.actions], ['bias'])

    def test_an_angle_constraint_reaches_the_plan(self):
        """A4: an angle is no longer a refusal, it is a condition row."""
        plan = _normalize({'constraints': [{'kind': 'angle',
                                            'bonds': [[1, 2, 3]]}]})
        spec = [item for item in plan.actions if item.kind == 'constraint'][0]
        self.assertEqual(spec.family, 'angle')
        self.assertIsNone(spec.pairs)
        self.assertEqual(spec.parameters['cv_kind'], 'angle')
        self.assertEqual(spec.parameters['rows'], [[0, 1, 2]])

    def test_a_dihedral_constraint_reaches_the_plan(self):
        """Four atoms, ordered: swapping the ends negates the variable."""
        plan = _normalize({'constraints': [{'kind': 'dihedral',
                                            'bonds': [[1, 2, 3, 4],
                                                      [4, 3, 2, 1]]}]})
        spec = [item for item in plan.actions if item.kind == 'constraint'][0]
        self.assertEqual(spec.family, 'dihedral')
        self.assertEqual(spec.parameters['cv_kind'], 'dihedral')
        self.assertEqual(spec.parameters['rows'],
                         [[0, 1, 2, 3], [3, 2, 1, 0]])
        self.assertIsNone(spec.pairs)

    def test_an_angle_constraint_with_a_pair_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _normalize({'constraints': [{'kind': 'angle', 'bonds': [[1, 2]]}]})
        self.assertIn('needs 3 atom indices', str(caught.exception))

    def test_an_angle_constraint_repeats_an_atom_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _normalize({'constraints': [{'kind': 'angle',
                                         'bonds': [[1, 2, 1]]}]})
        self.assertIn('repeats an atom', str(caught.exception))

    def test_an_rmsd_history_that_also_names_a_cv_is_refused(self):
        """The variable of an aligned RMSD is the deviation, not a cv.

        This refusal used to be broader: 'rmsd' was a later milestone and the
        whole declaration was turned down. What remains is the narrower one for
        naming a cv as well, which would be two variables for one hill list.
        """
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                 'atoms': [1, 2]}],
                        'metadynamics': [{'kind': 'rmsd', 'cvs': ['r'],
                                          'atoms': [1, 2, 3],
                                          'height': 1.0e-4,
                                          'width': [0.1]}]})

    def test_an_rmsd_history_normalizes_with_the_group_resolved(self):
        plan = _normalize({'metadynamics': [{'kind': 'rmsd',
                                             'atoms': [1, 2, 3],
                                             'height': 1.0e-4,
                                             'width': [0.1], 'pace': 5}]})
        spec = plan.actions[0]
        self.assertEqual(spec.family, 'rmsd')
        self.assertEqual(list(spec.atoms), [0, 1, 2])

    def test_one_sided_dihedral_region_needs_an_origin(self):
        with self.assertRaises(schema.UnsupportedCapabilityError):
            _normalize({'cvs': [{'name': 'd', 'kind': 'dihedral',
                                 'atoms': [1, 2, 3, 4]}],
                        'biases': [{'cv': 'd', 'target': 1.0,
                                    'force_constant': 0.01,
                                    'region': 'upper'}]})

    def test_interval_region_is_not_defined_yet(self):
        with self.assertRaises(schema.UnsupportedCapabilityError):
            _normalize({'cvs': [{'name': 'r', 'kind': 'bond',
                                 'atoms': [1, 2]}],
                        'biases': [{'cv': 'r', 'target': 1.0,
                                    'force_constant': 0.01,
                                    'region': 'inside'}]})

    def test_a_six_parameter_cell_agrees_with_the_matrix_spelling(self):
        """Both cell shapes land on one lattice, through one converter."""
        from pyrai3md.utilities.lattice import cell6_to_matrix
        from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

        six = _normalize({'walls': [{'geometry': 'triclinic',
                                     'cell': [20, 20, 20, 80, 70, 120],
                                     'target': 'com', 'group': 'g',
                                     'power': 2, 'force_constant': 0.001,
                                     'radius': 1.0}],
                          'groups': [{'name': 'g', 'atoms': [1, 2]}]})
        rows = [item for item in six.actions if item.kind == 'wall'][0]
        np.testing.assert_allclose(
            np.asarray(rows.parameters['cell'], dtype=np.float64),
            cell6_to_matrix([20, 20, 20, 80, 70, 120]) * ANGSTROM_TO_BOHR,
            rtol=0.0, atol=1.0e-12)
        nine = _normalize({'walls': [{'geometry': 'triclinic',
                                      'cell': list(
                                          np.asarray(
                                              cell6_to_matrix(
                                                  [20, 20, 20, 80, 70, 120]),
                                              dtype=float).reshape(-1)),
                                      'target': 'com', 'group': 'g',
                                      'power': 2, 'force_constant': 0.001,
                                      'radius': 1.0}],
                            'groups': [{'name': 'g', 'atoms': [1, 2]}]})
        rows_nine = [item for item in nine.actions if item.kind == 'wall'][0]
        np.testing.assert_allclose(np.asarray(rows.parameters['cell']),
                                   np.asarray(rows_nine.parameters['cell']),
                                   atol=1.0e-14)

    def test_a_six_parameter_cell_with_a_bad_length_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError) as caught:
            _normalize({'walls': [{'geometry': 'triclinic',
                                   'cell': [0, 20, 20, 90, 90, 90],
                                   'target': 'com', 'group': 'g',
                                   'power': 2, 'force_constant': 0.001,
                                   'radius': 1.0}],
                        'groups': [{'name': 'g', 'atoms': [1, 2]}]})
        self.assertIn('lengths must be positive', str(caught.exception))

    def test_unknown_field_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'biases': [{'atoms': [1, 2], 'target': 1.0,
                                    'force_constant': 0.01,
                                    'force_costant': 0.02}]})

    def test_unknown_unit_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'units': {'length': 'parsec'}})

    def test_com_wall_without_masses_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            schema.build_plan(
                {'walls': [{'geometry': 'sphere', 'radius': 1.0,
                            'target': 'com', 'group': 'g', 'power': 2,
                            'force_constant': 0.001}],
                 'groups': [{'name': 'g', 'atoms': [1, 2]}]},
                atoms=ATOMS, natom=4)

    def test_conflicting_targets_on_one_pair_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'constraints': [{'id': 'a', 'bonds': [[1, 2]],
                                         'target': 1.0},
                                        {'id': 'b', 'bonds': [[1, 2]],
                                         'target': 1.1}]})

    def test_a_constraint_may_declare_several_bonds(self):
        """Two pairs in one block, which a SHAKE run is made of.

        The selector base rewrites a lone inline bond into an atom pair for a
        bias or a cv, and inheriting that rewriting here refused a two-bond
        declaration as "takes one inline bond". A water has two.
        """
        plan = _normalize({'constraints': [{'id': 'oh',
                                            'bonds': [[1, 2], [1, 3]]}]})
        spec = plan.spec('oh')
        self.assertEqual(spec.pairs.tolist(), [[0, 1], [0, 2]])
        self.assertEqual(spec.reference, 'initial')

    def test_wall_power_below_two_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'walls': [{'geometry': 'sphere', 'radius': 1.0,
                                   'atoms': [1], 'power': 1,
                                   'force_constant': 0.001}]})

    def test_declaring_only_a_group_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'groups': [{'name': 'g', 'atoms': [1, 2]}]})

    def test_schedule_on_a_missing_parameter_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'schedules': [{'id': 's', 'parameter': 'height',
                                       'start_step': 0, 'end_step': 5,
                                       'value': [0.0, 1.0]}],
                        'cvs': [{'name': 'r', 'kind': 'bond',
                                 'atoms': [1, 2]}],
                        'biases': [{'cv': 'r', 'target': 1.0,
                                    'force_constant': 0.01,
                                    'schedule': 's'}]})


class TestGeometryInputs(unittest.TestCase):

    def test_box_and_triclinic_hold_their_numbers(self):
        plan = _normalize({'walls': [
            {'geometry': 'box', 'lengths': [2, 2, 2], 'atoms': [1],
             'power': 2, 'force_constant': 0.001},
            {'geometry': 'triclinic', 'cell': [[2, 0, 0], [0, 2, 0],
                                               [0, 0, 2]], 'atoms': [2],
             'power': 2, 'force_constant': 0.001}]})
        box = plan.spec('wall1').parameters
        tri = plan.spec('wall2').parameters
        np.testing.assert_allclose(box['lengths'],
                                   [2 * ANGSTROM_TO_BOHR] * 3, atol=1.0e-12)
        np.testing.assert_allclose(tri['cell'],
                                   np.diag([2, 2, 2]) * ANGSTROM_TO_BOHR,
                                   atol=1.0e-12)

    def test_ellipsoid_needs_three_positive_semiaxes(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'walls': [{'geometry': 'ellipsoid',
                                   'semiaxes': [1.0, 2.0], 'atoms': [1],
                                   'power': 2, 'force_constant': 0.001}]})

    def test_singular_cell_is_refused(self):
        with self.assertRaises(schema.ConstraintInputError):
            _normalize({'walls': [{'geometry': 'triclinic',
                                   'cell': [[1, 0, 0], [2, 0, 0], [0, 0, 1]],
                                   'atoms': [1], 'power': 2,
                                   'force_constant': 0.001}]})

    def test_explicit_wall_center_is_converted(self):
        plan = _normalize({'walls': [{'geometry': 'sphere',
                                      'center': [1.0, 0.0, 0.0],
                                      'radius': 2.0, 'atoms': [1], 'power': 2,
                                      'force_constant': 0.001}]})
        np.testing.assert_allclose(plan.spec('wall1').parameters['center'],
                                   [ANGSTROM_TO_BOHR, 0.0, 0.0], atol=1.0e-12)

    def test_direction_spelling_becomes_an_axis_list(self):
        plan = _normalize({'fixes': [{'atoms': [1, 2], 'direction': 'xz'}]})
        self.assertEqual(plan.spec('fix1').parameters['axes'], [0, 2])

    def test_overlapping_fixes_merge_by_component(self):
        plan = _normalize({'fixes': [{'atoms': [1], 'direction': 'x'},
                                     {'atoms': [1], 'direction': 'y'}]})
        self.assertEqual(len(plan.ids('fix')), 2)


class TestStructuredEquivalence(unittest.TestCase):

    def test_the_same_plan_from_a_mapping_or_a_config_object(self):
        mapping = {'cvs': [{'name': 'r', 'kind': 'bond', 'atoms': [1, 2]}],
                   'biases': [{'cv': 'r', 'target': 1.5,
                               'force_constant': 0.02, 'region': 'upper'}]}
        from_mapping = _normalize(mapping)
        config = schema.config_from_mapping(mapping)
        from_config = schema.normalize_config(config, atoms=ATOMS, natom=4,
                                              masses=MASSES)
        self.assertEqual(from_mapping.fingerprint, from_config.fingerprint)

    def test_a_scalar_block_is_read_as_a_one_item_list(self):
        plan = _normalize({'biases': {'atoms': [1, 2], 'target': 1.0,
                                      'force_constant': 0.01}})
        self.assertEqual(len(plan.ids('bias')), 1)


if __name__ == '__main__':
    unittest.main()
