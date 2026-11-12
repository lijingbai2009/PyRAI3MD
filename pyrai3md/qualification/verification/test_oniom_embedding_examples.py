"""
======================================================================
Tests for the shipped single-center ONIOM true-embedding examples.

Ten cases: the base sp/md set with ORCA high and PySCF outer, the Molcas
energy pair plus the Molcas nac leg (ORCA refuses NACME under charges), the
meci/neb algorithm copies, and the two ONIOM3 periodic legs. They are read through the real input pipeline, so a
keyword the loader cannot spell, a plan the expander cannot build, or a
queue the calculator cannot embed fails here rather than on a host that
has the engines.

The A/B pairs are checked to share geometry exactly: an energy
difference is the field's, not a different structure's.

Nov 5 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import unittest

_EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'examples'))

SINGLE_2 = (
    'multiscale_oniom2_orca_embedding_sp',
    'multiscale_oniom2_orca_no_embedding_sp',
    'multiscale_oniom2_orca_embedding_md',
    'multiscale_oniom2_molcas_embedding_nac',
    'multiscale_oniom2_orca_embedding_meci',
    'multiscale_oniom2_orca_embedding_neb',
    'multiscale_oniom2_molcas_embedding_sp',
    'multiscale_oniom2_molcas_no_embedding_sp',
)
SINGLE_3 = (
    'multiscale_oniom3_orca_embedding_npt',
    'multiscale_oniom3_orca_embedding_cell',
)


def _load(name):
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    raw = parse_input(os.path.join(_EXAMPLES_DIR, name + '.yaml'))
    return raw, validate(normalize(build_config(raw)))


def _queue(config, raw):
    from pyrai3md.data.build import build_state
    from pyrai3md.kernel.environment import Environment
    Environment().initialize()
    return build_state(config, raw_config=raw)


class TestShippedOniomEmbeddingExamples(unittest.TestCase):

    def test_two_layer_cases_plan_three_terms(self):
        for name in SINGLE_2:
            raw, config = _load(name)
            self.assertEqual(config.control.multiscale_type, 'oniom')
            self.assertEqual(config.control.theory, ['orca', 'pyscf']
                             if 'orca' in name else ['molcas', 'pyscf'])
            self.assertEqual(len(config.system.highlevel), 3)
            state = _queue(config, raw)
            self.assertEqual(len(state.theory), 3)
            self.assertEqual(
                [task.get('contribution') for task in state.theory],
                [1, 1, -1])
            self.assertEqual(
                [task.get('center_index') for task in state.theory],
                [None, 0, 0])

    def test_three_layer_cases_plan_five_terms(self):
        for name in SINGLE_3:
            raw, config = _load(name)
            self.assertEqual(config.control.theory,
                             ['orca', 'pyscf', 'lammps'])
            self.assertEqual(len(config.system.highlevel), 3)
            self.assertEqual(len(config.system.midlevel), 3)
            self.assertEqual(list(config.system.pbc), [1, 1, 1])
            state = _queue(config, raw)
            self.assertEqual(len(state.theory), 5)
            self.assertEqual(
                [task.get('contribution') for task in state.theory],
                [1, 1, -1, 1, -1])

    def test_a_b_pairs_share_geometry_and_differ_only_in_the_switch(self):
        for embedded, control in (
                ('multiscale_oniom2_orca_embedding_sp',
                 'multiscale_oniom2_orca_no_embedding_sp'),
                ('multiscale_oniom2_molcas_embedding_sp',
                 'multiscale_oniom2_molcas_no_embedding_sp')):
            _raw_e, config_e = _load(embedded)
            _raw_c, config_c = _load(control)
            self.assertTrue(config_e.system.embedding)
            self.assertFalse(config_c.system.embedding)
            self.assertEqual(config_e.system.atoms, config_c.system.atoms)
            self.assertEqual(config_e.system.coordinates,
                             config_c.system.coordinates)
            self.assertEqual(config_e.system.highlevel,
                             config_c.system.highlevel)

    def test_every_case_ships_an_embedding_consumer(self):
        # ORCA and Molcas both consume point charges; openqp does not, which
        # is why these cases do not use it.
        for name in SINGLE_2:
            _raw, config = _load(name)
            self.assertIn(config.control.theory[0], ('orca', 'molcas'), name)

    def test_ai_b_control_pairs_ship_the_switch_off(self):
        for name in ('multiscale_oniom2_orca_no_embedding_sp',
                     'multiscale_oniom2_molcas_no_embedding_sp'):
            _raw, config = _load(name)
            self.assertFalse(config.system.embedding, name)

    def test_neb_endpoints_are_shipped_and_readable(self):
        from pyrai3md.workflow.path import fixture_path, read_xyz_frame
        _raw, config = _load('multiscale_oniom2_orca_embedding_neb')
        for name in ('path_oniom_cluster_reactant.xyz',
                     'path_oniom_cluster_product.xyz'):
            path = fixture_path(name)
            self.assertTrue(os.path.isfile(path), path)
            symbols, coordinates, _lattice = read_xyz_frame(
                path, atoms=config.system.atoms)
            self.assertEqual(len(symbols), 6)
            self.assertEqual(len(coordinates), 6)


if __name__ == '__main__':
    unittest.main()