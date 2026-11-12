"""
======================================================================
Tests for the shipped multi-center ONIOM example inputs.

The six cases are the ONIOM counterparts of the multi-center QMMM
family: two ORCA (or Molcas, for the nac leg) high-level centers with a
quantum outer/mid (PySCF HF), and for the three-layer form a LAMMPS low
layer. They are read by the
real input pipeline - parse, normalize, validate, build_state - so a
keyword the loader cannot spell, a plan the expander cannot build, or a
queue the calculator cannot embed fails here rather than on a host that
has the engines.

The two-layer form is the new per-center outer-charge embedding: each
center's environment is every atom outside its own highlevel, filled by
the L(real) low-level row. The three-layer form embeds each center only
in its own midlevel. Nothing in this file starts an engine.

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

CASES_2 = (
    'multicenter_oniom2_orca_embedding_sp',
    'multicenter_oniom2_orca_no_embedding_sp',
    'multicenter_oniom2_orca_embedding_md',
    'multicenter_oniom2_molcas_embedding_nac_sp',
)
CASES_3 = (
    'multicenter_oniom3_orca_embedding_sp',
    'multicenter_oniom3_orca_embedding_npt',
)


def _load(name):
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    raw = parse_input(os.path.join(_EXAMPLES_DIR, name + '.yaml'))
    return raw, validate(normalize(build_config(raw)))


def _state(config, raw):
    from pyrai3md.data.build import build_state
    from pyrai3md.kernel.environment import Environment
    Environment().initialize()
    return build_state(config, raw_config=raw)


class TestShippedMulticenterOniomInput(unittest.TestCase):
    """Each shipped example validates and plans the queue it claims."""

    def test_two_layer_cases_plan_one_real_and_two_centers(self):
        for name in CASES_2:
            raw, config = _load(name)
            self.assertEqual(config.control.multiscale_type, 'oniom')
            self.assertIn(config.control.theory[0], ('orca', 'molcas'), name)
            self.assertEqual(config.control.theory[1], 'pyscf')
            self.assertEqual(len(config.system.highlevel), 2)
            self.assertFalse(config.system.read_charge)
            state = _state(config, raw)
            # real-first: L(real), then +H(c), -L(c) per center.
            self.assertEqual(len(state.theory), 5)
            self.assertEqual(
                [task.get('contribution') for task in state.theory],
                [1, 1, -1, 1, -1])
            self.assertEqual(
                [task.get('center_index') for task in state.theory],
                [None, 0, 0, 1, 1])
            self.assertEqual(
                [bool(task.get('provides_electronic'))
                 for task in state.theory],
                [False, True, False, True, False])

    def test_two_layer_control_turns_the_field_off(self):
        _raw, config = _load('multicenter_oniom2_orca_no_embedding_sp')
        self.assertFalse(config.system.embedding)
        _raw, embedded = _load('multicenter_oniom2_orca_embedding_sp')
        self.assertTrue(embedded.system.embedding)

    def test_three_layer_cases_plan_one_real_and_four_duties_per_center(self):
        for name in CASES_3:
            raw, config = _load(name)
            self.assertEqual(config.control.theory, ['orca', 'pyscf', 'lammps'])
            self.assertEqual(len(config.system.highlevel), 2)
            self.assertEqual(len(config.system.midlevel), 2)
            self.assertTrue(config.system.embedding)
            self.assertFalse(config.system.read_charge)
            state = _state(config, raw)
            # real-first: L(real), then M(I), -L(I), +H(M), -M(M).
            self.assertEqual(len(state.theory), 9)
            self.assertEqual(
                [task.get('contribution') for task in state.theory],
                [1, 1, -1, 1, -1, 1, -1, 1, -1])
            self.assertEqual(
                [task.get('center_index') for task in state.theory],
                [None, 0, 0, 0, 0, 1, 1, 1, 1])
            self.assertEqual(
                sum(1 for task in state.theory
                    if task.get('provides_electronic')), 2)

    def test_periodic_legs_declare_their_box(self):
        for name in ('multicenter_oniom3_orca_embedding_npt',
                     'multicenter_oniom3_orca_embedding_sp'):
            _raw, config = _load(name)
            self.assertEqual(list(config.system.pbc), [1, 1, 1])
            self.assertEqual(len(config.system.cell), 6)


class TestOniomEmbeddingRegions(unittest.TestCase):
    """The embedding source each center is handed."""

    def _calculator(self, name):
        from pyrai3md.workflow.calculator import Calculator
        raw, config = _load(name)
        return config, Calculator(state=_state(config, raw))

    def test_two_layer_environment_is_the_other_center(self):
        _config, calc = self._calculator('multicenter_oniom2_orca_embedding_sp')
        regions = calc._embedding_regions_by_center()
        self.assertEqual(regions, {0: [3, 4, 5], 1: [0, 1, 2]})

    def test_three_layer_environment_is_the_own_midlevel(self):
        _config, calc = self._calculator('multicenter_oniom3_orca_embedding_sp')
        regions = calc._embedding_regions_by_center()
        self.assertEqual(regions, {0: [6, 7, 8], 1: [9, 10, 11]})


CHARGED_2 = (
    'multicenter_oniom2_orca_charged_sp',
    'multicenter_oniom2_orca_charged_no_embedding_sp',
    'multicenter_oniom2_orca_charged_md',
)
CHARGED_3 = (
    'multicenter_oniom3_orca_charged_sp',
    'multicenter_oniom3_orca_charged_no_embedding_sp',
    'multicenter_oniom3_orca_charged_md',
)


class TestChargedMulticenterOniom(unittest.TestCase):
    """An ammonium (+1) centre beside a neutral water centre."""

    def test_charges_are_per_center_and_the_real_layer_takes_the_sum(self):
        for name in CHARGED_2 + CHARGED_3:
            raw, config = _load(name)
            state = _state(config, raw)
            plan = state.partition_plan
            self.assertEqual([plan.center(c).charge for c in range(2)],
                             [1, 0], name)
            self.assertEqual(plan.system_charge, 1, name)
            self.assertEqual([plan.center(c).multiplicity for c in range(2)],
                             [1, 1], name)

    def test_two_layer_tasks_carry_the_center_charge(self):
        for name in CHARGED_2:
            raw, config = _load(name)
            state = _state(config, raw)
            # real +1, then +H(c0)/-L(c0) at +1, +H(c1)/-L(c1) at 0.
            self.assertEqual(
                [task.get('region_charge') for task in state.theory],
                [1, 1, 1, 0, 0], name)

    def test_three_layer_tasks_carry_the_center_charge(self):
        for name in CHARGED_3:
            raw, config = _load(name)
            self.assertEqual(config.control.theory, ['orca', 'pyscf', 'orca'],
                             name)
            state = _state(config, raw)
            # real +1, then the four c0 duties at +1 and c1's four at 0.
            self.assertEqual(
                [task.get('region_charge') for task in state.theory],
                [1, 1, 1, 1, 1, 0, 0, 0, 0], name)
            self.assertEqual(len(state.theory), 9, name)

    def test_charged_embedding_regions(self):
        from pyrai3md.workflow.calculator import Calculator
        raw, config = _load('multicenter_oniom2_orca_charged_sp')
        calc = Calculator(state=_state(config, raw))
        self.assertEqual(calc._embedding_regions_by_center(),
                         {0: [5, 6, 7], 1: [0, 1, 2, 3, 4]})
        raw, config = _load('multicenter_oniom3_orca_charged_sp')
        calc = Calculator(state=_state(config, raw))
        self.assertEqual(calc._embedding_regions_by_center(),
                         {0: [5, 6, 7], 1: [11, 12, 13]})

    def test_control_pairs_share_geometry_and_charges(self):
        for embedded, control in (
                ('multicenter_oniom2_orca_charged_sp',
                 'multicenter_oniom2_orca_charged_no_embedding_sp'),
                ('multicenter_oniom3_orca_charged_sp',
                 'multicenter_oniom3_orca_charged_no_embedding_sp')):
            _raw_e, config_e = _load(embedded)
            _raw_c, config_c = _load(control)
            self.assertTrue(config_e.system.embedding, embedded)
            self.assertFalse(config_c.system.embedding, control)
            self.assertEqual(config_e.system.charges,
                             config_c.system.charges)
            self.assertEqual(config_e.system.atoms, config_c.system.atoms)
            self.assertEqual(config_e.system.coordinates,
                             config_c.system.coordinates)
            self.assertEqual(config_e.system.highlevel,
                             config_c.system.highlevel)
            self.assertEqual(config_e.system.midlevel,
                             config_c.system.midlevel)


if __name__ == '__main__':
    unittest.main()