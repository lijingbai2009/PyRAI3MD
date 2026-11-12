"""
======================================================================
Unit tests for nuclear-mass derivation in schema normalize.

The integrators, thermostats/barostats and the kinetic-energy records
all read system.mass and silently skip when it is empty; an example
input that carries atoms but no mass used to produce a frozen MD
trajectory (remote run 2026-09-25: mass [], Ekin 0, coordinates
constant over 10 steps). compute_derived_quantities now fills the
masses from the element list (schema/reference.py ATOMIC_MASS) when
the input does not provide a complete list.

Sep 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.schema.normalize import compute_derived_quantities
from pyrai3md.schema.reference import ATOMIC_MASS


class TestMassDerivation(unittest.TestCase):
    """compute_derived_quantities fills system.mass from atoms."""

    def test_mass_derived_when_absent(self):
        data = {'atoms': ['C', 'C', 'H', 'H', 'H', 'H'],
                'coordinates': [[0.0, 0.0, 0.0]] * 6}
        out = compute_derived_quantities(dict(data))
        self.assertEqual(out['mass'],
                         [ATOMIC_MASS['C']] * 2 + [ATOMIC_MASS['H']] * 4)

    def test_user_mass_of_correct_length_preserved(self):
        user = [13.0, 13.0, 2.0, 2.0, 2.0, 2.0]
        data = {'atoms': ['C', 'C', 'H', 'H', 'H', 'H'], 'mass': list(user)}
        out = compute_derived_quantities(dict(data))
        self.assertEqual(out['mass'], user)

    def test_length_mismatch_rederives(self):
        data = {'atoms': ['O', 'H', 'H'], 'mass': [15.999]}
        out = compute_derived_quantities(dict(data))
        self.assertEqual(out['mass'],
                         [ATOMIC_MASS['O'], ATOMIC_MASS['H'],
                          ATOMIC_MASS['H']])

    def test_isotope_label_stripped(self):
        data = {'atoms': ['C_13', 'H_2']}
        out = compute_derived_quantities(dict(data))
        self.assertEqual(out['mass'],
                         [ATOMIC_MASS['C'], ATOMIC_MASS['H']])

    def test_lowercase_symbol_normalized(self):
        data = {'atoms': ['o', 'he']}
        out = compute_derived_quantities(dict(data))
        self.assertEqual(out['mass'],
                         [ATOMIC_MASS['O'], ATOMIC_MASS['He']])

    def test_unknown_element_leaves_mass_untouched(self):
        data = {'atoms': ['C', 'Xx'], 'mass': []}
        out = compute_derived_quantities(dict(data))
        self.assertEqual(out.get('mass'), [])


class TestMassDerivationEndToEnd(unittest.TestCase):
    """The official examples reach the state with populated masses."""

    def test_example_yaml_yields_nonempty_mass(self):
        import os
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize

        examples = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, 'examples'))
        path = os.path.join(examples, 'md_bagel_fssh.yaml')
        raw = parse_input(path)
        config = normalize(build_config(raw))
        self.assertEqual(len(config.system.mass), config.system.natom)
        self.assertTrue(all(m > 0 for m in config.system.mass))


if __name__ == '__main__':
    unittest.main()
