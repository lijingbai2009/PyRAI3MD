"""
======================================================================
This module locks the two shared hosts in the benchmark family.

`PRODUCTION_QDM_KEYWORDS` was restated in fssh, gsh and lzsh, and the
half-linear crossing law was written twice, byte for byte, in lz2 and
lz3. Both are now single definitions (`fssh.PRODUCTION_QDM_KEYWORDS`
and `base.DoubleSlopeLandauZener`).

The criterion is identity, not equality: a copy would still compare
equal today and drift tomorrow. The numeric check is the second half -
the moved formula still answers the same probability - and lz1 keeps
its own, different law.

Nov 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.qualification.benchmark import fssh, gsh, lzsh
from pyrai3md.skills.qdm_skill.service.models import Lz1, Lz2, Lz3


class TestTheFamilySharesOneDefinition(unittest.TestCase):
    """The table and the crossing law each have one home."""

    def test_the_override_table_is_one_object(self):
        self.assertIs(gsh.PRODUCTION_QDM_KEYWORDS,
                      fssh.PRODUCTION_QDM_KEYWORDS)
        self.assertIs(lzsh.PRODUCTION_QDM_KEYWORDS,
                      fssh.PRODUCTION_QDM_KEYWORDS)

    def test_the_half_linear_models_share_the_method(self):
        self.assertIs(Lz2.landau_zener_probability,
                      Lz3.landau_zener_probability)
        self.assertIsNot(Lz1.landau_zener_probability,
                         Lz2.landau_zener_probability)

    def test_the_shared_law_still_answers_the_same_numbers(self):
        params = {'lz_a': 0.02, 'lz_b': 0.005}
        v = 0.01
        expected = float(np.exp(-2.0 * np.pi * 0.005 ** 2 / (0.02 * v)))
        self.assertEqual(Lz2(params).landau_zener_probability(v), expected)
        self.assertEqual(Lz3(params).landau_zener_probability(v), expected)
        self.assertEqual(
            Lz1(params).landau_zener_probability(v),
            float(np.exp(-np.pi * 0.005 ** 2 / (0.02 * v))))


if __name__ == '__main__':
    unittest.main()