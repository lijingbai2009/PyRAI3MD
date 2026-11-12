"""
======================================================================
Unit tests for the spin blocks the OpenQP input can and cannot ask for.

`state_spin` is a label written per state, so a 2 in it is a claim that
the row is a triplet. The input this skill writes carries one spin line
for the whole calculation, so the claim is either honoured by the engine
or it is a singlet wearing a triplet's name - and the second case keeps
the row count, which is the only thing a downstream search looks at. The
tests below pin both halves: the combination the input cannot serve is
refused by name, and the combinations it does serve are not refused.

These tests do NOT require an OpenQP installation.

Oct 14 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.skills.openqp_skill.schema.adapter import build_config
from pyrai3md.skills.openqp_skill.schema.normalize import normalize
from pyrai3md.skills.openqp_skill.schema.validate import validate

ATOMS = ['C', 'C', 'H', 'H', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.33], [0.0, 0.92, -0.57],
          [0.0, -0.92, -0.57], [0.92, 0.0, 1.9], [-0.92, 0.0, 1.9]]


def _config(**extra):
    """Run the real schema pipeline (build -> normalize -> validate)."""
    raw = {
        'atoms': list(ATOMS),
        'coordinates': [list(row) for row in COORDS],
        'charges': 0,
        'multiplicity': 1,
        'nstate': 2,
        'state_spin': [0, 0],
        'target_states': [0, 1],
        'profile': 'mrsf_gradient',
        'method': 'mrsf',
        'basis': 'sto-3g',
        'functional': 'bhhlyp',
        'execution_mode': 'local',
    }
    raw.update(extra)
    return validate(normalize(build_config(raw)))


class TestATripletRowOfASingletSystem(unittest.TestCase):
    """The list may name a triplet; the input cannot ask the engine for one."""

    def test_a_triplet_row_of_a_singlet_system_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            _config(state_spin=[0, 2])
        message = str(caught.exception)
        self.assertIn('state_spin', message)
        self.assertIn('triplet', message)
        self.assertIn('multiplicity', message)

    def test_the_same_claim_on_the_tddft_path_is_refused_too(self):
        # The refusal is about what the written input requests, and the tddft
        # path writes no multiplicity line at all - which is OpenQP's singlet
        # block, so the claim is unhonoured there exactly as it is for mrsf.
        with self.assertRaises(ValueError) as caught:
            _config(profile='tddft_gradient', method='tddft',
                    state_spin=[0, 2])
        self.assertIn('triplet', str(caught.exception))

    def test_the_singlet_list_still_passes(self):
        # Without this the refusal above could be a blanket ban on the keyword,
        # and the shipped cases would be the ones that noticed.
        config = _config(state_spin=[0, 0])
        self.assertEqual([int(x) for x in config.system.state_spin], [0, 0])

    def test_a_triplet_system_is_not_refused_by_this_guard(self):
        # A triplet reference is a different question: the multiplicity of the
        # calculation is 3, every row is of that spin, and the input asks for
        # exactly what the list says. The guard names the mixed case only.
        config = _config(multiplicity=3, state_spin=[2, 2])
        self.assertEqual([int(x) for x in config.system.state_spin], [2, 2])


if __name__ == '__main__':
    unittest.main()
