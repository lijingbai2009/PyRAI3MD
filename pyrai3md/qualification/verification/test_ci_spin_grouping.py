"""
======================================================================
This module locks the ci/spin grouping shared by the harness export and
the molcas skill.

Three converter sites derived ci/spin from state_spin with their own
copy of one grouping; they now call `_group_state_spin`. The grouping
is the skill's: one block per multiplicity, blocks in ascending 2S
order, which is what `molcas_skill/schema/normalize.py::verify_state_spin`
produces for a standalone input (it sorts the ladder and refuses one
that does not start at the ground multiplicity).

The premise that first-appearance grouping and the skill's run
grouping differ was measured false on every ladder the skill accepts:
`[0, 2, 0]` groups as `2 1` / `0 1` on both sides, and a ladder whose
first value is not the ground multiplicity (`[2, 0]`) is refused by
the skill outright. The two groupings do differ there (`1 0` before,
`0 1` now), so the export follows the skill's canonical order; that
no shipped example carries state_spin at all makes the change a no-op
on the shipped corpus.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.qualification.certification.converter import (
    _derive_agent_ci_spin, _render_molecule_section)
from pyrai3md.qualification.verification._source_paths import source_read


def _harness(state_spin):
    """(ci, spin) the agent-mode export derives for one ladder."""
    molecule = {'state_spin': list(state_spin)}
    _derive_agent_ci_spin(molecule)
    return molecule['ci'], molecule['spin']


def _skill(state_spin):
    """(ci, spin) the molcas skill's own normalize derives."""
    from pyrai3md.skills.molcas_skill.schema.adapter import build_config
    from pyrai3md.skills.molcas_skill.schema.normalize import normalize
    raw = {'atoms': ['C', 'H', 'H'],
           'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1],
                           [0.0, 0.0, -1.1]],
           'charges': 0, 'multiplicity': 1, 'nstate': len(state_spin),
           'state_spin': list(state_spin), 'profile': 'casscf_energy',
           'method': 'casscf', 'basis': 'ano-rcc-vdzp', 'nact_elec': 2,
           'nact_orb': 2, 'molcas_path': '/tmp'}
    out = normalize(build_config(raw))
    return list(out.ci_roots), [two_s // 2 for two_s in out.ci_spin]


class TestOneGroupingOnBothSides(unittest.TestCase):
    """The export and the skill read the ladder the same way."""

    ACCEPTED_LADDERS = ([0], [0, 0], [0, 2, 0], [0, 0, 2, 2, 2])

    def test_the_harness_and_the_skill_agree(self):
        for ladder in self.ACCEPTED_LADDERS:
            with self.subTest(ladder=ladder):
                self.assertEqual(_harness(ladder), _skill(ladder))

    def test_the_ungrounded_ladder_is_exported_in_skill_order(self):
        # [2, 0] is not a run the skill accepts (its first state must
        # carry the ground multiplicity), so only the export order can
        # be pinned here: first-appearance gave ('1 0'), the skill's
        # canonical order is ('0 1').
        self.assertEqual(_harness([2, 0]), ([1, 1], [0, 1]))
        with self.assertRaises(ValueError) as raised:
            _skill([2, 0])
        self.assertIn('first state spin', str(raised.exception))

    def test_the_classic_renderer_uses_the_same_grouping(self):
        text = _render_molecule_section('molecule', [
            ('state_spin', [0, 2, 0]), ('multiplicity', 1)])
        lines = {line.split()[0]: line.split()[1:]
                 for line in text.splitlines()
                 if line.split() and line.split()[0] in ('ci', 'spin')}
        self.assertEqual(lines['ci'], ['2', '1'])
        self.assertEqual(lines['spin'], ['0', '1'])

    def test_the_converter_carries_one_grouping(self):
        text = source_read('pyrai3md/qualification/certification/converter.py')
        self.assertEqual(text.count('def _group_state_spin('), 1)
        # One definition plus the three sites that used to group inline.
        self.assertEqual(text.count('_group_state_spin('), 4)
        self.assertNotIn('spin_groups', text)


if __name__ == '__main__':
    unittest.main()