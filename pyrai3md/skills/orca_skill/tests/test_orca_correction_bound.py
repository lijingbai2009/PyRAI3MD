"""
======================================================================
The ORCA state-energy resolver's sanity bound (#8.6).

`MAX_CORRECTION` (0.02 Eh) gates the property-file source: a recovered
additive correction larger than that means the reference energies were
mis-read, and the historic "final single point energy is the ground
state" path is safer than a shifted surface. The review asked whether
the bound still participates or is a vestige of the old positional
mapping.

It participates: the two cases below take different sources with the
same totals and only the size of the correction changed. The threshold
stays a module constant rather than a configuration key -- it is a
sanity gate on a mis-read, not a per-case cost, and exposing it invites
an input that hides the mis-read.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from ..service.parser.output_parser import (
    MAX_CORRECTION, _resolve_state_energies)

E_SCF = -100.0
EXCITATION = 0.1
TOTAL = -99.85


def _case(correction):
    """One property record whose totals agree with the final energy."""
    final = TOTAL + correction
    data = {'s0_energy': final, 'escf_energy': None, 'dispersion': None,
            'root_excitations': {}}
    prop = {'e_scf': E_SCF, 'state_totals': {1: TOTAL},
            'corrections': correction}
    return _resolve_state_energies(data, prop, [EXCITATION], 2)


class TestTheBoundDecidesTheSource(unittest.TestCase):
    """Same totals, different correction size, different source."""

    def test_a_correction_inside_the_bound_is_applied(self):
        energies = _case(0.015)
        self.assertAlmostEqual(energies[0], E_SCF + 0.015)
        # The labelled per-root total replaces the ladder value.
        self.assertAlmostEqual(energies[1], TOTAL + 0.015)

    def test_a_correction_past_the_bound_falls_back(self):
        energies = _case(MAX_CORRECTION + 0.01)
        # The final single point energy stands in as the ground state.
        self.assertAlmostEqual(energies[0], TOTAL + MAX_CORRECTION + 0.01)
        self.assertNotAlmostEqual(energies[0], E_SCF)

    def test_inconsistent_totals_fall_back_whatever_the_size(self):
        data = {'s0_energy': -50.0, 'escf_energy': None, 'dispersion': None,
                'root_excitations': {}}
        prop = {'e_scf': E_SCF, 'state_totals': {1: TOTAL},
                'corrections': 0.001}
        energies = _resolve_state_energies(data, prop, [EXCITATION], 2)
        self.assertAlmostEqual(energies[0], -50.0)


if __name__ == '__main__':
    unittest.main()