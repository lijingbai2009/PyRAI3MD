"""
======================================================================
One reading of `&system pbc` across the two consumers (#365, 4.5).

The constraint adapter read the flags itself (any non-zero element of
whatever sequence `pbc` happened to be) while the integrator went
through `utilities/lattice.normalize_pbc`. The adapter now shares the
normalizer, so a spelling the normalizer refuses (a flag of 2, a wrong
length) is refused by both instead of being read as "periodic" by one.

The image override stays where it was: an atom that has crossed a face
makes the run periodic even with the flags off, because at that point
the stored coordinate no longer says where the atom is.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest
from types import SimpleNamespace

import numpy as np

from pyrai3md.workflow.constraints.state_adapter import _periodic
from pyrai3md.workflow.integrator.operations import _is_periodic

SPELLINGS = (None, 0, 1, [0, 0, 0], [1, 0, 1], [True, False, True],
             (1, 1, 0), 1.0)


def _state(pbc, image=None):
    system = SimpleNamespace(pbc=pbc,
                             image=[[0, 0, 0]] if image is None else image)
    return system, SimpleNamespace(pmd=SimpleNamespace(system=system))


class TestOnePeriodicityReading(unittest.TestCase):
    """Both predicates answer the same for every accepted spelling."""

    def test_the_two_predicates_agree(self):
        cell = np.eye(3) * 10.0
        for pbc in SPELLINGS:
            with self.subTest(pbc=pbc):
                system, state = _state(pbc)
                self.assertEqual(_periodic(system, cell)[0],
                                 _is_periodic(state))

    def test_the_flags_are_the_normalized_integers(self):
        cell = np.eye(3) * 10.0
        system, _state_view = _state([True, False, True])
        self.assertEqual(_periodic(system, cell)[1].tolist(), [1, 0, 1])

    def test_an_invalid_flag_is_refused_by_both(self):
        cell = np.eye(3) * 10.0
        system, state = _state([2, 0, 0])
        with self.assertRaises(ValueError):
            _periodic(system, cell)
        with self.assertRaises(ValueError):
            _is_periodic(state)

    def test_a_crossed_atom_still_makes_the_run_periodic(self):
        cell = np.eye(3) * 10.0
        system, _view = _state([0, 0, 0], image=[[1, 0, 0]])
        self.assertTrue(_periodic(system, cell)[0])


if __name__ == '__main__':
    unittest.main()