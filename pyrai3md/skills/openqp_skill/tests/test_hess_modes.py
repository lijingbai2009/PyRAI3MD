"""
======================================================================
Test suite for the normal-mode fold of OpenQP's hessian export:
- pyrai3md/skills/openqp_skill/service/parser/output_parser.py
  (`_as_norm_modes`)

The skill's own schema declares `norm_mode` as
`(state, mode, atom, xyz)`, while the engine's `<project>.hess.json` writes
each mode as one flat row of `3 * natom` numbers. Unfolded, that 2-D array
was read by the result printer as nested state/mode lists and aborted a run
the engine had finished (measured in the test environment: `modes` is (12, 18) for
acetylene, round tmp_20261011_085800, openqp `hf_frequency_input` and
`mrsf_frequency_input`). These cases pin the fold for the layouts the engine
and its neighbours can produce, and the refusal to invent a layout it cannot
read.

Oct 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ..service.parser.output_parser import _as_norm_modes

NATOM = 6


def test_the_measured_flat_layout_folds_to_one_state():
    array = _as_norm_modes(np.zeros((12, 3 * NATOM)), NATOM)
    assert array.shape == (1, 12, NATOM, 3)


def test_a_state_axis_is_kept():
    assert _as_norm_modes(np.zeros((2, 12, 3 * NATOM)), NATOM).shape \
        == (2, 12, NATOM, 3)


def test_an_already_folded_array_passes_through():
    for array in (np.zeros((12, NATOM, 3)), np.zeros((1, 12, NATOM, 3))):
        out = _as_norm_modes(array, NATOM)
        assert out.ndim == 4 and out.shape[-2:] == (NATOM, 3)


def test_an_unreadable_layout_is_empty_rather_than_guessed():
    # (3 * atom, mode) would need a transposition this engine never asked
    # for; reading it anyway would print modes the run did not compute.
    assert _as_norm_modes(np.zeros((3 * NATOM, 12)), NATOM).size == 0
    assert _as_norm_modes([], NATOM).size == 0
