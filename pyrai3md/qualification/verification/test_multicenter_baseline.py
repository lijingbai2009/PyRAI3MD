"""
======================================================================
R0 baseline fixation for the multi-center unified base refactor
(dev/MULTICENTER_DEVELOPMENT.md section 13, acceptance matrix).

These are characterization tests: they lock the CURRENT behavior of
the single-center pipeline so later refactor phases (R1-R7) can be
checked against a fixed reference instead of drifting silently. They
do not change any production code path.

Three groups are frozen here:
1. Raw input index base (adapter): external indices are 1-based, the
   internal normalized form is 0-based. Two known defects are pinned
   and clearly labelled so R1 can flip them deliberately.
2. Single-center normalize derivation: nstate / state_spin / pair
   maps / current_state, plus the zip silent-truncation defect.
3. History timing sentinels: update_history snapshot semantics and
   the get_history_step(-1/-2/-3) chronological offsets the FSSH /
   GSH / LZSH integrators rely on.

Sep 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest


class TestR0AdapterIndexBase(unittest.TestCase):
    """Freeze the raw 1-based / normalized 0-based index contract."""

    def test_highlevel_raw_string_is_one_based(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        # '1-3 5' (external 1-based) -> single center [0,1,2,4]
        self.assertEqual(
            _parse_center_indices('1-3 5', 'highlevel', start=1),
            [[0, 1, 2, 4]])

    def test_highlevel_raw_semicolon_splits_centers(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        self.assertEqual(
            _parse_center_indices('1-3; 8-11', 'highlevel', start=1),
            [[0, 1, 2], [7, 8, 9, 10]])

    def test_highlevel_typed_flat_wraps_single_center(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        self.assertEqual(
            _parse_center_indices([1, 2, 3], 'highlevel', start=1),
            [[0, 1, 2]])

    def test_highlevel_typed_nested_subtracts_one(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        self.assertEqual(
            _parse_center_indices([[1, 2], [3, 4]], 'highlevel', start=1),
            [[0, 1], [2, 3]])

    def test_highlevel_empty_group_raises(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        with self.assertRaises(ValueError):
            _parse_center_indices('1-3;;8', 'highlevel', start=1)

    def test_coupling_raw_string_is_one_based(self):
        from pyrai3md.schema.adapter import _convert_types
        # coupling text state indices are 1-based -> 0-based pairs
        self.assertEqual(
            _convert_types({'coupling': '1 2, 3 4'})['coupling'],
            [[0, 1], [2, 3]])

    # --- R0 defects flipped by R1: the raw / normalized entry split ---

    def test_normalized_typed_highlevel_no_double_subtract(self):
        """R1 FIX: the normalized entry never decrements typed input.

        _parse_center_indices now takes an explicit ``raw`` flag (docs
        4.3). raw=False is the normalized programmatic entry: an
        already 0-based [[0, 1]] is passed through unchanged. The raw
        entry (default raw=True) still treats a typed list as 1-based
        external input and decrements it once, which is the documented
        raw behaviour rather than a silent double subtraction.
        """
        from pyrai3md.schema.adapter import _parse_center_indices
        # normalized entry: 0-based typed values pass through
        self.assertEqual(
            _parse_center_indices([[0, 1]], 'highlevel', start=1,
                                  raw=False),
            [[0, 1]])
        # raw entry: 1-based typed values are decremented exactly once
        self.assertEqual(
            _parse_center_indices([[1, 2]], 'highlevel', start=1,
                                  raw=True),
            [[0, 1]])

    def test_typed_coupling_is_normalized_zero_based_passthrough(self):
        """R1 CONTRACT: typed coupling is the normalized 0-based entry.

        The two entry points are now explicit (docs 4.3): a raw string
        carries 1-based state indices and is decremented once, while a
        typed array is already normalized 0-based and passes through
        unchanged. The agreement is on the SEMANTICS (both land on
        0-based pairs), not on re-running the raw parser over typed
        input -- which would corrupt the molcas fixtures and the
        converter-generated adjacent pairs.
        """
        from pyrai3md.schema.adapter import _convert_types
        # raw string entry: 1-based -> 0-based
        self.assertEqual(
            _convert_types({'coupling': '1 2, 3 4'})['coupling'],
            [[0, 1], [2, 3]])
        # normalized typed entry: 0-based passthrough (no decrement)
        self.assertEqual(
            _convert_types({'coupling': [[0, 1], [2, 3]]})['coupling'],
            [[0, 1], [2, 3]])


class TestR0NormalizeSingleCenter(unittest.TestCase):
    """Freeze the single-center derived-quantity contract."""

    def test_ci_spin_derive_nstate_and_state_spin(self):
        from pyrai3md.schema.normalize import compute_derived_quantities
        data = {'ci': [2, 2], 'spin': [0, 1], 'coupling': [],
                'atoms': ['H', 'H'], 'charges': 0, 'multiplicity': 1}
        data = compute_derived_quantities(data)
        self.assertEqual(data['nstate'], 4)
        self.assertEqual(data['state_spin'], [0, 0, 2, 2])

    def test_declared_pairs_classify_and_nothing_is_invented(self):
        from pyrai3md.schema.normalize import compute_derived_quantities
        pairs = [[0, 1], [1, 2], [2, 3]]
        data = compute_derived_quantities(
            {'ci': [2, 2], 'spin': [0, 1], 'coupling': pairs,
             'atoms': ['H', 'H'], 'charges': 0, 'multiplicity': 1})
        # same spin -> NAC, different spin -> SOC
        self.assertEqual(data['nac_pair_map'], [[0, 1], [2, 3]])
        self.assertEqual(data['soc_pair_map'], [[1, 2]])
        self.assertEqual(data['npair'], 3)
        self.assertEqual(data['nnac'], 2)
        self.assertEqual(data['nsoc'], 1)
        # The pair maps are the derived side of the same statement.
        self.assertEqual(data['coupling'], pairs)

        # An undeclared coupling stays undeclared: adjacent pairs used to
        # be auto-generated for any nstate >= 2, which read back as
        # couplings nobody asked about.
        empty = compute_derived_quantities(
            {'ci': [2, 2], 'spin': [0, 1], 'coupling': [],
             'atoms': ['H', 'H'], 'charges': 0, 'multiplicity': 1})
        self.assertEqual([], empty['coupling'])
        self.assertEqual([], empty['nac_pair_map'])
        self.assertEqual([], empty['soc_pair_map'])
        self.assertEqual(0, empty['npair'])

    def test_derive_current_state_from_root(self):
        from pyrai3md.schema.normalize import derive_current_state
        # default current_state (0) is filled from root
        self.assertEqual(
            derive_current_state({'current_state': 0, 'root': 2}),
            {'current_state': 2, 'root': 2})
        # explicit non-default current_state is never overridden
        self.assertEqual(
            derive_current_state({'current_state': 1, 'root': 2}),
            {'current_state': 1, 'root': 2})

    def test_history_normalized_to_at_least_four(self):
        from pyrai3md.schema.normalize import normalize_history
        self.assertEqual(normalize_history({'history': 1})['history'], 4)
        self.assertEqual(normalize_history({'history': 8})['history'], 8)

    # --- R0 defect fixed by R1: strict per-center structural check ---

    def test_ci_spin_length_mismatch_rejected(self):
        """R1 FIX: a ci/spin length mismatch is rejected, not truncated.

        The old zip(ci, spin) silently dropped the third ci block
        (ci=[2, 2, 2] with spin=[0, 1] gave nstate=6 but a length-4
        state_spin). The unified-center structural pre-check now
        requires one spin value per ci block and raises before any
        derived quantity is produced.
        """
        from pyrai3md.schema.normalize import compute_derived_quantities
        data = {'ci': [2, 2, 2], 'spin': [0, 1], 'atoms': [],
                'coupling': []}
        with self.assertRaises(ValueError):
            compute_derived_quantities(data)


class TestR0HistoryTiming(unittest.TestCase):
    """Freeze the history snapshot semantics the integrators read."""

    def _make_state(self, history_limit=4):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.system.atoms = ['H', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        config.dynamics.history = history_limit
        state = PyRAIState(runtime=RuntimeState(total_steps=10))
        state.pmd = config
        return state

    def test_update_history_appends_chronologically(self):
        state = self._make_state()
        for step in range(3):
            state.update_system({'current_state': step})
            state.update_history()
        self.assertEqual(state.history_size, 3)
        # index 0 oldest, -1 newest
        self.assertEqual(state.get_history_step(0)['system']['current_state'], 0)
        self.assertEqual(state.get_history_step(-1)['system']['current_state'], 2)

    def test_history_offsets_are_chronological_sentinels(self):
        """FSSH reads -1/-2/-3; GSH/LZSH read -2/-3 as t-dt/t-2dt."""
        state = self._make_state()
        for step in range(4):
            state.update_system({'current_state': step})
            state.update_history()
        self.assertEqual(state.get_history_step(-1)['system']['current_state'], 3)
        self.assertEqual(state.get_history_step(-2)['system']['current_state'], 2)
        self.assertEqual(state.get_history_step(-3)['system']['current_state'], 1)

    def test_history_limit_slices_oldest_first(self):
        state = self._make_state(history_limit=4)
        for step in range(6):
            state.update_system({'current_state': step})
            state.update_history()
        self.assertEqual(state.history_size, 4)
        # oldest two dropped; remaining are steps 2..5
        self.assertEqual(state.get_history_step(0)['system']['current_state'], 2)
        self.assertEqual(state.get_history_step(-1)['system']['current_state'], 5)

    def test_missing_step_returns_none(self):
        state = self._make_state()
        state.update_history()
        self.assertIsNone(state.get_history_step(-2))

    def test_copy_false_returns_readonly_reference(self):
        state = self._make_state()
        state.update_history()
        ref = state.get_history_step(-1, copy=False)
        ref2 = state.get_history_step(-1, copy=False)
        self.assertIs(ref, ref2)  # same internal object (hot-loop read)
        copied = state.get_history_step(-1, copy=True)
        self.assertIsNot(copied, ref)

    def test_history_snapshot_isolated_from_live_state(self):
        """A snapshot never aliases the live pmd (deepcopy on write)."""
        state = self._make_state()
        state.update_system({'current_state': 0})
        state.update_history()
        state.update_system({'current_state': 5})
        self.assertEqual(state.get_history_step(-1)['system']['current_state'], 0)
        self.assertEqual(state.current_state, 5)


if __name__ == '__main__':
    unittest.main()
