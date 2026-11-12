"""
======================================================================
Test suite for the NN prediction deviation reception channel
(pyrai3md/workflow/records/deviation.py) and the run-log prediction error
section (pyrai3md/utilities/print_results.py).

Tests cover:
- fetch_deviations: the complete key set with the tolerant 0.0 fill
  for absent/None entries and the untouched pass-through of values
  provided by the backend
- merge_deviations: absent keys filled, backend keys never
  overwritten
- judge_deviations: the legacy exceeded semantics (error >
  threshold), the equality boundary and non-numerical tolerance
- _format_prediction_error: section omission without deviations,
  threshold/judgment lines, keys without thresholds and
  non-numerical values

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.workflow.records.deviation import (
    DEVIATION_KEYS, fetch_deviations, merge_deviations,
    judge_deviations,
)


class TestFetchDeviations(unittest.TestCase):
    """Test the tolerant deviation fetch."""

    def test_missing_result_fills_all_keys(self):
        deviations = fetch_deviations(None)
        self.assertEqual(set(deviations), set(DEVIATION_KEYS))
        for key in DEVIATION_KEYS:
            self.assertEqual(deviations[key], 0.0)

    def test_absent_keys_filled_with_zero(self):
        deviations = fetch_deviations({'energy': -75.0})
        for key in DEVIATION_KEYS:
            self.assertEqual(deviations[key], 0.0)

    def test_none_values_filled_with_zero(self):
        deviations = fetch_deviations({'err_energy': None})
        self.assertEqual(deviations['err_energy'], 0.0)

    def test_present_values_pass_through(self):
        result = {'err_energy': 0.03, 'err_grad': [0.1, 0.2]}
        deviations = fetch_deviations(result)
        self.assertEqual(deviations['err_energy'], 0.03)
        self.assertEqual(deviations['err_grad'], [0.1, 0.2])
        # The input is never mutated.
        self.assertEqual(result, {'err_energy': 0.03,
                                  'err_grad': [0.1, 0.2]})

    def test_zero_fill_is_scalar(self):
        deviations = fetch_deviations({})
        for key in DEVIATION_KEYS:
            self.assertIsInstance(deviations[key], float)


class TestMergeDeviations(unittest.TestCase):
    """Test the deviation merge into the step results."""

    def test_absent_keys_filled(self):
        results = {'energy': -75.0}
        merge_deviations(results, fetch_deviations(None))
        for key in DEVIATION_KEYS:
            self.assertEqual(results[key], 0.0)
        self.assertEqual(results['energy'], -75.0)

    def test_backend_keys_kept(self):
        results = {'err_energy': 0.2}
        merge_deviations(results, fetch_deviations(None))
        self.assertEqual(results['err_energy'], 0.2)
        self.assertEqual(results['err_grad'], 0.0)

    def test_none_results_tolerated(self):
        merge_deviations(None, fetch_deviations(None))


class TestJudgeDeviations(unittest.TestCase):
    """Test the threshold judgment semantics."""

    def test_exceeded_flag(self):
        judged = judge_deviations({'err_energy': 0.2},
                                  {'err_energy': 0.05})
        error, threshold, exceeded = judged['err_energy']
        self.assertEqual(error, 0.2)
        self.assertEqual(threshold, 0.05)
        self.assertTrue(exceeded)

    def test_within_threshold_not_exceeded(self):
        judged = judge_deviations({'err_energy': 0.01},
                                  {'err_energy': 0.05})
        self.assertFalse(judged['err_energy'][2])

    def test_equality_boundary_not_exceeded(self):
        judged = judge_deviations({'err_energy': 0.05},
                                  {'err_energy': 0.05})
        self.assertFalse(judged['err_energy'][2])

    def test_missing_threshold_unjudged(self):
        judged = judge_deviations({'err_dipole': 0.3}, {})
        error, threshold, exceeded = judged['err_dipole']
        self.assertEqual(error, 0.3)
        self.assertIsNone(threshold)
        self.assertFalse(exceeded)

    def test_non_numerical_error_unjudged(self):
        judged = judge_deviations({'err_grad': [0.1, 0.2]},
                                  {'err_grad': 0.15})
        error, threshold, exceeded = judged['err_grad']
        self.assertEqual(error, [0.1, 0.2])
        self.assertIsNone(threshold)
        self.assertFalse(exceeded)

    def test_empty_inputs(self):
        self.assertEqual(judge_deviations({}, None), {})
        self.assertEqual(judge_deviations(None, None), {})


class TestPredictionErrorSection(unittest.TestCase):
    """Test the run-log prediction error section formatting."""

    def _format(self, results, thresholds=None):
        from pyrai3md.utilities.print_results import (
            _format_prediction_error)
        return _format_prediction_error(results, thresholds)

    def test_section_omitted_without_deviations(self):
        self.assertEqual(self._format({'energy': -75.0}), '')
        self.assertEqual(self._format({}), '')
        self.assertEqual(self._format(None), '')

    def test_threshold_and_judgment_lines(self):
        results = {'err_energy': 0.2, 'err_grad': 0.01}
        thresholds = {'err_energy': 0.05, 'err_grad': 0.15}
        block = self._format(results, thresholds)
        self.assertIn('&prediction error', block)
        self.assertIn('err_energy', block)
        self.assertIn('threshold     0.050000', block)
        self.assertIn('Exceeded: TRUE', block)
        self.assertIn('Exceeded: FALSE', block)

    def test_key_without_threshold_lists_value_only(self):
        block = self._format({'err_dipole': 0.0}, {})
        self.assertIn('err_dipole', block)
        self.assertNotIn('threshold', block)
        self.assertNotIn('Exceeded', block)

    def test_non_numerical_value_printed(self):
        block = self._format({'err_grad': [0.1, 0.2]},
                             {'err_grad': 0.15})
        self.assertIn('err_grad', block)
        self.assertIn('[0.1, 0.2]', block)

    def test_numeric_columns_are_right_aligned(self):
        # Labels differ in length, so every value must still occupy
        # the same columns across the section.
        results = {'err_energy': 0.2, 'err_grad': 0.01,
                   'err_trans_dipole': 0.0003}
        thresholds = {'err_energy': 0.05, 'err_grad': 0.15}
        lines = [line for line
                 in self._format(results, thresholds).split('\n')
                 if line.strip().startswith('err_')]

        self.assertEqual(len(lines), 3)
        for line in lines:
            label = line.strip().split()[0]
            self.assertEqual(line[:20], ('  ' + label).ljust(20))
            self.assertEqual(line[20:36], '%16.8f' % results[label])
            if 'threshold' in line:
                self.assertEqual(line.index('threshold'), 38)

    def test_format_result_block_inserts_section(self):
        from pyrai3md.utilities.print_results import format_result_block
        results = {'err_energy': 0.2}
        thresholds = {'err_energy': 0.05}
        block = format_result_block({}, results, step=1,
                                    thresholds=thresholds)
        self.assertIn('&prediction error', block)
        # Without deviation keys the section stays absent.
        plain = format_result_block({}, {'energy': -75.0}, step=1)
        self.assertNotIn('&prediction error', plain)



class TestProducerChannelAgreement(unittest.TestCase):
    """A producer that renames a deviation key must not drift silently.

    The framework receives deviations by key name and never computes them,
    so the skill's own key set and the reception channel have to be the same
    words. Nothing else in either direction catches a rename: an unknown key
    is simply absent, and an absent deviation reads as a perfect surface.
    """

    def test_the_nn_skill_emits_names_the_framework_reads(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            DEVIATION_KEYS as PRODUCER, DEVIATION_UNITS)
        from pyrai3md.workflow.records.deviation import DEVIATION_KEYS

        emitted = set(PRODUCER.values())
        self.assertTrue(emitted, 'the producer declares no deviation keys')
        self.assertEqual(emitted - set(DEVIATION_KEYS), set(),
                         'the skill emits deviation names the framework '
                         'never reads')

    def test_every_emitted_unit_is_a_declared_property_unit(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            DEVIATION_KEYS, DEVIATION_UNITS)
        self.assertEqual(set(DEVIATION_UNITS), set(DEVIATION_KEYS.values()))
        self.assertEqual(DEVIATION_UNITS['err_energy'], 'hartree')
        self.assertEqual(DEVIATION_UNITS['err_grad'], 'hartree/bohr')
        self.assertEqual(DEVIATION_UNITS['err_soc'], 'cm-1')

    def test_the_thresholds_of_the_input_map_onto_those_names(self):
        # &md's md_max* keywords are what the run compares against; an
        # adaptive md child has them overwritten from &adaptive's max* by
        # the parent workflow (test_adaptive_sampling covers that half).
        from pyrai3md.workflow.md import AIMD
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize

        class _State:
            pass

        state = _State()
        state.pmd = normalize(build_config({
            'control': {'jobtype': 'md', 'theory': ['nn']},
            'md': {'md_maxenergy': 0.05, 'md_maxgrad': 0.15,
                   'md_maxnac': 0.15, 'md_maxsoc': 50.0}}))
        thresholds = AIMD.state_kwargs(state)['deviation_thresholds']
        self.assertEqual(sorted(thresholds),
                         ['err_energy', 'err_grad', 'err_nac', 'err_soc'])
        self.assertEqual(thresholds['err_energy'], 0.05)

    def test_without_md_max_a_channel_is_left_unjudged(self):
        from pyrai3md.workflow.md import AIMD
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize

        class _State:
            pass

        state = _State()
        state.pmd = normalize(build_config({
            'control': {'jobtype': 'md', 'theory': ['nn']},
            'md': {'step': 10}}))
        self.assertEqual(
            AIMD.state_kwargs(state)['deviation_thresholds'], {})


if __name__ == '__main__':
    unittest.main()
