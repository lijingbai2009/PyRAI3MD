"""
======================================================================
Unit tests for the OpenQP skill spin-orbit routing (OpenQP 1.3).

Covers the framework-driven combination contract:
- the 'soc' jobtype token resolves to a SOC profile, in any token order;
- runtype=soc takes couplings only, runtype=data takes gradients with
  them, and the two ladders are addressed through grad_s / grad_t;
- a two-manifold deck pins Cartesian functions, sizes each ladder and
  never writes the legacy combined grad selector;
- SOC cannot be paired with a NAC of any flavour, and the refusal names
  the engine rule rather than saying 'unsupported';
- nstate_s / nstate_t fall back on the declared states and are checked
  when given;
- the pair maps classify the pairs the framework declared instead of
  rebuilding the full cross product, which is what the surface-hopping
  integrators index the returned rows by;
- the parser reads the labelled exports and collapses socme's three Ms
  rows into one magnitude per pair, refusing rather than zero-filling.

Measured anchors taken from OpenQP 1.3.1 on this machine (see
tests/mrsf_soc.yaml and tests/mrsf_soc_gradient.yaml for the decks).

Oct 23 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.skills.openqp_skill.schema.adapter import build_config
from pyrai3md.skills.openqp_skill.schema.normalize import normalize
from pyrai3md.skills.openqp_skill.schema.validate import validate
from pyrai3md.skills.openqp_skill.schema.method import resolve_profile
from pyrai3md.skills.openqp_skill.service.workflow.compiler import (
    compile_workflow)
from pyrai3md.skills.openqp_skill.service.builder.input_builder import (
    build_input)
from pyrai3md.skills.openqp_skill.service.parser.output_parser import (
    parse_output)

ATOMS = ['C', 'C', 'H', 'H', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.33], [0.0, 0.92, -0.57],
          [0.0, -0.92, -0.57], [0.92, 0.0, 1.9], [-0.92, 0.0, 1.9]]

# Two singlets and two triplets, with the cross-spin pairs the framework
# would declare for them.
SOC_INPUT = {
    'nstate': 4,
    'state_spin': [0, 0, 2, 2],
    'coupling': [[0, 1], [0, 2], [1, 3]],
    'target_states': [0, 3],
}


def _config(profile='mrsf_soc', **extra):
    raw = {
        'atoms': list(ATOMS),
        'coordinates': [list(row) for row in COORDS],
        'charges': 0,
        'multiplicity': 1,
        'profile': profile,
        'method': 'mrsf',
        'basis': 'sto-3g',
        'functional': 'bhhlyp',
        'execution_mode': 'local',
    }
    raw.update(SOC_INPUT)
    raw.update(extra)
    return validate(normalize(build_config(raw)))


def _workflow(config):
    from pyrai3md.skills.openqp_skill.schema.validate import (
        soc_is_requested)
    return compile_workflow(
        config.calculation.profile,
        nac_type=config.calculation.nac_type,
        is_soc=soc_is_requested(config.calculation),
        has_grad_t=bool(config.calculation.grad_t))


class TestProfileResolution(unittest.TestCase):
    """The soc token finds a SOC profile whatever order it arrives in."""

    def test_soc_tokens_map_to_a_soc_profile(self):
        self.assertEqual(resolve_profile('mrsf', 'soc'), 'mrsf_soc')
        self.assertEqual(resolve_profile('mrsf', 'energy soc'), 'mrsf_soc')
        self.assertEqual(resolve_profile('mrsf', 'energy gradient soc'),
                         'mrsf_soc_gradient')

    def test_token_order_does_not_change_the_profile(self):
        self.assertEqual(resolve_profile('mrsf', 'gradient energy soc'),
                         'mrsf_soc_gradient')
        self.assertEqual(resolve_profile('mrsf', 'soc gradient energy'),
                         'mrsf_soc_gradient')

    def test_coupling_and_nac_land_on_the_soc_profile(self):
        # Chosen so that validate, not this lookup, is what refuses the
        # request: the message has to name the engine rule.
        self.assertEqual(
            resolve_profile('mrsf', 'energy gradient nac soc'),
            'mrsf_soc_gradient')

    def test_dynamics_jobtype_still_maps_to_the_dynamics_profile(self):
        self.assertEqual(resolve_profile('mrsf', 'md'),
                         'mrsf_nonadiabatic_dynamics')


class TestWorkflowFlags(unittest.TestCase):
    """runtype and the two-manifold flags follow the mapping table."""

    def test_soc_only_profile_takes_no_gradient_path(self):
        wf = _workflow(_config('mrsf_soc'))
        self.assertEqual(wf.runtype, 'soc')
        self.assertTrue(wf.is_soc)
        self.assertTrue(wf.two_manifold)

    def test_soc_gradient_profile_runs_on_data(self):
        wf = _workflow(_config('mrsf_soc_gradient'))
        self.assertEqual(wf.runtype, 'data')
        self.assertTrue(wf.two_manifold)

    def test_dynamics_with_dosoc_runs_on_data(self):
        wf = _workflow(_config('mrsf_nonadiabatic_dynamics', dosoc=1))
        self.assertEqual(wf.runtype, 'data')
        self.assertTrue(wf.is_soc)
        self.assertTrue(wf.two_manifold)

    def test_plain_dynamics_is_not_two_manifold(self):
        config = _config('mrsf_nonadiabatic_dynamics', nac_type='ktdc',
                         nstate=3, state_spin=[0, 0, 0], coupling=[[0, 1]],
                         target_states=[0, 1, 2], dosoc=0)
        wf = _workflow(config)
        self.assertFalse(wf.two_manifold)
        self.assertFalse(wf.is_soc)


class TestDeckText(unittest.TestCase):
    """What the generated deck says for a spin-orbit run."""

    def _deck(self, config):
        return build_input(config, _workflow(config))

    def test_cartesian_functions_are_pinned_for_soc(self):
        deck = self._deck(_config('mrsf_soc'))
        self.assertIn('ispher=false', deck)
        self.assertIn('soc_2e=1', deck)

    def test_two_electron_term_can_be_dropped(self):
        deck = self._deck(_config('mrsf_soc', soc_2e=0))
        self.assertIn('soc_2e=0', deck)

    def test_ladder_sizes_are_written(self):
        deck = self._deck(_config('mrsf_soc'))
        self.assertIn('nstate_s=2', deck)
        self.assertIn('nstate_t=2', deck)
        # [tdhf] nstate is the configured ladder's count, not the total:
        # each manifold is published with 1 + nstate_label rows.
        tdhf = deck[deck.index('[tdhf]'):deck.index('[properties]')]
        self.assertIn('nstate=2\n', tdhf)
        self.assertNotIn('nstate=4', tdhf)

    def test_gradient_selectors_are_per_ladder(self):
        deck = self._deck(_config('mrsf_soc_gradient'))
        self.assertIn('grad_s=1', deck)
        self.assertIn('grad_t=2', deck)
        self.assertNotIn('\ngrad=', deck)

    def test_soc_switch_is_off_the_couplings_only_driver(self):
        # runtype=soc is the dedicated driver: the upstream examples write
        # no properties.soc for it, and it refuses a gradient selector.
        couplings = self._deck(_config('mrsf_soc'))
        self.assertNotIn('soc=true', couplings)
        self.assertNotIn('grad_s', couplings)
        both = self._deck(_config('mrsf_soc_gradient'))
        self.assertIn('soc=true', both)

    def test_explicit_lists_outrank_the_followed_states(self):
        deck = self._deck(_config('mrsf_soc_gradient',
                                  grad_s=[2], grad_t=[1]))
        self.assertIn('grad_s=2', deck)
        self.assertIn('grad_t=1', deck)
        self.assertNotIn('grad_s=1\n', deck)

    def test_scf_extras_are_written_only_when_stated(self):
        plain = self._deck(_config('mrsf_soc'))
        self.assertNotIn('escalation', plain)
        self.assertNotIn('converger_type', plain)
        self.assertNotIn('scal_rel', plain)

        stated = self._deck(_config('mrsf_soc', escalation='soscf,trah',
                                    scf_converger='soscf', scal_rel=1))
        self.assertIn('escalation=soscf,trah', stated)
        self.assertIn('converger_type=soscf', stated)
        self.assertIn('scal_rel=1', stated)

    def test_upstream_spelling_of_the_converger_is_accepted(self):
        config = _config('mrsf_soc', converger_type='soscf')
        self.assertEqual(config.calculation.scf_converger, 'soscf')
        self.assertIn('converger_type=soscf', self._deck(config))

    def test_a_non_soc_deck_is_left_alone(self):
        config = validate(normalize(build_config({
            'atoms': list(ATOMS),
            'coordinates': [list(row) for row in COORDS],
            'charges': 0, 'multiplicity': 1, 'nstate': 3,
            'state_spin': [0, 0, 0], 'target_states': [0, 1, 2],
            'profile': 'mrsf_gradient', 'method': 'mrsf',
            'basis': 'sto-3g', 'functional': 'bhhlyp'})))
        deck = build_input(config, _workflow(config))
        self.assertIn('\ngrad=1,2,3', deck)
        for absent in ('ispher', 'soc_2e', 'soc=true', 'grad_s', 'nstate_s'):
            self.assertNotIn(absent, deck)


class TestSpinOrbitRefusals(unittest.TestCase):
    """Combinations OpenQP 1.3 cannot answer are refused by name."""

    def test_nac_vectors_are_refused_alongside_soc(self):
        with self.assertRaises(ValueError) as ctx:
            _config('mrsf_soc', nac_type='nac')
        message = str(ctx.exception)
        self.assertIn('cannot be combined with NAC vectors', message)
        self.assertIn('singlet fold', message)

    def test_dosoc_counts_as_a_soc_request_for_the_refusal(self):
        with self.assertRaises(ValueError):
            _config('mrsf_nonadiabatic_dynamics', dosoc=1, nac_type='nac')

    def test_couplings_without_a_second_ladder_are_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _config('mrsf_soc', state_spin=[0, 0, 0], nstate=3,
                    coupling=[[0, 1]], target_states=[0, 2])
        self.assertIn('singlet-triplet block', str(ctx.exception))

    def test_a_triplet_row_is_allowed_for_a_two_manifold_run(self):
        # The rule this replaces refused any triplet row at multiplicity 1,
        # which is exactly what a singlet-triplet coupling run declares.
        config = _config('mrsf_soc')
        self.assertEqual(config.system.state_spin, [0, 0, 2, 2])

    def test_ladder_size_below_the_declared_states_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _config('mrsf_soc', nstate_t=1)
        self.assertIn('nstate_t', str(ctx.exception))

    def test_a_gradient_root_past_its_ladder_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _config('mrsf_soc_gradient', grad_s=[3])
        self.assertIn('grad_s', str(ctx.exception))

    def test_a_ladder_without_a_request_is_still_measured(self):
        # OpenQP only sizes a ladder that carries a gradient request, but
        # nstate_s x nstate_t x 3 is what socme is enumerated over, so a
        # short ladder with no request still breaks the reader.
        with self.assertRaises(ValueError):
            _config('mrsf_soc', nstate_s=1)


class TestPairMaps(unittest.TestCase):
    """The returned rows are ordered the way the integrators index them."""

    def test_declared_pairs_are_classified_not_cross_multiplied(self):
        config = _config('mrsf_soc')
        self.assertEqual(config.system.soc_pair_map, [[0, 2], [1, 3]])
        self.assertEqual(config.system.nac_pair_map, [[0, 1]])
        self.assertEqual(config.system.nsoc, 2)

    def test_cross_product_is_the_standalone_fallback(self):
        config = _config('mrsf_soc', coupling=[])
        self.assertEqual(config.system.soc_pair_map,
                         [[0, 2], [0, 3], [1, 2], [1, 3]])

    def test_framework_adjacent_default_yields_one_soc_row(self):
        # ci [2,2] with the default adjacent coupling gives the framework
        # one cross-spin pair; the skill must agree or the rows it returns
        # land on the wrong pairs.
        config = _config('mrsf_soc',
                         coupling=[[0, 1], [1, 2], [2, 3]])
        self.assertEqual(config.system.soc_pair_map, [[1, 2]])
        self.assertEqual(config.system.nsoc, 1)


class TestTwoManifoldParsing(unittest.TestCase):
    """Labelled exports and the socme table."""

    @staticmethod
    def _log(tmp):
        with open(os.path.join(tmp, 'openqp_calc.log'), 'w') as handle:
            handle.write('OpenQP finished normally\n')

    @staticmethod
    def _socme(rows):
        header = ('# S_index T_index Ms Re(cm-1) Im(cm-1) Absolute(cm-1) '
                  '[merged 1e+2e]\n')
        return header + ''.join(
            '%6d%6d%6d%24.16f%24.16f%24.16f\n' % row for row in rows)

    @classmethod
    def _surface_files(cls, tmp):
        for name, rows in (('energies_s', [-77.45, -77.41, -77.29]),
                           ('energies_t', [-77.45, -77.413, -77.10])):
            with open(os.path.join(tmp, name), 'w') as handle:
                handle.writelines('%24.16f\n' % value for value in rows)
        for name in ('grad_s_1', 'grad_t_2'):
            with open(os.path.join(tmp, name), 'w') as handle:
                for index in range(len(ATOMS)):
                    handle.write('%24.16f%24.16f%24.16f\n'
                                 % (0.1, 0.0, 0.0))

    def test_labels_and_ladders_are_read_from_the_named_files(self):
        config = _config('mrsf_soc_gradient')
        wf = _workflow(config)
        with tempfile.TemporaryDirectory() as tmp:
            self._log(tmp)
            self._surface_files(tmp)
            # S0/T0 forbidden, S1/T1 at 8.6352: the three Ms rows of a pair
            # combine as the square root of the summed squares, the shape of
            # the measured 2 x 2 x 3 table.
            rows = []
            values = {(0, 0): (0.0, 0.0, 0.0),
                      (0, 1): (1.0, 2.0, 3.0),
                      (1, 0): (0.5, 0.5, 0.5),
                      (1, 1): (6.1061, 0.0, 6.1061)}
            for ist in (0, 1):
                for jst in (0, 1):
                    for ms, magnitude in zip((-1, 0, 1), values[(ist, jst)]):
                        rows.append((ist, jst, ms, 0.0, 0.0, magnitude))
            with open(os.path.join(tmp, 'socme'), 'w') as handle:
                handle.write(self._socme(rows))
            results = parse_output(config, tmp, wf)
        self.assertAlmostEqual(float(results['energy'][0]), -77.41, places=6)
        self.assertAlmostEqual(float(results['energy'][2]), -77.413, places=6)
        self.assertAlmostEqual(float(results['energy'][3]), -77.10, places=6)
        self.assertAlmostEqual(float(results['gradient'][0][0][0]), 0.1,
                               places=9)
        self.assertAlmostEqual(float(results['gradient'][3][0][0]), 0.1,
                               places=9)
        self.assertEqual(len(results['soc']), 2)
        self.assertAlmostEqual(float(results['soc'][0]), 0.0, places=9)
        # sqrt(6.1061^2 + 6.1061^2) = 8.6352, the Ms-collapse convention
        self.assertAlmostEqual(float(results['soc'][1]), 8.6352, places=3)

    def test_missing_socme_raises_rather_than_reporting_zeros(self):
        config = _config('mrsf_soc_gradient')
        wf = _workflow(config)
        with tempfile.TemporaryDirectory() as tmp:
            self._log(tmp)
            self._surface_files(tmp)
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('socme', str(ctx.exception))

    def test_a_short_socme_table_raises(self):
        config = _config('mrsf_soc_gradient')
        wf = _workflow(config)
        with tempfile.TemporaryDirectory() as tmp:
            self._log(tmp)
            self._surface_files(tmp)
            with open(os.path.join(tmp, 'socme'), 'w') as handle:
                handle.write(self._socme([(0, 0, 0, 0.0, 0.0, 1.0),
                                          (0, 0, 1, 0.0, 0.0, 1.0),
                                          (1, 0, -1, 0.0, 0.0, 1.0)]))
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('x 3 Ms = 12', str(ctx.exception))

    def test_a_declared_pair_without_rows_raises(self):
        config = _config('mrsf_soc_gradient')
        wf = _workflow(config)
        with tempfile.TemporaryDirectory() as tmp:
            self._log(tmp)
            self._surface_files(tmp)
            # 12 rows, all of them for S0/T0: the S1/T3 pair the input
            # declares has nothing to read.
            rows = [(0, 0, ms, 0.0, 0.0, 1.0) for ms in (-1, 0, 1)] * 4
            with open(os.path.join(tmp, 'socme'), 'w') as handle:
                handle.write(self._socme(rows))
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('no row', str(ctx.exception))

    def test_missing_labelled_energy_raises(self):
        config = _config('mrsf_soc_gradient')
        wf = _workflow(config)
        with tempfile.TemporaryDirectory() as tmp:
            self._log(tmp)
            with open(os.path.join(tmp, 'energies_t'), 'w') as handle:
                handle.write('-77.45\n-77.41\n-77.10\n')
            for name in ('grad_s_1', 'grad_t_2'):
                with open(os.path.join(tmp, name), 'w') as handle:
                    for _ in ATOMS:
                        handle.write('0.0 0.0 0.0\n')
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('energies_s', str(ctx.exception))

    def test_couplings_only_run_reports_no_gradient_files(self):
        # runtype=soc computes no gradient, so the absence of grad_* is
        # not a failure there.
        config = _config('mrsf_soc')
        wf = _workflow(config)
        with tempfile.TemporaryDirectory() as tmp:
            self._log(tmp)
            for name, rows in (('energies_s', [-77.45, -77.41, -77.29]),
                               ('energies_t', [-77.45, -77.41, -77.10])):
                with open(os.path.join(tmp, name), 'w') as handle:
                    handle.writelines('%f\n' % v for v in rows)
            with open(os.path.join(tmp, 'socme'), 'w') as handle:
                handle.write(self._socme(
                    [(s, t, ms, 0.0, 0.0, 1.0)
                     for s in (0, 1) for t in (0, 1) for ms in (-1, 0, 1)]))
            results = parse_output(config, tmp, wf)
        self.assertEqual(np.abs(results['gradient']).max(), 0.0)
        self.assertEqual(len(results['soc']), 2)


if __name__ == '__main__':
    unittest.main()
