"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Contract tests of the serving layer against a stub container.

The stub returns arrays in exactly the shapes the real container returns
(measured in the migration acceptance run), so key names, state slicing,
coupling-channel ordering and the unit bridge are checked without
TensorFlow and without a trained model.

"""

import unittest

import numpy as np

from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
from pyrai3md.skills.nn_pes_skill.schema.validate import validate
from pyrai3md.skills.nn_pes_skill.schema.reference import BOHR_TO_ANGSTROM
from pyrai3md.skills.nn_pes_skill.service.calculator import evaluate

ATOMS = ['C', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
NATOM = len(ATOMS)
NSTATE = 3
# Two declared spin-orbit pairs, so the spin-orbit ensemble has two channels
NSOC = 2

# The predictor channel is additive: mode 0 writes these keys as None, and
# the prediction-key assertions below read the prediction channels only.
ANALYSIS_KEYS = {
    'alternative_energy', 'alternative_gradient', 'alternative_nac',
    'alternative_soc', 'alternative_model_ids', 'predict_model_id',
    'n_models',
}


def prediction_keys(results):
    """The prediction channels, without the additive comparison channel."""
    return sorted(set(results) - ANALYSIS_KEYS)


class StubContainer:
    """Return fixed outputs in the shapes of the real container."""

    def __init__(self, layout):
        self.layout = layout
        self.seen = None

    def call(self, batch):
        self.seen = np.asarray(batch)
        means = {}
        spreads = {}
        for name, kind in self.layout.items():
            if kind == 'mlp_e' and name == 'soc':
                # The spin-orbit folder holds one scalar per declared pair
                means[name] = np.arange(1.0, NSOC + 1.0)[None]
                spreads[name] = np.zeros_like(means[name])
            elif kind == 'mlp_e':
                means[name] = self._states()
                spreads[name] = np.zeros_like(means[name])
            elif kind == 'mlp_eg':
                means[name] = [self._states(), self._gradients()]
                spreads[name] = [np.zeros_like(means[name][0]),
                                 np.zeros_like(means[name][1])]
            elif kind == 'mlp_g2':
                means[name] = self._gradients()
                spreads[name] = np.zeros_like(means[name])
            else:
                means[name] = self._couplings()
                spreads[name] = np.zeros_like(means[name])
        return means, spreads

    def _states(self):
        return np.arange(1.0, NSTATE + 1.0)[None]

    def _gradients(self):
        # Distinct value per (state, atom, component)
        return np.tile(np.arange(NSTATE * NATOM * 3, dtype=np.float64)
                       .reshape(NSTATE, NATOM, 3), (1, 1, 1, 1))

    def _couplings(self):
        return np.tile(np.arange(NSTATE * NATOM * 3, dtype=np.float64)
                       .reshape(NSTATE, NATOM, 3), (1, 1, 1, 1))


def make_config(profile, method, **over):
    # The ensembles a profile loads are named by the profile itself: one
    # folder per role inside modeldir, 'energy_grad' and 'nac'.
    raw = {
        'profile': profile, 'method': method,
        'modeldir': '/tmp/unused', 'mult_nn': 1, 'nstate': NSTATE,
        'atoms': ATOMS, 'coordinates': COORDS,
        'target_states': list(range(NSTATE)), 'target_nac_pair': [],
        'scratch_path': '',
    }
    raw.update(over)
    return validate(normalize(build_config(raw)))


class TestEnergyGradient(unittest.TestCase):

    def setUp(self):
        self.config = make_config('mlp_eg_gradient', 'mlp_eg')
        self.stub = StubContainer({'energy_grad': 'mlp_eg'})
        self.results, self.spread = evaluate(self.config, COORDS,
                                             container=self.stub)

    def test_keys(self):
        self.assertEqual(prediction_keys(self.results), ['energy', 'gradient'])

    def test_energy_shape_and_values(self):
        self.assertEqual(np.shape(self.results['energy']), (NSTATE,))
        self.assertEqual(list(self.results['energy']), [1.0, 2.0, 3.0])

    def test_gradient_shape(self):
        self.assertEqual(np.shape(self.results['gradient']), (NSTATE, NATOM, 3))

    def test_gradient_is_converted_once(self):
        raw = np.arange(NSTATE * NATOM * 3, dtype=np.float64)
        expected = raw.reshape(NSTATE, NATOM, 3) * BOHR_TO_ANGSTROM
        np.testing.assert_allclose(self.results['gradient'], expected,
                                   rtol=0, atol=0)

    def test_batch_axis_is_dropped(self):
        # The framework consumes (nstate, natom, 3), never a batch axis
        self.assertEqual(np.asarray(self.results['gradient']).ndim, 3)

    def test_geometry_is_fed_unchanged(self):
        np.testing.assert_allclose(self.stub.seen[0],
                                   np.asarray(COORDS, dtype=np.float32),
                                   rtol=0, atol=1e-6)

    def test_requested_states_do_not_slice_the_output(self):
        # Properties.energy is (nstate,) and the propagator indexes it by
        # state number, so the full state axis is always reported.
        config = make_config('mlp_eg_gradient', 'mlp_eg', target_states=[2, 0])
        results, _spread = evaluate(config, COORDS, container=self.stub)
        self.assertEqual(list(results['energy']), [1.0, 2.0, 3.0])
        self.assertEqual(np.shape(results['gradient']), (NSTATE, NATOM, 3))


class TestEnergyOnly(unittest.TestCase):

    def test_energy_network_reports_no_gradient(self):
        config = make_config('mlp_e_energy', 'mlp_e')
        results, _spread = evaluate(config, COORDS,
                                    container=StubContainer({'energy_grad': 'mlp_e'}))
        self.assertEqual(prediction_keys(results), ['energy'])


class TestGradientOnly(unittest.TestCase):

    def test_direct_gradient_network_reports_no_energy(self):
        config = make_config('mlp_g2_gradient', 'mlp_g2')
        results, _spread = evaluate(config, COORDS,
                                    container=StubContainer({'energy_grad': 'mlp_g2'}))
        self.assertEqual(prediction_keys(results), ['gradient'])


class TestCouplings(unittest.TestCase):

    def setUp(self):
        self.stub = StubContainer({'nac': 'mlp_nac2'})

    def test_coupling_channel_count(self):
        config = make_config('mlp_nac2_nac', 'mlp_nac2',
                             nac_pair_map=[[0, 1], [0, 2], [1, 2]])
        results, _spread = evaluate(config, COORDS, container=self.stub)
        self.assertEqual(np.shape(results['nac']), (3, NATOM, 3))

    def test_requested_pairs_reorder_channels(self):
        config = make_config('mlp_nac2_nac', 'mlp_nac2',
                             nac_pair_map=[[0, 1], [0, 2], [1, 2]],
                             target_nac_pair=[[1, 2]])
        results, _spread = evaluate(config, COORDS, container=self.stub)
        self.assertEqual(np.shape(results['nac']), (1, NATOM, 3))
        # Channel 2 of the stub holds the third state block
        raw = (np.arange(NSTATE * NATOM * 3, dtype=np.float64)
               .reshape(NSTATE, NATOM, 3)[2])
        np.testing.assert_allclose(results['nac'][0],
                                   raw * BOHR_TO_ANGSTROM, rtol=0, atol=0)

    def test_pair_outside_the_trained_channels_raises(self):
        config = make_config('mlp_nac2_nac', 'mlp_nac2',
                             nac_pair_map=[[0, 1]],
                             target_nac_pair=[[1, 2]])
        with self.assertRaises(ValueError):
            evaluate(config, COORDS, container=self.stub)


class TestCompositeProfile(unittest.TestCase):

    def test_eg_and_nac_in_one_call(self):
        config = make_config('eg_nac_nac', 'eg_nac',
                             nac_pair_map=[[0, 1], [0, 2], [1, 2]])
        stub = StubContainer({'energy_grad': 'mlp_eg', 'nac': 'mlp_nac'})
        results, _spread = evaluate(config, COORDS, container=stub)
        self.assertEqual(prediction_keys(results), ['energy', 'gradient', 'nac'])
        self.assertEqual(np.shape(results['energy']), (NSTATE,))
        self.assertEqual(np.shape(results['nac']), (3, NATOM, 3))

    def test_no_soc_channel_is_reported(self):
        # The skill trains no spin-orbit model, and an empty placeholder
        # would read as a computed coupling to a consumer of the key.
        config = make_config('mlp_eg_gradient', 'mlp_eg')
        results, _spread = evaluate(config, COORDS,
                                    container=StubContainer({'energy_grad': 'mlp_eg'}))
        self.assertNotIn('soc', results)


class TestCouplingChannelDeclaration(unittest.TestCase):
    """The channels served are the pairs the caller declared.

    A coupling network stores no state-pair names: it carries a channel
    count, and which pair each channel means is a property of the training
    data. The framework passes that declaration in its coupling section, so
    the serving layer reads it there instead of assuming a pair set.
    """

    def setUp(self):
        self.stub = StubContainer({'nac': 'mlp_nac2'})

    def test_declared_pairs_set_the_channel_count(self):
        config = make_config('mlp_nac2_nac', 'mlp_nac2',
                             coupling=[[0, 1], [1, 2], [0, 2]])
        self.assertEqual(config.system.nnac, 3)
        results, _spread = evaluate(config, COORDS, container=self.stub)
        self.assertEqual(np.shape(results['nac']), (3, NATOM, 3))

    def test_a_pair_set_the_model_does_not_match_raises(self):
        # The declaration and the trained width have to agree, or the
        # numbers would go out under state names the model never carried.
        config = make_config('mlp_nac2_nac', 'mlp_nac2',
                             coupling=[[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            evaluate(config, COORDS, container=self.stub)

    def test_a_count_without_pairs_serves_channels_in_order(self):
        config = make_config('mlp_nac2_nac', 'mlp_nac2', nnac=3)
        self.assertEqual(config.system.nac_pair_map, [])
        results, _spread = evaluate(config, COORDS, container=self.stub)
        self.assertEqual(np.shape(results['nac']), (NSTATE, NATOM, 3))

    def test_naming_a_pair_without_a_map_raises(self):
        # Serving a channel under a pair nobody declared would attach the
        # wrong state names to the numbers.
        config = make_config('mlp_nac2_nac', 'mlp_nac2', nnac=3,
                             target_nac_pair=[[0, 1]])
        with self.assertRaises(ValueError):
            evaluate(config, COORDS, container=self.stub)

    def test_the_channel_count_is_not_forced_to_a_pair_enumeration(self):
        # One declared pair is one channel, not the three a full ladder of
        # three states would enumerate.
        config = make_config('mlp_nac2_nac', 'mlp_nac2', nstate=3,
                             coupling=[[0, 1]])
        self.assertEqual(config.system.nnac, 1)
        self.assertEqual(np.shape(config.properties.nac), (1, NATOM, 3))


class TestSpinOrbitServing(unittest.TestCase):
    """A spin-orbit request is answered by the energy family, as scalars.

    One SOC matrix element is one number per declared pair, in cm-1. It is a
    value that depends on geometry rather than a derivative of anything, so
    the ensemble that holds it is an `mlp_e` network in the 'soc' folder and
    no gradient channel comes with it.
    """

    def setUp(self):
        self.stub = StubContainer({'soc': 'mlp_e'})

    def _config(self, **over):
        base = dict(profile='mlp_e_soc', method='mlp_e', nsoc=NSOC,
                    soc_pair_map=[[0, 2], [1, 2]])
        base.update(over)
        return make_config(**base)

    def test_shape_values_and_absence_of_a_gradient(self):
        results, _spread = evaluate(self._config(), COORDS,
                                    container=self.stub)
        self.assertEqual(prediction_keys(results), ['soc'])
        self.assertEqual(np.shape(results['soc']), (NSOC, 1))
        self.assertEqual(results['soc'].tolist(), [[1.0], [2.0]])

    def test_values_are_not_rescaled_by_the_length_unit(self):
        # cm-1 is not a derivative, so the Bohr conversion that the gradient
        # and coupling channels go through must not touch it.
        results, _spread = evaluate(self._config(), COORDS,
                                    container=self.stub)
        self.assertEqual(results['soc'].tolist(), [[1.0], [2.0]])

    def test_requested_pairs_reorder_the_channels(self):
        results, _spread = evaluate(
            self._config(target_soc_pair=[[1, 2]]), COORDS,
            container=self.stub)
        self.assertEqual(results['soc'].tolist(), [[2.0]])

    def test_a_pair_outside_the_declaration_raises(self):
        with self.assertRaises(ValueError):
            evaluate(self._config(target_soc_pair=[[0, 1]]), COORDS,
                     container=self.stub)

    def test_a_count_without_pairs_serves_channels_in_order(self):
        config = make_config('mlp_e_soc', 'mlp_e', nsoc=NSOC)
        self.assertEqual(config.system.soc_pair_map, [])
        results, _spread = evaluate(config, COORDS, container=self.stub)
        self.assertEqual(results['soc'].tolist(), [[1.0], [2.0]])

    def test_the_composite_call_reports_all_three_channels(self):
        # The coupling channels are declared too, because the stub holds
        # one channel per state and the two counts are compared.
        config = make_config('eg_nac_soc', 'eg_nac',
                             nac_pair_map=[[0, 1], [0, 2], [1, 2]],
                             soc_pair_map=[[0, 2], [1, 2]])
        stub = StubContainer({'energy_grad': 'mlp_eg', 'nac': 'mlp_nac',
                              'soc': 'mlp_e'})
        results, _spread = evaluate(config, COORDS, container=stub)
        self.assertEqual(prediction_keys(results),
                         ['energy', 'gradient', 'nac', 'soc'])
        self.assertEqual(np.shape(results['soc']), (NSOC, 1))


class TestDeviationScalars(unittest.TestCase):
    """The ensemble spread reaches the framework as one number per property.

    The spread between the members is this skill's only error bar, and the
    channel the framework receives it on is a scalar per computed property:
    the largest member-to-member standard deviation over that property's
    components, in the property's own unit. The arrays stay in the
    uncertainty file; only the maximum is returned, because the result keys
    feed composite aggregation.
    """

    def setUp(self):
        self.spreads = {
            'energy': np.array([[0.001, 0.004]]),
            'gradient': np.array([[[0.0, 0.0, 0.02], [0.0, 0.05, 0.0]],
                                  [[0.0, 0.0, 0.01], [0.0, 0.03, 0.0]]]),
        }

    def test_one_scalar_per_computed_property(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            deviation_metrics)
        metrics = deviation_metrics(self.spreads, members=2)
        self.assertEqual(sorted(metrics), ['err_energy', 'err_grad'])
        self.assertAlmostEqual(metrics['err_energy'], 0.004)
        self.assertAlmostEqual(metrics['err_grad'], 0.05)

    def test_a_property_that_was_not_computed_gets_no_number(self):
        # Absent is not zero: a zero would read as a coupling the ensemble
        # agrees on perfectly.
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            deviation_metrics)
        metrics = deviation_metrics({'energy': self.spreads['energy']},
                                    members=2)
        self.assertEqual(list(metrics), ['err_energy'])

    def test_a_single_network_reports_no_spread(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            deviation_metrics)
        metrics = deviation_metrics(self.spreads, members=1)
        self.assertEqual(metrics['err_energy'], 0.0)
        self.assertEqual(metrics['err_grad'], 0.0)

    def test_the_backend_returns_them_with_the_properties(self):
        results, _spread = evaluate(
            make_config('mlp_eg_gradient', 'mlp_eg'), COORDS,
            container=StubContainer({'energy_grad': 'mlp_eg'}))
        formatted = dict(results)
        self.assertNotIn('err_energy', formatted)
        # The keys are attached by the backend, not the serving layer, so
        # drive the backend's own reduction here.
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            DEVIATION_KEYS, deviation_metrics)
        attached = deviation_metrics(_spread, members=2)
        self.assertEqual(sorted(attached), ['err_energy', 'err_grad'])
        self.assertEqual(set(DEVIATION_KEYS),
                         {'energy', 'gradient', 'nac', 'soc'})

    def test_the_coupling_and_spin_orbit_channels_use_their_own_keys(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            deviation_metrics)
        metrics = deviation_metrics({'nac': np.zeros((2, 3, 3)),
                                     'soc': np.array([[12.5, 3.25]])},
                                    members=2)
        self.assertEqual(sorted(metrics), ['err_nac', 'err_soc'])
        self.assertAlmostEqual(metrics['err_soc'], 12.5)

    def test_the_step_log_names_the_unit_of_every_number(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            deviation_metrics, format_step)
        spreads = {'energy': np.array([[0.001, 0.002]]),
                   'gradient': np.array([[[0.0, 0.0, 0.05]]])}
        metrics = deviation_metrics(spreads, members=2)
        text = format_step('mlp_eg_gradient', ATOMS,
                           {'energy': [[1.0, 2.0]],
                            'gradient': [[[0.0, 0.0, 3.0]]]},
                           metrics, natom=len(ATOMS))
        self.assertIn('states=2', text)
        self.assertIn('hartree', text)
        self.assertIn('hartree/bohr', text)
        self.assertIn('max member-to-member std', text)
        self.assertIn('err_energy', text)
        # A channel that was not computed is not printed, so a reader
        # cannot mistake an absent coupling for a perfect one.
        self.assertNotIn('err_nac', text)


class TestStepLogWithoutEnergy(unittest.TestCase):
    """A step that computed no energy logs no energy line.

    np.asarray(None) is a size-one array holding nan, so an absent channel
    converts into a state whose value is nan unless it is recognized first.
    A training run is exactly such a step: it fits networks and evaluates
    no geometry.
    """

    def test_no_energy_no_state_and_no_line(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            format_step)
        text = format_step('mlp_eg_train', ATOMS, {}, {})
        self.assertNotIn('energy (per state)', text)
        self.assertNotIn('nan', text)
        self.assertIn('states=0', text)

    def test_a_present_energy_still_prints(self):
        from pyrai3md.skills.nn_pes_skill.service.deviation import (
            format_step)
        text = format_step('mlp_eg_gradient', ATOMS,
                           {'energy': [[-1.5, -1.2]]}, {})
        self.assertIn('states=2', text)
        self.assertIn('energy (per state)', text)


TRAINING_RECORD = {
    'train_errors': {'energy_grad': {'energy': [0.1, 0.2],
                                     'gradient': [0.01, 0.02]}},
    'train_metric_units': {'energy': 'eV', 'gradient': 'eV/angstrom'},
    'train_wall_seconds': 12.5,
    'train_arm': 'local',
    'train_data': '/data/TRAIN.json',
    'train_modeldir': '/run/NN-TEST',
    'train_fitlogs': ['/run/NN-TEST/energy_grad/fitlog_0.txt',
                      '/run/NN-TEST/energy_grad/fitlog_1.txt'],
    'train_prediction_std': {'ndata': 4, 'energy': 0.001,
                             'gradient': 0.002, 'nac': None, 'soc': None},
    'train_summary': '/run/TEST_train.summary.txt',
}


def training_config():
    """A configuration carrying the counts the report heads with."""
    from types import SimpleNamespace
    return SimpleNamespace(system=SimpleNamespace(
        natom=4, nstate=3, nac_pair_map=[[0, 1]], soc_pair_map=[]))


class TestTrainingReport(unittest.TestCase):
    """The training record renders as the log sections of a result."""

    def test_both_sections_carry_the_rule_grammar(self):
        from pyrai3md.skills.nn_pes_skill.service.trainer import (
            format_training)
        text = format_training(training_config(), TRAINING_RECORD)
        self.assertIn('&nn training', text)
        self.assertIn('&nn validation mean absolute error', text)
        self.assertIn('-' * 79, text)

    def test_one_row_per_member_with_units_and_counts(self):
        from pyrai3md.skills.nn_pes_skill.service.trainer import (
            format_training)
        text = format_training(training_config(), TRAINING_RECORD)
        self.assertIn('energy_grad_v0', text)
        self.assertIn('energy_grad_v1', text)
        self.assertIn('Number of atoms:  4', text)
        self.assertIn('Number of state:  3', text)
        self.assertIn('Wall clock:       12.5 s', text)
        self.assertIn('eV/angstrom', text)
        self.assertIn('Training data:    /data/TRAIN.json', text)
        self.assertIn('Model folder:     /run/NN-TEST', text)
        self.assertIn('Fit log:          /run/NN-TEST/energy_grad/fitlog_0.txt',
                      text)
        self.assertIn('Number of NAC:    0', text)
        self.assertIn('&nn prediction max std', text)
        self.assertIn('0.001000', text)
        self.assertIn('--', text)
        self.assertNotIn('Summary file:', text)

    def test_an_untrained_channel_is_blank_not_zero(self):
        # A channel the folder never produced is not a channel fitted to
        # zero; printing 0.00000000 would claim a perfect fit.
        from pyrai3md.skills.nn_pes_skill.service.trainer import (
            format_training)
        text = format_training(training_config(), TRAINING_RECORD)
        rows = [line for line in text.splitlines()
                if line.strip().startswith('energy_grad_v')]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn('--', row)
            self.assertNotIn('0.00000000', row)


class MemberStub(StubContainer):
    """A stub whose members differ, so selection is observable."""

    def __init__(self, layout, members=2):
        super().__init__(layout)
        self.member_count = int(members)

    def _member_output(self, kind):
        if kind == 'mlp_eg':
            return [self._states(), self._gradients()]
        if kind == 'mlp_e':
            return self._states()
        if kind == 'mlp_g2':
            return self._gradients()
        return self._couplings()

    def call_with_members(self, batch):
        means, spreads = self.call(batch)
        members = {}
        for name, kind in self.layout.items():
            if kind == 'mlp_e' and name == 'soc':
                base = np.arange(1.0, NSOC + 1.0)[None]
            else:
                base = self._member_output(kind)
            if isinstance(base, list):
                members[name] = [
                    (base[0] + 0.1 * index, base[1] + 0.01 * index)
                    for index in range(self.member_count)]
            else:
                members[name] = [
                    base + 0.1 * index for index in range(self.member_count)]
        return means, spreads, members


class TestPredictorSelection(unittest.TestCase):

    def test_mean_mode_writes_none_alternatives(self):
        config = make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=2)
        results, _spread = evaluate(
            config, COORDS, container=MemberStub({'energy_grad': 'mlp_eg'}))
        self.assertEqual(results['predict_model_id'], 0)
        self.assertEqual(results['n_models'], 2)
        self.assertIsNone(results['alternative_energy'])
        self.assertIsNone(results['alternative_gradient'])
        self.assertIsNone(results['alternative_model_ids'])
        self.assertEqual(list(results['energy']), [1.0, 2.0, 3.0])

    def test_the_selected_member_is_the_prediction(self):
        config = make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=2,
                             nn_predict_mode=1)
        results, _spread = evaluate(
            config, COORDS, container=MemberStub({'energy_grad': 'mlp_eg'}))
        self.assertEqual(results['predict_model_id'], 1)
        self.assertEqual(list(results['energy']), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            results['alternative_energy'], [[1.1, 2.1, 3.1]])
        self.assertEqual(results['alternative_model_ids'], [2])
        self.assertEqual(
            np.shape(results['alternative_gradient']),
            (1, NSTATE, NATOM, 3))

    def test_the_second_member_can_be_the_primary(self):
        config = make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=2,
                             nn_predict_mode=2)
        results, _spread = evaluate(
            config, COORDS, container=MemberStub({'energy_grad': 'mlp_eg'}))
        self.assertEqual(list(results['energy']), [1.1, 2.1, 3.1])
        np.testing.assert_allclose(
            results['alternative_energy'], [[1.0, 2.0, 3.0]])
        self.assertEqual(results['alternative_model_ids'], [1])

    def test_three_members_keep_every_other_member(self):
        config = make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=3,
                             nn_predict_mode=3)
        results, _spread = evaluate(
            config, COORDS,
            container=MemberStub({'energy_grad': 'mlp_eg'}, members=3))
        self.assertEqual(results['alternative_model_ids'], [1, 2])
        self.assertEqual(np.shape(results['alternative_energy']), (2, NSTATE))

    def test_a_coupling_channel_gets_an_alternative(self):
        config = make_config('mlp_nac2_nac', 'mlp_nac2', nstate=3,
                             coupling=[[0, 1], [0, 2], [1, 2]],
                             mult_nn=2, nn_predict_mode=1)
        results, _spread = evaluate(
            config, COORDS, container=MemberStub({'nac': 'mlp_nac2'}))
        self.assertEqual(np.shape(results['alternative_nac']),
                         (1, 3, NATOM, 3))
        self.assertEqual(results['alternative_model_ids'], [2])

    def test_a_spin_orbit_channel_gets_an_alternative(self):
        config = make_config('mlp_e_soc', 'mlp_e', nsoc=NSOC,
                             soc_pair_map=[[0, 2], [1, 2]],
                             mult_nn=2, nn_predict_mode=1)
        results, _spread = evaluate(
            config, COORDS, container=MemberStub({'soc': 'mlp_e'}))
        self.assertEqual(np.shape(results['alternative_soc']),
                         (1, NSOC, 1))
        self.assertEqual(results['alternative_model_ids'], [2])

    def test_a_mode_outside_the_ensemble_is_refused(self):
        with self.assertRaises(ValueError):
            make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=2,
                        nn_predict_mode=3)

    def test_a_single_member_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=1,
                        nn_predict_mode=1)

    def test_a_folder_with_one_loaded_member_refuses_mode_one(self):
        from pyrai3md.skills.nn_pes_skill.service.calculator import (
            check_predict_mode)

        class Container:
            _addNN = 1

        config = make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=2,
                             nn_predict_mode=1)
        with self.assertRaises(ValueError):
            check_predict_mode(config, Container())

    def test_mean_mode_does_not_require_a_comparison_member(self):
        from pyrai3md.skills.nn_pes_skill.service.calculator import (
            check_predict_mode)

        class Container:
            _addNN = 1

        config = make_config('mlp_eg_gradient', 'mlp_eg', mult_nn=1)
        check_predict_mode(config, Container())
