"""
======================================================================
This is a standalone machine learning force field skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the test-set evaluation: the record it returns, the files it
writes, and the unit chain both depend on.

These are function tests. The engine route -- a fit in subprocesses and a
compiled model serving a dynamics step -- is certified by the case files in
this directory (``00_mlp_eg_train.yaml`` through ``07_mlp_eg_predict_eval``),
which the qualification suite runs through the three entry points, and by
``test_train_predict.py``, whose subject is that route. What is under test
here is this package's own arithmetic: which channels a prediction record
carries, what each statistic means, how the raw comparison file is laid
out, which directory the products land in, and what a request without
labels or with the wrong profile is answered with.

Because the surface answering the test file is known analytically, every
expectation below is a number the test computes itself: a label that is the
surface plus one Hartree must come back as a mean absolute error of
``HARTREE_TO_EV``, and the gradient assertion distinguishes the per-Angstrom
axis from the per-Bohr one by three significant figures rather than by
tolerance. The geometries are chosen to be exact in float32, because the
serving path casts them there and a coordinate like 0.7 would put a 6e-8
rounding into a claim that is otherwise exact.

"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.skills.nn_pes_skill.schema.reference import (BOHR_TO_ANGSTROM,
                                                            HARTREE_TO_EV,
                                                            METRIC_ENERGY_UNIT,
                                                            METRIC_GRADIENT_UNIT)
from pyrai3md.skills.nn_pes_skill.service import predictor
from pyrai3md.skills.nn_pes_skill.tests._fake_engine import (
    FakeContainer, coupling_label_file, label_file)

# Every value here is a dyadic rational: exact in float32, so the numbers
# the test predicts and the numbers the pipeline reports differ only by
# float64 rounding.
FRAMES = np.array([[[0.5, 0.25, -1.0], [0.75, -0.125, 0.0]],
                   [[-0.5, 1.0, 0.125], [0.25, 0.5, -0.25]]])
ATOMS = ['C', 'H']
ENERGY_SHIFT = 1.0
GRADIENT_SHIFT = 0.5

# The disagreement the fake ensemble reports per component, in Hartree and
# Hartree per Angstrom.
SPREAD = 0.05


def _config(profile, pred_data, **over):
    from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
    from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
    from pyrai3md.skills.nn_pes_skill.schema.validate import validate

    cfg = {
        'profile': profile,
        'method': 'mlp_eg',
        'modeldir': over.pop('modeldir', './model'),
        'pred_data': pred_data,
        'mult_nn': over.pop('mult_nn', 2),
        'nstate': over.pop('nstate', 2),
        'nnac': over.pop('nnac', 0),
        'target_states': [0, 1],
        'atoms': ATOMS,
        'natom': len(ATOMS),
        'coordinates': [list(row) for row in FRAMES[0]],
        'scratch_path': over.pop('scratch_path'),
        'project': 'UNIT',
        'execution_mode': 'local',
    }
    if cfg['nnac']:
        cfg['coupling'] = [[0, 1], [0, 2], [1, 2]]
        cfg['nac_pair_map'] = [[0, 1], [0, 2], [1, 2]]
        cfg['target_nac_pair'] = [[0, 1], [0, 2], [1, 2]]
    cfg.update(over)
    return validate(normalize(build_config(cfg)))


class PredictionRecordCase(unittest.TestCase):
    """One evaluation of a known surface against known labels."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='nn_predict_unit_')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.labels = os.path.join(self.dir, 'labels.json')
        label_file(self.labels, FRAMES, ENERGY_SHIFT, GRADIENT_SHIFT)
        self.calcdir = os.path.join(self.dir, 'calc')
        os.makedirs(self.calcdir)

    def _run(self, profile='mlp_eg_gradient', container=None, mult_nn=2):
        config = _config(profile, self.labels,
                         modeldir=os.path.join(self.dir, 'model'),
                         scratch_path=os.path.join(self.dir, 'run'),
                         mult_nn=mult_nn)
        container = container or FakeContainer(nstate=2, natom=2,
                                               mult_nn=mult_nn)
        return config, predictor.run(config, container, self.calcdir)


class TestRecordShape(PredictionRecordCase):
    """Which channels are reported, in which unit."""

    def test_only_the_served_channels_are_reported(self):
        _config, record = self._run()
        self.assertEqual(['energy', 'gradient'],
                         sorted(record['predict_errors']))
        self.assertNotIn('nac', record['predict_errors'])
        self.assertNotIn('nac', record['predict_raw_files'])

    def test_the_units_are_the_declared_metric_units(self):
        _config, record = self._run()
        self.assertEqual({'energy': METRIC_ENERGY_UNIT,
                          'gradient': METRIC_GRADIENT_UNIT},
                         record['predict_metric_units'])

    def test_the_frame_count_is_the_test_set(self):
        _config, record = self._run()
        self.assertEqual(len(FRAMES), record['predict_frames'])

    def test_the_serving_call_is_one_per_frame(self):
        container = FakeContainer(nstate=2, natom=2)
        self._run(container=container)
        self.assertEqual(len(FRAMES), container.calls,
                         'a frame left unserved would quietly shrink the set')


class TestUnitChain(PredictionRecordCase):
    """The factors between a label file, the served value and the report."""

    def test_the_energy_error_is_the_shift_in_electronvolts(self):
        _config, record = self._run()
        values = record['predict_errors']['energy']
        self.assertAlmostEqual(ENERGY_SHIFT * HARTREE_TO_EV, values['mae'],
                               places=8)

    def test_the_gradient_error_is_per_angstrom_not_per_bohr(self):
        _config, record = self._run()
        values = record['predict_errors']['gradient']
        # The label file carries Hartree per Bohr, the served gradient
        # arrives on that axis too, and the metric prints eV per Angstrom.
        # Read on the wrong axis the number is off by 0.529177 squared --
        # three significant figures, so this assertion cannot pass by
        # rounding, and it cannot pass if one side is subtracted from the
        # other before the axes are unified.
        expected = GRADIENT_SHIFT * HARTREE_TO_EV
        wrong_axis = expected * BOHR_TO_ANGSTROM ** 2
        self.assertAlmostEqual(expected, values['mae'], places=6)
        self.assertGreater(abs(values['mae'] - wrong_axis),
                           0.3 * expected)

    def test_the_spread_is_the_disagreement_in_the_same_unit(self):
        _config, record = self._run()
        for channel in ('energy', 'gradient'):
            self.assertAlmostEqual(SPREAD * HARTREE_TO_EV,
                                   record['predict_errors'][channel]['spread_max'],
                                   places=6)

    def test_a_single_member_reports_no_spread_rather_than_no_line(self):
        # One network has nothing to disagree with: the channel stays in the
        # record at 0.0, which reads as "no evidence of trouble", not as a
        # missing measurement.
        _config, record = self._run(mult_nn=1)
        self.assertEqual(0.0, record['predict_errors']['energy']['spread_max'])


class TestStatistics(PredictionRecordCase):
    """The reduction of one channel's differences."""

    def test_the_statistics_are_ordered_as_their_definitions_imply(self):
        _config, record = self._run()
        for channel, values in record['predict_errors'].items():
            with self.subTest(channel=channel):
                self.assertLessEqual(values['mae'], values['rmse'])
                self.assertLessEqual(values['rmse'], values['max'])
                self.assertGreater(values['mae'], 0.0)

    def test_a_uniform_shift_makes_the_three_statistics_agree(self):
        # Every component of the energy label is off by the same amount, so
        # mean, root mean square and maximum are one number -- a check that
        # the three are reductions of the same array rather than three
        # unrelated ones.
        _config, record = self._run()
        values = record['predict_errors']['energy']
        self.assertAlmostEqual(values['mae'], values['rmse'], places=8)
        self.assertAlmostEqual(values['rmse'], values['max'], places=8)

    def test_the_frame_table_holds_one_row_per_frame(self):
        _config, record = self._run()
        with open(record['predict_summary']) as handle:
            rows = [line for line in handle.read().splitlines()[1:]
                    if line.strip()]
        self.assertEqual(len(FRAMES), len(rows))


class TestRawFiles(PredictionRecordCase):
    """The printed record and the file it was reduced from."""

    def test_the_record_is_the_raw_columns_reduced(self):
        _config, record = self._run()
        for channel, values in record['predict_errors'].items():
            path = record['predict_raw_files'][channel]
            block = np.loadtxt(path)
            width = block.shape[1] // 4
            reference, predicted = block[:, :width], block[:, width:2 * width]
            deviation, spread = block[:, 2 * width:3 * width], block[:, 3 * width:]
            with self.subTest(channel=channel):
                self.assertEqual(len(FRAMES), block.shape[0])
                # One unit for all four blocks: the deviation is the
                # difference of the first two, so a reader can rebuild the
                # table from the file alone.
                self.assertEqual(width, reference.shape[1])
                self.assertEqual(4 * width, block.shape[1])
                np.testing.assert_allclose(np.abs(reference - predicted),
                                           deviation, rtol=1e-9)
                self.assertAlmostEqual(values['mae'], float(deviation.mean()),
                                       places=6)
                self.assertAlmostEqual(values['max'], float(deviation.max()),
                                       places=6)
                self.assertAlmostEqual(values['spread_max'],
                                       float(spread.max()), places=6)

    def test_the_products_land_in_the_run_directory_not_the_calculation_one(self):
        # `keep_tmp` clears the calculation directory, and what a test-set
        # evaluation exists to leave behind is not TensorFlow scratch.
        config, record = self._run()
        run_dir = os.path.abspath(config.environment.scratch_path)
        for path in list(record['predict_raw_files'].values()) + [
                record['predict_summary']]:
            self.assertEqual(run_dir, os.path.dirname(os.path.abspath(path)))
            self.assertFalse(path.startswith(self.calcdir))
            self.assertTrue(os.path.isfile(path))

    def test_the_coupling_channel_is_compared_channel_by_channel(self):
        couplings = os.path.join(self.dir, 'nac_labels.json')
        shift = coupling_label_file(couplings, FRAMES, nnac=3)
        container = FakeContainer(nstate=3, natom=2, nnac=3, mult_nn=2)
        config = _config('mlp_nac2_nac', couplings,
                         modeldir=os.path.join(self.dir, 'model'),
                         scratch_path=os.path.join(self.dir, 'run2'),
                         method='mlp_nac2', nnac=3, nstate=3)
        record = predictor.run(config, container, self.calcdir)
        self.assertEqual(['nac'], sorted(record['predict_errors']))
        self.assertEqual(METRIC_GRADIENT_UNIT,
                         record['predict_metric_units']['nac'])
        self.assertAlmostEqual(shift * HARTREE_TO_EV,
                               record['predict_errors']['nac']['mae'],
                               places=6)
        # (npair, natom, 3) per frame, in the declared pair order.
        block = np.loadtxt(record['predict_raw_files']['nac'])
        self.assertEqual(3 * len(ATOMS) * 3, block.shape[1] // 4)


class TestRefusals(PredictionRecordCase):
    """What the evaluator refuses rather than reports as zero."""

    def test_an_evaluation_without_a_test_file_names_the_field(self):
        config = _config('mlp_eg_gradient', '',
                         scratch_path=os.path.join(self.dir, 'run-empty'))
        with self.assertRaises(ValueError) as caught:
            predictor.run(config, FakeContainer(), self.calcdir)
        self.assertIn('pred_data is empty', str(caught.exception))

    def test_a_served_channel_without_labels_is_refused(self):
        bare = os.path.join(self.dir, 'bare.json')
        with open(self.labels) as handle:
            raw = json.load(handle)
        raw['grad'] = []
        with open(bare, 'w') as handle:
            json.dump(raw, handle)
        config = _config('mlp_eg_gradient', bare,
                         modeldir=os.path.join(self.dir, 'model'),
                         scratch_path=os.path.join(self.dir, 'run'))
        with self.assertRaises(ValueError) as caught:
            predictor.run(config, FakeContainer(nstate=2, natom=2),
                          self.calcdir)
        self.assertIn('no', str(caught.exception))
        self.assertIn('reference value', str(caught.exception))

    def test_the_skill_schema_refuses_a_test_set_on_a_training_profile(self):
        with self.assertRaises(ValueError) as caught:
            _config('mlp_eg_train', self.labels, train_data=self.labels,
                    scratch_path=os.path.join(self.dir, 'run-train'))
        self.assertIn('training profile', str(caught.exception))


class TestPredictionLog(unittest.TestCase):
    """The skill's own copy of the report."""

    def test_the_sections_name_every_served_channel_and_no_other(self):
        directory = tempfile.mkdtemp(prefix='nn_predict_log_')
        self.addCleanup(shutil.rmtree, directory, True)
        labels = os.path.join(directory, 'labels.json')
        label_file(labels, FRAMES)
        config = _config('mlp_eg_gradient', labels,
                         modeldir=os.path.join(directory, 'model'),
                         scratch_path=os.path.join(directory, 'run'))
        record = predictor.run(config,
                               FakeContainer(nstate=2, natom=2), directory)
        text = predictor.format_prediction(config, record)
        self.assertIn('&nn prediction error analysis', text)
        self.assertIn('&nn prediction mean absolute error', text)
        self.assertIn('&nn prediction ensemble spread', text)
        body = text.split('&nn prediction mean absolute error')[1]
        self.assertIn('energy', body)
        self.assertIn('gradient', body)
        self.assertNotIn('nac', body,
                         'a channel this profile does not serve prints no '
                         'column')


class TestPredictionVocabulary(unittest.TestCase):
    """The field the request is made of, and the record's statistics."""

    def test_pred_data_is_in_both_default_vocabulary_places(self):
        # A field missing from DEFAULT is dropped by the config builder
        # before the backend ever sees it, so the request would be mute.
        from pyrai3md.skills.nn_pes_skill.schema.config import DEFAULT, Calculation

        self.assertIn('pred_data', DEFAULT)
        self.assertIn('pred_data', Calculation.model_fields)
        self.assertEqual('', DEFAULT['pred_data'])

    def test_the_record_carries_the_four_statistics_and_no_more(self):
        self.assertEqual(('mae', 'rmse', 'max', 'spread_max'),
                         predictor._STATISTICS)


if __name__ == '__main__':
    unittest.main()
