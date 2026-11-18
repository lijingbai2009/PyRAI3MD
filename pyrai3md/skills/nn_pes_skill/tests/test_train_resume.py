"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the training continuation: which container calls a request
makes, in which order, and what a refusal must name.

The order is the whole subject. `create` builds freshly initialized
networks and `save` writes their weights unconditionally, so a
continuation that created before it saved would put random weights over
the trained surface and then have the fit subprocess load those and
report, truthfully, that it loaded old weights. A test that only checked
the fit subprocess's own log would pass on exactly that bug, so these
tests assert the call sequence of the container instead.

No TensorFlow is needed: the container is a recorder, and the folder
products are files a real fit leaves behind.

"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from pyrai3md.skills.nn_pes_skill.service import trainer

_PROFILE = 'mlp_eg_train'
_FOLDER = 'energy_grad'
_PRODUCTS = ('hyper_v%d.json', 'weights_v%d.weights.h5', 'scaler_v%d.json',
             'fiterr_valid_v%d.npy')


class _Recorder:
    """A container that records the calls a preparation makes."""

    def __init__(self, folder_root, members=1, model_type='mlp_eg',
                 depth=2, nn_size=24):
        self.folder_root = folder_root
        self.calls = []
        self.members = members
        self._models_hyper = {}
        self._seed = {'model': {'depth': depth, 'nn_size': nn_size,
                                'atoms': 2, 'states': 1},
                      'training': {'initialize_weights': True},
                      'retraining': {'initialize_weights': False},
                      'general': {'model_type': model_type}}

    def create(self, hyp_dict):
        self.calls.append('create')
        for name, hyper in hyp_dict.items():
            entries = hyper if isinstance(hyper, list) else [hyper]
            self._models_hyper[name] = [dict(entry) for entry in entries]

    def load(self, name):
        self.calls.append('load')
        self._models_hyper[name] = [dict(self._seed)
                                    for _ in range(self.members)]

    def update(self, hyp_dict):
        self.calls.append('update')
        for name, hyper in hyp_dict.items():
            entries = hyper if isinstance(hyper, list) else [hyper]
            stored = self._models_hyper.setdefault(name, [])
            for index, entry in enumerate(entries):
                # The container merges every category but 'model'.
                merged = dict(stored[index]) if index < len(stored) else {}
                for key, value in entry.items():
                    if key != 'model':
                        merged[key] = value
                if index < len(stored):
                    stored[index] = merged
                else:
                    stored.append(merged)

    def save(self, name):
        self.calls.append('save')


DEFAULT_MODEL = {'depth': 2, 'nn_size': 24, 'atoms': 2, 'states': 1}


class _Hyper:
    """A stand-in for the hyperparameter expansion.

    The switch is read from the request's own role section, the way the
    real expansion reads it, because which category the fit subprocess
    consults is decided by `train_mode` and a stub that ignored the config
    would test nothing about that. The model block is fixed here, and the
    architecture test overrides it.
    """

    def __init__(self, members=1, initialize_weights=None, model=None):
        self.members = members
        self.initialize_weights = initialize_weights
        self.model = model

    def __call__(self, config, folder, model_type, natom, indices):
        from pyrai3md.skills.nn_pes_skill.schema.method import ROLE_SECTIONS

        # A folder is named by the role it serves; its hyperparameters are
        # spelled in the role section the trainer reads.
        section = getattr(config.calculation, ROLE_SECTIONS[folder])
        written = dict(section or {}).get('initialize_weights')
        if self.initialize_weights is not None:
            written = self.initialize_weights
        # The container's own defaults differ between the two categories
        # a fit may read: a new training initializes weights, a retraining
        # does not. A keyword the caller wrote applies to both, which is
        # what the expansion mirrors; a keyword they did not write leaves
        # each category at its default.
        training = True if written is None else written
        retraining = False if written is None else written
        model = self.model if self.model is not None else DEFAULT_MODEL
        return [{'model': dict(model),
                 'general': {'model_type': model_type},
                 'training': {'initialize_weights': training},
                 'retraining': {'initialize_weights': retraining}}
                for _ in range(self.members)]


class ContinuationCase(unittest.TestCase):
    """Shared plumbing: a real data file, a folder, and a patched fit."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='nn_resume_')
        self.data = os.path.join(self.dir, 'egs.json')
        self._write_data()
        self.model_root = os.path.join(self.dir, 'model')
        os.makedirs(self.model_root, exist_ok=True)
        patches = [
            mock.patch.object(trainer, 'ensemble_hyper', _Hyper()),
            mock.patch.object(trainer.data_io, 'write_to_folder'),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write_data(self):
        with open(self.data, 'w') as handle:
            json.dump({'natom': 2, 'nstate': 1,
                       'xyz': [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]],
                       'energy': [[-1.0], [-1.1]],
                       'grad': [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]]},
                      handle)

    def _config(self, mode='training', initialize_weights=None,
                model=None):
        from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
        from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
        from pyrai3md.skills.nn_pes_skill.schema.validate import validate

        raw = {
            'profile': _PROFILE, 'method': 'mlp_eg',
            'modeldir': self.model_root, 'train_data': self.data,
            'train_mode': mode, 'mult_nn': 1, 'nstate': 1, 'nnac': 0,
            'target_states': [0], 'nsoc': 0,
            'atoms': ['C', 'H'], 'natom': 2,
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            'eg': ({'initialize_weights': initialize_weights}
                   if initialize_weights is not None else {}),
        }
        if model:
            raw['eg'].update(model)
        return validate(normalize(build_config(raw)))

    def _fill_folder(self, members=1, skip=()):
        path = os.path.join(self.model_root, _FOLDER)
        os.makedirs(path, exist_ok=True)
        for index in range(members):
            for pattern in _PRODUCTS:
                if pattern % index in skip:
                    continue
                with open(os.path.join(path, pattern % index), 'w') as handle:
                    handle.write('{}')
        return path

    def _prepare(self, container, config):
        return trainer.targets_with_data(config, container, self.model_root)


class TestCallOrder(ContinuationCase):
    """Which container methods a request calls, in what order."""

    def test_a_fresh_folder_creates_then_saves(self):
        container = _Recorder(self.model_root)
        self._prepare(container, self._config())
        self.assertEqual(['create', 'save'], container.calls)

    def test_a_continuation_loads_before_anything_is_saved(self):
        self._fill_folder()
        container = _Recorder(self.model_root)
        self._prepare(container, self._config(mode='retraining',
                                              initialize_weights=False))
        self.assertEqual(['load', 'update', 'save'], container.calls)

    def test_a_continuation_never_creates(self):
        # create() would rebuild the networks from the folder's own
        # architecture and reset them; a continuation must not reach it.
        self._fill_folder()
        container = _Recorder(self.model_root)
        self._prepare(container, self._config(mode='retraining',
                                              initialize_weights=False))
        self.assertNotIn('create', container.calls)

    def test_the_switch_is_read_from_the_category_the_fit_subprocess_reads(
            self):
        # train_mode names the category the subprocess reads, so a request
        # whose chosen category keeps initialize_weights true fits a new
        # network: no load, no update.
        container = _Recorder(self.model_root)
        self._prepare(container, self._config(mode='retraining',
                                              initialize_weights=True))
        self.assertEqual(['create', 'save'], container.calls)

    def test_the_container_default_of_the_category_decides(self):
        # 'retraining' with no switch written anywhere: the container's own
        # retraining category does not initialize weights, so the request is
        # a continuation, and the folder is loaded rather than replaced.
        self._fill_folder()
        container = _Recorder(self.model_root)
        self._prepare(container, self._config(mode='retraining'))
        self.assertEqual(['load', 'update', 'save'], container.calls)

    def test_a_plain_training_request_with_the_switch_off_is_a_continuation(
            self):
        # initialize_weights is a role-section keyword, so writing it under
        # eg applies it to the category the chosen mode reads.
        self._fill_folder()
        container = _Recorder(self.model_root)
        self._prepare(container, self._config(mode='training',
                                              initialize_weights=False))
        self.assertEqual(['load', 'update', 'save'], container.calls)


class TestRefusals(ContinuationCase):
    """What a refused request has to name."""

    def test_training_over_a_trained_folder_is_refused_not_attempted(self):
        path = self._fill_folder()
        container = _Recorder(self.model_root)
        with self.assertRaises(FileExistsError) as caught:
            self._prepare(container, self._config())
        message = str(caught.exception)
        self.assertIn(self.model_root, message)
        self.assertIn('initialize_weights: false', message)
        self.assertIn('eg', message, 'the section to write the switch in')
        self.assertEqual([], container.calls,
                         'a refusal must not touch the folder at all')
        self.assertTrue(os.path.isdir(path))

    def test_a_continuation_without_weights_names_the_missing_product(self):
        self._fill_folder(members=1, skip=('weights_v0.weights.h5',))
        container = _Recorder(self.model_root)
        with self.assertRaises(FileNotFoundError) as caught:
            self._prepare(container, self._config(mode='retraining',
                                                  initialize_weights=False))
        self.assertIn('weights_v0.weights.h5', str(caught.exception))
        self.assertEqual([], container.calls)

    def test_an_absent_folder_names_the_path(self):
        container = _Recorder(self.model_root)
        with self.assertRaises(FileNotFoundError) as caught:
            self._prepare(container, self._config(mode='retraining',
                                                  initialize_weights=False))
        self.assertIn(_FOLDER, str(caught.exception))
        self.assertEqual([], container.calls)

    def test_an_architecture_change_is_named_not_silently_dropped(self):
        # update() excludes the 'model' category, so a scan that varied
        # depth while continuing would run the old network and report the
        # new number. The refusal names the key and both values.
        self._fill_folder()
        container = _Recorder(self.model_root)
        with mock.patch.object(trainer, 'ensemble_hyper',
                               _Hyper(model={'depth': 5, 'nn_size': 24,
                                             'atoms': 2, 'states': 1})):
            with self.assertRaises(ValueError) as caught:
                self._prepare(container, self._config(mode='retraining',
                                                      initialize_weights=False))
        message = str(caught.exception)
        self.assertIn('model parameters are not updated', message)
        self.assertIn('depth', message)
        self.assertIn('5', message)
        self.assertIn('2', message)
        self.assertEqual(['load'], container.calls,
                         'the folder must not be saved over')

    def test_a_matching_architecture_is_allowed(self):
        self._fill_folder()
        container = _Recorder(self.model_root)
        with mock.patch.object(trainer, 'ensemble_hyper', _Hyper()):
            self._prepare(container, self._config(mode='retraining',
                                                  initialize_weights=False))
        self.assertEqual(['load', 'update', 'save'], container.calls)


class TestHelpers(unittest.TestCase):
    """The two predicates the preparation reads."""

    def test_resumable_needs_every_product_of_every_member(self):
        root = tempfile.mkdtemp(prefix='nn_products_')
        self.addCleanup(shutil.rmtree, root, True)
        path = os.path.join(root, _FOLDER)
        os.makedirs(path)

        self.assertIn('does not exist',
                      trainer.resumable(os.path.join(root, 'elsewhere'),
                                        [_FOLDER], 1)[1])
        self.assertIn('holds no hyper_v0.json',
                      trainer.resumable(root, [_FOLDER], 1)[1])
        open(os.path.join(path, 'hyper_v0.json'), 'w').close()
        open(os.path.join(path, 'hyper_v1.json'), 'w').close()
        self.assertIn('weights_v0', trainer.resumable(root, [_FOLDER], 2)[1])
        for name in ('weights_v0.weights.h5', 'weights_v1.weights.h5',
                     'scaler_v0.json', 'scaler_v1.json',
                     'fiterr_valid_v0.npy', 'fiterr_valid_v1.npy'):
            open(os.path.join(path, name), 'w').close()
        self.assertEqual((True, ''), trainer.resumable(root, [_FOLDER], 2))

    def test_wants_continuation_reads_the_category_the_mode_names(self):
        config = mock.Mock()
        config.calculation.train_mode = 'retraining'
        members = {'training': {'initialize_weights': True},
                   'retraining': {'initialize_weights': False}}
        self.assertTrue(trainer.wants_continuation(config, members))
        config.calculation.train_mode = 'training'
        self.assertFalse(trainer.wants_continuation(config, members))

    def test_a_missing_switch_defaults_to_a_new_fit(self):
        config = mock.Mock()
        config.calculation.train_mode = 'training'
        self.assertFalse(trainer.wants_continuation(config, {'training': {}}))

    def test_model_shape_changes_names_only_the_shared_keys(self):
        container = mock.Mock()
        container._models_hyper = {_FOLDER: [
            {'model': {'depth': 2, 'nn_size': 24, 'atoms': 2}}]}
        same = [{'model': {'depth': 2, 'nn_size': 24, 'atoms': 2,
                           'states': 1}}]
        self.assertEqual([], trainer.model_shape_changes(container, _FOLDER,
                                                         same))
        wider = [{'model': {'depth': 2, 'nn_size': 96, 'atoms': 2}}]
        changed = trainer.model_shape_changes(container, _FOLDER, wider)
        self.assertEqual(1, len(changed))
        self.assertIn('nn_size', changed[0])
        self.assertIn('96', changed[0])
        self.assertIn('24', changed[0])


if __name__ == '__main__':
    unittest.main()
