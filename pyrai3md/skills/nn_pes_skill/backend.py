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

This module provides the backend class for the nn_pes skill.

The backend is the public interface of the skill and is responsible for:
- Accepting configuration from multiple sources
- Initializing the runtime once (load the trained ensembles, or train
  them when the profile asks for it)
- Running property evaluations from coordinates
- Return standardized results

Core principle:
Open the ensembles once during initialization, then serve each dynamics
step with one compiled forward pass per ensemble member. A step measured
1.43 ms against 1274 ms for the uncompiled container path.

"""

import os
import shutil
import weakref

import numpy as np
from pydantic import BaseModel

from .schema.adapter import build_config, read_json
from .schema.normalize import normalize
from .schema.validate import validate
from .schema.method import resolve_profile
from .service.runner import (get_serving, validate_environment,
                             import_container, results_file_name,
                             configure_threads,
                             configure_tensorflow_threads)
from .service.calculator import open_container
from .service.deviation import deviation_metrics
from .service.utils import build_calcdir, flatten_extended_config, to_list


class NNPESBackend:
    """Main backend class for neural network potential calculations.

    Initializes once, then serves lightweight calculate() calls.

    Supports initialization from:
    1. Dictionary: task config carrying 'skill_config'
    2. JSON path: path to a JSON config file
    3. BaseModel: pydantic model instance (standalone flat config or
       a prebuilt extended PyRAIConfig from PyRAIState)

    Parameters
    ----------
    config : dict or str or BaseModel
        Configuration source.
    calcdir : str, optional
        Explicit calculation directory path.
    execution_id : str, optional
        Execution namespace identifier assigned by the Calculator
        (e.g. 'tmp_NN_PES' (single task) or 'tmp_high_qm_task1_NNPES'). Used
        directly as the calculation directory name; empty string falls
        back to the standalone tmp_NNPES naming.
    """

    def __init__(self, config, calcdir=None, execution_id=''):

        # Resolve the raw skill config from the input source.
        # Task config dict mode: dict with 'skill_config' from framework
        #   task queue (new design, independent per task).
        # Legacy state mode: PyRAIConfig from PyRAIState with _skill_config
        #   attached by the framework (defaults filled, profile resolved).
        # Standalone mode: dict, JSON path or flat config model.
        if isinstance(config, dict) and 'skill_config' in config:
            self._task_config = config
            raw_config = dict(config['skill_config'])
        elif isinstance(config, BaseModel) and 'control' in type(config).model_fields:
            # Legacy: extended config from PyRAIState with _skill_config
            self._task_config = None
            self._pyrai_config = config
            skill_config = getattr(config, '_skill_config', None)
            raw_config = (dict(skill_config) if skill_config is not None
                          else flatten_extended_config(config))
        else:
            # Standalone mode: dict, JSON path or flat config model
            self._task_config = None
            self._pyrai_config = None
            raw_config = config
            if isinstance(config, str):
                raw_config = read_json(config)

        # Derive profile from method + jobtype if not explicitly set
        if isinstance(raw_config, dict) and not raw_config.get('profile'):
            method = raw_config.get('method', '')
            jobtype = raw_config.get('jobtype', 'energy')
            if method:
                raw_config = dict(raw_config)
                raw_config['profile'] = resolve_profile(method, jobtype)

        # Run the skill schema pipeline (build -> normalize -> validate)
        # in both modes. The result is a nested SkillConfig held as
        # self.config throughout the backend lifetime.
        input_config = build_config(raw_config)
        normalized_config = normalize(input_config)
        self.config = validate(normalized_config)

        # The training-frame prediction spread is a post-fit diagnostic.
        # Only the training and prediction profiles run it; a serving
        # profile ignores the switch even if the input carried it.
        profile = str(self.config.calculation.profile)
        if not (profile.endswith('_train') or profile.endswith('_predict')):
            self.config.calculation.predict_std = 0

        # Ensure scratch_path exists (default to cwd if not set)
        scratch_path = self.config.environment.scratch_path or os.getcwd()
        scratch_path = os.path.expanduser(scratch_path)
        os.makedirs(scratch_path, exist_ok=True)
        self.config.environment.scratch_path = scratch_path

        # The serving thread budget is published before TensorFlow is
        # touched, so the runtime reads it at its first parallel region
        # (#306). Training children receive train_thread separately.
        configure_threads(self.config)

        # The runtime check happens before anything TensorFlow is touched:
        # a machine without the optional extra gets a named error here, not
        # a silent absence during discovery.
        validate_environment(self.config)
        configure_tensorflow_threads(self.config)

        # Resolve calculation directory under the scratch path
        if calcdir:
            self.calcdir = calcdir
        else:
            self.calcdir = build_calcdir(
                execution_id=execution_id, base_path=scratch_path,
                project=self.config.calculation.project,
            )
        os.makedirs(self.calcdir, exist_ok=True)

        # keep_tmp is applied when this backend is gone, not per step:
        # the calculation directory holds the files a run accumulates
        # across steps ({project}_{profile}.log and the uncertainty
        # dump), which every writer opens in append mode without
        # recreating the directory. Clearing it inside calculate() would
        # keep only the last step and then fail silently, so the teardown
        # follows the lifecycle LAMMPS' resident service uses (#307).
        self._finalizer = weakref.finalize(
            self, _discard_calcdir, self.calcdir,
            int(self.config.environment.keep_tmp))
        self._finalizer.atexit = True

        # The thread count is published before TensorFlow is
        # touched, so the runtime reads it at its first parallel
        # region (#306).
        self._serve = get_serving()
        self.container = None
        self.last_uncertainty = {}

        if self.config.calculation.profile.endswith('_train'):
            # A training run creates the folder it will write into; there
            # is nothing to load yet.
            NeuralNetPes = import_container()
            folder = os.path.abspath(os.path.expanduser(
                self.config.calculation.modeldir))
            os.makedirs(folder, exist_ok=True)
            self.container = NeuralNetPes(folder,
                                          mult_nn=self.config.calculation.mult_nn)
        else:
            self.container = open_container(self.config, import_container())

    def _trained_atoms(self):
        """Return the atom count the loaded networks were trained on.

        Returns
        -------
        int or None
        None when no ensemble is loaded yet (a training run).
        """
        hyper_map = getattr(self.container, '_models_hyper', None) or {}
        for name in sorted(hyper_map):
            members = hyper_map[name]
            if members:
                return int(members[0]['model']['atoms'])
        return None

    def check_thread_safe(self, runner, execution_mode, profile_semantics):
        """Decline a thread pool only for a training term.

        The training arm reconfigures TensorFlow for the whole process
        (nn_pes_src/device.py:48-52 sets the visible devices and the
        memory growth through service/trainer), so two terms that train
        at once would move each other's GPU. That is read from this
        backend's own profile, not from the contract: the training run
        is the one this skill spells '*_train'.

        A serving term is allowed, and #302 measured it rather than
        assuming it: two serving terms through one TensorFlow runtime in
        one thread pool reproduce the serial energy and gradient bitwise
        (skills/nn_pes_skill/tests/test_train_predict.py). The claim is
        that configuration -- an energy-gradient ensemble, the compiled
        serving path -- not NAC/SOC heads or a distributed TensorFlow
        build, neither of which the test covers.
        """
        if str(self.config.calculation.profile).endswith('_train'):
            raise ValueError(
                "nn_pes_skill trains one profile per process: the "
                "training arm sets the TensorFlow visible devices for the "
                "whole process. Run the training profiles serially or with "
                "remotetask (profile '%s' asked for runner='%s', "
                "execution_mode='%s', profile_semantics='%s')."
                % (self.config.calculation.profile, runner, execution_mode,
                   profile_semantics))

    def _store_results(self, results):
        """Write formatted results into the task config results dict.

        Results dict keys follow the Properties schema field names
        ('energy', 'gradient', 'nac') so the framework composite can
        aggregate them.

        Parameters
        ----------
        results : dict
            Formatted calculation results.
        """
        if self._task_config is not None:
            self._task_config['results'] = {
                key: (value.tolist() if hasattr(value, 'tolist') else value)
                for key, value in results.items()
            }

    def close(self):
        """Release the calculation directory according to keep_tmp.

        The finalizer normally runs when the backend is collected; the MD
        workflow also calls this at the end of a trajectory so a worker
        process reused for the next trajectory does not keep the previous
        one's temporary files alive.
        """
        finalizer = getattr(self, '_finalizer', None)
        if finalizer is not None and finalizer.alive:
            finalizer()
            finalizer.detach()
        self._finalizer = None

    def _log_path(self):
        """Return the path of this skill's own run log.

        One file per project and profile, appended to per step, so a
        remote run leaves its numbers beside the calculation directory
        even when the framework report stays on the parent side.
        """
        return os.path.join(
            self.calcdir,
            '%s_%s.log' % (self.config.calculation.project,
                           self.config.calculation.profile))

    def _write_training_log(self, record):
        """Record one training run in the skill's own log.

        The framework report prints the same numbers from the returned
        record; this is the copy the skill keeps, written with the same
        section grammar as a served step.

        Parameters
        ----------
        record : dict
            The training record from the service layer.
        """
        from .service.trainer import format_training
        try:
            with open(self._log_path(), 'a') as handle:
                handle.write(format_training(self.config, record))
        except OSError:
            pass

    def _write_prediction_log(self, record):
        """Record one test-set evaluation in the skill's own log.

        The framework report prints the same table from the returned
        record; this is the copy the skill keeps, so a remote run leaves
        its error numbers beside the folder that produced them.

        Parameters
        ----------
        record : dict
            The prediction record from the service layer.
        """
        from .service.predictor import format_prediction
        try:
            with open(self._log_path(), 'a') as handle:
                handle.write(format_prediction(self.config, record))
        except OSError:
            pass

    def _write_step_log(self, results, metrics, natom):
        """Record one served step's numbers in the skill's own log.

        The framework's report prints what the backend returned, and this
        file is the skill's copy of the same numbers: the properties, each
        with the unit it is reported in, and the member-to-member standard
        deviation behind them. It is appended per step, so a dynamics run
        leaves one line set per accepted geometry beside its uncertainty
        file. Writing it never fails a step: a diagnostic file that can
        stop a trajectory would cost more than the line it loses.

        Parameters
        ----------
        results : dict
            Formatted property values of this step.
        metrics : dict
            Deviation key to its scalar, from the ensemble spread.
        natom : int
            Number of atoms in the served geometry.
        """
        from .service.deviation import format_step
        path = self._log_path()
        try:
            with open(path, 'a') as handle:
                handle.write(format_step(
                    self.config.calculation.profile,
                    self.config.system.atoms, results, metrics, natom=natom))
        except OSError:
            pass

    def _write_uncertainty(self, uncertainty):
        """Record the ensemble spread of one step.

        The spread is a diagnostic of the ensemble, not a framework
        property channel, so it goes to a file in the calculation
        directory instead of the results dict.

        Parameters
        ----------
        uncertainty : dict
            Spread arrays keyed like the results.
        """
        self.last_uncertainty = uncertainty
        path = os.path.join(
            self.calcdir,
            '%s_%s.uncertainty.txt' % (self.config.calculation.project,
                                       self.config.calculation.profile))
        try:
            with open(path, 'a') as handle:
                handle.write('%s\n' % ' '.join(
                    '%s=%.6f' % (key, float(np.max(np.abs(value))))
                    for key, value in sorted(uncertainty.items())))
        except OSError:
            # A diagnostic file must never fail a dynamics step
            pass

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        """Execute a lightweight property step, or one training run.

        Serving only updates the dynamic information (geometry, target
        states); the ensembles were loaded during initialization.

        Parameters
        ----------
        atoms : list of str
            Atom symbols.
        coordinates : list of list of float
            Atomic coordinates in Angstrom.
        point_charges : list of list of float, optional
            Background point charges (accepted for interface
            compatibility; a neural network potential has no embedding).
        target_states : list of int, optional
            Per-step target states chosen by the workflow. None keeps the
            target_states fixed at initialization; otherwise overrides
            them for this step.
        cell : dict, optional
            Periodic-cell snapshot, accepted for interface
            compatibility. This is a molecular (non-periodic) backend
            and the skill validation rejects pbc, so cell is always
            ignored here.

        Returns
        -------
        dict
            Formatted calculation results.
        """
        # Update dynamic information via attribute access
        expected = self._trained_atoms()
        if expected is not None and len(atoms) != expected:
            raise ValueError(
                "The trained network covers %d atoms but a geometry of "
                "%d atoms was handed to calculate(); a potential cannot "
                "change its molecule between steps" % (expected, len(atoms)))
        self.config.system.atoms = atoms
        self.config.system.coordinates = coordinates
        self.config.system.natom = len(atoms)
        if target_states is not None:
            self.config.calculation.target_states = list(target_states)

        if self.config.calculation.profile.endswith('_train'):
            from .service.trainer import run as run_training_job
            record = run_training_job(self.config, self.container, self.calcdir)
            formatted = {'train_errors': record['train_errors'],
                         # The unit travels with the number: a validation
                         # error is meaningless without knowing whether it is
                         # scaled, Hartree or eV.
                         'train_metric_units': record['train_metric_units'],
                         'train_wall_seconds': record['train_wall_seconds'],
                         'train_arm': record['train_arm'],
                         'train_data': record['train_data'],
                         'train_modeldir': record['train_modeldir'],
                         'train_fitlogs': record['train_fitlogs'],
                         'train_prediction_std':
                             record['train_prediction_std'],
                         'train_summary': record['train_summary']}
            formatted.setdefault('fin_coord', [])
            formatted.setdefault('fin_velocity', [])
            self._store_results(formatted)
            self._write_training_log(record)
            return formatted

        if self.config.calculation.pred_data:
            # A serving request that carries a test set is an evaluation
            # of it: one pass over the file's frames against their labels,
            # reported as an error table. The field is the request, so no
            # second spelling of "predict" has to be kept in step with the
            # profile the framework already resolved.
            from .service.predictor import run as run_prediction
            record = run_prediction(self.config, self.container, self.calcdir)
            formatted = dict(record)
            formatted.setdefault('fin_coord', [])
            formatted.setdefault('fin_velocity', [])
            self._store_results(formatted)
            self._write_prediction_log(record)
            return formatted

        if point_charges is not None and len(point_charges) > 0:
            raise ValueError(
                "nn_pes does not accept point charges; a neural network "
                "potential is trained on nuclei only.")

        results, uncertainty = self._serve(self.config, self.calcdir,
                                           self.container)
        # The spread is the adaptive sampling signal and is always
        # computed; only the diagnostic files it feeds are optional.
        self.last_uncertainty = uncertainty
        write_step_log = int(self.config.environment.write_step_log)
        if write_step_log:
            self._write_uncertainty(uncertainty)

        # The ensemble's member-to-member spread is this skill's only
        # error bar, and the framework receives one number per computed
        # property: the largest spread over that property's components, in
        # the property's own unit.
        metrics = deviation_metrics(
            uncertainty,
            members=int(results.get('n_models')
                        or self.config.calculation.mult_nn))

        # Only the channels the profile's networks produce are reported:
        # an empty placeholder would look like a computed coupling to a
        # consumer that reads the key's presence (qdm does the same).
        formatted = to_list(results)
        formatted.update(metrics)
        if write_step_log:
            self._write_step_log(formatted, metrics, len(atoms))

        # Reserved return channels: this skill never relaxes the geometry
        # and never returns velocities, so both stay empty explicitly.
        formatted.setdefault('fin_coord', [])
        formatted.setdefault('fin_velocity', [])

        # Write results back into the config following the schema
        self._store_results(formatted)

        return formatted


def _discard_calcdir(calcdir, keep_tmp):
    """Remove one backend's calculation directory when keep_tmp is off.

    Reached through weakref.finalize, so it holds no reference to the
    backend: the directory is cleared once the run has dropped it, which
    is the only point at which a stepped serving run has really finished
    with it (#307).

    Parameters
    ----------
    calcdir : str
        Calculation directory of the backend that is going away.
    keep_tmp : int
        Resource-management setting: 0 clears the directory, 1 keeps it.
    """
    if keep_tmp == 0 and calcdir:
        shutil.rmtree(calcdir, ignore_errors=True)


def load_backend(config, calcdir=None, execution_id=''):
    """Construct a backend from a configuration source.

    Reserved compatibility wrapper: the framework resolves the class
    through ``skills.query.resolve_api``; no production caller uses this.

    Parameters
    ----------
    config : dict or str or BaseModel
        Configuration source.
    calcdir : str, optional
        Explicit calculation directory.
    execution_id : str, optional
        Execution namespace identifier.

    Returns
    -------
    NNPESBackend
        Initialized backend instance.
    """
    return NNPESBackend(config, calcdir=calcdir, execution_id=execution_id)
