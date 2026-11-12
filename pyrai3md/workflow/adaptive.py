"""
======================================================================
This module provides the AdaptiveSampling workflow: an iteration-based
adaptive-sampling loop.

One iteration is one directory. From iteration 2 on, the directory holds the
iteration's training data, the network trained from it, the MD trajectories
computed with that network, and the screening statistics. Iteration 1 is the
external starting point: the data and the network come from the input,
so its directory holds only the trajectories and the statistics.

Iteration order:
    1. save this iteration's training data (iteration >= 2)
    2. train the iteration's network from that file (iteration >= 2)
    3. run the MD trajectories with the iteration's network
    4. screen the retained history frames and write the statistics
    5. if nothing was selected or the iteration budget is spent: stop
    6. otherwise run the reference calculations; their labels are the
       next iteration's training data

The parent workflow never initializes a backend: every stage is a child
runtime materialized from the parent state, so each child owns its own
calculator. The MD stage keeps the full theory queue (PBC and multiscale
composites included), the reference stage keeps the full reference
queue, and the training stage keeps only the one trainable theory task.

Aug 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import shutil

import numpy as np

from ..data.geometry import continuous_coordinates
from ..kernel.logger import get_logger
from ..training_data import TrainingData, load as load_training_data
from .base import Workflow
from .grid.plan import skill_surface
from .grid.record import folder_complete
from .initcond import normalize_initcond, resolve_initial_conditions, write_init_xyz
from .records.adaptive import (
    TRAINING_UNITS, build_frame_dataset, energy_outlier_flags, frame_contract,
    labels_from_outcome, screen_iteration, screen_thresholds,
)
from .records.analysis import (
    MISSING, AnalysisFrame, analysis_active, analysis_plan,
    predictor_status, saturation, summarize_coverage, summarize_gap,
    summarize_pes, summarize_trace, trace_rows, trace_spec,
)
from .records.ml import declared_channels, ml_verdict

_UNIT_ALIASES = {
    'energy': ('hartree', 'ha', 'eh'),
    'gradient': ('hartree/bohr', 'ha/bohr', 'eh/bohr'),
    'nac': ('hartree/bohr', 'ha/bohr', 'eh/bohr'),
    'soc': ('cm-1', 'cm_1', '1/cm'),
}

_LOGGER = get_logger(__name__)


def _mae_text(value):
    """Print a validation MAE, or '--' when the fit reported none."""
    return '--' if value is None else '%.6f' % float(value)


def _child_log_name(child_state):
    """File stem of a child run log, when its product names it.

    A fit writes into ``NN-<title>-iter-<k>``; naming its log after the
    same stem keeps the two products readable as one pair, like the MD
    child whose log carries the trajectory's own title.

    Parameters
    ----------
    child_state : PyRAIState
        Materialized child state.

    Returns
    -------
    str or None
        ``os.path.basename(modeldir)`` for a training child; None
        otherwise (the run logger then uses the control title).
    """
    jobtype = str(getattr(child_state, 'jobtype', '') or '').split()
    if not jobtype or jobtype[0] != 'train':
        return None
    try:
        tasks = list(child_state.iter_tasks())
    except (AttributeError, TypeError) as error:
        _LOGGER.warning(
            'Child log name skipped: the child state gave no task queue '
            '(%s), so the run log keeps the control title', error)
        return None
    for _category, task in tasks:
        skill = ((task.get('config') or {}).get('skill_config')
                 if isinstance(task, dict) else None)
        if not isinstance(skill, dict):
            continue
        modeldir = str(skill.get('modeldir') or '')
        if modeldir:
            return os.path.basename(os.path.normpath(modeldir))
    return None


def _child_entry(child_state, runtime_idx=None):
    """Execute one materialized child runtime under its own run log.

    A dispatched adaptive stage is a framework task in every respect, so
    it leaves the same ``<title>.log`` an independent run does: start
    line, heading, the workflow's own blocks and the end line, all in
    the child's working directory. A batch of trajectories or reference
    points must not die as a whole because one child failed: the failure
    comes back as a marker that the parent records, exactly like
    GridSearch's child entry.
    """
    from . import execute, resolve_workflow
    from ..kernel.logger import RunLogger, stop_code_note

    workflow_cls = resolve_workflow(child_state)
    logger = None
    workdir = ''
    runtime = getattr(child_state, 'runtime', None)
    if runtime is not None:
        workdir = str(getattr(runtime, 'workdir', '') or '')
    if workdir:
        logger = RunLogger(child_state, log_dir=workdir,
                           log_name=_child_log_name(child_state))
        child_state.logger = logger
        logger.write_start_line()
        logger.write_heading(child_state)
        logger.log_workflow_title(workflow_cls, child_state.jobtype)

    try:
        outcome = execute(child_state, runtime_idx=runtime_idx)
    except Exception as error:
        code = int(getattr(runtime, 'stop_code', 0) or 0)
        if logger is not None:
            logger.finish('error: %s' % error, note=stop_code_note(code))
        return {'adaptive_status': 'error', 'stop_code': code,
                'adaptive_error': '%s: %s' % (type(error).__name__, error)}

    if logger is not None:
        kind = (getattr(workflow_cls, 'workflow_kind', 'single')
                if workflow_cls is not None else 'single')
        if kind != 'iterative':
            logger.log_result(child_state)
        status = 'completed'
        if (runtime is not None
                and getattr(runtime, 'status', '') == 'failed'):
            status = 'failed: %s' % getattr(
                runtime, 'error_message', '')
        logger.finish(status, note=stop_code_note(
            getattr(runtime, 'stop_code', 0) if runtime is not None else 0))
    return outcome


def _json_default(value):
    """JSON fallback for numpy scalars/arrays carried by screening rows."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('not JSON serializable: %r' % (type(value).__name__,))


def _is_error_outcome(outcome):
    return isinstance(outcome, dict) and outcome.get('adaptive_status') == 'error'


def _runtime_of(state):
    """Return the state's runtime record, creating the default one once."""
    runtime = getattr(state, 'runtime', None)
    if runtime is None:
        from ..data.runtime.state import RuntimeState
        runtime = RuntimeState()
        state.runtime = runtime
    return runtime


class AdaptiveSampling(Workflow):
    """Adaptive sampling workflow (iteration-based loop).

    Parameters
    ----------
    calculator : Calculator or None
        Parent calculator; it owns the parent state and is never used to
        evaluate a term in this process.
    maxiter : int
        Number of iterations.
    maxsample : int
        Maximum selected geometries per trajectory per iteration.
    load : int
        Iteration 1 uses the external data and network; only 1 is supported.
    record_step : int
        Screening window (last N history frames); 0 means 10.
    **kwargs
        Dispatcher parameters.
    """

    needs_backends = False
    name = 'adaptive'
    workflow_kind = 'sampling'
    multitask = False
    remotetask = False

    stages = ('md', 'base.theory', 'base.reference', 'train')

    stage_defaults = {
        'md': {},
        'base.theory': {'multitask': False},
        'base.reference': {},
        'train': {'multitask': False},
    }

    # Stage flag sources read by the planner: the phase keywords live in
    # &adaptive, not in &md / the skill sections.
    stage_flag_sources = {
        'md': ('adaptive', 'md'),
        'base.reference': ('adaptive', 'reference'),
    }
    # NN training needs no fan-out; it is one child runtime, remote or not.
    stage_remote_fields = {'train': ('adaptive', 'remote_train')}

    @classmethod
    def state_kwargs(cls, state):
        """Read the loop bounds from the ``&adaptive`` section."""
        adaptive = getattr(getattr(state, 'pmd', None), 'adaptive', None)
        if adaptive is None:
            return {}
        return {
            'maxiter': int(adaptive.maxiter),
            'maxsample': int(adaptive.maxsample),
            'load': int(adaptive.load),
            'record_step': int(adaptive.record_step),
        }

    def __init__(self, calculator=None, maxiter=1, maxsample=1, load=1,
                 record_step=10, **kwargs):
        super().__init__(name='adaptive', calculator=calculator,
                         maxiter=maxiter, **kwargs)
        self.maxsample = int(maxsample)
        self.load = int(load)
        self.record_step = int(record_step)
        self.logger = get_logger('pyrai3md.workflow.adaptive')
        self.executor_spec = ('pyrai3md.workflow.adaptive', '_child_entry')
        self.local_executor = _child_entry
        self._batch_root = os.getcwd()
        self._batch_name_format = 'traj_{idx}'
        self._root = os.getcwd()
        self._initial_conditions = []
        self._validation_history = []
        self._last_screen = None

    @classmethod
    def prepare(cls, state):
        """No pre-backend data work is required for this workflow."""
        return None

    @staticmethod
    def run_root(state):
        """Directory holding the adaptive run and its own run log.

        The project directory (the launch directory) holds the input and
        the user's assets; the adaptive work is the child directory named
        after the title, and its ``<title>.log`` sits there beside the
        iteration directories, not in the launch directory.

        Parameters
        ----------
        state : PyRAIState
            State carrying the control title.

        Returns
        -------
        str
            Absolute path of ``<title>.adaptive``.
        """
        title = str(getattr(state.pmd.control, 'title', '') or 'pyrai3md')
        return os.path.abspath('%s.adaptive' % title)

    # ---- iteration driver -------------------------------------------------

    def run(self):
        """Run the adaptive iterations and settle the parent runtime."""
        state = self.calculator.get_state()
        runtime = _runtime_of(state)
        title = str(getattr(state.pmd.control, 'title', '') or 'pyrai3md')
        root = self.run_root(state)
        self._root = root

        record = {
            'iterations': [],
            'converged': False,
            'reason': '',
            'root': root,
            'initial_dataset': '',
            'final_dataset': '',
            'final_modeldir': '',
            'n_iterations': 0,
            'keep_md': 1,
        }
        alignment = getattr(state, 'method_alignment', None)
        if alignment:
            record['method_alignment'] = dict(alignment)
            self._log_method_alignment(state, alignment)
        try:
            settings = self._settings(state)
            record['keep_md'] = int(settings['keep_md'])
            ml_slot, ml_task, surface = self._find_ml_task(state)
            ml_indices, ml_atoms = self._ml_region(state, ml_task)
            dataset = self._load_initial_dataset(ml_task)
            self._preflight(state, dataset, ml_task, surface, settings)
            self._initial_conditions = self._prepare_initial_conditions(state)
            record['initial_dataset'] = self._dataset_source(ml_task)
            record['final_dataset'] = self._dataset_source(ml_task)
            record['final_modeldir'] = self._configured_modeldir(ml_task)
        except Exception as error:
            record['reason'] = str(error)
            self._finish(state, record, failed=True)
            raise

        os.makedirs(root, exist_ok=True)
        if int(self.maxiter) <= 0:
            record['reason'] = 'maxiter is 0'
            return self._finish(state, record)

        selected_previous = None
        serving = None
        dataset_path = None
        external = True

        for iteration in range(1, int(self.maxiter) + 1):
            runtime.iteration = iteration
            iter_dir = os.path.join(root, 'iter-%d' % iteration)
            os.makedirs(iter_dir, exist_ok=True)
            labels, failures = [], []
            reference_ran = False
            reference_iteration = None
            trained = False
            received = 0
            verified = 0
            train_result = None
            gate = None
            kept_frames = 0
            base_nframe = 0
            new_positions = []

            if iteration == 1:
                # Iteration 1 is the external starting point: the configured
                # network and dataset, no reference and no training.
                serving = self._configured_modeldir(ml_task)
                self._require_servable(ml_task, surface, serving)
                dataset_path = None
                external = True
                record['final_dataset'] = self._dataset_source(ml_task)
                record['final_modeldir'] = serving
            else:
                external = False
                # The reference belongs to this iteration: it labels the
                # previous iteration's selected structures, and its result is
                # this iteration's training data. No iteration-number offset.
                reference_ran = bool(selected_previous)
                reference_iteration = iteration - 1
                received = len(selected_previous or [])
                base_nframe = int(dataset.nframe)
                if reference_ran:
                    contract = frame_contract(dataset)
                    labels, failures = self._reference_iteration(
                        state, iteration, iter_dir, selected_previous,
                        contract, settings, ml_indices)
                    verified = len(labels)
                    if labels:
                        new_data = build_frame_dataset(
                            labels,
                            ml_atoms
                            if contract.get('symbols') is not None else None,
                            unit=dict(dataset.unit or TRAINING_UNITS))
                        dataset.append(new_data)
                        new_positions = list(
                            range(base_nframe, int(dataset.nframe)))
                        new_path = os.path.join(
                            iter_dir, 'reference', 'new-samples.json')
                        self._save_dataset(new_data, new_path)
                if not labels:
                    record['reason'] = 'reference data collection failed'
                    record['converged'] = False
                    record['n_iterations'] = iteration
                    return self._finish(state, record, failed=True)

                dataset_path = os.path.join(
                    iter_dir, 'data_iter-%d_%d.json'
                    % (iteration, int(dataset.nframe)))
                self._save_dataset(dataset, dataset_path)
                modeldir = os.path.join(
                    iter_dir,
                    surface['query'].model_dir_name(
                        title, 'iter-%d' % iteration))
                trained, serving, train_note = self._train_iteration(
                    state, ml_slot, surface, dataset_path, modeldir,
                    iter_dir, settings, ml_indices)
                if not trained:
                    record['reason'] = train_note.get('reason') or (
                        'training failed')
                    record['final_dataset'] = dataset_path
                    return self._finish(state, record, failed=True)
                record['final_dataset'] = dataset_path
                record['final_modeldir'] = serving
                train_result = train_note.get('result')
                gate = self._validation_gate(
                    state, iteration, iter_dir, ml_slot, surface, title,
                    ml_indices, dataset, dataset_path, serving, train_result,
                    new_positions, base_nframe, settings)
                if gate['status'] == 'aborted':
                    # The run stops before MD: a surface that missed the
                    # target the user set must not propose new frames.
                    record['reason'] = gate['reason']
                    record['final_dataset'] = gate['dataset_path']
                    record['final_modeldir'] = gate['modeldir']
                    self._write_iteration(
                        state, iteration, iter_dir, external,
                        gate['dataset_path'], gate['modeldir'], None,
                        labels, failures, record,
                        reference_ran=reference_ran,
                        reference_iteration=reference_iteration,
                        trained=trained, received=received,
                        verified=verified, training=gate['training'],
                        ml_atoms=ml_atoms,
                        thresholds=self._screen_thresholds(settings),
                        gate=gate)
                    return self._finish(state, record, failed=True)
                dataset = gate['dataset']
                dataset_path = gate['dataset_path']
                serving = gate['modeldir']
                train_result = gate['training']
                kept_frames = int(gate['kept_frames'])
                record['final_dataset'] = dataset_path
                record['final_modeldir'] = serving

            outcomes = self._md_iteration(
                state, iteration, iter_dir, ml_slot, serving, settings)

            screen_settings = dict(settings)
            screen_settings['atoms'] = list(state.atoms)
            screen_settings['mdstep'] = int(state.pmd.dynamics.step)
            screen_settings['exploration'] = bool(
                settings.get('explore_udd') or settings.get('explore_gap'))
            screen = screen_iteration(outcomes, screen_settings)
            self._last_screen = screen

            # The same-spin order data is dynamics information, not optional
            # analysis: a trajectory that should carry it and does not is a
            # defect. Population is no longer screened here - the md guard
            # owns that judgment and returns its stop code, and the screen
            # only decides whether the stopping structure is collected.
            order_missing = list(screen.get('order_missing') or [])
            ntraj = int(screen.get('n_trajectory') or 0)
            for traj_id in order_missing:
                self.logger.warning(
                    "Adaptive iteration %d: trajectory %s produced no "
                    "energy/state-spin data for the same-spin order check",
                    iteration, traj_id)
            core_fatal = None
            if ntraj and len(order_missing) == ntraj:
                core_fatal = ('energy/state-order data missing from every '
                              'trajectory')

            if outcomes and all(_is_error_outcome(item) for item in outcomes):
                record['reason'] = 'all trajectories failed'
                self._write_iteration(
                    state, iteration, iter_dir, external, dataset_path,
                    serving, screen, labels, failures, record,
                    reference_ran=reference_ran,
                    reference_iteration=reference_iteration, trained=trained,
                    received=received, verified=verified,
                    training=train_result, ml_atoms=ml_atoms,
                    thresholds=self._screen_thresholds(settings))
                return self._finish(state, record, failed=True)

            # Analysis runs after screening and before the next iteration's
            # reference calculations. With every switch off it returns
            # (None, None, False) without touching a row or a file.
            if core_fatal:
                analysis, fatal, saturated = None, core_fatal, False
            else:
                analysis, fatal, saturated = self._analyze_iteration(
                    state, iteration, iter_dir, screen, outcomes, dataset,
                    settings, train_result, record=record,
                    new_labels=kept_frames)

            # The next iteration references these; the terminal iteration does not
            # get a consumer for its own selection.
            selected_previous = screen['selected']

            self._write_iteration(
                state, iteration, iter_dir, external, dataset_path,
                serving, screen, labels, failures, record,
                reference_ran=reference_ran,
                reference_iteration=reference_iteration, trained=trained,
                received=received, verified=verified,
                training=train_result, ml_atoms=ml_atoms,
                thresholds=self._screen_thresholds(settings),
                analysis=analysis, gate=gate)
            if int(settings['keep_md']) == 0:
                shutil.rmtree(os.path.join(iter_dir, 'md'),
                              ignore_errors=True)

            record['final_modeldir'] = serving
            record['n_iterations'] = iteration

            if fatal:
                record['converged'] = False
                record['reason'] = fatal
                return self._finish(state, record, failed=True)
            if not screen['selected']:
                record['converged'] = bool(screen['all_completed'])
                record['reason'] = 'no sample selected'
                break
            if saturated:
                record['converged'] = False
                record['reason'] = 'saturated'
                break
            if iteration >= int(self.maxiter):
                record['converged'] = False
                record['reason'] = 'maxiter exhausted'
                break

        if not record['reason']:
            record['reason'] = 'iterations finished'
        return self._finish(state, record)

    # ---- settings and preflight ---------------------------------------

    def _prepare_initial_conditions(self, state):
        """Resolve the starting points and materialize the warm-up files.

        A wigner/boltzmann run samples from ``<title>.freq.<format>`` once,
        before the iteration loop, and writes the same batch to ``<title>.init``
        (the legacy product) and ``<title>.init.xyz`` (the file
        ``initcond: read`` consumes later). The batch is the only source
        handed to the trajectory children, so they never re-sample.
        """
        batch = resolve_initial_conditions(state)
        dynamics = state.pmd.dynamics
        initcond = normalize_initcond(dynamics.initcond)
        if initcond in ('wigner', 'boltzmann'):
            title = str(getattr(state.pmd.control, 'title', '') or 'pyrai3md')
            comment = 'Init %s' % title
            for suffix in ('.init', '.init.xyz'):
                path = os.path.join(os.getcwd(), title + suffix)
                write_init_xyz(path, batch, list(state.atoms),
                               comment=comment)
            if getattr(state, 'logger', None) is not None:
                state.logger.write(
                    '\n  Warm-up: %d Wigner/Boltzmann initial condition(s) '
                    'written to %s.init and %s.init.xyz\n'
                    % (len(batch), title, title))
        return batch

    def _settings(self, state):
        """Read and validate the &adaptive settings of this run."""
        adaptive = getattr(getattr(state, 'pmd', None), 'adaptive', None)
        if adaptive is None:
            raise ValueError(
                "adaptive sampling needs the &adaptive section; the state "
                "carries none")
        for key in ('dynsample', 'refine', 'transfer'):
            value = int(getattr(adaptive, key, 0) or 0)
            if value != 0:
                raise ValueError(
                    "%s=%d is not ported in this adaptive build; set it to 0"
                    % (key, value))
        if int(self.load) != 1:
            raise ValueError(
                "this adaptive build starts every run from the data and "
                "network supplied in the input (load=1); train the initial "
                "network with jobtype 'train' first, then run adaptive")
        if self.maxsample < 0:
            raise ValueError(
                "maxsample must be >= 0, got %d" % self.maxsample)

        record_step = int(self.record_step)
        if record_step <= 0:
            record_step = 10
        maxima, minima = screen_thresholds(adaptive)
        analysis = analysis_plan(adaptive)
        dynamics = state.pmd.dynamics
        population_expected = (
            str(getattr(dynamics, 'sfhp', 'nosh')).lower() != 'nosh')
        channels = tuple(
            key for key, flag in (
                ('err_energy', 'sample_energy'),
                ('err_grad', 'sample_grad'),
                ('err_nac', 'sample_nac'),
                ('err_soc', 'sample_soc'),
            ) if int(getattr(adaptive, flag, 0) or 0))
        return {
            'maxsample': int(self.maxsample),
            'record_step': record_step,
            'mdstep': int(state.pmd.dynamics.step),
            # Validation energy MAE ceiling of a fit, eV; 0 disables the
            # training-data quality gate.
            'maxmae': float(getattr(adaptive, 'maxmae', 0.0) or 0.0),
            'maxima': maxima,
            'minima': minima,
            'channels': channels,
            'analysis': analysis,
            'analysis_active': analysis_active(analysis),
            'population_expected': population_expected,
            'predict_mode': int(getattr(adaptive, 'predict_mode', 0) or 0),
            'gap_threshold': float(getattr(adaptive, 'gap_threshold', 0.03)
                                   or 0.03),
            'explore_udd': int(getattr(adaptive, 'explore_udd', 0) or 0),
            'explore_gap': int(getattr(adaptive, 'explore_gap', 0) or 0),
            'udd_mode': str(getattr(adaptive, 'udd_mode', 'force')
                            or 'force'),
            'udd_amplitude': float(
                getattr(adaptive, 'udd_amplitude', 0.0) or 0.0),
            'udd_width': float(getattr(adaptive, 'udd_width', 0.0) or 0.0),
            'udd_start_iteration': int(
                getattr(adaptive, 'udd_start_iteration', 1) or 1),
            'gap_method': str(getattr(adaptive, 'gap_method', 'cv') or 'cv'),
            'gap_pair': list(getattr(adaptive, 'gap_pair', []) or []),
            'gap_height': float(getattr(adaptive, 'gap_height', 0.0) or 0.0),
            'gap_width': float(getattr(adaptive, 'gap_width', 0.0) or 0.0),
            'check_population': int(adaptive.check_population),
            'check_energy_order': int(adaptive.check_energy_order),
            'gap_tolerance': float(adaptive.gap_tolerance),
            'keep_md': int(adaptive.keep_md),
            'reference_remotetask': bool(adaptive.reference_remotetask),
            'md_multitask': self._pick(adaptive.md_multitask,
                                       adaptive.multitask),
            'md_remotetask': self._pick(adaptive.md_remotetask,
                                        adaptive.remotetask),
            'md_multipool': int(adaptive.md_multipool or 1),
            'reference_multitask': self._pick(adaptive.reference_multitask,
                                              adaptive.multitask),
            'reference_multipool': int(adaptive.reference_multipool or 1),
            'remote_train': bool(adaptive.remote_train),
        }

    @staticmethod
    def _pick(stage_value, generic_value):
        value = stage_value if stage_value is not None else generic_value
        return bool(value)

    def _find_ml_task(self, state):
        """Locate the one theory task that carries a trainable profile."""
        candidates = []
        for slot, task in enumerate(getattr(state, 'theory', []) or []):
            name = task.get('skill_name')
            if not name:
                continue
            try:
                surface = skill_surface(str(name))
            except ValueError:
                continue
            if surface.get('targets'):
                candidates.append((slot, task, surface))
        if not candidates:
            raise ValueError(
                "adaptive sampling needs one theory task whose skill "
                "declares a trainable profile; none of the queued theory "
                "tasks does")
        if len(candidates) > 1:
            raise ValueError(
                "adaptive sampling found %d trainable theory tasks (%s); "
                "it cannot decide which one the reference labels train"
                % (len(candidates),
                   ', '.join(str(task.get('skill_name'))
                             for _slot, task, _surface in candidates)))
        slot, task, surface = candidates[0]
        if int(task.get('contribution', 1)) != 1:
            raise ValueError(
                "the trainable theory task carries contribution=%s; the "
                "ML surface must be the +1 high-layer provider."
                % task.get('contribution'))
        if not task.get('provides_electronic'):
            raise ValueError(
                "the trainable theory task is not its region's electronic "
                "provider; adaptive sampling needs the high-layer surface")
        if task.get('region', 'all') not in ('all', 'model'):
            raise ValueError(
                "the trainable theory task serves region %r; adaptive "
                "sampling trains on the 'all' or 'model' surface"
                % task.get('region'))
        return slot, task, surface

    @staticmethod
    def _ml_region(state, task):
        """Indices and element labels of the trainable surface."""
        indices = [int(index) for index in (task.get('region_indices') or [])]
        if not indices:
            indices = list(range(len(state.atoms)))
        return indices, [state.atoms[index] for index in indices]

    @staticmethod
    def _json_plain(value):
        """Recursively turn numpy containers/scalars into JSON-safe values."""
        if isinstance(value, dict):
            return {key: AdaptiveSampling._json_plain(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AdaptiveSampling._json_plain(item) for item in value]
        if hasattr(value, 'tolist'):
            return value.tolist()
        if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except (TypeError, ValueError):
                return value
        return value

    @staticmethod
    def _plain_skill_geometry(state):
        """Make per-task skill configuration JSON-safe for slurm children.

        Planning stores geometry rows from the parent arrays, and the
        child may add numpy scalars. A skill's slurm runner serializes
        its config with `json.dump`, so every numpy value has to become
        a plain Python value before the child leaves the parent.
        """
        for _category, task in state.iter_tasks():
            config = task.get('config')
            skill = (config.get('skill_config')
                     if isinstance(config, dict) else None)
            if not isinstance(skill, dict):
                continue
            for key in list(skill):
                skill[key] = AdaptiveSampling._json_plain(skill[key])

    @staticmethod
    def _slice_child_system(child, indices):
        """Make one reference/training child a molecular region state.

        The reference and training children do not integrate the box: they
        compute the high-layer surface. Slicing their system makes the run
        log and the report describe those 6 atoms, and clearing the task
        region keeps Calculator from slicing the already-sliced arrays.
        """
        indices = [int(index) for index in indices]
        system = child.pmd.system
        geometry = continuous_coordinates(
            list(system.coordinates or []), system)
        def take(values, default=None):
            values = list(values or [])
            if values and len(values) == len(system.atoms):
                return [values[index] for index in indices]
            return default
        updates = {
            'atoms': [system.atoms[index] for index in indices],
            'natom': len(indices),
            'coordinates': [list(map(float, geometry[index]))
                            for index in indices],
            'velocities': take(system.velocities, [[0.0, 0.0, 0.0]
                                                   for _ in indices]),
            'mass': take(system.mass),
            'image': take(system.image, [[0, 0, 0] for _ in indices]),
            'pbc': [0, 0, 0],
            'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0],
            'lattice_vector': [[0.0, 0.0, 0.0] for _ in range(3)],
            'volume': 0.0,
            # A sliced child is the region itself: the original partition
            # indices and charge rows address atoms of the parent system,
            # not this one. Leaving them behind made a QMMM training child
            # index 1575+ into a six-atom array.
            'highlevel': [],
            'midlevel': [],
            'highref': [],
            'midref': [],
            'q_qm': [],
            'point_charges': [],
            'external_potential': [],
            'external_charges': '',
            'embedding': False,
            'read_charge': False,
        }
        child.update_system({key: value for key, value in updates.items()
                             if value is not None})
        # The region child owns no parent-sized electronic carrier: the
        # parent's per-center arrays, charge store and partition plan
        # address the box, and a training/reference child that carries
        # them indexes past its own atoms.
        child.electronic = []
        child.active_surface = None
        child.charge_store = None
        child.partition_plan = None
        for _category, task in child.iter_tasks():
            task['region'] = 'all'
            task['region_indices'] = []
            config = task.get('config')
            skill = (config.get('skill_config')
                     if isinstance(config, dict) else None)
            if isinstance(skill, dict):
                # The planning-time skill block carries the parent
                # partition too. It must address this sliced system, not
                # the box, or the skill's own preparation indexes past
                # the six atoms it was handed.
                for key in ('highlevel', 'midlevel', 'highref', 'midref',
                            'q_qm', 'point_charges', 'external_potential'):
                    if key in skill:
                        skill[key] = []
                for key in ('embedding', 'read_charge'):
                    if key in skill:
                        skill[key] = False
                if 'external_charges' in skill:
                    skill['external_charges'] = ''
        AdaptiveSampling._plain_skill_geometry(child)

    @staticmethod
    def _skill_config(task):
        config = task.get('config') or {}
        skill = config.get('skill_config')
        if not isinstance(skill, dict):
            raise ValueError(
                "adaptive sampling needs a flat skill_config on the "
                "trainable theory task; got %r" % (type(config).__name__,))
        return skill

    def _load_initial_dataset(self, task):
        """Load the unexpanded training data of the external network."""
        skill = self._skill_config(task)
        inline = skill.get('train_dataset')
        if isinstance(inline, dict) and inline:
            source = str(skill.get('train_data') or '<inline>')
            return TrainingData.from_mapping(inline, source=source)
        path = str(skill.get('train_data') or '')
        if not path:
            raise ValueError(
                "the trainable theory task carries neither train_data nor "
                "train_dataset; adaptive sampling needs the initial data")
        return load_training_data(path)

    @staticmethod
    def _dataset_source(task):
        skill = (task.get('config') or {}).get('skill_config') or {}
        return str(skill.get('train_data') or '<inline>')

    def _configured_modeldir(self, task):
        skill = self._skill_config(task)
        modeldir = str(skill.get('modeldir') or '')
        if not modeldir:
            raise ValueError(
                "the trainable theory task carries no modeldir; iteration 1 "
                "needs the network supplied in the input")
        return os.path.abspath(os.path.expanduser(modeldir))

    def _require_servable(self, task, surface, modeldir):
        skill = self._skill_config(task)
        context = {'query_module': surface['query_module'],
                   'members': int(skill.get('mult_nn') or 1)}
        try:
            complete = folder_complete(context, modeldir)
        except FileNotFoundError:
            complete = False
        if not complete:
            raise ValueError(
                "model directory %s holds no complete training; train the "
                "initial network with jobtype 'train' first" % modeldir)

    def _preflight(self, state, dataset, ml_task, surface, settings):
        """Refuse a run whose data cannot carry the labels it needs.

        Everything here reads an artifact that `schema/validate.py` cannot
        see: the loaded initial dataset (labels, units, atom order, cell),
        the trainable region and the reference queue. The checks that need
        only the config (which channels this build labels) live in
        `schema/normalize.reconcile_adaptive_properties`, where the final
        switch value is known. These stay for the two things a later
        failure cannot give: a message naming the artifact, and the run
        stopped before the first MD trajectory is paid for.
        """
        if dataset.energy is None:
            raise ValueError(
                "the initial training data carries no energy labels")
        if dataset.grad is None:
            raise ValueError(
                "the initial training data carries no gradient labels")
        adaptive = getattr(getattr(state, 'pmd', None), 'adaptive', None)
        if adaptive is not None:
            if int(adaptive.sample_nac) and dataset.nac is None:
                raise ValueError(
                    "adaptive sample_nac is on but the initial training "
                    "data carries no NAC labels")
            if int(adaptive.sample_soc) and dataset.soc is None:
                raise ValueError(
                    "adaptive sample_soc is on but the initial training "
                    "data carries no SOC labels")
        ml_indices, ml_atoms = self._ml_region(state, ml_task)
        if ml_atoms and dataset.natom != len(ml_atoms):
            raise ValueError(
                "the initial training data has %d atoms but the trainable "
                "region has %d" % (dataset.natom, len(ml_atoms)))
        if dataset.symbols is not None and ml_atoms:
            labels = [str(value) for value in dataset.symbols[0]]
            if labels != [str(value) for value in ml_atoms]:
                raise ValueError(
                    "the initial training data orders its atoms differently "
                    "from the trainable region")
        nstate = getattr(state, 'nstate', None)
        if nstate is not None and int(nstate) != dataset.nstate:
            raise ValueError(
                "the initial training data declares %d states but the "
                "system has %d" % (dataset.nstate, int(nstate)))
        for name in ('transition_dipole', 'transition_dipole_norm',
                     'osc_strength'):
            if getattr(dataset, name, None) is not None:
                raise ValueError(
                    "the initial training data carries '%s'; adaptive "
                    "sampling does not label that channel in this build"
                    % name)
        if dataset.charge is not None:
            raise ValueError(
                "the initial training data carries per-frame charge; "
                "adaptive sampling does not label that channel in this build")

        system = getattr(getattr(state, 'pmd', None), 'system', None)
        pbc = getattr(system, 'pbc', []) or []
        periodic = any(bool(value) for value in pbc)
        sliced = bool(ml_task.get('region_indices'))
        if periodic and not sliced and (dataset.cell is None
                                        or dataset.pbc is None):
            raise ValueError(
                "a periodic adaptive run needs cell/pbc in the initial "
                "training data")

        for channel, expected in _UNIT_ALIASES.items():
            declared = (dataset.unit or {}).get(channel)
            if declared is None:
                continue
            spelling = str(declared).strip().lower()
            if spelling not in expected:
                raise ValueError(
                    "the initial training data declares %s units as %r; "
                    "adaptive sampling labels frames as %s"
                    % (channel, declared, expected[0]))

        reference_tasks = [task for category, task in state.iter_tasks()
                           if category == 'reference']
        if not reference_tasks:
            raise ValueError(
                "adaptive sampling needs at least one reference task to "
                "label the selected structures")
        reference_caps = set()
        capability_seen = False
        plan = next(
            (task.get('label_plan') for task in reference_tasks
             if task.get('label_plan')), None)
        label_ids = (set(plan.get('source_task_ids') or [])
                     if plan and plan.get('source') == 'layer' else set())
        check_tasks = ([task for task in reference_tasks
                        if task.get('task_id') in label_ids]
                       if label_ids else reference_tasks)
        for task in check_tasks:
            caps = task.get('capabilities')
            if caps:
                capability_seen = True
                reference_caps.update(caps)
        if capability_seen:
            if dataset.nac is not None and 'nac' not in reference_caps:
                raise ValueError(
                    "the initial data carries NAC labels but the reference "
                    "queue does not plan the 'nac' capability")
            if dataset.soc is not None and 'soc' not in reference_caps:
                raise ValueError(
                    "the initial data carries SOC labels but the reference "
                    "queue does not plan the 'soc' capability")

        adaptive = getattr(getattr(state, 'pmd', None), 'adaptive', None)
        if adaptive is not None:
            predict_mode = int(getattr(adaptive, 'predict_mode', 0) or 0)
            explore_udd = int(getattr(adaptive, 'explore_udd', 0) or 0)
            explore_gap = int(getattr(adaptive, 'explore_gap', 0) or 0)
            constraints = getattr(
                getattr(state.pmd, 'system', None), 'constraints', None)
            metadynamics = list(
                getattr(constraints, 'metadynamics', None) or [])
            if (explore_udd or explore_gap) and metadynamics:
                raise ValueError(
                    "explore_udd/explore_gap cannot be combined with a "
                    "&system_constraint metadynamics bias: both add a "
                    "propagation potential")
            members = int(self._skill_config(ml_task).get('mult_nn') or 1)
            if predict_mode > members:
                raise ValueError(
                    "predict_mode=%d names a member outside the %d-member "
                    "ensemble" % (predict_mode, members))
            if predict_mode > 0 and members < 2:
                raise ValueError(
                    "predict_mode=%d needs at least 2 ensemble members; the "
                    "serving skill declares mult_nn=%d"
                    % (predict_mode, members))
            if explore_udd and predict_mode == 0:
                raise ValueError(
                    "explore_udd needs predict_mode>0: the UDD covariance "
                    "force is built from the primary prediction and the "
                    "alternative members")
            if explore_udd and int(getattr(state, 'n_center', 1) or 1) > 1:
                raise ValueError(
                    "explore_udd is limited to single-center adaptive runs")
            if (explore_udd and explore_gap
                    and str(getattr(adaptive, 'gap_method', 'cv')).lower()
                    == 'paper'):
                raise ValueError(
                    "gap_method=paper replaces the propagation force and "
                    "cannot be combined with explore_udd")
            if explore_gap and int(dataset.nstate) < 2:
                raise ValueError(
                    "explore_gap needs at least two electronic states")
            if (explore_gap
                    and str(getattr(adaptive, 'gap_method', 'cv')).lower()
                    == 'paper'
                    and str(getattr(getattr(state.pmd, 'dynamics', None),
                                    'sfhp', 'nosh')).lower() != 'nosh'):
                raise ValueError(
                    "gap_method=paper is an exploration trajectory: it "
                    "replaces the propagation force, so it requires "
                    "sfhp=nosh")
            pair = list(getattr(adaptive, 'gap_pair', []) or [])
            if explore_gap and pair:
                if len(pair) != 2 or any(
                        not 0 <= int(index) < int(dataset.nstate)
                        for index in pair):
                    raise ValueError(
                        "gap_pair must name two states inside the %d-state "
                        "ladder, got %r" % (int(dataset.nstate), pair))

    # ---- stage dispatch ------------------------------------------------

    @staticmethod
    def _screen_thresholds(settings):
        """The maximum threshold of every channel this run labels."""
        channels = set(settings.get('channels') or ())
        return {key: value
                for key, value in (settings.get('maxima') or {}).items()
                if key in channels}

    def _flags_for(self, state, stage_name, settings):
        flags = self._stage_flag_kwargs(state, stage_name)
        if stage_name == 'md':
            fallback = {
                'multitask': settings['md_multitask'],
                'remotetask': settings['md_remotetask'],
                'multipool': settings['md_multipool'],
            }
        elif stage_name == 'base.reference':
            fallback = {
                'multitask': settings['reference_multitask'],
                'remotetask': False,
                'multipool': settings['reference_multipool'],
            }
        else:
            fallback = {
                'multitask': False,
                'remotetask': settings['remote_train'],
                'multipool': 1,
            }
        return flags or fallback

    def _dispatch_batch(self, units, *, child_jobtype, active_role,
                        flags, root, name_format, workdir=None):
        """Dispatch one stage batch under that stage's own flags.

        ``workdir`` pins a one-unit batch (the training stage) to an
        existing directory instead of allocating a runtime subdirectory;
        the unit is still marked isolated so its scratch stays there.
        """
        if not units:
            return []
        os.makedirs(root, exist_ok=True)
        previous = (self.multitask, self.remotetask, self.multipool,
                    self._batch_root, self._batch_name_format)
        self.multitask = bool(flags.get('multitask', False))
        self.remotetask = bool(flags.get('remotetask', False))
        self.multipool = max(1, int(flags.get('multipool', 1)))
        self._batch_root = root
        self._batch_name_format = name_format
        try:
            if workdir is not None:
                workdir = os.path.abspath(workdir)
                os.makedirs(workdir, exist_ok=True)
                for unit in units:
                    unit['workdir'] = workdir
                    unit_state = unit.get('state')
                    runtime = getattr(unit_state, 'runtime', None)
                    if runtime is not None:
                        runtime.workdir = workdir
                    self._stamp_isolated_workdir(unit_state, workdir)
            elif len(units) == 1:
                self.allocate_runtime_identities(units)
            return self.dispatch(units=units, child_jobtype=child_jobtype,
                                 active_role=active_role)
        finally:
            (self.multitask, self.remotetask, self.multipool,
             self._batch_root, self._batch_name_format) = previous

    # ---- stage bodies --------------------------------------------------

    def _train_iteration(self, state, ml_slot, surface, data_path, modeldir,
                         iter_dir, settings, ml_indices,
                         name_format='train_{idx}'):
        """Train the iteration's network from the iteration's data file.

        The fit is one task in the iteration directory: its run log and the
        skill's output land beside the data file and the model directory
        rather than in a training subdirectory of their own.
        """
        unit = self.materialize_unit(
            {}, 0, child_jobtype='train', active_role='theory')
        child = unit['state']
        self._slice_child_system(child, ml_indices)
        child.theory = [child.theory[ml_slot]]
        # A training child is one skill call, not a composite evaluation.
        # The QMMM plan marks its QM task with the composite/duty keys so
        # the MD and reference children can stitch the classical surface;
        # keeping them on the lone training task made Calculator look for
        # an energy task that this child deliberately does not carry.
        for key in ('composite', 'qmmm_role', 'binding_role',
                    'partition_indices', 'probe_indices'):
            child.theory[0].pop(key, None)
        self._stamp_training(child.theory[0], surface, modeldir, data_path)

        flags = self._flags_for(state, 'train', settings)
        outcomes = self._dispatch_batch(
            [unit], child_jobtype='train', active_role='theory',
            flags=flags, root=iter_dir, name_format=name_format,
            workdir=iter_dir)
        outcome = outcomes[0] if outcomes else None
        if _is_error_outcome(outcome):
            return False, modeldir, {'reason': outcome.get('adaptive_error')}
        result = self._outcome_result(outcome)
        settled, reason = ml_verdict(
            'train', result, required=declared_channels(child, 'train'))
        if not settled:
            return False, modeldir, {'reason': reason}
        try:
            self._require_servable(child.theory[0], surface, modeldir)
        except (ValueError, FileNotFoundError) as error:
            return False, modeldir, {'reason': str(error)}
        return True, modeldir, {'reason': '', 'result': result}

    @staticmethod
    def _bias_spec(settings, iteration=1):
        """The adaptive-private propagation spec, or None when both are off."""
        udd = None
        if (int(settings.get('explore_udd') or 0)
                and int(iteration) >= int(
                    settings.get('udd_start_iteration') or 1)):
            udd = {
                'udd_mode': str(settings.get('udd_mode', 'force')),
                'udd_amplitude': float(
                    settings.get('udd_amplitude') or 0.0),
                'udd_width': float(settings.get('udd_width') or 0.0),
            }
        gap = None
        if int(settings.get('explore_gap') or 0):
            gap = {
                'gap_method': str(settings.get('gap_method', 'cv')),
                'gap_pair': list(settings.get('gap_pair') or []),
                'gap_threshold': float(
                    settings.get('gap_threshold') or 0.03),
                'gap_height': float(settings.get('gap_height') or 0.0),
                'gap_width': float(settings.get('gap_width') or 0.0),
            }
        if udd is None and gap is None:
            return None
        return {'udd': udd, 'gap': gap}

    def _gap_initial_conditions(self, settings):
        """Initial conditions for the paper gap exploration, or the batch.

        The paper launches gap trajectories from a pre-trigger step of the
        previous round's trajectories, half on the lower and half on the
        upper surface. The row carries its velocity only when an analysis
        switch is on, and `explore_gap` makes the layer active, so it is
        present here. A round with no such frame falls back to the
        configured batch with a warning rather than inventing a start.
        """
        if not (int(settings.get('explore_gap') or 0)
                and str(settings.get('gap_method', 'cv')).lower() == 'paper'):
            return self._initial_conditions
        screen = self._last_screen
        if not screen:
            return self._initial_conditions
        pair = list(settings.get('gap_pair') or [])
        if len(pair) != 2:
            pair = [0, 1]
        lower, upper = int(pair[0]), int(pair[1])
        conditions = []
        for result in screen.get('per_trajectory') or []:
            rows = result.get('frames') or []
            pre = [row for row in rows
                   if not (row.get('flags') or {}).get('trigger')]
            if not pre:
                continue
            row = pre[-1]
            coordinates = row.get('coordinates')
            velocities = row.get('velocity')
            if coordinates is None or velocities is None:
                continue
            for surface in (lower, upper):
                conditions.append({
                    'coordinates': coordinates,
                    'velocities': velocities,
                    'current_state': surface,
                })
        if not conditions:
            self.logger.warning(
                "Gap exploration found no pre-trigger frame with velocities; "
                "falling back to the configured initial conditions")
            return self._initial_conditions
        return conditions

    def _md_iteration(self, state, iteration, iter_dir, ml_slot, modeldir,
                  settings):
        """Dispatch one MD child runtime per initial condition."""
        units = []
        history = max(int(settings['record_step']), 4)
        # The adaptive loop owns the md child's detection thresholds: the
        # &adaptive max* values overwrite whatever &md declared. All four
        # channels are stamped, including the ones sample_* left off, so
        # the guard sees exactly the thresholds this run screens against.
        maxima = settings.get('maxima') or {}
        guard = {
            'md_maxenergy': maxima.get('err_energy'),
            'md_maxgrad': maxima.get('err_grad'),
            'md_maxnac': maxima.get('err_nac'),
            'md_maxsoc': maxima.get('err_soc'),
        }
        # The population bound belongs to the md child's own `&md md_maxpop`
        # (default 0.01). Adaptive does not overwrite it: the guard is md's
        # judgment, and `&adaptive check_population` only decides whether the
        # structure it stopped on is collected.
        conditions = self._gap_initial_conditions(settings)
        for index, condition in enumerate(conditions):
            unit = self.materialize_unit(
                {'coordinates': condition['coordinates'],
                 'velocities': condition['velocities']},
                index, child_jobtype='md', active_role='theory')
            child = unit['state']
            self._stamp_serving(child, ml_slot, modeldir,
                                predict_mode=settings.get('predict_mode', 0))
            updates = dict(guard)
            updates['history'] = history
            if condition.get('current_state') is not None:
                state_index = int(condition['current_state'])
                updates['root'] = state_index
                child.update_system({'current_state': state_index})
            child.update_dynamics(updates)
            spec = trace_spec(settings.get('analysis') or {})
            if spec is not None:
                # The trace rides RuntimeState into the child and back; it
                # is omitted entirely when no analysis is on.
                _runtime_of(child).analysis_trace = {
                    'spec': spec, 'rows': []}
            bias = self._bias_spec(settings, iteration)
            if bias is not None:
                _runtime_of(child).analysis_bias = bias
            units.append(unit)
        flags = self._flags_for(state, 'md', settings)
        return self._dispatch_batch(
            units, child_jobtype='md', active_role='theory',
            flags=flags,
            root=os.path.join(iter_dir, 'md'),
            name_format='traj_{idx}')

    def _analyze_iteration(self, state, iteration, iter_dir, screen,
                           outcomes, dataset, settings, training,
                           record=None, new_labels=0):
        """Build and write the round's analysis report.

        Returns
        -------
        tuple(dict or None, str or None, bool)
            (report, fatal_reason, saturated). With every analysis switch
            off the triple is (None, None, False) and nothing is written.
        """
        plan = settings.get('analysis') or {}
        if not analysis_active(plan):
            return None, None, False
        report = AnalysisFrame(
            iteration=int(iteration),
            plan=dict(plan),
            trajectories=[],
            summary={},
        )
        missing = []
        for index, result in enumerate(screen['per_trajectory']):
            rows = result.get('frames') or []
            entry = {'traj_id': result.get('traj_id'),
                     'n_frames': len(rows),
                     'exploration': bool(result.get('exploration')),
                     'population_status': result.get('population_status'),
                     'order_status': result.get('order_status')}
            bias_states = [row.get('bias_status') for row in rows
                           if row.get('bias_status')]
            if bias_states:
                entry['bias_status'] = bias_states[-1]
            for key in ('udd_energy', 'gap_bias_energy', 'gap_excess',
                        'gap_velocity_scale', 'gap_force_switched',
                        'bias_work', 'udd_force_ratio',
                        'udd_state_switches'):
                values = [row.get(key) for row in rows
                          if row.get(key) is not None]
                if values:
                    entry[key] = values[-1]
            if plan.get('gap', 'off') != 'off':
                entry['gap'] = summarize_gap(
                    rows, plan['gap'], settings.get('gap_threshold', 0.03))
            if plan.get('trace'):
                rows_trace = (trace_rows(outcomes[index])
                              if index < len(outcomes) else [])
                entry['trace'] = summarize_trace(rows_trace)
                if entry['trace'].get('status') == MISSING:
                    missing.append(entry['traj_id'])
            if plan.get('pes'):
                entry['pes'] = summarize_pes(rows, dataset)
            entry['predictor'] = predictor_status(
                rows, settings.get('predict_mode', 0))
            report['trajectories'].append(entry)

        selected = screen.get('selected') or []
        if plan.get('coverage'):
            lengths = [int(result['last_step'])
                       for result in screen['per_trajectory']
                       if result.get('last_step') is not None]
            average = (sum(lengths) / float(len(lengths))) if lengths else None
            report['summary']['coverage'] = summarize_coverage(
                selected, dataset, average_length=average,
                mdstep=settings.get('mdstep'))
        if plan.get('pes'):
            report['summary']['pes'] = summarize_pes(selected, dataset)

        validation = self._validation_mae(training)
        if validation is not None:
            self._validation_history.append(validation)
        saturation_record = saturation(
            self._validation_history, plan.get('window', 3),
            plan.get('global_tolerance', 0.0),
            local_tolerance=plan.get('local_tolerance', 0.0))
        report['summary']['saturation'] = saturation_record
        saturated = bool(saturation_record.get('saturated'))
        report['summary']['saturated'] = saturated
        previous_rounds = (record.get('iterations') or []
                           if isinstance(record, dict) else [])
        cumulative = sum(int(row.get('new_frames') or 0)
                         for row in previous_rounds) + int(new_labels or 0)
        rounds_since = 0
        if len(self._validation_history) >= 2:
            best = min(range(len(self._validation_history)),
                       key=lambda index: self._validation_history[index])
            rounds_since = len(self._validation_history) - 1 - best
        report['summary']['progress'] = {
            'cumulative_new_frames': cumulative,
            'rounds_since_improvement': rounds_since,
            'validation_history': list(self._validation_history),
        }

        fatal = None
        ntraj = int(screen.get('n_trajectory') or 0)
        if plan.get('trace') and ntraj and len(missing) == ntraj:
            fatal = ('analysis trace requested but no trajectory '
                     'produced trace rows')
        self._write_json(
            os.path.join(iter_dir, 'analysis_iter-%d.json' % iteration),
            report)
        for traj_id in missing:
            self.logger.warning(
                "Adaptive iteration %d: trajectory %s produced no analysis "
                "trace rows", iteration, traj_id)
        return report, fatal, saturated

    @staticmethod
    def _validation_mae(training, channels=('energy', 'gradient')):
        """Mean validation MAE of one fit over the requested channels, or None.

        The fit result carries `train_errors` as {ensemble: {channel:
        [member values]}}. The analysis layer prefers energy because its
        unit is comparable across rounds and falls back to the gradient;
        the quality gate asks for energy alone, because an eV target and
        an eV/Angstrom gradient are not the same statement.
        """
        if not isinstance(training, dict):
            return None
        errors = training.get('train_errors')
        if not isinstance(errors, dict):
            return None
        for channel in channels:
            values = []
            for channels in errors.values():
                if not isinstance(channels, dict):
                    continue
                series = channels.get(channel)
                if isinstance(series, (list, tuple)):
                    values.extend(float(value) for value in series
                                  if value is not None)
                elif series is not None:
                    values.append(float(series))
            if values:
                return float(np.mean(values))
        return None

    def _validation_gate(self, state, iteration, iter_dir, ml_slot, surface,
                         title, ml_indices, dataset, dataset_path, modeldir,
                         training, new_positions, base_nframe, settings):
        """Judge the fit's validation energy MAE and prune suspect frames.

        With ``maxmae`` at 0 the gate is off and nothing is evaluated.
        Otherwise a fit that misses the target is retried after removing
        one frame of this round at a time; the removal stays only when
        the retrained fit lowers the MAE, and the run stops when it does
        not. A frame is suspect when its QM energy lies outside the
        robust band of the frames the dataset held before this round,
        which is a statement about the reference labels rather than
        about a prediction.

        Parameters
        ----------
        dataset : TrainingData
            The accumulated dataset, this round's labels already at the
            tail.
        new_positions : list of int
            Frame positions this round appended, in dataset order.
        base_nframe : int
            Frame count before the append; the reference block of the
            outlier test ends here.
        settings : dict
            Adaptive settings, for `maxmae`.

        Returns
        -------
        dict
            The gate record. Its 'status' is 'off', 'passed' or
            'aborted'; 'dataset', 'dataset_path', 'modeldir' and
            'training' are what the iteration continues with.
        """
        target = float(settings.get('maxmae') or 0.0)
        mae = self._validation_mae(training, channels=('energy',))
        gate = {
            'status': 'off',
            'reason': '',
            'target': target,
            'unit': 'eV',
            'mae': mae,
            'dataset': dataset,
            'dataset_path': dataset_path,
            'modeldir': modeldir,
            'training': training,
            'new_frames': len(new_positions),
            'kept_frames': len(new_positions),
            'pruned': [],
            'attempts': [],
        }
        if target <= 0:
            return gate
        gate['attempts'].append({
            'round': 0,
            'modeldir': modeldir,
            'mae': mae,
            'pruned': None,
            'pruned_z': None,
            'decision': 'start',
            'detail': {},
        })
        if mae is None:
            return self._gate_abort(
                gate, iteration, target,
                'the fit reported no energy validation MAE')
        if mae <= target:
            gate['status'] = 'passed'
            return gate

        remaining = [int(index) for index in new_positions]
        energy = dataset.energy
        while remaining:
            round_index = len(gate['attempts'])
            reference = None if energy is None else energy[:base_nframe]
            candidates = None if energy is None else energy[remaining]
            flags, detail = energy_outlier_flags(reference, candidates)
            attempt = {
                'round': round_index,
                'modeldir': '',
                'mae': None,
                'pruned': None,
                'pruned_z': None,
                'decision': 'rejected',
                'detail': detail,
            }
            gate['attempts'].append(attempt)
            if not detail.get('testable'):
                return self._gate_abort(
                    gate, iteration, target,
                    'cannot judge the new frames: %s'
                    % (detail.get('reason') or 'no robust energy scale'))
            flagged = list(detail['flagged'])
            if len(flagged) >= len(remaining):
                return self._gate_abort(
                    gate, iteration, target,
                    'every frame added this iteration is an outlier')
            if not flagged:
                return self._gate_abort(
                    gate, iteration, target,
                    'no outlier among the %d new frame(s)' % len(remaining))
            worst = max(flagged,
                        key=lambda index: detail['max_abs_z'][index])
            victim = remaining[worst]
            candidate = dataset.subset(
                [index for index in range(int(dataset.nframe))
                 if index != victim])
            trial_path = os.path.join(
                iter_dir, 'prune', 'data_iter-%d_%d_r%d.json'
                % (iteration, int(candidate.nframe), round_index))
            self._save_dataset(candidate, trial_path)
            trial_dir = os.path.join(
                iter_dir,
                surface['query'].model_dir_name(
                    title, 'iter-%d-r%d' % (iteration, round_index)))
            trial_trained, trial_serving, note = self._train_iteration(
                state, ml_slot, surface, trial_path, trial_dir, iter_dir,
                settings, ml_indices,
                name_format='train_r%d_{idx}' % round_index)
            attempt['modeldir'] = trial_serving
            attempt['pruned'] = int(victim)
            attempt['pruned_z'] = detail['max_abs_z'][worst]
            if not trial_trained:
                return self._gate_abort(
                    gate, iteration, target,
                    'retraining without frame %d failed: %s'
                    % (victim, note.get('reason') or 'training failed'))
            trial_mae = self._validation_mae(
                note.get('result'), channels=('energy',))
            attempt['mae'] = trial_mae
            if trial_mae is None or trial_mae >= mae:
                return self._gate_abort(
                    gate, iteration, target,
                    'removing frame %d did not improve the energy MAE '
                    '(%s -> %s eV); the frame stays in the dataset'
                    % (victim, _mae_text(mae), _mae_text(trial_mae)))
            attempt['decision'] = 'accepted'
            dataset, dataset_path = candidate, trial_path
            modeldir, training = trial_serving, note.get('result')
            mae = trial_mae
            # The pruned frame is gone from the frame axis, so the new
            # frames after it move down one position and the cached label
            # array is read from the dataset that now holds them.
            energy = dataset.energy
            remaining = [index - 1 if index > victim else index
                         for index in remaining if index != victim]
            gate.update({
                'dataset': dataset,
                'dataset_path': dataset_path,
                'modeldir': modeldir,
                'training': training,
                'mae': mae,
                'kept_frames': len(remaining),
                'pruned': list(gate['pruned']) + [int(victim)],
            })
            if mae <= target:
                gate['status'] = 'passed'
                return gate
        return self._gate_abort(
            gate, iteration, target,
            'the target was not reached after removing every flagged outlier')

    @staticmethod
    def _gate_abort(gate, iteration, target, reason):
        """Mark the gate aborted and spell the stop reason in one place."""
        gate['status'] = 'aborted'
        gate['kept_frames'] = int(gate['new_frames']) - len(gate['pruned'])
        gate['reason'] = (
            'training quality gate (iteration %d): validation energy MAE '
            '%s eV exceeds %g eV; %s'
            % (iteration, _mae_text(gate.get('mae')), target, reason))
        return gate

    def _reference_iteration(self, state, iteration, iter_dir, selected,
                             contract, settings, ml_indices):
        """Run the reference calculations for the selected frames.

        The calculations live in this iteration's directory, beside the data
        file, the network and the MD trajectories they belong to. The
        reference labels the trainable region, so each child is sliced to
        those atoms before Calculator sees it.
        """
        units = []
        indices = [int(index) for index in ml_indices]
        plan = next(
            (task.get('label_plan') for task in getattr(state, 'reference', [])
             if task.get('label_plan')), None)
        task_ids = (list(plan.get('source_task_ids') or [])
                    if isinstance(plan, dict) else [])
        # A single reference task is the old sliced-high-region plan. A
        # composite reference needs the full geometry for its low layer /
        # MM environment; its tasks carry their own regions, and the label
        # channels are projected onto the ML atoms afterwards.
        slice_child = not bool(plan and plan.get('full_system'))
        for index, row in enumerate(selected):
            unit = self.materialize_unit(
                {'coordinates': row.get('coordinates')},
                index, child_jobtype='energy', active_role='reference')
            # The reference child must see the selected frame's cell,
            # periodicity and image counts, not the parent input's zero
            # image flags: a periodic bonded frame whose atoms crossed a
            # face is otherwise refused by the classical low layer's image
            # check.
            updates = {}
            coordinates = row.get('coordinates')
            if coordinates is not None:
                updates['coordinates'] = [
                    list(map(float, xyz)) for xyz in coordinates]
            for key in ('image', 'cell', 'pbc'):
                value = row.get(key)
                if value is not None:
                    if key == 'image':
                        updates[key] = [list(map(int, row)) for row in value]
                    elif key == 'cell':
                        updates[key] = list(map(float, value))
                    else:
                        updates[key] = [bool(item) for item in value]
            if updates and hasattr(unit['state'], 'update_system'):
                unit['state'].update_system(updates)
            if hasattr(unit['state'], 'iter_tasks'):
                self._plain_skill_geometry(unit['state'])
            if slice_child:
                self._slice_child_system(unit['state'], indices)
            self._stamp_reference(unit['state'],
                                  settings['reference_remotetask'])
            units.append(unit)

        flags = self._flags_for(state, 'base.reference', settings)
        outcomes = self._dispatch_batch(
            units, child_jobtype='energy', active_role='reference',
            flags=flags, root=os.path.join(iter_dir, 'reference'),
            name_format='ref_{idx}')

        labels = []
        failures = []
        aux_rows = []
        base_group = int(contract.get('nframe') or 0)
        for index, (row, outcome) in enumerate(zip(selected, outcomes)):
            values, reason = labels_from_outcome(outcome, contract, indices)
            if values is None:
                if _is_error_outcome(outcome):
                    reason = (outcome.get('adaptive_error')
                              or 'reference calculation failed')
                failures.append({
                    'traj_id': row.get('traj_id'),
                    'frame_index': row.get('frame_index'),
                    'step': row.get('step'),
                    'ref_index': index,
                    'source_task_ids': task_ids,
                    'reason': reason,
                })
                self.logger.warning(
                    "Adaptive iteration %d reference ref_%d (tasks %s) "
                    "failed for traj %s step %s: %s",
                    iteration, index, task_ids or 'composite total',
                    row.get('traj_id'), row.get('step'), reason)
                continue
            coordinates = row.get('coordinates')
            values['coordinates'] = ([coordinates[index] for index in indices]
                                     if coordinates is not None else None)
            if contract.get('has_cell'):
                values['cell'] = row.get('cell')
            if contract.get('has_pbc'):
                values['pbc'] = row.get('pbc')
            if contract.get('has_group_id'):
                values['group_id'] = base_group + index
            if contract.get('has_permutation_id'):
                values['permutation_id'] = 0
            labels.append(values)
            raw = self._outcome_result(outcome)
            channels = (raw.get('label_channels')
                        if isinstance(raw, dict) else None)
            if isinstance(channels, dict):
                auxiliary = {
                    key: channels.get(key)
                    for key in ('field', 'mm_energy', 'q_qm',
                                'mid_charges', 'low_charges')
                    if channels.get(key) is not None}
                if auxiliary:
                    auxiliary['region_indices'] = channels.get(
                        'region_indices')
                    auxiliary.update({
                        'traj_id': row.get('traj_id'),
                        'frame_index': row.get('frame_index'),
                        'step': row.get('step')})
                    aux_rows.append(auxiliary)
        if aux_rows:
            self._write_json(
                os.path.join(iter_dir, 'reference', 'aux-labels.json'),
                {'frames': aux_rows})
        if len(labels) < len(selected):
            self.logger.warning(
                "Adaptive iteration %d: %d of %d reference calculations yielded "
                "no usable label.", iteration, len(selected) - len(labels),
                len(selected))
        return labels, failures

    # ---- child configuration stamps ------------------------------------

    def _stamp_serving(self, child, ml_slot, modeldir, predict_mode=0):
        skill = self._skill_config(child.theory[ml_slot])
        skill['modeldir'] = modeldir
        # The adaptive predictor selection overrides whatever the skill
        # section carried, and both reach the same serving path.
        skill['nn_predict_mode'] = int(predict_mode or 0)
        # The training-frame prediction spread is a fit diagnostic, not a
        # per-step property: a serving child never pays for it.
        skill['predict_std'] = 0

    def _stamp_reference(self, child, remote_mode):
        if not remote_mode:
            return
        for _category, task in child.iter_tasks():
            config = task.get('config') or {}
            skill = config.get('skill_config')
            if isinstance(skill, dict):
                skill['execution_mode'] = 'slurm'

    def _stamp_training(self, task, surface, modeldir, data_path):
        skill = self._skill_config(task)
        # Which trainable profile this job fits is the skill's answer
        # (`query.scan_requirements`), the same call the grid scan makes.
        skill['profile'] = surface['query'].scan_requirements(
            skill)['profile']
        skill['modeldir'] = modeldir
        skill['train_data'] = data_path
        skill['train_dataset'] = {}
        skill['train_mode'] = 'training'
        skill['pred_data'] = ''
        task['capabilities'] = []

    # ---- iteration outputs -------------------------------------------------

    def _write_iteration(self, state, iteration, iter_dir, external,
                         dataset_path, modeldir, screen, labels, failures,
                         record, reference_ran=False,
                         reference_iteration=None, trained=False,
                         received=0, verified=0, training=None, ml_atoms=None,
                         thresholds=None, analysis=None, gate=None):
        """Write the iteration's statistics and append its formatted block.

        `screen` is None when the training-data quality gate stopped the
        run before MD; the block then reports the data, the last fit and
        the gate, and writes no trajectory product.
        """
        from ..utilities.print_results import format_adaptive_iteration

        system_info = self._fit_system_info(state, ml_atoms)
        kept = int(gate['kept_frames']) if gate else len(labels)
        pruned = len(gate['pruned']) if gate else 0
        # The kept/pruned counts are a gate product: with the gate off the
        # iteration record stays exactly what it was before the gate existed.
        armed = bool(gate) and gate.get('status') != 'off'
        if screen is None:
            compact = {
                'iteration': iteration,
                'external': bool(external),
                'dataset': dataset_path,
                'modeldir': modeldir,
                'trained': bool(trained),
                'reference_iteration': reference_iteration,
                'selection_iteration': reference_iteration,
                'received': int(received),
                'verified': int(verified),
                'n_passed': 0,
                'completed': 0,
                'triggered': False,
                'selected': 0,
                'reference_ran': bool(reference_ran),
                'reference_failed': len(failures),
                'new_frames': len(labels),
                'kept_frames': kept,
                'pruned_frames': pruned,
                'status': 'training quality gate aborted',
            }
            record['iterations'].append(compact)
            payload = dict(compact)
            payload['reference_failures'] = failures
            if armed:
                payload['validation_gate'] = self._gate_payload(gate)
            self._write_json(
                os.path.join(iter_dir, 'status_iter-%d.json' % iteration),
                payload)
            self._write_json(
                os.path.join(self._root, '%s.adaptive.json'
                             % str(getattr(state.pmd.control, 'title',
                                           '') or 'pyrai3md')), record)
            block = format_adaptive_iteration(
                compact, screen=None, thresholds=thresholds,
                training=training, system=system_info, gate=gate)
            if getattr(state, 'logger', None) is not None:
                state.logger.write(block)
            return

        lengths = [int(result['last_step'])
                   for result in screen['per_trajectory']
                   if result.get('last_step') is not None]
        # Exploration trajectories are not production statistics: the
        # ground-state fraction is computed over the non-exploration ones.
        production = [result for result in screen['per_trajectory']
                      if not result.get('exploration')]
        ground = sum(1 for result in production
                     if result.get('last_state') == 0)
        n_production = len(production)
        ntraj = int(screen['n_trajectory'])
        compact = {
            'ground': ground,
            'ground_percent': ((100.0 * ground / n_production)
                               if n_production else None),
            'exploration': bool(screen.get('exploration')),
            'iteration': iteration,
            'external': bool(external),
            'dataset': dataset_path,
            'modeldir': modeldir,
            'trained': bool(trained),
            'reference_iteration': reference_iteration,
            'selection_iteration': reference_iteration,
            'received': int(received),
            'verified': int(verified),
            'md_kept': int(record.get('keep_md', 1)) != 0,
            'ntraj': int(screen['n_trajectory']),
            'completed': int(screen['n_completed']),
            'triggered': bool(screen['triggered']),
            'selected': len(screen['selected']),
            'discarded': int(screen.get('n_discarded', 0)),
            'discarded_bad': int(screen.get('n_bad', 0)),
            'average_length': ((sum(lengths) / float(len(lengths)))
                               if lengths else 0.0),
            'reference_ran': bool(reference_ran),
            'reference_failed': len(failures),
            'new_frames': len(labels),
        }
        if armed:
            compact['kept_frames'] = kept
            compact['pruned_frames'] = pruned
        record['iterations'].append(compact)

        title = str(getattr(state.pmd.control, 'title', '') or 'pyrai3md')
        nstate = int(getattr(
            getattr(state.pmd, 'system', None), 'nstate', 0) or 0)
        population = {
            'iteration': iteration,
            'nstate': nstate,
            'trajectories': self._populations(
                iter_dir, title, nstate, int(screen['n_trajectory'])),
        }
        self._write_json(
            os.path.join(iter_dir, 'population_iter-%d.json' % iteration),
            population)
        self._write_selected_xyz(
            os.path.join(iter_dir, 'selected_iter-%d.xyz' % iteration),
            list(state.atoms), screen['selected'])

        payload = dict(compact)
        payload['per_trajectory'] = [
            self._trajectory_payload(item)
            for item in screen['per_trajectory']]
        payload['reference_failures'] = failures
        if analysis is not None:
            payload['analysis'] = analysis
        if armed:
            payload['validation_gate'] = self._gate_payload(gate)
        self._write_json(
            os.path.join(iter_dir, 'status_iter-%d.json' % iteration),
            payload)

        summary_path = os.path.join(self._root,
                                    '%s.adaptive.json'
                                    % str(getattr(state.pmd.control,
                                                  'title', 'pyrai3md')))
        self._write_json(summary_path, record)

        block = format_adaptive_iteration(
            compact, screen=screen, thresholds=thresholds,
            training=training, system=system_info, gate=gate)
        if getattr(state, 'logger', None) is not None:
            state.logger.write(block)
            if analysis is not None:
                from ..utilities.print_results import format_analysis
                state.logger.write(format_analysis(analysis))

    @staticmethod
    def _fit_system_info(state, ml_atoms):
        """The system record the training report heads with."""
        pmd_system = getattr(state.pmd, 'system', None)
        if ml_atoms is not None:
            # The fit's report describes the trainable region, not the box.
            return {
                'atoms': list(ml_atoms),
                'natom': len(ml_atoms),
                'nstate': getattr(pmd_system, 'nstate', None),
                'nac_pair_map': getattr(pmd_system, 'nac_pair_map', None),
                'soc_pair_map': getattr(pmd_system, 'soc_pair_map', None),
            }
        if hasattr(pmd_system, 'model_dump'):
            return pmd_system.model_dump()
        return {
            key: getattr(pmd_system, key, None)
            for key in ('natom', 'nstate', 'nac_pair_map', 'soc_pair_map')}

    @staticmethod
    def _gate_payload(gate):
        """The JSON-safe part of a gate record (the dataset is an object)."""
        if not gate:
            return None
        return {key: value for key, value in gate.items()
                if key not in ('dataset', 'training')}

    @staticmethod
    def _trajectory_payload(result):
        rows = []
        for row in result.get('frames', []):
            payload = {
                'traj_id': row.get('traj_id'),
                'frame_index': row.get('frame_index'),
                'step': row.get('step'),
                'errors': row.get('errors'),
                'flags': row.get('flags'),
                'selected': row.get('selected'),
                'reason': row.get('reason'),
                'coordinates': row.get('coordinates'),
                'population': row.get('population'),
                'cell': row.get('cell'),
                'pbc': row.get('pbc'),
            }
            # Analysis-only fields exist on the row only when a switch was
            # on, so a default run's status_iter keeps its old key set.
            for key in ('energy', 'velocity', 'current_state', 'alternatives',
                        'bias_status', 'udd_energy', 'gap_bias_energy',
                        'gap_excess', 'gap_velocity_scale',
                        'gap_force_switched', 'bias_work',
                        'udd_force_ratio', 'udd_state_switches'):
                if key in row:
                    payload[key] = row[key]
            rows.append(payload)
        return {
            'traj_id': result.get('traj_id'),
            'completed': result.get('completed'),
            'exploration': bool(result.get('exploration')),
            'population_status': result.get('population_status'),
            'order_status': result.get('order_status'),
            'last_step': result.get('last_step'),
            'last_state': result.get('last_state'),
            'triggered': result.get('triggered'),
            'max_errors': result.get('max_errors'),
            'n_sampled': result.get('n_sampled'),
            'n_candidates': result.get('n_candidates'),
            'n_selected': result.get('n_selected'),
            'n_discarded': result.get('n_discarded'),
            'n_bad': result.get('n_bad'),
            'frames': rows,
        }

    @staticmethod
    def _read_population(path, nstate):
        """Read one trajectory's md.pop file into population rows.

        The file's first row is a header; each later row is
        ``time pop_1 .. pop_nstate act_1 .. act_nstate``. Only the
        populations are kept, one list per recorded step, because the
        trajectory lengths differ and must stay separate.
        """
        if nstate <= 0 or not os.path.isfile(path):
            return []
        rows = []
        with open(path) as handle:
            next(handle, None)
            for line in handle:
                values = line.split()
                if len(values) < 1 + nstate:
                    continue
                try:
                    rows.append([float(value)
                                 for value in values[1:1 + nstate]])
                except ValueError:
                    continue
        return rows

    @classmethod
    def _populations(cls, iter_dir, title, nstate, ntraj):
        """One population list per trajectory, never averaged."""
        entries = []
        for index in range(int(ntraj)):
            path = os.path.join(
                iter_dir, 'md', 'traj_%d' % (index + 1),
                '%s.md.pop' % title)
            rows = cls._read_population(path, nstate)
            entries.append({
                'traj_id': index + 1,
                'steps': len(rows),
                'population': rows,
            })
        return entries

    @staticmethod
    def _write_selected_xyz(path, atoms, selected):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            for index, row in enumerate(selected):
                coordinates = row.get('coordinates') or []
                handle.write('%d\n' % len(atoms))
                handle.write(
                    'traj_id=%s step=%s selected_idx=%d\n'
                    % (row.get('traj_id'), row.get('step'), index))
                for atom, xyz in zip(atoms, coordinates):
                    handle.write(
                        '%s %.10f %.10f %.10f\n'
                        % (atom, float(xyz[0]), float(xyz[1]),
                           float(xyz[2])))

    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(payload, handle, indent=2, default=_json_default,
                      allow_nan=False)

    def _save_dataset(self, dataset, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataset.save(path)

    def _log_method_alignment(self, state, plan):
        """Write the planned label correspondence and its warnings."""
        notes = [str(note) for note in (plan.get('notes') or []) if note]
        for note in notes:
            self.logger.warning("Adaptive method alignment: %s", note)
        logger = getattr(state, 'logger', None)
        if logger is None:
            return
        source = str(plan.get('source') or '')
        layer = str(plan.get('source_layer') or '')
        if layer:
            source = '%s (%s)' % (source, layer)
        tasks = ', '.join(str(value)
                          for value in (plan.get('source_task_ids') or []))
        aux = ', '.join(str(value)
                        for value in (plan.get('aux') or [])) or 'none'
        text = '\n  &method alignment\n%s\n' % ('-' * 79)
        text += '  %-24s %d atom(s)\n' % (
            'theory surface', len(plan.get('region_indices') or []))
        text += '  %-24s %s\n' % (
            'reference type', plan.get('reference_type') or 'none')
        text += '  %-24s %s\n' % ('label source', source or 'none')
        text += '  %-24s %s\n' % ('source task(s)', tasks or 'composite total')
        text += '  %-24s %s\n' % ('auxiliary channels', aux)
        if notes:
            text += '  warnings:\n'
            for note in notes:
                text += '    %s\n' % note
        logger.write(text)

    def _finish(self, state, record, failed=False):
        """Write the run summary and settle the parent runtime."""
        record['n_iterations'] = len(record.get('iterations', []))
        summary_path = os.path.join(
            self._root,
            '%s.adaptive.json' % str(getattr(
                state.pmd.control, 'title', 'pyrai3md')))
        # A run refused before its first iteration has no work directory and
        # must not create one: the summary is the record of a run that
        # started, and writing it beside the input turned a refused test run
        # into a stray `<title>.adaptive/` in the launch directory.
        if os.path.isdir(self._root):
            try:
                self._write_json(summary_path, record)
            except Exception as error:
                self.logger.warning(
                    "Could not write the adaptive summary: %s", error)

        merged = {}
        if getattr(state, 'result', None) is not None:
            try:
                merged = dict(state.result)
            except (TypeError, ValueError):
                merged = {}
        merged['adaptive'] = record
        self.calculator.update({'results': merged})

        if getattr(state, 'runtime', None) is not None:
            if failed:
                state.runtime.mark_failed(record.get('reason') or 'failed')
            else:
                state.runtime.mark_completed()
        return state

    # ---- helper accessors ---------------------------------------------

    @staticmethod
    def _outcome_result(outcome):
        if outcome is None:
            return {}
        if isinstance(outcome, dict):
            pmd = outcome.get('pmd')
            nested = (pmd.get('results') if isinstance(pmd, dict)
                      else getattr(pmd, 'results', None))
            if isinstance(nested, dict) and nested:
                return dict(nested)
            return outcome
        # A local reference outcome is a Result, which is mapping-like but
        # exposes no `.result` attribute; reading it as a mapping first is
        # what keeps label_channels (and therefore aux-labels.json) alive.
        if hasattr(outcome, 'keys') and hasattr(outcome, '__getitem__'):
            try:
                mapped = dict(outcome)
                if mapped:
                    pmd = mapped.get('pmd')
                    if isinstance(pmd, dict):
                        nested = pmd.get('results')
                    else:
                        nested = getattr(pmd, 'results', None)
                    if isinstance(nested, dict) and nested:
                        # A dispatched training child comes back as a
                        # serialized state document; its skill result lives
                        # under pmd.results, not at the document top level.
                        return dict(nested)
                    return mapped
            except (TypeError, ValueError):
                pass
        result = getattr(outcome, 'result', None)
        if result is not None:
            try:
                mapped = dict(result)
                if mapped:
                    return mapped
            except (TypeError, ValueError):
                pass
        pmd = getattr(outcome, 'pmd', None)
        if pmd is not None:
            return dict(getattr(pmd, 'results', {}) or {})
        return {}

    # ---- dispatcher overrides -----------------------------------------

    def allocate_runtime_identities(self, units):
        """Allocate workdirs, then stage the example's launch fixtures."""
        super().allocate_runtime_identities(units)
        self._stage_runtime_fixtures(units)

    def _stage_runtime_fixtures(self, units):
        """Copy launch-directory fixtures into each dispatched workdir.

        A remote child runs with the allocated workdir as its cwd, while
        the LAMMPS topology (`.data` / `.in.init` / `.in.settings`) and
        an optional `.charge` file were staged beside the example. A
        skill resolves those by bare name from its launch directory, so
        the parent copies them into the child's workdir before dispatch.
        """
        source = (os.path.dirname(self._root) if self._root
                  else os.getcwd())
        if not os.path.isdir(source):
            return
        names = [name for name in os.listdir(source)
                 if name.endswith(('.data', '.in.init', '.in.settings',
                                   '.charge'))]
        if not names:
            return
        source_abs = os.path.abspath(source)
        for unit in units:
            workdir = unit.get('workdir')
            if not workdir or os.path.abspath(workdir) == source_abs:
                continue
            os.makedirs(workdir, exist_ok=True)
            for name in names:
                shutil.copy2(os.path.join(source, name),
                             os.path.join(workdir, name))

    def runtime_root(self):
        return self._batch_root

    def runtime_name_format(self):
        return self._batch_name_format

    def runtime_isolated_workdir(self):
        return True

    def _create_copy(self):
        clone = super()._create_copy()
        clone.calculator = None
        clone.local_executor = None
        return clone

    @staticmethod
    def _stage_flag_kwargs(state, stage_name):
        """Build one stage's flags from the frozen execution graph."""
        graph = getattr(state, 'execution_graph', None)
        for node in graph or []:
            if node.get('name') != 'adaptive':
                continue
            for stage in node.get('stages', []) or []:
                if stage.get('name') != stage_name:
                    continue
                return {
                    'multitask': bool(stage.get('multitask')),
                    'remotetask': bool(stage.get('remotetask')),
                    'multipool': max(1, int(stage.get('multipool') or 1)),
                }
        return {}

    def __repr__(self):
        return ("AdaptiveSampling(maxiter=%d, maxsample=%d, load=%d)"
                % (self.maxiter, self.maxsample, self.load))