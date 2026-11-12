"""
======================================================================
Verification tests for the control.driver execution model:
driver validation, planning dispatch, workflow resolution, and the
SingleShot workflow.

Tests cover:
- _validate_driver: legal values (main/skill/skill name), unknown
  driver, multi-token theory with a skill driver, driver/theory
  mismatch; applied uniformly to every jobtype
- Planning dispatch: driver=main + optimize -> skill jobtype
  'energy gradient', spelled from the task capabilities so a QMMM
  field/energy task keeps its duty token; driver=skill -> jobtype
  stays 'optimize';
  optimize.method misuse of 'skill' fails fast with guidance
- resolve_workflow: driver=main resolves iterative workflows or
  None; driver != main always resolves SingleShot
- SingleShot semantics (mock calculator): one calculator call,
  opt_converge decides completed/failed, optimization outputs are
  written
- get_optimizer resolves framework optimizers only

Aug 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import numpy as np


def _make_state(results=None):
    """Create a minimal PyRAIState with optional pmd results."""
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.data.runtime.state import RuntimeState
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    config.system.natom = 2
    if results:
        config.results.update(results)

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    return state


class _IsolatedCwdTestCase(unittest.TestCase):
    """Run each test inside a temporary directory (opt outputs)."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmpdir.cleanup()


class TestValidateDriver(unittest.TestCase):
    """Test planning-layer driver validation (_validate_driver)."""

    def _state(self, driver, jobtype='energy', theory=('pyscf',)):
        state = _make_state()
        state.pmd.control.driver = driver
        state.pmd.control.jobtype = jobtype
        state.pmd.control.theory = list(theory)
        return state

    def test_main_driver_passes(self):
        from pyrai3md.planning.assembly import _validate_driver
        _validate_driver(self._state('main'))

    def test_skill_keyword_passes(self):
        from pyrai3md.planning.assembly import _validate_driver
        _validate_driver(self._state('skill'))

    def test_skill_name_passes(self):
        from pyrai3md.planning.assembly import _validate_driver
        _validate_driver(self._state('pyscf'))

    def test_unknown_driver_raises(self):
        from pyrai3md.planning.assembly import _validate_driver
        with self.assertRaises(ValueError) as ctx:
            _validate_driver(self._state('bogus'))
        message = str(ctx.exception)
        self.assertIn('bogus', message)
        self.assertIn('skill', message)

    def test_skill_driver_multi_theory_raises(self):
        from pyrai3md.planning.assembly import _validate_driver
        with self.assertRaises(ValueError) as ctx:
            _validate_driver(self._state(
                'skill', theory=('pyscf', 'molcas')))
        self.assertIn('one theory skill', str(ctx.exception))

    def test_named_driver_mismatch_raises(self):
        from pyrai3md.planning.assembly import _validate_driver
        with self.assertRaises(ValueError) as ctx:
            _validate_driver(self._state('molcas', theory=('pyscf',)))
        message = str(ctx.exception)
        self.assertIn('molcas', message)
        self.assertIn('pyscf', message)

    def test_validation_applies_to_single_point_jobtype(self):
        # The driver check is uniform for every jobtype (energy/
        # gradient/...), not only iterative ones.
        from pyrai3md.planning.assembly import _validate_driver
        with self.assertRaises(ValueError):
            _validate_driver(self._state('bogus', jobtype='energy'))
        _validate_driver(self._state('skill', jobtype='energy'))


class TestPlanningDispatch(unittest.TestCase):
    """Test planning-layer jobtype dispatch via build_state."""

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        # Idempotent; populates the Maestro with skill discovery so
        # build_state can resolve the pyscf theory task.
        Environment().initialize()

    def _build_state(self, jobtype='optimize', driver='main',
                     method='minima', theory=('pyscf',)):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = jobtype
        config.control.theory = list(theory)
        config.control.driver = driver
        config.system.atoms = ['C', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        config.system.natom = 2
        if jobtype == 'optimize':
            config.optimize.opt_method = method
        return build_state(config)

    def test_main_driver_sets_energy_gradient(self):
        state = self._build_state(driver='main', method='minima')
        skill_config = state.theory[0]['config']['skill_config']
        self.assertEqual(skill_config.get('jobtype'), 'energy gradient')

    def test_skill_driver_keeps_optimize(self):
        state = self._build_state(driver='skill')
        skill_config = state.theory[0]['config']['skill_config']
        self.assertEqual(skill_config.get('jobtype'), 'optimize')

    def test_named_driver_keeps_optimize(self):
        state = self._build_state(driver='pyscf')
        skill_config = state.theory[0]['config']['skill_config']
        self.assertEqual(skill_config.get('jobtype'), 'optimize')

    def test_skill_driver_single_point_keeps_jobtype(self):
        state = self._build_state(jobtype='energy', driver='skill')
        skill_config = state.theory[0]['config']['skill_config']
        self.assertEqual(skill_config.get('jobtype'), 'energy')

    def test_method_misuse_of_skill_raises_with_guidance(self):
        # optimize.method now records framework algorithms only;
        # 'skill' belongs to control.driver.
        with self.assertRaises(ValueError) as ctx:
            self._build_state(driver='main', method='skill')
        message = str(ctx.exception)
        self.assertIn('skill', message)
        self.assertIn('control.driver', message)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build_state(driver='main', method='bogus')
        self.assertIn('bogus', str(ctx.exception))

    def _build_qmmm_state(self, method='minima'):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.theory = ['molcas', 'lammps']
        config.control.driver = 'main'
        config.control.multiscale_type = 'qmmm'
        config.system.atoms = ['C', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        config.system.natom = 2
        config.system.highlevel = [0]
        config.system.embedding = True
        config.optimize.opt_method = method
        return build_state(config)

    def test_qmmm_optimize_keeps_the_task_duty_tokens(self):
        # The classical provider is selected by its duty token: without
        # it the same task resolves to a plain force-field profile that
        # still carries qm_atoms, which the skill refuses.
        state = self._build_qmmm_state()
        by_role = {task.get('qmmm_role'): task['config']['skill_config']
                   for task in state.theory}
        self.assertEqual(sorted(by_role), ['energy', 'field', 'qm'])
        self.assertEqual(by_role['field'].get('jobtype'),
                         'energy gradient field')
        self.assertEqual(by_role['field'].get('profile'), 'qmmm_field')
        self.assertEqual(by_role['energy'].get('jobtype'),
                         'energy gradient qmmm')
        self.assertEqual(by_role['energy'].get('profile'), 'qmmm_energy')

    def test_qmmm_path_keeps_the_task_duty_tokens(self):
        # A path search drives the same per-step request through the
        # optimizer branch, image by image.
        state = self._build_qmmm_state(method='mep')
        by_role = {task.get('qmmm_role'): task['config']['skill_config']
                   for task in state.theory}
        self.assertEqual(by_role['field'].get('profile'), 'qmmm_field')
        self.assertEqual(by_role['energy'].get('profile'), 'qmmm_energy')


class _FakeState:
    """Minimal state stub for resolve_workflow resolution tests."""

    def __init__(self, driver, jobtype, optimize_params=None):
        self.driver = driver
        self.jobtype = jobtype
        if optimize_params is not None:
            self.optimize_params = optimize_params


class TestResolveWorkflowDriver(unittest.TestCase):
    """Test resolve_workflow driver awareness."""

    def _resolve(self, driver, jobtype):
        from pyrai3md.workflow import resolve_workflow
        return resolve_workflow(_FakeState(driver, jobtype))

    def test_main_optimize_resolves_optimize(self):
        from pyrai3md.workflow import Optimize
        self.assertIs(self._resolve('main', 'optimize'), Optimize)

    def test_main_md_resolves_aimd(self):
        from pyrai3md.workflow import AIMD
        self.assertIs(self._resolve('main', 'md'), AIMD)

    def test_main_single_point_resolves_none(self):
        self.assertIsNone(self._resolve('main', 'energy'))
        self.assertIsNone(self._resolve('main', 'energy gradient'))

    def test_skill_driver_always_resolves_singleshot(self):
        from pyrai3md.workflow import SingleShot
        for jobtype in ('optimize', 'md', 'energy',
                        'energy gradient', 'frequency'):
            self.assertIs(self._resolve('skill', jobtype), SingleShot,
                          jobtype)

    def test_named_driver_resolves_singleshot(self):
        from pyrai3md.workflow import SingleShot
        self.assertIs(self._resolve('pyscf', 'optimize'), SingleShot)

    def test_engine_owned_steps_refuse_a_framework_cell_variable(self):
        # `optimize_cell` is read by the framework's optimizer, and this route
        # never builds one: the engine gets a single evaluation and returns the
        # geometry it stopped on. Left alone, the keyword would sit in the input
        # claiming a variable box while a fixed-box profile ran - the same shape
        # of defect as the invented `opt_optimize_cell` that hid a dropped
        # keyword once (CHANGELOG.md #355).
        from pyrai3md.workflow import resolve_workflow
        state = _FakeState('skill', 'optimize',
                           {'optimize_cell': True})
        with self.assertRaises(ValueError) as caught:
            resolve_workflow(state)
        message = str(caught.exception)
        self.assertIn('optimize_cell', message)
        self.assertIn('driver', message)

    def test_the_refusal_moves_only_with_the_keyword_and_the_driver(self):
        # Two arms, one factor each: the keyword off under the same engine-owned
        # driver still resolves, and the keyword on under `main` still reaches
        # the optimizer. Without these the first test would pass for a router
        # that refused every cell run, which is not the claim.
        from pyrai3md.workflow import Optimize, SingleShot, resolve_workflow
        self.assertIs(resolve_workflow(
            _FakeState('skill', 'optimize', {'optimize_cell': False})),
            SingleShot)
        self.assertIs(resolve_workflow(
            _FakeState('main', 'optimize', {'optimize_cell': True})),
            Optimize)


class TestEngineOwnedIterationNeedsItsOwnProfile(unittest.TestCase):
    """A skill that cannot own the iteration refuses it by name.

    Under `driver != 'main'` the jobtype reaches the skill unchanged and the
    skill owns every step. A profile table with no row for that jobtype can
    only answer with a single point, so the refusal has to reach the caller:
    downgraded to a warning it would print a finished calculation that never
    searched anything (the `driver: skill` probe of #367, dev/TODO.md).
    """

    # Skills whose profile table has no optimize row, with a method each
    # declares. The refusal lives in the table; the framework only stops
    # swallowing it on this route.
    _UNSERVED = (
        ('xtb_skill', 'gfn_ff', 'optimize'),
        ('qdm_skill', 'tully1', 'optimize'),
        ('nn_pes_skill', 'mlp_eg', 'optimize'),
    )

    def _flat(self, skill, method, jobtype, driver):
        from pyrai3md.planning.assembly import _build_skill_flat_and_submodels
        merged = {'method': method, 'jobtype': jobtype, 'driver': driver}
        return _build_skill_flat_and_submodels(skill, merged, driver=driver)

    def test_a_skill_owned_iteration_with_no_profile_is_refused(self):
        for skill, method, jobtype in self._UNSERVED:
            with self.assertRaises(ValueError) as caught:
                self._flat(skill, method, jobtype, 'skill')
            message = str(caught.exception)
            self.assertIn(jobtype, message, skill)
            self.assertIn('skill', message, skill)

    def test_the_refusal_moves_with_the_driver_and_not_the_jobtype(self):
        # One factor each: under `main` the framework owns the steps, so the
        # same unresolvable combination only warns and keeps the default
        # profile - the documented behaviour for a whole-system layer handed
        # another layer's method (CHANGELOG.md #94). Without this arm the
        # refusal above would also pass for a gate that refused everything.
        for skill, method, jobtype in self._UNSERVED:
            with self.assertLogs('pyrai3md.planning.assembly', level='WARNING'):
                self._flat(skill, method, jobtype, 'main')

    def test_a_property_request_under_the_same_driver_still_resolves(self):
        # The positive control: the engine-owned route is refused for the
        # iteration, not for the evaluation the skill is built for.
        flat = self._flat('xtb_skill', 'gfn_ff', 'energy gradient', 'skill')
        self.assertEqual(flat.get('profile'), 'gfn_ff_gradient')

    def test_the_other_iteration_keywords_are_refused_too(self):
        # xtb spells 'md' as a gradient for the framework's per-step ask; on
        # the route that hands the steps over it owns the propagation, which
        # it cannot do either.
        for jobtype in ('md', 'surface_hopping'):
            with self.assertRaises(ValueError):
                self._flat('xtb_skill', 'gfn_ff', jobtype, 'skill')


class _SkillMockCalculator:
    """Mock calculator simulating a skill-internal job.

    run() returns a result carrying the skill's opt_converge flag
    and writes the results back into the state like the real
    Calculator does.
    """

    def __init__(self, opt_converge=True, energy=-75.0,
                 include_flag=True):
        self._opt_converge = opt_converge
        self._energy = energy
        self._include_flag = include_flag
        self.call_count = 0
        self.state = None

    def run(self):
        self.call_count += 1
        results = {'energy': [self._energy]}
        if self._include_flag:
            results['opt_converge'] = self._opt_converge
        if self.state is not None and self.state.pmd is not None:
            self.state.set_result(results)
        return dict(results)

    def get_state(self):
        return self.state


class _SummaryRecorder:
    """Recording stub for the RunLogger optimization summary channel."""

    def __init__(self):
        self.summary = None
        self.log_step_calls = 0
        self.notes = []

    def write(self, text):
        self.notes.append(text)

    def log_optimize_setup(self, setup):
        del setup

    def log_step(self, step, snapshot, thresholds=None):
        del step, snapshot, thresholds
        self.log_step_calls += 1

    def log_optimize_step(self, step, report, converged,
                          thresholds=None, details=None):
        del step, report, converged, thresholds, details

    def log_optimize_summary(self, converged, steps, energy=None,
                             message='', label='Geometry optimization'):
        self.summary = {'converged': converged, 'steps': steps,
                        'energy': energy, 'message': message,
                        'label': label}


class TestSingleShotWorkflow(_IsolatedCwdTestCase):
    """Test the SingleShot (skill-driven) workflow semantics."""

    def _run(self, jobtype='optimize', opt_converge=True,
             include_flag=True, logger=None):
        from pyrai3md.workflow.single import SingleShot
        calc = _SkillMockCalculator(opt_converge=opt_converge,
                                    include_flag=include_flag)
        state = _make_state()
        state.pmd.control.jobtype = jobtype
        state.pmd.control.driver = 'skill'
        calc.state = state
        if logger is not None:
            state.logger = logger
        wf = SingleShot(calculator=calc)
        final = wf.run()
        return calc, final

    def test_single_calculator_call(self):
        calc, _ = self._run()
        self.assertEqual(calc.call_count, 1)

    def test_true_flag_marks_completed(self):
        _, state = self._run(opt_converge=True)
        self.assertEqual(state.runtime.status, 'completed')

    def test_missing_flag_marks_completed(self):
        _, state = self._run(include_flag=False)
        self.assertEqual(state.runtime.status, 'completed')

    def test_false_flag_marks_failed_without_retry(self):
        calc, state = self._run(opt_converge=False)
        self.assertEqual(state.runtime.status, 'failed')
        # No framework-side retry: exactly one calculator call.
        self.assertEqual(calc.call_count, 1)

    def test_optimize_outputs_written_without_framework_table(self):
        # The final structure is a real product of an engine-iteration
        # run; opt_status.txt is not, because the framework neither took
        # a step nor measured a gradient on this path.
        self._run()
        for path in ('opt.xyz', 'opt_geom.xyz'):
            self.assertTrue(os.path.exists(path), path)
        self.assertFalse(os.path.exists('opt_status.txt'))
        with open('opt.xyz') as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip(), '2')
        self.assertIn('E = -75.0', lines[1])

    def test_single_point_jobtype_no_opt_outputs(self):
        self._run(jobtype='energy')
        self.assertFalse(os.path.exists('opt.xyz'))

    def test_md_jobtype_missing_flag_completes(self):
        # The skill-side md convergence flag is not defined yet;
        # a missing flag passes.
        _, state = self._run(jobtype='md', include_flag=False)
        self.assertEqual(state.runtime.status, 'completed')

    def test_md_jobtype_names_the_verdict_it_could_not_make(self):
        # Completing was always right; the silence was not, because a bare
        # `completed` line reads as though a convergence verdict had been
        # reached. The run log now states which judgement did not run.
        recorder = _SummaryRecorder()
        _, state = self._run(jobtype='md', include_flag=False,
                             logger=recorder)
        self.assertEqual(state.runtime.status, 'completed')
        self.assertEqual(len(recorder.notes), 1, recorder.notes)
        self.assertIn('md verdict not judged', recorder.notes[0])
        # The optimization path has its own summary banner and must not
        # gain this note.
        recorder = _SummaryRecorder()
        self._run(jobtype='optimize', logger=recorder)
        self.assertEqual(recorder.notes, [])

    def test_converged_summary_logged(self):
        recorder = _SummaryRecorder()
        self._run(opt_converge=True, logger=recorder)
        summary = recorder.summary
        self.assertIsNotNone(summary)
        self.assertTrue(summary['converged'])
        self.assertEqual(summary['steps'], 1)
        self.assertIn('Skill-driven', summary['label'])

    def test_failed_summary_logged(self):
        recorder = _SummaryRecorder()
        self._run(opt_converge=False, logger=recorder)
        summary = recorder.summary
        self.assertIsNotNone(summary)
        self.assertFalse(summary['converged'])
        self.assertIn('did not converge', summary['message'])

    def test_no_log_step_call(self):
        # The run-log result block is written once by the pmd tail
        # log_result for non-iterative workflows; SingleShot must
        # not emit a step block itself (would duplicate the block).
        recorder = _SummaryRecorder()
        self._run(logger=recorder)
        self.assertEqual(recorder.log_step_calls, 0)


class TestGetOptimizerFrameworkOnly(unittest.TestCase):
    """Test that get_optimizer resolves framework optimizers only."""

    def test_framework_name_unchanged(self):
        from pyrai3md.workflow.optimizer import (
            MinimaOptimizer, get_optimizer,
        )
        optimizer = get_optimizer('minima')
        self.assertIsInstance(optimizer, MinimaOptimizer)

    def test_rejects_skill_keyword(self):
        from pyrai3md.workflow.optimizer import get_optimizer
        with self.assertRaises(ValueError):
            get_optimizer('skill')

    def test_rejects_skill_name(self):
        from pyrai3md.workflow.optimizer import get_optimizer
        with self.assertRaises(ValueError):
            get_optimizer('pyscf')

    def test_optimize_workflow_rejects_skill_method(self):
        # optimize.method is a framework algorithm keyword only;
        # constructing the Optimize workflow with 'skill' fails.
        from pyrai3md.workflow.optimize import Optimize
        calc = _SkillMockCalculator()
        state = _make_state()
        state.pmd.optimize.opt_method = 'skill'
        calc.state = state
        with self.assertRaises(ValueError):
            Optimize(calculator=calc)




class TestMethodReadingsForLogAndStatus(unittest.TestCase):
    """What each method adds beyond the shared convergence report."""

    def test_the_base_optimizer_adds_nothing(self):
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        optimizer = MinimaOptimizer()
        self.assertEqual(optimizer.status_method_columns(), ())
        self.assertEqual(optimizer.status_method_values({}), ())
        self.assertEqual(optimizer.method_details({}), {})

    def test_a_variable_cell_run_adds_the_frozen_ride(self):
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        optimizer = MinimaOptimizer(cell_variables=True)
        self.assertEqual(optimizer.status_method_columns(), ('frozen_ride',))
        self.assertEqual(
            optimizer.status_method_values({'Frozen Ride Max': 0.2}), (0.2,))
        self.assertTrue(np.isnan(optimizer.status_method_values({})[0]))

    def test_the_saddle_reports_its_curvature_readings(self):
        from pyrai3md.workflow.optimizer.ts import TransitionStateOptimizer

        optimizer = TransitionStateOptimizer()
        self.assertEqual(optimizer.status_method_columns(),
                         ('neg_modes', 'mode_eig'))
        self.assertEqual(
            optimizer.status_method_values(
                {'Negative Modes (model)': 1.0,
                 'Mode Eigenvalue (model)': -0.02}),
            (1.0, -0.02))

    def test_the_crossing_reports_gap_and_coupling(self):
        from pyrai3md.workflow.optimizer.meci import MECIOptimizer

        optimizer = MECIOptimizer()
        optimizer.last_crossing = {'gap_norm': 0.02, 'coupling_norm': None,
                                   'mean_energy': -1.4}
        self.assertEqual(
            optimizer.status_method_columns(),
            ('cross_gap', 'branch_dirs', 'gap_norm', 'coupling_norm'))
        values = optimizer.status_method_values(
            {'Crossing Gap': 0.01, 'Branching Directions': 2})
        self.assertEqual(values[:3], (0.01, 2.0, 0.02))
        self.assertTrue(np.isnan(values[3]))
        details = optimizer.method_details({})
        self.assertEqual(details['Gap Norm'], 0.02)
        self.assertIsNone(details['Coupling Norm'])
        self.assertEqual(details['Mean Energy'], -1.4)

    def test_the_band_reports_its_barrier_and_image(self):
        from pyrai3md.workflow.optimizer.neb import NEBOptimizer

        details = NEBOptimizer().method_details(
            {'barrier': 1.2, 'image': 3, 'climb': 1})
        self.assertEqual(details['Barrier'], 1.2)
        self.assertEqual(details['Image'], 3)
        self.assertEqual(details['Climb'], 1)


class TestAnAssembledSurfaceIsTheRequestedState(unittest.TestCase):
    """A composite evaluation reports the state it was asked to assemble.

    A multi-center excited-state minimization names per-center state 1, and the
    composite assembles exactly that surface while recording the per-center
    configuration in the same result. Its energy is therefore a single number
    that already IS the requested state, whatever index that state carries in
    each center's ladder. Reading that single number as an index into itself
    (`energy[active_state]`) is what raised `index 1 is out of bounds for axis 0
    with size 1` on the first multi-center excited-state run (CHANGELOG #459).
    """

    def test_the_assembled_energy_is_read_whatever_the_index_is(self):
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        optimizer = MinimaOptimizer(active_state=1)
        energy, gradient = optimizer._extract_energy_gradient({
            'energy': -152.45400421,
            'gradient': np.zeros((2, 3)).tolist(),
            'active_configuration': (1, 1)})
        self.assertEqual(energy, -152.45400421)
        np.testing.assert_array_equal(gradient, np.zeros((2, 3)))

    def test_a_plain_result_still_indexes_its_state_axis(self):
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        optimizer = MinimaOptimizer(active_state=1)
        energy, gradient = optimizer._extract_energy_gradient({
            'energy': [-1.0, -0.5],
            'gradient': np.zeros((2, 2, 3)).tolist()})
        self.assertEqual(energy, -0.5)
        self.assertEqual(gradient.shape, (2, 3))

    def test_a_two_surface_composite_still_indexes_its_pair_axis(self):
        # A crossing search over a composite gets two assembled surfaces (one
        # per member of each center's pair) and picks one by index; the
        # single-surface rule above must not swallow that case.
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        optimizer = MinimaOptimizer(active_state=1)
        energy, gradient = optimizer._extract_energy_gradient({
            'energy': [-152.53, -152.45],
            'gradient': np.zeros((2, 2, 3)).tolist(),
            'active_configuration': (1, 1)})
        self.assertEqual(energy, -152.45)
        self.assertEqual(gradient.shape, (2, 3))

    def test_a_short_state_axis_without_the_composite_marker_still_refuses(self):
        # The marker is what says "this is one assembled surface"; a plain run
        # that names state 1 while its provider returned one state keeps the
        # loud failure instead of silently reading state 0.
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        optimizer = MinimaOptimizer(active_state=1)
        with self.assertRaises(IndexError):
            optimizer._extract_energy_gradient({
                'energy': [-1.0], 'gradient': np.zeros((1, 2, 3)).tolist()})


class TestVariableCellStepKeepsTheCellSpellings(_IsolatedCwdTestCase):
    """One box, three spellings: the lattice, the six parameters, the volume.

    A variable-cell step writes `lattice_vector`; the six cell parameters and
    the volume are that same box read another way. A state whose lattice moved
    while the other two kept the pre-strain numbers shows two boxes at once to
    every reader that names the cell through them - the &cell block of the run
    log, an xyz frame's `Lattice=` fallback, the volume column of the status
    record. The measured multi-center cell relaxation (CHANGELOG #455) printed
    a strained lattice beside the starting a/b/c and volume, and that is the
    shape of the failure: not a wrong box under the step, but a stale second
    spelling of it.
    """

    EDGE = 8.399091

    def _state(self):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'cell_spellings'
        config.system.atoms = ['O', 'H', 'H']
        config.system.natom = 3
        config.system.mass = [16.0, 1.0, 1.0]
        config.system.nstate = 1
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.pbc = [1, 1, 1]
        config.system.lattice_vector = [[self.EDGE, 0.0, 0.0],
                                        [0.0, self.EDGE, 0.0],
                                        [0.0, 0.0, self.EDGE]]
        config.system.cell = [self.EDGE] * 3 + [90.0, 90.0, 90.0]
        config.system.volume = self.EDGE ** 3
        config.system.coordinates = [[3.0, 4.0, 5.0], [3.9, 4.0, 5.0],
                                     [2.7, 4.6, 5.0]]
        config.optimize.optimize_cell = True
        state = PyRAIState(runtime=RuntimeState(total_steps=4))
        state.pmd = config
        # Zero nuclear gradient, non-zero isotropic stress: the only degree of
        # freedom this step can move is the box, so a reading that changed
        # belongs to the cell and not to the atoms.
        state.set_result({
            'energy': [-152.5],
            'gradient': np.zeros((1, 3, 3)).tolist(),
            'stress_tensor': (np.eye(3) * 6.0e-5).reshape(1, 3, 3).tolist()})
        return state

    def test_a_cell_step_moves_the_six_parameters_with_the_lattice(self):
        from pyrai3md.utilities.lattice import cell_volume, matrix_to_cell6
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer

        state = self._state()
        starting = [float(value) for value in state.pmd.system.cell]
        optimizer = MinimaOptimizer(maxiter=2, cell_variables=True)
        optimizer.note_evaluation(state)
        optimizer.run(state)
        system = state.pmd.system
        lattice = np.asarray(system.lattice_vector, dtype=np.float64)
        six = [float(value) for value in matrix_to_cell6(lattice)]
        moved = [float(value) for value in system.cell]
        # The step has to have strained the box, or the two assertions below
        # pass on a state nobody moved.
        self.assertGreater(abs(moved[0] - starting[0]), 1.0e-4,
                           msg='this step left the box where it found it; the '
                               'comparison below would be vacuous: %r -> %r'
                               % (starting, moved))
        np.testing.assert_allclose(moved, six, rtol=0.0, atol=1.0e-12)
        self.assertAlmostEqual(float(system.volume), cell_volume(lattice),
                               delta=1.0e-9)


if __name__ == '__main__':
    unittest.main()
