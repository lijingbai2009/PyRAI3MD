"""
======================================================================
The force-constant link end to end: the numerical batch, the analytical
route, the refusals between them, and the reading of the result.

The engine-free instrument is a harmonic model: energy 0.5 x^T H0 x with a
known symmetric H0. For a quadratic the central difference of the gradient is
exact, not convergent, so the assembled matrix has to reproduce H0 to
roundoff -- a criterion that a real backend, whose fourth derivative is not
zero, cannot offer. The one PySCF test kept here is the convergence of the
finite difference toward the analytical matrix, which is the statement that
the step size is a distance while the gradients are Hartree per Bohr.

The linear triatomic is there because it is the case the retired analytical
module got wrong: fixing six rigid directions costs it one real bending mode,
and the rank-based projection has to hand back four modes where the old
arithmetic returned three.

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.utilities.print_results import _format_normal_modes

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.build import build_state
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
from pyrai3md.workflow.hessian import Hessian
from pyrai3md.workflow.frequency import vibration


class HarmonicBackend:
    """A provider whose energy is exactly 0.5 x^T H0 x, x in Bohr."""

    def __init__(self, matrix, origin=None):
        self.h0 = np.asarray(matrix, dtype=np.float64)
        self.origin = (np.zeros_like(self.h0[0]) if origin is None
                       else np.asarray(origin, dtype=np.float64))
        self.calls = 0
        self.geometries = []

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        del atoms, point_charges, target_states, cell
        self.calls += 1
        x = (np.asarray(coordinates, dtype=np.float64).reshape(-1)
             * ANGSTROM_TO_BOHR - self.origin)
        # The derivative of that energy with respect to the Bohr coordinate is
        # already Hartree per Bohr, the unit every backend reports gradients in:
        # multiplying it by the Bohr conversion again inflates the whole
        # assembled matrix by 1.89, which is what this harness measured at first.
        gradient = (self.h0 @ x).reshape(-1, 3)
        energy = 0.5 * float(x @ self.h0 @ x)
        return {'energy': energy, 'gradient': [list(row) for row in gradient]}


def pair_matrix(natom, bonds, k):
    """A central-force force-constant matrix: exact, translation invariant."""
    size = 3 * natom
    matrix = np.zeros((size, size))
    for left, right in bonds:
        # Every Cartesian direction carries the same pair constant. A real
        # bond has no transverse force constant of its own, but leaving those
        # blocks out would make the bending modes of a linear chain exactly
        # zero for a reason that has nothing to do with what is being tested.
        for axis in range(3):
            for i, j, sign in ((left, left, 1), (right, right, 1),
                               (left, right, -1), (right, left, -1)):
                matrix[3 * i + axis, 3 * j + axis] += sign * k
    return matrix


def _state(coordinates, masses, natom):
    config = PyRAIConfig()
    config.control.jobtype = 'num_hess'
    config.control.driver = 'main'
    config.control.title = 'harness'
    config.system.atoms = ['C', 'O', 'O'][:natom]
    config.system.natom = natom
    config.system.mass = list(masses)
    config.system.coordinates = [list(row) for row in coordinates]
    config.system.nstate = 1
    config.system.charges = 0
    config.system.multiplicity = 1
    state = PyRAIState(runtime=RuntimeState(total_steps=2))
    state.pmd = config
    return state


def _executor(backend, drop_gradient=False):
    """The child-job entry the batch fan-out calls, on the injected model.

    `dispatch` hands every displaced geometry to a module-level entry that
    would build a real Calculator from the child state; a unit test that owns
    no engine replaces it here, so the batch runs through the same
    `collect`/`postprocess` code the engine path does.
    """
    from pyrai3md.data.runtime.result import Result

    def run_child(child_state, runtime_idx=None):
        del runtime_idx
        data = backend.calculate(child_state.atoms, child_state.coordinates)
        if drop_gradient:
            data = {'energy': data['energy']}
        return Result(data=data)

    return run_child


class _Calculator:
    """The slice of Calculator the workflow uses, with an injected provider."""

    def __init__(self, state, backend, extra=None):
        self.state = state
        self.backend = backend
        self.extra = extra or {}
        self.syncs = 0

    def run(self, target_states=None):
        data = self.backend.calculate(self.state.atoms,
                                      self.state.coordinates)
        data = dict(data)
        data.update(self.extra)
        self.state.set_result(data)
        return dict(self.state.result)

    def get_state(self):
        return self.state

    def sync(self):
        self.syncs += 1

    def update(self, updates):
        """The Calculator state-ownership interface the workflow writes to."""
        if 'results' in updates:
            self.state.set_result(updates['results'])
        elif 'system' in updates:
            self.state.update_system(updates['system'])


class _Isolated(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._dir = tempfile.mkdtemp()
        os.chdir(self._dir)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._dir, ignore_errors=True)


class TestNumericalBatch(_Isolated):
    """The displaced-gradient route, on a model it must reproduce exactly."""

    LINEAR = [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]
    MASSES = [12.011, 15.999, 15.999]

    def _workflow(self, coordinates, masses, matrix, **keywords):
        backend = HarmonicBackend(matrix)
        state = _state(coordinates, masses, len(coordinates))
        workflow = Hessian(calculator=_Calculator(state, backend),
                           **keywords)
        workflow.local_executor = _executor(backend)
        return workflow, state, backend

    def test_a_quadratic_is_reproduced_without_truncation_error(self):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        workflow, state, _backend = self._workflow(
            self.LINEAR, self.MASSES, matrix, step_size=0.01)
        final = workflow.run()
        got = np.asarray(final.result.to_dict()['hessian'],
                         dtype=np.float64)[0]
        self.assertLess(float(np.abs(got - matrix).max()), 1.0e-9,
                        'a central difference of a quadratic is exact')
        self.assertEqual(final.runtime.status, 'completed')

    def test_the_matrix_keeps_translational_invariance(self):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        workflow, _state, _backend = self._workflow(
            self.LINEAR, self.MASSES, matrix, step_size=0.01)
        final = workflow.run()
        got = np.asarray(final.result.to_dict()['hessian'],
                         dtype=np.float64)[0]
        self.assertLess(float(np.abs(got.sum(axis=0)).max()), 1.0e-9)
        self.assertLess(float(np.abs(got - got.T).max()), 1.0e-12)

    def test_a_linear_structure_keeps_its_four_modes(self):
        # 3 atoms, 9 components, five rigid directions: the retired fixed-six
        # projection returned three modes and lost a bending one with them.
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        modes = vibration(np.asarray(self.LINEAR) * ANGSTROM_TO_BOHR,
                          np.asarray(self.MASSES), matrix)
        self.assertEqual(modes['rigid_removed'], 5)
        self.assertTrue(modes['linear'])
        self.assertEqual(modes['frequencies'].shape[0], 4)
        self.assertEqual(int(np.count_nonzero(modes['frequencies'] == 0.0)),
                         0, 'the four are vibrations, not leftovers')

    def test_a_bent_triatomic_keeps_three(self):
        bent = [[0.0, 0.0, 0.1173], [0.0, 0.7575, -0.4657],
                [0.0, -0.7575, -0.4657]]
        matrix = pair_matrix(3, [(0, 1), (1, 2), (0, 2)], 0.04)
        modes = vibration(np.asarray(bent) * ANGSTROM_TO_BOHR,
                          np.asarray([15.999, 1.008, 1.008]), matrix)
        self.assertEqual(modes['rigid_removed'], 6)
        self.assertEqual(modes['frequencies'].shape[0], 3)

    def test_the_task_count_follows_the_mobile_selection(self):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        workflow, state, backend = self._workflow(
            self.LINEAR, self.MASSES, matrix, step_size=0.01,
            atoms=[1, 2])            # 1-based: the central atom and one O
        final = workflow.run()
        result = final.result.to_dict()
        self.assertEqual(sorted(result['hessian_atoms']), [0, 1])
        held = np.asarray(result['hessian'], dtype=np.float64)[0]
        self.assertEqual(float(np.abs(held[6:, :]).max()), 0.0,
                         'a held atom must not carry force constants')
        self.assertEqual(float(np.abs(held[:, 6:]).max()), 0.0)
        self.assertLessEqual(backend.calls, 12,
                            'two displacements per mobile component only')

    def test_a_missing_displaced_gradient_fails_loudly(self):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        backend = HarmonicBackend(matrix)
        state = _state(self.LINEAR, self.MASSES, 3)

        class Gradientless(_Calculator):
            def run(self, target_states=None):
                data = super().run(target_states)
                data.pop('gradient')
                self.state.set_result({'energy': data.get('energy')})
                return dict(self.state.result)

        workflow = Hessian(calculator=Gradientless(state, backend),
                           step_size=0.01)
        workflow.local_executor = _executor(backend, drop_gradient=True)
        with self.assertRaises(ValueError) as ctx:
            workflow.run()
        self.assertIn('gradient', str(ctx.exception))
        self.assertEqual(state.runtime.status, 'failed')


class TestAnalyticalRoute(_Isolated):
    """The route that reads a matrix the backend already produced."""

    def _run(self, source, matrix=None):
        state = _state([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0],
                        [-1.16, 0.0, 0.0]], [12.011, 15.999, 15.999], 3)
        backend = HarmonicBackend(pair_matrix(3, [(0, 1), (1, 2)], 0.05))
        extra = {}
        if matrix is not None:
            extra['hessian'] = [matrix.tolist()]
        calculator = _Calculator(state, backend, extra)
        workflow = Hessian(calculator=calculator, source=source,
                           analysis=False, thermo=False)
        return workflow, calculator, state

    def test_the_backend_matrix_is_used_and_nothing_is_displaced(self):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        workflow, calculator, _state = self._run('analytic', matrix)
        final = workflow.run()
        self.assertEqual(calculator.backend.calls, 1)
        got = np.asarray(final.result.to_dict()['hessian'],
                         dtype=np.float64)[0]
        np.testing.assert_allclose(got, matrix, atol=1.0e-12)

    def test_a_refused_route_names_the_missing_thing(self):
        workflow, _calculator, state = self._run('analytic', matrix=None)
        with self.assertRaises(ValueError) as ctx:
            workflow.run()
        self.assertIn('analytical Hessian', str(ctx.exception))
        self.assertEqual(state.runtime.status, 'failed')

    def test_auto_drops_to_the_batch_and_says_so(self):
        from pyrai3md.kernel.logger import RunLogger
        workflow, calculator, state = self._run('auto', matrix=None)
        state.logger = RunLogger(state)
        workflow.local_executor = lambda child, runtime_idx=None: None
        with self.assertRaises(ValueError):
            workflow.run()
        text = open('harness.log').read()
        self.assertIn('numerical displacement route is taken', text,
                      'a route switch has to be readable in the log')


class TestCurvatureProducer(_Isolated):
    """`&hessian hess_driver`: who produces the matrix, and which pairs exist.

    The producer and the method are two questions, and the pairings are not all
    real. The one worth a test of its own is `skill` with `numerical`: an engine
    that displaces its own atoms (ORCA's NumFreq, Molcas' MCKINLEY) is the
    backend producing a numerically differentiated matrix, so from this side it
    is one evaluation that answers with a matrix, and a workflow that read
    `numerical` as the framework's name would silently displace the geometry a
    second time instead.
    """

    def _workflow(self, matrix=None, jobtype='num_hess', **section):
        config = PyRAIConfig()
        config.control.jobtype = jobtype
        config.control.driver = 'main'
        config.control.title = 'producer'
        config.system.atoms = ['C', 'O', 'O']
        config.system.natom = 3
        config.system.mass = [12.011, 15.999, 15.999]
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0],
                                     [-1.16, 0.0, 0.0]]
        config.system.nstate = 1
        for key, value in section.items():
            setattr(config.hessian, key, value)
        state = PyRAIState(runtime=RuntimeState(total_steps=2))
        state.pmd = config
        backend = HarmonicBackend(pair_matrix(3, [(0, 1), (1, 2)], 0.05))
        extra = {}
        if matrix is not None:
            extra['hessian'] = [np.asarray(matrix).tolist()]
        calculator = _Calculator(state, backend, extra)
        return Hessian(calculator=calculator, analysis=False), state, calculator

    def test_the_named_producer_outranks_the_numerical_token(self):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        workflow, state, calculator = self._workflow(
            matrix, hess_driver='skill', hess_mode='numerical')
        self.assertEqual(workflow.driver, 'skill')
        self.assertEqual(workflow.source, 'numerical')
        # The other route would fan out 18 displaced geometries through the
        # executor, so recording here says what did not happen rather than
        # trusting the call count alone.
        displaced = []
        workflow.local_executor = (
            lambda child, runtime_idx=None: displaced.append(child))
        workflow.run()
        self.assertEqual(calculator.backend.calls, 1,
                         'one evaluation asked of the backend')
        self.assertEqual(displaced, [], 'and no displacement of its own')
        np.testing.assert_allclose(workflow.matrix, matrix, atol=1.0e-12)

    def test_the_driver_is_read_through_the_state_whitelist(self):
        # `hessian_params` is a whitelist: a field added to the model but not to
        # that defaults dict reads as its default however the input spelled it.
        _workflow, state, _calculator = self._workflow(hess_driver='skill')
        self.assertEqual(state.hessian_params['hess_driver'], 'skill')

    def test_the_framework_producer_needs_the_numerical_method(self):
        # The negative control for the check above and the pair refusal below:
        # the legal pairing must construct, or the guard would prove nothing.
        workflow, _state, _calculator = self._workflow(
            hess_driver='main', hess_mode='numerical')
        self.assertEqual(workflow.driver, 'main')

    def test_an_unnamed_pairing_is_left_to_the_mode(self):
        # `num_hess` names the framework, but a declared analytical mode is a
        # question about a backend that can answer it: the mode decides the
        # producer when the input did not name one.
        workflow, _state, _calculator = self._workflow(hess_mode='analytic')
        self.assertEqual(workflow.driver, 'skill')

    def test_a_loop_asks_the_backend_for_the_matrix_as_its_own_job(self):
        """`hess_driver: skill` inside an optimization run.

        The loop is evaluated for energy and gradient every step, so the matrix
        is not in the evaluation this run performs; the request goes out as one
        dispatched job on the geometry the search starts on, and the run's own
        configuration is left exactly as the planning layer wrote it.
        """
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        workflow, state, calculator = self._workflow(
            jobtype='optimize', hess_driver='skill', hess_mode='analytic')
        asked = []

        def answer(child_state, runtime_idx=None):
            asked.append((str(child_state.pmd.control.jobtype),
                          str(child_state.pmd.hessian.hess_mode),
                          bool(child_state.pmd.hessian.hess_vibration)))
            return {'hessian': [matrix.tolist()]}

        workflow.local_executor = answer
        final = workflow.run()
        self.assertEqual(
            asked, [('frequency', 'analytic', False)],
            'one job, spelling the matrix, with its own route pinned so an '
            'engine that cannot answer refuses instead of displacing here')
        self.assertEqual(calculator.backend.calls, 1,
                         'the run itself was still evaluated once, for energy '
                         'and gradient')
        self.assertEqual(str(state.pmd.control.jobtype), 'optimize',
                         'the request is a dispatched unit, not a rewrite')
        np.testing.assert_allclose(workflow.matrix, matrix, atol=1.0e-12)
        got = np.asarray(final.result.to_dict()['hessian'], dtype=np.float64)
        self.assertEqual(got.shape, (1, 9, 9),
                         'the result keeps the state axis every route writes')

    def test_a_backend_that_cannot_answer_names_the_other_producer(self):
        workflow, state, _calculator = self._workflow(
            jobtype='optimize', hess_driver='skill', hess_mode='analytic')
        workflow.local_executor = lambda child, runtime_idx=None: {
            'energy': -1.0}
        with self.assertRaises(ValueError) as ctx:
            workflow.run()
        self.assertIn("hess_driver 'main'", str(ctx.exception))
        self.assertEqual(state.runtime.status, 'failed',
                         'a job that refused is marked, not a silent success')


class TestCurvaturePairRefusals(unittest.TestCase):
    """The owner/method pairs that have no producer behind them."""

    def _config(self, driver, mode):
        return {
            'control': {'jobtype': 'hessian', 'driver': 'main',
                        'title': 'pair'},
            'system': {'atoms': ['C', 'O'], 'natom': 2,
                       'mass': [12.0, 16.0],
                       'coordinates': [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0]],
                       'nstate': 1, 'charges': 0, 'multiplicity': 1},
            'hessian': {'hess_driver': driver, 'hess_mode': mode},
        }

    def test_the_framework_cannot_be_asked_for_an_analytical_matrix(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError) as ctx:
            PyRAIConfig(**self._config('main', 'analytic'))
        self.assertIn('hess_driver', str(ctx.exception))

    def test_the_framework_cannot_be_asked_with_the_owner_unset(self):
        # `auto` is the statement that no producer has been settled on; it
        # cannot stand beside a named one.
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PyRAIConfig(**self._config('main', 'auto'))

    def test_an_unknown_producer_is_named_back(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError) as ctx:
            PyRAIConfig(**self._config('hybrid', 'numerical'))
        self.assertIn('main, skill', str(ctx.exception))

    def test_the_workflow_refuses_the_pair_it_could_not_serve(self):
        # The schema is the gate for an input, but a programmatic caller can
        # name the same contradiction, and building the wrong batch is cheaper
        # to refuse here than to run.
        from pyrai3md.workflow.hessian import Hessian as WorkflowHessian
        state = _state([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0]], [12.0, 16.0], 2)
        backend = HarmonicBackend(pair_matrix(2, [(0, 1)], 0.05))
        with self.assertRaises(ValueError) as ctx:
            WorkflowHessian(calculator=_Calculator(state, backend),
                            driver='main', source='analytic')
        self.assertIn('analytical', str(ctx.exception))


class TestKeywordRouting(_Isolated):
    """The section keywords reach the workflow instead of being dropped."""

    def _workflow(self, jobtype, **section):
        config = PyRAIConfig()
        config.control.jobtype = jobtype
        config.control.driver = 'main'
        config.control.title = 'routing'
        config.system.atoms = ['C', 'O', 'O']
        config.system.natom = 3
        config.system.mass = [12.011, 15.999, 15.999]
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0],
                                     [-1.16, 0.0, 0.0]]
        config.system.nstate = 1
        for key, value in section.items():
            setattr(config.hessian, key, value)
        state = PyRAIState(runtime=RuntimeState(total_steps=2))
        state.pmd = config
        backend = HarmonicBackend(pair_matrix(3, [(0, 1), (1, 2)], 0.05))
        return Hessian(calculator=_Calculator(state, backend))

    def test_the_step_and_selection_are_read_from_the_section(self):
        workflow = self._workflow('num_hess', hess_step=0.02,
                                 hess_atoms=[1, 2], hess_thermo=False)
        self.assertEqual(workflow.step_size, 0.02)
        self.assertFalse(workflow.thermo)
        self.assertEqual(workflow.atoms, [1, 2])

    def test_the_tokens_choose_the_route_and_the_analysis(self):
        from pyrai3md.workflow.hessian import _requested_source, _wants_analysis
        for jobtype, source in (('num_hess', 'numerical'),
                                ('num_freq', 'numerical'),
                                ('hessian', 'auto'),
                                ('frequency', 'auto')):
            workflow = self._workflow(jobtype)
            state = workflow.calculator.get_state()
            self.assertEqual(_requested_source(state), source, jobtype)
            self.assertEqual(_wants_analysis(state),
                             jobtype in ('num_freq', 'frequency'), jobtype)

    def test_a_declared_mode_outranks_the_token(self):
        # The token chooses the route, and the input's own `mode` chooses over
        # it: `num_hess` asked analytically is a legitimate question about a
        # backend that can answer it. A *default* in that field must not do the
        # same, which is what the None default is for.
        workflow = self._workflow('num_hess', hess_mode='analytic')
        self.assertEqual(workflow.source, 'analytic')
        untouched = self._workflow('num_hess')
        self.assertEqual(untouched.source, 'numerical')

    def test_a_section_keyword_does_not_land_in_the_dynamics_model(self):
        # The reason these keywords carry the `hess_` prefix: `step`,
        # `temperature` and `pressure` are also flat names in `&md`, and the
        # flattened routing gave the Hessian section's values to the dynamics
        # submodel, which reads as an integer error for `step` and as a
        # silently ignored keyword for the others.
        from pyrai3md.schema.adapter import _flatten_sections
        flat = _flatten_sections({'hessian': {'hess_step': 0.02,
                                              'hess_temperature': 300.0}})
        self.assertEqual(flat['hessian']['hess_step'], 0.02)
        self.assertNotIn('step', flat.get('dynamics', {}))


class TestMoldenRecord(_Isolated):
    """The record the initial-condition sampler reads, round tripped."""

    def test_the_written_record_comes_back_at_the_sampling_normalization(self):
        from pyrai3md.workflow.sampling import load_frequency_data
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        coordinates = [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]
        masses = [12.011, 15.999, 15.999]
        state = _state(coordinates, masses, 3)
        state.pmd.control.title = 'harness'
        backend = HarmonicBackend(matrix)
        workflow = Hessian(calculator=_Calculator(state, backend),
                           save_molden=True, thermo=False)
        workflow.local_executor = _executor(backend)
        workflow.run()
        data = load_frequency_data('harness', 'molden')
        self.assertEqual(data['nfreq'], 4)
        norm = np.einsum('kij,i->k', data['vib'] ** 2,
                         data['amass'].reshape(-1))
        np.testing.assert_allclose(norm, np.ones(4), atol=1.0e-6,
                                   err_msg='the sampler assumes sum m v^2 = 1')


class TestStepSizeIsABohrDistance(_Isolated):
    """What the displacement unit actually buys.

    The two claims that a wrong factor would break, both exact on a quadratic:
    the assembled matrix equals the model's own force constants, and it does
    not move when the displacement does. A missing Bohr factor leaves a
    constant 1.89 scale error on every entry; a divisor built from a gradient
    that is not Hartree per Bohr moves the matrix with the step size.
    """

    def _assembled(self, step):
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)
        backend = HarmonicBackend(matrix)
        state = _state(TestNumericalBatch.LINEAR, TestNumericalBatch.MASSES, 3)
        workflow = Hessian(calculator=_Calculator(state, backend),
                           step_size=step, analysis=False, thermo=False)
        workflow.local_executor = _executor(backend)
        final = workflow.run()
        return (np.asarray(final.result.to_dict()['hessian'],
                            dtype=np.float64)[0], matrix)

    def test_the_result_is_independent_of_the_displacement(self):
        coarse, truth = self._assembled(0.02)
        fine, _truth = self._assembled(0.005)
        np.testing.assert_allclose(coarse, truth, atol=1.0e-9)
        np.testing.assert_allclose(fine, truth, atol=1.0e-9)
        self.assertLess(float(np.abs(coarse - fine).max()), 1.0e-12)

    def test_a_gradient_in_the_wrong_unit_would_move_the_matrix(self):
        # The negative control for the assertion above: scale the reported
        # gradient and the assembled matrix follows it, so the agreement at a
        # factor of one is evidence and not an artifact of the comparison.
        matrix = pair_matrix(3, [(0, 1), (1, 2)], 0.05)

        class MisScaled(HarmonicBackend):
            def calculate(self, *args, **kwargs):
                data = super().calculate(*args, **kwargs)
                data['gradient'] = [[row[i] * ANGSTROM_TO_BOHR
                                     for i in range(3)] for row in
                                    data['gradient']]
                return data

        backend = MisScaled(matrix)
        state = _state(TestNumericalBatch.LINEAR, TestNumericalBatch.MASSES, 3)
        workflow = Hessian(calculator=_Calculator(state, backend),
                          step_size=0.01, analysis=False, thermo=False)
        workflow.local_executor = _executor(backend)
        got = np.asarray(workflow.run().result.to_dict()['hessian'],
                         dtype=np.float64)[0]
        self.assertGreater(float(np.abs(got - matrix).max()), 1.0e-3)


class TestNormalModePrintShape(unittest.TestCase):
    """The result printer reports a mode array it cannot read; it does not die.

    The block runs on the way out of a finished calculation, so an
    unexpected shape there used to abort a run the engine had completed:
    OpenQP's `hess.json` writes modes as (mode, 3 * atom) -- (12, 18) for
    acetylene -- and the loop below that fold read a 2-D array as nested
    states and modes and raised `IndexError: tuple index out of range` on
    `mode.shape[0]` (round tmp_20261011_085800, openqp
    `hf_frequency_input`/`mrsf_frequency_input`). The skill folds its own
    export now; this is the second line of defence, stated rather than
    silent, because a modes block that vanished without a word would be a
    worse report than one that says what it was given.
    """

    _ATOMS = ['C', 'C', 'H', 'H', 'H', 'H']

    def test_the_folded_shape_prints_every_mode(self):
        block = _format_normal_modes(
            np.zeros((1, 12, 6, 3)), self._ATOMS, 4)
        self.assertIn('mode    1', block)
        self.assertIn('mode   12', block)

    def test_an_unfolded_array_is_reported_not_raised(self):
        for shape in ((12, 18), (18, 12)):
            with self.subTest(shape=shape):
                block = _format_normal_modes(
                    np.zeros(shape), self._ATOMS, 4)
                self.assertIn('unreadable shape', block)
                self.assertIn(str(shape), block)

    def test_a_low_verbosity_still_prints_nothing(self):
        self.assertEqual(
            _format_normal_modes(np.zeros((12, 18)), self._ATOMS, 1), '')


if __name__ == '__main__':
    unittest.main()
