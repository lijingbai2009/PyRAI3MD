"""
======================================================================
Test suite for the workflow layer (pyrai3md/workflow/).

Tests cover:
- Module imports
- Workflow base class interface
- WORKFLOW_REGISTRY, JOBTYPE_MAP, resolve_workflow
- Optimize workflow run() with mock calculator
- AIMD workflow run() with mock calculator
- Hessian and Scan workflow construction
- Verification that workflows do NOT call external programs directly

Jul 30 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock result dict matching backend.calculate() return format.
MOCK_ENERGY = -75.123456
# Ensemble file name the runtime resolves for a state with no title.
ENSEMBLE_NAME = 'pyrai3md.init.xyz'

MOCK_GRADIENT = np.array([[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]])


class MockResult:
    """Minimal dict-like Result object for workflow tests."""

    def __init__(self, energy=None, gradient=None, populations=None):
        self._data = {}
        if energy is not None:
            self._data['energy'] = energy
        if gradient is not None:
            self._data['gradient'] = gradient
        if populations is not None:
            self._data['populations'] = populations

    def __getitem__(self, key):
        return self._data.get(key)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()


class MockCalculator:
    """Mock calculator that returns fixed results."""

    def __init__(self, energy=MOCK_ENERGY, gradient=None):
        self._energy = energy
        self._gradient = gradient
        self.call_count = 0
        self.run_calls = []  # target_states received per run()
        self.state = None  # Set by test or execute()

    def run(self, target_states=None):
        self.call_count += 1
        self.run_calls.append(target_states)
        # Mimic the real Calculator: write results back into pmd
        # through the set_result contract (numpy arrays become
        # lists so pmd stays serializable).
        if self.state is not None and self.state.pmd is not None:
            results = {'energy': self._energy}
            if self._gradient is not None:
                results['gradient'] = self._gradient
            self.state.set_result(results)
        return MockResult(energy=self._energy, gradient=self._gradient)

    def get_state(self):
        # Mimic the real Calculator state ownership interface
        return self.state

    def update(self, updates):
        # Mimic the real Calculator state update interface
        if self.state is None or not updates:
            return self.state
        for key, value in updates.items():
            if key == 'system' and isinstance(value, dict):
                self.state.update_system(value)
            elif key == 'results':
                self.state.set_result(value)
            else:
                setattr(self.state, key, value)
        return self.state

    def sync(self):
        # Mimic the real Calculator.sync() placeholder
        pass


class MockRunLogger:
    """Recording stub for the RunLogger optimization log channels."""

    def __init__(self):
        self.steps = []
        self.summary = None
        self.setup = None

    def log_optimize_setup(self, setup):
        self.setup = dict(setup)

    def log_step(self, step, snapshot, thresholds=None):
        del step, snapshot, thresholds

    def log_optimize_step(self, step, report, converged,
                          thresholds=None, details=None):
        self.steps.append(
            (step, dict(report), converged, dict(thresholds or {}),
             dict(details or {})))

    def log_optimize_summary(self, converged, steps, energy=None,
                             message='', label='Geometry optimization'):
        self.summary = {'converged': converged, 'steps': steps,
                        'energy': energy, 'message': message,
                        'label': label}


def _make_mock_state(total_steps=5):
    """Create a minimal PyRAIState with pmd as the data source."""
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.data.runtime.state import RuntimeState
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    config.system.natom = 2
    config.system.charges = 0
    config.system.multiplicity = 1
    config.system.nstate = 2

    state = PyRAIState(runtime=RuntimeState(total_steps=total_steps))
    state.pmd = config
    return state


class IsolatedCwdTestCase(unittest.TestCase):
    """Base test case that runs each test inside a temporary directory.

    Workflows such as Optimize write outputs (opt.xyz, opt_geom.xyz,
    opt_status.txt) through bare relative paths. Redirecting cwd keeps
    those scratch files out of the caller directory, matching the
    isolation policy of the subprocess example/skill tests.
    """

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmpdir.cleanup()


class TestWorkflowImports(unittest.TestCase):
    """Test that all workflow modules can be imported."""

    def test_import_base(self):
        from pyrai3md.workflow.base import Workflow
        self.assertTrue(callable(Workflow))

    def test_import_all_workflows(self):
        from pyrai3md.workflow import Optimize, AIMD, Hessian, Scan
        for cls in [Optimize, AIMD, Hessian, Scan]:
            self.assertTrue(callable(cls))

    def test_import_registry(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY, JOBTYPE_MAP, resolve_workflow
        self.assertIsInstance(WORKFLOW_REGISTRY, dict)
        self.assertIsInstance(JOBTYPE_MAP, dict)
        self.assertTrue(callable(resolve_workflow))


class TestWorkflowBase(IsolatedCwdTestCase):
    """Test the Workflow base class."""

    def test_default_construction(self):
        from pyrai3md.workflow.base import Workflow
        wf = Workflow()
        self.assertEqual(wf.name, '')
        self.assertIsNone(wf.calculator)
        self.assertEqual(wf.maxiter, 100)

    def test_named_construction(self):
        from pyrai3md.workflow.base import Workflow
        calc = MockCalculator()
        wf = Workflow(name='test', calculator=calc, maxiter=50)
        self.assertEqual(wf.name, 'test')
        self.assertEqual(wf.maxiter, 50)

    def test_run_runs_calculator_once(self):
        from pyrai3md.workflow.base import Workflow
        calc = MockCalculator()
        state = _make_mock_state(total_steps=1)
        calc.state = state
        wf = Workflow(name='test', calculator=calc)
        wf.run()
        # Base workflow runs the calculator exactly once
        self.assertEqual(calc.call_count, 1)
        self.assertEqual(state.runtime.status, 'completed')

    def test_check_convergence_returns_false(self):
        from pyrai3md.workflow.base import Workflow
        wf = Workflow()
        self.assertFalse(wf._check_convergence(None))

    def test_repr(self):
        from pyrai3md.workflow.base import Workflow
        wf = Workflow(name='test', maxiter=50)
        r = repr(wf)
        self.assertIn('test', r)
        self.assertIn('50', r)


class TestWorkflowRegistry(IsolatedCwdTestCase):
    """Test WORKFLOW_REGISTRY and resolution functions."""

    def test_registry_has_all_workflows(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        expected = ['optimize', 'md', 'frequency', 'scan']
        for name in expected:
            self.assertIn(name, WORKFLOW_REGISTRY)

    def test_registry_instantiates_workflow(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY, Workflow
        wf = WORKFLOW_REGISTRY['optimize']()
        self.assertIsInstance(wf, Workflow)
        self.assertEqual(wf.name, 'optimize')

    def test_resolve_workflow_sp_returns_none(self):
        from pyrai3md.workflow import resolve_workflow
        state = _make_mock_state()
        state.pmd.control.jobtype = 'energy'
        result = resolve_workflow(state)
        self.assertIsNone(result)

    def test_resolve_workflow_optimize(self):
        from pyrai3md.workflow import resolve_workflow, Optimize
        state = _make_mock_state()
        state.pmd.control.jobtype = 'optimize'
        cls = resolve_workflow(state)
        self.assertEqual(cls, Optimize)

    def test_resolve_workflow_opt_alias(self):
        from pyrai3md.workflow import resolve_workflow, Optimize
        state = _make_mock_state()
        state.pmd.control.jobtype = 'optimize'
        cls = resolve_workflow(state)
        self.assertEqual(cls, Optimize)

    def test_resolve_workflow_namd(self):
        from pyrai3md.workflow import resolve_workflow, AIMD
        state = _make_mock_state()
        state.pmd.control.jobtype = 'md'
        cls = resolve_workflow(state)
        self.assertEqual(cls, AIMD)

    def test_resolve_workflow_md_alias(self):
        from pyrai3md.workflow import resolve_workflow, AIMD
        state = _make_mock_state()
        state.pmd.control.jobtype = 'md'
        cls = resolve_workflow(state)
        self.assertEqual(cls, AIMD)

    def test_resolve_workflow_frequency(self):
        # `frequency` is the analytical force-constant request and reaches the
        # one Hessian workflow; the route is chosen inside it, not by a second
        # workflow class.
        from pyrai3md.workflow import resolve_workflow, Hessian
        state = _make_mock_state()
        state.pmd.control.jobtype = 'frequency'
        cls = resolve_workflow(state)
        self.assertEqual(cls, Hessian)

    def test_resolve_workflow_freq_alias(self):
        """The alias tokens reach the same workflow as the canonical one.

        The test used to write the canonical spelling under an alias name, so
        it proved nothing about the alias; `freq`, `hess`, `hessian`,
        `num_freq` and `num_hess` are checked at the parser and at the
        resolver here."""
        from pyrai3md.schema.validate import parse_jobtype
        from pyrai3md.workflow import resolve_workflow, Hessian
        for token, canonical in (('freq', 'frequency'),
                                 ('hess', 'hessian'),
                                 ('hessian', 'hessian'),
                                 ('numfreq', 'num_freq'),
                                 ('num_hess', 'num_hess')):
            tokens, _basic, _advanced = parse_jobtype(token)
            self.assertEqual(tokens, [canonical], token)
            state = _make_mock_state()
            state.pmd.control.jobtype = token
            self.assertIs(resolve_workflow(state), Hessian, token)

    def test_resolve_workflow_unknown_returns_none(self):
        from pyrai3md.workflow import resolve_workflow
        state = _make_mock_state()
        state.pmd.control.jobtype = 'unknown_jobtype'
        result = resolve_workflow(state)
        self.assertIsNone(result)

    def test_registry_classes_declare_the_graph_node_name(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        for registry_name, cls in sorted(WORKFLOW_REGISTRY.items()):
            declared = getattr(cls, 'name', None)
            self.assertIsInstance(
                declared, str,
                "%s (registry key '%s') declares no class-level name"
                % (cls.__name__, registry_name))
            self.assertTrue(
                declared,
                "%s (registry key '%s') declares an empty name; the "
                "planner would resolve no keyword section"
                % (cls.__name__, registry_name))

    def test_declared_name_matches_the_instance_name(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        checked = set()
        for registry_name, cls in sorted(WORKFLOW_REGISTRY.items()):
            if cls in checked:
                continue
            checked.add(cls)
            self.assertEqual(cls().name, cls.name,
                             "%s declares name '%s' on the class but "
                             "'%s' on the instance" % (
                                 cls.__name__, cls.name, cls().name))

    def test_planner_tables_have_no_unreachable_keys(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        from pyrai3md.workflow.dispatch.planner import (
            WORKFLOW_SECTIONS, WORKFLOW_NODE_DECLARATIONS,
            WORKFLOW_BASE_CHAINS,
        )
        # A node name is either a workflow class name (root runtimes)
        # or a declared stage name (multi-stage workflows). Anything
        # else in these tables can never be looked up.
        reachable = {cls.name for cls in WORKFLOW_REGISTRY.values()}
        for cls in WORKFLOW_REGISTRY.values():
            for stage in getattr(cls, 'stages', ()) or ():
                if not stage.startswith('base.'):
                    reachable.add(stage)
        for table in (WORKFLOW_SECTIONS, WORKFLOW_NODE_DECLARATIONS,
                      WORKFLOW_BASE_CHAINS):
            for key in sorted(table):
                self.assertIn(
                    key, reachable,
                    "planner table key '%s' matches no workflow node "
                    "name; its settings are dead" % key)

    def test_log_titles_are_resolvable_jobtypes(self):
        from pyrai3md.workflow import JOBTYPE_MAP
        from pyrai3md.kernel.logger import WORKFLOW_TITLES
        for key in sorted(WORKFLOW_TITLES):
            self.assertIn(key, JOBTYPE_MAP,
                          "banner title '%s' is not a jobtype token" % key)












class TestResolveTargetStates(IsolatedCwdTestCase):
    """Test the activestate/current_state -> target_states rule.

    Several tests run the AIMD workflow end to end, whose output
    writer falls back to the cwd; the isolation base keeps the
    trajectory files out of the caller directory.
    """

    def test_activestate_zero_returns_all_states(self):
        from pyrai3md.workflow.md import resolve_target_states
        self.assertEqual(resolve_target_states(0, 1, 4), [0, 1, 2, 3])

    def test_activestate_nonzero_returns_current_state_only(self):
        from pyrai3md.workflow.md import resolve_target_states
        self.assertEqual(resolve_target_states(1, 2, 4), [2])

    def test_single_state_boundary(self):
        from pyrai3md.workflow.md import resolve_target_states
        self.assertEqual(resolve_target_states(0, 0, 1), [0])
        self.assertEqual(resolve_target_states(1, 0, 1), [0])

    def test_current_state_out_of_range_raises(self):
        from pyrai3md.workflow.md import resolve_target_states
        with self.assertRaises(ValueError):
            resolve_target_states(0, 4, 4)
        with self.assertRaises(ValueError):
            resolve_target_states(1, -1, 4)

    def test_checkpoint_called_each_step(self):
        from pyrai3md.workflow.md import AIMD

        class CountingAIMD(AIMD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.checkpoint_count = 0

            def checkpoint(self, state):
                self.checkpoint_count += 1
                return super().checkpoint(state)

        calc = MockCalculator(gradient=MOCK_GRADIENT)
        wf = CountingAIMD(calculator=calc, maxiter=3)
        state = _make_mock_state(total_steps=3)
        calc.state = state
        wf.run()
        self.assertEqual(wf.checkpoint_count, 3)

    def test_run_updates_kinetic_energy(self):
        from pyrai3md.workflow.md import AIMD
        calc = MockCalculator(gradient=MOCK_GRADIENT)
        wf = AIMD(calculator=calc, maxiter=2, dt=1.0)
        state = _make_mock_state(total_steps=2)
        state.update_system({
            'mass': [12.0, 1.0],
            'velocities': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        })
        calc.state = state
        wf.run()
        # After the second step the velocities are non-zero, so the
        # stored kinetic energy must be positive.
        self.assertGreater(state.runtime.kinetic_energy, 0.0)

    def test_run_propagates_coordinates_with_verlet(self):
        from pyrai3md.workflow.md import AIMD
        from pyrai3md.utilities.units import AMU_TO_AU, BOHR_TO_ANGSTROM
        calc = MockCalculator(gradient=MOCK_GRADIENT)
        wf = AIMD(calculator=calc, maxiter=2, dt=1.0)
        state = _make_mock_state(total_steps=2)
        state.update_system({
            'mass': [12.0, 1.0],
            'velocities': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        })
        calc.state = state
        wf.run()

        # Constant gradient: analytic velocity-Verlet solution.
        # Step 1 keeps the initial structure (both integrators skip
        # internally); step 2 propagates R and completes V.
        dt = 1.0
        mass_au = np.array([12.0, 1.0])[:, None] * AMU_TO_AU
        grad = MOCK_GRADIENT
        r0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        r1 = r0 + (-0.5 * grad / mass_au * dt ** 2) * BOHR_TO_ANGSTROM
        v1 = -grad / mass_au * dt
        np.testing.assert_allclose(state.coordinates, r1)
        np.testing.assert_allclose(state.velocities, v1)

    def test_run_snapshots_history_each_step(self):
        from pyrai3md.workflow.md import AIMD
        calc = MockCalculator(gradient=MOCK_GRADIENT)
        wf = AIMD(calculator=calc, maxiter=3, dt=1.0)
        state = _make_mock_state(total_steps=3)
        state.update_system({
            'mass': [12.0, 1.0],
            'velocities': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        })
        calc.state = state
        wf.run()
        # One complete-timestep snapshot per step, bounded by
        # the history limit (>= 4 after normalization).
        self.assertEqual(state.history_size, 3)
        snapshot = state.get_history_step(-1)
        self.assertIn('gradient', snapshot['results'])
        # The first snapshot records the initial t = 0 structure
        # before any propagation took place.
        first = state.get_history_step(0)
        np.testing.assert_allclose(
            first['system']['coordinates'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            first['system']['velocities'],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


class _HessianMockCalculator(MockCalculator):
    """A provider that answers with an analytical force-constant matrix.

    The matrix is a diatomic stretch along x: identical masses, the k[[1,-1],
    [-1,1]] block in the two x components and zeros elsewhere, so the expected
    analysis is one vibration with the frequency `vibration` itself computes
    from it. Nothing in the class below reads a made-up number off a screen.
    """

    K = 0.02

    @classmethod
    def matrix(cls):
        h = np.zeros((6, 6))
        for left, right in ((0, 3), (3, 0)):
            pass
        h[0, 0] = cls.K
        h[3, 3] = cls.K
        h[0, 3] = -cls.K
        h[3, 0] = -cls.K
        return h

    def run(self, target_states=None):
        self.call_count += 1
        results = {'energy': self._energy,
                   'gradient': self._gradient,
                   'hessian': [self.matrix().tolist()]}
        self.state.set_result(results)
        return dict(results)


def _mass_state(total_steps=1):
    """A two-atom state with masses, on a jobtype that already asked for the matrix.

    `frequency` is spelled here because it is what a property run looks like to
    the Hessian workflow: planning resolved the skill's profile for the
    force-constant request, so the matrix is expected in the evaluation the run
    performs. A run of a different jobtype (an optimization loop, for example)
    asks the backend for the matrix as a job of its own instead, and that second
    channel is `test_hessian_link.py`'s subject, not this class's.
    """
    state = _make_mock_state(total_steps=total_steps)
    state.pmd.control.jobtype = 'frequency'
    state.pmd.system.mass = [12.0, 12.0]
    return state




class TestScanWorkflow(IsolatedCwdTestCase):
    """Test the Scan workflow construction."""

    def test_construction(self):
        from pyrai3md.workflow.scan import Scan
        calc = MockCalculator()
        wf = Scan(calculator=calc, maxiter=5)
        self.assertEqual(wf.name, 'scan')
        self.assertEqual(wf.maxiter, 5)

    def test_run_calls_calculator(self):
        from pyrai3md.workflow.scan import Scan
        calc = MockCalculator(energy=-50.0)
        wf = Scan(calculator=calc, maxiter=3)
        state = _make_mock_state(total_steps=10)
        calc.state = state
        wf.run()
        self.assertEqual(calc.call_count, 3)

    def test_run_records_energies(self):
        from pyrai3md.workflow.scan import Scan
        calc = MockCalculator(energy=-50.0)
        wf = Scan(calculator=calc, maxiter=2)
        state = _make_mock_state(total_steps=10)
        calc.state = state
        wf.run()
        self.assertIn('scan_energies', state.result.keys())
        self.assertEqual(len(state.result['scan_energies']), 2)


class TestWorkflowNoDirectExternalCalls(unittest.TestCase):
    """Verify that workflows do not directly call external programs."""

    def test_optimize_no_subprocess(self):
        from pyrai3md.workflow.optimize import Optimize
        import inspect
        source = inspect.getsource(Optimize)
        self.assertNotIn('subprocess', source)
        self.assertNotIn('os.system', source)

    def test_namd_no_subprocess(self):
        from pyrai3md.workflow.md import AIMD
        import inspect
        source = inspect.getsource(AIMD)
        self.assertNotIn('subprocess', source)
        self.assertNotIn('os.system', source)




class TestTopologyContinuationIsDerivedFromTheJobtype(unittest.TestCase):
    """Who is allowed to carry a GFN-FF topology from one evaluation to the next.

    A skill cannot make this call itself: planning hands every route the same
    re-spelled 'energy gradient' jobtype string, so dynamics and a geometry
    search arrive indistinguishable. The framework decides and says so through
    `keep_topology`, because the fixed calculation directory otherwise makes the
    engine read back the bond list some *other* geometry produced - measured at
    +0.3508098 Hartree on a healthy structure in a controlled rerun, and the
    shape of the block round tmp_20261017_234656 reported at rows 6-21.

    Read off the shipped examples through the real load chain, and off the same
    level the backend is built from (`task['config']['skill_config']`), so the
    assertion survives a refactor of the task dict's shape only if the value
    still reaches the skill.
    """

    def _flags(self, example):
        import yaml
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'examples', example + '.yaml')
        with open(path, 'r', encoding='utf-8') as handler:
            spec = yaml.safe_load(handler)
        state = build_state(validate(normalize(build_config(spec))),
                            raw_config=spec)
        out = []
        for _category, task in state.iter_tasks():
            skill = (task.get('config') or {}).get('skill_config') or {}
            out.append((task.get('skill_name'), skill.get('keep_topology')))
        return out

    def test_a_dynamics_run_asks_for_the_continuation(self):
        for example in ('md_lammps_nvt_nh', 'md_lammps_npt_nh_berendsen'):
            flags = self._flags(example)
            self.assertTrue(flags, example)
            for skill, value in flags:
                self.assertIs(value, True,
                              '%s (%s): a dynamics step is exactly the case '
                              'the carried topology is for' % (example, skill))

    def test_a_geometry_search_does_not(self):
        for example in ('optimize_cell_ts', 'optimize_cell_meci_multiscale'):
            flags = [f for f in self._flags(example) if f[0] == 'xtb_skill']
            self.assertTrue(flags, example)
            for skill, value in flags:
                self.assertIs(value, False,
                              '%s (%s): a search moves the geometry on its own '
                              'terms, so each trial must be scored with its '
                              'own bond list' % (example, skill))

    def test_the_precedent_the_injection_follows_still_holds(self):
        """The NPT stress request rides the same block, unchanged.

        #378 put `keep_topology` beside the existing `stress` derivation. If
        that block stopped running for md tasks, both would go missing at once
        and this is the reading that says so.
        """
        skills = {name for name, _ in self._flags('md_lammps_npt_nh_berendsen')}
        self.assertTrue(skills, 'no tasks planned for the NPT example')
        import yaml
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'examples',
            'md_lammps_npt_nh_berendsen.yaml')
        with open(path, 'r', encoding='utf-8') as handler:
            spec = yaml.safe_load(handler)
        state = build_state(validate(normalize(build_config(spec))),
                            raw_config=spec)
        stresses = [(task.get('skill_name'),
                     (task.get('config') or {}).get('skill_config', {})
                     .get('stress'))
                    for _category, task in state.iter_tasks()]
        self.assertIn(('lammps_skill', True), stresses,
                      'an NPT run no longer asks its engine for a stress '
                      'tensor: %s' % stresses)


if __name__ == '__main__':
    unittest.main()


COMPOSITE_TERMS = {'labels': ['high[qm]', 'low[qm]', 'low[mm]'],
                   'energies': [[-78.0], [-77.0], [-100.0]],
                   'contributions': [1, -1, 1]}
COMPOSITE_TOTAL = -101.0


class CompositeCalculator(MockCalculator):
    """One evaluation of a single-center composite, as the assembly returns it.

    `energy` and `composite_terms` arrive in the SAME result dictionary, because
    `Calculator` writes `pmd.results` whole (`data/state.py::set_result` assigns
    rather than merges) - a fixture that wrote them in two calls would erase the
    energy and then report the writer, not the fixture, as the defect.
    """

    def __init__(self, energy=MOCK_ENERGY, gradient=None, terms=None):
        super().__init__(energy=energy, gradient=gradient)
        self._terms = COMPOSITE_TERMS if terms is None else terms

    def run(self, target_states=None):
        self.call_count += 1
        self.run_calls.append(target_states)
        if self.state is not None and self.state.pmd is not None:
            results = {'energy': self._energy,
                       'composite_terms': dict(self._terms),
                       'composite_total': [COMPOSITE_TOTAL]}
            if self._gradient is not None:
                results['gradient'] = self._gradient
            self.state.set_result(results)
        return MockResult(energy=self._energy, gradient=self._gradient)


class TestTheAssemblyWritesItsTermsBesideTheEnergy(IsolatedCwdTestCase):
    """One row of a composite run, read back as the columns it was written in.

    A single-center composite keeps its assembled objective in the energy column,
    and until now the pieces of that assembly lived only in the `&composite
    energy decomposition` block of the run log: a reader of the machine-readable
    table saw the sum and nothing else. The term columns are what the ruling on
    2026-10-18 asked for, and the arithmetic that makes them worth having is the
    sum, so that is what is asserted here.

    The fixture is the ONIOM2 shape: the real system at the low level, the centre
    at the high level, and the centre again at the low level with a minus
    coefficient, which adds up to the energy the run records.
    """

    def _run(self, title, calculator, steps):
        import os
        from pyrai3md.workflow.optimize import Optimize
        state = _make_mock_state(total_steps=steps)
        state.pmd.control.title = title
        calculator.state = state
        Optimize(calculator=calculator, maxiter=steps).run()
        return os.path.join(os.getcwd(), 'opt_status.txt')

    def test_the_term_columns_are_appended_after_the_fixed_header(self):
        from pyrai3md.utilities.opt_formatter import STATUS_ATOM_COLUMNS
        from pyrai3md.utilities.print_results import WRITER_ATOM, read_status
        record = read_status(self._run(
            'terms', CompositeCalculator(energy=COMPOSITE_TOTAL,
                                         gradient=MOCK_GRADIENT), 2))
        self.assertEqual(record.writer, WRITER_ATOM)
        self.assertEqual(record.columns,
                         list(STATUS_ATOM_COLUMNS) + COMPOSITE_TERMS['labels'],
                         'the header widened, but not in the order the log '
                         'orders its terms')
        self.assertEqual(record.malformed, [])
        self.assertTrue(record.rows, 'no step row was written at all')

    def test_the_term_columns_add_up_to_the_energy_column_they_sit_beside(self):
        from pyrai3md.utilities.print_results import read_status
        record = read_status(self._run(
            'sums', CompositeCalculator(energy=COMPOSITE_TOTAL,
                                        gradient=MOCK_GRADIENT), 2))
        self.assertGreaterEqual(len(record.rows), 2, record.malformed)
        for row in record.rows:
            total = sum(row[name] for name in COMPOSITE_TERMS['labels'])
            self.assertAlmostEqual(total, row['energy'], delta=1.0e-9,
                                   msg='row %s: the terms add to %.10f and the '
                                       'energy column says %.10f'
                                       % (row['step'], total, row['energy']))
            self.assertAlmostEqual(row['energy'], COMPOSITE_TOTAL, delta=1e-9)

    def test_a_single_layer_run_keeps_the_header_it_has_always_written(self):
        # The other half of the same claim, and the one that keeps this change
        # from moving the bytes of every ordinary relaxation: no composite
        # breakdown in the result, so no term columns on disk.
        from pyrai3md.utilities.opt_formatter import status_header
        path = self._run('plain', MockCalculator(energy=MOCK_ENERGY,
                                                 gradient=MOCK_GRADIENT), 2)
        with open(path) as handler:
            first = handler.readline().rstrip(chr(10))
        self.assertEqual(first, status_header())

    def test_the_header_is_not_widened_under_rows_already_written(self):
        # The writer fits the term columns at step one only, so reaching the
        # widening with rows already in the file means some other order of
        # operations was written, and rewriting the header then would leave
        # those rows read against the wrong number of columns.
        import os
        from pyrai3md.workflow.optimize import Optimize
        state = _make_mock_state(total_steps=1)
        state.pmd.control.title = 'late'
        calc = MockCalculator(energy=MOCK_ENERGY, gradient=MOCK_GRADIENT)
        calc.state = state
        workflow = Optimize(calculator=calc, maxiter=1)
        workflow.run()
        path = os.path.join(os.getcwd(), 'opt_status.txt')
        with open(path, 'a') as handler:
            handler.write('     2  1.0\n')
        with self.assertRaises(ValueError) as caught:
            workflow._fit_status_header(['high[qm]'])
        self.assertIn('already', str(caught.exception))
        with open(path) as handler:
            self.assertTrue(handler.read().startswith('  step'),
                            'the refusal still rewrote the record')


class TestEachTermIsReadAtTheStateTheAssemblyRecords(IsolatedCwdTestCase):
    """Which state a term's column comes from, read back off the table.

    The remote round found this by crashing: an assembly keeps each term's RAW
    per-state energy row (`composite.py` collects it before the total's
    broadcast), while a crossing search drives the box on state 1 or 2. One
    global index for every row then either raises or writes the wrong number,
    and the local fixture could not see it because all its rows held one state
    and its driven state was 0.
    """

    def _run(self, title, terms, energy, driven=None, steps=2):
        import os
        from pyrai3md.utilities.print_results import read_status
        from pyrai3md.workflow.optimize import Optimize
        state = _make_mock_state(total_steps=steps)
        state.pmd.control.title = title
        calc = CompositeCalculator(energy=energy, gradient=MOCK_GRADIENT,
                                   terms=terms)
        calc.state = state
        workflow = Optimize(calculator=calc, maxiter=steps)
        # `Optimize.__init__` registers the optimized state as the driven one
        # (`opt_state_i`, default 0), so the state these rows are scored on is
        # set after it - the way a crossing run ends up on state 1 or 2.
        if driven is not None:
            state.update_system({'current_state': driven})
        workflow.run()
        return read_status(os.path.join(os.getcwd(), 'opt_status.txt'))

    def test_a_state_independent_term_survives_being_driven_on_another_state(self):
        # The shape that crashed: every term holds one energy (a classical
        # outer layer answers one state) while the run is driven on state 1.
        # The broadcast the total already applies is the same number here.
        terms = {'labels': ['high[qm]', 'low[qm]', 'low[mm]'],
                 'energies': [[-78.0], [-77.0], [-100.0]],
                 'contributions': [1, -1, 1],
                 'state_indices': [None, None, None]}
        record = self._run('onestate', terms, -101.0, driven=1)
        self.assertEqual(record.malformed, [])
        self.assertTrue(record.rows, 'no step row was written at all')
        for row in record.rows:
            self.assertAlmostEqual(row['high[qm]'], -78.0, delta=1.0e-9)
            self.assertAlmostEqual(row['low[qm]'], 77.0, delta=1.0e-9)
            self.assertAlmostEqual(row['low[mm]'], -100.0, delta=1.0e-9)
            self.assertAlmostEqual(sum(row[name] for name in terms['labels']),
                                   row['energy'], delta=1.0e-9)

    def test_a_multi_state_term_is_read_at_the_driven_state_not_the_first(self):
        terms = {'labels': ['high[qm]', 'low[qm]', 'low[mm]'],
                 'energies': [[-78.0, -71.0, -61.0],
                              [-77.0, -70.0, -60.0],
                              [-100.0, -90.0, -80.0]],
                 'contributions': [1, -1, 1]}
        # Without this the second assertion would also pass for a reader that
        # always took the first state.
        self.assertNotEqual(terms['energies'][0][1], terms['energies'][0][0])
        record = self._run('driven', terms, -91.0, driven=1)
        self.assertEqual(record.malformed, [])
        row = record.rows[0]
        self.assertAlmostEqual(row['high[qm]'], -71.0, delta=1.0e-9,
                               msg='state 1 was driven but the column holds '
                                   'state %s' % row)
        self.assertAlmostEqual(row['low[qm]'], 70.0, delta=1.0e-9)
        self.assertAlmostEqual(row['low[mm]'], -90.0, delta=1.0e-9)
        self.assertAlmostEqual(sum(row[name] for name in terms['labels']),
                               row['energy'], delta=1.0e-9)

    def test_a_multi_center_term_is_read_at_its_own_recorded_state(self):
        # C>1 has no global state axis: each term contributes at the local
        # state the assembly records, so the columns cannot share one index.
        # Driven on state 0 while the record says otherwise is the witness -
        # taking the driven state for all three adds up to -91.0, not -84.0.
        terms = {'labels': ['high[qm]', 'low[qm]', 'low[mm]'],
                 'energies': [[-78.0, -71.0], [-77.0, -70.0],
                              [-100.0, -90.0]],
                 'contributions': [1, -1, 1],
                 'state_indices': [1, 0, 1]}
        record = self._run('centers', terms, -84.0, driven=0)
        self.assertEqual(record.malformed, [])
        row = record.rows[0]
        self.assertAlmostEqual(row['high[qm]'], -71.0, delta=1.0e-9)
        self.assertAlmostEqual(row['low[qm]'], 77.0, delta=1.0e-9)
        self.assertAlmostEqual(row['low[mm]'], -90.0, delta=1.0e-9)
        self.assertAlmostEqual(sum(row[name] for name in terms['labels']),
                               row['energy'], delta=1.0e-9)

    def test_a_crossing_legs_columns_sum_to_the_face_not_to_the_objective(self):
        # The identity `sum of the term columns == the energy column` is a
        # property of ONE assembly, not of the table: it held exactly (worst
        # difference 0.0 over 64 rows of both carriers) on
        # `optimize_ts_multiscale` in round 16, because there the recorded energy
        # IS the sum of the contributions. A crossing leg records the objective
        # of the pair while its columns stay the raw per-term numbers the
        # backends computed, so the two differ by construction. Measured on this
        # host's own crossing legs on the force-field box
        # (`optimize_cell_meci_lammps`, 2026-10-19): the columns summed 1.28e-2 to
        # 1.51e-2 Hartree away from the energy row at every one of its five steps.
        #
        # What the table code has to get right is therefore not the identity but
        # its absence: a writer that reconciled the row, or a reader that
        # classified a non-summing row as malformed, would break the crossing legs
        # already on the plan. The witness that this is not an arbitrary fixture is
        # the criterion after it - the same columns with an energy that does add up
        # also come back whole.
        terms = {'labels': ['high[qm]', 'high[qm2]', 'low[qm2]'],
                 'energies': [[-78.0, -71.0, -61.0],
                              [-77.0, -70.0, -60.0],
                              [-100.0, -90.0, -80.0]],
                 'contributions': [1, -1, 1],
                 'state_indices': [1, 2, 1]}
        # Driven on state 1: the columns read -71.0, +60.0 (the second term
        # carries contribution -1) and -90.0, which sum to -101.0, while the row's
        # objective is the pair's own, -86.0.
        record = self._run('crossing', terms, -86.0, driven=1)
        self.assertEqual(record.malformed, [])
        self.assertEqual(record.columns[-3:], ['high[qm]', 'high[qm2]',
                                               'low[qm2]'])
        row = record.rows[0]
        self.assertAlmostEqual(row['high[qm]'], -71.0, delta=1.0e-9)
        self.assertAlmostEqual(row['high[qm2]'], 60.0, delta=1.0e-9)
        self.assertAlmostEqual(row['low[qm2]'], -90.0, delta=1.0e-9)
        total = sum(row[name] for name in terms['labels'])
        self.assertNotEqual(total, row['energy'],
                            'the fixture stopped being a crossing shape')
        self.assertAlmostEqual(total, -101.0, delta=1.0e-9)
        self.assertAlmostEqual(row['energy'], -86.0, delta=1.0e-9)

    def test_a_row_whose_columns_do_add_up_is_still_a_whole_row(self):
        terms = {'labels': ['high[qm]', 'high[qm2]', 'low[qm2]'],
                 'energies': [[-78.0, -71.0, -61.0],
                              [-77.0, -70.0, -60.0],
                              [-100.0, -90.0, -80.0]],
                 'contributions': [1, -1, 1],
                 'state_indices': [1, 2, 1]}
        record = self._run('summing', terms, -101.0, driven=1)
        self.assertEqual(record.malformed, [])
        row = record.rows[0]
        self.assertAlmostEqual(sum(row[name] for name in terms['labels']),
                               row['energy'], delta=1.0e-9)

    def test_mismatched_state_records_refuse_the_columns_rather_than_misread(self):
        # Fewer indices than terms means at least one column would be read
        # against the wrong term. The record keeps the header it always had and
        # the row stays whole; the refusal is the point, not a crash.
        from pyrai3md.utilities.opt_formatter import STATUS_ATOM_COLUMNS
        terms = {'labels': ['high[qm]', 'low[qm]', 'low[mm]'],
                 'energies': [[-78.0], [-77.0], [-100.0]],
                 'contributions': [1, -1, 1],
                 'state_indices': [1, 0]}
        record = self._run('short', terms, -101.0, driven=1)
        self.assertEqual(record.columns, list(STATUS_ATOM_COLUMNS))
        self.assertEqual(record.malformed, [])
        self.assertTrue(record.rows)
