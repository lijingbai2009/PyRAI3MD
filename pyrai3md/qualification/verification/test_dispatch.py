"""
======================================================================
Test suite for the dispatch subsystem and runtime loader:
- pyrai3md/workflow/dispatch/
- pyrai3md/data/runtime/loader.py

Tests cover:
- RuntimeDirectoryManager naming (%d, no zero padding) and
  globally unique index allocation with restart reuse
- Tri-state aggregation rule (aggregate_tristate)
- Launcher / Scheduler composition (multitask x remotetask)
- Dispatcher.dispatch() unified pipeline and unit materialization
- ExecutionGraphBuilder: root graph construction, modularized
  keyword overrides, stage sub-nodes with stage_defaults
- Universal child task rules (rule 1 worker label, rule 2 worker
  graphs reject remotetask, rule 3 calculator always local)
- Calculator backend role grouping, active role switching and
  composite matching
- AdaptiveSampling three-stage chain end-to-end
- .rt serialization round-trip (save -> load equivalence, task
  configs converted to dicts, no backend residue)

Aug 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import os
import shutil
import tempfile
import unittest

MOCK_ENERGY = -75.123456


def _mock_executor(state, runtime_idx=None):
    """Module-level picklable executor for Pool-based tests."""
    del runtime_idx
    return state.pmd.control.jobtype


_IDENTITY_RECORD = os.path.join(tempfile.gettempdir(),
                                'pyrai3md_identity_record.txt')


def _recording_executor(state, runtime_idx=None):
    """Module-level executor recording the received identities.

    Spawn pool workers cannot share memory with the parent process,
    so the received runtime_idx values are appended to a file.
    """
    del state
    with open(_IDENTITY_RECORD, 'a') as handler:
        handler.write('%s\n' % runtime_idx)
    return 'ok'


def _make_mock_state(jobtype='energy'):
    """Create a minimal PyRAIState with pmd as the data source."""
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.data.runtime.state import RuntimeState
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.control.jobtype = jobtype
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    config.system.natom = 2

    state = PyRAIState(runtime=RuntimeState(total_steps=1))
    state.pmd = config
    return state


def _make_task(skill_name, static_id, config=None):
    """Create one static task descriptor."""
    from pyrai3md.schema.config import PyRAIConfig

    return {
        'role': 'qm', 'task': 0, 'task_id': skill_name, 'layer_id': 0,
        'static_id': static_id,
        'contribution': 1,
        'region': 'all', 'region_indices': [], 'region_charge': None,
        'skill_name': skill_name,
        'api': 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend',
        'capabilities': ['energy'],
        'config': config if config is not None else PyRAIConfig(),
    }


def _make_state_with_task(jobtype='energy'):
    """Create a mock state with one task in the theory queue."""
    state = _make_mock_state(jobtype=jobtype)
    state.add_task('theory', _make_task('test_skill', 'qm_layer0_test'))
    return state


def _set_environment(state, **fields):
    """Set environment snapshot fields on a mock state."""
    for key, value in fields.items():
        setattr(state.pmd.environment, key, value)
    return state


class _TempWorkdir:
    """Context manager running the body in a fresh temp cwd.

    Batch dispatch allocates runtime sub directories under the
    current working directory; tests exercise that side effect in a
    throwaway directory instead of the repository root.
    """

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='pyrai3md_dispatch_cwd_')
        self._previous = os.getcwd()
        os.chdir(self.path)
        return self.path

    def __exit__(self, *exc_info):
        del exc_info
        os.chdir(self._previous)
        shutil.rmtree(self.path, ignore_errors=True)


class _CountingBackend:
    """Mock backend counting calculate() calls (per instance)."""

    def __init__(self, energy=MOCK_ENERGY, config=None, execution_id=None):
        del config, execution_id
        self.energy = energy
        self.calls = []

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        del point_charges, target_states
        self.calls.append((list(atoms), coordinates))
        return {'energy': self.energy}


class _RoleProbeBackend:
    """Importable backend that records its execution_id and first x.

    The class-level list survives the backend being rebuilt inside
    execute(), which is exactly what the reference-child test needs.
    """

    calls = []

    def __init__(self, config=None, execution_id=None):
        del config
        self.execution_id = execution_id or ''

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        del atoms, point_charges, target_states, cell
        _RoleProbeBackend.calls.append(
            (self.execution_id, float(coordinates[0][0])))
        return {'energy': -9.0}


class _CloseProbeBackend:
    """Importable backend that logs construction and release in order.

    The log is class-level for the same reason `_RoleProbeBackend.calls`
    is: the backend is rebuilt inside execute(), and the question is what
    one process saw across two units.
    """

    events = []

    def __init__(self, config=None, execution_id=None):
        del config
        self.execution_id = execution_id or ''
        _CloseProbeBackend.events.append(('open', self.execution_id))

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        del atoms, coordinates, point_charges, target_states, cell
        return {'energy': -9.0}

    def close(self):
        _CloseProbeBackend.events.append(('close', self.execution_id))


class TestRuntimeDirectoryManager(unittest.TestCase):
    """Test runtime directory layout and index allocation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrai3md_dispatch_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_runtime_name_no_zero_padding(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        self.assertEqual(manager.runtime_name(3), 'pyscf_skill_runtime_3')
        self.assertEqual(manager.runtime_name(12), 'pyscf_skill_runtime_12')
        self.assertEqual(manager.runtime_idx(3), 'runtime_3')

    def test_directory_and_file_paths(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        expected_dir = os.path.join(
            self.tmpdir, 'pyscf_skill_runtime_3')
        self.assertEqual(manager.runtime_dir(3), expected_dir)
        self.assertEqual(
            manager.runtime_file(3),
            os.path.join(expected_dir, 'pyscf_skill_runtime_3.rt'))
        self.assertEqual(
            manager.result_file(3),
            os.path.join(expected_dir, 'result.json'))

    def test_create_directory(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        path = manager.create(1)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(path, manager.runtime_dir(1))

    def test_next_index_unique(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        self.assertEqual(manager.next_index(), 1)
        manager.create(1)
        self.assertEqual(manager.next_index(), 2)
        manager.create(3)
        self.assertEqual(manager.next_index(), 2)

    def test_existing_indices_ignore_foreign_names(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        os.makedirs(os.path.join(self.tmpdir, 'other_skill_runtime_1'))
        os.makedirs(os.path.join(self.tmpdir, 'pyscf_skill_runtime_2'))
        self.assertEqual(manager.existing_indices(), {2})

    def test_allocate_indices_reuse_for_restart(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        manager.create(1)
        manager.create(2)
        self.assertEqual(manager.allocate_indices(3), [1, 2, 3])
        self.assertEqual(manager.allocate_indices(1), [1])

    def test_is_completed(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='pyscf_skill', root=self.tmpdir)
        manager.create(1)
        self.assertFalse(manager.is_completed(1))
        with open(manager.result_file(1), 'w') as handler:
            handler.write('{}')
        self.assertTrue(manager.is_completed(1))

    def test_custom_name_format(self):
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager,
        )
        manager = RuntimeDirectoryManager(
            skill_name='qdm_skill', root=self.tmpdir,
            name_format='benchmark_fssh_tully1_{idx}')
        self.assertEqual(manager.runtime_name(3),
                         'benchmark_fssh_tully1_3')
        # The workflow suffix stays 'runtime_{idx}' for every
        # naming template.
        self.assertEqual(manager.runtime_idx(3), 'runtime_3')
        self.assertEqual(
            manager.runtime_file(3),
            os.path.join(self.tmpdir, 'benchmark_fssh_tully1_3',
                         'benchmark_fssh_tully1_3.rt'))
        # The scan pattern is derived from the template: foreign
        # layouts (including the default one) never match.
        os.makedirs(os.path.join(
            self.tmpdir, 'benchmark_fssh_tully1_2'))
        os.makedirs(os.path.join(self.tmpdir, 'qdm_skill_runtime_1'))
        self.assertEqual(manager.existing_indices(), {2})
        self.assertEqual(manager.allocate_indices(3), [2, 3, 4])


class TestAggregateTristate(unittest.TestCase):
    """Test the tri-state aggregation rule of the graph builder."""

    def test_all_none_is_false(self):
        from pyrai3md.workflow.dispatch.planner import aggregate_tristate
        self.assertFalse(aggregate_tristate([None, None, None]))
        self.assertFalse(aggregate_tristate([]))

    def test_true_and_none_is_true(self):
        from pyrai3md.workflow.dispatch.planner import aggregate_tristate
        self.assertTrue(aggregate_tristate([True, None]))
        self.assertTrue(aggregate_tristate([None, True, None]))

    def test_false_and_none_is_false(self):
        from pyrai3md.workflow.dispatch.planner import aggregate_tristate
        self.assertFalse(aggregate_tristate([False, None]))
        self.assertFalse(aggregate_tristate([None, False, None]))

    def test_true_and_false_raises(self):
        from pyrai3md.workflow.dispatch.planner import aggregate_tristate
        with self.assertRaises(KeyError):
            aggregate_tristate([True, False])
        with self.assertRaises(KeyError):
            aggregate_tristate([None, True, False])

    def test_resolve_flag_priority_chain(self):
        from pyrai3md.workflow.dispatch.planner import resolve_flag
        # Declaration dict is the highest priority: a key that
        # appears forces its value over the input keyword.
        self.assertFalse(resolve_flag(
            {'multitask': False}, 'multitask', True, True))
        self.assertTrue(resolve_flag(
            {'multitask': True}, 'multitask', False, False))
        # No declaration: the input keyword wins over the default.
        self.assertTrue(resolve_flag({}, 'multitask', True, False))
        self.assertFalse(resolve_flag({}, 'multitask', False, True))
        # Keyword None keeps the class default.
        self.assertTrue(resolve_flag({}, 'multitask', None, True))
        self.assertFalse(resolve_flag({}, 'multitask', None, False))

    def test_resolve_pool_priority_chain(self):
        from pyrai3md.workflow.dispatch.planner import resolve_pool
        self.assertEqual(resolve_pool({'multipool': 2}, 8, 4), 2)
        self.assertEqual(resolve_pool({}, 8, 4), 8)
        self.assertEqual(resolve_pool({}, None, 4), 4)

    def test_aggregate_multipool_rules(self):
        from pyrai3md.workflow.dispatch.planner import aggregate_multipool
        self.assertEqual(aggregate_multipool([]), 1)
        self.assertEqual(aggregate_multipool([None, None]), 1)
        self.assertEqual(aggregate_multipool([None, 4]), 4)
        self.assertEqual(aggregate_multipool([4, 4]), 4)
        with self.assertRaises(KeyError):
            aggregate_multipool([2, 4])


class TestMoleculeDispatchFlags(unittest.TestCase):
    """R3a: molecule(System)-scope dispatch keyword carrier.

    The molecule section carries multitask/remotetask/multipool at
    center scope (dev/MULTICENTER_DEVELOPMENT.md). They must land on
    config.system, stay distinct from the same-named workflow-section
    keywords (md -> dynamics), and preserve the tri-state None
    (unspecified -> planner inherits from skills) vs explicit False.
    """

    def _build(self, raw):
        from pyrai3md.schema.adapter import _build_pydantic_config
        return _build_pydantic_config(raw, env_config=None)

    def test_molecule_flags_land_on_system(self):
        config = self._build({
            'control': {'jobtype': 'sp'},
            'molecule': {'multitask': True, 'remotetask': False,
                         'multipool': 2},
        })
        self.assertIs(config.system.multitask, True)
        self.assertIs(config.system.remotetask, False)
        self.assertEqual(config.system.multipool, 2)

    def test_molecule_and_md_flags_do_not_collide(self):
        # molecule -> system and md -> dynamics share the keyword
        # names; nesting keeps them from overwriting each other.
        config = self._build({
            'control': {'jobtype': 'md'},
            'molecule': {'multitask': True, 'multipool': 2},
            'md': {'multitask': False, 'multipool': 4},
        })
        self.assertIs(config.system.multitask, True)
        self.assertEqual(config.system.multipool, 2)
        self.assertIs(config.dynamics.multitask, False)
        self.assertEqual(config.dynamics.multipool, 4)

    def test_unspecified_molecule_flags_are_none(self):
        # Absent molecule keywords stay None (inherit), not False/1.
        config = self._build({
            'control': {'jobtype': 'sp'},
            'molecule': {'highlevel': [0, 1]},
        })
        self.assertIsNone(config.system.multitask)
        self.assertIsNone(config.system.remotetask)
        self.assertIsNone(config.system.multipool)

    def test_explicit_false_distinct_from_none(self):
        config = self._build({
            'control': {'jobtype': 'sp'},
            'molecule': {'multitask': False},
        })
        self.assertIs(config.system.multitask, False)
        self.assertIsNone(config.system.remotetask)

    def test_classic_string_flags_are_typed(self):
        # Classic text carries '1'/'0' strings; ReadVal('b')/('i')
        # type them on the nested molecule section.
        config = self._build({
            'control': {'jobtype': 'sp'},
            'molecule': {'multitask': '1', 'remotetask': '0',
                         'multipool': '3'},
        })
        self.assertIs(config.system.multitask, True)
        self.assertIs(config.system.remotetask, False)
        self.assertEqual(config.system.multipool, 3)


class TestLauncher(unittest.TestCase):
    """Test the Launcher: remotetask -> single-task runner."""

    def test_local_launch_returns_local_sentinel(self):
        from pyrai3md.workflow.dispatch.launcher import Launcher

        runner = Launcher()(False)
        self.assertEqual(runner, 'local')

    def test_remote_launch_returns_slurm_run(self):
        from pyrai3md.workflow.dispatch.launcher import Launcher
        from pyrai3md.workflow.dispatch.slurm_run import slurm_run

        runner = Launcher()(True)
        # remotetask=True always returns slurm_run
        self.assertIs(runner, slurm_run)


class TestScheduler(unittest.TestCase):
    """Test the Scheduler: multitask -> execution wrapper."""

    def test_multitask_false_sequential_local(self):
        from pyrai3md.workflow.dispatch.scheduler import Scheduler

        executed = []

        def _local_executor(state):
            executed.append(state.pmd.control.jobtype)
            return state.pmd.control.jobtype

        scheduler = Scheduler(state=_make_mock_state(),
                              local_executor=_local_executor)
        run_all = scheduler(False)('local')
        units = [
            {'state': _make_mock_state(jobtype='energy'), 'task_index': 0},
            {'state': _make_mock_state(jobtype='gradient'),
             'task_index': 1},
        ]
        self.assertEqual(run_all(units), ['energy', 'gradient'])
        self.assertEqual(executed, ['energy', 'gradient'])

    def test_multitask_true_local_runs_in_pool(self):
        from pyrai3md.workflow.dispatch.scheduler import Scheduler

        state = _make_mock_state()
        test_spec = (__name__, '_mock_executor')
        scheduler = Scheduler(state=state, executor_spec=test_spec,
                              multipool=2)
        # Local runner with multitask=True goes through the pool
        run_all = scheduler(True)('local')
        units = [
            {'state': _make_mock_state(jobtype='energy'),
             'task_index': 0, 'executor_spec': test_spec},
            {'state': _make_mock_state(jobtype='gradient'),
             'task_index': 1, 'executor_spec': test_spec},
            {'state': _make_mock_state(jobtype='nac'),
             'task_index': 2, 'executor_spec': test_spec},
        ]
        self.assertEqual(run_all(units),
                         ['energy', 'gradient', 'nac'])

    def test_remote_runner_returns_callable(self):
        from pyrai3md.workflow.dispatch.launcher import Launcher

        runner = Launcher()(True)
        # Remote runner is slurm_run, not 'local'
        self.assertNotEqual(runner, 'local')
        self.assertTrue(callable(runner))

    def test_worker_count_defaults_and_floor(self):
        from pyrai3md.workflow.dispatch.scheduler import Scheduler

        scheduler = Scheduler(state=_make_mock_state())
        self.assertEqual(scheduler._worker_count(), 1)
        # multipool is declared at Scheduler construction
        scheduler = Scheduler(state=_make_mock_state(), multipool=8)
        self.assertEqual(scheduler._worker_count(), 8)


class TestDispatcherPipeline(unittest.TestCase):
    """Test Dispatcher.dispatch() and unit materialization."""

    def test_run_local_units_in_order(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        executed = []

        def _executor(child_state, runtime_idx=None):
            del runtime_idx
            executed.append(child_state)
            return child_state.jobtype

        dispatcher = Dispatcher(state=_make_mock_state(),
                                local_executor=_executor)
        units = [
            {'state': _make_mock_state(jobtype='energy'), 'task_index': 0},
            {'state': _make_mock_state(jobtype='gradient'),
             'task_index': 1},
        ]
        with _TempWorkdir():
            results = dispatcher.dispatch(units=units)
        self.assertEqual(results, ['energy', 'gradient'])
        self.assertEqual(len(executed), 2)

    def test_run_materializes_raw_tasks(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        state = _make_state_with_task(jobtype='frequency')
        received = []

        def _executor(child_state, runtime_idx=None):
            received.append(runtime_idx)
            return child_state.jobtype

        dispatcher = Dispatcher(state=state, local_executor=_executor)
        tasks = [{'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                 {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]}]
        with _TempWorkdir() as workdir:
            results = dispatcher.dispatch(units=tasks,
                                          child_jobtype='energy')
            # Batch > 1: every unit gets a runtime identity and a
            # sub working directory (execution-location independent).
            self.assertEqual(received, ['runtime_1', 'runtime_2'])
            self.assertTrue(os.path.isdir(os.path.join(
                workdir, 'runtime', 'test_skill_runtime_1')))
            self.assertTrue(os.path.isdir(os.path.join(
                workdir, 'runtime', 'test_skill_runtime_2')))
        self.assertEqual(results, ['energy', 'energy'])

    def test_batch_single_unit_gets_no_identity(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        received = []

        def _executor(child_state, runtime_idx=None):
            received.append(runtime_idx)
            return child_state.jobtype

        dispatcher = Dispatcher(state=_make_mock_state(),
                                local_executor=_executor)
        with _TempWorkdir() as workdir:
            dispatcher.dispatch(
                units=[{'coordinates': [[0.1, 0.0, 0.0],
                                        [1.0, 0.0, 0.0]]}],
                child_jobtype='energy')
            self.assertEqual(received, [None])
            # No runtime directory layout for a single-unit batch
            self.assertFalse(os.path.isdir(
                os.path.join(workdir, 'runtime')))

    def test_materialize_stamps_backend_role(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        state = _make_state_with_task(jobtype='energy')
        dispatcher = Dispatcher(state=state)
        unit = dispatcher.materialize_unit(
            {'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]}, 0,
            child_jobtype='energy')
        self.assertEqual(unit['state'].runtime.active_role, 'theory')

    def test_explicit_role_prunes_inactive_queue(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        state = _make_mock_state()
        state.add_task('theory', _make_task('theory_skill',
                                            'theory_layer0'))
        reference = _make_task('ref_skill', 'ref_layer0')
        reference['config'].control.jobtype = 'energy gradient nac'
        state.add_task('reference', reference)
        dispatcher = Dispatcher(state=state)
        unit = dispatcher.materialize_unit(
            {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]}, 0,
            child_jobtype='gradient', active_role='reference')
        child = unit['state']
        self.assertEqual(child.runtime.active_role, 'reference')
        # Only the role being run keeps a queue: the theory backend of
        # a reference child must never be constructed.
        self.assertEqual(child.theory, [])
        self.assertEqual(len(child.reference), 1)
        # The child's control jobtype resolves a single-shot run, while
        # the planned reference capabilities are preserved (an aligned
        # jobtype would overwrite them).
        self.assertEqual(child.pmd.control.jobtype, 'gradient')
        self.assertEqual(child.reference[0]['config'].control.jobtype,
                         'energy gradient nac')
        # Parent untouched
        self.assertEqual(len(state.theory), 1)
        self.assertEqual(len(state.reference), 1)

    def test_explicit_role_requires_child_jobtype(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        dispatcher = Dispatcher(state=_make_state_with_task())
        with self.assertRaises(ValueError):
            dispatcher.materialize_unit(
                {'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]}, 0,
                active_role='reference')

    def test_execute_runs_a_reference_child(self):
        from pyrai3md.workflow import execute
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        # Resolve the probe through the very module the backend import
        # path names. unittest's discover imports this file as the bare
        # module `test_dispatch`, while the Calculator imports
        # `pyrai3md.qualification.verification.test_dispatch`: two module
        # objects, two `_RoleProbeBackend` classes, and the backend's
        # calls land in the second one. Checking the bare class (as this
        # test used to) reports 0 calls for a backend that ran.
        probe = importlib.import_module(
            'pyrai3md.qualification.verification.test_dispatch')
        api = probe.__name__ + '._RoleProbeBackend'
        state = _make_mock_state()
        theory = _make_task('theory_skill', 'theory_layer0')
        theory['api'] = api
        reference = _make_task('ref_skill', 'ref_layer0')
        reference['api'] = api
        state.add_task('theory', theory)
        state.add_task('reference', reference)
        unit = Dispatcher(state=state).materialize_unit(
            {'coordinates': [[0.4, 0.0, 0.0], [1.0, 0.0, 0.0]]}, 0,
            child_jobtype='gradient', active_role='reference')
        probe._RoleProbeBackend.calls = []
        with _TempWorkdir():
            result = execute(unit['state'])
        # execute() rebuilt the Calculator from RuntimeState.active_role:
        # one evaluation, on the reference execution id, at the
        # injected geometry.
        self.assertEqual(len(probe._RoleProbeBackend.calls), 1)
        self.assertEqual(probe._RoleProbeBackend.calls[0][0], 'ref_layer0')
        self.assertAlmostEqual(probe._RoleProbeBackend.calls[0][1], 0.4)
        self.assertAlmostEqual(result.to_dict()['energy'], -9.0)

    def test_execute_releases_the_unit_it_built(self):
        from pyrai3md.workflow import execute
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        # A batch whose pool has one worker runs its units one after
        # another in this process, and an isolated-workdir batch gives
        # every unit the same execution namespace. A backend that outlives
        # its unit therefore meets the next unit still holding the name,
        # which is how a numerical Hessian came to be refused on its
        # second displacement (round tmp_20261031_005246).
        probe = importlib.import_module(
            'pyrai3md.qualification.verification.test_dispatch')
        api = probe.__name__ + '._CloseProbeBackend'
        probe._CloseProbeBackend.events = []
        state = _make_mock_state()
        task = _make_task('theory_skill', 'theory_layer0')
        task['api'] = api
        state.add_task('theory', task)
        dispatcher = Dispatcher(state=state, local_executor=execute)
        with _TempWorkdir():
            dispatcher.dispatch(
                units=[{'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                       {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]}],
                child_jobtype='gradient')
        # Not just "a close happened": the second unit is constructed after
        # the first one is released, so the two never hold a live backend
        # at the same time.
        kinds = [kind for kind, _name in probe._CloseProbeBackend.events]
        self.assertEqual(kinds.count('open'), kinds.count('close'))
        self.assertGreaterEqual(kinds.count('open'), 2)
        self.assertEqual(kinds[:2], ['open', 'close'])

    def test_dispatch_forwards_role(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        state = _make_mock_state()
        state.add_task('theory', _make_task('theory_skill',
                                            'theory_layer0'))
        state.add_task('reference', _make_task('ref_skill', 'ref_layer0'))
        received = []

        def _executor(child_state, runtime_idx=None):
            del runtime_idx
            received.append((child_state.runtime.active_role,
                             child_state.pmd.control.jobtype,
                             len(child_state.reference)))
            return child_state.pmd.control.jobtype

        dispatcher = Dispatcher(state=state, local_executor=_executor)
        with _TempWorkdir():
            results = dispatcher.dispatch(
                units=[
                    {'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                    {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                ],
                child_jobtype='gradient', active_role='reference')
        self.assertEqual(received,
                         [('reference', 'gradient', 1),
                          ('reference', 'gradient', 1)])
        self.assertEqual(results, ['gradient', 'gradient'])

    def test_materialize_unit_stamps_worker(self):
        from pyrai3md.workflow.dispatch.dispatcher import (
            Dispatcher, worker_execution_graph,
        )
        from pyrai3md.workflow.hessian import NumericalHessian

        state = _make_state_with_task(jobtype='frequency')
        calculator = _dummy_calculator(state)
        workflow = NumericalHessian(calculator=calculator)
        displaced = [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]
        unit = workflow.materialize_unit(
            {'coordinates': displaced}, 0,
            child_jobtype=workflow.child_jobtype)
        self.assertEqual(unit['task_index'], 0)
        child = unit['state']
        # Rule 1: dispatched jobs are workers with a frozen graph
        self.assertTrue(child.runtime.worker)
        self.assertFalse(child.resolve_execution_graph)
        self.assertEqual(child.execution_graph, worker_execution_graph())
        # Child runs the basic jobtype -- no recursive dispatch
        self.assertEqual(child.pmd.control.jobtype, 'energy gradient')
        self.assertEqual(list(child.coordinates), displaced)
        # Parent state untouched
        self.assertEqual(state.pmd.control.jobtype, 'frequency')
        # Child task config aligned with the child jobtype
        child_config = child.theory[0]['config']
        self.assertEqual(child_config.control.jobtype, 'energy gradient')

    def test_materialize_unit_passthrough(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        dispatcher = Dispatcher(state=_make_mock_state())
        unit = {'state': _make_mock_state(), 'task_index': 5}
        self.assertIs(dispatcher.materialize_unit(unit, 0), unit)

    def test_worker_execution_graph_is_local(self):
        from pyrai3md.workflow.dispatch.dispatcher import (
            worker_execution_graph,
        )
        graph = worker_execution_graph()
        self.assertEqual(len(graph), 1)
        self.assertFalse(graph[0]['multitask'])
        self.assertFalse(graph[0]['remotetask'])

    def test_child_jobtype_default_energy(self):
        # A grid point trains rather than evaluates a geometry: the
        # scan's child jobtype is 'train', and the single-shot path is
        # what resolves it.
        from pyrai3md.workflow.search import GridSearch
        self.assertEqual(GridSearch.child_jobtype, 'train')
        self.assertEqual(GridSearch.name, 'search')
        self.assertEqual(GridSearch.workflow_kind, 'batch')

    def test_task_system_updates_inject_velocities(self):
        from pyrai3md.workflow.dispatch.dispatcher import (
            task_system_updates,
        )
        task = {'coordinates': [[0.1, 0.0, 0.0]],
                'velocities': [[0.0, 0.1, 0.0]]}
        self.assertEqual(task_system_updates(task), task)
        # Coordinates-only tasks stay geometry updates
        self.assertEqual(
            task_system_updates({'coordinates': [[0.1, 0.0, 0.0]]}),
            {'coordinates': [[0.1, 0.0, 0.0]]})

    def test_parallel_local_batch_forwards_identities(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        if os.path.isfile(_IDENTITY_RECORD):
            os.remove(_IDENTITY_RECORD)
        state = _make_mock_state()
        test_spec = (__name__, '_recording_executor')
        dispatcher = Dispatcher(state=state, multitask=True,
                                multipool=2,
                                executor_spec=test_spec)
        tasks = [{'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                 {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]}]
        with _TempWorkdir():
            results = dispatcher.dispatch(units=tasks,
                                          child_jobtype='energy')
        self.assertEqual(results, ['ok', 'ok'])
        with open(_IDENTITY_RECORD) as handler:
            received = sorted(handler.read().split())
        os.remove(_IDENTITY_RECORD)
        self.assertEqual(received, ['runtime_1', 'runtime_2'])

    def test_batch_identities_reused_on_restart(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        received = []

        def _executor(child_state, runtime_idx=None):
            del child_state
            received.append(runtime_idx)
            return 'ok'

        state = _make_state_with_task(jobtype='frequency')
        dispatcher = Dispatcher(state=state, local_executor=_executor)
        tasks = [{'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                 {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]}]
        with _TempWorkdir() as workdir:
            # First run allocates runtime_1/runtime_2
            dispatcher.dispatch(units=list(tasks), child_jobtype='energy')
            received.clear()
            # Restart: the existing directories are reused in order
            dispatcher.dispatch(units=list(tasks), child_jobtype='energy')
            self.assertEqual(received, ['runtime_1', 'runtime_2'])
            self.assertEqual(sorted(os.listdir(
                os.path.join(workdir, 'runtime'))),
                ['test_skill_runtime_1', 'test_skill_runtime_2'])

    def test_slurm_run_consumes_stamped_identity(self):
        import json
        import subprocess
        from unittest import mock

        # Direct submodule import: the package __init__ rebinds the
        # 'slurm_run' attribute to the function.
        import importlib
        slurm_module = importlib.import_module(
            'pyrai3md.workflow.dispatch.slurm_run')

        state = _make_state_with_task(jobtype='energy')
        unit = {'state': state, 'runtime_idx': 'runtime_7'}

        def fake_run(command, **kwargs):
            del kwargs
            if command[0] == 'sbatch':
                return subprocess.CompletedProcess(
                    command, 0, stdout='Submitted batch job 42\n')
            return subprocess.CompletedProcess(
                command, 0, stdout='COMPLETED\n')

        def fake_save_state(saved_state, path, runtime_idx=None):
            # Record the serialization target and suffix
            fake_save_state.calls.append((path, runtime_idx))

        fake_save_state.calls = []

        with _TempWorkdir() as workdir, \
                mock.patch.object(slurm_module.subprocess, 'run',
                                  side_effect=fake_run), \
                mock.patch.object(slurm_module, 'save_state',
                                  fake_save_state):
            # os.getcwd() resolves symlinked temp roots (macOS /var)
            workdir = os.getcwd()
            runtime_dir = os.path.join(workdir, 'runtime',
                                       'test_skill_runtime_7')
            os.makedirs(runtime_dir)
            with open(os.path.join(runtime_dir, 'result.json'),
                      'w') as handler:
                json.dump({'energy': MOCK_ENERGY}, handler)
            result = slurm_module.slurm_run(unit)

        # The stamped index is consumed: no fresh index allocated
        self.assertEqual(len(fake_save_state.calls), 1)
        path, suffix = fake_save_state.calls[0]
        self.assertEqual(
            path, os.path.join(runtime_dir,
                               'test_skill_runtime_7.rt'))
        self.assertEqual(suffix, 'runtime_7')
        self.assertEqual(result.energy, MOCK_ENERGY)

    def test_slurm_run_consumes_stamped_naming_policy(self):
        import json
        import subprocess
        from unittest import mock

        import importlib
        slurm_module = importlib.import_module(
            'pyrai3md.workflow.dispatch.slurm_run')

        state = _make_state_with_task(jobtype='energy')

        def fake_run(command, **kwargs):
            del kwargs
            if command[0] == 'sbatch':
                return subprocess.CompletedProcess(
                    command, 0, stdout='Submitted batch job 42\n')
            return subprocess.CompletedProcess(
                command, 0, stdout='COMPLETED\n')

        def fake_save_state(saved_state, path, runtime_idx=None):
            fake_save_state.calls.append((path, runtime_idx))

        fake_save_state.calls = []

        with _TempWorkdir(), \
                mock.patch.object(slurm_module.subprocess, 'run',
                                  side_effect=fake_run), \
                mock.patch.object(slurm_module, 'save_state',
                                  fake_save_state):
            workdir = os.getcwd()
            # The Dispatcher-stamped naming policy: custom root and
            # template reach slurm_run through the unit descriptor.
            runtime_root = os.path.join(workdir, 'trajectories')
            unit = {'state': state,
                    'runtime_idx': 'runtime_7',
                    'runtime_name': 'bench_7',
                    'runtime_root': runtime_root,
                    'runtime_name_format': 'bench_{idx}'}
            runtime_dir = os.path.join(runtime_root, 'bench_7')
            os.makedirs(runtime_dir)
            with open(os.path.join(runtime_dir, 'result.json'),
                      'w') as handler:
                json.dump({'energy': MOCK_ENERGY}, handler)
            result = slurm_module.slurm_run(unit)
            # .rt and slurm.sh land at the local-mode location.
            path, suffix = fake_save_state.calls[0]
            self.assertEqual(
                path, os.path.join(runtime_dir, 'bench_7.rt'))
            self.assertTrue(os.path.isfile(
                os.path.join(runtime_dir, 'slurm.sh')))
        self.assertEqual(suffix, 'runtime_7')
        self.assertEqual(result.energy, MOCK_ENERGY)

    def test_isolated_workdir_stamps_units(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        class _Isolated(Dispatcher):
            def runtime_root(self):
                return os.getcwd()

            def runtime_name_format(self):
                return 'bench_{idx}'

            def runtime_isolated_workdir(self):
                return True

        state = _make_state_with_task(jobtype='frequency')
        # Framework-mode task config: the scratch_path injection
        # lands in the flat skill_config dict consumed by the
        # backend.
        state.theory[0]['config'] = {
            'skill_config': {'method': 'tully1'}, 'results': {},
        }
        received = []

        def _executor(child_state, runtime_idx=None):
            del runtime_idx
            received.append(child_state)
            return child_state.jobtype

        dispatcher = _Isolated(state=state, local_executor=_executor)
        tasks = [{'coordinates': [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                 {'coordinates': [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]}]
        with _TempWorkdir():
            workdir = os.getcwd()
            dispatcher.dispatch(units=tasks, child_jobtype='energy')
            # Custom root + template: the unit directories sit right
            # under the cwd, no shared runtime/ subdirectory.
            self.assertEqual(sorted(os.listdir(workdir)),
                             ['bench_1', 'bench_2'])
            for child, name in zip(received, ['bench_1', 'bench_2']):
                unit_dir = os.path.join(workdir, name)
                self.assertEqual(child.runtime.workdir, unit_dir)
                # Isolation marks travel on the child runtime.
                self.assertTrue(child.runtime.isolated_workdir)
                # scratch_path injection points the backend calcdir
                # into the unit directory.
                skill_config = child.theory[0]['config']['skill_config']
                self.assertEqual(skill_config['scratch_path'], unit_dir)

    def test_isolated_workdir_drops_execution_suffix(self):
        from pyrai3md.workflow.calculator import Calculator

        state = _make_state_with_task(jobtype='energy')
        calculator = Calculator(state=state, runtime_idx='runtime_3')
        task = state.theory[0]
        self.assertEqual(calculator.compose_execution_id(task),
                         'qm_layer0_test_runtime_3')
        # Isolated units keep the plain static id (their directory
        # is already unique).
        state.runtime.isolated_workdir = True
        self.assertEqual(calculator.compose_execution_id(task),
                         'qm_layer0_test')

    def test_aimd_runtime_policy_uses_title_directories(self):
        from pyrai3md.workflow import AIMD

        state = _make_state_with_task(jobtype='md')
        state.pmd.control.title = 'benchmark_fssh_tully1'
        aimd = AIMD(calculator=_dummy_calculator(state))
        self.assertTrue(aimd.runtime_isolated_workdir())
        self.assertEqual(aimd.runtime_name_format(),
                         'benchmark_fssh_tully1_{idx}')
        with _TempWorkdir():
            self.assertEqual(aimd.runtime_root(), os.getcwd())


class TestExecutionGraphBuilder(unittest.TestCase):
    """Test graph construction and the universal rules."""

    def _build(self, state, workflow_cls=None):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )
        return ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, workflow_cls)

    def test_single_shot_graph_is_local_base_node(self):
        state = _make_state_with_task(jobtype='energy')
        graph = self._build(state, None)
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph[0]['node'], 'base')
        self.assertEqual(graph[0]['name'], 'base.theory')
        self.assertEqual(graph[0]['role'], 'theory')
        self.assertFalse(graph[0]['multitask'])
        self.assertFalse(graph[0]['remotetask'])
        self.assertEqual(graph[0]['multipool'], 1)
        self.assertIs(state.execution_graph, graph)

    def test_single_shot_graph_skips_roles_without_tasks(self):
        state = _make_mock_state(jobtype='energy')
        graph = self._build(state, None)
        # No theory/reference tasks: no base nodes are planned.
        self.assertEqual(graph, [])

    def test_mixed_top_level_and_staged_base_nodes(self):
        from pyrai3md.workflow import _graph_base_flags

        # The planner emits one shape per graph; the collector must
        # read both because a multi-stage workflow nests its base
        # nodes. Merging is by role, never across roles.
        graph = [
            {'node': 'base', 'role': 'theory', 'multitask': False,
             'remotetask': False, 'multipool': 1},
            {'node': 'workflow', 'name': 'adaptive', 'stages': [
                {'node': 'base', 'role': 'reference', 'multitask': True,
                 'remotetask': False, 'multipool': 3},
            ]},
        ]
        flags = _graph_base_flags(graph)
        self.assertEqual(set(flags), {'theory', 'reference'})
        self.assertEqual(flags['reference'], (True, False, 3))
        self.assertEqual(flags['theory'], (False, False, 1))

    def test_workflow_node_uses_class_defaults(self):
        from pyrai3md.workflow import Optimize

        state = _make_mock_state(jobtype='optimize')
        graph = self._build(state, Optimize)
        workflow_node = graph[0]
        self.assertEqual(workflow_node['node'], 'workflow')
        self.assertEqual(workflow_node['name'], 'optimize')
        self.assertFalse(workflow_node['multitask'])
        self.assertFalse(workflow_node['remotetask'])
        self.assertEqual(workflow_node['multipool'], 1)

    def test_md_section_flags_reach_the_workflow_node(self):
        from pyrai3md.workflow import AIMD

        # The &md keywords name the trajectory fan-out pool (the
        # ninitcond benchmarks set ncpus -> multitask/multipool).
        # They resolve only through the node name, because
        # WORKFLOW_SECTIONS maps 'md' -> the dynamics section.
        state = _make_state_with_task(jobtype='md')
        state.pmd.dynamics.multitask = True
        state.pmd.dynamics.multipool = 2
        graph = self._build(state, AIMD)
        workflow_node = graph[0]
        self.assertEqual(workflow_node['name'], 'md')
        self.assertTrue(workflow_node['multitask'])
        self.assertEqual(workflow_node['multipool'], 2)

    def test_node_name_out_of_the_tables_drops_the_section_flags(self):
        from pyrai3md.workflow import AIMD

        # Negative control for the case above: a workflow whose name
        # is not a WORKFLOW_SECTIONS key gets no section keywords at
        # all, so the &md request is silently lost. This is what the
        # class-name fallback used to produce ('aimd').
        class _MisnamedAIMD(AIMD):
            name = 'aimd'

        state = _make_state_with_task(jobtype='md')
        state.pmd.dynamics.multitask = True
        state.pmd.dynamics.multipool = 2
        workflow_node = self._build(state, _MisnamedAIMD)[0]
        self.assertEqual(workflow_node['name'], 'aimd')
        self.assertFalse(workflow_node['multitask'])
        self.assertEqual(workflow_node['multipool'], 1)

    def test_path_section_flags_reach_the_workflow_node(self):
        from pyrai3md.workflow import Path

        # The path family is asked for as `optimize.method`, so its node reads
        # the `&optimize` section (`WORKFLOW_SECTIONS['path']`) - one section
        # for both loops. What rides on it is the band's image fan-out: a band
        # evaluates its images as a batch, which is the fan-out `multipool`
        # exists for (`workflow/path.py::_run_band`), so an `&optimize
        # multipool` written for a NEB run has to reach this node and not be
        # dropped because the node's name is newer than the section's.
        state = _make_state_with_task(jobtype='optimize')
        state.pmd.optimize.multipool = 3
        workflow_node = self._build(state, Path)[0]
        self.assertEqual(workflow_node['name'], 'path')
        self.assertEqual(workflow_node['multipool'], 3)
        # The node forces no declaration, so the class default is what answers
        # when the input says nothing (`WORKFLOW_NODE_DECLARATIONS['path']`).
        quiet = _make_state_with_task(jobtype='optimize')
        self.assertEqual(self._build(quiet, Path)[0]['multipool'], 1)

    def test_path_name_out_of_the_tables_drops_the_section_flags(self):
        from pyrai3md.workflow import Path

        # The negative control for the case above: the keywords are resolved
        # through the node name, so a path workflow that did not carry its name
        # in WORKFLOW_SECTIONS would silently lose the request - the same
        # failure the AIMD case above pins, on the node whose name is not its
        # section's (a `path` node reads `optimize`).
        class _MisnamedPath(Path):
            name = 'path_unregistered'

        state = _make_state_with_task(jobtype='optimize')
        state.pmd.optimize.multipool = 3
        workflow_node = self._build(state, _MisnamedPath)[0]
        self.assertEqual(workflow_node['name'], 'path_unregistered')
        self.assertEqual(workflow_node['multipool'], 1)

    def test_workflow_keyword_section_overrides_defaults(self):
        from pyrai3md.workflow import AdaptiveSampling

        state = _make_mock_state(jobtype='adaptive')
        state.pmd.adaptive.multitask = False
        state.pmd.adaptive.remotetask = True
        state.pmd.adaptive.multipool = 4
        graph = self._build(state, AdaptiveSampling)
        workflow_node = graph[0]
        self.assertFalse(workflow_node['multitask'])
        self.assertTrue(workflow_node['remotetask'])
        self.assertEqual(workflow_node['multipool'], 4)

    def test_adaptive_stage_nodes_follow_stage_defaults(self):
        from pyrai3md.workflow import AdaptiveSampling

        state = _make_mock_state(jobtype='adaptive')
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.add_task('reference', _make_task('skill_b', 'b_layer0'))
        # The adaptive section owns the md-stage flag; &md decides only the
        # propagation keywords for an adaptive run.
        state.pmd.adaptive.md_multitask = True
        # reference skill keyword survives into base.reference (no
        # forced declaration); theory declares forced serial.
        state.reference[0]['multitask'] = True
        # The training stage is one remote child runtime when remote_train
        # requests it (it never fans out locally).
        state.pmd.adaptive.remote_train = 1
        graph = self._build(state, AdaptiveSampling)
        stages = {stage['name']: stage for stage in
                  graph[0]['stages']}
        self.assertEqual(
            set(stages),
            {'md', 'base.theory', 'base.reference', 'train'})
        self.assertTrue(stages['md']['multitask'])
        self.assertFalse(stages['md']['remotetask'])
        # base.theory forces serial execution and pins multipool.
        self.assertFalse(stages['base.theory']['multitask'])
        self.assertEqual(stages['base.theory']['multipool'], 1)
        # base.reference keeps the skill-section settings.
        self.assertTrue(stages['base.reference']['multitask'])
        self.assertFalse(stages['base.reference']['remotetask'])
        self.assertFalse(stages['train']['multitask'])
        self.assertTrue(stages['train']['remotetask'])

    def test_calculator_multitask_aggregated_from_skills(self):
        state = _make_state_with_task(jobtype='energy')
        # Dispatch-layer keywords at the task dict top level
        # (extracted from raw input by _build_task_queue, not
        # validated by the skill schema)
        state.theory[0]['multitask'] = True
        state.theory[0]['remotetask'] = True
        state.theory[0]['multipool'] = 3
        graph = self._build(state, None)
        calculator_node = graph[-1]
        # multitask aggregated; remotetask ignored (rule 3: the
        # keyword is reserved for the skill's own local/slurm run)
        self.assertTrue(calculator_node['multitask'])
        self.assertFalse(calculator_node['remotetask'])
        # multipool travels the same task-dict channel
        self.assertEqual(calculator_node['multipool'], 3)

    def test_base_node_skill_keyword_conflict_raises(self):
        state = _make_mock_state(jobtype='energy')
        # Conflicts are per backend role: two theory skills
        # declaring opposite multitask values.
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.add_task('theory', _make_task('skill_b', 'b_layer0'))
        state.theory[0]['multitask'] = True
        state.theory[1]['multitask'] = False
        with self.assertRaises(KeyError):
            self._build(state, None)

    def test_cross_role_flags_do_not_conflict(self):
        state = _make_mock_state(jobtype='energy')
        # Different roles aggregate independently: theory True and
        # reference False build two distinct base nodes.
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.add_task('reference', _make_task('skill_b', 'b_layer0'))
        state.theory[0]['multitask'] = True
        state.reference[0]['multitask'] = False
        graph = self._build(state, None)
        nodes = {node['name']: node for node in graph}
        self.assertTrue(nodes['base.theory']['multitask'])
        self.assertFalse(nodes['base.reference']['multitask'])

    def test_worker_accepts_frozen_local_graph(self):
        from pyrai3md.workflow.dispatch.dispatcher import (
            worker_execution_graph,
        )

        state = _make_mock_state(jobtype='energy')
        state.resolve_execution_graph = False
        state.execution_graph = worker_execution_graph()
        graph = self._build(state, None)
        self.assertEqual(graph, worker_execution_graph())

    def test_worker_rejects_remote_frozen_graph(self):
        state = _make_mock_state(jobtype='energy')
        state.resolve_execution_graph = False
        state.execution_graph = [{'node': 'workflow', 'name': 'train',
                                  'multitask': False,
                                  'remotetask': True}]
        with self.assertRaises(RuntimeError):
            self._build(state, None)

    def test_worker_requires_frozen_graph(self):
        state = _make_mock_state(jobtype='energy')
        state.resolve_execution_graph = False
        state.execution_graph = None
        with self.assertRaises(RuntimeError):
            self._build(state, None)


class TestBackendExecutionDescriptor(unittest.TestCase):
    """Test resolve_molecule_flags and backend_execution_descriptor."""

    def _build(self, state, workflow_cls=None):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )
        return ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, workflow_cls)

    # ---- resolve_molecule_flags ----

    def test_molecule_none_aggregates_skills(self):
        from pyrai3md.workflow.dispatch.planner import (
            resolve_molecule_flags,
        )
        state = _make_mock_state(jobtype='energy')
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.theory[0]['multitask'] = True
        state.theory[0]['multipool'] = 3
        multitask, remotetask, multipool, log = resolve_molecule_flags(
            state, 'theory')
        self.assertTrue(multitask)
        self.assertFalse(remotetask)
        self.assertEqual(multipool, 3)
        origins = {rec['flag']: rec['origin'] for rec in log}
        self.assertEqual(origins['multitask'], 'skill')
        self.assertEqual(origins['multipool'], 'skill')
        self.assertEqual(origins['remotetask'], 'default')

    def test_molecule_explicit_overrides_skills(self):
        from pyrai3md.workflow.dispatch.planner import (
            resolve_molecule_flags,
        )
        state = _make_mock_state(jobtype='energy')
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.theory[0]['multitask'] = True
        state.pmd.system.multitask = False
        state.pmd.system.multipool = 2
        multitask, _, multipool, log = resolve_molecule_flags(
            state, 'theory')
        self.assertFalse(multitask)
        self.assertEqual(multipool, 2)
        origins = {rec['flag']: rec['origin'] for rec in log}
        self.assertEqual(origins['multitask'], 'molecule')
        self.assertEqual(origins['multipool'], 'molecule')

    def test_molecule_explicit_resolves_skill_conflict(self):
        from pyrai3md.workflow.dispatch.planner import (
            resolve_molecule_flags,
        )
        state = _make_mock_state(jobtype='energy')
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.add_task('theory', _make_task('skill_b', 'b_layer0'))
        state.theory[0]['multitask'] = True
        state.theory[1]['multitask'] = False
        # Undecided molecule scope: the skills conflict.
        with self.assertRaises(KeyError):
            resolve_molecule_flags(state, 'theory')
        # An explicit molecule value settles the conflict.
        state.pmd.system.multitask = True
        multitask, _, _, _ = resolve_molecule_flags(state, 'theory')
        self.assertTrue(multitask)

    def test_remotetask_is_kept_not_dropped(self):
        from pyrai3md.workflow.dispatch.planner import (
            resolve_molecule_flags,
        )
        state = _make_mock_state(jobtype='energy')
        state.add_task('theory', _make_task('skill_a', 'a_layer0'))
        state.theory[0]['remotetask'] = True
        _, remotetask, _, _ = resolve_molecule_flags(state, 'theory')
        self.assertTrue(remotetask)

    # ---- backend_execution_descriptor ----

    def test_descriptor_shape_and_defaults(self):
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        tasks = [{'task_slot': 0, 'skill_name': 'a'},
                 {'task_slot': 1, 'skill_name': 'b'}]
        desc = backend_execution_descriptor(
            'theory', tasks, (True, False, 2, []))
        self.assertEqual(desc['scope'], 'backend_terms')
        self.assertEqual(desc['role'], 'theory')
        self.assertEqual(desc['task_slots'], [0, 1])
        self.assertEqual(desc['executor'], 'serial')
        # Serial local role provisions no workers.
        self.assertEqual(desc['max_workers'], 1)
        self.assertEqual(desc['failure_policy'],
                         'fail_fast_collect_diagnostics')
        self.assertTrue(desc['multitask'])
        self.assertFalse(desc['remotetask'])
        self.assertEqual(desc['multipool'], 2)

    def test_thread_executor_capacity(self):
        from pyrai3md.schema.config import Composite
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        composite = Composite(executor='thread', max_workers=4)
        desc = backend_execution_descriptor(
            'theory', [], (True, False, 1, []), composite=composite)
        self.assertEqual(desc['executor'], 'thread')
        self.assertEqual(desc['max_workers'], 4)

    def test_remote_terms_provision_workers(self):
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        # remotetask=True fans terms out even with a serial executor.
        desc = backend_execution_descriptor(
            'theory', [], (True, True, 3, []))
        self.assertTrue(desc['remotetask'])
        self.assertEqual(desc['max_workers'], 3)

    def test_molecule_false_multitask_with_thread_raises(self):
        from pyrai3md.schema.config import Composite, System
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        composite = Composite(executor='thread', max_workers=2)
        molecule = System(multitask=False)
        with self.assertRaises(ValueError):
            backend_execution_descriptor(
                'theory', [], (False, False, 1, []),
                composite=composite, molecule=molecule)

    def test_molecule_multipool_conflict_raises(self):
        from pyrai3md.schema.config import Composite, System
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        composite = Composite(executor='thread', max_workers=2)
        molecule = System(multipool=3)
        with self.assertRaises(ValueError):
            backend_execution_descriptor(
                'theory', [], (True, False, 3, []),
                composite=composite, molecule=molecule)

    def test_molecule_multipool_agreement_ok(self):
        from pyrai3md.schema.config import Composite, System
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        composite = Composite(executor='thread', max_workers=3)
        molecule = System(multipool=3)
        desc = backend_execution_descriptor(
            'theory', [], (True, False, 3, []),
            composite=composite, molecule=molecule)
        self.assertEqual(desc['max_workers'], 3)

    def test_unknown_executor_raises(self):
        from pyrai3md.schema.config import Composite
        from pyrai3md.workflow.dispatch.planner import (
            backend_execution_descriptor,
        )
        composite = Composite(executor='process')
        with self.assertRaises(ValueError):
            backend_execution_descriptor(
                'theory', [], (True, False, 1, []), composite=composite)

    # ---- base node integration ----

    def test_base_node_carries_descriptor(self):
        state = _make_state_with_task(jobtype='energy')
        graph = self._build(state, None)
        node = graph[0]
        self.assertEqual(node['node'], 'base')
        desc = node['backend_execution']
        self.assertEqual(desc['scope'], 'backend_terms')
        self.assertEqual(desc['role'], 'theory')
        # Rule 3: the whole base node stays local.
        self.assertFalse(node['remotetask'])

    def test_base_node_descriptor_keeps_remotetask(self):
        state = _make_state_with_task(jobtype='energy')
        state.theory[0]['remotetask'] = True
        node = self._build(state, None)[0]
        # Top level pinned local (rule 3) but the backend_terms
        # descriptor records the remote per-term intent (r3e).
        self.assertFalse(node['remotetask'])
        self.assertTrue(node['backend_execution']['remotetask'])

    def test_declared_serial_base_keeps_inner_thread(self):
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )
        state = _make_state_with_task(jobtype='md')
        state.pmd.composite.executor = 'thread'
        state.pmd.composite.max_workers = 2
        builder = ExecutionGraphBuilder()
        # The md base.theory declaration forces multitask=False.
        node = builder._base_node(state, 'theory', {'multitask': False})
        self.assertFalse(node['multitask'])
        self.assertEqual(node['multipool'], 1)
        # The outer multitask and the inner term executor are
        # orthogonal: the thread pool still stands (r3f).
        desc = node['backend_execution']
        self.assertEqual(desc['executor'], 'thread')
        self.assertEqual(desc['max_workers'], 2)
        self.assertFalse(desc['multitask'])

    def test_worker_graph_descriptor_is_serial(self):
        from pyrai3md.workflow.dispatch.dispatcher import (
            worker_execution_graph,
        )
        desc = worker_execution_graph()[0]['backend_execution']
        self.assertEqual(desc['scope'], 'backend_terms')
        self.assertEqual(desc['executor'], 'serial')
        self.assertFalse(desc['multitask'])
        self.assertFalse(desc['remotetask'])
        self.assertEqual(desc['max_workers'], 1)


class TestUniversalRules(unittest.TestCase):
    """Test the three universal child task rules."""

    def test_rule1_materialized_children_are_workers(self):
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        state = _make_state_with_task(jobtype='search')
        dispatcher = Dispatcher(state=state,
                                local_executor=lambda s: s.jobtype)
        unit = dispatcher.materialize_unit(
            {'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}, 0,
            child_jobtype='energy')
        self.assertTrue(unit['state'].runtime.worker)
        self.assertFalse(unit['state'].resolve_execution_graph)

    def test_rule2_worker_graph_rejects_remotetask(self):
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )

        state = _make_mock_state(jobtype='energy')
        state.resolve_execution_graph = False
        state.execution_graph = [{'node': 'calculator',
                                  'name': 'calculator',
                                  'multitask': False,
                                  'remotetask': True}]
        with self.assertRaises(RuntimeError):
            ExecutionGraphBuilder({}).build(state, None)

    def test_rule3_calculator_rejects_remotetask(self):
        from pyrai3md.workflow.calculator import Calculator

        with self.assertRaises(RuntimeError):
            Calculator(state=_make_mock_state(), remotetask=True)

    def test_rule3_graph_builder_keeps_calculator_local(self):
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder, resolve_molecule_flags,
        )

        state = _make_state_with_task(jobtype='energy')
        state.theory[0]['multitask'] = True
        state.theory[0]['remotetask'] = True
        # The molecule keywords aggregate multitask/multipool; the
        # remotetask keyword is reserved for the skill's own
        # local/slurm run (rule 3), which the base node below must not
        # inherit.
        multitask, _, multipool, _ = resolve_molecule_flags(state, 'theory')
        self.assertTrue(multitask)
        self.assertEqual(multipool, 1)
        graph = ExecutionGraphBuilder({}).build(state, None)
        base_node = graph[-1]
        self.assertEqual(base_node['node'], 'base')
        self.assertFalse(base_node['remotetask'])

    def test_runtime_state_worker_serialization(self):
        from pyrai3md.data.runtime.state import RuntimeState

        runtime = RuntimeState(worker=True)
        restored = RuntimeState.from_dict(runtime.to_dict())
        self.assertTrue(restored.worker)
        # Default root runtime is not a worker
        self.assertFalse(RuntimeState().worker)

    def test_runtime_state_legacy_dispatchable_compat(self):
        from pyrai3md.data.runtime.state import RuntimeState

        restored = RuntimeState.from_dict({'dispatchable': False})
        self.assertTrue(restored.worker)
        restored = RuntimeState.from_dict({'dispatchable': True})
        self.assertFalse(restored.worker)


def _dummy_calculator(state):
    """Create a Calculator serving a shared counting backend."""
    from pyrai3md.workflow.calculator import Calculator

    return Calculator(state=state, dummy_backend=_CountingBackend())


class TestCalculatorRoles(unittest.TestCase):
    """Test backend role grouping and active role switching."""

    def _make_two_role_state(self):
        state = _make_state_with_task(jobtype='energy')
        state.add_task('reference',
                       _make_task('ref_skill', 'ref_layer0_test'))
        return state

    def test_initialize_groups_backends_by_category(self):
        from pyrai3md.workflow.calculator import Calculator

        state = self._make_two_role_state()
        calculator = Calculator(state=state,
                                dummy_backend=_CountingBackend())
        calculator.initialize()
        self.assertEqual(len(calculator.backends['theory']), 1)
        self.assertEqual(len(calculator.backends['reference']), 1)
        self.assertEqual(calculator.execution_ids['theory'],
                         ['qm_layer0_test'])
        self.assertEqual(calculator.execution_ids['reference'],
                         ['ref_layer0_test'])

    def test_default_role_is_theory(self):
        from pyrai3md.workflow.calculator import Calculator

        calculator = Calculator(state=self._make_two_role_state())
        self.assertEqual(calculator.active_role, 'theory')

    def test_switch_role_unknown_raises(self):
        from pyrai3md.workflow.calculator import Calculator

        calculator = Calculator(state=self._make_two_role_state())
        with self.assertRaises(ValueError):
            calculator.switch_role('auxiliary')

    def test_initial_role_runs_reference(self):
        from pyrai3md.workflow.calculator import Calculator

        state = self._make_two_role_state()
        theory_backend = _CountingBackend(energy=-1.0)
        reference_backend = _CountingBackend(energy=-2.0)
        calculator = Calculator(state=state, initial_role='reference')
        calculator.backends = {'theory': [theory_backend],
                               'reference': [reference_backend]}
        calculator.run()
        # Negative control is test_default_role_is_theory above: the
        # same state with no initial_role runs theory only.
        self.assertEqual(calculator.active_role, 'reference')
        self.assertEqual(len(theory_backend.calls), 0)
        self.assertEqual(len(reference_backend.calls), 1)
        self.assertEqual(state.pmd.results['energy'], -2.0)

    def test_initial_role_unknown_raises(self):
        from pyrai3md.workflow.calculator import Calculator

        with self.assertRaises(ValueError):
            Calculator(state=self._make_two_role_state(),
                       initial_role='auxiliary')

    def test_initial_role_with_empty_queue_raises(self):
        from pyrai3md.workflow.calculator import Calculator

        # A reference runtime without reference tasks used to reach
        # _run_single and fail on a missing backend; it now names the
        # real cause at construction.
        state = _make_state_with_task(jobtype='energy')
        with self.assertRaises(ValueError):
            Calculator(state=state, initial_role='reference')

    def test_materialized_reference_unit_runs_reference(self):
        from pyrai3md.workflow.calculator import Calculator
        from pyrai3md.workflow.dispatch.dispatcher import Dispatcher

        state = self._make_two_role_state()
        unit = Dispatcher(state=state).materialize_unit(
            {'coordinates': [[0.3, 0.0, 0.0], [1.0, 0.0, 0.0]]}, 0,
            child_jobtype='gradient', active_role='reference')
        child = unit['state']
        backend = _CountingBackend(energy=-3.0)
        calculator = Calculator(state=child,
                                initial_role=child.runtime.active_role,
                                dummy_backend=backend)
        calculator.initialize()
        calculator.run()
        # The full unit path: role travels, the theory queue is gone,
        # and the evaluation runs at the injected geometry.
        self.assertEqual(calculator.active_role, 'reference')
        self.assertEqual(child.theory, [])
        self.assertEqual(len(backend.calls), 1)
        self.assertAlmostEqual(backend.calls[0][1][0][0], 0.3)

    def test_run_executes_active_role_only(self):
        from pyrai3md.workflow.calculator import Calculator

        state = self._make_two_role_state()
        theory_backend = _CountingBackend(energy=-1.0)
        reference_backend = _CountingBackend(energy=-2.0)
        calculator = Calculator(state=state)
        calculator.backends = {'theory': [theory_backend],
                               'reference': [reference_backend]}
        calculator.run()
        # The two roles are never executed simultaneously
        self.assertEqual(len(theory_backend.calls), 1)
        self.assertEqual(len(reference_backend.calls), 0)

        calculator.switch_role('reference')
        calculator.run()
        self.assertEqual(len(reference_backend.calls), 1)
        self.assertEqual(state.pmd.results['energy'], -2.0)

    def test_update_routes_state_updates(self):
        from pyrai3md.workflow.calculator import Calculator

        state = _make_mock_state(jobtype='energy')
        calculator = Calculator(state=state)
        calculator.update({'results': {'energy': MOCK_ENERGY}})
        self.assertEqual(state.result['energy'], MOCK_ENERGY)
        calculator.update({'system': {'charges': 1}})
        self.assertEqual(state.charges, 1)
        self.assertIs(calculator.get_state(), state)


class TestAdaptiveChain(unittest.TestCase):
    """Test the adaptive three-stage chain end-to-end."""

    def _make_adaptive_state(self):
        state = _make_mock_state(jobtype='adaptive')
        state.runtime.total_steps = 1
        state.add_task('theory', _make_task('theory_skill',
                                            'theory_layer0'))
        state.add_task('reference', _make_task('ref_skill',
                                               'ref_layer0'))
        return state

    def test_parent_calculator_is_not_used(self):
        """The adaptive node orchestrates children; it never evaluates."""
        from pyrai3md.workflow.adaptive import AdaptiveSampling

        state = self._make_adaptive_state()
        backend = _CountingBackend()
        from pyrai3md.workflow.calculator import Calculator
        calculator = Calculator(state=state, dummy_backend=backend)
        calculator.initialize()

        workflow = AdaptiveSampling(calculator=calculator)
        # No installed skill is named 'theory_skill', so the trainable
        # task lookup refuses the run. The refusal is the assertion: the
        # parent calculator was never asked to compute anything. It is
        # also the assertion that a refused run leaves nothing behind:
        # the summary used to be written (and its directory created)
        # before the refusal, which planted `pyrai3md.adaptive/` in
        # whatever directory the test ran in.
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as work:
            os.chdir(work)
            try:
                with self.assertRaises(ValueError):
                    workflow.run()
            finally:
                os.chdir(previous)
            self.assertEqual(os.listdir(work), [],
                             'a refused adaptive run left files behind')
        self.assertEqual(backend.calls, [])
        self.assertEqual(state.runtime.status, 'failed')

    def test_missing_reference_task_is_refused(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        from pyrai3md.workflow.calculator import Calculator

        state = _make_mock_state(jobtype='adaptive')
        state.add_task('theory', _make_task('theory_skill',
                                            'theory_layer0'))
        backend = _CountingBackend()
        calculator = Calculator(state=state, dummy_backend=backend)
        calculator.initialize()

        with self.assertRaises(ValueError):
            AdaptiveSampling(calculator=calculator).run()
        self.assertEqual(backend.calls, [])

    def test_execute_skips_parent_backend_initialization(self):
        """needs_backends=False keeps the parent process backend-free."""
        from pyrai3md.workflow import execute
        from pyrai3md.workflow.calculator import Calculator

        state = self._make_adaptive_state()
        original = Calculator.initialize
        calls = []

        def refuse(self):
            calls.append(True)
            raise AssertionError('the adaptive parent initialized a backend')

        Calculator.initialize = refuse
        try:
            with self.assertRaises(ValueError):
                execute(state)
        finally:
            Calculator.initialize = original
        self.assertEqual(calls, [])

    def test_adaptive_graph_reaches_base_nodes(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )

        state = self._make_adaptive_state()
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, AdaptiveSampling)
        # The base nodes live inside the adaptive stage list.
        node_types = [node['node'] for node in graph]
        self.assertEqual(node_types, ['workflow'])
        stages = {stage['name']: stage for stage in
                  graph[0]['stages']}
        self.assertIn('base.theory', stages)
        self.assertIn('base.reference', stages)
        for stage in stages.values():
            if stage['node'] == 'base':
                self.assertFalse(stage['remotetask'])

    def test_staged_base_nodes_reach_role_flags(self):
        from pyrai3md.workflow import (_graph_backend_execution,
                                       _graph_base_flags, WORKFLOW_REGISTRY)
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )

        state = self._make_adaptive_state()
        # base.theory is declared serial by stage_defaults; the
        # reference stage takes the task-level request. Different
        # numbers make a cross-role merge impossible to pass.
        state.reference[0]['multitask'] = True
        state.reference[0]['multipool'] = 2
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, AdaptiveSampling)
        flags = _graph_base_flags(graph)
        self.assertEqual(set(flags), {'theory', 'reference'})
        self.assertFalse(flags['theory'][0])
        self.assertEqual(flags['theory'][2], 1)
        self.assertTrue(flags['reference'][0])
        self.assertEqual(flags['reference'][2], 2)
        self.assertEqual(set(_graph_backend_execution(graph)),
                         {'theory', 'reference'})

    def test_reference_stage_applies_its_base_flags(self):
        from pyrai3md.workflow import (_graph_backend_execution,
                                       _graph_base_flags, WORKFLOW_REGISTRY)
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        from pyrai3md.workflow.calculator import Calculator
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )

        state = self._make_adaptive_state()
        state.reference[0]['multitask'] = True
        state.reference[0]['multipool'] = 2
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, AdaptiveSampling)
        descriptors = _graph_backend_execution(graph)
        calculator = Calculator(
            state=state, dummy_backend=_CountingBackend(),
            base_flags=_graph_base_flags(graph),
            backend_execution=descriptors)
        calculator.switch_role('reference')
        self.assertTrue(calculator.multitask)
        self.assertEqual(calculator.multipool, 2)
        # The term executor is the reference descriptor, not the
        # theory one (the two roles keep distinct contracts).
        self.assertIs(calculator._active_descriptor(),
                      descriptors['reference'])


class TestStateSerializer(unittest.TestCase):
    """Test .rt serialization round-trip via StateSerializer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrai3md_runtime_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip_state_equivalence(self):
        from pyrai3md.data.runtime.serializer import load_state, save_state

        state = _make_state_with_task(jobtype='gradient')
        state.pmd.results = {'energy': MOCK_ENERGY}
        path = os.path.join(self.tmpdir, 'test.rt')

        save_state(state, path, runtime_idx='runtime_3')
        loaded, runtime_idx = load_state(path)

        self.assertEqual(runtime_idx, 'runtime_3')
        self.assertEqual(loaded.jobtype, 'gradient')
        self.assertEqual(list(loaded.atoms), ['C', 'H'])
        self.assertEqual(list(loaded.coordinates),
                         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(loaded.pmd.results, {'energy': MOCK_ENERGY})
        self.assertEqual(len(loaded.theory), 1)
        # Task metadata preserved
        task = loaded.theory[0]
        self.assertEqual(task['skill_name'], 'test_skill')
        self.assertEqual(task['static_id'], 'qm_layer0_test')
        # Original state never mutated
        self.assertEqual(state.pmd.results, {'energy': MOCK_ENERGY})

    def test_worker_contract_travels_with_rt(self):
        from pyrai3md.data.runtime.serializer import load_state, save_state
        from pyrai3md.workflow.dispatch.dispatcher import (
            worker_execution_graph,
        )

        state = _make_state_with_task(jobtype='energy')
        state.runtime.worker = True
        state.resolve_execution_graph = False
        state.execution_graph = worker_execution_graph()
        path = os.path.join(self.tmpdir, 'worker.rt')

        save_state(state, path)
        loaded, _ = load_state(path)
        self.assertTrue(loaded.runtime.worker)
        self.assertFalse(loaded.resolve_execution_graph)
        self.assertEqual(loaded.execution_graph, worker_execution_graph())

    def test_backend_role_travels_with_rt(self):
        from pyrai3md.data.runtime.serializer import load_state, save_state
        from pyrai3md.workflow.dispatch.dispatcher import (
            worker_execution_graph,
        )

        state = _make_state_with_task(jobtype='energy')
        state.runtime.worker = True
        state.runtime.active_role = 'reference'
        state.resolve_execution_graph = False
        state.execution_graph = worker_execution_graph()
        path = os.path.join(self.tmpdir, 'reference_worker.rt')
        save_state(state, path)
        loaded, _ = load_state(path)
        self.assertEqual(loaded.runtime.active_role, 'reference')

        # Negative side: a root state with no explicit role still reads
        # as theory after the round trip.
        root = _make_state_with_task(jobtype='energy')
        root_path = os.path.join(self.tmpdir, 'root.rt')
        save_state(root, root_path)
        root_loaded, _ = load_state(root_path)
        self.assertEqual(root_loaded.runtime.active_role, 'theory')

    def test_task_config_converted_to_dict(self):
        from pydantic import BaseModel

        from pyrai3md.data.runtime.serializer import load_state, save_state

        state = _make_state_with_task(jobtype='energy')
        self.assertIsInstance(state.theory[0]['config'], BaseModel)
        path = os.path.join(self.tmpdir, 'test.rt')

        save_state(state, path)
        loaded, _ = load_state(path)

        # Dynamic extended configs cannot be pickled by reference;
        # they are stored as plain dictionaries.
        self.assertIsInstance(loaded.theory[0]['config'], dict)
        # Original config object untouched
        self.assertIsInstance(state.theory[0]['config'], BaseModel)

    def test_no_backend_residue(self):
        from pyrai3md.data.runtime.serializer import load_state, save_state

        state = _make_state_with_task(jobtype='energy')
        path = os.path.join(self.tmpdir, 'test.rt')
        save_state(state, path)
        loaded, _ = load_state(path)

        # Backends are runtime objects of Calculator, never stored
        self.assertFalse(hasattr(loaded, 'backends'))
        for task in loaded.theory:
            self.assertNotIn('backend', task)

    def test_load_missing_file_raises(self):
        from pyrai3md.data.runtime.serializer import load_state
        with self.assertRaises(FileNotFoundError):
            load_state(os.path.join(self.tmpdir, 'missing.rt'))

    def test_load_invalid_payload_raises(self):
        import pickle

        from pyrai3md.data.runtime.serializer import load_state
        path = os.path.join(self.tmpdir, 'invalid.rt')
        with open(path, 'wb') as handler:
            pickle.dump({'not_state': 1}, handler)
        with self.assertRaises(ValueError):
            load_state(path)


class TestTermWorker(unittest.TestCase):
    """Remote per-term worker (docs R3, r3e).

    Gate: the term-worker entry executes exactly ONE backend term and
    returns the unified TaskOutcome; it refuses a nested remote
    submission; the payload/outcome round-trip preserves identity,
    geometry and array-typed results.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrai3md_term_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _module(self):
        import importlib
        return importlib.import_module(
            'pyrai3md.workflow.dispatch.term_worker')

    def _api_path(self):
        return '%s._CountingBackend' % __name__

    def _make_work_item(self, task_id='qm_term', cell=None):
        import numpy as np

        from pyrai3md.workflow.dispatch.terms import make_work_item
        return make_work_item(
            task_id=task_id,
            backend=None,
            local_atoms=['C', 'H'],
            local_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            task_slot=0, center_index=0, evaluation_id=7,
            target_states=[0],
            cell=cell,
            identity={'trajectory': 'runtime_1', 'task_slot': 0,
                      'center_index': 0, 'evaluation_id': 7},
            backend_name='test_skill', runner='in_process',
            execution_mode='local', profile_semantics='energy')

    def _make_term_task(self, remotetask=False):
        from pyrai3md.schema.config import PyRAIConfig

        return {
            'role': 'qm', 'task_id': 'qm_term',
            'static_id': 'qm_layer0_test', 'skill_name': 'test_skill',
            'api': self._api_path(), 'capabilities': ['energy'],
            'config': PyRAIConfig(), 'region': 'all',
            'region_indices': [], 'remotetask': remotetask,
        }

    def test_execute_term_locally_runs_single_term(self):
        term = self._module()
        payload = term.build_term_payload(
            self._make_work_item(), self._make_term_task(), 'runtime_3')
        # The remote runner is stamped on the audit key, and the live
        # backend is never shipped (rebuilt from the api path).
        self.assertEqual(payload['audit_key']['runner'], 'term_worker')
        self.assertNotIn('backend', payload)
        self.assertEqual(payload['execution_id'],
                         'qm_layer0_test_runtime_3')

        outcome = term.execute_term_locally(payload)
        self.assertTrue(outcome['success'])
        self.assertEqual(outcome['task_id'], 'qm_term')
        self.assertEqual(outcome['evaluation_id'], 7)
        self.assertEqual(outcome['raw'], {'energy': MOCK_ENERGY})

    def test_run_term_worker_writes_outcome_json(self):
        import json

        term = self._module()
        payload = term.build_term_payload(
            self._make_work_item(), self._make_term_task(), 'runtime_3')
        payload_path = os.path.join(self.tmpdir, 'term.rtt')
        outcome_path = os.path.join(self.tmpdir, 'term_outcome.json')
        term.write_term_payload(payload, payload_path)

        outcome = term.run_term_worker(payload_path, outcome_path)
        self.assertTrue(outcome['success'])
        self.assertTrue(os.path.isfile(outcome_path))
        with open(outcome_path) as handler:
            written = json.load(handler)
        self.assertEqual(written['task_id'], 'qm_term')
        self.assertEqual(written['evaluation_id'], 7)
        self.assertEqual(written['raw']['energy'], MOCK_ENERGY)

    def test_payload_round_trip_preserves_identity_and_geometry(self):
        import numpy as np

        term = self._module()
        payload = term.build_term_payload(
            self._make_work_item(), self._make_term_task(), 'runtime_3')
        path = os.path.join(self.tmpdir, 'term.rtt')
        term.write_term_payload(payload, path)
        loaded = term.read_term_payload(path)

        self.assertEqual(loaded['identity'], payload['identity'])
        self.assertEqual(loaded['geometry']['local_atoms'], ['C', 'H'])
        self.assertTrue(np.array_equal(
            loaded['geometry']['local_coords'],
            payload['geometry']['local_coords']))
        self.assertEqual(loaded['audit_key'], payload['audit_key'])
        self.assertEqual(loaded['schema_version'],
                         term.TERM_PAYLOAD_VERSION)

    def test_payload_carries_cell_snapshot(self):
        # The frozen periodic-state snapshot travels in the geometry
        # section (plain dict, pickle-safe) so the remote term worker
        # evaluates with the same cell as the parent step. The dict is
        # opaque to the payload, which is why adding the crossing counts
        # it now carries did not move the payload version.
        term = self._module()
        cell = {'lattice_vector': [[10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]],
                'cell': [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
                'pbc': [1, 1, 1], 'volume': 1000.0,
                'image': [[1, 0, 0], [0, 0, 0]]}
        payload = term.build_term_payload(
            self._make_work_item(cell=cell), self._make_term_task(),
            'runtime_3')
        path = os.path.join(self.tmpdir, 'term_cell.rtt')
        term.write_term_payload(payload, path)
        loaded = term.read_term_payload(path)
        self.assertEqual(loaded['geometry']['cell'], cell)

    def test_payload_cell_defaults_none(self):
        term = self._module()
        payload = term.build_term_payload(
            self._make_work_item(), self._make_term_task(), 'runtime_3')
        self.assertIsNone(payload['geometry']['cell'])

    def test_build_term_payload_refuses_nested_remote(self):
        term = self._module()
        with self.assertRaises(RuntimeError):
            term.build_term_payload(
                self._make_work_item(),
                self._make_term_task(remotetask=True), 'runtime_3')

    def test_execute_term_locally_refuses_nested_remote(self):
        term = self._module()
        payload = term.build_term_payload(
            self._make_work_item(), self._make_term_task(), 'runtime_3')
        # A hand-crafted payload that still declares remotetask=True is
        # refused before any backend is rebuilt (one remote boundary).
        payload['task']['remotetask'] = True
        with self.assertRaises(RuntimeError):
            term.execute_term_locally(payload)

    def test_json_safe_covers_gradient_arrays(self):
        import json

        import numpy as np

        term = self._module()
        raw = {
            'energy': np.float64(MOCK_ENERGY),
            'gradient': np.ones((2, 3), dtype=np.float64),
            'nested': [np.arange(3, dtype=np.float64), {'soc': np.zeros(2)}],
        }
        safe = term._json_safe(raw)
        # Fully JSON-serializable (no numpy residue at any depth).
        json.dumps(safe)
        self.assertEqual(safe['energy'], MOCK_ENERGY)
        self.assertEqual(safe['gradient'], [[1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0]])
        # Array-typed keys are restored to ndarray on the parent side.
        restored = term._restore_arrays(
            {'gradient': safe['gradient'], 'energy': safe['energy']})
        self.assertIsInstance(restored['gradient'], np.ndarray)
        self.assertTrue(np.array_equal(
            restored['gradient'], np.ones((2, 3))))

    def test_slurm_term_run_collects_outcome(self):
        import json
        import subprocess
        from unittest import mock

        import numpy as np

        term = self._module()
        state = _make_state_with_task(jobtype='energy')
        work_item = self._make_work_item()
        task = self._make_term_task()

        def fake_run(command, **kwargs):
            if command[0] == 'sbatch':
                # Simulate the child worker writing its outcome into
                # the per-term runtime directory (the sbatch cwd).
                outcome = {
                    'task_id': 'qm_term', 'evaluation_id': 7,
                    'raw': {'energy': MOCK_ENERGY,
                            'gradient': np.ones((2, 3)).tolist()},
                    'success': True, 'error': None,
                }
                with open(os.path.join(kwargs['cwd'],
                                       'term_outcome.json'), 'w') as fh:
                    json.dump(outcome, fh)
                return subprocess.CompletedProcess(
                    command, 0, stdout='Submitted batch job 42\n')
            return subprocess.CompletedProcess(
                command, 0, stdout='COMPLETED\n')

        with _TempWorkdir() as workdir, \
                mock.patch.object(term.subprocess, 'run',
                                  side_effect=fake_run), \
                mock.patch.object(term, '_query_slurm_state',
                                  return_value='COMPLETED'):
            outcome = term.slurm_term_run(work_item, task, state, {})
            workdir = os.getcwd()
            runtime_root = os.path.join(workdir, 'runtime')
            names = sorted(os.listdir(runtime_root))

        self.assertTrue(outcome['success'])
        self.assertEqual(outcome['task_id'], 'qm_term')
        self.assertEqual(outcome['evaluation_id'], 7)
        self.assertEqual(outcome['raw']['energy'], MOCK_ENERGY)
        # The gradient array survives the JSON round-trip as ndarray.
        self.assertIsInstance(outcome['raw']['gradient'], np.ndarray)
        # Per-term identity (center/slot/evaluation) names the runtime
        # directory so concurrent terms never contend.
        self.assertEqual(len(names), 1)
        self.assertIn('test_skill_c0_s0_e7', names[0])

    def test_slurm_term_run_missing_outcome_reports_failure(self):
        import subprocess
        from unittest import mock

        term = self._module()
        state = _make_state_with_task(jobtype='energy')

        def fake_run(command, **kwargs):
            del kwargs
            if command[0] == 'sbatch':
                return subprocess.CompletedProcess(
                    command, 0, stdout='Submitted batch job 42\n')
            return subprocess.CompletedProcess(
                command, 0, stdout='FAILED\n')

        with _TempWorkdir(), \
                mock.patch.object(term.subprocess, 'run',
                                  side_effect=fake_run), \
                mock.patch.object(term, '_query_slurm_state',
                                  return_value='FAILED'):
            outcome = term.slurm_term_run(
                self._make_work_item(), self._make_term_task(), state, {})
        self.assertFalse(outcome['success'])
        self.assertIn('term_outcome.json', outcome['error'])

    def test_execute_remote_terms_fail_fast_stops_group(self):
        from unittest import mock

        term = self._module()
        state = _make_state_with_task(jobtype='energy')
        items = [self._make_work_item(task_id='t0'),
                 self._make_work_item(task_id='t1')]
        tasks = [self._make_term_task(), self._make_term_task()]
        calls = []

        def fake_slurm_term_run(work_item, task, state_, descriptor=None):
            del task, state_, descriptor
            calls.append(work_item['task_id'])
            success = work_item['task_id'] != 't0'
            return {'task_id': work_item['task_id'], 'evaluation_id': 7,
                    'raw': {}, 'success': success,
                    'error': None if success else 'boom'}

        with mock.patch.object(term, 'slurm_term_run',
                               side_effect=fake_slurm_term_run):
            outcomes = term.execute_remote_terms(items, tasks, state, {})
        # The first failing term stops the group (no second job).
        self.assertEqual(calls, ['t0'])
        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0]['success'])


if __name__ == '__main__':
    unittest.main()

class TestCouplingRequestReachesTheBackend(unittest.TestCase):
    """A requested coupling is delivered to the backend that reads one."""

    class _Stub:
        supports_external_potential = False

        def __init__(self, supports_nac, supports_soc):
            self.supports_nac_request = supports_nac
            self.supports_soc_request = supports_soc
            self.seen = None

        def calculate(self, atoms, coordinates, **kwargs):
            self.seen = kwargs
            return {'energy': 0.0}

    def _kwargs(self, supports_nac=True, supports_soc=True):
        from pyrai3md.workflow.dispatch.terms import _run_one, make_work_item
        backend = self._Stub(supports_nac, supports_soc)
        item = make_work_item(task_id='qm', backend=backend,
                              local_atoms=['O'],
                              local_coords=[[0.0, 0.0, 0.0]],
                              target_states=[0, 1], nac_pairs=[[0, 1]],
                              soc_pairs=[[0, 2]])
        outcome = _run_one(item)
        self.assertTrue(outcome['success'], outcome['error'])
        return backend.seen

    def test_a_backend_that_reads_pairs_is_given_them(self):
        # The pair used to arrive in the work item and stop there: no skill
        # `calculate()` took it, so `&ALASKA` never named one and the engine
        # printed no coupling for a search that asked for one.
        self.assertEqual(self._kwargs().get('nac_pairs'), [[0, 1]])

    def test_the_soc_channel_reaches_the_backend_that_addresses_rows_by_it(self):
        # The same dead channel on the spin-orbit side: the reader addresses
        # its rows by this list, so a subset has to travel or the step gets
        # every pair the config declares under the row numbers it expects.
        self.assertEqual(self._kwargs().get('soc_pairs'), [[0, 2]])

    def test_a_backend_that_does_not_is_not_sent_them(self):
        # A dynamics run's pair map is ordinary furniture for a skill that
        # names its couplings in its own input, and the alternative - raising
        # - would fail every FSSH case over a keyword it never needed.
        kwargs = self._kwargs(supports_nac=False, supports_soc=False)
        self.assertNotIn('nac_pairs', kwargs)
        self.assertNotIn('soc_pairs', kwargs)

    def test_the_two_channels_are_gated_apart(self):
        # One flag must not open the other channel: a backend that reads NAC
        # pairs but not SOC would be handed a subset it ignores, and the
        # reader would then address the wrong rows.
        only_nac = self._kwargs(supports_nac=True, supports_soc=False)
        self.assertIn('nac_pairs', only_nac)
        self.assertNotIn('soc_pairs', only_nac)
        only_soc = self._kwargs(supports_nac=False, supports_soc=True)
        self.assertIn('soc_pairs', only_soc)
        self.assertNotIn('nac_pairs', only_soc)
