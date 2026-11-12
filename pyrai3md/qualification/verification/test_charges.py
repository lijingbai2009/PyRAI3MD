"""
======================================================================
Tests for ONIOM charge management and electrostatic embedding.

Covers the charge store (data.charges), the charge-update routing
(workflow.multiscale.charges), the two-phase embedded execution and point-charge
sync (workflow.calculator), the static read_charge initialization, the
MD charge files (workflow.records.output), the &composite energy breakdown
(workflow.multiscale.composite + utilities.print_results) and the embedding schema
validation (schema.validate).

Sep 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.data.charges import (
    ChargeStore, ChargeStoreError, embedding_region_by_center,
    managed_region_indices)
from pyrai3md.workflow.multiscale.charges import (
    capture_outer_reference, embedding_for_task, region_charge_sources,
    update_from_outcomes)


# ---------------------------------------------------------------------------
# data.charges.ChargeStore
# ---------------------------------------------------------------------------

class TestChargeStore(unittest.TestCase):
    """Per-atom charge store: load/update/derive/serialize."""

    def test_init_zeros_and_managed_sorted_unique(self):
        store = ChargeStore(5, managed_indices=[3, 1, 1, 2])
        self.assertEqual(store.natom, 5)
        self.assertEqual(store.atom_charges.tolist(), [0.0] * 5)
        self.assertEqual(store.managed_indices.tolist(), [1, 2, 3])

    def test_update_writes_given_indices(self):
        store = ChargeStore(4, managed_indices=[0, 1, 2])
        store.update([0, 2], [-0.5, 0.25])
        self.assertEqual(store.atom_charges.tolist(),
                         [-0.5, 0.0, 0.25, 0.0])

    def test_update_length_mismatch_raises(self):
        store = ChargeStore(3, managed_indices=[0, 1])
        with self.assertRaises(ChargeStoreError):
            store.update([0, 1], [0.1])

    def test_update_out_of_range_raises(self):
        store = ChargeStore(3, managed_indices=[0, 1])
        with self.assertRaises(ChargeStoreError):
            store.update([0, 5], [0.1, 0.2])

    def test_embedding_charges_pairs_q_with_coords(self):
        store = ChargeStore(4, managed_indices=[1, 2])
        store.update([1, 2], [0.3, 0.4])
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                          dtype=float)
        emb = store.embedding_charges([1, 2], coords)
        np.testing.assert_allclose(
            emb, [[0.3, 1, 0, 0], [0.4, 2, 0, 0]])

    def test_point_charges_reports_whole_system(self):
        store = ChargeStore(4, managed_indices=[2, 0])
        store.update([0, 2], [-0.1, -0.2])
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                          dtype=float)
        pc = store.point_charges(coords)
        # Whole-system rows: row i is atom i, unmanaged atoms stay 0.
        np.testing.assert_allclose(
            pc, [[-0.1, 0, 0, 0], [0.0, 1, 1, 1],
                 [-0.2, 2, 2, 2], [0.0, 3, 3, 3]])

    def test_load_file_legacy_format(self):
        store = ChargeStore(5, managed_indices=[1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 't.charge')
            with open(path, 'w') as fh:
                fh.write('3\ncomment\n0.1 9 9 9\n0.2 8 8 8\n0.3 7 7 7\n')
            n = store.load_file(path, [1, 2, 3])
        self.assertEqual(n, 3)
        # Only q is consumed; file xyz ignored, mapped in row order.
        self.assertEqual(store.atom_charges.tolist(),
                         [0.0, 0.1, 0.2, 0.3, 0.0])

    def test_load_file_missing_raises_filenotfound(self):
        store = ChargeStore(3, managed_indices=[0])
        with self.assertRaises(FileNotFoundError):
            store.load_file('/no/such/file.charge', [0])

    def test_load_file_count_mismatch_raises(self):
        store = ChargeStore(5, managed_indices=[1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 't.charge')
            with open(path, 'w') as fh:
                fh.write('3\nc\n0.1 0 0 0\n0.2 0 0 0\n0.3 0 0 0\n')
            with self.assertRaises(ChargeStoreError):
                store.load_file(path, [1, 2])

    def test_dict_round_trip(self):
        store = ChargeStore(4, managed_indices=[0, 1, 2])
        store.update([0, 1, 2], [-0.5, 0.25, 0.25])
        restored = ChargeStore.from_dict(store.to_dict())
        self.assertEqual(restored.natom, 4)
        self.assertEqual(restored.managed_indices.tolist(), [0, 1, 2])
        np.testing.assert_allclose(restored.atom_charges,
                                   store.atom_charges)

    def test_from_dict_none_is_none(self):
        self.assertIsNone(ChargeStore.from_dict(None))


class TestEmbeddingRegions(unittest.TestCase):
    """embedding_region_by_center / managed_region_indices set logic."""

    def test_embedding_region_is_the_disjoint_midlevel(self):
        shells = embedding_region_by_center([[1, 2], [4, 5]])
        self.assertEqual(shells, {0: [1, 2], 1: [4, 5]})

    def test_managed_region_is_union_high_and_midlevel(self):
        managed = managed_region_indices([[0], [3]], [[1, 2], [4, 5]])
        self.assertEqual(managed, [0, 1, 2, 3, 4, 5])

    def test_embedding_region_empty_stays_empty(self):
        self.assertEqual(embedding_region_by_center([[]]), {0: []})


# ---------------------------------------------------------------------------
# workflow.multiscale.charges routing
# ---------------------------------------------------------------------------

class _Outcome:
    """Minimal routed-outcome double carrying a raw payload."""

    def __init__(self, raw, success=True):
        self.raw = raw
        self.success = success


def _oniom3_c1_tasks():
    """Real-first ONIOM3 C=1 queue: H=[0], I=[0,1,2]."""
    return [
        {'task_id': 'qm3', 'provides_electronic': False, 'layer_id': 2,
         'region': 'real', 'region_indices': [], 'center_index': None},
        {'task_id': 'qm2', 'provides_electronic': False, 'layer_id': 1,
         'region': 'inter', 'region_indices': [0, 1, 2], 'center_index': 0},
        {'task_id': 'qm3_task1', 'provides_electronic': False, 'layer_id': 2,
         'region': 'inter', 'region_indices': [0, 1, 2], 'center_index': 0},
        {'task_id': 'qm', 'provides_electronic': True, 'layer_id': 0,
         'region': 'model', 'region_indices': [0], 'center_index': 0},
        {'task_id': 'qm2_task1', 'provides_electronic': False, 'layer_id': 1,
         'region': 'model', 'region_indices': [0], 'center_index': 0},
    ]


class TestChargeRouting(unittest.TestCase):
    """region_charge_sources / update_from_outcomes / embedding_for_task."""

    def test_region_charge_sources_split(self):
        tasks = _oniom3_c1_tasks()
        high, mid_inter = region_charge_sources(tasks)
        self.assertEqual([t['task_id'] for t in high], ['qm'])
        self.assertEqual([t['task_id'] for t in mid_inter], ['qm2'])

    def test_highlevel_from_high_midshell_from_inter(self):
        store = ChargeStore(6, managed_indices=[0, 1, 2])
        tasks = _oniom3_c1_tasks()
        outcomes = [
            _Outcome({'energy': [1.0]}),                            # L(real)
            _Outcome({'computed_charges': np.array([[0.1, 0.2, 0.3]])}),  # +M(I)
            _Outcome({'computed_charges': np.array([[9, 9, 9]])}),  # -L(I)
            _Outcome({'computed_charges': np.array([[-0.9]])}),     # +H(M)
            _Outcome({'computed_charges': np.array([[5.0]])}),      # -M(M)
        ]
        n = update_from_outcomes(store, tasks, outcomes)
        self.assertEqual(n, 2)
        # highlevel idx0 <- high (-0.9); mid shell idx1,2 <- inter,
        # the H_c part (idx0's 0.1) ignored; -L(I)/-M(M) not sources.
        self.assertEqual(store.atom_charges.tolist(),
                         [-0.9, 0.2, 0.3, 0.0, 0.0, 0.0])

    def test_the_charge_row_follows_the_state_that_ran(self):
        """#450: a per-state payload is read at the task's own state.

        `computed_charges` is (nstate, nlocal); the ONIOM path used to take
        row 0 unconditionally, which embeds one surface's charges under
        another surface's name as soon as a high-level task runs an excited
        state. The QM/MM path already selects the row; this is the same rule
        on the charge scatter.
        """
        tasks = _oniom3_c1_tasks()
        outcomes = [
            _Outcome({'energy': [1.0]}),
            _Outcome({'computed_charges': np.array(  # +M(I), 3-atom region
                [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])}),
            _Outcome({'energy': [1.0]}),
            _Outcome({'computed_charges': np.array([[1.0], [2.0], [3.0]])}),
            _Outcome({'energy': [1.0]}),
        ]
        store = ChargeStore(3, managed_indices=[0, 1, 2])
        written = update_from_outcomes(
            store, tasks, outcomes,
            state=lambda task: 2 if task.get('provides_electronic') else 1)
        self.assertEqual(written, 2)
        # highlevel idx0 <- the provider's state-2 row (3.0), not 1.0;
        # mid shell idx1/2 <- the inter term's state-1 row (0.2 each),
        # with its highlevel part (0.2 at idx0) ignored by the mid write.
        self.assertEqual(store.atom_charges.tolist(), [3.0, 0.2, 0.2])

    def test_a_state_the_payload_does_not_carry_writes_no_row(self):
        """Out of range is a refusal, not row 0 (#450)."""
        tasks = [_oniom3_c1_tasks()[3]]  # the high provider
        store = ChargeStore(1, managed_indices=[0])
        with self.assertLogs('pyrai3md.charges', level='WARNING') as caught:
            written = update_from_outcomes(
                store, tasks,
                [_Outcome({'computed_charges': np.array([[1.0], [2.0]])})],
                state=5)
        self.assertEqual(written, 0)
        self.assertEqual(store.atom_charges.tolist(), [0.0])
        self.assertIn('state 5 was requested', caught.output[0])

    def test_missing_computed_charges_skipped(self):
        store = ChargeStore(3, managed_indices=[0, 1, 2])
        tasks = _oniom3_c1_tasks()
        outcomes = [_Outcome({'energy': [1.0]}) for _ in tasks]
        n = update_from_outcomes(store, tasks, outcomes)
        self.assertEqual(n, 0)
        self.assertEqual(store.atom_charges.tolist(), [0.0, 0.0, 0.0])

    def test_failed_outcome_skipped(self):
        store = ChargeStore(3, managed_indices=[0])
        tasks = [_oniom3_c1_tasks()[3]]  # the high provider
        outcomes = [_Outcome({}, success=False)]
        self.assertEqual(update_from_outcomes(store, tasks, outcomes), 0)

    def test_none_store_is_noop(self):
        self.assertEqual(
            update_from_outcomes(None, _oniom3_c1_tasks(), []), 0)

    def test_phase_local_queue_alone_loses_the_highlevel_rows(self):
        # The witness that the two-phase fix is not an identity. An embedded
        # run scatters its queue in two phases, and phase 1 holds no
        # provider; asked which rows the high level owns from that phase
        # alone, the answer is none, so the mid-level inter term writes its
        # own H_c part over the highlevel row. Read off the whole queue the
        # same term drops that part, and the row is left for theory[0].
        tasks = _oniom3_c1_tasks()
        outcomes = [
            _Outcome({'energy': [1.0]}),
            _Outcome({'computed_charges': np.array([[0.1, 0.2, 0.3]])}),
            _Outcome({'computed_charges': np.array([[9, 9, 9]])}),
            _Outcome({'computed_charges': np.array([[-0.9]])}),
            _Outcome({'computed_charges': np.array([[5.0]])}),
        ]
        phase1 = [(t, o) for t, o in zip(tasks, outcomes)
                  if not t.get('provides_electronic')]
        p_tasks = [t for t, _ in phase1]
        p_outs = [o for _, o in phase1]

        blind = ChargeStore(6, managed_indices=[0, 1, 2])
        update_from_outcomes(blind, p_tasks, p_outs)
        self.assertEqual(blind.atom_charges.tolist(),
                         [0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

        informed = ChargeStore(6, managed_indices=[0, 1, 2])
        update_from_outcomes(informed, p_tasks, p_outs,
                             sources=region_charge_sources(tasks))
        self.assertEqual(informed.atom_charges.tolist(),
                         [0.0, 0.2, 0.3, 0.0, 0.0, 0.0])

    def test_two_phase_order_leaves_the_provider_owning_its_rows(self):
        # What the calculator does: phase 1 with the partition read off the
        # whole queue, then phase 2 for the provider. The highlevel row ends
        # as theory[0] said whatever theory[1] reported for the same atom,
        # which is the property the printed `&highlevel point charges` block
        # claims. A provider that emits no charges (bagel, openqp) leaves
        # that row at zero instead of at the mid level's guess.
        tasks = _oniom3_c1_tasks()
        outcomes = [
            _Outcome({'energy': [1.0]}),
            _Outcome({'computed_charges': np.array([[0.1, 0.2, 0.3]])}),
            _Outcome({'computed_charges': np.array([[9, 9, 9]])}),
            _Outcome({'computed_charges': np.array([[-0.9]])}),
            _Outcome({'computed_charges': np.array([[5.0]])}),
        ]
        store = ChargeStore(6, managed_indices=[0, 1, 2])
        sources = region_charge_sources(tasks)
        for keep in (False, True):
            pairs = [(t, o) for t, o in zip(tasks, outcomes)
                     if bool(t.get('provides_electronic')) is keep]
            update_from_outcomes(store, [t for t, _ in pairs],
                                 [o for _, o in pairs], sources=sources)
        self.assertEqual(store.atom_charges.tolist(),
                         [-0.9, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.assertTrue(store.has_values)

    def test_oniom2_outer_region_excludes_highlevel_across_phases(self):
        # The two-layer queue says the same thing the other way: theory[1]
        # runs over the whole system and its charges become the embedding,
        # minus the region theory[0] owns. Read off phase 1 alone there is no
        # provider to subtract, so the outer update claims the high-level atom
        # as well - and for a provider that returns no computed_charges at all
        # that is what stays in the store, and what the run prints as
        # `&highlevel point charges`.
        real = {'task_id': 'qm2', 'provides_electronic': False, 'layer_id': 1,
                'region': 'real', 'region_indices': [], 'center_index': None}
        high = {'task_id': 'qm', 'provides_electronic': True, 'layer_id': 0,
                'region': 'model', 'region_indices': [0], 'center_index': 0}
        low = {'task_id': 'qm2_model', 'provides_electronic': False,
               'layer_id': 1, 'region': 'model', 'region_indices': [0],
               'center_index': 0}
        tasks = [real, high, low]
        outcomes = [_Outcome({'computed_charges': np.array([[-0.8, 0.4, 0.4]])}),
                    _Outcome({'computed_charges': np.array([[-0.9]])}),
                    _Outcome({'computed_charges': np.array([[9.0]])})]

        blind = ChargeStore(3, managed_indices=[0, 1, 2])
        update_from_outcomes(blind, [real], [outcomes[0]])
        self.assertEqual(blind.atom_charges.tolist(), [-0.8, 0.4, 0.4])

        informed = ChargeStore(3, managed_indices=[0, 1, 2])
        update_from_outcomes(informed, [real], [outcomes[0]],
                             sources=region_charge_sources(tasks))
        self.assertEqual(informed.atom_charges.tolist(),
                         [0.0, 0.4, 0.4])

    def test_embedding_only_for_high_provider(self):
        store = ChargeStore(6, managed_indices=[0, 1, 2])
        store.update([1, 2], [0.2, 0.3])
        tasks = _oniom3_c1_tasks()
        shells = {0: [1, 2]}
        coords = np.arange(18, dtype=float).reshape(6, 3)
        emb = embedding_for_task(store, tasks[3], coords, shells)
        np.testing.assert_allclose(emb[:, 0], [0.2, 0.3])
        # A non-provider embeds nothing.
        self.assertIsNone(embedding_for_task(store, tasks[1], coords, shells))

    def test_capture_outer_reference_reads_the_full_system_row(self):
        real = {'task_id': 'qm2', 'provides_electronic': False, 'layer_id': 1,
                'region': 'real', 'region_indices': [], 'center_index': None}
        values = capture_outer_reference(
            [real], [_Outcome(
                {'computed_charges': np.array([[1.0, 2.0, 3.0]])})])
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])

    def test_capture_outer_reference_warns_when_the_source_is_silent(self):
        real = {'task_id': 'qm2', 'provides_electronic': False, 'layer_id': 1,
                'region': 'real', 'region_indices': [], 'center_index': None}
        with self.assertLogs('pyrai3md.charges', level='WARNING'):
            self.assertIsNone(
                capture_outer_reference([real], [_Outcome({'energy': [1.0]})]))

    def test_embedding_per_center_uses_own_midshell(self):
        store = ChargeStore(6, managed_indices=[0, 1, 2, 3, 4, 5])
        store.update([1, 2, 4, 5], [0.1, 0.2, 0.4, 0.5])
        shells = {0: [1, 2], 1: [4, 5]}
        coords = np.arange(18, dtype=float).reshape(6, 3)
        high_c0 = {'provides_electronic': True, 'center_index': 0,
                   'task_id': 'qm'}
        high_c1 = {'provides_electronic': True, 'center_index': 1,
                   'task_id': 'qm_task1'}
        np.testing.assert_allclose(
            embedding_for_task(store, high_c0, coords, shells)[:, 0],
            [0.1, 0.2])
        np.testing.assert_allclose(
            embedding_for_task(store, high_c1, coords, shells)[:, 0],
            [0.4, 0.5])


# ---------------------------------------------------------------------------
# Two-phase embedded execution (workflow.calculator)
# ---------------------------------------------------------------------------

CALL_LOG = []


class _RecordingBackend:
    """Mock backend recording the embedding field and emitting charges."""

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.execution_id = execution_id
        self.received_point_charges = 'UNSET'

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        self.received_point_charges = point_charges
        CALL_LOG.append(self.execution_id)
        natom = len(atoms)
        return {
            'energy': np.array([-1.0 * natom]),
            'gradient': np.ones((1, natom, 3)) * 0.01,
            'computed_charges': np.array(
                [[0.1 * (k + 1) for k in range(natom)]]),
        }


def _build_oniom3_state(highlevel, midlevel, ci, spin, natom, embedding,
                        read_charge=False, title='pyrai3md'):
    """Build a runnable ONIOM3 state with recording mock backends."""
    from pyrai3md.data.center import build_center_plan
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.planning.task import resolve_task_queue
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.control.title = title
    config.system.atoms = ['C'] * natom
    config.system.coordinates = [[float(i), 0.0, 0.0] for i in range(natom)]
    config.system.natom = natom
    config.system.highlevel = highlevel
    config.system.midlevel = midlevel
    config.system.ci = ci
    config.system.spin = spin
    config.system.embedding = embedding
    config.system.read_charge = read_charge

    state = PyRAIState()
    state.pmd = config
    state.bind_partition_plan(build_center_plan(config))

    class _A:
        pass

    class _B:
        pass

    class _C:
        pass

    skills = {'theory': {
        'a_skill': {'role': 'qm', 'backend': _A,
                    'capabilities': ['energy', 'gradient', 'nac']},
        'b_skill': {'role': 'qm2', 'backend': _B,
                    'capabilities': ['energy', 'gradient']},
        'c_skill': {'role': 'qm3', 'backend': _C,
                    'capabilities': ['energy', 'gradient']},
    }, 'reference': {}}
    queue = resolve_task_queue(config, skills)
    for task in queue['theory']:
        task['api'] = _RecordingBackend
        task['config'] = {'results': {}}
        state.add_task('theory', task)
    return state


class TestTwoPhaseExecution(unittest.TestCase):
    """Phase order, embedding delivery and the charge-store update."""

    def setUp(self):
        CALL_LOG.clear()

    def test_phase_order_and_embedding_c1(self):
        from pyrai3md.data.charges import managed_region_indices
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True)
        state.charge_store = ChargeStore(
            3, managed_indices=managed_region_indices([[0]], [[1, 2]]))
        calc = Calculator(state=state).initialize()
        calc.run()
        # The high provider runs last (phase 2 after mid+low phase 1).
        self.assertTrue(CALL_LOG[-1].startswith('tmp_high_qm_'))
        for eid in CALL_LOG[:-1]:
            self.assertFalse(eid.startswith('tmp_high_qm_A'))
        # highlevel <- high, mid shell <- inter (H_c part ignored).
        self.assertEqual(state.charge_store.atom_charges.tolist(),
                         [0.1, 0.2, 0.30000000000000004])
        # The high task was embedded with the updated mid shell [1, 2].
        high_backend = calc.backends['theory'][
            [t.get('provides_electronic') for t in state.theory].index(True)]
        pc = np.asarray(high_backend.received_point_charges)
        np.testing.assert_allclose(pc[:, 0], [0.2, 0.3])
        np.testing.assert_allclose(pc[:, 1], [1.0, 2.0])

    def test_non_high_tasks_get_no_embedding(self):
        from pyrai3md.data.charges import managed_region_indices
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True)
        state.charge_store = ChargeStore(
            3, managed_indices=managed_region_indices([[0]], [[1, 2]]))
        calc = Calculator(state=state).initialize()
        calc.run()
        for task, backend in zip(state.theory, calc.backends['theory']):
            if not task.get('provides_electronic'):
                self.assertIsNone(backend.received_point_charges)

    def test_multicenter_high_embeds_own_center_only(self):
        from pyrai3md.data.charges import managed_region_indices
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([[0], [3]], [[1, 2], [4, 5]],
                                    [[1], [1]], [[0], [0]], 6,
                                    embedding=True)
        state.charge_store = ChargeStore(
            6, managed_indices=managed_region_indices(
                [[0], [3]], [[1, 2], [4, 5]]))
        calc = Calculator(state=state).initialize()
        calc.run()
        for task, backend in zip(state.theory, calc.backends['theory']):
            if not task.get('provides_electronic'):
                continue
            pc = np.asarray(backend.received_point_charges)
            center = task.get('center_index')
            if center == 0:
                # mid shell of center 0 = atoms [1, 2] at x = 1, 2
                np.testing.assert_allclose(pc[:, 1], [1.0, 2.0])
            else:
                # mid shell of center 1 = atoms [4, 5] at x = 4, 5
                np.testing.assert_allclose(pc[:, 1], [4.0, 5.0])

    def test_point_charges_synced_into_system(self):
        from pyrai3md.data.charges import managed_region_indices
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True)
        state.charge_store = ChargeStore(
            3, managed_indices=managed_region_indices([[0]], [[1, 2]]))
        Calculator(state=state).initialize().run()
        pc = state.pmd.system.point_charges
        self.assertEqual(len(pc), 3)
        np.testing.assert_allclose([row[0] for row in pc],
                                   [0.1, 0.2, 0.30000000000000004])
        # The state property derives the same managed-region rows.
        np.testing.assert_allclose(
            np.asarray(state.point_charges)[:, 0],
            [0.1, 0.2, 0.30000000000000004])

    def test_no_embedding_leaves_point_charges_untouched(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=False)
        self.assertIsNone(state.charge_store)
        state.pmd.system.point_charges = []
        Calculator(state=state).initialize().run()
        self.assertEqual(state.pmd.system.point_charges, [])


class TestReadCharge(unittest.TestCase):
    """Static read_charge initialization (loaded once, never updated)."""

    def test_init_charge_store_loads_midshell(self):
        from pyrai3md.data.build import _init_charge_store
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True, read_charge=True,
                                    title='mol')
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with open('mol.charge', 'w') as fh:
                    # mid shell = I \ H = [1, 2] -> two rows
                    fh.write('2\nstatic field\n0.7 1 0 0\n0.8 2 0 0\n')
                _init_charge_store(state, state.pmd)
            finally:
                os.chdir(cwd)
        self.assertIsNotNone(state.charge_store)
        # mid shell [1, 2] loaded; highlevel idx0 stays 0 (no update).
        self.assertEqual(state.charge_store.atom_charges.tolist(),
                         [0.0, 0.7, 0.8])

    def test_read_charge_missing_file_raises(self):
        from pyrai3md.data.build import _init_charge_store
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True, read_charge=True,
                                    title='absent')
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaises(FileNotFoundError):
                    _init_charge_store(state, state.pmd)
            finally:
                os.chdir(cwd)

    def test_read_charge_runs_single_phase_no_update(self):
        from pyrai3md.data.charges import managed_region_indices
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True, read_charge=True)
        state.charge_store = ChargeStore(
            3, managed_indices=managed_region_indices([[0]], [[1, 2]]))
        state.charge_store.update([1, 2], [0.7, 0.8])
        CALL_LOG.clear()
        calc = Calculator(state=state).initialize()
        calc.run()
        # Single batch: every task staged once, no phase split.
        self.assertEqual(len(CALL_LOG), len(state.theory))
        # Static field is never overwritten by the computed charges.
        self.assertEqual(state.charge_store.atom_charges.tolist(),
                         [0.0, 0.7, 0.8])
        # The high provider is still embedded, with the static charges.
        high_backend = calc.backends['theory'][
            [t.get('provides_electronic') for t in state.theory].index(True)]
        pc = np.asarray(high_backend.received_point_charges)
        np.testing.assert_allclose(pc[:, 0], [0.7, 0.8])


# ---------------------------------------------------------------------------
# &composite energy breakdown (workflow.multiscale.composite + print_results)
# ---------------------------------------------------------------------------

class TestCompositeDecomposition(unittest.TestCase):
    """composite_terms / composite_total collection and rendering."""

    def test_terms_collected_from_two_phase_run(self):
        from pyrai3md.data.charges import managed_region_indices
        from pyrai3md.workflow.calculator import Calculator
        state = _build_oniom3_state([0], [1, 2], [1], [0], 3,
                                    embedding=True)
        state.charge_store = ChargeStore(
            3, managed_indices=managed_region_indices([[0]], [[1, 2]]))
        Calculator(state=state).initialize().run()
        terms = state.pmd.results['composite_terms']
        self.assertEqual(
            terms['labels'],
            ['high[qm]', 'high[qm2]', 'mid[qm2]', 'mid[qm3]', 'low[qm3]'])
        self.assertEqual(terms['contributions'], [1, -1, 1, -1, 1])
        # Raw per-term energies: model terms span 1 atom, inter/real 3.
        self.assertEqual([row[0] for row in terms['energies']],
                         [-1.0, -1.0, -3.0, -3.0, -3.0])
        np.testing.assert_allclose(
            state.pmd.results['composite_total'], [-3.0])
        np.testing.assert_allclose(state.pmd.results['energy'], [-3.0])

    def test_build_composite_terms_orders_high_mid_low(self):
        from pyrai3md.workflow.multiscale.composite import build_composite_terms
        collected = [
            ({'region': 'real', 'role': 'qm3', 'center_index': None,
              'task_slot': 0}, [-3.0], 1),
            ({'region': 'inter', 'role': 'qm2', 'center_index': 0,
              'task_slot': 1}, [-3.0], 1),
            ({'region': 'model', 'role': 'qm', 'center_index': 0,
              'task_slot': 3}, [-1.0, -0.5], 1),
            ({'region': 'model', 'role': 'qm2', 'center_index': 0,
              'task_slot': 4}, [-1.0], -1),
        ]
        terms = build_composite_terms(collected, multi_center=False)
        self.assertEqual(terms['labels'],
                         ['high[qm]', 'high[qm2]', 'mid[qm2]', 'low[qm3]'])
        # The multi-state provider keeps both states on its row.
        self.assertEqual(terms['energies'][0], [-1.0, -0.5])

    def test_the_recorded_state_index_travels_with_its_own_term(self):
        # The consumer reads one number per term off the disk table, so the
        # state a term contributes at must arrive in the ORDER the labels are
        # in. Taken from the queue instead it would pair a term with another
        # term's state, which is a wrong column rather than a crash.
        from pyrai3md.workflow.multiscale.composite import build_composite_terms
        collected = [
            ({'region': 'real', 'role': 'qm3', 'center_index': None,
              'task_slot': 0}, [-3.0, -3.2, -3.4], 1),
            ({'region': 'inter', 'role': 'qm2', 'center_index': 0,
              'task_slot': 1, 'reference_state': 2}, [-3.0, -3.1, -3.2], 1),
            ({'region': 'model', 'role': 'qm', 'center_index': 0,
              'task_slot': 3, 'provides_electronic': True,
              'state_policy': 'active'}, [-1.0, -0.5, -0.2], 1),
            ({'region': 'model', 'role': 'qm2', 'center_index': 0,
              'task_slot': 4}, [-1.0, -0.9], -1),
        ]
        terms = build_composite_terms(collected, multi_center=True,
                                      active_configuration=(1,))
        self.assertEqual(terms['labels'],
                         ['high[qm][1]', 'high[qm2][1]', 'mid[qm2][1]',
                          'low[qm3]'])
        # The active provider follows the center's local state (1); the
        # reference terms keep their declared ones; the list is in the sorted
        # order the labels are in, not the queue's.
        self.assertEqual(list(terms['state_indices']), [1, 0, 2, 0])

    def test_a_single_center_assembly_records_no_state_of_its_own(self):
        # C=1 assembles one total over a global state axis, so no term has a
        # local state: the record says None for every term and the reader
        # falls back to the state the run is driven on.
        from pyrai3md.workflow.multiscale.composite import build_composite_terms
        collected = [
            ({'region': 'model', 'role': 'qm', 'center_index': 0,
              'task_slot': 3}, [-1.0, -0.5], 1),
            ({'region': 'real', 'role': 'qm3', 'center_index': None,
              'task_slot': 0}, [-3.0], 1),
        ]
        terms = build_composite_terms(collected, multi_center=False)
        self.assertEqual(len(terms['state_indices']), len(terms['labels']))
        self.assertEqual(list(terms['state_indices']), [None, None])

    def test_term_label_center_suffix(self):
        from pyrai3md.workflow.multiscale.composite import term_label
        task = {'region': 'model', 'role': 'qm', 'center_index': 1}
        self.assertEqual(term_label(task, multi_center=False), 'high[qm]')
        self.assertEqual(term_label(task, multi_center=True), 'high[qm][2]')
        real = {'region': 'real', 'role': 'qm3', 'center_index': None}
        self.assertEqual(term_label(real, multi_center=True), 'low[qm3]')

    def test_format_composite_renders_section(self):
        from pyrai3md.utilities.print_results import _format_composite
        terms = {'labels': ['high[qm]', 'low[qm3]'],
                 'energies': [[-1.0, -0.5], [-3.0]],
                 'contributions': [1, 1]}
        block = _format_composite(terms, [-4.0, -3.5])
        self.assertIn('&composite', block)
        self.assertIn('high[qm]', block)
        self.assertIn('total', block)
        lines = [ln for ln in block.splitlines() if 'high[qm]' in ln]
        # The multi-state term prints both states on one line.
        self.assertEqual(len(lines), 1)
        self.assertIn('-1.0000000000', lines[0])
        self.assertIn('-0.5000000000', lines[0])

    def test_the_state_record_does_not_move_the_rendered_block(self):
        # The log block prints the raw per-state rows and reads only labels and
        # energies, so the term columns' state record must not change one byte
        # of it: two renderings, the same text.
        from pyrai3md.utilities.print_results import _format_composite
        terms = {'labels': ['high[qm]', 'low[qm3]'],
                 'energies': [[-1.0, -0.5], [-3.0]],
                 'contributions': [1, 1]}
        with_record = dict(terms, state_indices=[1, None])
        self.assertEqual(_format_composite(with_record, [-4.0, -3.5]),
                         _format_composite(terms, [-4.0, -3.5]))

    def test_format_composite_omitted_without_terms(self):
        from pyrai3md.utilities.print_results import _format_composite
        self.assertEqual(_format_composite(None, None), '')
        self.assertEqual(_format_composite({'labels': [], 'energies': []},
                                           []), '')


# ---------------------------------------------------------------------------
# MD charge files (workflow.records.output)
# ---------------------------------------------------------------------------

def _charge_snapshot():
    return {
        'system': {
            'atoms': ['C', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0]],
            'velocities': [[0.0, 0.0, 0.0]] * 3,
            'mass': [12.0, 1.0, 1.0],
            'current_state': 0,
            'point_charges': [[-0.5, 0.0, 0.0, 0.0],
                              [0.25, 1.0, 0.0, 0.0],
                              [0.25, 2.0, 0.0, 0.0]],
        },
        'results': {'energy': [-10.0]},
    }


class TestMdChargeFiles(unittest.TestCase):
    """md.charges/sh.charges emission gated on charge management."""

    def test_no_charge_suffixes_when_disabled(self):
        from pyrai3md.workflow.records.output import MdOutputWriter
        with tempfile.TemporaryDirectory() as tmp:
            writer = MdOutputWriter(title='t', log_dir=tmp, total_steps=1)
            self.assertEqual(
                [s for s in writer._file_suffixes() if 'charges' in s], [])

    def test_global_md_charges_frame(self):
        from pyrai3md.workflow.records.output import MdOutputWriter
        with tempfile.TemporaryDirectory() as tmp:
            writer = MdOutputWriter(title='t', log_dir=tmp, total_steps=1,
                                    charge_output=True)
            self.assertIn('md.charges', writer._file_suffixes())
            writer.write_headers()
            writer.record(1, _charge_snapshot(), None, final=True)
            path = os.path.join(tmp, 't.md.charges')
            self.assertTrue(os.path.exists(path))
            lines = open(path).read().splitlines()
            # Legacy frame: ncharge, comment, then q x y z rows.
            self.assertEqual(lines[0], '3')
            self.assertIn('coord 1', lines[1])
            self.assertEqual(len(lines), 2 + 3)
            # No hop -> no sh.charges frame written.
            self.assertFalse(
                os.path.exists(os.path.join(tmp, 't.sh.charges')))

    def test_per_center_charge_slicing(self):
        from pyrai3md.workflow.records.output import MdOutputWriter
        with tempfile.TemporaryDirectory() as tmp:
            writer = MdOutputWriter(title='p', log_dir=tmp, total_steps=1,
                                    center_regions=[[0], [1, 2]],
                                    charge_output=True)
            self.assertIn('md.1.charges', writer._file_suffixes())
            writer.write_headers()
            writer.record(1, _charge_snapshot(), None, final=True)
            c1 = open(os.path.join(tmp, 'p.md.1.charges')).read().splitlines()
            c2 = open(os.path.join(tmp, 'p.md.2.charges')).read().splitlines()
            self.assertEqual(c1[0], '1')  # center 1 region [0] -> 1 charge
            self.assertEqual(c2[0], '2')  # center 2 region [1,2] -> 2


# ---------------------------------------------------------------------------
# Schema validation (schema.validate)
# ---------------------------------------------------------------------------

class TestEmbeddingValidation(unittest.TestCase):
    """embedding / read_charge keyword consistency."""

    def test_embedding_requires_midlevel(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertRaises(ValueError):
            validate_system({'atoms': ['C'], 'natom': 1,
                             'theory': ['a', 'b', 'c'],
                             'embedding': True, 'midlevel': []})

    def test_single_center_embedding_has_no_midlevel_requirement(self):
        # A lone QM calculation with background charges is not an
        # ONIOM job: the default empty midlevel list must not be read
        # as a missing ONIOM mid shell.
        from pyrai3md.schema.validate import validate_system
        result = validate_system({'atoms': ['C'], 'natom': 1,
                                  'embedding': True, 'midlevel': []})
        self.assertTrue(result['embedding'])

    def test_read_charge_without_embedding_is_ignored(self):
        # embedding=false means no embedding; read_charge has no effect
        # and is ignored rather than refused.
        from pyrai3md.schema.validate import validate_system
        result = validate_system({'atoms': ['C'], 'natom': 1,
                                  'read_charge': True, 'midlevel': [0]})
        self.assertFalse(result['read_charge'])

    def test_valid_combinations_pass(self):
        from pyrai3md.schema.validate import validate_system
        base = {'atoms': ['C', 'H'], 'natom': 2}
        self.assertIsNotNone(validate_system(
            dict(base, embedding=True, midlevel=[0, 1])))
        self.assertIsNotNone(validate_system(
            dict(base, embedding=True, read_charge=True, midlevel=[0, 1])))
        self.assertIsNotNone(validate_system(dict(base)))

    def test_standalone_read_charge_is_the_embedding_switch(self):
        # One theory and no partition: `embedding` is a multiscale switch with
        # nothing to steer, so it is not read at all - the run is embedded
        # exactly when read_charge names '{title}.charge'. Writing
        # embedding=false next to it therefore neither refuses nor clears it.
        from pyrai3md.schema.validate import validate_system
        alone = validate_system({'atoms': ['C'], 'natom': 1,
                                 'theory': ['molcas'],
                                 'read_charge': True})
        self.assertTrue(alone['read_charge'])
        self.assertFalse(alone['embedding'])
        with_false = validate_system({'atoms': ['C'], 'natom': 1,
                                      'theory': ['molcas'],
                                      'embedding': False,
                                      'read_charge': True})
        self.assertTrue(with_false['read_charge'])

    def test_two_background_sources_are_refused(self):
        # read_charge (which resolves '{title}.charge') and external_charges (a
        # path) answer the same question; one of them has to say it.
        from pyrai3md.schema.validate import validate_system
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bg.charge')
            with open(path, 'w') as fh:
                fh.write('1\nfield\n0.5 0 0 5\n')
            with self.assertRaises(ValueError):
                validate_system({'atoms': ['C'], 'natom': 1,
                                 'theory': ['molcas'],
                                 'read_charge': True,
                                 'external_charges': path})

    def test_standalone_predicate_is_shared_by_validator_and_queue(self):
        # The line between the two read_charge regimes is decided once; the
        # validator and the task queue cannot disagree about it.
        from pyrai3md.schema.validate import is_standalone_queue
        self.assertTrue(is_standalone_queue(['molcas'], []))
        self.assertTrue(is_standalone_queue(['molcas'], ['lammps']))
        self.assertFalse(is_standalone_queue(['molcas'], ['a', 'b']))
        self.assertFalse(is_standalone_queue(['pyscf', 'lammps'], []))
        self.assertFalse(is_standalone_queue(['molcas'], [],
                                             highlevel=[0]))
        self.assertFalse(is_standalone_queue(['molcas'], [],
                                             midlevel=[0]))


class TestChargeArrayValidation(unittest.TestCase):
    """Whole-system point_charges and the external_charges channel."""

    def test_point_charges_must_be_whole_system(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertRaises(ValueError):
            validate_system({'atoms': ['C', 'C'], 'natom': 2,
                             'point_charges': [[0.1, 0.0, 0.0, 0.0]]})
        out = validate_system({
            'atoms': ['C', 'C'], 'natom': 2,
            'point_charges': [[0.1, 0.0, 0.0, 0.0],
                              [0.2, 1.0, 0.0, 0.0]]})
        self.assertEqual(len(out['point_charges']), 2)

    def test_external_charges_cannot_mix_with_a_partition(self):
        from pyrai3md.schema.validate import validate_system
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ext.charge')
            with open(path, 'w') as fh:
                fh.write('1\nc\n0.5 5 5 5\n')
            with self.assertRaises(ValueError):
                validate_system({'atoms': ['C'], 'natom': 1,
                                 'highlevel': [0],
                                 'external_charges': path})

    def test_external_charges_reject_periodic_positions(self):
        from pyrai3md.schema.validate import validate_system
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ext.charge')
            with open(path, 'w') as fh:
                fh.write('1\nc\n0.5 5 5 5\n')
            with self.assertRaises(ValueError):
                validate_system({'atoms': ['C'], 'natom': 1,
                                 'pbc': [1, 1, 1],
                                 'external_charges': path})

    def test_external_charges_file_is_parsed(self):
        from pyrai3md.schema.validate import validate_system
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ext.charge')
            with open(path, 'w') as fh:
                fh.write('2\nc\n0.5 5 5 5\n-0.5 6 5 5\n')
            out = validate_system({'atoms': ['C'], 'natom': 1,
                                   'external_charges': path})
        self.assertEqual(out['external_charges'], path)

    def test_skill_vocabulary_folds_onto_the_framework_switch(self):
        # A sectioned config hands `System` the skill's word for the same
        # decision, because the flatten step cannot tell the sections apart.
        # 'charge'/'potential' embed, 'none' does not, and a file name is
        # refused instead of reading as false.
        from pyrai3md.schema.config import System
        self.assertTrue(System(embedding='charge').embedding)
        self.assertTrue(System(embedding='field').embedding)
        self.assertFalse(System(embedding='none').embedding)
        self.assertTrue(System(read_charge='legacy').read_charge)
        with self.assertRaises(Exception):
            System(read_charge='espf_charge.dat')

    def test_oniom_read_charge_still_needs_its_embedding(self):
        # The switch keeps its old meaning under a partition: read_charge fills
        # the mid shell, and only when embedding was asked for. The standalone
        # reading must not leak back into this branch.
        from pyrai3md.schema.validate import validate_system
        kept = validate_system({'atoms': ['C'], 'natom': 1,
                                'theory': ['a', 'b', 'c'],
                                'embedding': True, 'midlevel': [0],
                                'read_charge': True})
        self.assertTrue(kept['read_charge'])
        cleared = validate_system({'atoms': ['C'], 'natom': 1,
                                   'theory': ['a', 'b', 'c'],
                                   'embedding': False, 'midlevel': [0],
                                   'read_charge': True})
        self.assertFalse(cleared['read_charge'])


def _build_oniom2_state(highlevel, natom, embedding,
                        read_charge=False, title='pyrai3md'):
    """Build a runnable single-center ONIOM2 state (QM outer layer)."""
    from pyrai3md.data.center import build_center_plan
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.planning.task import resolve_task_queue
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.control.title = title
    config.control.multiscale_type = 'oniom'
    config.system.atoms = ['C'] * natom
    config.system.coordinates = [[float(i), 0.0, 0.0] for i in range(natom)]
    config.system.natom = natom
    config.system.highlevel = highlevel
    config.system.ci = [1]
    config.system.spin = [0]
    config.system.embedding = embedding
    config.system.read_charge = read_charge

    state = PyRAIState()
    state.pmd = config
    state.bind_partition_plan(build_center_plan(config))

    class _A:
        pass

    class _B:
        pass

    skills = {'theory': {
        'a_skill': {'role': 'qm', 'backend': _A,
                    'capabilities': ['energy', 'gradient']},
        'b_skill': {'role': 'qm2', 'backend': _B,
                    'capabilities': ['energy', 'gradient']},
    }, 'reference': {}}
    queue = resolve_task_queue(config, skills)
    for task in queue['theory']:
        task['api'] = _RecordingBackend
        task['config'] = {'results': {}}
        state.add_task('theory', task)
    return state


class _DistinctChargeBackend:
    """Mock backend whose high and low charges cannot be confused."""

    def __init__(self, config=None, execution_id=''):
        self.execution_id = execution_id
        self.received_point_charges = 'UNSET'

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        self.received_point_charges = point_charges
        CALL_LOG.append(self.execution_id)
        natom = len(atoms)
        base = -9.0 if self.execution_id.startswith('tmp_high') else 100.0
        return {
            'energy': np.array([-1.0 * natom]),
            'gradient': np.ones((1, natom, 3)) * 0.01,
            'computed_charges': np.array(
                [[base + k for k in range(natom)]]),
        }


def _run_oniom2_state(highlevel, natom, embedding, backend_cls):
    """Run one ONIOM2 evaluation and return the state and its high backends."""
    from pyrai3md.workflow.calculator import Calculator
    CALL_LOG.clear()
    state = _build_oniom2_state(highlevel, natom, embedding=embedding)
    state.charge_store = ChargeStore(
        natom, managed_indices=list(range(natom)))
    for task in state.theory:
        task['api'] = backend_cls
    calc = Calculator(state=state).initialize()
    calc.run()
    highs = [backend for backend, task
             in zip(calc.backends['theory'], state.theory)
             if task.get('provides_electronic')]
    return state, highs


class TestOniom2OuterEmbedding(unittest.TestCase):
    """Single-center ONIOM2 with a quantum outer layer embeds in its
    outer-region charges (computed by theory[1] or read from .charge)."""

    def test_validate_allows_qm_outer_embedding(self):
        from pyrai3md.schema.validate import validate_system
        out = validate_system({
            'atoms': ['C', 'C'], 'natom': 2,
            'theory': ['openqp', 'xtb'], 'highlevel': [0],
            'embedding': True})
        self.assertTrue(out['embedding'])

    def test_validate_allows_multi_center_qm_outer_embedding(self):
        from pyrai3md.schema.validate import validate_system
        out = validate_system({
            'atoms': ['C'] * 3, 'natom': 3,
            'theory': ['a', 'b'], 'highlevel': [[0], [1, 2]],
            'embedding': True})
        self.assertTrue(out['embedding'])

    def test_validate_refuses_multi_center_read_charge(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertRaises(ValueError) as ctx:
            validate_system({
                'atoms': ['C'] * 3, 'natom': 3,
                'theory': ['a', 'b'], 'highlevel': [[0], [1, 2]],
                'embedding': True, 'read_charge': True})
        self.assertIn('static charge file', str(ctx.exception))

    def test_validate_refuses_multi_center_classical_outer(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertRaises(ValueError) as ctx:
            validate_system({
                'atoms': ['C'] * 3, 'natom': 3,
                'theory': ['a', 'lammps'], 'highlevel': [[0], [1, 2]],
                'embedding': True})
        self.assertIn('qmmm', str(ctx.exception))

    def test_multi_center_environments_are_the_other_centers(self):
        state, highs = _run_oniom2_state(
            [[0], [1, 2]], 3, True, _DistinctChargeBackend)
        self.assertEqual(len(highs), 2)
        # L(real) row is [100, 101, 102]; center 0's environment is atoms
        # 1,2 (the other center's high atoms), center 1's is atom 0. Both
        # carry the low-level row, not the other center's high charges.
        np.testing.assert_allclose(
            np.asarray(highs[0].received_point_charges)[:, 0],
            [101.0, 102.0])
        np.testing.assert_allclose(
            np.asarray(highs[1].received_point_charges)[:, 0], [100.0])

    def test_multi_center_store_high_rows_stay_provider_owned(self):
        state, _ = _run_oniom2_state(
            [[0], [1, 2]], 3, True, _DistinctChargeBackend)
        # Every atom is high-level here, so the store holds the providers'
        # rows (-9, then -9/-8), never the outer 100/101/102 row (#450).
        self.assertEqual(state.charge_store.atom_charges.tolist(),
                         [-9.0, -9.0, -8.0])

    def test_multi_center_result_keeps_the_scheme_and_outer_rows(self):
        state, _ = _run_oniom2_state(
            [[0], [1, 2]], 3, True, _DistinctChargeBackend)
        self.assertEqual(state.pmd.results.get('composite_scheme'), 'oniom')
        rows = state.pmd.results.get('oniom_outer_charges')
        self.assertEqual([row[0] for row in rows], [100.0, 101.0, 102.0])

    def test_phase_order_and_outer_charges(self):
        from pyrai3md.data.charges import ChargeStore
        from pyrai3md.workflow.calculator import Calculator
        CALL_LOG.clear()
        state = _build_oniom2_state([0], 3, embedding=True)
        state.charge_store = ChargeStore(3, managed_indices=[0, 1, 2])
        calc = Calculator(state=state).initialize()
        calc.run()
        # L(real) phase first, high provider last.
        self.assertTrue(CALL_LOG[-1].startswith('tmp_high_qm_'))
        self.assertEqual(state.charge_store.atom_charges.tolist(),
                         [0.1, 0.2, 0.30000000000000004])
        high_backend = calc.backends['theory'][
            [t.get('provides_electronic') for t in state.theory].index(True)]
        pc = np.asarray(high_backend.received_point_charges)
        np.testing.assert_allclose(pc[:, 0], [0.2, 0.3])
        np.testing.assert_allclose(pc[:, 1], [1.0, 2.0])

    def test_read_charge_fills_the_outer_region(self):
        from pyrai3md.data.build import _init_charge_store
        from pyrai3md.workflow.calculator import Calculator
        CALL_LOG.clear()
        state = _build_oniom2_state([0], 3, embedding=True,
                                    read_charge=True, title='mol2')
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with open('mol2.charge', 'w') as fh:
                    # outer = atoms 1, 2
                    fh.write('2\nstatic\n0.7 1 0 0\n0.8 2 0 0\n')
                _init_charge_store(state, state.pmd)
            finally:
                os.chdir(cwd)
        calc = Calculator(state=state).initialize()
        calc.run()
        self.assertEqual(len(CALL_LOG), len(state.theory))
        self.assertEqual(state.charge_store.atom_charges.tolist(),
                         [0.0, 0.7, 0.8])
        high_backend = calc.backends['theory'][
            [t.get('provides_electronic') for t in state.theory].index(True)]
        pc = np.asarray(high_backend.received_point_charges)
        np.testing.assert_allclose(pc[:, 0], [0.7, 0.8])


class TestOniom2ClassicalOuter(unittest.TestCase):
    """ONIOM2 + MM: embedding=true is a QMMM spelling, false warns."""

    def _data(self, **overrides):
        data = {'atoms': ['C', 'C'], 'natom': 2,
                'theory': ['pyscf', 'lammps'], 'reference': [],
                'highlevel': [0], 'midlevel': [],
                'embedding': False}
        data.update(overrides)
        return data

    def test_embedding_true_points_to_qmmm(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertRaises(ValueError) as ctx:
            validate_system(self._data(embedding=True))
        self.assertIn('qmmm', str(ctx.exception))

    def test_embedding_false_warns_equivalent_to_qmmm(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertWarns(UserWarning):
            out = validate_system(self._data(embedding=False))
        self.assertFalse(out['embedding'])


if __name__ == '__main__':
    unittest.main()
