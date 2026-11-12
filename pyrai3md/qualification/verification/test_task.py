"""
======================================================================
Test suite for the planning task queue resolution
(pyrai3md/planning/task.py).

Tests cover:
- resolve_composite_type() inference from skills and system keywords
- resolve_task_queue() expansion for single, ONIOM2, ONIOM3,
  fragment and QM/MM workflows
- Region slice indices and contribution coefficients
- Capability restriction for composite non-primary tasks
- Static execution identifiers (task_id, layer_id, static_id)

Aug 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import unittest
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class MockBackendA:
    """Placeholder backend class for task descriptors."""


class MockBackendB:
    """Placeholder backend class for task descriptors."""


class MockBackendC:
    """Placeholder backend class for task descriptors."""


class MockSystem:
    """Minimal system-like object carrying region keywords."""

    def __init__(self, highlevel=None, midlevel=None, center_groups=None,
                 qmmm_key=None, natom=0, atoms=None, nstate=1,
                 constraints=None, embedding=False, read_charge=False):
        self.highlevel = highlevel if highlevel is not None else []
        self.midlevel = midlevel if midlevel is not None else []
        self.center_groups = [list(g) for g in (center_groups or [])]
        self.qmmm_key = qmmm_key
        self.natom = natom
        self.atoms = atoms or []
        self.nstate = nstate
        self.constraints = constraints
        self.embedding = embedding
        self.read_charge = read_charge


class MockConfig:
    """Minimal config-like object exposing the system submodel."""

    def __init__(self, system=None, jobtype='', multiscale_type=''):
        self.system = system or MockSystem()
        self.control = SimpleNamespace(jobtype=jobtype,
                                       multiscale_type=multiscale_type)
        self.optimize = None


def _skills(theory=None, reference=None):
    """Build a resolved-skills style dict."""
    return {'theory': theory or {}, 'reference': reference or {}}


def _skill(role, backend, capabilities=None):
    return {
        'role': role,
        'backend': backend,
        'capabilities': list(capabilities
                             or ['energy', 'gradient', 'nac']),
    }


class TestResolveCompositeType(unittest.TestCase):
    """Test composite type inference."""

    def test_single_backend_no_composite(self):
        from pyrai3md.planning.task import resolve_composite_type
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        self.assertIsNone(resolve_composite_type(MockConfig(), skills))

    def test_two_theory_oniom2(self):
        from pyrai3md.planning.task import resolve_composite_type
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        self.assertEqual(resolve_composite_type(MockConfig(), skills),
                         'oniom2')

    def test_three_theory_oniom3(self):
        from pyrai3md.planning.task import resolve_composite_type
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB),
                                 'c': _skill('qm3', MockBackendC)})
        self.assertEqual(resolve_composite_type(MockConfig(), skills),
                         'oniom3')

    def test_center_groups_fragment(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(system=MockSystem(center_groups=[[0], [1]]))
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        self.assertEqual(resolve_composite_type(config, skills),
                         'fragment')

    def test_qmmm_key_is_retired(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(system=MockSystem(qmmm_key='tinker'))
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        with self.assertRaises(ValueError) as context:
            resolve_composite_type(config, skills)
        self.assertIn('retired', str(context.exception))

    def test_theory_plus_reference_no_longer_qmmm(self):
        from pyrai3md.planning.task import resolve_composite_type
        skills = _skills(theory={'a': _skill('qm', MockBackendA)},
                         reference={'b': _skill('mm', MockBackendB)})
        self.assertIsNone(resolve_composite_type(MockConfig(), skills))

    def test_two_methods_selected_by_multiscale_type(self):
        from pyrai3md.planning.task import resolve_composite_type
        system = MockSystem(highlevel=[0, 1], natom=6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('mm', MockBackendB)})
        self.assertEqual(
            resolve_composite_type(
                MockConfig(system=system, jobtype='energy gradient'),
                skills), 'oniom2')
        self.assertEqual(
            resolve_composite_type(
                MockConfig(system=system, jobtype='energy gradient',
                           multiscale_type='oniom'),
                skills), 'oniom2')
        self.assertEqual(
            resolve_composite_type(
                MockConfig(system=system, jobtype='energy gradient',
                           multiscale_type='qmmm'),
                skills), 'qmmm')

    def test_qmmm_requires_two_theory_methods(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(
            system=MockSystem(highlevel=[0, 1], natom=6),
            jobtype='energy gradient', multiscale_type='qmmm')
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        with self.assertRaises(ValueError):
            resolve_composite_type(config, skills)

    def test_qmmm_requires_a_qm_region(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(
            system=MockSystem(natom=6),
            jobtype='energy gradient', multiscale_type='qmmm')
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('mm', MockBackendB)})
        with self.assertRaises(ValueError):
            resolve_composite_type(config, skills)

    def test_qmmm_accepts_advanced_jobtypes(self):
        from pyrai3md.planning.task import resolve_composite_type
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('mm', MockBackendB)})
        for jobtype in ('md', 'optimize', 'surface_hopping', 'num_hess'):
            config = MockConfig(
                system=MockSystem(highlevel=[0, 1], natom=6),
                jobtype=jobtype, multiscale_type='qmmm')
            self.assertEqual(resolve_composite_type(config, skills), 'qmmm')

    def test_qmmm_accepts_multi_state(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(
            system=MockSystem(highlevel=[0, 1], natom=6, nstate=2),
            jobtype='energy gradient', multiscale_type='qmmm')
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('mm', MockBackendB)})
        self.assertEqual(resolve_composite_type(config, skills), 'qmmm')

    def test_adaptive_layered_theory_is_oniom2(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(
            system=MockSystem(highlevel=[[0, 1]], natom=6),
            jobtype='adaptive')
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)},
                         reference={'c': _skill('qm', MockBackendA)})
        self.assertEqual(resolve_composite_type(config, skills), 'oniom2')

    def test_adaptive_single_theory_ignores_reference(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(jobtype='adaptive')
        skills = _skills(theory={'a': _skill('qm', MockBackendA)},
                         reference={'b': _skill('qm', MockBackendA)})
        self.assertIsNone(resolve_composite_type(config, skills))

    def test_adaptive_layered_theory_needs_highlevel(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(jobtype='adaptive')
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        with self.assertRaises(ValueError):
            resolve_composite_type(config, skills)

    def test_adaptive_multi_center_is_refused(self):
        from pyrai3md.planning.task import resolve_composite_type
        config = MockConfig(
            system=MockSystem(highlevel=[[0, 1], [2, 3]], natom=6),
            jobtype='adaptive')
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        with self.assertRaises(ValueError):
            resolve_composite_type(config, skills)


class TestResolveTaskQueueSingle(unittest.TestCase):
    """Test queue expansion without composite."""

    def test_single_task(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        queue = resolve_task_queue(MockConfig(), skills)
        self.assertEqual(len(queue['theory']), 1)
        task = queue['theory'][0]
        self.assertEqual(task['role'], 'qm')
        self.assertEqual(task['task'], 0)
        self.assertEqual(task['contribution'], 1)
        self.assertEqual(task['region'], 'all')
        self.assertEqual(task['region_indices'], [])
        self.assertIsNone(task['region_charge'])
        self.assertIs(task['backend'], MockBackendA)

    def test_reference_queue(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(reference={'b': _skill('mm', MockBackendB)})
        queue = resolve_task_queue(MockConfig(), skills)
        self.assertEqual(len(queue['reference']), 1)
        self.assertEqual(queue['reference'][0]['role'], 'mm')


class TestResolveTaskQueueOniom2(unittest.TestCase):
    """Test ONIOM2 queue expansion."""

    def _queue(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0, 1])
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        return resolve_task_queue(MockConfig(system=system), skills)

    def test_three_tasks(self):
        queue = self._queue()
        self.assertEqual(len(queue['theory']), 3)

    def test_contributions(self):
        queue = self._queue()
        contributions = [t['contribution'] for t in queue['theory']]
        self.assertEqual(contributions, [1, 1, -1])

    def test_roles_and_indices(self):
        queue = self._queue()
        tasks = queue['theory']
        # real-first order (docs 8): L(real) baseline first, then per
        # center +H(M_c), -L(M_c).
        self.assertEqual(tasks[0]['role'], 'qm2')
        self.assertEqual(tasks[0]['region'], 'real')
        self.assertEqual(tasks[0]['region_indices'], [])
        self.assertEqual(tasks[1]['role'], 'qm')
        self.assertEqual(tasks[1]['region'], 'model')
        self.assertEqual(tasks[1]['region_indices'], [0, 1])
        self.assertEqual(tasks[2]['role'], 'qm2')
        self.assertEqual(tasks[2]['task'], 1)
        self.assertEqual(tasks[2]['region'], 'model')
        self.assertEqual(tasks[2]['region_indices'], [0, 1])

    def test_capability_restriction(self):
        queue = self._queue()
        tasks = queue['theory']
        # The electronic provider H(c) (real-first slot 1) keeps full
        # capabilities; the L(real) baseline and the -L(M) reference
        # correction are restricted to energy + gradient.
        self.assertIn('nac', tasks[1]['capabilities'])
        self.assertEqual(tasks[0]['capabilities'], ['energy', 'gradient'])
        self.assertEqual(tasks[2]['capabilities'], ['energy', 'gradient'])

    def test_adaptive_reference_labels_the_high_region(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[1, 2], natom=6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)},
                         reference={'c': _skill('qm', MockBackendA)})
        queue = resolve_task_queue(
            MockConfig(system=system, jobtype='adaptive'), skills)
        self.assertEqual(len(queue['theory']), 3)
        self.assertEqual(len(queue['reference']), 1)
        reference = queue['reference'][0]
        self.assertEqual(reference['region'], 'model')
        self.assertEqual(reference['region_indices'], [1, 2])
        self.assertIsNone(reference['center_index'])


class TestResolveTaskQueueOniom3(unittest.TestCase):
    """Test ONIOM3 queue expansion."""

    def _queue(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0], midlevel=[1, 2])
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB),
                                 'c': _skill('qm3', MockBackendC)})
        return resolve_task_queue(MockConfig(system=system), skills)

    def test_five_tasks(self):
        queue = self._queue()
        self.assertEqual(len(queue['theory']), 5)

    def test_contributions(self):
        queue = self._queue()
        contributions = [t['contribution'] for t in queue['theory']]
        self.assertEqual(contributions, [1, 1, -1, 1, -1])

    def test_regions(self):
        queue = self._queue()
        regions = [t['region'] for t in queue['theory']]
        # real-first order (docs 8): L(real); per center +M(I_c),
        # -L(I_c), +H(M_c), -M(M_c).
        self.assertEqual(regions,
                         ['real', 'inter', 'inter', 'model', 'model'])

    def test_indices(self):
        queue = self._queue()
        tasks = queue['theory']
        # real-first: L(real)=[], +M(I)=[0,1,2], -L(I)=[0,1,2],
        # +H(M)=[0], -M(M)=[0]
        self.assertEqual(tasks[0]['region_indices'], [])
        self.assertEqual(tasks[1]['region_indices'], [0, 1, 2])
        self.assertEqual(tasks[3]['region_indices'], [0])
        self.assertEqual(tasks[4]['region_indices'], [0])


class TestResolveTaskQueueFragment(unittest.TestCase):
    """Test fragment queue expansion."""

    def test_one_task_per_group(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(center_groups=[[0, 1], [2], [3, 4]])
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        queue = resolve_task_queue(MockConfig(system=system), skills)
        tasks = queue['theory']
        self.assertEqual(len(tasks), 3)
        indices = [t['region_indices'] for t in tasks]
        self.assertEqual(indices, [[0, 1], [2], [3, 4]])
        contributions = [t['contribution'] for t in tasks]
        self.assertEqual(contributions, [1, 1, 1])
        task_ids = [t['task'] for t in tasks]
        self.assertEqual(task_ids, [0, 1, 2])


class TestResolveTaskQueueQmmm(unittest.TestCase):
    """Test QM/MM queue expansion (field -> QM -> energy)."""

    def test_field_qm_energy_stages(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0, 1], natom=6, embedding=True)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(
            MockConfig(system=system, jobtype='energy gradient',
                       multiscale_type='qmmm'), skills)
        tasks = queue['theory']
        self.assertEqual(len(tasks), 3)
        self.assertEqual(queue['reference'], [])
        self.assertEqual([task['qmmm_role'] for task in tasks],
                         ['field', 'qm', 'energy'])
        self.assertEqual([task['contribution'] for task in tasks],
                         [0, 1, 1])
        field, qm, energy = tasks
        self.assertEqual(field['role'], 'field')
        self.assertEqual(field['region'], 'all')
        self.assertEqual(field['binding_role'], 'qm2')
        self.assertFalse(field['provides_electronic'])
        self.assertNotIn('stress', field['capabilities'])
        self.assertEqual(qm['role'], 'qm')
        self.assertEqual(qm['region'], 'model')
        self.assertEqual(qm['region_indices'], [0, 1])
        self.assertEqual(qm['center_index'], 0)
        self.assertTrue(qm['provides_electronic'])
        self.assertNotIn('stress', qm['capabilities'])
        self.assertEqual(energy['role'], 'energy')
        self.assertEqual(energy['region'], 'all')
        self.assertEqual(energy['binding_role'], 'qm2')
        self.assertFalse(energy['provides_electronic'])
        self.assertNotIn('stress', energy['capabilities'])
        for task in tasks:
            self.assertEqual(task['composite'], 'qmmm')


class TestStaticExecutionId(unittest.TestCase):
    """Test semantic static execution identifiers (static_id).

    Single-shot jobs keep the plain skill working-directory name
    ('tmp_{SKILL}'); composite/multitask jobs are named by region,
    role and (when several tasks share them) a 1-based ordinal,
    then the skill tag ('tmp_{region}_{role}[_task{N}]_{SKILL}').
    The region is the human-facing layer name (model->high,
    inter->mid, real->low); the _task{N} ordinal appears only when
    several tasks share the same (region, role, skill), e.g. a
    multi-center ONIOM level or a multi-group fragment.
    """

    def test_single_task_id(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(theory={'pyscf_skill': _skill('qm', MockBackendA)})
        queue = resolve_task_queue(MockConfig(), skills)
        task = queue['theory'][0]
        self.assertEqual(task['task_id'], 'qm')
        self.assertEqual(task['layer_id'], 0)
        self.assertEqual(task['static_id'], 'tmp_PYSCF')

    def test_reference_task_id(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(reference={'molcas_skill': _skill('mm', MockBackendB)})
        queue = resolve_task_queue(MockConfig(), skills)
        task = queue['reference'][0]
        self.assertEqual(task['static_id'], 'tmp_MOLCAS')

    def test_oniom2_layer_ids(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0, 1])
        skills = _skills(theory={'pyscf_skill': _skill('qm', MockBackendA),
                                 'xtb_skill': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(MockConfig(system=system), skills)
        tasks = queue['theory']
        ids = [t['static_id'] for t in tasks]
        # real-first: L(real), +H(M_c), -L(M_c). Each (region, role,
        # skill) group holds one task, so no _task{N} ordinal.
        self.assertEqual(ids, ['tmp_low_qm2_XTB',
                               'tmp_high_qm_PYSCF',
                               'tmp_high_qm2_XTB'])

    def test_oniom2_same_skill_unique_ids(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0, 1])
        # Same skill serves both layers: ids must stay unique
        skills = _skills(theory={'pyscf_skill': _skill('qm', MockBackendA),
                                 'pyscf2_skill': _skill('qm2', MockBackendA)})
        queue = resolve_task_queue(MockConfig(system=system), skills)
        ids = [t['static_id'] for t in queue['theory']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_oniom3_layer_ids(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0], midlevel=[1, 2])
        skills = _skills(theory={'a_skill': _skill('qm', MockBackendA),
                                 'b_skill': _skill('qm2', MockBackendB),
                                 'c_skill': _skill('qm3', MockBackendC)})
        queue = resolve_task_queue(MockConfig(system=system), skills)
        layer_ids = [t['layer_id'] for t in queue['theory']]
        # real-first: L(real)=2, +M(I)=1, -L(I)=2, +H(M)=0, -M(M)=1
        self.assertEqual(layer_ids, [2, 1, 2, 0, 1])
        ids = [t['static_id'] for t in queue['theory']]
        self.assertEqual(ids, ['tmp_low_qm3_C', 'tmp_mid_qm2_B',
                               'tmp_mid_qm3_C', 'tmp_high_qm_A',
                               'tmp_high_qm2_B'])

    def test_fragment_layer_ids(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(center_groups=[[0, 1], [2]])
        skills = _skills(theory={'pyscf_skill': _skill('qm', MockBackendA)})
        queue = resolve_task_queue(MockConfig(system=system), skills)
        tasks = queue['theory']
        self.assertEqual([t['layer_id'] for t in tasks], [0, 0])
        # Both fragments share (region, role, skill): 1-based ordinal
        # in slot order disambiguates them.
        self.assertEqual(tasks[0]['static_id'], 'tmp_fragment_qm_task1_PYSCF')
        self.assertEqual(tasks[1]['static_id'], 'tmp_fragment_qm_task2_PYSCF')

    def test_qmmm_layer_ids(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0, 1], natom=6, embedding=True)
        skills = _skills(theory={'pyscf_skill': _skill('qm', MockBackendA),
                                 'openqp_skill': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(
            MockConfig(system=system, jobtype='energy gradient',
                       multiscale_type='qmmm'), skills)
        tasks = queue['theory']
        self.assertEqual(tasks[0]['layer_id'], 2)
        self.assertEqual(tasks[0]['static_id'], 'tmp_all_field_OPENQP')
        self.assertEqual(tasks[1]['layer_id'], 0)
        self.assertEqual(tasks[1]['static_id'], 'tmp_high_qm_PYSCF')
        self.assertEqual(tasks[2]['layer_id'], 2)
        self.assertEqual(tasks[2]['static_id'], 'tmp_all_energy_OPENQP')

    def test_multicenter_oniom3_region_ids(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[[0], [3]],
                            midlevel=[[1, 2], [4, 5]], natom=6)
        skills = _skills(theory={'a_skill': _skill('qm', MockBackendA),
                                 'b_skill': _skill('qm2', MockBackendB),
                                 'c_skill': _skill('qm3', MockBackendC)})
        queue = resolve_task_queue(MockConfig(system=system), skills)
        ids = [t['static_id'] for t in queue['theory']]
        # real-first, two centers: L(real) is center-less (no ordinal);
        # every per-center level carries a 1-based _task{N} ordinal
        # (center 0 -> task1, center 1 -> task2).
        self.assertEqual(ids, [
            'tmp_low_qm3_C',
            'tmp_mid_qm2_task1_B', 'tmp_mid_qm3_task1_C',
            'tmp_high_qm_task1_A', 'tmp_high_qm2_task1_B',
            'tmp_mid_qm2_task2_B', 'tmp_mid_qm3_task2_C',
            'tmp_high_qm_task2_A', 'tmp_high_qm2_task2_B'])
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()


class TestMultiCenterDetection(unittest.TestCase):
    """Test multi-center ONIOM type detection."""

    def test_nested_highlevel_oniom2(self):
        from pyrai3md.planning.task import resolve_composite_type
        system = MockSystem(highlevel=[[0, 1], [4, 5]], natom=6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        self.assertEqual(resolve_composite_type(MockConfig(system), skills),
                         'multi_center_oniom2')

    def test_nested_highlevel_oniom3(self):
        from pyrai3md.planning.task import resolve_composite_type
        system = MockSystem(highlevel=[[0], [3]],
                            midlevel=[[1], [4]], natom=6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB),
                                 'c': _skill('qm3', MockBackendC)})
        self.assertEqual(resolve_composite_type(MockConfig(system), skills),
                         'multi_center_oniom3')

    def test_flat_highlevel_not_multi_center(self):
        from pyrai3md.planning.task import resolve_composite_type
        system = MockSystem(highlevel=[0, 1])
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        self.assertEqual(resolve_composite_type(MockConfig(system), skills),
                         'oniom2')

    def test_multi_center_conflicts_with_center_groups(self):
        from pyrai3md.planning.task import resolve_composite_type
        system = MockSystem(highlevel=[[0, 1], [4, 5]],
                            center_groups=[[0], [1]])
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        with self.assertRaises(ValueError):
            resolve_composite_type(MockConfig(system), skills)


class TestMultiCenterExpansion(unittest.TestCase):
    """Test multi-center ONIOM task queue expansion."""

    def test_two_center_oniom2_task_count(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[[0, 1], [4, 5]],
                            natom=6, atoms=['C'] * 6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(MockConfig(system), skills)
        # 1+2N = 1+4 = 5
        self.assertEqual(len(queue['theory']), 5)

    def test_two_center_oniom2_contributions(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[[0, 1], [4, 5]],
                            natom=6, atoms=['C'] * 6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(MockConfig(system), skills)
        contributions = [t['contribution'] for t in queue['theory']]
        # L(real)=+1, per center: +H, -L, +H, -L
        self.assertEqual(contributions, [1, 1, -1, 1, -1])

    def test_two_center_oniom2_center_indices(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[[0, 1], [4, 5]],
                            natom=6, atoms=['C'] * 6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(MockConfig(system), skills)
        center_indices = [t.get('center_index') for t in queue['theory']]
        # real-first: L(real) has None, then c0, c0, c1, c1
        self.assertEqual(center_indices, [None, 0, 0, 1, 1])

    def test_three_layer_two_center_task_count(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[[0], [3]],
                            midlevel=[[1], [4]],
                            natom=6, atoms=['C'] * 6)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB),
                                 'c': _skill('qm3', MockBackendC)})
        queue = resolve_task_queue(MockConfig(system), skills)
        # 1+4N = 1+8 = 9
        self.assertEqual(len(queue['theory']), 9)


class TestRegionRulesHaveOneHome(unittest.TestCase):
    """The region invariants are the schema's; planning refuses only a
    blank region (`planning/task.py::_require_region`).

    Nine bad regions were measured: each one is refused by
    `schema/validate.py::validate_centers` with a field-path message,
    so the copy of the same rules that stood in planning could never
    answer first and was removed. The one case the schema cannot
    judge is a blank region -- with no index to look at it reads the
    run as having no region -- and planning still refuses it before
    the expander slices anything.
    """

    BAD_REGIONS = (
        ('high overlap', [[0, 1], [1, 2]], None),
        ('high duplicate', [[0, 0]], None),
        ('high out of range', [[0, 1], [4, 10]], None),
        ('empty center', [[0, 1], []], None),
        ('high/mid overlap', [[0, 1], [3]], [[0, 2], [3, 4]]),
        ('mid duplicate', [[0], [3]], [[1, 1], [4, 5]]),
        ('mid count mismatch', [[0], [3]], [[1]]),
        ('mid overlap', [[0], [5]], [[1], [1, 2]]),
        ('mid empty', [[0], [3]], [[], [4, 5]]),
    )

    def _schema_refusal(self, high, mid):
        from pyrai3md.schema.validate import validate_centers
        data = {'highlevel': high, 'midlevel': mid, 'natom': 6,
                'atoms': ['C'] * 6, 'ci': [1] * max(len(high), 1),
                'spin': [0] * max(len(high), 1), 'root': 0,
                'charges': 0, 'multiplicity': 1}
        with self.assertRaises(ValueError) as raised:
            validate_centers(data)
        return str(raised.exception)

    @staticmethod
    def _blank_region_config(high):
        import warnings
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        raw = {'control': {'title': 'blank_region', 'jobtype': 'energy',
                           'theory': ['orca', 'lammps']},
               'atoms': ['C'] * 6,
               'coordinates': [[float(i), 0.0, 0.0] for i in range(6)],
               'molecule': {'charges': 0, 'multiplicity': 1,
                            'highlevel': high},
               'orca': {'method': 'b3lyp', 'basis': 'def2-svp'},
               'lammps': {'method': 'uff'}}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return validate(normalize(build_config(raw)))

    def test_every_bad_region_is_refused_upstream(self):
        for name, high, mid in self.BAD_REGIONS:
            with self.subTest(name):
                self.assertIn('molecule.', self._schema_refusal(high, mid))

    def test_planning_keeps_only_the_blank_region_refusal(self):
        from pyrai3md.qualification.verification._source_paths import (
            source_read)
        text = source_read('pyrai3md/planning/task.py')
        for message in ('has duplicate indices', 'out of range [0, %d)',
                        'overlap at highlevel indices',
                        'overlap at midlevel indices', 'must be disjoint',
                        'midlevel region is empty'):
            self.assertNotIn(message, text)
        self.assertIn('highlevel region is empty', text)

    def test_a_blank_region_is_refused_per_center(self):
        from pyrai3md.planning.task import _require_region
        with self.assertRaises(ValueError) as raised:
            _require_region([[], []], 'QMMM')
        self.assertIn('QMMM center 0: highlevel region is empty.',
                      str(raised.exception))
        _require_region([[0, 1], [4, 5]], 'ONIOM')

    def test_planning_refuses_what_the_schema_reads_as_no_region(self):
        from pyrai3md.planning.task import (
            resolve_composite_type, resolve_task_queue)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm', MockBackendB)})
        for high in ([], [[], []]):
            with self.subTest(high=high):
                config = self._blank_region_config(high)
                resolve_composite_type(config, skills)
                with self.assertRaises(ValueError) as raised:
                    resolve_task_queue(config, skills)
                self.assertIn('highlevel region is empty',
                              str(raised.exception))


class TestPlanVersion(unittest.TestCase):
    """Test plan_version stamping."""

    def test_composite_tasks_get_version_1(self):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=[0, 1])
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        queue = resolve_task_queue(MockConfig(system), skills)
        for task in queue['theory']:
            self.assertEqual(task.get('plan_version'), 1)

    def test_single_task_gets_version_0(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        queue = resolve_task_queue(MockConfig(), skills)
        for task in queue['theory']:
            self.assertEqual(task.get('plan_version'), 0)


class TestR2UnifiedStaticPlan(unittest.TestCase):
    """R2 unified static plan acceptance (docs 6.1-6.3, gate 6-7).

    Single-center multi-task and multi-center flow through the SAME
    center-based expander: term counts scale as 1 / 1+2C / 1+4C, the
    full-system baseline L(real) is computed exactly once, every task
    carries a unique identity, and each center's electronics come from
    an explicit provider H(c) -- never from the mathematical position
    tasks[0] (which real-first reserves for L(real)).
    """

    def _single(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        return resolve_task_queue(MockConfig(), skills)

    def _oniom2(self, highlevel, natom=0):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=highlevel, natom=natom,
                            atoms=['C'] * natom)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        return resolve_task_queue(MockConfig(system), skills)

    def _oniom3(self, highlevel, midlevel, natom=0):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=highlevel, midlevel=midlevel,
                            natom=natom, atoms=['C'] * natom)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB),
                                 'c': _skill('qm3', MockBackendC)})
        return resolve_task_queue(MockConfig(system), skills)

    def test_term_counts_scale_with_centers(self):
        # 1 (single), 1+2C (ONIOM2), 1+4C (ONIOM3)
        self.assertEqual(len(self._single()['theory']), 1)
        self.assertEqual(len(self._oniom2([0, 1])['theory']), 3)    # C=1
        self.assertEqual(
            len(self._oniom3([0], [1, 2])['theory']), 5)         # C=1
        self.assertEqual(
            len(self._oniom2([[0, 1], [4, 5]], natom=6)['theory']),
            5)                                                      # C=2
        self.assertEqual(
            len(self._oniom3([[0], [3]], [[1], [4]],
                             natom=6)['theory']), 9)                # C=2

    def test_real_baseline_computed_once(self):
        for queue in (self._oniom2([0, 1]),
                      self._oniom2([[0, 1], [4, 5]], natom=6),
                      self._oniom3([0], [1, 2]),
                      self._oniom3([[0], [3]], [[1], [4]],
                                   natom=6)):
            tasks = queue['theory']
            real = [t for t in tasks if t['region'] == 'real']
            unbound = [t for t in tasks if t['center_index'] is None]
            # exactly one L(real), and it is the only center_index=None
            self.assertEqual(len(real), 1)
            self.assertEqual(len(unbound), 1)
            self.assertIs(real[0], unbound[0])

    def test_unique_task_identity_multi_center(self):
        for queue in (self._oniom2([[0, 1], [4, 5]], natom=6),
                      self._oniom3([[0], [3]], [[1], [4]],
                                   natom=6)):
            tasks = queue['theory']
            task_ids = [t['task_id'] for t in tasks]
            static_ids = [t['static_id'] for t in tasks]
            slots = [t['task_slot'] for t in tasks]
            self.assertEqual(len(task_ids), len(set(task_ids)))
            self.assertEqual(len(static_ids), len(set(static_ids)))
            self.assertEqual(slots, list(range(len(tasks))))

    def test_provider_per_center_and_map(self):
        tasks = self._oniom2([[0, 1], [4, 5]], natom=6)['theory']
        providers = [t for t in tasks if t['provides_electronic']]
        # exactly one provider H(c) per center, bound to its center
        self.assertEqual(len(providers), 2)
        self.assertEqual(
            sorted(t['center_index'] for t in providers), [0, 1])
        for t in providers:
            self.assertEqual(t['state_policy'], 'active')
            self.assertEqual(t['electronic_binding_id'],
                             'c%d' % t['center_index'])
        self.assertEqual(
            {t['center_index']: t['task_id'] for t in providers},
            {0: 'qm', 1: 'qm_task1'})

    def test_primary_task_is_provider_not_real(self):
        from pyrai3md.planning.task import resolve_primary_task
        tasks = self._oniom2([0, 1])['theory']
        primary = resolve_primary_task(tasks)
        # tasks[0] is the L(real) baseline; the primary must be H(c0)
        self.assertEqual(tasks[0]['region'], 'real')
        self.assertIsNot(primary, tasks[0])
        self.assertEqual(primary['task_id'], 'qm')
        self.assertEqual(primary['center_index'], 0)
        self.assertTrue(primary['provides_electronic'])

    def test_primary_task_lowest_center_wins(self):
        from pyrai3md.planning.task import resolve_primary_task
        tasks = self._oniom2([[0, 1], [4, 5]], natom=6)['theory']
        self.assertEqual(resolve_primary_task(tasks)['center_index'], 0)

    def test_single_method_binds_center_zero(self):
        from pyrai3md.planning.task import resolve_primary_task
        tasks = self._single()['theory']
        task = tasks[0]
        self.assertEqual(task['center_index'], 0)
        self.assertTrue(task['provides_electronic'])
        self.assertEqual(task['electronic_binding_id'], 'c0')
        self.assertEqual(task['state_policy'], 'active')
        self.assertIs(resolve_primary_task(tasks), task)

    def test_reference_terms_state_policy(self):
        real, high, low_model = self._oniom2([0, 1])['theory']
        # L(real): reference baseline, unbound, no binding id
        self.assertEqual(real['state_policy'], 'reference')
        self.assertEqual(real['reference_state'], 0)
        self.assertIsNone(real['center_index'])
        self.assertIsNone(real['electronic_binding_id'])
        # H(c0): active provider bound to the center
        self.assertEqual(high['state_policy'], 'active')
        self.assertEqual(high['electronic_binding_id'], 'c0')
        # -L(M_c0): reference correction bound to the same center
        self.assertEqual(low_model['state_policy'], 'reference')
        self.assertEqual(low_model['reference_state'], 0)
        self.assertEqual(low_model['center_index'], 0)
        self.assertEqual(low_model['electronic_binding_id'], 'c0')


class TestResolveSystemTask(unittest.TestCase):
    """Structural-track provider selection (dev/methods/
    ENSEMBLE_DEVELOPMENT.md section 4A).

    resolve_system_task is the parallel of resolve_primary_task for
    whole-system structural properties (stress/virial, later
    hessian/frequencies): a single method IS the whole system, a
    multiscale run resolves the ONIOM L(real) full-system baseline, and
    a multi-center run resolves the same center-less baseline so one
    global quantity (e.g. the 3x3 stress) is returned, never per center.
    """

    def _single(self):
        from pyrai3md.planning.task import resolve_task_queue
        skills = _skills(theory={'a': _skill('qm', MockBackendA)})
        return resolve_task_queue(MockConfig(), skills)

    def _oniom2(self, highlevel, natom=0):
        from pyrai3md.planning.task import resolve_task_queue
        system = MockSystem(highlevel=highlevel, natom=natom,
                            atoms=['C'] * natom)
        skills = _skills(theory={'a': _skill('qm', MockBackendA),
                                 'b': _skill('qm2', MockBackendB)})
        return resolve_task_queue(MockConfig(system), skills)

    def test_empty_queue_returns_none(self):
        from pyrai3md.planning.task import resolve_system_task
        self.assertIsNone(resolve_system_task([]))

    def test_single_method_is_the_whole_system(self):
        from pyrai3md.planning.task import resolve_system_task
        tasks = self._single()['theory']
        self.assertIs(resolve_system_task(tasks), tasks[0])

    def test_multiscale_resolves_real_baseline(self):
        from pyrai3md.planning.task import resolve_system_task
        tasks = self._oniom2([0, 1])['theory']
        system_task = resolve_system_task(tasks)
        self.assertEqual(system_task['region'], 'real')
        self.assertIsNone(system_task['center_index'])

    def test_multi_center_resolves_centerless_baseline(self):
        from pyrai3md.planning.task import resolve_system_task
        tasks = self._oniom2([[0, 1], [4, 5]], natom=6)['theory']
        system_task = resolve_system_task(tasks)
        self.assertEqual(system_task['region'], 'real')
        self.assertIsNone(system_task['center_index'])

    def test_distinct_from_primary_for_multiscale(self):
        # The two tracks resolve DIFFERENT tasks for a multiscale run:
        # the electronic primary is the high-level provider H(c), the
        # structural system task is the full-system L(real) baseline.
        from pyrai3md.planning.task import (resolve_system_task,
                                            resolve_primary_task)
        tasks = self._oniom2([0, 1])['theory']
        primary = resolve_primary_task(tasks)
        system_task = resolve_system_task(tasks)
        self.assertIsNot(primary, system_task)
        self.assertTrue(primary.get('provides_electronic'))
        self.assertEqual(system_task['region'], 'real')
