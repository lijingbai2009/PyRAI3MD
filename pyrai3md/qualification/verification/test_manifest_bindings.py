"""
======================================================================
Manifest-declared bindings: the backend class and a composite's roles.

Two hardcoded linkages are pinned here.

The backend class. `Maestro.load_skill` used to try the manifest
first, then a module-level `Backend` attribute, then a name built from
the package name (`orca_skill` -> `OrcaBackend`). The last two could
only fire when the manifest was wrong, and they turned that into a
loaded skill with a guessed class. The manifest is now the only source;
a skill that declares none fails to load by name.

The QMMM theory binding. `_expand_qmmm` read control.theory by
position (first entry QM, last MM), so a queue handed in the other
order gave the QM region to the classical engine. The binding now asks
the manifest roles through `skills.query.role_of`, the same reading
the ONIOM2 classical-outer gate uses.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.kernel.maestro import Maestro
from pyrai3md.qualification.verification._source_paths import source_read
from pyrai3md.qualification.verification.test_task import (
    MockBackendA, MockBackendB, MockConfig, MockSystem, _skill, _skills)


class TestTheBackendClassIsDeclared(unittest.TestCase):
    """No guessed class, no module-level fallback."""

    def test_a_skill_without_a_declared_class_does_not_load(self):
        import pyrai3md.skills.orca_skill.query as orca_query
        original = orca_query.get_metadata
        orca_query.get_metadata = lambda: {'backend_class': ''}
        try:
            loaded = Maestro().load_skill('orca_skill')
        finally:
            orca_query.get_metadata = original
        self.assertFalse(loaded)

    def test_the_fallbacks_are_gone(self):
        text = source_read('pyrai3md/kernel/maestro.py')
        self.assertNotIn("replace('_skill'", text)
        self.assertNotIn("getattr(skill_module, 'Backend'", text)
        self.assertIn('No backend class found', text)

    def test_every_shipped_skill_declares_one(self):
        maestro = Maestro()
        for name in ('bagel_skill', 'cp2k_skill', 'lammps_skill',
                     'molcas_skill', 'nn_pes_skill', 'openqp_skill',
                     'orca_skill', 'pyscf_skill', 'qdm_skill',
                     'xtb_skill'):
            with self.subTest(name):
                self.assertTrue(maestro.load_skill(name))


class TestTheCompositeRolesDecideTheBinding(unittest.TestCase):
    """control.theory order no longer decides which side a skill is."""

    @staticmethod
    def _queue(theory):
        system = MockSystem(highlevel=[[0, 1]], natom=6, atoms=['C'] * 6)
        config = MockConfig(system, jobtype='energy',
                            multiscale_type='qmmm')
        from pyrai3md.planning.task import resolve_task_queue
        return resolve_task_queue(config, _skills(theory=theory))

    def test_both_orders_bind_the_quantum_side_to_the_qm_skill(self):
        orders = (
            {'molcas': _skill('qm', MockBackendA),
             'lammps': _skill('mm', MockBackendB)},
            {'lammps': _skill('mm', MockBackendB),
             'molcas': _skill('qm', MockBackendA)},
        )
        for theory in orders:
            with self.subTest(order=list(theory)):
                roles = {task.get('qmmm_role'): task.get('skill_name')
                         for task in self._queue(theory)['theory']}
                self.assertEqual(roles['qm'], 'molcas')
                self.assertEqual(roles['energy'], 'lammps')

    def test_two_quantum_providers_are_refused(self):
        theory = {'molcas': _skill('qm', MockBackendA),
                  'orca': _skill('qm', MockBackendB)}
        with self.assertRaises(ValueError) as raised:
            self._queue(theory)
        self.assertIn('declared roles', str(raised.exception))

    def test_one_predicate_reads_the_roles(self):
        self.assertIn("role_of(label) == 'mm'",
                      source_read('pyrai3md/schema/validate.py'))
        self.assertIn('from ..skills.query import role_of',
                      source_read('pyrai3md/planning/task.py'))
        for label in ('molcas', 'lammps'):
            with self.subTest(label):
                from pyrai3md.skills.query import role_of
                self.assertIn(role_of(label), ('qm', 'mm'))


if __name__ == '__main__':
    unittest.main()