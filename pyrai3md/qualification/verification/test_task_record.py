"""
======================================================================
Guard for the persisted task field set.

`data/task.TASK_FIELDS` is the one list `PyRAIState.to_dict` reads. An
assembled task that carries a key outside it would be silently dropped
from the dump, which is how `multipool`, `external_charges` and
`external_potential` were lost before.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import unittest

from pyrai3md.qualification.verification._source_paths import source_path


class TestTaskFieldSet(unittest.TestCase):

    def test_an_assembled_task_has_no_key_outside_the_list(self):
        from pyrai3md.data.task import TASK_FIELDS

        here = os.getcwd()
        os.chdir(source_path('pyrai3md/examples'))
        try:
            from pyrai3md.interface.parser import parse_input
            from pyrai3md.schema.adapter import build_config
            from pyrai3md.schema.normalize import normalize
            from pyrai3md.schema.validate import validate
            from pyrai3md.data.build import build_state

            raw = parse_input('energy.yaml')
            config = validate(normalize(build_config(raw)))
            state = build_state(config, raw_config=raw)
        finally:
            os.chdir(here)

        dumped = state.to_dict()
        tasks = dumped['theory'] + dumped['reference']
        self.assertTrue(tasks)
        keys = set().union(*(set(task) for task in tasks))
        self.assertLessEqual(keys, set(TASK_FIELDS) | {'config'},
                             'a task key would be lost by the dump')
        # The three fields that used to be lost are carried explicitly.
        for name in ('multipool', 'external_charges', 'external_potential'):
            self.assertIn(name, keys)

    def test_list_fields_are_copied_not_shared(self):
        from pyrai3md.data.task import dump_task

        region = [0, 1]
        capabilities = ['energy']
        task = {'region_indices': region, 'capabilities': capabilities}
        dumped = dump_task(task)
        dumped['region_indices'].append(2)
        dumped['capabilities'].append('gradient')
        self.assertEqual(region, [0, 1])
        self.assertEqual(capabilities, ['energy'])


if __name__ == '__main__':
    unittest.main()
