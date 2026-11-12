"""
======================================================================
This module locks the failure of a broken skill default table.

`_skill_defaults` used to answer `{}` for any failure, so a query
module that raised rendered its section with none of that skill's
keywords and the round stayed green. The lookup now lets the error
through, and the planner turns it into a diagnostic row for that case:
the failure is recorded, it names the case, and the rest of the round
still plans.

The criterion is the injected `ValueError`: before the change it was
invisible, after it is a skip row whose reason names the case and the
error.

Nov 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest
from unittest import mock

from pyrai3md.qualification.certification import handler, planner


class TestABrokenDefaultTableIsVisible(unittest.TestCase):
    """The error is not read as an empty table."""

    def test_the_lookup_raises_instead_of_returning_empty(self):
        import importlib
        module = importlib.import_module('pyrai3md.skills.pyscf_skill.query')
        with mock.patch.object(module, 'get_defaults',
                               side_effect=ValueError('broken table')):
            with self.assertRaises(ValueError) as caught:
                handler._skill_defaults('pyscf_skill')
        self.assertIn('broken table', str(caught.exception))

    def test_the_planner_row_names_the_case_and_the_error(self):
        config = {'_runall': 'condition',
                  'skills': {'pyscf_skill': {'test': True}}}
        with tempfile.TemporaryDirectory() as work_dir, \
                mock.patch.object(handler, '_skill_defaults',
                                  side_effect=ValueError('broken table')):
            tasks = planner.plan_skill_tests(config, work_dir)
        diagnostics = [task for task in tasks
                       if task.get('name') == 'pyscf_skill_diagnostic']
        self.assertTrue(diagnostics, [task.get('name') for task in tasks])
        reason = diagnostics[0]['skip_reason']
        self.assertIn('test planning failed', reason)
        self.assertIn('broken table', reason)


if __name__ == '__main__':
    unittest.main()