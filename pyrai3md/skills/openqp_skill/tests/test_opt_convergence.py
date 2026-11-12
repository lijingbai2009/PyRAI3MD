"""
======================================================================
The OpenQP log's optimization verdict, and who decides on it.

The engine writes `PyOQP: Geometry Optimization Has Converged` or
`... Has Not Converged. Reached The Maximum Iteration`. A run that stops
on its own step budget is not a failed calculation: the other skills
return `opt_converged: False` and the framework reports NOT CONVERGED.
OpenQP used to raise on the second marker, which made the direct-skill
leg red while the framework legs passed the same run (#466).

Nov 7 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

from pyrai3md.skills.openqp_skill.service.parser.log_parser import parse_log


def _parse(text):
    holder = tempfile.TemporaryDirectory()
    path = os.path.join(holder.name, 'demo.log')
    with open(path, 'w') as handle:
        handle.write(text)
    try:
        return parse_log(path, SimpleNamespace(runtype='optimize'))
    finally:
        holder.cleanup()


class TestTheOptimizationVerdict(unittest.TestCase):
    def test_converged_marker_sets_the_flag(self):
        status = _parse('PyOQP: Geometry Optimization Has Converged\n')
        self.assertTrue(status['completed'])
        self.assertTrue(status['opt_converged'])

    def test_budget_stop_is_a_result_not_an_error(self):
        status = _parse(
            'PyOQP: Geometry Optimization Has Not Converged. '
            'Reached The Maximum Iteration\n')
        self.assertTrue(status['completed'])
        self.assertFalse(status['opt_converged'])
        self.assertEqual(status['error'], '')

    def test_a_real_failure_is_still_an_error(self):
        status = _parse('PyOQP: SCF energy is not converged\n')
        self.assertFalse(status['completed'])
        self.assertIn('SCF', status['error'])


if __name__ == '__main__':
    unittest.main()
