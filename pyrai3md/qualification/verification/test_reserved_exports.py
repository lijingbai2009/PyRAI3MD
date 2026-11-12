"""
======================================================================
Guard the reserved public symbols kept by the review.

The review removed dead implementations and kept the few symbols that
are public or reserved for a future entry point. They must stay
importable; a later cleanup that deletes one has to say so here.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest


class ReservedExportsStayImportable(unittest.TestCase):
    def test_reserved_symbols(self):
        from pyrai3md.skills.query import get_api
        from pyrai3md.skills.nn_pes_skill.backend import load_backend
        from pyrai3md.skills.molcas_skill.service.builder.module_builders \
            import build_mclr
        from pyrai3md.workflow.dispatch.terms import audit_key

        for function in (get_api, load_backend, build_mclr, audit_key):
            self.assertTrue(callable(function), function)


if __name__ == '__main__':
    unittest.main()
