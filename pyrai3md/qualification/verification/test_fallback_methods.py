"""
======================================================================
Guard the static theory-method fallback against the shipped skills.

`schema/validate.py::FALLBACK_THEORY_METHODS` is used when the skill
registry cannot be discovered (a partial installation). It is a second
list, so it drifts: a skill added or renamed must be reflected there.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.schema.validate import (
    BUILTIN_THEORY_METHODS, FALLBACK_THEORY_METHODS)
from pyrai3md.skills.query import discover_all


class FallbackMatchesShippedSkills(unittest.TestCase):
    def test_fallback_and_registry_agree(self):
        labels = set(discover_all())
        fallback = set(FALLBACK_THEORY_METHODS) - set(BUILTIN_THEORY_METHODS)
        self.assertEqual(fallback - labels, set(),
                         'fallback names a skill that is not installed')
        self.assertEqual(labels - set(FALLBACK_THEORY_METHODS), set(),
                         'a shipped skill is missing from the fallback')


if __name__ == '__main__':
    unittest.main()
