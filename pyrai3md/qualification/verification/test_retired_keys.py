"""
======================================================================
Every retired spelling is refused by name, with the sentence that says
what to write instead (#5.9).

Three tables hold retired keys, and each is exercised here against the
code that refuses it, so adding a key to a table is adding a criterion:

- RETIRED_JOBTYPES: refused where the alias table is applied (normalize),
  which is the first place a jobtype is read;
- CONSTRAINT_REFUSED_KEYS: refused by the `&constraint` keyword check;
- RETIRE_DYNAMICS_KEYS: already walked key by key in
  test_md_keywords.py::TestRetiredDynamicsKeywords, which is why this
  module does not repeat it -- one criterion per key, one table.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.schema.adapter import CONSTRAINT_REFUSED_KEYS, \
    _check_constraint_keywords
from pyrai3md.schema.normalize import normalize_jobtype
from pyrai3md.schema.validate import RETIRED_JOBTYPES


class TestRetiredJobtypes(unittest.TestCase):
    """`hop` and `hybrid` are refused with their replacement."""

    def test_normalize_refuses_each_retired_token_with_its_reason(self):
        for token, reason in RETIRED_JOBTYPES.items():
            with self.subTest(token):
                with self.assertRaises(ValueError) as raised:
                    normalize_jobtype({'jobtype': 'energy ' + token})
                self.assertIn("'%s' has been removed" % token,
                              str(raised.exception))
                self.assertIn(reason[:40], str(raised.exception))

    def test_a_live_alias_is_still_rewritten(self):
        from pyrai3md.schema.validate import JOBTYPE_ALIASES
        self.assertNotIn('hop', JOBTYPE_ALIASES)
        self.assertNotIn('hybrid', JOBTYPE_ALIASES)
        self.assertEqual(normalize_jobtype({'jobtype': 'sp'})['jobtype'],
                         'energy')


class TestRefusedConstraintKeys(unittest.TestCase):
    """Each refused keyword answers with its own sentence."""

    def test_every_refused_key_answers_with_its_reason(self):
        for key, reason in CONSTRAINT_REFUSED_KEYS.items():
            with self.subTest(key):
                with self.assertRaises(ValueError) as raised:
                    _check_constraint_keywords({key: '1'})
                message = str(raised.exception)
                self.assertIn("'%s' is refused" % key, message)
                self.assertIn(reason[:40], message)

    def test_an_unknown_key_that_is_not_retired_names_its_family(self):
        with self.assertRaises(ValueError) as raised:
            _check_constraint_keywords({'bias_bond_fc_typo': '1'})
        message = str(raised.exception)
        self.assertIn("does not know 'bias_bond_fc_typo'", message)
        self.assertNotIn('is refused', message)


if __name__ == '__main__':
    unittest.main()