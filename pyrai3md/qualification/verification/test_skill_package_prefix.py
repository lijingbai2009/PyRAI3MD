"""
======================================================================
The skills package name is read, not spelled (#470, 4.6).

`skills/query.py` and `skills/wrapper.py` built module paths as
`pyrai3md.skills.<name>...` at every site, so a build that places the
skills in another package (a third-party plugin following the same
layout) could not resolve them. Both now take the package from their
own import path: `query.SKILLS_PACKAGE` for the discovery layer and
`__name__` for the wrapper.

The criterion builds a throwaway package with one skill in it and asks
the discovery layer for that skill's registry by pointing the package
name at it, which the spelled path could not do.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import sys
import tempfile
import textwrap
import unittest

from pyrai3md.skills import query


class TestTheSkillsPackageIsRead(unittest.TestCase):
    """A skill outside pyrai3md.skills resolves through the same layer."""

    QUERY = textwrap.dedent('''
        def get_registry():
            return {'demo': {'description': 'demo skill',
                             'module': 'demo_backend',
                             'backend_class': 'DemoBackend',
                             'roles': {'qm': 'DemoBackend'},
                             'capabilities': ['energy'],
                             'tasks': {}}}
    ''')

    def _make_package(self, root):
        package = os.path.join(root, 'third_party_skills')
        skill = os.path.join(package, 'demo_skill')
        os.makedirs(skill)
        for path in (os.path.join(package, '__init__.py'),
                     os.path.join(skill, '__init__.py')):
            open(path, 'w').close()
        with open(os.path.join(skill, 'query.py'), 'w') as handle:
            handle.write(self.QUERY)
        return package

    def test_the_registry_follows_the_package_name(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_package(root)
            sys.path.insert(0, root)
            previous = query.SKILLS_PACKAGE
            query.SKILLS_PACKAGE = 'third_party_skills'
            try:
                registry = query.get_registry('demo_skill')
            finally:
                query.SKILLS_PACKAGE = previous
                sys.path.remove(root)
                for name in list(sys.modules):
                    if name.startswith('third_party_skills'):
                        del sys.modules[name]
        entry = registry['demo']
        self.assertTrue(entry['module'].startswith('third_party_skills.'),
                        entry['module'])
        self.assertEqual(entry['roles'], {'qm': 'DemoBackend'})

    def test_the_prefix_is_not_spelled_at_the_call_sites(self):
        from pyrai3md.qualification.verification._source_paths import (
            source_read)
        for path in ('pyrai3md/skills/query.py',
                     'pyrai3md/skills/wrapper.py'):
            text = source_read(path)
            code = '\n'.join(line for line in text.splitlines()
                             if 'pyrai3md.skills.' in line
                             and not line.lstrip().startswith('#')
                             and "e.g. '" not in line)
            self.assertEqual(code.strip(), '', path)


if __name__ == '__main__':
    unittest.main()