"""
======================================================================
Guard for the framework manifest (pyrai3md/manifest.yaml).

The manifest describes the framework itself with the same discovery
vocabulary a skill uses: the input schema a user may write and the code
entry points that consume it. These assertions keep the two halves from
drifting away from the code:

- every path it names resolves;
- every keyword it lists is accepted by the section's model or adapter
  table;
- the retired `qm`/`abinit` spellings stay routed to `theory`/`reference`.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import unittest

import yaml

from pyrai3md.qualification.verification._source_paths import source_read

MANIFEST = 'pyrai3md/manifest.yaml'


def _resolve(path):
    """Import a `module.attr` path, or the module itself when it has no attr."""
    try:
        return importlib.import_module(path)
    except ImportError:
        module, _, attr = path.rpartition('.')
        return getattr(importlib.import_module(module), attr)


def _walk_paths(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_paths(value)
    elif isinstance(node, str) and node.startswith('pyrai3md.'):
        yield node


class TestFrameworkManifest(unittest.TestCase):

    def setUp(self):
        self.manifest = yaml.safe_load(source_read(MANIFEST))

    def test_it_names_the_framework_and_its_schema_sections(self):
        self.assertEqual('pyrai3md', self.manifest['name'])
        sections = self.manifest['input_schema']['sections']
        for required in ('control', 'molecule', 'md', 'optimize',
                         'hessian', 'search', 'adaptive', 'constraint'):
            self.assertIn(required, sections)

    def test_every_code_path_it_names_resolves(self):
        for group in ('entry_points', 'pipeline', 'workflows'):
            for path in _walk_paths(self.manifest[group]):
                with self.subTest(path=path):
                    self.assertIsNotNone(_resolve(path))

    def test_every_keyword_is_accepted_by_its_model_or_table(self):
        from pyrai3md.schema import adapter, config

        tables = {
            'CONTROL_TYPES': adapter.CONTROL_TYPES,
            'MOLECULE_TYPES': adapter.MOLECULE_TYPES,
            'MD_TYPES': adapter.MD_TYPES,
            'OPTIMIZE_TYPES': adapter.OPTIMIZE_TYPES,
            'HESSIAN_TYPES': adapter.HESSIAN_TYPES,
            'SEARCH_TYPES': adapter.SEARCH_TYPES,
            'ADAPTIVE_TYPES': adapter.ADAPTIVE_TYPES,
            'COMPOSITE_TYPES': adapter.COMPOSITE_TYPES,
        }
        for section, spec in self.manifest['input_schema']['sections'].items():
            model_name = spec['model'].rsplit('.', 1)[-1]
            model = getattr(config, model_name)
            table_name = (spec.get('types') or '').rsplit('.', 1)[-1]
            table = tables.get(table_name, {})
            for keyword in spec['keywords']:
                with self.subTest(section=section, keyword=keyword):
                    self.assertTrue(
                        keyword in model.model_fields or keyword in table,
                        '%s.%s is listed in the manifest but neither the '
                        '%s model nor %s accepts it'
                        % (section, keyword, model_name, table_name or '-'))

    def test_retired_method_spellings_route_to_the_new_queues(self):
        aliases = self.manifest['input_schema']['keyword_aliases']
        self.assertEqual('theory', aliases['qm'])
        self.assertEqual('reference', aliases['abinit'])
        control = self.manifest['input_schema']['sections']['control']
        self.assertIn('theory', control['keywords'])
        self.assertIn('reference', control['keywords'])

    def test_capabilities_match_the_planning_vocabulary(self):
        from pyrai3md.planning import CAPABILITY_NAMES

        self.assertEqual(set(CAPABILITY_NAMES),
                         set(self.manifest['capabilities']))


if __name__ == '__main__':
    unittest.main()
