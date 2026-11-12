"""
======================================================================
Unit tests for skill resource packaging in setup.py.

Skill discovery reads manifest.yaml from the installed package, so
every skill directory carrying a manifest must be declared in
setup.py package_data (manifest.yaml, skill_guide.md, tests/*.yaml).
A missing entry ships the skill without its manifest and the remote
install reports 'skill_name not available' / 'method not found in
registry' even though the source tree works.

Aug 17 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import os
import unittest


def _repo_root():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


# The spelling the documentation uses for a count that small.
_NUMBER_WORDS = (
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
)


def _setup_path():
    return os.path.join(_repo_root(), 'setup.py')


def _load_package_data():
    """Extract the package_data dict from setup.py via AST."""
    with open(_setup_path(), 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'package_data':
                return ast.literal_eval(keyword.value)
    raise AssertionError('setup.py defines no package_data')


# setup.py only exists in the source tree, not in installed copies.
_HAS_SOURCE_TREE = os.path.exists(_setup_path())


@unittest.skipIf(not _HAS_SOURCE_TREE,
                 'setup.py is only shipped with the source tree')
class TestSkillPackageData(unittest.TestCase):
    """Every skill with a manifest.yaml declares its resources."""

    def setUp(self):
        self.skills_dir = os.path.join(
            _repo_root(), 'pyrai3md', 'skills')
        self.package_data = _load_package_data()

    def _skill_packages_with_manifest(self):
        packages = []
        for name in sorted(os.listdir(self.skills_dir)):
            skill_dir = os.path.join(self.skills_dir, name)
            manifest = os.path.join(skill_dir, 'manifest.yaml')
            if os.path.isdir(skill_dir) and os.path.exists(manifest):
                packages.append(f'pyrai3md.skills.{name}')
        return packages

    def test_all_skills_declare_manifest(self):
        for package in self._skill_packages_with_manifest():
            with self.subTest(package=package):
                self.assertIn(
                    package, self.package_data,
                    f'{package} missing from setup.py package_data; '
                    f'the installed skill would lose manifest.yaml '
                    f'and fail discovery remotely')
                self.assertIn(
                    'manifest.yaml', self.package_data[package])

    def test_skill_resources_cover_guide_and_tests(self):
        for package in self._skill_packages_with_manifest():
            with self.subTest(package=package):
                resources = self.package_data.get(package, [])
                self.assertIn('skill_guide.md', resources)
                self.assertIn('tests/*.yaml', resources)

    def test_framework_guide_is_packaged(self):
        resources = self.package_data.get('pyrai3md', [])
        self.assertIn('skill_guide.md', resources)
        self.assertTrue(os.path.exists(os.path.join(
            _repo_root(), 'pyrai3md', 'skill_guide.md')),
            'the framework guide is named in package_data but missing on disk')


@unittest.skipIf(not _HAS_SOURCE_TREE,
                 'setup.py is only shipped with the source tree')
class TestExamplePackageData(unittest.TestCase):
    """The LAMMPS topology fixtures shipped inside examples/ are declared in
    package_data, so an installed package can run the examples that read them
    (the `md_lammps_*` box, the `multicenter_pbc_*` single water) directly from
    the examples directory."""

    def setUp(self):
        self.examples_dir = os.path.join(
            _repo_root(), 'pyrai3md', 'examples')
        self.package_data = _load_package_data()

    def test_every_shipped_example_sidecar_is_declared(self):
        # Any file beside an example YAML that is not itself an example is
        # a fixture an input names by bare name -- a topology, an initial
        # geometry, a label file. setup.py must ship it, because an
        # installed package that lost it loses the example that reads it,
        # and the loss shows up only on someone else's machine. This
        # checks the rule instead of a list, so the next fixture cannot be
        # added to the directory and forgotten in the packaging.
        import fnmatch

        # The declarations already carry the 'examples/' prefix, so the
        # file name is matched as it is spelled in setup.py.
        globs = list(self.package_data.get('pyrai3md', []))
        for fname in sorted(os.listdir(self.examples_dir)):
            path = os.path.join(self.examples_dir, fname)
            if not os.path.isfile(path):
                continue
            if fname.endswith(('.yaml', '.yml')):
                continue
            with self.subTest(fixture=fname):
                self.assertTrue(
                    any(fnmatch.fnmatch('examples/' + fname, pattern)
                        for pattern in globs),
                    'examples/%s is shipped but no package_data pattern '
                    'covers it' % fname)

    def test_the_label_fixture_for_the_machine_learning_examples(self):
        # nn_train / nn_predict / nn_grid_search name ./egs_small.json,
        # resolved in the run directory (the qualification planner stages
        # it; a direct run reads it beside the YAML).
        resources = self.package_data.get('pyrai3md', [])
        self.assertTrue(
            os.path.exists(os.path.join(self.examples_dir, 'egs_small.json')))
        self.assertIn('examples/*.json', resources)

    def test_lammps_fixtures_present_and_declared(self):
        resources = self.package_data.get('pyrai3md', [])
        for fname in ('h2o.data', 'h2o.in.init', 'h2o.in.settings'):
            with self.subTest(fixture=fname):
                self.assertTrue(
                    os.path.exists(os.path.join(self.examples_dir, fname)),
                    f'{fname} missing from pyrai3md/examples/')
                self.assertIn(
                    f'examples/{fname}', resources,
                    f'examples/{fname} missing from setup.py '
                    f'package_data; the installed examples directory '
                    f'would lose the LAMMPS topology fixtures')


class TestDocumentationIndex(unittest.TestCase):
    """ARCHITECTURE.md's document index must cover every document on disk.

    The index drifted to 27 of 42 files while the docs themselves grew, and
    nothing noticed: a document missing from the only inventory is invisible
    to the next reader, which is how a stale section starts getting copied into
    code. The gate reads the count the tree states and recounts the disk.
    """

    def test_every_document_is_listed_and_the_count_is_true(self):
        root = _repo_root()
        docs = os.path.join(root, 'dev')
        on_disk = sorted(name for base, _dirs, files in os.walk(docs)
                         for name in files if name.endswith('.md'))
        self.assertGreater(len(on_disk), 30,
                           'found only %d documents; the walk is wrong'
                           % len(on_disk))
        with open(os.path.join(docs, 'ARCHITECTURE.md'), 'r',
                  encoding='utf-8') as handler:
            index = handler.read()
        missing = [name for name in on_disk if name not in index]
        self.assertEqual(missing, [],
                         'documents absent from the ARCHITECTURE.md index: %s'
                         % missing)
        stated = index.count('Design documentation: ') and int(
            index.split('Design documentation: ')[1].split(' files')[0])
        self.assertEqual(stated, len(on_disk),
                         'the index states %d documents, disk holds %d'
                         % (stated, len(on_disk)))


class TestOptimizeCellDocMatchesTheCode(unittest.TestCase):
    """The cell-optimization prose has to say what the code does.

    ARCHITECTURE.md kept "`optimize_cell` is a routing flag, not cell degrees
    of freedom ... no geometry optimizer in `workflow/optimizer/` moves the
    box" long after the optimizer grew that half, and both of its
    cross-references had drifted onto unrelated lines; the shipped-example
    count said twelve while sixteen files sat on disk (it lived in
    `OPTIMIZE_DEVELOPMENT.md` then and lives in `CELL_OPT_DEVELOPMENT.md` now).
    A paragraph like that is what the next reader copies into code, so the
    claims are recounted here rather than trusted.
    """

    def _read(self, *parts):
        with open(os.path.join(_repo_root(), *parts), 'r',
                  encoding='utf-8') as handler:
            return handler.read()

    def test_the_routing_flag_claim_is_gone_and_points_where_it_lives(self):
        # The optimizer really does carry the six movable lattice components
        # (`workflow/optimizer/base.py::_extend_to_cell`) and the schema field
        # says so (`OptimizeConfig.optimize_cell`), so the claim that it does
        # not is the stale half.
        text = self._read('dev', 'ARCHITECTURE.md')
        self.assertNotIn('not cell degrees of freedom', text)
        self.assertNotIn('no geometry optimizer in', text)
        start = text.index('`optimize_cell` says')
        paragraph = text[start:start + 900]
        self.assertIn('CELL_OPT_DEVELOPMENT.md', paragraph,
                      'the cell paragraph must point at the document that '
                      'describes the implemented cell half')

    def test_the_shipped_cell_example_count_is_recounted(self):
        import glob
        import re
        examples = os.path.join(_repo_root(), 'pyrai3md', 'examples')
        on_disk = len(glob.glob(os.path.join(examples,
                                             'optimize_cell_*.yaml')))
        self.assertGreater(on_disk, 10, 'the glob is wrong: %d' % on_disk)
        sentence = re.search(r'since #355, by (\w+) shipped examples',
                             self._read('dev', 'methods',
                                        'CELL_OPT_DEVELOPMENT.md'))
        self.assertIsNotNone(sentence, 'the count sentence changed shape')
        stated = _NUMBER_WORDS.index(sentence.group(1))
        self.assertEqual(stated, on_disk,
                         'the section states %s cell examples, disk holds %d'
                         % (sentence.group(1), on_disk))
