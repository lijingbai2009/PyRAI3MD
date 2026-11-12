"""
======================================================================
Unit tests for how a calculation directory is named across the skills.

Two runs of one skill that share a directory cannot attribute a
fixed-name output file: BAGEL writes `ENERGY.out`, `FORCE_*.out` and
`NACME_*_*.out` under names that carry no run identity, so whichever
job wrote last is what the next job reads and reports as its own. A
round measured this directly - thirteen bagel cases across three
carriers all ran inside one `tmp_BAGEL`, and the file found afterwards
had been written eight minutes after the job that read it.

The separation is the `project` keyword every skill already carries:
`build_calcdir` replaces the engine token of the directory name with
the project and keeps every other part of the name. These tests walk
all skills, because a skill that does not take the substitution is a
skill whose fixed-name files stay unattributable.

Oct 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import importlib
import inspect
import os
import unittest


def _repo_root():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _skills_dir():
    return os.path.join(_repo_root(), 'pyrai3md', 'skills')


def _skill_names():
    """Every packaged skill directory, read off the tree.

    Not a list kept here: a skill added later is checked without anyone
    remembering to add it, which is the only way this convention stays
    true for the skills that exist.
    """
    root = _skills_dir()
    return sorted(name for name in os.listdir(root)
                  if name.endswith('_skill')
                  and os.path.isfile(os.path.join(root, name, 'service',
                                                  'utils.py')))


def _builder(skill_name):
    module = importlib.import_module(
        'pyrai3md.skills.%s.service.utils' % skill_name)
    return module.build_calcdir


def _default(skill_name, argument):
    return inspect.signature(_builder(skill_name)).parameters[
        argument].default


def _schema_default(skill_name, key):
    """The value the skill's own schema fills in when nothing is declared."""
    module = importlib.import_module(
        'pyrai3md.skills.%s.schema.config' % skill_name)
    return module.DEFAULT[key]


class TestTheProjectReplacesTheEngineToken(unittest.TestCase):
    """The naming rule, on every skill's own builder."""

    def setUp(self):
        self.skills = _skill_names()
        self.assertTrue(self.skills, 'no skill directories discovered')

    def test_every_skill_builder_takes_a_project(self):
        for skill in self.skills:
            spec = inspect.signature(_builder(skill))
            self.assertIn('project', spec.parameters,
                          msg='%s cannot separate runs by project' % skill)
            self.assertEqual(spec.parameters['project'].default, '',
                             msg='%s should keep the old naming when no '
                                 'project is declared' % skill)

    def test_an_empty_project_keeps_the_directory_name_unchanged(self):
        # A run that never mentions a project is not silently renamed.
        for skill in self.skills:
            builder = _builder(skill)
            base = _default(skill, 'base_name')
            self.assertEqual(builder(base_path='/job'),
                             os.path.join('/job', base), msg=skill)
            self.assertEqual(builder(execution_id=base, base_path='/job',
                                     project=''),
                             os.path.join('/job', base), msg=skill)

    def test_the_schema_default_project_is_not_a_declaration(self):
        # `project` is a schema field with a default, so "not declared"
        # arrives here as that default string. Treating it as a declaration
        # would rename every run that never asked for a project, and would
        # take the framework's namespace away from the tools that read the
        # engine out of the directory name.
        for skill in self.skills:
            builder = _builder(skill)
            base = _default(skill, 'base_name')
            declared_default = _schema_default(skill, 'project')
            self.assertEqual(builder(base_path='/job',
                                     project=declared_default),
                             os.path.join('/job', base),
                             msg='%s: default project %r renamed the directory'
                                 % (skill, declared_default))
            self.assertEqual(
                builder(execution_id='tmp_qm_%s' % base[len('tmp_'):],
                        base_path='/job', project=declared_default),
                os.path.join('/job', 'tmp_qm_%s' % base[len('tmp_'):]),
                msg=skill)

    def test_a_project_replaces_only_the_engine_token(self):
        for skill in self.skills:
            builder = _builder(skill)
            token = _default(skill, 'base_name')[len('tmp_'):]
            self.assertEqual(builder(base_path='/job', project='case_one_test'),
                             os.path.join('/job', 'tmp_case_one_test'),
                             msg=skill)
            # A framework namespace keeps its region, role and runtime index:
            # the substitution is one token, not a rebuild of the name. The
            # token arrives in the framework's own case (`PySCF` against the
            # base name's `PYSCF`) and can span two segments (`NN_PES`), both
            # of which have to be found or the substitution silently does
            # nothing.
            self.assertEqual(
                builder(execution_id='tmp_qm_layer0_%s_3' % token,
                        base_path='/job', project='case_one_test'),
                os.path.join('/job', 'tmp_qm_layer0_case_one_test_3'),
                msg=skill)
            self.assertEqual(
                builder(execution_id='tmp_qm_%s' % token.capitalize(),
                        base_path='/job', project='case_one_test'),
                os.path.join('/job', 'tmp_qm_case_one_test'),
                msg='%s: the framework spelling of the token was not replaced'
                    % skill)

    def test_two_terms_of_one_project_do_not_share_a_directory(self):
        # The other direction of the same rule: the substitution must not
        # consume the part of the namespace that distinguishes the terms.
        # Measured first as a regression - an earlier revision rebuilt the
        # name from the project alone and collapsed `tmp_qm_<SKILL>` and
        # `tmp_qm2_<SKILL>` into one directory.
        for skill in self.skills:
            builder = _builder(skill)
            token = _default(skill, 'base_name')[len('tmp_'):]
            first = builder(execution_id='tmp_qm_%s' % token,
                            base_path='/job', project='h2o_test')
            second = builder(execution_id='tmp_qm2_%s' % token,
                             base_path='/job', project='h2o_test')
            self.assertNotEqual(first, second, msg=skill)
            self.assertEqual(
                [os.path.basename(first), os.path.basename(second)],
                ['tmp_qm_h2o_test', 'tmp_qm2_h2o_test'], msg=skill)

    def test_two_projects_of_one_skill_do_not_share_a_directory(self):
        # The defect this exists for: two cases of one skill, one directory,
        # and a fixed-name file that belongs to neither of them.
        for skill in self.skills:
            builder = _builder(skill)
            self.assertNotEqual(builder(base_path='/job', project='case_a_test'),
                                builder(base_path='/job', project='case_b_test'),
                                msg=skill)

    def test_a_project_that_is_not_one_path_segment_is_refused(self):
        # The project becomes a directory name under the scratch path, so a
        # value carrying a separator would write outside it.
        for skill in self.skills:
            builder = _builder(skill)
            for value in ('a/b', '..', 'a/../b'):
                with self.assertRaises(ValueError,
                                       msg='%s accepted %r' % (skill, value)):
                    builder(base_path='/job', project=value)


class TestEveryBackendFeedsItItsProject(unittest.TestCase):
    """The wiring, read off each backend's source.

    The builder is only half of it: a backend that calls it without a
    project gets the shared name back, and nothing else in the package
    would notice.
    """

    def _calcdir_call(self, skill_name):
        path = os.path.join(_skills_dir(), skill_name, 'backend.py')
        with open(path, encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, 'id', '') == 'build_calcdir']
        self.assertEqual(len(calls), 1,
                         msg='%s: expected one build_calcdir call, found %d'
                             % (skill_name, len(calls)))
        return calls[0]

    def test_the_project_is_passed_at_the_call_site(self):
        for skill in _skill_names():
            keywords = {keyword.arg
                        for keyword in self._calcdir_call(skill).keywords}
            self.assertIn('project', keywords,
                          msg='%s names its directory without the project'
                              % skill)

    def test_the_project_passed_is_the_configured_one(self):
        # Not a literal and not the skill label: the directory follows the
        # project the caller declared, which is what makes it per run.
        for skill in _skill_names():
            value = next(keyword.value
                         for keyword in self._calcdir_call(skill).keywords
                         if keyword.arg == 'project')
            self.assertEqual(ast.unparse(value),
                             'self.config.calculation.project',
                             msg='%s reads its directory name from elsewhere'
                                 % skill)


class TestThePlannedProjectReachesTheDirectory(unittest.TestCase):
    """The chain from a planned task's project to the directory name.

    Qualification writes one `<label>_project` per test case, the skill's
    adapter mirrors it onto `calculation.project`, and that value is what
    names the directory. Run on two shipped case names of one skill, so the
    claim is about the path a round takes and not about the helper alone.
    """

    def test_two_shipped_cases_land_in_two_directories(self):
        import yaml
        from pyrai3md.skills.bagel_skill.schema.adapter import build_config
        from pyrai3md.skills.bagel_skill.service.utils import build_calcdir

        root = os.path.join(_skills_dir(), 'bagel_skill', 'tests')
        directories = set()
        for case in ('casscf_frequency.yaml', 'casscf_gradient.yaml'):
            with open(os.path.join(root, case)) as handle:
                project = yaml.safe_load(handle)['name']
            config = build_config({
                'profile': 'casscf_frequency', 'method': 'casscf',
                'basis': 'cc-pvdz', 'execution_mode': 'local',
                'atoms': ['O', 'H', 'H'],
                'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                [0.0, 1.0, 1.0]],
                'bagel_project': project,
            })
            self.assertEqual(config.calculation.project, project,
                             msg='%s: the planned project never reached the '
                                 'schema' % case)
            directories.add(os.path.basename(build_calcdir(
                base_path='/job', project=config.calculation.project)))
        self.assertEqual(directories,
                         {'tmp_casscf_frequency_test',
                          'tmp_casscf_gradient_test'})


if __name__ == '__main__':
    unittest.main()
