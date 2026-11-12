"""
======================================================================
Test suite for the machine-learning examples and the carriers they are
exercised through:
- pyrai3md/examples/nn_train.yaml, nn_predict.yaml, nn_grid_search.yaml,
  nn_grid_search_local.yaml
- qualification/certification/converter.py (framework workflow sections)

Tests cover:
- the shipped ML examples reach one plan through the yaml and the
  classic text carriers, which is the pair qualification runs them
  through; the scan's `grid` must survive the conversion, because a
  dropped section would leave a classic scan planning no points at all
- the two scan arms: one grid, two dispatches, and the `slurm:` section
  the remote arm states reaching the framework snapshot from either
  carrier
- &SEARCH / &TRAIN / &PREDICTION rendering, and the refusals that keep a
  section from being silently dropped or invented
- a machine-learning jobtype emits no default &MD block: it propagates
  nothing, and a parsed dynamics section nobody declared is a claim about
  the run

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

_EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, 'examples')


def _example(name):
    return os.path.abspath(os.path.join(_EXAMPLES, '%s.yaml' % name))


def _pipeline(raw):
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate

    return validate(normalize(build_config(raw)))


class TestShippedMlExamples(unittest.TestCase):
    """The shipped examples state one plan in each of the two carriers."""

    def test_the_examples_ship_and_are_qualified(self):
        from pyrai3md.qualification.manager import POSTPONED_EXAMPLES

        for name in ('nn_train', 'nn_predict', 'nn_grid_search',
                     'nn_grid_search_local'):
            with self.subTest(example=name):
                self.assertTrue(os.path.isfile(_example(name)))
                self.assertNotIn(name, POSTPONED_EXAMPLES)

    def test_the_label_fixture_ships_beside_them(self):
        # The examples name `./egs_small.json`, resolved in the run
        # directory; the file is shipped in examples/ the way the LAMMPS
        # topologies are, and setup.py must declare it.
        self.assertTrue(os.path.isfile(os.path.join(_EXAMPLES,
                                                    'egs_small.json')))

    def test_yaml_carrier_reaches_the_plan(self):
        from pyrai3md.pmd import parse_input

        config = _pipeline(parse_input(_example('nn_grid_search')))
        self.assertEqual({'depth': [1, 2], 'nn_size': [8, 16]},
                         config.search.grid)
        self.assertEqual('./grid-search', config.search.grid_root)

    def test_classic_carrier_reaches_the_same_plan(self):
        from pyrai3md.qualification.certification.converter import convert
        from pyrai3md.pmd import parse_input

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, 'case.json')
            input_path = os.path.join(tmp, 'nn_grid_search.input')
            _convert(_example('nn_grid_search'), json_path, input_path)
            rendered = parse_input(input_path)
            classic = _pipeline(rendered)
        yaml_config = _pipeline(parse_input(_example('nn_grid_search')))
        self.assertEqual(yaml_config.search.grid, classic.search.grid)
        self.assertEqual(yaml_config.search.sort_stat,
                         classic.search.sort_stat)

    def test_the_scan_arms_ask_for_the_dispatch_they_are_named_for(self):
        # Two arms of one scan: the same grid, the same ranking, and the
        # difference is only how the points are run. Read back from the
        # config rather than from the YAML text, because what the run acts
        # on is what the plan resolved.
        from pyrai3md.pmd import parse_input

        remote = _pipeline(parse_input(_example('nn_grid_search'))).search
        local = _pipeline(parse_input(_example('nn_grid_search_local'))).search
        self.assertEqual(remote.grid, local.grid)
        self.assertEqual(remote.sort_by, local.sort_by)
        self.assertEqual(remote.max_grid, local.max_grid)
        self.assertEqual((remote.multitask, remote.remotetask), (True, True))
        self.assertEqual((local.multitask, local.remotetask), (True, None))
        self.assertEqual((remote.multipool, local.multipool), (4, 2))

    def test_the_slurm_directives_arrive_through_both_carriers(self):
        # The scan example is the one shipped input that states a `slurm:`
        # section, and the remote arm is the one case a Slurm-free host
        # never plans into a run. What the YAML declares is therefore
        # checked against what the framework freezes onto the state, in
        # both carriers: the classic text spells `exclusive false` as a
        # word, and a reader that took any non-empty string for truth would
        # reserve a whole node per grid point.
        import yaml as _yaml

        from pyrai3md.kernel.environment import Environment
        from pyrai3md.pmd import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        with open(_example('nn_grid_search'), encoding='utf-8') as handle:
            declared = _yaml.safe_load(handle)['slurm']
        self.assertTrue(declared, 'the scan example states no slurm section')

        env = Environment()
        env.initialize()
        env.reset_input_overrides()
        host = env.get_slurm()
        # Non-vacuity: an example that declared nothing but what the host
        # already states would reach the same snapshot with the merge
        # removed, so the assertion below would say nothing either way.
        differing = [key for key, value in declared.items()
                     if str(host.get(key)) != str(value)]
        self.assertTrue(differing,
                        'the example repeats the host section exactly: %s'
                        % sorted(declared))

        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, 'case.json')
            input_path = os.path.join(tmp, 'nn_grid_search.input')
            _convert(_example('nn_grid_search'), json_path, input_path)
            os.chdir(tmp)
            try:
                snapshots = {
                    'yaml': build_config(
                        parse_input(_example('nn_grid_search'))),
                    'input': build_config(parse_input(input_path)),
                }
            finally:
                os.chdir(previous)

        for label in sorted(snapshots):
            slurm = validate(
                normalize(snapshots[label])).environment.slurm
            for key in sorted(declared):
                self.assertEqual(str(slurm.get(key)), str(declared[key]),
                                 '%s carrier: %s' % (label, key))
            self.assertIs(slurm.get('exclusive'), False, label)

    def test_the_training_and_prediction_sections_state_their_keywords(
            self):
        from pyrai3md.qualification.certification.converter import convert
        from pyrai3md.pmd import parse_input

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, 'train.json')
            input_path = os.path.join(tmp, 'nn_train.input')
            _convert(_example('nn_train'), json_path, input_path)
            convert(json_path, input_path)
            with open(input_path) as handle:
                text = handle.read()
        self.assertIn('&TRAIN', text)
        self.assertIn('multipool', text)
        # A machine-learning job propagates nothing: an &MD block in its
        # input would be a dynamics request the example never made.
        self.assertNotIn('&MD', text)


def _convert(yaml_path, json_path, input_path):
    """Run the example through the converter the qualification planner uses."""
    from pyrai3md.qualification.certification.converter import (
        convert, yaml_to_config)

    yaml_to_config(yaml_path, json_path)
    convert(json_path, input_path)
    return json_path


class TestWorkflowSectionConversion(unittest.TestCase):
    """&SEARCH and friends: rendered faithfully, or refused by name."""

    def setUp(self):
        from pyrai3md.qualification.certification.converter import (
            _build_workflow_section_pairs, _build_workflow_sections,
            _render_grid_value)

        self.pairs = _build_workflow_section_pairs
        self.sections = _build_workflow_sections
        self.grid = _render_grid_value

    def test_grid_renders_as_semicolon_groups(self):
        self.assertEqual('depth 1 2; nn_size 8 16',
                         self.grid({'nn_size': [8, 16], 'depth': [1, 2]}))

    def test_a_scalar_candidate_is_one_entry_list_on_readback(self):
        self.assertEqual('depth 2', self.grid({'depth': 2}))

    def test_booleans_are_spelled_as_words_not_python_values(self):
        # 'False' would read back as a two-character string in a schema
        # that expects a boolean, so the text carrier states the value the
        # way the reader spells it.
        self.assertEqual('use_dropout false', self.grid({'use_dropout': [False]}))
        self.assertEqual('use_dropout true', self.grid({'use_dropout': [True]}))

    def test_a_rendered_grid_reads_back_as_the_same_mapping(self):
        # The refusal-then-render rules only mean something if the runtime
        # reader accepts what the converter wrote: a candidate that renders
        # as text it cannot parse would plan nothing at run time.
        from pyrai3md.schema.adapter import _parse_search_grid

        for grid in ({'depth': [1, 2], 'nn_size': [8, 16]},
                     {'use_dropout': [False, True]},
                     {'dropout': [0.005]},
                     {'activ': ['leaky_softplus']}):
            with self.subTest(grid=grid):
                self.assertEqual(grid,
                                 _parse_search_grid(self.grid(grid)))

    def test_a_nested_candidate_is_refused_not_dropped(self):
        with self.assertRaises(ValueError) as caught:
            self.grid({'loss_weights': [[1, 1], [1, 10]]})
        self.assertIn('no text spelling', str(caught.exception))

    def test_an_empty_grid_is_refused(self):
        for value in ({}, 42, {'depth': []}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.grid(value)

    def test_an_unknown_keyword_in_a_section_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.pairs('search', {'search': {'nonsense': 1}})
        message = str(caught.exception)
        self.assertIn('nonsense', message)
        self.assertIn('grid', message, 'the refusal lists what is known')

    def test_a_section_is_rendered_only_when_the_config_carries_it(self):
        self.assertEqual([], self.sections({'theory': ['nn']}))
        blocks = self.sections({'search': {'grid': {'depth': [1, 2]},
                                           'multipool': 2}})
        self.assertEqual(1, len(blocks))
        self.assertIn('&SEARCH', blocks[0])
        self.assertIn('grid           depth 1 2', blocks[0])

    def test_the_dispatch_keys_survive_into_the_section_pairs(self):
        pairs = dict(self.pairs('train', {'train': {'multipool': 4,
                                                   'remotetask': True}}))
        self.assertEqual({'multipool': 4, 'remotetask': True}, pairs)


class TestMlJobtypesEmitNoDynamicsSection(unittest.TestCase):
    """`train`, `prediction` and `search` ask for no &MD block."""

    def _render(self, jobtype):
        from pyrai3md.qualification.certification.converter import convert

        config = {
            'title': 'ml_case', 'jobtype': jobtype, 'theory': ['nn'],
            'skill_name': 'nn_pes_skill',
            'atoms': ['C'], 'coordinates': [[0.0, 0.0, 0.0]],
            'nn': {'method': 'mlp_eg', 'modeldir': './nn'},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'case.json')
            import json
            with open(path, 'w') as handle:
                json.dump(config, handle)
            return convert(path)

    def test_ml_jobtypes_state_no_md_section(self):
        for jobtype in ('train', 'prediction', 'search'):
            with self.subTest(jobtype=jobtype):
                self.assertNotIn('&MD', self._render(jobtype))

    def test_a_dynamics_job_still_does(self):
        # The guard is about the machine-learning jobtypes, not about the
        # advanced-jobtype branch as a whole.
        self.assertIn('&MD', self._render('md'))


if __name__ == '__main__':
    unittest.main()
