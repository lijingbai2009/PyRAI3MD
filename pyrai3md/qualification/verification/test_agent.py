"""
======================================================================
Test suite for the Agent entry point (agent.py) and the
interface.loader module.

Tests cover:
- Loader: JSON config loading and normalization
- Loader: error handling
- PyRAI3Agent: construction from dict and file
- PyRAI3Agent: pipeline integration (adapter -> normalize -> validate)
- Loader-parser consistency

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Inline mock input for agent tests (no classic file dependency)
MOCK_AGENT_INPUT = """\
&control
title agent_test
jobtype hybrid
qm nn

&molecule
ci 2 2
spin 0 1
natom 4

&md
step 2
sfhp gsh
"""


class TestLoaderNormalize(unittest.TestCase):
    """Test interface.loader normalize function."""

    def test_lowercase_keys(self):
        from pyrai3md.interface.loader import normalize
        config = {'Control': {'title': 'test'}, 'MD': {'step': '10'}}
        result = normalize(config)
        self.assertIn('control', result)
        self.assertIn('md', result)

    def test_empty_dict(self):
        from pyrai3md.interface.loader import normalize
        self.assertEqual(normalize({}), {})


class TestLoaderLoadConfig(unittest.TestCase):
    """Test interface.loader load_config function."""

    def test_load_json_file(self):
        from pyrai3md.interface.loader import load_config

        data = {'control': {'title': 'json_test', 'jobtype': 'sp'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            tmpfile = f.name

        try:
            result = load_config(tmpfile)
            self.assertEqual(result['control']['title'], 'json_test')
        finally:
            os.unlink(tmpfile)

    def test_file_not_found(self):
        from pyrai3md.interface.loader import load_config
        with self.assertRaises(FileNotFoundError):
            load_config('nonexistent_config.json')

    def test_invalid_json(self):
        from pyrai3md.interface.loader import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            tmpfile = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                load_config(tmpfile)
        finally:
            os.unlink(tmpfile)


class TestPyRAI3AgentConstruction(unittest.TestCase):
    """Test PyRAI3Agent class construction."""

    def test_init_from_dict(self):
        from pyrai3md.agent import PyRAI3Agent

        config = {'control': {'title': 'test', 'jobtype': 'sp'}}
        agent = PyRAI3Agent.from_dict(config)
        self.assertIs(agent.raw_config, config)
        self.assertIsNone(agent.config)
        self.assertIsNone(agent.state)
        self.assertIsNone(agent.skills)

    def test_from_file(self):
        from pyrai3md.agent import PyRAI3Agent

        data = {'control': {'title': 'file_test', 'jobtype': 'sp'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            tmpfile = f.name

        try:
            agent = PyRAI3Agent.from_file(tmpfile)
            self.assertIn('control', agent.raw_config)
        finally:
            os.unlink(tmpfile)

    def test_from_file_not_found(self):
        from pyrai3md.agent import PyRAI3Agent
        with self.assertRaises(FileNotFoundError):
            PyRAI3Agent.from_file('nonexistent.json')


class TestPyRAI3AgentPipeline(unittest.TestCase):
    """Test PyRAI3Agent pipeline integration."""

    def test_agent_uses_adapter(self):
        """Test that agent pipeline uses build_config from adapter."""
        from pyrai3md.agent import PyRAI3Agent
        from pyrai3md.schema.adapter import build_config

        raw = {'control': {'title': 'pipeline_test', 'jobtype': 'hybrid', 'qm': 'nn'}}
        agent = PyRAI3Agent.from_dict(raw)

        # Manually run adapter step
        config = build_config(agent.raw_config)
        self.assertEqual(config.title, 'pipeline_test')
        self.assertEqual(config.jobtype, 'hybrid')

    def test_agent_uses_maestro(self):
        """Test that agent uses Maestro for skill resolution."""
        from pyrai3md.kernel.maestro import Maestro
        maestro = Maestro()
        self.assertTrue(hasattr(maestro, 'resolve_skill'))
        self.assertTrue(hasattr(maestro, 'resolve_skills'))

    def test_agent_with_parsed_input(self):
        """Test agent construction with parser output."""
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.agent import PyRAI3Agent

        raw = _parse_text(MOCK_AGENT_INPUT.splitlines(keepends=True))
        agent = PyRAI3Agent.from_dict(raw)
        self.assertIs(agent.raw_config, raw)
        self.assertEqual(raw['control']['title'], 'agent_test')


class TestLoaderParserConsistency(unittest.TestCase):
    """Test that loader and parser produce consistent output formats."""

    def test_both_produce_dicts(self):
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.interface.loader import normalize

        text = "&CONTROL\ntitle test\njobtype sp\n"
        parser_result = _parse_text(text.splitlines(keepends=True))
        self.assertIsInstance(parser_result, dict)

        loader_result = normalize({'Control': {'title': 'test'}})
        self.assertIsInstance(loader_result, dict)

    def test_both_have_lowercase_sections(self):
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.interface.loader import normalize

        text = "&Control\ntitle test\n"
        parser_result = _parse_text(text.splitlines(keepends=True))
        self.assertIn('control', parser_result)

        loader_result = normalize({'Control': {'title': 'test'}})
        self.assertIn('control', loader_result)


class TestPartitionFieldsMergesNestedSection(unittest.TestCase):
    """_partition_fields merges a nested framework section into its
    submodel instead of replacing it, so top-level structure fields
    routed to the same submodel survive.

    Regression: the example YAML layout carries atoms/coordinates at
    the top level alongside a nested molecule section (both map to the
    system submodel). Replacing the system dict with the molecule
    section dropped the structure, leaving natom=0, which broke every
    region/composite run at validate_region ('Region index 0 out of
    range [0, 0)')."""

    def test_top_level_structure_survives_nested_molecule(self):
        from pyrai3md.schema.adapter import _partition_fields
        flat = {
            'title': 'x',
            'atoms': ['C', 'C'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            'molecule': {'charges': 0, 'ci': [1], 'spin': [0]},
        }
        sub = _partition_fields(flat)
        self.assertEqual(sub['system']['atoms'], ['C', 'C'])
        self.assertEqual(len(sub['system']['coordinates']), 2)
        # The nested molecule fields still land in the system submodel.
        self.assertEqual(sub['system']['charges'], 0)
        self.assertEqual(sub['system']['ci'], [1])
        self.assertEqual(sub['control']['title'], 'x')

    def test_build_config_keeps_toplevel_structure_with_molecule(self):
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        raw = {
            'control': {'title': 'x', 'jobtype': 'energy',
                        'theory': ['pyscf']},
            'atoms': ['C', 'C'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            'molecule': {'charges': 0, 'multiplicity': 1,
                         'ci': [1], 'spin': [0]},
        }
        cfg = normalize(build_config(raw))
        self.assertEqual(len(cfg.system.atoms), 2)
        self.assertEqual(cfg.system.natom, 2)
        self.assertEqual(cfg.system.charges, 0)


if __name__ == '__main__':
    unittest.main()


class TestTopLevelDispatchRouting(unittest.TestCase):
    """A top-level dispatch keyword follows the jobtype's section.

    Agent configs flatten the sections; the keyword itself does not say
    which workflow it configures. The jobtype does. `optimize` declares
    only `multipool`, so a top-level `multitask` there is the
    molecule-scope keyword instead.
    """

    @staticmethod
    def _config(jobtype, **extra):
        from pyrai3md.schema.adapter import build_config
        raw = {
            'control': {'title': 'dispatch_route', 'jobtype': jobtype},
            'molecule': {
                'atoms': ['H', 'H'],
                'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                'charges': 0,
                'multiplicity': 1,
            },
        }
        raw.update(extra)
        return build_config(raw)

    def test_dispatch_key_follows_jobtype(self):
        cases = (
            ('md', 'dynamics'),
            ('adaptive', 'adaptive'),
            ('frequency', 'hessian'),
            ('search', 'search'),
            ('train', 'train'),
            ('prediction', 'prediction'),
            ('energy', 'system'),
        )
        for jobtype, section in cases:
            with self.subTest(jobtype=jobtype):
                config = self._config(jobtype, multitask=True, multipool=3)
                submodel = getattr(config, section)
                self.assertTrue(submodel.multitask)
                self.assertEqual(submodel.multipool, 3)

    def test_optimize_keeps_multipool_and_sends_multitask_to_molecule(self):
        config = self._config('optimize', multitask=True, multipool=3)
        self.assertTrue(config.system.multitask)
        self.assertEqual(config.optimize.multipool, 3)
        self.assertFalse(hasattr(config.optimize, 'multitask'))

class TestAgentCliStatus(unittest.TestCase):
    """The agent CLI writes a run status; the library entry does not."""

    def test_run_cli_goes_through_the_run_logger(self):
        from pyrai3md.agent import PyRAI3Agent
        agent = PyRAI3Agent({})
        agent.state = object()
        with mock.patch('pyrai3md.pmd.run_with_logger',
                        return_value='result') as logger:
            self.assertEqual(agent.run_cli(), 'result')
        logger.assert_called_once_with(agent.state)

    def test_run_stays_the_programmatic_entry(self):
        from pyrai3md.agent import PyRAI3Agent
        agent = PyRAI3Agent({})
        agent.state = object()
        with mock.patch('pyrai3md.workflow.execute',
                        return_value='result') as execute:
            self.assertEqual(agent.run(), 'result')
        execute.assert_called_once_with(agent.state)

