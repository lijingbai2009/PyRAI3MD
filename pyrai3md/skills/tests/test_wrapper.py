"""
======================================================================
Unit tests for the pyrai3md_skill wrapper config parsing.

The wrapper resolves the skill agent module from the input config.
Flat skill configs carry 'skill_name' at the top level; older
sectioned configs nest it in the 'environment' section. Both
layouts must resolve so standalone skill runs stay compatible
with pre-#141 config files.

Aug 18 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import sys

import pytest

from pyrai3md.skills.wrapper import resolve_skill_name


def test_flat_config_top_level_skill_name():
    config = {'skill_name': 'xtb_skill', 'method': 'gfn0'}
    assert resolve_skill_name(config) == 'xtb_skill'


def test_sectioned_config_environment_skill_name():
    config = {
        'system': {'atoms': ['C']},
        'calculation': {'method': 'gfn0'},
        'environment': {'skill_name': 'xtb_skill'},
    }
    assert resolve_skill_name(config) == 'xtb_skill'


def test_top_level_wins_over_environment():
    config = {
        'skill_name': 'orca_skill',
        'environment': {'skill_name': 'xtb_skill'},
    }
    assert resolve_skill_name(config) == 'orca_skill'


def test_missing_skill_name_returns_empty():
    assert resolve_skill_name({'method': 'gfn0'}) == ''
    assert resolve_skill_name({'environment': {}}) == ''
    assert resolve_skill_name({'environment': 'backend'}) == ''


class TestAgentCommandLine:
    """The agent receives the config path and nothing else (#301).

    execution_mode is a skill keyword the skill reads out of its own
    config. A second positional on the command line could therefore
    only ever be a value nothing branches on, and it was logged as
    'Mode:' with a wrapper default ('config') that disagreed with the
    agent default ('local').
    """

    @staticmethod
    def _capture(monkeypatch, tmp_path):
        import json

        from pyrai3md.skills import wrapper

        config_file = tmp_path / 'case.config'
        config_file.write_text(json.dumps({'skill_name': 'xtb_skill'}),
                               encoding='utf-8')
        captured = {}

        class _Completed:
            returncode = 0

        def fake_run(command, **kwargs):
            captured['command'] = command
            return _Completed()

        monkeypatch.setattr(wrapper.subprocess, 'run', fake_run)
        monkeypatch.setattr(sys, 'argv',
                            ['pyrai3md_skill', str(config_file)])
        with pytest.raises(SystemExit):
            wrapper.main()
        return captured['command'], str(config_file)

    def test_command_is_module_and_config_only(self, monkeypatch, tmp_path):
        command, config_file = self._capture(monkeypatch, tmp_path)
        assert command == [sys.executable, '-m',
                           'pyrai3md.skills.xtb_skill.agent', config_file]

    def test_command_carries_no_execution_mode(self, monkeypatch, tmp_path):
        command, _ = self._capture(monkeypatch, tmp_path)
        for mode in ('local', 'slurm', 'config'):
            assert mode not in command
        assert 'execution_mode' not in command
