"""
======================================================================
Unit tests for the anchor a skill's local run script is launched from.

Every skill that hands its engine a shell deck composes the script path and
the child's working directory in the same place. When the calculation directory
is relative -- which `environment.scratch_path` legitimately allows, since it
is honoured as written -- a relative script path is resolved by the child
*inside* the directory it names, and bash reports the deck the runner just
wrote as missing. Round tmp_20261017_180206 hit it on the classic-input carrier
of a multi-layer example: `bash: ./scratch/tmp_low_qm2_XTB/XTB.sh: No such file
or directory`, with the term dead before the engine started.

Aug 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import contextlib
import importlib
import os
from types import SimpleNamespace

import pytest


def _shell_skills():
    """Skills whose local runner launches a shell deck, read from source.

    The set is derived instead of listed so a new skill joins the
    parametrization by shipping its own runner; the rule is the one the
    criteria need (the runner builds a bash command), not a name.
    ``test_the_shell_launching_set_is_the_parametrized_one`` keeps the
    derivation honest.
    """
    skills_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    launching = []
    for name in sorted(os.listdir(skills_root)):
        path = os.path.join(skills_root, name, 'service', 'runner',
                            'local_runner.py')
        if not os.path.isfile(path):
            continue
        with open(path, 'r', encoding='utf-8') as handler:
            text = handler.read()
        if "'bash'" in text or '"bash"' in text:
            launching.append(name[:-len('_skill')])
    return launching


SHELL_SKILLS = tuple(_shell_skills())


@contextlib.contextmanager
def _in_directory(path):
    """chdir for the launch, restored on exit (a skill test may not import
    the framework's own `working_directory`)."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _Recorder:
    """Stand-in for the `subprocess` module of one runner."""

    def __init__(self):
        self.argv = None
        self.cwd = None

    def run(self, argv, capture_output=False, cwd=None, **kwargs):
        self.argv = list(argv)
        self.cwd = cwd
        return SimpleNamespace(returncode=0)


class _StubConfig:
    """Config stand-in: every project name a runner asks for is supplied.

    The launch path reads `project` and the skill's own `*_project`
    key; answering any attribute keeps the stub from naming one skill,
    which is how it used to drift when a skill renamed its project key.
    """

    project = 'PRJ'
    profile = 'ff_energy'

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return '%s_project' % name


def _launch(skill, calcdir, work_dir):
    """Run one skill's run_local() with bash stubbed out.

    Returns (recorder, deck_path_written). The engine script body is replaced by
    a stub because only the launch anchor is under test here.
    """
    module_name = 'pyrai3md.skills.%s_skill' % skill
    runner = importlib.import_module(
        '%s.service.runner.local_runner' % module_name)
    templates = importlib.import_module(
        '%s.service.templates.run_templates' % module_name)
    recorder = _Recorder()
    original_run = runner.subprocess
    original_script = templates.build_local_script
    with _in_directory(work_dir):
        try:
            runner.subprocess = recorder
            templates.build_local_script = (
                lambda *a, **k: '#!/bin/bash\nexit 0\n')
            code = runner.run_local(_StubConfig(), calcdir)
        finally:
            runner.subprocess = original_run
            templates.build_local_script = original_script
    assert code == 0
    script_arg = recorder.argv[-1]
    return recorder, os.path.normpath(
        os.path.join(work_dir, calcdir, os.path.basename(script_arg)))


def _resolved_from_the_child(recorder, work_dir):
    """The path bash actually opens, seen from the cwd the runner asked for.

    A relative `cwd` is read from the directory the launch started in
    (`work_dir`), which is what bash does; the test restores its own cwd
    after the launch, so it cannot lean on the runner's chdir.
    """
    script_arg = recorder.argv[-1]
    if os.path.isabs(script_arg):
        return script_arg
    base = recorder.cwd or work_dir
    if not os.path.isabs(base):
        base = os.path.join(work_dir, base)
    return os.path.normpath(os.path.join(base, script_arg))


@pytest.mark.parametrize('skill', SHELL_SKILLS)
def test_a_relative_calc_dir_still_finds_the_deck_it_wrote(skill, tmp_path):
    work_dir = str(tmp_path)
    calcdir = os.path.join('scratch', 'tmp_%s' % skill)
    os.makedirs(os.path.join(work_dir, calcdir))

    recorder, deck = _launch(skill, calcdir, work_dir)

    assert os.path.isfile(deck), 'the runner wrote its deck somewhere else'
    assert os.path.realpath(_resolved_from_the_child(recorder, work_dir)) == \
        os.path.realpath(deck), (
            'bash was handed %r with cwd=%r, which opens %r rather than the '
            'deck at %r' % (recorder.argv[-1], recorder.cwd,
                            _resolved_from_the_child(recorder, work_dir), deck))


@pytest.mark.parametrize('skill', SHELL_SKILLS)
def test_an_absolute_calc_dir_keeps_the_same_anchor(skill, tmp_path):
    work_dir = str(tmp_path)
    calcdir = os.path.join(work_dir, 'tmp_%s' % skill)
    os.makedirs(calcdir)

    recorder, deck = _launch(skill, calcdir, work_dir)

    assert os.path.realpath(_resolved_from_the_child(recorder, work_dir)) == \
        os.path.realpath(deck)
    assert os.path.realpath(recorder.cwd) == os.path.realpath(calcdir), (
        'the child no longer starts in the calculation directory')


def test_the_shell_launching_set_is_the_derived_one():
    """The recipe above is the set the criteria exercise, and it discriminates.

    The parametrization is derived from the runners; this pins that the
    derivation finds the shell-launching skills and leaves the in-process
    engines out, so an empty or over-wide recipe cannot pass as coverage.
    """
    assert 'xtb' in SHELL_SKILLS
    assert 'lammps' in SHELL_SKILLS
    for in_process in ('pyscf', 'nn_pes', 'qdm'):
        assert in_process not in SHELL_SKILLS
    assert SHELL_SKILLS == tuple(_shell_skills())
