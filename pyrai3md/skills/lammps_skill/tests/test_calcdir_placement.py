"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Where a LAMMPS run puts its files, and who decides each name.

The directory is the framework's, not the skill's: ``tmp_LAMMPS`` is the
standalone fallback and every path a run writes hangs off whatever
``scratch_path`` and execution namespace the caller supplied. The names
inside it are the second half of the same contract, because a person who
found one of them finds the rest by pattern.

Run: pytest lammps_skill/tests/test_calcdir_placement.py -v
"""

import ast
import pathlib

from ..service.builder.ff_model import model_path
from ..service.persistent import launcher
from ..service.utils import (build_calcdir, output_basename,
                             step_data_files)

SKILL_DIR = pathlib.Path(__file__).resolve().parents[1]


def _code_strings(path: pathlib.Path):
    """String literals of one module that are not docstrings."""
    tree = ast.parse(path.read_text())
    skipped = set()
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and \
                isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant):
            skipped.add(id(body[0].value))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in skipped:
            found.append(node.value)
    return found


def test_the_directory_name_follows_the_caller():
    # The execution namespace assigned by the Calculator is the directory;
    # the skill's own name appears only when there is no namespace at all.
    assert build_calcdir(execution_id='tmp_mm_layer0_LAMMPS',
                         base_path='/job') == '/job/tmp_mm_layer0_LAMMPS'
    assert build_calcdir(execution_id='tmp_LAMMPS',
                         base_path='/job') == '/job/tmp_LAMMPS'
    assert build_calcdir(execution_id='', base_path='/job') == \
        '/job/tmp_LAMMPS'


def test_the_fallback_directory_name_is_decided_in_one_place():
    # A second module that wrote 'tmp_LAMMPS' itself would be a second
    # opinion about the layout, and the one that disagrees silently.
    offenders = []
    for path in sorted(SKILL_DIR.rglob('*.py')):
        if 'tests' in path.parts:
            continue
        if any('tmp_LAMMPS' in value for value in _code_strings(path)):
            offenders.append(path.name)
    assert offenders == ['utils.py'], offenders


def test_every_resident_artifact_follows_the_calculation_directory():
    # Nothing a warm run writes lives outside the directory a cold run
    # would have used, so one keep_tmp decides the fate of all of it.
    calcdir = '/job/tmp_mm_layer0_LAMMPS'
    paths = launcher.service_paths(calcdir, 'tmp_mm_layer0_LAMMPS',
                                   'h2o_force')
    assert paths.work_dir == calcdir
    for field in ('script', 'log', 'endpoint', 'owner_record'):
        assert getattr(paths, field).startswith(calcdir + '/'), field
    assert paths.endpoint == f'{calcdir}/{launcher.SOCKET_NAME}'
    assert paths.owner_record == f'{calcdir}/{launcher.OWNER_NAME}'


def test_the_model_cache_is_a_package_member():
    assert model_path('h2o', '/job/tmp_LAMMPS') == '/job/tmp_LAMMPS/h2o.ff.json'


def test_the_result_files_share_the_run_stem():
    # The same stem as the input, script and log: the profile decides the
    # name once, and each quantity a step produced hangs off it.
    title = output_basename('h2o', 'ff_gradient')
    assert title == 'h2o_force'
    assert sorted(step_data_files(title, gradient=True,
                                  stress=True).values()) == [
        'h2o_force.coords', 'h2o_force.energy', 'h2o_force.force',
        'h2o_force.stress',
    ]
    assert sorted(step_data_files(
            output_basename('h2o', 'ff_energy')).values()) == [
        'h2o_ff_energy.coords', 'h2o_ff_energy.energy',
    ]
