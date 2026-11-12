"""
Unit tests for the periodicity the LAMMPS test fixtures declare.

Every LAMMPS topology here is a periodic water box: the shared fixture
carries a box of its own and its ``in.init`` declares ``kspace_style``, so a
case that does not say ``pbc`` is computed periodically while its config, its
planned artifacts and its run log all claim the opposite. The framework
config is the single source of the cell for both execution paths, and
``method=skill`` is no exception to that: it hands the task to the engine,
not the parameters. These tests keep the fixtures honest about it -- the
declared box must equal the fixture's box, and the declared periodicity must
be what actually reaches the generated data file.

These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_pbc_declaration.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import re

import pytest
import yaml

from ..schema.adapter import build_config
from ..service.builder import write_data_file
from ..service.builder.input_builder import _resolve_periodic_lattice
from ..schema.normalize import normalize
from ..schema.validate import validate

_TESTS_DIR = os.path.dirname(__file__)
_FIXTURE = 'h2o'
_CASES = ('ff_energy', 'ff_gradient', 'ff_gradient_persistent', 'ff_stress',
          'ff_nve', 'ff_nvt', 'ff_npt', 'ff_optimize')


@pytest.fixture(autouse=True)
def _in_launch_directory(monkeypatch):
    """The skill reads its topology from the launch directory."""
    monkeypatch.chdir(_TESTS_DIR)


def _input_of(case):
    with open(os.path.join(_TESTS_DIR, f'{case}.yaml'),
              encoding='utf-8') as handle:
        return yaml.safe_load(handle)['input']


def _configured(case, **overrides):
    raw = {**_input_of(case), 'lammps_project': _FIXTURE, **overrides}
    return validate(normalize(build_config(raw)))


def _fixture_box():
    """The three axis lengths and the tilt factors of the topology file."""
    with open(os.path.join(_TESTS_DIR, f'{_FIXTURE}.data'),
              encoding='utf-8') as handle:
        lines = handle.read().splitlines()
    lengths = []
    tilts = [0.0, 0.0, 0.0]
    for line in lines:
        tokens = line.split()
        if len(tokens) == 4 and tokens[2:] in (['xlo', 'xhi'],
                                              ['ylo', 'yhi'],
                                              ['zlo', 'zhi']):
            lengths.append(float(tokens[1]) - float(tokens[0]))
        elif len(tokens) == 6 and tokens[3:] == ['xy', 'xz', 'yz']:
            tilts = [float(value) for value in tokens[:3]]
    assert len(lengths) == 3, f'{_FIXTURE}.data declares no complete box'
    return lengths, tilts


class TestFixturePeriodicityDeclaration:
    """A case that runs on a periodic topology declares itself periodic."""

    @pytest.mark.parametrize('case', _CASES)
    def test_pbc_is_declared_on(self, case):
        assert _input_of(case).get('pbc') == [1, 1, 1]

    @pytest.mark.parametrize('case', _CASES)
    def test_cell_is_declared(self, case):
        # Every case states a full six-parameter cell; whether it states
        # the *right* one is the topology test below, so the two do not
        # keep a second copy of the box here.
        cell = _input_of(case).get('cell')
        assert isinstance(cell, list) and len(cell) == 6

    @pytest.mark.parametrize('case', _CASES)
    def test_declared_cell_matches_the_topology(self, case):
        # The declaration is only truthful while it agrees with the box the
        # force field was built for: a mismatch would silently resize the
        # system every case runs on.
        lengths, tilts = _fixture_box()
        cell = _input_of(case)['cell']
        assert cell[:3] == lengths
        assert tilts == [0.0, 0.0, 0.0]
        assert cell[3:] == [90.0, 90.0, 90.0]


class TestDeclaredCellReachesTheDataFile:
    """The framework cell, not the template box, is what the run loads."""

    @pytest.mark.parametrize('case', _CASES)
    def test_lattice_resolves_from_the_config(self, case):
        assert _resolve_periodic_lattice(_configured(case)) is not None

    def test_data_file_box_follows_the_config(self, tmp_path):
        # A cell the fixture never had: if the framework cell really is the
        # source, the written box follows it instead of the template.
        write_data_file(_configured('ff_gradient',
                                    cell=[12.5, 12.5, 12.5,
                                          90.0, 90.0, 90.0]), str(tmp_path))
        text = (tmp_path / f'{_FIXTURE}.data').read_text()
        assert re.search(r'^0\.0+ 12\.50* xlo xhi$', text, re.M)
        assert '0.0 10.00000000 xlo xhi' not in text

    def test_without_pbc_the_template_box_survives(self, tmp_path):
        # The other branch, kept on purpose: a non-periodic run must not
        # have a box invented for it. This is also why an undeclared case
        # reads as non-periodic in its log while LAMMPS still runs the
        # fixture's periodic topology.
        config = _configured('ff_gradient', pbc=[0, 0, 0])
        assert _resolve_periodic_lattice(config) is None
        write_data_file(config, str(tmp_path))
        text = (tmp_path / f'{_FIXTURE}.data').read_text()
        # The untouched template line, read from the template itself: the
        # box a non-periodic run keeps is the one its topology declares,
        # whatever that happens to measure today.
        with open(os.path.join(_TESTS_DIR, f'{_FIXTURE}.data'),
                  encoding='utf-8') as handle:
            template_line = next(
                line for line in handle.read().splitlines()
                if line.endswith('xlo xhi'))
        assert template_line in text
