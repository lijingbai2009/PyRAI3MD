"""
======================================================================
Background point charges are extra `Q` atoms in BAGEL's geometry.

The builder appends them to the molecule block, so the engine's own atom
count is natom + n_q: log geometry blocks are that long, and the per-atom
files carry that many rows. This module pins the consequences that can
silently corrupt a result, plus the two places where counting the charges
must not ask for their truth value:

- a log block has to be sized by natom + n_q, or it closes early and the
  leftover Q rows are prepended to the next block, shifting every geometry
  after the first (the file readers take the leading natom rows, so only a
  multi-block log shows the defect);
- `parse_log` must pass that size to the block reader;
- ONIOM electrostatic embedding hands over an (n, 4) array, not a list of
  rows, so `if point_charges` in the builder and `charges or []` in the
  parser both raise instead of counting. The parser form is the one a
  remote round measured: every bagel charge case died inside the term
  before a single block was read.

Oct 31 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from types import SimpleNamespace

import numpy as np

from ..service.builder.input_builder import _build_molecule
from ..service.parser import log_parser


def _atom_line(symbol, x, y, z):
    return '{ "atom" : "%s", "xyz" : [ %r, %r, %r ] },' % (symbol, x, y, z)


def _log_two_blocks():
    """Two geometry blocks of 2 real atoms + 1 background charge each."""
    return [
        _atom_line('O', 0.0, 0.0, 0.0),
        _atom_line('H', 0.9, 0.0, 0.0),
        _atom_line('Q', 5.0, 5.0, 5.0),
        _atom_line('O', 1.0, 1.0, 1.0),
        _atom_line('H', 1.9, 1.0, 1.0),
        _atom_line('Q', 6.0, 6.0, 6.0),
    ]


class TestLogBlocksWithCharges:

    def test_the_expanded_size_keeps_every_block_aligned(self):
        blocks = log_parser._read_atom_lines(_log_two_blocks(), 3)
        # One block is natom + 1 rows; the trim to the real atoms happens in
        # `parse_log`, and what this test pins is that the SECOND block still
        # starts on its own first atom instead of the previous block's charge.
        assert blocks[0] == [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0],
                             [5.0, 5.0, 5.0]]
        assert blocks[1] == [[1.0, 1.0, 1.0], [1.9, 1.0, 1.0],
                             [6.0, 6.0, 6.0]]

    def test_the_unexpanded_size_misaligns_the_second_block(self):
        # Non-identity witness: the same log read with the engine's real-atom
        # count alone starts the second block on the first block's charge.
        blocks = log_parser._read_atom_lines(_log_two_blocks(), 2)
        assert blocks[1][0] == [5.0, 5.0, 5.0]


class TestParseOutputSizing:

    def _block_size_seen_by_the_reader(
            self, monkeypatch, tmp_path, point_charges):
        seen = {}

        def fake_coordinates(log, size):
            seen['size'] = size
            return [[[0.0, 0.0, 0.0]] * 2]

        monkeypatch.setattr(log_parser, '_parse_coordinates_from_log',
                            fake_coordinates)
        monkeypatch.setattr(log_parser, '_parse_energy',
                            lambda *a, **k: np.array([0.0]))
        monkeypatch.setattr(log_parser, '_parse_gradients',
                            lambda *a, **k: np.zeros((1, 2, 3)))
        monkeypatch.setattr(log_parser, '_parse_nac',
                            lambda *a, **k: np.zeros((0, 2, 3)))
        monkeypatch.setattr(log_parser, '_parse_hessian_from_log',
                            lambda log, n: np.zeros((3 * n, 3 * n)))
        monkeypatch.setattr(log_parser, '_parse_frequencies_from_log',
                            lambda log: np.zeros(3))
        monkeypatch.setattr(log_parser, '_check_log_completion',
                            lambda path: None)
        monkeypatch.setattr(log_parser, '_extract_velocities',
                            lambda *a, **k: np.zeros((2, 3)))
        log_path = tmp_path / 'run.log'
        log_path.write_text('log\n')
        config = SimpleNamespace(
            natom=2, nstate=1, method='casscf', profile='casscf_energy',
            target_nac_pair=[], point_charges=point_charges)
        results = log_parser.parse_log(str(log_path), config)
        assert np.asarray(results['coordinates']).shape == (1, 2, 3)
        return seen['size']

    def test_parse_output_reads_blocks_of_natom_plus_charges(
            self, monkeypatch, tmp_path):
        size = self._block_size_seen_by_the_reader(
            monkeypatch, tmp_path, [[-0.83, 5.0, 5.0, 5.0]])
        assert size == 3

    def test_the_embedding_array_sizes_the_block_the_same_way(
            self, monkeypatch, tmp_path):
        # ONIOM electrostatic embedding hands the parser an (n, 4) array,
        # not a list of rows. Counting it with `charges or []` raises
        # instead, which is what a remote round measured on every bagel
        # charge case: the term died before any block was read.
        rows = np.array([[-0.83, 5.0, 5.0, 5.0]])
        size = self._block_size_seen_by_the_reader(
            monkeypatch, tmp_path, rows)
        assert size == 3

    def test_no_charges_keeps_the_block_at_natom(
            self, monkeypatch, tmp_path):
        for empty in (None, [], np.zeros((0, 4))):
            size = self._block_size_seen_by_the_reader(
                monkeypatch, tmp_path, empty)
            assert size == 2


def _config(point_charges):
    return SimpleNamespace(
        atoms=['O', 'H', 'H'],
        coordinates=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]],
        point_charges=point_charges,
        basis='cc-pvdz', df_basis='svp-jkfit', bagel_path='')


class TestBuilderAcceptsArrayCharges:

    def test_an_array_of_rows_builds_the_same_q_atoms(self):
        rows = np.array([[-0.83, 5.0, 5.0, 5.0], [0.415, 6.0, 5.0, 5.0]])
        geometry = _build_molecule(_config(rows))['geometry']
        assert len(geometry) == 5
        assert [entry['atom'] for entry in geometry] == [
            'O', 'H', 'H', 'Q', 'Q']
        assert geometry[3]['charge'] == -0.83
        assert geometry[4]['charge'] == 0.415

    def test_the_list_form_is_unchanged(self):
        rows = [[-0.83, 5.0, 5.0, 5.0], [0.415, 6.0, 5.0, 5.0]]
        geometry = _build_molecule(_config(rows))['geometry']
        assert [entry['atom'] for entry in geometry] == [
            'O', 'H', 'H', 'Q', 'Q']

    def test_no_charges_leaves_the_real_atoms_alone(self):
        for empty in (None, [], np.zeros((0, 4))):
            geometry = _build_molecule(_config(empty))['geometry']
            assert len(geometry) == 3
