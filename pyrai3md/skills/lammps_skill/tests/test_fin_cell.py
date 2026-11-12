"""
Unit tests for the fin_cell channel: the cell an engine ended on, back to
the framework as six cell parameters.

Three things are pinned here. The inverse of the box the skill writes (a
data file and a dump state *bounding* planes, not the domain's true bounds)
must return the lattice the engine holds, so a cell that went out and came
back is the same cell. The second implementation of lattice-to-cell-parameters
that the isolation contract forces into the skill must agree with the
framework's ``utilities.lattice.matrix_to_cell6`` value for value, because a
silent divergence between the two would show up as a cell nobody asked for.
And the two readers -- the ``.cond`` dump of a subprocess run and the box the
resident service returns over the socket -- must produce the same six numbers
for the same domain, which is the same equivalence the rest of this skill is
judged on.

These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_fin_cell.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import numpy as np
import pytest

from ..service.builder.input_builder import (
    bound_and_tilt_to_lattice, box_to_bound_numbers, lattice_to_true_box)
from ..service.parser.output_parser import parse_cond_cell

# A lattice, its bounding planes as LAMMPS writes them, and the six cell
# parameters the framework should be told. The 90-degree case carries the
# float noise cos(radians(90)) leaves behind, which is what #235's tilt
# threshold had to learn to ignore.
LATTICES = {
    'untilted': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    'right-angle-noise': [[10.0, 0.0, 0.0],
                          [6.123233995736766e-16, 10.0, 0.0],
                          [6.123233995736766e-16,
                           6.123233995736766e-16, 10.0]],
    'tilted': [[11.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.5, 0.25, 10.0]],
    'tilted-negative': [[11.0, 0.0, 0.0], [-1.5, 9.0, 0.0],
                        [0.5, -0.7, 10.0]],
    'stretched': [[10.000054, 0.0, 0.0], [0.0, 10.000054, 0.0],
                  [0.0, 0.0, 9.999724]],
}


def _assert_lattice(actual, expected, **kwargs):
    """Row-by-row comparison: pytest.approx refuses nested sequences."""
    assert len(actual) == len(expected)
    for row, want in zip(actual, expected):
        assert list(row) == pytest.approx(list(want), **kwargs)


def _planes(lattice):
    box = lattice_to_true_box(lattice, 0.0, 0.0, 0.0)
    bound = box_to_bound_numbers(box)
    return ({key: bound[key] for key in ('xlo', 'xhi', 'ylo', 'yhi',
                                         'zlo', 'zhi')},
            {key: bound[key] for key in ('xy', 'xz', 'yz')})


class TestBoundToLattice:
    """The box the skill writes is the box the skill reads back."""

    @pytest.mark.parametrize('name', sorted(LATTICES))
    def test_round_trip_returns_the_lattice(self, name):
        lattice = LATTICES[name]
        bounds, tilts = _planes(lattice)
        _assert_lattice(bound_and_tilt_to_lattice(bounds, tilts), lattice,
                        abs=1e-12)

    @pytest.mark.parametrize('name', sorted(LATTICES))
    def test_a_moved_box_comes_back_moved(self, name):
        # The point of the channel: an engine that resized the domain has
        # to be able to say so, so the planes must not be collapsed back to
        # whatever the input cell was.
        lattice = [[row[0] * 1.03, row[1] * 1.03, row[2] * 1.03]
                   for row in LATTICES[name]]
        bounds, tilts = _planes(lattice)
        _assert_lattice(bound_and_tilt_to_lattice(bounds, tilts), lattice,
                        rel=1e-12)

    def test_negative_shear_is_not_read_as_a_bigger_box(self):
        # xy = -1 slides b back over a, so the bounding planes reach from
        # -1 to 11 while the a vector is 11 long: the plane *span* (12) is
        # neither the length nor the length plus the tilt, and undoing only
        # the outward pull the writer applied is what gets 11 back.
        bounds = {'xlo': -1.0, 'xhi': 11.0, 'ylo': 0.0, 'yhi': 10.0,
                  'zlo': 0.0, 'zhi': 10.0}
        lattice = bound_and_tilt_to_lattice(bounds, {'xy': -1.0, 'xz': 0.0,
                                                     'yz': 0.0})
        assert lattice[0][0] == pytest.approx(11.0)
        assert lattice[1] == pytest.approx([-1.0, 10.0, 0.0])

    def test_planes_that_enclose_nothing_are_refused(self):
        with pytest.raises(RuntimeError):
            bound_and_tilt_to_lattice(
                {'xlo': 0.0, 'xhi': 1.0, 'ylo': 0.0, 'yhi': 10.0,
                 'zlo': 0.0, 'zhi': 10.0},
                {'xy': 5.0, 'xz': 0.0, 'yz': 0.0})


def _dump(bounds, tilts=(), rows=None):
    """A one-frame custom dump with the box block LAMMPS writes.

    Three atoms, matching the fixture the sibling parser tests use, so a
    dump handed to ``parse_output`` passes its atom-count check.
    """
    header = 'ITEM: BOX BOUNDS pp pp pp'
    lines = [f'{lo} {hi}' for lo, hi in bounds]
    if tilts:
        header = 'ITEM: BOX BOUNDS xy xz yz pp pp pp'
        lines = [f'{lo} {hi} {tilt}'
                 for (lo, hi), tilt in zip(bounds, tilts)]
    body = rows if rows is not None else [
        '3 5.95927 4.36082 3.75416 0.0 -0.002 0.001',
        '1 5.36547 3.97334 3.11112 0.001 0.002 0.003',
        '2 5.03173 3.18409 3.53766 -0.001 0.0 0.0',
    ]
    return ('ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n3\n'
            + header + '\n' + '\n'.join(lines) + '\n'
            'ITEM: ATOMS id x y z vx vy vz\n'
            + '\n'.join(body) + '\n')


class TestCondCellReader:
    """The cell is taken from the frame the run ended on, not assumed."""

    def test_orthogonal_dump(self, tmp_path):
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]))
        _assert_lattice(parse_cond_cell(str(path)),
                        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                         [0.0, 0.0, 10.0]])

    def test_tilt_names_come_from_the_header(self, tmp_path):
        # xy on the x line, xz on the y line, yz on the z line: reading the
        # header is what tells these three apart from one another.
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(0.0, 12.0), (0.0, 10.0), (0.0, 10.0)],
                              tilts=[1.0, 0.5, 0.25]))
        lattice = parse_cond_cell(str(path))
        assert lattice[1][0] == pytest.approx(1.0)
        assert lattice[2][0] == pytest.approx(0.5)
        assert lattice[2][1] == pytest.approx(0.25)

    def test_a_moved_box_is_reported_moved(self, tmp_path):
        # The NPT case: 10 A in, a different box out. The state must be told.
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(0.0, 10.000054), (0.0, 10.000054),
                               (0.0, 9.999724)]))
        lattice = parse_cond_cell(str(path))
        assert lattice[0][0] == pytest.approx(10.000054)
        assert lattice[2][2] == pytest.approx(9.999724)

    def test_the_deployment_box_reads_at_full_precision(self, tmp_path):
        # A real ff_npt frame from the deployment (orthorhombic, origins
        # moved off zero by the barostat). The lengths are the plane
        # differences, and %.16e means nothing is lost on the way back.
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(-5.4270313555981886e-05,
                                1.0000054270313555e+01),
                               (-5.4047498548204231e-04,
                                1.0000540474985485e+01),
                               (2.7585204359503734e-04,
                                9.9997241479564067e+00)]))
        lattice = parse_cond_cell(str(path))
        # Written as the plane differences themselves, so the expectation
        # cannot be mis-transcribed out of a calculator.
        assert lattice[0][0] == pytest.approx(
            1.0000054270313555e+01 + 5.4270313555981886e-05, rel=1e-14)
        assert lattice[1][1] == pytest.approx(
            1.0000540474985485e+01 + 5.4047498548204231e-04, rel=1e-14)
        assert lattice[2][2] == pytest.approx(
            9.9997241479564067e+00 - 2.7585204359503734e-04, rel=1e-14)
        # ...and the cell the framework will store is not the input box:
        # utilities.lattice.matrix_to_cell6 reports row lengths, so its
        # a is this row's norm.
        assert float(np.linalg.norm(lattice[0])) > 10.0001

    def test_a_sheared_box_is_not_read_as_a_longer_one(self, tmp_path):
        # The planes span 12 A with xy=1; the lattice a-vector is 11.
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(-1.0, 11.0), (0.0, 10.0), (0.0, 10.0)],
                              tilts=[1.0, 0.0, 0.0]))
        assert parse_cond_cell(str(path))[0][0] == pytest.approx(11.0)

    def test_last_frame_wins(self, tmp_path):
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(0.0, 9.0), (0.0, 9.0), (0.0, 9.0)])
                        + _dump([(0.0, 11.0), (0.0, 11.0), (0.0, 11.0)]))
        assert parse_cond_cell(str(path))[0][0] == pytest.approx(11.0)

    def test_dump_without_a_box_block_reports_nothing(self, tmp_path):
        path = tmp_path / 'x.cond'
        path.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\n'
                        'ITEM: ATOMS id x y z vx vy vz\n'
                        '1 0 0 0 0 0 0\n2 0 0 0 0 0 0\n3 0 0 0 0 0 0\n')
        assert parse_cond_cell(str(path)) is None

    def test_unknown_tilt_name_is_refused(self, tmp_path):
        path = tmp_path / 'x.cond'
        path.write_text(_dump([(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                              tilts=[1.0, 0.0, 0.0]).replace(
                                  'xy xz yz', 'xy qz yz'))
        with pytest.raises(RuntimeError, match='unknown tilt'):
            parse_cond_cell(str(path))

    def test_truncated_box_block_is_refused(self, tmp_path):
        path = tmp_path / 'x.cond'
        # Drop only the last of the three box lines: they are identical, so
        # filtering by content would remove the whole block instead.
        lines = _dump([(0.0, 10.0), (0.0, 10.0), (0.0, 9.5)]).splitlines()
        path.write_text('\n'.join(lines[:-5]) + '\n')
        with pytest.raises(RuntimeError, match='truncated'):
            parse_cond_cell(str(path))


class TestParseOutputReportsTheCell:
    """MD returns fin_cell; an internal minimize, which cannot move a box, does not."""

    def _md(self, tmp_path, cond):
        from .test_output_parser import (MULTI_LOG, FINAL_XYZ, _make_config,
                                         _write_outputs)
        from ..service.parser.output_parser import parse_output
        from ..service.workflow import compile_workflow

        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       cond=cond, profile='ff_nvt')
        return parse_output(_make_config(profile='ff_nvt'), str(tmp_path),
                            compile_workflow('ff_nvt'))

    def test_md_reports_the_flattened_lattice(self, tmp_path):
        # Nine components, row-major: the six cell parameters are derived
        # from them by the framework, which owns that formula.
        results = self._md(
            tmp_path,
            _dump([(0.0, 10.5), (0.0, 10.5), (0.0, 10.5)]))
        assert results['fin_cell'] == pytest.approx(
            [10.5, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 10.5])

    def test_md_still_reports_the_other_channels(self, tmp_path):
        # Adding a channel must not cost the run its velocities.
        results = self._md(tmp_path,
                           _dump([(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]))
        assert results['fin_velocity'].shape == (3, 3)
        assert results['fin_coord'].shape == (3, 3)

    def test_optimize_does_not_report_a_cell(self, tmp_path):
        from .test_output_parser import (MULTI_LOG, FINAL_XYZ, _make_config,
                                         _write_outputs)
        from ..service.parser.output_parser import parse_output
        from ..service.workflow import compile_workflow

        # ff_optimize writes no .cond at all, and 'minimize' has no cell
        # degree of freedom: a cell reported here would be an invention.
        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       profile='ff_optimize')
        results = parse_output(_make_config(profile='ff_optimize'),
                              str(tmp_path), compile_workflow('ff_optimize'))
        assert 'fin_cell' not in results
