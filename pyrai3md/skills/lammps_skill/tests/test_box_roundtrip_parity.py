"""
======================================================================
Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

"""Is the cell the framework integrated the cell the engine was handed?

A variable-cell step ends by pushing a box into a live LAMMPS instance and
reading back energy, gradient and a virial stress. The stress is normalized
inside the engine, by the engine's own domain volume, and the barostat then
multiplies it by the framework's volume. So the whole handshake rests on one
claim: the two numbers name the same cell.

The claim is checked here without a LAMMPS installation, on the cells that
the qualification runs actually reached -- including the sheared and
negative-tilt endpoints of the anisotropic NPT examples of
tmp_20260929_091941, which are the shapes no hand-written fixture would have
picked. What is pinned:

- lattice -> true box -> bounding planes -> lattice round-trips. The engine
  writes planes (a data file's box lines, a dump's BOX BOUNDS), the
  framework reads lattices, and the two must be inverses rather than
  approximations of each other;
- the plane span is NOT the axis length when the cell is sheared, so a
  reader that confuses them silently shrinks or grows the box;
- the resident path's box producer and the data file's box writer are the
  same arithmetic, since a cold start and a resident step have to disagree
  about nothing;
- the re-triangularized representative the commit produces is the one that
  travels (dev/methods/ENSEMBLE_DEVELOPMENT.md section 8.1), so a positive
  diagonal and a possibly negative tilt are the only shape a box push has
  to survive.

Run: pytest lammps_skill/tests/test_box_roundtrip_parity.py -v
"""

import numpy as np
import pytest

from ..service.builder.input_builder import (
    bound_and_tilt_to_lattice, box_to_bound_lines, box_to_bound_numbers,
    lattice_to_true_box, resolve_live_box)

# The fixture cell (20 water molecules in 8.399091 A, the density the box was
# equilibrated at), then the shapes the anisotropic runs reached. The last
# three are read out of tmp_20260929_091941's md.xyz comments: nh_pr at its
# 10th frame, nh_mttk at its 8th, csvr_pr at its 6th -- all caught by the
# cumulative-strain gate, and all more sheared than any test written by hand.
LATTICES = {
    'fixture_cubic': [[8.399091, 0.0, 0.0],
                      [0.0, 8.399091, 0.0],
                      [0.0, 0.0, 8.399091]],
    'tilt_positive': [[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    'tilt_negative_xy': [[10.0, 0.0, 0.0], [-3.0, 10.0, 0.0],
                         [1.0, -2.0, 10.0]],
    'all_tilts_negative': [[9.0, 0.0, 0.0], [-1.5, 8.0, 0.0],
                           [-0.5, -1.0, 7.0]],
    # Read as the six cell parameters the run logged
    # (a 4.766674, b 9.130717, c 9.620471, 94.596/85.050/92.510 deg) and
    # converted with the framework's own cell6_to_matrix, so these are the
    # lower-triangular representative the commit would have handed on.
    'nh_pr_frame_10': [[4.766674, 0.0, 0.0],
                       [-0.399919, 9.121955, 0.0],
                       [0.830079, -0.735287, 9.556348]],
    'non_orthogonal_extreme': [[5.0, 0.0, 0.0],
                               [0.868241, 4.924039, 0.0],
                               [1.710101, 2.23703, 4.131737]],
}


def _round_trip(lattice, origin=(0.0, 0.0, 0.0)):
    """lattice -> true box -> bounding planes -> lattice."""
    box = lattice_to_true_box(lattice, *origin)
    bound = box_to_bound_numbers(box)
    bounds = {key: bound[key] for key in ('xlo', 'xhi', 'ylo', 'yhi',
                                          'zlo', 'zhi')}
    tilts = {key: bound[key] for key in ('xy', 'xz', 'yz')}
    return bound_and_tilt_to_lattice(bounds, tilts)


class TestRoundTripIsExact:
    """A cell that went out to the engine and came back is unchanged."""

    @pytest.mark.parametrize('name', sorted(LATTICES))
    def test_a_cell_comes_back_as_the_cell_it_was(self, name):
        lattice = np.array(LATTICES[name], dtype=np.float64)
        back = np.array(_round_trip(lattice), dtype=np.float64)
        assert back.shape == (3, 3)
        assert np.max(np.abs(back - lattice)) < 1e-10, (
            'round trip moved vertices by %.3e A'
            % np.max(np.abs(back - lattice)))

    @pytest.mark.parametrize('name', sorted(LATTICES))
    def test_the_metric_survives_the_plane_frame(self, name):
        # The bounding planes describe the same cell as the vectors, so the
        # Gram matrix -- which is every distance and angle in the cell --
        # has to be invariant. This catches a swap of tilt sign or axis that
        # a coordinate-by-coordinate comparison could still call close.
        lattice = np.array(LATTICES[name], dtype=np.float64)
        back = np.array(_round_trip(lattice), dtype=np.float64)
        before = lattice.T @ lattice
        after = back.T @ back
        assert np.max(np.abs(after - before)) < 1e-9

    def test_a_box_origin_that_is_not_zero_travels_too(self):
        # The template's lo planes are kept, not reset: a run whose data
        # file starts at xlo = -2 must not be shifted into the origin by
        # reading its own box back.
        lattice = np.array(LATTICES['tilt_negative_xy'], dtype=np.float64)
        back = np.array(_round_trip(lattice, origin=(-2.0, 3.0, -0.5)))
        assert np.max(np.abs(back - lattice)) < 1e-10

    def test_the_plane_span_is_not_the_axis_length_under_shear(self):
        # The bounding planes straddle a sheared domain, so their span is
        # longer than the vector along x: a tilt of -4 pulls xlo to -4 while
        # xhi stays at 10. A reader that divided the span by the axis (or
        # inferred a tilt from it) would hand the barostat a cell the engine
        # never had.
        lattice = [[10.0, 0.0, 0.0], [-4.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
        bound = box_to_bound_numbers(lattice_to_true_box(lattice, 0.0, 0.0,
                                                         0.0))
        assert bound['xy'] == pytest.approx(-4.0)
        assert bound['xlo'] == pytest.approx(-4.0)
        assert bound['xhi'] == pytest.approx(10.0)
        assert bound['xhi'] - bound['xlo'] == pytest.approx(14.0)
        # y is pulled only by yz, which this cell does not have.
        assert bound['ylo'] == pytest.approx(0.0)
        assert bound['yhi'] == pytest.approx(8.0)


class TestOneProducerPerFrame:
    """The resident push and the data file write are the same arithmetic.

    dev/skills/LAMMPS_SKILL_DEVELOPMENT.md section 11.3 claims one shared
    producer so a variable-cell step cannot evaluate in a box that differs
    from the one the cold path would have written. That claim is what the
    handshake rests on, so it is pinned here on the sheared cells rather
    than on the cube a fixture would normally supply.
    """

    @pytest.mark.parametrize('name', ('fixture_cubic', 'tilt_negative_xy',
                                      'nh_pr_frame_10',
                                      'non_orthogonal_extreme'))
    def test_live_box_and_the_data_file_line_agree(self, name):
        from ..schema.adapter import build_config
        from ..schema.normalize import normalize

        # normalize without validate: the box arithmetic needs the cell
        # fields, and validate also insists the topology template exists in
        # the launch directory, which a pure-geometry test has no reason to
        # stage.
        lattice = LATTICES[name]
        config = normalize(build_config({
            'atoms': ['O'], 'coordinates': [[0.0, 0.0, 0.0]],
            'profile': 'ff_gradient', 'method': 'ff',
            'lammps_project': 'LAMMPS', 'lammps_units': 'real',
            'pbc': [1, 1, 1],
            'lattice_vector': [list(row) for row in lattice]}))
        live = resolve_live_box(config, (0.0, 0.0, 0.0))
        assert live is not None, 'a periodic cell produced no box'
        stated = box_to_bound_lines(lattice_to_true_box(lattice, 0.0, 0.0,
                                                       0.0))
        assert [f"{live[key]:.10f}" for key in
                ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi')] == [
                    token for line in stated[:3] for token in
                    line.split()[:2]], name
        # And the box the engine will be moved to decodes back to the cell
        # the integrator committed.
        back = np.array(bound_and_tilt_to_lattice(
            {key: live[key] for key in ('xlo', 'xhi', 'ylo', 'yhi', 'zlo',
                                        'zhi')},
            {key: live[key] for key in ('xy', 'xz', 'yz')}), dtype=np.float64)
        assert np.max(np.abs(back - np.array(lattice, dtype=np.float64))
                      ) < 1e-10, name

    def test_a_negative_tilt_is_not_read_as_a_zero_tilt(self):
        # The commit re-triangularizes, and the Cholesky representative can
        # carry negative tilts. A threshold that discarded |tilt| below an
        # epsilon as noise would silently orthorhombic-such a cell, and the
        # engine's stress would then belong to a different shape.
        lattice = [[10.0, 0.0, 0.0], [-0.05, 10.0, 0.0], [0.0, 0.0, 10.0]]
        bound = box_to_bound_numbers(lattice_to_true_box(lattice, 0.0, 0.0,
                                                         0.0))
        assert bound['xy'] == pytest.approx(-0.05)
        back = bound_and_tilt_to_lattice(
            {key: bound[key] for key in ('xlo', 'xhi', 'ylo', 'yhi', 'zlo',
                                         'zhi')},
            {key: bound[key] for key in ('xy', 'xz', 'yz')})
        assert back[1][0] == pytest.approx(-0.05)
