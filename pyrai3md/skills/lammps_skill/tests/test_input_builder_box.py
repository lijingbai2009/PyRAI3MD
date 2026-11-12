"""
Unit tests for the LAMMPS box arithmetic shared by both execution paths.

service/builder keeps one producer of box numbers now: the data file the
subprocess path writes and the box the resident path pushes into a live
LAMMPS instance both come from lattice_to_true_box. These tests pin that
producer against the formula the pre-refactor writer used inline, and
against the LAMMPS read_data bounding-box convention, so a change here
cannot quietly move one path away from the other.

These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_input_builder_box.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder import write_data_file
from ..service.builder.input_builder import (
    box_to_bound_lines, lattice_to_true_box, parse_template_box_origin,
    resolve_live_box, write_static_input_files)
from ..service.workflow import compile_workflow

ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]

DATA = """LAMMPS Description

3  atoms
0  bonds
0  angles
0  dihedrals
0  impropers

2  atom types
0  bond types
0  angle types
0  dihedral types
0  improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 15.999
2 1.008

Atoms

1 1 1 -0.834 0.0000000000 0.0000000000 0.1173000000
2 1 2 0.417 0.0000000000 0.7572000000 -0.4692000000
3 1 2 0.417 0.0000000000 -0.7572000000 -0.4692000000
"""


def _make_config(**kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'profile': 'ff_gradient',
        'method': 'ff',
        'lammps_project': 'LAMMPS',
        'lammps_units': 'real',
    }
    raw.update(kwargs)
    return validate(normalize(build_config(raw)))


@pytest.fixture
def launch_dir(tmp_path, monkeypatch):
    """Launch directory carrying the topology template."""
    (tmp_path / 'LAMMPS.data').write_text(DATA)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestLatticeToTrueBox:
    """The lattice maps onto true box bounds plus LAMMPS tilt factors."""

    def test_orthorhombic_keeps_origin_and_lengths(self):
        lattice = [[12.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 7.0]]
        box = lattice_to_true_box(lattice, 1.0, 2.0, 3.0)
        assert box['xlo'] == 1.0 and box['ylo'] == 2.0 and box['zlo'] == 3.0
        assert box['xhi'] == 13.0 and box['yhi'] == 11.0 and box['zhi'] == 10.0
        assert (box['xy'], box['xz'], box['yz']) == (0.0, 0.0, 0.0)

    def test_lower_triangular_elements_become_the_tilts(self):
        lattice = [[10.0, 0.0, 0.0], [1.5, 8.0, 0.0], [0.5, 0.25, 6.0]]
        box = lattice_to_true_box(lattice, 0.0, 0.0, 0.0)
        assert (box['xy'], box['xz'], box['yz']) == (1.5, 0.5, 0.25)

    def test_non_positive_length_is_refused(self):
        lattice = [[0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]]
        with pytest.raises(RuntimeError, match='must be positive'):
            lattice_to_true_box(lattice, 0.0, 0.0, 0.0)

    def test_upper_triangular_lattice_is_refused(self):
        # A non-lower-triangular lattice cannot be written as a LAMMPS
        # data box; the framework triangularizes before it gets here.
        lattice = [[10.0, 1.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]]
        with pytest.raises(RuntimeError, match='lower-triangular'):
            lattice_to_true_box(lattice, 0.0, 0.0, 0.0)


class TestBoundLines:
    """True bounds render to the bounding planes read_data expects."""

    @staticmethod
    def _reference(box):
        """The formula _rewrite_data_box carried before the extraction."""
        xy, xz, yz = box['xy'], box['xz'], box['yz']
        return [
            f"{box['xlo'] + min(0.0, xy, xz, xy + xz):.10f} "
            f"{box['xhi'] + max(0.0, xy, xz, xy + xz):.10f} xlo xhi",
            f"{box['ylo'] + min(0.0, yz):.10f} "
            f"{box['yhi'] + max(0.0, yz):.10f} ylo yhi",
            f"{box['zlo']:.10f} {box['zhi']:.10f} zlo zhi",
        ]

    def test_orthorhombic_writes_three_lines_only(self):
        box = lattice_to_true_box([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]], 0.0, 0.0, 0.0)
        lines = box_to_bound_lines(box)
        assert len(lines) == 3
        assert lines == self._reference(box)

    def test_positive_tilt_moves_only_the_far_plane(self):
        box = lattice_to_true_box([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0],
                                   [1.0, 1.0, 10.0]], 0.0, 0.0, 0.0)
        lines = box_to_bound_lines(box)
        assert lines[0].split()[0] == '0.0000000000'
        # x carries the whole shear span (xy + xz = 3), y only yz = 1.
        assert float(lines[0].split()[1]) == pytest.approx(13.0)
        assert float(lines[1].split()[1]) == pytest.approx(11.0)
        assert lines[3] == '2.0000000000 1.0000000000 1.0000000000 xy xz yz'

    def test_negative_tilt_moves_the_near_plane(self):
        box = lattice_to_true_box([[10.0, 0.0, 0.0], [-2.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]], 0.0, 0.0, 0.0)
        lines = box_to_bound_lines(box)
        assert float(lines[0].split()[0]) == pytest.approx(-2.0)
        assert float(lines[0].split()[1]) == pytest.approx(10.0)
        assert float(lines[1].split()[0]) == pytest.approx(0.0)

    def test_rendered_lines_match_the_pre_refactor_formula(self):
        for xy, xz, yz in ((2.0, 1.0, 1.0), (-2.0, 1.0, -1.0),
                           (0.0, -3.0, 0.5), (-1.0, -2.0, -3.0)):
            box = lattice_to_true_box(
                [[10.0, 0.0, 0.0], [xy, 8.0, 0.0], [xz, yz, 6.0]],
                0.5, 1.5, 2.5)
            assert box_to_bound_lines(box)[:3] == self._reference(box)

    def test_zero_tilt_stays_orthorhombic(self):
        box = lattice_to_true_box([[10.0, 0.0, 0.0], [1e-12, 8.0, 0.0],
                                   [0.0, 0.0, 6.0]], 0.0, 0.0, 0.0)
        assert len(box_to_bound_lines(box)) == 3


class TestTemplateOrigin:
    """The writer keeps the template's lo planes."""

    def test_origin_and_indices_are_reported(self):
        lines = ['0.5 10.5 xlo xhi', '1.5 11.5 ylo yhi', '2.5 12.5 zlo zhi']
        origin, box_idx = parse_template_box_origin(lines, 'x.data')
        assert origin == (0.5, 1.5, 2.5)
        assert [box_idx[label][0] for label in
                ('xlo xhi', 'ylo yhi', 'zlo zhi')] == [0, 1, 2]

    def test_incomplete_box_is_named(self):
        with pytest.raises(RuntimeError, match='x.data'):
            parse_template_box_origin(['0.0 10.0 xlo xhi'], 'x.data')


class TestResolveLiveBox:
    """pbc stays the authority for whether the framework cell applies."""

    def test_non_periodic_run_keeps_the_template_box(self, launch_dir):
        config = _make_config(pbc=[0, 0, 0])
        assert resolve_live_box(config, (0.0, 0.0, 0.0)) is None

    def test_lattice_vector_is_preferred(self, launch_dir):
        config = _make_config(
            pbc=[1, 1, 1],
            lattice_vector=[[12.0, 0.0, 0.0], [1.0, 9.0, 0.0],
                            [0.5, 0.25, 7.0]])
        box = resolve_live_box(config, (0.0, 0.0, 0.0))
        assert box['xy'] == 1.0 and box['yz'] == 0.25
        # The planes of a data file, not the true bounds: the shear pulls
        # x by max(0, xy, xz, xy+xz) and y by max(0, yz).
        assert box['xhi'] == pytest.approx(13.5)
        assert box['yhi'] == pytest.approx(9.25)

    def test_the_step_box_is_the_data_file_box(self, launch_dir):
        # One convention, one source: whatever the resident path is told to
        # move the domain to is what the subprocess path would have read.
        lattice = [[12.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.5, 0.25, 7.0]]
        config = _make_config(pbc=[1, 1, 1], lattice_vector=lattice)
        box = resolve_live_box(config, (0.0, 0.0, 0.0))
        stated = box_to_bound_lines(lattice_to_true_box(lattice, 0.0, 0.0,
                                                        0.0))
        assert [f"{box[key]:.10f}" for key in
                ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi')] == [
                    token for line in stated[:3] for token in line.split()[:2]]

    def test_cell_parameters_are_the_fallback(self, launch_dir):
        config = _make_config(pbc=[1, 1, 1], lattice_vector=[],
                             cell=[10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        box = resolve_live_box(config, (0.0, 0.0, 0.0))
        assert box['xhi'] == pytest.approx(10.0)
        assert abs(box['xy']) < 1e-9 and abs(box['yz']) < 1e-9


class TestStaticPackage:
    """The one-time package matches the data file the cold path writes."""

    def test_package_files_and_script_shape(self, launch_dir, tmp_path):
        config = _make_config(pbc=[1, 1, 1],
                              lattice_vector=[[10.0, 0.0, 0.0],
                                              [0.0, 10.0, 0.0],
                                              [0.0, 0.0, 10.0]])
        workflow = compile_workflow('ff_gradient')
        session = str(tmp_path / 'session')
        package = write_static_input_files(config, workflow, session, None)

        assert sorted(package['files']) == ['LAMMPS.data', 'LAMMPS_force.in']
        assert package['order'] == [1, 2, 3]
        assert package['natom'] == 3
        assert package['box']['xhi'] == 10.0

        script = open(f"{session}/LAMMPS_force.in").read()
        # The loaded script sets up output control and stops there: the
        # service applies a box and a geometry before it evaluates, so a
        # run on load would answer for a geometry no step had asked for.
        assert 'run 0' not in script
        assert 'thermo_style custom step pe etotal' in script
        assert 'compute Pconf all pressure NULL virial' not in script
        # The evaluation is the sequence every step replays, and without
        # debug it names no output file.
        assert package['step_tail'] == ['run 0']
        # A step is a run of zero steps, on both paths. `run 1 pre no` skips
        # the run's opening and measures 1.135 ms where `run 0` measures
        # 1.624 ms, but it is not the same calculation to the last bit, so it
        # is not what ships; the measurement and the reason are in
        # build_run_section and in dev/skills/LAMMPS_SKILL_DEVELOPMENT.md.
        assert 'pre no' not in script
        assert 'pre no' not in '\n'.join(package['step_tail'])
        assert 'write_dump' not in script
        assert '.energy' not in '\n'.join(package['step_tail'])
        assert 'LAMMPS_force.force' not in script

    def test_debug_sends_the_cold_write_commands_per_step(self, launch_dir,
                                                          tmp_path):
        config = _make_config(pbc=[1, 1, 1], debug=True, stress=True,
                              lattice_vector=[[10.0, 0.0, 0.0],
                                              [0.0, 10.0, 0.0],
                                              [0.0, 0.0, 10.0]])
        workflow = compile_workflow('ff_gradient')
        session = str(tmp_path / 'debug')
        package = write_static_input_files(config, workflow, session, None)
        tail = '\n'.join(package['step_tail'])

        # debug=true means a resident step leaves the same files behind as
        # the subprocess step it stands in for, so the per-step sequence
        # carries exactly those write commands.
        assert 'run 0' in tail
        assert 'print "${pyrai_pe}" file LAMMPS_force.energy' in tail
        assert 'write_dump all custom LAMMPS_force.coords' in tail
        assert 'write_dump all custom LAMMPS_force.force' in tail
        assert 'file LAMMPS_force.stress' in tail
        # The setup stays in the script, issued once, as on the cold path.
        script = open(f"{session}/LAMMPS_force.in").read()
        assert 'compute Pconf all pressure NULL virial' in script
        assert 'run 0' not in script

    def test_data_file_matches_the_cold_writer(self, launch_dir, tmp_path):
        config = _make_config(pbc=[1, 1, 1],
                              lattice_vector=[[11.0, 0.0, 0.0],
                                              [1.0, 9.0, 0.0],
                                              [0.0, 0.5, 8.0]])
        workflow = compile_workflow('ff_gradient')
        cold = str(tmp_path / 'cold')
        session = str(tmp_path / 'session')
        import os
        os.makedirs(cold)
        write_data_file(config, cold, None)
        write_static_input_files(config, workflow, session, None)

        with open(f'{cold}/LAMMPS.data') as f:
            cold_text = f.read()
        with open(f'{session}/LAMMPS.data') as f:
            session_text = f.read()
        assert cold_text == session_text
        assert 'xy xz yz' in cold_text
