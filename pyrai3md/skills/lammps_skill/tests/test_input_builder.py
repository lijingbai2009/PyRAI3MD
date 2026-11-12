"""
Unit tests for lammps_skill input generation (input_builder).

Verifies the input script sections of the internal-iteration
profiles against the reference lammps_demo inputs:
  - ff_optimize: min_style + minimize + final write_dump + restart
  - ff_nve/ff_nvt/ff_npt: thermo control, velocity initialization,
    timestep, ensemble fix, run, final xyz/cond write_dump
  - template execution commands are dropped for these profiles
  - the Velocities section of the data file follows the system
    velocities
  - no 'modify element' clause is emitted: the dumps are
    element-independent and sorted by atom id
  - the main input template is optional: without '{project}.in' the
    input script is fully generated from the workflow
  - the generated input script follows the dump output basename
    ('{project}_{profile}.in', '{project}_force.in' for ff_gradient)
    so different tasks in the same directory never overwrite each
    other and the user template '{project}.in' stays untouched
These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_input_builder.py -v

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
from ..service.builder.ff_model import build_ff_model, is_subsystem
from ..service.builder.input_builder import (_SUBSYSTEM_BOX_HALF,
                                             _build_shake_section,
                                             write_input_file,
                                             write_static_input_files)
from ..service.utils import output_basename
from ..service.workflow import compile_workflow

# Water molecule keeps the input compact
ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]

# Template carrying execution commands that the internal-iteration
# profiles must drop (mirrors the reference h2o.in structure)
TEMPLATE_IN = (
    'include "LAMMPS.in.init"\n'
    'read_data "LAMMPS.data"\n'
    'include "LAMMPS.in.settings"\n'
    'thermo 100\n'
    'thermo_style multi\n'
    'thermo_modify norm yes\n'
    'dump dumpall all xyz 100 traj.xyz\n'
    'velocity all create 900 12345\n'
    'timestep 1.0\n'
    'fix old_name all nvt temp 300 300 100\n'
    'run 999\n'
    'unfix old_name\n'
)

TEMPLATE_DATA = (
    'LAMMPS Description\n'
    '\n'
    '3  atoms\n'
    '\n'
    '0.0 10.0 xlo xhi\n'
    '0.0 10.0 ylo yhi\n'
    '0.0 10.0 zlo zhi\n'
    '\n'
    'Masses\n'
    '\n'
    '1 15.999\n'
    '2 1.008\n'
    '\n'
    'Atoms\n'
    '\n'
    '1 1 1 -0.834 0.0000000000 0.0000000000 0.1173000000\n'
    '2 1 2 0.417 0.0000000000 0.7572000000 -0.4692000000\n'
    '3 1 2 0.417 0.0000000000 -0.7572000000 -0.4692000000\n'
    '\n'
    'Bonds\n'
    '\n'
    '1 1 1 2\n'
    '2 1 1 3\n'
)


@pytest.fixture
def launch_dir(tmp_path, monkeypatch):
    """Launch directory carrying the user templates."""
    (tmp_path / 'LAMMPS.in').write_text(TEMPLATE_IN)
    (tmp_path / 'LAMMPS.data').write_text(TEMPLATE_DATA)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def data_only_dir(tmp_path, monkeypatch):
    """Launch directory with only the topology (no main template)."""
    (tmp_path / 'LAMMPS.data').write_text(TEMPLATE_DATA)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_config(**kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'profile': 'ff_energy',
        'method': 'ff',
        'lammps_project': 'LAMMPS',
        'lammps_units': 'real',
    }
    raw.update(kwargs)
    return validate(normalize(build_config(raw)))


def _build(tmp_path, config, profile):
    workflow = compile_workflow(profile)
    write_input_file(config, workflow, str(tmp_path))
    basename = output_basename('LAMMPS', profile)
    return (tmp_path / f'{basename}.in').read_text()


def _atom_lines(text):
    """The Atoms rows of a data file as token lists, in file order."""
    lines = text.split('\n')
    count = int(next(line.split()[0] for line in lines[:16]
                     if line.endswith('atoms')))
    header = next(i for i, line in enumerate(lines)
                  if line.startswith('Atoms'))
    rows = []
    i = header + 1
    while len(rows) < count and i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 6 and parts[0].isdigit():
            rows.append(parts)
        i += 1
    return rows


# The template every image-column test below writes by hand: full style
# with three trailing flags, so the reader has to tell the flags apart
# from the coordinates.
_FLAGGED_DATA = (
    'LAMMPS Description\n\n3  atoms\n\n'
    '0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n'
    'Masses\n\n1 15.999\n2 1.008\n\n'
    'Atoms # full\n\n'
    '1 1 1 -0.834 0.0 0.0 0.0 0 0 0\n'
    '2 1 2 0.417 0.0 0.0 0.0 -1 0 2\n'
    '3 1 2 0.417 0.0 0.0 0.0 0 1 0\n\n'
    'Bonds\n\n1 1 1 2\n2 1 1 3\n'
)

CUBIC10 = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


class TestImageDataColumns:
    """The three trailing Atoms columns carry the run's crossing counts.

    A periodic run states its images instead of letting read_data infer
    them from box containment: the two agree only while every atom sits
    inside the cell, and a snapshot that was wrapped into the main image
    can hold bonds which close only across a face. The framework seeds
    those counts from the geometry (schema.normalize.seed_initial_image)
    and hands them over the same snapshot channel that carries the
    coordinates, so the columns this writer produces are the columns the
    resident path sends as 'set atom ... image'.
    """

    def test_periodic_run_states_its_image_columns(self, launch_dir):
        config = _make_config(profile='ff_energy', pbc=[1, 1, 1],
                              lattice_vector=CUBIC10)
        config.system.image = [[0, 0, 0], [1, 0, -2], [0, 0, 1]]
        _build(launch_dir, config, 'ff_energy')
        lines = _atom_lines((launch_dir / 'LAMMPS.data').read_text())
        assert [line[7:] for line in lines] == \
            [['0', '0', '0'], ['1', '0', '-2'], ['0', '0', '1']]
        # The coordinates and the identity columns are untouched by it.
        assert lines[1][:4] == ['2', '1', '2', '0.417']

    def test_periodic_run_without_crossings_writes_zeros(self, launch_dir):
        # The first step of a fresh run has crossed nothing: stating that
        # explicitly beats leaving it to inference.
        config = _make_config(profile='ff_energy', pbc=[1, 1, 1],
                              lattice_vector=CUBIC10)
        _build(launch_dir, config, 'ff_energy')
        lines = _atom_lines((launch_dir / 'LAMMPS.data').read_text())
        assert [line[7:] for line in lines] == [['0', '0', '0']] * 3

    def test_image_columns_survive_a_second_pass(self, launch_dir):
        # Rebuilding from a file this writer already produced must keep
        # three trailing columns, not append another three.
        config = _make_config(profile='ff_energy', pbc=[1, 1, 1],
                              lattice_vector=CUBIC10)
        config.system.image = [[0, 1, 0], [0, 0, 0], [-1, 0, 2]]
        _build(launch_dir, config, 'ff_energy')
        _build(launch_dir, config, 'ff_energy')
        lines = _atom_lines((launch_dir / 'LAMMPS.data').read_text())
        assert [len(line) for line in lines] == [10, 10, 10]
        assert [line[7:] for line in lines] == \
            [['0', '1', '0'], ['0', '0', '0'], ['-1', '0', '2']]

    def test_a_periodic_run_overstates_a_templates_own_flags(
            self, launch_dir):
        # The snapshot, not the topology template, is the authority: a
        # periodic run that crossed nothing rewrites the flags a template
        # happens to carry, because leaving them would describe a
        # geometry the coordinates no longer have.
        (launch_dir / 'LAMMPS.data').write_text(_FLAGGED_DATA)
        config = _make_config(profile='ff_energy', pbc=[1, 1, 1],
                              lattice_vector=CUBIC10,
                              coordinates=[[1.0, 2.0, 3.0],
                                           [4.0, 5.0, 6.0],
                                           [7.0, 8.0, 9.0]])
        _build(launch_dir, config, 'ff_energy')
        lines = _atom_lines((launch_dir / 'LAMMPS.data').read_text())
        assert [line[7:] for line in lines] == [['0', '0', '0']] * 3
        assert lines[1][4] == '4.0000000000'


class TestOptimizeSection:
    """ff_optimize follows the reference min input (minimize)."""

    def test_minimize_commands(self, launch_dir):
        config = _make_config(profile='ff_optimize')
        text = _build(launch_dir, config, 'ff_optimize')
        assert 'min_style cg' in text
        assert 'minimize 0.0001 1e-08 1000 1000000' in text

    def test_custom_minimize_parameters(self, launch_dir):
        config = _make_config(profile='ff_optimize', min_style='sd',
                              min_etol=1e-5, min_ftol=1e-9,
                              min_maxiter=50, min_maxeval=5000)
        text = _build(launch_dir, config, 'ff_optimize')
        assert 'min_style sd' in text
        assert 'minimize 1e-05 1e-09 50 5000' in text

    def test_final_structure_and_restart_outputs(self, launch_dir):
        config = _make_config(profile='ff_optimize')
        text = _build(launch_dir, config, 'ff_optimize')
        assert ('write_dump all xyz LAMMPS_ff_optimize.xyz '
                'modify sort id') in text
        assert 'write_data  restart.data' in text
        # dumps are element-independent
        assert 'element' not in text

    def test_thermo_control_aligned_with_reference(self, launch_dir):
        config = _make_config(profile='ff_optimize')
        text = _build(launch_dir, config, 'ff_optimize')
        assert 'thermo_style multi' in text
        assert 'thermo_modify norm yes' in text

    def test_template_execution_commands_dropped(self, launch_dir):
        config = _make_config(profile='ff_optimize')
        text = _build(launch_dir, config, 'ff_optimize')
        assert 'run 999' not in text
        assert 'fix old_name' not in text
        assert 'velocity all create 900' not in text
        # exactly one minimize and one run-free tail
        assert text.count('minimize ') == 1


class TestMdSection:
    """ff_nve/ff_nvt/ff_npt follow the reference MD inputs."""

    def test_nve_fix(self, launch_dir):
        config = _make_config(profile='ff_nve', md_nstep=10)
        text = _build(launch_dir, config, 'ff_nve')
        assert 'fix PYRAI_MD all nve' in text
        assert 'run 10' in text
        assert 'unfix PYRAI_MD' in text

    def test_nvt_fix(self, launch_dir):
        config = _make_config(profile='ff_nvt', temperature=298.15,
                              thermo_damp=100.0)
        text = _build(launch_dir, config, 'ff_nvt')
        assert 'fix PYRAI_MD all nvt temp 298.15 298.15 100' in text

    def test_npt_fix_couples_the_cell_isotropically_by_default(self,
                                                               launch_dir):
        # The framework's md.barostat_mode defaults to 'isotropic', so a run
        # that says nothing scales the volume as one scalar. Before the
        # field reached the skill, LAMMPS deformed three independent axes
        # whatever the input declared, which is how an equilibrated 20-water
        # box ended up with a 1.4 axis ratio instead of a cubic cell.
        config = _make_config(profile='ff_npt', pressure=1.0,
                              pressure_damp=1000.0)
        text = _build(launch_dir, config, 'ff_npt')
        assert ('fix PYRAI_MD all npt temp 298.15 298.15 100 '
                'iso 1 1 1000 drag 2.0') in text
        assert 'fix PYRAI_RECENTER all recenter INIT INIT INIT' in text

    def test_npt_fix_anisotropic_keeps_the_three_axis_form(self, launch_dir):
        # Written as three same-pressure axis clauses rather than the
        # equivalent 'aniso' keyword, so the input text a run produced
        # before the field existed is reproduced byte for byte.
        config = _make_config(profile='ff_npt', pressure=1.0,
                              pressure_damp=1000.0,
                              barostat_mode='anisotropic')
        text = _build(launch_dir, config, 'ff_npt')
        assert ('fix PYRAI_MD all npt temp 298.15 298.15 100 '
                'x 1 1 1000 y 1 1 1000 z 1 1 1000 drag 2.0') in text
        assert ' iso ' not in text

    def test_npt_fix_rejects_an_unknown_coupling(self, launch_dir):
        import pytest
        with pytest.raises(ValueError, match='barostat_mode'):
            _make_config(profile='ff_npt', barostat_mode='tri')

    def test_no_periodic_trajectory_dump(self, launch_dir):
        config = _make_config(profile='ff_nvt')
        text = _build(launch_dir, config, 'ff_nvt')
        # the generated section keeps only the final write_dump
        # outputs and emits no element information
        assert 'traj.xyz' not in text
        assert 'dump_modify' not in text
        assert 'element' not in text

    def test_final_structure_and_restart_outputs(self, launch_dir):
        config = _make_config(profile='ff_nvt')
        text = _build(launch_dir, config, 'ff_nvt')
        assert ('write_dump all xyz LAMMPS_ff_nvt.xyz '
                'modify sort id') in text
        assert ('write_dump all custom LAMMPS_ff_nvt.cond '
                'id x y z vx vy vz modify sort id') in text
        assert 'write_data  restart.data' in text

    def test_timestep_command(self, launch_dir):
        config = _make_config(profile='ff_nvt', md_timestep=0.5)
        text = _build(launch_dir, config, 'ff_nvt')
        assert 'timestep 0.5' in text

    def test_velocity_create_without_system_velocities(self, launch_dir):
        config = _make_config(profile='ff_nve', temperature=900.0,
                              md_seed=12345)
        text = _build(launch_dir, config, 'ff_nve')
        assert 'velocity all create 900 12345' in text

    def test_no_velocity_create_with_system_velocities(self, launch_dir):
        velocities = [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0],
                      [0.0, 0.0, 0.001]]
        config = _make_config(profile='ff_nve', velocities=velocities)
        text = _build(launch_dir, config, 'ff_nve')
        assert 'velocity all create' not in text

    def test_template_execution_commands_dropped(self, launch_dir):
        config = _make_config(profile='ff_nvt', md_nstep=50)
        text = _build(launch_dir, config, 'ff_nvt')
        assert 'run 999' not in text
        assert 'fix old_name' not in text
        assert 'velocity all create 900' not in text
        assert text.count('timestep') == 1
        assert text.count('run ') == 1


class TestDataFileVelocities:
    """The Velocities section follows the system velocities."""

    def test_velocities_written_into_data_file(self, launch_dir):
        # Bohr/au velocities; 1 au = 21.8769126364 Angstrom/fs (real)
        velocities = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
        config = _make_config(profile='ff_nvt', velocities=velocities)
        _build(launch_dir, config, 'ff_nvt')
        data_text = (launch_dir / 'LAMMPS.data').read_text()
        assert 'Velocities' in data_text
        # 0.1 Bohr/au = 2.1876912636 Angstrom/fs in real units
        assert '2.1876912636' in data_text

    def test_existing_velocities_section_replaced(self, launch_dir):
        data_text = (launch_dir / 'LAMMPS.data').read_text()
        data_text += '\nVelocities\n\n1 9.9 9.9 9.9\n2 9.9 9.9 9.9\n3 9.9 9.9 9.9\n'
        (launch_dir / 'LAMMPS.data').write_text(data_text)
        velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        config = _make_config(profile='ff_nvt', velocities=velocities)
        _build(launch_dir, config, 'ff_nvt')
        new_data = (launch_dir / 'LAMMPS.data').read_text()
        assert new_data.count('Velocities') == 1
        assert '9.9 9.9 9.9' not in new_data


class TestDataFileBox:
    """The simulation box follows the framework cell (PBC docs 3.3)."""

    def test_periodic_box_resized_from_lattice(self, launch_dir):
        # Template box is 10 A; a 20 A periodic cell must resize it.
        config = _make_config(
            profile='ff_energy', pbc=[1, 1, 1],
            lattice_vector=[[20.0, 0.0, 0.0],
                            [0.0, 20.0, 0.0],
                            [0.0, 0.0, 20.0]])
        _build(launch_dir, config, 'ff_energy')
        data = (launch_dir / 'LAMMPS.data').read_text()
        assert '0.0000000000 20.0000000000 xlo xhi' in data
        assert '0.0000000000 20.0000000000 ylo yhi' in data
        assert '0.0000000000 20.0000000000 zlo zhi' in data

    def test_non_periodic_keeps_template_box(self, launch_dir):
        config = _make_config(profile='ff_energy')
        _build(launch_dir, config, 'ff_energy')
        data = (launch_dir / 'LAMMPS.data').read_text()
        assert '0.0 10.0 xlo xhi' in data

    def test_triclinic_tilt_line_written(self, launch_dir):
        config = _make_config(
            profile='ff_energy', pbc=[1, 1, 1],
            lattice_vector=[[10.0, 0.0, 0.0],
                            [2.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
        _build(launch_dir, config, 'ff_energy')
        data = (launch_dir / 'LAMMPS.data').read_text()
        # xy = B[1][0] = 2.0, xz = yz = 0; the line is self-labelled
        # with the trailing 'xy xz yz' tokens.
        assert ('2.0000000000 0.0000000000 0.0000000000 xy xz yz'
                in data)

    def test_labelled_tilt_template_rewritten_not_duplicated(
            self, launch_dir):
        # A triclinic template already carrying a labelled tilt line is
        # detected and rewritten in place (no second tilt line).
        data = (launch_dir / 'LAMMPS.data').read_text()
        data = data.replace(
            '0.0 10.0 zlo zhi',
            '0.0 10.0 zlo zhi\n1.000 0.500 0.250 xy xz yz')
        (launch_dir / 'LAMMPS.data').write_text(data)
        config = _make_config(
            profile='ff_energy', pbc=[1, 1, 1],
            lattice_vector=[[10.0, 0.0, 0.0],
                            [2.0, 10.0, 0.0],
                            [0.5, 0.0, 10.0]])
        _build(launch_dir, config, 'ff_energy')
        out = (launch_dir / 'LAMMPS.data').read_text()
        assert out.count('xy xz yz') == 1
        assert '2.0000000000 0.5000000000 0.0000000000 xy xz yz' in out

    def test_upper_triangular_lattice_rejected(self, launch_dir):
        # The data-file box can only express a lower-triangular
        # lattice; an upstream convention violation must fail fast
        # instead of silently dropping the upper components.
        import pytest
        config = _make_config(
            profile='ff_energy', pbc=[1, 1, 1],
            lattice_vector=[[10.0, 1.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
        with pytest.raises(RuntimeError, match='lower-triangular'):
            _build(launch_dir, config, 'ff_energy')

    def test_image_flags_preserved(self, launch_dir):
        # full style WITH trailing image flags: id mol type q x y z ix iy iz
        (launch_dir / 'LAMMPS.data').write_text(_FLAGGED_DATA)
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        config = _make_config(profile='ff_energy', coordinates=coords)
        _build(launch_dir, config, 'ff_energy')
        out = (launch_dir / 'LAMMPS.data').read_text().splitlines()
        atom_lines = [l for l in out
                      if l.strip() and l.split()[0].isdigit()
                      and len(l.split()) == 10]
        assert len(atom_lines) == 3
        # atom 1 coords replaced, image flags '0 0 0' preserved
        t1 = atom_lines[0].split()
        assert float(t1[4]) == 1.0 and float(t1[5]) == 2.0 \
            and float(t1[6]) == 3.0
        assert t1[7:] == ['0', '0', '0']
        # atom 2 keeps its '-1 0 2' image flags
        t2 = atom_lines[1].split()
        assert float(t2[4]) == 4.0
        assert t2[7:] == ['-1', '0', '2']

    def test_out_of_order_ids_mapped_by_id(self, launch_dir):
        # Atoms listed out of order: coordinates must map by id-1.
        data = (
            'LAMMPS Description\n\n3  atoms\n\n'
            '0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n'
            'Masses\n\n1 15.999\n2 1.008\n\n'
            'Atoms # full\n\n'
            '3 1 2 0.417 0.0 0.0 0.0\n'
            '1 1 1 -0.834 0.0 0.0 0.0\n'
            '2 1 2 0.417 0.0 0.0 0.0\n\n'
            'Bonds\n\n1 1 1 2\n2 1 1 3\n'
        )
        (launch_dir / 'LAMMPS.data').write_text(data)
        coords = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        config = _make_config(profile='ff_energy', coordinates=coords)
        _build(launch_dir, config, 'ff_energy')
        out = (launch_dir / 'LAMMPS.data').read_text().splitlines()
        by_id = {}
        for l in out:
            t = l.split()
            if len(t) == 7 and t[0].isdigit():
                by_id[int(t[0])] = float(t[4])
        # id 1 -> coords[0][0]=1.0, id 3 -> coords[2][0]=3.0
        assert by_id[1] == 1.0
        assert by_id[2] == 2.0
        assert by_id[3] == 3.0

    def test_out_of_range_id_raises(self, launch_dir):
        data = (
            'LAMMPS Description\n\n2  atoms\n\n'
            '0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n'
            'Masses\n\n1 15.999\n\n'
            'Atoms # full\n\n'
            '1 1 1 -0.834 0.0 0.0 0.0\n'
            '5 1 1 -0.834 0.0 0.0 0.0\n\n'
        )
        (launch_dir / 'LAMMPS.data').write_text(data)
        config = _make_config(
            profile='ff_energy', atoms=['O', 'H'],
            coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(RuntimeError):
            _build(launch_dir, config, 'ff_energy')


class TestSinglePointRegression:
    """ff_energy / ff_gradient keep the legacy single-point behavior."""

    def test_single_point_run_zero(self, launch_dir):
        config = _make_config(profile='ff_energy')
        text = _build(launch_dir, config, 'ff_energy')
        assert 'run 0' in text
        assert 'thermo_style custom step pe etotal' in text
        # single point keeps the template body (legacy behavior)
        assert 'run 999' in text

    def test_gradient_force_dump(self, launch_dir):
        config = _make_config(profile='ff_gradient')
        text = _build(launch_dir, config, 'ff_gradient')
        assert ('write_dump all custom LAMMPS_force.force '
                'fx fy fz modify sort id') in text
        # the force dump follows 'run 0' so the forces are computed
        assert text.index('run 0') < text.index('LAMMPS_force.force')

    def test_energy_and_geometry_are_written_as_text(self, launch_dir):
        # The result of the step is a file per quantity, so reading the
        # calculation directory is reading the calculation.
        config = _make_config(profile='ff_energy')
        text = _build(launch_dir, config, 'ff_energy')
        assert 'variable pyrai_pe equal pe' in text
        assert 'print "${pyrai_pe}" file LAMMPS_ff_energy.energy' in text
        assert ('write_dump all custom LAMMPS_ff_energy.coords '
                'x y z modify sort id') in text
        # an energy-only profile writes no force file
        assert 'LAMMPS_ff_energy.force' not in text


class TestKspaceReissue:
    """kspace_style is replayed after read_data.

    A variable-cell (NPT) step rewrites the data-file box with tilt
    factors; LAMMPS then aborts with "Must redefine kspace_style after
    changing to triclinic box" unless the kspace command set earlier in
    the in.init template is re-issued after read_data.
    """

    @pytest.fixture
    def init_dir(self, launch_dir):
        (launch_dir / 'LAMMPS.in.init').write_text(
            'units real\n'
            'atom_style full\n'
            'kspace_style pppm 1e-4\n'
            'pair_modify mix arithmetic\n')
        return launch_dir

    def test_kspace_reissued_after_read_data(self, init_dir):
        config = _make_config(
            profile='ff_gradient', pbc=[1, 1, 1],
            # Mirrors the template's own token so this test keeps asking
            # only where the line is replayed, not what it says.
            ewald_accuracy='1e-4',
            lattice_vector=[[10.0, 0.0, 0.0],
                            [2.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
        text = _build(init_dir, config, 'ff_gradient')
        # the replayed command sits AFTER read_data in the script
        assert ('kspace_style pppm 1e-4' in text)
        assert text.index('read_data') < text.index('kspace_style pppm 1e-4')
        # the include directive is not inlined, so the command appears
        # exactly once (the replay) in the generated script
        assert text.count('kspace_style pppm 1e-4') == 1

    def test_accuracy_token_is_the_configs_not_the_templates(self, init_dir):
        # The replayed line keeps the template's reciprocal STYLE - the two
        # styles are different methods and the skill does not choose between
        # them - while the accuracy token is the configuration's. A case that
        # declares nothing therefore runs the converged default rather than
        # whatever looseness the template happens to carry.
        config = _make_config(
            profile='ff_gradient', pbc=[1, 1, 1], ewald_accuracy='1e-8',
            lattice_vector=[[10.0, 0.0, 0.0],
                            [2.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
        text = _build(init_dir, config, 'ff_gradient')
        assert 'kspace_style pppm 1e-8' in text
        assert '1e-4' not in text

    def test_default_replaces_a_loose_template_without_asking(
            self, launch_dir):
        # Non-identity witness on the other half: the shipped-style template
        # spelling for ewald at 1e-4 is overwritten by the default, so an
        # undeclared periodic case cannot reach the engine at the accuracy
        # that left 5.3e-4 of the probe's V unconverged.
        (launch_dir / 'LAMMPS.in.init').write_text(
            'units real\n'
            'atom_style full\n'
            'kspace_style ewald 0.0001\n'
            'pair_modify mix arithmetic\n')
        config = _make_config(profile='ff_gradient', pbc=[1, 1, 1])
        text = _build(launch_dir, config, 'ff_gradient')
        assert 'kspace_style ewald 1e-8' in text
        assert '0.0001' not in text

    def test_no_reissue_when_template_lacks_kspace(self, launch_dir):
        # no LAMMPS.in.init staged -> nothing to replay
        config = _make_config(profile='ff_gradient')
        text = _build(launch_dir, config, 'ff_gradient')
        assert 'kspace_style' not in text


class TestAccuracyValueIsChecked:
    """The token is validated where it is read, not where the engine chokes.

    A value LAMMPS cannot parse would abort inside `kspace_style` after the
    run has paid for the topology and the neighbor lists, with an error that
    names neither the key nor the file it came from.

    There is no way to ask for "keep whatever the template declared": the
    config merge re-applies a falsy value from the default table, so an empty
    string arrives here already rewritten to `1e-8` and is indistinguishable
    from an absent key. That is the adapter's standing rule, measured rather
    than assumed, and it is why a case that wants a loose sum writes the
    loose number.
    """

    @pytest.mark.parametrize('value', ['abc', '1.0', '0', '-1e-8'])
    def test_refused_before_any_deck_is_written(self, launch_dir, value):
        with pytest.raises(ValueError, match='ewald_accuracy'):
            _make_config(profile='ff_energy', ewald_accuracy=value)

    def test_empty_value_is_the_default_not_an_inheritance(
            self, launch_dir):
        # The witness for the paragraph above: '' does not reach the checker,
        # it is replaced on the way in. Asserted rather than left implicit,
        # because it decides how the shipped classical cases must declare
        # their accuracy.
        assert (_make_config(profile='ff_energy',
                             ewald_accuracy='').ewald_accuracy == '1e-8')

    def test_spelling_is_normalized(self, launch_dir):
        assert (_make_config(profile='ff_energy',
                             ewald_accuracy=' 1E-6 ').ewald_accuracy
                == '1e-6')

    def test_yaml_types_the_number_two_ways(self, launch_dir):
        # `0.0001` is a float to YAML and `1e-4` is a string, so a single key
        # arrives under two Python types depending on how the case spells the
        # number its template already carries. Both must reach the same token:
        # before the field folded it, the float died in model construction
        # while the string ran, which is a confusing failure to hand whoever
        # edits a template's spelling.
        assert (_make_config(profile='ff_energy',
                             ewald_accuracy=0.0001).ewald_accuracy
                == '0.0001')
        assert (_make_config(profile='ff_energy',
                             ewald_accuracy='0.0001').ewald_accuracy
                == '0.0001')

    def test_absent_key_answers_with_the_converged_default(self, launch_dir):
        assert _make_config(profile='ff_energy').ewald_accuracy == '1e-8'


class TestSubsystemInput:
    """A selected subsystem is generated as a non-periodic calculation.

    The multi-center difference term `L(c)` has to be the center's own
    internal energy: kept periodic, its Ewald self-image term would be
    subtracted along with it (measured on the shipped water template:
    0.280090 kcal/mol per center, dev/methods/PBC_DEVELOPMENT.md). So the
    subsystem gets `boundary f f f`, no kspace command anywhere, and the
    real-space pair style its template declares the cutoffs for.
    """

    _SUB_DATA = (
        'LAMMPS Description\n\n3  atoms\n2  bonds\n0  angles\n'
        '0  dihedrals\n0  impropers\n\n2  atom types\n1  bond types\n'
        '0  angle types\n0  dihedral types\n0  improper types\n\n'
        '0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n'
        'Masses\n\n1 15.999\n2 1.008\n\nAtoms\n\n'
        '1 1 1 -0.834 0.0000000000 0.0000000000 0.1173000000\n'
        '2 1 2 0.417 0.0000000000 0.7572000000 -0.4692000000\n'
        '3 1 2 0.417 0.0000000000 -0.7572000000 -0.4692000000\n\n'
        'Bonds\n\n1 1 1 2\n2 1 1 3\n')

    _SUB_INIT = (
        'units real\n'
        'atom_style full\n'
        'bond_style hybrid harmonic\n'
        'pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0\n'
        'kspace_style pppm 1e-4\n'
        'pair_modify mix arithmetic\n'
        'special_bonds amber\n')

    _SUB_SETTINGS = (
        'pair_coeff 1 1 lj/charmm/coul/long 0.102 3.188\n'
        'pair_coeff 2 2 lj/charmm/coul/long 0.00 0.4000\n'
        'pair_coeff 1 2 lj/charmm/coul/long 0.00 1.7753\n'
        'bond_coeff 1 harmonic 450.0 0.9572\n')

    @pytest.fixture
    def sub_dir(self, tmp_path, monkeypatch):
        """A system-level template whose pair style needs an Ewald sum."""
        (tmp_path / 'LAMMPS.in').write_text(TEMPLATE_IN)
        (tmp_path / 'LAMMPS.data').write_text(self._SUB_DATA)
        (tmp_path / 'LAMMPS.in.init').write_text(self._SUB_INIT)
        (tmp_path / 'LAMMPS.in.settings').write_text(self._SUB_SETTINGS)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    @staticmethod
    def _written(directory, config, profile, kind='sub'):
        """Generate the whole input set from the compressed model and read
        the script back."""
        calcdir = str(directory / kind)
        os.makedirs(calcdir)
        model = build_ff_model(config, profile)
        write_input_file(config, compile_workflow(profile), calcdir, model)
        basename = output_basename('LAMMPS', profile)
        return (model,
                open(os.path.join(calcdir, f'{basename}.in')).read(),
                open(os.path.join(calcdir, 'LAMMPS.in.init')).read(),
                open(os.path.join(calcdir, 'LAMMPS.in.settings')).read())

    def test_subsystem_runs_without_a_box(self, sub_dir):
        config = _make_config(profile='ff_gradient', pbc=[0, 0, 0],
                              atom_selection=[1, 2, 3], natom=3)
        model, script, init_text, _settings = self._written(
            sub_dir, config, 'ff_gradient')
        assert is_subsystem(model)
        assert 'boundary f f f' in script
        # The engine refuses `boundary` once the box exists, so it has to
        # precede read_data.
        assert script.index('boundary f f f') < script.index('read_data')
        assert 'kspace_style' not in script
        assert 'kspace_style' not in init_text

    def test_subsystem_pair_style_keeps_the_declared_cutoffs(self, sub_dir):
        config = _make_config(profile='ff_gradient', pbc=[0, 0, 0],
                              atom_selection=[1, 2, 3], natom=3)
        _model, _script, init_text, settings = self._written(
            sub_dir, config, 'ff_gradient')
        # The template's own numbers, in the positions that put the
        # declared cutoff in charge (measured argument-by-argument,
        # service/builder/ff_model.COULOMB_REAL_SPACE).
        assert 'pair_style hybrid lj/cut/coul/cut 9.0 10.0' in init_text
        assert 'lj/charmm/coul/long' not in init_text
        assert settings.count('pair_coeff') == 3
        assert 'lj/charmm/coul/long' not in settings

    def test_a_subsystem_is_given_the_fixed_container(self, sub_dir):
        # A cell is present (the run is periodic for the rest of the plan)
        # but the region task is handed pbc off, so the framework lattice must
        # never become this subsystem's box. What the box does get is the one
        # container a non-periodic run is evaluated in, centred on the origin
        # and independent of where this step's cluster happens to be: a box
        # fitted to the atoms at load time is a box they walk out of under the
        # integrator, and with more than one MPI rank the escaped atom is not
        # refused but lost - `Bond atoms 1 3 missing on proc 0`, measured on
        # the deployment for the shipped two-center example.
        config = _make_config(
            profile='ff_gradient', pbc=[0, 0, 0],
            atom_selection=[1, 2, 3], natom=3,
            cell=[20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
            lattice_vector=[[20.0, 0.0, 0.0], [0.0, 20.0, 0.0],
                            [0.0, 0.0, 20.0]])
        _model, _script, _init, _settings = self._written(
            sub_dir, config, 'ff_gradient')
        data = open(os.path.join(str(sub_dir), 'sub',
                                 'LAMMPS.data')).read()
        bounds = self._bounds(data)
        assert set(bounds) == {'xlo xhi', 'ylo yhi', 'zlo zhi'}
        half = _SUBSYSTEM_BOX_HALF
        assert all(bound == (-half, half)
                   for bound in bounds.values()), bounds
        # Neither the template's own lengths nor the cell the framework
        # handed reached the container.
        assert '0.0 10.0 xlo xhi' not in data
        assert '20.0' not in data
        rows = self._atom_rows(data)
        for axis in range(3):
            values = [row[axis] for row in rows]
            lo, hi = list(bounds.values())[axis]
            assert lo <= min(values) and max(values) <= hi, (axis, bounds)

    @staticmethod
    def _bounds(text):
        """The three bound lines of a data file, keyed by their label."""
        bounds = {}
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) == 4 and ' '.join(tokens[2:]) in (
                    'xlo xhi', 'ylo yhi', 'zlo zhi'):
                bounds[' '.join(tokens[2:])] = (float(tokens[0]),
                                                float(tokens[1]))
        return bounds

    @staticmethod
    def _atom_rows(text):
        """The Atoms rows of a data file as floats, in file order."""
        out, keep = [], False
        for line in text.splitlines():
            head = line.split('#')[0].strip()
            if head == 'Atoms':
                keep = True
                continue
            if keep:
                if not head:
                    continue
                if not head.split()[0].isdigit():
                    break
                tokens = head.split()
                out.append([float(value) for value in tokens[4:7]])
        return out

    def test_a_cluster_outside_the_container_is_refused(self, sub_dir):
        # The container is a premise about the system being a molecular
        # cluster of this cell, and the refusal is the other half of that: a
        # geometry further out than the box is suspect, and growing the box to
        # swallow it would let a broken rebuild run to completion and report a
        # number nobody asked for.
        config = _make_config(
            profile='ff_gradient', pbc=[0, 0, 0],
            atom_selection=[1, 2, 3], natom=3,
            coordinates=[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692],
                         [70.0, -0.7572, -0.4692]])
        with pytest.raises(RuntimeError, match='container'):
            self._written(sub_dir, config, 'ff_gradient', kind='far')

    def test_whole_system_keeps_the_long_range_declaration(self, sub_dir):
        # The control: with no selection nothing about the periodic
        # treatment moves, which is the half of the change that the 24
        # golden digests cannot see on their own.
        config = _make_config(profile='ff_gradient', pbc=[1, 1, 1],
                              ewald_accuracy='1e-4',
                              lattice_vector=[[10.0, 0.0, 0.0],
                                              [0.0, 10.0, 0.0],
                                              [0.0, 0.0, 10.0]])
        model, script, init_text, settings = self._written(
            sub_dir, config, 'ff_gradient', kind='whole')
        assert not is_subsystem(model)
        assert 'boundary' not in script
        assert 'kspace_style pppm 1e-4' in script
        assert 'lj/charmm/coul/long 9.0 10.0 10.0' in init_text
        assert 'lj/cut/coul/cut' not in settings

    def test_resident_and_subprocess_agree_on_the_subsystem(self, sub_dir):
        # One producer for both paths: the package a resident instance loads
        # and the files the subprocess path writes must be the same bytes,
        # including the subsystem's boundary decision and its empty kspace
        # line.
        config = _make_config(profile='ff_gradient', pbc=[0, 0, 0],
                              atom_selection=[1, 2, 3], natom=3)
        workflow = compile_workflow('ff_gradient')
        model = build_ff_model(config, 'ff_gradient')
        subprocess_dir = str(sub_dir / 'subprocess')
        os.makedirs(subprocess_dir)
        write_input_file(config, workflow, subprocess_dir, model)
        package_dir = str(sub_dir / 'resident')
        os.makedirs(package_dir)
        package = write_static_input_files(config, workflow, package_dir,
                                          model)
        assert package['kspace'] == ''
        for name in ('LAMMPS.in.init', 'LAMMPS.in.settings',
                     'LAMMPS.data'):
            assert (open(os.path.join(package_dir, name)).read()
                    == open(os.path.join(subprocess_dir,
                                         name)).read()), name


class TestStressFileOutput:
    """A stress request adds the pressure compute and a file write."""

    def test_stress_writes_dedicated_file(self, launch_dir):
        config = _make_config(profile='ff_gradient', stress=True)
        text = _build(launch_dir, config, 'ff_gradient')
        assert 'compute Pconf all pressure NULL virial' in text
        assert 'variable pyrai_sxx equal c_Pconf[1]' in text
        assert 'file LAMMPS_force.stress' in text
        # the print follows 'run 0' so the compute is evaluated
        assert text.index('run 0') < text.index('LAMMPS_force.stress')

    def test_no_stress_leaves_input_unchanged(self, launch_dir):
        config = _make_config(profile='ff_gradient')
        text = _build(launch_dir, config, 'ff_gradient')
        assert 'compute Pconf' not in text
        assert 'LAMMPS_force.stress' not in text
        assert 'thermo_style custom step pe etotal' in text


class TestNoMainTemplate:
    """Without '{project}.in' the input script is fully generated."""

    @pytest.mark.parametrize('profile', [
        'ff_energy', 'ff_gradient', 'ff_optimize',
        'ff_nve', 'ff_nvt', 'ff_npt',
    ])
    def test_input_generated_without_template(self, data_only_dir,
                                              profile):
        config = _make_config(profile=profile)
        text = _build(data_only_dir, config, profile)
        # header loads the topology (in.init/in.settings absent)
        assert 'read_data "LAMMPS.data"' in text
        assert 'include' not in text
        # workflow tails are generated in full
        if profile == 'ff_optimize':
            assert 'minimize ' in text
        elif profile.startswith('ff_n'):
            assert 'run ' in text
            assert 'LAMMPS_%s.cond' % profile in text
        else:
            assert 'run 0' in text


class TestPerTaskInputNaming:
    """The generated input shares the dump output basename."""

    @pytest.mark.parametrize('profile, expected', [
        ('ff_energy', 'LAMMPS_ff_energy.in'),
        ('ff_gradient', 'LAMMPS_force.in'),
        ('ff_optimize', 'LAMMPS_ff_optimize.in'),
        ('ff_nve', 'LAMMPS_ff_nve.in'),
        ('ff_nvt', 'LAMMPS_ff_nvt.in'),
        ('ff_npt', 'LAMMPS_ff_npt.in'),
    ])
    def test_input_file_follows_output_basename(self, launch_dir,
                                                profile, expected):
        config = _make_config(profile=profile)
        _build(launch_dir, config, profile)
        assert (launch_dir / expected).exists()
        # the user template keeps its original name and content
        assert (launch_dir / 'LAMMPS.in').read_text() == TEMPLATE_IN


SHAKE_SETTINGS = (
    'pair_coeff 1 1 lj/charmm/coul/long 0.1 3.0\n'
    'group tip3p type 1 2\n'
    'fix fShakeTIP3P tip3p shake 0.0001 10 100 b 1 a 1\n'
)


class TestShakeInvariant:
    """A constraint must be applied by whoever integrates the atoms."""

    def test_framework_profiles_reject_shake(self, launch_dir):
        # ff_energy / ff_gradient hand one force evaluation per step to
        # the framework integrator, which propagates and constrains on
        # its own account. The refusal has to name where the declaration
        # goes: a message that only says no teaches the user to delete
        # the keyword instead of moving it.
        for profile in ('ff_energy', 'ff_gradient'):
            with pytest.raises(ValueError,
                               match='lammps_shake=1 is inconsistent') as exc:
                _make_config(profile=profile, lammps_shake=1)
            message = str(exc.value)
            assert "'constraint' block" in message
            assert 'constraint engine' not in message
            assert profile in message

    def test_engine_profiles_accept_shake(self, launch_dir):
        for profile in ('ff_optimize', 'ff_nve', 'ff_nvt', 'ff_npt'):
            config = _make_config(profile=profile, lammps_shake=1)
            assert int(config.environment.lammps_shake) == 1


class TestModelDrivenShakeSection:
    """With a model the constraint state is decided once, in the model."""

    def test_disabled_constraint_emits_neither_fix_nor_unfix(self, launch_dir):
        config = _make_config(profile='ff_gradient')
        model = {'constraints': {'enabled': False}}
        assert _build_shake_section(config, str(launch_dir), model) == ''

    def test_enabled_constraint_travels_with_the_settings(self, launch_dir):
        config = _make_config(profile='ff_nvt', lammps_shake=1)
        model = {'constraints': {'enabled': True, 'decl': 'tip3p shake 1'}}
        assert _build_shake_section(config, str(launch_dir), model) == ''

    def test_enabled_without_a_fix_falls_back_to_keywords(self, launch_dir):
        config = _make_config(profile='ff_nvt', lammps_shake=1,
                              shake_keywords='fix myShake all shake 1 1 1')
        model = {'constraints': {'enabled': True, 'decl': ''}}
        assert _build_shake_section(config, str(launch_dir), model) == \
            'fix myShake all shake 1 1 1\n'

    def test_enabled_without_any_definition_raises(self, launch_dir):
        config = _make_config(profile='ff_nvt', lammps_shake=1)
        model = {'constraints': {'enabled': True, 'decl': ''}}
        with pytest.raises(RuntimeError, match='shake_keywords is empty'):
            _build_shake_section(config, str(launch_dir), model)

    def test_legacy_path_still_unfixes_the_template_fix(self, launch_dir):
        (launch_dir / 'LAMMPS.in.settings').write_text(SHAKE_SETTINGS)
        config = _make_config(profile='ff_gradient')
        assert _build_shake_section(config, str(launch_dir), None) == \
            'unfix fShakeTIP3P\n'
