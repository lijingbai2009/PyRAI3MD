"""
Unit tests for lammps_skill result collection (output_parser).

Uses output formats demo-verified against lammps_demo:
  - thermo_style multi log entries ('PotEng = ...', per-atom
    normalized) for the internal-iteration profiles
  - write_dump xyz final-structure frames ('{project}_{profile}.xyz')
    for the fin_coord channel
  - write_dump custom cond frames ('{project}_{profile}.cond',
    id x y z vx vy vz) for the fin_velocity channel of the MD
    profiles
  - write_dump custom force frames without the id column
    ('{title}.force', fx fy fz + modify sort id)
  - an energy file printed by the step itself ('{title}.energy'), which is
    where a single point reads its energy from; the thermo table log is the
    internal-iteration profiles' channel
These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_output_parser.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.parser.output_parser import parse_output
from ._result_units import apply_result_units
from ..service.parser.log_parser import parse_log
from ..service.utils import ENERGY_TO_HARTREE
from ..service.utils import FORCE_TO_HARTREE_PER_BOHR
from ..service.utils import LAMMPS_VELOCITY_TO_AU
from ..service.utils import PRESSURE_TO_BAR
from ..service.utils import step_data_files
from ..service.utils import output_basename
from ..service.workflow import compile_workflow

NATOM = 3

# Multi-style thermo log excerpt (min demo format, per-atom norm)
MULTI_LOG = [
    'LAMMPS (23 Jun 2022 - Update 1)',
    '------------ Step              0 ----- CPU =            0 (sec) -------------',
    'PotEng   =        -0.3758 E_bond   =         0.0000 E_angle  =         0.0000 ',
    '------------ Step            100 ----- CPU =   0.03676958 (sec) -------------',
    'PotEng   =        -1.5679 E_bond   =         0.0000 E_angle  =         0.0000 ',
    'Loop time of 0.284127 on 128 procs for 1000 steps with 60 atoms',
]

# write_dump xyz final structure frame (element-independent: the
# leading column is the atom type id, not an element symbol)
FINAL_XYZ = (
    '3\n'
    ' Atoms. Timestep: 100\n'
    '1 5.36547 3.97334 3.11112\n'
    '2 5.03173 3.18409 3.53766\n'
    '2 5.95927 4.36082 3.75416\n'
)

FINAL_COORDS = [
    [5.36547, 3.97334, 3.11112],
    [5.03173, 3.18409, 3.53766],
    [5.95927, 4.36082, 3.75416],
]

# write_dump custom cond frame (7 columns id x y z vx vy vz); the
# coordinates match the xyz dump, rows given out of id order to
# exercise the defensive id sort
FINAL_COND = (
    'ITEM: TIMESTEP\n'
    '100\n'
    'ITEM: NUMBER OF ATOMS\n'
    '3\n'
    'ITEM: BOX BOUNDS pp pp pp\n'
    '0 10\n'
    '0 10\n'
    '0 10\n'
    'ITEM: ATOMS id x y z vx vy vz\n'
    '3 5.95927 4.36082 3.75416 0.0 -0.002 0.001\n'
    '1 5.36547 3.97334 3.11112 0.001 0.002 0.003\n'
    '2 5.03173 3.18409 3.53766 -0.001 0.0 0.0\n'
)

# Native LAMMPS velocities (real: Angstrom/fs) in atom-id order
FINAL_VELOCITIES = [
    [0.001, 0.002, 0.003],
    [-0.001, 0.0, 0.0],
    [0.0, -0.002, 0.001],
]

# write_dump custom force frame without the id column (force demo
# format: fx fy fz + modify sort id)
FORCE_DUMP_NO_ID = (
    'ITEM: TIMESTEP\n'
    '0\n'
    'ITEM: NUMBER OF ATOMS\n'
    '3\n'
    'ITEM: BOX BOUNDS pp pp pp\n'
    '0.0000000000000000e+00 1.0000000000000000e+01\n'
    '0.0000000000000000e+00 1.0000000000000000e+01\n'
    '0.0000000000000000e+00 1.0000000000000000e+01\n'
    'ITEM: ATOMS fx fy fz\n'
    '10.2244 -3.61294 -4.59281\n'
    '-0.618265 -2.44341 3.71219\n'
    '-5.64704 -0.323485 -0.514467\n'
)

# Legacy single-point thermo table (skill-injected custom style)
ENERGY_VALUE = -10.502731791

TABLE_LOG = [
    'Step PotEng TotEng ',
    '       0   -10.502731791   -10.502731791 ',
    'Loop time of 0.01 on 1 procs for 0 steps with 3 atoms',
]


def _make_config(**kwargs):
    raw = {
        'atoms': ['O', 'H', 'H'],
        'coordinates': [
            [0.000000000, 0.000000000, 0.117300000],
            [0.000000000, 0.757200000, -0.469200000],
            [0.000000000, -0.757200000, -0.469200000],
        ],
        'profile': 'ff_energy',
        'method': 'ff',
        'lammps_project': 'LAMMPS',
        'lammps_units': 'real',
    }
    raw.update(kwargs)
    # Build + normalize only: the parser tests carry their own
    # output files and must not touch the launch directory
    return normalize(build_config(raw))


def _write_outputs(tmp_path, log_lines=None, xyz=None, cond=None,
                   force=None, profile='ff_energy', stress_file=None,
                   energy=None):
    """Lay out a calculation directory the way the builder writes one.

    The single-point names come from the same helper the writer uses, so a
    test cannot pass by reading a file the run never produced.
    """
    project = 'LAMMPS'
    basename = output_basename(project, profile)
    names = step_data_files(basename,
                            gradient=profile in ('ff_gradient',),
                            stress=stress_file is not None)
    if log_lines is not None:
        (tmp_path / f'{basename}.log').write_text(
            '\n'.join(log_lines) + '\n')
    if xyz is not None:
        (tmp_path / f'{project}_{profile}.xyz').write_text(xyz)
    if cond is not None:
        (tmp_path / f'{project}_{profile}.cond').write_text(cond)
    if force is not None:
        (tmp_path / names['force']).write_text(force)
    if stress_file is not None:
        (tmp_path / names['stress']).write_text(stress_file)
    if energy is not None:
        (tmp_path / names['energy']).write_text(f'{energy}\n')


class TestLogEnergyParsing:
    """parse_log covers the table and the multi thermo formats."""

    def test_table_format(self):
        assert parse_log(TABLE_LOG) == pytest.approx(-10.502731791)

    def test_multi_format(self):
        assert parse_log(MULTI_LOG) == pytest.approx(-1.5679)

    def test_latest_format_wins(self):
        # table first, multi later -> the multi value is the newest
        lines = TABLE_LOG + [''] + MULTI_LOG
        assert parse_log(lines) == pytest.approx(-1.5679)

    def test_no_energy_raises(self):
        with pytest.raises(RuntimeError, match='no thermo output'):
            parse_log(['LAMMPS (23 Jun 2022)'])


class TestInternalIterationResults:
    """ff_optimize / MD profiles collect energy and fin_coord."""

    @pytest.mark.parametrize('profile', ['ff_optimize', 'ff_nve',
                                         'ff_nvt', 'ff_npt'])
    def test_energy_norm_restored_to_total(self, tmp_path, profile):
        # multi log PotEng is a per-atom average (thermo_modify norm)
        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       cond=FINAL_COND, profile=profile)
        config = _make_config(profile=profile)
        workflow = compile_workflow(profile)
        results = parse_output(config, str(tmp_path), workflow)
        expected_total = -1.5679 * NATOM
        assert results['energy'][0] == pytest.approx(expected_total)

    @pytest.mark.parametrize('profile', ['ff_optimize', 'ff_nvt'])
    def test_fin_coord_from_xyz_dump(self, tmp_path, profile):
        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       cond=FINAL_COND, profile=profile)
        config = _make_config(profile=profile)
        workflow = compile_workflow(profile)
        results = parse_output(config, str(tmp_path), workflow)
        assert results['fin_coord'].shape == (NATOM, 3)
        assert np.allclose(results['fin_coord'], FINAL_COORDS)

    def test_multi_frame_xyz_takes_last(self, tmp_path):
        first_frame = (
            '3\n'
            ' Atoms. Timestep: 0\n'
            '1 0.0 0.0 0.0\n'
            '2 0.0 0.0 0.96\n'
            '2 0.0 0.96 0.0\n'
        )
        _write_outputs(tmp_path, log_lines=MULTI_LOG,
                       xyz=first_frame + FINAL_XYZ, cond=FINAL_COND,
                       profile='ff_nvt')
        config = _make_config(profile='ff_nvt')
        workflow = compile_workflow('ff_nvt')
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['fin_coord'], FINAL_COORDS)

    def test_missing_xyz_raises(self, tmp_path):
        _write_outputs(tmp_path, log_lines=MULTI_LOG, profile='ff_nvt')
        config = _make_config(profile='ff_nvt')
        workflow = compile_workflow('ff_nvt')
        with pytest.raises(RuntimeError, match='xyz dump file not found'):
            parse_output(config, str(tmp_path), workflow)


class TestCondVelocityChannel:
    """MD profiles report fin_velocity from the cond dump."""

    def _run_md(self, tmp_path, cond=FINAL_COND):
        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       cond=cond, profile='ff_nvt')
        config = _make_config(profile='ff_nvt')
        workflow = compile_workflow('ff_nvt')
        return parse_output(config, str(tmp_path), workflow)

    def test_fin_velocity_converted_to_au(self, tmp_path):
        results = self._run_md(tmp_path)
        expected = np.array(FINAL_VELOCITIES) \
            * LAMMPS_VELOCITY_TO_AU['real']
        assert results['fin_velocity'].shape == (NATOM, 3)
        assert np.allclose(results['fin_velocity'], expected)

    def test_cond_rows_sorted_by_id(self, tmp_path):
        # FINAL_COND rows are out of id order; the coordinates must
        # still match the id-sorted xyz frame
        results = self._run_md(tmp_path)
        assert np.allclose(results['fin_coord'], FINAL_COORDS)

    def test_missing_cond_raises(self, tmp_path):
        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       profile='ff_nvt')
        config = _make_config(profile='ff_nvt')
        workflow = compile_workflow('ff_nvt')
        with pytest.raises(RuntimeError, match='cond dump file not found'):
            parse_output(config, str(tmp_path), workflow)

    def test_optimize_has_no_fin_velocity(self, tmp_path):
        _write_outputs(tmp_path, log_lines=MULTI_LOG, xyz=FINAL_XYZ,
                       profile='ff_optimize')
        config = _make_config(profile='ff_optimize')
        workflow = compile_workflow('ff_optimize')
        results = parse_output(config, str(tmp_path), workflow)
        assert 'fin_velocity' not in results


class TestGradientCollection:
    """ff_gradient force dump parsing (with and without id column)."""

    FORCE_DUMP_WITH_ID = (
        'ITEM: TIMESTEP\n'
        '0\n'
        'ITEM: NUMBER OF ATOMS\n'
        '3\n'
        'ITEM: BOX BOUNDS pp pp pp\n'
        '0 10\n'
        '0 10\n'
        '0 10\n'
        'ITEM: ATOMS id fx fy fz\n'
        '1 10.2244 -3.61294 -4.59281\n'
        '2 -0.618265 -2.44341 3.71219\n'
        '3 -5.64704 -0.323485 -0.514467\n'
    )

    FORCES = [
        [10.2244, -3.61294, -4.59281],
        [-0.618265, -2.44341, 3.71219],
        [-5.64704, -0.323485, -0.514467],
    ]

    def _expected_forces(self):
        return np.array(self.FORCES)

    def test_force_dump_with_id_column(self, tmp_path):
        _write_outputs(tmp_path, log_lines=TABLE_LOG, energy=ENERGY_VALUE,
                       force=self.FORCE_DUMP_WITH_ID,
                       profile='ff_gradient')
        config = _make_config(profile='ff_gradient')
        workflow = compile_workflow('ff_gradient')
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['gradient'],
                           -np.asarray(self._expected_forces()))

    def test_force_dump_without_id_column(self, tmp_path):
        # Reference format: write_dump all custom ... fx fy fz
        # modify sort id (rows are already id-sorted)
        _write_outputs(tmp_path, log_lines=TABLE_LOG, energy=ENERGY_VALUE,
                       force=FORCE_DUMP_NO_ID, profile='ff_gradient')
        config = _make_config(profile='ff_gradient')
        workflow = compile_workflow('ff_gradient')
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['gradient'],
                           -np.asarray(self._expected_forces()))

    def test_energy_comes_from_its_own_file(self, tmp_path):
        # One number, fifteen digits: the log's thermo table quantizes to
        # eight, so the file is the source and the table beside it is not
        # consulted at all.
        _write_outputs(tmp_path, log_lines=TABLE_LOG,
                       energy=-10.5027317912345)
        config = _make_config(profile='ff_energy')
        workflow = compile_workflow('ff_energy')
        results = parse_output(config, str(tmp_path), workflow)
        expected = -10.5027317912345
        assert results['energy'][0] == pytest.approx(expected)

    def test_framework_scaling_applies_units_to_the_gradient(self, tmp_path):
        # The parser already returns `gradient` = dU/dR in native units; the
        # framework's job is the unit, and it reads it from this skill's
        # manifest declaration.
        _write_outputs(tmp_path, log_lines=TABLE_LOG, energy=ENERGY_VALUE,
                       force=self.FORCE_DUMP_WITH_ID,
                       profile='ff_gradient')
        config = _make_config(profile='ff_gradient')
        workflow = compile_workflow('ff_gradient')
        results = parse_output(config, str(tmp_path), workflow)
        converted = apply_result_units(
            dict(results, _native_units='real'), 'lammps_skill')
        expected = (-np.array(self.FORCES)
                    * FORCE_TO_HARTREE_PER_BOHR['real'])
        assert np.allclose(converted['gradient'][0], expected)
        assert converted['energy'][0] == pytest.approx(
            ENERGY_VALUE * ENERGY_TO_HARTREE['real'])

    def test_missing_energy_file_fails_with_the_log_quoted(self, tmp_path):
        _write_outputs(tmp_path, log_lines=TABLE_LOG)
        config = _make_config(profile='ff_energy')
        workflow = compile_workflow('ff_energy')
        with pytest.raises(RuntimeError, match='energy file not found'):
            parse_output(config, str(tmp_path), workflow)
        try:
            parse_output(config, str(tmp_path), workflow)
        except RuntimeError as exc:
            # The log is not the source, but it is still where the reason
            # for a missing result is written.
            assert 'LAMMPS log tail:' in str(exc)


# Single-point thermo table carrying the six virial pressure
# components (xx yy zz xy xz yz) of 'compute Pconf all pressure NULL
# virial'; 'real' units report pressure in atm.
STRESS_TABLE_LOG = [
    'Step PotEng Volume c_Pconf[1] c_Pconf[2] c_Pconf[3] '
    'c_Pconf[4] c_Pconf[5] c_Pconf[6] ',
    '       0   -10.502731791   250.047   '
    '100.0   200.0   300.0   10.0   20.0   30.0 ',
    'Loop time of 0.01 on 1 procs for 0 steps with 3 atoms',
]
STRESS_PCONF = np.array([[100.0, 10.0, 20.0],
                         [10.0, 200.0, 30.0],
                         [20.0, 30.0, 300.0]])


class TestStressTensor:
    """Whole-system stress tensor from the virial pressure compute."""

    STRESS_LINE = ('STRESS 100.0 200.0 300.0 10.0 20.0 30.0\n')

    def test_stress_from_its_own_file(self, tmp_path):
        _write_outputs(tmp_path, log_lines=TABLE_LOG, energy=ENERGY_VALUE,
                       force=FORCE_DUMP_NO_ID, profile='ff_gradient',
                       stress_file=self.STRESS_LINE)
        config = _make_config(profile='ff_gradient', stress=True)
        workflow = compile_workflow('ff_gradient')
        results = parse_output(config, str(tmp_path), workflow)
        expected = -STRESS_PCONF * PRESSURE_TO_BAR['real']
        assert results['stress_tensor'].shape == (1, 3, 3)
        assert np.allclose(results['stress_tensor'][0], expected)

    def test_stress_zero_filled_when_not_requested(self, tmp_path):
        _write_outputs(tmp_path, log_lines=TABLE_LOG, energy=ENERGY_VALUE,
                       force=FORCE_DUMP_NO_ID, profile='ff_gradient')
        config = _make_config(profile='ff_gradient')
        workflow = compile_workflow('ff_gradient')
        results = parse_output(config, str(tmp_path), workflow)
        assert results['stress_tensor'].shape == (1, 3, 3)
        assert np.allclose(results['stress_tensor'], 0.0)

    def test_missing_stress_file_is_fatal(self, tmp_path):
        # A stress request whose file is absent is a failed step, not a
        # step whose stress happens to be zero.
        _write_outputs(tmp_path, log_lines=TABLE_LOG, energy=ENERGY_VALUE,
                       force=FORCE_DUMP_NO_ID, profile='ff_gradient')
        config = _make_config(profile='ff_gradient', stress=True)
        workflow = compile_workflow('ff_gradient')
        with pytest.raises(RuntimeError, match='stress file'):
            parse_output(config, str(tmp_path), workflow)

    def test_thermo_pressure_columns_are_not_read(self, tmp_path):
        # The log's c_Pconf columns used to be a fallback. With the
        # dedicated file present they must not be consulted at all, and
        # with it absent nothing rescues the run: reading the file is the
        # whole contract.
        _write_outputs(tmp_path, log_lines=STRESS_TABLE_LOG,
                       energy=ENERGY_VALUE, force=FORCE_DUMP_NO_ID,
                       profile='ff_gradient',
                       stress_file='STRESS 1.0 2.0 3.0 0.0 0.0 0.0\n')
        config = _make_config(profile='ff_gradient', stress=True)
        workflow = compile_workflow('ff_gradient')
        results = parse_output(config, str(tmp_path), workflow)
        file_conf = np.array([[1.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0],
                              [0.0, 0.0, 3.0]])
        expected = -file_conf * PRESSURE_TO_BAR['real']
        assert np.allclose(results['stress_tensor'][0], expected)
