"""
======================================================================
Test suite for the MD trajectory output writer
(pyrai3md/workflow/records/output.py).

Tests cover:
- Legacy-format regression of the md.xyz / md.velo / md.energies /
  md.pop records (the pop rows carry the electronic populations
  followed by the 0/1 active state columns)
- Periodic frames stating their cell: the md.xyz comment carries the
  extended-XYZ Lattice key, md.velo does not, and a non-periodic or
  cell-less frame keeps the legacy comment byte for byte
- Hopping-event records (sh.xyz / sh.velo / sh.energies / sh.pop):
  written on hops only, post-hop velocities, never suppressed by the
  direct/buffer control
- Direct/buffer output control: direct steps written every step,
  buffered period afterwards, forced final-step output, and the gate
  shared with the run-log step block
- Header writing and the population-less (nosh) behavior

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import re
import tempfile
import unittest

import numpy as np

from pyrai3md.utilities.lattice import cell6_to_matrix
from pyrai3md.utilities.units import (AMU_TO_AU, au_pressure_to_bar,
                                      KELVIN_TO_HARTREE)
from pyrai3md.workflow.records.output import MdOutputWriter

TITLE = 'pyrai3md'
DT = 20.67
ATOMS = ['C', 'H']
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
VELOS = [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]
MASS = [12.0, 1.0]
ENERGIES = [0.0, 0.05]
POPULATION = [0.8, 0.2]

# Active-state whole-system stress (3, 3) in Hartree/Bohr^3, tensile
# positive. Trace = -0.006 -> P = -Tr/3 = 0.002 au.
STRESS = [[-0.001, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, -0.003]]
# A second state block used to verify active-state projection.
STRESS_S1 = [[-0.010, 0.0, 0.0], [0.0, -0.010, 0.0], [0.0, 0.0, -0.010]]

ENERGY_HEAD = '%20s%28s%28s%28s%28s\n' % (
    'time', 'Epot', 'Ekin', 'Etot', 'Epot1,2,3...')
POP_HEAD = '%20s%28s%28s\n' % ('time', 'Pop1,2,3...', 'Act1,2,3...')
STRESS_HEAD = ('%20s' % 'time'
               + ''.join('%20s' % c for c in MdOutputWriter._STRESS_COMPONENTS)
               + '%20s\n' % 'P(bar)')


class _MockDynamics:
    def __init__(self, population):
        self.state_population = list(population)


class _MockPmd:
    def __init__(self, population):
        self.dynamics = _MockDynamics(population)


class MockState:
    """Minimal live-state stand-in for the writer interface."""

    def __init__(self, current_state, velocities, population, runtime=None):
        self.current_state = current_state
        self.velocities = velocities
        self.pmd = _MockPmd(population)
        if runtime is not None:
            self.runtime = runtime


class _MockRuntime:
    """Live-runtime stand-in carrying the per-step ensemble record."""

    def __init__(self, ensemble):
        self.ensemble = dict(ensemble)


def _snapshot(current_state=0, stress=None, dynamics=None, runtime=None,
              **system_extra):
    """Build one history snapshot (pmd model dump) of a step."""
    results = {'energy': list(ENERGIES)}
    if stress is not None:
        results['stress_tensor'] = stress
    system = {
        'atoms': list(ATOMS),
        'coordinates': [list(row) for row in COORDS],
        'velocities': [list(row) for row in VELOS],
        'mass': list(MASS),
        'current_state': current_state,
    }
    system.update(system_extra)
    snapshot = {
        'system': system,
        'results': results,
    }
    if dynamics is not None:
        snapshot['dynamics'] = dict(dynamics)
    if runtime is not None:
        snapshot['runtime'] = dict(runtime)
    return snapshot


def _coord_lines(values):
    text = ''
    for element, (x, y, z) in zip(ATOMS, values):
        text += '%-5s%24.16f%24.16f%24.16f\n' % (
            element, float(x), float(y), float(z))
    return text


def _expected_ekin():
    mass_au = [m * AMU_TO_AU for m in MASS]
    velo_sq = [sum(v * v for v in row) for row in VELOS]
    return 0.5 * sum(m * s for m, s in zip(mass_au, velo_sq))


class MdOutputTestCase(unittest.TestCase):
    """Base case running the writer inside a temporary directory."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_dir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _writer(self, **kwargs):
        args = dict(title=TITLE, log_dir=self.log_dir, direct=2000,
                    buffer=500, dt=DT, total_steps=6)
        args.update(kwargs)
        return MdOutputWriter(**args)

    def _read(self, suffix):
        path = os.path.join(self.log_dir, '%s.%s' % (TITLE, suffix))
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return fh.read()


class TestHeaders(MdOutputTestCase):
    """Energies and pop files get one header; xyz/velo do not."""

    def test_headers_written_once(self):
        writer = self._writer()
        writer.write_headers()
        for prefix in ('md', 'sh'):
            self.assertEqual(self._read('%s.energies' % prefix),
                             ENERGY_HEAD)
            self.assertEqual(self._read('%s.pop' % prefix), POP_HEAD)
        # xyz/velo are self-describing and start empty.
        self.assertIsNone(self._read('md.xyz'))
        self.assertIsNone(self._read('sh.velo'))

    def test_headers_append_on_restart(self):
        writer = self._writer()
        writer.write_headers()
        writer.write_headers()
        self.assertEqual(self._read('md.energies'), ENERGY_HEAD * 2)

    def test_reset_removes_previous_run_records(self):
        # A fresh start wipes every md/sh record so repeated runs
        # never stack frames or headers on top of each other.
        writer = self._writer()
        writer.write_headers()
        writer.reset()
        for prefix in ('md', 'sh'):
            for suffix in ('xyz', 'velo', 'energies', 'pop'):
                self.assertIsNone(
                    self._read('%s.%s' % (prefix, suffix)))
        # The next start writes exactly one header again.
        writer.write_headers()
        self.assertEqual(self._read('md.energies'), ENERGY_HEAD)


class TestRecordFormat(MdOutputTestCase):
    """Frame and row formats match the legacy writer bit for bit."""

    def test_md_frame_formats(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(3, _snapshot(), state)

        cmmt = '%s coord %d state %d' % (TITLE, 3, 1)
        self.assertEqual(self._read('md.xyz'),
                         '2\n%s\n%s' % (cmmt, _coord_lines(COORDS)))
        self.assertEqual(self._read('md.velo'),
                         '2\n%s\n%s' % (cmmt, _coord_lines(VELOS)))

        ekin = _expected_ekin()
        epot = ENERGIES[0]
        time_text = '%20.2f' % (2 * DT)
        energy_row = time_text + '%28.16f%28.16f%28.16f' % (
            epot, ekin, epot + ekin)
        energy_row += ''.join('%28.16f' % e for e in ENERGIES)
        self.assertEqual(self._read('md.energies'), energy_row + '\n')

        pop_row = time_text + ''.join('%28.16f' % p
                                      for p in POPULATION)
        # The 0/1 active state columns mark the post-hop state
        # (state 0 here: no hop in this step).
        pop_row += ''.join('%28.16f' % a for a in (1.0, 0.0))
        self.assertEqual(self._read('md.pop'), pop_row + '\n')

    def test_no_hop_writes_no_sh_records(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(), state)
        for suffix in ('sh.xyz', 'sh.velo', 'sh.energies', 'sh.pop'):
            self.assertIsNone(self._read(suffix))

    def test_no_population_writes_no_pop_rows(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=[])
        writer.record(1, _snapshot(), state)
        # The md frame and energies still land; pop stays absent.
        self.assertIsNotNone(self._read('md.xyz'))
        self.assertIsNone(self._read('md.pop'))


class TestHoppingRecords(MdOutputTestCase):
    """sh.* records are written on hops only, with post-hop data."""

    def test_hop_records_all_four_files(self):
        writer = self._writer(direct=0, buffer=500)
        hop_velo = [[0.002, 0.0, 0.0], [0.0, 0.002, 0.0]]
        state = MockState(current_state=1, velocities=hop_velo,
                          population=POPULATION)
        write_md = writer.record(2, _snapshot(current_state=0), state)

        # Suppressed by the direct/buffer gate, but sh.* still land.
        self.assertFalse(write_md)
        self.assertIsNone(self._read('md.xyz'))

        cmmt = '%s coord %d state %d to %d CI' % (TITLE, 2, 1, 2)
        # Coordinates and energies stay pre-hop (computation-time
        # values); velocities carry the post-hop adjustment.
        self.assertEqual(self._read('sh.xyz'),
                         '2\n%s\n%s' % (cmmt, _coord_lines(COORDS)))
        self.assertEqual(self._read('sh.velo'),
                         '2\n%s\n%s' % (cmmt, _coord_lines(hop_velo)))

        ekin = _expected_ekin()
        epot = ENERGIES[0]
        time_text = '%20.2f' % (1 * DT)
        energy_row = time_text + '%28.16f%28.16f%28.16f' % (
            epot, ekin, epot + ekin)
        energy_row += ''.join('%28.16f' % e for e in ENERGIES)
        self.assertEqual(self._read('sh.energies'), energy_row + '\n')

        pop_row = time_text + ''.join('%28.16f' % p
                                      for p in POPULATION)
        # The active state columns mark the post-hop state (1).
        pop_row += ''.join('%28.16f' % a for a in (0.0, 1.0))
        self.assertEqual(self._read('sh.pop'), pop_row + '\n')

    def test_hop_comment_also_marks_md_frames(self):
        writer = self._writer()
        state = MockState(current_state=1, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(current_state=0), state)
        cmmt = '%s coord %d state %d to %d CI' % (TITLE, 1, 1, 2)
        self.assertEqual(self._read('md.xyz'),
                         '2\n%s\n%s' % (cmmt, _coord_lines(COORDS)))


class TestDirectBufferControl(MdOutputTestCase):
    """Legacy semantics: direct steps, buffered period, forced end."""

    def test_should_output_sequence(self):
        writer = self._writer(direct=2, buffer=2, total_steps=6)
        results = [writer.should_output(step, final=(step == 6))
                   for step in range(1, 7)]
        self.assertEqual(results, [True, True, False, True, False,
                                   True])

    def test_record_gating_matches_should_output(self):
        writer = self._writer(direct=2, buffer=2, total_steps=6)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        written = []
        for step in range(1, 7):
            if writer.record(step, _snapshot(), state,
                             final=(step == 6)):
                written.append(step)
        self.assertEqual(written, [1, 2, 4, 6])
        # One frame per written step.
        self.assertEqual(self._read('md.xyz').count('coord'), 4)

    def test_direct_steps_always_written(self):
        writer = self._writer(direct=3, buffer=100, total_steps=3)
        for step in range(1, 4):
            self.assertTrue(writer.should_output(step,
                                                 final=(step == 3)))

    def test_final_step_forced_after_buffer(self):
        writer = self._writer(direct=0, buffer=100, total_steps=2)
        self.assertFalse(writer.should_output(1, final=False))
        self.assertTrue(writer.should_output(2, final=True))

    def test_ten_picosecond_throttling(self):
        # The long-run example setting: the first step, every 1000th
        # step after it, and the last one -> 11 frames for 10000 steps.
        writer = self._writer(direct=1, buffer=1000, total_steps=10000)
        written = [step for step in range(1, 10001)
                   if writer.should_output(step, final=(step == 10000))]
        self.assertEqual(written,
                         [1] + list(range(1001, 9002, 1000)) + [10000])

    def test_buffer_longer_than_the_run_keeps_both_ends(self):
        writer = self._writer(direct=1, buffer=1000, total_steps=10)
        written = [step for step in range(1, 11)
                   if writer.should_output(step, final=(step == 10))]
        self.assertEqual(written, [1, 10])

    def test_phase_survives_an_unrecorded_step(self):
        # The gate is a pure function of the step number, so a step
        # that never reaches the writer cannot shift the following
        # buffered frames.
        writer = self._writer(direct=1, buffer=3, total_steps=12)
        full = [step for step in range(1, 13)
                if writer.should_output(step, final=(step == 12))]
        offered = [step for step in (1, 2, 5, 6, 7, 8, 12)
                   if writer.should_output(step, final=(step == 12))]
        self.assertEqual(full, [1, 4, 7, 10, 12])
        self.assertEqual(offered, [1, 7, 12])

    def test_first_frame_clock_reads_zero(self):
        # Step 1 records the initial condition: both velocity-Verlet
        # half kicks are skipped on the first iteration, so the frame
        # sits at t = 0 rather than one timestep ahead of itself.
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(), state)
        row = self._read('md.energies').splitlines()[0]
        self.assertEqual(row[:20], '%20.2f' % 0.0)
        self.assertEqual(self._read('md.pop').splitlines()[0][:20],
                         '%20.2f' % 0.0)


class TestStressRecords(MdOutputTestCase):
    """md.stress / sh.stress rows and their availability gating."""

    @staticmethod
    def _expected_row(step, tensor):
        import numpy as np
        arr = np.asarray(tensor, dtype=float).reshape(3, 3)
        pressure_au = -float(np.trace(arr)) / 3.0
        text = '%20.2f' % ((step - 1) * DT)
        text += ''.join('%20.12f' % float(v) for v in arr.reshape(-1))
        text += '%20.4f' % au_pressure_to_bar(pressure_au)
        return text + '\n'

    def test_md_stress_written(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(3, _snapshot(stress=[STRESS]), state)
        self.assertEqual(self._read('md.stress'),
                         STRESS_HEAD + self._expected_row(3, STRESS))

    def test_active_state_projection(self):
        # pre_state = 1 selects the second (nstate, 3, 3) block.
        writer = self._writer()
        state = MockState(current_state=1, velocities=VELOS,
                          population=POPULATION)
        writer.record(2, _snapshot(current_state=1,
                                   stress=[STRESS, STRESS_S1]), state)
        self.assertEqual(self._read('md.stress'),
                         STRESS_HEAD + self._expected_row(2, STRESS_S1))

    def test_no_stress_writes_no_file(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(), state)
        self.assertIsNone(self._read('md.stress'))
        self.assertIsNone(self._read('sh.stress'))

    def test_zero_stress_writes_no_file(self):
        writer = self._writer()
        zero = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(stress=zero), state)
        self.assertIsNone(self._read('md.stress'))

    def test_hop_writes_sh_stress_when_md_gated(self):
        writer = self._writer(direct=0, buffer=500)
        state = MockState(current_state=1, velocities=VELOS,
                          population=POPULATION)
        write_md = writer.record(2, _snapshot(current_state=0,
                                              stress=[STRESS, STRESS_S1]),
                                 state)
        # md.* suppressed by the gate, but the hop record still lands.
        self.assertFalse(write_md)
        self.assertIsNone(self._read('md.stress'))
        # sh.stress projects the pre-hop state (index 0).
        self.assertEqual(self._read('sh.stress'),
                         STRESS_HEAD + self._expected_row(2, STRESS))

    def test_md_stress_follows_direct_buffer_gate(self):
        writer = self._writer(direct=2, buffer=2, total_steps=6)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        for step in range(1, 7):
            writer.record(step, _snapshot(stress=[STRESS]), state,
                          final=(step == 6))
        # Steps 1, 2, 4, 6 written: one header plus four rows.
        content = self._read('md.stress')
        self.assertTrue(content.startswith(STRESS_HEAD))
        self.assertEqual(content.count('\n'), 5)


class TestEnsembleRecords(MdOutputTestCase):
    """md.ensemble / sh.ensemble rows, column planning and gating."""

    NHC_DYNAMICS = {
        'ensemble': 'nvt', 'thermostat': 'nose-hoover-chain',
        'thermostat_chain': 2, 'temperature': 300.0,
    }
    NHC_DIAG = {
        'ndof': 3, 'ke': 0.001, 'kt': 300.0 * KELVIN_TO_HARTREE,
        'q': [11.0, 12.0], 'scale_pre': 0.9, 'scale_post': 1.1,
        'thermostat_skipped': None,
    }
    NHC_DOF = [0.1, 0.2, 0.3, 0.4]   # [xi1, xi2, peta1, peta2]
    NHC_HEAD_NAMES = ('time', 'T_inst(K)', 'T_target(K)', 'N_df',
                      'Ekin(Ha)', 'V(A^3)', 'scale_pre', 'scale_post',
                      'Q1', 'Q2', 'xi1', 'xi2', 'peta1', 'peta2')

    def _nhc_snapshot(self):
        return _snapshot(dynamics=self.NHC_DYNAMICS,
                         runtime={'ensemble': dict(self.NHC_DIAG)},
                         thermostat_dof=list(self.NHC_DOF),
                         volume=100.0)

    def _expected_nhc_row(self, step):
        ekin = _expected_ekin()
        t_inst = 2.0 * ekin / (3 * KELVIN_TO_HARTREE)
        values = [t_inst, 300.0, 3.0, ekin, 100.0, 0.9, 1.1,
                  11.0, 12.0, 0.1, 0.2, 0.3, 0.4]
        text = '%20.2f' % ((step - 1) * DT)
        text += ''.join('%20.12f' % v for v in values)
        return text + '\n'

    def _head(self, names):
        return '%20s' % 'time' + ''.join('%20s' % n for n in names[1:]) + '\n'

    def test_no_dynamics_writes_no_file(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(), state)
        self.assertIsNone(self._read('md.ensemble'))
        self.assertIsNone(self._read('sh.ensemble'))

    def test_nve_writes_no_file(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        snapshot = _snapshot(dynamics={'ensemble': 'nve'},
                             runtime={'ensemble': {}})
        writer.record(1, snapshot, state)
        self.assertIsNone(self._read('md.ensemble'))

    def test_nvt_nhc_row_and_header(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(3, self._nhc_snapshot(), state)
        expected = (self._head(self.NHC_HEAD_NAMES)
                    + self._expected_nhc_row(3))
        self.assertEqual(self._read('md.ensemble'), expected)

    def test_live_runtime_diagnostics_preferred(self):
        # The writer reads state.runtime.ensemble first; the snapshot
        # runtime projection is the fallback (both carry the same step).
        writer = self._writer()
        diag = dict(self.NHC_DIAG, scale_post=0.5)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION,
                          runtime=_MockRuntime(diag))
        writer.record(1, self._nhc_snapshot(), state)
        content = self._read('md.ensemble')
        row = content.splitlines()[1]
        self.assertIn('%20.12f' % 0.5, row)

    def test_an_oversized_value_does_not_shift_the_other_columns(self):
        # The record is a whitespace-separated table, so a number wider
        # than its field does not merely look bad: it fuses with its
        # neighbour and every column after it shifts one. The diverging
        # NPT examples of tmp_20260928_223009 did exactly that (a 23-
        # character s column in a 20-wide field), and the barostat drive
        # read out of those files was nonsense -- kinetic pressure with a
        # negative sign, Pint pinned at P_ext. A wide value must fall
        # back to a form that fits, keeping the row parseable by column.
        writer = self._writer()
        huge = dict(self.NHC_DIAG, w=1.2676495335789e7, scale_post=0.5)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION,
                          runtime=_MockRuntime(huge))
        writer.record(1, self._nhc_snapshot(), state)
        content = self._read('md.ensemble')
        names = content.splitlines()[0].split()
        row = content.splitlines()[1].split()
        self.assertEqual(len(row), len(names),
                         'row has %d fields, header %d' % (len(row),
                                                           len(names)))
        for name, value in zip(names, row):
            if name == 'W':
                # Still the same number, and still in its own column.
                self.assertAlmostEqual(float(value), 1.2676495335789e7,
                                       delta=1e-1)
            self.assertNotEqual(value, '', name)

    def test_a_value_that_fills_the_field_still_gets_a_separator(self):
        # The gutter is the ONLY separator between columns, and it lives in
        # the padding -- so the fit test has to measure the digits, not the
        # padded field. tmp_20260929_07:1340 proved the difference: the
        # first pass at the fix above checked the padded width and so still
        # emitted `-10.996128489922` next to `9807535.803130459040`, two
        # fields of exactly 20 characters that read back as one 40-character
        # token. Ordinary values must stay byte-identical to '%20.12f'.
        ens_cell = MdOutputWriter._ens_cell
        self.assertEqual(ens_cell(2494.364976804596),
                         '%20.12f' % 2494.364976804596)
        filling = 9807535.803130459040          # 20 digits, no room left
        self.assertEqual(len('%.12f' % filling), 20)
        self.assertTrue(ens_cell(filling).startswith(' '),
                        ens_cell(filling))
        row = ens_cell(-10.996128489922) + ens_cell(filling)
        self.assertEqual(len(row.split()), 2, row)

    def test_missing_diagnostics_render_nan_columns(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        snapshot = self._nhc_snapshot()
        snapshot['runtime'] = {'ensemble': {}}
        writer.record(1, snapshot, state)
        row = self._read('md.ensemble').splitlines()[1]
        self.assertIn('nan', row)

    def test_npt_mttk_header_layout(self):
        dynamics = {
            'ensemble': 'npt', 'thermostat': 'nose-hoover',
            'thermostat_chain': 1, 'barostat': 'mttk',
            'temperature': 300.0, 'pressure': 1.0,
        }
        diag = {
            'ndof': 3, 'q': [5.0], 'scale_pre': 1.0, 'scale_post': 1.0,
            'w': 7.0, 'alpha_cell': 2.0, 'p_ext': 0.0,
            'p_kin': [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]],
            'p_vir': [[-1e-4, 0, 0], [0, -1e-4, 0], [0, 0, -1e-4]],
            'g_eps': [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            'ke_bar': 0.25, 'scale_bar': 0.98, 'q_bar': [9.0],
            'scale_particle_post': 1.0001,
        }
        strain = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]
        mom = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        vel = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
        snapshot = _snapshot(
            dynamics=dynamics, runtime={'ensemble': diag},
            thermostat_dof=[0.0, 0.0], barostat_strain=strain,
            barostat_momentum=mom, barostat_velocity=vel,
            barostat_thermostat_dof=[0.2, 0.3], volume=1000.0)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer = self._writer()
        writer.record(1, snapshot, state)
        content = self._read('md.ensemble')
        header = content.splitlines()[0]
        names = header.split()
        comps = [c for c in MdOutputWriter._MATRIX_COMPONENTS]
        expected_names = (
            ['time', 'T_inst(K)', 'T_target(K)', 'N_df', 'Ekin(Ha)',
             'V(A^3)', 'scale_pre', 'scale_post', 'Q1', 'xi1', 'peta1',
             'W', 'Pkin_bar', 'Pvir_bar', 'Pint_bar', 'Pext_bar']
            + ['geps_%s' % c for c in comps]
            + ['eps_%s' % c for c in comps]
            + ['peps_%s' % c for c in comps]
            + ['veps_%s' % c for c in comps]
            + ['scale_cell_pre', 'scale_cell_post',
               'ke_bar', 'scale_bar', 'Qbar1', 'xibar1', 'petabar1'])
        self.assertEqual(names, expected_names)
        values = [float(v) for v in content.splitlines()[1].split()]
        idx = {name: i for i, name in enumerate(names)}
        # eps columns read the persisted system strain (xx component).
        self.assertAlmostEqual(values[idx['eps_xx']], 0.01, places=12)
        self.assertAlmostEqual(values[idx['veps_yy']], 0.1, places=12)
        # Pkin_bar = +Tr(p_kin)/3 converted from au; Pvir_bar = -Tr(stress)/3.
        self.assertAlmostEqual(
            values[idx['Pkin_bar']], au_pressure_to_bar(1e-4), places=6)
        self.assertAlmostEqual(
            values[idx['Pvir_bar']], au_pressure_to_bar(1e-4), places=6)
        self.assertAlmostEqual(values[idx['W']], 7.0, places=12)
        self.assertAlmostEqual(values[idx['petabar1']], 0.3, places=12)

    def test_csvr_header_layout(self):
        dynamics = {'ensemble': 'nvt', 'thermostat': 'csvr',
                    'temperature': 300.0}
        diag = {'ndof': 3, 'ke': 0.001, 'csvr_alpha': 1.2,
                'scale_post': 1.2 ** 0.5, 'csvr_ratio': 0.9,
                'csvr_c': 0.5, 'csvr_r1': -0.3, 'csvr_s': 2.0}
        snapshot = _snapshot(dynamics=dynamics,
                             runtime={'ensemble': diag}, volume=100.0)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer = self._writer()
        writer.record(1, snapshot, state)
        content = self._read('md.ensemble')
        names = content.splitlines()[0].split()
        self.assertEqual(names, [
            'time', 'T_inst(K)', 'T_target(K)', 'N_df', 'Ekin(Ha)',
            'V(A^3)', 'alpha', 'scale_post', 'ratio', 'c', 'r1', 's'])

    def test_hop_writes_sh_ensemble_when_md_gated(self):
        writer = self._writer(direct=0, buffer=500)
        state = MockState(current_state=1, velocities=VELOS,
                          population=POPULATION)
        write_md = writer.record(2, self._nhc_snapshot(), state)
        self.assertFalse(write_md)
        self.assertIsNone(self._read('md.ensemble'))
        self.assertEqual(self._read('sh.ensemble'),
                         self._head(self.NHC_HEAD_NAMES)
                         + self._expected_nhc_row(2))

    def test_md_ensemble_follows_direct_buffer_gate(self):
        writer = self._writer(direct=2, buffer=2, total_steps=6)
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        for step in range(1, 7):
            writer.record(step, self._nhc_snapshot(), state,
                          final=(step == 6))
        content = self._read('md.ensemble')
        self.assertTrue(content.startswith(self._head(self.NHC_HEAD_NAMES)))
        self.assertEqual(content.count('\n'), 5)

    def test_reset_removes_ensemble_files(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, self._nhc_snapshot(), state)
        self.assertIsNotNone(self._read('md.ensemble'))
        writer.reset()
        self.assertIsNone(self._read('md.ensemble'))


class TestVeloZeroFill(MdOutputTestCase):
    """md.velo/sh.velo are recorded even when the snapshot carries no
    velocities (zero-filled), so every recorded step owns one velo
    frame in all three input modes (an agent .config without the
    velocities key used to skip the file entirely)."""

    def _snapshot_without_velo(self, current_state=0):
        snapshot = _snapshot(current_state=current_state)
        snapshot['system'].pop('velocities')
        return snapshot

    def test_md_velo_zero_filled(self):
        writer = self._writer()
        state = MockState(current_state=0, velocities=[],
                          population=POPULATION)
        writer.record(1, self._snapshot_without_velo(), state)
        content = self._read('md.velo')
        self.assertIsNotNone(content)
        lines = content.splitlines()
        self.assertEqual(lines[0].strip(), '2')
        for line in lines[2:4]:
            self.assertEqual(line.split()[1:],
                             ['0.0000000000000000'] * 3)

    def test_sh_velo_zero_filled_on_hop(self):
        writer = self._writer()
        state = MockState(current_state=1, velocities=[],
                          population=POPULATION)
        writer.record(2, self._snapshot_without_velo(current_state=0),
                      state)
        self.assertIsNotNone(self._read('sh.velo'))


class TestPeriodicFrameCarriesItsCell(MdOutputTestCase):
    """A main-image trajectory is only readable when it states its box.

    The geometry frames keep the wrapped (main image) coordinates, which
    is what lets a reader find bonds and contacts with the minimum-image
    convention. Without the cell in the file those same numbers look like
    a molecule chopped by a cell face, so the box travels in the frame's
    comment line under the extended XYZ convention.
    """

    # The O-H pair of the shipped water box that straddles the x face:
    # the raw difference is 9.24 Angstrom, its minimum image 0.78, and the
    # bond really is 0.9653.
    STRADDLING = [[0.7758370, 2.9046780, 7.8523340],
                  [9.9999999, 3.0545250, 7.2978220]]
    CUBIC = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

    def _record(self, **system_extra):
        writer = self._writer()
        state = MockState(current_state=0, velocities=VELOS,
                          population=POPULATION)
        writer.record(1, _snapshot(**system_extra), state)
        return writer

    @staticmethod
    def _comment(text):
        return text.split('\n')[1]

    @staticmethod
    def _lattice_of(text):
        """The nine Lattice numbers of one emitted frame, or None."""
        match = re.search(r'Lattice="([^"]+)"', text)
        if not match:
            return None
        return [float(value) for value in match.group(1).split()]

    def test_md_xyz_comment_carries_lattice(self):
        self._record(pbc=[1, 1, 1], lattice_vector=self.CUBIC,
                     coordinates=self.STRADDLING)
        xyz = self._read('md.xyz')
        self.assertEqual(self._lattice_of(xyz),
                         [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0])
        self.assertEqual(self._comment(xyz),
                         '%s coord 1 state 1 Lattice="%s"' % (
                             TITLE,
                             ' '.join('%.6f' % value for value in
                                      [10.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                                       0.0, 0.0, 10.0])))

    def test_velocity_frames_do_not_claim_a_lattice(self):
        # The box is the geometry frame's business; a velocity table with
        # a Lattice key would read as a second, possibly stale cell.
        self._record(pbc=[1, 1, 1], lattice_vector=self.CUBIC,
                     coordinates=self.STRADDLING)
        self.assertIsNone(self._lattice_of(self._read('md.velo')))

    def test_emitted_lattice_recovers_the_straddling_bond(self):
        # Before the cell travelled in the file this assertion had nothing
        # to work with, and a naive read of the frame reported the 9.24
        # Angstrom raw separation as the bond length.
        self._record(pbc=[1, 1, 1], lattice_vector=self.CUBIC,
                     coordinates=self.STRADDLING)
        rows = [line.split()[1:4] for line in
                self._read('md.xyz').split('\n')[2:2 + len(ATOMS)]]
        points = np.asarray([[float(value) for value in row] for row in rows])
        lattice = np.asarray(self._lattice_of(self._read('md.xyz')),
                             dtype=np.float64).reshape(3, 3)
        difference = points[0] - points[1]
        fraction = difference @ np.linalg.inv(lattice)
        fraction -= np.round(fraction)
        bond = float(np.linalg.norm(fraction @ lattice))
        self.assertGreater(bond, 0.90)
        self.assertLess(bond, 1.03)

    def test_non_periodic_frame_keeps_the_legacy_comment(self):
        self._record(pbc=[0, 0, 0], lattice_vector=self.CUBIC,
                     coordinates=self.STRADDLING)
        self.assertIsNone(self._lattice_of(self._read('md.xyz')))
        self.assertEqual(self._comment(self._read('md.xyz')),
                         '%s coord 1 state 1' % TITLE)

    def test_periodic_frame_without_a_cell_stays_silent(self):
        # Declaring periodicity without a usable cell is an input error the
        # integrator already refuses; the writer must not invent a box.
        self._record(pbc=[1, 1, 1], coordinates=self.STRADDLING)
        self.assertIsNone(self._lattice_of(self._read('md.xyz')))

    def test_six_parameter_cell_is_promoted_to_vectors(self):
        self._record(pbc=[1, 1, 1], cell=[8.399091, 8.399091, 8.399091,
                                          90.0, 90.0, 90.0],
                     lattice_vector=[], coordinates=self.STRADDLING)
        lattice = self._lattice_of(self._read('md.xyz'))
        self.assertAlmostEqual(lattice[0], 8.399091, places=6)
        self.assertAlmostEqual(lattice[4], 8.399091, places=6)
        self.assertEqual(lattice[1], 0.0)

    def test_triclinic_cell_is_emitted_as_its_own_rows(self):
        cell6 = [8.0, 9.0, 10.0, 80.0, 95.0, 100.0]
        expected = cell6_to_matrix(np.asarray(cell6)).reshape(-1)
        self._record(pbc=[1, 1, 1], lattice_vector=expected.reshape(3, 3)
                     .tolist(), coordinates=self.STRADDLING)
        np.testing.assert_allclose(self._lattice_of(self._read('md.xyz')),
                                   expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
