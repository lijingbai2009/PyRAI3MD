"""
======================================================================
Test suite for the qualification NPT cell-stability verdict
(pyrai3md/qualification/runner.py).

Tests cover:
- the product readers: the Lattice= comment of md.xyz, the named columns
  of md.ensemble, malformed and missing files
- the frame-rotation measure, cross-checked against an SVD polar factor
- the example gate: only YAML md runs whose ensemble is NPT are judged
- the three verdicts -- violation downgrades a pass, a wide-but-legal
  run leaves a trace, a run with too few cell frames is reported as
  unjudged instead of passing silently
- regression: the four shapes the 2026-09-28 run actually produced,
  including the example that ended with a 1.66 Angstrom axis and passed

Every trajectory is inlined; no calculation is executed.

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.qualification.runner import (
    BOX_NOTES, BOX_TOLERANCES, _check_npt_stability, _determinant,
    _frame_rotation, _invert3, _multiply, _orthogonal_factor,
    cell_frame_metrics, read_md_ensemble, read_md_lattice)

ENSEMBLE_HEADER = ('%20s%20s\n' % ('time', 'T_inst(K)'))


def _xyz(lattices, natom=1):
    """Render an md.xyz text from flat nine-element cell rows."""
    blocks = []
    for row in lattices:
        comment = ('run coord 1 state 1 Lattice="%s"'
                   % ' '.join('%.6f' % value for value in row))
        atoms = '\n'.join('Ar %.6f %.6f %.6f' % (0.1 * index, 0.2, 0.3)
                          for index in range(natom))
        blocks.append('%d\n%s\n%s\n' % (natom, comment, atoms))
    return ''.join(blocks)


def _ensemble(rows):
    """Render an md.ensemble text from (time, T_inst) pairs."""
    return ENSEMBLE_HEADER + ''.join('%20.2f%20.6f\n' % row for row in rows)


def _cube(side):
    """Flat nine-element row of a cubic cell."""
    return [side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side]


def _tilted(side, tilt):
    """Flat row of a lower-triangular cell with one tilted axis."""
    return [side, 0.0, 0.0, tilt, side, 0.0, 0.0, 0.0, side]


class _Scratch(unittest.TestCase):
    """Base class providing an emptied staging directory and a spec."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.yaml = os.path.join(self.dir, 'run.yaml')
        self._write_yaml({'ensemble': 'npt', 'temperature': 300.0})

    def _write_yaml(self, md):
        with open(self.yaml, 'w') as handle:
            handle.write('control:\n  jobtype: md\nmd:\n')
            for key, value in md.items():
                handle.write('  %s: %s\n' % (key, value))

    def staged(self, xyz_text, ensemble_text=None, title='run'):
        """Write the two products an NPT run leaves behind."""
        with open(os.path.join(self.dir, '%s.md.xyz' % title), 'w') as f:
            f.write(xyz_text)
        if ensemble_text is not None:
            with open(os.path.join(self.dir, '%s.md.ensemble' % title),
                      'w') as f:
                f.write(ensemble_text)
        return {'status': 'pass', 'details': ''}

    def judge(self, result, config=None):
        _check_npt_stability(result, self.yaml, config, self.dir, 'run')
        return result


class TestReadMdLattice(_Scratch):
    """The Lattice= comment is the cell history of the run."""

    def test_every_frame_row_is_read(self):
        path = os.path.join(self.dir, 'x.md.xyz')
        with open(path, 'w') as handle:
            handle.write(_xyz([_cube(10.0), _cube(10.1)]))
        self.assertEqual(len(read_md_lattice(path)), 2)
        self.assertEqual(read_md_lattice(path)[1][:3],
                         [10.1, 0.0, 0.0])

    def test_a_missing_or_cell_less_file_reads_empty(self):
        self.assertEqual(read_md_lattice(os.path.join(self.dir, 'nope')), [])
        path = os.path.join(self.dir, 'x.md.xyz')
        with open(path, 'w') as handle:
            handle.write(_xyz([_cube(10.0)]).replace('Lattice=', 'nope='))
        self.assertEqual(read_md_lattice(path), [])

    def test_ensemble_columns_follow_the_header(self):
        path = os.path.join(self.dir, 'x.md.ensemble')
        with open(path, 'w') as handle:
            handle.write(_ensemble([(0.0, 300.0), (41.34, 311.5)]))
        columns = read_md_ensemble(path, ('T_inst(K)',))
        self.assertEqual(columns['T_inst(K)'], [300.0, 311.5])
        self.assertNotIn('Pint_bar', read_md_ensemble(
            path, ('T_inst(K)', 'Pint_bar'))['T_inst(K)'])
        self.assertEqual(read_md_ensemble(path, ('absent',)), {})


class TestFrameRotationMeasure(unittest.TestCase):
    """The polar factor of the step's affine map, without numpy in the
    runner and cross-checked against numpy here."""

    def _svd_angle(self, before, after):
        b = np.asarray(before, dtype=np.float64).reshape(3, 3)
        a = np.asarray(after, dtype=np.float64).reshape(3, 3)
        u, _, vt = np.linalg.svd(a @ np.linalg.inv(b))
        trace = np.clip((np.trace(u @ vt) - 1.0) / 2.0, -1.0, 1.0)
        return math.degrees(math.acos(float(trace)))

    def test_a_pure_shape_change_rotates_nothing(self):
        self.assertAlmostEqual(_frame_rotation(_cube(10.0), _cube(9.0)), 0.0)

    def test_a_sheared_cell_agrees_with_the_svd(self):
        before = _cube(10.0)
        after = _tilted(10.0, 1.0)
        self.assertAlmostEqual(math.degrees(_frame_rotation(before, after)),
                               self._svd_angle(before, after), places=6)

    def test_accumulated_turning_of_a_real_trajectory(self):
        # tools/analyze_npt_frames.py measured 11.11 degrees for the
        # csvr_pr example; the same cell series must read the same here.
        lattices = [_tilted(8.399, tilt) for tilt in
                    [0.0, 0.4, 0.9, 1.5, 2.2, 3.0]]
        total = sum(_frame_rotation(a, b) for a, b in zip(lattices,
                                                          lattices[1:]))
        self.assertAlmostEqual(math.degrees(total),
                               sum(self._svd_angle(a, b) for a, b in
                                   zip(lattices, lattices[1:])), places=6)

    def test_the_polar_factor_is_orthogonal(self):
        matrix = [[1.2, 0.4, 0.0], [0.3, 0.9, 0.1], [0.0, 0.2, 1.1]]
        polar = np.asarray(_orthogonal_factor(matrix), dtype=np.float64)
        np.testing.assert_allclose(polar @ polar.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(polar)), 1.0, places=6)


class TestThreeByThreeAlgebra(unittest.TestCase):
    """The runner writes its own 3x3 algebra, so it is checked against
    numpy here rather than trusted."""

    def test_the_inverse_multiplies_back_to_the_identity(self):
        cells = [_cube(8.399), _tilted(8.399, 1.4), _tilted(9.1, -2.2),
                 [9.0, 0.3, 0.1, 0.4, 8.0, 0.0, 0.2, 0.5, 8.5]]
        for row in cells:
            matrix = [row[0:3], row[3:6], row[6:9]]
            inverse = _invert3(matrix)
            np.testing.assert_allclose(
                np.array(_multiply(matrix, inverse)), np.eye(3), atol=1e-12)
            np.testing.assert_allclose(
                inverse, np.linalg.inv(np.array(matrix)), rtol=1e-12)

    def test_the_determinant_is_the_cell_volume(self):
        row = _tilted(8.399, 1.4)
        self.assertAlmostEqual(
            _determinant([row[0:3], row[3:6], row[6:9]]),
            float(np.linalg.det(np.array(row).reshape(3, 3))), places=9)

    def test_a_collapsed_cell_inverts_to_nothing(self):
        self.assertIsNone(_invert3([[0.0] * 3, [0.0] * 3, [0.0] * 3]))


class TestCellFrameMetrics(unittest.TestCase):
    """The three numbers a variable-cell run is judged on."""

    def test_volume_drift_is_measured_from_the_first_frame(self):
        metrics = cell_frame_metrics([_cube(10.0), _cube(9.0),
                                      _cube(11.5)], {}, 300.0)
        self.assertAlmostEqual(metrics['volume_drift'],
                               abs(11.5 ** 3 / 1000.0 - 1.0))
        self.assertEqual(metrics['cell_frames'], 3.0)

    def test_the_settling_frame_is_not_a_temperature_violation(self):
        metrics = cell_frame_metrics([_cube(10.0), _cube(10.0)],
                                     {'T_inst(K)': [330.0, 296.0, 302.0]},
                                     300.0)
        self.assertAlmostEqual(metrics['temperature_drift'], 4.0 / 300.0)

    def test_one_frame_is_not_enough_to_judge_anything(self):
        metrics = cell_frame_metrics([_cube(10.0)], {}, 300.0)
        self.assertNotIn('volume_drift', metrics)
        self.assertEqual(metrics['cell_frames'], 1.0)


class TestNptStabilityGate(_Scratch):
    """Only NPT examples are judged, and only when the cell moved."""

    def test_a_microcanonical_run_is_left_to_the_energy_verdict(self):
        self._write_yaml({'ensemble': 'nve', 'temperature': 300.0})
        result = self.staged(_xyz([_cube(10.0), _cube(4.0)]))
        self.judge(result)
        self.assertEqual(result['status'], 'pass')
        self.assertNotIn('[NPT', result['details'])

    def test_a_canonical_run_has_no_cell_to_judge(self):
        self._write_yaml({'ensemble': 'nvt', 'temperature': 300.0})
        result = self.staged(_xyz([_cube(10.0), _tilted(10.0, 5.0)]))
        self.judge(result)
        self.assertEqual(result['status'], 'pass')

    def test_a_non_md_jobtype_is_not_read_at_all(self):
        with open(self.yaml, 'w') as handle:
            handle.write('control:\n  jobtype: optimize\nmd:\n'
                         '  ensemble: npt\n  temperature: 300.0\n')
        result = self.staged(_xyz([_cube(10.0), _cube(4.0)]))
        self.judge(result)
        self.assertEqual(result['status'], 'pass')

    def test_a_missing_trajectory_is_reported_not_skipped(self):
        result = {'status': 'pass', 'details': ''}
        self.judge(result)
        self.assertIn('[NPT CHECK] no run.md.xyz', result['details'])
        self.assertEqual(result['status'], 'pass')

    def test_one_cell_frame_is_stated_as_unjudged(self):
        result = self.staged(_xyz([_cube(10.0)]))
        self.judge(result)
        self.assertIn('unjudged, 1 cell frames', result['details'])
        self.assertEqual(result['status'], 'pass')


class TestNptStabilityVerdict(_Scratch):
    """The four shapes the 2026-09-28 run produced, and a healthy one."""

    def test_a_collapsed_box_fails_even_though_the_run_passed(self):
        # md_lammps_npt_nh_mttk: 21 frames, volume down to 0.425 of its
        # target, axis to 1.66 Angstrom, and the suite reported pass.
        lattices = [_cube(10.0)]
        for index in range(1, 21):
            lattices.append(_cube(10.0 * (1.0 - 0.045 * index)))
        result = self.staged(_xyz(lattices),
                             _ensemble([(0.0, 300.0)]
                                       + [(41.34 * i, 300.0)
                                          for i in range(1, 21)]))
        self.judge(result)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('[NPT STABILITY] volume_drift', result['details'])
        self.assertIn('(volume_drift %.3f'
                      % result['box_metrics']['metrics']['volume_drift'],
                      result['details'])

    def test_a_blowing_box_and_a_flying_temperature_fail(self):
        # md_lammps_npt_csvr_pr: 5 frames, V/V0 up to 1.293, the frame
        # turned 11 degrees, T_inst reached 335111 K.
        lattices = [_tilted(8.399, tilt)
                    for tilt in [0.0, 0.6, 1.4, 2.4, 3.6]]
        rows = [(0.0, 310.3), (41.34, 900.0), (82.68, 4000.0),
                (124.02, 60000.0), (165.36, 335111.0)]
        result = self.staged(_xyz(lattices), _ensemble(rows))
        self.judge(result)
        self.assertEqual(result['status'], 'fail')
        metrics = result['box_metrics']['metrics']
        # The shape change alone costs no volume (a tilt keeps det), so
        # this one fails on the runaway temperature and the turning
        # frame, exactly as the example did before its engine died.
        self.assertAlmostEqual(metrics['volume_drift'], 0.0, places=9)
        self.assertGreater(metrics['temperature_drift'],
                           BOX_TOLERANCES['temperature_drift'])
        # The frame turned 12 degrees and is reported, not judged: a
        # healthy run turns at a comparable rate (see BOX_NOTES).
        self.assertGreater(metrics['cell_rotation_rad'],
                           BOX_NOTES['cell_rotation_rad'])
        self.assertIn('cell_rotation_rad', result['details'])

    def test_a_wide_but_legal_run_passes_and_says_what_it_saw(self):
        # md_lammps_npt_nh_berendsen: 8-16% of volume, no threshold
        # crossed. It is a pass, and it is a pass that leaves a trace.
        lattices = [_cube(10.0 * (1.0 - 0.002 * index))
                    for index in range(21)]
        rows = [(41.34 * index, 300.0) for index in range(21)]
        result = self.staged(_xyz(lattices), _ensemble(rows))
        self.judge(result)
        self.assertEqual(result['status'], 'pass')
        self.assertIn('[NPT STABILITY] volume_drift 0.115 over 0.100',
                      result['details'])
        self.assertIn('volume_drift', result['details'])

    def test_a_healthy_run_leaves_a_clean_trace(self):
        lattices = [_cube(10.0 * (1.0 + 0.001 * ((-1) ** index)))
                    for index in range(21)]
        rows = [(41.34 * index, 300.0) for index in range(21)]
        result = self.staged(_xyz(lattices), _ensemble(rows))
        self.judge(result)
        self.assertEqual(result['status'], 'pass')
        self.assertIn('all cell thresholds held', result['details'])
        self.assertLess(result['box_metrics']['metrics']['cell_rotation_rad'],
                        BOX_NOTES['cell_rotation_rad'])

    def test_the_frame_count_is_always_stated(self):
        lattices = [_cube(10.0), _cube(10.01)]
        result = self.staged(_xyz(lattices))
        self.judge(result)
        self.assertIn('over 2 cell frames', result['details'])

    def test_a_failing_result_is_not_relabelled(self):
        lattices = [_cube(10.0 * (1.0 - 0.05 * index))
                    for index in range(21)]
        result = self.staged(_xyz(lattices))
        result['status'] = 'error'
        result['details'] = 'engine died'
        self.judge(result)
        self.assertEqual(result['status'], 'error')

    def test_the_threshold_is_overridable_from_pmd_env(self):
        lattices = [_cube(10.0 * (1.0 - 0.02 * index))
                    for index in range(21)]
        result = self.staged(_xyz(lattices))
        self.judge(result, config={'box': {'volume_drift': 0.9}})
        self.assertEqual(result['status'], 'pass')

    def test_the_worst_layer_of_a_multicentre_run_decides(self):
        for title, factor in (('run', 1.0), ('run.1', 0.5)):
            lattices = [_cube(10.0 * factor), _cube(10.0 * factor * 0.999)]
            with open(os.path.join(self.dir, '%s.md.xyz' % title),
                      'w') as handle:
                handle.write(_xyz(lattices))
            handle = open(os.path.join(self.dir, '%s.md.ensemble' % title),
                          'w')
            handle.write(_ensemble([(0.0, 300.0), (41.34, 300.0)]))
            handle.close()
        result = {'status': 'pass', 'details': ''}
        self.judge(result)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['box_metrics']['metrics']['cell_frames'], 2.0)



class TestTheBarostatAndTheCellSearchAnswerOnePressureTheSameWay(unittest.TestCase):
    """Static `sigma = -p I` is the rest point for both paths, at one load.

    `dev/methods/CELL_OPT_DEVELOPMENT.md` section 5.7 says the NPT run and the
    variable-cell search must agree at the same pressure. What is decidable
    without an engine is the premise of that claim - that the two read the SAME
    tensor out of one stored stress and one declared load - and it is checked
    here on the two real drivers (`integrator/ensemble._berendsen` and
    `Optimizer._cell_gradient`) rather than on algebra, because an algebraic
    statement would still pass if either side dropped its `p I` term, which is
    the defect #381 found in the objective.

    The comparison is made with the kinetic part zero, and that is not a
    convenience: the barostat relaxes the TOTAL pressure (`p_int = p_kin -
    sigma`), while a cell criterion is deliberately built on the static stress -
    the framework asks the engine for a kinetic-free tensor for exactly this
    reason (`skills/lammps_skill/input_builder.py`, `compute Pconf all pressure
    NULL virial`, and `dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` section 6.4). The
    last criterion in this class keeps that difference on the record by showing
    the two paths DO part company once a temperature is in play.

    What this class does not settle is the endpoint claim - that a converged NPT
    run and a converged cell search stop on the same box. Neither carrier
    converges at an example budget on this host (the LAMMPS `water25` box bottoms
    out at 220.189 bar of residual with a 0.1 Bohr step bound; the xtb 81-atom
    box passed through 181.527 bar between two frames), so that half stays a
    round reading.
    """

    PRESSURE_BAR = 1700.0
    # The three row-length derivatives inside the six-slot lattice vector:
    # `utilities/lattice.CELL_COMPONENT_SLOTS` is (0, 3, 4, 6, 7, 8) over the
    # flattened 3x3, so the flattened diagonal (0, 4, 8) sits at slots 0, 2, 5.
    LENGTH_SLOTS = (0, 2, 5)

    def _state(self, offset_bar, pressure_bar=PRESSURE_BAR, hot=False):
        """One NPT state whose stress sits `offset_bar` off `-p I`.

        Cold unless `hot`: with the kinetic part zero the two paths are
        comparable, which is the whole point of the first three criteria.
        """
        from pyrai3md.qualification.verification._ensemble_helpers import (
            NATOM, make_npt_state)
        from pyrai3md.utilities.units import bar_to_au_pressure

        state = make_npt_state(barostat='berendsen', barostat_mode='isotropic',
                               pressure=pressure_bar,
                               velos=None if hot else [[0.0, 0.0, 0.0]] * NATOM)
        stress = (bar_to_au_pressure(-pressure_bar + offset_bar)) * np.eye(3)
        state.set_result({'gradient': np.zeros((NATOM, 3)).tolist(),
                          'stress_tensor': stress[None].tolist()})
        return state

    def _drives(self, state, pressure_bar=PRESSURE_BAR):
        """(barostat length drive, search length drive), both signed per axis."""
        import copy

        from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
        from pyrai3md.workflow.integrator.ensemble import _berendsen
        from pyrai3md.workflow.optimizer import get_optimizer
        from pyrai3md.qualification.verification._ensemble_helpers import DT

        lattice = (np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
                   * ANGSTROM_TO_BOHR)
        searched = get_optimizer('minima', maxiter=2, cell_variables=True,
                                 external_pressure=pressure_bar)
        gradient = np.asarray(
            searched._cell_gradient(state, lattice),
            dtype=np.float64)[list(self.LENGTH_SLOTS)]
        scaled = copy.deepcopy(state)
        _berendsen(scaled, float(DT))
        mu = np.asarray(scaled.runtime.ensemble['mu'], dtype=np.float64)
        # The search moves against its gradient; the barostat reports the
        # rescaling it applied, whose departure from 1 is its own drive.
        return np.diag(mu) - 1.0, -gradient

    def test_a_displaced_box_is_pushed_back_the_same_way_by_both(self):
        for offset in (2000.0, -2000.0):
            baro, search = self._drives(self._state(offset))
            self.assertTrue(np.all(np.abs(baro) > 0.0)
                            and np.all(np.abs(search) > 0.0),
                            'one of the paths did not move at all, so the '
                            'comparison below would compare nothing')
            self.assertEqual([float(np.sign(v)) for v in baro],
                             [float(np.sign(v)) for v in search],
                             'offset %+g bar: the barostat drives %s while the '
                             'cell search drives %s' % (
                                 offset, baro.tolist(), search.tolist()))

    def test_the_rest_point_is_shared(self):
        # Drop the `p I` term on either side and this fails by the whole raw
        # stress, which is what makes it a witness rather than a tautology.
        baro, search = self._drives(self._state(0.0))
        self.assertTrue(np.all(np.abs(baro) < 1e-12), baro.tolist())
        self.assertTrue(np.all(np.abs(search) < 1e-9), search.tolist())

    def test_the_same_load_is_honoured_at_a_second_pressure(self):
        # The agreement must not be an accident of one number.
        for pressure in (0.0, 5000.0):
            baro, search = self._drives(
                self._state(1500.0, pressure_bar=pressure),
                pressure_bar=pressure)
            self.assertEqual([float(np.sign(v)) for v in baro],
                             [float(np.sign(v)) for v in search],
                             'at %g bar: %s against %s' % (
                                 pressure, baro.tolist(), search.tolist()))
            quiet, still = self._drives(self._state(0.0, pressure_bar=pressure),
                                        pressure_bar=pressure)
            self.assertTrue(np.all(np.abs(quiet) < 1e-12)
                            and np.all(np.abs(still) < 1e-9),
                            'at %g bar the rest point moved: %s / %s' % (
                                pressure, quiet.tolist(), still.tolist()))

    def test_a_thermal_box_is_where_the_two_paths_divide_by_design(self):
        # The keep-the-difference-on-record criterion: with the kinetic pressure
        # in play the barostat is at rest where `p_kin - sigma = -p`, which the
        # static search cannot see, so at the search's own rest point the two
        # drives no longer match. If this ever agrees, someone folded the
        # temperature into the cell criterion.
        hot = self._state(0.0, hot=True)
        baro, search = self._drives(hot)
        self.assertTrue(np.all(np.abs(search) < 1e-9), search.tolist())
        self.assertTrue(np.all(np.abs(baro) > 1e-9),
                        'the barostat stopped where the static search stops even '
                        'with a temperature on the box - the kinetic term has '
                        'gone missing from one of the two readings')


def _angstrom_to_bohr():
    from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
    return ANGSTROM_TO_BOHR


def copy_deep(state):
    import copy
    return copy.deepcopy(state)


if __name__ == '__main__':
    unittest.main()
