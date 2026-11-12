"""
======================================================================
Test suite for the qualification NVE energy-conservation verdict
(pyrai3md/qualification/runner.py).

Tests cover:
- md.energies column reading: header line, malformed rows, missing file
- The spread / drift-rate metric set and the frame budget gating the
  rate metric
- The example gate: only YAML md runs whose ensemble is microcanonical
  and whose thermostat is off are judged
- The tolerance verdict, the recorded metrics payload, the worst-of
  behaviour over the layer records of a multicentre trajectory
- The artifact anchor: an example run writes into its run directory
  under control.title, so the lookup must follow those anchors
- Regression: the 7.0e-2 Hartree slide an excited-state reference
  energy produces must fail the shipped tolerance

Every trajectory is inlined; no calculation is executed.

Sep 26 2026
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
from unittest import mock

from pyrai3md.qualification import runner
from pyrai3md.qualification.runner import (
    ENERGY_TOLERANCES, RATE_MIN_FRAMES, _check_energy_conservation,
    energy_conservation_metrics, read_md_energies)

HEADER = ('%20s%28s%28s%28s%28s\n'
          % ('time', 'Epot', 'Ekin', 'Etot', 'Epot1,2,3...'))


def _rows(etot):
    """Build parsed frames from an Etot column."""
    return [[20.67 * (i + 1), value - 0.01, 0.01, value]
            for i, value in enumerate(etot)]


def _table(etot):
    """Render the md.energies text of an Etot column."""
    return HEADER + ''.join(
        '%20.2f%28.10f%28.10f%28.10f%28.10f\n'
        % (time, epot, ekin, value, epot - 0.35)
        for time, epot, ekin, value in _rows(etot))


class _Scratch(unittest.TestCase):
    """Base class providing an emptied staging directory."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, 'w') as f:
            f.write(text)
        return path


class TestReadMdEnergies(_Scratch):
    """Column reading of the md.energies record."""

    def test_rows_exclude_the_header(self):
        path = self.write('run.md.energies', _table([-76.1, -76.2, -76.3]))
        rows = read_md_energies(path)
        self.assertEqual([row[3] for row in rows], [-76.1, -76.2, -76.3])
        self.assertEqual(rows[0], _rows([-76.1])[0])

    def test_short_and_unparsable_lines_are_dropped(self):
        path = self.write('run.md.energies',
                          HEADER + 'garbage line here\nincomplete\n')
        self.assertEqual(read_md_energies(path), [])

    def test_missing_file_reads_empty(self):
        self.assertEqual(
            read_md_energies(os.path.join(self.dir, 'absent.energies')), [])


class TestConservationMetrics(unittest.TestCase):
    """Spread and drift-rate summary of one trajectory."""

    def test_drift_is_the_spread(self):
        metrics = energy_conservation_metrics(_rows([-76.10, -76.25, -76.05]))
        self.assertEqual(metrics['etot_frames'], 3)
        self.assertAlmostEqual(metrics['etot_drift'], 0.20)
        self.assertNotIn('etot_drift_rate', metrics)

    def test_rate_needs_the_frame_budget(self):
        rows = _rows([-76.1 - 1e-4 * i for i in range(RATE_MIN_FRAMES - 1)])
        self.assertNotIn('etot_drift_rate',
                         energy_conservation_metrics(rows))

    def test_rate_measures_a_linear_leak(self):
        rows = _rows([-76.1 - 1e-4 * i for i in range(RATE_MIN_FRAMES + 10)])
        metrics = energy_conservation_metrics(rows)
        self.assertAlmostEqual(metrics['etot_drift_rate'], 1e-4, delta=1e-9)


class TestExampleGate(_Scratch):
    """Only microcanonical YAML runs are judged."""

    def stage(self, md_body, etot, name='case'):
        self.write(f'{name}.yaml',
                   'control:\n  title: %s\n  jobtype: md\n'
                   'md:\n%s' % (name, md_body))
        self.write(f'{name}_example.md.energies', _table(etot))
        return os.path.join(self.dir, f'{name}.yaml')

    def test_default_ensemble_is_judged(self):
        input_file = self.stage('  step: 10\n', [-76.1, -76.2])
        result = {'status': 'pass'}
        _check_energy_conservation(result, input_file, {})
        self.assertEqual(result['metrics']['metrics']['etot_frames'], 2)

    def test_coupled_ensembles_are_skipped(self):
        for body in ('  ensemble: nvt\n',
                     '  ensemble: npt\n  barostat: mttk\n',
                     '  ensemble: nve\n  thermostat: csvr\n'):
            input_file = self.stage(body, [-76.1, -76.9], name='case')
            result = {'status': 'pass'}
            _check_energy_conservation(result, input_file, {})
            self.assertNotIn('metrics', result)
            self.assertEqual(result['status'], 'pass')

    def test_non_md_input_is_skipped(self):
        path = self.write('sp.yaml', 'control:\n  title: sp\n'
                                     '  jobtype: energy\n')
        result = {'status': 'pass'}
        _check_energy_conservation(result, path, {})
        self.assertNotIn('metrics', result)

    def test_run_without_records_is_skipped(self):
        self.write('case.yaml', 'control:\n  title: case\n'
                                '  jobtype: md\nmd:\n  ensemble: nve\n')
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, 'case.yaml'), {})
        self.assertNotIn('metrics', result)

    def test_a_fanned_out_run_is_judged_from_its_trajectory_dirs(self):
        # ninitcond > 1 writes one directory per trajectory and nothing at
        # the level the task was launched from, which is how md_parallel and
        # md_batch landed in the qualification round: their records existed,
        # the check looked one level deep and reported the run as unjudged
        # while still scoring it PASS.
        self.write('case.yaml', 'control:\n  title: case\n'
                                '  jobtype: md\nmd:\n  ensemble: nve\n')
        for index in (1, 2):
            trajectory = os.path.join(self.dir, 'case_example_%d' % index)
            os.makedirs(trajectory, exist_ok=True)
            with open(os.path.join(trajectory,
                                   'case_example.md.energies'),
                      'w') as handle:
                handle.write(_table([-76.1, -76.2]))
        result = {'status': 'pass'}
        _check_energy_conservation(result, os.path.join(self.dir, 'case.yaml'),
                                   {})
        self.assertIn('metrics', result)
        self.assertEqual(result['metrics']['metrics']['etot_frames'], 2)
        self.assertNotIn('ENERGY CHECK] no', result.get('details', ''))

    def test_a_benchmark_leg_says_so_instead_of_missing_records(self):
        # Every calibration leg reported "[ENERGY CHECK] no ... energies":
        # the frame search sits beside the reference input, where a
        # benchmark never writes example records, and the leg's own verdict
        # is the population comparison printed under it. The exemption has
        # to state that, not go looking for files it will not judge.
        input_file = self.write('case.yaml', 'control:\n  title: case\n'
                                             '  jobtype: md\n'
                                             'md:\n  ensemble: nve\n')
        result = {'status': 'pass'}
        _check_energy_conservation(result, input_file, {}, mode='benchmark')
        self.assertIn('not judged for a benchmark leg', result['details'])
        self.assertNotIn('ENERGY CHECK] no', result['details'])
        self.assertEqual(result['status'], 'pass')

    def test_classic_input_files_are_skipped(self):
        result = {'status': 'pass'}
        _check_energy_conservation(result, os.path.join(self.dir, 'a.input'),
                                   {})
        self.assertNotIn('metrics', result)


class TestConstraintGate(_Scratch):
    """A constraint that pumps energy by construction is not judged on Etot.

    The band the microcanonical criterion asserts is a statement about a closed
    system. Three declared capabilities of the constraint subsystem are not that
    system, and each of them is measured on a real run before it is exempted
    here: a position-only projection (1.78e-03 Hartree over 5 steps aperiodic,
    9.10e-02 over 600 steps in the water box), a metadynamics history (1.44e-02
    over 60 steps), a parameter ramp. A static bias, a static wall or a freeze
    stays judged, because the same measurement puts them inside the band
    (8.05e-05, 2.66e-04, 3.75e-04).
    """

    LEAK = [-76.1, -76.1 + 0.09]

    def stage(self, block, name='case'):
        self.write(f'{name}.yaml',
                   'control:\n  title: %s\n  jobtype: md\n'
                   'md:\n  ensemble: nve\n  barostat: "off"\n'
                   'system_constraint:\n%s' % (name, block))
        self.write(f'{name}_example.md.energies', _table(self.LEAK))
        return os.path.join(self.dir, f'{name}.yaml')

    def judge(self, block, name='case'):
        input_file = self.stage(block, name=name)
        result = {'status': 'pass'}
        _check_energy_conservation(result, input_file, {})
        return result

    def test_position_only_projection_is_explained_not_failed(self):
        result = self.judge('  shake_bond: [[1, 2, 1, 3]]\n')
        self.assertEqual(result['status'], 'pass')
        self.assertNotIn('metrics', result)
        self.assertIn('not judged', result['details'])
        self.assertIn('positions only', result['details'])

    def test_a_rattle_on_every_record_is_still_judged(self):
        result = self.judge('  shake_bond: [[1, 2, 1, 3]]\n'
                            '  shake_vel: rattle\n')
        self.assertEqual(result['status'], 'fail')
        self.assertIn('ENERGY CONSERVATION', result['details'])

    def test_rattle_must_cover_every_record(self):
        # Two constraint records, one of them position-only: the plan projects
        # the whole set together, so the run still leaks.
        result = self.judge('  shake_bond: [[1, 2, 1, 3]]\n'
                            '  shake_angle: [[2, 1, 3]]\n'
                            '  shake_vel: [rattle, none]\n')
        self.assertNotIn('metrics', result)
        self.assertIn('not judged', result['details'])

    def test_metadynamics_is_explained_not_failed(self):
        result = self.judge('  cv_bond: [[1, 2]]\n  cv_name: oh\n'
                            '  meta_cv: oh\n  meta_height: 0.0005\n')
        self.assertNotIn('metrics', result)
        self.assertIn('history potential', result['details'])

    def test_a_parameter_ramp_is_explained_not_failed(self):
        result = self.judge('  wall_type: sphere\n'
                            '  wall_sphere_radius: 2.8\n'
                            '  wall_sphere_radius_start: 2.8\n'
                            '  wall_sphere_radius_end: 2.0\n'
                            '  wall_atoms: [[1, 2, 3]]\n')
        self.assertNotIn('metrics', result)
        self.assertIn('ramp', result['details'])

    def test_a_static_bias_is_judged(self):
        self.LEAK = [-76.1, -76.1 + 8.0e-05]
        result = self.judge('  cv_bond: [[1, 2]]\n  cv_name: oh\n'
                            '  bias_cv: oh\n  bias_cv_ref: 0.9\n'
                            '  bias_cv_fc: 0.02\n')
        self.assertAlmostEqual(result['metrics']['metrics']['etot_drift'],
                               8.0e-05, delta=1e-09)

    def test_a_leak_under_a_static_constraint_still_fails(self):
        result = self.judge('  fix_atoms: 1\n')
        self.assertEqual(result['status'], 'fail')

    def test_no_block_is_judged(self):
        self.write('plain.yaml', 'control:\n  title: plain\n'
                                 '  jobtype: md\nmd:\n  ensemble: nve\n')
        self.write('plain_example.md.energies', _table(self.LEAK))
        result = {'status': 'pass'}
        _check_energy_conservation(
            result, os.path.join(self.dir, 'plain.yaml'), {})
        self.assertEqual(result['status'], 'fail')
        self.assertIn('etot_drift', result['metrics']['metrics'])

    def test_the_shipped_examples_land_on_the_expected_side(self):
        """The classification, read off the inputs that ship.

        Each side was decided from a measurement on that run's own
        `.md.energies`, recorded in `runner._constraint_breaks_conservation`.
        The assertion here is the classification, not the number: it keeps the
        exempt set and the judged set from drifting apart from the rule that
        sorts them, which is the failure mode where a leak stops being reported
        because nobody asked which side of the gate the case sits on.
        """
        import yaml

        import pyrai3md
        root = os.path.dirname(os.path.abspath(pyrai3md.__file__))
        exempt = ('constraint_md_shake.yaml',
                  'constraint_md_restart.yaml',
                  'constraint_md_restart_base.yaml',
                  'constraint_md_metad.yaml',
                  'constraint_md_metad_rmsd.yaml',
                  'constraint_md_torsion_intermolecular.yaml',
                  'constraint_md_periodic_shake.yaml',
                  'constraint_md_periodic_metad_gaussian.yaml',
                  'constraint_md_periodic_metad_rmsd.yaml',
                  'constraint_md_periodic_wall.yaml')
        judged = ('constraint_md_bias.yaml',
                  'constraint_md_freeze.yaml',
                  'constraint_md_rattle.yaml',
                  'constraint_md_torsion.yaml',
                  'constraint_md_wall_nonsphere.yaml',
                  'constraint_wall_center.yaml',
                  'constraint_md_periodic_bias.yaml',
                  'constraint_md_periodic_freeze.yaml',
                  'constraint_md_periodic_restart.yaml',
                  'constraint_md_periodic_restart_base.yaml',
                  'constraint_md_periodic_torn_angle.yaml',
                  'constraint_md_periodic_wall_box.yaml',
                  'constraint_md_periodic_wall_ellipsoid.yaml',
                  'constraint_md_periodic_wall_triclinic.yaml',
                  'constraint_md_periodic_dihedral_ethanol.yaml',
                  'constraint_md_periodic_xh_ethanol.yaml',
                  # NPT carriers: the constraint does not break the
                  # constant of motion on them either (the ensemble
                  # check skips them separately, for its own reason).
                  'constraint_md_periodic_npt_bias.yaml',
                  'constraint_md_periodic_npt_wall.yaml',
                  'constraint_md_periodic_npt_shake_slow.yaml',
                  'constraint_md_periodic_npt_angle.yaml',
                  'constraint_md_periodic_npt_freeze.yaml',
                  'constraint_md_periodic_npt_shake.yaml',
                  'constraint_md_periodic_nvt_freeze.yaml',
                  'constraint_md_periodic_nvt_shake.yaml')
        for names, expected in ((exempt, False), (judged, True)):
            for name in names:
                with self.subTest(example=name):
                    path = os.path.join(root, 'examples', name)
                    with open(path, 'r', encoding='utf-8') as handle:
                        spec = yaml.safe_load(handle)
                    reason = runner._constraint_breaks_conservation(spec)
                    self.assertEqual(reason is None, expected,
                                     '%s: %s' % (name, reason))

        # The two lists are the oracle, so a new example must not slip
        # past them: every shipped constraint case is labeled on exactly
        # one side. Its verdict equals the label by the loop above, which
        # is what makes this a label for the rule rather than a second
        # copy of it.
        labeled = set(exempt) | set(judged)
        shipped = set()
        for name in os.listdir(os.path.join(root, 'examples')):
            if not (name.startswith('constraint_') and name.endswith('.yaml')):
                continue
            with open(os.path.join(root, 'examples', name),
                      encoding='utf-8') as handle:
                spec = yaml.safe_load(handle) or {}
            jobtype = str((spec.get('control') or {}).get('jobtype') or '')
            # Only a dynamics run reaches this gate; a constraint example
            # whose jobtype is optimize is outside its scope by the rule
            # itself, not by a naming convention.
            if 'md' in jobtype.split():
                shipped.add(name)
        self.assertEqual(shipped - labeled, set(),
                         'unlabeled constraint examples')
        self.assertEqual(labeled - shipped, set(),
                         'labeled cases the package no longer ships')


class TestVerdict(_Scratch):
    """Tolerance comparison and the resulting status."""

    def run_case(self, etot, config=None, name='case'):
        self.write(f'{name}.yaml',
                   'control:\n  title: %s\n  jobtype: md\n'
                   'md:\n  ensemble: nve\n' % name)
        self.write(f'{name}_example.md.energies', _table(etot))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, f'{name}.yaml'),
                                   config or {})
        return result

    def test_conservating_run_passes(self):
        result = self.run_case([-76.1, -76.1002, -76.1001])
        self.assertEqual(result['status'], 'pass')
        self.assertLess(result['metrics']['metrics']['etot_drift'],
                        ENERGY_TOLERANCES['etot_drift'])

    def test_shipped_tolerance_catches_a_reference_energy_leak(self):
        # The signature of a mis-assigned excited-state reference:
        # Etot slides with the state gap, here 7.0e-2 Hartree.
        result = self.run_case([-76.81048945 - 0.0078 * i for i in range(10)])
        self.assertEqual(result['status'], 'fail')
        self.assertIn('ENERGY CONSERVATION', result['details'])
        self.assertAlmostEqual(
            result['metrics']['metrics']['etot_drift'], 0.0702, delta=1e-4)

    def test_pmd_env_overrides_the_default_tolerance(self):
        result = self.run_case([-76.1, -76.1002],
                               config={'energy': {'etot_drift': 1e-6}})
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(
            result['metrics']['tolerances']['etot_drift'], 1e-6)

    def test_a_driven_run_says_this_judgement_did_not_run(self):
        # The branch used to return silently, which made an NVT run with no
        # energy verdict read exactly like one that had passed. The record
        # now says which judgement did not run; the deliberately wide Etot
        # band below is what a removed early return would fail on.
        self.write('nvt.yaml', 'control:\n  title: nvt\n  jobtype: md\n'
                               'md:\n  ensemble: nvt\n'
                               '  thermostat: nose-hoover\n')
        self.write('nvt_example.md.energies', _table([-76.1, -76.9]))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, 'nvt.yaml'), {})
        self.assertEqual(result['status'], 'pass')
        self.assertIn('not judged', result['details'])
        self.assertIn('ensemble nvt and thermostat nose-hoover',
                      result['details'])

    def test_a_barostat_run_is_reported_by_its_ensemble_alone(self):
        # One question per clause: a run with no thermostat key names only
        # the thing that actually drives it.
        self.write('npt.yaml', 'control:\n  title: npt\n  jobtype: md\n'
                               'md:\n  ensemble: npt\n')
        self.write('npt_example.md.energies', _table([-76.1, -76.9]))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, 'npt.yaml'), {})
        self.assertEqual(result['status'], 'pass')
        self.assertIn('driven by ensemble npt', result['details'])
        self.assertNotIn('thermostat', result['details'])

    def test_a_failed_run_is_not_rewritten(self):
        result = self.run_case([-76.1, -76.1001])
        result.update({'status': 'error', 'details': 'crashed'})
        self.assertEqual(result['status'], 'error')

    def test_an_already_failing_run_keeps_its_verdict(self):
        input_file = os.path.join(self.dir, 'broken.yaml')
        self.write('broken.yaml', 'control:\n  title: broken\n'
                                  '  jobtype: md\nmd:\n  ensemble: nve\n')
        self.write('broken_example.md.energies', _table([-76.1, -76.9]))
        result = {'status': 'error', 'details': 'crashed'}
        _check_energy_conservation(result, input_file, {})
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['details'], 'crashed')

    def test_only_the_trajectory_total_is_judged(self):
        # A center is not a closed subsystem. The composite propagates its
        # atoms with forces that come from the whole box, so the center's
        # own Epot + Ekin (`workflow/records/output.py::_write_center_files`)
        # moves by design and only the sum is a constant of the motion.
        # Measured on examples/multicenter_pbc_nve.yaml over 60 recorded
        # frames: the composite slope is 2.906e-06 Ha/frame against the
        # 1.0e-05 tolerance while the water torn across the z face sweeps
        # 8.82e-03 Ha of its own band -- judging the worst file failed a run
        # whose total was flat.
        self.write('case.yaml', 'control:\n  title: case\n'
                                '  jobtype: md\nmd:\n  ensemble: nve\n')
        self.write('case_example.md.energies', _table([-76.1, -76.1001]))
        self.write('case_example.md.1.energies', _table([-76.1, -76.2]))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, 'case.yaml'), {})
        self.assertEqual(result['status'], 'pass')
        metrics = result['metrics']['metrics']
        self.assertAlmostEqual(metrics['etot_drift'], 1.0e-4, places=8)
        self.assertAlmostEqual(metrics['etot_center_drift'], 0.1)
        self.assertNotIn('etot_center_drift',
                         result['metrics']['tolerances'])

    def test_a_leak_in_the_total_still_fails_beside_a_quiet_center(self):
        # The negative control for the rule above: excluding the center
        # records cannot be mistaken for a blanket pass, because a real
        # leak shows in the trajectory total.
        self.write('leak.yaml', 'control:\n  title: leak\n'
                               '  jobtype: md\nmd:\n  ensemble: nve\n')
        self.write('leak_example.md.energies', _table([-76.1, -76.9]))
        self.write('leak_example.md.1.energies', _table([-76.1, -76.1001]))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, 'leak.yaml'), {})
        self.assertEqual(result['status'], 'fail')
        self.assertIn('etot_drift', result['details'])

    def test_a_run_with_only_center_records_is_reported_unjudged(self):
        # Silently passing over a verdict nobody could reach is the failure
        # mode this check has been caught at once already, so the absence of
        # a trajectory total is stated rather than judged against it.
        self.write('parts.yaml', 'control:\n  title: parts\n'
                                 '  jobtype: md\nmd:\n  ensemble: nve\n')
        self.write('parts_example.md.1.energies', _table([-76.1, -76.2]))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, 'parts.yaml'), {})
        self.assertEqual(result['status'], 'pass')
        self.assertIn('no trajectory total', result['details'])


class TestLongRunJudgesTheSlope(_Scratch):
    """A recorded run of RATE_MIN_FRAMES or more is judged on its trend.

    Max minus min over a long NVE record is dominated by the bounded
    kinetic/potential exchange a symplectic integrator always shows, and
    it cannot tell that exchange from energy leaving the system. The
    shipped example is the case: a 2.004e-03 Hartree band whose width is
    the same in its second half as in its first, against a slope of
    3.36e-08 Hartree per frame. The band stays in the record; it no
    longer decides on its own once the trend is measurable.
    """

    def run_case(self, etot, name='case'):
        self.write(f'{name}.yaml',
                   'control:\n  title: %s\n  jobtype: md\n'
                   'md:\n  ensemble: nve\n' % name)
        self.write(f'{name}_example.md.energies', _table(etot))
        result = {'status': 'pass'}
        _check_energy_conservation(result,
                                   os.path.join(self.dir, f'{name}.yaml'), {})
        return result

    def test_bounded_oscillation_over_a_thousand_frames_passes(self):
        etot = [-0.2331 + 1.0e-3 * math.sin(2 * math.pi * i / 200.0)
                for i in range(1001)]
        result = self.run_case(etot)
        metrics = result['metrics']['metrics']
        self.assertAlmostEqual(metrics['etot_drift'], 2.0e-3, delta=2e-6)
        self.assertLess(metrics['etot_drift_rate'],
                        ENERGY_TOLERANCES['etot_drift_rate'])
        self.assertEqual(result['status'], 'pass')
        self.assertIn('judged on 1001 frames', result['details'])
        # Recorded, not judged: the band is visible to whoever reads it.
        self.assertIn('etot_drift', metrics)
        self.assertNotIn('etot_drift', result['metrics']['tolerances'])

    def test_a_real_leak_over_the_same_frames_still_fails(self):
        etot = [-0.2331 - 2.0e-5 * i for i in range(1001)]
        result = self.run_case(etot)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('etot_drift_rate', result['details'])

    def test_a_short_record_keeps_the_band_as_the_verdict(self):
        # Below RATE_MIN_FRAMES there is no trend to judge on, and the
        # handful of frames that exist are all there is.
        etot = [-0.2331, -0.2338, -0.2345, -0.2352, -0.2359, -0.2366,
                -0.2373, -0.2380, -0.2387, -0.2394, -0.2401]
        result = self.run_case(etot)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('etot_drift 7.000e-03', result['details'])
        self.assertIn('etot_drift', result['metrics']['tolerances'])

    def test_the_rate_threshold_is_the_boundary(self):
        # One waveform, two record lengths, both with the same 2e-3 band:
        # the only difference is whether the trend is measurable at all.
        # (A long-period sine would not do here: 49 frames of it never
        # reach the band the gate is being tested against.)
        def wave(count):
            return [-0.2331 + 3.5e-3 * math.sin(2 * math.pi * i / 20.0)
                    for i in range(count)]

        short = self.run_case(wave(RATE_MIN_FRAMES - 1), 'short')
        self.assertAlmostEqual(short['metrics']['metrics']['etot_drift'],
                               7.0e-3, delta=1e-4)
        self.assertEqual(short['status'], 'fail')
        self.assertIn('etot_drift', short['metrics']['tolerances'])

        edge = self.run_case(wave(RATE_MIN_FRAMES), 'edge')
        self.assertAlmostEqual(edge['metrics']['metrics']['etot_drift'],
                               7.0e-3, delta=1e-4)
        self.assertEqual(edge['status'], 'pass')
        self.assertNotIn('etot_drift', edge['metrics']['tolerances'])


class TestArtifactAnchors(_Scratch):
    """Example artifacts resolve from the run directory and the title.

    The classes above stage the YAML and its records side by side under
    the same stem, which is exactly the arrangement the remote never
    has: an example runs as 'cd <work_dir>/examples && pyrai3md
    <source_dir>/<name>.yaml' and writes everything there under its
    control.title. Resolving beside the input therefore judged nothing.
    """

    SOURCE = ('control:\n  title: case_example\n  jobtype: md\n'
              'md:\n  step: 10\n')

    def stage_split(self, etot):
        """Put the YAML in the source dir, the records in the run dir."""
        yaml_path = self.write('case.yaml', self.SOURCE)
        run_dir = os.path.join(self.dir, 'run')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'case_example.md.energies'),
                  'w') as handle:
            handle.write(_table(etot))
        return yaml_path, run_dir

    def test_run_dir_and_title_reach_the_records(self):
        yaml_path, run_dir = self.stage_split([-76.1, -76.2])
        result = {'status': 'pass'}
        _check_energy_conservation(result, yaml_path, {},
                                   base_dir=run_dir, title='case_example')
        self.assertEqual(result['metrics']['metrics']['etot_frames'], 2)

    def test_input_relative_lookup_judges_nothing(self):
        # The pre-fix anchor: same YAML, same records, resolved beside
        # the input under the input stem.
        yaml_path, run_dir = self.stage_split([-76.1, -76.2])
        result = {'status': 'pass'}
        _check_energy_conservation(result, yaml_path, {})
        self.assertNotIn('metrics', result)
        self.assertIn('[ENERGY CHECK]', result['details'])

    def test_missing_records_leave_a_trace_instead_of_silence(self):
        yaml_path = self.write('case.yaml', self.SOURCE)
        result = {'status': 'pass'}
        _check_energy_conservation(result, yaml_path, {},
                                   base_dir=self.dir, title='case_example')
        self.assertIn('[ENERGY CHECK] no case_example.md.energies',
                      result['details'])
        self.assertNotIn('metrics', result)
        self.assertEqual(result['status'], 'pass')

    def test_skill_test_anchor_is_unchanged(self):
        # No run_dir/title: records resolve beside the input under the
        # input stem, which is how skill tests stage them.
        path = self.write('case.yaml', self.SOURCE)
        self.write('case.md.energies', _table([-76.1, -76.2]))
        result = {'status': 'pass'}
        _check_energy_conservation(result, path, {})
        self.assertEqual(result['metrics']['metrics']['etot_frames'], 2)

    def test_task_anchors_carry_a_violation_end_to_end(self):
        # Through _run_skill_test: without the task's run_dir/title the
        # oversized drift below would never be seen and the run would
        # report a clean pass.
        yaml_path, run_dir = self.stage_split([-76.1] + [-76.6] * 9)
        task = {'name': 'example_case', 'mode': 'pyrai3md_yaml',
                'input_file': yaml_path, 'run_dir': run_dir,
                'title': 'case_example',
                # Every runnable task is planned with a script (the
                # planner writes one); the mocked subprocess stands in
                # for it, so its content is irrelevant here.
                'script': os.path.join(run_dir, 'case.sh')}
        proc = mock.Mock(returncode=0, stdout='', stderr='')
        with mock.patch.object(runner.subprocess, 'run',
                               return_value=proc):
            result = runner._run_skill_test(task, {}, {})
        self.assertEqual(result['status'], 'fail')
        self.assertIn('ENERGY CONSERVATION', result['details'])
        self.assertAlmostEqual(
            result['metrics']['metrics']['etot_drift'], 0.5)


class TestShippedTolerances(unittest.TestCase):
    """The defaults match what pmd_env.yaml documents."""

    def test_tolerance_set(self):
        self.assertEqual(ENERGY_TOLERANCES,
                         {'etot_drift': 5e-3, 'etot_drift_rate': 1e-5})


class TestMinimumImageTrace(_Scratch):
    """The cutoff-versus-half-box record: stated, never used to fail a run.

    The shipped force field declares a real-space cutoff the small example
    box cannot hold, and deciding that is a force-field question, not a
    suite's. What the suite owes is that the fact is visible in the result
    instead of being discovered later through an energy that will not
    conserve.
    """

    INIT = ('units real\n'
            'pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0 '
            'buck/coul/long 10.0\n'
            'kspace_style pppm 1e-4\n')

    def _case(self, cell):
        return self.write('case.yaml', (
            'control:\n  jobtype: md\n  theory: [lammps]\n'
            'molecule:\n  cell: [%s]\n'
            % ', '.join(repr(float(value)) for value in cell)))

    def test_cutoffs_come_from_the_style_line(self):
        path = self.write('h2o.in.init', self.INIT)
        self.assertEqual(runner.read_pair_cutoffs(path),
                         [9.0, 10.0, 10.0, 10.0])
        self.assertEqual(
            runner.read_pair_cutoffs(os.path.join(self.dir, 'absent')), [])

    def test_a_violating_cutoff_is_recorded_and_the_verdict_holds(self):
        result = {'status': 'pass', 'details': 'ran'}
        self.write('h2o.in.init', self.INIT)
        runner._check_minimum_image(result, self._case(
            [8.399, 8.399, 8.399, 90.0, 90.0, 90.0]), self.dir, 'case')
        self.assertEqual(result['status'], 'pass')
        self.assertIn('[PBC]', result['details'])
        self.assertIn('10.000 A', result['details'])
        self.assertIn('4.199', result['details'])

    def test_a_cutoff_inside_the_half_box_leaves_no_trace(self):
        result = {'status': 'pass', 'details': 'ran'}
        self.write('h2o.in.init', self.INIT)
        runner._check_minimum_image(result, self._case(
            [20.0, 20.0, 20.0, 90.0, 90.0, 90.0]), self.dir, 'case')
        self.assertEqual(result['details'], 'ran')

    def test_a_tilted_cell_is_left_alone(self):
        # Half of an axis is not the perpendicular width of a tilted cell,
        # and guessing with the axis would report a violation that may not
        # be one.
        result = {'status': 'pass', 'details': 'ran'}
        self.write('h2o.in.init', self.INIT)
        runner._check_minimum_image(result, self._case(
            [20.0, 20.0, 20.0, 80.0, 95.0, 100.0]), self.dir, 'case')
        self.assertEqual(result['details'], 'ran')

    def test_no_staged_template_leaves_no_trace(self):
        result = {'status': 'pass', 'details': 'ran'}
        runner._check_minimum_image(result, self._case(
            [8.399, 8.399, 8.399, 90.0, 90.0, 90.0]), self.dir, 'case')
        self.assertEqual(result['details'], 'ran')

    def test_classic_input_runs_are_not_traced(self):
        # The trace reads the example's own declaration; a classic .input
        # run has none in this shape, and the input mode is deliberately
        # outside the post-run gates.
        result = {'status': 'pass', 'details': 'ran'}
        self.write('h2o.in.init', self.INIT)
        runner._check_minimum_image(
            result, os.path.join(self.dir, 'case.input'), self.dir, 'case')
        self.assertEqual(result['details'], 'ran')


if __name__ == '__main__':
    unittest.main()
