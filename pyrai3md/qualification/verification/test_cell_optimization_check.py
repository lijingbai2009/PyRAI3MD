"""
======================================================================
Test suite for the qualification variable-cell verdict
(pyrai3md/qualification/runner.py::check_cell_optimization).

A run that declares `optimize_cell` and silently loses the switch looks
identical to a converged fixed-cell relaxation: the engine answers, the
atoms come back quieter, and every other judgement in the suite is green.
This happened for real in the shape of an invented keyword - the
optimizer read `optimize_cell` while the flattener wrote `opt_optimize_cell`
(dev/methods/OPTIMIZE_DEVELOPMENT.md section 11) - and nothing downstream
could see it. The two things this check reads are the framework's own
status record: the cell columns, and whether the volume moved.

Fixtures are inlined from a measured variable-cell relaxation on a 25
Angstrom TIP3P box (12 steps, CHANGELOG #341's vehicle), so the header and
the numbers are the bytes the optimizer actually wrote rather than a
hand-made table. No calculation is executed here.

Oct 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.qualification.runner import check_cell_optimization
from pyrai3md.utilities.opt_formatter import STATUS_CELL_COLUMNS
from pyrai3md.utilities.print_results import WRITER_CELL, read_status

# Verbatim from a measured run: the header the cell branch prints, and the
# first and last of its twelve rows.
CELL_HEADER = ('  step                energy                 shift'
               '                 max_d                 rms_d'
               '                 max_g                 rms_g'
               '           cell_grms               cell_gmax'
               '           cell_drms               cell_dmax'
               '              volume')
CELL_ROW_FIRST = ('     1  -8.265022119597521e+00'
                  '   0.000000000000000e+00'
                  '   1.501118855422821e-02'
                  '   3.835617011075697e-03'
                  '   5.003729518076071e-02'
                  '   1.278539003691899e-02'
                  '   1.248755456624326e-01'
                  '   1.904484196632695e-01'
                  '   3.746266369872979e-02'
                  '   5.713452589898085e-02'
                  '   1.572157528502303e+04')
CELL_ROW_LAST = ('    12  -8.748313606734245e+00'
                 '  -4.695139132770976e-05'
                 '   3.124903138044711e-02'
                 '   4.546205697565238e-04'
                 '   2.733211026401777e-02'
                 '   6.009317167019814e-03'
                 '   2.319898442599715e-02'
                 '   3.446061380443190e-02'
                 '   4.999064057336139e-06'
                 '   8.043283726668378e-06'
                 '   1.577465170314372e+04')

# The seven columns a fixed-cell run writes: this is what a dropped switch
# leaves behind, and the shape the check exists to catch.
ATOM_HEADER = ('  step                energy                 shift'
               '                 max_d                 rms_d'
               '                 max_g                 rms_g')
ATOM_ROW = ('     1  -8.265022119597521e+00   0.000000000000000e+00'
            '   1.501118855422821e-02   3.835617011075697e-03'
            '   5.003729518076071e-02   1.278539003691899e-02')


def _spec(**optimize):
    """An example YAML with the given `optimize` section."""
    lines = ['control:', '  title: cell_case', '  jobtype: optimize', 'optimize:']
    for key, value in optimize.items():
        rendered = ('true' if value is True
                    else 'false' if value is False else repr(value))
        lines.append('  %s: %s' % (key, rendered))
    return '\n'.join(lines) + '\n'


class CellCheckCase(unittest.TestCase):
    """Shared fixture: a launch directory, a spec, a status record."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='cell_check_')
        self.spec_file = os.path.join(self.root, 'cell_case.yaml')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _judge(self, spec, status=None, status_name='opt_status.txt',
               subdir=None, result=None):
        with open(self.spec_file, 'w') as handle:
            handle.write(spec)
        run_dir = self.root
        if subdir:
            run_dir = os.path.join(self.root, subdir)
            os.makedirs(run_dir, exist_ok=True)
        if status is not None:
            with open(os.path.join(run_dir, status_name), 'w') as handle:
                handle.write(status)
        outcome = {'status': 'pass'} if result is None else result
        check_cell_optimization(outcome, self.spec_file, run_dir, 'cell_case')
        return outcome

    def _cell_record(self, first=CELL_ROW_FIRST, last=CELL_ROW_LAST):
        return '\n'.join([CELL_HEADER, first, last]) + '\n'


class TestTheGate(CellCheckCase):
    """Which runs this check reads at all."""

    def test_a_fixed_cell_optimization_is_left_alone(self):
        outcome = self._judge(_spec(method='minima'),
                              status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass')
        self.assertNotIn('[CELL]', outcome.get('details') or '')

    def test_a_dynamics_run_is_out_of_this_verdict(self):
        # The spec carries a moving-box status record *and* an `optimize`
        # block with the switch written, so the only thing that keeps this run
        # unjudged is the jobtype: a barostat owns its cell and NPT already has
        # its own verdict (`_check_npt_stability`), and judging the same box
        # twice would report two answers to one question.
        spec = ('control:\n  title: md_case\n  jobtype: md\n'
                'md:\n  ensemble: npt\n'
                'optimize:\n  optimize_cell: true\n')
        outcome = self._judge(spec, status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass')
        self.assertNotIn('[CELL]', outcome.get('details') or '')
        self.assertNotIn('cell_metrics', outcome)

    def test_a_record_is_only_read_from_a_yaml_spec(self):
        # The input carrier hands the runner a generated classic input; the
        # judgement comes from the source YAML through the spec anchor, so a
        # non-YAML path must be ignored rather than guessed at.
        outcome = self._judge(_spec(optimize_cell=True),
                              status=self._cell_record())
        self.assertIn('[CELL]', outcome.get('details') or '')
        text_file = os.path.join(self.root, 'cell_case.input')
        with open(text_file, 'w') as handle:
            handle.write('&control\n  jobtype optimize\n&end\n')
        quiet = {'status': 'pass'}
        check_cell_optimization(quiet, text_file, self.root, 'cell_case')
        self.assertNotIn('[CELL]', quiet.get('details') or '')


class TestTheTwoMeasurements(CellCheckCase):
    """The cell columns and the volume, which are what the switch leaves."""

    def test_a_run_that_declares_the_switch_needs_a_record(self):
        outcome = self._judge(_spec(optimize_cell=True), status=None)
        self.assertEqual(outcome['status'], 'fail')
        self.assertIn('no opt_status.txt', outcome['details'])

    def test_a_record_without_cell_columns_fails_by_name(self):
        # The dropped switch, in the bytes it actually produces: a seven-column
        # table, atoms quiet, box untouched. Every other verdict is green.
        outcome = self._judge(_spec(optimize_cell=True),
                              status='\n'.join([ATOM_HEADER, ATOM_ROW,
                                                ATOM_ROW]) + '\n')
        self.assertEqual(outcome['status'], 'fail')
        self.assertIn('optimize_cell', outcome['details'])
        for name in STATUS_CELL_COLUMNS:
            self.assertIn(name, outcome['details'])

    def test_the_record_is_read_from_the_launch_directory(self):
        # The location the round actually produces: every status record of
        # tmp_20261015_100320 sits in the leg directory the run started in.
        outcome = self._judge(_spec(optimize_cell=True),
                              status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass')

    def test_a_record_in_a_subtree_does_not_answer_for_this_run(self):
        # A sibling carrier's record belongs to that carrier's run. Reading it
        # here would let one run's box answer another's question, which is the
        # same mistake as the shared calculation directory of #354 in a new place.
        other = os.path.join(self.root, 'pyrai3md_input')
        os.makedirs(other, exist_ok=True)
        with open(os.path.join(other, 'opt_status.txt'), 'w') as handle:
            handle.write(self._cell_record())
        outcome = self._judge(_spec(optimize_cell=True), status=None)
        self.assertEqual(outcome['status'], 'fail')
        self.assertIn('no opt_status.txt', outcome['details'])

    def test_a_moving_box_is_reported_with_the_numbers_it_gave(self):
        outcome = self._judge(_spec(optimize_cell=True),
                              status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass')
        metrics = outcome['cell_metrics']
        self.assertAlmostEqual(metrics['volume_first'], 15721.57528502303)
        self.assertAlmostEqual(metrics['volume_last'], 15774.65170314372)
        self.assertGreater(metrics['volume_drift'], 0.0)
        self.assertAlmostEqual(metrics['cell_gmax'],
                               3.446061380443190e-02)
        self.assertEqual(metrics['steps'], 2)

    def test_a_box_that_never_moved_fails_by_name(self):
        # The other half of the same defect: columns present because the box
        # was a variable, and a lattice that did not move.
        outcome = self._judge(_spec(optimize_cell=True),
                              status=self._cell_record(first=CELL_ROW_FIRST,
                                                       last=CELL_ROW_FIRST))
        self.assertEqual(outcome['status'], 'fail')
        self.assertIn('never moved', outcome['details'])

    def test_a_single_step_record_is_reported_as_unjudged(self):
        outcome = self._judge(_spec(optimize_cell=True),
                              status='\n'.join([CELL_HEADER,
                                                CELL_ROW_FIRST]) + '\n')
        self.assertEqual(outcome['status'], 'pass')
        self.assertIn('unjudged', outcome['details'])


class TestReaderAndPrecedence(CellCheckCase):
    """The reader, and who owns the message when a run already failed."""

    def test_the_reader_keeps_the_named_columns(self):
        record = _read_cell_status_from(self._cell_record(), self.root)
        self.assertEqual(record.columns, CELL_HEADER.split())
        self.assertEqual(record.writer, WRITER_CELL)
        self.assertEqual(len(record.rows), 2)
        self.assertAlmostEqual(record.rows[1]['volume'], 15774.65170314372)
        self.assertAlmostEqual(record.rows[0]['cell_grms'], 1.248755456624326e-01)

    def test_a_short_row_is_counted_not_filled(self):
        record = _read_cell_status_from(
            '\n'.join([CELL_HEADER, CELL_ROW_FIRST, ATOM_ROW]) + '\n',
            self.root)
        self.assertEqual(len(record.rows), 1)
        self.assertEqual(record.malformed, [{'line': 3, 'tokens': 7}])

    def test_a_line_that_does_not_match_is_reported_in_the_verdict(self):
        # A killed run leaves a cut final row. The judge still reads the steps
        # it has, but the sentence has to say a line was refused: reporting the
        # relaxation as if the record were complete is how a truncated run
        # reads as a short one.
        record = self._cell_record() + '     3  -8.5e+00\n'
        outcome = self._judge(_spec(optimize_cell=True), status=record)
        self.assertEqual(outcome['status'], 'pass')
        self.assertIn('do not match the 12-column header',
                      outcome['details'])
        self.assertEqual(outcome['cell_metrics']['malformed_rows'], 1)

    def test_an_engine_record_under_the_same_name_is_refused(self):
        # openqp writes its own convergence table under this plain file name
        # inside its calculation directory. Its rows are seven tokens against an
        # eleven-token header; taking one for the framework's would answer for a
        # box the framework never moved.
        engine = (' Step           Energy            Shift      RMSD Step'
                  '       Max Step      RMSD Grad       Max Grad\n'
                  + '-' * 40 + '\n'
                  + '    1     -76.85494948     -76.85494948       0.000000\n')
        outcome = self._judge(_spec(optimize_cell=True), status=engine)
        self.assertEqual(outcome['status'], 'fail')
        self.assertIn('not the framework table', outcome['details'])
        self.assertNotIn('cell_metrics', outcome)

    def test_an_existing_failure_owns_its_message(self):
        outcome = {'status': 'fail', 'details': 'engine died'}
        self._judge(_spec(optimize_cell=True), status=None, result=outcome)
        self.assertEqual(outcome['details'], 'engine died')


def _read_cell_status_from(record, root):
    path = os.path.join(root, 'status_probe.txt')
    with open(path, 'w') as handle:
        handle.write(record)
    return read_status(path)


if __name__ == '__main__':
    unittest.main()


def _progress_block(pressure, rms, maximum, converged=True, strain=6.0e-5):
    """One optimization-progress block, in the rows the logger prints.

    Written from `kernel/logger.py::log_optimize_step`'s own format string
    (`'  %-18s %20.8f'` plus the threshold pair), so a renamed row shows up
    here as the reader missing it rather than as a passing run.
    """
    rows = []
    for key, value in (('External Pressure', pressure),
                       ('Cell Stress RMS', rms),
                       ('Cell Stress Max', maximum),
                       ('Cell Strain RMS', strain),
                       ('Cell Strain Max', strain)):
        rows.append('  %-18s %20.8f' % (key + ':', value))
    rows.append('  %-18s %20s' % ('Converged:',
                                  'True' if converged else 'False'))
    return ('\n  &optimization progress\n%s\n  %-18s %20d\n%s\n'
            % ('-' * 79, 'Step:', 12, '\n'.join(rows)))


class TestAPressureLegIsJudgedAgainstItsPressure(CellCheckCase):
    """`opt_pressure` changes what the box is asked to stop at, and the verdict.

    Round tmp_20261017_094721's pressure leg is the reason this class exists:
    three steps into a 10 kbar compression it passed on the two facts every
    variable-cell leg is read on (the cell columns arrived, the box moved),
    while its own run log still carried 9972.6 bar of `sigma + p I` - the whole
    applied pressure. A leg that declares a pressure is now asked for the
    residual, the direction the volume took, the pressure the search says it is
    carrying, and its own word that it converged.
    """

    def _judge_pressure(self, rms=12.0, maximum=20.0, pressure=10000.0,
                        converged=True, growing=False, log=None, **extra):
        spec_args = {'method': 'minima', 'optimize_cell': True,
                     'opt_pressure': pressure, 'opt_csrms': 40.0,
                     'opt_csmax': 60.0}
        spec_args.update(extra)
        record = (self._cell_record() if growing else
                  self._cell_record(first=CELL_ROW_LAST, last=CELL_ROW_FIRST))
        # The shipped fixture grew from 15721.575 to 15774.652 A^3, so asking
        # for the two rows in the other order is what makes it shrink.
        if log is None:
            # Two blocks, because the direction clause weighs the first recorded
            # step against the last: a single block can say nothing about either,
            # and a fixture that wrote one would let the clause pass a run whose
            # residual never moved or let it accuse one that did.
            log = (_progress_block(pressure, rms * 10.0, maximum * 10.0)
                   + _progress_block(pressure, rms, maximum, converged))
        if log:
            # The block belongs to the run, so it is on disk before the single
            # call that reads both products - a second call would append a
            # second verdict sentence to the same result.
            with open(os.path.join(self.root, 'cell_case.log'), 'w') as h:
                h.write(log)
        return self._judge(_spec(**spec_args), status=record)

    def test_a_converged_pressure_leg_passes_and_says_the_numbers(self):
        outcome = self._judge_pressure()
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertIn('pressure 10000.0 bar', outcome['details'])
        self.assertIn('12.0 bar (rms) / 20.0 bar (max) against 40.0 / 60.0',
                      outcome['details'])
        metrics = outcome['cell_metrics']
        self.assertEqual(metrics['pressure_target_bar'], 10000.0)
        self.assertEqual(metrics['pressure_residual_max_bar'], 20.0)
        self.assertTrue(metrics['pressure_converged'])

    def test_a_five_step_run_outside_the_band_passes_and_still_prints_it(self):
        # Ruled on 2026-10-19: at the `maxit: 5` these legs run under, the band
        # and the convergence flag are read and printed, not judged. The fixture
        # is rounds 15-17's own reading - endpoint residual 9972.6 bar against
        # 40/60, Converged: False - arriving from ten times that at its first
        # recorded step, which is the direction the clause still judges. Round
        # tmp_20261019_122659 is why the band came out of the verdict: its
        # isotropic leg ran 10818.1 -> 9732.0 -> 6971.1 -> 181.5 -> 1319.9 bar,
        # so the endpoint that used to be failed had already crossed its rest
        # point one step earlier and come back. The witness that this is not the
        # clause going blind is the one after it: a residual that does NOT come
        # down still fails.
        outcome = self._judge_pressure(rms=9972.6, maximum=9972.6,
                                       converged=False)
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        details = outcome['details']
        self.assertIn('NOT JUDGED (functional-test budget)', details)
        self.assertIn('9972.6 bar of residual', details)
        self.assertIn('opt_csrms 40.0 bar', details)
        self.assertIn('Converged: False', details)
        self.assertNotIn('FAILED', details)
        metrics = outcome['cell_metrics']
        self.assertEqual(metrics['pressure_residual_rms_bar'], 9972.6)
        self.assertEqual(metrics['pressure_residual_first_bar'], 99726.0)
        self.assertFalse(metrics['pressure_converged'])

    def test_a_dropped_pressure_keyword_is_named(self):
        # The input asks for 10 kbar; the search reports it is carrying none.
        # Every number below the header would still read as a sane relaxation,
        # which is exactly why the row has to be compared with the request.
        outcome = self._judge_pressure(log=_progress_block(0.0, 12.0, 20.0))
        self.assertEqual(outcome['status'], 'fail', outcome.get('details'))
        self.assertIn('did not reach the search', outcome['details'])

    def test_a_box_that_grew_toward_its_rest_point_is_not_called_wrong(self):
        # The witness #391 replaced. Round tmp_20261018_200317's loaded leg grew
        # from 1157.625 to 1173.837737 A^3 under 1 kbar while its residual fell
        # 11518.1 -> 653.4 bar: the box starts more compressed than the load, so
        # its rest point lies at a larger volume and expansion is the correct
        # direction. The old clause accused exactly this run of a sign error.
        outcome = self._judge_pressure(growing=True, rms=30.0, maximum=40.0,
                                       pressure=1000.0)
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertNotIn('wrong way', outcome['details'])
        self.assertNotIn('grew the volume', outcome['details'])

    def test_a_residual_that_did_not_come_down_is_named(self):
        # The same fixture with the two blocks swapped: the endpoint residual is
        # worse than the first step's, which no threshold on the endpoint alone
        # would separate from a run heading the right way.
        outcome = self._judge_pressure(
            log=_progress_block(1000.0, 653.4, 653.4)
            + _progress_block(1000.0, 9000.0, 9000.0, converged=False))
        self.assertEqual(outcome['status'], 'fail', outcome.get('details'))
        self.assertIn('away from its rest point', outcome['details'])

    def test_a_leg_with_no_zero_pressure_twin_says_so(self):
        # This fixture lays its record in the run directory with no sibling
        # example beside it, so the paired half of the clause has nothing to
        # compare against and reports that rather than passing quietly.
        outcome = self._judge_pressure(pressure=1000.0, rms=30.0,
                                       maximum=40.0)
        self.assertIn('paired against nothing', outcome['details'])

    def test_a_leg_without_a_declared_pressure_keeps_its_verdict_byte_identical(self):
        # The other four variable-cell examples cap their budgets on purpose and
        # are judged on the machinery. Their sentence must not move when this
        # clause is added, because the post-run audit compares it against the
        # round log byte for byte.
        outcome = self._judge(_spec(method='minima', optimize_cell=True),
                              status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertEqual(outcome['details'].count('[CELL]'), 1)
        self.assertNotIn('pressure', outcome['details'])
        self.assertNotIn('pressure_target_bar', outcome['cell_metrics'])

    def test_a_missing_progress_block_is_reported_as_unjudged(self):
        # No log is not a zero residual: the clause says what it could not read
        # instead of passing quietly or inventing a failure.
        outcome = self._judge_pressure(log='')
        self.assertIn('unjudged', outcome['details'])
        self.assertNotIn('FAILED', outcome['details'])


class TestTheLoadIsPairedWithTheBoxsOwnMotion(CellCheckCase):
    """The second half of the direction clause: a reference run, when one exists.

    A pressure can only be called the wrong way against the same box with no load
    on it, so the clause looks for that leg in the round - same optimizer, same
    declared coupling, same starting volume, `opt_pressure` absent. Round
    tmp_20261018_200317 supplies the reading this fixture imitates: its loaded leg
    drifted 15.730721 A^3 under 1 kbar where the unloaded twin drifted
    17.087668 A^3, which is a load doing what a load does.
    """

    def _lay(self, name, volumes, coupling, method='minima', pressure=0.0,
             theory=None, mode='yaml'):
        leg = os.path.join(self.root, 'examples', name, mode)
        os.makedirs(leg, exist_ok=True)
        from pyrai3md.utilities.opt_formatter import status_header, status_line
        rows = [status_header(cell=True)]
        for step, volume in enumerate(volumes, start=1):
            rows.append(status_line(step, (-8.0 - step, 0.0, 1.0e-3, 1.0e-4,
                                           1.0e-3, 1.0e-4, 2.4e-2, 3.4e-2,
                                           2.7e-3, 6.3e-3, volume)))
        with open(os.path.join(leg, 'opt_status.txt'), 'w') as handle:
            handle.write('\n'.join(rows) + '\n')
        import json
        config = {'optimize_cell': True, 'opt_method': method,
                  'opt_cell_coupling': coupling, 'opt_pressure': pressure}
        if theory is not None:
            config['theory'] = theory
        with open(os.path.join(leg, '%s.json' % name), 'w') as handle:
            json.dump(config, handle)
        return leg

    def _judge_loaded(self, loaded, reference_legs, theory=None):
        # The loaded leg lives where a round puts it, two levels under the
        # examples root, so the pairing scan has real siblings to walk.
        leg = self._lay(loaded[0], loaded[1], loaded[2], pressure=1000.0)
        for entry in reference_legs:
            # (name, volumes, coupling) plus, when the case needs it, the engine
            # list as recorded and the mode directory the leg was written into.
            self._lay(entry[0], entry[1], entry[2],
                      theory=entry[3] if len(entry) > 3 else None,
                      mode=entry[4] if len(entry) > 4 else 'yaml')
        spec = ('control:\n  title: cell_case\n  jobtype: optimize\n')
        if theory is not None:
            spec += '  theory: %s\n' % (theory,)
        spec += ('optimize:\n  method: minima\n  optimize_cell: true\n'
                 '  opt_pressure: 1000.0\n  opt_cell_coupling: anisotropic\n'
                 '  opt_csrms: 100.0\n  opt_csmax: 100.0\n')
        with open(self.spec_file, 'w') as handle:
            handle.write(spec)
        with open(os.path.join(leg, 'cell_case.log'), 'w') as handle:
            handle.write(_progress_block(1000.0, 9000.0, 9000.0)
                         + _progress_block(1000.0, 30.0, 40.0, True))
        outcome = {'status': 'pass', 'details': ''}
        check_cell_optimization(outcome, self.spec_file, leg, 'cell_case')
        return outcome

    def test_a_load_that_let_the_box_drift_more_than_unloaded_is_named(self):
        outcome = self._judge_loaded(('loaded', [1157.625, 1173.838],
                                      'anisotropic'),
                                     [('quiet', [1157.625, 1158.0],
                                       'anisotropic')])
        self.assertEqual(outcome['status'], 'fail', outcome['details'])
        self.assertIn("failed to oppose the box's own motion",
                      outcome['details'])
        metrics = outcome['cell_metrics']
        self.assertEqual(metrics['pressure_reference_example'], 'quiet')
        self.assertAlmostEqual(metrics['pressure_reference_drift'],
                               1158.0 - 1157.625, delta=1.0e-6)

    def test_a_load_smaller_than_the_boxes_own_drift_passes_by_reference(self):
        # The round's own numbers, in the fixture's shape: unloaded 17.088,
        # loaded 15.731 A^3.
        outcome = self._judge_loaded(('loaded', [1157.625, 1173.356],
                                     'anisotropic'),
                                    [('quiet', [1157.625, 1174.713],
                                      'anisotropic')])
        self.assertEqual(outcome['status'], 'pass', outcome['details'])
        self.assertIn('paired against quiet', outcome['details'])
        self.assertNotIn('oppose', outcome['details'].split('FAILED')[-1])

    def test_two_twins_are_refused_rather_than_picked(self):
        # Two zero-pressure legs of the same box, method and coupling: choosing
        # between them would make the verdict depend on glob order.
        outcome = self._judge_loaded(('loaded', [1157.625, 1173.356],
                                      'anisotropic'),
                                     [('quiet_a', [1157.625, 1174.713],
                                       'anisotropic'),
                                      ('quiet_b', [1157.625, 1180.0],
                                       'anisotropic')])
        self.assertEqual(outcome['status'], 'pass', outcome['details'])
        self.assertIn('choosing one is not a measurement', outcome['details'])

    def test_a_twin_of_another_coupling_is_not_a_reference(self):
        # The pairing key is the coupling the run declared: a shape-preserving
        # relaxation of the same box does not say what this box does freely.
        outcome = self._judge_loaded(('loaded', [1157.625, 1173.356],
                                      'anisotropic'),
                                     [('quiet', [1157.625, 1174.713],
                                       'isotropic')])
        self.assertEqual(outcome['status'], 'pass', outcome['details'])
        self.assertIn('paired against nothing', outcome['details'])

    def test_a_twin_on_another_engine_is_not_a_reference(self):
        # Two arms of a plan can share optimizer, coupling and starting volume
        # while one is scored by xtb and the other by cp2k. The unloaded drift is
        # a property of the scoring function, so cp2k's own motion is not what the
        # xtb box would do with no load on it - and the round carries both.
        outcome = self._judge_loaded(
            ('loaded', [1157.625, 1173.356], 'anisotropic'),
            [('quiet', [1157.625, 1174.713], 'anisotropic', ['cp2k'])],
            theory=['xtb'])
        self.assertIn('paired against nothing', outcome['details'])

    def test_a_twin_on_the_same_engine_is_still_a_reference(self):
        # The other half of that gate, and the reason the clause above is a
        # measurement rather than a vacuous rejection: one item of the fixture,
        # the engine list, moves the verdict from "no reference" to "paired
        # against quiet".
        outcome = self._judge_loaded(
            ('loaded', [1157.625, 1173.356], 'anisotropic'),
            [('quiet', [1157.625, 1174.713], 'anisotropic', ['xtb'])],
            theory=['xtb'])
        self.assertIn('paired against quiet', outcome['details'])
        self.assertEqual(outcome['status'], 'pass', outcome['details'])

    def test_a_layered_twin_pairs_by_its_whole_engine_list(self):
        # A gap run scores its two layers in order; a single-engine box whose
        # centre happens to be pyscf is a different objective, so the comparison
        # runs over the whole list and not its first item.
        outcome = self._judge_loaded(
            ('loaded', [1157.625, 1173.356], 'anisotropic'),
            [('quiet', [1157.625, 1174.713], 'anisotropic', ['pyscf'])],
            theory=['pyscf', 'lammps'])
        self.assertIn('paired against nothing', outcome['details'])

    def test_a_twin_recorded_as_a_string_pairs_with_a_list_declaration(self):
        # The input YAML spells `control.theory` a list, the round's flat config
        # sometimes carries one space-separated string (`manager.py:954-960`
        # accepts both when it resolves skills). Comparing shapes instead of
        # contents would refuse every real twin and report "no reference".
        outcome = self._judge_loaded(
            ('loaded', [1157.625, 1173.356], 'anisotropic'),
            [('quiet', [1157.625, 1174.713], 'anisotropic', 'xtb')],
            theory=['xtb'])
        self.assertIn('paired against quiet', outcome['details'])

    def test_the_two_modes_of_one_example_are_one_reference(self):
        # A plan lays an example in two mode directories, and their records are
        # the same measurement twice rather than a second opinion: 17.087668 A^3
        # against 17.087669 agrees far inside the 1e-6 this scan already spends on
        # the starting volume, so the pair has to name the twin instead of
        # throwing up its hands. Before the tolerance was declared the two drifts
        # differed in the ninth decimal of a rounded key and the verdict was
        # "choosing one is not a measurement".
        outcome = self._judge_loaded(
            ('loaded', [1157.625, 1173.356], 'anisotropic'),
            [('quiet', [1157.625, 1174.712668], 'anisotropic', ['xtb'], 'yaml'),
             ('quiet', [1157.625, 1174.712669], 'anisotropic', ['xtb'], 'slurm')],
            theory=['xtb'])
        self.assertIn('paired against quiet', outcome['details'])
        self.assertNotIn('choosing one is not a measurement',
                         outcome['details'])

    def test_two_modes_that_disagree_are_refused_as_two_references(self):
        # The same fixture with the tolerance crossed: one example name holding
        # two unstressed boxes 4.8 A^3 apart is the round carrying two references,
        # and the count says two rather than the one directory they share.
        outcome = self._judge_loaded(
            ('loaded', [1157.625, 1173.356], 'anisotropic'),
            [('quiet', [1157.625, 1174.713], 'anisotropic', ['xtb'], 'yaml'),
             ('quiet', [1157.625, 1179.500], 'anisotropic', ['xtb'], 'slurm')],
            theory=['xtb'])
        self.assertIn('2 zero-pressure twins of this box are on the plan',
                      outcome['details'])
        self.assertIn('choosing one is not a measurement', outcome['details'])


class TestTheDeclaredCouplingIsAnsweredByTheShape(CellCheckCase):
    """The third reading: which directions the box was allowed to move in.

    A volume moves under either coupling, so the cell columns and the drift both
    stay green when a run answers the isotropic direction after being asked for
    the anisotropic one, or the other way round. What separates the two is the
    SHAPE: the isotropic step is one vector built from the three row lengths
    (`optimizer/base.py::_cell_direction`, :757-780), so it scales them together
    and their ratios do not move at all, while an anisotropic step is never
    projected onto that vector and the ratios are free. The fixtures below are
    the two trajectories, each handed to both declarations, which is what makes
    this reading falsifiable rather than a restatement of the volume.

    The numbers are synthetic on purpose: no engine runs here. The printed cell
    carries six decimals, so `%.6f` is how the writer of the real trajectory
    writes it and the ratios a reader can resolve are the ones this fixture can
    resolve too.
    """

    CUBIC_START = (10.0, 9.0, 8.0)
    PROPORTIONATE = (11.0, 9.9, 8.8)      # every row scaled by 1.1
    STRETCHED = (11.0, 9.0, 8.0)          # only the first row moved

    def _trajectory(self, cells):
        """An `opt_geom.xyz` whose frames carry the given row lengths."""
        text = ''
        for row in cells:
            lattice = [(row[0], 0.0, 0.0), (0.0, row[1], 0.0),
                       (0.0, 0.0, row[2])]
            flat = [value for triple in lattice for value in triple]
            text += '1\nStep 1 Lattice="%s"\nH 0.0 0.0 0.0\n' % (
                ' '.join('%.6f' % value for value in flat))
        with open(os.path.join(self.root, 'opt_geom.xyz'), 'w') as handle:
            handle.write(text)

    def _judge_shape(self, cells, **optimize):
        self._trajectory(cells)
        spec_args = {'method': 'minima', 'optimize_cell': True}
        spec_args.update(optimize)
        return self._judge(_spec(**spec_args), status=self._cell_record())

    def test_scaling_every_row_together_agrees_with_isotropic_coupling(self):
        outcome = self._judge_shape([self.CUBIC_START, self.PROPORTIONATE],
                                    opt_cell_coupling='isotropic')
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertIn('[COUPLING] isotropic coupling agrees', outcome['details'])
        self.assertLess(outcome['cell_metrics']['shape_ratio_change'],
                        1.0e-9, 'a proportionate box must read as no shape '
                                'change, not as a small one')

    def test_a_box_stretched_along_one_row_contradicts_isotropic_coupling(self):
        # The mutation arm of the reading above: one row moves further than the
        # other two, which is what a step that was never confined to the
        # coupling direction leaves. If this passed, the clause would be
        # measuring nothing but the volume it already had.
        outcome = self._judge_shape([self.CUBIC_START, self.STRETCHED],
                                    opt_cell_coupling='isotropic')
        self.assertEqual(outcome['status'], 'fail', outcome.get('details'))
        self.assertIn('not confined to it', outcome['details'])

    def test_a_box_that_changed_its_shape_agrees_with_anisotropic_coupling(self):
        outcome = self._judge_shape([self.CUBIC_START, self.STRETCHED])
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertIn('[COUPLING] anisotropic coupling agrees',
                      outcome['details'])

    def test_a_proportionate_box_contradicts_the_anisotropic_default(self):
        # The key is not written here at all, so this is the other four cell
        # examples: they inherit `anisotropic` (schema/config.py:193) and are
        # judged on it. A box that kept its ratios to the last printed digit
        # under that declaration is the signature of the projection still being
        # applied, and the verdict says which declaration it read.
        outcome = self._judge_shape([self.CUBIC_START, self.PROPORTIONATE])
        self.assertEqual(outcome['status'], 'fail', outcome.get('details'))
        self.assertIn('signature of the isotropic direction still being applied',
                      outcome['details'])

    def test_one_frame_is_reported_as_unjudged_rather_than_as_a_still_box(self):
        self._trajectory([self.CUBIC_START])
        outcome = self._judge(_spec(method='minima', optimize_cell=True),
                              status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertIn('[COUPLING] unjudged: 1 frame(s)', outcome['details'])

    def test_a_run_with_no_trajectory_says_it_could_not_read_the_shape(self):
        # A killed run that wrote its status record and died before the first
        # frame is not a run that kept its shape: the clause names the missing
        # product instead of inventing a verdict.
        outcome = self._judge(_spec(method='minima', optimize_cell=True),
                              status=self._cell_record())
        self.assertEqual(outcome['status'], 'pass', outcome.get('details'))
        self.assertIn('[COUPLING] unjudged: 0 frame(s)', outcome['details'])
