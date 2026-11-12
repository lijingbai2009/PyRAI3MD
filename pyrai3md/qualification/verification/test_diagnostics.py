"""
======================================================================
This module tests the owner of the run-health policy: pyrai3md.workflow.
diagnostics. What it pins is the split the user ruled on -- a commit that
cannot be computed stops every run, while a bound that merely says "this
box has left its own start" is a judgement, and a judgement interrupts
only a run started as a test (control.debug).

Why each case is here:

- the judgement must not stop a production trajectory, and must stop a
  test trajectory with the same message it would have written;
- the arithmetic cases (a collapsed cell, a non-finite lattice or
  extended coordinate) must stop with the framework's own words. The
  07:13 round's anisotropic failures surfaced as 'Value error,
  thermostat_dof contains a non-finite value' -- the schema validator
  naming its victim, two fields and one traceback away from the drive
  that produced it;
- a monitor that speaks every step is a monitor nobody reads. The Nose
  Hoover chain of that same run reached inf within one step of the last
  readable frame, so the emission is bounded and closed by a summary;
- the findings have to survive serialization, because a dispatched child
  trajectory is the one that sees them;
- control.debug has to reach both input modes and outrank a skill's
  spelling of the same name, which is how one keyword drives every module
  under test.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import unittest

import numpy as np

from pyrai3md.workflow.records.diagnostics import (
    CELL_GATE, CELL_IMPOSSIBLE, CELL_NOTE, CONSTRAINT_GEOMETRY,
    ENSEMBLE_DOF_IMPOSSIBLE, IMAGE_INTEGRITY, KINETIC_IDENTITY,
    EnsembleDivergenceError, RecordedStepError, check_finite,
    check_recorded_step, is_debug, report, summary)
from pyrai3md.workflow.integrator import CellDivergenceError
from pyrai3md.workflow.integrator.operations import verify_recorded
from pyrai3md.workflow.integrator.verlet import _matrix_expm, _scale_cell

# Absolute import: `unittest discover` is run from inside this directory on
# the deployment, where a relative import has no parent package to resolve.
from pyrai3md.qualification.verification._ensemble_helpers import (
    NATOM, make_npt_state)


class _StubMaestro:
    """Duck-typed Maestro exposing only what the merge looks up."""

    def __init__(self, runtime=None):
        self._runtime = runtime or {}

    def get_config(self, skill_name):
        return self._runtime.get(skill_name)

    def get_skill_info(self, skill_name):
        return {'name': 'lammps'}


class _CollectingLogger:
    """Stand-in for RunLogger that keeps what the monitor said."""

    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(line)


def _creeping_state(debug=False, strain=None):
    """A variable-cell state whose box is already past the bound."""
    state = make_npt_state()
    state.pmd.control.debug = bool(debug)
    epsilon = 0.8 * np.eye(3) if strain is None else strain
    state.update_system({'barostat_strain': epsilon.tolist()})
    state.set_result({'gradient': np.zeros((NATOM, 3)).tolist(),
                      'stress_tensor': np.zeros((1, 3, 3)).tolist()})
    state.update_history()
    state.logger = _CollectingLogger()
    return state


def _commit(state):
    """One cell commit small enough that only the accumulated bound sees it."""
    return _scale_cell(state, _matrix_expm(np.diag([0.001, 0.0, 0.0])),
                       coordinates=np.asarray(state.coordinates),
                       velocities=np.asarray(state.velocities))


class TestJudgementIsATestFeature(unittest.TestCase):
    """The accumulated bound interrupts a test run and warns a real one."""

    def test_a_production_run_reports_the_creep_and_keeps_going(self):
        state = _creeping_state(debug=False)
        updates = _commit(state)           # must not raise
        self.assertIn('lattice_vector', updates)
        message = state.runtime.diagnostics[CELL_GATE]
        self.assertEqual(message['count'], 1)
        self.assertFalse(message['aborted'])
        self.assertEqual(len(state.logger.lines), 1)
        self.assertIn('cell diverged at step', state.logger.lines[0])
        self.assertIn('[DIAG cell_gate]', state.logger.lines[0])

    def test_a_debugged_run_is_interrupted_with_the_same_words(self):
        state = _creeping_state(debug=True)
        with self.assertRaises(CellDivergenceError) as caught:
            _commit(state)
        self.assertIn('strain xx is e^0.80 = 2.2 of the box the run '
                      'started with', str(caught.exception))
        entry = state.runtime.diagnostics[CELL_GATE]
        self.assertTrue(entry['aborted'])
        # The aborting text is the record, and the record is not also
        # written as a log line: a run that died says it once, in the
        # Status line the qualification already reads.
        self.assertEqual(state.logger.lines, [])

    def test_an_early_note_is_said_once_not_every_step(self):
        # 0.3 log strain is past the note bound and under the bound that
        # means anything to the run; a monitor repeating it 1700 times
        # would bury the one line that matters.
        # A pure shear of 0.3: its trace is zero, so no component crosses
        # the bound that interrupts, and the note band is what is left.
        shear = np.zeros((3, 3))
        shear[0, 1] = 0.3
        state = _creeping_state(debug=False, strain=shear)
        for _ in range(50):
            _commit(state)
        self.assertEqual(len(state.logger.lines), 1, state.logger.lines)
        self.assertIn('[DIAG cell_note]', state.logger.lines[0])
        self.assertEqual(state.runtime.diagnostics[CELL_NOTE]['count'], 50)


class TestArithmeticIsNotAFeature(unittest.TestCase):
    """What cannot be computed stops in every mode, and says what it was."""

    def test_a_collapsed_cell_stops_without_the_switch(self):
        # A flattening deformation (one cell direction gone) is not a
        # judgement about the ensemble: det(B) = 0 has no successor, so it
        # stops a production run too.
        state = _creeping_state(debug=False)
        with self.assertRaises(CellDivergenceError) as caught:
            _scale_cell(state, np.diag([1.0, 1.0, 0.0]),
                        coordinates=np.asarray(state.coordinates),
                        velocities=np.asarray(state.velocities))
        self.assertIn('the box collapsed', str(caught.exception))
        self.assertTrue(state.runtime.diagnostics[CELL_IMPOSSIBLE]['aborted'])

    def test_a_non_finite_coordinate_names_its_drive_not_its_validator(self):
        for field in ('barostat_momentum', 'barostat_velocity',
                      'barostat_strain'):
            state = _creeping_state(debug=False)
            poison = [[float('nan'), 0.0, 0.0], [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
            with self.assertRaises(EnsembleDivergenceError) as caught:
                check_finite(state, {field: poison})
            message = str(caught.exception)
            self.assertIn(field, message)
            self.assertIn('ensemble diverged at step', message)
            self.assertNotIn('Value error', message,
                             'the validator must not be the reader\'s first word')
            self.assertTrue(state.runtime.diagnostics[
                ENSEMBLE_DOF_IMPOSSIBLE]['aborted'])

    def test_a_clean_write_is_not_a_finding(self):
        state = _creeping_state(debug=False)
        check_finite(state, {'barostat_strain': np.zeros((3, 3)).tolist(),
                             'coordinates': np.zeros((NATOM, 3)).tolist()})
        self.assertEqual(state.runtime.diagnostics, {})


class TestTheMonitorIsBoundedAndPersistent(unittest.TestCase):
    """Emission frequency, the closing summary, and what survives a copy."""

    def test_the_monitor_speaks_once_per_recorded_frame_and_no_more(self):
        # The cadence is not a second knob to tune: `md.direct`/`md.buffer`
        # already decide which steps this run records at all, and md.ensemble
        # holds the numbers a monitor line would be read against. 20 steps of
        # a 1/100 schedule is therefore 20 lines, and a violation that
        # persists between two of them says when it began.
        state = _creeping_state(debug=False)
        state.pmd.dynamics.direct = 1
        state.pmd.dynamics.buffer = 100
        for step in range(1, 2001):
            state.runtime.current_step = step
            report(state, CELL_GATE, 'cell diverged', worst=0.8)
        lines = state.logger.lines
        self.assertEqual(len(lines), 20, len(lines))
        self.assertNotIn('since step', lines[0],
                         'the first time this is said, its start step is now')
        self.assertIn('since step 1', lines[1])
        self.assertEqual(state.runtime.diagnostics[CELL_GATE]['count'], 2000)
        self.assertEqual(state.runtime.diagnostics[CELL_GATE]['last_step'],
                         2000)
        self.assertAlmostEqual(state.runtime.diagnostics[CELL_GATE]['worst'],
                               0.8)

    def test_the_run_closes_with_one_line_per_check(self):
        state = _creeping_state(debug=False)
        # A record schedule of "never" is what makes this case exist: a
        # finding that never landed on a recorded step would otherwise leave
        # the run silent about its own box.
        state.pmd.dynamics.direct = 0
        state.pmd.dynamics.buffer = 10 ** 6
        report(state, CELL_GATE, 'cell diverged', worst=0.8)
        report(state, CELL_GATE, 'cell diverged', worst=0.9)
        summary(state)
        closing = [line for line in state.logger.lines
                   if 'finding(s)' in line]
        self.assertEqual(len(closing), 1, state.logger.lines)
        self.assertIn('2 finding(s)', closing[0])
        self.assertIn('worst 0.9000', closing[0])
        self.assertIn('cell diverged', '\n'.join(state.logger.lines))
        # A run that watched nothing says nothing.
        quiet = _creeping_state(debug=False)
        summary(quiet)
        self.assertEqual(quiet.logger.lines, [])

    def test_a_state_with_no_logger_still_records(self):
        # The harness states of test_integrator and test_md_output carry no
        # run log at all; the monitor must record without trying to speak.
        state = _creeping_state(debug=False)
        state.logger = None
        self.assertFalse(report(state, CELL_GATE, 'cell diverged', worst=0.8))
        self.assertEqual(state.runtime.diagnostics[CELL_GATE]['count'], 1)

    def test_the_record_survives_the_serializer_and_a_deep_copy(self):
        state = _creeping_state(debug=False)
        report(state, CELL_GATE, 'cell diverged', worst=0.8)
        round_trip = copy.deepcopy(state.runtime).to_dict()
        restored = type(state.runtime).from_dict(round_trip)
        self.assertIn(CELL_GATE, restored.diagnostics)
        self.assertEqual(restored.diagnostics[CELL_GATE]['count'], 1)
        # A copy must not share its entries: a child that increments would
        # otherwise rewrite the parent's counts.
        clone = state.runtime.copy()
        report(state, CELL_GATE, 'cell diverged', worst=0.9)
        self.assertEqual(clone.diagnostics[CELL_GATE]['count'], 1)
        self.assertEqual(state.runtime.diagnostics[CELL_GATE]['count'], 2)

    def test_is_debug_tolerates_a_state_with_no_control_section(self):
        class Bare:
            pass

        self.assertFalse(is_debug(Bare()))
        self.assertFalse(is_debug(object()))


class TestControlDebugReachesEveryLayer(unittest.TestCase):
    """One keyword, both input modes, and the priority over a skill's own."""

    def _example(self, name):
        import os

        import pyrai3md
        return os.path.join(os.path.dirname(os.path.abspath(pyrai3md.__file__
                                                             )),
                            'examples', name)

    def test_the_schema_default_is_off(self):
        from pyrai3md.schema.config import DEFAULT, PyRAIConfig

        self.assertIs(DEFAULT['debug'], False)
        self.assertIs(PyRAIConfig().control.debug, False)

    def test_the_classic_keyword_is_a_boolean(self):
        from pyrai3md.schema.adapter import CONTROL_TYPES
        from pyrai3md.utilities.read_tools import ReadVal

        self.assertEqual(CONTROL_TYPES['debug'], 'b')
        self.assertIs(ReadVal('b')(['true']), True)
        self.assertIs(ReadVal('b')(['false']), False)

    def test_a_yaml_example_reaches_the_integrator(self):
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema import build_config

        raw = parse_input(self._example('md_lammps_npt_nh_pr.yaml'))
        self.assertEqual(str(raw['control']['debug']).lower(), 'true')
        self.assertIs(build_config(raw).control.debug, True)

    def test_the_global_switch_forces_a_skill_that_never_mentioned_it(self):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        maestro = _StubMaestro()
        on = PyRAIConfig()
        on.control.debug = True
        merged = _merge_one_skill_config(
            maestro, 'lammps_skill', config=on,
            raw_config={'control': {'debug': True},
                        'lammps_skill': {'lammps_units': 'real'}})
        self.assertIs(merged['debug'], True)

    def test_a_skill_that_asked_for_debug_is_not_silenced_by_a_default(self):
        # The framework's field is named `debug` too, and its default is
        # False. If the name alone carried priority, every skill-side test
        # that asks for its artifacts would be turned off by a framework
        # that never mentioned the switch.
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        maestro = _StubMaestro()
        merged = _merge_one_skill_config(
            maestro, 'lammps_skill', config=PyRAIConfig(),
            raw_config={'lammps_skill': {'debug': True,
                                         'lammps_units': 'real'}})
        self.assertIs(merged['debug'], True)


    def test_the_classic_input_carries_the_switch(self):
        # The pair table is what both the classic text and the agent config
        # are built from. Without the entry, `pyrai3md case.input` silently
        # runs the framework default and the two qualification modes judge
        # different behaviours -- the gap _MD_ECHO_KEYS documents for &MD.
        from pyrai3md.qualification.certification.converter import (
            _build_control_pairs)

        def keys(config):
            return [key for key, _ in
                    _build_control_pairs(config, 'case')[1]]

        self.assertIn('debug', keys({'title': 'case', 'debug': True}))
        self.assertNotIn('debug', keys({'title': 'case'}),
                         'a default input stays minimal, like the driver')

    def test_the_classic_spelling_reads_back_as_a_bool(self):
        # The parser hands over strings for a classic section, so the
        # keyword table is what turns 'debug true' into a bool. A key
        # missing from that table would reach the schema as the string
        # 'true', which pydantic coerces -- and a table entry with the
        # wrong code would not, which is how a silent default gets shipped.
        from pyrai3md.schema import build_config

        config = build_config({'control': {'title': 'case', 'jobtype': 'md',
                                           'debug': 'true'}})
        self.assertIs(config.control.debug, True)
        config = build_config({'control': {'title': 'case', 'jobtype': 'md',
                                           'debug': 'false'}})
        self.assertIs(config.control.debug, False)


class TestWhatTheQualificationShows(unittest.TestCase):
    """The test side of the same ruling: aligned, unclipped, never silent.

    Details are printed under the counter, not at column 0, and a
    warn-only run has to leave a trace even though it passed. Both are
    print-side properties: the `details` string itself stays unindented
    data, because six assertions in test_npt_stability select on its
    substrings and a record that carries layout cannot be read by
    anything else.
    """

    @staticmethod
    def _captured(result, idx, total):
        import contextlib
        import io

        from pyrai3md.qualification import runner

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            runner._print_single_result(result, idx, total)
        return buffer.getvalue().rstrip('\n').split('\n')

    def test_every_detail_line_starts_under_the_counter(self):
        result = {'name': 'example_case_yaml', 'mode': 'pyrai3md_yaml',
                  'status': 'fail',
                  'details': ('tail line\n[NPT STABILITY] volume_drift 0.115 '
                              'over 0.100\n[PBC] pair cutoff 10.000 A')}
        lines = self._captured(result, 51, 107)
        counter = lines[0].index('[51/107]') if '[51/107]' in lines[0] else \
            lines[0].index('[051/107]')
        self.assertEqual(counter, 2, 'the counter must sit at the left edge')
        for line in lines[1:]:
            self.assertEqual(len(line) - len(line.lstrip()),
                             2 + len('[051/107]') + 1, line)
        self.assertIn('[PBC]', lines[-1])
        self.assertFalse(lines[-1].startswith('[PBC]'),
                         'no tag line may start at column 0')

    def test_the_indent_follows_the_counter_width(self):
        # A 38-task round prints exactly what the shipped logs look like;
        # a 107-task round is one column wider and nothing is re-edited.
        result = {'name': 'case', 'mode': 'm', 'status': 'fail',
                  'details': 'a\nb'}
        for total, width in ((38, 10), (107, 12), (9, 8)):
            lines = self._captured(result, 1, total)
            self.assertEqual(len(lines[1]) - len(lines[1].lstrip()), width,
                             'total=%d -> %r' % (total, lines[1]))

    def test_a_clipped_stream_starts_on_a_whole_line(self):
        from pyrai3md.qualification import runner

        text = ''.join('line %03d %s\n' % (i, 'x' * 20) for i in range(40))
        tail = runner._clip_to_lines(text, 50, tail=True)
        self.assertTrue(tail.startswith('line '), repr(tail[:12]))
        head = runner._clip_to_lines('hello\nworld', 200)
        self.assertEqual(head, 'hello\nworld')
        self.assertEqual(runner._clip_to_lines('', 200, tail=True), '')

    def test_a_verdict_keeps_the_engine_tail_it_used_to_wipe(self):
        # The exceeding verdicts used to ASSIGN details, discarding the
        # captured stream that explains the verdict.
        from pyrai3md.qualification import runner

        result = {'status': 'pass', 'details': 'engine said something'}
        runner._check_run_diagnostics(result, 'x.yaml', base_dir='.',
                                      title='missing')
        self.assertEqual(result['details'], 'engine said something')

    def test_the_monitor_lines_are_read_back_out_of_the_run_log(self):
        import os
        import tempfile

        from pyrai3md.qualification import runner

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'case.log'), 'w') as handle:
                handle.write('[DIAG cell_gate] cell diverged at step 402\n')
                handle.write('Ensemble: npt\n')
                handle.write('[DIAG cell_gate] 3 finding(s), first step 402, '
                             'last step 1899, worst 0.8000\n')
            # Closed: the harvest reads the file the run left behind.
            result = {'status': 'pass', 'details': 'ran'}
            runner._check_run_diagnostics(result, 'case.yaml', tmp, 'case')
        self.assertIn('[DIAG cell_gate] 3 finding(s)', result['details'])
        self.assertIn('ran', result['details'])
        self.assertEqual(result['status'], 'pass',
                         'a monitor report does not fail a run')

    def test_the_relaxed_thermal_bound_still_states_the_drift(self):
        # Relaxation must not mean silence: the two numbers this round
        # failed on now pass and still have to appear in the record.
        import os
        import tempfile

        from pyrai3md.qualification import runner

        self.assertEqual(runner.BOX_TOLERANCES['temperature_drift'], 3.0)
        self.assertGreater(1.754, 1.0, 'the number that used to fail')
        with tempfile.TemporaryDirectory() as tmp:
            xyz = os.path.join(tmp, 'case_example.md.xyz')
            with open(xyz, 'w') as handle:
                for step, temp in ((1, 303.0), (101, 500.0), (201, 826.2)):
                    handle.write('0\ncase coord %d state %d '
                                 'Lattice="8.4 0.0 0.0 0.0 8.4 0.0 '
                                 '0.0 0.0 8.4"\n' % (step, step))
            with open(os.path.join(tmp, 'case_example.md.ensemble'),
                      'w') as handle:
                handle.write('%20s%20s\n' % ('time', 'T_inst(K)'))
                for temp in (303.0, 500.0, 826.2):
                    handle.write('%20.2f%20.4f\n' % (0.0, temp))
            result = {'status': 'pass', 'details': 'ran'}
            spec = os.path.join(tmp, 'case.yaml')
            with open(spec, 'w') as handle:
                handle.write('control:\n  jobtype: md\n'
                             'md:\n  ensemble: npt\n'
                             '  temperature: 300\n')
            runner._check_npt_stability(result, spec, {}, tmp,
                                        'case_example')
        self.assertEqual(result['status'], 'pass', result.get('details'))
        self.assertIn('temperature_drift', result['details'])

    def test_a_passing_unit_task_prints_one_counted_line(self):
        # Both kinds of unit test (the framework files and a skill's runtime
        # pytest) end their passing block with the line that counts them.
        from pyrai3md.qualification import runner

        for mode in ('local', 'skill_runtime'):
            lines = self._captured(
                {'name': 'case', 'mode': mode, 'status': 'pass',
                 'details': '[UNIT] ran=16 failures=0 errors=0'}, 1, 107)
            self.assertEqual(len(lines), 2, mode)
            self.assertEqual(lines[1].strip(),
                             '[UNIT] ran=16 failures=0 errors=0', mode)

    def test_a_passing_run_prints_what_qualifies_the_verdict(self):
        # The engine tail a passing task keeps in `details` stays out of the
        # log (500 lines between two verdicts is not a report), but the note
        # saying the verdict was reached without the records it wanted does
        # not get to disappear just because the status reads PASS.
        from pyrai3md.qualification import runner

        details = ('convergence line\nanother engine line\n'
                   '[ENERGY CHECK] no case.md.energies under /run')
        lines = self._captured({'name': 'case', 'mode': 'pyrai3md_yaml',
                                'status': 'pass', 'details': details}, 1, 107)
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[1].strip().startswith('[ENERGY CHECK]'), lines[1])
        self.assertNotIn('convergence line', '\n'.join(lines))

    def test_a_failing_task_still_prints_the_captured_block(self):
        # The note filter is a pass-side rule only: a failure shows the whole
        # stream it captured, tagged lines and plain lines alike.
        from pyrai3md.qualification import runner

        details = 'engine said no\n[ENERGY CHECK] judged 0 frames'
        lines = self._captured({'name': 'case', 'mode': 'pyrai3md_yaml',
                                'status': 'fail', 'details': details}, 1, 107)
        self.assertEqual(len(lines), 3, lines)
        self.assertEqual(lines[1].strip(), 'engine said no')
        self.assertEqual(lines[2].strip(), '[ENERGY CHECK] judged 0 frames')

    def test_the_header_is_flushed_before_the_first_task_line(self):
        # The banner used to land after the first block of results: the run
        # writes stdout to a file (block-buffered) while the test framework
        # writes stderr (unbuffered) into the same file. The header therefore
        # has to go out on its own, before any task line exists to push it.
        import contextlib
        import io

        from pyrai3md.qualification import runner

        class _Recorder(io.TextIOBase):
            def __init__(self):
                self.events = []

            def write(self, text):
                self.events.append(('w', text))
                return len(text)

            def flush(self):
                self.events.append(('f', ''))
                return None

        stream = _Recorder()
        with contextlib.redirect_stdout(stream):
            runner._print_run_header({'timestamp': 'ts'}, 1)
            runner._print_single_result({'name': 'case', 'mode': 'local',
                                         'status': 'pass', 'details': ''},
                                        1, 1)
        first_task = next(i for i, (kind, text) in enumerate(stream.events)
                          if '[001/1]' in text or '[1/1]' in text)
        self.assertTrue(any(kind == 'f' for kind, _ in
                            stream.events[:first_task]),
                        'the header must reach the file before the first task')

    def _run_probe_task(self, body):
        """Run _run_main_test over a planted test file, fd-captured outside.

        The outer capture is what makes this a statement about descriptors:
        contextlib.redirect_stdout would only prove the Python-level stream,
        while the noise in question is written by a process that inherited
        file descriptor 1.
        """
        import os
        import tempfile
        import uuid

        from pyrai3md.qualification import runner

        with tempfile.TemporaryDirectory() as tmp:
            # A unique module stem: loader.discover imports the file by name,
            # and a second probe reusing the first probe's name would resolve
            # against the module already in sys.modules (whose directory the
            # earlier TemporaryDirectory has since removed).
            stem = 'test_probe_%s' % uuid.uuid4().hex[:8]
            path = os.path.join(tmp, stem + '.py')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(body)
            saved = os.dup(1)
            outer = tempfile.TemporaryFile(mode='w+b')
            try:
                import sys
                sys.stdout.flush()
                os.dup2(outer.fileno(), 1)
                try:
                    result = runner._run_main_test({'name': 'probe',
                                                   'file': path})
                finally:
                    sys.stdout.flush()
                    os.dup2(saved, 1)
            finally:
                os.close(saved)
                outer.seek(0)
                report = outer.read().decode('utf-8', 'replace')
                outer.close()
        return result, report

    _PROBE_HEADER = (
        'import os\n'
        'import subprocess\n'
        'import sys\n'
        'import unittest\n'
        '\n'
        'class TestProbe(unittest.TestCase):\n'
        '    def test_probe(self):\n')

    def test_a_task_owns_what_its_children_write(self):
        # Three kinds of noise a task can leave on the shared descriptor: a
        # bare print, an INFO line from the framework logger, and a child
        # process writing to the fd it inherited. A passing task shows none of
        # them: the report is this process's, and the child's own log is
        # where its record lives.
        body = self._PROBE_HEADER + (
            '        print("BARE-PRINT-NOISE")\n'
            '        from pyrai3md.kernel.logger import get_logger\n'
            '        get_logger("pyrai3md.probe").info("LOGGER-NOISE")\n'
            '        subprocess.run([sys.executable, "-c",\n'
            '                        "print(\'CHILD-NOISE\')"],\n'
            '                       stdout=None)\n'
            '        os.write(2, b\'CHILD-STDERR-NOISE\\n\')\n')
        result, report = self._run_probe_task(body)
        self.assertEqual(result['status'], 'pass', result.get('details'))
        for noise in ('BARE-PRINT-NOISE', 'LOGGER-NOISE', 'CHILD-NOISE',
                      'CHILD-STDERR-NOISE'):
            self.assertNotIn(noise, report, noise)
        self.assertIn('[UNIT] ran=1 failures=0 errors=0',
                      result['details'])

    def test_a_failing_task_keeps_what_its_children_wrote(self):
        # The same descriptor is the failure path's evidence: when the test
        # breaks, whatever its children left on it stays with the block that
        # explains it.
        body = self._PROBE_HEADER + (
            '        subprocess.run([sys.executable, "-c",\n'
            '                        "print(\'CHILD-DIAGNOSTIC\')"])\n'
            '        self.fail("probe failed")\n')
        result, _report = self._run_probe_task(body)
        self.assertEqual(result['status'], 'fail', result.get('details'))
        self.assertIn('CHILD-DIAGNOSTIC', result['details'])

    def test_a_module_with_no_case_is_not_a_pass(self):
        # unittest's empty suite is wasSuccessful() true; a pytest-style
        # module placed here used to certify itself with zero cases.
        body = 'def helper():\n    return 1\n'
        result, _report = self._run_probe_task(body)
        self.assertEqual(result['status'], 'error', result.get('details'))
        self.assertIn('0 cases collected', result['details'])

    def test_a_module_that_skips_every_case_is_a_skip(self):
        # A module-level guard (a missing optional dependency) collects
        # one ModuleSkipped case: wasSuccessful() is true, but nothing ran.
        body = ('import unittest\n'
                'raise unittest.SkipTest("missing dependency")\n')
        result, _report = self._run_probe_task(body)
        self.assertEqual(result['status'], 'skip', result.get('details'))
        self.assertIn('missing dependency', result['details'])


class TestTheRoundVerifierIsNotVacuous(unittest.TestCase):
    """`tools/verify.py` must catch the errors it claims to catch.

    The verifier re-derives a finished round's numbers from the files the run
    left behind. Its own selftest plants a wrong energy, a wrong cell metric, a
    stale record, a broken held condition and a composite that does not
    reproduce its single-layer counterpart, and expects each to be named; this
    gate runs that negative control so the tool cannot degrade into something
    that reports "clean" over a round it never actually checked.
    """

    @staticmethod
    def _load():
        import importlib.util
        import os
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
        path = os.path.join(repo_root, 'tools', 'verify.py')
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location('verify_tool', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_planted_errors_are_all_detected(self):
        verify = self._load()
        if verify is None:
            self.skipTest('tools/verify.py is only shipped with the source '
                          'tree')
        caught = verify.selftest()
        self.assertEqual(caught, [], 'verifier missed: %s' % caught)
        self.assertEqual(verify.main(['verify.py', '--selftest']), 0)

    def test_the_log_section_titles_come_from_the_writer(self):
        # The verifier reads the step log with the writer's own section
        # titles. A reader that retypes them drifts silently and reports
        # nothing to compare over a log full of values.
        import os
        verify = self._load()
        if verify is None:
            self.skipTest('tools/verify.py is only shipped with the source '
                          'tree')
        from pyrai3md.utilities import print_results
        path = os.path.join(os.path.dirname(print_results.__file__),
                            'print_results.py')
        with open(path, 'r', encoding='utf-8') as handler:
            source = handler.read()
        for name in ('SECTION_ENERGIES', 'SECTION_KINETIC'):
            title = getattr(verify, name)
            self.assertIn('%s = ' % name, source)
            self.assertIn(title, source)


def _recorded_frame(state, kinetic=None, pbc=None, image=None):
    """One frame shaped like the history write, from the state's own numbers.

    `K_pre` defaults to the kinetic energy the frame's velocities give, which
    is the reading every check below compares the recorded one against.
    """
    velocities = np.asarray(state.velocities, dtype=np.float64)
    masses = np.asarray(state.mass_au, dtype=np.float64)
    computed = 0.5 * float(np.sum(masses[:, None] * velocities ** 2))
    system = {'mass': list(state.pmd.system.mass),
              'coordinates': copy.deepcopy(state.pmd.system.coordinates)}
    if pbc is not None:
        system['pbc'] = list(pbc)
    if image is not None:
        system['image'] = image
    return {'step': int(getattr(state.runtime, 'current_step', 0) or 0) + 1,
            'system': system,
            'nuclear_sample': {'V_pre': velocities.tolist(),
                               'K_pre': computed if kinetic is None
                               else kinetic}}


class TestRecordedStepSentinels(unittest.TestCase):
    """What a frame can be asked about itself, and what that costs."""

    def _state(self, debug=False):
        state = make_npt_state()
        state.pmd.control.debug = bool(debug)
        state.logger = _CollectingLogger()
        state.runtime.current_step = 3
        return state

    def test_a_consistent_frame_says_nothing(self):
        state = self._state()
        check_recorded_step(state, _recorded_frame(state))
        entries = state.runtime.diagnostics
        self.assertNotIn(KINETIC_IDENTITY, entries)
        self.assertNotIn(IMAGE_INTEGRITY, entries)

    def test_two_kinetic_readings_of_one_step_are_a_finding(self):
        # The defect this guards is the one #310 closed: a step that sampled
        # the kinetic energy twice, on either side of an action that rewrote
        # the velocities, printed a number its own velocities could not
        # reproduce. 1e-3 Hartree is three orders above the roundoff the
        # identity carries, so the band it is judged in is not the issue;
        # that the check fires at all is.
        state = self._state()
        velocities = np.asarray(state.velocities, dtype=np.float64)
        masses = np.asarray(state.mass_au, dtype=np.float64)
        computed = 0.5 * float(np.sum(masses[:, None] * velocities ** 2))
        check_recorded_step(
            state, _recorded_frame(state, kinetic=computed + 1e-3))
        entry = state.runtime.diagnostics[KINETIC_IDENTITY]
        self.assertEqual(1, entry['count'])
        self.assertFalse(entry['aborted'],
                         'a warn-only run keeps integrating')
        self.assertTrue(any(KINETIC_IDENTITY in line
                            for line in state.logger.lines),
                        'the finding has to reach the log, not only the record')

    def test_a_debug_run_stops_on_the_same_finding(self):
        state = self._state(debug=True)
        velocities = np.asarray(state.velocities, dtype=np.float64)
        masses = np.asarray(state.mass_au, dtype=np.float64)
        computed = 0.5 * float(np.sum(masses[:, None] * velocities ** 2))
        with self.assertRaises(RecordedStepError) as caught:
            check_recorded_step(
                state, _recorded_frame(state, kinetic=computed + 1e-3))
        self.assertIn('recorded Ekin', str(caught.exception))

    def test_a_rounding_of_the_identity_is_not_a_finding(self):
        # The tolerance is declared on the scale of what is being compared
        # (Ekin ~ 1e-2 Hartree), so the last bit of one addition cannot
        # become a report the qualification has to explain.
        state = self._state()
        frame = _recorded_frame(state)
        frame['nuclear_sample']['K_pre'] *= (1.0 + 1e-15)
        check_recorded_step(state, frame)
        self.assertNotIn(KINETIC_IDENTITY, state.runtime.diagnostics)

    def test_fractional_crossing_counts_are_a_finding(self):
        state = self._state()
        natom = len(state.pmd.system.atoms)
        image = [[0, 0, 0]] * natom
        image[1] = [0, 0.5, 0]
        check_recorded_step(state, _recorded_frame(state, pbc=[1, 1, 1],
                                                  image=image))
        entry = state.runtime.diagnostics[IMAGE_INTEGRITY]
        self.assertEqual(1, entry['count'])
        self.assertAlmostEqual(0.5, entry['worst'])

    def test_an_image_of_the_wrong_shape_is_a_finding(self):
        state = self._state()
        image = [[0, 0, 0]] * (len(state.pmd.system.atoms) - 1)
        check_recorded_step(state, _recorded_frame(state, pbc=[1, 1, 1],
                                                  image=image))
        self.assertIn(IMAGE_INTEGRITY, state.runtime.diagnostics)

    def test_a_periodic_frame_without_counts_is_left_alone(self):
        # A configuration whose bond perception refused leaves no counts
        # behind (utilities/bonds returns empty with a warning). The
        # absence of a bookkeeping entry is not a violation of one.
        state = self._state()
        check_recorded_step(state, _recorded_frame(state, pbc=[1, 1, 1]))
        self.assertNotIn(IMAGE_INTEGRITY, state.runtime.diagnostics)


class TestConstraintGeometryOfARecordedFrame(unittest.TestCase):
    """The recorded geometry is asked the question the projection answered."""

    def _shaken_state(self):
        from pyrai3md.data.build import build_state
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema import normalize, validate
        from pyrai3md.schema.adapter import build_config
        config = {'control': {'title': 't_diagnostics', 'jobtype': 'md'},
                  'system': {'atoms': ['O', 'H', 'H'],
                             'coordinates': [[0.0, 0.0, 0.11734],
                                             [0.0, 0.75754, -0.46936],
                                             [0.0, -0.75754, -0.46936]]},
                  'molecule': {'ci': [1], 'spin': [0]},
                  'md': {'step': 1, 'size': 1.0},
                  'system_constraint': {
                      'unit_length': 'angstrom', 'unit_angle': 'degree',
                      'unit_energy': 'hartree',
                      'shake_bond': [[1, 2], [1, 3]]}}
        state = build_state(validate(normalize(build_config(config))))
        # The runtime record is attached by the execution layer, and the
        # findings have to live somewhere: without it there is nothing to
        # count and nothing to assert on.
        state.runtime = RuntimeState(total_steps=1)
        state.logger = _CollectingLogger()
        from pyrai3md.workflow.constraints import runtime as constraint_runtime
        constraint_runtime.bind(state)
        return state

    def _frame_of(self, state):
        system = copy.deepcopy(state.pmd.model_dump()['system'])
        return {'step': 1, 'system': system, 'nuclear_sample': {}}

    def test_a_projected_frame_is_satisfied_by_construction(self):
        state = self._shaken_state()
        from pyrai3md.workflow.integrator import apply_constrain
        apply_constrain(state)
        verify_recorded(state, self._frame_of(state))
        self.assertNotIn(CONSTRAINT_GEOMETRY, state.runtime.diagnostics)

    def test_a_frame_written_after_the_projection_moves_is_a_finding(self):
        # This is the case the projection itself cannot see: something wrote
        # the geometry after the step was projected, and the frame the
        # trajectory files, the restart and the next velocity kick all read
        # no longer satisfies what the input declared.
        state = self._shaken_state()
        from pyrai3md.workflow.integrator import apply_constrain
        apply_constrain(state)
        frame = self._frame_of(state)
        coordinates = frame['system']['coordinates']
        coordinates[1][1] += 0.05           # 0.05 Angstrom, far past 1e-8 Bohr
        verify_recorded(state, frame)
        entry = state.runtime.diagnostics[CONSTRAINT_GEOMETRY]
        self.assertEqual(1, entry['count'])
        self.assertGreater(entry['worst'], 1e-4)

    def test_a_run_without_constraints_is_not_asked_the_question(self):
        state = make_npt_state()
        state.logger = _CollectingLogger()
        frame = _recorded_frame(state)
        verify_recorded(state, frame)
        self.assertNotIn(CONSTRAINT_GEOMETRY, state.runtime.diagnostics)


if __name__ == '__main__':
    unittest.main()
