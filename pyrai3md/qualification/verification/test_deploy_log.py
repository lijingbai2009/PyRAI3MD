"""
======================================================================
Unit tests for the deploy_test.log download logic in tools/deploy.py.

Covers the ssh-cat primary channel and the scp fallback, whose jump
usage must mirror the actual SSH connection (ssh via jump -> scp -J
jump; ssh direct -> scp direct).

Aug 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


def _load_deploy_module():
    """Load tools/deploy.py by path (tools/ is not a package).

    Returns None outside the source tree, where tools/deploy.py is
    not shipped, so installed copies skip these tests instead of
    crashing at import.
    """
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    deploy_path = os.path.join(repo_root, 'tools', 'deploy.py')
    if not os.path.exists(deploy_path):
        return None
    spec = importlib.util.spec_from_file_location('deploy_tool',
                                                  deploy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy = _load_deploy_module()

_SKIP_NO_DEPLOY = unittest.skipIf(
    deploy is None,
    'tools/deploy.py is only shipped with the source tree')


class _FakeProc:
    """Minimal CompletedProcess stand-in."""

    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@_SKIP_NO_DEPLOY
class TestScpDeployLog(unittest.TestCase):
    """ssh cat primary channel with a jump-consistent scp fallback."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.local_log = os.path.join(self.tmpdir, 'test_run.log')
        self.ssh_base = ['ssh', '-o', 'ConnectTimeout=5', 'user@host']

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _call(self, jump_host='', ssh_base=None):
        deploy._scp_deploy_log(
            'user@host', 22, jump_host, '/remote/proj',
            tmp_dir='tmp_20260815_120000', local_log=self.local_log,
            ssh_base=ssh_base)

    def test_ssh_cat_success_writes_file_and_skips_scp(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeProc(returncode=0, stdout='remote log body\n')

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            self._call(ssh_base=self.ssh_base)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], self.ssh_base
                         + ['cat /remote/proj/deploy_test.log'])
        with open(self.local_log, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), 'remote log body\n')

    def test_ssh_cat_failure_falls_back_to_scp_with_jump(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == 'ssh':
                return _FakeProc(returncode=1,
                                 stderr='No such file or directory')
            return _FakeProc(returncode=0)

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            self._call(jump_host='bastion@jump', ssh_base=self.ssh_base)

        self.assertEqual(len(calls), 2)
        scp_cmd = calls[1]
        self.assertEqual(scp_cmd[0], 'scp')
        self.assertIn('-J', scp_cmd)
        self.assertEqual(scp_cmd[scp_cmd.index('-J') + 1],
                         'bastion@jump')
        self.assertIn('user@host:/remote/proj/deploy_test.log',
                      scp_cmd)

    def test_ssh_cat_failure_falls_back_to_direct_scp(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == 'ssh':
                return _FakeProc(returncode=1)
            return _FakeProc(returncode=0)

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            self._call(jump_host='', ssh_base=self.ssh_base)

        scp_cmd = calls[1]
        self.assertEqual(scp_cmd[0], 'scp')
        self.assertNotIn('-J', scp_cmd)

    def test_ssh_cat_timeout_falls_back_to_scp(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == 'ssh':
                raise subprocess.TimeoutExpired(cmd, 60)
            return _FakeProc(returncode=0)

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            self._call(jump_host='', ssh_base=self.ssh_base)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], 'scp')

    def test_no_ssh_base_uses_scp_directly(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeProc(returncode=0)

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            self._call(jump_host='bastion@jump', ssh_base=None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'scp')
        self.assertIn('-J', calls[0])


@_SKIP_NO_DEPLOY
class TestRemoteTriggerWorkDir(unittest.TestCase):
    """The trigger waits for this run's work directory and echoes it.

    Returning right after the nohup would race with the poller: the
    remote python still imports before create_work_dir runs, so the
    first discovery could lock onto a leftover directory carrying a
    terminal status.json. The trigger therefore blocks until a tmp
    directory newer than the trigger marker appears.
    """

    def setUp(self):
        self.ssh_base = ['ssh', '-o', 'ConnectTimeout=5', 'user@host']

    def test_trigger_command_waits_and_reports_work_dir(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured['cmd'] = cmd
            captured['timeout'] = kwargs.get('timeout')
            return _FakeProc(returncode=0,
                             stdout='tmp_20260817_090000\n')

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            success, work_dir = deploy._remote_trigger(
                self.ssh_base, '/remote/proj', 'python3', '')

        self.assertTrue(success)
        self.assertEqual(work_dir, '/remote/proj/tmp_20260817_090000')
        remote_cmd = captured['cmd'][-1]
        # Marker file fixes the trigger instant for 'find -newer'
        self.assertIn('touch .deploy_trigger_marker', remote_cmd)
        self.assertIn('-newer .deploy_trigger_marker', remote_cmd)
        self.assertIn('rm -f .deploy_trigger_marker', remote_cmd)
        # Only timestamped work directories qualify
        self.assertIn('tmp_[0-9]', remote_cmd)
        # The wait loop may hold the channel beyond the old 30s cap
        self.assertGreater(captured['timeout'], 30)

    def test_trigger_without_work_dir_falls_back_to_discovery(self):
        def fake_run(cmd, **kwargs):
            return _FakeProc(returncode=0, stdout='\n')

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            success, work_dir = deploy._remote_trigger(
                self.ssh_base, '/remote/proj', 'python3', '')

        self.assertTrue(success)
        self.assertEqual(work_dir, '')

    def test_trigger_failure_returns_empty_work_dir(self):
        def fake_run(cmd, **kwargs):
            return _FakeProc(returncode=1, stderr='boom')

        with mock.patch.object(deploy.subprocess, 'run',
                               side_effect=fake_run):
            success, work_dir = deploy._remote_trigger(
                self.ssh_base, '/remote/proj', 'python3', '')

        self.assertFalse(success)
        self.assertEqual(work_dir, '')

    def test_leftover_dir_names_never_match_the_echoed_line(self):
        # Skill calc dirs and arbitrary leftovers are filtered by the
        # same pattern the trigger parse relies on.
        for name in ('tmp_MOLCAS', 'tmp_orca', 'tmp_2026_0817',
                     'tmp_20260817_0900', 'tmp_20260817_0900001'):
            self.assertIsNone(deploy.TMP_DIR_PATTERN.match(name))
        self.assertIsNotNone(
            deploy.TMP_DIR_PATTERN.match('tmp_20260817_090000'))


@_SKIP_NO_DEPLOY
class TestKillRemotePattern(unittest.TestCase):
    """The pattern that stops the remote test must not match its carrier.

    A plain 'python.*pyrai3md test' matches the command line of the shell
    running the pkill, because that line contains 'python' and 'pyrai3md
    test' too. The kill then took out its own session and --force reported a
    failed stop -- with OpenSSH's post-quantum notice pasted in as the
    reason -- for a test that had in fact been stopped.
    """

    def test_the_kill_command_does_not_match_itself(self):
        import re

        carrier = ('bash -c pkill -f \'%s\' 2>/dev/null; echo "PKILL_RC=$?"'
                   % deploy.REMOTE_TEST_PATTERN)
        self.assertIsNone(re.search(deploy.REMOTE_TEST_PATTERN, carrier),
                          carrier)

    def test_a_running_test_is_matched(self):
        import re

        for line in ('python3 -m pyrai3md test',
                     '/opt/conda/envs/myenv/bin/'
                     'python3 /home/user/.local/bin/pyrai3md test'):
            self.assertIsNotNone(re.search(deploy.REMOTE_TEST_PATTERN, line),
                                 line)

    def test_the_liveness_probe_shares_the_pattern(self):
        """The stop fix is worthless if the poller still self-matches.

        The same bracketed pattern has to drive the liveness check: an
        unbracketed one matches the shell running the pgrep, so a run that
        already died keeps reporting itself alive and the poller waits out its
        idle timeout on a process that is not there.
        """
        seen = {}

        def fake_run(cmd, **kwargs):
            seen['cmd'] = cmd
            return _FakeProc(returncode=0, stdout='dead\n')

        original = deploy.subprocess.run
        deploy.subprocess.run = fake_run
        try:
            deploy._check_remote_process_alive(['ssh', 'user@host'])
        finally:
            deploy.subprocess.run = original
        command = ' '.join(seen['cmd'])
        self.assertIn(deploy.REMOTE_TEST_PATTERN, command, command)
        self.assertNotIn("pgrep -f 'python.*pyrai3md test'", command, command)


@_SKIP_NO_DEPLOY
class TestRemoteInstallCleansBuild(unittest.TestCase):
    """The install must not inherit the server's stale build tree.

    build/ is excluded from the sync, and --delete does not remove an
    excluded path, so the server keeps an old build/lib forever; build_py
    then ships files that no longer exist in the source, and the remote plan
    stops matching the tree it was given.
    """

    def test_build_directory_is_removed_before_installing(self):
        seen = {}

        def fake_run_remote(ssh_base, command, timeout=120, check=False):
            seen['command'] = command
            return True, ''

        original = deploy._run_remote
        deploy._run_remote = fake_run_remote
        try:
            deploy._remote_install(['ssh', 'user@host'], '/srv/pmd3',
                                   'python3', 'myenv')
        finally:
            deploy._run_remote = original

        command = seen['command']
        self.assertIn('rm -rf build', command, command)
        self.assertLess(command.index('rm -rf build'),
                        command.index('pip install'),
                        'the clean has to happen before the build')
        self.assertTrue(command.startswith('conda activate myenv'),
                        command)


@_SKIP_NO_DEPLOY
class TestKillRemoteReporting(unittest.TestCase):
    """What --force says must distinguish stopped / nothing / unreachable.

    The operator decides whether to trust the stop from this line alone, and
    the version that shipped conflated all three: it printed a bare
    'Could not terminate:' plus whatever stderr held (an OpenSSH notice) for
    a test that had in fact been stopped.
    """

    def _run_kill(self, remote_output, success=True):
        lines = []

        def fake_run_remote(ssh_base, command, timeout=120, check=False):
            return success, remote_output

        original = deploy._run_remote
        deploy._run_remote = fake_run_remote
        import contextlib
        import io
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                stopped = deploy._kill_remote_test(['ssh', 'user@host'])
        finally:
            deploy._run_remote = original
        return stopped, buffer.getvalue()

    def test_a_stopped_process_is_reported_as_stopped(self):
        stopped, out = self._run_kill('PKILL_RC=0\n')
        self.assertTrue(stopped, out)
        self.assertIn('terminated', out)

    def test_nothing_running_is_not_reported_as_a_failure(self):
        stopped, out = self._run_kill('PKILL_RC=1\n')
        self.assertFalse(stopped, out)
        self.assertIn('No remote test was running', out)
        self.assertNotIn('Could not', out)

    def test_an_unreachable_host_says_so(self):
        _stopped, out = self._run_kill('', success=False)
        self.assertIn('Could not reach the remote host', out)


@_SKIP_NO_DEPLOY
class TestSummaryCountsEveryCategory(unittest.TestCase):
    """The local summary must describe the same run the remote report does.

    `runner.run_tests` aggregates four result categories; the deploy tool
    summed only three, so a 593-task round announced 568 and a failing
    benchmark appeared in neither the totals nor the failure listing. The
    two sides are joined only by this dict, so the count is pinned here
    against every key the runner writes.
    """

    def _summary(self, results):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            deploy._print_summary(results)
        return buffer.getvalue()

    def test_benchmark_results_reach_the_total(self):
        def entries(prefix, count, status='pass'):
            return [{'name': '%s_%d' % (prefix, i), 'status': status}
                    for i in range(count)]

        failing = {'name': 'bench_4', 'status': 'fail'}
        # The failure listing only quotes an entry that carries a captured
        # channel, which is what a benchmark leg writes when it misses.
        failing['stderr'] = 'reference population mismatch\n'
        results = {
            'status': 'fail',
            'main_results': entries('main', 2),
            'example_results': entries('example', 3),
            'skill_results': entries('skill', 4),
            'benchmark_results': entries('bench', 4) + [failing],
        }
        out = self._summary(results)
        self.assertIn('Total rows: 14', out, out)
        self.assertIn('Fail: 1', out, out)
        self.assertIn('FAIL: bench_4', out, out)

    def test_every_key_the_runner_aggregates_is_aggregated_here(self):
        runner_keys = {'main_results', 'example_results', 'skill_results',
                       'benchmark_results'}
        out = self._summary(dict((k, []) for k in runner_keys))
        self.assertIn('Total rows: 0', out, out)
        # Nothing outside those four categories may carry weight either.
        out = self._summary(dict(list((k, []) for k in runner_keys)
                                 + [('other_results',
                                     [{'name': 'x', 'status': 'pass'}])]))
        self.assertIn('Total rows: 0', out, out)


@_SKIP_NO_DEPLOY
class TestFoldedRowsAreCountedOnce(unittest.TestCase):
    """A folded row must be read as the tasks it stands for, on both sides.

    runner.run_tests now writes one row per bucket for everything its switches
    closed, carrying the number in 'skip_count'. Two places sit on the far side
    of that dict: the summary this tool prints, and the expected-task count it
    polls against. Either one reading rows instead of counts turns a 123-task
    round into an announced one-row skip, and a denominator that counted the
    390 planned entries would have the poller wait on 267 rows that never
    print. The summary names both units so the two counts cannot be read as
    one.
    """

    def _summary(self, results):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            deploy._print_summary(results)
        return buffer.getvalue()

    def _folded(self):
        return {'name': 'example_closed', 'status': 'skip', 'skip_count': 2,
                'skip_reason': 'closed by 2 switch(es)',
                'closed_reasons': {'disabled by examples.energy': 1,
                                   'disabled by examples.gradient': 1},
                'details': 'disabled by examples.energy x1, '
                           'disabled by examples.gradient x1'}

    def test_a_folded_row_prints_one_block_worth_its_tasks(self):
        results = {'status': 'pass',
                   'main_results': [], 'skill_results': [],
                   'benchmark_results': [],
                   'example_results': [
                       {'name': 'example_optimize_minima_yaml',
                        'status': 'pass'},
                       self._folded()]}
        out = self._summary(results)
        self.assertIn('Skip rows: 1 (2 skipped tasks)', out, out)
        self.assertEqual(out.count('SKIP:'), 1, out)
        self.assertIn('SKIP: example_closed', out, out)
        self.assertIn('disabled by examples.energy x1', out, out)


@_SKIP_NO_DEPLOY
class TestRemoteLogDenominator(unittest.TestCase):
    """The round's denominator comes from its own log, not a second plan.

    The tool no longer asks the host to plan before triggering: that ran the
    full planner twice (minutes on a full round). The header line
    'Total tests: N' is the denominator until the first row, and a row's
    '[i/N]' outranks it once seen.
    """

    def test_the_header_total_is_read_before_the_first_row(self):
        text = ('  Planning the test round...\n'
                '  Plan ready: 1119 planned tasks\n'
                '  Total tests: 1114\n')
        self.assertEqual(deploy._parse_log_progress(text), (0, 1114))

    def test_a_row_marker_outranks_the_header(self):
        text = '  Total tests: 1114\n  [12/1114]  + a -> PASS\n'
        self.assertEqual(deploy._parse_log_progress(text), (12, 1114))

    def test_the_log_total_outranks_the_plan_once_seen(self):
        self.assertEqual(deploy._next_denominator(593, 0), 593)
        self.assertEqual(deploy._next_denominator(593, 611), 611)
        self.assertEqual(deploy._next_denominator(593, 593), 593)



class _ScriptedClock:
    """A stand-in for the `time` module in which only sleeping moves the clock.

    The poll loop's two windows are measured in seconds, and a test that let the
    real clock run would have to wait them out; here a round advances the clock
    by exactly the interval the loop asked to sleep for, so a budget is checked
    in rounds rather than in wall time.
    """

    def __init__(self, start=1000.0):
        self.now = start

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def strftime(self, fmt):
        return '[0000-00-00 00:00:00]'


@_SKIP_NO_DEPLOY
class TestThePollSeparatesAnOutageFromASilence(unittest.TestCase):
    """`_poll_outcome`, the four readings, and the route an outage ends in.

    The poller used to have one clock. A round that could not be reached - ssh
    returncode 255, a dropped jump-host session, a timeout - fell into it beside
    a round that answered and said nothing new, and both aged the idle window.
    Round `tmp_20261018_173454` paid for that: the link went at about 19:2x, the
    poller declared the round lost at 19:37, and the remote stayed alive at 105
    of 243 rows with zero failures for another two hours.
    """

    def test_terminal_status_beats_every_other_reading(self):
        # A leg that finished with failures is FINISHED, not quiet: fetching it
        # is the whole point of the poll.
        self.assertEqual(
            deploy._poll_outcome(0, False, 7, 7, False, 'fail'),
            deploy.POLL_FINISHED)

    def test_a_failed_probe_is_not_evidence_about_the_run(self):
        # The same silence reached two ways: a call that came back with 255, a
        # call that never completed, and a call that answered. Only the last
        # says anything about the round.
        self.assertEqual(
            deploy._poll_outcome(255, True, 7, 7, False),
            deploy.POLL_TRANSPORT_FAILURE)
        self.assertEqual(deploy._poll_outcome(None, True, 7, 7, False),
                         deploy.POLL_TRANSPORT_FAILURE)
        self.assertEqual(deploy._poll_outcome(0, True, 7, 7, False),
                         deploy.POLL_QUIET)

    def test_a_new_completion_is_progress_and_a_frozen_log_is_quiet(self):
        self.assertEqual(
            deploy._poll_outcome(0, False, 8, 7, False, 'running'),
            deploy.POLL_PROGRESS)
        self.assertEqual(
            deploy._poll_outcome(0, False, 7, 7, False, 'running'),
            deploy.POLL_QUIET)

    def test_a_log_that_writes_something_else_is_still_moving(self):
        # The counter only ticks when a TASK ends. A long leg - thirteen engine
        # calls for one cp2k strain point - writes step lines that change no
        # counter, and reading those as silence is the other half of the
        # mistake this function exists to stop.
        self.assertEqual(
            deploy._poll_outcome(0, False, 7, 7, True),
            deploy.POLL_PROGRESS)

    def test_the_manual_route_carries_the_transport_it_was_given(self):
        # What the operator is handed when an outage outlasts the budget: the
        # round's own two artifacts, over the transport the poll was using - so
        # a jump-host deployment does not get a direct command to correct by
        # hand.
        direct = deploy._manual_collection_route('user@host', 22, '',
                                                 '/proj', '/proj/tmp_x')
        jump = deploy._manual_collection_route('user@host', 2200,
                                               'user@jump', '/proj',
                                               '/proj/tmp_x')
        self.assertEqual(len(direct), 2)
        self.assertIn('/proj/tmp_x/results.json', direct[0])
        self.assertIn('/proj/deploy_test.log', direct[1])
        self.assertIn('-p 2200', jump[0])
        self.assertIn('-J user@jump', jump[0])
        self.assertNotIn('-J', ' '.join(direct))


@_SKIP_NO_DEPLOY
class TestThePollKeepsWaitingThroughAnOutage(unittest.TestCase):
    """The two clocks, driven through the real poll loop with no connection.

    The classifier alone could not be the test: what round 16 lost was the
    policy of which readings age the idle window. These cases run
    `_collect_results` against a scripted `subprocess.run` and a clock that only
    moves when the loop sleeps, so a budget is measured in rounds.
    """

    INTERVAL = 50
    IDLE = 100

    def setUp(self):
        self.ssh_base = ['ssh', 'user@host']

    def _drive(self, rounds, jump_host=''):
        """Poll until the loop returns; each round answers `rounds[i]`.

        An entry names what the three probes say: `status` (or none, meaning
        `status.json` is not there yet), `log` as `(returncode, text)`, and
        `alive` as True, False, or None for a liveness probe that did not
        answer. The last entry repeats if the loop outlives the script.
        """
        state = {'index': 0, 'round': {}}
        printed = []

        def fake_run(cmd, **kwargs):
            command = ' '.join(cmd)
            if 'status.json' in command:
                state['index'] += 1
                state['round'] = rounds[min(state['index'] - 1,
                                            len(rounds) - 1)]
                status = state['round'].get('status')
                if status is None:
                    return _FakeProc(returncode=0, stdout='')
                return _FakeProc(returncode=0,
                                 stdout='{"status": "%s"}' % status)
            if 'tail -n 200' in command:
                code, text = state['round'].get('log', (0, ''))
                return _FakeProc(returncode=code, stdout=text)
            if 'pgrep' in command:
                alive = state['round'].get('alive', True)
                if alive is None:
                    return _FakeProc(returncode=255, stdout='')
                return _FakeProc(returncode=0,
                                 stdout='alive\n' if alive else 'dead\n')
            raise AssertionError('unexpected command: %s' % command)

        clock = _ScriptedClock()
        with mock.patch.object(deploy.subprocess, 'run', side_effect=fake_run), \
                mock.patch.object(deploy, 'time', clock), \
                mock.patch.object(deploy, '_count_remote_results',
                                  return_value=7), \
                mock.patch.object(deploy, '_fetch_remote_results',
                                  return_value={'status': 'pass'}), \
                mock.patch.object(deploy, '_scp_deploy_log'), \
                mock.patch('builtins.print') as reporter:
            reporter.side_effect = lambda *args, **kwargs: printed.append(
                ' '.join(str(value) for value in args))
            result = deploy._collect_results(
                self.ssh_base, '/proj', 10, interval=self.INTERVAL,
                idle_timeout=self.IDLE, port=22, jump_host=jump_host,
                resolved_dir='/proj/tmp_x')
        return result, printed, clock

    def test_an_outage_longer_than_the_idle_window_does_not_end_the_poll(self):
        # Four unreachable rounds is 200 s against a 100 s window - the shape of
        # the round-16 loss - and the poll has to still be asking when the
        # remote writes its terminal status.
        outage = {'status': None, 'log': (255, ''), 'alive': None}
        result, printed, clock = self._drive(
            [dict(outage) for _ in range(4)] + [{'status': 'pass'}])
        self.assertEqual(result, {'status': 'pass'}, '\n'.join(printed))
        joined = '\n'.join(printed)
        self.assertIn('not counting toward the idle window', joined)
        self.assertIn('attempt 4', joined)

    def test_an_outage_that_outlasts_its_budget_hands_over_the_collection(self):
        # The ceiling is what keeps "do not age the window" from meaning "wait
        # forever": past it the tool says so and prints the route, jump host
        # included, rather than polling into the night.
        outage = {'status': None, 'log': (255, ''), 'alive': None}
        result, printed, clock = self._drive(
            [dict(outage) for _ in range(12)] + [{'status': 'pass'}],
            jump_host='user@jump')
        self.assertIsNone(result, '\n'.join(printed))
        joined = '\n'.join(printed)
        self.assertIn('rsync -az', joined)
        self.assertIn('-J user@jump', joined)
        self.assertIn('results.json', joined)

    def test_a_log_that_keeps_writing_keeps_the_poll_alive(self):
        # Five rounds of movement with the TASK counter frozen at 3/10: the
        # cp2k-strain shape. Under a classifier that only reads counters these
        # 250 s age a 100 s window and the collection is abandoned mid-leg.
        steps = [{'status': None,
                  'log': (0, '[3/10] + demo [yaml] -> PASS\nstep %d of 13\n'
                          % index),
                  'alive': False} for index in range(5)]
        result, printed, clock = self._drive(steps + [{'status': 'pass'}])
        self.assertEqual(result, {'status': 'pass'}, '\n'.join(printed))

    def test_a_reachable_remote_that_says_nothing_still_times_out(self):
        # The other half, and the witness that the case above is not just the
        # poller being lenient: answered probes, one frozen log line, no process.
        quiet = {'status': None, 'log': (0, '[3/10] + demo [yaml] -> PASS\n'),
                 'alive': False}
        result, printed, clock = self._drive([dict(quiet) for _ in range(8)])
        self.assertIsNone(result)
        joined = '\n'.join(printed)
        self.assertIn('no progress from remote for', joined)
        self.assertNotIn('not counting toward the idle window', joined)


if __name__ == '__main__':
    unittest.main()
