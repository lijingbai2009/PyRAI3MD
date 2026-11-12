"""
Unit tests for the resident service launcher.

The launcher is the only place in the skill that spawns a process, so its
tests are about the command it builds, the socket it claims and the
conditions under which it declares a service ready. Readiness is the
subtle part: a socket file on disk is not a service, and these tests pin
that down rather than assuming it.

Nothing here needs a LAMMPS installation: the launcher answers for the
process it starts, and what the service can load is the service's own
question to answer, before it binds.

Run: pytest lammps_skill/tests/test_persistent_launcher.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile

import pytest

from ..service.persistent import launcher
from ..service.persistent.protocol import LammpsPersistentUnsupported

# The stub carries the folded names the templates read (schema/config.py
# Environment._resolve_resources produces them from the two spellings).
PROJECT_CONFIG = {'shell': '', 'python_path': sys.executable,
                  'nproc': 1, 'thread': 1}


@pytest.fixture
def short_dir():
    """A directory whose path fits the AF_UNIX limit.

    pytest's tmp_path is far too long for a socket path on macOS (104
    bytes), which is the same constraint service/persistent/launcher.py
    works around for long scratch paths.
    """
    root = '/tmp' if os.path.isdir('/tmp') else tempfile.gettempdir()
    path = tempfile.mkdtemp(prefix='lmpunit-', dir=root)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class _Config:
    """Attribute stand-in for the skill config the launcher reads."""

    def __init__(self, **overrides):
        values = dict(PROJECT_CONFIG)
        values.update(overrides)
        for key, value in values.items():
            setattr(self, key, value)


class TestServiceArgv:
    """One rank runs the interpreter; more ranks need a launcher."""

    def test_single_rank_has_no_launcher(self):
        argv = launcher.build_service_argv('/env/bin/python', '/tmp/e',
                                           '/tmp/s', 4242, 1)
        assert argv[0] == '/env/bin/python'
        # Launched as a file so the service interpreter needs no
        # installation of the framework.
        assert argv[1] == launcher.SERVER_FILE
        assert argv[1].endswith('persistent/server.py')
        assert '-m' not in argv
        assert 'mpirun' not in argv

    def test_parallel_world_is_launched_with_mpi(self):
        argv = launcher.build_service_argv('/env/bin/python', '/tmp/e',
                                           '/tmp/s', 4242, 4)
        assert argv[:3] == ['mpirun', '-np', '4']
        assert argv[3] == '/env/bin/python'
        assert argv[4] == launcher.SERVER_FILE

    def test_arguments_carry_the_ownership_contract(self):
        argv = launcher.build_service_argv('/env/bin/python', '/tmp/e',
                                           '/tmp/s', 4242, 2)
        joined = ' '.join(argv)
        for flag in ('--endpoint /tmp/e', '--session /tmp/s',
                     '--owner-pid 4242', '--nproc 2'):
            assert flag in joined
        assert os.path.isfile(launcher.SERVER_FILE)

    def test_zero_ranks_is_refused_not_launched(self):
        with pytest.raises(LammpsPersistentUnsupported, match='lammps_nproc'):
            launcher.build_service_argv('/env/bin/python', '/tmp/e',
                                        '/tmp/s', 4242, 0)


class TestServiceScript:
    """The script wraps the same environment lmp gets, then execs."""

    def test_shell_prologue_runs_before_the_service(self):
        config = _Config(shell='module load gcc\\nmodule load openmpi')
        script = launcher.build_service_script(
            config, launcher.build_service_argv(sys.executable, '/tmp/e',
                                               '/tmp/s', 1, 2))
        lines = [line.strip() for line in script.splitlines() if line.strip()]
        assert lines[0] == '#!/bin/bash'
        assert any('module load gcc' in line for line in lines)
        assert any('module load openmpi' in line for line in lines)
        exec_index = next(n for n, line in enumerate(lines)
                          if line.startswith('exec '))
        module_index = next(n for n, line in enumerate(lines)
                            if 'module load' in line)
        assert module_index < exec_index

    def test_thread_per_rank_matches_the_subprocess_convention(self):
        """Both arms export the thread count; the service is not pinned.

        #303 read the resident arm's literal 1 as an engine constraint and
        documented it as an exception to the run-wide thread floor. It was
        only the keyword default: the subprocess arm exports the same
        field, so the two have to move together (#304).
        """
        from ..service.templates.run_templates import _build_lammps_commands

        for thread in (1, 4):
            config = _Config(thread=thread, lammps_project='P',
                             profile='nve', lammps_path='/opt/lammps')
            service = launcher.build_service_script(config, ['a', 'b'])
            subprocess_block = _build_lammps_commands(config)
            assert ('export OMP_NUM_THREADS=%d' % thread) in service
            assert ('export OMP_NUM_THREADS=%d' % thread) in subprocess_block

    def test_slurm_allocates_ranks_times_threads(self):
        # One task's cores are its MPI ranks times the threads per rank;
        # with the default this is the number it always was.
        from ..service.templates.run_templates import build_slurm_script

        config = _Config(thread=4, nproc=2, slurm_partition='debug',
                         slurm_time='10', slurm_ntasks=1, slurm_memory='1GB',
                         lammps_project='P', profile='nve', lammps_path='/x')
        script = build_slurm_script(config, '')
        assert '#SBATCH --cpus-per-task=8' in script

        config = _Config(thread=1, nproc=2,
                         slurm_partition='debug', slurm_time='10',
                         slurm_ntasks=1, slurm_memory='1GB',
                         lammps_project='P', profile='nve', lammps_path='/x')
        assert '#SBATCH --cpus-per-task=2' in build_slurm_script(config, '')

    def test_exec_keeps_the_group_leader_killable(self):
        # Without exec, killpg() would signal bash and leave mpirun and
        # every rank behind as orphans.
        script = launcher.build_service_script(_Config(), ['mpirun', 'x'])
        assert 'exec mpirun x' in script

    def test_no_lmp_binary_is_referenced(self):
        config = _Config(lammps_path='/opt/lammps')
        script = launcher.build_service_script(
            config, launcher.build_service_argv(sys.executable, '/tmp/e',
                                                '/tmp/s', 1, 1))
        assert 'lmp' not in script.replace('ramp', '')
        assert 'LAMMPS_INPUT' not in script

    def test_quoting_survives_spaces(self):
        script = launcher.build_service_script(
            _Config(), ['/bin/python with space', '--endpoint',
                        '/tmp/a path/svc.sock'])
        assert "'/bin/python with space'" in script
        assert "'/tmp/a path/svc.sock'" in script


class TestServicePaths:
    """Every resident artifact uses the cold stem, in the cold place."""

    def test_script_and_log_use_the_cold_basename(self):
        paths = launcher.service_paths('/scratch/tmp_LAMMPS', 'tmp_LAMMPS',
                                       'h2o_force')
        assert paths.work_dir == '/scratch/tmp_LAMMPS'
        assert paths.script == '/scratch/tmp_LAMMPS/h2o_force.sh'
        assert paths.log == '/scratch/tmp_LAMMPS/h2o_force.log'

    def test_everything_lives_in_the_calculation_directory(self):
        # One directory, one kind of file list: a person who found the
        # inputs has already found the socket and the owner record, and a
        # keep_tmp that clears the run clears them with it.
        paths = launcher.service_paths('/scratch/tmp_LAMMPS', 'tmp_LAMMPS',
                                       'h2o_force')
        assert os.path.dirname(paths.endpoint) == '/scratch/tmp_LAMMPS'
        assert os.path.dirname(paths.owner_record) == '/scratch/tmp_LAMMPS'
        assert paths.owner_record == '/scratch/tmp_LAMMPS/owner.json'

    def test_the_process_files_are_named_for_what_they_are(self):
        # Not the run's stem: the directory already says which run, and a
        # thirty-one character name spends the AF_UNIX budget to repeat it.
        paths = launcher.service_paths('/scratch/x', 'x', 'h2o_force')
        assert os.path.basename(paths.endpoint) == launcher.SOCKET_NAME
        assert launcher.SOCKET_NAME == 'svc.sock'

    def test_the_endpoint_is_the_namespace_not_a_fresh_name(self):
        # Rebuilding a backend in one namespace reuses the socket name, so a
        # live service there is refused rather than shadowed, and the owner
        # record describes whoever holds the namespace now.
        first = launcher.service_paths('/scratch/x', 'x', 'h2o_force')
        second = launcher.service_paths('/scratch/x', 'x', 'h2o_force')
        assert first.endpoint == second.endpoint
        assert first.owner_record == second.owner_record

    def test_a_normal_path_keeps_the_socket_beside_the_products(self):
        paths = launcher.service_paths('/scratch/tmp_LAMMPS', 'tmp_LAMMPS',
                                       'h2o_force')
        assert os.path.dirname(paths.endpoint) == '/scratch/tmp_LAMMPS'

    def test_a_path_too_long_for_af_unix_relocates_only_the_socket(self):
        deep = '/' + '/'.join(['directory'] * 30)
        paths = launcher.service_paths(deep, 'tmp_LAMMPS', 'h2o_force')
        # The relocated name stays inside the AF_UNIX budget even on a
        # platform whose temp root is a long per-user directory, and it is
        # still a name a person can read: the directory is the namespace,
        # the file is the service.
        assert len(paths.endpoint.encode('utf-8')) < launcher._AF_UNIX_LIMIT - 8
        assert paths.endpoint == os.path.join(
            launcher._runtime_dir('tmp_LAMMPS', deep), launcher.SOCKET_NAME)
        # The products and the record do not move with it: they are what a
        # user reads, and they belong beside the input files.
        assert paths.script.startswith(deep)
        assert paths.log.startswith(deep)
        assert paths.owner_record.startswith(deep)

    def test_isolated_long_paths_get_distinct_sockets(self):
        # Adaptive trajectories all share the execution namespace (the
        # isolated workdir drops the runtime suffix); their deep calcdirs
        # still need separate resident sockets.
        deep = '/' + '/'.join(['directory'] * 30)
        first = launcher.service_paths(
            deep + '/traj_1/tmp_low_qm2_LAMMPS', 'tmp_low_qm2_LAMMPS',
            'h2o_force')
        second = launcher.service_paths(
            deep + '/traj_2/tmp_low_qm2_LAMMPS', 'tmp_low_qm2_LAMMPS',
            'h2o_force')
        assert first.endpoint != second.endpoint

    def test_the_runtime_directory_is_private(self, tmp_path, monkeypatch):
        # An AF_UNIX socket is reachable by anyone who can name it, so the
        # directory holding it is created 0700 rather than at umask mercy.
        monkeypatch.setattr(launcher.tempfile, 'gettempdir',
                            lambda: str(tmp_path))
        launcher._ensure_runtime_dir(launcher._runtime_dir('exec_one'))
        mode = os.stat(launcher._runtime_dir('exec_one')).st_mode & 0o777
        assert mode == 0o700

    def test_the_namespace_is_sanitised_into_the_directory_name(self):
        first = launcher._runtime_dir('../../etc/cron')
        second = launcher._runtime_dir('tmp_LAMMPS')
        assert '..' not in first and '/' not in first.split('/')[-1]
        assert first != second

    def test_no_generated_name_appears_in_any_path(self):
        # Nothing here is random, so a second look at the directory finds
        # the same files by name; uniqueness comes from the directory.
        paths = launcher.service_paths('/scratch/x', 'x', 'h2o_force')
        for path in (paths.script, paths.log, paths.endpoint,
                     paths.owner_record):
            assert os.path.basename(path) in (
                'h2o_force.sh', 'h2o_force.log', launcher.SOCKET_NAME,
                launcher.OWNER_NAME)

    def test_the_service_runs_where_its_package_is(self, monkeypatch,
                                                   tmp_path):
        # The service's cwd is the calculation directory, so a LAMMPS
        # write command with a relative name puts a dump under debug
        # beside the input files, as the subprocess path does.
        seen = {}
        monkeypatch.setattr(launcher.tempfile, 'gettempdir',
                            lambda: str(tmp_path))

        class _Process:
            pid = 4242

            def poll(self):
                return None

        def fake_popen(command, **kwargs):
            seen['command'] = command
            seen['cwd'] = kwargs.get('cwd')
            seen['start_new_session'] = kwargs.get('start_new_session')
            return _Process()

        monkeypatch.setattr(launcher.subprocess, 'Popen', fake_popen)
        calcdir = str(tmp_path / 'tmp_LAMMPS')
        paths = launcher.service_paths(calcdir, 'tmp_LAMMPS', 'h2o_force')
        launcher.start_service(_Config(shell='module load openmpi'), paths,
                               timeout=1.0)
        assert seen['command'] == ['bash', os.path.join(calcdir,
                                                        'h2o_force.sh')]
        assert seen['cwd'] == calcdir
        assert seen['start_new_session'] is True
        # The owner record names where the log and the script went, which
        # is the whole point of keeping it after a hard kill.
        record = open(paths.owner_record).read()
        assert paths.log in record and paths.script in record

    def test_a_relative_calc_dir_is_anchored_on_one_base(self, monkeypatch,
                                                         tmp_path):
        # `environment.scratch_path` can be written relative, and then the
        # child's cwd and the script's own path would be one relative base
        # applied twice: bash would look for
        # 'scratch/tmp_LAMMPS/scratch/tmp_LAMMPS/h2o_force.sh'. One absolute
        # base is what keeps the deck the launcher wrote findable - the same
        # fix `local_runner.py` records for the cold path (#375).
        monkeypatch.chdir(tmp_path)
        paths = launcher.service_paths('scratch/tmp_LAMMPS', 'tmp_LAMMPS',
                                       'h2o_force')
        assert paths.work_dir == str(tmp_path / 'scratch' / 'tmp_LAMMPS')
        for path in (paths.script, paths.log, paths.owner_record):
            assert os.path.isabs(path)
        assert paths.script == os.path.join(paths.work_dir, 'h2o_force.sh')
        # Anchoring must not move anything: the same directory, named
        # absolutely.
        assert os.path.realpath(paths.work_dir) == os.path.realpath(
            'scratch/tmp_LAMMPS')

    def test_a_relative_launch_resolves_to_the_deck_it_wrote(self,
                                                             monkeypatch,
                                                             tmp_path):
        # The end-to-end half: the (cwd, script) pair handed to bash has to
        # name the file that was just written. Resolving the pair the way
        # bash does is the whole judgement - an absolute-looking script
        # argument is not enough, because the child applies its own cwd to
        # it.
        seen = {}
        monkeypatch.setattr(launcher.tempfile, 'gettempdir',
                            lambda: str(tmp_path))
        monkeypatch.chdir(tmp_path)

        class _Process:
            pid = 4242

            def poll(self):
                return None

        def fake_popen(command, **kwargs):
            seen['command'] = command
            seen['cwd'] = kwargs.get('cwd')
            return _Process()

        monkeypatch.setattr(launcher.subprocess, 'Popen', fake_popen)
        paths = launcher.service_paths('scratch/tmp_LAMMPS', 'tmp_LAMMPS',
                                       'h2o_force')
        launcher.start_service(_Config(), paths, timeout=1.0)
        resolved = os.path.join(seen['cwd'], seen['command'][1])
        assert os.path.realpath(resolved) == os.path.realpath(paths.script)
        assert os.path.isfile(resolved)


class TestSocketClaim:
    """One live service per socket, per process."""

    def test_second_claim_of_one_socket_is_refused(self):
        launcher.release_endpoint('/tmp/claim.sock')
        launcher.claim_endpoint('/tmp/claim.sock', 'unit_claim')
        try:
            with pytest.raises(LammpsPersistentUnsupported,
                               match='already live'):
                launcher.claim_endpoint('/tmp/claim.sock', 'unit_claim')
        finally:
            launcher.release_endpoint('/tmp/claim.sock')

    def test_release_allows_a_rebuild(self):
        launcher.release_endpoint('/tmp/rebuild.sock')
        launcher.claim_endpoint('/tmp/rebuild.sock', 'unit_rebuild')
        launcher.release_endpoint('/tmp/rebuild.sock')
        launcher.claim_endpoint('/tmp/rebuild.sock', 'unit_rebuild')
        try:
            assert launcher.live_services()['/tmp/rebuild.sock'] == \
                'unit_rebuild'
        finally:
            launcher.release_endpoint('/tmp/rebuild.sock')
        assert '/tmp/rebuild.sock' not in launcher.live_services()

    def test_one_namespace_holds_two_sockets(self):
        # The shape an isolated-workdir batch has: every displacement owns
        # its calculation directory and its socket, and the namespace is
        # shared on purpose because the runtime suffix is dropped when the
        # directory is already unique. Keying the claim on the namespace
        # refused the second displacement of a numerical Hessian, and with
        # it the whole batch (round tmp_20261031_005246).
        for socket in ('/tmp/hess_0.sock', '/tmp/hess_1.sock'):
            launcher.release_endpoint(socket)
        launcher.claim_endpoint('/tmp/hess_0.sock', 'tmp_all_field_LAMMPS')
        launcher.claim_endpoint('/tmp/hess_1.sock', 'tmp_all_field_LAMMPS')
        try:
            live = launcher.live_services()
            assert live['/tmp/hess_0.sock'] == 'tmp_all_field_LAMMPS'
            assert live['/tmp/hess_1.sock'] == 'tmp_all_field_LAMMPS'
        finally:
            for socket in ('/tmp/hess_0.sock', '/tmp/hess_1.sock'):
                launcher.release_endpoint(socket)


class TestReadiness:
    """connect_service waits for a handshake-capable listener, not a file."""

    def test_socket_file_alone_is_not_readiness(self, short_dir):
        # A leftover inode from a dead generation looks like a service and
        # is not one: connect() keeps failing, and the launcher must say so
        # instead of handing the backend a session to write into.
        endpoint = os.path.join(short_dir, 'svc.sock')
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(endpoint)
        stale.close()
        alive = subprocess.Popen([sys.executable, '-c', 'import time; '
                                                        'time.sleep(30)'])
        try:
            handle = launcher.ServiceHandle(alive, endpoint=endpoint,
                                            owner_record=os.path.join(
                                                short_dir, 'h2o_force.owner.json'),
                                            owner_pid=os.getpid(), nproc=1,
                                            python_path=sys.executable,
                                            log_path=os.path.join(short_dir,
                                                                 'x.log'),
                                            script_path=os.path.join(
                                                short_dir, 'h2o_force.sh'))
            with pytest.raises(LammpsPersistentUnsupported,
                               match='no connection'):
                launcher.connect_service(handle, timeout=1.0)
        finally:
            alive.kill()
            alive.wait()

    def test_a_service_that_died_early_quotes_its_log(self, tmp_path):
        # The engine writes its own error into the log, and the quoting is
        # the one the subprocess parser already uses, so a resident failure
        # reads like a cold one instead of stopping at an exit code.
        log = tmp_path / 'h2o_force.log'
        log.write_text('Step PotEng\nERROR: Pair hybrid sub-style is '
                       'not used (src/pair_hybrid.cpp:572)\n'
                       'Last command: run 0\n')
        process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(3)'])
        process.wait()
        handle = launcher.ServiceHandle(
            process, endpoint=str(tmp_path / 'gone.sock'),
            owner_record=str(tmp_path / 'h2o_force.owner.json'),
            owner_pid=os.getpid(), nproc=1,
            python_path=sys.executable, log_path=str(log),
            script_path=str(tmp_path / 'h2o_force.sh'))
        with pytest.raises(LammpsPersistentUnsupported,
                           match='Pair hybrid sub-style is not used'):
            launcher.connect_service(handle, timeout=2.0)
        assert 'Last command: run 0' in handle.read_log_tail()
        assert 'ERROR' in handle.read_log_tail()

    def test_a_missing_log_is_quoted_as_silence(self, tmp_path):
        # A service killed before it wrote anything must not turn the
        # error path into a second failure.
        handle = launcher.ServiceHandle(
            None, endpoint=str(tmp_path / 'gone.sock'),
            owner_record=str(tmp_path / 'h2o_force.owner.json'),
            owner_pid=os.getpid(), nproc=1,
            python_path=sys.executable,
            log_path=str(tmp_path / 'h2o_force.log'),
            script_path=str(tmp_path / 'h2o_force.sh'))
        assert handle.read_log_tail() == ''




class TestTeardown:
    """A reaped service leaves nothing that looks like a live endpoint."""

    def test_terminate_removes_the_socket_and_the_owner_record(self,
                                                               short_dir):
        endpoint = os.path.join(short_dir, 'h2o_force.sock')
        record = os.path.join(short_dir, 'runtime',
                              'h2o_force-abcd1234.owner.json')
        os.makedirs(os.path.dirname(record))
        with open(record, 'w') as handle:
            handle.write('{}')
        # A bound socket nobody listens on is what a dead service leaves
        # behind, and it answers a later connect() with ECONNREFUSED, which
        # is exactly the thing readiness has to be able to tell apart.
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(endpoint)
        stale.close()
        process = subprocess.Popen([sys.executable, '-c', 'pass'])
        process.wait()
        handle = launcher.ServiceHandle(
            process, endpoint=endpoint, owner_record=record,
            owner_pid=os.getpid(), nproc=1, python_path=sys.executable,
            log_path=os.path.join(short_dir, 'h2o_force.log'),
            script_path=os.path.join(short_dir, 'h2o_force.sh'))

        launcher.terminate_service(handle)

        assert not os.path.exists(endpoint)
        assert not os.path.exists(record)
