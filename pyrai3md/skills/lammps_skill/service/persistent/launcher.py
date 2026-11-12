"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Start, recognize and reap the resident LAMMPS service process.

Only this module spawns anything, and it never imports LAMMPS or MPI:
the service runs in its own interpreter because the framework process
must stay free of liblammps and of an MPI runtime. Everything it needs
comes from the skill's own configuration: python_path for the
interpreter, lammps_nproc for the rank count and shell for the module
loads, so the service sees exactly the environment the lmp binary sees
on the subprocess path.

Readiness is established by a HELLO round trip, never by the presence of
a socket file: a stale inode from an earlier generation looks identical
from here and would hand the backend a dead session.
"""

import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time

from collections import namedtuple

from ..templates.run_templates import _shell_setup
from ..utils import LOG_TAIL_LINES
from ..utils import read_log_tail
from .protocol import (
    CLOSE_TIMEOUT_S,
    IDLE_TIMEOUT_S,
    STARTUP_TIMEOUT_S,
    LammpsPersistentUnsupported,
)

# Where one resident service puts its files. See :func:`service_paths`.
ServicePaths = namedtuple(
    'ServicePaths', 'work_dir script log endpoint owner_record')

# The service is launched as a file, not through ``python -m``: the
# interpreter that owns LAMMPS needs no PyRAI3MD installation, no pydantic
# and nothing but the standard library plus this directory's protocol.py.
# Running the file also guarantees the service executes the same source
# tree the client came from, rather than whichever copy that interpreter
# happens to resolve by package name.
SERVER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'server.py')

# Path of the AF_UNIX sun_path field, and the margin that keeps a long
# scratch_path from producing a socket the service cannot bind.
_AF_UNIX_LIMIT = 108
_RUNTIME_DIR_PREFIX = 'pyrai3md-lammps'
_OWNER_CMDLINE_LIMIT = 200

# Process-local registry of live sessions, keyed by the socket path and
# valued with the execution namespace that asked for it. The socket is the
# key because it is what a live service is reachable by, and because the
# namespace is not unique per service: an isolated-workdir batch drops the
# runtime suffix on purpose (the calculation directory is already unique),
# so an optimization loop and every displacement of its numerical Hessian
# share one namespace while owning one directory and one socket each.
_LIVE_ENDPOINTS = {}
_LIVE_LOCK = threading.Lock()


class ServiceHandle:
    """Bookkeeping for one running service process group.

    Attributes
    ----------
    process : subprocess.Popen
        Leader of the service process group (mpirun when parallel).
    endpoint : str
        Socket path.
    owner_record : str
        Path of the owner record, which lives in the per-user runtime
        directory rather than beside the results.
    owner_pid : int
        Process id the service watchdog follows.
    nproc : int
        Rank count the launcher asked for.
    python_path : str
        Service interpreter.
    log_path : str
        Combined stdout/stderr of the whole group.
    info : dict
        HELLO reply of the accepted connection.
    """

    def __init__(self, process, *, endpoint, owner_record, owner_pid, nproc,
                 python_path, log_path, script_path, info=None):
        self.process = process
        self.endpoint = endpoint
        self.owner_record = owner_record
        self.owner_pid = owner_pid
        self.nproc = nproc
        self.python_path = python_path
        self.log_path = log_path
        self.script_path = script_path
        self.info = info or {}

    @property
    def alive(self) -> bool:
        """Whether the leader process is still running."""
        return self.process is not None and self.process.poll() is None

    def read_log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        """Quote the end of the service log, where LAMMPS wrote its error.

        The same reading the subprocess parser applies to
        ``{basename}.log``, so a resident failure is reported the way a
        cold one is instead of stopping at an exit code.
        """
        return read_log_tail(self.log_path, lines)


def _runtime_dir(execution_id: str = '', calcdir: str = '') -> str:
    """Per-user directory for the files a live service owns in the kernel.

    The owner record, and any socket the AF_UNIX length limit moved out of
    the calculation directory, live here rather than beside the results:
    they describe a process, not a calculation, and they are unlinked when
    it is gone. Mode 0700 because an AF_UNIX socket is reachable by anyone
    who can name it.

    ``calcdir`` disambiguates isolated adaptive runtimes whose execution
    namespace is deliberately stripped of the runtime suffix: their deep
    paths all relocate here, and without the hash every trajectory would
    claim the same socket. Only a short digest is appended, so the AF_UNIX
    length budget is preserved.
    """
    namespace = ''.join(character if character.isalnum() else '-'
                        for character in str(execution_id))
    if calcdir:
        # The AF_UNIX budget is tight on a macOS temp root; keep the
        # readable namespace very short and let the calcdir digest carry
        # the uniqueness.
        digest = hashlib.sha1(
            os.path.abspath(calcdir).encode('utf-8')).hexdigest()[:8]
        namespace = '%s-%s' % ((namespace or 'standalone')[:8], digest)
    else:
        namespace = namespace[:40]
    return os.path.join(tempfile.gettempdir(),
                        f'{_RUNTIME_DIR_PREFIX}-{_uid()}',
                        namespace or 'standalone')


def _ensure_runtime_dir(path: str) -> None:
    """Create *path* privately, tolerating another process having done so."""
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError:
        # A pre-existing directory with wider modes is the administrator's
        # or an earlier umask's doing; the socket inside is still only as
        # reachable as this run can make it.
        os.makedirs(path, exist_ok=True)


# Names the resident service gives the two files that describe it rather
# than a result. They are fixed, not derived from the run's stem: the
# calculation directory is already one backend's namespace, and a name like
# 'ff_gradient_persistent_test_force.sock' spends thirty characters of the
# AF_UNIX path budget to say what 'svc.sock' says in a directory that has
# only one service in it.
SOCKET_NAME = 'svc.sock'
OWNER_NAME = 'owner.json'


def service_paths(calcdir: str, execution_id: str = '',
                  basename: str = '') -> ServicePaths:
    """Choose where each resident artifact lives.

    Everything lives in the backend's calculation directory, exactly as on
    the subprocess path: the service loads its input package from there,
    the script and the log keep the cold path's stem (``{basename}.sh``,
    ``{basename}.log``), and the two process files are ``svc.sock`` and
    ``owner.json`` beside them. One backend is either cold or warm, so the
    two paths never compete for a name, and nothing is written outside the
    directory a person already looks in.

    The single exception is the socket's name length. AF_UNIX caps the
    address at 108 bytes, and a scratch path can be long enough that even
    ``svc.sock`` does not fit; the socket then binds inside
    :func:`_runtime_dir`, which is per-user and per execution namespace, so
    the name stays short without colliding, and :func:`start_service` says
    so at the top of the log rather than leaving the relocation to be
    guessed. The owner record never moves: it is an ordinary file with no
    length limit on its name.

    The directory is made absolute once, here, and every artifact is named
    from that one base: a relative ``calcdir`` (``environment.scratch_path``
    may be written relatively) would otherwise be applied twice, because the
    child is started with ``cwd=work_dir`` and would resolve the script's
    relative path one level too deep. ``local_runner.py`` records the same
    fix for the subprocess path.

    Parameters
    ----------
    calcdir : str
        Backend calculation directory.
    execution_id : str
        Execution namespace, used only to name the relocation directory.
    basename : str
        ``output_basename(project, profile)``, the cold path's stem.

    Returns
    -------
    ServicePaths
        Work directory, script path, log path, endpoint and owner record.
    """
    stem = basename or 'service'
    base = os.path.abspath(calcdir)
    endpoint = os.path.join(base, SOCKET_NAME)
    if len(endpoint.encode('utf-8')) >= _AF_UNIX_LIMIT - 8:
        endpoint = os.path.join(_runtime_dir(execution_id, base), SOCKET_NAME)
    return ServicePaths(
        work_dir=base,
        script=os.path.join(base, f'{stem}.sh'),
        log=os.path.join(base, f'{stem}.log'),
        endpoint=endpoint,
        owner_record=os.path.join(base, OWNER_NAME),
    )


def _uid() -> int:
    """Effective user id, 0 where the platform has none."""
    return getattr(os, 'getuid', lambda: 0)()


def build_service_argv(python_path: str, endpoint: str, work_dir: str,
                       owner_pid: int, nproc: int,
                       startup_timeout: float = STARTUP_TIMEOUT_S) -> list:
    """Build the command that starts the service.

    A single rank runs the interpreter directly: wrapping it in mpirun
    would need a launcher the run may already be inside of.

    Parameters
    ----------
    nproc : int
        MPI rank count; 1 runs without a launcher.
    work_dir : str
        Directory the service runs in and loads its package from.

    Returns
    -------
    list of str
        Argument vector, interpreter first (or mpirun first).
    """
    nproc = int(nproc)
    if nproc < 1:
        raise LammpsPersistentUnsupported(
            f'lammps_nproc must be >= 1, got {nproc}')
    argv = [python_path, SERVER_FILE,
            '--endpoint', endpoint, '--session', work_dir,
            '--owner-pid', str(int(owner_pid)), '--nproc', str(nproc),
            '--startup-timeout', repr(float(startup_timeout)),
            '--idle-timeout', repr(float(IDLE_TIMEOUT_S))]
    if nproc == 1:
        return argv
    return ['mpirun', '-np', str(nproc)] + argv


def build_service_script(config, argv: list) -> str:
    """Wrap the service command in the skill's shell prologue.

    The same ``shell`` lines the subprocess path injects before lmp
    (module load of the compiler and the MPI stack) run here too, so the
    service interpreter resolves the same liblammps the binary would, and
    a parallel world is launched by the MPI that build links. They are
    copied verbatim into a script and run by bash exactly as on the cold
    path, because a site that needs a ``module load`` to see its LAMMPS
    needs it in the service process too.
    ``exec`` keeps the launcher's child as the process-group leader that
    killpg() can reach with every rank behind it. ``OMP_NUM_THREADS`` is
    exported from ``thread`` here as well, so a service rank gets the
    same thread count the cold subprocess path would have given it
    (#304; 1 remains the keyword default).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration carrying the shell prologue.
    argv : list of str
        Command from :func:`build_service_argv`.

    Returns
    -------
    str
        Bash script text.
    """
    command = ' '.join(shlex.quote(token) for token in argv)
    return (
        '#!/bin/bash\n'
        f'{_shell_setup(config)}'
        f'export OMP_NUM_THREADS={config.thread}\n'
        f'exec {command}\n'
    )


def start_service(config, paths: ServicePaths,
                  timeout: float = STARTUP_TIMEOUT_S) -> ServiceHandle:
    """Launch the service process group and wait for its socket.

    The script and the log carry the cold path's ``{basename}.sh`` and
    ``{basename}.log`` names inside the calculation directory, so a
    resident run is inspected with the habits the subprocess run already
    taught, and the process runs with that directory as its working
    directory: a LAMMPS ``dump`` or ``write_dump`` under ``debug`` then
    lands where the same command would have written it cold.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration (python_path, lammps_nproc, shell).
    paths : ServicePaths
        Where this service works, writes its script and log, and binds.
    timeout : float
        Startup deadline.

    Returns
    -------
    ServiceHandle
        Handle whose ``process`` is already listening.
    """
    python_path = getattr(config, 'python_path', '') or sys.executable
    nproc = int(getattr(config, 'lammps_nproc', 1) or 1)

    os.makedirs(paths.work_dir, exist_ok=True)
    if os.path.dirname(os.path.abspath(paths.endpoint)) != \
            os.path.abspath(paths.work_dir):
        # Only the relocation needs a directory of its own; the normal case
        # creates nothing outside the calculation directory.
        _ensure_runtime_dir(os.path.dirname(paths.endpoint))
    _remove_stale_socket(paths.endpoint)
    argv = build_service_argv(python_path, paths.endpoint, paths.work_dir,
                              os.getpid(), nproc, timeout)
    with open(paths.script, 'w') as handle:
        handle.write(build_service_script(config, argv))
    os.chmod(paths.script, 0o755)

    log_file = open(paths.log, 'w')
    if os.path.dirname(os.path.abspath(paths.endpoint)) != \
            os.path.abspath(paths.work_dir):
        log_file.write(f'# endpoint relocated: {paths.endpoint}\n')
        log_file.flush()
    try:
        process = subprocess.Popen(['bash', paths.script],
                                   cwd=paths.work_dir,
                                   stdout=log_file, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    except OSError as exc:
        raise LammpsPersistentUnsupported(
            f'cannot start the resident LAMMPS service: {exc}') from exc
    finally:
        log_file.close()

    handle = ServiceHandle(process, endpoint=paths.endpoint,
                           owner_record=paths.owner_record,
                           owner_pid=os.getpid(), nproc=nproc,
                           python_path=python_path, log_path=paths.log,
                           script_path=paths.script)
    _write_owner_record(handle)
    return handle


def connect_service(handle: ServiceHandle, timeout: float,
                    connect_interval: float = 0.05) -> socket.socket:
    """Connect to the service, refusing to trust a socket file alone.

    Returns once the endpoint accepts a connection; the HELLO round trip
    that proves the world is complete belongs to the client.

    Parameters
    ----------
    handle : ServiceHandle
        Started service.
    timeout : float
        Startup deadline.
    connect_interval : float
        Poll spacing.

    Returns
    -------
    socket.socket
        Connected AF_UNIX stream socket.
    """
    deadline = time.monotonic() + float(timeout)
    last_error = None
    while True:
        if not handle.alive:
            code = handle.process.returncode
            raise LammpsPersistentUnsupported(
                f'resident LAMMPS service exited with {code} before it was '
                f'ready ({handle.script_path}, {handle.log_path}). '
                f'LAMMPS log tail:\n{handle.read_log_tail()}')
        try:
            stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stream.settimeout(max(1.0, deadline - time.monotonic()))
            stream.connect(handle.endpoint)
            return stream
        except OSError as exc:
            last_error = exc
            stream.close()
        if time.monotonic() >= deadline:
            raise LammpsPersistentUnsupported(
                f'no connection to {handle.endpoint} within {timeout:g}s '
                f'(last error: {last_error}; {handle.log_path}). '
                f'LAMMPS log tail:\n{handle.read_log_tail()}')
        time.sleep(connect_interval)


def terminate_service(handle: ServiceHandle,
                      timeout: float = CLOSE_TIMEOUT_S) -> None:
    """Ensure the whole service process group is gone.

    Ordering matters: ask for a graceful stop first, then signal the
    group, then force. A bare ``process.kill()`` would orphan every rank
    but the leader and leave them sitting in a LAMMPS collective.

    Parameters
    ----------
    handle : ServiceHandle
        Service to stop.
    timeout : float
        Patience per stage.
    """
    process = handle.process
    if process is None:
        return
    if process.poll() is not None:
        _retire(handle)
        return

    _signal_group(process, 15, timeout)                 # SIGTERM
    if process.poll() is None:
        _signal_group(process, 9, timeout)              # SIGKILL
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:                   # pragma: no cover
        pass
    _retire(handle)


def _retire(handle: ServiceHandle) -> None:
    """Remove the two files that describe this process, not a result.

    The owner record is never a product, and a socket left behind answers
    a later connect() with ECONNREFUSED and looks like a service. The
    calculation directory is not touched here: that is keep_tmp's call.
    """
    _unlink(handle.endpoint)
    _unlink(handle.owner_record)


def _signal_group(process, signal: int, timeout: float) -> None:
    """Send *signal* to the service process group, ignoring a dead one."""
    try:
        os.killpg(os.getpgid(process.pid), signal)
    except (ProcessLookupError, PermissionError, OSError):
        # Already gone, or a platform without killpg: fall back to the
        # leader so the ranks are at least abandoned by their parent.
        try:
            process.send_signal(signal)
        except (OSError, ValueError):
            return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.02)


def _write_owner_record(handle: ServiceHandle) -> None:
    """Record who owns this service, for forensics after a hard kill.

    The record names the log and the script as well, because those are
    what a person looks for next, and both are under the calculation
    directory rather than here.
    """
    record = {
        'owner_pid': handle.owner_pid,
        'endpoint': handle.endpoint,
        'work_dir': os.path.dirname(handle.log_path),
        'script': handle.script_path,
        'log': handle.log_path,
        'python_path': handle.python_path,
        'nproc': handle.nproc,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    try:
        with open(handle.owner_record, 'w') as f:
            json.dump(record, f, indent=1)
    except OSError:
        pass


def _remove_stale_socket(endpoint: str) -> None:
    """Unlink a socket no one is listening on.

    A leftover inode from a previous generation answers connect() with
    ECONNREFUSED, which is what distinguishes it from a live service.
    """
    if not os.path.exists(endpoint):
        return
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(0.2)
    try:
        listener.connect(endpoint)
    except OSError:
        listener.close()
        _unlink(endpoint)
        return
    listener.close()
    raise LammpsPersistentUnsupported(
        f'an active service already listens on {endpoint}; refusing to '
        f'take it over')


def _unlink(path: str) -> None:
    """Remove *path* if present, ignoring a concurrent removal."""
    try:
        os.unlink(path)
    except OSError:
        pass


def claim_endpoint(endpoint: str, execution_id: str = '') -> str:
    """Reserve one socket path for one live service.

    Parameters
    ----------
    endpoint : str
        Socket the claimant will serve on.
    execution_id : str
        Execution namespace asking for it, recorded for diagnostics and
        named in the refusal. It is not the key: two services of one
        namespace in two directories are two processes with a socket each,
        which is what an isolated-workdir batch is.

    Returns
    -------
    str
        The claim token to release later.

    Raises
    ------
    LammpsPersistentUnsupported
        When another live service already holds this socket.
    """
    token = f'{execution_id}|{endpoint}'
    with _LIVE_LOCK:
        held = _LIVE_ENDPOINTS.get(endpoint)
        if held is not None:
            raise LammpsPersistentUnsupported(
                f'a resident LAMMPS service is already live on {endpoint} '
                f'(claimed for execution_id {held!r}); a second service '
                f'cannot bind that socket, and taking the live one over '
                f'would leave it unreachable to its own close(). Give the '
                f'second task its own calculation directory, or set '
                f'python_enabled=false.')
        _LIVE_ENDPOINTS[endpoint] = execution_id or endpoint
    return token


def release_endpoint(endpoint: str) -> None:
    """Hand the socket back so a rebuild can claim it again."""
    with _LIVE_LOCK:
        _LIVE_ENDPOINTS.pop(endpoint, None)


def live_services():
    """Snapshot of claimed sockets (diagnostics and tests).

    Returns
    -------
    dict
        ``{endpoint: execution_id}`` for every live claim of this process.
    """
    with _LIVE_LOCK:
        return dict(_LIVE_ENDPOINTS)


__all__ = [
    'ServiceHandle',
    'ServicePaths',
    'SERVER_FILE',
    'build_service_argv',
    'build_service_script',
    'claim_endpoint',
    'connect_service',
    'live_services',
    'release_endpoint',
    'service_paths',
    'start_service',
    'terminate_service',
]
