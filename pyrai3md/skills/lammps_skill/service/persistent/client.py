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

Backend-side handle of one resident LAMMPS session.

The session is the backend's own: it is established during
initialization, carries only dynamic geometry per calculation step, and
is released when the backend is closed. Nothing here imports LAMMPS or
MPI, so the framework process never links either of them, and nothing
here converts units: a reply is validated and handed over raw, and
service/parser turns it into framework results. That split keeps one
answer to "where does a LAMMPS number become a Hartree number".

Three properties are enforced here rather than trusted from the peer:

  single request in flight   one session serves one operation at a time,
                             so a reply can always be matched to it
  no silent success          a reply that is not exactly right raises;
                             it is never zero-filled or reused
  no retry after a timeout   the client cannot tell whether the service
                             consumed the operation, so the session is
                             poisoned instead of replaying it

"""

import os
import threading
import time

from .protocol import (
    OP_ERROR,
    OP_EVALUATE,
    OP_HELLO,
    OP_INIT,
    OP_RESULT,
    PROTOCOL_VERSION,
    REQUEST_TIMEOUT_S,
    LammpsProtocolError,
    LammpsServiceDead,
    LammpsServiceTimeout,
    build_close,
    build_evaluate,
    build_hello,
    build_init,
    check_fields,
    encode_frame,
    read_frame,
)

class PersistentSession:
    """One socket session against one resident LAMMPS instance.

    Parameters
    ----------
    stream : socket.socket
        Connected AF_UNIX stream socket owned by this session.
    session : str
        Session directory the service was told to work in.
    fingerprint : str
        Session identity recomputed by the service on every operation.
    natom : int
        Atom count the instance was loaded with.
    order : list of int
        Atom id order matching the framework atom list.
    units : str
        LAMMPS units style of the loaded input.
    capabilities : dict
        ``{'gradient': bool, 'stress': bool}`` requested extractions.
    expected_world : int, optional
        MPI rank count the service must report; checked at HELLO.
    request_timeout : float
        Deadline for one operation.
    """

    def __init__(self, stream, *, session, fingerprint, natom, order, units,
                 capabilities, expected_world=None,
                 request_timeout=REQUEST_TIMEOUT_S, log_path=''):
        self._stream = stream
        self._session = session
        self._fingerprint = fingerprint
        self._natom = int(natom)
        self._order = [int(tag) for tag in order]
        self._units = units
        self._capabilities = {'gradient': bool(capabilities.get('gradient')),
                              'stress': bool(capabilities.get('stress'))}
        self._expected_world = expected_world
        self._request_timeout = float(request_timeout)
        # The service's own log, and the last request that was put on the
        # wire. A dead session is reported by what it was doing when it
        # died, because the socket gives no other account of it: the
        # qualification run lost eight NPT examples to
        # "service closed the connection before a frame header" and the
        # step, the box and the engine's own error line were all still
        # sitting in files nobody read.
        self._log_path = str(log_path or '')
        self._last_request = None

        # Re-entrant: the failure path of an evaluate() closes the session
        # while still holding this lock.
        self._lock = threading.RLock()
        self._ids = iter(range(1, 1 << 31))
        self._generation = 0
        self._closed = False
        self._poison = None
        # Client-side socket time per finished operation (see _tally).
        self._perf = {}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def poisoned(self):
        """Reason the session died, or None while it is usable."""
        return self._poison

    @property
    def closed(self):
        """Whether close() has run."""
        return self._closed

    @property
    def generation(self):
        """Number of operations the service has accepted so far."""
        return self._generation

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def hello(self) -> dict:
        """Ask the service what it is, before sending anything to load.

        Returns
        -------
        dict
            The ``info`` block of the reply.

        Raises
        ------
        LammpsProtocolError
            Protocol mismatch, or a world size other than the one the
            configuration asked for.
        """
        reply = self._request(build_hello(next(self._ids)), OP_HELLO)
        info = reply.get('info')
        if not isinstance(info, dict):
            raise self._fail('HELLO reply carries no info block')
        if info.get('protocol') != PROTOCOL_VERSION:
            raise self._fail(
                f"service speaks protocol {info.get('protocol')!r}, this "
                f"client speaks {PROTOCOL_VERSION}")
        world = info.get('world_size')
        if (self._expected_world is not None and world is not None
                and int(world) != int(self._expected_world)):
            raise self._fail(
                f"service reports a world of {world} ranks, configuration "
                f"asked for {self._expected_world}")
        return info

    def initialize(self, *, cwd, script, files, profile, pbc, step_tail,
                   kspace: str = '', qmmm: dict = None) -> dict:
        """Load the static package and return the READY reply.

        Parameters
        ----------
        step_tail : list of str
            Commands the service issues for every step, including this
            initialization's smoke evaluation.

        Returns
        -------
        dict
            The reply payload: ``natom``, ``triclinic``, ``box`` and the
            finite smoke energy.
        """
        message = build_init(
            request_id=next(self._ids), session=self._session,
            fingerprint=self._fingerprint, cwd=cwd, script=script,
            files=files, profile=profile, natom=self._natom,
            order=self._order, units=self._units, pbc=pbc,
            capabilities=dict(self._capabilities), step_tail=step_tail,
            kspace=kspace, qmmm=qmmm)
        reply = self._request(message, OP_INIT)
        if reply.get('fingerprint') != self._fingerprint:
            raise self._fail('INIT reply echoed a different fingerprint')
        if int(reply.get('natom', -1)) != self._natom:
            raise self._fail(
                f"service loaded {reply.get('natom')} atoms, the model "
                f"declares {self._natom}")
        energy = reply.get('smoke_energy_raw')
        if not _is_finite_number(energy):
            raise self._fail(
                f'smoke evaluation returned a non-finite energy: {energy!r}')
        return reply

    def evaluate(self, coordinates, box, step: int, image=None,
                 q_qm=None) -> dict:
        """Send one geometry and return the raw LAMMPS numbers.

        Parameters
        ----------
        coordinates : list of list of float
            (natom, 3) wrapped positions in Angstrom, in the main image as
            the framework keeps them.
        box : dict or None
            Box numbers for this step in the frame the engine takes
            them in (see input_builder.resolve_live_box), or None to
            keep the current box.
        step : int
            Monotonic step index, echoed back for log correlation.
        image : list of list of int, optional
            (natom, 3) crossing counts belonging to these coordinates.
            They travel with the positions because the service's live
            instance still carries the previous step's flags; None is
            read as "no atom has crossed".

        Returns
        -------
        dict
            ``energy_raw`` plus ``forces_raw`` and ``pressure6_raw`` when
            those capabilities were requested, in native units.

        Raises
        ------
        LammpsServiceDead
            The session died earlier, or this reply failed validation.
        LammpsServiceTimeout
            No reply in time; the session is poisoned, never replayed.
        """
        message = build_evaluate(
            request_id=next(self._ids), session=self._session,
            fingerprint=self._fingerprint, step=step,
            coordinates=coordinates, box=box, image=image,
            q_qm=q_qm)
        self._last_request = {
            'step': int(step),
            # `is not None`, not `or`: the framework hands coordinates
            # and image flags over as numpy arrays, and `array or ()` is
            # not a falsy test -- it raises "the truth value of an array
            # with more than one element is ambiguous". The array case is
            # the only case that ever reaches a running MD, so this is
            # recorded as a regression (test_evaluate_accepts_numpy_geometry).
            'natom': len(coordinates) if coordinates is not None else 0,
            'box': box,
        }
        return self._request(message, OP_EVALUATE)

    def close(self, reason: str = 'owner_close') -> None:
        """Release the instance, idempotently and never raising.

        A CLOSE that the service cannot acknowledge is still a closed
        session locally: the caller has already decided to stop using it,
        and the launcher's process-group teardown is what guarantees the
        ranks actually go away.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._stream is None:
                return
            try:
                self._stream.sendall(
                    encode_frame(build_close(request_id=next(self._ids),
                                             reason=reason)))
            except OSError:
                pass
            self._stream = None

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    def _request(self, message, expect_for: str) -> dict:
        """Send one message and return the accepted RESULT.

        The lock spans send and receive: one session, one operation in
        flight, so a reply cannot be attributed to the wrong request.
        """
        with self._lock:
            if self._poison is not None:
                raise LammpsServiceDead(
                    f'resident LAMMPS session is unusable: {self._poison}')
            if self._stream is None or self._closed:
                raise LammpsServiceDead('resident LAMMPS session is closed')
            if expect_for not in (OP_HELLO, OP_INIT, OP_EVALUATE):
                raise LammpsProtocolError(
                    f'internal error: {expect_for} is not a request')
            try:
                check_fields(message, expect_for)
                # The deadline is only enforceable if the socket itself
                # gives bytes back: a blocking read would sit in the
                # kernel long after the request timed out, and a dead
                # service would hang the framework step instead of
                # raising.
                self._stream.settimeout(max(1.0, self._request_timeout))
                started = time.perf_counter()
                self._stream.sendall(encode_frame(message))
                answered_send = time.perf_counter()
            except OSError as exc:
                raise self._mark_dead(
                    f'sending {expect_for} failed: {exc}') from exc

            deadline = time.monotonic() + self._request_timeout
            try:
                reply = read_frame(self._stream, deadline=deadline)
            except LammpsServiceTimeout as exc:
                raise self._mark_dead(
                    f'{expect_for} timed out after '
                    f'{self._request_timeout:g}s; the service may still be '
                    f'running this operation, so the session is not reused'
                ) from exc
            except (LammpsProtocolError, LammpsServiceDead) as exc:
                raise self._mark_dead(str(exc)) from exc

            answered = time.perf_counter()
            accepted = self._accept(reply, expect_for, message)
            self._tally(expect_for, answered_send - started,
                        answered - answered_send,
                        time.perf_counter() - answered)
            return accepted

    def _tally(self, expect_for: str, send_s: float, wait_s: float,
               parse_s: float) -> None:
        """Accumulate the socket time of one finished operation.

        The service reports what a step costs it; this reports what the
        framework process itself waits for, which is the only way to tell
        a slow engine apart from a slow pipe.
        """
        record = self._perf.setdefault(expect_for, [0.0, 0.0, 0.0, 0])
        record[0] += send_s
        record[1] += wait_s
        record[2] += parse_s
        record[3] += 1

    def timings(self) -> dict:
        """Milliseconds of client-side time per finished operation.

        The three buckets are the three things a step can actually be
        waiting on, and their boundaries are deliberate:

        ``send_ms``
            Encoding the request and writing it to the socket. The frame
            is a few kilobytes of JSON, so this is where a payload that
            grew beyond its geometry would show up.
        ``wait_ms``
            From the first byte requested to the last byte of the reply:
            the service's own work plus the pipe around it. Read together
            with the service's phase table this separates an engine that
            is slow from a session that is merely far away.
        ``parse_ms``
            Decoding the reply and validating it - identity, generation,
            shape, finiteness. Every number the step receives is walked
            here, so a per-element check that belongs on the engine side
            cannot hide in the wait.

        Returns
        -------
        dict
            {operation: {'calls', 'send_ms', 'wait_ms', 'parse_ms'}} for
            every operation that completed; empty while none has.
        """
        return {
            name: {'calls': int(row[3]),
                   'send_ms': 1000.0 * row[0] / row[3],
                   'wait_ms': 1000.0 * row[1] / row[3],
                   'parse_ms': 1000.0 * row[2] / row[3]}
            for name, row in sorted(self._perf.items()) if row[3]
        }

    def _accept(self, reply, expect_for: str, request: dict) -> dict:
        """Validate the reply of one request, or raise."""
        op = check_fields(reply)
        request_id = reply.get('id')
        generation = int(reply.get('gen', -1))
        if request_id != request['id']:
            raise self._mark_dead(
                f"reply answers request {request_id}, this request was "
                f"{request['id']}")
        if op == OP_ERROR:
            if reply.get('for') != expect_for:
                raise self._mark_dead(
                    f"error names {reply.get('for')!r} while waiting for "
                    f"{expect_for}")
            message = (f"{expect_for} failed [{reply.get('kind')}/"
                       f"{reply.get('stage')}]: {reply.get('message')}")
            if reply.get('session_usable', True):
                # The service kept its control flow consistent; this
                # request simply did not run. The session survives, but
                # the generation still advanced past the rejected op.
                self._advance(generation, expect_for)
                raise LammpsProtocolError(message)
            raise self._mark_dead(message)
        if op != OP_RESULT:
            raise self._mark_dead(
                f'expected a RESULT reply, got {op} for {expect_for}')
        if reply.get('for') != expect_for:
            raise self._mark_dead(
                f"RESULT names {reply.get('for')!r} while waiting for "
                f"{expect_for}")
        self._advance(generation, expect_for)
        if expect_for == OP_EVALUATE:
            # The geometry checks live here rather than in evaluate()
            # because that is where the reply's validation window is: a
            # step that ships 180 numbers and gets 180 back is only
            # known to be the same 180 after these checks walk them, and
            # time spent proving it belongs to the same bucket as the
            # rest of the proof, not to the wait on the service.
            self._check_reply_identity(reply, request)
            self._check_geometry(reply, request['coordinates'])
        return reply

    def _advance(self, generation: int, expect_for: str) -> None:
        """Enforce the strictly increasing reply order."""
        if generation != self._generation + 1:
            raise self._mark_dead(
                f'service generation jumped from {self._generation} to '
                f'{generation} on {expect_for}')
        self._generation = generation

    def _check_reply_identity(self, reply, request: dict) -> None:
        """Verify the session identity echoed by an EVALUATE reply."""
        for field in ('session', 'fingerprint'):
            if reply.get(field) != request[field]:
                raise self._mark_dead(
                    f'{field} mismatch: the service answered for a '
                    f'different session')
        if int(reply.get('step', -1)) != int(request['step']):
            raise self._mark_dead(
                f"reply echoes step {reply.get('step')} for step "
                f"{request['step']}")
        if reply.get('order') != 'atom_id':
            raise self._mark_dead(
                f"reply does not declare its atom order as 'atom_id': "
                f"{reply.get('order')!r}")
        if reply.get('units') != 'lammps-native':
            raise self._mark_dead(
                f"reply units are {reply.get('units')!r}, not "
                f"'lammps-native'; converting again would double the factor")

    def _check_geometry(self, reply, coordinates) -> None:
        """Verify shapes and finiteness of everything about to be stored."""
        natom = self._natom
        if int(reply.get('natom', -1)) != natom:
            raise self._mark_dead(
                f"service evaluated {reply.get('natom')} atoms, the model "
                f"declares {natom}")

        energy = reply.get('energy_raw')
        if not _is_finite_number(energy):
            raise self._mark_dead(f'energy is not a finite number: {energy!r}')

        if self._capabilities['gradient']:
            forces = reply.get('forces_raw')
            if _shape_of(forces) != (natom, 3):
                raise self._mark_dead(
                    f'forces have shape {_shape_of(forces)}, expected '
                    f'({natom}, 3)')
            if not _all_finite(forces):
                raise self._mark_dead('forces contain a non-finite value')
        elif 'forces_raw' in reply:
            raise self._mark_dead(
                'service returned forces that were never requested')

        if self._capabilities['stress']:
            pressure = reply.get('pressure6_raw')
            if _shape_of(pressure) != (6,):
                raise self._mark_dead(
                    f'pressure has shape {_shape_of(pressure)}, expected '
                    f'(6,)')
            if not _all_finite(pressure):
                raise self._mark_dead(
                    'pressure components contain a non-finite value')
        elif 'pressure6_raw' in reply:
            raise self._mark_dead(
                'service returned a pressure tensor that was never requested')

    def _fail(self, message: str) -> LammpsProtocolError:
        """Poison the session and return the error to raise."""
        self._poison = message
        return LammpsProtocolError(f'{message} (session poisoned)')

    def _mark_dead(self, message: str) -> LammpsServiceDead:
        """Poison the session and return the dead-service error.

        The protocol error text stays as the service wrote it; what the
        socket cannot carry is appended, so the person reading the
        qualification report does not have to guess which step died or
        open a log file that the next task in the same directory has
        already overwritten.
        """
        self._poison = message
        return LammpsServiceDead(
            f'resident LAMMPS session died: {message}{self._death_context()}')

    def _death_context(self) -> str:
        """Describe what the session was doing when it stopped answering.

        Returns
        -------
        str
        A `; `-led tail: the last request sent (step, atom count, box)
        and the last lines of the service's own log. Empty when nothing
        was ever requested and no log exists, so a failure at startup
        stays exactly as terse as it was.
        """
        parts = []
        request = self._last_request
        if request:
            box = request.get('box')
            parts.append('last request step=%d natom=%d box=%s'
                         % (request['step'], request['natom'],
                            _short(box, 120)))
        tail = _read_tail(self._log_path)
        if tail:
            parts.append('service log ends: %s' % _short(tail, 400))
        return ('' if not parts else ' (' + '; '.join(parts) + ')')


# ----------------------------------------------------------------------
# Small data helpers, kept local so the client needs no numpy
# ----------------------------------------------------------------------

# How much of the service log a death report carries: the tail is where
# an engine states why it stopped, and the whole file can be megabytes of
# step lines.
LOG_TAIL_BYTES = 4096
LOG_TAIL_LINES = 6


def _short(value, limit: int) -> str:
    """Render a value in at most `limit` characters, on one line."""
    text = str(value).replace('\n', ' | ')
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _read_tail(path: str) -> str:
    """Return the last lines of a log file, or '' when there are none."""
    if not path:
        return ''
    try:
        size = os.path.getsize(path)
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            if size > LOG_TAIL_BYTES:
                handle.seek(size - LOG_TAIL_BYTES)
                handle.readline()
            lines = [line.rstrip() for line in handle if line.strip()]
    except OSError:
        return ''
    return '\n'.join(lines[-LOG_TAIL_LINES:])


def _is_finite_number(value) -> bool:
    """True for a real, finite number (bool excluded)."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float('inf'), float('-inf'))


def _shape_of(value):
    """Return the (rows, columns) shape of a nested list, or None."""
    if not isinstance(value, (list, tuple)):
        return None
    if all(isinstance(row, (list, tuple)) for row in value):
        widths = {len(row) for row in value}
        if len(widths) > 1:
            return (len(value), None)
        return (len(value), widths.pop() if widths else 0)
    return (len(value),)


def _all_finite(value) -> bool:
    """True when a nested list holds only finite numbers."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif not _is_finite_number(item):
            return False
    return True
