"""
A fake resident LAMMPS service for the unit tests.

The real service needs a LAMMPS build and an MPI runtime, which the
development machine does not have, so the protocol, the client and the
backend wiring are tested against this in-process stand-in instead. It
speaks the same frames through the same AF_UNIX socket and imitates a
correct service by default, and each test bends one behaviour (wrong
identity, wrong shape, NaN, silence, death) to check that the client
notices.

It deliberately knows the wire format only through service/persistent/
protocol.py, so a client bug cannot be hidden by a matching bug here.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import shutil
import socket
import tempfile
import threading
import time

from ..service.persistent import launcher
from ..service.persistent.protocol import (
    OP_CLOSE,
    OP_EVALUATE,
    OP_HELLO,
    OP_INIT,
    PROTOCOL_VERSION,
    build_error,
    build_result,
    check_fields,
    LammpsServiceTimeout,
    encode_frame,
    read_frame)

# Geometry the fake service "evaluates": a linear function of the
# coordinates, so a test can assert the reply actually depends on the
# request rather than on a constant the service memorized.
ENERGY_SCALE = -2.0


class FakeLammpsService:
    """Threaded stand-in for the resident service.

    Parameters
    ----------
    endpoint : str
        Socket path to bind.
    info : dict, optional
        HELLO payload overrides.
    on_evaluate : callable, optional
        ``(request, count) -> dict`` returning a replacement RESULT
        payload, an ERROR message, or raising to simulate a crash.
    on_init : callable, optional
        Same hook for INIT.
    generation_offset : int
        Added to every reply generation (to simulate a skipped or
        repeated operation).
    silence : bool
        Answer nothing at all, to test the request deadline.
    """

    def __init__(self, endpoint=None, *, info=None, on_evaluate=None,
                 on_init=None, generation_offset=0, silence=False):
        # AF_UNIX paths are short (104 bytes on macOS, 108 on Linux) and a
        # pytest tmp_path easily exceeds that, so the default home is a
        # directory of its own next to the other temporary roots.
        self._base = None
        if endpoint is None:
            root = '/tmp' if os.path.isdir('/tmp') else tempfile.gettempdir()
            self._base = tempfile.mkdtemp(prefix='lmpfake-', dir=root)
            endpoint = os.path.join(self._base, 'svc.sock')
        self.endpoint = endpoint
        self.info = {'protocol': PROTOCOL_VERSION, 'lammps_version': 'fake',
                     'has_mpi_support': True, 'mpi4py': True,
                     'world_size': 1, 'nproc_expected': 1,
                     'python': {'executable': 'fake', 'version': '0.0.0'},
                     'api_probe': {}}
        if info:
            self.info.update(info)
        self._on_evaluate = on_evaluate
        self._on_init = on_init
        self._generation_offset = generation_offset
        self._silence = silence

        self.operations = []
        self.init_request = None
        self.generation = 0
        self.stopped = False
        self.error = None
        self._listener = None
        self._thread = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 5.0):
        """Bind and start serving in a background thread."""
        parent = os.path.dirname(self.endpoint)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # start_service() clears an endpoint nobody is listening on before
        # it launches, and this class stands in for a started service: a
        # file left by an aborted run would otherwise make the suite
        # depend on the order it ran in.
        launcher._remove_stale_socket(self.endpoint)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.endpoint)
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not os.path.exists(self.endpoint):
            if time.monotonic() > deadline:
                raise RuntimeError('fake service did not create its endpoint')
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        """Close the listener and join the serving thread."""
        self.stopped = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._base:
            shutil.rmtree(self._base, ignore_errors=True)
        else:
            # A caller that handed over an endpoint owns the directory it
            # sits in, and that directory is usually a calculation
            # directory a later test reuses: leave the socket file behind
            # and the next bind reports a stranger's address as its own.
            try:
                os.unlink(self.endpoint)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def _run(self):
        """Accept one client and serve it until it goes away."""
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return
        conn.settimeout(0.5)
        try:
            while not self.stopped:
                try:
                    message = read_frame(conn)
                except LammpsServiceTimeout:
                    # The socket timeout here is only a poll interval: the
                    # fake must keep waiting when a test makes it silent, or
                    # it would hang up instead of letting the client time
                    # out the way the real service would.
                    continue
                except Exception:                    # noqa: BLE001
                    break
                try:
                    check_fields(message)
                except Exception as exc:             # noqa: BLE001
                    self._send(conn, build_error(
                        for_op='unknown', request_id=0, generation=0,
                        kind='protocol', stage='validate',
                        session_usable=True, message=str(exc)))
                    continue
                self.operations.append(message)
                if message['op'] == OP_INIT:
                    self.init_request = message
                self.generation += 1
                reply = self._reply(message)
                if reply is None:
                    continue
                self._send(conn, reply)
                if message['op'] == OP_CLOSE:
                    break
        finally:
            conn.close()

    def _send(self, conn, message):
        """Write one frame unless configured to stay silent."""
        if self._silence:
            return
        try:
            conn.sendall(encode_frame(message))
        except OSError:
            self.error = 'client vanished'

    def _reply(self, message):
        """Build the answer to one operation."""
        generation = self.generation + self._generation_offset
        op = message['op']
        if op == OP_HELLO:
            return build_result(for_op=OP_HELLO, request_id=message['id'],
                                generation=generation, info=self.info)
        if op == OP_INIT:
            if self._on_init is not None:
                outcome = self._on_init(message)
                if isinstance(outcome, dict) and 'op' in outcome:
                    return dict(outcome, gen=generation)
                # A hook may replace any field, the echoed identity
                # included, so the defaults go in first.
                payload = {'fingerprint': message['fingerprint']}
                payload.update(outcome)
                return build_result(for_op=OP_INIT, request_id=message['id'],
                                    generation=generation, **payload)
            return build_result(
                for_op=OP_INIT, request_id=message['id'],
                generation=generation, fingerprint=message['fingerprint'],
                natom=message['natom'], triclinic=False,
                box=[0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                smoke_energy_raw=ENERGY_SCALE)
        if op == OP_EVALUATE:
            payload = self._evaluate_payload(message)
            if self._on_evaluate is not None:
                replaced = self._on_evaluate(message, len(self.operations))
                if replaced is not None:
                    payload = replaced
            if isinstance(payload, dict) and 'op' in payload:
                return dict(payload, gen=generation)
            return build_result(
                for_op=OP_EVALUATE, request_id=message['id'],
                generation=generation,
                **{**{'session': message['session'],
                      'fingerprint': message['fingerprint']}, **payload})
        if op == OP_CLOSE:
            return build_result(for_op=OP_CLOSE, request_id=message['id'],
                                generation=generation, closed=True)
        return None

    def _evaluate_payload(self, message) -> dict:
        """Correct-by-construction numbers for one geometry."""
        coordinates = message['coordinates']
        natom = len(coordinates)
        payload = {
            'step': message['step'],
            'natom': natom,
            'order': 'atom_id',
            'units': 'lammps-native',
            'energy_raw': ENERGY_SCALE * natom
            + sum(value for row in coordinates for value in row),
        }
        capabilities = (self.init_request or {}).get('capabilities', {})
        if capabilities.get('gradient'):
            payload['forces_raw'] = [[-value, -value * 2.0, -value * 3.0]
                                      for value in
                                      (row[0] for row in coordinates)]
        if capabilities.get('stress'):
            payload['pressure6_raw'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        return payload


def error_reply(request, *, kind='eval_failed', stage='run', usable=True,
                message='fake failure'):
    """Build an ERROR reply to *request* for a test to force."""
    return build_error(for_op=request['op'], request_id=request['id'],
                       generation=0, kind=kind, stage=stage,
                       session_usable=usable, message=message)
