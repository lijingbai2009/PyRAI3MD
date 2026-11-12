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

Wire format of the resident LAMMPS service.

Everything here is plain data: no socket object outlives a call, no
numpy, no LAMMPS, no MPI. Both sides of the pipe import this module, and
the service side imports nothing else from the skill but the input
builders, so a message can be checked without a framework present.

Framing is a 4-byte big-endian length followed by one UTF-8 JSON
object. JSON rather than pickle because the two interpreters are allowed
to differ in version, and a length prefix rather than newline-delimited
records because an input script or a comment may contain a newline.

The session identity (the fingerprint) is deliberately separate from the
force-field cache identity in service/builder/ff_model.py: that cache is
keyed on template content and may be shared by several backends, while a
session belongs to exactly one live service and also pins the loaded
script, the atom order and the capabilities. Merging the two would put
per-session state into a shareable file.

"""

import hashlib
import json
import socket
import struct
import time

PROTOCOL_VERSION = 2

# A request carries the whole coordinate array; 64 MiB is far above any
# system this skill evaluates per step and below the point where a
# corrupt length becomes an allocation attempt.
MAX_FRAME_BYTES = 64 * 1024 * 1024

_FRAME_HEADER = struct.Struct('>I')

# Timeouts are deployment policy, not per-calculation input, so they are
# constants rather than config keywords. STARTUP covers launching the
# interpreter and loading the force field; REQUEST covers one evaluate;
# CLOSE covers the coordinated shutdown of every rank; IDLE is how long
# the service waits without a client before it stops by itself.
STARTUP_TIMEOUT_S = 120.0
REQUEST_TIMEOUT_S = 600.0
CLOSE_TIMEOUT_S = 30.0
IDLE_TIMEOUT_S = 1800.0
OWNER_POLL_INTERVAL_S = 2.0

OP_HELLO = 'HELLO'
OP_INIT = 'INIT'
OP_EVALUATE = 'EVALUATE'
OP_RESULT = 'RESULT'
OP_ERROR = 'ERROR'
OP_CLOSE = 'CLOSE'

# Every failure reports which class it belongs to and where it happened.
# The kinds are stable names a log can be searched for; the stages match
# the steps of the service loop.
ERROR_KINDS = frozenset({
    'protocol',          # undecodable or malformed message
    'dependency',        # lammps/mpi4py/world not usable
    'unsupported_model', # cannot be served by the resident path
    'init_failed',       # loading the static package failed
    'eval_failed',       # a LAMMPS call or an extraction failed
    'nan_detected',      # the numbers are not finite
    'dead_service',      # the session cannot serve another request
    'owner_exit',        # the owner went away; the service is closing
    'internal',          # a defect in the service itself (never input)
})
ERROR_STAGES = frozenset({
    'frame', 'decode', 'validate', 'precollective', 'bcast', 'load',
    'box', 'scatter', 'run', 'extract', 'close',
})

# op -> (required fields, additional allowed fields). A field outside
# this table is a bug on one side of the pipe, and refusing it is
# cheaper than discovering that a renamed key silently became a default.
# Every accepted operation on the service side advances a counter that
# travels back as 'gen', and the client requires gen == its own + 1. That
# catches a reply to a request it never sent, a dropped reply and a
# re-ordered stream, which is all a two-sided sequence check has to do
# here: this protocol never retries, so a timeout ends the session
# instead of raising the question of whether the operation ran.
_MESSAGE_FIELDS = {
    OP_HELLO: (('id',), ()),
    OP_INIT: (('id', 'session', 'fingerprint', 'cwd', 'script', 'files',
               'profile', 'natom', 'order', 'units', 'pbc',
               'capabilities', 'step_tail'),
              # The kspace line travels with INIT because making the domain
              # triclinic invalidates it, exactly as the cold path's
              # re-issue after read_data does. `qmmm` names the computes a
              # QM/MM step reads and the QM atom ids to read them for.
              ('kspace', 'qmmm')),
    OP_EVALUATE: (('id', 'session', 'fingerprint', 'step',
                  'coordinates', 'box'),
                 # The crossing counts of this step's wrapped coordinates.
                 # Optional on the wire so a message from before the image
                 # channel still validates; a service reads its absence as
                 # "nothing has crossed", which is what such a run meant.
                 # q_qm carries the qmmm_energy per-step 0-order charges.
                 ('image', 'q_qm')),
    OP_RESULT: (('for', 'id', 'gen'),
                ('info', 'fingerprint', 'natom', 'triclinic', 'box',
                 'smoke_energy_raw', 'session', 'step', 'order', 'units',
                 'energy_raw', 'forces_raw', 'pressure6_raw',
                 'qmmm_raw', 'closed')),
    OP_ERROR: (('for', 'id', 'gen', 'kind', 'stage', 'session_usable',
                'message'),
               ('detail',)),
    OP_CLOSE: (('id', 'reason'), ()),
}

BOX_FIELDS = ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'xy', 'xz', 'yz')

# Below this, a tilt factor is arithmetic noise rather than a sheared box.
# A cell declared with three 90-degree angles reaches these numbers
# through cos(radians(90)) and carries tilts of order 1e-16, and a domain
# that is made tilt-capable for a tilt of zero is not the same calculation:
# PPPM re-frames and re-grids (measured in the test environment: an orthorhombic
# 10 A water box of 6 grid points per side became 8, and the energy moved
# from -20.6642 to -20.5581 kcal/mol). The writer that decides whether to
# emit a tilt line and the service that decides whether to convert its
# domain must answer that question the same way, so the threshold lives in
# the one module both sides import.
TILT_EPSILON = 1e-10


class LammpsPersistentError(RuntimeError):
    """Base of every resident-path failure.

    It derives from RuntimeError on purpose: Calculator.initialize()
    retries a backend with ``backend_cls(task_config)`` when the
    constructor raises TypeError, so a TypeError escaping here would
    rebuild the backend without its execution namespace and start a
    second service.
    """


class LammpsPersistentUnsupported(LammpsPersistentError):
    """The resident path cannot serve this request; the cold path can."""


class LammpsProtocolError(LammpsPersistentError):
    """A message could not be decoded, or did not match its schema."""


class LammpsServiceTimeout(LammpsPersistentError):
    """The service did not answer in time. The session is no longer usable.

    After a timeout the client cannot tell whether the service consumed
    the operation, so the reply stream can no longer be matched to the
    request stream. The session is poisoned rather than retried.
    """


class LammpsServiceDead(LammpsPersistentError):
    """The session died; every later call reports the first failure."""


def encode_frame(message: dict) -> bytes:
    """Serialize one message into a length-prefixed frame.

    Parameters
    ----------
    message : dict
        Plain-data message.

    Returns
    -------
    bytes
        Frame ready to be handed to sendall().

    Raises
    ------
    LammpsProtocolError
        When the payload exceeds MAX_FRAME_BYTES or is not JSON.
    """
    try:
        payload = json.dumps(message, ensure_ascii=False,
                             separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise LammpsProtocolError(f'message is not JSON-serializable: '
                                  f'{type(exc).__name__}: {exc}') from exc
    if len(payload) > MAX_FRAME_BYTES:
        raise LammpsProtocolError(
            f'message of {len(payload)} bytes exceeds the '
            f'{MAX_FRAME_BYTES} byte frame limit')
    return _FRAME_HEADER.pack(len(payload)) + payload


def _recv_exactly(stream, count: int, buffer: bytearray):
    """Fill the buffer with at least count bytes from a stream socket.

    Returns the number of bytes available, or None on a clean EOF.
    """
    while len(buffer) < count:
        try:
            chunk = stream.recv(max(4096, count - len(buffer)))
        except socket.timeout:
            raise
        except OSError as exc:
            raise LammpsServiceDead(f'socket read failed: {exc}') from exc
        if not chunk:
            return None
        buffer.extend(chunk)
    return len(buffer)


def read_frame(stream, deadline: float = None) -> dict:
    """Read one frame from a stream socket.

    Parameters
    ----------
    stream : socket.socket
        Connected stream socket; a socket.timeout on it is translated
        into LammpsServiceTimeout once the deadline passed.
    deadline : float, optional
        Absolute time.monotonic() instant. Without it the call relies on
        the socket timeout alone.

    Returns
    -------
    dict
        Decoded message.

    Raises
    ------
    LammpsProtocolError
        Oversize, malformed or truncated frame.
    LammpsServiceTimeout
        No complete frame before the deadline.
    LammpsServiceDead
        The peer closed the connection.
    """
    buffer = bytearray()
    header_size = _FRAME_HEADER.size
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise LammpsServiceTimeout(
                'no complete frame before the request deadline')
        try:
            available = _recv_exactly(stream, header_size, buffer)
        except socket.timeout as exc:
            if deadline is None:
                raise LammpsServiceTimeout('socket read timed out') from exc
            continue
        if available is None:
            raise LammpsServiceDead(
                'service closed the connection before a frame header')
        length = _FRAME_HEADER.unpack(bytes(buffer[:header_size]))[0]
        if length > MAX_FRAME_BYTES:
            raise LammpsProtocolError(
                f'incoming frame claims {length} bytes, over the '
                f'{MAX_FRAME_BYTES} byte limit')
        try:
            remaining = _recv_exactly(stream, header_size + length, buffer)
        except socket.timeout as exc:
            if deadline is None:
                raise LammpsServiceTimeout('socket read timed out') from exc
            continue
        if remaining is None:
            raise LammpsServiceDead('service closed the connection mid-frame')
        if len(buffer) < header_size + length:
            continue
        body = bytes(buffer[header_size:header_size + length])
        del buffer[:header_size + length]
        try:
            return json.loads(body.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise LammpsProtocolError(
                f'frame is not a JSON object: {exc}') from exc


def check_fields(message, expect: str = None):
    """Validate a message against its field list.

    Parameters
    ----------
    message : dict
        Decoded message.
    expect : str, optional
        Operation the message must carry.

    Returns
    -------
    str
        The operation name.

    Raises
    ------
    LammpsProtocolError
        Not a mapping, wrong or unknown version, unknown or missing
        field.
    """
    if not isinstance(message, dict):
        raise LammpsProtocolError(
            f'message is a {type(message).__name__}, expected an object')
    version = message.get('v')
    if version != PROTOCOL_VERSION:
        raise LammpsProtocolError(
            f'unsupported protocol version {version!r}, this build '
            f'speaks {PROTOCOL_VERSION}')
    op = message.get('op')
    fields = _MESSAGE_FIELDS.get(op)
    if fields is None:
        raise LammpsProtocolError(f'unknown operation {op!r}')
    if expect is not None and op != expect:
        raise LammpsProtocolError(
            f'expected an {expect} message, got {op}')
    required, allowed = fields
    missing = [name for name in required if name not in message]
    if missing:
        raise LammpsProtocolError(
            f'{op} message lacks field(s): {", ".join(missing)}')
    unknown = sorted(set(message) - set(required) - set(allowed)
                     - {'v', 'op'})
    if unknown:
        raise LammpsProtocolError(
            f'{op} message carries unknown field(s): {", ".join(unknown)}')
    return op


def _new(envelope: dict) -> dict:
    """Return a copy of *envelope* carrying the protocol version."""
    message = {'v': PROTOCOL_VERSION}
    message.update(envelope)
    return message


def build_hello(request_id: int) -> dict:
    """Build the HELLO request."""
    return _new({'op': OP_HELLO, 'id': int(request_id)})


def build_init(*, request_id: int, session: str, fingerprint: str, cwd: str,
               script: str, files: dict, profile: str, natom: int,
               order: list, units: str, pbc: list, capabilities: dict,
               step_tail: list, kspace: str = '', qmmm: dict = None):
    """Build the INIT request that loads one resident instance.

    Parameters
    ----------
    files : dict
        ``{name: sha256}`` of every file of the static package.
    step_tail : list of str
        The command lines that evaluate one step, issued after the box and
        the coordinates have been applied. The loaded script deliberately
        contains no run of its own, so without this the instance would
        never evaluate the geometry of the step that asked for it.
    capabilities : dict
        ``{'gradient': bool, 'stress': bool}`` requested extractions.
    """
    message = _new({
        'op': OP_INIT, 'id': int(request_id), 'session': session,
        'fingerprint': fingerprint, 'cwd': cwd, 'script': script,
        'files': dict(files), 'profile': profile, 'natom': int(natom),
        'order': [int(tag) for tag in order], 'units': units,
        'pbc': [int(flag) for flag in pbc], 'capabilities': dict(capabilities),
        'step_tail': [str(line) for line in step_tail],
        'kspace': str(kspace or ''),
    })
    if qmmm is not None:
        message['qmmm'] = dict(qmmm)
    return message


def build_evaluate(*, request_id: int, session: str,
                   fingerprint: str, step: int, coordinates: list,
                   box: dict, image: list = None, q_qm: list = None):
    """Build one EVALUATE request.

    Parameters
    ----------
    coordinates : list of list of float
        (natom, 3) framework-order positions in Angstrom, wrapped into the
        main image exactly as the framework stores them.
    box : dict or None
        Box numbers from input_builder.resolve_live_box, in the frame
        a data file states them, or None to
        keep the loaded box (a non-periodic run).
    image : list of list of int, optional
        (natom, 3) count of cell lengths each atom crossed, from the same
        evaluation snapshot as the coordinates. The service has to set
        these with the positions: the live instance still holds the flags
        of the previous step, and a new wrapped coordinate written on a
        stale flag moves the atom's unwrapped position by a whole box.
        None reads as "no atom has crossed", which is what a run before
        the image channel meant.

    Raises
    ------
    LammpsProtocolError
        The image rows do not cover the atoms of this step.
    """
    payload = set(BOX_FIELDS)
    if box is not None and set(box) != payload:
        raise LammpsProtocolError(
            f'box must carry exactly {sorted(payload)}, got {sorted(box)}')
    # `is not None`, not a falsy test: the framework sends the image
    # flags as a numpy array, and `array or []` on one raises "the truth
    # value of an array with more than one element is ambiguous".
    rows = [[int(value) for value in row]
            for row in (image if image is not None else [])]
    if rows and len(rows) != len(coordinates):
        raise LammpsProtocolError(
            f'image describes {len(rows)} atoms, the coordinates describe '
            f'{len(coordinates)}')
    if any(len(row) != 3 for row in rows):
        raise LammpsProtocolError('every image row holds three counts')
    message = {
        'op': OP_EVALUATE, 'id': int(request_id), 'session': session,
        'fingerprint': fingerprint, 'step': int(step),
        'coordinates': [[float(v) for v in row] for row in coordinates],
        'box': None if box is None else {key: float(box[key])
                                         for key in BOX_FIELDS},
        'image': rows,
    }
    if q_qm is not None:
        values = [float(value) for value in q_qm]
        if any(value != value or value in (float('inf'), float('-inf'))
               for value in values):
            raise LammpsProtocolError('q_qm contains a non-finite value')
        message['q_qm'] = values
    return _new(message)


def build_close(*, request_id: int, reason: str) -> dict:
    """Build the CLOSE request that releases the instance."""
    return _new({'op': OP_CLOSE, 'id': int(request_id), 'reason': reason})


def build_result(*, for_op: str, request_id: int, generation: int, **payload):
    """Build a RESULT reply carrying *payload*."""
    message = {'op': OP_RESULT, 'for': for_op, 'id': int(request_id),
               'gen': int(generation)}
    message.update(payload)
    return _new(message)


def build_error(*, for_op: str, request_id: int, generation: int, kind: str,
                stage: str, session_usable: bool, message: str,
                detail: dict = None):
    """Build an ERROR reply.

    Raises
    ------
    LammpsProtocolError
        When kind or stage is outside the agreed vocabulary, which would
        leave the client guessing how to classify the failure.
    """
    if kind not in ERROR_KINDS:
        raise LammpsProtocolError(f'unknown error kind {kind!r}')
    if stage not in ERROR_STAGES:
        raise LammpsProtocolError(f'unknown error stage {stage!r}')
    payload = {'op': OP_ERROR, 'for': for_op, 'id': int(request_id),
               'gen': int(generation), 'kind': kind, 'stage': stage,
               'session_usable': bool(session_usable),
               'message': str(message)}
    if detail:
        payload['detail'] = dict(detail)
    return _new(payload)


def compute_fingerprint(*, model: dict, init_files: dict, script: str,
                        data: str, order: list, box_lines: list, units: str,
                        pbc: list, profile: str, stress: bool,
                        needs_gradient: bool, shake: dict, nproc: int,
                        step_tail: list) -> str:
    """Hash everything a loaded instance is, into one session identity.

    The client sends it with INIT and repeats it with every EVALUATE; the
    service recomputes nothing but compares, so a stale cache, a template
    edited between steps, a swapped project or a changed rank count is a
    refused request rather than a silently different calculation.

    Parameters
    ----------
    model : dict
        Compressed force-field model (its own content hashes are enough,
        so only the digests, the counts and the selection are folded in).
    init_files : dict
        ``{name: sha256}`` of the static package.
    shake : dict
        ``model['constraints']``: the constraint decision is part of the
        identity, because it changes what the instance actually applies.
    step_tail : list of str
        The per-step command sequence. It carries the ``debug`` decision
        (whether a step also writes the subprocess path's files), and a
        session must not be able to change that half way through.

    Returns
    -------
    str
        Hex sha256 digest.
    """
    payload = {
        'protocol': PROTOCOL_VERSION,
        'model_format': model.get('format'),
        'model_source_hash': model.get('source_hash'),
        'model_config_hash': model.get('config_hash'),
        'model_counts': model.get('counts'),
        # Two centers of the same size and the same atom types have identical
        # counts and an identical type map, so the selection is the only field
        # that says which subsystem this session is. Naming it here turns
        # 'a swapped selection' into a refused step by name.
        'model_selection': [int(tag) for tag in (model.get('selection')
                                                 or [])],
        'model_type_map': model.get('type_map'),
        'init_files': init_files,
        'script': script,
        'data_digest': _digest_text(data),
        'order': [int(tag) for tag in order],
        'box_lines': [str(line) for line in box_lines],
        'units': units,
        'pbc': [int(flag) for flag in pbc],
        'profile': profile,
        'stress': bool(stress),
        'needs_gradient': bool(needs_gradient),
        'shake': shake,
        'nproc': int(nproc),
        'step_tail': [str(line) for line in step_tail],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _digest_text(text: str) -> str:
    """Hash one template body."""
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()
