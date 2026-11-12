"""
Unit tests for the resident-service wire protocol.

The protocol module is the only code both sides of the process boundary
share, so its job is not to be convenient but to refuse: an unknown
field, a stale version or a truncated frame must fail here rather than
become a wrong number several layers up. These tests are therefore
mostly negative controls, and every fingerprint component gets its own
one, because a session identity that ignores one input silently accepts
the mismatch it was meant to catch.

These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_persistent_protocol.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import socket
import struct

import pytest

from ..service.persistent.protocol import (
    BOX_FIELDS,
    MAX_FRAME_BYTES,
    OP_CLOSE,
    OP_EVALUATE,
    OP_HELLO,
    OP_INIT,
    OP_RESULT,
    PROTOCOL_VERSION,
    LammpsProtocolError,
    LammpsServiceDead,
    LammpsServiceTimeout,
    build_close,
    build_error,
    build_evaluate,
    build_hello,
    build_init,
    build_result,
    check_fields,
    compute_fingerprint,
    encode_frame,
    read_frame)

SESSION = 's' * 64
FINGERPRINT = 'f' * 64


class TestFraming:
    """A frame is a length plus that many bytes of JSON, nothing else."""

    def test_round_trip(self):
        message = build_hello(7)
        assert read_frame(_Stream(encode_frame(message))) == message

    def test_length_prefix_is_big_endian_unsigned(self):
        frame = encode_frame(build_hello(1))
        payload = frame[4:]
        assert struct.unpack('>I', frame[:4])[0] == len(payload)

    def test_non_serializable_message_is_rejected(self):
        with pytest.raises(LammpsProtocolError, match='JSON'):
            encode_frame({'v': 1, 'op': OP_HELLO, 'id': object()})

    def test_oversize_message_is_rejected_before_sending(self):
        with pytest.raises(LammpsProtocolError, match='exceeds'):
            encode_frame({'v': 1, 'op': OP_EVALUATE, 'blob': 'x' * 100,
                          'pad': 'y' * (MAX_FRAME_BYTES + 10)})

    def test_oversize_incoming_length_is_refused_before_allocating(self):
        frame = struct.pack('>I', MAX_FRAME_BYTES + 1) + b'{}'
        with pytest.raises(LammpsProtocolError, match='over the'):
            read_frame(_Stream(frame))

    def test_truncated_header_is_a_dead_service(self):
        with pytest.raises(LammpsServiceDead, match='frame header'):
            read_frame(_Stream(b'\x00\x00'))

    def test_truncated_body_is_a_dead_service(self):
        frame = encode_frame(build_hello(1))[:-2]
        with pytest.raises(LammpsServiceDead, match='mid-frame'):
            read_frame(_Stream(frame))

    def test_non_json_body_is_a_protocol_error(self):
        body = b'nonsense'
        frame = struct.pack('>I', len(body)) + body
        with pytest.raises(LammpsProtocolError, match='not a JSON'):
            read_frame(_Stream(frame))

    def test_frame_split_across_reads_is_reassembled(self):
        # A stream that yields one byte per recv() must still deliver the
        # message: the framing cannot assume a single read returns a whole
        # frame, because AF_UNIX does not guarantee it.
        frame = encode_frame(build_hello(3))
        assert read_frame(_Chunked(frame, chunk=1)) == build_hello(3)

    def test_expired_deadline_is_a_timeout(self):
        with pytest.raises(LammpsServiceTimeout):
            read_frame(_Stream(b'', timeout_immediately=True), deadline=0.0)

    def test_real_socket_pair_round_trip(self):
        left, right = socket.socketpair()
        try:
            left.sendall(encode_frame(build_close(request_id=1,
                                                   reason='test')))
            assert read_frame(right)['reason'] == 'test'
        finally:
            left.close()
            right.close()


class TestFieldContract:
    """Only the agreed fields, in the agreed operations, may pass."""

    def test_hello_and_close_field_sets(self):
        assert check_fields(build_hello(1), OP_HELLO) == OP_HELLO
        assert check_fields(build_close(request_id=2, reason='x'),
                            OP_CLOSE) == OP_CLOSE

    def test_init_carries_everything_a_load_needs(self):
        message = _init_message()
        assert set(message) - {'v'} == {
            'op', 'id', 'session', 'fingerprint', 'cwd', 'script', 'files',
            'profile', 'natom', 'order', 'units', 'pbc', 'capabilities',
            # The per-step evaluation sequence: the loaded script stops
            # before it, so INIT is the only place it can arrive.
            'step_tail',
            # Optional: a project without a kspace_style simply sends ''.
            'kspace'}

    def test_evaluate_ships_only_dynamic_data(self):
        message = build_evaluate(request_id=1, session=SESSION,
                                 fingerprint=FINGERPRINT, step=1,
                                 coordinates=[[0.0, 0.0, 0.0]],
                                 box={key: 1.0 for key in BOX_FIELDS},
                                 image=[[1, 0, -1]])
        assert set(message) == {'v', 'op', 'id', 'session', 'fingerprint',
                                 'step', 'coordinates', 'box', 'image'}
        # The image counts belong here because they belong to the
        # coordinates: they say which copy of the cell this step's wrapped
        # positions mean, so they can only travel with them.
        assert message['image'] == [[1, 0, -1]]
        # Charges, topology and force-field numbers have no place here: a
        # step cannot change what the instance was loaded with.
        for forbidden in ('charges', 'coordinates_backup', 'coeffs',
                          'atoms', 'units'):
            assert forbidden not in message

    def test_evaluate_without_image_counts_sends_none(self):
        # Absence is a defined reading ("nothing has crossed"), not an
        # error, so a client from before the channel stays usable.
        message = build_evaluate(request_id=1, session=SESSION,
                                  fingerprint=FINGERPRINT, step=1,
                                  coordinates=[[0.0, 0.0, 0.0]],
                                  box={key: 1.0 for key in BOX_FIELDS})
        assert message['image'] == []

    def test_evaluate_refuses_counts_for_another_atom_count(self):
        with pytest.raises(LammpsProtocolError, match='describe'):
            build_evaluate(request_id=1, session=SESSION,
                           fingerprint=FINGERPRINT, step=1,
                           coordinates=[[0.0] * 3, [1.0] * 3],
                           box={key: 1.0 for key in BOX_FIELDS},
                           image=[[0, 0, 0]])

    def test_evaluate_refuses_a_malformed_image_row(self):
        with pytest.raises(LammpsProtocolError, match='three'):
            build_evaluate(request_id=1, session=SESSION,
                           fingerprint=FINGERPRINT, step=1,
                           coordinates=[[0.0] * 3],
                           box={key: 1.0 for key in BOX_FIELDS},
                           image=[[0, 0]])

    def test_wrong_box_fields_is_rejected(self):
        with pytest.raises(LammpsProtocolError, match='box must carry'):
            build_evaluate(request_id=1, session=SESSION,
                           fingerprint=FINGERPRINT, step=1,
                           coordinates=[[0.0, 0.0, 0.0]],
                           box={'xlo': 0.0, 'xhi': 1.0})

    def test_unknown_operation_is_refused(self):
        with pytest.raises(LammpsProtocolError, match='unknown operation'):
            check_fields({'v': PROTOCOL_VERSION, 'op': 'COMMAND', 'id': 1})

    def test_unknown_field_is_refused(self):
        message = build_hello(1)
        message['extra'] = 1
        with pytest.raises(LammpsProtocolError, match='unknown field'):
            check_fields(message, OP_HELLO)

    def test_missing_field_is_named(self):
        message = build_hello(1)
        del message['id']
        with pytest.raises(LammpsProtocolError, match='lacks field'):
            check_fields(message, OP_HELLO)

    def test_wrong_version_is_refused(self):
        with pytest.raises(LammpsProtocolError, match='protocol version'):
            check_fields({'v': PROTOCOL_VERSION + 1, 'op': OP_HELLO,
                          'id': 1})

    def test_wrong_operation_for_the_caller_is_refused(self):
        with pytest.raises(LammpsProtocolError, match='expected an'):
            check_fields(build_hello(1), OP_INIT)

    def test_non_mapping_is_refused(self):
        with pytest.raises(LammpsProtocolError, match='expected an object'):
            check_fields([1, 2])

    def test_error_vocabulary_is_closed(self):
        with pytest.raises(LammpsProtocolError, match='unknown error kind'):
            build_error(for_op=OP_INIT, request_id=1, generation=1,
                        kind='oops', stage='run', session_usable=False,
                        message='x')
        with pytest.raises(LammpsProtocolError, match='unknown error stage'):
            build_error(for_op=OP_INIT, request_id=1, generation=1,
                        kind='protocol', stage='nowhere',
                        session_usable=False, message='x')

    def test_result_replies_are_checkable(self):
        reply = build_result(for_op=OP_INIT, request_id=4, generation=2,
                             natom=3, triclinic=False, box=[0.0] * 9,
                             smoke_energy_raw=-1.0,
                             fingerprint=FINGERPRINT)
        assert check_fields(reply, OP_RESULT) == OP_RESULT


class TestFingerprint:
    """Every input changes the identity; a blind spot would be a bug."""

    BASE = None

    @classmethod
    def setup_class(cls):
        cls.BASE = _fingerprint_args()

    def _fingerprint(self, **overrides):
        arguments = dict(self.BASE)
        arguments.update(overrides)
        return compute_fingerprint(**arguments)

    def test_is_stable_and_hex(self):
        assert self._fingerprint() == self._fingerprint()
        assert len(self._fingerprint()) == 64

    def test_key_order_does_not_change_the_identity(self):
        # The digest is taken over canonically sorted JSON, so a dict that
        # merely got built in another order is the same session.
        model_a = {'format': 3, 'source_hash': {'data': 'abc'},
                   'config_hash': 'c', 'counts': {'natom': 6},
                   'type_map': {'atom': {'31': 1}}, 'box': [],
                   'constraints': {}}
        model_b = dict(reversed(list(model_a.items())))
        assert self._fingerprint(model=model_a) \
            == self._fingerprint(model=model_b)

    @pytest.mark.parametrize('field,value', [
        ('model', {'format': 3, 'source_hash': {'data': 'other'},
                   'config_hash': 'c', 'counts': {'natom': 6},
                   'type_map': {}, 'box': [], 'constraints': {}}),
        ('init_files', {'LAMMPS.data': 'zz'}),
        ('script', 'other_force'),
        ('data', 'different data'),
        ('order', [2, 1]),
        ('box_lines', ['0.0 9.0 xlo xhi', '0.0 10.0 ylo yhi',
                       '0.0 10.0 zlo zhi']),
        ('units', 'metal'),
        ('pbc', [1, 1, 0]),
        ('profile', 'ff_energy'),
        ('stress', True),
        ('needs_gradient', False),
        ('shake', {'enabled': True, 'engine': 'framework'}),
        ('nproc', 2),
        ('step_tail', ['run 0', 'write_dump all custom x.xyz fx fy fz']),
    ])
    def test_each_component_matters(self, field, value):
        assert self._fingerprint(**{field: value}) != self._fingerprint()

    def test_protocol_version_matters(self):
        from ..service.persistent import protocol

        before = self._fingerprint()
        protocol.PROTOCOL_VERSION += 1
        try:
            assert self._fingerprint() != before
        finally:
            protocol.PROTOCOL_VERSION -= 1


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _Stream:
    """Minimal stand-in for a connected socket."""

    def __init__(self, data: bytes, timeout_immediately: bool = False):
        self._data = data
        self._timeout = timeout_immediately

    def recv(self, count: int) -> bytes:
        if self._timeout:
            raise socket.timeout()
        chunk, self._data = self._data[:count], self._data[count:]
        return chunk


class _Chunked(_Stream):
    """A stream that hands over one byte at a time."""

    def __init__(self, data: bytes, chunk: int = 1):
        super().__init__(data)
        self._chunk = chunk

    def recv(self, count: int) -> bytes:
        return super().recv(min(count, self._chunk))


def _init_message():
    return build_init(request_id=1, session=SESSION, fingerprint=FINGERPRINT,
                      cwd='/tmp/s', script='LAMMPS_force',
                      files={'LAMMPS.data': 'a'}, profile='ff_gradient',
                      natom=6, order=[1, 2, 3, 4, 5, 6], units='real',
                      pbc=[1, 1, 1],
                      capabilities={'gradient': True, 'stress': False},
                      step_tail=['run 0'])


def _fingerprint_args():
    model = {'format': 3, 'source_hash': {'data': 'abc'},
             'config_hash': 'c', 'counts': {'natom': 6},
             'type_map': {'atom': {'31': 1}}, 'box': [], 'constraints': {}}
    return {
        'model': model,
        'init_files': {'LAMMPS.data': 'a'},
        'script': 'LAMMPS_force',
        'data': 'LAMMPS Description',
        'order': [1, 2, 3],
        'box_lines': ['0.0 10.0 xlo xhi', '0.0 10.0 ylo yhi',
                      '0.0 10.0 zlo zhi'],
        'units': 'real',
        'pbc': [1, 1, 1],
        'profile': 'ff_gradient',
        'stress': False,
        'needs_gradient': True,
        'shake': {'enabled': False, 'engine': 'framework'},
        'nproc': 1,
        'step_tail': ['run 0'],
    }
