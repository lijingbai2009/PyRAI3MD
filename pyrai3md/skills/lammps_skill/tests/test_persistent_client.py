"""
Unit tests for the backend side of a resident LAMMPS session.

The client is the layer that decides whether a reply may become a
framework result, so the interesting cases are all refusals: a reply that
answers a different request, carries a different session, is one shape
wrong, holds a NaN, arrives late or never arrives at all. Each of them
has to end with an exception and a poisoned session, and never with a
number, because the failure mode this protocol is designed against is a
calculation that quietly continues on stale or mismatched data.

A fake service (tests/_fake_lammps_service.py) speaks the real frames
over a real AF_UNIX socket; nothing here needs LAMMPS.

Run: pytest lammps_skill/tests/test_persistent_client.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import socket
import threading
import time

import numpy as np

import pytest

from ..service.persistent.client import PersistentSession
from ..service.persistent.protocol import (
    BOX_FIELDS,
    LammpsPersistentError,
    LammpsProtocolError,
    LammpsServiceDead,
    build_result)
from ._fake_lammps_service import FakeLammpsService, error_reply

COORDS = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
FINGERPRINT = 'f' * 64
SESSION = 's' * 64


def _service_factory(request_timeout=5.0, capabilities=None,
                     **fake_kwargs):
    """Start a fake service and return (service, session opener)."""

    service = FakeLammpsService(**fake_kwargs).start()

    def open_session(**overrides):
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stream.connect(service.endpoint)
        kwargs = dict(session=SESSION, fingerprint=FINGERPRINT,
                      natom=len(COORDS), order=[1, 2, 3], units='real',
                      capabilities=capabilities if capabilities is not None
                      else {'gradient': True, 'stress': True},
                      expected_world=1, request_timeout=request_timeout)
        kwargs.update(overrides)
        return PersistentSession(stream, **kwargs)

    return service, open_session


@pytest.fixture
def service():
    """A well-behaved fake service plus a client factory."""
    instance, opener = _service_factory()
    yield instance, opener
    instance.stop()


class TestHappyPath:
    """A healthy session asks, is answered, and reports the numbers."""

    def test_hello_checks_the_build_before_anything_is_loaded(self, service):
        session_client = _hello(*service)
        assert session_client.generation == 1

    def test_init_is_acked_with_the_loaded_state(self, service):
        session_client = _hello(*service)
        reply = session_client.initialize(
            cwd='/tmp/s', script='LAMMPS_force', files={'LAMMPS.data': 'a'},
            profile='ff_gradient', pbc=[1, 1, 1], step_tail=['run 0'])
        assert reply['natom'] == 3
        assert reply['fingerprint'] == FINGERPRINT
        assert session_client.generation == 2

    def test_evaluate_returns_native_numbers(self, service):
        session_client = _ready(*service)
        reply = session_client.evaluate(COORDS, None, 1)
        # The fake service evaluates a linear function of the geometry, so
        # a matching number proves the request arrived, not just a reply.
        # The fake evaluates a linear function: -2 per atom plus the sum
        # of every coordinate, i.e. -6 + 9 for this geometry.
        assert reply['energy_raw'] == pytest.approx(3.0)
        assert reply['forces_raw'] == [[-0.0, -0.0, -0.0],
                                       [-1.0, -2.0, -3.0],
                                       [-2.0, -4.0, -6.0]]
        assert reply['pressure6_raw'] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert reply['units'] == 'lammps-native'

    def test_a_new_geometry_gets_a_new_answer(self, service):
        session_client = _ready(*service)
        first = session_client.evaluate(COORDS, None, 1)['energy_raw']
        moved = [[value + 1.0 for value in row] for row in COORDS]
        second = session_client.evaluate(moved, None, 2)['energy_raw']
        assert first != second

    def test_box_travels_with_the_step(self, service):
        session_client = _ready(*service)
        box = {key: 5.0 for key in BOX_FIELDS}
        session_client.evaluate(COORDS, box, 1)
        assert service[0].operations[-1]['box'] == box

    def test_evaluate_accepts_numpy_geometry(self, service):
        # The framework passes coordinates and image flags as numpy
        # arrays, never as lists, and that is the only form a running MD
        # ever sends. A client-side `coordinates or ()`-style falsy test
        # on such an input raises "the truth value of an array with more
        # than one element is ambiguous" at evaluation 1 and takes the
        # whole example suite down with it -- which is exactly what the
        # 2026-09-28 22:05 remote run did. Tests that only ever send
        # lists cannot see this, so this one sends what the framework
        # actually sends.
        session_client = _ready(*service)
        response = session_client.evaluate(
            np.array(COORDS, dtype=float), None, 1,
            image=np.zeros((len(COORDS), 3), dtype=int))
        assert response['energy_raw']
        assert session_client._last_request['natom'] == len(COORDS)

    def test_timings_count_the_operations_that_finished(self, service):
        # The client half of the cost question: the framework's own wait,
        # separated from what the service says it spent. An operation that
        # never finished is not counted, because its time belongs to no
        # answer.
        session_client = _ready(*service)
        session_client.evaluate(COORDS, None, 1)
        session_client.evaluate(COORDS, None, 2)
        timings = session_client.timings()
        assert timings['EVALUATE']['calls'] == 2
        assert timings['INIT']['calls'] == 1
        for field in ('send_ms', 'wait_ms', 'parse_ms'):
            assert timings['EVALUATE'][field] >= 0.0


class TestRefusedReplies:
    """Anything that is not exactly right is an exception, not a number."""

    def test_reply_to_another_request_is_refused(self):
        def tamper(request, count):
            payload = {'step': request['step'], 'natom': 3,
                       'order': 'atom_id', 'units': 'lammps-native',
                       'energy_raw': -1.0, 'forces_raw': [[0.0] * 3] * 3,
                       'pressure6_raw': [0.0] * 6}
            # Answer the previous id: a stale reply must never be read as
            # the answer to this step.
            return _raw_result(request, payload, request_id=request['id'] - 1)

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(Exception, match='answers request'):
                session_client.evaluate(COORDS, None, 1)
        finally:
            service.stop()

    def test_generation_gap_is_refused(self):
        # The very first reply already skips a generation: the client has
        # nothing to align a stream against, so it refuses rather than
        # assuming the reply belongs to the request it just sent.
        service, opener = _service_factory(generation_offset=1)
        try:
            session_client = opener()
            # A reply whose generation is not the next one cannot be
            # matched to a request, so the session is dead, not merely
            # rejected: there is no safe way to keep reading it.
            with pytest.raises(LammpsServiceDead, match='generation'):
                session_client.hello()
        finally:
            service.stop()

    @pytest.mark.parametrize('field,value', [
        ('session', 'other' * 12),
        ('fingerprint', 'deadbeef'),
        ('step', 99),
        ('order', 'rank_local'),
        ('units', 'hartree'),
        ('natom', 2),
    ])
    def test_identity_and_shape_fields_are_required(self, field, value):
        def tamper(request, count):
            payload = {'step': request['step'], 'natom': 3,
                       'order': 'atom_id', 'units': 'lammps-native',
                       'energy_raw': -1.0, 'forces_raw': [[0.0] * 3] * 3,
                       'pressure6_raw': [0.0] * 6}
            payload[field] = value
            return payload

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsPersistentError):
                session_client.evaluate(COORDS, None, 1)
            assert session_client.poisoned
        finally:
            service.stop()

    def test_short_force_array_is_refused(self):
        def tamper(request, count):
            return {'step': request['step'], 'natom': 3,
                    'order': 'atom_id', 'units': 'lammps-native',
                    'energy_raw': -1.0,
                    'forces_raw': [[0.0, 0.0, 0.0]] * 2,
                    'pressure6_raw': [0.0] * 6}

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsPersistentError, match='shape'):
                session_client.evaluate(COORDS, None, 1)
        finally:
            service.stop()

    def test_missing_forces_are_refused_not_zero_filled(self):
        def tamper(request, count):
            payload = {'step': request['step'], 'natom': 3,
                       'order': 'atom_id', 'units': 'lammps-native',
                       'energy_raw': -1.0, 'pressure6_raw': [0.0] * 6}
            return payload

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsPersistentError, match='shape'):
                session_client.evaluate(COORDS, None, 1)
        finally:
            service.stop()

    def test_unrequested_forces_are_refused(self):
        def tamper(request, count):
            return {'step': request['step'], 'natom': 3,
                    'order': 'atom_id', 'units': 'lammps-native',
                    'energy_raw': -1.0, 'forces_raw': [[0.0] * 3] * 3}

        service, opener = _service_factory(
            capabilities={'gradient': False, 'stress': False},
            on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsPersistentError, match='never requested'):
                session_client.evaluate(COORDS, None, 1)
        finally:
            service.stop()

    @pytest.mark.parametrize('bad', [float('nan'), float('inf')])
    def test_non_finite_numbers_are_refused(self, bad):
        def tamper(request, count):
            return {'step': request['step'], 'natom': 3,
                    'order': 'atom_id', 'units': 'lammps-native',
                    'energy_raw': bad, 'forces_raw': [[0.0] * 3] * 3,
                    'pressure6_raw': [0.0] * 6}

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsPersistentError):
                session_client.evaluate(COORDS, None, 1)
        finally:
            service.stop()

    def test_smoke_energy_must_be_finite(self):
        def tamper(request):
            return {'natom': request['natom'], 'triclinic': False,
                    'box': [0.0] * 9, 'smoke_energy_raw': float('nan'),
                    'fingerprint': request['fingerprint']}

        service, opener = _service_factory(on_init=tamper)
        try:
            session_client = opener()
            session_client.hello()
            with pytest.raises(LammpsPersistentError, match='non-finite'):
                session_client.initialize(cwd='/tmp', script='x', files={},
                                          profile='ff_gradient',
                                          pbc=[0, 0, 0],
                                          step_tail=['run 0'])
        finally:
            service.stop()


class TestErrorReplies:
    """A usable refusal keeps the session; an unusable one ends it."""

    def test_recoverable_error_keeps_the_session_usable(self):
        state = {'failed': False}

        def tamper(request, count):
            if state['failed']:
                return None
            state['failed'] = True
            return error_reply(request, kind='eval_failed', stage='run',
                               usable=True, message='transient')

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsProtocolError, match='transient'):
                session_client.evaluate(COORDS, None, 1)
            assert not session_client.poisoned
            assert session_client.evaluate(COORDS, None, 2)['energy_raw']
        finally:
            service.stop()

    def test_dead_service_error_poisons_and_stops_talking(self):
        def tamper(request, count):
            return error_reply(request, kind='dead_service', stage='run',
                               usable=False, message='rank died')

        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(service, opener)
            with pytest.raises(LammpsServiceDead, match='rank died'):
                session_client.evaluate(COORDS, None, 1)
            seen = len(service.operations)
            with pytest.raises(LammpsServiceDead, match='rank died'):
                session_client.evaluate(COORDS, None, 2)
            # No further traffic: a dead session is not probed again.
            assert len(service.operations) == seen
        finally:
            service.stop()


class TestTimeoutsAndDisconnects:
    """A reply that never comes ends the session instead of being retried."""

    def test_timeout_poisons_without_retrying(self):
        service, opener = _service_factory(silence=True,
                                            request_timeout=0.5)
        try:
            session_client = opener()
            # A timeout ends the session rather than being reported as a
            # retryable condition: the client cannot know whether the
            # service consumed the operation, so re-sending it could
            # evaluate the step twice.
            with pytest.raises(LammpsServiceDead, match='timed out'):
                session_client.hello()
            seen = len(service.operations)
            with pytest.raises(LammpsServiceDead, match='timed out'):
                session_client.hello()
            assert len(service.operations) == seen
            assert 'not reused' in session_client.poisoned
        finally:
            service.stop()

    def test_peer_shutdown_is_reported_as_dead(self):
        service, opener = _service_factory()
        session_client = _ready(service, opener)
        service.stop()
        # The listener is gone; the live connection drains at the next
        # request, and the client must say 'died', not hand back a number.
        with pytest.raises(LammpsServiceDead):
            session_client.evaluate(COORDS, None, 1)
        assert service.init_request is not None


class TestCloseSemantics:
    """close() is safe to call repeatedly and closes the door behind it."""

    def test_repeated_close_sends_one_message(self, service):
        instance, opener = service
        session_client = _ready(instance, opener)
        session_client.close()
        session_client.close()
        session_client.close()
        closes = _wait_for(instance, lambda ops: [
            op for op in ops if op['op'] == 'CLOSE'])
        # Three close() calls, one message: the session is closed the
        # first time and every later call is a no-op.
        assert len(closes) == 1
        assert closes[0]['reason'] == 'owner_close'

    def test_evaluate_after_close_raises(self, service):
        instance, opener = service
        session_client = _ready(instance, opener)
        session_client.close()
        with pytest.raises(LammpsServiceDead, match='closed'):
            session_client.evaluate(COORDS, None, 1)

    def test_a_death_reports_the_step_it_died_on(self, tmp_path):
        # Eight NPT examples of the 2026-09-28 run died mid-trajectory
        # with "service closed the connection before a frame header" and
        # nothing else: not the step, not the box, not the line the
        # engine had written into its own log before it went.
        def tamper(request, count):
            return error_reply(request, kind='dead_service', stage='run',
                               usable=False, message='run 0 failed')

        log = tmp_path / 'lammps_resident.log'
        log.write_text('Step 450 total energy -76.1\n'
                       'ERROR: Pair system has no coefficients '
                       '(src/pair.cpp:249)\nAborting\n')
        service, opener = _service_factory(on_evaluate=tamper)
        try:
            session_client = _ready(
                service, lambda **kw: opener(log_path=str(log), **kw))
            with pytest.raises(LammpsServiceDead) as caught:
                session_client.evaluate(
                    COORDS, {'xlo': 0.0, 'xhi': 8.4, 'ylo': 0.0,
                             'yhi': 8.4, 'zlo': 0.0, 'zhi': 8.4,
                             'xy': 0.0, 'xz': 0.3, 'yz': 0.0}, 451)
            message = str(caught.value)
            assert 'run 0 failed' in message
            assert 'last request step=451 natom=3' in message
            assert 'Aborting' in message
            # The protocol text keeps its place: what the service said
            # first, then what the socket could not carry.
            assert (message.index('run 0 failed')
                    < message.index('last request'))
        finally:
            service.stop()

    def test_a_death_with_no_log_and_no_request_stays_terse(self, service):
        instance, opener = service
        session_client = _ready(instance, opener)
        session_client.close()
        with pytest.raises(LammpsServiceDead) as caught:
            session_client.evaluate(COORDS, None, 1)
        # The context is appended only when there is something to
        # append, so a failure that never reached a request or a log
        # does not acquire an empty parenthesis of its own.
        assert 'last request' not in str(caught.value)


class TestConcurrency:
    """One session serves one request at a time, however it is called."""

    def test_parallel_callers_are_serialized(self, service):
        instance, opener = service
        session_client = _ready(instance, opener)
        results, errors = [], []

        def worker(index):
            try:
                results.append(session_client.evaluate(
                    COORDS, None, index + 1)['energy_raw'])
            except Exception as exc:              # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,))
                   for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not errors, errors
        assert len(results) == 8
        steps = [operation['step'] for operation in instance.operations
                 if operation['op'] == 'EVALUATE']
        # Serialized, not interleaved: the service saw eight whole
        # requests and never two partial ones.
        assert sorted(steps) == list(range(1, 9))
        assert len({result for result in results}) <= 2


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _wait_for(service, predicate, timeout: float = 5.0):
    """Poll the fake service until *predicate* sees the operations."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = predicate(list(service.operations))
        if outcome:
            return outcome
        time.sleep(0.02)
    return predicate(list(service.operations))


def _hello(instance, opener):
    session_client = opener()
    session_client.hello()
    return session_client


def _ready(instance, opener):
    session_client = _hello(instance, opener)
    session_client.initialize(cwd='/tmp/s', script='LAMMPS_force',
                              files={'LAMMPS.data': 'a'},
                              profile='ff_gradient', pbc=[1, 1, 1],
                              step_tail=['run 0'])
    return session_client


def _raw_result(request, payload, request_id=None):
    """Build a RESULT the fake service will send verbatim."""
    message = build_result(for_op='EVALUATE',
                           request_id=request_id if request_id is not None
                           else request['id'],
                           generation=0, **payload)
    # The wrapper replaces the whole reply, so carry the frame fields the
    # test did not name.
    message.setdefault('session', request['session'])
    message.setdefault('fingerprint', request['fingerprint'])
    return message
