"""
======================================================================
The resident LAMMPS service's error boundaries (#5.5).

Three defects used to wear the same mask: a frame that could not be
decoded, a validator that raised, and a socket that went quiet all
ended as "nothing arrived", so each was reported as an idle timeout a
few seconds later (or not at all). The reader now tells them apart:

- a timeout consults the one owner/idle rule and keeps waiting;
- a closed peer stops the session as `client_closed`;
- an undecodable frame answers a protocol ERROR and stops this
  connection;
- a validator that raises anything but its own refusal is reported as
  `internal` and the connection stops, because that validator cannot be
  trusted with the next message.

`close()` prints a failure instead of swallowing it. The owner/idle rule
is one function, shared by the accept loop and the frame loop.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import time
import unittest
from types import SimpleNamespace

from ..service.persistent import protocol, server
from ..service.persistent.protocol import (
    LammpsServiceDead, LammpsServiceTimeout)


def _args():
    return SimpleNamespace(owner_pid=0,
                           idle_deadline=time.monotonic() + 60.0)


class TestTheFrameLoopTellsTheCausesApart(unittest.TestCase):
    """One injected defect per cause, one distinct answer."""

    def _run(self, read_frame, check_fields=None):
        original_read = server.read_frame
        original_check = server.check_fields
        server.read_frame = read_frame
        if check_fields is not None:
            server.check_fields = check_fields
        try:
            return server._read_operation(object(), _args())
        finally:
            server.read_frame = original_read
            server.check_fields = original_check

    def test_an_undecodable_frame_closes_with_a_protocol_error(self):
        def broken(conn, deadline=None):
            raise protocol.LammpsProtocolError('bad header')

        op, reply = self._run(broken)
        self.assertEqual(op['op'], protocol.OP_CLOSE)
        self.assertEqual(op['reason'], 'protocol')
        self.assertEqual(reply['kind'], 'protocol')
        self.assertFalse(reply['session_usable'])

    def test_a_closed_peer_stops_as_client_closed(self):
        def closed(conn, deadline=None):
            raise LammpsServiceDead('peer went away')

        op, reply = self._run(closed)
        self.assertEqual(op['reason'], 'client_closed')
        self.assertIsNone(reply)

    def test_a_timeout_past_the_idle_deadline_stops_as_idle(self):
        def quiet(conn, deadline=None):
            raise LammpsServiceTimeout('nothing yet')

        args = _args()
        args.idle_deadline = time.monotonic() - 1.0
        original = server.read_frame
        server.read_frame = quiet
        try:
            op, reply = server._read_operation(object(), args)
        finally:
            server.read_frame = original
        self.assertEqual(op['reason'], 'idle_exit')
        self.assertIsNone(reply)

    def test_a_defect_inside_the_reader_is_not_an_idle_timeout(self):
        # The AttributeError is not one of the protocol's own failures, so
        # it must leave this function at once instead of idling for three
        # seconds and reporting a quiet socket.
        def defect(conn, deadline=None):
            raise AttributeError('frame helper is gone')

        with self.assertRaises(AttributeError):
            self._run(defect)

    def test_a_broken_validator_is_internal_and_closes(self):
        def defect(message):
            raise AttributeError('field table is gone')

        def good(conn, deadline=None):
            return {'op': protocol.OP_EVALUATE, 'id': 7, 'v': 1}

        op, reply = self._run(good, check_fields=defect)
        self.assertEqual(op['reason'], 'internal')
        self.assertEqual(reply['kind'], 'internal')
        self.assertFalse(reply['session_usable'])

    def test_the_validators_own_refusal_is_a_protocol_error(self):
        def refusal(message):
            raise protocol.LammpsProtocolError('unknown field: x')

        def good(conn, deadline=None):
            return {'op': protocol.OP_EVALUATE, 'id': 7, 'v': 1}

        op, reply = self._run(good, check_fields=refusal)
        self.assertIsNone(op)
        self.assertEqual(reply['kind'], 'protocol')
        self.assertTrue(reply['session_usable'])


class TestOneWaitRule(unittest.TestCase):
    """Owner and idle are read in one place."""

    def test_the_accept_loop_and_the_frame_loop_share_it(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parents[1]
                / 'service/persistent/server.py').read_text()
        # One definition, two waits (the accept loop and the frame loop).
        self.assertEqual(text.count('def _stop_reason('), 1)
        self.assertEqual(text.count('_stop_reason(args)'), 3)
        self.assertNotIn('def close(self) -> None:\n        """Release the '
                         'instance without terminating MPI."""', text)


if __name__ == '__main__':
    unittest.main()