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

Resident (in-process LAMMPS module) execution support for the skill.

This package is split by which side of the OS process boundary it
belongs to:

  protocol   message framing, field lists, session identity, errors
  client     the backend's side of the socket
  launcher   starting and reaping the service process
  server     the service itself, the only module allowed to import
             lammps and mpi4py
  bringup    operator-run checks on a real LAMMPS deployment

Importing this package must therefore never touch a native library: the
framework process stays free of liblammps and of any MPI runtime, which
is the whole point of running LAMMPS in a separate process. The server
and the bringup tool are reachable only as module entry points, and
nothing here imports them.

"""

from .protocol import (
    CLOSE_TIMEOUT_S,
    ERROR_STAGES,
    IDLE_TIMEOUT_S,
    MAX_FRAME_BYTES,
    OP_CLOSE,
    OP_ERROR,
    OP_EVALUATE,
    OP_HELLO,
    OP_INIT,
    OP_RESULT,
    PROTOCOL_VERSION,
    REQUEST_TIMEOUT_S,
    STARTUP_TIMEOUT_S,
    LammpsPersistentError,
    LammpsPersistentUnsupported,
    LammpsProtocolError,
    LammpsServiceDead,
    LammpsServiceTimeout,
    encode_frame,
    read_frame,
)
from . import launcher

__all__ = [
    'launcher',
    'protocol',
    'PROTOCOL_VERSION',
    'MAX_FRAME_BYTES',
    'STARTUP_TIMEOUT_S',
    'REQUEST_TIMEOUT_S',
    'CLOSE_TIMEOUT_S',
    'IDLE_TIMEOUT_S',
    'OP_HELLO',
    'OP_INIT',
    'OP_EVALUATE',
    'OP_RESULT',
    'OP_ERROR',
    'OP_CLOSE',
    'ERROR_STAGES',
    'LammpsPersistentError',
    'LammpsPersistentUnsupported',
    'LammpsProtocolError',
    'LammpsServiceTimeout',
    'LammpsServiceDead',
    'encode_frame',
    'read_frame',
]
