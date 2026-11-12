"""
End-to-end tests of the resident service loop, against a stub LAMMPS.

service/persistent/server.py is the one module that imports LAMMPS, so it
cannot be exercised on a machine without a LAMMPS build. These tests still
cover what is this skill's own responsibility rather than LAMMPS's: the
request loop, the exact operation sequence the service sends to LAMMPS,
the identity checks, the box read-back gate and the owner watchdog. They
run the real server in a child interpreter whose 'lammps' module is a stub
that records every call it receives.

What is NOT verified here is the numerical behaviour of real LAMMPS calls
(scatter/gather/compute semantics on a decomposed system). That belongs to
service/persistent/bringup.py on the test environment.

Run: pytest lammps_skill/tests/test_persistent_server.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from ..service.persistent import launcher
from ..service.persistent.client import PersistentSession
from ..service.persistent.protocol import (
    LammpsPersistentError,
    LammpsServiceDead,
)

# The stub records every call into LMP_STUB_LOG and answers with numbers
# derived from whatever geometry was last scattered, so a service that
# ignored the request cannot pass these tests by luck.
_RECORDER_SOURCE = r'''
import json
import os

_PATH = os.environ['LMP_STUB_LOG']


def log(entry):
    with open(_PATH, 'a') as handle:
        handle.write(json.dumps(entry) + '\n')
'''

_STUB_SOURCE = r'''
import os

import ctypes
import re
import lammps_recorder as rec


def _stub_ids(spec, natom):
    """Expand a set-atom selection the way utils::bounds does."""
    if spec == '*':
        return range(1, natom + 1)
    if '*' not in spec:
        return [int(spec)]
    low, high = (int(value) for value in spec.split('*'))
    if low > high:
        raise AssertionError('set atom range %s counts down' % spec)
    return range(low, high + 1)


class lammps(object):

    def __init__(self, cmdargs=None, comm=None):
        self.natom = int(os.environ.get('LMP_STUB_NATOM', '3'))
        self.box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        self.x = [0.0] * (3 * self.natom)
        self.triclinic = False
        self.mode = os.environ.get('LMP_STUB_BOX', 'ok')
        rec.log({'op': 'create', 'comm': comm is not None})

    def file(self, name):
        # A real LAMMPS runs the script it is handed; the stub replays its
        # command lines, so an INIT package that lost 'run 0' or the
        # pressure compute cannot pass these tests either.
        rec.log({'op': 'file', 'name': str(name)})
        with open(str(name)) as handle:
            for line in handle:
                line = line.split('#')[0].strip()
                if line:
                    self.command(line)

    def command(self, text):
        text = ' '.join(str(text).split())
        rec.log({'op': 'command', 'text': text})
        # The grammar LAMMPS actually accepts: every dimension carries its
        # own keyword, so the bounds and the tilt factors arrive as two
        # commands. Anything else is an illegal change_box command, which
        # the real engine answers by aborting the process group; here it
        # leaves the box untouched, and the read-back check fails.
        bounds = re.fullmatch(
            r'change_box all x final (-?[\d.]+) (-?[\d.]+)'
            r' y final (-?[\d.]+) (-?[\d.]+)'
            r' z final (-?[\d.]+) (-?[\d.]+)', text)
        tilts = re.fullmatch(
            r'change_box all xy final (-?[\d.]+)'
            r' xz final (-?[\d.]+)'
            r' yz final (-?[\d.]+)', text)
        if bounds:
            numbers = [float(value) for value in bounds.groups()]
            if self.mode == 'wrong':
                numbers = [value + 1.0 for value in numbers]
            self.box = numbers + self.box[6:]
        elif tilts:
            if not self.triclinic:
                raise AssertionError(
                    'cannot change tilt factors of a non-triclinic box')
            values = [float(value) for value in tilts.groups()]
            if self.mode == 'wrong':
                values = [value + 1.0 for value in values]
            self.box = self.box[:6] + values
        elif text == 'change_box all triclinic':
            self.triclinic = True
        elif text.startswith('change_box all final'):
            raise AssertionError(
                'change_box all final <9 numbers> is not a LAMMPS command')
        elif text.startswith('set '):
            # The image flags of a step travel as 'set atom <spec> image
            # <nx> <ny> <nz>'. spec is one ID or a lo*hi range, never a
            # list: set.cpp parses it with utils::bounds as a single token,
            # so every other part of a space-separated list is read as a
            # keyword and aborts the job. The stub reproduces that, because
            # a grammar invented here is a grammar the machine rejects.
            parts = text.split()
            if (len(parts) != 7 or parts[1] != 'atom' or parts[3] != 'image'
                    or not re.fullmatch(r'\d+(\*\d+)?|\*', parts[2])
                    or not all(re.fullmatch(r'-?\d+', value)
                               for value in parts[4:])):
                raise AssertionError('unrecognized set command: %s' % text)
            held = getattr(self, 'image', {})
            for atom_id in _stub_ids(parts[2], self.natom):
                held[atom_id] = [int(value) for value in parts[4:]]
            self.image = held
        return 0

    def get_natoms(self):
        return self.natom

    # The deployed pylammps hands both arrays straight to the C function, so
    # a Python list raises ctypes.ArgumentError there. The stub reproduces
    # that, because an interface stand-in that accepts anything lets a real
    # TypeError walk onto the machine again.
    @staticmethod
    def _require_array(values):
        if not hasattr(values, '_type_'):
            raise ctypes.ArgumentError(
                'expected LP_c_int or LP_c_double instance instead of list')
        return list(values)

    def scatter_atoms(self, name, dtype, count, data):
        data = self._require_array(data)
        rec.log({'op': 'scatter', 'attr': name, 'dtype': dtype,
                 'count': count, 'length': len(data)})
        if name == 'x':
            self.x = [float(value) for value in data]
        return None

    def gather_atoms_subset(self, name, dtype, count, ndata, ids):
        # The deployed pylammps mirrors the C API here:
        # (name, type, count, ndata, ids). The stub enforces ndata == len(ids)
        # so a call that drops ndata fails here exactly as it does on the
        # machine, instead of passing on a lenient stand-in.
        ids = self._require_array(ids)
        if int(ndata) != len(ids):
            raise TypeError(
                f'gather_atoms_subset: ndata={ndata} does not match '
                f'{len(ids)} ids')
        rec.log({'op': 'gather_subset', 'attr': name, 'dtype': dtype,
                 'count': count, 'ndata': int(ndata), 'ids': ids})
        if dtype == 0:
            return (ctypes.c_int * len(ids))(*[int(tag) for tag in ids])
        out = []
        for tag in ids:
            index = (int(tag) - 1) * 3
            out.extend([-self.x[index + k] * 0.5 - 1.0 for k in range(3)])
        return (ctypes.c_double * len(out))(*out)

    def extract_compute(self, cid, style, kind):
        # The pybind11 interface passes the C integers through and answers
        # None for a combination it does not know, so a string spelling
        # must fail here rather than on a lenient stand-in. A global scalar
        # comes back as a value, a global vector as a pointer with no
        # length attached.
        rec.log({'op': 'extract_compute', 'cid': cid, 'style': style,
                 'kind': kind})
        if not isinstance(style, int) or not isinstance(kind, int):
            return None
        if cid == 'thermo_pe' and style == 0 and kind == 0:
            return ctypes.c_double(sum(self.x))
        if cid == 'Pconf' and style == 0 and kind == 1:
            values = (ctypes.c_double * 6)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            return ctypes.cast(values, ctypes.POINTER(ctypes.c_double))
        return None

    def extract_box(self):
        # The pybind11 answer is the C function's seven parts: boxlo(3),
        # boxhi(3), xy, yz, xz, periodicity(3), box_change. Flattening that
        # blindly reads a periodicity flag as a tilt factor, so the layout
        # is worth reproducing exactly.
        b = list(self.box)
        return ([b[0], b[2], b[4]], [b[1], b[3], b[5]],
                b[6], b[8], b[7], [1, 1, 1], 0)

    def extract_setting(self, name):
        if name == 'triclinic':
            return 1 if self.triclinic else 0
        return None

    def commands_list(self, lines):
        # The pure-Python interface offers both spellings; recording each
        # line as a command keeps the sequence assertions uniform.
        for line in lines:
            self.command(line)
        return 0

    def close(self):
        rec.log({'op': 'close'})
        return 0


LAMMPS_VERSION = 'stub-1'
has_mpi_support = True
'''


BOX = {'xlo': 0.0, 'xhi': 11.0, 'ylo': 0.0, 'yhi': 10.0, 'zlo': 0.0,
       'zhi': 10.0, 'xy': 1.0, 'xz': 0.5, 'yz': 0.25}
BOX_ORDER = ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'xy', 'xz', 'yz')
COORDS = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]

# The static package INIT is handed a real LAMMPS instance: the script
# text is what service/builder produces for a single-point step with the
# file-producing lines removed. The stub replays it, so the tests see the
# same load sequence the deployment will.
SCRIPT_NAME = 'LAMMPS_force'
SCRIPT_TEXT = (
    'include "LAMMPS.in.init"\n'
    'read_data "LAMMPS.data"\n'
    '\n'
    '# PyRAI3MD single-point section\n'
    'compute Pconf all pressure NULL virial\n'
    'thermo_style custom step pe vol '
    'c_Pconf[1] c_Pconf[2] c_Pconf[3] c_Pconf[4] c_Pconf[5] c_Pconf[6]\n'
    'thermo 1\n'
)

# What the service issues for every step, INIT's smoke evaluation
# included. The script above deliberately stops before it.
STEP_TAIL = ['run 0']


class _Service:
    """A spawned stub-backed service and the files that describe it."""

    def __init__(self, process, home, endpoint, session, log_path):
        self.process = process
        self.home = home
        self.endpoint = endpoint
        self.session = session
        self.log_path = log_path

    def stop(self):
        """Kill the service and remove its directory."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        shutil.rmtree(self.home, ignore_errors=True)

    def connect(self):
        """Open a client session against this service."""
        deadline = time.monotonic() + 30.0
        while not os.path.exists(self.endpoint):
            if time.monotonic() > deadline:
                raise RuntimeError('the stub service never came up')
            time.sleep(0.05)
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        while True:
            try:
                stream.connect(self.endpoint)
                break
            except ConnectionRefusedError:
                # The endpoint file appears a moment before the listener
                # accepts on it, so a refused connect this early is the
                # service still starting, not a service that has died.
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        handle = launcher.ServiceHandle(
            self.process, endpoint=self.endpoint,
            owner_record=os.path.join(self.home, 'h2o.owner.json'),
            owner_pid=os.getpid(), nproc=1, python_path=sys.executable,
            log_path=os.path.join(self.home, 'server.log'),
            script_path=launcher.SERVER_FILE)
        return PersistentSession(stream, session=self.session,
                                 fingerprint='f' * 64, natom=3,
                                 order=[1, 2, 3], units='real',
                                 capabilities={'gradient': True,
                                               'stress': True},
                                 expected_world=1, request_timeout=30.0), handle

    def calls(self):
        """The recorded LAMMPS calls, in order."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def commands(self):
        """The recorded LAMMPS command strings, in order."""
        return [entry['text'] for entry in self.calls()
                if entry['op'] == 'command']


def _start(box_mode='ok', owner_pid=None, name='svc.sock'):
    """Spawn one stub-backed service; the caller owns the cleanup."""
    root = '/tmp' if os.path.isdir('/tmp') else tempfile.gettempdir()
    home = tempfile.mkdtemp(prefix='lmpsvc-', dir=root)
    session = os.path.join(home, 'session')
    os.makedirs(session, exist_ok=True)
    with open(os.path.join(session, f'{SCRIPT_NAME}.in'), 'w') as handle:
        handle.write(SCRIPT_TEXT)
    # The stub is installed as a 'lammps' package on the child's path, so
    # the service runs exactly as it does in production: python <file>,
    # importing the real thing by name, with no framework installation and
    # no bootstrap of its own.
    stub_dir = os.path.join(home, 'stub')
    os.makedirs(os.path.join(stub_dir, 'lammps'), exist_ok=True)
    with open(os.path.join(stub_dir, 'lammps_recorder.py'), 'w') as handle:
        handle.write(_RECORDER_SOURCE)
    with open(os.path.join(stub_dir, 'lammps', '__init__.py'),
              'w') as handle:
        handle.write(_STUB_SOURCE)
    env = dict(os.environ)
    env.update({'LMP_STUB_LOG': os.path.join(home, 'calls.jsonl'),
                'LMP_STUB_NATOM': '3', 'LMP_STUB_BOX': box_mode,
                'PYTHONPATH': os.pathsep.join(
                    [stub_dir] + ([env['PYTHONPATH']]
                                  if env.get('PYTHONPATH') else []))})
    process = subprocess.Popen(
        [sys.executable, launcher.SERVER_FILE,
         '--endpoint', os.path.join(home, name), '--session', session,
         '--owner-pid', str(owner_pid or os.getpid()), '--nproc', '1',
         '--idle-timeout', '25'],
        cwd=home, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    return _Service(process, home, os.path.join(home, name), session,
                    env['LMP_STUB_LOG'])


@pytest.fixture
def service():
    """A healthy stub-backed service, stopped after the test."""
    instance = _start()
    try:
        yield instance
    finally:
        instance.stop()


def _ready(instance, **overrides):
    """Hello and INIT a session, returning it."""
    session_client, _handle = instance.connect()
    session_client.hello()
    kwargs = {'cwd': instance.session, 'script': SCRIPT_NAME,
              'files': {'LAMMPS.data': 'x'}, 'profile': 'ff_gradient',
              'pbc': [1, 1, 1], 'step_tail': list(STEP_TAIL)}
    kwargs.update(overrides)
    session_client.initialize(**kwargs)
    return session_client


def _is_number(token):
    """Whether one command token is one of the box numbers."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _kind(entry):
    """Name one recorded call for the sequence assertions."""
    if entry['op'] != 'command':
        return entry['op']
    text = entry['text']
    for marker in ('final', 'image', 'triclinic'):
        if marker in text:
            return marker
    return 'run' if text.startswith('run') else 'command'


class TestBoxSpelling:
    """The one string set a variable-cell step is allowed to send.

    Pinned verbatim because the deployment is the only place that can
    reject it: LAMMPS answers an illegal change_box with MPI_Abort, which
    takes the whole resident group down at the first step. Asserted on the
    recorded commands rather than on the helper, because importing
    service.persistent.server in this interpreter is what the isolation
    test exists to prevent.
    """

    def test_bounds_and_tilts_each_carry_their_own_keyword(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        assert [text for text in service.commands()
                if ' final ' in text] == [
                    'change_box all x final 0.0000000000 11.0000000000'
                    ' y final 0.0000000000 10.0000000000'
                    ' z final 0.0000000000 10.0000000000',
                    'change_box all xy final 1.0000000000'
                    ' xz final 0.5000000000 yz final 0.2500000000',
                ]

    def test_a_rectangular_cell_never_mentions_tilt_factors(self, service):
        # LAMMPS rejects a tilt change on a domain it keeps orthogonal, and
        # the conversion itself moves the reciprocal sum, so an untilted
        # cell must leave the domain exactly as the script loaded it.
        session_client = _ready(service)
        session_client.evaluate(COORDS, {**BOX, 'xy': 0.0, 'xz': 0.0,
                                         'yz': 0.0}, 1)
        commands = service.commands()
        assert 'change_box all triclinic' not in commands
        assert not [text for text in commands if 'xy final' in text]

    def test_remap_is_never_issued(self, service):
        # change_box ... remap wraps atoms into the new box and counts the
        # crossing into their flags. The service states the flags itself
        # with the coordinates of the same step, so letting the engine also
        # accrue them would count every crossing twice.
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        session_client.evaluate(COORDS, {**BOX, 'xy': 1.0, 'xz': 0.5,
                                         'yz': 0.25}, 2)
        assert not [text for text in service.commands() if 'remap' in text]


def _box_commands(instance):
    """The box-moving commands one service has been asked to run."""
    return [text for text in instance.commands()
            if text.startswith('change_box all ')]


def _flag_commands(instance):
    """The image-flag commands one service has been asked to run."""
    return [entry['text'] for entry in instance.calls()
            if entry['op'] == 'command'
            and entry['text'].startswith('set atom')]


class TestImageFlagsTravelWithTheCoordinates:
    """A warm-started instance holds the flags of the step it evaluates.

    The framework stores wrapped (main image) coordinates, so the crossing
    counts are the only thing that says which copy of the cell an atom
    means. Writing new positions while the instance keeps the flags of the
    previous step is what made LAMMPS report inconsistent image flags and
    evaluate a bond across the whole box.
    """

    def test_first_step_states_every_atom_once(self, service):
        # One command per run of a distinct target, because set takes one
        # value per command; the alternative is trusting what read_data
        # inferred from box containment for a run that may have crossed and
        # come back.
        session_client = _ready(service)
        session_client.evaluate(COORDS, None, 1,
                                image=[[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        assert _flag_commands(service) == ['set atom 1 image 0 0 0',
                                           'set atom 3 image 0 0 0',
                                           'set atom 2 image 1 0 -1']

    def test_a_run_of_atoms_shares_one_range_command(self, service):
        # set atom parses its selection as a single token, so contiguous
        # IDs travel as lo*hi rather than one command per atom, and the
        # first evaluation of a 648-atom box is one command rather than 648.
        session_client = _ready(service)
        session_client.evaluate(COORDS, None, 1,
                                image=[[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        assert _flag_commands(service) == ['set atom 3 image 0 0 0',
                                           'set atom 1*2 image 1 0 0']

    def test_a_step_that_crossed_nothing_sends_no_command(self, service):
        session_client = _ready(service)
        image = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
        session_client.evaluate(COORDS, None, 1, image=image)
        seen = len(_flag_commands(service))
        session_client.evaluate(COORDS, None, 2, image=image)
        assert len(_flag_commands(service)) == seen

    def test_only_the_crossing_atom_is_rewritten(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, None, 1,
                                image=[[0, 0, 0]] * 3)
        seen = len(_flag_commands(service))
        session_client.evaluate(COORDS, None, 2,
                                image=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert _flag_commands(service)[seen:] == ['set atom 2 image 0 1 0']

    def test_missing_counts_read_as_no_crossings(self, service):
        # A client that knows nothing about the channel still describes a
        # run where nothing crossed, which is what it meant before.
        session_client = _ready(service)
        session_client.evaluate(COORDS, None, 1)
        assert _flag_commands(service) == ['set atom 1*3 image 0 0 0']

    def test_a_non_periodic_run_owns_no_flags(self, service):
        session_client = _ready(service, pbc=[0, 0, 0])
        session_client.evaluate(COORDS, None, 1, image=[[0, 0, 0]] * 3)
        assert _flag_commands(service) == []

    def test_counts_for_the_wrong_atom_count_are_refused(self, service):
        # Refused on the client side: the message never reaches the service,
        # because a half-described step is not worth a round trip.
        session_client = _ready(service)
        with pytest.raises(LammpsPersistentError, match='describe'):
            session_client.evaluate(COORDS, None, 1, image=[[0, 0, 0]])


class TestHandshake:
    """The service describes itself before anything is loaded."""

    def test_hello_reports_the_build(self, service):
        session_client, _handle = service.connect()
        info = session_client.hello()
        assert info['lammps_version'] == 'stub-1'
        assert info['world_size'] == 1
        assert info['api_probe']['scatter_atoms'] is True

    def test_unexpected_world_size_is_refused(self, service):
        session_client, _handle = service.connect()
        session_client._expected_world = 4
        with pytest.raises(LammpsPersistentError, match='world'):
            session_client.hello()


class TestInitialization:
    """INIT loads the same script text the subprocess path would write."""

    def test_script_is_loaded_and_evaluated_once(self, service):
        reply_session = service.connect()[0]
        reply_session.hello()
        reply = reply_session.initialize(cwd=service.session,
                                         script=SCRIPT_NAME,
                                         files={'a': 'b'},
                                         profile='ff_gradient',
                                         pbc=[1, 1, 1],
                                         step_tail=list(STEP_TAIL))
        assert reply['natom'] == 3
        assert reply['smoke_energy_raw'] == 0.0
        ops = [entry['op'] for entry in service.calls()]
        assert 'file' in ops
        commands = service.commands()
        assert 'run 0' in commands
        # The script carries no run of its own, so the instance evaluates
        # exactly once at INIT: once on load plus once from the sequence
        # would report a smoke energy that nothing asked for.
        assert commands.count('run 0') == 1
        assert ops.index('file') < [
            index for index, entry in enumerate(service.calls())
            if entry.get('text') == 'run 0'][0]

    def test_load_does_not_convert_the_domain(self, service):
        # Conversion moves the reciprocal sum on this build, so a load that
        # converted "just in case" would not answer with the numbers the
        # subprocess path produces for the same untilted cell.
        _ready(service)
        assert 'change_box all triclinic' not in service.commands()

    def test_a_tilted_cell_converts_the_domain_before_the_tilt_line(
            self, service):
        session_client = _ready(service, kspace='kspace_style pppm 1e-4')
        session_client.evaluate(COORDS, dict(BOX), 1)
        commands = service.commands()
        convert = commands.index('change_box all triclinic')
        assert convert < next(n for n, text in enumerate(commands)
                              if text.startswith('kspace_style'))
        assert convert < next(n for n, text in enumerate(commands)
                              if text.startswith('change_box all xy'))

    def test_a_tilt_of_only_rounding_does_not_convert_the_domain(
            self, service):
        # A cell declared with three 90-degree angles reaches this side with
        # tilt factors of order 1e-16, because cos(radians(90)) is not zero.
        # Reading that as a sheared box converts the domain, PPPM re-grids
        # (measured in the test environment: 6x6x6 to 8x8x8) and the energy moves
        # by 0.106 kcal/mol, which is far more than the parity this path is
        # judged on. The writer side does not emit a tilt line for the same
        # numbers, so the two paths compute on different domains.
        session_client = _ready(service)
        noise = dict(BOX, xy=1e-16, xz=-3e-17, yz=0.0)
        session_client.evaluate(COORDS, noise, 1)
        commands = service.commands()
        assert 'change_box all triclinic' not in commands
        assert not [text for text in commands
                    if text.startswith('change_box all xy')]

    def test_the_writer_uses_the_same_tilt_threshold(self):
        # The two sides never import each other's decision, so the shared
        # threshold is pinned by giving both the same cell: if either side
        # changes what counts as tilted, one of these two assertions breaks.
        from ..service.builder.input_builder import (
            box_to_bound_lines, lattice_to_true_box)
        from ..service.persistent.protocol import TILT_EPSILON

        noise = lattice_to_true_box([[10.0, 0.0, 0.0],
                                     [1e-16, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]], 0.0, 0.0, 0.0)
        assert abs(noise['xy']) < TILT_EPSILON
        assert not any(line.endswith('xy xz yz')
                       for line in box_to_bound_lines(noise))
        sheared = lattice_to_true_box([[10.0, 0.0, 0.0],
                                       [1e-3, 10.0, 0.0],
                                       [0.0, 0.0, 10.0]], 0.0, 0.0, 0.0)
        assert any(line.endswith('xy xz yz')
                   for line in box_to_bound_lines(sheared))

    def test_aperiodic_load_does_not_switch_the_domain(self, service):
        _ready(service, pbc=[0, 0, 0])
        assert 'change_box all triclinic' not in service.commands()

    def test_atom_count_mismatch_is_refused(self, service):
        session_client, _handle = service.connect()
        session_client.hello()
        session_client._natom = 4
        session_client._order = [1, 2, 3, 4]
        with pytest.raises(LammpsPersistentError, match='atoms'):
            session_client.initialize(cwd=service.session,
                                      script=SCRIPT_NAME, files={},
                                      profile='ff_gradient', pbc=[1, 1, 1],
                                      step_tail=list(STEP_TAIL))


class TestEvaluation:
    """One step is box, positions, image flags, run, extraction, in order."""

    def test_numbers_follow_the_scattered_geometry(self, service):
        session_client = _ready(service)
        reply = session_client.evaluate(COORDS, None, 1)
        # 27 coordinates of value 1, 2, 3 in the stub's arithmetic.
        assert reply['energy_raw'] == pytest.approx(18.0)
        assert reply['forces_raw'][0] == pytest.approx([-1.5, -1.5, -1.5])
        assert reply['pressure6_raw'] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert reply['units'] == 'lammps-native'
        assert reply['order'] == 'atom_id'

    def test_box_step_sequence(self, service):
        session_client = _ready(service)
        before = len(service.calls())
        session_client.evaluate(COORDS, dict(BOX), 1)
        step = service.calls()[before:]
        kinds = [_kind(entry) for entry in step]
        assert kinds.index('scatter') > kinds.index('final')
        assert kinds.index('image') > kinds.index('scatter')
        assert kinds.index('run') > kinds.index('image')

    def test_true_box_numbers_reach_change_box(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        # The grammar LAMMPS accepts: bounds in one command with each
        # dimension spelled, tilt factors in the next.
        changes = [text for text in service.commands()
                   if text.startswith('change_box all ')]
        numbers = [float(token) for token in ' '.join(changes).split()
                   if _is_number(token)]
        assert numbers == pytest.approx([BOX[key] for key in BOX_ORDER])

    def test_scatter_uses_the_documented_shape(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, None, 1)
        entry = next(entry for entry in service.calls()
                     if entry['op'] == 'scatter')
        # The deployed signature is scatter_atoms(name, dtype, count, data),
        # with dtype the atom-property mask and count the values per atom.
        assert (entry['attr'], entry['dtype'], entry['count'],
                entry['length']) == ('x', 1, 3, 9)

    def test_forces_are_gathered_by_id(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, None, 1)
        entry = next(entry for entry in service.calls()
                     if entry['op'] == 'gather_subset' and entry['attr'] == 'f')
        assert entry['ids'] == [1, 2, 3]
        # ndata is the count of requested ids; a call that omits it is a
        # TypeError on the machine, and the stub raises the same way.
        assert (entry['dtype'], entry['count'], entry['ndata'],
                len(entry['ids'])) == (1, 3, 3, 3)

    def test_non_finite_numbers_end_the_session(self, service):
        session_client = _ready(service)
        with pytest.raises(LammpsServiceDead, match='finite'):
            session_client.evaluate([[float('nan'), 0.0, 0.0],
                                     [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                                    None, 1)
        assert session_client.poisoned


class TestBoxGate:
    """A domain that did not take the box stops the session."""

    def test_read_back_mismatch_is_fatal(self):
        service = _start(box_mode='wrong')
        try:
            session_client = _ready(service)
            with pytest.raises(LammpsPersistentError, match='read-back'):
                session_client.evaluate(COORDS, dict(BOX), 1)
            assert session_client.poisoned
        finally:
            service.stop()


class TestARepeatedCellIsAppliedOnce:
    """A step that repeats the cell already applied sends no box command.

    ``change_box`` makes LAMMPS re-initialise the reciprocal representation,
    so restating a box that has not changed is not a cheap no-op: one
    measured NVE example sent the same cell 10000 times and drew 10001
    ``PPPM initialization`` blocks with it. The first step still applies,
    because nothing is recorded yet and that step is where the read-back
    check earns its keep.
    """

    def test_the_first_step_states_the_cell(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        assert _box_commands(service)

    def test_a_step_with_the_same_cell_sends_nothing(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        before = len(_box_commands(service))
        session_client.evaluate(COORDS, dict(BOX), 2)
        assert len(_box_commands(service)) == before

    def test_a_changed_cell_is_applied_again(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        before = len(_box_commands(service))
        session_client.evaluate(COORDS, {**BOX, 'xhi': 12.0}, 2)
        assert len(_box_commands(service)) > before

    def test_tilt_after_a_skipped_step_still_converts(self, service):
        # The skip covers only a cell that is byte-for-byte the applied
        # one. The first tilted cell of a run is not that, and it has to
        # reach the one-way conversion, or the tilt factors below would
        # be sent to an orthogonal domain.
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        session_client.evaluate(COORDS, dict(BOX), 2)
        session_client.evaluate(COORDS, {**BOX, 'xy': 1.0}, 3)
        assert 'change_box all triclinic' in ' '.join(service.commands())

    def test_a_step_without_a_cell_is_unaffected(self, service):
        # cell=None means "keep what you hold", which is what the skip
        # now does explicitly for a repeated cell too.
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        before = len(_box_commands(service))
        session_client.evaluate(COORDS, None, 2)
        assert len(_box_commands(service)) == before


class TestSessionPhaseTable:
    """The engine's per-phase timings are left where the run can be read."""

    def test_the_table_lands_in_the_session_directory(self, service):
        session_client = _ready(service)
        session_client.evaluate(COORDS, dict(BOX), 1)
        session_client.evaluate(COORDS, dict(BOX), 2)
        session_client.close()
        path = os.path.join(service.session, 'lammps_resident.perf')
        deadline = time.monotonic() + 10.0
        while not os.path.exists(path) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert os.path.exists(path), (
            'the service answered CLOSE without leaving its phase table; '
            'a downloaded run directory would carry no engine timings')
        text = open(path, encoding='utf-8').read()
        assert '[perf] phase=evaluate steps=2' in text
        # The second step reused the cell, and the table says so, so an
        # A/B of the same example shows whether the skip is firing.
        assert '[perf] box_skipped=1 of 2 steps' in text


class TestThePhaseTableNamesItsEvaluationLine:
    """The engine's own table says which command answered for the run.

    This row earns its place by having caught a harness of mine: a driver that
    flipped a constant which never took effect produced ten samples of one side
    presented as two, and every digest agreed - so parity looked proven and the
    speed looked flat. The line printed per run is what showed the switch had
    never been thrown, and any later A/B of an engine-side candidate is
    unreadable without it.
    """

    def test_the_row_reports_the_installed_sequence(self, tmp_path,
                                                   monkeypatch):
        from ..service.persistent import server

        monkeypatch.setattr(server, 'PROFILE_EVERY', 2)
        _lmp, engine = _engine(tmp_path, ORTHO_BOX)
        engine.state['tail_line'] = engine.state['step_tail'][0]
        for step in (1, 2):
            engine.evaluate({'step': step, 'box': _box(),
                             'coordinates': [[1.0, 2.0, 3.0]] * 3})
        text = (tmp_path / server._PERF_FILE).read_text()
        assert "[perf] per_step_line='run 0'" in text

    def test_the_row_is_absent_before_anything_is_installed(self, tmp_path):
        # Nothing was loaded, so nothing may be claimed about the run.
        _lmp, engine = _engine(tmp_path, ORTHO_BOX)
        assert engine._perf_lines() == []


class TestOwnershipAndShutdown:
    """The service does not outlive its owner, and it closes the instance."""

    def test_owner_death_stops_the_service(self, service):
        owner = subprocess.Popen([sys.executable, '-c',
                                  'import time; time.sleep(120)'])
        watched = _start(owner_pid=owner.pid, name='second.sock')
        try:
            deadline = time.monotonic() + 30.0
            while not os.path.exists(watched.endpoint):
                if time.monotonic() > deadline:
                    pytest.fail('the watched service never came up')
                time.sleep(0.05)
            owner.kill()
            owner.wait()
            started = time.monotonic()
            while watched.process.poll() is None:
                if time.monotonic() - started > 30.0:
                    pytest.fail('the service survived the loss of its owner')
                time.sleep(0.2)
            assert watched.process.returncode == 0
        finally:
            if owner.poll() is None:
                owner.kill()
            watched.stop()

    def test_close_releases_the_instance_without_finalizing(self, service):
        session_client = _ready(service)
        before = len(service.calls())
        session_client.close()
        deadline = time.monotonic() + 10.0
        while 'close' not in [entry['op'] for entry in service.calls()] \
                and time.monotonic() < deadline:
            time.sleep(0.05)
        ops = [entry['op'] for entry in service.calls()[before:]]
        assert 'close' in ops
        # The service must never call the LAMMPS finalize: it would tear
        # down MPI for every sibling instance.
        assert 'finalize' not in ' '.join(service.commands())

    def test_service_exits_after_close(self, service):
        session_client = _ready(service)
        session_client.close()
        deadline = time.monotonic() + 15.0
        while service.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert service.process.returncode == 0


class TestTheRequestCwdIsAppliedOnce:
    """The session directory is resolved once, by the launcher.

    ``main()`` chdirs into ``--session`` before any native import, and the
    launcher resolved that path absolutely. The request names the same
    calculation directory, so a relative value there is the base already
    applied - skipping it is what keeps the deck findable, and it must not
    depend on whether a same-named directory happens to exist under the
    session (which is all ``os.path.isdir`` would have caught).
    """

    def test_a_relative_cwd_is_not_applied_a_second_time(self):
        from ..service.persistent.server import _session_cwd
        assert _session_cwd('scratch/tmp_LAMMPS') == ''
        assert _session_cwd('') == ''
        assert _session_cwd(None) == ''

    def test_an_absolute_cwd_is_still_honoured(self):
        from ..service.persistent.server import _session_cwd
        assert _session_cwd('/scratch/tmp_LAMMPS') == '/scratch/tmp_LAMMPS'


class TestInterfaceShapes:
    """Both Python interfaces are answered, and neither is assumed.

    The deployed pylammps and the older SWIG wrapper disagree about the
    gather signature, about whether a Python list is a valid array, and
    about the shape of extract_box(). The adapter learns which one it is
    talking to from the first call; these fakes let both answers be tested
    without a LAMMPS installation.
    """

    def test_the_swig_flat_box_still_reads(self):
        from ..service.persistent.server import _normalize_extract_box
        flat = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0,       # bounds
                0.0, 0.0, 0.0,                          # origin
                1.0, 0.5, 0.25]                        # tilts
        assert _normalize_extract_box(flat) == flat[:6] + flat[-3:]

    def test_the_pybind_box_maps_xy_xz_yz(self):
        from ..service.persistent.server import _normalize_extract_box
        # boxlo, boxhi, xy, yz, xz, periodicity, box_change: the two tilt
        # orders differ, and a flattened read would take the periodicity
        # flags for the tilt factors.
        raw = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0],
               1.0, 0.25, 0.5, [1, 1, 1], 0)
        assert _normalize_extract_box(raw) == [
            0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 1.0, 0.5, 0.25]

    def test_a_four_argument_gather_is_accepted(self):
        """An older interface takes (name, dtype, count, ids)."""
        from ..service.persistent.server import _ApiAdapter

        class Legacy:
            def __init__(self):
                self.calls = []

            def gather_atoms_subset(self, name, dtype, count, ids):
                self.calls.append((name, dtype, count, list(ids)))
                return [2.0, 3.0, 4.0]

            def gather_state(self):
                return None

        adapter = _ApiAdapter(Legacy())
        assert adapter.gather_subset('f', 1, 3, [1]) == [2.0, 3.0, 4.0]

    def test_a_list_scatter_is_accepted(self):
        from ..service.persistent.server import _ApiAdapter

        class Legacy:
            def __init__(self):
                self.seen = None

            def scatter_atoms(self, name, dtype, count, data):
                if hasattr(data, '_type_'):
                    raise ctypes.ArgumentError('expected a list here')
                self.seen = (name, dtype, count, list(data))

        adapter = _ApiAdapter(Legacy())
        adapter.scatter_positions([[1.0, 2.0, 3.0]])
        assert adapter._lmp.seen == ('x', 1, 3, [1.0, 2.0, 3.0])

    def test_an_interface_that_takes_neither_shape_is_reported(self):
        from ..service.persistent.server import _ApiAdapter

        class Rigid:
            def gather_atoms_subset(self, *args, **kwargs):
                raise TypeError('no shape fits')

        adapter = _ApiAdapter(Rigid())
        with pytest.raises(RuntimeError, match='none of the known call shapes'):
            adapter.gather_subset('f', 1, 3, [1])


class _FakeEngineLmp:
    """The few instance calls the resident engine makes, recorded."""

    def __init__(self, box):
        self.box = list(box)
        self.commands = []

    def command(self, line):
        self.commands.append(line)

    def extract_box(self):
        return self.box


class _FakeEngineAdapter:
    """The engine's extraction side, answering with what was asked."""

    def __init__(self, lmp, shift=0.0):
        self._lmp = lmp
        self.shift = shift

    def box(self):
        return [value + self.shift for value in self._lmp.box]

    def scatter_positions(self, coordinates):
        return None

    def commands(self, lines):
        self._lmp.commands.extend(lines)

    def energy(self):
        return -1.0

    def forces(self, order):
        return [0.0] * (3 * len(order))

    def pressure6(self):
        return [0.0] * 6


def _engine(tmp_path, box, shift=0.0):
    from ..service.persistent.server import ResidentEngine

    lmp = _FakeEngineLmp(box)
    state = {
        'natom': 3, 'order': [1, 2, 3], 'periodic': True,
        'triclinic': False, 'kspace_line': '', 'step_tail': list(STEP_TAIL),
        'capabilities': {'gradient': False, 'stress': False},
        'cwd': str(tmp_path),
    }
    return lmp, ResidentEngine(lmp, _FakeEngineAdapter(lmp, shift), state)


ORTHO_BOX = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0]


def _box(values=None):
    keys = ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'xy', 'xz', 'yz')
    return dict(zip(keys, values if values is not None else ORTHO_BOX))


class TestTheBoxIsAppliedOncePerCell:
    """A step that repeats the cell is not allowed to pay for a rebox.

    ``change_box`` makes LAMMPS re-initialise the reciprocal sum, so a
    fixed-cell run that restates its box every step pays for a k-space
    rebuild it never asked for: the measured NVE example drew 10000
    ``Changing box`` blocks and 10001 ``PPPM initialization`` blocks for a
    cell with exactly one distinct value.
    """

    def test_a_repeated_cell_is_applied_once(self, tmp_path):
        lmp, engine = _engine(tmp_path, ORTHO_BOX)
        for step in (1, 2, 3):
            engine.evaluate({'step': step, 'box': _box(),
                             'coordinates': [[1.0, 2.0, 3.0]] * 3})
        changes = [line for line in lmp.commands
                   if line.startswith('change_box')]
        # One application of an untilted cell is one bounds command.
        assert len(changes) == 1
        assert engine.perf['box_skipped'] == 2

    def test_a_changed_cell_is_applied_again(self, tmp_path):
        lmp, engine = _engine(tmp_path, ORTHO_BOX)
        engine.evaluate({'step': 1, 'box': _box(),
                         'coordinates': [[1.0, 2.0, 3.0]] * 3})
        first = len([line for line in lmp.commands
                     if line.startswith('change_box')])
        moved = _box()
        moved['xhi'] = 11.0
        lmp.box[1] = 11.0
        engine.evaluate({'step': 2, 'box': moved,
                         'coordinates': [[1.0, 2.0, 3.0]] * 3})
        after = len([line for line in lmp.commands
                     if line.startswith('change_box')])
        assert after > first
        assert engine.perf.get('box_skipped', 0) == 0

    def test_a_tilted_cell_after_a_skip_still_converts(self, tmp_path):
        lmp, engine = _engine(tmp_path, ORTHO_BOX)
        engine.evaluate({'step': 1, 'box': _box(),
                         'coordinates': [[1.0, 2.0, 3.0]] * 3})
        engine.evaluate({'step': 2, 'box': _box(),
                         'coordinates': [[1.0, 2.0, 3.0]] * 3})
        tilted = _box([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 1.0, 0.0, 0.0])
        lmp.box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 1.0, 0.0, 0.0]
        engine.evaluate({'step': 3, 'box': tilted,
                         'coordinates': [[1.0, 2.0, 3.0]] * 3})
        assert 'change_box all triclinic' in lmp.commands
        assert engine.state['triclinic'] is True

    def test_a_cell_the_domain_refused_is_not_recorded(self, tmp_path):
        # The signature is kept only after the read-back agrees, so a
        # domain that answered with a different box cannot make the next
        # identical step look like a repeat and skip the check.
        lmp, engine = _engine(tmp_path, ORTHO_BOX, shift=1.0)
        with pytest.raises(RuntimeError, match='read-back'):
            engine.evaluate({'step': 1, 'box': _box(),
                             'coordinates': [[1.0, 2.0, 3.0]] * 3})
        assert engine.state['applied_box'] is None


class TestThePhaseTableStaysWithTheRun:
    """The engine's timings must survive a session that never closes."""

    def test_the_table_is_written_during_the_run(self, tmp_path, monkeypatch):
        from ..service.persistent import server

        monkeypatch.setattr(server, 'PROFILE_EVERY', 2)
        _lmp, engine = _engine(tmp_path, ORTHO_BOX)
        for step in (1, 2):
            engine.evaluate({'step': step, 'box': _box(),
                             'coordinates': [[1.0, 2.0, 3.0]] * 3})
        path = tmp_path / server._PERF_FILE
        assert path.exists(), (
            'no CLOSE reached the service, and with it the phase table '
            'went missing from every downloaded example run')
        text = path.read_text()
        assert '[perf] phase=evaluate steps=2' in text
