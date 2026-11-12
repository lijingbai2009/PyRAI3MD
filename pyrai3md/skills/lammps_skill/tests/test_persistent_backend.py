"""
Unit tests for the backend's two execution paths.

The switch is small, but it decides which of two very different
mechanisms produces the numbers the framework integrates, so these tests
check three things: that turning it off leaves the old path exactly as it
was, that turning it on routes every step through the session instead of
the files, and that a resident session that cannot be served raises
instead of quietly falling back.

The service side is the fake from tests/_fake_lammps_service.py, reached
through a stubbed launcher; no LAMMPS installation is needed.

Run: pytest lammps_skill/tests/test_persistent_backend.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import socket
import sys
import tempfile

import pytest

from .. import backend as backend_module
from ._result_units import apply_result_units
from ..service.persistent import launcher
from ..service.persistent.profile import PROFILE_FILE
from ..service.persistent.protocol import (
    LammpsPersistentError,
    LammpsPersistentUnsupported,
)
from ._fake_lammps_service import FakeLammpsService

DATA = """LAMMPS Description

3  atoms
0  bonds
0  angles
0  dihedrals
0  impropers

1  atom types
0  bond types
0  angle types
0  dihedral types
0  improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 15.999

Atoms

1 1 1 -0.834 0.0000000000 0.0000000000 0.1173000000
2 1 1 0.417 0.0000000000 0.7572000000 -0.4692000000
3 1 1 0.417 0.0000000000 -0.7572000000 -0.4692000000
"""

INIT = """units real
atom_style charge
pair_style lj/cut 9.0
kspace_style pppm 1e-4
"""

SETTINGS = """pair_coeff 1 1 lj/cut 0.1 3.0
"""

ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692],
          [0.0, -0.7572, -0.4692]]


@pytest.fixture
def launch_dir(tmp_path, monkeypatch):
    """Launch directory carrying a compressible input package."""
    for name, text in (('LAMMPS.data', DATA), ('LAMMPS.in.init', INIT),
                       ('LAMMPS.in.settings', SETTINGS)):
        (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _raw(**overrides):
    raw = {'atoms': ATOMS, 'coordinates': COORDS, 'charges': 0,
           'multiplicity': 1, 'profile': 'ff_gradient', 'method': 'ff',
           'lammps_project': 'LAMMPS', 'lammps_units': 'real',
           'lammps_path': '', 'scratch_path': ''}
    raw.update(overrides)
    return raw


class _StubLauncher:
    """Route the launcher calls of one test through a fake service."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.services = []
        self.terminations = []
        self.binary_checks = 0

    def install(self):
        monkeypatch = self.monkeypatch
        monkeypatch.setattr(backend_module, 'validate_environment',
                            self._binary_check)
        monkeypatch.setattr(launcher, 'start_service', self._start)
        monkeypatch.setattr(launcher, 'connect_service', self._connect)
        monkeypatch.setattr(launcher, 'terminate_service', self._terminate)
        return self

    def _binary_check(self, config):
        self.binary_checks += 1

    def _start(self, config, paths, timeout=None):
        service = FakeLammpsService(paths.endpoint).start()
        self.services.append(service)
        # A real start_service leaves the script and the log behind in the
        # calculation directory; the fake has to, or a test about where a
        # user finds the failure would pass on an empty directory.
        os.makedirs(os.path.dirname(paths.owner_record), exist_ok=True)
        with open(paths.script, 'w') as written:
            written.write(launcher.build_service_script(config, ['true']))
        with open(paths.log, 'w') as written:
            written.write('lammps service stub listening\n')
        handle = launcher.ServiceHandle(
            _NoopProcess(), endpoint=paths.endpoint,
            owner_record=paths.owner_record,
            owner_pid=os.getpid(), nproc=1, python_path=sys.executable,
            log_path=paths.log, script_path=paths.script)
        handle.service = service
        return handle

    def _connect(self, handle, timeout=None, connect_interval=None):
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stream.connect(handle.endpoint)
        return stream

    def _terminate(self, handle, reason='owner_close', timeout=None):
        self.terminations.append(handle.endpoint)
        service = getattr(handle, 'service', None)
        if service is not None:
            service.stop()


class _NoopProcess:
    """A finished process: the backend never signals the fake service."""

    def __init__(self):
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


@pytest.fixture
def stub(monkeypatch):
    """A stubbed launcher for one test."""
    return _StubLauncher(monkeypatch).install()


class TestRouting:
    """The switch decides the path, and only the switch."""

    def test_default_stays_on_the_subprocess_path(self, launch_dir, stub,
                                                 monkeypatch):
        monkeypatch.setattr(backend_module, 'get_runner',
                            lambda config: lambda c, d: 0)
        monkeypatch.setattr(backend_module, 'parse_output',
                            lambda config, calcdir, workflow: {
                                'energy': [0.0], 'gradient': [[[0.0] * 3] * 3]})
        backend = backend_module.LammpsBackend(
            {'skill_config': _raw(), 'results': {}},
            execution_id='route_cold')
        assert backend._persistent is False
        assert stub.binary_checks == 1
        backend.close()

    def test_slurm_mode_stays_cold_and_says_so(self, launch_dir, stub,
                                               monkeypatch):
        monkeypatch.setattr(backend_module, 'get_runner',
                            lambda config: lambda c, d: 0)
        with pytest.warns(UserWarning, match='execution_mode=slurm'):
            backend = backend_module.LammpsBackend(
                {'skill_config': _raw(execution_mode='slurm',
                                      python_enabled=True),
                 'results': {}}, execution_id='route_slurm')
        assert backend._persistent is False
        # Slurm is the cold path, and the cold path is where the lmp
        # binary check belongs.
        assert stub.binary_checks == 1

    def test_internal_iteration_profile_is_not_resident(self, launch_dir,
                                                        stub):
        with pytest.raises(ValueError, match='ff_nvt'):
            backend_module.LammpsBackend(
                {'skill_config': _raw(profile='ff_nvt',
                                      python_enabled=True), 'results': {}},
                execution_id='route_nvt')

    def test_uncompressed_templates_refuse_residency(self, launch_dir, stub):
        # A wildcard coefficient row cannot be renumbered, so the compressor
        # degrades the cold path to the verbatim templates. Resident mode
        # evaluates the model, so it has to refuse instead.
        (launch_dir / 'LAMMPS.in.settings').write_text(
            'pair_coeff * * lj/cut 0.1 3.0\n')
        with pytest.warns(UserWarning, match='compression disabled'):
            with pytest.raises(LammpsPersistentUnsupported,
                               match='compressed force-field model'):
                backend_module.LammpsBackend(
                    {'skill_config': _raw(python_enabled=True),
                     'results': {}}, execution_id='route_unproven')
        assert stub.services == []


class TestResidentCalculation:
    """A warm step sends geometry, takes numbers, and writes no files."""

    def _warm(self, stub, **overrides):
        raw = _raw(python_enabled=True, **overrides)
        return backend_module.LammpsBackend(
            {'skill_config': raw, 'results': {}},
            execution_id='warm_' + (overrides.get('profile')
                                    or 'ff_gradient'))

    def test_the_resident_path_checks_no_lmp_binary(self, launch_dir, stub):
        # The resident path never executes lmp, so a deployment without it
        # must still start.
        backend = self._warm(stub)
        try:
            assert stub.binary_checks == 0
        finally:
            backend.close()

    def test_results_follow_the_framework_contract(self, launch_dir, stub):
        backend = self._warm(stub, stress=True)
        try:
            results = backend.calculate(ATOMS, COORDS)
            assert len(results['energy']) == 1
            assert results['gradient'].shape == (len(ATOMS), 3)
            assert len(results['stress_tensor'][0]) == 3
            converted = apply_result_units(results, 'lammps_skill')
            assert len(converted['gradient']) == 1
            assert len(converted['gradient'][0]) == 3
            # The fake evaluates -2 per atom plus the coordinate sum; the
            # native result is converted exactly once by the framework.
            from ..service.utils import ENERGY_TO_HARTREE
            expected = (-6.0 + sum(value for row in COORDS for value in row))
            assert results['energy'][0] == pytest.approx(expected)
            assert converted['energy'][0] == pytest.approx(
                expected * ENERGY_TO_HARTREE['real'])
        finally:
            backend.close()

    def test_results_reach_the_task_config(self, launch_dir, stub):
        task_config = {'skill_config': _raw(python_enabled=True),
                       'results': {}}
        backend = backend_module.LammpsBackend(task_config,
                                               execution_id='warm_store')
        try:
            results = backend.calculate(ATOMS, COORDS)
            assert task_config['results']['energy'] == list(
                results['energy'])
            assert task_config['results']['gradient'] == (
                results['gradient'].tolist())
            converted = apply_result_units(task_config['results'],
                                           'lammps_skill')
            assert len(converted['gradient']) == 1
        finally:
            backend.close()

    def test_the_hot_path_writes_no_per_step_file(self, launch_dir, stub):
        backend = self._warm(stub)
        try:
            backend.calculate(ATOMS, COORDS)
            after_first = sorted(os.listdir(backend.calcdir))
            backend.calculate(ATOMS, [[1.0] * 3] * 3)
            assert sorted(os.listdir(backend.calcdir)) == after_first
            # The one-time package follows the cold naming, so the place to
            # look is the place a subprocess run taught; nothing writes a
            # dump or a second script per step.
            assert 'LAMMPS_force.in' in after_first
            assert 'LAMMPS_force.sh' in after_first
            assert 'LAMMPS_force.log' in after_first
            assert not [name for name in after_first
                        if name.endswith(('.xyz', '.dat', '.cond'))]
        finally:
            backend.close()

    def test_cell_snapshot_becomes_the_step_box(self, launch_dir, stub):
        backend = self._warm(stub, pbc=[1, 1, 1],
                             lattice_vector=[[10.0, 0.0, 0.0],
                                             [0.0, 10.0, 0.0],
                                             [0.0, 0.0, 10.0]])
        try:
            backend.calculate(ATOMS, COORDS)
            cell = {'lattice_vector': [[12.0, 0.0, 0.0], [1.0, 10.0, 0.0],
                                       [0.5, 0.25, 10.0]],
                    'pbc': [1, 1, 1], 'cell': [], 'volume': 1200.0}
            backend.calculate(ATOMS, COORDS, cell=cell)
            evaluate = [op for op in stub.services[0].operations
                        if op['op'] == 'EVALUATE']
            last = evaluate[-1]['box']
            # The step box is stated the way the data file states it, which
            # is the frame change_box takes it back in: measured on the
            # deployment, applying the true bounds instead lands the live
            # domain one shear short of the cell the subprocess path
            # computed on (xhi 11 against the file's 12.5 for xy=1.0,
            # xz=0.5), and the two energies differ from the first tilted
            # step onwards.
            assert last['xhi'] == pytest.approx(13.5)
            assert last['yhi'] == pytest.approx(10.25)
            assert last['xy'] == pytest.approx(1.0)
            assert last['xz'] == pytest.approx(0.5)
            assert last['yz'] == pytest.approx(0.25)
        finally:
            backend.close()

    def test_atom_count_change_is_refused_before_sending(
            self, launch_dir, stub):
        backend = self._warm(stub)
        try:
            before = len(stub.services[0].operations)
            with pytest.raises(LammpsPersistentError, match='atoms'):
                backend.calculate(ATOMS + ['O'], COORDS + [[0.0, 0.0, 0.0]])
            assert len(stub.services[0].operations) == before
        finally:
            backend.close()


def _resident(backend_module, execution_id, **overrides):
    """Start one resident backend on the stubbed service."""
    return backend_module.LammpsBackend(
        {'skill_config': _raw(python_enabled=True, **overrides),
         'results': {}}, execution_id=execution_id)


class TestPhaseTable:
    """The framework-side step table is written where the run can read it."""

    def test_the_backend_rows_land_in_the_calculation_directory(
            self, launch_dir, stub):
        # Nothing in the framework closes a resident backend, so a table
        # written at shutdown is a table nobody collects: the cadence
        # inside the step is the only path that leaves evidence behind.
        from ..service.persistent.profile import (PROFILE_EVERY,
                                                  PROFILE_FILE)
        backend = _resident(backend_module, 'warm_profile')
        try:
            for _step in range(PROFILE_EVERY):
                backend.calculate(ATOMS, COORDS)
            path = os.path.join(backend.calcdir, PROFILE_FILE)
            assert os.path.exists(path)
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
            for name in ('apply_inputs', 'resolve_box', 'session_evaluate',
                         'format_results', 'store_results'):
                assert 'phase=%s ' % name in text
            assert 'phase=TOTAL' in text
            # The client's own buckets ride under their own label, so a
            # reader never adds them to the rows that contain them.
            assert '[client] op=evaluate' in text
            assert ('steps=%d' % PROFILE_EVERY) in text
        finally:
            backend.close()

    def test_a_subprocess_run_writes_no_table(self, launch_dir, stub):
        del stub
        raw = _raw()
        backend = backend_module.LammpsBackend(
            {'skill_config': raw, 'results': {}},
            execution_id='cold_profile')
        try:
            assert getattr(backend, '_profiler', None) is None
            assert not os.path.exists(
                os.path.join(backend.calcdir, 'lammps_backend.perf'))
        finally:
            backend.close()


class TestLifecycle:
    """No path leaves a service behind, and a live socket is not taken over."""

    def test_close_terminates_once_and_is_idempotent(self, launch_dir, stub):
        backend = backend_module.LammpsBackend(
            {'skill_config': _raw(python_enabled=True), 'results': {}},
            execution_id='life_close')
        endpoint = backend._paths.endpoint
        backend.close()
        backend.close()
        assert len(stub.terminations) == 1
        assert endpoint not in launcher.live_services()

    def test_context_manager_releases_the_service(self, launch_dir, stub):
        with backend_module.LammpsBackend(
                {'skill_config': _raw(python_enabled=True), 'results': {}},
                execution_id='life_with') as backend:
            backend.calculate(ATOMS, COORDS)
        assert len(stub.terminations) == 1

    def test_exception_inside_the_block_still_releases(
            self, launch_dir, stub):
        with pytest.raises(RuntimeError):
            with backend_module.LammpsBackend(
                    {'skill_config': _raw(python_enabled=True),
                     'results': {}}, execution_id='life_raise'):
                raise RuntimeError('workflow failed')
        assert len(stub.terminations) == 1

    def test_calculate_after_close_raises_without_restarting(
            self, launch_dir, stub):
        backend = backend_module.LammpsBackend(
            {'skill_config': _raw(python_enabled=True), 'results': {}},
            execution_id='life_after')
        backend.close()
        with pytest.raises(LammpsPersistentError, match='closed'):
            backend.calculate(ATOMS, COORDS)
        assert len(stub.services) == 1

    def test_second_live_backend_on_one_socket_is_refused(
            self, launch_dir, stub):
        first = backend_module.LammpsBackend(
            {'skill_config': _raw(python_enabled=True), 'results': {}},
            execution_id='life_double')
        try:
            with pytest.raises(LammpsPersistentUnsupported,
                               match='already live'):
                backend_module.LammpsBackend(
                    {'skill_config': _raw(python_enabled=True),
                     'results': {}}, execution_id='life_double')
            # The refused backend must not have taken the first one's
            # service down with it.
            assert len(stub.terminations) == 0
            assert first._session is not None
        finally:
            first.close()

    def test_two_directories_of_one_namespace_are_both_live(
            self, launch_dir, stub):
        # The numerical Hessian of an isolated-workdir batch: every
        # displacement owns a calculation directory, and the namespace is
        # shared because the runtime suffix is dropped when the directory
        # is already unique. The parent optimization loop holds one of
        # these services for its own steps while the batch runs, so
        # refusing the second one refuses the batch (round
        # tmp_20261031_005246, `multiscale_qmmm_ts`).
        def build(index):
            return backend_module.LammpsBackend(
                {'skill_config': _raw(python_enabled=True), 'results': {}},
                calcdir=str(launch_dir / ('hess_%d' % index)
                            / 'tmp_all_field_LAMMPS'),
                execution_id='tmp_all_field_LAMMPS')

        first = build(0)
        second = build(1)
        try:
            assert first._paths.endpoint != second._paths.endpoint
            live = launcher.live_services()
            assert live[first._paths.endpoint] == 'tmp_all_field_LAMMPS'
            assert live[second._paths.endpoint] == 'tmp_all_field_LAMMPS'
            assert len(stub.services) == 2
        finally:
            second.close()
            first.close()
        # Each close releases its own socket, and neither takes the other
        # down early: this is what keying the claim on the socket buys over
        # keying it on the namespace.
        assert sorted(stub.terminations) == sorted(
            [first._paths.endpoint, second._paths.endpoint])

    def test_a_rebuilt_backend_reclaims_its_socket(self, launch_dir,
                                                   stub):
        # A second backend on the same socket is refused only while the
        # first service is live (see the test above); once it is closed, the
        # rebuild takes over that socket. The owner record is part of the
        # directory, not of a generation: there is one file, and whoever
        # holds the directory now is who it describes.
        def build():
            return backend_module.LammpsBackend(
                {'skill_config': _raw(python_enabled=True), 'results': {}},
                execution_id='life_rebuild')

        first = build()
        record = first._service.owner_record
        first.close()
        second = build()
        try:
            assert second._service.owner_record == record
            assert os.path.dirname(record) == second.calcdir
        finally:
            second.close()

    def test_keep_tmp_zero_clears_the_calculation_directory(self, launch_dir,
                                                            stub):
        backend = backend_module.LammpsBackend(
            {'skill_config': _raw(python_enabled=True, keep_tmp=0),
             'results': {}}, execution_id='life_clean')
        assert os.path.isdir(backend.calcdir)
        backend.close()
        assert not os.path.exists(backend.calcdir)

    def test_keep_tmp_on_keeps_the_products_for_inspection(self, launch_dir,
                                                           stub):
        backend = backend_module.LammpsBackend(
            {'skill_config': _raw(python_enabled=True, keep_tmp=1),
             'results': {}}, execution_id='life_keep')
        calcdir = backend.calcdir
        backend.close()
        products = os.listdir(calcdir)
        # A reader looking for the failure finds the log where the
        # subprocess path would have put it, not in a private directory.
        assert 'LAMMPS.data' in products
        assert 'LAMMPS_force.log' in products
        assert 'LAMMPS_force.sh' in products

    def test_a_starting_failure_cannot_look_like_a_signature_problem(
            self, launch_dir, stub, monkeypatch):
        # Calculator.initialize() retries a backend whose constructor
        # raised TypeError, which would drop execution_id and start a
        # second service; a service failure must therefore never be one.
        def explode(*args, **kwargs):
            raise TypeError('boom inside the launcher')

        monkeypatch.setattr(launcher, 'connect_service', explode)
        with pytest.raises(LammpsPersistentError) as info:
            backend_module.LammpsBackend(
                {'skill_config': _raw(python_enabled=True), 'results': {}},
                execution_id='life_typeerror')
        assert not isinstance(info.value, TypeError)
        assert 'boom inside the launcher' in str(info.value)
        assert len(stub.terminations) == 1


class TestIsolation:
    """The framework process must stay free of the native stack."""

    def test_importing_the_backend_loads_no_native_module(self, launch_dir):
        import subprocess

        code = (
            'import sys; '
            'import pyrai3md.skills.lammps_skill.backend; '
            'assert "lammps" not in sys.modules, "lammps imported"; '
            'assert "mpi4py" not in sys.modules, "mpi4py imported"; '
            'print("clean")')
        completed = subprocess.run([sys.executable, '-c', code],
                                   capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert 'clean' in completed.stdout

    def test_only_the_server_module_may_import_the_native_stack(
            self, launch_dir):
        import ast

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Only the two entry points that run *inside* the service
        # interpreter may import the native stack: the server itself, and
        # the operator-run bringup tool. Everything else, including the
        # modules the framework imports, must not. Reading the syntax tree
        # rather than the text keeps a docstring or the probe program
        # (which runs in the other interpreter) from looking like a hit.
        allowed = {os.path.join('persistent', 'server.py'),
                   os.path.join('persistent', 'bringup.py')}
        offenders = []
        for directory, _dirs, files in os.walk(root):
            if os.sep + 'tests' in directory:
                continue
            for name in files:
                if not name.endswith('.py'):
                    continue
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, root)
                if any(relative.endswith(allowed_path)
                       for allowed_path in allowed):
                    continue
                with open(path) as handle:
                    tree = ast.parse(handle.read(), filename=path)
                for node in ast.walk(tree):
                    names = ()
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or '']
                    if any(name.split('.')[0] in ('lammps', 'mpi4py')
                           for name in names):
                        offenders.append(f'{relative}: {names}')
        assert offenders == []


class TestStepProfilerUnits:
    """The skill-local accumulator renders the service's row format."""

    def test_rows_divide_by_the_steps_the_caller_counts(self):
        from ..service.persistent.profile import StepProfiler
        profiler = StepProfiler()
        profiler.record('session_evaluate', 0.002)
        profiler.record('store_results', 0.0005)
        rows = profiler.rows(10)
        assert rows[0] == ('[perf] phase=session_evaluate steps=10 '
                           'total_s=0.0020 ms_per_step=0.200')
        assert rows[-1] == ('[perf] phase=TOTAL steps=10 total_s=0.0025 '
                             'ms_per_step=0.250')

    def test_the_client_buckets_never_join_the_total(self):
        # The nested rows answer a different question (how the wait for
        # the service broke down), so they ride under their own label and
        # stay out of the phase sum above them.
        from ..service.persistent.profile import StepProfiler
        profiler = StepProfiler()
        profiler.record('session_evaluate', 0.002)
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler.refresh(tmpdir, 1,
                             extra=['[client] op=evaluate calls=1 '
                                    'send_ms=0.010 wait_ms=1.900 '
                                    'parse_ms=0.020'])
            with open(os.path.join(tmpdir, PROFILE_FILE),
                      encoding='utf-8') as handle:
                lines = handle.read().splitlines()
        assert lines[-1].startswith('[client] op=evaluate')
        assert lines[-2] == ('# client buckets (inside session_evaluate '
                             'above)')
        assert lines[1] == ('[perf] phase=TOTAL steps=1 total_s=0.0020 '
                            'ms_per_step=2.000')

    def test_a_table_that_cannot_be_written_does_not_end_the_step(self):
        from ..service.persistent.profile import StepProfiler
        profiler = StepProfiler()
        profiler.record('session_evaluate', 0.002)
        profiler.refresh('/proc/this/cannot/be/written', 1)
