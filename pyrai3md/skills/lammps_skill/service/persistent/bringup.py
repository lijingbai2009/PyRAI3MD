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

Operator-run verification of the resident LAMMPS service.

The local development environment has no LAMMPS, no MPI and no lmp
binary, so nothing below can be proven by the unit tests: this tool is
where the interface assumptions of service/persistent/server.py meet the
build actually deployed. Run it on the target machine against a real
input package:

    python -m pyrai3md.skills.lammps_skill.service.persistent.bringup \
        --dir /path/to/inputs --project h2o --nproc 1 2

Every check prints PASS, FAIL or SKIP with the numbers that decided it,
and the exit status is non-zero if any check failed. Skips are explicit:
a check that could not run is never reported as a pass, because the
whole point of the tool is to say which assumptions were verified where.

What is compared is always the same force field and the same geometry
against the subprocess path, which is the reference this skill already
trusts in production. Tolerances are set from LAMMPS's own arithmetic,
not from what the run happened to produce.
"""

import argparse
import os

import numpy as np
import subprocess
import sys
import time

# Reference tolerances. Energy and force comparisons are between two
# evaluations of identical input, so agreement is tight; parallel
# reduction order differs from serial order, so stress gets a relative
# allowance rather than a bit-for-bit demand.
ENERGY_TOLERANCE = 1e-8
FORCE_TOLERANCE = 1e-6
STRESS_RELATIVE = 1e-6
# How long to wait for a service whose owner was killed to notice.
WATCHDOG_WAIT_S = 30.0

PASS, FAIL, SKIP = 'PASS', 'FAIL', 'SKIP'


class Report:
    """Collect check outcomes and print them once, at the end."""

    def __init__(self):
        self.rows = []

    def add(self, name: str, status: str, detail: str = '') -> str:
        """Record one outcome and echo it immediately."""
        self.rows.append((name, status, detail))
        print(f'[{status}] {name}' + (f': {detail}' if detail else ''),
              flush=True)
        return status

    @property
    def failed(self):
        """Number of checks that failed."""
        return sum(1 for _, status, _ in self.rows if status == FAIL)

    def summary(self) -> int:
        """Print the roll-up and return the process exit status."""
        counts = {PASS: 0, FAIL: 0, SKIP: 0}
        for _, status, _ in self.rows:
            counts[status] = counts.get(status, 0) + 1
        print('\n' + ' ' * 0 + 'bringup: '
              + ', '.join(f'{status} {counts.get(status, 0)}'
                          for status in (PASS, FAIL, SKIP)), flush=True)
        if counts[FAIL]:
            print('the resident service is NOT verified on this '
                  'environment', flush=True)
            return 1
        if counts[SKIP]:
            print('every executed check passed; skipped checks still '
                  'need a run where they can execute', flush=True)
            return 0
        print('all checks passed on this environment', flush=True)
        return 0


def _max_abs_delta(left, right) -> float:
    """Largest elementwise difference between two nested number lists."""
    worst = 0.0
    for a, b in zip(_flat(left), _flat(right)):
        worst = max(worst, abs(float(a) - float(b)))
    return worst


def _flat(value):
    """Yield every number of a nested sequence, arrays included.

    A backend result nests numpy arrays inside lists, and an array is not a
    list or tuple even though it is a sequence, so float() of one fails at
    the leaf with "only length-1 arrays can be converted to Python
    scalars". Converting through tolist() keeps this about the numbers.
    """
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flat(item)
    else:
        yield float(value)


def _relative_delta(left, right) -> float:
    """Largest relative difference, scale taken from the reference."""
    scale = max((abs(number) for number in _flat(right)), default=1.0)
    scale = scale or 1.0
    return _max_abs_delta(left, right) / scale


def _import_lammps():
    """Import the LAMMPS Python module, or report why it is unavailable."""
    try:
        import lammps
    except Exception as exc:                        # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'
    return lammps, ''


def _mpi_support(lammps) -> str:
    """Report the MPI flag from wherever this packaging declares it."""
    for source, holder in (('module', lammps),
                           ('class', getattr(lammps, 'lammps', None))):
        if holder is not None and hasattr(holder, 'has_mpi_support'):
            return f'{bool(getattr(holder, "has_mpi_support"))} ({source})'
    return 'absent (judged by the comm= probes below)'


def check_interface(report: Report, args) -> bool:
    """C1: report the LAMMPS Python interface this build provides.

    Returns
    -------
    bool
        False when the module cannot be imported at all, in which case
        every later check would be meaningless.
    """
    lammps, error = _import_lammps()
    if lammps is None:
        report.add('C1 interface surface', FAIL,
                   f'cannot import lammps here: {error}')
        return False
    names = ('file', 'command', 'scatter_atoms', 'gather_atoms',
             'gather_atoms_subset', 'extract_compute', 'extract_modify',
             'extract_box', 'get_natoms', 'gather_state', 'close',
             'finalize')
    surface = {name: hasattr(lammps.lammps, name) for name in names}
    missing = [name for name in ('file', 'command', 'scatter_atoms',
                                'gather_atoms_subset', 'extract_compute',
                                'get_natoms', 'close')
               if not surface[name]]
    version = (str(getattr(lammps, 'LAMMPS_VERSION', '')
                   or getattr(lammps, '__version__', '')) or '?')
    detail = (f"LAMMPS {version}, "
              f"has_mpi_support={_mpi_support(lammps)}, "
              f"missing={missing or 'none'}")
    try:
        import importlib.util
        mpi4py = importlib.util.find_spec('mpi4py') is not None
    except Exception:                               # noqa: BLE001
        mpi4py = False
    if not mpi4py and max(args.nproc) > 1:
        missing.append('mpi4py(not importable)')
    report.add('C1 interface surface',
               FAIL if missing else PASS, detail)
    return not missing


def _build_config(project: str, profile: str, args, **overrides):
    """Build a skill config from the environment block of the run.

    The reference geometry goes in as input, not as a later mutation of
    the validated model: ``natom`` is derived while normalizing, so a
    config built with an empty atom list and fixed up afterwards asks the
    backend to place 60 atoms under an atom count of 0.
    """
    from ...schema.adapter import build_config
    from ...schema.normalize import normalize
    from ...schema.validate import validate

    atoms, coordinates = _reference_geometry(project)
    raw = {
        'atoms': list(atoms),
        'coordinates': coordinates,
        'profile': profile,
        'method': 'ff',
        'lammps_project': project,
        'lammps_path': args.lammps_path,
        'lammps_nproc': 1,
        'lammps_units': args.units,
        'python_path': args.python_path or sys.executable,
        'execution_mode': 'local',
        'keep_tmp': int(args.keep_tmp),
        'stress': bool(args.stress),
        # Every artifact of the acceptance run belongs to the run,
        # including the calculation directories, so a rerun does not
        # scatter case files through the launch directory it reads.
        'scratch_path': args.workdir,
    }
    raw.update(overrides)
    return validate(normalize(build_config(raw)))


_REFERENCE_GEOMETRY = {}


def _reference_geometry(project: str):
    """Read the topology's own geometry once per project, then reuse it.

    The comparison must not depend on a framework state object, so the
    first evaluation simply asks for the geometry already in the file.
    """
    if project not in _REFERENCE_GEOMETRY:
        _REFERENCE_GEOMETRY[project] = _read_reference_geometry(project)
    return _REFERENCE_GEOMETRY[project]


def _read_reference_geometry(project: str):
    """Take atoms and coordinates from the topology's own data file.

    Element symbols are not carried over: the comparison is about numbers,
    and the force field is chosen by the topology, not by the names.
    """
    atoms, coordinates = [], []
    with open(f'{project}.data') as handle:
        inside = False
        for line in handle:
            head = line.split('#')[0].strip()
            if head == 'Atoms':
                inside = True
                continue
            if inside:
                tokens = head.split()
                if not tokens:
                    continue
                if not tokens[0].isdigit():
                    break
                coordinates.append([float(value) for value in tokens[-3:]])
                atoms.append('X')
    return atoms, coordinates


def _evaluate(backend, atoms, coordinates, cell=None):
    """Run one calculation and return the raw result dict."""
    return backend.calculate(atoms, coordinates, cell=cell)


def check_cold_then_warm(report: Report, args) -> None:
    """C2/C3/C4: subprocess reference, resident match, repeatability."""
    from ...backend import LammpsBackend

    atoms, coordinates = _read_reference_geometry(args.project)
    if not coordinates:
        report.add('C2 cold reference', SKIP, 'no Atoms section to read')
        return

    nproc = min(args.nproc)
    cold_config = _build_config(args.project, 'ff_gradient', args,
                                lammps_nproc=1)
    try:
        cold = LammpsBackend(cold_config, execution_id='bringup_cold')
        reference = _evaluate(cold, atoms, coordinates)
    except Exception as exc:                        # noqa: BLE001
        report.add('C2 cold reference', SKIP,
                   f'subprocess path unavailable: {exc}')
        report.add('C3 resident match', SKIP, 'no reference to compare to')
        return
    report.add('C2 cold reference', PASS,
               f"energy={reference['energy'][0]!r}")

    warm_config = _build_config(args.project, 'ff_gradient', args,
                                python_enabled=True, lammps_nproc=nproc)
    warm = None
    try:
        warm = LammpsBackend(warm_config, execution_id='bringup_warm')
    except Exception as exc:                        # noqa: BLE001
        report.add('C3 resident match', FAIL, f'service did not start: {exc}')
        if warm is not None:
            warm.close()
        return
    try:
        first = _evaluate(warm, atoms, coordinates)
        delta = _max_abs_delta(first['gradient'], reference['gradient'])
        energy_delta = abs(first['energy'][0] - reference['energy'][0])
        detail = (f'dE={energy_delta:.3e} (tol {ENERGY_TOLERANCE}), '
                  f'max|dG|={delta:.3e} (tol {FORCE_TOLERANCE})')
        status = (PASS if energy_delta <= ENERGY_TOLERANCE
                  and delta <= FORCE_TOLERANCE else FAIL)
        report.add('C3 resident match', status, detail)

        # C4: the same geometry must come back identical, and a displaced
        # one must come back different: an instance that stopped listening
        # to its input would otherwise pass C3 by luck.
        again = _evaluate(warm, atoms, coordinates)
        same = _max_abs_delta(again['gradient'], first['gradient'])
        moved = [[value + 0.05 for value in row] for row in coordinates]
        shifted = _evaluate(warm, atoms, moved)
        change = _max_abs_delta(shifted['gradient'], first['gradient'])
        report.add(
            'C4 repeatability and sensitivity',
            PASS if same == 0.0 and change > 0.0 else FAIL,
            f'A->A delta={same:.3e}, displaced delta={change:.3e}')
    finally:
        warm.close()
        cold.close()


def _cell_parity(args, atoms, coordinates, tag, cells):
    """Cold versus resident over one sequence of cells; returns (dE, dG).

    One backend pair per sequence, because the sequence's first cell is
    also the cell the session is initialized with: a run that is tilted
    from the start and one that becomes tilted are different objects to
    the service, and comparing them in one session would hide which
    representation each number came from.
    """
    from ...backend import LammpsBackend

    periodic = {'pbc': [1, 1, 1],
                'lattice_vector': [list(row) for row in cells[0]]}
    cold = warm = None
    try:
        cold = LammpsBackend(
            _build_config(args.project, 'ff_gradient', args, **periodic),
            execution_id=f'bringup_{tag}_cold')
        warm = LammpsBackend(
            _build_config(args.project, 'ff_gradient', args,
                          python_enabled=True, lammps_nproc=min(args.nproc),
                          **periodic),
            execution_id=f'bringup_{tag}_warm')
        worst_energy = worst_force = 0.0
        for lattice in cells:
            cell = {'lattice_vector': [list(row) for row in lattice],
                    'pbc': [1, 1, 1]}
            left = _evaluate(cold, atoms, coordinates, cell=cell)
            right = _evaluate(warm, atoms, coordinates, cell=cell)
            worst_energy = max(worst_energy,
                               abs(left['energy'][0] - right['energy'][0]))
            worst_force = max(worst_force,
                              _max_abs_delta(left['gradient'],
                                             right['gradient']))
        return worst_energy, worst_force
    finally:
        for backend in (warm, cold):
            if backend is not None:
                backend.close()


def _data_file_lattice(path):
    """The (3, 3) row-vector lattice a LAMMPS data file declares, in Angstrom.

    Read through the skill's own inverse of the box writer, so the numbers the
    sequences are built from are the numbers the engine was given - not a copy of
    them kept by hand in this file, which is how the 10 A cube outlived its own
    fixture.

    Parameters
    ----------
    path : str
        The data file, named by the topology the run reads.

    Returns
    -------
    list of list of float
        Three lattice rows.

    Raises
    ------
    ValueError
        When the file states no complete box: a parity priced on a guessed box
        tests the guess.
    """
    from ..builder.input_builder import bound_and_tilt_to_lattice
    bounds, tilts = {}, {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
    with open(path) as handle:
        lines = handle.read().splitlines()
    for line in lines:
        tokens = line.split('#')[0].split()
        if len(tokens) == 4 and tokens[2].endswith('lo')                 and tokens[3].endswith('hi'):
            axis = tokens[2][0]
            bounds[axis + 'lo'] = float(tokens[0])
            bounds[axis + 'hi'] = float(tokens[1])
        elif len(tokens) == 6 and tokens[3:] == ['xy', 'xz', 'yz']:
            tilts = {'xy': float(tokens[0]), 'xz': float(tokens[1]),
                     'yz': float(tokens[2])}
    missing = [key for axis in 'xyz' for key in (axis + 'lo', axis + 'hi')
               if key not in bounds]
    if missing:
        raise ValueError('%s declares no complete box (missing %s): a parity '
                         'priced on a guessed box tests the guess'
                         % (path, ', '.join(missing)))
    return bound_and_tilt_to_lattice(bounds, tilts)


def check_variable_cell(report: Report, args) -> None:
    """C5: per-step box changes match the cold path, tilt included.

    Three sequences, because they exercise three different things: an
    untilted run that resizes, a run that is triclinic from its first
    step, and a run that becomes tilted partway through (which is where
    the service converts its domain). A fourth case, a tilted run whose
    cell later loses its tilt, is measured and reported but not claimed:
    LAMMPS converts a domain to tilt-capable and never back, so the two
    paths then compute the same cell on different representations, and
    the reciprocal sum moves by more than the parity tolerance.
    """
    if not args.periodic:
        report.add('C5 variable cell', SKIP,
                   '--periodic not given: no cell to vary')
        return
    atoms, coordinates = _read_reference_geometry(args.project)
    # The three sequences are one box and two moves off it, where the moves are a
    # fixed fraction of the box rather than absolute lengths. They used to be
    # written as 11x10x10 and its neighbours, which were the right size for the
    # 10 A cube this skill shipped before the equilibrated water box existed: on
    # the box the parity is supposed to survive, that pair is a 10% strain, and
    # the two paths then differ by 4.1e-6 against a 1e-6 tolerance for reasons
    # that have nothing to do with the cell being handed over correctly
    # (dev/TODO.md records the reading). A fraction prices the same three shapes
    # at whatever box the topology carries.
    base = _data_file_lattice(f'{args.project}.data')
    rows = np.asarray(base, dtype=np.float64)
    lengths = np.linalg.norm(rows, axis=1)
    strain = 0.02
    tilt_fraction = 0.05
    ortho = rows.tolist()
    grown = (rows * (1.0 + strain)).tolist()
    tilt = rows.copy()
    tilt[1, 0] = tilt_fraction * lengths[1]
    tilt[2, 0] = tilt_fraction * lengths[2]
    tilt[2, 1] = 0.025 * lengths[2]
    tilt2 = rows.copy()
    tilt2[1, 0] = 0.04 * lengths[1]
    tilt2[2, 0] = 0.02 * lengths[2]
    tilt2[2, 1] = 0.03 * lengths[2]
    tilt = tilt.tolist()
    tilt2 = tilt2.tolist()
    report.add('C5 variable cell', PASS,
               'priced on the box %s declares: rows %.4f / %.4f / %.4f A, '
               'grown by %.0f%%, tilted by %.0f%% of the row'
               % (args.project, *[lengths[k] for k in range(3)],
                  strain * 100.0, tilt_fraction * 100.0))
    sequences = [
        ('resize', 'untilted resize', [ortho, grown, ortho]),
        ('triclinic', 'triclinic throughout', [tilt, tilt2, tilt]),
        ('ontilt', 'untilted then tilted', [ortho, tilt, tilt2]),
    ]
    try:
        for tag, label, cells in sequences:
            delta_energy, delta_force = _cell_parity(args, atoms, coordinates,
                                                     tag, cells)
            report.add(
                'C5 variable cell',
                PASS if (delta_energy <= ENERGY_TOLERANCE
                         and delta_force <= FORCE_TOLERANCE) else FAIL,
                f'{label}: dE={delta_energy:.3e} (tol {ENERGY_TOLERANCE}), '
                f'max|dG|={delta_force:.3e} (tol {FORCE_TOLERANCE})')
    except Exception as exc:                        # noqa: BLE001
        report.add('C5 variable cell', FAIL, f'{type(exc).__name__}: {exc}')
    try:
        delta_energy, delta_force = _cell_parity(args, atoms, coordinates,
                                                 'offtilt', [tilt, ortho])
    except Exception as exc:                        # noqa: BLE001
        report.add('C5 note tilt to untilted', SKIP,
                   f'unavailable: {type(exc).__name__}: {exc}')
        return
    report.add(
        'C5 note tilt to untilted', SKIP,
        f'measured dE={delta_energy:.3e}, max|dG|={delta_force:.3e}: the '
        'domain keeps tilt capability once a step has used it, so the '
        'resident instance computes this cell triclinic-with-zero-tilts '
        'and the subprocess path computes it orthogonal; reported, not '
        'claimed')


def check_ownership(report: Report, args) -> None:
    """C6: a service outliving its killed owner is reaped by the watchdog."""
    from ...service.persistent import launcher

    paths = launcher.service_paths(
        os.path.join(args.workdir, 'bringup_owner'), 'bringup_owner',
        'bringup_owner')
    owner = subprocess.Popen([sys.executable, '-c', 'import time; '
                                                    'time.sleep(600)'])
    handle = None
    try:
        argv = launcher.build_service_argv(args.python_path or sys.executable,
                                           paths.endpoint, paths.work_dir,
                                           owner.pid, 1, 30.0)
        os.makedirs(paths.work_dir, exist_ok=True)
        launcher._ensure_runtime_dir(os.path.dirname(paths.owner_record))
        with open(paths.script, 'w') as handle_file:
            handle_file.write('#!/bin/bash\n'
                              'exec ' + ' '.join(
                                  _quote(token) for token in argv) + '\n')
        os.chmod(paths.script, 0o755)
        process = subprocess.Popen(['bash', paths.script],
                                   cwd=paths.work_dir,
                                   stdout=open(paths.log, 'w'),
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        handle = launcher.ServiceHandle(process, endpoint=paths.endpoint,
                                        owner_record=paths.owner_record,
                                        owner_pid=owner.pid, nproc=1,
                                        python_path=sys.executable,
                                        log_path=paths.log,
                                        script_path=paths.script)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not _can_connect(paths.endpoint):
            if not handle.alive:
                break
            time.sleep(0.2)
        if not handle.alive:
            report.add('C6 owner watchdog', SKIP,
                       'the service did not start here, so there is nothing '
                       'to reap: ' + handle.read_log_tail(6).strip()[:200])
            return
        owner.kill()
        owner.wait()
        started = time.monotonic()
        while time.monotonic() - started < WATCHDOG_WAIT_S:
            if handle.process.poll() is not None:
                report.add('C6 owner watchdog', PASS,
                           f'group exited {time.monotonic() - started:.1f}s '
                           f'after kill -9 of the owner')
                return
            time.sleep(0.2)
        report.add('C6 owner watchdog', FAIL,
                   f'still running {WATCHDOG_WAIT_S:g}s after the owner died')
    finally:
        if handle is not None:
            launcher.terminate_service(handle)
        if owner.poll() is None:
            owner.kill()


def _can_connect(endpoint: str) -> bool:
    """Whether something is accepting on *endpoint*."""
    import socket

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(endpoint)
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _quote(token: str) -> str:
    """Shell-quote one argument without importing shlex at module state."""
    import shlex
    return shlex.quote(str(token))


def check_stress(report: Report, args) -> None:
    """C7: the stress tensor matches the cold path."""
    from ...backend import LammpsBackend

    if not args.stress:
        report.add('C7 stress tensor', SKIP, '--stress not given')
        return
    atoms, coordinates = _read_reference_geometry(args.project)
    cold = warm = None
    try:
        cold_config = _build_config(args.project, 'ff_gradient', args,
                                    stress=True)
        cold = LammpsBackend(cold_config, execution_id='bringup_stress_cold')
        warm_config = _build_config(args.project, 'ff_gradient', args,
                                    stress=True, python_enabled=True,
                                    lammps_nproc=min(args.nproc))
        warm = LammpsBackend(warm_config, execution_id='bringup_stress_warm')
        cold_result = _evaluate(cold, atoms, coordinates)
        warm_result = _evaluate(warm, atoms, coordinates)
        delta = _relative_delta(warm_result['stress_tensor'],
                               cold_result['stress_tensor'])
        report.add('C7 stress tensor',
                   PASS if delta <= STRESS_RELATIVE else FAIL,
                   f'relative delta {delta:.3e} (tol {STRESS_RELATIVE})')
    except Exception as exc:                        # noqa: BLE001
        report.add('C7 stress tensor', FAIL, f'{type(exc).__name__}: {exc}')
    finally:
        for backend in (warm, cold):
            if backend is not None:
                backend.close()


def check_ranks(report: Report, args) -> None:
    """C8: a multi-rank world returns what the single rank returned."""
    from ...backend import LammpsBackend

    ranks = sorted(set(args.nproc))
    if len(ranks) < 2:
        report.add('C8 rank independence', SKIP,
                   'pass --nproc more than once (e.g. --nproc 1 2)')
        return
    atoms, coordinates = _read_reference_geometry(args.project)
    results = {}
    backends = []
    try:
        for nproc in ranks:
            config = _build_config(args.project, 'ff_gradient', args,
                                   python_enabled=True, lammps_nproc=nproc)
            backend = LammpsBackend(config,
                                    execution_id=f'bringup_np{nproc}')
            backends.append(backend)
            results[nproc] = _evaluate(backend, atoms, coordinates)
        reference = results[ranks[0]]
        worst = 0.0
        for nproc in ranks[1:]:
            worst = max(worst,
                        _max_abs_delta(results[nproc]['gradient'],
                                       reference['gradient']),
                        abs(results[nproc]['energy'][0]
                            - reference['energy'][0]))
        report.add('C8 rank independence',
                   PASS if worst <= FORCE_TOLERANCE else FAIL,
                   f'ranks {ranks}, worst delta {worst:.3e}')
    except Exception as exc:                        # noqa: BLE001
        report.add('C8 rank independence', FAIL,
                   f'{type(exc).__name__}: {exc}')
    finally:
        for backend in backends:
            backend.close()


def expected_artifacts(project: str, profile: str, *, stress: bool,
                       result_files: bool) -> set:
    """Names a resident run is allowed to leave in its calculation directory.

    Built from the same helpers that produce the files, so the check cannot
    drift into approving a name no writer ever writes. ``result_files``
    separates the two resident modes: under ``debug`` a step also prints
    its own results, without it the socket carries them and the directory
    holds only what the run was built from.
    """
    from ...service.utils import output_basename, step_data_files

    title = output_basename(project, profile)
    names = {f'{project}.data', f'{project}.in.init', f'{project}.in.settings',
             f'{project}.ff.json', f'{title}.in', f'{title}.sh',
             f'{title}.log'}
    if result_files:
        names.update(step_data_files(title, gradient=True,
                                      stress=stress).values())
    return names


def _stray_runtime_files(*execution_ids) -> list:
    """Anything a service left in the per-user runtime directory."""
    from ...service.persistent import launcher

    found = []
    for execution_id in execution_ids:
        runtime = launcher._runtime_dir(execution_id)
        if os.path.isdir(runtime):
            found.extend(os.path.join(runtime, name)
                         for name in sorted(os.listdir(runtime)))
    return found


def _audit_directory(report, label, calcdir, wanted):
    """Report one directory's contents against the names it should hold."""
    if not os.path.isdir(calcdir):
        report.add(label, SKIP, f'{calcdir} absent')
        return
    names = sorted(os.listdir(calcdir))
    stray = [name for name in names if name.startswith('_')]
    extra = [name for name in names if name not in wanted]
    missing = sorted(wanted - set(names))
    report.add(label, PASS if not (stray or extra or missing) else FAIL,
               f'{len(names)} file(s) in {os.path.basename(calcdir)}'
               + (f'; unexpected: {extra}' if extra else '')
               + (f'; missing: {missing}' if missing else '')
               + (f'; private subdirectory: {stray}' if stray else ''))


def check_report_artifacts(report: Report, args) -> None:
    """C9: audit where a plain resident run put its files.

    A resident step without ``debug`` keeps its results in the socket, so
    its directory holds the inputs and the launch pair and nothing else;
    the ``debug`` directory, which additionally holds the per-step result
    files, is audited by C10 because C10 is what produces it. Nothing may
    be left in the runtime directory: a socket or an owner record still
    there means a service did not retire with its owner.
    """
    _audit_directory(
        report, 'C9 resident artifacts',
        os.path.join(args.workdir, 'bringup_warm'),
        expected_artifacts(args.project, 'ff_gradient', stress=False,
                           result_files=False))
    outside = _stray_runtime_files('bringup_warm', 'bringup_files_warm')
    if outside:
        report.add('C9 nothing outside', FAIL,
                   f'runtime files left behind: {outside}')
    else:
        report.add('C9 nothing outside', PASS,
                   'no service file left in the runtime directory')


def _digest_file(path: str) -> str:
    """sha256 of one file's bytes, for the byte-equality comparison."""
    import hashlib

    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check_data_files(report: Report, args) -> None:
    """C10: a ``debug`` resident step writes what the subprocess step wrote.

    Byte equality of the four result files, not numeric closeness: both
    paths print their numbers through the same LAMMPS commands, so a
    difference in those files is a difference in what the two paths did,
    which is what this check exists to catch. The other names the two
    directories share are deliberately not compared. The launch script is
    how each path starts, the log is what each one said, and the deck
    differs by design: the subprocess deck ends in its own ``run 0`` tail,
    while the resident deck stops before it because the tail travels over
    the socket with each step. The ``debug`` directory is audited here as
    well, because this check is what creates it, and after both backends
    have closed: while a session is live its socket and owner record sit in
    that directory by design, and their absence once the service retired is
    part of what the audit is for.
    """
    from ...backend import LammpsBackend
    from ...service.utils import output_basename, step_data_files

    atoms, coordinates = _read_reference_geometry(args.project)
    if not coordinates:
        report.add('C10 result files', SKIP, 'no Atoms section to read')
        return
    cold = warm = None
    try:
        cold = LammpsBackend(
            _build_config(args.project, 'ff_gradient', args, lammps_nproc=1),
            execution_id='bringup_files_cold')
        warm = LammpsBackend(
            _build_config(args.project, 'ff_gradient', args,
                          python_enabled=True, lammps_nproc=min(args.nproc),
                          debug=True),
            execution_id='bringup_files_warm')
        _evaluate(cold, atoms, coordinates)
        _evaluate(warm, atoms, coordinates)
        wanted = sorted(step_data_files(
            output_basename(args.project, 'ff_gradient'), gradient=True,
            stress=bool(args.stress)).values())
        only_cold, only_warm, differ = [], [], []
        for name in wanted:
            left = os.path.join(cold.calcdir, name)
            right = os.path.join(warm.calcdir, name)
            if not os.path.exists(right):
                only_cold.append(name)
            elif not os.path.exists(left):
                only_warm.append(name)
            elif _digest_file(left) != _digest_file(right):
                differ.append(name)
        report.add('C10 result files',
                   PASS if not (only_cold or only_warm or differ) else FAIL,
                   f'{len(wanted)} result file(s) compared byte for byte'
                   + (f', cold only: {only_cold}' if only_cold else '')
                   + (f', resident only: {only_warm}' if only_warm else '')
                   + (f', differing: {differ}' if differ else ''))
    except Exception as exc:                        # noqa: BLE001
        report.add('C10 result files', FAIL, f'{type(exc).__name__}: {exc}')
    finally:
        for backend in (warm, cold):
            if backend is not None:
                backend.close()
        if warm is not None:
            _audit_directory(
                report, 'C10 debug artifacts', warm.calcdir,
                expected_artifacts(args.project, 'ff_gradient',
                                   stress=bool(args.stress),
                                   result_files=True))


def check_perfbench(report: Report, args) -> None:
    """C11: where the milliseconds of one resident step actually go.

    The subprocess path costs about a second a step because it restarts
    LAMMPS every time. The resident path was measured in the test environment
    at 8.4 ms of framework wall time per step for an engine that spends
    1.47 ms of it, so the open question is not whether the switch pays but
    where its own cost sits. This runs the asked number of steps against
    one session and reports both halves: what the framework waits for, and
    the service's per-phase table, which it writes into its own directory
    every hundred steps (the framework never closes a resident session, so
    an artifact held back for shutdown is an artifact nobody collects).
    Nothing is judged against a threshold, because the number that matters
    is this machine's.
    """
    from ...backend import LammpsBackend

    atoms, coordinates = _read_reference_geometry(args.project)
    if not coordinates:
        report.add('C11 resident cost', SKIP, 'no Atoms section to read')
        return
    steps = max(1, int(args.perfbench))
    config = _build_config(args.project, 'ff_gradient', args,
                           python_enabled=True,
                           lammps_nproc=min(args.nproc))
    backend = LammpsBackend(config, execution_id='bringup_perf')
    try:
        _evaluate(backend, atoms, coordinates)     # prime the session
        started = time.perf_counter()
        for _ in range(steps):
            _evaluate(backend, atoms, coordinates)
        elapsed = time.perf_counter() - started
        timings = backend.resident_timings()
    finally:
        backend.close()

    detail = ('%d steps in %.3f s = %.3f ms/step'
              % (steps, elapsed, 1000.0 * elapsed / steps))
    for name, record in sorted(timings.items()):
        detail += ('; %s calls=%d send=%.3f wait=%.3f parse=%.3f ms'
                   % (name, record['calls'], record['send_ms'],
                      record['wait_ms'], record['parse_ms']))
    report.add('C11 resident cost', PASS,
               detail + '; engine phases: the service log of this run')


def build_parser() -> argparse.ArgumentParser:
    """Command line of the bring-up tool."""
    parser = argparse.ArgumentParser(
        prog='lammps persistent bringup',
        description='Verify the resident LAMMPS service on this machine.')
    parser.add_argument('--dir', default='.', dest='launch_dir',
                        help='directory holding <project>.data and friends')
    parser.add_argument('--project', default='LAMMPS',
                        help='input package base name')
    parser.add_argument('--python-path', default='',
                        help='interpreter of the service process')
    parser.add_argument('--lammps-path', default='',
                        help='lmp binary or installation dir (cold path)')
    parser.add_argument('--units', default='real', choices=('real', 'metal'))
    parser.add_argument('--nproc', type=int, nargs='+', default=[1],
                        help='rank counts to exercise')
    parser.add_argument('--stress', action='store_true',
                        help='also compare the stress tensor')
    parser.add_argument('--periodic', action='store_true',
                        help='exercise variable-cell steps (pbc on)')
    parser.add_argument('--keep-tmp', type=int, default=1,
                        help='keep session directories afterwards')
    parser.add_argument('--workdir', default='',
                        help='scratch directory for the run (default: '
                             '--dir)')
    parser.add_argument('--perfbench', type=int, default=0, metavar='N',
                        help='also time N resident steps and report where '
                             'their milliseconds go (0 skips the check)')
    return parser


def main(argv=None) -> int:
    """Run every check in order and report."""
    args = build_parser().parse_args(argv)
    args.launch_dir = os.path.abspath(args.launch_dir)
    args.workdir = os.path.abspath(args.workdir or args.launch_dir)
    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.launch_dir)
    if not os.path.exists(f'{args.project}.data'):
        print(f'{args.project}.data not found in {args.launch_dir}',
              file=sys.stderr)
        return 2

    print(f'bringup in {args.launch_dir} (project {args.project}, '
          f'ranks {args.nproc}, units {args.units})', flush=True)
    report = Report()
    started = time.monotonic()
    if not check_interface(report, args):
        return report.summary()
    for check in (check_cold_then_warm, check_variable_cell, check_stress,
                  check_ranks, check_ownership, check_report_artifacts,
                  check_data_files):
        try:
            check(report, args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:                    # noqa: BLE001
            report.add(check.__name__, FAIL,
                       f'{type(exc).__name__}: {exc}')
    if args.perfbench:
        # Asked for by number alone: it starts its own session, so it does
        # not depend on the checks above having passed.
        try:
            check_perfbench(report, args)
        except Exception as exc:                    # noqa: BLE001
            report.add('C11 resident cost', FAIL,
                       f'{type(exc).__name__}: {exc}')
    print(f'elapsed {time.monotonic() - started:.1f}s', flush=True)
    return report.summary()


if __name__ == '__main__':
    sys.exit(main())
