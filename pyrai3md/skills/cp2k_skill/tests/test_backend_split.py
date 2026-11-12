"""
Integration tests for the two-run CP2K path inside `Cp2kBackend.calculate`.

What the unit tests do NOT cover: `service/jobs.py` plans the runs,
`output_parser.merge_job_results` stitches them, and `backend.calculate` loops
over the jobs - but each of those was only ever exercised on its own. This file
runs the whole call sequence once per scenario, with the real input builder, the
real `local_runner` (so the per-project `CP2K_<project>.sh` naming and the actual
execution step are in the loop) and the real parser, and a stand-in for the one
piece that is not on this machine: CP2K itself.

The stand-in writes the engine log the way the deployment does, under the project
name the script asks for, and its two tables are transcribed from the A/B measured
on the cluster: the same ethylene input differing only in `RKS_TRIPLETS` printed
`multiplicity 1` with S1 1.49462 / S2 7.15881 eV and `multiplicity 3` with
T1 -1.41009 / T2 5.99128 eV, and BOTH runs reported the same SCF baseline
`Total energy: -13.56482679723119`. So the numbers asserted below are the numbers
the engine printed, and the round on the real machine has the same four values to
reproduce.

Why the failure arms matter as much as the success arm: a loop that runs two jobs
can fail in the second one while the first succeeded, and a merge that then packs
the missing rows leaves the framework holding an array of the right shape with
zeros in it - the exact shape this skill has spent two rounds refusing to produce.

Run: pytest cp2k_skill/tests/test_backend_split.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import numpy as np
import pytest

from ..service.runner import local_runner

# The A/B baselines and tables, verbatim from the deployment logs.
S0 = -13.56482679723119
EV_TO_HARTREE = 1.0 / 27.211386245988

TABLE_HEADER = [
    '         State    Excitation        Transition dipole (a.u.)'
    '        Oscillator',
    '         number   energy (eV)       x           y           z'
    '     strength (a.u.)',
    '         ----------------------------------------------------'
    '----',
]
SINGLET_TABLE = [
    ' *** SCF run converged ***',
    f' Total energy:                              {S0:18.10f}',
    '',
    ' R-TDDFPT states of multiplicity 1',
    ' Transition dipoles calculated using length formulation',
    '',
] + TABLE_HEADER + [
    ' TDDFPT|      1       1.49462   1.5144E-06  3.6088E-07  5.4721E+00'
    '   1.09646E+00',
    ' TDDFPT|      2       7.15881  -4.1730E-01 -4.1733E-01  2.3686E-05'
    '   6.10876E-02',
]
TRIPLET_TABLE = [
    ' *** SCF run converged ***',
    f' Total energy:                              {S0:18.10f}',
    '',
    ' R-TDDFPT states of multiplicity 3',
    ' Transition dipoles calculated using length formulation',
    '',
] + TABLE_HEADER + [
    ' TDDFPT|      1      -1.41009   0.0000E+00  0.0000E+00  0.0000E+00'
    '  -0.00000E+00',
    ' TDDFPT|      2       5.99128   0.0000E+00  0.0000E+00  0.0000E+00'
    '   0.00000E+00',
]

ATOMS = ['C', 'C', 'H', 'H', 'H', 'H']
COORDS = [
    [-1.120191050, 0.000000000, -2.661941306],
    [-1.120191050, 0.000000000, -1.331783306],
    [-1.120191050, 0.923341000, -3.234891306],
    [-1.120191050, -0.923341000, -3.234891306],
    [-2.043532050, 0.000000000, -0.758833306],
    [-0.196850050, 0.000000000, -0.758833306],
]


# The ground-state forces one SCF prints. Both runs of a split request compute the
# same closed-shell reference, so the two force files must agree; the engine below
# can be told to make them disagree to exercise the merge's ground-state check.
GROUND_FORCES = [
    [0.0010000000, 0.0020000000, -0.0030000000],
    [-0.0010000000, -0.0020000000, 0.0030000000],
    [0.0004000000, -0.0005000000, 0.0001000000],
    [-0.0004000000, 0.0005000000, -0.0001000000],
    [0.0002000000, 0.0003000000, 0.0004000000],
    [-0.0002000000, -0.0003000000, -0.0004000000],
]


def _force_file(lines=GROUND_FORCES):
    out = [' ATOMIC FORCES in [a.u.]', '',
           '   # Atom   Kind   Element          X             '
           '     Y                  Z']
    for index, (fx, fy, fz) in enumerate(lines, 1):
        out.append(f'   {index:5d}   {1 if index < 3 else 2:5d}   '
                   f'{"C" if index < 3 else "H":2s}   '
                   f'{fx:18.10f} {fy:18.10f} {fz:18.10f}')
    return '\n'.join(out) + '\n'


def _tdfrc(blocks):
    """Render tdfrc blocks the way CP2K prints them: header then force rows."""
    lines = []
    for state, excitation, osc, forces in blocks:
        lines.append(f'# STATE NR.    {state}   {excitation:.5f} eV  '
                     f'Oscillator strength:  {osc:.5e}')
        lines.append(str(len(forces)))
        lines.append('')
        for fx, fy, fz in forces:
            lines.append(f'{fx:18.10f} {fy:18.10f} {fz:18.10f}')
    return '\n'.join(lines) + '\n'


class _Engine:
    """Stand-in for the CP2K subprocess.

    Stands where the binary would run: reads the project out of the script the
    runner just wrote, files that project's log and its ground-state force file,
    and returns the code the scenario asked for. The runner's own behaviour (script
    naming, the `bash` call, the return code that `calculate` turns into an
    exception) stays real. A project with no table in `tables` produces NO output
    at all, which is how the half-written-run arms below reach the parser.
    """

    def __init__(self, tables=None, codes=None, forces=None):
        self.tables = tables or {}
        self.codes = codes or {}
        self.forces = forces or {}
        self.calls = []

    def __call__(self, argv, cwd=None, **_kwargs):
        script = str(argv[-1])
        project = os.path.basename(script)[:-3]
        self.calls.append(project)
        body = self.tables.get(project)
        if body is not None:
            # The run solves and prints exactly `NSTATES` roots, read out of the
            # input the builder just wrote: a table or a set of force blocks that
            # ran past that count is not something this engine produces, and the
            # parser is right to refuse it (it did, when the stub invented one).
            with open(os.path.join(cwd, f'{project}.inp')) as handle:
                text = handle.read()
            import re
            nstates = int(re.search(r'NSTATES\s+(\d+)', text).group(1))
            kept, seen = [], 0
            for line in body:
                if line.startswith(' TDDFPT|'):
                    seen += 1
                    if seen > nstates:
                        continue
                kept.append(line)
            body = kept
            with open(os.path.join(cwd, f'{project}.log'), 'w') as handle:
                handle.write('\n'.join(body) + '\n')
            lines = self.forces.get(project, GROUND_FORCES)
            with open(os.path.join(cwd, f'{project}-force-1_0.xyz'),
                      'w') as handle:
                handle.write(_force_file(lines))
            # A gradient run also prints one excited-state force block per root it
            # was asked for. Their headers carry the excitation energy of THIS
            # project's own table, so the attribution the parser does by
            # (number, energy) is exercised against the same numbers, not a
            # invented pair.
            rows = [line.split('|')[1].split() for line in body
                    if line.startswith(' TDDFPT|')]
            blocks = []
            for index, fields in enumerate(rows):
                state, energy = int(fields[0]), float(fields[1])
                blocks.append((state, energy, 0.0,
                               [[0.001 * (index + 1) * (k + 1),
                                 -0.001 * (index + 1) * (k + 1),
                                 0.0005 * (index + 1)]
                                for k in range(6)]))
            if blocks:
                with open(os.path.join(cwd,
                                       f'{project}-td_forces-1_0.tdfrc'),
                          'w') as handle:
                    handle.write(_tdfrc(blocks))
        code = self.codes.get(project, 0)

        class _Proc:
            returncode = code

        return _Proc()


def _request(**overrides):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 4,
        'state_spin': [0, 0, 0, 2],
        'target_states': [0, 1, 2, 3],
        'profile': 'tddft_energy',
        'method': 'tddft',
        'functional': 'pbe0',
        'cutoff': 400,
        'ref_cutoff': 50,
        'project': 'eth',
        'execution_mode': 'local',
        'keep_tmp': 1,
    }
    raw.update(overrides)
    return raw


@pytest.fixture()
def engine(monkeypatch, tmp_path):
    """Wire the stand-in into the real runner and the real environment check."""
    from .. import backend as backend_module

    monkeypatch.setattr(backend_module, 'validate_environment', lambda config: None)
    monkeypatch.setattr(local_runner, 'subprocess', type(
        '_Shim', (), {'run': None}))
    handle = _Engine()
    local_runner.subprocess.run = handle        # type: ignore[attr-defined]
    return handle


def _backend(tmp_path, engine, **overrides):
    from ..backend import Cp2kBackend

    scratch = tmp_path / 'scratch'
    scratch.mkdir(exist_ok=True)
    backend = Cp2kBackend(_request(scratch_path=str(scratch), **overrides))
    engine.tables.setdefault('eth', SINGLET_TABLE)
    return backend


class TestTwoRunsComeBackAsOneAnswer:
    """The loop, the merge and the export contract, on one call."""

    def test_a_two_manifold_request_returns_the_four_engine_numbers(
            self, tmp_path, engine):
        engine.tables = {'eth_S': SINGLET_TABLE, 'eth_T': TRIPLET_TABLE}
        backend = _backend(tmp_path, engine)
        results = backend.calculate(ATOMS, COORDS)

        assert engine.calls == ['eth_S', 'eth_T']
        energy = np.asarray(results['energy'], dtype=np.float64)
        assert energy.shape == (4,)
        assert energy[0] == pytest.approx(S0)
        assert energy[1] == pytest.approx(S0 + 1.49462 * EV_TO_HARTREE)
        assert energy[2] == pytest.approx(S0 + 7.15881 * EV_TO_HARTREE)
        assert energy[3] == pytest.approx(S0 - 1.41009 * EV_TO_HARTREE)
        excitation = np.asarray(results['excitation'], dtype=np.float64)
        assert excitation[1:] == pytest.approx([1.49462, 7.15881, -1.41009])

    def test_a_one_manifold_request_runs_one_job(self, tmp_path, engine):
        # The identity arm, counted in engine calls rather than in shapes: a
        # request naming no triplet must not pay for a second run.
        backend = _backend(tmp_path, engine, nstate=3,
                           state_spin=[0, 0, 0],
                           target_states=[0, 1, 2])
        engine.tables = {'eth': SINGLET_TABLE}
        backend.calculate(ATOMS, COORDS)
        assert engine.calls == ['eth']

    def test_each_run_gets_its_own_script_and_its_own_input(
            self, tmp_path, engine):
        engine.tables = {'eth_S': SINGLET_TABLE, 'eth_T': TRIPLET_TABLE}
        backend = _backend(tmp_path, engine, profile='tddft_gradient')
        backend.calculate(ATOMS, COORDS)
        calcdir = backend.calcdir
        for project, letter, states, listing in (
                ('eth_S', 'F', 'NSTATES 2', 'LIST 1 2'),
                ('eth_T', 'T', 'NSTATES 1', 'LIST 1')):
            script = open(os.path.join(calcdir, f'{project}.sh')).read()
            assert project in script, f'{project}.sh does not run {project}'
            text = open(os.path.join(calcdir, f'{project}.inp')).read()
            assert f'RKS_TRIPLETS {letter}' in text
            assert states in text
            assert listing in text

    def test_the_export_contract_holds_for_a_split_answer(self, tmp_path,
                                                          engine):
        engine.tables = {'eth_S': SINGLET_TABLE, 'eth_T': TRIPLET_TABLE}
        backend = _backend(tmp_path, engine, profile='tddft_gradient')
        results = backend.calculate(ATOMS, COORDS)
        assert len(results['energy']) == 4
        assert np.shape(results['gradient']) == (4, 6, 3)

    def test_narrowing_the_targets_does_not_change_the_job_set(
            self, tmp_path, engine):
        """One backend, two steps, and a count of what each step ran.

        `Optimize` hands the skill the pair it is following, one step at a time.
        Today the job set is planned from the DECLARED rows (`state_spin`), not
        from that step's `target_states`, so the second step here still runs both
        manifolds - and this test records the number that a ruling on per-step
        planning has to pay for. It is pinned so the behaviour cannot drift
        without a test going red.
        """
        engine.tables = {'eth_S': SINGLET_TABLE, 'eth_T': TRIPLET_TABLE}
        backend = _backend(tmp_path, engine)
        backend.calculate(ATOMS, COORDS, target_states=[0, 1])
        assert engine.calls == ['eth_S', 'eth_T']
        engine.calls.clear()
        backend.calculate(ATOMS, COORDS, target_states=[0, 1, 2, 3])
        assert engine.calls == ['eth_S', 'eth_T']


class TestTheSecondJobFailingIsNotReportedAsASuccess:
    """A loop can fail halfway, and a merge must not finish the sentence."""

    def test_a_non_zero_second_run_is_named_by_project(self, tmp_path, engine):
        engine.tables = {'eth_S': SINGLET_TABLE, 'eth_T': TRIPLET_TABLE}
        engine.codes = {'eth_T': 1}
        backend = _backend(tmp_path, engine)
        with pytest.raises(RuntimeError, match='eth_T') as caught:
            backend.calculate(ATOMS, COORDS)
        message = str(caught.value)
        assert 'return code 1' in message
        # The first job's output is still on disk. Naming the failing project is
        # what keeps a caller from reading `eth_S.log` and concluding the run
        # answered the whole request.
        assert os.path.exists(os.path.join(backend.calcdir, 'eth_S.log'))
        assert 'return code' in message and 'project' in message

    def test_a_missing_second_log_is_refused_not_packed_with_zeros(
            self, tmp_path, engine):
        # The script ran and produced nothing: only the singlet run answers, and
        # the triplet rows are left for a refusal, not for a zero.
        engine.tables = {'eth_S': SINGLET_TABLE}
        backend = _backend(tmp_path, engine)
        with pytest.raises(FileNotFoundError, match='eth_T.log') as caught:
            backend.calculate(ATOMS, COORDS)
        # The file, not a state row: this is the arm that used to come back as
        # "nobody answered row 3", three layers from the job that never wrote
        # anything at all.
        assert 'nothing about its states is known' in str(caught.value)

    def test_two_runs_disagreeing_on_the_ground_forces_are_refused(
            self, tmp_path, engine):
        # Both runs compute the same closed-shell reference, so their ground-state
        # forces are one number, not two opinions. A split that lands different
        # forces in the two files is not describing one geometry.
        shifted = [list(row) for row in GROUND_FORCES]
        shifted[0][0] += 1.0e-4
        engine.tables = {'eth_S': SINGLET_TABLE, 'eth_T': TRIPLET_TABLE}
        engine.forces = {'eth_T': shifted}
        backend = _backend(tmp_path, engine, profile='tddft_gradient')
        with pytest.raises(RuntimeError, match='row 0'):
            backend.calculate(ATOMS, COORDS)

    def test_a_swapped_pair_of_logs_fails_at_the_claim(self, tmp_path, engine):
        """The same two tables, filed under the other project.

        A merge that only ever took rows by owner would accept this silently: the
        singlet run would fill nothing it was asked to fill, the triplet table
        would land on the triplet's rows, and the row count would still agree.
        """
        engine.tables = {'eth_S': TRIPLET_TABLE, 'eth_T': SINGLET_TABLE}
        backend = _backend(tmp_path, engine)
        with pytest.raises(RuntimeError, match='belongs to the other run'):
            backend.calculate(ATOMS, COORDS)
