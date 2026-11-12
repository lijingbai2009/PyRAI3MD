"""
Unit tests for the bagel_skill active orbital index convention,
forces method block, conditional load_ref archive handling and
log completion checks.

Active orbital indices are 1-based at every boundary (user yaml /
classic input / BAGEL JSON) and stored 0-based internally. These
tests verify the -1 conversion on ingest (build_config), the 0-based
default derivation (normalize) and the +1 conversion on write
(input builder), without requiring a BAGEL installation.

Run: pytest bagel_skill/tests/test_skill.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import json
import warnings

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.method import PROFILE_STEP_MAP, resolve_profile
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder.input_builder import (
    _build_casscf,
    _build_forces,
    _build_forces_caspt2,
    _build_forces_dipole,
    _build_hessian_casscf,
    _build_hessian_caspt2,
    _build_molecule,
    _build_optimize,
    _build_smith_caspt2,
    build_input,
)
from ..service.parser.log_parser import _check_log_completion, parse_log
from ..service.templates.run_templates import (
    build_local_script,
    build_slurm_script,
)
from ..service.workflow import compile_workflow


# Minimal valid configuration mirroring tests/casscf_energy.yaml
# (ethylene, CASSCF(2,2), nclosed=7).
MINIMAL_CONFIG = {
    'atoms': ['C', 'C', 'H', 'H', 'H', 'H'],
    'coordinates': [
        [-1.120191050, 0.000000000, -2.661941306],
        [-1.120191050, 0.000000000, -1.331783306],
        [-1.120191050, 0.923341000, -3.234891306],
        [-1.120191050, -0.923341000, -3.234891306],
        [-2.043532050, 0.000000000, -0.758833306],
        [-0.196850050, 0.000000000, -0.758833306],
    ],
    'charges': 0,
    'multiplicity': 1,
    'nstate': 3,
    'state_spin': [0, 0, 0],
    'profile': 'casscf_energy',
    'method': 'casscf',
    'basis': 'cc-pvdz',
    'nact_elec': 2,
    'nact_orb': 2,
    'nclosed': 7,
}


class TestActiveIndexConvention:
    """Verify the 1-based boundary / 0-based internal convention."""

    def test_build_config_converts_to_zero_based(self):
        raw = dict(MINIMAL_CONFIG, active=[8, 9])
        config = build_config(raw)
        assert config.active == [7, 8]

    def test_build_config_empty_active_stays_empty(self):
        config = build_config(MINIMAL_CONFIG)
        assert config.active == []

    def test_normalize_derives_default_active(self):
        config = normalize(build_config(MINIMAL_CONFIG))
        # 0-based default: range(nclosed, nclosed + nact_orb)
        assert config.active == [7, 8]

    def test_normalize_preserves_user_active(self):
        raw = dict(MINIMAL_CONFIG, active=[8, 9])
        config = normalize(build_config(raw))
        assert config.active == [7, 8]

    def test_builder_writes_one_based(self):
        raw = dict(MINIMAL_CONFIG, active=[8, 9])
        config = normalize(build_config(raw))
        casscf = _build_casscf(config)
        assert casscf['active'] == [8, 9]

    def test_builder_default_active_one_based(self):
        config = normalize(build_config(MINIMAL_CONFIG))
        casscf = _build_casscf(config)
        assert casscf['active'] == [8, 9]

    def test_round_trip_no_drift(self):
        # 1-based input -> 0-based storage -> 1-based BAGEL JSON
        raw = dict(MINIMAL_CONFIG, active=[8, 9])
        config = normalize(build_config(raw))
        assert _build_casscf(config)['active'] == [8, 9]


def _make_config(**overrides):
    """Build a validated SkillConfig from MINIMAL_CONFIG."""
    raw = dict(MINIMAL_CONFIG, **overrides)
    return validate(normalize(build_config(raw)))


class TestForcesMethodBlock:
    """The forces block must carry the method definition array.

    BAGEL aborts with 'A required keyword is missing from the input
    for forces: method' when the method keyword is absent, and with
    'unknown method' when it is a bare dict instead of an array.
    """

    def test_forces_contains_casscf_method_array(self):
        config = _make_config()
        forces = _build_forces(config, nac=False)
        assert isinstance(forces['method'], list)
        method = forces['method'][0]
        assert method['title'] == 'casscf'
        assert method['nstate'] == config.nstate
        assert method['nclosed'] == config.nclosed
        assert method['nact'] == config.nact_orb

    def test_forces_method_writes_one_based_active(self):
        config = _make_config()
        # normalize derives 0-based [7, 8] for ethylene
        forces = _build_forces(config, nac=False)
        assert forces['method'][0]['active'] == [8, 9]

    def test_forces_method_maxiter_from_maxit_scf(self):
        config = _make_config(maxit_scf=500)
        forces = _build_forces(config, nac=False)
        assert forces['method'][0]['maxiter'] == 500

    def test_forces_nacme_keeps_method(self):
        config = _make_config()
        forces = _build_forces(config, nac=True)
        assert isinstance(forces['method'], list)
        titles = [g['title'] for g in forces['grads']]
        assert 'nacme' in titles


class TestCaspt2ForcesTemplate:
    """The XMS-CASPT2 forces block follows the official template."""

    def _method(self, **overrides):
        config = _make_config(profile='xmscaspt2_gradient',
                              method='xmscaspt2', **overrides)
        forces = _build_forces_caspt2(config, nac=False)
        assert isinstance(forces['method'], list)
        return forces['method'][0]

    def test_method_is_array_with_smith(self):
        method = self._method()
        assert method['title'] == 'caspt2'
        assert method['nstate'] == 3
        assert method['nact'] == 2
        assert method['nclosed'] == 7
        assert 'smith' in method

    def test_smith_matches_official_template(self):
        smith = self._method()['smith']
        # smith 'method' stays a plain string: it names the smith
        # algorithm, not the main-task method array
        assert smith['method'] == 'caspt2'
        assert smith['ms'] is True
        assert smith['xms'] is True
        assert smith['sssr'] is True
        assert smith['frozen'] is True
        assert smith['shift'] == 0.2
        assert smith['thresh'] == 1e-8
        assert smith['maxiter'] == 500
        assert 'davidson_subspace' not in smith

    def test_smith_shift_thresh_from_config(self):
        method = self._method(pt2_shift=0.4, scf_converge=1e-6,
                              maxit_scf=300)
        smith = method['smith']
        assert smith['shift'] == 0.4
        assert smith['thresh'] == 1e-6
        assert smith['maxiter'] == 300
        assert method['maxiter'] == 300

    def test_method_has_no_standalone_extras(self):
        method = self._method()
        assert 'natocc' not in method
        # The method-level maxiter aligns with maxit_scf
        assert method['maxiter'] == 500


class TestTemplateConsistency:
    """Cross-block template conventions.

    Every main-task method keyword is an array of definition blocks;
    all casscf/caspt2 keyword regions carry maxiter aligned with
    maxit_scf; the smith 'method' string is the only non-array
    method keyword.
    """

    def test_all_main_methods_are_arrays(self):
        config = _make_config(profile='xmscaspt2_gradient',
                              method='xmscaspt2')
        blocks = [
            _build_forces(config, nac=False),
            _build_forces_caspt2(config, nac=False),
            _build_hessian_casscf(config),
            _build_hessian_caspt2(config),
            _build_optimize(config, compile_workflow(
                'xmscaspt2_geometry_optimization')),
        ]
        for block in blocks:
            assert isinstance(block['method'], list)
            assert isinstance(block['method'][0], dict)

    def test_smith_caspt2_energy_block(self):
        config = _make_config(profile='xmscaspt2_energy',
                              method='xmscaspt2')
        smith = _build_smith_caspt2(config)
        # BAGEL has no standalone caspt2-titled block: the energy
        # chain runs casscf followed by a smith block naming the
        # caspt2 algorithm (official template: flags + shift only)
        assert smith == {
            'title': 'smith',
            'method': 'caspt2',
            'ms': True,
            'xms': True,
            'sssr': True,
            'shift': 0.2,
        }

    def test_smith_caspt2_shift_from_config(self):
        config = _make_config(profile='xmscaspt2_energy',
                              method='xmscaspt2', pt2_shift=0.4)
        assert _build_smith_caspt2(config)['shift'] == 0.4

    def test_xmscaspt2_energy_chain_steps(self, tmp_path):
        config = _make_config(profile='xmscaspt2_energy',
                              method='xmscaspt2')
        blocks = json.loads(build_input(
            config, compile_workflow('xmscaspt2_energy'),
            str(tmp_path)))['bagel']
        assert [b['title'] for b in blocks] == \
            ['molecule', 'casscf', 'smith']
        assert blocks[1]['maxiter'] == config.maxit_scf

    def test_df_basis_similarity_matching(self):
        # An empty df_basis resolves by similarity: same family for
        # svp/tzvpp/qzvpp, same cardinal number for Dunning bases
        # (6z downgrades to 5z), svp-jkfit when no match exists
        cases = {
            'svp': 'svp-jkfit',
            'tzvpp': 'tzvpp-jkfit',
            'cc-pvdz': 'cc-pvdz-jkfit',
            'cc-pcvdz': 'cc-pvdz-jkfit',
            'aug-cc-pvtz': 'cc-pvtz-jkfit',
            'd-aug-cc-pvqz': 'cc-pvqz-jkfit',
            'aug-cc-pwcv5z': 'cc-pv5z-jkfit',
            'cc-pv6z': 'cc-pv5z-jkfit',
            'sto-3g': 'svp-jkfit',
            '6-31g': 'svp-jkfit',
            'ano-rcc': 'svp-jkfit',
        }
        for basis, expected in cases.items():
            config = _make_config(basis=basis, df_basis='')
            assert _build_molecule(config)['df_basis'].endswith(
                expected), basis

    def test_molecule_df_basis_explicit(self):
        config = _make_config(df_basis='cc-pvtz-jkfit')
        assert 'cc-pvtz-jkfit' in \
            _build_molecule(config)['df_basis']

    def test_df_basis_invalid_value_rejected(self):
        with pytest.raises(ValueError):
            _make_config(df_basis='no')
        with pytest.raises(ValueError):
            _make_config(df_basis='cc-pv6z-jkfit')

    def test_hessian_casscf_method_matches_standalone(self):
        config = _make_config()
        method = _build_hessian_casscf(config)['method'][0]
        assert method == _build_casscf(config)
        assert method['maxiter'] == config.maxit_scf

    def test_hessian_caspt2_method_matches_forces_method(self):
        config = _make_config(profile='xmscaspt2_frequency',
                              method='xmscaspt2')
        method = _build_hessian_caspt2(config)['method'][0]
        assert method['smith']['frozen'] is True
        assert method['smith']['maxiter'] == config.maxit_scf
        assert method['maxiter'] == config.maxit_scf


class TestTheFrequencyChainAnswersForItsOwnEnergy:
    """`xmscaspt2_frequency` has to print the energy its run reports.

    Measured in the test environment: round `tmp_20261015_100320`'s
    `xmscaspt2_frequency_test.log` (392 lines) matches no
    `mrci|caspt2|energy|total` line and holds no number of the -77.xxxxxx
    magnitude at all, while the -77.0809867081 that case reported is the first
    line of the sibling cases' ENERGY.out. Round `tmp_20261015_172519`, where a
    calculation directory is named per project, had no foreign file to borrow
    and failed all three carriers. The chain now carries the solve that prints,
    so the number has one writer: this run.
    """

    # Verbatim lines 370-372 of the real `xmscaspt2_energy_test.log` of round
    # tmp_20261015_100320: what the block this chain now runs actually prints.
    ENERGY_BLOCK_LOG = (
        '    * MS-CASPT2 energy : state  0      -77.0809867081\n'
        '    * MS-CASPT2 energy : state  1      -76.9004450750\n'
        '    * MS-CASPT2 energy : state  2      -76.8903112551\n')

    # Verbatim excerpt of the real `xmscaspt2_frequency_test.log` of the same
    # round (its `++ Hessian ++` region): everything that job used to print.
    HESSIAN_ONLY_LOG = (
        '  ++ Hessian ++\n'
        '             0                   1                   2\n'
        '     0       -0.0081011281        0.0000000001       -0.0000000002\n'
        '     1       -0.0000000000        0.6653874909       -0.0000000039\n'
        '  ++ Mass Weighted Hessian ++\n'
        '  ++ Symmetrized Mass Weighted Hessian ++\n')

    def test_the_chain_is_the_energy_chain_plus_the_hessian(self):
        energy = PROFILE_STEP_MAP['xmscaspt2_energy']
        frequency = PROFILE_STEP_MAP['xmscaspt2_frequency']
        assert frequency[:len(energy)] == energy, (
            'the frequency chain no longer starts as the chain whose output '
            'this run reads: %s vs %s' % (frequency, energy))
        assert frequency[-1] == 'hessian_caspt2'

    def test_the_joblist_holds_the_reference_solve_before_the_hessian(
            self, tmp_path):
        # No `{project}_guess.archive` sits in this directory, so `load_ref` is
        # skipped by design (input_builder.py:151-161) and the blocks below are
        # the whole job: the reference solve and the correction are in it.
        config = _make_config(profile='xmscaspt2_frequency',
                              method='xmscaspt2', nstate=3,
                              state_spin=[0, 0, 0])
        blocks = json.loads(build_input(
            config, compile_workflow('xmscaspt2_frequency'),
            str(tmp_path)))['bagel']
        assert [block['title'] for block in blocks] == \
            ['molecule', 'casscf', 'smith', 'hessian']

    def test_the_printed_energies_are_what_the_parser_returns(self, tmp_path):
        log = tmp_path / 'frequency.log'
        log.write_text(self.ENERGY_BLOCK_LOG)
        config = _make_config(profile='xmscaspt2_frequency', method='xmscaspt2',
                              nstate=3, state_spin=[0, 0, 0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = parse_log(str(log), config)
        assert [round(value, 10) for value in results['energy']] == [
            -77.0809867081, -76.9004450750, -76.8903112551]
        borrowed = [str(item.message) for item in caught
                    if 'ENERGY.out' in str(item.message)]
        assert not borrowed, (
            'the log answered, yet the parser still reached for the shared '
            'file: %s' % borrowed)

    def test_a_hessian_only_log_is_still_refused_by_name(self, tmp_path):
        # The negative control: the fix is that the run prints an energy, not
        # that the parser started tolerating its absence. With nothing printed
        # and no ENERGY.out in this directory, the refusal stands.
        log = tmp_path / 'frequency.log'
        log.write_text(self.HESSIAN_ONLY_LOG)
        config = _make_config(profile='xmscaspt2_frequency', method='xmscaspt2',
                              nstate=3, state_spin=[0, 0, 0])
        with pytest.raises(RuntimeError, match='no state energies'):
            parse_log(str(log), config)


class TestAOneStateRunReadsItsOwnCaspt2Line:
    """One declared state leaves SMITH with no multistate Hamiltonian.

    Measured on round `tmp_20261015_211219`, the first round to run the
    frequency chain that carries the reference solve (#355): the job's own log
    prints `* METHOD: CASSCF`, then one CASPT2 row, then
    `* METHOD: SMITH` and **no** `* MS-CASPT2 energy` line at all - a 1x1
    multistate matrix is diagonalised without a printed table. The run declared
    one state, so the reading that answers for it is that CASPT2 row, whose
    value agrees with what the Hessian's own displaced points print in the
    sibling `freq.log` (-77.0713120427 against -77.0713120468). This is the
    engine's behaviour for `nstate == 1`, not a hole cut for a case: the three
    gates below hold the rule to that size.
    """

    # Verbatim excerpt of the real `xmscaspt2_frequency_test.log` of round
    # tmp_20261015_211219 (lines 253-260), including the shift-corrected row
    # directly under the energy: a loosely scanning reader would take
    # -77.0651978849 from it.
    ONE_STATE_LOG = (
        '       - CASPT2 energy evaluation                  0.52\n'
        '\n'
        '    * CASPT2 energy : state  0      -77.0713120427\n'
        '        w/o shift correction        -77.0651978849\n'
        '        reference weight              0.9703360611\n'
        '\n'
        '    * METHOD: SMITH                                0.68\n'
        '  The Hessian will be computed with central gradient '
        'differences (analytical gradients)\n')

    # Three CASPT2 rows, verbatim lines 340-348 of the real
    # `xmscaspt2_energy_test.log` of round tmp_20261015_100320: what a
    # multi-state run prints *before* the MS table.
    THREE_CASPT2_ROWS = (
        '    * CASPT2 energy : state  0      -76.8953781655\n'
        '    * CASPT2 energy : state  1      -77.0809867080\n'
        '    * CASPT2 energy : state  2      -76.8953781646\n')

    def _parse(self, tmp_path, text, **overrides):
        log = tmp_path / 'frequency.log'
        log.write_text(text)
        config = _make_config(profile='xmscaspt2_frequency',
                              method='xmscaspt2', **overrides)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = parse_log(str(log), config)
        borrowed = [str(item.message) for item in caught
                    if 'ENERGY.out' in str(item.message)]
        return results, borrowed

    def test_the_single_printed_row_answers_for_one_state(self, tmp_path):
        results, borrowed = self._parse(
            tmp_path, self.ONE_STATE_LOG, nstate=1, state_spin=[0])
        assert [round(value, 10) for value in results['energy']] == [
            -77.0713120427], (
            'the CASPT2 row this run printed in its own log is the surface it '
            'declares; -77.0651978849 from the shift row must not be read '
            'instead')
        assert not borrowed, (
            'the log answered, yet the parser still reached for the shared '
            'file: %s' % borrowed)

    def test_three_caspt2_rows_do_not_answer_for_three_states(self, tmp_path):
        # Gate one: the row is accepted for one state, not as a substitute
        # section for multistate runs. These three numbers are the un-mixed
        # CASPT2 roots (-76.8953781655 and -76.8953781646 are not the MS face
        # at -77.0809867081), so a three-state run must still refuse rather
        # than relabel them.
        with pytest.raises(RuntimeError, match='no state energies'):
            self._parse(tmp_path, self.THREE_CASPT2_ROWS,
                        nstate=3, state_spin=[0, 0, 0])

    def test_more_rows_than_declared_states_are_refused(self, tmp_path):
        # Gate two: exactly one row, not the first of several. One declared
        # state against two printed roots means the CI solved something else;
        # taking row zero would report a surface this run does not declare.
        with pytest.raises(RuntimeError, match='no state energies'):
            self._parse(tmp_path,
                        self.THREE_CASPT2_ROWS.splitlines()[0] + '\n'
                        + self.THREE_CASPT2_ROWS.splitlines()[1] + '\n',
                        nstate=1, state_spin=[0])

    def test_an_ms_table_still_wins_for_one_state(self, tmp_path):
        # Gate three: precedence. Where the engine does print the multistate
        # table, that is the run's answer even for a single declared state, so
        # the fallback cannot quietly displace it.
        results, _ = self._parse(
            tmp_path,
            self.ONE_STATE_LOG
            + '    * MS-CASPT2 energy : state  0      -77.0809867081\n',
            nstate=1, state_spin=[0])
        assert [round(value, 10) for value in results['energy']] == [
            -77.0809867081]


class TestRunScripts:
    """Run scripts follow the mpiexe dual-mode execution rules.

    MPI mode (default): BAGEL starts under the mpiexe launcher. Direct
    mode (mpiexe is no/none): BAGEL runs without a launcher. A local
    script uses bagel_nproc, while the skill's own slurm script reads
    $SLURM_NTASKS from the header it writes; thread variables come from
    `thread` in both modes (#305) -- the same rule lammps, cp2k and
    openqp follow. Before that MPI mode pinned threads to 1 while the
    SLURM header allocated --cpus-per-task from the same keyword, and
    direct mode read the count off the rank expression.
    """

    def test_default_mpi_mode(self):
        config = _make_config()
        script = build_local_script(config)
        nproc = config.bagel_nproc
        assert config.mpiexe == 'mpirun', 'the default launcher is mpirun'
        assert f'mpirun -np {nproc} $BAGEL/bin/BAGEL' in script
        # thread defaults to 1, so the literal here is the keyword
        # default rather than a pinned policy value.
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BAGEL_NUM_THREADS'):
            assert f'export {var}=1' in script

    def test_threads_follow_the_keyword_in_both_modes(self):
        """The same thread count reaches MPI and direct runs alike."""
        for mpiexe in ('mpirun', 'no'):
            config = _make_config(mpiexe=mpiexe, thread=4)
            script = build_local_script(config)
            for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'BAGEL_NUM_THREADS'):
                assert f'export {var}=4' in script, (mpiexe, var)

    def test_ranks_and_threads_are_separate_keywords(self):
        # bagel_nproc answers "how many ranks", thread answers "how
        # many threads per rank"; neither may stand in for the other.
        config = _make_config(nproc=8, bagel_thread=2)
        script = build_local_script(config)
        assert 'mpirun -np 8 ' in script
        for line in script.splitlines():
            if line.startswith('export ') and line.endswith('_NUM_THREADS=2'):
                continue
            if '_NUM_THREADS=' in line:
                raise AssertionError('a thread export is not the keyword: '
                                     + line)
        # the rank placeholder never reaches a thread variable
        assert 'NUM_THREADS=${' not in script

    def test_custom_mpiexe(self):
        # A site that names its own launcher still gets it; only the
        # default is standardized (#305).
        config = _make_config(mpiexe='mpiexec.hydra')
        script = build_local_script(config)
        assert f'mpiexec.hydra -np {config.nproc} ' in script

    def test_mpiexe_no_runs_directly(self):
        for token in ('no', 'NONE'):
            config = _make_config(mpiexe=token)
            script = build_local_script(config)
            assert f'$BAGEL/bin/BAGEL {config.project}.json' in script
            assert 'mpiexec' not in script
            assert 'mpirun' not in script
            for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'BAGEL_NUM_THREADS'):
                assert f'export {var}={config.thread}' in script

    def test_slurm_allocates_threads_per_rank(self):
        # The header and the run line have to agree: --cpus-per-task is
        # the thread count the exports set (CHANGELOG #303/#305).
        config = _make_config(thread=4)
        script = build_slurm_script(config)
        assert '#SBATCH --cpus-per-task=4' in script
        assert 'export OMP_NUM_THREADS=4' in script

    def test_slurm_script_same_run_rules(self):
        config = _make_config()
        script = build_slurm_script(config)
        assert '#SBATCH' in script
        assert 'mpirun -np ${SLURM_NTASKS:-' in script
        direct = build_slurm_script(_make_config(mpiexe='none'))
        assert '$BAGEL/bin/BAGEL' in direct
        assert 'mpirun -np' not in direct

    def test_mpiexe_routed_to_environment(self):
        # The framework input carries mpiexe as a flat keyword and
        # the adapter routes it into the Environment submodel
        config = build_config(dict(MINIMAL_CONFIG, mpiexe='mpirun'))
        assert config.environment.mpiexe == 'mpirun'
        assert config.mpiexe == 'mpirun'
        # An empty value falls back to the default launcher
        config = validate(normalize(build_config(
            dict(MINIMAL_CONFIG, mpiexe=''))))
        assert config.mpiexe == 'mpirun'


class TestTheFixedNameFilesAreClearedBeforeTheRun:
    """BAGEL's `ENERGY.out`, `FORCE_*.out` and `NACME_*_*.out` carry no job
    identity, so a run must not inherit them from whoever wrote last.

    Measured in round `tmp_20261015_100320`: thirteen bagel cases across three
    carriers shared one calculation directory, and `xmscaspt2_frequency`, whose
    own log prints no energy line, reported -77.08098671 - a value the file had
    received from a sibling case five minutes after that job ran. Naming the
    calculation directory per project (#354) separates the cases; this is the
    other half, and the half that holds even inside a directory one job owns:
    the files a previous invocation left are removed before the engine starts.

    The order is the load-bearing part. A `rm` after the run line would delete
    the results of the job that just produced them and every parse would then
    report nothing, so one criterion checks a foreign file is gone and another
    checks that a file this run wrote survives.
    """

    FIXED_NAMES = ('ENERGY.out', 'FORCE_0.out', 'NACME_0_1.out')

    def _stub_root(self, tmp_path, writes_energy):
        """A stand-in BAGEL install whose `bin/BAGEL` writes the log, and, when
        asked, this job's own `ENERGY.out`."""
        binary = tmp_path / 'bagel' / 'bin' / 'BAGEL'
        binary.parent.mkdir(parents=True, exist_ok=True)
        body = ['#!/bin/bash', 'echo "  STUB BAGEL finished" > "$2.log"']
        if writes_energy:
            body.append('echo "-77.0809867100" > ENERGY.out')
        binary.write_text('\n'.join(body) + '\n')
        binary.chmod(0o755)
        return str(binary.parent.parent)

    def _run_script(self, tmp_path, writes_energy):
        import subprocess
        root = self._stub_root(tmp_path, writes_energy)
        workdir = tmp_path / 'calcdir'
        workdir.mkdir(exist_ok=True)
        config = _make_config(bagel_path=root, mpiexe='no',
                              project='cellcheck')
        path = workdir / 'cellcheck.sh'
        path.write_text(build_local_script(config))
        path.chmod(0o755)
        for name in self.FIXED_NAMES:
            (workdir / name).write_text('-99.0000000000\n')
        proc = subprocess.run(['bash', str(path)], cwd=str(workdir),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return workdir

    def test_both_arms_clear_them_and_do_it_before_the_engine_starts(self):
        config = _make_config()
        for arm, script in (('local', build_local_script(config)),
                           ('slurm', build_slurm_script(config))):
            clear = [index for index, line in enumerate(script.splitlines())
                     if line.startswith('rm -f') and 'ENERGY.out' in line]
            assert clear, '%s arm never clears the fixed-name files' % arm
            run_line = [index for index, line in enumerate(script.splitlines())
                        if '$BAGEL/bin/BAGEL' in line]
            assert run_line, arm
            assert clear[0] < run_line[0], (
                '%s arm clears them after the run: that deletes the results of '
                'the job that just wrote them' % arm)
            for token in ('FORCE_*.out', 'NACME_*.out'):
                assert token in script.splitlines()[clear[0]], (arm, token)

    def test_a_file_left_by_another_run_does_not_survive_into_this_one(
            self, tmp_path):
        workdir = self._run_script(tmp_path, writes_energy=False)
        for name in self.FIXED_NAMES:
            assert not (workdir / name).exists(), (
                '%s survived: the parse would read another run\'s number as '
                'this one\'s answer' % name)

    def test_a_file_this_run_wrote_is_not_deleted(self, tmp_path):
        # The negative control: clearing before the engine must not turn into
        # clearing after it.
        workdir = self._run_script(tmp_path, writes_energy=True)
        energy = workdir / 'ENERGY.out'
        assert energy.exists(), 'the run lost its own output'
        assert energy.read_text().strip() == '-77.0809867100'
        assert (workdir / 'cellcheck.log').exists()
class TestOptimizeBlock:
    """The optimize block follows the official BAGEL template."""

    def _optimize(self, profile='casscf_geometry_optimization',
                  **overrides):
        method = 'xmscaspt2' if profile.startswith('xmscaspt2') \
            else 'casscf'
        config = _make_config(profile=profile, method=method,
                              **overrides)
        workflow = compile_workflow(profile)
        return config, _build_optimize(config, workflow)

    def test_casscf_optimize_structure(self):
        config, optimize = self._optimize(opt_state_i=1,
                                          opt_state_j=1)
        assert optimize['title'] == 'optimize'
        assert optimize['target'] == 1
        assert optimize['opttype'] == 'energy'
        assert 'target2' not in optimize
        assert optimize['maxiter'] == config.maxit_opt
        assert isinstance(optimize['method'], list)
        assert optimize['method'][0]['title'] == 'casscf'

    def test_caspt2_optimize_structure(self):
        config, optimize = self._optimize(
            profile='xmscaspt2_geometry_optimization',
            opt_state_i=0)
        assert optimize['target'] == 0
        assert optimize['opttype'] == 'energy'
        method = optimize['method'][0]
        assert method['title'] == 'caspt2'
        assert method['smith']['frozen'] is True

    def test_meci_opttype_and_target2(self):
        config, optimize = self._optimize(opt_state_i=0,
                                          opt_state_j=2)
        assert optimize['opttype'] == 'meci'
        assert optimize['target'] == 0
        assert optimize['target2'] == 2

    def test_equal_state_pair_stays_energy(self):
        _, optimize = self._optimize(opt_state_i=1, opt_state_j=1)
        assert optimize['opttype'] == 'energy'
        assert optimize['target'] == 1
        assert 'target2' not in optimize

    def test_maxiter_follows_maxit_opt(self):
        _, optimize = self._optimize(maxit_opt=42)
        assert optimize['maxiter'] == 42

    def test_resolve_profile_optimize(self):
        assert resolve_profile('casscf', 'optimize') == \
            'casscf_geometry_optimization'
        assert resolve_profile('xmscaspt2', 'optimize') == \
            'xmscaspt2_geometry_optimization'

    def test_workflow_graph_needs_optimize(self):
        workflow = compile_workflow('casscf_geometry_optimization')
        assert workflow.needs_optimize
        assert workflow.steps == \
            ['molecule', 'load_ref', 'casscf', 'optimize']


class TestLoadRefConditional:
    """load_ref is only written when the guess archive exists."""

    def _input_blocks(self, calcdir=''):
        config = _make_config(profile='xmscaspt2_energy',
                              method='xmscaspt2')
        workflow = compile_workflow('xmscaspt2_energy')
        content = build_input(config, workflow, calcdir)
        return json.loads(content)['bagel']

    def test_load_ref_skipped_without_archive(self, tmp_path):
        blocks = self._input_blocks(str(tmp_path))
        titles = [b['title'] for b in blocks]
        assert 'load_ref' not in titles

    def test_load_ref_written_with_archive(self, tmp_path):
        config = _make_config(profile='xmscaspt2_energy',
                              method='xmscaspt2')
        archive = tmp_path / f'{config.project}_guess.archive'
        archive.write_text('')
        blocks = self._input_blocks(str(tmp_path))
        titles = [b['title'] for b in blocks]
        assert 'load_ref' in titles

    def test_load_ref_written_without_calcdir(self):
        # No calcdir given: the existence check is disabled and the
        # block is written (standalone string-building behaviour).
        blocks = self._input_blocks()
        titles = [b['title'] for b in blocks]
        assert 'load_ref' in titles

    def test_casscf_chain_load_ref_conditional(self, tmp_path):
        # The CASSCF chains carry load_ref between molecule and
        # casscf, honouring the same skip-if-absent policy.
        config = _make_config(profile='casscf_gradient')
        workflow = compile_workflow('casscf_gradient')
        blocks = json.loads(
            build_input(config, workflow, str(tmp_path)))['bagel']
        assert 'load_ref' not in [b['title'] for b in blocks]

        archive = tmp_path / f'{config.project}_guess.archive'
        archive.write_text('')
        blocks = json.loads(
            build_input(config, workflow, str(tmp_path)))['bagel']
        titles = [b['title'] for b in blocks]
        assert titles == ['molecule', 'load_ref', 'casscf', 'forces']


class TestDipoleBlock:
    """Dipole properties come from a forces block with dipole:true.

    The dipole keyword switches dipole output on; the grads entries
    disable derivative evaluation via ciderivative. Force entries
    yield state dipoles, nacme entries yield transition dipoles.
    """

    def _dipole(self, profile='casscf_dipole', **overrides):
        method = 'xmscaspt2' if profile.startswith('xmscaspt2') \
            else 'casscf'
        config = _make_config(profile=profile, method=method,
                              **overrides)
        workflow = compile_workflow(profile)
        return config, _build_forces_dipole(config, workflow)

    def test_casscf_dipole_structure(self):
        _, forces = self._dipole('casscf_dipole')
        assert forces['title'] == 'forces'
        assert forces['dipole'] == 'true'
        assert 'export' not in forces
        assert isinstance(forces['method'], list)
        assert forces['method'][0]['title'] == 'casscf'
        # State dipoles: one force entry per state, derivatives off
        grads = forces['grads']
        assert len(grads) == 3
        assert grads[0] == {'title': 'force', 'target': 0,
                            'ciderivative': 'false'}
        assert [g['target'] for g in grads] == [0, 1, 2]

    def test_transition_dipole_grads_nacme(self):
        _, forces = self._dipole('casscf_transition_dipole')
        grads = forces['grads']
        assert all(g['title'] == 'nacme' for g in grads)
        assert all(g['ciderivative'] == 'false' for g in grads)
        assert all(g['nacmtype'] == 'interstate' for g in grads)
        # Default transition pairs are the declared ground-to-state pairs:
        # the self pair [0, 0] is not one of them (#448), so a 3-state run
        # asks BAGEL for two interstate derivatives, not three.
        assert [[g['target'], g['target2']] for g in grads] == \
            [[0, 1], [0, 2]]

    def test_an_unset_transition_table_asks_for_no_nacme_entry(self):
        # The deck builder is where the self pair used to be born: with an
        # unset table it filled `[[0, i] for i in range(nstate)]`, so BAGEL was
        # handed a "nacme" request from state 0 to state 0 before any reader
        # could see it (#448). This arm drives the builder with the marker a
        # config carries before normalization, which is the only way to reach
        # that branch - every fixture that goes through `normalize` already has
        # a real table. Restoring the `range(nstate)` fallback puts a [0, 0]
        # entry back and this fails.
        config = _make_config(profile='casscf_transition_dipole')
        config.system.transition_pair = [[]]
        workflow = compile_workflow('casscf_transition_dipole')
        forces = _build_forces_dipole(config, workflow)
        pairs = [[g['target'], g['target2']] for g in forces['grads']
                 if g['title'] == 'nacme']
        assert pairs == [[0, 1], [0, 2]]
        assert [p for p in pairs if p[0] == p[1]] == []

    def test_combined_request_flags(self):
        _, forces = self._dipole('casscf_dipole', req_dipole=True,
                                 req_transition_dipole=True)
        grads = forces['grads']
        titles = [g['title'] for g in grads]
        assert titles == ['force'] * 3 + ['nacme'] * 2
        assert all(g['ciderivative'] == 'false' for g in grads)

    def test_caspt2_dipole_structure(self):
        _, forces = self._dipole('xmscaspt2_transition_dipole')
        method = forces['method'][0]
        assert method['title'] == 'caspt2'
        assert method['smith']['frozen'] is True

    def test_build_input_emits_dipole_forces(self, tmp_path):
        config = _make_config(profile='casscf_dipole')
        workflow = compile_workflow('casscf_dipole')
        blocks = json.loads(
            build_input(config, workflow, str(tmp_path)))['bagel']
        dipole_blocks = [b for b in blocks
                         if b.get('title') == 'forces']
        assert len(dipole_blocks) == 1
        assert dipole_blocks[0]['dipole'] == 'true'

    def test_resolve_profile_dipole(self):
        assert resolve_profile('casscf', 'dipole') == 'casscf_dipole'
        assert resolve_profile('casscf', 'energy dipole') == \
            'casscf_dipole'
        assert resolve_profile('casscf', 'energy transition') == \
            'casscf_transition_dipole'
        assert resolve_profile('xmscaspt2', 'absorption') == \
            'xmscaspt2_transition_dipole'

    def test_workflow_graph_needs_dipole(self):
        workflow = compile_workflow('casscf_dipole')
        assert workflow.needs_dipole
        assert workflow.steps == \
            ['molecule', 'load_ref', 'casscf', 'forces_dipole']


class TestDipoleParsing:
    """Dipole results come from the log dipole-moments section.

    BAGEL prints state dipoles, transition dipoles and oscillator
    strengths under '* CASSCF dipole moments' / '* CASPT2 dipole
    moments'; it does not export dipole files.
    """

    DIPOLE_SECTION = (
        '  * CASSCF dipole moments\n'
        '\n'
        '    * State             0 :   (    0.100000,    '
        ' 0.200000,     0.300000) a.u.\n'
        '\n'
        '    * State             1 :   (    0.400000,    '
        ' 0.500000,     0.600000) a.u.\n'
        '\n'
        '    * Transition    1 - 0 :   (    1.000000,    '
        ' 0.000000,     0.000000) a.u.\n'
        '    * Oscillator strength :   0.012500 a.u.\n'
        '\n'
        '    * Transition    2 - 0 :   (    0.000000,    '
        ' 2.000000,     0.000000) a.u.\n'
        '    * Oscillator strength :   0.025000 a.u.\n'
    )

    def _parse(self, tmp_path, profile='casscf_transition_dipole',
               extra_log=''):
        log = tmp_path / 'BAGEL.log'
        log.write_text(
            '    * METHOD: CASSCF      0.18\n'
            + self.DIPOLE_SECTION + extra_log)
        (tmp_path / 'ENERGY.out').write_text(
            '-78.0\n-77.9\n-77.7\n')
        config = _make_config(profile=profile)
        return parse_log(str(log), config)

    def test_dipole_and_transition_dipole_parsed(self, tmp_path):
        results = self._parse(tmp_path)
        assert results['dipole'][0].tolist() == [0.1, 0.2, 0.3]
        assert results['dipole'][1].tolist() == [0.4, 0.5, 0.6]
        # State 2 not printed: zero-fills like other exports
        assert results['dipole'][2].tolist() == [0.0, 0.0, 0.0]
        # One row per declared transition pair, in declaration order:
        # row 0 is (0, 1), row 1 is (0, 2). There is no row for (0, 0).
        assert results['transition_dipole'][0].tolist() == \
            [1.0, 0.0, 0.0]
        assert results['transition_dipole'][1].tolist() == \
            [0.0, 2.0, 0.0]
        assert results['transition_dipole'].shape == (2, 3)

    def test_declared_pair_without_a_printed_transition_is_refused(
            self, tmp_path):
        """A declared pair the log answers for nobody is a missing value.

        Zero-filling it would put "this transition has no dipole" into the
        results of a run whose log never printed one (#448).
        """
        section = self.DIPOLE_SECTION.replace(
            '    * Transition    2 - 0 :   (    0.000000,    ', '')
        log = tmp_path / 'BAGEL.log'
        log.write_text('    * METHOD: CASSCF      0.18\n' + section)
        (tmp_path / 'ENERGY.out').write_text('-78.0\n-77.9\n-77.7\n')
        config = _make_config(profile='casscf_transition_dipole')
        with pytest.raises(RuntimeError,
                           match='prints no transition for it'):
            parse_log(str(log), config)

    def test_oscillator_strength_from_log(self, tmp_path):
        results = self._parse(tmp_path)
        excitation = results['excitation']
        assert excitation[0] == 0.0
        assert abs(excitation[1] - 0.1) < 1e-12
        # BAGEL prints oscillator strengths directly; the rows follow the
        # declared pairs, so row 0 answers (0, 1) and row 1 answers (0, 2).
        assert abs(results['osc_strength'][0] - 0.0125) < 1e-12
        assert abs(results['osc_strength'][1] - 0.0250) < 1e-12

    def test_no_declared_pair_leaves_the_channel_empty(self):
        """Nothing declared, nothing returned - not a zero row.

        Both selectors used to answer an empty table with one row of zeros
        (`return np.zeros(1)` / `np.zeros((1, 3))`), which is a statement that
        a transition exists and carries no dipole. Restoring either fill makes
        this arm red.
        """
        from ..service.parser.log_parser import (_select_osc_strengths,
                                                _select_transition_dipoles)

        config = _make_config(profile='casscf_transition_dipole')
        config.system.transition_pair = [[]]
        assert np.asarray(
            _select_transition_dipoles(config, {})).shape == (0, 3)
        assert np.asarray(_select_osc_strengths(config, {})).shape == (0,)

    def test_caspt2_section_takes_precedence(self, tmp_path):
        caspt2_section = self.DIPOLE_SECTION.replace(
            'CASSCF dipole moments', 'CASPT2 dipole moments').replace(
            '0.100000', '0.900000')
        results = self._parse(tmp_path, extra_log=caspt2_section)
        assert results['dipole'][0].tolist() == [0.9, 0.2, 0.3]


class TestResultParsing:
    """Result extraction follows the real cluster output formats."""

    # Three states are declared, so a log that answers for them carries an FCI
    # iteration block with states 0, 1 and 2. The gaps between these values are
    # what the NAC scaling test reads, so they stay -78.0 / -77.9 / -77.8.
    # Shape taken verbatim from a real deployment log (`casscf_energy_test.log:79-85`
    # of round tmp_20261014_185533): the state number is the SECOND column, the
    # first one counting the CI solves of the block, the `*` marks the selected
    # root and sits on the last print of its state, and the `* ci vector, state N`
    # banner that follows is the engine's own statement of how many states the
    # block answered for.
    FCI_BLOCK = (
        '    * METHOD: CASSCF      0.18\n'
        '=== FCI iteration ===\n'
        '\n'
        '      0  0 *     -78.00000000     2.78e-16      0.00\n'
        '      0  1 *     -77.90000000     5.55e-17      0.00\n'
        '      0  2 *     -77.80000000     5.55e-17      0.00\n'
        '\n'
        '     * ci vector, state   0, <S^2> = 0.0000\n'
        '       ab    -0.7071067812\n'
        '\n'
        '=== Davidson ===\n'
    )

    def _write_log(self, tmp_path, content=None):
        log = tmp_path / 'BAGEL.log'
        log.write_text(self.FCI_BLOCK if content is None else content)
        return str(log)

    def test_force_file_skips_row_index(self, tmp_path):
        (tmp_path / 'FORCE_0.out').write_text(
            '                         0                   1'
            '                   2\n'
            '     0        0.1000000000        0.2000000000'
            '  0.3000000000\n'
            '     1       -0.1000000000       -0.2000000000'
            ' -0.3000000000\n'
            + ''.join(
                f'     {i}        0.0000000000        0.0000000000'
                '  0.0000000000\n' for i in range(2, 6))
        )
        config = _make_config(profile='casscf_gradient')
        results = parse_log(self._write_log(tmp_path), config)
        assert np.allclose(results['gradient'][0][0], [0.1, 0.2, 0.3])
        assert np.allclose(results['gradient'][0][1],
                           [-0.1, -0.2, -0.3])

    def test_nac_scaled_by_energy_gap(self, tmp_path):
        # BAGEL exports the derivative coupling already divided by
        # the gap; the parser multiplies by (E_b - E_a) to match the
        # molcas CI-derivative convention
        (tmp_path / 'ENERGY.out').write_text('-78.0\n-77.9\n')
        (tmp_path / 'NACME_0_1.out').write_text(
            '                         0                   1'
            '                   2\n'
            '     0        1.0000000000        0.0000000000'
            '  0.0000000000\n'
            + ''.join(
                f'     {i}        0.0000000000        0.0000000000'
                '  0.0000000000\n' for i in range(1, 5))
            + '     5        0.0000000000        0.0000000000'
            '  3.0000000000\n'
        )
        config = _make_config(profile='casscf_nac',
                              target_nac_pair=[[0, 1]])
        results = parse_log(self._write_log(tmp_path), config)
        assert np.allclose(results['nac'][0][0], [0.1, 0.0, 0.0])
        assert np.allclose(results['nac'][0][5], [0.0, 0.0, 0.3])

    def test_nac_gap_falls_back_to_log(self, tmp_path):
        # Without ENERGY.out the log 'Energy gap' line (E_a - E_b in
        # eV) supplies the gap factor
        (tmp_path / 'NACME_0_1.out').write_text(
            '                         0                   1'
            '                   2\n'
            + ''.join(
                f'     {i}        1.0000000000        0.0000000000'
                '  0.0000000000\n' for i in range(6))
        )
        config = _make_config(profile='casscf_nac',
                              target_nac_pair=[[0, 1]])
        log = (self.FCI_BLOCK +
               '    * NACME Target states: 0 - 1\n'
               '    * Energy gap is:       -2.7211386245988 eV\n')
        results = parse_log(self._write_log(tmp_path, log), config)
        assert np.allclose(results['nac'][0][0], [0.1, 0.0, 0.0])

    def test_frequencies_from_log(self, tmp_path):
        config = _make_config(profile='casscf_frequency')
        log = (self.FCI_BLOCK +
               '    * Vibrational frequencies, IR intensities, and'
               ' corresponding cartesian eigenvectors\n'
               '      Freq (cm-1)             -718.14'
               '                0.00\n'
               '      Freq (cm-1)              419.05'
               '             1027.72\n')
        results = parse_log(self._write_log(tmp_path, log), config)
        assert results['frequencies'].tolist() == \
            [-718.14, 0.00, 419.05, 1027.72]

    def test_hessian_from_log(self, tmp_path):
        config = _make_config(profile='casscf_frequency')
        ndim = 3 * config.natom
        lines = [self.FCI_BLOCK,
                 '  ++ Hessian ++\n', '\n']
        for offset in range(0, ndim, 6):
            cols = list(range(offset, offset + 6))
            lines.append('  ' + ''.join(
                f'{c:20d}' for c in cols) + '\n')
            for row in range(ndim):
                values = ''.join(
                    f'{float(row * ndim + c):20.10f}' for c in cols)
                lines.append(f'{row:6d}{values}\n')
        lines.append('  ++ Mass Weighted Hessian ++\n')
        results = parse_log(self._write_log(tmp_path, ''.join(lines)),
                            config)
        hessian = results['hessian']
        assert hessian.shape == (ndim, ndim)
        assert hessian[0, 0] == 0.0
        assert hessian[1, 0] == float(ndim)
        assert hessian[ndim - 1, ndim - 1] == float(ndim * ndim - 1)

    def test_coordinates_read_as_angstrom(self, tmp_path):
        # The molecule block sets "angstrom": "true", so log
        # coordinates are read without any unit conversion
        config = _make_config()
        log = ''.join(
            f'  {{ "atom" : "C", "xyz" : [     {i}.000000,      '
            f'{i + 1}.000000,     {i + 2}.000000 ] }},\n'
            for i in range(1, 7))
        results = parse_log(self._write_log(tmp_path, self.FCI_BLOCK + log),
                            config)
        assert results['coordinates'][0] == [1.0, 2.0, 3.0]
        assert results['coordinates'][5] == [6.0, 7.0, 8.0]

    def _geometry_block(self, base):
        return ''.join(
            f'  {{ "atom" : "C", "xyz" : [     {base + i}.000000,      '
            f'{base + i}.000000,     {base + i}.000000 ] }},\n'
            for i in range(6))

    def test_fin_coord_from_last_geometry_block(self, tmp_path):
        # A converged optimization prints the final structure in the
        # '*** Geometry ***' block at the end of the log
        config = _make_config(profile='casscf_geometry_optimization')
        log = (self.FCI_BLOCK
               + '    * METHOD: OPTIMIZE      4.62\n'
               + self._geometry_block(1)
               + '\n  *** Geometry ***\n\n'
               + self._geometry_block(10))
        results = parse_log(self._write_log(tmp_path, log), config)
        assert results['fin_coord'][0].tolist() == [10.0, 10.0, 10.0]
        assert results['fin_coord'][5].tolist() == [15.0, 15.0, 15.0]
        assert results['opt_converge'] is True

    def test_fin_coord_falls_back_to_opt_molden(self, tmp_path):
        # Without a complete geometry block in the log the final
        # structure is recovered from opt.molden ([Atoms] Angs
        # section, sym/num/Z/x/y/z columns)
        config = _make_config(profile='casscf_geometry_optimization')
        molden = ('[Molden Format]\n'
                  '[Atoms] Angs\n'
                  + ''.join(
                      f' c       {i + 1}                    6 '
                      f'{float(i):19.10e}  0.0  0.0\n'
                      for i in range(6))
                  + '[GTO]\n')
        (tmp_path / 'opt.molden').write_text(molden)
        results = parse_log(self._write_log(tmp_path), config)
        assert results['fin_coord'][2].tolist() == [2.0, 0.0, 0.0]


class TestLogCompletion:
    """Failed or missing BAGEL runs must never parse as pass."""

    def test_missing_log_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='log file not found'):
            _check_log_completion(str(tmp_path / 'no_such.log'))

    def test_error_marker_raises_runtime_error(self, tmp_path):
        log = tmp_path / 'BAGEL.log'
        log.write_text(
            '    * METHOD: CASSCF                               0.18\n'
            '\n'
            '  ERROR: EXCEPTION RAISED:  A required keyword is '
            'missing from the input for forces:  method\n'
        )
        with pytest.raises(RuntimeError, match='required keyword'):
            _check_log_completion(str(log))

    def test_clean_log_passes(self, tmp_path):
        log = tmp_path / 'BAGEL.log'
        log.write_text('    * METHOD: CASSCF      0.18\n')
        _check_log_completion(str(log))

    def test_parse_log_rejects_failed_run(self, tmp_path):
        log = tmp_path / 'BAGEL.log'
        log.write_text(
            '  ERROR: EXCEPTION RAISED:  File not found: '
            'xmscaspt2_energy_guess.archive\n'
        )
        config = _make_config()
        with pytest.raises(RuntimeError, match='File not found'):
            parse_log(str(log), config)


class TestBackendReturnCode:
    """Non-zero runner return codes propagate as failures."""

    def test_calculate_raises_on_nonzero_returncode(
            self, tmp_path, monkeypatch):
        from .. import backend as backend_module
        from ..backend import BagelBackend

        monkeypatch.chdir(tmp_path)
        # Skip the external-program environment check and force a
        # failing runner so no BAGEL installation is needed.
        monkeypatch.setattr(
            backend_module, 'validate_environment', lambda config: None)
        monkeypatch.setattr(
            backend_module, 'get_runner',
            lambda config: (lambda cfg, calcdir: 1),
        )

        backend = BagelBackend(dict(MINIMAL_CONFIG))
        with pytest.raises(RuntimeError, match='return code 1'):
            backend.calculate(
                MINIMAL_CONFIG['atoms'], MINIMAL_CONFIG['coordinates'])


class TestASpinTheInputCannotName:
    """A mixed spin list is refused, because no input line asks for it."""

    def test_a_singlet_triplet_pair_is_refused_by_name(self):
        with pytest.raises(ValueError, match='no spin keyword'):
            _make_config(nstate=2, state_spin=[0, 2],
                         profile='casscf_gradient')

    def test_a_singlet_only_request_is_accepted(self):
        # The positive arm: the refusal is about a spin the input cannot ask
        # for, not about declaring several states.
        config = _make_config(nstate=3, state_spin=[0, 0, 0],
                              profile='casscf_gradient')
        assert config.state_spin == [0, 0, 0]


class TestStateNumberIsTheSecondColumn:
    """A row prints [ci solve] [state]; the state is column two, and the last
    print of a state is the converged one."""

    LOG = TestResultParsing.FCI_BLOCK

    def _log(self, rows, starred=-1):
        """Render FCI rows in the shape the real log prints.

        `rows` is a list of (solve, state, energy). The `*` marks the selected
        root and, measured on every block of every log in round
        tmp_20261014_185533, sits on the last print of its state - which is why
        the reader takes the last row per state rather than needing the star.
        """
        lines = ['=== FCI iteration ===\n', '\n']
        selected = starred % len(rows)
        for position, (solve, state, energy) in enumerate(rows):
            mark = ' *' if position == selected else '  '
            lines.append('  %d  %d%s  %18.10f     1.00e-16      0.00\n'
                         % (solve, state, mark, energy))
        lines.append('\n=== Davidson ===\n')
        return ''.join(lines)

    def test_energies_are_read_in_the_order_the_states_print(self, tmp_path):
        config = _make_config(profile='casscf_energy')
        log = tmp_path / 'BAGEL.log'
        log.write_text(self._log(
            [(0, 0, -78.0), (0, 1, -77.9), (0, 2, -77.8)]))
        results = parse_log(str(log), config)
        assert np.allclose(results['energy'], [-78.0, -77.9, -77.8])

    def test_a_single_state_reprinted_per_solve_keeps_the_converged_row(
            self, tmp_path):
        # Verbatim from `casscf_frequency_test.log:201-212` of round
        # tmp_20261014_142736: four CI solves of state 0, the last two already at
        # the converged value. Keying on column one read this as four states and
        # `energies[:nstate]` took row 0, -77.74685469 - 0.20 hartree off the
        # surface, while the case reported PASS.
        config = _make_config(profile='casscf_energy', nstate=1,
                              state_spin=[0])
        log = tmp_path / 'BAGEL.log'
        log.write_text(
            '=== FCI iteration ===\n'
            '\n'
            '      0  0       -77.74685469     4.96e-10      0.00\n'
            '      1  0       -77.94047011     1.78e-02      0.00\n'
            '    - linear dependency detected:    1 /    3    min eigenvalue:'
            '     2.0384e-13\n'
            '    * Using canonical orthogonalization due to linear dependency\n'
            '\n'
            '      2  0       -77.94689388     3.14e-08      0.00\n'
            '      3  0 *     -77.94689388     7.85e-17      0.00\n'
            '\n'
            '     * ci vector, state   0, <S^2> = 0.0000\n'
            '\n'
            '=== Davidson ===\n')
        results = parse_log(str(log), config)
        assert results['energy'].tolist() == [-77.94689388]
        assert abs(results['energy'][0] + 77.74685469) > 0.1

    def test_a_three_state_block_is_read_verbatim(self, tmp_path):
        # `casscf_energy_test.log:79-83` of round tmp_20261014_185533, the shape
        # that the previous column reading refused as "root indices [0, 0, 0]" and
        # that made 7 cases x 3 modes fail on the engine run.
        config = _make_config(profile='casscf_energy', nstate=3)
        log = tmp_path / 'BAGEL.log'
        log.write_text(
            '  === FCI iteration ===\n'
            '\n'
            '      0  0 *     -77.92051100     2.78e-16      0.00\n'
            '      0  1 *     -77.80674018     5.55e-17      0.00\n'
            '      0  2 *     -77.80062527     5.55e-17      0.00\n'
            '\n'
            '     * ci vector, state   0, <S^2> = 0.0000\n'
            '       ab    -0.7071067812\n'
            '       ba    -0.7071067812\n')
        results = parse_log(str(log), config)
        assert results['energy'].tolist() == [-77.92051100, -77.80674018,
                                              -77.80062527]

    def test_a_block_whose_state_numbers_jump_is_refused(self, tmp_path):
        # States 0, 1, 3 in one block: the row count still says three, so a
        # position-packed read would hand the third energy to the wrong state
        # and nothing downstream could see it.
        config = _make_config(profile='casscf_energy')
        log = tmp_path / 'BAGEL.log'
        log.write_text(self._log(
            [(0, 0, -78.0), (0, 1, -77.9), (0, 3, -77.8)]))
        with pytest.raises(RuntimeError, match='state numbers'):
            parse_log(str(log), config)

    def test_a_block_whose_states_are_out_of_order_is_refused(self, tmp_path):
        config = _make_config(profile='casscf_energy')
        log = tmp_path / 'BAGEL.log'
        log.write_text(self._log(
            [(0, 1, -78.0), (0, 0, -77.9), (0, 2, -77.8)]))
        with pytest.raises(RuntimeError, match='state numbers'):
            parse_log(str(log), config)

    def test_fewer_states_than_declared_are_refused(self, tmp_path):
        # Three states declared, one printed: the shortfall used to be filled with
        # zeros, which reads as flat surfaces to a crossing search.
        config = _make_config(profile='casscf_energy', nstate=3)
        log = tmp_path / 'BAGEL.log'
        log.write_text(self._log([(0, 0, -78.0)]))
        with pytest.raises(RuntimeError, match='no state energies'):
            parse_log(str(log), config)

    def test_a_log_and_an_energy_file_answering_for_nothing_is_refused(
            self, tmp_path):
        config = _make_config(profile='casscf_energy')
        log = tmp_path / 'BAGEL.log'
        log.write_text('    * METHOD: CASSCF      0.18\n')
        with pytest.raises(RuntimeError, match='no state energies'):
            parse_log(str(log), config)


class TestEnergyOutFallbackIsNamed:
    """`ENERGY.out` has a fixed name, so a value taken from it is not
    attributable to the run that read it."""

    # `xmscaspt2_frequency_test.log` of round tmp_20261014_185533 is a finite
    # difference job: it prints `Hessian evaluation ( k / 18)` rows and not one
    # energy line, so its recorded energy came out of the shared ENERGY.out.
    # Round tmp_20261015_100320 says whose: that job ran at 11:10 while the file
    # was last written at 11:15:50, and its three rows are the energies seven
    # sibling cases print for the same input - the bagel cases share one geometry
    # (verified: maximum coordinate difference 0.0 between the frequency input and
    # the energy/gradient/meci/nac inputs), which is why the foreign number looked
    # right and the case reported PASS.
    FD_LOG = (
        '    * METHOD: MOLECULE                             0.05\n'
        '  The Hessian will be computed with central gradient differences'
        ' (analytical gradients)\n'
        '       - Hessian evaluation ( 1 / 18)              1.23\n'
        '  ++ Hessian ++\n'
        '                         0                   1                   2\n'
        '     0        0.0100000000        0.0000000000        0.0000000000\n'
        '     1        0.0000000000        0.6600000000        0.0000000000\n'
        '     2        0.0000000000        0.0000000000        1.0800000000\n'
    )

    def test_a_log_answering_for_nothing_reads_the_shared_file_and_says_so(
            self, tmp_path):
        config = _make_config(profile='xmscaspt2_frequency', nstate=1,
                              state_spin=[0], method='xmscaspt2')
        (tmp_path / 'ENERGY.out').write_text('-77.08098671\n')
        log = tmp_path / 'BAGEL.log'
        log.write_text(self.FD_LOG)
        with pytest.warns(UserWarning, match='ENERGY.out'):
            results = parse_log(str(log), config)
        assert results['energy'].tolist() == [-77.08098671]

    def test_a_log_that_answers_is_not_warned_about_the_shared_file(
            self, tmp_path):
        # The negative control: the warning must fire on the fallback path and
        # nowhere else, or it becomes noise that nobody reads.
        config = _make_config(profile='xmscaspt2_frequency', nstate=1,
                              state_spin=[0], method='xmscaspt2')
        (tmp_path / 'ENERGY.out').write_text('-77.08098671\n')
        log = tmp_path / 'BAGEL.log'
        log.write_text('    * MS-CASPT2 energy : state    0'
                       '     -76.90000000\n')
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            results = parse_log(str(log), config)
        assert results['energy'].tolist() == [-76.9]
