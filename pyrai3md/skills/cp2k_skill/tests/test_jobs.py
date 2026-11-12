"""
Unit tests for cp2k_skill's job planner (`service/jobs.py`).

One CP2K run answers one spin manifold, so a request naming both singlet and
triplet excited states has to become two runs of the same reference. These tests
hold the two halves of that statement: the split happens exactly when the
request needs it, and a request that does not need it is untouched - same config
object, same project name, same input bytes - which is what makes every shipped
cp2k case the identity arm for this branch.

The engine facts behind it are measured, not reasoned: an A/B in the test environment
machine (CP2K 2025.2) took one input differing only in `RKS_TRIPLETS` and got
`R-TDDFPT states of multiplicity 3` with no singlet table at all from `T`, and
`multiplicity 1` from `F`.

Run: pytest cp2k_skill/tests/test_jobs.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder.input_builder import build_input
from ..service.jobs import Job, job_nstates, manifolds_needed, plan_jobs
from ..service.parser import state_map
from ..service.workflow import compile_workflow

ATOMS = ['C', 'C', 'H', 'H', 'H', 'H']
COORDS = [
    [-1.120191050, 0.000000000, -2.661941306],
    [-1.120191050, 0.000000000, -1.331783306],
    [-1.120191050, 0.923341000, -3.234891306],
    [-1.120191050, -0.923341000, -3.234891306],
    [-2.043532050, 0.000000000, -0.758833306],
    [-0.196850050, 0.000000000, -0.758833306],
]


def _tddft_config(profile='tddft_energy', **kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 3,
        'state_spin': [0, 0, 0],
        'target_states': [0, 1],
        'profile': profile,
        'method': 'tddft',
        'functional': 'pbe0',
        'project': 'eth',
    }
    raw.update(kwargs)
    return validate(normalize(build_config(raw)))


def _tddfpt_block(config, profile):
    """The `&TDDFPT` text this job's input would carry."""
    text = build_input(config, compile_workflow(profile))
    return text.split('&TDDFPT')[1].split('&END TDDFPT')[0]


def _keyword(block, name):
    lines = [line.strip() for line in block.splitlines()
             if line.strip().startswith(name)]
    return lines[0] if lines else None


class TestTheSplitHappensOnlyWhenTheRequestNeedsIt:
    """One manifold named is one job with the caller's own config."""

    def test_an_all_singlet_request_is_one_job_and_the_original_config(self):
        config = _tddft_config(state_spin=[0, 0, 0], target_states=[0, 1, 2])
        jobs = plan_jobs(config, compile_workflow(config.profile))
        assert len(jobs) == 1
        # Not a copy: the shipped cases must not acquire a second project name,
        # a retuned NSTATES or a zeroed root count, and `is` is how that is
        # pinned without reading every field.
        assert jobs[0].config is config
        assert jobs[0].two_s is None

    def test_an_all_singlet_input_is_byte_identical_to_the_unsplit_one(self):
        # The identity arm in bytes rather than in pointers: whatever the planner
        # decides, the input a single-manifold request writes cannot change.
        config = _tddft_config(profile='tddft_gradient',
                               state_spin=[0, 0, 0], target_states=[0, 1, 2])
        job = plan_jobs(config, compile_workflow(config.profile))[0]
        assert (_tddfpt_block(job.config, config.profile)
                == _tddfpt_block(config, config.profile))

    def test_a_ground_plus_triplet_request_is_still_one_run(self):
        # The shape round tmp_20261014_185533 ran: S0 and one triplet. One
        # manifold is named, so one run answers it and the ground state comes
        # from that same run's SCF.
        config = _tddft_config(nstate=2, state_spin=[0, 2],
                               target_states=[0, 1])
        jobs = plan_jobs(config, compile_workflow(config.profile))
        assert len(jobs) == 1
        assert jobs[0].config.project == 'eth'
        assert 'RKS_TRIPLETS T' in _tddfpt_block(jobs[0].config,
                                                 config.profile)

    def test_a_dft_request_names_no_manifold_and_stays_one_job(self):
        raw = {'atoms': ATOMS, 'coordinates': COORDS, 'charges': 0,
               'multiplicity': 1, 'nstate': 1, 'state_spin': [0],
               'profile': 'dft_energy', 'method': 'dft',
               'functional': 'pbe0', 'project': 'eth'}
        config = validate(normalize(build_config(raw)))
        assert manifolds_needed(config) == []
        assert len(plan_jobs(config, compile_workflow('dft_energy'))) == 1


class TestATwoManifoldRequestBecomesTwoRuns:
    """Singlets and triplets apart, each addressing only its own manifold."""

    @pytest.fixture()
    def split(self):
        config = _tddft_config(profile='tddft_gradient',
                               state_spin=[0, 0, 0, 2], nstate=4,
                               target_states=[0, 1, 2, 3])
        jobs = plan_jobs(config, compile_workflow(config.profile))
        return config, jobs

    def test_each_manifold_gets_its_own_project(self, split):
        config, jobs = split
        assert [job.two_s for job in jobs] == [0, 2]
        assert [job.config.project for job in jobs] == ['eth_S', 'eth_T']
        # CP2K names every file it writes after PROJECT, so the two runs cannot
        # read each other's leftovers out of one calculation directory.
        assert all(job.config is not config for job in jobs)

    def test_each_run_writes_the_letter_of_its_own_manifold(self, split):
        config, jobs = split
        letters = [_keyword(_tddfpt_block(job.config, config.profile),
                            'RKS_TRIPLETS') for job in jobs]
        assert letters == ['RKS_TRIPLETS F', 'RKS_TRIPLETS T']

    def test_each_run_asks_for_its_own_number_of_roots(self, split):
        config, jobs = split
        states = [_keyword(_tddfpt_block(job.config, config.profile),
                           'NSTATES') for job in jobs]
        # S1 and S2 in one run, T1 in the other. The shared declaration says
        # three excited rows in total, and a run that holds only triplets has no
        # use for the singlet count.
        assert states == ['NSTATES 2', 'NSTATES 1']
        assert [job_nstates(config, 0), job_nstates(config, 2)] == [2, 1]

    def test_each_runs_list_stays_inside_its_manifold(self, split):
        config, jobs = split
        lists = [_keyword(_tddfpt_block(job.config, config.profile), 'LIST')
                 for job in jobs]
        assert lists == ['LIST 1 2', 'LIST 1']
        rows = [state_map.requested_rows(job.config) for job in jobs]
        assert rows == [[1, 2], [3]]

    def test_a_triplet_row_does_not_steal_the_singlets_number(self, split):
        # Both manifolds print a root numbered 1, and the framework rows differ
        # (1 and 2 are singlets, 3 is the triplet). The LIST is per run precisely
        # so that the number 1 in the triplet input means the triplet's first
        # root; the row it belongs to is decided by the run, not by the number.
        config, jobs = split
        singlet_job, triplet_job = jobs
        assert state_map.list_numbers(singlet_job.config) == [1, 2]
        assert state_map.list_numbers(triplet_job.config) == [1]
        assert state_map.spin_of_row(config, 1) == 0
        assert state_map.spin_of_row(config, 3) == 2
        assert state_map.job_manifold(singlet_job.config) == 0
        assert state_map.job_manifold(triplet_job.config) == 2

    def test_an_explicit_nstates_bigger_than_anything_travels_to_both(
            self, split):
        # The only reading of an NSTATES that exceeds the total declared roots is
        # that the caller asked for more than the rows name; the per-manifold
        # count must not silently override it.
        config, _ = split
        widened = config.model_copy(deep=True)
        widened.calculation.tddft_nstates = 10
        jobs = plan_jobs(widened, compile_workflow(config.profile))
        assert [job.config.tddft_nstates for job in jobs] == [10, 10]

    def test_the_split_is_a_copy_and_leaves_the_request_alone(self, split):
        config, jobs = split
        assert config.ntrip_root == 1 and config.nsing_root == 2
        # Zeroing the other manifold's count is how the split tells the builder
        # and the parse which rows this run addresses; it happens on the variant.
        assert jobs[0].config.ntrip_root == 0
        assert jobs[1].config.nsing_root == 0
        assert state_map.excited_rows(jobs[0].config) == {0: [1, 2], 2: [3]}


class TestCombinationsThatHaveNoOwnerAreRefused:
    """A split only makes sense for channels labelled by state rows or pairs."""

    def test_a_two_manifold_optimization_is_refused_by_name(self):
        config = _tddft_config(profile='tddft_opt_min',
                               state_spin=[0, 0, 2], target_states=[0, 1, 2])
        with pytest.raises(ValueError, match='tddft_opt_min'):
            plan_jobs(config, compile_workflow(config.profile))

    def test_a_two_manifold_md_is_refused_by_name(self):
        config = _tddft_config(profile='tddft_md', state_spin=[0, 0, 2],
                               target_states=[0, 1, 2])
        with pytest.raises(ValueError, match='one run'):
            plan_jobs(config, compile_workflow(config.profile))

    def test_a_split_request_never_reaches_the_builders_refusal(self):
        # `require_addressable` keeps its engine-abort knowledge as an invariant:
        # it fires on an unsplit two-manifold LIST, which is exactly what the
        # planner prevents, and it stays silent on each job's own variant.
        config = _tddft_config(profile='tddft_gradient',
                               state_spin=[0, 0, 2], target_states=[0, 1, 2])
        with pytest.raises(ValueError, match='ONE run'):
            state_map.require_addressable(config)
        for job in plan_jobs(config, compile_workflow(config.profile)):
            state_map.require_addressable(job.config)


class TestPlannerShape:
    """The two small helpers the planner is built from."""

    def test_manifolds_are_read_from_declared_rows_not_from_targets(self):
        # A run that declares three singlets and one triplet and follows only the
        # ground state still needs both manifolds solved: energies are wanted for
        # every declared row, forces for the targeted ones.
        config = _tddft_config(state_spin=[0, 0, 2, 2], nstate=4,
                               target_states=[0])
        assert manifolds_needed(config) == [0, 2]
        jobs = plan_jobs(config, compile_workflow(config.profile))
        assert [state_map.requested_rows(job.config) for job in jobs] == [[], []]

    def test_a_job_is_a_named_tuple_with_the_manifold_and_the_config(self):
        config = _tddft_config()
        job = Job(None, config)
        assert job.two_s is None
        assert job.config is config
