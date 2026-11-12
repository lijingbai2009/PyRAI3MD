"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 14 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module decides HOW MANY CP2K runs one evaluation needs. It reads the
request and answers with the jobs; it runs nothing and writes nothing, so the
flow stays in `Cp2kBackend.calculate` and the split is testable on its own.

Why a split exists at all, measured on the test environment (CP2K 2025.2,
`src/qs_tddfpt2_methods.F`): a run solves ONE spin manifold. Without `&SOC` the
engine calls `tddfpt_energies` once (line 421) with a `mult` that the single
boolean `RKS_TRIPLETS` decides outright (lines 294-297), and it prints one
excitation table (line 1190). An A/B on the same input, differing in that one
letter and carrying no `&SOC`, showed it directly: `T` printed only
`R-TDDFPT states of multiplicity 3` and the word `singlet` appeared zero times,
`F` printed only `multiplicity 1`. The only code in the tree that solves both
manifolds is `tddfpt_soc_energies` (its `! First multiplicity` /
`! Second multiplicity` blocks), which exists to build the spin-orbit matrix -
a different task, and one that still hands back forces for only one manifold
(line 1255: "only the values of one multiplicity will be passed back for force
calculations").

So the two surfaces of different spin come from two runs of the same reference,
each asking for its own manifold, and `state_map` stays the one place where
CP2K's per-manifold numbering meets the framework's single `(nstate,)` axis.

A request that names one manifold gets ONE job carrying the ORIGINAL config: no
copy, no renamed project, no retuned `NSTATES`. That is what makes every shipped
cp2k case the identity test for this branch, and it is structural rather than a
test that happens to pass.

"""

from typing import NamedTuple, Optional

from .parser import state_map

# The framework's 2S spelling for the two manifolds CP2K's RKS linear response
# can serve from a closed-shell reference.
SINGLET = 0
TRIPLET = 2

# Per-job project suffix. CP2K names every file it writes after `PROJECT`
# (`<project>.log`, `<project>-force-1_0.xyz`, `<project>-td_forces-1_0.tdfrc`,
# `<project>-RESTART.wfn`), so a distinct project per job is what keeps two runs
# in one calculation directory from reading each other's leftovers.
PROJECT_SUFFIX = {SINGLET: '_S', TRIPLET: '_T'}

# Steps whose results have a single owner per run: the returned geometry, the
# cell that goes with it, and the convergence verdict all come out of one
# document, and a second run over the same rows would be asked to agree with the
# first on something neither was asked to produce. A two-manifold request on such
# a profile is refused where it is made instead of merged by guesswork.
SINGLE_OWNER_STEPS = ('optimize', 'md')


class Job(NamedTuple):
    """One CP2K run.

    Attributes
    ----------
    two_s : int or None
        The manifold this job solves, in the framework's 2S spelling, or None
        when the request needed no split and `config` is the caller's own.
    config : SkillConfig
        The configuration to build, run and parse this job with. The original
        object for a single job, a variant per manifold for a split.
    """

    two_s: Optional[int]
    config: object


def manifolds_needed(config) -> list:
    """The 2S values with DECLARED excited rows, ascending.

    Declared, not requested: an energy run wants every state it named even when
    it follows one of them, so the job count comes from `state_spin` while the
    per-job force list stays driven by `target_states` (see
    `state_map.requested_rows`).
    """
    return sorted(state_map.excited_rows(config))


def job_nstates(config, two_s: int) -> int:
    """How many roots this job has to solve.

    Each manifold asks for its OWN root count now that the manifolds are solved
    apart: `NSTATES` is per run, and a run that holds only triplets has no use for
    the singlet roots the shared declaration asked for. The value normalize
    derives (`nstate - 1`, the largest targeted index, the triplet count - see
    `schema/normalize.py`) is the TOTAL across manifolds, so it is not what a
    single-manifold run should ask for.

    An `NSTATES` bigger than every declared root put together is not a derivation
    though - nobody declares more excited states than exist - so that one is a
    deliberate override (asking for more roots than the rows name, to relax the
    Davidson space), and it travels to both jobs untouched.
    """
    groups = state_map.excited_rows(config)
    total = sum(len(rows) for rows in groups.values())
    explicit = int(getattr(config, 'tddft_nstates', 0) or 0)
    if explicit > total:
        return explicit
    return len(groups.get(two_s, []))


def plan_jobs(config, workflow) -> list:
    """The jobs this evaluation needs, in the order they must run.

    Parameters
    ----------
    config : SkillConfig
        The normalized, validated skill configuration.
    workflow : WorkflowGraph
        The compiled workflow, consulted for steps whose results have a single
        owner per run.

    Returns
    -------
    list of Job
        One job when the request names at most one spin manifold (carrying the
        caller's own config), two when it names both singlet and triplet excited
        rows.

    Raises
    ------
    ValueError
        When a two-manifold request rides a profile whose channels belong to one
        run (an optimization or an MD). Stress never reaches this point on a
        TDDFT run - `schema/validate.py::validate_stress` refuses both of its
        methods against the engine's own answer (the analytical tensor is
        declined, the numerical route deforms the cell the run cannot report on),
        so there is no combination left for this function to turn away.
    """
    manifolds = manifolds_needed(config)
    if len(manifolds) < 2:
        return [Job(None, config)]

    single_owner = [step for step in SINGLE_OWNER_STEPS
                    if workflow.has_step(step)]
    if single_owner:
        raise ValueError(
            'cp2k: a two-manifold request (2S %r) was compiled onto the %r '
            'profile, whose %s results come out of one run\'s documents (the '
            'returned geometry, its cell, the convergence verdict). Splitting '
            'the excited states over two runs would leave those channels '
            'unowned, so the combination is refused here rather than merged by '
            'guesswork' % (manifolds, workflow.profile,
                          '/'.join(single_owner)))

    jobs = []
    for two_s in manifolds:
        variant = config.model_copy(deep=True)
        variant.calculation.project = (
            f'{config.project}{PROJECT_SUFFIX[two_s]}')
        variant.calculation.tddft_nstates = job_nstates(config, two_s)
        # The other manifold's count goes to zero on this variant. Two existing
        # consumers then follow without a new keyword anywhere: the builder's
        # `RKS_TRIPLETS {T|F}` letter is read from `ntrip_root`, and
        # `state_map.job_manifold` reads both counts to decide which rows this run
        # addresses - so the `&PRINT &FORCES LIST` numbers and the rows the parse
        # is held to answer both land inside this job's own manifold, which is the
        # one thing a single run's LIST cannot do.
        if two_s == SINGLET:
            variant.system.ntrip_root = 0
        else:
            variant.system.nsing_root = 0
        jobs.append(Job(two_s, variant))
    return jobs
