"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

The two spellings of each compute resource (#306).

A skill reads `thread`/`nproc`; the framework forwards `lammps_thread`
and `lammps_nproc` from the &lammps section. Both are accepted, and the
larger of a pair is what the run gets, so neither spelling can shrink a
request made in the other. The resolved value lives in `thread`/`nproc`,
which is all the templates and the resident service ever read.

Run: pytest skills/lammps_skill/tests/test_resources.py -v
"""

from ..schema.adapter import build_config, reject_retired_resources
from ..schema.config import Environment


def _env(**overrides):
    """One Environment resolved through the schema fold."""
    return Environment(**overrides)


def test_defaults_are_one():
    assert (_env().thread, _env().nproc) == (1, 1)


def test_the_larger_spelling_wins_for_threads():
    assert _env(lammps_thread=8).thread == 8
    assert _env(thread=8).thread == 8
    assert _env(thread=8, lammps_thread=2).thread == 8
    assert _env(thread=2, lammps_thread=8).thread == 8


def test_the_larger_spelling_wins_for_ranks():
    assert _env(lammps_nproc=4).nproc == 4
    assert _env(nproc=4).nproc == 4
    assert _env(nproc=4, lammps_nproc=2).nproc == 4
    assert _env(nproc=2, lammps_nproc=4).nproc == 4


def test_both_spellings_stay_readable_on_the_model():
    # The framework spelling is kept on the model (nothing silently
    # deletes a user setting); consumers read the folded name.
    env = _env(lammps_thread=3, lammps_nproc=5)
    assert env.lammps_thread == 3 and env.lammps_nproc == 5
    assert (env.thread, env.nproc) == (3, 5)


def test_a_retired_spelling_is_refused_before_the_filter():
    import pytest

    with pytest.raises(ValueError, match='num_thread'):
        build_config({'skill_name': 'lammps_skill', 'num_thread': 4})
    with pytest.raises(ValueError, match='num_thread'):
        reject_retired_resources({'num_thread': 4})


def test_the_accepted_names_are_not_refused():
    reject_retired_resources({'thread': 2, 'lammps_thread': 4,
                              'nproc': 1, 'lammps_nproc': 2})
    reject_retired_resources({})
