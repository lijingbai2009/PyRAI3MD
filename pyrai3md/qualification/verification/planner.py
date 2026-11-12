"""
======================================================================
This module plans the main framework verification tests (split out
of the qualification manager).

It mirrors the planner -> test case file management pattern of
certification and benchmark: the planner discovers the test case
files and returns task descriptors, while the qualification manager
only aggregates the three plans and the runner executes them
unchanged.

The verification tests are the unittest files shipped in this
directory (test_*.py). They run in-process in a local round and as one
job in a slurm round: framework-level regressions must be
host-independent, and the job is only a different place to run the same
process.

Reference direction: planner -> verification test files only (no
manager imports, so the manager can import this module at load
time without a cycle).

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os


def plan_main_tests(config=None):
    """Plan the main framework tests.

    Each file is one in-process unittest task in a local round and one
    row of the unit-test job in a slurm round.

    Discovers every test_*.py file shipped in this directory and returns
    one local task per file. When ``config`` is given, the files in the
    sibling ``integration`` package are appended as well; that planner
    marks them skipped (folding to one closed row) unless
    ``test_integration`` or ``_runall`` opens them.

    Parameters
    ----------
    config : dict, optional
        Parsed pmd_env.yaml configuration. None plans only the unit
        tests, which is what a direct planner call or an old caller
        expects.

    Returns
    -------
    list of dict
        List of main test task descriptors.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = []

    if os.path.isdir(tests_dir):
        for fname in sorted(os.listdir(tests_dir)):
            if fname.startswith('test_') and fname.endswith('.py'):
                tasks.append({
                    'name': fname[:-3],
                    'file': os.path.join(tests_dir, fname),
                    'mode': 'local',
                    'job_group': 'main_unit',
                })

    if config is not None:
        from ..integration.planner import plan_integration_tests
        tasks.extend(plan_integration_tests(config))

    return tasks
