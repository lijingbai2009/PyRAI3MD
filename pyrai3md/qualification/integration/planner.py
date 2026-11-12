"""
======================================================================
Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

"""Plan the integration test files.

Integration tests are the checks that load every shipped example or
skill and compare the carriers against each other. They are not unit
tests: they are slower and they depend on the whole input tree, so the
top-level ``test_integration`` switch decides whether this round runs
them. The unit suite in ``verification`` always runs.

The master ``_runall`` switch sits above ``test_integration``: ``true``
forces integration on, ``false`` forces it off, and ``condition`` lets
``test_integration`` answer. A disabled integration test is still
planned, as a skipped task, so the report names the switch that closed
it instead of omitting the coverage silently.
"""

import os

from ..manager import _coerce_switch, _runall_mode

# Tests that partition their own work: file stem -> number of parts. The
# planner emits one task per part and tells the file which part it owns
# through PRAI3MD_INTEGRATION_PART; the variable is unset when the file is
# run directly (pytest, unittest), so a bare run still covers every case.
# The count is the single source: the planner enumerates every index, so a
# missing part cannot pass unnoticed.
SHARDED_TESTS = {
    # 213 shipped examples x 3 input modes x 2 contract levels took 16.5
    # min on one core on 2026-11-12; it was the round's whole tail.
    'test_integration_unification': 8,
}


def integration_enabled(config):
    """Resolve the integration switch under the global master.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    bool
        True when this round runs the integration tests.
    """
    master = _runall_mode(config)
    if master is False:
        return False
    if master is True:
        return True
    value = _coerce_switch(
        config.get('test_integration', False), 'test_integration', False)
    return value is True


def plan_integration_tests(config):
    """Plan one local task per integration test file.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    list of dict
        Integration test task descriptors. When the switch is off, each
        task carries ``skip`` and the switch name as ``skip_reason``; the
        runner folds them into one closed row.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    enabled = integration_enabled(config)
    tasks = []
    for fname in sorted(os.listdir(tests_dir)):
        if not (fname.startswith('test_') and fname.endswith('.py')):
            continue
        stem = fname[:-3]
        parts = int(SHARDED_TESTS.get(stem, 1))
        for part in range(parts):
            task = {
                'name': stem if parts == 1 else '%s_%d' % (stem, part),
                'file': os.path.join(tests_dir, fname),
                'mode': 'local',
                # One job per test file (or per part of one): the grouped
                # job runs its tasks in a serial loop on one core, and these
                # files are the round's longest tail (1.5-4.5 min each,
                # measured 2026-11-12), so a single group serialised them
                # into ~30 min on one node while the rest of the cluster
                # idled.
                'job_group': stem if parts == 1 else '%s_%d' % (stem, part),
            }
            if parts > 1:
                task['env'] = {
                    'PRAI3MD_INTEGRATION_PART': '%d/%d' % (part, parts)}
            if not enabled:
                task['skip'] = True
                task['skip_reason'] = 'disabled by test_integration'
            tasks.append(task)
    return tasks
