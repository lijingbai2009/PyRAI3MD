#!/usr/bin/env python3
"""
======================================================================
This module is the benchmark analysis tool for PyRAI3MD.

Thin user-facing wrapper around the qualification benchmark task
modules: it re-exports the evaluation entry points of
pyrai3md.qualification.benchmark.fssh and owns the command-line
entry point (the task modules themselves are programmatic only:
the planner imports their input builders and the qualification
runner imports evaluate_run). This tool is the external one-way
analysis entry point for secondary verification only: it reads an
already-computed benchmark run directory, analyzes the metrics and
renders the population plot, and never launches calculations.
The pyrai3md test qualification flow never calls this script: the
runner performs the complete calibration pipeline (planning,
compute, in-process analysis and plotting) itself.
The wrapper routes the run directory to the task module named by
the case YAML (the fssh, gsh and lzsh calibrations analyze the
reference and trajectory populations against the exact quantum
reference; the ktdc calibration reuses the fssh evaluator; the
perf case compares the two backend legs against each other) and
reports the calibration metrics.

This tool is located in tools/ and is run directly:
    python3 ./tools/benchmark.py <run_dir>
        # evaluate a computed benchmark/<case_key> directory

It does not participate in test planning or execution logic; the
qualification runner calls the task module's evaluate_run
in-process right after the compute task finished.

Aug 31 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import sys

import yaml

# Direct script launch puts tools/ on sys.path, not the project
# root: insert it so the source-tree pyrai3md package wins over an
# installed snapshot (same source-tree-first semantics as the
# qualification cwd config layer).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pyrai3md.qualification.benchmark.fssh import (  # noqa: E402
    collect_populations,
    emit,
    ensemble_average,
    evaluate_run,
    parse_pop,
    parse_reference_population,
)
from pyrai3md.qualification.benchmark import (  # noqa: E402
    fssh as _fssh_module,
    gsh as _gsh_module,
    ktdc as _ktdc_module,
    lzsh as _lzsh_module,
    perf as _perf_module,
)

__all__ = [
    'collect_populations',
    'emit',
    'ensemble_average',
    'evaluate_run',
    'main',
    'parse_pop',
    'parse_reference_population',
]


def _route_evaluate(run_dir):
    """Evaluate a run directory through its own task module.

    The case YAML inside the run directory names the task (its
    'case' key follows the '<task>[_<method>]' convention); the
    perf case routes to the perf task module, the gsh cases to the
    gsh task module, the lzsh cases to the lzsh task module and the
    ktdc cases to the ktdc task module (all three reuse the fssh
    population analysis), everything else to the fssh calibration
    module (the historical default).
    """
    try:
        case_path = _fssh_module._find_case_yaml(run_dir)
        with open(case_path, 'r', encoding='utf-8') as handler:
            case_name = str(
                (yaml.safe_load(handler) or {}).get('case', 'fssh'))
    except Exception:  # noqa: BLE001 - fall back to the fssh route
        case_name = 'fssh'
    if case_name == 'perf' or case_name.startswith('perf_'):
        return _perf_module.evaluate_run(run_dir)
    if case_name == 'gsh' or case_name.startswith('gsh_'):
        return _gsh_module.evaluate_run(run_dir)
    if case_name == 'lzsh' or case_name.startswith('lzsh_'):
        return _lzsh_module.evaluate_run(run_dir)
    if case_name == 'ktdc' or case_name.startswith('ktdc_'):
        return _ktdc_module.evaluate_run(run_dir)
    return evaluate_run(run_dir)


def main(argv=None):
    """Entry point: evaluate one computed benchmark run directory."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('usage: python3 ./tools/benchmark.py <run_dir>')
        return 2
    return emit(_route_evaluate(args[0]))


if __name__ == '__main__':
    sys.exit(main())
