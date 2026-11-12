"""
======================================================================
This module provides the traditional CLI entry point and the main
orchestration pipeline for PyRAI3MD.

It handles:
1. CLI entry function (processing sys.argv)
2. Orchestration flow: Interface -> Adapter -> Planning -> Merge
   -> Normalize -> Validate -> Build State -> Execute

The orchestration functions are reused by agent.py for the AI agent
entry point.

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import sys
import time

from .interface.parser import parse_input
from .schema.adapter import build_config
from .schema.normalize import normalize
from .schema.validate import validate
from .data.build import build_state
from .utilities.serialization import json_safe, write_json
from .utilities.logo import print_logo


def main():
    """CLI entry point for PyRAI3MD.

    Usage:
        pyrai3md input [--info]
        pyrai3md test
    """
    print(print_logo(), flush=True)

    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]

    # Forward test command to qualification runner
    if command == 'test':
        from .qualification.runner import cli_entry
        cli_entry()
        return

    # Traditional run mode
    input_file = command
    info_only = '--info' in sys.argv

    run(input_file, info_only=info_only)


def run(input_file, info_only=False):
    """Main execution pipeline.

    Flow:
        input file -> parse_input (Interface)
        -> build_config (Adapter, includes Environment init)
        -> normalize -> validate (Schema)
        -> build_state (PyRAIState with static task queue, no backends)
        -> execute (Workflow layer: Calculator.initialize() -> run())

    Parameters
    ----------
    input_file : str
        Path to the input file.
    info_only : bool, optional
        If True, only print parsed info without running.

    Returns
    -------
    Result or None
        Calculation results, or None if info_only.
    """
    # Additive runtime branch: .rtt files are remote per-term worker
    # payloads (docs R3, r3e). They execute exactly ONE backend term
    # and write term_outcome.json; they never run the whole
    # calculator.run + composite sum + set_result, so they bypass the
    # .rt runtime path entirely.
    if str(input_file).lower().endswith('.rtt'):
        from .workflow.dispatch.term_worker import run_term_worker
        return run_term_worker(input_file)

    # Additive runtime branch: .rt files are serialized PyRAIState
    # runtimes (restart/remote dispatch). They bypass the input
    # pipeline entirely and feed the same execute() path; the input
    # pipeline below is unchanged.
    if str(input_file).lower().endswith('.rt'):
        return _execute_runtime_file(input_file)

    # Wall-clock start of the whole run: the reported total time
    # covers input parsing, environment initialization and execute.
    run_start = time.time()

    # Step 1: Interface -- parse input file
    raw_config = parse_input(input_file)

    # Step 2: Build PyRAIConfig (includes Environment init)
    config = build_config(raw_config)

    # Step 3: Normalize -- compute derived quantities
    config = normalize(config)

    # Step 4: Validate -- check consistency
    config = validate(config)

    # Step 5: Build PyRAIState (static task queue; backend instances
    # are created at runtime by Calculator.initialize())
    state = build_state(config, raw_config=raw_config)

    # Step 6: Execute or print info
    if info_only:
        print_state_info(state)
        return None

    # A read run returns its frames and writes nothing: no run log, no
    # data artifact, no result file. The caller owns persistence.
    from .workflow.read import is_read_jobtype
    if is_read_jobtype(state):
        from .workflow import execute
        return execute(state)

    return run_with_logger(state, start_time=run_start)


def _execute_runtime_file(runtime_file):
    """Load a .rt runtime file and execute the restored state.

    The StateSerializer is the only source of the PyRAIState here;
    Interface/Schema/Planning are bypassed by design. The restored
    state runs through the unchanged workflow execution path, and
    the results are written to result.json in the current directory
    for collection by remote dispatchers.

    Parameters
    ----------
    runtime_file : str
        Path to the .rt runtime file.

    Returns
    -------
    Result or PyRAIState
        Calculation results from the restored runtime.
    """
    from .data.runtime.serializer import load_state

    run_start = time.time()
    state, runtime_idx = load_state(runtime_file)

    result = run_with_logger(state, runtime_idx=runtime_idx,
                              start_time=run_start)

    _write_result_json(result, state)
    return result


def run_with_logger(state, runtime_idx=None, start_time=None):
    """Execute the state through the workflow layer with run logging.

    Creates the RunLogger (<title>.log in the current directory),
    writes the header (start time, logo, input parameters and skill
    info) before execution and always writes the tail line (end
    time, wall time, status), also when the execution raises.

    Parameters
    ----------
    state : PyRAIState
        Fully initialized state.
    runtime_idx : str or None
        Runtime execution suffix for restored runtimes.
    start_time : float or None
        Wall-clock start of the whole run (run() entry); defaults to
        the RunLogger construction time when omitted.

    Returns
    -------
    Result or PyRAIState
        Calculation results.
    """
    from .workflow import AdaptiveSampling, execute, resolve_workflow
    from .kernel.logger import RunLogger, stop_code_note

    workflow_cls = resolve_workflow(state)

    # The framework log lives in the run's scratch root when one is set,
    # and beside the launch directory otherwise (#306). Adaptive sampling
    # is its own working directory: its log belongs beside round-*, not
    # in the project directory that holds the input.
    if workflow_cls is AdaptiveSampling:
        log_dir = AdaptiveSampling.run_root(state)
    else:
        log_dir = state.pmd.environment.scratch_path or None
    logger = RunLogger(state, start_time=start_time, log_dir=log_dir)
    state.logger = logger
    logger.write_start_line()
    logger.write_heading(state)
    logger.log_workflow_title(workflow_cls, state.jobtype)

    try:
        result = execute(state, runtime_idx=runtime_idx)
    except Exception as error:
        logger.finish('error: %s' % error, note=_stop_note(state))
        raise

    # Non-iterative workflows print the final result block here;
    # iterative workflows already recorded every step themselves.
    kind = (getattr(workflow_cls, 'workflow_kind', 'single')
            if workflow_cls is not None else 'single')
    if kind != 'iterative':
        # Adaptive sampling evaluates many structures; the parent state's
        # single geometry is not a product of that run.
        logger.log_result(
            state, include_system=workflow_cls is not AdaptiveSampling)

    status = 'completed'
    runtime = getattr(state, 'runtime', None)
    if runtime is not None and getattr(runtime, 'status', '') == 'failed':
        status = 'failed: %s' % getattr(runtime, 'error_message', '')
    logger.finish(status, note=_stop_note(state))

    return result


def _stop_note(state):
    """The md stop-code sentence for one run ('' when the code is zero)."""
    from .kernel.logger import stop_code_note
    runtime = getattr(state, 'runtime', None)
    return stop_code_note(getattr(runtime, 'stop_code', 0))


def _write_result_json(result, state=None, filename='result.json'):
    """Write calculation results to result.json (child runtimes).

    Parameters
    ----------
    result : Result, PyRAIState or None
        Execution output of the restored runtime.
    state : PyRAIState or None
        Restored state, used as fallback result source.
    filename : str
        Output file name in the current working directory.
    """
    if result is not None and hasattr(result, 'to_dict'):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = dict(result)
    elif state is not None and getattr(state, 'pmd', None) is not None:
        data = dict(getattr(state.pmd, 'results', {}) or {})
    else:
        data = {}

    write_json(filename, json_safe(data))


def print_state_info(state):
    """Print the parsed run summary for ``--info``.

    ``--info`` stops after ``build_state``, so this is the only place a
    user sees what the input resolved to without starting a calculation.
    It reports the control block and the task queue sizes; the per-task
    detail is left to the task-config logs a real run writes.

    Parameters
    ----------
    state : PyRAIState
        The constructed runtime state.
    """
    config = getattr(state, 'pmd', None)
    control = getattr(config, 'control', None)
    system = getattr(config, 'system', None)
    theory = ' '.join(getattr(control, 'theory', []) or []) or '-'
    reference = ' '.join(getattr(control, 'reference', []) or []) or '-'
    print('title:     %s' % getattr(control, 'title', ''))
    print('jobtype:   %s' % getattr(control, 'jobtype', ''))
    print('theory:    %s' % theory)
    print('reference: %s' % reference)
    print('driver:    %s' % getattr(control, 'driver', 'main'))
    print('natom:     %d' % int(getattr(system, 'natom', 0) or 0))
    print('tasks:     theory %d, reference %d'
          % (len(getattr(state, 'theory', {}) or {}),
             len(getattr(state, 'reference', {}) or {})))


def _print_usage():
    """Print CLI usage information."""
    print("""

  Usage: pyrai3md <input_file> [--info]
         pyrai3md test

  Commands:
    <input_file>    Run traditional CLI with input file
    test            Run test suite (qualification)

  Options:
    --info    Print parsed information without running
""")
