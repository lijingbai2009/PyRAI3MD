"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides local execution for PySCF calculations.
Since PySCF is a Python-native library, local execution means
directly calling the service functions in the current process.

"""

import os
import sys
import numpy as np

from ..builder import dispatch_calculation
from ..utils import set_stdout


class _Tee:
    """Duplicate one term's writes to a log file.

    PySCF is a Python-native library, so all calculation output
    (molecule build info, SCF iterations, post-HF results) is written
    through ``mol.stdout``. The runner hands the molecule this object --
    bound to the calling thread, never installed as ``sys.stdout`` --
    and every write lands on both the terminal and this term's log.
    """

    def __init__(self, filepath, original_stdout):
        self._file = open(filepath, 'w')
        self._stdout = original_stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return False

    def close(self):
        self._file.close()


def run_local(config, workflow, log_file='') -> dict:
    """Execute PySCF calculation locally (in-process).

    When *log_file* is given, all PySCF output (written to stdout
    via the logger) is captured and saved to that path alongside
    the terminal output.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    log_file : str, optional
        Path to the log file.  Derived from the input config
        file basename by the backend.

    Returns
    -------
    dict
        Runtime dictionary with results.
    """
    # ---- thread count ---------------------------------------------------
    # OpenMP reads OMP_NUM_THREADS once, at the first parallel region of
    # the process, so a per-term write cannot retune what is already
    # latched: the first term fixes it for the process (#302). Keep that
    # first value and say so when it is not the one this term asked for.
    requested = str(config.environment.thread)
    effective = os.environ.setdefault('OMP_NUM_THREADS', requested)
    if effective != requested:
        print('[PySCF] OMP_NUM_THREADS is already %s for this process, so '
              'thread=%s is ignored; the thread count is fixed at the '
              'first parallel region.' % (effective, requested),
              file=sys.stderr)

    # ---- log-file capture ------------------------------------------------
    tee = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        tee = _Tee(log_file, sys.stdout)

    # Bound to this thread only: another term of the skill keeps its own
    # handle and the process sys.stdout is never replaced.
    set_stdout(tee)
    try:
        runtime = dispatch_calculation(config, workflow)
    finally:
        set_stdout(None)
        if tee is not None:
            tee.close()

    if log_file:
        runtime['log_file'] = log_file

    return runtime


def to_list(results: dict) -> dict:
    """Convert numpy arrays in results to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Results dictionary with potential numpy arrays.

    Returns
    -------
    dict
        Results with numpy arrays converted to lists.
    """
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    return results
