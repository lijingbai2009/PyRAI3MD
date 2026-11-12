"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module checks the OpenQP log file for completion markers.

Result data is NOT parsed from the log: the skill enables
[properties]export=true, so results are read from the fixed-name
export text files (energies, grad_<i>, nacme, dcme, nac_<i>_<j>).
The log parser only verifies that the run completed without fatal
convergence failures.

"""

import os

# Fatal failure markers written by OpenQP (section='end' or failure dumps)
_FAILURE_MARKERS = [
    'PyOQP: SCF energy is not converged',
    'PyOQP: TD energy is not converged',
    'PyOQP: TD Z-vector is not converged',
    'PyOQP: numerical hessian calculations failed',
    'PyOQP: numerical nac calculations failed',
]

# Geometry optimization status markers
_OPT_CONVERGED_MARKER = 'PyOQP: Geometry Optimization Has Converged'
_OPT_FAILED_MARKER = 'PyOQP: Geometry Optimization Has Not Converged'


def parse_log(log_path: str, workflow) -> dict:
    """Check the OpenQP log for completion markers.

    Parameters
    ----------
    log_path : str
        Path to the OpenQP .log file.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Status dictionary with keys:
        - completed : bool, the log exists and no fatal marker found
        - error : str, error description ('' when completed)
        - opt_converged : bool, optimization converged marker found
    """
    status = {
        'completed': False,
        'error': '',
        'opt_converged': False,
    }

    if not os.path.exists(log_path):
        status['error'] = f'OpenQP log file not found: {log_path}'
        return status

    with open(log_path, 'r') as f:
        log = f.read()

    for marker in _FAILURE_MARKERS:
        if marker in log:
            status['error'] = f'OpenQP calculation failed: {marker}'
            return status

    if workflow.runtype == 'optimize':
        if _OPT_CONVERGED_MARKER in log:
            status['opt_converged'] = True
        elif _OPT_FAILED_MARKER in log:
            # The engine finished the optimize and says it stopped without
            # reaching its criterion (on this deployment: `Reached The
            # Maximum Iteration` at both 5 and 50). That is the step-budget
            # stop the other optimization skills report as
            # `opt_converged: False` and let the caller decide on; raising
            # here made the direct-skill leg a red while the framework legs
            # printed NOT CONVERGED for the same run (#466).
            status['opt_converged'] = False

    status['completed'] = True
    return status
