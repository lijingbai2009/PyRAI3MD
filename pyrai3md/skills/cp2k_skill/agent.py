"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This is the CLI entry point for the CP2K skill.
It wraps backend.calculate() with error-safe execution.

Usage:
    python -m pyrai3md.skills.cp2k_skill.agent <config.json>

"""

import os
import sys

from .backend import Cp2kBackend
from .service.utils import safe_run


def _run_calculation(backend):
    """Forward the dynamic calculation inputs to the backend.

    point_charges embeds background charges and target_states
    selects the states evaluated by the calculation (multistate
    gradients etc.). Both come from the agent config so the
    subprocess entry point matches the framework calculator call.

    Parameters
    ----------
    backend : Cp2kBackend
        Initialized skill backend.

    Returns
    -------
    dict
        Calculation results from backend.calculate().
    """
    config = backend.config
    return backend.calculate(
        config.atoms, config.coordinates,
        point_charges=config.point_charges or None,
        target_states=config.target_states or None)


if __name__ == "__main__":
    config_file = sys.argv[1]

    # Log file: same basename as config
    log_file = os.path.splitext(config_file)[0] + '.log'

    try:
        backend = Cp2kBackend(config_file)
    except Exception as e:
        print(f"[CP2K ERROR] Failed to initialize: {e}", file=sys.stderr)
        sys.exit(1)

    # Run calculation via safe wrapper
    def runner():
        return _run_calculation(backend)

    run_status = safe_run(runner)

    # Write log file with same basename as config
    with open(log_file, 'w', encoding='utf-8') as fh:
        fh.write(f"Config: {config_file}\n")
        fh.write(f"Mode: {backend.config.environment.execution_mode}\n")
        fh.write(f"Status: {run_status}\n")

    # Report status to screen
    if run_status == 'success':
        print(f"[CP2K] Run status: success")
        print(f"[CP2K] Log saved to: {log_file}")
    else:
        print(f"[CP2K] Run status: failed")
        print(f"[CP2K] Log saved to: {log_file}")
        error_lines = [l for l in run_status.strip().splitlines() if l.strip()]
        if error_lines:
            print(f"[CP2K] Error: {error_lines[-1]}")
        sys.exit(1)
