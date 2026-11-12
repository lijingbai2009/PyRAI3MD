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

This is the CLI entry point for the PySCF skill.
It wraps backend.calculate() with error-safe execution.

Usage:
    python -m pyrai3md.skills.pyscf_skill.agent <config.json>

"""

import json
import os
import sys

from .backend import PySCFBackend
from .service.utils import safe_run
from .service.runner import results_file_name


def _run_calculation(backend):
    """Forward the dynamic calculation inputs to the backend.

    point_charges embeds background charges and target_states
    selects the states evaluated by the calculation (multistate
    gradients etc.). Both come from the agent config so the
    subprocess entry point matches the framework calculator call.

    Parameters
    ----------
    backend : PySCFBackend
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
        backend = PySCFBackend(config_file)
    except Exception as e:
        print(f"[PySCF ERROR] Failed to initialize: {e}", file=sys.stderr)
        sys.exit(1)

    # Run calculation via safe wrapper
    results_holder = {}

    def runner():
        results_holder['results'] = _run_calculation(backend)

    run_status = safe_run(runner)

    # Persist the formatted results next to the config so a remote
    # (slurm) parent backend can collect them after submission.
    if run_status == 'success':
        with open(results_file_name(config_file), 'w',
                  encoding='utf-8') as fh:
            json.dump(
                results_holder.get('results', {}), fh, indent=2,
                default=lambda value: value.tolist()
                if hasattr(value, 'tolist') else str(value))

    # Write log file with same basename as config
    with open(log_file, 'w', encoding='utf-8') as fh:
        fh.write(f"Config: {config_file}\n")
        fh.write(f"Mode: {backend.config.environment.execution_mode}\n")
        fh.write(f"Status: {run_status}\n")

    # Report status to screen
    if run_status == 'success':
        print(f"[PySCF] Run status: success")
        print(f"[PySCF] Log saved to: {log_file}")
    else:
        print(f"[PySCF] Run status: failed")
        print(f"[PySCF] Log saved to: {log_file}")
        error_lines = [l for l in run_status.strip().splitlines() if l.strip()]
        if error_lines:
            print(f"[PySCF] Error: {error_lines[-1]}")
        sys.exit(1)
