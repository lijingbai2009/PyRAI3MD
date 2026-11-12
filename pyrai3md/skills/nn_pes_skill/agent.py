"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Standalone agent entry point of the nn_pes skill.

Reads a serialized skill configuration, runs one calculation through the
backend and writes the formatted results next to the config, so a
subprocess or cluster-side worker can use the skill without the
framework present.

Usage:
    python -m pyrai3md.skills.nn_pes_skill.agent <config.json>

"""

import json
import os
import sys

from .service.runner import results_file_name
from .service.utils import safe_run


def _run_calculation(backend):
    """Forward the dynamic calculation inputs to the backend.

    Parameters
    ----------
    backend : NNPESBackend
        Initialized skill backend.

    Returns
    -------
    dict
        Calculation results from backend.calculate().
    """
    config = backend.config
    return backend.calculate(
        config.atoms, config.coordinates,
        target_states=config.target_states or None)


if __name__ == "__main__":
    from .backend import NNPESBackend

    config_file = sys.argv[1]

    # Log file: same basename as config
    log_file = os.path.splitext(config_file)[0] + '.log'

    try:
        backend = NNPESBackend(config_file)
    except Exception as e:
        print(f"[NNPES ERROR] Failed to initialize: {e}", file=sys.stderr)
        sys.exit(1)

    results_holder = {}

    def runner():
        results_holder['results'] = _run_calculation(backend)

    run_status = safe_run(runner)

    # Persist the formatted results next to the config so a remote
    # (slurm) parent backend can collect them after submission.
    if run_status == 'success':
        with open(results_file_name(config_file), 'w', encoding='utf-8') as fh:
            json.dump(results_holder.get('results', {}), fh, indent=2,
                      default=lambda value: value.tolist()
                      if hasattr(value, 'tolist') else str(value))

    with open(log_file, 'w', encoding='utf-8') as fh:
        fh.write(f"Config: {config_file}\n")
        fh.write(f"Mode: {backend.config.environment.execution_mode}\n")
        fh.write(f"Status: {run_status}\n")
        # The numbers the skill returned belong in the log too: this file
        # is what a remote run leaves behind, and a status line alone says
        # nothing about what was computed or how far the members disagreed.
        served = results_holder.get('results') or {}
        if run_status == 'success' and served:
            if 'train_errors' in served:
                # A training run returned no properties, so it gets the
                # training report rather than an empty property block.
                from .service.trainer import format_training
                fh.write(format_training(backend.config, served))
            else:
                from .service.deviation import format_step
                properties = {key: served[key] for key in
                              ('energy', 'gradient', 'nac', 'soc')
                              if key in served}
                metrics = {key: served[key] for key in
                           ('err_energy', 'err_grad', 'err_nac', 'err_soc')
                           if key in served}
                fh.write(format_step(backend.config.calculation.profile,
                                     backend.config.system.atoms, properties,
                                     metrics,
                                     natom=len(backend.config.system.atoms)))

    if run_status == 'success':
        print("[NNPES] Run status: success")
        print(f"[NNPES] Log saved to: {log_file}")
    else:
        print("[NNPES] Run status: failed")
        print(f"[NNPES] Log saved to: {log_file}")
        error_lines = [line for line in run_status.strip().splitlines()
                       if line.strip()]
        if error_lines:
            print(f"[NNPES] Error: {error_lines[-1]}")
        sys.exit(1)
