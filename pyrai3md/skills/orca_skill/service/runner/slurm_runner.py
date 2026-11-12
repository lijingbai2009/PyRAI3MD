"""
======================================================================
This is a standalone ORCA calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module runs ORCA calculations via SLURM job scheduler.

"""

import os
import subprocess


def run_slurm(config, calcdir: str) -> int:
    """Execute an ORCA calculation via SLURM.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with ORCA runtime settings.
    calcdir : str
        Calculation directory containing the input file.

    Returns
    -------
    int
        Return code from the subprocess execution.
    """
    from ..templates.run_templates import build_slurm_script

    project = config.project

    # Check for user-provided SLURM template
    slurm_template_path = f'{project}.slurm'
    slurm_template = ''
    if os.path.exists(slurm_template_path):
        with open(slurm_template_path, 'r') as f:
            slurm_template = f.read()

    script = build_slurm_script(config, slurm_template)
    script_path = os.path.join(calcdir, f'{project}.sbatch')

    with open(script_path, 'w') as f:
        f.write(script)

    # The child is anchored with cwd=, not a process-wide chdir: two
    # terms of one skill share the interpreter, and chdir would resolve
    # the other term's script name in the wrong directory.
    proc = subprocess.run(
        ['sbatch', '-W', script_path],
        capture_output=False,
        cwd=calcdir,
    )
    return proc.returncode
