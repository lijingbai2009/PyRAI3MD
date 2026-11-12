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

This module runs CP2K calculations via SLURM job scheduler.

"""

import os
import subprocess


def run_slurm(config, calcdir: str) -> int:
    """Execute an CP2K calculation via SLURM.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with CP2K runtime settings.
    calcdir : str
        Calculation directory containing the input file.

    Returns
    -------
    int
        Return code from the subprocess execution.
    """
    from ..templates.run_templates import build_slurm_script

    project = config.project

    # Check for a user-provided SLURM template. The point's own
    # directory is looked in first: the numerical-stress driver copies
    # the central template there under the point's project name, which
    # is what lets a perturbed point -- whose file stem differs from the
    # central run -- still pick up the submission script the user wrote.
    # The working directory stays as the fallback the route has always
    # used, so a standalone run is unaffected.
    slurm_template = ''
    for template_path in (os.path.join(calcdir, f'{project}.slurm'),
                          f'{project}.slurm'):
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                slurm_template = f.read()
            break

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
