"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module runs Molcas calculations locally via subprocess.

"""

import os
import subprocess


def run_local(config, calcdir: str, workdir: str = '') -> int:
    """Execute a Molcas calculation locally via subprocess.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with Molcas runtime settings.
    calcdir : str
        Calculation directory containing the input file.
    workdir : str
        Molcas working directory for scratch files.

    Returns
    -------
    int
        Return code from the subprocess execution.
    """
    from ..templates.run_templates import build_local_script

    script = build_local_script(config, workdir=workdir)
    run_dir = os.path.abspath(calcdir)
    script_path = os.path.join(run_dir, f"{config.project}.sh")

    with open(script_path, 'w') as f:
        f.write(script)

    os.chmod(script_path, 0o755)

    # The child is anchored with cwd=, not a process-wide chdir: two
    # terms of one skill share the interpreter, and chdir would resolve
    # the other term's script name in the wrong directory.
    # The base is made absolute for the same reason: `calcdir` can be
    # relative when `environment.scratch_path` is written relative, and a
    # relative script path resolved against that cwd would name one level
    # too deep and bash would not find the deck it just wrote.
    proc = subprocess.run(
        ['bash', script_path],
        capture_output=False,
        cwd=run_dir,
    )
    return proc.returncode
