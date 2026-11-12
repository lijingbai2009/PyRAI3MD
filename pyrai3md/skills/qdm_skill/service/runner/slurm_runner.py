"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides SLURM execution for QDM calculations.

The runner serializes the flat configuration into the calculation
directory and submits a blocking sbatch job ('sbatch -W', same
convention as the bagel/molcas skills). The job runs the QDM skill
agent on the cluster node; the agent writes the formatted results
into '{project}_results.json' next to the config, which the backend
collects after the submission call returns. Reference trajectory
files are written into the same calculation directory by the agent.

"""

import json
import os
import subprocess
import sys

from ..templates.run_templates import SLURM_SCRIPT_TEMPLATE


def package_import_root():
    """Return the import root of the pyrai3md package.

    Derived from this file's location so the skill stays free of
    framework imports; cluster jobs prepend the path to PYTHONPATH
    so the agent runs the same skill code as the parent process.

    Returns
    -------
    str
        Absolute directory that contains the pyrai3md package.
    """
    path = os.path.abspath(__file__)
    for _ in range(6):
        path = os.path.dirname(path)
    return path


def run_slurm(config, calcdir: str) -> int:
    """Execute QDM calculation via SLURM.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container with QDM settings.
    calcdir : str
        Calculation directory receiving the config json, the sbatch
        script and the agent outputs.

    Returns
    -------
    int
        Return code from the sbatch submission.
    """
    project = config.project or 'QDM'

    # Serialize the flat configuration for the cluster-side agent.
    # model_dump() produces a nested dict; the agent rebuilds its
    # own SkillConfig from this flat JSON on the cluster node.
    flat_config = config.model_dump()
    # The job body must execute here: the parent already chose the
    # slurm arm. Leaving execution_mode='slurm' in this JSON made the
    # compute-node agent submit another job recursively.
    flat_config.setdefault('environment', {})['execution_mode'] = 'local'
    config_file = os.path.join(calcdir, f'{project}.json')
    with open(config_file, 'w') as f:
        json.dump(flat_config, f, indent=2)

    script = SLURM_SCRIPT_TEMPLATE.format(
        partition=config.environment.slurm_partition,
        time=config.environment.slurm_time,
        ntasks=config.environment.slurm_ntasks,
        cpus_per_task=config.environment.thread,
        memory=config.environment.slurm_memory,
        job_name=project,
        output=os.path.join(calcdir, f'{project}_slurm.out'),
        error=os.path.join(calcdir, f'{project}_slurm.err'),
        command=config.environment.slurm_command,
        pythonpath=package_import_root(),
        python_exec=sys.executable,
        module='pyrai3md.skills.qdm_skill.agent',
        config_file=config_file,
    )

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
