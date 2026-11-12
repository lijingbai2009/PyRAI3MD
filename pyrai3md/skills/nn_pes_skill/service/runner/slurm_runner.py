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

This module provides the slurm submission arm of the nn_pes training
run. One independent job is submitted per ensemble member, matching the
local arm, which also starts one process per member; the two arms differ
only in the launcher in front of the same command line.

The submission is blocking ('sbatch -W'), the convention the shipped
skills use, so the parent keeps a single wait loop instead of growing a
scheduler poller of its own.

"""

import os
import shutil
import subprocess

SBATCH_OPTIONS = {
    'slurm_partition': '-p',
    'slurm_time': '-t',
    'slurm_memory': '--mem',
    'slurm_ntasks': '-n',
}


def sbatch_preamble(config):
    """Build the #SBATCH lines of one training job.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    list of str
        Directive lines for the job script.
    """
    environment = config.environment
    lines = ['#!/bin/bash']
    for field, flag in SBATCH_OPTIONS.items():
        value = getattr(environment, field, None)
        if value is None or value == '':
            continue
        lines.append(f'#SBATCH {flag} {value}')
    lines.append('#SBATCH -J nn_pes_train')
    return lines


def write_job_script(command, workdir, config, name):
    """Write one job script that runs the given training command.

    Parameters
    ----------
    command : list of str
        Training command line, identical to the local arm's.
    workdir : str
        Directory the job runs in and where the script is written.
    config : SkillConfig
        Runtime configuration container.
    name : str
        Script base name.

    Returns
    -------
    str
        Path of the written script.
    """
    script = os.path.join(workdir, f'{name}.sbatch')
    lines = sbatch_preamble(config)
    # The job runs in its own process, so the thread count it inherits is
    # whatever this script says, not the parent's environment (#306). The
    # training lifetime uses train_thread, not the serving predict_thread.
    thread = str(config.environment.train_thread)
    lines.append('export OMP_NUM_THREADS=%s' % thread)
    lines.append('export TF_NUM_INTRAOP_THREADS=%s' % thread)
    lines.append('export TF_NUM_INTEROP_THREADS=%s' % thread)
    lines.append('')
    lines.append(' '.join(str(part) for part in command))
    with open(script, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return script


def submit_blocking(script_path, workdir):
    """Submit one job script and block until it leaves the queue.

    Parameters
    ----------
    script_path : str
        Path of the job script.
    workdir : str
        Directory to submit from.

    Returns
    -------
    object
        Popen handle of the blocking 'sbatch -W' call.

    Raises
    ------
    RuntimeError
        If the 'sbatch' command is not available in this environment.
    """
    if shutil.which('sbatch') is None:
        raise RuntimeError(
            "execution_mode is 'slurm' but the 'sbatch' command was not "
            "found in PATH; use execution_mode 'local' or run this job on "
            "the cluster head node.")
    return subprocess.Popen(['sbatch', '-W', os.path.basename(script_path)],
                            cwd=workdir)
