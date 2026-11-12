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

This module provides the training launcher of the nn_pes skill.

One subprocess per (ensemble, member) is started, which is the shape the
vendored container has always used, and all of them are awaited in one
loop. Two arms share the same command line: 'local' starts it directly,
'slurm' writes it into a job script and submits that with 'sbatch -W',
so a cluster run submits one independent job per member.

"""

import os
import subprocess

from ..nn_pes.nn_pes_src.fit import child_environment, train_command
from ...schema.hyper import member_list
from . import slurm_runner


def build_jobs(config, targets):
    """Expand the training work into one entry per ensemble member.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    targets : list of tuple
        (ensemble name, model type, model folder) to train.

    Returns
    -------
    list of dict
        Job entries with the command line each arm will start.
    """
    jobs = []
    members = config.calculation.mult_nn
    mode = config.calculation.train_mode
    devices = member_list(config.calculation.gpu, members, 'gpu')
    # The framework's control-level seed reaches the fit through this line, and
    # it advances by the member index: every member that drew the same numbers
    # would be the same network, and an ensemble of one network trained five
    # times is not an ensemble. Same rule an MD run uses for its trajectories
    # (workflow/md.py seeds `gl_seed + trajectory index`).
    gl_seed = getattr(config.calculation, 'gl_seed', None)
    for name, model_type, folder in targets:
        for index in range(members):
            seed = None if gl_seed is None else int(gl_seed) + index
            command = train_command(model_type, index, folder,
                                    int(devices[index]), mode, seed)
            jobs.append({
                'name': name,
                'model_type': model_type,
                'index': index,
                'folder': folder,
                'command': command,
            })
    return jobs


def start(job, config, arm):
    """Start one training job through the selected arm.

    Parameters
    ----------
    job : dict
        Entry produced by build_jobs().
    config : SkillConfig
        Runtime configuration container.
    arm : str
        'local' or 'slurm'.

    Returns
    -------
    object
        Popen handle of the started process or submission.
    """
    if arm == 'local':
        return subprocess.Popen(
            job['command'],
            env=child_environment(config.environment.train_thread),
            cwd=job['folder'])

    name = '%s_v%i' % (job['name'], job['index'])
    script = slurm_runner.write_job_script(job['command'], job['folder'],
                                           config, name)
    return slurm_runner.submit_blocking(script, job['folder'])


def run_training(config, targets, arm='local'):
    """Start every member of every target ensemble and wait for them.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    targets : list of tuple
        (ensemble name, model type, model folder) to train.
    arm : str, optional
        'local' starts processes directly, 'slurm' submits one blocking
        job per member.

    Returns
    -------
    list of dict
        One record per member: ensemble, index, return code.

    Raises
    ------
    ValueError
        If an unknown arm is requested.
    """
    if arm not in ('local', 'slurm'):
        raise ValueError(f"Unknown training arm {arm!r}")

    jobs = build_jobs(config, targets)
    processes = [(job, start(job, config, arm)) for job in jobs]

    records = []
    for job, process in processes:
        code = process.wait()
        records.append({
            'name': job['name'],
            'index': job['index'],
            'returncode': code,
            'log': os.path.join(job['folder'], 'fitlog_%i.txt' % job['index']),
        })
    return records


def failed_runs(records):
    """Return the members that did not finish cleanly.

    Parameters
    ----------
    records : list of dict
        Output of run_training().

    Returns
    -------
    list of dict
        Records with a non-zero return code.
    """
    return [record for record in records if record['returncode'] != 0]
