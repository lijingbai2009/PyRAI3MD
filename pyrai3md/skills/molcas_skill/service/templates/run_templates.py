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

This module defines the run script templates for Molcas execution,
including local bash scripts and SLURM submission scripts.

"""

import os


def _shell_setup(config) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.molcas_skill.shell of
    pmd_env.yaml or the test input; the global environment.shell is
    reserved for the main framework.  The SkillConfig normalizes this
    into a single string during build_config.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with runtime settings.

    Returns
    -------
    str
        One command per line followed by a blank line; '' when no
        commands are configured.
    """
    shell = config.shell
    if not shell:
        return ''
    return shell + '\n'


def build_local_script(config, workdir: str = '') -> str:
    """Build a local bash run script for Molcas execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with Molcas runtime settings.
    workdir : str
        Molcas working directory for scratch files.

    Returns
    -------
    str
        Bash script content.
    """
    molcas_path = config.molcas_path
    project = config.project
    nproc = config.nproc
    mem = config.molcas_mem
    molcas_print = config.molcas_print
    num_thread = config.thread
    pymolcas = os.path.join(molcas_path, 'bin', 'pymolcas')
    if not os.path.isfile(pymolcas):
        pymolcas = os.path.join(molcas_path, 'pymolcas')

    script = f"""#!/bin/bash
{_shell_setup(config)}export MOLCAS={molcas_path}
export MOLCAS_PROJECT={project}
export MOLCAS_WORKDIR={workdir}
export MOLCAS_NPROCS={nproc}
export MOLCAS_MEM={mem}
export MOLCAS_PRINT={molcas_print}
export OMP_NUM_THREADS={num_thread}
export PATH=$MOLCAS/bin:$MOLCAS:$PATH

mkdir -p $MOLCAS_WORKDIR/$MOLCAS_PROJECT
{pymolcas} -f $MOLCAS_PROJECT.inp -b 1
rc=$?
rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
exit $rc
"""
    return script


def build_slurm_script(config, slurm_template: str = '', workdir: str = '') -> str:
    """Build a SLURM submission script for Molcas execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with Molcas runtime settings.
    slurm_template : str
        Optional user-provided SLURM template content.
    workdir : str
        Molcas working directory for scratch files.

    Returns
    -------
    str
        SLURM script content.
    """
    project = config.project
    partition = config.slurm_partition
    time_limit = config.slurm_time
    ntasks = config.slurm_ntasks
    memory = config.slurm_memory
    num_thread = config.thread

    if slurm_template:
        script = slurm_template
    else:
        script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={num_thread}
#SBATCH --mem={memory}mb
#SBATCH --job-name={project}
#SBATCH --output={project}_slurm.out
#SBATCH --error={project}_slurm.err
"""

    # Append the Molcas execution commands
    molcas_path = config.molcas_path
    nproc = config.nproc
    mem = config.molcas_mem
    molcas_print = config.molcas_print
    pymolcas = os.path.join(molcas_path, 'bin', 'pymolcas')
    if not os.path.isfile(pymolcas):
        pymolcas = os.path.join(molcas_path, 'pymolcas')

    molcas_commands = f"""
{_shell_setup(config)}export MOLCAS={molcas_path}
export MOLCAS_PROJECT={project}
export MOLCAS_WORKDIR={workdir}
export MOLCAS_NPROCS={nproc}
export MOLCAS_MEM={mem}
export MOLCAS_PRINT={molcas_print}
export OMP_NUM_THREADS={num_thread}
export PATH=$MOLCAS/bin:$MOLCAS:$PATH

mkdir -p $MOLCAS_WORKDIR/$MOLCAS_PROJECT
{pymolcas} -f $MOLCAS_PROJECT.inp -b 1
rc=$?
rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
exit $rc
"""
    script += molcas_commands

    return script
