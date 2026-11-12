"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the run script templates for LAMMPS execution,
including local bash scripts and SLURM submission scripts.

"""

import os

from ..utils import output_basename


def _resolve_lammps_binary(lammps_path: str) -> str:
    """Resolve the LAMMPS executable from the configured path.

    The lammps_path may point to the installation directory (using
    '$lammps_path/bin/lmp', legacy convention), directly to the
    binary, or be empty (falling back to 'lmp' in the system PATH).

    Parameters
    ----------
    lammps_path : str
        LAMMPS installation directory or binary path.

    Returns
    -------
    str
        Resolved executable reference for the run script.
    """
    if not lammps_path:
        return 'lmp'
    if os.path.isdir(lammps_path):
        return f'{lammps_path}/bin/lmp'
    return lammps_path


def _shell_setup(config: dict) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.lammps_skill.shell of
    pmd_env.yaml or the test input; the global environment.shell is
    reserved for the main framework. A list of commands, a multi-line
    string, or a single line with ';' separators (classic input) are
    all accepted.

    Parameters
    ----------
    config : dict
        Configuration dictionary with runtime settings.

    Returns
    -------
    str
        One command per line followed by a blank line; '' when no
        commands are configured.
    """
    shell = config.shell
    fragments = shell if isinstance(shell, list) else [shell]
    commands = []
    for fragment in fragments:
        if fragment is None:
            continue
        for piece in str(fragment).split(';'):
            piece = piece.strip()
            if piece:
                commands.append(piece)
    if not commands:
        return ''
    return '\n'.join(commands) + '\n\n'


def _build_lammps_commands(config: dict) -> str:
    """Build the environment export and LAMMPS execution commands.

    Single-process jobs run the binary directly; multi-process jobs
    are launched with 'mpirun -np N' (legacy convention). The two
    parallelism axes are separate keywords and both travel: 
    ``lammps_nproc`` counts the MPI ranks, ``thread`` is exported as
    ``OMP_NUM_THREADS`` for the threads inside each rank (the same pair
    cp2k and openqp spell). #303 documented the export as pinned to 1 for
    an engine reason; it never was -- 1 is just the keyword default, and
    LAMMPS built with USER-OMP uses anything above it. The input
    and log files share the per-task dump output basename (see
    service.utils.output_basename) so different tasks placed in the
    same directory never overwrite each other.

    Parameters
    ----------
    config : dict
        Configuration dictionary with LAMMPS runtime settings.

    Returns
    -------
    str
        Shell command block shared by local and slurm scripts.
    """
    project = config.lammps_project
    basename = output_basename(project, config.profile)
    nproc = config.nproc
    num_thread = config.thread
    lammps_binary = _resolve_lammps_binary(config.lammps_path)

    if nproc > 1:
        run_command = f'mpirun -np {nproc} {lammps_binary}'
    else:
        run_command = lammps_binary

    commands = f"""
{_shell_setup(config)}export LAMMPS_INPUT={basename}
export OMP_NUM_THREADS={num_thread}

{run_command} -in $LAMMPS_INPUT.in > $LAMMPS_INPUT.log
"""
    return commands


def build_local_script(config: dict) -> str:
    """Build a local bash run script for LAMMPS execution.

    Parameters
    ----------
    config : dict
        Configuration dictionary with LAMMPS runtime settings.

    Returns
    -------
    str
        Bash script content.
    """
    script = '#!/bin/bash\n'
    script += _build_lammps_commands(config)
    return script


def build_slurm_script(config: dict, slurm_template: str = '') -> str:
    """Build a SLURM submission script for LAMMPS execution.

    Parameters
    ----------
    config : dict
        Configuration dictionary with LAMMPS runtime settings.
    slurm_template : str
        Optional user-provided SLURM template content.

    Returns
    -------
    str
        SLURM script content.
    """
    project = config.lammps_project
    partition = config.slurm_partition
    time_limit = config.slurm_time
    ntasks = config.slurm_ntasks
    memory = config.slurm_memory
    nproc = config.nproc
    num_thread = config.thread
    # One task's cores are its ranks times the OpenMP threads per rank;
    # asking for the ranks alone under-allocates as soon as `thread`
    # is above its default (#304).
    cpus_per_task = nproc * num_thread

    if slurm_template:
        script = slurm_template
    else:
        partition_line = ('#SBATCH --partition=%s\n' % partition
                          if partition else '')
        script = f"""#!/bin/bash
{partition_line}#SBATCH --time={time_limit}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}mb
#SBATCH --job-name={project}
#SBATCH --output={project}_slurm.out
#SBATCH --error={project}_slurm.err
"""

    # Append the LAMMPS execution commands
    script += _build_lammps_commands(config)

    return script
