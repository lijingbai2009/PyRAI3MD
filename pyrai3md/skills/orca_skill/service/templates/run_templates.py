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

This module defines the run script templates for ORCA execution,
including local bash scripts and SLURM submission scripts.

"""


def _shell_setup(config) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.orca_skill.shell of
    pmd_env.yaml or the test input; the global environment.shell is
    reserved for the main framework. A list of commands, a multi-line
    string, or a single line with ';' separators (classic input) are
    all accepted.

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


def _build_orca_commands(config) -> str:
    """Build the ORCA environment export and execution commands.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with ORCA runtime settings.

    Returns
    -------
    str
        Shell commands exporting ORCA/MPI paths and running ORCA.
    """
    orca_path = config.orca_path
    mpi = config.mpi
    project = config.project

    commands = f"""
{_shell_setup(config)}export ORCA_PROJECT={project}
export ORCA={orca_path}
export MPI={mpi}

export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

# stderr is merged into the log: ORCA reports input/keyword errors
# on stderr, and losing them leaves an empty failure log.
$ORCA/orca $ORCA_PROJECT.inp > $ORCA_PROJECT.out 2>&1
"""
    return commands


def build_local_script(config) -> str:
    """Build a local bash run script for ORCA execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with ORCA runtime settings.

    Returns
    -------
    str
        Bash script content.
    """
    script = '#!/bin/bash\n'
    script += _build_orca_commands(config)
    return script


def build_slurm_script(config, slurm_template: str = '') -> str:
    """Build a SLURM submission script for ORCA execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with ORCA runtime settings.
    slurm_template : str
        Optional user-provided SLURM template content.

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

    # Append the ORCA execution commands
    script += _build_orca_commands(config)

    return script
