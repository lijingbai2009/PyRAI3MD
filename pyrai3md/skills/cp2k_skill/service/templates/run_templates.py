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

This module defines the run script templates for CP2K execution,
including local bash scripts and SLURM submission scripts.

"""


def _shell_setup(config) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.cp2k_skill.shell of
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


def _build_cp2k_commands(config) -> str:
    """Build the CP2K environment export and execution commands.

    Follows the cp2k_demo/run.sh pattern: CP2K_HOME + executable
    resolution, optional toolchain setup script, OMP_NUM_THREADS,
    and mpirun -n N. stderr is merged into the log so input errors
    are never lost.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with CP2K runtime settings.

    Returns
    -------
    str
        Shell commands exporting the CP2K environment and running
        CP2K via mpirun.
    """
    cp2k_path = config.cp2k_path
    cp2k_exe = config.cp2k_exe or '$CP2K_HOME/exe/local/cp2k.psmp'
    setup_script = config.setup_script
    nproc = config.nproc
    num_thread = config.thread
    project = config.project

    setup_line = f'source {setup_script}\n' if setup_script else ''

    commands = f"""
{_shell_setup(config)}export CP2K_PROJECT={project}
export CP2K_HOME={cp2k_path}
export CP2K_EXE={cp2k_exe}
export OMP_NUM_THREADS={num_thread}
{setup_line}
# stderr is merged into the log: CP2K reports input errors on
# stderr, and losing them leaves an empty failure log.
mpirun -n {nproc} $CP2K_EXE -i $CP2K_PROJECT.inp -o $CP2K_PROJECT.log 2>&1
"""
    return commands


def build_local_script(config) -> str:
    """Build a local bash run script for CP2K execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with CP2K runtime settings.

    Returns
    -------
    str
        Bash script content.
    """
    script = '#!/bin/bash\n'
    script += _build_cp2k_commands(config)
    return script


def build_slurm_script(config, slurm_template: str = '') -> str:
    """Build a SLURM submission script for CP2K execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with CP2K runtime settings.
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

    # Append the CP2K execution commands
    script += _build_cp2k_commands(config)

    return script
