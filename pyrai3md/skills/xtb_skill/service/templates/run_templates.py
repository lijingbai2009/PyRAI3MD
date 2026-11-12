"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the run script templates for xTB execution,
including local bash scripts and SLURM submission scripts.

"""

import os


def _resolve_xtb_binary(xtb_path: str) -> str:
    """Resolve the xTB executable from the configured path.

    The xtb_path may point to the installation directory (using
    '$xtb_path/bin/xtb'), directly to the binary, or be empty
    (falling back to 'xtb' in the system PATH).

    Parameters
    ----------
    xtb_path : str
        xTB installation directory or binary path.

    Returns
    -------
    str
        Resolved executable reference for the run script.
    """
    if not xtb_path:
        return 'xtb'
    if os.path.isdir(xtb_path):
        return f'{xtb_path}/bin/xtb'
    return xtb_path


def _resolve_xtb_home(xtb_path: str) -> str:
    """Resolve the xTB installation root from the configured path.

    The install root feeds the legacy environment contract: XTBHOME
    points at the installation directory and XTBPATH at its
    parameter/data directory ({install}/share/xtb). An existing
    directory, or any path not ending in the 'xtb' binary name, is
    taken as the installation root; an existing file named 'xtb' is
    the binary and its grand parent is the install root; an empty
    xtb_path or a bare command name yields no export.

    Parameters
    ----------
    xtb_path : str
        xTB installation directory or binary path.

    Returns
    -------
    str
        Installation root, '' when xtb_path is empty.
    """
    if not xtb_path:
        return ''
    if os.path.isdir(xtb_path):
        install_root = xtb_path
    elif os.path.isfile(xtb_path) \
            and os.path.basename(xtb_path) == 'xtb':
        # Binary path: {install}/bin/xtb
        install_root = os.path.dirname(os.path.dirname(xtb_path))
    elif os.path.dirname(xtb_path):
        # Remote install root not visible on the local filesystem;
        # paths ending in the binary name still derive the grand
        # parent, anything else is taken as the install root.
        if os.path.basename(xtb_path) == 'xtb':
            install_root = os.path.dirname(os.path.dirname(xtb_path))
        else:
            install_root = xtb_path
    else:
        # Bare command name resolved via PATH: no install root
        return ''
    return install_root


def _shell_setup(config) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.xtb_skill.shell of
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


def _build_xtb_commands(config) -> str:
    """Build the environment export and xTB execution commands.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with xTB runtime settings.

    Returns
    -------
    str
        Shell command block shared by local and slurm scripts.
    """
    project = config.xtb_project
    mem = config.xtb_mem
    num_thread = config.thread
    xtb_binary = _resolve_xtb_binary(config.xtb_path)
    install_root = _resolve_xtb_home(config.xtb_path)
    xtb_args = config.xtb_args
    geometry_file = config.geometry_file or f'{project}.xyz'

    # Legacy environment contract (legacy qc_xtb.py): XTBHOME is the
    # installation root and XTBPATH its parameter/data directory;
    # both are exported only when the install path is known.
    if install_root:
        xtb_env_lines = (
            f'export XTBHOME={install_root}\n'
            f'export XTBPATH=$XTBHOME/share/xtb\n'
        )
    else:
        xtb_env_lines = ''

    commands = f"""
{_shell_setup(config)}export XTB_PROJECT={project}
{xtb_env_lines}export OMP_NUM_THREADS={num_thread}
export OMP_STACKSIZE={mem}MB
export KMP_STACKSIZE=$OMP_STACKSIZE
export OMP_MAX_ACTIVE_LEVELS=1

{xtb_binary} {xtb_args} -I $XTB_PROJECT.inp {geometry_file} > $XTB_PROJECT.out
"""
    return commands


def build_local_script(config) -> str:
    """Build a local bash run script for xTB execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with xTB runtime settings.

    Returns
    -------
    str
        Bash script content.
    """
    script = '#!/bin/bash\n'
    script += _build_xtb_commands(config)
    return script


def build_slurm_script(config, slurm_template: str = '') -> str:
    """Build a SLURM submission script for xTB execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with xTB runtime settings.
    slurm_template : str
        Optional user-provided SLURM template content.

    Returns
    -------
    str
        SLURM script content.
    """
    project = config.xtb_project
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

    # Append the xTB execution commands
    script += _build_xtb_commands(config)

    return script
