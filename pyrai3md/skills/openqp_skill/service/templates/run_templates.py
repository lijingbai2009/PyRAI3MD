"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the run script templates for OpenQP execution,
including local bash scripts and SLURM submission scripts.

The run command is controlled by the MPI switch:
- use_mpi=False (default): OpenMP-only mode, 'openqp ... --nompi'
- use_mpi=True: MPI mode, 'mpirun -n <openqp_nproc> openqp ...'
  (no --nompi); mpi_command overrides the launcher prefix.

"""


def _shell_setup(config) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.openqp_skill.shell of
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


def build_run_command(config) -> str:
    """Build the OpenQP run command based on the MPI switch.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with OpenQP runtime settings.

    Returns
    -------
    str
        Shell command line (uses the $OPENQP_PROJECT variable).
    """
    if config.use_mpi:
        mpi_command = str(config.mpi_command).strip()
        if not mpi_command:
            mpi_command = f"mpirun -n {config.openqp_nproc}"
        return (f'{mpi_command} openqp '
                '"$PWD/${OPENQP_PROJECT}.inp"')

    # The numerical Hessian spawns displacement children in
    # <project>_num_hess/ and hands them the JSON guess path derived from the
    # parent's log path. A relative input makes that path relative to the
    # child directory, so every child fails to find the guess and the Hessian
    # is empty. An absolute input keeps the derived guess path absolute.
    return 'openqp "$PWD/${OPENQP_PROJECT}.inp" --nompi'


def _build_env_block(config) -> str:
    """Build the shared environment export block for run scripts.

    OPENQP_PROJECT fixes the input/log/orbital-json/export file
    names; OPENQP_ROOT points to the OpenQP installation root when
    openqp_path is set, and $OPENQP_ROOT/lib is prepended to
    LD_LIBRARY_PATH. The 'openqp' command itself is expected in
    PATH after a proper OpenQP installation. The runner has already
    chdir'd into calcdir, so no separate workdir is needed.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with OpenQP runtime settings.

    Returns
    -------
    str
        Environment export lines for the run script.
    """
    project = config.project
    openqp_path = config.openqp_path
    num_thread = config.thread

    lines = []
    shell_setup = _shell_setup(config)
    if shell_setup:
        # Module loads must precede the export block.
        lines.append(shell_setup.rstrip('\n'))
        lines.append('')
    lines.append(f"export OPENQP_PROJECT={project}")
    if openqp_path:
        lines.append(f"export OPENQP_ROOT={openqp_path}")
        lines.append(
            "export LD_LIBRARY_PATH="
            "$OPENQP_ROOT/lib:$MKL_LIB:$GCC_LIB:$LD_LIBRARY_PATH"
        )
    lines.append(f"export OMP_NUM_THREADS={num_thread}")

    lines.append("")
    lines.append(build_run_command(config))

    return '\n'.join(lines)


def build_local_script(config) -> str:
    """Build a local bash run script for OpenQP execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with OpenQP runtime settings.

    Returns
    -------
    str
        Bash script content.
    """
    script = f"""#!/bin/bash
{_build_env_block(config)}
"""
    return script


def build_slurm_script(config, slurm_template: str = '') -> str:
    """Build a SLURM submission script for OpenQP execution.

    In MPI mode the number of MPI ranks (openqp_nproc) is mapped to
    '#SBATCH --ntasks' so that the scheduler allocates enough ranks.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with OpenQP runtime settings.
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

    # MPI mode: MPI ranks map to slurm ntasks
    if config.use_mpi:
        ntasks = max(ntasks, config.openqp_nproc)

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

    script += f"""
{_build_env_block(config)}
"""
    return script
