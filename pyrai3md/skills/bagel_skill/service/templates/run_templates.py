"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the run script templates for BAGEL execution,
including local bash scripts and SLURM submission scripts.

"""


def _shell_setup(config) -> str:
    """Return the normalized shell setup commands from the config.

    The 'shell' entry carries skill-level shell commands (module
    loads, activation lines, ...) from skills.bagel_skill.shell of
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


def _bagel_run_commands(config, from_slurm=False) -> str:
    """Return the environment exports and the BAGEL run line.

    Shared by the local and slurm scripts. BAGEL runs in MPI mode by
    default: threads are pinned to 1 and the binary is launched under
    the ``mpiexe`` launcher. When ``mpiexe`` is no/none
    (case-insensitive), BAGEL runs directly without a launcher and
    the thread variables scale with the process count instead.

    The rank count depends on who owns the allocation. The skill's own
    slurm script reads ``$SLURM_NTASKS`` because its header states that
    number (``slurm_ntasks``). A local script runs with ``bagel_nproc``:
    inside a harness job (the qualification's grouped submission) the
    environment carries the *harness* allocation, and reading it there
    launched ten ranks for a two-rank skill case and timed the BAGEL
    cases out (round tmp_20261111_132628).

    Parameters
    ----------
    config : SkillConfig
        Configuration container with BAGEL runtime settings.
    from_slurm : bool
        True when building the skill's own batch script.

    Returns
    -------
    str
        Export commands, optional MKL sourcing, and the run line.
    """
    bagel_path = config.bagel_path
    project = config.project
    blas = config.blas
    lapack = config.lapack
    boost = config.boost
    mpi = config.mpi
    mkl = config.mkl
    arch = config.arch

    if from_slurm:
        ntasks = f'${{SLURM_NTASKS:-{config.nproc}}}'
    else:
        ntasks = str(config.nproc)
    num_thread = config.thread
    mpiexe = str(config.mpiexe).strip().lower()
    if mpiexe in ('no', 'none'):
        # Direct execution: one process, no launcher to spread ranks
        run_line = f'$BAGEL/bin/BAGEL {project}.json > {project}.log'
    else:
        # MPI execution: the launcher spreads the ranks
        run_line = (f'{mpiexe} -np {ntasks} '
                    f'$BAGEL/bin/BAGEL {project}.json > {project}.log')

    # Threads per rank come from `thread` in both arms (#305), the same
    # rule lammps/cp2k/openqp follow: MPI mode used to pin them to 1 while
    # the SLURM header allocated --cpus-per-task from the same keyword,
    # job bought cores the run never used, and direct mode used to take
    # the thread count from the rank expression, which made a keyword for
    # processes answer a question about threads. A run that wants N
    # threads says `thread` N; a run that wants N ranks says
    # bagel_nproc/slurm_ntasks N.
    commands = f"""export BAGEL={bagel_path}
export BLAS={blas}
export LAPACK={lapack}
export BOOST={boost}
export MPI={mpi}
export OMP_NUM_THREADS={num_thread}
export MKL_NUM_THREADS={num_thread}
export BAGEL_NUM_THREADS={num_thread}
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BLAS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH
"""

    # Source MKL if provided
    if mkl and arch:
        commands += f"source {mkl} {arch}\n"

    # BAGEL writes three families of results under fixed names in the working
    # directory: ENERGY.out, FORCE_<state>.out and NACME_<a>_<b>.out. None of
    # them carries the identity of the job that wrote it, and the script runs
    # with the calculation directory as its working directory, so a file left by
    # an earlier job is read back as this job's answer. Measured in round
    # tmp_20261015_100320: `xmscaspt2_frequency` printed no energy in its own log
    # and reported -77.08098671, a number the file had received from a sibling
    # case five minutes later. Clear them where the engine will write them, in
    # the script, so both arms do it on the node that runs the job.
    commands += 'rm -f ENERGY.out FORCE_*.out NACME_*.out\n'

    commands += run_line + '\n'
    return commands


def build_local_script(config) -> str:
    """Build a local bash run script for BAGEL execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with BAGEL runtime settings.

    Returns
    -------
    str
        Bash script content.
    """
    return f"""#!/bin/bash
{_shell_setup(config)}{_bagel_run_commands(config)}"""


def build_slurm_script(config, slurm_template: str = '') -> str:
    """Build a SLURM submission script for BAGEL execution.

    Parameters
    ----------
    config : SkillConfig
        Configuration container with BAGEL runtime settings.
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

    # Append the BAGEL execution commands
    script += f"\n{_shell_setup(config)}{_bagel_run_commands(config, from_slurm=True)}"

    return script
