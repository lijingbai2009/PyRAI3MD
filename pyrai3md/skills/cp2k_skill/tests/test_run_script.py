"""
Unit tests for cp2k_skill run script generation (run_templates).

Verifies the run.sh-style environment exports (CP2K_HOME, CP2K_EXE,
OMP_NUM_THREADS), the optional setup script sourcing, and the mpirun
command line for both local and SLURM scripts.
These tests do NOT require a CP2K installation.

Run: pytest cp2k_skill/tests/test_run_script.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.templates.run_templates import (build_local_script,
                                               build_slurm_script)

ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]


def _make_config(**kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'profile': 'dft_energy',
        'method': 'dft',
        'functional': 'pbe',
        'cp2k_path': '/opt/cp2k',
    }
    raw.update(kwargs)
    return normalize(build_config(raw))


class TestLocalScript:
    """The local script mirrors the cp2k_demo/run.sh pattern."""

    def test_environment_exports(self):
        config = _make_config(project='water')
        script = build_local_script(config)
        assert script.startswith('#!/bin/bash')
        assert 'export CP2K_PROJECT=water' in script
        assert 'export CP2K_HOME=/opt/cp2k' in script
        assert 'export CP2K_EXE=$CP2K_HOME/exe/local/cp2k.psmp' in script
        assert 'export OMP_NUM_THREADS=' in script

    def test_mpirun_command_merges_stderr(self):
        # Both accepted spellings of the rank count fold to the larger
        # value (#306): nproc is what a standalone input writes,
        # cp2k_nproc what the framework forwards from &cp2k.
        config = _make_config(nproc=4)
        script = build_local_script(config)
        assert ('mpirun -n 4 $CP2K_EXE -i $CP2K_PROJECT.inp '
                '-o $CP2K_PROJECT.log 2>&1') in script

    def test_custom_exe_and_setup_script(self):
        config = _make_config(cp2k_exe='/usr/local/bin/cp2k.ssmp',
                              setup_script='/opt/cp2k/tools/toolchain/'
                                           'install/setup')
        script = build_local_script(config)
        assert 'export CP2K_EXE=/usr/local/bin/cp2k.ssmp' in script
        assert 'source /opt/cp2k/tools/toolchain/install/setup' in script

    def test_shell_commands_passthrough(self):
        config = _make_config(shell=['module load cp2k/2024.1'])
        script = build_local_script(config)
        assert 'module load cp2k/2024.1' in script


class TestSlurmScript:
    """The SLURM script carries the scheduler header plus the same
    CP2K command block."""

    def test_slurm_header_and_commands(self):
        config = _make_config(project='water', slurm_partition='normal',
                              slurm_time='24:00:00', slurm_ntasks=8,
                              slurm_memory=16000)
        script = build_slurm_script(config)
        assert '#SBATCH --partition=normal' in script
        assert '#SBATCH --time=24:00:00' in script
        assert '#SBATCH --ntasks=8' in script
        assert '#SBATCH --job-name=water' in script
        assert 'mpirun -n' in script
        assert 'export CP2K_HOME=/opt/cp2k' in script

    def test_user_template_is_respected(self):
        config = _make_config()
        template = '#!/bin/bash\n#SBATCH --account=myproject\n'
        script = build_slurm_script(config, slurm_template=template)
        assert '#SBATCH --account=myproject' in script
        assert 'mpirun -n' in script


def test_rank_spellings_fold_to_the_larger():
    """nproc and cp2k_nproc ask for the same resource; max wins (#306)."""
    from ..schema.config import Environment

    assert Environment(nproc=4).nproc == 4
    assert Environment(cp2k_nproc=6).nproc == 6
    assert Environment(nproc=6, cp2k_nproc=2).nproc == 6
    assert Environment(cp2k_thread=5).thread == 5
    assert Environment(thread=5, cp2k_thread=3).thread == 5


def test_retired_names_are_refused():
    import pytest

    from ..schema.adapter import build_config

    for retired in ('num_thread', 'mpi'):
        with pytest.raises(ValueError, match=retired):
            build_config({'skill_name': 'cp2k_skill', retired: 4})
