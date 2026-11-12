"""
Unit tests for xtb_skill run script generation (run_templates).

Verifies the environment export block of the generated bash/slurm
scripts: XTBHOME is the installation root and XTBPATH its
{install}/share/xtb parameter/data directory whenever the xtb_path
install location is known, and the skill-level shell commands ride
into the script ahead of the exports.
These tests do NOT require an xTB installation.

Run: pytest xtb_skill/tests/test_run_script.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.templates.run_templates import (
    build_local_script,
    build_slurm_script,
)

# Water molecule keeps the config minimal
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
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': 'gfn2_energy',
        'method': 'gfn2',
    }
    raw.update(kwargs)
    return normalize(build_config(raw))


class TestXtbEnvExport:
    """XTBHOME/XTBPATH follow the legacy runscript contract.

    XTBHOME is the installation root and XTBPATH the
    {install}/share/xtb parameter/data directory; both are emitted
    whenever the install location can be derived from xtb_path
    (installation directory or binary path); a bare command name
    resolved via PATH carries neither.
    """

    def test_install_directory_yields_both_exports(self, tmp_path):
        # tmp_path is a real directory -> treated as the install root
        config = _make_config(xtb_path=str(tmp_path))
        script = build_local_script(config)
        assert f'export XTBHOME={tmp_path}' in script
        assert 'export XTBPATH=$XTBHOME/share/xtb' in script
        assert f'{tmp_path}/bin/xtb' in script

    def test_binary_path_derives_install_root(self):
        binary = '/opt/xtb-6.7.1/bin/xtb'
        config = _make_config(xtb_path=binary)
        script = build_local_script(config)
        assert 'export XTBHOME=/opt/xtb-6.7.1' in script
        assert 'export XTBPATH=$XTBHOME/share/xtb' in script
        assert binary in script

    def test_empty_path_skips_both_exports(self):
        config = _make_config(xtb_path='')
        script = build_local_script(config)
        assert 'XTBHOME' not in script
        assert 'XTBPATH' not in script
        # Falls back to the PATH-resolved 'xtb' command
        assert '\nxtb ' in script

    def test_slurm_script_carries_both_exports(self, tmp_path):
        config = _make_config(xtb_path=str(tmp_path))
        script = build_slurm_script(config)
        assert f'export XTBHOME={tmp_path}' in script
        assert 'export XTBPATH=$XTBHOME/share/xtb' in script


class TestShellSetup:
    """Skill-level shell commands are copied ahead of the exports."""

    def test_shell_commands_precede_exports(self, tmp_path):
        config = _make_config(
            xtb_path=str(tmp_path),
            shell='module load gcc/11.3.0')
        script = build_local_script(config)
        assert 'module load gcc/11.3.0' in script
        assert (script.index('module load')
                < script.index('export XTB_PROJECT'))
