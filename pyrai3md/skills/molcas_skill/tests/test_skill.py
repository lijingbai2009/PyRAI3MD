"""
End-to-end tests for molcas_skill config refactoring.

Tests the SkillConfig nested model with flat access convenience layer,
verifying standalone mode (dict input) and service function compatibility.
These tests do NOT require a Molcas installation.

Run: pytest molcas_skill/tests/test_skill.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import numpy as np
import pytest
from unittest.mock import patch

from ..schema.config import (
    SkillConfig, Config, System, Properties, Dynamics,
    Calculation, Environment, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES,
)
from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate


# Minimal valid configuration for a CASSCF energy calculation
MINIMAL_CONFIG = {
    'atoms': ['C', 'H', 'H'],
    'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [0.0, 0.0, -1.1]],
    'charges': 0,
    'multiplicity': 1,
    'nstate': 2,
    'state_spin': [0, 0],
    'profile': 'casscf_energy',
    'method': 'casscf',
    'basis': 'ano-rcc-vdzp',
    'nact_elec': 2,
    'nact_orb': 2,
    'molcas_path': '/tmp',  # must exist for validate_environment
}


@pytest.fixture
def skill_config():
    """Build a normalized, validated SkillConfig from minimal input."""
    raw = build_config(MINIMAL_CONFIG)
    norm = normalize(raw)
    return validate(norm)


# ===================================================================
# 1. Config nested construction
# ===================================================================

class TestConfigNestedConstruction:
    """Verify flat dict -> SkillConfig nested model construction."""

    def test_build_config_returns_skill_config(self):
        config = build_config(MINIMAL_CONFIG)
        assert isinstance(config, SkillConfig)

    def test_submodel_types(self, skill_config):
        assert isinstance(skill_config.system, System)
        assert isinstance(skill_config.properties, Properties)
        assert isinstance(skill_config.dynamics, Dynamics)
        assert isinstance(skill_config.calculation, Calculation)
        assert isinstance(skill_config.environment, Environment)

    def test_system_fields_populated(self, skill_config):
        assert skill_config.system.atoms == ['C', 'H', 'H']
        assert len(skill_config.system.coordinates) == 3
        assert skill_config.system.charges == 0
        assert skill_config.system.multiplicity == 1
        assert skill_config.system.nstate == 2

    def test_calculation_fields_populated(self, skill_config):
        assert skill_config.calculation.profile == 'casscf_energy'
        assert skill_config.calculation.method == 'casscf'
        assert skill_config.calculation.basis == 'ano-rcc-vdzp'
        assert skill_config.calculation.nact_elec == 2
        assert skill_config.calculation.nact_orb == 2

    def test_normalize_derived_fields(self, skill_config):
        # normalize computes total_electron, nalpha, nbeta, inactive_orb
        assert skill_config.system.total_electron == 8  # C(6) + 2*H(1)
        assert skill_config.system.nalpha == 4
        assert skill_config.system.nbeta == 4
        assert skill_config.system.natom == 3

    def test_backward_compat_alias(self):
        assert Config is SkillConfig


# ===================================================================
# 2. Flat access convenience layer
# ===================================================================

class TestFlatAccess:
    """Verify flat access routes to the correct submodel field."""

    def test_flat_access_system_fields(self, skill_config):
        assert skill_config.atoms == skill_config.system.atoms
        assert skill_config.coordinates == skill_config.system.coordinates
        assert skill_config.charges == skill_config.system.charges
        assert skill_config.multiplicity == skill_config.system.multiplicity

    def test_flat_access_calculation_fields(self, skill_config):
        assert skill_config.basis == skill_config.calculation.basis
        assert skill_config.profile == skill_config.calculation.profile
        assert skill_config.method == skill_config.calculation.method
        assert skill_config.nact_elec == skill_config.calculation.nact_elec

    def test_flat_access_environment_fields(self, skill_config):
        assert skill_config.molcas_project == skill_config.environment.molcas_project
        assert skill_config.molcas_path == skill_config.environment.molcas_path
        assert skill_config.execution_mode == skill_config.environment.execution_mode

    def test_flat_access_properties_fields(self, skill_config):
        assert skill_config.energy == skill_config.properties.energy

    def test_flat_access_dynamics_fields(self, skill_config):
        assert skill_config.temperature == skill_config.dynamics.temperature

    def test_flat_access_nonexistent_raises(self, skill_config):
        with pytest.raises(AttributeError):
            _ = skill_config.nonexistent_field

    def test_flat_access_model_prefix_raises(self, skill_config):
        with pytest.raises(AttributeError):
            _ = skill_config.model_fields_custom


# ===================================================================
# 3. Dynamic updates via attribute access
# ===================================================================

class TestDynamicUpdates:
    """Verify attribute assignment on nested submodels."""

    def test_update_atoms(self, skill_config):
        new_atoms = ['N', 'H']
        skill_config.system.atoms = new_atoms
        assert skill_config.system.atoms == new_atoms
        assert skill_config.atoms == new_atoms  # flat access reflects change

    def test_update_coordinates(self, skill_config):
        new_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        skill_config.system.coordinates = new_coords
        assert skill_config.coordinates == new_coords

    def test_update_target_states(self, skill_config):
        skill_config.calculation.target_states = [1]
        assert skill_config.target_states == [1]

    def test_update_point_charges(self, skill_config):
        pcs = [[1.0, 5.0, 0.0, 0.0]]
        skill_config.system.point_charges = pcs
        assert skill_config.point_charges == pcs

    def test_update_scratch_path(self, skill_config):
        skill_config.environment.scratch_path = '/tmp/test'
        assert skill_config.scratch_path == '/tmp/test'


# ===================================================================
# 4. FIELD_TO_SUBMODEL mapping
# ===================================================================

class TestFieldToSubmodel:
    """Verify the FIELD_TO_SUBMODEL mapping is correct."""

    def test_all_fields_mapped(self):
        # Every field in every submodel should appear in FIELD_TO_SUBMODEL
        for sub_name, sub_cls in SUBMODEL_CLASSES.items():
            for field_name in sub_cls.model_fields:
                assert field_name in FIELD_TO_SUBMODEL, \
                    f'{field_name} from {sub_name} missing in FIELD_TO_SUBMODEL'
                assert FIELD_TO_SUBMODEL[field_name] == sub_name

    def test_no_duplicate_fields(self):
        # If two submodels share a field name, the last one wins.
        # This is expected behavior (same as PyRAIConfig).
        all_fields = []
        for sub_cls in SUBMODEL_CLASSES.values():
            all_fields.extend(sub_cls.model_fields)
        # FIELD_TO_SUBMODEL has unique keys (dict), so len may differ
        # Just verify the mapping is consistent
        for field, parent in FIELD_TO_SUBMODEL.items():
            assert parent in SUBMODEL_CLASSES


# ===================================================================
# 5. SkillConfig construction from flat kwargs
# ===================================================================

class TestSkillConfigInit:
    """Verify SkillConfig.__init__ auto-routes flat kwargs."""

    def test_flat_kwargs_routing(self):
        config = SkillConfig(basis='sto-3g', atoms=['H', 'H'])
        assert config.calculation.basis == 'sto-3g'
        assert config.system.atoms == ['H', 'H']

    def test_nested_kwargs_routing(self):
        config = SkillConfig(
            system={'atoms': ['O']},
            calculation={'basis': 'cc-pvdz'},
        )
        assert config.system.atoms == ['O']
        assert config.calculation.basis == 'cc-pvdz'

    def test_mixed_kwargs(self):
        config = SkillConfig(
            atoms=['N'],
            calculation={'basis': '6-31g'},
            charges=1,
        )
        assert config.system.atoms == ['N']
        assert config.system.charges == 1
        assert config.calculation.basis == '6-31g'


# ===================================================================
# 6. Service functions with SkillConfig
# ===================================================================

class TestServiceFunctions:
    """Verify service functions accept SkillConfig via flat access."""

    def test_build_gateway(self, skill_config):
        from ..service.builder.module_builders import build_gateway
        result = build_gateway(skill_config)
        assert isinstance(result, str)
        assert 'GATEWAY' in result
        assert skill_config.basis in result

    def test_build_seward(self, skill_config):
        from ..service.builder.module_builders import build_seward
        result = build_seward(skill_config)
        assert 'SEWARD' in result

    def test_build_rasscf(self, skill_config):
        from ..service.builder.module_builders import build_rasscf
        result = build_rasscf(skill_config)
        assert 'RASSCF' in result
        assert 'Spin=' in result

    def test_build_rassi_soc(self, skill_config):
        from ..service.builder.module_builders import build_rassi
        result = build_rassi(skill_config, soc=True, dual_rasscf=False)
        assert 'RASSI' in result

    def test_build_caspt2(self, skill_config):
        from ..service.builder.module_builders import build_caspt2
        result = build_caspt2(skill_config)
        assert 'CASPT2' in result

    def test_build_slapaf(self, skill_config):
        from ..service.builder.module_builders import build_slapaf
        result = build_slapaf(skill_config)
        assert 'SLAPAF' in result

    def test_build_dynamix(self, skill_config):
        from ..service.builder.module_builders import build_dynamix
        result = build_dynamix(skill_config)
        assert 'Dynamix' in result

    def test_write_xyz_file(self, skill_config, tmp_path):
        from ..service.builder.input_builder import write_xyz_file
        write_xyz_file(skill_config, str(tmp_path))
        project = skill_config.project
        xyz_path = tmp_path / f'{project}.xyz'
        assert xyz_path.exists()
        content = xyz_path.read_text()
        assert str(len(skill_config.atoms)) in content.split('\n')[0]

    def test_resolve_workdir_returns_string(self, skill_config):
        from ..service.runner.env_check import resolve_workdir
        workdir = resolve_workdir(skill_config, '')
        assert isinstance(workdir, str)
        assert len(workdir) > 0

    def test_resolve_workdir_separates_concurrent_runs(self, skill_config):
        # The AUTO path used to be keyed only by the term (the calculation
        # directory's basename), which repeats across examples and slurm
        # jobs; two jobs on one node then shared and deleted the same
        # runfile. The run id gives each calculation directory its own root,
        # and the same directory keeps the same root across restarts.
        from ..service.runner.env_check import resolve_workdir
        one = resolve_workdir(skill_config, '', term='tmp_high_qm_MOLCAS',
                              run_id='/work/example_a/yaml/tmp_high_qm_MOLCAS')
        two = resolve_workdir(skill_config, '', term='tmp_high_qm_MOLCAS',
                              run_id='/work/example_b/yaml/tmp_high_qm_MOLCAS')
        again = resolve_workdir(skill_config, '',
                                term='tmp_high_qm_MOLCAS',
                                run_id='/work/example_a/yaml/tmp_high_qm_MOLCAS')
        assert one != two
        assert one == again
        assert one.endswith('tmp_high_qm_MOLCAS')

    def test_get_runner_returns_callable(self, skill_config):
        from ..service.runner import get_runner
        runner = get_runner(skill_config)
        assert callable(runner)

    def test_build_local_script(self, skill_config):
        from ..service.templates.run_templates import build_local_script
        script = build_local_script(skill_config, workdir='/tmp/test')
        assert '#!/bin/bash' in script
        assert 'MOLCAS_PROJECT=' in script
        assert '/tmp/test' in script

    def test_build_slurm_script(self, skill_config):
        from ..service.templates.run_templates import build_slurm_script
        script = build_slurm_script(skill_config, workdir='/tmp/test')
        assert '#!/bin/bash' in script
        assert '#SBATCH' in script


# ===================================================================
# 7. Backend initialization (standalone mode)
# ===================================================================

class TestBackendInit:
    """Verify backend initialization without Molcas installation."""

    @patch('pyrai3md.skills.molcas_skill.backend.validate_environment')
    def test_backend_init_from_dict(self, mock_validate, tmp_path):
        """Backend initializes from a flat dict (standalone mode)."""
        from ..backend import MolcasBackend

        config = {**MINIMAL_CONFIG, 'molcas_path': '/tmp'}
        backend = MolcasBackend(
            config=config,
            calcdir=str(tmp_path / 'calc'),
        )

        assert isinstance(backend.config, SkillConfig)
        assert backend.config.system.atoms == ['C', 'H', 'H']
        assert backend.config.calculation.profile == 'casscf_energy'
        assert hasattr(backend, 'workdir')
        assert hasattr(backend, 'workflow')
        assert hasattr(backend, 'run')

    @patch('pyrai3md.skills.molcas_skill.backend.validate_environment')
    def test_backend_no_config_dict(self, mock_validate, tmp_path):
        """Backend holds SkillConfig, not config_dict."""
        from ..backend import MolcasBackend

        config = {**MINIMAL_CONFIG, 'molcas_path': '/tmp'}
        backend = MolcasBackend(
            config=config,
            calcdir=str(tmp_path / 'calc'),
        )

        # Verify no config_dict attribute
        assert not hasattr(backend, 'config_dict')
        # Verify config is SkillConfig
        assert isinstance(backend.config, SkillConfig)

    @patch('pyrai3md.skills.molcas_skill.backend.validate_environment')
    def test_backend_dynamic_update(self, mock_validate, tmp_path):
        """Backend supports dynamic attribute updates."""
        from ..backend import MolcasBackend

        config = {**MINIMAL_CONFIG, 'molcas_path': '/tmp'}
        backend = MolcasBackend(
            config=config,
            calcdir=str(tmp_path / 'calc'),
        )

        # Simulate what calculate() does
        new_atoms = ['N', 'H', 'H']
        new_coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [0.0, 0.0, -1.2]]
        backend.config.system.atoms = new_atoms
        backend.config.system.coordinates = new_coords

        assert backend.config.system.atoms == new_atoms
        assert backend.config.atoms == new_atoms  # flat access works


# ===================================================================
# 8. Geometry optimization input assembly (single/MECI/MECP)
# ===================================================================

class TestOptimizeInput:
    """Verify build_input assembles the optimization loop correctly."""

    @staticmethod
    def _optimize_config(**overrides):
        """Build a validated optimization config from the MECI yaml."""
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / 'casscf_meci.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw.update(overrides)
        config = validate(normalize(build_config(raw)))
        workflow = compile_workflow(config.profile)
        return config, workflow, build_input(config, workflow)

    def test_single_state_loop_structure(self):
        config, _, inp = self._optimize_config(
            opt_state_i=0, opt_state_j=0, opt_maxit=50)
        lines = inp.split('\n')
        assert lines[0] == '>> EXPORT MOLCAS_MAXITER=50'
        assert '>> Do while' in lines
        assert '>> End Do' in lines
        # GATEWAY stays outside the loop, SLAPAF closes it
        assert lines.index('>> Do while') > lines.index('&GATEWAY')
        assert lines.index('>> End Do') > lines.index('&SLAPAF')
        assert 'Rlxroot=1' in inp
        assert 'Constraints' not in inp
        assert inp.count('&RASSCF') == 1
        assert 'ITERATIONS=200,100' in inp
        # No staged StrOrb guess: RASSCF keeps its default orbitals
        assert 'fileorb=' not in inp
        # Single-state relaxation needs no ALASKA before SLAPAF
        assert '&ALASKA' not in inp

    def test_meci_constraint_and_rlxroot(self):
        config, _, inp = self._optimize_config(
            opt_state_i=1, opt_state_j=2, opt_maxit=5)
        assert '>> EXPORT MOLCAS_MAXITER=5' in inp
        # Same-spin crossing: Ediff carries the 1-based root pair
        assert 'a = Ediff 2 3' in inp
        assert 'Rlxroot=2' in inp
        assert inp.count('&RASSCF') == 1
        assert 'Iterations=5' in inp
        # Same-spin crossing relaxation needs no ALASKA before SLAPAF
        assert '&ALASKA' not in inp

    def test_mecp_dual_rasscf(self):
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / 'casscf_mecp.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        config = validate(normalize(build_config(raw)))
        inp = build_input(config, compile_workflow(config.profile))

        # Cross-spin crossing: bare Ediff constraint, no root pair
        assert 'a = Ediff\n' in inp
        assert 'Ediff ' not in inp.replace('End of', '')
        # Dual RASSCF jobs, singlet before triplet, own Rlxroot each;
        # the triplet root is renumbered by the singlet count.
        assert inp.count('&RASSCF') == 2
        assert inp.index('Spin=1') < inp.index('Spin=3')
        assert inp.count('Rlxroot=1') == 2
        assert 'RASSI' not in inp
        # Each spin job is followed by a bare ALASKA; the singlet
        # RunFile is locked as RUNFILE2 before the triplet job.
        assert inp.count('&ALASKA') == 2
        first_alaska = inp.index('&ALASKA')
        last_alaska = inp.rindex('&ALASKA')
        copy_runfile = inp.index('$WorkDir/RUNFILE2')
        assert inp.index('Spin=1') < first_alaska
        assert first_alaska < copy_runfile < inp.index('Spin=3')
        assert inp.index('Spin=3') < last_alaska
        assert last_alaska < inp.index('&SLAPAF')

    def test_mecp_spin_inverted_raises(self):
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / 'casscf_mecp.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw.update(opt_state_i=1, opt_state_j=0)  # triplet first
        config = validate(normalize(build_config(raw)))
        with pytest.raises(ValueError, match='singlet'):
            build_input(config, compile_workflow(config.profile))

    def test_opt_state_out_of_range_raises(self):
        with pytest.raises(ValueError, match='out of range'):
            self._optimize_config(opt_state_j=5)

    def test_ci_roots_insufficient_raises(self):
        # Bypass normalize (which derives ci_roots from state_spin)
        # to hit the coverage check directly.
        import yaml
        from pathlib import Path
        from ..service.builder.input_builder import _resolve_optimize_mode

        yaml_path = Path(__file__).parent / 'casscf_meci.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        config = build_config(raw)
        config.calculation.ci_roots = [2]  # needs root 3 for state j=2
        config.calculation.opt_state_j = 2
        with pytest.raises(ValueError, match='ci_roots'):
            _resolve_optimize_mode(config)


# ===================================================================
# 9. Dynamics input assembly (ab initio MD / surface hopping)
# ===================================================================

class TestDynamicsInput:
    """Verify build_input assembles the FOREACH dynamics loop."""

    @staticmethod
    def _dynamics_input(yaml_name='casscf_abinitio_dynamics', **overrides):
        """Build a validated dynamics input from a test yaml."""
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / f'{yaml_name}.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw.update(overrides)
        config = validate(normalize(build_config(raw)))
        return config, build_input(config, compile_workflow(config.profile))

    def test_abinitio_loop_structure(self):
        config, inp = self._dynamics_input()
        lines = inp.split('\n')
        # Header: copy coordinates/velocities, export the step limit
        assert lines[0] == ('>> COPY $MOLCAS_SUBMIT_DIR/$Project.xyz '
                            '$WorkDir/$Project.xyz')
        assert lines[1] == ('>> COPY $MOLCAS_SUBMIT_DIR/'
                            '$Project.velocity.xyz '
                            '$WorkDir/$Project.velocity.xyz')
        # MOLCAS_MAXITER auto limit: maxit_md x 10
        assert lines[2] == '>> EXPORT MOLCAS_MAXITER=100'
        # FOREACH loop wraps SEWARD..DYNAMIX, GATEWAY stays outside
        assert '>> FOREACH ITER in (1 .. 10)' in lines
        assert lines.index('>> FOREACH ITER in (1 .. 10)') > lines.index('&GATEWAY')
        assert lines.index('>> End do') > lines.index('&Dynamix')
        # RASSCF carries MDRLXR (current_state=0 -> S0); without a
        # staged StrOrb guess no fileorb line is emitted
        assert 'MDRLXR=1' in inp
        assert 'fileorb=' not in inp
        assert '$Project.StrOrb' not in inp
        # ALASKA stays bare inside the MD loop
        assert '&ALASKA\n' in inp
        assert 'ROOT=' not in inp
        # DYNAMIX keywords
        assert 'velver' in inp
        assert 'dt = 20.0' in inp
        assert 'velo = 1' in inp
        assert 'ther = 0' in inp
        assert 'Surfacehop' not in inp

    def test_surface_hopping_block(self):
        config, inp = self._dynamics_input('casscf_surface_hopping')
        assert '&Surfacehop' in inp
        assert 'tully' in inp
        assert 'decoherence = 0.1' in inp
        assert 'psub' in inp

    def test_md_maxiter_explicit(self):
        config, inp = self._dynamics_input(md_maxiter=200)
        assert '>> EXPORT MOLCAS_MAXITER=200' in inp

    def test_nose_hoover_thermostat(self):
        config, inp = self._dynamics_input(thermostat='nose-hoover')
        assert 'ther = 2' in inp

    def test_missing_velocities_raises(self):
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / 'casscf_surface_hopping.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw.pop('velocities')
        config = validate(normalize(build_config(raw)))
        with pytest.raises(ValueError, match='velocities'):
            build_input(config, compile_workflow(config.profile))

    def test_unsupported_thermostat_raises(self):
        with pytest.raises(ValueError, match='thermostat'):
            self._dynamics_input(thermostat='langevin')

    def test_md_maxiter_below_steps_raises(self):
        with pytest.raises(ValueError, match='md_maxiter'):
            self._dynamics_input(md_maxiter=5)

    def test_write_velocity_file(self, tmp_path):
        import yaml
        from pathlib import Path
        from ..service.builder.input_builder import write_velocity_file

        yaml_path = Path(__file__).parent / 'casscf_abinitio_dynamics.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        config = validate(normalize(build_config(raw)))
        write_velocity_file(config, str(tmp_path))
        velo_path = tmp_path / f'{config.project}.velocity.xyz'
        assert velo_path.exists()
        content = [l for l in velo_path.read_text().split('\n') if l.strip()]
        # Plain N x 3 numeric block: no atom count line, no comment
        # line, no element-symbol column
        assert len(content) == len(config.atoms)
        for line in content:
            values = line.split()
            assert len(values) == 3
            [float(v) for v in values]


# ===================================================================
# 10. Frequency input assembly (MCKINLEY numerical hessian)
# ===================================================================

class TestFrequencyInput:
    """Verify build_input exports a generous MOLCAS_MAXITER budget
    for frequency calculations."""

    @staticmethod
    def _frequency_input(yaml_name='casscf_frequency', **overrides):
        """Build a validated frequency input from a test yaml."""
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / f'{yaml_name}.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw.update(overrides)
        config = validate(normalize(build_config(raw)))
        return config, build_input(config, compile_workflow(config.profile))

    def test_maxiter_export_is_first_line(self):
        config, inp = self._frequency_input()
        lines = inp.split('\n')
        # N = 100 x natom keeps intermediate iteration budgets
        # generous while MCKINLEY displaces every atom.
        assert lines[0] == f'>> EXPORT MOLCAS_MAXITER={config.natom * 100}'
        assert lines[0] == '>> EXPORT MOLCAS_MAXITER=600'
        assert '&MCKINLEY' in inp


# ===================================================================
# 11. Final structure collection (fin_coord channel)
# ===================================================================

class TestFinCoord:
    """Verify parse_fin_coord collects the final structure for
    optimize/dynamics profiles through the framework fin_coord
    channel."""

    @staticmethod
    def _load_config(yaml_name):
        import yaml
        from pathlib import Path

        yaml_path = Path(__file__).parent / f'{yaml_name}.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        return validate(normalize(build_config(raw)))

    @staticmethod
    def _write_xyz(path, atoms, frames):
        """Write a multi-frame XYZ trajectory file."""
        with open(path, 'w') as f:
            for k, frame in enumerate(frames):
                f.write(f'{len(atoms)}\n')
                f.write(f'frame {k}\n')
                for atom, coord in zip(atoms, frame):
                    f.write(f'{atom:>4s}  {coord[0]:16.10f}  '
                            f'{coord[1]:16.10f}  {coord[2]:16.10f}\n')

    def test_optimize_reads_opt_xyz(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        config = self._load_config('casscf_geometry_optimization')
        final = [[0.1 * i, 0.2 * i, 0.3 * i]
                 for i in range(config.natom)]
        self._write_xyz(tmp_path / f'{config.project}.opt.xyz',
                        config.atoms, [final])
        coords = parse_fin_coord(config, str(tmp_path))
        assert np.allclose(coords, final)

    def test_optimize_reads_capital_Opt_xyz(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        config = self._load_config('casscf_geometry_optimization')
        final = [[0.1 * i, 0.2 * i, 0.3 * i]
                 for i in range(config.natom)]
        self._write_xyz(tmp_path / f'{config.project}.Opt.xyz',
                        config.atoms, [final])
        coords = parse_fin_coord(config, str(tmp_path))
        assert np.allclose(coords, final)

    def test_dynamics_reads_last_frame(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        config = self._load_config('casscf_abinitio_dynamics')
        frames = []
        for step in range(3):
            frames.append([[step + 0.1 * i, step + 0.2 * i,
                            step + 0.3 * i]
                           for i in range(config.natom)])
        self._write_xyz(tmp_path / f'{config.project}.md.xyz',
                        config.atoms, frames)
        coords = parse_fin_coord(config, str(tmp_path))
        assert np.allclose(coords, frames[-1])

    def test_surface_hopping_reads_md_xyz(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        config = self._load_config('casscf_surface_hopping')
        final = [[1.0 + i, 2.0, 3.0] for i in range(config.natom)]
        self._write_xyz(tmp_path / f'{config.project}.md.xyz',
                        config.atoms, [final])
        coords = parse_fin_coord(config, str(tmp_path))
        assert np.allclose(coords, final)

    def test_non_iterative_profile_returns_empty(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        # casscf_energy performs no internal iteration: no file read
        config = validate(normalize(build_config(MINIMAL_CONFIG)))
        coords = parse_fin_coord(config, str(tmp_path))
        assert len(coords) == 0

    def test_missing_file_raises(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        config = self._load_config('casscf_geometry_optimization')
        with pytest.raises(RuntimeError, match='[Oo]pt.xyz'):
            parse_fin_coord(config, str(tmp_path))

    def test_not_converged_missing_file_returns_empty(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        # SLAPAF stops at opt_maxit without writing Opt.xyz; that is
        # the reported opt_converge=False outcome, not a read failure.
        config = self._load_config('casscf_geometry_optimization')
        coords = parse_fin_coord(config, str(tmp_path), not_converged=True)
        assert len(coords) == 0

    def test_natom_mismatch_raises(self, tmp_path):
        from ..service.parser.output_parser import parse_fin_coord

        config = self._load_config('casscf_geometry_optimization')
        bad = [[0.0, 0.0, 0.0]]  # wrong atom count
        self._write_xyz(tmp_path / f'{config.project}.opt.xyz',
                        ['C'], [bad])
        with pytest.raises(ValueError, match='atoms'):
            parse_fin_coord(config, str(tmp_path))

    def test_parse_output_returns_fin_coord(self, tmp_path):
        from ..service.parser import output_parser

        config = self._load_config('casscf_geometry_optimization')
        final = [[0.5 * i, 0.5 * i, 0.5 * i]
                 for i in range(config.natom)]
        self._write_xyz(tmp_path / f'{config.project}.opt.xyz',
                        config.atoms, [final])
        with patch.object(output_parser, 'parse_log',
                          return_value={'energy': [-75.0]}), \
             patch.object(output_parser, 'manage_orbital_files'):
            results = output_parser.parse_output(
                config, str(tmp_path), workflow=None)
        assert 'fin_coord' in results
        assert np.allclose(results['fin_coord'], final)

    def test_parse_output_not_converged_omits_fin_coord(self, tmp_path):
        from ..service.parser import output_parser

        config = self._load_config('casscf_geometry_optimization')
        with patch.object(output_parser, 'parse_log',
                          return_value={'energy': [-75.0],
                                        'log_not_converged': True}), \
             patch.object(output_parser, 'manage_orbital_files'):
            results = output_parser.parse_output(
                config, str(tmp_path), workflow=None)
        assert results['opt_converge'] is False
        assert 'fin_coord' not in results

    def test_parse_output_omits_fin_coord_for_energy(self, tmp_path):
        from ..service.parser import output_parser

        config = validate(normalize(build_config(MINIMAL_CONFIG)))
        with patch.object(output_parser, 'parse_log',
                          return_value={'energy': [-75.0]}), \
             patch.object(output_parser, 'manage_orbital_files'):
            results = output_parser.parse_output(
                config, str(tmp_path), workflow=None)
        assert 'fin_coord' not in results


# ===================================================================
# 12. Initial orbital guess (title.StrOrb -> RasOrb channel)
# ===================================================================

class TestInitialOrbital:
    """Verify external orbital guesses are staged as RasOrb and
    RASSCF reads fileorb=$MOLCAS_PROJECT.RasOrb."""

    @staticmethod
    def _write_guess(tmp_path, name='guess.StrOrb', content='ORBITALS'):
        guess = tmp_path / name
        guess.write_text(content)
        return str(guess)

    @staticmethod
    def _build_input(raw):
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        config = validate(normalize(build_config(raw)))
        return config, build_input(config, compile_workflow(config.profile))

    def test_manage_orbitals_copies_as_rasorb(self, tmp_path):
        from ..service.builder.input_builder import manage_orbitals

        guess = self._write_guess(tmp_path)
        raw = dict(MINIMAL_CONFIG, guess_orbital=guess)
        config = validate(normalize(build_config(raw)))
        calcdir = tmp_path / 'calcdir'
        calcdir.mkdir()
        manage_orbitals(config, str(calcdir))
        rasorb = calcdir / f'{config.project}.RasOrb'
        assert rasorb.exists()
        assert rasorb.read_text() == 'ORBITALS'
        # The user guess is additionally locked under the StrOrb
        # suffix so the dynamics COPY line finds it in the submit dir
        strorb = calcdir / f'{config.project}.StrOrb'
        assert strorb.exists()
        assert strorb.read_text() == 'ORBITALS'

    def test_manage_orbitals_no_overwrite(self, tmp_path):
        from ..service.builder.input_builder import manage_orbitals

        guess = self._write_guess(tmp_path, content='EXTERNAL')
        raw = dict(MINIMAL_CONFIG, guess_orbital=guess)
        config = validate(normalize(build_config(raw)))
        calcdir = tmp_path / 'calcdir'
        calcdir.mkdir()
        # RasOrb converged by an earlier step must be preserved
        rasorb = calcdir / f'{config.project}.RasOrb'
        rasorb.write_text('CONVERGED')
        manage_orbitals(config, str(calcdir))
        assert rasorb.read_text() == 'CONVERGED'

    def test_manage_orbitals_jobiph_keeps_name(self, tmp_path):
        from ..service.builder.input_builder import manage_orbitals

        guess = self._write_guess(tmp_path, name='guess.JobIph')
        raw = dict(MINIMAL_CONFIG, guess_orbital=guess)
        config = validate(normalize(build_config(raw)))
        calcdir = tmp_path / 'calcdir'
        calcdir.mkdir()
        manage_orbitals(config, str(calcdir))
        assert (calcdir / 'guess.JobIph').exists()
        assert not (calcdir / f'{config.project}.RasOrb').exists()

    def test_single_point_fileorb_rasorb(self, tmp_path):
        guess = self._write_guess(tmp_path)
        raw = dict(MINIMAL_CONFIG, guess_orbital=guess)
        _, inp = self._build_input(raw)
        assert 'fileorb=$MOLCAS_PROJECT.RasOrb' in inp
        assert 'fileorb=$MOLCAS_PROJECT.StrOrb' not in inp

    def test_optimize_with_guess_reads_rasorb(self, tmp_path):
        import yaml
        from pathlib import Path

        guess = self._write_guess(tmp_path)
        yaml_path = Path(__file__).parent / 'casscf_geometry_optimization.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw['guess_orbital'] = guess
        _, inp = self._build_input(raw)
        # Optimization loop continues from the staged RasOrb
        assert inp.count('fileorb=$MOLCAS_PROJECT.RasOrb') == 1
        assert 'fileorb=$MOLCAS_PROJECT.StrOrb' not in inp

    def test_optimize_without_guess_no_fileorb(self):
        import yaml
        from pathlib import Path

        yaml_path = Path(__file__).parent / 'casscf_geometry_optimization.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        _, inp = self._build_input(raw)
        # No staged guess: the Do-while loop starts from default
        # orbitals and no fileorb line is written
        assert 'fileorb=' not in inp

    def test_mecp_with_guess_both_jobs_read_rasorb(self, tmp_path):
        import yaml
        from pathlib import Path

        guess = self._write_guess(tmp_path)
        yaml_path = Path(__file__).parent / 'casscf_mecp.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw['guess_orbital'] = guess
        _, inp = self._build_input(raw)
        # Both singlet and triplet RASSCF jobs read the staged guess
        assert inp.count('fileorb=$MOLCAS_PROJECT.RasOrb') == 2

    def test_dynamics_with_guess_single_fileorb(self, tmp_path):
        import yaml
        from pathlib import Path

        guess = self._write_guess(tmp_path)
        yaml_path = Path(__file__).parent / 'casscf_abinitio_dynamics.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw['guess_orbital'] = guess
        _, inp = self._build_input(raw)
        # MDRLXR continuation line only; no duplicate from the guess
        assert inp.count('fileorb=$MOLCAS_PROJECT.RasOrb') == 1
        assert 'MDRLXR=1' in inp

    @staticmethod
    def _deck(raw, calcdir=None):
        from ..service.builder.input_builder import build_input
        from ..service.workflow.compiler import compile_workflow

        config = validate(normalize(build_config(raw)))
        workflow = compile_workflow(config.profile, config=config)
        return config, build_input(config, workflow, calcdir=calcdir)

    def test_a_dual_deck_guess_reaches_every_spin_pass(self, tmp_path):
        """A spin-labelled pass must read the staged guess too (#448).

        The dual deck runs the singlet group before the triplet group
        against one {project}.RasOrb, which is the design: the second
        pass continues from the first and `manage_orbitals` saves the
        last RasOrb back as the next call's StrOrb. The `fileorb` line
        only ever served the unlabelled template, the Rlxroot line and
        the MDRLXR line, so both passes of a dual deck ran on Molcas'
        default orbitals with a guess staged - and said nothing.
        """
        import yaml
        from pathlib import Path

        with open(Path(__file__).parent / 'casscf_soc.yaml') as f:
            shipped = yaml.safe_load(f)['input']
        guess = self._write_guess(tmp_path)

        for label, raw in (('generic', dict(shipped)),
                           ('legacy casscf_soc',
                            dict(shipped, profile='casscf_soc'))):
            with_guess = self._deck(dict(raw, guess_orbital=guess))[1]
            without = self._deck(raw)[1]
            blocks = with_guess.count('&RASSCF')
            assert blocks == 2, (label, blocks)
            # One continuation line per pass, no duplicates
            assert with_guess.count('fileorb=$MOLCAS_PROJECT.RasOrb') == blocks, label
            # A bare run must not claim an orbital file it was never given
            assert without.count('fileorb=$MOLCAS_PROJECT.RasOrb') == 0, label

    def test_a_missing_guess_warns_on_both_builder_paths(self, tmp_path):
        """The hazard notice must reach the generic path as well (#448).

        The legacy builder printed a line that never appeared for a
        generic/ESPF run, which is the path QMMM and the shipped NAC
        example take; a bare `print` also stays out of the warning
        channel the rest of the skill uses.
        """
        import warnings
        import yaml
        from pathlib import Path

        with open(Path(__file__).parent / 'casscf_soc.yaml') as f:
            shipped = yaml.safe_load(f)['input']

        for profile in ('casscf_properties', 'casscf_gradient'):
            calcdir = tmp_path / profile
            calcdir.mkdir()
            config, _ = self._deck(dict(shipped, profile=profile),
                                   calcdir=str(calcdir))
            with pytest.warns(UserWarning, match='StrOrb'):
                self._deck(dict(shipped, profile=profile),
                           calcdir=str(calcdir))
            (calcdir / f'{config.project}.StrOrb').write_text('ORBITALS')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self._deck(dict(shipped, profile=profile),
                           calcdir=str(calcdir))
            assert not [w for w in caught if 'StrOrb' in str(w.message)], profile

    def test_manage_orbitals_probes_cwd_strorb(self, tmp_path, monkeypatch):
        from ..service.builder.input_builder import manage_orbitals

        # No explicit guess: the {project}.StrOrb locked next to the
        # launching directory is staged into calcdir instead.
        config = validate(normalize(build_config(dict(MINIMAL_CONFIG))))
        monkeypatch.chdir(tmp_path)
        (tmp_path / f'{config.project}.StrOrb').write_text('CWDGUESS')
        calcdir = tmp_path / 'calcdir'
        calcdir.mkdir()
        manage_orbitals(config, str(calcdir))
        assert (calcdir / f'{config.project}.StrOrb').read_text() == 'CWDGUESS'
        assert (calcdir / f'{config.project}.RasOrb').read_text() == 'CWDGUESS'

    def test_dynamics_staged_strorb_copy_and_fileorb(self, tmp_path):
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        yaml_path = Path(__file__).parent / 'casscf_abinitio_dynamics.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        config = validate(normalize(build_config(raw)))
        workflow = compile_workflow(config.profile)
        calcdir = tmp_path / 'calcdir'
        calcdir.mkdir()
        # No guess staged: no StrOrb COPY, no fileorb line
        inp = build_input(config, workflow, calcdir=str(calcdir))
        assert '$Project.StrOrb' not in inp
        assert 'fileorb=' not in inp
        # Stage the locked StrOrb: the header COPY renames it to
        # RasOrb and RASSCF reads fileorb=$MOLCAS_PROJECT.RasOrb
        (calcdir / f'{config.project}.StrOrb').write_text('GUESS')
        inp = build_input(config, workflow, calcdir=str(calcdir))
        assert ('>> COPY $MOLCAS_SUBMIT_DIR/$Project.StrOrb '
                '$WorkDir/$Project.RasOrb') in inp
        assert 'fileorb=$MOLCAS_PROJECT.RasOrb' in inp
        assert 'MDRLXR=1' in inp

    def test_surface_hopping_current_state_selects_mdrlxr(self):
        import yaml
        from pathlib import Path
        from ..service.workflow.compiler import compile_workflow
        from ..service.builder.input_builder import build_input

        # The surface hopping yaml starts on S1 (current_state=1),
        # so MDRLXR must point at root 2.
        yaml_path = Path(__file__).parent / 'casscf_surface_hopping.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        config = validate(normalize(build_config(raw)))
        inp = build_input(config, compile_workflow(config.profile))
        assert 'MDRLXR=2' in inp
        assert '&Surfacehop' in inp


class TestResolveProfileSynonyms:
    """Synonymous task translation in resolve_profile: dipole is a
    by-product of the energy evaluation (merged with energy);
    transition keeps all other tasks and only sets the flag."""

    def test_dipole_translates_to_energy_profile(self):
        from ..schema.method import resolve_profile
        assert resolve_profile('casscf', 'dipole') == \
            ('casscf_energy', False)
        assert resolve_profile('casscf', 'energy dipole') == \
            ('casscf_energy', False)

    def test_dipole_collapses_with_other_tasks(self):
        from ..schema.method import resolve_profile
        assert resolve_profile('casscf', 'energy gradient dipole') == \
            ('casscf_gradient', False)
        assert resolve_profile('casscf', 'energy gradient nac dipole') == \
            ('casscf_nac', False)
        assert resolve_profile('casscf',
                               'energy gradient nac soc dipole') == \
            ('casscf_soc', False)

    def test_bare_transition_collapses_to_energy(self):
        from ..schema.method import resolve_profile
        # The transition dipoles come from the appended RASSI step,
        # so the profile task itself is the energy evaluation.
        assert resolve_profile('casscf', 'transition') == \
            ('casscf_energy', True)
        assert resolve_profile('caspt2', 'transition') == \
            ('caspt2_energy', True)

    def test_transition_keeps_other_tasks(self):
        from ..schema.method import resolve_profile
        assert resolve_profile('casscf', 'energy gradient transition') == \
            ('casscf_gradient', True)
        assert resolve_profile(
            'casscf', 'energy gradient nac transition') == \
            ('casscf_nac', True)
        assert resolve_profile(
            'casscf', 'energy gradient nac soc transition') == \
            ('casscf_soc', True)

    def test_order_independent(self):
        from ..schema.method import resolve_profile
        assert resolve_profile('casscf', 'dipole gradient energy') == \
            ('casscf_gradient', False)
        assert resolve_profile('casscf', 'transition gradient energy') == \
            ('casscf_gradient', True)

    def test_query_interface_returns_profile_string(self):
        from ..query import resolve_profile
        assert resolve_profile('casscf', 'energy gradient') == \
            'casscf_gradient'
        assert resolve_profile('casscf', 'energy transition') == \
            'casscf_energy'


class TestTransitionWorkflow:
    """A transition request appends an absorption-mode RASSI step to
    the profile module chain (energy template upgraded to the
    absorption one) without changing the resolved profile task."""

    def test_compile_workflow_appends_rassi(self):
        from ..service.workflow.compiler import compile_workflow
        wf = compile_workflow('casscf_energy', transition=True)
        assert wf.modules == ['GATEWAY', 'SEWARD', 'RASSCF', 'RASSI']
        assert wf.requires_rassi_absorption is True

    def test_compile_workflow_without_transition(self):
        from ..service.workflow.compiler import compile_workflow
        wf = compile_workflow('casscf_energy')
        assert wf.modules == ['GATEWAY', 'SEWARD', 'RASSCF']
        assert wf.requires_rassi_absorption is False

    def test_compile_workflow_gradient_keeps_alaska(self):
        from ..service.workflow.compiler import compile_workflow
        wf = compile_workflow('casscf_gradient', transition=True)
        assert wf.modules == [
            'GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA', 'RASSI']
        assert wf.requires_rassi_absorption is True

    def test_compile_workflow_rassi_profiles_unchanged(self):
        from ..service.workflow.compiler import compile_workflow
        # Chains that already end in RASSI keep their own chain;
        # absorption profiles run RASSI in absorption mode anyway.
        wf = compile_workflow('casscf_absorption', transition=True)
        assert wf.modules == ['GATEWAY', 'SEWARD', 'RASSCF', 'RASSI']
        assert wf.requires_rassi_absorption is True
        wf = compile_workflow('casscf_soc', transition=True)
        assert wf.modules == [
            'GATEWAY', 'SEWARD', 'RASSCF', 'ALASKA', 'RASSI']
        assert wf.requires_rassi_absorption is False

    def test_build_input_emits_absorption_rassi(self):
        from ..service.builder.input_builder import build_input
        from ..service.workflow.compiler import compile_workflow

        raw = dict(MINIMAL_CONFIG, jobtype='energy transition')
        config = validate(normalize(build_config(raw)))
        workflow = compile_workflow('casscf_energy', transition=True)
        inp = build_input(config, workflow)
        # RASSI runs in absorption mode (MEES + DIPRint=0)
        assert '>> RASSI' in inp or '&RASSI' in inp
        assert 'DIPRint=0' in inp
        assert 'MEES' in inp

    def test_backend_workflow_from_jobtype(self):
        from ..schema.method import has_transition_request
        assert has_transition_request('energy gradient transition')
        assert has_transition_request('transition')
        assert not has_transition_request('energy gradient')
        assert not has_transition_request('')
        assert not has_transition_request(None)


# ===================================================================
# 13. Deprecation warning
# ===================================================================

class TestDeprecation:
    """Verify flatten_extended_config emits deprecation warning."""

    def test_flatten_deprecation_warning(self):
        import warnings
        from ..service.utils import flatten_extended_config
        from pydantic import BaseModel, Field

        # Create a minimal mock to pass to flatten_extended_config
        class MockConfig(BaseModel):
            dynamics: dict = Field(default_factory=dict)
            system: dict = Field(default_factory=dict)
            properties: dict = Field(default_factory=dict)
            calculation: dict = Field(default_factory=dict)
            environment: dict = Field(default_factory=dict)

        mock = MockConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flatten_extended_config(mock)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# ===================================================================
# 14. Unconverged optimization tolerance (opt_converge contract)
# ===================================================================

class TestUnconvergedOptimization:
    """Verify an unconverged geometry optimization (SLAPAF stops with
    /rc=_RC_NOT_CONVERGED_, pymolcas exit code 96) is returned as a
    result status through the opt_converge contract instead of
    aborting the run."""

    @staticmethod
    def _load_config(yaml_name):
        import yaml
        from pathlib import Path

        yaml_path = Path(__file__).parent / f'{yaml_name}.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        return validate(normalize(build_config(raw)))

    # ---- log completion check ----

    def test_completion_ok(self):
        from ..service.parser.log_parser import _check_log_completion

        lines = [
            'Stop Module: gateway at Fri Aug 14 /rc=0 /',
            'Happy landing!',
        ]
        assert _check_log_completion(lines) == 'ok'

    def test_completion_not_converged(self):
        from ..service.parser.log_parser import _check_log_completion

        lines = [
            '--- Stop Module: slapaf at Sat Aug 15 23:54:48 2026 '
            '/rc=_RC_NOT_CONVERGED_ /',
            '.#######################.',
            '.# Convergence problem #.',
            '.#######################.',
        ]
        assert _check_log_completion(lines) == 'not_converged'

    def test_completion_mixed_failure_still_raises(self):
        from ..service.parser.log_parser import _check_log_completion

        lines = [
            'Stop Module: slapaf at Sat Aug 15 /rc=_RC_NOT_CONVERGED_ /',
            'Stop Module: rasscf at Sat Aug 15 /rc=_RC_INPUT_ERROR_ /',
        ]
        with pytest.raises(RuntimeError, match='did not complete'):
            _check_log_completion(lines)

    def test_completion_incomplete_still_raises(self):
        from ..service.parser.log_parser import _check_log_completion

        with pytest.raises(RuntimeError, match='Happy landing'):
            _check_log_completion(['some unrelated output'])

    # ---- opt_converge mapping in parse_output ----

    def test_parse_output_maps_unconverged(self, tmp_path):
        from ..service.parser import output_parser

        config = self._load_config('casscf_geometry_optimization')
        final = [[0.5 * i, 0.5 * i, 0.5 * i]
                 for i in range(config.natom)]
        TestFinCoord._write_xyz(tmp_path / f'{config.project}.Opt.xyz',
                                config.atoms, [final])
        with patch.object(
                output_parser, 'parse_log',
                return_value={'energy': [-75.0],
                              'log_not_converged': True}), \
             patch.object(output_parser, 'manage_orbital_files'):
            results = output_parser.parse_output(
                config, str(tmp_path), workflow=None)
        assert results['opt_converge'] is False
        assert 'log_not_converged' not in results
        assert np.allclose(results['fin_coord'], final)

    def test_parse_output_maps_converged(self, tmp_path):
        from ..service.parser import output_parser

        config = self._load_config('casscf_geometry_optimization')
        final = [[0.5 * i, 0.5 * i, 0.5 * i]
                 for i in range(config.natom)]
        TestFinCoord._write_xyz(tmp_path / f'{config.project}.Opt.xyz',
                                config.atoms, [final])
        with patch.object(
                output_parser, 'parse_log',
                return_value={'energy': [-75.0],
                              'log_not_converged': False}), \
             patch.object(output_parser, 'manage_orbital_files'):
            results = output_parser.parse_output(
                config, str(tmp_path), workflow=None)
        assert results['opt_converge'] is True

    def test_parse_output_drops_marker_for_energy(self, tmp_path):
        from ..service.parser import output_parser

        config = validate(normalize(build_config(MINIMAL_CONFIG)))
        with patch.object(
                output_parser, 'parse_log',
                return_value={'energy': [-75.0],
                              'log_not_converged': False}), \
             patch.object(output_parser, 'manage_orbital_files'):
            results = output_parser.parse_output(
                config, str(tmp_path), workflow=None)
        assert 'opt_converge' not in results
        assert 'log_not_converged' not in results

    # ---- backend tolerance path ----

    def test_run_not_converged_flag(self, tmp_path):
        from ..backend import _run_not_converged

        config = self._load_config('casscf_geometry_optimization')
        # Missing log: fail-fast
        assert _run_not_converged(config, str(tmp_path)) is False

        log = tmp_path / f'{config.project}.log'
        log.write_text('--- Stop Module: slapaf at Sat Aug 15 '
                       '/rc=_RC_NOT_CONVERGED_ /\n')
        assert _run_not_converged(config, str(tmp_path)) is True

        log.write_text('Stop Module: gateway /rc=0 /\nHappy landing!\n')
        assert _run_not_converged(config, str(tmp_path)) is False

    @patch('pyrai3md.skills.molcas_skill.backend.validate_environment')
    @patch('pyrai3md.skills.molcas_skill.backend.write_input_file')
    @patch('pyrai3md.skills.molcas_skill.backend.manage_orbitals')
    def test_calculate_returns_unconverged_status(
            self, mock_orbitals, mock_input, mock_env, tmp_path):
        import yaml
        from pathlib import Path
        from ..backend import MolcasBackend

        yaml_path = Path(__file__).parent / 'casscf_geometry_optimization.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw['molcas_path'] = '/tmp'

        calcdir = str(tmp_path / 'calc')
        os.makedirs(calcdir)
        config = validate(normalize(build_config(raw)))
        # Fake SLAPAF outputs: unconverged stop marker (no Happy
        # landing) and the final structure written by SLAPAF.
        final = [[0.9 * i, 0.8 * i, 0.7 * i]
                 for i in range(config.natom)]
        TestFinCoord._write_xyz(
            os.path.join(calcdir, f'{config.project}.Opt.xyz'),
            config.atoms, [final])
        with open(os.path.join(calcdir, f'{config.project}.log'),
                  'w') as f:
            f.write('--- Stop Module: slapaf at Sat Aug 15 23:54:48 '
                    '2026 /rc=_RC_NOT_CONVERGED_ /\n')

        backend = MolcasBackend(config=raw, calcdir=calcdir)
        with patch.object(backend, 'run', return_value=96):
            results = backend.calculate(
                config.atoms, config.coordinates)
        assert results['opt_converge'] is False
        assert np.allclose(results['fin_coord'], final)

    @patch('pyrai3md.skills.molcas_skill.backend.validate_environment')
    @patch('pyrai3md.skills.molcas_skill.backend.write_input_file')
    @patch('pyrai3md.skills.molcas_skill.backend.manage_orbitals')
    def test_calculate_crash_still_raises(
            self, mock_orbitals, mock_input, mock_env, tmp_path):
        import yaml
        from pathlib import Path
        from ..backend import MolcasBackend

        yaml_path = Path(__file__).parent / 'casscf_geometry_optimization.yaml'
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)['input']
        raw['molcas_path'] = '/tmp'

        calcdir = str(tmp_path / 'calc')
        os.makedirs(calcdir)
        config = validate(normalize(build_config(raw)))
        # Log without the unconverged marker: a genuine crash must
        # keep raising.
        with open(os.path.join(calcdir, f'{config.project}.log'),
                  'w') as f:
            f.write('Stop Module: rasscf at Sat Aug 15 '
                    '/rc=_RC_INPUT_ERROR_ /\n')

        backend = MolcasBackend(config=raw, calcdir=calcdir)
        with patch.object(backend, 'run', return_value=16):
            with pytest.raises(RuntimeError, match='return code 16'):
                backend.calculate(config.atoms, config.coordinates)


# ===================================================================
# Active orbital selection (ALTEr)
# ===================================================================

class TestActiveOrbitalSelection:
    """Verify active orbital selection and ALTEr pair derivation.

    Indices are 1-based at the boundary and 0-based internally.
    The CH2 minimal config has 8 electrons and nact_elec=2, so
    normalize derives inactive_orb=3; with ras2=2 the Ras2 window
    is 0-based {3, 4} (1-based [4, 5]).
    """

    def test_build_config_converts_active_to_zero_based(self):
        raw = dict(MINIMAL_CONFIG, active=[2, 5])
        config = build_config(raw)
        assert config.active == [1, 4]

    def test_derive_alter_pairs_full_overlap_locked(self):
        from ..schema.normalize import derive_alter_pairs
        # Selection identical to the window: nothing to swap
        assert derive_alter_pairs([3, 4], 3, 0, 2) == []

    def test_derive_alter_pairs_partial_overlap(self):
        from ..schema.normalize import derive_alter_pairs
        # Orbital 1 (0-based) swapped into window position 3;
        # orbital 4 stays locked. Output 1-based: (sym=1, 2, 4)
        assert derive_alter_pairs([1, 4], 3, 0, 2) == [(1, 2, 4)]

    def test_derive_alter_pairs_disjoint_pairs(self):
        from ..schema.normalize import derive_alter_pairs
        assert derive_alter_pairs([1, 2], 3, 0, 2) == [
            (1, 2, 4), (1, 3, 5)]

    def test_derive_alter_pairs_ras1_shifts_window(self):
        from ..schema.normalize import derive_alter_pairs
        # ras1_orb=1 shifts the window to 0-based {4, 5}
        assert derive_alter_pairs([4, 5], 3, 1, 2) == []
        assert derive_alter_pairs([1, 5], 3, 1, 2) == [(1, 2, 5)]

    def test_build_rasscf_emits_alter(self):
        from ..service.builder.module_builders import build_rasscf
        # 1-based selection [2, 5]: orbital 5 locked in the window,
        # orbital 2 swapped into position 4 -> pair '1 2 4'.
        raw = dict(MINIMAL_CONFIG, active=[2, 5])
        config = validate(normalize(build_config(raw)))
        result = build_rasscf(config)
        assert ' ALTEr' in result
        assert '\n 1\n 1 2 4' in result

    def test_build_rasscf_no_alter_without_active(self, skill_config):
        from ..service.builder.module_builders import build_rasscf
        result = build_rasscf(skill_config)
        assert 'ALTEr' not in result

    def test_build_rasscf_no_alter_when_overlap_complete(self):
        from ..service.builder.module_builders import build_rasscf
        # 1-based [4, 5] equals the default window: locked, no ALTEr
        raw = dict(MINIMAL_CONFIG, active=[4, 5])
        config = validate(normalize(build_config(raw)))
        result = build_rasscf(config)
        assert 'ALTEr' not in result


class TestCISpaceLegality:
    """A RASSCF job may not ask for more roots than its space holds CSFs.

    Molcas refuses this itself: a `&RASSCF Spin=3 CIRoot=2` over a CAS(2,2)
    stops with "You can't ask for more roots than there are configurations"
    (measured in the test environment for the shipped `casscf_prop` fixture in
    rounds tmp_20261010_133424 and tmp_20261010_190544, where the engine ran
    GATEWAY/SEWARD/a full singlet RASSCF before dying in the second block).
    The counting is asserted here so an over-declared fixture fails locally,
    in the time it takes to read it.
    """

    @staticmethod
    def csf_count(nact_elec, nact_orb, spin):
        """Spin-adapted CSF count of CAS(nact_elec, nact_orb) at given S.

        Covers the <=2 active electrons the shipped fixtures declare: with
        `d` doubly occupied active orbitals the remaining `o = nact_elec - 2d`
        electrons sit one per orbital, the orbitals being chosen in
        C(nact_orb, d) * C(nact_orb - d, o) ways, and o open shells couple to
        S=0 only when o is 0, to S=1/2 when o is 1, and to one S=0 plus one
        S=1 when o is 2.
        """
        from math import comb

        if nact_elec > 2:
            raise AssertionError(
                'the counter only covers the <=2-electron spaces the '
                'fixtures declare; extend it before using it elsewhere')
        total = 0
        for doubles in range(nact_elec // 2 + 1):
            open_shells = nact_elec - 2 * doubles
            if doubles + open_shells > nact_orb:
                continue
            choices = (comb(nact_orb, doubles)
                       * comb(nact_orb - doubles, open_shells))
            if open_shells == 0:
                spins = (0,)
            elif open_shells == 1:
                spins = (0.5,)
            else:
                spins = (0, 1)
            total += choices * (1 if spin in spins else 0)
        return total

    def _fixture_request(self):
        import yaml

        from ..service.workflow import compile_workflow

        path = os.path.join(os.path.dirname(__file__), 'casscf_prop.yaml')
        spec = yaml.safe_load(open(path).read())
        raw = dict(spec['input'])
        raw.update({'name': spec['name'],
                    'version': spec.get('version', 1.0)})
        config = validate(normalize(build_config(raw)))
        compile_workflow(config.calculation.profile, config=config)
        return (int(config.nact_elec), int(config.nact_orb or config.ras2_orb),
                (config.singlet_ci_roots or [1])[0],
                (config.triplet_ci_roots or [1])[0])

    def test_the_shipped_soc_fixture_fits_its_own_space(self):
        electrons, orbitals, singlet_roots, triplet_roots = \
            self._fixture_request()
        assert singlet_roots <= self.csf_count(electrons, orbitals, 0), \
            (electrons, orbitals, singlet_roots)
        assert triplet_roots <= self.csf_count(electrons, orbitals, 1), \
            (electrons, orbitals, triplet_roots)

    def test_the_counter_sees_the_combination_the_fixture_used_to_ask(self):
        # CAS(2,2) holds one triplet CSF, which is why two triplet roots were
        # an engine refusal; CAS(2,3) holds three, which is what the case
        # declares now.
        assert self.csf_count(2, 2, 1) == 1
        assert self.csf_count(2, 3, 1) == 3
        assert self.csf_count(2, 2, 0) == 3
        assert self.csf_count(2, 3, 0) == 6


class TestDualSpinBlocks:
    """Two RASSCF jobs per optimization iteration, read block by block.

    The MECP case runs a singlet job and a triplet job, so one iteration prints
    two 'Final state energy(ies)' blocks and two gradient blocks, and the two
    blocks hold DIFFERENT numbers of roots when the manifolds are uneven
    (`ci_roots` is the per-group list). Reading every block with the first
    group's count is the defect this class holds: on an uneven request it either
    stops short or runs into the next section's text, where the parse that
    followed swallowed the error and returned a state table with a surface
    missing.

    The block layout here is written to the branch the parser itself takes when
    the root-number banner is absent and the table starts five lines below the
    header (`log_parser.py:206-216`, energies as the second token). No real
    Molcas dual-job log exists on this host, so these tests prove the COUNTING
    and the iteration boundary against the parser's own contract; which branch
    the engine actually prints is reading Q3 of the test round.
    """

    NATOM = 6

    @staticmethod
    def _config(state_spin, target_states, pair=(0, 1)):
        raw = dict(MINIMAL_CONFIG)
        raw.update({
            'atoms': ['C', 'C', 'H', 'H', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.33],
                            [0.0, 0.93, -0.57], [0.0, -0.93, -0.57],
                            [0.0, 0.93, 1.90], [0.0, -0.93, 1.90]],
            'nstate': len(state_spin),
            'state_spin': list(state_spin),
            'target_states': list(target_states),
            'profile': 'casscf_geometry_optimization',
            'opt_state_i': pair[0],
            'opt_state_j': pair[1],
        })
        return validate(normalize(build_config(raw)))

    @classmethod
    def _energy_block(cls, energies):
        lines = [
            '                Final state energy(ies)',
            '',
            '',
            '',
            '      state      energy',
        ]
        for index, value in enumerate(energies, 1):
            lines.append('      %d   %18.10f' % (index, value))
        return lines

    @classmethod
    def _gradient_block(cls, first):
        lines = ['      Molecular gradients for state']
        lines += ['', '', '', '', '', '', '']
        for index in range(cls.NATOM):
            value = first + index
            lines.append('   %2d   %18.10f %18.10f %18.10f'
                         % (index + 1, value, value + 0.5, value + 1.0))
        return lines

    @staticmethod
    def _tail():
        return ['Stop Module: slapaf at Fri Aug 14 /rc=_RC_ALL_IS_WELL_ /',
                'Happy landing!']

    def _write(self, directory, lines):
        path = directory / 'molcas.log'
        path.write_text('\n'.join(lines) + '\n')
        return str(path)

    def test_each_spin_group_is_read_with_its_own_root_count(self, tmp_path):
        from ..service.parser.log_parser import parse_log

        # Two excited singlets and one triplet: the second block holds one
        # root, and reading it with the first block's count would run past it.
        # The pair is the singlet row and the triplet row: two jobs, hence two
        # blocks per iteration, only when the two surfaces differ in spin.
        # The pair is the singlet row and the triplet row: two jobs, hence two
        # blocks per iteration, only when the two surfaces differ in spin. Each
        # job prints one force block (its relaxed root), so the run asks for the
        # two surfaces of the crossing rather than all four states.
        config = self._config([0, 0, 0, 2], [0, 3], pair=(0, 3))
        assert list(config.ci_roots) == [3, 1]
        assert config.state_spin == [0, 0, 0, 2]
        lines = (self._energy_block([-78.10, -78.20, -78.30])
                 + self._energy_block([-79.90])
                 + self._gradient_block(0.01)
                 + self._gradient_block(0.11)
                 + self._tail())
        result = parse_log(self._write(tmp_path, lines), config)
        energies = list(np.asarray(result['energy'], dtype=np.float64))
        assert len(energies) == 4, energies
        assert energies[:3] == pytest.approx([-78.10, -78.20, -78.30])
        assert energies[3] == pytest.approx(-79.90)

    def test_the_last_optimization_iteration_is_the_one_kept(self, tmp_path):
        from ..service.parser.log_parser import parse_log

        config = self._config([0, 2], [0, 1])
        first = (self._energy_block([-78.10]) + self._gradient_block(0.01)
                 + self._energy_block([-79.90]) + self._gradient_block(0.11))
        second = (self._energy_block([-78.55]) + self._gradient_block(1.01)
                  + self._energy_block([-79.25]) + self._gradient_block(1.11))
        result = parse_log(self._write(tmp_path, first + second + self._tail()),
                           config)
        energies = list(np.asarray(result['energy'], dtype=np.float64))
        assert energies == pytest.approx([-78.55, -79.25])
        gradient = np.asarray(result['gradient'], dtype=np.float64)
        assert gradient.shape == (2, self.NATOM, 3)
        # The first atom of each row carries the block's stamp: iteration two's
        # stamps, not iteration one's, which is what the search steps on.
        assert gradient[0][0][0] == pytest.approx(1.01)
        assert gradient[1][0][0] == pytest.approx(1.11)

    def test_a_requested_state_without_a_gradient_block_is_refused(self,
                                                                   tmp_path):
        from ..service.parser.log_parser import parse_log

        config = self._config([0, 2], [0, 1])
        lines = (self._energy_block([-78.10]) + self._energy_block([-79.90])
                 + self._gradient_block(0.01) + self._tail())
        with pytest.raises(RuntimeError, match='gradient block'):
            parse_log(self._write(tmp_path, lines), config)

    def test_extra_gradient_blocks_are_reported_rather_than_dropped_silently(
            self, tmp_path):
        from ..service.parser.log_parser import parse_log

        config = self._config([0, 2], [0])
        lines = (self._energy_block([-78.10]) + self._energy_block([-79.90])
                 + self._gradient_block(0.01) + self._gradient_block(0.11)
                 + self._tail())
        with pytest.warns(UserWarning, match='not packed'):
            result = parse_log(self._write(tmp_path, lines), config)
        gradient = np.asarray(result['gradient'], dtype=np.float64)
        assert gradient.shape == (2, self.NATOM, 3)
        assert gradient[0][0][0] == pytest.approx(0.01)

    def test_fewer_state_energies_than_declared_is_refused(self, tmp_path):
        from ..service.parser.log_parser import parse_log

        config = self._config([0, 0, 0, 2], [0, 1, 2, 3], pair=(0, 3))
        # The triplet job printed nothing: three of four surfaces are here.
        lines = self._energy_block([-78.10, -78.20, -78.30]) + self._tail()
        with pytest.raises(RuntimeError, match='state energy'):
            parse_log(self._write(tmp_path, lines), config)


class TestDeckShapeHasOneSource:
    """#448: who says how many RASSCF passes a deck holds, and it is one voice.

    Two defects met at that seam. The builder wrote a single `&ESPF` for a deck
    that runs one RASSCF job per spin manifold, so the second manifold embedded
    the density file the first pass left behind; and the reader re-derived the
    blocks-per-round from the profile *name*, so a grouped deck labelled its
    second pass with the first pass's spin. Both sides now call
    `rasscf_pass_count`, and these tests bind that function to the deck the
    builder actually wrote rather than to a constant.

    The block layout below is written to the branch the parser takes when the
    root-number banner is absent (`log_parser.py`, energies as the second
    token), as in `TestDualSpinBlocks`: what a real two-job deck prints is a
    test reading.
    """

    NATOM = 6

    @staticmethod
    def _config(state_spin, **over):
        raw = dict(MINIMAL_CONFIG)
        raw.update({
            'atoms': ['C', 'C', 'H', 'H', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.33],
                            [0.0, 0.93, -0.57], [0.0, -0.93, -0.57],
                            [0.0, 0.93, 1.90], [0.0, -0.93, 1.90]],
            'nstate': len(state_spin),
            'state_spin': list(state_spin),
            'profile': 'casscf_properties',
            'method': 'casscf',
            'jobtype': 'energy',
            'nact_elec': 2,
            'nact_orb': 3,
            # A named embedding plus a point-charge set, so the field really
            # resolves instead of coming back empty and hiding the question.
            'embedding': 'charge',
            'point_charges': [[0.5, 4.0, 0.0, 0.0], [-0.5, 6.0, 0.0, 0.0]],
        })
        point_charges = raw.pop('point_charges')
        embedding = over.pop('embedding', None)
        raw.update(over)
        if embedding == 'none':
            # An iterative profile refuses an embedding, and a gas-phase deck
            # keeps the question to one pass per round.
            raw['embedding'] = 'none'
        else:
            raw['point_charges'] = point_charges
        return validate(normalize(build_config(raw)))

    def _deck(self, config):
        import tempfile

        from ..service.builder.input_builder import build_input
        from ..service.workflow.compiler import compile_workflow

        workflow = compile_workflow(config.profile, config=config)
        with tempfile.TemporaryDirectory() as directory:
            return workflow, build_input(config, workflow, calcdir=directory)

    @classmethod
    def _energy_block(cls, energies):
        lines = ['                Final state energy(ies)',
                 '', '', '', '      state      energy']
        for index, value in enumerate(energies, 1):
            lines.append('      %d   %18.10f' % (index, value))
        return lines

    @staticmethod
    def _write(directory, lines):
        path = directory / 'molcas.log'
        path.write_text('\n'.join(lines) + '\n')
        return str(path)

    def test_an_espf_block_precedes_every_rasscf_pass(self):
        for label, spin, want in (('one manifold', [0, 0, 0], 1),
                                  ('two manifolds', [0, 0, 2], 2)):
            _, deck = self._deck(self._config(spin))
            modules = [line.strip() for line in deck.splitlines()
                       if line.strip().startswith('&')]
            assert modules.count('&RASSCF') == want, label
            assert deck.count('&ESPF') == want, label
            # Each field block belongs to the job that follows it, and the
            # ESP communication files are cleared before a second one is
            # written - otherwise the second pass reads the first density.
            assert all(previous == '&ESPF'
                       for name, previous in zip(modules[1:], modules)
                       if name == '&RASSCF'), label
            assert deck.count('RM -FORCE ESPF.DATA') == want - 1, label

    def test_the_grouping_follows_the_declaration_not_the_tokens(self):
        config = self._config([0, 0, 2], embedding='none',
                              jobtype='energy gradient nac',
                              coupling=[[0, 1]], target_states=[0, 1, 2])
        _, deck = self._deck(config)
        # The declared triplet state is a surface the run asked for; a pair
        # that lives only in the singlet manifold is no reason to drop it.
        assert deck.count('&RASSCF') == 2
        assert 'Spin=1' in deck and 'Spin=3' in deck

    def test_the_pass_count_describes_the_built_deck(self):
        from ..service.builder.input_builder import rasscf_pass_count

        cases = (
            ('energy, one manifold', self._config([0, 0, 0]), 1),
            ('energy, two manifolds', self._config([0, 0, 2]), 2),
            ('mecp optimization',
             self._config([0, 2], embedding='none',
                          profile='casscf_geometry_optimization',
                          jobtype='optimize', opt_state_i=0, opt_state_j=1),
             2),
            ('same-spin optimization',
             self._config([0, 0], embedding='none',
                          profile='casscf_geometry_optimization',
                          jobtype='optimize', opt_state_i=0, opt_state_j=1),
             1),
        )
        for label, config, want in cases:
            workflow, deck = self._deck(config)
            assert rasscf_pass_count(config, workflow) == want, label
            rounds = 2 if workflow.two_pass else 1
            assert deck.count('&RASSCF') == want * rounds, label

    def test_a_mixed_ladder_groups_on_the_named_per_step_profiles(self):
        """The per-step asks resolve to named profiles, and they group too.

        `energy gradient` resolves to `casscf_gradient` and
        `energy gradient nac` to `casscf_nac`; before #449 both took the
        legacy branch, which wrote one singlet job and then asked it for
        roots and pairs that live in the manifold it never built.
        """
        from ..query import resolve_profile

        cases = (
            ('energy gradient', None, []),
            ('energy gradient nac', [[0, 1]], ['NAC=1 2']),
        )
        for jobtype, coupling, want_nac in cases:
            extra = {'jobtype': jobtype, 'target_states': [0, 1, 2],
                     'profile': resolve_profile('casscf', jobtype)}
            if coupling:
                extra['coupling'] = coupling
            config = self._config([0, 0, 2], embedding='none', **extra)
            _, deck = self._deck(config)
            assert deck.count('&RASSCF') == 2, jobtype
            assert 'Spin=1' in deck and 'Spin=3' in deck, jobtype
            # Root numbers are local to the job that answers them
            assert [line.strip() for line in deck.splitlines()
                    if line.strip().startswith('ROOT=')] == [
                        'ROOT=1', 'ROOT=2', 'ROOT=1'], jobtype
            assert [line.strip() for line in deck.splitlines()
                    if line.strip().startswith('NAC=')] == want_nac, jobtype

    def test_a_property_deck_keeps_every_block_it_printed(self, tmp_path):
        from ..service.parser.log_parser import parse_log

        config = self._config([0, 0, 2])
        lines = (self._energy_block([-78.10, -78.20])
                 + self._energy_block([-79.90]) + ['Happy landing!'])
        result = parse_log(self._write(tmp_path, lines), config)
        energies = list(np.asarray(result['energy'], dtype=np.float64))
        # A property deck's blocks are read in declaration order and none is
        # dropped. This pins the read, not the reset gate: with two blocks and
        # a two-block round the reset could not have fired anyway, so the
        # gate's witness is the looping deck in
        # `TestDualSpinBlocks::test_the_last_optimization_iteration_is_the_one_kept`
        # (verified by removing the reset, which reddens that test and not this
        # one).
        assert energies == pytest.approx([-78.10, -78.20, -79.90])
