"""
======================================================================
Test suite for the planning layer (pyrai3md/planning/).

Tests cover:
- Module imports (resolve_capabilities, CAPABILITY_NAMES)
- Token-to-capability mapping (basic and advanced jobtypes)
- Dependency resolution
- Dynamics-derived capabilities (dosoc, dodipole, sfhp+nactype)
- Multi-backend constraints (hessian forbidden, scoping)
- Per-skill capability distribution

All capabilities are plain strings -- no Property objects.

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestPlanningImports(unittest.TestCase):
    """Test that planning modules can be imported with the new API."""

    def test_import_resolve_capabilities(self):
        from pyrai3md.planning import resolve_capabilities
        self.assertTrue(callable(resolve_capabilities))

    def test_import_capability_names(self):
        from pyrai3md.planning import CAPABILITY_NAMES
        self.assertIsInstance(CAPABILITY_NAMES, frozenset)
        self.assertIn('energy', CAPABILITY_NAMES)
        self.assertIn('gradient', CAPABILITY_NAMES)
        self.assertIn('nac', CAPABILITY_NAMES)

    def test_import_from_resolver(self):
        from pyrai3md.planning.resolver import (
            resolve_capabilities, CAPABILITY_NAMES,
            TOKEN_TO_CAPABILITY, _DEPENDENCIES, _ADVANCED_DEFAULTS,
        )
        self.assertTrue(callable(resolve_capabilities))
        self.assertIsInstance(CAPABILITY_NAMES, frozenset)


class TestBasicResolution(unittest.TestCase):
    """Test basic token-to-capability resolution."""

    def test_energy_only(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'])
        names = result['property_names']
        self.assertIn('energy', names)

    def test_energy_gradient(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy', 'gradient'])
        names = result['property_names']
        self.assertIn('energy', names)
        self.assertIn('gradient', names)

    def test_nac_token(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['nac'])
        names = result['property_names']
        self.assertIn('nac', names)
        self.assertIn('energy', names)  # dependency

    def test_soc_token(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['soc'])
        names = result['property_names']
        self.assertIn('soc', names)
        self.assertIn('energy', names)  # dependency

    def test_frequency_maps_to_hessian(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['frequency'])
        names = result['property_names']
        self.assertIn('hessian', names)
        self.assertIn('energy', names)
        self.assertIn('gradient', names)

    def test_absorption_maps_to_transition_dipole(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['absorption'])
        names = result['property_names']
        self.assertIn('transition_dipole', names)
        self.assertIn('energy', names)

    def test_is_basic_flag(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'])
        self.assertTrue(result['is_basic'])
        self.assertFalse(result['is_advanced'])


class TestAdvancedResolution(unittest.TestCase):
    """Test advanced jobtype expansion."""

    def test_optimize_expands(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['optimize'])
        names = result['property_names']
        self.assertIn('energy', names)
        self.assertIn('gradient', names)

    def test_md_expands(self):
        from pyrai3md.planning import resolve_capabilities
        # A dynamics run defaults to the E+G pair; ktdc builds the coupling
        # from the energy curvature and needs no NAC vector.
        result = resolve_capabilities(['md'])
        names = result['property_names']
        self.assertIn('energy', names)
        self.assertIn('gradient', names)
        self.assertNotIn('nac', names)

    def test_md_nac_method_adds_nac(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['md'], planning_context={'sfhp': 'fssh', 'nactype': 'nac'})
        self.assertIn('nac', result['property_names'])

    def test_is_advanced_flag(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['md'])
        self.assertTrue(result['is_advanced'])


class TestDynamicsDerivation(unittest.TestCase):
    """Test dynamics-derived capability additions."""

    def test_dosoc_adds_soc(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'], planning_context={'dosoc': 1})
        names = result['property_names']
        self.assertIn('soc', names)
        self.assertEqual(result['dosoc'], 1)

    def test_dosoc_string_true(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'], planning_context={'dosoc': 'true'})
        names = result['property_names']
        self.assertIn('soc', names)
        self.assertEqual(result['dosoc'], 1)

    def test_dodipole_adds_dipole(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'], planning_context={'dodipole': 1})
        names = result['property_names']
        self.assertIn('dipole', names)
        self.assertEqual(result['dodipole'], 1)

    def test_dodipole_string_yes(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'], planning_context={'dodipole': 'yes'})
        names = result['property_names']
        self.assertIn('dipole', names)
        self.assertEqual(result['dodipole'], 1)

    def test_fssh_dcm_adds_overlap(self):
        from pyrai3md.planning import resolve_capabilities
        dynamics = {'sfhp': 'fssh', 'nactype': 'dcm'}
        result = resolve_capabilities(['energy'], planning_context=dynamics)
        names = result['property_names']
        self.assertIn('overlap', names)

    def test_fssh_nac_adds_nac(self):
        from pyrai3md.planning import resolve_capabilities
        dynamics = {'sfhp': 'fssh', 'nactype': 'nac'}
        result = resolve_capabilities(['energy'], planning_context=dynamics)
        names = result['property_names']
        self.assertIn('nac', names)

    def test_radiative_adds_transition_dipole(self):
        from pyrai3md.planning import resolve_capabilities
        dynamics = {'radiative': 'true'}
        result = resolve_capabilities(['energy'], planning_context=dynamics)
        names = result['property_names']
        self.assertIn('transition_dipole', names)

    def test_no_dynamics_defaults(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'])
        self.assertEqual(result['dosoc'], 0)
        self.assertEqual(result['dodipole'], 0)

    def test_dynamics_schema_registers_property_triggers(self):
        # dodipole/radiative must survive the framework schema
        # pipeline so _extract_planning_context can consume them.
        from pyrai3md.schema.config import Dynamics
        dynamics = Dynamics(dodipole=1, radiative='true')
        self.assertEqual(dynamics.dodipole, 1)
        self.assertEqual(dynamics.radiative, 'true')
        self.assertEqual(Dynamics().dodipole, 0)
        self.assertEqual(Dynamics().radiative, '0')

    def test_md_types_registers_property_triggers(self):
        from pyrai3md.schema.adapter import MD_TYPES
        self.assertEqual(MD_TYPES['dodipole'], 'i')
        self.assertEqual(MD_TYPES['radiative'], 's')


class TestStressDerivation(unittest.TestCase):
    """Test whole-system stress derivation (NPT / optimize_cell).

    The stress tensor is never a user-facing jobtype token: it is
    derived from the ensemble (md + npt) and the cell-optimization
    switch (optimize + optimize_cell), then carried to the skill as a
    boolean config flag (see planning/assembly.py) rather than through the
    skill jobtype spelling.
    """

    def test_md_npt_derives_stress(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['md'], planning_context={'ensemble': 'npt'})
        self.assertIn('stress', result['property_names'])

    def test_md_nve_no_stress(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['md'], planning_context={'ensemble': 'nve'})
        self.assertNotIn('stress', result['property_names'])

    def test_md_nvt_no_stress(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['md'], planning_context={'ensemble': 'nvt'})
        self.assertNotIn('stress', result['property_names'])

    def test_md_default_ensemble_no_stress(self):
        # A missing ensemble key defaults to 'nve' -> no stress.
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['md'], planning_context={})
        self.assertNotIn('stress', result['property_names'])

    def test_ensemble_case_insensitive(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['md'], planning_context={'ensemble': 'NPT'})
        self.assertIn('stress', result['property_names'])

    def test_optimize_cell_derives_stress(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['optimize'], planning_context={'optimize_cell': True})
        self.assertIn('stress', result['property_names'])

    def test_optimize_no_cell_no_stress(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['optimize'], planning_context={'optimize_cell': False})
        self.assertNotIn('stress', result['property_names'])

    def test_optimize_default_no_stress(self):
        # A missing optimize_cell key defaults to False -> no stress.
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['optimize'], planning_context={})
        self.assertNotIn('stress', result['property_names'])

    def test_stress_pulls_energy_dependency(self):
        # stress depends on energy, so the NPT route keeps energy.
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['md'], planning_context={'ensemble': 'npt'})
        names = result['property_names']
        self.assertIn('stress', names)
        self.assertIn('energy', names)

    def test_stress_is_basic_capability(self):
        # stress must survive multi-backend scoping so the
        # whole-system baseline task receives it.
        from pyrai3md.planning.resolver import _BASIC_CAPABILITIES
        self.assertIn('stress', _BASIC_CAPABILITIES)

    def test_stress_not_in_jobtype_spelling(self):
        # stress is intentionally absent from the skill jobtype
        # spelling maps, so spell_skill_jobtype skips it and the
        # skill's resolve_profile is never altered.
        from pyrai3md.planning.assembly import (
            CAPABILITY_TO_TOKEN, JOBTYPE_TOKEN_ORDER, spell_skill_jobtype)
        self.assertNotIn('stress', CAPABILITY_TO_TOKEN)
        self.assertNotIn('stress', JOBTYPE_TOKEN_ORDER)
        spelled = spell_skill_jobtype(
            ['energy', 'gradient', 'stress'], 'md')
        self.assertEqual(spelled, 'energy gradient')

    def test_optimize_schema_registers_optimize_cell(self):
        # optimize_cell must survive the framework schema pipeline so
        # _extract_planning_context can consume it.
        from pyrai3md.schema.config import Optimize
        self.assertFalse(Optimize().optimize_cell)
        self.assertTrue(Optimize(optimize_cell=True).optimize_cell)


class TestMultiBackendConstraints(unittest.TestCase):
    """Test multi-backend capability constraints."""

    def test_hessian_forbidden_multi_theory(self):
        from pyrai3md.planning import resolve_capabilities
        skills = {
            'theory': {
                'skill_a': {'role': 'qm'},
                'skill_b': {'role': 'qm2'},
            },
            'reference': {},
        }
        with self.assertRaises(ValueError):
            resolve_capabilities(['frequency'], skills=skills)

    def test_hessian_forbidden_multi_reference(self):
        from pyrai3md.planning import resolve_capabilities
        skills = {
            'theory': {'skill_a': {'role': 'qm'}},
            'reference': {
                'skill_b': {'role': 'qm'},
                'skill_c': {'role': 'qm2'},
            },
        }
        with self.assertRaises(ValueError):
            resolve_capabilities(['frequency'], skills=skills)

    def test_hessian_ok_single_backend(self):
        from pyrai3md.planning import resolve_capabilities
        skills = {
            'theory': {'skill_a': {'role': 'qm'}},
            'reference': {},
        }
        result = resolve_capabilities(['frequency'], skills=skills)
        self.assertIn('hessian', result['property_names'])


class TestHessianRoute(unittest.TestCase):
    """Multiscale Hessians: only a backend-owned matrix is refused."""

    def _skills(self):
        return {
            'theory': {
                'skill_a': {'role': 'qm'},
                'skill_b': {'role': 'qm2'},
            },
            'reference': {},
        }

    def test_explicit_main_without_mode_is_numerical(self):
        from pyrai3md.schema.config import resolve_hessian_route
        self.assertEqual(
            resolve_hessian_route(['frequency'], {'hess_driver': 'main'}),
            ('numerical', 'main'))

    def test_token_defaults(self):
        from pyrai3md.schema.config import resolve_hessian_route
        self.assertEqual(resolve_hessian_route(['num_hess'], {}),
                         ('numerical', 'main'))
        self.assertEqual(resolve_hessian_route(['frequency'], {}),
                         ('auto', 'skill'))
        self.assertEqual(
            resolve_hessian_route(['frequency'],
                                  {'hess_driver': 'skill'}),
            ('auto', 'skill'))

    def test_frequency_main_is_allowed_multi_backend(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['frequency'], skills=self._skills(),
            planning_context={'hessian_params': {'hess_driver': 'main'}})
        self.assertIn('hessian', result['property_names'])

    def test_frequency_skill_is_refused_multi_backend(self):
        from pyrai3md.planning import resolve_capabilities
        with self.assertRaises(ValueError) as ctx:
            resolve_capabilities(
                ['frequency'], skills=self._skills(),
                planning_context={
                    'hessian_params': {'hess_driver': 'skill'}})
        self.assertIn('num_hess/num_freq', str(ctx.exception))

    def test_num_hess_is_allowed_multi_backend(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(
            ['num_hess'], skills=self._skills(),
            planning_context={'hessian_params': {}})
        self.assertNotIn('hessian', result['property_names'])


class TestCapabilityDistribution(unittest.TestCase):
    """Test per-skill capability distribution."""

    def test_single_backend_gets_all(self):
        from pyrai3md.planning import resolve_capabilities
        skills = {
            'theory': {'skill_a': {'role': 'qm'}},
            'reference': {},
        }
        result = resolve_capabilities(
            ['energy', 'gradient', 'nac'], skills=skills
        )
        bc = result['backend_capabilities']
        self.assertIn('skill_a', bc)
        self.assertIn('nac', bc['skill_a'])
        self.assertIn('energy', bc['skill_a'])

    def test_multi_backend_scoping(self):
        from pyrai3md.planning import resolve_capabilities
        skills = {
            'theory': {
                'skill_a': {'role': 'qm'},
                'skill_b': {'role': 'qm2'},
            },
            'reference': {},
        }
        result = resolve_capabilities(
            ['energy', 'gradient', 'nac', 'soc'], skills=skills
        )
        bc = result['backend_capabilities']
        # Primary gets full capabilities
        self.assertIn('nac', bc['skill_a'])
        self.assertIn('soc', bc['skill_a'])
        # Non-primary restricted to basic
        self.assertIn('energy', bc['skill_b'])
        self.assertIn('gradient', bc['skill_b'])
        self.assertNotIn('nac', bc['skill_b'])
        self.assertNotIn('soc', bc['skill_b'])


class TestReturnStructure(unittest.TestCase):
    """Test the return dict structure."""

    def test_has_all_keys(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'])
        self.assertIn('property_names', result)
        self.assertIn('is_basic', result)
        self.assertIn('is_advanced', result)
        self.assertIn('backend_capabilities', result)
        self.assertIn('dosoc', result)
        self.assertIn('dodipole', result)

    def test_property_names_sorted(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy', 'gradient', 'nac'])
        names = result['property_names']
        self.assertEqual(names, sorted(names))

    def test_backend_capabilities_is_dict(self):
        from pyrai3md.planning import resolve_capabilities
        result = resolve_capabilities(['energy'])
        self.assertIsInstance(result['backend_capabilities'], dict)


if __name__ == '__main__':
    unittest.main()
