"""
======================================================================
Test suite for the skill-side flatten_extended_config protection
(pyrai3md/skills/*/service/utils.py).

Tests cover:
- Cross-section collision protection: framework-only sections
  (control, adaptive) are skipped and framework-only dynamics
  keywords (initial-condition method, velocity scale, tri-state
  dispatch flags, ...) never reach the skill flat config
- Winner semantics: system.velocities, calculation.method,
  calculation.verbose and environment.scratch_path stay in charge
- Static scan: new same-name collisions between the framework
  Dynamics section and a skill vocabulary must be handled
  explicitly (whitelisted by the skill Dynamics schema or listed
  in the documented shadow set), otherwise these tests fail

Aug 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.schema.config import Dynamics as FrameworkDynamics

from pyrai3md.skills.bagel_skill.schema.config import (
    Config as BagelConfig, Dynamics as BagelDynamics,
)
from pyrai3md.skills.bagel_skill.service.utils import (
    FLATTEN_ORDER as BAGEL_FLATTEN_ORDER,
    flatten_extended_config as bagel_flatten,
)
from pyrai3md.skills.molcas_skill.schema.config import (
    Config as MolcasConfig, Dynamics as MolcasDynamics,
)
from pyrai3md.skills.molcas_skill.service.utils import (
    FLATTEN_ORDER as MOLCAS_FLATTEN_ORDER,
    flatten_extended_config as molcas_flatten,
)
from pyrai3md.skills.pyscf_skill.schema.config import (
    Config as PyscfConfig, Dynamics as PyscfDynamics,
)
from pyrai3md.skills.pyscf_skill.service.utils import (
    FLATTEN_ORDER as PYSCF_FLATTEN_ORDER,
    flatten_extended_config as pyscf_flatten,
)


class _StubExtendedConfig:
    """Minimal stand-in for an extended PyRAIConfig instance.

    flatten_extended_config only consumes model_dump(), so a nested
    dictionary is enough to exercise the section filtering.
    """

    def __init__(self, nested):
        self._nested = nested

    def model_dump(self):
        return self._nested


# Nested sections carrying every audited cross-section collision.
# Loser values are chosen so a leak would visibly corrupt the flat
# config (wrong method, wrong verbose, wrong scratch path, ...).
COLLISION_NESTED = {
    'control': {
        'verbose': 9,
        'jobtype': 'energy',
        'gl_seed': 42,
    },
    'adaptive': {
        'multitask': True,
        'remotetask': True,
        'load': 7,
    },
    'dynamics': {
        'method': 'wigner',
        'scale': 0.5,
        'multitask': True,
        'remotetask': True,
        'velocities': [[0.1, 0.0, 0.0]],
        'root': 3,
        'sfhp': 'fssh',
        'temperature': 350.0,
        'state_population': [0.2, 0.8],
    },
    'system': {
        'atoms': ['O', 'H', 'H'],
        'velocities': [[0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.1, 0.0, 0.0]],
        'scale': 2.0,
        'multiplicity': 1,
    },
    'properties': {
        'energy': [-76.0, -75.9],
    },
    'calculation': {
        'method': 'CASSCF',
        'verbose': 1,
        'scratch_path': '/tmp/calc',
        'target_states': [0],
    },
    'environment': {
        'scratch_path': '/tmp/env',
    },
}

# Framework-only keys that must never appear in the skill flat
# config (control/adaptive sections skipped, dynamics whitelist).
# 'root' is protected for skills that do not declare it; molcas_skill
# declares root (initial MD state -> RASSCF MDRLXR) and passes it.
PROTECTED_KEYS = ('jobtype', 'gl_seed', 'load', 'root', 'sfhp',
                  'multitask', 'remotetask')


class TestFlattenProtection(unittest.TestCase):
    """Behavior tests: audited collisions resolve to the right winner."""

    def _check(self, flatten, allow_root=False):
        flat = flatten(_StubExtendedConfig(COLLISION_NESTED))

        # Framework-only sections and dynamics keywords stay out
        for key in PROTECTED_KEYS:
            if key == 'root' and allow_root:
                continue
            self.assertNotIn(key, flat)
        if allow_root:
            self.assertEqual(flat['root'], 3)
        self.assertNotEqual(flat.get('method'), 'wigner')

        # Skill-declared dynamics keys pass the whitelist
        self.assertEqual(flat['temperature'], 350.0)
        self.assertEqual(flat['state_population'], [0.2, 0.8])

        # Winner semantics for the audited collisions
        self.assertEqual(flat['method'], 'CASSCF')
        self.assertEqual(flat['verbose'], 1)
        self.assertEqual(flat['velocities'],
                         COLLISION_NESTED['system']['velocities'])
        self.assertEqual(flat['scale'], 2.0)
        self.assertEqual(flat['scratch_path'], '/tmp/env')

        # Unrelated sections still reach the flat config
        self.assertEqual(flat['atoms'], ['O', 'H', 'H'])
        self.assertEqual(flat['energy'], [-76.0, -75.9])
        self.assertEqual(flat['target_states'], [0])

    def test_molcas_flatten_protection(self):
        self._check(molcas_flatten, allow_root=True)

    def test_bagel_flatten_protection(self):
        self._check(bagel_flatten)

    def test_pyscf_flatten_protection(self):
        self._check(pyscf_flatten)

    def test_missing_sections_are_harmless(self):
        flat = molcas_flatten(_StubExtendedConfig(
            {'calculation': {'method': 'CASSCF'}}))
        self.assertEqual(flat, {'method': 'CASSCF'})


class TestFlattenOrderLock(unittest.TestCase):
    """Static locks on the flatten order and collision handling."""

    def test_framework_only_sections_are_not_flattened(self):
        for order in (MOLCAS_FLATTEN_ORDER, BAGEL_FLATTEN_ORDER,
                      PYSCF_FLATTEN_ORDER):
            self.assertNotIn('control', order)
            self.assertNotIn('adaptive', order)
            # environment stays last so its scratch_path wins
            self.assertEqual(order[-1], 'environment')

    def test_dynamics_collisions_are_explicitly_handled(self):
        """Every name collision between the framework Dynamics
        section and a skill vocabulary must be handled explicitly:
        either whitelisted by the skill's own Dynamics schema (the
        skill owns the keyword) or listed in the documented shadow
        set (framework-only semantics, silently dropped).  Adding a
        colliding framework field anywhere else fails this test and
        forces a conscious decision.
        """
        # Framework dynamics keywords with different semantics that
        # the whitelist intentionally shadows for every skill.
        shadowed = {'method', 'scale'}

        framework_fields = set(FrameworkDynamics.model_fields)
        skills = (
            ('molcas', MolcasConfig, MolcasDynamics),
            ('bagel', BagelConfig, BagelDynamics),
            ('pyscf', PyscfConfig, PyscfDynamics),
        )
        for name, config_cls, dynamics_cls in skills:
            collisions = framework_fields & set(config_cls.model_fields)
            handled = set(dynamics_cls.model_fields) | shadowed
            unhandled = collisions - handled
            self.assertFalse(
                unhandled,
                f"{name}: framework Dynamics fields {sorted(unhandled)} "
                f"collide with the skill vocabulary outside the "
                f"Dynamics whitelist; add them to the skill Dynamics "
                f"schema or to the documented shadow set",
            )


class _StubMaestro:
    """Duck-typed Maestro exposing only the merge-time lookups."""

    def __init__(self, runtime=None, skill_info=None):
        self._runtime = runtime or {}
        self._skill_info = skill_info or {}

    def get_config(self, skill_name):
        return self._runtime.get(skill_name)

    def get_skill_info(self, skill_name):
        return self._skill_info.get(skill_name)


class TestMdStepLimitForwarding(unittest.TestCase):
    """maxit_md/md_maxiter written in the framework input must reach
    the skill config: the framework Config owns no submodel for them,
    so _merge_one_skill_config forwards them from the raw sections.
    """

    def _merge(self, raw_config, runtime=None):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        maestro = _StubMaestro(
            runtime={'molcas_skill': runtime or {'maxit_md': 1}},
            skill_info={'molcas_skill': {'name': 'molcas'}})
        return _merge_one_skill_config(
            maestro, 'molcas_skill', config=PyRAIConfig(),
            raw_config=raw_config)

    def test_md_section_maxit_md_forwarded(self):
        # The reported bug: yaml 'md: maxit_md: 10' yielded
        # FOREACH (1 .. 1) and MOLCAS_MAXITER=10 in the molcas input.
        merged = self._merge({'md': {'maxit_md': 10}})
        self.assertEqual(merged['maxit_md'], 10)

    def test_md_section_md_maxiter_forwarded(self):
        merged = self._merge({'md': {'maxit_md': 10,
                                     'md_maxiter': 200}})
        self.assertEqual(merged['md_maxiter'], 200)

    def test_skill_section_overrides_md_section(self):
        merged = self._merge({'md': {'maxit_md': 10},
                              'molcas': {'maxit_md': 5}})
        self.assertEqual(merged['maxit_md'], 5)

    def test_unset_keeps_runtime_default(self):
        merged = self._merge({'md': {'step': 10}})
        self.assertEqual(merged['maxit_md'], 1)

    def test_barostat_mode_reaches_the_lammps_skill(self):
        # The word is the framework's, and the skill has to see it: until
        # the keyword was registered in the skill's own DEFAULT table a
        # declared cell coupling was dropped on the way through the merge
        # and LAMMPS ran its own default instead.
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        for mode in ('isotropic', 'anisotropic'):
            config = PyRAIConfig()
            config.control.jobtype = 'md'
            config.control.theory = ['lammps']
            config.dynamics.ensemble = 'npt'
            config.dynamics.barostat_mode = mode
            maestro = _StubMaestro(
                runtime={'lammps_skill': {}},
                skill_info={'lammps_skill': {'name': 'lammps'}})
            merged = _merge_one_skill_config(
                maestro, 'lammps_skill', config=config,
                raw_config={'md': {'barostat_mode': mode}})
            self.assertEqual(merged['barostat_mode'], mode)

    @staticmethod
    def _lammps_merge(config):
        from pyrai3md.schema.adapter import _merge_one_skill_config

        maestro = _StubMaestro(
            runtime={'lammps_skill': {}},
            skill_info={'lammps_skill': {'name': 'lammps'}})
        return _merge_one_skill_config(maestro, 'lammps_skill',
                                       config=config, raw_config={})

    def test_seeded_image_reaches_the_lammps_skill(self):
        # The crossing counts are seeded framework-side, from the wrapped
        # geometry the input carries, and an engine learns them only
        # through the skill config. Dropping the field here would put the
        # all-zero flags back in front of LAMMPS, which is exactly the
        # state its own consistency check calls inconsistent.
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = 'md'
        config.control.theory = ['lammps']
        config.system.pbc = [1, 1, 1]
        seeded = [[0, 0, 0], [-1, 0, 0], [0, 0, 1]]
        config.system.image = seeded
        merged = self._lammps_merge(config)
        self.assertEqual(merged['image'], seeded)

    def test_an_unseeded_system_forwards_no_image_key(self):
        # The forwarding rule is "differ from the default", so a run that
        # crossed nothing adds no key at all rather than restating an
        # empty list through every skill section.
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = 'energy'
        config.control.theory = ['lammps']
        self.assertNotIn('image', self._lammps_merge(config))


class TestSkillFlatDictUnification(unittest.TestCase):
    """The framework hands every skill backend a key-by-key
    identically typed flat dict: unify_skill_flat_dict converts the
    classic-text spellings echoed into skill sections (&MOLCAS ci 3 /
    coupling '1 2, 2 3' from the certification converter, or the agent
    config skill section) into the canonical typed forms, at the
    single merge choke point every framework input mode passes
    through. Unification is not validation: unparsable values stay raw
    so the skill schema reports them (the agent feedback channel)."""

    def _merge(self, section, runtime=None):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        maestro = _StubMaestro(
            runtime={'molcas_skill': runtime or {}},
            skill_info={'molcas_skill': {'name': 'molcas'}})
        return _merge_one_skill_config(
            maestro, 'molcas_skill', config=PyRAIConfig(),
            raw_config={'molcas': dict(section)})

    def test_ci_spin_coupling_strings_unified(self):
        merged = self._merge({'ci': '3', 'spin': '0',
                              'coupling': '1 2, 2 3, 1 3'})
        self.assertEqual(merged['ci'], [3])
        self.assertEqual(merged['spin'], [0])
        # The external classic coupling form is 1-based; the skill
        # schema consumes the normalized 0-based pair list.
        self.assertEqual(merged['coupling'], [[0, 1], [1, 2], [0, 2]])

    def test_pbc_cell_charges_strings_unified(self):
        merged = self._merge({'pbc': '1 1 1',
                              'cell': '10.0 10.0 10.0 90.0 90.0 90.0',
                              'charges': '0', 'multiplicity': '1'})
        self.assertEqual(merged['pbc'], [1, 1, 1])
        self.assertEqual(merged['cell'],
                         [10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        self.assertEqual(merged['charges'], 0)
        self.assertEqual(merged['multiplicity'], 1)

    def test_scalar_and_list_state_keys_unified(self):
        merged = self._merge({'target_states': '0 1 2', 'active': '8 9',
                              'nact_elec': '2', 'temperature': '300',
                              'root': '2', 'size': '20.67'})
        self.assertEqual(merged['target_states'], [0, 1, 2])
        self.assertEqual(merged['active'], [8, 9])
        self.assertEqual(merged['nact_elec'], 2)
        self.assertEqual(merged['temperature'], 300.0)
        # root's classic spelling is external 1-based; the skill layer
        # consumes the internal 0-based value ('2' -> 1).
        self.assertEqual(merged['root'], 1)
        self.assertEqual(merged['size'], 20.67)

    def test_highlevel_range_string_unified(self):
        # Same semantics as the framework Config boundary: classic
        # 1-based ranges become 0-based index lists.
        merged = self._merge({'highlevel': '1-6'})
        self.assertEqual(merged['highlevel'], [0, 1, 2, 3, 4, 5])

    def test_keys_are_never_renamed(self):
        # FIELD_MAP applies to the type lookup only: the merged dict
        # keeps the skill-vocabulary spelling.
        merged = self._merge({'temp': '300'})
        self.assertEqual(merged['temp'], 300.0)
        self.assertNotIn('temperature', merged)

    def test_string_coded_keys_keep_their_form(self):
        merged = self._merge({'sfhp': 'fssh', 'method': 'casscf',
                              'jobtype': 'energy gradient'})
        self.assertEqual(merged['sfhp'], 'fssh')
        self.assertEqual(merged['method'], 'casscf')
        self.assertEqual(merged['jobtype'], 'energy gradient')

    def test_typed_values_pass_through(self):
        merged = self._merge({'ci': [3], 'spin': [0],
                              'coupling': [[0, 1]], 'pbc': [1, 1, 1]})
        self.assertEqual(merged['ci'], [3])
        self.assertEqual(merged['coupling'], [[0, 1]])
        self.assertEqual(merged['pbc'], [1, 1, 1])

    def test_multicenter_strings_left_untouched(self):
        # ';'-separated center strings travel through the per-center
        # projection in planning/assembly.py; unification must not rewrite
        # them (multicenter runs pass remotely on this contract).
        merged = self._merge({'ci': '3; 3', 'spin': '0; 0',
                              'coupling': '1 2; 1 2'})
        self.assertEqual(merged['ci'], '3; 3')
        self.assertEqual(merged['spin'], '0; 0')
        self.assertEqual(merged['coupling'], '1 2; 1 2')

    def test_unparsable_values_kept_raw(self):
        merged = self._merge({'ci': 'three', 'cell': 'a b c'})
        self.assertEqual(merged['ci'], 'three')
        self.assertEqual(merged['cell'], 'a b c')

    def test_skill_private_keywords_pass_through(self):
        # The framework is unknown to skill-private keys: no table
        # entry, no conversion, the skill normalize owns them.
        merged = self._merge({'lammps_shake': '0', 'molcas_mem': '4000'})
        self.assertEqual(merged['lammps_shake'], '0')
        self.assertEqual(merged['molcas_mem'], '4000')

    def test_numeric_shapes_canonicalized(self):
        # A JSON int and a YAML float spelling of the same number must
        # not differ across modes: float-coded keys cast to float.
        from pyrai3md.schema.adapter import unify_skill_flat_dict
        flat = {'temperature': 300, 'size': 20, 'pressure': 1,
                'cell': [10, 10, 10, 90, 90, 90], 'step': 10,
                'keep_tmp': 1}
        unified = unify_skill_flat_dict(dict(flat))
        self.assertEqual(unified['temperature'], 300.0)
        self.assertIsInstance(unified['temperature'], float)
        self.assertEqual(unified['size'], 20.0)
        self.assertIsInstance(unified['pressure'], float)
        self.assertEqual(unified['cell'],
                         [10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        # Int-coded keys and skill-private keys keep their int shape.
        self.assertIsInstance(unified['step'], int)
        self.assertIsInstance(unified['keep_tmp'], int)


class TestProvenanceOverlay(unittest.TestCase):
    """The framework Config overlay keeps user-written values even
    when they coincide with the framework default (raw provenance),
    while never letting unwritten framework defaults overwrite the
    Maestro/skill layer below."""

    def _merge(self, raw_config, runtime=None):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        maestro = _StubMaestro(
            runtime={'molcas_skill': runtime or {}},
            skill_info={'molcas_skill': {'name': 'molcas'}})
        config = PyRAIConfig()
        # Simulate the typed Config carrying the user's md section.
        md_section = raw_config.get('md') or {}
        if 'temperature' in md_section:
            config.dynamics.temperature = float(md_section['temperature'])
        return _merge_one_skill_config(
            maestro, 'molcas_skill', config=config,
            raw_config=raw_config)

    def test_user_value_equal_to_framework_default_survives(self):
        # temperature 300 == framework DEFAULT: the raw provenance
        # marks it user-set, so it must reach the skill flat dict
        # instead of falling through to the skill default (298.15).
        merged = self._merge({'md': {'temperature': 300}})
        self.assertEqual(merged['temperature'], 300.0)

    def test_user_value_wins_over_maestro_default(self):
        merged = self._merge({'md': {'temperature': 300}},
                             runtime={'temperature': 350.0})
        self.assertEqual(merged['temperature'], 300.0)

    def test_unwritten_framework_default_never_overlays(self):
        # No temperature anywhere in the raw input: the Maestro value
        # below must survive (the original default-protection bug).
        merged = self._merge({'molcas': {'method': 'casscf'}},
                             runtime={'temperature': 350.0})
        self.assertEqual(merged['temperature'], 350.0)

    def test_field_map_alias_marks_canonical_key_user_set(self):
        # Classic spelling 'temp' marks the canonical 'temperature'
        # field as user-set.
        merged = self._merge({'md': {'temp': 300}})
        self.assertEqual(merged['temperature'], 300.0)


if __name__ == '__main__':
    unittest.main()
