"""
======================================================================
The merged subsystem at the framework's own boundaries.

Three seams carry the constraint code into a live run, and each one has a way
to fail that no unit test of the numeric core can see:

- the classic `.input` text, where a nested block used to be impossible: a
  repeated key overwrote and a literal `&end` opened a section named `end` and
  quietly filed the rest of the file there (design document section 3.2);
- the checkpoint, where a restarted run must not re-read its bound targets from
  the geometry it resumed at, and must refuse a file whose input has since
  changed (section 12);
- the thermostat, which counts the degrees of freedom the constraints removed
  (section 10, D-06).

Every test here goes through the framework's own entry points - `parse_input`,
`StateSerializer.save` and `load`, `ensemble._dof` - because the point is the
seam, not the arithmetic behind it.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

try:
    import pytest
except ImportError:  # unittest-only hosts do not need pytest
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.integration

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.qualification.verification._constraint_helpers import (
    BLOCK, COORDS, _state)
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

CLASSIC_INPUT = """&control
  title constraint_classic
  jobtype md
  theory pyscf
&molecule
  atoms O H H
  coord 0.000000 0.000000  0.117340
  coord 0.000000  0.757540 -0.469360
  coord 0.000000 -0.757540 -0.469360
&end
&constraint
  cv_bond 1 2
  cv_name oh
  bias_cv oh
  bias_cv_ref 0.90
  bias_cv_fc 0.02
  bias_id hold
  wall_type sphere
  wall_sphere_radius 1.4
  wall_center 0 0 0
  wall_atoms 1-3
  wall_fc 0.001
  wall_id vessel
  shake_bond 1 3
  shake_vel rattle
  shake_id ohb
&end
"""












ATOMS = ['O', 'H', 'H', 'C']
MASSES = [15.999, 1.008, 1.008, 12.011]




class TestShippedConstraintExamples(unittest.TestCase):
    """Every constraint example that ships in `pyrai3md/examples/` must load.

    Two of this session's defects were of the kind no numeric test can see: an
    angle record written with `atoms:` instead of `bonds:` (silently no rows),
    and a declared quantity that a real run later proved impossible. Both are
    caught by pushing each file through the same four steps a run does -
    `parse_input`, `build_config`, `normalize`, `validate` - and then building
    the plan, without needing an engine.

    These inputs came from `draft_constraint_examples/`, which the user folded
    into the official example campaign: the files now ship in the package, so
    this loop reads them where a run reads them and no checkout can skip it.
    """

    @staticmethod
    def _example_paths():
        import glob

        import yaml
        import pyrai3md
        root = os.path.dirname(os.path.abspath(pyrai3md.__file__))
        paths = []
        for path in sorted(glob.glob(os.path.join(root, 'examples', '*.yaml'))):
            with open(path, 'r', encoding='utf-8') as handle:
                spec = yaml.safe_load(handle) or {}
            if any(isinstance(spec.get(name), dict) and spec.get(name)
                   for name in ('system_constraint', 'constraint')):
                paths.append(path)
        return paths

    def test_every_example_loads_and_declares_something(self):
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import build_plan, normalize
        from pyrai3md.schema.validate import validate
        from pyrai3md.workflow.constraints import state_adapter

        paths = self._example_paths()
        self.assertGreater(len(paths), 30)
        for path in paths:
            name = os.path.basename(path)
            with self.subTest(example=name):
                config = validate(normalize(build_config(parse_input(path))))
                system = config.system
                block = getattr(system, 'constraints', None)
                declared = [key for key in ('constraints', 'cvs', 'biases',
                                            'walls', 'fixes', 'metadynamics')
                            if getattr(block, key, None)]
                self.assertTrue(declared,
                                '%s declares no constraint' % name)
                plan = build_plan(block.model_dump(),
                                  atoms=list(system.atoms or []),
                                  natom=int(system.natom or 0),
                                  masses=list(system.mass or []) or None,
                                  pbc=list(system.pbc or []) or None,
                                  coordinates=list(system.coordinates or [])
                                  or None,
                                  lattice=list(system.lattice_vector or [])
                                  or None,
                                  md_step=int(getattr(
                                      getattr(config, 'dynamics', None),
                                      'step', 0) or 0) or None)
                self.assertFalse(plan.is_empty,
                                 '%s declares a block but plans nothing'
                                 % name)
                # The rows a strict constraint solves are read back through the
                # accessor the runtime itself uses, because bond rows and
                # angle/torsion rows are not stored the same way: an angle
                # written with `atoms:` instead of `bonds:` would land here as a
                # condition with no rows, which is the mistake this loop exists
                # to catch.
                for spec in plan.actions:
                    if spec.kind == 'constraint':
                        self.assertTrue(state_adapter._condition_rows(spec),
                                        '%s: %s has no condition rows'
                                        % (name, spec.id))
                    if spec.kind == 'wall':
                        self.assertIn(spec.parameters.get('geometry'),
                                      ('sphere', 'ellipsoid', 'box',
                                       'triclinic'),
                                      '%s: %s names no geometry' % (name,
                                                                    spec.id))

    def test_writing_a_shipped_block_out_and_back_in_keeps_the_plan(self):
        """The writer is the reader's inverse, on every input that ships.

        This is the criterion that folded the constraint drafts into the
        example set and it stays as a guard because the three defects it found
        were all of one kind: the writer producing a text input that reads
        back as a *different* run (all but the first bond row dropped, every
        number truncated to six digits, a retired key raising rather than
        refusing). A plan fingerprint compares the declarations after selector
        resolution and unit conversion, so it catches a value that changes, not
        merely a line that goes missing.
        """
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.qualification.certification.constraint_writer import write_constraint_section
        from pyrai3md.schema.adapter import (build_config,
                                             _constraint_declaration)
        from pyrai3md.schema.normalize import build_plan, normalize
        from pyrai3md.schema.validate import validate

        def _plan(mapping, system, config):
            return build_plan(
                mapping, atoms=list(system.atoms or []),
                natom=int(system.natom or 0),
                masses=list(system.mass or []) or None,
                pbc=list(system.pbc or []) or None,
                coordinates=list(system.coordinates or []) or None,
                lattice=list(system.lattice_vector or []) or None,
                md_step=int(getattr(getattr(config, 'dynamics', None),
                                    'step', 0) or 0) or None)

        for path in self._example_paths():
            name = os.path.basename(path)
            with self.subTest(example=name):
                config = validate(normalize(build_config(parse_input(path))))
                system = config.system
                dumped = system.constraints.model_dump()
                written = dict(write_constraint_section(dumped))
                self.assertTrue(written,
                                '%s writes no keyword line at all' % name)
                read_back = _constraint_declaration(written)
                self.assertEqual(
                    _plan(dumped, system, config).fingerprint,
                    _plan(read_back, system, config).fingerprint,
                    '%s: the section the writer emits does not read back as '
                    'the run that wrote it' % name)

    def test_the_classic_input_keeps_every_constraint_keyword(self):
        """The second entry point must carry the restraint the first one got.

        Qualification runs each example twice, through the YAML and through the
        classic text the converter renders from it. `yaml_to_config` treated
        every non-framework section as a skill's parameter block, so
        `system_constraint` went to `skill_sections`, the converter wrote a
        `&SYSTEM_CONSTRAINT` holding the skill-project line and the shared
        `scratch_path`, and the constraint keywords were dropped: the text mode
        ran the unconstrained physics, and re-reading that section is refused by
        the constraint reader anyway. The check is on the keyword set, because
        that is what a silent drop changes.
        """
        import yaml

        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config)

        for path in self._example_paths():
            name = os.path.basename(path)
            with open(path, 'r', encoding='utf-8') as handle:
                spec = yaml.safe_load(handle)
            declared = (spec.get('system_constraint')
                        or spec.get('constraint') or {})
            with tempfile.TemporaryDirectory() as work:
                json_path = os.path.join(work, 'case.json')
                config = yaml_to_config(path, json_path)
                self.assertNotIn('system_constraint',
                                 config.get('skill_sections') or {},
                                 '%s: the block was filed as a skill section'
                                 % name)
                self.assertTrue(config.get('constraints'),
                                '%s: no constraint block reached the config'
                                % name)
                text = convert(json_path)
            self.assertIn('&CONSTRAINT', text, name)
            self.assertNotIn('&SYSTEM_CONSTRAINT', text, name)
            rendered = _parse_text(
                [line.rstrip('\n') for line in text.splitlines()])
            emitted = rendered.get('constraint') or {}
            self.assertEqual(sorted(str(key).lower() for key in declared),
                             sorted(str(key).lower() for key in emitted),
                             '%s: the text mode declares a different run'
                             % name)


    def test_the_classic_input_keeps_every_md_keyword(self):
        """The dynamics section has to survive the second entry point too.

        The constraint keywords are guarded above; the same silent-drop failure
        was live on the dynamics side. `md.checkpoint` is accepted by the classic
        reader (`schema/adapter.MD_TYPES`) but was missing from the writer's
        `_MD_ECHO_KEYS`, so on 2026-10-02 the periodic restart examples ran their
        `.input` mode with no checkpoint at all: no `.rt` file, and a recorded
        frame set different from the YAML mode's (5 frames against 2). A keyword
        the reader knows and the writer does not emit is not an error anywhere on
        the path, so the check is on the key set.

        `temperature` renders as `temp`, which the reader's FIELD_MAP restores;
        that alias is the one name difference the assertion allows.
        """
        import yaml

        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config)

        for path in self._example_paths():
            name = os.path.basename(path)
            with open(path, 'r', encoding='utf-8') as handle:
                spec = yaml.safe_load(handle)
            declared = spec.get('md') or {}
            if not declared:
                continue
            with tempfile.TemporaryDirectory() as work:
                json_path = os.path.join(work, 'case.json')
                yaml_to_config(path, json_path)
                text = convert(json_path)
            self.assertIn('&MD', text, '%s: no dynamics section' % name)
            parsed = _parse_text(
                [line.rstrip('\n') for line in text.splitlines()])
            emitted = {str(key).lower() for key in (parsed.get('md') or {})}
            # The comparison is against the classic vocabulary the reader
            # accepts. `md.seed`, declared by ten of these files, is not in
            # `MD_TYPES` and not a field of the dynamics model either, so it is
            # outside this guard's question: nothing carries it, in either mode.
            from pyrai3md.schema.adapter import MD_TYPES
            expected = {('temp' if str(key).lower() == 'temperature'
                         else str(key).lower()) for key in declared
                        if str(key).lower() in MD_TYPES}
            self.assertEqual(expected, expected & emitted,
                             '%s: the text mode drops %s'
                             % (name, sorted(expected - emitted)))

    def test_the_classic_input_keeps_the_control_seed(self):
        """``control.gl_seed`` is the one seed the dynamics draw reads.

        The constraint inputs used to carry ``md.seed``, which names no field of
        the dynamics model and no classic keyword, so nothing read it and every
        entry point drew the default. The value now lives in ``control.gl_seed``,
        and the writer emits it presence-based -- the same gap ``_MD_ECHO_KEYS``
        documents for the dynamics section, and the reason this is asserted rather
        than trusted: a dropped seed makes two carriers draw different velocities
        while both still report a pass.
        """
        import yaml

        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config)

        seeded = 0
        for path in self._example_paths():
            name = os.path.basename(path)
            with open(path, 'r', encoding='utf-8') as handle:
                spec = yaml.safe_load(handle)
            gl_seed = (spec.get('control') or {}).get('gl_seed')
            if gl_seed is None:
                continue
            seeded += 1
            with tempfile.TemporaryDirectory() as work:
                json_path = os.path.join(work, 'case.json')
                yaml_to_config(path, json_path)
                text = convert(json_path)
            control = (_parse_text(
                [line.rstrip('\n') for line in text.splitlines()])
                .get('control') or {})
            self.assertIn('gl_seed', control,
                          '%s: &CONTROL lost the seed' % name)
            self.assertEqual(int(control['gl_seed']), int(gl_seed),
                             '%s: &CONTROL carries a different seed' % name)
        self.assertGreater(seeded, 10,
                           'no constraint input declares a seed')






if __name__ == '__main__':
    unittest.main()


