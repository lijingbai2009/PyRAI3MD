"""
======================================================================
Three-mode flat-dict unification contract tests.

The framework hands every skill backend a neutral flat dict
(skill_config). This suite locks the determinism contract of
pyrai3md.schema.adapter.unify_skill_flat_dict: for every official
example, the three input modes (yaml / classic .input / agent
.config) must produce a key-by-key identically typed merged flat
dict on the shared framework/skill vocabulary, so a skill schema
sees one canonical value shape regardless of the entry point.

Two levels are asserted:
- Merge level: for keys present in all three modes, value and type
  must match exactly. Multi-center (';'-separated) strings are
  skipped: they are owned by the per-center typed projection in
  planning/assembly.py, not by the flat merge.
- Skill level (single-center examples): each skill's own
  build_config(merged) model dump must agree across the three modes
  on the shared vocabulary, proving the effective SkillConfig the
  backend validates is mode-independent.

Sep 25 2026
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

import glob
import importlib
import json
import os
import tempfile
import unittest

from pyrai3md.interface.parser import parse_input
from pyrai3md.kernel.environment import Environment
from pyrai3md.qualification import manager
from pyrai3md.qualification.certification.converter import (
    _skill_folder, convert, convert_to_pyrai_config, parse_test_yaml,
    yaml_to_config)
from pyrai3md.schema import adapter as framework_adapter
from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.normalize import normalize

_EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'examples'))

# Shared framework/skill vocabulary whose value shape must be
# mode-independent (the unification type table plus the center-parsed
# field sets).
_VOCABULARY = (
    set(framework_adapter._SKILL_UNIFICATION_TYPES)
    | set(framework_adapter._CENTER_INDEX_FIELDS)
    | set(framework_adapter._CENTER_INT_LIST_FIELDS)
    | set(framework_adapter._CENTER_PAIR_FIELDS)
    | set(framework_adapter._CENTER_SCALAR_FIELDS)
    | set(framework_adapter._CENTER_SCALAR_LIST_FIELDS)
)


_PART_ENV = 'PRAI3MD_INTEGRATION_PART'


def _example_yamls():
    """The examples this run covers: all of them, or this part's share.

    The qualification planner splits this file into parts and exports
    `PRAI3MD_INTEGRATION_PART` as `<index>/<total>`; the same variable is
    unset for a direct pytest/unittest run, which then covers every
    example. Parts are interleaved by index so each one gets a mix of
    cheap and expensive examples.
    """
    paths = sorted(glob.glob(os.path.join(_EXAMPLES_DIR, '*.yaml')))
    raw = os.environ.get(_PART_ENV, '').strip()
    if not raw:
        return paths
    try:
        index, total = (int(part) for part in raw.split('/'))
    except ValueError:
        raise AssertionError('%s must read <index>/<total>, got %r'
                             % (_PART_ENV, raw))
    if not 0 <= index < total:
        raise AssertionError('%s names part %d of %d, out of range'
                             % (_PART_ENV, index, total))
    return [path for position, path in enumerate(paths)
            if position % total == index]


class TestThreeModeFlatDictUnification(unittest.TestCase):
    """yaml / classic input / agent config merge into identically
    typed skill flat dicts on the shared vocabulary."""

    @classmethod
    def setUpClass(cls):
        cls._env = Environment()
        cls._env.initialize()
        cls._maestro = (getattr(cls._env, 'maestro', None)
                        or getattr(cls._env, '_maestro', None))
        cls._pmd_env = manager.load_config()

    def _mode_raws(self, yaml_path, tmpdir):
        """Build the (label, raw_config) triples of one example."""
        stem = os.path.splitext(os.path.basename(yaml_path))[0]
        flat_path = os.path.join(tmpdir, stem + '.json')
        yaml_to_config(yaml_path, flat_path)
        manager._inject_example_skill_env(
            flat_path, self._pmd_env.get('skills', {}))
        with open(flat_path, encoding='utf-8') as handle:
            flat = json.load(handle)

        input_path = os.path.join(tmpdir, stem + '.input')
        convert(flat_path, input_path)
        agent_path = os.path.join(tmpdir, stem + '.config')
        with open(agent_path, 'w', encoding='utf-8') as handle:
            json.dump(convert_to_pyrai_config(flat_path), handle)

        with open(agent_path, encoding='utf-8') as handle:
            agent_raw = json.load(handle)
        return flat, [
            ('yaml', parse_input(yaml_path)),
            ('input', parse_input(input_path)),
            ('agent', agent_raw),
        ]

    def _merged_per_mode(self, raws, skill):
        merged = {}
        for label, raw in raws:
            config = normalize(build_config(raw))
            merged[label] = framework_adapter._merge_one_skill_config(
                self._maestro, skill, config=config, raw_config=raw) or {}
        return merged

    @staticmethod
    def _multicenter_string(value):
        return isinstance(value, str) and ';' in value

    def test_merge_level_shapes_agree_across_modes(self):
        for yaml_path in _example_yamls():
            stem = os.path.splitext(os.path.basename(yaml_path))[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                flat, raws = self._mode_raws(yaml_path, tmpdir)
                for theory in flat.get('theory') or []:
                    skill = '%s_skill' % theory
                    merged = self._merged_per_mode(raws, skill)
                    common = set(merged['yaml'])
                    for label in ('input', 'agent'):
                        common &= set(merged[label])
                    for key in sorted(common & _VOCABULARY):
                        values = {label: merged[label][key]
                                  for label in merged}
                        if any(self._multicenter_string(v)
                               for v in values.values()):
                            continue  # owned by the core.py projection
                        with self.subTest(example=stem, skill=skill,
                                          key=key):
                            for label in ('input', 'agent'):
                                self.assertEqual(
                                    values[label], values['yaml'],
                                    '%s: %s value differs between yaml '
                                    'and %s mode' % (stem, key, label))
                                self.assertIs(
                                    type(values[label]),
                                    type(values['yaml']),
                                    '%s: %s type differs between yaml '
                                    'and %s mode' % (stem, key, label))

    def test_skill_level_config_agrees_across_modes(self):
        # Single-center examples only: a multi-center merged dict still
        # carries ';' strings that planning/assembly.py projects per center
        # before the backend validates, which this direct build_config
        # comparison does not emulate.
        for yaml_path in _example_yamls():
            stem = os.path.splitext(os.path.basename(yaml_path))[0]
            if 'multicenter' in stem:
                continue
            with tempfile.TemporaryDirectory() as tmpdir:
                flat, raws = self._mode_raws(yaml_path, tmpdir)
                for theory in flat.get('theory') or []:
                    # The theory stem is the input spelling, not the
                    # package name: `nn` lives in `nn_pes_skill`, and a
                    # stem that differs from its folder makes this either
                    # skip the skill's adapter or import a module that does
                    # not exist. The registry knows the pair, so it is
                    # asked. Skipping the folder check instead would let an
                    # engine whose skill package is named differently go
                    # unverified in every mode but the yaml one.
                    skill = _skill_folder(theory)
                    merged = self._merged_per_mode(raws, skill)
                    if any(self._multicenter_string(
                            merged[label].get('ci'))
                            for label in merged):
                        continue
                    skill_adapter = importlib.import_module(
                        'pyrai3md.skills.%s.schema.adapter' % skill)
                    # Presence layering (which mode supplies a key at
                    # all) is the documented merge-priority semantics,
                    # not the unification contract: compare only the
                    # vocabulary keys every mode provides, and require
                    # identical value and type on those.
                    common = set(merged['yaml'])
                    for label in ('input', 'agent'):
                        common &= set(merged[label])
                    dumps = {}
                    for label in merged:
                        config = skill_adapter.build_config(
                            dict(merged[label]))
                        dumps[label] = config.model_dump()
                    for key in sorted(common & _VOCABULARY):
                        values = {}
                        for label, dump in dumps.items():
                            for section in dump.values():
                                if (isinstance(section, dict)
                                        and key in section):
                                    values[label] = section[key]
                                    break
                        if len(values) < 2:
                            continue
                        if any(self._multicenter_string(v)
                               for v in values.values()):
                            continue
                        with self.subTest(example=stem, skill=skill,
                                          key=key):
                            reference = values.get('yaml')
                            for label, value in values.items():
                                if reference is None:
                                    continue
                                self.assertEqual(
                                    value, reference,
                                    '%s: %s SkillConfig field differs '
                                    'in %s mode' % (stem, key, label))
                                self.assertIs(
                                    type(value), type(reference),
                                    '%s: %s SkillConfig field type '
                                    'differs in %s mode'
                                    % (stem, key, label))


_LAMMPS_TESTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'skills',
    'lammps_skill', 'tests'))
_LAMMPS_CASES = ('ff_energy', 'ff_gradient', 'ff_gradient_persistent',
                 'ff_stress', 'ff_nve', 'ff_nvt', 'ff_npt', 'ff_optimize',
                 'ff_optimize_cell', 'ff_box_relax_iso',
                 # The two `fix box/relax` spellings, one per engine line. Both
                 # carry the fixture box, which is what the assertion below is
                 # able to demand of them.
                 'ff_box_relax_aniso_pressure')
def _fixture_cell():
    """The box the h2o.data topology fixture declares, as a 6-parameter cell.

    A skill test YAML and its topology data file state the same box twice,
    which is exactly how the two drift apart, so the assertion below reads
    the expected numbers from the data file instead of repeating a literal
    here: refreshing the fixture then moves the test with it. The fixture
    is an untilted box, so the angles are the orthorhombic default.
    """
    sides = []
    with open(os.path.join(_LAMMPS_TESTS_DIR, 'h2o.data'),
              encoding='utf-8') as handle:
        for line in handle:
            line = line.rstrip('\n')
            for name in ('xlo xhi', 'ylo yhi', 'zlo zhi'):
                if line.endswith(name):
                    parts = line.split()
                    sides.append(float(parts[1]) - float(parts[0]))
    if len(sides) != 3:
        raise AssertionError('h2o.data declares %d box faces, expected 3'
                             % len(sides))
    return sides + [90.0, 90.0, 90.0]


def _find_key(node, key, seen=None):
    """First value stored under ``key`` anywhere inside a nested document.

    The three renderings keep a system field at different depths (flat in
    the skill ``.json``, sectioned in the classic ``.input``, nested under
    ``system`` in the agent ``.config``), so the comparison is about the
    value surviving, not about where it sits.
    """
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_key(value, key, seen)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key, seen)
            if found is not None:
                return found
    return None


def _as_numbers(value):
    """Coerce a system field to a number list: the classic .input keeps
    list-valued keywords as one whitespace-separated string, the JSON
    renderings keep them as lists."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    return [float(item) for item in value]


class TestSkillYamlPeriodicityReachesEveryArtifact(unittest.TestCase):
    """A skill test YAML that declares a periodic box must carry it into
    every planned artifact: the classic .input and the agent .config are
    read back by runs that never see the YAML."""

    def _artifacts(self, case, tmpdir):
        yaml_path = os.path.join(_LAMMPS_TESTS_DIR, case + '.yaml')
        flat = parse_test_yaml(yaml_path, 'lammps_skill')
        json_path = os.path.join(tmpdir, case + '.json')
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump(flat, handle)
        input_path = os.path.join(tmpdir, case + '.input')
        convert(json_path, input_path)
        agent_path = os.path.join(tmpdir, case + '.config')
        with open(agent_path, 'w', encoding='utf-8') as handle:
            json.dump(convert_to_pyrai_config(json_path), handle)
        return {
            'json': flat,
            'input': parse_input(input_path),
            'agent': self._load(agent_path),
        }

    @staticmethod
    def _load(path):
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)

    def test_pbc_declared_on_every_case(self):
        for case in _LAMMPS_CASES:
            with self.subTest(case=case):
                raw = parse_test_yaml(
                    os.path.join(_LAMMPS_TESTS_DIR, case + '.yaml'),
                    'lammps_skill')
                self.assertEqual(raw.get('pbc'), [1, 1, 1])
                self.assertEqual([float(v) for v in raw.get('cell') or []],
                                 _fixture_cell())

    def test_periodicity_survives_all_three_renderings(self):
        for case in _LAMMPS_CASES:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                for label, doc in self._artifacts(case, td).items():
                    pbc = _as_numbers(_find_key(doc, 'pbc'))
                    cell = _as_numbers(_find_key(doc, 'cell'))
                    self.assertEqual(
                        [int(value) for value in pbc], [1, 1, 1],
                        '%s: %s mode lost the declared periodicity'
                        % (case, label))
                    self.assertEqual(
                        cell[:6], _fixture_cell(),
                        '%s: %s mode lost the declared cell' % (case, label))


if __name__ == '__main__':
    unittest.main()
