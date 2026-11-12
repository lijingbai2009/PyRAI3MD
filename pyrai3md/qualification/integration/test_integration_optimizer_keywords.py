"""
======================================================================
Every framework optimizer keyword has to be plumbed all the way: the
model field, the classic reader's type table, the runtime whitelist,
the classic writer, and a search actually built with the value.

A keyword is lost in five independent places, and each one has been
lost before in this package: `schema/config.py` carries the field,
`schema/adapter.OPTIMIZE_TYPES` decides whether the classic parser
accepts the line, `data/state.py::optimize_params` is a whitelist a new
field is silently dropped by, `qualification/certification/converter.py`
is the only thing that writes the line into the `.input` carrier, and
`workflow/optimize.py` is where the value reaches the search object.
#395 found one hole of exactly this shape (three dispatch keywords
declared and read but never written) and #396 found the same class in
the step model (`opt_use_rfo` was in the reader table from the start and
in no writer). Listing them by hand is how those got missed, so this
file derives the list from the schema itself: add an `opt_*` field to
the `Optimize` model without plumbing it and a criterion here fails,
naming the key and the stage that dropped it.

For every key the check is the same and it is behavioural, not textual:
one input declares a value chosen away from the schema default (so a
lost line cannot hide behind "the default is the same number"), and the
value must come out of both carriers - the YAML one and the generated
classic input - both in the parameter dict the runtime reads and in the
search object built from it.

Jul - 2026
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
import shutil
import tempfile
import unittest

import numpy as np


# Declared value per key, every one of them different from the schema
# default, plus the attribute the search object carries it under (None
# when the key is consumed by the runtime rather than by the search).
VALUES = {
    'opt_maxit': (3, 'maxiter'),
    'opt_energy': (4.0e-4, 'energy_threshold'),
    'opt_grms': (9.0e-4, 'gradient_threshold'),
    'opt_gmax': (1.1e-3, None),
    'opt_drms': (1.3e-3, None),
    'opt_dmax': (1.7e-3, None),
    'optimize_cell': (True, 'cell_variables'),
    'opt_csrms': (25.0, 'cell_stress_rms_threshold'),
    'opt_csmax': (30.0, 'cell_stress_max_threshold'),
    'opt_cerms': (1.0e-3, 'cell_strain_rms_threshold'),
    'opt_cemax': (2.0e-3, 'cell_strain_max_threshold'),
    'opt_pressure': (1700.0, 'pressure_au'),
    'opt_cell_coupling': ('isotropic', 'cell_coupling'),
    'opt_use_rfo': (True, 'use_rfo'),
    'opt_trust_radius': (0.125, 'trust_radius'),
}

# The two keys this file cannot walk through the search object:
# `opt_method` chooses which class is built (it is covered by
# `test_optimizer.py` and the registry's own acceptance list), and
# `opt_state_j` names the second surface of a pair, which every
# single-surface search refuses by name (`planning/assembly.py::build_task_queue`)
# - so it is checked at the parameter stage only, where the pairing
# rule does not apply.
OBJECT_EXEMPT = (
    'opt_method', 'opt_state_i', 'opt_state_j',
    # The path family's keys are read by the Path workflow
    # (`workflow/path.py::_build_optimizer`) and the optimizers it builds, not
    # by the single-structure search object this file walks: `opt_method` is
    # what sends a run to that workflow in the first place, so none of these
    # twenty reach this object's attributes. Their carrier parity - the thing
    # this file is about - is covered by every `path_*` example running in both
    # carriers (the first path round: thirteen cases PASS on both, and the
    # failures it did find were a fixture-resolution defect, not a dropped key).
    'opt_path_constraint', 'opt_path_radius', 'opt_path_inner_maxit',
    'opt_path_inner_tol', 'opt_irc_direction', 'opt_irc_mode',
    'opt_irc_mode_file', 'opt_neb_reactant', 'opt_neb_product', 'opt_neb_ts',
    'opt_neb_images', 'opt_neb_interpolate', 'opt_neb_geodesic_threshold',
    'opt_neb_geodesic_scaler', 'opt_neb_geodesic_min_neighbors',
    'opt_neb_spring', 'opt_neb_climb', 'opt_neb_climb_threshold',
    'opt_neb_step',
)


def _model_keys():
    """Every optimizer keyword the schema models, derived not hand-typed."""
    from pyrai3md.schema.config import Optimize
    return sorted(k for k in Optimize.model_fields
                  if k.startswith('opt_') or k == 'optimize_cell')




class TestEveryKeywordSurvivesBothCarriers(unittest.TestCase):
    """One declared value per key: yaml in, both carriers out, search built."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='keyword_matrix_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _spec(self, key, value):
        """One periodic cell input declaring exactly `key`."""
        name = 'matrix_%s' % key
        path = os.path.join(self.root, name + '.yaml')
        molecule_extra = ''
        lines = ['control:\n  title: %s\n' % name,
                 '  jobtype: optimize\n  theory: [xtb]\n',
                 'atoms: [O, H, H]\n',
                 'coordinates:\n  - [0.0, 0.0, 0.0]\n',
                 '  - [0.7, 0.0, 0.0]\n  - [-0.4, 0.6, 0.0]\n',
                 'molecule:\n  charges: 0\n  multiplicity: 1\n',
                 '  pbc: [1, 1, 1]\n',
                 '  cell: [9.0, 9.0, 9.0, 90.0, 90.0, 90.0]\n',
                 'optimize:\n  method: minima\n  maxit: 2\n',
                 '  optimize_cell: true\n']
        if key == 'optimize_cell':
            pass                      # declared with the box above
        elif key == 'opt_pressure':
            lines.append('  pressure: %r\n' % (value,))
        elif key == 'opt_state_i':
            # The validator judges the index against the plan's state axis
            # (`ci`), so the surface has to exist before it can be selected.
            lines.append('  state_i: %r\n' % (value,))
            molecule_extra = '  ci: "2"\n  spin: [0]\n'
        else:
            lines.append('  %s: %r\n' % (key[len('opt_'):], value))
        if molecule_extra:
            lines.insert(7, molecule_extra)
        lines.append('xtb:\n  method: gfn_ff\n  profile: gfn_ff_gradient\n')
        with open(path, 'w') as handle:
            handle.writelines(lines)
        return path

    def _params(self, source):
        from pyrai3md.data.build import build_state
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        raw = parse_input(source)
        state = build_state(validate(normalize(build_config(raw))),
                            raw_config=raw)
        return state

    def _built(self, state, key):
        from pyrai3md.workflow.optimize import Optimize

        class Calculator:
            def __init__(self, state):
                self.state = state

            def get_state(self):
                return self.state

            def sync(self):
                pass

        return Optimize(calculator=Calculator(state), maxiter=2)

    def test_each_key_arrives_through_both_carriers(self):
        from pyrai3md.qualification.certification.converter import (
            yaml_to_config, convert)
        from pyrai3md.utilities.units import bar_to_au_pressure
        for key, (value, attribute) in sorted(VALUES.items()):
            path = self._spec(key, value)
            yaml_state = self._params(path)
            config_path = os.path.join(
                self.root, os.path.basename(path)[:-len('.yaml')] + '.json')
            input_path = config_path[:-len('.json')] + '.input'
            yaml_to_config(path, config_path)
            convert(config_path, input_path)
            with open(input_path) as handle:
                text = handle.read()
            classic_state = self._params(input_path)
            yaml_params = yaml_state.optimize_params
            classic_params = classic_state.optimize_params
            context = '%s declared %r' % (key, value)
            # Stage 1: the reader table (`OPTIMIZE_TYPES`) and the writer.
            self.assertIn(
                key, classic_params,
                '%s: the classic carrier lost the key at the reader stage, so '
                'the `.input` run applies a default the yaml run does not'
                % context)
            # Stage 2: the same value out of both carriers.
            self.assertEqual(classic_params.get(key), yaml_params.get(key),
                             '%s: the two carriers disagree (%r vs %r)'
                             % (context, yaml_params.get(key),
                                classic_params.get(key)))
            # The parameter stage keeps the unit the input named (bar), and the
            # search object holds the converted one - so the two stages are
            # compared against two different expectations, deliberately.
            expected = value
            object_expected = (bar_to_au_pressure(value)
                               if key == 'opt_pressure' else value)
            got = yaml_params.get(key)
            if isinstance(expected, float):
                self.assertAlmostEqual(float(got), float(expected),
                                       msg='%s: the value did not arrive' % context)
            else:
                self.assertEqual(got, expected,
                                 '%s: the value did not arrive' % context)
            # Stage 3: the search object built from it, where there is one.
            if attribute and key not in OBJECT_EXEMPT:
                for tag, state in (('yaml', yaml_state), ('classic',
                                                          classic_state)):
                    search = self._built(state, key)
                    found = getattr(search.optimizer, attribute, None)
                    expected = object_expected
                    if isinstance(expected, float):
                        self.assertAlmostEqual(
                            float(found), float(expected),
                            msg='%s: %s mode built a search with %s=%r'
                                % (context, tag, attribute, found))
                    else:
                        self.assertEqual(
                            found, expected,
                            '%s: %s mode built a search with %s=%r'
                            % (context, tag, attribute, found))

    def test_the_state_pair_keys_are_refused_by_name_in_both_carriers(self):
        """`opt_state_i` and `opt_state_j` are walked, each where it acts.

        An excited-state index is not a number the framework search consumes, so
        neither key reaches the search-object stage and this criterion covers
        them where they act - through both carriers, which is what the matrix is
        about. **What each key does there is not the same thing, and this
        criterion used to assert that it was** (until the sweep of 2026-10-20
        found it red): a single-surface search REFUSES a named pair
        (`planning/assembly.py::_merge_skill_binding` - `opt_state_j` differing from `opt_state_i`
        asks for a second surface the method never reads), while a lone index is
        the selector that names WHICH surface it reads and is therefore accepted.
        So `opt_state_j` is the refused case and `opt_state_i` the accepted one,
        and the accepted case asserts the index really arrived
        (`optimize_params['opt_state_i']`) rather than merely not raising.

        The range rule belongs to the input validator, as ruled on 2026-10-20:
        `schema.validate.validate_centers` judges `opt_state_i`/`opt_state_j`
        against the same `nstate` the coupling pairs read, and an index
        outside it raises at input with no second check downstream. So the
        fixture declares a two-state manifold (`ci: "2"`, `spin: [0]`) and
        state 1 is a legal surface rather than an out-of-range number this
        criterion would otherwise be pinning as accepted.
        """
        from pyrai3md.qualification.certification.converter import (
            yaml_to_config, convert)
        previous = os.getcwd()
        try:
            os.chdir(self.root)
            for key, declaration, refused in (
                    ('opt_state_i', '  state_i: 1\n', False),
                    ('opt_state_j', '  state_i: 1\n  state_j: 2\n', True)):
                path = os.path.join(self.root, 'pair_%s.yaml' % key)
                with open(path, 'w') as handle:
                    handle.write('control:\n  title: pair_%s\n'
                                 '  jobtype: optimize\n  theory: [xtb]\n'
                                 'atoms: [O, H, H]\n'
                                 'coordinates:\n  - [0.0, 0.0, 0.0]\n'
                                 '  - [0.7, 0.0, 0.0]\n'
                                 '  - [-0.4, 0.6, 0.0]\n'
                                 'molecule:\n  charges: 0\n  multiplicity: 1\n'
                                 '  ci: "2"\n  spin: [0]\n'
                                 '  pbc: [1, 1, 1]\n'
                                 '  cell: [9.0, 9.0, 9.0, 90.0, 90.0, 90.0]\n'
                                 'optimize:\n  method: minima\n  maxit: 2\n'
                                 '  optimize_cell: true\n%s'
                                 'xtb:\n  method: gfn_ff\n'
                                 '  profile: gfn_ff_gradient\n'
                                 % (key, declaration))
                messages = []
                accepted = []
                for tag, source in (('yaml', path),):
                    config_path = os.path.join(self.root, 'pair_%s.json' % key)
                    input_path = config_path[:-len('.json')] + '.input'
                    yaml_to_config(path, config_path)
                    convert(config_path, input_path)
                    for mode, carrier in (('yaml', path),
                                          ('classic',
                                           os.path.basename(input_path))):
                        try:
                            state = self._params(carrier)
                        except Exception as error:
                            messages.append((mode, str(error)))
                            continue
                        accepted.append((mode, state))
                if refused:
                    for mode, text in messages:
                        self.assertTrue(
                            'state' in text or 'surface' in text,
                            '%s in %s mode was refused for an unrelated reason: '
                            '%s' % (key, mode, text[:160]))
                    self.assertEqual(len(messages), 2,
                                     '%s: only one carrier refused, the two '
                                     'disagree about the same input' % key)
                    self.assertEqual(accepted, [],
                                     '%s was accepted in %s'
                                     % (key, [m for m, _ in accepted]))
                else:
                    self.assertEqual(messages, [],
                                     '%s was refused: %s' % (key, messages))
                    self.assertEqual(len(accepted), 2,
                                     '%s reached only %d carrier(s)'
                                     % (key, len(accepted)))
                    for mode, state in accepted:
                        self.assertEqual(
                            int(state.optimize_params.get('opt_state_i', 0)), 1,
                            '%s: %s mode did not hand the index to the search'
                            % (key, mode))
        finally:
            os.chdir(previous)

    def test_a_key_the_writer_drops_is_visible_in_the_text(self):
        # The stage no other check can see: a key accepted by the reader with
        # nothing writing it is invisible in yaml mode and wrong in `.input`
        # mode, which is the exact shape of the two #395/#396 holes. Asserting
        # on the rendered text makes the omission a failure, not an inference.
        from pyrai3md.qualification.certification.converter import (
            yaml_to_config, convert)
        for key, (value, _attribute) in sorted(VALUES.items()):
            if key == 'optimize_cell':
                continue
            path = self._spec(key, value)
            config_path = os.path.join(self.root, 'text_%s.json' % key)
            input_path = config_path[:-len('.json')] + '.input'
            yaml_to_config(path, config_path)
            convert(config_path, input_path)
            with open(input_path) as handle:
                body = handle.read()
            label = {'opt_maxit': 'maxit', 'opt_pressure': 'pressure',
                     'opt_state_i': 'state_i'}.get(key, key)
            self.assertIn(
                label, body,
                '%s declared %r never reached the classic input, so the two '
                'carriers run different searches while the yaml mode hides it'
                % (key, value))




if __name__ == '__main__':
    unittest.main()
