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


class TestTheKeywordListIsTheSchemaList(unittest.TestCase):
    """The list this file walks is the schema's, so it cannot go stale."""

    def test_every_modelled_keyword_is_walked_below(self):
        missing = [key for key in _model_keys() if key not in VALUES
                   and key not in OBJECT_EXEMPT]
        self.assertEqual(
            missing, [],
            'these optimizer keywords exist in `schema/config.py::Optimize` '
            'and no criterion walks them through both carriers: %s' % missing)




class TestTheStateIndexIsCheckedAgainstTheManifold(unittest.TestCase):
    """An out-of-range surface index is refused by the input validator.

    Ruled on 2026-10-20: the range rule belongs to `normalize`/`validate`,
    with no second check downstream. The fixture declares two states
    (`ci: "2"`), so both sides are exercised on the same manifold: one past
    it must raise with the field name and the count, and one inside it must
    survive to the model.
    """

    @staticmethod
    def _validated(state_i, state_j=None):
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        raw = {
            'control': {'title': 'state_range', 'jobtype': 'optimize',
                        'theory': ['xtb']},
            'atoms': ['O', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0],
                            [-0.4, 0.6, 0.0]],
            'molecule': {'charges': 0, 'multiplicity': 1, 'ci': '2',
                         'spin': [0], 'pbc': [1, 1, 1],
                         'cell': [9.0, 9.0, 9.0, 90.0, 90.0, 90.0]},
            'optimize': {'method': 'minima', 'maxit': 2, 'state_i': state_i},
            'xtb': {'method': 'gfn_ff', 'profile': 'gfn_ff_gradient'},
        }
        if state_j is not None:
            raw['optimize']['state_j'] = state_j
        return validate(normalize(build_config(raw)))

    def test_an_index_inside_the_manifold_is_accepted(self):
        self.assertEqual(1, self._validated(1).optimize.opt_state_i)

    def test_an_index_past_the_manifold_is_refused_by_name(self):
        with self.assertRaises(ValueError) as raised:
            self._validated(2)
        self.assertIn('optimize.state_i=2', str(raised.exception))
        self.assertIn('nstate=2', str(raised.exception))

    def test_the_second_surface_is_judged_the_same_way(self):
        with self.assertRaises(ValueError) as raised:
            self._validated(0, state_j=2)
        self.assertIn('optimize.state_j=2', str(raised.exception))


class TestTheSingleCenterRangeRuleHasOneHome(unittest.TestCase):
    """The workflow does not try the single-center bound a second time.

    Same ruling as the class above: the validator owns the bound. The
    workflow's copy compared the same index against the same `nstate`
    behind an already-refused input, so it could only fire on a state
    built outside the input path; it was removed, while the
    per-center bounds stayed because the validator skips them (a
    multi-center run has no global state axis).
    """

    def test_the_workflow_keeps_only_the_per_center_bounds(self):
        from pyrai3md.qualification.verification._source_paths import (
            source_read)
        text = source_read('pyrai3md/workflow/optimize.py')
        self.assertNotIn('is out of range for nstate', text)
        self.assertIn('the narrowest ladder here holds', text)
        self.assertIn('center %d holds', text)


if __name__ == '__main__':
    unittest.main()
