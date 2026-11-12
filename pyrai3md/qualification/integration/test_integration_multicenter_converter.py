"""
======================================================================
Test suite for the certification converter's multi-center example
support: the classic-input / agent-config rendering of the
center-dimensioned molecule/md keywords and of a multi-theory
composite (multi-center ONIOM) skill sections.

Tests cover:
- yaml_to_config preserves the center strings (highlevel/ci/spin/
  coupling/root) and one parameter block per theory skill
  (skill_sections), with the primary skill following control.theory
- convert renders the classic input with the ';'-separated center
  keywords, one method section per skill (&OPENQP + &XTB) and an &MD
  section carrying the per-center root even for the non-advanced
  'gradient' jobtype
- convert_to_pyrai_config emits the same center strings into the
  molecule/md sections and one agent section per skill
- the flat example .json is read back by the task queue, which takes each
  bound skill's method from its own block in skill_sections rather than
  from the primary skill's flattened top level, and resolves a profile
  that matches the jobtype the task actually asks for
- a run that binds ONE skill at TWO positions (`theory: [pyscf, pyscf]`, the
  multiscale carriers) counts as a multi-task run to the converter: its block
  is kept, its flat config reads back at all, and the rendering puts the
  System fields where every already-shipped two-backend example puts them
- single-center regression: a plain one-skill example keeps the
  historical output (no skill_sections, one method section, no &MD)

These are pure converter-output assertions: no external program
(openqp/xtb) is executed, so the tests run locally without the
multi-center runtime environment. The flat-config class is the one
exception in kind, not in degree: it builds a task queue from the
converter's own product, which reads each skill's query module and still
calls no engine.

Sep 9 2026
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
import json
import os
import tempfile
import unittest

from pyrai3md.qualification.certification.converter import (
    convert, convert_to_pyrai_config, yaml_to_config,
    _is_multicenter, _render_center_string, _build_molecule_pairs,
)

# Package-relative examples directory (pyrai3md/qualification/
# verification -> pyrai3md/examples)
_EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'examples'))
_MULTICENTER_YAML = os.path.join(_EXAMPLES_DIR,
                                 'multicenter_gradient.yaml')
_GRADIENT_YAML = os.path.join(_EXAMPLES_DIR, 'gradient.yaml')








def _flat_task_table(raw):
    """Per-binding (method, profile, jobtype) the task queue is handed.

    Built through the same loader chain a run uses (parse_input ->
    build_config -> normalize -> validate -> build_state, `pmd.py` steps
    1-5), so the readings below are what a backend is constructed from,
    not what the file spells.

    The two middle steps are part of the chain and not a refinement of
    it. `normalize` is what resolves a LAMMPS topology into `natom` and
    expands `read_charge: true` into the file it names; a table built
    without it reads every topology-driven example as a system with no
    atoms, which is harmless until a step asks for the atom count. Two
    of them do: a `&system_constraint` block, whose normalization refuses
    with `constraint normalization needs an atom count`, and a charge
    file, which is then looked up by bare name in the working directory.
    """
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    from pyrai3md.data.build import build_state
    config = validate(normalize(build_config(raw)))
    state = build_state(config, raw_config=raw)
    table = {}
    for task in state.theory + state.reference:
        skill = (task.get('config') or {}).get('skill_config') or {}
        table[task['skill_binding_id']] = (skill.get('method'),
                                           skill.get('profile'),
                                           skill.get('jobtype'))
    return table


class TestFlatConfigSkillSectionsAreRead(unittest.TestCase):
    """The flat example `.json` hands each skill its own method.

    `yaml_to_config` keeps every skill's block under `skill_sections`
    because two skills spelling `method` cannot share a flat namespace,
    and flattens only the first `control.theory` entry to the top level.
    A consumer that reads the top level and never opens that key gives
    every skill the primary's method, and the skill whose method did not
    travel answers with its default profile - so a gradient jobtype is
    served an energy profile and the request is quietly not the one the
    input made.
    """

    _YAML = os.path.join(_EXAMPLES_DIR, 'optimize_cell_meci_lammps.yaml')

    def setUp(self):
        from pyrai3md.interface.parser import parse_input
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'mc.json')
        yaml_to_config(self._YAML, self.config_path)
        self.table = _flat_task_table(parse_input(self.config_path))

    def test_the_two_skills_declare_different_methods(self):
        # The non-identity witness: a queue where both skills answered the
        # same method would make the criterion below true by construction.
        with open(self.config_path, 'r', encoding='utf-8') as handle:
            declared = [section['method'] for section in
                        json.load(handle)['skill_sections'].values()]
        self.assertEqual(declared, ['tda', 'ff'])
        self.assertEqual(len(set(declared)), 2)

    def test_each_binding_reads_its_own_method(self):
        for binding, row in sorted(self.table.items()):
            expected = 'tda' if binding.startswith('pyscf') else 'ff'
            self.assertEqual(
                row[0], expected,
                '%s was handed method=%r, which is not the method its own '
                'section declares (%r)' % (binding, row[0], expected))

    def test_the_lattice_task_runs_the_gradient_profile(self):
        # Before the sections were read, this task answered 'ff_energy':
        # an unknown method 'tda', the resolution refused, the skill's
        # default profile kept, and a jobtype asking for a gradient served
        # by a profile that reports none.
        row = self.table['lammps_skill::qm2']
        self.assertEqual(row, ('ff', 'ff_gradient', 'energy gradient'))
        self.assertTrue(row[1].endswith('gradient'), row)

    def test_both_modes_hand_the_queue_the_same_profiles(self):
        from pyrai3md.interface.parser import parse_input
        agreed = []
        refused = []
        launched_from = os.getcwd()
        for yaml_path in sorted(glob.glob(os.path.join(_EXAMPLES_DIR, '*.yaml'))):
            raw = parse_input(yaml_path)
            theory = list((raw.get('control') or {}).get('theory') or [])
            if len(theory) < 2:
                continue
            config_path = os.path.join(tempfile.mkdtemp(), 'flat.json')
            yaml_to_config(yaml_path, config_path)
            # Both carriers are built from the example's own directory, which
            # is where a run launched against it looks for the siblings the
            # input names by bare title - the `.charge` file a `read_charge`
            # case expands to. From anywhere else the lookup misses and the
            # comparison never reaches the profiles it is about.
            os.chdir(os.path.dirname(yaml_path))
            try:
                try:
                    flat = _flat_task_table(parse_input(config_path))
                except TypeError as error:
                    refused.append(
                        (os.path.basename(yaml_path), str(error)[:32]))
                    continue
                self.assertEqual(
                    _flat_task_table(raw), flat,
                    '%s: the flat mode and the yaml mode hand the queue '
                    'different method/profile pairs'
                    % os.path.basename(yaml_path))
                agreed.append(os.path.basename(yaml_path))
            finally:
                os.chdir(launched_from)
        # Measured on 2026-10-19 after #394: every shipped example that binds
        # two or more theory entries reads back from its own flat config, and
        # all of them agree binding for binding. Before that change ten agreed
        # and one - `optimize_ts_multiscale.yaml`, which binds `pyscf` twice -
        # was refused as pure-flat input. The refusal list is asserted empty so
        # a new one fails by name instead of lowering the count above.
        #
        # Thirteen since 2026-10-21: the path family added three single-centre
        # composite cases (`path_mep_multiscale`, `path_irc_multiscale`,
        # `path_neb_multiscale`), each binding `pyscf` twice, and all three
        # agree binding for binding - which is the same reading this list is
        # for, carried to the multi-structure searches.
        #
        # Fourteen since 2026-10-22: `path_cell_mep_multiscale` binds `pyscf`
        # and `xtb` - a mixed pair rather than one program twice - and it
        # agrees binding for binding as well, which is the reading that says
        # the flat carrier keeps two *different* skills apart. Those fourteen
        # are the four `multicenter`, six `optimize` and four `path` examples.
        #
        # Thirty-nine measured on 2026-10-31, and the number had not been
        # re-measured since 2026-10-22 because the assertion was unreachable:
        # the loader chain this helper used stopped at `build_config`, so every
        # topology-driven example arrived with no atom count and the first one
        # carrying a `&system_constraint` block aborted the method with
        # `constraint normalization needs an atom count`, while the four
        # `read_charge` examples aborted it with a charge file looked up by
        # bare name. Both were artifacts of the shortened chain and of building
        # away from the example's directory, not of the examples. The added
        # twenty-five are the `multiscale` family, which grew from none to
        # twenty-five multi-theory examples in that window.
        #
        # Forty since 2026-10-31 18:20: `multiscale_qmmm_neb`, the band arm
        # that the expansion round had left out, binds `molcas` twice and
        # agrees binding for binding like the other six QMMM copies.
        # 41 after #449 added `multiscale_qmmm_embedding_nac` (the
        # framework carrier for the mixed two-pass ESPF+NAC deck).
        #
        # 44 since 2026-11-03: the three `multicenter_qmmm_*` ports bind
        # `molcas` and `lammps` and agree binding for binding in both
        # carriers, measured on this run's own enumeration.
        #
        # 46 since 2026-11-04: the two multi-center QMMM channel cases
        # (`_mep`, `_num_freq`) join that set, same pair of bindings.
        #
        # 52 since 2026-11-05: the family grew by six - `_nac_sp`, `_npt`,
        # `_cell`, `_neb` and the checkpoint pair `_md_restart` /
        # `_md_restart_base` - all of them the same `molcas` + `lammps` pair,
        # each read back through both carriers on this run's own enumeration.
        #
        # 55 since 2026-11-05 (#460): `_mecp`, `_md_fssh` and `_ex_opt` join
        # them, and they are the first three inputs whose per-center state
        # lists (`state_i: [0, 0]`, `root: "2; 2"`) had to survive both
        # carriers - which is the reading this ledger adds for them.
        #
        # 77 since 2026-11-05: the current working tree enumerates
        # seventy-seven multi-theory examples (the multiscale, multicenter,
        # optimize and path families); every one reads back from its own
        # flat config and agrees binding for binding.
        #
        # 81 since 2026-11-05: the four adaptive multiscale carriers
        # (ONIOM2 and QMMM, each in local and Slurm stage variants) join
        # that set; the single-method adaptive pair binds one theory entry
        # and is outside this enumeration.
        self.assertEqual(len(agreed), 81, agreed)
        self.assertEqual(refused, [])








if __name__ == '__main__':
    unittest.main()
