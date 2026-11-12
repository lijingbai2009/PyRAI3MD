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


class TestClassicInput(unittest.TestCase):
    """`&constraint` inside a real `.input` file, split then read."""

    def _parse(self, text):
        from pyrai3md.interface.parser import _parse_text
        return _parse_text(text.split('\n'))

    def _declaration(self, text, name='constraint'):
        """The section as the schema adapter reads it from the file's text."""
        from pyrai3md.schema.adapter import _constraint_declaration
        return _constraint_declaration(self._parse(text)[name])

    def test_the_flat_loop_splits_and_the_adapter_reads(self):
        """Two duties, in two files: the interface knows only lines.

        The section reaches the parser's dict as `keyword value` strings, the
        same shape every other section arrives in, and everything the ';'
        segments mean is decided one layer down.
        """
        sections = self._parse(CLASSIC_INPUT)
        self.assertEqual(sections['constraint']['shake_bond'], '1 3')
        self.assertEqual(sections['constraint']['wall_atoms'], '1-3')
        block = self._declaration(CLASSIC_INPUT)
        self.assertEqual([item['id'] for item in block['biases']], ['hold'])
        self.assertEqual([item['id'] for item in block['walls']], ['vessel'])
        self.assertEqual([item['id'] for item in block['constraints']],
                         ['ohb'])

    def test_two_segments_are_two_objects_not_a_last_one_wins(self):
        block = self._declaration(
            CLASSIC_INPUT.replace('cv_bond 1 2', 'cv_bond 1 2; 1 3'))
        self.assertEqual(len(block['cvs']), 2)
        self.assertEqual([item['atoms'] for item in block['cvs']],
                         [[1, 2], [1, 3]])

    def test_the_retired_section_name_still_reads(self):
        """`system_constraint` is the old name of the same block.

        Renaming a section an archived input already writes would turn a
        spelling change into a broken run, so both names route to one reader
        and only the new one is documented as the way to write it.
        """
        block = self._declaration(
            CLASSIC_INPUT.replace('&constraint', '&system_constraint'),
            name='system_constraint')
        self.assertEqual([item['id'] for item in block['walls']], ['vessel'])

    def test_the_section_closers_do_not_open_a_phantom_section(self):
        """`&end` used to open a section named 'end' and every key after it was
        filed there (design document section 3.2). It closes instead.
        """
        sections = self._parse(CLASSIC_INPUT)
        self.assertNotIn('end', sections)
        self.assertEqual(sorted(sections),
                         ['constraint', 'control', 'molecule'])
        trailing = self._parse(CLASSIC_INPUT + '  stray_key 1\n')
        self.assertNotIn('stray_key', trailing.get('end', {}))

    def test_a_key_without_a_value_is_skipped_not_looped(self):
        """A line with one token ends the line, it does not end the scan.

        Pinned because the flat loop is an index loop, and the early `continue`
        that a `for` loop can afford silently spins a `while` loop forever. The
        symptom was not a wrong dict but a suite that never finished, so the
        guard is written as a termination test: this returns or it hangs.
        """
        text = CLASSIC_INPUT.replace('  jobtype md\n', '  jobtype\n')
        sections = self._parse(text)
        self.assertNotIn('jobtype', sections['control'])
        self.assertEqual(sections['control']['title'], 'constraint_classic')
        self.assertIn('constraint', sections)

    def test_a_key_the_grammar_does_not_know_is_refused_not_dropped(self):
        """The nested block grammar is gone, and a leftover line says so.

        An archived `&system_constraint` block still reaches this layer as flat
        lines, so the keyword it cannot name is the answer the user needs: not
        "unknown key" but what the section reads instead.
        """
        broken = CLASSIC_INPUT.replace('  wall_type sphere\n',
                                        '  wall sphere\n')
        from pyrai3md.schema.adapter import _constraint_declaration
        with self.assertRaises(ValueError) as caught:
            _constraint_declaration(self._parse(broken)['constraint'])
        message = str(caught.exception)
        self.assertIn('wall', message)
        self.assertIn('flat', message)

    def test_the_text_reaches_the_same_plan_as_the_mapping(self):
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema import build_plan
        from pyrai3md.schema.validate import validate
        import importlib
        import tempfile
        normalize = importlib.import_module('pyrai3md.schema.normalize')

        directory = tempfile.mkdtemp(prefix='constraint_input_')
        self.addCleanup(__import__('shutil').rmtree, directory, True)
        # The classic reader takes coordinates from a same-basename .xyz, which
        # is how a real key_molecule job arrives; the constraint section is what
        # this test is about, so the geometry comes in the normal way.
        source = os.path.join(directory, 'constraint_classic.input')
        geometry = os.path.join(directory, 'constraint_classic.xyz')
        with open(geometry, 'w') as handle:
            handle.write('3\nwater\n')
            for symbol, row in zip(ATOMS, COORDS):
                handle.write('%s %12.6f %12.6f %12.6f\n'
                             % (symbol, row[0], row[1], row[2]))
        with open(source, 'w') as handle:
            handle.write(CLASSIC_INPUT)
        sections = parse_input(source)
        config = validate(build_config(sections))
        config = normalize.normalize(config)
        plan = config.system.constraints
        self.assertIsNotNone(plan)
        self.assertEqual(sorted(item.id for item in plan.biases), ['hold'])
        built = build_plan(plan.model_dump(), atoms=list(config.system.atoms),
                           natom=config.system.natom,
                           masses=list(config.system.mass))
        self.assertEqual(sorted(built.ids()), ['hold', 'ohb', 'vessel'])
        self.assertEqual(
            built.spec('ohb').diagnostics['velocity_constraint'], 'rattle')


class TestRunRecord(unittest.TestCase):
    """The log prints what the reader decided, not only what was written."""

    def _log(self, block):
        from pyrai3md.kernel.logger import RunLogger
        import shutil
        state = _state(block, COORDS)
        directory = tempfile.mkdtemp(prefix='constraint_log_')
        self.addCleanup(shutil.rmtree, directory, True)
        logger = RunLogger(state, log_dir=directory)
        logger.write_heading(state)
        path = os.path.join(directory,
                            '%s.log' % state.pmd.control.title)
        with open(path) as handle:
            return handle.read()

    @staticmethod
    def _flat(**section):
        from pyrai3md.schema.adapter import _read_constraint_section
        return _read_constraint_section(section)

    def test_the_entities_the_objects_became_are_listed_with_their_ids(self):
        """An auto-numbered id is a name the input never wrote.

        It is what every later line of the run - the energy breakdown, a
        restart's refusal - calls the action by, so it belongs in the record
        beside the object it came from.
        """
        text = self._log(self._flat(
            wall_type='sphere', wall_sphere_radius='1.5; 1.2',
            wall_center='0 0 0', wall_atoms='4', wall_fc='0.01',
            bias_bond='1 2', bias_bond_ref='0.72', bias_bond_fc='0.05',
            fix_atoms='3'))
        plan = text[text.index('&constraint plan'):text.index('&skills')]
        for line in ('wall       sphere       wall1',
                     'wall       sphere       wall2',
                     'bias       harmonic     bias1',
                     'fix        mask         fix1'):
            self.assertIn(line, plan)

    def test_a_value_the_reader_filled_in_is_printed_as_a_note(self):
        """The broadcast is visible in the run, not only inferable from the file.

        Two walls and one `wall_atoms` line is the shorthand the format allows;
        a record that only printed the input would leave the reader to work out
        that the second wall shares the first one's atoms.
        """
        text = self._log(self._flat(
            wall_type='sphere', wall_sphere_radius='1.5; 1.2',
            wall_center='0 0 0', wall_atoms='4', wall_fc='0.01'))
        notes = text[text.index('  notes:'):]
        self.assertIn("'wall_atoms' broadcast over 2 objects", notes)
        self.assertIn("'wall_center' broadcast over 2 objects", notes)

    def test_a_wall_keyword_the_switch_does_not_read_is_said_not_dropped(self):
        text = self._log(self._flat(
            wall_type='sphere', wall_sphere_radius='1.5', wall_atoms='4',
            wall_fc='0.01', wall_box_lengths='3 3 3'))
        self.assertIn("'wall_box_lengths' ignored", text)
        self.assertNotIn('box', text[text.index('&constraint plan'):
                                     text.index('notes:')])

    def test_a_wall_line_names_the_three_choices_that_reached_the_run(self):
        """Which law, what each object's center is weighted by, how many objects
        there are, and where the center came from.

        Measured on the block below: the wall with two molecules of two prints
        '2 objects, center from selection', and the one that names anchor atoms
        but no partition prints '1 object, center from wall_ref'. Before the
        center had a rule, a record could not tell a wall around its own atoms
        from a wall around the origin, which is the difference between a cage and
        an external field.
        """
        text = self._log(self._flat(
            wall_type='sphere', wall_sphere_radius='4.0; 3.0',
            wall_atoms='1-4', wall_fc='0.001', wall_target='com; com',
            wall_groups='1-2 3-4 2; none', wall_law='soft; hard',
            wall_ref='none; 1 2 3'))
        plan = text[text.index('&constraint plan'):text.index('&skills')]
        self.assertIn('wall1          4 atoms (1 2 3 4) law soft, '
                      'target com, 2 objects, center from selection', plan)
        self.assertIn('wall2          4 atoms (1 2 3 4) law hard, '
                      'target com, 1 object, center from wall_ref', plan)

    def test_a_run_with_no_constraints_prints_no_section(self):
        """DC-1 reaches the log: absence is not an empty heading."""
        text = self._log(None)
        self.assertNotIn('&constraint plan', text)


class TestCheckpoint(unittest.TestCase):
    """Save plus restore equals the continuous run, or the file is refused."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='constraint_rt_')
        self.path = os.path.join(self.directory, 'run.rt')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def _save(self, state):
        from pyrai3md.data.runtime.serializer import StateSerializer
        StateSerializer.save(state, self.path)
        return self.path

    def test_a_restart_carries_the_bound_targets_unchanged(self):
        from pyrai3md.data.runtime.serializer import StateSerializer
        from pyrai3md.workflow.constraints import runtime

        state = _state(BLOCK, COORDS)
        runtime.bind(state)
        stored_targets = dict(state.constraint_state.targets)
        self._save(state)
        restored, _index = StateSerializer.load(self.path)
        self.assertEqual(restored.constraint_state.targets, stored_targets)
        self.assertEqual(restored.constraint_plan().fingerprint,
                         state.constraint_plan().fingerprint)

    def test_a_restarted_run_does_not_retarget_its_own_drift(self):
        """The expensive direction: a resumed geometry re-read as a reference.

        The stored target must survive a resume at a moved structure; if the
        restore re-captured it, every long-run drift number in the log would be
        measured against a moving baseline.
        """
        from pyrai3md.data.runtime.serializer import StateSerializer
        from pyrai3md.workflow.constraints import runtime

        state = _state(BLOCK, COORDS)
        runtime.bind(state)
        original = dict(state.constraint_state.targets)
        self._save(state)
        moved = (np.asarray(COORDS) + 0.25).tolist()
        restored, _ = StateSerializer.load(self.path)
        restored.pmd.system.coordinates = moved
        runtime.bind(restored)
        self.assertEqual(restored.constraint_state.targets, original)

    def test_a_checkpoint_from_a_changed_block_is_refused(self):
        from pyrai3md.data.runtime.serializer import StateSerializer
        from pyrai3md.workflow.constraints import runtime

        state = _state(BLOCK, COORDS)
        runtime.bind(state)
        self._save(state)
        # Rewriting the resumed input is what a restart with an edited block
        # looks like; the file must refuse it rather than resume under other
        # physics.
        added = dict(BLOCK)
        added['walls'] = list(BLOCK['walls']) + [
            {'id': 'second', 'geometry': 'sphere', 'center': [0.0, 0.0, 0.0],
             'radius': 2.0, 'atoms': [2], 'force_constant': 0.002}]
        from pyrai3md.schema import SystemConstraintConfig
        import pickle
        with open(self.path, 'rb') as handle:
            payload = pickle.load(handle)
        payload['state'].pmd.system.constraints = SystemConstraintConfig(**added)
        with open(self.path, 'wb') as handle:
            pickle.dump(payload, handle)
        with self.assertRaises(ValueError) as caught:
            StateSerializer.load(self.path)
        self.assertIn('fingerprint', str(caught.exception))

    def test_a_file_with_a_history_but_no_block_is_refused(self):
        from pyrai3md.data.runtime.serializer import StateSerializer
        from pyrai3md.workflow.constraints import runtime
        import pickle

        state = _state(BLOCK, COORDS)
        runtime.bind(state)
        self._save(state)
        with open(self.path, 'rb') as handle:
            payload = pickle.load(handle)
        payload['state'].pmd.system.constraints = None
        with open(self.path, 'wb') as handle:
            pickle.dump(payload, handle)
        with self.assertRaises(ValueError) as caught:
            StateSerializer.load(self.path)
        self.assertIn('no constraints', str(caught.exception))

    def test_an_unconstrained_run_writes_an_empty_block(self):
        from pyrai3md.data.runtime.serializer import StateSerializer
        import pickle

        state = _state(None, COORDS)
        self._save(state)
        with open(self.path, 'rb') as handle:
            payload = pickle.load(handle)
        self.assertEqual(payload['constraint'], {})
        restored, _ = StateSerializer.load(self.path)
        self.assertTrue(restored.constraint_plan().is_empty)
        self.assertIsNone(restored.constraint_state)

    def test_a_state_holding_the_run_logger_still_writes_a_file(self):
        """The checkpoint carries state, not the process that is printing it.

        The entry point attaches a RunLogger to the live state, and that object
        owns an open file handle, so a checkpoint written from a real run would
        fail on `pickle` the moment the first cadence ticked. The handle has to
        be dropped for the payload, and the restored state comes back without a
        logger because the resuming process opens its own log file.
        """
        from pyrai3md.data.runtime.serializer import StateSerializer
        from pyrai3md.kernel.logger import RunLogger

        state = _state(BLOCK, COORDS)
        state.logger = RunLogger(state, log_dir=self.directory)
        try:
            self._save(state)
            restored, _ = StateSerializer.load(self.path)
            self.assertFalse(hasattr(restored, 'logger'))
        finally:
            state.logger.finish()


class TestThermalDegreesOfFreedom(unittest.TestCase):
    """D-06: the thermostat counts the rank, not the atoms."""

    def _dof(self, block, natom=3, pbc=None, velocities=None):
        from pyrai3md.workflow.integrator.ensemble import _dof

        state = _state(block, [list(row) for row in
                               np.asarray(COORDS)[:natom]])
        state.pmd.system.natom = natom
        state.pmd.system.atoms = list(ATOMS[:natom])
        state.pmd.system.mass = list(MASSES[:natom])
        if pbc:
            state.pmd.system.pbc = list(pbc)
        if velocities is not None:
            state.pmd.system.velocities = [list(row) for row in velocities]
        else:
            state.pmd.system.velocities = [
                [0.0, 0.0, 0.0] for _ in range(natom)]
        from pyrai3md.workflow.constraints import runtime
        runtime.bind(state)
        return _dof(state)

    def test_a_bond_constraint_costs_one_degree_not_one_atom(self):
        mobile = self._dof(None)
        frozen = self._dof({'fixes': [{'id': 'f', 'atoms': [3]}]})
        bonded = self._dof({'constraints': [{'id': 'x', 'bonds': [[1, 2]]}],
                            'units': {'length': 'angstrom'}})
        self.assertEqual(mobile, 3 * 3 - 3)
        self.assertEqual(frozen, mobile - 3)
        self.assertEqual(bonded, mobile - 1)

    def test_two_bonds_on_a_shared_atom_cost_two(self):
        """Each independent row removes one degree, atom sharing or not."""
        block = {'constraints': [{'id': 'x', 'bonds': [[1, 2], [2, 3]], 'target': 1.8}],
                 'units': {'length': 'angstrom'}}
        self.assertEqual(self._dof(block), 3 * 3 - 3 - 2)

    def test_a_three_bond_chain_on_four_atoms_removes_three(self):
        block = {'constraints': [{'id': 'x',
                                  'bonds': [[1, 2], [2, 3], [3, 4]], 'target': 1.8}],
                 'units': {'length': 'angstrom'}}
        self.assertEqual(self._dof(block, natom=4), 3 * 4 - 3 - 3)

    def test_a_bond_between_two_frozen_atoms_adds_no_condition(self):
        """The redundancy case the rank arithmetic has to get right.

        Counting it as a condition would remove a degree of freedom the frozen
        pair already cannot use, and a run would thermostat against a number
        that does not describe its own phase space.
        """
        block = {'fixes': [{'id': 'f', 'atoms': [1, 2], 'direction': 'xyz'}],
                 'constraints': [{'id': 'x', 'bonds': [[1, 2]], 'target': 1.8}],
                 'units': {'length': 'angstrom'}}
        # Four atoms: six components frozen, the bond already satisfied by the
        # anchors, and the center-of-mass convention subtracts three.
        self.assertEqual(self._dof(block, natom=4), 6 - 3)

    def test_the_same_pair_with_two_targets_is_refused_not_ranked(self):
        from pyrai3md.schema import ConstraintInputError

        block = {'constraints': [
            {'id': 'x', 'bonds': [[1, 2]],
             'target': 1.8},
            {'id': 'y', 'bonds': [[1, 2]],
             'target': 1.6}],
            'units': {'length': 'angstrom'}}
        with self.assertRaises(ConstraintInputError):
            self._dof(block)

    def test_a_system_with_no_free_degree_is_refused_not_clamped(self):
        block = {'fixes': [{'id': 'f', 'atoms': [1, 2, 3],
                            'direction': 'xyz'}]}
        with self.assertRaises(ValueError) as caught:
            self._dof(block)
        self.assertIn('degrees of freedom', str(caught.exception))

    def test_the_unconstrained_path_still_uses_the_classic_formula(self):
        """DC-1 for the ensemble line: no plan, no change in its counting."""
        self.assertEqual(self._dof(None), 3 * 3 - 3)


class TestOneRestorePathOnly(unittest.TestCase):
    """D-16 closed by deletion: the restart guard lives in the serializer."""

    def test_the_runtime_module_has_no_restore_entry(self):
        from pyrai3md.workflow.constraints import runtime
        self.assertFalse(
            hasattr(runtime, 'restore'),
            'constraints/runtime.restore was dead code reading payload keys the '
            'writer never produces; the live refusal is '
            'serializer._check_constraint_payload, which '
            'test_a_checkpoint_from_a_changed_block_is_refused pins against a '
            'real checkpoint')


ATOMS = ['O', 'H', 'H', 'C']
MASSES = [15.999, 1.008, 1.008, 12.011]


class TestConstraintOwnership(unittest.TestCase):
    """D-27: one driver, one constraint engine, said out loud.

    The collision was real and measured: a run whose profile let the engine
    integrate and whose input also declared framework constraints held two
    different bond lengths at once (the engine's TIP3P 0.9572 A against the
    framework's bound 0.9815 A) with nothing to tell the reader. The rule the
    user gave is that the driver decides, in both directions, and the framework
    states whose engine is in force.
    """

    BLOCK = {'constraints': [{'id': 'oh', 'kind': 'bond',
                              'bonds': [[1, 2]],
                              'tolerance': 1.0e-10}]}

    def test_the_framework_clears_the_skill_side_keywords_it_owns(self):
        """`validate_shake` on driver='main': inert, and announced."""
        import warnings as warning_module

        from pyrai3md.skills.lammps_skill.schema.validate import validate_shake
        with warning_module.catch_warnings(record=True) as caught:
            warning_module.simplefilter('always')
            data = validate_shake({'lammps_shake': 1,
                                   'shake_fix_id': 'fShakeTIP3P',
                                   'driver': 'main'})
        self.assertEqual(int(data['lammps_shake']), 0)
        self.assertEqual(str(data['shake_fix_id']), '')
        self.assertEqual(str(data['shake_keywords']), '')
        messages = [str(row.message) for row in caught]
        self.assertTrue(any("control.driver='main'" in message
                            and 'framework' in message
                            for message in messages), messages)

    def test_the_skill_side_request_survives_when_the_skill_drives(self):
        """The opposite control: under driver='skill' the same input is the
        skill's own business, so nothing is cleared and nothing is refused here
        (the profile gate below it still applies)."""
        from pyrai3md.skills.lammps_skill.schema.validate import validate_shake
        data = validate_shake({'lammps_shake': 1,
                               'shake_fix_id': 'fShakeTIP3P',
                               'profile': 'ff_nvt',
                               'driver': 'skill'})
        self.assertEqual(int(data['lammps_shake']), 1)
        self.assertEqual(str(data['shake_fix_id']), 'fShakeTIP3P')

    def test_a_framework_block_is_inert_when_the_skill_owns_the_jobtype(self):
        """driver != 'main' hands the whole jobtype to the skill, so the
        framework plan has to be empty - and the input must not be allowed to
        believe its constraint is being applied."""
        import warnings as warning_module

        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        state = _state(block=self.BLOCK)
        state.pmd.control.driver = 'skill'
        state._constraint_plan = None
        with warning_module.catch_warnings(record=True) as caught:
            warning_module.simplefilter('always')
            plan = state.constraint_plan()
        self.assertTrue(plan.is_empty)
        messages = [str(row.message) for row in caught]
        self.assertTrue(any('inert' in message and 'skill' in message
                            for message in messages), messages)

    def test_the_run_log_names_the_engine_in_force(self):
        """The debug line the ruling asked for: a reader of the log learns the
        ownership before reading a single constraint row."""
        from pyrai3md.kernel.logger import RunLogger
        import shutil
        state = _state(block=self.BLOCK)
        # RunLogger opens <title>.log at construction; without a directory it
        # lands in the launch directory (#306).
        directory = tempfile.mkdtemp(prefix='constraint_log_')
        self.addCleanup(shutil.rmtree, directory, True)
        logger = RunLogger(state, log_dir=directory)
        written = []
        logger.write = written.append
        logger._write_constraint_plan(state)
        text = ''.join(written)
        self.assertIn("ownership          framework (control.driver='main')",
                      text)
        self.assertIn('constraint shake', text)




class TestRestartFrameCadence(unittest.TestCase):
    """The frame a segmented run writes is decided by the absolute step, not by
    which process happened to reach it (#281, D-19 and D-24).

    A restart has one piece of information the continuous run does not: it is
    not the end of anything. The old loop carried a per-process step count, so
    the resumed process wrote a label its predecessor had already written and
    the two products disagreed on both the numbers and the frames. The step
    numbering is absolute now; what still disagreed was the frame LIST, because
    the last step of every process forces a record. The rule that closes that
    gap is that a step which leaves a checkpoint leaves a frame too - the
    checkpoint cadence is the one thing both arms carry identically, and it is
    the cadence that decides where a run may be cut.
    """

    class _Spy:
        """Records the (step, forced) pairs the writer is asked about."""

        def __init__(self):
            self.calls = []

        def record(self, step, snapshot, state, final=False):
            self.calls.append((step, bool(final)))
            return None

    @staticmethod
    def _state(step, checkpoint_every):
        import types
        state = types.SimpleNamespace()
        state.runtime = types.SimpleNamespace(current_step=step, workdir='')
        state.pmd = types.SimpleNamespace(
            control=types.SimpleNamespace(title='cadence'),
            dynamics=types.SimpleNamespace(checkpoint=checkpoint_every))
        state.get_history_step = lambda idx, copy=False: None
        state.record_step_result = lambda: None
        return state

    def _checkpoint(self, step, checkpoint_every, total=8):
        from pyrai3md.workflow.md import AIMD
        workflow = AIMD(maxiter=total)
        workflow._total_steps = total
        workflow.output = self._Spy()
        # The serialization itself is another test's business; here only the
        # predicate it shares with the record gate matters.
        workflow._write_runtime_file = lambda state, step, final: None
        workflow.checkpoint(self._state(step, checkpoint_every))
        return workflow.output.calls

    def test_a_checkpoint_step_is_recorded_even_mid_run(self):
        """Step 4 of an eight-step run is not its last step, but it is where a
        segment ends, so it must be a frame in both arms."""
        self.assertEqual(self._checkpoint(4, 2), [(4, True)])

    def test_a_step_between_checkpoints_is_still_left_to_the_writer(self):
        """Default-closed on the frame gate: a step that is neither the run's
        end nor a cut point asks for nothing, so the sparse cadence of an
        ordinary run is unchanged by this rule. Two cadences, because "every
        step" and "every third step" both answer True to a single sample."""
        self.assertEqual(self._checkpoint(5, 2), [(5, False)])
        self.assertEqual(self._checkpoint(4, 3), [(4, False)])
        self.assertEqual(self._checkpoint(6, 3), [(6, True)])

    def test_a_run_with_no_checkpoint_cadence_gains_no_frames(self):
        """checkpoint: 0 writes no file at all, so it forces no frame either -
        only the true end does. A run that never checkpoints therefore records
        exactly what it recorded before this rule existed."""
        self.assertEqual(self._checkpoint(4, 0), [(4, False)])
        self.assertEqual(self._checkpoint(8, 0), [(8, True)])

    def test_the_frame_gate_and_the_file_use_one_predicate(self):
        """The two must not be able to drift: the file is the cut point and the
        frame is its witness. Read the written paths, not the predicate, so a
        change that makes one of them ask a different question shows up."""
        import tempfile
        from unittest import mock

        from pyrai3md.workflow.md import AIMD
        workflow = AIMD(maxiter=8)
        with tempfile.TemporaryDirectory() as workdir:
            with mock.patch('pyrai3md.data.runtime.serializer.save_state') \
                    as save:
                for step in range(1, 9):
                    for every in (0, 2, 3):
                        state = self._state(step, every)
                        state.runtime.workdir = workdir
                        written = workflow._write_runtime_file(
                            state, step, step == 8)
                        self.assertEqual(bool(written),
                                        workflow._runtime_file_due(
                                            state, step, step == 8),
                                        'step %d, cadence %d' % (step, every))
                        self.assertEqual(self._checkpoint(step, every,
                                                          total=8)[0][1],
                                         bool(written) or step == 8)


class TestConvergenceGradientCriterion(unittest.TestCase):
    """The gradient half of an optimization's convergence test, read on the
    directions the run can actually move (#282 D-26, replaced by the tangential
    criterion of this round).

    A constrained geometry's stationary point is not a zero gradient: the force
    a held condition absorbs sits in the gradient the engine returns, so a
    threshold read along that condition is unreachable. The first answer to this
    removed every atom a condition named from the count, which also threw away
    the *tangential* half of those atoms: a run could then be called converged
    over a large sideways force on a held bond. The split is directional instead,
    and the removed part is reported as the condition's multiplier.
    """

    # The bond is atoms 1-2 of the fixture, which lie along x, so an x
    # component on those two atoms is the condition's own direction and a y
    # component is the direction the run is free to move in.
    BLOCK = {'constraints': [{'id': 'pair01', 'kind': 'bond',
                              'bonds': [[1, 2]],
                              'tolerance': 1.0e-10}],
             'fixes': [{'id': 'frozen', 'atoms': [3], 'direction': 'xyz'}]}

    def _report(self, block, gradient):
        from pyrai3md.workflow.constraints import runtime
        from pyrai3md.workflow.optimizer import MinimaOptimizer
        state = _state(block, COORDS)
        optimizer = MinimaOptimizer(gradient_threshold=1.0e-3,
                                    gradient_max_threshold=1.0e-3,
                                    displacement_threshold=1.0e-3,
                                    displacement_max_threshold=1.0e-3)
        runtime.bind(state)
        state.set_result({'energy': -100.0, 'gradient': gradient})
        # The workflow's own order: the split is read at the geometry the
        # gradient belongs to, before the optimizer has moved the atoms.
        optimizer.note_evaluation(state)
        return optimizer.convergence(state)

    @staticmethod
    def _sideways():
        """A large force on the held pair, perpendicular to the bond, plus the
        frozen atom's own force and a quiet free atom."""
        return np.array([[0.0, 5.0, 0.0],
                         [0.0, -5.0, 0.0],
                         [3.0, 3.0, 3.0],
                         [1.0e-6, -1.0e-6, 1.0e-6]])

    @staticmethod
    def _along_the_bond():
        """The pair pulled apart along the bond: exactly what the condition
        holds, so none of it is the run's to answer."""
        return np.array([[5.0, 0.0, 0.0],
                         [-5.0, 0.0, 0.0],
                         [3.0, 3.0, 3.0],
                         [1.0e-6, -1.0e-6, 1.0e-6]])

    def test_a_sideways_force_on_a_held_atom_still_counts(self):
        """The reading the old mask got wrong, and it got it by declaring a run
        converged over 5 Hartree per Bohr of force the step can answer."""
        converged, report = self._report(self.BLOCK, self._sideways())
        self.assertEqual(report['Gradient Max'], 5.0)
        self.assertFalse(converged)

    def test_the_condition_absorbs_its_own_force_and_reports_the_multiplier(self):
        converged, report = self._report(self.BLOCK, self._along_the_bond())
        # What is left to judge is the one free atom's own quiet force.
        self.assertAlmostEqual(report['Gradient Max'], 1.0e-6, places=12)
        self.assertTrue(converged)
        rows = report['Constraint']['conditions']
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(abs(rows[0]['multiplier']), 5.0, places=9)
        # A held length is measured at the geometry the gradient was computed
        # on, so the residual is the solve's reading of the previous step.
        self.assertLess(abs(rows[0]['residual']), 1.0e-8)
        self.assertEqual(rows[0]['atoms'], [1, 2],
                         'the row is numbered the way the input numbered it')

    def test_a_frozen_atom_stays_out_of_the_criteria(self):
        """A frozen component is not a degree of freedom at all, so it carries
        no force the run could answer: the third atom's 3 Hartree per Bohr is
        simply not asked."""
        converged, report = self._report(self.BLOCK, self._along_the_bond())
        self.assertTrue(converged)
        self.assertEqual(report['Constraint']['frozen_components'], 3)
        self.assertEqual(report['Constraint']['mobile_components'], 9)

    def test_a_run_with_nothing_free_reports_a_zero_gradient(self):
        """Everything frozen leaves nothing to ask, and an empty selection must
        read as satisfied rather than as nan - a nan would make the criterion
        unmeetable in exactly the case it stops applying."""
        block = {'fixes': [{'id': 'all', 'atoms': [1, 2, 3, 4],
                         'direction': 'xyz'}]}
        converged, report = self._report(block, self._sideways())
        self.assertEqual(report['Gradient Max'], 0.0)
        self.assertEqual(report['Gradient RMS'], 0.0)
        self.assertTrue(converged)

    def test_the_same_gradient_is_not_converged_without_the_declarations(self):
        """The negative control, and it is the same arithmetic: with no plan the
        report reads every component, which is what an unconstrained run always
        did."""
        converged, report = self._report(None, self._sideways())
        self.assertEqual(report['Gradient Max'], 5.0)
        self.assertGreater(report['Gradient RMS'], 1.0)
        self.assertFalse(converged)
class TestConstraintEnergyBlock(unittest.TestCase):
    """The record of a constrained run names both halves of its potential
    (#283, D-18).

    `Epot` in the trajectory is the sum the integrator conserves - the engine's
    own potential plus every declared constraint potential - and the ruling is
    that the output has to show the addition, with each term under the name the
    input gave the constraint. Without the origin row a reader can only take the
    total on trust.
    """

    def _results(self, block):
        from pyrai3md.qualification.verification._constraint_helpers \
            import _run
        state, result, _backend = _run(block, COORDS)
        return result

    def test_the_origin_pot_travel_with_the_sum(self):
        """Read the two numbers the finalizer writes and check the addition on
        them, not on the printed page. The fixture block has a bias and a wall,
        so the sum has two rows under it."""
        result = self._results(BLOCK)
        contributions = result['energy_contributions']
        origin = float(np.asarray(result['energy_origin'],
                                  dtype=np.float64).reshape(-1)[0])
        total = float(np.asarray(result['energy'], dtype=np.float64).reshape(-1)[0])
        self.assertAlmostEqual(
            origin + sum(contributions.values()), total, places=12)
        self.assertIn('constraint.bias.stretch', contributions)
        self.assertIn('constraint.wall.vessel', contributions)

    def test_an_unconstrained_run_records_no_origin_row(self):
        """Default-closed: nothing contributed, so there is no decomposition to
        keep and no new key appears in the result."""
        self.assertNotIn('energy_origin', self._results(None))

    def test_the_block_prints_the_id_and_both_ends_of_the_sum(self):
        """The reader's view: origin pot, one line per constraint under the name
        the input wrote, and the Epot the trajectory column carries."""
        from pyrai3md.utilities.print_results import _format_constraint_energy
        text = _format_constraint_energy(
            [-74.9631292844484278], [-74.9631631182905238],
            {'constraint.bias.hold_oh': 0.0000338338420919})
        self.assertIn('&constraint energies in Hartree', text)
        self.assertIn('origin pot', text)
        self.assertIn('hold_oh', text)
        self.assertIn('Epot', text)
        # The flat key is storage, not a name to print.
        self.assertNotIn('constraint.bias.', text)

    def test_a_run_with_nothing_to_decompose_prints_no_block(self):
        """Two sides of the same rule. No origin row - an unconstrained run -
        means no section at all. An origin row with no constraint under it means
        a section whose two numbers are equal, which is the answer a purely
        geometric hold asks for."""
        from pyrai3md.utilities.print_results import _format_constraint_energy
        self.assertEqual(_format_constraint_energy([-1.0], None, {}), '')
        text = _format_constraint_energy(
            [-1.0], [-1.0], {'multiscale.low': 0.1})
        self.assertIn('origin pot', text)
        self.assertNotIn('multiscale', text)
        self.assertEqual(text.count('-1.0000000000000000'), 2)


if __name__ == '__main__':
    unittest.main()


class TestBarostatOrdering(unittest.TestCase):
    """The fourth seam: the geometry the step records is the one the barostat
    left behind, not the one the solver produced (#276, D-25).

    A Berendsen rescale multiplies coordinates and cell by the same tensor, so
    it is not a rigid motion: a bond the projection has just fixed comes back a
    fraction long, and a frozen atom comes back displaced. The loop therefore
    asks the geometry question again after the ensemble's post half step.
    """

    RESCALE = 1.001

    BLOCK = {'constraints': [{'id': 'pair12', 'kind': 'bond',
                              'bonds': [[1, 2]],
                              'tolerance': 1.0e-10}],
             'fixes': [{'id': 'frozen', 'atoms': [3], 'direction': 'xyz'}]}

    @staticmethod
    def _bond(state):
        rows = np.asarray(state.coordinates, dtype=np.float64)
        return float(np.linalg.norm(rows[0] - rows[1]))

    @staticmethod
    def _scaled(state):
        rows = np.asarray(state.coordinates, dtype=np.float64)
        state.update_system(
            {'coordinates': [list(row) for row in rows * TestBarostatOrdering.RESCALE]})

    def test_a_cell_rescale_reaches_the_recorded_geometry_off_the_manifold(self):
        """The negative control first: the rescale alone really does break both
        declarations, so the second projection has something to fix."""
        from pyrai3md.workflow.integrator import operations
        state = operations.apply_constrain(_state(block=self.BLOCK))
        held = self._bond(state)
        anchor = np.asarray(state.coordinates, dtype=np.float64)[2]
        self._scaled(state)
        self.assertAlmostEqual(self._bond(state) / held, self.RESCALE, places=12)
        # An affine rescale displaces a point by its own distance from the
        # origin times the scale, which is the number to pin rather than a
        # threshold picked to make the assertion pass.
        moved = np.linalg.norm(
            np.asarray(state.coordinates, dtype=np.float64)[2] - anchor)
        self.assertAlmostEqual(moved,
                               float(np.linalg.norm(anchor)) * (self.RESCALE - 1.0),
                               places=12)

    def test_the_second_projection_restores_both_declarations(self):
        """After the rescale the same entry brings the bond back to its target
        and the frozen atom exactly home; the pair together is why the loop's
        second call is one call and not a bond-only correction."""
        from pyrai3md.workflow.integrator import operations
        state = operations.apply_constrain(_state(block=self.BLOCK))
        held = self._bond(state)
        anchor = np.asarray(state.coordinates, dtype=np.float64)[2]
        self._scaled(state)
        state = operations.apply_constrain(state)
        self.assertLessEqual(abs(self._bond(state) - held), 1.0e-9)
        self.assertLessEqual(
            np.linalg.norm(np.asarray(state.coordinates,
                                      dtype=np.float64)[2] - anchor), 1.0e-12)

    def test_the_loop_projects_after_the_barostat_half_step(self):
        """The fix is a position in a call sequence, and no numeric test can see
        it: if someone moves the block back above the ensemble's post half step
        the arithmetic stays perfect while the recorded geometry is one
        rescale away from it. So the order itself is pinned."""
        import inspect

        from pyrai3md.workflow.md import AIMD
        source = inspect.getsource(AIMD._run_trajectory)
        pre = source.index("'ensemble_pre'")
        project = source.index("'apply_constrain'")
        post = source.index("'ensemble_post'")
        again = source.index("'constrain_after_cell'")
        history = source.index("'update_history'")
        self.assertTrue(pre < project < post < again < history)

    def test_an_unconstrained_step_pays_nothing_for_the_second_projection(self):
        """Default-closed still holds at the new seam: with no plan the entry
        returns the very same object, so the added call is free for every run
        that declares no constraint."""
        from pyrai3md.workflow.integrator import operations
        state = _state(block=None)
        self.assertIs(operations.apply_constrain(state), state)
