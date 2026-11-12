"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the schema pipeline: what the skill accepts, what it
rejects, and the geometry and state bookkeeping it derives.

These tests need no TensorFlow.

"""

import unittest

from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
from pyrai3md.skills.nn_pes_skill.schema.validate import validate

ATOMS = ['C', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def config(**over):
    """Build, normalize and validate a config with the given overrides.

    Parameters
    ----------
    over : dict
        Fields to set beyond the minimal serving config.

    Returns
    -------
    SkillConfig
        Validated configuration.
    """
    raw = {
        'profile': 'mlp_eg_gradient', 'method': 'mlp_eg',
        'modeldir': '/tmp/does-not-need-to-exist', 'mult_nn': 2,
        'nstate': 2, 'atoms': ATOMS, 'coordinates': COORDS,
        'scratch_path': '', 'target_states': [0, 1],
    }
    raw.update(over)
    return validate(normalize(build_config(raw)))


class TestGeometry(unittest.TestCase):

    def test_atom_count_derived(self):
        self.assertEqual(config().system.natom, 3)

    def test_atoms_and_coordinates_must_match(self):
        with self.assertRaises(ValueError):
            config(atoms=['C', 'H'])

    def test_single_atom_is_refused(self):
        # One atom has no pair, so the geometric features would be empty
        with self.assertRaises(ValueError):
            config(atoms=['C'], coordinates=[[0.0, 0.0, 0.0]])

    def test_bad_coordinate_shape(self):
        with self.assertRaises(ValueError):
            config(coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


class TestStateBookkeeping(unittest.TestCase):

    def test_pair_map_defaults_to_adjacent_states(self):
        self.assertEqual(config().system.nac_pair_map, [[0, 1]])

    def test_declared_pair_map_is_kept(self):
        out = config(nac_pair_map=[[0, 2], [1, 2], [0, 1]])
        self.assertEqual(out.system.nac_pair_map, [[0, 2], [1, 2], [0, 1]])
        self.assertEqual(out.system.nnac, 3)

    def test_target_states_out_of_range_are_refused(self):
        with self.assertRaises(ValueError):
            config(target_states=[0, 5])

    def test_empty_target_states_use_every_state(self):
        out = config(target_states=[])
        self.assertEqual(out.calculation.target_states, [0, 1])

    def test_spin_count_must_match_states(self):
        with self.assertRaises(ValueError):
            config(state_spin=[0, 0, 0])

    def test_state_labels_are_generated(self):
        self.assertEqual(config(state_spin=[0, 2]).system.state_label,
                         ['s', 't'])


class TestProfileValidation(unittest.TestCase):

    def test_method_and_profile_must_agree(self):
        with self.assertRaises(ValueError):
            config(profile='mlp_eg_gradient', method='mlp_nac2')

    def test_serving_needs_a_model_folder(self):
        with self.assertRaises(ValueError):
            config(modeldir='')

    def test_training_needs_data(self):
        with self.assertRaises(ValueError):
            config(profile='mlp_eg_train', method='mlp_eg', train_data='')

    def test_periodic_runs_are_refused(self):
        with self.assertRaises(ValueError):
            config(pbc=[True, False, False])

    def test_the_unit_convention_is_not_a_keyword(self):
        # A file is read as Hartree energies and Hartree/Bohr derivatives
        # and nothing else, so there is no scheme to declare and no way to
        # declare one that disagrees with the data.
        from pyrai3md.skills.nn_pes_skill.schema.config import DEFAULT
        self.assertNotIn('eg_unit', DEFAULT)
        self.assertNotIn('nac_unit', DEFAULT)

    def test_bad_train_numbers_are_refused(self):
        base = dict(profile='mlp_eg_train', method='mlp_eg',
                    train_data='/tmp/data.json')
        for bad in ({'epo': 0}, {'val_split': 1.5}, {'depth': 0},
                    {'nn_size': 0}, {'batch_size': -8}):
            with self.assertRaises(ValueError):
                config(eg=bad, **base)

    def test_bad_train_mode_is_refused(self):
        with self.assertRaises(ValueError):
            config(profile='mlp_eg_train', method='mlp_eg',
                   train_data='/tmp/data.json', train_mode='finetune')

    def test_training_needs_a_model_folder(self):
        # A fit writes one folder per ensemble, so it has to know where
        with self.assertRaises(ValueError):
            config(profile='mlp_eg_train', method='mlp_eg',
                   train_data='/tmp/data.json', modeldir='')

    def test_training_reads_the_role_section_of_its_ensemble(self):
        base = dict(profile='mlp_eg_train', method='mlp_eg',
                    train_data='/tmp/data.json')
        out = config(eg={'epo': '30;20', 'depth': 3}, **base)
        self.assertEqual(out.calculation.eg, {'epo': '30;20', 'depth': 3})
        # An unknown keyword is refused before a folder is made, but the
        # section a profile never trains is not read at all.
        with self.assertRaises(ValueError):
            config(eg={'nonsense': 1}, **base)
        config(nac={'nonsense': 1}, **base)

    def test_member_groups_follow_the_ensemble_size(self):
        base = dict(profile='mlp_eg_train', method='mlp_eg',
                    train_data='/tmp/data.json', mult_nn=2)
        # Surplus groups are ignored: mult_nn decides how many are read.
        config(eg={'epo': '30;20;10'}, **base)
        # A single group broadcasts to every member.
        config(eg={'epo': '30'}, **base)
        # Writing for some but not all members is the mistake.
        three = dict(base, mult_nn=3)
        with self.assertRaises(ValueError):
            config(eg={'epo': '30;20'}, **three)

    def test_ensemble_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            config(mult_nn=0)


class TestEnvironment(unittest.TestCase):

    def test_execution_mode_is_checked(self):
        with self.assertRaises(ValueError):
            config(execution_mode='lsf')

    def test_a_thread_count_above_this_machine_is_accepted(self):
        import os
        # #306: the machine a run lands on is not the machine that parsed
        # the input, so the skill keeps the number it is given and the
        # framework compares the demand with the CPUs it can actually
        # use (workflow/dispatch/scheduler.py) and warns.
        requested = os.cpu_count() + 1
        self.assertEqual(config(thread=requested).environment.thread,
                         requested)

    def test_the_retired_thread_keyword_is_refused(self):
        with self.assertRaises(ValueError):
            config(num_thread=4)

    def test_project_name_reaches_the_calculation_section(self):
        # The classic input writes nn_project; the adapter copies it into
        # Calculation.project, which names the output files.
        out = config(nn_project='MYRUN')
        self.assertEqual(out.calculation.project, 'MYRUN')


if __name__ == '__main__':
    unittest.main()
