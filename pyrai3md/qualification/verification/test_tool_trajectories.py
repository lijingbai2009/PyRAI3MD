"""
======================================================================
Guard for the migrated trajectory tools.

The analyzer's PyRAI3MD reader must take labels, populations and hops
from the framework records (not a log block), and the generator must
write a PyRAI3MD YAML plus `<title>.init.xyz`. These checks use inline
synthetic trajectories and a shipped frequency fixture.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / 'tools'
EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS
                                                  / ('%s.py' % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trajectory(folder, project='demo', hop=True):
    os.makedirs(folder, exist_ok=True)
    second = '%s coord 2 state 1%s' % (
        project, ' to 2 CI' if hop else '')
    with open(os.path.join(folder, '%s.md.xyz' % project), 'w') as handle:
        for comment, z in ((('%s coord 1 state 1' % project), 0.74),
                           (second, 0.75)):
            handle.write('2\n%s\n' % comment)
            handle.write('%-5s%16.8f%16.8f%16.8f\n' % ('H', 0.0, 0.0, 0.0))
            handle.write('%-5s%16.8f%16.8f%16.8f\n' % ('H', 0.0, 0.0, z))
    head = '%20s%28s%28s%28s%28s\n' % ('time', 'Epot', 'Ekin', 'Etot',
                                       'Epot1,2,3...')
    with open(os.path.join(folder, '%s.md.energies' % project), 'w') as handle:
        handle.write(head)
        handle.write('%20.2f%28.16f%28.16f%28.16f%28.16f%28.16f\n'
                     % (0.0, -1.0, 0.1, -0.9, -1.0, -0.9))
        handle.write('%20.2f%28.16f%28.16f%28.16f%28.16f%28.16f\n'
                     % (0.5, -1.01, 0.12, -0.89, -1.01, -0.89))
    pop_head = '%20s%28s%28s\n' % ('time', 'Pop1,2,3...', 'Act1,2,3...')
    with open(os.path.join(folder, '%s.md.pop' % project), 'w') as handle:
        handle.write(pop_head)
        handle.write('%20.2f%28.16f%28.16f%28.16f%28.16f\n'
                     % (0.0, 1.0, 0.0, 1.0, 0.0))
        handle.write('%20.2f%28.16f%28.16f%28.16f%28.16f\n'
                     % (0.5, 0.0, 1.0, 0.0, 1.0))
    if hop:
        with open(os.path.join(folder, '%s.sh.xyz' % project), 'w') as handle:
            handle.write('2\n%s coord 2 state 1 to 2 CI\n' % project)
            handle.write('H 0.0 0.0 0.0\nH 0.0 0.0 0.75\n')
        with open(os.path.join(folder, '%s.sh.energies' % project),
                  'w') as handle:
            handle.write(head)
            handle.write('%20.2f%28.16f%28.16f%28.16f%28.16f%28.16f\n'
                         % (0.5, -1.01, 0.12, -0.89, -1.01, -0.89))


class TestTrajectoryTools(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix='tools_traj_')
        self.addCleanup(shutil.rmtree, self.work)
        self._old_cwd = os.getcwd()
        os.chdir(self.work)
        if str(TOOLS) not in sys.path:
            sys.path.insert(0, str(TOOLS))

    def tearDown(self):
        os.chdir(self._old_cwd)

    def test_analyzer_reader_uses_framework_records(self):
        analyzer = _load('traj_analyzer')
        folder = os.path.join(self.work, 'demo_1')
        _write_trajectory(folder)
        result = analyzer.read_raw_data_pyrai3md(
            [0, folder, 0, [], [], []])
        (ntraj, natom, nstate, nstep, dtime, kin, pot, pop,
         label, coord, hop, state) = result
        self.assertEqual((natom, nstate, nstep), (2, 2, 2))
        self.assertAlmostEqual(dtime, 0.5)
        self.assertEqual(hop, [2])
        self.assertIn('to 1 CI', label[1])
        self.assertEqual(pop[1], [0.0, 1.0])
        self.assertEqual(state, 1)

    def test_analyzer_snapshot_reader(self):
        analyzer = _load('traj_analyzer')
        folder = os.path.join(self.work, 'demo_1')
        _write_trajectory(folder)
        (ntraj, natom, nstate, init, final, hop, init_t, final_t,
         hop_t, init_p, final_p, hop_p) = analyzer.read_pyrai3md(
            [0, folder, 0, [], [], []])
        self.assertEqual((natom, nstate), (2, 2))
        self.assertEqual(init_t, [1])
        self.assertEqual(final_t, [2])
        self.assertEqual(hop_t, [2])
        self.assertAlmostEqual(init_p[0][0], -1.0)

    def test_analyzer_snapshot_reader_without_hops(self):
        analyzer = _load('traj_analyzer')
        folder = os.path.join(self.work, 'demo_1')
        _write_trajectory(folder, hop=False)
        result = analyzer.read_pyrai3md([0, folder, 0, [], [], []])
        self.assertEqual(result[5], [])
        self.assertEqual(result[8], [])

    def test_analyzer_refuses_partitioned_comments(self):
        analyzer = _load('traj_analyzer')
        folder = os.path.join(self.work, 'demo_1')
        _write_trajectory(folder, hop=False)
        path = os.path.join(folder, 'demo.md.xyz')
        with open(path) as handle:
            lines = handle.read().splitlines()
        lines[1] = 'demo coord 1 states 1 2'
        with open(path, 'w') as handle:
            handle.write('\n'.join(lines) + '\n')
        with self.assertRaises(ValueError):
            analyzer.read_raw_data_pyrai3md([0, folder, 0, [], [], []])

    def test_generator_sampler_and_starting_frame(self):
        generator = _load('traj_generator')
        shutil.copy2(EXAMPLES / 'path_irc.freq.molden',
                     os.path.join(self.work, 'hcn.freq.molden'))
        ensemble = generator._sampling_pyrai3md(
            'hcn', 1, 1, 300.0, 'wigner', 'molden')
        self.assertEqual(len(ensemble), 1)
        self.assertEqual(set(ensemble[0][:, 0]), {'H', 'C', 'N'})
        self.assertTrue(os.path.exists(
            os.path.join(self.work, 'wigner-hcn-300.0.xyz')))

        template = {
            'control': {'title': 'hcn', 'jobtype': 'md',
                        'theory': ['pyscf']},
            'atoms': ['H', 'C', 'N'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2],
                            [0.0, 0.0, 0.1]],
            'molecule': {'charges': 0, 'multiplicity': 1,
                         'ci': [1], 'spin': [0]},
            'md': {'initcond': 'wigner', 'ninitcond': 1},
            'pyscf': {'method': 'hf', 'basis': 'sto3g'},
        }
        mol = np.zeros((3, 9), dtype=object)
        mol[:, 0] = ['H', 'C', 'N']
        mol[:, 1:4] = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2],
                       [0.0, 0.0, 0.1]]
        mol[:, 4:7] = 0.0
        folder = os.path.join(self.work, 'hcn-1')
        generator.pyrai3md([
            'hcn', 'hcn-1', folder, 1, 1, '1:00:00', 'normal', 100,
            template, mol, ''])
        self.assertTrue(os.path.exists(
            os.path.join(folder, 'hcn-1.yaml')))
        self.assertTrue(os.path.exists(
            os.path.join(folder, 'hcn-1.init.xyz')))
        self.assertTrue(os.path.exists(
            os.path.join(folder, 'run_pyrai3md.sh')))
        import yaml
        with open(os.path.join(folder, 'hcn-1.yaml')) as handle:
            payload = yaml.safe_load(handle)
        self.assertEqual(payload['control']['title'], 'hcn-1')
        self.assertEqual(payload['md']['initcond'], 'read')
        self.assertEqual(payload['md']['ninitcond'], 1)
        with open(os.path.join(
                folder, 'hcn-1.init.xyz')) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], '3')
        self.assertEqual([line.split()[0] for line in lines[2:5]],
                         ['H', 'C', 'N'])

    def test_multiscale_generator_uses_framework_reference(self):
        multiscale = _load('multiscale_generator')
        self.assertGreater(multiscale._get_mass('H'), 0.0)
        self.assertGreater(multiscale._get_radii('H'), 0.0)
        self.assertTrue(callable(multiscale._sampling_pyrai3md))


if __name__ == '__main__':
    unittest.main()