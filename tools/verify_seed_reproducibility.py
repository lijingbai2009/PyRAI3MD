"""End-to-end reproducibility check for the gl_seed wiring.

Runs the same md input (8 trajectories, fssh, deco off) twice in two
temp dirs and verifies that every trajectory's md.pop is byte-identical.
"""
import glob
import hashlib
import os
import subprocess
import sys
import tempfile

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_input(tmpdir):
    wf = {
        'control': {'title': 'seedtest', 'jobtype': 'md',
                    'theory': ['qdm'], 'driver': 'main', 'gl_seed': 7},
        'atoms': ['X'],
        'coordinates': [[-20 * 0.52917721092, 0.0, 0.0]],
        'velocities': [[0.005, 0.0, 0.0]],
        'mass': [2000.0 / 1822.888486209],
        'molecule': {'charges': 0, 'multiplicity': 1, 'ci': [2],
                     'spin': [0, 0]},
        'md': {'initcond': '0', 'ninitcond': 8, 'multitask': True,
               'multipool': 4, 'sfhp': 'fssh', 'nactype': 'nac',
               'deco': 'off', 'step': 220, 'size': 20.0, 'root': 1},  # root is 1-based (S0)
        'environment': {},
        'qdm': {'method': 'tully1', 'model_mass': 2000.0,
                'execution_mode': 'local'},
    }
    path = os.path.join(tmpdir, 'input.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(wf, f, sort_keys=False)
    return path


def run_once(tag):
    tmpdir = tempfile.mkdtemp(prefix='seedtest_%s_' % tag)
    inp = build_input(tmpdir)
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run(
        [sys.executable, '-m', 'pyrai3md', inp],
        cwd=tmpdir, capture_output=True, text=True, timeout=1200,
        env=env)
    if proc.returncode != 0:
        print('RUN %s FAILED rc=%d' % (tag, proc.returncode))
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        sys.exit(1)
    pops = sorted(glob.glob(os.path.join(tmpdir, '**', '*.md.pop'),
                            recursive=True))
    print('run %s: %d md.pop files' % (tag, len(pops)))
    digests = {}
    for p in pops:
        key = os.path.relpath(p, tmpdir)
        with open(p, 'rb') as f:
            digests[key] = hashlib.sha256(f.read()).hexdigest()
    return digests


d1 = run_once('a')
d2 = run_once('b')

if len(d1) == 0:
    print('FAIL: no md.pop produced')
    sys.exit(1)

common = sorted(set(d1) & set(d2))
mismatch = [k for k in common if d1[k] != d2[k]]
print('trajectories compared: %d' % len(common))
if mismatch:
    print('FAIL: mismatched trajectories: %s' % mismatch)
    sys.exit(1)
print('PASS: all trajectories byte-identical across the two runs')
