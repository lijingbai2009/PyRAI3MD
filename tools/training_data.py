#!/usr/bin/env python3
"""
======================================================================
Training-data tool: extract, shuffle, split, merge and select.

`extract` calls the framework's read-only jobtype (`control.jobtype
read` plus a `read` section) and packs the returned frames into the
canonical PyRAI3MD training-data JSON. The read run itself writes
nothing; this tool owns the filtering, statistics and persistence.

The data operations work on the canonical JSON (and read a legacy
`data.json` as well, because the container ignores its retired
`ncharge` key). They keep the legacy file naming:

    <stem>-shuffled.json   <stem>-set-1.json   <stem>-set-2.json
    <stem>-merged.json     <stem>-selected.json

Usage
-----
    python tools/training_data.py extract  read.yaml [-o data.json]
    python tools/training_data.py shuffle  data.json [-s 1234] [-g 200]
    python tools/training_data.py split    data.json -p 200 [-r 0]
    python tools/training_data.py merge    data.json data2.json [...]
    python tools/training_data.py select   data.json --index 1-100,200
    python tools/training_data.py describe data.json

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pyrai3md.interface.parser import parse_input
from pyrai3md.training_data import TrainingData, load as load_dataset
from pyrai3md.training_data.dataset import _FRAME_FIELDS

_LABELS = ('energy', 'gradient', 'nac', 'soc')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract and handle PyRAI3MD training data.')
    sub = parser.add_subparsers(dest='command', required=True)

    extract = sub.add_parser(
        'extract', help='read finished calculations into a data JSON')
    extract.add_argument('input', help='YAML or classic input with '
                        'control.jobtype read and a read section')
    extract.add_argument('-o', '--output', default='data.json',
                         help='output JSON (default: data.json)')

    shuffle = sub.add_parser('shuffle', help='shuffle the frames')
    shuffle.add_argument('data', help='training-data JSON')
    shuffle.add_argument('-s', '--seed', type=int, default=1234,
                         help='random seed (default: 1234)')
    shuffle.add_argument('-g', '--group', type=int, default=0,
                         help='shuffle the first N frames separately')

    split = sub.add_parser('split', help='split into two sets')
    split.add_argument('data', help='training-data JSON')
    split.add_argument('-p', '--size', type=int, required=True,
                       help='frames in the first set')
    split.add_argument('-r', '--shuffle', type=int, default=1,
                       help='shuffle before splitting (default: 1)')
    split.add_argument('-s', '--seed', type=int, default=1234,
                       help='random seed (default: 1234)')

    merge = sub.add_parser('merge', help='append other datasets')
    merge.add_argument('data', help='training-data JSON')
    merge.add_argument('others', nargs='+', help='datasets to append')
    merge.add_argument('-r', '--shuffle', type=int, default=1,
                       help='shuffle the input before merging (default: 1)')
    merge.add_argument('-s', '--seed', type=int, default=1234,
                       help='random seed (default: 1234)')

    select = sub.add_parser('select', help='keep a subset of frames')
    select.add_argument('data', help='training-data JSON')
    select.add_argument('--index', required=True,
                        help='1-based indices/ranges, e.g. 1-100,200')

    describe = sub.add_parser('describe',
                              help='print the dataset shapes')
    describe.add_argument('data', help='training-data JSON')

    args = parser.parse_args(argv)
    if args.command == 'extract':
        return _extract(args.input, args.output)
    if args.command == 'shuffle':
        return _shuffle(args.data, args.seed, args.group)
    if args.command == 'split':
        return _split(args.data, args.size, bool(args.shuffle), args.seed)
    if args.command == 'merge':
        return _merge(args.data, args.others, bool(args.shuffle),
                      args.seed)
    if args.command == 'select':
        return _select(args.data, args.index)
    if args.command == 'describe':
        return _describe_file(args.data)
    parser.error('unknown command %r' % args.command)


# ---- extract ---------------------------------------------------------

def _extract(input_file, output):
    raw = parse_input(input_file)
    jobtype = _jobtype(raw)
    if 'read' not in jobtype:
        raise SystemExit(
            "this input runs jobtype %r; set control.jobtype read and "
            "add a read section before extracting" % (jobtype,))

    from pyrai3md.pmd import run
    payload = run(input_file)
    dataset = _dataset_from_payload(payload)
    dataset.save(output)

    print('\n    --- Read summary ---')
    print('skill:     %s' % payload.get('skill', ''))
    print('paths:     %d' % len(payload.get('paths') or []))
    print('requested: %s' % ', '.join(payload.get('requested') or []))
    print('missing:   %s' % (', '.join(payload.get('missing') or [])
                             or 'none'))
    print('\n    --- Data ---')
    _describe(dataset)
    print('\n    Saved: %s' % output)
    return 0


def _jobtype(raw):
    control = raw.get('control') if isinstance(raw, dict) else None
    value = control.get('jobtype', '') if isinstance(control, dict) else ''
    return str(value).lower().split()


def _dataset_from_payload(payload):
    frames = list(payload.get('frames') or [])
    if not frames:
        raise SystemExit('the read run returned no frame')
    requested = list(payload.get('requested') or [])

    reference = [str(symbol) for symbol in frames[0]['symbols']]
    for frame in frames[1:]:
        if [str(symbol) for symbol in frame['symbols']] != reference:
            raise SystemExit(
                'the frames do not share one atom order; a training set '
                'needs one molecule (first frame: %s, %s: %s)'
                % (' '.join(reference), frame.get('path', 'another frame'),
                   ' '.join(str(symbol) for symbol in frame['symbols'])))

    units = {}
    for frame in frames:
        unit = dict(frame.get('unit') or {})
        if not units:
            units = unit
        elif units != unit:
            raise SystemExit(
                'the frames declare different unit blocks; one dataset '
                'carries one unit set (%r vs %r)'
                % (units, unit))

    xyz = [frame['coordinates'] for frame in frames]
    symbols = [[str(symbol) for symbol in frame['symbols']]
               for frame in frames]

    channels = {}
    for label in requested:
        present = [frame.get(label) is not None for frame in frames]
        count = sum(present)
        if count == 0:
            print('    omit %-8s: no frame carries it' % label)
            continue
        if count != len(frames):
            print('    omit %-8s: only %d/%d frames carry it'
                  % (label, count, len(frames)))
            continue
        key = 'grad' if label == 'gradient' else label
        channels[key] = [frame.get(label) for frame in frames]

    return TrainingData(
        xyz=xyz,
        symbols=symbols,
        unit=units,
        source=str(payload.get('title', '') or 'read'),
        **channels)


# ---- data operations -------------------------------------------------

def _load(path):
    if not os.path.isfile(path):
        raise SystemExit('data file not found: %s' % path)
    return load_dataset(path)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _subset(dataset, indices):
    index = np.asarray(indices, dtype=int)
    values = {'xyz': dataset.xyz[index]}
    for name in _FRAME_FIELDS:
        value = getattr(dataset, name)
        values[name] = None if value is None else value[index]
    values['symbols'] = (None if dataset.symbols is None
                         else dataset.symbols[index])
    values['unit'] = dict(dataset.unit or {})
    values['source'] = dataset.source
    values['nstate'] = dataset.nstate
    values['nnac'] = dataset.nnac
    values['nsoc'] = dataset.nsoc
    values['npair'] = dataset.npair
    return TrainingData(**values)


def _permutation(size, seed, group=0):
    np.random.seed(seed)
    index = np.arange(size)
    if size > group > 0:
        first = index[:group].copy()
        second = index[group:].copy()
        np.random.shuffle(first)
        np.random.shuffle(second)
        return np.concatenate((first, second))
    index = index.copy()
    np.random.shuffle(index)
    return index


def _shuffle(path, seed, group):
    dataset = _load(path)
    if group and not (0 < group < len(dataset)):
        print('    group %d ignored: dataset has %d frames'
              % (group, len(dataset)))
    shuffled = _subset(dataset, _permutation(len(dataset), seed, group))
    output = '%s-shuffled.json' % _stem(path)
    shuffled.save(output)
    print('    Shuffled %d frames (seed %d): %s'
          % (len(shuffled), seed, output))
    print('\n    --- Data ---')
    _describe(shuffled)
    return 0


def _split(path, size, do_shuffle, seed):
    dataset = _load(path)
    if size <= 0 or size >= len(dataset):
        raise SystemExit(
            'split size %d does not divide a dataset of %d frames'
            % (size, len(dataset)))
    if do_shuffle:
        dataset = _subset(dataset,
                          _permutation(len(dataset), seed))
    first = _subset(dataset, range(0, size))
    second = _subset(dataset, range(size, len(dataset)))
    stem = _stem(path)
    first.save('%s-set-1.json' % stem)
    second.save('%s-set-2.json' % stem)
    print('    Set 1: %d frames -> %s-set-1.json' % (len(first), stem))
    print('    Set 2: %d frames -> %s-set-2.json' % (len(second), stem))
    print('\n    --- Set 1 ---')
    _describe(first)
    print('\n    --- Set 2 ---')
    _describe(second)
    return 0


def _merge(path, others, do_shuffle, seed):
    dataset = _load(path)
    if do_shuffle:
        dataset = _subset(dataset, _permutation(len(dataset), seed))
    for other in others:
        dataset.append(_load(other))
    output = '%s-merged.json' % _stem(path)
    dataset.save(output)
    print('    Merged %d frames from %d files: %s'
          % (len(dataset), 1 + len(others), output))
    print('\n    --- Data ---')
    _describe(dataset)
    return 0


def _select(path, expression):
    dataset = _load(path)
    indices = _parse_indices(expression, len(dataset))
    selected = _subset(dataset, indices)
    output = '%s-selected.json' % _stem(path)
    selected.save(output)
    print('    Selected %d of %d frames: %s'
          % (len(selected), len(dataset), output))
    print('\n    --- Data ---')
    _describe(selected)
    return 0


def _parse_indices(expression, size):
    """1-based indices and inclusive ranges ('1-100,200')."""
    indices = []
    for part in str(expression).replace(',', ' ').split():
        if '-' in part:
            start, end = part.split('-', 1)
            try:
                start, end = int(start), int(end)
            except ValueError:
                raise SystemExit('malformed index range: %r' % part)
            indices.extend(range(start, end + 1))
        else:
            try:
                indices.append(int(part))
            except ValueError:
                raise SystemExit('malformed index: %r' % part)
    if not indices:
        raise SystemExit('no index selected')
    if min(indices) < 1 or max(indices) > size:
        raise SystemExit(
            'indices must be within 1..%d, got %d..%d'
            % (size, min(indices), max(indices)))
    return [index - 1 for index in indices]


_AXIS_NAMES = {
    'xyz': ('nmol', 'natom', '3'),
    'symbols': ('nmol', 'natom'),
    'energy': ('nmol', 'nstate'),
    'grad': ('nmol', 'nstate', 'natom', '3'),
    'nac': ('nmol', 'nnac', 'natom', '3'),
    'soc': ('nmol', 'nsoc'),
    'transition_dipole': ('nmol', 'npair', '3'),
    'transition_dipole_norm': ('nmol', 'npair'),
    'osc_strength': ('nmol', 'npair'),
    'charge': ('nmol',),
    'cell': ('nmol', '6'),
    'pbc': ('nmol', '3'),
    'group_id': ('nmol',),
    'permutation_id': ('nmol',),
}

_SHAPE_ORDER = (
    'xyz', 'symbols', 'energy', 'grad', 'nac', 'soc',
    'transition_dipole', 'transition_dipole_norm', 'osc_strength',
    'charge', 'cell', 'pbc', 'group_id', 'permutation_id',
)


def _describe(dataset):
    print('nmol:    %d' % len(dataset))
    print('natom:   %d' % dataset.natom)
    print('nstate:  %d' % dataset.nstate)
    print('nnac:    %d' % dataset.nnac)
    print('nsoc:    %d' % dataset.nsoc)
    print('npair:   %d' % dataset.npair)
    for name in _SHAPE_ORDER:
        if name == 'xyz':
            value = dataset.xyz
        elif name == 'symbols':
            value = dataset.symbols
        else:
            value = getattr(dataset, name)
        if value is None:
            continue
        shape = tuple(np.asarray(value).shape)
        axes = _AXIS_NAMES.get(name)
        label = '(%s)' % ', '.join(axes) if axes else ''
        print('%-22s %-24s %s' % (name + ':', str(shape), label))
    if dataset.unit:
        print('unit:    %s' % ', '.join(
            '%s=%s' % item for item in sorted(dataset.unit.items())))


def _describe_file(path):
    dataset = _load(path)
    print('\n    --- Data: %s ---' % path)
    _describe(dataset)
    return 0


if __name__ == '__main__':
    sys.exit(main())