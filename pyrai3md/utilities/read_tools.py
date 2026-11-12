"""
======================================================================
This module provides file reading and type conversion utilities for
PyRAI3MD. It includes the ReadVal class for value parsing and the
ReadIndex class for flexible index/range parsing from files or lists.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os


class ReadVal:
    """Parse and convert string values to specific data types.

    Parameters
    ----------
    data_type : str
        Conversion type indicator:
        's'  - string
        'i'  - integer
        'f'  - float
        'sl' - string list
        'il' - integer list
        'fl' - float list
        'b'  - boolean

    Examples
    --------
    >>> rv = ReadVal('f')
    >>> rv(['3.14'])
    3.14
    >>> rv = ReadVal('il')
    >>> rv(['1', '2', '3'])
    [1, 2, 3]
    """

    def __init__(self, data_type='s'):
        self.type = data_type

    def __call__(self, x):
        self._data_func = {
            's': self._string,
            'i': self._integer,
            'f': self._floatnum,
            'sl': self._string_list,
            'il': self._integer_list,
            'fl': self._floatnum_list,
            'b': self._boolean,
            'bl': self._boolean_list,
        }
        return self._data_func[self.type](x)

    @staticmethod
    def _string(x):
        return str(x[0])

    @staticmethod
    def _integer(x):
        return int(x[0])

    @staticmethod
    def _floatnum(x):
        return float(x[0])

    @staticmethod
    def _string_list(x):
        return [str(f) for f in x]

    @staticmethod
    def _integer_list(x):
        return [int(f) for f in x]

    @staticmethod
    def _floatnum_list(x):
        return [float(f) for f in x]

    @staticmethod
    def _boolean(x):
        b = {
            'true': True,
            '1': True,
            'false': False,
            '0': False,
        }
        return b[x[0].lower()]

    @staticmethod
    def _boolean_list(x):
        b = {
            'true': True,
            '1': True,
            'false': False,
            '0': False,
        }
        return [b[v.lower()] for v in x]


class ReadIndex:
    """Parse index specifications from lists or files.

    Supports single indices, ranges (e.g. '1-5'), and comma-separated
    groups. Can read from a file or directly from a list of strings.

    Parameters
    ----------
    index_type : str
        's' for single index parsing, 'g' for group index parsing.
    start : int
        Starting offset for index conversion (e.g. 1 for 1-based input).
    sort : bool
        Whether to sort and deduplicate the resulting index list.

    Examples
    --------
    >>> ri = ReadIndex('s', start=1)
    >>> ri(['1', '3', '5-7'])
    [0, 2, 4, 5, 6]
    """

    def __init__(self, index_type='s', start=0, sort=True):
        self.type = index_type
        self.start = start
        self.sort = sort

    def __call__(self, var):
        if self.type != 'g':
            return self._read_index(var)
        else:
            return self._read_index_group(var)

    def _get_index(self, index):
        """Parse single, range, and separated range indices into a list."""
        index_list = []
        for i in index:
            if '-' in i:
                a, b = i.split('-')
                a, b = int(a) - self.start, int(b) - self.start
                index_list += range(a, b + 1)
            else:
                index_list.append(int(i) - self.start)

        if self.sort:
            index_list = sorted(list(set(index_list)))  # remove duplicates and sort

        return index_list

    def _read_index(self, var):
        """Read a group of indices from a list or a file."""
        file = var[0]
        if os.path.exists(file):
            with open(file, 'r') as indexfile:
                indices = indexfile.read().split()
        else:
            indices = var

        index_list = self._get_index(indices)
        return index_list

    def _read_index_group(self, var):
        """Read groups of indices from a list or a file."""
        file = var[0]
        if os.path.exists(file):
            with open(file, 'r') as indexfile:
                indices = indexfile.read().splitlines()
            indices = [x.split() for x in indices]
        else:
            indices = ' '.join(var).split(',')
            indices = [x.split() for x in indices]

        index_group = [self._get_index(x) for x in indices]
        return index_group
