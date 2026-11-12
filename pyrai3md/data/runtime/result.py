"""
======================================================================
This module provides Result, a dict-like data container for computed
properties. It stores all calculation outputs as key-value pairs,
with uniform access through dict-style or attribute-style syntax.

Result objects are transferable between Skills.

Jul 30 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.serialization import json_safe


class Result:
    """Dict-like container for computed properties.

    Stores all calculation outputs uniformly as key-value pairs.
    Supports both dict-style access (result['energy']) and
    attribute-style access (result.energy) for convenience.

    Parameters
    ----------
    data : dict or None
        Initial data dictionary. Lists and tuples are automatically
        converted to numpy arrays.
    """

    def __init__(self, data=None):
        self._data = {}
        if data:
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    self._data[key] = np.asarray(value)
                else:
                    self._data[key] = value

    # ---- dict-like interface ----

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, (list, tuple)):
            self._data[key] = np.asarray(value)
        else:
            self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __bool__(self):
        return len(self._data) > 0

    def get(self, key, default=None):
        """Get a value by key with optional default."""
        return self._data.get(key, default)

    def keys(self):
        """Return result keys."""
        return self._data.keys()

    def values(self):
        """Return result values."""
        return self._data.values()

    def items(self):
        """Return result items."""
        return self._data.items()

    def update(self, d):
        """Update with another dict or Result."""
        if isinstance(d, Result):
            self._data.update(d._data)
        else:
            for key, value in d.items():
                if isinstance(value, (list, tuple)):
                    self._data[key] = np.asarray(value)
                else:
                    self._data[key] = value

    # ---- backward-compatible attribute access ----

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._data.get(name)

    # ---- serialization ----

    def to_dict(self):
        """Serialize to a plain dictionary.

        Every value goes through utilities.serialization json_safe, the
        one converter: this method used to carry its own top-level-only
        copy of the same rule, which left a nested array unconverted.

        Returns
        -------
        dict
            Dictionary with numpy arrays converted to lists.
        """
        return {key: json_safe(value) for key, value in self._data.items()}

    @classmethod
    def from_dict(cls, data):
        """Construct a Result from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with result data.

        Returns
        -------
        Result
            New instance.
        """
        return cls(data=data)

    def copy(self):
        """Create a deep copy.

        Returns
        -------
        Result
            Independent copy.
        """
        new_data = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                new_data[key] = value.copy()
            else:
                new_data[key] = value
        return Result(data=new_data)

    def __repr__(self):
        keys = list(self._data.keys())
        return "Result(keys=%s)" % keys


def task_config_results(task):
    """Read the results stored in one task config.

    A skill backend writes results into the task config following its own
    schema (``config.results``); a plain dict config (tests and the early
    task queue) uses the ``results`` key. This is the single reader both
    the Calculator and the composite assembler use.

    Parameters
    ----------
    task : dict
        Task descriptor carrying a ``config`` entry.

    Returns
    -------
    dict or None
        The result dictionary written by the backend, or None.
    """
    config = task.get('config')
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get('results')
    return getattr(config, 'results', None) or None
