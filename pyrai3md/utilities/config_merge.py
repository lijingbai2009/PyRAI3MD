"""
======================================================================
This module merges configuration layers: one rule for what "unset" means,
one rule for how two dictionaries combine.

Four framework modules each carried their own copy of that pair of rules
- `schema/adapter.py` twice (`_deep_update` and `_deep_merge`),
`kernel/environment.py` once, `qualification/manager.py` once - and the
copies disagreed about the one thing that matters to a caller: two of
them modified the base dictionary in place and returned it, the other
two returned a copy. A caller that assumed one contract while the code
it called honoured the other got either a shared mutation it did not ask
for or a merge that never reached the object it inspected.

`deep_merge` is the copying form. A layering helper that reaches into the
dictionary it was handed cannot be used where the caller keeps reading
its own view of the layer, so the copies stay and the call sites name
what they use.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy


def is_empty_value(value):
    """Whether a configuration value counts as unset.

    None, the empty string, an empty list and an empty dict are unset, so
    a higher-priority layer leaves the decision to the next lower one.
    False and 0 are answers, not absences, and never count as unset.

    Parameters
    ----------
    value : object
        Configuration value to test.

    Returns
    -------
    bool
        True when the value is considered unset.
    """
    return value is None or value == '' or value == [] or value == {}


def deep_merge(base, override):
    """Merge one configuration layer over another as a new dictionary.

    - A dict on both sides of one key merges recursively.
    - Any other value in `override` replaces the value in `base`.
    - An unset value in `override` (see :func:`is_empty_value`) keeps the
      `base` value, or leaves the key absent when `base` has no answer.
    - Neither argument is modified.

    Parameters
    ----------
    base : dict
        The lower-priority layer.
    override : dict
        The higher-priority layer.

    Returns
    -------
    dict
        The merged layer.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        elif is_empty_value(value):
            continue
        else:
            result[key] = copy.deepcopy(value)
    return result
