"""
======================================================================
This module provides the single JSON-safe conversion used by every place
PyRAI3MD writes a result document: a result dictionary of numpy arrays,
numpy scalars and nested containers becomes plain JSON data.

The walker descends into dicts, lists and tuples at any depth. That
matters because a result is not a flat table: an md record holds a list
of per-step dictionaries, a gradient is a (natom, 3) array, and the
coupling entries are lists of arrays. A converter that only looks at the
top level leaves those in place, and the failure does not surface where
the gap is -- it surfaces in whoever reads the file afterwards, as a
half-written document.

One implementation lives here on purpose. The dispatch layer and the
command-line entry each carried a copy, and only one of them learned to
descend into lists; the copy that did not is the one the Slurm child
used, and its crash left a truncated result.json for the parent to
misread.

Oct 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os

import numpy as np


def json_safe(data):
    """Convert numpy objects in `data` into JSON-serializable form.

    Parameters
    ----------
    data : object
        Any value: a scalar, an array, or a dict/list/tuple nesting them.

    Returns
    -------
    object
        The same structure with numpy leaves replaced by Python types.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, dict):
        return {str(key): json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(value) for value in data]
    return data


def write_json(path, data, indent=2):
    """Write `data` as JSON so a reader never sees a half-written file.

    The document is written beside its destination under a temporary name
    and then renamed, which is atomic on every filesystem this project
    writes results to. A direct open(path, 'w') would leave the truncated
    prefix of a crashed dump in place, and the next reader -- the parent
    runtime collecting a dispatched child's result -- gets a parse error
    that names the file instead of the failure that made it.

    Parameters
    ----------
    path : str
        Destination file path.
    data : object
        JSON-serializable structure (see json_safe() for the conversion).
    indent : int
        Indentation passed to json.dump().
    """
    temporary = '%s.%d.tmp' % (path, os.getpid())
    try:
        with open(temporary, 'w', encoding='utf-8') as handler:
            json.dump(data, handler, indent=indent)
            handler.flush()
            os.fsync(handler.fileno())
        os.replace(temporary, path)
    except BaseException:
        # A failed dump must not leave its partial file behind to be
        # mistaken for a result; the original error is what matters.
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
