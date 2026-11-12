"""
======================================================================
This module is the public API of the framework training-data layer:
the container that reads, saves and appends frames, and the
permutation-map expansion that the train workflow applies before a
skill backend is initialized.

Oct 23 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .dataset import (FORMAT_NAME, FORMAT_VERSION, TrainingData,
                      from_payload, load, save)
from .permutation import expand_permutations, read_permutation_map

__all__ = [
    'FORMAT_NAME',
    'FORMAT_VERSION',
    'TrainingData',
    'from_payload',
    'load',
    'save',
    'expand_permutations',
    'read_permutation_map',
]