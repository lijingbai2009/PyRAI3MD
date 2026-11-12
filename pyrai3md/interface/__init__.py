"""
======================================================================
Interface layer for PyRAI3MD.

This package provides external input conversion:
- parser: Parse traditional text-format input files into raw dict
- loader: Load agent task JSON configurations into raw dict

Both output a unified raw configuration dictionary for downstream
Schema validation and Skill processing.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from . import parser
from . import loader
