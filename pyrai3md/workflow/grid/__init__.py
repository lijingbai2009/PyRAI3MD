"""
======================================================================
This package plans and reads back a hyperparameter grid scan.

The grid belongs to the skill, not to the framework: which keywords may
be varied is a declaration the skill makes in its manifest, what shape a
candidate must have is a rule the skill enforces, and what a finished
training left on disk is a folder layout only the skill knows. The two
modules here therefore reach the skill through its query interface and
import nothing of its implementation.

    plan.py     turn a declared grid into points, and stamp one point
                onto a materialized child state
    record.py   read the records the points came back with, rank them,
                and write the tables a reader compares them from

Nothing here trains, evaluates or scores a model. A scan is a plan, a
fan-out over the existing single-shot training path, and a ranking of
the numbers the skill already reported.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .plan import (build_context, first_skill_name, plan_points,
                   point_key, point_model_dir, skill_surface, write_point)
from .record import (build_row, folder_complete, format_plan, format_summary,
                     point_failure, point_record, rank_points, skipped_row,
                     write_crashed, write_csv)

__all__ = [
    'build_context',
    'first_skill_name',
    'plan_points',
    'point_key',
    'point_model_dir',
    'skill_surface',
    'write_point',
    'build_row',
    'format_plan',
    'format_summary',
    'point_failure',
    'point_record',
    'rank_points',
    'folder_complete',
    'skipped_row',
    'write_crashed',
    'write_csv',
]
