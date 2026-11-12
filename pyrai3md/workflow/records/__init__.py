"""
======================================================================
This package collects the run records of the workflow layer: what one
execution leaves behind and reads back.

The modules are components of the execution chains, not workflows. None
of them declares a Workflow subclass and none appears in
WORKFLOW_REGISTRY or JOBTYPE_MAP:

    records/
        trajectory.py    result containers returned by iterative
                         workflows (the universal TrajectorySet, with
                         the single-trajectory run as its count-1 case)
        deviation.py     NN prediction deviation reception channel
                         (the err_* keys, tolerant fetch: the framework
                         never computes a deviation)
        ml.py            machine-learning outcome channel (does the
                         'train'/'prediction' call carry the record its
                         jobtype promises? no threshold, no arithmetic;
                         called by the Train and Prediction nodes)
        output.py        MD trajectory file writer (the title.md.* and
                         title.sh.* legacy records) plus record_due,
                         the single predicate that decides when a step
                         is one the run asked to have recorded, and
                         lattice_comment, the extended-XYZ cell key of
                         one periodic frame
        diagnostics.py   the run-health checks that watch a trajectory
                         without changing it

Design rules:
- trajectory.py, output.py and diagnostics.py serve the MD chain
  (workflow/md.py and workflow/integrator/); the multiscale chain never
  imports them, so the two chains stay independent. One formatter in
  output.py reaches past that chain: `lattice_comment`, because the
  optimization trajectory describes a box with the same line the MD
  trajectory does, and a second spelling of it would be a second answer.
  ml.py serves the
  two machine-learning nodes (workflow/train.py, workflow/prediction.py),
  which are not iterations.
- diagnostics.py reads the emit cadence from output.record_due, the one
  shared edge inside the package.
- Every module keeps the read-only policy of the layer: a record is
  taken from the state, the state is never changed by writing it. The
  one exception is deliberate and narrow: ml.py settles the runtime
  outcome of the run it is called from, and nothing else touches state.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .trajectory import TrajectoryResult, TrajectorySet
from .analysis import (
    ANALYSIS_VERSION, MISSING, NOT_APPLICABLE, NOT_REQUESTED, OK,
    AnalysisFrame, analysis_active, analysis_plan, predictor_status,
    saturation, summarize_coverage, summarize_gap, summarize_pes,
    summarize_trace, trace_payload, trace_rows, trace_spec,
)
from .adaptive import (
    SCREEN_RULES, TRAINING_UNITS, build_frame_dataset, frame_contract,
    history_frames, labels_from_outcome, screen_iteration, screen_thresholds,
    screen_trajectory,
)
from .deviation import (
    DEVIATION_KEYS, fetch_deviations, merge_deviations, judge_deviations,
)
from .ml import (
    ML_JOBTYPES, ml_detail, ml_verdict,
    settle_ml_outcome,
)
from .output import MdOutputWriter, lattice_comment, record_due
from .diagnostics import (
    CELL_GATE, CELL_IMPOSSIBLE, CELL_NOTE, CONSTRAINT_GEOMETRY,
    ENSEMBLE_DOF_IMPOSSIBLE, IMAGE_INTEGRITY, KINETIC_IDENTITY,
    MAX_CUMULATIVE_LOG_STRAIN, MAX_STEP_AXIS_RATIO,
    MAX_STEP_FRAME_ROTATION_RAD, REGION_RECONSTRUCTION,
    WARN_CUMULATIVE_LOG_STRAIN,
    CellDivergenceError, EnsembleDivergenceError, RecordedStepError,
    check_finite, check_recorded_step, frame_rotation_angle, is_debug, note,
    record, report, summary,
)

__all__ = [
    'TrajectoryResult',
    'TrajectorySet',
    'ANALYSIS_VERSION',
    'AnalysisFrame',
    'MISSING',
    'NOT_APPLICABLE',
    'NOT_REQUESTED',
    'OK',
    'analysis_active',
    'analysis_plan',
    'predictor_status',
    'saturation',
    'summarize_coverage',
    'summarize_gap',
    'summarize_pes',
    'summarize_trace',
    'trace_payload',
    'trace_rows',
    'trace_spec',
    'SCREEN_RULES',
    'TRAINING_UNITS',
    'build_frame_dataset',
    'frame_contract',
    'history_frames',
    'labels_from_outcome',
    'screen_iteration',
    'screen_thresholds',
    'screen_trajectory',
    'DEVIATION_KEYS',
    'fetch_deviations',
    'merge_deviations',
    'judge_deviations',
    'ML_JOBTYPES',
    'ml_detail',
    'ml_verdict',
    'settle_ml_outcome',
    'MdOutputWriter',
    'lattice_comment',
    'record_due',
    'CELL_GATE',
    'CELL_IMPOSSIBLE',
    'CELL_NOTE',
    'CONSTRAINT_GEOMETRY',
    'ENSEMBLE_DOF_IMPOSSIBLE',
    'IMAGE_INTEGRITY',
    'KINETIC_IDENTITY',
    'MAX_CUMULATIVE_LOG_STRAIN',
    'MAX_STEP_AXIS_RATIO',
    'MAX_STEP_FRAME_ROTATION_RAD',
    'REGION_RECONSTRUCTION',
    'WARN_CUMULATIVE_LOG_STRAIN',
    'CellDivergenceError',
    'EnsembleDivergenceError',
    'RecordedStepError',
    'check_finite',
    'check_recorded_step',
    'frame_rotation_angle',
    'is_debug',
    'note',
    'record',
    'report',
    'summary',
]
