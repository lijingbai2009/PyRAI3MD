"""
======================================================================
This module provides the machine-learning outcome channel for the
single-shot ML jobtypes.

A training run or a test-set evaluation returns a record, not a
property: the skill fits the networks and reports the fit error, and
the framework has nothing to iterate and no geometry to move. What it
can still settle honestly is whether the run produced the record its
jobtype is contracted to carry. Today that check does not exist: the
ML jobtypes fall through to an unconditional completion, so a call
that trained nothing reads back as a successful training run.

The framework computes nothing here and judges nothing scientific. It
looks at the channels the skill returned, states whether the record is
there, and marks the runtime accordingly. There is deliberately no
error threshold: a validation mean absolute error is not a convergence
criterion, and a number that decides pass from fail would have to be
invented. The comparative judgement of several trainings belongs to
the grid scan, which ranks the records the skill produced.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

# Jobtypes settled through this channel. 'predict' is aliased to
# 'prediction' by the input schema, so only the normalized spelling
# appears here.
ML_JOBTYPES = ('train', 'prediction')

# The channels each ML jobtype must come back with are declared by the
# skill that serves it (manifest `result_contract`, read through
# query.get_metadata; #3.6): the names are that skill's fields, not the
# framework's. A missing channel means the skill returned a result of
# some other kind; an empty one means it returned the right kind and
# filled nothing in.

# Section titles of the verdict banner, keyed by jobtype.
_VERDICT_LABELS = {
    'train': 'Model training',
    'prediction': 'Model prediction',
}


def _as_dict(record):
    """Read one result carrier as a plain dictionary.

    Parameters
    ----------
    record : Result or dict or None
        Anything the single-shot path hands back.

    Returns
    -------
    dict
        The record's keys and values; empty when there is nothing to
        read.
    """
    if record is None:
        return {}
    try:
        return dict(record)
    except (TypeError, ValueError):
        return {}


def _has_entries(value):
    """Whether a record channel actually carries a value.

    A channel that exists but holds no role, no channel or only empty
    collections is not a record: an untrained ensemble and an ensemble
    trained to a perfect zero are different statements, and only the
    second one is a value.

    Parameters
    ----------
    value : object
        The channel value as returned by the skill.

    Returns
    -------
    bool
        True when at least one leaf value is present.
    """
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_entries(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def ml_verdict(jobtype, record, required=None):
    """Judge whether one ML call produced its contracted record.

    Parameters
    ----------
    jobtype : str
        Jobtype token of the run ('train' or 'prediction').
    record : Result or dict or None
        The result the single call returned.
    required : sequence of str, optional
        The channels this jobtype must carry, as the serving skill
        declared them. Absent means no skill declared a contract, and
        the run is not settled as complete: an ML job with no declared
        record cannot be judged.

    Returns
    -------
    tuple(bool, str)
        (settled, reason). The reason names the missing or empty
        channel and is empty when the run is complete.
    """
    token = str(jobtype or '').split()[0] if jobtype else ''
    data = _as_dict(record)
    channels = tuple(required or ())
    if not channels:
        return False, ("%s returned no declared record contract: the skill "
                       "serving it declares no result channels for this "
                       "jobtype" % token)
    for channel in channels:
        if channel not in data:
            return False, ("%s returned no %s channel: the call produced "
                           "no %s record" % (token, channel, token))
        if not _has_entries(data[channel]):
            return False, ("%s returned an empty %s channel: nothing was "
                           "recorded for the %s run"
                           % (token, channel, token))
    frames = data.get('predict_frames')
    if token == 'prediction':
        try:
            count = int(frames)
        except (TypeError, ValueError):
            count = 0
        if count < 1:
            return False, ("prediction evaluated no test frame: an error "
                           "table over zero geometries is not a result")
    return True, ''


def ml_detail(jobtype, record):
    """Summarize the record for the verdict banner.

    Counts and paths the skill already reported, assembled into one
    line. No value is derived here beyond the number of ensembles and
    the number of channels they carry.

    Parameters
    ----------
    jobtype : str
        Jobtype token of the run.
    record : Result or dict or None
        The result the single call returned.

    Returns
    -------
    str
        Banner tail, e.g. '(2 ensembles, 4 channels, wall clock 63.1 s)'.
    """
    data = _as_dict(record)
    token = str(jobtype or '').split()[0] if jobtype else ''
    errors = data.get('train_errors') or data.get('predict_errors') or {}
    parts = []
    if isinstance(errors, dict) and errors:
        # The two records are keyed differently and the banner has to say
        # what the keys actually are: a training record is keyed by
        # ensemble folder (each holding channels), a prediction record by
        # channel (each holding statistics). Counting both as "ensembles"
        # would print a number that names nothing in the test-set case.
        if token == 'prediction':
            parts.append('%d channels' % len(errors))
        else:
            parts.append('%d ensembles' % len(errors))
            parts.append('%d channels' % sum(
                len(value) if isinstance(value, dict) else 1
                for value in errors.values()))
    if token == 'prediction' and data.get('predict_frames') is not None:
        parts.append('%s frames' % data['predict_frames'])
    seconds = data.get('train_wall_seconds')
    if seconds is None:
        seconds = data.get('predict_wall_seconds')
    try:
        parts.append('wall clock %.1f s' % float(seconds))
    except (TypeError, ValueError):
        pass
    if token == 'train':
        value = data.get('train_fitlogs')
        fitlogs = [str(path) for path in (value if value is not None else [])]
        if fitlogs:
            folders = sorted({os.path.dirname(path) for path in fitlogs})
            parts.append('fit logs %s' % ', '.join(folders))
    else:
        summary = data.get('predict_summary')
        if summary:
            parts.append('summary %s' % summary)
    return ', '.join(parts)


def declared_channels(state, token):
    """The record channels the run's skill declared for one jobtype."""
    skill_name = ''
    try:
        for _category, task in state.iter_tasks():
            skill_name = str(task.get('skill_name') or '')
            break
    except (AttributeError, TypeError):
        return ()
    if not skill_name:
        return ()
    from ...skills.query import discover_all

    try:
        meta = discover_all()
    except (ImportError, AttributeError):
        return ()
    entry = next((value for value in meta.values()
                  if value.get('skill_name') == skill_name), {})
    module_path = entry.get('_dir_name')
    if not module_path:
        return ()
    import importlib

    from ...skills.query import skill_module_path

    try:
        skill_query = importlib.import_module(
            skill_module_path(module_path))
        contract = skill_query.get_metadata().get('result_contract') or {}
    except (ImportError, AttributeError):
        return ()
    return tuple(contract.get(token) or ())


def settle_ml_outcome(state, result):
    """Settle the outcome of one machine-learning single call.

    Runs that are not ML jobs keep the plain completion of the
    single-shot path, byte for byte. An ML run is settled by whether
    its record came back, and the verdict is written to the run log so
    a reader sees which statement was made rather than inferring it
    from the absence of an error.

    Parameters
    ----------
    state : PyRAIState
        State carrying the skill-updated system and the result.
    result : Result or dict or None
        The result of the single call.
    """
    token = str(getattr(state, 'jobtype', '') or '').split()
    token = token[0] if token else ''
    record = result if result is not None else getattr(
        getattr(state, 'pmd', None), 'results', None)

    if token not in ML_JOBTYPES:
        if getattr(state, 'runtime', None) is not None:
            state.runtime.mark_completed()
        return

    settled, reason = ml_verdict(token, record,
                                 required=declared_channels(state, token))
    run_logger = getattr(state, 'logger', None)
    if run_logger is not None:
        run_logger.log_ml_summary(
            settled, detail=ml_detail(token, record), message=reason,
            label=_VERDICT_LABELS.get(token, 'Machine learning run'))

    if getattr(state, 'runtime', None) is not None:
        if settled:
            state.runtime.mark_completed()
        else:
            state.runtime.mark_failed(reason)
