"""
======================================================================
This module reads back the records of a hyperparameter grid scan,
ranks them, and writes the tables a reader compares them from.

The original project scraped a finished training out of its log file:
it read the last ten lines, looked for a section title and took the
numbers two and three lines below it by position. That works until a
report changes its layout, and it silently reports a fit that never
happened as a zero. The training this framework runs writes structured
products instead (the per-member fit error, the hyperparameters, the
scaler), so a point is read from those and a folder that lacks one is
a failed point, not a zero.

Ranking is a comparison of what the skill reported, never a judgement
of its own: this module picks the smallest error, it does not decide
that any error is acceptable.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import csv
import os

from .plan import query_of

# Row statuses of a scan. 'skipped' is a point whose folder was already
# complete when the scan reached it, and is reported rather than trained
# over, because a plain training request would destroy the surface that
# is already there.
STATUS_OK = 'ok'
STATUS_FAILED = 'failed'
STATUS_SKIPPED = 'skipped'


def point_record(outcome):
    """Read the result channels of one dispatched point.

    A point can come back three ways: as the live child state (local
    execution), as a result wrapped around the state dump (what a remote
    child leaves in its result file), or as a plain property result. The
    training record is under the same keys in all three; only where to
    look differs.

    Parameters
    ----------
    outcome : object or None
        One entry of the dispatch outcome list.

    Returns
    -------
    dict
        The result channels of the point; empty when it returned
        nothing readable.
    """
    if outcome is None:
        return {}
    pmd = getattr(outcome, 'pmd', None)
    if pmd is not None:
        return dict(getattr(pmd, 'results', None) or {})
    try:
        data = dict(outcome)
    except (TypeError, ValueError):
        return {}
    nested = data.get('pmd')
    if isinstance(nested, dict):
        return dict(nested.get('results') or {})
    return data


def point_failure(outcome):
    """What a dispatched point said about its own failure.

    A worker no longer writes the parent's run log (two handles on one
    file overwrite each other), so the sentence its runtime settled on --
    the record channel that came back empty, for instance -- has to reach
    the scan table some other way. This is that way: the child's runtime
    report, read from whichever carrier the point came back in.

    Parameters
    ----------
    outcome : object or None
        One entry of the dispatch outcome list.

    Returns
    -------
    str
        The point's failure message; empty when it completed or reported
        nothing. The message is read whatever the runtime's own status
        says, because `build_row` only consults it for a row that has no
        numbers, so a point that finished cannot be talked out of its
        status by a stale sentence.
    """
    runtime = getattr(outcome, 'runtime', None)
    if runtime is not None and not isinstance(runtime, dict):
        runtime = runtime.to_dict()
    if not isinstance(runtime, dict):
        # A `Result` answers every attribute question with its payload
        # (`data/runtime/result.py::__getattr__`), so a dispatched child's
        # runtime reaches this point as the dumped dict rather than a
        # `RuntimeState`, and a plain state dump holds it under the
        # top-level 'runtime' key. Neither is read through `point_record`,
        # which answers with the *results* payload and would drop this key.
        try:
            data = dict(outcome) if outcome is not None else {}
        except (TypeError, ValueError):
            data = {}
        nested = data.get('runtime')
        runtime = nested if isinstance(nested, dict) else {}
    return str(runtime.get('error_message') or '')


def record_errors(record):
    """Take the training errors out of one record.

    Parameters
    ----------
    record : dict
        Result channels of a point.

    Returns
    -------
    tuple(dict, dict, object)
        (errors keyed role then channel, units keyed channel, wall
        seconds). Empty dicts when the record carries no fit error.
    """
    errors = record.get('train_errors')
    if not isinstance(errors, dict) or not errors:
        return {}, {}, record.get('train_wall_seconds')
    units = record.get('train_metric_units') or {}
    return errors, units, record.get('train_wall_seconds')


def disk_errors(context, modeldir):
    """Read one point's folder from disk through the skill.

    Parameters
    ----------
    context : dict
        Scan context, for the skill's query module and ensemble size.
    modeldir : str
        The point's model directory.

    Returns
    -------
    tuple(dict, dict, str)
        (errors, units, reason). The reason is empty when the folder is
        complete and names what is missing when it is not.
    """
    record = query_of(context).read_training_record(
        modeldir, members=context['members'])
    if record is None:
        return {}, {}, 'no model directory at %s' % modeldir
    errors = record.get('errors') or {}
    units = record.get('units') or {}
    incomplete = record.get('incomplete') or []
    if not errors:
        return {}, {}, 'no fit error in %s (%s)' % (
            modeldir, '; '.join(incomplete) or 'no channel reported')
    if incomplete:
        return errors, units, 'incomplete: %s' % '; '.join(incomplete)
    return errors, units, ''


def folder_complete(context, modeldir):
    """Whether one point's folder already holds a finished training.

    A scan that reaches a complete folder does not train over it: the
    request that would overwrite it is a plain training one, and a plain
    training writes new weights before its fit starts. The point is
    reported as skipped and its numbers are read from the folder.

    Parameters
    ----------
    context : dict
        Scan context, for the skill's query module and ensemble size.
    modeldir : str
        The point's model directory.

    Returns
    -------
    bool
        True when every member product of every ensemble folder is
        there.
    """
    record = query_of(context).read_training_record(
        modeldir, members=context['members'])
    return bool(record and record.get('complete'))


def _flatten(errors):
    """Key one record's errors by column for the tables.

    Parameters
    ----------
    errors : dict
        Role to channel to the per-member values.

    Returns
    -------
    dict
        (role, channel, member index) to its value.
    """
    columns = {}
    for role in sorted(errors):
        channels = errors.get(role) or {}
        for channel in sorted(channels):
            values = channels[channel]
            items = values if isinstance(values, (list, tuple)) else [values]
            for index, value in enumerate(items):
                try:
                    columns[(role, channel, index)] = float(value)
                except (TypeError, ValueError):
                    continue
    return columns


def column_stat(columns, channel, statistic):
    """Reduce one channel's member values to the ranking statistic.

    Parameters
    ----------
    columns : dict
        Flattened columns from _flatten().
    channel : str
        Channel name to rank by.
    statistic : str
        'max' takes the worst member (the folder is deployed as a whole,
        so the member that serves the worst surface is the one that
        matters), 'mean' averages them.

    Returns
    -------
    float or None
        The statistic, or None when the channel reports nothing.
    """
    values = [value for (role, name, index), value in columns.items()
              if name == channel]
    if not values:
        return None
    if statistic == 'mean':
        return sum(values) / len(values)
    return max(values)


def build_row(point, context, record=None, modeldir='', failure=''):
    """Assemble one scan row from a record or from the folder.

    Parameters
    ----------
    point : dict
        Planned point.
    context : dict
        Scan context.
    record : dict or None
        The point's returned record; disk is read when it is None or
        carries nothing.
    modeldir : str
        The point's model directory.
    failure : str
        What the point's own runtime said about its failure, if it said
        anything. It wins over the folder's sentence: a point whose job
        never reached the cluster leaves an absent folder, and "no model
        directory" is the consequence of the death rather than its reason.

    Returns
    -------
    dict
        Row with 'key', 'values', 'modeldir', 'status', 'reason',
        'columns', 'units', 'wall' and 'stat'.
    """
    errors, units, wall = record_errors(record or {})
    reason = ''
    if not errors:
        errors, units, reason = disk_errors(context, modeldir)
        wall = None
    columns = _flatten(errors)
    if not columns:
        reason = failure or reason or 'no training record in %s' % modeldir
    row = {
        'key': point['key'],
        'values': dict(point['values']),
        'modeldir': modeldir,
        'columns': columns,
        'units': units,
        'wall': wall,
        # A folder that reports numbers but is missing a product is a
        # half-product, not a result: the same rule the serving path
        # applies when it refuses to load a folder whose fit never finished.
        'status': STATUS_OK if (columns and not reason) else STATUS_FAILED,
        'reason': reason,
    }
    return row


def skipped_row(point, context, modeldir):
    """Build the row of a point whose folder was already complete.

    Its numbers are read from the folder like any other: a scan that
    skips a finished point still has to rank it, or the ranking would
    describe only the work this invocation happened to do.

    Parameters
    ----------
    point : dict
        Planned point.
    context : dict
        Scan context.
    modeldir : str
        The point's model directory.

    Returns
    -------
    dict
        Row with the folder's errors, status 'skipped'.
    """
    row = build_row(point, context, None, modeldir)
    row['status'] = STATUS_SKIPPED
    row['reason'] = 'model directory already complete, not retrained'
    return row


def rank_points(rows, sort_by, sort_stat='max', sort_order='asc'):
    """Rank the points of a scan by one channel's error.

    A row whose channel reports nothing sorts last, and ties break on
    the point key so the same scan reads out the same table twice.

    Parameters
    ----------
    rows : list of dict
        Rows from build_row().
    sort_by : str
        Channel name to rank on.
    sort_stat : str
        'max' or 'mean' over the ensemble members.
    sort_order : str
        'asc' (smaller error first) or 'desc'.

    Returns
    -------
    list of dict
        The rows in rank order, each carrying 'stat' and 'rank'.
    """
    for row in rows:
        row['stat'] = column_stat(row['columns'], sort_by, sort_stat)

    def sort_key(row):
        value = row['stat']
        if sort_order == 'desc':
            # A point that reports nothing ranks last in either order.
            keyed = float('-inf') if value is None else -float(value)
        else:
            keyed = float('inf') if value is None else float(value)
        return (keyed, row['key'])

    ranked = sorted(rows, key=sort_key)
    for position, row in enumerate(ranked):
        row['rank'] = position + 1
    return ranked


def _columns_of(rows):
    """List the error columns the rows actually carry, in table order."""
    names = set()
    for row in rows:
        names.update(row['columns'])
    return sorted(names)


def format_plan(points, keywords):
    """Render the plan of a scan before it runs.

    Written first, so a scan that dies halfway still leaves the table of
    what was asked for.

    Parameters
    ----------
    points : list of dict
        Planned points.
    keywords : list of str
        Varied keyword names, in column order.

    Returns
    -------
    str
        The '&grid search' block.
    """
    rule = '-' * 79
    lines = ['  &grid search', rule,
             '  Number of search: %d' % len(points)]
    if not keywords:
        lines.append(rule)
        return '\n'.join(lines) + '\n'
    header = '  %-1s' % ''
    for keyword in keywords:
        header += '%16s' % keyword
    lines.append(header)
    for point in points:
        row = '  %-1s' % ''
        for keyword in keywords:
            row += '%16s' % _render(point['values'][keyword])
        row += '  %s' % point['key']
        lines.append(row)
    lines.append(rule)
    return '\n'.join(lines) + '\n'


def _render(value):
    """Print one candidate value compactly."""
    if isinstance(value, (list, tuple)):
        return ' '.join(_render(item) for item in value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return '%.3e' % value
    return str(value)


# Folder names are roles; the short alias a table column prints is the
# skill's declaration (manifest `scan_role_aliases`), passed in by the
# caller. A folder with no alias prints its own name.


def format_summary(rows, keywords, sort_by, aliases=None):
    """Render the ranked table of a completed scan.

    Parameters
    ----------
    rows : list of dict
        Ranked rows.
    keywords : list of str
        Varied keyword names.
    sort_by : str
        The ranked channel, named in the heading so a reader knows what
        'best' means.
    aliases : dict, optional
        Role -> short column alias, as the skill declares it.

    Returns
    -------
    str
        The '&grid search summary' block.
    """
    rule = '-' * 79
    columns = _columns_of(rows)
    lines = ['  &grid search summary', rule,
             '  Ranked by %s (worst member), %d points'
             % (sort_by, len(rows))]
    header = '  %4s %-28s' % ('rank', 'key')
    for keyword in keywords:
        header += '%16s' % keyword
    for role, channel, index in columns:
        header += '%16s' % ('%s_v%d %s' % ((aliases or {}).get(role, role),
                                           index, channel))
    header += '%12s %8s' % ('wall (s)', 'status')
    lines.append(header)
    if columns:
        units = '  %4s %-28s' % ('', '')
        for keyword in keywords:
            units += '%16s' % ''
        for _role, channel, _index in columns:
            unit = next((row['units'].get(channel, '')
                         for row in rows if row['units'].get(channel)), '')
            units += '%16s' % unit
        lines.append(units)
    for row in rows:
        line = '  %4d %-28s' % (row.get('rank', 0), row['key'][:28])
        for keyword in keywords:
            line += '%16s' % _render(row['values'].get(keyword, ''))
        for name in columns:
            value = row['columns'].get(name)
            line += '%16s' % ('%.8f' % value if value is not None else '--')
        wall = ('%.1f' % float(row['wall'])
                if row.get('wall') is not None else '--')
        line += '%12s %8s' % (wall, row['status'])
        lines.append(line)
    for row in rows:
        if row['status'] != STATUS_OK:
            lines.append('  %s (%s): %s' % (row['key'], row['status'],
                                            row['reason'] or 'no record'))
    lines.append(rule)
    return '\n'.join(lines) + '\n'


def write_csv(path, rows, keywords, sort_by):
    """Write the ranked scan as machine-readable rows.

    Parameters
    ----------
    path : str
        Destination file.
    rows : list of dict
        Ranked rows.
    keywords : list of str
        Varied keyword names, one column each.
    sort_by : str
        Ranked channel.

    Returns
    -------
    str
        The written path.
    """
    columns = _columns_of(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        header = ['rank', 'key', 'status', 'sort_by', 'sort_value',
                  'wall_seconds', 'modeldir']
        header += list(keywords)
        writer.writerow(header)
        for row in rows:
            values = [row.get('rank', ''), row['key'], row['status'],
                      sort_by, row['stat'],
                      row['wall'] if row['wall'] is not None else '',
                      row['modeldir']]
            values += [_render(row['values'].get(keyword, ''))
                       for keyword in keywords]
            writer.writerow(values)
        detail = ['rank', 'key', 'role', 'channel', 'member',
                  'value', 'unit']
        writer.writerow([])
        writer.writerow(detail)
        for row in rows:
            for (role, channel, index), value in sorted(
                    row['columns'].items()):
                writer.writerow([row.get('rank', ''), row['key'], role,
                                 channel, index, value,
                                 row['units'].get(channel, '')])
    return path


def write_crashed(path, rows):
    """Write the points that produced no record.

    Parameters
    ----------
    path : str
        Destination file.
    rows : list of dict
        All scan rows.

    Returns
    -------
    str or None
        The written path, or None when nothing failed: a file naming
        crashes that did not happen would read as if the scan had some.
    """
    failures = [row for row in rows if row['status'] == STATUS_FAILED]
    if not failures:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        for row in failures:
            handle.write('%s  %s\n' % (row['modeldir'],
                                       row['reason'] or 'no record'))
    return path
