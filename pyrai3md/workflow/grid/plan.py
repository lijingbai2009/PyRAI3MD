"""
======================================================================
This module plans a hyperparameter grid and stamps one point of it onto
a materialized child state.

The plan is pure and imports no TensorFlow: it reads what the skill
declares, expands the candidate lists the run supplied, and refuses a
grid the skill cannot train. The stamp writes into the flat skill
config of each task of one child state, which is the copy the Dispatcher
made of the parent, so no two points share a configuration and the
parent itself is never touched.

One shape decision carries the whole module. A skill reads a keyword
written as a list along its ensemble member axis (schema/hyper.py of
the nn skill says so in its own words), so a grid value must never be
handed over as a list of alternatives: the alternatives were expanded
into points already, and what remains for one point is one value.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import itertools
import os

# The keywords of a point that reach the skill as one member group
# rather than as a scalar: the keyword's own value is a list, so the
# outer list is what marks it as one alternative.
_GROUP_KINDS = ('list', 'pairs')

# Separator between the keyword=value pairs of a point key. It cannot be
# '.', because a rendered float carries one (0.005), and a key that cannot
# be split back into its pairs is not the reversible name a restarted scan
# re-derives its folders from.
_KEY_SEPARATOR = '~'


def skill_surface(skill_name):
    """Read the grid surface one installed skill declares.

    The skill is reached through its query module, the framework's only
    sanctioned interface to a skill, so a scan can be planned on a
    machine that cannot even run the skill it plans for.

    Parameters
    ----------
    skill_name : str
        Folder name of the skill ('nn_pes_skill'), as the task queue
        records it.

    Returns
    -------
    dict
        'label' (program label), 'query' (the skill's query module),
        'searchable' (keyword to its entry) and 'targets' (trainable
        profile to what a point stamps).

    Raises
    ------
    ValueError
        If no installed skill carries that folder name, or it declares
        nothing a scan can vary.
    """
    from ...skills.query import discover_all

    meta = None
    for _label, entry in (discover_all() or {}).items():
        if entry.get('skill_name') == skill_name:
            meta = entry
            break
    if meta is None:
        raise ValueError(
            "no installed skill is named '%s'; a grid scan is planned "
            "against the skill the run's task queue carries" % skill_name)
    searchable = meta.get('searchable') or {}
    targets = meta.get('search_targets') or {}
    if not searchable or not targets:
        raise ValueError(
            "skill '%s' declares no searchable hyperparameters, so there "
            "is nothing a grid scan could vary in it" % skill_name)
    from ...skills.query import skill_module_path

    module_path = skill_module_path(meta['_dir_name'])
    return {'label': meta.get('name', skill_name),
            'query': importlib.import_module(module_path),
            'query_module': module_path,
            'searchable': searchable, 'targets': targets,
            # Column aliases for the scan table, declared by the skill
            # (its own training folder names, #3.6).
            'role_aliases': dict(meta.get('scan_role_aliases') or {})}


def query_of(context):
    """Import the query module a scan context names.

    The context holds a module *path* rather than a module: a batch
    dispatch deep-copies the workflow that carries it, and a module
    object does not survive that copy (or the pickling a pooled worker
    needs). The path is resolved at the two points that ask the skill a
    question, which is where the import is actually wanted.

    Parameters
    ----------
    context : dict
        Scan context.

    Returns
    -------
    module
        The skill's query module.
    """
    return importlib.import_module(context['query_module'])


def first_skill_name(state):
    """Name the skill the scan plans for, read off the task queue.

    Parameters
    ----------
    state : PyRAIState
        Parent state of the scan.

    Returns
    -------
    str
        Folder name of the queued skill.

    Raises
    ------
    ValueError
        If the plan queued no task.
    """
    for _category, task in state.iter_tasks():
        name = task.get('skill_name')
        if name:
            return str(name)
    raise ValueError("a grid scan plans against a skill, and this run's "
                     "task queue carries none")



def point_key(values):
    """Name one point by the values it varies.

    The name is reversible: it is the only thing that identifies a
    point's model folder once the runtime directories have been
    re-assigned, so a restarted scan finds its own previous work by
    recomputing this key from the grid rather than by trusting an index.

    Parameters
    ----------
    values : dict
        Keyword to candidate value.

    Returns
    -------
    str
        'keyword=value' pairs joined by '~', in keyword order.
    """
    return _KEY_SEPARATOR.join('%s=%s' % (keyword, _token(values[keyword]))
                               for keyword in sorted(values))


def _token(value):
    """Render one candidate as a filesystem-safe token."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return '_'.join(_token(item) for item in value)
    if isinstance(value, int):
        return '%d' % value
    if isinstance(value, float):
        return '%g' % value
    return str(value).replace('/', '-').replace(' ', '').replace(
        _KEY_SEPARATOR, '+')


def plan_points(grid, searchable, query):
    """Expand a declared grid into the points of the scan.

    Every candidate asks the skill whether it admits it, so a value the
    container would reject inside a fit subprocess is refused before a
    single directory is made. Points come out in keyword-sorted order,
    which makes the same input produce the same sequence of folders.

    Parameters
    ----------
    grid : dict
        Keyword to its list of candidate values.
    searchable : dict
        Declared keywords, from skill_surface().
    query : module
        The skill's query module, for check_searchable().

    Returns
    -------
    list of dict
        One entry per point: 'values' and 'key'.

    Raises
    ------
    ValueError
        For an empty grid, an undeclared keyword, or a candidate the
        skill refuses.
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError(
            "&search grid names no hyperparameter to vary; write one "
            "keyword with its candidate list, e.g. grid: {depth: [2, 3]}")
    unknown = sorted(set(grid) - set(searchable))
    if unknown:
        raise ValueError(
            "&search grid names %s, which skill does not declare "
            "searchable; declared: %s"
            % (', '.join(unknown), ', '.join(sorted(searchable))))

    candidates = {}
    for keyword, values in grid.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                "&search grid key '%s' carries no candidate values; it "
                "needs a list of them" % keyword)
        checked = []
        for value in values:
            try:
                checked.append(query.check_searchable(keyword, value))
            except ValueError as error:
                raise ValueError(
                    "&search grid: %s" % error) from error
        candidates[keyword] = checked

    order = sorted(candidates)
    points = []
    for combo in itertools.product(*[candidates[key] for key in order]):
        values = dict(zip(order, combo))
        points.append({'values': values, 'key': point_key(values)})
    return points



def build_context(state, surface, search_section):
    """Collect everything one scan needs, read off the parent state.

    Parameters
    ----------
    state : PyRAIState
        Parent state of the scan.
    surface : dict
        Skill surface from skill_surface().
    search_section : dict
        The run's &search section as a plain dictionary.

    Returns
    -------
    dict
        Context for write_point(): 'searchable', 'profile', 'sections',
        'model_root', 'title', 'train_data' and 'members'.

    Raises
    ------
    ValueError
        If the training data the points fit is not a readable file.
    """
    tasks = list(state.iter_tasks())
    if not tasks:
        raise ValueError("a grid scan needs one task carrying a skill "
                         "configuration; the plan produced none")
    names = {task.get('skill_name') for _category, task in tasks}
    if len(names) > 1:
        raise ValueError(
            "a grid scan varies the hyperparameters of one training job; "
            "this plan queues several skills (%s) and there is no answer "
            "to which one the grid belongs to" % ', '.join(
                sorted(str(name) for name in names)))
    skill_config = (tasks[0][1].get('config') or {}).get('skill_config') or {}
    # The profile, the training mode and the member count are the
    # skill's to resolve (its own keyword names and trainable-profile
    # table); this layer asks `query.scan_requirements` and adds only
    # what is the scan's own business -- where the data is, where the
    # models go, and what the run is called.
    requirements = surface['query'].scan_requirements(skill_config)
    profile = requirements['profile']
    train_data = str(requirements['train_data'] or '')
    if not train_data or not os.path.isfile(train_data):
        raise ValueError(
            "a grid scan trains every point from %s, but train_data names "
            "no readable file; set train_data in the skill section"
            % (train_data or '(empty)'))
    title = str(getattr(getattr(getattr(state, 'pmd', None), 'control',
                                 None), 'title', '') or '')
    root = search_section.get('grid_root') or './grid-search'
    return {
        'query_module': surface['query_module'],
        'label': surface['label'],
        'searchable': surface['searchable'],
        'targets': surface['targets'],
        'profile': profile,
        'sections': surface['targets'][profile]['sections'],
        'channels': surface['targets'][profile]['channels'],
        'model_root': os.path.abspath(os.path.expanduser(root)),
        'title': title or 'NN',
        'train_data': os.path.abspath(train_data),
        'members': int(requirements['members']),
        'role_aliases': dict(surface.get('role_aliases') or {}),
    }


def point_model_dir(point, context):
    """Model directory of one point, from its key alone.

    The folder name is the skill's (`query.model_dir_name`): renaming a
    model directory is a change in the skill, not in this planner.
    """
    name = query_of(context).model_dir_name(context['title'], point['key'])
    return os.path.abspath(os.path.join(context['model_root'], name))


def write_point(state, point, context):
    """Stamp one grid point onto a materialized child state.

    Every task of the child gets the point's values in each role section
    the trained profile fits, its own model directory, the profile to
    train and no capability request: a training run reports a record, not
    a property, and leaving the parent's energy and gradient on the task
    would have the child judged against channels it never computed.

    Parameters
    ----------
    state : PyRAIState
        Child state built by Dispatcher.materialize_unit().
    point : dict
        One entry from plan_points().
    context : dict
        Scan context from build_context().

    Returns
    -------
    str
        The point's model directory, so the caller can carry the path.
    """
    key = point['key']
    directory = point_model_dir(point, context)
    for _category, task in state.iter_tasks():
        config = task.get('config')
        if not isinstance(config, dict):
            continue
        skill_config = config.get('skill_config')
        if not isinstance(skill_config, dict):
            continue
        for keyword, value in point['values'].items():
            kind = context['searchable'][keyword]['kind']
            written = [value] if kind in _GROUP_KINDS else value
            for section in context['sections']:
                entries = skill_config.setdefault(section, {})
                if isinstance(entries, dict):
                    entries[keyword] = written
        skill_config['profile'] = context['profile']
        skill_config['modeldir'] = directory
        skill_config['train_data'] = context['train_data']
        task['capabilities'] = []
    return directory
