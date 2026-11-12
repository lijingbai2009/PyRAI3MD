"""
======================================================================
The persisted field set of one task descriptor.

`planning/task._new_task` creates the descriptor, `planning/assembly`
fills the task config, and `PyRAIState.to_dict` writes what a run
leaves behind. Those three used to spell the field list separately; a
key added to the assembly silently disappeared from the dump. This
module is the one list the dump reads, and the guard in
test_task_record.py checks that an assembled task has no key outside
it.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from copy import deepcopy


# Every task field a state dump carries. `api` (the backend class) and
# `config` (handled below as a mapping) are deliberately not in the
# list: one is a runtime object and the other needs a model dump.
TASK_FIELDS = (
    'role',
    'task',
    'task_id',
    'task_slot',
    'layer_id',
    'static_id',
    'contribution',
    'region',
    'region_indices',
    'region_charge',
    'region_multiplicity',
    'skill_name',
    'capabilities',
    'multitask',
    'remotetask',
    'multipool',
    'center_index',
    'region_id',
    'provides_electronic',
    'provides_charges',
    'electronic_binding_id',
    'state_policy',
    'reference_state',
    'transform_spec',
    'skill_binding_id',
    'plan_version',
    'plan_fingerprint',
    'composite',
    'qmmm_role',
    'binding_role',
    'partition_indices',
    'probe_indices',
    'label_plan',
    'external_charges',
    'external_potential',
    'property_request_spec',
)

# Fields whose dump is always a list (never a shared reference).
LIST_FIELDS = frozenset((
    'region_indices', 'capabilities', 'partition_indices', 'probe_indices',
))

# Fields with a non-None default in the task queue.
BOOLEAN_DEFAULTS = {
    'provides_electronic': False,
    'provides_charges': False,
}
INTEGER_DEFAULTS = {'plan_version': 0}


def dump_task(task):
    """Return the persisted form of one task descriptor.

    Parameters
    ----------
    task : dict
        Task descriptor carried by PyRAIState.

    Returns
    -------
    dict
        The fields in `TASK_FIELDS` plus the config mapping, safe to
        serialize.
    """
    entry = {}
    for name in TASK_FIELDS:
        value = task.get(name)
        if name in LIST_FIELDS:
            entry[name] = list(value or [])
        else:
            entry[name] = value
    for name, default in BOOLEAN_DEFAULTS.items():
        entry[name] = bool(task.get(name, default))
    for name, default in INTEGER_DEFAULTS.items():
        entry[name] = task.get(name, default)
    config = task.get('config')
    if config is not None:
        if isinstance(config, dict):
            entry['config'] = deepcopy(config)
        elif hasattr(config, 'model_dump'):
            # Serialization failure is a defect, not a reason to drop the
            # only copy of the task configuration from a checkpoint.
            entry['config'] = config.model_dump()
    return entry
