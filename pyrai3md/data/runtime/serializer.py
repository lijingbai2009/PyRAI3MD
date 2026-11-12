"""
======================================================================
This module provides StateSerializer, the sole serialization layer
for PyRAIState. It is the ONLY module that knows the storage format
(currently pickle; future: HDF5 or other formats).

PyRAIState, Workflow, Calculator and Skill never know the format.
The dependency direction is:

    PyRAIState -> StateSerializer -> pickle / HDF5 / ...

When the format changes, only this module changes.

A runtime file is not another type of input file; it is another
source of PyRAIState. Restart and remote dispatch therefore bypass
the Interface/Schema/Planning pipeline entirely:

    New calculation:  Input -> Interface -> Schema -> Planning
                      -> build_state() -> PyRAIState
    Restart/remote:   Runtime (.rt) -> StateSerializer.load()
                      -> PyRAIState

Both branches produce exactly the same object, which is then passed
to the unchanged Workflow/Calculator execution path.

The .rt file stores the complete PyRAIState (task queue, pmd config,
runtime variables, previous results) plus the workflow runtime_idx.
Backend instances, loggers, file handles and caches are intentionally
excluded; they are reconstructed by Calculator.initialize().

Task configs are dumped to plain dictionaries before serialization:
the extended PyRAIConfig classes are created dynamically at planning
time and cannot be pickled by reference, while the skill backends
natively accept dict configs (standalone mode rebuilds the validated
config from them).

The internal serialization format (currently pickle) is hidden from
users; future implementations may switch to HDF5 or other formats
without changing the save_state/load_state interface.

The payload carries a file-level ``version`` block (the format version
plus the partition / history / electronic semantic versions) so a
restart can tell a current file from a legacy or incompatible one. A
file written before versioning has no block: it predates the
multi-center data model, so load() migrates it to the single-center
(C=1) record by rebuilding the canonical plan from the restored pmd and
back-filling the center carriers that unpickling (which bypasses
__init__) never created. On restore the persisted plan fingerprint is
checked against the fingerprint the history frames were recorded under;
a mismatch means the file is stale for the current center partition and
is refused instead of silently resumed (docs section 11, refactor R4e).

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import pickle

from pydantic import BaseModel


# File-level payload format version, bumped on any breaking change to
# the serialized layout. Persisted in every payload's ``version`` block
# so load() can tell a current file from a legacy or future one.
SERIALIZER_FORMAT_VERSION = '1.0'

# Semantic version of the per-center electronic carrier layout (the
# energy/gradient/nac/soc/density keyed dict). The partition and history
# semantic versions are read from their source-of-truth modules
# (data.center.CENTER_PLAN_VERSION, data.state.HISTORY_FRAME_VERSION) so
# they can never drift from the data model they describe.
ELECTRONIC_SEMANTICS_VERSION = '1.0'


class StaleRuntimeError(ValueError):
    """A runtime file does not match the current center partition.

    Raised by load() when the persisted plan fingerprint disagrees with
    the fingerprint the history frames were recorded under (or with an
    explicit expected fingerprint), i.e. the file is stale for this run
    and must not be silently resumed (docs section 11, refactor R4e).
    """


def _payload_version():
    """Build the file-level version block written into every payload.

    Reads the partition / history semantic versions from their
    source-of-truth modules; the import is lazy because data.state and
    data.center sit above this module and importing them at module load
    time would be circular.
    """
    from ..center import CENTER_PLAN_VERSION
    from ..state import HISTORY_FRAME_VERSION
    return {
        'format': SERIALIZER_FORMAT_VERSION,
        'partition': CENTER_PLAN_VERSION,
        'history': HISTORY_FRAME_VERSION,
        'electronic': ELECTRONIC_SEMANTICS_VERSION,
    }


# Nested PyRAIConfig submodel names flattened for skill consumption.
# Skill backends in standalone mode consume flat dictionaries; the
# same convention is used by the skill service utilities.
_SUBMODEL_NAMES = (
    'control', 'adaptive', 'dynamics', 'system',
    'properties', 'calculation', 'environment',
    'composite',
)


def _flatten_config(nested):
    """Flatten a nested config dump into a flat skill-consumable dict.

    Submodel entries (control/system/dynamics/...) are merged into
    the top level following the skill flatten convention; all other
    top-level entries (e.g. results) are preserved as is.

    Parameters
    ----------
    nested : dict
        model_dump() output of an extended PyRAIConfig.

    Returns
    -------
    dict
        Flat configuration dictionary.
    """
    flat = {}
    for name in _SUBMODEL_NAMES:
        sub_data = nested.get(name)
        if isinstance(sub_data, dict):
            flat.update(sub_data)
    for key, value in nested.items():
        if key not in _SUBMODEL_NAMES:
            flat[key] = value
    return flat


def _prepare_task(task):
    """Return a serializable copy of one task descriptor.

    Dynamic extended PyRAIConfig task configs are replaced by flat
    dictionaries; all other task fields (including the importable
    backend class in 'api') are kept unchanged.  New multi-center
    fields are back-filled with defaults so older consumers can
    read them safely.

    Parameters
    ----------
    task : dict
        Task descriptor from the state queue.

    Returns
    -------
    dict
        Copy of the task with a pickle-safe config entry.
    """
    prepared = dict(task)
    # Ensure the unified center identity fields have safe defaults
    # (docs 3.1: center_index replaces the former center_id).
    prepared.setdefault('center_index', None)
    prepared.setdefault('region_id', None)
    prepared.setdefault('plan_version', 1)
    config = prepared.get('config')
    if isinstance(config, BaseModel):
        prepared['config'] = _flatten_config(config.model_dump())
    return prepared


def _prepare_state(state):
    """Return a pickle-safe copy of a PyRAIState.

    Copies the state object and its task queues, replacing dynamic
    task configs with plain dictionaries. The original state is never
    modified.

    Parameters
    ----------
    state : PyRAIState
        State to prepare for serialization.

    Returns
    -------
    PyRAIState
        Pickle-safe state copy.
    """
    prepared = object.__new__(type(state))
    prepared.__dict__.update(state.__dict__)
    prepared.theory = [_prepare_task(t) for t in state.theory]
    prepared.reference = [_prepare_task(t) for t in state.reference]
    # The staging slot is transient: a checkpoint represents committed
    # state and commit_step is the sole writeback point (docs r4a/R6),
    # so a half-staged evaluation is never persisted. The static
    # partition_plan is immutable (a frozen dataclass) and travels by
    # reference; the mutable center carriers / history / task configs
    # are isolated by the pickle round trip itself.
    prepared._staged_evaluation = None
    # The run logger is attached to the state by the entry point, not by
    # __init__, and it owns an open file handle in the writing process.
    # A checkpoint carries state, not whoever is printing it: the reader
    # of a restored state opens its own log file.
    prepared.__dict__.pop('logger', None)
    return prepared


def _restore_task(task):
    """Back-fill unified center defaults for legacy task descriptors.

    Old .rt files written before the multi-center extension lack
    center_index, region_id and plan_version; files written before the
    center_id -> center_index rename (docs 3.1) still carry center_id.
    Loading them migrates center_id to center_index and fills the
    remaining keys with safe defaults so the rest of the pipeline
    (Calculator, composite) reads one canonical center identity.

    Parameters
    ----------
    task : dict
        Task descriptor loaded from a .rt file.

    Returns
    -------
    dict
        The same task dict (mutated in place for convenience).
    """
    # Legacy migration: move a pre-rename center_id value onto
    # center_index; an explicit center_index already present wins.
    if 'center_id' in task:
        task.setdefault('center_index', task.pop('center_id'))
    task.setdefault('center_index', None)
    task.setdefault('region_id', None)
    task.setdefault('plan_version', 0)  # legacy queue marker
    return task


def _restore_center_model(state):
    """Back-fill the multi-center data model onto a restored state.

    Unpickling rebuilds __dict__ but bypasses __init__, so a legacy .rt
    (written before the r4a center carriers existed) lacks
    partition_plan / electronic / active_surface / continuation
    entirely. A legacy file is a single-center run: it migrates to the
    C=1 record by rebuilding the canonical plan from the restored pmd
    (build_center_plan, the same call planning makes) and sizing the
    electronic carrier to it. The active surface stays null so
    get_active_configuration() falls back to the saved current_state --
    the true restart position -- instead of being reseeded to the plan's
    initial_root. For a versioned file every attribute already exists,
    so this is a no-op (docs r4e).

    Parameters
    ----------
    state : PyRAIState
        Restored state (mutated in place).

    Returns
    -------
    PyRAIState
        The same state, presenting the current center contract.
    """
    from ..center import build_center_plan, CenterPlanError

    if not hasattr(state, 'partition_plan'):
        plan = None
        pmd = getattr(state, 'pmd', None)
        if pmd is not None:
            try:
                plan = build_center_plan(pmd)
            except CenterPlanError:
                plan = None
        state.partition_plan = plan

    if not hasattr(state, 'electronic'):
        plan = state.partition_plan
        n_center = plan.n_center if plan is not None else 0
        state.electronic = [state._empty_center_electronic()
                            for _ in range(n_center)]

    if not hasattr(state, 'active_surface'):
        state.active_surface = None
    if not hasattr(state, 'continuation'):
        state.continuation = None
    if not hasattr(state, '_staged_evaluation'):
        state._staged_evaluation = None
    if not hasattr(state, 'charge_store'):
        # A legacy .rt predates ONIOM charge management; it restarts
        # without a bound store (charge management stays off).
        state.charge_store = None

    # Consistency guard: a bound plan sizes the electronic carrier to
    # exactly C entries (one independent array group per center).
    plan = state.partition_plan
    if plan is not None and len(state.electronic) != plan.n_center:
        raise ValueError(
            "Runtime file center mismatch: the partition plan has %d "
            "center(s) but the electronic carrier has %d entr(ies)."
            % (plan.n_center, len(state.electronic)))
    return state


def _check_version(version):
    """Validate a payload version block; None marks a legacy file.

    Parameters
    ----------
    version : dict or None
        The payload's version block (None for a pre-versioning file).

    Returns
    -------
    bool
        True when the file is legacy (no version block) and was handled
        by the migration path; False when it is a current file.

    Raises
    ------
    ValueError
        If a present version block is incompatible with this build.
    """
    if version is None:
        return True
    if not isinstance(version, dict):
        raise ValueError(
            "Malformed runtime version block: %r." % (version,))
    current = _payload_version()
    for key in ('format', 'partition', 'history', 'electronic'):
        stored = version.get(key)
        if stored != current[key]:
            raise ValueError(
                "Runtime file %s version %r is incompatible with this "
                "build (%r); the file cannot be resumed as-is."
                % (key, stored, current[key]))
    return False


def _check_plan_fingerprint(state, expected=None):
    """Refuse a runtime file whose plan disagrees with its own history.

    Every history frame records the plan fingerprint in force when it
    was taken (docs r4d). If a frame's fingerprint differs from the
    restored plan's, the recorded per-center electronic data belongs to
    a different partition and the file is stale. An explicit ``expected``
    fingerprint (remote dispatch, where the parent knows the plan the
    child must run) is checked the same way. Frames without a
    fingerprint (legacy, or a warmup C=1 frame) are skipped.

    Parameters
    ----------
    state : PyRAIState
        Restored state with its partition_plan and history.
    expected : str or None
        Optional fingerprint the restored plan must match.

    Raises
    ------
    StaleRuntimeError
        On any fingerprint disagreement.
    """
    plan = getattr(state, 'partition_plan', None)
    plan_fp = plan.plan_fingerprint if plan is not None else None
    if expected is not None and plan_fp != expected:
        raise StaleRuntimeError(
            "Runtime file plan fingerprint %r does not match the "
            "expected %r; the file is stale for this center partition."
            % (plan_fp, expected))
    if plan_fp is None:
        return
    for index, frame in enumerate(getattr(state, '_history', [])):
        frame_fp = (frame.get('plan_fingerprint')
                    if isinstance(frame, dict) else None)
        if frame_fp is not None and frame_fp != plan_fp:
            raise StaleRuntimeError(
                "History frame %d was recorded under plan fingerprint "
                "%r but the restored plan is %r; the file mixes center "
                "partitions and is stale." % (index, frame_fp, plan_fp))


# ---------------------------------------------------------------------------
# StateSerializer: the sole serialization layer
# ---------------------------------------------------------------------------

class StateSerializer:
    """PyRAIState serialization layer.

    The sole module that knows the storage format. Currently uses
    pickle internally; future versions may switch to HDF5 or other
    formats without changing any code outside this module.

    PyRAIState structure (fixed, explicit):
        state.runtime        -> RuntimeState
        state.theory         -> list of task dicts
        state.reference      -> list of task dicts
        state.pmd            -> PyRAIConfig (Pydantic model)
        state.execution_graph -> list of dicts or None
        state.resolve_execution_graph -> bool
        state._history       -> list of HistoryFrame dicts
        state.partition_plan -> PartitionPlan or None (static, shared)
        state.electronic     -> list of C per-center carrier dicts
        state.active_surface -> committed active surface dict or None
        state.continuation   -> staged next-step dict or None
    """

    @staticmethod
    def save(state, path, runtime_idx=None):
        """Serialize a PyRAIState to a runtime file.

        Parameters
        ----------
        state : PyRAIState
            State to serialize (task queue, pmd, runtime variables
            and previous results; backend instances are excluded).
        path : str
            Destination file path.
        runtime_idx : str or None
            Workflow runtime suffix carried by the child runtime
            (e.g. 'runtime_3'), restored by load().
        """
        prepared = _prepare_state(state)
        payload = {
            'version': _payload_version(),
            'state': prepared,
            'runtime_idx': runtime_idx,
        }
        payload['constraint'] = _constraint_payload(state)
        with open(path, 'wb') as handler:
            pickle.dump(payload, handler,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path, expected_fingerprint=None):
        """Restore a PyRAIState from a runtime file.

        Parameters
        ----------
        path : str
            Source file path.
        expected_fingerprint : str or None
            Optional center-plan fingerprint the restored plan must
            match (remote dispatch / sub-trajectory materialization,
            where the parent already knows the partition the child must
            run). A mismatch raises StaleRuntimeError instead of
            resuming on the wrong partition.

        Returns
        -------
        tuple(PyRAIState, str or None)
            The restored state and its runtime_idx. The state is
            ready for the standard execution path (Calculator
            .initialize() rebuilds the backend instances).

        Raises
        ------
        FileNotFoundError
            If the runtime file does not exist.
        ValueError
            If the file content is not a valid runtime payload, or its
            version block is incompatible with this build.
        StaleRuntimeError
            If the persisted plan fingerprint disagrees with the
            history frames recorded under it (or with
            expected_fingerprint).
        """
        with open(path, 'rb') as handler:
            payload = pickle.load(handler)

        if not isinstance(payload, dict) or 'state' not in payload:
            raise ValueError("Invalid runtime file: %s" % path)
        state = payload['state']
        # A missing version block marks a legacy file (migrated to the
        # C=1 record below); a present block must be compatible.
        _check_version(payload.get('version'))
        # Back-fill multi-center defaults for legacy task queues.
        for category in ('theory', 'reference'):
            for task in getattr(state, category, []):
                _restore_task(task)
        # Back-fill the center data model (unpickling bypasses __init__),
        # then refuse a file whose plan disagrees with its own history.
        _restore_center_model(state)
        _check_plan_fingerprint(state, expected=expected_fingerprint)
        # The constraint plan is derived data. Dropping whatever the pickle
        # carried forces it to be re-derived from the restored config, which is
        # the only way the comparison below can catch an input that changed
        # between the save and this resume.
        state._constraint_plan = None
        _check_constraint_payload(state, payload)
        return state, payload.get('runtime_idx')


def _check_constraint_payload(state, payload):
    """Refuse a checkpoint whose constraint history belongs to another run.

    Two different words collide here, so the boundary is stated: the
    fingerprint checked above belongs to the *center partition* plan
    (`PartitionPlan.plan_fingerprint`), while the one here belongs to the
    *constraint* plan (`ConstraintPlan.fingerprint`, a digest of the declared
    actions). Neither is the other's substitute.

    The constraint state travels by pickle, so save plus restore already gives
    a continuous run. What has to be checked is the case a pickle cannot:
    a file resumed after the input changed. A restarted run that re-reads a
    `&system_constraint` block with one wall added, or with the
    `velocity_constraint` keyword flipped, would otherwise continue under
    different physics while reporting the same title. An atom-count change is
    the same class of error and is caught by the plan rebuild, which indexes
    every declared atom against the restored structure.

    Raises
    ------
    ValueError
        When the restored config declares a plan that is not the one the
        checkpoint was written under, or when the stored history references
        atoms the restored system does not have.
    """
    from pyrai3md.data.constraint import PLAN_SEMANTIC_VERSION

    stored = payload.get('constraint') or {}
    if not stored:
        return
    plan = state.constraint_plan()
    if plan.is_empty:
        raise ValueError(
            'the runtime file carries a constraint history (fingerprint %s) '
            'but the restored input declares no constraints; the run would '
            'continue without the restraints that produced the file'
            % stored.get('fingerprint'))
    if int(stored.get('semantic_version', PLAN_SEMANTIC_VERSION)) != \
            PLAN_SEMANTIC_VERSION:
        raise ValueError(
            'the runtime file carries constraint semantic version %r while '
            'this build reads %r; the stored meaning has changed'
            % (stored.get('semantic_version'), PLAN_SEMANTIC_VERSION))
    if stored.get('fingerprint') != plan.fingerprint:
        raise ValueError(
            'the runtime file was written under constraint plan fingerprint '
            '%s but the restored input builds %s; a restart cannot silently '
            'change the constraints in force'
            % (stored.get('fingerprint'), plan.fingerprint))
    recorded = stored.get('state')
    if recorded:
        # The payload is the authority, not the pickled attribute: a bound
        # target re-read from the resumed geometry would be measuring the very
        # drift it exists to prevent.
        from pyrai3md.data.constraint import ConstraintState
        state.constraint_state = ConstraintState.from_dict(recorded)


def _constraint_payload(state):
    """The constraint block of one runtime file: version, fingerprint, history.

    The plan itself is not stored. It is derived from `pmd.system.constraints`,
    which the config already carries, and a second copy would give a restart two
    things to disagree about. What is stored is the digest of that derivation -
    enough to refuse a file whose input has since changed - plus the
    per-trajectory state, which nothing else can reconstruct.
    """
    from pyrai3md.data.constraint import PLAN_SEMANTIC_VERSION

    plan = state.constraint_plan()
    if plan.is_empty:
        return {}
    recorded = getattr(state, 'constraint_state', None)
    return {'semantic_version': PLAN_SEMANTIC_VERSION,
            'fingerprint': plan.fingerprint,
            'state': None if recorded is None else recorded.to_dict()}


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

def save_state(state, path, runtime_idx=None):
    """Serialize a PyRAIState to a file (convenience wrapper).

    See StateSerializer.save() for full documentation.
    """
    StateSerializer.save(state, path, runtime_idx=runtime_idx)


def load_state(path, expected_fingerprint=None):
    """Restore a PyRAIState from a file (convenience wrapper).

    See StateSerializer.load() for full documentation.
    """
    return StateSerializer.load(
        path, expected_fingerprint=expected_fingerprint)


def prepare_state(state):
    """Return a pickle-safe copy of PyRAIState (public API).

    Used by the Scheduler to prepare pickle-safe payloads for
    multiprocessing workers.
    """
    return _prepare_state(state)


def shallow_copy_state(state):
    """Shallow copy of PyRAIState (shared numpy array memory).

    Suitable for workflow transfer and temporary modification
    where the original data should not be duplicated.

    Parameters
    ----------
    state : PyRAIState
        The state object to copy.

    Returns
    -------
    PyRAIState
        Shallow copy with shared arrays.
    """
    return copy.copy(state)


def deep_copy_state(state):
    """Deep copy of PyRAIState (independent numpy arrays).

    Suitable for branch calculations, independent simulations,
    and optimization trials where data must not be shared.

    Parameters
    ----------
    state : PyRAIState
        The state object to copy.

    Returns
    -------
    PyRAIState
        Fully independent deep copy.
    """
    return copy.deepcopy(state)
