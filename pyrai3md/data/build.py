"""
======================================================================
Build a PyRAIState from a validated framework Config.

This is the composition root: it creates the state object, asks Maestro
for the resolved skill bindings, calls the planning-side assembly, and
binds the per-atom charge store. The planning work itself lives in
planning/assembly.py.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from ..kernel.environment import Environment
from .state import PyRAIState
from .charges import (
    ChargeStore, embedding_region_by_center, managed_region_indices,
)


def _init_charge_store(state, config):
    """Bind a ChargeStore for multiscale charge management.

    ONIOM3 binds the store when `embedding` (compute and embed) or
    `read_charge` (load '{title}.charge') is requested; its managed
    region is highlevel ∪ midlevel, and read_charge fills the midlevel.
    A single-center ONIOM2 with a quantum outer method binds it for its
    outer-region embedding (read_charge fills all atoms outside
    highlevel; otherwise the low full-system term fills them).
    QMMM always binds one over the highlevel so the QM 0-order charges
    (`q_qm = point_charges[highlevel]`) have a single home.

    A whole-system `system.point_charges` seeds the store first (row i
    is atom i); the outer calculation / read_charge then overwrite the
    regions they own. A run with neither a multiscale partition nor
    supplied charges leaves state.charge_store None so its behavior is
    unchanged.

    Parameters
    ----------
    state : PyRAIState
        State with its partition plan already bound.
    config : PyRAIConfig
        Validated configuration (system and control keywords).
    """
    system = getattr(config, 'system', None)
    if system is None:
        return
    embedding = bool(getattr(system, 'embedding', False))
    read_charge = bool(getattr(system, 'read_charge', False))
    multiscale = str(getattr(getattr(config, 'control', None),
                             'multiscale_type', '') or '').strip().lower()
    plan = getattr(state, 'partition_plan', None)
    if plan is None:
        return
    qmmm = plan.n_layer == 2 and multiscale == 'qmmm'
    oniom2 = (plan.n_layer == 2 and not qmmm
              and (embedding or read_charge))
    if plan.n_layer != 3 and not qmmm and not oniom2:
        return
    if plan.n_layer == 3 and not embedding and not read_charge:
        return
    high_centers = [list(plan.center(c).high_indices)
                    for c in range(plan.n_center)]
    mid_centers = [list(plan.center(c).mid_indices)
                   for c in range(plan.n_center)]
    natom = int(getattr(system, 'natom', 0)
                or len(getattr(system, 'coordinates', []) or [])
                or len(getattr(system, 'atoms', []) or []))
    if natom <= 0:
        # No geometry loaded yet: cannot size the per-atom array.
        return
    high_flat = [int(i) for group in high_centers for i in group]
    if qmmm:
        managed = sorted(set(high_flat))
    elif oniom2:
        # The outer region (all atoms outside highlevel) is the
        # embedding source; the highlevel is updated by theory[0].
        managed = list(range(natom))
    else:
        managed = managed_region_indices(high_centers, mid_centers)
    store = ChargeStore(natom, managed_indices=managed)
    point_charges = list(getattr(system, 'point_charges', []) or [])
    if len(point_charges) == natom:
        store.update(range(natom),
                     [float(row[0]) for row in point_charges])
    if qmmm:
        q_qm = [float(value) for value in
                (getattr(system, 'q_qm', []) or [])]
        if q_qm and len(q_qm) == len(high_flat) and not point_charges:
            store.update(high_flat, q_qm)
    elif oniom2 and read_charge:
        title = str(getattr(getattr(config, 'control', None), 'title', '') or '')
        high = set(high_flat)
        outer = [index for index in range(natom) if index not in high]
        store.load_file('%s.charge' % title, outer)
    elif read_charge:
        title = str(getattr(getattr(config, 'control', None), 'title', '') or '')
        shells = embedding_region_by_center(mid_centers)
        mid_indices = []
        for c in sorted(shells):
            mid_indices.extend(shells[c])
        store.load_file('%s.charge' % title, mid_indices)
    state.charge_store = store


def build_state(config, raw_config=None):
    """Create fully initialized PyRAIState from validated Config.

    Internally:
    1. Get Maestro from Environment
    2. Resolve skills via Maestro.resolve_skills()
    3. Resolve the task queue (composite expansion) and for each
       task: extend PyRAIConfig (no backend instantiation; backends
       are runtime objects created by Calculator.initialize())
    4. Register tasks to PyRAIState
    5. Attach the framework PyRAIConfig as state.pmd (the sole
       runtime data source: jobtype, system and results are all
       read from it through read-only properties)

    Parameters
    ----------
    config : PyRAIConfig
        Validated configuration.
    raw_config : dict, optional
        Original raw configuration from interface layer.
        Passed through to skill backend initialization for
        extracting skill-specific settings.

    Returns
    -------
    PyRAIState
        Fully initialized state with task queues (no backend
        instances yet).
    """
    env = Environment()
    maestro = env.get_maestro()

    state = PyRAIState()

    # Framework PyRAIConfig as the sole runtime data source
    # (jobtype/system/results are derived from it via properties)
    state.pmd = config

    # Resolve task queue and register tasks
    skills = maestro.resolve_skills(config)
    from ..planning.assembly import build_task_queue
    build_task_queue(state, config, skills, maestro, raw_config=raw_config)

    # The adaptive label plan is carried by the reference tasks; surface
    # it on the state so the run log can record the correspondence once.
    for task in getattr(state, 'reference', []) or []:
        plan = task.get('label_plan')
        if plan:
            state.method_alignment = dict(plan)
            break

    # Bind the per-atom charge store when ONIOM electrostatic embedding
    # or a static read_charge field is requested (docs charge mgmt).
    _init_charge_store(state, config)

    return state
