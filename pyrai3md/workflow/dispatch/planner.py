"""
======================================================================
This module provides the ExecutionGraphBuilder, the single planning
stage of the PyRAI3MD runtime.

Graph construction (root runtime, resolve_execution_graph=True):
1. The jobtype resolves the primary workflow.
2. The graph is a node chain assembled from the workflow's node-chain
   template (WORKFLOW_NODE_DECLARATIONS / WORKFLOW_BASE_CHAINS):
   - workflow nodes: the iterative orchestrators (md, hessian, ...);
     multi-stage workflows (adaptive) carry their stage sub-nodes.
   - base nodes: single-shot execution wrapping the calculator, one
     per backend role (base.theory / base.reference); each carries a
     backend_execution descriptor, the finer-grained backend_terms
     contract (serial / local thread pool / remote per-term workers)
     consumed by the calculator term execution.
3. Every node resolves its multitask/remotetask/multipool triple
   through the same priority chain:
   - declaration dict (stage_defaults / node-chain template): highest
     priority, a key that appears forces its value;
   - input keywords: workflow nodes read the workflow's own keyword
     section (&adaptive -> state.pmd.adaptive, &md ->
     state.pmd.dynamics, ...); base nodes read the dispatch-layer
     keywords of the skill sections carried by the task queue;
   - class-level defaults: the fallback.
   The lookup key of all three layers is the node NAME: a root
   workflow node takes it from the class-level ``name`` the workflow
   declares (never from the class spelling, which can differ), a
   stage node takes it from its declared stage name. A key in the
   tables that no node can reach is dead settings, and a workflow
   whose name is absent from the tables silently resolves no keyword
   section; the verification suite guards both directions.
4. The frozen graph is stored into state.execution_graph.

Worker runtimes (resolve_execution_graph=False) never build a graph:
they only verify the parent-frozen graph against the universal rules
(workers and base nodes never carry remotetask=True).

Universal rules enforced here:
- Rule 2: a worker execution graph may not contain remotetask=True.
- Rule 3: base nodes are always local (remotetask=False); remote
  single-point execution is controlled by the skill.

Aug 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# Modularized workflow keyword sections: workflow name -> PyRAIConfig
# submodel carrying the multitask/remotetask/multipool keywords
# (&adaptive, &md, &optimize, &hessian, &search). Workflows without
# a section keep their class defaults.
WORKFLOW_SECTIONS = {
    'adaptive': 'adaptive',
    'md': 'dynamics',
    'optimize': 'optimize',
    # The path family is asked for as `optimize.method`, so its node reads the
    # same &optimize section the single-structure loop does - one section, one
    # set of thresholds, and `opt_method` is what decides which of the two loops
    # a run gets (`workflow.resolve_workflow`).
    'path': 'optimize',
    'hessian': 'hessian',
    'search': 'search',
    # The two machine-learning nodes: their own sections are what let a
    # long fit or a test-set pass be steered (&train remotetask) instead
    # of inheriting a class default.
    'train': 'train',
    'prediction': 'prediction',
}

# Node-chain declaration layer of the workflow nodes (highest
# priority): a key that appears forces its value over the input
# keywords and the class defaults.
WORKFLOW_NODE_DECLARATIONS = {
    'adaptive': {'multitask': False},
    'md': {},
    # The node stays declared but forces no flag: a Hessian batch is 6N
    # independent displaced evaluations, which is the case multitask exists
    # for, and a default here would outrank the input's own `&hessian
    # multitask` because the declaration table wins over keywords.
    'hessian': {},
    'optimize': {'multitask': False},
    'scan': {'multitask': False},
    # The path node forces no flag. A band is N independent image evaluations per
    # round, which is the fan-out `multipool` exists for, and a declaration here
    # would outrank the section keyword `read_section_flags` hands this node -
    # `&optimize multipool`, the one dispatch keyword the optimize section
    # carries - the same reason the hessian node above forces nothing. The
    # `multitask` half is not the section's to give: it lives on `&molecule`
    # (`system`), whose tri-state reaches the node through the tasks. A walker
    # (mep, irc) never fans out, so an undeclared node costs it nothing either
    # way.
    'path': {},
    'train': {},
    'prediction': {},
    # A scan is a fan-out by construction: one point per child runtime,
    # so the node takes the user's multitask/remotetask keywords rather
    # than being pinned serial. Forcing False here would outvote the
    # &search section and make a grid scan permanently one point at a
    # time.
    'search': {},
}

# Base nodes attached to each workflow's node chain: backend role ->
# declaration dict (empty dict keeps the pure skill-section
# settings). Multi-stage workflows declare their base nodes inside
# the stage list instead (see AdaptiveSampling.stages).
WORKFLOW_BASE_CHAINS = {
    # AIMD serves both the 'md' and 'surface_hopping' jobtypes, so the
    # node name 'md' is the only entry either jobtype can reach.
    'md': {'theory': {'multitask': False}},
    'hessian': {'theory': {}},
    'optimize': {'theory': {'multitask': False}},
    'scan': {'theory': {'multitask': False}},
    # A band's per-image evaluations are independent, so the base chain takes
    # the section's settings rather than pinning the theory role serial: the
    # band fan-out is what `multipool` is for (`workflow/path.py::_run_band`).
    'path': {'theory': {}},
    'search': {'theory': {}},
    'train': {'theory': {}},
    'prediction': {'theory': {}},
}

# Base chain of workflows without a dedicated entry (property jobs and
# the single-shot node 'single', which is what the train and prediction
# jobtypes resolve to): every role keeps its skill-section settings.
DEFAULT_BASE_CHAIN = {'theory': {}, 'reference': {}}


def aggregate_tristate(values):
    """Aggregate tri-state flags of all involved skills.

    Rule (multitask and remotetask aggregated independently):
    - all None          -> False
    - {True, None}      -> True
    - {False, None}     -> False
    - {True, False}     -> KeyError (conflicting configuration)

    Parameters
    ----------
    values : list
        Tri-state values (None / True / False), one per skill.

    Returns
    -------
    bool
        Aggregated flag.

    Raises
    ------
    KeyError
        When True and False appear simultaneously.
    """
    assigned = {value for value in values if value is not None}
    if not assigned:
        return False
    if True in assigned and False in assigned:
        raise KeyError(
            "Conflicting multitask/remotetask keywords: both True "
            "and False are set across the involved skills.")
    return True in assigned


def aggregate_multipool(values):
    """Aggregate the multipool declarations of one backend role.

    Rule aligned with aggregate_tristate:
    - all None               -> 1 (serial default)
    - consistent declarations -> the declared value
    - conflicting values     -> KeyError

    Parameters
    ----------
    values : list
        Declared pool sizes (None when undeclared), one per skill.

    Returns
    -------
    int
        Aggregated pool size (>= 1).

    Raises
    ------
    KeyError
        When different pool sizes are declared simultaneously.
    """
    declared = {int(value) for value in values if value is not None}
    if not declared:
        return 1
    if len(declared) > 1:
        raise KeyError(
            "Conflicting multipool keywords: %r declared across the "
            "involved skills." % (sorted(declared),))
    return max(1, declared.pop())


def _flag_value(source, key):
    """Read one tri-state flag from a model or dict source.

    Parameters
    ----------
    source : BaseModel or dict or None
        Keyword section to read.
    key : str
        'multitask', 'remotetask' or 'multipool'.

    Returns
    -------
    bool or int or None
        Declared flag value (None when not declared/set).
    """
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def read_prefixed_flags(state, section_name, prefix):
    """Read one stage's prefixed dispatch flags from a workflow section.

    A multi-stage workflow can keep per-stage keywords in its own section
    (adaptive: md_multitask, reference_multipool, ...). This reader
    returns only the prefixed values; a workflow stage node may fall back
    to the section's generic triple, while a base node leaves its
    task-level flags in charge when no prefixed key is set.

    Parameters
    ----------
    state : PyRAIState
        State carrying the pmd configuration.
    section_name : str
        PyRAIConfig submodel name ('adaptive').
    prefix : str
        Stage prefix ('md' or 'reference').

    Returns
    -------
    tuple(bool or None, bool or None, int or None)
        (multitask, remotetask, multipool) tri-state values; None when a
        prefixed key is unset. The caller decides whether the section's
        generic triple is the fallback.
    """
    pmd = getattr(state, 'pmd', None)
    section = getattr(pmd, section_name, None) if pmd is not None else None
    if section is None:
        return None, None, None
    return (_flag_value(section, prefix + '_multitask'),
            _flag_value(section, prefix + '_remotetask'),
            _flag_value(section, prefix + '_multipool'))


def read_section_flags(state, workflow_name):
    """Read the modularized keywords of one workflow section.

    Parameters
    ----------
    state : PyRAIState
        State carrying the pmd configuration.
    workflow_name : str
        Workflow name mapped to its keyword section.

    Returns
    -------
    tuple(bool or None, bool or None, int or None)
        (multitask, remotetask, multipool) tri-state values;
        (None, None, None) when the workflow has no keyword
        section.
    """
    section_name = WORKFLOW_SECTIONS.get(workflow_name)
    if section_name is None:
        return None, None, None
    pmd = getattr(state, 'pmd', None)
    section = getattr(pmd, section_name, None) if pmd is not None else None
    return (_flag_value(section, 'multitask'),
            _flag_value(section, 'remotetask'),
            _flag_value(section, 'multipool'))


def _task_dispatch_flags(task):
    """Read dispatch-layer keywords from one task dict.

    multitask/remotetask are dispatch-layer keywords extracted from
    the raw input during task queue construction (_build_task_queue).
    They are NOT validated by the skill schema; the planner reads
    them directly from the task dict. multipool (the base node's
    multitask pool size) travels the same channel.

    Parameters
    ----------
    task : dict
        Task descriptor from state.iter_tasks().

    Returns
    -------
    tuple
        (multitask_value, remotetask_value, multipool_value) from
        the task dict.
    """
    return (task.get('multitask'), task.get('remotetask'),
            task.get('multipool'))



def resolve_flag(declared, key, keyword_value, default_value):
    """Resolve one boolean flag through the priority chain.

    Priority: declaration dict (a key that appears forces its value)
    > input keyword (None keeps the lower layers) > class default.

    Parameters
    ----------
    declared : dict
        Node declaration dict (stage_defaults / chain template).
    key : str
        'multitask' or 'remotetask'.
    keyword_value : bool or None
        Keyword override from the input section.
    default_value : bool
        Class-level default.

    Returns
    -------
    bool
        Resolved flag.
    """
    if key in declared:
        return bool(declared[key])
    if keyword_value is not None:
        return bool(keyword_value)
    return bool(default_value)


def resolve_pool(declared, keyword_value, default_value):
    """Resolve the multipool through the priority chain.

    Priority: declaration dict > input keyword > class default.

    Parameters
    ----------
    declared : dict
        Node declaration dict (stage_defaults / chain template).
    keyword_value : int or None
        Keyword override from the input section.
    default_value : int
        Class-level default.

    Returns
    -------
    int
        Resolved pool size (>= 1).
    """
    if 'multipool' in declared:
        return max(1, int(declared['multipool']))
    if keyword_value is not None:
        return max(1, int(keyword_value))
    return max(1, int(default_value))


# Backend-terms execution descriptor (r3b): the finer-grained
# execution contract of one backend role, attached to its base node
# and frozen with the execution graph.
BACKEND_TERMS_SCOPE = 'backend_terms'
DEFAULT_FAILURE_POLICY = 'fail_fast_collect_diagnostics'
VALID_EXECUTORS = ('serial', 'thread')


def _role_tasks(state, role):
    """Collect the task dicts of one backend role in queue order.

    Parameters
    ----------
    state : PyRAIState or None
        State with the static task queue.
    role : str
        Backend role ('theory' or 'reference').

    Returns
    -------
    list of dict
        Task descriptors of the role (empty when state is None or the
        role carries no task).
    """
    if state is None:
        return []
    return [task for category, task in state.iter_tasks()
            if category == role]


def _flag_source_record(flag, value, origin, detail=None):
    """Build one serializable provenance record of the source log.

    Parameters
    ----------
    flag : str
        Resolved item name (multitask/remotetask/multipool/executor/
        max_workers).
    value : object
        Chosen value.
    origin : str
        Where the value came from ('molecule', 'skill', 'default',
        'declared', 'composite', 'forced', 'worker').
    detail : dict or None
        Extra provenance (per-skill declarations, override reason).

    Returns
    -------
    dict
        JSON-serializable provenance record.
    """
    record = {'flag': flag, 'value': value, 'origin': origin}
    if detail:
        record['detail'] = detail
    return record


def resolve_molecule_flags(state, role, skill_names=None):
    """Resolve the backend-terms execution flags of one backend role.

    Priority per flag (multitask/remotetask/multipool resolved
    independently):

    1. molecule (System) explicit value -- highest; an explicit value
       also settles a conflicting skill declaration (the skills are
       not aggregated when the molecule scope already decided);
    2. the involved skills' tri-state declarations carried by the
       role task queue (aggregate_tristate / aggregate_multipool);
    3. the serial/local default (False / False / 1).

    Unlike the base-node flag reader this replaced (it dropped
    remotetask because universal rule 3 pins the *whole* base node
    local), this resolver keeps remotetask: it describes the
    finer-grained backend_terms scope where an individual term may run
    as a remote worker (r3e). The base node still stamps
    remotetask=False at its top level.

    Parameters
    ----------
    state : PyRAIState or None
        State carrying pmd.system (molecule scope) and the task queue.
    role : str
        Backend role ('theory' or 'reference').
    skill_names : list of str or None
        Skill names involved in the role; None derives them from the
        role task queue. Only used to shape the provenance log.

    Returns
    -------
    tuple(bool, bool, int, list of dict)
        (multitask, remotetask, multipool, source_log). multipool is
        >= 1; source_log records the chosen value and origin of each
        flag.

    Raises
    ------
    KeyError
        When the molecule scope leaves a boolean flag undecided and
        the involved skills declare both True and False.
    """
    del skill_names  # provenance only; the values come from the queue
    pmd = getattr(state, 'pmd', None)
    system = getattr(pmd, 'system', None)
    molecule = {
        'multitask': _flag_value(system, 'multitask'),
        'remotetask': _flag_value(system, 'remotetask'),
        'multipool': _flag_value(system, 'multipool'),
    }
    tasks = _role_tasks(state, role)
    skill_flags = {}
    for task in tasks:
        multitask, remotetask, multipool = _task_dispatch_flags(task)
        skill_flags[task.get('skill_name')] = {
            'multitask': multitask, 'remotetask': remotetask,
            'multipool': multipool,
        }

    source_log = []
    resolved = {}
    for flag in ('multitask', 'remotetask'):
        if molecule[flag] is not None:
            value = bool(molecule[flag])
            source_log.append(_flag_source_record(flag, value, 'molecule'))
        else:
            values = [item[flag] for item in skill_flags.values()]
            value = aggregate_tristate(values)
            declared = {name: item[flag]
                        for name, item in skill_flags.items()
                        if item[flag] is not None}
            source_log.append(_flag_source_record(
                flag, value, 'skill' if declared else 'default',
                {'skills': declared} if declared else None))
        resolved[flag] = value

    if molecule['multipool'] is not None:
        value = max(1, int(molecule['multipool']))
        source_log.append(_flag_source_record('multipool', value, 'molecule'))
    else:
        values = [item['multipool'] for item in skill_flags.values()]
        value = aggregate_multipool(values)
        declared = {name: item['multipool']
                    for name, item in skill_flags.items()
                    if item['multipool'] is not None}
        source_log.append(_flag_source_record(
            'multipool', value, 'skill' if declared else 'default',
            {'skills': declared} if declared else None))
    resolved['multipool'] = value

    return (resolved['multitask'], resolved['remotetask'],
            resolved['multipool'], source_log)


def backend_execution_descriptor(role, tasks, flags,
                                 composite=None, molecule=None):
    """Build the serializable backend-terms execution descriptor.

    The descriptor is the finer-grained execution contract of one
    backend role: how its individual terms (task slots) run -- serial
    in-process, in a local thread pool, or as remote per-term workers
    (r3e). It is attached to the role base node and travels with the
    frozen execution graph.

    composite.executor / composite.max_workers stay the mechanism and
    capacity entry points; the molecule (System) keywords express the
    user intent. Two contradictions fail before any backend directory,
    thread, or Slurm submission:

    - molecule.multitask explicitly False with composite.executor
      'thread' (a thread pool the user asked not to fan out);
    - molecule.multipool explicitly set and disagreeing with an
      explicitly set composite.max_workers (one capacity, two names).

    The term-level executor is orthogonal to the base node multitask
    (the outer Dispatcher fan-out): an MD base node keeps
    multitask=False while its terms may still run in a thread pool.

    Parameters
    ----------
    role : str
        Backend role ('theory' or 'reference').
    tasks : list of dict
        Role task descriptors (for the task-slot inventory).
    flags : tuple
        (multitask, remotetask, multipool, source_log) as returned by
        resolve_molecule_flags (after any declaration override).
    composite : Composite or None
        Composite mechanism/capacity section (state.pmd.composite).
    molecule : System or None
        Molecule scope section (state.pmd.system) carrying the
        explicit tri-state keywords.

    Returns
    -------
    dict
        JSON-serializable descriptor: scope, role, task_slots,
        executor, max_workers, failure_policy, multitask, remotetask,
        multipool, source.

    Raises
    ------
    ValueError
        On an unknown executor name, or an executor/multitask or
        multipool/max_workers contradiction.
    """
    multitask, remotetask, multipool, source_log = flags
    source_log = list(source_log or [])
    multipool = max(1, int(multipool or 1))

    executor = _flag_value(composite, 'executor') or 'serial'
    if executor not in VALID_EXECUTORS:
        raise ValueError(
            "Unknown composite.executor %r: expected one of %s."
            % (executor, list(VALID_EXECUTORS)))
    max_workers = max(1, int(_flag_value(composite, 'max_workers') or 1))
    composite_workers_set = (composite is not None and 'max_workers'
                             in getattr(composite, 'model_fields_set',
                                        set()))

    molecule_multitask = _flag_value(molecule, 'multitask')
    molecule_multipool = _flag_value(molecule, 'multipool')
    if molecule_multitask is False and executor == 'thread':
        raise ValueError(
            "molecule.multitask=False contradicts "
            "composite.executor='thread': a thread pool fans the "
            "backend terms out, which multitask=False forbids.")
    if (molecule_multipool is not None and composite_workers_set
            and int(molecule_multipool) != max_workers):
        raise ValueError(
            "molecule.multipool=%d disagrees with "
            "composite.max_workers=%d: both express the backend-terms "
            "worker capacity and must match."
            % (int(molecule_multipool), max_workers))

    # Effective worker capacity: the composite mechanism entry and the
    # resolved multipool intent both express capacity; the larger wins
    # when the terms fan out (thread pool or remote workers). A serial
    # local role never provisions workers.
    if executor == 'thread' or remotetask:
        effective_workers = max(multipool, max_workers)
    else:
        effective_workers = 1

    source_log.append(_flag_source_record('executor', executor, 'composite'))
    source_log.append(_flag_source_record(
        'max_workers', effective_workers, 'composite',
        {'composite_max_workers': max_workers, 'multipool': multipool}))

    return {
        'scope': BACKEND_TERMS_SCOPE,
        'role': role,
        'task_slots': [task.get('task_slot') for task in (tasks or [])],
        'executor': executor,
        'max_workers': effective_workers,
        'failure_policy': DEFAULT_FAILURE_POLICY,
        'multitask': bool(multitask),
        'remotetask': bool(remotetask),
        'multipool': multipool,
        'source': source_log,
    }


def default_backend_execution_descriptor(role=None):
    """Build the serial/local descriptor stamped on worker graphs.

    Workers execute one local calculator node (universal rules 2 and
    3): no thread fan-out and no remote per-term submission. The
    descriptor keeps the same schema as the root descriptor so the
    frozen worker graph verifies and serializes uniformly.

    Parameters
    ----------
    role : str or None
        Backend role; None for the role-agnostic worker calculator.

    Returns
    -------
    dict
        Serial/local backend-terms descriptor.
    """
    return {
        'scope': BACKEND_TERMS_SCOPE,
        'role': role,
        'task_slots': [],
        'executor': 'serial',
        'max_workers': 1,
        'failure_policy': DEFAULT_FAILURE_POLICY,
        'multitask': False,
        'remotetask': False,
        'multipool': 1,
        'source': [_flag_source_record('executor', 'serial', 'worker'),
                   _flag_source_record('max_workers', 1, 'worker')],
    }


class ExecutionGraphBuilder:
    """Build (root) or verify (worker) the execution graph.

    The builder never executes anything: it produces the immutable
    execution graph (a list of node descriptors) frozen into
    state.execution_graph.

    Parameters
    ----------
    workflow_registry : dict or None
        Mapping of workflow name to workflow class, injected by the
        workflow entry (the dispatch package never imports the
        workflow package); used to resolve multi-stage sub-workflows.
    """

    def __init__(self, workflow_registry=None):
        self.workflow_registry = dict(workflow_registry or {})

    def build(self, state, workflow_cls=None):
        """Run the planning stage for one runtime.

        Parameters
        ----------
        state : PyRAIState
            Fully initialized state.
        workflow_cls : type or None
            Resolved workflow class (None for single-shot jobs).

        Returns
        -------
        list of dict
            Frozen execution graph stored into
            state.execution_graph.

        Raises
        ------
        RuntimeError
            When a worker carries an illegal frozen graph.
        """
        if not getattr(state, 'resolve_execution_graph', True):
            return self._verify_worker_graph(state)
        graph = self._build_root(state, workflow_cls)
        state.execution_graph = graph
        return graph

    def _build_root(self, state, workflow_cls):
        """Build the execution graph of the root runtime.

        Parameters
        ----------
        state : PyRAIState
            Root runtime state.
        workflow_cls : type or None
            Resolved workflow class.

        Returns
        -------
        list of dict
            Node chain (workflow node with optional stage sub-nodes,
            then the base nodes of the chain; base nodes are skipped
            for roles without tasks).
        """
        graph = []
        workflow_name = None
        if workflow_cls is not None:
            node = self._workflow_node(state, workflow_cls)
            graph.append(node)
            workflow_name = node['name']
            if 'stages' in node:
                # Multi-stage chains declare their base nodes inside
                # the stage list (adaptive).
                return graph
        for role, declared in self._base_chain(state, workflow_name):
            if self._role_has_tasks(state, role):
                graph.append(self._base_node(state, role, declared))
        return graph

    def _base_chain(self, state, workflow_name):
        """Resolve the base-node chain of one workflow.

        The node-chain template supplies the declared roles; every
        extra role present in the task queue is appended with an
        empty declaration (pure skill-section settings).

        Parameters
        ----------
        state : PyRAIState or None
            Root runtime state.
        workflow_name : str or None
            Workflow node name (None for single-shot jobs).

        Returns
        -------
        list of tuple(str, dict)
            (role, declaration dict) pairs in chain order.
        """
        chain = dict(WORKFLOW_BASE_CHAINS.get(
            workflow_name, DEFAULT_BASE_CHAIN))
        if state is not None:
            for category, _ in state.iter_tasks():
                chain.setdefault(category, {})
        return list(chain.items())

    @staticmethod
    def _role_has_tasks(state, role):
        """Check whether the task queue carries tasks of one role.

        Parameters
        ----------
        state : PyRAIState or None
            Root runtime state.
        role : str
            Backend role ('theory' or 'reference').

        Returns
        -------
        bool
            True when at least one task of the role exists.
        """
        if state is None:
            return False
        for category, _ in state.iter_tasks():
            if category == role:
                return True
        return False

    def _workflow_node(self, state, workflow_cls):
        """Build one workflow node descriptor.

        Parameters
        ----------
        state : PyRAIState
            Root runtime state.
        workflow_cls : type
            Workflow class; its declared class-level ``name`` is the
            node name, and the flag tables are keyed on it.

        Returns
        -------
        dict
            Workflow node descriptor (multi-stage workflows carry
            their stage sub-nodes).
        """
        name = workflow_cls.name
        declared = dict(WORKFLOW_NODE_DECLARATIONS.get(name, {}))
        (keyword_multitask, keyword_remotetask,
         keyword_multipool) = read_section_flags(state, name)
        multitask = resolve_flag(
            declared, 'multitask', keyword_multitask,
            getattr(workflow_cls, 'multitask', False))
        remotetask = resolve_flag(
            declared, 'remotetask', keyword_remotetask,
            getattr(workflow_cls, 'remotetask', False))
        multipool = resolve_pool(
            declared, keyword_multipool,
            getattr(workflow_cls, 'multipool', 1))
        node = {
            'node': 'workflow',
            'name': name,
            'kind': getattr(workflow_cls, 'workflow_kind', 'single'),
            'multitask': multitask,
            'remotetask': remotetask,
            'multipool': multipool,
        }
        stages = getattr(workflow_cls, 'stages', None)
        if stages:
            stage_defaults = getattr(workflow_cls, 'stage_defaults', {}) or {}
            stage_sources = getattr(
                workflow_cls, 'stage_flag_sources', {}) or {}
            stage_remote = getattr(
                workflow_cls, 'stage_remote_fields', {}) or {}
            node['stages'] = []
            for stage_name in stages:
                if stage_name.startswith('base.'):
                    role = stage_name.split('.', 1)[1]
                    if not self._role_has_tasks(state, role):
                        continue
                    node['stages'].append(self._base_node(
                        state, role, stage_defaults.get(stage_name),
                        flag_source=stage_sources.get(stage_name)))
                else:
                    node['stages'].append(self._stage_node(
                        state, stage_name, stage_defaults,
                        flag_source=stage_sources.get(stage_name),
                        remote_field=stage_remote.get(stage_name)))
        return node

    def _stage_node(self, state, stage_name, stage_defaults=None,
                    flag_source=None, remote_field=None):
        """Build one sub-workflow (stage) node descriptor.

        Parameters
        ----------
        state : PyRAIState
            Root runtime state.
        stage_name : str
            Stage name resolved through the injected registry.
        stage_defaults : dict or None
            Parent workflow stage-level declaration dicts
            ({stage_name: {...}}); a key that appears forces its
            value over the input keywords and the stage class
            defaults.

        Returns
        -------
        dict
            Stage node descriptor (registry-missing stages keep the
            declared values so graph construction never fails).
        """
        declared = dict((stage_defaults or {}).get(stage_name, {}))
        stage_cls = self.workflow_registry.get(stage_name)
        if stage_cls is None:
            return {
                'node': 'workflow',
                'name': stage_name,
                'kind': 'single',
                'multitask': bool(declared.get('multitask', False)),
                'remotetask': bool(declared.get('remotetask', False)),
                'multipool': max(1, int(declared.get('multipool', 1))),
            }
        if flag_source is None:
            (keyword_multitask, keyword_remotetask,
             keyword_multipool) = read_section_flags(state, stage_name)
        else:
            section_name, prefix = flag_source
            (keyword_multitask, keyword_remotetask,
             keyword_multipool) = read_prefixed_flags(
                 state, section_name, prefix)
            if (keyword_multitask is None and keyword_remotetask is None
                    and keyword_multipool is None):
                section = getattr(getattr(state, 'pmd', None),
                                  section_name, None)
                (keyword_multitask, keyword_remotetask,
                 keyword_multipool) = (
                    _flag_value(section, 'multitask'),
                    _flag_value(section, 'remotetask'),
                    _flag_value(section, 'multipool'))
        if remote_field is not None:
            section_name, key = remote_field
            section = getattr(getattr(state, 'pmd', None),
                              section_name, None)
            value = _flag_value(section, key)
            if value is not None:
                keyword_remotetask = bool(value)
        multitask = resolve_flag(
            declared, 'multitask', keyword_multitask,
            getattr(stage_cls, 'multitask', False))
        remotetask = resolve_flag(
            declared, 'remotetask', keyword_remotetask,
            getattr(stage_cls, 'remotetask', False))
        multipool = resolve_pool(
            declared, keyword_multipool,
            getattr(stage_cls, 'multipool', 1))
        return {
            'node': 'workflow',
            'name': stage_name,
            'kind': getattr(stage_cls, 'workflow_kind', 'single'),
            'multitask': multitask,
            'remotetask': remotetask,
            'multipool': multipool,
        }

    def _base_node(self, state, role, declared=None, flag_source=None):
        """Build one base node descriptor (always local).

        The base node wraps the calculator single-shot execution of
        one backend role. Its top-level multitask/remotetask/multipool
        triple is resolved through resolve_molecule_flags (molecule
        scope first, then the skill-section keywords of the role
        tasks) and overridden by the declaration dict (a declared
        multitask=False forces serial execution and pins multipool to
        1). Universal rule 3 pins the top-level remotetask=False: the
        *whole* base node runs locally.

        The node also carries a backend_execution descriptor: the
        finer-grained backend_terms contract (serial / local thread
        pool / remote per-term workers) consumed by the calculator
        term execution. Its executor/max_workers come from the
        composite mechanism entry and are orthogonal to the top-level
        multitask (an MD base node keeps multitask=False while its
        terms may still run in a thread pool).

        Parameters
        ----------
        state : PyRAIState or None
            State with the static task queue.
        role : str
            Backend role ('theory' or 'reference').
        declared : dict or None
            Node declaration dict (highest priority).

        Returns
        -------
        dict
            Base node descriptor with the backend_execution
            descriptor attached.
        """
        declared = dict(declared or {})
        tasks = _role_tasks(state, role)
        (multitask, remotetask, multipool,
         source_log) = resolve_molecule_flags(state, role)
        # A multi-stage parent may own this role's fan-out keywords in its
        # own section (adaptive: reference_multitask/reference_multipool).
        # The source is read before the declaration dict so a declared
        # stage value still wins.
        if flag_source is not None:
            keyword_multitask, _keyword_remotetask, keyword_multipool = (
                read_prefixed_flags(state, *flag_source))
            if keyword_multitask is not None:
                multitask = bool(keyword_multitask)
                if not multitask:
                    multipool = 1
                source_log.append(_flag_source_record(
                    'multitask', multitask, 'workflow section'))
            if keyword_multipool is not None:
                multipool = max(1, int(keyword_multipool))
                source_log.append(_flag_source_record(
                    'multipool', multipool, 'workflow section'))
        # Declaration dict (node-chain template / stage defaults) is
        # the highest priority: a key that appears forces its value.
        if 'multitask' in declared:
            multitask = bool(declared['multitask'])
            if not multitask:
                multipool = 1
            source_log.append(_flag_source_record(
                'multitask', multitask, 'declared'))
        if 'multipool' in declared:
            multipool = max(1, int(declared['multipool']))
            source_log.append(_flag_source_record(
                'multipool', multipool, 'declared'))
        if 'remotetask' in declared:
            remotetask = bool(declared['remotetask'])
            source_log.append(_flag_source_record(
                'remotetask', remotetask, 'declared'))

        pmd = getattr(state, 'pmd', None)
        descriptor = backend_execution_descriptor(
            role, tasks,
            (multitask, remotetask, multipool, source_log),
            composite=getattr(pmd, 'composite', None),
            molecule=getattr(pmd, 'system', None))
        # Universal rule 3: the whole base node is always local; remote
        # single-point execution (local run vs slurm run) is decided
        # by the skill itself through execution_mode. The
        # backend_terms descriptor may still route individual terms to
        # remote workers (r3e).
        return {
            'node': 'base',
            'name': 'base.%s' % role,
            'role': role,
            'multitask': multitask,
            'remotetask': False,
            'multipool': multipool,
            'backend_execution': descriptor,
        }

    def _verify_worker_graph(self, state):
        """Verify the parent-frozen graph of a worker runtime.

        Workers never re-plan: the frozen graph must exist and may
        not contain remotetask=True anywhere (universal rule 2).

        Parameters
        ----------
        state : PyRAIState
            Worker runtime state.

        Returns
        -------
        list of dict
            The frozen execution graph.

        Raises
        ------
        RuntimeError
            When the graph is missing or carries remotetask=True.
        """
        graph = getattr(state, 'execution_graph', None)
        if not graph:
            raise RuntimeError(
                "Worker runtime (resolve_execution_graph=False) "
                "requires a parent-frozen execution graph.")
        for node in self._iter_nodes(graph):
            if node.get('remotetask'):
                raise RuntimeError(
                    "Worker runtime rejects the frozen graph: workers "
                    "never set remotetask=True (node %r)." % (
                        node.get('name'),))
        return graph

    @staticmethod
    def _iter_nodes(graph):
        """Iterate over all nodes including nested stage nodes.

        Parameters
        ----------
        graph : list of dict
            Execution graph.

        Yields
        ------
        dict
            Every node descriptor (depth first).
        """
        for node in graph:
            yield node
            for stage in node.get('stages', []) or []:
                yield stage

    def __repr__(self):
        return "ExecutionGraphBuilder()"
