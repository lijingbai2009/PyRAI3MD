"""
======================================================================
This module provides the skill coordination mechanism for PyRAI3MD.
The Maestro discovers available skills, loads their backend modules,
resolves skill APIs, and manages skill runtime configurations.

Maestro is the single entry point for all skill-related operations:
- Skill discovery via the query mechanism (populate)
- Skill resolution from config (resolve_skill, resolve_skills)
- Skill runtime config storage (register_config, get_config)
- Backend class loading (_load_backend_class)

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import importlib

from .logger import get_logger
from .exceptions import SkillNotFoundError

logger = get_logger(__name__)

# Role sequence for theory and reference methods
_ROLE_SEQUENCE = ['qm', 'qm2', 'mm', 'mm2', 'mm3']


def _binding_key(bindings, skill_name, role):
    """The key one resolved skill is stored under.

    The skill name, unless it already holds a position of this category, in
    which case the role is appended so a composite can put one program on two
    layers. Consumers that index by name are unaffected in the ordinary case.
    """
    if skill_name not in bindings:
        return skill_name
    return '%s@%s' % (skill_name, role)


def find_binding(bindings, skill_name, role):
    """The resolved entry for one (skill, role) pair, whichever key it is under.

    A task descriptor carries the skill name and the role it was expanded from,
    which together identify the binding whether or not the key was disambiguated
    (an entry built without a `skill_name` field is keyed by that name).
    """
    for key, info in (bindings or {}).items():
        if (info.get('skill_name', key) == skill_name
                and info.get('role') == role):
            return info
    return {}


class Maestro:
    """Coordinates skill discovery, resolution, and configuration.

    The Maestro is the single coordination point for skills in PyRAI3MD.
    It combines skill discovery (query mechanism), skill resolution
    (config -> backend classes), and runtime config storage.

    Data flow:
        skills/query.py (query mechanism)
            -> Maestro.populate() (cache)
            -> resolve / resolve_skills (from cache)
    """

    def __init__(self):
        # Skill discovery cache (from Registry._skill_registry_cache)
        self._skill_cache = {}
        self._skill_dir_map = {}

        # Skill runtime config store (from Registry._configs)
        self._configs = {}

        # Legacy: skill class registry (for qualification)
        self._skills = {}
        self._skill_modules = {}
        self._discovered = []

    # ---- Skill Discovery (from Registry) ----

    def populate(self):
        """Populate skill cache via the query mechanism.

        This is the single entry point for discovering all available
        skills. It calls skills/query.discover_all() once and caches
        the result. All subsequent skill resolution goes through this
        cached data.

        Data flow:
            skills/query.py (query mechanism)
                -> Maestro.populate() (cache)
                -> resolve / resolve_skills (from cache)
        """
        from ..skills.query import discover_all
        self._skill_cache = discover_all()
        for label, meta in self._skill_cache.items():
            skill_name = meta.get('skill_name', '')
            dir_name = meta.get('_dir_name', '')
            if skill_name and dir_name:
                self._skill_dir_map[skill_name] = dir_name

    def resolve_label(self, label):
        """Resolve a method label to skill info via cached registry.

        Parameters
        ----------
        label : str
            Method label, e.g. 'molcas', 'pyscf'.

        Returns
        -------
        dict or None
            Skill metadata dict if found, None otherwise.
        """
        if not self._skill_cache:
            self.populate()
        return self._skill_cache.get(label, None)

    def get_skill_info(self, skill_name):
        """Get skill info by skill_name from cached registry.

        Parameters
        ----------
        skill_name : str
            Skill identifier, e.g. 'molcas_skill'.

        Returns
        -------
        dict or None
            Skill metadata dict if found, None otherwise.
        """
        if not self._skill_cache:
            self.populate()
        for label, meta in self._skill_cache.items():
            if meta.get('skill_name') == skill_name:
                return meta
        return None

    def is_populated(self):
        """Check whether the skill cache has been populated.

        Returns
        -------
        bool
            True if populate() has been called.
        """
        return bool(self._skill_cache)

    def list_known_skill_names(self):
        """Return the set of all known skill_name values.

        Used by validate to check skill_name validity.

        Returns
        -------
        set of str
            All known skill_name values from the cache.
        """
        if not self._skill_cache:
            self.populate()
        return {meta.get('skill_name', '') for meta in self._skill_cache.values()}

    # ---- Config Store (from Registry) ----

    def register_config(self, name, config):
        """Register a skill runtime configuration entry.

        Parameters
        ----------
        name : str
            Configuration name (skill name).
        config : dict
            Configuration data.
        """
        self._configs[name] = config

    def get_config(self, name):
        """Retrieve a registered skill runtime configuration.

        Parameters
        ----------
        name : str
            Configuration name (skill name).

        Returns
        -------
        dict
            Configuration data, or empty dict if not found.
        """
        return self._configs.get(name, {})

    # ---- Skill Resolution (from pmd.py) ----

    def resolve_skill(self, config):
        """Discover primary skill based on validated config.

        Resolution priority:
        1. config.control.skill_name  -- explicit skill identifier (agent mode)
        2. config.control.theory[0]   -- fallback: resolve label via cache
        3. ''                         -- empty, caller should raise error

        Parameters
        ----------
        config : Config
            Validated configuration model.

        Returns
        -------
        str
            Name of the skill to use (skill_name from cache).
        """
        # Access nested control submodel
        control = config.control if hasattr(config, 'control') else config

        # Priority 1: explicit skill_name (set by agent or normalize)
        skill_name = getattr(control, 'skill_name', '')
        if skill_name:
            return skill_name

        # Priority 2: try to resolve from theory[0] label
        theory = getattr(control, 'theory', []) or []
        if theory and isinstance(theory, list) and len(theory) > 0:
            from ..skills.query import method_label
            label = method_label(theory[0])
            skill_info = self.resolve_label(label)
            if skill_info:
                return skill_info['skill_name']

        # No skill found
        return ''

    def resolve_skills(self, config):
        """Resolve all skill APIs from config and plan capabilities.

        Iterates over theory methods (ordered: 1st=qm, 2nd=qm2, ...)
        and reference methods (same role sequence), resolving each to a
        Backend class. Then resolves capabilities from jobtype tokens
        and dynamics keywords.

        Parameters
        ----------
        config : Config
            Framework configuration model.

        Returns
        -------
        dict
            Nested dict {'theory': {...}, 'reference': {...}}.
            Each skill: {skill_name: {'role': str, 'backend': class,
                                       'capabilities': [str]}}.
        """
        from ..planning import resolve_capabilities
        from ..schema.validate import parse_jobtype

        # Access nested control submodel
        control = config.control if hasattr(config, 'control') else config

        skills = {'theory': {}, 'reference': {}}

        # A binding is keyed by the skill it serves - unless the same skill
        # holds two positions of one composite, in which case the role is
        # appended (`pyscf@qm2`). A plain name-keyed dict dropped every repeat
        # (`if skill_name in seen: continue`), so `theory: [pyscf, pyscf]` came
        # out as a single backend and a single-center composite never built its
        # task queue. Everything that indexes by skill name keeps working for
        # the ordinary one-binding-per-skill case; the descriptor lookup finds a
        # key through (skill_name, role), which is unique either way.
        #
        # Resolve theory methods (ordered)
        from ..skills.query import method_label
        theory_list = getattr(control, 'theory', []) or []
        for i, method_name in enumerate(theory_list):
            label = method_label(method_name)
            skill_info = self.resolve_label(label)
            if skill_info is None:
                raise ValueError(
                    f"Theory method '{method_name}' not found in registry. "
                    f"Available skills: {list(self._skill_cache.keys())}"
                )
            skill_name = skill_info['skill_name']

            role = _ROLE_SEQUENCE[i] if i < len(_ROLE_SEQUENCE) else f'theory{i+1}'
            backend_api_path = self._get_backend_api_path(skill_info)
            if backend_api_path is None:
                raise ValueError(
                    f"Cannot construct backend API path for skill '{skill_name}'."
                )
            skills['theory'][_binding_key(skills['theory'], skill_name,
                                           role)] = {
                'role': role, 'skill_name': skill_name,
                'backend': backend_api_path,
            }

        # Resolve reference methods (ordered, same role sequence)
        ref_list = getattr(control, 'reference', []) or []
        for i, ref_name in enumerate(ref_list):
            label = method_label(ref_name)
            skill_info = self.resolve_label(label)
            if skill_info is None:
                raise ValueError(
                    f"Reference method '{ref_name}' not found in registry."
                )
            skill_name = skill_info['skill_name']

            role = _ROLE_SEQUENCE[i] if i < len(_ROLE_SEQUENCE) else f'ref{i+1}'
            backend_api_path = self._get_backend_api_path(skill_info)
            if backend_api_path is None:
                raise ValueError(
                    f"Cannot construct backend API path for skill '{skill_name}'."
                )
            skills['reference'][_binding_key(skills['reference'], skill_name,
                                             role)] = {
                'role': role, 'skill_name': skill_name,
                'backend': backend_api_path,
            }

        # Fallback: if config has explicit skill_name but no theory/reference list
        if not skills['theory'] and not skills['reference']:
            skill_name = getattr(control, 'skill_name', '')
            if skill_name:
                skill_info = self.get_skill_info(skill_name)
                if skill_info:
                    backend_api_path = self._get_backend_api_path(skill_info)
                    if backend_api_path:
                        skills['theory']['qm'] = {
                            'role': 'qm', 'skill_name': skill_name,
                            'backend': backend_api_path,
                        }

        # Planning: resolve capabilities and distribute to skills
        jobtype_str = getattr(control, 'jobtype', 'energy') or 'energy'
        tokens, is_basic, is_advanced = parse_jobtype(jobtype_str)

        planning_context = self._extract_planning_context(config)

        planning_result = resolve_capabilities(
            tokens, planning_context=planning_context, skills=skills,
        )

        # Store per-skill capabilities in skills dict, under the binding's key
        backend_caps = planning_result['backend_capabilities']
        for category in ('theory', 'reference'):
            for key, info in skills[category].items():
                info['capabilities'] = backend_caps.get(
                    key, ['energy', 'gradient']
                )

        return skills

    def _get_backend_api_path(self, skill_info):
        """Construct backend API import path string.

        Parameters
        ----------
        skill_info : dict
            Skill metadata from cache.

        Returns
        -------
        str or None
            Full import path (e.g. 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend'),
            or None if construction fails.
        """
        skill_path = skill_info.get('skill_path', '')
        backend_class_name = skill_info.get('backend_class', '')
        dir_name = skill_info.get('_dir_name', '')

        if not skill_path or not backend_class_name or not dir_name:
            return None

        # Construct full import path
        from ..skills.query import skill_module_path

        full_module = skill_module_path(dir_name, skill_path)
        return f'{full_module}.{backend_class_name}'

    @staticmethod
    def _extract_planning_context(config):
        """Scan the input keywords that state what has to be computed.

        Every section listed by `planning.resolver._REQUIREMENT_KEYWORDS`
        is read here, so a need added by any workflow section reaches the
        one resolver instead of being discovered later by whichever layer
        happened to build the task (docs PLANNING 4.2). The criterion walks
        that table against this dict's keys.

        Parameters
        ----------
        config : Dynamics or Config
            Dynamics submodel or full Config.

        Returns
        -------
        dict
            Planning context: jobtype and driver, the dynamics keywords
            (sfhp, nactype, radiative, dosoc, dodipole, ensemble), the
            optimize keywords (opt_method, opt_state_i/j, optimize_cell),
            the Hessian route, and the state declaration (nstate,
            state_spin, coupling, nac_pair_map, soc_pair_map).
        """
        # Support both Dynamics submodel and full Config
        dynamics = config.dynamics if hasattr(config, 'dynamics') else config
        control = getattr(config, 'control', None)
        optimize = getattr(config, 'optimize', None)
        hessian = getattr(config, 'hessian', None)
        system = getattr(config, 'system', None)
        return {
            # Which run this is, and who plans it: a skill-driven search
            # adds no request of its own to the skill's jobtype.
            'jobtype': getattr(control, 'jobtype', '') if control else '',
            'driver': getattr(control, 'driver', 'main') if control else 'main',
            'sfhp': getattr(dynamics, 'sfhp', ''),
            'nactype': getattr(dynamics, 'nactype', ''),
            'radiative': getattr(dynamics, 'radiative', '0'),
            'dosoc': getattr(dynamics, 'dosoc', 0),
            'dodipole': getattr(dynamics, 'dodipole', 0),
            # ensemble drives the NPT stress requirement; optimize_cell
            # drives the cell-optimization stress requirement.
            'ensemble': getattr(dynamics, 'ensemble', 'nve'),
            'opt_method': getattr(optimize, 'opt_method', '')
                          if optimize is not None else '',
            'opt_state_i': getattr(optimize, 'opt_state_i', 0)
                           if optimize is not None else 0,
            'opt_state_j': getattr(optimize, 'opt_state_j', None)
                           if optimize is not None else None,
            'optimize_cell': bool(getattr(optimize, 'optimize_cell', False)
                                  if optimize is not None else False),
            # The Hessian producer/mode decide whether a multi-backend
            # queue may run the request (only main / numerical does).
            'hessian_params': (hessian.model_dump()
                               if hessian is not None else {}),
            'hess_driver': getattr(hessian, 'hess_driver', 'main')
                           if hessian is not None else 'main',
            # The state declaration, for the check that a run needing a
            # coupling names the pairs it wants computed.
            'nstate': getattr(system, 'nstate', 1) if system else 1,
            'state_spin': list(getattr(system, 'state_spin', [])
                               or []) if system else [],
            'coupling': list(getattr(system, 'coupling', [])
                             or []) if system else [],
            'nac_pair_map': list(getattr(system, 'nac_pair_map', [])
                                 or []) if system else [],
            'soc_pair_map': list(getattr(system, 'soc_pair_map', [])
                                 or []) if system else [],
        }

    # ---- Legacy: Skill Class Registry (for qualification) ----

    def register_skill(self, name, skill_class, metadata=None):
        """Register a skill with its implementing class.

        Parameters
        ----------
        name : str
            Unique skill name.
        skill_class : type
            The class implementing the skill.
        metadata : dict or None
            Optional metadata describing the skill.
        """
        self._skills[name] = {
            'class': skill_class,
            'metadata': metadata or {},
        }

    def get_skill(self, name):
        """Retrieve a registered skill class by name.

        Parameters
        ----------
        name : str
            The skill name.

        Returns
        -------
        type
            The skill class.

        Raises
        ------
        SkillNotFoundError
            If the skill is not registered.
        """
        if name not in self._skills:
            raise SkillNotFoundError(f"Skill '{name}' is not registered.")
        return self._skills[name]['class']

    def list_skills(self):
        """Return a list of all registered skill names.

        Returns
        -------
        list of str
            Registered skill names.
        """
        return list(self._skills.keys())

    def register_skill_module(self, name, module):
        """Register a skill module.

        Parameters
        ----------
        name : str
            Skill name.
        module : module
            The skill module object.
        """
        self._skill_modules[name] = module

    def get_skill_module(self, name):
        """Retrieve a registered skill module.

        Parameters
        ----------
        name : str
            Skill name.

        Returns
        -------
        module
            The skill module.

        Raises
        ------
        SkillNotFoundError
            If the skill module is not registered.
        """
        if name not in self._skill_modules:
            raise SkillNotFoundError(
                f"Skill module '{name}' is not registered.")
        return self._skill_modules[name]

    def list_skill_modules(self):
        """Return a list of all registered skill module names.

        Returns
        -------
        list of str
            Registered skill module names.
        """
        return list(self._skill_modules.keys())

    # ---- Legacy: Maestro Discovery (for qualification) ----

    def discover(self, search_paths=None):
        """Discover available skills from the skills directory.

        Parameters
        ----------
        search_paths : list of str or None
            Directories to search for skills. If None, uses
            the global skill registry via list_skills().

        Returns
        -------
        list of str
            Names of discovered skills.
        """
        if search_paths is None:
            from ..skills.query import list_skills
            discovered = list_skills()
        else:
            discovered = []
            for search_dir in search_paths:
                if not os.path.isdir(search_dir):
                    continue
                for name in os.listdir(search_dir):
                    skill_dir = os.path.join(search_dir, name)
                    manifest = os.path.join(skill_dir, 'manifest.yaml')
                    if os.path.isdir(skill_dir) and os.path.exists(manifest):
                        if name not in discovered:
                            discovered.append(name)

        self._discovered = discovered
        logger.info("Discovered skills: %s", discovered)
        return discovered

    def load_skill(self, name):
        """Load a single skill by name.

        Parameters
        ----------
        name : str
            Skill name (e.g. 'molcas_skill').

        Returns
        -------
        bool
            True if the skill was loaded successfully.
        """
        from ..skills.query import skill_module_path

        try:
            skill_module = importlib.import_module(
                skill_module_path(name, ''))

            # The backend class comes from the skill's manifest
            # (`backend.class`, read through query.get_metadata). The two
            # fallbacks that used to stand here -- a module-level `Backend`
            # attribute, then a name built from the package name -- could
            # only fire when the manifest was wrong, and they turned that
            # into a silently loaded skill with a guessed class.
            try:
                skill_query = importlib.import_module(
                    skill_module_path(name))
                meta = skill_query.get_metadata()
            except (ImportError, AttributeError):
                meta = {}
            backend_class_name = str(meta.get('backend_class') or '')
            backend_class = (getattr(skill_module, backend_class_name, None)
                             if backend_class_name else None)

            if backend_class is None:
                logger.warning("No backend class found for skill: %s", name)
                return False

            self.register_skill(name, backend_class)
            logger.info("Loaded skill: %s", name)

            self.register_skill_module(name, skill_module)
            return True

        except (ImportError, AttributeError) as e:
            logger.warning("Failed to load skill '%s': %s", name, e)
            return False

    def load_all(self):
        """Load all discovered skills.

        Returns
        -------
        list of str
            Names of successfully loaded skills.
        """
        if not self._discovered:
            self.discover()

        loaded = []
        for name in self._discovered:
            if self.load_skill(name):
                loaded.append(name)

        logger.info("Loaded %d skills: %s", len(loaded), loaded)
        return loaded
