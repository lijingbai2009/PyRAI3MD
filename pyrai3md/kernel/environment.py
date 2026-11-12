"""
======================================================================
This module initializes the PyRAI3MD runtime environment before task
execution. It is the single bootstrap entry point that satisfies all
prerequisites for downstream skill resolution and loading.

It locates and loads package-level and user-level configuration files,
ensures the user configuration exists on first run, combines the two
layers, creates and populates the Maestro via the query mechanism,
and stores skill runtime configurations in the Maestro.

After Environment.initialize(), downstream code obtains:
  - env.get_config()  → combined config for schema/adapter
  - env.get_maestro() → populated Maestro for skill resolution

The environment module does NOT parse user input files, validate
task parameters, or perform the full three-tier configuration
resolution. Configuration resolution (merging environment defaults
with runtime input) is handled by schema/adapter.py.

Jul 31 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from ..utilities.config_merge import deep_merge
import copy
import os

import yaml


# User configuration directory name
_USER_DIR = '.pyrai3md'
_USER_CONFIG = 'pmd_env.yaml'

# Template for first-time user configuration (fallback if pip install was skipped)
_USER_CONFIG_TEMPLATE = """\
# PyRAI3MD User Configuration
# Override package defaults here.
# Priority: Runtime Input > This File > Package Config
# Note: This file should have been created during pip install.
# If not, run: pip install pyrai3md

# environment:
#   scratch: /scratch/user
#   export:
#     OMP_NUM_THREADS: [64]

# skills:
#   molcas_skill:
#     molcas_path: /opt/openmolcas
#     molcas_mem: 8000
"""


# Module-level singleton instance
_instance = None

# The slurm directives whose value is a flag rather than a string, and the
# text spellings of "off" the classic input may carry for them.
_BOOL_SLURM_KEYS = frozenset({'exclusive'})
_FALSE_TOKENS = frozenset({'false', 'no', 'off', '0', 'none', ''})


class Environment:
    """PyRAI3MD runtime environment bootstrap.

    Single entry point that satisfies all prerequisites for downstream
    skill resolution and loading:
      - Loads package and user configurations
      - Creates and populates Maestro via query mechanism
      - Stores skill runtime configs in Maestro

    After initialize(), downstream code obtains:
      - env.get_config()  → combined config for schema/adapter
      - env.get_maestro() → populated Maestro for skill resolution

    Does NOT parse input, validate parameters, or resolve configuration.
    Configuration resolution is handled by schema/adapter.py.

    Environment is a singleton: multiple Environment() calls return
    the same instance, and initialize() is idempotent.

    Attributes
    ----------
    package_config : dict
        Configuration loaded from pyrai3md/pmd_env.yaml.
    user_config : dict
        Configuration loaded from ~/.pyrai3md/pmd_env.yaml.
    combined_config : dict
        Combined configuration (user overrides package).
    """

    def __new__(cls):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
        return _instance

    def __init__(self):
        if hasattr(self, '_init_done'):
            return
        self.package_config = {}
        self.user_config = {}
        self.combined_config = {}
        self._disk_config = None
        self._maestro = None
        self._initialized = False
        self._init_done = True

    def initialize(self):
        """Main entry point: locate, load, ensure user config, populate Maestro.

        Three-layer config loading:
          1. qualification/pmd_env.yaml (package defaults)
          2. ~/.pyrai3md/pmd_env.yaml (user overrides)
          3. Runtime input (config.json / input file) — handled by adapter

        After initialize(), the Environment satisfies all prerequisites
        for downstream skill resolution and loading:
          - Config files loaded and merged (get_config())
          - Maestro populated with skill discovery (get_maestro())
          - Skill runtime configs stored in Maestro

        This method is idempotent. Calling it multiple times has no
        additional effect after the first successful call.
        """
        if self._initialized:
            return

        # Step 1: Load package config (qualification/pmd_env.yaml)
        package_path = _locate_package_config()
        self.package_config = _load_yaml(package_path)

        # Step 2: Load user config (~/.pyrai3md/pmd_env.yaml)
        user_path = _locate_user_config()
        if not os.path.isfile(user_path):
            _ensure_user_config(user_path)
        self.user_config = _load_yaml(user_path)

        # Step 3: Combine package + user (user overrides package)
        self.combined_config = _combine_configs(
            self.package_config, self.user_config
        )
        # The on-disk state, kept aside as what an input's overrides are
        # rolled back to. See `reset_input_overrides`.
        self._disk_config = copy.deepcopy(self.combined_config)

        # Step 4: Create and populate Maestro with skill discovery + config
        from .maestro import Maestro
        self._maestro = Maestro()
        self._maestro.populate()

        # Store skill runtime config in Maestro
        skills_config = self.combined_config.get('skills', {})
        for skill_name, skill_conf in skills_config.items():
            if isinstance(skill_conf, dict):
                self._maestro.register_config(skill_name, skill_conf)

        self._initialized = True

    def get_config(self):
        """Return combined environment dict for schema/adapter to resolve.

        Returns
        -------
        dict
            Combined configuration (package + user, user takes priority).
        """
        return copy.deepcopy(self.combined_config)

    def get_maestro(self):
        """Return the populated Maestro instance.

        The Maestro contains all discovered skills and their runtime
        configurations. Available after initialize() is called.

        Returns
        -------
        Maestro
            The populated Maestro instance.
        """
        return self._maestro

    def get_skills(self):
        """Return skills section from combined config.

        Returns
        -------
        dict
            Skills configuration dictionary.
        """
        return copy.deepcopy(self.combined_config.get('skills', {}))

    def get_slurm(self):
        """Return slurm section from combined config.

        Returns
        -------
        dict
            SLURM configuration dictionary.
        """
        return copy.deepcopy(self.combined_config.get('slurm', {}))

    def apply_slurm_overrides(self, overrides):
        """Merge a runtime input's `slurm` section into the top-level one.

        The Slurm directives are a **top-level** section (`get_slurm` reads
        `combined_config['slurm']`), because they describe the allocation a
        dispatched child asks the cluster for, not the shell a process starts
        in. `apply_runtime_overrides` writes into the `environment` section,
        which is a different dictionary: routing a `&slurm` block through it
        would leave the section read by the job-script builder untouched, so
        the input would read as configured while every child job went out on
        cluster defaults.

        Merge is additive and per key, in the documented priority order
        (package < ~/.pyrai3md < input), so an input that pins one directive
        keeps the rest of the host's section.

        A directive is a scalar or a boolean. The classic text states every
        value as a string, so the one boolean directive is read back as one:
        `exclusive: false` written as text would otherwise be truthy and ask
        the cluster to reserve the whole node.

        Parameters
        ----------
        overrides : dict
            The input's `slurm` section.
        """
        if not isinstance(overrides, dict) or not overrides:
            return
        section = self.combined_config.setdefault('slurm', {})
        if not isinstance(section, dict):
            section = {}
            self.combined_config['slurm'] = section
        for key, value in overrides.items():
            name = str(key)
            if name in _BOOL_SLURM_KEYS and isinstance(value, str):
                value = value.strip().lower() not in _FALSE_TOKENS
            section[name] = copy.deepcopy(value)

    def get_export(self):
        """Return environment export variables.

        Returns
        -------
        dict
            Export variables from the environment section.
        """
        env_section = self.combined_config.get('environment', {})
        return copy.deepcopy(env_section.get('export', {}))

    def get_shell(self):
        """Return the normalized shell section for dispatch scripts.

        The shell content is copied verbatim into generated
        submission scripts (module loads, activation commands, ...).
        Sources differ in shape, so all forms are normalized here:
        - YAML list of commands (pmd_env.yaml)
        - Single-line string with ';' separators (classic input
          '&environment shell:', which cannot span multiple lines)
        - Multi-line string

        Returns
        -------
        str
            One command per line; '' when unset.
        """
        env_section = self.combined_config.get('environment', {}) or {}
        shell = env_section.get('shell')

        fragments = shell if isinstance(shell, list) else [shell]
        commands = []
        for fragment in fragments:
            if fragment is None:
                continue
            for piece in str(fragment).split(';'):
                piece = piece.strip()
                if piece:
                    commands.append(piece)

        return '\n'.join(commands)

    def get_environment_variables(self):
        """Return the flattened environment export variables.

        Returns
        -------
        dict
            Export variable name -> string value.
        """
        flat = {}
        for key, value in self.get_export().items():
            if isinstance(value, (list, tuple)):
                if value:
                    flat[str(key)] = str(value[0])
            elif value is not None:
                flat[str(key)] = str(value)
        return flat

    def reset_input_overrides(self):
        """Roll the merged config back to what the disk files state.

        `Environment` is a process singleton and `initialize` is
        idempotent, so `combined_config` outlives the run that wrote into
        it through `apply_runtime_overrides` or `apply_slurm_overrides`.
        The second input built in the same process then reads the first
        input's directives as its own host defaults, which is the priority
        chain inverted: an input cannot override a setting it never
        declared. Every real route hits this -- `agent.py` builds a config
        per instance, and `lammps_skill`'s bringup builds a cold and a warm
        one in the same process and compares them -- so the merge hooks
        reset to the on-disk state before applying what the current input
        declares.

        Only the sections the two merge hooks write are involved; the skill
        runtime configs registered into Maestro during `initialize` are not
        part of this dictionary and stay registered.
        """
        if self._disk_config is None:
            return
        self.combined_config = copy.deepcopy(self._disk_config)

    def apply_runtime_overrides(self, overrides):
        """Apply runtime input overrides to the environment section.

        Additive hook for the classic input '&environment' section
        (and config.json environment entries): shell, program paths
        and free-form keys land here so the dispatch subsystem reads
        a single merged configuration source. The 'export' dict is
        merged instead of replaced. The retired dispatch keys are
        refused by the adapter before reaching this hook
        (schema.config.RETIRE_ENVIRONMENT_KEYS).

        Parameters
        ----------
        overrides : dict
            Environment key-value pairs from the runtime input.
        """
        if not isinstance(overrides, dict) or not overrides:
            return
        env_section = self.combined_config.setdefault('environment', {})
        if not isinstance(env_section, dict):
            env_section = {}
            self.combined_config['environment'] = env_section
        for key, value in overrides.items():
            if key == 'export' and isinstance(value, dict):
                export = env_section.setdefault('export', {})
                if isinstance(export, dict):
                    export.update(copy.deepcopy(value))
                else:
                    env_section['export'] = copy.deepcopy(value)
            else:
                env_section[key] = copy.deepcopy(value)


def _locate_package_config():
    """Find pmd_env.yaml within the installed pyrai3md package.

    Uses os.path relative to this file:
        kernel/environment.py -> pyrai3md/qualification/pmd_env.yaml

    Returns
    -------
    str
        Absolute path to the package configuration file.
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'qualification', 'pmd_env.yaml'
    )


def _locate_user_config():
    """Return path to ~/.pyrai3md/pmd_env.yaml.

    Returns
    -------
    str
        Absolute path to the user configuration file.
    """
    return os.path.join(
        os.path.expanduser('~'), _USER_DIR, _USER_CONFIG
    )


def _load_yaml(path):
    """Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content. Empty dict if file does not exist
        or contains no data.
    """
    if not os.path.isfile(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def _ensure_user_config(user_path):
    """Create ~/.pyrai3md/pmd_env.yaml with template if not exists.

    Creates the directory and a commented template config file on
    first run. Does nothing if the file already exists.

    Parameters
    ----------
    user_path : str
        Path to the user configuration file.
    """
    if os.path.isfile(user_path):
        return

    user_dir = os.path.dirname(user_path)
    os.makedirs(user_dir, exist_ok=True)

    with open(user_path, 'w', encoding='utf-8') as f:
        f.write(_USER_CONFIG_TEMPLATE)


def ensure_user_config():
    """Ensure ~/.pyrai3md/pmd_env.yaml exists, creating it if necessary.

    Public entry point for external callers (e.g. qualification cli_entry())
    that need the user config file to exist without running the full
    Environment.initialize() bootstrap.
    """
    user_path = _locate_user_config()
    _ensure_user_config(user_path)


def _combine_configs(package_config, user_config):
    """Combine package and user configs (user overrides package).

    Performs a recursive deep merge where user values take priority
    over package values. Dict values are merged recursively; non-dict
    values in user_config replace those in package_config. Empty
    override values (None, '', [], {}) count as unset and fall back
    to the package value; False and 0 are valid values and never
    fall back.

    Parameters
    ----------
    package_config : dict
        Package-level configuration (lowest priority).
    user_config : dict
        User-level configuration (overrides package).

    Returns
    -------
    dict
        Combined configuration dictionary.
    """
    return deep_merge(package_config, user_config)




