"""
======================================================================
This module provides the RuntimeDirectoryManager, which owns the
runtime directory layout used by remote dispatchers.

Layout (relative to the PyRAI3MD main directory, i.e. the current
working directory of the parent runtime):

    runtime/
      {skill_name}_runtime_{idx}/          child runtime directory
        {skill_name}_runtime_{idx}.rt      serialized PyRAIState
        result.json                        result of the child runtime
        slurm.sh                           submission script (slurm)

The runtime_idx uses plain %d formatting (runtime_3, not runtime_000003)
and is globally unique within the runtime root: the manager scans
existing directories before allocating a new index, so restarts never
collide with previous runtimes.

The directory name follows a configurable name_format template
(default '{skill_name}_runtime_{idx}'); the scan pattern of
existing_indices() is derived from the template so custom layouts
(e.g. '{title}_{idx}' trajectory directories) keep restart reuse.

The same runtime_idx string ('runtime_{idx}') is passed to the child
Calculator so backend directories append the identical suffix
(e.g. tmp_PYSCF_runtime_3).

Aug 05 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import re

from ...kernel.logger import get_logger

# Sentinel standing in for {idx} while the name template is turned
# into a scan pattern (must never appear in a real directory name).
_IDX_SENTINEL = '\x00IDX\x00'

DEFAULT_NAME_FORMAT = '{skill_name}_runtime_{idx}'

_LOGGER = get_logger(__name__)


def resolve_skill_name(state):
    """Derive the skill name used in runtime directory names.

    Reads the skill_name of the first task in the state queue;
    falls back to the first jobtype token, then 'pyrai3md'.

    Parameters
    ----------
    state : PyRAIState or None
        Parent state of the batch workflow.

    Returns
    -------
    str
        Skill name for directory naming.
    """
    if state is not None:
        try:
            for _, task in state.iter_tasks():
                skill_name = task.get('skill_name') or ''
                if skill_name:
                    return str(skill_name)
                break
        except (AttributeError, TypeError) as error:
            _LOGGER.warning(
                'Skill name read from the task queue failed (%s); the '
                'runtime directory name falls back to the jobtype token',
                error)
        jobtype = getattr(state, 'jobtype', '') or ''
        tokens = jobtype.split()
        if tokens:
            return tokens[0]
    return 'pyrai3md'


def package_source_root():
    """Return the directory that contains the pyrai3md package.

    Child runtimes must import the same pyrai3md as the parent
    process; dispatchers prepend this path to PYTHONPATH so an
    installed copy of the package cannot shadow the source tree.

    Returns
    -------
    str
        Absolute path of the package parent directory.
    """
    import pyrai3md
    return os.path.dirname(os.path.dirname(os.path.abspath(
        pyrai3md.__file__)))


def flatten_export_variables(state=None):
    """Flatten the environment export variables to plain strings.

    The sole source is the environment snapshot carried by the state
    (state.pmd.environment of the main PyRAIConfig), so dispatch
    never depends on the kernel Environment singleton or
    pmd_env.yaml. A state without the snapshot yields no exports.

    Parameters
    ----------
    state : PyRAIState or None
        State carrying the environment snapshot.

    Returns
    -------
    dict
        Export variable name -> string value; empty dict when the
        state carries no environment snapshot or no export section.
    """
    env_model = getattr(getattr(state, 'pmd', None),
                        'environment', None)
    if env_model is None:
        return {}
    return {str(key): str(value)
            for key, value in (env_model.environment or {}).items()}


class RuntimeDirectoryManager:
    """Manage runtime directories, .rt files and result files.

    Parameters
    ----------
    skill_name : str
        Skill name component of the directory/file names.
    root : str or None
        Runtime root directory. Defaults to '<cwd>/runtime'.
    name_format : str or None
        Directory name template with the placeholders
        '{skill_name}' and '{idx}'. Defaults to
        '{skill_name}_runtime_{idx}'.
    """

    def __init__(self, skill_name='pyrai3md', root=None, name_format=None):
        self.skill_name = str(skill_name or 'pyrai3md')
        self.root = root if root is not None else os.path.join(
            os.getcwd(), 'runtime')
        self.name_format = name_format or DEFAULT_NAME_FORMAT

    def runtime_name(self, idx):
        """Return the runtime name for one index.

        Parameters
        ----------
        idx : int
            Runtime index (plain %d, no zero padding).

        Returns
        -------
        str
            The name_format template filled with skill_name and idx.
        """
        return self.name_format.format(
            skill_name=self.skill_name, idx=int(idx))

    def runtime_idx(self, idx):
        """Return the workflow runtime suffix for one index.

        Parameters
        ----------
        idx : int
            Runtime index (plain %d, no zero padding).

        Returns
        -------
        str
            'runtime_{idx}', appended to backend execution ids.
        """
        return "runtime_%d" % int(idx)

    def runtime_dir(self, idx):
        """Return the child runtime directory path.

        Parameters
        ----------
        idx : int
            Runtime index.

        Returns
        -------
        str
            '<root>/{skill_name}_runtime_{idx}'.
        """
        return os.path.join(self.root, self.runtime_name(idx))

    def runtime_file(self, idx):
        """Return the .rt file path inside the child runtime directory.

        Parameters
        ----------
        idx : int
            Runtime index.

        Returns
        -------
        str
            '<runtime_dir>/{skill_name}_runtime_{idx}.rt'.
        """
        return os.path.join(self.runtime_dir(idx),
                            self.runtime_name(idx) + '.rt')

    def result_file(self, idx):
        """Return the result.json path inside the child runtime directory.

        Parameters
        ----------
        idx : int
            Runtime index.

        Returns
        -------
        str
            '<runtime_dir>/result.json'.
        """
        return os.path.join(self.runtime_dir(idx), 'result.json')

    def existing_indices(self):
        """Scan the runtime root for already allocated indices.

        Returns
        -------
        set of int
            Indices found in existing directories matching the
            name_format template.
        """
        indices = set()
        if not os.path.isdir(self.root):
            return indices
        pattern = self._name_pattern()
        for entry in os.listdir(self.root):
            match = pattern.match(entry)
            if match:
                indices.add(int(match.group(1)))
        return indices

    def _name_pattern(self):
        """Compile the scan pattern derived from the name template.

        The template is filled with the skill_name and a sentinel for
        '{idx}', escaped, then the escaped sentinel is replaced by a
        digit group, so every template yields its exact match rule.

        Returns
        -------
        re.Pattern
            Anchored pattern with one group capturing the index.
        """
        literal = self.name_format.format(
            skill_name=self.skill_name, idx=_IDX_SENTINEL)
        escaped = re.escape(literal)
        return re.compile('^%s$' % escaped.replace(
            re.escape(_IDX_SENTINEL), r'(\d+)'))

    def next_index(self):
        """Allocate the next globally unique runtime index.

        Returns
        -------
        int
            Smallest positive index not present in the runtime root.
        """
        existing = self.existing_indices()
        idx = 1
        while idx in existing:
            idx += 1
        return idx

    def allocate_indices(self, count):
        """Allocate indices for a whole batch with restart reuse.

        Existing runtime indices are reused first (ascending order,
        matching batch order), so a restarted parent runtime maps
        its batch tasks back onto the previous runtimes; remaining
        tasks get fresh indices beyond the largest existing one.

        Parameters
        ----------
        count : int
            Number of batch tasks.

        Returns
        -------
        list of int
            One index per task, in batch order.
        """
        count = max(int(count), 0)
        existing = sorted(self.existing_indices())
        indices = existing[:count]
        if len(indices) < count:
            used = set(existing)
            idx = (existing[-1] + 1) if existing else 1
            while len(indices) < count:
                while idx in used:
                    idx += 1
                indices.append(idx)
                used.add(idx)
                idx += 1
        return indices

    def create(self, idx):
        """Create the child runtime directory.

        Parameters
        ----------
        idx : int
            Runtime index.

        Returns
        -------
        str
            Absolute path of the created directory.
        """
        path = self.runtime_dir(idx)
        os.makedirs(path, exist_ok=True)
        return path

    def is_completed(self, idx):
        """Check whether a runtime finished (result.json present).

        Used by remote dispatchers to skip already completed
        runtimes on restart.

        Parameters
        ----------
        idx : int
            Runtime index.

        Returns
        -------
        bool
            True when result.json exists in the runtime directory.
        """
        return os.path.isfile(self.result_file(idx))

    def __repr__(self):
        return ("RuntimeDirectoryManager(skill_name='%s', root='%s', "
                "name_format='%s')") % (
            self.skill_name, self.root, self.name_format)
