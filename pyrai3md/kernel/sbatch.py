"""
======================================================================
This module owns the framework's single Slurm submission dialect: the
mapping from a `slurm:` section key to its sbatch directive, and the
rules that turn one section into `#SBATCH` lines.

Two callers ship different jobs - `qualification/manager.py` submits
one test-case group and `workflow/dispatch` submits one child runtime
or one backend term - but they read the same section and must ask the
cluster for the same allocation. The wrapper scripts stay separate; the
key names, the directive spellings and the completion rules do not.

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""


# Environment key -> sbatch directive. The keys are the snake_case names
# the `slurm:` section documents; the directives are the spellings sbatch
# itself accepts.
SBATCH_OPTIONS = {
    'partition': '--partition',
    'nodes': '--nodes',
    'ntasks': '--ntasks',
    'cpus_per_task': '--cpus-per-task',
    'mem': '--mem',
    'mem_per_cpu': '--mem-per-cpu',
    'time': '--time',
    'job_name': '--job-name',
    'output': '--output',
    'error': '--error',
    'account': '--account',
    'qos': '--qos',
    'constraint': '--constraint',
    'gres': '--gres',
    'mail_type': '--mail-type',
    'mail_user': '--mail-user',
    'array': '--array',
    # A boolean directive: `exclusive: true` asks for the whole allocation,
    # the shipped default `false` writes nothing.
    'exclusive': '--exclusive',
}

# Keys of the slurm section that are used by the script builders but are not
# '#SBATCH' directives.
SLURM_PASSTHROUGH_KEYS = ('extra',)


def unknown_slurm_keys(slurm_config):
    """Name the slurm keys this dialect would silently ignore.

    A directive spelled wrong (`memmory: 8G`) would land in the snapshot,
    reach no `#SBATCH` line, and let the job run on cluster defaults while
    the input reads as if it had said otherwise. Both script builders
    refuse it at script time, which is the earliest point that knows the
    job is really going out.
    """
    known = set(SBATCH_OPTIONS) | set(SLURM_PASSTHROUGH_KEYS)
    return sorted(str(key) for key in (slurm_config or {})
                  if str(key) not in known)


def render_sbatch_directives(slurm_config):
    """Render one `slurm:` section as `#SBATCH` lines.

    Parameters
    ----------
    slurm_config : dict
        Slurm parameters from the environment configuration or
        pmd_env.yaml.

    Returns
    -------
    list of str
        Directive lines in `SBATCH_OPTIONS` order, followed by the raw
        `extra` lines.

    Raises
    ------
    ValueError
        When the section carries a key that maps to no directive and to no
        known pass-through key.
    """
    config = dict(slurm_config or {})
    unknown = unknown_slurm_keys(config)
    if unknown:
        raise ValueError(
            'slurm section carries %s, which maps to no #SBATCH '
            'directive and to no known pass-through key; known: %s'
            % (', '.join(unknown),
               ', '.join(sorted(SBATCH_OPTIONS)
                         + list(SLURM_PASSTHROUGH_KEYS))))

    lines = []
    for key, directive in SBATCH_OPTIONS.items():
        value = config.get(key)
        if value in ('', None):
            continue
        if isinstance(value, bool):
            if value:
                lines.append('#SBATCH %s' % directive)
            continue
        lines.append('#SBATCH %s=%s' % (directive, value))

    extra = config.get('extra')
    if extra not in ('', None):
        lines.extend(str(extra).splitlines())
    return lines
