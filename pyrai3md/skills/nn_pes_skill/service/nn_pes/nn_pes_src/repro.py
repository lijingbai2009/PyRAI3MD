"""
======================================================================
This is a standalone NN-PES skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,

Oct 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Reproducibility for a fit that happens in a child process.

Every network in this skill is trained by a subprocess
(`python -m ...training_mlp_*`), which is what makes a run reproducible or not:
the parent's random state does not travel into a child. A seed has to be carried
in the command line - the same line a cluster arm writes into its job script -
and applied in the child before any weight is drawn.

`apply_seed` is that application point, and it is one call per child entry. It
seeds all three streams a Keras fit consumes (Python's `random`, NumPy, and
TensorFlow), because the fit shuffles batches through the framework's own state
and not through NumPy alone: seeding one of them leaves the others drawing fresh
numbers, which is how a "seeded" training can still differ run to run.

"""


def apply_seed(seed):
    """Seed this process before a model is built or a batch order is drawn.

    Parameters
    ----------
    seed : int or None
        The value to seed with - the framework's `control.gl_seed`, carried into
        the child as the `-s` argument. None leaves the process unseeded, which is
        what a run without a seed does today, and is reported as such by the
        return value so a caller can tell the two apart.

    Returns
    -------
    int or None
        The seed applied, or None when nothing was applied.
    """
    if seed is None or seed == '':
        return None
    import keras
    value = int(seed)
    # One call, three streams: keras.utils.set_random_seed seeds Python's random,
    # NumPy and TensorFlow, which is the whole surface a fit draws from.
    keras.utils.set_random_seed(value)
    return value
