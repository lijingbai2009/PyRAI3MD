"""
======================================================================
This module starts the training scripts of NNsMD in parallel.

Only local subprocesses are supported: each network instance is trained by
its own python process so that several GPUs and several fits can run at
the same time. Cluster submission would plug in here. The child is started
with the interpreter that runs this code and with the project root on
PYTHONPATH, because a training run launched through a bare "python3" from
PATH silently lands in whichever environment happens to come first.

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import subprocess
import sys



# The model families this skill can train, and the child module each one runs
# in. It is module data rather than a local dict because the registry test in
# `tests/test_seed_repro.py` walks these keys to prove that every training
# module - including one added later - takes the seed argument and seeds its
# own process; a family that cannot be enumerated cannot be checked.
FIT_MODULES = {"mlp_eg": "training_mlp_eg",
               "mlp_nac": "training_mlp_nac",
               "mlp_nac2": "training_mlp_nac2",
               "mlp_e": "training_mlp_e",
               "mlp_g2": "training_mlp_g2"}


def get_module_for_fit_script(model_type):
    """
    Interface to find the training module of a model type.

    The container starts training through "python -m <module>", which keeps
    the package importable for the child without knowing where it was
    installed. The prefix is read from this module's own import name, so
    the launcher works inside any enclosing package and no installation
    path is written into the source.

    Args:
        model_type (str): Name of the model.

    Returns:
        str: Importable module name of the training script.

    """
    here = __name__.rsplit('.', 1)[0]
    return "%s.training.%s" % (here, FIT_MODULES[model_type])


def child_environment(thread=None):
    """
    Return the environment of a training subprocess.

    The child is started with ``sys.executable -m <module>``, so it needs
    the same import paths the parent had. They are carried over through
    PYTHONPATH instead of a guessed project root, which keeps the launcher
    working whether this package sits inside a larger distribution or is
    used on its own.

    Args:
        thread (int, optional): Training thread budget for this member.
            When given, it overrides the parent's inherited TensorFlow
            settings so each member's training process is independently
            sized.

    Returns:
        dict: Environment mapping.

    """
    entries = [path for path in sys.path
               if path and os.path.isabs(path)]
    existing = os.environ.get("PYTHONPATH", "")
    merged = os.pathsep.join(entries + ([existing] if existing else []))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = merged
    if thread is not None:
        value = str(max(1, int(thread)))
        environment["OMP_NUM_THREADS"] = value
        environment["TF_NUM_INTRAOP_THREADS"] = value
        environment["TF_NUM_INTEROP_THREADS"] = value
    return environment


def train_command(model_type, i, filepath, g, m, seed=None):
    """
    Build the subprocess command line of one training instance.

    Both execution arms use this list unchanged: the local arm starts it
    with Popen, a cluster arm writes it into a job script. Keeping the
    construction here means the two arms cannot drift apart - and it is
    also why the seed travels as an argument and not as an environment
    variable: the command line is the one thing the cluster arm copies
    into the job script verbatim.

    Args:
        model_type (str): Name of the model.
        i (int): Index of model.
        filepath (str): Filepath to model.
        g (int): GPU index to use.
        m (str): Fitmode.
        seed (int, optional): RNG seed for this instance. The caller passes
            the framework's ``gl_seed`` advanced by the member index - the
            same rule an MD run applies to its trajectories
            (``workflow/md.py`` seeds with ``gl_seed + trajectory``), because
            every member of an ensemble that drew the same numbers would be
            the same network. None leaves the child unseeded, as before.

    Returns:
        list: Command line for subprocess.

    """
    command = [sys.executable, "-m", get_module_for_fit_script(model_type),
               "-i", str(i), "-f", filepath, "-g", str(g), "-m", str(m)]
    if seed is not None:
        command += ["-s", str(int(seed))]
    return command


def fit_model_by_modeltype(model_type, dist_method, i, filepath, g, m,
                           seed=None, thread=None):
    """
    Run a training module for one network instance.

    Args:
        model_type (str): Name of the model.
        dist_method (bool): Start asynchronously with Popen.
        i (int): Index of model.
        filepath (str): Filepath to model.
        g (int): GPU index to use.
        m (str): Fitmode.
        seed (int, optional): RNG seed carried to the child as `-s`.

    Returns:
        object: Popen instance for asynchronous runs, otherwise the
                CompletedProcess result.

    """
    print("Run:", filepath, "Instance:", i, "on GPU:", g, m)
    command = train_command(model_type, i, filepath, g, m, seed)
    environment = child_environment(thread)
    if dist_method:
        return subprocess.Popen(command, env=environment)
    return subprocess.run(command, capture_output=False, shell=False, env=environment)
