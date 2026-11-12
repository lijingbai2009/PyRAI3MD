"""
======================================================================
This module provides the Launcher, which converts the remotetask
boolean into a single-task runner function.

    runner = Launcher()(remotetask)

- remotetask=False: returns the 'local' sentinel; the Scheduler
  resolves local execution through the executor_spec (direct skill
  backend API call in the current process or in a Pool worker).
- remotetask=True: returns slurm_run, which serializes the child
  state and submits it to a Slurm cluster.

The runner is consumed by the Scheduler which wraps it in either
a sequential or multiprocessing-pool execution strategy.

The launcher never imports the workflow package: the in-process
execution entry is injected (local_executor / executor_spec).

Aug 13 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""


class Launcher:
    """Convert the remotetask flag into a single-task runner."""

    def __init__(self):
        pass

    def __call__(self, remotetask):
        """Return the runner for the given remotetask flag.

        Parameters
        ----------
        remotetask : bool
            False: 'local' sentinel for direct skill API execution.
            True: slurm_run for Slurm cluster submission.

        Returns
        -------
        str or callable
            'local' for in-process execution, or slurm_run.
        """
        if not remotetask:
            return 'local'
        from .slurm_run import slurm_run
        return slurm_run

    def __repr__(self):
        return "Launcher()"
