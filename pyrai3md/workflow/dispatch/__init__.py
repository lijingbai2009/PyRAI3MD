"""
======================================================================
Dispatch subsystem for the PyRAI3MD workflow layer (Draft v3).

Architecture:
- Dispatcher: the unified execution base. Every workflow and the
  calculator inherit from it; execution mode is declared through two
  orthogonal booleans (multitask / remotetask) and resolved by the
  composed Launcher/Scheduler pipeline inside Dispatcher.dispatch().
- Launcher: converts remotetask into a single-task runner function
  ('local' sentinel for in-process, slurm_run for Slurm cluster
  submission).
- Scheduler: converts multitask into an execution wrapper
  (sequential or spawn multiprocessing pool); all execution
  (local and remote) goes through the same pool managed by the
  Scheduler, with dispatch_worker resolving the actual entry.
- ExecutionGraphBuilder: the single planning stage. The root runtime
  builds the execution graph once and freezes it into
  state.execution_graph; worker runtimes (resolve_execution_graph
  =False) only verify the parent-frozen graph.

Single-task runners (Launcher-selected, Scheduler-managed):
- slurm_run: child runtimes submitted to a Slurm cluster.

Supporting components:
- RuntimeDirectoryManager: runtime directory layout, .rt and
  result.json paths, unique runtime index allocation.
- terms: the term axis of execution, orthogonal to the two booleans
  above. A Dispatcher unit is a materialized child PyRAIState (it can
  be deep-copied, pickled into a pool worker and given its own runtime
  directory); a backend term carries a live backend instance, so it
  runs in-process (serial or thread, gated by the thread-safety audit
  registry) or as a dedicated remote term worker (term_worker.py). The
  work_item/outcome contracts are shared by both.

Universal child task rules:
1. Dispatched jobs are labeled as workers in the PyRAIState.
2. Worker execution graphs never carry remotetask=True.
3. Calculator nodes are always local (remotetask=False); remote
   single-point execution is controlled by the skill itself.

The dispatch package never imports the workflow package; the
in-process execution entry is injected by the workflow layer
(local_executor / executor_spec). MPI/SSH/PBS backends are reserved
for future implementations.

Aug 13 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .dispatcher import Dispatcher, worker_execution_graph
from .launcher import Launcher
from .scheduler import Scheduler, dispatch_worker
from .planner import (
    ExecutionGraphBuilder, aggregate_tristate, aggregate_multipool,
    read_section_flags, resolve_flag, resolve_pool,
    WORKFLOW_SECTIONS,
)
from .slurm_run import slurm_run
from .runtime import (
    RuntimeDirectoryManager, resolve_skill_name,
)
from .terms import (
    TaskExecutor, apply_failure_policy, audit_key, check_thread_safe,
    execute_with_failure_handling, make_outcome, make_work_item,
)
