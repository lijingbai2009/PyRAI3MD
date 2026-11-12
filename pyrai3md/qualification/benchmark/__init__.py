"""
======================================================================
Benchmark package: application-level algorithm and performance
calibration of pyrai3md (the third pillar of the qualification
system, after verification/ and certification/).

Flat layout: one single-script module per benchmark task type,
named after the task (<task>.py, e.g. fssh.py), paired with its
calibration case YAMLs <task>_<case>.yaml (e.g. fssh_tully1.yaml).
The task module owns the complete calibration task (input
generation, independent reference handling and analysis); adding a
new case of an existing task only adds a YAML file.

The planner expands every case into a single compute task chaining
the reference and production runs in ONE generated script (input
files + local .sh, '&&' guarantees production
starts only after the reference). Right after the compute task
finished, the runner calls the task module's evaluate_run
in-process to analyze the run directory; evaluation is a
qualification tool, never a framework calculation task. The task
modules are programmatic only; the CLI entry point for manual use
lives in the public wrapper tools/benchmark.py (external tools
directory, next to deploy.py).

Aug 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .planner import plan_benchmark_tests

__all__ = ['plan_benchmark_tests']
