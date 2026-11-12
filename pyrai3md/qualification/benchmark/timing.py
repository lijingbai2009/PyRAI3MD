"""
======================================================================
This module is the wall-clock timing wrapper of the perf benchmark
task (the perf.py task module generates one leg command per
backend). It runs one pyrai3md calculation as a subprocess inside
the leg directory, captures the calculation output into a
persisted stdout file, measures the wall time of the calculation
and saves it next to the output, then exits with the subprocess
return code so the planner's '&&' chain keeps its fail-fast
semantics.

The wrapper is a tiny portable launcher (identical behavior in the
local .sh a job runs) and never touches the
calculation itself:

    python3 -m pyrai3md.qualification.benchmark.timing \
        <input.yaml> <leg_dir> <name>

    <leg_dir>/<name>.stdout   combined stdout/stderr of the run
    <leg_dir>/leg.timing      wall time of the run in seconds
                              (one float line)

The module is programmatic plus a single command-line entry used by
the generated leg commands; it is not a qualification task.

Sep 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import subprocess
import sys
import time


def run_timed(input_path, leg_dir, name):
    """Run one pyrai3md calculation with wall-time measurement.

    Parameters
    ----------
    input_path : str
        Path of the framework YAML input of the leg.
    leg_dir : str
        Leg working directory (the subprocess cwd; holds the
        trajectory outputs exactly like the plain chain command).
    name : str
        Leg name; the combined calculation output is saved as
        <leg_dir>/<name>.stdout.

    Returns
    -------
    int
        The pyrai3md subprocess return code (0 on success).
    """
    os.makedirs(leg_dir, exist_ok=True)
    stdout_path = os.path.join(leg_dir, '%s.stdout' % name)
    timing_path = os.path.join(leg_dir, 'leg.timing')

    started = time.time()
    with open(stdout_path, 'w', encoding='utf-8') as out:
        code = subprocess.call(
            ['pyrai3md', input_path], cwd=leg_dir,
            stdout=out, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    with open(timing_path, 'w', encoding='utf-8') as timing:
        timing.write('%.6f\n' % elapsed)
    return code


def main(argv=None):
    """Command-line entry of the timing wrapper."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print('usage: python3 -m pyrai3md.qualification.benchmark'
              '.timing <input.yaml> <leg_dir> <name>')
        return 2
    return run_timed(args[0], args[1], args[2])


if __name__ == '__main__':
    sys.exit(main())
