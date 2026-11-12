#!/usr/bin/env python3
"""
======================================================================
This module is the remote deployment tool for PyRAI3MD.

It is a development utility responsible for:
1. Reading qualification/pmd_env.yaml for deployment configuration
2. Interactive configuration if deploy section is incomplete
3. Resolving the SSH connection mode: direct connection first,
   falling back to a jump host (ssh -J) when configured
4. Synchronizing source code to the remote server via rsync
   (through the jump host via ssh -J when needed)
5. Installing PyRAI3MD on the remote server
6. Triggering pyrai3md test on the remote server and waiting
   for its work directory to appear before returning
7. Periodically polling test status via SSH and downloading
   deploy_test.log as test_<stamp>.log on completion

This tool is located in tools/ and is run directly:
    python tools/deploy.py

It does not participate in test planning or execution logic.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

# Prefer the local source tree over any installed pyrai3md package so
# estimates and imports always reflect the current checkout. Without
# this, running 'python tools/deploy.py' puts tools/ on sys.path but
# not the repo root, and an outdated site-packages copy would shadow
# the working tree.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml


# Path to pmd_env.yaml relative to this script
ENV_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'pyrai3md', 'qualification', 'pmd_env.yaml'
)

# Qualification test work directories are named tmp_<YYYYMMDD_HHMMSS>
# by create_work_dir; skill calculation directories (tmp_MOLCAS, ...)
# share the tmp_ prefix but must never be polled as test work dirs.
TMP_DIR_PATTERN = re.compile(r'^tmp_\d{8}_\d{6}$')

# What pkill matches to stop the remote test. The bracketed digit is load
# bearing: 'python.*pyrai3md test' also matches the command line of the
# shell that runs the pkill, which kills its own session and reports a
# failed stop for a test that did stop. Pinned by
# test_deploy_log.py::TestKillRemotePattern.
REMOTE_TEST_PATTERN = 'python.*pyrai[3]md test'

# Seconds the remote trigger waits for the test process to create its
# work directory before giving up and letting the poller discover it.
_TRIGGER_WAIT_SECS = 120

# deploy's own polling idle base (seconds). The effective idle timeout
# is this base plus the per-test _timeout from pmd_env.yaml, so a single
# remote task running its full budget is never misjudged as a hang.
_POLL_IDLE_BASE = 300


def main():
    """Entry point for the deployment tool.

    Checks location, loads configuration, and runs the full
    deployment workflow: sync, install, trigger test, collect results.
    """
    if '--help' in sys.argv or '-h' in sys.argv:
        _print_help()
        return

    if '--force' in sys.argv:
        _force_stop_remote_test()
        return

    _print_header()
    _check_location()

    config = _load_config()
    config = _ensure_deploy_config(config)

    deploy_config = config.get('deploy', {})
    env_config = config.get('environment', {})

    host = deploy_config.get('host', '')
    username = deploy_config.get('username', '')
    port = deploy_config.get('port', 22)
    install_path = deploy_config.get('install_path', '')
    python = env_config.get('python', 'python3')
    conda_env = env_config.get('conda', '')

    if not host or not install_path:
        print('Error: deploy.host and deploy.install_path are required.')
        sys.exit(1)

    # Step 0: Resolve SSH connection mode (direct first, then jump host)
    ssh_base, use_jump = _resolve_ssh_base(config)

    # The jump section may have been updated interactively during
    # connection resolution; re-read it for rsync/scp wiring.
    deploy_config = config.get('deploy', {})
    jump_target = _build_jump_target(deploy_config.get('jump_host', ''),
                                     deploy_config.get('jump_username', ''),
                                     username)
    rsync_base = _build_rsync_command(host, username, port,
                                      jump_target if use_jump else '')

    # Step 1: Print prominent SSH/conda configuration banner
    _print_config_banner(config, use_jump)

    # A nohup test outlives the local poller. Refuse a second round while
    # the first is still submitting work; --force is the explicit stop.
    if _remote_test_alive(ssh_base):
        print('\nError: a remote pyrai3md test is already running.')
        print('Stop it with `python tools/deploy.py --force` before starting '
              'a new round.')
        sys.exit(1)

    # Step 2: Resolve conda environment (with interactive prompt)
    conda_env = _resolve_conda_env(ssh_base, conda_env, config)

    # Step 3: Sync source code
    print('Syncing source code to remote...')
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not _sync_code(rsync_base, source_dir, install_path):
        print('\nError: failed to sync source code to remote.')
        print('Check rsync output above for details.')
        sys.exit(1)

    # Step 4: Remote install
    print('Installing PyRAI3MD on remote...')
    if not _remote_install(ssh_base, install_path, python, conda_env):
        print('\nError: remote installation failed.')
        print(f'Check {install_path} on the remote server.')
        sys.exit(1)

    # Step 5: Trigger remote test
    # The round's own log is the denominator: its header prints
    # "Total tests: N" after planning and every row prints "[i/N]". Running
    # the planner here first would plan the whole round a second time
    # (minutes on a full round), so no pre-trigger probe is sent; the poller
    # adopts the header total as soon as it appears.
    expected = 0
    print('Triggering remote test...')
    trigger_ok, work_dir = _remote_trigger(ssh_base, install_path, python,
                                           conda_env)
    if not trigger_ok:
        print('\nError: failed to trigger remote test.')
        print(f'Check {install_path}/deploy_test.log on the remote server.')
        sys.exit(1)

    # Step 6: Periodically collect results
    print('Waiting for test results... (Ctrl-C stops the remote test)')
    # deploy's polling idle timeout = its own base + the timeout every task in
    # this round is given, so a leg that runs its full allowance is not
    # misjudged as a hung remote. The poll's clock resets on NEW completions,
    # not on polls that merely answer, so the window has to clear one task:
    # `_timeout` is the only budget, per-example ones were removed on
    # 2026-10-20 after the most expensive leg measured inside it.
    test_timeout = int(config.get('_timeout', 300) or 300)
    poll_idle_timeout = _POLL_IDLE_BASE + test_timeout
    try:
        results = _collect_results(ssh_base, install_path, expected,
                                   idle_timeout=poll_idle_timeout,
                                   port=port,
                                   jump_host=jump_target if use_jump else '',
                                   resolved_dir=work_dir)
    except KeyboardInterrupt:
        # Ctrl-C during polling: stop the remote test as well so the
        # detached nohup process does not keep running on its own.
        print('\nInterrupted by user. Stopping remote test...')
        _kill_remote_test(ssh_base, install_path)
        sys.exit(130)

    if results:
        _print_summary(results)
    else:
        print('Deploy completed but no results collected.')


def _print_help():
    """Print help message showing configuration file location and options."""
    from pyrai3md.qualification.manager import EXAMPLE_FAMILIES
    print('PyRAI3MD Remote Deploy Tool')
    print()
    print('Usage:  python tools/deploy.py [--help] [--force]')
    print()
    print('Options:')
    print('  --help, -h   Show this help message')
    print('  --force      Force-stop any running remote test and exit')
    print('               (cancels the newest round\'s submitted jobs first)')
    print()
    print('  During result polling, Ctrl-C also terminates the remote')
    print('  test before exiting.')
    print()
    print(f'Configuration file:')
    print(f'  {ENV_YAML_PATH}')
    print()
    print('Configurable variables:')
    print()
    print('  deploy:')
    print('    host          Remote server hostname or IP')
    print('    username      SSH login username')
    print('    port          SSH port (default: 22)')
    print('    install_path  Remote installation directory')
    print('    jump_host     Jump/bastion host for ssh -J. Empty = direct')
    print('                  connection without a jump host.')
    print('    jump_username Login username on the jump host. Empty = use')
    print('                  the remote username (deploy.username).')
    print()
    print('Connection strategy:')
    print('  The tool first tries a direct SSH connection (5s timeout).')
    print('  If that fails and deploy.jump_host is configured, it retries')
    print('  through the jump host using ssh -J. If no jump host is')
    print('  configured yet, the tool asks for one interactively; an')
    print('  empty answer aborts with an error, a non-empty answer is')
    print('  saved to the config file and used for the retry. The chosen')
    print('  mode is applied to all ssh/scp/rsync operations of the run.')
    print()
    print('  environment:')
    print('    conda         Conda environment name (empty = use default)')
    print('    python        Python interpreter command (default: python3)')
    print('    scheduler     Round execution mode: \'\' (or local) runs the')
    print('                  round in one local process; \'slurm\' groups the')
    print('                  test cases into independent jobs and submits them')
    print('                  (jobs under tmp_*/jobs/; the orchestrator is still')
    print('                  the pyrai3md test process). Skill self-submission')
    print('                  stays skills.<name>.execution_mode, independent')
    print('                  of this key.')
    print('    dependencies  List of extra pip packages to install on remote')
    print('    export        Environment variables to set on remote')
    print('      MPI_NP            MPI process count')
    print('      OMP_NUM_THREADS   OpenMP thread count')
    print()
    print('  _timeout:')
    print('    Per-test execution timeout in seconds (default 300).')
    print('    The runner subprocess uses it as a backstop; each')
    print('    generated execution script embeds timeout (_timeout-5)s')
    print('    so a timed-out engine process group is killed with no')
    print('    residual processes. deploy polling idle timeout equals')
    print('    300 (its own base) + _timeout.')
    print()
    print('  _runall:')
    print('    Top-level master switch, three values (a missing key')
    print('    means condition). The same three values govern the')
    for family in EXAMPLE_FAMILIES:
        print('    family switch examples.%s' % family)
    print('    and skills.<name>.test.')
    print('    true       force everything below it on')
    print('    false    force everything below it off (the main')
    print('             framework tests carry no switch and run anyway).')
    print('             It is decided above the families, so a `false`')
    print('             master with `optimize_test: true` still runs no')
    print('             example - an optimize round is `condition` plus')
    print('             the family key.')
    print('    condition  let the family / per-test switches decide')
    print()
    print('  skills:')
    print('    Dict of skill names to their environment config')
    print()
    print('  benchmark:')
    print('    Dict of benchmark case switches (case key -> true/false,')
    print('    e.g. fssh_tully1; a missing key keeps the case enabled)')
    print()
    print('  slurm:')
    print('    Slurm scheduler options passed to sbatch:')
    print('      partition       Queue/partition name (e.g. normal, gpu)')
    print('      nodes           Number of compute nodes (default: 1)')
    print('      ntasks          Total MPI tasks (default: 1)')
    print('      cpus_per_task   CPU cores per task for OpenMP (default: 1)')
    print('      mem             Memory per node (e.g. 8G, 16G)')
    print('      mem_per_cpu     Memory per CPU (e.g. 2G); exclusive with mem')
    print('      time            Wall time limit (DD-HH:MM:SS or MM:SS)')
    print('      job_name        Job name shown in squeue (default: pyrai3md)')
    print('      output          Stdout file pattern (%j = job ID)')
    print('      error           Stderr file pattern (%j = job ID)')
    print('      account         Billing account/project code')
    print('      qos             Quality of service tier')
    print('      exclusive       Request exclusive node access (true/false)')
    print('      constraint      Node feature constraint (e.g. intel)')
    print('      gres            Generic resources (e.g. gpu:2)')
    print('      mail_type       Email notify: NONE/BEGIN/END/FAIL/ALL')
    print('      mail_user       Email address for notifications')
    print('      array           Job array spec (e.g. 0-9, 1-100%10)')
    print('      extra           Additional raw sbatch options (verbatim)')
    print('    Empty string values are omitted from the generated script.')
    print()
    print('Interactive prompts:')
    print('  If deploy.host or deploy.username is missing, the tool will')
    print('  prompt for these values and save them to the config file.')
    print('  When deploy.host is missing (first use), the tool also asks')
    print('  for a jump host after the direct connection information;')
    print('  press Enter to keep a direct connection. An existing host is')
    print('  assumed to work directly - the jump host is only requested')
    print('  at runtime when the direct connection fails.')
    print('  If environment.conda is empty, the tool will ask for a conda')
    print('  environment name. Press Enter to skip; the prompt will not')
    print('  appear again on subsequent runs.')
    print()


def _remote_test_alive(ssh_base):
    """Whether a remote pyrai3md test process is already running.

    The test is triggered with nohup, so stopping the local deploy tool
    does not stop it. Starting a second round then leaves two test
    processes submitting reference work into one machine, which is how a
    previous round accumulated hundreds of slurm jobs. Refuse to start
    while one is already alive; --force is the explicit stop.
    """
    probe = ("pgrep -f '%s' >/dev/null 2>&1 && echo ALIVE || echo DEAD"
             % REMOTE_TEST_PATTERN)
    success, output = _run_remote(ssh_base, probe, timeout=20)
    return bool(success and 'ALIVE' in output)


def _remote_scancel_round(ssh_base, remote_dir):
    """Cancel the jobs recorded by the newest round's ledger, if any.

    A slurm round's jobs outlive the orchestrator, so stopping the
    orchestrator alone would leave queued work behind. The ledger
    `tmp_*/jobs/jobs.json` names exactly the jobs this round submitted;
    this reads the newest one and cancels those ids. Best effort: an
    absent ledger (a local round) or an unreachable host is not an
    error.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    remote_dir : str
        Remote project directory (empty = nothing to do).

    Returns
    -------
    int
        Number of job ids cancelled.
    """
    if not remote_dir:
        return 0
    success, output = _run_remote(
        ssh_base,
        f'ls -dt {shlex.quote(remote_dir)}/tmp_*/ 2>/dev/null | head -1',
        timeout=20)
    work_dir = output.strip().splitlines()[-1].strip() \
        if success and output.strip() else ''
    if not work_dir:
        return 0
    ledger = os.path.join(work_dir, 'jobs', 'jobs.json')
    success, text = _run_remote(
        ssh_base, 'cat %s 2>/dev/null' % shlex.quote(ledger), timeout=20)
    if not success or not text.strip():
        return 0
    try:
        data = json.loads(text)
    except ValueError:
        return 0
    job_ids = []
    for group in data.get('groups', []) or []:
        job_id = str(group.get('job_id') or '').strip()
        if job_id.isdigit():
            job_ids.append(job_id)
    job_ids = list(dict.fromkeys(job_ids))
    if not job_ids:
        return 0
    _run_remote(ssh_base, 'scancel %s' % ' '.join(job_ids), timeout=60)
    return len(job_ids)


def _kill_remote_test(ssh_base, remote_dir=''):
    """Kill the remote pyrai3md test process over a resolved SSH channel.

    Shared by the --force entry point and the Ctrl-C cleanup path so
    both terminate the remote test with the same command.

    The pattern carries a bracket on purpose: a plain
    "pkill -f 'python.*pyrai3md test'" matches the very shell that runs it,
    because that shell's own command line contains "python" and
    "pyrai3md test". It then killed its own session, ssh exited non-zero, and
    --force printed "Could not terminate:" followed by OpenSSH's
    post-quantum notice -- for a test that had in fact been stopped. The
    bracketed form does not match itself.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    remote_dir : str
        Remote project directory; the newest round's job ledger is
        read from it and its jobs cancelled before the orchestrator is
        stopped.

    Returns
    -------
    bool
        True when a running test was stopped, False when none was running
        or the host could not be reached.
    """
    cancelled = _remote_scancel_round(ssh_base, remote_dir)
    if cancelled:
        print(f'  Cancelled {cancelled} submitted job(s) from the latest '
              f'round.')
    kill_cmd = ("pkill -f '%s' 2>/dev/null; "
                'echo "PKILL_RC=$?"' % REMOTE_TEST_PATTERN)
    success, output = _run_remote(ssh_base, kill_cmd, timeout=30)
    if not success:
        print('  Could not reach the remote host to stop the test: %s'
              % output.strip()[:200])
        return False

    # pkill answers 0 when it matched a process, 1 when nothing matched, and
    # above that only for a bad pattern or a permission it does not have.
    matched = re.search(r'PKILL_RC=(\d+)', output)
    code = int(matched.group(1)) if matched else None
    if code == 0:
        print('  Remote test process terminated.')
    elif code == 1:
        print('  No remote test was running.')
    else:
        print('  Could not terminate the remote test (pkill rc=%s).' % code)
    return code == 0


def _force_stop_remote_test():
    """Force-stop any running remote pyrai3md test process."""
    config = _load_config()
    deploy_config = config.get('deploy', {})

    host = deploy_config.get('host', '')
    install_path = deploy_config.get('install_path', '')

    if not host or not install_path:
        print('Error: deploy.host and deploy.install_path are required.')
        sys.exit(1)

    ssh_base, _ = _resolve_ssh_base(config)

    print(f'Stopping remote test on {host}...')
    _kill_remote_test(ssh_base, install_path)


def _print_header():
    """Print the PyRAI3MD logo and remote deploy testing header."""
    from pyrai3md.utilities.logo import print_logo
    print(print_logo())
    print('  Remote Deploy Testing')
    print('  ' + '-' * 50)
    print()


def _check_location():
    """Verify this script is located in the tools/ directory.

    Prints a warning if not in the expected location.
    """
    script_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if script_dir != 'tools':
        print(f'Warning: deploy.py should be located in the tools directory '
              f'(currently in {script_dir}/).')


def _load_config():
    """Load the pmd_env.yaml configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.exists(ENV_YAML_PATH):
        print(f'Error: Configuration file not found: {ENV_YAML_PATH}')
        sys.exit(1)

    with open(ENV_YAML_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _ensure_deploy_config(config):
    """Check deploy section and prompt for missing values.

    If deploy.host or deploy.username is missing, enters interactive
    mode to collect the information and updates pmd_env.yaml.

    Parameters
    ----------
    config : dict
        Current configuration.

    Returns
    -------
    dict
        Updated configuration with complete deploy section.
    """
    deploy = config.get('deploy', {})
    missing = []

    if not deploy.get('host'):
        missing.append('host')
    if not deploy.get('username'):
        missing.append('username')

    if not missing:
        return config

    # Ask for the direct connection information first.
    print('\nDeployment configuration incomplete.')

    if 'host' in missing:
        deploy['host'] = input('Host: ').strip()
    if 'username' in missing:
        deploy['username'] = input('Username: ').strip()
    if not deploy.get('install_path'):
        deploy['install_path'] = input('Install path: ').strip()

    port_input = input('Port [22]: ').strip()
    if port_input:
        deploy['port'] = int(port_input)

    # First-use only (host was missing): the jump host is auxiliary to
    # the direct connection and not required every run, so it is asked
    # once together with the initial direct connection setup. An
    # existing host is assumed to work directly; the jump host is only
    # revisited at runtime when the direct connection fails.
    if 'host' in missing and 'jump_host' not in deploy:
        print('\nDeployment configuration: jump host')
        answer = input(
            'Jump host / bastion (press Enter for direct connection): '
        ).strip()
        deploy['jump_host'] = answer
        if answer:
            deploy['jump_username'] = input(
                f'Jump host username [default: {deploy["username"]}]: '
            ).strip()
        else:
            deploy['jump_username'] = ''

    config['deploy'] = deploy
    _save_config(config)
    print(f'Configuration updated: {ENV_YAML_PATH}\n')

    return config


def _save_config(config):
    """Write configuration back to pmd_env.yaml.

    Parameters
    ----------
    config : dict
        Configuration to save.
    """
    with open(ENV_YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _print_config_banner(config, use_jump):
    """Print a prominent banner with SSH and conda configuration.

    Shows the active SSH connection parameters (including whether a
    jump host is in use) and the conda environment, together with
    instructions on where to modify them.

    Parameters
    ----------
    config : dict
        Full pmd_env.yaml configuration.
    use_jump : bool
        Whether the jump host connection mode was selected.
    """
    deploy_config = config.get('deploy', {})
    env_config = config.get('environment', {})

    host = deploy_config.get('host', '')
    port = deploy_config.get('port', 22)
    username = deploy_config.get('username', '')
    jump_host = deploy_config.get('jump_host', '')
    jump_username = deploy_config.get('jump_username', '')
    conda_env = env_config.get('conda', '')

    jump_target = _build_jump_target(jump_host, jump_username, username)
    jump_info = f'{jump_target} (ssh -J)' if use_jump else 'none (direct)'
    conda_info = conda_env if conda_env else '(default environment)'
    round_mode = str(env_config.get('scheduler', '') or '').strip().lower()
    round_info = ('slurm grouped jobs' if round_mode == 'slurm'
                  else 'one local process')

    print()
    print('=' * 62)
    print('  SSH CONNECTION  (to modify, edit the deploy section in:')
    print(f'    {ENV_YAML_PATH})')
    print(f'    host      : {host}:{port}')
    print(f'    username  : {username}')
    print(f'    jump host : {jump_info}')
    print('  CONDA ENVIRONMENT  (to modify, edit environment.conda in the')
    print('    same file)')
    print(f'    conda     : {conda_info}')
    print('  TEST ROUND  (environment.scheduler)')
    print(f'    mode      : {round_info}')
    print('=' * 62)
    print()


def _build_jump_target(jump_host, jump_username='', fallback_username=''):
    """Combine jump host and jump username into an ssh -J target.

    Parameters
    ----------
    jump_host : str
        Jump/bastion host (may already carry user@ or :port parts).
    jump_username : str
        Login username on the jump host.
    fallback_username : str
        Username used when jump_username is empty (the target host
        username); the local username is used only when both are
        empty.

    Returns
    -------
    str
        ssh -J target string ('' when no jump host is configured).
    """
    if not jump_host:
        return ''
    if '@' in jump_host:
        return jump_host
    user = jump_username or fallback_username
    return f'{user}@{jump_host}' if user else jump_host


def _test_ssh(ssh_cmd, timeout=10):
    """Test an SSH connection by running 'echo ok' on the remote.

    Uses BatchMode so a missing key or a password prompt fails fast
    instead of hanging (all deploy operations are non-interactive).
    This is the only place where connection timeouts apply; interactive
    questions are handled separately and never count against them.

    Parameters
    ----------
    ssh_cmd : list of str
        Full SSH command prefix (options + target).
    timeout : int
        Local subprocess timeout in seconds.

    Returns
    -------
    bool
        True if the remote echoed 'ok', False otherwise.
    """
    cmd = list(ssh_cmd) + ['-o', 'BatchMode=yes', '-o',
                           'ConnectTimeout=5', '-o', 'LogLevel=ERROR',
                           'echo ok']
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode == 0 and 'ok' in proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def _connection_error(host, port, jump_target):
    """Print connection failure diagnostics and exit.

    Parameters
    ----------
    host : str
        Remote hostname.
    port : int
        SSH port.
    jump_target : str
        Jump host target string (empty = no jump host was involved).
    """
    print(f'\nError: cannot reach {host}:{port}'
          + (f' (direct or via {jump_target}).' if jump_target else '.'))
    print('Possible causes:')
    print('  - Server is offline or unreachable')
    print('  - VPN or campus network is not connected')
    print('  - Firewall is blocking the connection')
    print('  - Host, port, jump_host, or jump_username is misconfigured')
    print('    in pmd_env.yaml')
    print('  - SSH key authentication is not set up (BatchMode is')
    print('    required; password login is not supported)')
    print(f'\nConfiguration: {ENV_YAML_PATH}')
    sys.exit(1)


def _ask_jump_config(config):
    """Ask interactively for jump host information.

    Called only after the direct connection has failed and no jump
    host is configured. This function performs no SSH operations:
    it is plain input() without any timeout, and the SSH connection
    timeouts only start when the probe runs after the answers are
    received. A non-empty answer is saved to the config file.

    Parameters
    ----------
    config : dict
        Full pmd_env.yaml configuration (the deploy section is
        updated when the user supplies a jump host).

    Returns
    -------
    str
        The ssh -J jump target, or '' when the user chose to abort.
    """
    deploy_config = config.setdefault('deploy', {})
    username = deploy_config.get('username', '')

    print('\n  Direct connection failed. A jump host (bastion) may '
          'be required.')
    answer = input(
        '  Jump host / bastion (press Enter to abort): '
    ).strip()
    if not answer:
        return ''

    deploy_config['jump_host'] = answer
    deploy_config['jump_username'] = input(
        f'  Jump host username [default: {username}]: '
    ).strip()
    _save_config(config)

    jump_target = _build_jump_target(deploy_config['jump_host'],
                                     deploy_config['jump_username'],
                                     username)
    print(f'  Saved jump host configuration: {jump_target}')
    print(f'  Configuration updated: {ENV_YAML_PATH}')
    return jump_target


def _resolve_ssh_base(config):
    """Resolve the SSH connection mode: direct first, then jump host.

    Orchestration only: every SSH probe goes through _test_ssh (which
    alone carries the connection timeout), while the interactive jump
    host question lives in _ask_jump_config and is never subject to a
    timeout. First probes a direct SSH connection. If it fails and a
    jump host is configured, probes again through the jump host
    (ssh -J). If no jump host is configured, asks for one via
    _ask_jump_config and probes again with the answer. On total
    failure prints diagnostics and exits.

    Parameters
    ----------
    config : dict
        Full pmd_env.yaml configuration (the deploy section may be
        updated with interactively supplied jump host information).

    Returns
    -------
    tuple of (list of str, bool)
        (ssh_base, use_jump) — SSH command prefix for the chosen mode
        and whether the jump host is used.
    """
    deploy_config = config.setdefault('deploy', {})
    host = deploy_config.get('host', '')
    username = deploy_config.get('username', '')
    port = deploy_config.get('port', 22)
    jump_target = _build_jump_target(deploy_config.get('jump_host', ''),
                                     deploy_config.get('jump_username', ''),
                                     username)

    print(f'Checking direct SSH connection to {host}:{port} '
          f'(timeout 5s)...')
    direct_cmd = _build_ssh_command(host, username, port)
    if _test_ssh(direct_cmd):
        print(f'  Direct connection to {host}:{port} OK')
        return direct_cmd, False

    print(f'  Direct connection to {host}:{port} failed.')

    # Jump host already configured: probe through it directly.
    if jump_target:
        print(f'  Retrying through jump host {jump_target} (ssh -J)...')
        jump_cmd = _build_ssh_command(host, username, port, jump_target)
        # Two-hop connections need two TCP connect phases
        # (ConnectTimeout=5 per hop) plus the handshake, so give the
        # jump probe a longer local timeout than the direct probe.
        if _test_ssh(jump_cmd, timeout=20):
            print(f'  Connection to {host}:{port} via {jump_target} OK')
            return jump_cmd, True
        print(f'  Connection through jump host {jump_target} failed.')
        _connection_error(host, port, jump_target)

    # No jump host configured: ask interactively (timeout-free), then
    # start a fresh probe with the answer.
    jump_target = _ask_jump_config(config)
    if not jump_target:
        _connection_error(host, port, '')

    print(f'  Retrying through jump host {jump_target} (ssh -J)...')
    jump_cmd = _build_ssh_command(host, username, port, jump_target)
    if _test_ssh(jump_cmd, timeout=20):
        print(f'  Connection to {host}:{port} via {jump_target} OK')
        return jump_cmd, True
    print(f'  Connection through jump host {jump_target} failed.')

    _connection_error(host, port, jump_target)


def _build_ssh_command(host, username, port, jump_host=''):
    """Build SSH command prefix from configuration.

    Parameters
    ----------
    host : str
        Remote hostname.
    username : str
        Remote username.
    port : int
        SSH port.
    jump_host : str
        Jump/bastion host for ssh -J (empty = direct connection).

    Returns
    -------
    list of str
        SSH command components.
    """
    # LogLevel=ERROR keeps OpenSSH's own notices out of the captured
    # stderr: a two-hop connection prints a post-quantum key-exchange
    # warning on every call, and those lines were being reported back as
    # tool failures by whoever inspected the stderr half.
    cmd = ['ssh', '-o', 'ConnectTimeout=5', '-o', 'LogLevel=ERROR']
    if jump_host:
        cmd.extend(['-J', jump_host])
    if port and port != 22:
        cmd.extend(['-p', str(port)])
    target = f'{username}@{host}' if username else host
    cmd.append(target)
    return cmd


def _build_rsync_command(host, username, port, jump_host=''):
    """Build rsync command prefix from configuration.

    Parameters
    ----------
    host : str
        Remote hostname.
    username : str
        Remote username.
    port : int
        SSH port.
    jump_host : str
        Jump/bastion host tunneled via the rsync ssh transport
        (empty = direct connection).

    Returns
    -------
    list of str
        rsync command components.
    """
    ssh_opts = ['-o', 'ConnectTimeout=5', '-o', 'LogLevel=ERROR']
    if jump_host:
        ssh_opts.extend(['-J', jump_host])
    if port and port != 22:
        ssh_opts.extend(['-p', str(port)])

    cmd = ['rsync', '-avz', '--delete', '-e', ' '.join(['ssh'] + ssh_opts)]

    target = f'{username}@{host}' if username else host
    cmd.append(target)
    return cmd


def _build_scp_command(target, port, jump_host=''):
    """Build scp command prefix from configuration.

    Parameters
    ----------
    target : str
        user@host string of the remote server.
    port : int
        SSH port (scp uses the uppercase -P flag).
    jump_host : str
        Jump/bastion host for scp -J (empty = direct connection).

    Returns
    -------
    list of str
        scp command components (without source/destination).
    """
    cmd = ['scp', '-o', 'ConnectTimeout=5', '-o', 'LogLevel=ERROR']
    if jump_host:
        cmd.extend(['-J', jump_host])
    if port and port != 22:
        cmd.extend(['-P', str(port)])
    return cmd


def _run_remote(ssh_base, command, timeout=120, check=False):
    """Execute a command on the remote server via SSH.

    Provides consistent error handling for all remote operations.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    command : str
        Command string to execute on remote.
    timeout : int
        Command timeout in seconds.
    check : bool
        If True, return False on failure instead of printing error.

    Returns
    -------
    tuple of (bool, str)
        (success, output) — success flag and combined stdout/stderr.
    """
    cmd = list(ssh_base) + [command]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            return False, output
        return True, proc.stdout
    except subprocess.TimeoutExpired:
        return False, f'Command timed out after {timeout}s'
    except FileNotFoundError:
        return False, 'ssh command not found. Is openssh installed?'
    except Exception as e:
        return False, str(e)


def _resolve_conda_env(ssh_base, conda_env, config):
    """Resolve conda environment on the remote server.

    If conda_env is empty and the user has not previously dismissed
    the prompt, ask interactively. If the user presses Enter (empty),
    record the dismissal so the question is not asked again.

    If conda_env is not empty, try to activate it on the remote.
    If activation fails, fall back to the default environment.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    conda_env : str
        Configured conda environment name (may be empty).
    config : dict
        Full configuration (used to track prompt dismissal).

    Returns
    -------
    str
        Resolved conda environment name (empty string means default).
    """
    if not conda_env:
        prompts = config.setdefault('_prompts', {})
        if not prompts.get('conda_asked', False):
            answer = input(
                'Conda environment name (press Enter to use default): '
            ).strip()
            if answer:
                conda_env = answer
                config['environment']['conda'] = conda_env
                _save_config(config)
                print(f'  Saved conda environment: {conda_env}')
            else:
                prompts['conda_asked'] = True
                _save_config(config)
                print('  Using default environment. (Prompt dismissed)')
        else:
            print('  Conda environment: not specified, using default.')
        return conda_env if conda_env else ''

    print(f'Conda environment: {conda_env}, verifying on remote...')
    check_cmd = (
        f'source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null '
        f'&& conda activate {conda_env} 2>/dev/null && echo OK'
    )
    success, output = _run_remote(ssh_base, check_cmd, timeout=30)
    if success and 'OK' in output:
        print(f'Conda environment {conda_env} activated successfully.')
        return conda_env

    print(f'Warning: failed to activate {conda_env} on remote, using default environment.')
    return ''


def _sync_code(rsync_base, source_dir, remote_dir):
    """Synchronize source code to remote server.

    Parameters
    ----------
    rsync_base : list of str
        rsync command prefix (includes target).
    source_dir : str
        Local source directory.
    remote_dir : str
        Remote target directory.

    Returns
    -------
    bool
        True if sync succeeded, False otherwise.
    """
    dest = f'{rsync_base[-1]}:{remote_dir}/'

    cmd = list(rsync_base[:-1]) + [
        source_dir + '/',
        dest,
        '--exclude', '.git/',
        '--exclude', '__pycache__/',
        '--exclude', '*.egg-info/',
        '--exclude', '.idea/',
        '--exclude', '.vscode/',
        '--exclude', 'legacy/',
        '--exclude', 'skill_demo/',
        # Local run artifacts: never upload, but keep remote copies
        # (--delete does not remove excluded paths).
        '--exclude', 'tmp_*/',
        '--exclude', 'test_*.log',
        '--exclude', 'build/',
        '--exclude', '*.egg',
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f'rsync error (code {proc.returncode}):')
            stderr = proc.stderr.strip()
            if stderr:
                print(f'  {stderr}')
            return False
        return True
    except subprocess.TimeoutExpired:
        print('rsync timed out after 300s.')
        return False
    except FileNotFoundError:
        print('rsync command not found. Is rsync installed?')
        return False
    except Exception as e:
        print(f'rsync failed: {e}')
        return False


def _remote_install(ssh_base, remote_dir, python, conda_env):
    """Install PyRAI3MD on the remote server.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    remote_dir : str
        Remote project directory.
    python : str
        Python interpreter command.
    conda_env : str
        Conda environment name (empty for current env).

    Returns
    -------
    bool
        True if installation succeeded, False otherwise.
    """
    prefix = f'conda activate {conda_env} && ' if conda_env else ''
    # The remote's own build directory goes first. build/ is excluded from
    # the sync and --delete never removes an excluded path, so the server
    # keeps whatever an old build left under build/lib, and setuptools'
    # build_py copies changed files without pruning the ones that no longer
    # exist in the source. Stale example and skill-test YAMLs therefore get
    # installed and planned as live cases: the remote round runs a different
    # number of tasks than the tree it was given, and a removed input keeps
    # passing under a name nothing ships any more.
    install_cmd = (f'{prefix}cd {remote_dir} && rm -rf build *.egg-info && '
                   f'{python} -m pip install . 2>&1')
    success, output = _run_remote(ssh_base, install_cmd, timeout=300)
    if output.strip():
        for line in output.strip().splitlines()[-10:]:
            print(f'  {line}')
    if not success:
        print('  Installation failed.')
    return success


def _remote_trigger(ssh_base, remote_dir, python, conda_env):
    """Trigger pyrai3md test on the remote server.

    The test is backgrounded inside an explicit subshell with all
    stdio redirected. A bare 'a && b && nohup cmd > log 2>&1 &'
    backgrounds the whole AND-list: the implicit subshell inherits
    the SSH channel stdout/stderr and waits for the test process,
    keeping the channel open until the trigger timeout expires.

    The foreground shell then waits (up to _TRIGGER_WAIT_SECS) until
    the test process creates its timestamped work directory and
    echoes the directory path. The work directory is created only
    after the python interpreter finishes importing, so returning
    immediately after the nohup would race with the poller and let
    it lock onto a leftover directory from a previous run. The
    marker file makes 'newer' refer to the trigger instant, so
    leftover timestamped directories can never be reported.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    remote_dir : str
        Remote project directory.
    python : str
        Python interpreter command.
    conda_env : str
        Conda environment name.

    Returns
    -------
    tuple of (bool, str)
        (success, work_dir) — trigger success flag and the echoed
        work directory path ('' when the directory did not appear
        within the wait window; polling falls back to discovery).
    """
    prefix = f'conda activate {conda_env} && ' if conda_env else ''
    marker = '.deploy_trigger_marker'
    wait_loop = (
        f'found=""; i=0; '
        f'while [ $i -lt {_TRIGGER_WAIT_SECS} ]; do '
        f'found=$(find . -maxdepth 1 -type d -newer {marker} '
        f'-name \'tmp_*\' 2>/dev/null | sed \'s|^\\./||\' | '
        f'grep -E \'^tmp_[0-9]{{8}}_[0-9]{{6}}$\' | head -1); '
        f'[ -n "$found" ] && break; i=$((i+1)); sleep 1; done; '
        f'rm -f {marker}; echo "$found"'
    )
    trigger_cmd = (
        f'{prefix}cd {remote_dir} && '
        f'touch {marker} && '
        # -u keeps the remote log in the order things happened: without it
        # the banner and header sit in a block-buffered stdout and reach the
        # file after the first stderr-written test lines.
        f'(nohup {python} -u -m pyrai3md test '
        f'> {remote_dir}/deploy_test.log 2>&1 < /dev/null &) && '
        f'{wait_loop}'
    )
    # The wait loop may hold the channel for the whole window.
    success, output = _run_remote(
        ssh_base, trigger_cmd, timeout=_TRIGGER_WAIT_SECS + 30)
    if not success:
        print(f'  Trigger failed: {output.strip()}')
        return False, ''

    work_dir = ''
    for line in output.splitlines():
        line = line.strip()
        if TMP_DIR_PATTERN.match(line):
            work_dir = f'{remote_dir}/{line}'
            break
    if not work_dir:
        print(f'  Note: work directory not reported within '
              f'{_TRIGGER_WAIT_SECS}s; falling back to discovery.')
    return True, work_dir


def _progress_line(completed, total, elapsed, detail=''):
    """Render one remote polling progress line.

    Parameters
    ----------
    completed : int
        Number of finished tasks.
    total : int
        Total number of tasks; 0 while still unknown, which drops the
        percentage and the counter.
    elapsed : int
        Seconds since the run was triggered.
    detail : str
        Trailing diagnostic text (liveness, warnings), may be empty.

    Returns
    -------
    str
        Line shaped '[YYYY-MM-DD HH:MM:SS] RUNNING TESTS  34%
        (043 of 125) ELAPSED: 34 seconds <detail>'; the completed
        count is zero padded to the width of the total.
    """
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if total > 0:
        counter = (f'RUNNING TESTS {100 * completed // total:3d}% '
                   f'({completed:0{len(str(total))}d} of {total})')
    else:
        counter = 'RUNNING TESTS'
    line = f'[{stamp}] {counter} ELAPSED: {elapsed} seconds'
    return f'{line} {detail}' if detail else line


def _next_denominator(current, log_total):
    """Adopt the runner's own task count as soon as it is visible.

    The runner writes `[index/total]` per task, and that `total` is the plan
    the round is actually executing. Once it can be read it outranks any
    estimate, so every later progress line divides by the same number the
    final report will.

    Parameters
    ----------
    current : int
        Denominator in use (0 until the round's own header is read).
    log_total : int
        Total parsed from the streamed progress lines; 0 when unknown.

    Returns
    -------
    int
        The denominator to keep using.
    """
    return log_total if log_total else current


def _collect_results(ssh_base, remote_dir, expected_count, interval=30, idle_timeout=300, tmp_wait_timeout=120, port=22, jump_host='', resolved_dir=''):
    """Periodically poll the remote test status via SSH.

    The work directory is normally resolved synchronously by the
    trigger (which waits until the test process creates it) and
    passed in via resolved_dir. When it is missing, the directory is
    discovered on the remote via SSH and the first timestamped match
    is locked. On completion
    only deploy_test.log is downloaded into the current directory as
    test_<stamp>.log; the remote tmp_* work directory itself is not
    copied.

    Uses idle timeout (no-response timeout) instead of fixed total timeout.
    The function will keep waiting as long as the remote test is making
    progress. Only times out if there is no response for idle_timeout seconds.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    remote_dir : str
        Remote project directory.
    expected_count : int
        Expected total number of test tasks.
    interval : int
        Polling interval in seconds (default 30).
    idle_timeout : int
        Maximum wait time without any response in seconds (default 300).
        Resets on each successful response from remote.
    tmp_wait_timeout : int
        Maximum wait time for tmp directory to appear in seconds (default 120).
    port : int
        SSH port forwarded to the scp log download.
    jump_host : str
        Jump host forwarded to the scp log download (empty = direct).
    resolved_dir : str
        Work directory path already resolved by the trigger; empty
        falls back to remote discovery.

    Returns
    -------
    dict or None
        Final test results, or None if timeout or error.
    """
    target = ssh_base[-1]
    # One denominator for every progress line. The remote runner plans its
    # own task count, and that is the only number the round's own report
    # will use, so it wins as soon as it is known; the caller's value is a
    # floor, never a second competing total.
    denominator = expected_count
    if resolved_dir:
        print(f'  Remote work dir: {resolved_dir}')

    start_time = time.time()
    # Two clocks, because one clock conflated two different facts. A round is
    # QUIET when the remote answers and says nothing new, and unreachable when
    # the probe itself fails; only the first ages the idle window. Reading the
    # second as the first is what stopped round `tmp_20261018_173454`'s
    # collection ~2 h early while the remote sat at 105 of 243 rows with zero
    # failures - the link had gone, not the run.
    last_completed = 0  # Track progress for idle detection
    quiet_since = time.time()
    unreachable_since = None
    unreachable_attempts = 0
    # An outage this long is not a network blip; stop waiting and hand the
    # collection over to the route a person can run by hand.
    unreachable_budget = idle_timeout * 4
    last_log_text = ''

    def _record(outcome):
        """Fold one round's classification into the two clocks.

        Returns
        -------
        bool
            False when the poller is to give up and return None.
        """
        nonlocal quiet_since, unreachable_since, unreachable_attempts
        if outcome == POLL_PROGRESS:
            quiet_since = time.time()
            unreachable_since = None
            unreachable_attempts = 0
            return True
        if outcome == POLL_TRANSPORT_FAILURE:
            unreachable_attempts += 1
            if unreachable_since is None:
                unreachable_since = time.time()
            print(f'  WARNING: cannot reach {target} (attempt '
                  f'{unreachable_attempts}), not counting toward the idle '
                  f'window')
            if time.time() - unreachable_since > unreachable_budget:
                print(f'  Error: {target} has answered nothing for '
                      f'{int(time.time() - unreachable_since)}s '
                      f'(budget {unreachable_budget}s).')
                print('  Collecting by hand once the link is back:')
                for line in _manual_collection_route(target, port, jump_host,
                                                     remote_dir, resolved_dir):
                    print(f'    {line}')
                return False
            return True
        if outcome == POLL_QUIET:
            quiet = int(time.time() - quiet_since)
            if quiet > idle_timeout:
                print(f'Error: no progress from remote for {quiet}s '
                      f'(idle_timeout={idle_timeout}s).')
                print(f'The remote test may have hung. '
                      f'Check {remote_dir}/deploy_test.log on remote.')
                print(f'Use --force to terminate the remote test.')
                # Best-effort log retrieval: the remote log usually
                # exists at this point and helps diagnose the hang.
                _scp_deploy_log(target, port, jump_host, remote_dir,
                                tmp_dir=resolved_dir, ssh_base=ssh_base)
                return False
        return True

    while True:
        elapsed = int(time.time() - start_time)

        # Discover the latest tmp directory on remote (fallback only:
        # the trigger normally resolves the directory synchronously)
        if not resolved_dir:
            if elapsed > tmp_wait_timeout:
                print(f'Error: remote tmp directory not found after {elapsed}s.')
                print(f'The remote test may have failed to start. '
                      f'Check {remote_dir}/deploy_test.log on remote.')
                # Best-effort log retrieval: the trigger log may
                # already exist and explains the failed start.
                _scp_deploy_log(target, port, jump_host, remote_dir,
                                ssh_base=ssh_base)
                return None

            resolved_dir = _resolve_remote_tmp_dir(ssh_base, remote_dir)
            if resolved_dir:
                print(f'  Remote work dir: {resolved_dir}')
                quiet_since = time.time()
            else:
                print(_progress_line(last_completed, denominator,
                                     elapsed,
                                     'waiting for this run\'s tmp '
                                     'directory to appear...'))
                time.sleep(interval)
                continue

        # Check remote status
        check_cmd = list(ssh_base) + [
            f'cat {resolved_dir}/status.json 2>/dev/null'
        ]
        try:
            proc = subprocess.run(
                check_cmd, capture_output=True, text=True, timeout=30
            )
            if proc.returncode == 0 and proc.stdout.strip():
                status_data = json.loads(proc.stdout.strip())
                status = status_data.get('status', 'unknown')

                # Count completed tasks from results.json if available
                completed = _count_remote_results(ssh_base, resolved_dir)

                outcome = _poll_outcome(0, False, completed, last_completed,
                                        False, status)
                if outcome == POLL_FINISHED:
                    # Fetch full results via SSH
                    full_results = _fetch_remote_results(ssh_base, resolved_dir)
                    _scp_deploy_log(target, port, jump_host, remote_dir,
                                    tmp_dir=resolved_dir, ssh_base=ssh_base)
                    return full_results if full_results else status_data

                if completed > last_completed:
                    progress_info = f'+{completed - last_completed} new'
                    last_completed = completed
                else:
                    progress_info = 'no new results'

                # Report progress with completion info and liveness
                if status == 'running':
                    alive_rc, alive = _probe_liveness(ssh_base)
                    if alive_rc == 0:
                        alive_info = ('process alive' if alive
                                      else 'WARNING: process not found!')
                    else:
                        alive_info = 'WARNING: liveness probe did not answer'
                else:
                    alive_info = f'status: {status}'
                print(_progress_line(
                    completed, denominator, elapsed,
                    f'{progress_info}, {alive_info}'))
                if not _record(outcome):
                    return None

            else:
                # status.json not found or empty. The runner writes it only at
                # the end, so the trigger log is the live channel: it streams
                # one line per finished task as '[3/70] + name [mode] -> PASS'.
                log_rc, log_text = _tail_remote_log(ssh_base, remote_dir)
                completed, log_total = _parse_log_progress(log_text)
                denominator = _next_denominator(denominator, log_total)
                log_advanced = bool(log_text) and log_text != last_log_text
                last_log_text = log_text
                outcome = _poll_outcome(log_rc, not log_text.strip(),
                                        completed, last_completed,
                                        log_advanced)
                if outcome == POLL_PROGRESS:
                    last_completed = max(last_completed, completed)
                    print(_progress_line(completed, denominator, elapsed,
                                         'waiting for status.json, log '
                                         'advancing'))
                elif outcome == POLL_TRANSPORT_FAILURE:
                    print(_progress_line(last_completed, denominator, elapsed,
                                         'WARNING: remote probes did not '
                                         'answer'))
                else:
                    # Reachable, answered, nothing new. The name pattern of
                    # pgrep is one guess at how the command line looks (conda
                    # run, a wrapper, a symlinked entry point all defeat it),
                    # and on 2026-09-28 23:29 it declared a healthy suite
                    # crashed at 59% - the process was alive at task 71/107 and
                    # wrote results.json four minutes later. So silence is
                    # judged by the two clocks, not by one pgrep miss, and the
                    # liveness answer below only words the line: the round was
                    # already classified by what the log probe returned.
                    alive_rc, alive = _probe_liveness(ssh_base)
                    if alive_rc != 0:
                        detail = 'liveness probe did not answer'
                    elif alive:
                        detail = 'waiting for status.json...'
                    else:
                        detail = 'WARNING: remote process not found, log quiet'
                    print(_progress_line(last_completed, denominator, elapsed,
                                         detail))
                    if not alive and alive_rc == 0:
                        print(f'  Check {remote_dir}/deploy_test.log on '
                              f'remote if this persists.')
                if not _record(outcome):
                    return None

        except subprocess.TimeoutExpired:
            print(_progress_line(last_completed, denominator, elapsed,
                                 'SSH command timed out'))
            if not _record(POLL_TRANSPORT_FAILURE):
                return None
        except json.JSONDecodeError:
            print(_progress_line(last_completed, denominator, elapsed,
                                 'status.json parsing error'))
            if not _record(POLL_QUIET):
                return None
        except Exception as e:
            print(_progress_line(last_completed, denominator, elapsed,
                                 f'error: {e}'))
            if not _record(POLL_TRANSPORT_FAILURE):
                return None

        time.sleep(interval)


def _resolve_remote_tmp_dir(ssh_base, remote_dir):
    """Find the latest test work directory on the remote server.

    Uses ls -dt to sort by modification time and returns the most
    recent directory whose name matches tmp_<YYYYMMDD_HHMMSS>.
    Skill calculation directories (e.g. tmp_MOLCAS) share the tmp_
    prefix but carry no timestamp, so the pattern filter keeps them
    out of the polling target.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    remote_dir : str
        Remote project directory.

    Returns
    -------
    str
        Absolute path to the latest timestamped tmp directory, or
        empty string.
    """
    check_cmd = list(ssh_base) + [
        f'ls -dt {remote_dir}/tmp_*/ 2>/dev/null'
    ]
    try:
        proc = subprocess.run(
            check_cmd, capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                path = line.strip().rstrip('/')
                if not path:
                    continue
                if TMP_DIR_PATTERN.match(os.path.basename(path)):
                    return path
    except (subprocess.TimeoutExpired, Exception):
        pass

    return ''


def _check_remote_process_alive(ssh_base):
    """Check if the remote pyrai3md test process is still running.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.

    Returns
    -------
    bool
        True if process is running, False if not found or check failed.
    """
    return _probe_liveness(ssh_base)[1]


def _probe_liveness(ssh_base):
    """Ask the remote whether the test process is there, and say whether it answered.

    Returns `(returncode, alive)`. The second value alone cannot be trusted as
    evidence about the run: `False` is what a dropped connection reads as, and
    the poller used to count that against the round's idle window exactly like a
    process that is really gone. Round `tmp_20261018_173454` lost its collection
    that way at 19:37 while the remote was alive at 105 of 243 rows with zero
    failures; the link, not the run, had gone.
    """
    # Same pattern the stop uses: an unbracketed "pyrai3md test" also matches
    # the command line of the shell running the pgrep, which reports a dead
    # run as alive and lets the poller wait out its idle timeout on a process
    # that is already gone.
    check_cmd = list(ssh_base) + [
        "pgrep -f '%s' >/dev/null 2>&1 && echo alive || echo dead"
        % REMOTE_TEST_PATTERN
    ]
    try:
        proc = subprocess.run(
            check_cmd, capture_output=True, text=True, timeout=15
        )
    except Exception:
        # No answer at all. `None` is what the classifier reads as a transport
        # failure rather than as a dead process.
        return None, False
    if proc.returncode != 0:
        return proc.returncode, False
    return 0, 'alive' in proc.stdout.strip()


def _parse_log_progress(text):
    """The `(completed, total)` of the last marker in one log tail.

    The runner streams one line per finished task, formatted as
    '[idx/total] <marker> name [mode] -> STATUS', and the counter is zero-padded
    to the digit width of the total (the padding is what lets the detail lines
    align under it), so the pattern stays r'\\[(\\d+)/(\\d+)\\]' and int() reads
    it. Nothing here may switch to space padding: '[ 1/38]' does not match.

    Before the first row the round publishes its own denominator as the
    header line 'Total tests: N'. Reading that here is what lets the tool
    skip a pre-trigger planning run: the number the poller divides by comes
    from the round that is running, not from a second planning of it.
    """
    text = text or ''
    markers = re.findall(r'\[(\d+)/(\d+)\]', text)
    if markers:
        idx, total = markers[-1]
        return int(idx), int(total)
    header = re.search(r'Total tests:\s*(\d+)', text)
    if header:
        return 0, int(header.group(1))
    return 0, 0


def _tail_remote_log(ssh_base, remote_dir):
    """The last two hundred lines of the remote trigger log, and whether they came.

    Returns `(returncode, text)`; `returncode` is None when the call itself did
    not complete, which the poll classifies as a transport failure rather than
    as a log that stopped growing.
    """
    tail_cmd = list(ssh_base) + [
        f'tail -n 200 {remote_dir}/deploy_test.log 2>/dev/null'
    ]
    try:
        proc = subprocess.run(
            tail_cmd, capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None, ''
    if proc.returncode != 0:
        return proc.returncode, ''
    return 0, proc.stdout


# The four readings one polling round can produce, and the only place that
# decides between them.
POLL_PROGRESS = 'PROGRESS'
POLL_QUIET = 'QUIET'
POLL_TRANSPORT_FAILURE = 'TRANSPORT_FAILURE'
POLL_FINISHED = 'FINISHED'
_TERMINAL_STATUSES = ('pass', 'fail', 'error', 'skip')


def _poll_outcome(ssh_returncode, stdout_empty, completed_now, completed_prev,
                  log_advanced, status=''):
    """Classify one polling round: did the run move, go quiet, or stop answering?

    Parameters
    ----------
    ssh_returncode : int or None
        Exit code of the probe this round relied on; None or non-zero means the
        call itself failed.
    stdout_empty : bool
        The probe answered but carried nothing.
    completed_now : int
        Finished tasks read this round.
    completed_prev : int
        Finished tasks at the last reset of the idle clock.
    log_advanced : bool
        The remote log holds different text than it did last round, whether or
        not a task finished - a long leg writes step lines that change no
        counter, and treating those as silence is how a working run is declared
        lost.
    status : str
        The remote's own status field when `status.json` could be read.

    Returns
    -------
    str
        One of `POLL_PROGRESS`, `POLL_QUIET`, `POLL_TRANSPORT_FAILURE`,
        `POLL_FINISHED`. Only a QUIET round ages the idle window: a round that
        could not be reached says nothing about the run, and the poller that
        conflated the two stopped round `tmp_20261018_173454`'s collection ~2 h
        early while the remote sat at 105/243 with no failures.
    """
    if str(status or '').lower() in _TERMINAL_STATUSES:
        return POLL_FINISHED
    if ssh_returncode is None or int(ssh_returncode) != 0:
        return POLL_TRANSPORT_FAILURE
    if stdout_empty:
        return POLL_QUIET
    if int(completed_now) > int(completed_prev) or log_advanced:
        return POLL_PROGRESS
    return POLL_QUIET


def _manual_collection_route(target, port, jump_host, remote_dir, work_dir):
    """The commands that collect a finished round by hand, spelled out.

    What the poller prints when an outage outlasts its budget. They name the
    same two artifacts the automatic path fetches - the round's `results.json`
    in the work directory and the trigger log above it - and rebuild the
    transport from the same three values, so a jump-host deployment does not get
    a direct command the operator would have to correct by hand.
    """
    transport = 'ssh -p %s' % port
    if jump_host:
        transport += ' -J %s' % jump_host
    return [
        'rsync -az -e "%s" %s:%s/results.json .'
        % (transport, target, work_dir),
        'rsync -az -e "%s" %s:%s/deploy_test.log .'
        % (transport, target, remote_dir),
    ]


def _count_remote_results(ssh_base, tmp_dir):
    """Count completed test results from remote results.json.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    tmp_dir : str
        Remote tmp working directory path.

    Returns
    -------
    int
        Number of completed test results, or 0 if unavailable.
    """
    results_cmd = list(ssh_base) + [
        f'cat {tmp_dir}/results.json 2>/dev/null'
    ]
    try:
        proc = subprocess.run(
            results_cmd, capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout.strip())
            # The same four buckets the run writes; leaving benchmark out
            # made this count short of the denominator it is compared with.
            return sum(len(data.get(key, [])) for key in
                       ('main_results', 'example_results', 'skill_results',
                        'benchmark_results'))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

    return 0


def _fetch_remote_results(ssh_base, tmp_dir):
    """Fetch full test results from remote results.json via SSH.

    Parameters
    ----------
    ssh_base : list of str
        SSH command prefix.
    tmp_dir : str
        Remote tmp working directory path.

    Returns
    -------
    dict or None
        Full test results, or None if unavailable.
    """
    results_cmd = list(ssh_base) + [
        f'cat {tmp_dir}/results.json 2>/dev/null'
    ]
    try:
        proc = subprocess.run(
            results_cmd, capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

    return None


def _scp_deploy_log(target, port, jump_host, remote_dir, tmp_dir='',
                    local_log='', ssh_base=None):
    """Download the remote deploy_test.log into the current directory.

    The primary channel is an 'ssh cat' redirection over ssh_base,
    the same verified connection used for status polling, so the
    download inherits its direct/jump mode exactly. When ssh_base is
    unavailable or the cat fails, the download falls back to scp,
    whose jump usage must mirror the actual SSH connection (ssh via
    jump -> scp -J jump; ssh direct -> scp direct).

    The log is saved as test_<stamp>.log, where <stamp> is derived
    from the remote tmp_* work directory name (e.g.
    tmp_20260812_092706 -> test_20260812_092706.log) so each run
    keeps its own log. Falls back to the current timestamp when the
    tmp directory name is unavailable. Only the test log is fetched;
    the remote tmp_* work directory with full calculation outputs is
    intentionally left on the server.

    Parameters
    ----------
    target : str
        user@host string (scp fallback source).
    port : int
        SSH port (scp uses the uppercase -P flag).
    jump_host : str
        Jump/bastion host for scp -J (empty = direct connection);
        must match the jump state of the actual SSH connection.
    remote_dir : str
        Remote project directory containing deploy_test.log.
    tmp_dir : str
        Remote tmp_* work directory path (for naming the local log).
    local_log : str
        Explicit local file name (overrides the derived name).
    ssh_base : list of str or None
        Verified SSH command prefix; preferred download channel.
    """
    if not local_log:
        base = os.path.basename(tmp_dir.rstrip('/'))
        if base.startswith('tmp_'):
            local_log = f'test_{base[len("tmp_"):].strip("/")}.log'
        else:
            local_log = f'test_{time.strftime("%Y%m%d_%H%M%S")}.log'

    # Preferred channel: cat over the verified SSH connection. The
    # remote command may fail without a log file present, so every
    # failure path falls through to the scp attempt.
    if ssh_base:
        try:
            cat_cmd = list(ssh_base) + [
                f'cat {remote_dir}/deploy_test.log'
            ]
            proc = subprocess.run(
                cat_cmd, capture_output=True, text=True, timeout=60
            )
            if proc.returncode == 0 and proc.stdout:
                with open(local_log, 'w', encoding='utf-8') as f:
                    f.write(proc.stdout)
                print(f'  Deploy log saved to: '
                      f'{os.path.abspath(local_log)}')
                return
        except subprocess.TimeoutExpired:
            print('  Warning: ssh cat of deploy_test.log timed out; '
                  'falling back to scp.')
        except Exception as e:
            print(f'  Warning: ssh cat of deploy_test.log failed '
                  f'({e}); falling back to scp.')

    scp_cmd = _build_scp_command(target, port, jump_host)
    scp_cmd.append(f'{target}:{remote_dir}/deploy_test.log')
    scp_cmd.append(local_log)

    try:
        proc = subprocess.run(
            scp_cmd, capture_output=True, text=True, timeout=60
        )
        if proc.returncode == 0:
            print(f'  Deploy log saved to: {os.path.abspath(local_log)}')
        else:
            print('  Warning: failed to download deploy_test.log '
                  'from remote.')
    except subprocess.TimeoutExpired:
        print('  Warning: deploy_test.log download timed out.')
    except Exception as e:
        print(f'  Warning: deploy_test.log download failed: {e}')


def _format_result_name(result):
    """Format a result display name with its theory when present.

    Example qualification results carry the theory (e.g. pyscf)
    resolved from the example YAML; appending it keeps the local
    summary consistent with the remote per-test output.

    Parameters
    ----------
    result : dict
        Single test result entry from results.json.

    Returns
    -------
    str
        Display name, e.g. 'example_energy_yaml [pyscf]'.
    """
    name = result.get('name', 'unknown')
    theory = result.get('theory', '')
    return f'{name} [{theory}]' if theory else name


def _print_summary(results):
    """Print a brief summary of collected results.

    Parameters
    ----------
    results : dict
        Test results from remote.
    """
    print('\n' + '=' * 50)
    print('  Remote Test Summary')
    print('=' * 50)

    main_results = results.get('main_results', [])
    example_results = results.get('example_results', [])
    skill_results = results.get('skill_results', [])
    # runner.run_tests aggregates four categories (its own report line
    # sums the same four); leaving benchmark_results out of this one made
    # the tool announce 25 fewer tasks than the run it was describing, and
    # a failing benchmark would have been counted in neither the totals nor
    # the failure listing below.
    benchmark_results = results.get('benchmark_results', [])
    all_results = (main_results + example_results + skill_results
                   + benchmark_results)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get('status') == 'pass')
    failed = sum(1 for r in all_results if r.get('status') == 'fail')
    errors = sum(1 for r in all_results if r.get('status') == 'error')
    # runner.run_tests folds a bucket's switch-closed tasks into one row that
    # carries their number in 'skip_count', so the skip total reads that and
    # not the row count. Both units are named, because `Total` alone reads as
    # if Pass + Skip should add up to it.
    skip_rows = sum(1 for r in all_results if r.get('status') == 'skip')
    skipped = sum(r.get('skip_count', 1) for r in all_results
                  if r.get('status') == 'skip')

    # Overall verdict is binary (pass/fail); the skip count is shown
    # as a bracketed suffix, e.g. PASS[SKIP 4].
    status = results.get('status', 'unknown').upper()
    if skipped:
        status += f'[SKIP {skipped}]'
    print(f'  Status: {status}')

    summary = (f'  Total rows: {total}  Pass: {passed}  Fail: {failed}  '
               f'Error: {errors}')
    if skip_rows:
        summary += f'  Skip rows: {skip_rows} ({skipped} skipped tasks)'
    print(summary)
    print('=' * 50)

    # Print error and failure details
    for r in all_results:
        name = _format_result_name(r)
        for tb in r.get('error_tracebacks', []):
            print(f'\n  ERROR: {tb.get("test", name)}')
            print(f'  {tb.get("traceback", "").strip()}')
        for tb in r.get('failure_tracebacks', []):
            print(f'\n  FAIL: {tb.get("test", name)}')
            print(f'  {tb.get("traceback", "").strip()}')
        # Print stderr for subprocess failures
        if r.get('status') == 'fail' and r.get('stderr'):
            print(f'\n  FAIL: {name}')
            print(f'  {r["stderr"].strip()[-1000:]}')

    # Print skip reasons. A folded row speaks for every task its bucket's
    # switches closed and lists the keys in `details`, so it gets one block;
    # printing its tasks individually is the 267-line tail this replaced.
    for r in all_results:
        if r.get('status') != 'skip' or not r.get('skip_reason'):
            continue
        name = _format_result_name(r)
        print(f'\n  SKIP: {name}')
        print(f'  Reason: {r["skip_reason"]}')
        if r.get('closed_reasons'):
            # The same wrapped key list the runner's own report prints, so the
            # two readers of results.json say the same sentence.
            print(textwrap.fill(r.get('details', ''), width=76,
                                initial_indent='  Closed: ',
                                subsequent_indent='    '))
        elif r.get('input_file'):
            print(f'  Input:  {r["input_file"]}')

    print()


if __name__ == '__main__':
    main()
