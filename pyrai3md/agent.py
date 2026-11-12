"""
======================================================================
This module provides the AI agent entry point for PyRAI3MD.

It is parallel to pmd.py but uses interface.loader for agent task
configuration instead of interface.parser. Shares the same
Build Config -> Normalize -> Validate -> Build Skill Config ->
Build State -> Execute pipeline with pmd.py.

Usage (CLI):
    python -m pyrai3md.agent config.json

Usage (programmatic):
    from pyrai3md.agent import PyRAI3Agent
    result = PyRAI3Agent.from_dict({...}).run()

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import sys

from .interface.loader import load_config
from .schema.adapter import build_config
from .schema.normalize import normalize
from .schema.validate import validate
from .data.build import build_state


class PyRAI3Agent:
    """Agent entry point for PyRAI3MD.

    Parallel to pmd.py but uses interface.loader for
    agent task configuration instead of interface.parser.
    Shares the same pipeline with pmd.py.

    Parameters
    ----------
    raw_config : dict
        Raw configuration from interface.loader.

    Attributes
    ----------
    raw_config : dict
        The raw configuration dictionary.
    configs : dict or None
        The pipeline configs dict (after initialize()).
    state : PyRAIState or None
        The constructed runtime state (after initialize()).
    """

    def __init__(self, raw_config):
        self.raw_config = raw_config
        self.config = None
        self.configs = None
        self.state = None
        self.skills = None

    @classmethod
    def from_file(cls, config_file):
        """Create agent from a JSON config file.

        Uses interface.loader.load_config to load.

        Parameters
        ----------
        config_file : str
            Path to the JSON configuration file.

        Returns
        -------
        PyRAI3Agent
            A new agent instance.
        """
        raw_config = load_config(config_file)
        return cls(raw_config)

    @classmethod
    def from_dict(cls, config):
        """Create agent from a dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        PyRAI3Agent
            A new agent instance.
        """
        return cls(config)

    def initialize(self):
        """Run the initialization pipeline.

        build_config -> normalize -> validate -> build_state

        Returns
        -------
        self
            The initialized agent instance.
        """
        # Step 1: Build PyRAIConfig (includes Environment init)
        self.config = build_config(self.raw_config)

        # Step 2: Normalize -- compute derived quantities
        self.config = normalize(self.config)

        # Step 3: Validate -- check consistency
        self.config = validate(self.config)

        # Step 4: Build PyRAIState (static task queue; backend
        # instances are created at runtime by Calculator.initialize())
        self.state = build_state(self.config, raw_config=self.raw_config)

        return self

    def run(self):
        """Execute the task and return results (programmatic entry).

        Writes no run log: a library caller owns its output. The CLI
        (`run_cli`, below) uses the same logger as `pyrai3md` so a
        skill-driven optimization that stopped on its step budget carries
        the same `Status: failed: ... did not converge` line on this
        carrier as on the classic-input carrier.

        Returns
        -------
        Result
            Calculation results.
        """
        if self.state is None:
            self.initialize()
        from .workflow import execute
        return execute(self.state)

    def run_cli(self):
        """Execute the CLI run with the framework's RunLogger.

        `run` is the programmatic entry and writes nothing; the command
        line is a run like any other, so it goes through
        `pmd.run_with_logger` (header, result block, closing status). Before
        this split `pyrai3md_agent` exited 0 without a status artifact, and
        a non-converged optimization was only visible on the
        `pyrai3md_input` leg (round tmp_20261110_202158).

        Returns
        -------
        Result
            Calculation results.
        """
        if self.state is None:
            self.initialize()
        from .pmd import run_with_logger
        return run_with_logger(self.state)


def main():
    """Main function for running the agent as a script."""
    if len(sys.argv) < 2:
        print("Usage: python -m pyrai3md.agent <input.config>[JSON]")
        sys.exit(1)

    config_file = sys.argv[1]
    agent = PyRAI3Agent.from_file(config_file)
    result = agent.run_cli()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
