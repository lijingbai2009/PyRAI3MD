"""
======================================================================
This module provides the agent task configuration loader for PyRAI3MD.
It loads agent task configurations from JSON files and performs minimal
normalization (e.g., lowercase section names).

This is the counterpart to interface.parser for the AI agent entry point.
Named 'loader' to avoid conflict with schema/config.py (Pydantic models).

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os


def load_config(config_file):
    """Load an agent task configuration from a JSON file.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Raw configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError("Config file not found: %s" % config_file)

    with open(config_file, 'r') as f:
        config = json.load(f)

    return normalize(config)


def normalize(config):
    """Normalize a configuration dictionary.

    Minimal processing: normalize section names to lowercase.
    No validation or type conversion is performed here.

    Parameters
    ----------
    config : dict
        Raw configuration dictionary.

    Returns
    -------
    dict
        Normalized configuration dictionary with lowercase keys.
    """
    normalized = {}
    for key, value in config.items():
        normalized[key.lower()] = value
    return normalized
