"""
======================================================================
Unit tests for the skill agent calculate() forwarding.

Every skill agent entry point must forward the dynamic calculation
inputs (target_states, point_charges) from the agent config to
backend.calculate(), matching the framework calculator call.

Aug 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
from types import SimpleNamespace
from unittest import mock

import pytest


AGENT_MODULES = (
    'pyrai3md.skills.bagel_skill.agent',
    'pyrai3md.skills.cp2k_skill.agent',
    'pyrai3md.skills.lammps_skill.agent',
    'pyrai3md.skills.molcas_skill.agent',
    'pyrai3md.skills.openqp_skill.agent',
    'pyrai3md.skills.orca_skill.agent',
    'pyrai3md.skills.pyscf_skill.agent',
    'pyrai3md.skills.xtb_skill.agent',
)


def _stub_backend(point_charges, target_states):
    """Build a backend stub exposing the flat config attributes."""
    config = SimpleNamespace(
        atoms=['H', 'H'],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        point_charges=point_charges,
        target_states=target_states,
    )
    calculate = mock.Mock(return_value={'energy': [0.0]})
    return SimpleNamespace(config=config, calculate=calculate)


@pytest.mark.parametrize('module_name', AGENT_MODULES)
def test_run_calculation_forwards_target_states(module_name):
    module = importlib.import_module(module_name)
    backend = _stub_backend(
        point_charges=[[1.0, 0.0, 0.0, 0.5]], target_states=[0, 1])

    results = module._run_calculation(backend)

    assert results == {'energy': [0.0]}
    backend.calculate.assert_called_once_with(
        ['H', 'H'], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        point_charges=[[1.0, 0.0, 0.0, 0.5]],
        target_states=[0, 1])


@pytest.mark.parametrize('module_name', AGENT_MODULES)
def test_run_calculation_empty_inputs_become_none(module_name):
    module = importlib.import_module(module_name)
    backend = _stub_backend(point_charges=[], target_states=[])

    module._run_calculation(backend)

    # Empty config values must not leak empty lists into the backend
    backend.calculate.assert_called_once_with(
        ['H', 'H'], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        point_charges=None, target_states=None)


def test_molcas_agent_forwards_external_potential():
    """The molcas backend is the skill that consumes the ESPF potential."""
    module = importlib.import_module('pyrai3md.skills.molcas_skill.agent')
    backend = _stub_backend(point_charges=[], target_states=[])
    potential = [[0.1, 0.0, 0.0, 0.0],
                 [0.2, 0.0, 0.1, 0.0]]
    backend.config.external_potential = potential

    module._run_calculation(backend)

    backend.calculate.assert_called_once_with(
        ['H', 'H'], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        point_charges=None, target_states=None,
        external_potential=potential)
