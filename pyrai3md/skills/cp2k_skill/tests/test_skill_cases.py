"""
Loads every test case shipped in this skill's tests/ directory through
the skill's own schema pipeline (build_config -> normalize -> validate).

A case file that the pipeline refuses, or that carries a keyword the
schema does not list, is not a pending run: it is a case that will never
certify anything, because the framework filter drops unknown keys in
silence and a wrong keyword reads as an unset one. The same argument
covers the 'expected' block: a shape that contradicts the case's own
atoms/nstate is a gate that can never be met.

CP2K is not needed: nothing is executed here.

Run: pytest cp2k_skill/tests/test_skill_cases.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import glob
import os

import pytest
import yaml

from ..schema.adapter import build_config
from ..schema.config import FIELD_TO_SUBMODEL
from ..schema.normalize import normalize
from ..schema.validate import validate

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_FILES = sorted(glob.glob(os.path.join(TESTS_DIR, '*.yaml')))


def _case_id(path):
    return os.path.splitext(os.path.basename(path))[0]


@pytest.fixture(params=CASE_FILES, ids=_case_id)
def case(request):
    """One shipped case: its raw input, and the config the skill builds."""
    with open(request.param, 'r') as handle:
        document = yaml.safe_load(handle)
    raw = dict(document['input'])
    # 'name' and 'version' describe the case file, not the calculation.
    for meta in ('name', 'version'):
        raw.pop(meta, None)
    unknown = sorted(key for key in raw if key not in FIELD_TO_SUBMODEL)
    assert not unknown, (
        f'{_case_id(request.param)} sets {unknown}, which this skill\'s '
        'schema does not list: the framework filter drops it, so the '
        'case would run without the setting it claims to certify.')
    config = validate(normalize(build_config(raw)))
    return config, document


def test_cases_are_shipped():
    # The parametrized fixture above is worth nothing if discovery comes
    # back empty: an empty list silently turns every case assertion into
    # a pass.
    assert len(CASE_FILES) >= 7, CASE_FILES


class TestCaseLoads:
    """Each case survives the real pipeline and says what it means."""

    def test_profile_and_method_agree(self, case):
        config, document = case
        assert config.calculation.profile == document['input']['profile']
        assert config.calculation.method == document['input']['method']

    def test_expected_shapes_match_the_case_system(self, case):
        config, document = case
        nstate = int(config.system.nstate)
        natom = len(config.system.atoms)
        assert nstate == int(document['input'].get('nstate', 1))
        for name, block in (document.get('expected') or {}).items():
            if 'shape' not in block:
                # A case may pin a value and a tolerance instead of a
                # shape (dft_opt_min does that for the energy); there is
                # no shape here to check against the system.
                continue
            shape = list(block['shape'])
            if name == 'stress_tensor':
                assert shape == [nstate, 3, 3], (name, shape)
            elif name == 'gradient':
                assert shape == [nstate, natom, 3], (name, shape)
            elif name == 'energy':
                assert shape == [nstate], (name, shape)

    def test_stress_cases_name_a_supported_route(self, case):
        config, _document = case
        if not config.calculation.stress:
            return
        route = config.calculation.stress_method
        if config.calculation.method == 'tddft':
            # The engine refuses a tensor for an excited state, so only
            # the finite-difference route may ask for stress here.
            assert route == 'num', (config.calculation.method, route)
        assert route in ('analytical', 'num')

    def test_numerical_stress_cases_ask_for_a_periodic_cell(self, case):
        config, _document = case
        if config.calculation.stress_method != 'num':
            return
        assert list(config.system.pbc) == [1, 1, 1]
        assert config.calculation.stress_scope in ('full', 'iso')
        assert 0.0 < config.calculation.stress_delta <= 1.0e-2


class TestStressCaseCoverage:
    """The two strain scopes and the default fill are both certified."""

    @staticmethod
    def _inputs():
        for path in CASE_FILES:
            with open(path, 'r') as handle:
                yield os.path.basename(path), yaml.safe_load(handle)['input']

    def test_both_stress_scopes_are_covered(self):
        scopes = {raw.get('stress_scope', 'full')
                  for _, raw in self._inputs()
                  if raw.get('stress_method') == 'num'}
        assert scopes == {'full', 'iso'}, scopes

    def test_one_case_leaves_the_strain_settings_to_their_defaults(self):
        # A value pinned in every case is a value no case exercises.
        num = [raw for _, raw in self._inputs()
                if raw.get('stress_method') == 'num']
        assert num, 'no num case is shipped'
        assert any('stress_delta' not in raw and 'stress_scope' not in raw
                   for raw in num), num
