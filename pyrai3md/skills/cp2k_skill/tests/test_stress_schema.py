"""
Unit tests for the stress route settings of the CP2K skill schema
(schema/config.py, schema/reference.py, schema/validate.py).

The stress channel has three evaluation routes and only one shape of
input reaches each of them, so the tests below pin both halves: that a
usable request is routed to the fields the service layer reads, and that
an unusable one is refused by name instead of running and returning a
zero-filled tensor.

The refusal of `analytical` stress under `method: tddft` is not
a style preference: CP2K aborts with 'Stress Tensor not available for
TDDFT calculations' (response_solver.F:943, measured on the
deployment), while `num` asks the engine for no tensor at all and is
therefore the route that may carry an excited state.

These tests do NOT require a CP2K installation.

Run: pytest cp2k_skill/tests/test_stress_schema.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.reference import STRESS_METHODS, STRESS_SCOPES
from ..schema.validate import validate

# A 6.3 Angstrom cubic box: the strain route only means something for a
# full 3D periodic system, so every case below carries one.
LATTICE = [[6.3, 0.0, 0.0], [0.0, 6.3, 0.0], [0.0, 0.0, 6.3]]
CELL = [6.3, 6.3, 6.3, 90.0, 90.0, 90.0]
ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]]


def _config(**overrides):
    """Run the real schema pipeline (build -> normalize -> validate)."""
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': 'dft_energy',
        'method': 'dft',
        'functional': 'pbe',
        'project': 'stress_route',
        'pbc': [1, 1, 1],
        'cell': CELL,
        'lattice_vector': LATTICE,
    }
    raw.update(overrides)
    return validate(normalize(build_config(raw)))


def _tddft(**overrides):
    overrides.setdefault('profile', 'tddft_energy')
    overrides.setdefault('method', 'tddft')
    overrides.setdefault('nstate', 3)
    overrides.setdefault('state_spin', [0, 0, 0])
    return _config(**overrides)


class TestDefaults:
    """An untouched config keeps the analytical ground-state route."""

    def test_stress_off_with_analytical_method(self):
        config = _config()
        assert config.calculation.stress is False
        assert config.calculation.stress_method == 'analytical'

    def test_strain_parameters_carry_the_documented_start(self):
        config = _config()
        # 1e-3 is the recommended starting step of
        # dev/skills/CP2K_SKILL_DEVELOPMENT.md section 6.6 (step size),
        # not a tuned value.
        assert config.calculation.stress_delta == pytest.approx(1.0e-3)
        assert config.calculation.stress_scope == 'full'

    def test_vocabularies_are_declared_once(self):
        assert list(STRESS_METHODS) == ['analytical', 'num']
        assert list(STRESS_SCOPES) == ['full', 'iso']


class TestTheRouteIsResolvedFromTheMethod:
    """A run that leaves `stress_method` unwritten gets the route its method has.

    The framework asks for a tensor through the `stress` capability and knows
    nothing about engines (``planning/resolver.py::_derive_stress``), so before
    this resolution a planned variable-cell run on a TDDFT surface arrived here,
    fell on the default 'analytical', and was refused - the route CP2K aborts on
    ('Stress Tensor not available for TDDFT calculations', response_solver.F:943,
    measured in rounds tmp_20261010_133424 and tmp_20261010_190544) - even though
    the strain route answers the same request with (nstate, 3, 3).
    """

    def test_a_ground_state_run_still_resolves_to_the_engine_tensor(self):
        config = _config()
        assert config.calculation.stress_method == 'analytical'

    def test_an_excited_state_run_resolves_to_the_strain_route(self):
        config = _tddft()
        assert config.calculation.stress_method == 'num'

    def test_the_resolution_alone_says_nothing(self):
        # The route follows from the method, so it is resolved even when nobody
        # asks for a tensor. It is not an event, and a run that wants no stress
        # must not be warned about one: any warning at all fails this case.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            config = _tddft()
        assert config.calculation.stress_method == 'num'

    def test_resolving_it_for_a_requested_tensor_says_why(self):
        with pytest.warns(UserWarning, match='resolved it to .*num'):
            _tddft(stress=True)

    def test_a_written_analytical_on_an_excited_state_is_still_refused(self):
        # The negative control: resolution fills what is absent, it does not
        # overrule what was asked for.
        with pytest.raises(ValueError, match='not available'):
            _tddft(stress=True, stress_method='analytical')

    def test_a_written_route_on_a_ground_state_run_is_kept(self):
        config = _config(stress=True, stress_method='num')
        assert config.calculation.stress_method == 'num'

    def test_an_unknown_spelling_is_refused_before_any_resolution(self):
        # The empty string means "not written"; a misspelling must not be
        # swallowed by the same branch.
        with pytest.raises(ValueError, match='unknown stress_method'):
            _config(stress=True, stress_method='numm')


class TestNum2Routing:
    """The finite-difference route reaches the fields the driver reads."""

    def test_num_fields_are_kept(self):
        config = _config(stress=True, stress_method='num',
                         stress_delta=5.0e-4, stress_scope='iso')
        assert config.calculation.stress is True
        assert config.calculation.stress_method == 'num'
        assert config.calculation.stress_delta == pytest.approx(5.0e-4)
        assert config.calculation.stress_scope == 'iso'

    def test_explicit_false_on_stress_is_not_overwritten(self):
        # An explicit False on a bool switch must survive the default
        # fill, or a run could not turn stress off once a preset sets it.
        config = _config(stress=False, stress_method='num')
        assert config.calculation.stress is False

    def test_tddft_may_carry_the_num_route(self):
        # This is the point of the route: the engine refuses a tensor,
        # the energy difference does not ask for one.
        config = _tddft(stress=True, stress_method='num')
        assert config.calculation.stress_method == 'num'
        assert config.calculation.tddft_nstates == 2


class TestKeywordRouting:
    """The &cp2k dispatch keywords reach the skill, and nothing else."""

    def test_dispatch_keywords_are_received(self):
        config = _config(multitask=True, multipool=4, remotetask=True,
                         execution_mode='slurm')
        assert config.environment.multitask is True
        assert config.environment.multipool == 4
        assert config.environment.remotetask is True

    def test_dispatch_keywords_are_undeclared_by_default(self):
        config = _config()
        assert config.environment.multitask is None
        assert config.environment.multipool is None
        assert config.environment.remotetask is None

    def test_framework_none_values_are_accepted(self):
        # The framework merge hands these keys over with a None value
        # whenever the user did not write them; a strict bool/int field
        # would reject that and break every framework-driven CP2K run.
        config = _config(multitask=None, remotetask=None, multipool=None)
        assert config.environment.multitask is None
        assert config.environment.multipool is None

    def test_zero_pool_is_refused(self):
        # Unlike the numeric settings whose DEFAULT is a real number
        # (`stress_delta`, where a written 0 is the unset marker), the
        # dispatch keywords default to None: a written 0 is not the empty
        # marker here, it reaches the model and is refused.
        with pytest.raises(Exception, match='multipool'):
            _config(multitask=True, multipool=0)

    def test_explicit_pool_below_one_is_refused(self):
        # A config that reaches the model directly (bypassing the
        # default fill) is refused, not silently run serially.
        from ..schema.config import Environment
        with pytest.raises(Exception, match='multipool'):
            Environment(multipool=0)

    def test_remotetask_without_the_slurm_arm_is_refused(self):
        # remotetask asks for the remote arm; execution_mode selects it.
        # Neither outranks the other, so a mismatch is reported instead
        # of silently submitting jobs the input did not ask for.
        with pytest.raises(Exception, match='execution_mode'):
            _config(remotetask=True)

    def test_slurm_arm_without_remotetask_is_allowed(self):
        # The legacy standalone spelling stays valid: asking for slurm
        # through execution_mode alone is not a contradiction.
        config = _config(execution_mode='slurm')
        assert config.environment.remotetask is None


class TestRefusals:
    """Unusable stress requests are refused by name, not run empty."""

    def test_unknown_stress_method_is_refused(self):
        with pytest.raises(ValueError, match='unknown stress_method'):
            _config(stress=True, stress_method='numerical')

    def test_unknown_stress_scope_is_refused(self):
        with pytest.raises(ValueError, match='unknown stress_scope'):
            _config(stress=True, stress_method='num', stress_scope='shear')

    def test_strain_step_out_of_range_is_refused(self):
        for delta in (-1.0e-3, 0.5):
            with pytest.raises(ValueError, match='stress_delta'):
                _config(stress=True, stress_method='num',
                        stress_delta=delta)

    def test_zero_strain_step_is_read_as_unset(self):
        # Shared schema convention: a zero/empty numeric value is the
        # default-fill trigger, not a refused value (cutoff behaves the
        # same), so writing `stress_delta: 0` asks for the documented
        # starting step of 1e-3 rather than failing the run.
        config = _config(stress=True, stress_method='num', stress_delta=0.0)
        assert config.calculation.stress_delta == pytest.approx(1.0e-3)

    def test_num_needs_full_periodicity(self):
        with pytest.raises(ValueError, match='full 3D periodicity'):
            _config(stress=True, stress_method='num', pbc=[0, 0, 0])

    def test_num_needs_a_non_singular_lattice(self):
        flat = [[6.3, 0.0, 0.0], [0.0, 6.3, 0.0], [0.0, 0.0, 0.0]]
        with pytest.raises(ValueError, match='lattice_vector'):
            _config(stress=True, stress_method='num', lattice_vector=flat)

    def test_tddft_with_an_engine_tensor_request_is_refused(self):
        with pytest.raises(ValueError, match='TDDFT'):
            _tddft(stress=True, stress_method='analytical')

    def test_retired_numerical_route_is_refused_by_name(self):
        # `num1` (CP2K's own STRESS_TENSOR NUMERICAL) is retired: the
        # engine gives no numerical stress for a TDDFT run and the
        # analytical tensor already covers DFT, so the vocabulary has two
        # members. An old input asking for it must be refused by name,
        # not read as the surviving route.
        with pytest.raises(ValueError, match='unknown stress_method'):
            _config(stress=True, stress_method='num1')

    def test_stress_off_needs_no_periodic_cell(self):
        # The refusals are conditional on the request: a molecular DFT
        # run without stress stays untouched by the periodic checks.
        config = _config(pbc=[0, 0, 0],
                         lattice_vector=[[0.0] * 3 for _ in range(3)])
        assert config.calculation.stress is False
