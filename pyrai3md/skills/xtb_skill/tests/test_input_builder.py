"""
Unit tests for xtb_skill command-line construction (input_builder).

Verifies that every profile runs xTB with '--grad': the output
parser reads the total energy from the engrad file, which xTB only
writes for gradient runs, so energy-only profiles must request the
gradient as well (remote gfn0_energy failures traced to a missing
engrad file).
These tests do NOT require an xTB installation.

Run: pytest xtb_skill/tests/test_input_builder.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.builder.input_builder import build_xtb_args
from ..service.builder.input_builder import write_input_file
from ..service.workflow import compile_workflow

# Water molecule keeps the config minimal
ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]


def _make_config(method, profile, **kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': profile,
        'method': method,
    }
    raw.update(kwargs)
    return normalize(build_config(raw))


class TestGradFlagAlwaysPresent:
    """--grad stays on for energy profiles: the parser reads the
    total energy from the engrad file, which xTB only writes when
    the gradient is requested."""

    @pytest.mark.parametrize('method, profile', [
        ('gfn0', 'gfn0_energy'),
        ('gfn1', 'gfn1_energy'),
        ('gfn2', 'gfn2_energy'),
        ('gfn_ff', 'gfn_ff_energy'),
    ])
    def test_energy_profiles_request_engrad(self, method, profile):
        config = _make_config(method, profile)
        workflow = compile_workflow(profile)
        assert workflow.needs_gradient is False

        args = build_xtb_args(config, workflow)

        assert '--grad' in args.split()
        # The charges are a returned property (computed_charges), and xTB
        # only writes the charge file when the population is requested.
        assert '--pop' in args.split()

    @pytest.mark.parametrize('method, profile', [
        ('gfn0', 'gfn0_gradient'),
        ('gfn2', 'gfn2_gradient'),
    ])
    def test_gradient_profiles_keep_grad(self, method, profile):
        config = _make_config(method, profile)
        args = build_xtb_args(config, compile_workflow(profile))
        assert '--grad' in args.split()

    def test_method_flag_matches_method(self):
        config = _make_config('gfn0', 'gfn0_energy')
        args = build_xtb_args(config, compile_workflow('gfn0_energy'))
        assert '--gfn 0' in args

        config = _make_config('gfn_ff', 'gfn_ff_energy')
        args = build_xtb_args(config, compile_workflow('gfn_ff_energy'))
        assert '--gfnff' in args

    def test_charge_and_uhf_flags(self):
        # charges enters through the raw config; uhf is derived by
        # normalize as multiplicity - 1. The OH radical (9 electrons)
        # keeps the doublet multiplicity parity-consistent.
        raw = {
            'atoms': ['O', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
            'charges': -2,
            'multiplicity': 2,
            'nstate': 1,
            'state_spin': [1],
            'profile': 'gfn2_energy',
            'method': 'gfn2',
        }
        config = normalize(build_config(raw))
        args = build_xtb_args(config, compile_workflow('gfn2_energy'))
        assert '--chrg -2' in args
        assert '--uhf 1' in args


CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


class TestAPeriodicCellNeedsAMethodThatCarriesIt:
    """The cell is refused where it becomes geometry, not only at load time.

    `schema/validate.py` already asks whether the method supports `pbc`, but it
    can only see a `pbc` the input wrote on the skill itself. A framework run
    carries the cell on the system and hands it to the backend per evaluation,
    so an example that declared its box under `molecule:` reached xtb with
    `$periodic 3` on a GFN2 run: round `tmp_20261015_211219` stopped inside the
    engine's own SCF with `scf: Multipoles not available with PBC`, two carriers
    and no diagnosis available before the launch.
    """

    def test_a_non_periodic_method_with_a_cell_is_refused_before_the_run(
            self, tmp_path):
        # The load-time check cannot fire here: this config carries no pbc of
        # its own, only the cell the framework applies per evaluation.
        config = _make_config('gfn2', 'gfn2_gradient')
        config.system.pbc = [1, 1, 1]
        config.system.lattice_vector = CELL
        with pytest.raises(ValueError, match='does not support pbc'):
            write_input_file(config, compile_workflow('gfn2_gradient'),
                             str(tmp_path))
        assert not list(tmp_path.glob('*.coord')), (
            'the periodic coord was written anyway, so the engine call the '
            'round made would still be reachable')

    def test_the_periodic_method_still_writes_its_cell(self, tmp_path):
        # The control that keeps the refusal above from being a blanket ban:
        # GFN-FF is the method the capability table marks periodic, and this
        # is the path `examples/optimize_cell_xtb.yaml` runs on.
        config = _make_config('gfn_ff', 'gfn_ff_gradient', pbc=[1, 1, 1],
                              lattice_vector=CELL, stress=True)
        write_input_file(config, compile_workflow('gfn_ff_gradient'),
                         str(tmp_path))
        coord = next(iter(tmp_path.glob('*.coord')), None)
        assert coord is not None, 'GFN-FF lost its periodic geometry writer'
        text = coord.read_text()
        assert '$periodic 3' in text and '$lattice' in text

    def test_the_two_checks_read_one_capability_source(self):
        # A second table would drift, and the drift is invisible until an
        # engine stops: both the schema and the geometry writer must consult
        # the predicate, not restate the lookup.
        import inspect

        from ..schema import validate as schema_validate
        writer = inspect.getsource(write_input_file)
        guard = inspect.getsource(schema_validate.validate_system)
        assert 'method_supports_pbc(' in writer, writer
        assert 'method_supports_pbc(' in guard, guard
        for source in (writer, guard):
            assert "get('supports_pbc'" not in source, (
                'a consumer still reads the capability table directly: '
                + source[:120])
