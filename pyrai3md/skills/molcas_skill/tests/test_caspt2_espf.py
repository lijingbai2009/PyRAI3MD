"""
Unit tests for CASPT2 variants, ESPF embedding and generic profiles.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.method import resolve_profile
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder.input_builder import build_input, nac_deck_order
from ..service.builder.module_builders import build_alaska, build_caspt2
from ..service.builder.embedding import build_esfp_block, resolve_embedding_field
from ..service.parser.log_parser import parse_log
from ..service.workflow.compiler import compile_workflow


def _config(**overrides):
    data = dict(
        project='t', method='caspt2', profile='caspt2_properties',
        basis='ano-l-vdzp', jobtype='energy',
        atoms=['O', 'H', 'H'],
        coordinates=[[0, 0, 0.1173], [0, 0.7572, -0.4692],
                     [0, -0.7572, -0.4692]],
        charges=0, multiplicity=1, nstate=3,
        state_spin=[0, 0, 0], ci_roots=[3], ci_spin=[0],
        target_states=[0], target_nac_pair=[[]], target_soc_pair=[[]],
        nact_elec=6, nact_orb=5, inactive_orb=2, ras2_orb=5,
        embedding='none', read_charge='none',
    )
    data.update(overrides)
    return validate(normalize(build_config(data)))


def test_caspt2_variant_keywords():
    checks = {
        'ss': None,
        'ms': 'MULTISTATE',
        'xms': 'XMULTISTATE',
        'rms': 'RMULTISTATE',
        'xdw': 'XMULTISTATE',
    }
    for ctype, keyword in checks.items():
        kwargs = dict(caspt2_type=ctype)
        if ctype == 'ss':
            kwargs.update(nstate=1, state_spin=[0], ci_roots=[1],
                          target_states=[0])
        block = build_caspt2(_config(**kwargs), gradient=True)
        if keyword is None:
            assert 'MULTISTATE' not in block
        else:
            assert keyword in block
    xdw = build_caspt2(_config(caspt2_type='xdw'))
    assert 'DWMS' in xdw and 'DWTYPE' in xdw


def test_caspt2_nac_nocsf_and_gradient_markers():
    cfg = _config(jobtype='energy gradient nac',
                  target_nac_pair=[[0, 2]], nocsf=True)
    caspt2 = build_caspt2(cfg, gradient=True)
    for token in ('GRDT', 'DORT', 'IMAG'):
        assert token in caspt2
    assert 'NAC' not in caspt2
    alaska = build_alaska(
        cfg, nac_mode=True, root_offset=0, nac_pairs=[[0, 2]],
        nocsf=True)
    for token in ('ALASKA', 'NAC', 'NOCSF'):
        assert token in alaska


def test_generic_profile_resolution():
    assert resolve_profile('caspt2', 'energy gradient nac soc')[0] == \
        'caspt2_properties'
    assert resolve_profile('casscf', 'gradient nac')[0] == \
        'casscf_properties'


def test_espf_charge_and_potential_blocks():
    charge = _config(embedding='charge',
                     point_charges=[[1.0, 0.0, 0.0, 5.0]])
    field = resolve_embedding_field(charge)
    text = build_esfp_block(field)
    assert 'External = 1 Angstrom' in text
    assert '0.0 0.0 0.0' in text

    potential = _config(
        embedding='field',
        external_potential=[[0.01, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
    text = build_esfp_block(resolve_embedding_field(potential))
    assert 'External = 0' in text
    assert '0.01' in text
    assert text.count('0.0') >= 6


def test_charge_embedding_accepts_the_framework_array():
    # ONIOM electrostatic embedding assigns the (n, 4) array straight onto
    # the config each step, after validation, so the rows are an ndarray
    # and not the List[List[float]] the schema declares. Counting them
    # with `charges or []` raises instead of building the block.
    rows = [[1.0, 0.0, 0.0, 5.0], [-1.0, 0.0, 0.0, 6.0]]
    listed = _config(embedding='charge', point_charges=rows)
    arrayed = _config(embedding='charge', point_charges=rows)
    arrayed.system.point_charges = np.array(rows)
    assert resolve_embedding_field(arrayed) == resolve_embedding_field(listed)
    assert 'External = 2 Angstrom' in build_esfp_block(
        resolve_embedding_field(arrayed))


def test_espf_blocks_pin_a_lossless_grid_step():
    # OpenMolcas stores DELTAR in ESPF.DATA with 9 decimals and a later
    # module rebuilds the grid from that value with int(range/DeltaR).
    # A step with more decimals changes the bucket count between the two
    # calls and MkGrid aborts, so every &ESPF we emit pins a value that
    # survives the round trip exactly.
    potential = _config(
        embedding='field',
        external_potential=[[0.01, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
    for text in (build_esfp_block(resolve_embedding_field(potential)),
                 build_esfp_block(resolve_embedding_field(
                     _config(embedding='charge',
                             point_charges=[[1.0, 0.0, 0.0, 5.0]])))):
        assert 'GRID\nPNT 4 ' in text
        step = float(text.split('PNT 4 ')[1].split()[0])
        assert step == float('%.9f' % step)


def test_read_charge_file_declares_its_representation(tmp_path):
    path = tmp_path / 'charge.dat'
    path.write_text('charge angstrom\n0.0 0.0 5.0 1.0\n')
    cfg = _config(embedding='charge', read_charge=str(path))
    field = resolve_embedding_field(cfg)
    assert field['mode'] == 'charge'

    # A file declares its own representation, so `embedding` is consulted
    # only when no file is given (resolve_embedding_field docstring). A
    # field document handed to an embedding='charge' config is read as a
    # field rather than refused.
    field_file = tmp_path / 'potential.dat'
    field_file.write_text('potential au\n0.0 0.0 0.0 0.0\n'
                          '0.0 0.0 0.0 0.0\n0.0 0.0 0.0 0.0\n')
    assert resolve_embedding_field(
        _config(embedding='charge', read_charge=str(field_file)))['mode'] == \
        'field'


def test_legacy_charge_file_reads_as_external_points(tmp_path):
    """A framework `.charge` file is readable as the background field.

    `read_charge: true` expands to '<project>.charge', and that file is written
    and read by the framework's own charge store: a count line, a comment line,
    then `q x y z` rows. Two rows on purpose - a reader that handled only the
    first row of a file would pass a one-charge fixture.
    """
    path = tmp_path / 't.charge'
    path.write_text('2\nbackground\n1.0 0.0 0.0 5.0\n-1.0 0.0 0.0 -5.0\n')
    rows = resolve_embedding_field(
        _config(embedding='charge', read_charge=str(path)))['rows']
    assert rows == [[1.0, 0.0, 0.0, 5.0], [-1.0, 0.0, 0.0, -5.0]]
    # The self-describing spelling of the same two points answers the same rows.
    other = tmp_path / 't.dat'
    other.write_text('charge angstrom\n0.0 0.0 5.0 1.0\n0.0 0.0 -5.0 -1.0\n')
    assert resolve_embedding_field(
        _config(embedding='charge', read_charge=str(other)))['rows'] == rows


def test_legacy_charge_file_count_must_match_rows(tmp_path):
    path = tmp_path / 'short.charge'
    path.write_text('3\nbackground\n1.0 0.0 0.0 5.0\n-1.0 0.0 0.0 -5.0\n')
    with pytest.raises(ValueError, match='declares 3 charges'):
        resolve_embedding_field(_config(embedding='charge',
                                         read_charge=str(path)))


def test_read_charge_true_expands_to_the_project_name():
    """The switch names the file the framework writes, not a search path."""
    for spelling in (True, 'true', 'legacy'):
        cfg = _config(embedding='charge', read_charge=spelling, project='proj')
        assert cfg.read_charge == 'proj.charge'


def test_espf_nac_uses_two_passes():
    cfg = _config(embedding='charge',
                  jobtype='energy gradient nac',
                  target_nac_pair=[[0, 2]],
                  point_charges=[[1.0, 0.0, 0.0, 5.0]])
    workflow = compile_workflow(cfg.profile, config=cfg)
    assert workflow.two_pass
    assert len(workflow.pass_plan) == 2
    assert workflow.pass_plan[0]['espf'] is True
    assert workflow.pass_plan[1]['espf'] is False
    text = build_input(cfg, workflow)
    assert text.count('&GATEWAY') == 2
    assert text.count('&ESPF') == 1


def test_caspt2_energy_parser_uses_variant_block(tmp_path):
    log = tmp_path / 'x.log'
    log.write_text(
        '  Total XMS-CASPT2 energies:\n'
        '::    XMS-CASPT2 Root  1     Total energy:    -1.10000000\n'
        '::    XMS-CASPT2 Root  2     Total energy:    -1.00000000\n'
        '::    XMS-CASPT2 Root  3     Total energy:    -0.90000000\n'
        ' Happy landing\n')
    cfg = _config(nstate=3, jobtype='energy')
    result = parse_log(str(log), cfg)
    assert result['energy'] == pytest.approx([-1.1, -1.0, -0.9])
    assert result['energy_source'] == 'caspt2_xms'


def test_caspt2_nocsf_nac_parser_prefers_total(tmp_path):
    log = tmp_path / 'n.log'
    block = (' *              Total derivative coupling              *\n'
             ' H1 0.1 0.0 0.0\n H2 0.0 0.2 0.0\n O3 0.0 0.0 0.3\n')
    log.write_text(block + ' Happy landing\n')
    cfg = _config(method='casscf', profile='casscf_properties',
                  nstate=1, state_spin=[0], ci_roots=[1],
                  jobtype='energy nac', target_nac_pair=[[0, 0]])
    result = parse_log(str(log), cfg)
    assert result['nac_source'] == 'total_derivative'
    assert result['nac_csf_included'] is False

def test_no_coupling_block_is_no_rows_not_a_zero_coupling(tmp_path):
    # A run whose input never asked `&ALASKA` for a derivative coupling
    # prints none, and the reader used to answer that with one row of zeros.
    # Zero is not "absent": `meci` read it as a measured coupling of zero,
    # dropped the direction below its floor, and reported a one-dimensional
    # branching plane as if the engine had been asked and had said no. The
    # two logs below are the same case with and without a coupling table, so
    # the empty result is read against a non-empty one rather than asserted
    # on its own.
    cfg = _config(method='casscf', profile='casscf_properties',
                  nstate=1, state_spin=[0], ci_roots=[1],
                  jobtype='energy gradient')
    quiet = tmp_path / 'quiet.log'
    quiet.write_text('--- Start Module: alaska at Sun Nov 01 00:00:00 2026 ---\n'
                     ' Molecular gradients\n'
                     ' Happy landing\n')
    loud = tmp_path / 'loud.log'
    # The reader takes the table eight lines under the banner line, which is
    # where the engine's own box and column headings leave it.
    loud.write_text(' *              Total derivative coupling              *\n'
                    + '\n'.join(' heading %d' % k for k in range(7)) + '\n'
                    + ' O1 0.1 0.0 0.0\n H2 0.0 0.2 0.0\n H3 0.0 0.0 0.3\n'
                    ' Happy landing\n')
    quiet_result = parse_log(str(quiet), cfg)
    assert np.asarray(quiet_result['nac']).size == 0
    assert quiet_result['nac_source'] == ''
    loud_result = parse_log(str(loud), cfg)
    rows = np.asarray(loud_result['nac'])
    assert rows.shape == (1, 3, 3)
    assert rows[0][0].tolist() == pytest.approx([0.1, 0.0, 0.0])
    assert loud_result['nac_source'] == 'total_derivative'


def test_a_coupling_is_requested_on_a_table_that_has_both_states():
    # What decides whether the engine is asked for a derivative coupling at
    # all. The skill refuses to build a deck that names `nac` on a one-state
    # table, and prints the coupling block once the table covers the pair -
    # so a run whose own `ci`/`spin` declare three states cannot get a
    # coupling while its state table is handed over empty. That is how
    # `multiscale_qmmm_meci` came to search a one-dimensional plane: the
    # request was never made, and the reader answered the missing table with
    # a row of zeros. The one-state arm is refused by name at validation -
    # which the per-step request channel deliberately does not run, so the
    # plan has to hand the leg the table it derives.
    def deck(nstate, state_spin, pair):
        cfg = _config(method='casscf', profile='casscf_nac',
                      jobtype='energy gradient nac', nstate=nstate,
                      state_spin=state_spin, ci_roots=[nstate],
                      ci_spin=[0], target_states=[0, 1][:nstate],
                      target_nac_pair=pair)
        return build_input(cfg, compile_workflow(cfg.profile, config=cfg))

    with pytest.raises(ValueError, match='target_nac_pair'):
        deck(1, [], [[0, 1]])
    assert 'NAC=1 2' in deck(3, [0, 0, 0], [[0, 1]])


def test_espf_readback_gate_uses_the_skills_own_vocabulary():
    # The backend reads post-ESPF 0-order charges only for the embedding
    # words listed in `_ESPF_CHARGE_EMBEDDINGS`. That list went stale once
    # already: it named 'potential', which this schema refuses, and omitted
    # 'field', which is what a QMMM run actually sends - so the reader never
    # fired on the QMMM path and every shipped QMMM example had to hand the
    # framework a hand-written molecule.q_qm instead of a computed charge.
    # Checked against the model that decides the vocabulary rather than
    # against a second list, so the next word added here has to be one the
    # config would actually accept.
    import pytest
    from pydantic import ValidationError

    from ..backend import _ESPF_CHARGE_EMBEDDINGS
    from ..schema.config import System

    for word in _ESPF_CHARGE_EMBEDDINGS:
        assert System(embedding=word).embedding == word
        assert word != 'none', ('a non-embedding must not trigger the '
                                'charge readback')
    with pytest.raises(ValidationError):
        System(embedding='potential')
    assert 'potential' not in _ESPF_CHARGE_EMBEDDINGS
    # The witness that this is not an empty-Set pass: the word the framework
    # puts on a QMMM task has to be one of the gated ones.
    assert 'field' in _ESPF_CHARGE_EMBEDDINGS


def test_the_coupling_rows_are_addressed_by_the_requested_pairs(tmp_path):
    # `&ALASKA` prints one coupling table per requested pair, in the deck's own
    # order, and the table names no states: row k *is* the k-th pair. A log
    # whose block count does not match the request cannot be mapped onto the
    # pairs, and the reader used to hand back row k for whatever pair happened
    # to be printed k-th - a mis-pairing no caller can see.
    def block(seed):
        return (' *              Total derivative coupling              *\n'
                + '\n'.join(' heading %d' % k for k in range(7)) + '\n'
                + ' O1 %.1f 0.0 0.0\n H2 0.0 %.1f 0.0\n H3 0.0 0.0 %.1f\n'
                % (seed, seed, seed))

    cfg = _config(method='casscf', profile='casscf_properties',
                  nstate=3, state_spin=[0, 0, 0], ci_roots=[3],
                  jobtype='energy nac', target_nac_pair=[[0, 1], [1, 2]])

    matched = tmp_path / 'matched.log'
    matched.write_text(block(0.1) + block(0.2) + ' Happy landing\n')
    rows = np.asarray(parse_log(str(matched), cfg)['nac'])
    assert rows.shape == (2, 3, 3)
    assert rows[1][1].tolist() == pytest.approx([0.0, 0.2, 0.0])

    extra = tmp_path / 'extra.log'
    extra.write_text(block(0.1) + block(0.2) + block(0.3) + ' Happy landing\n')
    with pytest.raises(RuntimeError,
                       match='3 derivative-coupling block.*2\\s+requested'):
        parse_log(str(extra), cfg)


def test_no_declared_transition_pair_leaves_the_channel_empty(tmp_path):
    """An undeclared transition channel returns no rows, not a zero row.

    The reader used to hand back `np.zeros(1)` / `np.zeros((1, 3))` when
    `transition_pair` was empty, so a run with nothing to report printed a
    transition dipole of zero and an oscillator strength of zero - the two
    numbers an absorption spectrum is supposed to have. Restoring either
    default makes this arm red.
    """
    cfg = _config(method='casscf', profile='casscf_properties',
                  jobtype='energy', nstate=1, state_spin=[0],
                  ci_roots=[1], target_states=[0])
    assert [list(pair) for pair in cfg.transition_pair] == []
    log = tmp_path / 'single.log'
    log.write_text(' Happy landing\n')
    results = parse_log(str(log), cfg)
    assert np.asarray(results['osc_strength']).shape == (0,)
    assert np.asarray(results['transition_dipole']).shape == (0, 3)


def test_a_deck_grouped_by_spin_still_returns_the_declared_row_order(
        tmp_path):
    """Grouping decides where a block sits in the deck, not which pair it is.

    `&ALASKA` is written once per spin group, so three pairs spread over two
    manifolds print in the order singlet-singlet, singlet-singlet,
    triplet-triplet whatever the input declared first. The reader has to put
    them back: the framework indexes `results['nac']` by the declared pair
    list (`workflow/integrator/fssh.py`), and a run that declared the triplet
    pair first would otherwise have that pair's row filled with a singlet
    coupling - the same number of rows, all of them wrong, nothing raised.

    Witness: delete the re-permutation in `log_parser.py` (or restore the
    `sorted()` this skill used to apply to `target_nac_pair`) and `rows[0]`
    comes back carrying the 0.1 block instead of the 0.3 one.
    """
    def block(seed):
        return (' *              Total derivative coupling              *\n'
                + '\n'.join(' heading %d' % k for k in range(7)) + '\n'
                + ' O1 %.1f 0.0 0.0\n H2 0.0 %.1f 0.0\n H3 0.0 0.0 %.1f\n'
                % (seed, seed, seed))

    declared = [[3, 4], [0, 1], [1, 2]]
    cfg = _config(method='casscf', profile='casscf_properties',
                  nstate=5, state_spin=[0, 0, 0, 2, 2], ci_roots=[3, 2],
                  ci_spin=[0, 1], target_states=[0, 1, 2, 3, 4],
                  jobtype='energy nac',
                  target_nac_pair=[list(pair) for pair in declared])

    # The declaration survives the skill untouched, and the deck's own order
    # is the grouped one - the two are deliberately different.
    assert [list(pair) for pair in cfg.target_nac_pair] == declared
    deck = build_input(cfg, compile_workflow(cfg.profile, config=cfg),
                       str(tmp_path))
    assert nac_deck_order(cfg.target_nac_pair,
                          list(cfg.state_spin)) == [[0, 1], [1, 2], [3, 4]]
    assert deck.count('NAC=') == 3

    log = tmp_path / 'grouped.log'
    log.write_text(block(0.1) + block(0.2) + block(0.3) + ' Happy landing\n')
    rows = np.asarray(parse_log(str(log), cfg)['nac'])

    assert rows.shape == (3, 3, 3)
    # Row 0 is the declared triplet pair, whose block the deck printed last.
    assert rows[0][0].tolist() == pytest.approx([0.3, 0.0, 0.0])
    assert rows[1][0].tolist() == pytest.approx([0.1, 0.0, 0.0])
    assert rows[2][0].tolist() == pytest.approx([0.2, 0.0, 0.0])


def test_a_declared_coupling_reaches_the_step_request_past_the_unset_marker():
    # `target_nac_pair` arrives from the submodel as the "not filled" marker
    # `[[]]`, which is truthy. Reading "did the step name a pair?" as a truth
    # test therefore saw a declaration where there was none, left the request
    # empty, and the CASPT2 contract then refused the case outright
    # (`CASPT2 NAC requires at least one same-spin pair`) - which is how all six
    # shipped CASPT2 nac/prop cases failed the loading chain when #448 made the
    # pair maps declaration-driven. The test asks for the coupling in the words
    # a framework run uses and checks the request the deck will be built from.
    cfg = _config(method='casscf', profile='casscf_properties',
                  jobtype='energy nac', nstate=3, state_spin=[0, 0, 0],
                  ci_roots=[3], coupling=[[0, 1], [1, 2]])
    assert [list(pair) for pair in cfg.target_nac_pair] == [[0, 1], [1, 2]]
    assert [list(pair) for pair in cfg.target_soc_pair] == []
    assert [list(pair) for pair in cfg.nac_pair_map] == [[0, 1], [1, 2]]

    # The other arm: a step that named pairs of its own keeps them, so the
    # declaration is a default and not an overwrite.
    chosen = _config(method='casscf', profile='casscf_properties',
                     jobtype='energy nac', nstate=3, state_spin=[0, 0, 0],
                     ci_roots=[3], coupling=[[0, 1], [1, 2]],
                     target_nac_pair=[[1, 2]])
    assert [list(pair) for pair in chosen.target_nac_pair] == [[1, 2]]
