"""
Unit tests for the LAMMPS force-field compressor (ff_model).

Verifies that the JSON model keeps only the referenced force-field
types, renumbers them onto 1..N, resolves duplicate coefficient keys
last-write-wins with an audit trail, owns the constraint decision, and
refuses (rather than guesses) every template shape it cannot prove
equivalent. Also covers the cache invalidation and the sibling-of-calcdir
placement that keeps the model out of reach of keep_tmp.
These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_ff_model.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import json
import os
import types

import pytest

from ..service.builder.ff_model import (FFModelUnsupported, build_ff_model,
                                        is_subsystem, load_cached_model,
                                        model_path, read_templates,
                                        render_data, render_init,
                                        render_settings, save_model)

# A water box whose force-field template declares far more types than
# the system references, mirroring the shipped h2o fixture: atom types
# 31 (H) and 76 (O) out of 111, bond type 952, angle type 5355, with
# generated placeholder coefficients shadowed by the re-fitted block at
# the end of the file.
DATA = """LAMMPS Description

6  atoms
4  bonds
2  angles
0  dihedrals
0  impropers

111  atom types
952  bond types
5355  angle types
1004  dihedral types
38  improper types

0.0 10.00000000 xlo xhi
0.0 10.00000000 ylo yhi
0.0 10.00000000 zlo zhi

Masses

31 1.008  # hw
76 15.9994  # ow

Atoms

1 1 76 -0.830 1.00000000 1.00000000 1.00000000
2 1 31 0.415 2.00000000 1.00000000 1.00000000
3 1 31 0.415 1.00000000 2.00000000 1.00000000
4 2 76 -0.830 5.00000000 5.00000000 5.00000000
5 2 31 0.415 6.00000000 5.00000000 5.00000000
6 2 31 0.415 5.00000000 6.00000000 5.00000000

Bonds

1 952 1 2
2 952 1 3
3 952 4 5
4 952 4 6

Angles

1 5355 2 1 3
2 5355 5 4 6
"""

CROSS_PAIR = 'pair_coeff 31 76 lj/charmm/coul/long 0.00 1.7753\n'

SETTINGS = """# generated library defaults
pair_coeff 31 31 lj/charmm/coul/long 0.0 1.0 # (default parameters)
pair_coeff 76 76 lj/charmm/coul/long 0.0 1.0 # (default parameters)
pair_coeff 26 26 lj/charmm/coul/long 0.0208 2.600176998764394
bond_coeff   952   harmonic      450.0  0.9572
bond_coeff   111   harmonic      300.0  1.5000
angle_coeff  5355 harmonic      55.0   104.52

#   For the 2004 Price & Brooks version of TIP3P use:
pair_coeff  76  76  lj/charmm/coul/long 0.102 3.188
pair_coeff  31  31  lj/charmm/coul/long 0.00 0.4000
pair_coeff 31 76 lj/charmm/coul/long 0.00 1.7753

# Constrain tip3p water
group tip3p type  76  31
fix fShakeTIP3P tip3p shake 0.0001 10 100 b 952 a 5355

# Buckham coefficients of the library types no atom of this box uses,
# exactly as the shipped template lists them.
pair_coeff 26 26 buck/coul/long 21641046.65 0.205 145.31
pair_coeff 101 101 buck/coul/long 371321.8 0.099492 25480.4
"""

# The five hybrid declarations of the shipped template. The 60-atom box
# uses only lj/charmm/coul/long (its buck rows belong to types 101/106)
# and no dihedral or improper at all, so those sub-styles are starved by
# the type census and must leave the declaration: LAMMPS aborts on a
# hybrid wrapper listing a sub-style no coefficient uses.
INIT = """units real
atom_style full
bond_style hybrid harmonic
angle_style hybrid harmonic
dihedral_style hybrid fourier
improper_style hybrid cvff
pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0 buck/coul/long 10.0
kspace_style pppm 1e-4
pair_modify mix arithmetic
special_bonds amber
"""


@pytest.fixture
def launch_dir(tmp_path, monkeypatch):
    """Launch directory carrying the three user templates."""
    (tmp_path / 'h2o.data').write_text(DATA)
    (tmp_path / 'h2o.in.init').write_text(INIT)
    (tmp_path / 'h2o.in.settings').write_text(SETTINGS)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _config(**kwargs):
    """Minimal stand-in for the skill config the model builder reads."""
    values = {'lammps_project': 'h2o', 'shake_fix_id': 'fShakeTIP3P',
              'lammps_shake': 0, 'shake_keywords': '', 'stress': False,
              # Mirrors the accuracy INIT declares, so a fresh model and a
              # reloaded one are asked to render the same deck text.
              'ewald_accuracy': '1e-4',
              'natom': 6, 'coordinates': BACKEND_COORDS,
              'pbc': [0, 0, 0], 'velocities': None}
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def _model(profile='ff_gradient', **kwargs):
    return build_ff_model(_config(**kwargs), profile)


class TestCompression:
    """Only the referenced types survive, renumbered onto 1..N."""

    def test_counts_collapse_to_the_used_types(self, launch_dir):
        model = _model()
        assert model['counts'] == {
            'natom': 6, 'nbond': 4, 'nangle': 2, 'ndihedral': 0,
            'nimproper': 0, 'natom_types': 2, 'nbond_types': 1,
            'nangle_types': 1, 'ndihedral_types': 0, 'nimproper_types': 0}
        assert model['used_types']['atom'] == [31, 76]
        assert model['type_map']['atom'] == {'31': 1, '76': 2}

    def test_settings_keep_only_the_referenced_coefficients(self, launch_dir):
        text = render_settings(_model())
        assert 'pair_coeff 1 1 lj/charmm/coul/long 0.00 0.4000' in text
        assert 'pair_coeff 2 2 lj/charmm/coul/long 0.102 3.188' in text
        assert 'pair_coeff 1 2 lj/charmm/coul/long 0.00 1.7753' in text
        assert 'bond_coeff 1 harmonic 450.0 0.9572' in text
        assert 'angle_coeff 1 harmonic 55.0 104.52' in text
        # Unused library entries never reach LAMMPS.
        assert '26' not in text.replace('# src', '')
        assert text.count('\n') <= 7

    def test_duplicate_keys_resolve_last_write_wins(self, launch_dir):
        model = _model()
        pairs = {tuple(int(t) for t in row[:2]): row[2:]
                 for row in model['coeffs']['pair']}
        # The generated placeholder (0.0 1.0) is shadowed by the
        # re-fitted block, exactly as the verbatim template behaves.
        assert pairs[(2, 2)][1:] == ['0.102', '3.188']
        assert pairs[(1, 1)][1:] == ['0.00', '0.4000']

    def test_audit_records_every_superseded_line(self, launch_dir):
        audit = _model()['audit']
        dropped = {(tuple(d['key']), d['superseded_line'], d['kept_line'])
                   for d in audit['dropped_duplicates']}
        assert dropped == {
            ((31, 31, 'lj/charmm/coul/long'), 2, 11),
            ((76, 76, 'lj/charmm/coul/long'), 3, 10)}

    def test_two_substyles_of_one_pair_are_not_duplicates(self, launch_dir):
        # A hybrid table legitimately carries buck/coul/long beside
        # lj/charmm/coul/long for the same type pair (L4 and L20 of the
        # template): those are two parameter sets, not one shadowing the
        # other, so neither may be dropped as superseded.
        audit = _model()['audit']
        keys = {tuple(d['key']) for d in audit['dropped_duplicates']}
        assert not any(key[0] == 26 for key in keys)
        assert audit['dropped_unused_coeffs'] >= 1

    def test_rendered_provenance_points_at_the_kept_line(self, launch_dir):
        text = render_settings(_model())
        assert 'pair_coeff 2 2 lj/charmm/coul/long 0.102 3.188   # src L10' \
            in text

    def test_data_renumbers_only_the_type_columns(self, launch_dir):
        rendered = render_data(_model()).splitlines()
        template = DATA.splitlines()

        def rows(lines, section):
            out, keep = [], False
            for line in lines:
                if line.split('#')[0].strip() == section:
                    keep = True
                    continue
                if keep and line.split()[:1] and not line.split()[0].isdigit():
                    break
                if keep and line.strip():
                    out.append(line.split())
            return out

        for section, column in (('Atoms', 2), ('Bonds', 1), ('Angles', 1)):
            old, new = rows(template, section), rows(rendered, section)
            assert len(old) == len(new)
            for o, n in zip(old, new):
                assert [t for i, t in enumerate(o) if i != column] == \
                       [t for i, t in enumerate(n) if i != column]
        assert '2  atom types' in rendered
        assert '111  atom types' not in rendered

    def test_masses_follow_the_renumbering(self, launch_dir):
        rendered = render_data(_model())
        assert '1 1.008 # hw' in rendered
        assert '2 15.9994 # ow' in rendered


class TestSelection:
    """A row selection cuts the template down to one task's subsystem.

    This is the channel a multi-center run's low-layer correction term
    `L(c)` needs: the user ships one system-level `{project}.data` and each
    LAMMPS task is handed the rows of its own region (docs
    /skills/LAMMPS_SKILL_DEVELOPMENT.md).
    """

    @staticmethod
    def _rows(model, section):
        out, keep = [], False
        for line in render_data(model).splitlines():
            head = line.split('#')[0].strip()
            if head == section:
                keep = True
                continue
            if keep and not head:
                continue
            if keep and not head.split()[0].isdigit():
                break
            if keep:
                out.append(head.split())
        return out

    def test_no_selection_leaves_the_model_unmarked(self, launch_dir):
        model = _model()
        assert 'selection' not in model
        assert not is_subsystem(model)

    def test_a_selection_that_asks_every_row_moves_no_byte(self, launch_dir):
        # The negative control of the whole feature, and the reason the 24
        # golden input digests of test_input_builder_golden stay put: cutting
        # the template to every row in template order is the identity.
        assert (render_data(_model(atom_selection=[1, 2, 3, 4, 5, 6]))
                == render_data(_model()))

    def test_one_water_of_two_survives_the_cut(self, launch_dir):
        model = _model(atom_selection=[1, 2, 3])
        assert is_subsystem(model)
        assert model['selection'] == [1, 2, 3]
        assert model['counts']['natom'] == 3
        assert model['counts']['nbond'] == 2
        assert model['counts']['nangle'] == 1
        assert model['audit']['torn_entities'] == [
            {'section': 'Bonds', 'row': '3 952 4 5', 'outside': ['4', '5']},
            {'section': 'Bonds', 'row': '4 952 4 6', 'outside': ['4', '6']},
            {'section': 'Angles', 'row': '2 5355 5 4 6',
             'outside': ['5', '4', '6']}]
        # Renumbered onto 1..natom, which is what the per-step writer's
        # `atom id - 1 == coordinate row` mapping needs.
        assert [row[0] for row in self._rows(model, 'Atoms')] == ['1', '2',
                                                                  '3']
        assert [row[2:] for row in self._rows(model, 'Bonds')] == \
               [['1', '2'], ['1', '3']]
        assert [row[2:] for row in self._rows(model, 'Angles')] == \
               [['2', '1', '3']]

    def test_the_new_numbering_follows_the_handed_order(self, launch_dir):
        # The skill's rule is positional: new id = position in the selection.
        # The framework canonicalises a center to ascending before it gets
        # here (schema/adapter.py::_parse_center_indices), so this deliberately
        # non-ascending selection is the case that proves the skill itself adds
        # no sorting of its own - a re-sorted copy would put the center's atoms
        # on each other's coordinates.
        model = _model(atom_selection=[3, 1, 2])
        rows = self._rows(model, 'Atoms')
        assert [row[0] for row in rows] == ['1', '2', '3']
        assert rows[0][4:] == ['1.00000000', '2.00000000', '1.00000000']
        assert rows[1][4:] == ['1.00000000', '1.00000000', '1.00000000']
        assert [row[2:] for row in self._rows(model, 'Bonds')] == \
               [['2', '3'], ['2', '1']]
        assert self._rows(model, 'Atoms') != self._rows(
            _model(atom_selection=[1, 2, 3]), 'Atoms')

    def test_an_entity_broken_by_the_cut_is_recorded_not_lost(self,
                                                              launch_dir):
        # Crossing interactions belong to L(real); dropping them from the
        # subsystem is the point, but it has to be visible in the audit.
        model = _model(atom_selection=[1, 2, 4])
        assert [row[2:] for row in self._rows(model, 'Bonds')] == [['1', '2']]
        assert self._rows(model, 'Angles') == []
        torn = {(entry['section'], entry['row'])
                for entry in model['audit']['torn_entities']}
        assert ('Bonds', '2 952 1 3') in torn
        assert ('Angles', '1 5355 2 1 3') in torn
        assert len(model['audit']['torn_entities']) == 5

    def test_a_row_the_template_does_not_have_is_refused(self, launch_dir):
        with pytest.raises(FFModelUnsupported, match='has no row'):
            _model(atom_selection=[1, 2, 7])
        with pytest.raises(FFModelUnsupported, match='selected twice'):
            _model(atom_selection=[1, 1, 2])

    def test_the_selection_is_part_of_the_cache_identity(self, launch_dir,
                                                         tmp_path):
        first = _config(atom_selection=[1, 2, 3])
        second = _config(atom_selection=[4, 5, 6])
        reversed_first = _config(atom_selection=[3, 2, 1])
        digests = {build_ff_model(config, 'ff_gradient')['config_hash']
                   for config in (first, second, reversed_first, _config())}
        assert len(digests) == 4

        path = model_path('h2o', str(tmp_path))
        save_model(path, build_ff_model(first, 'ff_gradient'))
        sources = read_templates('h2o')
        assert load_cached_model(path, sources, first, 'ff_gradient')
        # Two centers of the same size and the same types are the case the
        # counts cannot separate: only the selection says which one this is.
        assert (build_ff_model(second, 'ff_gradient')['counts']
                == build_ff_model(first, 'ff_gradient')['counts'])
        assert load_cached_model(path, sources, second,
                                 'ff_gradient') is None

    def test_the_reciprocal_accuracy_is_part_of_the_cache_identity(
            self, launch_dir, tmp_path):
        # A cached model stands for the deck it renders, and that deck now
        # carries an accuracy token, so a scratch directory shared by a 1e-4
        # run and a converged 1e-8 run must not answer one with the other.
        loose = _config(ewald_accuracy='1e-4')
        tight = _config(ewald_accuracy='1e-8')
        assert (build_ff_model(loose, 'ff_gradient')['config_hash']
                != build_ff_model(tight, 'ff_gradient')['config_hash'])

        path = model_path('h2o', str(tmp_path))
        save_model(path, build_ff_model(loose, 'ff_gradient'))
        sources = read_templates('h2o')
        assert load_cached_model(path, sources, loose, 'ff_gradient')
        assert load_cached_model(path, sources, tight,
                                 'ff_gradient') is None


class TestConstraintOwnership:
    """The model is the single source of the constraint keywords."""

    def test_framework_integration_emits_no_constraint(self, launch_dir):
        model = _model('ff_gradient')
        assert model['constraints']['enabled'] is False
        assert model['constraints']['engine'] == 'framework'
        text = render_settings(model)
        assert 'fix' not in text and 'group' not in text
        assert model['audit']['dropped_constraints'][0]['line'] == 16

    def test_lammps_integration_keeps_a_renumbered_fix(self, launch_dir):
        model = _model('ff_nvt', lammps_shake=1)
        assert model['constraints']['enabled'] is True
        assert model['constraints']['engine'] == 'lammps'
        lines = model['constraints']['lines']
        assert lines[0] == 'group tip3p type 2 1'
        assert lines[1] == 'fix fShakeTIP3P tip3p shake 0.0001 10 100 b 1 a 1'
        assert 'group tip3p type 2 1' in render_settings(model)

    def test_engine_profile_with_shake_off_stays_unconstrained(
            self, launch_dir):
        model = _model('ff_nvt', lammps_shake=0)
        assert model['constraints']['enabled'] is False
        assert 'fix' not in render_settings(model)


class TestStyleDeclarations:
    """The model owns the '*_style hybrid' declarations of the init file.

    LAMMPS refuses a hybrid wrapper listing a sub-style no coefficient
    uses ("Pair hybrid sub-style is not used"), which the trimmed
    coefficient table would otherwise leave behind for three of the five
    families of the shipped template.
    """

    def rewrite(self, launch_dir, init=None, settings=None):
        (launch_dir / 'h2o.in.init').write_text(init if init is not None
                                                else INIT)
        if settings is not None:
            (launch_dir / 'h2o.in.settings').write_text(settings)

    def test_starved_substyle_leaves_the_declaration(self, launch_dir):
        text = render_init(_model())
        assert ('pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0\n'
                in text)
        assert 'buck/coul/long' not in text

    def test_family_with_nothing_to_evaluate_is_switched_off(self,
                                                             launch_dir):
        text = render_init(_model())
        # The box carries no dihedral and no improper: the wrapper's only
        # sub-style can never be used, so the family is switched off.
        assert 'dihedral_style none' in text
        assert 'improper_style none' in text

    def test_surviving_substyle_keeps_its_exact_arguments(self,
                                                          launch_dir):
        # Getting the name/argument boundary wrong would silently change
        # the cutoffs, so the kept segment is compared in full.
        line = [row for row in render_init(_model()).splitlines()
                if row.startswith('pair_style')][0]
        assert line == 'pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0'

    def test_untouched_declaration_is_byte_identical(self, launch_dir):
        init = INIT.replace('dihedral_style hybrid fourier\n', '') \
                   .replace('improper_style hybrid cvff\n', '') \
                   .replace(' buck/coul/long 10.0', '')
        self.rewrite(launch_dir, init=init)
        assert render_init(_model()) == init

    def test_audit_lists_every_pruned_substyle(self, launch_dir):
        audit = {entry['family']: entry
                 for entry in _model()['audit']['pruned_substyles']}
        assert set(audit) == {'pair', 'dihedral', 'improper'}
        assert audit['pair']['dropped'] == ['buck/coul/long']
        assert audit['pair']['kept'] == ['lj/charmm/coul/long']
        assert audit['dihedral']['kept'] == ['none']

    def test_hybrid_suffix_survives_the_rewrite(self, launch_dir):
        # hybrid/overlay differs from hybrid in both grammar and physics;
        # the model may not choose between them.
        self.rewrite(launch_dir, init=INIT.replace(
            'pair_style hybrid lj', 'pair_style hybrid/overlay lj'))
        line = [row for row in render_init(_model()).splitlines()
                if row.startswith('pair_style')][0]
        assert line.startswith('pair_style hybrid/overlay ')

    def test_trailing_comment_survives_the_rewrite(self, launch_dir):
        self.rewrite(launch_dir, init=INIT.replace(
            'buck/coul/long 10.0', 'buck/coul/long 10.0 # library extra'))
        line = [row for row in render_init(_model()).splitlines()
                if row.startswith('pair_style')][0]
        assert line == ('pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0'
                        ' # library extra')

    def test_non_hybrid_declaration_is_never_rewritten(self, launch_dir):
        self.rewrite(launch_dir, init=INIT.replace(
            'pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0 '
            'buck/coul/long 10.0', 'pair_style lj/charmm/coul/long 9.0'))
        assert 'pair_style lj/charmm/coul/long 9.0\n' in render_init(_model())

    def test_refuses_a_family_losing_its_last_used_substyle(self,
                                                            launch_dir):
        # Angles are there to evaluate, so answering their vanished
        # wrapper with 'none' would drop real interactions.
        settings = '\n'.join(row for row in SETTINGS.splitlines()
                             if not row.startswith('angle_coeff'))
        self.rewrite(launch_dir, settings=settings)
        with pytest.raises(FFModelUnsupported,
                           match='angle_style hybrid keeps no sub-style'):
            _model()

    def test_refuses_a_keyword_argument_it_cannot_place(self, launch_dir):
        # 'linear 100' belongs to table; without a coefficient row naming
        # it the model cannot tell an argument from a sub-style.
        self.rewrite(launch_dir, init=INIT.replace(
            'buck/coul/long 10.0', 'buck/coul/long 10.0 table linear 100'))
        with pytest.raises(FFModelUnsupported, match='cannot be told from an'):
            _model()


class TestRefusal:
    """Anything unprovable falls back to the verbatim templates."""

    def _rewrite(self, launch_dir, name, text):
        (launch_dir / name).write_text(text)

    def test_wildcard_pair_coeff(self, launch_dir):
        self._rewrite(launch_dir, 'h2o.in.settings',
                      SETTINGS + 'pair_coeff * * lj/charmm/coul/long 0 1\n')
        with pytest.raises(FFModelUnsupported, match='wildcard'):
            _model()

    def test_unknown_settings_command(self, launch_dir):
        self._rewrite(launch_dir, 'h2o.in.settings',
                      SETTINGS + 'region box block 0 10 0 10 0 10\n')
        with pytest.raises(FFModelUnsupported, match='outside the model'):
            _model()

    def test_missing_coefficient_for_a_used_type(self, launch_dir):
        self._rewrite(launch_dir, 'h2o.in.settings',
                      SETTINGS.replace('bond_coeff   952   harmonic'
                                       '      450.0  0.9572\n', ''))
        with pytest.raises(FFModelUnsupported, match='bond'):
            _model()

    def test_a_cross_pair_derived_by_mixing_is_not_a_hole(self, launch_dir):
        """Same-type rows plus `pair_modify mix` cover the cross pair.

        The ethanol template is why this exists: it lists 98 `pair_coeff` rows
        and not one cross combination, because LAMMPS derives them from the two
        diagonal rows under `pair_modify mix arithmetic`. The coverage check
        read that as a missing coefficient, so ethanol had no compressed model
        and the whole project fell to one subprocess per step. The fixture here
        is the same shape -- h2o with its explicit `31 76` row removed.
        """
        self._rewrite(launch_dir, 'h2o.in.settings', SETTINGS.replace(CROSS_PAIR, ''))
        model = _model()
        # Every kept pair row is a same-type row, so the cross term the box
        # needs is the one LAMMPS derives -- and both used types still have
        # their diagonal, which is what makes the derivation defined.
        rows = [row for row in model['coeffs']['pair'] if len(row) > 1]
        assert rows, 'no pair coefficients survived'
        assert all(row[0] == row[1] for row in rows)
        assert {row[0] for row in rows} == {'1', '2'}

    def test_without_a_mixing_rule_the_cross_pair_is_required(self, launch_dir):
        """The relaxation is bound to the statement, not to optimism.

        Take the mixing line away and the same template is refused naming the
        pair, because nothing in the file derives it any more.
        """
        self._rewrite(launch_dir, 'h2o.in.settings', SETTINGS.replace(CROSS_PAIR, ''))
        self._rewrite(launch_dir, 'h2o.in.init',
                      INIT.replace('pair_modify mix arithmetic\n', ''))
        with pytest.raises(FFModelUnsupported, match='no pair_coeff entry'):
            _model()

    def test_data_without_declared_type_counts(self, launch_dir):
        self._rewrite(launch_dir, 'h2o.data',
                      DATA.replace('111  atom types\n', ''))
        with pytest.raises(FFModelUnsupported, match='atom types'):
            _model()

    def test_velocities_section_is_not_owned_by_the_model(self, launch_dir):
        self._rewrite(launch_dir, 'h2o.data',
                      DATA + '\nVelocities\n\n1 0.0 0.0 0.0\n')
        with pytest.raises(FFModelUnsupported, match='Velocities'):
            _model()

    def test_dihedral_entities_compress_and_renumber(self, launch_dir):
        """A box that carries dihedrals is compressed, not refused.

        This path exists for the ethanol fixture: 12 dihedral rows over sparse
        library type ids, with `dihedral_style hybrid fourier` live. The
        refusal it replaces treated dihedral renumbering as out of scope, and
        that left ethanol with no compressed model, so the whole project fell
        back to one subprocess per step while h2o stayed resident -- h2o passed
        only because it has no dihedral entity at all.
        """
        body = DATA.replace('0  dihedrals', '2  dihedrals')
        body += '\nDihedrals\n\n1 777 1 2 3 4\n2 777 4 5 6 1\n'
        self._rewrite(launch_dir, 'h2o.data', body)
        self._rewrite(launch_dir, 'h2o.in.settings',
                      SETTINGS + 'dihedral_coeff 777 fourier 3.0 2.0 1.0 1\n')

        model = _model()
        assert model['counts']['ndihedral'] == 2
        assert model['counts']['ndihedral_types'] == 1
        assert model['type_map']['dihedral'] == {'777': 1}
        # Only the type column moves; the atom ids of the row stay as written.
        assert '1 1 1 2 3 4' in model['sections']['Dihedrals']
        assert '2 1 4 5 6 1' in model['sections']['Dihedrals']
        # A family with rows to evaluate is never switched off.
        assert 'dihedral_style none' not in render_init(model)
        row = model['coeffs']['dihedral'][0]
        assert row[0] == '1' and row[1:] == ['fourier', '3.0', '2.0', '1.0', '1']
        assert 'dihedral_coeff 1 fourier 3.0 2.0 1.0 1' in render_settings(model)
        assert 'Dihedrals' in render_data(model)

    def test_improper_entities_compress_too(self, launch_dir):
        body = DATA.replace('0  impropers', '1  impropers')
        body += '\nImpropers\n\n1 901 1 2 3 4\n'
        self._rewrite(launch_dir, 'h2o.data', body)
        self._rewrite(launch_dir, 'h2o.in.settings',
                      SETTINGS + 'improper_coeff 901 cvff 1.0 2.0 3.0 1\n')

        model = _model()
        assert model['counts']['nimproper'] == 1
        assert model['type_map']['improper'] == {'901': 1}
        assert '1 1 1 2 3 4' in model['sections']['Impropers']
        assert 'improper_style none' not in render_init(model)

    def test_missing_topology_template(self, launch_dir):
        (launch_dir / 'h2o.data').unlink()
        with pytest.raises(FFModelUnsupported, match='absent'):
            _model()

    def test_optional_templates_may_be_absent(self, launch_dir):
        (launch_dir / 'h2o.in.settings').unlink()
        with pytest.raises(FFModelUnsupported):
            _model()


class TestCache:
    """The model survives the calculation directory."""

    def test_path_is_a_sibling_of_the_calcdir(self):
        assert model_path('h2o', '/scratch') == '/scratch/h2o.ff.json'
        assert model_path('h2o') == './h2o.ff.json'

    def test_round_trip_and_stale_source(self, launch_dir, tmp_path):
        model = _model()
        path = str(tmp_path / 'h2o.ff.json')
        assert save_model(path, model)

        sources = read_templates('h2o')
        loaded = load_cached_model(path, sources, _config(), 'ff_gradient')
        assert loaded['counts'] == model['counts']

        # Editing a template invalidates the cache.
        (launch_dir / 'h2o.in.settings').write_text(SETTINGS + '\n')
        assert load_cached_model(path, read_templates('h2o'), _config(),
                                 'ff_gradient') is None

    def test_constraint_configuration_is_part_of_the_identity(
            self, launch_dir, tmp_path):
        # The constraint state depends on the profile and the shake
        # keywords, so a cache built for the framework-integrated
        # profile must not be handed to an engine-integrated one that
        # shares the scratch directory.
        path = str(tmp_path / 'h2o.ff.json')
        assert save_model(path, _model('ff_gradient'))
        sources = read_templates('h2o')

        assert load_cached_model(path, sources, _config(),
                                 'ff_gradient') is not None
        assert load_cached_model(path, sources, _config(lammps_shake=1),
                                 'ff_nvt') is None
        assert load_cached_model(path, sources,
                                 _config(shake_fix_id='other'),
                                 'ff_gradient') is None

    def test_unreadable_cache_is_ignored(self, launch_dir, tmp_path):
        path = str(tmp_path / 'h2o.ff.json')
        with open(path, 'w') as f:
            f.write('not json')
        assert load_cached_model(path, read_templates('h2o'), _config(),
                                 'ff_gradient') is None
        with open(path, 'w') as f:
            json.dump({'format': 999}, f)
        assert load_cached_model(path, read_templates('h2o'), _config(),
                                 'ff_gradient') is None
        assert load_cached_model(str(tmp_path / 'absent.json'),
                                 read_templates('h2o'), _config(),
                                 'ff_gradient') is None


# Atom order of the DATA fixture above (2 waters, O first).
BACKEND_ATOMS = ['O', 'H', 'H', 'O', 'H', 'H']
BACKEND_COORDS = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
                  [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]]


def _backend_raw(**kwargs):
    raw = {'atoms': BACKEND_ATOMS, 'coordinates': BACKEND_COORDS,
           'charges': 0, 'multiplicity': 1, 'nstate': 1, 'state_spin': [0],
           'profile': 'ff_gradient', 'method': 'ff',
           'lammps_project': 'h2o'}
    raw.update(kwargs)
    return raw


class TestBackendWiring:
    """The backend owns the model lifecycle, not the framework."""

    @pytest.fixture
    def backend_module(self, monkeypatch):
        from .. import backend as module
        # No LAMMPS binary is required to build the input files.
        monkeypatch.setattr(module, 'validate_environment', lambda cfg: None)
        return module

    def _backend(self, module, launch_dir, **kwargs):
        return module.LammpsBackend(
            _backend_raw(**kwargs),
            calcdir=str(launch_dir / 'tmp_LAMMPS'))

    def test_model_is_written_inside_the_calculation_directory(
            self, launch_dir, backend_module):
        # The model the package was rendered from is part of the package:
        # it sits beside the data and settings files it produced, so one
        # keep_tmp decides whether all of them survive the run.
        backend = self._backend(backend_module, launch_dir)
        assert backend.ff_model is not None
        cache = launch_dir / 'tmp_LAMMPS' / 'h2o.ff.json'
        assert cache.exists()
        assert os.path.dirname(str(cache)) == backend.calcdir

    def test_second_backend_reuses_the_cache(self, launch_dir, backend_module):
        first = self._backend(backend_module, launch_dir)
        cache = launch_dir / 'tmp_LAMMPS' / 'h2o.ff.json'
        stamp = cache.read_text()
        second = self._backend(backend_module, launch_dir)
        assert cache.read_text() == stamp
        assert second.ff_model['counts'] == first.ff_model['counts']

    def test_compression_is_unconditional(self, launch_dir, backend_module):
        # There is no switch: every backend compresses, and the only way
        # back to the verbatim templates is a refused model.
        backend = self._backend(backend_module, launch_dir)
        assert backend.ff_model is not None
        assert not hasattr(backend.config.environment, 'ff_compress')

    def test_unprovable_templates_warn_and_fall_back(
            self, launch_dir, backend_module):
        (launch_dir / 'h2o.data').write_text(
            DATA.replace('111  atom types\n', ''))
        with pytest.warns(UserWarning, match='compression disabled'):
            backend = self._backend(backend_module, launch_dir)
        assert backend.ff_model is None


class TestRebuildFromCache:
    """keep_tmp may remove the calcdir; the JSON alone rebuilds it."""

    def test_reloaded_model_renders_identical_inputs(self, launch_dir):
        from ..service.builder import write_input_file
        from ..service.workflow import compile_workflow

        config = _config()
        model = build_ff_model(config, 'ff_gradient')
        path = model_path('h2o', str(launch_dir))
        assert save_model(path, model)
        with open(path) as f:
            reloaded = json.load(f)

        workflow = compile_workflow('ff_gradient')
        live, rebuilt = launch_dir / 'live', launch_dir / 'rebuilt'
        live.mkdir()
        rebuilt.mkdir()
        write_input_file(config, workflow, str(live), model)
        write_input_file(config, workflow, str(rebuilt), reloaded)

        names = sorted(os.listdir(live))
        assert names and names == sorted(os.listdir(rebuilt))
        for name in names:
            assert (live / name).read_bytes() == (rebuilt / name).read_bytes()


# A tilted box: the bounding planes LAMMPS stores in the data file plus
# the tilt line. Both spellings are valid, and the resident path has to
# carry either through the model unchanged.
TILTED_DATA = DATA.replace(
    '0.0 10.00000000 zlo zhi\n',
    '0.0 10.00000000 zlo zhi\n0.0 0.50000000 0.25000000 xy xz yz\n')
BARE_TILTED_DATA = DATA.replace(
    '0.0 10.00000000 zlo zhi\n',
    '0.0 10.00000000 zlo zhi\n0.0 0.50000000 0.25000000\n')


class TestTriclinicBox:
    """A tilted template compresses instead of degrading to verbatim."""

    def test_labelled_tilt_line_is_part_of_the_model(self, launch_dir):
        (launch_dir / 'h2o.data').write_text(TILTED_DATA)
        model = _model()
        assert model['box'][-1] == '0.0 0.50000000 0.25000000 xy xz yz'

    def test_bare_tilt_line_is_part_of_the_model(self, launch_dir):
        (launch_dir / 'h2o.data').write_text(BARE_TILTED_DATA)
        model = _model()
        assert model['box'][-1] == '0.0 0.50000000 0.25000000'

    def test_rendered_data_keeps_the_tilt_line(self, launch_dir):
        (launch_dir / 'h2o.data').write_text(TILTED_DATA)
        rendered = render_data(_model())
        assert '0.0 0.50000000 0.25000000 xy xz yz' in rendered
        # The tilt line stays between the bounds and the Atoms section,
        # so the per-step box rewrite can still find both by position.
        lines = rendered.splitlines()
        z_index = next(n for n, line in enumerate(lines)
                       if line.endswith('zlo zhi'))
        assert lines[z_index + 1].split()[3:] == ['xy', 'xz', 'yz']

    def test_tilt_wider_than_the_box_is_refused(self, launch_dir):
        # xlo..xhi spans 10 A while the xy shear alone needs 20 A: the
        # true x length goes negative, which is LAMMPS's "Triclinic box
        # is invalid" abort. Say it here, naming the line.
        (launch_dir / 'h2o.data').write_text(DATA.replace(
            '0.0 10.00000000 zlo zhi\n',
            '0.0 10.00000000 zlo zhi\n20.0 0.0 0.0 xy xz yz\n'))
        with pytest.raises(FFModelUnsupported, match='x axis'):
            _model()


class TestCachePersistence:
    """The shared cache survives every reader because no write is partial."""

    def test_save_leaves_only_the_cache_file(self, launch_dir):
        path = model_path('h2o', str(launch_dir))
        assert save_model(path, _model())
        assert sorted(name for name in os.listdir(launch_dir)
                      if name.startswith('h2o.ff.json')) == ['h2o.ff.json']

    def test_save_creates_a_missing_directory(self, launch_dir):
        path = model_path('h2o', str(launch_dir / 'deep' / 'scratch'))
        assert save_model(path, _model())
        assert os.path.exists(path)

    def test_second_save_replaces_the_model(self, launch_dir):
        path = model_path('h2o', str(launch_dir))
        assert save_model(path, _model())
        assert save_model(path, _model(profile='ff_nvt', lammps_shake=1))
        with open(path) as f:
            assert json.load(f)['constraints']['enabled']

    def test_format_bump_invalidates_older_caches(self, launch_dir):
        from ..service.builder import ff_model as module

        path = model_path('h2o', str(launch_dir))
        stale = _model()
        stale['format'] = module.MODEL_FORMAT - 1
        assert save_model(path, stale)
        sources = read_templates('h2o')
        assert load_cached_model(path, sources, _config(), 'ff_gradient') is None
