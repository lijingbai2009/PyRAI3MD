"""
======================================================================
Test suite for the periodic-input data contract of the schema layer
(dev/methods/PBC_DEVELOPMENT.md stage A):

- lattice six-parameter input through both entry formats (classic
  string and typed YAML/JSON list) and the 3x3 pass-through.
- cell six-parameter and nine-component derivation in normalize,
  including the [99, 99, 99, 90, 90, 90] "not filled" sentinel that
  must not derive a fake lattice.
- pbc normalization and the System structural validators (three 0/1
  flags, cell length 6/9, finite lattice_vector).
- conflicting cell/lattice_vector representations are rejected.
- the validate layer rejects PBC enabled without a valid cell and
  partial periodicity in this stage.

Sep 25 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np
from pydantic import ValidationError

from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.normalize import normalize
from pyrai3md.schema.validate import validate

_TRICLINIC6 = [8.0, 9.0, 10.0, 80.0, 95.0, 100.0]


def _molecule(**extra):
    base = {'atoms': ['O', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0],
                            [0.96, 0.0, 0.0],
                            [-0.24, 0.93, 0.0]]}
    base.update(extra)
    return base


class TestLatticeInput(unittest.TestCase):
    """lattice keyword through both entry formats (stage A2)."""

    def test_classic_string_six_parameters(self):
        config = build_config(
            {'molecule': _molecule(lattice='10 10 10 90 90 90',
                                   pbc='1 1 1')})
        lattice = np.asarray(config.system.lattice_vector)
        np.testing.assert_allclose(lattice, np.eye(3) * 10.0, atol=1e-9)
        self.assertEqual(config.system.pbc, [1, 1, 1])

    def test_typed_six_parameters(self):
        config = build_config(
            {'molecule': _molecule(lattice=list(_TRICLINIC6),
                                   pbc=[1, 1, 1])})
        lattice = np.asarray(config.system.lattice_vector)
        self.assertEqual(lattice.shape, (3, 3))
        # Standard orientation: a along x, b in the xy plane.
        self.assertAlmostEqual(lattice[0][1], 0.0, places=9)
        self.assertAlmostEqual(lattice[1][2], 0.0, places=9)
        self.assertAlmostEqual(float(lattice[0][0]), 8.0, places=9)

    def test_typed_matrix_passthrough(self):
        matrix = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]]
        config = build_config(
            {'molecule': _molecule(lattice=matrix, pbc=[1, 1, 1])})
        self.assertEqual(config.system.lattice_vector, matrix)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            build_config({'molecule': _molecule(lattice='10 10 10')})
        with self.assertRaises(ValueError):
            build_config({'molecule': _molecule(lattice=[1.0] * 5)})

    def test_classic_and_typed_equivalent(self):
        string_form = build_config(
            {'molecule': _molecule(lattice='8 9 10 80 95 100')})
        typed_form = build_config(
            {'molecule': _molecule(lattice=list(_TRICLINIC6))})
        np.testing.assert_allclose(
            np.asarray(string_form.system.lattice_vector),
            np.asarray(typed_form.system.lattice_vector), atol=1e-12)


class TestCellDerivation(unittest.TestCase):
    """cell six/nine component derivation and the sentinel (stage A3)."""

    def test_cell_six_parameters_derives_lattice(self):
        config = normalize(build_config(
            {'molecule': _molecule(cell='9 9 9 90 90 90')}))
        lattice = np.asarray(config.system.lattice_vector)
        np.testing.assert_allclose(lattice, np.eye(3) * 9.0, atol=1e-9)
        self.assertAlmostEqual(config.system.volume, 729.0, places=9)

    def test_cell_nine_components_derives_lattice(self):
        config = normalize(build_config(
            {'molecule': _molecule(
                cell='10 0 0 0 12 0 0 0 14', pbc='1 1 1')}))
        expected = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]]
        np.testing.assert_allclose(
            np.asarray(config.system.lattice_vector), expected, atol=1e-12)
        self.assertAlmostEqual(
            config.system.volume, 10.0 * 12.0 * 14.0, places=9)

    def test_default_sentinel_derives_nothing(self):
        config = normalize(build_config({'molecule': _molecule()}))
        lattice = np.asarray(config.system.lattice_vector)
        self.assertFalse(np.any(lattice))
        self.assertEqual(config.system.cell, [99.0, 99.0, 99.0,
                                              90.0, 90.0, 90.0])
        self.assertEqual(config.system.volume, 0)

    def test_conflicting_cell_and_lattice_rejected(self):
        with self.assertRaises(ValueError):
            normalize(build_config({'molecule': _molecule(
                cell='9 9 9 90 90 90',
                lattice_vector=[[5.0, 0.0, 0.0],
                                [0.0, 5.0, 0.0],
                                [0.0, 0.0, 5.0]])}))

    def test_consistent_cell_and_lattice_accepted(self):
        matrix = np.eye(3) * 9.0
        config = normalize(build_config({'molecule': _molecule(
            cell='9 9 9 90 90 90', lattice_vector=matrix.tolist())}))
        np.testing.assert_allclose(
            np.asarray(config.system.lattice_vector), matrix, atol=1e-12)


class TestPbcNormalizationAndValidation(unittest.TestCase):
    """pbc structural rules (stages A3/A4)."""

    def test_pbc_normalized_to_three_flags(self):
        config = normalize(build_config(
            {'molecule': _molecule(pbc=[1, 1, 1])}))
        self.assertEqual(config.system.pbc, [1, 1, 1])

    def test_out_of_range_pbc_rejected(self):
        with self.assertRaises(ValidationError):
            build_config({'molecule': _molecule(pbc=[2, 0, 0])})

    def test_wrong_length_pbc_rejected(self):
        with self.assertRaises(ValidationError):
            build_config({'molecule': _molecule(pbc=[1, 0])})

    def test_non_finite_cell_rejected(self):
        with self.assertRaises(ValidationError):
            build_config({'molecule': _molecule(
                cell=[float('inf')] * 6)})

    def test_non_finite_lattice_rejected(self):
        with self.assertRaises(ValidationError):
            build_config({'molecule': _molecule(
                lattice_vector=[[float('nan'), 0.0, 0.0],
                                [0.0, 10.0, 0.0],
                                [0.0, 0.0, 10.0]])})

    def test_bad_cell_length_rejected(self):
        with self.assertRaises(ValidationError):
            build_config({'molecule': _molecule(cell=[1.0] * 7)})


class TestValidatePeriodicSemantics(unittest.TestCase):
    """validate-layer semantic rules (stage A5)."""

    def test_pbc_without_cell_rejected(self):
        config = normalize(build_config(
            {'control': {'jobtype': 'sp'},
             'molecule': _molecule(pbc=[1, 1, 1])}))
        with self.assertRaises(ValueError):
            validate(config)

    def test_pbc_with_singular_cell_rejected(self):
        config = normalize(build_config({'molecule': _molecule(
            pbc=[1, 1, 1],
            lattice_vector=[[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [5.0, 0.0, 0.0]])}))
        with self.assertRaises(ValueError):
            validate(config)

    def test_partial_periodicity_rejected_this_stage(self):
        config = normalize(build_config({'molecule': _molecule(
            pbc=[1, 1, 0],
            lattice_vector=(np.eye(3) * 10.0).tolist())}))
        with self.assertRaises(ValueError):
            validate(config)

    def test_non_periodic_passes(self):
        config = normalize(build_config(
            {'control': {'jobtype': 'sp'},
             'molecule': _molecule()}))
        validate(config)

    def test_full_periodic_with_valid_cell_passes(self):
        config = normalize(build_config(
            {'control': {'jobtype': 'energy'},
             'molecule': _molecule(
                 pbc=[1, 1, 1],
                 lattice_vector=(np.eye(3) * 10.0).tolist())}))
        validate(config)

    def test_molecular_backend_with_pbc_rejected(self):
        # Capability gate (PBC docs 5.4): a molecular backend that
        # declares supports_pbc=False is refused when pbc is enabled,
        # instead of silently ignoring the cell.
        config = normalize(build_config(
            {'control': {'jobtype': 'energy', 'theory': 'pyscf'},
             'molecule': _molecule(
                 pbc=[1, 1, 1],
                 lattice_vector=(np.eye(3) * 10.0).tolist())}))
        with self.assertRaises(ValueError):
            validate(config)


class TestMethodCombinationPbcGate(unittest.TestCase):
    """The 2026-10-25 theory/reference length matrix."""

    @staticmethod
    def _config(theory, reference, **control):
        base = {'jobtype': 'energy', 'theory': theory,
                'reference': reference}
        base.update(control)
        return normalize(build_config({
            'control': base,
            'molecule': _molecule(
                pbc=[1, 1, 1],
                lattice_vector=(np.eye(3) * 10.0).tolist()),
        }))

    def test_layered_theory_single_center_reference_passes(self):
        # nn is the center; lammps carries the box; openqp labels the
        # center and need not support PBC.
        validate(self._config(['nn', 'lammps'], ['openqp']))

    def test_layered_theory_reference_does_not_carry_the_box(self):
        # Inverted order puts the molecular nn last, so the theory tail
        # (the whole-system provider) is refused.
        with self.assertRaises(ValueError):
            validate(self._config(['lammps', 'nn'], ['openqp']))

    def test_single_theory_requires_the_reference_to_support_pbc(self):
        validate(self._config(['lammps'], ['xtb']))
        with self.assertRaises(ValueError):
            validate(self._config(['lammps'], ['openqp']))

    def test_single_theory_multiple_references_gate_both_tails(self):
        validate(self._config(['lammps'], ['openqp', 'xtb']))
        with self.assertRaises(ValueError):
            validate(self._config(['xtb'], ['xtb', 'openqp']))

    def test_layered_theory_multiple_references_is_illegal(self):
        with self.assertRaises(ValueError):
            validate(self._config(['nn', 'lammps'], ['openqp', 'xtb']))

    def test_reference_only_with_molecular_skill_is_refused(self):
        with self.assertRaises(ValueError):
            validate(self._config([], ['openqp'], skill_name='openqp'))


class TestPbcCapabilityRegistry(unittest.TestCase):
    """The manifest supports_pbc flag surfaces through discover_all."""

    def test_discover_all_exposes_supports_pbc(self):
        from pyrai3md.skills.query import discover_all
        registry = discover_all()
        for periodic in ('cp2k', 'xtb', 'lammps'):
            self.assertIn(periodic, registry)
            self.assertTrue(registry[periodic].get('supports_pbc'),
                            "%s should declare supports_pbc" % periodic)
        for molecular in ('molcas', 'orca', 'pyscf', 'bagel',
                          'openqp', 'qdm'):
            self.assertIn(molecular, registry)
            self.assertFalse(registry[molecular].get('supports_pbc'),
                             "%s should not declare supports_pbc" % molecular)


class TestInitialImageSeed(unittest.TestCase):
    """normalize derives the crossings a wrapped periodic input implies.

    A snapshot that was wrapped into the main image can hold bonds which
    only close across a cell face. The integrator cannot recover those
    crossings later (it counts from the moment it starts watching), and
    LAMMPS keys its unwrapped geometry on them, so the count has to be
    read off the geometry once, at configuration time.
    """

    @staticmethod
    def _wrapped_water(**extra):
        # One hydrogen wrapped to the far x face: the bond is 0.96
        # Angstrom long only when that hydrogen is counted one cell back.
        base = {'atoms': ['O', 'H', 'H'],
                'coordinates': [[0.10, 5.00, 5.00],
                                [9.14, 5.00, 5.00],
                                [0.10, 5.96, 5.00]],
                'pbc': [1, 1, 1],
                'lattice_vector': (np.eye(3) * 10.0).tolist()}
        base.update(extra)
        return base

    def test_a_straddling_bond_is_counted_at_configuration_time(self):
        config = validate(normalize(
            build_config({'molecule': self._wrapped_water()})))
        self.assertEqual(config.system.image,
                         [[0, 0, 0], [-1, 0, 0], [0, 0, 0]])

    def test_a_molecule_wholly_inside_is_left_unstated(self):
        # Nothing crossed, so there is nothing to seed: an empty field
        # already means "all zero" downstream, and stating rows of zeros
        # would let two sources of truth drift apart.
        config = validate(normalize(build_config(
            {'molecule': self._wrapped_water(
                coordinates=[[0.10, 5.00, 5.00],
                             [1.06, 5.00, 5.00],
                             [0.10, 5.96, 5.00]])})))
        self.assertEqual(config.system.image, [])

    def test_a_declared_image_is_not_overwritten(self):
        declared = [[0, 0, 0], [3, 0, 0], [0, 0, 0]]
        config = validate(normalize(build_config(
            {'molecule': self._wrapped_water(image=declared)})))
        self.assertEqual(config.system.image, declared)

    def test_a_non_periodic_input_owns_no_crossings(self):
        # A cell may be present without being periodic; the flags, not
        # the box, decide whether a crossing can happen at all.
        config = validate(normalize(build_config(
            {'molecule': self._wrapped_water(pbc=[0, 0, 0])})))
        self.assertEqual(config.system.image, [])

    def test_an_unreadable_structure_is_left_as_it_arrived(self):
        # The seed guards on shape instead of raising: reading the
        # structure is validate's job, which the last two lines check
        # together. A configuration-time derivation must not become a
        # new way to fail a run that used to load.
        config = normalize(build_config(
            {'molecule': self._wrapped_water(
                coordinates=[[0.10, 5.00, 5.00], [9.14, 5.00, 5.00]])}))
        self.assertEqual(len(config.system.atoms), 3)
        self.assertEqual(config.system.image, [])
        with self.assertRaises(ValueError):
            validate(config)


if __name__ == '__main__':
    unittest.main()
