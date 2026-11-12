"""
======================================================================
R1 unified-center schema acceptance tests
(dev/MULTICENTER_DEVELOPMENT.md section 13, input & schema acceptance).

These tests lock the R1 contract that the "center" is the single
partition axis of the calculation data: a single center is C=1, a
multi-center run is C>1, and both flow through the SAME adapter ->
normalize -> build_center_plan path. The five acceptance groups are:

1. Three input encodings (classic text / JSON / YAML native types)
   produce the identical canonical center plan.
2. The flat single-center form and the single-group nested form are
   equivalent for C=1.
3. Illegal configurations fail while the plan is built, i.e. BEFORE
   any backend directory, thread or Slurm submission.
4. Per-center derivation is strict: distinct state counts, no zip
   truncation; an incomplete multi-center electronic/state setting
   (ci/spin/coupling/root/charges/multiplicity) copies the first
   center, while region fields (highlevel/midlevel) are never
   broadcast and midlevel must match the highlevel center count.
5. raw -> normalized -> serialize -> restore never decrements an
   already 0-based index a second time.

Sep 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import unittest


def _plan_from_classic():
    """Build a C=2 plan from classic raw-string molecule fields.

    Classic strings carry EXTERNAL 1-based indices for highlevel,
    coupling and root; ci/spin are state counts / spin quantum numbers
    (never indices). The adapter decrements the 1-based fields exactly
    once.
    """
    from pyrai3md.schema.adapter import _convert_types
    from pyrai3md.data.center import build_center_plan

    flat = _convert_types({
        'highlevel': '1-2; 3-4',
        'ci': '2 2; 2 2',
        'spin': '0 1; 0 1',
        'coupling': '1 2, 3 4; 1 2',
        'root': '1; 2',
    })
    return build_center_plan(flat)


def _plan_from_native():
    """Build the same C=2 plan from JSON/YAML native typed fields.

    Native typed input reaches the adapter without a raw-string parse.
    Per docs 4.3 the two index fields keep their documented entry base:
    highlevel typed input is still the EXTERNAL 1-based form and is
    decremented once, while coupling typed input is the NORMALIZED
    0-based entry and passes through unchanged (this protects the
    molcas fixtures and the converter-generated adjacent pairs from a
    second decrement). ci/spin/root are not raw indices.
    """
    from pyrai3md.schema.adapter import _convert_types
    from pyrai3md.data.center import build_center_plan

    flat = _convert_types({
        'highlevel': [[1, 2], [3, 4]],
        'ci': [[2, 2], [2, 2]],
        'spin': [[0, 1], [0, 1]],
        'coupling': [[[0, 1], [2, 3]], [[0, 1]]],
        'root': [1, 2],
    })
    return build_center_plan(flat)


class TestThreeInputEquivalence(unittest.TestCase):
    """Acceptance 1: classic text / JSON / YAML -> one canonical plan."""

    def test_classic_and_native_plans_are_identical(self):
        classic = _plan_from_classic()
        json_like = _plan_from_native()
        # YAML native typing is the same as JSON native typing at the
        # adapter boundary (both are already-typed Python objects), so
        # one native plan represents both encodings.
        yaml_like = _plan_from_native()

        self.assertEqual(classic.plan_fingerprint, json_like.plan_fingerprint)
        self.assertEqual(classic.plan_fingerprint, yaml_like.plan_fingerprint)
        self.assertEqual(classic.to_dict(), json_like.to_dict())

    def test_two_centers_derived(self):
        plan = _plan_from_classic()
        self.assertEqual(plan.n_center, 2)
        self.assertFalse(plan.is_single)
        # center 0: ci=[2,2] spin=[0,1] -> nstate 4, pairs [0,1],[2,3]
        c0 = plan.center(0).electronic_spec
        self.assertEqual(c0.nstate, 4)
        self.assertEqual(c0.state_spin, (0, 0, 2, 2))
        self.assertEqual([list(p) for p in c0.coupling], [[0, 1], [2, 3]])
        self.assertEqual([list(p) for p in c0.nac_pair_map], [[0, 1], [2, 3]])
        self.assertEqual(c0.nsoc, 0)
        # center 1: coupling [0,1] only, root 1
        c1 = plan.center(1).electronic_spec
        self.assertEqual([list(p) for p in c1.coupling], [[0, 1]])
        self.assertEqual(c1.initial_root, 1)

    def test_json_roundtrip_through_disk_matches(self):
        """Serialize the plan to JSON text and rebuild the same plan."""
        plan = _plan_from_native()
        blob = json.dumps(plan.to_dict())
        restored = json.loads(blob)

        from pyrai3md.data.center import build_center_plan
        rebuilt = build_center_plan({
            'highlevel': [list(c['high_indices']) for c in restored['centers']],
            'ci': [list(c['electronic_spec']['ci'])
                   for c in restored['centers']],
            'spin': [list(c['electronic_spec']['spin'])
                     for c in restored['centers']],
            'coupling': [[list(p) for p in c['electronic_spec']['coupling']]
                         for c in restored['centers']],
            'root': [c['electronic_spec']['initial_root']
                     for c in restored['centers']],
        })
        self.assertEqual(rebuilt.plan_fingerprint, plan.plan_fingerprint)


class TestSingleCenterFlatNestedEquivalence(unittest.TestCase):
    """Acceptance 2: flat C=1 == single-group nested C=1."""

    def test_flat_and_nested_c1_are_equivalent(self):
        from pyrai3md.data.center import build_center_plan

        flat_plan = build_center_plan({
            'highlevel': [0, 1],
            'ci': [2, 2],
            'spin': [0, 1],
            'root': 0,
        })
        nested_plan = build_center_plan({
            'highlevel': [[0, 1]],
            'ci': [[2, 2]],
            'spin': [[0, 1]],
            'root': [0],
        })
        self.assertEqual(flat_plan.n_center, 1)
        self.assertTrue(flat_plan.is_single)
        self.assertEqual(flat_plan.plan_fingerprint,
                         nested_plan.plan_fingerprint)
        self.assertEqual(flat_plan.to_dict(), nested_plan.to_dict())

    def test_c1_declared_pairs_match_r0_baseline(self):
        """The C=1 classification reproduces the R0 snapshot.

        Adjacent pairs used to be auto-generated here, which is why this
        case carried no coupling. It declares them now: nothing is invented
        for a center that names no pair (docs MULTICENTER 4.4.5).
        """
        from pyrai3md.data.center import build_center_plan

        pairs = [[0, 1], [1, 2], [2, 3]]
        spec = build_center_plan({'ci': [2, 2], 'spin': [0, 1],
                                  'coupling': pairs}
                                 ).center(0).electronic_spec
        self.assertEqual(spec.nstate, 4)
        self.assertEqual(spec.state_spin, (0, 0, 2, 2))
        self.assertEqual([list(p) for p in spec.coupling], pairs)
        self.assertEqual([list(p) for p in spec.nac_pair_map],
                         [[0, 1], [2, 3]])
        self.assertEqual([list(p) for p in spec.soc_pair_map], [[1, 2]])

    def test_c1_without_pairs_declares_none(self):
        from pyrai3md.data.center import build_center_plan

        spec = build_center_plan({'ci': [2, 2], 'spin': [0, 1]}
                                 ).center(0).electronic_spec
        self.assertEqual([], [list(p) for p in spec.coupling])
        self.assertEqual([], [list(p) for p in spec.nac_pair_map])
        self.assertEqual([], [list(p) for p in spec.soc_pair_map])


class TestIllegalConfigFailsBeforeBackend(unittest.TestCase):
    """Acceptance 3: illegal configs raise while the plan is built."""

    def test_root_out_of_range_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError) as ctx:
            build_center_plan({'ci': [2, 2], 'spin': [0, 1], 'root': 4})
        msg = str(ctx.exception)
        # error carries the field path, both ordinals and the range
        self.assertIn('md.root', msg)
        self.assertIn('nstate=4', msg)

    def test_negative_ci_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError):
            build_center_plan({'ci': [-1, 2], 'spin': [0, 1]})

    def test_ci_spin_length_mismatch_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError):
            build_center_plan({'ci': [2, 2, 2], 'spin': [0, 1]})

    def test_duplicate_coupling_pair_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError):
            build_center_plan({
                'ci': [2, 2], 'spin': [0, 1],
                'coupling': [[0, 1], [1, 0]],
            })

    def test_self_coupling_pair_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError):
            build_center_plan({
                'ci': [2, 2], 'spin': [0, 1], 'coupling': [[1, 1]],
            })

    def test_coupling_out_of_range_raises_not_nac_fallback(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        # state 9 is outside [0, 4); it must NOT silently become a NAC
        with self.assertRaises(CenterPlanError):
            build_center_plan({
                'ci': [2, 2], 'spin': [0, 1], 'coupling': [[0, 9]],
            })

    def test_overlapping_highlevel_regions_raise(self):
        from pyrai3md.schema.validate import validate_centers
        data = {
            'highlevel': [[0, 1], [1, 2]],
            'ci': [[1], [1]], 'spin': [[0], [0]], 'root': [0, 0],
            'natom': 4, 'atoms': ['H', 'H', 'H', 'H'],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_centers(data)
        self.assertIn('overlap', str(ctx.exception))

    def test_empty_highlevel_region_raises(self):
        from pyrai3md.schema.validate import validate_centers
        data = {
            'highlevel': [[0, 1], []],
            'ci': [[1], [1]], 'spin': [[0], [0]], 'root': [0, 0],
            'natom': 4, 'atoms': ['H', 'H', 'H', 'H'],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_centers(data)
        self.assertIn('empty high-level region', str(ctx.exception))

    def test_multicenter_with_groups_refused(self):
        from pyrai3md.schema.validate import validate_centers
        data = {
            'highlevel': [[0, 1], [2, 3]],
            'ci': [[1], [1]], 'spin': [[0], [0]], 'root': [0, 0],
            'natom': 4, 'atoms': ['H', 'H', 'H', 'H'],
            'center_groups': [[0, 1]],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_centers(data)
        self.assertIn('center_groups', str(ctx.exception))

    def test_fullwidth_semicolon_rejected_with_hint(self):
        from pyrai3md.schema.adapter import _convert_types
        with self.assertRaises(ValueError) as ctx:
            _convert_types({'ci': '2 2\uff1b 2 2'})
        self.assertIn(';', str(ctx.exception))

    def test_empty_center_group_rejected(self):
        from pyrai3md.schema.adapter import _convert_types
        with self.assertRaises(ValueError):
            _convert_types({'ci': '2 2;; 2 2'})


class TestPerCenterDerivation(unittest.TestCase):
    """Acceptance 4: per-center derivation with first-center broadcast."""

    def test_distinct_state_counts_per_center(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[2, 2], [3]],
            'spin': [[0, 1], [0]],
            'root': [0, 2],
        })
        self.assertEqual(plan.n_center, 2)
        self.assertEqual(plan.center(0).nstate, 4)
        self.assertEqual(plan.center(1).nstate, 3)
        self.assertEqual(plan.center(1).electronic_spec.initial_root, 2)

    def test_flat_ci_broadcast_to_multi_center(self):
        from pyrai3md.data.center import build_center_plan
        # An incomplete multi-center setting copies the first center
        # (docs 4.4): a single flat ci/spin group is replicated to
        # every center instead of being rejected.
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'ci': [2, 2],
            'spin': [0, 1],
        })
        self.assertEqual(plan.n_center, 2)
        self.assertEqual(plan.center(0).nstate, 4)
        self.assertEqual(plan.center(1).nstate, 4)
        self.assertEqual(list(plan.center(1).electronic_spec.ci), [2, 2])

    def test_scalar_root_broadcast_to_multi_center(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'ci': [[1], [1]],
            'spin': [[0], [0]],
            'root': 0,
        })
        self.assertEqual(plan.n_center, 2)
        self.assertEqual(plan.center(0).electronic_spec.initial_root, 0)
        self.assertEqual(plan.center(1).electronic_spec.initial_root, 0)

    def test_center_index_out_of_range_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        plan = build_center_plan({'ci': [2, 2], 'spin': [0, 1]})
        with self.assertRaises(CenterPlanError):
            plan.center(1)

    def test_explicit_empty_coupling_group_means_no_pairs(self):
        """C>1 protocol: an explicit [] group is 'no pairs', not default."""
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'ci': [[2], [2]],
            'spin': [[0], [0]],
            'coupling': [[], []],
            'root': [0, 0],
        })
        self.assertEqual(plan.center(0).electronic_spec.npair, 0)
        self.assertEqual(plan.center(1).electronic_spec.npair, 0)

    def test_classic_none_token_gives_empty_group(self):
        from pyrai3md.schema.adapter import _convert_types
        converted = _convert_types({'coupling': '1 2; none'})
        self.assertEqual(converted['coupling'], [[[0, 1]], []])


class TestNoSecondDecrement(unittest.TestCase):
    """Acceptance 5: normalized indices are never decremented twice."""

    def test_highlevel_normalized_entry_passthrough(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        # normalized (raw=False) typed 0-based input is untouched
        self.assertEqual(
            _parse_center_indices([[0, 1]], 'highlevel', start=1,
                                  raw=False),
            [[0, 1]])

    def test_plan_high_indices_are_zero_based(self):
        plan = _plan_from_classic()
        self.assertEqual(list(plan.center(0).high_indices), [0, 1])
        self.assertEqual(list(plan.center(1).high_indices), [2, 3])

    def test_rebuild_from_normalized_indices_is_stable(self):
        """Feeding a plan's own 0-based output back yields the same plan."""
        from pyrai3md.data.center import build_center_plan
        plan = _plan_from_native()
        rebuilt = build_center_plan({
            'highlevel': [list(c.high_indices) for c in plan.centers],
            'ci': [list(c.electronic_spec.ci) for c in plan.centers],
            'spin': [list(c.electronic_spec.spin) for c in plan.centers],
            'coupling': [[list(p) for p in c.electronic_spec.coupling]
                         for c in plan.centers],
            'root': [c.electronic_spec.initial_root for c in plan.centers],
        })
        self.assertEqual(rebuilt.plan_fingerprint, plan.plan_fingerprint)
        self.assertEqual([list(c.high_indices) for c in rebuilt.centers],
                         [[0, 1], [2, 3]])


class TestCenterChargeSpin(unittest.TestCase):
    """Region charge / spin multiplicity: per-center, broadcast, aggregate."""

    def test_scalar_charge_multiplicity_broadcast(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'ci': [[1], [1]], 'spin': [[0], [0]],
            'charges': 0, 'multiplicity': 1,
        })
        self.assertEqual(plan.center(0).charge, 0)
        self.assertEqual(plan.center(1).charge, 0)
        self.assertEqual(plan.center(0).multiplicity, 1)
        self.assertEqual(plan.center(1).multiplicity, 1)
        self.assertEqual(plan.system_charge, 0)
        self.assertEqual(plan.system_multiplicity, 1)

    def test_per_center_charge_multiplicity_and_aggregate(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'ci': [[1], [1]], 'spin': [[0], [0]],
            'charges': [0, -1], 'multiplicity': [1, 2],
        })
        self.assertEqual(plan.center(0).charge, 0)
        self.assertEqual(plan.center(1).charge, -1)
        self.assertEqual(plan.center(1).multiplicity, 2)
        # system charge is the sum; multiplicity pairs off unpaired
        # electrons: (0 + 1) mod 2 + 1 = 2 -> doublet.
        self.assertEqual(plan.system_charge, -1)
        self.assertEqual(plan.system_multiplicity, 2)

    def test_partial_charge_multiplicity_broadcast_from_first(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3], [4, 5]],
            'ci': [[1], [1], [1]], 'spin': [[0], [0], [0]],
            'charges': [1], 'multiplicity': [3],
        })
        self.assertEqual([c.charge for c in plan.centers], [1, 1, 1])
        self.assertEqual([c.multiplicity for c in plan.centers], [3, 3, 3])
        self.assertEqual(plan.system_charge, 3)
        # triplet x3: (2+2+2) mod 2 + 1 = 1 -> singlet.
        self.assertEqual(plan.system_multiplicity, 1)

    def test_charge_group_count_exceeds_centers_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError):
            build_center_plan({
                'highlevel': [[0, 1], [2, 3]],
                'ci': [[1], [1]], 'spin': [[0], [0]],
                'charges': [0, 0, 0],
            })

    def test_multiplicity_below_one_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError):
            build_center_plan({
                'highlevel': [[0, 1], [2, 3]],
                'ci': [[1], [1]], 'spin': [[0], [0]],
                'multiplicity': [1, 0],
            })

    def test_to_dict_carries_new_fields(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'ci': [[1], [1]], 'spin': [[0], [0]],
            'charges': [0, -1], 'multiplicity': [3, 2],
        })
        blob = plan.to_dict()
        self.assertEqual(blob['system_charge'], -1)
        # (2 + 1) mod 2 + 1 = 2 -> doublet.
        self.assertEqual(blob['system_multiplicity'], 2)
        self.assertEqual(blob['centers'][1]['charge'], -1)
        self.assertEqual(blob['centers'][1]['multiplicity'], 2)


class TestAggregateSystemMultiplicity(unittest.TestCase):
    """Unpaired-electron pairing rule for the L(real) multiplicity."""

    def test_pairing_rules(self):
        from pyrai3md.data.center import aggregate_system_multiplicity
        # Low-spin parity rule: total_unpaired = sum(mult_c - 1) % 2,
        # system_multiplicity = total_unpaired + 1 (docs MULTICENTER
        # 4.4.1). The aggregate is only consumed for C>1 (a C=1 run
        # skips region injection and normalize uses the scalar).
        self.assertEqual(aggregate_system_multiplicity([3, 3]), 1)  # T+T->S
        self.assertEqual(aggregate_system_multiplicity([2, 2]), 1)  # D+D->S
        self.assertEqual(aggregate_system_multiplicity([3, 2]), 2)  # T+D->D
        self.assertEqual(aggregate_system_multiplicity([1, 1]), 1)  # S+S->S
        self.assertEqual(aggregate_system_multiplicity([2, 2, 2]), 2)
        self.assertEqual(aggregate_system_multiplicity([1, 1, 1]), 1)


class TestMidlevelConstraint(unittest.TestCase):
    """midlevel region count must equal the highlevel center count C."""

    def test_midlevel_count_mismatch_raises(self):
        from pyrai3md.data.center import build_center_plan, CenterPlanError
        with self.assertRaises(CenterPlanError) as ctx:
            build_center_plan({
                'highlevel': [[0, 1], [2, 3]],
                'midlevel': [[0, 1, 2, 3]],
                'ci': [[1], [1]], 'spin': [[0], [0]],
            })
        msg = str(ctx.exception)
        self.assertIn('midlevel', msg)
        self.assertIn('one-to-one', msg)

    def test_midlevel_matching_count_builds_three_layer(self):
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3]],
            'midlevel': [[0, 1, 4], [2, 3, 5]],
            'ci': [[1], [1]], 'spin': [[0], [0]],
        })
        self.assertEqual(plan.n_center, 2)
        self.assertEqual(plan.n_layer, 3)
        self.assertEqual(list(plan.center(0).mid_indices), [0, 1, 4])

    def test_validate_centers_midlevel_mismatch_raises(self):
        from pyrai3md.schema.validate import validate_centers
        data = {
            'highlevel': [[0, 1], [2, 3]],
            'midlevel': [[0, 1, 2, 3]],
            'ci': [[1], [1]], 'spin': [[0], [0]], 'root': [0, 0],
            'natom': 6, 'atoms': ['H'] * 6,
        }
        with self.assertRaises((ValueError, Exception)) as ctx:
            validate_centers(data)
        self.assertIn('midlevel', str(ctx.exception))


class TestChargeSpinSchema(unittest.TestCase):
    """normalize aggregate derivation + validate_electronic per-center."""

    def test_normalize_uses_aggregate_for_electron_counts(self):
        from pyrai3md.schema.normalize import compute_derived_quantities
        out = compute_derived_quantities({
            'atoms': ['H', 'H'], 'charges': [1, -1],
            'multiplicity': [2, 2],
        })
        # aggregate charge 0 -> 2 electrons; doublet+doublet -> singlet
        # (unpaired 0) -> nalpha == nbeta == 1.
        self.assertEqual(out['total_electron'], 2)
        self.assertEqual(out['nalpha'], 1)
        self.assertEqual(out['nbeta'], 1)

    def test_validate_electronic_per_center_below_one_raises(self):
        from pyrai3md.schema.validate import validate_electronic
        with self.assertRaises(ValueError) as ctx:
            validate_electronic({
                'multiplicity': [1, 0], 'total_electron': 2,
            })
        self.assertIn('multiplicity', str(ctx.exception))

    def test_validate_electronic_uses_aggregate(self):
        from pyrai3md.schema.validate import validate_electronic
        # triplet + triplet on 2 electrons: aggregate singlet (unpaired
        # 0) passes the global check even though each center is a
        # triplet; the per-center values stay >= 1.
        data = validate_electronic({
            'multiplicity': [3, 3], 'total_electron': 2,
        })
        self.assertEqual(data['multiplicity'], [3, 3])


if __name__ == '__main__':
    unittest.main()
