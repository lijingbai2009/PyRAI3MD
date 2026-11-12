"""
======================================================================
State declaration and property-request criteria.

Three tables are checked here, all three because a request can be lost
between the input and the deck:

* every property a jobtype asks for must survive the resolution to a
  profile (`tddft_soc` and `tddft_nac` each dropped `gradient`, which is
  how a crossing search ran with two identical state gradients - the
  engine printed no `CARTESIAN GRADIENT` at all, and the gap direction
  came back exactly zero while the row passed);
* one state declaration: `ci`/`spin` say how many states there are and
  of which spin, `coupling` says which pairs are coupled, and
  `nac_pair_map`/`soc_pair_map` are the derived internal labels - an
  extra entry available to the nested input formats, never a second
  source of truth. Either side fills the other in, a disagreement is
  refused, and nothing is invented for a run that declared no pair;
* every input keyword that states a computation need reaches the planner,
  and the generated input renders the declaration without extending it.

Nov 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import yaml

from pyrai3md.schema.capabilities import TOKEN_TO_CAPABILITY


class TestRequestedStepsSurviveTheProfile(unittest.TestCase):
    """A compound jobtype may not resolve to a profile that drops one of it."""

    # `transition_dipole` and `dipole` are served by the same evaluation as
    # the energies they differentiate, and `frequency` by its own module; the
    # tokens below are the ones whose producer is a distinct step, so a
    # profile that omits every producer of one answers nothing.
    #
    # Every token needs its own step: nothing here substitutes for another.
    # A frequency profile is not an answer to a `gradient` token - ORCA will
    # not put EnGrad beside a Hessian keyword (`service/utils.py::
    # writes_engrad`, measured in the test environment as the rc=64 refusal of
    # EnGrad+NACME in the same deck), and the framework never spells the two
    # together anyway, because a Hessian request absorbs the gradient it
    # implies (`planning/assembly.py::spell_skill_jobtype`, IMPLIED_BY_HESSIAN) and
    # the numerical route asks for gradients instead. The compound jobtype
    # that claimed otherwise is removed from the ORCA vocabulary, so writing
    # it by hand now fails at profile resolution.
    PRODUCERS = {
        'gradient': ('gradient',),
        'nac': ('nac',),
        'soc': ('soc',),
    }

    # The same question on the Molcas side, where a property is a module of
    # the deck rather than a step keyword.
    REQUIRED_MODULE = {
        'gradient': ('ALASKA',),
        'nac': ('ALASKA',),
        'soc': ('RASSI',),
    }

    def _served(self, token, steps):
        return any(producer in steps for producer in self.PRODUCERS[token])

    def _entries(self, module, suffix_map, producer):
        for jobtype in suffix_map:
            if ' ' not in jobtype:
                continue          # a single token is the profile's own name
            for base in module.AVAILABLE_METHOD:
                try:
                    profile = module.resolve_profile(base, jobtype)
                except ValueError:
                    continue      # a combination the skill refuses by design
                if profile not in producer:
                    continue
                yield jobtype, profile, set(producer[profile])

    def test_orca_keeps_every_requested_step(self):
        from pyrai3md.skills.orca_skill.schema import method as m

        missing = []
        for jobtype, profile, steps in self._entries(
                m, m._JOBERTYPE_SUFFIX_MAP, m.PROFILE_STEP_MAP):
            for token in jobtype.split():
                if token in self.PRODUCERS and not self._served(token, steps):
                    missing.append((jobtype, profile, token))
        self.assertEqual([], missing)

    def test_molcas_keeps_every_requested_module(self):
        from pyrai3md.skills.molcas_skill.schema import method as m

        missing = []
        for jobtype, profile, modules in self._entries(
                m, m._JOBBERTYPE_SUFFIX_MAP, m.PROFILE_MODULE_MAP):
            for token in jobtype.split():
                for step in self.REQUIRED_MODULE.get(token, ()):
                    if step not in modules:
                        missing.append((jobtype, profile, step))
        self.assertEqual([], missing)


class TestStateDeclarationIsDerivedOnce(unittest.TestCase):
    """ci/spin + coupling are the authority; the pair maps follow."""

    @staticmethod
    def _derive(**overrides):
        from pyrai3md.schema.normalize import compute_derived_quantities
        data = {'atoms': ['O', 'H', 'H'], 'natom': 3, 'ci': [2],
                'spin': [0], 'coupling': []}
        data.update(overrides)
        return compute_derived_quantities(data)

    def test_coupling_is_classified_into_the_two_pair_maps(self):
        out = self._derive(coupling=[[0, 1]])
        self.assertEqual([[0, 1]], out['nac_pair_map'])
        self.assertEqual([], out['soc_pair_map'])
        self.assertEqual(1, out['nnac'])
        self.assertEqual(0, out['nsoc'])
        self.assertEqual(1, out['npair'])

    def test_cross_spin_coupling_becomes_soc_pairs(self):
        out = self._derive(ci=[1, 1], spin=[0, 1], coupling=[[0, 1]])
        self.assertEqual([], out['nac_pair_map'])
        self.assertEqual([[0, 1]], out['soc_pair_map'])

    def test_a_declared_pair_map_must_match_the_coupling(self):
        # The two sides are one statement, so a disagreement is a broken
        # input, not a preference. It used to be settled by overwriting
        # whichever side the input cared about.
        with self.assertRaises(ValueError):
            self._derive(ci=[3], coupling=[[0, 1]], nac_pair_map=[[1, 2]])
        # Same pair, wrong channel: [0, 1] is singlet-singlet, so it is a
        # NAC and declaring it as SOC contradicts the spin list.
        with self.assertRaises(ValueError):
            self._derive(coupling=[[0, 1]], soc_pair_map=[[0, 1]])

    def test_pairs_alone_derive_the_coupling(self):
        # The nested input format's extra entry: declare the pairs and the
        # authority is filled in from them, and only then checked.
        out = self._derive(coupling=[], nac_pair_map=[[0, 1]])
        self.assertEqual([[0, 1]], out['coupling'])
        self.assertEqual([[0, 1]], out['nac_pair_map'])

    def test_nothing_is_invented_when_a_coupling_is_needed(self):
        # Adjacent pairs used to be silently supplied for any run with two
        # or more states, which reads back as a computed coupling for a pair
        # nobody asked about. A run that needs a coupling says so; the rest
        # keep an empty pair map.
        out = self._derive(coupling=[], nac_pair_map=[], soc_pair_map=[])
        self.assertEqual([], out['coupling'])
        self.assertEqual([], out['nac_pair_map'])
        self.assertEqual([], out['soc_pair_map'])
        self.assertEqual(0, out['npair'])
        self.assertNotIn([[0, 1]], out.get('transition_pair', []))

    def test_a_dynamics_run_that_asks_for_couplings_without_declaring_them(self):
        # The derivation itself never refuses an empty declaration - a run
        # that asks for no coupling has none, and that is the shielding rule -
        # so the refusal belongs to the requester, which is the only place
        # that knows a coupling is consumed. Covered at that layer by
        # `TestEveryInputNeedReachesThePlanner
        # .test_a_requested_coupling_without_a_declaration_is_refused`;
        # asserted here that no `needs_coupling` knob crept back into the
        # derivation, where nothing would set it.
        out = self._derive(coupling=[], nac_pair_map=[])
        self.assertEqual([], out['coupling'])
        with self.assertRaises(TypeError):
            from pyrai3md.schema.normalize import compute_derived_quantities
            compute_derived_quantities({}, needs_coupling=True)

    def test_a_self_pair_is_refused_not_dropped(self):
        # [1, 1] used to classify into neither map, so nsoc silently went to
        # zero and the declaration disappeared.
        with self.assertRaises(ValueError):
            self._derive(ci=[2], coupling=[[1, 1]])

    def test_jobtype_tokens_do_not_become_state_declarations(self):
        # The request side is a separate question: `dosoc`/`nactype` say what
        # to compute, never which states exist. Checked here so the two
        # vocabularies cannot drift into each other again.
        self.assertEqual('nac', TOKEN_TO_CAPABILITY['nac'])
        self.assertEqual('soc', TOKEN_TO_CAPABILITY['soc'])


class TestEveryInputNeedReachesThePlanner(unittest.TestCase):
    """A need stated anywhere in the input is scanned before planning.

    The jobtype token is not the only place a computation is asked for: a
    hopping integrator, a crossing optimizer and the SOC switch each state
    one in their own section. Every one of those keywords has to reach
    `resolve_capabilities`, which is the sole expander of the request - a
    need read only by whichever layer built the task is invisible to
    capability distribution, to the multi-backend ban and to the
    declaration check below.
    """

    def test_every_table_keyword_reaches_the_context(self):
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.planning.resolver import _REQUIREMENT_KEYWORDS
        from pyrai3md.schema.config import PyRAIConfig

        context = Maestro._extract_planning_context(PyRAIConfig())
        missing = [(section, keyword)
                   for section, keywords in _REQUIREMENT_KEYWORDS.items()
                   for keyword in keywords if keyword not in context]
        self.assertEqual([], missing)

    def test_a_crossing_optimizer_states_the_coupling_need(self):
        from pyrai3md.planning.resolver import (
            crossing_coupling_pair, resolve_capabilities)

        meci = {'driver': 'main', 'jobtype': 'optimize', 'opt_method': 'meci',
                'opt_state_i': 0, 'opt_state_j': 1}
        self.assertEqual([0, 1], crossing_coupling_pair(meci))
        self.assertIn('nac',
                      resolve_capabilities(['optimize'], meci)['property_names'])

        # The run has to be the one that searches: a crossing method left in
        # an input whose jobtype is `md` never steps on a plane.
        self.assertIsNone(crossing_coupling_pair(dict(meci, jobtype='md')))
        self.assertNotIn('nac', resolve_capabilities(
            ['md'], dict(meci, jobtype='md'))['property_names'])

        # A spin-forbidden pair does not consume the derivative coupling, so
        # the request must not carry it: an extra evaluation per step that
        # the search throws away.
        self.assertIsNone(crossing_coupling_pair(dict(meci, opt_method='mecp')))
        self.assertNotIn('nac', resolve_capabilities(
            ['optimize'], dict(meci, opt_method='mecp'))['property_names'])
        # Chasing one surface needs no coupling either.
        self.assertIsNone(crossing_coupling_pair(dict(meci, opt_state_j=0)))
        self.assertIsNone(crossing_coupling_pair(dict(meci, opt_state_j=None)))
        # driver names a skill: that skill plans its own search.
        self.assertIsNone(crossing_coupling_pair(dict(meci, driver='pyscf_skill')))

    def test_the_scanned_need_survives_to_the_task_request(self):
        # The pair a crossing search spans its plane on travels only if the
        # declaration names it (docs MULTICENTER 4.4.5): `molecule.coupling`
        # says which couplings are computed, opt_state_i/j only which two
        # surfaces are followed.
        from pyrai3md.data.center import build_center_plan
        from pyrai3md.planning.assembly import _task_property_request

        descriptor = {'center_index': 0, 'provides_electronic': True,
                      'capabilities': ['energy', 'gradient', 'nac'],
                      'skill_name': 'molcas_skill'}
        plan = build_center_plan({'ci': [3], 'spin': [0],
                                  'coupling': [[1, 2]]})
        spec = _task_property_request(descriptor, plan, crossing_pair=[1, 2])
        self.assertEqual([[1, 2]], spec['nac_pairs'])
        self.assertEqual([1, 2], spec['target_states'])
        # Direction is not the identity of a pair; its declared spelling is
        # what the engine's table and the parser's rows agree on.
        self.assertEqual(
            [[1, 2]], _task_property_request(descriptor, plan,
                                             crossing_pair=[2, 1])['nac_pairs'])
        with self.assertRaises(ValueError) as ctx:
            _task_property_request(descriptor, plan, crossing_pair=[0, 1])
        self.assertIn('molecule.coupling', str(ctx.exception))

    def test_a_requested_coupling_without_a_declaration_is_refused(self):
        from pyrai3md.data.center import build_center_plan
        from pyrai3md.planning.assembly import _task_property_request

        descriptor = {'center_index': 0, 'provides_electronic': True,
                      'capabilities': ['energy', 'gradient', 'nac'],
                      'skill_name': 'molcas_skill'}
        plan = build_center_plan({'ci': [3], 'spin': [0]})
        with self.assertRaises(ValueError) as ctx:
            _task_property_request(descriptor, plan)
        self.assertIn('molecule.coupling', str(ctx.exception))

        # The same plan asked for nothing else asks for nothing: the pair
        # map is empty because the run shields every coupling.
        quiet = {'center_index': 0, 'provides_electronic': True,
                 'capabilities': ['energy', 'gradient']}
        self.assertEqual([], _task_property_request(quiet, plan)['nac_pairs'])

    def test_a_channel_is_declared_against_its_own_pairs(self):
        # [0, 1] here is singlet-triplet, so it is a SOC pair: a run that
        # asks for NAC has no declaration behind the request even though
        # `coupling` is not empty. That is the case a single "is anything
        # declared" test would pass by accident.
        from pyrai3md.data.center import build_center_plan
        from pyrai3md.planning.assembly import _task_property_request

        plan = build_center_plan({'ci': [1, 1], 'spin': [0, 1],
                                  'coupling': [[0, 1]]})
        base = {'center_index': 0, 'provides_electronic': True,
                'skill_name': 'molcas_skill',
                'capabilities': ['energy', 'gradient']}
        soc = dict(base, capabilities=['energy', 'gradient', 'soc'])
        nac = dict(base, capabilities=['energy', 'gradient', 'nac'])
        self.assertEqual([[0, 1]],
                         _task_property_request(soc, plan)['soc_pairs'])
        with self.assertRaises(ValueError):
            _task_property_request(nac, plan)


class TestGeneratedInputCarriesTheDeclaration(unittest.TestCase):
    """The classic input renders the declaration, it does not extend it.

    Adjacent pairs used to be injected for every multistate case, so a
    generated deck asked the engine for couplings nobody named and the run
    log printed NAC labels for them. A case that declares the internal pair
    maps still has to reach the engine - those pairs are rendered into
    `coupling` - but a case that declares nothing gets no coupling line.
    """

    def _flat(self, tmpdir, input_section):
        from pyrai3md.qualification.certification.converter import (
            parse_test_yaml)
        path = os.path.join(tempfile.mkdtemp(dir=tmpdir), 'case.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump({'name': 'case_test',
                            'input': input_section}, handle)
        return parse_test_yaml(path, 'molcas_skill')

    _BASE = {'nstate': 3, 'state_spin': [0, 0, 0], 'project': 'case'}

    def test_declared_pairs_are_rendered_as_coupling(self):
        with tempfile.TemporaryDirectory() as tmp:
            flat = self._flat(tmp, dict(self._BASE, nac_pair_map=[[0, 1]]))
        self.assertEqual([[0, 1]], flat.get('coupling'))

    def test_nothing_is_rendered_for_an_undeclared_coupling(self):
        with tempfile.TemporaryDirectory() as tmp:
            flat = self._flat(tmp, dict(self._BASE))
        self.assertIsNone(flat.get('coupling'))

    def test_a_declared_coupling_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            flat = self._flat(tmp, dict(
                self._BASE, coupling=[[1, 2]], nac_pair_map=[[0, 1]]))
        # The authority the case wrote wins; the pair map is the derived side
        # and cannot replace it (and a disagreement is refused when the
        # config is loaded, not silently resolved here).
        self.assertEqual([[1, 2]], flat.get('coupling'))


class TestDeclaredPairsSurviveTheSkillEnumeration(unittest.TestCase):
    """A declared subset of pairs reaches the deck and the rows unchanged.

    The five skills that carry a spin ladder used to answer a coupling
    request by enumerating their own sorted state list: for three singlets
    and three triplets that is nine SOC pairs, no matter how few the input
    named. The framework then reads row *k* as the *k*-th declared pair
    (`workflow/integrator/fssh.py`), so the enumeration and the addressing
    were two index spaces with nothing between them.

    Witness: restore the enumeration (the `build_spin_pair_maps` branch
    below the declaration check) and the first test reads nine pairs where
    two were declared; restore the parser's `config.soc_pair_map` fallback
    and the packed rows are addressed by a table nobody declared.
    """

    SPIN = [0, 0, 0, 2, 2, 2]
    DECLARED = [[1, 3], [2, 4]]
    ALL_CROSS = [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]

    @staticmethod
    def _orca(profile, **overrides):
        from pyrai3md.skills.orca_skill.schema.adapter import build_config
        from pyrai3md.skills.orca_skill.schema.normalize import normalize
        raw = {'atoms': ['O', 'H', 'H'],
               'coordinates': [[0.0, 0.0, 0.1173],
                               [0.0, 0.7572, -0.4692],
                               [0.0, -0.7572, -0.4692]],
               'charges': 0, 'multiplicity': 1, 'nstate': 3,
               'state_spin': [0, 0, 0], 'profile': profile,
               'method': profile.split('_')[0],
               'functional': 'pbe0', 'basis': 'sto-3g'}
        raw.update(overrides)
        return normalize(build_config(raw))

    def _soc_config(self):
        return self._orca('tddft_soc', nstate=len(self.SPIN),
                          state_spin=list(self.SPIN),
                          coupling=[list(pair) for pair in self.DECLARED],
                          target_states=[1, 2, 3, 4])

    @staticmethod
    def _socme_table(rows):
        """A DoSOC SOCME block in the shape the deployment prints."""
        rule = '      ' + '-' * 100
        lines = [rule,
                 '                     CALCULATED SOCME BETWEEN TRIPLETS'
                 ' AND SINGLETS',
                 rule,
                 '           Root                          <T|HSO|S>  (Re, Im)'
                 ' cm-1',
                 '         T      S             Z                     X'
                 '                   Y',
                 rule]
        for triplet_root, singlet_root, value in rows:
            lines.append(
                '         %d      %d    (   0.00 ,%8.2f )    (   0.00 ,'
                '    0.00 )    (  -0.00 ,   -0.00 )'
                % (triplet_root, singlet_root, value))
        return lines

    @staticmethod
    def _gradient_block(scale):
        lines = ['CARTESIAN GRADIENT',
                 '---------------------------------------------', '']
        for index, atom in enumerate(['O', 'H', 'H'], start=1):
            lines.append('   %d   %s   :   %12.6f %12.6f %12.6f'
                         % (index, atom, scale * 0.1, scale * 0.2,
                            scale * 0.3))
        return lines

    def test_only_the_declared_pairs_become_rows(self):
        config = self._soc_config()
        self.assertEqual([list(pair) for pair in self.DECLARED],
                         [list(pair) for pair in config.soc_pair_map])
        self.assertEqual([], [list(pair) for pair in config.nac_pair_map])
        self.assertEqual(2, int(config.nsoc))
        self.assertEqual(0, int(config.nnac))
        # The placeholder the run returns into has the same row count as the
        # declaration: a longer one would be a row the input never asked for.
        self.assertEqual(2, len(config.soc))

    def test_the_deck_names_only_the_roots_the_pairs_touch(self):
        from pyrai3md.skills.orca_skill.service.builder.input_builder \
            import build_input
        from pyrai3md.skills.orca_skill.service.workflow import \
            compile_workflow

        config = self._soc_config()
        deck = build_input(config, compile_workflow('tddft_soc'))
        named = [line.strip() for line in deck.splitlines()
                 if 'rootlist' in line]
        # Singlet roots 1 and 2, triplet roots 1 and 2 - the four states the
        # two declared pairs name, and no others.
        self.assertIn('irootlist 1, 2', named)
        self.assertIn('trootlist 1, 2', named)

    def test_the_packed_rows_are_the_declaration_in_order(self):
        import numpy as np
        from pyrai3md.skills.orca_skill.service.parser.output_parser \
            import parse_output
        from pyrai3md.skills.orca_skill.service.workflow import \
            compile_workflow

        config = self._soc_config()
        nsinglet = int(config.nsing_root) + 1
        # The engine answers every cross pair it solved; the run reports only
        # the declared ones, in the declared order.
        rows = [(triplet - nsinglet + 1, singlet, 10.0 * singlet + triplet)
                for singlet, triplet in self.ALL_CROSS]
        log = ['FINAL SINGLE POINT ENERGY  -76.000000000']
        for root in (1, 2, 3, 4):
            log += self._gradient_block(float(root))
        log += self._socme_table(rows)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, '%s.out' % config.project), 'w') as h:
                h.write('\n'.join(log) + '\n')
            results = parse_output(config, tmp,
                                   compile_workflow('tddft_soc'))
        values = np.asarray(results['soc']).tolist()
        self.assertEqual([13.0, 24.0], values)

    def test_a_declared_pair_the_log_does_not_answer_is_refused(self):
        import numpy as np
        from pyrai3md.skills.orca_skill.service.parser.output_parser \
            import parse_output
        from pyrai3md.skills.orca_skill.service.workflow import \
            compile_workflow

        config = self._soc_config()
        nsinglet = int(config.nsing_root) + 1
        answered = [[1, 3]]
        rows = [(triplet - nsinglet + 1, singlet, 10.0 * singlet + triplet)
                for singlet, triplet in answered]
        log = ['FINAL SINGLE POINT ENERGY  -76.000000000']
        for root in (1, 2, 3, 4):
            log += self._gradient_block(float(root))
        log += self._socme_table(rows)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, '%s.out' % config.project), 'w') as h:
                h.write('\n'.join(log) + '\n')
            with self.assertRaisesRegex(RuntimeError,
                                        'prints no spin-orbit matrix element'):
                parse_output(config, tmp, compile_workflow('tddft_soc'))


class TestSkillCannotPermuteADeclaredLadder(unittest.TestCase):
    """Ordering is a check when the input declared pairs, a rewrite when not.

    Two sorts of the same list are two index spaces: the framework already
    handed the skill the state numbers it will address the returned rows by,
    so a skill that permutes them re-numbers every declared pair. A skill
    running on its own - validating what an agent typed - still sorts.
    """

    UNASCENDED = [0, 0, 2, 0]
    SORTED = [0, 0, 0, 2]
    SKILLS = ('orca', 'molcas', 'cp2k', 'pyscf', 'openqp')

    @staticmethod
    def _normalize(skill):
        import importlib
        module = importlib.import_module(
            'pyrai3md.skills.%s_skill.schema.normalize' % skill)
        return module.verify_state_spin

    def _data(self, **overrides):
        data = {'natom': 3, 'nstate': len(self.UNASCENDED), 'multiplicity': 1,
                'state_spin': list(self.UNASCENDED),
                'coupling': [], 'nac_pair_map': [], 'soc_pair_map': [],
                'target_states': [0, 1], 'target_nac_pair': [],
                'target_soc_pair': [], 'profile': 'tddft_soc',
                'method': 'tddft'}
        data.update(overrides)
        return data

    def test_a_declaration_freezes_the_ladder(self):
        # Collected, not sub-tested: one assertion line names every skill that
        # let the permutation through, so the reading is not "some skill failed".
        accepted = []
        for skill in self.SKILLS:
            try:
                self._normalize(skill)(self._data(coupling=[[0, 1]]))
            except ValueError as error:
                if 'ascending 2S' not in str(error):
                    raise
            else:
                accepted.append(skill)
        self.assertEqual([], accepted)

    def test_no_declaration_keeps_the_skill_own_sort(self):
        wrong = []
        for skill in self.SKILLS:
            data = self._normalize(skill)(self._data())
            if list(data['state_spin']) != self.SORTED:
                wrong.append((skill, list(data['state_spin'])))
        self.assertEqual([], wrong)


class TestEveryProfileStepHasAProducer(unittest.TestCase):
    """A profile may not name a step its own deck cannot produce.

    `dft_frequency` carried a `gradient` step while
    `writes_engrad` returns False for a frequency workflow (ORCA will not
    put `EnGrad` beside a Hessian keyword - measured in the test environment as
    the return-code-64 refusal of `EnGrad` with `NACME` in one deck). The
    step was inert and the parser's `is_frequency` guard left the
    pre-allocated zeros in place, so the run reported flat surfaces it had
    never differentiated. The combination is refused now, on both sides.

    Witness: put `gradient` back into `PROFILE_STEP_MAP['dft_frequency']`
    and the ORCA sweep below reports it; put `energy frequency gradient`
    back into the vocabulary and the refusal test reports it.
    """

    # Steps whose producer is a keyword the deck writes and a reader that
    # knows whether to expect the block. The remaining steps (`energy`,
    # `dipole`, `tddft`, `optimize`) are the profile's own evaluation and
    # have no second producer to be substituted for them.
    FILE_PRODUCED = ('gradient', 'hessian', 'nac', 'soc')

    def _orca_produced(self, step, workflow, config):
        from pyrai3md.skills.orca_skill.service.utils import (writes_engrad,
                                                              writes_nacme)
        if step == 'gradient':
            # A NACME deck cannot carry the gradient keyword; the reader then
            # hands back an empty axis, which is absence, not a zero.
            return bool(writes_engrad(workflow, config)
                        or writes_nacme(workflow, config))
        if step == 'hessian':
            return bool(workflow.is_frequency)
        if step == 'nac':
            return bool(workflow.has_step('nac'))
        if step == 'soc':
            return bool(getattr(workflow, 'is_soc', False))
        return True

    def test_every_orca_profile_step_has_a_producer(self):
        from pyrai3md.skills.orca_skill.schema import method as orca_method
        from pyrai3md.skills.orca_skill.schema.adapter import build_config
        from pyrai3md.skills.orca_skill.schema.normalize import normalize
        from pyrai3md.skills.orca_skill.service.workflow import \
            compile_workflow

        inert = []
        for profile, steps in orca_method.PROFILE_STEP_MAP.items():
            raw = {'atoms': ['O', 'H', 'H'],
                   'coordinates': [[0.0, 0.0, 0.1173],
                                   [0.0, 0.7572, -0.4692],
                                   [0.0, -0.7572, -0.4692]],
                   'charges': 0, 'multiplicity': 1, 'nstate': 3,
                   'state_spin': [0, 0, 0], 'profile': profile,
                   'method': orca_method.PROFILE_METHOD_MAP[profile],
                   'functional': 'pbe0', 'basis': 'sto-3g'}
            config = normalize(build_config(raw))
            workflow = compile_workflow(profile)
            for step in steps:
                if step in self.FILE_PRODUCED and \
                        not self._orca_produced(step, workflow, config):
                    inert.append((profile, step))
        self.assertEqual([], inert)

    def test_a_frequency_jobtype_cannot_ask_for_a_gradient(self):
        from pyrai3md.skills.orca_skill.schema import method as orca_method

        # The vocabulary key that claimed otherwise is gone, so the compound
        # is refused by name rather than served by a profile that drops it.
        self.assertNotIn('energy frequency gradient',
                         orca_method._JOBERTYPE_SUFFIX_MAP)
        with self.assertRaises(ValueError):
            orca_method.resolve_profile('dft', 'energy frequency gradient')

    def test_no_molcas_frequency_deck_names_a_gradient_module(self):
        from pyrai3md.skills.molcas_skill.schema import method as molcas_method

        frequency = [profile for profile in molcas_method.PROFILE_MODULE_MAP
                     if 'frequency' in profile]
        self.assertTrue(frequency)
        for profile in frequency:
            modules = molcas_method.PROFILE_MODULE_MAP[profile]
            self.assertNotIn('ALASKA', modules,
                             '%s asks MCKINLEY for a Hessian and ALASKA for a'
                             ' gradient in one deck; nothing here produces the'
                             ' gradient' % profile)


class TestTransitionRowsAreDeclaredOnly(unittest.TestCase):
    """One row per declared transition pair, and never one for a self pair.

    Every skill rebuilt `transition_pair` as [[0, i] for i in
    range(nstate)], which opens with (0, 0) - a transition dipole of the
    ground state to itself, which no engine computes - and the readers then
    padded a second row when the table was empty (`npair = len(pairs) if
    pairs else 1`). Both halves made rows nobody asked for.

    Witness: restore the `range(nstate)` default and the two-state fixture
    below returns two rows; restore the `else 1` in a reader and a run with
    no declared pair returns one.
    """

    def _derived(self, **overrides):
        from pyrai3md.schema.normalize import compute_derived_quantities
        data = {'atoms': ['O', 'H', 'H'], 'natom': 3, 'ci': [2], 'spin': [0],
                'coupling': []}
        data.update(overrides)
        return compute_derived_quantities(data)

    def test_the_framework_derives_ground_to_state_rows(self):
        data = self._derived()
        self.assertEqual([[0, 1]], data['transition_pair'])

    def test_a_self_pair_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'names one state twice'):
            self._derived(transition_pair=[[0, 0]])

    def test_a_declaration_is_read_through_by_every_skill(self):
        import importlib
        skills = {'orca': {}, 'molcas': {}, 'cp2k': {}, 'openqp': {},
                  'pyscf': {'method': 'tddft'},
                  'bagel': {'opt_state_i': 0, 'opt_state_j': 1}}
        base = {'natom': 3, 'nstate': 3, 'multiplicity': 1,
                'state_spin': [0, 0, 0], 'target_states': [0, 1, 2],
                'target_nac_pair': [], 'target_soc_pair': []}
        rewritten = []
        for skill, extra in sorted(skills.items()):
            module = importlib.import_module(
                'pyrai3md.skills.%s_skill.schema.normalize' % skill)
            declared = module.verify_state_num(
                dict(base, transition_pair=[[0, 2]], **extra))
            standalone = module.verify_state_num(dict(base, **extra))
            if (list(declared['transition_pair']) != [[0, 2]]
                    or len(declared['osc_strength']) != 1
                    or len(declared['transition_dipole']) != 1
                    or list(standalone['transition_pair'])
                    != [[0, 1], [0, 2]]):
                rewritten.append(
                    (skill, list(declared['transition_pair']),
                     len(declared['osc_strength']),
                     list(standalone['transition_pair'])))
        self.assertEqual([], rewritten)

    def test_a_two_state_absorption_run_has_one_row(self):
        import numpy as np
        results = self._orca_absorption(2, self._absorption_rows(1, 2))
        self.assertEqual((1, 3), np.asarray(
            results['transition_dipole']).shape)
        self.assertEqual((1,), np.asarray(results['osc_strength']).shape)
        self.assertAlmostEqual(0.01, float(results['osc_strength'][0]))

    def test_a_run_with_no_declared_transition_has_no_row(self):
        import numpy as np
        # One state: there is no ground-to-excited pair to declare, and the
        # log's absorption section is simply not asked for. The reader used to
        # answer one zero row anyway (`npair = len(pairs) if pairs else 1`);
        # restoring that here is what makes this arm go red, which is why the
        # two-state fixture above is not enough on its own.
        results = self._orca_absorption(1, [])
        self.assertEqual((0, 3), np.asarray(
            results['transition_dipole']).shape)
        self.assertEqual((0,), np.asarray(results['osc_strength']).shape)

    @staticmethod
    def _absorption_rows(first, last):
        lines = ['ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS',
                 'state          X           Y           Z']
        for root in range(first, last):
            lines.append('   %d( %dA  )   0.100000   0.000000   0.000000'
                         '   0.010000   0.100000   5.000000   0.010000'
                         % (root, root))
        return lines

    def _orca_absorption(self, nstate, extra_lines):
        from pyrai3md.skills.orca_skill.schema.adapter import build_config
        from pyrai3md.skills.orca_skill.schema.normalize import normalize
        from pyrai3md.skills.orca_skill.service.parser.output_parser \
            import parse_output
        from pyrai3md.skills.orca_skill.service.workflow import \
            compile_workflow

        config = normalize(build_config(
            {'atoms': ['O', 'H', 'H'],
             'coordinates': [[0.0, 0.0, 0.1173],
                             [0.0, 0.7572, -0.4692],
                             [0.0, -0.7572, -0.4692]],
             'charges': 0, 'multiplicity': 1, 'nstate': nstate,
             'state_spin': [0] * nstate, 'profile': 'tddft_energy',
             'method': 'tddft', 'functional': 'pbe0', 'basis': 'sto-3g'}))
        log = list(extra_lines)
        log.append('FINAL SINGLE POINT ENERGY       -76.100000000')
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, '%s.out' % config.project), 'w') as h:
                h.write('\n'.join(log) + '\n')
            return parse_output(config, tmp,
                                compile_workflow('tddft_energy'))


class TestEngineLimitsFallOnTheRequestedChannel(unittest.TestCase):
    """ORCA's structural limits refuse what a deck asks for, not what a run names.

    #448 turned the two engine filters (NACME only S0-Sn, DoSOC never S0) into
    refusals, and applied them to the declaration - which killed `md_orca_fssh`,
    three singlets whose full pair set is declared while `md.nactype: ktdc`
    estimates the coupling from the energy curvature and writes no `NACME`
    switch at all (round `tmp_20261102_114641`, both carriers). A pair nobody
    asks the engine for cannot be answered wrongly; the rule is restored to
    what it guards: the request, not the record.
    """

    PAIRS = [[0, 1], [1, 2], [0, 2]]

    def _orca(self, profile):
        from pyrai3md.skills.orca_skill.schema.adapter import build_config
        from pyrai3md.skills.orca_skill.schema.normalize import normalize
        return normalize(build_config(
            {'atoms': ['C', 'C', 'H', 'H', 'H', 'H'],
             'coordinates': [[-1.120191, 0.0, -2.661941],
                             [-1.120191, 0.0, -1.331783],
                             [-0.590584, 0.756357, -3.234891],
                             [-1.649798, -0.756357, -3.234891],
                             [-1.876548, -0.529607, -0.758833],
                             [-0.363834, 0.529607, -0.758833]],
             'charges': 0, 'multiplicity': 1, 'nstate': 3,
             'state_spin': [0, 0, 0],
             'coupling': [list(pair) for pair in self.PAIRS],
             'profile': profile, 'method': 'tddft',
             'functional': 'pbe0', 'basis': 'sto-3g',
             'target_states': [0, 1, 2]}))

    def test_a_declared_pair_is_kept_when_the_deck_does_not_ask(self):
        config = self._orca('tddft_energy')
        self.assertEqual(self.PAIRS,
                         [list(pair) for pair in config.nac_pair_map])
        self.assertEqual(0, int(config.nsoc))

    def test_the_same_declaration_is_refused_once_the_channel_is_asked(self):
        with self.assertRaisesRegex(ValueError, 'asks for the declared pair'):
            self._orca('tddft_nac')

    def test_a_multi_center_pair_set_is_read_as_one_center_statement(self):
        # The framework hands a term the active center's own `coupling` and its
        # own classification. `openqp_skill` used to add those two together and
        # then compare the sum against the maps, which can never agree, so 26
        # shipped ONIOM2/3 legs were refused for a disagreement that did not
        # exist (round `tmp_20261102_114641`): the same three pairs written
        # twice, against one singlet and two singlet-triplet maps.
        from pyrai3md.skills.openqp_skill.schema.normalize import \
            verify_state_spin
        center = {'natom': 6, 'nstate': 3, 'multiplicity': 1,
                  'state_spin': [0, 0, 2],
                  'coupling': [[0, 1], [1, 2], [0, 2]],
                  'nac_pair_map': [[0, 1]],
                  'soc_pair_map': [[1, 2], [0, 2]]}
        agreed = verify_state_spin(dict(center))
        self.assertEqual([[0, 1]], [list(p) for p in agreed['nac_pair_map']])
        self.assertEqual([[1, 2], [0, 2]],
                         [list(p) for p in agreed['soc_pair_map']])
        # The maps alone are a statement too: `coupling` is filled back in.
        from_maps = verify_state_spin(dict(
            center, coupling=[]))
        self.assertEqual([[0, 1], [1, 2], [0, 2]],
                         [list(p) for p in from_maps['coupling']])
        # And a real disagreement is still refused.
        with self.assertRaisesRegex(ValueError, 'disagree'):
            verify_state_spin(dict(center, soc_pair_map=[[0, 2]]))

    def test_the_shipped_ktdc_example_still_loads(self):
        # Not the witness for the gate above - that is the arm which builds the
        # same declaration through the skill's own normalize. This one holds the
        # other half: the shipped input `md_orca_fssh` (three singlets, all
        # three pairs, kTDC) must still load through the framework chain, so a
        # fix that only satisfies a stand-in dict cannot pass.
        import os
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config as fw_build
        from pyrai3md.schema.normalize import normalize as fw_normalize
        from pyrai3md.schema.validate import validate as fw_validate
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'examples', 'md_orca_fssh.yaml')
        raw = parse_input(path)
        fw_validate(fw_normalize(fw_build(raw)))


class TestAnMdExampleAsksWhatItClaims(unittest.TestCase):
    """A dynamics example must run dynamics, and ask for the coupling it names.

    `md_molcas_fssh_nac` shipped with a `jobtype` line inside its `molcas:`
    section, and that key overwrites `control.jobtype` in the flat merge: the
    case ran **one** single-point evaluation (`jobtype energy gradient nac`,
    `task id qm`), produced no `md.*` trajectory record at all, and still
    reported PASS - the runner prints `[ENERGY CHECK] no <title>.md.energies`
    as a trace instead of a failure. Restoring the skill-section line, or
    dropping `md.nactype`, is what this criterion catches (#448).
    """

    def _state(self, stem):
        import os
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        examples = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'examples')
        raw = parse_input(os.path.join(examples, stem + '.yaml'))
        cfg = validate(normalize(build_config(raw)))
        return cfg, build_state(cfg, raw_config=raw)

    def test_every_md_example_keeps_its_dynamics_jobtype(self):
        import glob
        import os
        examples = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'examples')
        offenders = []
        checked = 0
        for path in sorted(glob.glob(os.path.join(examples, 'md_*.yaml'))):
            stem = os.path.basename(path)[:-5]
            try:
                cfg, state = self._state(stem)
            except FileNotFoundError:
                continue          # a staged sibling the loader needs first
            checked += 1
            if not str(cfg.control.jobtype).startswith('md'):
                offenders.append((stem, cfg.control.jobtype))
        self.assertGreater(checked, 10)
        self.assertEqual([], offenders)

    def test_the_nac_leg_asks_for_nac_and_the_ktdc_leg_does_not(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        asks = {}
        for stem in ('md_molcas_fssh', 'md_molcas_fssh_nac'):
            cfg, state = self._state(stem)
            for _category, task in state.iter_tasks():
                asks[stem] = spell_skill_jobtype(
                    task.get('capabilities') or [], 'energy gradient')
        self.assertEqual('energy gradient', asks['md_molcas_fssh'])
        self.assertEqual('energy gradient nac', asks['md_molcas_fssh_nac'])


if __name__ == '__main__':
    unittest.main()
