"""
======================================================================
Tests for the shipped multi-center QMMM example inputs.

`multicenter_qmmm_sp.yaml` and `multicenter_qmmm_sc_charge.yaml` are the
QM/MM ports of the multi-center ONIOM box example: the same 20-water
periodic box, the same two water centers, Molcas for the high layer and
LAMMPS for the classical one. They are read by the real input pipeline -
parse, normalize, validate, build_state - so a keyword the loader cannot
spell, a plan the expander cannot build, or a queue the calculator cannot
run fails here rather than on a host that has the engines.

The five shipped QMMM cases divide the charge question three ways, and the
duty count of each queue is where that division is visible: the converged
fixed point (`sc_charge` true, `maxiter 5`, seven duties), one fitted round
(`sc_charge` true, `maxiter 1`, seven duties - `sp`, and every case whose
forces are integrated or whose geometry is walked: `md`, `mep`), and no
charge round at all (`sc_charge` off, five duties - `num_freq`, the
displacement batch, where the charges are not the subject). A case that
integrates has to name one electronic structure per evaluation; the zero-tier
surface does not.

Three more inputs ride the same evaluation and ask a different question of it:
`nac_sp` (each center answers its own coupling), `npt` and `cell` (the box is a
variable, and its conjugate force exists on the whole-system duty alone).

Nothing in this file starts an engine. The queue is inspected as the
planner leaves it.

Nov 3 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import unittest

_EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'examples'))

# The two centers are the waters at rows 10-12 and 22-24 (1-based, as the
# input spells them), both of which straddle a box face.
CENTER_A = [9, 10, 11]
CENTER_B = [21, 22, 23]
UNION = sorted(CENTER_A + CENTER_B)


def _load(name):
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    raw = parse_input(os.path.join(_EXAMPLES_DIR, name))
    return raw, validate(normalize(build_config(raw)))


def _state(config, raw):
    """Build the queue the way a run does: the raw input rides along.

    `build_state` merges each skill's own section out of the raw text, so a
    queue built without it puts one skill's `method` on every task and the
    skill-side profile resolution refuses it. The examples are what a run
    reads, so they have to be built the way a run builds.
    """
    from pyrai3md.data.build import build_state
    from pyrai3md.kernel.environment import Environment
    Environment().initialize()
    return build_state(config, raw_config=raw)


class TestShippedMulticenterQmmmInput(unittest.TestCase):
    """Each shipped example validates and plans the queue it claims."""

    def _checked(self, name, sc_charge):
        raw, config = _load(name)
        self.assertEqual(config.control.multiscale_type, 'qmmm')
        self.assertEqual(config.control.theory, ['molcas', 'lammps'])
        self.assertIs(config.system.embedding, True)
        self.assertEqual(config.system.highlevel,
                         [CENTER_A, CENTER_B])
        self.assertIs(bool(config.system.sc_charge), sc_charge)
        state = _state(config, raw)
        return config, state

    def test_one_round_input_plans_two_probes_and_two_solves(self):
        _config, state = self._checked(
            'multicenter_qmmm_sp.yaml', sc_charge=True)
        roles = [task.get('qmmm_role') for task in state.theory]
        # One fitted round, not a fixed point: the cheap 0-order solve runs
        # once per center, the accepted charges are probed again, and only
        # then does the gradient solve see the polarized field.
        self.assertEqual(roles, ['field', 'field', 'qm_charge', 'qm_charge',
                                 'qm', 'qm', 'energy'])
        self.assertEqual([task.get('center_index') for task in state.theory],
                         [0, 1, 0, 1, 0, 1, None])
        self.assertEqual([task.get('contribution') for task in state.theory],
                         [0, 0, 0, 0, 1, 1, 1])
        _raw, config = _load('multicenter_qmmm_sp.yaml')
        # The budget is the whole difference from the converged case: one
        # round, and nothing in the input asks for a second.
        self.assertEqual(config.system.sc_charge_maxiter, 1)

    def test_each_probe_narrows_to_its_own_center(self):
        _config, state = self._checked(
            'multicenter_qmmm_sp.yaml', sc_charge=True)
        probes = [list(task.get('probe_indices') or [])
                  for task in state.theory
                  if task.get('qmmm_role') == 'field']
        self.assertEqual(probes, [CENTER_A, CENTER_B])
        # The two MM duties address the whole QM set: the probe puts its
        # environment charges into one array over it, and the classical
        # surface removes the in-cell Q-Q pairs of every center. A QM task
        # keeps its own region, which is the slice it is solved on.
        for task in state.theory:
            if task.get('qmmm_role') in ('field', 'energy'):
                self.assertEqual(
                    sorted(int(index) for index in
                           (task.get('partition_indices') or [])), UNION)
            else:
                self.assertEqual(
                    sorted(int(index) for index in
                           (task.get('region_indices') or [])),
                    CENTER_A if task.get('center_index') == 0 else CENTER_B)

    def test_self_consistent_input_adds_one_charge_round_per_center(self):
        _config, state = self._checked(
            'multicenter_qmmm_sc_charge.yaml', sc_charge=True)
        roles = [task.get('qmmm_role') for task in state.theory]
        self.assertEqual(roles, ['field', 'field', 'qm_charge', 'qm_charge',
                                 'qm', 'qm', 'energy'])
        self.assertEqual([task.get('contribution') for task in state.theory],
                         [0, 0, 0, 0, 1, 1, 1])
        # A charge round asks for energy only: no gradient token reaches the
        # skill, which is what leaves &ALASKA out of the cheap deck.
        for task in state.theory:
            if task.get('qmmm_role') == 'qm_charge':
                self.assertEqual(task.get('capabilities'), ['energy'])
                self.assertTrue(task.get('provides_charges'))
                self.assertFalse(task.get('provides_electronic'))
        _raw, config = _load('multicenter_qmmm_sc_charge.yaml')
        self.assertEqual(config.system.sc_charge_maxiter, 5)
        self.assertAlmostEqual(config.system.sc_charge_threshold, 1e-3)

    def test_the_plan_carries_two_centers(self):
        _config, state = self._checked(
            'multicenter_qmmm_sc_charge.yaml', sc_charge=True)
        plan = state.partition_plan
        self.assertEqual(plan.n_center, 2)
        self.assertEqual([list(plan.center(c).high_indices)
                          for c in (0, 1)], [CENTER_A, CENTER_B])
        # Both centers are neutral, so the whole-system charge is zero and
        # the classical stage may keep the Ewald convention of a neutral cell.
        self.assertEqual([plan.center(c).charge for c in (0, 1)], [0, 0])

    def test_the_classic_spelling_round_trips(self):
        # The example is written in the ';'-separated center form; the
        # generated classic input must say it back, or a classic carrier
        # loses the second center and the charge iteration along with it.
        import tempfile
        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config)
        work = tempfile.mkdtemp(prefix='multicenter_qmmm_classic_')
        try:
            path = os.path.join(_EXAMPLES_DIR,
                                'multicenter_qmmm_sc_charge.yaml')
            config_path = os.path.join(work, 'mc.json')
            yaml_to_config(path, config_path)
            text = convert(config_path, os.path.join(work, 'mc.input'))
            block = text.split('&MOLECULE')[1].split('&')[0]
            rows = {row.split(None, 1)[0]: row.split(None, 1)[1].strip()
                    for row in block.splitlines() if row.strip()}
            self.assertEqual(rows['highlevel'], '10 11 12; 22 23 24')
            self.assertEqual(rows['sc_charge'], 'true')
            self.assertEqual(rows['sc_charge_maxiter'], '5')
            self.assertEqual(rows['sc_charge_threshold'], '1e-3')
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_no_example_pins_q_qm(self):
        # Under several centers the classical charges are the union of what
        # each center's own fit reported; a pinned set would describe a
        # field nobody built.
        for name in ('multicenter_qmmm_sp.yaml',
                     'multicenter_qmmm_sc_charge.yaml',
                     'multicenter_qmmm_md.yaml'):
            _raw, config = _load(name)
            self.assertEqual(list(config.system.q_qm or []), [])

    def test_the_md_input_plans_one_fitted_round_per_step(self):
        # MD integrates the surface, so the field and the classical stage
        # have to name one electronic structure: one fitted round per
        # evaluation, never a fixed point. `sc_charge_maxiter: 1` is the
        # budget, and the queue therefore carries the charge duties.
        raw, config = _load('multicenter_qmmm_md.yaml')
        self.assertEqual(config.control.jobtype, 'md')
        self.assertIs(bool(config.system.sc_charge), True)
        self.assertEqual(config.system.sc_charge_maxiter, 1)
        state = _state(config, raw)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm_charge', 'qm_charge',
                          'qm', 'qm', 'energy'])
        self.assertEqual([task.get('contribution') for task in state.theory],
                         [0, 0, 0, 0, 1, 1, 1])
class TestMulticenterQmmmCouplingScope(unittest.TestCase):
    """A coupling is per center: its own states, its own pair, its own hop.

    Each center computes its own NAC/SOC between its own states,
    `collect_center_electronic` stores them in `state.electronic[c]` in that
    center's local framing, and `ElectronicStepper` runs one ledger per
    center, so nothing is aggregated across centers. What the run still
    refuses is a coupling request with no declared pair: a pair that is not
    named is shielded, and there would be nothing to read back.
    """

    def test_a_coupling_request_needs_a_declared_pair(self):
        raw, config = _load('multicenter_qmmm_sc_charge.yaml')
        config.control.jobtype = 'energy gradient nac'
        with self.assertRaises(ValueError) as caught:
            _state(config, raw)
        self.assertIn('declares no pair', str(caught.exception))

    def test_the_same_rule_covers_spin_orbit(self):
        raw, config = _load('multicenter_qmmm_sc_charge.yaml')
        config.control.jobtype = 'energy gradient soc'
        with self.assertRaises(ValueError) as caught:
            _state(config, raw)
        self.assertIn('declares no pair', str(caught.exception))

    def test_a_declared_pair_reaches_every_center(self):
        raw, config = _load('multicenter_qmmm_sc_charge.yaml')
        config.control.jobtype = 'energy gradient nac'
        config.system.ci = [2, 2]
        config.system.state_spin = [0, 0]
        config.system.coupling = [[0, 1]]
        state = _state(config, raw)
        caps = [task.get('capabilities') for task in state.theory
                if task.get('qmmm_role') == 'qm']
        self.assertEqual(len(caps), 2)
        for entry in caps:
            self.assertIn('nac', entry)

    def test_one_center_still_answers_the_coupling(self):
        # multiscale_qmmm_embedding_nac.yaml is the single-center carrier of
        # the same request; if this planned, the rule above is about the
        # declared pair and not about the center count.
        raw, config = _load('multiscale_qmmm_embedding_nac.yaml')
        self.assertIn('nac', config.control.jobtype.split())
        state = _state(config, raw)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'qm', 'energy'])


class TestMulticenterQmmmSearchAxis(unittest.TestCase):
    """What a multi-center search may name, and what bounds it.

    A saddle search runs on the composite surface, so C>1 is searchable; the
    state index it names is not global, because each center answers for its own
    ladder and every center is solved at that index.
    """

    NATOM = 7

    def _state(self, state_i, state_j=None, method='minima',
               ladders=(2, 1), coupling=None):
        import os
        import tempfile
        from pyrai3md.data.build import build_state
        from pyrai3md.kernel.environment import Environment
        from pyrai3md.schema.config import PyRAIConfig
        Environment().initialize()
        os.chdir(tempfile.mkdtemp(prefix='mc_search_axis_'))
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.theory = ['molcas', 'lammps']
        config.control.multiscale_type = 'qmmm'
        config.system.atoms = ['O', 'H', 'H', 'O', 'H', 'H', 'O']
        config.system.coordinates = [
            [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
            [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0],
            [9.0, 9.0, 9.0],
        ]
        config.system.natom = self.NATOM
        # Two centers, ladders of different size on purpose: the bound the
        # search has to respect is the narrower one, and an equal-ladder input
        # would not show which center the refusal was read from.
        config.system.charges = [0, 0]
        config.system.multiplicity = [1, 1]
        config.system.highlevel = [[0, 1, 2], [3, 4, 5]]
        config.system.ci = [[size] for size in ladders]
        config.system.spin = [[0] for _ in ladders]
        config.system.embedding = True
        if coupling is not None:
            config.system.coupling = coupling
        config.optimize.opt_method = method
        config.optimize.opt_state_i = state_i
        if state_j is not None:
            config.optimize.opt_state_j = state_j
        return build_state(config)

    def _optimize(self, state):
        from pyrai3md.workflow.optimize import Optimize

        class _Calculator:
            def __init__(self, state):
                self.state = state

            def get_state(self):
                return self.state

        return Optimize(calculator=_Calculator(state), maxiter=2)

    def test_the_queue_asks_the_curvature_of_the_whole_system(self):
        state = self._state(state_i=0)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm', 'qm', 'energy'])
        self.assertEqual(state.center(0).nstate, 2)
        self.assertEqual(state.center(1).nstate, 1)
        # Reaching here is the point: a search on two centers is built, not
        # refused. The state index it names is bounded below.
        self._optimize(state)

    def test_a_per_center_pair_names_each_center_its_own_states(self):
        # The crossing-search form: a list per center, each bounded by that
        # center's own ladder. Center 0 has two states and asks for the pair
        # 0->1; center 1 has one and stays on it.
        state = self._state((0, 0), state_j=(1, 1), method='meci',
                            ladders=(2, 2), coupling=[[0, 1]])
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm', 'qm', 'energy'])
        optimize = self._optimize(state)
        self.assertEqual(optimize.target_states, {0: [0, 1], 1: [0, 1]})

    def test_a_pair_list_of_the_wrong_length_is_refused(self):
        # One entry per center, or the list cannot name every center's pair.
        state = self._state((0, 0, 0), state_j=(1, 1, 1), method='meci',
                            ladders=(2, 2), coupling=[[0, 1]])
        with self.assertRaises(ValueError) as caught:
            self._optimize(state)
        self.assertIn('names 3 centers', str(caught.exception))

    def test_a_state_the_narrower_ladder_cannot_answer_is_refused(self):
        state = self._state(state_i=1)
        with self.assertRaises(ValueError) as caught:
            self._optimize(state)
        message = str(caught.exception)
        self.assertIn('narrowest ladder', message)
        self.assertIn('1 state', message)
class TestMulticenterQmmmCrossingChannels(unittest.TestCase):
    """The carriers of a coupling request, and the pair a search drives.

    `nac`/`soc` reach the queue through two channels: the spelled jobtype
    (`jobtype: energy gradient nac`) and the QM duty's capabilities, which an
    `optimize` or `md` carrier fills from its own request. Both are open at any
    center count: each center computes its own coupling between its own states
    and the composite keeps the blocks apart. A crossing search (`meci`/`mecp`)
    is open for the same reason - each center names a pair, the assembly builds
    the two composite surfaces (first state of every pair, then the second) and
    the search follows the sum of the per-center gaps.
    """

    NATOM = 7

    def _state(self, method=None, jobtype='optimize', centers=2):
        import os
        import tempfile
        from pyrai3md.data.build import build_state
        from pyrai3md.kernel.environment import Environment
        from pyrai3md.schema.config import PyRAIConfig
        Environment().initialize()
        os.chdir(tempfile.mkdtemp(prefix='mc_crossing_'))
        config = PyRAIConfig()
        config.control.jobtype = jobtype
        config.control.driver = 'main'
        config.control.theory = ['molcas', 'lammps']
        config.control.multiscale_type = 'qmmm'
        config.system.atoms = ['O', 'H', 'H', 'O', 'H', 'H', 'O']
        config.system.coordinates = [
            [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
            [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0],
            [9.0, 9.0, 9.0],
        ]
        config.system.natom = self.NATOM
        config.system.charges = [0, 0]
        config.system.multiplicity = [1, 1]
        if centers == 1:
            config.system.highlevel = [0, 1, 2]
            config.system.ci = [2, 2]
            config.system.spin = [0, 0]
            config.system.charges = 0
            config.system.multiplicity = 1
        else:
            config.system.highlevel = [[0, 1, 2], [3, 4, 5]]
            config.system.ci = [[2], [2]]
            config.system.spin = [[0], [0]]
        config.system.coupling = [[0, 1]]
        config.system.embedding = True
        if method is not None:
            config.optimize.opt_method = method
            config.optimize.opt_state_i = 0
            config.optimize.opt_state_j = 1
        return build_state(config)

    def test_an_optimize_carrier_plans_the_coupling(self):
        # Nothing in this input spells `nac`: the request is the optimizer's
        # own, and it reaches the queue as a capability of the QM duty. It
        # plans now - the pair is per center, not across centers.
        state = self._state(method='meci')
        caps = [task.get('capabilities') for task in state.theory
                if task.get('qmmm_role') == 'qm']
        self.assertTrue(caps)
        for entry in caps:
            self.assertIn('nac', entry)

    def test_mecp_plans_its_pair_per_center_too(self):
        # `mecp` asks for no coupling at all; what it needs is a pair, and each
        # center names its own through `opt_state_i/j`.
        state = self._state(method='mecp')
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm', 'qm', 'energy'])

    def test_the_same_carrier_stays_available_to_one_center(self):
        # The two refusals are about the center count, not about the method:
        # `multiscale_qmmm_meci.yaml` and `_mecp.yaml` are shipped and have
        # been green in the test environment, so a one-center run must still get its
        # coupling duty built.
        state = self._state(method='meci', centers=1)
        roles = [task.get('qmmm_role') for task in state.theory]
        self.assertEqual(roles, ['field', 'qm', 'energy'])
        self.assertIn('nac', state.theory[1].get('capabilities'))
class TestShippedMulticenterQmmmChannels(unittest.TestCase):
    """The channels built on top of one multi-center evaluation.

    A band image and a numerical-Hessian displacement are not new mathematics:
    each is one multi-center evaluation, so what these two shipped inputs have
    to prove is that the queue and the atom set come out per *system*, not per
    center, and that the classical region stays held by the constraint plan
    rather than by a second keyword.
    """

    def test_the_band_input_plans_one_fitted_round_per_image(self):
        raw, config = _load('multicenter_qmmm_mep.yaml')
        self.assertEqual(config.control.jobtype, 'optimize')
        self.assertEqual(config.optimize.opt_method, 'mep')
        # A band image gets the same one-fitted-round evaluation an MD step
        # gets; the batch case (`num_freq`) is the one that keeps it off.
        self.assertIs(bool(config.system.sc_charge), True)
        self.assertEqual(config.system.sc_charge_maxiter, 1)
        state = _state(config, raw)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm_charge', 'qm_charge',
                          'qm', 'qm', 'energy'])

    def test_the_curvature_input_takes_only_the_mobile_block(self):
        from pyrai3md.workflow.hessian import _active_rows
        raw, config = _load('multicenter_qmmm_num_freq.yaml')
        self.assertEqual(config.control.jobtype, 'num_freq')
        state = _state(config, raw)
        # No charge duty: this case measures the displacement batch, and the
        # charges are not its subject - a round per evaluation would multiply
        # the batch by the number of centers without measuring anything new.
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm', 'qm', 'energy'])
        rows = _active_rows(state, list(range(config.system.natom)))
        self.assertEqual(len(rows), 18, rows)
        # The mobile block IS the union of the centers, and it is read against
        # them rather than against a literal atom list: an earlier revision of
        # this input froze `7-60`, which left atoms 1-6 free - the same 18 jobs
        # taken on six atoms that belong to no center, while the case's own
        # header claimed the block spanned both centers (#458). A criterion
        # pinned to the partition cannot let that drift back in.
        union = sorted({index for group in config.system.highlevel
                        for index in group})
        self.assertEqual(sorted({index for index, _ in rows}), union,
                         'the curvature block must be exactly the QM centers')


class TestShippedMulticenterQmmmBoxAndCouplingInputs(unittest.TestCase):
    """The three inputs that move the box or answer per center.

    `npt` and `cell` put the whole composite on a new box each evaluation, and
    the box's conjugate force lives on exactly one duty: `stress` is in
    `planning/task.py::_SYSTEM_ONLY_CAPABILITIES`, so it is carried by the
    whole-system classical task and never by a center's own solve - a three-atom
    cluster cut out of a periodic box has no cell to be stressed. Reading which
    duty holds it is the whole of what these two cases can prove without an
    engine: the barostat and the cell optimizer both take their force from that
    one place. `_nac_sp` is the shipped carrier of the per-center coupling, so
    the reading is that both centers' property solves ask for it, each on the
    pair its own center declares.
    """

    def test_the_npt_input_carries_stress_on_the_classical_duty_only(self):
        raw, config = _load('multicenter_qmmm_npt.yaml')
        self.assertEqual(config.dynamics.ensemble, 'npt')
        state = _state(config, raw)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm_charge', 'qm_charge',
                          'qm', 'qm', 'energy'])
        carriers = [task.get('qmmm_role') for task in state.theory
                    if 'stress' in (task.get('capabilities') or [])]
        self.assertEqual(carriers, ['energy'],
                         'the barostat reads state.active_stress(), which is '
                         'the whole-system task; a per-center duty carrying it '
                         'would put a cell on a cluster')

    def test_the_cell_input_asks_for_the_box_and_its_conjugate(self):
        raw, config = _load('multicenter_qmmm_cell.yaml')
        self.assertEqual(config.control.jobtype, 'optimize')
        self.assertIs(bool(config.optimize.optimize_cell), True)
        self.assertEqual(config.optimize.opt_cell_coupling, 'anisotropic')
        state = _state(config, raw)
        carriers = [task.get('qmmm_role') for task in state.theory
                    if 'stress' in (task.get('capabilities') or [])]
        self.assertEqual(carriers, ['energy'])
        # The box the run declares is the box the state must start on: a cell
        # relaxation with no lattice in the state is refused by name in
        # `optimizer/base.py::_cell_lattice`, so an input that lost its `cell`
        # would never reach the six entries it means to relax.
        self.assertEqual([float(value) for value in config.system.cell],
                         [8.399091, 8.399091, 8.399091, 90.0, 90.0, 90.0])

    def test_the_band_input_names_the_centers_the_endpoints_move(self):
        raw, config = _load('multicenter_qmmm_neb.yaml')
        self.assertEqual(config.optimize.opt_method, 'neb')
        self.assertEqual(config.system.highlevel, [[0, 1, 2], [3, 4, 5]])
        # The centers are the two waters the band's endpoints displace, which
        # is what makes this a test of the per-center bookkeeping along a
        # reaction coordinate rather than a band whose QM terms sit still.
        self.assertEqual([task.get('qmmm_role') for task in
                          _state(config, raw).theory],
                         ['field', 'field', 'qm_charge', 'qm_charge',
                          'qm', 'qm', 'energy'])

    def test_the_restart_pair_is_one_trajectory_split_at_a_checkpoint(self):
        raw, config = _load('multicenter_qmmm_md_restart.yaml')
        _raw2, base = _load('multicenter_qmmm_md_restart_base.yaml')
        # Same box, same centers, same seed, same cadence: the two legs differ
        # only in how the four steps are divided between processes. A drift in
        # any of the other five would make the frame comparison meaningless.
        for label, a, b in (
                ('gl_seed', config.control.gl_seed, base.control.gl_seed),
                ('atoms', config.system.atoms, base.system.atoms),
                ('coordinates', config.system.coordinates,
                 base.system.coordinates),
                ('highlevel', config.system.highlevel,
                 base.system.highlevel),
                ('cell', config.system.cell, base.system.cell)):
            self.assertEqual(a, b,
                             'the two halves disagree on %r' % label)
        self.assertEqual(config.dynamics.checkpoint, 2)
        self.assertEqual(base.dynamics.checkpoint, 2)
        self.assertEqual(config.dynamics.step, 2)
        self.assertEqual(base.dynamics.step, 4)

    def test_the_coupling_input_asks_it_of_each_center(self):
        raw, config = _load('multicenter_qmmm_nac_sp.yaml')
        self.assertEqual(config.control.jobtype, 'energy gradient nac')
        # One pair list per center, not one list for the system: the same
        # nesting `ci`/`spin` use, and what keeps the two blocks apart.
        self.assertEqual(config.system.coupling, [[[0, 1]], [[0, 1]]])
        state = _state(config, raw)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm_charge', 'qm_charge',
                          'qm', 'qm', 'energy'])
        for task in state.theory:
            if task.get('qmmm_role') == 'qm':
                self.assertIn('nac', task.get('capabilities') or [])


class TestMulticenterGlobalElectronicScalars(unittest.TestCase):
    """The consumers that still read one global electronic scalar.

    A multi-center input keeps `multiplicity` as one value per center, so any
    consumer that calls `int()` on it takes a list. The thermodynamic block of
    a numerical Hessian did exactly that on the shipped two-center case: the
    18-job batch and its symmetrization finished, and the analysis died on the
    spin multiplicity of the electronic entropy.
    """

    def test_the_entropy_reads_the_aggregate_not_the_per_center_list(self):
        from pyrai3md.workflow.hessian import _multiplicity
        raw, config = _load('multicenter_qmmm_num_freq.yaml')
        config.system.multiplicity = [1, 3]
        state = _state(config, raw)
        self.assertEqual(list(state.pmd.system.multiplicity), [1, 3])
        # Two unpaired electrons pair off against none: the aggregate is the
        # plan's, and the consumer must not see the list.
        self.assertEqual(_multiplicity(state),
                         state.partition_plan.system_multiplicity)
        self.assertIsInstance(_multiplicity(state), int)

    def test_a_single_center_run_keeps_reading_its_own_scalar(self):
        from pyrai3md.workflow.hessian import _multiplicity
        _raw, config = _load('multiscale_qmmm_embedding_sp.yaml')
        state = _state(config, _raw)
        self.assertEqual(_multiplicity(state),
                         int(config.system.multiplicity))


class TestShippedCarriersOfTheThreeChannels(unittest.TestCase):
    """The three inputs that close the channel matrix (#460).

    A crossing search over each center's own pair (MECP - a MECI's branching
    plane needs a coupling between the two summed surfaces, which no per-center
    block carries), one FSSH ledger per center with the engine's couplings, and
    an excited-state minimization on the per-center ladder. What each input has
    to prove here is that the per-center spelling reaches the planner: these are
    the cases the input layer could not express before #460.
    """

    def test_the_crossing_input_names_one_pair_per_center(self):
        raw, config = _load('multicenter_qmmm_mecp.yaml')
        self.assertEqual(config.optimize.opt_method, 'mecp')
        self.assertEqual(config.optimize.opt_state_i, [0, 0])
        self.assertEqual(config.optimize.opt_state_j, [1, 1])
        # No coupling is declared and none is asked for: this search reads
        # energies and gradients, so a NAC capability here would be a request
        # nothing consumes.
        state = _state(config, raw)
        for task in state.theory:
            if task.get('qmmm_role') == 'qm':
                self.assertNotIn('nac', task.get('capabilities') or [])

    def test_the_hopping_input_gives_each_center_its_own_root(self):
        raw, config = _load('multicenter_qmmm_md_fssh.yaml')
        self.assertEqual(config.dynamics.sfhp, 'fssh')
        self.assertEqual(config.dynamics.nactype, 'nac')
        self.assertEqual(list(config.dynamics.root), [1, 1])
        self.assertEqual(config.system.coupling, [[[0, 1]], [[0, 1]]])
        state = _state(config, raw)
        self.assertEqual([task.get('qmmm_role') for task in state.theory],
                         ['field', 'field', 'qm_charge', 'qm_charge',
                          'qm', 'qm', 'energy'])

    def test_the_excited_state_input_asks_state_one_of_each_center(self):
        raw, config = _load('multicenter_qmmm_ex_opt.yaml')
        self.assertEqual(config.optimize.opt_method, 'minima')
        self.assertEqual(config.optimize.opt_state_i, [1, 1])
        self.assertIsNone(config.optimize.opt_state_j)


class TestSkillMergeSkipsPerCenterSequences(unittest.TestCase):
    """A per-center list must not travel into a skill's config (#460).

    The framework flattens its sections into every skill's config, and a skill
    solves one region with its states delivered through `target_states`. A list
    there is dead data the skill schema rejects, which is how the three #460
    carriers failed to load at first; scalars keep being forwarded because a
    single-center run's root and state index are what a skill has always been
    handed.
    """

    class _Maestro:
        def get_config(self, name):
            return {}

    def _merge(self, **fields):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        for key, value in fields.items():
            if key in ('opt_state_i', 'opt_state_j'):
                setattr(config.optimize, key, value)
            else:
                setattr(config.dynamics, key, value)
        return (_merge_one_skill_config(self._Maestro(), 'molcas_skill',
                                        config) or {})

    def test_a_per_center_pair_is_not_forwarded(self):
        merged = self._merge(opt_state_i=[0, 0], opt_state_j=[1, 1])
        self.assertNotIn('opt_state_i', merged)
        self.assertNotIn('opt_state_j', merged)

    def test_a_per_center_root_is_not_forwarded(self):
        merged = self._merge(root=[1, 1])
        self.assertNotIn('root', merged)

    def test_a_scalar_root_is_still_forwarded(self):
        merged = self._merge(root=2)
        self.assertEqual(merged.get('root'), 2)


class TestChargeDamping(unittest.TestCase):
    """The damping key mixes the fit in; it never changes the stopping test.

    A multi-center fixed point solves each center in the field the other
    centers' charges produce, and that map is not contractive in general: two
    centers polarizing each other can overshoot and oscillate instead of
    settling. A weight below 1 mixes each round's fit into the charges the round
    was solved in. What has to stay untouched is the residual - it is read on the
    raw fit against the previous raw fit - because a damped move is shorter by
    construction and a tolerance read on it would call every iteration
    converged.
    """

    def test_the_undamped_weight_is_the_fit_itself(self):
        from pyrai3md.workflow.multiscale.charges import damp_charges

        self.assertEqual(damp_charges([0.0, 1.0], [0.4, -0.2], 1.0),
                         [0.4, -0.2])

    def test_a_half_weight_lands_half_way(self):
        from pyrai3md.workflow.multiscale.charges import damp_charges

        self.assertEqual(damp_charges([0.0, 0.0], [1.0, -0.5], 0.5),
                         [0.5, -0.25])

    def test_damping_settles_an_iteration_the_raw_map_sends_diverging(self):
        # The map q -> 1 - 2q has the fixed point 1/3 and an iteration matrix
        # of -2: the undamped update oscillates and grows. A half weight turns
        # the iteration matrix into -1 + 0.5*... i.e. 1 + 0.5*(-2 - 1) = -0.5,
        # which converges. Both halves are computed here, so the claim is a
        # property of the arithmetic and not of a fit to one example.
        from pyrai3md.workflow.multiscale.charges import damp_charges

        def step(q, weight):
            return damp_charges([q], [1.0 - 2.0 * q], weight)[0]

        raw = 0.0
        damped = 0.0
        for _ in range(24):
            raw = step(raw, 1.0)
            damped = step(damped, 0.5)
        self.assertGreater(abs(raw), 100.0,
                           'the undamped map was expected to run away: %.3f'
                           % raw)
        self.assertAlmostEqual(damped, 1.0 / 3.0, places=6)

    def test_the_schema_owns_the_range(self):
        from pyrai3md.schema.config import System

        self.assertEqual(System().sc_charge_damping, 1.0)
        self.assertEqual(System(sc_charge_damping=0.5).sc_charge_damping, 0.5)
        for bad in (0.0, 1.5, -0.1):
            with self.assertRaises(Exception):
                System(sc_charge_damping=bad)

    def test_the_classic_carrier_declares_the_key_too(self):
        # The framework's own key table for the classic `&molecule` block is
        # where an alias is declared; a key that only the yaml path knows would
        # be silently dropped by every classic-spelled input.
        from pyrai3md.schema.adapter import MOLECULE_TYPES

        self.assertEqual(MOLECULE_TYPES.get('sc_charge_damping'), 'f')

    def test_the_log_says_the_weight_only_when_it_is_used(self):
        from pyrai3md.utilities.print_results import (
            _format_qmmm_charge_iteration)

        plain = _format_qmmm_charge_iteration({'sc_charge': True, 'rounds_run': 2,
                                        'maxiter': 5, 'tolerance': 1e-3,
                                        'residuals': [0.2, 1e-4],
                                        'stopped_by': 'tolerance'})
        damped = _format_qmmm_charge_iteration({'sc_charge': True, 'rounds_run': 2,
                                         'maxiter': 5, 'tolerance': 1e-3,
                                         'residuals': [0.2, 1e-4],
                                         'stopped_by': 'tolerance',
                                         'damping': 0.5})
        self.assertNotIn('damping', plain)
        self.assertIn('damping', damped)
        self.assertIn('0.5000', damped)


if __name__ == '__main__':
    unittest.main()
