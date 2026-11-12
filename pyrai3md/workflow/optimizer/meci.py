"""
======================================================================
This module provides the MECIOptimizer for geometry optimization towards
a minimum energy conical intersection (MECI), and the crossing machinery
`MECPOptimizer` is built on.

Two surfaces meet where their average energy is as low as it can be and
their difference has closed. The optimizer therefore steps on the gradient
of one objective built out of both: the average gradient, with the
branching plane taken out of it, plus the gradient of the gap's own square.
Inside that plane the average gradient is deliberately not followed, because
it is the two surfaces' difference that has to be answered there.

The plane is spanned by the gradient difference and the derivative coupling,
which is what makes a conical intersection conical: with only the difference
available the search still walks the gap down, but it has no second direction
to keep the two surfaces apart in energy and degenerate in the same place, so
the run prints how many directions it actually had rather than claiming two.

The objective's energy is the average of the two surfaces, so the history,
the BFGS updates and the step rejection all read one number; the gap is
judged separately and its target is `opt_energy`, the same keyword that says
how small an energy change the run counts as none.

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.lattice import (CELL_COMPONENT_SLOTS,
                                  stress_to_cell_force)
from .minima import MinimaOptimizer

# The weight of the gap's square in the objective. The gap enters as
# 0.5 * GAP_MIXING * (E_j - E_i)^2, so its gradient is GAP_MIXING * gap along
# the gap direction; with gap in Hartree and the gradient in Hartree per Bohr,
# a weight of 1.0 per Bohr puts the two halves of the objective on the same
# scale for a gap of the order of one Bohr's worth of motion.
GAP_MIXING = 1.0

# A direction this small is not a direction: a gradient difference that
# vanishes with the gap does so at the crossing itself, and a derivative
# coupling that is numerically zero carries no plane to span.
DIRECTION_FLOOR = 1.0e-12


def _layered_plan(state):
    """Whether this geometry's energy is assembled from more than one layer.

    Read off the plan itself rather than off a name: a single-center composite
    puts the real whole-system task and the model tasks in different layers, and
    it is that difference which decides who can be asked for a stress.
    """
    tasks = getattr(state, 'theory', None) or []
    return len({task.get('layer_id') for task in tasks}) > 1


class CrossingOptimizer(MinimaOptimizer):
    """Common machinery for a search that walks two surfaces together.

    Parameters
    ----------
    Uses the same options as `MinimaOptimizer`; the pair of surfaces comes
    from `active_state` and `active_state_j`.
    """

    # A crossing search climbs the gap term on purpose, and the average energy
    # may rise while it does: the minima rule would reject exactly the steps
    # that are working.
    reject_on_energy_rise = False
    # Whether a derivative coupling is part of the branching plane. Physics
    # answers this per pair: two states of the same spin are coupled by nuclear
    # motion, two of different spin are not, and a search that had no coupling
    # to use says so in its own report rows.
    uses_derivative_coupling = True
    # This is the family that reads the pair: the branching plane is spanned by
    # the two surfaces' energy difference and, when there is one, the coupling.
    uses_state_pair = True
    # The report row the gap criterion is read against, so the log can print
    # the threshold beside it.
    gap_report_key = 'Crossing Gap'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_crossing = {}

    def _cell_stress(self, state):
        """The mean of the two surfaces' conjugates (the base cell force).

        The crossing objective's lattice half has two parts: this mean, and the
        gap term's own derivative, which `_cell_gap_conjugate` adds beside it in
        `_cell_gradient`. This method therefore does not include the gap
        derivative - not because it is unavailable, but because it is a separate
        producer call that needs both state-resolved tensors. For a
        non-degenerate spectrum the CP2K numerical-strain route supplies those
        pages, `_cell_gap_conjugate` adds `V (sigma_j - sigma_i)^T B^-T` scaled
        by `GAP_MIXING * gap`, and `Cell Gap Term` reports its norm (round 20:
        0.01705835 on both carriers). At a degenerate central spectrum that
        route refuses to track the states. The exact intersection is not a
        target of this optimizer: at the degeneracy the roots swap, only the
        gap-space gradient and stress vanish, and the individual surfaces are
        neither smooth nor local minima - they cannot serve as convergence
        criteria - while TDDFT itself carries error at the crossing. The method
        therefore stops at an approximate structure near the crossing, and the
        gap derivative is not claimed there.

        Reading the active state's tensor twice is not a fallback here: it would
        put the box and the atoms on two different objectives inside one
        generalized vector. A run whose backend answers with a single whole-system
        tensor is refused by name and told which shape would serve it.
        """
        if _layered_plan(state):
            # Ruled 2026-10-17: in a single-center composite the two surfaces are
            # the centre's business and the box is the outer layer's. The whole
            # system's stress is then the lattice derivative of the objective as
            # far as this plan can know it - the model layers that would carry a
            # surface-by-surface answer are non-periodic by construction (a
            # centre is a region, and only the real layer holds the cell), so
            # asking them for `sigma_j - sigma_i` would be asking a question the
            # plan has no producer for. The gap row says so by name rather than
            # leaving a NaN to be read either way.
            whole = state.active_stress()
            if whole is None:
                raise ValueError(
                    "optimize_cell with a crossing search over a single-center "
                    "composite needs the whole-system stress of the layer that "
                    "carries the cell, and this evaluation reports none. Drop "
                    "optimize_cell, or make the outermost layer a program that "
                    "answers with a stress")
            self.last_crossing['cell_gap_source'] = (
                'none: box carried by the outer layer')
            self.last_crossing['cell_gap_norm'] = 0.0
            return 0.5 * (np.asarray(whole, dtype=np.float64)
                          + np.asarray(whole, dtype=np.float64))
        first = state.state_stress(self.active_state)
        second = state.state_stress(self.active_state_j)
        if first is None or second is None:
            raise ValueError(
                "optimize_cell with a crossing search needs the stress of both "
                "surfaces and this evaluation carries %s: the cell force is the "
                "conjugate of a lattice entry on the energy being minimized, and "
                "this objective is the average of two surfaces. A backend answers "
                "with a state-resolved (nstate, 3, 3) stress tensor; a single "
                "(3, 3) whole-system reading says what the box is under, not which "
                "surface put it there. Drop optimize_cell, or run the crossing in a "
                "fixed cell" % ('no tensor' if first is None and second is None
                                else 'only one state\'s tensor'))
        return 0.5 * (np.asarray(first, dtype=np.float64)
                      + np.asarray(second, dtype=np.float64))

    def _cell_gap_conjugate(self, state, lattice_bohr):
        """The gap's lattice derivative, in the six relaxed components.

        `d(gap)/dB = V (sigma_j - sigma_i)^T B^-T`, the same conjugate the mean is
        stepped on and the same call the atomic half makes on the gradient
        difference. It is added to the cell force the way the gap's slope is added
        to the atomic one: as `GAP_MIXING * gap` along the unit derivative, so the
        term that vanishes at the crossing is the same size on both halves of one
        step. `Cell Gap Term` reports its norm, which is the number a reader needs
        to tell a closed gap from a missing producer.
        """
        crossing = self.last_crossing
        if str(crossing.get('cell_gap_source') or '').startswith('none'):
            # A composite's box is stepped on the outer layer's stress, so this
            # term is zero by the shape of the plan, not by a missing producer.
            return np.zeros(len(CELL_COMPONENT_SLOTS))
        gap = crossing.get('gap')
        if gap is None:
            return None
        if 'cell_gap_source' not in crossing:
            crossing['cell_gap_source'] = 'state-resolved stress'
        if self.active_state == self.active_state_j:
            # One surface named, so the difference of its tensor with itself would
            # come back as an exact zero: the honest reading is that this run has
            # no gap derivative to report, which is what the NaN row says.
            return None
        first = state.state_stress(self.active_state)
        second = state.state_stress(self.active_state_j)
        if first is None or second is None:
            return None
        difference = (np.asarray(second, dtype=np.float64)
                      - np.asarray(first, dtype=np.float64))
        row = stress_to_cell_force(difference, lattice_bohr).reshape(-1)
        row = row[list(CELL_COMPONENT_SLOTS)]
        norm = float(np.linalg.norm(row))
        self.last_crossing['cell_gap_norm'] = norm
        if norm < DIRECTION_FLOOR:
            return np.zeros_like(row)
        return (GAP_MIXING * float(gap)) * row / norm

    # ---- Objective ----

    def _objective(self, state, result):
        """The average surface and the gradient of the crossing objective.

        Returns
        -------
        tuple
            (mean energy, effective gradient in Hartree per Bohr). The
            effective gradient is the average gradient with the branching plane
            removed, plus the gap's own slope. At the crossing the second term
            vanishes and the first is what a stationary point of the crossing
            condition asks to be zero.
        """
        energy_i, energy_j, gradient_i, gradient_j = self._extract_pair(state)
        gap = energy_j - energy_i
        # The two surfaces' average, on the same objective the gradient is
        # stepped on: under a pressure that is `+ pV` as well (`_enthalpy`), so
        # the rejection and the criteria read the function the box is moving in.
        mean = self._enthalpy(state, 0.5 * (energy_i + energy_j))
        mean_gradient = 0.5 * (gradient_i + gradient_j)
        difference = gradient_j - gradient_i

        directions = [('gap', difference)]
        coupling = (self._derivative_coupling(state)
                    if self.uses_derivative_coupling else None)
        if coupling is not None:
            directions.append(('coupling', coupling))

        gradient, kept = self._project_out(mean_gradient, directions)
        gap_row = next((row for name, row, _weight in kept if name == 'gap'),
                       None)
        if gap_row is not None:
            gradient = gradient + (
                GAP_MIXING * gap) * gap_row / float(np.linalg.norm(gap_row))
        self.last_crossing = {
            'gap': float(gap),
            'mean_energy': float(mean),
            'directions': [name for name, _row, _weight in kept],
            'gap_norm': float(np.linalg.norm(difference)),
            'coupling_norm': (float(np.linalg.norm(coupling))
                              if coupling is not None else None),
        }
        return mean, gradient

    @staticmethod
    def _project_out(gradient, directions):
        """Remove the branching-plane part of the average gradient.

        Gram-Schmidt over the declared directions, so a pair that has become
        parallel (the usual thing as the gap closes) contributes one direction
        and is reported as one.

        Returns
        -------
        tuple
            (remaining gradient, [(name, orthogonal row, squared norm)] for the
            directions that were actually kept).
        """
        vector = np.asarray(gradient, dtype=np.float64)
        kept = []
        for name, candidate in directions:
            row = np.asarray(candidate, dtype=np.float64)
            if float(np.linalg.norm(row)) < DIRECTION_FLOOR:
                continue
            for _previous_name, previous, weight in kept:
                row = row - float(np.dot(row, previous)) / weight * previous
            norm = float(np.dot(row, row))
            if norm < DIRECTION_FLOOR * DIRECTION_FLOOR:
                continue
            vector = vector - float(np.dot(vector, row)) / norm * row
            kept.append((name, row, norm))
        return vector, kept

    # ---- Criterion ----

    def status_method_columns(self):
        """The crossing's gap and plane readings, for `opt_status.txt`."""
        columns = list(super().status_method_columns())
        columns += ['cross_gap', 'branch_dirs', 'gap_norm', 'coupling_norm']
        return tuple(columns)

    def status_method_values(self, report):
        """Values matching the crossing columns, in the same order."""
        values = list(super().status_method_values(report))
        crossing = self.last_crossing or {}
        crossing_gap = report.get(self.gap_report_key)
        values.append(float('nan') if crossing_gap is None
                      else float(crossing_gap))
        directions = report.get('Branching Directions')
        values.append(float('nan') if directions is None
                      else float(directions))
        for key in ('gap_norm', 'coupling_norm'):
            value = crossing.get(key)
            values.append(float('nan') if value is None else float(value))
        return tuple(values)

    def method_details(self, report):
        """Log-only crossing readings the convergence table does not carry."""
        crossing = self.last_crossing or {}
        details = {}
        for label, key in (('Gap Norm', 'gap_norm'),
                           ('Coupling Norm', 'coupling_norm'),
                           ('Mean Energy', 'mean_energy')):
            value = crossing.get(key)
            details[label] = None if value is None else float(value)
        return details

    def _extra_criteria(self, report):
        """Close the gap, and say what the plane was made of.

        The gap's target is `opt_energy`: the same keyword that names how small
        an energy change the run counts as none also names how small an energy
        *difference* it counts as a crossing, and one keyword answers one
        question.
        """
        crossing = self.last_crossing
        gap = crossing.get('gap')
        report[self.gap_report_key] = (abs(float(gap))
                                       if gap is not None else 0.0)
        report['Branching Directions'] = float(
            len(crossing.get('directions') or ()))
        if self.cell_variables:
            # What the box was stepped on, in two numbers. The mean of the two
            # surfaces' stresses drives the step, and the gap's own lattice
            # derivative now rides beside it: this row reports that derivative's
            # norm, so a run that closed the gap can be told from a run whose
            # backend never answered with a state-resolved tensor (the second is
            # reported as missing, not as a zero).
            norm = crossing.get('cell_gap_norm')
            report['Cell Gap Term'] = (float(norm) if norm is not None
                                       else float('nan'))
            report['Cell Gap Term Source'] = crossing.get(
                'cell_gap_source', 'not measured this step')
        if gap is None:
            return False
        return abs(float(gap)) < self.energy_threshold


class MECIOptimizer(CrossingOptimizer):
    """Optimizer for minimum energy conical intersection search.

    Same-spin pair: the branching plane is spanned by the gradient difference
    and the derivative coupling, and both are used when the backend returns
    them.
    """

    uses_derivative_coupling = True
