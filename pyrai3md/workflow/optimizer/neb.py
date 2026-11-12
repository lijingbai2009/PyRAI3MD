"""
======================================================================
This module provides the NEBOptimizer for nudged elastic band (NEB)
calculations: the minimum energy path between two known structures, found
by relaxing a chain of images that connects them.

The band is a set of geometries rather than one, which is why it is driven
by the Path workflow (`workflow/path.py`) and not by `Optimize.run`: every
round evaluates all the images and moves all of them. The forces are the
literature's, and the three parts of it are:

    the tangent      Henkelman & Jonsson, J. Chem. Phys. 113, 9978 (2000).
                     The tangent at an image is the energy-weighted average
                     of the two segment directions, so a band whose images
                     are unevenly spaced still estimates the path direction
                     from the images around it rather than from one side.
    the expansion    A parallel spring force `k (|R_{i+1} - R_i| -
                     |R_i - R_{i-1}|) tau` keeps the images apart, and the
                     physical force is used perpendicular to the tangent
                     only, so the band does not slide downhill off the path
                     it is trying to describe. The endpoints are held.
    the climb        Henkelman, Uberuaga & Jonsson, J. Chem. Phys. 113, 9901
                     (2000). With `neb_climb`, once the band is relaxed
                     enough the highest image has its parallel force
                     inverted, which drives it onto the saddle instead of
                     letting the spring hold it short of it.

The relaxation integrator is FIRE (Bitzek et al., Phys. Rev. Lett. 97,
170201 (2006)), which is the reference's choice for the same reason it is
the right one here: the nudged force is not the gradient of anything, so a
quasi-Newton step built on a curvature model would be fitting a model to a
field that has none.

A variable-cell band relaxes the atoms and the six relaxed lattice entries
together, the way the single-structure searches do: the lattice half of the
force is the stress conjugate (`optimizer/base.py::_cell_gradient` read on
each image's own evaluation), the objective under an applied pressure is the
enthalpy, and the two halves are judged by the two threshold pairs the
`optimize_cell` keywords already name.

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic Institute
======================================================================
"""

import numpy as np

from .pathbase import BandPathOptimizer
from ...utilities.lattice import (CELL_COMPONENT_SLOTS,
                                  min_image_displacement)


class NEBOptimizer(BandPathOptimizer):
    """Nudged elastic band between two structures.

    Parameters
    ----------
    neb_spring : float
        The spring constant, in Hartree per Bohr^2.
    neb_climb : bool
        Whether the highest image becomes a climbing image once the band is
        relaxed enough (`optimize.neb_climb`).
    neb_climb_threshold : float
        The perpendicular force max that arms the climbing image.
    neb_step : float
        The FIRE integrator's initial timestep, in atomic units.
    """

    is_implemented = True
    path_kind = 'band'

    def __init__(self, neb_spring=0.01, neb_climb=False,
                 neb_climb_threshold=1.0e-2, neb_step=0.1, **kwargs):
        super().__init__(neb_spring=neb_spring, neb_climb=neb_climb,
                         neb_climb_threshold=neb_climb_threshold,
                         neb_step=neb_step, **kwargs)
        self.last_steps = None
        self._previous_highest = None
        self._previous_barrier = None

    # ---- The band force ----

    def _tangent(self, index):
        """The improved tangent at one interior image (Henkelman-Jonsson).

        The three cases are the energy ordering of the three images the tangent
        is built from: a monotone climb uses the uphill segment alone, a
        monotone descent the downhill one, and an image between two higher (or
        two lower) neighbours uses both, weighted by how much the farther one
        differs. Equal energies fall back to the geometric secant, which is what
        keeps a flat region from producing a zero direction.
        """
        energies = [self.energies[index - 1], self.energies[index],
                    self.energies[index + 1]]
        forward = self._delta(index + 1, index)
        backward = self._delta(index, index - 1)
        above = energies[2] - energies[1]
        below = energies[1] - energies[0]
        if above > below > 0.0:
            tangent = forward + backward * (above / below)
        elif above < below < 0.0:
            tangent = forward + backward * (below / above)
        else:
            largest, smallest = max(abs(above), abs(below)), min(abs(above),
                                                                abs(below))
            if energies[2] > energies[0]:
                tangent = forward * largest + backward * smallest
            else:
                tangent = forward * smallest + backward * largest
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            # Equal neighbouring energies make the weighted expression vanish;
            # the secant across the two outer images still defines a direction,
            # and failing that a segment is better than no direction at all.
            tangent = self._delta(index + 1, index - 1)
            length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            tangent = (forward if np.linalg.norm(forward)
                       >= np.linalg.norm(backward) else backward)
            length = float(np.linalg.norm(tangent))
        return tangent / length if length > 0.0 else tangent

    def _delta(self, first, second):
        """`R_first - R_second`, its atomic half in the nearest periodic image."""
        delta = self.images[first] - self.images[second]
        if self.lattices is None or not any(self._pbc):
            return delta
        moved = delta.copy()
        moved[:self._atomic_size] = min_image_displacement(
            delta[:self._atomic_size].reshape(-1, 3),
            self.lattice_of(first)).reshape(-1)
        return moved

    def _band_forces(self, climbing=None):
        """The nudged force on every image, and the perpendicular maximum.

        Returns
        -------
        tuple
            `(forces, perpendicular_max)`: the relaxed images carry the spring
            force parallel and the physical force perpendicular; the endpoints
            carry nothing (they are the two structures the run was given), the
            climbing image carries the physical force with its parallel part
            inverted and no spring, and every other image the parallel spring
            plus the perpendicular physical force.
        """
        forces = np.zeros_like(self.images)
        perpendicular_max = 0.0
        for index in self.interior:
            tangent = self._tangent(index)
            gradient = self.gradients[index]
            parallel = float(np.dot(gradient, tangent)) * tangent
            perpendicular = gradient - parallel
            perpendicular_max = max(perpendicular_max,
                                    float(np.max(np.abs(perpendicular))))
            if climbing is not None and index == climbing:
                forces[index] = -gradient + 2.0 * parallel
                continue
            forward = float(np.linalg.norm(self._delta(index + 1, index)))
            backward = float(np.linalg.norm(self._delta(index, index - 1)))
            forces[index] = (-perpendicular
                             + self.spring * (forward - backward) * tangent)
        return forces, perpendicular_max

    # ---- The round ----

    def _band_reading(self):
        """Compute the forces and decide whether the band has settled.

        The climbing image is armed once, at the first round whose
        perpendicular force max falls below `neb_climb_threshold`, and stays
        armed: a band that is being driven towards a saddle and then handed back
        to the springs would take the image off the saddle it just found. Before
        that round the band is a plain nudged elastic band, which is what the
        reference calls its relax-then-climb schedule.
        """
        if self.climb and not self.latched:
            _forces, perpendicular_max = self._band_forces()
            if perpendicular_max < self.climb_threshold:
                self.latched = True
                self.climbing_index = min(
                    self.interior, key=lambda index: -self.energies[index])
                # A latched climber inherits no momentum from the relaxation
                # phase: its parallel force has just flipped sign, and a
                # velocity built on the old one would carry it past the saddle.
                for index in self.interior:
                    self.velocity[index] = np.zeros_like(self.velocity[index])
                self.dt = self.dt0
                self.alpha = 0.1
                self.n_up = 0
        climbing = self.climbing_index if self.latched else None
        forces, perpendicular_max = self._band_forces(climbing)
        self.forces = forces

        # The readings the thresholds are judged on. The climbing image is
        # judged on its own force, which vanishes only at the saddle: a
        # perpendicular-only reading there is zero on a symmetric path while the
        # parallel gradient still pushes the image off it.
        interior = self.interior
        atomic = slice(0, self._atomic_size)
        judged = [forces[index] for index in interior]
        atomic_forces = np.concatenate([force[atomic] for force in judged])
        force_max = float(np.max(np.abs(atomic_forces))) if atomic_forces.size else 0.0
        force_rms = (float(np.sqrt(np.mean(atomic_forces ** 2)))
                     if atomic_forces.size else 0.0)
        highest = max(interior, key=lambda index: self.energies[index])
        highest_energy = float(self.energies[highest])
        endpoints = (float(self.energies[0]), float(self.energies[-1]))
        barrier = highest_energy - max(endpoints)

        row = {
            'energy': highest_energy,
            'barrier': barrier,
            'shift': (0.0 if self._previous_barrier is None
                      else barrier - self._previous_barrier),
            'max_d': 0.0,
            'rms_d': 0.0,
            'max_g': force_max,
            'rms_g': force_rms,
            'image': highest,
            'climb': 1 if self.latched else 0,
            'converged': False,
        }
        if self.last_steps is not None:
            shifted = (self.last_steps[:self._atomic_size]
                       * self.disp_to_angstrom)
            row['max_d'] = (float(np.max(np.abs(shifted)))
                            if shifted.size else 0.0)
            row['rms_d'] = (float(np.sqrt(np.mean(shifted ** 2)))
                            if shifted.size else 0.0)
        settled = (force_rms < self.gradient_threshold
                   and force_max < self.gradient_max_threshold)
        if self.cell_variables:
            cell = np.concatenate([force[self._atomic_size:] for force in judged])
            cell_max = float(np.max(np.abs(cell))) if cell.size else 0.0
            cell_rms = float(np.sqrt(np.mean(cell ** 2))) if cell.size else 0.0
            strain = self._strain()
            strain_max = float(np.max(np.abs(strain))) if strain.size else 0.0
            strain_rms = (float(np.sqrt(np.mean(strain ** 2)))
                          if strain.size else 0.0)
            row.update({
                'cell_gmax': cell_max,
                'cell_grms': cell_rms,
                'cell_dmax': strain_max,
                'cell_drms': strain_rms,
                'volume': self._volume(highest),
            })
            # The box is judged on the pressure its force stands for and on the
            # strain its step took, the two readings a variable-cell relaxation
            # takes (`optimizer/base.py::convergence`): a band whose atoms are
            # quiet while the box is still expanding has not settled.
            settled = settled and cell_rms < self.cell_stress_rms_threshold \
                and cell_max < self.cell_stress_max_threshold \
                and strain_rms < self.cell_strain_rms_threshold \
                and strain_max < self.cell_strain_max_threshold
        row['converged'] = bool(settled)
        self._previous_barrier = barrier
        self._previous_highest = highest_energy
        return row

    def _strain(self):
        """The cell half of the last FIRE step, as a strain.

        The step of one lattice entry over the length of its own row, which is
        the dimensionless reading `optimize.opt_cerms` and `opt_cemax` are
        written in (`optimizer/base.py::convergence`), taken on each image's own
        box because a variable-cell band's images do not share one.
        """
        if self.last_steps is None or self.lattices is None:
            return np.zeros(0)
        values = []
        for index in self.interior:
            step = self.last_steps[index][self._atomic_size:]
            if not step.size:
                continue
            rows = np.linalg.norm(self.lattice_of(index), axis=1)
            lengths = np.array([rows[slot // 3]
                                for slot in CELL_COMPONENT_SLOTS])
            values.append(np.abs(step) / np.where(lengths > 0.0, lengths, 1.0))
        return np.concatenate(values) if values else np.zeros(0)

    def _volume(self, index):
        """The box one image stands for, in cubic Angstrom."""
        if self.lattices is None:
            return 0.0
        return float(abs(np.linalg.det(self.lattice_of(index)))
                     * self.disp_to_angstrom ** 3)

    def _fire_step(self, reading):
        """One FIRE update, recording the per-image step it took."""
        before = self.images.copy()
        super()._fire_step(reading)
        self.last_steps = self.images - before

    def path_points(self):
        """The band's images, in order, as profile rows."""
        points = super().path_points()
        if self.lattices is not None:
            for index, point in enumerate(points):
                point['volume'] = self._volume(index)
        return points

    def results(self):
        block = super().results()
        block.update({
            'path_climbing_image': (self.climbing_index
                                    if self.latched else None),
            'path_roles': list(self.roles or []),
        })
        return block


__all__ = ['NEBOptimizer']