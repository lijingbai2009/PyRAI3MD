"""
======================================================================
This module provides the Scan workflow for potential energy surface
scans along reaction coordinates. It steps through a series of
geometries and computes energies at each point.

The coordinate generation and interpolation are placeholders.

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .base import Workflow


class Scan(Workflow):
    """Potential energy surface scan workflow.

    Steps through a series of geometries along a reaction coordinate
    and computes energies at each point.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator for energy computations.
    maxiter : int
        Number of scan points.
    scan_coordinates : list of np.ndarray or None
        List of coordinate arrays for each scan point.
        If None, placeholder generates linear interpolation.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask,
        local_executor, executor_spec).
    """

    name = 'scan'
    workflow_kind = 'iterative'

    def __init__(self, calculator=None, maxiter=10, scan_coordinates=None,
                 **kwargs):
        super().__init__(name='scan', calculator=calculator, maxiter=maxiter,
                         **kwargs)
        self.scan_coordinates = scan_coordinates

    def run(self):
        """Execute potential energy scan.

        All info from self.calculator.state.

        Returns
        -------
        PyRAIState
            The state at the last scan point with results recorded.
        """
        state = self.calculator.get_state()

        # Generate scan coordinates if not provided
        coords_list = self.scan_coordinates
        if coords_list is None:
            coords_list = self._generate_coordinates(state)

        energies = []
        for i, coords in enumerate(coords_list):
            # Update the framework system geometry through the
            # calculator state ownership interface
            self.calculator.update({
                'system': {'coordinates': np.asarray(coords).tolist()},
            })
            self.calculator.sync()

            # Compute energy (writes results to pmd)
            result = self.calculator.run()
            energies.append(result['energy'])

            # Write the run-log step block for this scan point
            logger = getattr(state, 'logger', None)
            if logger is not None:
                logger.log_step(i + 1, state.pmd.model_dump())

        # Append scan energies to the results through the
        # calculator state ownership interface
        merged = dict(state.result.to_dict()) if state.result else {}
        merged['scan_energies'] = energies
        self.calculator.update({'results': merged})

        state.runtime.mark_completed()
        return state

    def _generate_coordinates(self, state):
        """Generate scan coordinates (placeholder).

        Parameters
        ----------
        state : PyRAIState
            Initial state with starting geometry.

        Returns
        -------
        list of np.ndarray
            List of coordinate arrays for each scan point.
        """
        # Placeholder: linear interpolation between initial and target
        initial = np.asarray(state.coordinates, dtype=np.float64).copy()
        coords_list = [initial.copy() for _ in range(self.maxiter)]
        return coords_list
