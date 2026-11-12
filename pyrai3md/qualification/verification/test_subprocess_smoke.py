"""
======================================================================
End-to-end smoke test of the subprocess_run single-task runner
with a minimal pyscf calculation.

The test builds a real PyRAIState through the unchanged input
pipeline (Interface -> Schema -> Planning -> build_state), creates
one dispatch task through the Dispatcher unit materialization
contract, and executes it as an independent child runtime:

    runtime directory creation -> .rt serialization ->
    subprocess 'python3 -m pyrai3md {name}.rt' -> RuntimeLoader ->
    standard execute() path -> result.json collection

Skipped automatically when pyscf is not installed.

Aug 05 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

INPUT_TEXT = """
&CONTROL
title dispatch_smoke
jobtype energy gradient
theory pyscf

&MOLECULE
ci 1

&PYSCF
method hf
basis sto-3g
"""

COORDINATES = """3
water molecule
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
"""


def _pyscf_available():
    try:
        import pyscf  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_pyscf_available(), "pyscf is not installed")
class TestSubprocessDispatchSmoke(unittest.TestCase):
    """Dispatch one minimal pyscf gradient task through a subprocess."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrai3md_smoke_')
        self._cwd = os.getcwd()
        os.chdir(self.tmpdir)

        with open('smoke.inp', 'w') as handler:
            handler.write(INPUT_TEXT)
        with open('smoke.xyz', 'w') as handler:
            handler.write(COORDINATES)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_state(self):
        from pyrai3md.data.build import build_state
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        raw_config = parse_input('smoke.inp')
        config = build_config(raw_config)
        config = normalize(config)
        config = validate(config)
        return build_state(config, raw_config=raw_config)

    def test_dispatch_child_runtime_round_trip(self):
        from pyrai3md.workflow.calculator import Calculator
        from pyrai3md.workflow.dispatch.subprocess_run import subprocess_run
        from pyrai3md.workflow.hessian import NumericalHessian

        state = self._build_state()
        self.assertEqual(sum(1 for _ in state.iter_tasks()), 1)

        calculator = Calculator(state)
        workflow = NumericalHessian(calculator=calculator)

        # One displaced geometry as the batch task
        displaced = [list(row) for row in state.coordinates]
        displaced[0][0] += workflow.step_size
        dispatch_task = workflow.materialize_unit(
            {'coordinates': displaced}, 0,
            child_jobtype=workflow.child_jobtype)

        # subprocess_run is a single-task function taking a unit dict
        result = subprocess_run(dispatch_task)

        # Runtime directory layout is managed by RuntimeDirectoryManager
        # inside subprocess_run; verify the result structure instead.
        self.assertIn('energy', result.keys())
        self.assertIn('gradient', result.keys())
        gradient = np.asarray(result['gradient'], dtype=np.float64)
        self.assertEqual(gradient.shape[-1], 3)

        # Parent state untouched by the remote execution
        self.assertEqual(state.pmd.results, {})


if __name__ == '__main__':
    unittest.main()
