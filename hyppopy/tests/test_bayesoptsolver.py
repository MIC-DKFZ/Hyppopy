# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical Image Computing.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE

import unittest
import matplotlib.pylab as plt

from hyppopy.solvers.BayesOptSolver import *
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.HyppopyProject import HyppopyProject


class BayesOptSolverTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_solver_complete(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [0, 800],
                    "type": "float"
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [-1, 1],
                    "type": "float"
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10],
                    "type": "float"
                }
            },
            "settings": {
                "solver": {"max_iterations": 10},
                "custom": {}
            }}

        project = HyppopyProject(config)
        solver = BayesOptSolver(project)
        vfunc = VirtualFunction()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(0 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))


if __name__ == '__main__':
    unittest.main()
