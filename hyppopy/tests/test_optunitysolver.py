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
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import unittest
import matplotlib.pylab as plt

from hyppopy.solver.OptunitySolver import *
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.HyppopyProject import HyppopyProject


class OptunitySolverTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_solver_complete(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [300, 800],
                    "type": "float"
                },
                "axis_01": {
                    "domain": "normal",
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
                "solver": {"max_iterations": 800},
                "custom": {}
            }}

        project = HyppopyProject(config)
        solver = OptunitySolver(project)
        vfunc = VirtualFunction()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(570 < best['axis_00'] < 590)
        self.assertTrue(0.1 < best['axis_01'] < 0.8)
        self.assertTrue(4.5 < best['axis_02'] < 6)


if __name__ == '__main__':
    unittest.main()
