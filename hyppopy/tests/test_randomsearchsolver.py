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

from hyppopy.solver.RandomsearchSolver import *
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.HyppopyProject import HyppopyProject


class RandomsearchTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_draw_uniform_sample(self):
        param = {"data": [0, 1, 10],
                 "type": "float"}
        values = []
        for i in range(10000):
            values.append(draw_uniform_sample(param))
            self.assertTrue(0 <= values[-1] <= 1)
            self.assertTrue(isinstance(values[-1], float))
        hist = plt.hist(values, bins=10, normed=True)
        std = np.std(hist[0])
        mean = np.mean(hist[0])
        self.assertTrue(std < 0.05)
        self.assertTrue(0.9 < mean < 1.1)

        param = {"data": [0, 10, 11],
                 "type": "int"}
        values = []
        for i in range(10000):
            values.append(draw_uniform_sample(param))
            self.assertTrue(0 <= values[-1] <= 10)
            self.assertTrue(isinstance(values[-1], int))
        hist = plt.hist(values, bins=11, normed=True)
        std = np.std(hist[0])
        mean = np.mean(hist[0])
        self.assertTrue(std < 0.05)
        self.assertTrue(0.09 < mean < 0.11)

    def test_draw_normal_sample(self):
        param = {"data": [0, 10, 11],
                 "type": "int"}
        values = []
        for i in range(10000):
            values.append(draw_normal_sample(param))
            self.assertTrue(0 <= values[-1] <= 10)
            self.assertTrue(isinstance(values[-1], int))
        hist = plt.hist(values, bins=11, normed=True)
        for i in range(1, 5):
            self.assertTrue(hist[0][i-1]-hist[0][i] < 0)
        for i in range(5, 10):
            self.assertTrue(hist[0][i] - hist[0][i+1] > 0)

    def test_draw_loguniform_sample(self):
        param = {"data": [1, 1000, 11],
                 "type": "float"}
        values = []
        for i in range(10000):
            values.append(draw_loguniform_sample(param))
            self.assertTrue(1 <= values[-1] <= 1000)
            self.assertTrue(isinstance(values[-1], float))
        hist = plt.hist(values, bins=11, normed=True)
        for i in range(4):
            self.assertTrue(hist[0][i] > hist[0][i+1])
            self.assertTrue((hist[0][i] - hist[0][i+1]) > 0)

    def test_draw_categorical_sample(self):
        param = {"data": [1, 2, 3],
                 "type": int}
        values = []
        for i in range(10000):
            values.append(draw_categorical_sample(param))
            self.assertTrue(values[-1] == 1 or values[-1] == 2 or values[-1] == 3)
            self.assertTrue(isinstance(values[-1], int))
        hist = plt.hist(values, bins=3, normed=True)
        for i in range(3):
            self.assertTrue(0.45 < hist[0][i] < 0.55)

    def test_solver_complete(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [300, 800],
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
                "solver": {"max_iterations": 100},
                "custom": {}
            }}

        project = HyppopyProject(config)
        solver = RandomsearchSolver(project)
        vfunc = VirtualFunction()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(300 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))


if __name__ == '__main__':
    unittest.main()
