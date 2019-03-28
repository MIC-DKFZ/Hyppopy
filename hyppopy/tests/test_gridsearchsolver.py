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

from hyppopy.solver.GridsearchSolver import *
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.HyppopyProject import HyppopyProject


class GridsearchTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_uniform_axis_sample(self):
        drange = [0, 10]
        N = 11
        data = get_uniform_axis_sample(drange[0], drange[1], N, "float")
        for i in range(11):
            self.assertEqual(float(i), data[i])

        drange = [-10, 10]
        N = 21
        data = get_uniform_axis_sample(drange[0], drange[1], N, "int")
        self.assertEqual(data[0], -10)
        self.assertEqual(data[20], 10)
        self.assertEqual(data[10], 0)

    def test_get_norm_cdf(self):
        res = [0, 0.27337265, 0.4331928, 0.48777553, 0.4986501, 0.5013499, 0.51222447, 0.5668072, 0.72662735, 1]
        f = get_norm_cdf(10)
        for n, v in enumerate(res):
            self.assertAlmostEqual(v, f[n])

        res = [0.0, 0.27337264762313174, 0.4331927987311419, 0.48777552734495533, 0.4986501019683699, 0.5,
               0.5013498980316301, 0.5122244726550447, 0.5668072012688581, 0.7266273523768683, 1.0]
        f = get_norm_cdf(11)
        for n, v in enumerate(res):
            self.assertAlmostEqual(v, f[n])

    def test_get_gaussian_axis_sampling(self):
        res = [-5.0,
               -2.2662735237686826,
               -0.6680720126885813,
               -0.12224472655044671,
               -0.013498980316301257,
               0.013498980316301257,
               0.12224472655044671,
               0.6680720126885813,
               2.2662735237686826,
               5.0]

        bounds = (-5, 5)
        N = 10
        data = get_gaussian_axis_sample(bounds[0], bounds[1], N, "float")
        for n in range(N):
            self.assertAlmostEqual(res[n], data[n])

        res = [-5.0,
               -2.2662735237686826,
               -0.6680720126885813,
               -0.12224472655044671,
               -0.013498980316301257,
               0.0,
               0.013498980316301257,
               0.12224472655044671,
               0.6680720126885813,
               2.2662735237686826,
               5.0]

        bounds = (-5, 5)
        N = 11
        data = get_gaussian_axis_sample(bounds[0], bounds[1], N, "float")
        for n in range(N):
            self.assertAlmostEqual(res[n], data[n])

    def test_get_logarithmic_axis_sample(self):
        res = [0.0010000000000000002,
               0.0035938136638046297,
               0.012915496650148841,
               0.046415888336127795,
               0.1668100537200059,
               0.5994842503189414,
               2.154434690031884,
               7.7426368268112675,
               27.825594022071247,
               100.00000000000004]
        bounds = (0.001, 1e2)
        N = 10
        data = get_logarithmic_axis_sample(bounds[0], bounds[1], N, "float")
        for n in range(N):
            self.assertAlmostEqual(res[n], data[n])

        res = [0.0010000000000000002,
               0.003162277660168382,
               0.010000000000000004,
               0.03162277660168381,
               0.10000000000000006,
               0.31622776601683833,
               1.0000000000000009,
               3.1622776601683813,
               10.00000000000001,
               31.622776601683846,
               100.00000000000004]
        bounds = (0.001, 1e2)
        N = 11
        data = get_logarithmic_axis_sample(bounds[0], bounds[1], N, "float")
        for n in range(N):
            self.assertAlmostEqual(res[n], data[n])

    def test_solver(self):
        config = {
            "hyperparameter": {
                "value 1": {
                    "domain": "uniform",
                    "data": [0, 20, 11],
                    "type": "int"
                },
                "value 2": {
                    "domain": "normal",
                    "data": [0, 20.0, 11],
                    "type": "float"
                },
                "value 3": {
                    "domain": "loguniform",
                    "data": [1, 10000, 11],
                    "type": "float"
                },
                "categorical": {
                    "domain": "categorical",
                    "data": ["a", "b"],
                    "type": "str"
                }
            },
            "settings": {
                "solver": {},
                "custom": {}
            }}
        res_labels = ['value 1', 'value 2', 'value 3', 'categorical']
        res_values = [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      [0.0, 5.467452952462635, 8.663855974622837, 9.755510546899107, 9.973002039367397, 10.0,
                       10.026997960632603, 10.244489453100893, 11.336144025377163, 14.532547047537365, 20.0],
                      [1.0, 2.51188643150958, 6.309573444801933, 15.848931924611136, 39.810717055349734,
                        100.00000000000004, 251.18864315095806, 630.9573444801938, 1584.8931924611143,
                        3981.071705534977, 10000.00000000001],
                      ['a', 'b']
                      ]
        solver = GridsearchSolver(config)
        searchspace = solver.convert_searchspace(config["hyperparameter"])
        for n in range(len(res_labels)):
            self.assertEqual(res_labels[n], searchspace[0][n])
        for i in range(3):
            self.assertAlmostEqual(res_values[i], searchspace[1][i])
        self.assertEqual(res_values[3], searchspace[1][3])

    def test_solver_complete(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [300, 800, 11],
                    "type": "float"
                },
                "axis_01": {
                    "domain": "normal",
                    "data": [-1, 1, 11],
                    "type": "float"
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10, 11],
                    "type": "float"
                }
            },
            "settings": {
                "solver": {},
                "custom": {}
            }}

        project = HyppopyProject(config)
        solver = GridsearchSolver(project)
        vfunc = VirtualFunction()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertAlmostEqual(best['axis_00'], 583.40, places=1)
        self.assertAlmostEqual(best['axis_01'], 0.45, places=1)
        self.assertAlmostEqual(best['axis_02'], 5.0, places=1)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))


if __name__ == '__main__':
    unittest.main()
