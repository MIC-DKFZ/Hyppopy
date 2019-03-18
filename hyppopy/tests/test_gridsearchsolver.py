# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical and Biological Informatics.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE.txt or http://www.mitk.org for details.
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import os
import unittest
import numpy as np

from ..solver.GridsearchSolver import *
from ..globals import TESTDATA_DIR


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


if __name__ == '__main__':
    unittest.main()
