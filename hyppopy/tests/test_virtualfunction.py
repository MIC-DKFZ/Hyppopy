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

from ..VirtualFunction import VirtualFunction
from ..globals import TESTDATA_DIR


class VirtualFunctionTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_imagereading(self):
        vfunc = VirtualFunction()
        vfunc.load_images(os.path.join(TESTDATA_DIR, 'functionsimulator'))
        self.assertTrue(isinstance(vfunc.data, np.ndarray))
        self.assertEqual(vfunc.data.shape[0], 5)
        self.assertEqual(vfunc.data.shape[1], 512)
        gt = [0.83984375*5, 0.44140625*20-10, 0.25390625*20, 0.81640625*8-10, 0.67578125*2+2]
        for i in range(5):
            self.assertAlmostEqual(vfunc.data[i][0], gt[i])
        gt = [[0, 1], [-10, 10], [0, 20], [-30, 5], [5, 10]]
        for i in range(5):
            self.assertEqual(vfunc.axis[i][0], gt[i][0])
            self.assertEqual(vfunc.axis[i][1], gt[i][1])

    def test_data_adding(self):
        gt = [[-10, 10], [-30, 5]]
        vfunc = VirtualFunction()
        dim0 = np.arange(0, 1.1, 0.1)
        dim1 = np.arange(1.0, -0.1, -0.1)
        vfunc.add_dimension(dim0, gt[0])
        self.assertEqual(len(vfunc.data.shape), 2)
        self.assertEqual(vfunc.data.shape[0], 1)
        self.assertEqual(vfunc.data.shape[1], 11)
        vfunc.add_dimension(dim1, gt[1])
        self.assertEqual(vfunc.data.shape[0], 2)
        self.assertEqual(vfunc.data.shape[1], 11)
        for n in range(11):
            self.assertAlmostEqual(dim0[n], vfunc.data[0, n])
            self.assertAlmostEqual(dim1[n], vfunc.data[1, n])
        for i in range(2):
            self.assertEqual(vfunc.axis[i][0], gt[i][0])
            self.assertEqual(vfunc.axis[i][1], gt[i][1])

    def test_sampling(self):
        vfunc = VirtualFunction()
        vfunc.load_images(os.path.join(TESTDATA_DIR, 'functionsimulator'))
        ranges = [[0, 1], [-10, 10], [0, 20], [-30, 5], [5, 10]]
        x_ranges = []
        for r in ranges:
            dr = (r[1]-r[0])/512.0
            x_ranges.append(np.arange(r[0], r[1], dr))
        data = [[], [], [], [], []]
        for n in range(x_ranges[0].shape[0]):
            x = [x_ranges[0][n], x_ranges[1][n], x_ranges[2][n], x_ranges[3][n], x_ranges[4][n]]
            f = vfunc(*x)
            for i in range(5):
                data[i].append(f[i])

        sum = 0
        for i in range(512):
            for n in range(5):
                sum += vfunc.data[n][i]-data[n][i]
        self.assertTrue(sum < 18)

    def test_minima(self):
        vfunc = VirtualFunction()
        vfunc.load_images(os.path.join(TESTDATA_DIR, 'functionsimulator'))
        minima = vfunc.minima()

        gt = [[[0.7265625], 0.48828125], [[-4.0234375], -7.890625], [[2.265625], 0.859375], [
            [-17.421875, -17.353515625, -17.28515625, -17.216796875, -17.1484375, -17.080078125, -17.01171875,
             -16.943359375, -16.875, -16.806640625, -16.73828125, -16.669921875, -16.6015625, -16.533203125,
             -16.46484375, -16.396484375, -16.328125, -16.259765625, -16.19140625, -16.123046875, -16.0546875,
             -15.986328125, -15.91796875, -15.849609375, -15.78125, -15.712890625, -15.64453125, -15.576171875,
             -15.5078125, -15.439453125, -15.37109375, -15.302734375, -15.234375, -15.166015625, -15.09765625,
             -15.029296875, -14.9609375, -14.892578125, -14.82421875, -14.755859375, -14.6875, -14.619140625,
             -14.55078125, -14.482421875, -14.4140625, -14.345703125, -14.27734375, -14.208984375, -14.140625,
             -14.072265625, -14.00390625, -13.935546875, -13.8671875, -13.798828125, -13.73046875, -13.662109375,
             -13.59375, -13.525390625, -13.45703125, -13.388671875, -13.3203125, -13.251953125, -13.18359375,
             -13.115234375, -13.046875, -12.978515625, -12.91015625, -12.841796875, -12.7734375, -12.705078125,
             -12.63671875, -12.568359375, -12.5, -12.431640625, -12.36328125, -12.294921875, -12.2265625, -12.158203125,
             -12.08984375, -12.021484375, -11.953125, -11.884765625, -11.81640625, -11.748046875, -11.6796875,
             -11.611328125, -11.54296875, -11.474609375, -11.40625, -11.337890625, -11.26953125, -11.201171875,
             -11.1328125, -11.064453125, -10.99609375, -10.927734375, -10.859375, -10.791015625, -10.72265625,
             -10.654296875, -10.5859375, -10.517578125, -10.44921875, -10.380859375, -10.3125, -10.244140625,
             -10.17578125, -10.107421875, -10.0390625, -9.970703125, -9.90234375, -9.833984375, -9.765625, -9.697265625,
             -9.62890625, -9.560546875, -9.4921875, -9.423828125, -9.35546875, -9.287109375, -9.21875, -9.150390625,
             -9.08203125, -9.013671875, -8.9453125, -8.876953125, -8.80859375, -8.740234375, -8.671875, -8.603515625,
             -8.53515625, -8.466796875, -8.3984375, -8.330078125, -8.26171875, -8.193359375, -8.125, -8.056640625,
             -7.98828125, -7.919921875, -7.8515625, -7.783203125, -7.71484375, -7.646484375, -7.578125, -7.509765625,
             -7.44140625, -7.373046875, -7.3046875, -7.236328125, -7.16796875, -7.099609375, -7.03125], -9.125],
         [[5.44921875, 5.458984375, 5.46875, 5.478515625, 5.48828125, 5.498046875, 5.5078125, 5.517578125, 5.52734375],
          2.09375]]

        self.assertAlmostEqual(minima, gt)


if __name__ == '__main__':
    unittest.main()
