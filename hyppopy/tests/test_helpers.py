# -*- coding: utf-8 -*-
#
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

import unittest

from hyppopy.helpers import NestedDictUnfolder


class SolverFactoryTestSuite(unittest.TestCase):

    def setUp(self):
        self.p1 = {"uni1": [1, 2], "uni2": [11, 12]}
        self.p2 = {"cat": {"a": {"uni1": [1, 2], "uni2": [11, 12]}, "b": {"uni1": [1, 2], "uni2": [11, 12]}}}
        self.p3 = {"cat1": {
            "a1": {"cat2": {"a2": {"uni1": [1, 2], "uni2": [11, 12]}, "b2": {"uni1": [1, 2], "uni2": [11, 12]}}},
            "b1": {"cat2": {"a2": {"uni1": [1, 2], "uni2": [11, 12]}, "b2": {"uni1": [1, 2], "uni2": [11, 12]}}}}}

        self.output_p3 = [{'cat1': 'a1', 'cat2': 'a2', 'uni1': 1, 'uni2': 11},
                         {'cat1': 'a1', 'cat2': 'a2', 'uni1': 1, 'uni2': 12},
                         {'cat1': 'a1', 'cat2': 'a2', 'uni1': 2, 'uni2': 11},
                         {'cat1': 'a1', 'cat2': 'a2', 'uni1': 2, 'uni2': 12},
                         {'cat1': 'a1', 'cat2': 'b2', 'uni1': 1, 'uni2': 11},
                         {'cat1': 'a1', 'cat2': 'b2', 'uni1': 1, 'uni2': 12},
                         {'cat1': 'a1', 'cat2': 'b2', 'uni1': 2, 'uni2': 11},
                         {'cat1': 'a1', 'cat2': 'b2', 'uni1': 2, 'uni2': 12},
                         {'cat1': 'b1', 'cat2': 'a2', 'uni1': 1, 'uni2': 11},
                         {'cat1': 'b1', 'cat2': 'a2', 'uni1': 1, 'uni2': 12},
                         {'cat1': 'b1', 'cat2': 'a2', 'uni1': 2, 'uni2': 11},
                         {'cat1': 'b1', 'cat2': 'a2', 'uni1': 2, 'uni2': 12},
                         {'cat1': 'b1', 'cat2': 'b2', 'uni1': 1, 'uni2': 11},
                         {'cat1': 'b1', 'cat2': 'b2', 'uni1': 1, 'uni2': 12},
                         {'cat1': 'b1', 'cat2': 'b2', 'uni1': 2, 'uni2': 11},
                         {'cat1': 'b1', 'cat2': 'b2', 'uni1': 2, 'uni2': 12}]

        self.output_p2 = [{'cat': 'a', 'uni1': 1, 'uni2': 11},
                         {'cat': 'a', 'uni1': 1, 'uni2': 12},
                         {'cat': 'a', 'uni1': 2, 'uni2': 11},
                         {'cat': 'a', 'uni1': 2, 'uni2': 12},
                         {'cat': 'b', 'uni1': 1, 'uni2': 11},
                         {'cat': 'b', 'uni1': 1, 'uni2': 12},
                         {'cat': 'b', 'uni1': 2, 'uni2': 11},
                         {'cat': 'b', 'uni1': 2, 'uni2': 12}]

        self.output_p1 = [{'uni1': 1, 'uni2': 11},
                         {'uni1': 1, 'uni2': 12},
                         {'uni1': 2, 'uni2': 11},
                         {'uni1': 2, 'uni2': 12}]

    def test_nested_dict_unfolder_p1(self):
        unfolder = NestedDictUnfolder(self.p1)
        unfolded = unfolder.unfold()

        for it1, it2 in zip(unfolded, self.output_p1):
            self.assertEqual(it1, it2)

    def test_nested_dict_unfolder_p2(self):
        unfolder = NestedDictUnfolder(self.p2)
        unfolded = unfolder.unfold()

        for it1, it2 in zip(unfolded, self.output_p2):
            self.assertEqual(it1, it2)

    def test_nested_dict_unfolder_p3(self):
        unfolder = NestedDictUnfolder(self.p3)
        unfolded = unfolder.unfold()
        for it1, it2 in zip(unfolded, self.output_p3):
            self.assertEqual(it1, it2)



if __name__ == '__main__':
    unittest.main()

