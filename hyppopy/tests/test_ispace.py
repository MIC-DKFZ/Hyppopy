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

from hyppopy.ispace import ISpace


class TestSpace(ISpace):
    def convert(self, *args, **kwargs):
        pass


class ISpaceTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_IO(self):
        ispace = TestSpace()
        tdict = {
                "a": {
                    "b": {
                        "3": 2,
                        "43": 30,
                        "c": [],
                        "d": ['red', 'buggy', 'bumpers'],
                    }
                },
                "A": {"X": 1}
        }
        ispace.set(tdict)
        self.assertEqual(ispace.get_section('a/b/d')[0], 'red')


if __name__ == '__main__':
    unittest.main()
