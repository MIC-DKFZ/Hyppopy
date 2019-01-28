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

from hyppopy.solver_factory import SolverFactory


class SolverFactoryTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_plugin_load(self):
        factory = SolverFactory.instance()
        for solver_name in factory.get_solver_names():
            self.assertTrue(factory.get_solver(solver_name).is_ready())


if __name__ == '__main__':
    unittest.main()
