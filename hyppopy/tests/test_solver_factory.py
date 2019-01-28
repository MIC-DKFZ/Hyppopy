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


class PluginMechanismTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_factory_build(self):
        factory = SolverFactory.instance()
        print(factory.get_solver_names())
        factory.get_solver("HyperoptPlugin")


if __name__ == '__main__':
    unittest.main()
