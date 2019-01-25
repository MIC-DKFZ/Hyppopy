# DKFZ
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

# -*- coding: utf-8 -*-

import unittest

from hyppopy.solver_factory import SolverFactory


class PluginMechanismTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def setUp(self):
        pass

    def test_1(self):
        """

        :return:
        """
        factory = SolverFactory.instance()


if __name__ == '__main__':
    unittest.main()
