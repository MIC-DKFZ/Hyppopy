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

from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.globals import TESTDATA_DIR


def foo(a, b):
    return a + b


class VirtualFunctionTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_project_creation(self):
        config = {
            "hyperparameter": {
                "C": {
                    "domain": "uniform",
                    "data": [0.0001, 20],
                    "type": "float"
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": "str"
                }
            },
            "settings": {
                "solver": {
                    "max_iterations": 300
                },
                "custom": {
                    "param1": 1,
                    "param2": 2,
                    "function": foo
                }
            }}

        project = HyppopyProject()
        project.set_config(config)
        self.assertEqual(project.hyperparameter["C"]["domain"], "uniform")
        self.assertEqual(project.hyperparameter["C"]["data"], [0.0001, 20])
        self.assertEqual(project.hyperparameter["C"]["type"], "float")
        self.assertEqual(project.hyperparameter["kernel"]["domain"], "categorical")
        self.assertEqual(project.hyperparameter["kernel"]["data"], ["linear", "sigmoid", "poly", "rbf"])
        self.assertEqual(project.hyperparameter["kernel"]["type"], "str")

        self.assertEqual(project.solver_max_iterations, 300)
        self.assertEqual(project.custom_param1, 1)
        self.assertEqual(project.custom_param2, 2)
        self.assertEqual(project.custom_function(2, 3), 5)

        self.assertTrue(project.get_typeof("C") is float)
        self.assertTrue(project.get_typeof("kernel") is str)

        project.clear()
        self.assertTrue(len(project.hyperparameter) == 0)
        self.assertTrue(len(project.settings) == 0)
        self.assertTrue("solver_max_iterations" not in project.__dict__.keys())
        self.assertTrue("custom_param1" not in project.__dict__.keys())
        self.assertTrue("custom_param2" not in project.__dict__.keys())
        self.assertTrue("custom_function" not in project.__dict__.keys())

        project.add_hyperparameter(name="C", domain="uniform", data=[0.0001, 20], dtype="float")
        project.add_hyperparameter(name="kernel", domain="categorical", data=["linear", "sigmoid", "poly", "rbf"], dtype="str")

        self.assertEqual(project.hyperparameter["C"]["domain"], "uniform")
        self.assertEqual(project.hyperparameter["C"]["data"], [0.0001, 20])
        self.assertEqual(project.hyperparameter["C"]["type"], "float")
        self.assertEqual(project.hyperparameter["kernel"]["domain"], "categorical")
        self.assertEqual(project.hyperparameter["kernel"]["data"], ["linear", "sigmoid", "poly", "rbf"])
        self.assertEqual(project.hyperparameter["kernel"]["type"], "str")

        project.add_settings("solver", "max_iterations", 500)
        self.assertEqual(project.solver_max_iterations, 500)
        project.add_settings("solver", "max_iterations", 200)
        self.assertEqual(project.solver_max_iterations, 200)

