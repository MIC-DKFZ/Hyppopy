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

from hyppopy.HyppopyProject import HyppopyProject


def foo(a, b):
    return a + b


class HyppopyProjectTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_project_creation(self):
        config = {
            "hyperparameter": {
                "C": {
                    "domain": "uniform",
                    "data": [0.0001, 20],
                    "type": float
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": str
                }
            },
            "max_iterations": 300,
            "param1": 1,
            "param2": 2,
            "function": foo
        }

        project = HyppopyProject()
        project.set_config(config)
        self.assertEqual(project.hyperparameter["C"]["domain"], "uniform")
        self.assertEqual(project.hyperparameter["C"]["data"], [0.0001, 20])
        self.assertTrue(project.hyperparameter["C"]["type"] is float)
        self.assertEqual(project.hyperparameter["kernel"]["domain"], "categorical")
        self.assertEqual(project.hyperparameter["kernel"]["data"], ["linear", "sigmoid", "poly", "rbf"])
        self.assertTrue(project.hyperparameter["kernel"]["type"] is str)

        self.assertEqual(project.max_iterations, 300)
        self.assertEqual(project.param1, 1)
        self.assertEqual(project.param2, 2)
        self.assertEqual(project.function(2, 3), 5)

        self.assertTrue(project.get_typeof("C") is float)
        self.assertTrue(project.get_typeof("kernel") is str)

        project = HyppopyProject()

        project.add_hyperparameter(name="C", domain="uniform", data=[0.0001, 20], type=float)
        project.add_hyperparameter(name="kernel", domain="categorical", data=["linear", "sigmoid", "poly", "rbf"], type=str)

        self.assertEqual(project.hyperparameter["C"]["domain"], "uniform")
        self.assertEqual(project.hyperparameter["C"]["data"], [0.0001, 20])
        self.assertTrue(project.hyperparameter["C"]["type"] is float)
        self.assertEqual(project.hyperparameter["kernel"]["domain"], "categorical")
        self.assertEqual(project.hyperparameter["kernel"]["data"], ["linear", "sigmoid", "poly", "rbf"])
        self.assertTrue(project.hyperparameter["kernel"]["type"] is str)

        project.set_settings(max_iterations=500)
        self.assertEqual(project.max_iterations, 500)
        project.add_setting("my_param", 42)
        self.assertEqual(project.my_param, 42)
        project.add_setting("max_iterations", 200)
        self.assertEqual(project.max_iterations, 200)

