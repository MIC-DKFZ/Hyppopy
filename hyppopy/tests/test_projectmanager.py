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
import tempfile
import unittest
from hyppopy.projectmanager import ProjectManager


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class ProjectManagerTestSuite(unittest.TestCase):

    def setUp(self):
        self.config = {
            "hyperparameter": {
                "C": {
                    "domain": "uniform",
                    "data": [0, 20],
                    "type": "float"
                },
                "gamma": {
                    "domain": "uniform",
                    "data": [0.0001, 20.0],
                    "type": "float"
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": "str"
                },
                "decision_function_shape": {
                    "domain": "categorical",
                    "data": ["ovo", "ovr"],
                    "type": "str"
                }
            },
            "settings": {
                "solver_plugin": {
                    "max_iterations": 300,
                    "use_plugin": "hyperopt",
                    "output_dir": os.path.join(tempfile.gettempdir(), 'results')
                },
                "custom": {
                    "the_answer": 42
                }
            }}

    def test_read_attrs(self):
        ProjectManager.read_config(os.path.join(DATA_PATH, *('Titanic', 'rf_config.xml')))
        self.assertEqual(ProjectManager.data_name, 'train_cleaned.csv')
        self.assertEqual(ProjectManager.labels_name, 'Survived')
        self.assertEqual(ProjectManager.max_iterations, 3)
        self.assertEqual(ProjectManager.use_plugin, 'optunity')

        hp = ProjectManager.get_hyperparameter()
        self.assertTrue("n_estimators" in hp.keys())
        self.assertTrue("domain" in hp["n_estimators"].keys())
        self.assertTrue("data" in hp["n_estimators"].keys())
        self.assertTrue("type" in hp["n_estimators"].keys())
        self.assertEqual(hp["n_estimators"]["domain"], "uniform")
        self.assertEqual(hp["n_estimators"]["type"], "int")
        self.assertEqual(hp["n_estimators"]["data"], [3, 200])

        self.assertTrue("max_depth" in hp.keys())
        self.assertTrue("domain" in hp["max_depth"].keys())
        self.assertTrue("data" in hp["max_depth"].keys())
        self.assertTrue("type" in hp["max_depth"].keys())
        self.assertEqual(hp["max_depth"]["domain"], "uniform")
        self.assertEqual(hp["max_depth"]["type"], "int")
        self.assertEqual(hp["max_depth"]["data"], [3, 50])

        self.assertTrue("criterion" in hp.keys())
        self.assertTrue("domain" in hp["criterion"].keys())
        self.assertTrue("data" in hp["criterion"].keys())
        self.assertTrue("type" in hp["criterion"].keys())
        self.assertEqual(hp["criterion"]["domain"], "categorical")
        self.assertEqual(hp["criterion"]["type"], "str")
        self.assertEqual(hp["criterion"]["data"], ["gini", "entropy"])

    def test_set_attrs(self):
        self.assertTrue(ProjectManager.set_config(self.config))
        self.assertEqual(ProjectManager.max_iterations, 300)
        self.assertEqual(ProjectManager.use_plugin, 'hyperopt')
        self.assertEqual(ProjectManager.the_answer, 42)

        hp = ProjectManager.get_hyperparameter()
        self.assertTrue("C" in hp.keys())
        self.assertTrue("domain" in hp["C"].keys())
        self.assertTrue("data" in hp["C"].keys())
        self.assertTrue("type" in hp["C"].keys())
        self.assertEqual(hp["C"]["domain"], "uniform")
        self.assertEqual(hp["C"]["type"], "float")
        self.assertEqual(hp["C"]["data"], [0, 20])

        self.assertTrue("gamma" in hp.keys())
        self.assertTrue("domain" in hp["gamma"].keys())
        self.assertTrue("data" in hp["gamma"].keys())
        self.assertTrue("type" in hp["gamma"].keys())
        self.assertEqual(hp["gamma"]["domain"], "uniform")
        self.assertEqual(hp["gamma"]["type"], "float")
        self.assertEqual(hp["gamma"]["data"], [0.0001, 20.0])

        self.assertTrue("kernel" in hp.keys())
        self.assertTrue("domain" in hp["kernel"].keys())
        self.assertTrue("data" in hp["kernel"].keys())
        self.assertTrue("type" in hp["kernel"].keys())
        self.assertEqual(hp["kernel"]["domain"], "categorical")
        self.assertEqual(hp["kernel"]["type"], "str")
        self.assertEqual(hp["kernel"]["data"], ["linear", "sigmoid", "poly", "rbf"])

        self.assertTrue("decision_function_shape" in hp.keys())
        self.assertTrue("domain" in hp["decision_function_shape"].keys())
        self.assertTrue("data" in hp["decision_function_shape"].keys())
        self.assertTrue("type" in hp["decision_function_shape"].keys())
        self.assertEqual(hp["decision_function_shape"]["domain"], "categorical")
        self.assertEqual(hp["decision_function_shape"]["type"], "str")
        self.assertEqual(hp["decision_function_shape"]["data"], ["ovo", "ovr"])


if __name__ == '__main__':
    unittest.main()
