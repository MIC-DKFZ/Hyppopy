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
import shutil
import unittest
import tempfile
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.svc_usecase.svc_usecase import svc_usecase
from hyppopy.workflows.knc_usecase.knc_usecase import knc_usecase
from hyppopy.workflows.lda_usecase.adaboost_usecase import lda_usecase
from hyppopy.workflows.randomforest_usecase.randomforest_usecase import randomforest_usecase


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class ProjectManagerTestSuite(unittest.TestCase):

    def setUp(self):
        breast_cancer_data = load_breast_cancer()
        x = breast_cancer_data.data
        y = breast_cancer_data.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=23)

        self.root = os.path.join(tempfile.gettempdir(), 'test_data')
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        x_train_fname = os.path.join(self.root, 'x_train.npy')
        y_train_fname = os.path.join(self.root, 'y_train.npy')
        np.save(x_train_fname, x_train)
        np.save(y_train_fname, y_train)

        self.test = [x_test, y_test]
        self.config = {
            "hyperparameter": {},
            "settings": {
                "solver_plugin": {
                    "max_iterations": 50,
                    "use_plugin": "hyperopt",
                    "output_dir": os.path.join(self.root, 'test_results')
                },
                "custom": {
                    "data_path": self.root,
                    "data_name": "x_train.npy",
                    "labels_name": "y_train.npy"
                }
            }}

    def test_svc_usecase(self):
        hyperparameter = {
            "C": {
                "domain": "uniform",
                "data": [0.0001, 300.0],
                "type": "float"
            },
            "kernel": {
                "domain": "categorical",
                "data": ["linear", "poly", "rbf"],
                "type": "str"
            }
        }

        self.config["hyperparameter"] = hyperparameter
        ProjectManager.set_config(self.config)
        uc = svc_usecase()
        uc.run(save=True)
        res, best = uc.get_results()
        print("="*30)
        print(best)
        print("=" * 30)
        clf = SVC(**best)
        train_predictions = clf.predict(self.test[0])
        acc = accuracy_score(self.test[1], train_predictions)
        print("Accuracy: {:.4%}".format(acc))
        print("=" * 30)

    def test_randomforest_usecase(self):
        hyperparameter = {
            "n_estimators": {
                "domain": "uniform",
                "data": [1, 500],
                "type": "int"
            },
            "criterion": {
                "domain": "categorical",
                "data": ["gini", "entropy"],
                "type": "str"
            },
            "max_depth": {
                "domain": "uniform",
                "data": [1, 50],
                "type": "int"
            },
            "max_features": {
                "domain": "categorical",
                "data": ["auto", "sqrt", "log2"],
                "type": "str"
            }
        }

        self.config["hyperparameter"] = hyperparameter
        ProjectManager.set_config(self.config)
        uc = randomforest_usecase()
        uc.run(save=True)
        res, best = uc.get_results()
        print(best)

    def test_lda_usecase(self):
        hyperparameter = {
            "solver": {
                "domain": "categorical",
                "data": ["svd", "lsqr", "eigen"],
                "type": "str"
            },
            "tol": {
                "domain": "uniform",
                "data": [0.00000001, 1.0],
                "type": "float"
            }
        }

        self.config["hyperparameter"] = hyperparameter
        ProjectManager.set_config(self.config)
        uc = lda_usecase()
        uc.run(save=True)
        res, best = uc.get_results()
        print(best)

    def test_knc_usecase(self):
        hyperparameter = {
                "n_neighbors": {
                    "domain": "uniform",
                    "data": [1, 100],
                    "type": "int"
                },
                "weights": {
                    "domain": "categorical",
                    "data": ["uniform", "distance"],
                    "type": "str"
                },
                "algorithm": {
                    "domain": "categorical",
                    "data": ["auto", "ball_tree", "kd_tree", "brute"],
                    "type": "str"
                }
            }

        self.config["hyperparameter"] = hyperparameter
        ProjectManager.set_config(self.config)
        uc = knc_usecase()
        uc.run(save=True)
        res, best = uc.get_results()
        print(best)

    def tearDown(self):
        pass
        # if os.path.isdir(self.root):
        #     shutil.rmtree(self.root)


if __name__ == '__main__':
    unittest.main()
