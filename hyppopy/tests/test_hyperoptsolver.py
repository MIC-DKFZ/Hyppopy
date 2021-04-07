# Hyppopy - A Hyper-Parameter Optimization Toolbox
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

from hyppopy.solvers.HyperoptSolver import *
from hyppopy.FunctionSimulator import FunctionSimulator
from hyppopy.HyppopyProject import HyppopyProject


class HyperoptSolverTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_solver_complete(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "uniform",
                    "data": [300, 700],
                    "type": float
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [0, 0.8],
                    "type": float
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [3.5, 6.5],
                    "type": float
                }
            },
            "max_iterations": 500
            }

        project = HyppopyProject(config)
        solver = HyperoptSolver(project)
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(575 <= best['axis_00'] <= 585)
        self.assertTrue(0.1 <= best['axis_01'] <= 0.8)
        self.assertTrue(4.7 <= best['axis_02'] <= 5.3)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_solver_normal(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [500, 650],
                    "type": float
                },
                "axis_01": {
                    "domain": "normal",
                    "data": [0.1, 0.8],
                    "type": float
                },
                "axis_02": {
                    "domain": "normal",
                    "data": [4.5, 5.5],
                    "type": float
                }
            },
            "max_iterations": 500,
            }

        project = HyppopyProject(config)
        solver = HyperoptSolver(project)
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(575 <= best['axis_00'] <= 585)
        self.assertTrue(0.1 <= best['axis_01'] <= 0.8)
        self.assertTrue(4.7 <= best['axis_02'] <= 5.3)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_solver_loguniform(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [500, 650],
                    "type": float
                },
                "axis_01": {
                    "domain": "loguniform",
                    "data": [0.001, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "normal",
                    "data": [4.5, 5.5],
                    "type": float
                }
            },
            "max_iterations": 500,
            }

        project = HyppopyProject(config)
        solver = HyperoptSolver(project)
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(575 <= best['axis_00'] <= 585)
        self.assertTrue(0.001 <= best['axis_01'] <= 1.0)
        self.assertTrue(4.7 <= best['axis_02'] <= 5.3)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_categorical(self):
        config = {
            "hyperparameter": {
                "C": {
                    "domain": "uniform",
                    "data": [1, 20],
                    "type": int
                },
                "gamma": {
                    "domain": "loguniform",
                    "data": [0.0, 20.0],
                    "type": float
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": str
                },
                "with_ovr": {
                    "domain": "categorical",
                    "data": [True, False],
                    "type": bool
                }
            },
            "max_iterations": 300
        }
        project = HyppopyProject(config=config)
        from sklearn.svm import SVC
        from sklearn.datasets import load_iris
        from sklearn.model_selection import cross_val_score
        iris_data = load_iris()
        data = [iris_data.data, iris_data.target]

        def my_loss_function(data, params):
            if params["with_ovr"]:
                params["decision_function_shape"] = "ovr"
            else:
                params["decision_function_shape"] = "ovo"
            del params["with_ovr"]
            clf = SVC(**params)
            return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()

        blackbox = BlackboxFunction(blackbox_func=my_loss_function, data=data)
        solver = HyperoptSolver(project=project)
        solver.blackbox = blackbox
        solver.run()


if __name__ == '__main__':
    unittest.main()
