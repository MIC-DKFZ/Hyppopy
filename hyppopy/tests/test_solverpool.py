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

from hyppopy.SolverPool import SolverPool
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.FunctionSimulator import FunctionSimulator
from hyppopy.solvers.HyperoptSolver import HyperoptSolver
from hyppopy.solvers.OptunitySolver import OptunitySolver
from hyppopy.solvers.OptunaSolver import OptunaSolver
from hyppopy.solvers.RandomsearchSolver import RandomsearchSolver
from hyppopy.solvers.QuasiRandomsearchSolver import QuasiRandomsearchSolver
from hyppopy.solvers.GridsearchSolver import GridsearchSolver


class SolverPoolTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_PoolContent(self):
        names = SolverPool.get_solver_names()
        self.assertTrue("hyperopt" in names)
        self.assertTrue("optunity" in names)
        self.assertTrue("optuna" in names)
        self.assertTrue("randomsearch" in names)
        self.assertTrue("quasirandomsearch" in names)
        self.assertTrue("gridsearch" in names)

    def test_getHyperoptSolver(self):
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
        solver = SolverPool.get("hyperopt", project)
        self.assertTrue(isinstance(solver, HyperoptSolver))
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(300 <= best['axis_00'] <= 700)
        self.assertTrue(0 <= best['axis_01'] <= 0.8)
        self.assertTrue(3.5 <= best['axis_02'] <= 6.5)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_getOptunitySolver(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "uniform",
                    "data": [300, 800],
                    "type": float
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [-1, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10],
                    "type": float
                }
            },
            "max_iterations": 100
        }

        project = HyppopyProject(config)
        solver = SolverPool.get("optunity", project)
        self.assertTrue(isinstance(solver, OptunitySolver))
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(300 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_getOptunaSolver(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "uniform",
                    "data": [300, 800],
                    "type": float
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [-1, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10],
                    "type": float
                }
            },
            "max_iterations": 100
        }

        project = HyppopyProject(config)
        solver = SolverPool.get("optuna", project)
        self.assertTrue(isinstance(solver, OptunaSolver))
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(300 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_getRandomsearchSolver(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "uniform",
                    "data": [0, 800],
                    "type": float
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [-1, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10],
                    "type": float
                }
            },
            "max_iterations": 300
        }

        project = HyppopyProject(config)
        solver = SolverPool.get("randomsearch", project)
        self.assertTrue(isinstance(solver, RandomsearchSolver))
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(0 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_getQuasiRandomsearchSolver(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "uniform",
                    "data": [0, 800],
                    "type": float
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [-1, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10],
                    "type": float
                }
            },
            "max_iterations": 300
        }

        project = HyppopyProject(config)
        solver = SolverPool.get("quasirandomsearch", project)
        self.assertTrue(isinstance(solver, QuasiRandomsearchSolver))
        vfunc = FunctionSimulator()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(0 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_getGridsearchSolver(self):
        config = {
            "hyperparameter": {
                "value 1": {
                    "domain": "uniform",
                    "data": [0, 20],
                    "type": int,
                    "frequency": 11
                },
                "value 2": {
                    "domain": "normal",
                    "data": [0, 20.0],
                    "type": float,
                    "frequency": 11
                },
                "value 3": {
                    "domain": "loguniform",
                    "data": [1, 10000],
                    "type": float,
                    "frequency": 11
                },
                "categorical": {
                    "domain": "categorical",
                    "data": ["a", "b"],
                    "type": str,
                    "frequency": 1
                }
            }}
        res_labels = ['value 1', 'value 2', 'value 3', 'categorical']
        res_values = [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      [0.0, 5.467452952462635, 8.663855974622837, 9.755510546899107, 9.973002039367397, 10.0,
                       10.026997960632603, 10.244489453100893, 11.336144025377163, 14.532547047537365, 20.0],
                      [1.0, 2.51188643150958, 6.309573444801933, 15.848931924611136, 39.810717055349734,
                       100.00000000000004, 251.18864315095806, 630.9573444801938, 1584.8931924611143,
                       3981.071705534977, 10000.00000000001],
                      ['a', 'b']]
        project = HyppopyProject(config)
        solver = SolverPool.get("gridsearch", project)
        self.assertTrue(isinstance(solver, GridsearchSolver))
        searchspace = solver.convert_searchspace(config["hyperparameter"])
        for n in range(len(res_labels)):
            self.assertEqual(res_labels[n], searchspace[0][n])
        for i in range(3):
            self.assertAlmostEqual(res_values[i], searchspace[1][i])
        self.assertEqual(res_values[3], searchspace[1][3])

    def test_projectNone(self):
        solver = SolverPool.get("hyperopt")
        solver = SolverPool.get("optunity")
        solver = SolverPool.get("optuna")
        solver = SolverPool.get("randomsearch")
        solver = SolverPool.get("quasirandomsearch")
        solver = SolverPool.get("gridsearch")

        self.assertRaises(AssertionError, SolverPool.get, "foo")
