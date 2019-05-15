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

from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.solvers.HyppopySolver import HyppopySolver


class FooSolver1(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None


class FooSolver2(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass


class FooSolver3(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def define_interface(self):
        pass


class FooSolver4(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def define_interface(self):
        pass

    def loss_function_call(self, params):
        return 1


class GooSolver1(HyppopySolver):
    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def define_interface(self):
        self._add_member("max_iterations", int, 1.0, 100)

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass


class GooSolver2(HyppopySolver):
    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def define_interface(self):
        self._add_member("max_iterations", int, 100, 5.0)

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass


class TestSolver1(HyppopySolver):
    def __init__(self, project=None):
        config = {
            "hyperparameter": {
                "gamma": {
                    "domain": "uniform",
                    "data": [0.0001, 20.0],
                    "type": float
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": str
                }
            },
            "foo1": 300,
            "goo": 1.0
        }
        project = HyppopyProject(config)

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass

    def define_interface(self):
        self._add_member("foo", int)
        self._add_member("goo", float)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)


class TestSolver2(HyppopySolver):
    def __init__(self, project=None):
        config = {
            "hyperparameter": {
                "gamma": {
                    "domain": "normal",
                    "data": [0.0001, 20.0],
                    "type": float
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": str
                }
            },
            "foo": 300,
            "goo": 1.0
        }
        project = HyppopyProject(config)

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass

    def define_interface(self):
        self._add_member("foo", int)
        self._add_member("goo", float)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)


class TestSolver3(HyppopySolver):
    def __init__(self, project=None):
        config = {
            "hyperparameter": {
                "gamma": {
                    "domain": 100,
                    "data": [0.0001, 20.0],
                    "type": float
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": str
                }
            },
            "foo": 300,
            "goo": 1.0
        }
        project = HyppopyProject(config)

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass

    def define_interface(self):
        self._add_member("foo", int)
        self._add_member("goo", float)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)


class TestSolver4(HyppopySolver):
    def __init__(self, project=None):
        config = {
            "hyperparameter": {
                "gamma": {
                    "domina": "uniform",
                    "data": [0.0001, 20.0],
                    "type": float
                },
                "kernel": {
                    "domain": "categorical",
                    "data": ["linear", "sigmoid", "poly", "rbf"],
                    "type": str
                }
            },
            "foo": 300,
            "goo": 1.0
        }
        project = HyppopyProject(config)

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass

    def define_interface(self):
        self._add_member("foo", int)
        self._add_member("goo", float)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)


class TestRunSolver1(HyppopySolver):
    def __init__(self, project=None):
        project = HyppopyProject({})

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        raise EnvironmentError("ForTesting")

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        pass

    def define_interface(self):
        pass


class TestRunSolver2(HyppopySolver):
    def __init__(self, project=None):
        project = HyppopyProject({})

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        return 1

    def execute_solver(self, searchspace):
        raise EnvironmentError("ForTesting")

    def define_interface(self):
        pass


class TestLossFuncSolver1(HyppopySolver):
    def __init__(self, project=None):
        project = HyppopyProject({})

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        raise Exception("For testing")

    def execute_solver(self, searchspace):
        self.loss_function(**{})

    def define_interface(self):
        pass


class TestLossFuncSolver2(HyppopySolver):
    def __init__(self, project=None):
        project = HyppopyProject({})

        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def convert_searchspace(self, hyperparameter):
        pass

    def loss_function_call(self, params):
        from numpy import nan as npnan
        return npnan

    def execute_solver(self, searchspace):
        self.loss_function(**{})

    def define_interface(self):
        pass


class HyppopySolverTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_class(self):
        self.assertRaises(NotImplementedError, HyppopySolver)
        self.assertRaises(NotImplementedError, FooSolver1)
        self.assertRaises(NotImplementedError, FooSolver2)
        foo = FooSolver3()
        self.assertRaises(NotImplementedError, foo.loss_function_call, {})
        foo = FooSolver4()
        self.assertEqual(foo.loss_function_call({}), 1)
        self.assertRaises(NotImplementedError, foo.execute_solver, {})

        self.assertRaises(AssertionError, GooSolver1)
        self.assertRaises(AssertionError, GooSolver2)

    def test_check_project(self):
        self.assertRaises(LookupError, TestSolver1)
        self.assertRaises(LookupError, TestSolver2)
        self.assertRaises(TypeError, TestSolver3)
        self.assertRaises(LookupError, TestSolver4)

    def test_run(self):
        solver = TestRunSolver1()
        self.assertRaises(AssertionError, solver.run)
        solver = TestRunSolver2()
        self.assertRaises(AssertionError, solver.run)
        self.assertRaises(TypeError, solver.project, 100)
        self.assertRaises(TypeError, solver.blackbox, 100)
        self.assertRaises(TypeError, solver.best, 100)

    def test_lossfunccall(self):
        TestLossFuncSolver1().run(print_stats=False)
        TestLossFuncSolver2().run(print_stats=False)
