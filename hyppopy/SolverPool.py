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

from .Singleton import *

import os
import logging
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.solvers.OptunaSolver import OptunaSolver
from hyppopy.solvers.BayesOptSolver import BayesOptSolver
from hyppopy.solvers.HyperoptSolver import HyperoptSolver
from hyppopy.solvers.OptunitySolver import OptunitySolver
from hyppopy.solvers.GridsearchSolver import GridsearchSolver
from hyppopy.solvers.RandomsearchSolver import RandomsearchSolver
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


@singleton_object
class SolverPool(metaclass=Singleton):

    def __init__(self):
        self._solver_list = ["hyperopt",
                             "optunity",
                             "bayesopt",
                             "optuna",
                             "randomsearch",
                             "gridsearch"]

    def get_solver_names(self):
        return self._solver_list

    def get(self, solver_name=None, project=None):
        if solver_name is not None:
            assert isinstance(solver_name, str), "precondition violation, solver_name type str expected, got {} instead!".format(type(solver_name))
        if project is not None:
            assert isinstance(project, HyppopyProject), "precondition violation, project type HyppopyProject expected, got {} instead!".format(type(project))
            if "custom_use_solver" in project.__dict__:
                solver_name = project.custom_use_solver
        if solver_name not in self._solver_list:
            raise AssertionError("Solver named [{}] not implemented!".format(solver_name))

        if solver_name == "hyperopt":
            if project is not None:
                return HyperoptSolver(project)
            return HyperoptSolver()
        elif solver_name == "optunity":
            if project is not None:
                return OptunitySolver(project)
            return OptunitySolver()
        elif solver_name == "bayesopt":
            if project is not None:
                return BayesOptSolver(project)
            return BayesOptSolver()
        elif solver_name == "optuna":
            if project is not None:
                return OptunaSolver(project)
            return OptunaSolver()
        elif solver_name == "gridsearch":
            if project is not None:
                return GridsearchSolver(project)
            return GridsearchSolver()
        elif solver_name == "randomsearch":
            if project is not None:
                return RandomsearchSolver(project)
            return RandomsearchSolver()

