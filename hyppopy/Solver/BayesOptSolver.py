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

import os
import logging
import warnings
import numpy as np
from pprint import pformat
from hyperopt import Trials
from bayes_opt import BayesianOptimization

from hyppopy.globals import DEBUGLEVEL
from hyppopy.solver.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class BayesOptSolver(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None
        self._idx = None

    def reformat_parameter(self, params):
        out_params = {}
        for name, value in params.items():
            if self._searchspace[name]["domain"] == "categorical":
                out_params[name] = self._searchspace[name]["data"][int(np.round(value))]
            else:
                if self._searchspace[name]["type"] == "int":
                    out_params[name] = int(np.round(value))
                else:
                    out_params[name] = value
        return out_params

    def loss_function_call(self, params):
        params = self.reformat_parameter(params)
        for key in params.keys():
            if self.project.get_typeof(key) is int:
                params[key] = int(round(params[key]))
        return self.blackbox(**params)

    def execute_solver(self, searchspace):
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        self.trials = Trials()
        self._idx = 0

        try:
            optimizer = BayesianOptimization(f=self.loss_function, pbounds=searchspace, verbose=0)
            optimizer.maximize(init_points=2, n_iter=self.max_iterations)
            self.best = self.reformat_parameter(optimizer.max["params"])
        except Exception as e:
            LOG.error("internal error in bayes_opt maximize occured. {}".format(e))
            raise BrokenPipeError("internal error in bayes_opt maximize occured. {}".format(e))

    def convert_searchspace(self, hyperparameter):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        self._searchspace = hyperparameter
        pbounds = {}
        for name, param in hyperparameter.items():
            if param["domain"] != "categorical":
                if param["domain"] != "uniform":
                    msg = "Warning: BayesOpt cannot handle {} domain. Only uniform and categorical domains are supported!".format(
                        param["domain"])
                    warnings.warn(msg)
                    LOG.warning(msg)
                pbounds[name] = (param["data"][0], param["data"][1])
            else:
                pbounds[name] = (0, len(param["data"])-1)
        return pbounds
