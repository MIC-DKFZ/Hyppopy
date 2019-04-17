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
import optuna
import logging
import warnings
import numpy as np
from pprint import pformat

from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class OptunaSolver(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def define_interface(self):
        self.add_member("max_iterations", int)
        self.add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self.add_hyperparameter_signature(name="data", dtype=list)
        self.add_hyperparameter_signature(name="type", dtype=type)

    def reformat_parameter(self, params):
        out_params = {}
        for name, value in params.items():
            if self._searchspace[name]["domain"] == "categorical":
                out_params[name] = self._searchspace[name]["data"][int(np.round(value))]
            else:
                if self._searchspace[name]["type"] is int:
                    out_params[name] = int(np.round(value))
                else:
                    out_params[name] = value
        return out_params

    def trial_cache(self, trial):
        params = {}
        for name, param in self._searchspace.items():
            if param["domain"] == "categorical":
                params[name] = trial.suggest_categorical(name, param["data"])
            else:
                params[name] = trial.suggest_uniform(name, param["data"][0], param["data"][1])
        return self.loss_function(**params)

    def loss_function_call(self, params):
        for key in params.keys():
            if self.project.get_typeof(key) is int:
                params[key] = int(round(params[key]))
        return self.blackbox(**params)

    def execute_solver(self, searchspace):
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        self._searchspace = searchspace

        try:
            study = optuna.create_study()
            study.optimize(self.trial_cache, n_trials=self.max_iterations)
            self.best = study.best_trial.params
        except Exception as e:
            LOG.error("internal error in bayes_opt maximize occured. {}".format(e))
            raise BrokenPipeError("internal error in bayes_opt maximize occured. {}".format(e))

    def convert_searchspace(self, hyperparameter):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        for name, param in hyperparameter.items():
            if param["domain"] != "categorical" and param["domain"] != "uniform":
                msg = "Warning: Optuna cannot handle {} domain. Only uniform and categorical domains are supported!".format(param["domain"])
                warnings.warn(msg)
                LOG.warning(msg)
        return hyperparameter
