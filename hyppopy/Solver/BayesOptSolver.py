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
import copy
import logging
import datetime
import numpy as np
from pprint import pformat
from hyperopt import Trials
from bayes_opt import BayesianOptimization

from hyppopy.globals import DEBUGLEVEL
from hyppopy.BlackboxFunction import BlackboxFunction
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

    def loss_function(self, **params):
        self._idx += 1
        params = self.reformat_parameter(params)
        vals = {}
        idx = {}
        for key, value in params.items():
            vals[key] = [value]
            idx[key] = [self._idx]
        trial = {'tid': self._idx,
                 'result': {'loss': None, 'status': 'ok'},
                 'misc': {
                     'tid': self._idx,
                     'idxs': idx,
                     'vals': vals
                 },
                 'book_time': datetime.datetime.now(),
                 'refresh_time': None
                 }
        try:
            loss = self.blackbox(**params)
            trial['result']['loss'] = loss
            trial['result']['status'] = 'ok'
        except Exception as e:
            LOG.error("computing loss failed due to:\n {}".format(e))
            loss = np.nan
            trial['result']['loss'] = np.nan
            trial['result']['status'] = 'failed'
        trial['refresh_time'] = datetime.datetime.now()
        self._trials.trials.append(trial)
        if isinstance(self.blackbox, BlackboxFunction) and self.blackbox.callback_func is not None:
            cbd = copy.deepcopy(params)
            cbd['iterations'] = self._idx
            cbd['loss'] = loss
            cbd['status'] = trial['result']['status']
            self.blackbox.callback_func(**cbd)
        return loss

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
                pbounds[name] = (param["data"][0], param["data"][1])
            else:
                pbounds[name] = (0, len(param["data"])-1)
        return pbounds
