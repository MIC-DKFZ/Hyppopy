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
import copy
import logging
import numpy as np
from pprint import pformat
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

from hyppopy.globals import DEBUGLEVEL
from hyppopy.solver.HyppopySolver import HyppopySolver
from hyppopy.BlackboxFunction import BlackboxFunction

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyperoptSolver(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def loss_function(self, params):
        for name, p in self._searchspace.items():
            if p["domain"] != "categorical":
                if params[name] < p["data"][0]:
                    params[name] = p["data"][0]
                if params[name] > p["data"][1]:
                    params[name] = p["data"][1]
        status = STATUS_FAIL
        try:
            loss = self.blackbox(**params)
            if loss is not None:
                status = STATUS_OK
            else:
                loss = 1e9
        except Exception as e:
            LOG.error("execution of self.blackbox(**params) failed due to:\n {}".format(e))
            status = STATUS_FAIL
            loss = 1e9
        cbd = copy.deepcopy(params)
        cbd['iterations'] = self._trials.trials[-1]['tid'] + 1
        cbd['loss'] = loss
        cbd['status'] = status
        cbd['book_time'] = self._trials.trials[-1]['book_time']
        cbd['refresh_time'] = self._trials.trials[-1]['refresh_time']
        if isinstance(self.blackbox, BlackboxFunction) and self.blackbox.callback_func is not None:
            # cbd = copy.deepcopy(params)
            # cbd['iterations'] = self._trials.trials[-1]['tid'] + 1
            # cbd['loss'] = loss
            # cbd['status'] = status
            self.blackbox.callback_func(**cbd)
        if self._visdom_viewer is not None:
            self._visdom_viewer.update(cbd)
        return {'loss': loss, 'status': status}

    def execute_solver(self, searchspace):
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        self.trials = Trials()

        try:
            self.best = fmin(fn=self.loss_function,
                             space=searchspace,
                             algo=tpe.suggest,
                             max_evals=self.max_iterations,
                             trials=self.trials)
        except Exception as e:
            msg = "internal error in hyperopt.fmin occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)

    def convert_searchspace(self, hyperparameter):
        self._searchspace = hyperparameter
        solution_space = {}
        for name, content in hyperparameter.items():
            param_settings = {'name': name}
            for key, value in content.items():
                if key == 'domain':
                    param_settings['domain'] = value
                elif key == 'data':
                    param_settings['data'] = value
                elif key == 'type':
                    param_settings['dtype'] = value
            solution_space[name] = self.convert(param_settings)
        return solution_space

    def convert(self, param_settings):
        name = param_settings["name"]
        domain = param_settings["domain"]
        dtype = param_settings["dtype"]
        data = param_settings["data"]

        assert isinstance(data, list), "precondition violation. data of type {} not allowed!".format(type(data))
        assert len(data) >= 2, "precondition violation, data must be of length 2, [left_bound, right_bound]"
        assert isinstance(domain, str), "precondition violation. domain of type {} not allowed!".format(type(domain))
        assert isinstance(dtype, str), "precondition violation. dtype of type {} not allowed!".format(type(dtype))

        if domain == "uniform":
            if dtype == "float" or dtype == "double":
                return hp.uniform(name, data[0], data[1])
            elif dtype == "int":
                data = list(np.arange(int(data[0]), int(data[1] + 1)))
                return hp.choice(name, data)
            else:
                msg = "cannot convert the type {} in domain {}".format(dtype, domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif domain == "loguniform":
            if dtype == "float" or dtype == "double":
                if data[0] == 0:
                    data[0] += 1e-23
                assert data[0] > 0, "precondition Violation, a < 0!"
                assert data[0] < data[1], "precondition Violation, a > b!"
                assert data[1] > 0, "precondition Violation, b < 0!"
                lexp = np.log(data[0])
                rexp = np.log(data[1])
                assert lexp is not np.nan, "precondition violation, left bound input error, results in nan!"
                assert rexp is not np.nan, "precondition violation, right bound input error, results in nan!"

                return hp.loguniform(name, lexp, rexp)
            else:
                msg = "cannot convert the type {} in domain {}".format(dtype, domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif domain == "normal":
            if dtype == "float" or dtype == "double":
                mu = (data[1] - data[0]) / 2.0
                sigma = mu / 3
                return hp.normal(name, data[0] + mu, sigma)
            else:
                msg = "cannot convert the type {} in domain {}".format(dtype, domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif domain == "categorical":
            if dtype == 'str':
                return hp.choice(name, data)
            elif dtype == 'bool':
                data = []
                for elem in data:
                    if elem == "true" or elem == "True" or elem == 1 or elem == "1":
                        data.append(True)
                    elif elem == "false" or elem == "False" or elem == 0 or elem == "0":
                        data.append(False)
                    else:
                        msg = "cannot convert the type {} in domain {}, unknown bool type value".format(dtype, domain)
                        LOG.error(msg)
                        raise LookupError(msg)
                return hp.choice(name, data)
        else:
            msg = "Precondition violation, domain named {} not available!".format(domain)
            LOG.error(msg)
            raise IOError(msg)
