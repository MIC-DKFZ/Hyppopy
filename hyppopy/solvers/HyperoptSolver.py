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

import os
import copy
import logging
import numpy as np
from pprint import pformat
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver
from hyppopy.BlackboxFunction import BlackboxFunction

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyperoptSolver(HyppopySolver):

    def __init__(self, project=None):
        """
        The constructor accepts a HyppopyProject.

        :param project: [HyppopyProject] project instance, default=None
        """
        HyppopySolver.__init__(self, project)
        self._searchspace = None

    def define_interface(self):
        """
        This function is called when HyppopySolver.__init__ function finished. Child classes need to define their
        individual parameter here by calling the _add_member function for each class member variable need to be defined.
        Using _add_hyperparameter_signature the structure of a hyperparameter the solver expects must be defined.
        Both, members and hyperparameter signatures are later get checked, before executing the solver, ensuring
        settings passed fullfill solver needs.
        """
        self._add_member("max_iterations", int)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "normal", "loguniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)

    def clamp_parameters(self, params):
        """
        helper function, that ensures that all non categorical parameters
        are within there defined bounds.
        It seems that this is not always ensured by hyperopt.

        :param params: [dict] hyperparameter set

        :return: [dict] clamped hyperparameters as a copy
        """
        clamped_params = params.copy()
        for name, p in self._searchspace.items():
            if p["domain"] != "categorical":
                if clamped_params[name] < p["data"][0]:
                    clamped_params[name] = p["data"][0]
                if clamped_params[name] > p["data"][1]:
                    clamped_params[name] = p["data"][1]
        return clamped_params

    def loss_function(self, params):
        """
        Loss function wrapper function.

        :param params: [dict] hyperparameter set

        :return: [float] loss
        """
        params = self.clamp_parameters(params)
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
            self.blackbox.callback_func(**cbd)
        if self._visdom_viewer is not None:
            self._visdom_viewer.update(cbd)
        return {'loss': loss, 'status': status}

    def loss_func_cand_preprocess(self, params):
        """
        Loss function wrapper function.

        :param params: [dict] hyperparameter set

        :return: [float] loss
        """
        return self.clamp_parameters(params)

    def loss_func_postprocess(self, loss):
        """
        Loss function wrapper function.

        :param params: [dict] hyperparameter set

        :return: [float] loss
        """

        if loss is not None:
            status = STATUS_OK
        else:
            loss = 1e9

        # return {'loss': loss, 'status': status}
        return loss

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and get the output of the latter as input. It's
        purpose is to call the solver libs main optimization function.

        :param searchspace: converted hyperparameter space
        """
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        self.trials = Trials()

        try:
            hyperopt_best = fmin(fn=self.loss_function,
                             space=searchspace,
                             algo=tpe.suggest,
                             max_evals=self.max_iterations,
                             trials=self.trials)
            self.best = self.convert_params_from_hyperopt(hyperopt_best)
        except Exception as e:
            msg = "internal error in hyperopt.fmin occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)

    def convert_searchspace(self, hyperparameter):
        """
        This function gets the unified hyppopy-like parameterspace description as input and, if necessary, should
        convert it into a solver lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':'float'}, ...}

        :return: [object] converted hyperparameter space
        """
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
        """
        Convert searchspace to hyperopt specific searchspace

        :param param_settings: [dict] hyperparameter description

        :return: [object] hyperopt description
        """
        name = param_settings["name"]
        domain = param_settings["domain"]
        dtype = param_settings["dtype"]
        data = param_settings["data"]

        if domain == "uniform":
            if dtype is float:
                return hp.uniform(name, data[0], data[1])
            elif dtype is int:
                data = list(np.arange(int(data[0]), int(data[1] + 1)))
                return hp.choice(name, data)
            else:
                msg = "cannot convert the type {} in domain {}".format(dtype, domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif domain == "loguniform":
            if dtype is float:
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
            if dtype is float:
                mu = (data[1] - data[0]) / 2.0
                sigma = mu / 3
                return hp.normal(name, data[0] + mu, sigma)
            else:
                msg = "cannot convert the type {} in domain {}".format(dtype, domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif domain == "categorical":
            if dtype is str:
                return hp.choice(name, data)
            elif dtype is bool:
                conv = []
                for elem in data:
                    if elem == "true" or elem == "True" or elem == 1 or elem == "1" or elem == True:
                        conv.append(True)
                    elif elem == "false" or elem == "False" or elem == 0 or elem == "0" or elem == False:
                        conv.append(False)
                    else:
                        msg = "cannot convert the type {} in domain {}, unknown bool type value".format(dtype, domain)
                        LOG.error(msg)
                        raise LookupError(msg)
                return hp.choice(name, conv)
        else:
            msg = "Precondition violation, domain named {} not available!".format(domain)
            LOG.error(msg)
            raise IOError(msg)

    def convert_params_from_hyperopt(self, hyperopt_params):
        """
        Convert params of hyperopt search space into gneral hyppopy specification

        :param hyperopt_params: [dict] hyperopt parameter

        :return: [object] hyppopy parameters
        """
        result = self.clamp_parameters(hyperopt_params)

        for name, p in self._searchspace.items():
            if p["domain"] == "categorical":
                #convert from index (used hyperopt intern) to categorical value
                result[name] = p['data'][result[name]]

        return result
