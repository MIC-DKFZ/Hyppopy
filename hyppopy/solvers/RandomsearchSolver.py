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

__all__ = ['RandomsearchSolver',
           'draw_uniform_sample',
           'draw_normal_sample',
           'draw_loguniform_sample',
           'draw_categorical_sample',
           'draw_sample']

import os
import copy
import random
import logging
import numpy as np
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def draw_uniform_sample(param):
    """
    Function draws a random sample from a uniform range

    :param param: [dict] input hyperparameter discription

    :return: random sample value of type data['type']
    """
    assert param['type'] is not str, "cannot sample a string list!"
    assert param['data'][0] < param['data'][1], "precondition violation: data[0] > data[1]!"
    s = random.random()
    s *= np.abs(param['data'][1] - param['data'][0])
    s += param['data'][0]
    if param['type'] is int:
        s = int(np.round(s))
        if s < param['data'][0]:
            s = int(param['data'][0])
        if s > param['data'][1]:
            s = int(param['data'][1])
    return s


def draw_normal_sample(param):
    """
    Function draws a random sample from a normal distributed range

    :param param: [dict] input hyperparameter discription

    :return: random sample value of type data['type']
    """
    assert param['type'] is not str, "cannot sample a string list!"
    assert param['data'][0] < param['data'][1], "precondition violation: data[0] > data[1]!"
    mu = (param['data'][1] - param['data'][0]) / 2
    sigma = mu / 3
    s = np.random.normal(loc=param['data'][0] + mu, scale=sigma)
    if s > param['data'][1]:
        s = param['data'][1]
    if s < param['data'][0]:
        s = param['data'][0]
    s = float(s)
    if param["type"] is int:
        s = int(np.round(s))
    return s


def draw_loguniform_sample(param):
    """
    Function draws a random sample from a logarithmic distributed range

    :param param: [dict] input hyperparameter discription

    :return: random sample value of type data['type']
    """
    assert param['type'] is not str, "cannot sample a string list!"
    assert param['data'][0] < param['data'][1], "precondition violation: data[0] > data[1]!"
    p = copy.deepcopy(param)
    p['data'][0] = np.log(param['data'][0])
    p['data'][1] = np.log(param['data'][1])
    assert p['data'][0] is not np.nan, "Precondition violation, left bound input error, results in nan!"
    assert p['data'][1] is not np.nan, "Precondition violation, right bound input error, results in nan!"
    x = draw_uniform_sample(p)
    s = np.exp(x)
    if s > param['data'][1]:
        s = param['data'][1]
    if s < param['data'][0]:
        s = param['data'][0]
    return s


def draw_categorical_sample(param):
    """
    Function draws a random sample from a categorical list

    :param param: [dict] input hyperparameter discription

    :return: random sample value of type data['type']
    """
    return random.sample(param['data'], 1)[0]


def draw_sample(param):
    """
    Function draws a sample from the input hyperparameter descriptor depending on it's domain

    :param param: [dict] input hyperparameter discription

    :return: random sample value of type data['type']
    """
    assert isinstance(param, dict), "input error, hyperparam descriptors of type {} not allowed!".format(type(param))
    if param['domain'] == "uniform":
        return draw_uniform_sample(param)
    elif param['domain'] == "normal":
        return draw_normal_sample(param)
    elif param['domain'] == "loguniform":
        return draw_loguniform_sample(param)
    elif param['domain'] == "categorical":
        return draw_categorical_sample(param)
    else:
        raise LookupError("Unknown domain {}".format(param['domain']))


class RandomsearchSolver(HyppopySolver):
    """
    The RandomsearchSolver class implements a randomsearch optimization. The randomsearch supports
    categorical, uniform, normal and loguniform sampling. The solver draws an independent sample
    from the parameter space each iteration.
    """
    def __init__(self, project=None):
        """
        The constructor accepts a HyppopyProject.

        :param project: [HyppopyProject] project instance, default=None
        """
        HyppopySolver.__init__(self, project)

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

    def loss_function_call(self, params):
        """
        This function is called within the function loss_function and encapsulates the actual blackbox function call
        in each iteration. The function loss_function takes care of the iteration driving and reporting, but each solver
        lib might need some special treatment between the parameter set selection and the calling of the actual blackbox
        function, e.g. parameter converting.

        :param params: [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...}

        :return: [float] loss
        """
        loss = self.blackbox(**params)
        if loss is None:
            return np.nan
        return loss

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and get the output of the latter as input. It's
        purpose is to call the solver libs main optimization function.

        :param searchspace: converted hyperparameter space
        """
        N = self.max_iterations
        try:
            for n in range(N):
                params = {}
                for name, p in searchspace.items():
                    params[name] = draw_sample(p)
                self.loss_function(**params)
        except Exception as e:
            msg = "internal error in randomsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        self.best = self._trials.argmin

    def convert_searchspace(self, hyperparameter):
        """
        This function gets the unified hyppopy-like parameterspace description as input and, if necessary, should
        convert it into a solver lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':'float'}, ...}

        :return: [object] converted hyperparameter space
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        return hyperparameter
