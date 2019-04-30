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
    function draws a random sample from a uniform range
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
    function draws a random sample from a normal distributed range
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
    function draws a random sample from a logarithmic distributed range
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
    function draws a random sample from a categorical list
    :param param: [dict] input hyperparameter discription
    :return: random sample value of type data['type']
    """
    return random.sample(param['data'], 1)[0]


def draw_sample(param):
    """
    function draws a sample from the input hyperparameter descriptor depending on it's domain
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
    from the parameter space each iteration."""
    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)

    def define_interface(self):
        self.add_member("max_iterations", int)
        self.add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "normal", "loguniform", "categorical"])
        self.add_hyperparameter_signature(name="data", dtype=list)
        self.add_hyperparameter_signature(name="type", dtype=type)

    def loss_function_call(self, params):
        loss = self.blackbox(**params)
        if loss is None:
            return np.nan
        return loss

    def execute_solver(self, searchspace):
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
        this function simply pipes the input parameter through, the sample
        drawing functions are responsible for interpreting the parameter.
        :param hyperparameter: [dict] hyperparameter space
        :return: [dict] hyperparameter space
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        return hyperparameter
