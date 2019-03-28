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
import numpy as np
from pprint import pformat
from scipy.stats import norm
from itertools import product
from hyppopy.globals import DEBUGLEVEL
from hyppopy.solver.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def get_uniform_axis_sample(a, b, N, dtype):
    """
    returns a uniform sample x(n) in the range [a,b] sampled at N pojnts
    :param a: left value range bound
    :param b: right value range bound
    :param N: discretization of intervall [a,b]
    :param dtype: data type
    :return: [list] axis range
    """
    assert a < b, "condition a < b violated!"
    assert isinstance(N, int), "condition N of type int violated!"
    assert isinstance(dtype, str), "condition type of type str violated!"
    if dtype == "int":
        return list(np.linspace(a, b, N).astype(int))
    elif dtype == "float" or dtype == "double":
        return list(np.linspace(a, b, N))
    else:
        raise AssertionError("dtype {} not supported for uniform sampling!".format(dtype))


def get_norm_cdf(N):
    """
    returns a normed gaussian cdf (range [0,1]) with N sampling points
    :param N: sampling points
    :return: [ndarray] gaussian cdf function values
    """
    assert isinstance(N, int), "condition N of type int violated!"
    even = True
    if N % 2 != 0:
        N -= 1
        even = False
    N = int(N/2)
    sigma = 1/3
    x = np.linspace(0, 1, N)
    y1 = norm.cdf(x, loc=0, scale=sigma)-0.5
    if not even:
        y1 = np.append(y1, [0.5])
    y2 = 1-(norm.cdf(x, loc=0, scale=sigma)-0.5)
    y2 = np.flip(y2, axis=0)
    y = np.concatenate((y1, y2), axis=0)
    return y


def get_gaussian_axis_sample(a, b, N, dtype):
    """
    returns a function value f(n) where f is a gaussian cdf in range [a, b] and N sampling points
    :param a: left value range bound
    :param b: right value range bound
    :param N: discretization of intervall [a,b]
    :param dtype: data type
    :return: [list] axis range
    """
    assert a < b, "condition a < b violated!"
    assert isinstance(N, int), "condition N of type int violated!"
    assert isinstance(dtype, str), "condition type of type str violated!"

    data = []
    for n in range(N):
        x = a + get_norm_cdf(N)[n]*(b-a)
        if dtype == "int":
            data.append(int(x))
        elif dtype == "float" or dtype == "double":
            data.append(x)
        else:
            raise AssertionError("dtype {} not supported for uniform sampling!".format(dtype))
    return data


def get_logarithmic_axis_sample(a, b, N, dtype):
    """
    returns a function value f(n) where f is logarithmic function e^x sampling
    the exponent range [log(a), log(b)] linear at N sampling points.
    The function values returned are in the range [a, b].
    :param a: left value range bound
    :param b: right value range bound
    :param N: discretization of intervall [a,b]
    :param dtype: data type
    :return: [list] axis range
    """
    assert a < b, "condition a < b violated!"
    assert a > 0, "condition a > 0 violated!"
    assert isinstance(N, int), "condition N of type int violated!"
    assert isinstance(dtype, str), "condition type of type str violated!"

    # convert input range into exponent range
    lexp = np.log(a)
    rexp = np.log(b)
    exp_range = np.linspace(lexp, rexp, N)

    data = []
    for n in range(exp_range.shape[0]):
        x = np.exp(exp_range[n])
        if dtype == "int":
            data.append(int(x))
        elif dtype == "float" or dtype == "double":
            data.append(x)
        else:
            raise AssertionError("dtype {} not supported for uniform sampling!".format(dtype))
    return data


class GridsearchSolver(HyppopySolver):
    """
    The GridsearchSolver class implements a gridsearch optimization. The gridsearch supports
    categorical, uniform, normal and loguniform sampling. To use the GridsearchSolver, besides
    a range, one must specifiy the number of samples in the domain, e.g. 'data': [0, 1, 100]
    """
    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._has_maxiteration_field = False

    def loss_function_call(self, params):
        loss = self.blackbox(**params)
        if loss is None:
            return np.nan
        return loss

    def execute_solver(self, searchspace):
        for x in product(*searchspace[1]):
            params = {}
            for name, value in zip(searchspace[0], x):
                params[name] = value
            try:
                self.loss_function(**params)
            except Exception as e:
                msg = "internal error in randomsearch execute_solver occured. {}".format(e)
                LOG.error(msg)
                raise BrokenPipeError(msg)
        self.best = self._trials.argmin

    def convert_searchspace(self, hyperparameter):
        """
        the function converts the standard parameter input into a range list depending
        on the domain. These rangelists are later used with itertools product to create
        a paramater space sample of each combination.
        :param hyperparameter: [dict] hyperparameter space
        :return: [list] name and range for each parameter space axis
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        searchspace = [[], []]
        for name, param in hyperparameter.items():
            if param["domain"] == "categorical":
                searchspace[0].append(name)
                searchspace[1].append(param["data"])
            elif param["domain"] == "uniform":
                searchspace[0].append(name)
                searchspace[1].append(get_uniform_axis_sample(param["data"][0], param["data"][1], param["data"][2], param["type"]))
            elif param["domain"] == "normal":
                searchspace[0].append(name)
                searchspace[1].append(get_gaussian_axis_sample(param["data"][0], param["data"][1], param["data"][2], param["type"]))
            elif param["domain"] == "loguniform":
                searchspace[0].append(name)
                searchspace[1].append(get_logarithmic_axis_sample(param["data"][0], param["data"][1], param["data"][2], param["type"]))
        return searchspace
