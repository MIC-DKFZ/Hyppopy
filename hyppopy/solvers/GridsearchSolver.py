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
import logging
import warnings
import numpy as np
from pprint import pformat

from scipy.stats import norm
from itertools import product
from hyppopy.globals import DEBUGLEVEL, DEFAULTGRIDFREQUENCY
from hyppopy.solvers.HyppopySolver import HyppopySolver, CandidateDescriptor

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def get_uniform_axis_sample(a, b, N, dtype):
    """
    Returns a uniform sample x(n) in the range [a,b] sampled at N pojnts

    :param a: left value range bound
    :param b: right value range bound
    :param N: discretization of intervall [a,b]
    :param dtype: data type

    :return: [list] axis range
    """
    assert a < b, "condition a < b violated!"
    assert isinstance(N, int), "condition N of type int violated!"
    if dtype is int:
        return list(np.linspace(a, b, N).astype(int))
    elif dtype is float:
        return list(np.linspace(a, b, N))
    else:
        raise AssertionError("dtype {} not supported for uniform sampling!".format(dtype))


def get_norm_cdf(N):
    """
    Returns a normed gaussian cdf (range [0,1]) with N sampling points

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
    Returns a function value f(n) where f is a gaussian cdf in range [a, b] and N sampling points

    :param a: left value range bound
    :param b: right value range bound
    :param N: discretization of intervall [a,b]
    :param dtype: data type

    :return: [list] axis range
    """
    assert a < b, "condition a < b violated!"
    assert isinstance(N, int), "condition N of type int violated!"

    data = []
    for n in range(N):
        x = a + get_norm_cdf(N)[n]*(b-a)
        if dtype is int:
            data.append(int(x))
        elif dtype is float:
            data.append(x)
        else:
            raise AssertionError("dtype {} not supported for uniform sampling!".format(dtype))
    return data


def get_logarithmic_axis_sample(a, b, N, dtype):
    """
    Returns a function value f(n) where f is logarithmic function e^x sampling
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

    # convert input range into exponent range
    lexp = np.log(a)
    rexp = np.log(b)
    exp_range = np.linspace(lexp, rexp, N)

    data = []
    for n in range(exp_range.shape[0]):
        x = np.exp(exp_range[n])
        if dtype is int:
            data.append(int(x))
        elif dtype is float:
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
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "normal", "loguniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="frequency", dtype=int)
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

    #RALF: Ich würde allgemein candidaten listen zu dict machen, damit jeder candidate auch
    # eine ID (key) hat, die der Solver vergeben kann. Hier würde ich einfach die stringifizierten params
    # zum key machen oder du führst die CandidateDescritorClass (siehe HyppopySolver.py) als Convinience.
    # TODO: Kommentar wieder entfernen
    def get_candidates(self, searchspace):
        """
        This function converts the searchspace to a candidate_list that can then be used to distribute via MPI.

        :param searchspace: converted hyperparameter space
        """
        candidates_list = list()
        candidates = [x for x in product(*searchspace[1])]
        # [print(c) for c in candidates]
        for c in candidates:
            params = {}
            for name, value in zip(searchspace[0], c):
                params[name] = value
            candidates_list.append(CandidateDescriptor(**params))

        return candidates_list

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and get the output of the latter as input. It's
        purpose is to call the solver libs main optimization function.

        :param searchspace: converted hyperparameter space
        """
        candidates = self.get_candidates(searchspace)

        # RALF: Hier wird get_candidate_list gebraucht um einen loss_function_batch aufzurufen
        # entsprechend muss auch HypopySolver erweitert werden, dass sie die loss_function_batch unterstützen.
        # TODO: Kommentar wieder entfernen
        try:
            self.loss_function_batch(candidates)
        except Exception as e:
            msg = "internal error in grdsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        self.best = self._trials.argmin

    def convert_searchspace(self, hyperparameter):
        """
        The function converts the standard parameter input into a range list depending
        on the domain. These rangelists are later used with itertools product to create
        a paramater space sample of each combination.

        :param hyperparameter: [dict] hyperparameter space

        :return: [list] name and range for each parameter space axis
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        searchspace = [[], []]
        for name, param in hyperparameter.items():
            if param["domain"] != "categorical" and "frequency" not in param.keys():
                param["frequency"] = DEFAULTGRIDFREQUENCY
                warnings.warn("No frequency field found, used default gridsearch frequency {}".format(DEFAULTGRIDFREQUENCY))

            if param["domain"] == "categorical":
                searchspace[0].append(name)
                searchspace[1].append(param["data"])
            elif param["domain"] == "uniform":
                searchspace[0].append(name)
                searchspace[1].append(get_uniform_axis_sample(param["data"][0],
                                                              param["data"][1],
                                                              param["frequency"],
                                                              param["type"]))
            elif param["domain"] == "normal":
                searchspace[0].append(name)
                searchspace[1].append(get_gaussian_axis_sample(param["data"][0],
                                                               param["data"][1],
                                                               param["frequency"],
                                                               param["type"]))
            elif param["domain"] == "loguniform":
                searchspace[0].append(name)
                searchspace[1].append(get_logarithmic_axis_sample(param["data"][0],
                                                                  param["data"][1],
                                                                  param["frequency"],
                                                                  param["type"]))
        return searchspace
