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
from hyperopt import Trials
from scipy.stats import norm
from itertools import product
from hyppopy.globals import DEBUGLEVEL
from .HyppopySolver import HyppopySolver

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
    assert isinstance(N, int), "condition N of type int violated!"
    assert isinstance(dtype, str), "condition type of type str violated!"
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

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._tid = None

    def loss_function(self, params):
        loss = None
        vals = {}
        idx = {}
        for key, value in params.items():
            vals[key] = [value]
            idx[key] = [self._tid]
        trial = {'tid': self._tid,
                 'result': {'loss': None, 'status': 'ok'},
                 'misc': {
                     'tid': self._tid,
                     'idxs': idx,
                     'vals': vals
                 },
                 'book_time': datetime.datetime.now(),
                 'refresh_time': None
                 }
        try:
            loss = self.blackbox(**params)
            if loss is None:
                trial['result']['loss'] = np.nan
                trial['result']['status'] = 'failed'
            else:
                trial['result']['loss'] = loss
        except Exception as e:
            LOG.error("execution of self.blackbox(**params) failed due to:\n {}".format(e))
            trial['result']['loss'] = np.nan
            trial['result']['status'] = 'failed'
        trial['refresh_time'] = datetime.datetime.now()
        self._trials.trials.append(trial)
        if self.blackbox.callback_func is not None:
            cbd = copy.deepcopy(params)
            cbd['iterations'] = self._tid + 1
            cbd['loss'] = loss
            cbd['status'] = trial['result']['status']
            self.blackbox.callback_func(**cbd)
        return

    def execute_solver(self, searchspace):
        self._tid = 0
        self._trials = Trials()

        for x in product(*searchspace[1]):
            params = {}
            for name, value in zip(searchspace[0], x):
                params[name] = value
            try:
                self.loss_function(params)
                self._tid += 1
            except Exception as e:
                msg = "internal error in randomsearch execute_solver occured. {}".format(e)
                LOG.error(msg)
                raise BrokenPipeError(msg)
        self.best = self._trials.argmin

    def convert_searchspace(self, hyperparameter):
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




# def get_uniform_axis_sample(n, a, b, N):
#     """
#     returns a uniform sample x(n) in the range [a,b] sampled at N pojnts
#     :param n: input position within range [0,N]
#     :param a: left value range bound
#     :param b: right value range bound
#     :param N: discretization of intervall [a,b]
#     :return: [float] x(n)
#     """
#     assert a < b, "condition a < b violated!"
#     assert n >= 0, "condition n >= 0 violated!"
#     assert n < N, "condition n < N violated!"
#     assert isinstance(n, int), "condition n of type int violated!"
#     assert isinstance(N, int), "condition N of type int violated!"
#     return np.linspace(a, b, N)[n]
#
#
# def get_norm_cdf(N):
#     """
#     returns a normed gaussian cdf (range [0,1]) with N sampling points
#     :param N: sampling points
#     :return: [ndarray] gaussian cdf function values
#     """
#     assert isinstance(N, int), "condition N of type int violated!"
#     even = True
#     if N % 2 != 0:
#         N -= 1
#         even = False
#     N = int(N/2)
#     sigma = 1/3
#     x = np.linspace(0, 1, N)
#     y1 = norm.cdf(x, loc=0, scale=sigma)-0.5
#     if not even:
#         y1 = np.append(y1, [0.5])
#     y2 = 1-(norm.cdf(x, loc=0, scale=sigma)-0.5)
#     y2 = np.flip(y2, axis=0)
#     y = np.concatenate((y1, y2), axis=0)
#     return y
#
#
# def get_gaussian_axis_sample(n, a, b, N):
#     """
#     returns a function value f(n) where f is a gaussian cdf in range [a, b] and N sampling points
#     :param n: input position within range [0,N]
#     :param a: left value range bound
#     :param b: right value range bound
#     :param N: discretization of intervall [a,b]
#     :return: [float] f(n)
#     """
#     assert a < b, "condition a < b violated!"
#     assert n >= 0, "condition n >= 0 violated!"
#     assert n < N, "condition n < N violated!"
#     assert isinstance(n, int), "condition n of type int violated!"
#     assert isinstance(N, int), "condition N of type int violated!"
#     return a + get_norm_cdf(N)[n]*(b-a)
#
#
# def get_logarithmic_axis_sample(n, a, b, N):
#     """
#     returns a function value f(n) where f is logarithmic function e^x sampling
#     the exponent range [log(a), log(b)] linear at N sampling points.
#     The function values returned are in the range [a, b].
#     :param n: sampling point [0, N-1]
#     :param a: left range bound
#     :param b: right range bound
#     :param N: discretization of intervall [log(a),log(b)]
#     :return: [float] f(x)
#     """
#     assert a < b, "condition a < b violated!"
#     assert n >= 0, "condition n >= 0 violated!"
#     assert n < N, "condition n < N violated!"
#     assert isinstance(n, int), "condition n of type int violated!"
#     assert isinstance(N, int), "condition N of type int violated!"
#     lexp = np.log(a)
#     rexp = np.log(b)
#     exp_range = np.linspace(lexp, rexp, N)
#     return np.exp(exp_range[n])
#
#
# class GridAxis(object):
#     _data = None
#     _name = None
#     _type = None
#     _domain = None
#     _sampling = None
#     _is_categorical = False
#     _current_pos = 0
#
#     def __init__(self, name, param):
#         self._name = name
#         self._domain = param["domain"]
#         self.data = param["data"]
#         self.type = param["type"]
#         if param["domain"] == "categorical":
#             self._is_categorical = True
#
#     def elems_left(self):
#         return self._sampling - self._current_pos - 1
#
#     def increment(self):
#         self._current_pos += 1
#         if self._current_pos > self._sampling - 1:
#             self._current_pos = 0
#
#     def get_value(self):
#         if self._domain == "categorical":
#             return self.data[self._current_pos]
#         elif self._domain == "uniform":
#             return get_uniform_axis_sample(self._current_pos, self.data[0], self.data[1], self._sampling)
#         elif self._domain == "normal":
#             return get_gaussian_axis_sample(self._current_pos, self.data[0], self.data[1], self._sampling)
#         elif self._domain == "loguniform":
#             return get_logarithmic_axis_sample(self._current_pos, self.data[0], self.data[1], self._sampling)
#
#     @property
#     def name(self):
#         return self._name
#
#     @property
#     def data(self):
#         return self._data
#
#     @data.setter
#     def data(self, value):
#         if self._domain == "categorical":
#             assert len(value) > 0, "Precondition violation, empty data cannot be handled!"
#             self._data = value
#             self._sampling = len(value)
#         else:
#             assert len(value) == 3, "precondition violation, gridsearch axis needs low, high and sampling value!"
#             self._data = value[0:2]
#             self._sampling = value[2]
#
#     @property
#     def sampling(self):
#         return self._sampling
#
#     @property
#     def type(self):
#         return self._type
#
#     @type.setter
#     def type(self, value):
#         assert isinstance(value, str), "precondition violation, value expects a str!"
#         if value == "str":
#             self._type = str
#         elif value == "int":
#             self._type = int
#         if value == "float" or value == "double":
#             self._type = float
#
#
# class GridSampler(object):
#
#     def __init__(self):
#         self._axis = []
#         self._loops = []
#
#     def get_gridsize(self):
#         n = 1
#         for ax in self._axis:
#             n *= ax.sampling
#         return n
#
#     def add_axis(self, axis):
#         self._axis.append(axis)
#         self.update_loops()
#
#     def update_loops(self):
#         if len(self._axis) == 1:
#             self._loops.append(1)
#         else:
#             lens = []
#             for ax in self._axis:
#                 lens.append(ax.sampling)
#             self._loops.append(np.cumprod(lens))
#
#     def get_sample(self):
#         sample = []
#         for ax in self._axis:
#             sample.append(ax.get_value())
#         return sample
#
#
# class GridsearchSolver(HyppopySolver):
#
#     def __init__(self, project=None):
#         HyppopySolver.__init__(self, project)
#         self._tid = None
#
#     def loss_function(self, params):
#         loss = None
#         vals = {}
#         idx = {}
#         for key, value in params.items():
#             vals[key] = [value]
#             idx[key] = [self._tid]
#         trial = {'tid': self._tid,
#                  'result': {'loss': None, 'status': 'ok'},
#                  'misc': {
#                      'tid': self._tid,
#                      'idxs': idx,
#                      'vals': vals
#                  },
#                  'book_time': datetime.datetime.now(),
#                  'refresh_time': None
#                  }
#         try:
#             loss = self.blackbox(**params)
#             if loss is None:
#                 trial['result']['loss'] = np.nan
#                 trial['result']['status'] = 'failed'
#             else:
#                 trial['result']['loss'] = loss
#         except Exception as e:
#             LOG.error("execution of self.blackbox(**params) failed due to:\n {}".format(e))
#             trial['result']['loss'] = np.nan
#             trial['result']['status'] = 'failed'
#         trial['refresh_time'] = datetime.datetime.now()
#         self._trials.trials.append(trial)
#         if self.blackbox.callback_func is not None:
#             cbd = copy.deepcopy(params)
#             cbd['iterations'] = self._tid + 1
#             cbd['loss'] = loss
#             cbd['status'] = trial['result']['status']
#             self.blackbox.callback_func(**cbd)
#         return
#
#     def execute_solver(self, searchspace):
#         self._tid = 0
#         self._trials = Trials()
#
#         while True:
#             params = {}
#             for axis in searchspace:
#                 params[axis.name] = axis.next()
#                 if params[axis.name] is None:
#                     break
#             try:
#                 self.loss_function(params)
#                 self._tid += 1
#             except Exception as e:
#                 msg = "internal error in randomsearch execute_solver occured. {}".format(e)
#                 LOG.error(msg)
#                 raise BrokenPipeError(msg)
#         self.best = self._trials.argmin
#
#     def convert_searchspace(self, hyperparameter):
#         searchspace = []
#         for name, param in hyperparameter.items():
#             if param["domain"] != "categorical":
#                 searchspace.append(GridAxis(name, param))
#         for name, param in hyperparameter.items():
#             if param["domain"] == "categorical":
#                 searchspace.append(GridAxis(name, param))
#         searchspace[-1].is_looping = False
#         return searchspace
