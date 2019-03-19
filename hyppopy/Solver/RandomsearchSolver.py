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
import random
import logging
import datetime
import numpy as np
from pprint import pformat
from hyperopt import Trials
from hyppopy.globals import DEBUGLEVEL
from .HyppopySolver import HyppopySolver
from ..BlackboxFunction import BlackboxFunction

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def draw_uniform_sample(param):
    assert param['type'] != 'str', "Cannot sample a string list uniformly!"
    assert param['data'][0] < param['data'][1], "Precondition violation: data[0] > data[1]!"
    s = random.random()
    s *= np.abs(param['data'][1] - param['data'][0])
    s += param['data'][0]
    if param['type'] == 'int':
        s = int(np.round(s))
        if s < param['data'][0]:
            s = int(param['data'][0])
        if s > param['data'][1]:
            s = int(param['data'][1])
    return s


def draw_normal_sample(param):
    mu = (param['data'][1] - param['data'][0]) / 2
    sigma = mu / 3
    s = np.random.normal(loc=param['data'][0] + mu, scale=sigma)
    if s > param['data'][1]:
        s = param['data'][1]
    if s < param['data'][0]:
        s = param['data'][0]
    return s


def draw_loguniform_sample(param):
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
    return random.sample(param['data'], 1)[0]


def draw_sample(param):
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
        if isinstance(self.blackbox, BlackboxFunction) and self.blackbox.callback_func is not None:
            cbd = copy.deepcopy(params)
            cbd['iterations'] = self._tid + 1
            cbd['loss'] = loss
            cbd['status'] = trial['result']['status']
            self.blackbox.callback_func(**cbd)
        return

    def execute_solver(self, searchspace):
        self._tid = 0
        self._trials = Trials()
        N = self.max_iterations
        try:
            for n in range(N):
                params = {}
                for name, p in searchspace.items():
                    params[name] = draw_sample(p)
                self.loss_function(params)
                self._tid += 1
        except Exception as e:
            msg = "internal error in randomsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        self.best = self._trials.argmin

    def convert_searchspace(self, hyperparameter):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        return hyperparameter
