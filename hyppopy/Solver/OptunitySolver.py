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
import optunity
import datetime
import numpy as np
from pprint import pformat
from hyperopt import Trials
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from hyppopy.solver.HyppopySolver import HyppopySolver
from hyppopy.BlackboxFunction import BlackboxFunction


class OptunitySolver(HyppopySolver):

    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._solver_info = None
        self.opt_trials = None
        self._idx = None

    def loss_function(self, **params):
        self._idx += 1
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
            for key in params.keys():
                if self.project.get_typeof(key) is int:
                    params[key] = int(round(params[key]))
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
            self.best, self.opt_trials, self._solver_info = optunity.minimize_structured(f=self.loss_function,
                                                                                         num_evals=self.max_iterations,
                                                                                         search_space=searchspace)
        except Exception as e:
            LOG.error("internal error in optunity.minimize_structured occured. {}".format(e))
            raise BrokenPipeError("internal error in optunity.minimize_structured occured. {}".format(e))

    def split_categorical(self, pdict):
        categorical = {}
        uniform = {}
        for name, pset in pdict.items():
            for key, value in pset.items():
                if key == 'domain' and value == 'categorical':
                    categorical[name] = pset
                elif key == 'domain':
                    uniform[name] = pset
        return categorical, uniform

    def convert_searchspace(self, hyperparameter):
        solution_space = {}
        # split input in categorical and non-categorical data
        cat, uni = self.split_categorical(hyperparameter)
        # build up dictionary keeping all non-categorical data
        uniforms = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == 'data':
                    if len(value2) == 3:
                        uniforms[key] = value2[0:2]
                    elif len(value2) == 2:
                        uniforms[key] = value2
                    else:
                        raise AssertionError("precondition violation, optunity searchspace needs list with left and right range bounds!")

        if len(cat) == 0:
            return uniforms
        # build nested categorical structure
        inner_level = uniforms
        for key, value in cat.items():
            tmp = {}
            tmp2 = {}
            for key2, value2 in value.items():
                if key2 == 'data':
                    for elem in value2:
                        tmp[elem] = inner_level
            tmp2[key] = tmp
            inner_level = tmp2
        solution_space = tmp2
        return solution_space
