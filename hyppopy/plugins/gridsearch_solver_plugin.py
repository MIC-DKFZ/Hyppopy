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
import time
import logging
from numpy import argmin, argmax, unique
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from pprint import pformat
from yapsy.IPlugin import IPlugin

from hyppopy.helpers import NestedDictUnfolder
from hyppopy.solverpluginbase import SolverPluginBase


class Trials(object):

    def __init__(self):
        self.loss = []
        self.duration = []
        self.status = []
        self.parameter = []
        self.best = None
        self._tick = None

    def start_iteration(self):
        self._tick = time.process_time()

    def stop_iteration(self):
        if self._tick is None:
            return
        self.duration.append(time.process_time()-self._tick)
        self._tick = None

    def set_status(self, status=True):
        self.status.append(status)

    def set_parameter(self, params):
        self.parameter.append(params)

    def set_loss(self, value):
        self.loss.append(value)

    def get(self):
        if len(self.loss) <= 0:
            raise Exception("Empty solver results!")
        if len(self.loss) != len(self.duration) or len(self.loss) != len(self.parameter) or len(self.loss) != len(self.status):
            raise Exception("Inconsistent results in gridsearch solver!")
        best_index = argmin(self.loss)
        best = self.parameter[best_index]
        worst_loss = self.loss[argmax(self.loss)]
        for n in range(len(self.status)):
            if not self.status[n]:
                self.loss[n] = worst_loss

        res = {
            'losses': self.loss,
            'duration': self.duration
        }
        is_string = []
        for key, value in self.parameter[0].items():
            res[key] = []
            if isinstance(value, str):
                is_string.append(key)

        for p in self.parameter:
            for key, value in p.items():
                res[key].append(value)

        for key in is_string:
            uniques = unique(res[key])
            lookup = {}
            for n, p in enumerate(uniques):
                lookup[p] = n
            for n in range(len(res[key])):
                res[key][n] = lookup[res[key][n]]

        return res, best


class gridsearch_Solver(SolverPluginBase, IPlugin):
    trials = None
    best = None

    def __init__(self):
        SolverPluginBase.__init__(self)
        LOG.debug("initialized")

    def blackbox_function(self, params):
        loss = None
        self.trials.set_parameter(params)
        try:
            self.trials.start_iteration()
            loss = self.blackbox_function_template(self.data, params)
            self.trials.stop_iteration()
            if loss is None:
                self.trials.set_status(False)
        except Exception as e:
            LOG.error("execution of self.loss(self.data, params) failed due to:\n {}".format(e))
            self.trials.set_status(False)
            self.trials.stop_iteration()
        self.trials.set_status(True)
        self.trials.set_loss(loss)
        return

    def execute_solver(self, parameter):
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(parameter)))

        self.trials = Trials()
        unfolder = NestedDictUnfolder(parameter)
        parameter_set = unfolder.unfold()
        N = len(parameter_set)
        print("")
        try:
            for n, params in enumerate(parameter_set):
                self.blackbox_function(params)
                print("\r{}% done".format(int(round(100.0/N*n))), end="")
        except Exception as e:
            msg = "internal error in gridsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        print("")

    def convert_results(self):
        return self.trials.get()
