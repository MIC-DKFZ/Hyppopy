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
import numpy as np
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from pprint import pformat
from yapsy.IPlugin import IPlugin

from hyppopy.helpers import Trials
from hyppopy.globals import DEFAULTITERATIONS
from hyppopy.projectmanager import ProjectManager
from hyppopy.solverpluginbase import SolverPluginBase


def drawUniformSample(param):
    assert param['type'] != 'str', "Cannot sample a string list uniformly!"
    assert param['data'][0] < param['data'][1], "Precondition violation: data[0] > data[1]!"
    s = random.random()
    s *= np.abs(param['data'][1]-param['data'][0])
    s += param['data'][0]
    if param['type'] == 'int':
        s = int(np.round(s))
        if s < param['data'][0]:
            s = int(param['data'][0])
        if s > param['data'][1]:
            s = int(param['data'][1])
    return s


def drawNormalSample(param):
    mu = (param['data'][1]-param['data'][0])/2
    sigma = mu/3
    s = np.random.normal(loc=param['data'][0] + mu, scale=sigma)
    return s


def drawLoguniformSample(param):
    p = copy.deepcopy(param)
    p['data'][0] = np.log(param['data'][0])
    p['data'][1] = np.log(param['data'][1])
    assert p['data'][0] is not np.nan, "Precondition violation, left bound input error, results in nan!"
    assert p['data'][1] is not np.nan, "Precondition violation, right bound input error, results in nan!"
    x = drawUniformSample(p)
    s = np.exp(x)
    return s


def drawCategoricalSample(param):
    return random.sample(param['data'], 1)[0]


def drawSample(param):
    if param['domain'] == "uniform":
        return drawUniformSample(param)
    elif param['domain'] == "normal":
        return drawNormalSample(param)
    elif param['domain'] == "loguniform":
        return drawLoguniformSample(param)
    elif param['domain'] == "categorical":
        return drawCategoricalSample(param)
    else:
        raise LookupError("Unknown domain {}".format(param['domain']))


class randomsearch_Solver(SolverPluginBase, IPlugin):
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
                self.trials.stop_iteration()
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
        if 'max_iterations' not in ProjectManager.__dict__:
            msg = "Missing max_iteration entry in config, used default {}!".format(DEFAULTITERATIONS)
            LOG.warning(msg)
            print("WARNING: {}".format(msg))
            setattr(ProjectManager, 'max_iterations', DEFAULTITERATIONS)
        N = ProjectManager.max_iterations
        #print("")
        try:
            for n in range(N):
                params = {}
                for name, p in parameter.items():
                    params[name] = drawSample(p)
                self.blackbox_function(params)
                #print("\r{}% done".format(int(round(100.0 / N * n))), end="")
        except Exception as e:
            msg = "internal error in randomsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        #print("\r{}% done".format(100), end="")
        #print("")

    def convert_results(self):
        return self.trials.get()
