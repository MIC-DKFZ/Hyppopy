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
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from pprint import pformat
from yapsy.IPlugin import IPlugin

from hyppopy.helpers import Trials
from hyppopy.projectmanager import ProjectManager
from hyppopy.solverpluginbase import SolverPluginBase


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
        N = ProjectManager.max_iterations
        print("")
        try:
            for n in range(N):
                params = {}
                for key, value in parameter.items():
                    params[key] = value[n]
                self.blackbox_function(params)
                print("\r{}% done".format(int(round(100.0 / N * n))), end="")
        except Exception as e:
            msg = "internal error in randomsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        print("\r{}% done".format(100), end="")
        print("")

    def convert_results(self):
        return self.trials.get()
