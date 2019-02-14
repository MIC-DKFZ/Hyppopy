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
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from yapsy.IPlugin import IPlugin


from hyppopy.projectmanager import ProjectManager
from hyppopy.solverpluginbase import SolverPluginBase


class hyperopt_Solver(SolverPluginBase, IPlugin):
    trials = None
    best = None

    def __init__(self):
        SolverPluginBase.__init__(self)
        LOG.debug("initialized")

    def loss_function(self, params):
        try:
            loss = self.loss(self.data, params)
            status = STATUS_OK
        except Exception as e:
            LOG.error("execution of self.loss(self.data, params) failed due to:\n {}".format(e))
            status = STATUS_FAIL
        return {'loss': loss, 'status': status}

    def execute_solver(self, parameter):
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(parameter)))
        self.trials = Trials()

        try:
            self.best = fmin(fn=self.loss_function,
                             space=parameter,
                             algo=tpe.suggest,
                             max_evals=ProjectManager.max_iterations,
                             trials=self.trials)
        except Exception as e:
            msg = "internal error in hyperopt.fmin occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)

    def convert_results(self):
        txt = ""
        solution = dict([(k, v) for k, v in self.best.items() if v is not None])
        txt += 'Solution Hyperopt Plugin\n========\n'
        txt += "\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items()))
        txt += "\n"
        return txt
