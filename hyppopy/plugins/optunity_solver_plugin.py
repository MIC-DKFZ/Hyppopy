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

import optunity
from yapsy.IPlugin import IPlugin
from hyppopy.projectmanager import ProjectManager
from hyppopy.solverpluginbase import SolverPluginBase


class optunity_Solver(SolverPluginBase, IPlugin):
    solver_info = None
    trials = None
    best = None
    status = None

    def __init__(self):
        SolverPluginBase.__init__(self)
        LOG.debug("initialized")

    def loss_function(self, **params):
        try:
            loss = self.loss(self.data, params)
            self.status.append('ok')
            return loss
        except Exception as e:
            LOG.error(f"computing loss failed due to:\n {e}")
            self.status.append('fail')
            return 1e9

    def execute_solver(self, parameter):
        LOG.debug(f"execute_solver using solution space:\n\n\t{pformat(parameter)}\n")
        self.status = []
        try:
            self.best, self.trials, self.solver_info = optunity.minimize_structured(f=self.loss_function,
                                                                                    num_evals=ProjectManager.max_iterations,
                                                                                    search_space=parameter)
        except Exception as e:
            LOG.error(f"internal error in optunity.minimize_structured occured. {e}")
            raise BrokenPipeError(f"internal error in optunity.minimize_structured occured. {e}")

    def convert_results(self):
        solution = dict([(k, v) for k, v in self.best.items() if v is not None])

        txt = ""
        txt += 'Solution Optunity Plugin\n========\n'
        txt += "\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items()))
        txt += f"\nSolver used: {self.solver_info['solver_name']}"
        txt += f"\nOptimum: {self.trials.optimum}"
        txt += f"\nIterations used: {self.trials.stats['num_evals']}"
        txt += f"\nDuration: {self.trials.stats['time']} s\n"
        return txt
