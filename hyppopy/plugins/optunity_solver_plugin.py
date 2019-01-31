# -*- coding: utf-8 -*-
#
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

try:
    import optunity
    from yapsy.IPlugin import IPlugin
except:
    LOG.warning("optunity package not installed, will ignore this plugin!")
    print("optunity package not installed, will ignore this plugin!")

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
            self.status.append('fail')
            return 1e9

    def execute_solver(self, parameter):
        LOG.debug(f"execute_solver using solution space:\n\n\t{pformat(parameter)}\n")
        self.status = []
        try:
            self.best, self.trials, self.solver_info = optunity.minimize_structured(f=self.loss_function,
                                                                                    num_evals=50,
                                                                                    search_space=parameter)
        except Exception as e:
            LOG.error(f"internal error in optunity.minimize_structured occured. {e}")
            raise BrokenPipeError(f"internal error in optunity.minimize_structured occured. {e}")

    def convert_results(self):
        solution = dict([(k, v) for k, v in self.best.items() if v is not None])
        print('Solution\n========')
        print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))
        print(f"Solver used: {self.solver_info['solver_name']}")
        print(f"Optimum: {self.trials.optimum}")
        print(f"Iterations used: {self.trials.stats['num_evals']}")
        print(f"Duration: {self.trials.stats['time']} s")
