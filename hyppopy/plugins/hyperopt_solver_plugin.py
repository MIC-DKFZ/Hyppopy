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
        # currently converting results in a way that this function returns a dict
        # keeping all useful parameter as key/list item. This will be automatically
        # converted to a pandas dataframe in the solver class
        results = {'timing ms': [], 'losses': []}
        pset = self.trials.trials[0]['misc']['vals']
        for p in pset.keys():
            results[p] = []

        for n, trial in enumerate(self.trials.trials):
            t1 = trial['book_time']
            t2 = trial['refresh_time']
            results['timing ms'].append((t2 - t1).microseconds/1000.0)
            results['losses'].append(trial['result']['loss'])
            pset = trial['misc']['vals']
            for p in pset.items():
                results[p[0]].append(p[1][0])
        return results, self.best
