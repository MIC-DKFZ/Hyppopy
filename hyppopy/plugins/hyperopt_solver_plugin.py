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

import logging
LOG = logging.getLogger('hyppopy')
from pprint import pformat

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
    from yapsy.IPlugin import IPlugin
except:
    LOG.warning("Hyperopt package not installed, will ignore this plugin!")
    print("Hyperopt package not installed, will ignore this plugin!")

from hyppopy.solverpluginbase import SolverPluginBase


class hyperopt_Solver(SolverPluginBase, IPlugin):
    trials = None
    best = None

    def __init__(self):
        LOG.debug("hyperopt_Solver.__init__()")
        SolverPluginBase.__init__(self)

    def loss_function(self, params):
        try:
            loss = self.loss(self.data, params)
            status = STATUS_OK
        except Exception as e:
            status = STATUS_FAIL
        return {'loss': loss, 'status': status}

    def convert_parameter(self, params):
        LOG.debug(f"convert_parameter({params})")

        self.solution_space = {}
        for name, content in params.items():
            data = None
            domain = None
            domain_fn = None
            for key, value in content.items():
                if key == 'domain':
                    domain = value
                    if value == 'uniform':
                        domain_fn = hp.uniform
                    if value == 'categorical':
                        domain_fn = hp.choice
                if key == 'data':
                    data = value
            if domain == 'categorical':
                self.solution_space[name] = domain_fn(name, data)
            else:
                self.solution_space[name] = domain_fn(name, data[0], data[1])

    def execute_solver(self):
        LOG.debug(f"execute_solver using solution space -> {pformat(self.solution_space)}")
        self.trials = Trials()
        try:
            self.best = fmin(fn=self.loss_function,
                             space=self.solution_space,
                             algo=tpe.suggest,
                             max_evals=50,
                             trials=self.trials)
        except Exception as e:
            LOG.error(f"Internal error in hyperopt.fmin occured. {e}")
            raise BrokenPipeError(f"Internal error in hyperopt.fmin occured. {e}")

    def convert_results(self):
        solution = dict([(k, v) for k, v in self.best.items() if v is not None])
        print('Solution\n========')
        print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))
