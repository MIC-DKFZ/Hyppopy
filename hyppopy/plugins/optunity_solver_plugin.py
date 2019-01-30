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

    def convert_parameter(self, params):
        LOG.debug(f"convert input parameter\n\n\t{pformat(params)}\n")

        # define function spliting input dict
        # into categorical and non-categorical
        def split_categorical(pdict):
            categorical = {}
            uniform = {}
            for name, pset in pdict.items():
                for key, value in pset.items():
                    if key == 'domain' and value == 'categorical':
                        categorical[name] = pset
                    elif key == 'domain':
                        uniform[name] = pset
            return categorical, uniform

        self.solution_space = {}
        # split input in categorical and non-categorical data
        cat, uni = split_categorical(params)
        # build up dictionary keeping all non-categorical data
        uniforms = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == 'data':
                    uniforms[key] = value2

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
        self.solution_space = tmp2

    def execute_solver(self):
        LOG.debug(f"execute_solver using solution space:\n\n\t{pformat(self.solution_space)}\n")
        self.status = []
        try:
            self.best, self.trials, self.solver_info = optunity.minimize_structured(f=self.loss_function,
                                                                                    num_evals=50,
                                                                                    search_space=self.solution_space)
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
