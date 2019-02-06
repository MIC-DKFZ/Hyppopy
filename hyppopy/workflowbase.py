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

import hyppopy.solverfactory as sfac
from hyppopy.deepdict import DeepDict
from hyppopy.globals import SETTINGSPATH

import os
import abc
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class Workflow(object):
    _solver = None
    _args = None

    def __init__(self, args):
        self._args = args
        factory = sfac.SolverFactory.instance()
        if args.plugin is None or args.plugin == '':
            dd = DeepDict(args.config)
            ppath = "use_plugin"
            if not dd.has_section(ppath):
                msg = f"invalid config file, missing section {ppath}"
                LOG.error(msg)
                raise LookupError(msg)
            plugin = dd[SETTINGSPATH+'/'+ppath]
        else:
            plugin = args.plugin
        self._solver = factory.get_solver(plugin)
        self.solver.read_parameter(args.config)

    def run(self):
        self.setup()
        self.solver.set_loss_function(self.blackbox_function)
        self.solver.run()
        self.test()

    def get_results(self):
        return self.solver.get_results()

    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError('the user has to implement this function')

    @abc.abstractmethod
    def blackbox_function(self):
        raise NotImplementedError('the user has to implement this function')

    @abc.abstractmethod
    def test(self):
        pass

    @property
    def solver(self):
        return self._solver

    @property
    def args(self):
        return self._args
