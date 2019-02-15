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

from hyppopy.deepdict import DeepDict
from hyppopy.solverfactory import SolverFactory
from hyppopy.projectmanager import ProjectManager
from hyppopy.globals import SETTINGSCUSTOMPATH, SETTINGSSOLVERPATH

import os
import abc
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class WorkflowBase(object):

    def __init__(self):
        self._solver = SolverFactory.get_solver()

    def run(self, save=True):
        self.setup()
        self.solver.set_loss_function(self.blackbox_function)
        self.solver.run()
        if save:
            self.solver.save_results()
        self.test()

    def get_results(self):
        return self.solver.get_results()

    @abc.abstractmethod
    def setup(self, **kwargs):
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

