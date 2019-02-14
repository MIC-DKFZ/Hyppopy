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


from hyppopy.projectmanager import ProjectManager

import os
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class Solver(object):
    _name = None
    _solver_plugin = None
    _settings_plugin = None

    def __init__(self):
        pass

    def set_data(self, data):
        self.solver.set_data(data)

    def set_hyperparameters(self, params):
        self.settings.set_hyperparameter(params)

    def set_loss_function(self, loss_func):
        self.solver.set_loss_function(loss_func)

    def run(self):
        if not ProjectManager.is_ready():
            LOG.error("No config data found to initialize PluginSetting object")
            raise IOError("No config data found to initialize PluginSetting object")
        hyps = ProjectManager.get_hyperparameter()
        self.settings.set_hyperparameter(hyps)
        self.solver.settings = self.settings
        self.solver.run()

    def get_results(self):
        return self.solver.get_results()

    @property
    def is_ready(self):
        return self.solver is not None and self.settings is not None

    @property
    def solver(self):
        return self._solver_plugin

    @solver.setter
    def solver(self, value):
        self._solver_plugin = value

    @property
    def settings(self):
        return self._settings_plugin

    @settings.setter
    def settings(self, value):
        self._settings_plugin = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            msg = "Invalid input, str type expected for value, got {} instead".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._name = value

