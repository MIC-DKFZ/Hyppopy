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


class Solver(object):
    _name = None
    _solver_plugin = None
    _settings_plugin = None

    def __init__(self):
        pass

    def set_data(self, data):
        self.solver.set_data(data)

    def set_parameters(self, params):
        self.settings.set(params)
        self.settings.set_attributes(self.solver)
        self.settings.set_attributes(self.settings)

    def read_parameter(self, fname):
        self.settings.read(fname)
        self.settings.set_attributes(self.solver)
        self.settings.set_attributes(self.settings)

    def set_loss_function(self, loss_func):
        self.solver.set_loss_function(loss_func)

    def run(self):
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
            LOG.error(f"Invalid input, str type expected for value, got {type(value)} instead")
            raise IOError(f"Invalid input, str type expected for value, got {type(value)} instead")
        self._name = value

