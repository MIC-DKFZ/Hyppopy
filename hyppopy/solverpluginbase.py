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

import abc

import os
import time
import logging
from hyppopy.globals import DEBUGLEVEL
from hyppopy.settingspluginbase import SettingsPluginBase

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class SolverPluginBase(object):
    data = None
    loss = None
    _settings = None
    _name = None

    def __init__(self):
        pass

    @abc.abstractmethod
    def loss_function(self, params):
        raise NotImplementedError('users must define loss_func to use this base class')

    @abc.abstractmethod
    def execute_solver(self):
        raise NotImplementedError('users must define execute_solver to use this base class')

    @abc.abstractmethod
    def convert_results(self):
        raise NotImplementedError('users must define convert_results to use this base class')

    def set_data(self, data):
        self.data = data

    def set_loss_function(self, func):
        self.loss = func

    def get_results(self):
        return self.convert_results()

    def run(self):
        self.execute_solver(self.settings.get_hyperparameter())

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

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        if not isinstance(value, SettingsPluginBase):
            msg = "Invalid input, SettingsPluginBase type expected for value, got {} instead".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._settings = value


