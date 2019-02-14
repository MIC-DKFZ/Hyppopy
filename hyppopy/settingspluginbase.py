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
import copy
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from hyppopy.globals import SETTINGSSOLVERPATH, SETTINGSCUSTOMPATH

from hyppopy.deepdict import DeepDict


class SettingsPluginBase(object):
    _data = None
    _name = None

    def __init__(self):
        self._data = {}

    @abc.abstractmethod
    def convert_parameter(self):
        raise NotImplementedError('users must define convert_parameter to use this base class')

    def get_hyperparameter(self):
        return self.convert_parameter(self.data)

    def set_hyperparameter(self, input_data):
        self.data.clear()
        self.data = copy.deepcopy(input_data)

    def read(self, fname):
        self.data.clear()
        self.data.from_file(fname)

    def write(self, fname):
        self.data.to_file(fname)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, dict):
            self._data = value
        elif isinstance(value, DeepDict):
            self._data = value.data
        else:
            raise IOError("unexpected input type({}) for data, needs to be of type dict or DeepDict!".format(type(value)))


    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            LOG.error("Invalid input, str type expected for value, got {} instead".format(type(value)))
            raise IOError("Invalid input, str type expected for value, got {} instead".format(type(value)))
        self._name = value
