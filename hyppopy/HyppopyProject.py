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

from .globals import *

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyppopyProject(object):

    def __init__(self, config=None):
        self._hyperparameter = None
        self._settings = None
        self._extmembers = []
        if config is not None:
            self.set_config(config)

    def clear(self):
        self._hyperparameter = None
        self._settings = None
        for added in self._extmembers:
            if added in self.__dict__.keys():
                del self.__dict__[added]
        self._extmembers = []

    def set_config(self, config):
        self.clear()
        assert isinstance(config, dict), "Input Error, config of type {} not supported!".format(type(config))
        assert HYPERPARAMETERPATH in config.keys(), "Missing hyperparameter section in config dict"
        assert SETTINGSPATH in config.keys(), "Missing settings section in config dict"
        self._hyperparameter = config[HYPERPARAMETERPATH]
        self._settings = config[SETTINGSPATH]
        self.parse_members()

    def parse_members(self):
        for section_name, content in self.settings.items():
            for name, value in content.items():
                member_name = section_name + "_" + name
                setattr(self, member_name, value)
                self._extmembers.append(member_name)

    def get_typeof(self, hyperparametername):
        if not hyperparametername in self.hyperparameter.keys():
            return None
        dtype = self.hyperparameter[hyperparametername]["type"]
        if dtype == 'str':
            return str
        if dtype == 'int':
            return int
        if dtype == 'float':
            return float

    @property
    def hyperparameter(self):
        return self._hyperparameter

    @property
    def settings(self):
        return self._settings


