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

import warnings

from hyppopy.globals import *

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyppopyProject(object):

    def __init__(self, config=None):
        self._hyperparameter = {}
        self._settings = {}
        self._extmembers = []
        if config is not None:
            self.set_config(config)

    def clear(self):
        self._hyperparameter = {}
        self._settings = {}
        for added in self._extmembers:
            if added in self.__dict__.keys():
                del self.__dict__[added]
        self._extmembers = []

    def set_config(self, config):
        self.clear()
        assert isinstance(config, dict), "Input Error, config of type {} not supported!".format(type(config))
        assert HYPERPARAMETERPATH in config.keys(), "Missing hyperparameter section in config dict"
        #assert SETTINGSPATH in config.keys(), "Missing settings section in config dict"
        if not SETTINGSPATH in config.keys():
            config[SETTINGSPATH] = {"solver": {"max_iterations": DEFAULTITERATIONS}}
            msg = "config dict had no section {0}/solver/max_iterations, set default value: {1}".format(SETTINGSPATH, DEFAULTITERATIONS)
            warnings.warn(msg)
            LOG.warning(msg)
        self._hyperparameter = config[HYPERPARAMETERPATH]
        self._settings = config[SETTINGSPATH]
        self.parse_members()

    def add_hyperparameter(self, name, domain, data, dtype):
        assert isinstance(name, str), "precondition violation, name of type {} not allowed, expect str!".format(type(name))
        assert isinstance(domain, str), "precondition violation, domain of type {} not allowed, expect str!".format(type(domain))
        assert domain in SUPPORTED_DOMAINS, "domain {} not supported, expect {}!".format(domain, SUPPORTED_DOMAINS)
        assert isinstance(data, list) or isinstance(data, tuple), "precondition violation, data of type {} not allowed, expect list or tuple!".format(type(data))
        if domain != "categorical":
            assert len(data) == 3 or len(data) == 2, "precondition violation, data must be a list of len 2 or 3"
        assert isinstance(dtype, str), "precondition violation, dtype of type {} not allowed, expect str!".format(type(dtype))
        assert dtype in SUPPORTED_DTYPES, "precondition violation, dtype {} not supported, expect {}!".format(dtype, SUPPORTED_DTYPES)
        self._hyperparameter[name] = {"domain": domain, "data": data, "type": dtype}

    def add_settings(self, section, name, value):
        assert isinstance(section, str), "precondition violation, section of type {} not allowed, expect str!".format(type(section))
        assert isinstance(name, str), "precondition violation, name of type {} not allowed, expect str!".format(type(name))
        if section not in self._settings.keys():
            self._settings[section] = {}
        self._settings[section][name] = value
        self.parse_members()

    def parse_members(self):
        for section_name, content in self.settings.items():
            for name, value in content.items():
                member_name = section_name + "_" + name
                if member_name not in self._extmembers:
                    setattr(self, member_name, value)
                    self._extmembers.append(member_name)
                else:
                    self.__dict__[member_name] = value

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


