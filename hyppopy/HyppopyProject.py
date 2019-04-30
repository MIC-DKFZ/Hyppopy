# Hyppopy - A Hyper-Parameter Optimization Toolbox
#
# Copyright (c) German Cancer Research Center,
# Division of Medical Image Computing.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE

import copy

from hyppopy.globals import *

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyppopyProject(object):

    def __init__(self, config=None):
        self._data = {HYPERPARAMETERPATH: {}, SETTINGSPATH: {}}
        if config is not None:
            self.set_config(config)

    def set_config(self, config):
        assert isinstance(config, dict), "precondition violation, config needs to be of type dict, got {}".format(type(config))
        confic_cp = copy.deepcopy(config)
        if HYPERPARAMETERPATH in confic_cp.keys():
            self._data[HYPERPARAMETERPATH] = confic_cp[HYPERPARAMETERPATH]
            del confic_cp[HYPERPARAMETERPATH]
        self._data[SETTINGSPATH] = confic_cp
        self.parse_members()

    def set_hyperparameter(self, params):
        assert isinstance(params, dict), "precondition violation, params needs to be of type dict, got {}".format(type(params))
        self._data[HYPERPARAMETERPATH] = params

    def set_settings(self, **kwargs):
        self._data[SETTINGSPATH] = kwargs
        self.parse_members()

    def add_hyperparameter(self, name, **kwargs):
        assert isinstance(name, str), "precondition violation, name needs to be of type str, got {}".format(type(name))
        self._data[HYPERPARAMETERPATH][name] = kwargs

    def add_setting(self, name, value):
        assert isinstance(name, str), "precondition violation, name needs to be of type str, got {}".format(type(name))
        self._data[SETTINGSPATH][name] = value
        self.parse_members()

    def parse_members(self):
        for name, value in self.settings.items():
            if name not in self.__dict__.keys():
                setattr(self, name, value)
            else:
                self.__dict__[name] = value

    def get_typeof(self, name):
        if not name in self.hyperparameter.keys():
            raise LookupError("Typechecking failed, couldn't find hyperparameter {}!".format(name))
        if not "type" in self.hyperparameter[name].keys():
            raise LookupError("Typechecking failed, couldn't find hyperparameter signature type!")
        dtype = self.hyperparameter[name]["type"]
        return dtype

    @property
    def hyperparameter(self):
        return self._data[HYPERPARAMETERPATH]

    @property
    def settings(self):
        return self._data[SETTINGSPATH]

