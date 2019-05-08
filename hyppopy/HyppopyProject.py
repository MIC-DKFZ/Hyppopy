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

__all__ = ['HyppopyProject']

import copy

from hyppopy.globals import *

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyppopyProject(object):
    """
    The HyppopyProject class takes care of the optimization settings. An instance can be configured using a config
    dictionary or by using the hyperparameter and settings methods. In case of initializing via dicts those can be
    passed to the constructor or by using the set_config method. After initialization a HyppopyProject instance is
    passed to a solver class which internally checks for consistency with it's needs. The class distinguished
    between two categories, hyperparameter and general settings.

    The hyperparameter are a dictionary structure as follows and can be accessed via hyperparameter
    {'param_name: {'domain': 'uniform', ...}, ...}

    General settings are internally converted to class attributes and can accessed directly or via settings

    An example config could look like:
    config = {'hyperparameter': {'myparam': {'domain': 'uniform', 'data': [0, 100], 'type': float}, ...},
              'my_setting_1': 3.1415,
              'my_setting_2': 'hello world'}
    project = HyppopyProject(config)

    The same can be achieved using:
    project = HyppopyProject()
    project.add_hyperparameter(name='myparam', domain='uniform', data=[0, 100], type=float})
    project.add_setting('my_setting_1', 3.1415)
    project.add_setting('my_setting_2', 'hello world')
    """

    def __init__(self, config=None):
        """
        Constructor

        :param config: [dict] config dictionary of the form {'hyperparameter': {...}, ...}
        """
        self._data = {HYPERPARAMETERPATH: {}, SETTINGSPATH: {}}
        if config is not None:
            self.set_config(config)

    def __parse_members(self):
        """
        The function converts settings into class attributes
        """
        for name, value in self.settings.items():
            if name not in self.__dict__.keys():
                setattr(self, name, value)
            else:
                self.__dict__[name] = value

    def set_config(self, config):
        """
        Set a config dict

        :param config: [dict] configuration dict defining hyperparameter and general settings
        """
        assert isinstance(config, dict), "precondition violation, config needs to be of type dict, got {}".format(type(config))
        confic_cp = copy.deepcopy(config)
        if HYPERPARAMETERPATH in confic_cp.keys():
            self._data[HYPERPARAMETERPATH] = confic_cp[HYPERPARAMETERPATH]
            del confic_cp[HYPERPARAMETERPATH]
        self._data[SETTINGSPATH] = confic_cp
        self.__parse_members()

    def set_hyperparameter(self, params):
        """
        This function can be used to set the hyperparameter description directly by passing the hyperparameter section
        of a config dict (see class description). Alternatively use add_hyperparameter to add one after each other.

        :param params: [dict] configuration dict defining hyperparameter
        """
        assert isinstance(params, dict), "precondition violation, params needs to be of type dict, got {}".format(type(params))
        self._data[HYPERPARAMETERPATH] = params

    def add_hyperparameter(self, name, **kwargs):
        """
        This function can be used to set hyperparameter descriptions. Alternatively use set_hyperparameter to set all at
        once.

        :param name: [str] hyperparameter name
        :param **kwargs: [dict] configuration dict defining a hyperparameter e.g. domain='uniform', data=[1,100], ...
        """
        assert isinstance(name, str), "precondition violation, name needs to be of type str, got {}".format(type(name))
        self._data[HYPERPARAMETERPATH][name] = kwargs

    def set_settings(self, **kwargs):
        """
        This function can be used to set the general settings directly by passing the settings as name=value pairs.
        Alternatively use add_setting to add one after each other.

        :param **kwargs: [dict] settings dict e.g. my_setting_1=3.1415, my_setting_2='hello world', ...
        """
        self._data[SETTINGSPATH] = kwargs
        self.__parse_members()

    def add_setting(self, name, value):
        """
        This function can be used to set a general settings. Alternatively use set_settings to set all at once.

        :param name: [str] setting name
        :param value: [object] settings value
        """
        assert isinstance(name, str), "precondition violation, name needs to be of type str, got {}".format(type(name))
        self._data[SETTINGSPATH][name] = value
        self.__parse_members()

    def get_typeof(self, name):
        """
        Returns a hyperparameter type by name

        :param name: [str] hyperparameter name
        :return: [type] hyperparameter type
        """
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

