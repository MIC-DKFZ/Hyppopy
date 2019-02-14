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

from hyppopy.singleton import *
from hyppopy.deepdict import DeepDict
from hyppopy.globals import SETTINGSCUSTOMPATH, SETTINGSSOLVERPATH

import os
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


@singleton_object
class ProjectManager(metaclass=Singleton):

    def __init__(self):
        self.configfilename = None
        self.config = None
        self._extmembers = []

    def clear(self):
        self.configfilename = None
        self.config = None
        self.remove_externals()

    def is_ready(self):
        return self.config is not None

    def remove_externals(self):
        for added in self._extmembers:
            if added in self.__dict__.keys():
                del self.__dict__[added]
        self._extmembers = []

    def get_hyperparameter(self):
        return self.config["hyperparameter"]

    def test_config(self):
        if not isinstance(self.config, DeepDict):
            msg = "test_config failed, config is not of type DeepDict"
            LOG.error(msg)
            return False
        sections = ["hyperparameter"]
        sections += SETTINGSSOLVERPATH.split("/")
        sections += SETTINGSCUSTOMPATH.split("/")
        for sec in sections:
            if not self.config.has_section(sec):
                msg = "test_config failed, config has no section {}".format(sec)
                LOG.error(msg)
                return False
        return True

    def set_config(self, config):
        self.clear()
        if isinstance(config, dict):
            self.config = DeepDict()
            self.config.data = config
        elif isinstance(config, DeepDict):
            self.config = config
        else:
            msg = "unknown type ({}) for config passed, expected dict or DeepDict".format(type(config))
            LOG.error(msg)
            raise IOError(msg)

        if not self.test_config():
            self.clear()
            return False

        try:
            self._extmembers += self.config.transfer_attrs(self, SETTINGSCUSTOMPATH.split("/")[-1])
            self._extmembers += self.config.transfer_attrs(self, SETTINGSSOLVERPATH.split("/")[-1])
        except Exception as e:
            msg = "transfering custom section as class attributes failed, " \
                "is the config path to your custom section correct? {}. Exception {}".format(SETTINGSCUSTOMPATH, e)
            LOG.error(msg)
            raise LookupError(msg)

        return True

    def read_config(self, configfile):
        self.clear()
        self.configfilename = configfile
        self.config = DeepDict(configfile)
        if not self.test_config():
            self.clear()
            return False

        try:
            self._extmembers += self.config.transfer_attrs(self, SETTINGSCUSTOMPATH.split("/")[-1])
            self._extmembers += self.config.transfer_attrs(self, SETTINGSSOLVERPATH.split("/")[-1])
        except Exception as e:
            msg = "transfering custom section as class attributes failed, " \
                "is the config path to your custom section correct? {}. Exception {e}".format(SETTINGSCUSTOMPATH, e)
            LOG.error(msg)
            raise LookupError(msg)

        return True
