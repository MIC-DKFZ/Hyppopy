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
import datetime
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


@singleton_object
class ProjectManager(metaclass=Singleton):

    def __init__(self):
        self.configfilename = None
        self.config = None
        self._extmembers = []
        self._identifier = None

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
            raise IOError(msg)
        sections = ["hyperparameter"]
        sections += [SETTINGSSOLVERPATH.split("/")[-1]]
        sections += [SETTINGSCUSTOMPATH.split("/")[-1]]
        sections_available = [True, True, True]
        for n, sec in enumerate(sections):
            if not self.config.has_section(sec):
                msg = "WARNING: config has no section {}".format(sec)
                LOG.warning(msg)
                sections_available[n] = False
        return sections_available


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

        sections_available = self.test_config()
        if not sections_available[0]:
            msg = "Missing section {}".format("hyperparameter")
            LOG.error(msg)
            raise LookupError(msg)
        if not sections_available[1]:
            msg = "Missing section {}".format(SETTINGSSOLVERPATH)
            LOG.error(msg)
            raise LookupError(msg)
        else:
            try:
                self._extmembers += self.config.transfer_attrs(self, SETTINGSCUSTOMPATH.split("/")[-1])
            except Exception as e:
                msg = "transfering custom section as class attributes failed, " \
                      "is the config path to your custom section correct? {}. Exception {}".format(SETTINGSCUSTOMPATH,
                                                                                                   e)
                LOG.error(msg)
                raise LookupError(msg)
        if sections_available[2]:
            try:
                self._extmembers += self.config.transfer_attrs(self, SETTINGSSOLVERPATH.split("/")[-1])
            except Exception as e:
                msg = "transfering custom section as class attributes failed, " \
                      "is the config path to your custom section correct? {}. Exception {}".format(SETTINGSCUSTOMPATH,
                                                                                                   e)
                LOG.error(msg)
                raise LookupError(msg)
        return True

    def read_config(self, configfile):
        self.clear()
        self.configfilename = configfile
        self.config = DeepDict(configfile)
        sections_available = self.test_config()
        if not sections_available[0]:
            msg = "Missing section {}".format("hyperparameter")
            LOG.error(msg)
            raise LookupError(msg)
        if not sections_available[1]:
            msg = "Missing section {}".format(SETTINGSSOLVERPATH)
            LOG.error(msg)
            raise LookupError(msg)
        else:
            try:
                self._extmembers += self.config.transfer_attrs(self, SETTINGSSOLVERPATH.split("/")[-1])
            except Exception as e:
                msg = "transfering custom section as class attributes failed, " \
                      "is the config path to your custom section correct? {}. Exception {}".format(SETTINGSSOLVERPATH, e)
                LOG.error(msg)
                raise LookupError(msg)
        if sections_available[2]:
            try:
                self._extmembers += self.config.transfer_attrs(self, SETTINGSCUSTOMPATH.split("/")[-1])
            except Exception as e:
                msg = "transfering custom section as class attributes failed, " \
                      "is the config path to your custom section correct? {}. Exception {}".format(SETTINGSCUSTOMPATH, e)
                LOG.error(msg)
                raise LookupError(msg)

        return True

    def identifier(self, force=False):
        if self._identifier is None or force:
            self._identifier = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        return self._identifier
