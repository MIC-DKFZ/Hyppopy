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

    def test_config(self):
        #TODO test the config structure to fullfill the needs, throwing useful error is not
        return True

    def read_config(self, configfile):
        self.configfilename = configfile
        self.config = DeepDict(configfile)
        if not self.test_config():
            self.configfilename = None
            self.config = None
            return False

        try:
            self.config.transfer_attrs(self, SETTINGSCUSTOMPATH.split("/")[-1])
            self.config.transfer_attrs(self, SETTINGSSOLVERPATH.split("/")[-1])
        except Exception as e:
            msg = f"transfering custom section as class attributes failed, " \
                f"is the config path to your custom section correct? {SETTINGSCUSTOMPATH}. Exception {e}"
            LOG.error(msg)
            raise LookupError(msg)

        return True
