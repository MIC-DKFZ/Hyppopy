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

from pprint import pformat

try:
    from hyperopt import hp
    from yapsy.IPlugin import IPlugin
except:
    LOG.warning("hyperopt package not installed, will ignore this plugin!")
    print("hyperopt package not installed, will ignore this plugin!")

from hyppopy.settingspluginbase import SettingsPluginBase


class hyperopt_Settings(SettingsPluginBase, IPlugin):

    def __init__(self):
        SettingsPluginBase.__init__(self)
        LOG.debug("initialized")

    def convert_parameter(self, input_dict):
        LOG.debug(f"convert input parameter\n\n\t{pformat(input_dict)}\n")

        solution_space = {}
        for name, content in input_dict.items():
            data = None
            domain = None
            domain_fn = None
            for key, value in content.items():
                if key == 'domain':
                    domain = value
                    if value == 'uniform':
                        domain_fn = hp.uniform
                    if value == 'categorical':
                        domain_fn = hp.choice
                if key == 'data':
                    data = value
            if domain == 'categorical':
                solution_space[name] = domain_fn(name, data)
            else:
                solution_space[name] = domain_fn(name, data[0], data[1])
        return solution_space

