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
    import optunity
    from yapsy.IPlugin import IPlugin
except:
    LOG.warning("optunity package not installed, will ignore this plugin!")
    print("optunity package not installed, will ignore this plugin!")

from hyppopy.settingspluginbase import SettingsPluginBase
from hyppopy.settingsparticle import split_categorical


class optunity_Settings(SettingsPluginBase, IPlugin):

    def __init__(self):
        SettingsPluginBase.__init__(self)
        LOG.debug("initialized")

    def convert_parameter(self, input_dict):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(input_dict)))

        solution_space = {}
        # split input in categorical and non-categorical data
        cat, uni = split_categorical(input_dict)
        # build up dictionary keeping all non-categorical data
        uniforms = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == 'data':
                    uniforms[key] = value2

        if len(cat) == 0:
            return uniforms
        # build nested categorical structure
        inner_level = uniforms
        for key, value in cat.items():
            tmp = {}
            tmp2 = {}
            for key2, value2 in value.items():
                if key2 == 'data':
                    for elem in value2:
                        tmp[elem] = inner_level
            tmp2[key] = tmp
            inner_level = tmp2
        solution_space = tmp2
        return solution_space


# class optunity_SettingsParticle(SettingsParticle):
#
#     def __init__(self, name=None, domain=None, dtype=None, data=None):
#         SettingsParticle.__init__(self, name, domain, dtype, data)
#
#     def convert(self):
#         if self.domain == "uniform":
#             if self.dtype == "float" or self.dtype == "double":
#                 pass
#             elif self.dtype == "int":
#                 pass
#             else:
#                 msg = f"cannot convert the type {self.dtype} in domain {self.domain}"
#                 LOG.error(msg)
#                 raise LookupError(msg)
#         elif self.domain == "loguniform":
#             if self.dtype == "float" or self.dtype == "double":
#                 pass
#             else:
#                 msg = f"cannot convert the type {self.dtype} in domain {self.domain}"
#                 LOG.error(msg)
#                 raise LookupError(msg)
#         elif self.domain == "normal":
#             if self.dtype == "float" or self.dtype == "double":
#                 pass
#             else:
#                 msg = f"cannot convert the type {self.dtype} in domain {self.domain}"
#                 LOG.error(msg)
#                 raise LookupError(msg)
#         elif self.domain == "categorical":
#             if self.dtype == 'str':
#                 pass
#             elif self.dtype == 'bool':
#                 pass
