# DKFZ
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

# -*- coding: utf-8 -*-

from yapsy.IPlugin import IPlugin
import logging
LOG = logging.getLogger('hyppopy')

from hyppopy.isolver import ISolver


class OptunityPlugin(IPlugin, ISolver):

    def __init__(self):
        self.__name__ = "OptunityPlugin"

    def execute(self, *args, **kwargs):
        pass