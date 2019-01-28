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
from hyppopy.ispace import ISpace


class HyperoptSpace(ISpace):

    def convert(self):
        pass


class HyperoptPlugin(IPlugin, ISolver):

    def execute(self, *args, **kwargs):
        pass

