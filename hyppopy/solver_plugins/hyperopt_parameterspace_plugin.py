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

from yapsy.IPlugin import IPlugin

from hyppopy.isolver import ISolver
from hyppopy.iparameterspace import IParameterSpace

import logging
LOG = logging.getLogger('hyppopy')


class hyperopt_ParameterSpace(IPlugin, IParameterSpace):

    def convert(self):
        """
        Converts the internal IParameterSpace data into the form preferred by
        the Hyperopt package. Hyperopt has it's own parameter space definition
        for each type of hyperparameter:

        https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions
        :return:
        """
        pass

