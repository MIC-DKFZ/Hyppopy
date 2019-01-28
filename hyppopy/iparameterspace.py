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

from hyppopy.deepdict.deepdict import DeepDict

import abc
import logging
LOG = logging.getLogger('hyppopy')


class IParameterSpace(DeepDict, metaclass=abc.ABCMeta):

    def __init__(self, in_data=None):
        DeepDict.__init__(self, in_data=in_data, path_sep='/')

    def status(self):
        return "ok"

    @abc.abstractmethod
    def convert(self):
        raise NotImplementedError('users must define convert to use this base class')
