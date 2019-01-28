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

import abc
import logging
LOG = logging.getLogger('hyppopy')


class ISolver(object, metaclass=abc.ABCMeta):
    loss_function = None
    space = None

    def set_loss_function(self, func):
        """
        set loss function
        """
        self.loss_function = func

    def set_space(self, space):
        """
        set loss function
        """
        self.space = space

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError('users must define execute to use this base class')

