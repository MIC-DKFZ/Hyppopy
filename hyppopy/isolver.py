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

import logging
LOG = logging.getLogger('hyppopy')


class ISolver(object):
    loss_function = None

    def set_loss_function(self, func):
        """
        set loss function
        """
        self.loss_function = func

    def execute(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__name__