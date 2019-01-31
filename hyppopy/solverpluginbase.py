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

import os
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class SolverPluginBase(object):
    data = None
    loss = None
    _name = None

    def __init__(self):
        pass

    @abc.abstractmethod
    def loss_function(self, params):
        raise NotImplementedError('users must define loss_func to use this base class')

    @abc.abstractmethod
    def execute_solver(self):
        raise NotImplementedError('users must define execute_solver to use this base class')

    @abc.abstractmethod
    def convert_results(self):
        raise NotImplementedError('users must define convert_results to use this base class')

    def set_data(self, data):
        self.data = data

    def set_loss_function(self, func):
        self.loss = func

    def get_results(self):
        self.convert_results()

    def run(self, parameter):
        self.execute_solver(parameter)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            LOG.error(f"Invalid input, str type expected for value, got {type(value)} instead")
            raise IOError(f"Invalid input, str type expected for value, got {type(value)} instead")
        self._name = value

