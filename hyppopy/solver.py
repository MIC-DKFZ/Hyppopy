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

import logging
LOG = logging.getLogger('hyppopy')


class Solver(object):
    _name = None
    _solver = None
    _parameter = None

    def __init__(self, name=None):
        self.set_name(name)

    def __str__(self):
        txt = f"\nSolver Instance {self._name}:"
        if self._solver is None:
            txt += f"\n - Status solver: None"
        else:
            txt += f"\n - Status solver: ok"
        if self._parameter is None:
            txt += f"\n - Status parameter: None"
        else:
            txt += f"\n - Status parameter: ok"
        return txt

    def is_ready(self):
        return self._solver is not None and self._parameter is not None

    def set_name(self, name):
        LOG.debug(f"set_name({name})")
        self._name = name

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, value):
        if not type(value).__name__.endswith("Solver"):
            LOG.error("Input Error, value is not of type Solver")
            raise IOError("Input Error, value is not of type Solver")
        self._solver = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        if not type(value).__name__.endswith("ParameterSpace"):
            LOG.error("Input Error, value is not of type ParameterSpace")
            raise IOError("Input Error, value is not of type ParameterSpace")
        self._parameter = value

