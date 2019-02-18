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
import abc
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class SettingsParticle(object):
    domains = ["uniform", "loguniform", "normal", "categorical"]
    _name = None
    _domain = None
    _dtype = None
    _data = None

    def __init__(self, name=None, domain=None, dtype=None, data=None):
        if name is not None:
            self.name = name
        if domain is not None:
            self.domain = domain
        if dtype is not None:
            self.dtype = dtype
        if data is not None:
            self.data = data

    @abc.abstractmethod
    def convert(self):
        raise NotImplementedError("the user has to implement this function")

    def get(self):
        msg = None
        if self.name is None:  msg = "cannot convert unnamed parameter"
        if self.domain is None: msg = "cannot convert parameter of empty domain"
        if self.dtype is None: msg = "cannot convert parameter with unknown dtype"
        if self.data is None: msg = "cannot convert parameter having no data"
        if msg is not None:
            LOG.error(msg)
            raise LookupError(msg)
        return self.convert()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
