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


class DataLoaderBase(object):

    def __init__(self):
        self.data = None

    def start(self, **kwargs):
        self.read(**kwargs)
        if self.data is None:
            raise AttributeError("data is empty, did you missed to assign it while implementing read...?")
        self.preprocess(**kwargs)

    @abc.abstractmethod
    def read(self, **kwargs):
        raise NotImplementedError("the read method has to be implemented in classes derived from DataLoader")

    @abc.abstractmethod
    def preprocess(self, **kwargs):
        pass
