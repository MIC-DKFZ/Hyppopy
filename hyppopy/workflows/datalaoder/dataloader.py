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


class DataLoader(object):

    def __init__(self):
        self.data = None

    @abc.abstractmethod
    def read(self, **kwargs):
        raise NotImplementedError("the read method has to be implemented in classes derived from DataLoader")

    @abc.abstractmethod
    def preprocess(self):
        pass

    def get(self):
        self.preprocess()
        return self.data
