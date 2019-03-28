# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical Image Computing.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE

from .Singleton import *

import os
import logging
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


@singleton_object
class ProjectManager(metaclass=Singleton):

    def __init__(self):
        self._current_project = None
        self._projects = {}

    def clear_all(self):
        pass

    def new_project(self, name="HyppopyProject", config=None):
        if name in self._projects.keys():
            name = self.check_projectname(name)
        self._projects[name] = HyppopyProject(config)
        self._current_project = self._projects[name]
        return self._current_project

    def check_projectname(self, name):
        split = name.split(".")
        if len(split) == 0:
            return split[0] + "." + str(0).zfill(3)
        else:
            try:
                number = int(split[-1])
                del split[-1]
            except:
                number = 0
            return '.'.join(split) + "." + str(number).zfill(3)

    def get_current(self):
        if self._current_project is None:
            self.new_project()
        return self._current_project

    def get_project(self, name):
        if name in self._projects.keys():
            self._current_project = self._projects[name]
            return self.get_current()
        return self.new_project(name)

    def get_projectnames(self):
        return self._projects.keys()

