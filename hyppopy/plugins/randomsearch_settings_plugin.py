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
import random
import logging
import numpy as np
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from yapsy.IPlugin import IPlugin

from hyppopy.helpers import sample_domain
from hyppopy.projectmanager import ProjectManager
from hyppopy.settingsparticle import SettingsParticle
from hyppopy.settingspluginbase import SettingsPluginBase
from hyppopy.globals import RANDOMSAMPLES, DEFAULTITERATIONS


class randomsearch_Settings(SettingsPluginBase, IPlugin):

    def __init__(self):
        SettingsPluginBase.__init__(self)
        LOG.debug("initialized")

    def convert_parameter(self, input_dict):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(input_dict)))

        solution_space = {}
        for name, content in input_dict.items():
            particle = randomsearch_SettingsParticle(name=name)
            for key, value in content.items():
                if key == 'domain':
                    particle.domain = value
                elif key == 'data':
                    particle.data = value
                elif key == 'type':
                    particle.dtype = value
            solution_space[name] = particle.get()
        return solution_space


class randomsearch_SettingsParticle(SettingsParticle):

    def __init__(self, name=None, domain=None, dtype=None, data=None):
        SettingsParticle.__init__(self, name, domain, dtype, data)

    def convert(self):
        assert isinstance(self.data, list), "Precondition Violation, invalid input type for data!"
        N = DEFAULTITERATIONS
        if "max_iterations" in ProjectManager.__dict__.keys():
            N = ProjectManager.max_iterations
        else:
            setattr(ProjectManager, 'max_iterations', N)
            ProjectManager.max_iterations
            msg = "No max_iterrations set, set it to default [{}]".format(DEFAULTITERATIONS)
            LOG.warning(msg)
            print("WARNING: {}".format(msg))

        if self.domain == "categorical":
            samples = []
            for n in range(N):
                samples.append(random.sample(self.data, 1)[0])
            return samples
        else:
            assert len(self.data) >= 2, "Precondition Violation, invalid input data!"

            full_range = list(sample_domain(start=self.data[0], stop=self.data[1], count=RANDOMSAMPLES, ftype=self.domain))
            if self.dtype == "int":
                data = []
                for s in full_range:
                    val = int(np.round(s))
                    if len(data) > 0:
                        if val == data[-1]:
                            continue
                    data.append(val)
                full_range = data
            samples = []
            for n in range(N):
                samples.append(random.sample(full_range, 1)[0])
            return samples
