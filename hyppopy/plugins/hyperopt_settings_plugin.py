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
import logging
import numpy as np
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from pprint import pformat

try:
    from hyperopt import hp
    from yapsy.IPlugin import IPlugin
except:
    LOG.warning("hyperopt package not installed, will ignore this plugin!")
    print("hyperopt package not installed, will ignore this plugin!")

from hyppopy.settingspluginbase import SettingsPluginBase
from hyppopy.settingsparticle import SettingsParticle


class hyperopt_Settings(SettingsPluginBase, IPlugin):

    def __init__(self):
        SettingsPluginBase.__init__(self)
        LOG.debug("initialized")

    def convert_parameter(self, input_dict):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(input_dict)))

        solution_space = {}
        for name, content in input_dict.items():
            particle = hyperopt_SettingsParticle(name=name)
            for key, value in content.items():
                if key == 'domain':
                    particle.domain = value
                elif key == 'data':
                    particle.data = value
                elif key == 'type':
                    particle.dtype = value
            solution_space[name] = particle.get()
        return solution_space


class hyperopt_SettingsParticle(SettingsParticle):

    def __init__(self, name=None, domain=None, dtype=None, data=None):
        SettingsParticle.__init__(self, name, domain, dtype, data)

    def convert(self):
        if self.domain == "uniform":
            if self.dtype == "float" or self.dtype == "double":
                return hp.uniform(self.name, self.data[0], self.data[1])
            elif self.dtype == "int":
                data = list(np.arange(int(self.data[0]), int(self.data[1]+1)))
                return hp.choice(self.name, data)
            else:
                msg = "cannot convert the type {} in domain {}".format(self.dtype, self.domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif self.domain == "loguniform":
            if self.dtype == "float" or self.dtype == "double":
                if self.data[0] == 0:
                    self.data[0] += 1e-23
                assert self.data[0] > 0, "Precondition Violation, a < 0!"
                assert self.data[0] < self.data[1], "Precondition Violation, a > b!"
                assert self.data[1] > 0, "Precondition Violation, b < 0!"
                lexp = np.log(self.data[0])
                rexp = np.log(self.data[1])
                assert lexp is not np.nan, "Precondition violation, left bound input error, results in nan!"
                assert rexp is not np.nan, "Precondition violation, right bound input error, results in nan!"

                return hp.loguniform(self.name, lexp, rexp)
            else:
                msg = "cannot convert the type {} in domain {}".format(self.dtype, self.domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif self.domain == "normal":
            if self.dtype == "float" or self.dtype == "double":
                mu = (self.data[1] - self.data[0])/2.0
                sigma = mu/3
                return hp.normal(self.name, self.data[0] + mu, sigma)
            else:
                msg = "cannot convert the type {} in domain {}".format(self.dtype, self.domain)
                LOG.error(msg)
                raise LookupError(msg)
        elif self.domain == "categorical":
            if self.dtype == 'str':
                return hp.choice(self.name, self.data)
            elif self.dtype == 'bool':
                data = []
                for elem in self.data:
                    if elem == "true" or elem == "True" or elem == 1 or elem == "1":
                        data .append(True)
                    elif elem == "false" or elem == "False" or elem == 0 or elem == "0":
                        data .append(False)
                    else:
                        msg = "cannot convert the type {} in domain {}, unknown bool type value".format(self.dtype, self.domain)
                        LOG.error(msg)
                        raise LookupError(msg)
                return hp.choice(self.name, data)
