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


from hyppopy.projectmanager import ProjectManager

import os
import datetime
import logging
import pandas as pd
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class Solver(object):
    _name = None
    _solver_plugin = None
    _settings_plugin = None

    def __init__(self):
        pass

    def set_data(self, data):
        self.solver.set_data(data)

    def set_hyperparameters(self, params):
        self.settings.set_hyperparameter(params)

    def set_loss_function(self, loss_func):
        self.solver.set_loss_function(loss_func)

    def run(self):
        if not ProjectManager.is_ready():
            LOG.error("No config data found to initialize PluginSetting object")
            raise IOError("No config data found to initialize PluginSetting object")
        self.settings.set_hyperparameter(ProjectManager.get_hyperparameter())
        self.solver.settings = self.settings
        self.solver.run()

    def save_results(self, savedir=None, savename=None):
        df, best = self.get_results()
        dir = None
        if savename is None:
            savename = "hypopy"
        if savedir is None:
            if 'output_dir' in ProjectManager.__dict__.keys():
                if not os.path.isdir(ProjectManager.output_dir):
                    os.mkdir(ProjectManager.output_dir)
                dir = ProjectManager.output_dir
            else:
                print("WARNING: No solver option output_dir found, cannot save results!")
                LOG.warning("WARNING: No solver option output_dir found, cannot save results!")
        else:
            dir = savedir
            if not os.path.isdir(savedir):
                os.mkdir(savedir)

        tstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = savename + "_all_" + tstr + ".csv"
        fname = os.path.join(dir, name)
        df.to_csv(fname)
        name = savename + "_best_" + tstr + ".txt"
        fname = os.path.join(dir, name)
        with open(fname, "w") as text_file:
            for item in best.items():
                text_file.write("{}\t:\t{}\n".format(item[0], item[1]))

    def get_results(self):
        results, best = self.solver.get_results()
        df = pd.DataFrame.from_dict(results)
        return df, best

    @property
    def is_ready(self):
        return self.solver is not None and self.settings is not None

    @property
    def solver(self):
        return self._solver_plugin

    @solver.setter
    def solver(self, value):
        self._solver_plugin = value

    @property
    def settings(self):
        return self._settings_plugin

    @settings.setter
    def settings(self, value):
        self._settings_plugin = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            msg = "Invalid input, str type expected for value, got {} instead".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._name = value

