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
#from hyppopy.resultviewer import ResultViewer

import os
import logging
import pandas as pd
from hyppopy.globals import LIBNAME
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
        self._solver_plugin.set_data(data)

    def set_hyperparameters(self, params):
        self.settings_plugin.set_hyperparameter(params)

    def set_loss_function(self, func):
        self._solver_plugin.set_blackbox_function(func)

    def run(self):
        if not ProjectManager.is_ready():
            LOG.error("No config data found to initialize PluginSetting object")
            raise IOError("No config data found to initialize PluginSetting object")
        self.settings_plugin.set_hyperparameter(ProjectManager.get_hyperparameter())
        self._solver_plugin.settings = self.settings_plugin
        self._solver_plugin.run()

    def save_results(self, savedir=None, savename=None, overwrite=True):#, show=False):
        df, best = self.get_results()
        dir = None
        if savename is None:
            savename = LIBNAME
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

        appendix = ""
        if not overwrite:
            appendix = "_" + ProjectManager.identifier(True)
        name = savename + "_all" + appendix + ".csv"
        fname_all = os.path.join(dir, name)
        df.to_csv(fname_all)
        name = savename + "_best" + appendix + ".txt"
        fname_best = os.path.join(dir, name)
        with open(fname_best, "w") as text_file:
            for item in best.items():
                text_file.write("{}\t:\t{}\n".format(item[0], item[1]))

        # if show:
        #     viewer = ResultViewer(fname_all)
        #     viewer.show()
        # else:
        #     viewer = ResultViewer(fname_all, save_only=True)
        #     viewer.show()

    def get_results(self):
        results, best = self._solver_plugin.get_results()
        df = pd.DataFrame.from_dict(results)
        return df, best

    @property
    def is_ready(self):
        return self._solver_plugin is not None and self.settings_plugin is not None

    @property
    def solver_plugin(self):
        return self._solver_plugin

    @solver_plugin.setter
    def solver_plugin(self, value):
        self._solver_plugin = value

    @property
    def settings_plugin(self):
        return self._settings_plugin

    @settings_plugin.setter
    def settings_plugin(self, value):
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

