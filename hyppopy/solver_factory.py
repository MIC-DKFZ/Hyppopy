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

from yapsy.PluginManager import PluginManager

from .settings import PLUGIN_DEFAULT_DIR
from .solver import Solver

import logging
LOG = logging.getLogger('hyppopy')


class SolverFactory(object):
    _instance = None
    _plugin_dirs = []
    _plugins = {}

    def __init__(self):
        if SolverFactory._instance is not None:
            pass
        else:
            LOG.debug("__init__()")
            SolverFactory._instance = self
            self.reset()
            self.load_plugins()

    @staticmethod
    def instance():
        """
        Singleton instance access
        :return: [SolverFactory] instance
        """
        LOG.debug("instance()")
        if SolverFactory._instance is None:
            SolverFactory()
        return SolverFactory._instance

    def get_solver_names(self):
        return list(self._plugins.keys())

    def get_solver(self, name, **kwargs):
        if name not in self._plugins.keys():
            LOG.error(f"Solver plugin {name} not available")
            raise KeyError(f"Solver plugin {name} not available")
        return self._plugins[name]

    def add_plugin_dir(self, dir):
        """
        Add plugin directory
        """
        LOG.debug(f"add_plugin_dir({dir})")
        self._plugin_dirs.append(dir)

    def reset(self):
        """
        Reset solver factory
        """
        LOG.debug("reset()")
        self._plugins = {}
        self._plugin_dirs = []
        self.add_plugin_dir(PLUGIN_DEFAULT_DIR)

    def load_plugins(self):
        """
        Load plugin modules from plugin paths
        """
        LOG.debug("load_plugins()")
        LOG.debug(f"setPluginPlaces(" + " ".join(map(str, self._plugin_dirs)))
        manager = PluginManager()
        manager.setPluginPlaces(self._plugin_dirs)
        manager.collectPlugins()
        for plugin in manager.getAllPlugins():
            name, type = plugin.plugin_object.__class__.__name__.split("_")
            if name not in self._plugins.keys():
                self._plugins[name] = Solver(name)
            if type == "Solver":
                self._plugins[name].set_solver(plugin.plugin_object.__class__())
                LOG.info(f"Plugin: {name} Solver loaded")
            elif type == "ParameterSpace":
                self._plugins[name].set_parameter(plugin.plugin_object.__class__())
                LOG.info(f"Plugin: {name} ParameterSpace loaded")
            else:
                LOG.error(f"Failed loading plugin {name}! Please check if naming conventions are kept!")
                raise IOError(f"Failed loading plugin {name}! Please check if naming conventions are kept!")

