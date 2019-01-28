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
            LOG.error("Solver plugin name not available")
            raise KeyError("Solver plugin name not available")

        if name == "HyperoptPlugin":
            pass
        elif name == "OptunityPlugin":
            pass
        else:
            LOG.error("Solver plugin name does not match with key")
            raise KeyError("Solver plugin name does not match with key")

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
            self._plugins[plugin.plugin_object.__class__.__name__] = plugin.plugin_object
            LOG.info(f"Plugin: {plugin.plugin_object.__class__.__name__} loaded")

