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

from hyppopy.settings import PLUGIN_DEFAULT_DIR
from hyppopy.solver import Solver

import os
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
            name_elements = plugin.plugin_object.__class__.__name__.split("_")
            if len(name_elements) != 2 or ("Solver" not in name_elements and "Settings" not in name_elements):
                LOG.error(f"Invalid plugin class naming for class {plugin.plugin_object.__class__.__name__}, the convention is libname_Solver or libname_Settings.")
                raise NameError(f"Invalid plugin class naming for class {plugin.plugin_object.__class__.__name__}, the convention is libname_Solver or libname_Settings.")
            if name_elements[0] not in self._plugins.keys():
                self._plugins[name_elements[0]] = Solver()
                self._plugins[name_elements[0]].name = name_elements[0]
            if name_elements[1] == "Solver":
                try:
                    self._plugins[name_elements[0]].solver = plugin.plugin_object.__class__()
                    LOG.info(f"Plugin: {name_elements[0]} Solver loaded")
                except Exception as e:
                    LOG.error(f"Failed to instanciate class {plugin.plugin_object.__class__.__name__}")
                    raise ImportError(f"Failed to instanciate class {plugin.plugin_object.__class__.__name__}")
            elif type == "Settings":
                try:
                    self._plugins[name_elements[0]].settings = plugin.plugin_object.__class__()
                    LOG.info(f"Plugin: {name_elements[0]} ParameterSpace loaded")
                except Exception as e:
                    LOG.error(f"Failed to instanciate class {plugin.plugin_object.__class__.__name__}")
                    raise ImportError(f"Failed to instanciate class {plugin.plugin_object.__class__.__name__}")
            else:
                LOG.error(f"Failed loading plugin {name_elements[0]}! Please check if naming conventions are kept!")
                raise IOError(f"Failed loading plugin {name_elements[0]}! Please check if naming conventions are kept!")

    def reset(self):
        """
        Reset solver factory
        """
        LOG.debug("reset()")
        self._plugins = {}
        self._plugin_dirs = []
        self.add_plugin_dir(os.path.abspath(PLUGIN_DEFAULT_DIR))

    def add_plugin_dir(self, dir):
        """
        Add plugin directory
        """
        LOG.debug(f"add_plugin_dir({dir})")
        self._plugin_dirs.append(dir)

    def get_solver(self, name):
        LOG.debug(f"get_solver({name})")
        return self._plugins[name]
