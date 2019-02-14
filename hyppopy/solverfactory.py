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

from hyppopy.projectmanager import ProjectManager
from hyppopy.globals import PLUGIN_DEFAULT_DIR
from hyppopy.deepdict import DeepDict
from hyppopy.solver import Solver
from hyppopy.singleton import *

import os
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

@singleton_object
class SolverFactory(metaclass=Singleton):
    """
    This class is responsible for grabbing all plugins from the plugin folder arranging them into a
    Solver class instances. These Solver class instances can be requested from the factory via the
    get_solver method. The SolverFactory class is a Singleton class, so try not to instantiate it using
    SolverFactory(), the consequences will be horrific. Instead use is like a class having static
    functions only, SolverFactory.method().
    """
    _plugin_dirs = []
    _plugins = {}

    def __init__(self):
        self.reset()
        self.load_plugins()
        LOG.debug("Solverfactory initialized")

    def load_plugins(self):
        """
        Load plugin modules from plugin paths
        """
        LOG.debug("load_plugins()")
        manager = PluginManager()
        LOG.debug("setPluginPlaces(" + " ".join(map(str, self._plugin_dirs)))
        manager.setPluginPlaces(self._plugin_dirs)
        manager.collectPlugins()
        for plugin in manager.getAllPlugins():
            name_elements = plugin.plugin_object.__class__.__name__.split("_")
            LOG.debug("found plugin " + " ".join(map(str, name_elements)))
            print("Solverfactory: found plugins " + " ".join(map(str, name_elements)))
            if len(name_elements) != 2 or ("Solver" not in name_elements and "Settings" not in name_elements):
                msg = "invalid plugin class naming for class {}, the convention is libname_Solver or libname_Settings.".format(plugin.plugin_object.__class__.__name__)
                LOG.error(msg)
                raise NameError(msg)
            if name_elements[0] not in self._plugins.keys():
                self._plugins[name_elements[0]] = Solver()
                self._plugins[name_elements[0]].name = name_elements[0]
            if name_elements[1] == "Solver":
                try:
                    obj = plugin.plugin_object.__class__()
                    obj.name = name_elements[0]
                    self._plugins[name_elements[0]].solver = obj
                    LOG.info("plugin: {} Solver loaded".format(name_elements[0]))
                except Exception as e:
                    msg = "failed to instanciate class {}".format(plugin.plugin_object.__class__.__name__)
                    LOG.error(msg)
                    raise ImportError(msg)
            elif name_elements[1] == "Settings":
                try:
                    obj = plugin.plugin_object.__class__()
                    obj.name = name_elements[0]
                    self._plugins[name_elements[0]].settings = obj
                    LOG.info("plugin: {} ParameterSpace loaded".format(name_elements[0]))
                except Exception as e:
                    msg = "failed to instanciate class {}".format(plugin.plugin_object.__class__.__name__)
                    LOG.error(msg)
                    raise ImportError(msg)
            else:
                msg = "failed loading plugin {}, please check if naming conventions are kept!".format(name_elements[0])
                LOG.error(msg)
                raise IOError(msg)
        if len(self._plugins) == 0:
            msg = "no plugins found, please check your plugin folder names or your plugin scripts for errors!"
            LOG.error(msg)
            raise IOError(msg)

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
        LOG.debug("add_plugin_dir({})".format(dir))
        self._plugin_dirs.append(dir)

    def list_solver(self):
        """
        list all solvers available
        :return: [list(str)]
        """
        return list(self._plugins.keys())

    def from_settings(self, settings):
        if isinstance(settings, str):
            if not os.path.isfile(settings):
                LOG.error("input error, file {} not found!".format(settings))
            if not ProjectManager.read_config(settings):
                LOG.error("failed to read config in ProjectManager!")
                return None
        else:
            if not ProjectManager.set_config(settings):
                LOG.error("failed to set config in ProjectManager!")
                return None

        if not ProjectManager.is_ready():
            LOG.error("failed to set config in ProjectManager!")
            return None

        try:
            solver = self.get_solver(ProjectManager.use_plugin)
        except Exception as e:
            msg = "failed to create solver, reason {}".format(e)
            LOG.error(msg)
            return None
        return solver

    def get_solver(self, name):
        """
        returns a solver by name tag
        :param name: [str] solver name
        :return: [Solver] instance
        """
        if not isinstance(name, str):
            msg = "Invalid input, str type expected for name, got {} instead".format(type(name))
            LOG.error(msg)
            raise IOError(msg)
        if name not in self.list_solver():
            msg = "failed solver request, a solver called {} is not available, check for typo or if your plugin failed while loading!".format(name)
            LOG.error(msg)
            raise LookupError(msg)
        LOG.debug("get_solver({})".format(name))
        return self._plugins[name]
