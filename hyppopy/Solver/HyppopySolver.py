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

import abc

import os
import copy
import types
import logging
import datetime
import numpy as np
import pandas as pd
from hyperopt import Trials
from hyppopy.globals import DEBUGLEVEL
from hyppopy.VisdomViewer import VisdomViewer
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.BlackboxFunction import BlackboxFunction
from hyppopy.VirtualFunction import VirtualFunction

from hyppopy.globals import DEBUGLEVEL, DEFAULTITERATIONS

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyppopySolver(object):
    """
    The HyppopySolver class is the base class for all solver addons. It defines virtual functions a child class has
    to implement to deal with the front-end communication, orchestrating the optimization process and ensuring a proper
    process information storing.
    The key idea is that the HyppopySolver class defines an interface to configure and run an object instance of itself
    independently from the concrete solver lib used to optimize in the background. To achieve this goal an addon
    developer needs to implement the abstract methods 'convert_searchspace', 'execute_solver' and 'loss_function_call'.
    These methods abstract the peculiarities of the solver libs to offer, on the user side, a simple and consistent
    parameter space configuration and optimization procedure. The method 'convert_searchspace' transforms the hyppopy
    parameter space description into the solver lib specific description. The method loss_function_call is used to
    handle solver lib specifics of calling the actual blackbox function and execute_solver is executed when the run
    method is invoked und takes care of calling the solver lib solving routine.
    """
    def __init__(self, project=None):
        self._idx = None
        self._best = None
        self._trials = None
        self._blackbox = None
        self._max_iterations = None
        self._project = project
        self._total_duration = None
        self._solver_overhead = None
        self._time_per_iteration = None
        self._accumulated_blackbox_time = None
        self._has_maxiteration_field = True
        self._visdom_viewer = None

    @abc.abstractmethod
    def convert_searchspace(self, hyperparameter):
        """
        This function gets the unified hyppopy-like parameterspace description as input and, if necessary, should
        convert it into a solver lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.
        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':'float'}, ...}
        :return: [object] converted hyperparameter space
        """
        raise NotImplementedError('users must define convert_searchspace to use this class')

    @abc.abstractmethod
    def execute_solver(self, searchspace):
        """
        This function is called immediatly after convert_searchspace and get the output of the latter as input. It's
        purpose is to call the solver libs main optimization function.
        :param searchspace: converted hyperparameter space
        """
        raise NotImplementedError('users must define execute_solver to use this class')

    @abc.abstractmethod
    def loss_function_call(self, params):
        """
        This function is called within the function loss_function and encapsulates the actual blackbox function call
        in each iteration. The function loss_function takes care of the iteration driving and reporting, but each solver
        lib might need some special treatment between the parameter set selection and the calling of the actual blackbox
        function, e.g. parameter converting.
        :param params: [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...}
        :return: [float] loss
        """
        raise NotImplementedError('users must define convert_searchspace to use this class')

    def loss_function(self, **params):
        """
        This function is called each iteration with a selected parameter set. The parameter set selection is driven by
        the solver lib itself. The purpose of this function is to take care of the iteration reporting and the calling
        of the callback_func is available. As a developer you might want to overwrite this function completely (e.g.
        HyperoptSolver) but then you need to take care for iteration reporting for yourself. The alternative is to only
        implement loss_function_call (e.g. OptunitySolver).
        :param params: [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...}
        :return: [float] loss
        """
        self._idx += 1
        vals = {}
        idx = {}
        for key, value in params.items():
            vals[key] = [value]
            idx[key] = [self._idx]
        trial = {'tid': self._idx,
                 'result': {'loss': None, 'status': 'ok'},
                 'misc': {
                     'tid': self._idx,
                     'idxs': idx,
                     'vals': vals
                 },
                 'book_time': datetime.datetime.now(),
                 'refresh_time': None
                 }
        try:
            loss = self.loss_function_call(params)
            trial['result']['loss'] = loss
            trial['result']['status'] = 'ok'
            if loss == np.nan:
                trial['result']['status'] = 'failed'
        except Exception as e:
            LOG.error("computing loss failed due to:\n {}".format(e))
            loss = np.nan
            trial['result']['loss'] = np.nan
            trial['result']['status'] = 'failed'
        trial['refresh_time'] = datetime.datetime.now()
        self._trials.trials.append(trial)
        cbd = copy.deepcopy(params)
        cbd['iterations'] = self._idx
        cbd['loss'] = loss
        cbd['status'] = trial['result']['status']
        cbd['book_time'] = trial['book_time']
        cbd['refresh_time'] = trial['refresh_time']
        if isinstance(self.blackbox, BlackboxFunction) and self.blackbox.callback_func is not None:
            self.blackbox.callback_func(**cbd)
        if self._visdom_viewer is not None:
            self._visdom_viewer.update(cbd)
        return loss

    def run(self, print_stats=True):
        """
        This function starts the optimization process.
        :param print_stats: [bool] en- or disable console output
        """
        self._idx = 0
        self.trials = Trials()
        if self._has_maxiteration_field:
            if 'solver_max_iterations' not in self.project.__dict__:
                msg = "Missing max_iteration entry in project, use default {}!".format(DEFAULTITERATIONS)
                LOG.warning(msg)
                print("WARNING: {}".format(msg))
                setattr(self.project, 'solver_max_iterations', DEFAULTITERATIONS)
            self._max_iterations = self.project.solver_max_iterations

        start_time = datetime.datetime.now()
        try:
            search_space = self.convert_searchspace(self.project.hyperparameter)
        except Exception as e:
            msg = "Failed to convert searchspace, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
        try:
            self.execute_solver(search_space)
        except Exception as e:
            msg = "Failed to execute solver, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
        end_time = datetime.datetime.now()
        dt = end_time - start_time
        days = divmod(dt.total_seconds(), 86400)
        hours = divmod(days[1], 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)
        milliseconds = divmod(seconds[1], 0.001)
        self._total_duration = [int(days[0]), int(hours[0]), int(minutes[0]), int(seconds[0]), int(milliseconds[0])]
        if print_stats:
            self.print_best()
            self.print_timestats()

    def get_results(self):
        """
        This function returns a complete optimization history as pandas DataFrame and a dict with the optimal parameter set.
        :return: [DataFrame], [dict] history and optimal parameter set
        """
        assert isinstance(self.trials, Trials), "precondition violation, wrong trials type! Maybe solver was not yet executed?"
        results = {'duration': [], 'losses': [], 'status': []}
        pset = self.trials.trials[0]['misc']['vals']
        for p in pset.keys():
            results[p] = []

        for n, trial in enumerate(self.trials.trials):
            t1 = trial['book_time']
            t2 = trial['refresh_time']
            results['duration'].append((t2 - t1).microseconds / 1000.0)
            results['losses'].append(trial['result']['loss'])
            results['status'].append(trial['result']['status'] == 'ok')
            losses = np.array(results['losses'])
            results['losses'] = list(losses)
            pset = trial['misc']['vals']
            for p in pset.items():
                results[p[0]].append(p[1][0])
        return pd.DataFrame.from_dict(results), self.best

    def print_best(self):
        print("\n")
        print("#" * 40)
        print("###       Best Parameter Choice      ###")
        print("#" * 40)
        for name, value in self.best.items():
            print(" - {}\t:\t{}".format(name, value))
        print("\n - number of iterations\t:\t{}".format(self.trials.trials[-1]['tid']+1))
        print(" - total time\t:\t{}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
                                                              self._total_duration[1],
                                                              self._total_duration[2],
                                                              self._total_duration[3],
                                                              self._total_duration[4]))
        print("#" * 40)

    def compute_time_statistics(self):
        dts = []
        for trial in self._trials.trials:
            if 'book_time' in trial.keys() and 'refresh_time' in trial.keys():
                dt = trial['refresh_time'] - trial['book_time']
                dts.append(dt.total_seconds())
        self._time_per_iteration = np.mean(dts) * 1e3
        self._accumulated_blackbox_time = np.sum(dts) * 1e3
        tmp = self.total_duration - self._accumulated_blackbox_time
        self._solver_overhead = int(np.round(100.0 / (self.total_duration+1e-12) * tmp))

    def print_timestats(self):
        print("\n")
        print("#" * 40)
        print("###        Timing Statistics        ###")
        print("#" * 40)
        print(" - per iteration: {}ms".format(int(self.time_per_iteration*1e4)/10000))
        print(" - total time: {}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
                                                           self._total_duration[1],
                                                           self._total_duration[2],
                                                           self._total_duration[3],
                                                           self._total_duration[4]))
        print("#" * 40)
        print(" - solver overhead: {}%".format(self.solver_overhead))

    def start_viewer(self, port=8097, server="http://localhost"):
        try:
            self._visdom_viewer = VisdomViewer(self._project, port, server)
        except Exception as e:
            import warnings
            warnings.warn("Failed starting VisdomViewer. Is the server running? If not start it via $visdom")
            LOG.error("Failed starting VisdomViewer: {}".format(e))
            self._visdom_viewer = None

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, value):
        if not isinstance(value, HyppopyProject):
            msg = "Input error, project_manager of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._project = value

    @property
    def blackbox(self):
        return self._blackbox

    @blackbox.setter
    def blackbox(self, value):
        if isinstance(value, types.FunctionType) or isinstance(value, BlackboxFunction) or isinstance(value, VirtualFunction):
            self._blackbox = value
        else:
            self._blackbox = None
            msg = "Input error, blackbox of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        if not isinstance(value, dict):
            msg = "Input error, best of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._best = value

    @property
    def trials(self):
        return self._trials

    @trials.setter
    def trials(self, value):
        self._trials = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int):
            msg = "Input error, max_iterations of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        if value < 1:
            msg = "Precondition violation, max_iterations < 1!"
            LOG.error(msg)
            raise IOError(msg)
        self._max_iterations = value

    @property
    def total_duration(self):
        return (self._total_duration[0]*86400 + self._total_duration[1] * 3600 + self._total_duration[2] * 60 + self._total_duration[3]) * 1000 + self._total_duration[4]

    @property
    def solver_overhead(self):
        if self._solver_overhead is None:
            self.compute_time_statistics()
        return self._solver_overhead

    @property
    def time_per_iteration(self):
        if self._time_per_iteration is None:
            self.compute_time_statistics()
        return self._time_per_iteration

    @property
    def accumulated_blackbox_time(self):
        if self._accumulated_blackbox_time is None:
            self.compute_time_statistics()
        return self._accumulated_blackbox_time
