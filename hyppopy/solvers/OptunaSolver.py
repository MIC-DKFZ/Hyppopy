# Hyppopy - A Hyper-Parameter Optimization Toolbox
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

import os
import optuna
import logging
import warnings
import numpy as np
from pprint import pformat

from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver

from hyppopy.CandidateDescriptor import CandidateDescriptor

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class OptunaSolver(HyppopySolver):

    def __init__(self, project=None):
        """
        The constructor accepts a HyppopyProject.

        :param project: [HyppopyProject] project instance, default=None
        """
        HyppopySolver.__init__(self, project)
        self._searchspace = None
        self.candidates_list = list()

    def define_interface(self):
        """
        This function is called when HyppopySolver.__init__ function finished. Child classes need to define their
        individual parameter here by calling the _add_member function for each class member variable need to be defined.
        Using _add_hyperparameter_signature the structure of a hyperparameter the solver expects must be defined.
        Both, members and hyperparameter signatures are later get checked, before executing the solver, ensuring
        settings passed fullfill solver needs.
        """
        self._add_member("max_iterations", int)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)

    def get_candidates(self, trial=None):
        """
        This function converts the searchspace to a candidate_list that can then be used to distribute via MPI.

        :param searchspace: converted hyperparameter space
        """

        candidates_list = list()
        N = self.max_iterations
        for n in range(N):
            # Todo: Ugly hack that does not even work...
            from optuna import trial as trial_module
            # temp_study = optuna.create_study()
            trial_id = self.study._storage.create_new_trial_id(0)
            trial = trial_module.Trial(self.study, trial_id)
            ## trial.report(result)
            ## self._storage.set_trial_state(trial_id, structs.TrialState.COMPLETE)
            ## self._log_completed_trial(trial_number, result)

            params = {}
            for name, param in self._searchspace.items():
                if param["domain"] == "categorical":
                    params[name] = trial.suggest_categorical(name, param["data"])
                else:
                    params[name] = trial.suggest_uniform(name, param["data"][0], param["data"][1])
            candidates_list.append(CandidateDescriptor(**params))

        return candidates_list

        N = self.max_iterations
        for n in range(N):
            params = {}
            for name, param in self._searchspace.items():
                if param["domain"] == "categorical":
                    params[name] = trial.suggest_categorical(name, param["data"])
                else:
                    params[name] = trial.suggest_uniform(name, param["data"][0], param["data"][1])
            candidates_list.append(CandidateDescriptor(**params))

        return candidates_list

    def trial_cache(self, trial):
        """
        Optuna specific loss function wrapper

        :param trial: [Trial] instance

        :return: [function] loss function
        """

        params = {}

        for name, param in self._searchspace.items():
            if param["domain"] == "categorical":
                params[name] = trial.suggest_categorical(name, param["data"])
            else:
                params[name] = trial.suggest_uniform(name, param["data"][0], param["data"][1])

        return self.loss_function(**params)

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and get the output of the latter as input. It's
        purpose is to call the solver libs main optimization function.

        :param searchspace: converted hyperparameter space
        """
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        self._searchspace = searchspace

        try:
            study = optuna.create_study()
            study.optimize(self.trial_cache, n_trials=self.max_iterations)
            self.best = study.best_trial.params
        except Exception as e:
            LOG.error("internal error in bayes_opt maximize occured. {}".format(e))
            raise BrokenPipeError("internal error in bayes_opt maximize occured. {}".format(e))

    def convert_searchspace(self, hyperparameter):
        """
        This function gets the unified hyppopy-like parameterspace description as input and, if necessary, should
        convert it into a solver lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':'float'}, ...}

        :return: [object] converted hyperparameter space
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        for name, param in hyperparameter.items():
            if param["domain"] != "categorical" and param["domain"] != "uniform":
                msg = "Warning: Optuna cannot handle {} domain. Only uniform and categorical domains are supported!".format(param["domain"])
                warnings.warn(msg)
                LOG.warning(msg)
        return hyperparameter
