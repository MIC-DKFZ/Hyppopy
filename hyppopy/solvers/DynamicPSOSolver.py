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
import sys
import numpy
import datetime
import logging
import optunity
from pprint import pformat

from hyppopy.CandidateDescriptor import CandidateDescriptor, CandicateDescriptorWrapper
from hyppopy.globals import DEBUGLEVEL
from hyperopt import Trials

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from hyppopy.solvers.HyppopySolver import HyppopySolver
from .OptunitySolver import OptunitySolver

class DynamicPSOSolver(OptunitySolver):
    """Dynamic PSO HyppoPy Solver Class"""
    
    def define_interface(self):
        """
        Function called after instantiation to define individual parameters for child solver class by calling
        _add_member function for each class member variable to be defined. When designing your own solver class,
        you need to implement this method to define custom solver options that are automatically converted
        to class attributes.
        """
        super().define_interface()
        self._add_method("update_param")                # Pass function used to adapt parameters during dynamic PSO as specified by user.
        self._add_method("combine_obj")                 # Pass function indicating how to combine obj. func. arguments and parameters to obtain scalar value.
        self._add_member("num_args_obj", int)           # Pass number of arguments/terms contributing to obj. func.
        self._add_member("num_params_obj", int)         # Pass number of parameters of obj. func.
        self._add_member("phi1", float, default=1.5)    # Pass first PSO acceleration coefficient.
        self._add_member("phi2", float, default=2.0)    # Pass second PSO acceleration coefficient.
        self._add_hyperparameter_signature(name="domain", dtype=str, options=["uniform", "loguniform", "categorical"])

    def _add_method(self, name, func=None, default=None):
        """
        When designing your child solver class you need to implement the define_interface abstract method where you can
        call _add_member_function to define custom solver options, here of Python callable type, which are automatically 
        converted to class methods.

        :param func: [callable] function object to be passed to solver
        """
        assert isinstance(name, str), "Precondition violation, name needs to be of type str, got {}.".format(type(name))
        if func is not None:
            assert callable(func), "Precondition violation, passed object is not callable!"
        if default is not None:
            assert callable(default), "Precondition violation, passed object is not callable!"
        setattr(self, name, func)
        self._child_members[name] = {"type": "callable", "function": func, "default": default}

    def convert_searchspace(self, hyperparameter):
        """
        Get unified hyppopy-like parameter space description as input and, if necessary,
        convert it into a solver-lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':float}, ...}

        :return: [object] converted hyperparameter space
        :return: [dict] dict keeping domains for different hyperparameters.
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        # Split input in categorical and non-categorical data.
        cat, uni = self.split_categorical(hyperparameter)
        # Build up dict keeping all non-categorical data.
        uniforms = {}
        domains = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == "data":
                    if len(value2) == 3:
                        uniforms[key] = value2[0:2]
                    elif len(value2) == 2:
                        uniforms[key] = value2
                    else:
                        raise AssertionError("precondition violation, optunity searchspace needs list with left and right range bounds!")
                if key2 == "domain":
                    domains[key] = value2

        if len(cat) == 0:
            return uniforms, domains
        # Build nested categorical structure.
        inner_level = uniforms
        for key, value in cat.items():
            tmp = {}
            optunity_space = {}
            for key2, value2 in value.items():
                if key2 == "data":
                    for elem in value2:
                        tmp[elem] = inner_level
                if key2 == "domain":
                    domains[key] = value2
            optunity_space[key] = tmp
            inner_level = optunity_space
        return optunity_space, domains

    def hyppopy_optunity_solver_pmap(self, f, seq):
        # Check if seq is empty. I so, return an empty result list.
        if len(seq) == 0:
            return []

        candidates = []
        for elem in seq:
            can = CandidateDescriptor(**elem)
            candidates.append(can)

        cand_list = CandicateDescriptorWrapper(keys=seq[0].keys())
        cand_list.set(candidates)

        f_result = f(cand_list)

        # If one candidate does not match the constraints, f() returns a single default value.
        # This is a problem as all the other candidates are not calculated either.
        # The following is a workaround. We split the candidate_list into 2 lists and call the map function recursively until all valid parameters are processed.
        if not isinstance(f_result, list):
            # First half
            seq_A = seq[:len(seq) // 2]
            temp_result_a = self.hyppopy_optunity_solver_pmap(f, seq_A)

            seq_B = seq[len(seq) // 2:]
            temp_result_b = self.hyppopy_optunity_solver_pmap(f, seq_B)
            # f_result = [42]

            f_result = temp_result_a + temp_result_b

        return f_result

    def execute_solver(self, searchspace, domains):
        """
        This function is called immediately after convert_searchspace and uses the output of the latter as input. Its
        purpose is to call the solver lib's main optimization function.

        :param searchspace: converted hyperparameter space
        """
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        tree = optunity.search_spaces.SearchTree(searchspace)   # Set up tree structure to model search space.
        box = tree.to_box()                                     # Create set of box constraints to define given search space.
        f = optunity.functions.logged(self.loss_function_batch)       # Call log here because function signature used later on is internal logic.
        f = tree.wrap_decoder(f)                                # Wrap decoder and constraints for internal search space rep.
        f = optunity.constraints.wrap_constraints(f, default=sys.float_info.max*numpy.ones(self.num_args_obj), range_oo=box)
        # 'wrap_constraints' decorates function f with given input domain constraints. default [float] gives a 
        # function value to default to in case of constraint violations. range_oo [dict] gives open range 
        # constraints lb and lu, i.e. lb < x < ub and range = (lb, ub), respectively.

        try:
            self.best, _ = optunity.optimize_dyn_PSO(func=f,
                                                     box=box,
                                                     domains=domains,
                                                     maximize=False,
                                                     max_evals=self.max_iterations,
                                                     num_args_obj=self.num_args_obj,
                                                     num_params_obj=self.num_params_obj,
                                                     pmap=self.hyppopy_optunity_solver_pmap, #map,#optunity.pmap,
                                                     decoder=tree.decode,
                                                     update_param=self.update_param,
                                                     eval_obj=self.combine_obj,   
                                                     phi1=self.phi1,
                                                     phi2=self.phi2
                                                     )
            # Workaround: Unpack best result, im max_iterations was reached.
            try:
                for key in self.best:
                    self.best[key] = self.best[key].get()[0]
            except:
                pass
            """
            optimize_dyn_PSO(func, maximize=False, max_evals=0, pmap=map, decoder=None, update_param=None, eval_obj=None)
            Optimize func with dynamic PSO solver.
            :param func: [callable] objective function
            :param maximize: [bool] maximize or minimize
            :param max_evals: [int] maximum number of permitted function evaluations
            :param pmap: [function] map() function to use
            :param update_param: [function] function to update parameters of objective function
                                 based on current state of knowledge
            :param eval_obj: [function] function giving functional form of objective function, i.e.
                             how to combine parameters and terms to obtain scalar fitness/loss.
            
            :return: solution, named tuple with further details
            optimize_dyn_PSO function (api.py) internally uses 'optimize' function from dynamic PSO solver module.
            """
        except Exception as e:
            LOG.error("Internal error in optunity.optimize_dyn_PSO occured. {}".format(e))
            raise BrokenPipeError("Internal error in optunity.optimize_dyn_PSO occured. {}".format(e))

    def print_best(self):
        """
        Optimization result console output printing.
        """
        print("\n")
        print("#" * 40)
        print("###       Best Parameter Choice      ###")
        print("#" * 40)
        for name, value in self.best.items():
            print(" - {}\t:\t{}".format(name, value))
        #print("\n - number of iterations\t:\t{}".format(self.trials.trials[-1]['tid']+1))
        #print(" - total time\t:\t{}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
        #                                                      self._total_duration[1],
        #                                                      self._total_duration[2],
        #                                                      self._total_duration[3],
        #                                                      self._total_duration[4]))
        print("#" * 40)

    def run(self, print_stats=True):
        """
        This function starts the optimization process.
        :param print_stats: [bool] en- or disable console output
        """
        self._idx = 0
        self.trials = Trials()

        start_time = datetime.datetime.now()
        try:
            search_space, domains = self.convert_searchspace(self.project.hyperparameter)
        except Exception as e:
            msg = "Failed to convert searchspace, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
        try:
            self.execute_solver(search_space, domains)
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
        self.print_best()
        if print_stats:
            self.print_timestats()

