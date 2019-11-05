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
import datetime
import os
import logging

from mpi4py import MPI
from hyppopy.globals import DEBUGLEVEL, MPI_TAGS

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class MPISolverWrapper:
    """
    TODO
    """
    def __init__(self, solver=None):
        """
        The constructor accepts a HyppopySolver.

        :param solver: [HyppopySolver] solver instance, default=None
        """
        self._solver = solver

    @property
    def blackbox(self):
        """
        Get the BlackboxFunction object.

        :return: [object] BlackboxFunction instance or function of member solver
        """
        return self._solver.blackbox

    @blackbox.setter
    def blackbox(self, value):
        """
        Set the BlackboxFunction wrapper class encapsulating the loss function or a function accepting a hyperparameter set
        and returning a float.

        :return:
        """
        self._solver.blackbox = value

    def get_results(self):
        """
        Just call get_results of the member solver and return the result.
        :return: return value of self._solver.get_results()
        """

        comm = MPI.COMM_WORLD
        size = comm.size

        # This loop collects the results from the worker processes.
        # If we start this without MPI, this loop is skipped and the results are already in the member variable.
        # Genius? Maybe... or maybe it just turned out this way. :-P
        for i in range(size - 1):
            rec_trials = comm.recv(source=i+1, tag=MPI_TAGS.MPI_SEND_TRIALS.value)
            for trial in rec_trials.trials:
                self._solver._trials.trials.append(trial)
        self._solver.best = self._solver._trials.argmin

        print('Number of processes: {}'.format(size))
        print('Best result: {}'.format(self._solver.best))
        return self._solver.get_results()

    def define_interface(self):
        """
        This function is called when HyppopySolver.__init__ function finished. Child classes need to define their
        individual parameter here by calling the _add_member function for each class member variable need to be defined.
        Using _add_hyperparameter_signature the structure of a hyperparameter the solver expects must be defined.
        Both, members and hyperparameter signatures are later get checked, before executing the solver, ensuring
        settings passed fullfill solver needs.
        """
        self._solver.define_interface()

    def loss_function_call(self, candidates):
        """
        This function is called within the function loss_function and encapsulates the actual blackbox function call
        in each iteration. The function loss_function takes care of the iteration driving and reporting, but each solver
        lib might need some special treatment between the parameter set selection and the calling of the actual blackbox
        function, e.g. parameter converting.

        :param candidates: TODO params [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...} TODO remove

        :return: [float] loss
        """
        try:
            self.call_batch(candidates)
        except:
            for params in candidates:
                self.blackbox(**params)
                # TODO: Why do we need the loss as a return here?
                # loss = self.blackbox(**params)
                # if loss is None:
                #     loss = np.nan
        return

    @staticmethod
    def call_batch(candidates):
        size = MPI.COMM_WORLD.Get_size()
        for i, candidate in enumerate(candidates):
            dest = (i % (size-1)) +1
            MPI.COMM_WORLD.send(candidate, dest=dest, tag=MPI_TAGS.MPI_SEND_DATA.value)

    def call_worker(self):
        """
        This function calls a worker for a specific MPI rank.
        It receives messages for the following tags:
        tag==MPI_SEND_DATA: parameters for the loss calculation. It param==None, the worker finishes.
        It sends messages for the following tags:
        tag==99: trials of this mpi process.

        :return:
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print("Starting worker {}. Waiting for param...".format(rank))

        while True:
            param = comm.recv(source=0, tag=MPI_TAGS.MPI_SEND_DATA.value)  # Wait here till params are received

            # param == None indicates that the worker should finish.
            if param is None:
                MPI.COMM_WORLD.send(self._solver._trials, dest=0, tag=MPI_TAGS.MPI_SEND_TRIALS.value)
                # trials = comm.gather(self._solver._trials, root=0)  # TODO can we solve this with gather as well?

                self._solver.best = self._solver._trials.argmin
                # TODO: Printing does not work for now.
                # self.print_best()
                # self.print_timestats()
                return
            self.loss_function(param)  # No **params here... I overwrote this method.

            # TODO: Do we need this here?
            # loss = self.loss_function(param)  # No **params here... I overwrote this method.
            # print('{}: param = {}, loss={}'.format(rank, param, loss))

    @staticmethod
    def signal_worker_finished():
        """
        This function sends data==None to all workers from the master. This is the signal that tells the workers to finish.

        :return:
        """
        print('signal_worker_finished')
        size = MPI.COMM_WORLD.Get_size()
        for i in range(size - 1):
            MPI.COMM_WORLD.send(None, dest=i + 1, tag=MPI_TAGS.MPI_SEND_DATA.value)

    def run(self):
        """
        This function starts the optimization process.

        TODO:
        This is a copy paste of the HyppopySolver method. Maybe not the most elegant solution, but works for now.

        :param print_stats: [bool] en- or disable console output
        """
        self._idx = 0

        start_time = datetime.datetime.now()
        try:
            search_space = self.convert_searchspace(self._solver.project.hyperparameter)
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

        # TODO: Do print later... Workers might not be finished
        # if print_stats:
        #     self.print_best()
        #     self.print_timestats()

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and get the output of the latter as input. It's
        purpose is to call the solver libs main optimization function.

        :param searchspace: converted hyperparameter space
        """

        candidates_list = self._solver.get_candidate_list(searchspace)

        try:
            self.call_batch(candidates_list)
        except:
            for params in candidates_list:
                self.loss_function(params)  # No **params here... I overwrote this method.

        return

        # results are gathered in the get_results() function

    def convert_searchspace(self, hyperparameter):
        """
        Just a call of the convert_searchspace function of the member solver.

        The function converts the standard parameter input into a range list depending
        on the domain. These rangelists are later used with itertools product to create
        a paramater space sample of each combination.

        :param hyperparameter: [dict] hyperparameter space

        :return: [list] name and range for each parameter space axis
        """

        searchspace = self._solver.convert_searchspace(hyperparameter)

        return searchspace

    def loss_function(self, params):
        return self._solver.loss_function(**params)

    def print_best(self):
        self._solver.print_best()

    def print_timestats(self):
        print('Implement me!')
        # TODO
        # self._solver.print_timestats()
