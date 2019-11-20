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
import logging

from mpi4py import MPI
from hyppopy.globals import DEBUGLEVEL, MPI_TAGS

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class MPISolverWrapper:
    """
    TODO
    The MPISolverWrapper class wraps the functionality of solvers in Hyppopy to extend them with MPI functionality.
    It builds upon the interface defined by the HyppopySolver class.
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
        # TODO: This is ugly.
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank == 0:
            return self._solver.get_results()
        return None, None

    def define_interface(self):
        """
        This function is called when HyppopySolver.__init__ function finished. Child classes need to define their
        individual parameter here by calling the _add_member function for each class member variable need to be defined.
        Using _add_hyperparameter_signature the structure of a hyperparameter the solver expects must be defined.
        Both, members and hyperparameter signatures are later get checked, before executing the solver, ensuring
        settings passed fullfill solver needs.
        """
        self._solver.define_interface()

    def loss_function_call(self, params):
        """
        TODO: Do we even need this here? Don't think so.
        """
        return self._solver.loss_function_call(params)

    def run_worker_mode(self):
        """
        This function is called if the wrapper should run as a worker for a specific MPI rank.
        It receives messages for the following tags:
        tag==MPI_SEND_CANDIDATE: parameters for the loss calculation. It param==None, the worker finishes.
        It sends messages for the following tags:
        tag==MPI_SEND_RESULT: result of an evaluated candidate.

        :return: the evaluated loss of the candidate
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print("Starting worker {}. Waiting for param...".format(rank))

        while True:
            candidate = comm.recv(source=0, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)  # Wait here till params are received

            if candidate == None:
                print(print("process {} received finish signal.".format(rank)))
                return

            id = candidate.ID
            params = candidate._definingValues

            loss = self._solver.blackbox.blackbox_func(None, params)
            # RALF: Ergebnisse müssen zurück geschickt werden und nicht local in einem trial gespeichert werden
            # TODO
            # print('ID: {}, param: {}'.format(id, params))
            comm.send((id,loss), dest=0, tag=MPI_TAGS.MPI_SEND_RESULTS.value)

    @staticmethod
    def signal_worker_finished():
        """
        This function sends data==None to all workers from the master. This is the signal that tells the workers to finish.

        :return:
        """
        print('signal_worker_finished')
        size = MPI.COMM_WORLD.Get_size()
        for i in range(size - 1):
            MPI.COMM_WORLD.send(None, dest=i + 1, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)

    def run(self, *args, **kwargs):
        """
        This function starts the optimization process.
        :param print_stats: [bool] en- or disable console output
        """

        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank == 0:
            # This is the master process. From here we run the solver and start all the other processes.
            self._solver.run(*args, **kwargs)
            self.signal_worker_finished()  # Tell the workers to finish.
        else:
            # this script execution should be in worker mode as it is an mpi worker.
            self.run_worker_mode()

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
        self._solver.print_timestats()
