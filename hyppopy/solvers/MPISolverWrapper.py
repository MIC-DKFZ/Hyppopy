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
    TODO Class description
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
        # Only rank==0 returns results, the workers return None.
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank == 0:
            return self._solver.get_results()
        return None, None

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

        cand_results = dict()
        while True:
            candidate = comm.recv(source=0, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)  # Wait here till params are received

            if candidate is None:
                print("[RECEIVE] Process {} received finish signal.".format(rank))
                return

            # if candidate.ID == 9999:
            #     comm.gather(losses, root=0)
            #     continue

            print("[WORKING] Process {} is actually doing things.".format(rank))
            cand_id = candidate.ID
            params = candidate.get_values()

            cand_results['book_time'] = datetime.datetime.now()
            loss = self._solver.blackbox.blackbox_func(params)
            cand_results['loss'] = loss  # Write loss to dictionary. This dictionary will be send back to the master via gather
            cand_results['refresh_time'] = datetime.datetime.now()

            comm.send((cand_id, cand_results), dest=0, tag=MPI_TAGS.MPI_SEND_RESULTS.value)

    @staticmethod
    def signal_worker_finished():
        """
        This function sends data==None to all workers from the master. This is the signal that tells the workers to finish.

        :return:
        """
        print('[SEND] signal_worker_finished')
        size = MPI.COMM_WORLD.Get_size()
        for i in range(size - 1):
            MPI.COMM_WORLD.send(None, dest=i + 1, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)

    def run(self, *args, **kwargs):
        """
        This function starts the optimization process of the underlying solver and takes care of the MPI awareness.
        """

        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank == 0:
            # This is the master process. From here we run the solver and start all the other processes.
            self._solver.run(*args, **kwargs)
            self.signal_worker_finished()  # Tell the workers to finish.
        else:
            # this script execution should be in worker mode as it is an mpi worker.
            self.run_worker_mode()

    @staticmethod
    def is_master():
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank == 0:
            return True
        else:
            return False

    @staticmethod
    def is_worker(self):
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank != 0:
            return True
        else:
            return False
