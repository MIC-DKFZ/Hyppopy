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

import numpy as np
from mpi4py import MPI
from hyppopy.globals import DEBUGLEVEL, MPI_TAGS
from hyppopy.MPIBlackboxFunction import MPIBlackboxFunction

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class MPISolverWrapper:
    """
    TODO Class description
    The MPISolverWrapper class wraps the functionality of solvers in Hyppopy to extend them with MPI functionality.
    It builds upon the interface defined by the HyppopySolver class.
    """
    def __init__(self, solver=None, mpi_comm=None):
        """
        The constructor accepts a HyppopySolver.

        :param solver: [HyppopySolver] solver instance, default=None
        :param mpi_comm: [MPI communicator] MPI communicator instance. If None, we create a new MPI.COMM_WORLD, default=None
        """
        self._solver = solver
        self._mpi_comm = None
        if mpi_comm is None:
            print('MPISolverWrapper: No mpi_comm given: Using MPI.COMM_WORLD')
            self._mpi_comm = MPI.COMM_WORLD
        else:
            self._mpi_comm = mpi_comm

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
        Set the BlackboxFunction wrapper class encapsulating the loss function or a function accepting a hyperparameter
        set and returning a float.
        If the passed value is not an instance of MPIBlackboxFunction (or a derived class) it will automatically
        wrapped by an MPIBackboxFunction.
        :return:
        """
        if isinstance(value, MPIBlackboxFunction):
            self._solver.blackbox = value
        else:
            self._solver.blackbox = MPIBlackboxFunction(blackbox_func=value, mpi_comm=self._mpi_comm)

    def get_results(self):
        """
        Just call get_results of the member solver and return the result.
        :return: return value of self._solver.get_results()
        """
        # Only rank==0 returns results, the workers return None.
        mpi_rank = self._mpi_comm.Get_rank()
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
        rank = self._mpi_comm.Get_rank()
        print("Starting worker {}. Waiting for param...".format(rank))

        cand_results = dict()

        while True:
            try:
                candidate = self._mpi_comm.recv(source=0, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)  # Wait here till params are received

                if candidate is None:
                    print("[RECEIVE] Process {} received finish signal.".format(rank))
                    return

                # if candidate.ID == 9999:
                #     comm.gather(losses, root=0)
                #     continue

                # print("[WORKING] Process {} is actually doing things.".format(rank))
                cand_id = candidate.ID
                params = candidate.get_values()

                loss = self._solver.blackbox.blackbox(params)

            except Exception as e:
                msg = "Error in Worker(rank={}): {}".format(rank, e)
                LOG.error(msg)
                print(msg)

                loss = np.nan
            finally:
                cand_results['book_time'] = datetime.datetime.now()
                cand_results['loss'] = loss  # Write loss to dictionary. This dictionary will be send back to the master via gather
                cand_results['refresh_time'] = datetime.datetime.now()

                cand_results['book_time'] = datetime.datetime.now()

                cand_results['loss'] = loss  # Write loss to dictionary. This dictionary will be send back to the master via gather
                cand_results['refresh_time'] = datetime.datetime.now()

                self._mpi_comm.send((cand_id, cand_results), dest=0, tag=MPI_TAGS.MPI_SEND_RESULTS.value)

    def signal_worker_finished(self):
        """
        This function sends data==None to all workers from the master. This is the signal that tells the workers to finish.

        :return:
        """
        print('[SEND] signal_worker_finished')
        size = self._mpi_comm.Get_size()
        for i in range(size - 1):
            self._mpi_comm.send(None, dest=i + 1, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)

    def run(self, *args, **kwargs):
        """
        This function starts the optimization process of the underlying solver and takes care of the MPI awareness.
        """

        mpi_rank = self._mpi_comm.Get_rank()
        if mpi_rank == 0:
            # This is the master process. From here we run the solver and start all the other processes.
            self._solver.run(*args, **kwargs)
            self.signal_worker_finished()  # Tell the workers to finish.
        else:
            # this script execution should be in worker mode as it is an mpi worker.
            self.run_worker_mode()

    def is_master(self):
        mpi_rank = self._mpi_comm.Get_rank()
        if mpi_rank == 0:
            return True
        else:
            return False

    def is_worker(self):
        mpi_rank = self._mpi_comm.Get_rank()
        if mpi_rank != 0:
            return True
        else:
            return False
