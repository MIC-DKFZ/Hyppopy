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
from hyppopy.BlackboxFunction import BlackboxFunction

__all__ = ['MPIBlackboxFunction']

import os
import logging
import functools
from hyppopy.globals import DEBUGLEVEL, MPI_TAGS
from mpi4py import MPI

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def default_kwargs(**defaultKwargs):
    """
    Decorator defining default args in **kwargs arguments
    """
    def actual_decorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            defaultKwargs.update(kwargs)
            return fn(*args, **defaultKwargs)
        return g
    return actual_decorator


class MPIBlackboxFunction(BlackboxFunction):
    """
    This class is a BlackboxFunction wrapper class encapsulating the loss function.
    # TODO: complete class documentation
    The constructor accepts several function pointers or a data object which are all None by default (see below).
    Additionally one can define an arbitrary number of arg pairs. These are passed as input to each function pointer as
    arguments.

    :param dataloader_func: data loading function pointer, default=None
    :param preprocess_func: data preprocessing function pointer, default=None
    :param callback_func: callback function pointer, default=None
    :param data: data object, default=None
    :param kwargs: additional arg=value pairs
    """

    @default_kwargs(blackbox_func=None, dataloader_func=None, preprocess_func=None, callback_func=None, data=None, mpi_comm=None)
    def __init__(self, **kwargs):
        mpi_comm = kwargs['mpi_comm']
        del kwargs['mpi_comm']
        self._mpi_comm = None

        if mpi_comm is None:
            print('MPIBlackboxFunction: No mpi_comm given: Using MPI.COMM_WORLD')
            self._mpi_comm = MPI.COMM_WORLD
        else:
            self._mpi_comm = mpi_comm

        super().__init__(**kwargs)

    @staticmethod
    def call_batch(candidates):
        results = dict()
        size = MPI.COMM_WORLD.Get_size()

        for i, candidate in enumerate(candidates):
            dest = (i % (size-1)) + 1
            MPI.COMM_WORLD.send(candidate, dest=dest, tag=MPI_TAGS.MPI_SEND_CANDIDATE.value)

        while True:
            for i in range(size - 1):
                if len(candidates) == len(results):
                    print('All results received!')
                    return results
                cand_id, result_dict = MPI.COMM_WORLD.recv(source=i + 1, tag=MPI_TAGS.MPI_SEND_RESULTS.value)
                results[cand_id] = result_dict