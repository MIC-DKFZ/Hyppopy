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


class MPIBlackboxFunction(object):
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

    @default_kwargs(blackbox_func=None, dataloader_func=None, preprocess_func=None, callback_func=None, data=None)
    def __init__(self, **kwargs):
        self._blackbox_func = None
        self._preprocess_func = None
        self._dataloader_func = None
        self._callback_func = None
        self._raw_data = None
        self._data = None
        self.setup(kwargs)

    def __call__(self, **kwargs):
        """
        Call method calls blackbox_func passing the data object and the args passed

        :param kwargs: [dict] args

        :return: blackbox_func(data, kwargs)
        """
        return self.blackbox_func(self.data, kwargs)

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

    def setup(self, kwargs):
        """
        Alternative to Constructor, kwargs signature see __init__

        :param kwargs: (see __init__)
        """
        self._blackbox_func = kwargs['blackbox_func']
        del kwargs['blackbox_func']

    @property
    def blackbox_func(self):
        """
        BlackboxFunction wrapper class encapsulating the loss function or a function accepting a hyperparameter set and
        returning a float.

        :return: [object] pointer to blackbox_func
        """
        return self._blackbox_func

    @property
    def preprocess_func(self):
        """
        Data preprocessing is called after dataloader_func, the functions signature must be foo(data, params) and must
        return a data object. The input is the data object set directly or via dataloader_func, the params are passed
        from constructor params.

        :return: [object] preprocess_func
        """
        return self._preprocess_func

    @property
    def dataloader_func(self):
        """
        Data loading, the function must return a data object and is called first when the solver is executed. The data
        object returned will be the input of the blackbox function.

        :return: [object] dataloader_func
        """
        return self._dataloader_func

    @property
    def callback_func(self):
        """
        This function is called at each iteration step getting passed the trail info content, can be used for
        custom visualization

        :return: [object] callback_func
        """
        return self._callback_func

    @property
    def raw_data(self):
        """
        This data structure is used to store the return from dataloader_func to serve as input for preprocess_func if
        available.

        :return: [object] raw_data
        """
        return self._raw_data

    @property
    def data(self):
        """
        Datastructure keeping the input data.

        :return: [object] data
        """
        return self._data
