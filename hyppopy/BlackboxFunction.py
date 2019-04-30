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
import functools
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def default_kwargs(**defaultKwargs):
    def actual_decorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            defaultKwargs.update(kwargs)
            return fn(*args, **defaultKwargs)
        return g
    return actual_decorator


class BlackboxFunction(object):

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
        return self.blackbox_func(self.data, kwargs)

    def setup(self, kwargs):
        self._blackbox_func = kwargs['blackbox_func']
        self._preprocess_func = kwargs['preprocess_func']
        self._dataloader_func = kwargs['dataloader_func']
        self._callback_func = kwargs['callback_func']
        self._raw_data = kwargs['data']
        self._data = self._raw_data
        del kwargs['blackbox_func']
        del kwargs['preprocess_func']
        del kwargs['dataloader_func']
        del kwargs['data']
        params = kwargs

        if self.dataloader_func is not None:
            self._raw_data = self.dataloader_func(params=params)
        assert self._raw_data is not None, "Missing data exception!"
        assert self.blackbox_func is not None, "Missing blackbox fucntion exception!"
        if self.preprocess_func is not None:
            result = self.preprocess_func(data=self._raw_data, params=params)
            if result is not None:
                self._data = result
            else:
                self._data = self._raw_data
        else:
            self._data = self._raw_data

    @property
    def blackbox_func(self):
        return self._blackbox_func

    @property
    def preprocess_func(self):
        return self._preprocess_func

    @property
    def dataloader_func(self):
        return self._dataloader_func

    @property
    def callback_func(self):
        return self._callback_func

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data
