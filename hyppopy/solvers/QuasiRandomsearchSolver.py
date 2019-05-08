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

__all__ = ['HaltonSequenceGenerator', 'QuasiRandomSampleGenerator', 'QuasiRandomsearchSolver']

import os
import logging
import warnings
import numpy as np
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HaltonSequenceGenerator(object):
    """
    This class generates Halton sequences (https://en.wikipedia.org/wiki/Halton_sequence). The class needs a total
    number of samples and the number of dimensions to generate a quasirandom sequence for each axis. The method
    get_unit_space returns a sequence list with N_samples for each axis representing N_samples vectors on a unit sphere.
    """
    def __init__(self):
        pass

    def __next_prime(self):
        """
        Checks if num is a prime value
        """
        def is_prime(num):
            for i in range(2, int(num ** 0.5) + 1):
                if (num % i) == 0: return False
            return True

        prime = 3
        while 1:
            if is_prime(prime):
                yield prime
            prime += 2

    def __vdc(self, n, base):
        vdc, denom = 0, 1
        while n:
            denom *= base
            n, remainder = divmod(n, base)
            vdc += remainder / float(denom)
        return vdc

    def get_unit_space(self, N_samples, N_dims):
        """
        Returns a unit space in form of a sequence list keeping N_dims sequences with N_sample samplings. Each sample
        represents a N_dims dimensional vector on a unit sphere.

        :param N_samples: [int] Number of samples
        :param N_dims: [int] Number of dimensions

        :return: [list] samples list of length N_dims keeping lists each of length N_samples
        """
        seq = []
        primeGen = self.__next_prime()
        next(primeGen)
        for d in range(N_dims):
            base = next(primeGen)
            seq.append([self.__vdc(i, base) for i in range(N_samples)])
        return seq


class QuasiRandomSampleGenerator(object):
    """
    This class takes care of the hyperparameter space creation and next sample delivery.
    """
    def __init__(self, N_samples=None):
        self._axis = None
        self._samples = []
        self._numerical = []
        self._categorical = []
        self._N_samples = N_samples

    def set_axis(self, name, data, domain, dtype):
        """
        Add an axis description.

        :param name: [str] axis name
        :param data: [list] axis range [min, max]
        :param domain: [str] axis domain
        :param dtype: [type] axis data type
        """
        if domain == "categorical":
            if dtype is int:
                data = [int(i) for i in data]
            elif dtype is str:
                data = [str(i) for i in data]
            elif dtype is float:
                data = [float(i) for i in data]
            self._categorical.append({"name": name, "data": data, "type": dtype})
        else:
            self._numerical.append({"name": name, "data": data, "type": dtype, "domain": domain})

    def generate_samples(self, N_samples=None):
        """
        This function is called once when the first sample is requested. It generates the halton sequence space.

        :param N_samples: [int] number of samples
        """
        self._axis = []
        if N_samples is None:
            assert isinstance(self._N_samples, int), "Precondition violation, no number of samples specified!"
        else:
            self._N_samples = N_samples

        axis_samples = {}
        if len(self._numerical) > 0:
            generator = HaltonSequenceGenerator()
            unit_space = generator.get_unit_space(self._N_samples, len(self._numerical))
            for n, axis in enumerate(self._numerical):
                width = abs(axis["data"][1] - axis["data"][0])
                unit_space[n] = [x * width for x in unit_space[n]]
                unit_space[n] = [x + axis["data"][0] for x in unit_space[n]]
                if axis["type"] is int:
                    unit_space[n] = [int(round(x)) for x in unit_space[n]]
                axis_samples[axis["name"]] = unit_space[n]
        else:
            warnings.warn("No numerical axis defined, this warning can be ignored if searchspace is categorical only, otherwise check if axis was set!")

        for n in range(self._N_samples):
            sample = {}
            for name, data in axis_samples.items():
               sample[name] = data[n]
            for cat in self._categorical:
                choice = np.random.choice(len(cat["data"]), 1)[0]
                sample[cat["name"]] = cat["data"][choice]
            self._samples.append(sample)

    def next(self):
        """
        Returns the next sample. Returns None if all samples are requested.

        :return: [dict] sample dict {'name':value, ...}
        """
        if len(self._samples) == 0:
            self.generate_samples()
        if len(self._samples) == 0:
            return None
        next_index = np.random.choice(len(self._samples), 1)[0]
        sample = self._samples.pop(next_index)
        return sample


class QuasiRandomsearchSolver(HyppopySolver):
    """
    The QuasiRandomsearchSolver class implements a quasi randomsearch optimization. The quasi randomsearch supports
    categorical and uniform sampling. The solver defines a Halton Sequence distributed hyperparameter space. This
    means a rather evenly distributed space sampling but no real randomness.
    """
    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._sampler = None

    def define_interface(self):
        self._add_member("max_iterations", int)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)

    def loss_function_call(self, params):
        loss = self.blackbox(**params)
        if loss is None:
            return np.nan
        return loss

    def execute_solver(self, searchspace):
        N = self.max_iterations
        self._sampler = QuasiRandomSampleGenerator(N)
        for name, axis in searchspace.items():
            self._sampler.set_axis(name, axis["data"], axis["domain"], axis["type"])
        try:
            for n in range(N):
                params = self._sampler.next()
                if params is None:
                    break
                self.loss_function(**params)
        except Exception as e:
            msg = "internal error in randomsearch execute_solver occured. {}".format(e)
            LOG.error(msg)
            raise BrokenPipeError(msg)
        self.best = self._trials.argmin

    def convert_searchspace(self, hyperparameter):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        return hyperparameter
