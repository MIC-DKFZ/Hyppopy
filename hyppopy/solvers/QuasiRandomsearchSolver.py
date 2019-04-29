# DKFZ
#
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
import warnings
import numpy as np
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def get_loguniform_ranges(a, b, N):
    aL = np.log(a)
    bL = np.log(b)
    exp_range = np.linspace(aL, bL, N+1)
    ranges = []
    for i in range(N):
        ranges.append([np.exp(exp_range[i]), np.exp(exp_range[i+1])])
    return ranges


class HaltonSequenceGenerator(object):

    def __init__(self, N_samples, dimensions):
        self._N = N_samples
        self._dims = dimensions

    def next_prime(self):
        def is_prime(num):
            "Checks if num is a prime value"
            for i in range(2, int(num ** 0.5) + 1):
                if (num % i) == 0: return False
            return True

        prime = 3
        while 1:
            if is_prime(prime):
                yield prime
            prime += 2

    def vdc(self, n, base):
        vdc, denom = 0, 1
        while n:
            denom *= base
            n, remainder = divmod(n, base)
            vdc += remainder / float(denom)
        return vdc

    def get_sequence(self):
        seq = []
        primeGen = self.next_prime()
        next(primeGen)
        for d in range(self._dims):
            base = next(primeGen)
            seq.append([self.vdc(i, base) for i in range(self._N)])
        return seq


class QuasiRandomSampleGenerator(object):

    def __init__(self, N_samples=None):
        self._axis = None
        self._samples = []
        self._numerical = []
        self._categorical = []
        self._N_samples = N_samples

    def set_axis(self, name, data, domain, dtype):
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
        self._axis = []
        if N_samples is None:
            assert isinstance(self._N_samples, int), "Precondition violation, no number of samples specified!"
        else:
            self._N_samples = N_samples

        axis_samples = {}
        if len(self._numerical) > 0:
            generator = HaltonSequenceGenerator(self._N_samples, len(self._numerical))
            unit_space = generator.get_sequence()
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
    categorical, uniform, normal and loguniform sampling. The solver defines a grid which size and appearance depends
    on the max_iterations parameter and the domain. The at each grid box a random value is drawn. This ensures both,
    random parameter samples with the cosntraint that the space is evenly sampled and cluster building prevention."""
    def __init__(self, project=None):
        HyppopySolver.__init__(self, project)
        self._sampler = None

    def define_interface(self):
        self.add_member("max_iterations", int)
        self.add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self.add_hyperparameter_signature(name="data", dtype=list)
        self.add_hyperparameter_signature(name="type", dtype=type)

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
        """
        this function simply pipes the input parameter through, the sample
        drawing functions are responsible for interpreting the parameter.
        :param hyperparameter: [dict] hyperparameter space
        :return: [dict] hyperparameter space
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        return hyperparameter
