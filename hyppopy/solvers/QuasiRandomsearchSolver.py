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
import itertools
import numpy as np
from random import choice
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL
from hyppopy.solvers.HyppopySolver import HyppopySolver

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


def get_gaussian_ranges(a, b, N):
    r = abs(b-a)/2
    if N % 2 == 0:
        _N = int(N/2)
    else:
        _N = int((N-1)/2)
    dr = r/_N
    sigma = r/2.5
    mu = a + r
    cuts = []
    csum = 0
    for n in range(_N):
        x = a+r+n*dr
        c = sigma*np.sqrt(2.0*np.pi)/(np.exp(-0.5*((x-mu)/sigma)**2))
        cuts.append(c)
        cuts.insert(0, c)
        csum += 2*c
    for n in range(len(cuts)):
        cuts[n] /= csum
        cuts[n] *= abs(b-a)
    ranges = []
    end = a
    for n, c in enumerate(cuts):
        start = end
        end = start + c
        ranges.append([start, end])
    return ranges


def get_loguniform_ranges(a, b, N):
    aL = np.log(a)
    bL = np.log(b)
    exp_range = np.linspace(aL, bL, N+1)
    ranges = []
    for i in range(N):
        ranges.append([np.exp(exp_range[i]), np.exp(exp_range[i+1])])
    return ranges


class QuasiRandomSampleGenerator(object):

    def __init__(self, N_samples=None, border_frac=0.1):
        self._grid = None
        self._axis = None
        self._numerical = []
        self._categorical = []
        self._N_samples = N_samples
        self._border_frac = border_frac

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

    def build_grid(self, N_samples=None):
        self._axis = []
        if N_samples is None:
            assert isinstance(self._N_samples, int), "Precondition violation, no number of samples specified!"
        else:
            self._N_samples = N_samples

        if len(self._numerical) > 0:
            axis_steps = int(round(self._N_samples**(1.0/len(self._numerical))))
            self._N_samples = int(axis_steps**(len(self._numerical)))

            for axis in self._numerical:
                self._axis.append(None)
                n = len(self._axis)-1
                boxes = None
                if axis["domain"] == "uniform":
                    boxes = self.add_uniform_axis(n, axis_steps)
                elif axis["domain"] == "normal":
                    boxes = self.add_normal_axis(n, axis_steps)
                elif axis["domain"] == "loguniform":
                    boxes = self.add_loguniform_axis(n, axis_steps)

                assert isinstance(boxes, list), "failed to compute axis ranges!"
                for k in range(len(boxes)):
                    dx = abs(boxes[k][1] - boxes[k][0])
                    boxes[k][0] += self._border_frac * dx
                    boxes[k][1] -= self._border_frac * dx
                self._axis[n] = boxes
            self._grid = list(itertools.product(*self._axis))
        else:
            warnings.warn("No numerical axis defined, this warning can be ignored if searchspace is categorical only, otherwise check if axis was set!")

    def add_uniform_axis(self, n, axis_steps):
        drange = self._numerical[n]["data"]
        width = abs(drange[1]-drange[0])
        dx = width / axis_steps
        boxes = []
        for k in range(1, axis_steps+1):
            bl = drange[0] + (k-1)*dx
            br = drange[0] + k*dx
            boxes.append([bl, br])
        return boxes

    def add_normal_axis(self, n, axis_steps):
        drange = self._numerical[n]["data"]
        boxes = get_gaussian_ranges(drange[0], drange[1], axis_steps)
        for k in range(len(boxes)):
            dx = abs(boxes[k][1] - boxes[k][0])
            boxes[k][0] += self._border_frac * dx
            boxes[k][1] -= self._border_frac * dx
        return boxes

    def add_loguniform_axis(self, n, axis_steps):
        drange = self._numerical[n]["data"]
        boxes = get_loguniform_ranges(drange[0], drange[1], axis_steps)
        for k in range(len(boxes)):
            dx = abs(boxes[k][1] - boxes[k][0])
            boxes[k][0] += self._border_frac * dx
            boxes[k][1] -= self._border_frac * dx
        return boxes

    def next(self):
        if self._grid is None:
            self.build_grid()
        if len(self._grid) == 0:
            return None
        next_index = np.random.randint(0, len(self._grid), 1)[0]
        next_range = self._grid.pop(next_index)
        pset = {}
        for n, rng in enumerate(next_range):
            name = self._numerical[n]["name"]
            rnd = np.random.random()
            param = rng[0] + rnd*abs(rng[1]-rng[0])
            if self._numerical[n]["type"] is int:
                param = int(np.floor(param))
            pset[name] = param
        for cat in self._categorical:
            pset[cat["name"]] = choice(cat["data"])
        return pset


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
                                          options=["uniform", "normal", "loguniform", "categorical"])
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
