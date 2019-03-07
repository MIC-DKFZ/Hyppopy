# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical and Biological Informatics.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE.txt or http://www.mitk.org for details.
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import copy
import time
import itertools
import numpy as np
from numpy import argmin, argmax, unique
from collections import OrderedDict, abc


def gaussian(x, mu, sigma):
    return 1.0/(sigma * np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))


def gaussian_axis_sampling(a, b, N):
    center = a + (b - a) / 2.0
    delta = (b - a) / N
    bn = b - center
    xn = np.arange(0, bn, delta)
    dn = []
    for x in xn:
        dn.append(1/gaussian(x, 0, bn/2.5))
    dn = np.array(dn)
    dn /= np.sum(dn)
    dn *= bn

    axis = [0]
    for x in dn:
        axis.append(x+axis[-1])
        axis.insert(0, -axis[-1])
    axis = np.array(axis)
    axis += center
    return axis


def log_axis_sampling(a, b, N):
    if a == 0:
        a += 1e-23
    assert a > 0, "Precondition Violation, a < 0!"
    assert a < b, "Precondition Violation, a > b!"
    assert b > 0, "Precondition Violation, b < 0!"
    lexp = np.log(a)
    rexp = np.log(b)
    assert lexp is not np.nan, "Precondition violation, left bound input error, results in nan!"
    assert rexp is not np.nan, "Precondition violation, right bound input error, results in nan!"

    delta = (rexp - lexp) / N
    logrange = np.arange(lexp, rexp + delta, delta)
    for n in range(logrange.shape[0]):
        logrange[n] = np.exp(logrange[n])
    return logrange


def sample_domain(start, stop, count, ftype="uniform"):
    assert stop > start, "Precondition Violation, stop <= start not allowed!"
    assert count > 0, "Precondition Violation, N <= 0 not allowed!"
    if ftype == 'uniform':
        delta = (stop - start)/count
        return np.arange(start, stop + delta, delta)
    elif ftype == 'loguniform':
        return log_axis_sampling(start, stop, count)
    elif ftype == 'normal':
        return gaussian_axis_sampling(start, stop, count)
    raise IOError("Precondition Violation, unknown sampling function type!")


class Trials(object):

    def __init__(self):
        self.loss = []
        self.duration = []
        self.status = []
        self.parameter = []
        self.best = None
        self._tick = None

    def start_iteration(self):
        self._tick = time.process_time()

    def stop_iteration(self):
        if self._tick is None:
            return
        self.duration.append(time.process_time()-self._tick)
        self._tick = None

    def set_status(self, status=True):
        self.status.append(status)

    def set_parameter(self, params):
        self.parameter.append(params)

    def set_loss(self, value):
        self.loss.append(value)

    def get(self):
        msg = None
        if len(self.loss) <= 0:
            msg = "Empty solver results!"
        if len(self.loss) != len(self.duration):
            msg = "Inconsistent results! len(self.loss) != len(self.duration) -> {} != {}".format(len(self.loss), len(self.duration))
        if len(self.loss) != len(self.parameter):
            msg = "Inconsistent results! len(self.loss) != len(self.parameter) -> {} != {}".format(len(self.loss), len(self.parameter))
        if len(self.loss) != len(self.status):
            msg = "Inconsistent results! len(self.loss) != len(self.status) -> {} != {}".format(len(self.loss), len(self.status))
        if msg is not None:
            raise Exception(msg)

        best_index = argmin(self.loss)
        best = self.parameter[best_index]
        worst_loss = self.loss[argmax(self.loss)]
        for n in range(len(self.status)):
            if not self.status[n]:
                self.loss[n] = worst_loss

        res = {
            'losses': self.loss,
            'duration': self.duration
        }
        is_string = []
        for key, value in self.parameter[0].items():
            res[key] = []
            if isinstance(value, str):
                is_string.append(key)

        for p in self.parameter:
            for key, value in p.items():
                res[key].append(value)

        for key in is_string:
            uniques = unique(res[key])
            lookup = {}
            for n, p in enumerate(uniques):
                lookup[p] = n
            for n in range(len(res[key])):
                res[key][n] = lookup[res[key][n]]

        return res, best


class NestedDictUnfolder(object):

    def __init__(self, nested_dict):
        self._nested_dict = nested_dict
        self._categories = []
        self._values = OrderedDict()
        self._tree_leafs = []

        NestedDictUnfolder.nested_dict_iter(self._nested_dict, self)

    @staticmethod
    def nested_dict_iter(nested, unfolder):
        for key, value in nested.items():
            if isinstance(value, abc.Mapping):
                unfolder.add_category(key)
                NestedDictUnfolder.nested_dict_iter(value, unfolder)
            else:
                unfolder.add_values(key, value)
                unfolder.mark_leaf()

    def find_parent_nodes(self, nested, node, last_node=""):
        for key, value in nested.items():
            if key == node:
                self._tree_leafs.append(last_node)
                return
            else:
                last_node = key
            if isinstance(value, abc.Mapping):
                self.find_parent_nodes(value, node, last_node)
            else:
                return

    def find_parent_node(self, leaf_names):
        if not isinstance(leaf_names, list):
            leaf_names = [leaf_names]
        for ln in leaf_names:
            try:
                pos = self._categories.index(ln) - 1
                candidate = self._categories[pos]
                if candidate not in leaf_names:
                    return candidate
            except:
                pass
        return None

    def add_category(self, name):
        self._categories.append(name)

    def add_values(self, name, values):
        self._values[name] = values

    def mark_leaf(self):
        if len(self._categories) > 0:
            if not self._categories[-1] in self._tree_leafs:
                self._tree_leafs.append(self._categories[-1])

    def permutate_values(self):
        pset = list(self._values.values())
        pset = list(itertools.product(*pset))
        permutations = []
        okeys = list(self._values.keys())
        for ps in pset:
            permutations.append({})
            for i in range(len(okeys)):
                permutations[-1][okeys[i]] = ps[i]
        return permutations

    def add_categories(self, values_permutated):
        while True:
            parent = self.find_parent_node(self._tree_leafs)
            if parent is None:
                return
            result = []
            for tl in self._tree_leafs:
                for elem in values_permutated:
                    new = copy.deepcopy(elem)
                    new[parent] = tl
                    result.append(new)
                while tl in self._categories:
                    self._categories.remove(tl)
            while parent in self._categories:
                self._categories.remove(parent)
            self._tree_leafs = []
            self.find_parent_nodes(self._nested_dict, parent)
            if len(self._tree_leafs) == 1 and self._tree_leafs[0] == "":
                break
            values_permutated = copy.deepcopy(result)
        return result

    def unfold(self):
        values_permutated = self.permutate_values()
        if len(self._categories) > 0:
            return self.add_categories(values_permutated)
        return values_permutated
