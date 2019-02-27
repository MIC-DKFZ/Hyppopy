import copy
import itertools
from collections import OrderedDict, abc


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
