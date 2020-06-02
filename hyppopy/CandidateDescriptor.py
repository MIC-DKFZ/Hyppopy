class CandidateDescriptor(object):
    """
    Descriptor that defines an candidate the solver wants to be checked.
    It is used to lable/identify the candidates and their results in the case of batch processing.
    """

    def __init__(self, **definingValues):
        """
        @param definingValues Class assumes that all variables passed to the computer are parameters of the candidate
        the instance should represent.
        """
        import uuid

        self._definingValues = definingValues

        self._definingStr = str()

        for item in sorted(definingValues.items()):
            self._definingStr = self._definingStr + "'" + str(item[0]) + "':'" + str(item[1]) + "',"

        self.ID = str(uuid.uuid4())

    def __missing__(self, key):
        return None

    def __len__(self):
        return len(self._definingValues)

    def __contains__(self, key):
        return key in self._definingValues

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._definingValues == other._definingValues
        else:
            return False

    def __hash__(self):
        return hash(self._definingStr)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'EvalInstanceDescriptor(%s)' % (self._definingValues)

    def __str__(self):
        return '(%s)' % (self._definingValues)

    def keys(self):
        return self._definingValues.keys()

    def __getitem__(self, key):
        if key in self._definingValues:
            return self._definingValues[key]
        raise KeyError('Unkown defining value key was requested. Key: {}; self: {}'.format(key, self))

    def get_values(self):
        return self._definingValues


class CandicateDescriptorWrapper:

    class InternalCandidateValueWrapper:
        def __init__(self, value_list):
            self._value_list = value_list

        def __gt__(self, other):
            boundary_condition = True
            for value in self._value_list:
                if value > other:
                    continue
                else:
                    boundary_condition = False
                    break
            return boundary_condition

        def __lt__(self, other):
            boundary_condition = True
            for value in self._value_list:
                if value < other:
                    continue
                else:
                    boundary_condition = False
                    break
            return boundary_condition

        def get(self):
            return self._value_list

    def __init__(self, keys):
        self._cand = None
        self._keys = keys

    def __iter__(self):
        return iter(self._cand)

    def __getitem__(self, key):
        return self.InternalCandidateValueWrapper([x[key] for x in self._cand])

    def keys(self):
        return self._keys

    def set(self, obj):
        self._cand = obj

    def get(self):
        return self._cand
