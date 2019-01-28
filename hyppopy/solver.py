class Solver(object):
    _name = None
    _solver = None
    _parameter = None

    def __init__(self, name=None):
        self.set_name(name)

    def __str__(self):
        txt = f"\nSolver Instance {self._name}:"
        if self._solver is None:
            txt += f"\n - Status solver: None"
        else:
            txt += f"\n - Status solver: ok"
        if self._parameter is None:
            txt += f"\n - Status parameter: None"
        else:
            txt += f"\n - Status parameter: ok"
        return txt

    def is_ready(self):
        return self._solver is not None and self._parameter is not None

    def set_name(self, name):
        self._name = name

    def set_parameter(self, obj):
        self._parameter = obj

    def set_solver(self, obj):
        self._solver = obj

