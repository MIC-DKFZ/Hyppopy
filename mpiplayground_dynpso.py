# Minimal setup to test dynPSO code: Reproduce normal PSO with dynamic PSO.

# Insert path to Marie's optunity
import sys

dir = None
assert dir != None, 'Please adapt the path to the location of specialized Optunity'
sys.path.insert(1, dir)


from mpi4py import MPI

from hyppopy.MPIBlackboxFunction import MPIBlackboxFunction
from hyppopy.solvers.MPISolverWrapper import MPISolverWrapper

import hyppopy.HyppopyProject
import hyppopy.solvers.DynamicPSOSolver
import numpy


def updateParam(pop_history, num_params_obj):
    return numpy.ones(num_params_obj)


def combineObj(args, params):
    return sum([a * p for a, p in zip(args, params)])


def f(x, y):
    return [x ** 2, y ** 2]


project = hyppopy.HyppopyProject.HyppopyProject()
project.add_hyperparameter(name="x", domain="uniform", data=[-10, 10], type=float)
project.add_hyperparameter(name="y", domain="uniform", data=[-10, 10], type=float)
project.add_setting(name="max_iterations", value=300)
project.add_setting(name="num_params_obj", value=2)
project.add_setting(name="num_args_obj", value=2)
project.add_setting(name="combine_obj", value=combineObj)
project.add_setting(name="update_param", value=updateParam)
project.add_setting(name="phi1", value=1.5)
project.add_setting(name="phi2", value=2.0)

# ======================================================================================
my_solver = hyppopy.solvers.DynamicPSOSolver.DynamicPSOSolver(project)
# solver.blackbox = f

comm = MPI.COMM_WORLD
solver = MPISolverWrapper(solver=my_solver, mpi_comm=comm)
blackbox = MPIBlackboxFunction(blackbox_func=f, mpi_comm=comm)
solver.blackbox = blackbox
# ======================================================================================

solver.run()