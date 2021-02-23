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

# A hyppopy minimal example optimizing a simple demo function f(x,y) = x**2+y**2

from hyppopy.BlackboxFunction import BlackboxFunction
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.SolverPool import SolverPool
from hyppopy.solvers.MPISolverWrapper import MPICascadedSolverWrapper

config = {
    "hyperparameter": {
        "x": {
            "domain": "uniform",
            "data": [-10.0, 10.0],
            "type": float,
            "frequency": 10
        },
        "y": {
            "domain": "uniform",
            "data": [-10.0, 10.0],
            "type": float,
            "frequency": 10
        }
    },
    "max_iterations": 500,
    "solver": "optunity"
}

project = HyppopyProject(config=config)

# The user defined loss function
# When using the MPICascadeSolverWrapper the loss function contains the
# worker code on tier 0 and have to communicate with its workers on tier 1.
# For that reason an additional named argument is passed: downstream_comm.
# downstream_comm can be used to communicate with the workers on tier 1
# associated with the respective tier 0 worker, their tier head node.
def my_loss_function(x, y, downstream_comm):
    return x**2+y**3

def my_cascade_callback(tier_id, upstream_comm, downstream_comm, blackbox):
    """Example callback that is used for the worker code on subsequent tiers.
       This function is called by MPICascadedSolverWrapper for the respective
       tier.
       If this function returns, it is assumed that the node has finished.
       :param tier_id: index of the tier the node is as a worker. As tier 0
        is handled by the MPIBlackboxFunction, this callback is
        used for all subsequent tiers.
       :param upstream_comm: MPI communicator that can be used to communicate
       upstream, so with the tier head node responsible for this node (in the
       role of a worker).
       :param downstream_comm: MPI communicator that can be used to communicate
       downstream, so with the workers of the tier below (where this node is
       the tier head). This parameter may be None if we are on the lowest tier
       of the cascade layout. In the downstream_comm the node has always the
       rank 0
       :param blackbox: the instance of the blackbox function (not in its MPI
       wrapper).
    """
    #here you can manage the worker code.
    #upstream_comm is the communicator used to talk with the primary of the current node.on the sdd

solver = MPICascadedSolverWrapper(solver=SolverPool.get(project=project), layout = [4,10,5], cascade_callback = my_cascade_callback)
solver.blackbox = my_loss_function

#alternative call
solver = MPICascadedSolverWrapper(solver=SolverPool.get(project=project), layout = [4,10,5])
solver.layout = [3,4,5]
solver.cascade_callback = my_cascade_callback
# maybe also
# solver.cascade_callbacks[2] = dedicated my_tier_calback for the 3rd tier
# or would MPI developers only work with just one callback?

solver.run()

df, best = solver.get_results()

if solver.is_master() is True:
    print("\n")
    print("*" * 100)
    print("Best Parameter Set:\n{}".format(best))
    print("*" * 100)
