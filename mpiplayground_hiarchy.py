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
    # here one can communicate downstream to distribute work into tier 1
    # or receive results from tier 1
    size = downstream_comm.Get_size()

    #distribute work downstream
    for i in range(size - 1):
        downstream_comm.send({'x':x,'y':y}, dest=i+1)

    # do own work

    #collect stuff from worker
    z = 0
    for i in range(size - 1):
        z += MPI.downstream_comm.recv(source=i + 1)

    # finally the loss function has to return the computed loss
    return z

def my_cascade_callback(tier_id, upstream_comm, downstream_comm, blackbox):
    """Example callback that is used for the worker code on subsequent tiers.
       This function is called by MPICascadedSolverWrapper for the respective
       tier.
       If this function returns, it is assumed that the node has finished.
       For tier 0 the function only is called after all evaluations have been
       made and the optimization process has finished.
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
    # here one can manage the worker code.
    # Depending on the tier id you can implement different communication patterns
    # with upstream or downstream as well as the workload code for the node.
    # In this example we only have 2 tiers so only tier 1 will be handled
    # here and we don't need tier specific logics
    if tier_id == 0:
        # worker on tier 0 only reach this point if all evaluations have been done
        # end it is just about cleaning up and releasing all nodes.
        size = downstream_comm.Get_size()
        for i in range(size - 1):
            downstream_comm.send(None, dest=i + 1)
    else:
        # Here we are on tier 1 and do the additional computations for the upstream
        while True:
            params = upstream_comm.recv(source=0)

            if params is None:
                #nothing to do any more
                return

            rank = upstream_comm.Get_rank()
            z = rank + params['x']**2 + params['y']**3
            upstream_comm.send(z, dest=0)


solver = MPICascadedSolverWrapper(solver=SolverPool.get(project=project), blackbox = my_loss_function, layout = [4,10,5], cascade_callback = my_cascade_callback)

#alternative call/init pattern
solver = MPICascadedSolverWrapper(solver=SolverPool.get(project=project))
solver.blackbox = my_loss_function
solver.layout = [3,5]
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
