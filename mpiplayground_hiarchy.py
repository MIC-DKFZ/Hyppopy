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

# master                       root = 0                     0   1   2   3   4   5   6   7   8   9   10  12  13  14  15
# ----------------------------------------------------------------------
# tier 0             1                     2                0   1   2
# ----------------------------------------------------------------------
# tier 1        11       12          21        22               10  20  11  12  21  22
# ----------------------------------------------------------------------
# tier 2    112   113 122   123  211   212 221  222                     110 120 210 220 111 112 121 122 211 212 221 222
# ----------------------------------------------------------------------
def my_loss_function_all_in_one(x, y, tier_address, upstream_comm, downstream_comm):
    size = downstream_comm.Get_size()

    z = None
    if len(tier_address) == 1: # tier_id == 0: # tier_id = len(tier_address), e.g. [0] -> len = 1
        for i in range(size - 1): # Why 'range(size-1)' and not 'enumerate(candidates)' (or similar)?
            downstream_comm.send({'x':x,'y':y}, dest=i+1)

        # ==================================================
        # do own work
        # ==================================================

        # collect stuff from worker
        z = 0
        for i in range(size - 1):
            z += downstream_comm.recv(source=i + 1)

    elif len(tier_address) == 2:
        # Here we are on tier 1 and do the additional computations for the upstream
        while True:
            params = upstream_comm.recv(source=0)

            if params is None:
                # nothing to do any more
                # Signal to finish worker send at the end.
                break

            rank = upstream_comm.Get_rank()
            z = rank + params['x'] ** 2 + params['y'] ** 3
            upstream_comm.send(z, dest=0)
    else:
        # can be implemented for as many tiers as one likes.


    if downstream_comm != MPI.COMM_NULL:
        # If we decide to create the comms via splitting of the initial one,
        # we can probably leave this out. At the end we stop all processes via signal_worker_finished().
        for i in range(size - 1):
            downstream_comm.send(None, dest=i + 1)

    # finally the loss function has to return the computed loss
    return z

# The use derfined loss function
# When using the MPICascadeSolverWrapper the loss function contains the
# worker code on tier 0 and has to communicate with its workers on tier 1.
# For that reason an additional named argument is passed: downstream_comm.
# downstream_comm can be used to communicate with the workers on tier 1
# associated with the respective tier 0 worker, their tier head node.
def my_loss_function_ver2(x, y, downstream_comm):   # lets assume layout [3,4]
    # here one can communicate downstream to distribute work into tier 1
    # or receive results from tier 1
    size = downstream_comm.Get_size()  # = 4

    #distribute work downstream
    for i in range(size - 1):
        downstream_comm.send({'x':x,'y':y}, dest=i+1)

    # ==================================================
    # do own work
    # ==================================================

    #collect stuff from worker
    z = 0
    for i in range(size - 1):
        z += downstream_comm.recv(source=i + 1)

    # finally the loss function has to return the computed loss
    return z



# @cascade_clean_up is a convenience decorator that can be used to ensures that
# the clean up code will always be executed at the end of the callback.
# so basically:
#    size = downstream_comm.Get_size()
#    for i in range(size - 1):
#        downstream_comm.send(termination_msg, dest=i + 1)
#
# We could also think to allow to specify a specific msg for each tier layer.
@cascade_clean_up(termination_msg=None)
def my_cascade_callback(tier_address, upstream_comm, downstream_comm, blackbox):
    """Example callback that is used for the worker code on subsequent tiers.
       This function is called by MPICascadedSolverWrapper for the respective
       tier.
       If this function returns, it is assumed that the node has finished.
       As tier 0 is handled by the MPIBlackboxFunction, this callback is used
       for the working code of all subsequent tiers.
       For tier 0 the function only is called after all evaluations have been
       made and the optimization process has finished, to allow the execution
       of clean up code (e.g. communicating downstream nodes to stop).
       :param tier_address: address of the node. An address is the list of upstream
       ranks (see upstream_comm below) of the node up to the master node of the
       hierarchy; all in reverse order. So len(tier_address) is always >0 and
       tier_address[0] is the upstream_comm rank of the current node. tier_address[-1]
       is the rank of the tier0 node, and so on. The the addresses in a hirarchy
       would look like:
              [1]                  [2]          -> "tier0"
            /  |  \              /  |  \
           /   |   \            /   |   \
       [1,1] [2,1] [3,1]    [1,2] [2,2] [3,2]   -> "tier1"
                  /  |  \
                 /   |   \
                /    |    \
          [1,3,1] [2,3,1] [3,3,1]               -> "tier3"
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
    # Depending on the tier address you can implement different communication patterns
    # with upstream or downstream as well as the workload code for the node.
    # In this example we only have 2 tiers so only tier 1 will be handled
    # here with worker code.
    # Worker on tier 0 (len(tier_address) == 0) only reach this point, if all evaluations
    # have been done and it is just about cleaning up and releasing all nodes (in this
    # example the cleaning up is ensured by the decorator cascade_clean_up.
    if len(tier_address) == 2:
        # Here we are on tier 1 and do the additional computations for the upstream
        while True:
            params = upstream_comm.recv(source=0)

            if params is None:
                #nothing to do any more
                break

            rank = upstream_comm.Get_rank()
            z = rank + params['x']**2 + params['y']**3
            upstream_comm.send(z, dest=0)

# Layout:
# -> [2*[3*[2]]]== [[2,2,2][2,2,2]]. In the previous notation this would have been [2,3,2].
# -> This notation enables us to create non-symmetric layouts.
# -> For convenience, we will provide a symmetric_layout() method to make the creation of symmetric layout easier:
#    [2*[3*[2]]] = symmetric_layout(2,3,2).
solver = MPICascadedSolverWrapper(solver=SolverPool.get(project=project), blackbox = my_loss_function, layout = symmetric_layout(2,3,2), cascade_callback = my_cascade_callback)

#alternative call/init pattern
solver = MPICascadedSolverWrapper(solver=SolverPool.get(project=project))
solver.blackbox = my_loss_function
solver.layout = [2*[3*[2]]]
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
