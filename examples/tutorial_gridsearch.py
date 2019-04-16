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

# In this tutorial we solve an optimization problem using the GridsearchSolver
# Gridsearch is very inefficient a Randomsearch might most of the time be the
# better choice.

# import the HyppopyProject class keeping track of inputs
from hyppopy.HyppopyProject import HyppopyProject

# import the GridsearchSolver classes
from hyppopy.solvers.GridsearchSolver import GridsearchSolver

# import the Blackboxfunction class wrapping your problem for Hyppopy
from hyppopy.BlackboxFunction import BlackboxFunction

# To configure the GridsearchSolver we only need the hyperparameter section. Another
# difference to the other solvers is that we need to define a gridsampling in addition
# to the range: 'data': [0, 1, 100] which means sampling the space from 0 to 1 in 100
# intervals. Gridsearch also supports categorical, uniform, normal and lognormal sampling
config = {
"hyperparameter": {
    "C": {
        "domain": "uniform",
        "data": [0.0001, 20, 20],
        "type": "float"
    },
    "gamma": {
        "domain": "uniform",
        "data": [0.0001, 20.0, 20],
        "type": "float"
    },
    "kernel": {
        "domain": "categorical",
        "data": ["linear", "sigmoid", "poly", "rbf"],
        "type": "str"
    }
},
"settings": {
    "solver": {},
    "custom": {}
}}

# When creating a HyppopyProject instance we
# pass the config dictionary to the constructor.
project = HyppopyProject(config=config)

# Hyppopy offers a class called BlackboxFunction to wrap your problem for Hyppopy.
# The function signature is as follows:
# BlackboxFunction(blackbox_func=None,
#                  dataloader_func=None,
#                  preprocess_func=None,
#                  callback_func=None,
#                  data=None,
#                  **kwargs)
#
# Means we can set a couple of function pointers, a data object and an arbitrary number of custom parameter via kwargs.
#
# - blackbox_func: a function pointer to the actual, user defined, blackbox function that is computing our loss
# - dataloader_func: a function pointer to a function handling the dataloading
# - preprocess_func: a function pointer to a function automatically executed before starting the optimization process
# - callback_func: a function pointer to a function that is called after each iteration with the trail object as input
# - data: setting data can be done via dataloader_func or directly
# - kwargs are passed to all functions above and thus can be used for parameter sharing between the functions
#
# (more details see in the documentation)
#
# Below we demonstrate the usage of all the above by defining a my_dataloader_function which in fact only grabs the
# iris dataset from sklearn and returns it. A my_preprocess_function which also does nothing useful here but
# demonstrating that a custom parameter can be set via kwargs and used in all of our functions when called within
# Hyppopy. The my_callback_function gets as input the dictionary containing the state of the iteration and thus can be
# used to access the current state of each solver iteration. Finally we define the actual loss_function
# my_loss_function, which gets as input a data object and params. Both parameter are fixed, the first is defined by
# the user depending on what is dataloader returns or the data object set in the constructor, the second is a dictionary
# with a sample of your hyperparameter space which content is in the choice of the solver.

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score


def my_dataloader_function(**kwargs):
    print("Dataloading...")
    iris_data = load_iris()
    return [iris_data.data, iris_data.target]


def my_callback_function(**kwargs):
    print("\r{}".format(kwargs), end="")
    
    
def my_loss_function(data, params):
    clf = SVC(**params)
    return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()


# We now create the BlackboxFunction object and pass all function pointers defined above,
# as well as 2 dummy parameter (my_preproc_param, my_dataloader_input) for demonstration purposes.
blackbox = BlackboxFunction(blackbox_func=my_loss_function,
                            dataloader_func=my_dataloader_function,
                            callback_func=my_callback_function)


# create a solver instance
solver = GridsearchSolver(project)
# pass the loss function to the solver
solver.blackbox = blackbox
# run the solver
solver.run()
# get the result via get_result() which returns a pandas dataframe
# containing the complete history and a dict best containing the
# best parameter set.
df, best = solver.get_results()

print("\n")
print("*"*100)
print("Best Parameter Set:\n{}".format(best))
print("*"*100)

