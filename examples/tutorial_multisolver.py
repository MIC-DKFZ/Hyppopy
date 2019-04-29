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

# In this tutorial we solve an optimization problem using the Hyperopt Solver (http://hyperopt.github.io/hyperopt/).
# Hyperopt uses a Baysian - Tree Parzen Estimator - Optimization approach, which means that each iteration computes a
# new function value of the blackbox, interpolates a guess for the whole energy function and predicts a point to
# compute the next function value at. This next point is not necessarily a "better" value, it's only the value with
# the highest uncertainty for the function interpolation.
#
# See a visual explanation e.g. here (http://philipperemy.github.io/visualization/)


# import the HyppopyProject class keeping track of inputs
from hyppopy.HyppopyProject import HyppopyProject

# import the SolverPool singleton class
from hyppopy.SolverPool import SolverPool

# import the Blackboxfunction class wrapping your problem for Hyppopy
from hyppopy.BlackboxFunction import BlackboxFunction


# Next step is defining the problem space and all settings Hyppopy needs to optimize your problem.
# The config is a simple nested dictionary with two obligatory main sections, hyperparameter and settings.
# The hyperparameter section defines your searchspace. Each hyperparameter is again a dictionary with:
# 
# - a domain ['categorical', 'uniform', 'normal', 'loguniform']
# - the domain data [left bound, right bound] and 
# - a type of your domain ['str', 'int', 'float']
# 
# The settings section has two subcategories, solver and custom. The first contains settings for the solver,
# here 'max_iterations' - is the maximum number of iteration.
# 
# The custom section allows defining custom parameter. An entry here is transformed to a member variable of the
# HyppopyProject class. These can be useful when implementing new solver classes or for control your hyppopy script.
# Here we use it as a solver switch to control the usage of our solver via the config. This means with the script
# below your can try out every solver by changing use_solver to 'optunity', 'randomsearch', 'gridsearch',...
# It can be used like so: project.custom_use_plugin (see below) If using the gridsearch solver, max_iterations is
# ignored, instead each hyperparameter must specifiy a number of samples additionally to the range like so:
# 'data': [0, 1, 100] which means sampling the space from 0 to 1 in 100 intervals.

config = {
"hyperparameter": {
    "C": {
        "domain": "uniform",
        "data": [0.0001, 20],
        "type": float
    },
    "gamma": {
        "domain": "uniform",
        "data": [0.0001, 20.0],
        "type": float
    },
    "kernel": {
        "domain": "categorical",
        "data": ["linear", "sigmoid", "poly", "rbf"],
        "type": str
    },
    "decision_function_shape": {
        "domain": "categorical",
        "data": ["ovo", "ovr"],
        "type": str
    }
},
"max_iterations": 300,
"solver": "quasirandomsearch"
}


# When creating a HyppopyProject instance we
# pass the config dictionary to the constructor.
project = HyppopyProject(config=config)

# demonstration of the custom parameter access
print("-"*30)
print("max_iterations:\t{}".format(project.max_iterations))
print("solver chosen -> {}".format(project.solver))
print("-"*30)


# The BlackboxFunction signature is as follows:
# BlackboxFunction(blackbox_func=None,
#                  dataloader_func=None,
#                  preprocess_func=None,
#                  callback_func=None,
#                  data=None,
#                  **kwargs)
#
# - blackbox_func: a function pointer to the users loss function
# - dataloader_func: a function pointer for handling dataloading. The function is called once before
#                    optimizing. What it returns is passed as first argument to your loss functions
#                    data argument.
# - preprocess_func: a function pointer for data preprocessing. The function is called once before
#                    optimizing and gets via kwargs['data'] the raw data object set directly or returned
#                    from dataloader_func. What this function returns is then what is passed as first
#                    argument to your loss function.
# - callback_func: a function pointer called after each iteration. The input kwargs is a dictionary
#                  keeping the parameters used in this iteration, the 'iteration' index, the 'loss'
#                  and the 'status'. The function in this example is used for realtime printing it's
#                  input but can also be used for realtime visualization.
# - data: if not done via dataloader_func one can set a raw_data object directly
# - kwargs: dict that whose content is passed to all functions above.

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score


def my_dataloader_function(**kwargs):
    print("Dataloading...")
    # kwargs['params'] allows accessing additional parameter passed, see below my_preproc_param, my_dataloader_input.
    print("my loading argument: {}".format(kwargs['params']['my_dataloader_input']))
    iris_data = load_iris()
    return [iris_data.data, iris_data.target]


def my_preprocess_function(**kwargs):
    print("Preprocessing...")
    # kwargs['data'] allows accessing the input data
    print("data:", kwargs['data'][0].shape, kwargs['data'][1].shape)
    # kwargs['params'] allows accessing additional parameter passed, see below my_preproc_param, my_dataloader_input.
    print("kwargs['params']['my_preproc_param']={}".format(kwargs['params']['my_preproc_param']), "\n")
    # if the preprocessing function returns something,
    # the input data will be replaced with the data returned by this function.
    x = kwargs['data'][0]
    y = kwargs['data'][1]
    for i in range(x.shape[0]):
        x[i, :] += kwargs['params']['my_preproc_param']
    return [x, y]


def my_callback_function(**kwargs):
    print("\r{}".format(kwargs), end="")
    
    
def my_loss_function(data, params):
    clf = SVC(**params)
    return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()


# We now create the BlackboxFunction object and pass all function pointers defined above,
# as well as 2 dummy parameter (my_preproc_param, my_dataloader_input) for demonstration purposes.
blackbox = BlackboxFunction(blackbox_func=my_loss_function,
                            dataloader_func=my_dataloader_function,
                            preprocess_func=my_preprocess_function,
                            callback_func=my_callback_function,
                            my_preproc_param=1,
                            my_dataloader_input='could/be/a/path')


# Last step, is we use our SolverPool which automatically returns the correct solver.
# There are multiple ways to get the desired solver from the solver pool.
# 1. solver = SolverPool.get('hyperopt')
#    solver.project = project
# 2. solver = SolverPool.get('hyperopt', project)
# 3. The SolverPool will look for the field 'use_solver' in the project instance, if
# it is present it will be used to specify the solver so that in this case it is enough
# to pass the project instance.
solver = SolverPool.get(project=project)

# Give the solver your blackbox and run it. After execution we can get the result
# via get_result() which returns a pandas dataframe containing the complete history
# The dict best contains the best parameter set.
solver.blackbox = blackbox
#solver.start_viewer()
solver.run()
df, best = solver.get_results()

print("\n")
print("*"*100)
print("Best Parameter Set:\n{}".format(best))
print("*"*100)

