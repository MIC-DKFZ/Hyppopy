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
        "type": "float"
    },
    "gamma": {
        "domain": "normal",
        "data": [0.0001, 20.0],
        "type": "float"
    },
    "kernel": {
        "domain": "categorical",
        "data": ["linear", "sigmoid", "poly", "rbf"],
        "type": "str"
    },
    "decision_function_shape": {
        "domain": "categorical",
        "data": ["ovo", "ovr"],
        "type": "str"
    }
},
"settings": {
    "solver": {
        "max_iterations": 500
    },
    "custom": {
        "use_solver": "hyperopt"
    }
}}


# When creating a HyppopyProject instance we
# pass the config dictionary to the constructor.
project = HyppopyProject(config=config)

# demonstration of the custom parameter access
print("-"*30)
print("max_iterations:\t{}".format(project.solver_max_iterations))
print("solver chosen -> {}".format(project.custom_use_solver))
print("-"*30)


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
solver.run()
df, best = solver.get_results()

print("\n")
print("*"*100)
print("Best Parameter Set:\n{}".format(best))
print("*"*100)

