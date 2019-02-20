# coding: utf-8
import os
import time
import tempfile
import hyppopy as hp


# ### Get Some Data

from sklearn.datasets import load_iris
iris_data = load_iris()
input_data = [iris_data.data, iris_data.target]


# ### Setup ProjectManager
# 
# We could read the configuration from .json or .xml using ProjectManager.read_config, but parameters can can also be set via a dictionary. 
# All subsections in the section hyperparameter represent a hyperparameter. Top-level of each hyperparameter is the name. Additionally one 
# needs to specifiy a domain [uniform, categorical, loguniform, normal], a type [str, int, float] and the data, wich is either a range [from, to] 
# or a list of categories. All key value pairs in the section settings are added to the ProjectManager as member variables. If you add the 
# section custom you can add your own workflow specific parameter. The ProjectManager is a Singleton, thus need no instanciation and can be used
# everywhere but exists only once!

config = {
"hyperparameter": {
    "C": {
        "domain": "uniform",
        "data": [0, 20],
        "type": "float"
    },
    "gamma": {
        "domain": "uniform",
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
    "solver_plugin": {
        "max_iterations": 300,
        "use_plugin" : "hyperopt",
        "output_dir": os.path.join(tempfile.gettempdir(), 'results')
    },
    "custom": {
        "the_answer": 42
    }
}}

if hp.ProjectManager.set_config(config):
    print("Valid config dict set!")
else:
    print("Invalid config dict!")

print("--------------------------------------------------------------")
print("max_iterations:\t{}".format(hp.ProjectManager.max_iterations))
print("use_plugin:\t{}".format(hp.ProjectManager.use_plugin))
print("output_dir:\t{}".format(hp.ProjectManager.output_dir))
print("the_answer:\t{}".format(hp.ProjectManager.the_answer))


# ### Define the problem
# 
# We define a blackbox function with the signature func(data, params). The first parameter data is whatever we tell the solver later. 
# So we are free in defining the type of data we want to give our blackbox function. However, the parameter params is fixed and of type 
# dict. Each iteration, the solver will create a sample of each of the hyperparameter defined via the config set and throw it into our 
# blackbox function. 
# E.g. in our case above params in one round could look like {"C": 0.3, "gamma": 2.8, "kernel": "poly", "decision_function_shape", "ovo"}.

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def my_blackbox_function(data, params):
    clf = SVC(**params)
    return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()


# ### Feeding the Solver
# 
# Now everything is prepared to set up the solver. First we request the solver from the SolverFactory which assembled a solver from the 
# plugin parts we specified via the use_plugin parameter. Then we only need to set the our blackbox function and the data.

solver = hp.SolverFactory.get_solver()
solver.set_loss_function(my_blackbox_function)
solver.set_data(input_data)


# ### Start the Solver and get the results

print("\nStart optimization...")
start = time.process_time()
solver.run()
end = time.process_time()
print("Finished optimization!\n")
print("Total Time: {}s\n".format(end-start))
res, best = solver.get_results()
print("---- Optimal Parameter -----\n")
for p in best.items():
    print(" - {} : {}".format(p[0], p[1]))

solver.save_results()

