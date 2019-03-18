# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical and Biological Informatics.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE.txt or http://www.mitk.org for details.
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import os
import sys
import tempfile
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.solver.HyperoptSolver import HyperoptSolver
from hyppopy.solver.OptunitySolver import OptunitySolver
from hyppopy.solver.RandomsearchSolver import RandomsearchSolver
from hyppopy.solver.GridsearchSolver import GridsearchSolver
from hyppopy.BlackboxFunction import BlackboxFunction

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score


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
    },
    "decision_function_shape": {
        "domain": "categorical",
        "data": ["ovo", "ovr"],
        "type": "str"
    }
},
"settings": {
    "solver": {
        "max_iterations": 300,
        "plugin": "gridsearch",
        "output_dir": os.path.join(tempfile.gettempdir(), 'results')
    },
    "custom": {
        "the_answer": 42
    }
}}

project = HyppopyProject(config=config)

print("--------------------------------------------------------------")
print("max_iterations:\t{}".format(project.solver_max_iterations))
print("plugin:\t{}".format(project.solver_plugin))
print("output_dir:\t{}".format(project.solver_output_dir))
print("the_answer:\t{}".format(project.custom_the_answer))


def my_loss_function(data, params):
    clf = SVC(**params)
    return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()


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
    print("kwargs['params']['my_preproc_param']={}".format(kwargs['params']['my_preproc_param']))
    # if the preprocessing function returns something,
    # the input data will be replaced with the data returned by this function.
    x = kwargs['data'][0]
    y = kwargs['data'][1]
    for i in range(x.shape[0]):
        x[i, :] += kwargs['params']['my_preproc_param']
    return [x, y]


def my_callback_function(**kwargs):
    print("\r{}".format(kwargs), end="")


blackbox = BlackboxFunction(blackbox_func=my_loss_function,
                            dataloader_func=my_dataloader_function,
                            preprocess_func=my_preprocess_function,
                            callback_func=my_callback_function,
                            #data=input_data, # data can be set directly or via a dataloader function
                            my_preproc_param=1,
                            my_dataloader_input='could/be/a/path')


if project.solver_plugin == "hyperopt":
    solver = HyperoptSolver(project)
elif project.solver_plugin == "optunity":
    solver = OptunitySolver(project)
elif project.solver_plugin == "randomsearch":
    solver = RandomsearchSolver(project)
elif project.solver_plugin == "gridsearch":
    solver = GridsearchSolver(project)

if solver is not None:
    solver.blackbox = blackbox
solver.run()
