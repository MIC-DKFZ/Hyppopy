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
#
# Author: Sven Wanner (s.wanner@dkfz.de)

# In this tutorial we demonstrate the HyppopyProject class usage

# import the HyppopyProject class
from hyppopy.HyppopyProject import HyppopyProject

# To configure a solver we need to instanciate a HyppopyProject class.
# This class can be configured using a nested dict. This dict has two
# obligatory  sections, hyperparameter and settings. A hyperparameter
# is described using a dict containing a section, data and type field
# and thus the hyperparameter section is a collection of hyperparameter
# dicts. The settings section keeps solver settings. These might depend
# on the solver used and need to be checked for each. E.g. Randomsearch,
# Hyperopt and Optunity need a solver setting max_iterations, the Grid-
# searchSolver don't.
config = {
"hyperparameter": {
    "C": {
        "domain": "uniform",
        "data": [0.0001, 20],
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
    }
},
"settings": {
    "solver": {
        "max_iterations": 500
    },
    "custom": {}
}}

# When creating a HyppopyProject instance we
# pass the config dictionary to the constructor.
project = HyppopyProject(config=config)

# When building the project programmatically we can also use the methods
# add_hyperparameter and add_settings
project.clear()
project.add_hyperparameter(name="C", domain="uniform", data=[0.0001, 20], dtype="float")
project.add_hyperparameter(name="kernel", domain="categorical", data=["linear", "sigmoid"], dtype="str")
project.add_settings(section="solver", name="max_iterations", value=500)

# The custom section can be used freely
project.add_settings(section="custom", name="my_var", value=10)

# Settings are automatically transformed to member variables of the project class with the section as prefix
if project.solver_max_iterations < 1000 and project.custom_my_var == 10:
    print("Project configured!")
