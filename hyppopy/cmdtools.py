# -*- coding: utf-8 -*-
#
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

import argparse

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import logging
LOG = logging.getLogger('hyppopy')

from hyppopy.solverfactory import SolverFactory


def cmd_workflow():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-v', '--verbosity', type=int, required=False, default=0,
                        help='')


    args_dict = vars(parser.parse_args())

    iris = datasets.load_iris()
    X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42)
    my_IRIS_dta = [X, y]

    my_SVC_parameter = {
        'C': {'domain': 'uniform', 'data': [0, 20]},
        'gamma': {'domain': 'uniform', 'data': [0.0001, 20.0]},
        'kernel': {'domain': 'categorical', 'data': ['linear', 'sigmoid', 'poly', 'rbf']}
    }

    def my_SVC_loss_func(data, params):
        clf = SVC(**params)
        return -cross_val_score(clf, data[0], data[1], cv=3).mean()

    factory = SolverFactory()

    solver = factory.get_solver('optunity')
    solver.set_data(my_IRIS_dta)
    solver.set_parameters(my_SVC_parameter)
    solver.set_loss_function(my_SVC_loss_func)
    solver.run()
    solver.get_results()

    solver = factory.get_solver('hyperopt')
    solver.set_data(my_IRIS_dta)
    solver.set_parameters(my_SVC_parameter)
    solver.set_loss_function(my_SVC_loss_func)
    solver.run()
    solver.get_results()
