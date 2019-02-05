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

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import hyppopy.solverfactory as sfac


def data_loader(path, data_name, labels_name):
    if data_name.endswith(".npy"):
        if not labels_name.endswith(".npy"):
            raise IOError("Expect both data_name and labels_name being of type .npy!")
        data = [np.load(os.path.join(path, data_name)), np.load(os.path.join(path, labels_name))]
    elif data_name.endswith(".csv"):
        try:
            dataset = pd.read_csv(os.path.join(path, data_name))
            y = dataset[labels_name].values
            X = dataset.drop([labels_name], axis=1).values
            data = [X, y]
        except Exception as e:
            print("Precondition violation, this usage case expects as data_name a "
                  "csv file and as label_name a name of a column in this csv table!")
    else:
        raise NotImplementedError("This combination of data_name and labels_name "
                                  "does not yet exist, feel free to add it")
    return data


def randomforest_usecase(args):
    print("Execute Random Forest UseCase...")

    factory = sfac.SolverFactory.instance()
    solver = factory.get_solver(args.plugin)
    solver.read_parameter(args.config)

    data = data_loader(args.data, solver.settings.data_name, solver.settings.labels_name)
    solver.set_data(data)

    def rf_loss(data, params):
        clf = RandomForestClassifier(**params)
        return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()

    solver.set_loss_function(rf_loss)
    solver.run()
    solver.get_results()
