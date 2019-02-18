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
# Author:

#------------------------------------------------------
# this needs to be imported, dont remove these
from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.workflowbase import WorkflowBase
#------------------------------------------------------

# import your external packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# import your custom DataLoader
from hyppopy.workflows.dataloader.simpleloader import SimpleDataLoader # This is a dataloader class create your own


class imageregistration_usecase(WorkflowBase):

    def setup(self, **kwargs):
        # here you create your own DataLoader instance
        dl = SimpleDataLoader()
        # call the start function of your DataLoader
        dl.start(path=ProjectManager.data_path,
                 data_name=ProjectManager.data_name,
                 labels_name=ProjectManager.labels_name)
        # pass the data to the solver
        self.solver.set_data(dl.data)

    def blackbox_function(self, data, params):
        # converting number back to integers is an ugly hack that will be removed in the future
        if "n_estimators" in params.keys():
            params["n_estimators"] = int(round(params["n_estimators"]))

        # Do your training
        clf = RandomForestClassifier(**params)
        # compute your loss
        loss = -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()
        # return loss
        return loss
