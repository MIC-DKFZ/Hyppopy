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


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.workflowbase import WorkflowBase
from hyppopy.workflows.dataloader.simpleloader import SimpleDataLoader


class randomforest_usecase(WorkflowBase):

    def setup(self, **kwargs):
        dl = SimpleDataLoader()
        dl.start(path=ProjectManager.data_path,
                 data_name=ProjectManager.data_name,
                 labels_name=ProjectManager.labels_name)
        self.solver.set_data(dl.data)

    def blackbox_function(self, data, params):
        if "n_estimators" in params.keys():
            params["n_estimators"] = int(round(params["n_estimators"]))
        clf = RandomForestClassifier(**params)
        return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()
