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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from hyppopy.workflows.workflowbase import Workflow
from hyppopy.workflows.datalaoder.simpleloader import SimpleDataLoader


class svc_usecase(Workflow):

    def __init__(self, args):
        Workflow.__init__(self, args)

    def setup(self):
        dl = SimpleDataLoader()
        dl.read(path=self.args.data, data_name=self.solver.settings.data_name,
                labels_name=self.solver.settings.labels_name)
        self.solver.set_data(dl.get())

    def blackbox_function(self, data, params):
        clf = SVC(**params)
        return -cross_val_score(estimator=clf, X=data[0], y=data[1], cv=3).mean()
