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

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.workflowbase import WorkflowBase
from hyppopy.workflows.dataloader.unetloader import UnetDataLoader


class unet_usecase(WorkflowBase):

    def setup(self):
        dl = UnetDataLoader()
        dl.start(data_path=ProjectManager.data_path,
                 data_name=ProjectManager.data_name,
                 image_dir=ProjectManager.image_dir,
                 labels_dir=ProjectManager.labels_dir,
                 split_dir=ProjectManager.split_dir,
                 output_dir=ProjectManager.data_path,
                 num_classes=ProjectManager.num_classes)
        self.solver.set_data(dl.data)

    def blackbox_function(self, data, params):
        pass
