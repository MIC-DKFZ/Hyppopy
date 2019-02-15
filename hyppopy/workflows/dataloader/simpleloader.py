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

from hyppopy.workflows.dataloader.dataloaderbase import DataLoaderBase


class SimpleDataLoader(DataLoaderBase):

    def read(self, **kwargs):
        if kwargs['data_name'].endswith(".npy"):
            if not kwargs['labels_name'].endswith(".npy"):
                raise IOError("Expect both data_name and labels_name being of type .npy!")
            self.data = [np.load(os.path.join(kwargs['path'], kwargs['data_name'])), np.load(os.path.join(kwargs['path'], kwargs['labels_name']))]
        elif kwargs['data_name'].endswith(".csv"):
            try:
                dataset = pd.read_csv(os.path.join(kwargs['path'], kwargs['data_name']))
                y = dataset[kwargs['labels_name']].values
                X = dataset.drop([kwargs['labels_name']], axis=1).values
                self.data = [X, y]
            except Exception as e:
                print("Precondition violation, this usage case expects as data_name a "
                      "csv file and as label_name a name of a column in this csv table!")
        else:
            raise NotImplementedError("This combination of data_name and labels_name "
                                      "does not yet exist, feel free to add it")
