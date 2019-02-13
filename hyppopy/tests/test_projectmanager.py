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
import unittest
from hyppopy.projectmanager import ProjectManager


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class ProjectManagerTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_attr_transfer(self):
        ProjectManager.read_config(os.path.join(DATA_PATH, *('Titanic', 'rf_config.xml')))
        self.assertEqual(ProjectManager.data_name, 'train_cleaned.csv')
        self.assertEqual(ProjectManager.labels_name, 'Survived')
        self.assertEqual(ProjectManager.max_iterations, 3)
        self.assertEqual(ProjectManager.use_plugin, 'optunity')


if __name__ == '__main__':
    unittest.main()
