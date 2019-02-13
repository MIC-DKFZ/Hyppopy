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
from hyppopy.globals import TESTDATA_DIR
IRIS_DATA = os.path.join(TESTDATA_DIR, 'Iris')
TITANIC_DATA = os.path.join(TESTDATA_DIR, 'Titanic')

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.svc_usecase.svc_usecase import svc_usecase
from hyppopy.workflows.randomforest_usecase.randomforest_usecase import randomforest_usecase


class WorkflowTestSuite(unittest.TestCase):

    def setUp(self):
        self.results = []

    def test_workflow_svc_on_iris_from_xml(self):
        ProjectManager.read_config(os.path.join(IRIS_DATA, 'svc_config.xml'))
        uc = svc_usecase()
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_iris_from_xml(self):
        ProjectManager.read_config(os.path.join(IRIS_DATA, 'rf_config.xml'))
        uc = svc_usecase()
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_svc_on_iris_from_json(self):
        ProjectManager.read_config(os.path.join(IRIS_DATA, 'svc_config.json'))
        uc = svc_usecase()
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_iris_from_json(self):
        ProjectManager.read_config(os.path.join(IRIS_DATA, 'rf_config.json'))
        uc = randomforest_usecase()
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    # def test_workflow_svc_on_titanic_from_xml(self):
    #     ProjectManager.read_config(os.path.join(TITANIC_DATA, 'svc_config.xml'))
    #     uc = svc_usecase()
    #     uc.run()
    #     self.results.append(uc.get_results())
    #     self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_titanic_from_xml(self):
        ProjectManager.read_config(os.path.join(TITANIC_DATA, 'rf_config.xml'))
        uc = randomforest_usecase()
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    # def test_workflow_svc_on_titanic_from_json(self):
    #     ProjectManager.read_config(os.path.join(TITANIC_DATA, 'svc_config.json'))
    #     uc = svc_usecase()
    #     uc.run()
    #     self.results.append(uc.get_results())
    #     self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_titanic_from_json(self):
        ProjectManager.read_config(os.path.join(TITANIC_DATA, 'rf_config.json'))
        uc = randomforest_usecase()
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def tearDown(self):
        print("")
        for r in self.results:
            print(r)


if __name__ == '__main__':
    unittest.main()

