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
import unittest
from hyppopy.globals import TESTDATA_DIR
IRIS_DATA = os.path.join(TESTDATA_DIR, 'Iris')
TITANIC_DATA = os.path.join(TESTDATA_DIR, 'Titanic')

from hyppopy.workflows.svc_usecase.svc_usecase import svc_usecase
from hyppopy.workflows.randomforest_usecase.randomforest_usecase import randomforest_usecase


class Args(object):

    def __init__(self):
        pass

    def set_arg(self, name, value):
        setattr(self, name, value)


class WorkflowTestSuite(unittest.TestCase):

    def setUp(self):
        self.results = []

    def test_workflow_svc_on_iris_from_xml(self):
        svc_args_xml = Args()
        svc_args_xml.set_arg('plugin', '')
        svc_args_xml.set_arg('data', IRIS_DATA)
        svc_args_xml.set_arg('config', os.path.join(IRIS_DATA, 'svc_config.xml'))
        uc = svc_usecase(svc_args_xml)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_iris_from_xml(self):
        rf_args_xml = Args()
        rf_args_xml.set_arg('plugin', '')
        rf_args_xml.set_arg('data', IRIS_DATA)
        rf_args_xml.set_arg('config', os.path.join(IRIS_DATA, 'rf_config.xml'))
        uc = svc_usecase(rf_args_xml)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_svc_on_iris_from_json(self):
        svc_args_json = Args()
        svc_args_json.set_arg('plugin', '')
        svc_args_json.set_arg('data', IRIS_DATA)
        svc_args_json.set_arg('config', os.path.join(IRIS_DATA, 'svc_config.json'))
        uc = svc_usecase(svc_args_json)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_iris_from_json(self):
        rf_args_json = Args()
        rf_args_json.set_arg('plugin', '')
        rf_args_json.set_arg('data', IRIS_DATA)
        rf_args_json.set_arg('config', os.path.join(IRIS_DATA, 'rf_config.json'))
        uc = randomforest_usecase(rf_args_json)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_svc_on_titanic_from_xml(self):
        svc_args_xml = Args()
        svc_args_xml.set_arg('plugin', '')
        svc_args_xml.set_arg('data', TITANIC_DATA)
        svc_args_xml.set_arg('config', os.path.join(TITANIC_DATA, 'svc_config.xml'))
        uc = svc_usecase(svc_args_xml)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_titanic_from_xml(self):
        rf_args_xml = Args()
        rf_args_xml.set_arg('plugin', '')
        rf_args_xml.set_arg('data', TITANIC_DATA)
        rf_args_xml.set_arg('config', os.path.join(TITANIC_DATA, 'rf_config.xml'))
        uc = randomforest_usecase(rf_args_xml)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_svc_on_titanic_from_json(self):
        svc_args_json = Args()
        svc_args_json.set_arg('plugin', '')
        svc_args_json.set_arg('data', TITANIC_DATA)
        svc_args_json.set_arg('config', os.path.join(TITANIC_DATA, 'svc_config.json'))
        uc = svc_usecase(svc_args_json)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def test_workflow_rf_on_titanic_from_json(self):
        rf_args_json = Args()
        rf_args_json.set_arg('plugin', '')
        rf_args_json.set_arg('data', TITANIC_DATA)
        rf_args_json.set_arg('config', os.path.join(TITANIC_DATA, 'rf_config.json'))
        uc = randomforest_usecase(rf_args_json)
        uc.run()
        self.results.append(uc.get_results())
        self.assertTrue(uc.get_results().find("Solution") != -1)

    def tearDown(self):
        print("")
        for r in self.results:
            print(r)


if __name__ == '__main__':
    unittest.main()

